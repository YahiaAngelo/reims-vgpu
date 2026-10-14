//! The indirect command buffer (object type 20): a Metal ICB with its command layout and
//! flags, paged on the CPU through the task's resource manager.
//!
//! It has no encoder paging: a needs-sync ICB on an in-process exec's blit list raises
//! [`RaiseKind::Unrecognized`](crate::resources::RaiseKind::Unrecognized) there. The CPU paging
//! runs inside its own real pool, so what Metal hands back +0 during the walk is released when
//! the walk returns, and a raise leaves the pool pushed for the enclosing pop.

use std::sync::Arc;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::guest::records::GuestPages;
use crate::platform::{Handle, Platform};
use crate::raise::RaiseResult;
use crate::resources::manager::icb::{
    prepare_indirect_command_buffer_on_null, synchronize_indirect_command_buffer_on_null,
};
use crate::resources::resource::AnyResource;
use crate::resources::resource::{Resource, ResourceClass, ResourceKind, TaskRef, read_descriptor};
use crate::rt::{Held, Hold};

/// The command layout record (52 bytes): where each part of a command lies in the guest's
/// command stride. The field names are the guest ABI's.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
#[expect(missing_docs, reason = "the guest ABI's field names")]
pub struct IndirectCommandLayout {
    pub command_type_offset: u16,
    pub barrier_offset: u16,
    pub kernel_dispatch_arguments_offset: u16,
    pub tessellation_factor_offset: u16,
    pub pipeline_state_offset: u32,
    pub vertex_buffer_bind_offset: u32,
    pub fragment_buffer_bind_offset: u32,
    pub object_buffer_bind_offset: u32,
    pub mesh_buffer_bind_offset: u32,
    pub kernel_buffer_bind_offset: u32,
    pub attribute_stride_offset: u32,
    pub object_threadgroup_memory_length_offset: u32,
    pub threadgroup_memory_length_offset: u32,
    pub command_arguments_offset: u32,
    pub command_size: u32,
}

const _: () = assert!(size_of::<IndirectCommandLayout>() == 52);

/// The ICB's flags record (40 bytes). The first word holds three one-bit fields: inherit
/// pipeline state, inherit buffers, dynamic attribute stride.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
#[expect(missing_docs, reason = "the guest ABI's field names")]
pub struct IndirectCommandBufferFlags {
    pub bits: u32,
    pub command_types: u32,
    pub max_vertex_buffer_bind_count: u64,
    pub max_fragment_buffer_bind_count: u64,
    pub max_kernel_buffer_bind_count: u64,
    pub max_kernel_threadgroup_memory_bind_count: u64,
}

const _: () = assert!(size_of::<IndirectCommandBufferFlags>() == 40);

impl IndirectCommandBufferFlags {
    /// Bit 0: commands inherit the encoder's pipeline state.
    pub fn inherit_pipeline_state(&self) -> bool {
        self.bits & 1 != 0
    }
    /// Bit 1: commands inherit the encoder's buffers.
    pub fn inherit_buffers(&self) -> bool {
        self.bits.wrapping_shr(1) & 1 != 0
    }
    /// Bit 2: dynamic attribute stride.
    pub fn support_dynamic_attribute_stride(&self) -> bool {
        self.bits.wrapping_shr(2) & 1 != 0
    }
}

/// The ICB record's first 12 bytes, packed: the length at +0 and the page at +8 (all this
/// kind reads).
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct IcbDescriptor {
    length: u64,
    page: u32,
}

const _: () = assert!(size_of::<IcbDescriptor>() == 12);

/// An indirect command buffer with its layout and flags, paged through the task's resource
/// manager.
pub struct IcbResource<P: Platform> {
    /// Unretained.
    task: TaskRef<P>,
    length: u64,
    /// The page, as a task VA offset (`page << page shift`).
    virtual_offset: u64,
    /// `None` when Metal made none, which the factory does not check without a resource ID.
    icb: Option<P::Icb>,
    layout: IndirectCommandLayout,
    flags: IndirectCommandBufferFlags,
}

impl<P: Platform> IcbResource<P> {
    /// An ICB resource from the record's bytes (12 are read; a shorter slice reads zero past
    /// its end). Starts host-invalid and guest-valid; never `None` and never a raise.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(
        task: TaskRef<P>,
        pages: GuestPages,
        descriptor: &[u8],
        icb: Option<P::Icb>,
        layout: IndirectCommandLayout,
        flags: IndirectCommandBufferFlags,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let d: IcbDescriptor = read_descriptor(descriptor);
        let (length, page) = (d.length, d.page);
        Ok(Some(Resource::new(ResourceKind::Icb(IcbResource {
            task,
            length,
            virtual_offset: pages.address(page),
            icb,
            layout,
            flags,
        }))))
    }

    /// The layout.
    pub fn layout(&self) -> &IndirectCommandLayout {
        &self.layout
    }
}

/// Runs a CPU paging walk in its own real pool: popped when the walk returns (`return`
/// included), left pushed for the enclosing pop when it raises. The walk is also a landing
/// frame ([`Held::raising`]), under a hold of the frame's holder: a deep hold its guest reads
/// make (the page-table read's device handle) is released at the walk's end, before the pool
/// pops, or goes on with its raise.
fn with_pool<T>(body: impl FnOnce(Hold<'_>) -> RaiseResult<T>) -> RaiseResult<T> {
    crate::rt::pool::around_here(|| Held::raising(|held| body(Hold::of(held))))
}

impl<P: Platform> ResourceClass<P> for IcbResource<P> {
    fn initial_guest_valid(&self) -> bool {
        true
    }

    /// Retained for the caller.
    fn get_indirect_command_buffer(&self, this: &Resource<P>) -> RaiseResult<Option<P::Icb>> {
        let _ = this;
        Ok(self.icb.as_ref().map(Handle::clone_ref))
    }

    fn get_resource(&self, this: &Resource<P>) -> RaiseResult<Option<AnyResource<P>>> {
        let _ = this;
        Ok(self.icb.as_ref().map(|i| AnyResource::Icb(i.clone_ref())))
    }

    /// Nothing: a discard of an ICB is answered and does nothing.
    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        // Opcode 63 on an ICB does nothing.
        Ok(())
    }

    fn needs_cpu_paging(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    /// The guest's commands into the ICB, through the task's resource manager. A missing
    /// cursor or a refused command logs and leaves it host-invalid (the next exec tries
    /// again); commands encoded before a refused one stay.
    fn prepare_with_cpu(&self, _hold: Hold<'_>, this: &Resource<P>) -> RaiseResult<()> {
        let state = &this.state;
        // A prepared ICB ignores guest writes until invalidated.
        if state.host_valid() {
            return Ok(());
        }
        if !state.guest_valid() {
            log::warn!(
                "preparing an indirect command buffer whose guest copy is not valid (offset {:#x})",
                self.virtual_offset
            );
        }
        let prepared = with_pool(|hold| {
            // A torn-down task gives no cursor.
            let task = self.task.upgrade();
            let cursor = match &task {
                Some(t) => t.cursor_from_virtual_offset(hold, self.virtual_offset, self.length)?,
                None => None,
            };
            let (Some(cursor), Some(task)) = (cursor, task) else {
                log::warn!(
                    "no cursor over the indirect command buffer's guest memory to prepare from \
                     (offset {:#x}, length {:#x})",
                    self.virtual_offset,
                    self.length
                );
                return Ok(false);
            };
            let Some(icb) = self.icb.as_ref() else {
                // The walk reads only the flags before its loops, which run 0 times for no ICB:
                // it succeeds even with no manager (a torn-down task), and the ICB turns
                // host-valid.
                return Ok(true);
            };
            // After teardown there is no manager: the walk reads the cursor before it first
            // needs the manager, and dies there.
            let result = match task.resource_manager() {
                Some(manager) => manager.prepare_indirect_command_buffer(
                    hold,
                    icb,
                    &self.layout,
                    &self.flags,
                    cursor,
                )?,
                None => prepare_indirect_command_buffer_on_null::<P>(
                    hold,
                    icb,
                    &self.layout,
                    &self.flags,
                    cursor,
                )?,
            };
            match result {
                Ok(()) => Ok(true),
                Err(error) => {
                    // The refusal's text may hold guest bytes, printed lossily.
                    log::warn!("indirect command buffer not prepared: {error}");
                    Ok(false)
                }
            }
        })?;
        if prepared {
            // The guest flag is untouched.
            state.set_host_valid(true);
        }
        Ok(())
    }

    /// The ICB's commands back to guest memory. The manager's result is dropped: the guest is
    /// marked valid even when the write-back failed. In-process the task's writable cursor dies.
    fn synchronize_with_cpu(&self, this: &Resource<P>) -> RaiseResult<()> {
        let state = &this.state;
        if state.guest_valid() {
            return Ok(());
        }
        if !state.host_valid() {
            log::warn!(
                "synchronizing an indirect command buffer whose host copy is not valid \
                 (offset {:#x})",
                self.virtual_offset
            );
        }
        let made_cursor = with_pool(|hold| {
            let task = self.task.upgrade();
            let cursor = match &task {
                Some(t) => {
                    t.writable_cursor_from_virtual_offset(hold, self.virtual_offset, self.length)?
                }
                None => None,
            };
            let (Some(cursor), Some(task)) = (cursor, task) else {
                log::warn!(
                    "no cursor over the indirect command buffer's guest memory to synchronize to \
                     (offset {:#x}, length {:#x})",
                    self.virtual_offset,
                    self.length
                );
                return Ok(false);
            };
            // No ICB: the loops run 0 times before the manager is needed; nothing is written.
            let Some(icb) = self.icb.as_ref() else {
                return Ok(true);
            };
            // With no manager the walk dies as the prepare's does.
            // The write-back's error is dropped.
            let _ = match task.resource_manager() {
                Some(manager) => manager.synchronize_indirect_command_buffer(
                    hold,
                    icb,
                    &self.layout,
                    &self.flags,
                    cursor,
                )?,
                None => synchronize_indirect_command_buffer_on_null::<P>(
                    hold,
                    icb,
                    &self.layout,
                    &self.flags,
                    cursor,
                )?,
            };
            Ok(true)
        })?;
        if made_cursor {
            state.set_guest_valid(true);
        }
        Ok(())
    }

    /// A CPU synchronize (asked of the resource) when the raw flags say host-only.
    fn evict_with_cpu(&self, this: &Resource<P>) -> RaiseResult<()> {
        if !this.state.guest_valid() && this.state.host_valid() {
            this.synchronize_with_cpu()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guest::records::APV_PAGE_SHIFT;
    use crate::memory::cursor::{Cursor, WritableCursor};
    use crate::raise::OnFault;
    use crate::resources::resource::test_support::kind as raised;
    use crate::resources::{Op, RaiseKind};
    use crate::seams::ResourceHost;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::objects::Model;
    use crate::testkit::resource_host::FakeResourceHost;
    use crate::testkit::{FakePlatform, Fk, Step, kind};

    type P = FakePlatform;

    struct Env {
        p: Arc<P>,
        rec: Arc<FakeResourceHost>,
        host: Arc<dyn ResourceHost<P>>,
    }

    fn env() -> Env {
        let p = Arc::new(FakePlatform::new());
        let rec = FakeResourceHost::new(p.clone());
        let host: Arc<dyn ResourceHost<P>> = rec.clone();
        Env { p, rec, host }
    }

    fn record(length: u64, page: u32) -> Vec<u8> {
        let mut v = length.to_le_bytes().to_vec();
        v.extend_from_slice(&page.to_le_bytes());
        v
    }

    fn make(e: &Env, icb: Option<Fk<kind::Icb>>) -> Arc<Resource<P>> {
        IcbResource::new(
            Arc::downgrade(&e.host),
            crate::guest::records::GuestPages::HOST,
            &record(0x200, 3),
            icb,
            IndirectCommandLayout {
                command_size: 64,
                ..IndirectCommandLayout::default()
            },
            IndirectCommandBufferFlags {
                command_types: 1,
                ..IndirectCommandBufferFlags::default()
            },
        )
        .unwrap()
        .unwrap()
    }

    fn icb(e: &Env) -> Fk<kind::Icb> {
        e.p.world().new_obj(Model::None)
    }

    fn kind_of(r: &Resource<P>) -> &IcbResource<P> {
        match &r.kind {
            ResourceKind::Icb(x) => x,
            _ => panic!("not an ICB"),
        }
    }

    /// The walk's real pool is pushed while the body runs, so it catches Metal's +0 returns, and
    /// pops on a normal exit; a raise leaves it pushed for the enclosing pop
    #[test]
    fn walk_pool_pops_on_return_and_stays_pushed_on_a_raise() {
        use crate::rt::pool::test_native::{events, note, with_test_native};
        with_test_native(|| {
            let r: RaiseResult<u32> = with_pool(|_| {
                note("body");
                Ok(7)
            });
            assert_eq!(r.unwrap(), 7);
            assert_eq!(events(), ["push 0x100", "body", "pop 0x100"]);
            {
                let r: RaiseResult<()> =
                    with_pool(|_| crate::resources::raise(RaiseKind::NoGuestBuffer));
                assert!(r.is_err());
                assert_eq!(events(), ["push 0x101"]);
            }
        });
    }

    /// A deep hold the walk's guest reads make (the page-table read's device handle) is released
    /// at the walk's end, before its pool pops; on a raise it goes on with the raise, and the
    /// catcher's holder releases it after the report
    #[test]
    fn a_deep_hold_lands_in_the_walk() {
        use crate::rt::pool::test_native::{events, note, with_test_native};
        struct Rel;
        impl Drop for Rel {
            fn drop(&mut self) {
                note("release");
            }
        }
        with_test_native(|| {
            let r: RaiseResult<()> = with_pool(|hold| {
                hold.keep(Rel);
                note("walked");
                Ok(())
            });
            assert!(r.is_ok());
            assert_eq!(events(), ["push 0x100", "walked", "release", "pop 0x100"]);
            let catcher = Held::new();
            {
                let hold = Hold::of(&catcher);
                let r: RaiseResult<()> = with_pool(|hold| {
                    hold.keep(Rel);
                    crate::resources::raise(RaiseKind::NoGuestBuffer)
                });
                note("report");
                hold.land(r.unwrap_err());
            }
            drop(catcher);
            assert_eq!(events(), ["push 0x101", "report", "release"]);
        });
    }

    #[test]
    fn init_and_getters() {
        let e = env();
        let b = icb(&e);
        let id = b.id();
        let r = make(&e, Some(b.clone_ref()));
        assert!(!r.state.host_valid());
        assert!(r.state.guest_valid());
        let x = kind_of(&r);
        assert_eq!(x.length, 0x200);
        assert_eq!(x.virtual_offset, 3 << APV_PAGE_SHIFT);
        assert_eq!(x.layout().command_size, 64);
        assert_eq!(x.flags.command_types, 1);
        assert_eq!(r.get_indirect_command_buffer().unwrap().unwrap().id(), id);
        match r.get_resource().unwrap().unwrap() {
            AnyResource::Icb(i) => assert_eq!(i.id(), id),
            other => panic!("{other:?}"),
        }
        assert!(r.needs_cpu_paging());
        assert!(r.supports_discard());
        r.discard().unwrap();
        // No encoder paging: what an in-process exec that puts a needs-sync ICB on its blit
        // list raises.
        {
            let hold = Hold::NONE;
            assert_eq!(
                raised(r.synchronize_in_encoder(hold, None)),
                Some(RaiseKind::Unrecognized(Op::Synchronize))
            );
            assert_eq!(
                raised(r.prepare_in_encoder(hold, None)),
                Some(RaiseKind::Unrecognized(Op::Prepare))
            );
            assert_eq!(b.retain_count(), 2);
            drop(r);
            assert_eq!(b.retain_count(), 1);
            // A nil ICB, and the largest page (no overflow).
            let r = IcbResource::<P>::new(
                Arc::downgrade(&e.host),
                crate::guest::records::GuestPages::HOST,
                &record(1, u32::MAX),
                None,
                IndirectCommandLayout::default(),
                IndirectCommandBufferFlags::default(),
            )
            .unwrap()
            .unwrap();
            let x = kind_of(&r);
            assert_eq!(x.virtual_offset, u64::from(u32::MAX) << APV_PAGE_SHIFT);
            assert!(r.get_resource().unwrap().is_none());
        };
    }

    /// The fields at +0 and +8, little-endian; bytes past 12 unread
    #[test]
    fn init_reads_descriptor() {
        let e = env();
        let mut rec = record(0x0102_0304_0506_0708, 0x0a0b_0c0d);
        rec.extend_from_slice(&[0xff; 4]);
        let r = IcbResource::<P>::new(
            Arc::downgrade(&e.host),
            crate::guest::records::GuestPages::HOST,
            &rec,
            None,
            IndirectCommandLayout::default(),
            IndirectCommandBufferFlags::default(),
        )
        .unwrap()
        .unwrap();
        let x = kind_of(&r);
        assert_eq!(x.length, 0x0102_0304_0506_0708);
        assert_eq!(x.virtual_offset, 0x0a0b_0c0d << APV_PAGE_SHIFT);
    }

    /// A host-valid ICB returns at once
    #[test]
    fn prepare_host_valid_returns() {
        let e = env();
        let r = make(&e, Some(icb(&e)));
        r.state.set_host_valid(true);
        let mark = e.p.call_log().mark();
        r.prepare_with_cpu(Hold::NONE).unwrap();
        assert!(e.p.call_log().since(mark).is_empty());
    }

    /// A null cursor leaves it host-invalid (guest-invalid only logs)
    #[test]
    fn prepare_null_cursor() {
        let e = env();
        let r = make(&e, Some(icb(&e)));
        r.state.set_guest_valid(false);
        r.prepare_with_cpu(Hold::NONE).unwrap();
        assert!(!r.state.host_valid());
        let c = e.p.call_log().calls_to("cursor_from_virtual_offset");
        assert_eq!(
            c[0].args,
            vec![Arg::U64(3 << APV_PAGE_SHIFT), Arg::U64(0x200)]
        );
        assert_eq!(e.p.call_log().count("resource_manager"), 0);
    }

    /// An invalid state and a missing cursor each log their own line
    #[test]
    fn invalid_state_and_null_cursor_lines() {
        use crate::resources::resource::test_support::logged;
        let e = env();
        let r = make(&e, Some(icb(&e)));
        r.state.set_guest_valid(false);
        let ((), lines) = logged(|| {
            r.prepare_with_cpu(Hold::NONE).unwrap();
            assert!(!r.state.host_valid());
            r.synchronize_with_cpu().unwrap();
            assert!(!r.state.guest_valid());
        });
        // Page 3: offset 0xc000 on a 16 KiB guest page.
        let offset = 3_u64 << APV_PAGE_SHIFT;
        assert_eq!(
            lines,
            vec![
                format!(
                    "preparing an indirect command buffer whose guest copy is not valid \
                     (offset {offset:#x})"
                ),
                format!(
                    "no cursor over the indirect command buffer's guest memory to prepare from \
                     (offset {offset:#x}, length 0x200)"
                ),
                format!(
                    "synchronizing an indirect command buffer whose host copy is not valid \
                     (offset {offset:#x})"
                ),
                format!(
                    "no cursor over the indirect command buffer's guest memory to synchronize to \
                     (offset {offset:#x}, length 0x200)"
                ),
            ]
        );
    }

    /// A torn-down task gives no cursor
    #[test]
    fn prepare_task_gone() {
        let e = env();
        let gone: TaskRef<P> = {
            let other: Arc<dyn ResourceHost<P>> = e.rec.sibling();
            Arc::downgrade(&other)
        };
        let r = IcbResource::new(
            gone,
            crate::guest::records::GuestPages::HOST,
            &record(0x200, 3),
            Some(icb(&e)),
            IndirectCommandLayout::default(),
            IndirectCommandBufferFlags::default(),
        )
        .unwrap()
        .unwrap();
        r.prepare_with_cpu(Hold::NONE).unwrap();
        assert!(!r.state.host_valid());
        r.state.set_guest_valid(false);
        r.synchronize_with_cpu().unwrap();
        assert!(!r.state.guest_valid());
        assert!(e.p.call_log().is_empty());
    }

    /// The cursor's raise propagates, host still invalid
    #[test]
    fn prepare_cursor_raise() {
        let e = env();
        let r = make(&e, Some(icb(&e)));
        e.p.script().push(
            "cursor_from_virtual_offset",
            Step::raise("NSGenericException", "x"),
        );
        let hold = Hold::NONE;
        assert!(r.prepare_with_cpu(hold).is_err());
        assert!(!r.state.host_valid());
    }

    /// A fake ICB of `size` commands.
    fn sized_icb(e: &Env, size: u64) -> Fk<kind::Icb> {
        use crate::platform::device::GpuDevice;
        use crate::platform::values::IcbDesc;
        e.p.device()
            .new_indirect_command_buffer(&IcbDesc::default(), size, 0)
            .unwrap()
            .unwrap()
    }

    /// The manager gone (after teardown): the walk reads the cursor first, so a short one is a
    /// logged refusal, not a death
    #[test]
    fn no_manager_and_a_short_cursor_is_a_refusal() {
        // subCursor fails before `this` is touched; logs and returns.
        let e = env();
        let r = make(&e, Some(sized_icb(&e, 1)));
        e.p.script().push(
            "cursor_from_virtual_offset",
            Step::value(Some(Cursor::with_ranges(Vec::new()))),
        );
        let d = catch_death(|| {
            let _ = r.prepare_with_cpu(Hold::NONE);
        });
        assert!(d.is_ok(), "{d:?}");
        assert!(!r.state.host_valid());
    }

    /// The manager gone: the walk dies at its first use of the manager
    #[test]
    fn prepare_manager_gone_dies_at_the_first_member_access() {
        use crate::resources::manager::icb::testing::cursor;
        let e = env();
        let r = make(&e, Some(sized_icb(&e, 1)));
        // Command type 1 (no reset) and pipeline-state reference 1, both read at offset 0.
        let mut command = vec![0u8; 64];
        command[0] = 1;
        e.p.script().push(
            "cursor_from_virtual_offset",
            Step::value(Some(cursor(&command))),
        );
        let d = catch_death(|| {
            let _ = r.prepare_with_cpu(Hold::NONE);
        });
        assert_eq!(
            d,
            Err(crate::resources::manager::icb::DeathKind::NoManager.site())
        );
    }

    /// A nil ICB with the manager gone runs 0 rounds: no crash, host-valid
    #[test]
    fn no_icb_and_no_manager_prepares() {
        // Only the flags are read; both loops run 0 times before the manager is needed: no
        // death, and host-valid.
        let e = env();
        let r = make(&e, None);
        e.p.script().push(
            "cursor_from_virtual_offset",
            Step::value(Some(Cursor::with_ranges(Vec::new()))),
        );
        let d = catch_death(|| {
            let _ = r.prepare_with_cpu(Hold::NONE);
        });
        assert!(d.is_ok(), "{d:?}");
        assert!(r.state.host_valid());
    }

    /// A nil ICB with the manager gone runs 0 rounds: no crash, guest-valid
    #[test]
    fn no_icb_and_no_manager_synchronizes() {
        // Only the flags are read; both loops run 0 times: guest-valid.
        let e = env();
        let r = make(&e, None);
        r.state.set_guest_valid(false);
        e.p.script().push(
            "writable_cursor_from_virtual_offset",
            Step::value(Some(WritableCursor::with_ranges(Vec::new()))),
        );
        let d = catch_death(|| {
            let _ = r.synchronize_with_cpu();
        });
        assert!(d.is_ok(), "{d:?}");
        assert!(r.state.guest_valid());
    }

    /// Guest-valid returns; a null cursor leaves it guest-invalid
    #[test]
    fn synchronize_paths() {
        let e = env();
        let r = make(&e, Some(icb(&e)));
        r.synchronize_with_cpu().unwrap();
        assert!(e.p.call_log().is_empty());
        r.state.set_guest_valid(false);
        r.synchronize_with_cpu().unwrap();
        assert!(!r.state.guest_valid());
        let c =
            e.p.call_log()
                .calls_to("writable_cursor_from_virtual_offset");
        assert_eq!(
            c[0].args,
            vec![Arg::U64(3 << APV_PAGE_SHIFT), Arg::U64(0x200)]
        );
    }

    /// The in-process task's writable cursor dies
    #[test]
    fn synchronize_in_process_aborts() {
        let e = env();
        let r = make(&e, Some(icb(&e)));
        r.state.set_guest_valid(false);
        e.p.script()
            .push("writable_cursor_from_virtual_offset", Step::crash());
        let d = catch_death(|| {
            let _ = r.synchronize_with_cpu();
        });
        assert!(d.is_err());
    }

    /// The manager gone: the walk fails at the cursor before it needs the manager; the result is
    /// dropped and the guest turns valid
    #[test]
    fn no_manager_and_a_short_cursor_still_synchronizes() {
        // writableSubCursor first;.
        let e = env();
        let r = make(&e, Some(sized_icb(&e, 1)));
        r.state.set_guest_valid(false);
        e.p.script().push(
            "writable_cursor_from_virtual_offset",
            Step::value(Some(WritableCursor::with_ranges(Vec::new()))),
        );
        let d = catch_death(|| {
            let _ = r.synchronize_with_cpu();
        });
        assert!(d.is_ok(), "{d:?}");
        assert!(r.state.guest_valid());
    }

    /// Synchronizes only when guest-invalid and host-valid
    #[test]
    fn evict_tests_the_raw_flags() {
        let e = env();
        let r = make(&e, Some(icb(&e)));
        for (guest, host, syncs) in [
            (true, true, false),
            (true, false, false),
            (false, false, false),
            (false, true, true),
        ] {
            r.state.set_guest_valid(guest);
            r.state.set_host_valid(host);
            let mark = e.p.call_log().mark();
            r.evict_with_cpu().unwrap();
            let n =
                e.p.call_log()
                    .since(mark)
                    .iter()
                    .filter(|c| c.sel == "writable_cursor_from_virtual_offset")
                    .count();
            assert_eq!(n, usize::from(syncs), "guest {guest} host {host}");
        }
    }

    /// No ICB: the manager's loops run 0 times and succeed, so it turns
    /// host-valid
    #[test]
    fn no_icb_prepare_sets_host_valid() {
        use crate::resources::manager::ResourceManager;
        use crate::tasks::object_lists::{ObjectListKind, ObjectLists};
        let e = env();
        let olm = ObjectLists::new(
            ObjectListKind::MemoryMap,
            0x10000,
            false,
            Arc::downgrade(&e.host),
            crate::guest::records::GuestPages::HOST,
        );
        let m = ResourceManager::new(
            e.p.clone(),
            Some(e.p.device()),
            Arc::downgrade(&e.host),
            olm,
            true,
        );
        let r = make(&e, None);
        e.p.script().push(
            "cursor_from_virtual_offset",
            Step::value(Some(Cursor::with_ranges(Vec::new()))),
        );
        e.p.script()
            .push("resource_manager", Step::value(Some(m.clone())));
        r.prepare_with_cpu(Hold::NONE).unwrap();
        // After a size-0 success.
        assert!(r.state.host_valid());
        // And the write-back to a nil ICB writes nothing and marks the guest valid.
        r.state.set_guest_valid(false);
        e.p.script().push(
            "writable_cursor_from_virtual_offset",
            Step::value(Some(WritableCursor::with_ranges(Vec::new()))),
        );
        e.p.script()
            .push("resource_manager", Step::value(Some(m.clone())));
        r.synchronize_with_cpu().unwrap();
        assert!(r.state.guest_valid());
    }

    /// A refusal's text is logged, the guest record's bytes printed lossily
    #[test]
    fn a_refusal_logs_its_text() {
        use crate::platform::device::GpuDevice;
        use crate::resources::buffer::child::HeapBufferResource;
        use crate::resources::manager::ResourceManager;
        use crate::resources::manager::create::Resources;
        use crate::resources::manager::icb::DrawPatches;
        use crate::resources::manager::icb::testing::cursor;
        use crate::tasks::object_lists::{ObjectListKind, ObjectLists};
        let e = env();
        let olm = ObjectLists::new(
            ObjectListKind::MemoryMap,
            0x10000,
            false,
            Arc::downgrade(&e.host),
            crate::guest::records::GuestPages::HOST,
        );
        let m = ResourceManager::new(
            e.p.clone(),
            Some(e.p.device()),
            Arc::downgrade(&e.host),
            olm,
            true,
        );
        // The patch index buffer, reference 5, so the walk reaches the tessellation read.
        let b =
            e.p.device()
                .new_buffer_of_length(OnFault::Die, 64, 0)
                .unwrap()
                .unwrap();
        let holder = make(&e, None);
        assert!(m.register_object::<Resources>(5, HeapBufferResource::new(holder, b, false)));
        // One DrawPatches command (type 4) whose tessellation factor (0x20 bytes at 0x30) runs past
        // the 0x40-byte command. Byte 23 of the record (0x02) is the short string's length, so the
        // text is the record's first two bytes, 0x80 0x01.
        let draw = DrawPatches {
            number_of_patch_control_points: 0x0180,
            patch_index_buffer_reference: 5,
            patch_index_buffer_gpu_address: 0x0200,
            ..Default::default()
        };
        let mut command = vec![0u8; 0x40];
        command[0] = 4;
        command[8..0x40].copy_from_slice(draw.as_bytes());
        let r = IcbResource::new(
            Arc::downgrade(&e.host),
            crate::guest::records::GuestPages::HOST,
            &record(0x40, 3),
            Some(sized_icb(&e, 1)),
            IndirectCommandLayout {
                command_type_offset: 0,
                barrier_offset: 4,
                command_arguments_offset: 8,
                tessellation_factor_offset: 0x30,
                command_size: 0x40,
                ..IndirectCommandLayout::default()
            },
            IndirectCommandBufferFlags {
                bits: 0b11,
                command_types: 4,
                ..IndirectCommandBufferFlags::default()
            },
        )
        .unwrap()
        .unwrap();
        e.p.script().push(
            "cursor_from_virtual_offset",
            Step::value(Some(cursor(&command))),
        );
        e.p.script()
            .push("resource_manager", Step::value(Some(m.clone())));
        let logs = crate::testkit::logs::capture();
        crate::resources::resource::with_platform(&e.p, || r.prepare_with_cpu(Hold::NONE)).unwrap();
        // Returns before host-valid is set.
        assert!(!r.state.host_valid());
        assert_eq!(
            logs.take(),
            vec![(
                log::Level::Warn,
                "reims_vgpu_core::resources::icb".to_owned(),
                String::from_utf8_lossy(
                    b"indirect command buffer not prepared: \
                      tessellation factor not readable: \x80\x01"
                )
                .into_owned()
            )]
        );
    }
    mod layout {
        use super::super::*;
        use std::mem::offset_of;

        #[test]
        fn layout_offsets() {
            assert_eq!(
                offset_of!(IndirectCommandLayout, tessellation_factor_offset),
                6
            );
            assert_eq!(offset_of!(IndirectCommandLayout, pipeline_state_offset), 8);
            assert_eq!(offset_of!(IndirectCommandLayout, command_size), 48);
            let l = IndirectCommandLayout {
                command_type_offset: 0x0102,
                command_size: 0x0a0b_0c0d,
                ..IndirectCommandLayout::default()
            };
            let b = l.as_bytes();
            assert_eq!(&b[0..2], &[2, 1]);
            assert_eq!(&b[48..52], &[0x0d, 0x0c, 0x0b, 0x0a]);
            assert_eq!(IndirectCommandLayout::read_from_bytes(b), Ok(l));
        }

        /// `commandTypes` at 4
        #[test]
        fn flags_offsets() {
            assert_eq!(offset_of!(IndirectCommandBufferFlags, command_types), 4);
            assert_eq!(
                offset_of!(IndirectCommandBufferFlags, max_vertex_buffer_bind_count),
                8
            );
            assert_eq!(
                offset_of!(
                    IndirectCommandBufferFlags,
                    max_kernel_threadgroup_memory_bind_count
                ),
                32
            );
        }

        /// The three one-bit fields of the first word, each alone, and the bits above them ignored
        #[test]
        fn flag_bits() {
            let f = |bits| IndirectCommandBufferFlags {
                bits,
                ..IndirectCommandBufferFlags::default()
            };
            let read = |x: IndirectCommandBufferFlags| {
                (
                    x.inherit_pipeline_state(),
                    x.inherit_buffers(),
                    x.support_dynamic_attribute_stride(),
                )
            };
            assert_eq!(read(f(0)), (false, false, false));
            assert_eq!(read(f(1)), (true, false, false));
            assert_eq!(read(f(2)), (false, true, false));
            assert_eq!(read(f(4)), (false, false, true));
            assert_eq!(read(f(7)), (true, true, true));
            assert_eq!(read(f(0xffff_fff8)), (false, false, false));
        }
    }
}
