//! The resource, resource-manager and deserializer areas composed on
//! `FakePlatform`, with no seam fakes between them.
//!
//! The one seam left is `ResourceHost` (the task, `LocalTask` in the product), which this file
//! implements over one flat region: task offset N is `region + N`. It hands out the real manager
//! and deserializer and records what it is asked.

#![cfg(feature = "testkit")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation
)] // A failure is the assertion

use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use zerocopy::IntoBytes;

use reims_vgpu_core::deserializer::Deserializer;
use reims_vgpu_core::deserializer::records::{
    CmdCopyFromBufferToBuffer, CmdHeader, CmdSetBufferEntry, CmdSetBuffers, CmdSetSamplerStates,
    CmdUseResources, SegmentHeader,
};
use reims_vgpu_core::guest::GuestPtr;
use reims_vgpu_core::guest::records::{
    APV_PAGE_SIZE, FeaturesRef, GuestPages, ObjectEntry, SharedFeatures,
};
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::memory::cursor::{Cursor, CursorRange, WritableCursor};
use reims_vgpu_core::memory::mapping::{Mapping, Teardown};
use reims_vgpu_core::platform::command::CommandQueue;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::{Handle, ObjId};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::raise::{Origin, Raise};
use reims_vgpu_core::resources::host::{CallBorrow, MappedAddress, ResourceHost};
use reims_vgpu_core::resources::manager::ResourceManager;
use reims_vgpu_core::resources::manager::create::Resources;
use reims_vgpu_core::resources::manager::states::SamplerStates;
use reims_vgpu_core::resources::resource::Resource;
use reims_vgpu_core::rt::{Held, Hold, Pool};
use reims_vgpu_core::tasks::object_lists::{ObjectListKind, ObjectLists};
use reims_vgpu_core::testkit::calllog::Arg;
use reims_vgpu_core::testkit::guest_ram::RawRegion;
use reims_vgpu_core::testkit::impls::{Backing, new_buffer};
use reims_vgpu_core::testkit::platform::FakePlatform;
use reims_vgpu_core::testkit::{Fk, kind};

type P = FakePlatform;

/// The task's length, and where its parts live (task offsets).
const TASK_LEN: u64 = 0x40000;
const ROOT: u64 = 0x100;
const OL: u64 = 0x4000;
const REC: u64 = 0x8000;
const CMD: u64 = 0xc000;

/// The object list, by reference (index): `(type, record length, record)`.
/// - 1, 2: shared buffers (type 1) over pages 4 and 5;
/// - 3: a private buffer (type 1, `isPrivate`) of page 6;
/// - 4: a sampler state (type 7, `newSamplerStateWithDescriptor`);
/// - 5: an invalid entry (type 0).
const BUF_LEN: u64 = 0x100;

fn buffer_record(length: u64, page: u32, flags: u8) -> Vec<u8> {
    let mut r = Vec::new();
    r.extend_from_slice(&length.to_le_bytes());
    r.extend_from_slice(&page.to_le_bytes());
    r.push(flags);
    r.extend_from_slice(&[0; 3]);
    r
}

/// `PGOperationHeader { commandType 3 }` + `PGOperationNewSamplerStateWithDescriptor`: min
/// filter 1, mag filter 1, `lodMaxClamp` 4.
fn sampler_record() -> Vec<u8> {
    let mut r = Vec::new();
    r.extend_from_slice(&3u32.to_le_bytes());
    r.extend_from_slice(&0u32.to_le_bytes());
    r.extend_from_slice(&0u32.to_le_bytes()); // field0
    r.extend_from_slice(&0b0101u32.to_le_bytes()); // bits0
    r.extend_from_slice(&0u32.to_le_bytes()); // bits1
    r.extend_from_slice(&0f32.to_le_bytes());
    r.extend_from_slice(&4f32.to_le_bytes());
    r.extend_from_slice(&[0; 8]); // field14
    assert_eq!(r.len(), 8 + 0x1c);
    r
}

fn object_list() -> Vec<(u8, Vec<u8>)> {
    vec![
        (0, Vec::new()),
        (1, buffer_record(BUF_LEN, 4, 0)),
        (1, buffer_record(BUF_LEN, 5, 0)),
        (1, buffer_record(BUF_LEN, 6, 1)),
        (7, sampler_record()),
        (0, Vec::new()),
    ]
}

// ------------------------------------------------------------------------------ the task

/// The task (`PGTask_Resource`, `PGTask_ObjectListManager`) over one region.
struct Task {
    platform: Arc<P>,
    region: Arc<RawRegion>,
    features: FeaturesRef,
    manager: OnceLock<Arc<ResourceManager<P>>>,
    deserializer: OnceLock<Arc<Deserializer<P>>>,
    asked: Mutex<Vec<String>>,
}

impl Task {
    fn addr(&self, offset: u64) -> Option<HostPtr> {
        (offset < self.region.len()).then(|| self.region.addr().wrapping_add(offset))
    }

    fn fits(&self, offset: u64, length: u64) -> bool {
        offset
            .checked_add(length)
            .is_some_and(|end| end <= self.region.len())
    }

    fn ask(&self, s: String) {
        self.asked.lock().push(s);
    }
}

impl ResourceHost<P> for Task {
    fn guest_pages(&self) -> GuestPages {
        GuestPages::HOST
    }

    fn new_buffer_for_virtual_page(
        &self,
        _hold: Hold<'_>,
        page: u32,
        length: u64,
    ) -> RaiseResult<Fk<kind::Buffer>> {
        self.ask(format!("new_buffer_for_virtual_page({page}, {length:#x})"));
        let at = self.addr(u64::from(page) * APV_PAGE_SIZE).unwrap();
        Ok(new_buffer(self.platform.world(), length, 0, Backing::At(at), None).unwrap())
    }

    fn prepare_backing(
        &self,
        _hold: Hold<'_>,
        _backing: &Resource<P>,
        _encoder: Option<&Fk<kind::BlitEncoder>>,
    ) -> RaiseResult<()> {
        panic!("no backing in these scenarios")
    }

    fn synchronize_backing(
        &self,
        _hold: Hold<'_>,
        _backing: &Resource<P>,
        _encoder: Option<&Fk<kind::BlitEncoder>>,
    ) -> RaiseResult<()> {
        panic!("no backing in these scenarios")
    }

    fn retain_io_surface_with_mapping_id(
        &self,
        _identifier: u64,
    ) -> RaiseResult<Option<Fk<kind::Surface>>> {
        panic!("no surface in these scenarios")
    }

    fn release_io_surface_with_mapping_id(
        &self,
        _identifier: u64,
        _surface: Option<Fk<kind::Surface>>,
    ) {
        panic!("no surface in these scenarios")
    }

    fn address_ranges_for_virtual_page(
        &self,
        _hold: Hold<'_>,
        _page: u32,
        _length: u64,
        _read_only: &mut bool,
    ) -> RaiseResult<Option<Fk<kind::AddressRangeArray>>> {
        panic!("the task does not support address ranges")
    }

    fn new_shared_texture_handle_for_id(
        &self,
        _hold: Hold<'_>,
        _identifier: u32,
    ) -> RaiseResult<Option<Fk<kind::SharedTextureHandle>>> {
        panic!("no shared texture in these scenarios")
    }

    fn data_from_virtual_offset(
        &self,
        _hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Vec<u8>>> {
        Ok(self.region.read_vec(offset, length))
    }

    fn backing_for_id(
        &self,
        _hold: Hold<'_>,
        _identifier: u32,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        panic!("no backing in these scenarios")
    }

    fn cursor_from_virtual_offset(
        &self,
        _hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Cursor>> {
        self.ask(format!(
            "cursor_from_virtual_offset({offset:#x}, {length:#x})"
        ));
        if !self.fits(offset, length) {
            return Ok(None);
        }
        Ok(Some(Cursor::with_ranges(vec![CursorRange {
            address: self.region.addr().wrapping_add(offset),
            length,
        }])))
    }

    fn writable_cursor_from_virtual_offset(
        &self,
        _hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<WritableCursor>> {
        if !self.fits(offset, length) {
            return Ok(None);
        }
        Ok(Some(WritableCursor::with_ranges(vec![CursorRange {
            address: self.region.addr().wrapping_add(offset),
            length,
        }])))
    }

    fn deserializer(&self) -> Option<CallBorrow<'_, Deserializer<P>>> {
        self.deserializer.get().cloned().map(CallBorrow::new)
    }

    fn mtl_device(&self) -> Option<Fk<kind::Device>> {
        Some(self.platform.device())
    }

    fn features(&self) -> FeaturesRef {
        self.features.clone()
    }

    fn resource_manager(&self) -> Option<CallBorrow<'_, ResourceManager<P>>> {
        self.manager.get().cloned().map(CallBorrow::new)
    }

    fn blit_in_rgb_2p_xr10_a8_pipeline(&self) -> Option<Fk<kind::ComputePipeline>> {
        None
    }

    fn blit_out_rgb_2p_xr10_a8_pipeline(&self) -> Option<Fk<kind::ComputePipeline>> {
        None
    }

    fn supports_buffer_with_address_ranges(&self) -> bool {
        false
    }

    fn supports_heap_with_address_ranges(&self) -> bool {
        false
    }

    fn supports_resource_detach_backing(&self) -> bool {
        false
    }

    fn paging_queue(&self) -> Option<Fk<kind::CommandQueue>> {
        self.ask("paging_queue".to_owned());
        None
    }

    fn new_virtual_mapping(
        &self,
        _hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<Option<Arc<Mapping<P>>>> {
        self.ask(format!("new_virtual_mapping({offset:#x}, {length:#x})"));
        if !self.fits(offset, length) {
            return Ok(None);
        }
        Ok(Some(Mapping::new(
            self.region.addr().wrapping_add(offset),
            length,
            need_writable,
            Teardown::None,
        )))
    }

    fn mapped_address_for_offset(
        &self,
        _hold: Hold<'_>,
        offset: u64,
        length: u64,
        _need_writable: bool,
    ) -> RaiseResult<MappedAddress<P>> {
        Ok(MappedAddress::unmapped(if self.fits(offset, length) {
            self.region.addr().wrapping_add(offset)
        } else {
            HostPtr::NULL
        }))
    }
}

// ------------------------------------------------------------------------------ the world

struct World {
    p: Arc<P>,
    task: Arc<Task>,
    m: Arc<ResourceManager<P>>,
    d: Arc<Deserializer<P>>,
    queue: Fk<kind::CommandQueue>,
}

impl World {
    /// The task, its object list mapped, its manager (in-process: no GPU address registry) and
    /// its deserializer over the manager's delegate, as `PGLocalTask` wires them.
    fn new() -> World {
        let p = Arc::new(FakePlatform::new());
        let task = Arc::new(Task {
            platform: p.clone(),
            region: RawRegion::new(TASK_LEN).unwrap(),
            features: Arc::new(SharedFeatures::new()),
            manager: OnceLock::new(),
            deserializer: OnceLock::new(),
            asked: Mutex::new(Vec::new()),
        });
        let dyn_task: Arc<dyn ResourceHost<P>> = task.clone();

        // The object list and its records.
        let mut rec = REC;
        let list = object_list();
        for (i, (ty, record)) in list.iter().enumerate() {
            let e = ObjectEntry {
                type_and_length: u32::from(*ty) | ((record.len() as u32) << 8),
                offset: rec,
            };
            assert!(task.region.write(OL + 12 * i as u64, e.as_bytes()));
            assert!(task.region.write(rec, record));
            rec += 0x40;
        }
        let olm = ObjectLists::new(
            ObjectListKind::MemoryMap,
            TASK_LEN,
            false,
            Arc::downgrade(&dyn_task),
            GuestPages::HOST,
        );
        assert!(olm.set_object_list(
            Hold::NONE,
            OL,
            (list.len() * 12) as u32,
            0,
            0,
            GuestPtr::new(task.region.addr().wrapping_add(ROOT)),
        ));
        assert!(olm.map_object_list(Hold::NONE));

        let m = ResourceManager::new(
            p.clone(),
            Some(p.device()),
            Arc::downgrade(&dyn_task),
            olm,
            false,
        );
        let d = Deserializer::new(p.clone(), Some(p.device()), m.object_delegate());
        assert!(task.manager.set(m.clone()).is_ok());
        assert!(task.deserializer.set(d.clone()).is_ok());
        let queue = p.device().new_command_queue().unwrap();
        task.asked.lock().clear();
        p.call_log().clear();
        World {
            p,
            task,
            m,
            d,
            queue,
        }
    }

    /// Writes `stream` at `CMD` and decodes it as one command buffer's segments, under a live
    /// handler, with an autorelease pool and a holder, as the task's exec does in its FIFO
    /// command.
    fn exec(&self, stream: &[u8]) -> Result<(), Raise> {
        assert!(self.task.region.write(CMD, stream));
        let cb = self.queue.command_buffer(OnFault::Die).unwrap().unwrap();
        let _pool = Pool::push(&*self.p);
        let held = Held::new();
        {
            let hold = Hold::NONE;
            let cursor = self
                .task
                .cursor_from_virtual_offset(hold, CMD, stream.len() as u64)
                .unwrap();
            self.task.asked.lock().clear();
            self.d
                .decode_segments(Hold::of(&held), &mut [cursor], Some(&cb))
                .map(drop)
        }
    }

    fn asked(&self) -> Vec<String> {
        std::mem::take(&mut *self.task.asked.lock())
    }

    /// The `MTLBuffer` the manager's cached resource `reference` holds.
    fn cached_buffer(&self, reference: u32) -> Option<ObjId> {
        let r = self.m.get_existing_object::<Resources>(reference)??;
        r.get_buffer().unwrap().map(|b| b.id())
    }

    /// The selectors and arguments sent to objects of `class`, in order.
    fn sent(&self, class: &str) -> Vec<(&'static str, Vec<Arg>)> {
        self.p
            .call_log()
            .calls_of(class)
            .into_iter()
            .map(|c| (c.sel, c.args))
            .collect()
    }
}

/// A segment of `records`, each `(commandID, record bytes)`.
fn segment(encoder_type: u8, records: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (id, bytes) in records {
        let h = CmdHeader {
            command_id: *id,
            length: 8 + bytes.len() as u32,
        };
        body.extend_from_slice(h.as_bytes());
        body.extend_from_slice(bytes);
    }
    let h = SegmentHeader {
        length: 8 + body.len() as u32,
        encoder_type,
        continues_previous: 0,
        continued_by_next: 0,
        unknown07: 0,
    };
    let mut s = h.as_bytes().to_vec();
    s.extend_from_slice(&body);
    s
}

fn copy(source: u32, destination: u32) -> (u32, Vec<u8>) {
    (
        301,
        CmdCopyFromBufferToBuffer {
            source_buffer_reference: source,
            destination_buffer_reference: destination,
            source_offset: 0x10,
            destination_offset: 0x20,
            size: 0x40,
        }
        .as_bytes()
        .to_vec(),
    )
}

fn raise_origin(t: &Raise) -> (&'static str, u32) {
    match t.origin() {
        Origin::Platform(sel) => panic!("a Metal raise from {sel}"),
        Origin::Boundary(selector) => panic!("a Metal raise from {selector}"),
        Origin::Core(_) => (t.site().file, t.site().line),
    }
}

// ------------------------------------------------------------------------------ scenarios

/// A blit copy's two buffer references are created on first use through the delegate, the manager
/// and `PGSharedBufferResource`, in lookup order (source, then destination), and the second command
/// buffer finds them cached.
#[test]
fn a_blit_creates_its_buffers_once_through_the_manager() {
    let w = World::new();
    w.exec(&segment(2, &[copy(1, 2)])).unwrap();
    assert_eq!(
        w.asked(),
        vec![
            "cursor_from_virtual_offset(0x8040, 0x10)",
            "new_buffer_for_virtual_page(4, 0x100)",
            "cursor_from_virtual_offset(0x8080, 0x10)",
            "new_buffer_for_virtual_page(5, 0x100)",
        ]
    );
    let (b1, b2) = (w.cached_buffer(1).unwrap(), w.cached_buffer(2).unwrap());
    assert_ne!(b1, b2);
    let expect = vec![
        (
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
            vec![
                Arg::Obj(Some(b1)),
                Arg::U64(0x10),
                Arg::Obj(Some(b2)),
                Arg::U64(0x20),
                Arg::U64(0x40),
            ],
        ),
        ("endEncoding", vec![]),
    ];
    assert_eq!(w.sent("MTLBlitCommandEncoder"), expect);

    // A second command buffer: the objects are cached, nothing is read or created again.
    w.p.call_log().clear();
    w.exec(&segment(2, &[copy(1, 2)])).unwrap();
    assert!(w.asked().is_empty());
    assert_eq!(w.sent("MTLBlitCommandEncoder"), expect);
    assert_eq!(w.p.call_log().count("newBufferWithLength:options:"), 0);
}

/// A compute segment binds a shared and a private buffer and a sampler state, all created by the
/// manager's real creators, and `useResources` hands the encoder the resources' buffers, with the
/// invalid reference as nil (the delegate's `getResourceForReference:` swallows the error). No
/// dispatch: that needs a pipeline state, whose function a library compiled from guest IR
/// provides.
#[test]
fn a_compute_segment_binds_objects_the_manager_creates() {
    let w = World::new();
    let set_buffers = CmdSetBuffers {
        range_location: 0,
        range_length: 2,
    };
    let mut set_buffers_rec = set_buffers.as_bytes().to_vec();
    for (reference, offset) in [(1u32, 0u64), (3, 8)] {
        set_buffers_rec.extend_from_slice(
            CmdSetBufferEntry {
                buffer_reference: reference,
                offset,
            }
            .as_bytes(),
        );
    }
    let mut set_samplers_rec = CmdSetSamplerStates {
        range_location: 2,
        range_length: 1,
    }
    .as_bytes()
    .to_vec();
    set_samplers_rec.extend_from_slice(&4u32.to_le_bytes());
    let mut use_rec = CmdUseResources { count: 2, usage: 1 }.as_bytes().to_vec();
    use_rec.extend_from_slice(&3u32.to_le_bytes());
    use_rec.extend_from_slice(&5u32.to_le_bytes());
    let stream = segment(
        1,
        &[
            (0xcb, set_buffers_rec),
            (0xcc, set_samplers_rec),
            (0x87, use_rec),
        ],
    );
    w.exec(&stream).unwrap();

    // The private buffer is the device's (private storage), never the task's pages.
    let made: Vec<Vec<Arg>> =
        w.p.call_log()
            .calls_to("newBufferWithLength:options:")
            .into_iter()
            .map(|c| c.args)
            .collect();
    assert_eq!(made, vec![vec![Arg::U64(BUF_LEN), Arg::U64(0x20)]]);
    assert_eq!(w.p.call_log().count("newSamplerStateWithDescriptor:"), 1);
    assert_eq!(
        w.asked(),
        vec![
            "cursor_from_virtual_offset(0x8040, 0x10)",
            "new_buffer_for_virtual_page(4, 0x100)",
            "cursor_from_virtual_offset(0x80c0, 0x10)",
            "cursor_from_virtual_offset(0x8100, 0x24)",
            // reference 5 is refused before any cursor: its type is 0
        ]
    );

    let b1 = w.cached_buffer(1).unwrap();
    let b3 = w.cached_buffer(3).unwrap();
    let sampler =
        w.m.get_existing_object::<SamplerStates>(4)
            .unwrap()
            .unwrap();
    let sent = w.sent("MTLComputeCommandEncoder");
    let sels: Vec<&str> = sent.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        sels,
        vec![
            "setBuffers:offsets:withRange:",
            "setSamplerStates:withRange:",
            "useResources:count:usage:",
            "endEncoding",
        ]
    );
    assert_eq!(sent[0].1[0], Arg::Objs(vec![Some(b1), Some(b3)]));
    assert_eq!(sent[0].1[1], Arg::U64s(vec![0, 8]));
    assert_eq!(sent[1].1[0], Arg::Objs(vec![Some(sampler.id())]));
    assert_eq!(sent[2].1[0], Arg::Objs(vec![Some(b3), None]));
    assert_eq!(sent[2].1[1], Arg::U64(1));
    // Nothing needed a paging queue.
    assert!(w.asked().is_empty());
}

/// A lookup error: the delegate logs and raises the buffer's lookup failure, the blit decoder
/// faults (`endEncoding`) and rethrows, and `decodeSegments` rethrows. The source looked up
/// before the failure stays cached.
#[test]
fn a_failed_lookup_faults_the_decoder_and_keeps_what_was_made() {
    let w = World::new();
    let t = w.exec(&segment(2, &[copy(1, 5)])).unwrap_err();
    assert_eq!(raise_origin(&t), ("manager.lookup.buffer", 0));
    assert_eq!(t.detail().unwrap_or_default(), "buffer lookup failed");
    assert_eq!(
        w.sent("MTLBlitCommandEncoder"),
        vec![("endEncoding", vec![])]
    );
    assert!(w.cached_buffer(1).is_some());
    assert!(w.m.get_existing_object::<Resources>(5).is_none());
}

/// A delete erases the cached resource; the next command buffer's lookup reads the record
/// again and makes a new buffer.
#[test]
fn a_deleted_buffer_is_made_again_on_its_next_use() {
    let w = World::new();
    w.exec(&segment(2, &[copy(1, 2)])).unwrap();
    let before = w.cached_buffer(1).unwrap();
    w.asked();
    w.m.delete_object::<Resources>(Hold::NONE, 1, true);
    assert!(w.m.get_existing_object::<Resources>(1).is_none());
    assert!(w.cached_buffer(2).is_some());

    w.exec(&segment(2, &[copy(1, 2)])).unwrap();
    assert_eq!(
        w.asked(),
        vec![
            "cursor_from_virtual_offset(0x8040, 0x10)",
            "new_buffer_for_virtual_page(4, 0x100)",
        ]
    );
    let after = w.cached_buffer(1).unwrap();
    assert_ne!(before, after);
}
