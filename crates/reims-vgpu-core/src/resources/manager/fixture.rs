//! The manager's test fixture, shared by the manager's, the lookup's and the resolver's tests
//! (and the tasks' through the compat path): a `ResourceManager<FakePlatform>` over a
//! `FakeResourceHost` task whose object and placement lists live in one block of real memory, and
//! a test kind whose creator the test scripts.

use std::sync::Arc;

use zerocopy::IntoBytes;

use crate::guest::GuestPtr;
use crate::guest::records::ObjectEntry;
use crate::host::HostPtr;
use crate::memory::cursor::Cursor;
use crate::memory::cursor::CursorRange;
use crate::memory::mapping::{Mapping, Teardown};
use crate::platform::Handle;
use crate::platform::device::GpuDevice;
use crate::raise::RaiseResult;
use crate::resources::host::ResourceHost;
use crate::resources::manager::{Cache, Caches, ManagedKind, ResourceManager};
use crate::rt::Hold;
use crate::tasks::object_lists::ObjectListKind;
use crate::tasks::object_lists::ObjectLists;
use crate::testkit::guest_ram::RawRegion;
use crate::testkit::objects::{BufferModel, Model};
use crate::testkit::resource_host::FakeResourceHost;
use crate::testkit::{CallLog, FakePlatform, Fk, Step, kind};
use std::cell::RefCell;
use std::collections::VecDeque;

pub type P = FakePlatform;

/// The task's length and where its parts live (task offsets = offsets into the region).
pub const TASK_LEN: u64 = 0x40000;
pub const ROOT: u64 = 0x100;
pub const OL: u64 = 0x4000;
pub const PL: u64 = 0x8000;
pub const REC: u64 = 0x10000;

/// How the placement list is set up.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// `supportsArgumentBuffers` false: every placement is `Ok(nullptr)`.
    Off,
    /// Mapped at `PL`.
    Mapped,
    /// Declared but its mapping failed: the placement list is not mapped.
    Unmapped,
}

pub struct Fx {
    pub platform: Arc<P>,
    pub task: Arc<FakeResourceHost>,
    pub region: Arc<RawRegion>,
    pub olm: Arc<ObjectLists<P>>,
    pub m: Arc<ResourceManager<P>>,
    pub task_dyn: Arc<dyn ResourceHost<P>>,
}

impl Fx {
    /// A memory-map object list of `entries` (`(type, length, offset)`), mapped.
    pub fn new(entries: &[(u8, u32, u64)], placement: Placement) -> Fx {
        Fx::with_kind(entries, placement, ObjectListKind::MemoryMap)
    }

    pub fn with_kind(entries: &[(u8, u32, u64)], placement: Placement, kind: ObjectListKind) -> Fx {
        let platform = Arc::new(FakePlatform::new());
        let task = FakeResourceHost::new(platform.clone());
        let region = RawRegion::new(TASK_LEN).unwrap();
        let task_dyn: Arc<dyn ResourceHost<P>> = task.clone();
        let legacy = matches!(kind, ObjectListKind::Legacy { .. });
        let olm = ObjectLists::new(
            kind,
            TASK_LEN,
            placement != Placement::Off,
            Arc::downgrade(&task_dyn),
            crate::guest::records::GuestPages::HOST,
        );
        for (i, (t, len, off)) in entries.iter().enumerate() {
            let e = ObjectEntry {
                type_and_length: u32::from(*t) | (len << 8),
                offset: *off,
            };
            assert!(region.write(OL + 12 * i as u64, e.as_bytes()));
        }
        let ol_len = (entries.len() * 12) as u32;
        let pl_len = (entries.len() * 16) as u32;
        let script = platform.script();
        if legacy {
            script.push(
                "mapped_address_for_offset",
                Step::value(region.addr().wrapping_add(OL)),
            );
        } else {
            match placement {
                Placement::Mapped => script.push(
                    "new_virtual_mapping",
                    Step::value(Some(Mapping::<P>::new(
                        region.addr().wrapping_add(PL),
                        0x4000,
                        true,
                        Teardown::None,
                    ))),
                ),
                Placement::Unmapped => script.push("new_virtual_mapping", Step::Nil),
                Placement::Off => {}
            }
            script.push(
                "new_virtual_mapping",
                Step::value(Some(Mapping::<P>::new(
                    region.addr().wrapping_add(OL),
                    0x4000,
                    false,
                    Teardown::None,
                ))),
            );
        }
        let (pl_off, pl) = if placement == Placement::Off {
            (0, 0)
        } else {
            (PL, pl_len)
        };
        assert!(olm.set_object_list(
            Hold::NONE,
            OL,
            ol_len,
            pl_off,
            pl,
            GuestPtr::new(region.addr().wrapping_add(ROOT))
        ));
        assert!(olm.map_object_list(Hold::NONE));
        let m = ResourceManager::new(
            platform.clone(),
            Some(platform.device()),
            Arc::downgrade(&task_dyn),
            olm.clone(),
            true,
        );
        Fx {
            platform,
            task,
            region,
            olm,
            m,
            task_dyn,
        }
    }

    /// Runs `f` inside a `with_platform` scope; answers the messages of the lines this area
    /// logged meanwhile, through the `log` crate (`test_support::logged`).
    pub fn logged<T>(&self, f: impl FnOnce() -> T) -> (T, Vec<String>) {
        crate::resources::resource::test_support::logged(|| {
            crate::resources::resource::with_platform(&self.platform, f)
        })
    }

    pub fn log(&self) -> &CallLog {
        self.platform.call_log()
    }

    pub fn clear_log(&self) -> usize {
        self.log().calls().len()
    }

    /// The ordered calls ([`CallLog::ordered`]) after mark `from` (a [`clear_log`] count).
    ///
    /// [`clear_log`]: Fx::clear_log
    pub fn sels_since(&self, from: usize) -> Vec<&'static str> {
        let calls = self.log().calls();
        let Some(first) = calls.get(from) else {
            return Vec::new();
        };
        self.log()
            .since(first.seq)
            .into_iter()
            .map(|c| c.sel)
            .collect()
    }

    pub fn addr(&self, off: u64) -> HostPtr {
        self.region.addr().wrapping_add(off)
    }

    pub fn write(&self, off: u64, bytes: &[u8]) {
        assert!(self.region.write(off, bytes));
    }

    pub fn placement(&self, reference: u32) -> (u64, u64) {
        let at = PL + 16 * u64::from(reference);
        (
            self.region.read_u64(at).unwrap(),
            self.region.read_u64(at + 8).unwrap(),
        )
    }

    pub fn set_placement(&self, reference: u32, field0: u64, id: u64) {
        let at = PL + 16 * u64::from(reference);
        assert!(self.region.write_u64(at, field0));
        assert!(self.region.write_u64(at + 8, id));
    }

    /// The next `cursor_from_virtual_offset` returns a cursor over `len` bytes at `REC+off`.
    pub fn next_cursor(&self, off: u64, len: u64) {
        self.platform.script().push(
            "cursor_from_virtual_offset",
            Step::value(Some(Cursor::with_ranges(vec![CursorRange {
                address: self.addr(REC + off),
                length: len,
            }]))),
        );
    }

    /// A fake object of kind `K` whose last release records `tag` in `into`.
    pub fn tracked<K: crate::testkit::objects::Kind>(
        &self,
        tag: &'static str,
        into: &Arc<parking_lot::Mutex<Vec<&'static str>>>,
    ) -> Fk<K> {
        let into = into.clone();
        self.platform
            .world()
            .new_obj::<K>(Model::Buffer(BufferModel {
                deallocator: Some(Box::new(move || into.lock().push(tag))),
                ..Default::default()
            }))
    }
}

/// What the test kind's next creation does.
pub enum Create {
    /// A new fence (`newFence`).
    Fence,
    /// `Err(text)`.
    Fail(&'static str),
    /// Raises `NSGenericException`, as a caught send (`Err`).
    Throw,
    /// Looks `reference` up (a nested creation), then makes a fence.
    Nested(u32),
    /// Registers another object under the same reference first (unreachable from the guest).
    Duplicate,
}

/// What the test kind's creator saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seen {
    pub reference: u32,
    pub entry_type: u8,
    pub cursor_len: u64,
    pub placement: Option<usize>,
}

thread_local! {
    pub static CREATE: RefCell<VecDeque<Create>> = const { RefCell::new(VecDeque::new()) };
    pub static SEEN: RefCell<Vec<Seen>> = const { RefCell::new(Vec::new()) };
}

pub fn script_create(c: Create) {
    CREATE.with(|q| q.borrow_mut().push_back(c));
}

pub fn seen() -> Vec<Seen> {
    SEEN.with(|s| s.borrow().clone())
}

/// A kind over the `fences` map whose creator the test scripts.
pub struct TestKind;

impl ManagedKind<P> for TestKind {
    type Obj = Fk<kind::Fence>;
    fn sub_manager(c: &Caches<P>) -> &Cache<Self::Obj> {
        &c.fences
    }
    fn sub_manager_mut(c: &mut Caches<P>) -> &mut Cache<Self::Obj> {
        &mut c.fences
    }
    fn retain(o: &Self::Obj) -> Self::Obj {
        o.clone_ref()
    }
    fn create_object(
        hold: Hold<'_>,
        m: &ResourceManager<P>,
        reference: u32,
        entry: &ObjectEntry,
        cursor: Cursor,
        placement: Option<GuestPtr>,
    ) -> RaiseResult<Result<Self::Obj, String>> {
        SEEN.with(|s| {
            s.borrow_mut().push(Seen {
                reference,
                entry_type: entry.object_type(),
                cursor_len: cursor.remaining(),
                placement: placement.map(|p| p.addr().addr()),
            })
        });
        let next = CREATE
            .with(|q| q.borrow_mut().pop_front())
            .unwrap_or(Create::Fence);
        match next {
            Create::Fence => Ok(Ok(crate::nil_send!(m.device(), new_fence())?.unwrap())),
            Create::Fail(s) => Ok(Err(s.to_owned())),
            Create::Throw => Err(crate::testkit::capture::raise(
                crate::raise::OnFault::Return,
                "site_test",
                "boom",
            )),
            Create::Nested(r) => {
                let inner = m.get_object::<TestKind>(hold, r)?;
                assert!(inner.is_ok());
                Ok(Ok(crate::nil_send!(m.device(), new_fence())?.unwrap()))
            }
            Create::Duplicate => {
                let other = crate::nil_send!(m.device(), new_fence())?.unwrap();
                assert!(m.register_object::<TestKind>(reference, other));
                Ok(Ok(crate::nil_send!(m.device(), new_fence())?.unwrap()))
            }
        }
    }
}

/// A new fence from the manager's device.
pub fn new_fence(m: &ResourceManager<P>) -> Fk<kind::Fence> {
    m.device().unwrap().new_fence().unwrap().unwrap()
}
