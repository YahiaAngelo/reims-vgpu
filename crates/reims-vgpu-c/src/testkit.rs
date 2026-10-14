//! Unit-test rigs: the fake host's callbacks as a C table, a recorder of C calls, and a device
//! made through the C API on this test thread's platform.

use std::ffi::c_void;
use std::sync::Arc;

use parking_lot::Mutex;
use reims_vgpu::{HostPtr, HostTask, HostTraceRange, PhysicalMemoryRange};
use reims_vgpu_core::host::HostBlocks;
use reims_vgpu_core::testkit::host::FakeHost;
use reims_vgpu_core::testkit::{FakePlatform, GuestRam};

use crate::device::*;
use crate::status::RVG_OK;
use crate::trace::{
    RvgTraceRangeHandler, rvg_trace_range_handler_invoke, rvg_trace_range_handler_release,
    rvg_trace_range_handler_retain,
};

/// This test thread's platform: the process's, as the API sees it.
pub(crate) fn test_platform() -> Arc<FakePlatform> {
    <FakePlatform as reims_vgpu::ApiPlatform>::process()
}

/// What C functions saw: one line per call, and the thread of each destroy.
#[derive(Default)]
pub(crate) struct Rec {
    lines: Mutex<Vec<String>>,
    destroy_threads: Mutex<Vec<std::thread::ThreadId>>,
}

impl Rec {
    pub(crate) fn new() -> Arc<Rec> {
        Arc::new(Rec::default())
    }

    /// The recorder as a ctx (the test keeps the `Arc`).
    pub(crate) fn ctx(self: &Arc<Rec>) -> *mut c_void {
        Arc::as_ptr(self).cast_mut().cast()
    }

    /// The recorder behind a ctx.
    pub(crate) fn of<'a>(ctx: *mut c_void) -> &'a Rec {
        // SAFETY: every recorder ctx is `Arc::as_ptr` of a `Rec` the test keeps alive.
        unsafe { &*ctx.cast::<Rec>() }
    }

    pub(crate) fn push(ctx: *mut c_void, line: String) {
        Rec::of(ctx).lines.lock().push(line);
    }

    pub(crate) fn lines(&self) -> Vec<String> {
        self.lines.lock().clone()
    }

    pub(crate) fn destroys(&self) -> Vec<std::thread::ThreadId> {
        self.destroy_threads.lock().clone()
    }

    /// The destroys, once `n` have run: waits up to 10 s for those a library thread runs after
    /// the call that ended their tables has returned. A display's runs on the display thread,
    /// where the cancel of its new-frame source frees the last of its closures, and the header
    /// lets it come after the release. A test that drops the recorder first leaves that destroy
    /// a freed ctx, so it waits here. If they do not all come, the recorder is kept for good, and
    /// a destroy that comes later still finds it: the failure is the caller's assertion.
    pub(crate) fn wait_destroys(self: &Arc<Rec>, n: usize) -> Vec<std::thread::ThreadId> {
        for _ in 0..1000 {
            let got = self.destroys();
            if got.len() >= n {
                return got;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        std::mem::forget(self.clone());
        self.destroys()
    }
}

/// A table's destroy: records its thread.
pub(crate) unsafe extern "C-unwind" fn t_destroy(ctx: *mut c_void) {
    Rec::of(ctx)
        .destroy_threads
        .lock()
        .push(std::thread::current().id());
}

/// The fake host's blocks and the recorder, as one table's ctx.
pub(crate) struct HostCtx {
    pub(crate) rec: Arc<Rec>,
    pub(crate) blocks: HostBlocks,
}

fn host<'a>(ctx: *mut c_void) -> &'a HostCtx {
    // SAFETY: every host ctx is `Arc::as_ptr` of a `HostCtx` the test keeps alive.
    unsafe { &*ctx.cast::<HostCtx>() }
}

unsafe extern "C-unwind" fn h_destroy(ctx: *mut c_void) {
    // SAFETY: the recorder's ctx is its own pointer.
    unsafe { t_destroy(host(ctx).rec.ctx()) };
}
unsafe extern "C-unwind" fn h_raise(ctx: *mut c_void, v: u32) {
    host(ctx).blocks.raise_interrupt.as_ref().unwrap()(v);
}
unsafe extern "C-unwind" fn h_create(
    ctx: *mut c_void,
    size: u64,
    base: *mut *mut c_void,
) -> *mut c_void {
    // SAFETY: the caller's out-parameter.
    let mut b = HostPtr::from_ptr(unsafe { *base });
    let t = host(ctx).blocks.create_task.as_ref().unwrap()(size, &mut b);
    // SAFETY: as above.
    unsafe { *base = b.as_ptr() };
    std::ptr::with_exposed_provenance_mut(t.addr())
}
unsafe extern "C-unwind" fn h_destroy_task(ctx: *mut c_void, t: *mut c_void) {
    host(ctx).blocks.destroy_task.as_ref().unwrap()(HostTask::from_addr(t.expose_provenance()));
}
unsafe extern "C-unwind" fn h_map(
    ctx: *mut c_void,
    t: *mut c_void,
    n: u32,
    off: u64,
    ro: bool,
    r: *const RvgPhysicalMemoryRange,
) -> bool {
    let ranges: Vec<PhysicalMemoryRange> = if n == 0 {
        Vec::new()
    } else {
        // SAFETY: `n` ranges, borrowed for the call.
        unsafe { std::slice::from_raw_parts(r, n as usize) }
            .iter()
            .map(|&r| r.into())
            .collect()
    };
    host(ctx).blocks.map_memory.as_ref().unwrap()(
        HostTask::from_addr(t.expose_provenance()),
        n,
        off,
        ro,
        &ranges,
    )
}
unsafe extern "C-unwind" fn h_unmap(ctx: *mut c_void, t: *mut c_void, off: u64, len: u64) -> bool {
    host(ctx).blocks.unmap_memory.as_ref().unwrap()(
        HostTask::from_addr(t.expose_provenance()),
        off,
        len,
    )
}
unsafe extern "C-unwind" fn h_read(ctx: *mut c_void, pa: u64, len: u64, dst: *mut c_void) -> bool {
    host(ctx).blocks.read_memory.as_ref().unwrap()(pa, len, HostPtr::from_ptr(dst))
}

/// A retained `RvgTraceRangeHandler` the fake host keeps; released with it.
struct Kept(*mut RvgTraceRangeHandler);
// SAFETY: the handler is callable from any thread.
unsafe impl Send for Kept {}
// SAFETY: as above.
unsafe impl Sync for Kept {}
impl Drop for Kept {
    fn drop(&mut self) {
        // SAFETY: the reference retained in h_add_trace.
        unsafe { rvg_trace_range_handler_release(self.0) };
    }
}

unsafe extern "C-unwind" fn h_add_trace(
    ctx: *mut c_void,
    r: RvgPhysicalMemoryRange,
    h: *mut RvgTraceRangeHandler,
) -> *mut c_void {
    // SAFETY: the borrowed handler, retained to keep.
    let kept = Arc::new(Kept(unsafe { rvg_trace_range_handler_retain(h) }));
    let handler: reims_vgpu::TraceRangeHandler = Arc::new(move |dirty| {
        // SAFETY: kept alive by `kept`.
        unsafe { rvg_trace_range_handler_invoke(kept.0, dirty.into()) }
    });
    let t = host(ctx).blocks.add_trace_range.as_ref().unwrap()(r.into(), handler);
    std::ptr::with_exposed_provenance_mut(t.addr())
}
unsafe extern "C-unwind" fn h_remove_trace(ctx: *mut c_void, r: *mut c_void) {
    host(ctx).blocks.remove_trace_range.as_ref().unwrap()(HostTraceRange::from_addr(
        r.expose_provenance(),
    ));
}

/// The fake host's callbacks as a C table over `ctx`, with its destroy.
pub(crate) fn host_table(ctx: &Arc<HostCtx>) -> RvgHostCallbacks {
    RvgHostCallbacks {
        struct_size: crate::sized::size_u32::<RvgHostCallbacks>(),
        ctx: Arc::as_ptr(ctx).cast_mut().cast(),
        destroy: Some(h_destroy),
        create_task: Some(h_create),
        destroy_task: Some(h_destroy_task),
        map_memory: Some(h_map),
        unmap_memory: Some(h_unmap),
        read_memory: Some(h_read),
        raise_interrupt: Some(h_raise),
        add_trace_range: Some(h_add_trace),
        remove_trace_range: Some(h_remove_trace),
    }
}

/// A config over `table` with `ports` display ports and nothing else set.
pub(crate) fn config(ports: u32, table: &RvgHostCallbacks) -> RvgDeviceConfig {
    RvgDeviceConfig {
        struct_size: crate::sized::size_u32::<RvgDeviceConfig>(),
        display_port_count: ports,
        metal_registry_id: 0,
        feature_level: 0,
        display_port_connection_types: 0,
        use_iosurface_mapper: false,
        process_isolation: false,
        disable_argument_buffers: false,
        memory_map: std::ptr::null(),
        callbacks: table,
        memory_files: std::ptr::null(),
        process_isolation_follows_guest: false,
        guest_page_shift: 0,
    }
}

/// A device made through the C API on this thread's platform, over a fake host.
pub(crate) struct Rig {
    pub(crate) p: Arc<FakePlatform>,
    pub(crate) _host: Arc<FakeHost>,
    pub(crate) ctx: Arc<HostCtx>,
    pub(crate) device: *mut RvgDevice,
}

impl Rig {
    /// A device with every host callback and `ports` display ports.
    pub(crate) fn new(ports: u32) -> Rig {
        Rig::with(ports, |_| {})
    }

    /// A device with every host callback and `ports` display ports, its config as `edit` leaves
    /// it.
    pub(crate) fn with(ports: u32, edit: impl FnOnce(&mut RvgDeviceConfig)) -> Rig {
        let p = test_platform();
        let host = FakeHost::new(&p, GuestRam::with_range(0, 0x40_0000).unwrap());
        let ctx = Arc::new(HostCtx {
            rec: Rec::new(),
            blocks: host.blocks(),
        });
        let table = host_table(&ctx);
        let mut c = config(ports, &table);
        edit(&mut c);
        let mut device = std::ptr::null_mut();
        // SAFETY: a live config; `device` is writable.
        assert_eq!(unsafe { rvg_device_create(&c, &mut device) }, RVG_OK);
        Rig {
            p,
            _host: host,
            ctx,
            device,
        }
    }

    /// Releases the rig's device now.
    pub(crate) fn release(&mut self) {
        // SAFETY: the rig's reference, given back once.
        unsafe { rvg_device_release(std::mem::replace(&mut self.device, std::ptr::null_mut())) };
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.release();
    }
}

/// An `add_trace_range` that keeps nothing and answers NULL.
pub(crate) unsafe extern "C-unwind" fn t_add_trace(
    _ctx: *mut c_void,
    _r: RvgPhysicalMemoryRange,
    _h: *mut RvgTraceRangeHandler,
) -> *mut c_void {
    std::ptr::null_mut()
}
