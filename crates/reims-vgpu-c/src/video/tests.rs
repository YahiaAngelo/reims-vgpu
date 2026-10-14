use super::*;
use crate::device::{RvgMemoryMap, RvgMemoryRange};
use crate::iosurface::{
    RvgIOSurfaceDeviceConfig, rvg_iosurface_device_create, rvg_iosurface_device_release,
};
use crate::testkit::{Rec, t_destroy, test_platform};

unsafe extern "C-unwind" fn t_send(ctx: *mut c_void, _message: *const u8, length: usize) {
    Rec::push(ctx, format!("send {length}"));
}

fn callbacks(rec: &Arc<Rec>) -> RvgVideoCallbacks {
    RvgVideoCallbacks {
        struct_size: sized::size_u32::<RvgVideoCallbacks>(),
        ctx: rec.ctx(),
        destroy: Some(t_destroy),
        send_to_guest: Some(t_send),
    }
}

/// The C API's create, with a device that opens none of the host's backends: what these tests
/// check is the C API's and the device's, the same whatever the host decodes on, and their
/// platform's clock is virtual, which no backend's wait for its GPU would pass. The C harness
/// makes its device through rvg_video_device_create itself, on the host's backend.
///
/// # Safety
/// As rvg_video_device_create's.
unsafe fn create_without_backend(
    config: *const RvgVideoDeviceConfig,
    device: *mut *mut RvgVideoDevice,
) -> RvgStatus {
    let make = |config| {
        VideoDevice::without_backend(
            <Plat as reims_vgpu::ApiPlatform>::process(),
            config,
            "a test's, on the testkit platform's virtual clock",
        )
    };
    // SAFETY: the caller's contract.
    unsafe { create(config, device, &make) }
}

/// The guest's `decs`, with no surfaces: a `vtpv` request it expects a reply to (flags 1), of the
/// zero UUID and reply identifier 41, with `,ver` as `=u32` 6.
fn decs() -> Vec<u8> {
    let mut m = Vec::new();
    m.extend_from_slice(&0x30u32.to_be_bytes());
    m.extend_from_slice(b"vtpvdecs");
    m.extend_from_slice(&1u32.to_be_bytes());
    m.resize(0x20, 0);
    m.extend_from_slice(&41u64.to_be_bytes());
    m.resize(0x30, 0);
    m.extend_from_slice(&20u32.to_be_bytes());
    m.extend_from_slice(b"para,ver=u32");
    m.extend_from_slice(&6u32.to_be_bytes());
    m
}

/// No config, or no out-pointer, is refused, the latter with the table's destroy; a device
/// answers RVG_VIDEO_INVALID_SESSION until it starts and after a reset, its session answers
/// `decs` with one reply through send_to_guest, and an empty message, or any once invalidated,
/// with an error, and the device ends its table at its release
#[test]
fn a_device_runs_its_session_from_start_to_reset() {
    let _p = test_platform();
    let rec = Rec::new();
    let cb = callbacks(&rec);
    let c = RvgVideoDeviceConfig {
        struct_size: sized::size_u32::<RvgVideoDeviceConfig>(),
        iosurface_device: std::ptr::null(),
        callbacks: &cb,
    };
    let mut d = std::ptr::null_mut();
    let m = decs();
    // SAFETY: live configs; `d` is writable; the device made is used until its release.
    unsafe {
        assert_eq!(
            rvg_video_device_create(std::ptr::null(), &mut d),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(rec.destroys().len(), 0);
        assert_eq!(
            rvg_video_device_create(&c, std::ptr::null_mut()),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(rec.destroys().len(), 1);
        assert_eq!(create_without_backend(&c, &mut d), RVG_OK);
        assert_eq!(
            rvg_video_device_deliver(d, m.as_ptr(), m.len()),
            RVG_VIDEO_INVALID_SESSION
        );
        rvg_video_device_start(d);
        assert_eq!(rvg_video_device_deliver(d, m.as_ptr(), m.len()), 0);
        assert_ne!(rvg_video_device_deliver(d, std::ptr::null(), 0), 0);
        rvg_video_device_invalidate(d);
        assert_ne!(
            rvg_video_device_deliver(d, m.as_ptr(), m.len()),
            0,
            "an invalidated session answers no message"
        );
        rvg_video_device_reset(d);
        assert_eq!(
            rvg_video_device_deliver(d, m.as_ptr(), m.len()),
            RVG_VIDEO_INVALID_SESSION
        );
        rvg_video_device_start(d);
        rvg_video_device_release(d);
        rvg_video_device_release(std::ptr::null_mut());
    }
    assert_eq!(rec.destroys().len(), 2);
    let sends: Vec<_> = (rec.lines().into_iter())
        .filter(|l| l.starts_with("send"))
        .collect();
    assert_eq!(
        sends.len(),
        1,
        "one reply, before the invalidation: {sends:?}"
    );
}

/// A device made over an IOSurface device keeps no reference to it: the IOSurface device's
/// release is its teardown, and the video device's lookups find no surface after it
#[test]
fn an_iosurface_device_is_borrowed() {
    let p = test_platform();
    let vm = reims_vgpu_core::platform::Platform::vm(&*p);
    let len = vm.page_size();
    let host = vm.allocate_anywhere(len).unwrap();
    let range = RvgMemoryRange {
        physical_address: 0,
        physical_length: len,
        host_address: host.as_ptr(),
    };
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    let ic = RvgIOSurfaceDeviceConfig {
        struct_size: sized::size_u32::<RvgIOSurfaceDeviceConfig>(),
        memory_map: &map,
        callbacks: std::ptr::null(),
        guest_page_shift: 0,
    };
    let mut io = std::ptr::null_mut();
    let mut d = std::ptr::null_mut();
    let mut m = Vec::new();
    m.extend_from_slice(&0x30u32.to_be_bytes());
    m.extend_from_slice(b"subs");
    m.extend_from_slice(&7u64.to_be_bytes());
    m.resize(0x30, 0);
    m.extend_from_slice(&decs());
    // SAFETY: live configs; the handles made are used until their release.
    unsafe {
        assert_eq!(rvg_iosurface_device_create(&ic, &mut io), RVG_OK);
        let c = RvgVideoDeviceConfig {
            struct_size: sized::size_u32::<RvgVideoDeviceConfig>(),
            iosurface_device: io,
            callbacks: std::ptr::null(),
        };
        assert_eq!(create_without_backend(&c, &mut d), RVG_OK);
        rvg_video_device_start(d);
        assert_eq!(rvg_video_device_deliver(d, m.as_ptr(), m.len()), 0);
        rvg_iosurface_device_release(io);
        assert_eq!(rvg_video_device_deliver(d, m.as_ptr(), m.len()), 0);
        rvg_video_device_release(d);
    }
}
