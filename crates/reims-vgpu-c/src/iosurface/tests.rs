use reims_vgpu_core::testkit::FakePlatform;

use super::*;
use crate::device::RvgMemoryRange;
use crate::status::*;
use crate::testkit::{Rec, t_destroy, test_platform};

unsafe extern "C-unwind" fn t_raise(ctx: *mut c_void, vector: u32) {
    Rec::push(ctx, format!("raise {vector}"));
}

fn callbacks(rec: &Arc<Rec>) -> RvgIOSurfaceCallbacks {
    RvgIOSurfaceCallbacks {
        struct_size: sized::size_u32::<RvgIOSurfaceCallbacks>(),
        ctx: rec.ctx(),
        destroy: Some(t_destroy),
        raise_interrupt: Some(t_raise),
    }
}

/// One page of the fake VM's memory.
fn page(p: &FakePlatform) -> RvgMemoryRange {
    let vm = reims_vgpu_core::platform::Platform::vm(p);
    let len = vm.page_size();
    let host = vm.allocate_anywhere(len).unwrap();
    RvgMemoryRange {
        physical_address: 0,
        physical_length: len,
        host_address: host.as_ptr(),
    }
}

/// No memory map is refused with the table's destroy; one is made, answers core's registers,
/// and ends its table at its release
#[test]
fn made_over_a_memory_map() {
    let p = test_platform();
    let rec = Rec::new();
    let cb = callbacks(&rec);
    let mut c = RvgIOSurfaceDeviceConfig {
        struct_size: sized::size_u32::<RvgIOSurfaceDeviceConfig>(),
        memory_map: std::ptr::null(),
        callbacks: &cb,
        guest_page_shift: 0,
    };
    let mut d = std::ptr::null_mut();
    // SAFETY: a live config; `d` is writable.
    unsafe {
        assert_eq!(
            rvg_iosurface_device_create(std::ptr::null(), &mut d),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            rvg_iosurface_device_create(&c, &mut d),
            RVG_ERR_NO_MEMORY_MAP
        );
    }
    assert_eq!(rec.destroys().len(), 1);
    let range = page(&p);
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    c.memory_map = &map;
    // SAFETY: as above.
    unsafe {
        assert_eq!(
            rvg_iosurface_device_create(&c, std::ptr::null_mut()),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(rvg_iosurface_device_create(&c, &mut d), RVG_OK);
    }
    assert_eq!(rec.destroys().len(), 2);
    // SAFETY: the device made above.
    let core = unsafe { &*d }.device.core();
    for off in (0x1000..0x1038).step_by(8) {
        // SAFETY: as above.
        assert_eq!(
            unsafe { rvg_iosurface_device_mmio_read(d, off) },
            core.mmio_read(off)
        );
    }
    // SAFETY: as above; an offset that names no register is ignored.
    unsafe {
        rvg_iosurface_device_mmio_write(d, 0, 1);
        rvg_iosurface_device_release(d);
        rvg_iosurface_device_release(std::ptr::null_mut());
    }
    assert_eq!(rec.destroys().len(), 3);
}

/// The config's first version ends at `callbacks`, 24 bytes; `guest_page_shift` makes it 32
/// with its padding, and a host built against the first passes 24 and gets the library's pages
#[test]
fn a_first_version_config_is_24_bytes() {
    assert_eq!(<RvgIOSurfaceDeviceConfig as HostStruct>::V1_SIZE, 24);
    assert_eq!(
        std::mem::offset_of!(RvgIOSurfaceDeviceConfig, guest_page_shift),
        24
    );
    assert_eq!(size_of::<RvgIOSurfaceDeviceConfig>(), 32);
}

/// 16 KiB guest pages at `B`: the ring, the descriptor table, the descriptor page, a surface's
/// root table and its two pages.
const P16: u64 = 0x4000;
const B: u64 = 0x10_0000;
const RING: u64 = B;
const TABLE: u64 = B + P16;
const DPAGE: u64 = B + 2 * P16;
const ROOT: u64 = B + 3 * P16;
const SURF: u64 = B + 4 * P16;

/// A page-table entry naming `pa` in 16 KiB pages.
fn pte16(pa: u64) -> u32 {
    (u32::try_from(pa >> 14).unwrap() << 2) | 1
}

/// A device made through the C API with `shift` over eight 16 KiB pages at `B` that an arm64
/// guest laid out for a two-page second-form surface at mapping id 15, which the guest then maps
/// through the ring: the status the device stores back, and whether the device holds a surface
/// at that id.
fn map_through_the_ring(shift: u32) -> (i32, bool) {
    let p = test_platform();
    let rec = Rec::new();
    let cb = callbacks(&rec);
    let vm = reims_vgpu_core::platform::Platform::vm(&*p);
    let len = 8 * P16;
    let host = vm.allocate_anywhere(len).unwrap();
    let range = RvgMemoryRange {
        physical_address: B,
        physical_length: len,
        host_address: host.as_ptr(),
    };
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    let put = |pa: u64, bytes: &[u8]| {
        let at = usize::try_from(pa - B).unwrap();
        // SAFETY: inside the fake VM's `len` bytes at `host`, which live with the platform.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), host.as_ptr::<u8>().add(at), bytes.len());
        }
    };
    put(TABLE, &pte16(DPAGE).to_le_bytes());
    put(ROOT, &pte16(SURF).to_le_bytes());
    put(ROOT + 4, &pte16(SURF + P16).to_le_bytes());
    // The second-form descriptor's base, at descriptor 15 of the page.
    let mut d = reims_vgpu_core::iosurface::IoSurfaceDeviceDescriptor2::default();
    d.base.word0 = pte16(ROOT);
    d.base.pixel_format = 0x4247_5241;
    d.base.alloc_size = P16 + 1;
    d.base.width = 64;
    d.base.height = 32;
    d.base.bytes_per_row = 256;
    d.base.bytes_per_element = 4;
    d.base.element_width = 1;
    d.base.element_height = 1;
    // SAFETY: the descriptor is plain repr(C) data, read as its bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(&d).cast::<u8>(),
            size_of::<reims_vgpu_core::iosurface::IoSurfaceDeviceDescriptor2>(),
        )
    };
    put(DPAGE + 15 * 0x400, bytes);
    // Command 1 (map) of mapping id 15 in ring slot 0, its status pre-set.
    put(RING, &1u32.to_le_bytes());
    put(RING + 4, &15u32.to_le_bytes());
    put(RING + 8, &0x5555u32.to_le_bytes());
    p.fake_vm().sync();
    let c = RvgIOSurfaceDeviceConfig {
        struct_size: sized::size_u32::<RvgIOSurfaceDeviceConfig>(),
        memory_map: &map,
        callbacks: &cb,
        guest_page_shift: shift,
    };
    let mut dev = std::ptr::null_mut();
    // SAFETY: a live config and map; `dev` is writable; then the device made.
    unsafe {
        assert_eq!(rvg_iosurface_device_create(&c, &mut dev), RVG_OK);
        rvg_iosurface_device_mmio_write(dev, 0x1030, 1); // the second form
        rvg_iosurface_device_mmio_write(dev, 0x1000, RING);
        rvg_iosurface_device_mmio_write(dev, 0x1008, 1024); // 16 KiB of commands
        rvg_iosurface_device_mmio_write(dev, 0x1010, TABLE);
        rvg_iosurface_device_mmio_write(dev, 0x1018, 1); // the doorbell
    }
    let mut tries = 0;
    while !rec.lines().iter().any(|l| l == "raise 0") {
        tries += 1;
        assert!(tries < 10_000, "the ring never raised");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    p.fake_vm().sync();
    // SAFETY: inside the region, as above.
    let status = unsafe {
        host.as_ptr::<u8>()
            .add(usize::try_from(RING + 8 - B).unwrap())
            .cast::<i32>()
            .read_unaligned()
    };
    // SAFETY: the device made above.
    let mapped = unsafe { &*dev }
        .device
        .core()
        .retain_surface_with_mapping_id(15)
        .is_some();
    // SAFETY: as above.
    unsafe { rvg_iosurface_device_release(dev) };
    (status, mapped)
}

/// With `guest_page_shift` 14 an arm64 guest's 16 KiB page numbers map its surface through the
/// ring, on any build; the same layout read in this x86-64 build's own 4 KiB pages names pages
/// outside its memory, and the map is NoMemory
#[test]
fn an_arm64_guest_maps_a_surface_in_16k_pages() {
    assert_eq!(map_through_the_ring(14), (0, true));
    if cfg!(target_arch = "x86_64") {
        assert_eq!(
            map_through_the_ring(0),
            (0xe000_02bd_u32.cast_signed(), false)
        );
    }
}

/// A shift other than 0, 12 and 14 is an invalid argument, and the table's destroy runs
#[test]
fn another_guest_page_shift_is_refused() {
    let p = test_platform();
    let rec = Rec::new();
    let cb = callbacks(&rec);
    let range = page(&p);
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    for (n, shift) in [1u32, 13, 16].into_iter().enumerate() {
        let c = RvgIOSurfaceDeviceConfig {
            struct_size: sized::size_u32::<RvgIOSurfaceDeviceConfig>(),
            memory_map: &map,
            callbacks: &cb,
            guest_page_shift: shift,
        };
        let mut d = std::ptr::null_mut();
        // SAFETY: a live config; `d` is writable.
        assert_eq!(
            unsafe { rvg_iosurface_device_create(&c, &mut d) },
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(rec.destroys().len(), n + 1);
    }
}
