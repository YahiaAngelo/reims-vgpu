//! The tests of `ycbcr` that compare with Metal on this host: macOS only.

use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::descriptor::TextureDescriptor as _;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _, Texture as _};
use reims_vgpu_core::platform::types::{SizeAndAlign, Swizzle};
use reims_vgpu_core::platform::values::{BufferDesc, HeapDesc, OptionSet, TextureDesc};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_metal::MetalPlatform;

use super::*;
use crate::os::tests::bgra_surface;
use crate::platform::tests::platform;

/// The two formats measured first (the module's docs): `YCBCR10_420_2P` (505) and
/// `RGB10_420_2P` (543).
const FORMATS: [u64; 2] = [505, 543];

/// The raw values of the two-plane formats ([`super::super::two_plane`]): the 69 of the runtime's
/// `ycbcr_format`, 505 and 543 among them.
fn two_plane_formats() -> Vec<u64> {
    let formats: Vec<u64> = (0..2048u32)
        .filter(|&raw| {
            metal2vulkan::metal::PixelFormat::try_from(raw).is_ok_and(super::super::two_plane)
        })
        .map(u64::from)
        .collect();
    assert_eq!(formats.len(), 69);
    assert!(FORMATS.iter().all(|f| formats.contains(f)));
    formats
}

/// A 2D descriptor of `format`, 16 x 16, one level, usage `ShaderRead`, storage `Private`,
/// then `change`.
fn ycbcr(format: u64, change: fn(&mut TextureDesc)) -> TextureDesc {
    let mut desc = TextureDesc {
        texture_type: 2,
        pixel_format: format,
        width: 16,
        height: 16,
        usage: 1,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    };
    change(&mut desc);
    desc
}

/// (B, G, R, A).
const BGRA: Option<Swizzle> = Some(Swizzle {
    red: 4,
    green: 3,
    blue: 2,
    alpha: 5,
});

type Case = (&'static str, fn(&mut TextureDesc));

/// A shared buffer of 1 MiB on `d`.
fn buffer<P: Platform>(d: &P::Device) -> P::Buffer {
    let desc = BufferDesc {
        length: 1 << 20,
        options: vec![OptionSet::StorageMode(0)],
        pinned_gpu_address: None,
        contents: None,
        no_copy: false,
        deallocator: None,
    };
    d.new_buffer(desc).unwrap().unwrap()
}

/// A placement heap of 1 MiB on `d`, of storage mode `storage`.
fn heap<P: Platform>(d: &P::Device, storage: u64) -> P::Heap {
    let desc = HeapDesc {
        size: 1 << 20,
        heap_type: 1,
        options: vec![OptionSet::StorageMode(storage)],
        address_ranges: None,
        pinned_gpu_address: None,
    };
    d.new_heap(&desc).unwrap().heap.unwrap()
}

/// What every creator but the IOSurface one answers of `desc` on `d`, whose storage mode is
/// `storage`: the descriptor's format; whether `newTextureWithDescriptor:`,
/// `newSharedTextureWithDescriptor:`, a buffer's `newTextureWithDescriptor:offset:bytesPerRow:`
/// and `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:` (at offset 0 and 1,
/// with rows of 4096 bytes and of 1, and past the buffer's end), and a placement heap's
/// `newTextureWithDescriptor:offset:` (of the descriptor's storage mode at offset 0, 1 and
/// past its end, and of another mode) made a texture, in that order; and
/// `heapTextureSizeAndAlignWithDescriptor:`.
fn answers<P: Platform>(
    d: &P::Device,
    desc: &P::TextureDescriptor,
    storage: u64,
) -> (u64, [bool; 13], SizeAndAlign) {
    let b = buffer::<P>(d);
    let h = heap::<P>(d, storage);
    let other = heap::<P>(d, if storage == 0 { 2 } else { 0 });
    let past = (1 << 20) + 4096;
    let made = [
        d.new_texture(OnFault::Return, desc).unwrap().is_some(),
        d.new_shared_texture(desc).unwrap().is_some(),
        b.new_texture(OnFault::Return, desc, 0, 4096)
            .unwrap()
            .is_some(),
        b.new_texture(OnFault::Return, desc, 1, 4096)
            .unwrap()
            .is_some(),
        b.new_texture(OnFault::Return, desc, 0, 1)
            .unwrap()
            .is_some(),
        b.new_texture(OnFault::Return, desc, past, 4096)
            .unwrap()
            .is_some(),
        b.new_linear_texture(desc, 0, 4096, 65536)
            .unwrap()
            .is_some(),
        b.new_linear_texture(desc, 1, 4096, 65536)
            .unwrap()
            .is_some(),
        b.new_linear_texture(desc, 0, 1, 65536).unwrap().is_some(),
        h.new_texture(desc, 0).unwrap().is_some(),
        h.new_texture(desc, 1).unwrap().is_some(),
        h.new_texture(desc, past).unwrap().is_some(),
        other.new_texture(desc, 0).unwrap().is_some(),
    ];
    let size = d.heap_texture_size_and_align(desc).unwrap();
    (desc.pixel_format(), made, size)
}

/// A descriptor takes each two-plane format and answers it; every creator but the IOSurface
/// one answers nil of one Metal's validation passes, and the heap's size query 0 bytes at
/// Metal's alignment, on both platforms, whatever the creator's own checks would refuse
/// (`framebufferOnly`, a rotation, an offset or rows off their alignment, bytes past the end,
/// a storage mode other than the heap's): 505 and 543, and each the runtime has, 500 among
/// them
#[test]
fn textures_but_over_a_surface_are_nil_as_metals_are() {
    let nil: [Case; 26] = [
        ("2D", |_| {}),
        ("1D", |d| (d.texture_type, d.height) = (0, 1)),
        ("1D array of 2048", |d| {
            (d.texture_type, d.height, d.array_length) = (1, 1, 2048);
        }),
        ("2D array of 2", |d| {
            (d.texture_type, d.array_length) = (3, 2)
        }),
        ("cube", |d| d.texture_type = 5),
        ("cube array of 341", |d| {
            (d.texture_type, d.array_length) = (6, 341)
        }),
        ("3D, 4 levels", |d| {
            (
                d.texture_type,
                d.width,
                d.height,
                d.depth,
                d.mipmap_level_count,
            ) = (7, 8, 8, 2, 4);
        }),
        ("texture buffer, 7 levels", |d| {
            (d.texture_type, d.width, d.height, d.mipmap_level_count) = (9, 64, 1, 7);
        }),
        ("5 levels", |d| d.mipmap_level_count = 5),
        ("16384 wide", |d| d.width = 16384),
        ("1 x 1", |d| (d.width, d.height) = (1, 1)),
        ("1921 x 1081", |d| (d.width, d.height) = (1921, 1081)),
        ("shared", |d| d.resource_options = Some(0)),
        ("managed", |d| d.resource_options = Some(0x10)),
        ("memoryless render target", |d| {
            (d.usage, d.resource_options) = (4, Some(0x30));
        }),
        ("read and write", |d| d.usage = 3),
        ("render target", |d| d.usage = 4),
        ("usage 0", |d| d.usage = 0),
        ("pixel format view", |d| d.usage = 0x13),
        ("usage bits Metal takes", |d| d.usage = 0x1_c00d),
        ("read swizzle", |d| d.swizzle = BGRA),
        ("written swizzle, enabled", |d| {
            (d.usage, d.swizzle, d.write_swizzle_enabled) = (3, BGRA, Some(true));
        }),
        ("framebufferOnly", |d| d.framebuffer_only = Some(true)),
        ("rotation 2, framebufferOnly", |d| {
            (d.rotation, d.framebuffer_only) = (Some(2), Some(true));
        }),
        ("rotation 4, framebufferOnly", |d| {
            (d.rotation, d.framebuffer_only) = (Some(4), Some(true));
        }),
        ("drawable", |d| d.is_drawable = Some(true)),
    ];
    let d = dev();
    let platform = platform();
    let md = MetalPlatform::system_default_device();
    for format in two_plane_formats() {
        for (what, change) in nil {
            let desc = ycbcr(format, change);
            // A heap of storage mode 1 (managed) is refused on both platforms.
            let storage = match desc.resource_options {
                Some(0) => 0,
                _ => 2,
            };
            let ours = platform.texture_descriptor(&desc);
            assert_eq!(ours.pixel_format(), format, "{what}");
            let answered = answers::<crate::platform::VulkanPlatform>(&d, &ours, storage);
            let align = if desc.texture_type == 3 { 128 } else { 16 };
            let expected = (format, [false; 13], SizeAndAlign { size: 0, align });
            assert_eq!(answered, expected, "{format} {what}");
            if let Some(md) = &md {
                let metal = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc);
                assert_eq!(
                    answers::<MetalPlatform>(md, &metal, storage),
                    answered,
                    "Metal: {format} {what}"
                );
            }
        }
    }
}

/// Every creator but the IOSurface one and `newSharedTextureWithDescriptor:` dies where
/// Metal's validation asserts (measured with 505 and 543: the module's docs; Metal's side
/// cannot run, as its assert ends the process), and that one answers nil, as Metal's does:
/// of each two-plane format, where the runtime answers no texture first for one it has
#[test]
fn textures_but_over_a_surface_die_where_metal_asserts() {
    let die: [Case; 22] = [
        ("multisample", |d| (d.texture_type, d.sample_count) = (4, 4)),
        ("multisample of 2", |d| {
            (d.texture_type, d.sample_count) = (4, 2)
        }),
        ("multisample of 1 sample", |d| d.texture_type = 4),
        ("multisample array", |d| {
            (d.texture_type, d.sample_count, d.array_length) = (8, 4, 2);
        }),
        ("atomic", |d| d.usage = 0x21),
        ("2D of 4 samples", |d| d.sample_count = 4),
        ("written swizzle", |d| (d.usage, d.swizzle) = (3, BGRA)),
        ("rendered swizzle", |d| (d.usage, d.swizzle) = (4, BGRA)),
        ("16385 wide", |d| d.width = 16385),
        ("3D 2049 wide", |d| {
            (d.texture_type, d.width, d.depth) = (7, 2049, 2)
        }),
        ("texture buffer past its largest", |d| {
            (d.texture_type, d.width, d.height) = (9, 268_435_457, 1);
        }),
        ("1D of height 2", |d| (d.texture_type, d.height) = (0, 2)),
        ("2D of depth 2", |d| d.depth = 2),
        ("cube 16 x 8", |d| (d.texture_type, d.height) = (5, 8)),
        ("2D of length 2", |d| d.array_length = 2),
        ("cube array of 342", |d| {
            (d.texture_type, d.array_length) = (6, 342)
        }),
        ("6 levels", |d| d.mipmap_level_count = 6),
        ("1D of 2 levels", |d| {
            (d.texture_type, d.height, d.mipmap_level_count) = (0, 1, 2);
        }),
        ("0 wide", |d| d.width = 0),
        ("rotation 2", |d| d.rotation = Some(2)),
        ("rotation 5, framebufferOnly", |d| {
            (d.rotation, d.framebuffer_only) = (Some(5), Some(true));
        }),
        ("usage bit 0x40", |d| d.usage = 0x45),
    ];
    type V = crate::platform::VulkanPlatform;
    let d = dev();
    let platform = platform();
    let md = MetalPlatform::system_default_device();
    let b = buffer::<V>(&d);
    let h = heap::<V>(&d, 2);
    for format in two_plane_formats() {
        for (what, change) in die {
            let desc = ycbcr(format, change);
            let ours = platform.texture_descriptor(&desc);
            let deaths = [
                (
                    "newTextureWithDescriptor:",
                    catch_death(|| d.new_texture(OnFault::Return, &ours)).map(drop),
                ),
                (
                    "heapTextureSizeAndAlignWithDescriptor:",
                    catch_death(|| d.heap_texture_size_and_align(&ours)).map(drop),
                ),
                (
                    "newTextureWithDescriptor:offset:bytesPerRow:",
                    catch_death(|| b.new_texture(OnFault::Return, &ours, 0, 4096)).map(drop),
                ),
                (
                    "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
                    catch_death(|| b.new_linear_texture(&ours, 0, 4096, 65536)).map(drop),
                ),
                (
                    "newTextureWithDescriptor:offset:",
                    catch_death(|| h.new_texture(&ours, 0)).map(drop),
                ),
            ];
            for (sel, death) in deaths {
                let site = death.unwrap_err();
                assert_eq!(
                    (site.kind, site.file),
                    (DeathKind::Abort, sel),
                    "{format} {what}"
                );
            }
            assert!(d.new_shared_texture(&ours).unwrap().is_none(), "{what}");
            if let Some(md) = &md {
                let metal = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc);
                let shared = md.new_shared_texture(&metal).unwrap();
                assert!(shared.is_none(), "Metal: {format} {what}");
            }
        }
    }
}

/// A texture of each format over a surface of one plane, which Metal makes, dies at its
/// selector; the descriptor takes another format after it, and the one-plane siblings, which
/// the runtime refuses, and answers each, a texture of which dies where it is made
#[test]
fn textures_die_where_made() {
    for format in FORMATS {
        die_where_made(format);
    }
}

/// [`textures_die_where_made`] of `format`.
fn die_where_made(format: u64) {
    let d = dev();
    let s = bgra_surface(16, 16);
    let desc = ycbcr(format, |_| {});
    if let Some(md) = MetalPlatform::system_default_device() {
        let metal = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc);
        let made = md.new_texture_on_surface(&metal, Some(&s), 0).unwrap();
        assert_eq!(made.map(|t| t.pixel_format()), Some(format), "Metal");
    }
    let ours = platform().texture_descriptor(&desc);
    let site = catch_death(|| d.new_texture_on_surface(&ours, Some(&s), 0)).unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (
            DeathKind::Abort,
            "newTextureWithDescriptor:iosurface:plane:"
        )
    );
    ours.set_pixel_format(80);
    assert_eq!(ours.pixel_format(), 80);
    assert!(d.new_texture(OnFault::Return, &ours).unwrap().is_some());
    ours.set_pixel_format(format);
    assert_eq!(ours.pixel_format(), format);
    for sibling in [501, 504, 521, 524] {
        ours.set_pixel_format(sibling);
        assert_eq!(ours.pixel_format(), sibling);
        let site = catch_death(|| d.new_texture(OnFault::Return, &ours)).unwrap_err();
        assert_eq!(site.file, "newTextureWithDescriptor:", "{sibling}");
    }
    ours.set_pixel_format(format);
    assert_eq!(ours.pixel_format(), format);
}
