//! The tests of `caps` that compare with Metal on this host: macOS only.

use reims_vgpu_core::platform::caps::LINEAR_ALIGNMENT_FORMAT;
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_metal::MetalPlatform;

use super::*;

/// The runtime's answers are its device's, and the computed ones follow the documented facts:
/// the M3 Max under MoltenVK, Metal's argument tables, the five public primitive types;
/// `isDepth24Stencil8PixelFormatSupported` is no, as Metal answers here
#[test]
fn caps_are_the_runtimes_and_the_documented_facts() {
    let d = dev();
    let caps = d.caps();
    assert!(Arc::ptr_eq(&caps, &d.caps()), "read once, then shared");
    let props = d.raw.properties();
    assert_eq!(caps.name.as_deref(), Some(props.name.as_str()));
    assert_eq!(
        Some(caps.registry_id),
        crate::os::default_registry_id(),
        "Metal's default device's"
    );
    assert_eq!(
        caps.gpu_core_count,
        crate::os::default_gpu_core_count(),
        "Metal's default device's"
    );
    assert!(caps.gpu_core_count.is_some_and(|n| n > 0));
    assert_eq!(caps.limits.max_compute_buffers, 31);
    assert_eq!(caps.limits.max_vertex_textures, 128);
    assert_eq!(caps.limits.max_fragment_samplers, 16);
    assert_eq!(
        caps.limits.max_predicated_nesting_depth,
        u64::from(u32::MAX)
    );
    assert_eq!(caps.limits.max_rasterization_rate_layer_count, 2);
    assert_eq!(
        caps.max_vertex_amplification_count, 2,
        "the stand-in, under 8"
    );
    assert!(caps.limits.max_texture_width_2d >= 16384);
    assert_eq!(
        caps.limits.max_threadgroup_memory_length,
        props.max_threadgroup_memory_length
    );
    assert!(caps.flags.supports_tile_shaders && caps.flags.supports_image_blocks);
    assert!(
        !caps.flags.is_depth24_stencil8_pixel_format_supported,
        "an Apple GPU's no, on every device (the runtime's D663)"
    );
    if let Some(metal) = MetalPlatform::system_default_device() {
        assert!(
            !metal
                .caps()
                .flags
                .is_depth24_stencil8_pixel_format_supported
        );
    }
    assert!(
        caps.flags.supports_command_buffer_jump,
        "the adapter's predication"
    );
    let f = &caps.flags;
    assert!(
        f.supports_buffer_with_address_ranges
            && f.supports_heap_with_address_ranges
            && f.supports_resource_detach_backing,
        "MoltenVK imports host memory"
    );
    assert_eq!(caps.primitive_types[..5], [true; 5].map(Ok));
    assert_eq!(
        caps.heap_buffer_page_private,
        Ok(SizeAndAlign {
            size: 16384,
            align: 256
        })
    );
    assert_eq!(caps.min_linear_alignment_bgra8, Ok(16));
    assert_eq!(
        LINEAR_ALIGNMENT_FORMAT,
        u64::from(PixelFormat::BGRA8Unorm.as_raw())
    );
}

/// Key 33 of the device info a guest that knows every key reads of `caps`, under `snapshot`.
fn key_33(
    caps: &DeviceCaps,
    snapshot: &reims_vgpu_core::guest::abi::device_info::DeviceInfoSnapshot,
) -> u32 {
    use reims_vgpu_core::guest::abi::device_info::{
        HOST_INFO_MAX_KEY, HOST_INFO_PAIRS, write_device_info,
    };
    let mut value = None;
    write_device_info(
        OnFault::Die,
        caps,
        snapshot,
        HOST_INFO_MAX_KEY,
        HOST_INFO_PAIRS,
        |_, pair| {
            if pair.key == 33 {
                value = Some(pair.value);
            }
        },
    )
    .unwrap();
    value.unwrap()
}

/// The device info's key 33 is Metal's on this host under every snapshot bit, its SIMD
/// reduction and shuffle-and-fill bits (0x40 and 0x100) included, which answer the runtime's
/// family rows ([`simd_rows_exact`]): 0x3c4 under no snapshot bit and 0xbff under all of them
#[test]
fn device_info_key_33_is_metals_but_the_simd_bits() {
    use reims_vgpu_core::guest::abi::device_info::DeviceInfoSnapshot;
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let every = DeviceInfoSnapshot {
        supports_argument_buffers: true,
        supports_shared_textures: true,
        supports_large_user_tasks: true,
        supports_large_kernel_tasks: true,
        supports_range_buffer: true,
        supports_shared_memory_heap: true,
        ..DeviceInfoSnapshot::default()
    };
    let d = dev();
    let ours = d.caps();
    for (snapshot, want) in [(DeviceInfoSnapshot::default(), 0x3c4), (every, 0xbff)] {
        let key = key_33(&ours, &snapshot);
        let metals = key_33(&metal.caps(), &snapshot);
        assert_eq!((key, metals), (want, want), "{key:#x} {metals:#x}");
    }
}

/// Every stand-in is the Metal platform's answer on this host, but `supportsMemoryOrderAtomics`
/// at macOS 27.0.1: the runtime has no memory-order atomics, so ours stays false, Metal's answer
/// at 26.7, where Metal 27.0.1 on the M3 Max answers true ([`stand_in::MEMORY_ORDER_ATOMICS`])
#[test]
fn stand_ins_are_the_metal_platforms_here() {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let m = metal.caps();
    let ours = dev().caps();
    let f = &m.flags;
    assert_eq!(f.is_large_mrt_supported, stand_in::LARGE_MRT);
    assert_eq!(f.is_rgb10a2_gamma_supported, stand_in::RGB10A2_GAMMA);
    assert_eq!(
        (
            f.supports_command_buffer_jump,
            m.limits.max_predicated_nesting_depth
        ),
        (
            ours.flags.supports_command_buffer_jump,
            ours.limits.max_predicated_nesting_depth
        ),
        "the adapter's predication answers Metal's"
    );
    assert_eq!(
        f.supports_compute_compressed_texture_write,
        stand_in::COMPUTE_COMPRESSED_TEXTURE_WRITE
    );
    assert_eq!(
        f.supports_conditional_load_store,
        stand_in::CONDITIONAL_LOAD_STORE
    );
    assert_eq!(
        f.supports_float16_bcubic_filtering,
        stand_in::FLOAT16_BCUBIC_FILTERING
    );
    assert!(!ours.flags.supports_memory_order_atomics);
    let metals = match crate::platform::metal_level("supportsMemoryOrderAtomics") {
        metal2vulkan_runtime::MetalLevel::MacOS26_7 => false,
        metal2vulkan_runtime::MetalLevel::MacOS27_0_1 => true,
        level => panic!("no answer measured at {level}"),
    };
    assert_eq!(f.supports_memory_order_atomics, metals);
    assert_eq!(
        f.supports_texture_2d_multisample_array,
        stand_in::TEXTURE_2D_MULTISAMPLE_ARRAY
    );
    assert_eq!(
        m.primitive_types[5..],
        stand_in::PRIVATE_PRIMITIVE_TYPES.map(Ok)
    );
    assert_eq!(
        m.limits.max_compute_threadgroup_memory,
        stand_in::MAX_COMPUTE_THREADGROUP_MEMORY
    );
    assert_eq!(
        m.max_vertex_amplification_count,
        stand_in::MAX_VERTEX_AMPLIFICATION_COUNT
    );
    assert_eq!(
        (f.supports_tile_shaders, f.supports_non_square_tile_shaders),
        (stand_in::TILE_SHADERS, stand_in::TILE_SHADERS)
    );
    assert_eq!(f.supports_image_blocks, stand_in::IMAGE_BLOCKS);
    assert_eq!(
        ours.registry_id, m.registry_id,
        "the default device's, Metal's"
    );
    assert_eq!(m.families, stand_in::FAMILIES.map(Ok));
    assert_eq!(
        f.supports_extended_xr10_formats,
        stand_in::EXTENDED_XR10_FORMATS
    );
    assert_eq!(
        m.write_rounding_modes,
        stand_in::WRITE_ROUNDING_MODES.map(Ok)
    );
    // The answers both platforms take from the device agree where the runtime reproduces
    // Metal's, and `gpuCoreCount` is Metal's own for this GPU.
    assert_eq!(ours.gpu_core_count, m.gpu_core_count);
    assert_eq!(
        ours.max_threads_per_threadgroup,
        m.max_threads_per_threadgroup
    );
    assert_eq!(
        ours.min_linear_alignment_bgra8,
        m.min_linear_alignment_bgra8
    );
    assert_eq!(ours.heap_buffer_page_private, m.heap_buffer_page_private);
    assert_eq!(
        ours.limits.max_compute_buffers,
        m.limits.max_compute_buffers
    );
    assert_eq!(ours.limits.max_texture_layers, m.limits.max_texture_layers);
    assert_eq!(
        (
            ours.limits.device_linear_texture_alignment_bytes,
            ours.limits.linear_texture_alignment_bytes,
            ours.limits.iosurface_texture_alignment_bytes,
            ours.limits.iosurface_read_only_texture_alignment_bytes,
        ),
        (
            m.limits.device_linear_texture_alignment_bytes,
            m.limits.linear_texture_alignment_bytes,
            m.limits.iosurface_texture_alignment_bytes,
            m.limits.iosurface_read_only_texture_alignment_bytes,
        ),
        "the runtime's alignments"
    );
    assert_eq!(
        ours.limits.max_threadgroup_memory_length,
        m.limits.max_threadgroup_memory_length
    );
}
