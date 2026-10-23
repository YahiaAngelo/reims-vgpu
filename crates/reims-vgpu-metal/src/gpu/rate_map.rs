//! Rasterization rate maps: core's `RateMap`, `RateMapDescriptor` and `RateMapLayerDescriptor` in
//! typed calls.
//!
//! Every method is one call through the message layer: a getter's or a descriptor property's raise ends
//! the process at the call; a raise inside a method with arguments is the caller's to catch.
//!
//! A render pass over a map that is not immutable, which the guest ends having drawn no triangles,
//! ends with a draw the guest did not encode ([`mark_pass`], [`drew`], [`end_pass`]), which keeps
//! the host's GPU from hanging. Measured on the M3 Max, macOS 27.0.1 (26A434), one case a process:
//! - A pass over a mutable map that clears an attachment and draws nothing hangs the GPU: its
//!   command buffer ends about 29 ms in with `MTLCommandBufferErrorDomain` 2, "Caused GPU Hang
//!   Error (00000003:kIOGPUCommandBufferCallbackErrorHang)". The map may be made with no layers or
//!   one, reset to either, or never reset. It hangs with a `Shared` BGRA8Unorm colour target, alone
//!   or with a `Private` Depth32Float_Stencil8 one, with a `Private` BGRA8Unorm target of 4 samples,
//!   and with a `Shared` Depth32Float one; it does not with a 2D array target of two slices, with no
//!   attachments, or with an RGBA16Float target of 4 samples resolved into a `Shared` one.
//! - A pipeline set without a draw, a memory barrier, or a blit after the pass in its command buffer
//!   does not keep it from hanging; any draw does, of no vertices or no instances included. A pass
//!   over an immutable map, a pass that loads, and a pass over no map do not hang.
//! - Once a pass that draws over a mutable map has run in the process, later drawless passes over
//!   new mutable maps do not hang, 20 s idle later included. Nothing here relies on that.
//! - At macOS 26.7 (25G229), a drawless pass that cleared over a mutable map reset to no layers
//!   completed, storing its clear.
//!
//! The draw is of no vertices, of triangles (the only primitive Metal's API validation lets a pass
//! over a map draw), through one pipeline a device whose vertex function returns nothing, whose
//! rasterization is off, which names no attachment formats, and whose largest vertex amplification
//! count is the device's (Metal's API validation refuses a draw whose pipeline's is below the
//! encoder's count). It draws nothing. Measured on the same host, every attachment's stored bytes
//! and the visibility buffer's, counting and boolean, with a draw before it or none, equal those of
//! the same pass without it, or over an immutable map where that one hangs. The passes measured have
//! one or eight colour attachments of mixed formats, 2 and 4 samples, depth, stencil or both, a
//! two-slice target, vertex amplification, two viewports, a memoryless attachment, imageblocks, a
//! tile dispatch, or the guest's fill mode, depth bias, scissor, cull mode and depth-stencil state.
//! Under `MTL_DEBUG_LAYER`, the draw raises no validation error in any of them.
//!
//! The draw is the pass's last command. The pipeline it binds is then bound for nothing the guest
//! encodes: a guest draw with no pipeline set dies at the draw, as it does without it, and
//! `setRenderPipelineState:` of nil dies (both measured). The visibility mode stays the guest's:
//! `setVisibilityResultMode:offset:` of `Disabled` at offset 0 writes 8 zero bytes at offset 0
//! (measured). A map does not say whether it is mutable through Metal's API; its `mutability` (SPI)
//! answers 1 for a mutable map and 2 for an immutable one, and every map whose answer is not 2 is
//! marked.
//!
//! **Only a pass that drew nothing gets the draw**, since making its pipeline moves identifiers the
//! guest reads. Measured on the same host, one process a case, 3 runs each, every run alike:
//! - Metal keeps a record for each pipeline that supports indirect command buffers in one region of
//!   GPU memory, and such a pipeline's `uniqueIdentifier`, which opcode 460 hands the guest, is its
//!   record's address there (`0x1000000440` for the first in a process). A render pipeline whose
//!   vertex function's variant is new takes 0x80 bytes of that region too, and a mesh pipeline
//!   0x2c0, so each moves the records made after it: made between two pairs of ICB-capable render
//!   and compute pipelines, the no-draw pipeline moves the second pair's identifiers by 0x80,
//!   whether its function is in a library of its own or a shared one, and whatever its
//!   amplification count. How far depends on what the region holds and has freed: 0x200 in
//!   `only_a_drawless_pass_moves_later_pipelines_identifiers` (6 runs alike).
//! - What takes nothing there: a library or a function, a compute pipeline that does not support
//!   ICBs, and a render pipeline whose vertex variant is already made (a new fragment function or
//!   attachment format over the same vertex function). No pipeline moves `gpuResourceID` or
//!   `resourceIndex` (a counter of ICB-capable pipelines alone), nor the identifiers and addresses
//!   of later ICBs, textures and buffers. A pipeline made and released before the next ones moves
//!   nothing: its block is freed with it.
//! - No pipeline the draw could bind is known to take nothing. With rasterization off, Metal
//!   requires a vertex function that returns nothing (`AGXMetalG15X_M1` error 3,
//!   "RasterizationEnabled is false but the vertex shader's return type is not void"), whose
//!   variant only the draw's own pipeline makes; one that rasterizes names attachment formats
//!   every pass would have to match.
//!
//! So the platform's draw comes only where the pass would hang without it: [`mark_pass`] marks
//! the encoder, a `drawPrimitives` the guest encodes on it of the shape measured to keep a pass
//! from hanging, which Metal takes, removes the mark ([`drew`]), and [`end_pass`] makes the
//! no-draw pipeline, once per device, for a pass still marked. That shape is at least one whole
//! triangle of a triangle list or strip (3 vertices or more), once or more: the measured draw is 3
//! vertices of a triangle list, one instance. From the first such pass on, the ICB-capable
//! pipelines made later have identifiers other than Metal's own; a run whose passes over mutable
//! maps all draw sees Metal's identifiers. A pass of the guest's that draws over a mutable map
//! completes with no draw of the platform's (measured, the same test). Other draws keep the mark
//! (indexed, indirect and mesh draws, patches, ICB executions, points and lines, and draws of no
//! vertices, fewer than three, or no instances): nothing measured shows that a guest's draw of
//! those keeps the pass from hanging. The platform's own draw of no vertices was measured to, but
//! only through its no-draw pipeline, with rasterization off.

use std::cell::Cell;
use std::ptr;

use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_metal::*;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::types::*;
use reims_vgpu_core::platform::values::RateMapDesc;
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::sync::Mutex;

use super::device::create_or_die;
use super::encode_render::MetalRenderEncoder;
use super::resource::MetalRenderPipeline;
use crate::msg::encode::{FromMtl, ToMtl};
use crate::msg::{self, Policy};
use crate::platform::{MetalPlatform, Obj};
use crate::spi::methods::{DeviceSpi, RateLayerDescriptorSpi, RateMapDescriptorSpi, RateMapSpi};

/// `id<MTLRasterizationRateMap>`.
pub type MetalRateMap = Obj<ProtocolObject<dyn MTLRasterizationRateMap>>;

impl reims_vgpu_core::platform::rate_map::RateMap<MetalPlatform> for MetalRateMap {
    /// Metal checks the buffer's room itself.
    fn copy_parameter_data(
        &self,
        buffer: &<MetalPlatform as Platform>::Buffer,
        offset: u64,
    ) -> RaiseResult<()> {
        let (m, buffer, offset) = (self.get(), buffer.get(), offset.to_usize());
        // SAFETY: raw guest values go to Metal unvalidated; Metal checks them itself.
        msg::send(
            "copyParameterDataToBuffer:offset:",
            Policy::Caught,
            || unsafe { m.copyParameterDataToBuffer_offset(buffer, offset) },
        )
    }

    fn layer_count(&self) -> u64 {
        let m = self.get();
        msg::get("layerCount", || m.layerCount()).to_u64()
    }

    fn physical_to_screen(
        &self,
        physical_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        let (m, point, layer) = (
            self.get(),
            physical_coordinates.mtl(),
            layer_index.to_usize(),
        );
        // SAFETY: Metal checks the layer index itself.
        msg::send(
            "mapPhysicalToScreenCoordinates:forLayer:",
            Policy::Caught,
            || unsafe { m.mapPhysicalToScreenCoordinates_forLayer(point, layer) },
        )
        .map(MtlCoordinate2D::from_mtl)
    }

    fn screen_to_physical(
        &self,
        screen_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        let (m, point, layer) = (self.get(), screen_coordinates.mtl(), layer_index.to_usize());
        // SAFETY: as above.
        msg::send(
            "mapScreenToPhysicalCoordinates:forLayer:",
            Policy::Caught,
            || unsafe { m.mapScreenToPhysicalCoordinates_forLayer(point, layer) },
        )
        .map(MtlCoordinate2D::from_mtl)
    }

    fn mutability(&self) -> u64 {
        let m = self.get();
        msg::get("mutability", || RateMapSpi::mutability(m))
    }

    fn parameter_buffer_size_and_align(&self) -> SizeAndAlign {
        let m = self.get();
        SizeAndAlign::from_mtl(msg::get("parameterBufferSizeAndAlign", || {
            m.parameterBufferSizeAndAlign()
        }))
    }

    fn physical_granularity(&self) -> MtlSize {
        let m = self.get();
        MtlSize::from_mtl(msg::get("physicalGranularity", || m.physicalGranularity()))
    }

    fn physical_size(&self, layer_index: u64) -> RaiseResult<MtlSize> {
        let (m, layer) = (self.get(), layer_index.to_usize());
        // SAFETY: as above.
        msg::send("physicalSizeForLayer:", Policy::Caught, || unsafe {
            m.physicalSizeForLayer(layer)
        })
        .map(MtlSize::from_mtl)
    }

    fn reset(&self, descriptor: &<MetalPlatform as Platform>::RateMapDescriptor) {
        let (m, descriptor) = (self.get(), descriptor.get());
        msg::get("resetUsingDescriptor:", || {
            m.resetUsingDescriptor(descriptor)
        });
    }
}

/// `MTLRasterizationRateMapDescriptor *`.
pub type MetalRateMapDescriptor = Obj<MTLRasterizationRateMapDescriptor>;

impl reims_vgpu_core::platform::rate_map::RateMapDescriptor<MetalPlatform>
    for MetalRateMapDescriptor
{
    fn set_layer(
        &self,
        on: OnFault,
        layer: &<MetalPlatform as Platform>::RateMapLayerDescriptor,
        layer_index: u64,
    ) -> RaiseResult<()> {
        let (d, layer, index) = (self.get(), layer.get(), layer_index.to_usize());
        // SAFETY: Metal checks the index itself.
        msg::send("setLayer:atIndex:", Policy::On(on), || unsafe {
            d.setLayer_atIndex(Some(layer), index)
        })
    }

    fn layer_count(&self) -> u64 {
        let d = self.get();
        msg::get("layerCount", || d.layerCount()).to_u64()
    }
    /// `layers` (`+0`, retained), then its `objectAtIndexedSubscript:`.
    fn layer(
        &self,
        on: OnFault,
        layer_index: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::RateMapLayerDescriptor>> {
        let d = self.get();
        let layers: MetalRateMapLayerArray = Obj::from_retained(msg::get("layers", || d.layers()));
        layer_at(on, &layers, layer_index)
    }
}

/// A rasterization-rate-map descriptor holding `desc`: `+new`, then its screen size, its
/// mutability when set, and its minimum factor when set, in that order. Its layers are set on it
/// after.
pub fn rate_map_descriptor(desc: &RateMapDesc) -> MetalRateMapDescriptor {
    let d = crate::platform::class_new(MTLRasterizationRateMapDescriptor::new);
    let o = d.get();
    let size = desc.screen_size.mtl();
    // SAFETY: Metal checks the size when it makes the map.
    msg::get("setScreenSize:", || unsafe { o.setScreenSize(size) });
    if let Some(mutability) = desc.mutability {
        msg::get("setMutability:", || o.setMutability(mutability));
    }
    if let Some(factor) = desc.min_factor {
        msg::get("setMinFactor:", || o.setMinFactor(factor));
    }
    d
}

/// A layer array's `objectAtIndexedSubscript:`: the layer at `+0`, retained, nil for an index
/// past the layers.
fn layer_at(
    on: OnFault,
    layers: &MetalRateMapLayerArray,
    layer_index: u64,
) -> RaiseResult<Option<MetalRateMapLayerDescriptor>> {
    let (a, index) = (layers.get(), layer_index.to_usize());
    // SAFETY: Metal answers nil past the layers.
    msg::send("objectAtIndexedSubscript:", Policy::On(on), || unsafe {
        a.objectAtIndexedSubscript(index)
    })
    .map(|l| l.map(Obj::from_retained))
}

/// `MTLRasterizationRateLayerDescriptor *`.
pub type MetalRateMapLayerDescriptor = Obj<MTLRasterizationRateLayerDescriptor>;

impl reims_vgpu_core::platform::rate_map::RateMapLayerDescriptor for MetalRateMapLayerDescriptor {
    fn horizontal_sample_storage(&self) -> HostPtr {
        let l = self.get();
        HostPtr::from_ptr(msg::get("horizontalSampleStorage", || {
            l.horizontalSampleStorage_or_null()
        }))
    }

    fn sample_count(&self) -> MtlSize {
        let l = self.get();
        MtlSize::from_mtl(msg::get("sampleCount", || {
            RateLayerDescriptorSpi::sampleCount(l)
        }))
    }

    fn vertical_sample_storage(&self) -> HostPtr {
        let l = self.get();
        HostPtr::from_ptr(msg::get("verticalSampleStorage", || {
            l.verticalSampleStorage_or_null()
        }))
    }
}

/// `MTLRasterizationRateLayerArray *`.
pub type MetalRateMapLayerArray = Obj<MTLRasterizationRateLayerArray>;

/// The source of the no-draw pipeline's vertex function, which returns nothing.
const NO_DRAW_SOURCE: &str = "vertex void reims_vgpu_no_draw() {}";

/// The no-draw pipelines made so far, each with its device's registry ID. Each stays for the life
/// of the process.
static NO_DRAW: Mutex<Vec<(u64, MetalRenderPipeline)>> = Mutex::new(Vec::new());

/// The key of the association [`mark_pass`] makes, and its value: its address. The association
/// assigns, so nothing is retained.
static MARK: u8 = 0;

thread_local! {
    /// The encoder [`mark_pass`] last marked on this thread, by address, until a draw of the
    /// guest's or [`end_pass`] takes the mark (0 for none): a draw reads this, not the association.
    /// A pass carried to another thread, or marked before another on this one, keeps its mark
    /// through its draws and gets the platform's draw whether or not the guest drew.
    static MARKED_HERE: Cell<usize> = const { Cell::new(0) };
}

/// Sets `encoder`'s mark (`marked`) or takes it.
fn set_mark(encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>, marked: bool) {
    let value = if marked {
        (&raw const MARK).cast_mut().cast()
    } else {
        ptr::null_mut()
    };
    // SAFETY: a live encoder and a static key; the association assigns, so the value, the key's
    // own address or nil, is never retained or released.
    unsafe {
        objc2::ffi::objc_setAssociatedObject(
            ptr::from_ref(encoder).cast_mut().cast(),
            (&raw const MARK).cast(),
            value,
            objc2::ffi::OBJC_ASSOCIATION_ASSIGN,
        );
    }
}

/// When the pass `descriptor` describes is over a map that is not immutable, marks `encoder`, its
/// encoder, for [`end_pass`].
pub(crate) fn mark_pass(encoder: &MetalRenderEncoder, descriptor: &MTLRenderPassDescriptor) {
    let Some(map) = msg::get("rasterizationRateMap", || descriptor.rasterizationRateMap()) else {
        return;
    };
    if msg::get("mutability", || RateMapSpi::mutability(&*map))
        == MTLMutability::Immutable.0.to_u64()
    {
        return;
    }
    let e = encoder.get();
    set_mark(e, true);
    MARKED_HERE.set(ptr::from_ref(e).addr());
}

/// After a guest's draw of `vertices` vertices of `primitive`, `instances` times, on `encoder`,
/// answered `drawn`: a draw of the shape measured to keep a pass from hanging, which Metal took,
/// takes the encoder's mark when [`mark_pass`] set it on this thread, as the pass no longer needs
/// the platform's draw. That shape is a draw of triangles or a triangle strip of at least one
/// whole triangle (3 vertices or more), once or more; a draw of no vertices, of fewer than three,
/// or of no instances keeps the mark, as it may give the GPU no draw at all.
pub(crate) fn drew(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    primitive: MTLPrimitiveType,
    vertices: usize,
    instances: usize,
    drawn: &RaiseResult<()>,
) {
    let at = ptr::from_ref(encoder).addr();
    if MARKED_HERE.get() != at
        || drawn.is_err()
        || !matches!(
            primitive,
            MTLPrimitiveType::Triangle | MTLPrimitiveType::TriangleStrip
        )
        || vertices < 3
        || instances == 0
    {
        return;
    }
    MARKED_HERE.set(0);
    set_mark(encoder, false);
}

/// Before `encoder`'s `endEncoding`: when the encoder is still marked, takes the mark, so a second
/// `endEncoding` draws nothing more, then binds the device's no-draw pipeline, made on its first
/// use, and draws no vertices. A raise at either send is as `on` says.
pub(crate) fn end_pass(
    on: OnFault,
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
) -> RaiseResult<()> {
    let object = ptr::from_ref(encoder).cast::<AnyObject>();
    // SAFETY: a live encoder and the static key; the value is nil or the key's address.
    let marked = unsafe {
        !objc2::ffi::objc_getAssociatedObject(object, (&raw const MARK).cast()).is_null()
    };
    if !marked {
        return Ok(());
    }
    set_mark(encoder, false);
    if MARKED_HERE.get() == object.addr() {
        MARKED_HERE.set(0);
    }
    let pipeline = no_draw_pipeline(&msg::get("device", || encoder.device()));
    let pipeline = pipeline.get();
    msg::send("setRenderPipelineState:", Policy::On(on), || {
        encoder.setRenderPipelineState(pipeline)
    })?;
    // SAFETY: a draw of no vertices.
    msg::send(
        "drawPrimitives:vertexStart:vertexCount:",
        Policy::On(on),
        || unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 0)
        },
    )
}

/// The no-draw pipeline of `device` (a new reference), made on the first call for its registry ID.
/// A library, function or pipeline Metal does not make ends the process at its call.
fn no_draw_pipeline(device: &ProtocolObject<dyn MTLDevice>) -> MetalRenderPipeline {
    use reims_vgpu_core::platform::Handle as _;

    let id = msg::get("registryID", || device.registryID());
    // Held through the make, so two first asks make one.
    let mut made = NO_DRAW.lock();
    if let Some((_, pipeline)) = made.iter().find(|(r, _)| *r == id) {
        return pipeline.clone_ref();
    }
    let source = msg::new_string(NO_DRAW_SOURCE);
    // SAFETY: `create_or_die`'s out-parameter.
    let library = made_or_die(
        "newLibraryWithSource:options:error:",
        create_or_die("newLibraryWithSource:options:error:", |error| unsafe {
            DeviceSpi::newLibraryWithSource_options_error(device, &source, None, error)
        }),
    );
    let name = msg::new_string("reims_vgpu_no_draw");
    let function = msg::get("newFunctionWithName:", || {
        library.get().newFunctionWithName(&name)
    });
    let descriptor = crate::platform::class_new(MTLRenderPipelineDescriptor::new);
    let d = descriptor.get();
    msg::get("setVertexFunction:", || {
        d.setVertexFunction(function.as_deref())
    });
    msg::get("setRasterizationEnabled:", || {
        d.setRasterizationEnabled(false)
    });
    let amplification = (1..=16)
        .rev()
        .find(|&n| {
            msg::get("supportsVertexAmplificationCount:", || {
                device.supportsVertexAmplificationCount(n)
            })
        })
        .unwrap_or(1);
    // SAFETY: a count the device supports.
    msg::get("setMaxVertexAmplificationCount:", || unsafe {
        d.setMaxVertexAmplificationCount(amplification)
    });
    // SAFETY: `create_or_die`'s out-parameter.
    let pipeline = made_or_die(
        "newRenderPipelineStateWithDescriptor:error:",
        create_or_die(
            "newRenderPipelineStateWithDescriptor:error:",
            |error| unsafe {
                DeviceSpi::newRenderPipelineStateWithDescriptor_error(device, d, error)
            },
        ),
    );
    made.push((id, pipeline.clone_ref()));
    pipeline
}

/// What a creator made, or the process ends at `call`, naming the creator's error.
fn made_or_die<T: objc2::Message>(
    call: &'static str,
    made: Result<Option<Obj<T>>, NsErrorText>,
) -> Obj<T> {
    match made {
        Ok(Some(o)) => o,
        Ok(None) => fatal_with_detail(
            DeathId::Platform {
                kind: DeathKind::Abort,
                call,
            },
            "nil",
        ),
        Err(e) => fatal_with_detail(
            DeathId::Platform {
                kind: DeathKind::Abort,
                call,
            },
            &e.description,
        ),
    }
}

#[cfg(test)]
mod tests {

    use objc2::rc::Retained;
    use objc2_metal::MTLRasterizationRateMapDescriptor;
    use reims_vgpu_core::platform::Handle as _;
    use reims_vgpu_core::platform::rate_map::*;

    use super::*;
    use crate::gpu::resource::testkit::*;

    // BEGIN GENERATED TESTS

    /// Every RateMap method
    #[test]
    fn rate_map_sends() {
        let k = kit();
        let x = k.rate_map;
        let r = x.copy_parameter_data(&k.buffer, 0);
        assert!(r.is_ok(), "copy_parameter_data: {r:?}");
        assert_eq!(x.layer_count(), raw_u64(obj(&x), "layerCount"));
        let r = x.physical_to_screen(MtlCoordinate2D { x: 1.0, y: 1.0 }, 0);
        assert!(r.is_ok(), "physical_to_screen: {r:?}");
        let r = x.screen_to_physical(MtlCoordinate2D { x: 1.0, y: 1.0 }, 0);
        assert!(r.is_ok(), "screen_to_physical: {r:?}");
        assert_eq!(x.mutability(), raw_u64(obj(&x), "mutability"));
        let _ = x.parameter_buffer_size_and_align();
        let _ = x.physical_granularity();
        let r = x.physical_size(0);
        assert!(r.is_ok(), "physical_size: {r:?}");
        x.reset(&k.rate_map_descriptor);
    }

    /// Every RateMapDescriptor method, and its layer's
    #[test]
    fn rate_map_descriptor_sends() {
        let k = kit();
        let x = rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(1, 1, 1),
            ..RateMapDesc::default()
        });
        assert_eq!(x.layer_count(), raw_u64(obj(&x), "layerCount"));
        let r = x.set_layer(OnFault::Return, &k.rate_layer, 0);
        assert!(r.is_ok(), "set_layer: {r:?}");
        let l = x.layer(OnFault::Return, 0).unwrap().unwrap();
        assert_eq!(l.sample_count(), MtlSize::new(2, 2, 0));
        assert!(!l.horizontal_sample_storage().is_null());
        assert!(!l.vertical_sample_storage().is_null());
    }
    // END GENERATED TESTS

    /// The rate map's values: one layer of the 64x64 screen, a parameter buffer Metal sizes, a
    /// physical size no larger than the screen, the identity at the origin both ways, and nil for a
    /// layer index past the descriptor's layers
    #[test]
    fn rate_map_values() {
        let k = kit();
        let m = &k.rate_map;
        assert_eq!(m.layer_count(), 1);
        let sa = m.parameter_buffer_size_and_align();
        assert!(sa.size > 0 && sa.align > 0, "{sa:?}");
        let g = m.physical_granularity();
        assert!(g.width > 0 && g.height > 0, "{g:?}");
        let p = m.physical_size(0).unwrap();
        assert!(p.width <= 64 && p.height <= 64 && p.width > 0, "{p:?}");
        let origin = MtlCoordinate2D { x: 0.0, y: 0.0 };
        assert_eq!(m.screen_to_physical(origin, 0).unwrap(), origin);
        assert_eq!(m.physical_to_screen(origin, 0).unwrap(), origin);
        // A map made from a descriptor that set none is immutable (2, measured).
        assert_eq!(m.mutability(), 2);
    }

    /// A rate map descriptor built from a value holds its screen size, mutability and minimum
    /// factor; the typed reads count and find the layers set on it
    #[test]
    fn rate_map_descriptor_from_a_value() {
        let k = kit();
        let d = rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(32, 16, 0),
            mutability: Some(1),
            min_factor: Some(0.5),
        });
        assert_eq!(
            MtlSize::from_mtl(raw::<MTLSize>(obj(&d), "screenSize")),
            MtlSize::new(32, 16, 0)
        );
        assert_eq!(raw_u64(obj(&d), "mutability"), 1);
        assert_eq!(raw::<f32>(obj(&d), "minFactor"), 0.5);
        assert_eq!(d.layer_count(), 0);
        assert!(d.layer(OnFault::Return, 0).unwrap().is_none());
        d.set_layer(OnFault::Return, &k.rate_layer, 0).unwrap();
        assert_eq!(d.layer_count(), 1);
        let l = d.layer(OnFault::Return, 0).unwrap().unwrap();
        assert_eq!(l.id(), k.rate_layer.id());
        assert!(d.layer(OnFault::Return, 3).unwrap().is_none());
        // What is not set keeps Metal's own value.
        let plain = rate_map_descriptor(&RateMapDesc::default());
        let fresh: MetalRateMapDescriptor =
            Obj::from_retained(MTLRasterizationRateMapDescriptor::new());
        assert_eq!(
            raw_u64(obj(&plain), "mutability"),
            raw_u64(obj(&fresh), "mutability")
        );
        assert_eq!(
            raw::<f32>(obj(&plain), "minFactor"),
            raw::<f32>(obj(&fresh), "minFactor")
        );
    }

    /// The side of the screens and targets of the drawless-pass tests.
    const SIDE: usize = 16;

    /// A map of a `SIDE` by `SIDE` screen with `layers` layers of one sample of 1 on each axis,
    /// mutable or immutable.
    fn side_map(k: &Kit, layers: usize, mutable: bool) -> MetalRateMap {
        use reims_vgpu_core::platform::device::GpuDevice as _;

        let d = side_descriptor(layers, mutable);
        k.device.new_rate_map(&d).unwrap().unwrap()
    }

    /// The descriptor [`side_map`] makes a map from.
    fn side_descriptor(layers: usize, mutable: bool) -> MetalRateMapDescriptor {
        let d = rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(SIDE.to_u64(), SIDE.to_u64(), 1),
            mutability: mutable.then_some(1),
            ..RateMapDesc::default()
        });
        for at in 0..layers {
            let layer: MetalRateMapLayerDescriptor = Obj::from_retained(rate_layer(1, 1));
            for storage in [
                layer.horizontal_sample_storage(),
                layer.vertical_sample_storage(),
            ] {
                // SAFETY: the layer's storage holds one sample on each axis.
                unsafe { *storage.as_ptr::<f32>() = 1.0 };
            }
            d.set_layer(OnFault::Return, &layer, at.to_u64()).unwrap();
        }
        d
    }

    /// What a drawless pass clears.
    #[derive(Clone, Copy)]
    enum Target {
        /// A `Shared` BGRA8Unorm colour target, to (0.25, 0.5, 0.75, 1).
        Color,
        /// A `Shared` Depth32Float target, to 0.375.
        Depth,
        /// A `Private` BGRA8Unorm colour target of 4 samples, as [`Target::Color`], read through a
        /// resolve after the pass.
        Multisample,
    }

    /// A `SIDE` by `SIDE` texture of `format` and `samples` samples.
    fn side_texture(
        k: &Kit,
        format: MTLPixelFormat,
        samples: usize,
        mode: MTLStorageMode,
    ) -> Retained<ProtocolObject<dyn MTLTexture>> {
        let td = MTLTextureDescriptor::new();
        // SAFETY: plain property sets on a fresh descriptor.
        unsafe {
            td.setWidth(SIDE);
            td.setHeight(SIDE);
            td.setSampleCount(samples);
        }
        td.setTextureType(if samples > 1 {
            MTLTextureType::Type2DMultisample
        } else {
            MTLTextureType::Type2D
        });
        td.setPixelFormat(format);
        td.setStorageMode(mode);
        td.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        k.device.get().newTextureWithDescriptor(&td).unwrap()
    }

    /// Encodes, through the platform, a pass over `map` that clears `target` and draws nothing,
    /// after setting `pipeline` when given; then copies what it stored to a buffer (through a
    /// resolve for [`Target::Multisample`]), commits, and waits. Answers the command buffer's error
    /// and the stored bytes.
    fn clear_over(
        k: &Kit,
        map: &MetalRateMap,
        target: Target,
        pipeline: Option<&MetalRenderPipeline>,
    ) -> (Option<NsErrorText>, Vec<u8>) {
        pass_over(k, map, target, pipeline, false)
    }

    /// [`clear_over`], whose pass, when `draws`, draws through the platform after setting the
    /// pipeline: a triangle of the kit's vertex function, whose three vertices are one point, so
    /// it covers no texel.
    fn pass_over(
        k: &Kit,
        map: &MetalRateMap,
        target: Target,
        pipeline: Option<&MetalRenderPipeline>,
        draws: bool,
    ) -> (Option<NsErrorText>, Vec<u8>) {
        use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};

        let (format, samples, mode) = match target {
            Target::Color => (MTLPixelFormat::BGRA8Unorm, 1, MTLStorageMode::Shared),
            Target::Depth => (MTLPixelFormat::Depth32Float, 1, MTLStorageMode::Shared),
            Target::Multisample => (MTLPixelFormat::BGRA8Unorm, 4, MTLStorageMode::Private),
        };
        let texture = side_texture(k, format, samples, mode);
        let pass = MTLRenderPassDescriptor::new();
        if let Target::Depth = target {
            let a = pass.depthAttachment();
            a.setTexture(Some(&texture));
            a.setLoadAction(MTLLoadAction::Clear);
            a.setStoreAction(MTLStoreAction::Store);
            a.setClearDepth(0.375);
        } else {
            // SAFETY: index 0 of the colour attachments.
            let a = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            a.setTexture(Some(&texture));
            a.setLoadAction(MTLLoadAction::Clear);
            a.setStoreAction(MTLStoreAction::Store);
            a.setClearColor(MTLClearColor {
                red: 0.25,
                green: 0.5,
                blue: 0.75,
                alpha: 1.0,
            });
        }
        pass.setRasterizationRateMap(Some(map.get()));
        let pass: crate::gpu::desc::MetalRenderPassDescriptor = Obj::from_retained(pass);
        let cb = k.queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
        if let Some(p) = pipeline {
            let set = RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(p),
            };
            e.encode(OnFault::Return, set).unwrap();
            if draws {
                let draw = RenderCmd::DrawPrimitives {
                    primitive_type: MTLPrimitiveType::Triangle.0.to_u64(),
                    vertex_start: 0,
                    vertex_count: 3,
                };
                e.encode(OnFault::Return, draw).unwrap();
            }
        }
        e.encode(OnFault::Return, RenderCmd::EndEncoding).unwrap();
        let c = cb.get();
        let stored = if let Target::Multisample = target {
            // A pass of no map that loads the target and resolves it.
            let resolved = side_texture(k, format, 1, MTLStorageMode::Shared);
            let resolve = MTLRenderPassDescriptor::new();
            // SAFETY: index 0 of the colour attachments.
            let a = unsafe { resolve.colorAttachments().objectAtIndexedSubscript(0) };
            a.setTexture(Some(&texture));
            a.setLoadAction(MTLLoadAction::Load);
            a.setStoreAction(MTLStoreAction::MultisampleResolve);
            a.setResolveTexture(Some(&resolved));
            c.renderCommandEncoderWithDescriptor(&resolve)
                .unwrap()
                .endEncoding();
            resolved
        } else {
            texture
        };
        let len = SIDE * SIDE * 4;
        let out = k
            .device
            .get()
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let blit = c.blitCommandEncoder().unwrap();
        // SAFETY: the whole of a one-slice texture of 4 bytes a texel, into a buffer that holds it.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &stored,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize {
                    width: SIDE,
                    height: SIDE,
                    depth: 1,
                },
                &out,
                0,
                SIDE * 4,
                len,
            );
        }
        blit.endEncoding();
        c.commit();
        c.waitUntilCompleted();
        // SAFETY: a `Shared` buffer's bytes, read after its command buffer completed.
        let bytes =
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr().cast::<u8>(), len) };
        (cb.error(), bytes.to_vec())
    }

    /// A drawless pass that clears `target` over a mutable map of `layers` layers, reset to
    /// `reset` layers when set, after setting a pipeline when `pipeline`: its command buffer
    /// completes, where Metal at macOS 27.0.1 hangs the GPU without the platform's draw, and it
    /// stores what the same pass over an immutable map of as many layers stores, which is the
    /// clear on every texel. Each case is a test of its own, for a process of its own under
    /// nextest: the hang shows only in a process where no pass over a mutable map has drawn.
    fn completes(layers: usize, reset: Option<usize>, target: Target, pipeline: bool) {
        let k = kit();
        let map = side_map(&k, layers, true);
        if let Some(to) = reset {
            map.reset(&side_descriptor(to, false));
        }
        let pipeline = pipeline.then_some(&k.render_pipeline);
        let (error, stored) = clear_over(&k, &map, target, pipeline);
        assert_eq!(error, None, "the pass over the mutable map");
        let oracle = side_map(&k, reset.unwrap_or(layers), false);
        let (error, theirs) = clear_over(&k, &oracle, target, pipeline);
        assert_eq!(error, None, "the pass over the immutable map");
        assert!(stored == theirs, "the stores differ");
        let texel: [u8; 4] = match target {
            Target::Depth => 0.375f32.to_le_bytes(),
            _ => [0xbf, 0x80, 0x40, 0xff],
        };
        assert!(
            stored.chunks(4).all(|t| t == texel),
            "{:02x?}",
            &stored[..8]
        );
    }

    /// A clear over a mutable map of no layers completes
    #[test]
    fn a_clear_over_a_mutable_map_of_no_layers_completes() {
        completes(0, None, Target::Color, false);
    }

    /// A clear over a mutable map of one layer completes
    #[test]
    fn a_clear_over_a_mutable_map_of_one_layer_completes() {
        completes(1, None, Target::Color, false);
    }

    /// A clear over a mutable map of one layer reset to none completes
    #[test]
    fn a_clear_over_a_mutable_map_reset_to_no_layers_completes() {
        completes(1, Some(0), Target::Color, false);
    }

    /// A clear over a mutable map of one layer reset to one completes
    #[test]
    fn a_clear_over_a_mutable_map_reset_to_one_layer_completes() {
        completes(1, Some(1), Target::Color, false);
    }

    /// A clear over a mutable map with a pipeline set and no draw completes
    #[test]
    fn a_clear_over_a_mutable_map_with_a_pipeline_and_no_draw_completes() {
        completes(0, None, Target::Color, true);
    }

    /// A depth clear over a mutable map completes
    #[test]
    fn a_depth_clear_over_a_mutable_map_completes() {
        completes(0, None, Target::Depth, false);
    }

    /// A clear of a target of 4 samples over a mutable map completes
    #[test]
    fn a_multisampled_clear_over_a_mutable_map_completes() {
        completes(0, None, Target::Multisample, false);
    }

    /// A draw of at least one whole triangle of a list or a strip, once or more, that Metal took,
    /// on the encoder marked on this thread, takes its mark; one of points or lines, of fewer
    /// than three vertices, of no instances, or one Metal refused, keeps it, as does a draw on
    /// another encoder than the one marked last here
    #[test]
    fn a_triangle_draw_the_guest_encodes_takes_the_mark() {
        use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};

        let k = kit();
        let map = side_map(&k, 1, true);
        let texture = side_texture(&k, MTLPixelFormat::BGRA8Unorm, 1, MTLStorageMode::Shared);
        let pass = MTLRenderPassDescriptor::new();
        // SAFETY: index 0 of the colour attachments.
        let a = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        a.setTexture(Some(&texture));
        a.setLoadAction(MTLLoadAction::Load);
        a.setStoreAction(MTLStoreAction::Store);
        pass.setRasterizationRateMap(Some(map.get()));
        let pass: crate::gpu::desc::MetalRenderPassDescriptor = Obj::from_retained(pass);
        // Each encoder is ended without its command buffer committed: nothing runs on the GPU.
        let cb = k.queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let marked = |e: &MetalRenderEncoder| {
            let object = ptr::from_ref(e.get()).cast::<AnyObject>();
            // SAFETY: a live encoder and the static key.
            unsafe {
                !objc2::ffi::objc_getAssociatedObject(object, (&raw const MARK).cast()).is_null()
            }
        };
        let refused = || {
            Err(reims_vgpu_core::testkit::capture::raise(
                OnFault::Return,
                "draw",
                "refused",
            ))
        };
        let (triangles, strip) = (MTLPrimitiveType::Triangle, MTLPrimitiveType::TriangleStrip);
        let e = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
        assert!(marked(&e));
        for primitive in [
            MTLPrimitiveType::Point,
            MTLPrimitiveType::Line,
            MTLPrimitiveType::LineStrip,
        ] {
            drew(e.get(), primitive, 3, 1, &Ok(()));
        }
        drew(e.get(), triangles, 3, 1, &refused());
        for (primitive, vertices, instances) in [
            (triangles, 0, 1),
            (triangles, 2, 1),
            (strip, 2, 1),
            (triangles, 3, 0),
            (strip, 4, 0),
        ] {
            drew(e.get(), primitive, vertices, instances, &Ok(()));
        }
        assert!(marked(&e), "kept");
        drew(e.get(), strip, 3, 1, &Ok(()));
        assert!(!marked(&e), "taken by a strip of one triangle");
        e.get().endEncoding();
        // Through the platform's encode, with a pipeline set: no vertices, then no instances,
        // keep the mark, and a triangle takes it.
        let e = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
        let set = RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&k.render_pipeline),
        };
        e.encode(OnFault::Return, set).unwrap();
        let draw = |vertex_count| RenderCmd::DrawPrimitives {
            primitive_type: triangles.0.to_u64(),
            vertex_start: 0,
            vertex_count,
        };
        e.encode(OnFault::Return, draw(0)).unwrap();
        assert!(marked(&e), "kept by a draw of no vertices");
        let no_instances = RenderCmd::DrawPrimitivesInstances {
            primitive_type: triangles.0.to_u64(),
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 0,
        };
        e.encode(OnFault::Return, no_instances).unwrap();
        assert!(marked(&e), "kept by a draw of no instances");
        e.encode(OnFault::Return, draw(3)).unwrap();
        assert!(!marked(&e), "taken by a draw of a triangle");
        e.get().endEncoding();
        // A second marked encoder on this thread: a draw on the first keeps the first's mark.
        let first = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
        first.get().endEncoding();
        let second = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
        drew(first.get(), triangles, 3, 1, &Ok(()));
        assert!(marked(&first) && marked(&second));
        drew(second.get(), triangles, 3, 1, &Ok(()));
        assert!(marked(&first) && !marked(&second));
        second.get().endEncoding();
    }

    /// An ICB-capable pipeline made after a pass over a mutable map in which the guest drew has
    /// the identifier Metal gives with no such pass: the pass makes no pipeline. A pass that draws
    /// nothing makes the no-draw pipeline, which moves that identifier, by 0x200 in this sequence
    /// (measured on the M3 Max, macOS 27.0.1, 6 runs alike)
    #[test]
    fn only_a_drawless_pass_moves_later_pipelines_identifiers() {
        use reims_vgpu_core::platform::resource::RenderPipeline as _;

        use crate::gpu::resource::testkit::render_pipeline;

        let k = kit();
        let mutable = side_map(&k, 1, true);
        let immutable = side_map(&k, 1, false);
        // Whatever Metal makes for a pass of this shape the first time, made first.
        let (error, _) = pass_over(
            &k,
            &immutable,
            Target::Color,
            Some(&k.render_pipeline),
            true,
        );
        assert_eq!(error, None);
        // The kit's pipeline made again and released: its record goes with it.
        let identifier = || render_pipeline(&k.device).unique_identifier();
        let metals = identifier();
        assert_eq!(
            identifier(),
            metals,
            "a released pipeline's record is freed"
        );
        let (error, stored) =
            pass_over(&k, &mutable, Target::Color, Some(&k.render_pipeline), true);
        assert_eq!(error, None, "the pass over the mutable map that drew");
        assert!(
            stored.chunks(4).all(|t| t == [0xbf, 0x80, 0x40, 0xff]),
            "{:02x?}",
            &stored[..8]
        );
        assert_eq!(identifier(), metals, "a pass that drew makes nothing");
        let (error, _) = clear_over(&k, &mutable, Target::Color, None);
        assert_eq!(error, None, "the drawless pass");
        assert_eq!(identifier(), metals + 0x200, "the no-draw pipeline's block");
    }

    /// A pass over a mutable map is marked, its end draws once, and a second end draws no more; a
    /// pass over an immutable map or none is not marked
    #[test]
    fn only_a_pass_over_a_map_not_immutable_is_marked() {
        use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};

        let k = kit();
        let marked = |map: Option<&MetalRateMap>| {
            let texture = side_texture(&k, MTLPixelFormat::BGRA8Unorm, 1, MTLStorageMode::Shared);
            let pass = MTLRenderPassDescriptor::new();
            // SAFETY: index 0 of the colour attachments.
            let a = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            a.setTexture(Some(&texture));
            a.setLoadAction(MTLLoadAction::Load);
            a.setStoreAction(MTLStoreAction::Store);
            pass.setRasterizationRateMap(map.map(Obj::get));
            let pass: crate::gpu::desc::MetalRenderPassDescriptor = Obj::from_retained(pass);
            let cb = k.queue.command_buffer(OnFault::Return).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
            let object = ptr::from_ref(e.get()).cast::<AnyObject>();
            // SAFETY: a live encoder and the static key.
            let mark = || unsafe {
                !objc2::ffi::objc_getAssociatedObject(object, (&raw const MARK).cast()).is_null()
            };
            let before = mark();
            end_pass(OnFault::Return, e.get()).unwrap();
            let after = mark();
            e.get().endEncoding();
            (before, after)
        };
        let mutable = side_map(&k, 1, true);
        let immutable = side_map(&k, 1, false);
        assert_eq!(mutable.mutability(), 1);
        assert_eq!(immutable.mutability(), 2);
        assert_eq!(marked(Some(&mutable)), (true, false));
        assert_eq!(marked(Some(&immutable)), (false, false));
        assert_eq!(marked(None), (false, false));
    }
}
