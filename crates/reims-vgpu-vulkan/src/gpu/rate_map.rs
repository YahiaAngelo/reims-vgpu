//! `id<MTLRasterizationRateMap>`, its descriptor and its layers: core's `RateMap`,
//! `RateMapDescriptor` and `RateMapLayerDescriptor`, over the runtime's map.
//!
//! What Metal answers, measured on this host (macOS 26.7, M3 Max) without the API validation
//! layer:
//!
//! - **A layer descriptor** (`initWithSampleCount:`) is nil for a count of 0 on either axis, and
//!   otherwise holds that many zeroed samples on each axis, the horizontal then the vertical in
//!   one allocation; its `sampleCount` is the count with depth 0, whatever depth was asked.
//! - **A map descriptor** holds its layers by reference: the layer `layers` answers at an index is
//!   the one set there, nil past them. Its `layerCount` counts the layers set from index 0 up to
//!   the first index with none. Its mutability is Immutable (2) until set.
//! - **The minimum factor** (`setMinFactor:`) changes nothing a map answers (physical sizes,
//!   coordinate maps, parameter data), set on the map's descriptor or a reset's, so it is not
//!   kept.
//! - **An immutable map** is compiled at creation: the runtime's compile, whose parameter data
//!   is Metal's bytes and whose coordinate maps are Apple's arithmetic over them, bit for bit.
//!   Where the runtime refuses the descriptor it dies. Metal crashes there too,
//!   at three layers and at a screen side past 16384. It makes a map of a screen side of 0.
//! - **A map of no layers** is made, at the creation and at a reset, as Metal makes it, and the
//!   runtime makes it too: its layer count is 0, its parameter data Metal's (no
//!   horizontal tables, a vertical axis of the least quality). A pass over it loads, clears and
//!   stores as Metal's does. Metal's draws in such a pass rasterize by what earlier maps left
//!   (measured: nothing in one run, a corner of 1,018 fragments of a 64-pixel target after other
//!   maps' passes in another; the runtime's measurement records rows 0 to 15 drawn as with no map
//!   after a released map of one layer, here too), where the runtime's draw nothing, so a draw and
//!   an execution of indirect commands in it die, and a tile dispatch, which only draws were
//!   measured against, relays it, and dies in strict mode ([`crate::gpu::encode_render`]).
//!   Its coordinate maps end Metal's process, so they die here
//!   ([`unmapped`]).
//! - **A mutable map** (`setMutability:` Mutable, 1) is not compiled until it is reset. Until then
//!   it answers the descriptor's screen size and layer count, a physical size of 0 by 0 for every
//!   layer, screen-to-physical (0, 0), and parameter data of zeros. Metal makes it whatever its
//!   descriptor holds (three layers, a screen side of 0 or 40000), and crashes at
//!   physical-to-screen. `resetUsingDescriptor:` compiles the descriptor given into the map, which
//!   keeps its own mutability. A reset the runtime refuses dies, as no path reaches it under a
//!   handler.
//! - **`parameterBufferSizeAndAlign`** is 24,624 bytes aligned to 4 for every map, compiled or
//!   not. A compiled map's parameter data is the runtime's, which is Metal's, in Apple's layout,
//!   byte for byte, and which the translated decoder of a function reads with
//!   Apple's arithmetic; its copy writes all of it. A map not yet compiled writes 24,624 zeros,
//!   as Metal's does.
//! - **The granularity** is 32 by 32, depth 1, compiled or not; a physical size has depth 1.
//!
//! **In a pass.** A pass descriptor holds the map by reference (`desc.rs`).
//! - **A pass of a map whose rates vary** (a tile spanning other than as many screen pixels as
//!   physical ones) loads, clears and stores as Metal's does, and the map's parameter data and
//!   coordinate maps are Metal's, but the runtime's draws in it rasterize by regions whose lines
//!   are the map's tiles' exact lines, where Metal's are its fixed-point ones. The runtime
//!   announces it as it compiles the map, at its creation or a
//!   reset (`rasterization-rate-map`, [`Regions`]), which is relayed, and a draw, an execution of
//!   indirect commands that is not empty and a tile dispatch in such a pass relay it again, and
//!   die in strict mode ([`crate::answer::Inexact::RateMap`], [`crate::gpu::encode_render`]).
//! - **A pass of a mutable map not yet reset** rasterizes nothing, as on Metal (measured: a
//!   full-screen triangle's vertex function writes its buffer, its fragment function never runs,
//!   and the colour, depth and stencil attachments keep their clears), so its encoder keeps its
//!   scissor rectangles empty, and a tile dispatch in it, which only draws were measured
//!   against, dies ([`crate::gpu::encode_render`]).
//! - **When the map is read.** Metal's pass reads the map as the GPU runs it (measured: a reset
//!   after the encoder is made, after its draw, after `endEncoding` and just after `commit` each
//!   shows in the pass, from a map not yet reset and from a compiled one). The runtime plans each
//!   draw by the map as it is encoded, so an encoder takes the map as it is at its creation, and
//!   the map keeps the command buffers whose passes took it, by their progress
//!   ([`crate::gpu::command::Progress`]), which outlives core's handle: a reset while one of them
//!   may still run on the GPU (not committed, or committed and not complete) dies, as this
//!   platform would draw by the map Metal no longer holds. One that goes uncommitted runs nothing,
//!   and no longer counts. What an encoder takes and the registration of its command buffer, once
//!   the encoder exists, are under the map's lock, which a reset's check and compile hold too.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use metal2vulkan::bind;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::Handle;
use reims_vgpu_core::platform::rate_map::{RateMap, RateMapDescriptor, RateMapLayerDescriptor};
use reims_vgpu_core::platform::types::{MtlCoordinate2D, MtlSize, SizeAndAlign};
use reims_vgpu_core::platform::values::RateMapDesc;
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::sync::Mutex;

use crate::answer::{Policy, get, hearing, plain, send, unrepresentable};
use crate::gpu::command::{Progress, VulkanCommandBuffer};
use crate::gpu::device::VulkanDevice;
use crate::gpu::resource::VulkanBuffer;
use crate::platform::{Obj, VulkanPlatform};

/// `MTLMutabilityImmutable`: a descriptor's mutability until it is set.
const IMMUTABLE: u64 = 2;

/// `MTLMutabilityMutable`: a map made of a descriptor of it is compiled only at its reset.
const MUTABLE: u64 = 1;

/// Metal's `parameterBufferSizeAndAlign` of every map on this host, which the runtime's compiled
/// maps answer too.
const PARAMETER_DATA: SizeAndAlign = SizeAndAlign {
    size: 24624,
    align: 4,
};

/// The physical granularity of every map, compiled or not, as Metal answers it (the runtime's
/// compiled maps answer it too).
const GRANULARITY: u64 = 32;

// ---- Layers.

/// `MTLRasterizationRateLayerDescriptor *`: Metal's sample count and its sample storage.
#[derive(Debug)]
pub struct LayerObject {
    /// The horizontal samples' count, then the vertical's.
    counts: (usize, usize),
    /// The horizontal samples, then the vertical, as `f32` bits: core writes them in place
    /// through the storage getters' pointers.
    samples: Box<[AtomicU32]>,
}

/// `MTLRasterizationRateLayerDescriptor *`.
pub type VulkanRateMapLayerDescriptor = Obj<LayerObject>;

impl LayerObject {
    /// The samples from `at` on, `n` of them.
    fn read(&self, at: usize, n: usize) -> Vec<f32> {
        self.samples[at..at + n]
            .iter()
            .map(|s| f32::from_bits(s.load(Ordering::Relaxed)))
            .collect()
    }

    /// The compile layer's form of the layer, its samples as they stand now.
    fn samples(&self) -> bind::RateMapLayerDescriptor {
        let (w, h) = self.counts;
        bind::RateMapLayerDescriptor::new(self.read(0, w), self.read(w, h))
    }

    /// The address of sample `at`.
    fn storage(&self, at: usize) -> HostPtr {
        HostPtr::from_ptr(self.samples[at..].as_ptr().cast::<u32>().cast_mut())
    }
}

/// `initWithSampleCount:` of `sample_count`: nil for a count of 0 on either axis, or one of
/// samples it has no memory for, as Metal answers.
pub(crate) fn new_layer(sample_count: MtlSize) -> Option<VulkanRateMapLayerDescriptor> {
    let w = usize::try_from(sample_count.width).ok()?;
    let h = usize::try_from(sample_count.height).ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    let n = w.checked_add(h)?;
    let mut samples = Vec::new();
    samples.try_reserve_exact(n).ok()?;
    samples.resize_with(n, || AtomicU32::new(0));
    Some(Obj::new(LayerObject {
        counts: (w, h),
        samples: samples.into_boxed_slice(),
    }))
}

impl RateMapLayerDescriptor for VulkanRateMapLayerDescriptor {
    fn horizontal_sample_storage(&self) -> HostPtr {
        plain(|| self.storage(0))
    }

    /// The counts, depth 0, as Metal answers them.
    fn sample_count(&self) -> MtlSize {
        let (w, h) = self.counts;
        MtlSize::new(w as u64, h as u64, 0)
    }

    fn vertical_sample_storage(&self) -> HostPtr {
        plain(|| self.storage(self.counts.0))
    }
}

// ---- Descriptors.

/// `MTLRasterizationRateMapDescriptor *`: its screen size, mutability and layers.
#[derive(Debug)]
pub struct DescriptorObject {
    /// `screenSize`.
    screen_size: MtlSize,
    /// `mutability`.
    mutability: u64,
    /// The layers set, by index.
    layers: Mutex<Vec<Option<VulkanRateMapLayerDescriptor>>>,
}

/// `MTLRasterizationRateMapDescriptor *`.
pub type VulkanRateMapDescriptor = Obj<DescriptorObject>;

/// A descriptor of `desc`'s screen size and mutability (Immutable when unset), with no layers;
/// its minimum factor, which changes nothing a map answers (the module docs), is not kept.
pub(crate) fn new_descriptor(desc: &RateMapDesc) -> VulkanRateMapDescriptor {
    Obj::new(DescriptorObject {
        screen_size: desc.screen_size,
        mutability: desc.mutability.unwrap_or(IMMUTABLE),
        layers: Mutex::new(Vec::new()),
    })
}

impl DescriptorObject {
    /// The layers set from index 0 up to the first index with none: Metal's `layerCount`.
    fn counted(&self) -> Vec<VulkanRateMapLayerDescriptor> {
        plain(|| {
            self.layers
                .lock()
                .iter()
                .map_while(|l| l.as_ref().map(Handle::clone_ref))
                .collect()
        })
    }

    /// The compile layer's form of the descriptor, its layers' samples as they stand now, for
    /// `sel`.
    fn compiled(&self, sel: &'static str) -> bind::RateMapDescriptor {
        let side = |v: u64| {
            u32::try_from(v)
                .unwrap_or_else(|_| unrepresentable(sel, &format!("a screen side of {v}")))
        };
        let mut d = bind::RateMapDescriptor::new([
            side(self.screen_size.width),
            side(self.screen_size.height),
        ]);
        d.layers = self.counted().iter().map(|l| l.samples()).collect();
        d
    }
}

/// Dies at `sel`, a coordinate map of `map`, where it has no layers: Metal's ends the process
/// there, where the runtime's answers 0 (the module docs).
fn unmapped(sel: &'static str, map: &rt::RasterizationRateMap) {
    if map.layer_count() == 0 {
        unrepresentable(
            sel,
            "a coordinate map of a rasterization rate map of no layers, which ends Metal's \
             process",
        );
    }
}

impl RateMapDescriptor<VulkanPlatform> for VulkanRateMapDescriptor {
    /// The layer is kept by reference, as Metal keeps it.
    fn set_layer(
        &self,
        _on: OnFault,
        layer: &VulkanRateMapLayerDescriptor,
        layer_index: u64,
    ) -> RaiseResult<()> {
        let sel = "setLayer:atIndex:";
        let at = usize::try_from(layer_index)
            .ok()
            .filter(|&at| at < 1 << 16)
            .unwrap_or_else(|| unrepresentable(sel, &format!("layer {layer_index}")));
        plain(|| {
            let mut layers = self.layers.lock();
            if layers.len() <= at {
                layers.resize_with(at + 1, || None);
            }
            layers[at] = Some(layer.clone_ref());
        });
        Ok(())
    }

    fn layer_count(&self) -> u64 {
        self.counted().len() as u64
    }

    /// The layer set at the index, nil where none is.
    fn layer(
        &self,
        _on: OnFault,
        layer_index: u64,
    ) -> RaiseResult<Option<VulkanRateMapLayerDescriptor>> {
        Ok(plain(|| {
            let layers = self.layers.lock();
            usize::try_from(layer_index)
                .ok()
                .and_then(|at| layers.get(at))
                .and_then(|l| l.as_ref().map(Handle::clone_ref))
        }))
    }
}

// ---- Maps.

/// What a map is now.
#[derive(Debug)]
enum State {
    /// The runtime's compile of its descriptor, and whether its rates vary, which the runtime
    /// announced as it compiled it (`rasterization-rate-map`, [`Regions`]).
    Compiled(rt::RasterizationRateMap, Regions),
    /// A mutable map not yet reset: its descriptor's screen size and layer count.
    Unset { screen_size: MtlSize, layers: u64 },
}

/// Whether a compiled map's rates vary, which the runtime announces as it compiles one
/// (`rasterization-rate-map`, [`crate::answer::HEARD`]): a pass over such a map draws by regions
/// whose lines are its tiles' exact lines, where Apple's are its fixed-point ones, so a draw, an
/// execution of indirect commands that is not empty and a tile dispatch in it relay it, and die
/// in strict mode ([`crate::answer::Inexact::RateMap`], [`crate::gpu::encode_render`]). The runtime announces nothing for a map whose every tile spans
/// as many screen pixels as physical ones, whose passes draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Regions {
    /// The identity inside the screen: the runtime announced nothing.
    Identity,
    /// Its rates vary.
    Varying,
}

impl Regions {
    /// What the runtime announced as it compiled a map, `heard` inside the call.
    fn of(heard: &[rt::Degradation]) -> Regions {
        if heard.contains(&rt::Degradation::RasterizationRateMap) {
            Regions::Varying
        } else {
            Regions::Identity
        }
    }
}

/// `id<MTLRasterizationRateMap>`: the runtime's map once compiled, and its mutability.
#[derive(Debug)]
pub struct RateMapObject {
    /// The device, which compiles a reset.
    device: rt::Device,
    /// `mutability`: its descriptor's, which a reset keeps.
    mutability: u64,
    /// What it is now, and the passes that took it, under one lock.
    inner: Mutex<Inner>,
}

/// A map's state and its passes.
#[derive(Debug)]
struct Inner {
    /// What it is now.
    state: State,
    /// The progress of the command buffers whose passes took the map, while the GPU may run them.
    passes: Vec<Arc<Progress>>,
}

/// `id<MTLRasterizationRateMap>`.
pub type VulkanRateMap = Obj<RateMapObject>;

/// `newRasterizationRateMapWithDescriptor:`: a mutable descriptor's map not yet compiled, and an
/// immutable one's compiled, dying where the runtime refuses it (the module docs). Metal answers
/// no nil here.
pub(crate) fn new_map(
    device: &VulkanDevice,
    descriptor: &VulkanRateMapDescriptor,
) -> RaiseResult<Option<VulkanRateMap>> {
    let sel = "newRasterizationRateMapWithDescriptor:";
    let state = if descriptor.mutability == MUTABLE {
        State::Unset {
            screen_size: descriptor.screen_size,
            layers: descriptor.layer_count(),
        }
    } else {
        let d = descriptor.compiled(sel);
        let (made, heard) = hearing(|| {
            send(sel, Policy::Dies, || {
                device.raw.new_rasterization_rate_map(&d)
            })
        });
        State::Compiled(made?, Regions::of(&heard))
    };
    Ok(Some(Obj::new(RateMapObject {
        device: device.raw.clone(),
        mutability: descriptor.mutability,
        inner: Mutex::new(Inner {
            state,
            passes: Vec::new(),
        }),
    })))
}

impl RateMapObject {
    /// `open`, the creation of an encoder of a pass on `command_buffer` that takes the map, given
    /// the runtime's map as it is now and whether its rates vary (none for a mutable map not yet
    /// reset, whose pass rasterizes nothing), under the map's lock. Once it answers an encoder,
    /// the map keeps the command buffer's progress while the GPU may run it; one that answers
    /// none leaves it unregistered. No reset comes between what the encoder takes and the
    /// registration.
    pub(crate) fn for_pass<T>(
        &self,
        command_buffer: &VulkanCommandBuffer,
        open: impl FnOnce(Option<(rt::RasterizationRateMap, Regions)>) -> RaiseResult<Option<T>>,
    ) -> RaiseResult<Option<T>> {
        let mut inner = plain(|| self.inner.lock());
        let map = match &inner.state {
            State::Compiled(map, regions) => Some((map.clone(), *regions)),
            State::Unset { .. } => None,
        };
        let opened = open(map)?;
        if opened.is_some() {
            let progress = command_buffer.progress("renderCommandEncoderWithDescriptor:");
            inner.passes.retain(|p| p.running());
            if !inner.passes.iter().any(|p| Arc::ptr_eq(p, &progress)) {
                inner.passes.push(progress);
            }
        }
        Ok(opened)
    }

    /// `answer` of the compiled map, or `unset`'s of one not yet reset.
    fn of<T>(
        &self,
        answer: impl FnOnce(&rt::RasterizationRateMap) -> T,
        unset: impl FnOnce(MtlSize, u64) -> T,
    ) -> T {
        match &plain(|| self.inner.lock()).state {
            State::Compiled(map, _) => answer(map),
            &State::Unset {
                screen_size,
                layers,
            } => unset(screen_size, layers),
        }
    }
}

/// Core's coordinate as the runtime's.
fn coordinate(c: MtlCoordinate2D) -> rt::Coordinate2D {
    rt::Coordinate2D::new(c.x, c.y)
}

/// The runtime's coordinate as core's.
fn mtl_coordinate(c: rt::Coordinate2D) -> MtlCoordinate2D {
    MtlCoordinate2D { x: c.x, y: c.y }
}

/// The runtime's size as core's.
fn mtl_size(s: rt::Size) -> MtlSize {
    MtlSize::new(s.width, s.height, s.depth)
}

impl RateMap<VulkanPlatform> for VulkanRateMap {
    /// Metal's 24,624 bytes at `offset`: a compiled map's parameter data, the runtime's, which is
    /// Metal's; zeros for a map not yet compiled. Core checks the buffer's room for them first. A
    /// refusal of the runtime's (a `Private` buffer, where Metal's copy ends the process) dies.
    fn copy_parameter_data(&self, buffer: &VulkanBuffer, offset: u64) -> RaiseResult<()> {
        let sel = "copyParameterDataToBuffer:offset:";
        let compiled = self.of(
            |map| {
                send(sel, Policy::Dies, || {
                    map.copy_parameter_data_to_buffer(&buffer.raw, offset)
                })
                .map(|()| true)
            },
            |_, _| Ok(false),
        )?;
        if compiled {
            return Ok(());
        }
        let contents = plain(|| buffer.raw.contents())
            .unwrap_or_else(|| unrepresentable(sel, "a Private buffer, where Metal's copy ends"));
        let end = offset + PARAMETER_DATA.size;
        if end > plain(|| buffer.raw.length()) {
            unrepresentable(sel, "parameter data past the buffer's end");
        }
        let (from, end) = (
            usize::try_from(offset).unwrap_or(usize::MAX),
            usize::try_from(end).unwrap_or(usize::MAX),
        );
        // SAFETY: the buffer's mapping holds its length's bytes from `contents`, and `end` is at
        // most that length. A command that reads or writes those bytes meanwhile races, as on
        // Metal.
        unsafe { std::ptr::write_bytes(contents.as_ptr().add(from), 0, end - from) };
        Ok(())
    }

    fn layer_count(&self) -> u64 {
        self.of(rt::RasterizationRateMap::layer_count, |_, layers| layers)
    }

    /// Metal crashes at a map not yet compiled.
    fn physical_to_screen(
        &self,
        physical_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        let sel = "mapPhysicalToScreenCoordinates:forLayer:";
        Ok(self.of(
            |map| {
                unmapped(sel, map);
                mtl_coordinate(map.map_physical_to_screen_coordinates(
                    coordinate(physical_coordinates),
                    layer_index,
                ))
            },
            |_, _| {
                unrepresentable(
                    sel,
                    "a mutable rate map not yet reset, which Metal crashes at",
                )
            },
        ))
    }

    fn screen_to_physical(
        &self,
        screen_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        let sel = "mapScreenToPhysicalCoordinates:forLayer:";
        Ok(self.of(
            |map| {
                unmapped(sel, map);
                mtl_coordinate(map.map_screen_to_physical_coordinates(
                    coordinate(screen_coordinates),
                    layer_index,
                ))
            },
            |_, _| MtlCoordinate2D { x: 0.0, y: 0.0 },
        ))
    }

    fn mutability(&self) -> u64 {
        self.mutability
    }

    /// Metal's (the module docs), which a compiled map's runtime answers too: a runtime answer
    /// of another size dies, as its copy would write other bytes than Metal's.
    fn parameter_buffer_size_and_align(&self) -> SizeAndAlign {
        let size = self.of(
            |map| map.parameter_buffer_size_and_align().size,
            |_, _| PARAMETER_DATA.size,
        );
        if size != PARAMETER_DATA.size {
            unrepresentable(
                "parameterBufferSizeAndAlign",
                &format!(
                    "parameter data of {size} bytes, not Metal's {}",
                    PARAMETER_DATA.size
                ),
            );
        }
        PARAMETER_DATA
    }

    fn physical_granularity(&self) -> MtlSize {
        MtlSize::new(GRANULARITY, GRANULARITY, 1)
    }

    fn physical_size(&self, layer_index: u64) -> RaiseResult<MtlSize> {
        Ok(self.of(
            |map| mtl_size(map.physical_size_for_layer(layer_index)),
            |_, _| MtlSize::new(0, 0, 1),
        ))
    }

    /// The descriptor compiled into the map, which keeps its mutability; a refusal dies, and so
    /// does a reset while a pass that took the map may still run on the GPU (the module docs).
    /// The check and the compile hold the map's lock, which a pass's creation holds too.
    fn reset(&self, descriptor: &VulkanRateMapDescriptor) {
        let sel = "resetUsingDescriptor:";
        let mut inner = plain(|| self.inner.lock());
        inner.passes.retain(|p| p.running());
        if !inner.passes.is_empty() {
            unrepresentable(
                sel,
                "a reset before a pass of the map has run, which Metal's pass would read and the \
                 runtime's, planned as it was encoded, would not",
            );
        }
        let d = descriptor.compiled(sel);
        let (map, heard) = hearing(|| get(sel, || self.device.new_rasterization_rate_map(&d)));
        inner.state = State::Compiled(map, Regions::of(&heard));
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::Platform;
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::device::tests::dev;
    use crate::platform::tests::platform;

    /// A layer of `horizontal` and `vertical` samples, written through its storage as core
    /// writes them.
    fn layer<P: Platform>(
        p: &P,
        horizontal: &[f32],
        vertical: &[f32],
    ) -> P::RateMapLayerDescriptor {
        let l = p
            .new_rate_map_layer(
                OnFault::Die,
                MtlSize::new(horizontal.len() as u64, vertical.len() as u64, 0),
            )
            .unwrap()
            .unwrap();
        let write = |at: HostPtr, samples: &[f32]| {
            // SAFETY: the layer's storage holds as many samples as it was made with.
            unsafe {
                std::ptr::copy_nonoverlapping(samples.as_ptr(), at.as_ptr::<f32>(), samples.len())
            }
        };
        write(l.horizontal_sample_storage(), horizontal);
        write(l.vertical_sample_storage(), vertical);
        l
    }

    /// A descriptor of a 256 by 128 screen, mutable when asked, of one layer whose samples
    /// are `quality` then 1 across and `quality` down.
    fn descriptor<P: Platform>(p: &P, quality: f32, mutable: bool) -> P::RateMapDescriptor {
        let d = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(256, 128, 1),
            mutability: mutable.then_some(MUTABLE),
            min_factor: mutable.then_some(0.5),
        });
        d.set_layer(OnFault::Die, &layer(p, &[quality, 1.0], &[quality]), 0)
            .unwrap();
        d
    }

    /// What Metal crashes at dies at the selector: a map of three layers, which the runtime
    /// refuses, and physical-to-screen of a mutable map not yet reset
    #[test]
    fn what_metal_crashes_at_dies() {
        let p = platform();
        let d = dev();
        let three = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(64, 32, 1),
            ..RateMapDesc::default()
        });
        for i in 0..3 {
            three
                .set_layer(OnFault::Die, &layer(&p, &[1.0], &[1.0]), i)
                .unwrap();
        }
        assert_eq!(three.layer_count(), 3);
        let site = catch_death(|| d.new_rate_map(&three)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "newRasterizationRateMapWithDescriptor:")
        );
        let unset = d.new_rate_map(&descriptor(&p, 1.0, true)).unwrap().unwrap();
        let site = catch_death(|| unset.physical_to_screen(MtlCoordinate2D { x: 1.0, y: 1.0 }, 0))
            .unwrap_err();
        assert_eq!(site.file, "mapPhysicalToScreenCoordinates:forLayer:");
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
