//! What GPU work may write: the host memory each object it binds or makes resident lies in
//! ([`Memory`]), and the slots its pipeline state's functions write, by the state's reflection
//! ([`Reflection`], [`Slots`]), with whether a write through a slot happens ([`Whether`]).
//!
//! A write through an object on a surface whose GPU writes never reach its memory
//! ([`crate::os::gpu_read_only`]) dies where Metal faults or drops it otherwise than this platform
//! ([`crate::os::dropped_write`]): a blit's (`crate::gpu::encode_blit`), a dispatch's
//! (`crate::gpu::predication`) and a draw's (`crate::gpu::encode_render`). So does a dispatch's
//! or a draw's write through a texture of a 4:4:4 two-plane YCbCr format whose writes the runtime
//! refuses ([`crate::gpu::ycbcr::refused_write`]). A render state that
//! rounds its texture writes as the runtime does not reads its fragment function's slots too
//! ([`crate::gpu::serialized::rounded_texture_writes`]).
//!
//! A write through an object over memory the runtime imports through a writable twin of
//! read-only pages ([`Memory::kept`]; on Linux, [`crate::os::TWINS_READ_ONLY`]) would reach those
//! pages, where Metal's GPU drops it or faults, so the command that may make it dies where it is
//! encoded, at every extent ([`kept_write`]): a blit into such an object; a dispatch or a draw
//! through one bound at a slot its function writes, by its reflection (every slot where the
//! reflection cannot tell), or made resident with write usage; and a pass whose attachment lies
//! in one, whatever its actions, or that resolves into one or keeps its visibility results in one
//! (`crate::gpu::encode_blit`, `crate::gpu::predication`, `crate::gpu::encode_render`); and a
//! CPU write of such a texture (`replaceRegion:`, `crate::gpu::resource`). A
//! function's write through an address it loads (an argument buffer's, a pointer's) into such an
//! object not made resident for writing names nothing and is not seen: on Linux it reaches the
//! pages, as the runtime imports no read-only memory there.
use std::ops::Range;
use std::sync::Arc;

use metal2vulkan::metal::{BindingAccess, BindingType, PixelFormat};
use metal2vulkan::pipeline::ComputePipelineDescriptor;
use reims_vgpu_core::num::ToU64;

use crate::answer::plain;
use crate::gpu::over_pages::Seen;
use crate::gpu::resource::{BufferObject, HeapObject, RenderDescriptor, TextureObject, Under};
use crate::os::OsSurface;

/// The memory an object's bytes lie in, which a write of the object may change.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Memory {
    /// The host memory: the first byte and end of each range. An object of none (a `Private`
    /// buffer, a texture of memory of its own) is written nowhere a guest's surface lies.
    pub ranges: Vec<Range<u64>>,
    /// Whether the GPU's writes through the object never reach the memory: the surface it was
    /// made on is one of [`crate::os::gpu_read_only`].
    pub dropped: bool,
    /// The format of the object, a texture over a surface, where the runtime refuses the GPU's
    /// writes through it: one of a 4:4:4 format of the two-plane YCbCr family whose writes it
    /// does not store as Metal's GPU does ([`crate::gpu::ycbcr::writes_refused`]).
    pub refused: Option<PixelFormat>,
    /// Whether the runtime imports the memory through a writable twin of read-only pages, where
    /// the GPU's writes through the object reach them and Metal's GPU drops them ([`kept_write`]).
    pub kept: Kept,
}

/// Whether the runtime imports an object's memory through a writable twin of read-only pages
/// ([`Memory::kept`]), asked as work that may write through the object is encoded
/// ([`Kept::holds`]).
#[derive(Clone, Debug, Default)]
pub enum Kept {
    /// It does not.
    #[default]
    No,
    /// It does for the object's life: a surface's memory ([`surface_kept`]).
    Always,
    /// It does until a detach leaves the object, a buffer or heap, in no memory
    /// ([`crate::gpu::backing::Backing::writes_kept`]); no replace gives it other memory while
    /// work encoded holds it, as the runtime refuses the replace (`crate::gpu::backing`).
    Until(Arc<Seen>),
}

impl Kept {
    /// [`Kept::Always`] where `kept`, else [`Kept::No`].
    pub fn always(kept: bool) -> Kept {
        if kept { Kept::Always } else { Kept::No }
    }

    /// Whether the GPU's writes through the object reach read-only memory now.
    pub fn holds(&self) -> bool {
        match self {
            Kept::No => false,
            Kept::Always => true,
            Kept::Until(seen) => !seen.zeroed(),
        }
    }
}

impl PartialEq for Kept {
    fn eq(&self, other: &Kept) -> bool {
        match (self, other) {
            (Kept::No, Kept::No) | (Kept::Always, Kept::Always) => true,
            (Kept::Until(a), Kept::Until(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for Kept {}

impl Memory {
    /// Whether it holds no memory.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The memory of an object made on `surface`: all of the surface's
    /// ([`crate::os::surface_memory`]).
    pub fn of_surface(surface: &OsSurface) -> Memory {
        Memory {
            ranges: crate::os::surface_memory(surface),
            dropped: crate::os::gpu_read_only(surface),
            refused: None,
            kept: Kept::always(surface_kept(surface)),
        }
    }

    /// The memory of the bytes `bytes` (a first byte and an end, from its base address) of
    /// `surface` ([`crate::os::surface_bytes_memory`]).
    pub fn of_surface_bytes(surface: &OsSurface, bytes: Range<u64>) -> Memory {
        Memory {
            ranges: crate::os::surface_bytes_memory(surface, std::slice::from_ref(&bytes)),
            dropped: crate::os::gpu_read_only(surface),
            refused: None,
            kept: Kept::always(surface_kept(surface)),
        }
    }

    /// The memory of an object made over the host memory `ranges`, through a writable twin of
    /// read-only pages where `kept`.
    pub fn over(ranges: Vec<Range<u64>>, kept: Kept) -> Memory {
        Memory {
            ranges,
            dropped: false,
            refused: None,
            kept,
        }
    }

    /// What a bind keeps of it ([`Bound`]).
    pub fn bound(self) -> Bound {
        Bound {
            meets: !self.is_empty(),
            dropped: self.dropped,
            refused: self.refused,
            kept: self.kept,
        }
    }
}

/// What a bind or a residency keeps of an object's [`Memory`]: whether it lies in any, and what
/// the GPU's writes through it do there. What work encoded later may write is decided from these
/// alone (`crate::gpu::encode_render`'s `Reach`), never from the ranges.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bound {
    /// Whether it lies in any memory: [`Memory::is_empty`] answers `false`.
    pub meets: bool,
    /// [`Memory::dropped`].
    pub dropped: bool,
    /// [`Memory::refused`].
    pub refused: Option<PixelFormat>,
    /// [`Memory::kept`].
    pub kept: Kept,
}

impl Bound {
    /// [`Memory::over`]'s, of memory it lies in where `meets`.
    fn over(meets: bool, kept: Kept) -> Bound {
        Bound {
            meets,
            kept,
            ..Bound::default()
        }
    }
}

/// Whether the runtime imports the memory of an object made on `surface` through a writable
/// twin of read-only pages: one whose GPU writes never reach its memory
/// ([`crate::os::gpu_read_only`]), every buffer over which lies over a read-only alias of its
/// allocation ([`crate::os::surface_alias`]), where the runtime twins read-only memory
/// ([`crate::os::TWINS_READ_ONLY`]).
fn surface_kept(surface: &OsSurface) -> bool {
    crate::os::TWINS_READ_ONLY && crate::os::gpu_read_only(surface)
}

/// Whether the GPU's writes through `t` reach read-only memory where Metal's drop them
/// ([`Memory::kept`]), as [`texture_memory`] answers, without the ranges.
pub fn texture_kept(t: &TextureObject) -> bool {
    match (&t.surface, &t.under) {
        (Some(s), _) => surface_kept(s),
        (None, Some(Under::Buffer(b, _))) => buffer_kept(b),
        (None, Some(Under::Heap(h, _))) => h.backing.writes_kept(),
        _ => false,
    }
}

/// Whether the GPU's writes through `b` reach read-only memory where Metal's drop them
/// ([`Memory::kept`]), as [`buffer_memory`] answers, without the ranges.
pub fn buffer_kept(b: &BufferObject) -> bool {
    match (&b.surface, &b.heap) {
        (Some(s), _) => surface_kept(s),
        (None, Some(h)) => h.backing.writes_kept(),
        (None, None) => b.backing.writes_kept(),
    }
}

/// Dies at `sel`, a command that may write `what` an object over memory the runtime imports
/// through a writable twin of read-only pages ([`Memory::kept`]): read-only guest pages (a
/// read-only replace or heap, a resource made with `MTLResourceOptions` bit 24, a range core maps
/// read-only, a surface the GPU only reads), or the zeros past a read-only alias's pages. The
/// write would land in those pages or in the twin, where later reads see it, and Metal's GPU
/// drops it or faults the command buffer, and its CPU's write through the pages' read-only
/// mapping ends the process; the runtime imports no read-only memory on this host
/// (Linux drivers pin imported pages for writing), so the write is never sent
/// (`crate::os::import_view`).
pub(crate) fn kept_write(sel: &'static str, what: &str) -> ! {
    crate::answer::unrepresentable(
        sel,
        &format!(
            "{what} read-only memory (a read-only replace, heap, resource, guest range or \
             surface), which this host's runtime imports through a writable twin, so the write \
             would reach it, where Metal's never does: its GPU drops the write or faults, and its \
             CPU's write through the read-only mapping ends the process"
        ),
    )
}

/// The host memory `t`'s texels lie in, which a write of `t` may change: its plane's bytes of its
/// surface ([`plane_bytes`]), or its bytes of its buffer's or its heap's ([`Under`]); none for one
/// of memory of its own.
pub fn texture_memory(t: &TextureObject) -> Memory {
    match (&t.surface, &t.under) {
        (Some(s), _) => match &t.plane {
            Some(p) => {
                // A rotated texture's rows may run along either side.
                let (width, height) = plain(|| (t.raw.width(), t.raw.height()));
                let rows = if plain(|| t.raw.rotation()) == 0 {
                    height
                } else {
                    width.max(height)
                };
                let pf = plain(|| t.raw.pixel_format());
                Memory {
                    refused: crate::gpu::ycbcr::writes_refused(pf).then_some(pf),
                    ..Memory::of_surface_bytes(s, plane_bytes(p.offset, p.bytes_per_row, rows))
                }
            }
            None => Memory::of_surface(s),
        },
        (None, Some(Under::Buffer(b, bytes))) => buffer_bytes_memory(b, bytes.clone()),
        (None, Some(Under::Heap(h, bytes))) => heap_bytes_memory(h, bytes.clone()),
        _ => Memory::default(),
    }
}

/// The bytes of a surface, from its base address, that a texture of `rows` rows over a plane
/// at `offset` with rows `pitch` bytes apart (its `bytesPerRow`) reads and writes: its rows from
/// the offset, `pitch` apart, as the CPU lays them out, and from the offset rounded down to 16,
/// rows `pitch` rounded down to 16 apart, or 16 where that is 0, as the GPU does (the runtime's
/// `linear_plan.rs`, measured).
fn plane_bytes(offset: u64, pitch: u64, rows: u64) -> Range<u64> {
    let start = offset - offset % 16;
    let gpu_pitch = match pitch - pitch % 16 {
        0 => 16,
        p => p,
    };
    let cpu_end = offset.saturating_add(pitch.saturating_mul(rows));
    let gpu_end = start.saturating_add(gpu_pitch.saturating_mul(rows));
    start..cpu_end.max(gpu_end)
}

/// The host memory `b`'s bytes lie in now ([`buffer_bytes_memory`]).
pub fn buffer_memory(b: &BufferObject) -> Memory {
    buffer_bytes_memory(b, 0..plain(|| b.raw.length()))
}

/// The host memory `b`'s bytes `bytes` (a first byte and an end) lie in now: in its surface's
/// memory (`newBufferWithIOSurface:`, from the surface's base address), at its place in its
/// heap's, in the guest's pages of its last replace, or in the memory it was made over; none
/// where a detach left it in no memory, or for a `Private` one.
pub fn buffer_bytes_memory(b: &BufferObject, bytes: Range<u64>) -> Memory {
    if let Some(s) = &b.surface {
        return Memory::of_surface_bytes(s, bytes);
    }
    if let Some(h) = &b.heap {
        // A heap buffer's GPU address is its heap's plus its offset there.
        let at = plain(|| b.raw.gpu_address().0.saturating_sub(h.raw.gpu_address().0));
        return heap_bytes_memory(
            h,
            at.saturating_add(bytes.start)..at.saturating_add(bytes.end),
        );
    }
    let kept = b.backing.kept();
    let paged = b.backing.in_pages(&bytes);
    if paged.is_some() || b.backing.kept_contents().is_some() {
        return Memory::over(paged.unwrap_or_default(), kept);
    }
    let contents = plain(|| b.raw.contents()).map(|first| {
        let first = first.as_ptr().addr().to_u64();
        std::iter::once(first.saturating_add(bytes.start)..first.saturating_add(bytes.end))
            .collect()
    });
    Memory::over(contents.unwrap_or_default(), kept)
}

/// What a bind keeps of `b`'s memory: [`buffer_memory`]'s [`Bound`], read as it reads it, but
/// without collecting the ranges of its bytes in the guest's pages, one a page run, as each draw
/// binds its buffers anew.
pub fn buffer_bound(b: &BufferObject) -> Bound {
    buffer_bytes_bound(b, 0..plain(|| b.raw.length()))
}

/// [`buffer_bytes_memory`]'s [`Bound`] of `b`'s bytes `bytes`, read as it reads them, without
/// collecting their ranges ([`buffer_bound`]).
fn buffer_bytes_bound(b: &BufferObject, bytes: Range<u64>) -> Bound {
    if let Some(s) = &b.surface {
        return surface_bound(s, None);
    }
    if let Some(h) = &b.heap {
        // As `buffer_bytes_memory`'s: its bytes at its place in its heap's.
        let at = plain(|| b.raw.gpu_address().0.saturating_sub(h.raw.gpu_address().0));
        return heap_bytes_bound(
            h,
            &(at.saturating_add(bytes.start)..at.saturating_add(bytes.end)),
        );
    }
    let kept = b.backing.kept();
    let paged = b.backing.meets_pages(&bytes);
    if paged.is_some() || b.backing.kept_contents().is_some() {
        return Bound::over(paged.unwrap_or(false), kept);
    }
    Bound::over(plain(|| b.raw.contents()).is_some(), kept)
}

/// [`heap_bytes_memory`]'s [`Bound`] of `h`'s bytes `bytes`, without collecting their ranges.
fn heap_bytes_bound(h: &HeapObject, bytes: &Range<u64>) -> Bound {
    Bound::over(
        h.backing.meets_pages(bytes).unwrap_or(false),
        h.backing.kept(),
    )
}

/// The [`Bound`] of an object made on `surface`, of the `refused` format where the runtime
/// refuses its writes: [`Memory::of_surface`]'s and [`Memory::of_surface_bytes`]'s, which always
/// hold the surface's own allocation ([`crate::os::surface_memory`]), so it meets memory.
fn surface_bound(surface: &OsSurface, refused: Option<PixelFormat>) -> Bound {
    Bound {
        meets: true,
        dropped: crate::os::gpu_read_only(surface),
        refused,
        kept: Kept::always(surface_kept(surface)),
    }
}

/// What a bind or a residency keeps of `t`'s memory: [`texture_memory`]'s [`Bound`], read as it
/// reads it, without collecting the ranges of its texels in the guest's pages, one a page run,
/// as each draw binds its textures anew.
pub fn texture_bound(t: &TextureObject) -> Bound {
    match (&t.surface, &t.under) {
        (Some(s), _) => {
            let refused = t.plane.as_ref().and_then(|_| {
                let pf = plain(|| t.raw.pixel_format());
                crate::gpu::ycbcr::writes_refused(pf).then_some(pf)
            });
            surface_bound(s, refused)
        }
        (None, Some(Under::Buffer(b, bytes))) => buffer_bytes_bound(b, bytes.clone()),
        (None, Some(Under::Heap(h, bytes))) => heap_bytes_bound(h, bytes),
        _ => Bound::default(),
    }
}

/// The host memory of `h`'s bytes `bytes`, a first byte and an end: in the guest's pages it was
/// made over or last replaced with; none for one of memory of its own.
pub(crate) fn heap_bytes_memory(h: &HeapObject, bytes: Range<u64>) -> Memory {
    Memory::over(
        h.backing.in_pages(&bytes).unwrap_or_default(),
        h.backing.kept(),
    )
}

/// The host memory all of `h`'s bytes lie in ([`heap_bytes_memory`]).
pub(crate) fn heap_memory(h: &HeapObject) -> Memory {
    heap_bytes_memory(h, 0..plain(|| h.raw.size()))
}

/// The slots of one stage's function that it may write, by its pipeline's reflection: each
/// texture or buffer argument it uses whose access is not read-only, as Metal's reflection
/// answers `access::write` or `access::read_write` for a texture and a `device` pointer not
/// `const` for a buffer (an array's slots, each of its elements'); and whether it reaches
/// textures through handles.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Slots {
    /// The texture slots it may write.
    pub textures: Vec<Range<u64>>,
    /// The buffer slots it may write.
    pub buffers: Vec<Range<u64>>,
    /// Whether a buffer it uses holds argument buffers (`elementIsArgumentBuffer`), whose
    /// handles reach any texture the work holds resident, one bound at any slot included.
    pub handles: bool,
}

impl Slots {
    /// The slots `bindings`, one stage's reflection, write.
    pub fn of(bindings: &[metal2vulkan::reflection::Binding]) -> Slots {
        let slots = |kind: BindingType| {
            bindings
                .iter()
                .filter(|b| b.binding_type() == kind && b.is_used())
                .filter(|b| b.access() != BindingAccess::ReadOnly)
                .map(|b| {
                    let first = u64::from(b.index());
                    let count = b
                        .texture()
                        .map_or(1, |t| u64::from(t.array_length().max(1)));
                    first..first + count
                })
                .collect()
        };
        let handles = (bindings.iter())
            .filter(|b| b.binding_type() == BindingType::Buffer && b.is_used())
            .filter_map(|b| b.buffer())
            .filter_map(|b| b.buffer_pointer_type())
            .any(|p| p.element_is_argument_buffer());
        Slots {
            textures: slots(BindingType::Texture),
            buffers: slots(BindingType::Buffer),
            handles,
        }
    }

    /// Whether the function writes texture slot `slot`.
    pub fn texture(&self, slot: u64) -> bool {
        self.textures.iter().any(|r| r.contains(&slot))
    }

    /// Whether the function writes buffer slot `slot`.
    pub fn buffer(&self, slot: u64) -> bool {
        self.buffers.iter().any(|r| r.contains(&slot))
    }
}

/// The vertex function's index in a render pipeline state's [`Reflection::slots`].
pub const VERTEX: usize = 0;
/// The fragment function's.
pub const FRAGMENT: usize = 1;
/// The tile function's.
pub const TILE: usize = 2;

/// The descriptor a pipeline state was made of, whose reflection tells its functions' [`Slots`].
#[derive(Debug)]
pub enum Reflected {
    /// A compute pipeline state's: its kernel's.
    Compute(Box<ComputePipelineDescriptor>),
    /// A render or tile pipeline state's: its vertex, fragment and tile functions', in that
    /// order ([`VERTEX`]).
    Render(RenderDescriptor),
}

/// The [`Slots`] of a pipeline state's functions, by its descriptor's reflection, read the first
/// time something asks. The state holds it, and so does an encoder that set the state, as
/// Metal's encoder holds the state it runs, the guest's state gone or not.
#[derive(Debug)]
pub struct Reflection {
    /// The descriptor; `None` for a state no creator of the guest's made.
    descriptor: Option<Reflected>,
    /// Each function's slots, read once.
    slots: std::sync::OnceLock<Result<Vec<Slots>, String>>,
}

impl Reflection {
    /// The reflection of a state made of `descriptor`.
    pub fn new(descriptor: Option<Reflected>) -> Arc<Reflection> {
        Arc::new(Reflection {
            descriptor,
            slots: std::sync::OnceLock::new(),
        })
    }

    /// The descriptor the state was made of; `None` for a state no creator of the guest's made.
    pub fn descriptor(&self) -> Option<&Reflected> {
        self.descriptor.as_ref()
    }

    /// Each function's [`Slots`]: a compute state's kernel's alone, a render state's vertex,
    /// fragment and tile functions' ([`VERTEX`]); `Err` naming why the reflection cannot tell,
    /// for a state of no guest descriptor's or a reflection the compiler refuses. A render
    /// state with rasterization off runs no fragment function, whose slots it writes none of,
    /// though its descriptor keeps the function (`crate::gpu::serialized`).
    pub fn slots(&self) -> Result<&[Slots], &str> {
        let read = self.slots.get_or_init(|| {
            let budget = metal2vulkan::Budget::standard();
            match &self.descriptor {
                None => Err("a state of no guest descriptor's".to_owned()),
                Some(Reflected::Compute(d)) => plain(|| d.reflection(&budget))
                    .map(|r| vec![Slots::of(r.bindings())])
                    .map_err(|e| e.to_string()),
                Some(Reflected::Render(d)) => {
                    let reflection = match d {
                        RenderDescriptor::Render(d) => plain(|| d.reflection(&budget)),
                        RenderDescriptor::Tile(d) => plain(|| d.reflection(&budget)),
                    };
                    let rasterizes = match d {
                        RenderDescriptor::Render(d) => d.rasterization_enabled,
                        RenderDescriptor::Tile(_) => true,
                    };
                    reflection
                        .map(|r| {
                            let fragment = match rasterizes {
                                true => Slots::of(r.fragment_bindings()),
                                false => Slots::default(),
                            };
                            vec![
                                Slots::of(r.vertex_bindings()),
                                fragment,
                                Slots::of(r.tile_bindings()),
                            ]
                        })
                        .map_err(|e| e.to_string())
                }
            }
        });
        read.as_deref().map_err(String::as_str)
    }
}

/// Whether work writes an object it binds or holds resident, where it may.
#[derive(Clone, Debug)]
pub enum Whether {
    /// It may, whatever its functions: a resource made resident with write usage, and a binding
    /// whose functions are not known.
    Writes,
    /// Where a function of a pipeline state writes the slot the object is bound at, by the
    /// state's reflection ([`Reflection::slots`]).
    Slot(Arc<Reflection>, Slot),
}

/// A slot of a function of a pipeline state, and whether the function writes it where the
/// state's reflection cannot tell ([`Whether::Slot`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    /// The function's index in [`Reflection::slots`].
    pub stage: usize,
    /// A texture slot, or a buffer slot.
    pub texture: bool,
    /// The slot.
    pub slot: u64,
    /// Whether the function writes it where the reflection cannot tell.
    pub untold: bool,
}

impl Whether {
    /// What work writes an object through, where it does ([`Whether::holds`]), for a death that
    /// names it. A buffer's slot is written where its function's argument there is a `device`
    /// pointer that is not `const`, which Metal's reflection answers read-write whether or not
    /// the function stores through it.
    pub(crate) fn through(&self) -> &'static str {
        match self {
            Whether::Writes => {
                "a resource made resident for writing, or a buffer bound for indirect commands \
                 that run pipeline states of their own, which no reflection here tells"
            }
            Whether::Slot(reflection, s) => match (reflection.slots().is_ok(), s.texture) {
                (false, _) => {
                    "a texture or buffer at a slot the function may write, which the pipeline \
                     state's reflection cannot tell"
                }
                (true, true) => {
                    "a texture at a slot the function writes (`access::write` or \
                     `access::read_write`)"
                }
                (true, false) => {
                    "a buffer at a slot whose argument the function declares a `device` pointer \
                     that is not `const`, read-write by Metal's reflection, whether or not it \
                     stores through it"
                }
            },
        }
    }

    /// Whether the work writes the object.
    pub(crate) fn holds(&self) -> bool {
        match self {
            Whether::Writes => true,
            Whether::Slot(reflection, s) => match reflection.slots() {
                Ok(slots) => slots.get(s.stage).is_some_and(|f| {
                    if s.texture {
                        f.texture(s.slot)
                    } else {
                        f.buffer(s.slot)
                    }
                }),
                Err(_) => s.untold,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::device::GpuDevice as _;
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _};
    use reims_vgpu_core::platform::values::TextureDesc;
    use reims_vgpu_core::raise::OnFault;

    use super::*;
    use crate::gpu::resource::tests::{guest_pages, heap_at, over_pages};

    /// What a bind keeps of a buffer's or a texture's memory, read without collecting ranges, is
    /// what the memory collected answers: for a buffer of the device's memory, one placed in a
    /// heap over the guest's pages (at its bytes and past the pages), one on a surface, textures
    /// placed in the heap, over a buffer, on surfaces over the guest's pages and over memory of
    /// their own, and one of memory of its own; and once the heap is detached, in no memory
    #[test]
    fn bound_memory_is_the_memory_collected() {
        let (p, d) = (
            crate::platform::tests::platform(),
            crate::gpu::device::tests::dev(),
        );
        let bgra = |options: u64| {
            p.texture_descriptor(&TextureDesc {
                pixel_format: 80,
                width: 16,
                height: 16,
                usage: 0x1,
                resource_options: Some(options),
                ..TextureDesc::default()
            })
        };
        let size = over_pages();
        let heap = heap_at(&p, &d, guest_pages());
        let options = heap.resource_options();
        let placed = heap
            .new_buffer_of_length(0x1000, options, 0x2000)
            .unwrap()
            .unwrap();
        let in_heap = heap.new_texture(&bgra(options), 0x4000).unwrap().unwrap();
        let own = d
            .new_buffer_of_length(OnFault::Die, 0x4000, 0)
            .unwrap()
            .unwrap();
        let over_own = own
            .new_texture(OnFault::Die, &bgra(0), 0, 0x40)
            .unwrap()
            .unwrap();
        let alone = d.new_texture(OnFault::Die, &bgra(0)).unwrap().unwrap();
        let surface = crate::os::tests::bgra_surface(16, 16);
        let (over, pages, length) = crate::os::tests::bgra_surface_over_at(16, 16, 0x40);
        let on_surface = (d.new_texture_on_surface(&bgra(0), Some(&surface), 0))
            .unwrap()
            .unwrap();
        let on_pages = (d.new_texture_on_surface(&bgra(0), Some(&over), 0))
            .unwrap()
            .unwrap();
        let surface_buffer = d.new_buffer_on_surface(Some(&surface)).unwrap().unwrap();
        let agree = |when: &str| {
            for (name, b) in [
                ("placed", &placed),
                ("own", &own),
                ("surface", &surface_buffer),
            ] {
                assert_eq!(buffer_bound(b), buffer_memory(b).bound(), "{when}: {name}");
                for bytes in [0..0x10, 0xff0..0x1010, size..size + 0x10, 0x10..0x10] {
                    assert_eq!(
                        buffer_bytes_bound(b, bytes.clone()),
                        buffer_bytes_memory(b, bytes.clone()).bound(),
                        "{when}: {name} {bytes:x?}"
                    );
                }
            }
            for (name, t) in [
                ("in the heap", &in_heap),
                ("over a buffer", &over_own),
                ("of its own", &alone),
                ("on a surface", &on_surface),
                ("on the guest's pages", &on_pages),
            ] {
                assert_eq!(
                    texture_bound(t),
                    texture_memory(t).bound(),
                    "{when}: {name}"
                );
            }
        };
        agree("attached");
        assert!(buffer_bound(&placed).meets && texture_bound(&in_heap).meets);
        heap.detach_backing().unwrap();
        agree("detached");
        assert!(!buffer_bound(&placed).meets && !texture_bound(&in_heap).meets);
        drop((on_pages, over));
        crate::os::tests::free(pages, length);
    }

    /// A texture over a surface's plane reaches the plane's rows where the CPU lays them out and
    /// where the GPU reads them, to whichever ends later: from the offset rounded down to 16, a
    /// GPU's row `bytesPerRow` rounded down to 16, or 16 where that is 0
    #[test]
    fn a_plane_texture_reaches_its_rows_where_the_cpu_and_the_gpu_lay_them() {
        // Four rows of 0x48 bytes from 0x4008: the CPU's end is the later.
        assert_eq!(super::plane_bytes(0x4008, 0x48, 4), 0x4000..0x4128);
        // Four rows of 8 bytes from 0x4008, 16 apart on the GPU: its end is the later.
        assert_eq!(super::plane_bytes(0x4008, 8, 4), 0x4000..0x4040);
    }
}
