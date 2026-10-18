//! Resources: the guest's buffers, heaps, indirect command buffers and textures, each backed by a
//! Metal object that is made, paged and released at fixed points.
//!
//! A [`Resource`](resource::Resource) is its validity flags ([`validity`]) and its kind
//! ([`ResourceKind`](resource::ResourceKind)). Which kind answers which operation is the
//! [`ResourceClass`](resource::ResourceClass) trait: its default methods are the answers a kind
//! gives when it does not override them, and an operation a kind does not answer raises
//! [`RaiseKind::Unrecognized`]. The kinds are [`buffer`], [`heap`], [`icb`] and [`texture`];
//! [`host`] is what a resource asks of its task.
//!
//! # Contract
//!
//! What callers and the guest observe, each held by the tests named (under each module's
//! `tests`):
//! - **Bind at first use.** A kind that reads guest pages asks the task for them only while it
//!   holds none (or is detached), once per life; only a discard, a failed plane and the last
//!   release drop them (`buffer::shared::prepare_with_cpu_rebinds_once_after_discard`,
//!   `heap::legacy::prepare::transfer_buffer_is_kept_until_discard`).
//! - **Validity is written after encoding.** A prepare sets host-valid after its copy is
//!   encoded, a synchronize guest-valid after; they mean "encoded", not "completed"
//!   (`buffer::private::prepare::binds_the_guest_pages_once_and_copies_up`).
//! - **Devolatilizing comes first** in a private buffer's and a legacy heap's prepare, and may
//!   clear host-valid before it is tested; a synchronize never devolatilizes
//!   (`buffer::private::discard_devolatilize::devolatilize_recopies_only_when_emptied`).
//! - **Placement write-back** (detach buffer and heap) happens before the nil check, only into a
//!   zero slot (`buffer::detach::init::write_back_only_into_a_zero_slot`).
//! - **Mapper surfaces**: retained before the buffer is made, released once per made buffer at
//!   a discard or the last release, and at once when Metal refuses the buffer
//!   (`buffer::mapper_ref::discard::discard_releases_the_surface_once`).
//! - **Which kind answers which operation**, including which raise
//!   [`RaiseKind::Unrecognized`]: `resource::responds_table`.
//! - **Values kept for good** ([`crate::raise::kept`]), one test per rule this area has: a Metal
//!   object whose resource index or GPU VA later placements are numbered after,
//!   `MetalNumbering` (the detach heap's cover buffer,
//!   `heap::detach::accessors::last_release_leaks_the_cover_buffer_and_frees_the_dummy`; what a
//!   creation made before a raise, `buffer::detach::init::a_detach_raise_leaks_the_buffer`); a
//!   surface retain nobody takes back, `SurfaceUseCount`
//!   (`buffer::mapper_ref::discard::a_gone_task_leaks_the_extra_retain`,
//!   `buffer::mapper_ref::prepare::a_make_raise_leaks_the_surface_retain`). What a creation made
//!   that holds nothing Metal, or that no path reaches, is released on its raise
//!   (`buffer::shared::tests::a_task_raise_propagates_out_of_the_creation`).
//!
//! Metal's objects are made in order and released at their last release: a kind holding two or more
//! handles takes them in a fixed order in its `Drop`.

pub mod buffer;
pub mod heap;
pub mod host;
pub mod icb;
pub mod manager;
pub mod resource;
pub mod texture;
pub mod validity;

use std::fmt;

use crate::raise::Raise;
use crate::raise::RaiseResult;
use crate::raise::death::DeathId;
use crate::raise::death::DeathSite;

/// An operation a resource kind may not answer. Asked of a kind that does not, it raises
/// [`RaiseKind::Unrecognized`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// Guest to host through a blit encoder.
    Prepare,
    /// Host to guest through a blit encoder.
    Synchronize,
    /// Guest to host through a compute encoder.
    PrepareCompute,
    /// Host to guest through a compute encoder.
    SynchronizeCompute,
    /// Drop the host copy's contents (opcode 63).
    Discard,
    /// The Metal object an encoder binds.
    GetResource,
    /// The Metal buffer.
    GetBuffer,
    /// The Metal texture.
    GetTexture,
    /// The Metal heap.
    GetHeap,
    /// The Metal indirect command buffer.
    GetIcb,
    /// Guest to host on the CPU.
    PrepareCpu,
    /// Host to guest on the CPU.
    SynchronizeCpu,
    /// Write back on the CPU before an eviction.
    EvictCpu,
    /// A backing's IOSurface.
    IoSurface,
    /// Prepare a shared texture backing on a command buffer of its own.
    PrepareSharedBacking,
    /// A new handle for a shared texture backing.
    NewSharedTextureHandle,
}

impl Op {
    /// The operation's name.
    pub const fn name(self) -> &'static str {
        match self {
            Op::Prepare => "prepare",
            Op::Synchronize => "synchronize",
            Op::PrepareCompute => "prepare_compute",
            Op::SynchronizeCompute => "synchronize_compute",
            Op::Discard => "discard",
            Op::GetResource => "get_resource",
            Op::GetBuffer => "get_buffer",
            Op::GetTexture => "get_texture",
            Op::GetHeap => "get_heap",
            Op::GetIcb => "get_icb",
            Op::PrepareCpu => "prepare_cpu",
            Op::SynchronizeCpu => "synchronize_cpu",
            Op::EvictCpu => "evict_cpu",
            Op::IoSurface => "io_surface",
            Op::PrepareSharedBacking => "prepare_shared_backing",
            Op::NewSharedTextureHandle => "new_shared_texture_handle",
        }
    }
}

/// What the resources area raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// A buffer getter on a buffer that has no handle (discarded, or never made).
    BufferInvalidated,
    /// The task gave no buffer over the guest pages (a torn-down task).
    NoGuestBuffer,
    /// A legacy heap's transfer buffer over the guest pages could not be made.
    NoTransferBuffer,
    /// A buffer or heap record shorter than the factory's minimum.
    DescriptorLength,
    /// Metal refused to give a detached buffer its guest pages back.
    BufferBackingRefused,
    /// Metal refused to give a detached heap its guest pages back.
    HeapBackingRefused,
    /// The mapper named no surface for a mapper-ref buffer (the detail is the mapping id).
    MapperSurfaceMissing,
    /// An operation the resource's kind does not answer.
    Unrecognized(Op),
    /// The texture kinds' ([`texture::RaiseKind`]).
    Texture(texture::RaiseKind),
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RaiseKind::BufferInvalidated => f.write_str("buffer.invalidated"),
            RaiseKind::NoGuestBuffer => f.write_str("buffer.no_guest_pages"),
            RaiseKind::NoTransferBuffer => f.write_str("heap.no_transfer_buffer"),
            RaiseKind::DescriptorLength => f.write_str("descriptor.length"),
            RaiseKind::BufferBackingRefused => f.write_str("buffer.backing_refused"),
            RaiseKind::HeapBackingRefused => f.write_str("heap.backing_refused"),
            RaiseKind::MapperSurfaceMissing => f.write_str("mapper.no_surface"),
            RaiseKind::Unrecognized(op) => write!(f, "unrecognized.{}", op.name()),
            RaiseKind::Texture(k) => fmt::Display::fmt(k, f),
        }
    }
}

impl From<RaiseKind> for crate::raise::RaiseId {
    fn from(k: RaiseKind) -> Self {
        crate::raise::RaiseId::Resources(k)
    }
}

/// How the resources area dies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// A resource of a torn-down task reads the task's features (a NULL features pointer
    /// dereference).
    GoneTaskFeatures,
    /// A NULL placement slot is read or written.
    NullPlacement,
    /// The task detaches backings but cannot back a buffer with address ranges (an assert).
    DetachBufferWithoutRanges,
    /// The task detaches backings but cannot back a heap with address ranges (an assert).
    DetachHeapWithoutRanges,
    /// The texture kinds' ([`texture::DeathKind`]).
    Texture(texture::DeathKind),
}

impl DeathKind {
    /// Its dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::GoneTaskFeatures => "resources.task.features_gone",
            DeathKind::NullPlacement => "resources.placement.null",
            DeathKind::DetachBufferWithoutRanges => "resources.buffer.detach_without_ranges",
            DeathKind::DetachHeapWithoutRanges => "resources.heap.detach_without_ranges",
            DeathKind::Texture(k) => k.id(),
        }
    }

    /// The site `testkit::capture` records and the stderr line names: the kind of death, and the
    /// dotted id.
    pub const fn site(self) -> DeathSite {
        let kind = match self {
            DeathKind::GoneTaskFeatures | DeathKind::NullPlacement => {
                crate::raise::death::DeathKind::NullDeref
            }
            DeathKind::DetachBufferWithoutRanges | DeathKind::DetachHeapWithoutRanges => {
                crate::raise::death::DeathKind::Assert
            }
            DeathKind::Texture(k) => k.kind(),
        };
        DeathSite {
            kind,
            file: self.id(),
            line: 0,
        }
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id().trim_start_matches("resources."))
    }
}

impl From<DeathKind> for DeathId {
    fn from(k: DeathKind) -> Self {
        DeathId::Resources(k)
    }
}

impl From<DeathKind> for DeathSite {
    fn from(k: DeathKind) -> Self {
        k.site()
    }
}

/// Raises `kind` here: with no handler live the process dies at this call, and with one the raise
/// goes to it through the raise channel, which this area's callers answer.
#[track_caller]
pub(crate) fn raise<T>(kind: RaiseKind) -> RaiseResult<T> {
    Err(Raise::new(kind))
}

/// [`raise`] with a detail (diagnostics only).
#[track_caller]
pub(crate) fn raise_with<T>(kind: RaiseKind, detail: impl fmt::Display) -> RaiseResult<T> {
    Err(Raise::new(kind).with_detail(detail))
}

/// The area's raise a `Raise` carries, if it carries one (tests).
#[cfg(test)]
pub(crate) fn raised_kind(t: &Raise) -> Option<RaiseKind> {
    match t.id() {
        crate::raise::RaiseId::Resources(k) => Some(k),
        _ => None,
    }
}

/// How long a retired object outlives its last use by core ([`retire`]).
const RETIRE_GRACE: std::time::Duration = std::time::Duration::from_secs(3);

/// Objects core let go of whose last use may still be in flight on the GPU, each with when it
/// was retired ([`retire`]).
static RETIRED: crate::sync::Mutex<
    std::collections::VecDeque<(std::time::Instant, Box<dyn std::any::Any + Send>)>,
> = crate::sync::Mutex::new(std::collections::VecDeque::new());

/// Lets `object` go [`RETIRE_GRACE`] from now, not at once (vphone-linux). A transfer buffer core
/// drops at a discard may still be the destination of a paging copy a submitted command buffer
/// has not run: Metal's command buffer keeps it alive, the Vulkan runtime's does not where the
/// copy reaches it through a texture over its bytes, and its pages freed under the copy fault the
/// device (NVIDIA Xid 31, a copy engine write). Older retired objects are dropped here, outside
/// the queue's lock.
pub(crate) fn retire<T: Send + 'static>(object: Option<T>) {
    let Some(object) = object else { return };
    // The unit tests check the releases core makes, at once.
    if cfg!(test) {
        drop(object);
        return;
    }
    // Host time, not the `Clock`: the grace is a margin past the GPU's work, which no test
    // observes (they release at once, above).
    #[allow(clippy::disallowed_methods)]
    let now = std::time::Instant::now();
    let expired: Vec<_> = {
        let mut queue = RETIRED.lock();
        queue.push_back((now, Box::new(object)));
        let mut out = Vec::new();
        while queue.front().is_some_and(|(at, _)| now.duration_since(*at) >= RETIRE_GRACE) {
            out.extend(queue.pop_front());
        }
        out
    };
    drop(expired);
}
