//! `id<MTLIndirectCommandBuffer>` and its commands: core's `IndirectCommandBuffer`,
//! `IndirectRenderCommand` and `IndirectComputeCommand`, over the runtime's.
//!
//! - **The commands** are the runtime's: each setter writes the command's record, and each
//!   getter reads it back typed (`IndirectCommandBuffer::commands`, the runtime's stand-in for
//!   Metal's private getters), as the records stand now, whoever wrote them.
//! - **What Metal answers that a record does not hold** is kept beside the commands
//!   ([`CommandSide`]), as Metal's getters answer it on this host without the API validation
//!   layer (measured, macOS 26.7, M3 Max):
//!   - a buffer slot core set whose record names no buffer ([`SetSlot`]): one past the slots the
//!     record holds (32 of a stage), whose buffer Metal keeps for its getters alone. The address
//!     getter answers the buffer's address plus the offset's low 32 bits, or those bits alone for
//!     nil, and the stride getter all ones after a plain set, the stride after a stride form; a
//!     slot core never set, or one a reset cleared, answers 0 and 0. A nil buffer at a slot the
//!     record holds the record names, as Metal keeps it (the offset's
//!     low 32 bits as its address, and the stride): an execution the GPU replays binds those bits
//!     as the address Metal's function sees, and one replayed on the host binds none and
//!     announces `indirect-nil-buffer-offset`, which dies ([`crate::answer::DIES`]). Its getters
//!     still answer what is kept, which a blit moves as below ([`named`]);
//!   - a threadgroup memory length set past the 31 a record holds, which Metal keeps for its
//!     getter alone;
//!   - the pipeline state core set on each command, by its numbers, whose `uniqueIdentifier`
//!     the getter answers. The numbers, not the handle: the guest's release of the state lets it
//!     go, as Metal's indirect command keeps no state alive ([`SetPipeline`]).
//!     A record the GPU wrote (a kernel's, or a blit's copy) names a state no setter gave the
//!     command, by its resource ID: the getter answers the identifier of the state this platform
//!     made with that ID ([`crate::gpu::unique_id`]), as Metal's answers that state's own
//!     (measured: a kernel's set of a compute and of a render state read from an argument
//!     buffer). A record names its state by the state's resource ID, which the runtime reads
//!     whether or not the state still lives (`pipeline_id`), and one naming none answers 0, as
//!     Metal's does. Metal answers the identifier of the state the record was written with, a
//!     released one included, though a state made later took its ID (measured), and the record
//!     does not say which state of its ID it was written with. So the getter answers the state
//!     that held the ID when the record can have been written ([`named_state`]): core's set,
//!     and each state that held the ID during a span since core's set in which the GPU may have
//!     written the buffer's records ([`Window`]: from a dispatch of a kernel that can encode
//!     indirect commands, or any execution of an indirect command buffer, in a compute encoder
//!     that made the buffer resident for writing, or from a render encoder's making it so, to the
//!     end of its command buffer; a blit's copy carries the source's spans). Where those name one
//!     state, it answers that state's identifier;
//!     where they name two or more, or a state no longer kept
//!     ([`crate::gpu::unique_id::HISTORY`]), it dies. Core's set keeps the state's numbers
//!     ([`CommandSide`]), not the state, and the runtime's command names its state without
//!     holding it, as Metal's does: the guest's release gives the ID back, and a state made later
//!     takes it, on Metal with the released one's identifier too, which the command then answers
//!     (measured). Where a later state may hold that identifier, the getter dies
//!     ([`IcbObject::pipeline_identifier`]).
//! - **A blit's copy and reset of commands** move and reset what is kept beside them once the
//!   blit's command buffer completed, as Metal's GPU has run the blit by then. Until it commits,
//!   the GPU ran none of it: a getter answers what stands, and a setter comes before the blit.
//!   While it may run, Metal's answer depends on whether the GPU ran the blit yet, and a getter
//!   of what a blit writes dies, as does a setter of what one copies or writes ([`Blit`]). Once
//!   its command buffer failed, which may or may not have run the blit, each value the blit would
//!   have changed is unknown, and its getter dies until core sets it again or resets the command,
//!   or a blit that completed resets it or copies a known one over it ([`Unknown`]).
//! - **A blit's copy past the destination's end** lands the commands that fit, as Metal's does,
//!   and takes every command before the index as written over ([`IcbObject::overrun`]): Metal
//!   writes over some of those records in its own layout, by a shape this platform does not
//!   follow, and no other buffer changes. A getter, an execution or a copy of one dies until a
//!   reset or a copy writes its whole record again, and so does a `useResource` form that makes
//!   the buffer resident, as a function of the encoder may copy any of its commands
//!   (`copy_command`), where this platform's copy would carry the intact record
//!   ([`IcbObject::resident`]); a copy past the end in a shape nothing measured dies at the
//!   copy (`encode_blit::overrun_measured`).
//! - **Getters Metal reads raw** answer the values Metal's do: a nil patch or control-point index
//!   buffer is Metal's address `0x1eeee1111` ([`NIL_BUFFER_ADDRESS`]), a base vertex is its low 32
//!   bits, zero-extended, a buffer's offset is its low 32 bits ([`kept_offset`]), a buffer set
//!   without a stride has the stride all ones, every stride of a buffer without dynamic attribute
//!   strides is all ones, and a patch draw's factor scale is 1. Metal reads past the descriptor's
//!   buffers and lengths, which core never asks for; they answer 0 here. Metal's arguments getters
//!   read a record of any kind as their own; core reads each only for a command of its kind, and
//!   another kind's dies here ([`another_kind`]).
//! - **An indexed patch draw of no control-point index buffer** is a patch draw, as Metal records
//!   it (measured: `getCommandType` answers `DrawPatches` and `drawPatchesArguments` its
//!   arguments, whatever the descriptor's types); a nil index or tessellation factor buffer,
//!   which Metal crashes at, dies.
//! - **Numbers the guest sees** are Metal's ([`Numbers`]): the `gpuResourceID`, which is also
//!   the `resourceIndex`, from a space of IDs of Metal's indirect command buffers, and the
//!   `uniqueIdentifier`, Metal's address of the buffer in its range of the GPU's address space,
//!   of a buffer made before the process's first encoder. Of one made after an encoder of any
//!   kind, Metal's address depends on what the encoders took of the range, which this platform
//!   does not follow, and the read dies ([`Space`], [`encoder_made`]).
//!   The runtime's `gpu_resource_id` is the ID a translated kernel encoding through an argument
//!   buffer reaches a buffer's records by. It follows Metal's pool, placement and hold, as these
//!   numbers do ([`Space`]): a placed buffer takes the same ID in the runtime, and each
//!   reservation Metal takes reaches the runtime, but one holding an ID of [`RECORDS_ID_LIMIT`]
//!   (2^23) or more, which the runtime would refuse, or whose buffers it would refuse
//!   ([`reserve`]). So it is the guest's, but after such a reservation, which parts them
//!   ([`IcbObject::reaches_its_records`]): a kernel encoding through the guest's ID of such a
//!   buffer would not reach its records, so a direct dispatch in a compute encoder that made one
//!   resident for writing dies ([`crate::gpu::encode_compute`]). A buffer is made, its ID given
//!   back and the runtime's buffer dropped, under the lock of [`Space`], and a reservation reaches
//!   the runtime under it too, so the runtime takes and gives back its IDs in the order these
//!   are. One window stays: a buffer released while a command buffer that executes it or makes it
//!   resident is pending gives its ID back here in the command buffer's completed handler
//!   ([`crate::gpu::command`]), and in the runtime just after, as the runtime lets go of what the
//!   commands used once its completed handlers ran; a buffer made between the two, on another
//!   thread, may take that ID here and another in the runtime, which parts them as above.
//! - **Refusals** die at the selector, naming the runtime's error: Metal without its validation
//!   layer raises at none of them, and goes on at those measured (an index at or past the size,
//!   a command of the other kind, and a Private buffer's command each answer a command object).

use std::collections::{BTreeMap, BTreeSet};
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use metal2vulkan::bind::indirect_commands::{BufferBinding, RECORDS_ID_LIMIT};
use metal2vulkan::metal::{IndexType, IndirectCommandType, PrimitiveType};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::Truncate;
use reims_vgpu_core::platform::Handle;
use reims_vgpu_core::platform::blit::ResourceRef;
use reims_vgpu_core::platform::icb::{
    DispatchThreadgroupsArgs, DispatchThreadsArgs, DrawArgs, DrawIndexedArgs,
    DrawIndexedPatchesArgs, DrawPatchesArgs, IcbComputeCmd, IcbRenderCmd, ImageBlockArgs,
    IndirectCommandBuffer, IndirectComputeCommand, IndirectRenderCommand, TessellationFactorArgs,
};
use reims_vgpu_core::platform::resource::{ComputePipeline as _, RenderPipeline as _};
use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
use reims_vgpu_core::platform::values::IcbDesc;
use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};
use reims_vgpu_core::sync::{Mutex, MutexGuard};

use crate::answer::{
    Policy, Unfollowed, get, hang, metal_aborts, plain, send, unfollowed, unrepresentable,
};
use crate::gpu::command::{Progress, Stage};
use crate::gpu::device::VulkanDevice;
use crate::gpu::metal_value;
use crate::gpu::resource::{Named, VulkanBuffer, VulkanComputePipeline, VulkanRenderPipeline};
use crate::platform::{Obj, VulkanPlatform};

/// The address Metal's patch and control-point index getters answer for a nil buffer (measured:
/// a patch draw of a nil patch index buffer, and an indexed one of nil buffers, on a fresh
/// command and on one that held a buffer).
pub const NIL_BUFFER_ADDRESS: u64 = 0x1_eeee_1111;

// ---- Metal's numbers.

/// The base of Metal's `uniqueIdentifier` of an indirect command buffer: the identifier is this
/// plus the buffer's address in Metal's range for them (measured: a fresh process's first buffer
/// answers `0x1000018000` with its private `gpuAddress` `0x18000`).
const UNIQUE_IDENTIFIER_BASE: u64 = 0x10_0000_0000;

/// Where the first buffer of a process lies in that range (measured: so too after textures,
/// buffers and pipeline states with indirect command buffer support; what MoltenVK makes in a
/// process moves Metal's own, which this platform's numbers do not follow).
const FIRST_ADDRESS: u64 = 0x1_8000;

/// Metal's page: a buffer's range holds one more past its `allocatedSize`.
const PAGE: u64 = 0x4000;

/// The alignment of a buffer's range, which holds its pages and one page more.
const SLOT: u64 = 0x8000;

/// The first resource ID a placement cannot take: Metal writes past its table of IDs at 16384 or
/// more, which ends its process at random (measured: 16383 always made, 16384 to 50367 made or
/// crashed from run to run, 100000 crashed).
const ID_LIMIT: u64 = 16384;

/// What Metal numbers one indirect command buffer by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Numbers {
    /// `gpuResourceID`, which `resourceIndex` answers too.
    id: u64,
    /// Whether plain allocation handed the ID out (a placed one does not count against it).
    plain: bool,
    /// Its address in Metal's range for these buffers.
    address: u64,
}

/// The size of Metal's first pool of indirect command buffer IDs ([`Pool`]): plain allocation
/// hands out every ID below it before the pool doubles (measured, in the tests on both
/// platforms: with 1 to 25 handed out and 2 to 9 given back, the next ID is 26, from the free
/// range of 26 to 31, the smaller, where a pool of 64 would hand out 2; and 10 reserved alone
/// leaves the order from 1, so the pool holds at least 20).
const FIRST_POOL: u64 = 32;

/// Metal's numbers of the live indirect command buffers of the process (one device's, as the
/// host has one). Isolated tasks share them, as every task's device is the one object of the
/// process's device ([`crate::gpu::device::adopt`]), and as every task's work is Metal's one
/// device's in a process (the suite's `interplay_decodewalk_44` records, on the M3, an isolated
/// task's two buffers at `0x68000` and `0x70000` of the range, after another task's two at
/// `0x18000` and `0x20000` and that task's encoders).
///
/// Measured on this host: plain allocation hands out IDs from Metal's pool ([`Pool`]); a
/// placement (`setResourceIndex:` of a nonzero index) takes the index's low 32 bits whoever
/// holds them, reserved or not, and takes no ID from the pool. Addresses are the lowest gap of
/// the range, from [`FIRST_ADDRESS`], that holds the buffer's `allocatedSize` and one more page,
/// aligned to [`SLOT`]. The runtime's `allocated_size` is Metal's (compared with Metal on this
/// host for 8160 descriptors: 17 sets of command types, none and render beside compute among
/// them, each inheriting its pipeline state and buffers or not, with dynamic strides or not, of
/// 0 to 31 buffers a stage and 0 to 3000 commands).
///
/// A reservation (`reserveResourceIndicesForResourceType:indices:indexCount:` of type 5) keeps
/// its IDs from plain allocation for good. Only the process's first call with an ID counts, and
/// only before a buffer was made, placed or not: Metal ignores any other, whatever it holds,
/// saying "reserve() must be called before any allocation takes place". A call holding an ID of
/// 2^32 or more is dropped whole, and a later call is the reservation. 0 is no ID: beside others
/// the call is theirs (0 and 20 hand out what 20 alone does: 21 to 31, 1 to 19, then 32 on), and
/// alone it reserves nothing (plain IDs from 1). Of the calls Metal takes, two kinds go
/// otherwise ([`Reservation`]):
/// - one holding an ID of 2^31 to 2^32 - 1 never returns: Metal spins in
///   `AGX::Mempool<32u,0u,true,0u,0u,unsigned long long>::reserve` at full CPU (measured at 2^31,
///   2^31 + 5 and 2^32 - 1, the longest run 5 minutes, and at 2^31 + 5 after a call of 0 alone),
///   where 2^31 - 1 is taken and hands out 1 next. While it spins it holds its pool's lock
///   (measured from a second thread, 5 s a case, the waits sampled): another buffer's creation,
///   placed or not, waits on it in `AGX::Mempool<32>::RangeAllocation`, one placed at 16384 or
///   100000 too, past the table of IDs Metal ends its process at ([`ID_LIMIT`]), and another
///   reservation of type 5 in `AGX::Mempool<32>::reserve`. A device texture and a command queue
///   wait too where they are the process's first, in `-[AGXG15XFamilyDevice setupDeferred]`, which
///   core's command queues, made with the device, run before any reservation; buffers, textures
///   over a buffer, heaps and their textures, sampler states, command buffers' creation, commit and
///   wait, and texture reservations go on. No buffer lives while it spins: one made first makes
///   Metal ignore the reservation. The thread parks for good holding the lock of [`Space`]
///   ([`reserve_in`], [`crate::answer::hang`]), as a hang stays a hang, so every later creation and
///   reservation of type 5, on any thread, an isolated task's among them, waits on it for good,
///   before any death of its own ([`new_indirect_command_buffer`]). After a buffer was made Metal
///   ignores such a call, as any other;
/// - one after a call of 0 alone is taken, and Metal then hands out IDs twice (measured, in order
///   of making: 0 then 20, 21 to 31, 1 to 19, then 1 to 10; 0 then 10, 1 to 9, 11 to 31, then 1
///   to 15; 0 then 100, 101 to 127, 1 to 31, then 1 to 82; 0 then 2^31 - 1, 1 to 31, then 1 to
///   31 and 32 on). [`Pool`] does not follow Metal's past such a call, so it dies at the
///   selector.
///
/// A buffer released is not free while a command buffer that uses it may run (measured, a
/// command buffer blocked on an event, then completed): a compute or render encoder's execution
/// or `useResource` of it keeps the buffer, its ID and range, until the command buffer completes,
/// or goes uncommitted; a blit's copy (from it or into it) or reset keeps only its range, its ID
/// free at once; a blit's optimization keeps neither.
///
/// Metal's own work takes ranges here too, by the GPU, the encoder's kind, its work and whether
/// its command buffer was committed. Where the next buffer lands after the process's first
/// encoder, with two live buffers ending at `0x28000` (measured, the runtime's `API-icbenc1`
/// rows; on the M3 Max with Metal's validation layer and without alike):
///
/// | the first encoder | M3 Max | M4 |
/// |---|---|---|
/// | compute: empty, making resources resident alone, or never committed | `0x28000` (nothing taken) | `0x28000` |
/// | compute: a plain dispatch | `0x30000` (one slot) | `0x30000` |
/// | compute: executing a buffer | `0x38000` | one slot past that buffer |
/// | blit: empty | `0x28000` | `0x68000` (`0x40000` taken) |
/// | blit: copying buffers | `0x70000` | `0x70000` |
/// | render: never committed | `0x88000` (`0x60000` taken) | `0x88000` |
/// | render: empty, committed | `0x4f0000` | `0x2c0000` |
///
/// Earlier M3 Max rows, whose live buffers they do not state: after a blit that copies or resets
/// indirect commands, at `0x4d0000`; after one that fills a buffer, at `0x70000`; after a render
/// encoder that executes a buffer, at `0x508000`.
///
/// These numbers do not follow that work. So a buffer made after an encoder of any kind was made
/// in the process, counted from the encoder's creation whether or not it is committed
/// ([`encoder_made`]), answers no `uniqueIdentifier`: the read dies, named
/// ([`Unfollowed::IcbIdentifierAfterEncoder`]), and the buffer's creation and every other number
/// of it go on. A buffer made before the first encoder answers Metal's. The death is wider than
/// Metal's difference where the encoder took nothing: a compute encoder that neither dispatches
/// nor executes a buffer, on both GPUs, and an empty blit encoder on the M3 Max.
struct Space {
    /// The plain IDs, once a buffer was made or a reservation taken.
    pool: Option<Pool>,
    /// The ranges of the live buffers: start, then bytes.
    ranges: BTreeMap<u64, u64>,
}

static SPACE: Mutex<Space> = Mutex::new(Space {
    pool: None,
    ranges: BTreeMap::new(),
});

/// Whether an encoder of any kind was made in the process ([`encoder_made`]).
static ENCODED: AtomicBool = AtomicBool::new(false);

/// Says an encoder of any kind is being made, before the runtime's encoder is, whether or not it
/// is committed later or the creator answers nil: each encoder creator core sends calls it first
/// (`blitCommandEncoder`, `computeCommandEncoder`, `computeCommandEncoderWithDescriptor:` and
/// `renderCommandEncoderWithDescriptor:`). An indirect command buffer made after it answers no
/// `uniqueIdentifier` ([`Space`]). The buffer's creation reads it after the runtime made the
/// buffer, so a buffer that reads it unset was made before every encoder whose creation had not
/// set it.
pub(crate) fn encoder_made() {
    if !ENCODED.load(Ordering::Relaxed) {
        ENCODED.store(true, Ordering::SeqCst);
    }
}

impl Space {
    /// The pool of plain IDs, made at its first use.
    fn pool(&mut self) -> &mut Pool {
        self.pool.get_or_insert_with(|| Pool::new(FIRST_POOL))
    }

    /// The numbers of a new buffer of `size` bytes, its `allocatedSize`, placed at `placed` when
    /// given.
    fn take(&mut self, placed: Option<u64>, size: u64) -> Numbers {
        let (id, plain) = match placed {
            Some(id) => {
                self.pool().started = true;
                (id, false)
            }
            None => (self.pool().take(), true),
        };
        let footprint = size.saturating_add(PAGE).next_multiple_of(SLOT);
        let mut address = FIRST_ADDRESS;
        for (&start, &bytes) in &self.ranges {
            if start >= address.saturating_add(footprint) {
                break;
            }
            address = address.max(start.saturating_add(bytes));
        }
        self.ranges.insert(address, footprint);
        Numbers { id, plain, address }
    }

    /// Frees the ID of a buffer released.
    fn give_back(&mut self, numbers: &Numbers) {
        if numbers.plain {
            self.pool().give(numbers.id);
        }
    }

    /// Keeps `ids` from plain allocation, where Metal takes the reservation ([`Pool::reserve`]):
    /// what Metal does with it. A call holding an ID of 2^32 or more is dropped whole.
    fn reserve(&mut self, ids: &[u64]) -> Reservation {
        if ids.iter().any(|&id| id > u64::from(u32::MAX)) {
            return Reservation::Ignored;
        }
        self.pool().reserve(ids)
    }
}

/// What Metal does with a reservation of indirect command buffer IDs ([`Space`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reservation {
    /// Its IDs are kept from plain allocation for good.
    Taken,
    /// Nothing: Metal ignores or drops it, or it holds no ID but 0.
    Ignored,
    /// Metal spins in its pool for good: an ID of 2^31 or more in a call it takes.
    Hangs,
    /// Metal takes it, after a call of 0 alone, then hands out IDs twice.
    HandsOutTwice,
}

/// Metal's plain allocation of indirect command buffer IDs (measured on this host, each case in
/// the tests on both platforms): free ranges inside a pool of [`FIRST_POOL`] IDs whose size
/// doubles. Each ID handed out is the lowest of the smallest free range, the lower of two alike,
/// and where none is free the pool doubles, its new IDs one free range. An ID given back joins
/// the free ranges either side of it, so with 2, 3 and 5 of 1 to 8 given back, the next are 5,
/// 2, 3, then 9. A reservation doubles the pool until it holds each reserved ID, the new IDs
/// joining the free range that ends at the old size, and splits the free range holding the ID:
/// 100 alone starts the IDs at 101, where the range above it is the smaller, and 2 and 16382
/// hand out 1, 16383, then 3.
#[derive(Debug)]
struct Pool {
    /// Every ID below it is free, handed out or reserved.
    size: u64,
    /// The free ranges, start to end (exclusive), none touching another.
    free: BTreeMap<u64, u64>,
    /// The same ranges as length and start: the first is the one an ID is handed out from.
    smallest: BTreeSet<(u64, u64)>,
    /// Whether a buffer was made, placed or not, or a reservation taken: Metal ignores a
    /// reservation then.
    started: bool,
    /// Whether a call of 0 alone came before that: Metal takes a later reservation, then hands
    /// out IDs twice.
    lone_zero: bool,
}

impl Pool {
    /// A pool of `size` IDs, every one above 0 free.
    fn new(size: u64) -> Pool {
        let mut pool = Pool {
            size,
            free: BTreeMap::new(),
            smallest: BTreeSet::new(),
            started: false,
            lone_zero: false,
        };
        pool.add(1, size);
        pool
    }

    /// `[start, end)` free, unless empty.
    fn add(&mut self, start: u64, end: u64) {
        if start < end {
            self.free.insert(start, end);
            self.smallest.insert((end - start, start));
        }
    }

    /// The free range at `start` taken out: its end.
    fn cut(&mut self, start: u64) -> Option<u64> {
        let end = self.free.remove(&start)?;
        self.smallest.remove(&(end - start, start));
        Some(end)
    }

    /// The pool doubled, its new IDs joining the free range that ends where it ended.
    fn grow(&mut self) {
        let old = self.size;
        self.size = old.saturating_mul(2);
        let start = match self.free.range(..old).next_back() {
            Some((&start, &end)) if end == old => {
                self.cut(start);
                start
            }
            _ => old,
        };
        self.add(start, self.size);
    }

    /// The next ID, now handed out.
    fn take(&mut self) -> u64 {
        self.started = true;
        if self.smallest.is_empty() {
            self.grow();
        }
        let start = self.smallest.first().map_or(self.size, |&(_, start)| start);
        if let Some(end) = self.cut(start) {
            self.add(start + 1, end);
        }
        start
    }

    /// `id`, handed out, free again.
    fn give(&mut self, id: u64) {
        let below = (self.free.range(..id).next_back())
            .filter(|&(_, &end)| end == id)
            .map(|(&start, _)| start);
        let start = match below {
            Some(start) => {
                self.cut(start);
                start
            }
            None => id,
        };
        let end = self.cut(id + 1).unwrap_or(id + 1);
        self.add(start, end);
    }

    /// Keeps `ids`, each below 2^32, from plain allocation for good, unless the pool started:
    /// what Metal does with them ([`Space`]). 0 is no ID: a call of no other ID leaves the pool
    /// as it was, and one of 0 alone marks it. One Metal hangs at or hands out IDs twice after
    /// leaves the pool as it was too.
    fn reserve(&mut self, ids: &[u64]) -> Reservation {
        if self.started {
            return Reservation::Ignored;
        }
        let zero = ids.contains(&0);
        let ids: Vec<u64> = ids.iter().copied().filter(|&id| id != 0).collect();
        if ids.is_empty() {
            self.lone_zero |= zero;
            return Reservation::Ignored;
        }
        if ids.iter().any(|&id| id >= 1 << 31) {
            return Reservation::Hangs;
        }
        if self.lone_zero {
            return Reservation::HandsOutTwice;
        }
        self.started = true;
        for id in ids {
            while id >= self.size {
                self.grow();
            }
            if let Some((&start, &end)) = self.free.range(..=id).next_back()
                && id < end
            {
                self.cut(start);
                self.add(start, id);
                self.add(id + 1, end);
            }
        }
        Reservation::Taken
    }
}

/// `reserveResourceIndicesForResourceType:indices:indexCount:` (`sel`) of indirect command
/// buffers (type 5) on `device`: `ids` kept from plain allocation where Metal takes the
/// reservation, else nothing, as Metal ignores or drops it; the thread parks for good where Metal
/// spins, and the process dies where Metal goes on to hand out IDs twice ([`Space`]). A
/// reservation Metal takes reaches the runtime too, so its IDs stay Metal's, unless it holds an
/// ID of [`RECORDS_ID_LIMIT`] (2^23) or more: the runtime refuses one of 2^30 or more, which
/// Metal takes to 2^31 - 1, and a buffer whose own ID is 2^23 or more, which plain allocation
/// may hand out past such a reservation. The runtime's numbers part from Metal's then (the module
/// docs). Its refusal of one it is given dies.
pub(crate) fn reserve(device: &VulkanDevice, sel: &'static str, ids: &[u64]) -> RaiseResult<()> {
    // The space stays held until the runtime has the reservation too, so no creation comes
    // between the two ([`IcbObject::reaches_its_records`]).
    let (reservation, _space) = plain(|| reserve_in(&SPACE, ids));
    match reservation {
        Reservation::Ignored => Ok(()),
        Reservation::Taken if ids.iter().any(|&id| id >= RECORDS_ID_LIMIT) => Ok(()),
        Reservation::Taken => {
            let ids: Vec<rt::ResourceId> = ids.iter().map(|&id| rt::ResourceId(id)).collect();
            send(sel, Policy::Dies, || {
                device
                    .raw
                    .reserve_resource_ids(rt::ResourceIdKind::IndirectCommandBuffer, &ids)
            })
        }
        // `reserve_in` parked this thread before it could answer this.
        Reservation::Hangs => hang(),
        Reservation::HandsOutTwice => fatal_with_detail(
            DeathId::Platform {
                kind: DeathKind::Abort,
                call: sel,
            },
            "Metal takes a reservation after one of 0 alone, then hands out IDs twice",
        ),
    }
}

/// What Metal does with `ids` reserved in `space` ([`Space::reserve`]), and `space`, still held.
/// Where Metal spins, this thread parks for good holding `space`'s lock, as Metal's spinning
/// thread holds its pool's: a later take of an ID or reservation waits on it for good.
fn reserve_in<'a>(space: &'a Mutex<Space>, ids: &[u64]) -> (Reservation, MutexGuard<'a, Space>) {
    let mut held = space.lock();
    let reservation = held.reserve(ids);
    if reservation == Reservation::Hangs {
        let _held = held;
        hang();
    }
    (reservation, held)
}

/// A buffer's range in [`Space`], free again once the buffer and every blit of its commands that
/// may still run let go of it.
#[derive(Debug)]
pub(crate) struct Reserved {
    address: u64,
}

impl Drop for Reserved {
    fn drop(&mut self) {
        SPACE.lock().ranges.remove(&self.address);
    }
}

// ---- The buffer.

/// What Metal answers of one command that its record does not hold (the module docs).
#[derive(Debug, Default)]
struct CommandSide {
    /// What the getters answer of each buffer slot core set, by stage and index, where its record
    /// names no buffer.
    slots: BTreeMap<(Slots, u32), SetSlot>,
    /// Each threadgroup memory length core set, by index, which the getter answers past the
    /// lengths the record holds.
    lengths: BTreeMap<u32, u32>,
    /// The pipeline state core set on the command.
    pipeline: Option<SetPipeline>,
    /// The point of [`crate::gpu::unique_id::now`]'s order at which core set it.
    pipeline_at: u64,
    /// What of the above a blit of a failed command buffer may or may not have changed.
    unknown: Unknown,
    /// Whether Metal's copy of commands past the buffer's end may have written over the record
    /// in its own layout ([`IcbObject::overrun`]): a getter, an execution or a copy of the command
    /// dies until a reset or a copy writes the whole record again.
    corrupted: bool,
}

impl CommandSide {
    /// A copy.
    fn copy(&self) -> CommandSide {
        CommandSide {
            slots: self.slots.clone(),
            lengths: self.lengths.clone(),
            pipeline: self.pipeline,
            pipeline_at: self.pipeline_at,
            unknown: self.unknown.clone(),
            corrupted: self.corrupted,
        }
    }

    /// What the getters may answer apart for this command and `other`: what either does not
    /// know, and each value they keep differently, one core never set counting as a fresh
    /// command's (0, and no pipeline state).
    fn differs(&self, other: &CommandSide) -> Unknown {
        let mut unknown = self.unknown.clone();
        unknown.join(&other.unknown);
        for key in self.slots.keys().chain(other.slots.keys()) {
            let slot = |side: &CommandSide| side.slots.get(key).copied().unwrap_or_default();
            if slot(self) != slot(other) {
                unknown.slots.insert(*key);
            }
        }
        for key in self.lengths.keys().chain(other.lengths.keys()) {
            let length = |side: &CommandSide| side.lengths.get(key).copied().unwrap_or(0);
            if length(self) != length(other) {
                unknown.lengths.insert(*key);
            }
        }
        unknown.pipeline |= match (&self.pipeline, &other.pipeline) {
            (None, None) => false,
            (Some(a), Some(b)) => a != b,
            _ => true,
        };
        unknown
    }
}

/// What is kept of a command, by value, that a blit of a failed command buffer may or may not
/// have changed: Metal's answer depends on whether its GPU ran the blit before the command buffer
/// failed, which nothing here tells (the runtime's records answer as its GPU left them, and
/// Metal's failure is not measured, which only a GPU fault brings). Its getter dies until core
/// sets it again or resets the command, or a blit that completed resets it or copies a known one
/// over it; one that copies an unknown one over it carries what is unknown.
#[derive(Debug, Clone, Default)]
struct Unknown {
    slots: BTreeSet<(Slots, u32)>,
    lengths: BTreeSet<u32>,
    pipeline: bool,
}

impl Unknown {
    /// `other`'s are unknown too.
    fn join(&mut self, other: &Unknown) {
        self.slots.extend(other.slots.iter().copied());
        self.lengths.extend(other.lengths.iter().copied());
        self.pipeline |= other.pipeline;
    }

    /// Whether `kept` is unknown.
    fn holds(&self, kept: Kept) -> bool {
        match kept {
            Kept::Slot(slots, at) => self.slots.contains(&(slots, at)),
            Kept::Length(at) => self.lengths.contains(&at),
            Kept::Pipeline => self.pipeline,
            Kept::All => !self.is_empty(),
        }
    }

    /// `kept` is known again: core set it.
    fn settle(&mut self, kept: Kept) {
        match kept {
            Kept::Slot(slots, at) => {
                self.slots.remove(&(slots, at));
            }
            Kept::Length(at) => {
                self.lengths.remove(&at);
            }
            Kept::Pipeline => self.pipeline = false,
            Kept::All => *self = Unknown::default(),
        }
    }

    /// Whether nothing is unknown.
    fn is_empty(&self) -> bool {
        self.slots.is_empty() && self.lengths.is_empty() && !self.pipeline
    }
}

/// What a getter reads of what is kept of a command, or a setter writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kept {
    /// A buffer slot of a stage.
    Slot(Slots, u32),
    /// A threadgroup memory length, by index.
    Length(u32),
    /// The pipeline state.
    Pipeline,
    /// All of it: a reset of the command.
    All,
}

/// A command's buffer slots of one stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Slots {
    Vertex,
    Fragment,
    Kernel,
}

/// What Metal's getters answer of a buffer slot core set (measured: vertex, fragment and kernel
/// buffers and nil at offsets of 0, 0x30, 0x1234_5678_9abc and 2^64 - 8, plain and with a
/// stride, on buffers with and without dynamic attribute strides, and buffers set at slots 30 to
/// 63 of a descriptor of 64).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SetSlot {
    /// The address: the buffer's plus the offset's low 32 bits, or those bits alone for nil.
    address: u64,
    /// The attribute stride: all ones after a plain set, the stride after a stride form.
    stride: u64,
}

/// A buffer offset as Metal keeps it: its low 32 bits, which its address getters add to the
/// buffer's address (measured: an offset of 2^40 answers the buffer's address), and which the
/// runtime's setters are sent, so the record, and what the GPU reads of it, holds that address.
const fn kept_offset(offset: u64) -> u64 {
    offset & 0xffff_ffff
}

/// A pipeline state core set on a command, by its numbers: its runtime state's resource ID and
/// its `uniqueIdentifier`, one per creation, so two are equal only for one state. Keeping no
/// handle lets the guest's release of the state drop this platform's object of it, as Metal's
/// indirect command keeps no state alive, so the release is noted
/// ([`crate::gpu::unique_id::released`]), which decides which state a record names once a
/// state made after it took its ID. The runtime's command names the runtime's state without
/// holding it too, as Metal's does, so the resource ID goes back at the release: a state made
/// after the guest released one a command still names takes its ID, as on Metal (measured
/// through core on the M3 Max).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SetPipeline {
    /// The runtime state's resource ID.
    id: u64,
    /// Its `uniqueIdentifier`.
    identifier: u64,
}

impl SetPipeline {
    /// `state`'s numbers.
    fn compute(state: &VulkanComputePipeline) -> SetPipeline {
        SetPipeline {
            id: state.raw.gpu_resource_id().0,
            identifier: state.unique_identifier(),
        }
    }

    /// `state`'s numbers.
    fn render(state: &VulkanRenderPipeline) -> SetPipeline {
        SetPipeline {
            id: state.raw.gpu_resource_id().0,
            identifier: state.unique_identifier(),
        }
    }

    /// Its resource ID and its `uniqueIdentifier`.
    fn numbers(self) -> (u64, u64) {
        (self.id, self.identifier)
    }
}

/// What an indirect command buffer's commands take of the encoder that executes them: the
/// textures always, as no command sets one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inherits {
    /// The pipeline state (`inheritPipelineState`).
    pub pipeline: bool,
    /// The buffers (`inheritBuffers`).
    pub buffers: bool,
}

/// An indirect command buffer: the runtime's, with Metal's numbers and what Metal answers of its
/// commands that their records do not hold.
#[derive(Debug)]
pub struct IcbObject {
    /// The runtime's buffer, dropped as its ID here is given back ([`IcbObject::drop`]).
    pub raw: ManuallyDrop<rt::IndirectCommandBuffer>,
    /// Metal's numbers of it.
    numbers: Numbers,
    /// Whether it was made after an encoder ([`encoder_made`]): its `uniqueIdentifier` dies.
    after_encoder: bool,
    /// Its range in [`Space`].
    range: Arc<Reserved>,
    /// Whether its commands take attribute strides.
    strides: bool,
    /// What its commands take of the encoder that executes them.
    pub(crate) inherits: Inherits,
    /// What Metal answers of its commands that their records do not hold.
    sides: Arc<Mutex<Sides>>,
    /// Which shape of those Metal's copies past a buffer's end were measured at it is, if any
    /// ([`OverrunShape::of`]).
    pub(crate) overrun_shape: Option<OverrunShape>,
}

/// A buffer of a shape Metal's copies of commands past a buffer's end were measured at, on this
/// host (`encode_blit::overrun_measured`, which also takes only a source and a destination of
/// one count of commands, as every measurement's were), made with options 0 (Shared storage),
/// of one of two descriptors with nothing else set, each of 2 buffers a stage and dynamic
/// attribute strides (two runs a shape, buffers of 2 to 64 commands, 1 to 63 commands past the
/// end). Where Metal's copy past the end writes depends on its layout of the descriptor's
/// records, which nothing here models, so a buffer of any other descriptor is of none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OverrunShape {
    /// Both dispatch kinds (`CONCURRENT_DISPATCH | CONCURRENT_DISPATCH_THREADS`: threadgroups
    /// and threads, both concurrent), 2 kernel buffers and 2 threadgroup memory lengths; the
    /// buffers measured held thread dispatches alone.
    Compute,
    /// The four draw kinds, 2 vertex and 2 fragment buffers.
    Render,
}

impl OverrunShape {
    /// The fewest and the most commands of a buffer of [`OverrunShape::Compute`] or
    /// [`OverrunShape::Render`] measured.
    const COMMANDS: std::ops::RangeInclusive<u64> = 2..=64;

    /// The shape of a buffer of `desc`, `max_count` commands and raw `options`, where it is one
    /// that was measured.
    fn of(desc: &IcbDesc, max_count: u64, options: u64) -> Option<OverrunShape> {
        let compute = IcbDesc {
            command_types: 0x60,
            max_kernel_buffer_bind_count: 2,
            max_kernel_threadgroup_memory_bind_count: 2,
            support_dynamic_attribute_stride: true,
            ..IcbDesc::default()
        };
        let render = IcbDesc {
            command_types: 0xf,
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 2,
            support_dynamic_attribute_stride: true,
            ..IcbDesc::default()
        };
        if options != 0 {
            return None;
        }
        match desc {
            d if *d == compute && Self::COMMANDS.contains(&max_count) => {
                Some(OverrunShape::Compute)
            }
            d if *d == render && Self::COMMANDS.contains(&max_count) => Some(OverrunShape::Render),
            _ => None,
        }
    }
}

/// What Metal answers of a buffer's commands that their records do not hold, and the blits of
/// them that may still change it.
#[derive(Debug, Default)]
struct Sides {
    /// What Metal answers of each command core reached, by index.
    commands: BTreeMap<u32, CommandSide>,
    /// The blits that copy or reset commands of the buffer, until they can change no answer.
    blits: Vec<Blit>,
    /// The progress of each command buffer that executes the buffer or makes it resident, once
    /// each, while the GPU may run it ([`IcbObject::executed`], [`IcbObject::resident`]).
    executions: Vec<std::sync::Weak<Progress>>,
    /// The spans in which the GPU may have written its records, oldest first.
    writes: Vec<Window>,
    /// The registry ID of the runtime's device the buffer was made on, whose states its records
    /// name.
    device: u64,
    /// Whether a command was ever taken as written over by Metal's copy past the buffer's end
    /// ([`CommandSide::corrupted`]): until one is, no command is, and an execution, a residency
    /// or a copy asks none ([`Sides::overrun`]).
    overran: bool,
}

/// A span of [`crate::gpu::unique_id::now`]'s order in which the GPU may write a buffer's records:
/// from the encoding of a command that may write them to its command buffer's end, as it
/// completed, failed or went ([`IcbObject::may_be_written`]). A record the GPU wrote in it names
/// a state that held its resource ID during it.
#[derive(Debug, Clone)]
struct Window {
    /// Where it starts.
    start: u64,
    /// Where it ends: [`OPEN`] until its command buffer ended ([`Closes`]).
    end: Arc<AtomicU64>,
    /// Its command buffer's progress, by address, which tells a second command of that command
    /// buffer from one of another while it is open; 0 for spans joined into one.
    of: usize,
}

/// The end of a [`Window`] whose command buffer has not ended.
const OPEN: u64 = u64::MAX;

/// How many [`Window`]s a buffer keeps apart: past them, two that ended are joined into one span
/// covering both where that names the same states for every record ([`Sides::write`]).
const WINDOWS: usize = 64;

impl Window {
    /// Its start and its end.
    fn span(&self) -> (u64, u64) {
        (self.start, self.end.load(Ordering::SeqCst))
    }
}

/// Ends a [`Window`] as its command buffer's [`Progress`] lets go of what it holds: once the
/// command buffer completed, failed or went.
struct Closes(Arc<AtomicU64>);

impl Drop for Closes {
    fn drop(&mut self) {
        self.0.store(crate::gpu::unique_id::now(), Ordering::SeqCst);
    }
}

/// The `uniqueIdentifier` of the state a record names by resource ID `id`, the module docs'
/// rule: of `own`, the numbers (resource ID, identifier) of the state core set on the command
/// and the point it set it at; `writes`, the spans in which the GPU may have written the
/// buffer's records ([`Window::span`]); and `history`, the states made with `id` on the record's
/// device, each holding it from its creation to its release. Where that names no state, the only
/// state ever made with `id`; else `Err` names why it cannot tell.
fn named_state(
    own: Option<((u64, u64), u64)>,
    writes: &[(u64, u64)],
    id: u64,
    history: Option<&crate::gpu::unique_id::History>,
) -> Result<u64, &'static str> {
    let mut states = BTreeSet::new();
    // Core's set is one candidate, and each span that ends after it adds the states that held the
    // ID during it; with no set, every span counts.
    let set_at = own.map_or(0, |(_, at)| at);
    if let Some(((own_id, identifier), _)) = own
        && own_id == id
    {
        states.insert(identifier);
    }
    if let Some(history) = history {
        for &(start, end) in writes.iter().filter(|&&(_, end)| end > set_at) {
            if start < history.lost_until {
                return Err(
                    "the pipeline state a record the GPU may have written names, by a resource ID \
                     whose states that held it while it may have been written are no longer kept",
                );
            }
            // Each state that held the ID during the span, from its creation to its release.
            for holder in &history.holders {
                if holder.made < end && start < holder.gone {
                    states.insert(holder.identifier);
                }
            }
        }
    }
    match (states.len(), history) {
        (1, _) => Ok(states.into_iter().next().unwrap_or_default()),
        (0, Some(history)) if history.lost_until == 0 && history.holders.len() == 1 => {
            Ok(history.holders[0].identifier)
        }
        (0, None) => Err("a pipeline state this platform did not make"),
        (0, Some(_)) => Err(
            "the pipeline state a record names, by a resource ID several states held, where \
             neither core's set nor a span in which the GPU may have written the record tells \
             which",
        ),
        _ => Err(
            "the pipeline state a record the GPU may have written names, by a resource ID two or \
             more states held while it may have been written, whose identifiers Metal answers \
             apart",
        ),
    }
}

/// Whether two ended spans `a` and `b` of a buffer of the device of registry ID `device`, `a`
/// starting first, name the same states for every record as the span covering both
/// ([`named_state`]): no state of the device was made and released between them, which only the
/// covering span would overlap, and no command's set (`sets`, core's points) lies between their
/// ends, which counts a span only where it ends after the set.
fn joinable(
    device: u64,
    (_, a_end): (u64, u64),
    (b_start, b_end): (u64, u64),
    sets: &[u64],
) -> bool {
    let (low, high) = (a_end.min(b_end), a_end.max(b_end));
    if sets.iter().any(|&at| low <= at && at < high) {
        return false;
    }
    b_start <= a_end || !plain(|| crate::gpu::unique_id::held_within(device, a_end, b_start))
}

/// Whether the state of `identifier` among `history`'s was released before a state of
/// indirect command buffer support was made on the device of registry ID `device`, so that Metal
/// may answer its identifier for that state ([`crate::gpu::unique_id`]); one `history` no longer
/// keeps counts as released whenever a state was made since the first kept.
fn superseded(
    device: u64,
    identifier: u64,
    history: Option<&crate::gpu::unique_id::History>,
) -> bool {
    use crate::gpu::unique_id::{LIVE, made_since};
    let Some(history) = history else {
        return false;
    };
    let gone = (history.holders.iter())
        .find(|holder| holder.identifier == identifier)
        .map_or(
            history
                .lost_until
                .min(history.holders.front().map_or(0, |h| h.made)),
            |h| h.gone,
        );
    gone != LIVE && made_since(device, gone)
}

/// The death of a getter, an execution or a copy of a command whose record Metal's copy past the
/// buffer's end may have written over ([`IcbObject::overrun`]).
const OVERRUN: &str = "a command whose record Metal's copy of commands past the indirect command \
                       buffer's end may have written over in its own layout, which this \
                       platform's copy does not";

/// What a `useResource` form that makes such a buffer resident dies naming
/// ([`IcbObject::resident`]).
const RESIDENT_OVERRUN: &str = "an indirect command buffer made resident for the encoder's \
                                functions, which may copy a command (copy_command) whose record \
                                Metal's copy of commands past the buffer's end may have written \
                                over in its own layout, where this platform's copy carries the \
                                intact record";

/// A blit that copies commands from a buffer, into one or resets them, kept by that buffer's
/// [`Sides`] while its command buffer may run it: it moves what Metal answers of them once its
/// command buffer completed, and leaves what it may have moved unknown once it failed
/// ([`Unknown`]).
#[derive(Debug)]
struct Blit {
    /// Its command buffer's progress.
    progress: Arc<Progress>,
    /// The commands it reaches.
    commands: Range<u64>,
    /// Whether it writes them (a copy's destination, a reset), not only reads them.
    writes: bool,
    /// Whether it may write over them in Metal's own layout: a copy past the buffer's end
    /// ([`IcbObject::overrun`]).
    corrupts: bool,
}

/// Whether a getter reads what is kept of a command, or a setter writes it, and what.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    Read(Kept),
    Write(Kept),
}

impl Blit {
    /// Whether it can still change an answer: its command buffer may run it. Once that failed,
    /// what it may have changed is unknown ([`Unknown`]), and it no longer counts.
    fn matters(&self) -> bool {
        matches!(self.progress.stage(), Stage::Encoding | Stage::Committed)
    }

    /// What Metal's answer to `access` of command `index` depends on, if it depends on this
    /// blit: whether the GPU ran it yet, while its command buffer may run. Before the commit the
    /// GPU ran none of it: a read answers what stands, and a write comes before it.
    fn races(&self, access: Access, index: u32) -> Option<&'static str> {
        if !self.commands.contains(&u64::from(index)) {
            return None;
        }
        match (self.progress.stage(), access, self.writes) {
            (Stage::Committed, Access::Read(_), true) => Some(
                "a read of a command a blit of a committed command buffer may be writing, whose \
                 answer on Metal depends on when the GPU runs it",
            ),
            (Stage::Committed, Access::Write(_), _) => Some(
                "a write of a command a blit of a committed command buffer may be reaching, whose \
                 answer on Metal depends on when the GPU runs it",
            ),
            _ => None,
        }
    }
}

impl Sides {
    /// Keeps `window`; one whose end is already kept, a copy's, is kept once. Past [`WINDOWS`],
    /// two spans that ended and are next to each other by their starts are joined into the span
    /// covering both, where that names the same states for every record ([`joinable`]); where no
    /// two can be, they stay apart. What keeps two apart is a state made and released between
    /// them or core's set of a command between their ends, so few stay apart.
    fn write(&mut self, window: Window) {
        if self.writes.iter().any(|w| Arc::ptr_eq(&w.end, &window.end)) {
            return;
        }
        self.writes.push(window);
        if self.writes.len() <= WINDOWS {
            return;
        }
        let mut ended: Vec<(u64, u64, usize)> = (self.writes.iter().enumerate())
            .map(|(i, w)| {
                let (start, end) = w.span();
                (start, end, i)
            })
            .filter(|&(_, end, _)| end != OPEN)
            .collect();
        ended.sort_unstable();
        let sets: Vec<u64> = (self.commands.values())
            .filter(|side| side.pipeline.is_some())
            .map(|side| side.pipeline_at)
            .collect();
        let pair = ended.windows(2).find(|pair| {
            let [(a_start, a_end, _), (b_start, b_end, _)] = [pair[0], pair[1]];
            joinable(self.device, (a_start, a_end), (b_start, b_end), &sets)
        });
        if let Some(&[(a_start, a_end, a), (_, b_end, b)]) = pair {
            self.writes[a] = Window {
                start: a_start,
                end: Arc::new(AtomicU64::new(a_end.max(b_end))),
                of: 0,
            };
            self.writes.remove(b);
        }
    }

    /// Copies of what is kept of commands `from`.
    fn copies(&self, from: Range<u64>) -> Vec<(u64, Option<CommandSide>)> {
        from.map(|i| {
            let side = u32::try_from(i).ok().and_then(|i| self.commands.get(&i));
            (i, side.map(CommandSide::copy))
        })
        .collect()
    }

    /// Commands from `to` are `moved`, copies of commands from `from`.
    fn take(&mut self, from: u64, to: u64, moved: Vec<(u64, Option<CommandSide>)>) {
        for (i, side) in moved {
            let Ok(at) = u32::try_from(to + (i - from)) else {
                continue;
            };
            match side {
                Some(side) => {
                    self.overran |= side.corrupted;
                    self.commands.insert(at, side)
                }
                None => self.commands.remove(&at),
            };
        }
    }

    /// Commands from `to` may or may not be `moved`, copies of commands from `from`: a blit of a
    /// failed command buffer may have copied them. What the two would answer apart is unknown.
    fn unsettle(&mut self, from: u64, to: u64, moved: Vec<(u64, Option<CommandSide>)>) {
        for (i, side) in moved {
            if let Ok(at) = u32::try_from(to + (i - from)) {
                self.may_be(at, side.as_ref());
            }
        }
    }

    /// Command `at` may or may not be `side` (none for a fresh command's): what the two would
    /// answer apart is unknown.
    fn may_be(&mut self, at: u32, side: Option<&CommandSide>) {
        let fresh = CommandSide::default();
        let unknown = self
            .commands
            .get(&at)
            .unwrap_or(&fresh)
            .differs(side.unwrap_or(&fresh));
        let corrupted = side.is_some_and(|s| s.corrupted);
        self.overran |= corrupted;
        if !unknown.is_empty() || corrupted || self.commands.contains_key(&at) {
            let kept = self.commands.entry(at).or_default();
            kept.unknown = unknown;
            kept.corrupted |= corrupted;
        }
    }

    /// Whether Metal's copy past the buffer's end may have written over a record of `range` (all
    /// of them, `None`), or may yet: one that completed or failed, or one of a command buffer the
    /// GPU may still run ([`IcbObject::overrun`]).
    fn overrun(&mut self, range: Option<&Range<u64>>) -> bool {
        self.blits.retain(Blit::matters);
        let reaches = |commands: &Range<u64>| {
            range.is_none_or(|r| r.start < commands.end && commands.start < r.end)
        };
        let marked = self.overran
            && (self.commands.iter())
                .any(|(&i, side)| side.corrupted && reaches(&(u64::from(i)..u64::from(i) + 1)));
        marked || (self.blits.iter()).any(|b| b.corrupts && reaches(&b.commands))
    }
}

impl Drop for IcbObject {
    /// Its ID is free again, as Metal's is once the buffer is released, and its range once no
    /// blit that may still run holds it. The ID is given back and the runtime's buffer dropped
    /// under the lock of [`Space`], which a creation holds while the runtime makes its buffer, so
    /// no creation takes an ID between the two ([`IcbObject::reaches_its_records`]).
    fn drop(&mut self) {
        let mut space = SPACE.lock();
        space.give_back(&self.numbers);
        // SAFETY: `raw` is dropped once, here, as `self` goes, and never reached again.
        unsafe { ManuallyDrop::drop(&mut self.raw) };
        drop(space);
    }
}

impl std::ops::Deref for IcbObject {
    type Target = rt::IndirectCommandBuffer;
    fn deref(&self) -> &rt::IndirectCommandBuffer {
        &self.raw
    }
}

impl IcbObject {
    /// Whether a kernel encoding commands through the guest's ID of the buffer reaches its
    /// records: whether the runtime's ID of it, which the records' address follows, is Metal's
    /// (the module docs).
    pub(crate) fn reaches_its_records(&self) -> bool {
        plain(|| self.raw.gpu_resource_id().0 == self.numbers.id)
    }

    /// Its range, which a blit of its commands holds until its command buffer has run (the docs
    /// of [`Space`]).
    pub(crate) fn range(&self) -> Arc<Reserved> {
        Arc::clone(&self.range)
    }

    /// Runs `edit`, `sel`'s `access`, on what Metal answers of command `index`; a write leaves
    /// what it writes known. Where Metal's answer depends on whether the GPU ran a blit of the
    /// command yet ([`Blit::races`]), or a read's on whether it ran one of a failed command
    /// buffer ([`Unknown`]), it dies instead.
    fn side<T>(
        &self,
        sel: &'static str,
        access: Access,
        index: u32,
        edit: impl FnOnce(&mut CommandSide) -> T,
    ) -> T {
        let edited = plain(|| {
            let mut sides = self.sides.lock();
            sides.blits.retain(Blit::matters);
            if let Some(race) = sides.blits.iter().find_map(|b| b.races(access, index)) {
                return Err(race);
            }
            let side = sides.commands.entry(index).or_default();
            match access {
                Access::Read(_) if side.corrupted => Err(OVERRUN),
                Access::Read(kept) if side.unknown.holds(kept) => Err(
                    "a read of what a blit of a failed command buffer may or may not have \
                     changed, whose answer on Metal depends on whether the GPU ran it before the \
                     command buffer failed",
                ),
                Access::Read(_) => Ok(edit(side)),
                Access::Write(kept) => {
                    let edited = edit(side);
                    side.unknown.settle(kept);
                    Ok(edited)
                }
            }
        });
        edited.unwrap_or_else(|why| {
            fatal_with_detail(
                DeathId::Platform {
                    kind: DeathKind::Abort,
                    call: sel,
                },
                why,
            )
        })
    }

    /// What is kept of slot `at` of `slots` of command `index`, which `sel` reads: a fresh
    /// slot's 0 and 0 where core set none.
    fn set_slot(&self, sel: &'static str, index: u32, slots: Slots, at: u64) -> SetSlot {
        let Ok(at) = u32::try_from(at) else {
            return SetSlot::default();
        };
        self.side(sel, Access::Read(Kept::Slot(slots, at)), index, |s| {
            s.slots.get(&(slots, at)).copied().unwrap_or_default()
        })
    }

    /// The address getter `sel`'s answer for slot `at` of `slots` of command `index`, whose
    /// record holds `buffers`: the buffer's address plus its offset where the record names one,
    /// else what is kept of the slot.
    fn buffer_address(
        &self,
        sel: &'static str,
        index: u32,
        (slots, buffers): (Slots, &[Option<rt::IndirectBufferBinding>]),
        at: u64,
    ) -> u64 {
        match named(buffers, at) {
            Some(b) => b.address.0,
            None => self.set_slot(sel, index, slots, at).address,
        }
    }

    /// The stride getter `sel`'s answer for slot `at` of `slots` of command `index`, whose
    /// record holds `buffers`: all ones on a buffer without dynamic attribute strides, as Metal
    /// answers every stride there; else the stride of the buffer the record names, all ones for
    /// the pipeline's own, or what is kept of the slot.
    fn attribute_stride(
        &self,
        sel: &'static str,
        index: u32,
        (slots, buffers): (Slots, &[Option<rt::IndirectBufferBinding>]),
        at: u64,
    ) -> u64 {
        if !self.strides {
            return u64::MAX;
        }
        match named(buffers, at) {
            Some(b) => b.attribute_stride.unwrap_or(u64::MAX),
            None => self.set_slot(sel, index, slots, at).stride,
        }
    }

    /// Slot `at` of `slots` of command `index` was set by `sel`, the getters answering `set`
    /// where its record names no buffer.
    fn keep_slot(&self, sel: &'static str, index: u32, slots: Slots, at: u32, set: SetSlot) {
        self.side(sel, Access::Write(Kept::Slot(slots, at)), index, |s| {
            s.slots.insert((slots, at), set)
        });
    }

    /// The `uniqueIdentifier` of the pipeline state a record names by `id`, made on the device
    /// of registry ID `device`, by core's set on command `index`, the spans in which the GPU may
    /// have written the buffer's records and the states made with `id` ([`named_state`]); where
    /// those do not tell which state, it dies.
    ///
    /// It dies too for a state released before a state of indirect command buffer support was
    /// made on its device ([`superseded`]): Metal may have given that one the released state's
    /// identifier, by which core's map of identifiers to the guest's references, the newest
    /// registration winning, answers the newer state, where this platform's identifier, never
    /// handed out twice, would answer the released one's reference.
    fn pipeline_identifier(&self, sel: &'static str, index: u32, (device, id): (u64, u64)) -> u64 {
        let own = self.side(sel, Access::Read(Kept::Pipeline), index, |s| {
            s.pipeline.map(|p| (p.numbers(), s.pipeline_at))
        });
        let writes: Vec<_> = plain(|| self.sides.lock().writes.iter().map(Window::span).collect());
        let history = plain(|| crate::gpu::unique_id::holders(device, id));
        let identifier = named_state(own, &writes, id, history.as_ref())
            .unwrap_or_else(|why| unrepresentable(sel, why));
        if plain(|| superseded(device, identifier, history.as_ref())) {
            unrepresentable(
                sel,
                "the pipeline state a record names, released before a state that supports \
                 indirect command buffers was made, which Metal may have given its identifier, so \
                 that the guest's reference by it is the newer state's",
            );
        }
        identifier
    }

    /// The `uniqueIdentifier` getter `sel`'s answer for command `index`, whose record names no
    /// live pipeline state: by `named`, the resource ID it names (the runtime's `pipeline_id`), a
    /// released state's identifier ([`IcbObject::pipeline_identifier`]), as Metal's getter
    /// answers it; 0 where it names none, as Metal's.
    fn released_pipeline(
        &self,
        sel: &'static str,
        index: u32,
        named: Option<rt::ResourceId>,
    ) -> u64 {
        named.map_or(0, |id| {
            let device = self.raw.device().registry_id();
            self.pipeline_identifier(sel, index, (device, id.0))
        })
    }

    /// A command of the command buffer of `progress` (none once core let go of it, when it runs
    /// nothing) may write the buffer's records: a span from now to the command buffer's end
    /// ([`Window`]), one for each command buffer.
    pub(crate) fn may_be_written(&self, progress: Option<Arc<Progress>>) {
        let Some(progress) = progress else {
            return;
        };
        let of = Arc::as_ptr(&progress).addr();
        let opened = plain(|| {
            let mut sides = self.sides.lock();
            let open = |w: &Window| w.of == of && w.end.load(Ordering::SeqCst) == OPEN;
            if sides.writes.iter().any(open) {
                return None;
            }
            let end = Arc::new(AtomicU64::new(OPEN));
            sides.write(Window {
                start: crate::gpu::unique_id::now(),
                end: Arc::clone(&end),
                of,
            });
            Some(end)
        });
        if let Some(end) = opened {
            progress.hold(Closes(end));
        }
    }

    /// A blit of the command buffer of `progress` (none once it went) copies commands `from` over
    /// the commands from `to` of `destination`: the GPU writes those, and what Metal answers of
    /// them moves with them once it completed, or is unknown where it would move once it failed.
    pub(crate) fn copied(
        &self,
        progress: Option<Arc<Progress>>,
        from: Range<u64>,
        destination: &IcbObject,
        to: u64,
    ) {
        let Some(progress) = &progress else {
            return;
        };
        let reached = to..to + (from.end - from.start);
        self.blit(progress, from.clone(), false);
        destination.blit(progress, reached, true);
        type Land = fn(&mut Sides, u64, u64, Vec<(u64, Option<CommandSide>)>);
        let copy = |land: Land| {
            let (source, destination) = (Arc::clone(&self.sides), Arc::clone(&destination.sides));
            let from = from.clone();
            move || {
                let (moved, writes) = {
                    let source = source.lock();
                    (source.copies(from.clone()), source.writes.clone())
                };
                let mut destination = destination.lock();
                land(&mut destination, from.start, to, moved);
                // A copied record may name a state the GPU wrote it with in the source.
                for window in writes {
                    destination.write(window);
                }
            }
        };
        progress.at_completion(copy(Sides::take));
        progress.at_failure(copy(Sides::unsettle));
    }

    /// A blit of the command buffer of `progress` (none once it went) resets commands `range`: the
    /// GPU writes them, and what Metal answers of them is a fresh command's once it completed, or
    /// unknown where that differs once it failed.
    pub(crate) fn reset(&self, progress: Option<Arc<Progress>>, range: Range<u64>) {
        let Some(progress) = &progress else {
            return;
        };
        self.blit(progress, range.clone(), true);
        let (sides, failed) = (Arc::clone(&self.sides), Arc::clone(&self.sides));
        let reached = range.clone();
        progress.at_completion(move || {
            let mut sides = sides.lock();
            for i in range {
                if let Ok(i) = u32::try_from(i) {
                    sides.commands.remove(&i);
                }
            }
        });
        progress.at_failure(move || {
            let mut sides = failed.lock();
            for i in reached {
                if let Ok(i) = u32::try_from(i) {
                    sides.may_be(i, None);
                }
            }
        });
    }

    /// A blit `sel` of the command buffer of `progress` (none once it went) copies commands into
    /// this buffer past its end, the commands that fit landing from `range`'s end on: Metal's
    /// writes over the records before them, `range`, in its own layout (measured on this host,
    /// two runs a shape, compute and render buffers of 2 to 64 commands, 1 to 63 commands past the
    /// end: some of those records turn into a dispatch of another kind, at shapes where others
    /// stay whole, and no other buffer changes). Every command of `range` is taken as overwritten
    /// once the command buffer completed or failed: a getter of one dies, as do an execution and
    /// a copy that reach one, until a reset or a copy writes its whole record again. While the
    /// GPU may run the blit, a getter of one races it and dies, and so does a setter of what is
    /// kept of it beside its record (`side`): its pipeline state, a buffer and its attribute
    /// stride, a threadgroup memory length, and a reset; the setters of its record alone (a
    /// dispatch or a draw, `setBarrier` and the like) check nothing. An execution or a copy that
    /// reaches one dies too, which Metal may run before or after it. An execution of the buffer,
    /// or a residency ([`IcbObject::resident`]), that another command buffer holds while the GPU
    /// may run it, which may run after the blit, dies here.
    pub(crate) fn overrun(
        &self,
        sel: &'static str,
        progress: Option<Arc<Progress>>,
        range: Range<u64>,
    ) {
        let Some(progress) = &progress else {
            return;
        };
        let elsewhere = plain(|| {
            let mut sides = self.sides.lock();
            sides
                .executions
                .retain(|p| p.upgrade().is_some_and(|p| p.running()));
            let elsewhere = (sides.executions.iter())
                .any(|p| p.upgrade().is_some_and(|p| !Arc::ptr_eq(&p, progress)));
            sides.blits.retain(Blit::matters);
            sides.blits.push(Blit {
                progress: Arc::clone(progress),
                commands: range.clone(),
                writes: true,
                corrupts: true,
            });
            elsewhere
        });
        if elsewhere {
            unrepresentable(
                sel,
                "a copy of commands past an indirect command buffer's end, which Metal writes \
                 over its other commands' records with, while another command buffer that may run \
                 after it executes the buffer or makes it resident",
            );
        }
        let mark = |sides: Arc<Mutex<Sides>>, range: Range<u64>| {
            move || {
                let mut sides = sides.lock();
                sides.overran = true;
                for i in range {
                    if let Ok(i) = u32::try_from(i) {
                        sides.commands.entry(i).or_default().corrupted = true;
                    }
                }
            }
        };
        progress.at_completion(mark(Arc::clone(&self.sides), range.clone()));
        progress.at_failure(mark(Arc::clone(&self.sides), range));
    }

    /// Dies at `sel`, which reads command `index`'s record, where Metal's copy past the buffer's
    /// end may have written over it ([`IcbObject::overrun`]): one that completed or failed, or
    /// one of a committed command buffer the GPU may be running. Before the commit the GPU ran
    /// none of it, and a read answers what stands.
    fn intact(&self, sel: &'static str, index: u32) {
        let at = u64::from(index);
        let overrun = plain(|| {
            let mut sides = self.sides.lock();
            sides.blits.retain(Blit::matters);
            let committed = |b: &Blit| {
                b.corrupts && b.progress.stage() == Stage::Committed && b.commands.contains(&at)
            };
            sides.commands.get(&index).is_some_and(|s| s.corrupted)
                || sides.blits.iter().any(committed)
        });
        if overrun {
            unrepresentable(sel, OVERRUN);
        }
    }

    /// `sel` executes commands `range` of the buffer (all of them, `None`: an execution whose
    /// range a buffer holds) in the command buffer of `progress`: it dies where Metal's copy past
    /// the buffer's end may have written over one of their records, or may yet before they run
    /// ([`IcbObject::overrun`]); else the execution is kept while the GPU may run it.
    pub(crate) fn executed(
        &self,
        sel: &'static str,
        range: Option<Range<u64>>,
        progress: Option<&Arc<Progress>>,
    ) {
        self.reached(sel, range, progress, OVERRUN);
    }

    /// `sel`, a `useResource` form, makes the buffer resident for the functions of an encoder of
    /// the command buffer of `progress`, which may copy any of its commands (`copy_command`), as
    /// an execution of the whole buffer reaches them: it dies where Metal's copy past the
    /// buffer's end may have written over one of their records, or may yet before the functions
    /// run ([`IcbObject::overrun`]), as this platform's copy would carry the intact record; else
    /// the residency is kept while the GPU may run it, so such a copy in another command buffer
    /// dies at the copy.
    pub(crate) fn resident(&self, sel: &'static str, progress: Option<&Arc<Progress>>) {
        self.reached(sel, None, progress, RESIDENT_OVERRUN);
    }

    /// [`IcbObject::executed`] and [`IcbObject::resident`], whose death names `why`.
    fn reached(
        &self,
        sel: &'static str,
        range: Option<Range<u64>>,
        progress: Option<&Arc<Progress>>,
        why: &str,
    ) {
        let overrun = plain(|| {
            let mut sides = self.sides.lock();
            if let Some(progress) = progress {
                sides
                    .executions
                    .retain(|p| p.upgrade().is_some_and(|p| p.running()));
                // Once a command buffer: another execution or residency in it changes nothing.
                let listed = (sides.executions.iter()).any(|p| p.as_ptr() == Arc::as_ptr(progress));
                if !listed {
                    sides.executions.push(Arc::downgrade(progress));
                }
            }
            sides.overrun(range.as_ref())
        });
        if overrun {
            unrepresentable(sel, why);
        }
    }

    /// Dies at `sel`, which copies commands `range` of the buffer, where Metal's copy past the
    /// buffer's end may have written over one of their records, or may yet before the copy runs
    /// ([`IcbObject::overrun`]).
    pub(crate) fn copied_intact(&self, sel: &'static str, range: &Range<u64>) {
        if plain(|| self.sides.lock().overrun(Some(range))) {
            unrepresentable(sel, OVERRUN);
        }
    }

    /// Keeps a blit of `progress`'s command buffer that reaches `commands`, and `writes` them or
    /// not, until it can change no answer.
    fn blit(&self, progress: &Arc<Progress>, commands: Range<u64>, writes: bool) {
        plain(|| {
            let mut sides = self.sides.lock();
            sides.blits.retain(Blit::matters);
            sides.blits.push(Blit {
                progress: Arc::clone(progress),
                commands,
                writes,
                corrupts: false,
            });
        });
    }
}

/// `id<MTLIndirectCommandBuffer>`.
pub type VulkanIcb = Obj<IcbObject>;

/// Another reference to each indirect command buffer of `list`, which a `useResource` form made
/// resident: its command buffer holds them until it has run (the docs of [`Space`]).
pub(crate) fn made_resident(list: &[Option<ResourceRef<'_, VulkanPlatform>>]) -> Vec<VulkanIcb> {
    list.iter()
        .flatten()
        .filter_map(|r| match r {
            ResourceRef::Icb(icb) => Some(icb.clone_ref()),
            _ => None,
        })
        .collect()
}

/// A count of a descriptor's, sent with `sel`, as the runtime's `u32`.
fn count(sel: &'static str, value: u64) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| unrepresentable(sel, &format!("a count of {value}")))
}

/// The runtime's descriptor of `desc`, its fields set in the value's order: the object and mesh
/// buffer counts among them, which mesh draws a kernel encodes take, every inheritance bit,
/// Apple's private ones among them (stencil reference values, depth test bounds, scissor
/// rectangles, viewports, blend colour), the scissor rectangle and viewport counts and the
/// render state overrides. Not sent: ray tracing, which the runtime has no form for, and the
/// object threadgroup memory count, which no command core encodes reaches and the runtime reads
/// nowhere, and which the runtime's `u32` might not hold. Core's descriptor has no
/// colour-attachment mapping.
///
/// A render command runs with the encoder's render state, as the runtime's do on a device that
/// claims no Apple10, MoltenVK among them, whatever the bits say (measured: an ICB without
/// `inheritCullMode`, `inheritFrontFacingWinding`, `inheritScissorRects` or `inheritViewports`
/// draws with the encoder's cull mode, winding, scissor rectangle and viewport, as one with them
/// does; the runtime's counts and overrides are kept and read by nothing, as Metal's draws are
/// the encoder's under any). Core's render commands set none of those states, nor object
/// threadgroup lengths.
fn descriptor(desc: &IcbDesc) -> rt::IndirectCommandBufferDescriptor {
    let mut d = rt::IndirectCommandBufferDescriptor::new();
    d.command_types = metal_value("setCommandTypes:", "command types", desc.command_types);
    d.max_vertex_buffer_bind_count = count(
        "setMaxVertexBufferBindCount:",
        desc.max_vertex_buffer_bind_count,
    );
    d.max_fragment_buffer_bind_count = count(
        "setMaxFragmentBufferBindCount:",
        desc.max_fragment_buffer_bind_count,
    );
    d.max_kernel_buffer_bind_count = count(
        "setMaxKernelBufferBindCount:",
        desc.max_kernel_buffer_bind_count,
    );
    d.max_object_buffer_bind_count = count(
        "setMaxObjectBufferBindCount:",
        desc.max_object_buffer_bind_count,
    );
    d.max_mesh_buffer_bind_count = count(
        "setMaxMeshBufferBindCount:",
        desc.max_mesh_buffer_bind_count,
    );
    d.max_kernel_threadgroup_memory_bind_count = count(
        "setMaxKernelThreadgroupMemoryBindCount:",
        desc.max_kernel_threadgroup_memory_bind_count,
    );
    d.inherit_pipeline_state = desc.inherit_pipeline_state;
    d.inherit_buffers = desc.inherit_buffers;
    d.support_dynamic_attribute_stride = desc.support_dynamic_attribute_stride;
    d.max_scissor_rect_count = desc.max_scissor_rect_count;
    d.max_viewport_count = desc.max_viewport_count;
    d.inherit_stencil_reference_values = desc.inherit_stencil_reference_values;
    d.inherit_depth_test_bounds = desc.inherit_depth_test_bounds;
    d.inherit_scissor_rects = desc.inherit_scissor_rects;
    d.inherit_viewports = desc.inherit_viewports;
    d.inherit_blend_color = desc.inherit_blend_color;
    d.allow_override_render_states = desc.allow_override_render_states;
    d.inherit_depth_stencil_state = desc.inherit_depth_stencil_state;
    d.inherit_depth_bias = desc.inherit_depth_bias;
    d.inherit_depth_clip_mode = desc.inherit_depth_clip_mode;
    d.inherit_cull_mode = desc.inherit_cull_mode;
    d.inherit_front_facing_winding = desc.inherit_front_facing_winding;
    d.inherit_triangle_fill_mode = desc.inherit_triangle_fill_mode;
    d
}

/// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:`: the runtime's buffer, with
/// Metal's numbers (the module docs), placed at the descriptor's resource index when its low 32
/// bits are nonzero, here and in the runtime. Metal answers no nil here. A placement past Metal's
/// table of IDs dies, as Metal ends its process there; so does every refusal of the runtime's, as
/// Metal crashes at memory it cannot back. The runtime makes what Metal makes of the descriptors
/// its validation stops (no command types, render and compute types together, more than 31 buffers
/// of a stage).
pub(crate) fn new_indirect_command_buffer(
    device: &VulkanDevice,
    desc: &IcbDesc,
    max_count: u64,
    options: u64,
) -> RaiseResult<Option<VulkanIcb>> {
    let sel = "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:";
    let overrun_shape = OverrunShape::of(desc, max_count, options);
    // The space first, so every death here comes after the wait: Metal's creation waits on its
    // pool's lock while a reservation spins, a placement past its table of IDs too ([`Space`]).
    // The runtime's buffer is made under it, so the runtime takes its IDs in the order these are
    // taken ([`IcbObject::reaches_its_records`]).
    let mut space = plain(|| SPACE.lock());
    let mut d = descriptor(desc);
    let options = crate::gpu::resource_options(sel, options);
    let placed = desc
        .resource_index
        .map(|index| index & 0xffff_ffff)
        .filter(|&id| id != 0);
    if let Some(id) = placed {
        d.placement = rt::IdPlacement::At(rt::ResourceId(id));
    }
    if let Some(id) = placed
        && id >= ID_LIMIT
    {
        metal_aborts(
            sel,
            &format!("an indirect command buffer placed at ID {id}"),
        );
    }
    let raw = send(sel, Policy::Dies, || {
        device
            .raw
            .new_indirect_command_buffer(&d, max_count, options)
    })?;
    let after_encoder = ENCODED.load(Ordering::SeqCst);
    let numbers = space.take(placed, raw.allocated_size());
    drop(space);
    Ok(Some(Obj::new(IcbObject {
        raw: ManuallyDrop::new(raw),
        numbers,
        after_encoder,
        range: Arc::new(Reserved {
            address: numbers.address,
        }),
        strides: d.support_dynamic_attribute_stride,
        inherits: Inherits {
            pipeline: d.inherit_pipeline_state,
            buffers: d.inherit_buffers,
        },
        sides: Arc::new(Mutex::new(Sides {
            device: device.raw.registry_id(),
            ..Sides::default()
        })),
        overrun_shape,
    })))
}

impl IndirectCommandBuffer<VulkanPlatform> for VulkanIcb {
    /// Metal's ID of the buffer, not the runtime's (the module docs).
    fn gpu_resource_id(&self) -> u64 {
        plain(|| self.numbers.id)
    }

    /// The runtime's setters of the command, which Metal hands out at any index and of a buffer
    /// of either kind: the runtime's refusal of those dies.
    fn compute_command(
        &self,
        command_index: u64,
    ) -> RaiseResult<Option<VulkanIndirectComputeCommand>> {
        let sel = "indirectComputeCommandAtIndex:";
        let raw = send(sel, Policy::Dies, || {
            self.raw.indirect_compute_command_at(command_index)
        })?;
        Ok(Some(Obj::new(CommandObject {
            icb: self.clone_ref(),
            index: command_index.low_u32(),
            raw,
        })))
    }

    /// As [`compute_command`](Self::compute_command), for a render command.
    fn render_command(
        &self,
        command_index: u64,
    ) -> RaiseResult<Option<VulkanIndirectRenderCommand>> {
        let sel = "indirectRenderCommandAtIndex:";
        let raw = send(sel, Policy::Dies, || {
            self.raw.indirect_render_command_at(command_index)
        })?;
        Ok(Some(Obj::new(CommandObject {
            icb: self.clone_ref(),
            index: command_index.low_u32(),
            raw,
        })))
    }

    /// Metal's `resourceIndex` of an indirect command buffer is its `gpuResourceID` (measured,
    /// placed or not).
    fn resource_index(&self) -> u64 {
        plain(|| self.numbers.id)
    }

    fn size(&self) -> u64 {
        plain(|| self.raw.size())
    }

    /// Metal's base plus the buffer's address in Metal's range (the module docs). Of a buffer
    /// made after an encoder it dies, as Metal's address then depends on what the encoders took
    /// ([`Space`]).
    fn unique_identifier(&self) -> u64 {
        let address = plain(|| self.numbers.address);
        if self.after_encoder {
            unfollowed(
                "uniqueIdentifier",
                Unfollowed::IcbIdentifierAfterEncoder,
                "the uniqueIdentifier of an indirect command buffer made after an encoder, which \
                 Metal places by what each encoder took of its range",
            );
        }
        UNIQUE_IDENTIFIER_BASE + address
    }
}

// ---- The commands.

/// One command of an indirect command buffer: the runtime's setters, and the buffer.
#[derive(Debug)]
pub struct CommandObject<C> {
    /// The buffer it is a command of.
    icb: VulkanIcb,
    /// Its index there.
    index: u32,
    /// The runtime's setters.
    raw: C,
}

/// `id<MTLIndirectRenderCommand>`.
pub type VulkanIndirectRenderCommand = Obj<CommandObject<rt::IndirectRenderCommand>>;

/// `id<MTLIndirectComputeCommand>`.
pub type VulkanIndirectComputeCommand = Obj<CommandObject<rt::IndirectComputeCommand>>;

/// A runtime setter's answer, for `sel`: every refusal dies (the module docs).
fn set(sel: &'static str, call: impl FnOnce() -> Result<(), rt::Error>) -> RaiseResult<()> {
    send(sel, Policy::Dies, call)
}

/// A binding index core passes as `NSUInteger`, which the runtime takes as `u32`.
fn slot(sel: &'static str, index: u64) -> u32 {
    u32::try_from(index).unwrap_or_else(|_| unrepresentable(sel, &format!("index {index}")))
}

/// `object`, which core may pass as nil, for `sel`: Metal crashes at a nil one (measured: an
/// indexed draw's index buffer, a patch draw's tessellation factor buffer).
fn some<'a, T>(sel: &'static str, object: Option<&'a T>, what: &str) -> &'a T {
    object.unwrap_or_else(|| unrepresentable(sel, &format!("a nil {what}, which Metal crashes at")))
}

/// The death of a getter of another kind's arguments: Metal reads the record as its own kind,
/// which no record of the runtime's holds, and core asks only for a command's own kind.
fn another_kind(sel: &'static str) -> ! {
    unrepresentable(sel, "Metal's reading of a command of another kind")
}

/// The address an arguments getter answers for a buffer of a record: Metal's for nil.
fn address_or_nil(address: Option<rt::GpuAddress>) -> u64 {
    address.map_or(NIL_BUFFER_ADDRESS, |a| a.0)
}

/// The buffer a record's `buffers` name at slot `index`: none where it names none, past the
/// slots it holds, and where it holds a nil buffer, whose address is below
/// `BufferBinding::MIN_BUFFER_ADDRESS` (its offset's low 32 bits): the getters answer a nil
/// buffer's slot by what is kept beside the record ([`SetSlot`]), so one dies while a blit that
/// writes it may run, and after one that failed ([`Unknown`]).
fn named(
    buffers: &[Option<rt::IndirectBufferBinding>],
    index: u64,
) -> Option<rt::IndirectBufferBinding> {
    usize::try_from(index)
        .ok()
        .and_then(|i| buffers.get(i).copied().flatten())
        .filter(|b| b.address.0 >= BufferBinding::MIN_BUFFER_ADDRESS)
}

impl CommandObject<rt::IndirectRenderCommand> {
    /// The command's record, read back, for `sel`.
    fn record(&self, sel: &'static str) -> rt::EncodedRenderCommand {
        self.icb.intact(sel, self.index);
        let i = u64::from(self.index);
        let commands = get(sel, || self.icb.raw.commands(i..i + 1));
        match commands.into_iter().next() {
            Some(rt::IndirectCommand::Render(c)) => c,
            _ => unrepresentable(sel, "a render command the runtime reads as none"),
        }
    }

    /// A pipeline state core set with `sel`: the record names it, and Metal answers its
    /// identifier.
    fn set_pipeline(&self, sel: &'static str, state: &VulkanRenderPipeline) {
        let set = SetPipeline::render(state);
        self.icb
            .side(sel, Access::Write(Kept::Pipeline), self.index, |s| {
                s.pipeline = Some(set);
                s.pipeline_at = crate::gpu::unique_id::now();
            });
    }

    /// Slot `at` of `slots` was set by `sel` to `buffer`, or nil, at `offset` and `stride`.
    fn keep_slot(
        &self,
        sel: &'static str,
        slots: Slots,
        at: u32,
        (buffer, offset, stride): (Option<&VulkanBuffer>, u64, u64),
    ) {
        let set = SetSlot {
            address: buffer
                .map_or(0, |b| b.raw.gpu_address().0)
                .wrapping_add(kept_offset(offset)),
            stride,
        };
        self.icb.keep_slot(sel, self.index, slots, at, set);
    }
}

impl CommandObject<rt::IndirectComputeCommand> {
    /// The command's record, read back, for `sel`.
    fn record(&self, sel: &'static str) -> rt::EncodedComputeCommand {
        self.icb.intact(sel, self.index);
        let i = u64::from(self.index);
        let commands = get(sel, || self.icb.raw.commands(i..i + 1));
        match commands.into_iter().next() {
            Some(rt::IndirectCommand::Compute(c)) => c,
            _ => unrepresentable(sel, "a compute command the runtime reads as none"),
        }
    }

    /// As the render command's.
    fn keep_slot(
        &self,
        sel: &'static str,
        slots: Slots,
        at: u32,
        (buffer, offset, stride): (Option<&VulkanBuffer>, u64, u64),
    ) {
        let set = SetSlot {
            address: buffer
                .map_or(0, |b| b.raw.gpu_address().0)
                .wrapping_add(kept_offset(offset)),
            stride,
        };
        self.icb.keep_slot(sel, self.index, slots, at, set);
    }
}

/// A raw `MTLPrimitiveType`, sent with `sel`.
fn primitive(sel: &'static str, raw: u64) -> PrimitiveType {
    metal_value(sel, "primitive type", raw)
}

impl IndirectRenderCommand<VulkanPlatform> for VulkanIndirectRenderCommand {
    /// One runtime setter per command. A nil pipeline state, index buffer or tessellation factor
    /// buffer, which the runtime's setters take none of, dies; what Metal's getters answer of a
    /// buffer slot is kept ([`SetSlot`]), for one past the slots its record holds, which the
    /// runtime's setter drops. A nil buffer the record holds at its offset (the module docs).
    fn encode(&self, cmd: IcbRenderCmd<'_, VulkanPlatform>) -> RaiseResult<()> {
        let sel = cmd.selector();
        // Each buffer named, with the bytes an indexed draw's indices take
        // ([`crate::gpu::encode_render::indices`]); all of any other.
        let named: Vec<Named<'_>> = match cmd {
            IcbRenderCmd::DrawIndexedPatches {
                patch_index_buffer,
                control_point_index_buffer,
                buffer,
                ..
            } => vec![
                (patch_index_buffer, None),
                (control_point_index_buffer, None),
                (buffer, None),
            ],
            IcbRenderCmd::DrawPatches {
                patch_index_buffer,
                buffer,
                ..
            } => vec![(patch_index_buffer, None), (buffer, None)],
            IcbRenderCmd::DrawIndexedPrimitives {
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                ..
            } => vec![(
                index_buffer,
                crate::gpu::encode_render::indices(index_type, index_buffer_offset, index_count),
            )],
            IcbRenderCmd::SetFragmentBuffer { buffer, .. }
            | IcbRenderCmd::SetVertexBuffer { buffer, .. }
            | IcbRenderCmd::SetVertexBufferAttributeStride { buffer, .. } => vec![(buffer, None)],
            _ => Vec::new(),
        };
        for (buffer, bytes) in named {
            if let Some(buffer) = buffer {
                buffer.used(sel, bytes);
            }
        }
        let c = &self.raw;
        match cmd {
            IcbRenderCmd::ClearBarrier => set(sel, || c.clear_barrier()),
            IcbRenderCmd::SetBarrier => set(sel, || c.set_barrier()),
            IcbRenderCmd::DrawIndexedPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                control_point_index_buffer,
                control_point_index_buffer_offset,
                instance_count,
                base_instance,
                buffer,
                offset,
                instance_stride,
            } => {
                let factors = some(sel, buffer, "tessellation factor buffer");
                let patch_index = patch_index_buffer.map(|b| &b.raw);
                set(sel, || match control_point_index_buffer {
                    Some(control_points) => c.draw_indexed_patches(
                        number_of_patch_control_points,
                        patch_start,
                        patch_count,
                        patch_index,
                        patch_index_buffer_offset,
                        Some(&control_points.raw),
                        control_point_index_buffer_offset,
                        instance_count,
                        base_instance,
                        &factors.raw,
                        offset,
                        instance_stride,
                    ),
                    None => c.draw_patches(
                        number_of_patch_control_points,
                        patch_start,
                        patch_count,
                        patch_index,
                        patch_index_buffer_offset,
                        instance_count,
                        base_instance,
                        &factors.raw,
                        offset,
                        instance_stride,
                    ),
                })
            }
            IcbRenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
                base_vertex,
                base_instance,
            } => {
                let primitive = primitive(sel, primitive_type);
                let index_type: IndexType = metal_value(sel, "index type", index_type);
                let indices = some(sel, index_buffer, "index buffer");
                set(sel, || {
                    c.draw_indexed_primitives(
                        primitive,
                        index_count,
                        index_type,
                        &indices.raw,
                        index_buffer_offset,
                        instance_count,
                        base_vertex,
                        base_instance,
                    )
                })
            }
            IcbRenderCmd::DrawPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                instance_count,
                base_instance,
                buffer,
                offset,
                instance_stride,
            } => {
                let factors = some(sel, buffer, "tessellation factor buffer");
                set(sel, || {
                    c.draw_patches(
                        number_of_patch_control_points,
                        patch_start,
                        patch_count,
                        patch_index_buffer.map(|b| &b.raw),
                        patch_index_buffer_offset,
                        instance_count,
                        base_instance,
                        &factors.raw,
                        offset,
                        instance_stride,
                    )
                })
            }
            IcbRenderCmd::DrawPrimitives {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => {
                let primitive = primitive(sel, primitive_type);
                set(sel, || {
                    c.draw_primitives(
                        primitive,
                        vertex_start,
                        vertex_count,
                        instance_count,
                        base_instance,
                    )
                })
            }
            IcbRenderCmd::Reset => {
                set(sel, || c.reset())?;
                self.icb
                    .side(sel, Access::Write(Kept::All), self.index, |s| {
                        *s = CommandSide::default();
                    });
                Ok(())
            }
            IcbRenderCmd::SetFragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let at = slot(sel, index);
                set(sel, || {
                    c.set_fragment_buffer(buffer.map(|b| &b.raw), kept_offset(offset), at)
                })?;
                self.keep_slot(sel, Slots::Fragment, at, (buffer, offset, u64::MAX));
                Ok(())
            }
            IcbRenderCmd::SetRenderPipelineState { pipeline_state } => {
                let state = some(sel, pipeline_state, "pipeline state");
                set(sel, || c.set_render_pipeline_state(&state.raw))?;
                self.set_pipeline(sel, state);
                Ok(())
            }
            IcbRenderCmd::SetVertexBuffer {
                buffer,
                offset,
                index,
            } => {
                let at = slot(sel, index);
                set(sel, || {
                    c.set_vertex_buffer(buffer.map(|b| &b.raw), kept_offset(offset), at)
                })?;
                self.keep_slot(sel, Slots::Vertex, at, (buffer, offset, u64::MAX));
                Ok(())
            }
            IcbRenderCmd::SetVertexBufferAttributeStride {
                buffer,
                offset,
                stride,
                index,
            } => {
                let at = slot(sel, index);
                set(sel, || {
                    let raw = buffer.map(|b| &b.raw);
                    c.set_vertex_buffer_with_stride(raw, kept_offset(offset), stride, at)
                })?;
                self.keep_slot(sel, Slots::Vertex, at, (buffer, offset, stride));
                Ok(())
            }
        }
    }

    /// The record's draw kind, as `MTLIndirectCommandType`; 0 for none.
    fn command_type(&self) -> u64 {
        let sel = "getCommandType";
        let kind = match self.record(sel).draw {
            None => IndirectCommandType::empty(),
            Some(rt::IndirectDraw::Primitives { .. }) => IndirectCommandType::DRAW,
            Some(rt::IndirectDraw::IndexedPrimitives { .. }) => IndirectCommandType::DRAW_INDEXED,
            Some(rt::IndirectDraw::Patches { .. }) => IndirectCommandType::DRAW_PATCHES,
            Some(rt::IndirectDraw::IndexedPatches { .. }) => {
                IndirectCommandType::DRAW_INDEXED_PATCHES
            }
            Some(_) => unrepresentable(sel, "a draw of a kind Metal has no type for"),
        };
        u64::from(kind.as_raw())
    }

    fn fragment_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getFragmentBufferAddressAtIndex:";
        let buffers = (Slots::Fragment, &*self.record(sel).fragment_buffers);
        Ok(self.icb.buffer_address(sel, self.index, buffers, index))
    }

    /// The identifier of the pipeline state the record names, live or released; 0 for none
    /// ([`IcbObject::released_pipeline`]).
    fn pipeline_state_unique_identifier(&self) -> u64 {
        let sel = "getPipelineStateUniqueIdentifier";
        let record = self.record(sel);
        match record.pipeline {
            None => self
                .icb
                .released_pipeline(sel, self.index, record.pipeline_id),
            Some(state) => {
                let named = (state.device().registry_id(), state.gpu_resource_id().0);
                self.icb.pipeline_identifier(sel, self.index, named)
            }
        }
    }

    fn vertex_attribute_stride(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getVertexAttributeStrideAtIndex:";
        let buffers = (Slots::Vertex, &*self.record(sel).vertex_buffers);
        Ok(self.icb.attribute_stride(sel, self.index, buffers, index))
    }

    fn vertex_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getVertexBufferAddressAtIndex:";
        let buffers = (Slots::Vertex, &*self.record(sel).vertex_buffers);
        Ok(self.icb.buffer_address(sel, self.index, buffers, index))
    }

    fn has_barrier(&self) -> bool {
        self.record("hasBarrier").barrier
    }

    fn draw_args(&self) -> Option<DrawArgs> {
        let sel = "drawArguments";
        match self.record(sel).draw {
            Some(rt::IndirectDraw::Primitives {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            }) => Some(DrawArgs {
                primitive_type: u64::from(primitive_type.as_raw()),
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            }),
            _ => another_kind(sel),
        }
    }

    /// Metal's `baseVertex` answers the record's 32 bits, zero-extended (measured: -2 reads
    /// 4294967294).
    fn draw_indexed_args(&self) -> Option<DrawIndexedArgs> {
        let sel = "drawIndexedArguments";
        match self.record(sel).draw {
            Some(rt::IndirectDraw::IndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                instance_count,
                base_vertex,
                base_instance,
            }) => Some(DrawIndexedArgs {
                index_buffer_gpu_virtual_address: index_buffer.0,
                primitive_type: u64::from(primitive_type.as_raw()),
                index_type: u64::from(index_type.as_raw()),
                index_count,
                instance_count,
                base_vertex: u64::from(base_vertex.cast_unsigned().low_u32()),
                base_instance,
            }),
            _ => another_kind(sel),
        }
    }

    fn draw_patches_args(&self) -> Option<DrawPatchesArgs> {
        let sel = "drawPatchesArguments";
        match self.record(sel).draw {
            Some(rt::IndirectDraw::Patches {
                control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                instance_count,
                base_instance,
                ..
            }) => Some(DrawPatchesArgs {
                patch_index_buffer_gpu_virtual_address: address_or_nil(patch_index_buffer),
                number_of_patch_control_points: control_points,
                patch_start,
                patch_count,
                instance_count,
                base_instance,
            }),
            _ => another_kind(sel),
        }
    }

    fn draw_indexed_patches_args(&self) -> Option<DrawIndexedPatchesArgs> {
        let sel = "drawIndexedPatchesArguments";
        match self.record(sel).draw {
            Some(rt::IndirectDraw::IndexedPatches {
                control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                control_point_index_buffer,
                instance_count,
                base_instance,
                ..
            }) => Some(DrawIndexedPatchesArgs {
                patch_index_buffer_gpu_virtual_address: address_or_nil(patch_index_buffer),
                control_point_index_buffer_gpu_virtual_address: address_or_nil(
                    control_point_index_buffer,
                ),
                number_of_patch_control_points: control_points,
                patch_start,
                patch_count,
                instance_count,
                base_instance,
            }),
            _ => another_kind(sel),
        }
    }

    /// A patch draw's factors: the buffer's address plus its offset, the instance stride, and
    /// the scale 1, which every patch draw a command's setter encodes holds.
    fn tessellation_factor_args(&self) -> Option<TessellationFactorArgs> {
        let sel = "getTessellationFactorArguments";
        match self.record(sel).draw {
            Some(
                rt::IndirectDraw::Patches {
                    factor_buffer,
                    factor_instance_stride,
                    ..
                }
                | rt::IndirectDraw::IndexedPatches {
                    factor_buffer,
                    factor_instance_stride,
                    ..
                },
            ) => Some(TessellationFactorArgs {
                gpu_virtual_address: factor_buffer.0,
                instance_stride: factor_instance_stride,
                scale: 1.0,
            }),
            _ => another_kind(sel),
        }
    }
}

impl IndirectComputeCommand<VulkanPlatform> for VulkanIndirectComputeCommand {
    /// One runtime setter per command. A nil pipeline state, which the runtime's setter takes
    /// none of, dies; what Metal's getters answer of a buffer slot is kept ([`SetSlot`]), as the
    /// render command's are.
    fn encode(&self, cmd: IcbComputeCmd<'_, VulkanPlatform>) -> RaiseResult<()> {
        let sel = cmd.selector();
        if let IcbComputeCmd::SetKernelBuffer {
            buffer: Some(buffer),
            ..
        }
        | IcbComputeCmd::SetKernelBufferAttributeStride {
            buffer: Some(buffer),
            ..
        } = cmd
        {
            buffer.used(sel, None);
        }
        let c = &self.raw;
        match cmd {
            IcbComputeCmd::ClearBarrier => set(sel, || c.clear_barrier()),
            IcbComputeCmd::ConcurrentDispatchThreadgroups {
                threadgroups_per_grid,
                threads_per_threadgroup,
            } => set(sel, || {
                c.concurrent_dispatch_threadgroups(
                    crate::gpu::size(threadgroups_per_grid),
                    crate::gpu::size(threads_per_threadgroup),
                )
            }),
            IcbComputeCmd::ConcurrentDispatchThreads {
                threads_per_grid,
                threads_per_threadgroup,
            } => set(sel, || {
                c.concurrent_dispatch_threads(
                    crate::gpu::size(threads_per_grid),
                    crate::gpu::size(threads_per_threadgroup),
                )
            }),
            IcbComputeCmd::Reset => {
                set(sel, || c.reset())?;
                self.icb
                    .side(sel, Access::Write(Kept::All), self.index, |s| {
                        *s = CommandSide::default();
                    });
                Ok(())
            }
            IcbComputeCmd::SetBarrier => set(sel, || c.set_barrier()),
            IcbComputeCmd::SetComputePipelineState { pipeline_state } => {
                let state = some(sel, pipeline_state, "pipeline state");
                set(sel, || c.set_compute_pipeline_state(&state.raw))?;
                let pipeline = SetPipeline::compute(state);
                self.icb
                    .side(sel, Access::Write(Kept::Pipeline), self.index, |s| {
                        s.pipeline = Some(pipeline);
                        s.pipeline_at = crate::gpu::unique_id::now();
                    });
                Ok(())
            }
            IcbComputeCmd::SetImageblockWidth { width, height } => {
                set(sel, || c.set_imageblock_size(width, height))
            }
            IcbComputeCmd::SetKernelBuffer {
                buffer,
                offset,
                index,
            } => {
                let at = slot(sel, index);
                set(sel, || {
                    c.set_kernel_buffer(buffer.map(|b| &b.raw), kept_offset(offset), at)
                })?;
                self.keep_slot(sel, Slots::Kernel, at, (buffer, offset, u64::MAX));
                Ok(())
            }
            IcbComputeCmd::SetKernelBufferAttributeStride {
                buffer,
                offset,
                stride,
                index,
            } => {
                let at = slot(sel, index);
                set(sel, || {
                    let raw = buffer.map(|b| &b.raw);
                    c.set_kernel_buffer_with_stride(raw, kept_offset(offset), stride, at)
                })?;
                self.keep_slot(sel, Slots::Kernel, at, (buffer, offset, stride));
                Ok(())
            }
            IcbComputeCmd::SetStageInRegion { region } => {
                set(sel, || c.set_stage_in_region(crate::gpu::region(region)))
            }
            IcbComputeCmd::SetThreadgroupMemoryLength { length, index } => {
                let length = u32::try_from(length)
                    .unwrap_or_else(|_| unrepresentable(sel, &format!("a length of {length}")));
                let at = slot(sel, index);
                set(sel, || c.set_threadgroup_memory_length(length, at))?;
                self.icb
                    .side(sel, Access::Write(Kept::Length(at)), self.index, |s| {
                        s.lengths.insert(at, length)
                    });
                Ok(())
            }
        }
    }

    /// The record's dispatch kind, as `MTLIndirectCommandType`; 0 for none.
    fn command_type(&self) -> u64 {
        let sel = "getCommandType";
        let kind = match self.record(sel).dispatch {
            None => IndirectCommandType::empty(),
            Some(rt::IndirectDispatch::Threadgroups { .. }) => {
                IndirectCommandType::CONCURRENT_DISPATCH
            }
            Some(rt::IndirectDispatch::Threads { .. }) => {
                IndirectCommandType::CONCURRENT_DISPATCH_THREADS
            }
            Some(_) => unrepresentable(sel, "a dispatch of a kind Metal has no type for"),
        };
        u64::from(kind.as_raw())
    }

    fn kernel_attribute_stride(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getKernelAttributeStrideAtIndex:";
        let buffers = (Slots::Kernel, &*self.record(sel).kernel_buffers);
        Ok(self.icb.attribute_stride(sel, self.index, buffers, index))
    }

    fn kernel_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getKernelBufferAddressAtIndex:";
        let buffers = (Slots::Kernel, &*self.record(sel).kernel_buffers);
        Ok(self.icb.buffer_address(sel, self.index, buffers, index))
    }

    /// The identifier of the pipeline state the record names, live or released; 0 for none, a
    /// reset's retained one included, as Metal's getter reports it
    /// ([`IcbObject::released_pipeline`]).
    fn pipeline_state_unique_identifier(&self) -> u64 {
        let sel = "getPipelineStateUniqueIdentifier";
        let record = self.record(sel);
        match record.pipeline {
            None => self
                .icb
                .released_pipeline(sel, self.index, record.pipeline_id),
            Some(state) => {
                let named = (state.device().registry_id(), state.gpu_resource_id().0);
                self.icb.pipeline_identifier(sel, self.index, named)
            }
        }
    }

    fn stage_in_region(&self) -> MtlRegion {
        let r = self.record("getStageInRegion").stage_in_region;
        MtlRegion::new(
            MtlOrigin::new(r.origin.x, r.origin.y, r.origin.z),
            MtlSize::new(r.size.width, r.size.height, r.size.depth),
        )
    }

    /// The length set; past the descriptor's lengths, 0.
    /// The record's length where it holds the index, else the one core set there, or 0.
    fn threadgroup_memory_length(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getThreadgroupMemoryLengthAtIndex:";
        let lengths = self.record(sel).threadgroup_memory_lengths;
        let held = usize::try_from(index).ok().and_then(|i| lengths.get(i));
        let length = match (held, u32::try_from(index)) {
            (Some(&l), _) => l,
            (None, Ok(at)) => self
                .icb
                .side(sel, Access::Read(Kept::Length(at)), self.index, |s| {
                    s.lengths.get(&at).copied().unwrap_or(0)
                }),
            (None, Err(_)) => 0,
        };
        Ok(u64::from(length))
    }

    fn has_barrier(&self) -> bool {
        self.record("hasBarrier").barrier
    }

    fn dispatch_threadgroups_args(&self) -> Option<DispatchThreadgroupsArgs> {
        let sel = "dispatchThreadgroupsArguments";
        match self.record(sel).dispatch {
            Some(rt::IndirectDispatch::Threadgroups {
                threadgroups_per_grid: g,
                threads_per_threadgroup: t,
            }) => Some(DispatchThreadgroupsArgs {
                threadgroups_per_grid: MtlSize::new(g.width, g.height, g.depth),
                threads_per_threadgroup: MtlSize::new(t.width, t.height, t.depth),
            }),
            _ => another_kind(sel),
        }
    }

    fn dispatch_threads_args(&self) -> Option<DispatchThreadsArgs> {
        let sel = "dispatchThreadsArguments";
        match self.record(sel).dispatch {
            Some(rt::IndirectDispatch::Threads {
                threads_per_grid: g,
                threads_per_threadgroup: t,
            }) => Some(DispatchThreadsArgs {
                threads_per_grid: MtlSize::new(g.width, g.height, g.depth),
                threads_per_threadgroup: MtlSize::new(t.width, t.height, t.depth),
            }),
            _ => another_kind(sel),
        }
    }

    fn image_block_args(&self) -> Option<ImageBlockArgs> {
        let c = self.record("getImageBlockSize");
        Some(ImageBlockArgs {
            width: c.imageblock_width,
            height: c.imageblock_height,
        })
    }
}

/// The tests of identifiers need a process each, as `cargo nextest` (the repository's runner,
/// `cargo xtask check`) gives them: what a record names is decided by process-wide state (the
/// states kept by resource ID, the devices a state was released on, [`crate::gpu::unique_id`])
/// over the one runtime device of the process, so under `cargo test`'s threads one test's states,
/// releases and spans reach another's.
#[cfg(test)]
mod tests {
    use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
    use reims_vgpu_core::platform::Platform;
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _, ResourceRef};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::platform::types::NsRange;
    use reims_vgpu_core::raise::OnFault;
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::command::VulkanCommandBuffer;
    use crate::gpu::device::tests::dev;

    const ADD: &[u8] =
        include_bytes!("../../../../tests/suite/fixtures/shaders/compute_add.metallib");

    /// `MTLCommandBufferStatusCompleted`.
    const COMPLETED: u64 = 4;

    /// A descriptor of both dispatch kinds, two kernel buffers with dynamic strides and two
    /// threadgroup lengths.
    fn compute_desc() -> IcbDesc {
        IcbDesc {
            command_types: 0x60,
            max_kernel_buffer_bind_count: 2,
            max_kernel_threadgroup_memory_bind_count: 2,
            support_dynamic_attribute_stride: true,
            ..IcbDesc::default()
        }
    }

    /// A descriptor of the four draw kinds, two vertex buffers with dynamic strides and two
    /// fragment buffers.
    fn render_desc() -> IcbDesc {
        IcbDesc {
            command_types: 0xf,
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 2,
            support_dynamic_attribute_stride: true,
            ..IcbDesc::default()
        }
    }

    /// A buffer of `n` commands of `desc`, Shared.
    fn make<P: Platform>(d: &P::Device, desc: &IcbDesc, n: u64) -> P::Icb {
        d.new_indirect_command_buffer(desc, n, 0).unwrap().unwrap()
    }

    /// A buffer made before the process's first encoder answers Metal's `uniqueIdentifier`, the
    /// first one's `0x1000018000`, before and after `encode` made an encoder of a command buffer
    /// never committed; one made after it dies at the read, naming
    /// `icb-identifier-after-encoder`, its creation and its other numbers gone on
    fn identifiers_after(encode: impl FnOnce(&VulkanCommandBuffer)) {
        let d = dev();
        let before = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        assert_eq!(before.unique_identifier(), 0x10_0001_8000);
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        encode(&cb);
        let after = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        assert_eq!(before.unique_identifier(), 0x10_0001_8000);
        assert_eq!((after.gpu_resource_id(), after.resource_index()), (2, 2));
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| after.unique_identifier()).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "uniqueIdentifier")
        );
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        let code = crate::answer::Unfollowed::IcbIdentifierAfterEncoder.code();
        assert!(line.contains(code), "{line}");
    }

    /// [`identifiers_after`] an empty blit encoder
    #[test]
    fn identifiers_after_a_blit_encoder_die() {
        identifiers_after(|cb| {
            let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        });
    }

    /// [`identifiers_after`] an empty compute encoder, which took nothing of Metal's range on the
    /// M4
    #[test]
    fn identifiers_after_a_compute_encoder_die() {
        identifiers_after(|cb| {
            let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        });
    }

    /// [`identifiers_after`] an empty compute encoder of a pass descriptor
    #[test]
    fn identifiers_after_a_compute_pass_encoder_die() {
        identifiers_after(|cb| {
            let desc = reims_vgpu_core::platform::values::ComputePassDesc::default();
            let e = cb.compute_encoder_for_pass(&desc).unwrap().unwrap();
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        });
    }

    /// [`identifiers_after`] an empty render encoder
    #[test]
    fn identifiers_after_a_render_encoder_die() {
        use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
        identifiers_after(|cb| {
            let p = crate::platform::tests::platform();
            let (_target, pass) = crate::gpu::encode_render::tests::pass_of(&p, &dev(), (4, 4));
            let e = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
            e.encode(OnFault::Return, RenderCmd::EndEncoding).unwrap();
        });
    }

    /// The runtime's ID of a buffer, which its records' address follows, is the guest's, past a
    /// reservation and for a placed buffer too; IDs are free again once their buffer goes, in
    /// both
    #[test]
    fn the_runtimes_ids_are_metals() {
        let d = dev();
        d.reserve_resource_indices(5, &[2]).unwrap();
        let placed = IcbDesc {
            resource_index: Some(9),
            ..compute_desc()
        };
        let ids = |b: &VulkanIcb| (b.gpu_resource_id(), b.raw.gpu_resource_id().0);
        let a = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        let b = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        let mut seen = vec![ids(&a), ids(&b)];
        drop((a, b));
        let c = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        let p = make::<VulkanPlatform>(&d, &placed, 1);
        let e = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        seen.extend([ids(&c), ids(&p), ids(&e)]);
        let reach: Vec<bool> = [&c, &p, &e].map(|b| b.reaches_its_records()).to_vec();
        assert_eq!(seen, [(1, 1), (3, 3), (1, 1), (9, 9), (3, 3)]);
        assert_eq!(reach, [true, true, true]);
    }

    /// A device the host releases and opens again is the one runtime device, whose IDs go on with
    /// these, as Metal's device lives for the process: 2 reserved on a handle dropped after, the
    /// device opened again hands out 1 then 3, the runtime's too, where a runtime device made
    /// again would hand out 1 then 2. It holds no other handle ([`dev`]) before the drop.
    #[test]
    fn a_device_opened_again_keeps_its_ids() {
        let open = || VulkanPlatform::system_default_device().expect("a Vulkan device (MoltenVK)");
        let first = open();
        first.reserve_resource_indices(5, &[2]).unwrap();
        drop(first);
        let d = open();
        let made: Vec<VulkanIcb> = (0..2)
            .map(|_| make::<VulkanPlatform>(&d, &compute_desc(), 1))
            .collect();
        let seen: Vec<(u64, u64)> = made
            .iter()
            .map(|b| (b.gpu_resource_id(), b.raw.gpu_resource_id().0))
            .collect();
        assert_eq!(seen, [(1, 1), (3, 3)]);
    }

    /// A reservation Metal spins at parks its thread for good holding the space's lock, as
    /// Metal's spinning thread holds its pool's, so a take or another reservation would wait on
    /// it for good: here the lock stays held past 500 ms. One Metal takes lets the lock go. The
    /// test leaves the parked thread behind, holding a space of its own (leaked, not the
    /// process's), until the test's process exits
    #[test]
    fn a_reservation_metal_spins_at_holds_the_space() {
        use std::time::Duration;

        let space = || -> &'static Mutex<Space> {
            Box::leak(Box::new(Mutex::new(Space {
                pool: None,
                ranges: BTreeMap::new(),
            })))
        };
        let taken = space();
        let (reservation, held) = reserve_in(taken, &[10]);
        assert_eq!(reservation, Reservation::Taken);
        drop(held);
        assert!(!taken.is_locked());
        let spun = space();
        std::thread::Builder::new()
            .name("spun reservation".into())
            .spawn(move || {
                // Parks for good: it never answers.
                let _never = reserve_in(spun, &[(1 << 31) + 5]);
            })
            .unwrap();
        // Up to 10 s for the thread to take the lock, which it then holds for good.
        let held = (0..10_000).any(|_| {
            std::thread::sleep(Duration::from_millis(1));
            spun.is_locked()
        });
        assert!(held, "never held");
        assert!(spun.try_lock_for(Duration::from_millis(500)).is_none());
    }

    /// The pool's order without a device: a free range below an ID in the second half of the
    /// pool is handed out after the smaller one above it, a full pool doubles, and IDs given back
    /// join their free neighbours
    #[test]
    fn the_pool_hands_out_the_smallest_free_range_first() {
        let mut pool = Pool::new(16);
        assert_eq!(pool.reserve(&[10]), Reservation::Taken);
        let first: Vec<u64> = (0..16).map(|_| pool.take()).collect();
        assert_eq!(
            first,
            [11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17]
        );
        assert_eq!(pool.reserve(&[20]), Reservation::Ignored);
        for id in [3, 4, 6] {
            pool.give(id);
        }
        assert_eq!([pool.take(), pool.take(), pool.take()], [6, 3, 4]);
        let mut empty = Pool::new(32);
        assert_eq!(empty.reserve(&[0]), Reservation::Ignored);
        assert!(!empty.started);
    }

    /// What the pool answers for the calls Metal spins at or hands out IDs twice after, without
    /// a device, where the selector would park or die: an ID of 2^31 to 2^32 - 1 in a call Metal
    /// takes hangs, after a call of 0 alone too, and a later call with an ID below 2^31 hands out
    /// twice; neither changes the pool, and once a buffer was made both are ignored
    #[test]
    fn the_pool_knows_the_calls_metal_spins_at_or_hands_out_twice_after() {
        for ids in [
            &[1 << 31][..],
            &[0, (1 << 31) + 5],
            &[3, u64::from(u32::MAX)],
        ] {
            let mut pool = Pool::new(FIRST_POOL);
            assert_eq!(pool.reserve(ids), Reservation::Hangs, "{ids:?}");
            assert_eq!(pool.take(), 1);
        }
        let mut edge = Pool::new(FIRST_POOL);
        assert_eq!(edge.reserve(&[(1 << 31) - 1]), Reservation::Taken);
        assert_eq!(edge.take(), 1);
        let mut zero = Pool::new(FIRST_POOL);
        assert_eq!(zero.reserve(&[0, 0]), Reservation::Ignored);
        assert_eq!(zero.reserve(&[(1 << 31) + 5]), Reservation::Hangs);
        assert_eq!(zero.reserve(&[20]), Reservation::HandsOutTwice);
        assert_eq!(zero.reserve(&[(1 << 31) - 1]), Reservation::HandsOutTwice);
        assert!(!zero.started);
        assert_eq!(zero.take(), 1);
        assert_eq!(zero.reserve(&[1 << 31]), Reservation::Ignored);
        assert_eq!(zero.reserve(&[20]), Reservation::Ignored);
        assert_eq!(zero.take(), 2);
    }

    /// The runtime's pipeline state of `add_one`, with indirect command buffer support, as core's
    /// handle.
    fn icb_pipeline(d: &VulkanDevice) -> VulkanComputePipeline {
        let library = d.raw.new_library_with_data(ADD).unwrap();
        let function = library.new_function_with_name("add_one").unwrap();
        let mut desc =
            ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        desc.support_indirect_command_buffers = true;
        let raw = d
            .raw
            .new_compute_pipeline_state(&desc, &rt::PipelineOptions::default())
            .unwrap();
        Obj::new(crate::gpu::resource::ComputePipelineObject::new(raw))
    }

    /// A Shared buffer holding `w`.
    fn holding(d: &VulkanDevice, w: &[u32]) -> VulkanBuffer {
        let b = d
            .new_buffer_of_length(OnFault::Return, u64::try_from(w.len() * 4).unwrap(), 0)
            .unwrap()
            .unwrap();
        // SAFETY: the buffer's Shared bytes, as long as `w`, written before any command uses them.
        unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), b.contents().as_ptr::<u32>(), w.len()) };
        b
    }

    /// The words of a Shared buffer.
    fn words(b: &VulkanBuffer) -> Vec<u32> {
        let n = usize::try_from(b.length() / 4).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the GPU's writes completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u32>(), n) }.to_vec()
    }

    /// A compute command names the pipeline state core set by its handle's identifier, 0 after
    /// a reset; a blit's copy carries the command, its pipeline and barrier, to another buffer,
    /// whose execution runs it; a blit's reset leaves a fresh command
    #[test]
    fn commands_run_copy_and_reset() {
        let d = dev();
        let pipeline = icb_pipeline(&d);
        let desc = IcbDesc {
            command_types: 0x40,
            max_kernel_buffer_bind_count: 2,
            ..IcbDesc::default()
        };
        let (a, b) = (
            make::<VulkanPlatform>(&d, &desc, 1),
            make::<VulkanPlatform>(&d, &desc, 2),
        );
        let input = holding(&d, &[10, 20, 30, 40]);
        let out = holding(&d, &[0; 4]);
        let c = a.compute_command(0).unwrap().unwrap();
        for cmd in [
            IcbComputeCmd::SetComputePipelineState {
                pipeline_state: Some(&pipeline),
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(&out),
                offset: 0,
                index: 0,
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(&input),
                offset: 0,
                index: 1,
            },
            IcbComputeCmd::ConcurrentDispatchThreads {
                threads_per_grid: MtlSize::new(4, 1, 1),
                threads_per_threadgroup: MtlSize::new(4, 1, 1),
            },
            IcbComputeCmd::SetBarrier,
        ] {
            c.encode(cmd).unwrap();
        }
        assert_eq!(
            c.pipeline_state_unique_identifier(),
            pipeline.unique_identifier()
        );
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyIndirectCommandBuffer {
                source: &a,
                source_range: NsRange::new(0, 1),
                destination: &b,
                destination_index: 1,
            },
            BlitCmd::OptimizeIndirectCommandBuffer {
                indirect_command_buffer: &b,
                range: NsRange::new(0, 2),
            },
            BlitCmd::ResetCommandsInBuffer {
                buffer: &a,
                range: NsRange::new(0, 1),
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Return, cmd).unwrap();
        }
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            ComputeCmd::UseResources {
                resources: &[
                    Some(ResourceRef::Buffer(&input)),
                    Some(ResourceRef::Buffer(&out)),
                    Some(ResourceRef::Icb(&b)),
                ],
                usage: 3,
            },
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &b,
                execution_range: NsRange::new(1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
        assert_eq!(words(&out), [11, 21, 31, 41]);
        let copied = b.compute_command(1).unwrap().unwrap();
        assert_eq!(
            (
                copied.command_type(),
                copied.pipeline_state_unique_identifier(),
                copied.has_barrier()
            ),
            (0x40, pipeline.unique_identifier(), true)
        );
        let reset = a.compute_command(0).unwrap().unwrap();
        assert_eq!(
            (
                reset.command_type(),
                reset.pipeline_state_unique_identifier(),
                reset.has_barrier()
            ),
            (0, 0, false)
        );
        c.encode(IcbComputeCmd::SetComputePipelineState {
            pipeline_state: Some(&pipeline),
        })
        .unwrap();
        c.encode(IcbComputeCmd::Reset).unwrap();
        assert_eq!(c.pipeline_state_unique_identifier(), 0);
    }

    /// The tests' kernels that encode indirect commands, which `build.rs` compiled from
    /// `icb_encode.metal`.
    const ICB_ENCODE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/icb_encode.metallib"));

    /// Two states of `nothing` and two of `a_vertex` and `a_colour`, each made from a guest
    /// descriptor that supports indirect command buffers (compute key 7, render key 25).
    fn twin_states<P: Platform>(
        d: &P::Device,
        library: &P::Library,
    ) -> (Vec<P::ComputePipeline>, Vec<P::RenderPipeline>) {
        use crate::gpu::serialized::tests::{hex, pipeline_blob};
        use reims_vgpu_core::platform::resource::Library as _;
        use reims_vgpu_core::rt::Hold;
        let function = |name: &str| {
            library
                .new_function(OnFault::Return, name)
                .unwrap()
                .unwrap()
        };
        let (kernel, vertex, fragment) = (
            function("nothing"),
            function("a_vertex"),
            function("a_colour"),
        );
        let compute = hex("02 07 04 01000000 00 04 05000000");
        // The basic blob with one more entry, key 25 set, before its three: the colour
        // attachments' block, at the offset its key 8 holds (byte 3 on), moves past it.
        let mut render = pipeline_blob("basic", 0x21, 0x22, 0);
        render[0] += 1;
        render[3] += 6;
        render.splice(1..1, [25, 4, 1, 0, 0, 0]);
        let computes = (0..2)
            .map(|_| {
                let mut resolve = |_: Hold<'_>, r: u64| Ok((r == 5).then(|| kernel.clone_ref()));
                let desc = d
                    .deserialize_compute_pipeline_descriptor(Hold::NONE, &compute, &mut resolve)
                    .unwrap()
                    .unwrap();
                d.new_compute_pipeline_state(Some(&desc), 0)
                    .unwrap()
                    .unwrap()
                    .unwrap()
            })
            .collect();
        let renders = (0..2)
            .map(|_| {
                let mut resolve = |_: Hold<'_>, r: u64| {
                    Ok(match r {
                        0x21 => Some(vertex.clone_ref()),
                        0x22 => Some(fragment.clone_ref()),
                        _ => None,
                    })
                };
                let desc = d
                    .deserialize_render_pipeline_descriptor(Hold::NONE, &render, &mut resolve)
                    .unwrap()
                    .unwrap();
                d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
                    .unwrap()
                    .unwrap()
                    .unwrap()
            })
            .collect();
        (computes, renders)
    }

    /// Runs the kernel `name` of `library` once on `d`, its argument buffer holding `icb`'s ID
    /// and `state`, a pipeline state's ID, which it encodes `icb`'s command 0 with.
    fn encode_by_kernel<P: Platform>(
        d: &P::Device,
        library: &P::Library,
        name: &str,
        icb: &P::Icb,
        state: u64,
    ) {
        use reims_vgpu_core::platform::resource::Library as _;
        let f = library
            .new_function(OnFault::Return, name)
            .unwrap()
            .unwrap();
        let kernel = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let held = d
            .new_buffer_of_length(OnFault::Return, 16, 0)
            .unwrap()
            .unwrap();
        let words = [icb.gpu_resource_id(), state];
        // SAFETY: the buffer's 16 Shared bytes, written before any command uses them.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), held.contents().as_ptr::<u64>(), 2)
        };
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let buffers = [Some(&held)];
        let resident = [Some(ResourceRef::Icb(icb))];
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&kernel),
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::UseResources {
                resources: &resident,
                usage: 3,
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
    }

    /// Past [`WINDOWS`], two spans that ended are joined only where the span covering both names
    /// the same states for every record: not across a state made and released between them, nor
    /// across core's set of a command between their ends
    #[test]
    fn spans_are_joined_only_where_that_names_the_same_states() {
        use crate::gpu::unique_id::{next_of, now, released};
        // A registry ID no device answers, so no other test's states meet these.
        let device = u64::MAX - 9;
        assert!(joinable(device, (10, 20), (15, 30), &[]));
        assert!(!joinable(device, (10, 20), (30, 40), &[25]));
        assert!(joinable(device, (10, 20), (30, 40), &[19, 40]));
        assert!(!joinable(device, (10, 20), (30, 40), &[20]));
        let before = now();
        let made = next_of(device, 5);
        released(device, 5, made);
        let after = now();
        assert!(!joinable(device, (0, before), (after, after + 1), &[]));
        assert!(joinable(
            device,
            (after, after + 1),
            (after + 2, after + 3),
            &[]
        ));
        let ended = |start: u64| Window {
            start,
            end: Arc::new(AtomicU64::new(start + 1)),
            of: 0,
        };
        let mut sides = Sides {
            device,
            ..Sides::default()
        };
        let base = now();
        for i in 0..=WINDOWS as u64 {
            sides.write(ended(base + 10 * i));
        }
        assert_eq!(sides.writes.len(), WINDOWS);
        let joined = sides.writes[0].span();
        assert_eq!(joined, (base, base + 11));
        // A state made and released between two spans keeps them apart.
        let mut apart = Sides {
            device,
            ..Sides::default()
        };
        apart.write(ended(0));
        apart.write(ended(before - 1));
        apart.write(ended(after + 1));
        for i in 0..WINDOWS as u64 - 2 {
            apart.write(Window {
                start: after + 10 + i,
                end: Arc::new(AtomicU64::new(OPEN)),
                of: usize::try_from(i + 1).unwrap(),
            });
        }
        // The first two were joined, and the state lies between what is left of them.
        assert_eq!(apart.writes.len(), WINDOWS);
        let spans: Vec<_> = apart.writes[..2].iter().map(Window::span).collect();
        assert_eq!(spans, [(0, before), (after + 1, after + 2)]);
        apart.write(Window {
            start: after + 10 + WINDOWS as u64,
            end: Arc::new(AtomicU64::new(OPEN)),
            of: WINDOWS,
        });
        assert_eq!(apart.writes.len(), WINDOWS + 1);
    }

    /// The states a record can name by its resource ID, by when core set it and when the GPU may
    /// have written it: one names its identifier, two or more, or one past those kept, none
    #[test]
    fn a_record_names_the_state_that_held_its_id_when_it_was_written() {
        use crate::gpu::unique_id::{History, Holder, LIVE};
        let history = |holders: &[(u64, u64, u64)], lost_until| History {
            holders: (holders.iter())
                .map(|&(identifier, made, gone)| Holder {
                    identifier,
                    made,
                    gone,
                })
                .collect(),
            lost_until,
        };
        // States A and B held ID 7, A from point 10 to its release at 18 and B from point 20.
        let ab = history(&[(0xa, 10, 18), (0xb, 20, LIVE)], 0);
        let ab = Some(&ab);
        let set_a = Some(((7, 0xa), 15));
        // Core's set alone, though B took the ID since: the record is core's.
        assert_eq!(named_state(set_a, &[], 7, ab), Ok(0xa));
        // A span that ended before core's set changes nothing.
        assert_eq!(named_state(set_a, &[(11, 14)], 7, ab), Ok(0xa));
        // A span after core's set while A alone held the ID.
        assert_eq!(named_state(set_a, &[(16, 18)], 7, ab), Ok(0xa));
        // One while B held it: the GPU may have written B over core's A.
        assert!(named_state(set_a, &[(16, 22)], 7, ab).is_err());
        assert!(named_state(set_a, &[(21, OPEN)], 7, ab).is_err());
        // No set: the spans decide, each naming the states that held the ID during it.
        assert_eq!(named_state(None, &[(11, 19)], 7, ab), Ok(0xa));
        assert_eq!(named_state(None, &[(21, 30)], 7, ab), Ok(0xb));
        assert!(named_state(None, &[(11, 19), (21, 30)], 7, ab).is_err());
        assert!(named_state(None, &[(17, 21)], 7, ab).is_err());
        // A span from A's release to B's creation, when no state held the ID, names neither.
        assert!(named_state(None, &[(18, 20)], 7, ab).is_err());
        assert!(named_state(None, &[(11, OPEN)], 7, ab).is_err());
        // A set of another state, which the record no longer names: the GPU wrote it since.
        assert_eq!(
            named_state(Some(((8, 0xc), 15)), &[(16, 18)], 7, ab),
            Ok(0xa)
        );
        // Nothing tells which: the only state of the ID, else none.
        let a = history(&[(0xa, 10, LIVE)], 0);
        assert_eq!(named_state(None, &[], 7, Some(&a)), Ok(0xa));
        assert!(named_state(None, &[], 7, ab).is_err());
        assert!(named_state(None, &[], 7, None).is_err());
        assert_eq!(named_state(set_a, &[], 7, None), Ok(0xa));
        // A span from before a dropped state's release, of a history that dropped older ones.
        let kept = history(&[(0xa, 10, 18), (0xb, 20, LIVE)], 10);
        assert!(named_state(None, &[(5, 12)], 7, Some(&kept)).is_err());
        assert_eq!(named_state(None, &[(21, 30)], 7, Some(&kept)), Ok(0xb));
        let dropped = history(&[(0xa, 10, LIVE)], 10);
        assert!(named_state(None, &[], 7, Some(&dropped)).is_err());
        // A state placed at the ID of one still live holds it beside that one: a span after both
        // were made names both, and core's set no longer tells.
        let placed = history(&[(0xa, 10, LIVE), (0xb, 20, LIVE)], 0);
        assert!(named_state(None, &[(21, 30)], 7, Some(&placed)).is_err());
        assert!(named_state(set_a, &[(21, 30)], 7, Some(&placed)).is_err());
        assert_eq!(named_state(set_a, &[], 7, Some(&placed)), Ok(0xa));
    }

    /// A command a kernel may have encoded after a state made later took the resource ID of the
    /// one its argument buffer held, released while the dispatch waited to be committed, dies:
    /// its record names the ID, which either state held while the GPU may have written it
    #[test]
    fn a_command_written_while_two_states_held_its_id_dies() {
        use reims_vgpu_core::platform::resource::{ComputePipeline as _, Library as _};
        let d = dev();
        let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
        let (mut computes, renders) = twin_states::<VulkanPlatform>(&d, &library);
        let icb = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        let id = computes[1].gpu_resource_id();
        let f = library
            .new_function(OnFault::Return, "set_compute_state")
            .unwrap()
            .unwrap();
        let kernel = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let held = d
            .new_buffer_of_length(OnFault::Return, 16, 0)
            .unwrap()
            .unwrap();
        let words = [icb.gpu_resource_id(), id];
        // SAFETY: the buffer's 16 Shared bytes, written before any command uses them.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), held.contents().as_ptr::<u64>(), 2)
        };
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let buffers = [Some(&held)];
        let resident = [Some(ResourceRef::Icb(&icb))];
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&kernel),
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::UseResources {
                resources: &resident,
                usage: 3,
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        computes.truncate(1);
        let (later, later_renders) = twin_states::<VulkanPlatform>(&d, &library);
        assert!(later.iter().any(|state| state.gpu_resource_id() == id));
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let site = catch_death(|| {
            icb.compute_command(0)
                .unwrap()
                .unwrap()
                .pipeline_state_unique_identifier()
        })
        .unwrap_err();
        let at = (site.kind, site.file);
        assert_eq!(at, (DeathKind::Abort, "getPipelineStateUniqueIdentifier"));
        drop((renders, later_renders));
    }

    /// A state of `mark` of `library`, made from a guest descriptor that supports indirect
    /// command buffers (compute key 7).
    fn marking_state<P: Platform>(d: &P::Device, library: &P::Library) -> P::ComputePipeline {
        use crate::gpu::serialized::tests::hex;
        use reims_vgpu_core::platform::resource::Library as _;
        use reims_vgpu_core::rt::Hold;
        let mark = library
            .new_function(OnFault::Return, "mark")
            .unwrap()
            .unwrap();
        let compute = hex("02 07 04 01000000 00 04 05000000");
        let mut resolve = |_: Hold<'_>, r: u64| Ok((r == 5).then(|| mark.clone_ref()));
        let desc = d
            .deserialize_compute_pipeline_descriptor(Hold::NONE, &compute, &mut resolve)
            .unwrap()
            .unwrap();
        d.new_compute_pipeline_state(Some(&desc), 0)
            .unwrap()
            .unwrap()
            .unwrap()
    }

    /// Commits, on `d`, an execution of `icb`'s command 0, which writes `out`, and waits for it.
    fn execute_writing<P: Platform>(d: &P::Device, icb: &P::Icb, out: &P::Buffer) {
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Icb(icb))],
                usage: 1,
            },
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(out))],
                usage: 3,
            },
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: icb,
                execution_range: NsRange::new(0, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
    }

    /// An execution of a command a kernel encoded with a state the program released since, whose
    /// ID nothing took, dies where the runtime announces it, at the host step that reads the
    /// record the GPU wrote, naming its notice (`ReleasedCommandState`,
    /// [`crate::answer::DIES`]): the runtime runs nothing of it, where Metal runs the released
    /// state's program. The host step runs on the scheduler's thread, outside every call here,
    /// where the death ends the process; the test capture's death there unwinds into the
    /// runtime, which catches it, so the test reads the death's line
    #[test]
    fn an_execution_of_a_record_the_gpu_wrote_naming_a_released_state_dies() {
        let d = dev();
        let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
        let state = marking_state::<VulkanPlatform>(&d, &library);
        let out = d
            .new_buffer_of_length(OnFault::Return, 16, 0)
            .unwrap()
            .unwrap();
        let icb = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        let command = icb.compute_command(0).unwrap().unwrap();
        command
            .encode(IcbComputeCmd::SetKernelBuffer {
                buffer: Some(&out),
                offset: 0,
                index: 0,
            })
            .unwrap();
        let id = state.gpu_resource_id();
        encode_by_kernel::<VulkanPlatform>(&d, &library, "set_compute_state", &icb, id);
        drop(state);
        *crate::answer::LAST_LINE.lock() = None;
        let ran = catch_death(|| execute_writing::<VulkanPlatform>(&d, &icb, &out));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert!(
            ran.is_ok() && line.contains("(metal2vulkan-degrade-released-command-state)"),
            "{ran:?}: {line}"
        );
        // SAFETY: the buffer's first Shared word, read after the execution completed.
        assert_eq!(unsafe { *out.contents().as_ptr::<u32>() }, 0, "{ran:?}");
    }

    /// A state the guest released while a command core set it on still names it goes, as Metal's
    /// does: a state made after it, whose identifier Metal answers as the released one's
    /// (measured), dies where the guest reads it (info 460), as every state made with indirect
    /// command buffer support does, and so does the command's getter, by whose answer core's map
    /// would reach the released state where Metal's reaches the later
    #[test]
    fn a_state_a_command_names_goes_at_the_guests_release() {
        use reims_vgpu_core::platform::resource::ComputePipeline as _;
        let d = dev();
        let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
        let (mut computes, renders) = twin_states::<VulkanPlatform>(&d, &library);
        let icb = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        let command = icb.compute_command(0).unwrap().unwrap();
        command
            .encode(IcbComputeCmd::SetComputePipelineState {
                pipeline_state: Some(&computes[1]),
            })
            .unwrap();
        computes.truncate(1);
        let (later, later_renders) = twin_states::<VulkanPlatform>(&d, &library);
        drop((renders, later_renders));
        let died = |site: reims_vgpu_core::raise::death::DeathSite| (site.kind, site.file);
        let read = catch_death(|| later[0].guest_unique_identifier()).unwrap_err();
        assert_eq!(died(read), (DeathKind::Abort, "uniqueIdentifier"));
        let answer = catch_death(|| command.pipeline_state_unique_identifier()).unwrap_err();
        assert_eq!(
            died(answer),
            (DeathKind::Abort, "getPipelineStateUniqueIdentifier")
        );
    }

    /// An execution range past the buffer's size, which Metal runs without validation, dies at
    /// its selector, naming the runtime's refusal
    #[test]
    fn an_execution_past_the_size_dies() {
        let d = dev();
        let icb = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let site = catch_death(|| {
            e.encode(
                OnFault::Return,
                ComputeCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: &icb,
                    execution_range: NsRange::new(0, 2),
                },
            )
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "executeCommandsInBuffer:withRange:")
        );
    }

    /// The copy's selector.
    const COPY: &str = "copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:";

    /// Buffers of `n` threads dispatches each on `d`, of 10 threads up, 20 up and 30 up (draws of
    /// that many vertices where `render`): a source, a destination and a neighbour made after it.
    fn three_buffers<P: Platform>(d: &P::Device, render: bool, n: u64) -> [P::Icb; 3] {
        let desc = if render {
            render_desc()
        } else {
            compute_desc()
        };
        let buffers = [(); 3].map(|()| make::<P>(d, &desc, n));
        for (icb, base) in buffers.iter().zip([10, 20, 30]) {
            for i in 0..n {
                if render {
                    let draw = IcbRenderCmd::DrawPrimitives {
                        primitive_type: 3,
                        vertex_start: 0,
                        vertex_count: base + i,
                        instance_count: 1,
                        base_instance: 0,
                    };
                    icb.render_command(i)
                        .unwrap()
                        .unwrap()
                        .encode(draw)
                        .unwrap();
                } else {
                    let dispatch = IcbComputeCmd::ConcurrentDispatchThreads {
                        threads_per_grid: MtlSize::new(base + i, 1, 1),
                        threads_per_threadgroup: MtlSize::new(1, 1, 1),
                    };
                    icb.compute_command(i)
                        .unwrap()
                        .unwrap()
                        .encode(dispatch)
                        .unwrap();
                }
            }
        }
        buffers
    }

    /// Command `i` of `icb` as its type and width: a dispatch's threads or threadgroups across, a
    /// draw's vertex count, and `u64::MAX` for a command of no draw or dispatch (type 0, a reset
    /// one), whose arguments core never asks for (a getter of another kind's dies here).
    fn kind_and_width<P: Platform>(icb: &P::Icb, render: bool, i: u64) -> (u64, u64) {
        if render {
            let c = icb.render_command(i).unwrap().unwrap();
            let width = match c.command_type() {
                0 => None,
                _ => c.draw_args().map(|a| a.vertex_count),
            };
            (c.command_type(), width.unwrap_or(u64::MAX))
        } else {
            let c = icb.compute_command(i).unwrap().unwrap();
            let width = match c.command_type() {
                0 => None,
                0x40 => c.dispatch_threads_args().map(|a| a.threads_per_grid.width),
                _ => c
                    .dispatch_threadgroups_args()
                    .map(|a| a.threadgroups_per_grid.width),
            };
            (c.command_type(), width.unwrap_or(u64::MAX))
        }
    }

    /// `blit` of `cmds` on a new command buffer of `d`'s, committed and run.
    fn blits<P: Platform>(d: &P::Device, cmds: Vec<BlitCmd<'_, P>>) -> P::CommandBuffer {
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in cmds {
            blit.encode(OnFault::Return, cmd).unwrap();
        }
        blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        cb
    }

    /// `k` commands of `source` copied to index `to` of `destination`.
    fn copy_to<'a, P: Platform>(
        source: &'a P::Icb,
        k: u64,
        destination: &'a P::Icb,
        to: u64,
    ) -> BlitCmd<'a, P> {
        BlitCmd::CopyIndirectCommandBuffer {
            source,
            source_range: NsRange::new(0, k),
            destination,
            destination_index: to,
        }
    }

    /// What a copy past the end may have written over stays so until its whole record is
    /// written again: an execution or a copy that reaches it dies, and an execution of the
    /// commands that landed does not; a reset of the command on the CPU, a blit's reset of it and
    /// a blit's copy into it make it answer again. Until its command buffer commits, an execution
    /// of one, which Metal runs after the copy, dies; and a copy past the end while another
    /// command buffer that has not run yet executes the buffer dies at the copy
    #[test]
    fn what_a_copy_past_the_end_overwrote_dies_until_written_again() {
        const EXECUTE: &str = "executeCommandsInBuffer:withRange:";
        let d = dev();
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 4);
        blits::<VulkanPlatform>(&d, vec![copy_to(&a, 2, &b, 3)]);
        let q = d.new_command_queue().unwrap();
        let executes = |range: NsRange| {
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
            let execution = ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &b,
                execution_range: range,
            };
            e.encode(OnFault::Return, execution).unwrap();
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        };
        for range in [NsRange::new(0, 1), NsRange::new(2, 2)] {
            let site = catch_death(|| executes(range)).unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, EXECUTE),
                "{range:?}"
            );
        }
        executes(NsRange::new(3, 1));
        let other = make::<VulkanPlatform>(&d, &compute_desc(), 4);
        let site = catch_death(|| blits::<VulkanPlatform>(&d, vec![copy_to(&b, 1, &other, 0)]));
        assert_eq!(site.unwrap_err().file, COPY);
        // Each way the whole record is written again.
        let command = b.compute_command(0).unwrap().unwrap();
        command.encode(IcbComputeCmd::Reset).unwrap();
        assert_eq!(command.command_type(), 0);
        let reset = BlitCmd::ResetCommandsInBuffer {
            buffer: &b,
            range: NsRange::new(1, 1),
        };
        let copied = BlitCmd::CopyIndirectCommandBuffer {
            source: &a,
            source_range: NsRange::new(3, 1),
            destination: &b,
            destination_index: 2,
        };
        blits::<VulkanPlatform>(&d, vec![reset, copied]);
        assert_eq!(kind_and_width::<VulkanPlatform>(&b, false, 1).0, 0);
        assert_eq!(kind_and_width::<VulkanPlatform>(&b, false, 2), (0x40, 13));
        executes(NsRange::new(0, 4));
        // Before the commit: an execution after the copy, in its command buffer.
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 2);
        let site = catch_death(|| {
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            blit.encode(OnFault::Return, copy_to(&a, 2, &b, 1)).unwrap();
            blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
            let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
            let execution = ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &b,
                execution_range: NsRange::new(0, 1),
            };
            e.encode(OnFault::Return, execution).unwrap();
        });
        assert_eq!(site.unwrap_err().file, EXECUTE);
        // Another command buffer, not run yet, executes the buffer.
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 2);
        let pending = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = pending.compute_encoder(OnFault::Return).unwrap().unwrap();
        let execution = ComputeCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &b,
            execution_range: NsRange::new(0, 2),
        };
        e.encode(OnFault::Return, execution).unwrap();
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        let site = catch_death(|| blits::<VulkanPlatform>(&d, vec![copy_to(&a, 2, &b, 1)]));
        assert_eq!(site.unwrap_err().file, COPY);
        drop(pending);
    }

    /// A `useResource` form that makes a buffer resident while a copy past its end may have
    /// written over one of its records dies, as a function of the encoder may copy that command
    /// (`copy_command`), where this platform's copy would carry the intact record: in a compute
    /// encoder at each usage and in a render encoder, once the copy completed, and after the copy
    /// in its command buffer, before the commit. A copy past the end while another command buffer
    /// that has not run yet makes the buffer resident dies at the copy. A buffer whose records are
    /// intact is made resident
    #[test]
    fn a_buffer_a_copy_past_its_end_overwrote_dies_where_made_resident() {
        use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
        const USE: &str = "useResources:count:usage:";
        let d = dev();
        let p = crate::platform::tests::platform();
        let q = d.new_command_queue().unwrap();
        /// `icb` made resident in `e` with `usage`.
        fn use_in(e: &<VulkanPlatform as Platform>::ComputeEncoder, icb: &VulkanIcb, usage: u64) {
            let resources = [Some(ResourceRef::Icb(icb))];
            let used = ComputeCmd::UseResources {
                resources: &resources,
                usage,
            };
            e.encode(OnFault::Return, used).unwrap();
        }
        let resident = |icb: &VulkanIcb, usage| {
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
            use_in(&e, icb, usage);
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        };
        let [a, b, c] = three_buffers::<VulkanPlatform>(&d, false, 2);
        resident(&c, 1);
        blits::<VulkanPlatform>(&d, vec![copy_to(&a, 2, &b, 1)]);
        for usage in [1, 2, 3] {
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| resident(&b, usage)).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, USE),
                "{usage}: {line}"
            );
            assert!(line.contains("copy_command"), "{line}");
        }
        let (_target, pass) = crate::gpu::encode_render::tests::pass_of(&p, &d, (4, 4));
        let site = catch_death(|| {
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
            let resources = [Some(ResourceRef::Icb(&b))];
            let used = RenderCmd::UseResources {
                resources: &resources,
                usage: 1,
            };
            e.encode(OnFault::Return, used).unwrap();
        });
        assert_eq!(site.unwrap_err().file, USE);
        // After the copy in its command buffer, before the commit.
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 2);
        let site = catch_death(|| {
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            blit.encode(OnFault::Return, copy_to(&a, 2, &b, 1)).unwrap();
            blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
            use_in(
                &cb.compute_encoder(OnFault::Return).unwrap().unwrap(),
                &b,
                1,
            );
        });
        assert_eq!(site.unwrap_err().file, USE);
        // Another command buffer, not run yet, makes the buffer resident.
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 2);
        let pending = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = pending.compute_encoder(OnFault::Return).unwrap().unwrap();
        use_in(&e, &b, 1);
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        let site = catch_death(|| blits::<VulkanPlatform>(&d, vec![copy_to(&a, 2, &b, 1)]));
        assert_eq!(site.unwrap_err().file, COPY);
        drop(pending);
    }

    /// A copy past the destination's end in a shape nothing measured dies at the copy: more
    /// commands than the destination holds, an index at its end, more than 63 commands past it,
    /// and a buffer into itself; and, each in an otherwise measured shape, a destination of 65
    /// commands, one of 3 kernel buffers, one of Private storage, a render buffer's commands
    /// into a compute buffer, and a source of 64 commands into a destination of 4
    #[test]
    fn copies_past_the_end_in_unmeasured_shapes_die() {
        let d = dev();
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 4);
        let [big, ..] = three_buffers::<VulkanPlatform>(&d, false, 128);
        let [wide, ..] = three_buffers::<VulkanPlatform>(&d, false, 128);
        let [render, ..] = three_buffers::<VulkanPlatform>(&d, true, 4);
        let long = make::<VulkanPlatform>(&d, &compute_desc(), 65);
        let sixty_four = make::<VulkanPlatform>(&d, &compute_desc(), 64);
        let three = IcbDesc {
            max_kernel_buffer_bind_count: 3,
            ..compute_desc()
        };
        let wider = make::<VulkanPlatform>(&d, &three, 4);
        let private = d
            .new_indirect_command_buffer(&compute_desc(), 4, 0x20)
            .unwrap()
            .unwrap();
        let cases: [BlitCmd<'_, VulkanPlatform>; 9] = [
            copy_to(&big, 5, &b, 0),
            copy_to(&a, 1, &b, 4),
            copy_to(&big, 65, &wide, 127),
            copy_to(&b, 2, &b, 3),
            copy_to(&a, 2, &long, 64),
            copy_to(&a, 2, &wider, 3),
            copy_to(&a, 2, &private, 3),
            copy_to(&render, 2, &b, 3),
            copy_to(&sixty_four, 2, &b, 3),
        ];
        for cmd in cases {
            let what = format!("{cmd:?}");
            let site = catch_death(|| blits::<VulkanPlatform>(&d, vec![cmd])).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, COPY), "{what}");
        }
    }

    /// A way a command buffer uses an indirect command buffer.
    #[derive(Debug, Clone, Copy)]
    enum Use {
        Execution,
        Resident,
        CopyFrom,
        CopyTo,
        Optimize,
        Reset,
    }

    /// Encodes `how` on `cb`, of `icb`, a copy's other buffer `other`, and an execution's
    /// `pipeline` and `out`.
    fn encode_use(
        cb: &VulkanCommandBuffer,
        how: Use,
        icb: &VulkanIcb,
        other: &VulkanIcb,
        pipeline: &VulkanComputePipeline,
        out: &VulkanBuffer,
    ) {
        let one = NsRange::new(0, 1);
        let blit = match how {
            Use::Execution | Use::Resident => None,
            Use::CopyFrom => Some(BlitCmd::CopyIndirectCommandBuffer {
                source: icb,
                source_range: one,
                destination: other,
                destination_index: 0,
            }),
            Use::CopyTo => Some(BlitCmd::CopyIndirectCommandBuffer {
                source: other,
                source_range: one,
                destination: icb,
                destination_index: 0,
            }),
            Use::Optimize => Some(BlitCmd::OptimizeIndirectCommandBuffer {
                indirect_command_buffer: icb,
                range: one,
            }),
            Use::Reset => Some(BlitCmd::ResetCommandsInBuffer {
                buffer: icb,
                range: one,
            }),
        };
        if let Some(cmd) = blit {
            let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            e.encode(OnFault::Return, cmd).unwrap();
            e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
            return;
        }
        if let Use::Resident = how {
            let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
            e.encode(
                OnFault::Return,
                ComputeCmd::UseResources {
                    resources: &[Some(ResourceRef::Icb(icb))],
                    usage: 1,
                },
            )
            .unwrap();
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
            return;
        }
        let c = icb.compute_command(0).unwrap().unwrap();
        for cmd in [
            IcbComputeCmd::SetComputePipelineState {
                pipeline_state: Some(pipeline),
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(out),
                offset: 0,
                index: 0,
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(out),
                offset: 0,
                index: 1,
            },
            IcbComputeCmd::ConcurrentDispatchThreads {
                threads_per_grid: MtlSize::new(4, 1, 1),
                threads_per_threadgroup: MtlSize::new(4, 1, 1),
            },
        ] {
            c.encode(cmd).unwrap();
        }
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(out))],
                usage: 3,
            },
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: icb,
                execution_range: one,
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
    }

    /// A buffer core released while a command buffer that uses it may still run keeps its ID and
    /// its range until that command buffer completes where an encoder executes it or makes it
    /// resident, only its range where a blit copies from it or into it or resets it, and neither
    /// where a blit optimizes it, as Metal's do; an execution in a command buffer never committed
    /// keeps them until the command buffer goes. The range is read as [`Space`] holds it: the
    /// `uniqueIdentifier` of a buffer made after an encoder dies ([`encoder_made`]), as it does
    /// here once the first command buffer used one
    #[test]
    fn a_buffer_keeps_its_numbers_while_a_command_buffer_uses_it() {
        use std::sync::mpsc;

        use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};

        let d = dev();
        let pipeline = icb_pipeline(&d);
        let desc = IcbDesc {
            command_types: 0x40,
            max_kernel_buffer_bind_count: 2,
            ..IcbDesc::default()
        };
        let out = holding(&d, &[0; 4]);
        let q = d.new_command_queue().unwrap();
        let numbers = |b: &VulkanIcb| (b.gpu_resource_id(), b.numbers.address);
        for (how, keeps_id, keeps_range) in [
            (Use::Execution, true, true),
            (Use::Resident, true, true),
            (Use::CopyFrom, false, true),
            (Use::CopyTo, false, true),
            (Use::Optimize, false, false),
            (Use::Reset, false, true),
        ] {
            let icb = make::<VulkanPlatform>(&d, &desc, 1);
            let other = make::<VulkanPlatform>(&d, &desc, 1);
            let old = numbers(&icb);
            let event = d.new_shared_event().unwrap();
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
            encode_use(&cb, how, &icb, &other, &pipeline, &out);
            let (tx, rx) = mpsc::channel();
            cb.add_completed_handler(OnFault::Return, Box::new(move |_| tx.send(()).unwrap()))
                .unwrap();
            cb.commit(OnFault::Return).unwrap();
            drop(icb);
            let during = numbers(&make::<VulkanPlatform>(&d, &desc, 1));
            assert_eq!(
                (during.0 != old.0, during.1 != old.1),
                (keeps_id, keeps_range),
                "{how:?}"
            );
            event.set_signaled_value(1).unwrap();
            rx.recv().unwrap();
            assert_eq!((cb.status().0, cb.error()), (COMPLETED, None), "{how:?}");
            let after = numbers(&make::<VulkanPlatform>(&d, &desc, 1));
            assert_eq!(after, old, "{how:?}");
        }
        let icb = make::<VulkanPlatform>(&d, &desc, 1);
        let other = make::<VulkanPlatform>(&d, &desc, 1);
        let old = numbers(&icb);
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        encode_use(&cb, Use::Execution, &icb, &other, &pipeline, &out);
        drop(icb);
        assert_ne!(numbers(&make::<VulkanPlatform>(&d, &desc, 1)).0, old.0);
        drop(cb);
        assert_eq!(numbers(&make::<VulkanPlatform>(&d, &desc, 1)), old);
        let after = make::<VulkanPlatform>(&d, &desc, 1);
        let site = catch_death(|| after.unique_identifier()).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "uniqueIdentifier")
        );
    }

    /// A blit's reset and copy change what Metal answers beside the records once the command
    /// buffer completed: before the commit the getters answer what stands, while it may run a
    /// getter of what a blit writes dies, as does a setter of what one reaches, and a getter of
    /// a copy's source answers. What the records hold, a barrier, answers as they stand, as the
    /// GPU left them
    #[test]
    fn blits_move_what_is_kept_beside_the_records_once_they_ran() {
        use std::sync::mpsc;

        use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};

        let d = dev();
        let (a, b) = (
            make::<VulkanPlatform>(&d, &render_desc(), 2),
            make::<VulkanPlatform>(&d, &render_desc(), 2),
        );
        let command = |icb: &VulkanIcb, i| icb.render_command(i).unwrap().unwrap();
        let stride = |stride| IcbRenderCmd::SetVertexBufferAttributeStride {
            buffer: None,
            offset: 0,
            stride,
            index: 0,
        };
        for i in [0, 1] {
            command(&a, i).encode(IcbRenderCmd::SetBarrier).unwrap();
            command(&a, i).encode(stride(20)).unwrap();
        }
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyIndirectCommandBuffer {
                source: &a,
                source_range: NsRange::new(0, 1),
                destination: &b,
                destination_index: 1,
            },
            BlitCmd::ResetCommandsInBuffer {
                buffer: &a,
                range: NsRange::new(1, 1),
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Return, cmd).unwrap();
        }
        let barriers = |icb: &VulkanIcb| [0, 1].map(|i| command(icb, i).has_barrier());
        let strides =
            |icb: &VulkanIcb| [0, 1].map(|i| command(icb, i).vertex_attribute_stride(0).unwrap());
        assert_eq!((barriers(&a), barriers(&b)), ([true, true], [false, false]));
        assert_eq!((strides(&a), strides(&b)), ([20, 20], [0, 0]));
        let (tx, rx) = mpsc::channel();
        cb.add_completed_handler(OnFault::Return, Box::new(move |_| tx.send(()).unwrap()))
            .unwrap();
        cb.commit(OnFault::Return).unwrap();
        for (icb, i, write) in [(&a, 1, false), (&b, 1, false), (&a, 0, true)] {
            let site = catch_death(|| {
                let c = command(icb, i);
                if write {
                    c.encode(stride(24)).unwrap();
                } else {
                    c.vertex_attribute_stride(0).unwrap();
                }
            })
            .unwrap_err();
            let sel = if write {
                "setVertexBuffer:offset:attributeStride:atIndex:"
            } else {
                "getVertexAttributeStrideAtIndex:"
            };
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, sel),
                "{i} {write}"
            );
        }
        assert_eq!(
            (
                command(&a, 0).vertex_attribute_stride(0).unwrap(),
                command(&b, 0).vertex_attribute_stride(0).unwrap()
            ),
            (20, 0)
        );
        assert_eq!((barriers(&a), barriers(&b)), ([true, true], [false, false]));
        event.set_signaled_value(1).unwrap();
        rx.recv().unwrap();
        assert_eq!((barriers(&a), barriers(&b)), ([true, false], [false, true]));
        assert_eq!((strides(&a), strides(&b)), ([20, 0], [0, 20]));
        command(&a, 0).encode(stride(24)).unwrap();
        assert_eq!(command(&a, 0).vertex_attribute_stride(0).unwrap(), 24);
    }

    /// Once the command buffer of a blit's copy and reset failed, which may or may not have run
    /// them, the getter of each value they would have changed dies, until core sets it again or
    /// resets the command; a value they would have left as it stands answers it, and so do a
    /// copy's source and the values and commands they do not reach
    #[test]
    fn a_failed_blit_leaves_what_it_would_change_unknown_until_set() {
        use reims_vgpu_core::platform::types::CommandBufferStatus;

        let d = dev();
        let (a, b) = (
            make::<VulkanPlatform>(&d, &render_desc(), 2),
            make::<VulkanPlatform>(&d, &render_desc(), 2),
        );
        let command = |icb: &VulkanIcb, i| icb.render_command(i).unwrap().unwrap();
        let stride = |stride| IcbRenderCmd::SetVertexBufferAttributeStride {
            buffer: None,
            offset: 0,
            stride,
            index: 0,
        };
        for (icb, i) in [(&a, 0), (&a, 1), (&b, 0)] {
            command(icb, i).encode(stride(20)).unwrap();
        }
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyIndirectCommandBuffer {
                source: &a,
                source_range: NsRange::new(0, 2),
                destination: &b,
                destination_index: 0,
            },
            BlitCmd::ResetCommandsInBuffer {
                buffer: &a,
                range: NsRange::new(1, 1),
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit_failed(rt::CommitFailure::new(2));
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(cb.status(), CommandBufferStatus::ERROR);
        let read = |icb: &VulkanIcb, i, at| {
            catch_death(|| command(icb, i).vertex_attribute_stride(at).unwrap())
                .map_err(|site| (site.kind, site.file))
        };
        let unknown = Err((DeathKind::Abort, "getVertexAttributeStrideAtIndex:"));
        assert_eq!(
            [
                read(&a, 0, 0),
                read(&a, 1, 0),
                read(&b, 0, 0),
                read(&b, 1, 0)
            ],
            [Ok(20), unknown, Ok(20), unknown]
        );
        assert_eq!((read(&a, 1, 1), read(&b, 1, 1)), (Ok(0), Ok(0)));
        command(&b, 1).encode(stride(24)).unwrap();
        command(&a, 1).encode(IcbRenderCmd::Reset).unwrap();
        assert_eq!((read(&a, 1, 0), read(&b, 1, 0)), (Ok(0), Ok(24)));
    }

    /// What two commands would answer apart is unknown: each slot and length kept differently,
    /// one never set counting as 0, a pipeline state of another handle or of none, and what either
    /// does not know; a value set again is known
    #[test]
    fn what_commands_would_answer_apart_is_unknown() {
        let d = dev();
        let (p, q) = (icb_pipeline(&d), icb_pipeline(&d));
        let side = |stride, length, pipeline: Option<&VulkanComputePipeline>| {
            let mut side = CommandSide::default();
            side.slots
                .insert((Slots::Kernel, 0), SetSlot { address: 0, stride });
            side.lengths.insert(40, length);
            side.pipeline = pipeline.map(SetPipeline::compute);
            side
        };
        assert!(
            side(4, 64, Some(&p))
                .differs(&side(4, 64, Some(&p)))
                .is_empty()
        );
        assert!(side(0, 0, None).differs(&CommandSide::default()).is_empty());
        let apart = side(4, 64, Some(&p)).differs(&side(8, 32, Some(&q)));
        for kept in [
            Kept::Slot(Slots::Kernel, 0),
            Kept::Length(40),
            Kept::Pipeline,
        ] {
            assert!(apart.holds(kept), "{kept:?}");
        }
        for kept in [
            Kept::Slot(Slots::Kernel, 1),
            Kept::Slot(Slots::Vertex, 0),
            Kept::Length(0),
        ] {
            assert!(!apart.holds(kept), "{kept:?}");
        }
        let mut unknown = side(4, 64, None).differs(&side(4, 64, Some(&p)));
        assert!(unknown.holds(Kept::Pipeline) && unknown.holds(Kept::All));
        let mut joined = side(4, 64, None);
        joined.unknown = apart;
        assert!(joined.differs(&joined.copy()).holds(Kept::Length(40)));
        unknown.settle(Kept::Pipeline);
        assert!(unknown.is_empty());
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
