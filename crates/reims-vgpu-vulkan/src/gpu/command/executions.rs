//! The commands an execution of an indirect command buffer runs, checked at its command buffer's
//! commit where a direct command of their kind dies at its call.
//!
//! - **What is checked.** Two deaths of a direct command depend, for an execution, on the
//!   records of the commands it runs: their draws, and the pipeline states they name. The
//!   runtime runs the records as they stand at the commit (`crate::gpu::icb`), so the commit
//!   reads them (`IndirectCommandBuffer::commands`, of the execution's range, or of every
//!   command where a buffer holds the range) and dies, at `commit`, naming the execution and the
//!   command, where a direct command of its kind dies:
//!   - a draw of points, of some vertices and instances, under a state whose points Metal sizes
//!     none of ([`Points`], `crate::gpu::encode_render`'s `sized`);
//!   - a dispatch or draw under a state that can encode indirect commands, in an encoder that
//!     made resident for writing an indirect command buffer whose ID is not the runtime's
//!     ([`Execution::guest_id`], `crate::gpu::encode_compute`'s `dispatchable`).
//!
//!   Commands that inherit the encoder's state are decided at the execution's call where the
//!   state alone decides (`crate::gpu::encode_compute`, `crate::gpu::encode_render`). A command
//!   of no draw or dispatch, or of a record that names no state, runs nothing, and is not asked;
//!   nor is an execution in an encoder its command buffer's failure runs nothing of
//!   ([`rt::FailedEncoders`]). The state a command runs is the live state of the ID its record
//!   names, where that state's identifier is the one the record answers; else the runtime runs
//!   the state the program set on the command and released since, of which nothing here holds
//!   what decides, and a command a death would ask of dies.
//! - **Records the GPU may write** ([`gpu_writes`]): a blit's copy into a buffer or reset of
//!   it, and a command, direct or executed, whose state can encode indirect commands in an
//!   encoder that made the buffer resident for writing, may write its records as the GPU runs,
//!   and what it wrote may stay. Once the runtime's buffer says so (`gpu_written`, which each
//!   of those sets at its call, as does any encoder's making the buffer resident for writing),
//!   it records an execution committed since once the work before it completed (its host step),
//!   from the records as the GPU and the CPU left them, and so it does an execution whose range
//!   a buffer holds, and a compute encoder's execution under predication. The commit then reads
//!   nothing: an execution that a death could ask of hands the runtime a check at its call
//!   ([`Execution::host_step_check`]), which the host step runs on each command it will record,
//!   before it records any, and which dies there where the commit would have
//!   ([`ask`]). Where the commit reads the records, the runtime records them as the commit read
//!   them and never runs the check. An execution whose own commands may write its records dies
//!   at the commit. A write encoded while a committed command buffer that has not ended executes
//!   the buffer as its commit read it dies at its own call: it may land before those commands
//!   run.

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use metal2vulkan::metal::PrimitiveType;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::icb::IndirectCommandBuffer as _;
use reims_vgpu_core::sync::Mutex;

use crate::answer::{get, plain, unrepresentable};
use crate::gpu::command::Progress;
use crate::gpu::icb::{IcbObject, VulkanIcb};

/// Which states size the points an execution's commands draw ([`Execution::check`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Points {
    /// Not asked: a compute encoder's execution, or commands that inherit a state that sizes
    /// them, or no state.
    Unasked,
    /// The commands inherit the encoder's state, whose points Metal sizes none of: a draw of
    /// points dies.
    Unsized,
    /// Each command's own state decides ([`crate::gpu::states::points_sized`]).
    Own,
}

/// An execution of an indirect command buffer whose commands its commit reads (the module docs).
#[derive(Debug)]
pub(crate) struct Execution {
    /// Its selector, which a death names.
    pub(crate) sel: &'static str,
    /// The buffer it executes.
    pub(crate) icb: VulkanIcb,
    /// Its range; `None` for one a buffer holds, which the GPU reads as it runs, so every command
    /// of the buffer is asked.
    pub(crate) range: Option<Range<u64>>,
    /// Metal's ID of an indirect command buffer its encoder made resident for writing that is
    /// not the runtime's ID of it, where its commands name their own states: a command whose
    /// state can encode indirect commands dies.
    pub(crate) guest_id: Option<u64>,
    /// Which states size its draws of points.
    pub(crate) points: Points,
    /// The indirect command buffers its encoder made resident for writing, where its commands
    /// name their own states: a command whose state can encode indirect commands may write their
    /// records ([`gpu_writes`]).
    pub(crate) writable: Vec<VulkanIcb>,
}

impl Execution {
    /// Whether its commit reads its commands.
    pub(crate) fn asks(&self) -> bool {
        self.decides() || !self.writable.is_empty()
    }

    /// Whether its commands decide a death.
    fn decides(&self) -> bool {
        self.guest_id.is_some() || self.points != Points::Unasked
    }

    /// Checks its commands at the commit (the module docs): dies where one would die as a direct
    /// command, marks the buffers its commands may write, and answers the buffer whose records
    /// the commit read for a death, which [`reads`] then holds.
    pub(super) fn check(self) -> Option<VulkanIcb> {
        let decides = self.decides();
        let Execution {
            sel: execution,
            icb,
            range,
            guest_id,
            points,
            writable,
        } = self;
        if plain(|| icb.raw.gpu_written()) {
            // The host step reads its commands, and asks them where a death could
            // ([`Execution::host_step_check`]); here they are not known: any of them may write.
            writable.iter().for_each(|w| gpu_writes("commit", w));
            return None;
        }
        let range = range.unwrap_or(0..icb.size());
        let commands = get("commit", || icb.raw.commands(range.clone()));
        let mut writes = false;
        for (index, command) in (range.start..).zip(&commands) {
            writes |= ask(execution, index, command, (points, guest_id), true) == Some(true);
        }
        if writes && !writable.is_empty() {
            if decides && writable.iter().any(|w| same(w, &icb)) {
                unrepresentable(
                    "commit",
                    &format!(
                        "{execution} of an indirect command buffer its encoder made resident \
                         for writing, a command of which may write its records as the GPU runs \
                         them: what its later commands run is not what the commit read"
                    ),
                );
            }
            writable.iter().for_each(|w| gpu_writes("commit", w));
        }
        decides.then_some(icb)
    }

    /// The check the runtime's host step runs on each command it records of this execution,
    /// where its commit reads nothing (the module docs); `None` where no death could ask of its
    /// commands, which leaves the runtime its other paths. It dies where [`Execution::check`]
    /// would, at `commit`, on the scheduler's thread; it never unwinds into the runtime.
    pub(crate) fn host_step_check(&self) -> Option<rt::IndirectCommandCheck> {
        if !self.decides() {
            return None;
        }
        let (execution, points, guest_id) = (self.sel, self.points, self.guest_id);
        Some(Box::new(move |index, command| {
            let asked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ask(execution, index, command, (points, guest_id), false);
            }));
            // A death unwinds only under the tests' capture; the runtime is not unwound through.
            if asked.is_err() {
                std::process::abort();
            }
        }))
    }
}

/// Asks command `index` of `execution`'s indirect command buffer, `command`, of the death a direct
/// command of its kind dies of (the module docs), `points` saying which states size its draws
/// of points and `guest_id` the ID through which a state that can encode indirect commands would
/// write: dies where one would, at `commit`. Where `encodes` is asked (the commit's), answers
/// whether its state may encode indirect commands (one released since may); `None` for a
/// command of no draw or dispatch, or of no state, which runs nothing, and wherever `encodes` is
/// not asked and the answer decides no death. A command's state is looked at only where a death
/// or the answer needs it: a host step asks each of tens of thousands of commands a frame.
fn ask(
    execution: &str,
    index: u64,
    command: &rt::IndirectCommand,
    (points, guest_id): (Points, Option<u64>),
    encodes: bool,
) -> Option<bool> {
    let at = || format!("command {index} of {execution}'s indirect command buffer");
    let answered = encodes || guest_id.is_some();
    let (named, draw) =
        match command {
            rt::IndirectCommand::Compute(c) if c.dispatch.is_some() => {
                if c.pipeline_id.is_none() || !answered {
                    return None;
                }
                let state = c.pipeline.as_ref().filter(|s| {
                    Some(plain(|| s.unique_identifier())) == c.pipeline_unique_identifier
                });
                (
                    state.map(|s| plain(|| s.can_encode_indirect_commands())),
                    None,
                )
            }
            rt::IndirectCommand::Render(r) if r.draw.is_some() => {
                let of_points = points != Points::Unasked && points_drawn(r.draw);
                if points == Points::Unsized && of_points {
                    dies_unsized(&at());
                }
                let sized_asked = points == Points::Own && of_points;
                if r.pipeline_id.is_none() || !(answered || sized_asked) {
                    return None;
                }
                let state = r.pipeline.as_ref().filter(|s| {
                    Some(plain(|| s.unique_identifier())) == r.pipeline_unique_identifier
                });
                if sized_asked {
                    match state.map(crate::gpu::states::points_sized) {
                        Some(Some(true)) => {}
                        Some(Some(false)) => dies_unsized(&at()),
                        Some(None) | None => unrepresentable(
                            "commit",
                            &format!(
                                "{}, a draw of points under a pipeline state the program released, \
                             which the runtime runs: its input primitive topology class, which \
                             decides whether Metal sizes its points, went with it",
                                at()
                            ),
                        ),
                    }
                }
                if !answered {
                    return None;
                }
                (
                    state.map(|s| plain(|| s.can_encode_indirect_commands())),
                    r.draw,
                )
            }
            _ => return None,
        };
    // `None`: a state released since, whose functions nothing here holds.
    let may_encode = named != Some(false);
    if let Some(id) = guest_id
        && may_encode
    {
        let what = format!(
            "{}, a {} under a pipeline state that can encode indirect commands, or one released \
             since, which the runtime runs in its place",
            at(),
            if draw.is_some() { "draw" } else { "dispatch" },
        );
        crate::gpu::encode_compute::encodes_through_guest_id("commit", &what, id);
    }
    Some(may_encode)
}

/// Whether `draw` is of points, of some vertices or indices and instances.
fn points_drawn(draw: Option<rt::IndirectDraw>) -> bool {
    match draw {
        Some(rt::IndirectDraw::Primitives {
            primitive_type: PrimitiveType::Point,
            vertex_count,
            instance_count,
            ..
        }) => vertex_count != 0 && instance_count != 0,
        Some(rt::IndirectDraw::IndexedPrimitives {
            primitive_type: PrimitiveType::Point,
            index_count,
            instance_count,
            ..
        }) => index_count != 0 && instance_count != 0,
        _ => false,
    }
}

/// Dies at `commit` at `at`, a draw of points under a state whose points Metal sizes none of.
fn dies_unsized(at: &str) -> ! {
    unrepresentable(
        "commit",
        &format!(
            "{at}, a draw of {}",
            crate::gpu::encode_render::UNSIZED_POINTS
        ),
    )
}

/// What the commits know of an indirect command buffer's records.
#[derive(Debug, Default)]
struct Records {
    /// How many committed command buffers that have not ended execute them as their commit read
    /// them ([`reads`]).
    readers: AtomicUsize,
}

/// The [`Records`] of each live indirect command buffer one was asked of.
static RECORDS: Mutex<Vec<(Weak<IcbObject>, Arc<Records>)>> = Mutex::new(Vec::new());

/// Whether `a` and `b` are one buffer.
fn same(a: &VulkanIcb, b: &VulkanIcb) -> bool {
    std::ptr::eq::<IcbObject>(&**a, &**b)
}

/// The [`Records`] of `icb`, added where it had none.
fn records(icb: &VulkanIcb) -> Arc<Records> {
    plain(|| {
        let mut records = RECORDS.lock();
        records.retain(|(weak, _)| weak.strong_count() > 0);
        let at = records
            .iter()
            .find(|(weak, _)| std::ptr::eq(weak.as_ptr(), &**icb));
        if let Some((_, r)) = at {
            return Arc::clone(r);
        }
        let r = Arc::<Records>::default();
        records.push((icb.downgrade(), Arc::clone(&r)));
        r
    })
}

/// A command encoded at `sel` may write `icb`'s records on the GPU (the module docs). It dies
/// where a committed command buffer that has not ended executes them as its commit read them
/// ([`reads`]): the write may land before those run.
pub(crate) fn gpu_writes(sel: &'static str, icb: &VulkanIcb) {
    let r = records(icb);
    if r.readers.load(Ordering::SeqCst) != 0 {
        unrepresentable(
            sel,
            "a write on the GPU of the records of an indirect command buffer that a committed \
             command buffer that has not ended executes, whose commands its commit read and \
             checked: the write may land before they run, behind an event or on another queue, \
             and what they run then is not checked",
        );
    }
}

/// A committed command buffer of `progress` executes `icb` as its commit read its records,
/// until it ends ([`gpu_writes`]).
pub(super) fn reads(icb: &VulkanIcb, progress: &Progress) {
    let r = records(icb);
    r.readers.fetch_add(1, Ordering::SeqCst);
    progress.hold(Read(r));
}

/// A reader of [`Records`], which ends as its command buffer's progress lets go of it.
struct Read(Arc<Records>);

impl Drop for Read {
    fn drop(&mut self) {
        self.0.readers.fetch_sub(1, Ordering::SeqCst);
    }
}
