//! Which pages of a task the host has mapped (the blocks setup): page runs, sorted, in up to
//! 128 blocks of up to 256 runs each.
//!
//! The structure and its edges show through to the host and the guest, so they are kept exactly:
//! adds stop filling a block at 240 runs while `remove` can grow one to 256; a full tracker
//! compacts, then splits, then raises; the binary searches return the last probe and scans go
//! forward from it. Every `u32` sum wraps; the searches are an `i32` loop, never
//! `slice::binary_search`.

use crate::guest::records::GuestPages;
use crate::memory::{DeathKind, RaiseKind};
use crate::num::{ToUsize, Truncate};
use crate::raise::RaiseResult;
use crate::sync::WriterPrefRwLock;

/// Block slots in a tracker.
const MAX_BLOCKS: u32 = 128;
/// Run slots in a block.
const RUNS_PER_BLOCK: u32 = 256;
/// A block holding this many runs takes no more adds, and compaction fills blocks to it.
const FULL_BLOCK: u32 = 240;
/// The unmapped runs a visit can hold.
const UNMAPPED_RUNS: usize = 16;

/// What [`Run::compare`] says about a run against another (values -3..6). Only the name
/// `kCmpLessAdjacent` is attested (an assert's text); the other names are ours, after the case each
/// value covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    /// -3: self ends strictly before other starts.
    Less,
    /// -2: self ends exactly where other starts.
    LessAdjacent,
    /// -1: self starts at or before other and ends inside it.
    LessOverlap,
    /// 0.
    Equal,
    /// 1: strictly inside, touching neither edge.
    Inside,
    /// 2: inside, sharing an edge.
    InsideEdge,
    /// 3: other is strictly inside self.
    Contains,
    /// 4: self starts inside other and ends at or after it.
    GreaterOverlap,
    /// 5: self starts exactly where other ends.
    GreaterAdjacent,
    /// 6: self starts strictly after other ends.
    Greater,
}

impl Cmp {
    /// The integer value (the split's assert orders by it).
    pub const fn value(self) -> i32 {
        match self {
            Cmp::Less => -3,
            Cmp::LessAdjacent => -2,
            Cmp::LessOverlap => -1,
            Cmp::Equal => 0,
            Cmp::Inside => 1,
            Cmp::InsideEdge => 2,
            Cmp::Contains => 3,
            Cmp::GreaterOverlap => 4,
            Cmp::GreaterAdjacent => 5,
            Cmp::Greater => 6,
        }
    }
}

/// A page run `{page_offset, page_length}`, `u32`, wrapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Run {
    /// The first page.
    pub page_offset: u32,
    /// The pages.
    pub page_length: u32,
}

impl Run {
    /// No pages.
    pub const fn is_empty(&self) -> bool {
        self.page_length == 0
    }

    const fn end(&self) -> u32 {
        self.page_offset.wrapping_add(self.page_length)
    }

    /// Where this run lies against `other` (the order of the tests decides).
    pub fn compare(&self, other: &Run) -> Cmp {
        if self.page_offset == other.page_offset && self.page_length == other.page_length {
            return Cmp::Equal;
        }
        // An emptied block's {0,0} hull compares Greater.
        if self.page_offset == 0 && self.page_length == 0 {
            return Cmp::Greater;
        }
        let end = self.end();
        if end < other.page_offset {
            return Cmp::Less;
        }
        if end == other.page_offset {
            return Cmp::LessAdjacent;
        }
        let other_end = other.end();
        if self.page_offset >= other.page_offset && end <= other_end {
            if self.page_offset == other.page_offset || end == other_end {
                return Cmp::InsideEdge;
            }
            return Cmp::Inside;
        }
        if self.page_offset < other.page_offset && end > other_end {
            return Cmp::Contains;
        }
        if self.page_offset <= other.page_offset && end <= other_end {
            return Cmp::LessOverlap;
        }
        if self.page_offset >= other.page_offset {
            if self.page_offset == other_end {
                return Cmp::GreaterAdjacent;
            }
            if self.page_offset > other_end {
                return Cmp::Greater;
            }
            if end >= other_end {
                return Cmp::GreaterOverlap;
            }
        }
        // Unreachable: with `self` starting first, Contains or LessOverlap answered; otherwise
        // Inside, InsideEdge or the tests just above answered.
        crate::fatal!(DeathKind::RunCompareUnordered)
    }

    /// Takes `other`'s pages off this run's start or end; raises when `other` lies strictly
    /// inside it.
    pub fn subtract_overlap(&mut self, other: &Run) -> RaiseResult<()> {
        let start = self.page_offset;
        let other_start = other.page_offset;
        let difference = other_start.wrapping_sub(start);
        if other_start <= start {
            let end = start.wrapping_add(self.page_length);
            let other_end = other_start.wrapping_add(other.page_length);
            let remaining = end.wrapping_sub(other_end);
            if end <= other_end {
                *self = Run::default();
                return Ok(());
            }
            self.page_offset = other_end;
            self.page_length = remaining;
            return Ok(());
        }
        let other_end = other_start.wrapping_add(other.page_length);
        let end = start.wrapping_add(self.page_length);
        if other_end < end {
            crate::raise!(
                RaiseKind::RunSubtractInside,
                "a run subtracted from inside another"
            );
        }
        self.page_length = difference;
        Ok(())
    }

    /// Cuts `other`, which lies inside this run, out of it: this run keeps the part before, and
    /// the part after is returned.
    pub fn subtract_inside(&mut self, other: &Run) -> Run {
        let end = self.end();
        self.page_length = other.page_offset.wrapping_sub(self.page_offset);
        let remainder_offset = other.end();
        Run {
            page_offset: remainder_offset,
            page_length: end.wrapping_sub(remainder_offset),
        }
    }

    /// `(offset, length)` in bytes, of `pages`.
    pub fn get_range(&self, pages: GuestPages) -> (u64, u64) {
        (
            pages.address(self.page_offset),
            pages.address(self.page_length),
        )
    }

    /// The hull, even of disjoint runs; both ends are taken before `pageOffset` moves.
    pub fn merge(&mut self, other: &Run) {
        let end = self.end();
        let other_end = other.end();
        if self.page_length == 0 {
            *self = *other;
            return;
        }
        if self.page_offset > other.page_offset {
            self.page_offset = other.page_offset;
        }
        if end > other_end {
            self.page_length = end.wrapping_sub(self.page_offset);
            return;
        }
        self.page_length = other_end.wrapping_sub(self.page_offset);
    }

    /// The overlap; wraps when disjoint. Only its unit test calls it.
    #[cfg(test)]
    pub fn intersect(&mut self, other: &Run) {
        let mut end = self.end();
        let other_end = other.end();
        if self.page_offset < other.page_offset {
            self.page_offset = other.page_offset;
        }
        if end > other_end {
            end = other_end;
        }
        self.page_length = end.wrapping_sub(self.page_offset);
    }
}

/// `first` merged with `last`.
fn merged_run(mut first: Run, last: &Run) -> Run {
    first.merge(last);
    first
}

/// Up to 256 sorted runs and their approximate hull.
#[derive(Debug, Clone)]
struct RunBlock {
    /// The runs; entries past `run_count` are zeroed, and no decision reads them.
    runs: [Run; 256],
    /// The hull, kept approximately.
    block_run: Run,
    /// The runs in use.
    run_count: u32,
}

impl RunBlock {
    fn new() -> Box<RunBlock> {
        Box::new(RunBlock {
            runs: [Run::default(); 256],
            block_run: Run::default(),
            run_count: 0,
        })
    }

    /// Run `i`. Every index formed is below 256, because the full-block deaths hold the count
    /// below 256 before any store; an index past it panics, which is a death.
    #[expect(clippy::indexing_slicing, reason = "every index is below 256 (above)")]
    fn at(&self, i: u32) -> Run {
        self.runs[i.to_usize()]
    }

    /// Run `i`, for writing (see [`RunBlock::at`]).
    #[expect(clippy::indexing_slicing, reason = "every index is below 256 (at)")]
    fn at_mut(&mut self, i: u32) -> &mut Run {
        &mut self.runs[i.to_usize()]
    }

    /// Where the hull lies against `run`.
    fn compare(&self, run: &Run) -> Cmp {
        self.block_run.compare(run)
    }

    /// The binary search of `add`, `remove` and `test`: the last probed index, not the insertion
    /// point, and 0 for an empty block.
    fn probe(&self, run: &Run) -> i32 {
        let mut upper = self.run_count.cast_signed();
        let mut lower: i32 = 0;
        let mut mid: i32 = 0;
        while lower < upper {
            mid = lower.wrapping_add(upper).wrapping_div(2);
            let probe = self.at(mid.cast_unsigned()).page_offset;
            if run.page_offset < probe {
                upper = mid;
            } else if run.page_offset <= probe {
                break;
            } else {
                lower = mid.wrapping_add(1);
            }
        }
        mid
    }

    fn add(&mut self, run: &Run) -> RaiseResult<()> {
        crate::fatal_unless!(self.run_count < RUNS_PER_BLOCK, DeathKind::BlockAddFull);
        if self.run_count == 0 {
            *self.at_mut(0) = *run;
            self.block_run = *run;
            self.run_count = 1;
            return Ok(());
        }
        let mut index = self.probe(run).cast_unsigned();
        if self.at(index).page_offset < run.page_offset {
            index = index.wrapping_add(1);
        }
        let mut merged = false;
        if index > 0 {
            let before = index.wrapping_sub(1);
            let cmp = run.compare(&self.at(before));
            if cmp == Cmp::GreaterOverlap {
                self.at_mut(before).merge(run);
                index = before;
                merged = true;
            } else if matches!(cmp, Cmp::Equal | Cmp::Inside | Cmp::InsideEdge) {
                // The hull is not merged.
                return Ok(());
            }
        }
        if !merged {
            if index < self.run_count && self.at(index).page_offset < run.page_offset {
                index = index.wrapping_add(1);
            }
            self.insert_at_index(index, run);
        }
        // Merged: clean up the runs after it.
        let mut remove_count: u32 = 0;
        let mut done = false;
        let mut i = index.wrapping_add(1);
        while i < self.run_count && !done {
            let current = self.at(index);
            let next = self.at(i);
            match current.compare(&next) {
                // Adjacent runs are not coalesced.
                Cmp::Less | Cmp::LessAdjacent => done = true,
                Cmp::LessOverlap => {
                    self.at_mut(i).subtract_overlap(&current)?;
                    if self.at(i).is_empty() {
                        remove_count = remove_count.wrapping_add(1);
                    }
                    done = true;
                }
                Cmp::Equal => {
                    remove_count = remove_count.wrapping_add(1);
                    done = true;
                }
                Cmp::InsideEdge => {
                    self.at_mut(index).merge(&next);
                    remove_count = remove_count.wrapping_add(1);
                    done = true;
                }
                Cmp::Contains => remove_count = remove_count.wrapping_add(1),
                Cmp::GreaterOverlap => {
                    self.at_mut(index).merge(&next);
                    remove_count = remove_count.wrapping_add(1);
                }
                // An abort(), not an assert.
                Cmp::Inside | Cmp::GreaterAdjacent | Cmp::Greater => {
                    crate::fatal!(DeathKind::BlockAddUnordered)
                }
            }
            i = i.wrapping_add(1);
        }
        if remove_count != 0 {
            self.remove_at_index(index.wrapping_add(1), remove_count);
        }
        self.block_run.merge(run);
        Ok(())
    }

    fn remove(&mut self, run: &Run) -> RaiseResult<()> {
        // A block `remove` grew to 256 runs dies here.
        crate::fatal_unless!(self.run_count < RUNS_PER_BLOCK, DeathKind::BlockRemoveFull);
        let mid = self.probe(run);
        let mut remove_start = u32::MAX;
        let mut remove_count: u32 = 0;
        let mut done = false;
        // The scan starts at the last probe and only goes forward, so it can miss the run that
        // holds the range.
        let mut i = mid.cast_unsigned();
        while i < self.run_count && !done {
            match self.at(i).compare(run) {
                Cmp::LessOverlap => {
                    self.at_mut(i).subtract_overlap(run)?;
                    if self.at(i).is_empty() {
                        if remove_start == u32::MAX {
                            remove_start = i;
                        }
                        remove_count = remove_count.wrapping_add(1);
                    }
                }
                Cmp::Equal => {
                    crate::fatal_unless!(
                        remove_start == u32::MAX,
                        DeathKind::BlockRemoveEqualAfterMark
                    );
                    remove_start = i;
                    remove_count = 1;
                    done = true;
                }
                Cmp::Inside | Cmp::InsideEdge => {
                    if remove_start == u32::MAX {
                        remove_start = i;
                    }
                    remove_count = remove_count.wrapping_add(1);
                }
                Cmp::Contains => {
                    // The split grows the block past 240, up to 256; no removal is applied and the
                    // hull is not recomputed.
                    let remainder = self.at_mut(i).subtract_inside(run);
                    self.insert_at_index(i.wrapping_add(1), &remainder);
                    return Ok(());
                }
                Cmp::GreaterOverlap => {
                    self.at_mut(i).subtract_overlap(run)?;
                    if self.at(i).is_empty() {
                        if remove_start == u32::MAX {
                            remove_start = i;
                        }
                        remove_count = remove_count.wrapping_add(1);
                    }
                    done = true;
                }
                Cmp::Less | Cmp::LessAdjacent | Cmp::GreaterAdjacent | Cmp::Greater => {}
            }
            i = i.wrapping_add(1);
        }
        if remove_count != 0 {
            self.remove_at_index(remove_start, remove_count);
        }
        if self.run_count == 0 {
            // The emptied block keeps its slot with a {0,0} hull, and compares as after
            // everything.
            self.block_run = Run::default();
        } else {
            self.block_run = merged_run(self.at(0), &self.at(self.run_count.wrapping_sub(1)));
        }
        Ok(())
    }

    fn remove_at_index(&mut self, i: u32, count: u32) {
        crate::fatal_unless!(
            i.wrapping_add(count) <= self.run_count,
            DeathKind::BlockRemovePastCount
        );
        self.run_count = self.run_count.wrapping_sub(count);
        let mut j = i;
        while j < self.run_count {
            *self.at_mut(j) = self.at(j.wrapping_add(count));
            j = j.wrapping_add(1);
        }
    }

    fn insert_at_index(&mut self, i: u32, run: &Run) {
        crate::fatal_unless!(self.run_count < RUNS_PER_BLOCK, DeathKind::BlockInsertFull);
        let mut j = self.run_count;
        while j > i {
            *self.at_mut(j) = self.at(j.wrapping_sub(1));
            j = j.wrapping_sub(1);
        }
        *self.at_mut(i) = *run;
        self.run_count = self.run_count.wrapping_add(1);
    }

    /// Whether this block's runs cover `run`, trimming the covered pages off it.
    fn test(&self, run: &mut Run) -> RaiseResult<bool> {
        if self.run_count == 0 {
            return Ok(false);
        }
        if run.end() < self.block_run.page_offset || self.block_run.end() < run.page_offset {
            return Ok(false);
        }
        let mut i = self.probe(run).cast_unsigned();
        while i < self.run_count {
            let current = self.at(i);
            match current.compare(run) {
                Cmp::Less | Cmp::LessAdjacent => {}
                Cmp::LessOverlap | Cmp::InsideEdge => run.subtract_overlap(&current)?,
                Cmp::Inside => return Ok(false),
                Cmp::Equal | Cmp::Contains | Cmp::GreaterOverlap => {
                    run.subtract_overlap(&current)?;
                    return Ok(run.is_empty());
                }
                Cmp::GreaterAdjacent | Cmp::Greater => return Ok(run.is_empty()),
            }
            i = i.wrapping_add(1);
        }
        Ok(run.is_empty())
    }

    fn split(&mut self) -> Box<RunBlock> {
        let mut new_block = RunBlock::new();
        let old_count = self.run_count;
        let half = old_count.wrapping_div(2);
        let mut i = half;
        while i < old_count {
            let run = self.at(i);
            *new_block.at_mut(i.wrapping_sub(half)) = run;
            new_block.block_run.merge(&run);
            new_block.run_count = new_block.run_count.wrapping_add(1);
            i = i.wrapping_add(1);
        }
        // Keeps `old - half`, not `half`; an odd count dies below.
        self.run_count = old_count.wrapping_sub(half);
        crate::fatal_unless!(
            self.run_count.wrapping_add(new_block.run_count) == old_count,
            DeathKind::BlockSplitCount
        );
        let last = self.at(self.run_count.wrapping_sub(1));
        self.block_run.page_length = last.end().wrapping_sub(self.block_run.page_offset);
        crate::fatal_unless!(
            self.block_run.compare(&new_block.block_run).value() <= Cmp::LessAdjacent.value(),
            DeathKind::BlockSplitOverlap
        );
        new_block
    }
}

/// The tracker's blocks.
#[derive(Debug)]
struct Blocks {
    /// The slots in use.
    count: u32,
    /// `Some` exactly for the slots below `count`.
    blocks: [Option<Box<RunBlock>>; 128],
}

impl Default for Blocks {
    fn default() -> Blocks {
        Blocks {
            count: 0,
            blocks: std::array::from_fn(|_| None),
        }
    }
}

impl Blocks {
    /// Slot `i`. Every index formed is below 128 (the count check and the compaction's death
    /// guard the stores).
    #[expect(clippy::indexing_slicing, reason = "every index is below 128 (above)")]
    fn slot(&mut self, i: u32) -> &mut Option<Box<RunBlock>> {
        &mut self.blocks[i.to_usize()]
    }

    /// Block `i`, read without a check: an empty slot there is a NULL read (a death).
    #[expect(clippy::indexing_slicing, reason = "every index is below 128 (slot)")]
    fn block(&self, i: u32) -> &RunBlock {
        match &self.blocks[i.to_usize()] {
            Some(b) => b,
            None => crate::fatal!(DeathKind::TrackerSlotEmpty),
        }
    }

    /// Block `i`, for writing (see [`Blocks::block`]).
    #[expect(clippy::indexing_slicing, reason = "every index is below 128 (slot)")]
    fn block_mut(&mut self, i: u32) -> &mut RunBlock {
        match &mut self.blocks[i.to_usize()] {
            Some(b) => b,
            None => crate::fatal!(DeathKind::TrackerSlotEmpty),
        }
    }

    /// An add, under the tracker's write lock.
    fn add_to_range(&mut self, run: &Run) -> RaiseResult<()> {
        if self.count == 0 {
            *self.slot(0) = Some(RunBlock::new());
            self.count = 1;
            return self.block_mut(0).add(run);
        }
        let mut compacted = false;
        let mut target_index: u32;
        // `cmp` keeps the value of the last block examined, which need not be the target's.
        let mut cmp = Cmp::Equal;
        loop {
            target_index = 0;
            let mut target: Option<*const RunBlock> = None;
            let mut found = false;
            let mut i: u32 = 0;
            while i < self.count && !found {
                let block = self.block(i);
                cmp = block.compare(run);
                match cmp {
                    Cmp::Less | Cmp::GreaterOverlap => {
                        target = Some(std::ptr::from_ref(block));
                        target_index = i;
                    }
                    Cmp::GreaterAdjacent | Cmp::Greater if target.is_some() => {}
                    Cmp::GreaterAdjacent
                    | Cmp::Greater
                    | Cmp::LessAdjacent
                    | Cmp::LessOverlap
                    | Cmp::Equal
                    | Cmp::Inside
                    | Cmp::InsideEdge
                    | Cmp::Contains => {
                        target = Some(std::ptr::from_ref(block));
                        target_index = i;
                        found = true;
                    }
                }
                i = i.wrapping_add(1);
            }
            // The first block always sets the target, so this never reads an empty slot.
            if self.block(target_index).run_count < FULL_BLOCK {
                break;
            }
            crate::fatal_unless!(
                target.is_some_and(|t| std::ptr::eq(t, self.block(target_index))),
                DeathKind::TrackerTargetMoved
            );
            if self.count == MAX_BLOCKS {
                if compacted {
                    // Raised after the compaction or split already made, and after the caller's
                    // host mapping (the task's order).
                    crate::raise!(
                        RaiseKind::TrackerTooComplex,
                        "the range needs more blocks than the tracker holds"
                    );
                }
                self.compact();
            } else if matches!(cmp, Cmp::Less | Cmp::LessAdjacent)
                && target_index == self.count.wrapping_sub(1)
            {
                target_index = target_index.wrapping_add(1);
                *self.slot(target_index) = Some(RunBlock::new());
                self.count = self.count.wrapping_add(1);
                break;
            } else {
                self.split_at_index(target_index);
            }
            compacted = true;
            if self.count == 0 {
                return Ok(());
            }
        }
        self.block_mut(target_index).add(run)?;
        let mut done = false;
        let mut i = target_index.wrapping_add(1);
        while i < self.count && !done {
            match self.block(i).compare(run) {
                Cmp::InsideEdge | Cmp::GreaterOverlap => {
                    // An InsideEdge block can be emptied here and stays: a new run ending where a
                    // later block ends empties it.
                    self.block_mut(i).remove(run)?;
                    done = true;
                }
                // The block that moves into slot i is skipped by `i++`.
                Cmp::Equal | Cmp::Inside => self.remove_at_index(i),
                Cmp::Contains | Cmp::GreaterAdjacent | Cmp::Greater => done = true,
                // An abort(), not an assert.
                Cmp::Less | Cmp::LessAdjacent | Cmp::LessOverlap => {
                    crate::fatal!(DeathKind::TrackerAddUnordered)
                }
            }
            i = i.wrapping_add(1);
        }
        Ok(())
    }

    fn split_at_index(&mut self, index: u32) {
        // Slot i - 1 moves up for i = count..index+1; the last copy (into index + 1) is overwritten
        // by the split below, so only the slots above `index` move.
        let mut i = self.count;
        while i > index.wrapping_add(1) {
            let moved = self.slot(i.wrapping_sub(1)).take();
            *self.slot(i) = moved;
            i = i.wrapping_sub(1);
        }
        let new_block = self.block_mut(index).split();
        *self.slot(index.wrapping_add(1)) = Some(new_block);
        self.count = self.count.wrapping_add(1);
    }

    fn remove_at_index(&mut self, index: u32) {
        crate::fatal_unless!(
            self.slot(index).is_some(),
            DeathKind::TrackerRemoveEmptySlot
        );
        crate::fatal_unless!(self.count != 0, DeathKind::TrackerRemoveNoBlocks);
        // The block's drop.
        *self.slot(index) = None;
        let mut i = index.wrapping_add(1);
        while i < self.count {
            let moved = self.slot(i).take();
            *self.slot(i.wrapping_sub(1)) = moved;
            i = i.wrapping_add(1);
        }
        self.count = self.count.wrapping_sub(1);
        *self.slot(self.count) = None;
    }

    fn compact(&mut self) {
        let mut i: u32 = 0;
        while i < self.count.wrapping_sub(1) {
            while self.block(i).run_count < FULL_BLOCK && i < self.count.wrapping_sub(1) {
                crate::fatal_unless!(
                    i.wrapping_add(1) < MAX_BLOCKS,
                    DeathKind::TrackerCompactPastBlocks
                );
                let next_index = i.wrapping_add(1);
                let run_count = self.block(i).run_count;
                let next_count = self.block(next_index).run_count;
                let to_steal = next_count.min(FULL_BLOCK.wrapping_sub(run_count));
                // An empty next block dies here.
                crate::fatal_unless!(to_steal != 0, DeathKind::TrackerCompactEmptyNext);
                let stolen: Vec<Run> = (0..to_steal)
                    .map(|j| self.block(next_index).at(j))
                    .collect();
                let block = self.block_mut(i);
                for (j, run) in (0u32..).zip(stolen) {
                    *block.at_mut(run_count.wrapping_add(j)) = run;
                }
                block.run_count = block.run_count.wrapping_add(to_steal);
                let last = block.at(block.run_count.wrapping_sub(1));
                // The hull's first page is kept (0 for an emptied block).
                block.block_run.page_length = last.end().wrapping_sub(block.block_run.page_offset);
                if to_steal < next_count {
                    let next = self.block_mut(next_index);
                    let mut j: u32 = 0;
                    while j < next.run_count.wrapping_sub(to_steal) {
                        *next.at_mut(j) = next.at(to_steal.wrapping_add(j));
                        j = j.wrapping_add(1);
                    }
                    next.run_count = next.run_count.wrapping_sub(to_steal);
                    next.block_run.page_offset = next.at(0).page_offset;
                    let next_last = next.at(next.run_count.wrapping_sub(1));
                    next.block_run.page_length = next_last
                        .page_offset
                        .wrapping_sub(next.block_run.page_offset)
                        .wrapping_add(next_last.page_length);
                } else {
                    self.remove_at_index(next_index);
                }
            }
            i = i.wrapping_add(1);
        }
    }

    /// A remove, under the tracker's write lock.
    fn remove_from_range(&mut self, run: &Run) -> RaiseResult<()> {
        let mut done = false;
        let mut i: u32 = 0;
        while i < self.count && !done {
            match self.block(i).compare(run) {
                Cmp::Less | Cmp::LessAdjacent => {}
                Cmp::LessOverlap | Cmp::GreaterOverlap => self.block_mut(i).remove(run)?,
                Cmp::Equal | Cmp::Inside | Cmp::InsideEdge => {
                    // Stops after deleting the first whole block: later blocks keep their runs,
                    // so a block opened past a full last block survives an unmap of everything.
                    self.remove_at_index(i);
                    done = true;
                }
                Cmp::Contains => {
                    self.block_mut(i).remove(run)?;
                    done = true;
                }
                Cmp::GreaterAdjacent | Cmp::Greater => done = true,
            }
            i = i.wrapping_add(1);
        }
        Ok(())
    }
}

/// One run block as the tracker holds it: its hull and its runs. For tests only.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSnapshot {
    /// The hull.
    pub block_run: Run,
    /// The runs in use.
    pub runs: Vec<Run>,
}

/// Which task pages are host-mapped (blocks setup). The blocks sit behind a writer-preferring
/// lock: written by adds and removes, read by tests and the visit's scan; the visitor runs with
/// the lock released, since it re-enters the tracker. Dropping it drops the blocks, with no lock.
#[derive(Debug)]
pub struct RangeTracker {
    blocks: WriterPrefRwLock<Blocks>,
    /// The task's guest pages, the runs' unit.
    pages: GuestPages,
}

impl RangeTracker {
    /// No blocks, over runs of `pages`.
    pub fn new(pages: GuestPages) -> RangeTracker {
        RangeTracker {
            blocks: WriterPrefRwLock::default(),
            pages,
        }
    }

    /// The pages of `[offset, offset + length)`, which must be whole pages: raises for a length
    /// of 0 or not whole pages, then for an offset not page-aligned.
    pub fn calculate_page_run_aligned(&self, offset: u64, length: u64) -> RaiseResult<Run> {
        let (shift, mask) = (self.pages.shift(), self.pages.mask());
        if length == 0 || (length & mask) != 0 {
            crate::raise!(
                RaiseKind::TrackerAlignedLength,
                format_args!("length {length:#x} is 0 or not whole pages")
            );
        }
        if (offset & mask) != 0 {
            crate::raise!(
                RaiseKind::TrackerAlignedOffset,
                format_args!("offset {offset:#x} is not page aligned")
            );
        }
        Ok(Run {
            page_offset: offset.wrapping_shr(shift).low_u32(),
            page_length: length.wrapping_shr(shift).low_u32(),
        })
    }

    /// The pages `[offset, offset + length)` touches, rounded outward; raises for a length of 0.
    pub fn calculate_page_run_unaligned(&self, offset: u64, length: u64) -> RaiseResult<Run> {
        if length == 0 {
            crate::raise!(RaiseKind::TrackerZeroLength, "length 0");
        }
        // Rounds with a page less 2 (0x3ffe of 16 KiB pages), the page start subtracted last:
        // one page short for a range ending one byte into a page.
        let (shift, mask) = (self.pages.shift(), self.pages.mask());
        let sum = offset
            .wrapping_add(length)
            .wrapping_add(self.pages.size().wrapping_sub(2))
            .wrapping_sub(offset & !mask);
        Ok(Run {
            page_offset: offset.wrapping_shr(shift).low_u32(),
            page_length: sum.wrapping_shr(shift).low_u32(),
        })
    }

    /// Marks `[offset, offset + length)` mapped. "the range needs more blocks than the tracker holds" raises with the compaction and
    /// splits made before it kept. The add is a handler frame with no cleanup (an empty `finally`):
    /// a raise inside it goes on from the frame's end. Every caller (the local task's map and
    /// unmap) is under a handler, so the raise is `Err`, and the lock goes as this returns.
    pub fn add_to_range(&self, offset: u64, length: u64) -> RaiseResult<()> {
        // The calculation and its raises precede the lock.
        let run = self.calculate_page_run_aligned(offset, length)?;
        let mut blocks = self.blocks.write();
        blocks.add_to_range(&run)
    }

    /// Marks `[offset, offset + length)` unmapped; a handler frame as [`RangeTracker::add_to_range`]
    /// is.
    pub fn remove_from_range(&self, offset: u64, length: u64) -> RaiseResult<()> {
        let run = self.calculate_page_run_aligned(offset, length)?;
        let mut blocks = self.blocks.write();
        blocks.remove_from_range(&run)
    }

    /// Whether every page `[offset, offset + length)` touches is mapped. Raises for a length of
    /// 0, and a run's subtraction can raise; there is no handler frame here.
    pub fn test_range(&self, offset: u64, length: u64) -> RaiseResult<bool> {
        let mut new_run = self.calculate_page_run_unaligned(offset, length)?;
        let blocks = self.blocks.read();
        let mut done = false;
        let mut i: u32 = 0;
        while i < blocks.count && !done {
            let block = blocks.block(i);
            match block.compare(&new_run) {
                Cmp::Less | Cmp::LessAdjacent => {}
                // A range continuing past this block tests false here.
                Cmp::LessOverlap | Cmp::InsideEdge => {
                    if !block.test(&mut new_run)? {
                        return Ok(false);
                    }
                }
                Cmp::Inside => return Ok(false),
                Cmp::Equal | Cmp::Contains => {
                    if !block.test(&mut new_run)? {
                        return Ok(false);
                    }
                    crate::fatal_unless!(new_run.is_empty(), DeathKind::TrackerTestLeftover);
                    return Ok(true);
                }
                Cmp::GreaterOverlap => {
                    return Ok(block.test(&mut new_run)? && new_run.is_empty());
                }
                Cmp::GreaterAdjacent | Cmp::Greater => {
                    if !new_run.is_empty() {
                        return Ok(false);
                    }
                    done = true;
                }
            }
            i = i.wrapping_add(1);
        }
        Ok(new_run.is_empty())
    }

    /// Visits the unmapped runs of the pages `[offset, offset + length)` touches, in order,
    /// until the visitor answers false. Raises "the range's unmapped parts are more than the tracker holds" (more than 16 runs)
    /// before any visit, and visits with the lock released. The visitor's `Err` is a raise out
    /// of it (the lazy map's visitor maps, which can raise): it leaves the loop at once and
    /// skips the remaining runs.
    pub fn visit_unmapped_ranges(
        &self,
        offset: u64,
        length: u64,
        visitor: &mut dyn FnMut(u64, u64) -> RaiseResult<bool>,
    ) -> RaiseResult<()> {
        let (unmapped, count) = self.unmapped_runs(offset, length)?;
        for run in unmapped.iter().take(count) {
            let (range_offset, range_length) = run.get_range(self.pages);
            // A raise leaves the loop.
            if !visitor(range_offset, range_length)? {
                break;
            }
        }
        Ok(())
    }

    /// The unmapped runs of the visit: the blocks scanned under the read lock, the last run
    /// placed after it is released.
    fn unmapped_runs(
        &self,
        offset: u64,
        length: u64,
    ) -> RaiseResult<([Run; UNMAPPED_RUNS], usize)> {
        let mut unmapped = [Run::default(); UNMAPPED_RUNS];
        let mut unmapped_count: usize = 0;
        let mut current;
        {
            let blocks = self.blocks.read();
            // Inside the lock, unlike add, remove and test.
            current = self.calculate_page_run_unaligned(offset, length)?;
            let mut keep_going = true;
            let mut i: u32 = 0;
            while i < blocks.count && keep_going {
                let block = blocks.block(i);
                let scan = match block.compare(&current) {
                    Cmp::Less | Cmp::LessAdjacent => false,
                    Cmp::GreaterAdjacent | Cmp::Greater => {
                        keep_going = false;
                        false
                    }
                    Cmp::Equal | Cmp::Contains | Cmp::GreaterOverlap => {
                        keep_going = false;
                        true
                    }
                    Cmp::LessOverlap | Cmp::Inside | Cmp::InsideEdge => true,
                };
                let mut more = scan;
                // From run 0, not from a binary search.
                let mut j: u32 = 0;
                while j < block.run_count && more {
                    let run = block.at(j);
                    match run.compare(&current) {
                        Cmp::Less | Cmp::LessAdjacent => {}
                        Cmp::LessOverlap | Cmp::InsideEdge => current.subtract_overlap(&run)?,
                        Cmp::Equal | Cmp::Contains => {
                            current = Run::default();
                            more = false;
                        }
                        Cmp::Inside => {
                            let remainder = current.subtract_inside(&run);
                            crate::fatal_unless!(
                                !current.is_empty(),
                                DeathKind::TrackerVisitEmptyRun
                            );
                            match unmapped.get_mut(unmapped_count) {
                                Some(slot) => *slot = current,
                                // Raised before any visit.
                                None => crate::raise!(
                                    RaiseKind::TrackerUnmappedTooComplex,
                                    "the range's unmapped parts are more than the tracker holds"
                                ),
                            }
                            unmapped_count = unmapped_count.wrapping_add(1);
                            current = remainder;
                        }
                        Cmp::GreaterOverlap => {
                            current.subtract_overlap(&run)?;
                            more = false;
                        }
                        Cmp::GreaterAdjacent | Cmp::Greater => more = false,
                    }
                    j = j.wrapping_add(1);
                }
                i = i.wrapping_add(1);
            }
        }
        if !current.is_empty() {
            match unmapped.get_mut(unmapped_count) {
                Some(slot) => *slot = current,
                // Raised (now unlocked) before any visit.
                None => crate::raise!(
                    RaiseKind::TrackerUnmappedTooComplexTail,
                    "the range's unmapped parts are more than the tracker holds"
                ),
            }
            unmapped_count = unmapped_count.wrapping_add(1);
        }
        Ok((unmapped, unmapped_count))
    }

    /// The blocks in slot order, for tests.
    #[cfg(test)]
    pub fn snapshot(&self) -> Vec<BlockSnapshot> {
        let blocks = self.blocks.read();
        (0..blocks.count)
            .map(|i| {
                let b = blocks.block(i);
                BlockSnapshot {
                    block_run: b.block_run,
                    runs: (0..b.run_count).map(|j| b.at(j)).collect(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    // The blocks are boxed as the tracker holds them.
    #![allow(clippy::vec_box)]

    use super::*;
    use crate::guest::records::{APV_PAGE_MASK, APV_PAGE_SHIFT, APV_PAGE_SIZE};

    use crate::raise::death::DeathSite;
    use crate::raise::{Raise, RaiseId};
    use crate::testkit::capture::catch_death;

    const PAGE: u64 = APV_PAGE_SIZE;

    fn r(page_offset: u32, page_length: u32) -> Run {
        Run {
            page_offset,
            page_length,
        }
    }

    /// Runs `f` under a handler, as the callers' do.
    fn caught<T>(f: impl FnOnce() -> RaiseResult<T>) -> RaiseResult<T> {
        f()
    }

    /// Runs a visit under a handler, as the local task's is.
    fn caught_visit<T>(f: impl FnOnce() -> RaiseResult<T>) -> RaiseResult<T> {
        f()
    }

    /// `r` is the raise `kind` with `reason`.
    fn raised_at(r: &Raise, kind: RaiseKind, reason: &str) {
        assert_eq!(r.id(), RaiseId::Memory(kind));
        assert_eq!(r.detail(), Some(reason));
    }

    /// `t` is the raise `kind` with `reason`, as a handler gets it.
    fn thrown_at(t: &Raise, kind: RaiseKind, reason: &str) {
        assert!(matches!(t.origin(), crate::raise::Origin::Core(_)));
        assert_eq!(t.detail().unwrap_or_default(), reason);
        assert_eq!((t.site().file, t.site().line), (kind.name(), 0));
    }

    /// Where an uncaught `kind` dies: at the raise.
    fn uncaught(kind: RaiseKind) -> DeathSite {
        DeathSite {
            kind: crate::raise::death::DeathKind::UnhandledRaise,
            file: kind.name(),
            line: 0,
        }
    }

    /// A block holding `runs` with the hull `hull` (the hull is kept approximately).
    fn block(runs: &[Run], hull: Run) -> Box<RunBlock> {
        let mut b = RunBlock::new();
        for (i, run) in runs.iter().enumerate() {
            b.runs[i] = *run;
        }
        b.run_count = runs.len() as u32;
        b.block_run = hull;
        b
    }

    fn runs_of(b: &RunBlock) -> Vec<Run> {
        (0..b.run_count).map(|i| b.at(i)).collect()
    }

    /// `n` single-page runs at pages `start, start + 2, …` (never adjacent, never coalesced).
    fn spaced(start: u32, n: u32) -> Vec<Run> {
        (0..n).map(|i| r(start + 2 * i, 1)).collect()
    }

    fn hull_of(runs: &[Run]) -> Run {
        let first = runs[0];
        let last = runs[runs.len() - 1];
        r(first.page_offset, last.end() - first.page_offset)
    }

    fn tracker_with(blocks: Vec<Box<RunBlock>>) -> RangeTracker {
        let t = RangeTracker::new(GuestPages::HOST);
        {
            let mut b = t.blocks.write();
            b.count = blocks.len() as u32;
            for (i, blk) in blocks.into_iter().enumerate() {
                b.blocks[i] = Some(blk);
            }
        }
        t
    }

    fn add_pages(t: &RangeTracker, page: u64, pages: u64) {
        caught(|| t.add_to_range(page * PAGE, pages * PAGE)).unwrap();
    }

    fn remove_pages(t: &RangeTracker, page: u64, pages: u64) {
        caught(|| t.remove_from_range(page * PAGE, pages * PAGE)).unwrap();
    }

    fn tested(t: &RangeTracker, page: u64, pages: u64) -> bool {
        caught(|| t.test_range(page * PAGE, pages * PAGE)).unwrap()
    }

    fn holes(t: &RangeTracker, page: u64, pages: u64) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        caught_visit(|| {
            t.visit_unmapped_ranges(page * PAGE, pages * PAGE, &mut |o, l| {
                out.push((o / PAGE, l / PAGE));
                Ok(true)
            })
        })
        .unwrap();
        out
    }

    mod run_compare {
        use super::*;

        #[test]
        fn every_result_in_order() {
            let other = r(10, 10); // [10, 20)
            assert_eq!(r(10, 10).compare(&other), Cmp::Equal);
            assert_eq!(r(0, 0).compare(&other), Cmp::Greater); // an emptied block's hull
            assert_eq!(r(0, 0).compare(&r(0, 0)), Cmp::Equal);
            assert_eq!(r(0, 9).compare(&other), Cmp::Less);
            assert_eq!(r(5, 5).compare(&other), Cmp::LessAdjacent);
            assert_eq!(r(12, 3).compare(&other), Cmp::Inside);
            assert_eq!(r(10, 3).compare(&other), Cmp::InsideEdge);
            assert_eq!(r(15, 5).compare(&other), Cmp::InsideEdge);
            assert_eq!(r(9, 12).compare(&other), Cmp::Contains);
            assert_eq!(r(5, 10).compare(&other), Cmp::LessOverlap);
            assert_eq!(r(5, 15).compare(&other), Cmp::LessOverlap);
            assert_eq!(r(20, 3).compare(&other), Cmp::GreaterAdjacent);
            assert_eq!(r(21, 3).compare(&other), Cmp::Greater);
            assert_eq!(r(15, 10).compare(&other), Cmp::GreaterOverlap);
            // Same start, self longer: GreaterOverlap.
            assert_eq!(r(10, 11).compare(&other), Cmp::GreaterOverlap);
        }

        #[test]
        fn empty_run_at_page_zero_is_greater_than_everything_else() {
            assert_eq!(r(0, 0).compare(&r(1_000_000, 1)), Cmp::Greater);
            // An empty run elsewhere is not special.
            assert_eq!(r(5, 0).compare(&r(10, 1)), Cmp::Less);
        }

        #[test]
        fn ends_wrap_in_u32() {
            // end = 0xffff_fff0 + 0x20 wraps to 0x10: Less than a run at 0x20.
            assert_eq!(r(0xffff_fff0, 0x20).compare(&r(0x20, 1)), Cmp::Less);
        }

        #[test]
        fn values_are_rs() {
            let all = [
                Cmp::Less,
                Cmp::LessAdjacent,
                Cmp::LessOverlap,
                Cmp::Equal,
                Cmp::Inside,
                Cmp::InsideEdge,
                Cmp::Contains,
                Cmp::GreaterOverlap,
                Cmp::GreaterAdjacent,
                Cmp::Greater,
            ];
            let v: Vec<i32> = all.iter().map(|c| c.value()).collect();
            assert_eq!(v, (-3..=6).collect::<Vec<_>>());
        }
    }

    mod run_subtract_overlap {
        use super::*;

        #[test]
        fn other_starts_at_or_before_self() {
            let mut a = r(10, 10);
            caught(|| a.subtract_overlap(&r(5, 20))).unwrap();
            assert_eq!(a, r(0, 0));
            let mut a = r(10, 10);
            caught(|| a.subtract_overlap(&r(8, 5))).unwrap();
            assert_eq!(a, r(13, 7));
        }

        #[test]
        fn other_starts_after_self() {
            let mut a = r(10, 10);
            caught(|| a.subtract_overlap(&r(15, 10))).unwrap();
            assert_eq!(a, r(10, 5));
            let mut a = r(10, 10);
            caught(|| a.subtract_overlap(&r(15, 5))).unwrap();
            assert_eq!(a, r(10, 5));
        }

        #[test]
        fn inside_throws() {
            let mut a = r(10, 10);
            let e = caught(|| a.subtract_overlap(&r(12, 2))).unwrap_err();
            raised_at(
                &e,
                RaiseKind::RunSubtractInside,
                "a run subtracted from inside another",
            );
            assert_eq!(a, r(10, 10));
        }

        #[test]
        fn inside_names_the_throw_site() {
            let mut a = r(10, 10);
            let e = caught(|| a.subtract_overlap(&r(12, 2))).unwrap_err();
            assert_eq!(e.site(), uncaught(RaiseKind::RunSubtractInside));
        }
    }

    #[test]
    fn run_subtract_inside() {
        let mut a = r(10, 10);
        assert_eq!(a.subtract_inside(&r(12, 3)), r(15, 5));
        assert_eq!(a, r(10, 2));
    }

    #[test]
    fn run_get_range() {
        assert_eq!(r(3, 2).get_range(GuestPages::HOST), (3 * PAGE, 2 * PAGE));
        assert_eq!(
            r(u32::MAX, u32::MAX).get_range(GuestPages::HOST),
            (
                u64::from(u32::MAX) << APV_PAGE_SHIFT,
                u64::from(u32::MAX) << APV_PAGE_SHIFT
            )
        );
    }

    #[test]
    fn run_merge() {
        let mut a = r(7, 0);
        a.merge(&r(3, 2));
        assert_eq!(a, r(3, 2), "empty self takes other");
        let mut a = r(10, 10);
        a.merge(&r(5, 2));
        assert_eq!(a, r(5, 15), "hull of disjoint runs, end is self's");
        let mut a = r(10, 2);
        a.merge(&r(20, 5));
        assert_eq!(a, r(10, 15), "end is other's");
        let mut a = r(10, 10);
        a.merge(&r(12, 8));
        assert_eq!(a, r(10, 10), "equal ends take other's");
    }

    #[test]
    fn run_intersect() {
        let mut a = r(10, 10);
        a.intersect(&r(15, 10));
        assert_eq!(a, r(15, 5));
        let mut a = r(10, 10);
        a.intersect(&r(5, 10));
        assert_eq!(a, r(10, 5));
        let mut a = r(10, 2);
        a.intersect(&r(20, 5));
        assert_eq!(a, r(20, 12u32.wrapping_sub(20)), "disjoint wraps");
    }

    #[test]
    fn run_is_empty() {
        assert!(r(5, 0).is_empty());
        assert!(!r(0, 1).is_empty());
    }

    #[test]
    fn merged_run_is_the_hull() {
        assert_eq!(merged_run(r(2, 1), &r(9, 3)), r(2, 10));
    }

    #[test]
    fn block_new_and_compare() {
        let b = RunBlock::new();
        assert_eq!(b.run_count, 0);
        assert_eq!(b.block_run, r(0, 0));
        assert_eq!(b.compare(&r(5, 1)), Cmp::Greater);
        let b = block(&[r(10, 2)], r(10, 2));
        assert_eq!(b.compare(&r(12, 1)), Cmp::LessAdjacent);
    }

    #[test]
    fn block_search_returns_the_last_probe() {
        // Runs [0,1), [4,5), [8,12), [16,17); page 10 probes 2 then 3.
        let b = block(&[r(0, 1), r(4, 1), r(8, 4), r(16, 1)], r(0, 17));
        assert_eq!(b.probe(&r(10, 1)), 3);
        assert_eq!(b.probe(&r(4, 1)), 1);
        assert_eq!(RunBlock::new().probe(&r(4, 1)), 0);
    }

    mod block_add {
        use super::*;

        fn add(b: &mut RunBlock, run: Run) {
            caught(|| b.add(&run)).unwrap();
        }

        #[test]
        fn first_run() {
            let mut b = RunBlock::new();
            add(&mut b, r(5, 2));
            assert_eq!(runs_of(&b), vec![r(5, 2)]);
            assert_eq!(b.block_run, r(5, 2));
        }

        #[test]
        fn inserts_in_order_and_merges_the_hull() {
            let mut b = block(&[r(2, 1), r(10, 1)], r(2, 9));
            add(&mut b, r(6, 1));
            assert_eq!(runs_of(&b), vec![r(2, 1), r(6, 1), r(10, 1)]);
            add(&mut b, r(0, 1));
            add(&mut b, r(20, 1));
            assert_eq!(
                runs_of(&b),
                vec![r(0, 1), r(2, 1), r(6, 1), r(10, 1), r(20, 1)]
            );
            assert_eq!(b.block_run, r(0, 21));
        }

        #[test]
        fn adjacent_runs_are_not_coalesced() {
            // Adjacent runs are never coalesced
            let mut b = block(&[r(2, 1)], r(2, 1));
            add(&mut b, r(3, 1));
            add(&mut b, r(1, 1));
            assert_eq!(runs_of(&b), vec![r(1, 1), r(2, 1), r(3, 1)]);
        }

        #[test]
        fn greater_overlap_merges_into_the_previous_run() {
            let mut b = block(&[r(2, 4), r(10, 1)], r(2, 9));
            add(&mut b, r(4, 4));
            assert_eq!(runs_of(&b), vec![r(2, 6), r(10, 1)]);
        }

        #[test]
        fn inside_the_previous_run_changes_nothing() {
            let mut b = block(&[r(2, 6)], r(2, 6));
            b.block_run = r(2, 1); // the hull is not merged on this return
            // Inside and InsideEdge (shared end) of the run before the insertion point.
            for run in [r(3, 2), r(4, 4)] {
                add(&mut b, run);
                assert_eq!(runs_of(&b), vec![r(2, 6)]);
                assert_eq!(b.block_run, r(2, 1));
            }
        }

        #[test]
        fn cleanup_less_overlap_trims_the_next_run() {
            let mut b = block(&[r(2, 1), r(10, 5)], r(2, 13));
            add(&mut b, r(8, 4));
            assert_eq!(runs_of(&b), vec![r(2, 1), r(8, 4), r(12, 3)]);
        }

        #[test]
        fn cleanup_contains_and_equal_remove() {
            // The new run [4,12) contains [5,6) and [7,8), then equals nothing; [12,13) is Less.
            let mut b = block(&[r(5, 1), r(7, 1), r(12, 1)], r(5, 8));
            add(&mut b, r(4, 8));
            assert_eq!(runs_of(&b), vec![r(4, 8), r(12, 1)]);
        }

        #[test]
        fn cleanup_equal_removes_the_duplicate() {
            // The probe breaks on the equal start, so the new run goes in first.
            let mut b = block(&[r(5, 3)], r(5, 3));
            add(&mut b, r(5, 3));
            assert_eq!(runs_of(&b), vec![r(5, 3)]);
        }

        #[test]
        fn cleanup_inside_edge_merges_the_next_run() {
            let mut b = block(&[r(5, 4)], r(5, 4));
            add(&mut b, r(5, 2)); // [5,7) before [5,9): InsideEdge -> merged, next removed
            assert_eq!(runs_of(&b), vec![r(5, 4)]);
        }

        #[test]
        fn cleanup_greater_overlap_merges_and_continues() {
            // [5,10) before [5,7): GreaterOverlap (merge), then [8,9): Contains, then Less.
            let mut b = block(&[r(5, 2), r(8, 1), r(30, 1)], r(5, 26));
            add(&mut b, r(5, 5));
            assert_eq!(runs_of(&b), vec![r(5, 5), r(30, 1)]);
            assert_eq!(b.block_run, r(5, 26));
        }

        #[test]
        fn merged_previous_run_trims_the_next() {
            // An unsorted block: [6,9) merges into [5,7) -> [5,9), which empties [8,9).
            let mut b = block(&[r(0, 1), r(5, 2), r(8, 1), r(1, 1)], r(0, 9));
            add(&mut b, r(6, 3));
            assert_eq!(runs_of(&b), vec![r(0, 1), r(5, 4), r(1, 1)]);
        }

        #[test]
        fn cleanup_abort_on_an_impossible_order() {
            // Only an unsorted block reaches it. [6,11) merges into [5,7) -> [5,11); [8,9) is
            // Contained; [1,2) is Greater from [5,11)'s side.
            let mut b = block(&[r(0, 1), r(5, 2), r(8, 1), r(1, 1)], r(0, 9));
            assert_eq!(
                catch_death(|| b.add(&r(6, 5))).unwrap_err(),
                DeathKind::BlockAddUnordered.site()
            );
        }

        #[test]
        fn full_block_dies() {
            let mut b = block(&spaced(0, 256), r(0, 511));
            assert_eq!(
                catch_death(|| b.add(&r(1000, 1))).unwrap_err(),
                DeathKind::BlockAddFull.site()
            );
        }
    }

    mod block_remove {
        use super::*;

        fn remove(b: &mut RunBlock, run: Run) {
            caught(|| b.remove(&run)).unwrap();
        }

        #[test]
        fn equal_removes_one_and_recomputes_the_hull() {
            let mut b = block(&[r(0, 1), r(4, 1), r(8, 1)], r(0, 9));
            remove(&mut b, r(8, 1));
            assert_eq!(runs_of(&b), vec![r(0, 1), r(4, 1)]);
            assert_eq!(b.block_run, r(0, 5));
        }

        #[test]
        fn inside_and_overlaps() {
            // [3,11) removes [4,5) and [8,9) (InsideEdge / Inside) and trims [10,12).
            let mut b = block(&[r(0, 4), r(4, 1), r(8, 1), r(10, 2)], r(0, 12));
            // The probe of 3 stops at index 1 ([4]) -> upper 1, mid 0 ([0]) -> lower 1: mid
            // 0, so [0,4) (LessOverlap of [3,11)) is trimmed to [0,3).
            remove(&mut b, r(3, 8));
            assert_eq!(runs_of(&b), vec![r(0, 3), r(11, 1)]);
            assert_eq!(b.block_run, r(0, 12));
        }

        #[test]
        fn less_overlap_emptied_is_removed() {
            let mut b = block(&[r(4, 2), r(9, 1)], r(4, 6));
            remove(&mut b, r(4, 4));
            assert_eq!(runs_of(&b), vec![r(9, 1)]);
            assert_eq!(b.block_run, r(9, 1));
        }

        #[test]
        fn greater_overlap_emptied_is_removed_and_ends_the_scan() {
            let mut b = block(&[r(2, 2), r(6, 2)], r(2, 6));
            // [6,8) vs [5,8): off 6 >= 5, end 8 <= 8 -> InsideEdge (removed). Use [5,7):
            // [6,8) vs [5,7): GreaterOverlap -> trimmed to [7,8).
            remove(&mut b, r(5, 2));
            assert_eq!(runs_of(&b), vec![r(2, 2), r(7, 1)]);
            let mut b = block(&[r(6, 2)], r(6, 2));
            remove(&mut b, r(6, 1)); // [6,8) vs [6,7): GreaterOverlap -> [7,8)
            assert_eq!(runs_of(&b), vec![r(7, 1)]);
        }

        #[test]
        fn contains_splits_in_place_without_recomputing_the_hull() {
            // Adds stop at 240 runs per block but `remove` can grow a block to 256
            let mut b = block(&[r(0, 10)], r(0, 99));
            remove(&mut b, r(4, 2));
            assert_eq!(runs_of(&b), vec![r(0, 4), r(6, 4)]);
            assert_eq!(b.block_run, r(0, 99));
        }

        #[test]
        fn the_last_probe_misses_the_containing_run() {
            // testing.md 7.1 `mutgap_mem_tracker_remove_search`.
            let mut b = block(&[r(0, 1), r(4, 1), r(8, 4), r(16, 1)], r(0, 17));
            remove(&mut b, r(10, 1));
            assert_eq!(runs_of(&b), vec![r(0, 1), r(4, 1), r(8, 4), r(16, 1)]);
        }

        #[test]
        fn emptied_block_gets_a_zero_hull() {
            // An emptied block stays, with `fBlockRun = {0,0}`, which compares Greater than any
            // non-`{0,0}` run and so stops lookups and attracts adds (testing.md 7.1
            // `tasks_gap_tracker_empty_block`, `mutgap_mem_tracker_block_end_edge`)
            let mut b = block(&[r(5, 1)], r(5, 1));
            remove(&mut b, r(5, 1));
            assert_eq!(b.run_count, 0);
            assert_eq!(b.block_run, r(0, 0));
        }

        #[test]
        fn a_block_of_256_runs_dies() {
            // testing.md 7.4 `mutgap_mem_tracker_block_256_runs`.
            let mut runs = spaced(0, 255);
            runs[0] = r(0, 1);
            let mut b = block(&runs, hull_of(&runs));
            remove(&mut b, r(1000, 1)); // 255 runs: allowed, nothing found
            let mut runs = spaced(0, 254);
            runs.push(r(600, 3));
            let mut b = block(&runs, hull_of(&runs));
            remove(&mut b, r(601, 1)); // Contains: 256 runs
            assert_eq!(b.run_count, 256);
            assert_eq!(
                catch_death(|| b.remove(&r(0, 1))).unwrap_err(),
                DeathKind::BlockRemoveFull.site()
            );
        }

        #[test]
        fn equal_after_a_marked_run_dies() {
            // [2,3) is InsideEdge of [2,4) (marks removeStart), then a duplicate [2,4) is Equal.
            // An unsorted block: the probe of 2 stops at [1,2); [3,4) is Inside of [2,6) (marks
            // removeStart), then [2,6) is Equal.
            let mut b = block(&[r(1, 1), r(3, 1), r(2, 4)], r(1, 5));
            assert_eq!(
                catch_death(|| b.remove(&r(2, 4))).unwrap_err(),
                DeathKind::BlockRemoveEqualAfterMark.site()
            );
        }
    }

    #[test]
    fn block_remove_and_insert_at_index() {
        let mut b = block(&[r(0, 1), r(2, 1), r(4, 1), r(6, 1)], r(0, 7));
        b.remove_at_index(1, 2);
        assert_eq!(runs_of(&b), vec![r(0, 1), r(6, 1)]);
        b.insert_at_index(1, &r(3, 1));
        assert_eq!(runs_of(&b), vec![r(0, 1), r(3, 1), r(6, 1)]);
        b.insert_at_index(3, &r(9, 1));
        assert_eq!(runs_of(&b), vec![r(0, 1), r(3, 1), r(6, 1), r(9, 1)]);
        assert_eq!(
            catch_death(|| b.remove_at_index(3, 2)).unwrap_err(),
            DeathKind::BlockRemovePastCount.site()
        );
        let mut full = block(&spaced(0, 256), r(0, 511));
        assert_eq!(
            catch_death(|| full.insert_at_index(0, &r(0, 1))).unwrap_err(),
            DeathKind::BlockInsertFull.site()
        );
    }

    mod block_test {
        use super::*;

        fn test(b: &RunBlock, run: Run) -> (bool, Run) {
            let mut run = run;
            let ok = caught(|| b.test(&mut run)).unwrap();
            (ok, run)
        }

        #[test]
        fn empty_block_and_outside_the_hull() {
            assert_eq!(test(&RunBlock::new(), r(1, 1)), (false, r(1, 1)));
            let b = block(&[r(10, 2)], r(10, 2));
            assert!(!test(&b, r(2, 3)).0);
            assert!(!test(&b, r(13, 1)).0);
        }

        #[test]
        fn covered_runs() {
            let b = block(&[r(10, 4), r(14, 2)], r(10, 6));
            assert_eq!(test(&b, r(10, 4)), (true, r(0, 0))); // Equal
            assert_eq!(test(&b, r(11, 2)), (true, r(0, 0))); // Contains
            assert_eq!(test(&b, r(10, 6)), (true, r(0, 0))); // InsideEdge then GreaterOverlap
            assert_eq!(test(&b, r(12, 4)), (true, r(0, 0))); // LessOverlap then Equal
        }

        #[test]
        fn uncovered_runs() {
            let b = block(&[r(10, 2), r(14, 2)], r(10, 6));
            // [10,16): [10,12) InsideEdge -> [12,16); [14,16) InsideEdge -> [12,14); end.
            assert_eq!(test(&b, r(10, 6)), (false, r(12, 2)));
            // Inside -> false at once.
            assert!(!test(&b, r(9, 8)).0);
            // [12,14): the probe stops on [14,16), which is GreaterAdjacent: returns isEmpty().
            assert_eq!(test(&b, r(12, 2)), (false, r(12, 2)));
        }
    }

    mod block_split {
        use super::*;

        #[test]
        fn even_split() {
            let runs = spaced(0, 240);
            let mut b = block(&runs, hull_of(&runs));
            let nb = b.split();
            assert_eq!(runs_of(&b), runs[..120].to_vec());
            assert_eq!(runs_of(&nb), runs[120..].to_vec());
            assert_eq!(b.block_run, r(0, 239));
            assert_eq!(nb.block_run, r(240, 239));
        }

        #[test]
        fn odd_count_dies() {
            // testing.md 7.4 `mutgap2_res_tracker_split_odd_death`.
            let runs = spaced(0, 241);
            let mut b = block(&runs, hull_of(&runs));
            assert_eq!(
                catch_death(|| b.split()).unwrap_err(),
                DeathKind::BlockSplitCount.site()
            );
        }

        #[test]
        fn overlapping_halves_die() {
            // The first half's last run reaches past the second half's first run.
            let mut runs = spaced(0, 4);
            runs[1] = r(2, 10);
            let mut b = block(&runs, r(0, 12));
            assert_eq!(
                catch_death(|| b.split()).unwrap_err(),
                DeathKind::BlockSplitOverlap.site()
            );
        }
    }

    #[test]
    fn calculate_page_run_aligned() {
        let t = RangeTracker::new(GuestPages::HOST);
        assert_eq!(
            caught(|| t.calculate_page_run_aligned(3 * PAGE, 2 * PAGE)).unwrap(),
            r(3, 2)
        );
        for len in [0, 1, PAGE + 1] {
            let e = caught(|| t.calculate_page_run_aligned(0, len)).unwrap_err();
            raised_at(
                &e,
                RaiseKind::TrackerAlignedLength,
                &format!("length {len:#x} is 0 or not whole pages"),
            );
        }
        let e = caught(|| t.calculate_page_run_aligned(1, PAGE)).unwrap_err();
        raised_at(
            &e,
            RaiseKind::TrackerAlignedOffset,
            "offset 0x1 is not page aligned",
        );
        // Truncated to 32 bits.
        let wrap = 1u64 << (32 + APV_PAGE_SHIFT);
        assert_eq!(
            caught(|| t.calculate_page_run_aligned(wrap + PAGE, wrap)).unwrap(),
            r(1, 0)
        );
    }

    #[test]
    fn calculate_page_run_unaligned() {
        let t = RangeTracker::new(GuestPages::HOST);
        let e = caught(|| t.calculate_page_run_unaligned(5, 0)).unwrap_err();
        raised_at(&e, RaiseKind::TrackerZeroLength, "length 0");
        let run = |o, l| caught(|| t.calculate_page_run_unaligned(o, l)).unwrap();
        assert_eq!(run(0, PAGE), r(0, 1));
        assert_eq!(run(PAGE + 10, 20), r(1, 1));
        // A page less 2 rounding: one page short when the range ends one byte into a page, and 0 pages
        // for one byte at a page start.
        assert_eq!(run(0, PAGE + 1), r(0, 1));
        assert_eq!(run(0, 1), r(0, 0));
        assert_eq!(run(0, 2), r(0, 1));
        // testing.md 7.1 `mutgap3_sys_heap_reply_one_past_page`: 16 bytes at 2 pages - 15.
        assert_eq!(run(2 * PAGE - 15, 16), r(1, 1));
        // Wrapping u64 sum.
        assert_eq!(run(!APV_PAGE_MASK, PAGE), r(u32::MAX, 1));
    }

    #[test]
    fn tracker_new_and_drop() {
        let t = RangeTracker::new(GuestPages::HOST);
        assert!(t.snapshot().is_empty());
        add_pages(&t, 1, 1);
        drop(t); // the blocks drop with it
    }

    mod add_to_range {
        use super::*;

        #[test]
        fn first_run_makes_the_first_block() {
            let t = RangeTracker::new(GuestPages::HOST);
            add_pages(&t, 4, 2);
            assert_eq!(
                t.snapshot(),
                vec![BlockSnapshot {
                    block_run: r(4, 2),
                    runs: vec![r(4, 2)]
                }]
            );
        }

        #[test]
        fn refusals_precede_the_lock_and_change_nothing() {
            let t = RangeTracker::new(GuestPages::HOST);
            let e = caught(|| t.add_to_range(0, 0)).unwrap_err();
            raised_at(
                &e,
                RaiseKind::TrackerAlignedLength,
                "length 0x0 is 0 or not whole pages",
            );
            let e = caught(|| t.add_to_range(1, PAGE)).unwrap_err();
            raised_at(
                &e,
                RaiseKind::TrackerAlignedOffset,
                "offset 0x1 is not page aligned",
            );
            assert!(t.snapshot().is_empty());
        }

        #[test]
        fn runs_before_an_empty_block_go_to_the_block_before_it() {
            // The empty block's {0,0} hull is Greater than everything.
            let first = spaced(0, 3);
            let t = tracker_with(vec![block(&first, hull_of(&first)), block(&[], r(0, 0))]);
            add_pages(&t, 100, 1);
            let s = t.snapshot();
            assert_eq!(s[0].runs.last(), Some(&r(100, 1)));
            assert!(s[1].runs.is_empty());
        }

        #[test]
        fn a_full_last_block_opens_a_new_block() {
            // testing.md 7.1 `mutgap_mem_tracker_full_last_block`
            let t = RangeTracker::new(GuestPages::HOST);
            for i in 0..240 {
                add_pages(&t, 2 * i, 1);
            }
            add_pages(&t, 1000, 1);
            let s = t.snapshot();
            assert_eq!(s.len(), 2);
            assert_eq!(s[0].runs.len(), 240);
            assert_eq!(s[1].runs, vec![r(1000, 1)]);
            // An unmap of everything clears only the first block.
            remove_pages(&t, 0, 2000);
            let s = t.snapshot();
            assert_eq!(s.len(), 1);
            assert_eq!(s[0].runs, vec![r(1000, 1)]);
        }

        #[test]
        fn a_full_block_in_the_middle_splits() {
            // testing.md 7.4 `mutgap2_res_tracker_split_adjacent` (the even control).
            let t = RangeTracker::new(GuestPages::HOST);
            for i in 0..240 {
                add_pages(&t, 2 * i, 1);
            }
            add_pages(&t, 1000, 1); // second block
            add_pages(&t, 101, 1); // into the full first block: split 120 + 120
            let s = t.snapshot();
            assert_eq!(s.len(), 3);
            assert_eq!(s[0].runs.len(), 121);
            assert!(s[0].runs.contains(&r(101, 1)));
            assert_eq!(s[1].runs.len(), 120);
            assert_eq!(s[2].runs, vec![r(1000, 1)]);
        }

        #[test]
        fn an_odd_full_block_dies_at_the_split() {
            // testing.md 7.4 `mutgap2_res_tracker_split_odd_death`: 239 single-page runs and a
            // three-page run, an unmap inside it (241 runs), then a map into the block.
            let t = RangeTracker::new(GuestPages::HOST);
            for i in 0..239 {
                add_pages(&t, 4 * i, 1);
            }
            add_pages(&t, 2000, 3);
            add_pages(&t, 5000, 1); // a second block after it
            remove_pages(&t, 2001, 1);
            assert_eq!(t.snapshot()[0].runs.len(), 241);
            let d = catch_death(|| caught(|| t.add_to_range(PAGE, PAGE))).unwrap_err();
            assert_eq!(d, DeathKind::BlockSplitCount.site());
        }

        #[test]
        fn cleanup_removes_inside_blocks_and_skips_the_next() {
            // [5,15) goes to block 1 ([10,11) is Inside it). The cleanup deletes block 2 ([12,13),
            // Inside), and the block that moves into slot 2 ([14,15), InsideEdge) is skipped by
            // `i++` and keeps its run.
            let t = tracker_with(vec![
                block(&[r(0, 1)], r(0, 1)),
                block(&[r(10, 1)], r(10, 1)),
                block(&[r(12, 1)], r(12, 1)),
                block(&[r(14, 1)], r(14, 1)),
                block(&[r(30, 1)], r(30, 1)),
            ]);
            add_pages(&t, 5, 10);
            let s = t.snapshot();
            assert_eq!(s.len(), 4);
            assert_eq!(s[0].runs, vec![r(0, 1)]);
            assert_eq!(
                s[1],
                BlockSnapshot {
                    block_run: r(5, 10),
                    runs: vec![r(5, 10)]
                }
            );
            assert_eq!(s[2].runs, vec![r(14, 1)]);
            assert_eq!(s[3].runs, vec![r(30, 1)]);
        }

        #[test]
        fn cleanup_empties_an_inside_edge_block_that_stays() {
            // testing.md 7.1 `mutgap_mem_tracker_block_end_edge`: a new run that ends where a later
            // block ends empties that block, and the empty block stays.
            let t = tracker_with(vec![
                block(&[r(0, 6)], r(0, 6)),
                block(&[r(10, 1)], r(10, 1)),
                block(&[r(30, 1)], r(30, 1)),
            ]);
            add_pages(&t, 5, 6); // [5, 11): block 0 LessOverlap (target), block 1 InsideEdge
            let s = t.snapshot();
            assert_eq!(s.len(), 3);
            assert_eq!(s[0].runs, vec![r(0, 11)]);
            assert_eq!(
                s[1],
                BlockSnapshot {
                    block_run: r(0, 0),
                    runs: vec![]
                }
            );
            // The empty block then stops lookups: [30,31) is never found.
            assert!(!tested(&t, 30, 1));
        }

        #[test]
        fn cleanup_abort_on_an_earlier_block() {
            // A later block whose hull ends before the new run.
            let t = tracker_with(vec![
                block(&[r(10, 1)], r(10, 1)),
                block(&[r(2, 1)], r(2, 1)),
            ]);
            let d = catch_death(|| caught(|| t.add_to_range(10 * PAGE, 2 * PAGE))).unwrap_err();
            assert_eq!(d, DeathKind::TrackerAddUnordered.site());
        }

        #[test]
        fn cleanup_stops_at_contains_and_greater() {
            let t = tracker_with(vec![
                block(&[r(0, 1)], r(0, 1)),
                block(&[r(10, 10)], r(10, 10)),
            ]);
            add_pages(&t, 12, 2); // block 1 contains it: added to block 1 (inside: no change)
            add_pages(&t, 4, 1); // block 1 Greater: stops
            let s = t.snapshot();
            assert_eq!(s[0].runs, vec![r(0, 1), r(4, 1)]);
            assert_eq!(s[1].runs, vec![r(10, 10)]);
        }

        #[test]
        fn a_full_tracker_compacts_then_refuses() {
            // testing.md 7.1 `newops_range_tracker_limit` (30,720 runs).
            let mut blocks = Vec::new();
            for b in 0..128u32 {
                let runs = spaced(b * 1000, 240);
                blocks.push(block(&runs, hull_of(&runs)));
            }
            let t = tracker_with(blocks);
            let e = caught(|| t.add_to_range(PAGE, PAGE)).unwrap_err();
            raised_at(
                &e,
                RaiseKind::TrackerTooComplex,
                "the range needs more blocks than the tracker holds",
            );
            // The raise leaves the tracker as the compaction left it (nothing to move here).
            assert_eq!(t.snapshot().len(), 128);
        }

        #[test]
        fn compaction_makes_room_and_the_add_lands() {
            // 128 blocks, one of them short: the target is full, compact() packs the blocks,
            // and the loop retries (then splits the still-full target: already compacted).
            let mut blocks = Vec::new();
            for b in 0..128u32 {
                let n = if b == 127 { 10 } else { 240 };
                let runs = spaced(b * 1000, n);
                blocks.push(block(&runs, hull_of(&runs)));
            }
            let t = tracker_with(blocks);
            // Into block 1 (full): compact() moves nothing (every earlier block is full),
            // compacted = true, the retry finds it full again at 128 blocks: refused.
            let e = caught(|| t.add_to_range(1001 * PAGE, PAGE)).unwrap_err();
            raised_at(
                &e,
                RaiseKind::TrackerTooComplex,
                "the range needs more blocks than the tracker holds",
            );
        }

        #[test]
        fn range_too_complex_goes_on_from_the_frame() {
            // The add's handler frame raises again at its end, to its caller's handler, with the
            // raise's own site.
            let mut blocks = Vec::new();
            for b in 0..128u32 {
                let runs = spaced(b * 1000, 240);
                blocks.push(block(&runs, hull_of(&runs)));
            }
            let t = tracker_with(blocks);
            let e = caught(|| t.add_to_range(PAGE, PAGE)).unwrap_err();
            assert_eq!(e.site(), uncaught(RaiseKind::TrackerTooComplex));
        }
    }

    #[test]
    fn split_at_index() {
        let full = spaced(100, 240);
        let t = tracker_with(vec![
            block(&[r(0, 1)], r(0, 1)),
            block(&full, hull_of(&full)),
            block(&[r(9000, 1)], r(9000, 1)),
        ]);
        t.blocks.write().split_at_index(1);
        let s = t.snapshot();
        assert_eq!(s.len(), 4);
        assert_eq!(s[0].runs, vec![r(0, 1)]);
        assert_eq!(s[1].runs, full[..120].to_vec());
        assert_eq!(s[2].runs, full[120..].to_vec());
        assert_eq!(s[3].runs, vec![r(9000, 1)]);
    }

    #[test]
    fn remove_at_index() {
        let t = tracker_with(vec![
            block(&[r(0, 1)], r(0, 1)),
            block(&[r(5, 1)], r(5, 1)),
            block(&[r(9, 1)], r(9, 1)),
        ]);
        let mut b = t.blocks.write();
        b.remove_at_index(0);
        assert_eq!(b.count, 2);
        assert!(b.blocks[2].is_none());
        assert_eq!(b.block(0).at(0), r(5, 1));
        drop(b);
        let d = catch_death(|| t.blocks.write().remove_at_index(5)).unwrap_err();
        assert_eq!(d, DeathKind::TrackerRemoveEmptySlot.site());
        let empty = RangeTracker::new(GuestPages::HOST);
        empty.blocks.write().blocks[0] = Some(RunBlock::new());
        let d = catch_death(|| empty.blocks.write().remove_at_index(0)).unwrap_err();
        assert_eq!(d, DeathKind::TrackerRemoveNoBlocks.site());
    }

    mod compact {
        use super::*;

        #[test]
        fn steals_into_short_blocks_and_drops_emptied_ones() {
            let a = spaced(0, 200);
            let b = spaced(1000, 30);
            let c = spaced(2000, 100);
            let t = tracker_with(vec![
                block(&a, hull_of(&a)),
                block(&b, hull_of(&b)),
                block(&c, hull_of(&c)),
            ]);
            t.blocks.write().compact();
            let s = t.snapshot();
            assert_eq!(s.len(), 2);
            assert_eq!(s[0].runs.len(), 240);
            // The receiving block keeps its hull's start.
            assert_eq!(s[0].block_run, r(0, 2019));
            assert_eq!(s[1].runs.len(), 90);
            assert_eq!(s[1].block_run, hull_of(&c[10..]));
        }

        #[test]
        fn an_emptied_receiver_keeps_offset_zero() {
            // `compact()` keeps `fBlockRun.off` of the receiving block (0 for an empty one) and
            // asserts `toSteal` when the next block is empty (testing.md 7.4)
            let b = spaced(1000, 5);
            let t = tracker_with(vec![block(&[], r(0, 0)), block(&b, hull_of(&b))]);
            t.blocks.write().compact();
            let s = t.snapshot();
            assert_eq!(s.len(), 1);
            assert_eq!(s[0].block_run, r(0, 1009));
        }

        #[test]
        fn an_empty_next_block_dies() {
            // testing.md 7.4 `mutgap_mem_tracker_compact_empty_block`
            let a = spaced(0, 10);
            let t = tracker_with(vec![block(&a, hull_of(&a)), block(&[], r(0, 0))]);
            let d = catch_death(|| t.blocks.write().compact()).unwrap_err();
            assert_eq!(d, DeathKind::TrackerCompactEmptyNext.site());
        }

        #[test]
        fn the_last_block_is_not_a_receiver() {
            let a = spaced(0, 240);
            let b = spaced(1000, 3);
            let t = tracker_with(vec![block(&a, hull_of(&a)), block(&b, hull_of(&b))]);
            t.blocks.write().compact();
            assert_eq!(t.snapshot()[1].runs.len(), 3);
        }
    }

    mod remove_from_range {
        use super::*;

        #[test]
        fn every_case() {
            let t = tracker_with(vec![
                block(&[r(0, 2)], r(0, 2)),
                block(&[r(4, 2), r(8, 2)], r(4, 6)),
                block(&[r(20, 2)], r(20, 2)),
                block(&[r(30, 1)], r(30, 1)),
            ]);
            // [1, 9): block 0 LessOverlap (trimmed), block 1 GreaterOverlap... [4,10) vs
            // [1,9): GreaterOverlap: remove trims; block 2 Greater: done.
            remove_pages(&t, 1, 8);
            let s = t.snapshot();
            assert_eq!(s[0].runs, vec![r(0, 1)]);
            assert_eq!(s[1].runs, vec![r(9, 1)]);
            assert_eq!(s[2].runs, vec![r(20, 2)]);
            // Contains: the block removes inside it and stops.
            remove_pages(&t, 20, 1);
            assert_eq!(t.snapshot()[2].runs, vec![r(21, 1)]);
            // Equal / Inside / InsideEdge: the whole block goes, and the scan stops.
            remove_pages(&t, 0, 40);
            let s = t.snapshot();
            assert_eq!(s.len(), 3);
            assert_eq!(s[0].runs, vec![r(9, 1)]);
        }

        #[test]
        fn refusals_precede_the_lock() {
            let t = RangeTracker::new(GuestPages::HOST);
            let e = caught(|| t.remove_from_range(0, 5)).unwrap_err();
            raised_at(
                &e,
                RaiseKind::TrackerAlignedLength,
                "length 0x5 is 0 or not whole pages",
            );
        }

        #[test]
        fn tasks_map_tracker_blocks() {
            // testing.md 7.1: an unmap spanning blocks clears only the first.
            let t = tracker_with(vec![block(&[r(0, 1)], r(0, 1)), block(&[r(5, 1)], r(5, 1))]);
            remove_pages(&t, 0, 10);
            assert_eq!(t.snapshot().len(), 1);
            assert!(tested(&t, 5, 1));
        }
    }

    mod test_range {
        use super::*;

        #[test]
        fn every_case() {
            let t = tracker_with(vec![
                block(&[r(0, 4)], r(0, 4)),
                block(&[r(10, 4), r(14, 2)], r(10, 6)),
            ]);
            assert!(tested(&t, 0, 4)); // Equal
            assert!(tested(&t, 1, 2)); // Contains
            assert!(tested(&t, 10, 6)); // Equal (hull)
            assert!(!tested(&t, 4, 1)); // Greater after Less: not empty
            assert!(!tested(&t, 9, 8)); // Inside
            assert!(tested(&t, 12, 4)); // GreaterOverlap of block 1
            assert!(!tested(&t, 12, 5)); // LessOverlap: test leaves [16,17)
            // A range spanning both blocks.
            assert!(!tested(&t, 0, 16));
            assert!(!tested(&RangeTracker::new(GuestPages::HOST), 0, 1));
        }

        #[test]
        fn unaligned_and_invalid() {
            let t = tracker_with(vec![block(&[r(0, 1)], r(0, 1))]);
            // A byte range within page 0.
            assert!(caught(|| t.test_range(10, 100)).unwrap());
            // One byte at a page start is 0 pages: tested true even with nothing mapped.
            assert!(caught(|| RangeTracker::new(GuestPages::HOST).test_range(PAGE, 1)).unwrap());
            let e = caught(|| t.test_range(0, 0)).unwrap_err();
            raised_at(&e, RaiseKind::TrackerZeroLength, "length 0");
        }

        #[test]
        fn empty_block_stops_lookups() {
            // testing.md 7.1 `tasks_gap_tracker_empty_block`
            let t = tracker_with(vec![
                block(&[r(0, 1)], r(0, 1)),
                block(&[], r(0, 0)),
                block(&[r(20, 1)], r(20, 1)),
            ]);
            assert!(!tested(&t, 20, 1));
        }
    }

    mod visit_unmapped_ranges {
        use super::*;

        #[test]
        fn holes_in_order() {
            let t = tracker_with(vec![block(&[r(2, 2), r(6, 1)], r(2, 5))]);
            assert_eq!(holes(&t, 0, 10), vec![(0, 2), (4, 2), (7, 3)]);
            assert_eq!(holes(&t, 2, 2), vec![]); // Equal
            assert_eq!(holes(&t, 3, 4), vec![(4, 2)]); // InsideEdge then GreaterOverlap
            assert_eq!(
                holes(&RangeTracker::new(GuestPages::HOST), 5, 1),
                vec![(5, 1)]
            );
        }

        #[test]
        fn later_blocks_and_the_stop() {
            let t = tracker_with(vec![
                block(&[r(0, 1)], r(0, 1)),
                block(&[r(3, 1)], r(3, 1)),
                block(&[r(8, 1)], r(8, 1)),
            ]);
            // Block 1 is Inside [2,6): scanned; block 2 Greater: the loop stops.
            assert_eq!(holes(&t, 2, 4), vec![(2, 1), (4, 2)]);
            // Block 1 Contains [3,4)? [3,4) vs [3,4) Equal: nothing unmapped, stop.
            assert_eq!(holes(&t, 3, 1), vec![]);
        }

        #[test]
        fn the_visitor_stops_at_false_and_runs_unlocked() {
            let t = tracker_with(vec![block(&[r(2, 1), r(5, 1)], r(2, 4))]);
            let mut seen = Vec::new();
            caught_visit(|| {
                t.visit_unmapped_ranges(0, 8 * PAGE, &mut |o, l| {
                    seen.push((o / PAGE, l / PAGE));
                    // Re-enters the tracker exclusively, as the lazy map's visitor does.
                    t.add_to_range(o, l).unwrap();
                    Ok(seen.len() < 2)
                })
            })
            .unwrap();
            assert_eq!(seen, vec![(0, 2), (3, 2)]);
            assert!(tested(&t, 0, 6));
            assert!(!tested(&t, 6, 1));
        }

        #[test]
        fn more_than_16_holes_throw_before_any_visit() {
            // testing.md 7.1 `tasks_map_unmapped_holes`.
            let runs = spaced(1, 16); // pages 1, 3, …, 31: holes 0, 2, …, 30 and [32, 40)
            let t = tracker_with(vec![block(&runs, hull_of(&runs))]);
            let mut visits = 0;
            let e = caught_visit(|| {
                t.visit_unmapped_ranges(0, 40 * PAGE, &mut |_, _| {
                    visits += 1;
                    Ok(true)
                })
            })
            .unwrap_err();
            thrown_at(
                &e,
                RaiseKind::TrackerUnmappedTooComplexTail,
                "the range's unmapped parts are more than the tracker holds",
            );
            assert_eq!(visits, 0);
            // 17 inside holes throw during the scan.
            let runs = spaced(1, 17);
            let t = tracker_with(vec![block(&runs, hull_of(&runs))]);
            let e = caught_visit(|| t.visit_unmapped_ranges(0, 40 * PAGE, &mut |_, _| Ok(true)))
                .unwrap_err();
            thrown_at(
                &e,
                RaiseKind::TrackerUnmappedTooComplex,
                "the range's unmapped parts are more than the tracker holds",
            );
            // Exactly 16 holes pass.
            let runs = spaced(1, 16);
            let t = tracker_with(vec![block(&runs, hull_of(&runs))]);
            assert_eq!(holes(&t, 0, 32).len(), 16);
        }

        #[test]
        fn a_raise_from_the_visitor_leaves_the_loop() {
            // With the lazy map's visitor, whose mapping raises: the raise leaves the loop and the
            // third run is never visited.
            let t = RangeTracker::new(GuestPages::HOST);
            add_pages(&t, 0, 1);
            add_pages(&t, 2, 1);
            add_pages(&t, 4, 1);
            let mut seen = Vec::new();
            let e = caught_visit(|| {
                t.visit_unmapped_ranges(0, 7 * PAGE, &mut |o, l| {
                    seen.push((o / PAGE, l / PAGE));
                    if seen.len() == 2 {
                        return Err(crate::testkit::capture::raise(
                            crate::raise::OnFault::Return,
                            "test-visitor", // a synthetic raise site
                            "the range needs more blocks than the tracker holds",
                        ));
                    }
                    Ok(true)
                })
            })
            .unwrap_err();
            assert_eq!(
                e.detail().unwrap_or_default(),
                "the range needs more blocks than the tracker holds"
            );
            assert_eq!(seen, vec![(1, 1), (3, 1)]);
        }

        #[test]
        fn invalid_length_throws() {
            let e = caught_visit(|| {
                RangeTracker::new(GuestPages::HOST)
                    .visit_unmapped_ranges(0, 0, &mut |_, _| Ok(true))
            })
            .unwrap_err();
            thrown_at(&e, RaiseKind::TrackerZeroLength, "length 0");
        }
    }

    /// testing.md 7.4 `tasks_map_tracker_multiblock`: 481 single-page runs, a map of [0, 1000),
    /// then a lazy probe of [0, 1010). The suite did not establish which `abort()` fires. With the
    /// runs at every other page (an assumption: testing.md does not give their pages): the map of
    /// [0, 1000) lands in block 0, and its cleanup leaves block 1 (hull [480, 959), Contained, so
    /// the loop stops) behind it. The probe then finds [1000, 1010) unmapped; its add targets block
    /// 0 (LessAdjacent), and block 1 is Less than the run: the tracker's add aborts
    /// (`DeathKind::TrackerAddUnordered`).
    #[test]
    fn tasks_map_tracker_multiblock() {
        let t = RangeTracker::new(GuestPages::HOST);
        for i in 0..481 {
            add_pages(&t, 2 * i, 1);
        }
        add_pages(&t, 0, 1000);
        let s = t.snapshot();
        assert_eq!(s[0].runs, vec![r(0, 1000)]);
        assert_eq!(s[1].block_run, r(480, 479));
        assert!(!tested(&t, 0, 1010));
        assert_eq!(holes(&t, 0, 1010), vec![(1000, 10)]);
        let d = catch_death(|| add_pages(&t, 1000, 10)).unwrap_err();
        assert_eq!(d, DeathKind::TrackerAddUnordered.site());
    }

    /// Runs count the tracker's pages, whatever the build's: 16 KiB for an arm64 guest, 4 KiB
    /// for an x86-64 one
    #[test]
    fn runs_count_the_tracker_s_pages() {
        let t = RangeTracker::new(GuestPages::ARM64);
        assert_eq!(
            caught(|| t.calculate_page_run_aligned(3 * 0x4000, 2 * 0x4000)).unwrap(),
            r(3, 2)
        );
        assert!(caught(|| t.calculate_page_run_aligned(0x1000, 0x4000)).is_err());
        assert!(caught(|| t.calculate_page_run_aligned(0, 0x1000)).is_err());
        assert_eq!(
            caught(|| t.calculate_page_run_unaligned(0x10, 0x4000)).unwrap(),
            r(0, 2)
        );
        assert_eq!(r(3, 2).get_range(GuestPages::ARM64), (0xc000, 0x8000));
        let t = RangeTracker::new(GuestPages::X86_64);
        assert_eq!(
            caught(|| t.calculate_page_run_aligned(0x3000, 0x2000)).unwrap(),
            r(3, 2)
        );
        assert_eq!(r(3, 2).get_range(GuestPages::X86_64), (0x3000, 0x2000));
    }
}
