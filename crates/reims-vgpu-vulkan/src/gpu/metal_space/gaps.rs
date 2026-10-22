//! The gaps of Metal's space between its held runs, for the lowest start where a range fits
//! ([`super::MetalSpace::lowest_gap`]): a search and each change of the held runs take
//! O(log n), where a walk of every held run from the space's base, under the process's lock at
//! each placement, took O(n) (1.6, 8.3 and 33 µs at 1k, 5k and 20k held runs).
//!
//! The search answers what that walk answers, in every case. The walk tries [`BASE`] first, and
//! after each held run, in order, the run's end rounded up to [`SLOT`] where that lies past the
//! start it tried last, and takes the first start whose `span` bytes end at or before the next
//! run's start, or the last start tried, past every run. So the starts it can take are one before
//! every run, `BASE`, and one after each run, the larger of `BASE` and the run's end rounded up:
//! a run lying wholly below the start tried before it changes nothing, as its own start is no
//! larger. Each start's room is the bytes from it to the next run's start, none where that run
//! starts below it, and every byte after the last run. The walk takes the first start, in the
//! runs' order, whose room holds the span, and so does the search: a treap of the gaps after the
//! runs, by the run's start, each node holding the most room in its subtree, below the gap before
//! every run, kept apart.

use std::collections::BTreeMap;
use std::ops::Bound;

use super::{BASE, SLOT};

/// No node.
const NIL: usize = usize::MAX;

/// The gap after one held run.
#[derive(Debug, Clone, Copy)]
struct Node {
    /// The run's start, which orders the gaps as the runs.
    key: u64,
    /// The gap's start.
    at: u64,
    /// Its room ([`room`]).
    room: u64,
    /// The most room of a gap in its subtree.
    most: u64,
    priority: u32,
    left: usize,
    right: usize,
}

/// The gaps after the held runs, and the gap before every run (module docs).
#[derive(Debug)]
pub(super) struct Gaps {
    nodes: Vec<Node>,
    /// The places in `nodes` of the nodes removed, for the next made.
    free: Vec<usize>,
    root: usize,
    /// The room of the gap before every run, from [`BASE`].
    before: u64,
    /// The state the nodes' priorities are drawn from.
    seed: u32,
}

/// A gap's room, from `at` to `next`, the next run's start, where one follows: one more than its
/// bytes, so that a span of `span` bytes fits where the room is larger than `span`, and 0 where
/// the next run starts below `at`, which no span fits; every byte where none follows.
fn room(at: u64, next: Option<u64>) -> u64 {
    match next {
        None => u64::MAX,
        Some(next) => next
            .checked_sub(at)
            .map_or(0, |bytes| bytes.saturating_add(1)),
    }
}

/// The start of the gap after a run that ends at `end`: the larger of [`BASE`] and `end` rounded
/// up to [`SLOT`]; past every address where that rounding would pass the end of the 64-bit space,
/// which no span ends past (module docs).
fn gap_at(end: u64) -> u64 {
    end.checked_next_multiple_of(SLOT)
        .map_or(u64::MAX, |at| at.max(BASE))
}

impl Default for Gaps {
    fn default() -> Gaps {
        Gaps::new()
    }
}

impl Gaps {
    /// No held run: one gap, from [`BASE`], of every byte.
    pub(super) const fn new() -> Gaps {
        Gaps {
            nodes: Vec::new(),
            free: Vec::new(),
            root: NIL,
            before: u64::MAX,
            seed: 0x9e37_79b9,
        }
    }

    /// The lowest start the walk takes for `span` bytes (module docs); `None` where no gap's room
    /// holds them, which only a span of every byte meets.
    pub(super) fn lowest(&self, span: u64) -> Option<u64> {
        if self.before > span {
            return Some(BASE);
        }
        let at = self.first_fit(self.root, span)?;
        Some(self.node(at).at)
    }

    /// The run `start` of `held`, which now holds it: its gap, and the room of the gap before it.
    pub(super) fn made(&mut self, held: &BTreeMap<u64, u64>, start: u64) {
        let Some(&end) = held.get(&start) else {
            return;
        };
        let next = (held
            .range((Bound::Excluded(start), Bound::Unbounded))
            .next())
        .map(|(&next, _)| next);
        let at = gap_at(end);
        self.insert(start, at, room(at, next));
        self.ends_before(held, start, Some(start));
    }

    /// The run `start` let go of, which `held` now lacks: its gap goes, and the gap before it
    /// runs on to the next run's start.
    pub(super) fn gone(&mut self, held: &BTreeMap<u64, u64>, start: u64) {
        self.remove(start);
        let next = held.range(start..).next().map(|(&next, _)| next);
        self.ends_before(held, start, next);
    }

    /// The gap before `start` among `held`'s runs, the gap before every run where none starts
    /// below it, ends at `next`, a run's start, or runs on past every run.
    fn ends_before(&mut self, held: &BTreeMap<u64, u64>, start: u64, next: Option<u64>) {
        match held.range(..start).next_back() {
            None => self.before = room(BASE, next),
            Some((&prev, &end)) => self.set_room(self.root, prev, room(gap_at(end), next)),
        }
    }

    fn node(&self, at: usize) -> &Node {
        &self.nodes[at]
    }

    fn node_mut(&mut self, at: usize) -> &mut Node {
        &mut self.nodes[at]
    }

    /// The most room in the subtree at `at`; none for no node.
    fn most(&self, at: usize) -> u64 {
        if at == NIL { 0 } else { self.node(at).most }
    }

    /// The node at `at`'s most room, from its own and its children's.
    fn update(&mut self, at: usize) {
        let node = *self.node(at);
        let most = node
            .room
            .max(self.most(node.left))
            .max(self.most(node.right));
        self.node_mut(at).most = most;
    }

    /// The first gap, in the runs' order, in the subtree at `at` whose room holds `span` bytes.
    fn first_fit(&self, at: usize, span: u64) -> Option<usize> {
        if self.most(at) <= span {
            return None;
        }
        let node = self.node(at);
        if let Some(found) = self.first_fit(node.left, span) {
            return Some(found);
        }
        if node.room > span {
            return Some(at);
        }
        self.first_fit(node.right, span)
    }

    /// The next priority: a xorshift step.
    fn priority(&mut self) -> u32 {
        let mut x = self.seed;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.seed = x;
        x
    }

    /// The subtree at `at` split by `key`: the gaps after runs that start below it, and the rest.
    fn split(&mut self, at: usize, key: u64) -> (usize, usize) {
        if at == NIL {
            return (NIL, NIL);
        }
        if self.node(at).key < key {
            let (low, high) = self.split(self.node(at).right, key);
            self.node_mut(at).right = low;
            self.update(at);
            (at, high)
        } else {
            let (low, high) = self.split(self.node(at).left, key);
            self.node_mut(at).left = high;
            self.update(at);
            (low, at)
        }
    }

    /// The subtrees at `low` and `high` as one, every key of `low` below every key of `high`.
    fn merge(&mut self, low: usize, high: usize) -> usize {
        if low == NIL {
            return high;
        }
        if high == NIL {
            return low;
        }
        if self.node(low).priority > self.node(high).priority {
            let right = self.merge(self.node(low).right, high);
            self.node_mut(low).right = right;
            self.update(low);
            low
        } else {
            let left = self.merge(low, self.node(high).left);
            self.node_mut(high).left = left;
            self.update(high);
            high
        }
    }

    /// The gap after the run `key`, from `at`, of `room`.
    fn insert(&mut self, key: u64, at: u64, room: u64) {
        let node = Node {
            key,
            at,
            room,
            most: room,
            priority: self.priority(),
            left: NIL,
            right: NIL,
        };
        let made = match self.free.pop() {
            Some(place) => {
                *self.node_mut(place) = node;
                place
            }
            None => {
                self.nodes.push(node);
                self.nodes.len() - 1
            }
        };
        let (low, high) = self.split(self.root, key);
        let low = self.merge(low, made);
        self.root = self.merge(low, high);
    }

    /// The gap after the run `key` goes.
    fn remove(&mut self, key: u64) {
        let (low, rest) = self.split(self.root, key);
        let (gone, high) = match key.checked_add(1) {
            Some(after) => self.split(rest, after),
            None => (rest, NIL),
        };
        if gone != NIL {
            self.free.push(gone);
        }
        self.root = self.merge(low, high);
    }

    /// The gap after the run `key`, in the subtree at `at`, has `room` from now on.
    fn set_room(&mut self, at: usize, key: u64, room: u64) {
        if at == NIL {
            return;
        }
        let node = *self.node(at);
        if key < node.key {
            self.set_room(node.left, key, room);
        } else if key > node.key {
            self.set_room(node.right, key, room);
        } else {
            self.node_mut(at).room = room;
        }
        self.update(at);
    }
}
