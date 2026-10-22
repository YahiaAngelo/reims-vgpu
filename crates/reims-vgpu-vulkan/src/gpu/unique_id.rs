//! The pipeline states made with each resource ID, which decide which state an indirect
//! command's record names ([`crate::gpu::icb`]).
//!
//! A state's `uniqueIdentifier` (SPI, the guest's info command 460) is the runtime's
//! (`ComputePipelineState::unique_identifier`, `RenderPipelineState::unique_identifier`), read
//! as the state is made: one nonzero value for the state's life, no other live state's, and a
//! second state made from an identical record a new one. A state made without
//! `supportIndirectCommandBuffers` answers one of a process-wide count, where Metal's is an
//! address that differs from run to run (`plain-state-unique-identifier`); one made with it
//! answers 2^36 plus an offset in the runtime's allocation of such states, which reuses a
//! released state's identifier, the last released first, by kind, as Metal's does, at offsets
//! that are not Metal's (`unmeasured-unique-identifier`: Metal's follow its allocator and the
//! process's history, as how the process was started).
//!
//! **By resource ID.** An indirect command's record names its pipeline state by the state's
//! resource ID, and a record the GPU wrote (a kernel's `set_compute_pipeline_state` or
//! `set_render_pipeline_state` of a state it read from an argument buffer, or a blit's copy) names
//! one no setter gave the command. Metal's `getPipelineStateUniqueIdentifier` answers that
//! state's own identifier (measured on the M3 Max, macOS 26.7: of a compute and of a render
//! command a kernel encoded, and of the second of two states made from one descriptor), and still
//! answers it once the state was released, though a state made later took its ID (measured). So
//! each state of a nonzero resource ID, one made with `supportIndirectCommandBuffers`, is kept by
//! that ID and its device with its identifier and the points of [`now`]'s order it was made and
//! released at ([`made`], [`released`], [`holders`]): a state holds its ID from its creation to
//! its release. The runtime gives a state's ID back only once the state goes, so states its plain
//! allocation numbered hold one ID one after another; a state placed at the ID of one still live
//! holds it beside that one. The [`HISTORY`] states made last with each ID are kept. Which of
//! them a record names is decided by when it can have been written ([`crate::gpu::icb`]); the
//! runtime's own answer there (`pipeline_unique_identifier`) names a live holder, and none where
//! a released state's ID was taken while the GPU could write the record.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use reims_vgpu_core::sync::Mutex;

/// The last point handed out of the order of what decides which state a record names: each
/// state's creation and each span in which the GPU may write records ([`now`]).
static ORDER: AtomicU64 = AtomicU64::new(0);

/// The next point of that order, later than every point handed out before, on any thread.
pub fn now() -> u64 {
    // One location's read-modify-writes take one total order, so a point taken after another
    // returned is larger.
    ORDER.fetch_add(1, Ordering::SeqCst) + 1
}

/// How many of the states made last with one resource ID on one device are kept ([`History`]).
pub const HISTORY: usize = 16;

/// A point of [`now`]'s order no point reaches: the release of a state still live.
pub const LIVE: u64 = u64::MAX;

/// A state made with a resource ID, which held the ID from `made` to `gone`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Holder {
    /// Its identifier.
    pub identifier: u64,
    /// The point of [`now`]'s order it was made at.
    pub made: u64,
    /// The point it was released at; [`LIVE`] while it lives.
    pub gone: u64,
}

/// The states made with one resource ID on one device, oldest first: the [`HISTORY`] made last,
/// those released first giving way to newer ones.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct History {
    /// The states kept.
    pub holders: VecDeque<Holder>,
    /// The point until which a state no longer kept may have held the ID: 0 where none was
    /// dropped, [`LIVE`] where one was dropped while it lived.
    pub lost_until: u64,
}

/// The states made with each nonzero resource ID, by the registry ID of the runtime's device
/// they were made on and that ID.
static OF_ID: Mutex<BTreeMap<(u64, u64), History>> = Mutex::new(BTreeMap::new());

/// Keeps the state of identifier `identifier` the runtime just made on the device of registry ID
/// `device` with resource ID `id` by that ID, with the point it was made at ([`holders`]). A
/// state of ID 0, which no indirect command can name, is kept by none.
pub fn made(device: u64, id: u64, identifier: u64) {
    if id == 0 {
        return;
    }
    let mut of_id = OF_ID.lock();
    let history = of_id.entry((device, id)).or_default();
    // Taken under the lock, so each ID's states are kept in the order they were made.
    let made = now();
    history.holders.push_back(Holder {
        identifier,
        made,
        gone: LIVE,
    });
    if history.holders.len() > HISTORY {
        // The oldest released one, else the oldest.
        let at = (history.holders.iter())
            .position(|holder| holder.gone != LIVE)
            .unwrap_or(0);
        if let Some(dropped) = history.holders.remove(at) {
            history.lost_until = history.lost_until.max(dropped.gone);
        }
    }
}

/// The states made on the device of registry ID `device` with resource ID `id`; `None` where
/// none was.
pub fn holders(device: u64, id: u64) -> Option<History> {
    OF_ID.lock().get(&(device, id)).cloned()
}

/// Notes that the live state of identifier `identifier` made on the device of registry ID
/// `device` with resource ID `id` was released, now: it no longer holds the ID. One of ID 0
/// changes nothing. A released state's identifier may be a later state's too, so the live one
/// goes.
pub fn released(device: u64, id: u64, identifier: u64) {
    if id == 0 {
        return;
    }
    let gone = now();
    let mut of_id = OF_ID.lock();
    let holder = (of_id.get_mut(&(device, id))).and_then(|history| {
        history
            .holders
            .iter_mut()
            .find(|h| h.identifier == identifier && h.gone == LIVE)
    });
    if let Some(holder) = holder {
        holder.gone = gone;
    }
}

/// Whether a state kept by its nonzero resource ID on the device of registry ID `device` was both
/// made and released from point `from` up to point `to` of [`now`]'s order.
pub fn held_within(device: u64, from: u64, to: u64) -> bool {
    let of_id = OF_ID.lock();
    let of_device = of_id.range((device, 0)..=(device, u64::MAX));
    of_device
        .flat_map(|(_, history)| &history.holders)
        .any(|holder| holder.made >= from && holder.gone != LIVE && holder.gone <= to)
}

/// What these keep is process-wide, so tests over a real device's states need a process each, as
/// `cargo nextest` gives them; these use registry IDs no device answers.
#[cfg(test)]
mod tests {
    use reims_vgpu_core::num::ToU64;

    use super::*;

    /// The states made with a nonzero resource ID are kept by it, in the order they were made,
    /// with when each was released, on their own device only, the [`HISTORY`] made last, a
    /// released one giving way first; one of ID 0 is kept by none, and its release changes
    /// nothing; a release takes the live state of an identifier a released one had too
    #[test]
    fn states_are_kept_by_resource_id_in_the_order_they_were_made() {
        // A registry ID no device answers, so no other test's states meet these.
        let device = u64::MAX - 7;
        let before = now();
        let first = 101;
        made(device, 3, first);
        let kept = holders(device, 3).unwrap();
        assert_eq!((kept.holders.len(), kept.lost_until), (1, 0));
        let holder = kept.holders[0];
        assert_eq!((holder.identifier, holder.gone), (first, LIVE));
        assert!(holder.made > before && holder.made < now());
        released(device, 0, first);
        assert_eq!(holders(device, 3).unwrap().holders[0].gone, LIVE);
        released(device, 3, first);
        // A later state takes the released one's identifier, as the runtime and Metal reuse it.
        let later = first;
        made(device, 3, later);
        released(device, 3, later);
        made(device, 3, later);
        let kept = holders(device, 3).unwrap();
        let (a, b, c) = (kept.holders[0], kept.holders[1], kept.holders[2]);
        assert_eq!([a.identifier, b.identifier, c.identifier], [first; 3]);
        assert!(a.made < a.gone && a.gone < b.made && b.made < b.gone && b.gone < c.made);
        assert_eq!(c.gone, LIVE);
        assert_eq!(holders(device - 1, 3), None);
        made(device, 0, 7);
        assert_eq!(holders(device, 0), None);
        // Past HISTORY, the released first states give way, then the oldest live one.
        let identifiers: Vec<u64> = (0..HISTORY.to_u64() - 1).map(|i| 200 + i).collect();
        for &identifier in &identifiers {
            made(device, 3, identifier);
        }
        let kept = holders(device, 3).unwrap();
        assert_eq!(kept.lost_until, b.gone);
        let kept_identifiers: Vec<_> = kept.holders.iter().map(|h| h.identifier).collect();
        assert_eq!(kept_identifiers[0], first);
        assert_eq!(kept_identifiers[1..], identifiers[..]);
        made(device, 3, 999);
        assert_eq!(holders(device, 3).unwrap().lost_until, LIVE);
    }
}
