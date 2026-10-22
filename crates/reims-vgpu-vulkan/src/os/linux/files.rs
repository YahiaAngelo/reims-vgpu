//! The process's backing table: which file, from which offset, each byte of the memory the
//! library may make a view of shows, so that a view is made by mapping files and never by
//! remapping addresses, which Linux has no call for (Mach's `mach_vm_remap`).
//!
//! Two kinds of entry:
//! - **Registrations** of memory the library does not map itself: the guest's RAM, a shared
//!   mapping of the files the host names in its memory map (the `guest_memory` lease of the
//!   memory map's device), and the memory of IOSurface's stand-in. One registered again over the
//!   same descriptor and offset is counted, and goes with the last. The C API duplicates the
//!   host's descriptors for each device it makes, so devices over the same RAM register it each
//!   under its own: a byte resolves through the first registration that holds it, and every one
//!   shows the same file pages.
//! - **Runs** the library mapped: every view [`super::vm::LinuxVm`] made, and every range of
//!   [`super::vm::allocate_aliasable`]. A run shows `end - first` bytes of its file from its
//!   offset, with the protection last set on it. Runs never overlap: a new run replaces what of
//!   another it covers, and an unmap or an overwrite drops what of the runs it covers.
//!
//! A byte in a run is found there first; a registration is looked at only for bytes no run
//! holds. Because a view of a view resolves to the file, never to the first view's address, the
//! second view keeps showing the pages after the first goes, as Mach's aliases do.
//!
//! **Mappings.** The table counts the kernel's mappings (VMAs) its runs take ([`mappings`]): one
//! per run, less one for each pair of neighbours the kernel holds as one, a run that starts where
//! the one before it ends, in the same file at the offset that one ends at, with the same
//! protection ([`joined`]). The kernel merges such neighbours whenever it maps or protects one of
//! them, and splits a mapping where part of it is unmapped or protected apart, as the table cuts
//! its runs, so the count follows every view, protect and unmap. Registrations map nothing.
//!
//! The table is a leaf lock: nothing is called under it.

use std::collections::BTreeMap;
use std::os::fd::OwnedFd;
use std::sync::Arc;

use reims_vgpu_core::num::ToU64 as _;
use reims_vgpu_core::sync::Mutex;

use crate::os::GuestRamFile;

/// The file behind a range: an open descriptor, and the file offset of the range's first byte.
#[derive(Clone, Debug)]
pub(crate) struct Backing {
    /// The file, held while any entry or view names it.
    pub(crate) fd: Arc<OwnedFd>,
    /// The file offset of the range's first byte.
    pub(crate) offset: u64,
}

/// A run of the library's own mapping: its end, its file and the protection set on it
/// (`PROT_*`).
#[derive(Clone, Debug)]
struct Run {
    end: u64,
    backing: Backing,
    prot: i32,
}

/// A registration: `first..end` shows `backing`, read and write, `count` times registered.
#[derive(Debug)]
struct Registered {
    first: u64,
    end: u64,
    backing: Backing,
    count: usize,
}

/// One piece of a range as the table resolves it: `length` bytes, `at` bytes into the range,
/// showing `backing`, with protection `prot`.
#[derive(Clone, Debug)]
pub(crate) struct Piece {
    pub(crate) at: u64,
    pub(crate) length: u64,
    pub(crate) backing: Backing,
    pub(crate) prot: i32,
}

/// The table (module docs).
#[derive(Debug, Default)]
struct Table {
    registered: Vec<Registered>,
    runs: BTreeMap<u64, Run>,
    /// The pairs of neighbouring runs the kernel holds as one mapping ([`joined`]).
    joins: usize,
}

/// The process's table.
static TABLE: Mutex<Table> = Mutex::new(Table {
    registered: Vec::new(),
    runs: BTreeMap::new(),
    joins: 0,
});

/// Whether the kernel holds the run `a` from `a_first` and the run `b` from `b_first` as one
/// mapping: `b` starts where `a` ends, in the same file at the offset `a` ends at, with the same
/// protection (module docs).
fn joined(a_first: u64, a: &Run, b_first: u64, b: &Run) -> bool {
    a.end == b_first
        && Arc::ptr_eq(&a.backing.fd, &b.backing.fd)
        && a.backing.offset + (a.end - a_first) == b.backing.offset
        && a.prot == b.prot
}

impl Table {
    /// The kernel's mappings the runs take (module docs).
    fn mappings(&self) -> usize {
        self.runs.len() - self.joins
    }

    /// How many of the runs on either side of `first..run.end` join a run there ([`joined`]).
    fn joins_beside(&self, first: u64, run: &Run) -> usize {
        let before = (self.runs.range(..first).next_back())
            .is_some_and(|(&at, before)| joined(at, before, first, run));
        let after = (self.runs.range(run.end..).next())
            .is_some_and(|(&at, after)| joined(first, run, at, after));
        usize::from(before) + usize::from(after)
    }

    /// Adds a run over bytes no run holds, counting the neighbours it joins.
    fn insert(&mut self, first: u64, run: Run) {
        self.joins += self.joins_beside(first, &run);
        self.runs.insert(first, run);
    }

    /// Removes the run from `first`, uncounting the neighbours it joined.
    fn remove(&mut self, first: u64) -> Option<Run> {
        let run = self.runs.remove(&first)?;
        self.joins -= self.joins_beside(first, &run);
        Some(run)
    }

    /// The run holding byte `at`: its first byte and itself.
    fn run_at(&self, at: u64) -> Option<(u64, &Run)> {
        (self.runs.range(..=at).next_back())
            .filter(|(_, run)| at < run.end)
            .map(|(&first, run)| (first, run))
    }

    /// The first run starting after `at`, before `end`: its first byte.
    fn next_run(&self, at: u64, end: u64) -> Option<u64> {
        self.runs.range(at..end).next().map(|(&first, _)| first)
    }

    /// The registration holding byte `at`.
    fn registered_at(&self, at: u64) -> Option<&Registered> {
        self.registered.iter().find(|r| r.first <= at && at < r.end)
    }

    /// The file behind byte `at`: the run or registration holding it, the file offset of `at`, its
    /// protection, and the end of the bytes from `at` that show the same file on, with the same
    /// protection, up to `end` (a registration's stop where a run starts); `None` where neither
    /// holds it.
    fn file_at(&self, at: u64, end: u64) -> Option<(&Arc<OwnedFd>, u64, i32, u64)> {
        if let Some((first, run)) = self.run_at(at) {
            let offset = run.backing.offset + (at - first);
            return Some((&run.backing.fd, offset, run.prot, run.end.min(end)));
        }
        let registered = self.registered_at(at)?;
        let to = (registered.end.min(end)).min(self.next_run(at, end).unwrap_or(end));
        let offset = registered.backing.offset + (at - registered.first);
        Some((&registered.backing.fd, offset, READ_WRITE, to))
    }

    /// Cuts the run holding `at` in two there, where `at` lies past its first byte: two runs the
    /// kernel holds as one mapping, so the count of mappings stays.
    fn cut(&mut self, at: u64) {
        let Some(first) = self.run_at(at).map(|(first, _)| first).filter(|&f| f < at) else {
            return;
        };
        let Some(run) = self.remove(first) else {
            return;
        };
        let tail = Run {
            end: run.end,
            backing: Backing {
                fd: Arc::clone(&run.backing.fd),
                offset: run.backing.offset + (at - first),
            },
            prot: run.prot,
        };
        self.insert(first, Run { end: at, ..run });
        self.insert(at, tail);
    }

    /// Drops what of the runs lies in `first..end`, keeping the rest of each.
    fn forget(&mut self, first: u64, end: u64) {
        if first >= end {
            return;
        }
        self.cut(first);
        self.cut(end);
        while let Some(at) = self.runs.range(first..end).next().map(|(&at, _)| at) {
            self.remove(at);
        }
    }

    /// `first..end` shows `backing` with protection `prot` from now on (`record`).
    fn record(&mut self, first: u64, end: u64, backing: Backing, prot: i32) {
        self.forget(first, end);
        if first < end {
            self.insert(first, Run { end, backing, prot });
        }
    }

    /// Sets the protection of what of the runs lies in `first..end` (`protect`), cutting a run it
    /// changes where the range ends inside it. A run that has the protection already stays as it
    /// is, as the kernel leaves a mapping that a protect does not change.
    fn protect(&mut self, first: u64, end: u64, prot: i32) {
        if first >= end {
            return;
        }
        for at in [first, end] {
            if self.run_at(at).is_some_and(|(_, run)| run.prot != prot) {
                self.cut(at);
            }
        }
        let mut from = first;
        while let Some(at) = (self.runs.range(from..end))
            .find(|(_, run)| run.prot != prot)
            .map(|(&at, _)| at)
        {
            let Some(mut run) = self.remove(at) else {
                break;
            };
            from = run.end;
            run.prot = prot;
            self.insert(at, run);
        }
    }

    /// Whether every byte of `first..end` has `bit` (`PROT_*`) in its protection: `None` where
    /// a byte lies in neither a run nor a registration.
    fn all_have(&self, first: u64, end: u64, bit: i32) -> Option<bool> {
        let mut all = true;
        let mut at = first;
        while at < end {
            let (_, _, prot, to) = self.file_at(at, end)?;
            all &= prot & bit != 0;
            at = to;
        }
        Some(all)
    }
}

/// The protection of a registration's bytes: read and write.
const READ_WRITE: i32 = libc::PROT_READ | libc::PROT_WRITE;

/// The kernel's mappings the table's runs take (module docs): every view the library made and
/// every range a view can be made of.
pub(crate) fn mappings() -> u64 {
    TABLE.lock().mappings().to_u64()
}

/// The kernel's mappings the table's runs in `first..end` take, as [`mappings`] counts them
/// (tests: a range's count to set beside the kernel's).
#[cfg(test)]
pub(crate) fn mappings_between(first: u64, end: u64) -> u64 {
    let table = TABLE.lock();
    let mut count = 0u64;
    let mut before: Option<(u64, &Run)> = None;
    for (&at, run) in table.runs.range(first..end) {
        if !before.is_some_and(|(b_at, b)| joined(b_at, b, at, run)) {
            count += 1;
        }
        before = Some((at, run));
    }
    count
}

/// The runs the table holds in `first..end`, each a remap's piece or what a cut left of one
/// (tests).
#[cfg(test)]
pub(crate) fn runs_between(first: u64, end: u64) -> usize {
    TABLE.lock().runs.range(first..end).count()
}

/// `length` bytes from `first` show `backing` from now on, with protection `prot`: what of
/// another run they cover goes.
pub(crate) fn record(first: u64, length: u64, backing: Backing, prot: i32) {
    TABLE
        .lock()
        .record(first, first.saturating_add(length), backing, prot);
}

/// `length` bytes from `first` show no file from now on.
pub(crate) fn forget(first: u64, length: u64) {
    TABLE.lock().forget(first, first.saturating_add(length));
}

/// The pieces of `length` bytes from `first`, in order, each bytes of one file that follow on in
/// it with one protection, through the runs and registrations over them (one piece where the runs
/// a view of following pages left follow on, as the kernel maps them as one); `None` where a byte
/// has no file behind it.
pub(crate) fn pieces(first: u64, length: u64) -> Option<Vec<Piece>> {
    let end = first.checked_add(length)?;
    let table = TABLE.lock();
    let mut pieces: Vec<Piece> = Vec::new();
    let mut at = first;
    while at < end {
        let (fd, offset, prot, to) = table.file_at(at, end)?;
        match pieces.last_mut() {
            Some(last)
                if Arc::ptr_eq(&last.backing.fd, fd)
                    && last.backing.offset + last.length == offset
                    && last.prot == prot =>
            {
                last.length += to - at;
            }
            _ => pieces.push(Piece {
                at: at - first,
                length: to - at,
                backing: Backing {
                    fd: Arc::clone(fd),
                    offset,
                },
                prot,
            }),
        }
        at = to;
    }
    Some(pieces)
}

/// `pieces` of a range, as [`pieces`] gave them, show from `target` on from now on, each
/// `piece.at` bytes in: what of other runs they cover goes (a remap's record, under one lock).
pub(crate) fn record_pieces(target: u64, pieces: Vec<Piece>) {
    let mut table = TABLE.lock();
    for piece in pieces {
        let first = target.saturating_add(piece.at);
        let end = first.saturating_add(piece.length);
        table.record(first, end, piece.backing, piece.prot);
    }
}

/// Sets the protection of what of the runs lies in `length` bytes from `first` to `prot`,
/// cutting a run it changes at its ends.
pub(crate) fn protect(first: u64, length: u64, prot: i32) {
    TABLE
        .lock()
        .protect(first, first.saturating_add(length), prot);
}

/// Whether every byte of `length` bytes from `first` lies in a run whose protection is `prot`
/// already, so a protect of them to `prot` changes no mapping: the table's runs have the
/// protections the kernel's mappings of them have, as each map and protect of them records what
/// it set. Bytes of a registration, which the table does not map, never are.
pub(crate) fn protected(first: u64, length: u64, prot: i32) -> bool {
    let Some(end) = first.checked_add(length) else {
        return false;
    };
    let table = TABLE.lock();
    let mut at = first;
    while at < end {
        match table.run_at(at) {
            Some((_, run)) if run.prot == prot => at = run.end,
            _ => return false,
        }
    }
    true
}

/// Whether the table knows every byte of `length` bytes from `first`: `Some(readable)` where
/// each lies in a run or a registration, readable where each has read access; `None` where one
/// lies in neither.
pub(crate) fn readable(first: u64, length: u64) -> Option<bool> {
    let end = first.checked_add(length)?;
    TABLE.lock().all_have(first, end, libc::PROT_READ)
}

/// As [`readable`], for write access.
pub(crate) fn writable(first: u64, length: u64) -> Option<bool> {
    let end = first.checked_add(length)?;
    TABLE.lock().all_have(first, end, libc::PROT_WRITE)
}

/// Whether a byte of `length` bytes from `first` lies in a registration.
pub(crate) fn meets_registration(first: u64, length: u64) -> bool {
    let end = first.saturating_add(length);
    let table = TABLE.lock();
    table
        .registered
        .iter()
        .any(|r| r.first < end && first < r.end)
}

/// A registration of `length` bytes from `first` as showing `backing` (module docs), until it
/// drops.
#[derive(Debug)]
pub(crate) struct Registration {
    first: u64,
    end: u64,
    backing: Backing,
}

impl Registration {
    /// Registers `length` bytes from `first` as showing `backing`: counted again where the same
    /// bytes are registered over the same file and offset.
    pub(crate) fn new(first: u64, length: u64, backing: Backing) -> Registration {
        let end = first.saturating_add(length);
        let mut table = TABLE.lock();
        match table
            .registered
            .iter_mut()
            .find(|r| r.is(first, end, &backing))
        {
            Some(r) => r.count += 1,
            None => table.registered.push(Registered {
                first,
                end,
                backing: backing.clone(),
                count: 1,
            }),
        }
        Registration {
            first,
            end,
            backing,
        }
    }
}

/// The guest's RAM registered while a device made over it lives: the `guest_memory` lease its
/// device holds, made of the host's memory map, and dropped after core's teardown. Views of the
/// guest's pages are made of these files.
#[derive(Debug)]
pub struct GuestMemoryLease {
    _registrations: Vec<Registration>,
}

/// Registers each range of the guest's RAM the host names a file behind: counted where the same
/// range was registered already over the same descriptor (module docs). Warns where the views'
/// budget of mappings is short of the guest's RAM (the `budget` module).
pub fn lease_guest_memory(ranges: impl IntoIterator<Item = GuestRamFile>) -> GuestMemoryLease {
    let registrations: Vec<Registration> = ranges
        .into_iter()
        .map(|r| {
            let backing = Backing {
                fd: r.fd,
                offset: r.offset,
            };
            Registration::new(r.host, r.length, backing)
        })
        .collect();
    let guest_ram = registrations
        .iter()
        .map(|r| r.end - r.first)
        .fold(0u64, u64::saturating_add);
    super::budget::warn_if_short(guest_ram);
    GuestMemoryLease {
        _registrations: registrations,
    }
}

impl Registered {
    /// Whether this registers `first..end` over `backing`.
    fn is(&self, first: u64, end: u64, backing: &Backing) -> bool {
        (self.first, self.end, self.backing.offset) == (first, end, backing.offset)
            && Arc::ptr_eq(&self.backing.fd, &backing.fd)
    }
}

impl Drop for Registration {
    /// Counts the registration down, and drops it at the last.
    fn drop(&mut self) {
        let mut table = TABLE.lock();
        if let Some(at) =
            (table.registered.iter()).position(|r| r.is(self.first, self.end, &self.backing))
        {
            let r = &mut table.registered[at];
            r.count -= 1;
            if r.count == 0 {
                table.registered.remove(at);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file for the tests' backings.
    fn file() -> Arc<OwnedFd> {
        let name = c"rvg-files-test";
        // SAFETY: a NUL-terminated name; the descriptor is owned from here on.
        let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0);
        // SAFETY: a descriptor this test just made, owned once.
        Arc::new(unsafe { <OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(fd) })
    }

    /// The pieces' (offset in the range, length, file offset).
    fn shape(pieces: &[Piece]) -> Vec<(u64, u64, u64)> {
        pieces
            .iter()
            .map(|p| (p.at, p.length, p.backing.offset))
            .collect()
    }

    // The table is the process's: each test uses address ranges of its own, far apart.

    /// A range resolves through the runs over it, cut where they end, then the registrations;
    /// a byte of neither leaves it unresolved
    #[test]
    fn a_range_resolves_through_runs_then_registrations() {
        let f = file();
        let base = 0x7a00_0000_0000;
        let backing = |offset| Backing {
            fd: Arc::clone(&f),
            offset,
        };
        let held = Registration::new(base, 0x10_000, backing(0x100_000));
        record(base + 0x20_000, 0x2000, backing(0x5000), libc::PROT_READ);
        assert_eq!(
            shape(&pieces(base + 0x1000, 0x2000).unwrap()),
            [(0, 0x2000, 0x101_000)]
        );
        assert_eq!(
            shape(&pieces(base + 0x20_000, 0x2000).unwrap()),
            [(0, 0x2000, 0x5000)]
        );
        assert!(pieces(base + 0x10_000, 0x1000).is_none());
        assert!(pieces(base + 0x1f_000, 0x2000).is_none());
        assert_eq!(readable(base + 0x20_000, 0x2000), Some(true));
        assert_eq!(readable(base, 0x10_000), Some(true));
        protect(base + 0x21_000, 0x1000, libc::PROT_NONE);
        assert_eq!(readable(base + 0x20_000, 0x1000), Some(true));
        assert_eq!(readable(base + 0x20_000, 0x2000), Some(false));
        assert_eq!(
            shape(&pieces(base + 0x20_000, 0x2000).unwrap()),
            [(0, 0x1000, 0x5000), (0x1000, 0x1000, 0x6000)]
        );
        assert_eq!(readable(base + 0x30_000, 1), None);
        drop(held);
        assert!(pieces(base, 0x1000).is_none());
        forget(base + 0x20_000, 0x2000);
    }

    /// A run replaces what of another it covers, and an unmap drops what it covers, keeping
    /// each run's other bytes at their file offsets
    #[test]
    fn runs_never_overlap() {
        let f = file();
        let base = 0x7b00_0000_0000;
        let backing = |offset| Backing {
            fd: Arc::clone(&f),
            offset,
        };
        record(
            base,
            0x4000,
            backing(0x10_000),
            libc::PROT_READ | libc::PROT_WRITE,
        );
        record(base + 0x1000, 0x1000, backing(0x90_000), libc::PROT_READ);
        assert_eq!(
            shape(&pieces(base, 0x4000).unwrap()),
            [
                (0, 0x1000, 0x10_000),
                (0x1000, 0x1000, 0x90_000),
                (0x2000, 0x2000, 0x12_000)
            ]
        );
        forget(base + 0x2000, 0x1000);
        assert!(pieces(base, 0x4000).is_none());
        assert_eq!(
            shape(&pieces(base + 0x3000, 0x1000).unwrap()),
            [(0, 0x1000, 0x13_000)]
        );
        forget(base, 0x4000);
        assert!(pieces(base, 1).is_none());
    }

    /// A lease registers each range of the guest's RAM at its file offset, until it drops
    #[test]
    fn a_lease_registers_the_guests_ram_while_held() {
        let f = file();
        let base = 0x7d00_0000_0000;
        let ram = |host, length, offset| GuestRamFile {
            host,
            length,
            fd: Arc::clone(&f),
            offset,
        };
        let lease =
            lease_guest_memory([ram(base, 0x4000, 0), ram(base + 0x10_000, 0x2000, 0x4000)]);
        assert_eq!(
            shape(&pieces(base + 0x1000, 0x1000).unwrap()),
            [(0, 0x1000, 0x1000)]
        );
        assert_eq!(
            shape(&pieces(base + 0x11_000, 0x1000).unwrap()),
            [(0, 0x1000, 0x5000)]
        );
        assert!(pieces(base + 0x4000, 0x1000).is_none());
        drop(lease);
        assert!(pieces(base, 0x1000).is_none());
        assert!(pieces(base + 0x10_000, 0x1000).is_none());
    }

    /// The table counts its runs' mappings as the kernel holds them: neighbours in one file at
    /// following offsets with one protection are one, and a protect or an unmap inside one cuts
    /// it
    #[test]
    fn the_table_counts_the_kernels_mappings() {
        let (f, g) = (file(), file());
        let mut t = Table::default();
        let base = 0x1000_0000;
        let page = 0x1000;
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let at = |i: u64| base + i * page;
        let b = |fd: &Arc<OwnedFd>, offset| Backing {
            fd: Arc::clone(fd),
            offset,
        };
        // Four pages of one file in order: one mapping.
        for i in 0..4 {
            t.record(at(i), at(i + 1), b(&f, i * page), rw);
        }
        assert_eq!(t.mappings(), 1);
        // A page of the file out of order after them, then a page of another file: three.
        t.record(at(4), at(5), b(&f, 0x9000), rw);
        t.record(at(5), at(6), b(&g, 0xa000), rw);
        assert_eq!(t.mappings(), 3);
        // The second page read-only cuts the first mapping in three; read and write again, one.
        t.protect(at(1), at(2), libc::PROT_READ);
        assert_eq!(t.mappings(), 5);
        t.protect(at(1), at(2), rw);
        assert_eq!(t.mappings(), 3);
        // An unmap of the second page cuts it in two; the page mapped again as it was, one.
        t.forget(at(1), at(2));
        assert_eq!(t.mappings(), 4);
        t.record(at(1), at(2), b(&f, page), rw);
        assert_eq!(t.mappings(), 3);
        // Another page of the file over the third: the first mapping in three.
        t.record(at(2), at(3), b(&f, 0x20_000), rw);
        assert_eq!(t.mappings(), 5);
        t.forget(base, at(6));
        assert_eq!((t.mappings(), t.joins, t.runs.len()), (0, 0, 0));
    }

    /// A protect to the protection a run has leaves it whole; one that changes part of a run cuts
    /// it there, and the runs it leaves count as the mappings the kernel holds
    #[test]
    fn a_protect_cuts_only_the_runs_it_changes() {
        let f = file();
        let mut t = Table::default();
        let at = |i: u64| 0x2000_0000 + i * 0x1000;
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let backing = Backing {
            fd: Arc::clone(&f),
            offset: 0,
        };
        t.record(at(0), at(4), backing, rw);
        t.protect(at(1), at(3), rw);
        assert_eq!((t.runs.len(), t.mappings()), (1, 1), "unchanged");
        t.protect(at(1), at(2), libc::PROT_READ);
        assert_eq!((t.runs.len(), t.mappings()), (3, 3), "its second page");
        t.protect(at(0), at(4), libc::PROT_READ);
        assert_eq!(t.mappings(), 1, "all of it read-only: one mapping");
        assert!(t.runs.values().all(|run| run.prot == libc::PROT_READ));
        t.forget(at(0), at(4));
        assert_eq!((t.mappings(), t.joins, t.runs.len()), (0, 0, 0));
    }

    /// Runs that follow on in one file with one protection resolve as one piece, as the kernel
    /// maps them as one; a protection or a file apart cuts it
    #[test]
    fn runs_that_follow_on_resolve_as_one_piece() {
        let (f, g) = (file(), file());
        let base = 0x7e00_0000_0000;
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let backing = |fd: &Arc<OwnedFd>, offset| Backing {
            fd: Arc::clone(fd),
            offset,
        };
        for i in 0..4 {
            record(
                base + i * 0x1000,
                0x1000,
                backing(&f, 0x8000 + i * 0x1000),
                rw,
            );
        }
        assert_eq!(
            shape(&pieces(base + 0x800, 0x3000).unwrap()),
            [(0, 0x3000, 0x8800)]
        );
        protect(base + 0x2000, 0x1000, libc::PROT_READ);
        record(base + 0x3000, 0x1000, backing(&g, 0xb000), rw);
        assert_eq!(
            shape(&pieces(base, 0x4000).unwrap()),
            [
                (0, 0x2000, 0x8000),
                (0x2000, 0x1000, 0xa000),
                (0x3000, 0x1000, 0xb000)
            ]
        );
        forget(base, 0x4000);
    }

    /// A range is protected where every byte lies in a run of that protection: not where one
    /// lies in a registration or in neither
    #[test]
    fn protected_is_runs_of_the_protection_alone() {
        let f = file();
        let base = 0x7f00_0000_0000;
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let backing = || Backing {
            fd: Arc::clone(&f),
            offset: 0,
        };
        record(base, 0x2000, backing(), rw);
        record(base + 0x2000, 0x1000, backing(), rw);
        assert!(protected(base, 0x3000, rw));
        assert!(!protected(base, 0x3000, libc::PROT_READ));
        assert!(!protected(base, 0x4000, rw), "a byte of no run");
        let held = Registration::new(base + 0x3000, 0x1000, backing());
        assert!(!protected(base, 0x4000, rw), "a byte of a registration");
        assert_eq!(writable(base, 0x4000), Some(true));
        protect(base + 0x1000, 0x1000, libc::PROT_READ);
        assert_eq!(
            (readable(base, 0x4000), writable(base, 0x4000)),
            (Some(true), Some(false))
        );
        assert_eq!(writable(base, 0x5000), None);
        drop(held);
        forget(base, 0x3000);
    }

    /// The same bytes registered again over the same file and offset are counted: they stay
    /// until the last registration drops
    #[test]
    fn registrations_are_counted() {
        let f = file();
        let base = 0x7c00_0000_0000;
        let backing = || Backing {
            fd: Arc::clone(&f),
            offset: 0,
        };
        let first = Registration::new(base, 0x1000, backing());
        let second = Registration::new(base, 0x1000, backing());
        drop(first);
        assert!(pieces(base, 0x1000).is_some());
        drop(second);
        assert!(pieces(base, 0x1000).is_none());
    }
}
