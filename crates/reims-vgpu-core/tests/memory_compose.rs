//! The memory area composed with the order containers, on
//! `FakePlatform`, with no seam fakes between them.

#![cfg(feature = "testkit")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation
)] // A failure is the assertion

use std::sync::Arc;

use parking_lot::Mutex;

use reims_vgpu_core::collections::BucketOrderMap;
use reims_vgpu_core::collections::ListOrderMap;
use reims_vgpu_core::guest::records::GuestPages;
use reims_vgpu_core::host::PhysRange;
use reims_vgpu_core::memory::cursor::{Cursor, CursorRange, WritableCursor};
use reims_vgpu_core::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
use reims_vgpu_core::memory::map::{MemoryMap, MemoryMapRange};
use reims_vgpu_core::memory::mapping::Mapping;
use reims_vgpu_core::memory::page_table::{PageTableLog, PageTableRead, PageTableWalker};
use reims_vgpu_core::memory::tracker::RangeTracker;
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::rt::Hold;
use reims_vgpu_core::testkit::calllog::Arg;
use reims_vgpu_core::testkit::guest_ram::GuestRam;
use reims_vgpu_core::testkit::platform::FakePlatform;

type Map = MemoryMap<FakePlatform>;

/// The guest's page, which page tables name.
const PAGE: u64 = reims_vgpu_core::guest::records::APV_PAGE_SIZE;

fn plat() -> Arc<FakePlatform> {
    Arc::new(FakePlatform::new())
}

fn pattern(n: u64, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(13).wrapping_add(seed))
        .collect()
}

fn pr(pa: u64, len: u64) -> PhysRange {
    PhysRange {
        physical_address: pa,
        physical_length: len,
    }
}

/// A map over `ram`, whose ranges enter the descriptor in `order` (indices into `ram.ranges()`).
fn map_in_order(p: &Arc<FakePlatform>, ram: &GuestRam, order: &[usize]) -> Arc<Map> {
    p.fake_vm().adopt(ram);
    let all = ram.ranges();
    let mut d = MemoryMapDescriptor::new();
    for &i in order {
        let r = &all[i];
        d.add_range(GuestPhysicalRange {
            physical_address: r.physical_address,
            physical_length: r.region.len(),
            host_address: r.host(),
        });
    }
    Map::new(p.clone(), &d, GuestPages::HOST).unwrap()
}

/// The descriptor's order is not the map's: init sorts by start, then length, and every later
/// consumer (an isolated task's copy, lookups) sees the sorted order.
#[test]
fn a_scrambled_descriptor_is_sorted_and_copied_in_that_order() {
    let p = plat();
    let ram = GuestRam::new();
    // Nine ranges, one of them ending mid-page (0x6000 bytes).
    for i in 1..=9u64 {
        let len = if i == 4 { 0x6000 } else { PAGE };
        ram.add_range(i * 0x10_0000, len).unwrap();
    }
    let scrambled = [4usize, 1, 8, 0, 6, 2, 7, 3, 5];
    let m = map_in_order(&p, &ram, &scrambled);

    // The map's order: by start, then length.
    let all = ram.ranges();
    let mut expect: Vec<MemoryMapRange> = scrambled
        .iter()
        .map(|&i| MemoryMapRange {
            physical_address: all[i].physical_address,
            physical_end: all[i].physical_address + all[i].region.len(),
            physical_length: all[i].region.len(),
            virtual_address: all[i].host(),
        })
        .collect();
    expect.sort_by_key(|r| (r.physical_address, r.physical_length));
    assert_eq!(m.ranges(), expect.as_slice());
    assert!(
        m.ranges()
            .windows(2)
            .all(|w| w[0].physical_end <= w[1].physical_address)
    );

    // An isolated task's service copies the map: the same ranges, in the sorted order.
    let service = m.copy();
    assert_eq!(service.ranges(), m.ranges());
    let starts: Vec<u64> = service
        .ranges()
        .iter()
        .map(|r| r.physical_address)
        .collect();
    assert_eq!(
        starts,
        (1..=9u64).map(|i| i * 0x10_0000).collect::<Vec<_>>()
    );

    // Both sides translate to the same host memory: a host write is a service read.
    let bytes = pattern(PAGE, 3);
    assert!(m.write(0x70_0000, &bytes));
    let mut back = vec![0u8; PAGE as usize];
    assert!(service.read(0x70_0000, &mut back));
    assert_eq!(back, bytes);
    // First containing range decides; a straddle across two ranges is refused.
    assert!(
        service
            .virtual_address_for_physical(0x70_0000 + 0x3000, 0x2000)
            .is_null()
    );
}

/// The pages a task's page table maps, walked through the memory map's `read`, become
/// mappings, tracker runs and cursor ranges; a write through a cursor lands in guest RAM.
#[test]
fn a_page_table_walk_over_the_map_feeds_mappings_the_tracker_and_a_cursor() {
    let p = plat();
    // 16 guest pages at 0x100000 (page 0x40 on a 16 KiB guest page): the first holds the
    // (depth 1) table, the next ones hold data.
    let ram = GuestRam::with_range(0x10_0000, 16 * PAGE).unwrap();
    let m = map_in_order(&p, &ram, &[0]);
    let root = (0x10_0000 / PAGE) as u32;
    // Virtual page i -> guest page, deliberately out of physical order.
    let leaves: [u32; 4] = [root + 5, root + 2, root + 7, root + 3];
    for (i, &page) in leaves.iter().enumerate() {
        assert!(ram.write_u32(u64::from(root) * PAGE + 4 * i as u64, page));
        assert!(ram.write(u64::from(page) * PAGE, &pattern(PAGE, i as u8 + 1)));
    }

    let read_map = m.clone();
    let reads: Arc<Mutex<Vec<(u64, usize)>>> = Arc::default();
    let seen = reads.clone();
    let read: PageTableRead = Arc::new(
        move |_: Hold<'_>, _: OnFault, pa: u64, dst: &mut [u8]| -> RaiseResult<bool> {
            seen.lock().push((pa, dst.len()));
            Ok(read_map.read(pa, dst))
        },
    );
    let log: PageTableLog = Arc::new(|_, _| {});
    let walker = PageTableWalker::new(GuestPages::HOST, root, 1, false, read, log);

    let mut visits: Vec<(u64, u32, u32)> = Vec::new();
    walker
        .visit_pages_in_range(
            Hold::NONE,
            OnFault::Die,
            0,
            4 * PAGE,
            &mut |off, page, flag| {
                visits.push((off, page, flag));
                Ok(true)
            },
        )
        .unwrap();
    // One root read covering the four touched entries, through the map.
    assert_eq!(*reads.lock(), vec![(u64::from(root) * PAGE, 16)]);
    assert_eq!(
        visits,
        leaves
            .iter()
            .enumerate()
            .map(|(i, &pg)| (i as u64 * PAGE, pg, 0))
            .collect::<Vec<_>>()
    );

    // Map each visited page (virtual pages 0, 1 and 3 first), tracking what is mapped.
    let tracker = RangeTracker::new(GuestPages::HOST);
    let mut by_page: Vec<Arc<Mapping<FakePlatform>>> = Vec::new();
    for &(off, page, _) in &visits {
        let mp = m
            .new_contiguous_mapping(u64::from(page) * PAGE, PAGE, 0)
            .unwrap();
        if off != 2 * PAGE {
            tracker.add_to_range(off, PAGE).unwrap();
        }
        by_page.push(mp);
    }
    assert!(!tracker.test_range(0, 4 * PAGE).unwrap());
    let mut holes = Vec::new();
    tracker
        .visit_unmapped_ranges(0, 4 * PAGE, &mut |off, len| {
            holes.push((off, len));
            Ok(true)
        })
        .unwrap();
    assert_eq!(holes, vec![(2 * PAGE, PAGE)]);
    tracker.add_to_range(2 * PAGE, PAGE).unwrap();
    assert!(tracker.test_range(0, 4 * PAGE).unwrap());

    // A cursor over the four mappings reads the task's virtual range in order, across the
    // page boundaries; one range mapping of the same pages reads the same bytes.
    let ranges: Vec<CursorRange> = by_page
        .iter()
        .map(|mp| CursorRange {
            address: mp.virtual_address(),
            length: mp.length(),
        })
        .collect();
    let expect: Vec<u8> = (0..4u8).flat_map(|i| pattern(PAGE, i + 1)).collect();
    let mut c = Cursor::with_ranges(ranges.clone());
    let mut got = vec![0u8; expect.len()];
    c.read_bytes(&mut got).unwrap();
    assert_eq!(got, expect);
    assert_eq!(c.remaining(), 0);

    let phys: Vec<PhysRange> = leaves
        .iter()
        .map(|&pg| pr(u64::from(pg) * PAGE, PAGE))
        .collect();
    let whole = m.new_range_mapping(&phys, 4 * PAGE, 1).unwrap();
    assert!(!whole.writable());
    let mut c2 = Cursor::with_ranges(vec![CursorRange {
        address: whole.virtual_address(),
        length: whole.length(),
    }]);
    let mut got2 = vec![0u8; expect.len()];
    c2.read_bytes(&mut got2).unwrap();
    assert_eq!(got2, expect);

    // A write straddling virtual pages 1 and 2 goes through two mappings (shared remaps) to
    // guest pages 0x42 and 0x47 (on a 16 KiB guest page), where the memory map reads it back.
    let mut w = WritableCursor::with_ranges(ranges);
    let note = pattern(0x20, 0xa0);
    w.write_bytes_at(2 * PAGE - 0x10, &note).unwrap();
    p.fake_vm().sync();
    let mut tail = vec![0u8; 0x10];
    assert!(m.read(u64::from(root + 2) * PAGE + PAGE - 0x10, &mut tail));
    assert_eq!(tail, note[..0x10]);
    let mut head = vec![0u8; 0x10];
    assert!(m.read(u64::from(root + 7) * PAGE, &mut head));
    assert_eq!(head, note[0x10..]);
}

/// Mappings held in the order containers are released in the container's order: draining a
/// `ListOrderMap` deallocates in its iteration order, and draining a `BucketOrderMap` in
/// ascending key order, not insertion order.
#[test]
fn mappings_in_the_order_containers_tear_down_in_container_order() {
    let p = plat();
    let ram = GuestRam::with_range(0x10_0000, 64 * PAGE).unwrap();
    let m = map_in_order(&p, &ram, &[0]);
    // Keys spread so that ascending order is not insertion order.
    let keys: Vec<u32> = vec![17, 3, 40, 9, 33, 1, 25, 12, 48, 6, 30, 21];

    let fill = |i: usize| m.new_contiguous_mapping(0x10_0000 + i as u64 * PAGE, PAGE, 0);

    // The hash map.
    let mut map: ListOrderMap<Arc<Mapping<FakePlatform>>> = ListOrderMap::new();
    for (i, &k) in keys.iter().enumerate() {
        assert!(map.try_insert(k, fill(i).unwrap()).is_ok());
    }
    let order: Vec<usize> = map
        .iter()
        .map(|(_, mp)| mp.virtual_address().addr())
        .collect();
    let mark = p.call_log().mark();
    drop(map.drain());
    assert!(map.is_empty());
    let freed: Vec<usize> = freed_addrs(&p, mark);
    assert_eq!(freed, order);
    assert_eq!(freed.len(), keys.len());

    // Ascending keys.
    let mut dict: BucketOrderMap<Arc<Mapping<FakePlatform>>> = BucketOrderMap::new();
    let mut by_key = Vec::new();
    for (i, &k) in keys.iter().enumerate() {
        let mp = fill(i).unwrap();
        by_key.push((k, mp.virtual_address().addr()));
        assert!(dict.insert(k, mp).is_none());
    }
    by_key.sort_unstable();
    let ascending: Vec<usize> = by_key.iter().map(|&(_, a)| a).collect();
    let order: Vec<usize> = dict
        .values()
        .map(|mp| mp.virtual_address().addr())
        .collect();
    assert_eq!(order, ascending);
    let mark = p.call_log().mark();
    drop(dict.drain());
    assert!(dict.is_empty());
    let freed = freed_addrs(&p, mark);
    assert_eq!(freed, ascending);

    // A mapping still retained elsewhere outlives its removal from a container.
    let mut dict: BucketOrderMap<Arc<Mapping<FakePlatform>>> = BucketOrderMap::new();
    let kept = fill(0).unwrap();
    dict.insert(7, kept.clone());
    let mark = p.call_log().mark();
    drop(dict.remove(7));
    assert!(freed_addrs(&p, mark).is_empty());
    drop(kept);
    assert_eq!(freed_addrs(&p, mark).len(), 1);
}

/// The host addresses `mach_vm_deallocate` was called with since `mark`, in call order.
fn freed_addrs(p: &FakePlatform, mark: u64) -> Vec<usize> {
    p.call_log()
        .since(mark)
        .into_iter()
        .filter(|c| c.sel == "mach_vm_deallocate")
        .map(|c| match c.args.first() {
            Some(Arg::Ptr(a)) => *a,
            other => panic!("unexpected deallocate argument {other:?}"),
        })
        .collect()
}
