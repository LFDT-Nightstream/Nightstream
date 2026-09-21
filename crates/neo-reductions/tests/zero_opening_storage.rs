use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use neo_math::{D, F, K};
use neo_reductions::superneo_eval::{SuperneoEvalCacheBuilder, SuperneoZBlocks};
use p3_field::PrimeCharacteristicRing;

struct AllocationProbe;
static RECORD: AtomicBool = AtomicBool::new(false);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

fn record(bytes: usize) {
    if RECORD.load(Ordering::Relaxed) {
        LARGEST.fetch_max(bytes, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for AllocationProbe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: AllocationProbe = AllocationProbe;

fn zero_opening_allocation(blocks: usize) -> usize {
    let width = blocks * D;
    let mut builder = SuperneoEvalCacheBuilder::new(1, width, 1).unwrap();
    builder.push_row(0, 0, [(width - 1, F::ONE)]).unwrap();
    let cache = builder.finish().unwrap();
    let zero = SuperneoZBlocks::with_block_len(blocks);
    let point = vec![K::from(F::from_u64(7)); width.next_power_of_two().ilog2() as usize];
    LARGEST.store(0, Ordering::Relaxed);
    RECORD.store(true, Ordering::Relaxed);
    let result = std::hint::black_box(&cache)
        .eval_real_v1_1_openings(
            std::hint::black_box(&point),
            std::slice::from_ref(std::hint::black_box(&zero)),
        )
        .unwrap();
    RECORD.store(false, Ordering::Relaxed);
    let allocated = LARGEST.load(Ordering::Relaxed);
    assert_eq!(result[0].eval_k, vec![K::ZERO; D]);
    assert_eq!(result[0].eval_a, vec![vec![K::ZERO; D]]);
    assert!(cache
        .eval_real_v1_1_openings(&point[1..], std::slice::from_ref(&zero))
        .is_err());
    assert!(cache
        .eval_real_v1_1_openings(&point, &[SuperneoZBlocks::with_block_len(blocks + 1)])
        .is_err());
    allocated
}

#[test]
fn zero_opening_workspace_does_not_grow_with_carrier_width() {
    // Output dimensions are fixed. A zero input needs no carrier-sized ring
    // scratch, even though the matrix contains a nonzero last-column entry.
    let small = zero_opening_allocation(1);
    // Make a dense scratch plane exceed the measured fixed output allocation.
    let blocks = small.div_ceil(std::mem::size_of::<neo_math::Rq>()) + 1;
    let wider = zero_opening_allocation(blocks);
    assert_eq!(small, wider, "zero opening allocation depends on carrier width");
}
