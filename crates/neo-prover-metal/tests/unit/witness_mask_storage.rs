use super::*;
use neo_ccs::Mat;
use neo_math::F;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

struct AllocationProbe;
thread_local! {
    static RECORD: Cell<bool> = const { Cell::new(false) };
    static ALLOCATED: Cell<usize> = const { Cell::new(0) };
}

fn record(bytes: usize) {
    if RECORD.get() {
        ALLOCATED.set(ALLOCATED.get() + bytes);
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

fn measured_masks(session: &MetalSession, sources: &[SuperneoZBlocks]) -> (MetalWitnessMasks, usize) {
    ALLOCATED.set(0);
    RECORD.set(true);
    let result = session.prepare_joint_witness_masks(sources, 2, D + 1);
    RECORD.set(false);
    (result.unwrap(), ALLOCATED.get())
}

#[test]
fn zero_suffix_does_not_allocate_host_mask_words() {
    let session = MetalSession::new().unwrap();
    let matrix = Mat::<F>::compact_signed_unit_from_column_masks(D, 2, &[1, 2], &[4, 8]).unwrap();
    let active = SuperneoZBlocks::from_witness_mat(&matrix, D + 1).unwrap();
    let count = neo_params::NeoParams::nightstream_goldilocks_k16().k_rho as usize + 1;
    let mut sources = vec![SuperneoZBlocks::with_block_len(2); count];
    sources[0] = active;
    // Initialize the driver before comparing Rust host allocations.
    drop(measured_masks(&session, &sources[..1]));
    let (single, single_bytes) = measured_masks(&session, &sources[..1]);
    let (family, family_bytes) = measured_masks(&session, &sources);
    assert_eq!(family.active_witnesses(), &[0]);
    assert!(family.matches_joint(count, 2));
    assert_eq!(family.words().length(), single.words().length());
    assert_eq!(session.read_buffer::<u64>(family.words(), 4), vec![1, 4, 2, 8]);
    assert_eq!(family_bytes, single_bytes, "zero suffix allocated host mask words");
}
