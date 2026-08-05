//! Retained red-team regressions for verifier resource bounds.

#[path = "../support/mod.rs"]
mod support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use neo_fold_clean::paper::construction2::ProofState;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static MAX_TRACKED_ALLOCATION: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;

impl TrackingAllocator {
    fn record(size: usize) {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            MAX_TRACKED_ALLOCATION.fetch_max(size, Ordering::Relaxed);
        }
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

/// Commitment dimensions are proof-controlled metadata. The verifier should
/// compare them with the Ajtai setup before allocating an all-zero commitment
/// of that shape. The zero-witness fast path currently allocates `d * kappa`
/// field elements first, allowing a tiny malformed proof to request unbounded
/// memory before its inevitable shape rejection.
#[test]
fn final_witness_authority_rejects_commitment_shape_before_attacker_sized_allocation() {
    const ATTACKER_D: usize = 1 << 14;
    const MAX_REASONABLE_REJECTION_ALLOCATION: usize = 256 * 1024;

    let prep = support::toy_preprocessing();
    let proof = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 0)]]).expect("one-step proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, proof).expect("finalized proof");
    let ProofState::Active { running, .. } = &finished.state.proof else {
        panic!("finalized proof must be active");
    };
    let mut running = running.materialize().expect("materialized final running");
    assert!(
        running.witnesses[0]
            .as_slice()
            .iter()
            .all(|&entry| entry == F::ZERO),
        "fixture must exercise the zero-witness fast path"
    );

    running.claims[0].c.d = ATTACKER_D;
    MAX_TRACKED_ALLOCATION.store(0, Ordering::Relaxed);
    TRACK_ALLOCATIONS.store(true, Ordering::Release);
    let result = neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running);
    TRACK_ALLOCATIONS.store(false, Ordering::Release);

    assert!(result.is_err(), "malformed commitment dimensions must reject");
    let largest = MAX_TRACKED_ALLOCATION.load(Ordering::Relaxed);
    assert!(
        largest <= MAX_REASONABLE_REJECTION_ALLOCATION,
        "verifier resource-exhaustion failure: malformed d={ATTACKER_D} caused a {largest}-byte allocation before the shape check"
    );
}
