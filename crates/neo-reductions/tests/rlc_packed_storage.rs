use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use neo_ccs::Mat;
use neo_math::{D, F};
use neo_reductions::optimized_engine::rlc_mix_witnesses;
use p3_field::PrimeCharacteristicRing;

struct AllocationProbe;
static RECORD: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

fn record(bytes: usize) {
    if RECORD.load(Ordering::Relaxed) {
        ALLOCATED.fetch_add(bytes, Ordering::Relaxed);
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

fn measured_mix(witness: &Mat<F>, rhos: &[Mat<F>]) -> (Mat<F>, usize) {
    ALLOCATED.store(0, Ordering::Relaxed);
    RECORD.store(true, Ordering::Relaxed);
    let result = rlc_mix_witnesses(
        D * witness.cols(),
        std::hint::black_box(rhos),
        &[std::hint::black_box(witness)],
    );
    RECORD.store(false, Ordering::Relaxed);
    (result, ALLOCATED.load(Ordering::Relaxed))
}

#[test]
fn packed_rlc_needs_no_nonzero_entry_lists() {
    // One worker isolates scratch allocation from Rayon scheduling and other tests.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let columns = D + 1;
    let all_lanes = (1u64 << D) - 1;
    let positive: Vec<_> = (0..columns)
        .map(|column| if column % 2 == 0 { all_lanes } else { 0 })
        .collect();
    let negative: Vec<_> = positive.iter().map(|&mask| all_lanes ^ mask).collect();
    let packed = Mat::compact_signed_unit_from_column_masks(D, columns, &positive, &negative).unwrap();
    let zero = Mat::virtual_constant(D, columns, F::ZERO);
    let rhos = [Mat::identity(D)];
    pool.install(|| {
        let (zero_result, zero_bytes) = measured_mix(&zero, &rhos);
        assert!(zero_result.as_slice().iter().all(|&value| value == F::ZERO));
        let (actual, nonzero_bytes) = measured_mix(&packed, &rhos);
        assert_eq!(actual, packed);
        assert_eq!(nonzero_bytes, zero_bytes, "packed RLC allocates data beyond its output");
    });
    // Mixed signs, zero lanes, non-diagonal challenges and both packed layouts.
    let positive: Vec<_> = (0..columns).map(|column| 1u64 << (column % D)).collect();
    let negative: Vec<_> = (0..columns)
        .map(|column| 1u64 << ((column + D / 2) % D))
        .collect();
    let first = Mat::compact_signed_unit_from_column_masks(D, columns, &positive, &negative).unwrap();
    let second = Mat::compact_signed_unit(D, columns, packed.to_dense_vec());
    let witnesses = [&first, &zero, &second];
    let rhos: Vec<_> = (0..witnesses.len())
        .map(|source| {
            Mat::from_row_major(
                D,
                D,
                (0..D * D)
                    .map(|index| F::from_u64((index + source * D) as u64))
                    .collect(),
            )
        })
        .collect();
    let mut expected = Mat::zero(D, columns, F::ZERO);
    for (rho, witness) in rhos.iter().zip(witnesses) {
        for row in 0..D {
            for column in 0..columns {
                for lane in 0..D {
                    expected[(row, column)] += rho[(row, lane)] * witness[(lane, column)];
                }
            }
        }
    }
    assert_eq!(rlc_mix_witnesses(D * columns, &rhos, &witnesses), expected);
    pool.install(|| assert_eq!(rlc_mix_witnesses(D * columns, &rhos, &witnesses), expected));
}
