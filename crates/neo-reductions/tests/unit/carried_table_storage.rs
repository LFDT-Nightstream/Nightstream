use super::*;
use crate::superneo_eval::{weighted_identity_projection, CachedMatrixRows, MatrixWindow, SuperneoEvalCacheBuilder};
use neo_math::KExtensions;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

struct AllocationProbe;
thread_local! {
    static CARRIER_BYTES: Cell<usize> = const { Cell::new(0) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static REAL_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record(bytes: usize) {
    if CARRIER_BYTES.get() != 0 && bytes == CARRIER_BYTES.get() {
        ALLOCATIONS.set(ALLOCATIONS.get() + 1);
    } else if CARRIER_BYTES.get() != 0 && bytes == CARRIER_BYTES.get() / 2 {
        REAL_ALLOCATIONS.set(REAL_ALLOCATIONS.get() + 1);
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

#[test]
fn carried_table_has_no_flat_combined_witness() {
    let rows = 2;
    let logical = D + 1;
    let width = 2 * D;
    let mut builder = SuperneoEvalCacheBuilder::new(rows, logical, 2).unwrap();
    for row in 0..rows {
        builder
            .push_row(0, row, [(row, F::ONE), (D, -F::ONE)])
            .unwrap();
        builder
            .push_row(1, row, [(row + 1, F::from_u64(3))])
            .unwrap();
    }
    let cache = builder.finish().unwrap();
    let source = CachedMatrixRows::new(&cache).unwrap();
    let workspace_bytes = MatrixWindow::required_workspace(&source, 0..rows, 0).unwrap();
    let packed = Mat::compact_signed_unit_from_column_masks(D, 2, &[1, 2], &[4, 8]).unwrap();
    let mut dense = Mat::zero(D, 2, F::ZERO);
    dense[(D - 1, 1)] = -F::ONE;
    dense[(1, 0)] = F::ONE;
    let zero = Mat::virtual_constant(D, 2, F::ZERO);
    let witnesses = [zero.clone(), packed, zero, dense];
    let running = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, logical).unwrap())
        .collect::<Vec<_>>();
    let dims = JointDims {
        assignment_width: width,
        row_count: width.next_power_of_two(),
        variables: width.next_power_of_two().ilog2() as usize,
        matrix_count: 2,
        degree: 4,
    };
    let gamma = K::from_coeffs([F::from_u64(5), F::from_u64(7)]);
    let combined = (0..width)
        .map(|index| {
            witnesses
                .iter()
                .enumerate()
                .map(|(source, witness)| gamma_power(gamma, source) * K::from(witness[(index % D, index / D)]))
                .sum::<K>()
        })
        .collect::<Vec<_>>();
    let blocks = SuperneoZBlocks::from_z(&combined);
    let matrix_weights = std::array::from_fn(|lane| gamma_power(gamma, witnesses.len() * dims.matrix_count * lane));
    let matrix_coefficients = (0..dims.matrix_count)
        .map(|matrix| gamma_power(gamma, witnesses.len() * matrix))
        .collect::<Vec<_>>();
    // One worker keeps every counted allocation on the probe's thread.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        // Warm the shared projection bases and compute the old scalar reference.
        let matrix = cache.eval_weighted_row_table(&blocks, &matrix_weights, &matrix_coefficients, rows, rows);
        let pad_weights = std::array::from_fn(|lane| gamma_power(gamma, witnesses.len() * lane));
        let mut expected = weighted_identity_projection(&blocks, &pad_weights);
        let matrix_shift = gamma_power(gamma, witnesses.len() * D);
        for (out, value) in expected.iter_mut().zip(matrix) {
            *out += matrix_shift * value;
        }
        while expected.last().is_some_and(|value| *value == K::ZERO) {
            expected.pop();
        }
        ALLOCATIONS.set(0);
        REAL_ALLOCATIONS.set(0);
        CARRIER_BYTES.set(width * size_of::<K>());
        let actual = carried_table(&source, &running, gamma, dims, rows, workspace_bytes).unwrap();
        CARRIER_BYTES.set(0);
        assert_eq!(
            actual, expected,
            "complete carried table and zero-source gamma positions"
        );
        assert_eq!(
            (ALLOCATIONS.get(), REAL_ALLOCATIONS.get()),
            (1, 0),
            "only one reused projection may span the carrier; combined witness planes must remain local"
        );
        let zeros = [SuperneoZBlocks::with_block_len(width / D)];
        let mut dirty = vec![K::ONE; width];
        fill_combined_projection(&zeros, &[gamma], &matrix_weights, &mut dirty);
        assert!(
            dirty.iter().all(|value| *value == K::ZERO),
            "reused zero blocks must be cleared"
        );
    });
}

#[test]
fn carried_rows_beyond_the_witness_carrier_are_retained() {
    let rows = D + 1;
    let mut builder = SuperneoEvalCacheBuilder::new(rows, 1, 1).unwrap();
    for row in 0..rows {
        builder.push_row(0, row, [(0, F::ONE)]).unwrap();
    }
    let cache = builder.finish().unwrap();
    let matrix_rows = CachedMatrixRows::new(&cache).unwrap();
    let workspace_bytes = MatrixWindow::required_workspace(&matrix_rows, 0..rows, 0).unwrap();
    let witness = Mat::<F>::compact_signed_unit_from_column_masks(D, 1, &[1], &[0]).unwrap();
    let source = SuperneoZBlocks::from_witness_mat(&witness, 1).unwrap();
    let gamma = K::from_coeffs([F::from_u64(5), F::from_u64(7)]);
    let weights = std::array::from_fn(|lane| gamma_power(gamma, lane));
    let projection = weighted_identity_projection(&source, &weights);
    let mut expected = projection.clone();
    expected.resize(rows, K::ZERO);
    for value in &mut expected {
        *value += gamma_power(gamma, D) * projection[0];
    }
    let dims = JointDims {
        assignment_width: D,
        row_count: rows.next_power_of_two(),
        variables: rows.next_power_of_two().ilog2() as usize,
        matrix_count: 1,
        degree: 4,
    };
    assert_eq!(
        carried_table(&matrix_rows, &[source], gamma, dims, rows, workspace_bytes).unwrap(),
        expected
    );
}

#[test]
fn dense_prefix_folding_reuses_its_storage() {
    let challenge = K::from_coeffs([F::from_u64(5), F::from_u64(7)]);
    let values = (0..2 * D)
        .map(|index| K::from_coeffs([F::from_usize(index + 1), F::from_usize(2 * index + 3)]))
        .collect::<Vec<_>>();
    let expected = values
        .chunks(2)
        .map(|pair| prefix::interpolate(pair[0], pair[1], challenge))
        .collect::<Vec<_>>();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let mut actual = values.clone();
        ALLOCATIONS.set(0);
        REAL_ALLOCATIONS.set(0);
        CARRIER_BYTES.set(actual.len() * size_of::<K>());
        prefix::fold(&mut actual, challenge);
        CARRIER_BYTES.set(0);
        assert_eq!(actual, expected);
        assert_eq!(
            (ALLOCATIONS.get(), REAL_ALLOCATIONS.get()),
            (0, 0),
            "folding must not allocate another whole or half table"
        );
    });

    // Two workers are the minimum needed to exercise chunk compaction.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        for len in [0, 1, 2, 3, D - 1, D, D + 1, 2 * D, 2 * D + 1] {
            let mut actual = (0..len)
                .map(|index| K::from_coeffs([F::from_usize(index + 1), F::from_usize(2 * index + 3)]))
                .collect::<Vec<_>>();
            let expected = actual
                .chunks(2)
                .map(|pair| prefix::interpolate(pair[0], pair.get(1).copied().unwrap_or(K::ZERO), challenge))
                .collect::<Vec<_>>();
            prefix::fold(&mut actual, challenge);
            assert_eq!(actual, expected, "prefix length {len}");
        }
    });
}

#[test]
fn openings_reuse_the_consumed_carrier_buffer() {
    let logical = 2 * D + 1;
    let width = 3 * D;
    let mut builder = SuperneoEvalCacheBuilder::new(2, logical, 2).unwrap();
    for row in 0..2 {
        builder
            .push_row(0, row, [(row, F::from_u64(3)), (2 * D, -F::ONE)])
            .unwrap();
        builder
            .push_row_with_runs(
                1,
                row,
                [(row, F::ONE)],
                [neo_ccs::GeometricRowRun::new(
                    row,
                    D - 2,
                    D + 3,
                    F::from_u64(7),
                    F::from_u64(3),
                )],
            )
            .unwrap();
    }
    let cache = builder.finish().unwrap();
    let witness = Mat::<F>::compact_signed_unit_from_column_masks(D, 3, &[1, 0, 1 << (D - 1)], &[4, 0, 0]).unwrap();
    let source = SuperneoZBlocks::from_witness_mat(&witness, logical).unwrap();
    let sources = [source, SuperneoZBlocks::with_block_len(3)];
    let point = vec![K::from_coeffs([F::from_u64(5), F::from_u64(7)]); width.next_power_of_two().ilog2() as usize];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let expected = cache.eval_real_v1_1_openings(&point, &sources).unwrap();
        let mut storage = vec![K::ONE; width];
        storage.truncate(1); // Completed in-place SumCheck retains this capacity.
        ALLOCATIONS.set(0);
        REAL_ALLOCATIONS.set(0);
        CARRIER_BYTES.set(width * size_of::<K>());
        let actual = cache
            .eval_real_v1_1_openings_reusing(&point, &sources, storage)
            .unwrap();
        CARRIER_BYTES.set(0);
        assert_eq!(actual, expected, "dirty reused storage changed complete openings");
        assert_eq!(
            (ALLOCATIONS.get(), REAL_ALLOCATIONS.get()),
            (0, 0),
            "openings allocated another carrier-sized coefficient buffer"
        );
    });
}

#[test]
fn encoded_prefix_folding_reuses_its_codes() {
    let witness = Mat::compact_signed_unit_from_column_masks(D, 2, &[1, 1 << (D - 1)], &[4, 2]).unwrap();
    let challenge = K::from_coeffs([F::from_u64(5), F::from_u64(7)]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let mut actual = Assignment::new(&witness, 2 * D);
        actual.fold(challenge);
        let expected = (0..actual.len().div_ceil(2))
            .map(|index| {
                let (low, high) = actual.pair(index);
                prefix::interpolate(low, high, challenge)
            })
            .collect::<Vec<_>>();
        ALLOCATIONS.set(0);
        REAL_ALLOCATIONS.set(0);
        CARRIER_BYTES.set(actual.len() * size_of::<u16>());
        actual.fold(challenge);
        CARRIER_BYTES.set(0);
        assert!(matches!(actual, Assignment::Encoded { .. }));
        for (index, expected) in expected.into_iter().enumerate() {
            assert_eq!(actual.get(index), expected);
        }
        assert_eq!(
            (ALLOCATIONS.get(), REAL_ALLOCATIONS.get()),
            (0, 0),
            "encoded folding must not allocate a replacement code buffer"
        );
    });
}
