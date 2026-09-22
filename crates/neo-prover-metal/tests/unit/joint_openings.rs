use super::*;
use neo_ccs::{GeometricRowRun, Mat};
use neo_reductions::superneo_eval::{
    CachedMatrixRows, MatrixRows, MatrixWindow, SuperneoEvalCacheBuilder, SuperneoZBlocks,
};
use objc2_metal::MTLBuffer;
use std::{ops::Range, sync::Arc};

pub(crate) fn matrix_workspace(source: &dyn MatrixRows, rows: Range<usize>) -> usize {
    let host = MatrixWindow::required_workspace(source, rows.clone(), 0).unwrap();
    let window = MatrixWindow::load_next(source, rows, host).unwrap();
    let (upload, staging) = super::super::matrix_window::upload_size(window.cache().matrix_caches()).unwrap();
    let descriptors = source.shape().matrices * size_of::<MetalCompactMatrix>();
    let uploaded = window.storage_bytes() + upload + staging + descriptors;
    let opening = OpeningLayout::measure(window.cache().matrix_caches(), source.shape().columns / D, true).unwrap();
    window.workspace_peak_bytes().max(uploaded) + opening.metadata_bytes
}

#[test]
fn matrix_window_retries_bound_rebuilt_rows_with_fixed_overhead() {
    let initial_rows = FORM_TILE_ENTRIES + 1;
    let row_bytes = size_of::<K>();
    let fixed_bytes = initial_rows * initial_rows * row_bytes;
    let available = fixed_bytes + row_bytes;
    let mut rows = initial_rows;
    let mut rebuilt_rows = 0;
    loop {
        rebuilt_rows += rows;
        assert!(
            rebuilt_rows < 2 * initial_rows,
            "fixed metadata must not cause repeated scans of nearly the same row range"
        );
        let required = fixed_bytes + rows * row_bytes;
        if required <= available {
            break;
        }
        rows = smaller_window(rows, required, available).unwrap();
    }
    assert_eq!(rows, 1);
    assert!(matches!(
        smaller_window(1, available + row_bytes, available),
        Err(MetalError::MemoryLimit { .. })
    ));
}

#[test]
fn compact_openings_match_cpu_across_parallel_and_tiled_lists() {
    // Cross the actual device list thresholds. All matrices share columns,
    // so each matrix must overwrite the reused form storage correctly.
    let rows = FORM_TILE_ENTRIES + 1;
    let mut builder = SuperneoEvalCacheBuilder::new(rows, 2 * D, 3).unwrap();
    for row in 0..rows {
        builder
            .push_row(0, row, [(1, F::from_u64(3)), (2, F::from_u64(5)), (D + 1, -F::ONE)])
            .unwrap();
        builder
            .push_row(
                1,
                row,
                (row < PARALLEL_FORM_LIST_THRESHOLD)
                    .then_some([(1, -F::ONE), (D + 1, F::ONE)])
                    .into_iter()
                    .flatten(),
            )
            .unwrap();
        builder
            .push_row(
                2,
                row,
                (0..D)
                    .filter(|_| row < 2)
                    .map(|local| (D + local, F::from_u64((local + 2) as u64))),
            )
            .unwrap();
    }
    let cache = Arc::new(builder.finish().unwrap());
    let mut first = Mat::zero(D, 2, F::ZERO);
    first[(1, 0)] = F::ONE;
    first[(2, 0)] = -F::ONE;
    first[(1, 1)] = F::ONE;
    let mut last = first.clone();
    last[(D - 1, 1)] = -F::ONE;
    let witnesses = [first, Mat::zero(D, 2, F::ZERO), last];
    let variables = rows.next_power_of_two().ilog2() as usize;
    let point: Vec<_> = (0..variables)
        .map(|i| K::from_coeffs([F::from_u64((i + 2) as u64), F::from_u64((i + 3) as u64)]))
        .collect();
    let blocks: Vec<_> = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, 2 * D).unwrap())
        .collect();
    let expected = cache.eval_real_v1_1_openings(&point, &blocks).unwrap();
    let session = MetalSession::new().unwrap();
    let source = CachedMatrixRows::new(&cache).unwrap();
    let workspace = matrix_workspace(&source, 0..rows);
    let before = session.activity().allocated_bytes;
    let plan = session
        .prepare_joint_matrix_plan(&source, workspace)
        .unwrap();
    assert_eq!(
        session.activity().allocated_bytes,
        before,
        "a plan must not upload matrix rows"
    );
    let empty = session.buffer(size_of::<u64>()).unwrap();
    let device_matrices = cache
        .matrix_caches()
        .iter()
        .map(|matrix| session.upload_matrix_metadata(matrix, &empty).unwrap())
        .collect::<Vec<_>>();
    let layout = OpeningLayout::measure(cache.matrix_caches(), 2, true).unwrap();
    let opening = session
        .prepare_joint_opening_plan(cache.matrix_caches(), &device_matrices, 2 * D, 0, true, &layout)
        .unwrap();
    assert!(opening.parallel_form_list_count > 0);
    assert!(opening.tiled_form_tile_count > 0);
    let masks = session
        .prepare_joint_witness_masks(&blocks, 2, 2 * D)
        .unwrap();
    let direct = session
        .eval_joint_openings(&opening, &masks, &point, witnesses.len(), 2 * D)
        .unwrap();
    for (actual, expected) in direct.iter().zip(&expected) {
        assert_eq!(actual.eval_k, expected.eval_k);
        assert_eq!(actual.eval_a, expected.eval_a);
    }
    drop(opening);
    drop(device_matrices);
    let actual = session
        .eval_joint_dec_openings(&plan, &witnesses, &point, 2 * D)
        .unwrap()
        .unwrap();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.eval_k, expected.eval_k);
        assert_eq!(actual.eval_a, expected.eval_a);
    }
}

#[test]
fn geometric_openings_share_block_coordinates_and_dispatch_each_matrix_once() {
    let rows = 3;
    let columns = 3 * D + 1;
    let mut compact = SuperneoEvalCacheBuilder::new(rows, columns, 3).unwrap();
    let mut scalar = SuperneoEvalCacheBuilder::new(rows, columns, 3).unwrap();
    for row in 0..rows {
        for matrix in 0..3 {
            let runs = match (matrix, row) {
                (0, 0) => vec![
                    GeometricRowRun::new(row, 0, columns, F::from_u64(19), F::from_u64(2)),
                    GeometricRowRun::new(row, D - 2, 41, F::from_u64(7), F::from_u64(3)),
                ],
                (0, 1) => vec![GeometricRowRun::new(row, 3 * D, 1, F::from_u64(11), F::ONE)],
                (2, 0) => vec![GeometricRowRun::new(row, D + 2, D + 5, F::from_u64(13), -F::ONE)],
                (2, 2) => vec![GeometricRowRun::new(row, 0, 41, F::from_u64(17), F::ZERO)],
                _ => vec![],
            };
            let entries = if matrix == 1 || (matrix == 0 && row == 1) {
                vec![(0, -F::ONE)]
            } else {
                vec![]
            };
            let mut coefficients = vec![F::ZERO; columns];
            for &(column, value) in &entries {
                coefficients[column] += value;
            }
            for run in &runs {
                run.for_each_term(|_, column, value| coefficients[column] += value);
            }
            compact
                .push_row_with_runs(matrix, row, entries, runs)
                .unwrap();
            scalar
                .push_row(
                    matrix,
                    row,
                    coefficients
                        .into_iter()
                        .enumerate()
                        .filter(|(_, value)| *value != F::ZERO),
                )
                .unwrap();
        }
    }
    let compact = Arc::new(compact.finish().unwrap());
    let scalar = scalar.finish().unwrap();
    let mut first = Mat::zero(D, columns.div_ceil(D), F::ZERO);
    for column in 0..columns {
        first[(column % D, column / D)] = if column % 2 == 0 { F::ONE } else { -F::ONE };
    }
    let mut last = first.clone();
    last[(D - 1, columns / D)] = -F::ONE;
    let witnesses = [first, Mat::zero(D, columns.div_ceil(D), F::ZERO), last];
    let point: Vec<_> = (0..(columns.div_ceil(D) * D).next_power_of_two().ilog2())
        .map(|i| K::from_coeffs([F::from_u64(u64::from(i) + 2), F::from_u64(u64::from(i) + 5)]))
        .collect();
    let blocks: Vec<_> = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, columns).unwrap())
        .collect();
    let expected = scalar.eval_real_v1_1_openings(&point, &blocks).unwrap();
    let session = MetalSession::new().unwrap();
    let source = CachedMatrixRows::new(&compact).unwrap();
    let workspace = matrix_workspace(&source, 0..rows);
    let plan = session
        .prepare_joint_matrix_plan(&source, workspace)
        .unwrap();
    let empty = session.buffer(size_of::<u64>()).unwrap();
    let device_matrices = compact
        .matrix_caches()
        .iter()
        .map(|matrix| session.upload_matrix_metadata(matrix, &empty).unwrap())
        .collect::<Vec<_>>();
    let layout = OpeningLayout::measure(compact.matrix_caches(), columns.div_ceil(D), true).unwrap();
    let opening = session
        .prepare_joint_opening_plan(
            compact.matrix_caches(),
            &device_matrices,
            columns.div_ceil(D) * D,
            0,
            true,
            &layout,
        )
        .unwrap();
    assert_eq!(opening.geometric.len(), 2);
    for geometric in &opening.geometric {
        assert_eq!(
            geometric.groups.length(),
            (geometric.group_count + 1) * size_of::<[u32; 2]>()
        );
    }
    let masks = session
        .prepare_joint_witness_masks(&blocks, 2, columns)
        .unwrap();
    let before = session.activity().dispatches;
    let actual = session
        .eval_joint_openings(&opening, &masks, &point, witnesses.len(), columns)
        .unwrap();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.eval_k, expected.eval_k);
        assert_eq!(actual.eval_a, expected.eval_a);
    }
    // One tensor stage per point coordinate, one row-weight dispatch, six
    // common opening stages per live matrix, and one dispatch per run plan.
    let expected_dispatches = point.len() + 1 + 6 * opening.matrix_count + opening.geometric.len();
    assert_eq!(session.activity().dispatches - before, expected_dispatches as u64);
    drop(opening);
    drop(device_matrices);
    let streamed = session
        .eval_joint_dec_openings(&plan, &witnesses, &point, columns)
        .unwrap()
        .unwrap();
    let row_workspace = (0..rows)
        .map(|row| matrix_workspace(&source, row..row + 1))
        .max()
        .unwrap();
    assert!(row_workspace < workspace);
    let plan = session
        .prepare_joint_matrix_plan(&source, row_workspace)
        .unwrap();
    let windowed = session
        .eval_joint_dec_openings(&plan, &witnesses, &point, columns)
        .unwrap()
        .unwrap();
    for ((streamed, windowed), expected) in streamed.iter().zip(&windowed).zip(&expected) {
        assert_eq!(streamed.eval_k, expected.eval_k);
        assert_eq!(streamed.eval_a, expected.eval_a);
        assert_eq!(windowed.eval_k, expected.eval_k);
        assert_eq!(windowed.eval_a, expected.eval_a);
    }
}
