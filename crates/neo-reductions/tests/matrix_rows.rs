use std::{
    mem::size_of,
    ops::{ControlFlow, Range},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

use neo_ccs::{poly::Term, CcsMatrix, CcsStructure, CscMat, GeometricRowRun, Mat, SeededPhi81LinearBlock, SparsePoly};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::{
    superneo_eval::{
        build_superneo_eval_cache, check_ccs_relation_zero_cached_with_blocks, eval_real_v1_1_openings_from_rows,
        first_unsatisfied_row_from_rows, CachedMatrixRows, MatrixRowSink, MatrixRows, MatrixShape, MatrixWindow,
        SuperneoCachedRelationError, SuperneoCompactRowOffsets, SuperneoEvalCache, SuperneoZBlocks,
    },
    PiCcsError,
};
use p3_field::PrimeCharacteristicRing;

struct PatternRows {
    rows: usize,
    ignore_stop: bool,
    visits: Mutex<Vec<Range<usize>>>,
}

impl PatternRows {
    fn new(rows: usize) -> Self {
        Self {
            rows,
            ignore_stop: false,
            visits: Mutex::new(Vec::new()),
        }
    }
}

impl MatrixRows for PatternRows {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            columns: 2 * D,
            matrices: 3,
        }
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.visits.lock().unwrap().push(rows.clone());
        for row in rows {
            // Intentionally put the scalar before an overlapping run whose
            // column starts earlier. Only row/matrix order is constrained.
            sink.push_run(row, 0, GeometricRowRun::new(row, D, 1, -F::from_usize(row + 1), F::ONE))?;
            sink.push_run(
                row,
                0,
                GeometricRowRun::new(row, D - 2, 5, F::from_usize(row + 2), F::from_u64(3)),
            )?;
            if sink.finish_matrix_row(row, 0)?.is_break() && !self.ignore_stop {
                return Ok(());
            }
            // Matrix one has an explicit empty slot in every row.
            if sink.finish_matrix_row(row, 1)?.is_break() && !self.ignore_stop {
                return Ok(());
            }
            sink.push_run(row, 2, GeometricRowRun::new(row, row + 2, 1, F::ONE, F::ONE))?;
            if sink.finish_matrix_row(row, 2)?.is_break() && !self.ignore_stop {
                return Ok(());
            }
        }
        Ok(())
    }
}

struct DenseRows(Vec<Mat<F>>);

impl MatrixRowSink for DenseRows {
    fn push_run(&mut self, row: usize, matrix: usize, run: GeometricRowRun<F>) -> Result<(), PiCcsError> {
        assert_eq!(row, run.row());
        run.for_each_term(|row, column, coefficient| self.0[matrix][(row, column)] += coefficient);
        Ok(())
    }

    fn finish_matrix_row(&mut self, _: usize, _: usize) -> Result<ControlFlow<()>, PiCcsError> {
        Ok(ControlFlow::Continue(()))
    }
}

fn expanded_cache(source: &dyn MatrixRows) -> SuperneoEvalCache {
    let shape = source.shape();
    let mut dense = DenseRows(vec![Mat::zero(shape.rows, shape.columns, F::ZERO); shape.matrices]);
    source.visit_rows(0..shape.rows, &mut dense).unwrap();
    build_superneo_eval_cache(&CcsStructure::new(dense.0, SparsePoly::new(shape.matrices, vec![])).unwrap()).unwrap()
}

fn compare_rows(window: &MatrixWindow, expected: &SuperneoEvalCache, columns: usize) {
    let values = (0..columns)
        .map(|column| K::from_coeffs([F::from_usize(column + 1), F::from_usize(2 * column + 1)]))
        .collect::<Vec<_>>();
    let witness = SuperneoZBlocks::from_z(&values);
    let rows = window.rows();
    assert_eq!(
        window.cache().relation_shape(),
        Some((rows.len(), columns, expected.matrix_caches().len()))
    );
    for (local_row, global_row) in rows.enumerate() {
        for matrix in 0..expected.matrix_caches().len() {
            assert_eq!(
                window
                    .cache()
                    .matrix(matrix)
                    .unwrap()
                    .row_dot_ring_with_blocks(local_row, &witness),
                expected
                    .matrix(matrix)
                    .unwrap()
                    .row_dot_ring_with_blocks(global_row, &witness),
                "matrix {matrix}, global row {global_row}",
            );
        }
    }
}

#[test]
fn original_runs_preserve_global_columns_overlap_and_empty_slots_across_windows() {
    let source = PatternRows::new(5);
    let expected = expanded_cache(&source);
    // Later scalar coefficients use dense patterns, unlike row zero's -1.
    // Derive the workspace from two of those complete rows. usize::MAX here
    // removes a budget ceiling for this small, explicitly bounded request.
    let budget = MatrixWindow::load_next(&source, 2..4, usize::MAX)
        .unwrap()
        .workspace_peak_bytes();
    source.visits.lock().unwrap().clear();
    let mut start = 0;
    let mut ranges = Vec::new();
    while start < source.shape().rows {
        let window = MatrixWindow::load_next(&source, start..source.shape().rows, budget).unwrap();
        compare_rows(&window, &expected, source.shape().columns);
        assert!(window.workspace_peak_bytes() <= budget);
        assert!(window.storage_bytes() < window.workspace_peak_bytes());
        start = window.rows().end;
        ranges.push(window.rows());
    }
    assert_eq!(ranges, vec![0..2, 2..4, 4..5]);
    assert_eq!(*source.visits.lock().unwrap(), vec![0..5, 0..2, 2..5, 2..4, 4..5, 4..5]);
}

#[test]
fn scalar_runs_use_compact_entries_and_keep_geometric_runs_unexpanded() {
    let source = PatternRows::new(2);
    let required = MatrixWindow::required_workspace(&source, 0..2, 0).unwrap();
    let window = MatrixWindow::load_next(&source, 0..2, required).unwrap();
    assert_eq!(window.workspace_peak_bytes(), required);
    let mixed = window
        .cache()
        .matrix(0)
        .unwrap()
        .compact_device_parts()
        .unwrap();
    assert!(matches!(mixed.row_offsets, SuperneoCompactRowOffsets::U32([0, 1, 2])));
    assert!(matches!(
        mixed.geometric_row_offsets,
        SuperneoCompactRowOffsets::U32([0, 1, 2])
    ));
    // Existing device ABI: sign bit 30; dense-reference bit 31. The first
    // scalar is -1 at global column D, and the second has coefficient -2.
    assert_eq!(mixed.row_blocks, &[(1 << 30) | 1, 1 << 31]);
    assert_eq!(mixed.dense_row_blocks, &[[1, 0]]);
    assert_eq!(mixed.dense_offsets, &[0, 1]);
    assert_eq!(mixed.dense_locals, &[0]);
    assert_eq!(mixed.dense_coefficients, &[-F::from_u64(2)]);
    assert_eq!(mixed.geometric_runs.len(), 2);
    assert!(mixed.geometric_runs.iter().all(|run| run[0] >> 32 == 5));
    let empty = window
        .cache()
        .matrix(1)
        .unwrap()
        .compact_device_parts()
        .unwrap();
    assert!(matches!(empty.row_offsets, SuperneoCompactRowOffsets::Empty));
    assert!(matches!(empty.geometric_row_offsets, SuperneoCompactRowOffsets::Empty));
    let unit = window
        .cache()
        .matrix(2)
        .unwrap()
        .compact_device_parts()
        .unwrap();
    assert_eq!(unit.row_blocks, &[2 << 24, 3 << 24]);
    assert!(unit.dense_row_blocks.is_empty());
    assert!(unit.geometric_runs.is_empty());
    compare_rows(&window, &expanded_cache(&source), source.shape().columns);
}

#[test]
fn supplied_workspace_includes_construction_counts_and_reserved_row_values() {
    let source = PatternRows::new(3);
    let one = MatrixWindow::load_next(&source, 0..1, usize::MAX).unwrap();
    let two = MatrixWindow::load_next(&source, 0..2, usize::MAX).unwrap();
    let payload = source.shape().matrices * size_of::<K>();
    let reserved = MatrixWindow::load_next_with_payload(&source, 0..3, two.workspace_peak_bytes(), payload).unwrap();
    assert_eq!(reserved.rows(), 0..1);
    assert_eq!(reserved.storage_bytes(), one.storage_bytes());
    assert_eq!(reserved.workspace_peak_bytes(), one.workspace_peak_bytes() + payload);
    let required = reserved.workspace_peak_bytes();
    assert!(matches!(
        MatrixWindow::load_next_with_payload(&source, 0..1, required - 1, payload),
        Err(PiCcsError::MatrixWorkspace { required: actual, available })
            if actual == required && available == required - 1
    ));
    assert!(matches!(
        MatrixWindow::load_next(&source, 0..1, one.workspace_peak_bytes() - 1),
        Err(PiCcsError::MatrixWorkspace { required, .. }) if required == one.workspace_peak_bytes()
    ));
    assert!(matches!(
        MatrixWindow::load_next_with_payload(&source, 0..1, usize::MAX, usize::MAX),
        Err(PiCcsError::InvalidInput(_))
    ));
}

#[test]
fn required_workspace_counts_once_and_matches_the_full_requested_window() {
    let source = PatternRows::new(5);
    let payload = source.shape().matrices * size_of::<K>();
    let required = MatrixWindow::required_workspace(&source, 1..3, payload).unwrap();
    assert_eq!(*source.visits.lock().unwrap(), vec![1..3]);
    let window = MatrixWindow::load_next_with_payload(&source, 1..3, required, payload).unwrap();
    assert_eq!(window.rows(), 1..3);
    assert_eq!(window.workspace_peak_bytes(), required);
    assert!(matches!(
        MatrixWindow::required_workspace(&source, 1..3, usize::MAX),
        Err(PiCcsError::InvalidInput(_))
    ));
}

#[test]
fn workspace_prefixes_charge_offset_families_when_they_first_appear() {
    struct LateFamilies;

    impl MatrixRows for LateFamilies {
        fn shape(&self) -> MatrixShape {
            MatrixShape {
                rows: 11,
                columns: 2 * D,
                matrices: 3,
            }
        }

        fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
            for row in rows {
                for matrix in 0..self.shape().matrices {
                    let run = match (row, matrix) {
                        (4, 1) => Some((D + 1, 1, F::from_u64(7), F::ONE)),
                        (6, 0) => Some((D - 1, 2, F::ONE, F::from_u64(3))),
                        (6, 1) => Some((3, 3, F::from_u64(5), -F::ONE)),
                        (7, 1) => Some((D + 2, 1, F::from_u64(3), F::ONE)),
                        (7, 2) => Some((5, 1, -F::ONE, F::ONE)),
                        (9, 0) => Some((2, 1, F::ONE, F::ONE)),
                        _ => None,
                    };
                    if let Some((column, count, coefficient, ratio)) = run {
                        sink.push_run(
                            row,
                            matrix,
                            GeometricRowRun::new(row, column, count, coefficient, ratio),
                        )?;
                    }
                    if sink.finish_matrix_row(row, matrix)?.is_break() {
                        return Ok(());
                    }
                }
            }
            Ok(())
        }
    }

    let source = LateFamilies;
    let start = 2;
    let end = source.shape().rows;
    let payload = source.shape().matrices * size_of::<K>();
    // An empty allocation isolates the fixed cache descriptors, initial dense
    // offsets, and construction counts without using the prefix count result.
    let empty = MatrixWindow::load_next(&source, start..start + 1, usize::MAX).unwrap();
    let fixed = empty.storage_bytes();
    let construction = empty.workspace_peak_bytes() - fixed;
    // Public compact device layout: each scalar has a u32 reference; a
    // non-unit scalar also has [block, pattern], one offset/local/coefficient.
    let explicit_bytes = size_of::<u32>();
    let dense_bytes = size_of::<[u32; 2]>() + size_of::<u32>() + size_of::<u8>() + size_of::<F>();
    let geometric_bytes = size_of::<[u64; 3]>();
    // Independently enumerated cumulative (explicit, dense, geometric, offset
    // families) after each global row. Empty rows before and after first runs
    // must get offsets once that family becomes live.
    let prefixes = [
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (1, 1, 0, 1),
        (1, 1, 0, 1),
        (1, 1, 2, 3),
        (3, 2, 2, 4),
        (3, 2, 2, 4),
        (4, 2, 2, 5),
        (4, 2, 2, 5),
    ];
    let mut previous_required = None;
    for (index, (explicit, dense, geometric, families)) in prefixes.into_iter().enumerate() {
        let rows = index + 1;
        let prefix_end = start + rows;
        let storage = fixed
            + explicit * explicit_bytes
            + dense * dense_bytes
            + geometric * geometric_bytes
            + families * (rows + 1) * size_of::<u32>();
        let required = storage + construction + rows * payload;
        let complete = MatrixWindow::load_next_with_payload(&source, start..prefix_end, usize::MAX, payload).unwrap();
        assert_eq!(complete.storage_bytes(), storage, "prefix ending at {prefix_end}");
        assert_eq!(complete.workspace_peak_bytes(), required);
        assert_eq!(
            MatrixWindow::required_workspace(&source, start..prefix_end, payload).unwrap(),
            required,
        );
        let exact = MatrixWindow::load_next_with_payload(&source, start..end, required, payload).unwrap();
        assert_eq!(exact.rows(), start..prefix_end);
        assert_eq!(exact.workspace_peak_bytes(), required);
        if rows == 1 {
            assert!(matches!(
                MatrixWindow::load_next_with_payload(&source, start..end, required - 1, payload),
                Err(PiCcsError::MatrixWorkspace { required: actual, available })
                    if actual == required && available == required - 1
            ));
        } else {
            let below = MatrixWindow::load_next_with_payload(&source, start..end, required - 1, payload).unwrap();
            assert_eq!(below.rows(), start..prefix_end - 1);
            assert_eq!(below.workspace_peak_bytes(), previous_required.unwrap());
        }
        previous_required = Some(required);
    }
    // Starting directly at a populated row must report its complete row cost,
    // including newly live offset families, rather than only the empty floor.
    for (row, explicit, dense, geometric, families) in
        [(4, 1, 1, 0, 1), (6, 0, 0, 2, 2), (7, 2, 1, 0, 2), (9, 1, 0, 0, 1)]
    {
        let required = fixed
            + construction
            + explicit * explicit_bytes
            + dense * dense_bytes
            + geometric * geometric_bytes
            + families * 2 * size_of::<u32>()
            + payload;
        assert_eq!(
            MatrixWindow::required_workspace(&source, row..row + 1, payload).unwrap(),
            required
        );
        assert!(matches!(
            MatrixWindow::load_next_with_payload(&source, row..row + 1, required - 1, payload),
            Err(PiCcsError::MatrixWorkspace { required: actual, available })
                if actual == required && available == required - 1
        ));
    }
}

#[test]
fn streamed_complete_openings_match_resident_with_global_imaginary_weights() {
    let source = PatternRows::new(5);
    let cache = expanded_cache(&source);
    let columns = source.shape().columns;
    let variables = columns.next_power_of_two().ilog2() as usize;
    let point = (0..variables)
        .map(|index| K::from_coeffs([F::from_usize(index + 3), F::from_usize(index + 1)]))
        .collect::<Vec<_>>();
    let witnesses = (0..2)
        .map(|source| {
            SuperneoZBlocks::from_z(
                &(0..columns)
                    .map(|column| K::from(F::from_usize((source + 2 * column) % 3) - F::ONE))
                    .collect::<Vec<_>>(),
            )
        })
        .chain(std::iter::once(SuperneoZBlocks::from_z(&vec![K::ZERO; columns])))
        .collect::<Vec<_>>();
    let expected = cache.eval_real_v1_1_openings(&point, &witnesses).unwrap();
    assert!(expected[0]
        .eval_a
        .iter()
        .flatten()
        .any(|value| value.as_coeffs()[1] != F::ZERO));
    let budget = MatrixWindow::required_workspace(&source, 0..2, 0).unwrap();
    source.visits.lock().unwrap().clear();
    let actual = eval_real_v1_1_openings_from_rows(&source, &point, &witnesses, budget).unwrap();
    assert_eq!(actual, expected);
    assert!(source
        .visits
        .lock()
        .unwrap()
        .iter()
        .any(|range| range.start > 0));
    assert!(actual[2]
        .eval_k
        .iter()
        .chain(actual[2].eval_a.iter().flatten())
        .all(|value| *value == K::ZERO));
}

#[test]
fn zero_openings_skip_rows_only_after_point_and_witness_shape_checks() {
    let source = PatternRows::new(5);
    let columns = source.shape().columns;
    let point = vec![K::from_coeffs([F::ONE, F::ONE]); columns.next_power_of_two().ilog2() as usize];
    let zero = SuperneoZBlocks::from_z(&vec![K::ZERO; columns]);
    let result = eval_real_v1_1_openings_from_rows(&source, &point, &[zero], 0).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].eval_k, vec![K::ZERO; D]);
    assert_eq!(result[0].eval_a, vec![vec![K::ZERO; D]; source.shape().matrices]);
    let zero = SuperneoZBlocks::from_z(&vec![K::ZERO; columns]);
    assert!(eval_real_v1_1_openings_from_rows(&source, &point[..point.len() - 1], &[zero], 0).is_err());
    let wrong_width = SuperneoZBlocks::from_z(&vec![K::ZERO; columns - D]);
    assert!(eval_real_v1_1_openings_from_rows(&source, &point, &[wrong_width], 0).is_err());
    assert!(source.visits.lock().unwrap().is_empty());
}

#[test]
fn streamed_terminal_check_reports_failure_in_the_last_global_window() {
    let source = PatternRows::new(5);
    let cache = expanded_cache(&source);
    let polynomial = SparsePoly::new(
        3,
        vec![Term {
            coeff: F::ONE,
            exps: vec![0, 0, 1],
        }],
    );
    let row_values = source.shape().matrices * size_of::<F>();
    let budget = MatrixWindow::required_workspace(&source, 0..2, row_values).unwrap();
    let mut values = vec![K::ZERO; source.shape().columns];
    let assignment = SuperneoZBlocks::from_z(&values);
    assert_eq!(
        first_unsatisfied_row_from_rows(&source, &polynomial, &assignment, budget).unwrap(),
        None
    );
    let last = source.shape().rows - 1;
    values[last + 2] = K::ONE;
    let assignment = SuperneoZBlocks::from_z(&values);
    assert!(matches!(
        check_ccs_relation_zero_cached_with_blocks(&cache, &polynomial, &assignment),
        Err(SuperneoCachedRelationError::UnsatisfiedRow { row }) if row == last
    ));
    assert_eq!(
        first_unsatisfied_row_from_rows(&source, &polynomial, &assignment, budget).unwrap(),
        Some(last)
    );
}

#[derive(Clone, Copy)]
enum BadOrder {
    WrongRow,
    WrongMatrix,
    WrongRunRow,
    BadColumnRange,
    EmptyRun,
    DuplicateSlot,
    MissingEmptySlot,
}

impl MatrixRows for BadOrder {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: 2,
            columns: D,
            matrices: 2,
        }
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        let row = rows.start;
        match self {
            Self::WrongRow => sink.finish_matrix_row(row + 1, 0).map(|_| ()),
            Self::WrongMatrix => sink.finish_matrix_row(row, 1).map(|_| ()),
            Self::WrongRunRow => sink.push_run(row, 0, GeometricRowRun::new(row + 1, 0, 1, F::ONE, F::ONE)),
            Self::BadColumnRange => sink.push_run(row, 0, GeometricRowRun::new(row, D - 1, 2, F::ONE, F::ONE)),
            Self::EmptyRun => {
                let mut encoded = serde_json::to_value(GeometricRowRun::new(row, 0, 1, F::ONE, F::ONE)).unwrap();
                encoded["len"] = serde_json::json!(0);
                sink.push_run(row, 0, serde_json::from_value(encoded).unwrap())
            }
            Self::DuplicateSlot => {
                let _ = sink.finish_matrix_row(row, 0)?;
                sink.finish_matrix_row(row, 0).map(|_| ())
            }
            Self::MissingEmptySlot => sink.finish_matrix_row(row, 0).map(|_| ()),
        }
    }
}

#[test]
fn malformed_order_range_and_ignored_stop_are_not_workspace_failures() {
    for source in [
        BadOrder::WrongRow,
        BadOrder::WrongMatrix,
        BadOrder::WrongRunRow,
        BadOrder::BadColumnRange,
        BadOrder::EmptyRun,
        BadOrder::DuplicateSlot,
        BadOrder::MissingEmptySlot,
    ] {
        assert!(matches!(
            MatrixWindow::load_next(&source, 0..2, usize::MAX),
            Err(PiCcsError::InvalidInput(_))
        ));
    }
    let mut source = PatternRows::new(5);
    for rows in [3..2, 0..6, 0..0] {
        assert!(matches!(
            MatrixWindow::load_next(&source, rows, usize::MAX),
            Err(PiCcsError::InvalidInput(_))
        ));
    }
    let budget = MatrixWindow::load_next(&source, 0..1, usize::MAX)
        .unwrap()
        .workspace_peak_bytes();
    source.ignore_stop = true;
    assert!(matches!(
        MatrixWindow::load_next(&source, 0..5, budget),
        Err(PiCcsError::InvalidInput(_))
    ));
}

struct ChangedSource(AtomicUsize);

impl MatrixRows for ChangedSource {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: 1,
            columns: D,
            matrices: 1,
        }
    }

    fn visit_rows(&self, _: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        let visits = self.0.fetch_add(1, Ordering::Relaxed);
        sink.push_run(0, 0, GeometricRowRun::new(0, 0, 1, F::ONE, F::ONE))?;
        if visits != 0 {
            sink.push_run(0, 0, GeometricRowRun::new(0, 1, 1, F::ONE, F::ONE))?;
        }
        sink.finish_matrix_row(0, 0).map(|_| ())
    }
}

#[test]
fn changed_run_counts_cannot_grow_the_fill_allocation() {
    let source = ChangedSource(AtomicUsize::new(0));
    assert!(matches!(
        MatrixWindow::load_next(&source, 0..1, usize::MAX),
        Err(PiCcsError::InvalidInput(_))
    ));
}

struct ChangedEncoding(AtomicUsize);

impl MatrixRows for ChangedEncoding {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: 1,
            columns: D,
            matrices: 1,
        }
    }

    fn visit_rows(&self, _: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        let coefficient = if self.0.fetch_add(1, Ordering::Relaxed) == 0 {
            F::ONE
        } else {
            F::from_u64(2)
        };
        sink.push_run(0, 0, GeometricRowRun::new(0, 0, 1, coefficient, F::ONE))?;
        sink.finish_matrix_row(0, 0).map(|_| ())
    }
}

#[test]
fn changed_scalar_encoding_cannot_add_an_uncounted_dense_pattern() {
    assert!(matches!(
        MatrixWindow::load_next(&ChangedEncoding(AtomicUsize::new(0)), 0..1, usize::MAX),
        Err(PiCcsError::InvalidInput(_))
    ));
}

#[test]
fn borrowed_cache_preserves_compact_identity_and_original_seeded_coefficients() {
    let (chunk_size, seeds) = neo_ajtai::seeded_pp_chunk_seeds([0x5C; 32], 1, 1);
    let block = SeededPhi81LinearBlock::new_with_word_width(0, vec![0, 2], 1, 1, 1, chunk_size, seeds).unwrap();
    let seeded = CcsMatrix::csc_with_compact_rows(
        CscMat::from_triplets(vec![(1, 0, -F::ONE)], D, D),
        vec![block.clone()],
        vec![GeometricRowRun::new(1, D - 3, 3, F::from_u64(7), -F::ONE)],
    )
    .unwrap();
    let structure =
        CcsStructure::new_sparse(vec![seeded, CcsMatrix::Identity { n: D }], SparsePoly::new(2, vec![])).unwrap();
    let cache = build_superneo_eval_cache(&structure).unwrap();
    let source = CachedMatrixRows::new(&cache).unwrap();
    let expected = expanded_cache(&source);
    let budget = MatrixWindow::load_next(&source, 1..2, usize::MAX)
        .unwrap()
        .workspace_peak_bytes();
    let mut start = 1;
    while start < 4 {
        let window = MatrixWindow::load_next(&source, start..4, budget).unwrap();
        compare_rows(&window, &expected, D);
        // Also compare against the original seeded evaluator, not just the
        // adapter-expanded dense matrix.
        compare_rows(&window, &cache, D);
        start = window.rows().end;
    }
    let transformed: CcsMatrix<F> = CcsMatrix::csc_with_seeded_phi81(
        CscMat::from_triplets(Vec::new(), D, D),
        vec![block.with_superneo_transformed_columns()],
    )
    .unwrap();
    let transformed = CcsStructure::new_sparse(vec![transformed], SparsePoly::new(1, vec![])).unwrap();
    let transformed = build_superneo_eval_cache(&transformed).unwrap();
    assert!(CachedMatrixRows::new(&transformed).is_err());
}
