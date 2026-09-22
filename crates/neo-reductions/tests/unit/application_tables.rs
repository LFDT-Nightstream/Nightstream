use super::*;
use crate::superneo_eval::MatrixRowSink;
use neo_ccs::{poly::Term, GeometricRowRun, Mat};
use neo_math::KExtensions;
use std::{
    ops::{ControlFlow, Range},
    sync::atomic::{AtomicUsize, Ordering},
};

struct OriginalRows {
    rows: usize,
    visits: AtomicUsize,
    fail: bool,
}

impl OriginalRows {
    fn new(rows: usize) -> Self {
        Self {
            rows,
            visits: AtomicUsize::new(0),
            fail: false,
        }
    }
}

impl MatrixRows for OriginalRows {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            columns: 2 * D,
            matrices: 4,
        }
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.visits.fetch_add(1, Ordering::Relaxed);
        if self.fail {
            return Err(invalid("synthetic row source failure"));
        }
        for row in rows {
            for matrix in 0..self.shape().matrices {
                match matrix {
                    0 => {
                        sink.push_run(
                            row,
                            matrix,
                            GeometricRowRun::new(row, 0, 1, F::from_usize(row + 1), F::ONE),
                        )?;
                        sink.push_run(
                            row,
                            matrix,
                            GeometricRowRun::new(row, D - 1, 3, F::from_usize(row + 2), F::from_u64(3)),
                        )?;
                    }
                    1 => sink.push_run(
                        row,
                        matrix,
                        GeometricRowRun::new(row, D, 1, F::from_usize(row + 3), F::ONE),
                    )?,
                    2 if row + 1 == self.rows => sink.push_run(
                        row,
                        matrix,
                        GeometricRowRun::new(row, D + 1, 1, F::from_usize(row + 5), F::ONE),
                    )?,
                    _ => {}
                }
                if sink.finish_matrix_row(row, matrix)? == ControlFlow::Break(()) {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

fn witnesses() -> Vec<Mat<F>> {
    vec![
        Mat::compact_signed_unit_from_column_masks(D, 2, &[1, 3], &[1 << (D - 1), 0]).unwrap(),
        Mat::virtual_constant(D, 2, F::ZERO),
        Mat::compact_signed_unit_from_column_masks(D, 2, &[1 << (D - 1), 0], &[1, 3]).unwrap(),
    ]
}

fn polynomial() -> SparsePoly<F> {
    SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 1, 0, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 1, 0],
            },
            Term {
                coeff: F::ONE,
                exps: vec![0, 0, 0, 2],
            },
        ],
    )
}

fn original_values(rows: usize, witnesses: &[Mat<F>]) -> Vec<Vec<Vec<K>>> {
    witnesses
        .iter()
        .map(|witness| {
            let value = |column| witness[(column % D, column / D)];
            (0..4)
                .map(|matrix| {
                    (0..rows)
                        .map(|row| {
                            K::from(match matrix {
                                0 => {
                                    F::from_usize(row + 1) * value(0)
                                        + F::from_usize(row + 2)
                                            * (value(D - 1) + F::from_u64(3) * value(D) + F::from_u64(9) * value(D + 1))
                                }
                                1 => F::from_usize(row + 3) * value(D),
                                2 if row + 1 == rows => F::from_usize(row + 5) * value(D + 1),
                                _ => F::ZERO,
                            })
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn reference_round(tables: &[Vec<Vec<K>>], points: &[K], gamma: K, weights: &EqualityWeights) -> Vec<K> {
    points
        .iter()
        .map(|&point| {
            let mut result = K::ZERO;
            let mut source_weight = K::ONE;
            for matrices in tables {
                for pair in 0..matrices[0].len().div_ceil(2) {
                    let values: Vec<K> = matrices
                        .iter()
                        .map(|matrix| {
                            let low = matrix[2 * pair];
                            let high = matrix.get(2 * pair + 1).copied().unwrap_or(K::ZERO);
                            (K::ONE - point) * low + point * high
                        })
                        .collect();
                    result +=
                        source_weight * weights.at(pair) * (values[0] * values[1] - values[2] + values[3] * values[3]);
                }
                source_weight *= gamma;
            }
            result
        })
        .collect()
}

fn budgets(source: &OriginalRows, fresh_count: usize, point_count: usize) -> (usize, usize) {
    let payload = size_of::<F>() + size_of::<K>();
    let one_row = (0..source.rows)
        .map(|row| MatrixWindow::required_workspace(source, row..row + 1, payload).unwrap())
        .max()
        .unwrap();
    let complete = MatrixWindow::required_workspace(source, 0..source.rows, payload).unwrap();
    let row_bytes = fresh_count * source.shape().matrices * size_of::<K>();
    let scratch = (point_count + source.shape().matrices) * size_of::<K>();
    (
        scratch + 2 * row_bytes + one_row,
        scratch + source.rows * row_bytes + complete,
    )
}

#[test]
fn fresh_application_replay_matches_retained_and_independent_rows() {
    // Seventeen is the first odd row count beyond sixteen. A two-row output
    // workspace therefore keeps replay active through three preceding folds.
    let source = OriginalRows::new(17);
    let witnesses = witnesses();
    let blocks = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, 2 * D).unwrap())
        .collect::<Vec<_>>();
    let mut reference = original_values(source.rows, &witnesses);
    let variables = (2 * D).next_power_of_two().ilog2() as usize;
    let alpha = (0..variables)
        .map(|round| K::from_coeffs([F::from_usize(round + 9), F::from_usize(round + 10)]))
        .collect::<Vec<_>>();
    let points = [K::ZERO, K::ONE, K::from_coeffs([F::from_u64(5), F::from_u64(7)])];
    let gamma = K::from_coeffs([F::from_u64(11), F::from_u64(13)]);
    let (replay_budget, resident_budget) = budgets(&source, blocks.len(), points.len());
    let mut replay = ApplicationTables::new(&source, &blocks, replay_budget).unwrap();
    let mut retained = ApplicationTables::new(&source, &blocks, resident_budget).unwrap();
    let polynomial = polynomial();
    for round in 0..variables {
        let weights = EqualityWeights::new(&alpha[round + 1..]);
        let expected = reference_round(&reference, &points, gamma, &weights);
        assert_eq!(
            replay
                .evals_at(&source, &blocks, &polynomial, gamma, &points, &weights)
                .unwrap(),
            expected,
            "replay round {round}"
        );
        assert_eq!(
            retained
                .evals_at(&source, &blocks, &polynomial, gamma, &points, &weights)
                .unwrap(),
            expected,
            "retained round {round}"
        );
        assert!(retained.resident.is_some());
        if round <= 3 {
            assert!(
                replay.resident.is_none(),
                "three prior challenges must still use replay"
            );
        }
        assert!(replay.peak_bytes <= replay_budget);
        assert!(retained.peak_bytes <= resident_budget);
        let challenge = K::from_coeffs([F::from_usize(round + 2), F::from_usize(round + 3)]);
        replay.fold(challenge);
        retained.fold(challenge);
        for matrix in reference.iter_mut().flatten() {
            *matrix = matrix
                .chunks(2)
                .map(|pair| (K::ONE - challenge) * pair[0] + challenge * pair.get(1).copied().unwrap_or(K::ZERO))
                .collect();
        }
    }
    assert!(
        replay.resident.is_some(),
        "the final prefix fits the same supplied budget"
    );
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
#[test]
fn pair_workers_match_serial_with_exact_payload_budgets() {
    let source = OriginalRows::new(17);
    let witnesses = witnesses();
    let blocks = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, 2 * D).unwrap())
        .collect::<Vec<_>>();
    let reference = original_values(source.rows, &witnesses);
    let points = [K::ZERO, K::ONE, K::from_coeffs([F::from_u64(5), F::from_u64(7)])];
    let gamma = K::from_coeffs([F::from_u64(11), F::from_u64(13)]);
    let variables = (2 * D).next_power_of_two().ilog2() as usize;
    let alpha = (1..variables)
        .map(|round| K::from_coeffs([F::from_usize(round + 9), F::from_usize(round + 10)]))
        .collect::<Vec<_>>();
    let weights = EqualityWeights::new(&alpha);
    let scratch = (points.len() + source.shape().matrices) * size_of::<K>();
    // The first nonzero pair boundary also checks a replay Values window.
    for row_start in [0, 2] {
        let mut expected = reference_round(&reference, &points, gamma, &weights);
        if row_start != 0 {
            let prefix = reference
                .iter()
                .map(|matrices| {
                    matrices
                        .iter()
                        .map(|matrix| matrix[..row_start].to_vec())
                        .collect()
                })
                .collect::<Vec<_>>();
            for (total, prefix) in expected
                .iter_mut()
                .zip(reference_round(&prefix, &points, gamma, &weights))
            {
                *total -= prefix;
            }
        }
        let rows = source.rows - row_start;
        let workers = rayon::current_num_threads().min(rows.div_ceil(2));
        // Zero spare payload forces serial work. Two is the smallest parallel
        // chunk count; the last case uses every worker up to the pair count.
        for extra_workers in [0, workers.min(2), workers] {
            let values = Values {
                words: reference
                    .iter()
                    .flatten()
                    .flat_map(|matrix| matrix[row_start..].iter().copied())
                    .collect(),
                rows,
            };
            let live = values.words.capacity() * size_of::<K>() + scratch;
            let workspace_bytes = live + extra_workers * scratch;
            let mut owner = ApplicationTables::new(&source, &blocks, workspace_bytes).unwrap();
            let actual = if row_start == 0 {
                owner.resident = Some(values);
                owner
                    .evals_at(&source, &blocks, &polynomial(), gamma, &points, &weights)
                    .unwrap()
            } else {
                let mut result = vec![K::ZERO; points.len()];
                let mut coordinates = vec![K::ZERO; source.shape().matrices];
                let peak = owner
                    .accumulate(
                        &values,
                        row_start,
                        &polynomial(),
                        gamma,
                        &points,
                        &weights,
                        &mut coordinates,
                        &mut result,
                    )
                    .unwrap();
                owner.record(peak).unwrap();
                result
            };
            assert_eq!(actual, expected);
            let expected_peak = if extra_workers > 1 { workspace_bytes } else { live };
            assert_eq!(owner.peak_bytes, expected_peak, "worker payload must be counted");
            assert!(owner.peak_bytes <= workspace_bytes);
        }
    }
    assert_eq!(source.visits.load(Ordering::Relaxed), 0);
}

#[test]
fn fresh_table_build_uses_existing_output_slices() {
    let source = OriginalRows::new(17);
    let witnesses = witnesses();
    let blocks = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, 2 * D).unwrap())
        .collect::<Vec<_>>();
    let expected = original_values(source.rows, &witnesses)
        .into_iter()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();
    let output_bytes = expected.len() * size_of::<K>();
    let workspace_bytes = output_bytes + MatrixWindow::required_workspace(&source, 0..source.rows, 0).unwrap();
    let mut owner = ApplicationTables::new(&source, &blocks, workspace_bytes).unwrap();
    source.visits.store(0, Ordering::Relaxed);
    let values = owner.values(&source, &blocks, 0, source.rows, 0).unwrap();
    assert_eq!(values.words, expected);
    assert_eq!(owner.peak_bytes, workspace_bytes);
    assert_eq!(
        source.visits.load(Ordering::Relaxed),
        2,
        "the complete fixture fits one count/fill window without worker row buffers"
    );
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
#[test]
fn table_folds_match_serial_without_another_value_allocation() {
    let source = OriginalRows::new(17);
    let witnesses = witnesses();
    let blocks = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, 2 * D).unwrap())
        .collect::<Vec<_>>();
    let mut reference = original_values(source.rows, &witnesses);
    let words = reference
        .iter()
        .flatten()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let owner = || {
        let values = Values {
            words: words.clone(),
            rows: source.rows,
        };
        let workspace_bytes = values.words.capacity() * size_of::<K>();
        let mut owner = ApplicationTables::new(&source, &blocks, workspace_bytes).unwrap();
        owner.record(workspace_bytes).unwrap();
        owner.resident = Some(values);
        owner
    };
    let mut serial = owner();
    let mut parallel = owner();
    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon::current_num_threads())
        .build()
        .unwrap();
    let serial_pointer = serial.resident.as_ref().unwrap().words.as_ptr();
    let parallel_pointer = parallel.resident.as_ref().unwrap().words.as_ptr();
    let serial_capacity = serial.resident.as_ref().unwrap().words.capacity();
    let parallel_capacity = parallel.resident.as_ref().unwrap().words.capacity();
    let rounds = (2 * D).next_power_of_two().ilog2() as usize;
    for round in 0..rounds {
        let challenge = K::from_coeffs([F::from_usize(round + 2), F::from_usize(round + 3)]);
        serial_pool.install(|| serial.fold(challenge));
        parallel_pool.install(|| parallel.fold(challenge));
        for matrix in reference.iter_mut().flatten() {
            *matrix = matrix
                .chunks(2)
                .map(|pair| (K::ONE - challenge) * pair[0] + challenge * pair.get(1).copied().unwrap_or(K::ZERO))
                .collect();
        }
        let expected = reference
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let serial_values = serial.resident.as_ref().unwrap();
        let parallel_values = parallel.resident.as_ref().unwrap();
        assert_eq!(serial_values.words, expected, "serial fold {round}");
        assert_eq!(parallel_values.words, expected, "parallel fold {round}");
        assert_eq!(serial_values.words.as_ptr(), serial_pointer);
        assert_eq!(parallel_values.words.as_ptr(), parallel_pointer);
        assert_eq!(serial_values.words.capacity(), serial_capacity);
        assert_eq!(parallel_values.words.capacity(), parallel_capacity);
        assert_eq!(serial.challenges, parallel.challenges);
        assert_eq!(serial.challenges.len(), round + 1);
        assert_eq!(serial.peak_bytes, serial.workspace_bytes);
        assert_eq!(parallel.peak_bytes, parallel.workspace_bytes);
    }
}

#[test]
fn zero_polynomial_needs_no_matrix_windows_and_source_errors_are_preserved() {
    let mut source = OriginalRows::new(3);
    let witnesses = witnesses();
    let blocks = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, 2 * D).unwrap())
        .collect::<Vec<_>>();
    let points = [K::ZERO, K::ONE];
    let (_, resident_budget) = budgets(&source, blocks.len(), points.len());
    source.visits.store(0, Ordering::Relaxed);
    source.fail = true;
    let mut owner = ApplicationTables::new(&source, &blocks, resident_budget).unwrap();
    let weights = EqualityWeights::new(&[K::ONE]);
    assert_eq!(
        owner
            .evals_at(&source, &blocks, &SparsePoly::new(4, vec![]), K::ONE, &points, &weights)
            .unwrap(),
        vec![K::ZERO; points.len()]
    );
    assert_eq!(source.visits.load(Ordering::Relaxed), 0);
    assert!(matches!(
        owner.evals_at(&source, &blocks, &polynomial(), K::ONE, &points, &weights),
        Err(PiCcsError::InvalidInput(reason)) if reason == "synthetic row source failure"
    ));
    assert_eq!(source.visits.load(Ordering::Relaxed), 1);
}

#[test]
fn application_values_reject_a_budget_that_cannot_hold_a_pair() {
    let source = OriginalRows::new(3);
    let witnesses = witnesses();
    let blocks = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, 2 * D).unwrap())
        .collect::<Vec<_>>();
    let points = [K::ZERO, K::ONE];
    let scratch = (points.len() + source.shape().matrices) * size_of::<K>();
    let mut owner = ApplicationTables::new(&source, &blocks, scratch).unwrap();
    assert!(owner
        .evals_at(
            &source,
            &blocks,
            &polynomial(),
            K::ONE,
            &points,
            &EqualityWeights::new(&[K::ONE])
        )
        .is_err());
    assert!(owner.peak_bytes <= scratch);
}
