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
    sbox: bool,
}

impl OriginalRows {
    fn new(rows: usize) -> Self {
        Self {
            rows,
            visits: AtomicUsize::new(0),
            fail: false,
            sbox: false,
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
                    3 if self.sbox && row % 8 >= 3 => {
                        let coefficient = match row % 8 {
                            3..=5 => F::from_u64(5),
                            6 => -F::ONE,
                            _ => F::from_usize(row),
                        };
                        sink.push_run(row, matrix, GeometricRowRun::new(row, D, 1, coefficient, F::ONE))?;
                    }
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

#[test]
fn poseidon2_gate_rounds_match_sparse_evaluation_in_resident_and_replay_tables() {
    let source = OriginalRows {
        sbox: true,
        ..OriginalRows::new(37)
    };
    let blocks = witnesses()
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, 2 * D).unwrap())
        .collect::<Vec<_>>();
    let mut terms = polynomial().terms().to_vec();
    terms[2].exps[3] = neo_ccs::crypto::poseidon2::SBOX_DEGREE as u32;
    let gate = SparsePoly::new(4, terms.clone());
    terms.rotate_left(1);
    let sparse = SparsePoly::new(4, terms);
    assert!(is_poseidon2_gate(&gate));
    assert!(!is_poseidon2_gate(&sparse));
    assert!(!is_poseidon2_gate(&SparsePoly::new(5, gate.terms().to_vec())));

    let variables = source.rows.next_power_of_two().ilog2() as usize;
    let alpha = (0..variables)
        .map(|round| K::from_coeffs([F::from_usize(round + 11), F::from_usize(round + 17)]))
        .collect::<Vec<_>>();
    let mut points: [K; 9] = std::array::from_fn(|i| K::from(F::from_usize(i)));
    points[7] = K::from_coeffs([F::from_u64(43), F::from_u64(41)]);
    points[8] = K::from_coeffs([-F::ONE, F::from_u64(59)]);
    let gamma = K::from_coeffs([F::from_u64(67), F::from_u64(71)]);
    let (replay_budget, resident_budget) = budgets(&source, blocks.len(), points.len());
    let mut replay = ApplicationTables::new(&source, &blocks, replay_budget).unwrap();
    let mut resident = ApplicationTables::new(&source, &blocks, resident_budget).unwrap();
    let mut reference = ApplicationTables::new(&source, &blocks, resident_budget).unwrap();
    let single = &blocks[..1];
    let mut fused = ApplicationTables::new(&source, single, resident_budget).unwrap();
    let mut single_reference = ApplicationTables::new(&source, single, resident_budget).unwrap();
    for round in 0..variables {
        let weights = EqualityWeights::new(&alpha[round + 1..]);
        let expected = reference
            .evals_at(&source, &blocks, &sparse, gamma, &points, &weights)
            .unwrap();
        let challenge = K::from_coeffs([F::from_usize(round + 2), F::from_usize(round + 3)]);
        for tables in [&mut replay, &mut resident] {
            assert_eq!(
                tables
                    .evals_at(&source, &blocks, &gate, gamma, &points, &weights)
                    .unwrap(),
                expected,
                "round {round}"
            );
            tables.fold(challenge);
        }
        reference.fold(challenge);
        assert_eq!(
            fused
                .evals_at(&source, single, &gate, gamma, &points, &weights)
                .unwrap(),
            single_reference
                .evals_at(&source, single, &sparse, gamma, &points, &weights)
                .unwrap(),
        );
        let current = fused.resident.as_ref().unwrap();
        let (address, capacity) = (current.words.as_ptr(), current.words.capacity());
        if round + 1 < variables {
            let next = EqualityWeights::new(&alpha[round + 2..]);
            fused.fold_next_gate(challenge, &gate, &next);
        } else {
            fused.fold(challenge);
        }
        single_reference.fold(challenge);
        let actual = fused.resident.as_ref().unwrap();
        assert_eq!(actual.words, single_reference.resident.as_ref().unwrap().words);
        assert_eq!(actual.words.as_ptr(), address);
        assert_eq!(actual.words.capacity(), capacity);
    }
    assert!(replay.peak_bytes <= replay_budget);
    assert!(resident.peak_bytes <= resident_budget);
}

#[test]
fn complete_first_round_window_serves_the_openings_without_another_visit() {
    let source = OriginalRows::new(17);
    let witnesses = witnesses();
    let blocks = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, 2 * D).unwrap())
        .collect::<Vec<_>>();
    let variables = (2 * D).next_power_of_two().ilog2() as usize;
    let alpha = (0..variables)
        .map(|round| K::from_coeffs([F::from_usize(round + 9), F::from_usize(round + 10)]))
        .collect::<Vec<_>>();
    let points = [K::ZERO, K::ONE];
    let gamma = K::from_coeffs([F::from_u64(11), F::from_u64(13)]);
    let weights = EqualityWeights::new(&alpha[1..]);
    let (replay_budget, resident_budget) = budgets(&source, blocks.len(), points.len());

    let mut replay = ApplicationTables::new(&source, &blocks, replay_budget).unwrap();
    replay
        .evals_at(&source, &blocks, &polynomial(), gamma, &points, &weights)
        .unwrap();
    assert!(
        replay.take_complete_window().is_none(),
        "a partial window is never kept"
    );

    let mut retained = ApplicationTables::new(&source, &blocks, resident_budget).unwrap();
    retained
        .evals_at(&source, &blocks, &polynomial(), gamma, &points, &weights)
        .unwrap();
    assert!(retained.peak_bytes <= resident_budget);
    let window = retained
        .take_complete_window()
        .expect("complete first-round window");
    assert_eq!(window.rows(), 0..source.rows);

    let expected =
        crate::superneo_eval::eval_real_v1_2_openings_from_rows(&source, &alpha, &blocks, resident_budget).unwrap();
    let visits = source.visits.load(Ordering::Relaxed);
    let actual = crate::superneo_eval::eval_real_v1_2_openings_from_rows_reusing(
        &source,
        &alpha,
        &blocks,
        resident_budget,
        Vec::new(),
        Some(window),
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(source.visits.load(Ordering::Relaxed), visits, "no second row visit");
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
#[test]
fn pair_workers_match_serial_with_exact_payload_budgets() {
    let source = OriginalRows::new(2 * prefix::MIN_JOB_PAIRS + 1);
    let witnesses = witnesses();
    let blocks = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, 2 * D).unwrap())
        .collect::<Vec<_>>();
    let reference = original_values(source.rows, &witnesses);
    let points = [K::ZERO, K::ONE, K::from_coeffs([F::from_u64(5), F::from_u64(7)])];
    let gamma = K::from_coeffs([F::from_u64(11), F::from_u64(13)]);
    let variables = source.rows.next_power_of_two().ilog2() as usize;
    let alpha = (1..variables)
        .map(|round| K::from_coeffs([F::from_usize(round + 9), F::from_usize(round + 10)]))
        .collect::<Vec<_>>();
    let weights = EqualityWeights::new(&alpha);
    let scratch = (points.len() + source.shape().matrices) * size_of::<K>();
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
        let workers =
            rayon::current_num_threads().min((rows.div_ceil(2) * points.len()).div_ceil(prefix::MIN_JOB_PAIRS));
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
                        false,
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
    let source = OriginalRows::new(2 * prefix::MIN_JOB_PAIRS + 1);
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

#[test]
fn checked_fresh_rows_match_and_are_consumed_once() {
    struct Prepared {
        source: OriginalRows,
        values: std::sync::Mutex<Option<Vec<[F; 4]>>>,
    }
    impl MatrixRows for Prepared {
        fn shape(&self) -> MatrixShape {
            self.source.shape()
        }
        fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
            self.source.visit_rows(rows, sink)
        }
        fn take_checked_fresh_rows(&self) -> Option<Vec<[F; 4]>> {
            self.values.lock().unwrap().take()
        }
    }
    let rows = 73;
    let inputs = witnesses();
    let blocks = [SuperneoZBlocks::from_witness_mat(&inputs[0], 2 * D).unwrap()];
    let point: Vec<K> = (0..7)
        .map(|i| K::from_coeffs([F::from_usize(i + 2), F::from_usize(i + 5)]))
        .collect();
    let values = original_values(rows, &inputs[..1]);
    let prepared = Prepared {
        source: OriginalRows::new(rows),
        values: std::sync::Mutex::new(Some(
            (0..rows)
                .map(|row| std::array::from_fn(|matrix| values[0][matrix][row].real()))
                .collect(),
        )),
    };
    let (_, budget) = budgets(&prepared.source, blocks.len(), 4);
    let budget = budget + rows * size_of::<[F; 4]>();
    let points = [K::ZERO, K::ONE, K::from(F::from_u8(2)), K::from(F::from_u8(3))];
    let poly = polynomial();
    let mut actual = ApplicationTables::new(&prepared, &blocks, budget).unwrap();
    let mut reference = ApplicationTables::new(&prepared, &blocks, budget).unwrap();
    for round in 0..point.len() {
        let weights = EqualityWeights::new(&point[round + 1..]);
        let visits = prepared.source.visits.load(Ordering::Relaxed);
        let cached = actual
            .evals_at(&prepared, &blocks, &poly, K::ONE, &points, &weights)
            .unwrap();
        assert_eq!(prepared.source.visits.load(Ordering::Relaxed), visits);
        let replayed = reference
            .evals_at(&prepared, &blocks, &poly, K::ONE, &points, &weights)
            .unwrap();
        assert_eq!(cached, replayed);
        if round == 0 {
            assert!(prepared.source.visits.load(Ordering::Relaxed) > visits);
        }
        reference.fold(point[round]);
        actual.fold(point[round]);
    }
    assert!(prepared.take_checked_fresh_rows().is_none());
    assert!(actual.peak_bytes <= budget);
}
