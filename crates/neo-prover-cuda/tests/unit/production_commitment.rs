use super::*;
use neo_ajtai::nightstream_fprime_setup::{
    coefficient, commit_production_signed_unit_prefix_matrices, MAX_MESSAGE_COLUMNS, PRODUCTION_MESSAGE_COLUMNS,
};
use p3_field::PrimeField64;

#[test]
fn production_key_commitments_match_cpu_across_groups_blocks_and_representations() {
    let session = CudaSession::new().unwrap();
    let columns = 2 * COLUMNS_PER_GROUP + 1;
    let positive = Mat::virtual_constant(D, columns, F::ONE);
    let negative = Mat::virtual_constant(D, columns, -F::ONE);
    let mut dense = Mat::zero(D, columns, F::ZERO);
    for column in 0..columns {
        for lane in 0..D {
            dense[(lane, column)] = match (column + lane) % 3 {
                0 => F::ONE,
                1 => -F::ONE,
                _ => F::ZERO,
            };
        }
    }
    let masks = (0..columns)
        .map(|column| {
            let mut pair = [0; 2];
            for lane in 0..D {
                match dense[(lane, column)] {
                    value if value == F::ONE => pair[0] |= 1 << lane,
                    value if value == -F::ONE => pair[1] |= 1 << lane,
                    _ => {}
                }
            }
            pair
        })
        .collect::<Vec<_>>();
    let packed = Mat::compact_signed_unit_from_column_masks(
        D,
        columns,
        &masks.iter().map(|pair| pair[0]).collect::<Vec<_>>(),
        &masks.iter().map(|pair| pair[1]).collect::<Vec<_>>(),
    )
    .unwrap();
    let short = Mat::virtual_constant(D, 1, -F::ONE);
    let mut witnesses = vec![
        positive,
        Mat::virtual_constant(D, columns, F::ZERO),
        negative,
        dense,
        packed,
        short,
    ];
    while witnesses.len() <= MAX_WITNESSES_PER_GROUP + 2 {
        witnesses.push(witnesses[3 + witnesses.len() % 3].clone());
    }
    let expected = commit_production_signed_unit_prefix_matrices(&witnesses).unwrap();
    let actual = session.commit_production_prefixes(&witnesses).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual[3], actual[4]);
    assert!(session.activity().launches > PRODUCTION_VERIFIER_ROWS);
}

/// Witnesses whose nonzero columns are `columns`, so occupied positions and
/// key columns differ.
fn sparse_witnesses(width: usize, columns: &[usize]) -> Vec<Mat<F>> {
    let lanes = (1u64 << D) - 1;
    (0..3u64)
        .map(|seed| {
            let mut positive = vec![0; width];
            let mut negative = vec![0; width];
            for (index, &column) in columns.iter().enumerate() {
                let bits = (0x9e37_79b9_7f4a_7c15u64.rotate_left((seed * 7 + index as u64) as u32)) & lanes;
                positive[column] = bits & 0x5555_5555_5555_5555;
                negative[column] = bits & !0x5555_5555_5555_5555;
            }
            Mat::compact_signed_unit_from_column_masks(D, width, &positive, &negative).unwrap()
        })
        .collect()
}

#[test]
fn kept_production_key_matches_cpu_and_grows_for_a_wider_witness() {
    let mut session = CudaSession::new().unwrap();
    session.keep_production_key();
    let wide = 2 * COLUMNS_PER_GROUP + 1;
    let narrow = sparse_witnesses(10, &[2, 9]);
    let calls = [
        narrow.clone(),
        sparse_witnesses(wide, &[0, 5, COLUMNS_PER_GROUP, wide - 1]),
        narrow,
    ];
    let mut launches = Vec::new();
    for witnesses in &calls {
        let before = session.activity().launches;
        let expected = commit_production_signed_unit_prefix_matrices(witnesses).unwrap();
        assert_eq!(session.commit_production_prefixes(witnesses).unwrap(), expected);
        launches.push(session.activity().launches - before);
    }
    // The first two calls expand all key rows; the last reuses them and only
    // accumulates and sums each row.
    let rows = PRODUCTION_VERIFIER_ROWS;
    assert_eq!(launches, [3 * rows, 3 * rows, 2 * rows]);
}

#[test]
fn a_released_production_key_expands_again_at_the_next_commitment() {
    let mut session = CudaSession::new().unwrap();
    session.keep_production_key();
    let witnesses = sparse_witnesses(10, &[2, 9]);
    let expected = commit_production_signed_unit_prefix_matrices(&witnesses).unwrap();
    let mut launches = Vec::new();
    for release in [false, true, false] {
        if release {
            session.release_production_key_rows();
        }
        let before = session.activity().launches;
        assert_eq!(session.commit_production_prefixes(&witnesses).unwrap(), expected);
        launches.push(session.activity().launches - before);
    }
    // Expanding all key rows costs one launch per row on top of the two
    // that accumulate and sum it.
    let rows = PRODUCTION_VERIFIER_ROWS;
    assert_eq!(launches, [3 * rows, 3 * rows, 2 * rows]);
}

#[test]
fn production_key_coefficients_use_exact_first_and_last_indexed_addresses() {
    let session = CudaSession::new().unwrap();
    for column in [0, PRODUCTION_MESSAGE_COLUMNS as usize - 1] {
        let mut positive = vec![0; column + 1];
        positive[column] = 1;
        let witness =
            Mat::compact_signed_unit_from_column_masks(D, column + 1, &positive, &vec![0; column + 1]).unwrap();
        // Odd batches leave the shared scratch unaligned to 16 bytes.
        for count in [1, 3] {
            let actual = session
                .commit_production_prefixes(&vec![witness.clone(); count])
                .unwrap();
            assert_eq!(actual.len(), count);
            for commitment in actual {
                for row in 0..PRODUCTION_VERIFIER_ROWS as usize {
                    for lane in 0..D {
                        assert_eq!(
                            commitment.data()[row * D + lane].as_canonical_u64(),
                            coefficient(&PRODUCTION_SEED, row as u32, column as u64, lane as u32),
                            "witnesses={count} row={row} column={column} lane={lane}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn production_commitment_checks_all_inputs_before_device_work() {
    let session = CudaSession::new().unwrap();
    let before = session.activity();
    let zero = Mat::virtual_constant(D, PRODUCTION_MESSAGE_COLUMNS as usize, F::ZERO);
    assert_eq!(
        session
            .commit_production_prefixes(std::slice::from_ref(&zero))
            .unwrap(),
        vec![Commitment::zeros(PRODUCTION_VERIFIER_ROWS as usize)]
    );
    assert!(session
        .commit_production_prefixes::<Mat<F>>(&[])
        .unwrap()
        .is_empty());
    let mut invalid = Mat::zero(D, 2, F::ZERO);
    invalid[(D - 1, 1)] = F::from_u64(2);
    for invalid in [
        invalid,
        Mat::virtual_constant(D - 1, 1, F::ZERO),
        Mat::virtual_constant(D, 0, F::ZERO),
        Mat::virtual_constant(D, MAX_MESSAGE_COLUMNS as usize + 1, F::ZERO),
    ] {
        let expected = signed_unit_prefix_blocks(&invalid).err().unwrap();
        let error = session
            .commit_production_prefixes(&[Mat::virtual_constant(D, 1, F::ONE), invalid, zero.clone()])
            .unwrap_err();
        assert!(matches!(error, CudaError::Commitment(actual) if actual == expected));
    }
    assert_eq!(session.activity().launches, before.launches);
    assert_eq!(session.activity().allocated_bytes, before.allocated_bytes);
}

#[test]
#[ignore = "timing evidence at production width; run on its own under the 300 s cap"]
fn production_commitment_cost_per_witness() {
    let streamed = CudaSession::new().unwrap();
    let mut kept = CudaSession::new().unwrap();
    kept.keep_production_key();
    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let lanes = (1u64 << D) - 1;
    let witnesses: Vec<Mat<F>> = (0..16)
        .map(|_| {
            let (positive, negative): (Vec<_>, Vec<_>) = (0..columns)
                .map(|_| {
                    let (signs, nonzero) = (next(), next() & lanes);
                    (nonzero & signs, nonzero & !signs)
                })
                .unzip();
            Mat::compact_signed_unit_from_column_masks(D, columns, &positive, &negative).unwrap()
        })
        .collect();
    for (key, session) in [("streamed", &streamed), ("kept", &kept)] {
        for count in [1usize, 1, 4, 16] {
            let started = std::time::Instant::now();
            session
                .commit_production_prefixes(&witnesses[..count])
                .unwrap();
            eprintln!(
                "commit key={key} witnesses={count} seconds={:.3}",
                started.elapsed().as_secs_f64()
            );
        }
    }
}

/// SHAKE128 on the device: every coefficient of key rows `0`, `1` and the
/// last equals `coefficient_block` on the CPU, at columns that set each byte
/// of a 32-bit column index (the key-row kernel's column type).
#[test]
fn shake128_key_rows_match_the_cpu() {
    use neo_ajtai::nightstream_fprime_setup::coefficient_block;
    let session = CudaSession::new().unwrap();
    let narrow: Vec<u32> = [0, 1, 255, 256, 65_537, PRODUCTION_MESSAGE_COLUMNS as u32 - 1, u32::MAX]
        .into_iter()
        .chain((0..4).map(|byte| 0xa5u32 << (8 * byte)))
        .collect();
    let prefix = session.production_key_prefix().unwrap();
    let device_columns = session.buffer_from_slice(&narrow).unwrap();
    let slab = session.buffer(narrow.len() * D * size_of::<u64>()).unwrap();
    for row in [0, 1, PRODUCTION_VERIFIER_ROWS - 1] {
        session
            .launch_key_row(&prefix, &device_columns, &slab, narrow.len(), row)
            .unwrap();
        session.finish().unwrap();
        let words = session.read_buffer::<u64>(&slab, narrow.len() * D).unwrap();
        for (&column, coefficients) in narrow.iter().zip(words.chunks_exact(D)) {
            assert_eq!(
                coefficients,
                coefficient_block(&PRODUCTION_SEED, row as u32, u64::from(column)),
                "row {row} column {column}"
            );
        }
    }
}
