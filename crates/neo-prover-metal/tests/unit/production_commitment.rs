use super::*;
use neo_ajtai::nightstream_fprime_setup::{
    coefficient, commit_production_signed_unit_prefix_matrices, MAX_MESSAGE_COLUMNS, PRODUCTION_MESSAGE_COLUMNS,
};
use p3_field::PrimeField64;

#[test]
fn production_key_commitments_match_cpu_across_groups_blocks_and_representations() {
    let session = MetalSession::new().unwrap();
    // Two full column ranges and one column give three group partials, the
    // last one short. All signed sums exceed 64 bits before reduction.
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
    // More active witnesses than one threadgroup serves, so a second witness
    // block reads the same key row.
    let block = session
        .production_ajtai_accumulate
        .maxTotalThreadsPerThreadgroup()
        / 64;
    while witnesses.len() <= block + 2 {
        witnesses.push(witnesses[3 + witnesses.len() % 3].clone());
    }
    let expected = commit_production_signed_unit_prefix_matrices(&witnesses).unwrap();
    let actual = session.commit_production_prefixes(&witnesses).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual[3], actual[4]);
    assert!(session.activity().dispatches > PRODUCTION_VERIFIER_ROWS);
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
    let mut session = MetalSession::new().unwrap();
    session.keep_production_key();
    let wide = 2 * COLUMNS_PER_GROUP + 1;
    let narrow = sparse_witnesses(10, &[2, 9]);
    let calls = [
        narrow.clone(),
        sparse_witnesses(wide, &[0, 5, COLUMNS_PER_GROUP, wide - 1]),
        narrow,
    ];
    let mut dispatches = Vec::new();
    for witnesses in &calls {
        let before = session.activity().dispatches;
        let expected = commit_production_signed_unit_prefix_matrices(witnesses).unwrap();
        assert_eq!(session.commit_production_prefixes(witnesses).unwrap(), expected);
        dispatches.push(session.activity().dispatches - before);
    }
    // The first two calls expand all key rows; the last reuses them and only
    // accumulates and sums each row.
    let rows = PRODUCTION_VERIFIER_ROWS;
    assert_eq!(dispatches, [3 * rows, 3 * rows, 2 * rows]);
}

#[test]
fn production_key_coefficients_use_exact_first_and_last_indexed_addresses() {
    let session = MetalSession::new().unwrap();
    for column in [0, PRODUCTION_MESSAGE_COLUMNS as usize - 1] {
        let mut positive = vec![0; column + 1];
        positive[column] = 1;
        let witness =
            Mat::compact_signed_unit_from_column_masks(D, column + 1, &positive, &vec![0; column + 1]).unwrap();
        // Odd batches exercise Metal's 16-byte threadgroup memory alignment.
        for count in [1, 3] {
            let actual = session
                .commit_production_prefixes(&vec![witness.clone(); count])
                .unwrap();
            assert_eq!(actual.len(), count);
            for commitment in actual {
                for row in 0..PRODUCTION_VERIFIER_ROWS as usize {
                    for lane in 0..D {
                        assert_eq!(
                            commitment.data[row * D + lane].as_canonical_u64(),
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
    let session = MetalSession::new().unwrap();
    let before = session.activity();
    let zero = Mat::virtual_constant(D, PRODUCTION_MESSAGE_COLUMNS as usize, F::ZERO);
    assert_eq!(
        session
            .commit_production_prefixes(std::slice::from_ref(&zero))
            .unwrap(),
        vec![Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize)]
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
        assert!(matches!(error, MetalError::Commitment(actual) if actual == expected));
    }
    assert_eq!(session.activity().dispatches, before.dispatches);
    assert_eq!(session.activity().allocated_bytes, before.allocated_bytes);
}

/// Commit time for 1, 4 and 16 dense random witnesses at production width,
/// with streamed and with kept key rows, for comparing commitment kernels.
/// It only prints timings; the first kept call expands the key.
#[test]
#[ignore = "timing evidence at production width; run on its own under the 300 s cap"]
fn production_commitment_cost_per_witness() {
    let streamed = MetalSession::new().unwrap();
    let mut kept = MetalSession::new().unwrap();
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
