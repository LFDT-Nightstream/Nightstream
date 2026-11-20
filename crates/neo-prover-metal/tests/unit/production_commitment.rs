use super::*;
use neo_ajtai::nightstream_fprime_setup::{
    coefficient, commit_production_signed_unit_prefix_matrices, MAX_MESSAGE_COLUMNS, PRODUCTION_MESSAGE_COLUMNS,
};
use p3_field::PrimeField64;

#[test]
fn production_key_commitments_match_cpu_across_groups_blocks_and_representations() {
    let session = MetalSession::new().unwrap();
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
    let block = session
        .production_ajtai_accumulate
        .maxTotalThreadsPerThreadgroup()
        / D;
    while witnesses.len() <= block + 2 {
        witnesses.push(witnesses[3 + witnesses.len() % 3].clone());
    }
    let expected = commit_production_signed_unit_prefix_matrices(&witnesses).unwrap();
    let actual = session.commit_production_prefixes(&witnesses).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual[3], actual[4]);
    assert!(session.activity().dispatches > PRODUCTION_VERIFIER_ROWS);
}

fn sparse_witnesses(width: usize, columns: &[usize]) -> Vec<Mat<F>> {
    (0..3u64)
        .map(|seed| {
            let mut positive = vec![0; width];
            let mut negative = vec![0; width];
            for (index, &column) in columns.iter().enumerate() {
                let bits = 0x9e37_79b9_7f4a_7c15u64.rotate_left((seed * 7 + index as u64) as u32);
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
    let rows = PRODUCTION_VERIFIER_ROWS;
    assert_eq!(dispatches, [3 * rows, 3 * rows, 2 * rows]);
}

#[test]
fn concurrent_commitment_rows_match_cpu_across_partial_groups() {
    let mut session = MetalSession::new().unwrap();
    session.keep_production_key();
    let columns = 3 * COLUMNS_PER_GROUP + 1;
    let witnesses = sparse_witnesses(columns, &(0..columns).filter(|i| i % 13 != 0).collect::<Vec<_>>());
    let fresh = &witnesses[..1];
    let expected = commit_production_signed_unit_prefix_matrices(fresh).unwrap();
    assert_eq!(session.commit_production_prefixes(fresh).unwrap(), expected);
}

#[test]
fn production_key_coefficients_use_exact_first_and_last_indexed_addresses() {
    let session = MetalSession::new().unwrap();
    for column in [0, PRODUCTION_MESSAGE_COLUMNS as usize - 1] {
        let mut positive = vec![0; column + 1];
        positive[column] = 1;
        let witness =
            Mat::compact_signed_unit_from_column_masks(D, column + 1, &positive, &vec![0; column + 1]).unwrap();
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
    let session = MetalSession::new().unwrap();
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
        assert!(matches!(error, MetalError::Commitment(actual) if actual == expected));
    }
    assert_eq!(session.activity().dispatches, before.dispatches);
    assert_eq!(session.activity().allocated_bytes, before.allocated_bytes);
}

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
    let witnesses: Vec<Mat<F>> = (0..16)
        .map(|_| {
            let (positive, negative): (Vec<_>, Vec<_>) = (0..columns)
                .map(|_| {
                    let (signs, nonzero) = (next(), next());
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

#[test]
fn matrix_commitment_accumulates_full_positive_and_negative_byte_limbs() {
    let session = MetalSession::new().unwrap();
    let Some(matrix) = &session.production_ajtai_matrix else {
        return;
    };
    let columns = COLUMNS_PER_GROUP;
    let value = F::ORDER_U64 - 1;
    let key = session.buffer_from_slice(&[value; D]).unwrap();
    let indices = session.buffer_from_slice(&vec![0_u32; columns]).unwrap();
    let shape = session
        .buffer_from_slice(&[columns as u64, 1, 1, columns as u64])
        .unwrap();
    let result = session.buffer(D * size_of::<u64>()).unwrap();
    for (mask, sign) in [([u64::MAX, 0], F::ONE), ([0, u64::MAX], -F::ONE)] {
        let masks = session.buffer_from_slice(&vec![mask; columns]).unwrap();
        let command = session
            .command_buffer("nightstream.test.matrix_commitment")
            .unwrap();
        let encoder = command.computeCommandEncoder().unwrap();
        session.launch(
            &encoder,
            matrix,
            &[
                Arg::Buffer(0, &key),
                Arg::Buffer(1, &masks),
                Arg::Buffer(2, &shape),
                Arg::Buffer(3, &result),
                Arg::Buffer(4, &indices),
            ],
            Grid::Groups {
                groups: 1,
                threads: 256,
            },
        );
        encoder.endEncoding();
        session.finish(&command).unwrap();
        for (lane, &word) in session.host_words(&result, D).iter().enumerate() {
            let shifts = F::from_usize(2 * (lane + 1)) - F::from_usize(D);
            assert_eq!(
                F::from_u64(word),
                sign * shifts * F::from_usize(columns) * F::from_u64(value)
            );
        }
    }
}

#[test]
fn metal_reuses_the_expanded_key_pages_and_retains_their_owner() {
    let session = MetalSession::new().unwrap();
    let columns = 8192;
    let key = ExpandedKey::of_pairs(columns as u64, |row, pair| {
        [2 * pair, 2 * pair + 1]
            .map(|column| std::array::from_fn(|lane| (row as u64) << 48 | column << 6 | lane as u64))
    });
    let words = key.shared_words();
    let rows = session.shared_key_rows(&key, columns / 2).unwrap();
    if !words
        .as_ptr()
        .addr()
        .is_multiple_of(objc2_foundation::NSPageSize())
    {
        assert!(rows.is_none());
        return;
    }
    let rows = rows.unwrap();
    for (row, buffer) in rows.iter().enumerate() {
        let expected = &words[row * columns * D..][..columns / 2 * D];
        let view = session.host_words(buffer, expected.len());
        assert_eq!(view.as_ptr(), expected.as_ptr());
        assert_eq!(view.last(), expected.last());
    }
    drop((key, words));
    let last_row = PRODUCTION_VERIFIER_ROWS - 1;
    let last = session.host_words(rows.last().unwrap(), columns / 2 * D);
    assert_eq!(
        last.last(),
        Some(&((last_row << 48) | ((columns as u64 / 2 - 1) << 6) | 63))
    );
}
