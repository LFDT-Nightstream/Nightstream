use super::*;
use neo_ajtai::nightstream_fprime_setup::{
    coefficient, commit_production_signed_unit_prefix_matrices, MAX_MESSAGE_COLUMNS, PRODUCTION_MESSAGE_COLUMNS,
};
use p3_field::PrimeField64;

#[test]
fn production_key_commitments_match_cpu_across_tiles_and_representations() {
    let session = MetalSession::new().unwrap();
    // Two full hardware tiles and one column exercise a partial key tile and
    // an odd reduction level. All signed sums include carries beyond 64 bits.
    let columns = 2 * session.production_ajtai_partials.threadExecutionWidth() + 1;
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
    let witnesses = [
        positive,
        Mat::virtual_constant(D, columns, F::ZERO),
        negative,
        dense,
        packed,
        short,
    ];
    let expected = commit_production_signed_unit_prefix_matrices(&witnesses).unwrap();
    let actual = session.commit_production_prefixes(&witnesses).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual[3], actual[4]);
    assert!(session.activity().dispatches > PRODUCTION_VERIFIER_ROWS);
}

#[test]
fn production_key_coefficients_use_exact_first_and_last_indexed_addresses() {
    let session = MetalSession::new().unwrap();
    for column in [0, PRODUCTION_MESSAGE_COLUMNS as usize - 1] {
        let mut positive = vec![0; column + 1];
        positive[column] = 1;
        let witness =
            Mat::compact_signed_unit_from_column_masks(D, column + 1, &positive, &vec![0; column + 1]).unwrap();
        let actual = session
            .commit_production_prefixes(&[witness])
            .unwrap()
            .remove(0);
        for row in 0..PRODUCTION_VERIFIER_ROWS as usize {
            for lane in 0..D {
                assert_eq!(
                    actual.data[row * D + lane].as_canonical_u64(),
                    coefficient(&PRODUCTION_SEED, row as u32, column as u64, lane as u32),
                    "row={row} column={column} lane={lane}"
                );
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
    assert!(session.commit_production_prefixes(&[]).unwrap().is_empty());
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
