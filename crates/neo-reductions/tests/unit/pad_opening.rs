use super::*;
use neo_ccs::Mat;
use neo_math::{Rq, F};

/// The original Pad formula: two full ring products per block.
fn dense_products(value_at: impl Fn(usize, usize) -> F, blocks: usize, weights: &EqualityWeights) -> [K; D] {
    let mut out = [K::ZERO; D];
    for block in 0..blocks {
        let value = Rq(std::array::from_fn(|lane| value_at(block, lane)));
        let weights: [K; D] = std::array::from_fn(|lane| weights.at(block * D + lane));
        let real = Rq(superneo_bar_block(weights.map(|weight| weight.as_coeffs()[0]))).mul(&value);
        let imaginary = Rq(superneo_bar_block(weights.map(|weight| weight.as_coeffs()[1]))).mul(&value);
        for lane in 0..D {
            out[lane] += K::from_coeffs([real.0[lane], imaginary.0[lane]]);
        }
    }
    out
}

#[test]
fn batched_pad_openings_match_dense_ring_products_for_every_witness_storage() {
    let blocks = 3;
    let point: Vec<K> = (0..8u64)
        .map(|index| K::from_coeffs([F::from_u64(7 * index + 3), F::from_u64(5 * index + 1)]))
        .collect();
    let weights = EqualityWeights::new(&point);

    // Unit, small non-unit and full-field digits use the dense digit storage.
    let digits: Vec<F> = (0..blocks * D)
        .map(|index| match index % 7 {
            0 | 6 => F::ONE,
            1 => -F::ONE,
            2 => F::from_u64(2),
            4 => -F::from_u64(3),
            5 if index % 2 == 1 => F::from_u64(0x1234_5678_9abc_def0),
            _ => F::ZERO,
        })
        .collect();
    let dense = SuperneoZBlocks::from_z(
        &digits
            .iter()
            .map(|&digit| K::from(digit))
            .collect::<Vec<_>>(),
    );
    // Packed signed-unit masks, including a zero block and the top lane.
    let positive = [0b1011u64, 0, 1 << (D - 1)];
    let negative = [0b0100u64, 0, 1];
    let packed = Mat::compact_signed_unit_from_column_masks(D, blocks, &positive, &negative).unwrap();
    let signed = SuperneoZBlocks::from_witness_mat(&packed, blocks * D).unwrap();
    // A zero witness between them keeps its place in the batch.
    let zero = SuperneoZBlocks::from_z(&vec![K::ZERO; blocks * D]);
    assert_eq!(
        pad_openings(&[dense, zero, signed], &point),
        vec![
            dense_products(|block, lane| digits[block * D + lane], blocks, &weights),
            [K::ZERO; D],
            dense_products(|block, lane| packed[(lane, block)], blocks, &weights),
        ]
    );
}

#[test]
fn tensor_pad_matches_ring_products_across_phases_tasks_and_boolean_points() {
    for (blocks, variables) in [(1, 6), (33, 11), (TASK_BLOCKS + 3, 18)] {
        let positive: Vec<_> = (0..blocks)
            .map(|block| (block as u64 * 0x1234_5678_9abc + 1) & ((1 << D) - 1))
            .collect();
        let negative: Vec<_> = positive
            .iter()
            .enumerate()
            .map(|(block, positive)| (!(block as u64 * 0x0fed_cba9_8765) & !positive) & ((1 << D) - 1))
            .collect();
        let packed = Mat::compact_signed_unit_from_column_masks(D, blocks, &positive, &negative).unwrap();
        let witness = SuperneoZBlocks::from_witness_mat(&packed, blocks * D).unwrap();
        for kind in 0..3 {
            let point: Vec<_> = (0..variables)
                .map(|index| match kind {
                    0 => K::ZERO,
                    1 => K::ONE,
                    _ => K::from_coeffs([F::from_u64(index * 5 + 3), F::from_u64(index * 7 + 1)]),
                })
                .collect();
            let weights = EqualityWeights::new(&point);
            assert_eq!(
                pad_openings(std::slice::from_ref(&witness), &point)[0],
                dense_products(|block, lane| packed[(lane, block)], blocks, &weights),
                "blocks {blocks}, point kind {kind}"
            );
        }
    }
}
