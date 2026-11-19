use std::sync::{Mutex, PoisonError};

use neo_ajtai::{
    nightstream_fprime_setup::{
        coefficient_block, commit_production_signed_unit_prefix_matrices, commit_production_signed_unit_prefix_matrix,
        expanded_key_bytes, ExpandedKey, ELEMENT_OUTPUT_BYTES, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS,
    },
    Commitment,
};
use neo_ccs::Mat;
use neo_math::{Fq as F, D};
use p3_field::PrimeCharacteristicRing;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

static SERIAL: Mutex<()> = Mutex::new(());

const KEY_BLOCKS: u64 = 512;
const LAST: usize = KEY_BLOCKS as usize - 1;
const COLUMNS: usize = 2 * KEY_BLOCKS as usize + 3;

fn masks(columns: usize, block: impl FnMut(usize) -> (u64, u64)) -> Mat<F> {
    let (positive, negative): (Vec<u64>, Vec<u64>) = (0..columns).map(block).unzip();
    Mat::compact_signed_unit_from_column_masks(D, columns, &positive, &negative).unwrap()
}

fn only(blocks: &'static [usize]) -> Mat<F> {
    masks(COLUMNS, |block| {
        if blocks.contains(&block) {
            (1 | 1 << 40 | 1 << 63, 1 << 7)
        } else {
            (0, 0)
        }
    })
}

fn random(columns: usize, density: u32) -> Mat<F> {
    let mut rng = ChaCha8Rng::seed_from_u64((columns as u64) << 8 | u64::from(density));
    masks(columns, |_| {
        if rng.random_range(0..100) >= density {
            return (0, 0);
        }
        let positive: u64 = rng.random();
        (positive, rng.random::<u64>() & !positive)
    })
}

fn witnesses() -> Vec<Mat<F>> {
    let mut dense = Mat::zero(D, COLUMNS, F::ZERO);
    for (lane, column, value) in [
        (0, 0, F::ONE),
        (D - 1, LAST, -F::ONE),
        (3, LAST + 1, F::ONE),
        (D - 1, COLUMNS - 1, -F::ONE),
    ] {
        dense[(lane, column)] = value;
    }
    vec![
        Mat::virtual_constant(D, COLUMNS, F::ZERO),
        masks(COLUMNS, |_| (u64::MAX, 0)),
        masks(COLUMNS, |_| (0, u64::MAX)),
        only(&[0]),
        only(&[LAST]),
        only(&[LAST + 1]),
        only(&[COLUMNS - 1]),
        only(&[LAST - 2, LAST - 1, LAST]),
        only(&[LAST, LAST + 1, LAST + 2]),
        only(&[LAST + 1, LAST + 2, COLUMNS - 1]),
        dense,
        random(COLUMNS, 100),
        random(COLUMNS, 30),
        random(LAST, 50),
        random(LAST + 1, 50),
        random(LAST + 2, 50),
    ]
}

fn commitments(witnesses: &[Mat<F>]) -> Vec<Commitment> {
    witnesses
        .iter()
        .map(|witness| commit_production_signed_unit_prefix_matrix(witness).unwrap())
        .collect()
}

#[test]
fn a_commitment_from_the_expanded_key_equals_the_commitment_from_shake() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let witnesses = witnesses();
    let from_shake = commitments(&witnesses);
    assert_eq!(from_shake[0], Commitment::zeros(PRODUCTION_VERIFIER_ROWS as usize));
    assert!(from_shake[1..].iter().all(|commitment| {
        (0..PRODUCTION_VERIFIER_ROWS as usize).all(|row| commitment.col(row).iter().any(|&value| value != F::ZERO))
    }));
    assert!(commit_production_signed_unit_prefix_matrices(&witnesses).unwrap() == from_shake);

    for blocks in [KEY_BLOCKS, LAST as u64, 2, COLUMNS as u64 + 1] {
        let key = ExpandedKey::production(blocks);
        let batch = commit_production_signed_unit_prefix_matrices(&witnesses).unwrap();
        for (index, (one, many)) in commitments(&witnesses).iter().zip(&batch).enumerate() {
            assert!(one == &from_shake[index], "witness {index}, a key of {blocks} blocks");
            assert!(
                many == &from_shake[index],
                "witness {index} of a batch, a key of {blocks} blocks"
            );
        }
        drop(key);
    }
}

#[test]
fn a_commitment_expands_only_the_blocks_after_the_expanded_key() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let expanded = |witness: &Mat<F>| {
        let before = expanded_key_bytes();
        commit_production_signed_unit_prefix_matrix(witness).unwrap();
        commit_production_signed_unit_prefix_matrices(&[witness, witness]).unwrap();
        expanded_key_bytes() - before
    };
    let element_bytes = PRODUCTION_VERIFIER_ROWS * ELEMENT_OUTPUT_BYTES as u64;
    let inside = masks(KEY_BLOCKS as usize, |_| (1, 0));
    let across = masks(COLUMNS, |_| (0, 1));
    let without_key = expanded(&across);
    assert!(without_key >= 2 * COLUMNS as u64 * element_bytes);
    assert!(expanded(&inside) >= 2 * KEY_BLOCKS * element_bytes);

    let key = ExpandedKey::production(KEY_BLOCKS);
    assert_eq!(key.blocks(), KEY_BLOCKS);
    assert_eq!(expanded(&inside), 0);
    let after_key = expanded(&across);
    assert!(after_key >= 2 * (COLUMNS as u64 - KEY_BLOCKS) * element_bytes);
    assert!(after_key < without_key);

    let smaller = ExpandedKey::production(2);
    assert_eq!(expanded(&inside), 0, "a smaller key does not replace a live larger key");
    drop((key, smaller));
    assert_eq!(expanded(&across), without_key, "a dropped key is not read");
}

fn element(key: &ExpandedKey, row: u64, block: u64) -> [u64; D] {
    let start = (row * key.blocks() + block) as usize * D;
    key.words()[start..start + D].try_into().unwrap()
}

#[test]
fn each_word_of_the_expanded_key_is_its_shake_coefficient() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    for blocks in [1, 2, 5, KEY_BLOCKS] {
        let key = ExpandedKey::production(blocks);
        let rounded = blocks.next_multiple_of(2);
        assert_eq!(key.blocks(), rounded);
        assert_eq!(key.words().len() as u64, PRODUCTION_VERIFIER_ROWS * rounded * D as u64);
        for block in 0..rounded {
            for row in 0..PRODUCTION_VERIFIER_ROWS {
                assert_eq!(
                    element(&key, row, block),
                    coefficient_block(&PRODUCTION_SEED, row as u32, block),
                    "row {row}, block {block}"
                );
            }
        }
    }
    let seeded = ExpandedKey::of_pairs(4, |row, pair| {
        [2 * pair, 2 * pair + 1].map(|block| coefficient_block(&[7; 32], row, block))
    });
    for (row, block) in [(0, 0), (1, 2), (PRODUCTION_VERIFIER_ROWS - 1, 3)] {
        assert_eq!(
            element(&seeded, row, block),
            coefficient_block(&[7; 32], row as u32, block)
        );
    }
}
