//! Parity pins for the compact NC digit table against the dense reference.
//!
//! `build_nc_digit_table_compact` returns `Lane0`/`Diagonal` tables (one K
//! per logical column) whose implicit dense form must match
//! `build_witness_nc_digit_table_with_masks` (the paper-exact reference
//! also used by `tests-paper-exact/oracle_self_check.rs`) row-for-row and
//! mask-for-mask — before folding and across folds, where `Diagonal`
//! materializes its `Dense` form. Any divergence changes the NC sumcheck's
//! prover messages, i.e. the Fiat-Shamir transcript.

#![allow(non_snake_case)]

use neo_ccs::Mat;
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::common::build_witness_nc_digit_table_with_masks;
use neo_reductions::engines::optimized_engine::{build_nc_digit_table_compact, NcDigitMasks, NcDigitTable};
use p3_field::PrimeCharacteristicRing;

const COLS: usize = 7;
const M: usize = D * COLS - 5; // ragged tail on purpose

fn params() -> NeoParams {
    NeoParams::goldilocks_paper_b2()
}

/// Witness with digits scattered across many ring lanes (forces the
/// strided diagonal variant).
fn multi_lane_witness_cols(cols: usize) -> Mat<F> {
    let mut z = Mat::zero(D, cols, F::ZERO);
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    for rho in 0..D {
        for blk in 0..cols {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            match state >> 62 {
                0 => z[(rho, blk)] = F::ONE,
                1 => z[(rho, blk)] = F::ZERO - F::ONE,
                _ => {}
            }
        }
    }
    z
}

fn multi_lane_witness() -> Mat<F> {
    multi_lane_witness_cols(COLS)
}

/// Witness live only in ring lane 0 (takes the `Lane0` variant).
fn lane0_witness() -> Mat<F> {
    let mut z = Mat::zero(D, COLS, F::ZERO);
    for blk in 0..COLS {
        if blk % 2 == 0 {
            z[(0, blk)] = F::ONE;
        }
    }
    z
}

fn assert_table_parity(
    compact: &NcDigitTable,
    compact_masks: &NcDigitMasks,
    dense: &[[K; D]],
    dense_masks: &NcDigitMasks,
) {
    assert_eq!(compact.len(), dense.len(), "table length mismatch");
    assert_eq!(compact_masks.len(), dense_masks.len(), "mask length mismatch");
    assert_eq!(compact_masks.to_dense(), dense_masks.to_dense(), "mask mismatch");
    for idx in 0..dense.len() {
        assert_eq!(compact.row(idx), dense[idx], "row {idx} mismatch");
        for rho in 0..D {
            assert_eq!(compact.lane(idx, rho), dense[idx][rho], "lane ({idx},{rho}) mismatch");
        }
    }
}

fn run_parity(z: &Mat<F>) {
    let challenges: Vec<K> = (1u64..=9)
        .map(|step| K::from_coeffs([F::from_u64(0x1000 + 37 * step), F::from_u64(step)]))
        .collect();
    run_parity_at(z, M, &challenges);
}

fn run_parity_at(z: &Mat<F>, m: usize, challenges: &[K]) {
    let params = params();
    let (mut compact, mut compact_masks) = build_nc_digit_table_compact(&params, z, m).expect("compact table");
    let (dense_rows, dense_masks) = build_witness_nc_digit_table_with_masks(&params, z, m).expect("dense table");
    let mut dense = NcDigitTable::Dense(dense_rows.clone());
    let mut dense_masks = NcDigitMasks::Dense(dense_masks);
    {
        let NcDigitTable::Dense(rows) = &dense else {
            unreachable!()
        };
        assert_table_parity(&compact, &compact_masks, rows, &dense_masks);
    }

    // Fold with the given schedule: the first folds exercise the in-place
    // strided widths (1 -> 2 -> 4 -> 8 -> 16 -> 32), then the strided ->
    // Dense materialization (2·32 > D), then the Dense fold. The dense
    // reference folds the same schedule.
    for &r in challenges {
        compact.fold_inplace(&mut compact_masks, r);
        dense.fold_inplace(&mut dense_masks, r);
        let NcDigitTable::Dense(rows) = &dense else {
            panic!("dense table must stay dense across folds")
        };
        assert_table_parity(&compact, &compact_masks, rows, &dense_masks);
    }
}

#[test]
fn diagonal_table_matches_dense_reference_across_folds() {
    run_parity(&multi_lane_witness());
}

#[test]
fn lane0_table_matches_dense_reference_across_folds() {
    run_parity(&lane0_witness());
}

#[test]
fn zero_witness_table_matches_dense_reference_across_folds() {
    run_parity(&Mat::zero(D, COLS, F::ZERO));
}

/// Degenerate challenges: r = 0 and r = 1 collapse the fold's lo/hi terms
/// (`v·(1-r)` / `v·r` become identity/zero), and a purely imaginary r has
/// no base-field part. The compact folds' shortcut paths must still match
/// the dense reference exactly.
#[test]
fn degenerate_challenges_match_dense_reference() {
    let z = multi_lane_witness();
    let challenges: Vec<K> = vec![
        K::from_coeffs([F::ZERO, F::ZERO]), // r = 0
        K::from_coeffs([F::ONE, F::ZERO]),  // r = 1
        K::from_coeffs([F::ZERO, F::ONE]),  // purely imaginary
        K::from_coeffs([F::ONE, F::ONE]),
        K::from_coeffs([F::ZERO, F::ZERO]), // r = 0 again, post-materialization widths
        K::from_coeffs([F::from_u64(7), F::from_u64(11)]),
        K::from_coeffs([F::ONE, F::ZERO]),
        K::from_coeffs([F::from_u64(3), F::ZERO]),
        K::from_coeffs([F::ZERO, F::from_u64(5)]),
    ];
    run_parity_at(&z, M, &challenges);
}

/// Shape edges: a single logical column, exactly one ring block, and an
/// exact multiple of D (no ragged tail). Each folds to exhaustion.
#[test]
fn boundary_shapes_match_dense_reference() {
    let challenges: Vec<K> = (1u64..=9)
        .map(|step| K::from_coeffs([F::from_u64(0x2000 + 13 * step), F::from_u64(step + 1)]))
        .collect();
    // Single column: only var 0 exists; lane 0 of block 0.
    let mut single = Mat::zero(D, 1, F::ZERO);
    single[(0, 0)] = F::ZERO - F::ONE;
    run_parity_at(&single, 1, &challenges);
    // Exactly one block (m = D): every lane live.
    let mut one_block = Mat::zero(D, 1, F::ZERO);
    for rho in 0..D {
        one_block[(rho, 0)] = if rho % 2 == 0 { F::ONE } else { F::ZERO - F::ONE };
    }
    run_parity_at(&one_block, D, &challenges);
    // Exact multiple of D, no ragged tail.
    run_parity_at(&multi_lane_witness_cols(4), 4 * D, &challenges);
}

/// Densest possible signed witness: every digit is -1 (all lanes live,
/// every mask bit set, every value the balanced negative unit).
#[test]
fn all_minus_one_witness_matches_dense_reference() {
    let cols = COLS;
    let mut z = Mat::zero(D, cols, F::ZERO);
    for rho in 0..D {
        for blk in 0..cols {
            z[(rho, blk)] = F::ZERO - F::ONE;
        }
    }
    run_parity_at(
        &z,
        D * cols,
        &(1u64..=9)
            .map(|step| K::from_coeffs([F::from_u64(0x3000 + 7 * step), F::from_u64(2 * step)]))
            .collect::<Vec<_>>(),
    );
}

/// Only the last ring lane (rho = D-1) is live: exercises the lane-window
/// wrap-around in the strided accessor (`(idx * width + j) % D`).
#[test]
fn last_lane_only_witness_matches_dense_reference() {
    let cols = COLS;
    let mut z = Mat::zero(D, cols, F::ZERO);
    for blk in 0..cols {
        z[(D - 1, blk)] = F::ONE;
    }
    run_parity_at(
        &z,
        D * cols,
        &(1u64..=9)
            .map(|step| K::from_coeffs([F::from_u64(0x4000 + 5 * step), F::from_u64(step)]))
            .collect::<Vec<_>>(),
    );
}
