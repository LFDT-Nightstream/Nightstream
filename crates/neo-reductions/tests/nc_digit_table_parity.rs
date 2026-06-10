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
use neo_reductions::engines::optimized_engine::{build_nc_digit_table_compact, NcDigitTable};
use p3_field::PrimeCharacteristicRing;

const COLS: usize = 7;
const M: usize = D * COLS - 5; // ragged tail on purpose

fn params() -> NeoParams {
    NeoParams::goldilocks_paper_b2()
}

/// Witness with digits scattered across many ring lanes (forces `Diagonal`).
fn multi_lane_witness() -> Mat<F> {
    let mut z = Mat::zero(D, COLS, F::ZERO);
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    for rho in 0..D {
        for blk in 0..COLS {
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

fn assert_table_parity(compact: &NcDigitTable, compact_masks: &[u64], dense: &[[K; D]], dense_masks: &[u64]) {
    assert_eq!(compact.len(), dense.len(), "table length mismatch");
    assert_eq!(compact_masks, dense_masks, "mask mismatch");
    for idx in 0..dense.len() {
        assert_eq!(compact.row(idx), dense[idx], "row {idx} mismatch");
        for rho in 0..D {
            assert_eq!(compact.lane(idx, rho), dense[idx][rho], "lane ({idx},{rho}) mismatch");
        }
    }
}

fn run_parity(z: &Mat<F>) {
    let params = params();
    let (mut compact, mut compact_masks) = build_nc_digit_table_compact(&params, z, M).expect("compact table");
    let (dense_rows, mut dense_masks) = build_witness_nc_digit_table_with_masks(&params, z, M).expect("dense table");
    let mut dense = NcDigitTable::Dense(dense_rows.clone());
    {
        let NcDigitTable::Dense(rows) = &dense else {
            unreachable!()
        };
        assert_table_parity(&compact, &compact_masks, rows, &dense_masks);
    }

    // Fold to a single row with distinct challenges: folds 1-5 exercise the
    // in-place strided widths (1 -> 2 -> 4 -> 8 -> 16 -> 32), fold 6 the
    // strided -> Dense materialization (2·32 > D), and the rest the Dense
    // fold. The dense reference folds the same schedule.
    for step in 1u64..=9 {
        let r = K::from_coeffs([F::from_u64(0x1000 + 37 * step), F::from_u64(step)]);
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
