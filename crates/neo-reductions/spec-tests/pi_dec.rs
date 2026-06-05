//! Spec-derived invariant tests for PiDEC.spec.md
//!
//! Each test corresponds to a row in the Invariant Obligations table.

#[path = "common/mod.rs"]
mod common;

use common::seeded_rng;
use neo_ccs::Mat;
use neo_math::F;
use neo_reductions::split_b_matrix_k;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::Rng;

// ---------------------------------------------------------------------------
// 1. DEC round-trip: Sigma b^i * Z_i = Z
// ---------------------------------------------------------------------------

/// PiDEC.spec.md: DEC round-trip: Sigma b^i * Z_i = Z
#[test]
fn dec_round_trip_balanced() {
    let mut rng = seeded_rng(0xDE01);
    let rows = 2;
    let cols = 8;
    let b: u32 = 4;
    let k: usize = 6;

    let data: Vec<F> = (0..rows * cols)
        .map(|_| {
            let v = (rng.random::<u64>() % 200) as i64 - 100;
            if v >= 0 {
                F::from_u64(v as u64)
            } else {
                F::ZERO - F::from_u64((-v) as u64)
            }
        })
        .collect();
    let z = Mat::from_row_major(rows, cols, data);

    let digits = split_b_matrix_k(&z, k, b).expect("split should succeed for small values");
    assert_eq!(digits.len(), k, "should have k={k} digit matrices");

    let mut z_recon = Mat::zero(rows, cols, F::ZERO);
    let mut power = F::ONE;
    let b_f = F::from_u64(b as u64);
    for digit_mat in &digits {
        for r in 0..rows {
            for c in 0..cols {
                z_recon[(r, c)] += power * digit_mat[(r, c)];
            }
        }
        power *= b_f;
    }

    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(z_recon[(r, c)], z[(r, c)], "DEC round-trip failed at ({r},{c})");
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Digit bound: entries lie in [-floor(b/2), +floor(b/2)]
// ---------------------------------------------------------------------------

/// PiDEC.spec.md: Digit bound: entries of Z_i lie in balanced range
#[test]
fn dec_digit_bound() {
    let mut rng = seeded_rng(0xDE02);
    let rows = 2;
    let cols = 4;
    let b: u32 = 8;
    let k: usize = 8;
    let bound = (b - 1) as u64;
    let p = F::ORDER_U64;

    let data: Vec<F> = (0..rows * cols)
        .map(|_| {
            let v = (rng.random::<u64>() % 50) as i64 - 25;
            if v >= 0 {
                F::from_u64(v as u64)
            } else {
                F::ZERO - F::from_u64((-v) as u64)
            }
        })
        .collect();
    let z = Mat::from_row_major(rows, cols, data);

    let digits = split_b_matrix_k(&z, k, b).expect("split should succeed");

    for (i, digit_mat) in digits.iter().enumerate() {
        for r in 0..rows {
            for c in 0..cols {
                let val = digit_mat[(r, c)].as_canonical_u64();
                let in_range = val <= bound || val >= p - bound;
                assert!(
                    in_range,
                    "digit[{i}][{r},{c}] = {val} out of SuperNeo digit range [-{bound}, {bound}] (mod p)"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. DEC with b=2 (binary decomposition)
// ---------------------------------------------------------------------------

/// PiDEC.spec.md: Binary decomposition (b=2) round-trip
#[test]
fn dec_binary_round_trip() {
    let rows = 2;
    let cols = 4;
    let b: u32 = 2;
    let k: usize = 64;

    let data: Vec<F> = (0..rows * cols)
        .map(|i| F::from_u64((i as u64 + 1) * 7))
        .collect();
    let z = Mat::from_row_major(rows, cols, data);

    let digits = split_b_matrix_k(&z, k, b).expect("binary split should succeed");

    let mut z_recon = Mat::zero(rows, cols, F::ZERO);
    let mut power = F::ONE;
    let b_f = F::from_u64(2);
    for digit_mat in &digits {
        for r in 0..rows {
            for c in 0..cols {
                z_recon[(r, c)] += power * digit_mat[(r, c)];
            }
        }
        power *= b_f;
    }

    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(z_recon[(r, c)], z[(r, c)], "binary DEC round-trip failed at ({r},{c})");
        }
    }
}

// ---------------------------------------------------------------------------
// 4. DEC with k=1 (single digit = identity for small values)
// ---------------------------------------------------------------------------

/// PiDEC.spec.md: k=1 decomposition is identity for small values
#[test]
fn dec_k1_identity() {
    let rows = 2;
    let cols = 4;
    let b: u32 = 256;
    let k: usize = 1;

    // Values within balanced range: [-127, 127]
    let data: Vec<F> = (0..rows * cols)
        .map(|i| {
            let v = (i as i64) - 4;
            if v >= 0 {
                F::from_u64(v as u64)
            } else {
                F::ZERO - F::from_u64((-v) as u64)
            }
        })
        .collect();
    let z = Mat::from_row_major(rows, cols, data);

    let digits = split_b_matrix_k(&z, k, b).expect("k=1 split should succeed for small values");
    assert_eq!(digits.len(), 1);

    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(
                digits[0][(r, c)],
                z[(r, c)],
                "k=1 digit should equal original at ({r},{c})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. DEC with zero matrix
// ---------------------------------------------------------------------------

/// PiDEC.spec.md: zero matrix decomposes to all-zero digits
#[test]
fn dec_zero_matrix() {
    let rows = 2;
    let cols = 4;
    let b: u32 = 4;
    let k: usize = 3;

    let z = Mat::zero(rows, cols, F::ZERO);
    let digits = split_b_matrix_k(&z, k, b).expect("zero matrix split should succeed");

    for (i, digit_mat) in digits.iter().enumerate() {
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(
                    digit_mat[(r, c)],
                    F::ZERO,
                    "zero matrix digit[{i}][{r},{c}] should be zero"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. DEC nonzero flags
// ---------------------------------------------------------------------------

/// PiDEC.spec.md: nonzero flags correctly identify all-zero digit layers
#[test]
fn dec_nonzero_flags() {
    let rows = 2;
    let cols = 4;
    let b: u32 = 4;
    let k: usize = 6;

    let z = Mat::zero(rows, cols, F::ZERO);
    let (digits, flags) = neo_reductions::split_b_matrix_k_with_nonzero_flags(&z, k, b).expect("split should succeed");

    for (i, &flag) in flags.iter().enumerate() {
        let has_nonzero = (0..rows).any(|r| (0..cols).any(|c| digits[i][(r, c)] != F::ZERO));
        assert_eq!(
            flag, has_nonzero,
            "nonzero flag for digit {i} should match actual content"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. DEC X decomposition formula
// ---------------------------------------------------------------------------

/// PiDEC.spec.md: X decomposition: parent = Sigma b^i * child_i
#[test]
fn dec_x_decomposition_formula() {
    let mut rng = seeded_rng(0xDE07);
    let b: u32 = 4;
    let k: usize = 4;
    let n = 8;

    // Generate parent values and decompose each
    let parent_vals: Vec<F> = (0..n)
        .map(|_| {
            let v = (rng.random::<u64>() % 20) as i64 - 10;
            if v >= 0 {
                F::from_u64(v as u64)
            } else {
                F::ZERO - F::from_u64((-v) as u64)
            }
        })
        .collect();

    let b_f = F::from_u64(b as u64);

    for (idx, &parent) in parent_vals.iter().enumerate() {
        let single = Mat::from_row_major(1, 1, vec![parent]);
        let digits = split_b_matrix_k(&single, k, b).expect("split should succeed");

        let mut recon = F::ZERO;
        let mut power = F::ONE;
        for d in &digits {
            recon += power * d[(0, 0)];
            power *= b_f;
        }
        assert_eq!(recon, parent, "decomposition failed at index {idx}");
    }
}
