//! Spec-derived tests for Matrix.spec.md invariant obligations.
//!
//! Covers: Mat layout, identity construction/detection, CSC round-trip,
//! CcsMatrix identity mul, column selector detection.

#[path = "common/mod.rs"]
mod common;

use neo_ccs::{CcsMatrix, CscMat, Mat, SeededPhi81LinearBlock};
use neo_math::D;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

// ---------------------------------------------------------------------------
// 1. row_major_layout
// ---------------------------------------------------------------------------
#[test]
fn row_major_layout() {
    // Mat[i,j] == data[i*cols + j]
    let rows = 3;
    let cols = 4;
    let data: Vec<F> = (0..12).map(|k| F::from_u64(k as u64)).collect();
    let m = Mat::from_row_major(rows, cols, data.clone());

    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(
                m[(i, j)],
                data[i * cols + j],
                "Mat[{i},{j}] should equal data[{idx}]",
                idx = i * cols + j
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. identity_construction_and_detection
// ---------------------------------------------------------------------------
#[test]
fn identity_construction_and_detection() {
    for n in [1, 2, 3, 5, 8] {
        let id = Mat::<F>::identity(n);
        assert!(id.is_identity(), "Mat::identity({n}) should be detected as identity");
        assert!(
            id.is_identity_hint(),
            "Mat::identity({n}) should have identity hint set"
        );

        // Verify diagonal is 1, off-diagonal is 0.
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_eq!(id[(i, j)], F::ONE, "diagonal should be 1");
                } else {
                    assert_eq!(id[(i, j)], F::ZERO, "off-diagonal should be 0");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. identity_hint_cleared_on_mut
// ---------------------------------------------------------------------------
#[test]
fn identity_hint_cleared_on_mut() {
    let mut id = Mat::<F>::identity(3);
    assert!(id.is_identity_hint(), "identity_hint should be set initially");

    // Mutate via set()
    id.set(0, 0, F::ONE); // same value, but set() clears hint unconditionally
    assert!(!id.is_identity_hint(), "identity_hint should be cleared after set()");

    // Still structurally identity (data unchanged), so is_identity() should still pass.
    assert!(id.is_identity(), "should still be structurally identity");
}

// ---------------------------------------------------------------------------
// 4. csc_round_trip_mul
// ---------------------------------------------------------------------------
#[test]
fn csc_round_trip_mul() {
    // Build a 3x3 dense matrix:
    // [2, 0, 1]
    // [0, 3, 0]
    // [4, 0, 5]
    let m = Mat::from_row_major(
        3,
        3,
        vec![
            F::from_u64(2),
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::from_u64(3),
            F::ZERO,
            F::from_u64(4),
            F::ZERO,
            F::from_u64(5),
        ],
    );

    // Convert to CSC.
    let csc = CscMat::from_dense_row_major(&m);

    // Test M*x via add_mul_into.
    let x = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];

    // Expected: M*x = [2*1 + 0*2 + 1*3, 0*1 + 3*2 + 0*3, 4*1 + 0*2 + 5*3]
    //                = [5, 6, 19]
    let mut y = vec![F::ZERO; 3];
    csc.add_mul_into(&x, &mut y, 3);

    assert_eq!(y[0], F::from_u64(5), "M*x[0] should be 5");
    assert_eq!(y[1], F::from_u64(6), "M*x[1] should be 6");
    assert_eq!(y[2], F::from_u64(19), "M*x[2] should be 19");

    // Verify against manual dense computation.
    let mut y_dense = vec![F::ZERO; 3];
    for i in 0..3 {
        for j in 0..3 {
            y_dense[i] += m[(i, j)] * x[j];
        }
    }
    assert_eq!(y, y_dense, "CSC mul should match dense mul");
}

// ---------------------------------------------------------------------------
// 5. csc_from_triplets_dedup
// ---------------------------------------------------------------------------
#[test]
fn csc_from_triplets_dedup() {
    // Duplicate (row, col) entries should be summed.
    let triplets = vec![
        (0, 0, F::from_u64(3)),
        (0, 0, F::from_u64(7)), // same (0,0) => sum = 10
        (1, 1, F::from_u64(5)),
    ];

    let csc = CscMat::from_triplets(triplets, 2, 2);

    // Check M*[1, 1] = [10, 5]
    let x = vec![F::ONE, F::ONE];
    let mut y = vec![F::ZERO; 2];
    csc.add_mul_into(&x, &mut y, 2);

    assert_eq!(y[0], F::from_u64(10), "duplicate entries should be summed");
    assert_eq!(y[1], F::from_u64(5));
}

// ---------------------------------------------------------------------------
// 6. csc_transpose_mul
// ---------------------------------------------------------------------------
#[test]
fn csc_transpose_mul() {
    // M = [2, 0, 1]
    //     [0, 3, 0]
    //     [4, 0, 5]
    // M^T * x for x = [1, 2, 3]:
    // M^T = [2, 0, 4]   => M^T*x = [2*1+0*2+4*3, 0*1+3*2+0*3, 1*1+0*2+5*3]
    //       [0, 3, 0]              = [14, 6, 16]
    //       [1, 0, 5]
    let m = Mat::from_row_major(
        3,
        3,
        vec![
            F::from_u64(2),
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::from_u64(3),
            F::ZERO,
            F::from_u64(4),
            F::ZERO,
            F::from_u64(5),
        ],
    );

    let csc = CscMat::from_dense_row_major(&m);

    let x = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
    let mut y = vec![F::ZERO; 3];
    csc.add_mul_transpose_into(&x, &mut y, 3);

    assert_eq!(y[0], F::from_u64(14), "M^T*x[0] should be 14");
    assert_eq!(y[1], F::from_u64(6), "M^T*x[1] should be 6");
    assert_eq!(y[2], F::from_u64(16), "M^T*x[2] should be 16");
}

// ---------------------------------------------------------------------------
// 7. ccs_matrix_identity_mul
// ---------------------------------------------------------------------------
#[test]
fn ccs_matrix_identity_mul() {
    // CcsMatrix::Identity should act as I in both mul and transpose_mul.
    let n = 4;
    let id = CcsMatrix::<F>::Identity { n };

    let x: Vec<F> = (1..=4).map(|k| F::from_u64(k)).collect();

    // I * x = x
    let mut y = vec![F::ZERO; n];
    id.add_mul_into(&x, &mut y, n);
    assert_eq!(y, x, "Identity mul should return x");

    // I^T * x = x
    let mut y2 = vec![F::ZERO; n];
    id.add_mul_transpose_into(&x, &mut y2, n);
    assert_eq!(y2, x, "Identity transpose mul should return x");
}

// ---------------------------------------------------------------------------
// 8. is_column_selector_detection
// ---------------------------------------------------------------------------
#[test]
fn is_column_selector_detection() {
    // A column selector: each column has exactly one 1, rest 0.
    // 3x2 selector: selects rows 1 and 2
    let m = Mat::from_row_major(3, 2, vec![F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE]);
    assert!(m.is_column_selector(), "valid column selector should be detected");

    // Identity is also a column selector.
    let id = Mat::<F>::identity(3);
    assert!(id.is_column_selector(), "identity should be a column selector");

    // Not a column selector: column 0 has two 1s.
    let not_sel = Mat::from_row_major(2, 2, vec![F::ONE, F::ZERO, F::ONE, F::ONE]);
    assert!(
        !not_sel.is_column_selector(),
        "matrix with two 1s in a column should not be a column selector"
    );

    // Not a column selector: column has a non-0/1 entry.
    let not_sel2 = Mat::from_row_major(2, 1, vec![F::from_u64(2), F::ZERO]);
    assert!(
        !not_sel2.is_column_selector(),
        "matrix with entry != 0 or 1 should not be a column selector"
    );
}

// ---------------------------------------------------------------------------
// 9. csr_round_trip
// ---------------------------------------------------------------------------
#[test]
fn csr_round_trip() {
    // Build a sparse 3x4 dense matrix using neo_math::F (CsrMatrix is F-specific).
    use neo_math::F as Fq;

    let m = Mat::from_row_major(
        3,
        4,
        vec![
            Fq::from_u64(2u64),
            Fq::ZERO,
            Fq::ONE,
            Fq::ZERO,
            Fq::ZERO,
            Fq::from_u64(3u64),
            Fq::ZERO,
            Fq::ZERO,
            Fq::from_u64(4u64),
            Fq::ZERO,
            Fq::from_u64(5u64),
            Fq::from_u64(7u64),
        ],
    );

    let csr = m.to_csr();

    // Verify dimensions preserved.
    assert_eq!(csr.rows, 3);
    assert_eq!(csr.cols, 4);

    // spmv_transpose: compute M^T * r  where r is given as (re, im) pairs.
    // Using r = [(1,0), (2,0), (3,0)] (real-only):
    //   M^T * [1,2,3] = [2*1+0*2+4*3, 0*1+3*2+0*3, 1*1+0*2+5*3, 0*1+0*2+7*3]
    //                  = [14, 6, 16, 21]
    let r_pairs: Vec<(Fq, Fq)> = vec![
        (Fq::ONE, Fq::ZERO),
        (Fq::from_u64(2u64), Fq::ZERO),
        (Fq::from_u64(3u64), Fq::ZERO),
    ];

    let (v_re, v_im) = csr.spmv_transpose(&r_pairs);

    assert_eq!(v_re[0], Fq::from_u64(14u64), "M^T*r[0]");
    assert_eq!(v_re[1], Fq::from_u64(6u64), "M^T*r[1]");
    assert_eq!(v_re[2], Fq::from_u64(16u64), "M^T*r[2]");
    assert_eq!(v_re[3], Fq::from_u64(21u64), "M^T*r[3]");
    // Imaginary parts should all be zero.
    for (i, &vi) in v_im.iter().enumerate() {
        assert_eq!(vi, Fq::ZERO, "v_im[{i}] should be zero");
    }
}

// ---------------------------------------------------------------------------
// 10. csr_row_nz_and_nnz
// ---------------------------------------------------------------------------
#[test]
fn csr_row_nz_and_nnz() {
    use neo_math::F as Fq;

    let m = Mat::from_row_major(
        3,
        3,
        vec![
            Fq::from_u64(2u64),
            Fq::ZERO,
            Fq::ONE,
            Fq::ZERO,
            Fq::ZERO,
            Fq::ZERO, // all-zero row
            Fq::from_u64(4u64),
            Fq::ZERO,
            Fq::from_u64(5u64),
        ],
    );

    let csr = m.to_csr();

    // Row 0: two non-zeros at cols 0 and 2.
    let (cols, vals) = csr.row_nz(0);
    assert_eq!(cols, &[0, 2]);
    assert_eq!(vals, &[Fq::from_u64(2u64), Fq::ONE]);
    assert_eq!(csr.row_nnz(0), 2);

    // Row 1: all zeros.
    let (cols1, vals1) = csr.row_nz(1);
    assert!(cols1.is_empty());
    assert!(vals1.is_empty());
    assert_eq!(csr.row_nnz(1), 0);

    // Row 2: two non-zeros.
    assert_eq!(csr.row_nnz(2), 2);

    // Total nnz.
    assert_eq!(csr.nnz(), 4);

    // Also test the dense Mat methods.
    assert_eq!(m.nnz(), 4);
    assert_eq!(m.row_nnz(0), 2);
    assert_eq!(m.row_nnz(1), 0);
    assert_eq!(m.row_nnz(2), 2);

    // Dense row_nz iterator.
    let dense_nz: Vec<(usize, &Fq)> = m.row_nz(0).collect();
    assert_eq!(dense_nz.len(), 2);
    assert_eq!(dense_nz[0].0, 0); // col 0
    assert_eq!(dense_nz[1].0, 2); // col 2
}

// ---------------------------------------------------------------------------
// 11. n_eff_limits_rows
// ---------------------------------------------------------------------------
#[test]
fn n_eff_limits_rows() {
    // Build a 4x3 matrix and verify that n_eff < nrows restricts which rows are processed.
    //
    // M = [1, 0, 0]
    //     [0, 2, 0]
    //     [0, 0, 3]
    //     [4, 4, 4]
    let m = Mat::from_row_major(
        4,
        3,
        vec![
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::from_u64(2),
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(4),
            F::from_u64(4),
        ],
    );

    let csc = CscMat::from_dense_row_major(&m);
    let x = vec![F::ONE, F::ONE, F::ONE];

    // Full mul: y = M*x = [1, 2, 3, 12]
    let mut y_full = vec![F::ZERO; 4];
    csc.add_mul_into(&x, &mut y_full, 4);
    assert_eq!(y_full[0], F::ONE);
    assert_eq!(y_full[1], F::from_u64(2));
    assert_eq!(y_full[2], F::from_u64(3));
    assert_eq!(y_full[3], F::from_u64(12));

    // n_eff = 2: only update y[0..2], rows 2 and 3 are excluded.
    let mut y_limited = vec![F::ZERO; 4];
    csc.add_mul_into(&x, &mut y_limited, 2);
    assert_eq!(y_limited[0], F::ONE, "row 0 should be computed");
    assert_eq!(y_limited[1], F::from_u64(2), "row 1 should be computed");
    assert_eq!(y_limited[2], F::ZERO, "row 2 should be untouched");
    assert_eq!(y_limited[3], F::ZERO, "row 3 should be untouched");

    // Transpose with n_eff = 2: y = M^T * x[0..2]
    // Only rows 0 and 1 of M contribute:
    //   M^T * [1, 1, -, -] = [1*1+0*1, 0*1+2*1, 0*1+0*1] = [1, 2, 0]
    let x4 = vec![F::ONE, F::ONE, F::from_u64(99), F::from_u64(99)];
    let mut y_t = vec![F::ZERO; 3];
    csc.add_mul_transpose_into(&x4, &mut y_t, 2);
    assert_eq!(y_t[0], F::ONE, "M^T col 0 from rows 0..2");
    assert_eq!(y_t[1], F::from_u64(2), "M^T col 1 from rows 0..2");
    assert_eq!(y_t[2], F::ZERO, "M^T col 2 from rows 0..2");
}

// ---------------------------------------------------------------------------
// 12. append_zero_rows
// ---------------------------------------------------------------------------
#[test]
fn append_zero_rows() {
    let mut m = Mat::from_row_major(
        2,
        3,
        vec![
            F::ONE,
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
            F::from_u64(6),
        ],
    );

    m.append_zero_rows(2, F::ZERO);

    assert_eq!(m.rows(), 4, "should have 4 rows after appending 2");
    assert_eq!(m.cols(), 3, "cols unchanged");

    // Original data preserved.
    assert_eq!(m[(0, 0)], F::ONE);
    assert_eq!(m[(1, 2)], F::from_u64(6));

    // New rows are zero.
    for r in 2..4 {
        for c in 0..3 {
            assert_eq!(m[(r, c)], F::ZERO, "appended row {r}, col {c} should be zero");
        }
    }

    // Appending 0 rows is a noop.
    let rows_before = m.rows();
    m.append_zero_rows(0, F::ZERO);
    assert_eq!(m.rows(), rows_before);
}

// ---------------------------------------------------------------------------
// 13. sparse_cache_from_csc
// ---------------------------------------------------------------------------
#[test]
fn sparse_cache_from_csc() {
    use neo_ccs::SparseCache;

    // Build two CSC matrices: one real, one identity sentinel (None).
    let real = CscMat::from_dense_row_major(&Mat::from_row_major(
        2,
        2,
        vec![F::from_u64(3), F::ZERO, F::ZERO, F::from_u64(7)],
    ));

    let cache = SparseCache::from_csc(vec![
        Some(real), // matrix 0: real sparse
        None,       // matrix 1: identity sentinel
    ]);

    assert_eq!(cache.len(), 2);
    assert!(!cache.is_empty());

    // matrix 0: accessible as CSC
    let m0 = cache.csc(0);
    assert!(m0.is_some(), "matrix 0 should be Some");
    let m0 = m0.unwrap();
    let mut y = vec![F::ZERO; 2];
    m0.add_mul_into(&[F::ONE, F::ONE], &mut y, 2);
    assert_eq!(y[0], F::from_u64(3));
    assert_eq!(y[1], F::from_u64(7));

    // matrix 1: identity sentinel returns None
    assert!(cache.csc(1).is_none(), "identity sentinel should be None");

    // out of bounds: returns None
    assert!(cache.csc(5).is_none());
}

// ---------------------------------------------------------------------------
// 14. sparse_cache_from_triplets
// ---------------------------------------------------------------------------
#[test]
fn sparse_cache_from_triplets() {
    use neo_ccs::SparseCache;

    let matrices: Vec<Option<Vec<(usize, usize, F)>>> = vec![
        // matrix 0: 2x2 diagonal [5, 0; 0, 9]
        Some(vec![(0, 0, F::from_u64(5)), (1, 1, F::from_u64(9))]),
        // matrix 1: identity sentinel
        None,
        // matrix 2: single entry at (0, 1)
        Some(vec![(0, 1, F::from_u64(11))]),
    ];

    let cache = SparseCache::from_triplets(2, 2, matrices);

    assert_eq!(cache.len(), 3);

    // matrix 0: [5,0;0,9] * [1,1] = [5, 9]
    let m0 = cache.csc(0).unwrap();
    let mut y = vec![F::ZERO; 2];
    m0.add_mul_into(&[F::ONE, F::ONE], &mut y, 2);
    assert_eq!(y[0], F::from_u64(5));
    assert_eq!(y[1], F::from_u64(9));

    // matrix 1: sentinel
    assert!(cache.csc(1).is_none());

    // matrix 2: [0,11;0,0] * [1,1] = [11, 0]
    let m2 = cache.csc(2).unwrap();
    let mut y2 = vec![F::ZERO; 2];
    m2.add_mul_into(&[F::ONE, F::ONE], &mut y2, 2);
    assert_eq!(y2[0], F::from_u64(11));
    assert_eq!(y2[1], F::ZERO);
}

// ---------------------------------------------------------------------------
// 15. compact_seeded_phi81_matches_native_and_expanded_csc
// ---------------------------------------------------------------------------
#[test]
fn compact_seeded_phi81_matches_native_and_expanded_csc() {
    let seed = [0xA5; 32];
    let kappa = 2;
    let words = [0x0123_4567_89AB_CDEFu64, 0x0F0F_F0F0_55AA_AA55, 7];
    let word_starts = vec![1, 65, 129];
    let message_cols = (words.len() * 64).div_ceil(D);
    let (chunk_size, chunk_seeds) = neo_ajtai::seeded_pp_chunk_seeds(seed, kappa, message_cols);
    let row_start = 3;
    let rows = row_start + D * kappa + 2;
    let cols = 193;
    let block = SeededPhi81LinearBlock::new(
        row_start,
        word_starts.clone(),
        kappa,
        message_cols,
        chunk_size,
        chunk_seeds,
    )
    .expect("valid compact block");

    let base = CscMat::from_triplets(vec![(0, 0, F::from_u64(9))], rows, cols);
    let compact = CcsMatrix::csc_with_seeded_phi81(base, vec![block.clone()]).expect("compact matrix");
    let mut assignment = vec![F::ZERO; cols];
    assignment[0] = F::ONE;
    for (&word, &start) in words.iter().zip(&word_starts) {
        for bit in 0..64 {
            assignment[start + bit] = F::from_u64((word >> bit) & 1);
        }
    }

    let mut column_bits = vec![0u64; message_cols];
    for bit_index in 0..words.len() * 64 {
        let bit = (words[bit_index / 64] >> (bit_index % 64)) & 1;
        if bit == 1 {
            column_bits[bit_index % message_cols] |= 1u64 << (bit_index / message_cols);
        }
    }
    let native = neo_ajtai::commit_row_major_seeded_binary_cols(seed, D, kappa, message_cols, &column_bits);

    let mut compact_y = vec![F::ZERO; rows];
    compact.add_mul_into(&assignment, &mut compact_y, rows);
    assert_eq!(compact_y[0], F::from_u64(9));
    assert_eq!(&compact_y[row_start..row_start + D * kappa], native.data);

    let mut trips = vec![(0, 0, F::from_u64(9))];
    block.for_each_term::<F, _>(|row, col, coefficient| trips.push((row, col, coefficient)));
    let expanded = CcsMatrix::Csc(CscMat::from_triplets(trips, rows, cols));
    let mut expanded_y = vec![F::ZERO; rows];
    expanded.add_mul_into(&assignment, &mut expanded_y, rows);
    assert_eq!(compact_y, expanded_y);

    let row_weights: Vec<F> = (0..rows)
        .map(|row| F::from_u64((row as u64 * 17 + 3) % 97))
        .collect();
    let mut compact_t = vec![F::ZERO; cols];
    let mut expanded_t = vec![F::ZERO; cols];
    compact.add_mul_transpose_into(&row_weights, &mut compact_t, rows);
    expanded.add_mul_transpose_into(&row_weights, &mut expanded_t, rows);
    assert_eq!(compact_t, expanded_t);
}
