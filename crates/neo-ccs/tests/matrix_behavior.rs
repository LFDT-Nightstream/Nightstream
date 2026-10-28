//! Direct behavior checks for dense and sparse matrix operations.

use neo_ccs::{CcsMatrix, CscMat, Mat};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

#[test]
fn row_major_layout() {
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

#[test]
fn identity_construction_and_detection() {
    for n in [1, 2, 3, 5, 8] {
        let id = Mat::<F>::identity(n);
        assert!(id.is_identity(), "Mat::identity({n}) should be detected as identity");
        assert!(
            id.is_identity_hint(),
            "Mat::identity({n}) should have identity hint set"
        );

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

#[test]
fn identity_hint_cleared_on_mut() {
    let mut id = Mat::<F>::identity(3);
    assert!(id.is_identity_hint(), "identity_hint should be set initially");

    id.set(0, 0, F::ONE);
    assert!(!id.is_identity_hint(), "identity_hint should be cleared after set()");

    assert!(id.is_identity(), "should still be structurally identity");
}

#[test]
fn csc_round_trip_mul() {
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
    csc.add_mul_into(&x, &mut y, 3);

    assert_eq!(y[0], F::from_u64(5), "M*x[0] should be 5");
    assert_eq!(y[1], F::from_u64(6), "M*x[1] should be 6");
    assert_eq!(y[2], F::from_u64(19), "M*x[2] should be 19");

    let mut y_dense = vec![F::ZERO; 3];
    for i in 0..3 {
        for j in 0..3 {
            y_dense[i] += m[(i, j)] * x[j];
        }
    }
    assert_eq!(y, y_dense, "CSC mul should match dense mul");
}

#[test]
fn csc_from_triplets_dedup() {
    let triplets = vec![(0, 0, F::from_u64(3)), (0, 0, F::from_u64(7)), (1, 1, F::from_u64(5))];

    let csc = CscMat::from_triplets(triplets, 2, 2);

    let x = vec![F::ONE, F::ONE];
    let mut y = vec![F::ZERO; 2];
    csc.add_mul_into(&x, &mut y, 2);

    assert_eq!(y[0], F::from_u64(10), "duplicate entries should be summed");
    assert_eq!(y[1], F::from_u64(5));
}

#[test]
fn csc_transpose_mul() {
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

#[test]
fn ccs_matrix_identity_mul() {
    let n = 4;
    let id = CcsMatrix::<F>::Identity { n };

    let x: Vec<F> = (1..=4).map(F::from_u64).collect();

    let mut y = vec![F::ZERO; n];
    id.add_mul_into(&x, &mut y, n);
    assert_eq!(y, x, "Identity mul should return x");

    let mut y2 = vec![F::ZERO; n];
    id.add_mul_transpose_into(&x, &mut y2, n);
    assert_eq!(y2, x, "Identity transpose mul should return x");
}

#[test]
fn is_column_selector_detection() {
    let m = Mat::from_row_major(3, 2, vec![F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE]);
    assert!(m.is_column_selector(), "valid column selector should be detected");

    let id = Mat::<F>::identity(3);
    assert!(id.is_column_selector(), "identity should be a column selector");

    let not_sel = Mat::from_row_major(2, 2, vec![F::ONE, F::ZERO, F::ONE, F::ONE]);
    assert!(
        !not_sel.is_column_selector(),
        "matrix with two 1s in a column should not be a column selector"
    );

    let not_sel2 = Mat::from_row_major(2, 1, vec![F::from_u64(2), F::ZERO]);
    assert!(
        !not_sel2.is_column_selector(),
        "matrix with entry != 0 or 1 should not be a column selector"
    );
}

#[test]
fn csr_round_trip() {
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

    assert_eq!(csr.rows, 3);
    assert_eq!(csr.cols, 4);

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
    for (i, &vi) in v_im.iter().enumerate() {
        assert_eq!(vi, Fq::ZERO, "v_im[{i}] should be zero");
    }
}

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
            Fq::ZERO,
            Fq::from_u64(4u64),
            Fq::ZERO,
            Fq::from_u64(5u64),
        ],
    );

    let csr = m.to_csr();

    let (cols, vals) = csr.row_nz(0);
    assert_eq!(cols, &[0, 2]);
    assert_eq!(vals, &[Fq::from_u64(2u64), Fq::ONE]);
    assert_eq!(csr.row_nnz(0), 2);

    let (cols1, vals1) = csr.row_nz(1);
    assert!(cols1.is_empty());
    assert!(vals1.is_empty());
    assert_eq!(csr.row_nnz(1), 0);

    assert_eq!(csr.row_nnz(2), 2);

    assert_eq!(csr.nnz(), 4);

    assert_eq!(m.nnz(), 4);
    assert_eq!(m.row_nnz(0), 2);
    assert_eq!(m.row_nnz(1), 0);
    assert_eq!(m.row_nnz(2), 2);

    let dense_nz: Vec<(usize, &Fq)> = m.row_nz(0).collect();
    assert_eq!(dense_nz.len(), 2);
    assert_eq!(dense_nz[0].0, 0);
    assert_eq!(dense_nz[1].0, 2);
}

#[test]
fn n_eff_limits_rows() {
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

    let mut y_full = vec![F::ZERO; 4];
    csc.add_mul_into(&x, &mut y_full, 4);
    assert_eq!(y_full[0], F::ONE);
    assert_eq!(y_full[1], F::from_u64(2));
    assert_eq!(y_full[2], F::from_u64(3));
    assert_eq!(y_full[3], F::from_u64(12));

    let mut y_limited = vec![F::ZERO; 4];
    csc.add_mul_into(&x, &mut y_limited, 2);
    assert_eq!(y_limited[0], F::ONE, "row 0 should be computed");
    assert_eq!(y_limited[1], F::from_u64(2), "row 1 should be computed");
    assert_eq!(y_limited[2], F::ZERO, "row 2 should be untouched");
    assert_eq!(y_limited[3], F::ZERO, "row 3 should be untouched");

    let x4 = vec![F::ONE, F::ONE, F::from_u64(99), F::from_u64(99)];
    let mut y_t = vec![F::ZERO; 3];
    csc.add_mul_transpose_into(&x4, &mut y_t, 2);
    assert_eq!(y_t[0], F::ONE, "M^T col 0 from rows 0..2");
    assert_eq!(y_t[1], F::from_u64(2), "M^T col 1 from rows 0..2");
    assert_eq!(y_t[2], F::ZERO, "M^T col 2 from rows 0..2");
}

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

    assert_eq!(m[(0, 0)], F::ONE);
    assert_eq!(m[(1, 2)], F::from_u64(6));

    for r in 2..4 {
        for c in 0..3 {
            assert_eq!(m[(r, c)], F::ZERO, "appended row {r}, col {c} should be zero");
        }
    }

    let rows_before = m.rows();
    m.append_zero_rows(0, F::ZERO);
    assert_eq!(m.rows(), rows_before);
}

#[test]
fn matrix_storage_forms_round_trip_through_serde() {
    let dense = Mat::from_row_major(2, 2, vec![F::ZERO, F::ONE, F::from_u64(2), F::from_u64(3)]);
    let constant = Mat::virtual_constant(2, 3, F::from_u64(7));
    let packed = Mat::compact_signed_unit_from_column_masks(2, 2, &[0, 1], &[2, 0]).expect("valid signed-unit masks");

    for matrix in [&dense, &constant, &packed] {
        let encoded = serde_json::to_vec(matrix).expect("serialize matrix");
        let decoded: Mat<F> = serde_json::from_slice(&encoded).expect("deserialize canonical matrix");
        assert_eq!(&decoded, matrix);
    }
}
