//! Capacity test: build an augmented CCS with a very large `y_len` and
//! ensure the builder succeeds without a hard D-row cap.
//!
//! EV rows are 2*y_len. With y_len=500 this implies 1000 EV rows
//! before any step rows or binders. Since the augmented CCS no longer
//! enforces a hard D=54 cap, this should succeed. A future EV
//! aggregation can further reduce EV rows to 2, but is not required
//! for this test to pass.

use neo::{F};
use p3_field::PrimeCharacteristicRing;
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use neo_ccs::traits::SModuleHomomorphism; // bring trait into scope for DummySModule methods

/// Build a trivial CCS with `rows` identical R1CS constraints of 1*1=1
/// and `m = 1 + y_len` columns where column 0 is the const-1 witness.
fn trivial_step_ccs_rows(y_len: usize, rows: usize) -> CcsStructure<F> {
    let m = 1 + y_len; // const-1 + y_step slots
    let n = rows.max(1);
    let mut a = vec![F::ZERO; n * m];
    let mut b = vec![F::ZERO; n * m];
    let mut c = vec![F::ZERO; n * m];
    for r in 0..n { a[r * m] = F::ONE; b[r * m] = F::ONE; c[r * m] = F::ONE; }
    let a_mat = Mat::from_row_major(n, m, a);
    let b_mat = Mat::from_row_major(n, m, b);
    let c_mat = Mat::from_row_major(n, m, c);
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

#[test]
fn large_y_len_augmented_ccs_should_eventually_fit_after_ev_aggregation() {
    // Set a very large y_len so that current EV rows = 2*y_len >> D.
    // "Equivalent to 1000 rows" today means y_len=500 ⇒ 2*y_len=1000 EV rows.
    let y_len = 500usize;

    // Keep the step computation tiny (1 row) so EV dominates the row budget.
    let step_ccs = trivial_step_ccs_rows(y_len, 1);

    // Binding: put y_step in witness indices [1..=y_len]; const1 at 0.
    let y_step_offsets: Vec<usize> = (1..=y_len).collect();
    let y_prev_witness_indices: Vec<usize> = vec![]; // no y_prev binder for this test
    let app_input_witness_indices: Vec<usize> = vec![]; // no app-input binder
    let const1_idx = 0usize;
    let step_x_len = 0usize; // empty step_x for this test

    // Attempt to build the augmented CCS. With the D-cap removed, this should succeed
    // even though EV rows dominate. (Future EV aggregation reduces EV to 2 rows.)
    let augmented = neo::build_augmented_ccs_linked_with_rlc(
        &step_ccs,
        step_x_len,
        &y_step_offsets,
        &y_prev_witness_indices,
        &app_input_witness_indices,
        y_len,
        const1_idx,
        None,
    );

    assert!(
        augmented.is_ok(),
        "Expected large-y_len augmented CCS to build after EV aggregation; got error: {:?}",
        augmented.err()
    );
}

/// Dummy S-module for ME checks in tests: commits as sum(Z) and projects first m_in columns
struct DummySModule;

impl neo_ccs::traits::SModuleHomomorphism<F, F> for DummySModule {
    fn commit(&self, z: &neo_ccs::Mat<F>) -> F {
        z.as_slice().iter().copied().fold(F::ZERO, |acc, v| acc + v)
    }
    fn project_x(&self, z: &neo_ccs::Mat<F>, m_in: usize) -> neo_ccs::Mat<F> {
        let rows = z.rows();
        let cols = m_in.min(z.cols());
        if cols == 0 { return neo_ccs::Mat::from_row_major(rows, 0, Vec::new()); }
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let row = z.row(r);
            data.extend_from_slice(&row[..cols]);
        }
        neo_ccs::Mat::from_row_major(rows, cols, data)
    }
}

/// Helper: compute y = Z * (M^T * χ_r) using the same semantics as check_me_consistency
fn compute_y<FK: p3_field::Field + From<F>>(z: &neo_ccs::Mat<F>, m_mat: &neo_ccs::Mat<F>, n_rows: usize, r: &[FK]) -> Vec<FK> {
    // χ_r over length 2^ell (implicit); consume first n entries only
    let rb = neo_ccs::utils::tensor_point::<FK>(r);
    // v = M^T * χ_r  (size m)
    let mut v = vec![FK::ZERO; m_mat.cols()];
    for rix in 0..n_rows {
        let wr = rb[rix];
        let row = m_mat.row(rix);
        for c in 0..m_mat.cols() {
            v[c] += FK::from(row[c]) * wr;
        }
    }
    // y = Z * v  (size d)
    let mut y = vec![FK::ZERO; z.rows()];
    for drow in 0..z.rows() {
        let row = z.row(drow);
        let mut acc = FK::ZERO;
        for c in 0..z.cols() { acc += FK::from(row[c]) * v[c]; }
        y[drow] = acc;
    }
    y
}

#[test]
fn non_power_of_two_n_me_smoke_test() {
    // Build CCS with n=6, m=3, t=1
    let n = 6usize; let m = 3usize; let t = 1usize;
    let mut a = vec![F::ZERO; n * m];
    // Simple pattern: each row selects one column (r mod m)
    for r in 0..n { a[r * m + (r % m)] = F::ONE; }
    let m0 = Mat::from_row_major(n, m, a);
    let f = neo_ccs::SparsePoly::new(t, Vec::new());
    let s = CcsStructure::new(vec![m0.clone()], f).unwrap();

    // Z: choose small d=2 for speed
    let d = 2usize;
    let mut z_data = vec![F::ZERO; d * m];
    // Z = [1 2 3; 4 5 6] in row-major
    z_data[0] = F::ONE; z_data[1] = F::from_u64(2); z_data[2] = F::from_u64(3);
    z_data[3] = F::from_u64(4); z_data[4] = F::from_u64(5); z_data[5] = F::from_u64(6);
    let z_mat = Mat::from_row_major(d, m, z_data);

    // r: len 3 since next_power_of_two(6) = 8 ⇒ ell=3
    let r: Vec<neo_math::K> = [1u64, 2, 3].iter().map(|&u| neo_math::K::from(F::from_u64(u))).collect();
    let y = compute_y::<neo_math::K>(&z_mat, &m0, n, &r);

    // Build ME instance with dummy commitment and X projection
    let l = DummySModule;
    let c = l.commit(&z_mat);
    let x_mat = l.project_x(&z_mat, 0);
    let inst = neo_ccs::MeInstance::<F, F, neo_math::K> {
        c,
        X: x_mat,
        r: r.clone(),
        y: vec![y.clone()],
        y_scalars: vec![neo_math::K::ZERO],
        m_in: 0,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    };
    let wit = neo_ccs::MeWitness { Z: z_mat.clone() };

    neo_ccs::relations::check_me_consistency(&s, &l, &inst, &wit).expect("ME check should pass for n=6 with ell=3");
}

#[test]
fn roundtrip_vs_padded_power_of_two() {
    // Same CCS data, once at n=6 and once padded to n=8 with zero rows
    let n6 = 6usize; let n8 = 8usize; let m = 3usize;
    let mut a6 = vec![F::ZERO; n6 * m];
    for r in 0..n6 { a6[r * m + (r % m)] = F::ONE; }
    let m6 = Mat::from_row_major(n6, m, a6);

    let mut a8 = vec![F::ZERO; n8 * m];
    for r in 0..n6 { a8[r * m + (r % m)] = F::ONE; }
    let m8 = Mat::from_row_major(n8, m, a8);

    let f6 = neo_ccs::SparsePoly::new(1, Vec::new());
    let f8 = neo_ccs::SparsePoly::new(1, Vec::new());
    let s6 = CcsStructure::new(vec![m6.clone()], f6).unwrap();
    let s8 = CcsStructure::new(vec![m8.clone()], f8).unwrap();

    // Z small d=2
    let d = 2usize;
    let z_mat = Mat::from_row_major(d, m, vec![F::ONE, F::from_u64(2), F::from_u64(3), F::from_u64(4), F::from_u64(5), F::from_u64(6)]);

    let r: Vec<neo_math::K> = [1u64, 2, 3].iter().map(|&u| neo_math::K::from(F::from_u64(u))).collect();
    let y6 = compute_y::<neo_math::K>(&z_mat, &m6, n6, &r);
    let y8 = compute_y::<neo_math::K>(&z_mat, &m8, n8, &r);
    assert_eq!(y6, y8, "Padding zero rows should not change y");

    // Sanity via ME checker for both variants
    let l = DummySModule;
    let c = l.commit(&z_mat);
    let x_mat = l.project_x(&z_mat, 0);
    let base_inst = neo_ccs::MeInstance::<F, F, neo_math::K> {
        c,
        X: x_mat,
        r: r.clone(),
        y: vec![y6.clone()],
        y_scalars: vec![neo_math::K::ZERO],
        m_in: 0,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    };
    let wit = neo_ccs::MeWitness { Z: z_mat.clone() };
    neo_ccs::relations::check_me_consistency(&s6, &l, &base_inst, &wit).expect("n=6 should pass");
    neo_ccs::relations::check_me_consistency(&s8, &l, &base_inst, &wit).expect("n=8 padded should pass");
}

#[test]
fn augmented_ccs_padding_regression() {
    // total_rows = step_rows (1) + ev_rows (2*y_len=32) + const1_enforce_rows (1) = 34 ⇒ target_rows should be 64
    // NOTE: const1_enforce_rows added by soundness fix (w_const1 * ρ = ρ constraint)
    let y_len = 16usize;
    let step_ccs = trivial_step_ccs_rows(y_len, 1);

    let y_step_offsets: Vec<usize> = (1..=y_len).collect();
    let y_prev_witness_indices: Vec<usize> = vec![];
    let app_input_witness_indices: Vec<usize> = vec![];
    let const1_idx = 0usize; // first witness column treated as constant 1 in tests
    let step_x_len = 0usize;

    let augmented = neo::build_augmented_ccs_linked_with_rlc(
        &step_ccs,
        step_x_len,
        &y_step_offsets,
        &y_prev_witness_indices,
        &app_input_witness_indices,
        y_len,
        const1_idx,
        None,
    ).expect("augmented CCS should build");

    assert_eq!(augmented.n, 64, "target_rows must be next power-of-two (34 → 64)");

    // Verify rows [34..64) are all zeros in each matrix (row 33 is the const-1 enforcement row)
    let used = 34usize;
    for (j, mtx) in augmented.matrices.iter().enumerate() {
        for r in used..augmented.n {
            let row = mtx.row(r);
            assert!(row.iter().all(|&v| v == F::ZERO), "matrix {} has non-zero in padded row {}", j, r);
        }
    }
}
