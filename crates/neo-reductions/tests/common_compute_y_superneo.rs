#![allow(non_snake_case)]

use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::utils::tensor_point;
use neo_ccs::{CcsStructure, Mat};
use neo_math::{D, F, K};
use neo_reductions::common::{
    compute_y_from_Z_and_r, decode_superneo_coeffs_from_witness_mat, validate_superneo_witness_mat,
};
use neo_reductions::superneo_eval::build_superneo_eval_cache;
use p3_field::PrimeCharacteristicRing;

fn dense_mat(rows: usize, cols: usize, seed: u64) -> Mat<F> {
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let v = if (r + 2 * c + 1) % 5 == 0 {
                F::from_u64(seed + (r as u64) * 17 + (c as u64) * 23 + 1)
            } else {
                F::ZERO
            };
            data.push(v);
        }
    }
    Mat::from_row_major(rows, cols, data)
}

fn make_z(seed: u64, m: usize) -> Mat<F> {
    let cols = m.div_ceil(D);
    let mut data = Vec::with_capacity(D * cols);
    for rho in 0..D {
        for blk in 0..cols {
            let c = blk * D + rho;
            if c < m {
                data.push(F::from_u64(seed + (rho as u64) * 13 + (c as u64) * 19 + 1));
            } else {
                data.push(F::ZERO);
            }
        }
    }
    Mat::from_row_major(D, cols, data)
}

fn manual_compute_y(s: &CcsStructure<F>, Z: &Mat<F>, r: &[K], ell_d: usize, b: u32) -> (Vec<Vec<K>>, Vec<K>) {
    let d_pad = 1usize << ell_d;
    let rb = tensor_point::<K>(r);
    let n_eff = core::cmp::min(s.n, rb.len());
    let cache = build_superneo_eval_cache(s).expect("expected SuperNeo cache");
    let z = decode_superneo_coeffs_from_witness_mat(Z, s.m).expect("decode packed coefficients");
    let mut y_ring: Vec<Vec<K>> = Vec::with_capacity(s.t());
    let y_raw = neo_reductions::superneo_eval::eval_all_mats_ring_cached(&cache, &z, &rb, n_eff);
    for coeffs in y_raw.into_iter().take(s.t()) {
        let mut row = coeffs.to_vec();
        if d_pad > row.len() {
            row.resize(d_pad, K::ZERO);
        }
        y_ring.push(row);
    }

    let params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs(s.n).expect("params");
    let mut params = params;
    params.b = b;
    let ct = neo_reductions::common::ct_from_y_ring_for_ccs_m(&y_ring, &params, s.m);
    (y_ring, ct)
}

#[test]
fn compute_y_from_Z_and_r_superneo_compatible_matches_manual() {
    let n = 16usize;
    let m = D; // SuperNeo-compatible width
    let s = CcsStructure::new(
        vec![dense_mat(n, m, 100), dense_mat(n, m, 200)],
        SparsePoly::new(
            2,
            vec![Term {
                coeff: F::ONE,
                exps: vec![1, 0],
            }],
        ),
    )
    .expect("valid CCS");

    assert!(
        build_superneo_eval_cache(&s).is_some(),
        "expected SuperNeo cache for compatible width"
    );

    let Z = make_z(300, m);
    let r = vec![
        K::from(F::from_u64(3)),
        K::from(F::from_u64(5)),
        K::from(F::from_u64(7)),
        K::from(F::from_u64(11)),
    ]; // n_pad = 16
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    let got = compute_y_from_Z_and_r(&s, &Z, &r, ell_d, 2);
    let want = manual_compute_y(&s, &Z, &r, ell_d, 2);
    assert_eq!(got, want);
}

#[test]
fn compute_y_from_Z_and_r_nondiv_width_uses_packed_layout_without_cache() {
    let n = 8usize;
    let m = D + 1; // non-divisible width uses packed ceil(m/D) layout.
    let s = CcsStructure::new(
        vec![dense_mat(n, m, 700), dense_mat(n, m, 900)],
        SparsePoly::new(
            2,
            vec![Term {
                coeff: F::ONE,
                exps: vec![1, 0],
            }],
        ),
    )
    .expect("valid CCS");

    assert!(
        build_superneo_eval_cache(&s).is_some(),
        "expected SuperNeo cache to support non-divisible packed width"
    );

    let Z = make_z(1200, m);
    validate_superneo_witness_mat(&Z, s.m).expect("packed layout must be accepted");

    let r = vec![
        K::from(F::from_u64(2)),
        K::from(F::from_u64(13)),
        K::from(F::from_u64(17)),
    ]; // n_pad = 8
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let (y_ring, ct) = compute_y_from_Z_and_r(&s, &Z, &r, ell_d, 2);
    assert_eq!(y_ring.len(), ct.len());
    for j in 0..y_ring.len() {
        assert_eq!(ct[j], y_ring[j][0], "ct must be constant term for j={j}");
    }
}
