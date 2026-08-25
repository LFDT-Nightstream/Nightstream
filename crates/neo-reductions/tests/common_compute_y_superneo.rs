#![allow(non_snake_case)]

use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::utils::tensor_point;
use neo_ccs::{CcsStructure, Mat, V1_1Evaluations};
use neo_math::{superneo_bar_block, Fq, KExtensions, Rq, D, F, K};
use neo_reductions::common::{
    compute_v1_1_evaluations_from_z_and_r, decode_superneo_coeffs_from_witness_mat, validate_superneo_witness_mat,
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

fn manual_compute_v1_1_evaluations(s: &CcsStructure<F>, Z: &Mat<F>, r: &[K], ell_d: usize) -> V1_1Evaluations<K> {
    let d_pad = 1usize << ell_d;
    let rb = tensor_point::<K>(r);
    let n_eff = core::cmp::min(s.n, rb.len());
    let cache = build_superneo_eval_cache(s).expect("expected SuperNeo cache");
    let z = decode_superneo_coeffs_from_witness_mat(Z, s.m).expect("decode packed coefficients");
    let mut identity = [K::ZERO; D];
    for (row, &weight) in rb.iter().take(z.len()).enumerate() {
        let block = row / D;
        let mut basis = [Fq::ZERO; D];
        basis[row % D] = Fq::ONE;
        let transformed = Rq(superneo_bar_block(basis));
        let mut real = [Fq::ZERO; D];
        let mut imaginary = [Fq::ZERO; D];
        for lane in 0..D {
            let [low, high] = z[block * D + lane].as_coeffs();
            real[lane] = low;
            imaginary[lane] = high;
        }
        let real_product = transformed.mul(&Rq(real));
        let imaginary_product = transformed.mul(&Rq(imaginary));
        for coefficient in 0..D {
            identity[coefficient] +=
                weight * K::from_coeffs([real_product.0[coefficient], imaginary_product.0[coefficient]]);
        }
    }
    let mut eval_k = identity.to_vec();
    eval_k.resize(d_pad, K::ZERO);

    let y_raw = neo_reductions::superneo_eval::eval_all_mats_ring_cached(&cache, &z, &rb, n_eff);
    let mut eval_a = Vec::with_capacity(s.t());
    for coeffs in y_raw.into_iter().take(s.t()) {
        let mut row = coeffs.to_vec();
        if d_pad > row.len() {
            row.resize(d_pad, K::ZERO);
        }
        eval_a.push(row);
    }
    V1_1Evaluations { eval_k, eval_a }
}

#[test]
fn compute_v1_1_evaluations_superneo_compatible_match_manual() {
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

    let got = compute_v1_1_evaluations_from_z_and_r(&s, &Z, &r, ell_d);
    let want = manual_compute_v1_1_evaluations(&s, &Z, &r, ell_d);
    assert_eq!(got, want);
}

#[test]
fn compute_v1_1_evaluations_nondiv_width_use_packed_layout() {
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
    let evaluations = compute_v1_1_evaluations_from_z_and_r(&s, &Z, &r, ell_d);
    let d_pad = D.next_power_of_two();
    assert_eq!(evaluations.eval_k.len(), d_pad);
    assert_eq!(evaluations.eval_a.len(), s.t());
    for family in std::iter::once(&evaluations.eval_k).chain(&evaluations.eval_a) {
        assert_eq!(
            family.len(),
            d_pad,
            "each v1_1 evaluation family must use the padded ring width"
        );
        assert!(family[D..].iter().all(|value| *value == K::ZERO));
    }
}
