#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, CcsWitness, Mat, SparsePoly};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::build_dims_and_policy;
use neo_reductions::optimized_engine::oracle::NcOracle;
use neo_reductions::optimized_engine::Challenges;
use neo_reductions::sumcheck::{interpolate_from_evals, RoundOracle};
use p3_field::PrimeCharacteristicRing;

#[inline]
fn k_cplx(re: u64, im: u64) -> K {
    K::from_coeffs([F::from_u64(re), F::from_u64(im)])
}

fn identity_left(n: usize, m: usize) -> Mat<F> {
    let mut mat = Mat::zero(n, m, F::ZERO);
    for i in 0..n.min(m) {
        mat.set(i, i, F::ONE);
    }
    mat
}

fn run_fast_vs_generic(b: u32) {
    let n = D;
    let m = D;

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.b = b;

    let s = CcsStructure::new(vec![identity_left(n, m)], SparsePoly::new(1, vec![])).expect("ccs");
    let dims = build_dims_and_policy(&params, &s).expect("dims");

    let packed_cols = m / D;
    let mut data = Vec::with_capacity(D * packed_cols);
    for rho in 0..D {
        for blk in 0..packed_cols {
            let c = blk * D + rho;
            data.push(F::from_u64(7 + (rho as u64) * 19 + (c as u64) * 23));
        }
    }
    let Z = Mat::from_row_major(D, packed_cols, data);
    let mcs_witnesses = vec![CcsWitness { w: vec![F::ZERO; m], Z }];

    let ch = Challenges {
        alpha: (0..dims.ell_d)
            .map(|i| K::from(F::from_u64(100 + i as u64)))
            .collect(),
        beta_a: (0..dims.ell_d)
            .map(|i| K::from(F::from_u64(200 + i as u64)))
            .collect(),
        beta_r: (0..dims.ell_n)
            .map(|i| K::from(F::from_u64(300 + i as u64)))
            .collect(),
        beta_m: (0..dims.ell_m)
            .map(|i| K::from(F::from_u64(400 + i as u64)))
            .collect(),
        gamma: K::from(F::from_u64(777)),
    };

    let mut oracle = NcOracle::new(&s, &params, &mcs_witnesses, &[], ch, dims.ell_d, dims.ell_m, dims.d_sc);
    let xs = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
        K::from(F::from_u64(9)),
    ];

    for round in 0..dims.ell_m {
        let (fast, generic) = oracle
            .__test_col_phase_fast_vs_generic(&xs)
            .expect("must be in NC column phase");
        assert_eq!(
            fast, generic,
            "NcOracle fast col-phase mismatch at b={b}, round={round}"
        );
        oracle.fold(K::from(F::from_u64(900 + round as u64)));
    }
}

fn run_direct_coeffs_match_interpolated_generic(b: u32) {
    let n = D;
    let m = D;

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.b = b;

    let s = CcsStructure::new(vec![identity_left(n, m)], SparsePoly::new(1, vec![])).expect("ccs");
    let dims = build_dims_and_policy(&params, &s).expect("dims");

    let packed_cols = m / D;
    let mut data = Vec::with_capacity(D * packed_cols);
    for rho in 0..D {
        for blk in 0..packed_cols {
            let c = blk * D + rho;
            data.push(F::from_u64(11 + (rho as u64) * 13 + (c as u64) * 29));
        }
    }
    let Z = Mat::from_row_major(D, packed_cols, data);
    let mcs_witnesses = vec![CcsWitness { w: vec![F::ZERO; m], Z }];

    let ch = Challenges {
        alpha: (0..dims.ell_d)
            .map(|i| K::from(F::from_u64(123 + i as u64)))
            .collect(),
        beta_a: (0..dims.ell_d)
            .map(|i| K::from(F::from_u64(223 + i as u64)))
            .collect(),
        beta_r: (0..dims.ell_n)
            .map(|i| K::from(F::from_u64(323 + i as u64)))
            .collect(),
        beta_m: (0..dims.ell_m)
            .map(|i| K::from(F::from_u64(423 + i as u64)))
            .collect(),
        gamma: K::from(F::from_u64(877)),
    };

    let mut oracle = NcOracle::new(&s, &params, &mcs_witnesses, &[], ch, dims.ell_d, dims.ell_m, dims.d_sc);

    for round in 0..dims.ell_m {
        let direct = oracle
            .optimized_col_phase_round_coeffs()
            .expect("must be in NC column phase");
        let deg = oracle.degree_bound();
        let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
        let generic = interpolate_from_evals(&xs, &oracle.evals_at(&xs));
        assert_eq!(
            direct, generic,
            "NcOracle direct col-phase coeff mismatch at b={b}, round={round}"
        );
        oracle.fold(K::from(F::from_u64(1700 + round as u64)));
    }
}

#[test]
fn nc_col_phase_fast_path_matches_generic_b2() {
    run_fast_vs_generic(2);
}

#[test]
fn nc_col_phase_fast_path_matches_generic_b3() {
    run_fast_vs_generic(3);
}

#[test]
fn nc_col_phase_direct_coeffs_match_interpolated_generic_b2() {
    run_direct_coeffs_match_interpolated_generic(2);
}

#[test]
fn nc_col_phase_direct_coeffs_match_interpolated_generic_b3() {
    run_direct_coeffs_match_interpolated_generic(3);
}

/// Like `run_fast_vs_generic(2)`, but every transcript challenge has nonzero
/// imaginary part. This forces `digit_tables_all_real` to flip to `false`
/// after the first fold and exercises the K branch of
/// `accumulate_inner_b2_at`, which the base-field-only variant of the
/// test never reaches.
///
/// Hardcoded to `b = 2`: that is the SuperNeo Goldilocks profile this
/// performance work targets, and this test covers the optimized b=2 K path.
#[test]
fn nc_col_phase_fast_path_matches_generic_b2_k_complex() {
    let n = D;
    let m = D;

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.b = 2;

    let s = CcsStructure::new(vec![identity_left(n, m)], SparsePoly::new(1, vec![])).expect("ccs");
    let dims = build_dims_and_policy(&params, &s).expect("dims");

    let packed_cols = m / D;
    let mut data = Vec::with_capacity(D * packed_cols);
    for rho in 0..D {
        for blk in 0..packed_cols {
            let c = blk * D + rho;
            data.push(F::from_u64(7 + (rho as u64) * 19 + (c as u64) * 23));
        }
    }
    let Z = Mat::from_row_major(D, packed_cols, data);
    let mcs_witnesses = vec![CcsWitness { w: vec![F::ZERO; m], Z }];

    let ch = Challenges {
        alpha: (0..dims.ell_d)
            .map(|i| k_cplx(100 + i as u64, 11 + i as u64))
            .collect(),
        beta_a: (0..dims.ell_d)
            .map(|i| k_cplx(200 + i as u64, 17 + i as u64))
            .collect(),
        beta_r: (0..dims.ell_n)
            .map(|i| k_cplx(300 + i as u64, 19 + i as u64))
            .collect(),
        beta_m: (0..dims.ell_m)
            .map(|i| k_cplx(400 + i as u64, 23 + i as u64))
            .collect(),
        gamma: k_cplx(777, 29),
    };

    let mut oracle = NcOracle::new(&s, &params, &mcs_witnesses, &[], ch, dims.ell_d, dims.ell_m, dims.d_sc);
    let xs = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        k_cplx(2, 1),
        k_cplx(5, 3),
        k_cplx(9, 7),
    ];

    for round in 0..dims.ell_m {
        let (fast, generic) = oracle
            .__test_col_phase_fast_vs_generic(&xs)
            .expect("must be in NC column phase");
        assert_eq!(
            fast, generic,
            "NcOracle fast col-phase mismatch (K-complex, b=2) at round={round}"
        );
        // Fold with a K-complex challenge so digit_tables_all_real flips
        // to false after round 0.
        oracle.fold(k_cplx(900 + round as u64, 31 + round as u64));
    }
}
