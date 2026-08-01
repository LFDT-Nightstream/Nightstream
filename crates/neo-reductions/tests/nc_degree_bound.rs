#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, CcsWitness, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::optimized_engine::legacy_split_nc::oracle::NcOracle;
use neo_reductions::optimized_engine::Challenges;
use neo_reductions::sumcheck::{interpolate_from_evals, poly_eval_k, RoundOracle};
use p3_field::PrimeCharacteristicRing;

fn identity_left(n: usize, m: usize) -> Mat<F> {
    let mut mat = Mat::zero(n, m, F::ZERO);
    for i in 0..n.min(m) {
        mat.set(i, i, F::ONE);
    }
    mat
}

#[test]
fn nc_oracle_round_polys_respect_degree_bound() {
    // Use a deliberately invalid witness (out-of-range digits) so that the NC polynomial
    // is generically non-zero and exercises the full `range_product` degree.
    let n = D;
    let m = D;

    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let s = CcsStructure::new(vec![identity_left(n, m)], SparsePoly::new(1, vec![])).expect("ccs");
    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &s).expect("dims");

    // Vary digits across the table so the MLE isn't constant.
    let packed_cols = m / D;
    let mut data = Vec::with_capacity(D * packed_cols);
    for rho in 0..D {
        for blk in 0..packed_cols {
            let c = blk * D + rho;
            data.push(F::from_u64(5 + (rho as u64) * 17 + (c as u64) * 31));
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

    let deg = oracle.degree_bound();
    let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
    let x_extra_0 = K::from(F::from_u64((deg as u64) + 1));
    let x_extra_1 = K::from(F::from_u64((deg as u64) + 2));

    for round_idx in 0..oracle.num_rounds() {
        let ys = oracle.evals_at(&xs);
        let coeffs = interpolate_from_evals(&xs, &ys);

        let ys_extra = oracle.evals_at(&[x_extra_0, x_extra_1]);
        assert_eq!(
            poly_eval_k(&coeffs, x_extra_0),
            ys_extra[0],
            "NC round {} violates declared degree bound at x=deg+1",
            round_idx
        );
        assert_eq!(
            poly_eval_k(&coeffs, x_extra_1),
            ys_extra[1],
            "NC round {} violates declared degree bound at x=deg+2",
            round_idx
        );

        oracle.fold(K::from(F::from_u64(1000 + round_idx as u64)));
    }
}
