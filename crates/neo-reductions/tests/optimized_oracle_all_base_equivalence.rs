#![allow(non_snake_case)]

use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::{CcsStructure, Mat, McsWitness};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::oracle::{OptimizedOracle, SparseCache};
use neo_reductions::Challenges;
use p3_field::PrimeField64;
use p3_field::PrimeCharacteristicRing;
use std::sync::Arc;

fn build_params_for_b(b: u32, m: usize) -> NeoParams {
    let q: u64 = <F as PrimeField64>::ORDER_U64;
    let eta: u32 = neo_math::ETA as u32;
    let d: u32 = neo_math::D as u32;
    let kappa: u32 = 2;
    let m_u: u64 = m as u64;
    let k_rho: u32 = 12;
    let T: u32 = 216;
    let s: u32 = 2;
    let lambda: u32 = 96;
    NeoParams::new(q, eta, d, kappa, m_u, b, k_rho, T, s, lambda).expect("params")
}

fn build_oracle(b: u32) -> OptimizedOracle<'static, F> {
    // Small square CCS with M0 = I to keep row-stream indexing valid (Zi[(rho,row)]).
    let n = 8usize;
    let mat = Mat::identity(n);
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let s = Box::leak(Box::new(CcsStructure::new(vec![mat], f).expect("CCS")));

    let params = Box::leak(Box::new(build_params_for_b(b, n)));

    // One witness (MCS) is enough to exercise row-phase logic.
    let mut data = Vec::with_capacity(D * n);
    for rho in 0..D {
        for c in 0..n {
            let x = (rho as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                ^ 0x94D0_49BB_1331_11EB;
            data.push(F::from_u64(x));
        }
    }
    let Z = Mat::from_row_major(D, n, data);
    let mcs_wit = Box::leak(Box::new(McsWitness { w: vec![F::ZERO; n], Z }));

    let ell_n = 3usize; // n_pad = 8
    let ell_d = 6usize; // d_pad = 64 >= D
    let d_sc = 5usize;

    let ch = Challenges {
        alpha: (0..ell_d).map(|i| K::from(F::from_u64((i as u64) + 1))).collect(),
        beta_a: (0..ell_d).map(|i| K::from(F::from_u64((i as u64) + 11))).collect(),
        beta_r: (0..ell_n).map(|i| K::from(F::from_u64((i as u64) + 21))).collect(),
        gamma: K::from(F::from_u64(7)),
    };

    let sparse = Arc::new(SparseCache::build(s));
    OptimizedOracle::new_with_sparse(
        s,
        params,
        core::slice::from_ref(mcs_wit),
        &[],
        ch,
        ell_d,
        ell_n,
        d_sc,
        None,
        sparse,
    )
}

#[test]
fn optimized_oracle_all_base_matches_generic_b2() {
    let oracle = build_oracle(2);
    assert!(oracle.__test_row_stream_all_base(), "expected all_base=true");

    let xs: Vec<K> = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
    ];
    let (base, generic) = oracle.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(base, generic);
}

#[test]
fn optimized_oracle_all_base_matches_generic_b3() {
    let oracle = build_oracle(3);
    assert!(oracle.__test_row_stream_all_base(), "expected all_base=true");

    let xs: Vec<K> = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
    ];
    let (base, generic) = oracle.__test_row_phase_base_vs_generic(&xs);
    assert_eq!(base, generic);
}
