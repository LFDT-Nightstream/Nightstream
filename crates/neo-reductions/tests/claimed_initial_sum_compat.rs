#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{poly::SparsePoly, poly::Term, CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_reductions::{optimized_engine, paper_exact_engine, Challenges};
use p3_field::PrimeCharacteristicRing;

#[test]
fn claimed_initial_sum_optimized_matches_paper_exact() {
    // Small CCS: n=4 (power of two), m=3, t=1, f(y)=y0 (linear)
    let n = 4usize;
    let m = 3usize;
    let m0 = Mat::from_row_major(
        n,
        m,
        vec![
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ONE,
            F::ONE,
        ],
    );
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    let s = CcsStructure::new(vec![m0], f).unwrap();

    let d_pad = D.next_power_of_two();
    let mut y0 = vec![K::ZERO; d_pad];
    for rho in 0..D {
        y0[rho] = K::from(F::from_u64((rho as u64) + 1));
    }

    let me_inputs: Vec<CeClaim<Cmt, F, K>> = vec![CeClaim {
        adv: None,
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: Cmt::zeros(D, 1),
        X: Mat::zero(D, 1, F::ZERO),
        r: vec![K::from(F::from_u64(3)), K::from(F::from_u64(5))],
        s_col: vec![],
        y_ring: vec![y0],
        ct: vec![K::ZERO],
        aux_openings: Vec::new(),
        y_zcol: vec![],
        m_in: 1,
        fold_digest: [0u8; 32],
    }];

    let ch = Challenges {
        alpha: (0..6)
            .map(|i| K::from(F::from_u64((i as u64) + 7)))
            .collect(),
        beta_a: vec![],
        beta_r: vec![],
        beta_m: vec![],
        gamma: K::from(F::from_u64(11)),
    };

    let t_opt = optimized_engine::claimed_initial_sum_from_inputs_with_k_mcs(&s, &ch, 1, &me_inputs);
    let t_paper = paper_exact_engine::claimed_initial_sum_from_inputs_with_k_mcs(&s, &ch, 1, &me_inputs);
    assert_eq!(t_opt, t_paper);
}
