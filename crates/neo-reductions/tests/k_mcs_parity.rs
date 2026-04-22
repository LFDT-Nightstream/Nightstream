#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{poly::SparsePoly, poly::Term, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::{optimized_engine, paper_exact_engine, Challenges};
use p3_field::PrimeCharacteristicRing;

fn k(v: u64) -> K {
    K::from(F::from_u64(v))
}

fn vec_k(start: u64, len: usize) -> Vec<K> {
    (0..len).map(|i| k(start + i as u64)).collect()
}

fn dense_mat(rows: usize, cols: usize, seed: u64) -> Mat<F> {
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let v = if (r + 2 * c) % 3 == 0 {
                F::from_u64(seed + (r as u64) * 17 + (c as u64) * 11 + 1)
            } else {
                F::ZERO
            };
            data.push(v);
        }
    }
    Mat::from_row_major(rows, cols, data)
}

fn build_structure(n: usize, m: usize) -> CcsStructure<F> {
    let matrices = vec![Mat::<F>::identity(n), dense_mat(n, m, 10), dense_mat(n, m, 20)];
    let f = SparsePoly::new(
        3,
        vec![
            Term {
                coeff: F::from_u64(2),
                exps: vec![1, 0, 0],
            },
            Term {
                coeff: F::from_u64(3),
                exps: vec![0, 1, 1],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 2],
            },
        ],
    );
    CcsStructure::new(matrices, f).expect("valid CCS structure")
}

fn make_y_ring(seed: u64, t: usize, d_pad: usize) -> Vec<Vec<K>> {
    (0..t)
        .map(|j| {
            (0..d_pad)
                .map(|rho| k(seed + (j as u64) * 100 + (rho as u64) + 1))
                .collect::<Vec<K>>()
        })
        .collect()
}

fn make_ce_claim(seed: u64, t: usize, d_pad: usize, ell_n: usize, m_in: usize) -> CeClaim<Cmt, F, K> {
    CeClaim {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: Cmt::zeros(D, 1),
        X: Mat::zero(D, m_in, F::ZERO),
        r: vec_k(seed + 500, ell_n),
        s_col: vec![],
        y_ring: make_y_ring(seed, t, d_pad),
        ct: vec![K::ZERO; t],
        aux_openings: vec![k(seed + 999)],
        y_zcol: vec![],
        m_in,
        fold_digest: [0u8; 32],
    }
}

fn make_witness(seed: u64, m: usize) -> Mat<F> {
    if m.is_multiple_of(D) {
        let cols = m / D;
        let mut data = Vec::with_capacity(D * cols);
        for rho in 0..D {
            for blk in 0..cols {
                let c = blk * D + rho;
                data.push(F::from_u64(seed + (rho as u64) * 13 + (c as u64) * 7 + 1));
            }
        }
        Mat::from_row_major(D, cols, data)
    } else {
        let mut data = Vec::with_capacity(D * m);
        for rho in 0..D {
            for c in 0..m {
                data.push(F::from_u64(seed + (rho as u64) * 13 + (c as u64) * 7 + 1));
            }
        }
        Mat::from_row_major(D, m, data)
    }
}

#[test]
fn claimed_initial_sum_k_mcs_parity_k1_k2_k4_k61() {
    let n = 4usize;
    let m = 4usize;
    let s = build_structure(n, m);
    let d_pad = D.next_power_of_two();

    let ch = Challenges {
        alpha: vec_k(1000, 6),
        beta_a: vec_k(2000, 6),
        beta_r: vec_k(3000, 2),
        beta_m: vec![],
        gamma: k(11),
    };

    let me_inputs: Vec<CeClaim<Cmt, F, K>> = vec![
        make_ce_claim(100, s.t(), d_pad, 2, 1),
        make_ce_claim(200, s.t(), d_pad, 2, 1),
        make_ce_claim(300, s.t(), d_pad, 2, 1),
    ];

    for &k_mcs in &[1usize, 2, 4, 61] {
        let t_opt = optimized_engine::claimed_initial_sum_from_inputs_with_k_mcs(&s, &ch, k_mcs, &me_inputs);
        let t_paper = paper_exact_engine::claimed_initial_sum_from_inputs_with_k_mcs(&s, &ch, k_mcs, &me_inputs);
        assert_eq!(t_opt, t_paper, "claimed sum mismatch at k_mcs={k_mcs}");
    }
}

#[test]
fn terminal_fe_k_mcs_parity_k1_k2_k4() {
    let n = 4usize;
    let m = 4usize;
    let s = build_structure(n, m);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let d_pad = D.next_power_of_two();

    let ch = Challenges {
        alpha: vec_k(101, 6),
        beta_a: vec_k(201, 6),
        beta_r: vec_k(301, 2),
        beta_m: vec![],
        gamma: k(17),
    };

    let r_prime = vec_k(401, 2);
    let alpha_prime = vec_k(501, 6);
    let r_inputs = vec_k(601, 2);

    for &k_mcs in &[1usize, 2, 4] {
        let k_me = 2usize;
        let k_total = k_mcs + k_me;
        let mut out_me = Vec::with_capacity(k_total);
        for i in 0..k_total {
            out_me.push(make_ce_claim(10_000 + i as u64 * 1000, s.t(), d_pad, 2, 1));
        }

        let rhs_opt = optimized_engine::rhs_terminal_identity_fe_with_k_mcs(
            &s,
            &params,
            &ch,
            &r_prime,
            &alpha_prime,
            &out_me,
            k_mcs,
            Some(&r_inputs),
        );
        let rhs_paper = paper_exact_engine::rhs_terminal_identity_fe_paper_exact_with_k_mcs(
            &s,
            &params,
            &ch,
            &r_prime,
            &alpha_prime,
            &out_me,
            k_mcs,
            Some(&r_inputs),
        );
        assert_eq!(rhs_opt, rhs_paper, "terminal FE mismatch at k_mcs={k_mcs}");
    }
}

#[test]
fn terminal_fe_k_mcs_parity_superneo_shape_k1_k2_k4() {
    let n = D;
    let m = D;
    let s = build_structure(n, m);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let d_pad = D.next_power_of_two();

    let ch = Challenges {
        alpha: vec_k(1301, 6),
        beta_a: vec_k(1401, 6),
        beta_r: vec_k(1501, 6),
        beta_m: vec![],
        gamma: k(29),
    };

    let r_prime = vec_k(1601, 6);
    let alpha_prime = vec_k(1701, 6);
    let r_inputs = vec_k(1801, 6);

    for &k_mcs in &[1usize, 2, 4] {
        let k_me = 2usize;
        let k_total = k_mcs + k_me;
        let mut out_me = Vec::with_capacity(k_total);
        for i in 0..k_total {
            let mut out = make_ce_claim(20_000 + i as u64 * 1000, s.t(), d_pad, 6, 1);
            out.ct = out.y_ring.iter().take(s.t()).map(|row| row[0]).collect();
            out_me.push(out);
        }

        let rhs_opt = optimized_engine::rhs_terminal_identity_fe_with_k_mcs(
            &s,
            &params,
            &ch,
            &r_prime,
            &alpha_prime,
            &out_me,
            k_mcs,
            Some(&r_inputs),
        );
        let rhs_paper = paper_exact_engine::rhs_terminal_identity_fe_paper_exact_with_k_mcs(
            &s,
            &params,
            &ch,
            &r_prime,
            &alpha_prime,
            &out_me,
            k_mcs,
            Some(&r_inputs),
        );
        assert_eq!(
            rhs_opt, rhs_paper,
            "terminal FE mismatch (SuperNeo shape) at k_mcs={k_mcs}"
        );
    }
}

#[test]
fn terminal_fe_ignores_stale_ct_shell() {
    let n = D;
    let m = D;
    let s = build_structure(n, m);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let d_pad = D.next_power_of_two();

    let ch = Challenges {
        alpha: vec_k(2301, 6),
        beta_a: vec_k(2401, 6),
        beta_r: vec_k(2501, 6),
        beta_m: vec![],
        gamma: k(31),
    };

    let r_prime = vec_k(2601, 6);
    let alpha_prime = vec_k(2701, 6);
    let r_inputs = vec_k(2801, 6);

    let k_mcs = 2usize;
    let k_me = 2usize;
    let k_total = k_mcs + k_me;
    let mut canonical = Vec::with_capacity(k_total);
    for i in 0..k_total {
        let mut out = make_ce_claim(29_000 + i as u64 * 1000, s.t(), d_pad, 6, 1);
        out.ct = out.y_ring.iter().take(s.t()).map(|row| row[0]).collect();
        canonical.push(out);
    }
    let mut stale_ct = canonical.clone();
    for out in &mut stale_ct {
        for ct in &mut out.ct {
            *ct += K::ONE;
        }
    }

    let rhs_opt_canonical = optimized_engine::rhs_terminal_identity_fe_with_k_mcs(
        &s,
        &params,
        &ch,
        &r_prime,
        &alpha_prime,
        &canonical,
        k_mcs,
        Some(&r_inputs),
    );
    let rhs_opt_stale = optimized_engine::rhs_terminal_identity_fe_with_k_mcs(
        &s,
        &params,
        &ch,
        &r_prime,
        &alpha_prime,
        &stale_ct,
        k_mcs,
        Some(&r_inputs),
    );
    let rhs_paper_stale = paper_exact_engine::rhs_terminal_identity_fe_paper_exact_with_k_mcs(
        &s,
        &params,
        &ch,
        &r_prime,
        &alpha_prime,
        &stale_ct,
        k_mcs,
        Some(&r_inputs),
    );

    assert_eq!(
        rhs_opt_canonical, rhs_opt_stale,
        "optimized terminal FE should ignore stale ct shell"
    );
    assert_eq!(
        rhs_opt_stale, rhs_paper_stale,
        "paper and optimized FE should agree when ct shell is stale"
    );
}

#[test]
fn q_eval_ext_point_k_mcs_parity_k2_k4_k61() {
    let n = D;
    let m = D;
    let s = build_structure(n, m);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    let ch = Challenges {
        alpha: vec_k(701, 6),
        beta_a: vec_k(801, 6),
        beta_r: vec_k(901, 6),
        beta_m: vec![],
        gamma: k(23),
    };

    let alpha_prime = vec_k(1001, 6);
    let r_prime = vec_k(1101, 6);
    let r_inputs = vec_k(1201, 6);

    for &k_mcs in &[2usize, 4, 61] {
        let mcs_witnesses: Vec<CcsWitness<F>> = (0..k_mcs)
            .map(|i| CcsWitness {
                w: vec![F::ZERO; m.saturating_sub(1)],
                Z: make_witness(2_000 + (i as u64) * 50, m),
            })
            .collect();
        let me_witnesses = vec![make_witness(9_000, m), make_witness(10_000, m)];

        let (lhs_opt, _) = optimized_engine::q_eval_at_ext_point_paper_exact_with_inputs(
            &s,
            &params,
            &mcs_witnesses,
            &me_witnesses,
            &alpha_prime,
            &r_prime,
            &ch,
            Some(&r_inputs),
        );
        let (lhs_paper, _) = paper_exact_engine::q_eval_at_ext_point_paper_exact_with_inputs(
            &s,
            &params,
            &mcs_witnesses,
            &me_witnesses,
            &alpha_prime,
            &r_prime,
            &ch,
            Some(&r_inputs),
        );

        assert_eq!(lhs_opt, lhs_paper, "Q(α',r') mismatch at k_mcs={k_mcs}");
    }
}

#[test]
fn q_eval_ext_point_k_mcs_parity_superneo_shape_k2_k4() {
    let n = D;
    let m = D;
    let s = build_structure(n, m);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    let ch = Challenges {
        alpha: vec_k(1901, 6),
        beta_a: vec_k(2001, 6),
        beta_r: vec_k(2101, 6),
        beta_m: vec![],
        gamma: k(31),
    };

    let alpha_prime = vec_k(2201, 6);
    let r_prime = vec_k(2301, 6);
    let r_inputs = vec_k(2401, 6);

    for &k_mcs in &[2usize, 4] {
        let mcs_witnesses: Vec<CcsWitness<F>> = (0..k_mcs)
            .map(|i| CcsWitness {
                w: vec![F::ZERO; m.saturating_sub(1)],
                Z: make_witness(25_000 + (i as u64) * 50, m),
            })
            .collect();
        let me_witnesses = vec![make_witness(26_000, m), make_witness(27_000, m)];

        let (lhs_opt, _) = optimized_engine::q_eval_at_ext_point_paper_exact_with_inputs(
            &s,
            &params,
            &mcs_witnesses,
            &me_witnesses,
            &alpha_prime,
            &r_prime,
            &ch,
            Some(&r_inputs),
        );
        let (lhs_paper, _) = paper_exact_engine::q_eval_at_ext_point_paper_exact_with_inputs(
            &s,
            &params,
            &mcs_witnesses,
            &me_witnesses,
            &alpha_prime,
            &r_prime,
            &ch,
            Some(&r_inputs),
        );

        assert_eq!(
            lhs_opt, lhs_paper,
            "Q(α',r') mismatch (SuperNeo shape) at k_mcs={k_mcs}"
        );
    }
}

#[test]
fn ct_constant_term_guard_rejects_mismatch() {
    let n = D;
    let m = D;
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let s = build_structure(n, m);
    let d_pad = D.next_power_of_two();

    let mut out = make_ce_claim(33_000, s.t(), d_pad, 6, 1);
    out.ct = out.y_ring.iter().take(s.t()).map(|row| row[0]).collect();
    out.ct[0] += K::ONE; // tamper

    let err = neo_reductions::engines::utils::validate_ct_constant_term(&s, &params, &[out]).unwrap_err();
    assert!(err.to_string().contains("does not match"), "unexpected error: {err}");
}
