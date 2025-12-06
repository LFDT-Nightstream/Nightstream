#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, McsInstance, McsWitness, MeInstance, SparsePoly, Term};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::pi_ccs_paper_exact as refimpl;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

/// --- Helpers for literal Q(α', r') from witnesses (paper-accurate) ----------

fn build_chi(p: &[K]) -> Vec<K> {
    let sz = 1usize << p.len();
    let mut chi = vec![K::ZERO; sz];
    for row in 0..sz {
        let mut w = K::ONE;
        for bit in 0..p.len() {
            let val = p[bit];
            let is_one = ((row >> bit) & 1) == 1;
            w *= if is_one { val } else { K::ONE - val };
        }
        chi[row] = w;
    }
    chi
}

fn build_vjs_ext(s: &CcsStructure<F>, r_prime: &[K]) -> Vec<Vec<K>> {
    let chi_rp = build_chi(r_prime);
    let n_sz = chi_rp.len();
    let mut vjs: Vec<Vec<K>> = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let mut vj = vec![K::ZERO; s.m];
        for row in 0..n_sz {
            let wr = if row < s.n { chi_rp[row] } else { K::ZERO };
            if wr == K::ZERO {
                continue;
            }
            for c in 0..s.m {
                vj[c] += K::from(s.matrices[j][(row, c)]) * wr;
            }
        }
        vjs.push(vj);
    }
    vjs
}

fn range_prod(val: K, b: u32) -> K {
    let lo = -((b as i64) - 1);
    let hi = (b as i64) - 1;
    let mut prod = K::ONE;
    for t in lo..=hi {
        prod *= val - K::from(F::from_i64(t));
    }
    prod
}

fn q_ext_from_witnesses_lit(
    s: &CcsStructure<F>,
    params: &NeoParams,
    mcs_w: &[McsWitness<F>],
    me_w: &[Mat<F>],
    alpha_p: &[K],
    r_p: &[K],
    ch: &neo_reductions::Challenges,
    me_inputs_r: Option<&[K]>,
) -> K {
    let k_total = mcs_w.len() + me_w.len();
    let chi_alpha = build_chi(alpha_p);
    let vjs = build_vjs_ext(s, r_p);

    // F'
    let z1 = refimpl::recomposed_z_from_Z(params, &mcs_w[0].Z);
    let mut m_vals = vec![K::ZERO; s.t()];
    for j in 0..s.t() {
        let vj = &vjs[j];
        let mut acc = K::ZERO;
        for c in 0..s.m {
            acc += vj[c] * z1[c];
        }
        m_vals[j] = acc;
    }
    let f_prime = s.f.eval_in_ext::<K>(&m_vals);

    // Σ γ^i · N_i'
    let mut nc_sum = K::ZERO;
    {
        let mut g = ch.gamma;
        let v1 = &vjs[0];
        let eval_y1 = |zi: &Mat<F>| {
            let mut y = vec![K::ZERO; D];
            for rho in 0..D {
                let mut a = K::ZERO;
                for c in 0..s.m {
                    a += K::from(zi[(rho, c)]) * v1[c];
                }
                y[rho] = a;
            }
            let mut yv = K::ZERO;
            let lim = core::cmp::min(D, chi_alpha.len());
            for rho in 0..lim {
                yv += y[rho] * chi_alpha[rho];
            }
            yv
        };
        for w in mcs_w {
            let yv = eval_y1(&w.Z);
            nc_sum += g * range_prod(yv, params.b);
            g *= ch.gamma;
        }
        for zi in me_w {
            let yv = eval_y1(zi);
            nc_sum += g * range_prod(yv, params.b);
            g *= ch.gamma;
        }
    }

    // Eval'
    let mut eval_sum = K::ZERO;
    if me_inputs_r.is_some() && k_total >= 2 {
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total {
            gamma_to_k *= ch.gamma;
        }
        let eq_ar = refimpl::eq_points(alpha_p, &ch.alpha) * refimpl::eq_points(r_p, me_inputs_r.unwrap());
        let eval_yj = |zi: &Mat<F>, j: usize| {
            let vj = &vjs[j];
            let mut y = vec![K::ZERO; D];
            for rho in 0..D {
                let mut a = K::ZERO;
                for c in 0..s.m {
                    a += K::from(zi[(rho, c)]) * vj[c];
                }
                y[rho] = a;
            }
            let mut yv = K::ZERO;
            let lim = core::cmp::min(D, chi_alpha.len());
            for rho in 0..lim {
                yv += y[rho] * chi_alpha[rho];
            }
            yv
        };
        let mut inner = K::ZERO;
        for j in 0..s.t() {
            for (i_abs, zi) in mcs_w
                .iter()
                .map(|w| &w.Z)
                .chain(me_w.iter())
                .enumerate()
                .skip(1)
            {
                let mut weight = K::ONE;
                for _ in 0..i_abs {
                    weight *= ch.gamma;
                }
                for _ in 0..j {
                    weight *= gamma_to_k;
                }
                inner += weight * eq_ar * eval_yj(zi, j);
            }
        }
        eval_sum = gamma_to_k * inner;
    }

    let eq_beta = refimpl::eq_points(alpha_p, &ch.beta_a) * refimpl::eq_points(r_p, &ch.beta_r);
    eq_beta * (f_prime + nc_sum) + eval_sum
}

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

fn tiny_ccs_id(n: usize, m: usize) -> CcsStructure<F> {
    assert_eq!(n, m, "use square tiny ccs");
    let m0 = Mat::identity(n);
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    ); // f(y0) = y0
    CcsStructure::new(vec![m0], f).unwrap()
}

fn rand_k() -> K {
    K::from(F::from_u64(3))
}

/// --- Π_CCS tests ----------------------------------------------------------

#[test]
fn paper_exact_rhs_matches_direct_eval_k1() {
    let params = NeoParams::goldilocks_127();
    let n = 2usize;
    let m = 2usize;
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z = Mat::from_row_major(D, m, vec![F::ONE; D * m]);
    let w = McsWitness { w: vec![], Z: z };
    let c = l.commit(&w.Z);
    let inst = McsInstance { c, x: vec![], m_in: 0 };

    let ch = neo_reductions::Challenges {
        alpha: vec![rand_k(); 6],
        beta_a: vec![rand_k(); 6],
        beta_r: vec![rand_k(); 1],
        gamma: rand_k(),
    };
    let alpha_p = vec![K::from(F::from_u64(5)); 6];
    let r_p = vec![K::from(F::from_u64(7)); 1];

    let out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst.clone()],
        &[w.clone()],
        &[],
        &[],
        &r_p,
        6,
        [0u8; 32],
        &l,
    );

    let rhs = refimpl::rhs_terminal_identity_paper_exact(&s, &params, &ch, &r_p, &alpha_p, &out, None);

    let lhs = q_ext_from_witnesses_lit(&s, &params, &[w], &[], &alpha_p, &r_p, &ch, None);

    assert_eq!(lhs, rhs, "paper-exact RHS must match direct Q(α',r') for k=1");
}

#[test]
fn paper_exact_rhs_matches_direct_eval_with_eval_block() {
    let params = NeoParams::goldilocks_127();
    let n = 2usize;
    let m = 2usize;
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z0 = Mat::from_row_major(D, m, vec![F::ONE; D * m]);
    let z1 = Mat::from_row_major(D, m, vec![F::from_u64(2); D * m]);
    let w0 = McsWitness { w: vec![], Z: z0 };
    let me_z = z1.clone();

    let c0 = l.commit(&w0.Z);
    let inst0 = McsInstance {
        c: c0,
        x: vec![],
        m_in: 0,
    };

    let me_r = vec![K::from(F::from_u64(9)); 1];
    let me_in = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: l.commit(&me_z),
        X: l.project_x(&me_z, 0),
        r: me_r.clone(),
        y: vec![vec![K::ZERO; D]],
        y_scalars: vec![K::ZERO],
        m_in: 0,
        fold_digest: [0u8; 32],
    };

    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(11)); 6],
        beta_a: vec![K::from(F::from_u64(13)); 6],
        beta_r: vec![K::from(F::from_u64(15)); 1],
        gamma: K::from(F::from_u64(17)),
    };
    let alpha_p = vec![K::from(F::from_u64(19)); 6];
    let r_p = vec![K::from(F::from_u64(21)); 1];

    let out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst0.clone()],
        &[w0.clone()],
        &[me_in.clone()],
        &[me_z.clone()],
        &r_p,
        6,
        [0u8; 32],
        &l,
    );

    let rhs = refimpl::rhs_terminal_identity_paper_exact(&s, &params, &ch, &r_p, &alpha_p, &out, Some(&me_r));

    let lhs = q_ext_from_witnesses_lit(&s, &params, &[w0], &[me_z], &alpha_p, &r_p, &ch, Some(&me_r));

    assert_eq!(lhs, rhs, "paper-exact RHS must match direct Q(α',r') with Eval block");
}

#[test]
fn paper_exact_k2_end_to_end_fold_identity() {
    let params = NeoParams::goldilocks_127();
    let n = 2usize;
    let m = 2usize;
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z0 = Mat::from_row_major(D, m, vec![F::from_u64(3); D * m]);
    let z1 = Mat::from_row_major(D, m, vec![F::from_u64(4); D * m]);
    let w0 = McsWitness { w: vec![], Z: z0 };
    let me_z = z1.clone();
    let c0 = l.commit(&w0.Z);
    let inst0 = McsInstance {
        c: c0,
        x: vec![],
        m_in: 0,
    };

    let me_r = vec![K::from(F::from_u64(23)); 1];
    let me_in = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: l.commit(&me_z),
        X: l.project_x(&me_z, 0),
        r: me_r.clone(),
        y: vec![vec![K::ZERO; D]],
        y_scalars: vec![K::ZERO],
        m_in: 0,
        fold_digest: [0u8; 32],
    };

    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(29)); 6],
        beta_a: vec![K::from(F::from_u64(31)); 6],
        beta_r: vec![K::from(F::from_u64(37)); 1],
        gamma: K::from(F::from_u64(41)),
    };
    let alpha_p = vec![K::from(F::from_u64(43)); 6];
    let r_p = vec![K::from(F::from_u64(47)); 1];

    let out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst0.clone()],
        &[w0.clone()],
        &[me_in.clone()],
        &[me_z.clone()],
        &r_p,
        6,
        [0u8; 32],
        &l,
    );

    let rhs = refimpl::rhs_terminal_identity_paper_exact(&s, &params, &ch, &r_p, &alpha_p, &out, Some(&me_r));
    let lhs = q_ext_from_witnesses_lit(&s, &params, &[w0], &[me_z], &alpha_p, &r_p, &ch, Some(&me_r));
    assert_eq!(lhs, rhs, "end-to-end terminal identity must hold for k=2");
}

#[test]
fn paper_exact_k2_invalid_outputs_break_identity() {
    let params = NeoParams::goldilocks_127();
    let n = 2usize;
    let m = 2usize;
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z0 = Mat::from_row_major(D, m, vec![F::from_u64(5); D * m]);
    let z1 = Mat::from_row_major(D, m, vec![F::from_u64(6); D * m]);
    let w0 = McsWitness { w: vec![], Z: z0 };
    let me_z = z1.clone();
    let inst0 = McsInstance {
        c: l.commit(&w0.Z),
        x: vec![],
        m_in: 0,
    };

    let me_r = vec![K::from(F::from_u64(51)); 1];
    let me_in = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: l.commit(&me_z),
        X: l.project_x(&me_z, 0),
        r: me_r.clone(),
        y: vec![vec![K::ZERO; D]],
        y_scalars: vec![K::ZERO],
        m_in: 0,
        fold_digest: [0u8; 32],
    };

    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(53)); 6],
        beta_a: vec![K::from(F::from_u64(57)); 6],
        beta_r: vec![K::from(F::from_u64(59)); 1],
        gamma: K::from(F::from_u64(61)),
    };
    let alpha_p = vec![K::from(F::from_u64(67)); 6];
    let r_p = vec![K::from(F::from_u64(71)); 1];

    let mut out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst0.clone()],
        &[w0.clone()],
        &[me_in.clone()],
        &[me_z.clone()],
        &r_p,
        6,
        [0u8; 32],
        &l,
    );

    // Tamper a digit in y' for the ME output (i=2)
    if let Some(me_out) = out.get_mut(1) {
        if let Some(yj0) = me_out.y.get_mut(0) {
            yj0[0] += K::ONE;
        }
    }

    let rhs_tampered = refimpl::rhs_terminal_identity_paper_exact(&s, &params, &ch, &r_p, &alpha_p, &out, Some(&me_r));
    let lhs_true = q_ext_from_witnesses_lit(&s, &params, &[w0], &[me_z], &alpha_p, &r_p, &ch, Some(&me_r));

    assert_ne!(lhs_true, rhs_tampered, "tampering outputs must break terminal identity");
}

#[test]
fn paper_exact_k2_ivc_two_steps() {
    let params = NeoParams::goldilocks_127();
    let n = 2usize;
    let m = 2usize;
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z_step0 = Mat::from_row_major(D, m, vec![F::from_u64(8); D * m]);
    let w_step0 = McsWitness {
        w: vec![],
        Z: z_step0.clone(),
    };
    let inst_step0 = McsInstance {
        c: l.commit(&z_step0),
        x: vec![],
        m_in: 0,
    };
    let r0 = vec![K::from(F::from_u64(73)); 1];
    let out0 = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst_step0.clone()],
        &[w_step0.clone()],
        &[],
        &[],
        &r0,
        6,
        [0u8; 32],
        &l,
    );
    let me_input = out0[0].clone();

    let z_step1 = Mat::from_row_major(D, m, vec![F::from_u64(9); D * m]);
    let w_step1 = McsWitness {
        w: vec![],
        Z: z_step1.clone(),
    };
    let inst_step1 = McsInstance {
        c: l.commit(&z_step1),
        x: vec![],
        m_in: 0,
    };

    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(79)); 6],
        beta_a: vec![K::from(F::from_u64(83)); 6],
        beta_r: vec![K::from(F::from_u64(89)); 1],
        gamma: K::from(F::from_u64(97)),
    };
    let alpha_p = vec![K::from(F::from_u64(101)); 6];
    let r_p = vec![K::from(F::from_u64(103)); 1];

    let out1 = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst_step1.clone()],
        &[w_step1.clone()],
        &[me_input.clone()],
        &[z_step0.clone()],
        &r_p,
        6,
        [0u8; 32],
        &l,
    );

    let rhs = refimpl::rhs_terminal_identity_paper_exact(&s, &params, &ch, &r_p, &alpha_p, &out1, Some(&me_input.r));
    let lhs = q_ext_from_witnesses_lit(
        &s,
        &params,
        &[w_step1],
        &[z_step0],
        &alpha_p,
        &r_p,
        &ch,
        Some(&me_input.r),
    );
    assert_eq!(lhs, rhs, "IVC-like two-step composition should hold under paper-exact");
}

#[test]
fn paper_exact_k2_mismatched_mcs_and_outputs() {
    let params = NeoParams::goldilocks_127();
    let n = 2usize;
    let m = 2usize;
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z0 = Mat::from_row_major(D, m, vec![F::from_u64(13); D * m]);
    let z1 = Mat::from_row_major(D, m, vec![F::from_u64(14); D * m]);
    let w0 = McsWitness {
        w: vec![],
        Z: z0.clone(),
    };
    let me_z = z1.clone();
    let inst0 = McsInstance {
        c: l.commit(&z0),
        x: vec![],
        m_in: 0,
    };
    let me_r = vec![K::from(F::from_u64(17)); 1];
    let me_in = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: l.commit(&me_z),
        X: l.project_x(&me_z, 0),
        r: me_r.clone(),
        y: vec![vec![K::ZERO; D]],
        y_scalars: vec![K::ZERO],
        m_in: 0,
        fold_digest: [0u8; 32],
    };

    let alpha_p = vec![K::from(F::from_u64(19)); 6];
    let r_p = vec![K::from(F::from_u64(23)); 1];
    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(29)); 6],
        beta_a: vec![K::from(F::from_u64(31)); 6],
        beta_r: vec![K::from(F::from_u64(37)); 1],
        gamma: K::from(F::from_u64(41)),
    };

    let out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst0.clone()],
        &[w0.clone()],
        &[me_in.clone()],
        &[me_z.clone()],
        &r_p,
        6,
        [0u8; 32],
        &l,
    );

    let z_bad = Mat::from_row_major(D, m, vec![F::from_u64(1); D * m]);
    let w_bad = McsWitness {
        w: vec![],
        Z: z_bad.clone(),
    };

    let rhs = refimpl::rhs_terminal_identity_paper_exact(&s, &params, &ch, &r_p, &alpha_p, &out, Some(&me_r));
    let lhs = q_ext_from_witnesses_lit(&s, &params, &[w_bad], &[me_z], &alpha_p, &r_p, &ch, Some(&me_r));
    assert_ne!(
        lhs, rhs,
        "Terminal identity must fail if outputs don't match witness used in Q"
    );
}

#[test]
fn paper_exact_boolean_corner_matches_extension_eval() {
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let z = Mat::from_row_major(D, m, vec![F::from_u64(7); D * m]);
    let w = McsWitness { w: vec![], Z: z };
    let ell_d_full = D.next_power_of_two().trailing_zeros() as usize;
    let mut alpha_vec = vec![K::ZERO; ell_d_full];
    alpha_vec[0] = K::ONE;
    let ch = neo_reductions::Challenges {
        alpha: alpha_vec.clone(),
        beta_a: vec![K::from(F::from_u64(5)); ell_d_full],
        beta_r: vec![K::from(F::from_u64(11)); 1],
        gamma: K::from(F::from_u64(13)),
    };

    let alpha_p = alpha_vec;
    let r_p = vec![K::ZERO];

    let lhs = refimpl::q_at_point_paper_exact::<F>(
        &s,
        &params,
        &[w.clone()],
        &[],
        &ch.alpha,
        &ch.beta_a,
        &ch.beta_r,
        ch.gamma,
        None,
        1,
        0,
    );
    let rhs = q_ext_from_witnesses_lit(&s, &params, &[w], &[], &alpha_p, &r_p, &ch, None);

    assert_eq!(lhs, rhs, "Boolean corner must match extension evaluation");
}

#[test]
fn paper_exact_outputs_equal_literal_definition() {
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let m0 = Mat::identity(n);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    m1.set(0, 0, F::ONE);
    m1.set(1, 1, F::ONE);
    let f = SparsePoly::new(
        2,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1, 0],
        }],
    );
    let s = CcsStructure::new(vec![m0.clone(), m1.clone()], f).unwrap();
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z = Mat::from_row_major(
        D,
        m,
        (0..D * m)
            .map(|i| F::from_u64((i % 7) as u64 + 1))
            .collect(),
    );
    let w = McsWitness {
        w: vec![],
        Z: z.clone(),
    };
    let inst = McsInstance {
        c: l.commit(&z),
        x: vec![],
        m_in: 0,
    };

    let r_p = vec![K::from(F::from_u64(5)); 1];

    let ell_d_full = D.next_power_of_two().trailing_zeros() as usize;
    let _out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst.clone()],
        &[w.clone()],
        &[],
        &[],
        &r_p,
        /*ell_d=*/ 1,
        [0u8; 32],
        &l,
    );
    let out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst],
        &[w.clone()],
        &[],
        &[],
        &r_p,
        ell_d_full,
        [0u8; 32],
        &l,
    );

    let n_sz = 1usize << r_p.len();
    let mut chi_rp = vec![K::ZERO; n_sz];
    for row in 0..n_sz {
        let mut wgt = K::ONE;
        for bit in 0..r_p.len() {
            let rb = r_p[bit];
            let is_one = ((row >> bit) & 1) == 1;
            wgt *= if is_one { rb } else { K::ONE - rb };
        }
        chi_rp[row] = wgt;
    }
    let mut vjs = vec![vec![K::ZERO; m]; s.t()];
    for j in 0..s.t() {
        for row in 0..n_sz {
            for c in 0..m {
                vjs[j][c] += K::from(s.matrices[j][(row, c)]) * chi_rp[row];
            }
        }
    }

    for j in 0..s.t() {
        let mut y_nav = vec![K::ZERO; D];
        for rho in 0..D {
            let mut acc = K::ZERO;
            for c in 0..m {
                acc += K::from(z[(rho, c)]) * vjs[j][c];
            }
            y_nav[rho] = acc;
        }
        assert_eq!(&y_nav[..], &out[0].y[j][..D]);
    }
}

#[test]
fn paper_exact_f_term_matches_mle_and_yprime_recomposition() {
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let m0 = Mat::identity(n);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    m1.set(0, 1, F::ONE);
    m1.set(1, 0, F::ONE);
    let f = SparsePoly::new(
        2,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1, 1],
        }],
    );
    let s = CcsStructure::new(vec![m0, m1], f).unwrap();
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z = Mat::from_row_major(D, m, vec![F::from_u64(1); D * m]);
    let w = McsWitness {
        w: vec![],
        Z: z.clone(),
    };
    let inst = McsInstance {
        c: l.commit(&z),
        x: vec![],
        m_in: 0,
    };

    let r_p = vec![K::from(F::from_u64(3)); 1];

    let ell_d_full = D.next_power_of_two().trailing_zeros() as usize;
    let out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst],
        &[w.clone()],
        &[],
        &[],
        &r_p,
        ell_d_full,
        [0; 32],
        &l,
    );

    let b_k = K::from(F::from_u64(params.b as u64));
    let mut pow = vec![K::ONE; D];
    for i in 1..D {
        pow[i] = pow[i - 1] * b_k;
    }
    let mut m_from_y = vec![K::ZERO; s.t()];
    for j in 0..s.t() {
        let mut acc = K::ZERO;
        for rho in 0..D {
            acc += out[0].y[j][rho] * pow[rho];
        }
        m_from_y[j] = acc;
    }
    let f_yprime = s.f.eval_in_ext::<K>(&m_from_y);

    let z1 = refimpl::recomposed_z_from_Z(&params, &w.Z);
    let n_sz = 1usize << r_p.len();
    let mut chi_rp = vec![K::ZERO; n_sz];
    for row in 0..n_sz {
        let mut wgt = K::ONE;
        for bit in 0..r_p.len() {
            let rb = r_p[bit];
            let is_one = ((row >> bit) & 1) == 1;
            wgt *= if is_one { rb } else { K::ONE - rb };
        }
        chi_rp[row] = wgt;
    }
    let mut m_from_mle = vec![K::ZERO; s.t()];
    for j in 0..s.t() {
        let mut vj = vec![K::ZERO; m];
        for row in 0..n_sz {
            for c in 0..m {
                vj[c] += K::from(s.matrices[j][(row, c)]) * chi_rp[row];
            }
        }
        for c in 0..m {
            m_from_mle[j] += z1[c] * vj[c];
        }
    }
    let f_mle = s.f.eval_in_ext::<K>(&m_from_mle);

    assert_eq!(f_yprime, f_mle, "F' must be f(M̃_j z_1(r'))");
}

#[test]
fn paper_exact_gamma_zero_kills_nc_and_eval() {
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let z0 = Mat::from_row_major(D, m, vec![F::from_u64(2); D * m]);
    let w0 = McsWitness {
        w: vec![],
        Z: z0.clone(),
    };

    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(2)); 1],
        beta_a: vec![K::from(F::from_u64(3)); 1],
        beta_r: vec![K::from(F::from_u64(5)); 1],
        gamma: K::ZERO,
    };
    let alpha_p = vec![K::from(F::from_u64(7)); 1];
    let r_p = vec![K::from(F::from_u64(11)); 1];

    let q = q_ext_from_witnesses_lit(&s, &params, &[w0.clone()], &[], &alpha_p, &r_p, &ch, None);

    let eq_beta = refimpl::eq_points(&alpha_p, &ch.beta_a) * refimpl::eq_points(&r_p, &ch.beta_r);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let inst = McsInstance {
        c: l.commit(&z0),
        x: vec![],
        m_in: 0,
    };
    let ell_d_full = D.next_power_of_two().trailing_zeros() as usize;
    let out =
        refimpl::build_me_outputs_paper_exact(&s, &params, &[inst], &[w0], &[], &[], &r_p, ell_d_full, [0; 32], &l);
    let b_k = K::from(F::from_u64(params.b as u64));
    let mut pow = vec![K::ONE; D];
    for i in 1..D {
        pow[i] = pow[i - 1] * b_k;
    }
    let mut m_vals = vec![K::ZERO; s.t()];
    for j in 0..s.t() {
        let mut acc = K::ZERO;
        for rho in 0..D {
            acc += out[0].y[j][rho] * pow[rho];
        }
        m_vals[j] = acc;
    }
    let f_prime = s.f.eval_in_ext::<K>(&m_vals);

    assert_eq!(q, eq_beta * f_prime, "γ=0 should zero out NC and Eval");
}

#[test]
fn paper_exact_ajtai_padding_is_zero() {
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z = Mat::from_row_major(D, m, vec![F::from_u64(1); D * m]);
    let w = McsWitness {
        w: vec![],
        Z: z.clone(),
    };
    let inst = McsInstance {
        c: l.commit(&z),
        x: vec![],
        m_in: 0,
    };

    let r_p = vec![K::from(F::from_u64(5)); 1];

    let ell_d_base = (D.next_power_of_two().trailing_zeros() as usize) + 1;
    let out =
        refimpl::build_me_outputs_paper_exact(&s, &params, &[inst], &[w], &[], &[], &r_p, ell_d_base, [0; 32], &l);

    let want = 1usize << ell_d_base;
    for j in 0..s.t() {
        assert_eq!(out[0].y[j].len(), want, "y' must be padded to 2^ell_d");
        assert!(
            out[0].y[j][D..].iter().all(|&v| v == K::ZERO),
            "padding tail must be zero"
        );
    }
}
