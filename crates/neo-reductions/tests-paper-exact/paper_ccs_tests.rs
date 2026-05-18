#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat, SparsePoly, Term};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::pi_ccs_paper_exact as refimpl;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn q_ext_from_witnesses_lit(
    s: &CcsStructure<F>,
    params: &NeoParams,
    mcs_w: &[CcsWitness<F>],
    me_w: &[Mat<F>],
    alpha_p: &[K],
    r_p: &[K],
    ch: &neo_reductions::Challenges,
    me_inputs_r: Option<&[K]>,
) -> K {
    let (q, _unused_rhs) =
        refimpl::q_eval_at_ext_point_paper_exact_with_inputs(s, params, mcs_w, me_w, alpha_p, r_p, ch, me_inputs_r);
    q
}

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, 4, packed_cols(m)).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

fn packed_cols(m: usize) -> usize {
    m.div_ceil(D)
}

fn ajtai_for_dims(m: usize) -> AjtaiSModule {
    AjtaiSModule::from_global_for_dims(D, packed_cols(m)).unwrap()
}

fn packed_constant(m: usize, value: F) -> Mat<F> {
    let mut z = Mat::zero(D, packed_cols(m), F::ZERO);
    for col in 0..m {
        z.set(col % D, col / D, value);
    }
    z
}

fn packed_from_values(values: impl IntoIterator<Item = F>) -> Mat<F> {
    let vals: Vec<F> = values.into_iter().collect();
    let mut z = Mat::zero(D, packed_cols(vals.len()), F::ZERO);
    for (col, value) in vals.into_iter().enumerate() {
        z.set(col % D, col / D, value);
    }
    z
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
    let params = NeoParams::goldilocks_paper_b2();
    let n = D;
    let m = D;
    let m_commit = m / D;
    setup_ajtai_for_dims(m_commit);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m_commit).unwrap();

    let z = Mat::from_row_major(D, m_commit, vec![F::ONE; D * m_commit]);
    let w = CcsWitness { w: vec![], Z: z };
    let c = l.commit(&w.Z);
    let inst = CcsClaim { c, x: vec![], m_in: 0 };

    let ch = neo_reductions::Challenges {
        alpha: vec![rand_k(); 6],
        beta_a: vec![rand_k(); 6],
        beta_r: vec![rand_k(); 6],
        beta_m: Vec::new(),
        gamma: rand_k(),
    };
    let alpha_p = vec![K::from(F::from_u64(5)); 6];
    let r_p = vec![K::from(F::from_u64(7)); 6];

    let out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst.clone()],
        &[w.clone()],
        &[],
        &[],
        &r_p,
        &[],
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
    let params = NeoParams::goldilocks_paper_b2();
    let n = D;
    let m = D;
    let m_commit = m / D;
    setup_ajtai_for_dims(m_commit);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m_commit).unwrap();

    let z0 = Mat::from_row_major(D, m_commit, vec![F::ONE; D * m_commit]);
    let z1 = Mat::from_row_major(D, m_commit, vec![F::from_u64(2); D * m_commit]);
    let w0 = CcsWitness { w: vec![], Z: z0 };
    let me_z = z1.clone();

    let c0 = l.commit(&w0.Z);
    let inst0 = CcsClaim {
        c: c0,
        x: vec![],
        m_in: 0,
    };

    let me_r = vec![K::from(F::from_u64(9)); 6];
    let me_in = CeClaim {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: l.commit(&me_z),
        X: Mat::zero(D, 0, F::ZERO),
        r: me_r.clone(),
        s_col: vec![],
        y_ring: vec![vec![K::ZERO; D]],
        ct: vec![K::ZERO],
        aux_openings: Vec::new(),
        y_zcol: vec![],
        m_in: 0,
        fold_digest: [0u8; 32],
    };

    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(11)); 6],
        beta_a: vec![K::from(F::from_u64(13)); 6],
        beta_r: vec![K::from(F::from_u64(15)); 6],
        beta_m: Vec::new(),
        gamma: K::from(F::from_u64(17)),
    };
    let alpha_p = vec![K::from(F::from_u64(19)); 6];
    let r_p = vec![K::from(F::from_u64(21)); 6];

    let out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst0.clone()],
        &[w0.clone()],
        &[me_in.clone()],
        &[me_z.clone()],
        &r_p,
        &[],
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
    let params = NeoParams::goldilocks_paper_b2();
    let n = D;
    let m = D;
    let m_commit = m / D;
    setup_ajtai_for_dims(m_commit);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m_commit).unwrap();

    let z0 = Mat::from_row_major(D, m_commit, vec![F::from_u64(3); D * m_commit]);
    let z1 = Mat::from_row_major(D, m_commit, vec![F::from_u64(4); D * m_commit]);
    let w0 = CcsWitness { w: vec![], Z: z0 };
    let me_z = z1.clone();
    let c0 = l.commit(&w0.Z);
    let inst0 = CcsClaim {
        c: c0,
        x: vec![],
        m_in: 0,
    };

    let me_r = vec![K::from(F::from_u64(23)); 6];
    let me_in = CeClaim {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: l.commit(&me_z),
        X: Mat::zero(D, 0, F::ZERO),
        r: me_r.clone(),
        s_col: vec![],
        y_ring: vec![vec![K::ZERO; D]],
        ct: vec![K::ZERO],
        aux_openings: Vec::new(),
        y_zcol: vec![],
        m_in: 0,
        fold_digest: [0u8; 32],
    };

    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(29)); 6],
        beta_a: vec![K::from(F::from_u64(31)); 6],
        beta_r: vec![K::from(F::from_u64(37)); 6],
        beta_m: Vec::new(),
        gamma: K::from(F::from_u64(41)),
    };
    let alpha_p = vec![K::from(F::from_u64(43)); 6];
    let r_p = vec![K::from(F::from_u64(47)); 6];

    let out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst0.clone()],
        &[w0.clone()],
        &[me_in.clone()],
        &[me_z.clone()],
        &r_p,
        &[],
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
    let params = NeoParams::goldilocks_paper_b2();
    let n = D;
    let m = D;
    let m_commit = m / D;
    setup_ajtai_for_dims(m_commit);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m_commit).unwrap();

    let z0 = Mat::from_row_major(D, m_commit, vec![F::from_u64(5); D * m_commit]);
    let z1 = Mat::from_row_major(D, m_commit, vec![F::from_u64(6); D * m_commit]);
    let w0 = CcsWitness { w: vec![], Z: z0 };
    let me_z = z1.clone();
    let inst0 = CcsClaim {
        c: l.commit(&w0.Z),
        x: vec![],
        m_in: 0,
    };

    let me_r = vec![K::from(F::from_u64(51)); 6];
    let me_in = CeClaim {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: l.commit(&me_z),
        X: Mat::zero(D, 0, F::ZERO),
        r: me_r.clone(),
        s_col: vec![],
        y_ring: vec![vec![K::ZERO; D]],
        ct: vec![K::ZERO],
        aux_openings: Vec::new(),
        y_zcol: vec![],
        m_in: 0,
        fold_digest: [0u8; 32],
    };

    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(53)); 6],
        beta_a: vec![K::from(F::from_u64(57)); 6],
        beta_r: vec![K::from(F::from_u64(59)); 6],
        beta_m: Vec::new(),
        gamma: K::from(F::from_u64(61)),
    };
    let alpha_p = vec![K::from(F::from_u64(67)); 6];
    let r_p = vec![K::from(F::from_u64(71)); 6];

    let mut out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst0.clone()],
        &[w0.clone()],
        &[me_in.clone()],
        &[me_z.clone()],
        &r_p,
        &[],
        6,
        [0u8; 32],
        &l,
    );

    // Tamper a digit in y' for the ME output (i=2)
    if let Some(me_out) = out.get_mut(1) {
        if let Some(yj0) = me_out.y_ring.get_mut(0) {
            yj0[0] += K::ONE;
        }
    }

    let rhs_tampered = refimpl::rhs_terminal_identity_paper_exact(&s, &params, &ch, &r_p, &alpha_p, &out, Some(&me_r));
    let lhs_true = q_ext_from_witnesses_lit(&s, &params, &[w0], &[me_z], &alpha_p, &r_p, &ch, Some(&me_r));

    assert_ne!(lhs_true, rhs_tampered, "tampering outputs must break terminal identity");
}

#[test]
fn paper_exact_k2_ivc_two_steps() {
    let params = NeoParams::goldilocks_paper_b2();
    let n = D;
    let m = D;
    let m_commit = m / D;
    setup_ajtai_for_dims(m_commit);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m_commit).unwrap();

    let z_step0 = Mat::from_row_major(D, m_commit, vec![F::from_u64(8); D * m_commit]);
    let w_step0 = CcsWitness {
        w: vec![],
        Z: z_step0.clone(),
    };
    let inst_step0 = CcsClaim {
        c: l.commit(&z_step0),
        x: vec![],
        m_in: 0,
    };
    let r0 = vec![K::from(F::from_u64(73)); 6];
    let out0 = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst_step0.clone()],
        &[w_step0.clone()],
        &[],
        &[],
        &r0,
        &[],
        6,
        [0u8; 32],
        &l,
    );
    let me_input = out0[0].clone();

    let z_step1 = Mat::from_row_major(D, m_commit, vec![F::from_u64(9); D * m_commit]);
    let w_step1 = CcsWitness {
        w: vec![],
        Z: z_step1.clone(),
    };
    let inst_step1 = CcsClaim {
        c: l.commit(&z_step1),
        x: vec![],
        m_in: 0,
    };

    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(79)); 6],
        beta_a: vec![K::from(F::from_u64(83)); 6],
        beta_r: vec![K::from(F::from_u64(89)); 6],
        beta_m: Vec::new(),
        gamma: K::from(F::from_u64(97)),
    };
    let alpha_p = vec![K::from(F::from_u64(101)); 6];
    let r_p = vec![K::from(F::from_u64(103)); 6];

    let out1 = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst_step1.clone()],
        &[w_step1.clone()],
        &[me_input.clone()],
        &[z_step0.clone()],
        &r_p,
        &[],
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
    let params = NeoParams::goldilocks_paper_b2();
    let n = 2usize;
    let m = 2usize;
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = ajtai_for_dims(m);

    let z0 = packed_constant(m, F::from_u64(13));
    let z1 = packed_constant(m, F::from_u64(14));
    let w0 = CcsWitness {
        w: vec![],
        Z: z0.clone(),
    };
    let me_z = z1.clone();
    let inst0 = CcsClaim {
        c: l.commit(&z0),
        x: vec![],
        m_in: 0,
    };
    let me_r = vec![K::from(F::from_u64(17)); 1];
    let me_in = CeClaim {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: l.commit(&me_z),
        X: Mat::zero(D, 0, F::ZERO),
        r: me_r.clone(),
        s_col: vec![],
        y_ring: vec![vec![K::ZERO; D]],
        ct: vec![K::ZERO],
        aux_openings: Vec::new(),
        y_zcol: vec![],
        m_in: 0,
        fold_digest: [0u8; 32],
    };

    let alpha_p = vec![K::from(F::from_u64(19)); 6];
    let r_p = vec![K::from(F::from_u64(23)); 1];
    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(29)); 6],
        beta_a: vec![K::from(F::from_u64(31)); 6],
        beta_r: vec![K::from(F::from_u64(37)); 1],
        beta_m: Vec::new(),
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
        &[],
        6,
        [0u8; 32],
        &l,
    );

    let z_bad = packed_constant(m, F::from_u64(1));
    let w_bad = CcsWitness {
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
    let params = NeoParams::goldilocks_paper_b2();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let z = packed_constant(m, F::ONE);
    let w = CcsWitness { w: vec![], Z: z };
    let ell_d_full = D.next_power_of_two().trailing_zeros() as usize;
    let mut alpha_vec = vec![K::ZERO; ell_d_full];
    alpha_vec[0] = K::ONE;
    let ch = neo_reductions::Challenges {
        alpha: alpha_vec.clone(),
        beta_a: vec![K::from(F::from_u64(5)); ell_d_full],
        beta_r: vec![K::from(F::from_u64(11)); 1],
        beta_m: Vec::new(),
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
    let params = NeoParams::goldilocks_paper_b2();
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
    let l = ajtai_for_dims(m);

    let z = packed_from_values((0..m).map(|i| F::from_u64((i % 7) as u64 + 1)));
    let w = CcsWitness {
        w: vec![],
        Z: z.clone(),
    };
    let inst = CcsClaim {
        c: l.commit(&z),
        x: vec![],
        m_in: 0,
    };

    let r_p = vec![K::from(F::from_u64(5)); 1];

    let ell_d_full = D.next_power_of_two().trailing_zeros() as usize;
    let out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst],
        &[w.clone()],
        &[],
        &[],
        &r_p,
        &[],
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
    let n_eff = core::cmp::min(s.n, chi_rp.len());
    let z1 = refimpl::recomposed_z_from_Z(&params, s.m, &w.Z);
    let superneo_cache =
        neo_reductions::superneo_eval::build_superneo_eval_cache(&s).expect("fixture should be SuperNeo-compatible");
    let expected = neo_reductions::superneo_eval::eval_all_mats_ring_cached(&superneo_cache, &z1, &chi_rp, n_eff);
    for (j, yj) in expected.iter().enumerate().take(s.t()) {
        assert_eq!(
            &yj[..],
            &out[0].y_ring[j][..D],
            "y_ring must match literal SuperNeo ring eval for matrix j={j}"
        );
    }
}

#[test]
fn paper_exact_f_term_matches_mle_and_yprime_recomposition() {
    let params = NeoParams::goldilocks_paper_b2();
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
    let l = ajtai_for_dims(m);

    let z = packed_constant(m, F::from_u64(1));
    let w = CcsWitness {
        w: vec![],
        Z: z.clone(),
    };
    let inst = CcsClaim {
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
        &[],
        ell_d_full,
        [0; 32],
        &l,
    );

    let z1 = refimpl::recomposed_z_from_Z(&params, s.m, &w.Z);
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
    let superneo_cache =
        neo_reductions::superneo_eval::build_superneo_eval_cache(&s).expect("fixture should be SuperNeo-compatible");
    let n_eff = core::cmp::min(s.n, chi_rp.len());
    let m_from_mle = neo_reductions::superneo_eval::eval_all_mats_cached(&superneo_cache, &z1, &chi_rp, n_eff);
    let f_mle = s.f.eval_in_ext::<K>(&m_from_mle);
    let f_from_ct = s.f.eval_in_ext::<K>(&out[0].ct);

    assert_eq!(f_from_ct, f_mle, "F' must be computed from ct(M̃_j z_1(r'))");
}

#[test]
fn paper_exact_gamma_zero_kills_nc_and_eval() {
    let params = NeoParams::goldilocks_paper_b2();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let z0 = packed_constant(m, F::from_u64(2));
    let w0 = CcsWitness {
        w: vec![],
        Z: z0.clone(),
    };
    let ell_d_full = D.next_power_of_two().trailing_zeros() as usize;

    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(2)); ell_d_full],
        beta_a: vec![K::from(F::from_u64(3)); ell_d_full],
        beta_r: vec![K::from(F::from_u64(5)); 1],
        beta_m: Vec::new(),
        gamma: K::ZERO,
    };
    let alpha_p = vec![K::from(F::from_u64(7)); ell_d_full];
    let r_p = vec![K::from(F::from_u64(11)); 1];

    let q = q_ext_from_witnesses_lit(&s, &params, &[w0.clone()], &[], &alpha_p, &r_p, &ch, None);

    let eq_beta = refimpl::eq_points(&alpha_p, &ch.beta_a) * refimpl::eq_points(&r_p, &ch.beta_r);
    let l = ajtai_for_dims(m);
    let inst = CcsClaim {
        c: l.commit(&z0),
        x: vec![],
        m_in: 0,
    };
    let out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst],
        &[w0],
        &[],
        &[],
        &r_p,
        &[],
        ell_d_full,
        [0; 32],
        &l,
    );
    let f_prime = s.f.eval_in_ext::<K>(&out[0].ct);

    assert_eq!(q, eq_beta * f_prime, "γ=0 should zero out NC and Eval");
}

#[test]
fn paper_exact_ajtai_padding_is_zero() {
    let params = NeoParams::goldilocks_paper_b2();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = ajtai_for_dims(m);

    let z = packed_constant(m, F::from_u64(1));
    let w = CcsWitness {
        w: vec![],
        Z: z.clone(),
    };
    let inst = CcsClaim {
        c: l.commit(&z),
        x: vec![],
        m_in: 0,
    };

    let r_p = vec![K::from(F::from_u64(5)); 1];

    let ell_d_base = (D.next_power_of_two().trailing_zeros() as usize) + 1;
    let out =
        refimpl::build_me_outputs_paper_exact(&s, &params, &[inst], &[w], &[], &[], &r_p, &[], ell_d_base, [0; 32], &l);

    let want = 1usize << ell_d_base;
    for j in 0..s.t() {
        assert_eq!(out[0].y_ring[j].len(), want, "y' must be padded to 2^ell_d");
        assert!(
            out[0].y_ring[j][D..].iter().all(|&v| v == K::ZERO),
            "padding tail must be zero"
        );
    }
}
