#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use neo_reductions::pi_ccs_paper_exact as refimpl;

use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat, SparsePoly, Term};

use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing};
use rand_chacha::rand_core::SeedableRng;

/// --- Shared helpers ----------------------------------------

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

fn build_chi_r(r: &[K]) -> Vec<K> {
    let n_sz = 1usize << r.len();
    let mut chi = vec![K::ZERO; n_sz];
    for row in 0..n_sz {
        let mut w = K::ONE;
        for bit in 0..r.len() {
            let rb = r[bit];
            let is_one = ((row >> bit) & 1) == 1;
            w *= if is_one { rb } else { K::ONE - rb };
        }
        chi[row] = w;
    }
    chi
}

fn build_vjs<Ff: Field + PrimeCharacteristicRing + Copy>(s: &CcsStructure<Ff>, r: &[K]) -> Vec<Vec<K>>
where
    K: From<Ff>,
{
    let chi = build_chi_r(r);
    let n_eff = core::cmp::min(s.n, chi.len());
    let mut vjs = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let mut vj = vec![K::ZERO; s.m];
        s.matrices[j].add_mul_transpose_into(&chi, &mut vj, n_eff);
        vjs.push(vj);
    }
    vjs
}

fn mul_Z_vec_digits<Ff: Field + PrimeCharacteristicRing + Copy>(Z: &Mat<Ff>, v: &[K]) -> Vec<K>
where
    K: From<Ff>,
{
    let expected_m = v.len();
    neo_reductions::common::validate_superneo_witness_mat(Z, expected_m)
        .expect("fixture witness must use packed SuperNeo layout");
    let mut out = vec![K::ZERO; D];
    for rho in 0..D {
        let mut acc = K::ZERO;
        for (c, &vc) in v.iter().enumerate() {
            acc += neo_reductions::common::witness_mat_get_k(Z, expected_m, rho, c) * vc;
        }
        out[rho] = acc;
    }
    out
}

fn mat_eq<Ff: Field + PrimeCharacteristicRing + Copy>(a: &Mat<Ff>, b: &Mat<Ff>) -> bool {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return false;
    }
    for r in 0..a.rows() {
        for c in 0..a.cols() {
            if a[(r, c)] != b[(r, c)] {
                return false;
            }
        }
    }
    true
}

fn public_inputs_from_witness(Z: &Mat<F>, expected_m: usize, m_in: usize) -> Vec<F> {
    neo_reductions::common::validate_superneo_witness_mat(Z, expected_m)
        .expect("fixture witness must use packed SuperNeo layout");
    (0..m_in)
        .map(|c| neo_reductions::common::witness_mat_get_f(Z, expected_m, c % D, c))
        .collect()
}

fn project_x_first_cols(Z: &Mat<F>, expected_m: usize, m_in: usize) -> Mat<F> {
    neo_reductions::common::project_x_from_witness_mat(Z, expected_m, m_in)
        .expect("fixture X projection should succeed")
}

fn tiny_ccs_t2(n: usize, m: usize) -> CcsStructure<F> {
    assert_eq!(n, m, "use square tiny ccs");
    // M0 = I, M1 = shift/swap
    let m0 = Mat::identity(n);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for i in 0..n {
        let j = (i + 1) % n;
        m1.set(i, j, F::ONE);
    }
    // f(y0,y1) = y0 + y1
    let f = SparsePoly::new(
        2,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 0],
            },
            Term {
                coeff: F::ONE,
                exps: vec![0, 1],
            },
        ],
    );
    CcsStructure::new(vec![m0, m1], f).unwrap()
}

/// --- Π_RLC tests ----------------------------------------------------------

#[test]
fn paper_exact_rlc_matches_direct_opening_and_eval() {
    let params = NeoParams::goldilocks_paper_b2();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = ajtai_for_dims(m);

    // Two inputs with same r
    let z1 = packed_constant(m, F::ONE);
    let z2 = packed_constant(m, F::from_u64(2));
    let w1 = CcsWitness {
        w: vec![],
        Z: z1.clone(),
    };
    let w2 = CcsWitness {
        w: vec![],
        Z: z2.clone(),
    };
    // choose m_in=1 so X is non-trivial
    let inst1 = CcsClaim {
        c: l.commit(&z1),
        x: public_inputs_from_witness(&z1, m, 1),
        m_in: 1,
    };
    let inst2 = CcsClaim {
        c: l.commit(&z2),
        x: public_inputs_from_witness(&z2, m, 1),
        m_in: 1,
    };

    let r = vec![K::from(F::from_u64(5)); 1];
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    // Get ME(b, L) inputs (y_(i,j), X_i) literally from the refimpl builder
    let out1 = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst1.clone()],
        &[w1.clone()],
        &[],
        &[],
        &r,
        &[],
        ell_d,
        [0; 32],
        &l,
    );
    let out2 = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst2.clone()],
        &[w2.clone()],
        &[],
        &[],
        &r,
        &[],
        ell_d,
        [0; 32],
        &l,
    );

    let inputs = vec![out1[0].clone(), out2[0].clone()];
    let Zs = vec![z1.clone(), z2.clone()];

    // ρ_1 = I, ρ_2 = 2·I  (commuting S-elements)
    let rho1 = Mat::identity(D);
    let mut rho2 = Mat::identity(D);
    for i in 0..D {
        rho2.set(i, i, F::from_u64(2));
    }
    let rhos = vec![rho1.clone(), rho2.clone()];

    // Run RLC
    let (combined_me, combined_Z) = refimpl::rlc_reduction_paper_exact::<F>(&s, &params, &rhos, &inputs, &Zs, ell_d);

    // Expected Z = Z1 + 2·Z2
    let mut Z_exp = Mat::zero(D, packed_cols(m), F::ZERO);
    for r_ in 0..D {
        for c_ in 0..Z_exp.cols() {
            let v = z1[(r_, c_)] + F::from_u64(2) * z2[(r_, c_)];
            Z_exp.set(r_, c_, v);
        }
    }

    assert!(mat_eq(&combined_Z, &Z_exp), "RLC combined Z must be Σ ρ_i Z_i");
    // X must be projection of combined Z
    let X_exp = project_x_first_cols(&Z_exp, m, /*m_in=*/ 1);
    assert!(
        mat_eq(&combined_me.X, &X_exp),
        "RLC combined X must equal projection of combined Z"
    );

    // y_j/ct must match a direct literal rebuild from the combined witness.
    let inst_exp = CcsClaim {
        c: l.commit(&Z_exp),
        x: public_inputs_from_witness(&Z_exp, m, 1),
        m_in: 1,
    };
    let w_exp = CcsWitness {
        w: vec![],
        Z: Z_exp.clone(),
    };
    let out_exp = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst_exp],
        &[w_exp],
        &[],
        &[],
        &r,
        &[],
        ell_d,
        [0; 32],
        &l,
    );
    for j in 0..s.t() {
        assert_eq!(
            combined_me.y_ring[j], out_exp[0].y_ring[j],
            "RLC y_ring mismatch for j={j}"
        );
    }
    assert_eq!(combined_me.ct, out_exp[0].ct, "RLC ct mismatch");

    // ct must match layout-aware CE scalar semantics.
    for j in 0..s.t() {
        let want = neo_reductions::common::ct_from_y_digits_for_ccs_m(&combined_me.y_ring[j], &params, s.m);
        assert_eq!(combined_me.ct[j], want, "RLC ct[j] mismatch");
    }
}

/// ---------------- Test: one full loop step (CCS → RLC → DEC) ----------------

#[test]
fn paper_exact_full_loop_k2_one_step_roundtrip() {
    let params = NeoParams::goldilocks_paper_b2();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_t2(n, m);
    let l = ajtai_for_dims(m);
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    // MCS witness Z_mcs and prior ME witness Z_prev (state)
    let Z_mcs = packed_constant(m, F::from_u64(2));
    let Z_prev = packed_constant(m, F::from_u64(3));

    let w_mcs = CcsWitness {
        w: vec![],
        Z: Z_mcs.clone(),
    };
    let inst_mcs = CcsClaim {
        c: l.commit(&Z_mcs),
        x: public_inputs_from_witness(&Z_mcs, m, 1),
        m_in: 1,
    };

    let me_r_prev = vec![K::from(F::from_u64(7)); 1];
    let me_prev = CeClaim {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: l.commit(&Z_prev),
        X: project_x_first_cols(&Z_prev, m, 1),
        r: me_r_prev.clone(),
        s_col: vec![],
        y_ring: vec![vec![K::ZERO; D]; s.t()],
        ct: vec![K::ZERO; s.t()],
        aux_openings: Vec::new(),
        y_zcol: vec![],
        m_in: 1,
        fold_digest: [0u8; 32],
    };

    // Π_CCS: build outputs at r'
    let r_prime = vec![K::from(F::from_u64(5)); 1];
    let out_ccs = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst_mcs.clone()],
        &[w_mcs.clone()],
        &[me_prev.clone()],
        &[Z_prev.clone()],
        &r_prime,
        &[],
        ell_d,
        [0; 32],
        &l,
    );

    // Optional: verify terminal identity of Π_CCS in this loop step
    if std::env::var("NEO_PAPER_CHECK_ID").ok().as_deref() == Some("1") {
        let ch = neo_reductions::Challenges {
            alpha: vec![K::from(F::from_u64(11)); ell_d],
            beta_a: vec![K::from(F::from_u64(13)); ell_d],
            beta_r: vec![K::from(F::from_u64(17)); 1],
            beta_m: Vec::new(),
            gamma: K::from(F::from_u64(19)),
        };
        let rhs = refimpl::rhs_terminal_identity_paper_exact(
            &s,
            &params,
            &ch,
            &r_prime,
            &ch.alpha,
            &out_ccs,
            Some(&me_r_prev),
        );
        let (lhs, _rhs_unused) = refimpl::q_eval_at_ext_point_paper_exact(
            &s,
            &params,
            &[w_mcs.clone()],
            &[Z_prev.clone()],
            &ch.alpha,
            &r_prime,
            &ch,
        );
        assert_eq!(lhs, rhs, "Π_CCS terminal identity must hold in the loop");
    }

    // Π_RLC with weights ρ_i = b^{i-1}·I  (i.e., [I, b·I])
    let bF = F::from_u64(params.b as u64);
    let rho1 = Mat::identity(D);
    let mut rho2 = Mat::identity(D);
    for d_ in 0..D {
        rho2.set(d_, d_, bF);
    }
    let rhos = vec![rho1, rho2];

    let inputs = vec![out_ccs[0].clone(), out_ccs[1].clone()];
    let Zs = vec![Z_mcs.clone(), Z_prev.clone()];

    let (combined_me, _combined_Z) = refimpl::rlc_reduction_paper_exact::<F>(&s, &params, &rhos, &inputs, &Zs, ell_d);

    // Π_DEC using the natural split (Z_1 = Z_mcs, Z_2 = Z_prev)
    let (children, ok_y, ok_X) =
        refimpl::dec_reduction_paper_exact::<F>(&s, &params, &combined_me, &[Z_mcs.clone(), Z_prev.clone()], ell_d);
    assert!(ok_y && ok_X, "Π_DEC reconstruction checks must pass");

    // After RLC (with powers of b) followed by DEC, we get back the CCS outputs
    for j in 0..s.t() {
        assert_eq!(
            children[0].y_ring[j], out_ccs[0].y_ring[j],
            "child[0].y_ring must match CCS output for j={}",
            j
        );
        assert_eq!(
            children[1].y_ring[j], out_ccs[1].y_ring[j],
            "child[1].y_ring must match CCS output for j={}",
            j
        );
    }
    assert!(
        mat_eq(&children[0].X, &out_ccs[0].X),
        "child[0].X must match CCS output"
    );
    assert!(
        mat_eq(&children[1].X, &out_ccs[1].X),
        "child[1].X must match CCS output"
    );
}

/// ---------------- Test: two consecutive loop steps (state carries forward) ----------------

#[test]
fn paper_exact_full_loop_k2_two_steps_chain() {
    let params = NeoParams::goldilocks_paper_b2();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_t2(n, m);
    let l = ajtai_for_dims(m);
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    // Step 1 witnesses
    let Z_mcs0 = packed_constant(m, F::from_u64(2));
    let Z_prev0 = packed_constant(m, F::from_u64(3));

    let w_mcs0 = CcsWitness {
        w: vec![],
        Z: Z_mcs0.clone(),
    };
    let inst_mcs0 = CcsClaim {
        c: l.commit(&Z_mcs0),
        x: public_inputs_from_witness(&Z_mcs0, m, 1),
        m_in: 1,
    };

    let me_r0 = vec![K::from(F::from_u64(23)); 1];
    let me0 = CeClaim {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: l.commit(&Z_prev0),
        X: project_x_first_cols(&Z_prev0, m, 1),
        r: me_r0.clone(),
        s_col: vec![],
        y_ring: vec![vec![K::ZERO; D]; s.t()],
        ct: vec![K::ZERO; s.t()],
        aux_openings: Vec::new(),
        y_zcol: vec![],
        m_in: 1,
        fold_digest: [0u8; 32],
    };

    let r_prime1 = vec![K::from(F::from_u64(29)); 1];
    let out1 = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst_mcs0.clone()],
        &[w_mcs0.clone()],
        &[me0.clone()],
        &[Z_prev0.clone()],
        &r_prime1,
        &[],
        ell_d,
        [0; 32],
        &l,
    );

    // RLC (ρ = [I, b·I]) then DEC back to two children
    let bF = F::from_u64(params.b as u64);
    let rho1 = Mat::identity(D);
    let mut rho2 = Mat::identity(D);
    for d_ in 0..D {
        rho2.set(d_, d_, bF);
    }
    let rhos = vec![rho1, rho2];

    let (combined_me1, _combined_Z1) = refimpl::rlc_reduction_paper_exact::<F>(
        &s,
        &params,
        &rhos,
        &vec![out1[0].clone(), out1[1].clone()],
        &vec![Z_mcs0.clone(), Z_prev0.clone()],
        ell_d,
    );
    let (children1, ok_y1, ok_X1) =
        refimpl::dec_reduction_paper_exact::<F>(&s, &params, &combined_me1, &[Z_mcs0.clone(), Z_prev0.clone()], ell_d);
    assert!(ok_y1 && ok_X1, "step 1 DEC checks must pass");

    // Carry forward the ME state as the child that corresponds to prior ME witness.
    // Here, child[1] corresponds to Z_prev0 (since split order was [Z_mcs0, Z_prev0]).
    let carried_me = children1[1].clone();
    let carried_Z = Z_prev0.clone();

    // Step 2: new MCS witness, same carried ME state
    let Z_mcs1 = packed_constant(m, F::from_u64(4));
    let w_mcs1 = CcsWitness {
        w: vec![],
        Z: Z_mcs1.clone(),
    };
    let inst_mcs1 = CcsClaim {
        c: l.commit(&Z_mcs1),
        x: public_inputs_from_witness(&Z_mcs1, m, 1),
        m_in: 1,
    };

    let r_prime2 = vec![K::from(F::from_u64(31)); 1];
    let out2 = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst_mcs1.clone()],
        &[w_mcs1.clone()],
        &[carried_me.clone()],
        &[carried_Z.clone()],
        &r_prime2,
        &[],
        ell_d,
        [0; 32],
        &l,
    );

    // Optional: verify Π_CCS terminal identity again at step 2
    if std::env::var("NEO_PAPER_CHECK_ID").ok().as_deref() == Some("1") {
        let ch2 = neo_reductions::Challenges {
            alpha: vec![K::from(F::from_u64(37)); ell_d],
            beta_a: vec![K::from(F::from_u64(41)); ell_d],
            beta_r: vec![K::from(F::from_u64(43)); 1],
            beta_m: Vec::new(),
            gamma: K::from(F::from_u64(47)),
        };
        let rhs2 = refimpl::rhs_terminal_identity_paper_exact(
            &s,
            &params,
            &ch2,
            &r_prime2,
            &ch2.alpha,
            &out2,
            Some(&carried_me.r),
        );
        let (lhs2, _rhs_unused2) = refimpl::q_eval_at_ext_point_paper_exact(
            &s,
            &params,
            &[w_mcs1.clone()],
            &[carried_Z.clone()],
            &ch2.alpha,
            &r_prime2,
            &ch2,
        );
        assert_eq!(lhs2, rhs2, "Π_CCS terminal identity must hold at step 2");
    }

    // RLC (same weights) then DEC back; must equal current CCS outputs
    let (combined_me2, _combined_Z2) = refimpl::rlc_reduction_paper_exact::<F>(
        &s,
        &params,
        &rhos,
        &vec![out2[0].clone(), out2[1].clone()],
        &vec![Z_mcs1.clone(), carried_Z.clone()],
        ell_d,
    );
    let (children2, ok_y2, ok_X2) = refimpl::dec_reduction_paper_exact::<F>(
        &s,
        &params,
        &combined_me2,
        &[Z_mcs1.clone(), carried_Z.clone()],
        ell_d,
    );
    assert!(ok_y2 && ok_X2, "step 2 DEC checks must pass");
    for j in 0..s.t() {
        assert_eq!(
            children2[0].y_ring[j], out2[0].y_ring[j],
            "step 2: child[0].y_ring must match CCS output j={}",
            j
        );
        assert_eq!(
            children2[1].y_ring[j], out2[1].y_ring[j],
            "step 2: child[1].y_ring must match CCS output j={}",
            j
        );
    }
    assert!(
        mat_eq(&children2[0].X, &out2[0].X),
        "step 2: child[0].X must match CCS output"
    );
    assert!(
        mat_eq(&children2[1].X, &out2[1].X),
        "step 2: child[1].X must match CCS output"
    );
}

#[test]
fn paper_exact_rlc_tampered_input_y_breaks_consistency() {
    let params = NeoParams::goldilocks_paper_b2();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = ajtai_for_dims(m);

    let z1 = packed_constant(m, F::ONE);
    let z2 = packed_constant(m, F::from_u64(3));
    let w1 = CcsWitness {
        w: vec![],
        Z: z1.clone(),
    };
    let w2 = CcsWitness {
        w: vec![],
        Z: z2.clone(),
    };
    let inst1 = CcsClaim {
        c: l.commit(&z1),
        x: public_inputs_from_witness(&z1, m, 1),
        m_in: 1,
    };
    let inst2 = CcsClaim {
        c: l.commit(&z2),
        x: public_inputs_from_witness(&z2, m, 1),
        m_in: 1,
    };

    let r = vec![K::from(F::from_u64(7)); 1];
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    let out1 = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst1.clone()],
        &[w1.clone()],
        &[],
        &[],
        &r,
        &[],
        ell_d,
        [0; 32],
        &l,
    );
    let mut out2 = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst2.clone()],
        &[w2.clone()],
        &[],
        &[],
        &r,
        &[],
        ell_d,
        [0; 32],
        &l,
    );

    // Tamper second input's first digit
    out2[0].y_ring[0][0] += K::ONE;

    let rhos = {
        let rho1 = Mat::identity(D);
        let mut rho2 = Mat::identity(D);
        for i in 0..D {
            rho2.set(i, i, F::from_u64(2));
        }
        vec![rho1, rho2]
    };

    let inputs = vec![out1[0].clone(), out2[0].clone()];
    let Zs = vec![z1.clone(), z2.clone()];

    let (combined_me, combined_Z) = refimpl::rlc_reduction_paper_exact::<F>(&s, &params, &rhos, &inputs, &Zs, ell_d);

    // Combined y (from tampered inputs) should NOT match eval from the (correctly combined) Z.
    let vjs = build_vjs(&s, &r);
    for j in 0..s.t() {
        let y_from_Z = mul_Z_vec_digits(&combined_Z, &vjs[j]);
        assert_ne!(
            &combined_me.y_ring[j][..D],
            &y_from_Z[..],
            "tampering inputs' y must break RLC consistency for j={}",
            j
        );
    }
}
