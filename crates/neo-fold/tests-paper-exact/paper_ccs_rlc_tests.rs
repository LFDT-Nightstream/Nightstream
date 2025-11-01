#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use neo_fold::pi_ccs_paper_exact as refimpl;

use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat, SparsePoly, Term};
use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;

use rand_chacha::rand_core::SeedableRng;
use neo_params::NeoParams;
use neo_math::{F, K, D};
use p3_field::{Field, PrimeCharacteristicRing};

/// --- Shared helpers ----------------------------------------

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

fn tiny_ccs_id(n: usize, m: usize) -> CcsStructure<F> {
    assert_eq!(n, m, "use square tiny ccs");
    let m0 = Mat::identity(n);
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]); // f(y0) = y0
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

fn build_vjs<Ff: Field + PrimeCharacteristicRing + Copy>(
    s: &CcsStructure<Ff>,
    r: &[K],
) -> Vec<Vec<K>>
where K: From<Ff>
{
    let chi = build_chi_r(r);
    let n_sz = chi.len();
    let mut vjs = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let mut vj = vec![K::ZERO; s.m];
        for row in 0..n_sz {
            let wr = if row < s.n { chi[row] } else { K::ZERO };
            if wr == K::ZERO { continue; }
            for c in 0..s.m {
                vj[c] += K::from(s.matrices[j][(row, c)]) * wr;
            }
        }
        vjs.push(vj);
    }
    vjs
}

fn mul_Z_vec_digits<Ff: Field + PrimeCharacteristicRing + Copy>(Z: &Mat<Ff>, v: &[K]) -> Vec<K>
where K: From<Ff>
{
    let mut out = vec![K::ZERO; D];
    for rho in 0..D {
        let mut acc = K::ZERO;
        for c in 0..Z.cols() {
            acc += K::from(Z[(rho, c)]) * v[c];
        }
        out[rho] = acc;
    }
    out
}

fn mat_eq<Ff: Field + PrimeCharacteristicRing + Copy>(a: &Mat<Ff>, b: &Mat<Ff>) -> bool {
    if a.rows()!=b.rows() || a.cols()!=b.cols() { return false; }
    for r in 0..a.rows() {
        for c in 0..a.cols() {
            if a[(r,c)] != b[(r,c)] { return false; }
        }
    }
    true
}

fn project_x_first_cols<Ff: Field + PrimeCharacteristicRing + Copy>(Z: &Mat<Ff>, m_in: usize) -> Mat<Ff> {
    let mut X: Mat<Ff> = Mat::zero(D, m_in, Ff::ZERO);
    for r in 0..D {
        for c in 0..m_in {
            X.set(r, c, Z[(r, c)]);
        }
    }
    X
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
    let f = SparsePoly::new(2, vec![
        Term { coeff: F::ONE, exps: vec![1, 0] },
        Term { coeff: F::ONE, exps: vec![0, 1] },
    ]);
    CcsStructure::new(vec![m0, m1], f).unwrap()
}

/// --- Π_RLC tests ----------------------------------------------------------

#[test]
fn paper_exact_rlc_matches_direct_opening_and_eval() {
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s  = tiny_ccs_id(n, m);
    let l  = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    // Two inputs with same r
    let z1 = Mat::from_row_major(D, m, vec![F::ONE; D*m]);
    let z2 = Mat::from_row_major(D, m, vec![F::from_u64(2); D*m]);
    let w1 = McsWitness { w: vec![], Z: z1.clone() };
    let w2 = McsWitness { w: vec![], Z: z2.clone() };
    // choose m_in=1 so X is non-trivial
    let inst1 = McsInstance { c: l.commit(&z1), x: vec![], m_in: 1 };
    let inst2 = McsInstance { c: l.commit(&z2), x: vec![], m_in: 1 };

    let r = vec![K::from(F::from_u64(5)); 1];
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    // Get ME(b, L) inputs (y_(i,j), X_i) literally from the refimpl builder
    let out1 = refimpl::build_me_outputs_paper_exact(&s, &params, &[inst1.clone()], &[w1.clone()], &[], &[], &r, ell_d, [0;32], &l);
    let out2 = refimpl::build_me_outputs_paper_exact(&s, &params, &[inst2.clone()], &[w2.clone()], &[], &[], &r, ell_d, [0;32], &l);

    let inputs = vec![out1[0].clone(), out2[0].clone()];
    let Zs     = vec![z1.clone(), z2.clone()];

    // ρ_1 = I, ρ_2 = 2·I  (commuting S-elements)
    let rho1 = Mat::identity(D);
    let mut rho2 = Mat::identity(D);
    for i in 0..D { rho2.set(i, i, F::from_u64(2)); }
    let rhos = vec![rho1.clone(), rho2.clone()];

    // Run RLC
    let (combined_me, combined_Z) = refimpl::rlc_reduction_paper_exact::<F>(&s, &params, &rhos, &inputs, &Zs, ell_d);

    // Expected Z = Z1 + 2·Z2
    let mut Z_exp = Mat::zero(D, m, F::ZERO);
    for r_ in 0..D { for c_ in 0..m {
        let v = z1[(r_,c_)] + F::from_u64(2) * z2[(r_, c_)];
        Z_exp.set(r_, c_, v);
    }}

    assert!(mat_eq(&combined_Z, &Z_exp), "RLC combined Z must be Σ ρ_i Z_i");
    // X must be projection of combined Z
    let X_exp = project_x_first_cols(&Z_exp, /*m_in=*/1);
    assert!(mat_eq(&combined_me.X, &X_exp), "RLC combined X must equal projection of combined Z");

    // y_j must equal (combined Z)·(M_j^T χ_r)
    let vjs = build_vjs(&s, &r);
    for j in 0..s.t() {
        let y_from_Z = mul_Z_vec_digits(&Z_exp, &vjs[j]);
        assert_eq!(&combined_me.y[j][..D], &y_from_Z[..], "RLC y_j must equal combined-Z eval for j={}", j);
        // padding tail must be zero
        assert!(combined_me.y[j][D..].iter().all(|&v| v == K::ZERO), "padding tail must be zero");
    }

    // y_scalars must recompose from digits with base b
    let bK = K::from(F::from_u64(params.b as u64));
    let mut pow = vec![K::ONE; D]; for i in 1..D { pow[i] = pow[i-1] * bK; }
    for j in 0..s.t() {
        let mut recomposed = K::ZERO;
        for rho in 0..D { recomposed += combined_me.y[j][rho] * pow[rho]; }
        assert_eq!(combined_me.y_scalars[j], recomposed, "RLC y_scalars[j] must recompose digits");
    }
}

/// ---------------- Test: one full loop step (CCS → RLC → DEC) ----------------

#[test]
fn paper_exact_full_loop_k2_one_step_roundtrip() {
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s  = tiny_ccs_t2(n, m);
    let l  = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    // MCS witness Z_mcs and prior ME witness Z_prev (state)
    let Z_mcs  = Mat::from_row_major(D, m, vec![F::from_u64(2); D*m]);
    let Z_prev = Mat::from_row_major(D, m, vec![F::from_u64(3); D*m]);

    let w_mcs  = McsWitness { w: vec![], Z: Z_mcs.clone() };
    let inst_mcs = McsInstance { c: l.commit(&Z_mcs), x: vec![], m_in: 1 };

    let me_r_prev = vec![K::from(F::from_u64(7)); 1];
    let me_prev = MeInstance {
        c_step_coords: vec![], u_offset: 0, u_len: 0,
        c: l.commit(&Z_prev),
        X: l.project_x(&Z_prev, 1),
        r: me_r_prev.clone(),
        y: vec![vec![K::ZERO; D]; s.t()],
        y_scalars: vec![K::ZERO; s.t()],
        m_in: 1,
        fold_digest: [0u8; 32],
    };

    // Π_CCS: build outputs at r'
    let r_prime = vec![K::from(F::from_u64(5)); 1];
    let out_ccs = refimpl::build_me_outputs_paper_exact(
        &s, &params,
        &[inst_mcs.clone()], &[w_mcs.clone()],
        &[me_prev.clone()], &[Z_prev.clone()],
        &r_prime, ell_d, [0; 32], &l,
    );

    // Optional: verify terminal identity of Π_CCS in this loop step
    if std::env::var("NEO_PAPER_CHECK_ID").ok().as_deref() == Some("1") {
        let ch = neo_fold::pi_ccs::transcript::Challenges {
            alpha:  vec![K::from(F::from_u64(11)); ell_d],
            beta_a: vec![K::from(F::from_u64(13)); ell_d],
            beta_r: vec![K::from(F::from_u64(17)); 1],
            gamma:  K::from(F::from_u64(19)),
        };
        let rhs = refimpl::rhs_terminal_identity_paper_exact(
            &s, &params, &ch, &r_prime, &ch.alpha, &out_ccs, Some(&me_r_prev)
        );
        let (lhs, _rhs_unused) = refimpl::q_eval_at_ext_point_paper_exact(
            &s, &params, &[w_mcs.clone()], &[Z_prev.clone()], &ch.alpha, &r_prime, &ch
        );
        assert_eq!(lhs, rhs, "Π_CCS terminal identity must hold in the loop");
    }

    // Π_RLC with weights ρ_i = b^{i-1}·I  (i.e., [I, b·I])
    let bF = F::from_u64(params.b as u64);
    let rho1 = Mat::identity(D);
    let mut rho2 = Mat::identity(D);
    for d_ in 0..D { rho2.set(d_, d_, bF); }
    let rhos = vec![rho1, rho2];

    let inputs = vec![out_ccs[0].clone(), out_ccs[1].clone()];
    let Zs     = vec![Z_mcs.clone(), Z_prev.clone()];

    let (combined_me, _combined_Z) = refimpl::rlc_reduction_paper_exact::<F>(
        &s, &params, &rhos, &inputs, &Zs, ell_d
    );

    // Π_DEC using the natural split (Z_1 = Z_mcs, Z_2 = Z_prev)
    let (children, ok_y, ok_X) = refimpl::dec_reduction_paper_exact::<F>(
        &s, &params, &combined_me, &[Z_mcs.clone(), Z_prev.clone()], ell_d
    );
    assert!(ok_y && ok_X, "Π_DEC reconstruction checks must pass");

    // After RLC (with powers of b) followed by DEC, we get back the CCS outputs
    for j in 0..s.t() {
        assert_eq!(children[0].y[j], out_ccs[0].y[j], "child[0].y must match CCS output for j={}", j);
        assert_eq!(children[1].y[j], out_ccs[1].y[j], "child[1].y must match CCS output for j={}", j);
    }
    assert!(mat_eq(&children[0].X, &out_ccs[0].X), "child[0].X must match CCS output");
    assert!(mat_eq(&children[1].X, &out_ccs[1].X), "child[1].X must match CCS output");
}

/// ---------------- Test: two consecutive loop steps (state carries forward) ----------------

#[test]
fn paper_exact_full_loop_k2_two_steps_chain() {
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s  = tiny_ccs_t2(n, m);
    let l  = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    // Step 1 witnesses
    let Z_mcs0  = Mat::from_row_major(D, m, vec![F::from_u64(2); D*m]);
    let Z_prev0 = Mat::from_row_major(D, m, vec![F::from_u64(3); D*m]);

    let w_mcs0   = McsWitness { w: vec![], Z: Z_mcs0.clone() };
    let inst_mcs0 = McsInstance { c: l.commit(&Z_mcs0), x: vec![], m_in: 1 };

    let me_r0 = vec![K::from(F::from_u64(23)); 1];
    let me0 = MeInstance {
        c_step_coords: vec![], u_offset: 0, u_len: 0,
        c: l.commit(&Z_prev0),
        X: l.project_x(&Z_prev0, 1),
        r: me_r0.clone(),
        y: vec![vec![K::ZERO; D]; s.t()],
        y_scalars: vec![K::ZERO; s.t()],
        m_in: 1,
        fold_digest: [0u8; 32],
    };

    let r_prime1 = vec![K::from(F::from_u64(29)); 1];
    let out1 = refimpl::build_me_outputs_paper_exact(
        &s, &params,
        &[inst_mcs0.clone()], &[w_mcs0.clone()],
        &[me0.clone()], &[Z_prev0.clone()],
        &r_prime1, ell_d, [0; 32], &l,
    );

    // RLC (ρ = [I, b·I]) then DEC back to two children
    let bF = F::from_u64(params.b as u64);
    let rho1 = Mat::identity(D);
    let mut rho2 = Mat::identity(D);
    for d_ in 0..D { rho2.set(d_, d_, bF); }
    let rhos = vec![rho1, rho2];

    let (combined_me1, _combined_Z1) = refimpl::rlc_reduction_paper_exact::<F>(
        &s, &params, &rhos, &vec![out1[0].clone(), out1[1].clone()],
        &vec![Z_mcs0.clone(), Z_prev0.clone()], ell_d
    );
    let (children1, ok_y1, ok_X1) = refimpl::dec_reduction_paper_exact::<F>(
        &s, &params, &combined_me1, &[Z_mcs0.clone(), Z_prev0.clone()], ell_d
    );
    assert!(ok_y1 && ok_X1, "step 1 DEC checks must pass");

    // Carry forward the ME state as the child that corresponds to prior ME witness.
    // Here, child[1] corresponds to Z_prev0 (since split order was [Z_mcs0, Z_prev0]).
    let carried_me = children1[1].clone();
    let carried_Z  = Z_prev0.clone();

    // Step 2: new MCS witness, same carried ME state
    let Z_mcs1  = Mat::from_row_major(D, m, vec![F::from_u64(4); D*m]);
    let w_mcs1  = McsWitness { w: vec![], Z: Z_mcs1.clone() };
    let inst_mcs1 = McsInstance { c: l.commit(&Z_mcs1), x: vec![], m_in: 1 };

    let r_prime2 = vec![K::from(F::from_u64(31)); 1];
    let out2 = refimpl::build_me_outputs_paper_exact(
        &s, &params,
        &[inst_mcs1.clone()], &[w_mcs1.clone()],
        &[carried_me.clone()], &[carried_Z.clone()],
        &r_prime2, ell_d, [0; 32], &l,
    );

    // Optional: verify Π_CCS terminal identity again at step 2
    if std::env::var("NEO_PAPER_CHECK_ID").ok().as_deref() == Some("1") {
        let ch2 = neo_fold::pi_ccs::transcript::Challenges {
            alpha:  vec![K::from(F::from_u64(37)); ell_d],
            beta_a: vec![K::from(F::from_u64(41)); ell_d],
            beta_r: vec![K::from(F::from_u64(43)); 1],
            gamma:  K::from(F::from_u64(47)),
        };
        let rhs2 = refimpl::rhs_terminal_identity_paper_exact(
            &s, &params, &ch2, &r_prime2, &ch2.alpha, &out2, Some(&carried_me.r)
        );
        let (lhs2, _rhs_unused2) = refimpl::q_eval_at_ext_point_paper_exact(
            &s, &params, &[w_mcs1.clone()], &[carried_Z.clone()], &ch2.alpha, &r_prime2, &ch2
        );
        assert_eq!(lhs2, rhs2, "Π_CCS terminal identity must hold at step 2");
    }

    // RLC (same weights) then DEC back; must equal current CCS outputs
    let (combined_me2, _combined_Z2) = refimpl::rlc_reduction_paper_exact::<F>(
        &s, &params, &rhos, &vec![out2[0].clone(), out2[1].clone()],
        &vec![Z_mcs1.clone(), carried_Z.clone()], ell_d
    );
    let (children2, ok_y2, ok_X2) = refimpl::dec_reduction_paper_exact::<F>(
        &s, &params, &combined_me2, &[Z_mcs1.clone(), carried_Z.clone()], ell_d
    );
    assert!(ok_y2 && ok_X2, "step 2 DEC checks must pass");
    for j in 0..s.t() {
        assert_eq!(children2[0].y[j], out2[0].y[j], "step 2: child[0].y must match CCS output j={}", j);
        assert_eq!(children2[1].y[j], out2[1].y[j], "step 2: child[1].y must match CCS output j={}", j);
    }
    assert!(mat_eq(&children2[0].X, &out2[0].X), "step 2: child[0].X must match CCS output");
    assert!(mat_eq(&children2[1].X, &out2[1].X), "step 2: child[1].X must match CCS output");
}

#[test]
fn paper_exact_rlc_tampered_input_y_breaks_consistency() {
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s  = tiny_ccs_id(n, m);
    let l  = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    let z1 = Mat::from_row_major(D, m, vec![F::ONE; D*m]);
    let z2 = Mat::from_row_major(D, m, vec![F::from_u64(3); D*m]);
    let w1 = McsWitness { w: vec![], Z: z1.clone() };
    let w2 = McsWitness { w: vec![], Z: z2.clone() };
    let inst1 = McsInstance { c: l.commit(&z1), x: vec![], m_in: 1 };
    let inst2 = McsInstance { c: l.commit(&z2), x: vec![], m_in: 1 };

    let r = vec![K::from(F::from_u64(7)); 1];
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    let out1 = refimpl::build_me_outputs_paper_exact(&s, &params, &[inst1.clone()], &[w1.clone()], &[], &[], &r, ell_d, [0;32], &l);
    let mut out2 = refimpl::build_me_outputs_paper_exact(&s, &params, &[inst2.clone()], &[w2.clone()], &[], &[], &r, ell_d, [0;32], &l);

    // Tamper second input's first digit
    out2[0].y[0][0] += K::ONE;

    let rhos = {
        let rho1 = Mat::identity(D);
        let mut rho2 = Mat::identity(D);
        for i in 0..D { rho2.set(i, i, F::from_u64(2)); }
        vec![rho1, rho2]
    };

    let inputs = vec![out1[0].clone(), out2[0].clone()];
    let Zs     = vec![z1.clone(), z2.clone()];

    let (combined_me, combined_Z) = refimpl::rlc_reduction_paper_exact::<F>(&s, &params, &rhos, &inputs, &Zs, ell_d);

    // Combined y (from tampered inputs) should NOT match eval from the (correctly combined) Z.
    let vjs = build_vjs(&s, &r);
    for j in 0..s.t() {
        let y_from_Z = mul_Z_vec_digits(&combined_Z, &vjs[j]);
        assert_ne!(&combined_me.y[j][..D], &y_from_Z[..], "tampering inputs' y must break RLC consistency for j={}", j);
    }
}

