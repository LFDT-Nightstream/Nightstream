#![cfg(feature = "paper-exact")]

// Paper-exact used to validate engine precomputations (intermediate values)

use neo_fold::pi_ccs_paper_exact as refimpl;
use neo_fold::pi_ccs::{precompute, sparse_matrix::to_csr};
use neo_fold::{pi_ccs_prove, pi_ccs_verify, pi_ccs_prove_simple};
use neo_ccs::{CcsStructure, McsInstance, McsWitness, Mat, SparsePoly, Term, r1cs_to_ccs};
use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule};
use neo_math::{F, K, D};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use neo_ccs::traits::SModuleHomomorphism;
use neo_transcript::Transcript;
use neo_fold::pi_ccs::terminal::rhs_Q_apr as engine_rhs_terminal;
use neo_fold::pi_ccs::context::build_dims_and_policy;

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

fn z_to_z_row_major(params: &NeoParams, z_full: &[F]) -> Mat<F> {
    let d = D;
    let m = z_full.len();
    let z_digits = neo_ajtai::decomp_b(z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m { for row in 0..d { row_major[row*m + col] = z_digits[col*d + row]; } }
    Mat::from_row_major(d, m, row_major)
}

fn mcs_from_z(
    params: &NeoParams,
    z_full: Vec<F>,
    m_in: usize,
    l: &AjtaiSModule,
) -> (McsInstance<neo_ajtai::Commitment, F>, McsWitness<F>) {
    let z = z_to_z_row_major(params, &z_full);
    let c = l.commit(&z);
    let inst = McsInstance { c, x: z_full[..m_in].to_vec(), m_in };
    let wit = McsWitness { w: z_full[m_in..].to_vec(), Z: z };
    (inst, wit)
}

// ==================== Dummy S-module helpers for engine folding smoke ====================
struct DummyS;
impl neo_ccs::traits::SModuleHomomorphism<F, neo_ajtai::Commitment> for DummyS {
    fn commit(&self, z: &Mat<F>) -> neo_ajtai::Commitment {
        let d = z.rows();
        neo_ajtai::Commitment::zeros(d, 4)
    }
    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let cols = m_in.min(z.cols());
        let mut result = Mat::zero(rows, cols, F::ZERO);
        for r in 0..rows { for c in 0..cols { result[(r,c)] = z[(r,c)]; } }
        result
    }
}

fn create_add_ccs() -> CcsStructure<F> {
    let rows: usize = 4; let cols: usize = 5;
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];
    a[0 * cols + 4] = F::ONE; // out
    a[0 * cols + 1] = -F::ONE; // -x1
    a[0 * cols + 2] = -F::ONE; // -x2
    b[0 * cols + 0] = F::ONE;  // *1
    for row in 1..rows { b[row * cols + 0] = F::ONE; }
    r1cs_to_ccs(Mat::from_row_major(rows, cols, a), Mat::from_row_major(rows, cols, b), Mat::from_row_major(rows, cols, c))
}

fn create_mcs_from_witness_dummy(
    params: &NeoParams,
    z_full: Vec<F>,
    m_in: usize,
    l: &DummyS,
) -> (McsInstance<neo_ajtai::Commitment, F>, McsWitness<F>) {
    let d = D; let m = z_full.len();
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m { for row in 0..d { row_major[row*m + col] = z_digits[col*d + row]; } }
    let z = Mat::from_row_major(d, m, row_major);
    let c = l.commit(&z);
    (McsInstance { c, x: z_full[..m_in].to_vec(), m_in }, McsWitness { w: z_full[m_in..].to_vec(), Z: z })
}

#[test]
fn engine_k2_folding_end_to_end_succeeds() {
    // Use DummyS and a tiny addition CCS (same pattern as integration test) to avoid PP dependencies
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let s = create_add_ccs();
    let l = DummyS;

    // Step 0: 7 + 11 = 18
    let z2_full = vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)];
    let m_in = 1;
    let (mcs2, wit2) = create_mcs_from_witness_dummy(&params, z2_full, m_in, &l);
    let mut tr0 = neo_transcript::Poseidon2Transcript::new(b"engine/k1");
    let (me_step0, _p0) = pi_ccs_prove_simple(&mut tr0, &params, &s, &[mcs2.clone()], &[wit2.clone()], &l).unwrap();
    assert_eq!(me_step0.len(), 1);

    // Step 1: 2 + 3 = 5 fused with prior ME
    let z1_full = vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)];
    let (mcs1, wit1) = create_mcs_from_witness_dummy(&params, z1_full, m_in, &l);

    let mut tr = neo_transcript::Poseidon2Transcript::new(b"engine/k2");
    let (me_outs, proof) = pi_ccs_prove(&mut tr, &params, &s, &[mcs1.clone()], &[wit1], &[me_step0[0].clone()], &[wit2.Z.clone()], &l).unwrap();
    assert_eq!(me_outs.len(), 2);

    let mut tr_v = neo_transcript::Poseidon2Transcript::new(b"engine/k2");
    let ok = pi_ccs_verify(&mut tr_v, &params, &s, &[mcs1], &[me_step0[0].clone()], &me_outs, &proof).unwrap();
    assert!(ok, "engine folding k=2 should verify");
}

#[test]
fn engine_k2_invalid_mcs_witness_rejected() {
    // Prepare a bad MCS witness (c != L(Z)) and ensure prover errors.
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let m0 = Mat::identity(n);
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    let s = CcsStructure::new(vec![m0], f).unwrap();
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    // Good ME input to satisfy API shape
    let (mcs0, wit0) = mcs_from_z(&params, vec![F::from_u64(4); m], 0, &l);
    let mut tr0 = neo_transcript::Poseidon2Transcript::new(b"engine/k1-bad");
    let (me_step0, _p0) = pi_ccs_prove_simple(&mut tr0, &params, &s, &[mcs0.clone()], &[wit0.clone()], &l).unwrap();

    // Bad MCS: alter Z after committing
    let z_full_bad = vec![F::from_u64(5); m];
    let (mcs_bad, mut wit_bad) = mcs_from_z(&params, z_full_bad, 0, &l);
    // Flip one digit in Z so that commit(L(Z)) != inst.c
    wit_bad.Z.as_mut_slice()[0] += F::ONE;
    // Keep c from before (stale)

    let mut tr = neo_transcript::Poseidon2Transcript::new(b"engine/k2-bad");
    let res = pi_ccs_prove(&mut tr, &params, &s, &[mcs_bad.clone()], &[wit_bad], &[me_step0[0].clone()], &[wit0.Z.clone()], &l);
    assert!(res.is_err(), "prover must reject invalid MCS witness (c != L(Z))");
}

#[test]
fn intermediates_beta_f_at_beta_r_matches_naive() {
    // Compare F(β_r) from engine precompute to a naive literal computation
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    // t=2 with nontrivial M1
    let m0 = Mat::identity(n);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    m1.set(0, 1, F::ONE);
    m1.set(1, 0, F::ONE);
    let f = SparsePoly::new(2, vec![Term { coeff: F::ONE, exps: vec![1, 1] }]);
    let s = CcsStructure::new(vec![m0, m1], f).unwrap();

    // One MCS witness built consistently from z_full
    let z_full: Vec<F> = (0..m).map(|i| F::from_u64((i as u64)+1)).collect();
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let (inst, w) = mcs_from_z(&params, z_full.clone(), 0, &l);

    // Challenges: pick β_r with ell_n=1
    let ell_n = 1usize; let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let ch = neo_fold::pi_ccs::transcript::Challenges {
        alpha: vec![K::from(F::from_u64(3)); ell_d],
        beta_a: vec![K::from(F::from_u64(5)); ell_d],
        beta_r: vec![K::from(F::from_u64(7)); ell_n],
        gamma: K::from(F::from_u64(11)),
    };

    // Engine precompute
    let mats_csr: Vec<_> = s
        .matrices
        .iter()
        .map(|dmat| to_csr(dmat, s.n, s.m))
        .collect();
    let w_arr = [w.clone()];
    let insts = precompute::prepare_instances(&s, &params, &[inst.clone()], &w_arr, &mats_csr, &l).unwrap();
    let beta = precompute::precompute_beta_block(&s, &params, &insts, &w_arr, &[], &ch, ell_d, ell_n).unwrap();

    // Naive literal: F(β_r) = Σ_row χ_{β_r}(row)·f((M_j·z)[row])
    let chi_beta_r = neo_ccs::utils::tensor_point::<K>(&ch.beta_r);
    let mut f_naive = K::ZERO;
    for row in 0..s.n {
        let mut m_vals = vec![K::ZERO; s.t()];
        for j in 0..s.t() {
            let mut acc = K::ZERO;
            for c in 0..s.m {
                // z recomposed at column c
                let mut zc = K::ZERO;
                let b_k = K::from(F::from_u64(params.b as u64));
                let mut pow = K::ONE;
                for rho in 0..D { zc += K::from(w.Z[(rho, c)]) * pow; pow *= b_k; }
                acc += K::from(s.matrices[j][(row, c)]) * zc;
            }
            m_vals[j] = acc;
        }
        let f_row = s.f.eval_in_ext::<K>(&m_vals);
        f_naive += chi_beta_r[row] * f_row;
    }

    assert_eq!(beta.f_at_beta_r, f_naive, "F(β_r) precompute must match naive literal");
}

#[test]
fn intermediates_nc_hypercube_sum_matches_paper_exact() {
    // Compare engine NC hypercube sum with paper-exact hypercube of Q when f≡0 and no Eval
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let m0 = Mat::identity(n);
    let f0 = SparsePoly::new(1, vec![Term { coeff: F::ZERO, exps: vec![0] }]);
    let s = CcsStructure::new(vec![m0], f0).unwrap();

    // one MCS and one ME witness (k=2), but Eval suppressed (no r passed to paper-exact sum)
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let z0_full = vec![F::from_u64(2); m];
    let (inst0, w0) = mcs_from_z(&params, z0_full, 0, &l);
    let z1_full = vec![F::from_u64(3); m];
    let me_wits = vec![z_to_z_row_major(&params, &z1_full)];

    let ell_d = D.next_power_of_two().trailing_zeros() as usize; let ell_n = 1usize;
    let ch = neo_fold::pi_ccs::transcript::Challenges {
        alpha: vec![K::from(F::from_u64(5)); ell_d],
        beta_a: vec![K::from(F::from_u64(7)); ell_d],
        beta_r: vec![K::from(F::from_u64(11)); ell_n],
        gamma: K::from(F::from_u64(13)),
    };

    let mats_csr: Vec<_> = s
        .matrices
        .iter()
        .map(|dmat| to_csr(dmat, s.n, s.m))
        .collect();
    let w_arr = [w0.clone()];
    let insts = precompute::prepare_instances(&s, &params, &[inst0.clone()], &w_arr, &mats_csr, &l).unwrap();
    let beta = precompute::precompute_beta_block(&s, &params, &insts, &w_arr, &me_wits, &ch, ell_d, ell_n).unwrap();

    let sum_paper = refimpl::sum_q_over_hypercube_paper_exact::<F>(&s, &params, &w_arr.to_vec(), &me_wits, &ch, ell_d, ell_n, None);
    assert_eq!(beta.nc_sum_hypercube, sum_paper, "NC hypercube sums must match");
}

#[test]
fn intermediates_eval_row_partial_matches_naive() {
    // Compare engine Eval row aggregator G with a naive literal computation
    let _params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    // s with t=2
    let m0 = Mat::identity(n);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    m1.set(0, 0, F::ONE);
    m1.set(1, 1, F::ONE);
    let f0 = SparsePoly::new(2, vec![Term { coeff: F::ZERO, exps: vec![0,0] }]);
    let s = CcsStructure::new(vec![m0, m1], f0).unwrap();

    // one ME witness (no MCS needed for this test)
    let z1 = Mat::from_row_major(D, m, (0..D*m).map(|i| F::from_u64((i%4+1) as u64)).collect());
    let me_wits = vec![z1.clone()];

    let ell_n = 1usize;
    let k_total = 2usize; // pretend there was one MCS + one ME overall
    let ch = neo_fold::pi_ccs::transcript::Challenges {
        alpha: vec![K::from(F::from_u64(3)); D.next_power_of_two().trailing_zeros() as usize],
        beta_a: vec![K::ZERO; D.next_power_of_two().trailing_zeros() as usize],
        beta_r: vec![K::ZERO; ell_n],
        gamma: K::from(F::from_u64(7)),
    };

    let g_engine = precompute::precompute_eval_row_partial(&s, &me_wits, &ch, k_total, ell_n).unwrap();

    // Naive literal G[row] = Σ_{j,i} γ^{(i-1)+j·k_total} · (M_j · u_i)[row], u_i = Z_i^T χ_α
    let chi_alpha = neo_ccs::utils::tensor_point::<K>(&ch.alpha);
    let mut g_naive = vec![K::ZERO; 1<<ell_n];
    for (i_off, zi) in me_wits.iter().enumerate() {
        // u_i
        let mut u_i = vec![K::ZERO; s.m];
        for c in 0..s.m {
            let mut acc = K::ZERO;
            for rho in 0..D { let w = if rho < chi_alpha.len() { chi_alpha[rho] } else { K::ZERO }; acc += K::from(zi[(rho,c)]) * w; }
            u_i[c] = acc;
        }
        for j in 0..s.t() {
            // g_ij = M_j · u_i
            let mut g_ij = vec![K::ZERO; s.n];
            for row in 0..s.n { let mut acc = K::ZERO; for c in 0..s.m { acc += K::from(s.matrices[j][(row,c)])*u_i[c]; } g_ij[row] = acc; }
            // weight: γ^{(i-1)} * (γ^k)^(j+1)
            let mut w_pow = K::ONE;
            for _ in 0..(i_off+1) { w_pow *= ch.gamma; } // γ^{i-1}
            for _ in 0..k_total { w_pow *= ch.gamma; }   // initial (γ^k)
            for _ in 0..j { for _ in 0..k_total { w_pow *= ch.gamma; } } // × (γ^k)^j
            for row in 0..s.n { g_naive[row] += w_pow * g_ij[row]; }
        }
    }
    // Pad to 2^ell_n
    g_naive.resize(1<<ell_n, K::ZERO);

    assert_eq!(g_engine, g_naive, "Eval row partial must match naive literal vector");
}

#[test]
fn intermediates_initial_sum_matches_paper_exact_total() {
    // Full initial sum via engine precompute must equal paper-exact hypercube sum
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let m0 = Mat::identity(n);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    m1.set(0, 0, F::ONE); m1.set(1, 1, F::ONE);
    let f = SparsePoly::new(2, vec![Term { coeff: F::ONE, exps: vec![1,0] }]);
    let s = CcsStructure::new(vec![m0, m1], f).unwrap();

    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let (inst0, w0) = mcs_from_z(&params, vec![F::from_u64(2); m], 0, &l);
    let me_wits = vec![z_to_z_row_major(&params, &vec![F::from_u64(3); m])];
    let r_in = vec![K::from(F::from_u64(5)); 1];

    let ell_d = D.next_power_of_two().trailing_zeros() as usize; let ell_n = 1usize;
    let ch = neo_fold::pi_ccs::transcript::Challenges {
        alpha: vec![K::from(F::from_u64(7)); ell_d],
        beta_a: vec![K::from(F::from_u64(11)); ell_d],
        beta_r: vec![K::from(F::from_u64(13)); ell_n],
        gamma: K::from(F::from_u64(17)),
    };

    let mats_csr: Vec<_> = s
        .matrices
        .iter()
        .map(|dmat| to_csr(dmat, s.n, s.m))
        .collect();
    let w_arr = [w0.clone()];
    let insts = precompute::prepare_instances(&s, &params, &[inst0.clone()], &w_arr, &mats_csr, &l).unwrap();
    let beta = precompute::precompute_beta_block(&s, &params, &insts, &w_arr, &me_wits, &ch, ell_d, ell_n).unwrap();
    let g = precompute::precompute_eval_row_partial(&s, &me_wits, &ch, 2, ell_n).unwrap();
    let initial_sum = precompute::compute_initial_sum_components(&beta, Some(&r_in), &g).unwrap();

    let sum_paper = refimpl::sum_q_over_hypercube_paper_exact::<F>(&s, &params, &w_arr.to_vec(), &me_wits, &ch, ell_d, ell_n, Some(&r_in));
    assert_eq!(initial_sum, sum_paper, "Initial sum must equal paper-exact hypercube total");
}

#[test]
fn engine_paper_full_terminal_rhs_k2() {
    // Full pipeline k=2: compare engine RHS vs paper-exact RHS and against sumcheck_final
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    // CCS: t=1, f(y0)=y0
    let m0 = Mat::identity(n);
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    let s = CcsStructure::new(vec![m0], f).unwrap();

    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();

    // Build k=2 inputs: one MCS and one ME (derived via k=1 fold)
    let (mcs0, wit0) = mcs_from_z(&params, vec![F::from_u64(2); m], 0, &l);
    let mut tr0 = neo_transcript::Poseidon2Transcript::new(b"term/k1");
    let (me_step0, _p0) = pi_ccs_prove_simple(&mut tr0, &params, &s, &[mcs0.clone()], &[wit0.clone()], &l).unwrap();
    let (mcs1, wit1) = mcs_from_z(&params, vec![F::from_u64(3); m], 0, &l);

    // Run k=2 engine prove
    let mut tr = neo_transcript::Poseidon2Transcript::new(b"term/k2");
    let (me_outs, proof) = pi_ccs_prove(
        &mut tr, &params, &s,
        &[mcs1.clone()], &[wit1.clone()],
        &[me_step0[0].clone()], &[wit0.Z.clone()],
        &l
    ).unwrap();
    assert_eq!(me_outs.len(), 2);

    // Extract (alpha', r') from proof
    let dims = build_dims_and_policy(&params, &s).unwrap();
    let ell_n = dims.ell_n; let ell_d = dims.ell_d;
    let r_prime: Vec<K> = proof.sumcheck_challenges[..ell_n].to_vec();
    let alpha_prime: Vec<K> = proof.sumcheck_challenges[ell_n..(ell_n+ell_d)].to_vec();
    let ch = &proof.challenges_public;

    // Engine RHS
    let engine_rhs = engine_rhs_terminal(
        &s, ch, &r_prime, &alpha_prime,
        &[mcs1.clone()], &[me_step0[0].clone()], &me_outs, &params
    ).unwrap();
    // Paper-exact RHS
    let paper_rhs = refimpl::rhs_terminal_identity_paper_exact(
        &s, &params, ch, &r_prime, &alpha_prime, &me_outs, Some(&me_step0[0].r)
    );

    // Sumcheck final value
    let v_final = proof.sumcheck_final;

    assert_eq!(engine_rhs, paper_rhs, "engine RHS must equal paper RHS");
    assert_eq!(engine_rhs, v_final, "RHS must equal sumcheck final");

    // Tamper outputs: flip one Ajtai digit in ME output i=2
    let mut out_tampered = me_outs.clone();
    if let Some(me2) = out_tampered.get_mut(1) {
        if let Some(y0) = me2.y.get_mut(0) { y0[0] += K::ONE; }
    }

    let engine_rhs_bad = engine_rhs_terminal(
        &s, ch, &r_prime, &alpha_prime,
        &[mcs1], &[me_step0[0].clone()], &out_tampered, &params
    ).unwrap();
    let paper_rhs_bad = refimpl::rhs_terminal_identity_paper_exact(
        &s, &params, ch, &r_prime, &alpha_prime, &out_tampered, Some(&me_step0[0].r)
    );
    assert_eq!(engine_rhs_bad, paper_rhs_bad, "tampered: engine RHS must match paper RHS");
    assert_ne!(engine_rhs_bad, v_final, "tampered: RHS must differ from sumcheck final");
}

#[test]
fn engine_paper_rhs_match_mismatched_outputs() {
    // Build outputs from (inst0, me_in), then tamper y' and ensure engine RHS == paper RHS.
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    // Simple CCS: t=1, f(y0)=y0
    let m0 = Mat::identity(n);
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    let s = CcsStructure::new(vec![m0], f).unwrap();

    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let (inst0, w0) = mcs_from_z(&params, vec![F::from_u64(3); m], 0, &l);
    let me_z = z_to_z_row_major(&params, &vec![F::from_u64(4); m]);
    let _me_in: McsInstance<neo_ajtai::Commitment, F> = McsInstance { c: inst0.c.clone(), x: vec![], m_in: 0 }; // placeholder for count
    let me_input = neo_ccs::MeInstance {
        c_step_coords: vec![], u_offset: 0, u_len: 0,
        c: l.commit(&me_z),
        X: l.project_x(&me_z, 0),
        r: vec![K::from(F::from_u64(9)); 1],
        y: vec![vec![K::ZERO; D]],
        y_scalars: vec![K::ZERO],
        m_in: 0,
        fold_digest: [0u8; 32],
    };

    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let r_prime = vec![K::from(F::from_u64(7)); 1];
    let alpha_prime = vec![K::from(F::from_u64(5)); ell_d];
    let ch = neo_fold::pi_ccs::transcript::Challenges {
        alpha: vec![K::from(F::from_u64(11)); ell_d],
        beta_a: vec![K::from(F::from_u64(13)); ell_d],
        beta_r: vec![K::from(F::from_u64(15)); 1],
        gamma: K::from(F::from_u64(17)),
    };

    // Build outputs at r'
    let mut out = refimpl::build_me_outputs_paper_exact(
        &s, &params,
        &[inst0.clone()], &[w0.clone()],
        &[me_input.clone()], &[me_z.clone()],
        &r_prime, ell_d, [0u8; 32], &l,
    );
    // Tamper an Ajtai digit of ME output (i=2)
    if let Some(me_out) = out.get_mut(1) {
        if let Some(y0) = me_out.y.get_mut(0) { y0[0] += K::ONE; }
    }

    let engine_rhs = engine_rhs_terminal(&s, &ch, &r_prime, &alpha_prime, &[inst0.clone()], &[me_input.clone()], &out, &params).unwrap();
    let paper_rhs = refimpl::rhs_terminal_identity_paper_exact(&s, &params, &ch, &r_prime, &alpha_prime, &out, Some(&me_input.r));
    assert_eq!(engine_rhs, paper_rhs, "Engine RHS must match paper-exact RHS under tampered outputs");
}

#[test]
fn engine_paper_rhs_match_mismatched_r_gate() {
    // Use a different r for ME inputs than r' used to build outputs; both RHS must match.
    let params = NeoParams::goldilocks_small_circuits();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let m0 = Mat::identity(n);
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    let s = CcsStructure::new(vec![m0], f).unwrap();

    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let (inst0, w0) = mcs_from_z(&params, vec![F::from_u64(2); m], 0, &l);
    let me_z = z_to_z_row_major(&params, &vec![F::from_u64(3); m]);
    let me_input_base = neo_ccs::MeInstance {
        c_step_coords: vec![], u_offset: 0, u_len: 0,
        c: l.commit(&me_z),
        X: l.project_x(&me_z, 0),
        r: vec![K::from(F::from_u64(19)); 1],
        y: vec![vec![K::ZERO; D]],
        y_scalars: vec![K::ZERO],
        m_in: 0,
        fold_digest: [0u8; 32],
    };

    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let r_prime = vec![K::from(F::from_u64(23)); 1]; // different than me_input_base.r
    let alpha_prime = vec![K::from(F::from_u64(29)); ell_d];
    let ch = neo_fold::pi_ccs::transcript::Challenges {
        alpha: vec![K::from(F::from_u64(31)); ell_d],
        beta_a: vec![K::from(F::from_u64(37)); ell_d],
        beta_r: vec![K::from(F::from_u64(41)); 1],
        gamma: K::from(F::from_u64(43)),
    };

    // Build outputs at r'
    let out = refimpl::build_me_outputs_paper_exact(
        &s, &params,
        &[inst0.clone()], &[w0.clone()],
        &[me_input_base.clone()], &[me_z.clone()],
        &r_prime, ell_d, [0u8; 32], &l,
    );

    // Use a different r in me_inputs to test eq((α',r'),(α,r)) factor
    let me_input_alt = neo_ccs::MeInstance { r: vec![K::from(F::from_u64(47)); 1], ..me_input_base.clone() };

    let engine_rhs = engine_rhs_terminal(&s, &ch, &r_prime, &alpha_prime, &[inst0.clone()], &[me_input_alt.clone()], &out, &params).unwrap();
    let paper_rhs = refimpl::rhs_terminal_identity_paper_exact(&s, &params, &ch, &r_prime, &alpha_prime, &out, Some(&me_input_alt.r));
    assert_eq!(engine_rhs, paper_rhs, "Engine RHS must match paper-exact RHS when r gate uses different input r");
}
