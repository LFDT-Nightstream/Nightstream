#![allow(non_snake_case)]

use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{prove, verify, FoldingMode};
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn identity_ccs(n: usize) -> CcsStructure<F> {
    CcsStructure::new(vec![Mat::identity(n)], SparsePoly::new(1, vec![])).expect("valid CCS")
}

#[inline]
fn commit_cols_for_ccs_m(ccs_m: usize) -> usize {
    ccs_m.div_ceil(D)
}

fn setup_ajtai_committer(params: &NeoParams, m: usize) -> AjtaiSModule {
    let m_commit = commit_cols_for_ccs_m(m);
    let mut rng = ChaCha8Rng::seed_from_u64(19);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m_commit).expect("Ajtai setup");
    AjtaiSModule::new(Arc::new(pp))
}

fn build_mcs_step(
    _params: &NeoParams,
    l: &AjtaiSModule,
    m: usize,
    m_in: usize,
    base: i64,
) -> (CcsClaim<neo_ajtai::Commitment, F>, CcsWitness<F>) {
    let z: Vec<F> = (0..m)
        .map(|i| if ((i as i64) + base) % 2 == 0 { F::ONE } else { -F::ONE })
        .collect();
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    let mut Z = Mat::zero(D, m.div_ceil(D), F::ZERO);
    for (c, val) in z.iter().copied().enumerate() {
        Z[(c % D, c / D)] = val;
    }
    let c = l.commit(&Z);
    (CcsClaim { adv: None, c, x, m_in }, CcsWitness { w, Z })
}

fn build_mcs_step_packed_digits(
    l: &AjtaiSModule,
    m: usize,
    m_in: usize,
    seed: u64,
) -> (CcsClaim<neo_ajtai::Commitment, F>, CcsWitness<F>) {
    let mut z_cols = vec![F::ZERO; m];
    for (c, out) in z_cols.iter_mut().enumerate().take(m) {
        *out = match ((seed as usize) + c * 11) % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        };
    }
    let mut Z = Mat::zero(D, m.div_ceil(D), F::ZERO);
    for (c, val) in z_cols.iter().copied().enumerate().take(m) {
        Z[(c % D, c / D)] = val;
    }
    let x: Vec<F> = z_cols[..m_in].to_vec();
    let w = z_cols[m_in..].to_vec();
    let c = l.commit(&Z);
    (CcsClaim { adv: None, c, x, m_in }, CcsWitness { w, Z })
}

fn run_case_with_n(n: usize, k_mcs: usize) {
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let l = setup_ajtai_committer(&params, ccs.m);

    let mut mcs_list = Vec::with_capacity(k_mcs);
    let mut mcs_wits = Vec::with_capacity(k_mcs);
    for i in 0..k_mcs {
        let (inst, wit) = build_mcs_step(&params, &l, ccs.m, D, 50 + (i as i64) * 7);
        mcs_list.push(inst);
        mcs_wits.push(wit);
    }

    let mut tr_p = Poseidon2Transcript::new(b"neo.reductions/k_mcs_e2e");
    let (out, proof) = prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &ccs,
        &mcs_list,
        &mcs_wits,
        &[],
        &[],
        &l,
    )
    .expect("pi_ccs prove");

    let mut tr_v = Poseidon2Transcript::new(b"neo.reductions/k_mcs_e2e");
    let ok = verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        &mcs_list,
        &[],
        &out,
        &proof,
    )
    .expect("pi_ccs verify");
    assert!(ok, "pi_ccs verify should pass for k_mcs={k_mcs}");
}

fn run_case(k_mcs: usize) {
    run_case_with_n(D, k_mcs);
}

fn make_dummy_me_input(m_in: usize, r: Vec<K>) -> CeClaim<neo_ajtai::Commitment, F, K> {
    CeClaim {
        adv: None,
        c: neo_ajtai::Commitment::zeros(D, 1),
        X: Mat::zero(D, neo_ccs::superneo_public_x_cols(m_in), F::ZERO),
        r,
        eval_k: vec![K::ZERO; D.next_power_of_two()],
        eval_a: vec![vec![K::ZERO; D.next_power_of_two()]],
        m_in,
        fold_digest: [0u8; 32],
    }
}

#[test]
fn pi_ccs_prove_verify_k_mcs_1_2_4_nonzero_digits() {
    for &k_mcs in &[1usize, 2, 4] {
        run_case(k_mcs);
    }
}

#[test]
fn pi_ccs_prove_verify_k_mcs_61_boundary() {
    run_case(61);
}

#[test]
fn pi_ccs_prove_verify_superneo_shape_k_mcs_2() {
    run_case_with_n(D, 2);
}

#[test]
fn pi_ccs_prove_verify_superneo_shape_nonzero_digits_k_mcs_2() {
    let n = D;
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let l = setup_ajtai_committer(&params, ccs.m);

    let mut mcs_list = Vec::with_capacity(2);
    let mut mcs_wits = Vec::with_capacity(2);
    for i in 0..2 {
        let (inst, wit) = build_mcs_step_packed_digits(&l, ccs.m, D, 500 + (i as u64) * 17);
        mcs_list.push(inst);
        mcs_wits.push(wit);
    }

    let mut tr_p = Poseidon2Transcript::new(b"neo.reductions/superneo_packed_digits");
    let (out, proof) = prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &ccs,
        &mcs_list,
        &mcs_wits,
        &[],
        &[],
        &l,
    )
    .expect("pi_ccs prove");

    let mut tr_v = Poseidon2Transcript::new(b"neo.reductions/superneo_packed_digits");
    let ok = verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        &mcs_list,
        &[],
        &out,
        &proof,
    )
    .expect("pi_ccs verify");
    assert!(ok, "pi_ccs verify should pass for SuperNeo packed witness");
}

#[test]
fn pi_ccs_prove_rejects_non_shared_me_r() {
    let n = D;
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let l = setup_ajtai_committer(&params, ccs.m);
    let (mcs_inst, mcs_wit) = build_mcs_step(&params, &l, ccs.m, D, 71);

    let r_len = ccs.n.next_power_of_two().trailing_zeros() as usize;
    let me_inputs = vec![
        make_dummy_me_input(D, vec![K::ZERO; r_len]),
        make_dummy_me_input(D, vec![K::ONE; r_len]),
    ];
    let me_witnesses = vec![Mat::zero(D, ccs.m / D, F::ZERO), Mat::zero(D, ccs.m / D, F::ZERO)];

    let mut tr = Poseidon2Transcript::new(b"neo.reductions/non_shared_r");
    let err = prove(
        FoldingMode::Optimized,
        &mut tr,
        &params,
        &ccs,
        &[mcs_inst],
        &[mcs_wit],
        &me_inputs,
        &me_witnesses,
        &l,
    )
    .expect_err("prove must reject me_inputs with distinct r points");

    assert!(
        err.to_string()
            .contains("all ME inputs must share the same r"),
        "unexpected error: {err}"
    );
}

#[test]
fn pi_ccs_verify_rejects_tampered_mcs_output_x_recomposition() {
    let n = D;
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let l = setup_ajtai_committer(&params, ccs.m);

    let mut mcs_list = Vec::with_capacity(2);
    let mut mcs_wits = Vec::with_capacity(2);
    for i in 0..2 {
        let (inst, wit) = build_mcs_step(&params, &l, ccs.m, D, 90 + (i as i64) * 5);
        mcs_list.push(inst);
        mcs_wits.push(wit);
    }

    let mut tr_p = Poseidon2Transcript::new(b"neo.reductions/tamper_mcs_x");
    let (mut out, proof) = prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &ccs,
        &mcs_list,
        &mcs_wits,
        &[],
        &[],
        &l,
    )
    .expect("pi_ccs prove");

    out[0].X[(0, 0)] += F::ONE;

    let mut tr_v = Poseidon2Transcript::new(b"neo.reductions/tamper_mcs_x");
    let err = verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        &mcs_list,
        &[],
        &out,
        &proof,
    )
    .expect_err("verify must reject tampered MCS output X");

    assert!(
        err.to_string().contains("does not match mcs_list"),
        "unexpected error: {err}"
    );
}

#[test]
fn pi_ccs_verify_rejects_noncanonical_extra_x_column() {
    let n = D;
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let l = setup_ajtai_committer(&params, ccs.m);

    let mut mcs_list = Vec::with_capacity(1);
    let mut mcs_wits = Vec::with_capacity(1);
    let (inst, wit) = build_mcs_step(&params, &l, ccs.m, D, 90);
    mcs_list.push(inst);
    mcs_wits.push(wit);

    let mut tr_p = Poseidon2Transcript::new(b"neo.reductions/tamper_mcs_x_permute");
    let (mut out, proof) = prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &ccs,
        &mcs_list,
        &mcs_wits,
        &[],
        &[],
        &l,
    )
    .expect("pi_ccs prove");

    // Add a noncanonical column and move the active input into it.
    let old_x = out[0].X.clone();
    out[0].X = Mat::zero(D, old_x.cols() + 1, F::ZERO);
    for rho in 0..out[0].X.rows() {
        out[0].X[(rho, 1)] = old_x[(rho, 0)];
    }

    let mut tr_v = Poseidon2Transcript::new(b"neo.reductions/tamper_mcs_x_permute");
    let err = verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        &mcs_list,
        &[],
        &out,
        &proof,
    )
    .expect_err("verify must reject permuted MCS output X columns");

    assert!(err.to_string().contains("X has shape"), "unexpected error: {err}");
}
