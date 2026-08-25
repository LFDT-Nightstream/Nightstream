use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn identity_left(n: usize, m: usize) -> Mat<F> {
    let mut mat = Mat::zero(n, m, F::ZERO);
    for i in 0..n.min(m) {
        mat.set(i, i, F::ONE);
    }
    mat
}

fn zero_poly(t: usize) -> SparsePoly<F> {
    SparsePoly::new(t, Vec::new())
}

fn build_fixture(
    label: &'static [u8],
    n: usize,
    m: usize,
) -> (
    NeoParams,
    CcsStructure<F>,
    AjtaiSModule,
    CcsClaim<neo_ajtai::Commitment, F>,
    CcsWitness<F>,
    Poseidon2Transcript,
) {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    let s = CcsStructure::new(vec![identity_left(n, m)], zero_poly(1)).expect("ccs");

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m / D).expect("ajtai setup");
    let l = AjtaiSModule::new(Arc::new(pp));

    // Z entries are all zero digits (in-range for any b>=2).
    let z = Mat::from_row_major(D, m / D, vec![F::ZERO; D * (m / D)]);
    let c = l.commit(&z);
    let mcs_inst = CcsClaim {
        adv: None,
        c,
        x: vec![],
        m_in: 0,
    };
    let mcs_wit = CcsWitness {
        w: vec![F::ZERO; m],
        Z: z,
    };

    let tr = Poseidon2Transcript::new(label);
    (params, s, l, mcs_inst, mcs_wit, tr)
}

#[test]
fn padded_row_identity_rejects_an_extra_sumcheck_round() {
    let label = b"test/padded_row_identity/second_sumcheck";
    let (params, s, l, mcs_inst, mcs_wit, mut tr_p) = build_fixture(label, 4, D);

    let (out_me, mut proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    proof.sumcheck_rounds.push(vec![K::ZERO]);

    let mut tr_v = Poseidon2Transcript::new(label);
    let res = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &[],
        &out_me,
        &proof,
    );
    assert!(res.is_err(), "selected verifier accepted an extra SumCheck round");
}

#[test]
fn padded_row_identity_uses_one_joint_row_cube() {
    let label = b"test/padded_row_identity/joint_cube";
    let (params, s, l, mcs_inst, mcs_wit, mut tr_p) = build_fixture(label, 4, D);
    let assignment_width = neo_reductions::common::superneo_carrier_width(s.m);
    let variables =
        s.n.max(assignment_width)
            .next_power_of_two()
            .max(2)
            .trailing_zeros() as usize;

    let (out_me, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    assert_eq!(proof.sumcheck_rounds.len(), variables);
    assert!(!out_me.is_empty());
    assert_eq!(out_me[0].r.len(), variables);

    let mut tr_v = Poseidon2Transcript::new(label);
    let ok = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &[],
        &out_me,
        &proof,
    )
    .expect("verify should not error");
    assert!(ok);
}

#[test]
fn padded_row_identity_verify_rejects_eval_k_mutation() {
    let label = b"test/padded_row_identity/eval_k_mutation";
    let (params, s, l, mcs_inst, mcs_wit, mut tr_p) = build_fixture(label, 4, D);

    let (mut out_me, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    assert!(!out_me.is_empty());
    out_me[0].eval_k[0] += K::ONE;

    let mut tr_v = Poseidon2Transcript::new(label);
    let result = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &[],
        &out_me,
        &proof,
    );
    assert!(!matches!(result, Ok(true)), "mutated Eval_K was accepted");
}

#[test]
fn padded_row_identity_raw_verify_rejects_eval_a_mutation() {
    let label = b"test/padded_row_identity/redteam/eval_a_mutation";
    let (params, s, l, mcs_inst, mcs_wit, mut tr_p) = build_fixture(label, 4, D);

    let (mut outputs, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    assert!(!outputs[0].eval_a.is_empty() && !outputs[0].eval_a[0].is_empty());
    outputs[0].eval_a[0][0] += K::ONE;

    let mut tr_v = Poseidon2Transcript::new(label);
    let result = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &[],
        &outputs,
        &proof,
    );

    assert!(
        !matches!(result, Ok(true)),
        "raw PiCCS accepted a mutated Eval_A output"
    );
}

#[test]
fn padded_row_identity_raw_verify_rejects_noncanonical_extra_output_x_column() {
    let label = b"test/padded_row_identity/redteam/inactive_output_x";
    let (params, s, l, mut mcs, mut wit, mut tr_p) = build_fixture(label, 4, D);
    mcs.m_in = D;
    mcs.x = vec![F::ZERO; D];
    wit.w.clear();

    let (mut outputs, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs),
        core::slice::from_ref(&wit),
        &[],
        &[],
        &l,
    )
    .expect("prove");

    assert_eq!(outputs[0].X.rows(), D);
    assert_eq!(outputs[0].X.cols(), 1);
    outputs[0].X = Mat::zero(D, 2, F::ZERO);
    outputs[0].X[(0, 1)] = F::ONE;

    let mut tr_v = Poseidon2Transcript::new(label);
    let result = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs),
        &[],
        &outputs,
        &proof,
    );

    assert!(
        !matches!(result, Ok(true)),
        "raw Pi_CCS accepted a CE claim with a noncanonical extra X column"
    );
}

#[test]
fn raw_pi_ccs_rejects_fresh_count_above_parameter_profile() {
    let label = b"test/padded_row_identity/redteam/fresh_count_policy";
    let (params, s, l, claim, witness, mut tr_p) = build_fixture(label, 4, D);
    let count = neo_params::goldilocks_paper_b2::MAX_FRESH_K as usize + 1;
    let claims = vec![claim; count];
    let witnesses = vec![witness; count];

    let result = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        &claims,
        &witnesses,
        &[],
        &[],
        &l,
    );

    assert!(
        result.is_err(),
        "raw Pi_CCS accepted {} fresh claims under a profile capped at {}",
        count,
        neo_params::goldilocks_paper_b2::MAX_FRESH_K
    );
}

#[test]
fn raw_pi_ccs_rejects_running_count_above_parameter_profile() {
    let seed_label = b"test/padded_row_identity/redteam/running_count_seed";
    let (params, s, l, claim, witness, mut seed_tr) = build_fixture(seed_label, 4, D);
    let running_witness = witness.Z.clone();
    let (seed_outputs, _) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut seed_tr,
        &params,
        &s,
        core::slice::from_ref(&claim),
        core::slice::from_ref(&witness),
        &[],
        &[],
        &l,
    )
    .expect("seed one valid running claim");

    let count = params.k_rho as usize + 1;
    let running = vec![seed_outputs[0].clone(); count];
    let running_witnesses = vec![running_witness; count];
    let label = b"test/padded_row_identity/redteam/running_count_policy";
    let mut tr_p = Poseidon2Transcript::new(label);
    let result = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&claim),
        core::slice::from_ref(&witness),
        &running,
        &running_witnesses,
        &l,
    );

    assert!(
        result.is_err(),
        "raw Pi_CCS accepted {count} running claims under a profile fixed to k_rho={}",
        params.k_rho
    );
}
