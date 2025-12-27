#![allow(non_snake_case)]
#![allow(deprecated)]

#[path = "common/fixtures.rs"]
mod fixtures;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{
    matrix::Mat,
    relations::{CcsStructure, MeInstance},
};
use neo_fold::memory_sidecar::claim_plan::RouteATimeClaimPlan;
use neo_fold::shard::CommitMixers;
use neo_fold::shard::{fold_shard_prove as fold_shard_prove_legacy, fold_shard_verify as fold_shard_verify_legacy};
use neo_math::K;
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::engines::utils;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

fn build_single_chunk_inputs() -> (
    NeoParams,
    CcsStructure<F>,
    StepWitnessBundle<Cmt, F, K>,
    Vec<MeInstance<Cmt, F, K>>,
    Vec<Mat<F>>,
    fixtures::HashCommit,
    CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>,
) {
    let fx = fixtures::build_twist_shout_2step_fixture(1);
    let step_bundle = fx.steps_witness[0].clone();
    (
        fx.params,
        fx.ccs,
        step_bundle,
        fx.acc_init,
        fx.acc_wit_init,
        fx.l,
        fx.mixers,
    )
}

#[test]
fn tamper_batched_time_round_poly_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-batched-time-round");
    let mut proof = fold_shard_prove_legacy(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    proof.steps[0].batched_time.round_polys[0][0][0] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-batched-time-round");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify_legacy(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered batched_time round poly must fail verification"
    );
}

#[test]
fn tamper_ccs_header_digest_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-ccs-header-digest");
    let mut proof = fold_shard_prove_legacy(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    proof.steps[0].fold.ccs_proof.header_digest[0] ^= 1;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-ccs-header-digest");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify_legacy(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered CCS header digest must fail verification"
    );
}

#[test]
fn tamper_ccs_challenges_public_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-ccs-challenges-public");
    let mut proof = fold_shard_prove_legacy(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    proof.steps[0].fold.ccs_proof.challenges_public.gamma += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-ccs-challenges-public");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify_legacy(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered CCS challenges_public must fail verification"
    );
}

#[test]
fn tamper_rlc_inputs_changes_rho_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let ccs_only_step: StepWitnessBundle<Cmt, F, K> = StepWitnessBundle::from(step_bundle.mcs.clone());

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-rlc-inputs-changes-rho");
    let mut proof = fold_shard_prove_legacy(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[ccs_only_step.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    proof.steps[0].fold.ccs_out[0].c.data[0] += F::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-rlc-inputs-changes-rho");
    let steps_public = [StepInstanceBundle::from(&ccs_only_step)];
    assert!(
        fold_shard_verify_legacy(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered RLC inputs must fail rho transcript binding"
    );
}

#[test]
fn tamper_rlc_rho_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-rlc-rho");
    let mut proof = fold_shard_prove_legacy(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    proof.steps[0].fold.rlc_rhos[0][(0, 0)] += F::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-rlc-rho");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify_legacy(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered RLC rho must fail verification"
    );
}

#[test]
fn tamper_rlc_parent_y_scalars_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-rlc-parent-y-scalars");
    let mut proof = fold_shard_prove_legacy(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    assert!(
        !proof.steps[0].fold.rlc_parent.y_scalars.is_empty(),
        "fixture should produce a non-empty y_scalars table"
    );
    proof.steps[0].fold.rlc_parent.y_scalars[0] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-rlc-parent-y-scalars");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify_legacy(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered RLC parent y_scalars must fail verification"
    );
}

#[test]
fn tamper_dec_child_y_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-dec-child-y");
    let mut proof = fold_shard_prove_legacy(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let child0 = proof
        .steps
        .get_mut(0)
        .and_then(|s| s.fold.dec_children.get_mut(0))
        .expect("fixture should produce at least one DEC child");
    child0.y[0][0] += K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-dec-child-y");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify_legacy(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered DEC child y must fail verification"
    );
}

#[test]
fn tamper_batched_time_label_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-batched-time-label");
    let mut proof = fold_shard_prove_legacy(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    proof.steps[0].batched_time.labels[0] = b"bad/label".as_slice();

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-batched-time-label");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify_legacy(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered batched_time label must fail verification"
    );
}

#[test]
fn tamper_batched_time_degree_bound_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-batched-time-degree");
    let mut proof = fold_shard_prove_legacy(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    proof.steps[0].batched_time.degree_bounds[0] += 1;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-batched-time-degree");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    assert!(
        fold_shard_verify_legacy(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "tampered batched_time degree bound must fail verification"
    );
}

#[test]
fn tamper_batched_time_static_claim_sum_nonzero_fails() {
    let (params, ccs, step_bundle, acc_init, acc_wit_init, l, mixers) = build_single_chunk_inputs();

    let mut tr_prove = Poseidon2Transcript::new(b"soundness/tamper-batched-time-static-sum");
    let mut proof = fold_shard_prove_legacy(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");
    let step_inst = StepInstanceBundle::from(&step_bundle);
    let metas = RouteATimeClaimPlan::time_claim_metas_for_step(&step_inst, dims.d_sc, None);
    let static_idx = metas
        .iter()
        .enumerate()
        .find_map(|(i, m)| (!m.is_dynamic).then_some(i))
        .expect("fixture should include at least one static claim");

    proof.steps[0].batched_time.claimed_sums[static_idx] = K::ONE;

    let mut tr_verify = Poseidon2Transcript::new(b"soundness/tamper-batched-time-static-sum");
    let steps_public = [step_inst];
    assert!(
        fold_shard_verify_legacy(
            FoldingMode::PaperExact,
            &mut tr_verify,
            &params,
            &ccs,
            &steps_public,
            &acc_init,
            &proof,
            mixers,
        )
        .is_err(),
        "non-zero claimed_sum for static batched claim must fail verification"
    );
}
