#[path = "../support/mod.rs"]
mod support;

use neo_ccs::Mat;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::{nifs, pi_ccs, pi_rlc};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

#[test]
fn nifs_prove_verify_round_trip_matches_children() {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, 7), support::toy_instance(&prep, 11)];
    let fresh_claims = fresh.iter().map(|i| i.claim.clone()).collect::<Vec<_>>();
    let running = RunningInstance::default();

    let mut prover_tr = Transcript::session();
    let (next_running, proof) = nifs::prove(
        &mut prover_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P");

    let mut verifier_tr = Transcript::session();
    let verified_children = nifs::verify(
        &mut verifier_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &proof,
    )
    .expect("NIFS.V");

    assert_eq!(verified_children.claims, next_running.claims);
    assert_eq!(verified_children.parent_authority, next_running.parent_authority);
}

#[test]
fn nifs_cpu_adapter_matches_prover_contract() {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, 43), support::toy_instance(&prep, 47)];
    let fresh_claims = fresh.iter().map(|i| i.claim.clone()).collect::<Vec<_>>();
    let running = RunningInstance::default();

    let mut adapter = nifs::OptimizedCpuNifsProver;
    let mut prover_tr = Transcript::session();
    let (next_running, proof) = nifs::prove_with_adapter(
        &mut adapter,
        &mut prover_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P CPU adapter");

    let mut verifier_tr = Transcript::session();
    let verified_children = nifs::verify(
        &mut verifier_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &proof,
    )
    .expect("NIFS.V");

    assert_eq!(verified_children.claims, next_running.claims);
    assert_eq!(verified_children.parent_authority, next_running.parent_authority);
}

#[test]
fn pi_rlc_rho_derivation_replays_after_pi_ccs() {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, 53), support::toy_instance(&prep, 59)];
    let fresh_claims = fresh.iter().map(|i| i.claim.clone()).collect::<Vec<_>>();
    let running = RunningInstance::default();

    let mut prover_tr = Transcript::session();
    let (_, proof) = nifs::prove(
        &mut prover_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P");

    let mut replay_tr = Transcript::session();
    let ccs_outputs = pi_ccs::verify(
        &mut replay_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof.pi_ccs,
    )
    .expect("Pi_CCS replay");
    let rhos = pi_rlc::derive_rhos_for_inputs(&mut replay_tr, &prep.params, &ccs_outputs).expect("Pi_RLC rhos");
    let mix = prep.mix_rhos_commits();
    let ok = neo_fold_clean::engine::optimized::verify_pi_rlc(
        &prep.params,
        prep.structure(),
        &rhos,
        &ccs_outputs,
        &proof.pi_rlc.combined,
        |zs, cs| mix(zs, cs),
    )
    .expect("Pi_RLC replay verify");
    assert!(ok, "replayed Pi_RLC rhos must verify the prover parent");

    let mut verifier_tr = Transcript::session();
    nifs::verify(
        &mut verifier_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &proof,
    )
    .expect("NIFS.V");
}

#[test]
fn nifs_verify_rejects_tampered_running_parent_authority() {
    let prep = support::toy_preprocessing();
    let first = vec![support::toy_instance(&prep, 23)];
    let mut tr0 = Transcript::session();
    let (running, _) = nifs::prove(
        &mut tr0,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        first,
        &RunningInstance::default(),
    )
    .expect("first NIFS.P");
    assert!(
        running.parent_authority.is_some(),
        "non-empty running must carry Pi_RLC parent"
    );

    let second = vec![support::toy_instance(&prep, 29)];
    let second_claims = second.iter().map(|i| i.claim.clone()).collect::<Vec<_>>();
    let mut tr1 = Transcript::session();
    let (_, proof) = nifs::prove(
        &mut tr1,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        second,
        &running,
    )
    .expect("second NIFS.P");

    let mut baseline_tr = Transcript::session();
    nifs::verify(
        &mut baseline_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &second_claims,
        &running,
        &proof,
    )
    .expect("baseline NIFS.V accepts");

    let mut bad_running = running.clone();
    support::mutate_ce_claim(
        bad_running
            .parent_authority
            .as_mut()
            .expect("parent authority"),
    );
    let mut verifier_tr = Transcript::session();
    assert!(
        nifs::verify(
            &mut verifier_tr,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            &second_claims,
            &bad_running,
            &proof,
        )
        .is_err(),
        "NIFS.V accepted a running accumulator with a tampered Pi_RLC parent authority"
    );
}

#[test]
fn nifs_verify_rejects_running_child_changed_under_same_parent_authority() {
    let prep = support::toy_preprocessing();
    let first = vec![support::toy_instance(&prep, 31)];
    let mut tr0 = Transcript::session();
    let (running, _) = nifs::prove(
        &mut tr0,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        first,
        &RunningInstance::default(),
    )
    .expect("first NIFS.P");

    let second = vec![support::toy_instance(&prep, 37)];
    let second_claims = second.iter().map(|i| i.claim.clone()).collect::<Vec<_>>();
    let mut tr1 = Transcript::session();
    let (_, proof) = nifs::prove(
        &mut tr1,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        second,
        &running,
    )
    .expect("second NIFS.P");

    let mut bad_running = running.clone();
    support::mutate_ce_claim(&mut bad_running.claims[0]);
    let mut verifier_tr = Transcript::session();
    assert!(
        nifs::verify(
            &mut verifier_tr,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            &second_claims,
            &bad_running,
            &proof,
        )
        .is_err(),
        "NIFS.V accepted changed running children under the original parent authority"
    );
}

#[test]
fn nifs_verify_rejects_tampered_pi_ccs_output() {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, 13)];
    let fresh_claims = fresh.iter().map(|i| i.claim.clone()).collect::<Vec<_>>();
    let running = RunningInstance::default();

    let mut prover_tr = Transcript::session();
    let (_, mut proof) = nifs::prove(
        &mut prover_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P");

    let mut baseline_tr = Transcript::session();
    nifs::verify(
        &mut baseline_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &proof,
    )
    .expect("baseline NIFS.V must accept before tamper");

    support::mutate_ce_claim(&mut proof.pi_ccs.outputs[0]);

    let mut verifier_tr = Transcript::session();
    assert!(
        nifs::verify(
            &mut verifier_tr,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            &fresh_claims,
            &running,
            &proof,
        )
        .is_err(),
        "NIFS.V accepted a tampered Pi_CCS output"
    );
}

#[test]
fn pi_ccs_verify_rejects_output_y_not_bound_to_sumcheck_terminal_value() {
    let m = D;
    let mut a = Mat::zero(1, m, F::ZERO);
    a[(0, 1)] = F::ONE;
    let mut b = Mat::zero(1, m, F::ZERO);
    b[(0, 0)] = F::ONE;
    let mut c = Mat::zero(1, m, F::ZERO);
    c[(0, 1)] = F::ONE;
    let r1cs = R1cs { a, b, c, m_in: D };
    let prep = direct_ccs::preprocess_seeded(&r1cs, 41).expect("preprocess nontrivial R1CS");
    let mut z = vec![F::ZERO; prep.structure().m];
    z[0] = F::ONE;
    z[1] = F::ONE;
    let fresh = vec![direct_ccs::build_instance(&prep, &r1cs, &z).expect("fresh instance")];
    let fresh_claims = fresh.iter().map(|i| i.claim.clone()).collect::<Vec<_>>();
    let running = RunningInstance::default();

    let mut prover_tr = Transcript::session();
    let (_, mut proof) = nifs::prove(
        &mut prover_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P");

    let mut baseline_tr = Transcript::session();
    pi_ccs::verify(
        &mut baseline_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof.pi_ccs,
    )
    .expect("baseline Pi_CCS.V must accept before tamper");

    let y0 = proof.pi_ccs.outputs[0]
        .y_ring
        .get_mut(0)
        .and_then(|row| row.get_mut(0))
        .expect("nontrivial Pi_CCS output must carry a y_ring constant term");
    *y0 += K::ONE;
    proof.pi_ccs.outputs[0].ct[0] += K::ONE;

    let mut verifier_tr = Transcript::session();
    assert!(
        pi_ccs::verify(
            &mut verifier_tr,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            &fresh_claims,
            &running,
            &proof.pi_ccs,
        )
        .is_err(),
        "Pi_CCS.V accepted an output y value that is not bound to the verified sumcheck terminal value"
    );
}

#[test]
fn nifs_verify_rejects_tampered_pi_dec_child() {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, 17)];
    let fresh_claims = fresh.iter().map(|i| i.claim.clone()).collect::<Vec<_>>();
    let running = RunningInstance::default();

    let mut prover_tr = Transcript::session();
    let (_, mut proof) = nifs::prove(
        &mut prover_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P");

    let mut baseline_tr = Transcript::session();
    nifs::verify(
        &mut baseline_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &running,
        &proof,
    )
    .expect("baseline NIFS.V must accept before tamper");

    support::mutate_ce_claim(&mut proof.pi_dec.children[0]);

    let mut verifier_tr = Transcript::session();
    assert!(
        nifs::verify(
            &mut verifier_tr,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            &fresh_claims,
            &running,
            &proof,
        )
        .is_err(),
        "NIFS.V accepted a tampered Pi_DEC child"
    );
}

#[test]
fn nifs_verify_rejects_pi_dec_child_count_drift() {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, 19)];
    let fresh_claims = fresh.iter().map(|i| i.claim.clone()).collect::<Vec<_>>();
    let running = RunningInstance::default();

    let mut prover_tr = Transcript::session();
    let (_, mut proof) = nifs::prove(
        &mut prover_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P");

    proof.pi_dec.children = vec![proof.pi_rlc.combined.clone()];

    let mut verifier_tr = Transcript::session();
    assert!(
        nifs::verify(
            &mut verifier_tr,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            &fresh_claims,
            &running,
            &proof,
        )
        .is_err(),
        "NIFS.V accepted a Π_DEC proof with one child instead of k_rho children"
    );
}
