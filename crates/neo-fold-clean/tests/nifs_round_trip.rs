mod support;

use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs;

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
        &prep.structure,
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        fresh,
        &running,
    )
    .expect("NIFS.P");

    let mut verifier_tr = Transcript::session();
    let verified_children = nifs::verify(
        &mut verifier_tr,
        &prep.params,
        &prep.structure,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &fresh_claims,
        &[],
        &proof,
    )
    .expect("NIFS.V");

    assert_eq!(verified_children, next_running.claims);
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
        &prep.structure,
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        fresh,
        &running,
    )
    .expect("NIFS.P");

    let mut baseline_tr = Transcript::session();
    nifs::verify(
        &mut baseline_tr,
        &prep.params,
        &prep.structure,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &fresh_claims,
        &[],
        &proof,
    )
    .expect("baseline NIFS.V must accept before tamper");

    support::mutate_ce_claim(&mut proof.pi_ccs.outputs[0]);

    let mut verifier_tr = Transcript::session();
    assert!(
        nifs::verify(
            &mut verifier_tr,
            &prep.params,
            &prep.structure,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &fresh_claims,
            &[],
            &proof,
        )
        .is_err(),
        "NIFS.V accepted a tampered Pi_CCS output"
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
        &prep.structure,
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        fresh,
        &running,
    )
    .expect("NIFS.P");

    let mut baseline_tr = Transcript::session();
    nifs::verify(
        &mut baseline_tr,
        &prep.params,
        &prep.structure,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &fresh_claims,
        &[],
        &proof,
    )
    .expect("baseline NIFS.V must accept before tamper");

    support::mutate_ce_claim(&mut proof.pi_dec.children[0]);

    let mut verifier_tr = Transcript::session();
    assert!(
        nifs::verify(
            &mut verifier_tr,
            &prep.params,
            &prep.structure,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &fresh_claims,
            &[],
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
        &prep.structure,
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
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
            &prep.structure,
            prep.mix_rhos_commits,
            prep.combine_b_pows,
            &fresh_claims,
            &[],
            &proof,
        )
        .is_err(),
        "NIFS.V accepted a Π_DEC proof with one child instead of k_rho children"
    );
}
