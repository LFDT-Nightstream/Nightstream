use std::sync::OnceLock;

use neo_fold_next::rv64im::{
    build_phase2_collapse_result, build_rv64im_accepted_proof_artifact,
    build_rv64im_eval_claim_witnesses_from_accepted_artifact, build_rv64im_opening_convergence_artifact_from_proof,
    build_rv64im_opening_convergence_artifact_from_witnesses, build_rv64im_opening_convergence_proof_from_witnesses,
    build_rv64im_phase0_binding_surface_from_accepted_artifact, derive_phase0_point, parity_source_cases,
    prove_rv64im_public_proof, verify_rv64im_opening_convergence_artifact,
    verify_rv64im_opening_convergence_artifact_from_proof, verify_rv64im_opening_convergence_proof,
    ClaimReductionError, FamilyEvalClaim, FamilyEvalClaimWitness, FamilyEvalPayload, FamilyEvalSchemaId,
    FinalOpeningError, Rv64imAcceptedProofArtifact, Rv64imOpeningConvergenceArtifact, Rv64imOpeningConvergenceProof,
    Rv64imPhase0BindingSurface, Rv64imProofInput,
};
use neo_math::K;
use p3_field::PrimeCharacteristicRing;

fn source_case(name: &str) -> neo_fold_next::rv64im::Rv64imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn proof_input(name: &str) -> Rv64imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

fn artifact() -> &'static Rv64imAcceptedProofArtifact {
    static ARTIFACT: OnceLock<Rv64imAcceptedProofArtifact> = OnceLock::new();
    ARTIFACT.get_or_init(|| {
        let proof = prove_rv64im_public_proof(&proof_input("aligned_negative_offset_roundtrip"))
            .expect("public proof should build for final-opening tests");
        build_rv64im_accepted_proof_artifact(&proof).expect("accepted artifact should build for final-opening tests")
    })
}

fn claim_witnesses() -> &'static Vec<FamilyEvalClaimWitness> {
    static WITNESSES: OnceLock<Vec<FamilyEvalClaimWitness>> = OnceLock::new();
    WITNESSES.get_or_init(|| {
        build_rv64im_eval_claim_witnesses_from_accepted_artifact(artifact())
            .expect("phase0 witnesses should build for final-opening tests")
    })
}

fn binding_surface() -> &'static Rv64imPhase0BindingSurface {
    static SURFACE: OnceLock<Rv64imPhase0BindingSurface> = OnceLock::new();
    SURFACE.get_or_init(|| build_rv64im_phase0_binding_surface_from_accepted_artifact(artifact()))
}

fn convergence_proof() -> &'static Rv64imOpeningConvergenceProof {
    static PROOF: OnceLock<Rv64imOpeningConvergenceProof> = OnceLock::new();
    PROOF.get_or_init(|| {
        build_rv64im_opening_convergence_proof_from_witnesses(binding_surface(), claim_witnesses())
            .expect("final opening convergence proof should build from real phase0 witnesses")
    })
}

fn convergence_artifact() -> &'static Rv64imOpeningConvergenceArtifact {
    static ARTIFACT: OnceLock<Rv64imOpeningConvergenceArtifact> = OnceLock::new();
    ARTIFACT.get_or_init(|| {
        build_rv64im_opening_convergence_artifact_from_proof(convergence_proof())
            .expect("compact convergence artifact should project from the full convergence proof")
    })
}

fn direct_convergence_artifact() -> &'static Rv64imOpeningConvergenceArtifact {
    static ARTIFACT: OnceLock<Rv64imOpeningConvergenceArtifact> = OnceLock::new();
    ARTIFACT.get_or_init(|| {
        build_rv64im_opening_convergence_artifact_from_witnesses(binding_surface(), claim_witnesses())
            .expect("compact convergence artifact should build directly from real phase0 witnesses")
    })
}

fn rebuild_compact_artifact_after_phase1_proof_mutation(artifact: &mut Rv64imOpeningConvergenceArtifact) {
    let rebuilt_phase2 = build_phase2_collapse_result(&artifact.phase1_results).expect("rebuild phase2");
    for (target, claim) in artifact
        .final_openings
        .iter_mut()
        .zip(rebuilt_phase2.reduced_claims.iter())
    {
        target.digest = target.expected_digest(claim);
    }
    artifact.phase2 = rebuilt_phase2;
    artifact.digest = artifact.expected_digest();
}

fn rebound_binding_digest_claim_witnesses() -> Vec<FamilyEvalClaimWitness> {
    let mut witnesses = claim_witnesses().clone();
    let original = &witnesses[0];
    let mut rebound_binding_digest = original.claim.binding_digest;
    rebound_binding_digest[0] ^= 1;
    let rebound_point = derive_phase0_point(
        &original.claim.opened_object,
        &original.claim.commitment_context,
        original.claim.payload.schema,
        original.claim.id.slot,
        rebound_binding_digest,
    );
    let rebound_payload = FamilyEvalPayload::new(
        original.claim.payload.schema,
        original
            .witness
            .evaluate_payload(&rebound_point)
            .expect("rebound point payload should evaluate"),
    )
    .expect("rebound payload should build");
    let rebound_claim = FamilyEvalClaim::new(
        original.claim.opened_object.clone(),
        original.claim.id.slot,
        original.claim.commitment_context,
        rebound_point,
        rebound_payload,
        rebound_binding_digest,
    )
    .expect("rebound claim should build");
    witnesses[0] = FamilyEvalClaimWitness::new(rebound_claim, original.witness.clone())
        .expect("rebound claim witness should stay self-consistent");
    witnesses
}

#[test]
fn final_bundle_roundtrips_from_real_phase0_witnesses() {
    let proof = convergence_proof();

    verify_rv64im_opening_convergence_proof(proof)
        .expect("final opening convergence proof should verify from carried artifacts alone");
    assert_eq!(proof.final_openings.len(), 6);
    assert_eq!(proof.phase2.reduced_claims.len(), 6);

    let rebuilt = build_rv64im_opening_convergence_proof_from_witnesses(binding_surface(), claim_witnesses())
        .expect("final opening convergence proof should rebuild deterministically");
    assert_eq!(*proof, rebuilt);
}

#[test]
fn final_opening_target_commitment_digest_matches_opened_object() {
    let target = &convergence_proof().final_openings[0];

    assert_eq!(
        target.opened_commitment.opened_object,
        target.reduced_claim.opened_object
    );
    assert_eq!(
        target.opened_commitment.digest,
        target.opened_commitment.expected_digest()
    );
    assert_eq!(target.opening_proof.digest, target.opening_proof.expected_digest());
}

#[test]
fn final_bundle_reconstructs_phase0_coverage() {
    let proof = convergence_proof();
    let reconstructed_phase0 = proof
        .phase1_results
        .iter()
        .flat_map(|result| result.bucket.claims.clone())
        .collect::<Vec<FamilyEvalClaim>>();

    assert_eq!(reconstructed_phase0, proof.phase0.claims);
    assert_eq!(
        proof
            .phase1_results
            .iter()
            .map(|result| result.bucket.schema)
            .collect::<Vec<_>>(),
        vec![
            FamilyEvalSchemaId::Stage1Rows,
            FamilyEvalSchemaId::Stage2RegisterReads,
            FamilyEvalSchemaId::Stage2RegisterWrites,
            FamilyEvalSchemaId::Stage2RamEvents,
            FamilyEvalSchemaId::Stage2TwistLinks,
            FamilyEvalSchemaId::Stage3Continuity,
        ]
    );
}

#[test]
fn final_bundle_rejects_missing_bucket_or_duplicate_bucket_claim() {
    let mut proof = convergence_proof().clone();
    proof.phase1_results.pop();

    let err = verify_rv64im_opening_convergence_proof(&proof)
        .expect_err("missing phase1 bucket must reject the final convergence proof");
    match err {
        FinalOpeningError::Phase1VerificationFailed(ClaimReductionError::ResultCountMismatch { expected, actual }) => {
            assert_eq!(expected, 6);
            assert_eq!(actual, 5);
        }
        other => panic!("unexpected final-opening phase1 coverage error: {other:?}"),
    }
}

#[test]
fn final_openings_follow_reduced_claim_order() {
    let proof = convergence_proof();

    assert_eq!(
        proof
            .final_openings
            .iter()
            .map(|target| target.reduced_claim.expected_digest())
            .collect::<Vec<_>>(),
        proof
            .phase2
            .reduced_claims
            .iter()
            .map(|claim| claim.expected_digest())
            .collect::<Vec<_>>()
    );

    let mut swapped = proof.clone();
    swapped.final_openings.swap(0, 1);
    let err =
        verify_rv64im_opening_convergence_proof(&swapped).expect_err("final openings must stay in reduced-claim order");
    match err {
        FinalOpeningError::UnexpectedReducedClaimAtIndex { index } => assert_eq!(index, 0),
        other => panic!("unexpected final-opening order error: {other:?}"),
    }
}

#[test]
fn final_bundle_digest_binds_phase0_digest() {
    let mut proof = convergence_proof().clone();
    proof.phase0.digest[0] ^= 1;

    let err = verify_rv64im_opening_convergence_proof(&proof)
        .expect_err("tampered phase0 digest must reject the final convergence proof");
    match err {
        FinalOpeningError::Phase0BundleMismatch { .. } => {}
        other => panic!("unexpected final-opening phase0 digest error: {other:?}"),
    }
}

#[test]
fn compact_convergence_artifact_roundtrips_from_full_proof() {
    let artifact = convergence_artifact();

    verify_rv64im_opening_convergence_artifact(artifact)
        .expect("compact convergence artifact should be self-consistent");
    verify_rv64im_opening_convergence_artifact_from_proof(artifact, convergence_proof())
        .expect("compact convergence artifact should match the full convergence proof");

    let rebuilt = build_rv64im_opening_convergence_artifact_from_proof(convergence_proof())
        .expect("compact convergence artifact should rebuild deterministically");
    assert_eq!(*artifact, rebuilt);
}

#[test]
fn direct_compact_convergence_artifact_matches_full_proof_projection() {
    let direct = direct_convergence_artifact();
    let projected = convergence_artifact();

    verify_rv64im_opening_convergence_artifact(direct)
        .expect("direct compact convergence artifact should be self-consistent");
    verify_rv64im_opening_convergence_artifact_from_proof(direct, convergence_proof())
        .expect("direct compact convergence artifact should match the full convergence proof");
    assert_eq!(*direct, *projected);
}

#[test]
fn compact_convergence_artifact_reconstructs_phase0_digest_from_phase1_results() {
    let artifact = convergence_artifact();

    assert_eq!(artifact.phase0_digest, convergence_proof().phase0.digest);
    assert_eq!(artifact.phase2, convergence_proof().phase2);
    assert_eq!(artifact.final_openings.len(), artifact.phase2.reduced_claims.len());
}

#[test]
fn compact_convergence_artifact_is_smaller_than_full_proof() {
    let full_len = bincode::serialize(convergence_proof())
        .expect("serialize full convergence proof")
        .len();
    let compact_len = bincode::serialize(convergence_artifact())
        .expect("serialize compact convergence artifact")
        .len();

    println!(
        "rv64im opening convergence sizes: full_proof={} compact_artifact={}",
        full_len, compact_len
    );

    assert!(compact_len < full_len);
}

#[test]
fn compact_convergence_artifact_rejects_rebound_final_target_tamper() {
    let mut artifact = convergence_artifact().clone();
    artifact.final_openings[0].opening_proof.digest[0] ^= 1;
    artifact.final_openings[0].digest = artifact.final_openings[0].expected_digest(&artifact.phase2.reduced_claims[0]);
    artifact.digest = artifact.expected_digest();

    let err = verify_rv64im_opening_convergence_artifact_from_proof(&artifact, convergence_proof())
        .expect_err("rebound compact final target tamper must fail against the full proof projection");
    match err {
        FinalOpeningError::ArtifactProjectionMismatch { .. } | FinalOpeningError::OpeningProofDigestMismatch { .. } => {
        }
        other => panic!("unexpected compact convergence artifact mismatch: {other:?}"),
    }
}

#[test]
fn compact_convergence_artifact_rejects_tampered_phase0_digest() {
    let mut artifact = convergence_artifact().clone();
    artifact.phase0_digest[0] ^= 1;
    artifact.digest = artifact.expected_digest();

    let err = verify_rv64im_opening_convergence_artifact(&artifact)
        .expect_err("tampered compact artifact phase0 digest must fail");
    match err {
        FinalOpeningError::ArtifactPhase0DigestMismatch { .. } => {}
        other => panic!("unexpected compact artifact phase0 digest error: {other:?}"),
    }
}

#[test]
fn compact_convergence_artifact_rejects_rebound_round_poly_forgery() {
    let mut artifact = convergence_artifact().clone();
    artifact.phase1_results[0].proof.round_polys[0].a0 += K::ONE;
    artifact.phase1_results[0].proof.digest = artifact.phase1_results[0].proof.expected_digest();
    rebuild_compact_artifact_after_phase1_proof_mutation(&mut artifact);

    let err = verify_rv64im_opening_convergence_artifact(&artifact)
        .expect_err("forged round poly with rebuilt digest chain must fail");
    assert!(
        format!("{err}").contains("phase1") || format!("{err}").contains("round") || format!("{err}").contains("poly")
    );
}

#[test]
fn compact_convergence_artifact_rejects_rebound_scalar_eval_forgery() {
    let mut artifact = convergence_artifact().clone();
    artifact.phase1_results[0].proof.scalar_evals_at_r_star[0] += K::ONE;
    artifact.phase1_results[0].proof.digest = artifact.phase1_results[0].proof.expected_digest();
    rebuild_compact_artifact_after_phase1_proof_mutation(&mut artifact);

    let err = verify_rv64im_opening_convergence_artifact(&artifact)
        .expect_err("forged scalar eval with rebuilt digest chain must fail");
    assert!(
        format!("{err}").contains("phase1") || format!("{err}").contains("scalar") || format!("{err}").contains("eval")
    );
}

#[test]
fn compact_convergence_artifact_rejects_rebound_eta_forgery() {
    let mut artifact = convergence_artifact().clone();
    artifact.phase1_results[0].proof.eta += K::ONE;
    artifact.phase1_results[0].proof.digest = artifact.phase1_results[0].proof.expected_digest();
    rebuild_compact_artifact_after_phase1_proof_mutation(&mut artifact);

    let err = verify_rv64im_opening_convergence_artifact(&artifact)
        .expect_err("forged eta with rebuilt digest chain must fail");
    assert!(format!("{err}").contains("phase1") || format!("{err}").contains("eta"));
}

#[test]
fn compact_convergence_artifact_rejects_rebound_rho_forgery() {
    let mut artifact = convergence_artifact().clone();
    artifact.phase1_results[0].proof.rho += K::ONE;
    artifact.phase1_results[0].proof.digest = artifact.phase1_results[0].proof.expected_digest();
    rebuild_compact_artifact_after_phase1_proof_mutation(&mut artifact);

    let err = verify_rv64im_opening_convergence_artifact(&artifact)
        .expect_err("forged rho with rebuilt digest chain must fail");
    assert!(format!("{err}").contains("phase1") || format!("{err}").contains("rho"));
}

#[test]
fn standalone_convergence_proof_rejects_rebound_binding_digest_forgery() {
    let forged = build_rv64im_opening_convergence_proof_from_witnesses(
        binding_surface(),
        &rebound_binding_digest_claim_witnesses(),
    )
    .expect("forged convergence proof should still build from rebound witnesses");

    verify_rv64im_opening_convergence_proof(&forged)
        .expect_err("standalone convergence proof must reject rebound binding-digest forgery");
}

#[test]
fn standalone_convergence_artifact_rejects_rebound_binding_digest_forgery() {
    let forged = build_rv64im_opening_convergence_artifact_from_witnesses(
        binding_surface(),
        &rebound_binding_digest_claim_witnesses(),
    )
    .expect("forged compact convergence artifact should still build from rebound witnesses");

    verify_rv64im_opening_convergence_artifact(&forged)
        .expect_err("standalone compact convergence artifact must reject rebound binding-digest forgery");
}
