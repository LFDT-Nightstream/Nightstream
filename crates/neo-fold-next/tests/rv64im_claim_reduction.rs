use std::sync::OnceLock;

use neo_fold_next::rv64im::{
    build_claim_reduction_buckets, build_claim_reduction_results_from_witnesses, build_rv64im_accepted_proof_artifact,
    build_rv64im_eval_claim_bundle_from_accepted_artifact, build_rv64im_eval_claim_witnesses_from_accepted_artifact,
    build_rv64im_phase0_binding_surface_from_accepted_artifact, derive_phase0_point, parity_source_cases,
    prove_rv64im_public_proof, verify_claim_reduction_result_with_binding_surface,
    verify_claim_reduction_results_with_binding_surface, ClaimReductionBucket, ClaimReductionError,
    ClaimReductionProof, ClaimReductionResult, CommitmentContextId, FamilyEvalClaim, FamilyEvalClaimId,
    FamilyEvalClaimWitness, FamilyEvalPayload, FamilyEvalSchemaId, OpenedAjtaiObjectId, QuadraticRoundPoly,
    Rv64imAcceptedProofArtifact, Rv64imEvalClaimBundle, Rv64imPhase0BindingSurface, Rv64imProofInput,
};
use neo_math::K;
use p3_field::PrimeCharacteristicRing;

fn digest(byte: u8) -> [u8; 32] {
    [byte; 32]
}

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
            .expect("public proof should build for claim-reduction tests");
        build_rv64im_accepted_proof_artifact(&proof).expect("accepted artifact should build for claim-reduction tests")
    })
}

fn eval_claim_bundle() -> &'static Rv64imEvalClaimBundle {
    static BUNDLE: OnceLock<Rv64imEvalClaimBundle> = OnceLock::new();
    BUNDLE.get_or_init(|| {
        build_rv64im_eval_claim_bundle_from_accepted_artifact(artifact())
            .expect("phase0 eval-claim bundle should build for claim-reduction tests")
    })
}

fn claim_witnesses() -> &'static Vec<neo_fold_next::rv64im::FamilyEvalClaimWitness> {
    static WITNESSES: OnceLock<Vec<neo_fold_next::rv64im::FamilyEvalClaimWitness>> = OnceLock::new();
    WITNESSES.get_or_init(|| {
        build_rv64im_eval_claim_witnesses_from_accepted_artifact(artifact())
            .expect("phase0 eval-claim witnesses should build for claim-reduction tests")
    })
}

fn phase0_binding_surface() -> &'static Rv64imPhase0BindingSurface {
    static SURFACE: OnceLock<Rv64imPhase0BindingSurface> = OnceLock::new();
    SURFACE.get_or_init(|| build_rv64im_phase0_binding_surface_from_accepted_artifact(artifact()))
}

fn claim_reduction_results() -> &'static Vec<ClaimReductionResult> {
    static RESULTS: OnceLock<Vec<ClaimReductionResult>> = OnceLock::new();
    RESULTS.get_or_init(|| {
        build_claim_reduction_results_from_witnesses(claim_witnesses())
            .expect("phase1 claim-reduction results should build from witnesses")
    })
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

fn stage1_bucket() -> ClaimReductionBucket {
    build_claim_reduction_buckets(eval_claim_bundle())
        .expect("phase1 buckets should build")
        .into_iter()
        .find(|bucket| bucket.schema == FamilyEvalSchemaId::Stage1Rows)
        .expect("stage1 bucket should exist")
}

fn zero_round_polys(point_arity: usize) -> Vec<QuadraticRoundPoly> {
    (0..point_arity)
        .map(|_| QuadraticRoundPoly {
            a0: K::ZERO,
            a1: K::ZERO,
            a2: K::ZERO,
        })
        .collect()
}

fn stage1_unified_claims(bucket: &ClaimReductionBucket, mismatched_payload: bool) -> Vec<FamilyEvalClaim> {
    let unified_point = vec![K::ZERO; bucket.point_arity()];
    let baseline_payload = bucket.claims[0].payload.clone();

    bucket
        .claims
        .iter()
        .enumerate()
        .map(|(index, claim)| {
            let mut unified = claim.clone();
            unified.point = unified_point.clone();
            unified.payload = baseline_payload.clone();
            if mismatched_payload && index == 1 {
                let coeff = &mut unified.payload.column_evals[0].coeffs[0];
                *coeff = if *coeff == K::ZERO { K::ONE } else { K::ZERO };
            }
            unified
        })
        .collect()
}

#[test]
fn claim_reduction_bucket_builder_partitions_phase0_bundle_by_schema() {
    let buckets = build_claim_reduction_buckets(eval_claim_bundle()).expect("phase1 buckets should build");

    assert_eq!(
        buckets
            .iter()
            .map(|bucket| bucket.schema)
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
    assert_eq!(
        buckets
            .iter()
            .map(|bucket| bucket.claims.len())
            .collect::<Vec<_>>(),
        vec![4, 1, 1, 1, 1, 1]
    );
    assert_eq!(buckets[0].payload_width(), 2);
    assert!(buckets[1..]
        .iter()
        .all(|bucket| bucket.payload_width() == 1));
}

#[test]
fn claim_reduction_bucket_digest_is_input_order_stable() {
    let original_claims = eval_claim_bundle()
        .claims
        .iter()
        .filter(|claim| claim.payload.schema == FamilyEvalSchemaId::Stage1Rows)
        .cloned()
        .collect::<Vec<_>>();
    let mut reversed_claims = original_claims.clone();
    reversed_claims.reverse();

    let original = ClaimReductionBucket::new(original_claims).expect("original stage1 bucket should build");
    let reversed = ClaimReductionBucket::new(reversed_claims).expect("reversed stage1 bucket should build");

    assert_eq!(original.claims, reversed.claims);
    assert_eq!(original.expected_digest(), reversed.expected_digest());
}

#[test]
fn claim_reduction_bucket_builder_rejects_mixed_point_arities_within_schema() {
    let mut claims = eval_claim_bundle().claims.clone();
    let index = claims
        .iter()
        .position(|claim| claim.payload.schema == FamilyEvalSchemaId::Stage1Rows && claim.id.slot == 3)
        .expect("stage1 slot 3 claim should exist");
    let claim = &mut claims[index];

    let opened_object = OpenedAjtaiObjectId::new(
        claim.opened_object.family,
        &claim.commitment_context,
        claim.opened_object.commitment_root_digest,
        claim.opened_object.layout_version,
        0,
    );
    claim.opened_object = opened_object.clone();
    claim.id = FamilyEvalClaimId::new(opened_object.digest, claim.id.slot);
    claim.point.clear();
    claim
        .validate()
        .expect("tampered claim should stay individually valid");

    let bundle = Rv64imEvalClaimBundle::new(claims).expect("bundle should still canonicalize");
    let expected_point_arity = eval_claim_bundle()
        .claims
        .iter()
        .find(|claim| claim.payload.schema == FamilyEvalSchemaId::Stage1Rows && claim.id.slot == 0)
        .expect("stage1 slot 0 claim should exist")
        .point
        .len();

    let err = build_claim_reduction_buckets(&bundle).expect_err("mixed point arity must reject");
    match err {
        ClaimReductionError::MixedPointArity { expected, actual, .. } => {
            assert!(
                (expected == expected_point_arity && actual == 0) || (expected == 0 && actual == expected_point_arity)
            );
        }
        other => panic!("unexpected mixed-point-arity error: {other:?}"),
    }
}

#[test]
fn claim_reduction_bucket_builder_rejects_mixed_commitment_contexts_within_schema() {
    let mut claims = eval_claim_bundle().claims.clone();
    let index = claims
        .iter()
        .position(|claim| claim.payload.schema == FamilyEvalSchemaId::Stage1Rows && claim.id.slot == 3)
        .expect("stage1 slot 3 claim should exist");
    let claim = &mut claims[index];

    let commitment_context = CommitmentContextId::new(digest(90), claim.commitment_context.module_shape_digest);
    let opened_object = OpenedAjtaiObjectId::new(
        claim.opened_object.family,
        &commitment_context,
        claim.opened_object.commitment_root_digest,
        claim.opened_object.layout_version,
        claim.opened_object.row_domain_log_size,
    );
    claim.commitment_context = commitment_context;
    claim.opened_object = opened_object.clone();
    claim.id = FamilyEvalClaimId::new(opened_object.digest, claim.id.slot);
    claim
        .validate()
        .expect("tampered claim should stay individually valid");

    let bundle = Rv64imEvalClaimBundle::new(claims).expect("bundle should still canonicalize");
    let err = build_claim_reduction_buckets(&bundle).expect_err("mixed commitment context must reject");
    assert_eq!(err, ClaimReductionError::MixedCommitmentContext { index: 3 });
}

#[test]
fn claim_reduction_result_rejects_missing_gamma_for_stage1_bucket() {
    let bucket = stage1_bucket();
    let unified_point = vec![K::ZERO; bucket.point_arity()];
    let unified_claims = stage1_unified_claims(&bucket, false);
    let mut proof = ClaimReductionProof {
        bucket_digest: bucket.expected_digest(),
        eta: K::ZERO,
        gamma: None,
        rho: K::ZERO,
        round_polys: zero_round_polys(unified_point.len()),
        scalar_evals_at_r_star: vec![K::ZERO; bucket.claims.len()],
        digest: [0; 32],
    };
    proof.digest = proof.expected_digest();

    let result = ClaimReductionResult {
        bucket,
        unified_point,
        unified_claims,
        proof,
    };

    let err = result
        .validate()
        .expect_err("stage1 proof without gamma must reject");
    assert_eq!(err, ClaimReductionError::MissingGamma { payload_width: 2 });
}

#[test]
fn claim_reduction_result_rejects_same_object_same_point_payload_mismatch() {
    let bucket = stage1_bucket();
    let unified_point = vec![K::ZERO; bucket.point_arity()];
    let unified_claims = stage1_unified_claims(&bucket, true);
    let mut proof = ClaimReductionProof {
        bucket_digest: bucket.expected_digest(),
        eta: K::ZERO,
        gamma: Some(K::ZERO),
        rho: K::ZERO,
        round_polys: zero_round_polys(unified_point.len()),
        scalar_evals_at_r_star: vec![K::ZERO; bucket.claims.len()],
        digest: [0; 32],
    };
    proof.digest = proof.expected_digest();

    let result = ClaimReductionResult {
        bucket: bucket.clone(),
        unified_point,
        unified_claims,
        proof,
    };

    let err = result
        .validate()
        .expect_err("same object + same point + different payloads must reject");
    assert_eq!(
        err,
        ClaimReductionError::SameObjectPayloadMismatch {
            opened_object_digest: bucket.claims[0].opened_object.digest,
        }
    );
}

#[test]
fn claim_reduction_results_roundtrip_from_real_witnesses() {
    let results = claim_reduction_results();

    verify_claim_reduction_results_with_binding_surface(results, claim_witnesses(), phase0_binding_surface())
        .expect("phase1 results should verify against their real witnesses");
    assert_eq!(
        results
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
    assert!(results[0].proof.gamma.is_some());
    assert!(results[1..]
        .iter()
        .all(|result| result.proof.gamma.is_none()));
}

#[test]
fn claim_reduction_verifier_rejects_tampered_scalar_eval() {
    let mut result = claim_reduction_results()[0].clone();
    result.proof.scalar_evals_at_r_star[0] += K::ONE;
    result.proof.digest = result.proof.expected_digest();

    let err = verify_claim_reduction_result_with_binding_surface(&result, claim_witnesses(), phase0_binding_surface())
        .expect_err("tampered scalar eval must reject");
    match err {
        ClaimReductionError::ScalarEvalMismatch { index, .. } => assert_eq!(index, 0),
        other => panic!("unexpected scalar-eval error: {other:?}"),
    }
}

#[test]
fn claim_reduction_verifier_rejects_tampered_unified_point() {
    let mut result = claim_reduction_results()[0].clone();
    result.unified_point[0] += K::ONE;
    for claim in &mut result.unified_claims {
        claim.point = result.unified_point.clone();
    }

    let err = verify_claim_reduction_result_with_binding_surface(&result, claim_witnesses(), phase0_binding_surface())
        .expect_err("tampered unified point must reject");
    assert_eq!(err, ClaimReductionError::UnifiedPointTranscriptMismatch);
}

#[test]
fn claim_reduction_results_verifier_rejects_rebound_binding_digest_witness_forgery() {
    let forged_witnesses = rebound_binding_digest_claim_witnesses();
    let forged_results = build_claim_reduction_results_from_witnesses(&forged_witnesses)
        .expect("forged claim-reduction results should still build from rebound witnesses");

    verify_claim_reduction_results_with_binding_surface(&forged_results, &forged_witnesses, phase0_binding_surface())
        .expect_err("claim-reduction verifier must reject rebound binding-digest witness forgery");
}
