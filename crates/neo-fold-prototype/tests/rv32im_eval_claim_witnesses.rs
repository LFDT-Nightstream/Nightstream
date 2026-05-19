use std::collections::BTreeSet;
use std::sync::OnceLock;

use neo_fold_prototype::rv32im::kernel::{
    build_rv32im_eval_claim_bundle_from_accepted_artifact, build_rv32im_eval_claim_witnesses_from_accepted_artifact,
    build_stage1_claim_witnesses, build_stage2_claim_witnesses, build_stage3_claim_witness,
    verify_rv32im_eval_claim_bundle_from_accepted_artifact, EvalClaimError, FamilyEvalClaimWitness, FamilyEvalSchemaId,
    OpenedAjtaiObjectWitness,
};
use neo_fold_prototype::rv32im::{
    build_rv32im_accepted_proof_artifact, parity_source_cases, prove_rv32im_public_proof, Rv32imAcceptedProofArtifact,
    Rv32imProofInput,
};
use neo_math::{from_complex, F, K};
use p3_field::PrimeCharacteristicRing;

#[path = "support/rv32im_phase0_memory.rs"]
mod rv32im_phase0_memory;

fn digest(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn k(real: u64, imag: u64) -> K {
    from_complex(F::from_u64(real), F::from_u64(imag))
}

fn source_case(name: &str) -> neo_fold_prototype::rv32im::Rv32imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn proof_input(name: &str) -> Rv32imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv32imProofInput { source, max_steps }
}

fn stage1_artifact() -> &'static Rv32imAcceptedProofArtifact {
    static ARTIFACT: OnceLock<Rv32imAcceptedProofArtifact> = OnceLock::new();
    ARTIFACT.get_or_init(|| {
        let proof = prove_rv32im_public_proof(&proof_input("native_add_chain_x0_ecall"))
            .expect("public proof should build for stage1 claim witness tests");
        build_rv32im_accepted_proof_artifact(&proof)
            .expect("accepted artifact should build for stage1 claim witness tests")
    })
}

fn aligned_memory_artifact() -> &'static Rv32imAcceptedProofArtifact {
    static ARTIFACT: OnceLock<Rv32imAcceptedProofArtifact> = OnceLock::new();
    ARTIFACT.get_or_init(|| rv32im_phase0_memory::phase0_memory_artifact().clone())
}

fn control_flow_artifact() -> &'static Rv32imAcceptedProofArtifact {
    static ARTIFACT: OnceLock<Rv32imAcceptedProofArtifact> = OnceLock::new();
    ARTIFACT.get_or_init(|| {
        let proof = prove_rv32im_public_proof(&proof_input("control_flow_ecall_only"))
            .expect("public proof should build for empty-family claim witness tests");
        build_rv32im_accepted_proof_artifact(&proof)
            .expect("accepted artifact should build for empty-family claim witness tests")
    })
}

fn stage1_claim_witnesses() -> Vec<FamilyEvalClaimWitness> {
    let artifact = stage1_artifact();
    build_stage1_claim_witnesses(&artifact.stage_claims.claims.stage1, &artifact.stage1)
        .expect("stage1 phase0 claim witnesses should build")
}

fn stage2_claim_witnesses() -> Vec<FamilyEvalClaimWitness> {
    let artifact = aligned_memory_artifact();
    build_stage2_claim_witnesses(&artifact.stage_claims.claims.stage2, &artifact.stage2)
        .expect("stage2 phase0 claim witnesses should build")
}

fn stage3_claim_witness() -> FamilyEvalClaimWitness {
    let artifact = aligned_memory_artifact();
    build_stage3_claim_witness(&artifact.stage_claims.claims.stage3, &artifact.stage3)
        .expect("stage3 phase0 claim witness should build")
}

#[test]
fn stage1_claim_witnesses_emit_four_slot_bound_claims() {
    let claims = stage1_claim_witnesses();

    assert_eq!(claims.len(), 4);
    assert_eq!(
        claims
            .iter()
            .map(|claim| claim.claim.id.slot)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert!(claims
        .iter()
        .all(|claim| claim.claim.payload.schema == FamilyEvalSchemaId::Stage1Rows));
    assert!(claims
        .iter()
        .all(|claim| claim.claim.payload.column_evals.len() == 1));

    let first_object = claims[0].claim.opened_object.digest;
    assert!(claims
        .iter()
        .all(|claim| claim.claim.opened_object.digest == first_object));

    for left in 0..claims.len() {
        for right in (left + 1)..claims.len() {
            assert_ne!(
                claims[left].claim.point, claims[right].claim.point,
                "slot-bound points should stay distinct"
            );
        }
    }
}

#[test]
fn stage1_claim_witness_payload_matches_witness_evaluation() {
    let claims = stage1_claim_witnesses();

    for claim in &claims {
        let expected = claim
            .witness
            .evaluate_payload(&claim.claim.point)
            .expect("witness evaluation should succeed");
        assert_eq!(claim.claim.payload.column_evals, expected);
    }
}

#[test]
fn stage1_claim_witness_rejects_tampered_payload() {
    let mut claim_witness = stage1_claim_witnesses()
        .into_iter()
        .next()
        .expect("stage1 claim witness");
    claim_witness.claim.payload.column_evals[0].coeffs[0] = k(77, 0);

    let err = FamilyEvalClaimWitness::new(claim_witness.claim, claim_witness.witness)
        .expect_err("tampered payload must reject");

    assert_eq!(
        err,
        EvalClaimError::WitnessPayloadMismatch {
            schema: FamilyEvalSchemaId::Stage1Rows,
            slot: 0,
        }
    );
}

#[test]
fn opened_object_witness_rejects_tampered_commitment_root() {
    let base = stage1_claim_witnesses()
        .into_iter()
        .next()
        .expect("stage1 claim witness")
        .witness;
    let mut opened_object = base.opened_object.clone();
    opened_object.commitment_root_digest = digest(99);

    let err = OpenedAjtaiObjectWitness::new(
        opened_object,
        base.commitment_context.clone(),
        base.packed_columns.clone(),
        base.commitment_vector.clone(),
    )
    .expect_err("tampered commitment root must reject");

    match err {
        EvalClaimError::WitnessCommitmentRootMismatch { .. } => {}
        other => panic!("unexpected witness error: {other:?}"),
    }
}

#[test]
fn stage2_singleton_claim_witnesses_emit_slot0_per_family() {
    let claims = stage2_claim_witnesses();

    assert_eq!(claims.len(), 4);
    assert_eq!(
        claims
            .iter()
            .map(|claim| claim.claim.id.slot)
            .collect::<Vec<_>>(),
        vec![0, 0, 0, 0]
    );
    assert_eq!(
        claims
            .iter()
            .map(|claim| claim.claim.payload.schema)
            .collect::<Vec<_>>(),
        vec![
            FamilyEvalSchemaId::Stage2RegisterReads,
            FamilyEvalSchemaId::Stage2RegisterWrites,
            FamilyEvalSchemaId::Stage2RamEvents,
            FamilyEvalSchemaId::Stage2TwistLinks,
        ]
    );
    assert!(claims
        .iter()
        .all(|claim| claim.claim.payload.column_evals.len() == 1));

    let object_count = claims
        .iter()
        .map(|claim| claim.claim.opened_object.digest)
        .collect::<BTreeSet<_>>()
        .len();
    assert_eq!(
        object_count, 4,
        "singleton family claims should each bind a distinct opened object"
    );
}

#[test]
fn stage2_singleton_claim_witness_payloads_match_witness_evaluation() {
    let claims = stage2_claim_witnesses();

    for claim in &claims {
        let expected = claim
            .witness
            .evaluate_payload(&claim.claim.point)
            .expect("witness evaluation should succeed");
        assert_eq!(claim.claim.payload.column_evals, expected);
    }
}

#[test]
fn stage2_singleton_claims_still_emit_when_some_families_are_empty() {
    let artifact = control_flow_artifact();
    let surface = &artifact.stage_claims.claims.stage2.claim;
    assert!(
        surface.register_read_count == 0
            || surface.register_write_count == 0
            || surface.ram_event_count == 0
            || surface.twist_link_count == 0,
        "control-flow fixture should exercise at least one empty Stage2 family"
    );

    let claims = build_stage2_claim_witnesses(&artifact.stage_claims.claims.stage2, &artifact.stage2)
        .expect("stage2 phase0 claim witnesses should still build when some families are empty");

    let expected_nonempty_count = [
        surface.register_read_count,
        surface.register_write_count,
        surface.ram_event_count,
        surface.twist_link_count,
    ]
    .into_iter()
    .filter(|count| *count != 0)
    .count();
    let expected_nonempty_schemas = [
        (surface.register_read_count, FamilyEvalSchemaId::Stage2RegisterReads),
        (surface.register_write_count, FamilyEvalSchemaId::Stage2RegisterWrites),
        (surface.ram_event_count, FamilyEvalSchemaId::Stage2RamEvents),
        (surface.twist_link_count, FamilyEvalSchemaId::Stage2TwistLinks),
    ]
    .into_iter()
    .filter_map(|(count, schema)| (count != 0).then_some(schema))
    .collect::<Vec<_>>();
    assert_eq!(claims.len(), expected_nonempty_count);
    assert!(claims.iter().all(|claim| claim.claim.id.slot == 0));
    assert_eq!(
        claims
            .iter()
            .map(|claim| claim.claim.payload.schema)
            .collect::<Vec<_>>(),
        expected_nonempty_schemas
    );
}

#[test]
fn stage3_continuity_claim_witness_emits_slot0_and_matches_payload() {
    let claim = stage3_claim_witness();

    assert_eq!(claim.claim.id.slot, 0);
    assert_eq!(claim.claim.payload.schema, FamilyEvalSchemaId::Stage3Continuity);
    assert_eq!(claim.claim.payload.column_evals.len(), 1);

    let expected = claim
        .witness
        .evaluate_payload(&claim.claim.point)
        .expect("witness evaluation should succeed");
    assert_eq!(claim.claim.payload.column_evals, expected);
}

#[test]
fn accepted_artifact_eval_claim_witnesses_cover_all_phase0_families() {
    let artifact = aligned_memory_artifact();
    let claims = build_rv32im_eval_claim_witnesses_from_accepted_artifact(artifact)
        .expect("accepted artifact should build the full phase0 eval-claim witness set");

    assert_eq!(claims.len(), 9);
    assert_eq!(
        claims
            .iter()
            .map(|claim| claim.claim.payload.schema)
            .collect::<Vec<_>>(),
        vec![
            FamilyEvalSchemaId::Stage1Rows,
            FamilyEvalSchemaId::Stage1Rows,
            FamilyEvalSchemaId::Stage1Rows,
            FamilyEvalSchemaId::Stage1Rows,
            FamilyEvalSchemaId::Stage2RegisterReads,
            FamilyEvalSchemaId::Stage2RegisterWrites,
            FamilyEvalSchemaId::Stage2RamEvents,
            FamilyEvalSchemaId::Stage2TwistLinks,
            FamilyEvalSchemaId::Stage3Continuity,
        ]
    );
    assert_eq!(
        claims
            .iter()
            .map(|claim| claim.claim.id.slot)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 0, 0, 0, 0, 0]
    );
}

#[test]
fn accepted_artifact_eval_claim_bundle_matches_claim_witness_projection() {
    let artifact = aligned_memory_artifact();
    let witnesses = build_rv32im_eval_claim_witnesses_from_accepted_artifact(artifact)
        .expect("accepted artifact should build the full phase0 eval-claim witness set");
    let bundle = build_rv32im_eval_claim_bundle_from_accepted_artifact(artifact)
        .expect("accepted artifact should build the canonical phase0 claim bundle");

    let expected_claims = witnesses
        .into_iter()
        .map(|claim| claim.claim)
        .collect::<Vec<_>>();
    assert_eq!(bundle.claims, expected_claims);
    assert_ne!(bundle.digest, [0; 32]);
    verify_rv32im_eval_claim_bundle_from_accepted_artifact(artifact, &bundle)
        .expect("accepted artifact should verify the canonical phase0 claim bundle");
}

#[test]
fn accepted_artifact_eval_claim_bundle_rejects_tampered_claim_payload() {
    let artifact = aligned_memory_artifact();
    let mut bundle = build_rv32im_eval_claim_bundle_from_accepted_artifact(artifact)
        .expect("accepted artifact should build the canonical phase0 claim bundle");
    bundle.claims[0].payload.column_evals[0].coeffs[0] = k(31337, 0);

    let err = verify_rv32im_eval_claim_bundle_from_accepted_artifact(artifact, &bundle)
        .expect_err("tampered phase0 claim bundle must reject");

    assert!(
        err.to_string()
            .contains("RV32IM Phase 0 eval-claim bundle digest mismatch"),
        "unexpected error: {err}"
    );
}
