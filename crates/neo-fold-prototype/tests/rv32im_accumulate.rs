use std::sync::OnceLock;

use neo_fold_prototype::rv32im::kernel::{
    build_claim_reduction_results_from_witnesses, build_phase2_collapse_result,
    build_rv32im_eval_claim_witnesses_from_accepted_artifact, verify_phase2_collapse_result, ClaimReductionResult,
    FamilyEvalSchemaId, Phase2CollapseError, Phase2CollapseResult,
};
use neo_fold_prototype::rv32im::Rv32imAcceptedProofArtifact;
use neo_math::K;
use p3_field::PrimeCharacteristicRing;

#[path = "support/rv32im_phase0_memory.rs"]
mod rv32im_phase0_memory;

fn artifact() -> &'static Rv32imAcceptedProofArtifact {
    static ARTIFACT: OnceLock<Rv32imAcceptedProofArtifact> = OnceLock::new();
    ARTIFACT.get_or_init(|| rv32im_phase0_memory::phase0_memory_artifact().clone())
}

fn phase1_results() -> &'static Vec<ClaimReductionResult> {
    static RESULTS: OnceLock<Vec<ClaimReductionResult>> = OnceLock::new();
    RESULTS.get_or_init(|| {
        let witnesses = build_rv32im_eval_claim_witnesses_from_accepted_artifact(artifact())
            .expect("phase0 eval-claim witnesses should build for phase2 tests");
        build_claim_reduction_results_from_witnesses(&witnesses).expect("phase1 results should build for phase2 tests")
    })
}

fn phase2_result() -> &'static Phase2CollapseResult {
    static RESULT: OnceLock<Phase2CollapseResult> = OnceLock::new();
    RESULT.get_or_init(|| {
        build_phase2_collapse_result(phase1_results()).expect("phase2 result should build from phase1 results")
    })
}

#[test]
fn phase2_collapse_roundtrips_from_real_phase1_results() {
    let result = phase2_result();

    verify_phase2_collapse_result(result, phase1_results())
        .expect("phase2 result should verify against the canonical phase1 results");
    assert_eq!(result.reduced_claims.len(), 6);
    assert_eq!(result.records.len(), 6);
    assert_eq!(
        result
            .reduced_claims
            .iter()
            .map(|claim| claim.payload.schema)
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

    let rebuilt = build_phase2_collapse_result(phase1_results()).expect("phase2 rebuild should stay stable");
    assert_eq!(*result, rebuilt);
}

#[test]
fn phase2_stage1_rows_collapse_preserves_bucket_order() {
    let phase1_stage1 = &phase1_results()[0];
    let reduced = &phase2_result().reduced_claims[0];

    assert_eq!(reduced.payload.schema, FamilyEvalSchemaId::Stage1Rows);
    assert_eq!(reduced.source_claim_ids.len(), 4);
    assert_eq!(
        reduced.source_claim_ids,
        phase1_stage1
            .unified_claims
            .iter()
            .map(|claim| claim.id)
            .collect::<Vec<_>>()
    );
    assert_eq!(reduced.point, phase1_stage1.unified_point);
    assert_eq!(reduced.payload, phase1_stage1.unified_claims[0].payload);
}

#[test]
fn phase2_singleton_families_remain_one_to_one() {
    let result = phase2_result();

    assert!(result.reduced_claims[1..]
        .iter()
        .all(|claim| claim.source_claim_ids.len() == 1));
    assert_eq!(
        result.reduced_claims[1..]
            .iter()
            .map(|claim| claim.payload.schema)
            .collect::<Vec<_>>(),
        vec![
            FamilyEvalSchemaId::Stage2RegisterReads,
            FamilyEvalSchemaId::Stage2RegisterWrites,
            FamilyEvalSchemaId::Stage2RamEvents,
            FamilyEvalSchemaId::Stage2TwistLinks,
            FamilyEvalSchemaId::Stage3Continuity,
        ]
    );
}

#[test]
fn phase2_rejects_conflicting_same_object_same_point_payloads() {
    let mut results = phase1_results().clone();
    results[0].unified_claims[1].payload.column_evals[0].coeffs[0] += K::ONE;

    let err = build_phase2_collapse_result(&results).expect_err("conflicting Stage1 payloads must reject");
    match err {
        Phase2CollapseError::InvalidPhase1Result { index, .. } => assert_eq!(index, 0),
        other => panic!("unexpected phase2 conflict error: {other:?}"),
    }
}

#[test]
fn phase2_rejects_duplicate_source_claim_ids_across_groups() {
    let mut result = phase2_result().clone();
    let duplicate = result.reduced_claims[0].source_claim_ids[0];
    result.reduced_claims[1].source_claim_ids[0] = duplicate;

    let err = verify_phase2_collapse_result(&result, phase1_results())
        .expect_err("duplicate source ids across groups must reject");
    assert_eq!(err, Phase2CollapseError::DuplicateSourceClaimId { claim_id: duplicate });
}
