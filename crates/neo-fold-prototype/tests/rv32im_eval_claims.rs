use neo_fold_prototype::rv32im::kernel::{
    AjtaiFamilyKind, CommitmentContextId, EvalClaimError, FamilyEvalClaim, FamilyEvalClaimId, FamilyEvalPayload,
    FamilyEvalSchemaId, OpenedAjtaiObjectId, OpeningClaimAccumulator, PackedColumnEval, Rv32imEvalClaimBundle,
};
use neo_fold_prototype::rv32im::{
    build_rv32im_accepted_proof_artifact, parity_source_cases, prove_rv32im_public_proof, Rv32imProofInput,
};
use neo_math::{from_complex, F, K};
use p3_field::PrimeCharacteristicRing;

fn digest(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn k(real: u64, imag: u64) -> K {
    from_complex(F::from_u64(real), F::from_u64(imag))
}

fn payload(schema: FamilyEvalSchemaId, seed: u64) -> FamilyEvalPayload {
    let column_evals = (0..schema.packed_column_count())
        .map(|column_idx| PackedColumnEval {
            coeffs: std::array::from_fn(|coeff_idx| {
                k(seed + column_idx as u64 + coeff_idx as u64, seed + coeff_idx as u64)
            }),
        })
        .collect();
    FamilyEvalPayload::new(schema, column_evals).expect("valid phase0 payload")
}

fn claim(
    schema: FamilyEvalSchemaId,
    commitment_context: CommitmentContextId,
    commitment_root_digest: [u8; 32],
    layout_version: u64,
    row_domain_log_size: u32,
    slot: u32,
    binding_digest: [u8; 32],
    seed: u64,
) -> FamilyEvalClaim {
    let opened_object = OpenedAjtaiObjectId::new(
        schema.family_kind(),
        &commitment_context,
        commitment_root_digest,
        layout_version,
        row_domain_log_size,
    );
    let point = (0..row_domain_log_size as usize)
        .map(|idx| k(seed + idx as u64, seed + 100 + idx as u64))
        .collect();
    FamilyEvalClaim::new(
        opened_object,
        slot,
        commitment_context,
        point,
        payload(schema, seed + 1000),
        binding_digest,
    )
    .expect("valid phase0 claim")
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

#[test]
fn family_eval_payload_rejects_wrong_packed_column_count() {
    let err = FamilyEvalPayload::new(FamilyEvalSchemaId::Stage1Rows, Vec::new())
        .expect_err("stage1 payload must carry exactly one packed-column evaluation");

    assert_eq!(
        err,
        EvalClaimError::PayloadWidthMismatch {
            schema: FamilyEvalSchemaId::Stage1Rows,
            expected: 1,
            actual: 0,
        }
    );
}

#[test]
fn opening_claim_accumulator_rejects_family_schema_mismatch() {
    let commitment_context = CommitmentContextId::new(digest(1), digest(2));
    let opened_object = OpenedAjtaiObjectId::new(AjtaiFamilyKind::Stage1Rows, &commitment_context, digest(3), 1, 2);
    let opened_object_digest = opened_object.digest;
    let claim = FamilyEvalClaim {
        opened_object,
        id: FamilyEvalClaimId::new(opened_object_digest, 0),
        commitment_context,
        point: vec![k(7, 0), k(8, 0)],
        payload: payload(FamilyEvalSchemaId::Stage2RegisterReads, 9),
        binding_digest: digest(4),
    };

    let mut accumulator = OpeningClaimAccumulator::default();
    let err = accumulator
        .insert(claim)
        .expect_err("accumulator must reject family/schema mismatches");

    assert_eq!(
        err,
        EvalClaimError::FamilySchemaMismatch {
            family: AjtaiFamilyKind::Stage1Rows,
            schema: FamilyEvalSchemaId::Stage2RegisterReads,
        }
    );
}

#[test]
fn opening_claim_accumulator_rejects_opened_object_commitment_context_mismatch() {
    let commitment_context = CommitmentContextId::new(digest(5), digest(6));
    let other_context = CommitmentContextId::new(digest(7), digest(8));
    let opened_object = OpenedAjtaiObjectId::new(AjtaiFamilyKind::Stage1Rows, &commitment_context, digest(9), 1, 2);
    let opened_object_digest = opened_object.digest;
    let claim = FamilyEvalClaim {
        opened_object,
        id: FamilyEvalClaimId::new(opened_object_digest, 0),
        commitment_context: other_context,
        point: vec![k(1, 0), k(2, 0)],
        payload: payload(FamilyEvalSchemaId::Stage1Rows, 10),
        binding_digest: digest(11),
    };

    let mut accumulator = OpeningClaimAccumulator::default();
    let err = accumulator
        .insert(claim)
        .expect_err("opened-object digests must stay bound to their commitment context");

    assert_eq!(
        err,
        EvalClaimError::OpenedObjectDigestMismatch {
            expected_digest: OpenedAjtaiObjectId::new(AjtaiFamilyKind::Stage1Rows, &other_context, digest(9), 1, 2,)
                .digest,
            object_digest: opened_object_digest,
        }
    );
}

#[test]
fn opening_claim_accumulator_rejects_same_object_slot_with_different_binding() {
    let commitment_context = CommitmentContextId::new(digest(10), digest(11));
    let first = claim(
        FamilyEvalSchemaId::Stage2RegisterReads,
        commitment_context,
        digest(12),
        1,
        2,
        0,
        digest(13),
        20,
    );
    let conflicting = claim(
        FamilyEvalSchemaId::Stage2RegisterReads,
        commitment_context,
        digest(12),
        1,
        2,
        0,
        digest(14),
        20,
    );

    let mut accumulator = OpeningClaimAccumulator::default();
    accumulator.insert(first.clone()).expect("first claim");
    let err = accumulator
        .insert(conflicting)
        .expect_err("same object+slot with different binding must fail");

    assert_eq!(
        err,
        EvalClaimError::SlotBindingConflict {
            opened_object_digest: first.opened_object.digest,
            slot: first.id.slot,
            existing_binding_digest: first.binding_digest,
            new_binding_digest: digest(14),
        }
    );
}

#[test]
fn eval_claim_bundle_canonicalizes_order_and_dedups_identical_claims() {
    let commitment_context = CommitmentContextId::new(digest(21), digest(22));
    let stage2_high = claim(
        FamilyEvalSchemaId::Stage2RegisterReads,
        commitment_context,
        digest(40),
        1,
        2,
        0,
        digest(23),
        30,
    );
    let stage1 = claim(
        FamilyEvalSchemaId::Stage1Rows,
        commitment_context,
        digest(30),
        1,
        2,
        1,
        digest(24),
        40,
    );
    let stage2_low = claim(
        FamilyEvalSchemaId::Stage2RegisterReads,
        commitment_context,
        digest(20),
        1,
        2,
        0,
        digest(25),
        50,
    );

    let bundle = Rv32imEvalClaimBundle::new(vec![
        stage2_high.clone(),
        stage1.clone(),
        stage2_low.clone(),
        stage1.clone(),
    ])
    .expect("valid phase0 bundle");

    assert_eq!(bundle.claims.len(), 3, "identical claims should dedup");
    let expected = {
        let mut claims = vec![stage2_high, stage1, stage2_low];
        claims.sort_by_key(|claim| (claim.payload.schema, claim.opened_object.digest, claim.id.slot));
        claims
    };
    assert_eq!(bundle.claims, expected);
    assert_ne!(bundle.digest, [0; 32]);
}

#[test]
fn phase0_binding_sources_use_exact_existing_stage_fields() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv32im_public_proof(&input).expect("prove rv32im public proof");
    let artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build accepted artifact");

    let stage1_family_binding_anchor_digest = artifact.stage_claims.claims.stage1.rows.rows_digest;
    let stage1_stage_proof_binding_digest = artifact.stage1.digest;
    assert_eq!(
        stage1_family_binding_anchor_digest, artifact.stage1.linkage.rows_digest,
        "Stage1 proof linkage must stay bound to the exact rows digest"
    );
    assert_ne!(stage1_stage_proof_binding_digest, [0; 32]);

    let stage2_register_reads_anchor = artifact
        .stage_claims
        .claims
        .stage2
        .families
        .register_reads_digest;
    let stage2_register_writes_anchor = artifact
        .stage_claims
        .claims
        .stage2
        .families
        .register_writes_digest;
    let stage2_ram_events_anchor = artifact
        .stage_claims
        .claims
        .stage2
        .families
        .ram_events_digest;
    let stage2_twist_links_anchor = artifact
        .stage_claims
        .claims
        .stage2
        .families
        .twist_links_digest;
    let stage2_stage_proof_binding_digest = artifact.stage2.digest;
    assert_eq!(
        stage2_register_reads_anchor, artifact.stage2.linkage.register_reads_family_digest,
        "Stage2 read-family binding must stay on the exact family digest field"
    );
    assert_eq!(
        stage2_register_writes_anchor, artifact.stage2.linkage.register_writes_family_digest,
        "Stage2 write-family binding must stay on the exact family digest field"
    );
    assert_eq!(
        stage2_ram_events_anchor, artifact.stage2.linkage.ram_events_family_digest,
        "Stage2 RAM-family binding must stay on the exact family digest field"
    );
    assert_eq!(
        stage2_twist_links_anchor, artifact.stage2.linkage.twist_links_family_digest,
        "Stage2 twist-family binding must stay on the exact family digest field"
    );
    assert_ne!(stage2_stage_proof_binding_digest, [0; 32]);

    let stage3_family_binding_anchor_digest = artifact
        .stage_claims
        .claims
        .stage3
        .continuity
        .continuity_digest;
    let stage3_stage_proof_binding_digest = artifact.stage3.digest;
    assert_eq!(
        stage3_family_binding_anchor_digest, artifact.stage3.linkage.continuity_family_digest,
        "Stage3 proof linkage must stay bound to the exact continuity digest field"
    );
    assert_ne!(stage3_stage_proof_binding_digest, [0; 32]);
}
