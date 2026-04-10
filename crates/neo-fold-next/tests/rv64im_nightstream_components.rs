use neo_fold_next::nightstream::rv64im::{
    audit::{
        build_rv64im_kernel_opening_claim_from_side_proof_bundle, build_rv64im_opening_artifact_from_accepted_artifact,
        build_rv64im_phase0_opened_object_bundle_from_claim_witnesses,
        build_rv64im_side_claim_relation_from_accepted_artifact, build_rv64im_side_claim_relation_statement,
        build_rv64im_side_claim_relation_witness_from_accepted_artifact,
        build_rv64im_side_eval_claim_artifact_from_accepted_artifact,
        build_rv64im_side_eval_claim_relation_from_accepted_artifact,
        build_rv64im_side_eval_claim_relation_statement_from_artifact,
        build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact,
        build_rv64im_side_opening_relation_from_accepted_artifact, build_rv64im_side_opening_relation_statement,
        build_rv64im_side_opening_relation_witness_from_accepted_artifact,
        build_rv64im_side_proof_bundle_from_accepted_artifact, build_rv64im_stage_claim_bundle_from_side_proof_bundle,
        verify_rv64im_hybrid_side_bridge_artifact, verify_rv64im_opening_artifact_from_side_proof_bundle,
        verify_rv64im_side_claim_relation, verify_rv64im_side_eval_claim_artifact,
        verify_rv64im_side_eval_claim_relation, verify_rv64im_side_opening_relation,
        Rv64imWitnessBackedSideBridgeStatement,
    },
    build_rv64im_nightstream_from_public_proof,
};
use neo_fold_next::rv64im::{
    build_phase2_collapse_result, build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof,
    Rv64imEvalClaimBundle, Rv64imProofInput, Rv64imProofStatement,
};
use neo_math::K;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn source_case(name: &str) -> neo_fold_next::rv64im::Rv64imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn alternate_case_name(exclude: &str) -> String {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name != exclude)
        .unwrap_or_else(|| panic!("missing alternate parity source case for {exclude}"))
        .manifest
        .name
        .to_string()
}

fn proof_input(name: &str) -> Rv64imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

fn rebind_hybrid_side_bridge_artifact(
    nightstream_statement: &neo_fold_next::nightstream::NightstreamStatement,
    public_statement: &Rv64imProofStatement,
    nightstream_proof: &mut neo_fold_next::nightstream::rv64im::Rv64imNightstreamProof,
) {
    let bridge_artifact = &mut nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact;
    let bridge_statement = Rv64imWitnessBackedSideBridgeStatement {
        nightstream_statement: nightstream_statement.clone(),
        public_statement: public_statement.clone(),
        side_bundle_digest: bridge_artifact.witness.side_bundle.digest,
        opening_artifact_digest: bridge_artifact.witness.opening_artifact.digest,
        bridge_handoff_digests: nightstream_proof
            .main_residual_proof
            .bridge_handoff_digests
            .clone(),
    };
    bridge_artifact.digest = bridge_artifact.expected_digest(bridge_statement.digest());
    nightstream_proof.hybrid_side_bridge_artifact.digest = nightstream_proof
        .hybrid_side_bridge_artifact
        .expected_digest();
}

fn rebind_hybrid_side_bundle_digests(
    nightstream_statement: &neo_fold_next::nightstream::NightstreamStatement,
    public_statement: &Rv64imProofStatement,
    nightstream_proof: &mut neo_fold_next::nightstream::rv64im::Rv64imNightstreamProof,
) {
    let side_bundle = &mut nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle;
    side_bundle
        .kernel_opening_bridge
        .prepared_step_bindings
        .digest = side_bundle
        .kernel_opening_bridge
        .prepared_step_bindings
        .expected_digest();
    side_bundle.kernel_opening_bridge.digest = side_bundle.kernel_opening_bridge.expected_digest();
    side_bundle.digest = side_bundle.expected_digest();
    rebind_hybrid_side_bridge_artifact(nightstream_statement, public_statement, nightstream_proof);
}

fn rebind_hybrid_opening_artifact_to_side_bundle(
    public_statement: &Rv64imProofStatement,
    nightstream_proof: &mut neo_fold_next::nightstream::rv64im::Rv64imNightstreamProof,
) {
    let bridge_artifact = &mut nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact;
    let phase0_opened_objects_digest = build_rv64im_side_eval_claim_relation_statement_from_artifact(
        public_statement,
        &bridge_artifact.witness.side_bundle,
        &bridge_artifact.witness.opening_artifact.phase0_artifact,
    )
    .expect("rebuild side-eval-claim relation statement before rebinding opening artifact")
    .phase0_opened_objects
    .digest;
    let opening_artifact = &mut bridge_artifact.witness.opening_artifact;
    opening_artifact.phase0_artifact.statement_digest = side_eval_claim_relation_statement_digest(
        public_statement.digest,
        bridge_artifact.witness.side_bundle.digest,
        phase0_opened_objects_digest,
        opening_artifact.phase0_artifact.eval_claim_bundle.digest,
    );
    opening_artifact.phase0_artifact.digest = opening_artifact.phase0_artifact.expected_digest();
    opening_artifact.digest = opening_artifact.expected_digest();
}

fn side_eval_claim_relation_statement_digest(
    public_statement_digest: [u8; 32],
    side_bundle_digest: [u8; 32],
    phase0_opened_objects_digest: [u8; 32],
    eval_claim_bundle_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/side_eval_claim_relation");
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/side_eval_claim_relation/public_statement_digest",
        &public_statement_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/side_eval_claim_relation/side_bundle_digest",
        &side_bundle_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/side_eval_claim_relation/phase0_opened_objects_digest",
        &phase0_opened_objects_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/side_eval_claim_relation/eval_claim_bundle_digest",
        &eval_claim_bundle_digest,
    );
    tr.digest32()
}

fn forge_opening_phase0_payload(
    public_statement: &Rv64imProofStatement,
    side_bundle: &neo_fold_next::nightstream::rv64im::Rv64imSideProofBundle,
    opening_artifact: &mut neo_fold_next::nightstream::rv64im::Rv64imOpeningArtifact,
) {
    let phase0_opened_objects_digest = build_rv64im_side_eval_claim_relation_statement_from_artifact(
        public_statement,
        side_bundle,
        &opening_artifact.phase0_artifact,
    )
    .expect("rebuild phase0 relation statement before forgery")
    .phase0_opened_objects
    .digest;
    let convergence = &mut opening_artifact.convergence_artifact;
    convergence.phase1_results[0].bucket.claims[0]
        .payload
        .column_evals[0]
        .coeffs[0] += K::ONE;
    convergence.phase1_results[0].proof.bucket_digest = convergence.phase1_results[0].bucket.expected_digest();
    convergence.phase1_results[0].proof.digest = convergence.phase1_results[0].proof.expected_digest();
    convergence.phase0_digest = Rv64imEvalClaimBundle::new(
        convergence
            .phase1_results
            .iter()
            .flat_map(|result| result.bucket.claims.clone())
            .collect(),
    )
    .expect("rebuild phase0 bundle")
    .digest;
    let rebuilt_phase2 = build_phase2_collapse_result(&convergence.phase1_results).expect("rebuild phase2");
    for (target, claim) in convergence
        .final_openings
        .iter_mut()
        .zip(rebuilt_phase2.reduced_claims.iter())
    {
        target.digest = target.expected_digest(claim);
    }
    convergence.phase2 = rebuilt_phase2;
    convergence.digest = convergence.expected_digest();
    opening_artifact.phase0_artifact.eval_claim_bundle = Rv64imEvalClaimBundle::new(
        convergence
            .phase1_results
            .iter()
            .flat_map(|result| result.bucket.claims.clone())
            .collect(),
    )
    .expect("rebuild phase0 bundle from forged convergence artifact");
    opening_artifact.phase0_artifact.statement_digest = side_eval_claim_relation_statement_digest(
        public_statement.digest,
        side_bundle.digest,
        phase0_opened_objects_digest,
        opening_artifact.phase0_artifact.eval_claim_bundle.digest,
    );
    opening_artifact.phase0_artifact.digest = opening_artifact.phase0_artifact.expected_digest();
    opening_artifact.digest = opening_artifact.expected_digest();
}

#[test]
fn rv64im_side_bundle_rebuilds_exact_stage_claim_bundle() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let bundle = build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let rebuilt = build_rv64im_stage_claim_bundle_from_side_proof_bundle(&bundle, proof.statement.execution_digest)
        .expect("rebuild exact stage-claim bundle from side proof bundle");

    assert_eq!(rebuilt, artifact.stage_claims.claims);
}

#[test]
fn rv64im_side_bundle_rebuilds_exact_kernel_opening_claim() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let bundle = build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let rebuilt = build_rv64im_kernel_opening_claim_from_side_proof_bundle(&bundle, &proof.statement)
        .expect("rebuild exact kernel-opening claim from side proof bundle");

    assert_eq!(rebuilt, artifact.kernel_opening.opening.claim);
}

#[test]
fn rv64im_side_opening_relation_roundtrips_from_accepted_artifact() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    verify_rv64im_side_opening_relation(&statement, &witness).expect("verify side-opening relation");
}

#[test]
fn rv64im_side_opening_relation_rejects_tampered_stage1_selected_row_witness() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, mut witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    witness.stage1_selected_rows.first.fetched_word ^= 1;

    let err = verify_rv64im_side_opening_relation(&statement, &witness)
        .expect_err("tampered stage1 selected row witness must fail");
    assert!(format!("{err}")
        .contains("RV64IM side-opening relation stage1 selected rows do not match the carried opening claim"));
}

#[test]
fn rv64im_side_opening_relation_rejects_tampered_stage1_package_statement_step_witness() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, mut witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&artifact).expect("build side-opening relation");

    witness.stage1_packaged.step.label.push_str("/tamper");

    let err = verify_rv64im_side_opening_relation(&statement, &witness)
        .expect_err("tampered stage1 package statement step must fail");
    assert!(
        format!("{err}").contains("rv64im/stage1 selected-claim package statement digest mismatch")
            || format!("{err}").contains("rv64im/stage1 selected-claim package public step mismatch")
    );
}

#[test]
fn rv64im_side_opening_relation_rejects_self_consistent_binding_summary_swap() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let mut statement =
        build_rv64im_side_opening_relation_statement(&proof.statement, &side_bundle).expect("build relation statement");
    let witness = build_rv64im_side_opening_relation_witness_from_accepted_artifact(&artifact);

    let last_binding = statement
        .side_bundle
        .kernel_opening_bridge
        .prepared_step_bindings
        .last_binding_digest
        .as_mut()
        .expect("last binding digest");
    last_binding[0] ^= 1;
    statement
        .side_bundle
        .kernel_opening_bridge
        .prepared_step_bindings
        .digest = statement
        .side_bundle
        .kernel_opening_bridge
        .prepared_step_bindings
        .expected_digest();
    statement.side_bundle.kernel_opening_bridge.digest = statement
        .side_bundle
        .kernel_opening_bridge
        .expected_digest();
    statement.side_bundle.digest = statement.side_bundle.expected_digest();

    let err = verify_rv64im_side_opening_relation(&statement, &witness)
        .expect_err("self-consistent binding-summary swap must fail");
    assert!(format!("{err}").contains(
        "RV64IM Nightstream compact kernel-opening proof surface does not match the carried public statement"
    ));
}

#[test]
fn rv64im_side_opening_relation_compact_witness_is_smaller_than_full_opening_bundles() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let witness = build_rv64im_side_opening_relation_witness_from_accepted_artifact(&artifact);

    let full_len = bincode::serialize(&(
        artifact.stage_packages.packages.clone(),
        artifact.kernel_opening.opening.clone(),
    ))
    .expect("serialize full opening-side witness material")
    .len();
    let compact_len = bincode::serialize(&witness)
        .expect("serialize compact opening-side witness")
        .len();

    println!(
        "rv64im side-opening compact witness sizes: full_opening={} compact_opening={}",
        full_len, compact_len
    );

    assert!(compact_len < full_len);
}

#[test]
fn rv64im_side_eval_claim_relation_roundtrips_from_accepted_artifact() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) = build_rv64im_side_eval_claim_relation_from_accepted_artifact(&artifact)
        .expect("build side-eval-claim relation");

    verify_rv64im_side_eval_claim_relation(&statement, &witness).expect("verify side-eval-claim relation");
}

#[test]
fn rv64im_side_eval_claim_relation_rejects_tampered_side_bundle_stage2_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (mut statement, witness) = build_rv64im_side_eval_claim_relation_from_accepted_artifact(&artifact)
        .expect("build side-eval-claim relation");

    statement.side_bundle.stage2.digest[0] ^= 1;
    statement.side_bundle.digest = statement.side_bundle.expected_digest();

    let err = verify_rv64im_side_eval_claim_relation(&statement, &witness)
        .expect_err("tampered side-bundle stage2 digest must fail");
    assert!(format!("{err}").contains("stage2 verified-claims digest mismatch"));
}

#[test]
fn rv64im_side_eval_claim_relation_rejects_tampered_opened_object_summary() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (mut statement, witness) = build_rv64im_side_eval_claim_relation_from_accepted_artifact(&artifact)
        .expect("build side-eval-claim relation");

    let stage1 = &mut statement.phase0_opened_objects.objects[0];
    stage1.opened_object.commitment_root_digest[0] ^= 1;
    stage1.opened_object.digest = stage1
        .opened_object
        .expected_digest(&stage1.commitment_context);
    stage1.digest = stage1.expected_digest();
    statement.phase0_opened_objects.digest = statement.phase0_opened_objects.expected_digest();

    let err = verify_rv64im_side_eval_claim_relation(&statement, &witness)
        .expect_err("tampered opened-object summary must fail");
    assert!(format!("{err}").contains("opened-object summaries do not match the carried Phase 0 bundle"));
}

#[test]
fn rv64im_side_eval_claim_relation_opened_objects_match_claim_witness_projection() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let witness = build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact(&artifact)
        .expect("build side-eval-claim witness");
    let opened_objects = build_rv64im_phase0_opened_object_bundle_from_claim_witnesses(&witness.claim_witnesses)
        .expect("build phase0 opened-object bundle");

    assert_eq!(opened_objects.objects.len(), 6);
    assert_ne!(opened_objects.digest, [0; 32]);
}

#[test]
fn rv64im_side_eval_claim_artifact_roundtrips_from_accepted_artifact() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let phase0_artifact = build_rv64im_side_eval_claim_artifact_from_accepted_artifact(&artifact)
        .expect("build side eval claim artifact");

    verify_rv64im_side_eval_claim_artifact(&proof.statement, &side_bundle, &phase0_artifact)
        .expect("verify side eval claim artifact");
}

#[test]
fn rv64im_side_eval_claim_artifact_reconstructs_relation_statement() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let (statement, _) = build_rv64im_side_eval_claim_relation_from_accepted_artifact(&artifact)
        .expect("build side eval claim relation");
    let phase0_artifact = build_rv64im_side_eval_claim_artifact_from_accepted_artifact(&artifact)
        .expect("build side eval claim artifact");

    let rebuilt =
        build_rv64im_side_eval_claim_relation_statement_from_artifact(&proof.statement, &side_bundle, &phase0_artifact)
            .expect("rebuild side eval claim relation statement from artifact");

    assert_eq!(rebuilt, statement);
}

#[test]
fn rv64im_side_eval_claim_artifact_rejects_duplicate_phase0_opening_target() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let mut phase0_artifact = build_rv64im_side_eval_claim_artifact_from_accepted_artifact(&artifact)
        .expect("build side eval claim artifact");

    let duplicate = phase0_artifact.phase0_opening_targets.targets[0].clone();
    phase0_artifact
        .phase0_opening_targets
        .targets
        .push(duplicate);
    phase0_artifact.phase0_opening_targets.digest = phase0_artifact.phase0_opening_targets.expected_digest();
    phase0_artifact.digest = phase0_artifact.expected_digest();

    let err = verify_rv64im_side_eval_claim_artifact(&proof.statement, &side_bundle, &phase0_artifact)
        .expect_err("duplicate phase0 opening target must fail");
    assert!(
        format!("{err}").contains("canonical")
            || format!("{err}").contains("exactly")
            || format!("{err}").contains("opening-target bundle")
    );
}

#[test]
fn rv64im_side_eval_claim_artifact_rejects_tampered_statement_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let mut phase0_artifact = build_rv64im_side_eval_claim_artifact_from_accepted_artifact(&artifact)
        .expect("build side eval claim artifact");

    phase0_artifact.statement_digest[0] ^= 1;
    phase0_artifact.digest = phase0_artifact.expected_digest();

    let err = verify_rv64im_side_eval_claim_artifact(&proof.statement, &side_bundle, &phase0_artifact)
        .expect_err("tampered statement digest must fail");
    assert!(format!("{err}").contains("statement digest does not match the carried relation statement"));
}

#[test]
fn rv64im_side_eval_claim_artifact_rejects_tampered_eval_claim_bundle() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let mut phase0_artifact = build_rv64im_side_eval_claim_artifact_from_accepted_artifact(&artifact)
        .expect("build side eval claim artifact");

    phase0_artifact.eval_claim_bundle.claims[0].binding_digest[0] ^= 1;
    phase0_artifact.eval_claim_bundle.digest = phase0_artifact.eval_claim_bundle.expected_digest();
    phase0_artifact.digest = phase0_artifact.expected_digest();

    let err = verify_rv64im_side_eval_claim_artifact(&proof.statement, &side_bundle, &phase0_artifact)
        .expect_err("tampered eval claim bundle must fail");
    assert!(format!("{err}").contains("statement digest does not match the carried relation statement"));
}

#[test]
fn rv64im_side_eval_claim_artifact_rejects_rebound_payload_forgery() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let mut phase0_artifact = build_rv64im_side_eval_claim_artifact_from_accepted_artifact(&artifact)
        .expect("build side eval claim artifact");
    let phase0_opened_objects_digest =
        build_rv64im_side_eval_claim_relation_statement_from_artifact(&proof.statement, &side_bundle, &phase0_artifact)
            .expect("rebuild relation statement before payload forgery")
            .phase0_opened_objects
            .digest;

    phase0_artifact.eval_claim_bundle.claims[0]
        .payload
        .column_evals[0]
        .coeffs[0] += K::ONE;
    phase0_artifact.eval_claim_bundle.digest = phase0_artifact.eval_claim_bundle.expected_digest();
    phase0_artifact.statement_digest = side_eval_claim_relation_statement_digest(
        proof.statement.digest,
        side_bundle.digest,
        phase0_opened_objects_digest,
        phase0_artifact.eval_claim_bundle.digest,
    );
    phase0_artifact.digest = phase0_artifact.expected_digest();

    let err = verify_rv64im_side_eval_claim_artifact(&proof.statement, &side_bundle, &phase0_artifact)
        .expect_err("forged phase0 eval payload must fail");
    assert!(
        format!("{err}").contains("payload") || format!("{err}").contains("eval") || format!("{err}").contains("claim")
    );
}

#[test]
fn rv64im_opening_artifact_rejects_rebound_phase0_payload_forgery() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let mut opening_artifact =
        build_rv64im_opening_artifact_from_accepted_artifact(&proof.statement, &side_bundle, &artifact)
            .expect("build opening artifact");

    forge_opening_phase0_payload(&proof.statement, &side_bundle, &mut opening_artifact);

    let err = verify_rv64im_opening_artifact_from_side_proof_bundle(&proof.statement, &side_bundle, &opening_artifact)
        .expect_err("forged opening-artifact phase0 payload must fail");
    assert!(
        format!("{err}").contains("payload")
            || format!("{err}").contains("opening")
            || format!("{err}").contains("claim")
    );
}

#[test]
fn rv64im_opening_artifact_rejects_duplicate_phase0_opening_target() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let mut opening_artifact =
        build_rv64im_opening_artifact_from_accepted_artifact(&proof.statement, &side_bundle, &artifact)
            .expect("build opening artifact");

    let duplicate = opening_artifact
        .phase0_artifact
        .phase0_opening_targets
        .targets[0]
        .clone();
    opening_artifact
        .phase0_artifact
        .phase0_opening_targets
        .targets
        .push(duplicate);
    opening_artifact
        .phase0_artifact
        .phase0_opening_targets
        .digest = opening_artifact
        .phase0_artifact
        .phase0_opening_targets
        .expected_digest();
    opening_artifact.phase0_artifact.digest = opening_artifact.phase0_artifact.expected_digest();
    opening_artifact.digest = opening_artifact.expected_digest();

    let err = verify_rv64im_opening_artifact_from_side_proof_bundle(&proof.statement, &side_bundle, &opening_artifact)
        .expect_err("duplicate phase0 opening target in opening artifact must fail");
    assert!(
        format!("{err}").contains("canonical")
            || format!("{err}").contains("exactly")
            || format!("{err}").contains("opening-target bundle")
    );
}

#[test]
fn rv64im_side_claim_relation_roundtrips_from_accepted_artifact() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, witness) =
        build_rv64im_side_claim_relation_from_accepted_artifact(&artifact).expect("build side-claim relation");

    verify_rv64im_side_claim_relation(&statement, &witness).expect("verify side-claim relation");
}

#[test]
fn rv64im_side_claim_relation_rejects_tampered_stage_claim_witness() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, mut witness) =
        build_rv64im_side_claim_relation_from_accepted_artifact(&artifact).expect("build side-claim relation");

    witness.stage_claims_packaged.proof_digest[0] ^= 1;

    let err =
        verify_rv64im_side_claim_relation(&statement, &witness).expect_err("tampered stage-claim witness must fail");
    assert!(format!("{err}")
        .contains("RV64IM side-claim relation stage-claim witness does not match the carried side bundle"));
}

#[test]
fn rv64im_side_claim_relation_rejects_tampered_stage_claim_statement_step_witness() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, mut witness) =
        build_rv64im_side_claim_relation_from_accepted_artifact(&artifact).expect("build side-claim relation");

    witness.stage_claims_packaged.step.label.push_str("/tamper");

    let err = verify_rv64im_side_claim_relation(&statement, &witness)
        .expect_err("tampered stage-claim statement step must fail");
    assert!(
        format!("{err}").contains("rv64im/stage_claim_bundle selected-claim package statement digest mismatch")
            || format!("{err}").contains("rv64im/stage_claim_bundle selected-claim package public step mismatch")
    );
}

#[test]
fn rv64im_side_claim_relation_rejects_self_consistent_kernel_claim_witness_swap() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build side proof bundle");
    let statement =
        build_rv64im_side_claim_relation_statement(&proof.statement, &side_bundle).expect("build relation statement");
    let mut witness = build_rv64im_side_claim_relation_witness_from_accepted_artifact(&artifact);

    let alternate_proof = prove_rv64im_public_proof(&proof_input(&alternate_case_name("control_flow_jal_skip_ecall")))
        .expect("prove alternate rv64im public proof");
    let alternate_artifact =
        build_rv64im_accepted_proof_artifact(&alternate_proof).expect("build alternate accepted artifact");
    witness.kernel_claims_packaged =
        build_rv64im_side_claim_relation_witness_from_accepted_artifact(&alternate_artifact).kernel_claims_packaged;

    let err = verify_rv64im_side_claim_relation(&statement, &witness)
        .expect_err("self-consistent kernel-claim witness swap must fail");
    assert!(
        format!("{err}")
            .contains("RV64IM side-claim relation kernel-claim witness does not match the carried side bundle")
            || format!("{err}").contains("rv64im/kernel_claim_bundle selected-claim package public step mismatch")
    );
}

#[test]
fn rv64im_side_claim_relation_compact_witness_is_smaller_than_full_packaged_proofs() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let witness = build_rv64im_side_claim_relation_witness_from_accepted_artifact(&artifact);

    let full_stage_len = bincode::serialize(&artifact.stage_claims.packaged)
        .expect("serialize full stage packaged proof")
        .len();
    let compact_stage_len = bincode::serialize(&witness.stage_claims_packaged)
        .expect("serialize compact stage packaged witness")
        .len();
    let full_kernel_len = bincode::serialize(&artifact.kernel_claims.packaged)
        .expect("serialize full kernel packaged proof")
        .len();
    let compact_kernel_len = bincode::serialize(&witness.kernel_claims_packaged)
        .expect("serialize compact kernel packaged witness")
        .len();

    println!(
        "rv64im side-claim compact witness sizes: full_stage={} compact_stage={} full_kernel={} compact_kernel={}",
        full_stage_len, compact_stage_len, full_kernel_len, compact_kernel_len
    );

    assert!(compact_stage_len < full_stage_len);
    assert!(compact_kernel_len < full_kernel_len);
}

#[test]
fn rv64im_hybrid_side_bridge_proof_artifact_rejects_wrong_opening_artifact() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build nightstream proof");
    let mut wrong_bridge_artifact = nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .clone();
    wrong_bridge_artifact.witness.opening_artifact.digest[0] ^= 1;
    let bridge_statement = Rv64imWitnessBackedSideBridgeStatement {
        nightstream_statement: nightstream_statement.clone(),
        public_statement: proof.statement.clone(),
        side_bundle_digest: wrong_bridge_artifact.witness.side_bundle.digest,
        opening_artifact_digest: wrong_bridge_artifact.witness.opening_artifact.digest,
        bridge_handoff_digests: nightstream_proof
            .main_residual_proof
            .bridge_handoff_digests
            .clone(),
    };
    wrong_bridge_artifact.digest = wrong_bridge_artifact.expected_digest(bridge_statement.digest());

    let mut wrong_hybrid_side_bridge_artifact = nightstream_proof.hybrid_side_bridge_artifact.clone();
    wrong_hybrid_side_bridge_artifact.bridge_artifact = wrong_bridge_artifact;
    wrong_hybrid_side_bridge_artifact.digest = wrong_hybrid_side_bridge_artifact.expected_digest();

    let err = verify_rv64im_hybrid_side_bridge_artifact(
        &nightstream_statement,
        &nightstream_proof.main_residual_proof.bridge_handoff_digests,
        &proof.statement,
        &wrong_hybrid_side_bridge_artifact,
    )
    .expect_err("hybrid-side-bridge proof verifier must reject a wrong opening artifact");
    assert!(
        format!("{err}").contains("opening"),
        "unexpected error for wrong opening artifact: {err}"
    );
}

#[test]
fn rv64im_hybrid_side_bridge_artifact_rejects_tampered_stage_claim_witness() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, mut nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build nightstream proof");

    nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .claim_witness
        .stage_claims_packaged
        .proof_digest[0] ^= 1;
    rebind_hybrid_side_bridge_artifact(&nightstream_statement, &proof.statement, &mut nightstream_proof);

    let err = verify_rv64im_hybrid_side_bridge_artifact(
        &nightstream_statement,
        &nightstream_proof.main_residual_proof.bridge_handoff_digests,
        &proof.statement,
        &nightstream_proof.hybrid_side_bridge_artifact,
    )
    .expect_err("hybrid-side-bridge proof verifier must reject a tampered stage-claim witness");
    assert!(
        format!("{err}")
            .contains("RV64IM side-claim relation stage-claim witness does not match the carried side bundle"),
        "unexpected error for tampered stage-claim witness: {err}"
    );
}

#[test]
fn rv64im_hybrid_side_bridge_artifact_rejects_self_consistent_kernel_claim_witness_swap() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, mut nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build nightstream proof");

    let alternate_proof = prove_rv64im_public_proof(&proof_input(&alternate_case_name("control_flow_jal_skip_ecall")))
        .expect("prove alternate rv64im public proof");
    let alternate_artifact =
        build_rv64im_accepted_proof_artifact(&alternate_proof).expect("build alternate accepted artifact");
    nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .claim_witness
        .kernel_claims_packaged =
        build_rv64im_side_claim_relation_witness_from_accepted_artifact(&alternate_artifact).kernel_claims_packaged;
    rebind_hybrid_side_bridge_artifact(&nightstream_statement, &proof.statement, &mut nightstream_proof);

    let err = verify_rv64im_hybrid_side_bridge_artifact(
        &nightstream_statement,
        &nightstream_proof.main_residual_proof.bridge_handoff_digests,
        &proof.statement,
        &nightstream_proof.hybrid_side_bridge_artifact,
    )
    .expect_err("hybrid-side-bridge proof verifier must reject a swapped kernel-claim witness");
    assert!(
        format!("{err}")
            .contains("RV64IM side-claim relation kernel-claim witness does not match the carried side bundle")
            || format!("{err}").contains("rv64im/kernel_claim_bundle selected-claim package public step mismatch"),
        "unexpected error for swapped kernel-claim witness: {err}"
    );
}

#[test]
fn rv64im_hybrid_side_bridge_artifact_rejects_tampered_stage1_selected_row_witness() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, mut nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build nightstream proof");

    nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .opening_witness
        .stage1_selected_rows
        .first
        .fetched_word ^= 1;
    rebind_hybrid_side_bridge_artifact(&nightstream_statement, &proof.statement, &mut nightstream_proof);

    let err = verify_rv64im_hybrid_side_bridge_artifact(
        &nightstream_statement,
        &nightstream_proof.main_residual_proof.bridge_handoff_digests,
        &proof.statement,
        &nightstream_proof.hybrid_side_bridge_artifact,
    )
    .expect_err("hybrid-side-bridge proof verifier must reject a tampered stage1 selected row witness");
    assert!(
        format!("{err}")
            .contains("RV64IM side-opening relation stage1 selected rows do not match the carried opening claim"),
        "unexpected error for tampered stage1 selected row witness: {err}"
    );
}

#[test]
fn rv64im_hybrid_side_bridge_artifact_rejects_self_consistent_binding_summary_swap() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, mut nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build nightstream proof");

    let last_binding = nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .prepared_step_bindings
        .last_binding_digest
        .as_mut()
        .expect("last binding digest");
    last_binding[0] ^= 1;
    rebind_hybrid_side_bundle_digests(&nightstream_statement, &proof.statement, &mut nightstream_proof);

    let err = verify_rv64im_hybrid_side_bridge_artifact(
        &nightstream_statement,
        &nightstream_proof.main_residual_proof.bridge_handoff_digests,
        &proof.statement,
        &nightstream_proof.hybrid_side_bridge_artifact,
    )
    .expect_err("hybrid-side-bridge proof verifier must reject a self-consistent binding-summary swap");
    assert!(
        format!("{err}").contains(
            "RV64IM Nightstream compact kernel-opening proof surface does not match the carried public statement"
        ) || format!("{err}").contains(
            "RV64IM Nightstream opening artifact does not match the verified compact Phase 0 opening surface"
        ) || format!("{err}")
            .contains("RV64IM side-eval-claim artifact statement digest does not match the carried relation statement"),
        "unexpected error for binding-summary swap: {err}"
    );
}

#[test]
fn rv64im_hybrid_side_bridge_artifact_rejects_tampered_root_execution_surface() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, mut nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build nightstream proof");

    let side_bundle = &mut nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle;
    side_bundle.row_local_ccs_acceptance_digest[0] ^= 1;
    side_bundle.digest = side_bundle.expected_digest();
    rebind_hybrid_opening_artifact_to_side_bundle(&proof.statement, &mut nightstream_proof);
    rebind_hybrid_side_bridge_artifact(&nightstream_statement, &proof.statement, &mut nightstream_proof);

    let err = verify_rv64im_hybrid_side_bridge_artifact(
        &nightstream_statement,
        &nightstream_proof.main_residual_proof.bridge_handoff_digests,
        &proof.statement,
        &nightstream_proof.hybrid_side_bridge_artifact,
    )
    .expect_err("hybrid-side-bridge proof verifier must reject a tampered root-execution surface");
    assert!(
        format!("{err}").contains(
            "RV64IM Nightstream compact side-proof root-execution surface does not match the carried statement surfaces"
        ),
        "unexpected error for tampered root-execution surface: {err}"
    );
}

#[test]
fn rv64im_hybrid_side_bridge_artifact_rejects_tampered_kernel_export_surface() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, mut nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build nightstream proof");

    let side_bundle = &mut nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle;
    side_bundle.kernel_export_bridge.kernel_export_source_digest[0] ^= 1;
    side_bundle.kernel_export_bridge.digest = side_bundle.kernel_export_bridge.expected_digest();
    side_bundle.digest = side_bundle.expected_digest();
    rebind_hybrid_opening_artifact_to_side_bundle(&proof.statement, &mut nightstream_proof);
    rebind_hybrid_side_bridge_artifact(&nightstream_statement, &proof.statement, &mut nightstream_proof);

    let err = verify_rv64im_hybrid_side_bridge_artifact(
        &nightstream_statement,
        &nightstream_proof.main_residual_proof.bridge_handoff_digests,
        &proof.statement,
        &nightstream_proof.hybrid_side_bridge_artifact,
    )
    .expect_err("hybrid-side-bridge proof verifier must reject a tampered kernel-export surface");
    assert!(
        format!("{err}").contains(
            "RV64IM Nightstream compact kernel-export source surface does not match the carried public statement"
        ),
        "unexpected error for tampered kernel-export surface: {err}"
    );
}

#[test]
fn rv64im_hybrid_side_bridge_artifact_rejects_mismatched_bridge_handoff_digests() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build nightstream proof");
    let mut bridge_handoff_digests = nightstream_proof
        .main_residual_proof
        .bridge_handoff_digests
        .clone();
    bridge_handoff_digests[0][0] ^= 1;

    let err = verify_rv64im_hybrid_side_bridge_artifact(
        &nightstream_statement,
        &bridge_handoff_digests,
        &proof.statement,
        &nightstream_proof.hybrid_side_bridge_artifact,
    )
    .expect_err("hybrid-side-bridge proof verifier must reject mismatched bridge handoff digests");
    assert!(
        format!("{err}").contains("RV64IM witness-backed side bridge artifact digest mismatch"),
        "unexpected error for mismatched bridge handoff digests: {err}"
    );
}
