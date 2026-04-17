//! Focused tests for the RV64IM folded/final relation seam above the accepted artifact.

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::build_rv64im_decider_relation_from_final_surface;
use neo_fold_next::rv64im::final_relation::{
    build_rv64im_terminal_chunk_fold_witness, prove_rv64im_final_statement_from_accepted,
    prove_rv64im_folded_statement_from_accepted, verify_rv64im_final_statement, verify_rv64im_folded_statement,
};
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, build_rv64im_kernel_export_source_from_accepted_artifact,
    build_rv64im_kernel_export_witness, parity_source_cases, prove_rv64im_accepted_proof,
    prove_rv64im_accepted_proof_with_options, prove_rv64im_public_proof,
    prove_rv64im_public_proof_and_published_seam_with_perf, Rv64imProofInput, Rv64imPublicProofOptions,
};

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

#[test]
fn rv64im_folded_statement_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (folded, proof) =
        prove_rv64im_folded_statement_from_accepted(&artifact).expect("prove rv64im folded statement");

    assert_eq!(folded.chunk_count, artifact.statement.chunk_count);
    assert_eq!(folded.fold_schedule, artifact.statement.fold_schedule);
    assert_ne!(folded.kernel_relation_digest, [0; 32]);
    assert_eq!(proof.steps.len(), folded.chunk_count as usize);

    verify_rv64im_folded_statement(&folded, &proof).expect("verify rv64im folded statement");
}

#[test]
fn rv64im_final_statement_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    assert_eq!(statement.public_statement_digest, artifact.statement.digest);
    assert_eq!(proof.steps.len(), statement.folded.chunk_count as usize);
    assert_ne!(proof.proof_digest, [0; 32]);
    assert_ne!(statement.digest, [0; 32]);

    verify_rv64im_final_statement(&statement, &proof).expect("verify rv64im final statement");
}

#[test]
fn rv64im_terminal_chunk_fold_witness_matches_final_accumulator() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let witness =
        build_rv64im_terminal_chunk_fold_witness(&statement, &proof).expect("build terminal chunk-fold witness");

    assert_eq!(
        witness.accumulator_final(),
        statement.folded.final_accumulator,
        "terminal chunk-fold witness must carry the authoritative final accumulator"
    );
    assert_eq!(
        witness.running_last.terminal_handle.0, witness.step_public.state_in,
        "terminal chunk-fold witness carry_in handle must match the terminal step public input"
    );
    assert_eq!(
        witness.running_final.terminal_handle.0, witness.step_public.state_out,
        "terminal chunk-fold witness carry_out handle must match the terminal step public output"
    );
    assert_eq!(
        witness.running_final.terminal_handle.0, statement.folded.final_accumulator.terminal_handle.0,
        "terminal chunk-fold witness carry_out handle must match the authoritative final accumulator handle"
    );
    assert!(
        witness.halted_out,
        "terminal chunk-fold witness must identify the last chunk as halted_out"
    );
    assert_eq!(
        witness.step_public.halted_out, witness.halted_out,
        "terminal chunk-fold witness halted flag must agree with the carried terminal step public"
    );
    assert!(
        !witness.fresh_last.fresh_claims.is_empty(),
        "terminal chunk-fold witness must carry a non-empty final fresh claim set"
    );
}

#[test]
#[ignore = "direct published-seam build remains too expensive for routine final-relation regression"]
fn rv64im_direct_prover_seam_matches_reconstructive_final_path() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let ((proof, published_seam), _) =
        prove_rv64im_public_proof_and_published_seam_with_perf(&input).expect("prove rv64im public proof and seam");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted rv64im artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    assert_eq!(published_seam.accepted_artifact.digest, artifact.digest);
    assert_eq!(
        published_seam
            .main_proof
            .final_statement_cache()
            .expect("locally built published seam should retain the final-statement cache")
            .digest,
        statement.digest
    );
    assert_ne!(
        published_seam
            .main_proof
            .published_statement()
            .expected_digest(),
        [0; 32]
    );
    assert_eq!(
        published_seam.main_proof.linkage_anchor_digest(),
        statement.public_statement_digest
    );

    verify_rv64im_final_statement(
        published_seam
            .main_proof
            .final_statement_cache()
            .expect("locally built published seam should retain the final-statement cache"),
        &final_proof,
    )
    .expect("verify rv64im direct final statement");
}

#[test]
fn rv64im_final_statement_kernel_export_matches_public_export_components() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted rv64im artifact");
    let (_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let expected_source =
        build_rv64im_kernel_export_source_from_accepted_artifact(&artifact).expect("build export source");
    let expected_witness = build_rv64im_kernel_export_witness(&proof).expect("build export witness");

    assert_eq!(final_proof.kernel_export.source.digest, expected_source.digest);
    assert_eq!(
        final_proof.kernel_export.source.public_statement_digest(),
        expected_source.public_statement_digest()
    );
    assert_eq!(final_proof.kernel_export.witness.digest, expected_witness.digest);
}

#[test]
fn rv64im_final_statement_rejects_tampered_accepted_artifact() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, mut proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    proof.kernel_export.source.root_execution.digest[0] ^= 1;

    let err = verify_rv64im_final_statement(&statement, &proof)
        .expect_err("tampered accepted artifact must fail final verification");
    assert!(format!("{err}").contains("accepted proof artifact") || format!("{err}").contains("digest"));
}

#[test]
fn rv64im_final_statement_rejects_tampered_chunk_replay_witness() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, mut proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    proof.steps[0].replay_witness.ccs_replay_proof.header_digest[0] ^= 1;

    let err = verify_rv64im_final_statement(&statement, &proof)
        .expect_err("tampered replay witness must fail final verification");
    assert!(format!("{err}").contains("final proof digest") || format!("{err}").contains("chunk"));
}

#[test]
fn rv64im_final_statement_digest_ignores_non_authoritative_replay_header_transport() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    let baseline =
        build_rv64im_decider_relation_from_final_surface(&statement, &proof).expect("build baseline decider relation");

    let mut tampered_proof = proof.clone();
    tampered_proof.steps[0]
        .replay_witness
        .ccs_replay_proof
        .header_digest[0] ^= 1;
    let tampered = build_rv64im_decider_relation_from_final_surface(&statement, &tampered_proof)
        .expect("build tampered decider relation");

    assert_eq!(
        baseline.final_proof_digest, tampered.final_proof_digest,
        "final proof digest must ignore non-authoritative Pi_CCS replay header transport"
    );
}

#[test]
fn rv64im_final_statement_rejects_tampered_bridge_binding() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, mut proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    proof.kernel_export.witness.chunk_handoffs[0]
        .bridge_handoff
        .step_bindings[0]
        .prepared_step_digest[0] ^= 1;

    let err = verify_rv64im_final_statement(&statement, &proof)
        .expect_err("tampered bridge binding must fail final verification");
    assert!(
        format!("{err}").contains("final proof digest")
            || format!("{err}").contains("bridge")
            || format!("{err}").contains("chunk transition")
    );
}

#[test]
fn rv64im_final_statement_rejects_tampered_transcript_surface() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, mut proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    proof
        .kernel_export
        .source
        .transcript
        .initial_state
        .registers[0] ^= 1;

    let err = verify_rv64im_final_statement(&statement, &proof)
        .expect_err("tampered transcript surface must fail final verification");
    assert!(format!("{err}").contains("transcript") || format!("{err}").contains("digest"));
}

#[test]
fn rv64im_final_statement_rejects_tampered_transcript_mix_surface() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, mut proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    proof.kernel_export.source.transcript.challenges.stage1_mix ^= 1;

    let err = verify_rv64im_final_statement(&statement, &proof)
        .expect_err("tampered transcript mix surface must fail final verification");
    assert!(format!("{err}").contains("transcript") || format!("{err}").contains("digest"));
}

#[test]
fn rv64im_final_statement_rejects_tampered_root_execution_rows() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, mut proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    proof.kernel_export.source.root_execution.execution_rows[0].next_pc ^= 1;
    proof.kernel_export.source.root_execution.digest = proof.kernel_export.source.root_execution.expected_digest();

    let err = verify_rv64im_final_statement(&statement, &proof)
        .expect_err("tampered root execution rows must fail final verification");
    assert!(
        format!("{err}").contains("root execution")
            || format!("{err}").contains("stage")
            || format!("{err}").contains("digest")
    );
}

#[test]
fn rv64im_final_statement_rejects_tampered_export_kernel_claim_surface() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, mut proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    proof.kernel_export.source.kernel_claims.final_state_digest[0] ^= 1;

    let err = verify_rv64im_final_statement(&statement, &proof)
        .expect_err("tampered export kernel claim surface must fail final verification");
    assert!(
        format!("{err}").contains("final state")
            || format!("{err}").contains("execution")
            || format!("{err}").contains("digest")
            || format!("{err}").contains("kernel opening")
    );
}

#[test]
fn rv64im_final_statement_rejects_tampered_export_terminal_row_endpoint() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, mut proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    let last_row = proof
        .kernel_export
        .source
        .root_execution
        .execution_rows
        .last_mut()
        .expect("root execution must carry a terminal row");
    last_row.next_pc ^= 1;
    proof.kernel_export.source.root_execution.digest = proof.kernel_export.source.root_execution.expected_digest();

    let err = verify_rv64im_final_statement(&statement, &proof)
        .expect_err("tampered export terminal row endpoint must fail final verification");
    assert!(
        format!("{err}").contains("root execution")
            || format!("{err}").contains("final pc")
            || format!("{err}").contains("kernel opening")
            || format!("{err}").contains("stage3")
            || format!("{err}").contains("root execution")
            || format!("{err}").contains("digest")
    );
}

#[test]
fn rv64im_final_statement_rejects_tampered_export_main_lane_surface() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof(&input).expect("prove accepted rv64im proof");
    let (statement, mut proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    proof
        .kernel_export
        .source
        .main_lane
        .packaged
        .statement
        .chunks[0]
        .start_index ^= 1;

    let err = verify_rv64im_final_statement(&statement, &proof)
        .expect_err("tampered export main-lane surface must fail final verification");
    assert!(
        format!("{err}").contains("main-lane")
            || format!("{err}").contains("digest")
            || format!("{err}").contains("public chunk")
            || format!("{err}").contains("packaged proof")
            || format!("{err}").contains("row-to-chunk routing"),
        "{err}"
    );
}

#[test]
fn rv64im_final_statement_rejects_reordered_chunk_witnesses() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let (artifact, _audit) = prove_rv64im_accepted_proof_with_options(
        &input,
        Rv64imPublicProofOptions {
            root_fold_schedule: FoldSchedule::RowsPerChunk(1),
        },
    )
    .expect("prove chunked accepted rv64im proof");
    let (statement, mut proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    assert!(proof.steps.len() > 1, "expected multi-chunk final proof");
    proof.steps.swap(0, 1);

    let err = verify_rv64im_final_statement(&statement, &proof)
        .expect_err("reordered chunk witnesses must fail final verification");
    assert!(
        format!("{err}").contains("final proof digest")
            || format!("{err}").contains("transition")
            || format!("{err}").contains("verify failed")
    );
}
