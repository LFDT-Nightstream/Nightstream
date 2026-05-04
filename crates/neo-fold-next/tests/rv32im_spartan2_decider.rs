//! Focused tests for the live RV32IM main-recursion Spartan SNARK.

use std::sync::Arc;

use neo_fold_next::nightstream::rv32im::audit::build_rv32im_nightstream_statement_from_final;
use neo_fold_next::nightstream::rv32im::rv32im_verifier_context_digest;
use neo_fold_next::rv32im::audit::{
    build_rv32im_ivc_recursion_snark_setup_shape_from_components, debug_check_rv32im_ivc_recursion_snark_circuit,
};
use neo_fold_next::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_next::rv32im::main_proof::Rv32imCompressedMainProof;
use neo_fold_next::rv32im::{
    build_rv32im_accepted_proof_artifact, parity_source_cases, prove_rv32im_public_proof,
    setup_rv32im_ivc_snark_from_final, setup_rv32im_ivc_snark_from_final_cached, Rv32imChunkStepIvcStatement,
    Rv32imIvcPublicImage, Rv32imIvcSnark, Rv32imIvcSnarkVerifierKey, Rv32imProofInput, SimpleKernelError,
};
use neo_math::{F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn source_case(name: &str) -> neo_fold_next::rv32im::Rv32imParitySourceCase {
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

fn proof_input_with_transcript_suffix(name: &str, suffix: &[u8]) -> Rv32imProofInput {
    let mut source = source_case(name);
    source.transcript_seed.extend_from_slice(suffix);
    let max_steps = source.program_words.len();
    Rv32imProofInput { source, max_steps }
}

fn final_fixture_from_input(
    input: Rv32imProofInput,
) -> (
    neo_fold_next::rv32im::Rv32imProof,
    neo_fold_next::rv32im::final_relation::Rv32imFinalStatement,
    neo_fold_next::rv32im::final_relation::Rv32imFinalBuildProof,
    neo_fold_next::nightstream::NightstreamStatement,
) {
    let proof = prove_rv32im_public_proof(&input).expect("prove rv32im public proof");
    let artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv32im_final_statement_from_accepted(&artifact).expect("prove rv32im final statement");
    let published_statement = neo_fold_next::rv32im::Rv32imAccumulatorPublicStatement::from_final_artifacts(
        &statement,
        &final_proof,
        proof.statement.final_pc,
    )
    .expect("build published statement");
    let ivc_recursion_snark_keys =
        setup_rv32im_ivc_snark_from_final_cached(&statement, &final_proof).expect("setup IVC recursion SNARK");
    let nightstream_statement = build_rv32im_nightstream_statement_from_final(
        proof.statement.digest,
        rv32im_verifier_context_digest(
            proof.statement.root_params_id,
            &published_statement,
            &ivc_recursion_snark_keys.as_ref().1,
        )
        .expect("digest verifier context"),
        &statement,
        &final_proof,
        [0; 32],
    )
    .expect("build nightstream statement");
    (proof, statement, final_proof, nightstream_statement)
}

fn final_fixture(
    name: &str,
) -> (
    neo_fold_next::rv32im::Rv32imProof,
    neo_fold_next::rv32im::final_relation::Rv32imFinalStatement,
    neo_fold_next::rv32im::final_relation::Rv32imFinalBuildProof,
    neo_fold_next::nightstream::NightstreamStatement,
) {
    final_fixture_from_input(proof_input(name))
}

fn compressed_main_proof_fixture(
    statement: &neo_fold_next::rv32im::final_relation::Rv32imFinalStatement,
    final_proof: &neo_fold_next::rv32im::final_relation::Rv32imFinalBuildProof,
    final_pc: u64,
) -> Rv32imCompressedMainProof {
    Rv32imCompressedMainProof::from_final_artifacts(statement, final_proof, final_pc)
        .expect("build compressed main proof")
}

fn verify_compressed_main_proof(
    proof: &Rv32imCompressedMainProof,
    vk: &Rv32imIvcSnarkVerifierKey,
) -> Result<(), SimpleKernelError> {
    let public_image = proof.expected_ivc_public_image()?;
    proof.ivc_snark().verify(vk, &public_image)
}

fn expect_compressed_verify_accepts(result: Result<(), SimpleKernelError>) {
    result.expect("compressed verifier must accept the final SuperNeo Construction-2 boundary");
}

fn flip_first_byte(bytes: &mut Vec<u8>) {
    if let Some(first) = bytes.first_mut() {
        *first ^= 1;
    } else {
        bytes.push(1);
    }
}

fn verified_step_statement_digest(statement: &Rv32imChunkStepIvcStatement) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement");
    tr.append_message(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/version",
        b"v2",
    );
    tr.append_u64s(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/meta",
        &[
            statement.step_public.chunk_index,
            statement.step_public.step_lo,
            statement.step_public.step_hi,
            u64::from(statement.step_public.halted_out),
        ],
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/state_in",
        &digest32_as_fields(statement.step_public.state_in),
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/state_out",
        &digest32_as_fields(statement.step_public.state_out),
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/public_chunk_digest",
        &digest32_as_fields(statement.chunk_summary.public_chunk_digest),
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/chunk_relation_digest",
        &digest32_as_fields(statement.chunk_summary.chunk_relation_digest),
    );
    tr.digest32()
}

fn digest32_as_fields(digest: [u8; 32]) -> [F; 4] {
    [
        F::from_u64(u64::from_le_bytes(digest[0..8].try_into().expect("digest limb 0"))),
        F::from_u64(u64::from_le_bytes(digest[8..16].try_into().expect("digest limb 1"))),
        F::from_u64(u64::from_le_bytes(digest[16..24].try_into().expect("digest limb 2"))),
        F::from_u64(u64::from_le_bytes(digest[24..32].try_into().expect("digest limb 3"))),
    ]
}

fn expect_direct_public_image_tamper_rejected(
    snark: &Rv32imIvcSnark,
    vk: &Rv32imIvcSnarkVerifierKey,
    mutate: impl FnOnce(&mut Rv32imIvcPublicImage),
    message: &str,
) {
    let mut tampered_snark = snark.clone();
    let mut public_image = tampered_snark.public_image().clone();
    mutate(&mut public_image);
    *tampered_snark.public_image_mut() = public_image.clone();
    tampered_snark.verify(vk, &public_image).expect_err(message);
}

#[test]
fn rv32im_spartan2_decider_setup_shape_is_transcript_seed_invariant() {
    let (_proof_a, statement_a, final_proof_a, _) = final_fixture("control_flow_jal_skip_ecall");
    let (_proof_b, statement_b, final_proof_b, _) = final_fixture_from_input(proof_input_with_transcript_suffix(
        "control_flow_jal_skip_ecall",
        b"/goal3-alt",
    ));

    let shape_a = build_rv32im_ivc_recursion_snark_setup_shape_from_components(
        &statement_a,
        final_proof_a.proof_digest,
        &final_proof_a.kernel_export,
        &final_proof_a.chunk_summaries,
        &final_proof_a.steps,
    )
    .expect("build setup shape A");
    let shape_b = build_rv32im_ivc_recursion_snark_setup_shape_from_components(
        &statement_b,
        final_proof_b.proof_digest,
        &final_proof_b.kernel_export,
        &final_proof_b.chunk_summaries,
        &final_proof_b.steps,
    )
    .expect("build setup shape B");

    assert_eq!(
        shape_a, shape_b,
        "same control-flow program with a different transcript seed must derive the same recursion SNARK setup shape"
    );
}

#[test]
fn rv32im_spartan2_decider_setup_shape_rejects_tampered_public_statement_digest() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let mut tampered_statement = statement.clone();
    tampered_statement.public_statement_digest[0] ^= 1;

    let err = build_rv32im_ivc_recursion_snark_setup_shape_from_components(
        &tampered_statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect_err("tampered final statement digest must fail setup-shape derivation");
    assert!(
        format!("{err}").contains("final statement digest mismatch"),
        "expected final statement digest mismatch, got: {err}"
    );
}

#[test]
fn rv32im_spartan2_decider_setup_shape_rejects_tampered_replay_header_transport() {
    let (_proof, statement, mut final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    final_proof.steps[0]
        .replay_witness
        .ccs_replay_proof
        .header_digest[0] ^= 1;

    let err = build_rv32im_ivc_recursion_snark_setup_shape_from_components(
        &statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect_err("tampered replay header transport must fail setup-shape derivation");
    assert!(
        format!("{err}").contains("header digest does not match transcript replay"),
        "expected replay header digest mismatch, got: {err}"
    );
}

#[test]
#[ignore = "expensive: main-recursion SNARK debug synthesis exceeds developer-memory budget"]
fn rv32im_spartan2_decider_debug_check_only() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    debug_check_rv32im_ivc_recursion_snark_circuit(&statement, &final_proof)
        .expect("rv32im main-recursion SNARK circuit must be satisfied");
}

#[test]
#[ignore = "expensive: main-recursion Spartan setup exceeds developer-memory budget"]
fn rv32im_spartan2_decider_setup_only() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let _ = setup_rv32im_ivc_snark_from_final(&statement, &final_proof).expect("setup rv32im recursion SNARK");
}

#[test]
#[ignore = "expensive: main-recursion Spartan setup exceeds developer-memory budget"]
fn rv32im_spartan2_decider_cached_setup_reuses_same_final_seam() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let first = setup_rv32im_ivc_snark_from_final_cached(&statement, &final_proof)
        .expect("setup cached rv32im recursion SNARK");
    let second = setup_rv32im_ivc_snark_from_final_cached(&statement, &final_proof)
        .expect("setup cached rv32im recursion SNARK");
    assert!(
        Arc::ptr_eq(&first, &second),
        "exact same final seam should reuse cached setup"
    );
}

#[test]
#[ignore = "expensive: main-recursion Spartan setup exceeds developer-memory budget"]
fn rv32im_spartan2_decider_setup_from_shape_is_reproducible() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let (_first_pk, first_vk) =
        setup_rv32im_ivc_snark_from_final(&statement, &final_proof).expect("setup recursion SNARK");
    let (_second_pk, second_vk) =
        setup_rv32im_ivc_snark_from_final(&statement, &final_proof).expect("setup recursion SNARK");
    let compressed_main_proof = compressed_main_proof_fixture(&statement, &final_proof, proof.statement.final_pc);

    expect_compressed_verify_accepts(verify_compressed_main_proof(&compressed_main_proof, &first_vk));
    expect_compressed_verify_accepts(verify_compressed_main_proof(&compressed_main_proof, &second_vk));
}

#[test]
#[ignore = "expensive: main-recursion Spartan setup exceeds developer-memory budget"]
fn rv32im_spartan2_decider_cached_shape_setup_is_deterministic() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let first = setup_rv32im_ivc_snark_from_final_cached(&statement, &final_proof).expect("cached recursion setup");
    let second = setup_rv32im_ivc_snark_from_final_cached(&statement, &final_proof).expect("cached recursion setup");
    assert!(
        Arc::ptr_eq(&first, &second),
        "exact same setup shape should reuse cached setup"
    );
}

#[test]
#[ignore = "expensive: main-recursion Spartan round-trip exceeds developer-memory budget"]
fn rv32im_spartan2_decider_round_trip_without_replay_verifier_input() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");

    let (_pk, vk) = setup_rv32im_ivc_snark_from_final(&statement, &final_proof).expect("setup rv32im recursion SNARK");
    let compressed_main_proof = compressed_main_proof_fixture(&statement, &final_proof, proof.statement.final_pc);

    expect_compressed_verify_accepts(verify_compressed_main_proof(&compressed_main_proof, &vk));
    assert!(
        compressed_main_proof
            .ivc_recursion_snark_proof()
            .snark_bytes_len()
            > 0
    );
}

#[test]
#[ignore = "expensive: main-recursion Spartan setup exceeds developer-memory budget"]
fn rv32im_spartan2_decider_rejects_tampered_chunk_relation_digest() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let (_pk, vk) = setup_rv32im_ivc_snark_from_final(&statement, &final_proof).expect("setup rv32im recursion SNARK");
    let mut compressed_main_proof = compressed_main_proof_fixture(&statement, &final_proof, proof.statement.final_pc);

    compressed_main_proof
        .published_statement_mut()
        .terminal_step_statement_mut()
        .chunk_summary
        .chunk_relation_digest[0] ^= 1;
    let err = verify_compressed_main_proof(&compressed_main_proof, &vk)
        .expect_err("tampered terminal chunk relation digest must fail");
    assert!(format!("{err}").contains("relation digest") || format!("{err}").contains("chunk"));
}

#[test]
#[ignore = "expensive: main-recursion Spartan setup exceeds developer-memory budget"]
fn rv32im_spartan2_decider_rejects_tampered_final_claim() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");

    let (_pk, vk) = setup_rv32im_ivc_snark_from_final(&statement, &final_proof).expect("setup rv32im recursion SNARK");
    let mut compressed_main_proof = compressed_main_proof_fixture(&statement, &final_proof, proof.statement.final_pc);

    expect_compressed_verify_accepts(verify_compressed_main_proof(&compressed_main_proof, &vk));

    compressed_main_proof
        .ivc_snark_mut()
        .public_image_mut()
        .folded_accumulator_digest[0] ^= 1;
    let err = verify_compressed_main_proof(&compressed_main_proof, &vk)
        .expect_err("tampered folded accumulator digest must fail");
    assert!(
        format!("{err}").contains("public image") || format!("{err}").contains("digest"),
        "expected public-image failure, got: {err}"
    );
}

#[test]
#[ignore = "expensive: main-recursion Spartan setup exceeds developer-memory budget"]
fn rv32im_spartan2_decider_rejects_public_image_and_proof_tampers() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let (_pk, vk) = setup_rv32im_ivc_snark_from_final(&statement, &final_proof).expect("setup rv32im recursion SNARK");
    let compressed_main_proof = compressed_main_proof_fixture(&statement, &final_proof, proof.statement.final_pc);

    expect_compressed_verify_accepts(verify_compressed_main_proof(&compressed_main_proof, &vk));

    let mut tampered_x = compressed_main_proof.clone();
    tampered_x
        .ivc_snark_mut()
        .public_image_mut()
        .x_i
        .bytes_mut()[0] ^= 1;
    verify_compressed_main_proof(&tampered_x, &vk)
        .expect_err("tampered public-image x_i must fail without native replay fallback");

    let mut tampered_construction2_x = compressed_main_proof.clone();
    tampered_construction2_x
        .ivc_snark_mut()
        .public_image_mut()
        .construction2_u_i
        .x_i
        .bytes_mut()[0] ^= 1;
    verify_compressed_main_proof(&tampered_construction2_x, &vk)
        .expect_err("tampered public Construction-2 u_i.x_i must fail without native replay fallback");

    let mut tampered_z0 = compressed_main_proof.clone();
    tampered_z0.ivc_snark_mut().public_image_mut().z_0[0] ^= 1;
    verify_compressed_main_proof(&tampered_z0, &vk)
        .expect_err("tampered public-image z0 must fail without native replay fallback");

    let mut tampered_zi = compressed_main_proof.clone();
    tampered_zi.ivc_snark_mut().public_image_mut().z_i[0] ^= 1;
    verify_compressed_main_proof(&tampered_zi, &vk)
        .expect_err("tampered public-image zi must fail without native replay fallback");

    let mut tampered_pc = compressed_main_proof.clone();
    tampered_pc.ivc_snark_mut().public_image_mut().pc = 2;
    verify_compressed_main_proof(&tampered_pc, &vk)
        .expect_err("tampered public-image pc must fail without native replay fallback");

    let mut tampered_bridge = compressed_main_proof.clone();
    tampered_bridge
        .ivc_snark_mut()
        .public_image_mut()
        .terminal_bridge_handoff_digest[0] ^= 1;
    verify_compressed_main_proof(&tampered_bridge, &vk)
        .expect_err("tampered terminal bridge handoff digest must fail without native replay fallback");

    let mut tampered_verified_step_digest = compressed_main_proof.clone();
    tampered_verified_step_digest
        .ivc_snark_mut()
        .public_image_mut()
        .terminal_verified_step_statement_digest[0] ^= 1;
    verify_compressed_main_proof(&tampered_verified_step_digest, &vk)
        .expect_err("tampered terminal verified-step digest must fail without native replay fallback");

    let mut tampered_proof = compressed_main_proof.clone();
    flip_first_byte(
        &mut tampered_proof
            .ivc_snark_mut()
            .proof_mut()
            .terminal_f_prime_committed_step_proof
            .snark_data,
    );
    verify_compressed_main_proof(&tampered_proof, &vk)
        .expect_err("tampered terminal F' committed-step proof bytes must fail without native replay fallback");

    let mut tampered_final_ce_proof = compressed_main_proof.clone();
    flip_first_byte(
        &mut tampered_final_ce_proof
            .ivc_snark_mut()
            .proof_mut()
            .final_ce_proof
            .snark_data,
    );
    verify_compressed_main_proof(&tampered_final_ce_proof, &vk)
        .expect_err("tampered final CE bundle proof bytes must fail without native replay fallback");

    let mut tampered_final_ce_claim = compressed_main_proof.clone();
    tampered_final_ce_claim
        .ivc_snark_mut()
        .proof_mut()
        .final_main_claims[0]
        .c
        .data[0] += F::from_u64(1);
    verify_compressed_main_proof(&tampered_final_ce_claim, &vk)
        .expect_err("tampered final carried CE commitment must fail without native replay fallback");

    let mut tampered_final_ce_x = compressed_main_proof.clone();
    tampered_final_ce_x
        .ivc_snark_mut()
        .proof_mut()
        .final_main_claims[0]
        .X[(0, 0)] += F::ONE;
    verify_compressed_main_proof(&tampered_final_ce_x, &vk)
        .expect_err("tampered final carried CE X field must fail without native replay fallback");

    let mut tampered_final_ce_r = compressed_main_proof.clone();
    tampered_final_ce_r
        .ivc_snark_mut()
        .proof_mut()
        .final_main_claims[0]
        .r[0] += K::ONE;
    verify_compressed_main_proof(&tampered_final_ce_r, &vk)
        .expect_err("tampered final carried CE evaluation point must fail without native replay fallback");

    let mut tampered_final_ce_y_ring = compressed_main_proof.clone();
    tampered_final_ce_y_ring
        .ivc_snark_mut()
        .proof_mut()
        .final_main_claims[0]
        .y_ring[0][0] += K::ONE;
    verify_compressed_main_proof(&tampered_final_ce_y_ring, &vk)
        .expect_err("tampered final carried CE y_ring field must fail without native replay fallback");

    let mut tampered_final_ce_transport = compressed_main_proof.clone();
    tampered_final_ce_transport
        .ivc_snark_mut()
        .proof_mut()
        .final_main_claims[0]
        .ct
        .push(K::ONE);
    verify_compressed_main_proof(&tampered_final_ce_transport, &vk)
        .expect_err("theorem-facing final CE claim must reject non-authoritative transport cargo");

    let mut tampered_construction2_commitment = compressed_main_proof.clone();
    tampered_construction2_commitment
        .ivc_snark_mut()
        .public_image_mut()
        .construction2_u_i
        .commitment_data[0] += F::from_u64(1);
    verify_compressed_main_proof(&tampered_construction2_commitment, &vk)
        .expect_err("tampered public Construction-2 u_i.C commitment data must fail without native replay fallback");
}

#[test]
#[ignore = "expensive: main-recursion Spartan setup exceeds developer-memory budget"]
fn rv32im_spartan2_decider_rejects_coherent_terminal_metadata_tamper_at_snark_boundary() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let (_pk, vk) = setup_rv32im_ivc_snark_from_final(&statement, &final_proof).expect("setup rv32im recursion SNARK");
    let compressed_main_proof = compressed_main_proof_fixture(&statement, &final_proof, proof.statement.final_pc);
    let mut snark = compressed_main_proof.ivc_snark().clone();
    let mut public_image = snark.public_image().clone();

    public_image
        .terminal_statement
        .as_mut()
        .expect("terminal statement")
        .chunk_summary
        .chunk_relation_digest[0] ^= 1;
    public_image.terminal_verified_step_statement_digest = verified_step_statement_digest(
        public_image
            .terminal_statement
            .as_ref()
            .expect("terminal statement"),
    );
    public_image
        .validate_final_construction2_public_boundary()
        .expect("coherently re-digested terminal metadata remains a structurally valid public image");
    *snark.public_image_mut() = public_image.clone();

    snark
        .verify(&vk, &public_image)
        .expect_err("coherently tampered terminal metadata must fail against recursive-step SNARK public IO");
}

#[test]
#[ignore = "expensive: main-recursion Spartan setup exceeds developer-memory budget"]
fn rv32im_spartan2_decider_direct_snark_verify_rejects_public_boundary_tampers() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let (_pk, vk) = setup_rv32im_ivc_snark_from_final(&statement, &final_proof).expect("setup rv32im recursion SNARK");
    let compressed_main_proof = compressed_main_proof_fixture(&statement, &final_proof, proof.statement.final_pc);
    let snark = compressed_main_proof.ivc_snark();
    let public_image = snark.public_image().clone();

    expect_compressed_verify_accepts(snark.verify(&vk, &public_image));

    expect_direct_public_image_tamper_rejected(
        snark,
        &vk,
        |image| image.folded_accumulator_digest[0] ^= 1,
        "direct IVC SNARK verify must reject a bad folded accumulator digest",
    );
    expect_direct_public_image_tamper_rejected(
        snark,
        &vk,
        |image| image.terminal_bridge_handoff_digest[0] ^= 1,
        "direct IVC SNARK verify must reject a bad terminal bridge handoff digest",
    );
    expect_direct_public_image_tamper_rejected(
        snark,
        &vk,
        |image| {
            image.construction2_u_i.commitment_data[0] += F::ONE;
            image.construction2_u_i.commitment_digest = image.construction2_u_i.expected_commitment_digest();
            image.construction2_u_i.fresh_instance_digest = image.construction2_u_i.expected_fresh_instance_digest();
        },
        "direct IVC SNARK verify must reject a coherently re-digested bad u_i.C",
    );
    expect_direct_public_image_tamper_rejected(
        snark,
        &vk,
        |image| {
            image
                .terminal_statement
                .as_mut()
                .expect("terminal statement")
                .chunk_summary
                .public_chunk_digest[0] ^= 1;
            image.terminal_verified_step_statement_digest = verified_step_statement_digest(
                image
                    .terminal_statement
                    .as_ref()
                    .expect("terminal statement"),
            );
        },
        "direct IVC SNARK verify must reject coherently re-digested terminal public chunk metadata",
    );
}

#[test]
#[ignore = "expensive: builds two main-recursion Spartan proofs"]
fn rv32im_spartan2_decider_rejects_terminal_proof_from_unrelated_chain() {
    let (proof_a, statement_a, final_proof_a, _) = final_fixture("control_flow_jal_skip_ecall");
    let (proof_b, statement_b, final_proof_b, _) = final_fixture_from_input(proof_input_with_transcript_suffix(
        "control_flow_jal_skip_ecall",
        b"/wrong-terminal-proof",
    ));
    let (_pk, vk) = setup_rv32im_ivc_snark_from_final(&statement_a, &final_proof_a).expect("setup recursion SNARK");
    let mut compressed_a = compressed_main_proof_fixture(&statement_a, &final_proof_a, proof_a.statement.final_pc);
    let compressed_b = compressed_main_proof_fixture(&statement_b, &final_proof_b, proof_b.statement.final_pc);

    expect_compressed_verify_accepts(verify_compressed_main_proof(&compressed_a, &vk));
    compressed_a
        .ivc_snark_mut()
        .proof_mut()
        .terminal_f_prime_committed_step_proof = compressed_b
        .ivc_recursion_snark_proof()
        .terminal_f_prime_committed_step_proof
        .clone();
    verify_compressed_main_proof(&compressed_a, &vk)
        .expect_err("terminal F' committed-step proof from another chain must fail");
}
