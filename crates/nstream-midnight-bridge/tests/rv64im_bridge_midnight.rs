use neo_fold_prototype::nightstream::rv64im::build_rv64im_nightstream_from_public_proof;
use neo_fold_prototype::proof::FoldSchedule;
use neo_fold_prototype::rv64im::{build_mixed_opcode_perf_source_case, prove_rv64im_public_proof, Rv64imProofInput};
use nstream_midnight_bridge::rv64im::{
    build_rv64im_nightstream_bridge_preimage, build_rv64im_nightstream_midnight_proof_preimage,
    build_rv64im_nightstream_verifier_ir_v2, check_rv64im_nightstream_verifier_ir_v2,
    decode_rv64im_nightstream_bridge_private_witness_fields, encode_rv64im_nightstream_bridge_private_witness_fields,
    rv64im_nightstream_bridge_binding_input, Rv64imNightstreamBridgePrivateWitness,
    Rv64imNightstreamBridgePublicInputs, RV64IM_NIGHTSTREAM_BRIDGE_KEY_LOCATION,
};
use transient_crypto::curve::Fr;

const DIGEST32_FIELD_WORDS: usize = 5;
const PROOF_BINDING_CLAIM_DIGESTS: usize = 5;

fn private_claim_words() -> usize {
    DIGEST32_FIELD_WORDS + DIGEST32_FIELD_WORDS + 2 + 1 + (PROOF_BINDING_CLAIM_DIGESTS * DIGEST32_FIELD_WORDS)
}

fn proof_binding_main_proof_digest_offset() -> usize {
    DIGEST32_FIELD_WORDS + DIGEST32_FIELD_WORDS + 2 + 1 + DIGEST32_FIELD_WORDS
}

fn statement_proof_binding_root_offset() -> usize {
    private_claim_words() + DIGEST32_FIELD_WORDS + DIGEST32_FIELD_WORDS + 2 + 1
}

fn statement_public_io_digest_offset() -> usize {
    private_claim_words()
}

fn proof_main_digest_offset() -> usize {
    private_claim_words() + DIGEST32_FIELD_WORDS + DIGEST32_FIELD_WORDS + 2 + 1 + DIGEST32_FIELD_WORDS
}

fn proof_published_statement_digest_offset() -> usize {
    private_claim_words()
        + DIGEST32_FIELD_WORDS
        + DIGEST32_FIELD_WORDS
        + 2
        + 1
        + DIGEST32_FIELD_WORDS
        + DIGEST32_FIELD_WORDS
        + DIGEST32_FIELD_WORDS
}

fn proof_input(_name: &str) -> Rv64imProofInput {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

#[test]
fn rv64im_bridge_builds_real_midnight_proof_preimage() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");
    let midnight_preimage =
        build_rv64im_nightstream_midnight_proof_preimage(&bridge_preimage).expect("build midnight preimage");

    assert_eq!(midnight_preimage.inputs.len(), bridge_preimage.inputs.len());
    assert_eq!(
        midnight_preimage.private_transcript.len(),
        bridge_preimage.private_transcript.len()
    );
    assert_eq!(
        midnight_preimage.binding_input,
        Fr::from(rv64im_nightstream_bridge_binding_input(
            Rv64imNightstreamBridgePublicInputs::new(&statement),
        ))
    );
    assert_eq!(
        midnight_preimage.key_location.0.as_ref(),
        RV64IM_NIGHTSTREAM_BRIDGE_KEY_LOCATION
    );
}

#[test]
fn rv64im_bridge_midnight_verifier_ir_checks_current_preimage() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");

    let pi_skips = check_rv64im_nightstream_verifier_ir_v2(&bridge_preimage).expect("check verifier ir");
    assert_eq!(pi_skips, Vec::<Option<usize>>::new());
}

#[test]
fn rv64im_bridge_midnight_verifier_ir_rejects_wrong_version_word() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");
    let mut midnight_preimage =
        build_rv64im_nightstream_midnight_proof_preimage(&bridge_preimage).expect("build midnight preimage");
    let ir = build_rv64im_nightstream_verifier_ir_v2(&bridge_preimage).expect("build verifier ir");

    midnight_preimage.inputs[0] = Fr::from(2u64);
    let err = midnight_preimage
        .check(&ir)
        .expect_err("wrong version word must fail verifier ir");
    let msg = format!("{err}");
    assert!(msg.contains("assert") || msg.contains("Assertion"));
}

#[test]
fn rv64im_bridge_midnight_verifier_ir_rejects_tampered_public_digest_word() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");
    let mut midnight_preimage =
        build_rv64im_nightstream_midnight_proof_preimage(&bridge_preimage).expect("build midnight preimage");
    let ir = build_rv64im_nightstream_verifier_ir_v2(&bridge_preimage).expect("build verifier ir");

    midnight_preimage.inputs[1] = Fr::from(0u64);
    let err = midnight_preimage
        .check(&ir)
        .expect_err("tampered public digest word must fail verifier ir");
    let msg = format!("{err}");
    assert!(msg.contains("assert") || msg.contains("Assertion"));
}

#[test]
fn rv64im_bridge_midnight_verifier_ir_rejects_tampered_main_proof_digest_claim() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");
    let ir = build_rv64im_nightstream_verifier_ir_v2(&bridge_preimage).expect("build verifier ir");
    let mut midnight_preimage =
        build_rv64im_nightstream_midnight_proof_preimage(&bridge_preimage).expect("build midnight preimage");

    let proof_digest_word = proof_binding_main_proof_digest_offset();
    midnight_preimage.private_transcript[proof_digest_word] = Fr::from(0u64);

    let err = midnight_preimage
        .check(&ir)
        .expect_err("tampered main proof digest must fail verifier ir");
    let msg = format!("{err}");
    assert!(msg.contains("assert") || msg.contains("Assertion"));
}

#[test]
fn rv64im_bridge_midnight_verifier_ir_rejects_tampered_statement_verifier_context_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");
    let ir = build_rv64im_nightstream_verifier_ir_v2(&bridge_preimage).expect("build verifier ir");
    let mut midnight_preimage =
        build_rv64im_nightstream_midnight_proof_preimage(&bridge_preimage).expect("build midnight preimage");

    let mut private_witness =
        decode_rv64im_nightstream_bridge_private_witness_fields(&bridge_preimage.private_transcript)
            .expect("decode private witness");
    private_witness.statement.verifier_context_digest[0] ^= 1;
    let tampered_private_words = encode_rv64im_nightstream_bridge_private_witness_fields(private_witness.borrowed())
        .expect("re-encode tampered private witness");
    midnight_preimage.private_transcript = tampered_private_words.into_iter().map(Fr::from).collect();

    let err = midnight_preimage
        .check(&ir)
        .expect_err("tampered verifier context digest must fail verifier ir");
    let msg = format!("{err}");
    assert!(msg.contains("assert") || msg.contains("Assertion"));
}

#[test]
fn rv64im_bridge_midnight_verifier_ir_rejects_tampered_statement_fold_schedule() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");
    let ir = build_rv64im_nightstream_verifier_ir_v2(&bridge_preimage).expect("build verifier ir");
    let mut midnight_preimage =
        build_rv64im_nightstream_midnight_proof_preimage(&bridge_preimage).expect("build midnight preimage");

    let mut private_witness =
        decode_rv64im_nightstream_bridge_private_witness_fields(&bridge_preimage.private_transcript)
            .expect("decode private witness");
    private_witness.statement.fold_schedule = match private_witness.statement.fold_schedule {
        FoldSchedule::WholeTrace => FoldSchedule::RowsPerChunk(1),
        FoldSchedule::RowsPerChunk(rows) => FoldSchedule::RowsPerChunk(rows + 1),
    };
    let tampered_private_words = encode_rv64im_nightstream_bridge_private_witness_fields(private_witness.borrowed())
        .expect("re-encode tampered private witness");
    midnight_preimage.private_transcript = tampered_private_words.into_iter().map(Fr::from).collect();

    let err = midnight_preimage
        .check(&ir)
        .expect_err("tampered fold schedule must fail verifier ir");
    let msg = format!("{err}");
    assert!(msg.contains("assert") || msg.contains("Assertion"));
}

#[test]
fn rv64im_bridge_midnight_verifier_ir_rejects_tampered_statement_public_io_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");
    let ir = build_rv64im_nightstream_verifier_ir_v2(&bridge_preimage).expect("build verifier ir");
    let mut midnight_preimage =
        build_rv64im_nightstream_midnight_proof_preimage(&bridge_preimage).expect("build midnight preimage");

    let public_io_word = statement_public_io_digest_offset();
    midnight_preimage.private_transcript[public_io_word] = Fr::from(0u64);

    let err = midnight_preimage
        .check(&ir)
        .expect_err("tampered statement public IO digest must fail verifier ir");
    let msg = format!("{err}");
    assert!(msg.contains("assert") || msg.contains("Assertion"));
}

#[test]
fn rv64im_bridge_midnight_verifier_ir_rejects_tampered_statement_proof_binding_root() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");
    let ir = build_rv64im_nightstream_verifier_ir_v2(&bridge_preimage).expect("build verifier ir");
    let mut midnight_preimage =
        build_rv64im_nightstream_midnight_proof_preimage(&bridge_preimage).expect("build midnight preimage");

    let proof_binding_root_word = statement_proof_binding_root_offset();
    midnight_preimage.private_transcript[proof_binding_root_word] = Fr::from(0u64);

    let err = midnight_preimage
        .check(&ir)
        .expect_err("tampered statement proof binding root must fail verifier ir");
    let msg = format!("{err}");
    assert!(msg.contains("assert") || msg.contains("Assertion"));
}

#[test]
fn rv64im_bridge_midnight_verifier_ir_rejects_tampered_proof_main_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");
    let ir = build_rv64im_nightstream_verifier_ir_v2(&bridge_preimage).expect("build verifier ir");
    let mut midnight_preimage =
        build_rv64im_nightstream_midnight_proof_preimage(&bridge_preimage).expect("build midnight preimage");

    let proof_main_digest_word = proof_main_digest_offset();
    midnight_preimage.private_transcript[proof_main_digest_word] = Fr::from(0u64);

    let err = midnight_preimage
        .check(&ir)
        .expect_err("tampered proof main digest must fail verifier ir");
    let msg = format!("{err}");
    assert!(msg.contains("assert") || msg.contains("Assertion"));
}

#[test]
fn rv64im_bridge_midnight_verifier_ir_rejects_tampered_proof_published_statement_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");
    let ir = build_rv64im_nightstream_verifier_ir_v2(&bridge_preimage).expect("build verifier ir");
    let mut midnight_preimage =
        build_rv64im_nightstream_midnight_proof_preimage(&bridge_preimage).expect("build midnight preimage");

    let published_statement_digest_word = proof_published_statement_digest_offset();
    midnight_preimage.private_transcript[published_statement_digest_word] = Fr::from(0u64);

    let err = midnight_preimage
        .check(&ir)
        .expect_err("tampered proof published statement digest must fail verifier ir");
    let msg = format!("{err}");
    assert!(msg.contains("assert") || msg.contains("Assertion"));
}

#[test]
fn rv64im_bridge_midnight_verifier_ir_rejects_tampered_statement_semantic_step_count() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let bridge_preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&statement),
        Rv64imNightstreamBridgePrivateWitness::new(
            &statement,
            &proof,
            public_proof.statement.root_params_id,
            public_proof.statement.recompute_digest(),
        ),
    )
    .expect("build bridge preimage");
    let ir = build_rv64im_nightstream_verifier_ir_v2(&bridge_preimage).expect("build verifier ir");
    let mut midnight_preimage =
        build_rv64im_nightstream_midnight_proof_preimage(&bridge_preimage).expect("build midnight preimage");

    let semantic_step_count_word = private_claim_words() + DIGEST32_FIELD_WORDS + DIGEST32_FIELD_WORDS + 2;
    midnight_preimage.private_transcript[semantic_step_count_word] = Fr::from(0u64);

    let err = midnight_preimage
        .check(&ir)
        .expect_err("tampered statement semantic step count must fail verifier ir");
    let msg = format!("{err}");
    assert!(msg.contains("assert") || msg.contains("Assertion"));
}
