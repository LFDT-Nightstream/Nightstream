use neo_fold_prototype::nightstream::rv64im::build_rv64im_nightstream_from_public_proof;
use neo_fold_prototype::rv64im::{build_mixed_opcode_perf_source_case, prove_rv64im_public_proof, Rv64imProofInput};
use nstream_midnight_bridge::rv64im::{
    verify_rv64im_nightstream_bridge_input, Rv64imNightstreamBridgePrivateWitness, Rv64imNightstreamBridgePublicInputs,
    RV64IM_NIGHTSTREAM_BRIDGE_VERSION,
};

fn proof_input(_name: &str) -> Rv64imProofInput {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

fn bridge_witness<'a>(
    statement: &'a neo_fold_prototype::nightstream::NightstreamStatement,
    proof: &'a neo_fold_prototype::nightstream::rv64im::Rv64imNightstreamProof,
    public_proof: &'a neo_fold_prototype::rv64im::Rv64imProof,
) -> Rv64imNightstreamBridgePrivateWitness<'a> {
    Rv64imNightstreamBridgePrivateWitness {
        statement,
        proof,
        trusted_root_params_id: public_proof.statement.root_params_id,
        public_statement_digest: public_proof.statement.recompute_digest(),
    }
}

#[test]
fn rv64im_bridge_reference_verifier_accepts_current_nightstream_boundary() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let public_inputs = Rv64imNightstreamBridgePublicInputs::new(&statement);
    let private_witness = bridge_witness(&statement, &proof, &public_proof);
    verify_rv64im_nightstream_bridge_input(public_inputs, private_witness)
        .expect("bridge reference verifier accepts current boundary");
}

#[test]
fn rv64im_bridge_reference_verifier_rejects_wrong_version() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    let public_inputs = Rv64imNightstreamBridgePublicInputs {
        version: RV64IM_NIGHTSTREAM_BRIDGE_VERSION + 1,
        statement_digest: statement.digest(),
    };
    let private_witness = bridge_witness(&statement, &proof, &public_proof);
    let err =
        verify_rv64im_nightstream_bridge_input(public_inputs, private_witness).expect_err("wrong version must fail");
    assert!(format!("{err}").contains("unsupported RV64IM Nightstream bridge version"));
}

#[test]
fn rv64im_bridge_reference_verifier_rejects_tampered_public_boundary() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (statement, mut proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    proof
        .main_proof_mut()
        .ivc_recursion_snark_proof_mut()
        .snark_data[0] ^= 1;
    let public_inputs = Rv64imNightstreamBridgePublicInputs::new(&statement);
    let private_witness = bridge_witness(&statement, &proof, &public_proof);
    let err = verify_rv64im_nightstream_bridge_input(public_inputs, private_witness)
        .expect_err("tampered boundary must fail");
    assert!(format!("{err}").contains("proof binding root"));
}

#[test]
fn rv64im_bridge_reference_verifier_rejects_verifier_context_mismatch() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (mut statement, proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");

    statement.verifier_context_digest[0] ^= 1;

    let public_inputs = Rv64imNightstreamBridgePublicInputs::new(&statement);
    let private_witness = bridge_witness(&statement, &proof, &public_proof);
    let err = verify_rv64im_nightstream_bridge_input(public_inputs, private_witness)
        .expect_err("verifier-context mismatch must fail");
    assert!(format!("{err}").contains("verifier-context digest"));
}
