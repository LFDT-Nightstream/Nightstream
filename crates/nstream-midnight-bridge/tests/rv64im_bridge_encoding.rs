use neo_fold_prototype::nightstream::rv64im::{build_rv64im_nightstream_from_public_proof, Rv64imNightstreamProof};
use neo_fold_prototype::nightstream::NightstreamStatement;
use neo_fold_prototype::rv64im::{
    build_mixed_opcode_perf_source_case, prove_rv64im_public_proof, Rv64imProof, Rv64imProofInput,
};
use nstream_midnight_bridge::rv64im::{
    build_rv64im_nightstream_bridge_preimage, build_rv64im_nightstream_check_request,
    build_rv64im_nightstream_check_request_body, build_rv64im_nightstream_proof_server_request_body,
    build_rv64im_nightstream_prove_request, decode_rv64im_nightstream_bridge_private_witness_fields,
    decode_rv64im_nightstream_bridge_public_inputs_fields, encode_rv64im_nightstream_bridge_private_witness_fields,
    encode_rv64im_nightstream_bridge_public_inputs_fields, execute_rv64im_nightstream_check_request,
    execute_rv64im_nightstream_prove_request, execute_rv64im_proof_server_request,
    parse_rv64im_nightstream_check_response_body, parse_rv64im_nightstream_prove_response_body,
    rv64im_nightstream_bridge_binding_input, verify_rv64im_nightstream_bridge_payload,
    verify_rv64im_nightstream_bridge_preimage, Rv64imBridgeError, Rv64imNightstreamBridgePrivateWitness,
    Rv64imNightstreamBridgePublicInputs, Rv64imProofServerCheckRequestPolicy, Rv64imProofServerCheckResponse,
    Rv64imProofServerPreimageVersioned, Rv64imProofServerProofVersioned, Rv64imProofServerProveRequestPolicy,
    Rv64imProofServerProvider, Rv64imProofServerRequest, Rv64imProofServerResolverPolicy, Rv64imProofServerResponse,
    Rv64imProofServerRoute, Rv64imProofServerWrappedIr, RV64IM_NIGHTSTREAM_BRIDGE_KEY_LOCATION,
};
use serialize::{tagged_deserialize, tagged_serialize};
use std::sync::OnceLock;
use transient_crypto::curve::Fr;
use transient_crypto::proofs::{Proof, ProvingKeyMaterial};

fn proof_input(_name: &str) -> Rv64imProofInput {
    let source = build_mixed_opcode_perf_source_case(0);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

struct BridgeFixture {
    public_proof: Rv64imProof,
    statement: NightstreamStatement,
    proof: Rv64imNightstreamProof,
}

fn bridge_fixture() -> &'static BridgeFixture {
    static FIXTURE: OnceLock<BridgeFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let input = proof_input("control_flow_jal_skip_ecall");
        let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
        let (statement, proof) =
            build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream boundary");
        BridgeFixture {
            public_proof,
            statement,
            proof,
        }
    })
}

fn bridge_witness(fixture: &BridgeFixture) -> Rv64imNightstreamBridgePrivateWitness<'_> {
    Rv64imNightstreamBridgePrivateWitness {
        statement: &fixture.statement,
        proof: &fixture.proof,
        trusted_root_params_id: fixture.public_proof.statement.root_params_id,
        public_statement_digest: fixture.public_proof.statement.recompute_digest(),
    }
}

struct MockProofServerProvider {
    expected_path: &'static str,
    expected_body: Vec<u8>,
    response_body: Vec<u8>,
}

impl Rv64imProofServerProvider for MockProofServerProvider {
    fn execute(&self, request: &Rv64imProofServerRequest) -> Result<Vec<u8>, String> {
        assert_eq!(request.path(), self.expected_path);
        assert_eq!(request.body(), self.expected_body.as_slice());
        Ok(self.response_body.clone())
    }
}

struct FailingProofServerProvider;

impl Rv64imProofServerProvider for FailingProofServerProvider {
    fn execute(&self, _request: &Rv64imProofServerRequest) -> Result<Vec<u8>, String> {
        Err("mock transport failure".into())
    }
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_public_field_encoding_round_trips() {
    let fixture = bridge_fixture();
    let public_inputs = Rv64imNightstreamBridgePublicInputs::new(&fixture.statement);
    let fields = encode_rv64im_nightstream_bridge_public_inputs_fields(public_inputs);
    let decoded = decode_rv64im_nightstream_bridge_public_inputs_fields(&fields).expect("decode public field encoding");
    assert_eq!(decoded, public_inputs);
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_private_witness_field_encoding_round_trips() {
    let fixture = bridge_fixture();
    let fields = encode_rv64im_nightstream_bridge_private_witness_fields(bridge_witness(fixture))
        .expect("encode private witness field encoding");
    let decoded = decode_rv64im_nightstream_bridge_private_witness_fields(&fields)
        .expect("decode private witness field encoding");
    let reencoded = encode_rv64im_nightstream_bridge_private_witness_fields(decoded.borrowed())
        .expect("re-encode private witness field encoding");
    assert_eq!(fields, reencoded);
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_encoded_payload_verifier_accepts_current_boundary() {
    let fixture = bridge_fixture();
    let public_inputs = encode_rv64im_nightstream_bridge_public_inputs_fields(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
    );
    let private_witness = encode_rv64im_nightstream_bridge_private_witness_fields(bridge_witness(fixture))
        .expect("encode private witness");

    verify_rv64im_nightstream_bridge_payload(&public_inputs, &private_witness)
        .expect("encoded payload verifier accepts current boundary");
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_encoded_payload_verifier_rejects_tampered_public_input() {
    let fixture = bridge_fixture();
    let mut public_inputs = encode_rv64im_nightstream_bridge_public_inputs_fields(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
    );
    let private_witness = encode_rv64im_nightstream_bridge_private_witness_fields(bridge_witness(fixture))
        .expect("encode private witness");

    let last = public_inputs
        .last_mut()
        .expect("public field encoding must not be empty");
    *last ^= 1;
    let err = verify_rv64im_nightstream_bridge_payload(&public_inputs, &private_witness)
        .expect_err("tampered public input must fail");
    assert!(format!("{err}").contains("statement digest mismatch"));
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_preimage_builder_matches_v1_shape() {
    let fixture = bridge_fixture();

    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");

    assert_eq!(preimage.public_transcript_inputs, Vec::<u64>::new());
    assert_eq!(preimage.public_transcript_outputs, Vec::<u64>::new());
    assert_eq!(preimage.communications_commitment, None);
    assert_eq!(
        preimage.binding_input,
        rv64im_nightstream_bridge_binding_input(Rv64imNightstreamBridgePublicInputs::new(&fixture.statement))
    );
    assert_eq!(preimage.key_location, RV64IM_NIGHTSTREAM_BRIDGE_KEY_LOCATION);
    verify_rv64im_nightstream_bridge_preimage(&preimage).expect("verify bridge preimage");
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_preimage_verifier_rejects_wrong_key_location() {
    let fixture = bridge_fixture();

    let mut preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");
    preimage.key_location = "nstream-midnight-bridge/rv64im/nightstream/v0".to_owned();

    let err = verify_rv64im_nightstream_bridge_preimage(&preimage).expect_err("wrong key location must fail");
    assert!(format!("{err}").contains(RV64IM_NIGHTSTREAM_BRIDGE_KEY_LOCATION));
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_preimage_builder_derives_binding_input_from_statement_digest() {
    let fixture = bridge_fixture();

    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");
    assert_eq!(
        preimage.binding_input,
        rv64im_nightstream_bridge_binding_input(Rv64imNightstreamBridgePublicInputs::new(&fixture.statement))
    );
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_preimage_verifier_rejects_tampered_binding_input() {
    let fixture = bridge_fixture();

    let mut preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");
    preimage.binding_input ^= 1;

    let err = verify_rv64im_nightstream_bridge_preimage(&preimage).expect_err("tampered binding input must fail");
    assert!(format!("{err}").contains("requires binding_input"));
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_proof_server_request_body_uses_derived_binding_input_without_override() {
    let fixture = bridge_fixture();
    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");
    let proving_data = ProvingKeyMaterial {
        prover_key: vec![1, 2, 3],
        verifier_key: vec![4, 5],
        ir_source: vec![6, 7, 8],
    };

    let request_policy = Rv64imProofServerProveRequestPolicy::embedded_proving_data(proving_data.clone());
    assert_eq!(request_policy.resolver_policy(), None);
    let body = build_rv64im_nightstream_proof_server_request_body(&preimage, request_policy)
        .expect("build proof-server request body");
    let (versioned_preimage, decoded_data, binding_input_override): (
        Rv64imProofServerPreimageVersioned,
        Option<ProvingKeyMaterial>,
        Option<Fr>,
    ) = tagged_deserialize(&body[..]).expect("decode proof-server request body");

    let Rv64imProofServerPreimageVersioned::V2(decoded_preimage) = versioned_preimage;
    assert_eq!(
        decoded_preimage.binding_input,
        Fr::from(rv64im_nightstream_bridge_binding_input(
            Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        ))
    );
    assert_eq!(
        decoded_preimage.key_location.0.as_ref(),
        RV64IM_NIGHTSTREAM_BRIDGE_KEY_LOCATION
    );
    let decoded_data = decoded_data.expect("request body must carry proving data");
    assert_eq!(decoded_data.prover_key, proving_data.prover_key);
    assert_eq!(decoded_data.verifier_key, proving_data.verifier_key);
    assert_eq!(decoded_data.ir_source, proving_data.ir_source);
    assert_eq!(binding_input_override, None);
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_check_request_body_uses_derived_binding_input_and_optional_wrapped_ir() {
    let fixture = bridge_fixture();
    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");
    let ir_source = vec![9, 8, 7, 6];

    let request_policy = Rv64imProofServerCheckRequestPolicy::embedded_ir(ir_source.clone());
    assert_eq!(request_policy.resolver_policy(), None);
    let body = build_rv64im_nightstream_check_request_body(&preimage, request_policy)
        .expect("build proof-server check request body");
    let (versioned_preimage, wrapped_ir): (Rv64imProofServerPreimageVersioned, Option<Rv64imProofServerWrappedIr>) =
        tagged_deserialize(&body[..]).expect("decode proof-server check request body");

    let Rv64imProofServerPreimageVersioned::V2(decoded_preimage) = versioned_preimage;
    assert_eq!(
        decoded_preimage.binding_input,
        Fr::from(rv64im_nightstream_bridge_binding_input(
            Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        ))
    );
    assert_eq!(
        decoded_preimage.key_location.0.as_ref(),
        RV64IM_NIGHTSTREAM_BRIDGE_KEY_LOCATION
    );
    let wrapped_ir = wrapped_ir.expect("check request body must carry wrapped ir");
    assert_eq!(wrapped_ir.0, ir_source);
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_proof_server_request_body_resolver_backed_omits_proving_data() {
    let fixture = bridge_fixture();
    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");

    let request_policy = Rv64imProofServerProveRequestPolicy::resolver_backed();
    assert_eq!(
        request_policy.resolver_policy(),
        Some(Rv64imProofServerResolverPolicy::BridgeKeyLocation)
    );
    let body = build_rv64im_nightstream_proof_server_request_body(&preimage, request_policy)
        .expect("build proof-server request body");
    let (_versioned_preimage, decoded_data, binding_input_override): (
        Rv64imProofServerPreimageVersioned,
        Option<ProvingKeyMaterial>,
        Option<Fr>,
    ) = tagged_deserialize(&body[..]).expect("decode proof-server request body");

    assert!(decoded_data.is_none());
    assert_eq!(binding_input_override, None);
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_check_request_body_resolver_backed_omits_wrapped_ir() {
    let fixture = bridge_fixture();
    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");

    let request_policy = Rv64imProofServerCheckRequestPolicy::resolver_backed();
    assert_eq!(
        request_policy.resolver_policy(),
        Some(Rv64imProofServerResolverPolicy::BridgeKeyLocation)
    );
    let body = build_rv64im_nightstream_check_request_body(&preimage, request_policy)
        .expect("build proof-server check request body");
    let (_versioned_preimage, wrapped_ir): (Rv64imProofServerPreimageVersioned, Option<Rv64imProofServerWrappedIr>) =
        tagged_deserialize(&body[..]).expect("decode proof-server check request body");

    assert_eq!(wrapped_ir, None);
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_prove_request_uses_bridge_owned_route_and_body() {
    let fixture = bridge_fixture();
    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");
    let proving_data = ProvingKeyMaterial {
        prover_key: vec![1, 2, 3],
        verifier_key: vec![4, 5],
        ir_source: vec![6, 7, 8],
    };
    let request_policy = Rv64imProofServerProveRequestPolicy::embedded_proving_data(proving_data);
    let expected_body = build_rv64im_nightstream_proof_server_request_body(&preimage, request_policy.clone())
        .expect("build proof-server request body");

    let request = build_rv64im_nightstream_prove_request(&preimage, request_policy).expect("build prove request");
    assert_eq!(request.path(), Rv64imProofServerRoute::Prove.path());
    assert_eq!(request.body(), expected_body.as_slice());
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_check_request_uses_bridge_owned_route_and_body() {
    let fixture = bridge_fixture();
    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");
    let request_policy = Rv64imProofServerCheckRequestPolicy::resolver_backed();
    let expected_body = build_rv64im_nightstream_check_request_body(&preimage, request_policy.clone())
        .expect("build proof-server check request body");

    let request = build_rv64im_nightstream_check_request(&preimage, request_policy).expect("build check request");
    assert_eq!(request.path(), Rv64imProofServerRoute::Check.path());
    assert_eq!(request.body(), expected_body.as_slice());
}

#[test]
fn rv64im_bridge_prove_response_parser_round_trips_versioned_proof() {
    let proof = Rv64imProofServerProofVersioned::V2(Proof(vec![7, 8, 9]));
    let mut body = Vec::new();
    tagged_serialize(&proof, &mut body).expect("serialize prove response body");

    let parsed = parse_rv64im_nightstream_prove_response_body(&body).expect("parse prove response body");
    assert_eq!(parsed, proof);
}

#[test]
fn rv64im_bridge_check_response_parser_round_trips_skip_vector() {
    let skipped_output_blocks = vec![Some(0u64), None, Some(3u64)];
    let mut body = Vec::new();
    tagged_serialize(&skipped_output_blocks, &mut body).expect("serialize check response body");

    let parsed = parse_rv64im_nightstream_check_response_body(&body).expect("parse check response body");
    assert_eq!(parsed, Rv64imProofServerCheckResponse::new(skipped_output_blocks));
}

#[test]
fn rv64im_bridge_route_parser_rejects_mismatched_response_shape() {
    let proof = Rv64imProofServerProofVersioned::V2(Proof(vec![1, 2, 3]));
    let mut body = Vec::new();
    tagged_serialize(&proof, &mut body).expect("serialize prove response body");

    let err = Rv64imProofServerRoute::Check
        .parse_response_body(&body)
        .expect_err("check route must reject prove response body");
    assert!(format!("{err}").contains("response decode"));
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_request_parser_uses_route_for_response_decoding() {
    let fixture = bridge_fixture();
    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");
    let request =
        build_rv64im_nightstream_prove_request(&preimage, Rv64imProofServerProveRequestPolicy::resolver_backed())
            .expect("build prove request");
    let response = Rv64imProofServerProofVersioned::V2(Proof(vec![4, 5, 6]));
    let mut body = Vec::new();
    tagged_serialize(&response, &mut body).expect("serialize prove response body");

    let parsed = request
        .parse_response_body(&body)
        .expect("parse route-aware response");
    assert_eq!(parsed, Rv64imProofServerResponse::Prove(response));
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_provider_executes_prove_request_and_parses_response() {
    let fixture = bridge_fixture();
    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");
    let request_policy = Rv64imProofServerProveRequestPolicy::resolver_backed();
    let expected_request =
        build_rv64im_nightstream_prove_request(&preimage, request_policy.clone()).expect("build prove request");
    let response = Rv64imProofServerProofVersioned::V2(Proof(vec![9, 9, 9]));
    let mut response_body = Vec::new();
    tagged_serialize(&response, &mut response_body).expect("serialize prove response body");
    let provider = MockProofServerProvider {
        expected_path: Rv64imProofServerRoute::Prove.path(),
        expected_body: expected_request.into_body(),
        response_body,
    };

    let parsed =
        execute_rv64im_nightstream_prove_request(&provider, &preimage, request_policy).expect("execute prove request");
    assert_eq!(parsed, response);
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_provider_executes_check_request_and_parses_response() {
    let fixture = bridge_fixture();
    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");
    let request_policy = Rv64imProofServerCheckRequestPolicy::embedded_ir(vec![1, 2, 3]);
    let expected_request =
        build_rv64im_nightstream_check_request(&preimage, request_policy.clone()).expect("build check request");
    let response = Rv64imProofServerCheckResponse::new(vec![Some(1), None, Some(4)]);
    let mut response_body = Vec::new();
    tagged_serialize(&response.clone().into_skipped_output_blocks(), &mut response_body)
        .expect("serialize check response body");
    let provider = MockProofServerProvider {
        expected_path: Rv64imProofServerRoute::Check.path(),
        expected_body: expected_request.into_body(),
        response_body,
    };

    let parsed =
        execute_rv64im_nightstream_check_request(&provider, &preimage, request_policy).expect("execute check request");
    assert_eq!(parsed, response);
}

#[test]
#[ignore = "builds the full RV64IM/Nightstream proof fixture; run explicitly when bridge boundary encoding changes"]
fn rv64im_bridge_provider_maps_transport_failures() {
    let fixture = bridge_fixture();
    let preimage = build_rv64im_nightstream_bridge_preimage(
        Rv64imNightstreamBridgePublicInputs::new(&fixture.statement),
        bridge_witness(fixture),
    )
    .expect("build bridge preimage");
    let request =
        build_rv64im_nightstream_prove_request(&preimage, Rv64imProofServerProveRequestPolicy::resolver_backed())
            .expect("build prove request");

    let err =
        execute_rv64im_proof_server_request(&FailingProofServerProvider, &request).expect_err("transport must fail");
    assert!(matches!(err, Rv64imBridgeError::Transport(_)));
}
