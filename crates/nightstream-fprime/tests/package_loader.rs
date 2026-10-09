use std::{fs, path::PathBuf};

use nightstream_fprime::{
    derive_pi_ccs_v1_2_transcript, load_per_application_package, load_poseidon2_hash_chain_v1_package, CcsMatrixSource,
    PackageError, PiCcsV1_2OutputEvaluations, PiCcsV1_2PackageInputs, PiDecV1_2PackageInputs,
    PI_CCS_V1_2_COEFFICIENT_COUNT, PI_CCS_V1_2_FRESH_COMMITMENT_WORDS, PI_CCS_V1_2_MATRIX_COUNT,
    PI_CCS_V1_2_PRIOR_CHILDREN_WORDS, PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT,
    PI_CCS_V1_2_ROUND_COUNT, PI_CCS_V1_2_SOURCE_COUNT, PI_CCS_V1_2_STATE_PREIMAGE_WORDS, PI_DEC_V1_2_CHILD_COUNT,
    PI_DEC_V1_2_COMMITMENT_WORDS_PER_CHILD, PI_DEC_V1_2_EVAL_A_MATRICES_PER_CHILD, PI_DEC_V1_2_EVAL_K_VALUES_PER_CHILD,
    PI_DEC_V1_2_PUBLIC_INPUT_WORDS_PER_CHILD,
};
use p3_field::PrimeField64;
use serde_json::{json, Value};

const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;

fn sealed_artifact_bytes() -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .expect("run formal/nightstream-fprime/scripts/validate.sh emit first")
}

fn pi_ccs_parity_bytes() -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-piccs-parity-v1.json"),
    )
    .expect("run formal/nightstream-fprime/scripts/validate.sh pi-ccs-parity first")
}

fn canonical_bytes(value: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).expect("canonical package JSON");
    bytes.push(b'\n');
    bytes
}

#[test]
fn sealed_package_builds_the_package_owned_logical_relation_header() {
    let package =
        load_poseidon2_hash_chain_v1_package(&sealed_artifact_bytes()).expect("verifier-owned production package");
    let relation = package.ccs_relation();
    let header = package
        .ccs_structure_header()
        .expect("Lean-owned logical CCS header");

    assert_eq!(package.physical_row_count(), 11_573_009);
    assert_eq!(package.total_column_count(), 11_659_967);
    assert_eq!(package.private_input_count(), 102_802);
    assert_eq!(package.public_input_count(), 278);
    assert_eq!(relation.row_count(), 1_032_323);
    // Poseidon2HashChainV1Package.logicalWidth, after shared-value wiring.
    assert_eq!(relation.column_count(), 45_140_532);
    assert_eq!(relation.cube_variables(), PI_CCS_V1_2_ROUND_COUNT);
    assert_eq!(
        relation.matrix_sources(),
        [
            CcsMatrixSource::A,
            CcsMatrixSource::B,
            CcsMatrixSource::C,
            CcsMatrixSource::SboxInput,
        ]
    );
    assert_eq!(relation.degree_bound(), 8);
    assert_eq!(relation.terms().len(), 3);
    assert_eq!(
        relation
            .terms()
            .iter()
            .map(|term| term.exponents().iter().sum::<usize>())
            .max(),
        Some(7)
    );
    assert_eq!(
        package.logical_public_input_count(),
        PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS
    );
    assert!(header.is_verifier_artifact_header());
    assert_eq!(header.n, relation.row_count());
    assert_eq!(header.m, relation.column_count());
    // The rows and the carrier fit 2^27; the declared cube keeps 28 rounds.
    assert!(
        relation
            .row_count()
            .max(relation.column_count().div_ceil(54) * 54)
            <= 1 << 27
    );
    assert_eq!(header.domain_rows(), 1 << PI_CCS_V1_2_ROUND_COUNT);
    assert_eq!(header.t(), relation.matrix_sources().len());
    assert_eq!(header.max_degree() as usize + 1, relation.degree_bound());
    assert_eq!(header.f.terms().len(), relation.terms().len());
    for (actual, expected) in header.f.terms().iter().zip(relation.terms()) {
        assert_eq!(actual.coeff.as_canonical_u64(), expected.coefficient());
        assert_eq!(
            actual
                .exps
                .iter()
                .copied()
                .map(|value| value as usize)
                .collect::<Vec<_>>(),
            expected.exponents(),
        );
    }
}

#[test]
fn sealed_stage1_encoder_appends_the_exact_application_witness() {
    let package =
        load_poseidon2_hash_chain_v1_package(&sealed_artifact_bytes()).expect("verifier-owned production package");
    let binding = package
        .production_verifier_binding()
        .expect("fixed production verifier binding");
    let output_evaluations = PiCcsV1_2OutputEvaluations::new(
        vec![vec![[0, 0]; PI_CCS_V1_2_COEFFICIENT_COUNT]; PI_CCS_V1_2_SOURCE_COUNT],
        vec![vec![vec![[0, 0]; PI_CCS_V1_2_COEFFICIENT_COUNT]; PI_CCS_V1_2_MATRIX_COUNT]; PI_CCS_V1_2_SOURCE_COUNT],
    )
    .expect("zero output evaluations");
    let mut prior_public_input = vec![0; PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS];
    prior_public_input[0] = 1;
    let pi_ccs = PiCcsV1_2PackageInputs::new(
        vec![0; PI_CCS_V1_2_STATE_PREIMAGE_WORDS],
        vec![0; PI_CCS_V1_2_STATE_PREIMAGE_WORDS],
        vec![0; PI_CCS_V1_2_PRIOR_CHILDREN_WORDS],
        vec![0; PI_CCS_V1_2_FRESH_COMMITMENT_WORDS],
        vec![vec![[0, 0]; PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT]; PI_CCS_V1_2_ROUND_COUNT],
        output_evaluations,
        prior_public_input,
        [0; 4],
        binding.verifier_context().clone(),
    )
    .expect("zero PiCCS inputs");
    let pi_dec = PiDecV1_2PackageInputs::new(
        vec![vec![0; PI_DEC_V1_2_COMMITMENT_WORDS_PER_CHILD]; PI_DEC_V1_2_CHILD_COUNT],
        vec![vec![[0, 0]; PI_DEC_V1_2_EVAL_K_VALUES_PER_CHILD]; PI_DEC_V1_2_CHILD_COUNT],
        vec![
            vec![vec![[0, 0]; PI_CCS_V1_2_COEFFICIENT_COUNT]; PI_DEC_V1_2_EVAL_A_MATRICES_PER_CHILD];
            PI_DEC_V1_2_CHILD_COUNT
        ],
        vec![vec![0; PI_DEC_V1_2_PUBLIC_INPUT_WORDS_PER_CHILD]; PI_DEC_V1_2_CHILD_COUNT],
    )
    .expect("zero PiDEC inputs");
    let message = [11, 12, 13, 14];
    let encoded_pi_ccs = package
        .encode_pi_ccs_v1_2_inputs(&pi_ccs)
        .expect("typed PiCCS inputs");
    let encoded = package
        .encode_stage1_v1_2_inputs(&pi_ccs, &pi_dec, &message)
        .expect("complete typed Stage 1 inputs");

    assert_eq!(encoded.private_values().len(), package.private_input_count());
    assert!(encoded.private_values().ends_with(&message));
    assert_eq!(encoded.public_values().len(), package.public_input_count());
    assert_eq!(encoded.public_values(), encoded_pi_ccs.public_values());
    let verifier_context_start = PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS + 4;
    assert_eq!(
        &encoded.public_values()[verifier_context_start..],
        binding.verifier_context().digest().as_slice(),
    );
    assert!(package
        .encode_stage1_v1_2_inputs(&pi_ccs, &pi_dec, &message[..3])
        .is_err());
    assert!(package
        .encode_stage1_v1_2_inputs(&pi_ccs, &pi_dec, &[GOLDILOCKS_MODULUS, 0, 0, 0])
        .is_err());
}

#[test]
fn lean_emitted_v1_2_pi_ccs_output_keeps_eval_k_and_eval_a_separate() {
    let package =
        load_poseidon2_hash_chain_v1_package(&sealed_artifact_bytes()).expect("verifier-owned production package");
    let binding = package
        .production_verifier_binding()
        .expect("fixed production verifier binding");
    let mut expected_eval_k = Vec::with_capacity(PI_CCS_V1_2_SOURCE_COUNT);
    let mut expected_eval_a = Vec::with_capacity(PI_CCS_V1_2_SOURCE_COUNT);

    for source in 0..PI_CCS_V1_2_SOURCE_COUNT {
        let mut source_eval_k = Vec::with_capacity(PI_CCS_V1_2_COEFFICIENT_COUNT);
        for coefficient in 0..PI_CCS_V1_2_COEFFICIENT_COUNT {
            let value = [
                1_000_000 + (source * 1_000 + coefficient * 2) as u64,
                1_000_001 + (source * 1_000 + coefficient * 2) as u64,
            ];
            source_eval_k.push(value);
        }
        expected_eval_k.push(source_eval_k);

        let mut source_eval_a = Vec::with_capacity(PI_CCS_V1_2_MATRIX_COUNT);
        for matrix in 0..PI_CCS_V1_2_MATRIX_COUNT {
            let mut matrix_values = Vec::with_capacity(PI_CCS_V1_2_COEFFICIENT_COUNT);
            for coefficient in 0..PI_CCS_V1_2_COEFFICIENT_COUNT {
                let ordinal = source * PI_CCS_V1_2_MATRIX_COUNT * PI_CCS_V1_2_COEFFICIENT_COUNT
                    + matrix * PI_CCS_V1_2_COEFFICIENT_COUNT
                    + coefficient;
                let value = [2_000_000 + (ordinal * 2) as u64, 2_000_001 + (ordinal * 2) as u64];
                matrix_values.push(value);
            }
            source_eval_a.push(matrix_values);
        }
        expected_eval_a.push(source_eval_a);
    }

    let output = PiCcsV1_2OutputEvaluations::new(expected_eval_k.clone(), expected_eval_a.clone())
        .expect("separate v1_2 output families");
    let mut prior_public_input = vec![0; PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS];
    prior_public_input[0] = 1;
    let inputs = PiCcsV1_2PackageInputs::new(
        vec![0; PI_CCS_V1_2_STATE_PREIMAGE_WORDS],
        vec![0; PI_CCS_V1_2_STATE_PREIMAGE_WORDS],
        vec![0; PI_CCS_V1_2_PRIOR_CHILDREN_WORDS],
        vec![0; PI_CCS_V1_2_FRESH_COMMITMENT_WORDS],
        vec![vec![[0, 0]; PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT]; PI_CCS_V1_2_ROUND_COUNT],
        output,
        prior_public_input,
        [0; 4],
        binding.verifier_context().clone(),
    )
    .expect("fixed v1_2 package inputs");
    let encoded = package
        .encode_pi_ccs_v1_2_inputs(&inputs)
        .expect("package-owned v1_2 input encoding");
    let decoded = package
        .pi_ccs_v1_2_output_evaluations(encoded.private_values())
        .expect("exact v1_2 PiCCS output layout");
    assert_eq!(decoded.eval_k(), expected_eval_k);
    assert_eq!(decoded.eval_a(), expected_eval_a);
}

#[test]
fn rust_v1_2_pi_ccs_transcript_matches_lean_emitted_vector() {
    let parity: Value = serde_json::from_slice(&pi_ccs_parity_bytes()).expect("Lean PiCCS parity vector");
    let parity = parity.as_array().expect("PiCCS parity tuple");
    assert_eq!(parity[0].as_u64(), Some(8));
    let input = parity[1].as_array().expect("PiCCS input tuple");
    let result = parity[2].as_array().expect("PiCCS result tuple");
    let state_preimage: Vec<u64> = serde_json::from_value(input[0].clone()).expect("prior preimage");
    let output_preimage: Vec<u64> = serde_json::from_value(input[1].clone()).expect("output preimage");
    let state_public_input: Vec<u64> = serde_json::from_value(input[2].clone()).expect("prior public input");
    let state_digest: [u64; 4] = serde_json::from_value(input[3].clone()).expect("output digest");
    let verifier_context_digest: [u64; 4] = serde_json::from_value(input[4].clone()).expect("verifier-context digest");
    let fresh_commitment: Vec<u64> = serde_json::from_value(input[5].clone()).expect("fresh commitment");
    let rounds: Vec<Vec<[u64; 2]>> = serde_json::from_value(input[6].clone()).expect("round messages");
    let expected_eval_k: Vec<Vec<[u64; 2]>> = serde_json::from_value(input[7].clone()).expect("output Eval_K");
    let expected_eval_a: Vec<Vec<Vec<[u64; 2]>>> = serde_json::from_value(input[8].clone()).expect("output Eval_A");
    let public: Vec<Vec<u64>> = serde_json::from_value(input[9].clone()).expect("public statement blocks");
    let verifier: Vec<Vec<u64>> = serde_json::from_value(input[10].clone()).expect("verifier input blocks");
    let alpha: Vec<[u64; 2]> = serde_json::from_value(result[1].clone()).expect("alpha");
    let gamma: [u64; 2] = serde_json::from_value(result[2].clone()).expect("gamma");
    let round_point: Vec<[u64; 2]> = serde_json::from_value(result[6].clone()).expect("round point");
    let outgoing_state: [u64; 16] = serde_json::from_value(result[14].clone()).expect("outgoing state");

    assert_eq!(state_preimage.len(), PI_CCS_V1_2_STATE_PREIMAGE_WORDS);
    assert_eq!(output_preimage.len(), PI_CCS_V1_2_STATE_PREIMAGE_WORDS);
    assert_eq!(state_public_input.len(), PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS);
    assert_eq!(public[0], state_digest, "prior digest statement block");

    let mut output = Vec::with_capacity(PI_CCS_V1_2_SOURCE_COUNT * 540);
    for source in 0..PI_CCS_V1_2_SOURCE_COUNT {
        for value in &expected_eval_k[source] {
            output.extend(value);
        }
        for matrix in &expected_eval_a[source] {
            for value in matrix {
                output.extend(value);
            }
        }
    }

    let actual =
        derive_pi_ccs_v1_2_transcript(&public, &verifier, &rounds, &output).expect("exact v1_2 transcript replay");
    assert_eq!(actual.alpha(), alpha);
    assert_eq!(actual.gamma(), gamma);
    assert_eq!(actual.round_point(), round_point);
    assert_eq!(actual.outgoing_state(), outgoing_state);

    let package =
        load_poseidon2_hash_chain_v1_package(&sealed_artifact_bytes()).expect("verifier-owned production package");
    let binding = package
        .production_verifier_binding()
        .expect("fixed production binding");
    assert_eq!(binding.verifier_context().digest(), verifier_context_digest);
    let derived_context = binding.verifier_context().clone();
    let output_evaluations = PiCcsV1_2OutputEvaluations::new(expected_eval_k.clone(), expected_eval_a.clone())
        .expect("nonzero output evaluations");
    let inputs = PiCcsV1_2PackageInputs::new(
        state_preimage,
        output_preimage,
        vec![0; PI_CCS_V1_2_PRIOR_CHILDREN_WORDS],
        fresh_commitment,
        rounds,
        output_evaluations,
        state_public_input,
        state_digest,
        derived_context,
    )
    .expect("nonzero package inputs");
    let encoded = package
        .encode_pi_ccs_v1_2_inputs(&inputs)
        .expect("package-owned v1_2 input encoding");
    let decoded = package
        .pi_ccs_v1_2_output_evaluations(encoded.private_values())
        .expect("accepted nonzero-proof output layout");
    assert_eq!(decoded.eval_k(), expected_eval_k);
    assert_eq!(decoded.eval_a(), expected_eval_a);
}

#[test]
fn v1_2_input_encoder_rejects_a_missing_eval_a_matrix() {
    let eval_k = vec![vec![[0, 0]; PI_CCS_V1_2_COEFFICIENT_COUNT]; PI_CCS_V1_2_SOURCE_COUNT];
    let eval_a =
        vec![vec![vec![[0, 0]; PI_CCS_V1_2_COEFFICIENT_COUNT]; PI_CCS_V1_2_MATRIX_COUNT - 1]; PI_CCS_V1_2_SOURCE_COUNT];
    assert!(PiCcsV1_2OutputEvaluations::new(eval_k, eval_a).is_err());
}

#[test]
fn loader_rejects_a_different_verifier_owned_identity() {
    assert!(matches!(
        load_per_application_package(&sealed_artifact_bytes(), [0; 4]),
        Err(PackageError::ExpectedIdentityMismatch { .. })
    ));
}

#[test]
fn loader_binds_the_lean_owned_ccs_polynomial_to_the_relation_identity() {
    let mut value: Value = serde_json::from_slice(&sealed_artifact_bytes()).expect("sealed production package");
    value.as_array_mut().expect("sealed package")[1]
        .as_array_mut()
        .expect("inner package")[4]
        .as_array_mut()
        .expect("CCS relation array")[5]
        .as_array_mut()
        .expect("CCS polynomial terms")[0]
        .as_array_mut()
        .expect("CCS polynomial term")[0] = json!(2);

    assert!(matches!(
        load_poseidon2_hash_chain_v1_package(&canonical_bytes(&value)),
        Err(PackageError::ExpectedIdentityMismatch { .. })
    ));
}

#[test]
fn loader_rejects_noncanonical_json_bytes() {
    let mut bytes = sealed_artifact_bytes();
    bytes.insert(0, b' ');

    assert!(matches!(
        load_poseidon2_hash_chain_v1_package(&bytes),
        Err(PackageError::NonCanonicalBytes)
    ));
}

#[test]
fn loader_rejects_a_missing_sampler_coefficient_witness_batch() {
    let bytes = sealed_artifact_bytes();
    load_poseidon2_hash_chain_v1_package(&bytes).expect("current complete package");
    let mut value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value[4][0], json!(3));
    let challenge_block = value[4][2][5].as_u64().unwrap() as usize;
    let first_word = value[4][1][challenge_block][4][0][0].as_u64().unwrap();
    let batches = value[1][10].as_array_mut().unwrap();
    let index = batches
        .iter()
        .position(|batch| batch[0].as_u64() == Some(first_word))
        .expect("sampler coefficient words have their own witness batch");
    assert_eq!(batches[index][1].as_array().unwrap().len(), 54);
    batches.remove(index);
    let changed = canonical_bytes(&value);
    // A self-consistent identity must not authorize a missing witness writer.
    // Replay the computed identity so the second load reaches structural checks.
    let identity = match load_per_application_package(&changed, [0; 4]) {
        Err(PackageError::ExpectedIdentityMismatch { computed, .. }) => computed,
        _ => panic!("the mutation must change the structural identity"),
    };
    let error = load_per_application_package(&changed, identity)
        .err()
        .expect("a rehashed incomplete witness program must be rejected");
    assert!(
        matches!(error, PackageError::Invalid("witness column coverage")),
        "{error:?}"
    );
}

#[test]
fn loader_rejects_a_rehashed_matrix_slot_permutation() {
    let mut value: Value = serde_json::from_slice(&sealed_artifact_bytes()).expect("sealed production package");
    let sources = &mut value[1][4][3];
    assert_eq!(*sources, json!([0, 1, 2, 3]));
    *sources = json!([1, 0, 2, 3]);
    let changed = canonical_bytes(&value);
    let identity = match load_per_application_package(&changed, [0; 4]) {
        Err(PackageError::ExpectedIdentityMismatch { computed, .. }) => computed,
        _ => panic!("the mutation must change the structural identity"),
    };
    let error = load_per_application_package(&changed, identity)
        .err()
        .expect("a rehashed matrix permutation must be rejected");
    assert!(
        matches!(error, PackageError::Invalid("CCS relation matrix order")),
        "{error:?}"
    );
}

#[test]
fn lean_test_error_counts_match_the_sealed_relation() {
    // VerifierErrorBudget exports the selected test error for these counts.
    // They must come from the relation the verifier loads, not from the artifact.
    let bytes = fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-piccs-test-error-v1.json"),
    )
    .expect("Lean PiCCS test-error artifact");
    let [schema, cube_variables, width, _, _, _, matrices, _]: [u64; 8] =
        serde_json::from_slice(&bytes).expect("schema-1 Lean PiCCS test-error artifact");
    let package =
        load_poseidon2_hash_chain_v1_package(&sealed_artifact_bytes()).expect("verifier-owned production package");
    let relation = package.ccs_relation();

    assert_eq!(schema, 1);
    assert_eq!(cube_variables as usize, relation.cube_variables());
    assert_eq!(width as usize, relation.degree_bound());
    assert_eq!(matrices as usize, relation.matrix_sources().len());
}
