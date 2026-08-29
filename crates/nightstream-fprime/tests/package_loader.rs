use std::{fs, path::PathBuf};

use nightstream_fprime::{
    derive_pi_ccs_v1_1_transcript, load, CcsMatrixSource, LoadedPackage, PackageError, PiCcsV1_1OutputEvaluations,
    PiCcsV1_1PackageInputs, PI_CCS_V1_1_COEFFICIENT_COUNT, PI_CCS_V1_1_FRESH_COMMITMENT_WORDS,
    PI_CCS_V1_1_MATRIX_COUNT, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT,
    PI_CCS_V1_1_ROUND_COUNT, PI_CCS_V1_1_SOURCE_COUNT, PI_CCS_V1_1_STATE_PREIMAGE_WORDS,
};
use serde_json::{json, Value};

// Lean-emitted Pilot + PiCCS + PiRLC + PiDEC + running-transition package
// identity. Phase conformance remains open until every required gate passes
// on these bytes.
const EXPECTED_IDENTITY: [u64; 4] = [
    3_355_019_049_079_043_662,
    4_920_201_927_044_277_974,
    5_339_237_732_450_517_664,
    894_111_819_037_169_888,
];
const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;

fn artifact_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-v1.json")
}

fn artifact_bytes() -> Vec<u8> {
    fs::read(artifact_path()).expect("run formal/nightstream-fprime/scripts/validate.sh emit first")
}

fn artifact_value() -> Value {
    serde_json::from_slice(&artifact_bytes()).expect("Lean-emitted package JSON")
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

fn package_array(value: &mut Value) -> &mut Vec<Value> {
    let plan = value.as_array_mut().expect("package plan array");
    assert_eq!(plan[0], json!(8), "package plan schema");
    plan[1].as_array_mut().expect("embedded package array")
}

fn assert_canonical_package_matrix(matrix: &nightstream_fprime::PackageSparseMatrix, rows: usize, columns: usize) {
    assert_eq!(matrix.rows(), rows);
    assert_eq!(matrix.columns(), columns);
    assert_eq!(matrix.values().len(), matrix.column_indices().len());
    assert_eq!(matrix.row_offsets().len(), rows + 1);
    assert_eq!(matrix.row_offsets().first().copied(), Some(0));
    assert_eq!(matrix.row_offsets().last().copied(), Some(matrix.nonzero_count()));
    assert!(matrix
        .row_offsets()
        .windows(2)
        .all(|pair| pair[0] <= pair[1]));
    assert!(matrix
        .column_indices()
        .iter()
        .all(|column| *column < columns));
    assert!(matrix
        .values()
        .iter()
        .all(|value| *value != 0 && *value < GOLDILOCKS_MODULUS));
}

fn package_inputs(
    package: &LoadedPackage,
    output_evaluations: PiCcsV1_1OutputEvaluations,
    digest: [u64; 4],
    commitment_key_words: &[u64],
) -> PiCcsV1_1PackageInputs {
    let mut prior_public_input = vec![0; PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS];
    prior_public_input[0] = 1;
    for (word, value) in digest.iter().copied().enumerate() {
        for bit in 0..64 {
            prior_public_input[1 + word * 64 + bit] = (value >> bit) & 1;
        }
    }
    let verifier_context = package
        .derive_pi_ccs_v1_1_verifier_context(commitment_key_words)
        .expect("fixed verifier context");
    PiCcsV1_1PackageInputs::new(
        vec![0; PI_CCS_V1_1_STATE_PREIMAGE_WORDS],
        vec![0; PI_CCS_V1_1_STATE_PREIMAGE_WORDS],
        vec![0; PI_CCS_V1_1_FRESH_COMMITMENT_WORDS],
        vec![vec![[0, 0]; PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT]; PI_CCS_V1_1_ROUND_COUNT],
        output_evaluations,
        prior_public_input,
        digest,
        verifier_context,
    )
    .expect("fixed v1_1 package inputs")
}

#[test]
fn lean_emitted_stage1_package_loads_with_verifier_owned_identity() {
    let package = load(&artifact_bytes(), EXPECTED_IDENTITY).expect("strict package load");

    assert_eq!(package.relation_identifier(), EXPECTED_IDENTITY);
    assert_eq!(package.row_count(), 27_584_180);
    assert_eq!(package.private_column_count(), 27_695_694);
    assert_eq!(package.private_input_count(), 166_738);
    assert_eq!(package.public_column_count(), 278);
    assert_eq!(package.total_column_count(), 27_695_973);
    assert_eq!(package.template_row_count(), 592);
    assert_eq!(package.permutation_invocation_count(), 7_703);
    assert_eq!(package.compact_template_count(), 326);
    assert_eq!(package.compact_invocation_count(), 167_246);
    assert_eq!(
        package.witness_instruction_count() + package.assertion_row_count(),
        1_380_435
    );
}

#[test]
fn verifier_context_derivation_binds_the_raw_commitment_setup() {
    let package = load(&artifact_bytes(), EXPECTED_IDENTITY).expect("strict package load");
    let original = package
        .derive_pi_ccs_v1_1_verifier_context(&[1, 2, 3])
        .expect("canonical verifier context");
    let changed = package
        .derive_pi_ccs_v1_1_verifier_context(&[1, 2, 4])
        .expect("changed verifier context");

    assert_eq!(original.relation_words(), changed.relation_words());
    assert_eq!(original.application_words(), changed.application_words());
    assert_ne!(original.nifs_key_words(), changed.nifs_key_words());
    assert_ne!(original.commitment_key_words(), changed.commitment_key_words());
    assert_ne!(original.digest(), changed.digest());
    assert!(package
        .derive_pi_ccs_v1_1_verifier_context(&[GOLDILOCKS_MODULUS])
        .is_err());
}

#[test]
fn lean_emitted_package_exports_canonical_r1cs_matrices() {
    let package = load(&artifact_bytes(), EXPECTED_IDENTITY).expect("strict package load");
    let relation = package
        .r1cs_matrices()
        .expect("package-owned R1CS expansion");
    let rows = 1usize << package.ccs_relation().cube_variables();
    let columns = rows + 1 + package.public_column_count();
    for matrix in [relation.a(), relation.b(), relation.c()] {
        assert_canonical_package_matrix(matrix, rows, columns);
        assert!(matrix.nonzero_count() > 0);
    }
}

#[test]
fn lean_emitted_package_exports_exact_v1_1_ccs_relation() {
    let package = load(&artifact_bytes(), EXPECTED_IDENTITY).expect("strict package load");
    let relation = package.ccs_relation();

    assert_eq!(relation.row_count(), package.row_count());
    assert_eq!(relation.column_count(), package.total_column_count());
    assert_eq!(relation.cube_variables(), PI_CCS_V1_1_ROUND_COUNT);
    assert_eq!(
        relation.matrix_sources(),
        [
            CcsMatrixSource::Bit,
            CcsMatrixSource::GeneralSelector,
            CcsMatrixSource::A,
            CcsMatrixSource::B,
            CcsMatrixSource::C,
            CcsMatrixSource::SboxInput,
            CcsMatrixSource::CenteredUnit,
            CcsMatrixSource::EvalSelector,
            CcsMatrixSource::Class0,
            CcsMatrixSource::Class1,
            CcsMatrixSource::Class2,
            CcsMatrixSource::Class3,
            CcsMatrixSource::Class4,
            CcsMatrixSource::Zero,
        ]
    );
    assert_eq!(relation.degree_bound(), 9);
    assert_eq!(relation.terms().len(), 74);
    assert_eq!(
        relation
            .terms()
            .iter()
            .map(|term| term.exponents().iter().sum::<usize>())
            .max(),
        Some(8)
    );
}

#[test]
fn lean_emitted_v1_1_pi_ccs_output_keeps_eval_k_and_eval_a_separate() {
    let package = load(&artifact_bytes(), EXPECTED_IDENTITY).expect("strict package load");
    let mut expected_eval_k = Vec::with_capacity(PI_CCS_V1_1_SOURCE_COUNT);
    let mut expected_eval_a = Vec::with_capacity(PI_CCS_V1_1_SOURCE_COUNT);

    for source in 0..PI_CCS_V1_1_SOURCE_COUNT {
        let mut source_eval_k = Vec::with_capacity(PI_CCS_V1_1_COEFFICIENT_COUNT);
        for coefficient in 0..PI_CCS_V1_1_COEFFICIENT_COUNT {
            let value = [
                1_000_000 + (source * 1_000 + coefficient * 2) as u64,
                1_000_001 + (source * 1_000 + coefficient * 2) as u64,
            ];
            source_eval_k.push(value);
        }
        expected_eval_k.push(source_eval_k);

        let mut source_eval_a = Vec::with_capacity(PI_CCS_V1_1_MATRIX_COUNT);
        for matrix in 0..PI_CCS_V1_1_MATRIX_COUNT {
            let mut matrix_values = Vec::with_capacity(PI_CCS_V1_1_COEFFICIENT_COUNT);
            for coefficient in 0..PI_CCS_V1_1_COEFFICIENT_COUNT {
                let ordinal = source * PI_CCS_V1_1_MATRIX_COUNT * PI_CCS_V1_1_COEFFICIENT_COUNT
                    + matrix * PI_CCS_V1_1_COEFFICIENT_COUNT
                    + coefficient;
                let value = [2_000_000 + (ordinal * 2) as u64, 2_000_001 + (ordinal * 2) as u64];
                matrix_values.push(value);
            }
            source_eval_a.push(matrix_values);
        }
        expected_eval_a.push(source_eval_a);
    }

    let output = PiCcsV1_1OutputEvaluations::new(expected_eval_k.clone(), expected_eval_a.clone())
        .expect("separate v1_1 output families");
    let inputs = package_inputs(&package, output, [0; 4], &[1]);
    let encoded = package
        .encode_pi_ccs_v1_1_inputs(&inputs)
        .expect("package-owned v1_1 input encoding");
    let decoded = package
        .pi_ccs_v1_1_output_evaluations(encoded.private_values())
        .expect("exact v1_1 PiCCS output layout");
    assert_eq!(decoded.eval_k(), expected_eval_k);
    assert_eq!(decoded.eval_a(), expected_eval_a);
}

#[test]
fn rust_v1_1_pi_ccs_transcript_matches_lean_emitted_vector() {
    let parity: Value = serde_json::from_slice(&pi_ccs_parity_bytes()).expect("Lean PiCCS parity vector");
    let parity = parity.as_array().expect("PiCCS parity tuple");
    assert_eq!(parity[0].as_u64(), Some(7));
    let input = parity[1].as_array().expect("PiCCS input tuple");
    let result = parity[2].as_array().expect("PiCCS result tuple");
    let state_preimage: Vec<u64> = serde_json::from_value(input[0].clone()).expect("prior preimage");
    let output_preimage: Vec<u64> = serde_json::from_value(input[1].clone()).expect("output preimage");
    let state_public_input: Vec<u64> = serde_json::from_value(input[2].clone()).expect("prior public input");
    let state_digest: [u64; 4] = serde_json::from_value(input[3].clone()).expect("output digest");
    let verifier_context: [u64; 4] = serde_json::from_value(input[4].clone()).expect("verifier context");
    let fresh_commitment: Vec<u64> = serde_json::from_value(input[5].clone()).expect("fresh commitment");
    let rounds: Vec<Vec<[u64; 2]>> = serde_json::from_value(input[6].clone()).expect("round messages");
    let expected_eval_k: Vec<Vec<[u64; 2]>> = serde_json::from_value(input[7].clone()).expect("output Eval_K");
    let expected_eval_a: Vec<Vec<Vec<[u64; 2]>>> = serde_json::from_value(input[8].clone()).expect("output Eval_A");
    let public: Vec<Vec<u64>> = serde_json::from_value(input[9].clone()).expect("public statement blocks");
    let verifier: Vec<Vec<u64>> = serde_json::from_value(input[10].clone()).expect("verifier input blocks");
    let authority: Vec<Vec<u64>> = serde_json::from_value(input[11].clone()).expect("verifier context authority");
    let alpha: Vec<[u64; 2]> = serde_json::from_value(result[1].clone()).expect("alpha");
    let gamma: [u64; 2] = serde_json::from_value(result[2].clone()).expect("gamma");
    let round_point: Vec<[u64; 2]> = serde_json::from_value(result[6].clone()).expect("round point");
    let outgoing_state: [u64; 8] = serde_json::from_value(result[14].clone()).expect("outgoing state");

    assert_eq!(state_preimage.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    assert_eq!(output_preimage.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    assert_eq!(state_public_input.len(), PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS);
    assert_eq!(public[0], state_digest, "prior digest statement block");

    let mut output = Vec::with_capacity(PI_CCS_V1_1_SOURCE_COUNT * 1_620);
    for source in 0..PI_CCS_V1_1_SOURCE_COUNT {
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
        derive_pi_ccs_v1_1_transcript(&public, &verifier, &rounds, &output).expect("exact v1_1 transcript replay");
    assert_eq!(actual.alpha(), alpha);
    assert_eq!(actual.gamma(), gamma);
    assert_eq!(actual.round_point(), round_point);
    assert_eq!(actual.outgoing_state(), outgoing_state);

    let package = load(&artifact_bytes(), EXPECTED_IDENTITY).expect("strict package load");
    assert_eq!(authority.len(), 4);
    let derived_context = package
        .derive_pi_ccs_v1_1_verifier_context(&authority[3])
        .expect("package-bound verifier context");
    assert_eq!(derived_context.relation_words(), authority[0]);
    assert_eq!(derived_context.application_words(), authority[1]);
    assert_eq!(derived_context.nifs_key_words(), authority[2]);
    assert_eq!(derived_context.commitment_key_words(), authority[3]);
    assert_eq!(derived_context.digest(), verifier_context);
    let output_evaluations = PiCcsV1_1OutputEvaluations::new(expected_eval_k.clone(), expected_eval_a.clone())
        .expect("nonzero output evaluations");
    let inputs = PiCcsV1_1PackageInputs::new(
        state_preimage,
        output_preimage,
        fresh_commitment,
        rounds,
        output_evaluations,
        state_public_input,
        state_digest,
        derived_context,
    )
    .expect("nonzero package inputs");
    let encoded = package
        .encode_pi_ccs_v1_1_inputs(&inputs)
        .expect("package-owned v1_1 input encoding");
    let decoded = package
        .pi_ccs_v1_1_output_evaluations(encoded.private_values())
        .expect("accepted nonzero-proof output layout");
    assert_eq!(decoded.eval_k(), expected_eval_k);
    assert_eq!(decoded.eval_a(), expected_eval_a);
}

#[test]
fn v1_1_input_encoder_rejects_a_missing_eval_a_matrix() {
    let eval_k = vec![vec![[0, 0]; PI_CCS_V1_1_COEFFICIENT_COUNT]; PI_CCS_V1_1_SOURCE_COUNT];
    let eval_a =
        vec![vec![vec![[0, 0]; PI_CCS_V1_1_COEFFICIENT_COUNT]; PI_CCS_V1_1_MATRIX_COUNT - 1]; PI_CCS_V1_1_SOURCE_COUNT];
    assert!(PiCcsV1_1OutputEvaluations::new(eval_k, eval_a).is_err());
}

#[test]
fn loader_rejects_a_different_verifier_owned_identity() {
    assert!(matches!(
        load(&artifact_bytes(), [0; 4]),
        Err(PackageError::ExpectedIdentityMismatch { .. })
    ));
}

#[test]
fn loader_rejects_a_mutated_package_under_the_expected_identity() {
    let mut value = artifact_value();
    package_array(&mut value)[12]
        .as_array_mut()
        .expect("assertion rows")[0]
        .as_array_mut()
        .expect("assertion row")[1]
        .as_array_mut()
        .expect("A combination")[1]
        .as_array_mut()
        .expect("A terms")[0]
        .as_array_mut()
        .expect("A term")[1] = json!(2);

    assert!(matches!(
        load(&canonical_bytes(&value), EXPECTED_IDENTITY),
        Err(PackageError::ExpectedIdentityMismatch { .. })
    ));
}

#[test]
fn loader_binds_the_lean_owned_ccs_polynomial_to_the_relation_identity() {
    let mut value = artifact_value();
    package_array(&mut value)[4]
        .as_array_mut()
        .expect("CCS relation array")[5]
        .as_array_mut()
        .expect("CCS polynomial terms")[0]
        .as_array_mut()
        .expect("CCS polynomial term")[0] = json!(2);

    assert!(matches!(
        load(&canonical_bytes(&value), EXPECTED_IDENTITY),
        Err(PackageError::ExpectedIdentityMismatch { .. })
    ));
}

#[test]
fn loader_rejects_noncanonical_json_bytes() {
    let mut bytes = artifact_bytes();
    bytes.insert(0, b' ');

    assert!(matches!(
        load(&bytes, EXPECTED_IDENTITY),
        Err(PackageError::NonCanonicalBytes)
    ));
}

#[test]
fn loader_rejects_a_malformed_profile_array() {
    let mut value = artifact_value();
    package_array(&mut value)[1]
        .as_array_mut()
        .expect("profile array")
        .push(json!(0));

    assert!(matches!(
        load(&canonical_bytes(&value), EXPECTED_IDENTITY),
        Err(PackageError::Json(_))
    ));
}

#[test]
fn loader_rejects_a_layout_above_the_stage1_joint_domain_limit() {
    let mut value = artifact_value();
    package_array(&mut value)[3]
        .as_array_mut()
        .expect("layout array")[0] = json!((1u64 << 28) + 1);

    assert!(matches!(
        load(&canonical_bytes(&value), EXPECTED_IDENTITY),
        Err(PackageError::Invalid("2^28 joint domain"))
    ));
}

#[test]
fn loader_rejects_a_noncanonical_matrix_coefficient() {
    let mut value = artifact_value();
    let package = package_array(&mut value);
    package[5].as_array_mut().expect("permutation array")[3]
        .as_array_mut()
        .expect("template rows")[0]
        .as_array_mut()
        .expect("template row")[1]
        .as_array_mut()
        .expect("A combination")[1]
        .as_array_mut()
        .expect("A terms")[0]
        .as_array_mut()
        .expect("A term")[1] = json!(GOLDILOCKS_MODULUS);

    assert!(matches!(
        load(&canonical_bytes(&value), EXPECTED_IDENTITY),
        Err(PackageError::NonCanonicalField { .. })
    ));
}
