use std::{fs, path::PathBuf};

use neo_fold_clean::frontends::r1cs_f_prime::ivc::load_pi_ccs_v1_1_package;
use nightstream_fprime::{
    PiCcsV1_1OutputEvaluations, PiCcsV1_1PackageInputs, PiCcsV1_1VerifierContext, PiDecV1_1PackageInputs,
    PI_CCS_V1_1_COEFFICIENT_COUNT, PI_CCS_V1_1_FRESH_COMMITMENT_WORDS, PI_CCS_V1_1_MATRIX_COUNT,
    PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT,
    PI_CCS_V1_1_SOURCE_COUNT, PI_CCS_V1_1_STATE_PREIMAGE_WORDS, PI_DEC_V1_1_CHILD_COUNT,
};
use serde_json::Value;

const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;
const PACKAGE_IDENTITY: [u64; 4] = [
    5_326_948_389_888_638_380,
    15_945_253_772_729_055_182,
    12_038_831_075_978_321_435,
    4_066_786_242_110_063_495,
];

struct PiDecFixture {
    inputs: PiDecV1_1PackageInputs,
    output_preimage: Vec<u64>,
    output_digest: [u64; 4],
}

fn artifact_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts")
        .join(name)
}

fn json_words(value: &Value, location: &str) -> Vec<u64> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{location} array"))
        .iter()
        .map(|value| {
            let word = value
                .as_u64()
                .unwrap_or_else(|| panic!("{location} canonical word"));
            assert!(word < GOLDILOCKS_MODULUS, "{location} canonical word");
            word
        })
        .collect()
}

fn json_extensions(value: &Value, location: &str) -> Vec<[u64; 2]> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{location} array"))
        .iter()
        .map(|value| {
            json_words(value, location)
                .try_into()
                .unwrap_or_else(|_| panic!("{location} extension width"))
        })
        .collect()
}

fn parity_input(name: &str, schema: u64, input_length: usize) -> Vec<Value> {
    let bytes = fs::read(artifact_path(name)).expect("Lean parity bytes");
    let parity: Value = serde_json::from_slice(&bytes).expect("Lean parity JSON");
    let parity = parity.as_array().expect("Lean parity tuple");
    assert_eq!(parity.len(), 3, "Lean parity tuple length");
    assert_eq!(parity[0].as_u64(), Some(schema), "Lean parity schema");
    let input = parity[1].as_array().expect("Lean parity input").clone();
    assert_eq!(input.len(), input_length, "Lean parity input length");
    input
}

fn commitment_key_words() -> Vec<u64> {
    let input = parity_input("nightstream-fprime-stage1-piccs-parity-v1.json", 7, 12);
    let authority = input[11]
        .as_array()
        .expect("Lean verifier-context authority");
    json_words(&authority[3], "Lean commitment-key authority")
}

fn pi_ccs_inputs(
    verifier_context: PiCcsV1_1VerifierContext,
    output_preimage: Vec<u64>,
    output_digest: [u64; 4],
) -> (PiCcsV1_1PackageInputs, Vec<u64>) {
    let input = parity_input("nightstream-fprime-stage1-piccs-parity-v1.json", 7, 12);
    let authority = input[11]
        .as_array()
        .expect("Lean verifier-context authority");
    assert_eq!(json_words(&authority[0], "Lean package identity"), PACKAGE_IDENTITY);
    assert_eq!(
        json_words(&input[4], "Lean verifier-context digest"),
        verifier_context.digest()
    );

    let prior_preimage = json_words(&input[0], "Lean prior preimage");
    let phase_output_preimage = json_words(&input[1], "Lean phase output preimage");
    let prior_public = json_words(&input[2], "Lean prior public input");
    let phase_output_digest: [u64; 4] = json_words(&input[3], "Lean phase output digest")
        .try_into()
        .expect("four-word phase output digest");
    let fresh_commitment = json_words(&input[5], "Lean fresh commitment");
    let round_messages = input[6]
        .as_array()
        .expect("Lean round messages")
        .iter()
        .map(|round| json_extensions(round, "Lean round message"))
        .collect::<Vec<_>>();
    let eval_k = input[7]
        .as_array()
        .expect("Lean Eval_K")
        .iter()
        .map(|source| json_extensions(source, "Lean Eval_K source"))
        .collect::<Vec<_>>();
    let eval_a = input[8]
        .as_array()
        .expect("Lean Eval_A")
        .iter()
        .map(|source| {
            source
                .as_array()
                .expect("Lean Eval_A source")
                .iter()
                .map(|matrix| json_extensions(matrix, "Lean Eval_A matrix"))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert_eq!(prior_preimage.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    assert_eq!(phase_output_preimage.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    assert_eq!(output_preimage.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    assert_ne!(
        phase_output_digest, output_digest,
        "running transition changes output digest"
    );
    assert_eq!(prior_public.len(), PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS);
    assert_eq!(fresh_commitment.len(), PI_CCS_V1_1_FRESH_COMMITMENT_WORDS);
    assert_eq!(round_messages.len(), PI_CCS_V1_1_ROUND_COUNT);
    assert!(round_messages
        .iter()
        .all(|round| round.len() == PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT));
    assert_eq!(eval_k.len(), PI_CCS_V1_1_SOURCE_COUNT);
    assert!(eval_k
        .iter()
        .all(|source| source.len() == PI_CCS_V1_1_COEFFICIENT_COUNT));
    assert_eq!(eval_a.len(), PI_CCS_V1_1_SOURCE_COUNT);
    assert!(eval_a.iter().all(|source| {
        source.len() == PI_CCS_V1_1_MATRIX_COUNT
            && source
                .iter()
                .all(|matrix| matrix.len() == PI_CCS_V1_1_COEFFICIENT_COUNT)
    }));

    let mut expected_public = prior_public.clone();
    expected_public.extend_from_slice(&output_digest);
    expected_public.extend_from_slice(&verifier_context.digest());
    let evaluations = PiCcsV1_1OutputEvaluations::new(eval_k, eval_a).expect("typed Lean output evaluations");
    let inputs = PiCcsV1_1PackageInputs::new(
        prior_preimage,
        output_preimage,
        fresh_commitment,
        round_messages,
        evaluations,
        prior_public,
        output_digest,
        verifier_context,
    )
    .expect("typed Lean PiCCS inputs");
    (inputs, expected_public)
}

fn pi_dec_inputs() -> PiDecFixture {
    let bytes =
        fs::read(artifact_path("nightstream-fprime-stage1-pidec-parity-v1.json")).expect("Lean PiDEC parity bytes");
    let parity: Value = serde_json::from_slice(&bytes).expect("Lean PiDEC parity JSON");
    let parity = parity.as_array().expect("Lean PiDEC parity tuple");
    assert_eq!(parity.len(), 3, "Lean PiDEC parity tuple length");
    assert_eq!(parity[0].as_u64(), Some(2), "Lean PiDEC parity schema");
    let input = parity[1].as_array().expect("Lean PiDEC parity input");
    let result = parity[2].as_array().expect("Lean PiDEC parity result");
    assert_eq!(input.len(), 7, "Lean PiDEC parity input length");
    assert_eq!(result.len(), 19, "Lean PiDEC parity result length");
    assert_eq!(result[0].as_u64(), Some(1), "Lean PiDEC acceptance");
    assert_eq!(json_words(&input[6], "Lean PiDEC package identity"), PACKAGE_IDENTITY);
    let child_commitments = input[1]
        .as_array()
        .expect("Lean child commitments")
        .iter()
        .map(|child| json_words(child, "Lean child commitment"))
        .collect::<Vec<_>>();
    let child_eval_k = input[2]
        .as_array()
        .expect("Lean child Eval_K")
        .iter()
        .map(|child| json_extensions(child, "Lean child Eval_K"))
        .collect::<Vec<_>>();
    let child_eval_a = input[3]
        .as_array()
        .expect("Lean child Eval_A")
        .iter()
        .map(|child| {
            child
                .as_array()
                .expect("Lean child Eval_A matrices")
                .iter()
                .map(|matrix| json_extensions(matrix, "Lean child Eval_A matrix"))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let child_public_inputs = input[4]
        .as_array()
        .expect("Lean child public inputs")
        .iter()
        .map(|child| json_words(child, "Lean child public input"))
        .collect::<Vec<_>>();
    assert_eq!(child_commitments.len(), PI_DEC_V1_1_CHILD_COUNT);
    let output_preimage = json_words(&result[17], "Lean running-transition output preimage");
    assert_eq!(output_preimage.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    let output_digest = json_words(&result[18], "Lean running-transition output digest")
        .try_into()
        .expect("four-word running-transition output digest");
    PiDecFixture {
        inputs: PiDecV1_1PackageInputs::new(child_commitments, child_eval_k, child_eval_a, child_public_inputs)
            .expect("typed Lean PiDEC inputs"),
        output_preimage,
        output_digest,
    }
}

#[test]
fn application_consumes_identity_bound_lean_package_for_canonical_state() {
    let bytes = fs::read(artifact_path("nightstream-fprime-stage1-v1.json")).expect("Lean-emitted package bytes");
    let (prover, verifier) = load_pi_ccs_v1_1_package(&bytes, PACKAGE_IDENTITY, &commitment_key_words())
        .expect("identity-bound application package runtime");
    assert_eq!(prover.relation_identifier(), PACKAGE_IDENTITY);
    assert_eq!(verifier.relation_identifier(), PACKAGE_IDENTITY);
    assert_eq!(prover.verifier_context(), verifier.verifier_context());

    let PiDecFixture {
        inputs: pi_dec,
        output_preimage,
        output_digest,
    } = pi_dec_inputs();
    let (pi_ccs, expected_public) = pi_ccs_inputs(prover.verifier_context().clone(), output_preimage, output_digest);
    let proof = prover
        .prove(&pi_ccs, &pi_dec)
        .expect("application package proof");
    verifier
        .verify(&proof, &expected_public)
        .expect("application package verification");

    let mut changed_public = expected_public;
    changed_public[0] ^= 1;
    assert!(verifier.verify(&proof, &changed_public).is_err());

    let stats = prover.matrix_stats();
    assert!(stats.a_nonzeros() > 0);
    assert!(stats.b_nonzeros() > 0);
    assert!(stats.c_nonzeros() > 0);
}
