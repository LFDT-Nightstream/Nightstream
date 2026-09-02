//! Nonzero assignment execution for the Lean-authored concrete application.
//!
//! The package witness IR generates the assignment. This target does not
//! claim independent row evaluation; that remains a separate gate.

use std::{fs, path::PathBuf};

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use nightstream_fprime::load_poseidon2_hash_chain_v1_package;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde_json::Value;

#[allow(dead_code, unused_imports)]
#[path = "../src/bin/check_package_conformance/support.rs"]
mod conformance_support;

const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;
const PRIVATE_INPUT_COUNT: usize = 177_326;
const PUBLIC_INPUT_COUNT: usize = 278;
const TOTAL_COLUMN_COUNT: usize = 29_344_425;
const STATE_PREIMAGE_WORDS: usize = 49_393;
const OUTPUT_DIGEST_PUBLIC_START: usize = 270;
const ITERATION_WORD: usize = 28;
const INITIAL_STATE_WORD_START: usize = 30;
const CURRENT_WORD_START: usize = 35;
const APPLICATION_TAG: &[u8; 40] = b"Nightstream/Stage1/Poseidon2HashChain/v1";
const APPLICATION_MESSAGE: [u64; 4] = [7, 11, 13, 17];

fn artifact_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts")
        .join(name)
}

fn parity(name: &str, schema: u64) -> (Vec<Value>, Vec<Value>) {
    let bytes = fs::read(artifact_path(name)).expect("Lean parity bytes");
    let value: Value = serde_json::from_slice(&bytes).expect("Lean parity JSON");
    let fields = value.as_array().expect("Lean parity tuple");
    assert_eq!(fields.len(), 3, "Lean parity tuple length");
    assert_eq!(fields[0].as_u64(), Some(schema), "Lean parity schema");
    (
        fields[1].as_array().expect("Lean parity input").clone(),
        fields[2].as_array().expect("Lean parity output").clone(),
    )
}

fn append_words(value: &Value, output: &mut Vec<u64>) {
    if let Some(word) = value.as_u64() {
        assert!(word < GOLDILOCKS_MODULUS, "canonical fixture word");
        output.push(word);
        return;
    }
    for child in value.as_array().expect("nested word array") {
        append_words(child, output);
    }
}

fn words(value: &Value) -> Vec<u64> {
    let mut output = Vec::new();
    append_words(value, &mut output);
    output
}

fn hash_words(input: &[u64]) -> [u64; 4] {
    let fields = input
        .iter()
        .copied()
        .map(Goldilocks::from_u64)
        .collect::<Vec<_>>();
    poseidon2_hash(&fields).map(|value| value.as_canonical_u64())
}

fn concrete_inputs(
    expected_verifier_context: [u64; 4],
    expected_package_identity: [u64; 4],
) -> (Vec<u64>, Vec<u64>, [u64; 4], [u64; 4]) {
    let (pi_ccs, _) = parity("nightstream-fprime-stage1-piccs-parity-v1.json", 8);
    let (pi_dec, pi_dec_output) = parity("nightstream-fprime-stage1-pidec-parity-v1.json", 2);
    let (application, application_result) = parity("nightstream-fprime-stage1-poseidon2-hash-chain-v1-parity.json", 2);

    assert_eq!(words(&pi_ccs[4]), expected_verifier_context);
    assert_eq!(words(&pi_dec[6]), expected_package_identity);

    let tag = APPLICATION_TAG
        .iter()
        .copied()
        .map(u64::from)
        .collect::<Vec<_>>();
    assert_eq!(words(&application[0]), tag);
    let prior_preimage = words(&application[1]);
    assert_eq!(prior_preimage, words(&pi_ccs[0]));
    let message: [u64; 4] = words(&application[2])
        .try_into()
        .expect("four application message words");
    assert_eq!(message, APPLICATION_MESSAGE);
    let application_output: [u64; 4] = words(&application_result[0])
        .try_into()
        .expect("four application output words");
    let output_preimage = words(&application_result[1]);
    let output_digest: [u64; 4] = words(&application_result[2])
        .try_into()
        .expect("four final digest words");
    let terminal_layout: [u64; 4] = words(&application_result[3])
        .try_into()
        .expect("four terminal-layout words");

    let prior_current: [u64; 4] = prior_preimage[CURRENT_WORD_START..CURRENT_WORD_START + 4]
        .try_into()
        .expect("four prior current-state words");
    let mut application_input = tag;
    application_input.extend_from_slice(&prior_current);
    application_input.extend_from_slice(&message);
    assert_eq!(hash_words(&application_input), application_output);
    assert_eq!(output_preimage[ITERATION_WORD], prior_preimage[ITERATION_WORD] + 1);
    assert_eq!(
        &output_preimage[CURRENT_WORD_START..CURRENT_WORD_START + 4],
        application_output,
    );
    assert_eq!(hash_words(&output_preimage), output_digest);
    let transition_preimage = words(&pi_dec_output[17]);
    assert_eq!(transition_preimage.len(), output_preimage.len());
    for index in 0..output_preimage.len() {
        if index != ITERATION_WORD && !(CURRENT_WORD_START..CURRENT_WORD_START + 4).contains(&index) {
            assert_eq!(output_preimage[index], transition_preimage[index]);
        }
    }

    let mut private_inputs = Vec::with_capacity(PRIVATE_INPUT_COUNT);
    private_inputs.extend_from_slice(&prior_preimage);
    private_inputs.extend_from_slice(&output_preimage);
    append_words(&pi_ccs[5], &mut private_inputs);
    append_words(&pi_ccs[6], &mut private_inputs);
    let eval_k = pi_ccs[7].as_array().expect("PiCCS Eval_K sources");
    let eval_a = pi_ccs[8].as_array().expect("PiCCS Eval_A sources");
    assert_eq!(eval_k.len(), eval_a.len(), "PiCCS output source count");
    for (source_k, source_a) in eval_k.iter().zip(eval_a) {
        append_words(source_k, &mut private_inputs);
        append_words(source_a, &mut private_inputs);
    }
    for index in 1..=4 {
        append_words(&pi_dec[index], &mut private_inputs);
    }
    private_inputs.extend_from_slice(&message);
    assert_eq!(private_inputs.len(), PRIVATE_INPUT_COUNT);

    let mut public_inputs = words(&pi_ccs[2]);
    public_inputs.extend_from_slice(&output_digest);
    append_words(&pi_ccs[4], &mut public_inputs);
    assert_eq!(public_inputs.len(), PUBLIC_INPUT_COUNT);

    (private_inputs, public_inputs, application_output, terminal_layout)
}

#[test]
#[ignore = "exact-cut matrix and assignment conformance; run this target explicitly under the 300-second cap"]
fn package_generates_the_complete_nonzero_hash_chain_assignment() {
    let bytes = fs::read(artifact_path("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"))
        .expect("Lean-emitted concrete package");
    let package = load_poseidon2_hash_chain_v1_package(&bytes).expect("verifier-owned production package");
    assert_eq!(package.private_input_count(), PRIVATE_INPUT_COUNT);
    assert_eq!(package.public_input_count(), PUBLIC_INPUT_COUNT);
    assert_eq!(package.total_column_count(), TOTAL_COLUMN_COUNT);

    let binding = package
        .production_verifier_binding()
        .expect("fixed production verifier binding");
    let (private_inputs, public_inputs, expected_output, expected_terminal) =
        concrete_inputs(binding.verifier_context().digest(), binding.package_identity());
    let terminal = package.terminal().expect("Lean-owned terminal layout");
    assert_eq!(terminal.row_start() as u64, expected_terminal[0]);
    assert_eq!(terminal.row_count() as u64, expected_terminal[1]);
    assert_eq!(terminal.running_claim_count() as u64, expected_terminal[2]);
    assert_eq!(terminal.fresh_claim_count() as u64, expected_terminal[3]);
    let assignment = package
        .execute_witness(&private_inputs, &public_inputs)
        .expect("complete concrete assignment");
    assert_eq!(
        assignment.private_values().len(),
        TOTAL_COLUMN_COUNT - PUBLIC_INPUT_COUNT - 1
    );
    assert_eq!(assignment.public_values(), public_inputs);
    for (&column, &value) in package
        .application()
        .witness_columns()
        .iter()
        .zip(&APPLICATION_MESSAGE)
    {
        assert_eq!(assignment.private_values()[column], value);
    }
    for (column, value) in package
        .application()
        .output_columns()
        .into_iter()
        .zip(expected_output)
    {
        assert_eq!(assignment.private_values()[column], value);
    }
    assert!(package
        .application()
        .private_range()
        .any(|column| assignment.private_values()[column] != 0));
    assert_eq!(
        conformance_support::evaluate_sealed_assignment(&bytes, &assignment),
        package.physical_row_count(),
    );
    let matrix_nonzeros = {
        let matrices = package
            .r1cs_matrices()
            .expect("final concrete A/B/C matrices");
        conformance_support::compare_sealed_matrices(&bytes, &matrices)
    };
    assert_eq!(matrix_nonzeros, [93_701_820, 39_358_148, 28_868_018]);
    eprintln!("concrete_final_matrix_nonzeros={matrix_nonzeros:?}");

    let message_start = private_inputs.len() - APPLICATION_MESSAGE.len();
    for lane in 0..APPLICATION_MESSAGE.len() {
        let mut changed = private_inputs.clone();
        changed[message_start + lane] += 1;
        assert!(package.execute_witness(&changed, &public_inputs).is_err());
    }

    for output_word in std::iter::once(ITERATION_WORD).chain(INITIAL_STATE_WORD_START..CURRENT_WORD_START) {
        let mut changed_private = private_inputs.clone();
        changed_private[STATE_PREIMAGE_WORDS + output_word] += 1;
        let output_preimage = &changed_private[STATE_PREIMAGE_WORDS..2 * STATE_PREIMAGE_WORDS];
        let mut changed_public = public_inputs.clone();
        changed_public[OUTPUT_DIGEST_PUBLIC_START..OUTPUT_DIGEST_PUBLIC_START + 4]
            .copy_from_slice(&hash_words(output_preimage));
        assert!(package
            .execute_witness(&changed_private, &changed_public)
            .is_err());
    }
}
