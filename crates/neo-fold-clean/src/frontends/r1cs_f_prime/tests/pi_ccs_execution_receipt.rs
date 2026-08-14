//! Drift-checked selected-profile receipt for the independent Lean checker.

use std::fmt::Display;

use neo_math::{KExtensions, D, F, K};
use neo_reductions::engines::pi_ccs_joint::{equality, gamma_power};
use neo_reductions::{PiCcsCanonicalStatement, PiCcsExecutionProof, PiCcsExecutionReceipt, PiCcsProof, PiCcsReceiptK};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use super::*;

const ARTIFACT_REL_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/PiCcsExecution/Generated/SelectedReceipt.lean";
const VARIABLES: usize = 24;
const FRESH_COUNT: usize = 1;
const RUNNING_COUNT: usize = 14;
const MATRIX_COUNT: usize = 14;
const DEGREE: usize = 9;
const SOURCE_COUNT: usize = FRESH_COUNT + RUNNING_COUNT;
const CLAIMED_COUNT: usize = RUNNING_COUNT * MATRIX_COUNT * D;
const OUTPUT_COUNT: usize = SOURCE_COUNT * MATRIX_COUNT * D;

fn append_u64(transcript: &mut Poseidon2Transcript, fields: &[u64]) {
    let fields = fields.iter().copied().map(F::from_u64).collect::<Vec<_>>();
    transcript.append_fields_unframed(&fields);
}

fn squeeze(transcript: &mut Poseidon2Transcript, tag: u64, index: Option<usize>) -> K {
    match index {
        Some(index) => append_u64(transcript, &[tag, index as u64]),
        None => append_u64(transcript, &[tag]),
    }
    let fields = transcript.challenge_fields_raw(2);
    neo_math::from_complex(fields[0], fields[1])
}

fn selected_statement_fields() -> Vec<u64> {
    let polynomial = selective_polynomial();
    assert_eq!(polynomial.arity(), 13);
    assert_eq!(polynomial.max_degree(), 8);
    assert_eq!(polynomial.terms().len(), 74);

    let mut fields = vec![41, 24, 1, 14, 14, D as u64, 8, 74];
    for term in polynomial.terms() {
        fields.push(term.coeff.as_canonical_u64());
        fields.push(0);
        fields.push(0);
        fields.extend(term.exps.iter().map(|&exponent| u64::from(exponent)));
    }
    fields.push(47);
    fields
}

fn selected_receipt() -> PiCcsExecutionReceipt {
    assert_eq!(D, 54);
    assert_eq!(CLAIMED_COUNT, 10584);
    assert_eq!(OUTPUT_COUNT, 11340);

    // This fixed identifier names the selected-profile test relation. It is
    // not the missing production matrix-artifact identifier.
    let relation_id = [701, 702, 703, 704];
    let public_fields = [
        vec![
            40,
            2,
            VARIABLES as u64,
            FRESH_COUNT as u64,
            RUNNING_COUNT as u64,
            MATRIX_COUNT as u64,
            D as u64,
            11_437_038,
            16_777_216,
            DEGREE as u64,
            14_944_219,
            11_437_038,
        ],
        relation_id.to_vec(),
        vec![47, 801, 802, 803, 804, RUNNING_COUNT as u64, 1, 901, 902, 903, 904],
    ]
    .concat();
    let statement_fields = selected_statement_fields();

    let mut transcript = Poseidon2Transcript::new(b"nightstream/pi-ccs-execution-receipt/v1");
    let transcript_state = transcript.state().map(|value| value.as_canonical_u64());
    let transcript_absorbed = transcript.absorbed();
    append_u64(&mut transcript, &public_fields);
    append_u64(&mut transcript, &statement_fields);
    let _alpha = (0..VARIABLES)
        .map(|index| squeeze(&mut transcript, 42, Some(index)))
        .collect::<Vec<_>>();
    let gamma = squeeze(&mut transcript, 43, None);

    let half = K::from(F::from_u64(2).inverse());
    let mut claim = gamma_power(gamma, 2 * FRESH_COUNT + RUNNING_COUNT);
    let mut rounds = Vec::with_capacity(VARIABLES);
    let mut round_point = Vec::with_capacity(VARIABLES);
    for round in 0..VARIABLES {
        let constant = claim * half;
        let mut coefficients = vec![K::ZERO; DEGREE + 1];
        coefficients[0] = constant;
        let mut fields = vec![45, round as u64, coefficients.len() as u64];
        for coefficient in &coefficients {
            let (low, high) = coefficient.to_limbs_u64();
            fields.extend([low, high]);
        }
        append_u64(&mut transcript, &fields);
        let challenge = squeeze(&mut transcript, 46, Some(round));
        round_point.push(challenge);
        rounds.push(coefficients);
        claim = constant;
    }

    // Use the derived point as the prior point. This makes the nonzero fixture
    // cover the complete carried-evaluation terminal branch.
    let prior_point = round_point.clone();
    let equality_value = equality(&round_point, &prior_point);
    let carried_factor = equality_value * gamma_power(gamma, 2 * FRESH_COUNT + RUNNING_COUNT);
    assert_ne!(carried_factor, K::ZERO, "selected receipt carried factor");
    let terminal_value = claim * carried_factor.inv();
    assert_eq!(carried_factor * terminal_value, claim);

    let mut claimed_coefficients = vec![PiCcsReceiptK::from(K::ZERO); CLAIMED_COUNT];
    claimed_coefficients[0] = PiCcsReceiptK::from(K::ONE);
    let mut full_output = vec![PiCcsReceiptK::from(K::ZERO); OUTPUT_COUNT];
    full_output[MATRIX_COUNT * D] = terminal_value.into();
    let proof = PiCcsProof::new(rounds);

    PiCcsExecutionReceipt {
        statement: PiCcsCanonicalStatement {
            relation_id,
            transcript_state,
            transcript_absorbed,
            public_fields,
            pi_ccs_statement_fields: statement_fields,
            prior_point: prior_point.into_iter().map(Into::into).collect(),
            claimed_coefficients,
        },
        proof: PiCcsExecutionProof {
            proof_bytes: proof.canonical_bytes(),
            full_output,
        },
    }
}

fn render_list<T: Display>(values: &[T]) -> String {
    if values.is_empty() {
        return "[]".to_owned();
    }
    let mut output = String::from("[\n");
    for (chunk_index, chunk) in values.chunks(16).enumerate() {
        output.push_str("    ");
        output.push_str(
            &chunk
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        );
        if chunk_index + 1 != values.len().div_ceil(16) {
            output.push(',');
        }
        output.push('\n');
    }
    output.push_str("  ]");
    output
}

fn render() -> String {
    let receipt = selected_receipt();
    let relation_id = render_list(&receipt.statement.relation_id);
    let transcript_state = render_list(&receipt.statement.transcript_state);
    let public_fields = render_list(&receipt.statement.public_fields);
    let statement_fields = render_list(&receipt.statement.pi_ccs_statement_fields);
    let proof_bytes = render_list(&receipt.proof.proof_bytes);
    let terminal = receipt.proof.full_output[MATRIX_COUNT * D];

    format!(
        "import Nightstream.Implementation.Rust.PiCcsExecution\n\n\
/-!\n\
GENERATED FILE - do not edit by hand.\n\n\
Selected-profile Rust transcript receipt for the independent Lean checker.\n\
Regenerated and drift-checked by\n\
`cargo test -p neo-fold-clean --lib --release pi_ccs_execution_receipt`.\n\n\
This fixture uses the exact 24-variable, 14-matrix, 74-term relation profile.\n\
Its relation ID names the test fixture, not the production matrix artifact.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.PiCcsExecution.Generated.SelectedReceipt\n\n\
open Nightstream.Implementation.Rust.PiCcsExecution\n\n\
def zeroK : RawK := {{ low := 0, high := 0 }}\n\
def oneK : RawK := {{ low := 1, high := 0 }}\n\
def terminalK : RawK := {{ low := {}, high := {} }}\n\n\
def expectedRelationId : List Nat :=\n  {relation_id}\n\n\
def statement : PiCcsCanonicalStatement where\n\
\x20\x20relationId := expectedRelationId\n\
\x20\x20transcriptState :=\n  {transcript_state}\n\
\x20\x20transcriptAbsorbed := {}\n\
\x20\x20publicFields :=\n  {public_fields}\n\
\x20\x20piCcsStatementFields :=\n  {statement_fields}\n\
\x20\x20priorPoint :=\n    {}\n\
\x20\x20claimedCoefficients := [oneK] ++ List.replicate 10583 zeroK\n\n\
def rustProof : PiCcsExecutionProof where\n\
\x20\x20proofBytes :=\n  {proof_bytes}\n\
\x20\x20fullOutput :=\n\
\x20\x20\x20\x20List.replicate 756 zeroK ++ [terminalK] ++ List.replicate 10583 zeroK\n\n\
end Nightstream.Implementation.R1CS.Artifacts.PiCcsExecution.Generated.SelectedReceipt\n",
        terminal.low,
        terminal.high,
        receipt.statement.transcript_absorbed,
        render_list(
            &receipt
                .statement
                .prior_point
                .iter()
                .map(|value| format!("{{ low := {}, high := {} }}", value.low, value.high))
                .collect::<Vec<_>>()
        )
    )
}

#[test]
fn pi_ccs_execution_receipt_matches_generated_lean() {
    let emitted = render();
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != emitted {
        let expected_path = format!("{path}.expected");
        let parent = std::path::Path::new(&expected_path)
            .parent()
            .expect("artifact path has a parent");
        std::fs::create_dir_all(parent).expect("create generated receipt directory");
        std::fs::write(&expected_path, emitted).expect("write expected receipt artifact");
        panic!("PiCCS receipt artifact drifted; inspect and promote {expected_path}");
    }
}
