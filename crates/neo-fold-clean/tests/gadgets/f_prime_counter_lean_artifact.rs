//! Exact production-used F' counter block export and Rust conformance gate.
//!
//! The exported rows are emitted by the same input-binding and recursive
//! transition helpers used by the F' circuit. Lean projects these rows onto
//! the canonical-u64, increment, and addition artifacts, then proves the two
//! counter equations for every satisfying assignment.

#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_fold_clean::paper::f_prime::r1cs::{
    enforce_f_prime_counter_input_binding, enforce_f_prime_recursive_counter_transition, FPrimeCounterInputWires,
    FPrimeCounterTransitionWires,
};
use neo_fold_clean::paper::f_prime::source_image::FPrimeSourceImage;
use neo_fold_clean::paper::f_prime::source_image_circuit::SourceImageWires;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const CHUNK_IN: u64 = 5;
const STEP_IN: u64 = 9;
const ROWS_IN_CHUNK: u64 = 7;
const CHUNK_OUT: u64 = 6;
const STEP_OUT: u64 = 16;

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrime/Generated/FPrimeCounterArtifact.lean";

#[derive(Clone, Debug)]
struct CounterLayout {
    chunk_in_var: usize,
    step_in_var: usize,
    chunk_input_canonical_map: Vec<usize>,
    step_input_canonical_map: Vec<usize>,
    chunk_output_canonical_map: Vec<usize>,
    step_output_canonical_map: Vec<usize>,
    increment_map: Vec<usize>,
    add_map: Vec<usize>,
    rows_in_chunk_bits: [usize; 64],
}

struct BuiltCounter {
    builder: R1csBuilder,
    layout: CounterLayout,
}

fn cols(vars: &[Var; 64]) -> [usize; 64] {
    std::array::from_fn(|i| vars[i].col())
}

fn canonical_map(var: Var, bits: &[Var; 64], hi_is_max_col: usize, inverse_col: usize) -> Vec<usize> {
    std::iter::once(0)
        .chain(std::iter::once(var.col()))
        .chain(bits.iter().map(|bit| bit.col()))
        .chain([hi_is_max_col, inverse_col])
        .collect()
}

fn increment_map(input: &[Var; 64], output: &[Var; 64], carry_start: usize) -> Vec<usize> {
    std::iter::once(0)
        .chain(input.iter().map(|bit| bit.col()))
        .chain(output.iter().map(|bit| bit.col()))
        .chain(carry_start..carry_start + 63)
        .collect()
}

fn add_map(lhs: &[Var; 64], rhs: &[Var; 64], output: &[Var; 64], carry_start: usize) -> Vec<usize> {
    std::iter::once(0)
        .chain(lhs.iter().map(|bit| bit.col()))
        .chain(rhs.iter().map(|bit| bit.col()))
        .chain(output.iter().map(|bit| bit.col()))
        .chain(carry_start..carry_start + 63)
        .collect()
}

fn build_counter(
    chunk_source: u64,
    step_source: u64,
    chunk_var_value: u64,
    step_var_value: u64,
    chunk_out_value: u64,
    step_out_value: u64,
) -> BuiltCounter {
    let mut image = FPrimeSourceImage::new();
    let chunk_word = image.push_u64_le(chunk_source);
    let step_word = image.push_u64_le(step_source);

    let mut builder = R1csBuilder::new();
    let chunk_in = builder.alloc(F::from_u64(chunk_var_value));
    let step_in = builder.alloc(F::from_u64(step_var_value));
    let source_wires = SourceImageWires::alloc(&mut builder, &image);

    let input_aux_start = builder.cols();
    let input =
        enforce_f_prime_counter_input_binding(&mut builder, &source_wires, chunk_word, step_word, chunk_in, step_in);
    assert_eq!(builder.cols(), input_aux_start + 4, "source canonical layout changed");

    let transition = enforce_f_prime_recursive_counter_transition(
        &mut builder,
        chunk_in,
        step_in,
        &input,
        ROWS_IN_CHUNK,
        chunk_out_value,
        step_out_value,
    );
    let layout = layout_from_wires(chunk_in, step_in, input_aux_start, &input, &transition);

    BuiltCounter { builder, layout }
}

fn layout_from_wires(
    chunk_in: Var,
    step_in: Var,
    input_aux_start: usize,
    input: &FPrimeCounterInputWires,
    transition: &FPrimeCounterTransitionWires,
) -> CounterLayout {
    let chunk_out_last = transition.chunk_count_out_bits[63].col();
    let chunk_out_hi = chunk_out_last + 1;
    let chunk_out_inv = chunk_out_last + 2;
    let increment_carry_start = chunk_out_last + 3;
    let step_out_last = transition.step_count_out_bits[63].col();
    let step_out_hi = step_out_last + 1;
    let step_out_inv = step_out_last + 2;
    let add_carry_start = step_out_last + 3;

    CounterLayout {
        chunk_in_var: chunk_in.col(),
        step_in_var: step_in.col(),
        chunk_input_canonical_map: canonical_map(
            chunk_in,
            &input.chunk_count_bits,
            input_aux_start,
            input_aux_start + 1,
        ),
        step_input_canonical_map: canonical_map(
            step_in,
            &input.step_count_bits,
            input_aux_start + 2,
            input_aux_start + 3,
        ),
        chunk_output_canonical_map: canonical_map(
            transition.chunk_count_out,
            &transition.chunk_count_out_bits,
            chunk_out_hi,
            chunk_out_inv,
        ),
        step_output_canonical_map: canonical_map(
            transition.step_count_out,
            &transition.step_count_out_bits,
            step_out_hi,
            step_out_inv,
        ),
        increment_map: increment_map(
            &input.chunk_count_bits,
            &transition.chunk_count_out_bits,
            increment_carry_start,
        ),
        add_map: add_map(
            &input.step_count_bits,
            &transition.rows_in_chunk_bits,
            &transition.step_count_out_bits,
            add_carry_start,
        ),
        rows_in_chunk_bits: cols(&transition.rows_in_chunk_bits),
    }
}

fn emit_lean(honest: &BuiltCounter, wrong_source: &[F], wrong_step: &[F], wrong_rows: &[F]) -> String {
    let layout = &honest.layout;
    let mut payload = String::new();
    payload.push_str(&format!("def schemaVersion : Nat := {SCHEMA_VERSION}\n"));
    payload.push_str("def artifactKind : String := \"r1cs/f-prime-recursive-counter\"\n");
    payload.push_str("def sourceAnchorInput : String := \"enforce_f_prime_counter_input_binding\"\n");
    payload.push_str("def sourceAnchorTransition : String := \"enforce_f_prime_recursive_counter_transition\"\n");
    payload.push_str(&format!("def rowsInChunk : Nat := {ROWS_IN_CHUNK}\n"));
    payload.push_str(&format!("def chunkInputVarCol : Nat := {}\n", layout.chunk_in_var));
    payload.push_str(&format!("def stepInputVarCol : Nat := {}\n", layout.step_in_var));
    payload.push_str(&format!(
        "def chunkInputCanonicalMap : List Nat := {}\n",
        lean_nat_list(layout.chunk_input_canonical_map.iter().copied())
    ));
    payload.push_str(&format!(
        "def stepInputCanonicalMap : List Nat := {}\n",
        lean_nat_list(layout.step_input_canonical_map.iter().copied())
    ));
    payload.push_str(&format!(
        "def chunkOutputCanonicalMap : List Nat := {}\n",
        lean_nat_list(layout.chunk_output_canonical_map.iter().copied())
    ));
    payload.push_str(&format!(
        "def stepOutputCanonicalMap : List Nat := {}\n",
        lean_nat_list(layout.step_output_canonical_map.iter().copied())
    ));
    payload.push_str(&format!(
        "def incrementMap : List Nat := {}\n",
        lean_nat_list(layout.increment_map.iter().copied())
    ));
    payload.push_str(&format!(
        "def addMap : List Nat := {}\n",
        lean_nat_list(layout.add_map.iter().copied())
    ));
    payload.push_str(&format!(
        "def rowsInChunkBitCols : List Nat := {}\n",
        lean_nat_list(layout.rows_in_chunk_bits)
    ));
    payload.push_str(&format!("def rowCount : Nat := {}\n", honest.builder.rows()));
    payload.push_str(&format!("def colCount : Nat := {}\n\n", honest.builder.cols()));
    payload.push_str(&format!("def rows : List Row :=\n  {}\n\n", lean_rows(&honest.builder)));
    payload.push_str(&lean_witness("honestWitness", honest.builder.witness()));
    payload.push_str("\n");
    payload.push_str(&lean_witness("wrongSourceWitness", wrong_source));
    payload.push_str("\n");
    payload.push_str(&lean_witness("wrongStepWitness", wrong_step));
    payload.push_str("\n");
    payload.push_str(&lean_witness("wrongRowsWitness", wrong_rows));

    let hash = sha256_hex(&payload);
    format!(
        "import Nightstream.Implementation.R1CS.Core.Semantics\n\n\
         /-!\nGENERATED FILE — do not edit by hand.\n\n\
         Schema-v1 exact sparse rows emitted by the production-used F' counter\n\
         input-binding and recursive-transition helpers. Regenerated and\n\
         drift-checked by `gadgets_f_prime_counter_lean_artifact`.\n-/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeCounter\n\n\
         def artifactSha256 : String := \"{hash}\"\n{payload}\n\
         end Nightstream.Implementation.R1CS.FPrimeCounter\n"
    )
}

#[test]
fn production_counter_block_accepts_honest_witness() {
    let built = build_counter(CHUNK_IN, STEP_IN, CHUNK_IN, STEP_IN, CHUNK_OUT, STEP_OUT);
    assert_eq!(built.builder.rows(), 660, "F' counter row count changed");
    assert_eq!(built.builder.cols(), 459, "F' counter column count changed");
    assert!(built.builder.unconstrained_columns().is_empty());
    assert!(built.builder.is_satisfied());
}

#[test]
fn production_counter_block_rejects_disconnected_source_word() {
    let built = build_counter(CHUNK_IN, STEP_IN, CHUNK_IN + 1, STEP_IN, CHUNK_OUT, STEP_OUT);
    assert_eq!(built.builder.first_unsatisfied_row(), Some(132));
}

#[test]
fn production_counter_block_rejects_wrong_step_output() {
    let built = build_counter(CHUNK_IN, STEP_IN, CHUNK_IN, STEP_IN, CHUNK_OUT, STEP_OUT + 1);
    assert_eq!(built.builder.first_unsatisfied_row(), Some(139));
}

#[test]
fn production_counter_block_rejects_forged_rows_in_chunk() {
    let mut built = build_counter(CHUNK_IN, STEP_IN, CHUNK_IN, STEP_IN, CHUNK_OUT, STEP_OUT);
    let bit_zero = built.layout.rows_in_chunk_bits[0];
    built.builder.tamper_witness(bit_zero, F::ZERO);
    assert_eq!(built.builder.first_unsatisfied_row(), Some(400));
}

#[test]
fn lean_f_prime_counter_artifact_matches_committed_file() {
    let honest = build_counter(CHUNK_IN, STEP_IN, CHUNK_IN, STEP_IN, CHUNK_OUT, STEP_OUT);
    let wrong_source = build_counter(CHUNK_IN, STEP_IN, CHUNK_IN + 1, STEP_IN, CHUNK_OUT, STEP_OUT);
    let wrong_step = build_counter(CHUNK_IN, STEP_IN, CHUNK_IN, STEP_IN, CHUNK_OUT, STEP_OUT + 1);
    let mut wrong_rows = build_counter(CHUNK_IN, STEP_IN, CHUNK_IN, STEP_IN, CHUNK_OUT, STEP_OUT);
    wrong_rows
        .builder
        .tamper_witness(wrong_rows.layout.rows_in_chunk_bits[0], F::ZERO);
    let emitted = emit_lean(
        &honest,
        wrong_source.builder.witness(),
        wrong_step.builder.witness(),
        wrong_rows.builder.witness(),
    );

    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != emitted {
        panic!("frozen Lean reference differs: {path:?}");
    }
}
