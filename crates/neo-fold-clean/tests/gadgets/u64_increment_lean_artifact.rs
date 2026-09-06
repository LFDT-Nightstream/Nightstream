//! Lean artifact exporter and drift gate for the no-wrap u64 increment gadget.
//!
//! This is the next F' counter-path slice after canonical decomposition. It
//! exports the exact rows emitted by `alloc_u64_bits` and
//! `enforce_u64_increment`, plus an honest witness and the minimized overflow
//! witness `u64::MAX -> 0`. Lean proves that every satisfying assignment
//! increments by exactly one over the integers, so wraparound cannot satisfy
//! the exported artifact.

use neo_fold_clean::engine::r1cs_circuit::{alloc_u64_bits, enforce_u64_increment, R1csBuilder};
use neo_math::F;
use p3_field::PrimeField64;

const HONEST_IN: u64 = 0xFFFF_FFFF;
const HONEST_OUT: u64 = 0x1_0000_0000;

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/U64/Generated/U64IncrementArtifact.lean";

fn build_increment(input: u64, output: u64) -> R1csBuilder {
    let mut builder = R1csBuilder::new();
    let input_bits = alloc_u64_bits(&mut builder, input);
    let output_bits = alloc_u64_bits(&mut builder, output);
    enforce_u64_increment(&mut builder, &input_bits, &output_bits);
    builder
}

fn lean_terms(trips: &[(usize, usize, F)], row: usize) -> String {
    let terms: Vec<String> = trips
        .iter()
        .filter(|&&(r, _, _)| r == row)
        .map(|&(_, col, coeff)| format!("({}, {})", col, coeff.as_canonical_u64()))
        .collect();
    format!("[{}]", terms.join(", "))
}

fn lean_witness(name: &str, witness: &[F]) -> String {
    let values: Vec<String> = witness
        .iter()
        .map(|value| value.as_canonical_u64().to_string())
        .collect();
    format!("def {name} : List Nat :=\n  [{}]\n", values.join(", "))
}

fn emit_lean(honest: &R1csBuilder, overflow_witness: &[F]) -> String {
    let (a, b, c) = honest.sparse_triplets();
    let mut out = String::new();
    out.push_str("import Nightstream.Implementation.R1CS.Core.Semantics\n\n");
    out.push_str("/-!\nGENERATED FILE — do not edit by hand.\n\n");
    out.push_str("Exact sparse R1CS rows emitted by `alloc_u64_bits` followed by\n");
    out.push_str("`enforce_u64_increment`, plus honest and overflow witnesses.\n");
    out.push_str("Regenerated and drift-checked by\n");
    out.push_str("`cargo test -p neo-fold-clean --release --test gadgets_u64_increment_lean_artifact`.\n-/\n\n");
    out.push_str("namespace Nightstream.Implementation.R1CS.U64Increment\n\n");
    out.push_str("def inputBitCol (i : Nat) : Nat := i + 1\n");
    out.push_str("def outputBitCol (i : Nat) : Nat := i + 65\n");
    out.push_str("def carryCol (i : Nat) : Nat := i + 129\n\n");
    out.push_str(&format!("def rowCount : Nat := {}\n", honest.rows()));
    out.push_str(&format!("def colCount : Nat := {}\n\n", honest.cols()));
    out.push_str("def rows : List Row :=\n  [");
    let rows: Vec<String> = (0..honest.rows())
        .map(|row| {
            format!(
                "⟨{}, {}, {}⟩",
                lean_terms(a, row),
                lean_terms(b, row),
                lean_terms(c, row)
            )
        })
        .collect();
    out.push_str(&rows.join(",\n   "));
    out.push_str("]\n\n");
    out.push_str("/-- Witness for `0xFFFFFFFF + 1 = 0x100000000`. -/\n");
    out.push_str(&lean_witness("honestWitness", honest.witness()));
    out.push_str("\n/-- Overflow attempt `u64::MAX + 1 = 0`; rejected by the final row. -/\n");
    out.push_str(&lean_witness("overflowWitness", overflow_witness));
    out.push_str("\nend Nightstream.Implementation.R1CS.U64Increment\n");
    out
}

#[test]
fn honest_increment_satisfies_gadget() {
    let builder = build_increment(HONEST_IN, HONEST_OUT);
    assert_eq!(builder.rows(), 255, "u64 increment row count changed");
    assert_eq!(builder.cols(), 192, "u64 increment column count changed");
    assert!(builder.unconstrained_columns().is_empty());
    assert!(builder.is_satisfied());
}

#[test]
fn overflow_is_rejected_by_final_row() {
    let builder = build_increment(u64::MAX, 0);
    assert_eq!(
        builder.first_unsatisfied_row(),
        Some(254),
        "no-wrap increment must reject only at the final carry equation"
    );
}

#[test]
fn lean_increment_artifact_matches_committed_file() {
    let honest = build_increment(HONEST_IN, HONEST_OUT);
    let overflow = build_increment(u64::MAX, 0);
    let emitted = emit_lean(&honest, overflow.witness());

    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != emitted {
        panic!("frozen Lean reference differs: {path:?}");
    }
}
