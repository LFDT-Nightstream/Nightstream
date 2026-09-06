//! Canonical-u64 witness checks and read-only comparison with the frozen Lean artifact.

use neo_fold_clean::engine::r1cs_circuit::builder::R1csBuilder;
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Goldilocks modulus, as u128 so `x + P` stays exact.
const P: u128 = 18_446_744_069_414_584_321;

/// Honest sample value for the exported witness.
const SAMPLE: u64 = 0x0123_4567_89AB_CDEF;

/// Forged sample: `x = 5` re-encoded through the non-canonical bit pattern
/// `x + p < 2^64`. Rows 0..=67 accept it; only the canonicity gate (row 68)
/// rejects it.
const FORGED_BASE: u64 = 5;

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/CanonicalU64/Generated/CanonicalU64Artifact.lean";

/// Build the gadget exactly as the F' counter path does: one allocated field
/// var, then the canonical 64-bit decomposition.
fn build_gadget(value: u64) -> (R1csBuilder, usize) {
    let mut builder = R1csBuilder::new();
    let var = builder.alloc(F::from_u64(value));
    let _bits = decompose_var_to_u64_bits(&mut builder, var);
    (builder, var.col())
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
    let vals: Vec<String> = witness
        .iter()
        .map(|v| v.as_canonical_u64().to_string())
        .collect();
    format!("def {} : List Nat :=\n  [{}]\n", name, vals.join(", "))
}

fn emit_lean(builder: &R1csBuilder, var_col: usize, forged_witness: &[F]) -> String {
    let (a, b, c) = builder.sparse_triplets();
    let mut out = String::new();
    out.push_str("import Nightstream.Implementation.R1CS.Core.Semantics\n\n");
    out.push_str("/-!\nGENERATED FILE — do not edit by hand.\n\n");
    out.push_str("Exact sparse R1CS rows emitted by the production gadget\n");
    out.push_str("`decompose_var_to_u64_bits` (crates/neo-fold-clean/src/engine/r1cs_circuit/u64.rs),\n");
    out.push_str("plus its witness layout and one honest / one forged assignment.\n");
    out.push_str("Regenerated and drift-checked by\n");
    out.push_str("`cargo test -p neo-fold-clean --release --test gadgets_lean_artifact`.\n-/\n\n");
    out.push_str("namespace Nightstream.Implementation.R1CS.CanonicalU64\n\n");
    out.push_str("/-- Column of the decomposed field element. -/\n");
    out.push_str(&format!("def varCol : Nat := {}\n\n", var_col));
    out.push_str("/-- Column of little-endian bit `i` (valid for `i < 64`). -/\n");
    out.push_str("def bitCol (i : Nat) : Nat := i + 2\n\n");
    out.push_str(&format!("def rowCount : Nat := {}\n", builder.rows()));
    out.push_str(&format!("def colCount : Nat := {}\n\n", builder.cols()));
    out.push_str("def rows : List Row :=\n  [");
    let rows: Vec<String> = (0..builder.rows())
        .map(|r| format!("⟨{}, {}, {}⟩", lean_terms(a, r), lean_terms(b, r), lean_terms(c, r)))
        .collect();
    out.push_str(&rows.join(",\n   "));
    out.push_str("]\n\n");
    out.push_str("/-- Witness of the canonical decomposition of `0x0123456789ABCDEF`. -/\n");
    out.push_str(&lean_witness("honestWitness", builder.witness()));
    out.push_str("\n/-- Witness re-encoding `5` through the non-canonical bits of `5 + p`.\n");
    out.push_str("Satisfies every row except the canonicity gate (row 68). -/\n");
    out.push_str(&lean_witness("forgedWitness", forged_witness));
    out.push_str("\nend Nightstream.Implementation.R1CS.CanonicalU64\n");
    out
}

/// Forged assignment: `var = FORGED_BASE`, bits encode `FORGED_BASE + p`,
/// `hi_is_max = 1` (its 32 high bits are all set), `inv = 0` (unused branch).
fn forged_builder() -> R1csBuilder {
    let (mut builder, var_col) = build_gadget(FORGED_BASE);
    let noncanonical = u64::try_from(FORGED_BASE as u128 + P).expect("5 + p fits in u64");
    for i in 0..64 {
        builder.tamper_witness(2 + i, F::from_u64((noncanonical >> i) & 1));
    }
    let hi_is_max_col = 66;
    let inv_col = 67;
    builder.tamper_witness(hi_is_max_col, F::ONE);
    builder.tamper_witness(inv_col, F::ZERO);
    assert_eq!(var_col, 1);
    builder
}

#[test]
fn honest_witness_satisfies_gadget() {
    let (builder, _) = build_gadget(SAMPLE);
    assert_eq!(builder.rows(), 69, "canonical-u64 gadget row count changed");
    assert_eq!(builder.cols(), 68, "canonical-u64 gadget column count changed");
    assert!(
        builder.unconstrained_columns().is_empty(),
        "gadget allocated an unreferenced column"
    );
    assert!(builder.is_satisfied());
}

#[test]
fn forged_noncanonical_witness_fails_exactly_at_canonicity_row() {
    let builder = forged_builder();
    assert_eq!(
        builder.first_unsatisfied_row(),
        Some(68),
        "the non-canonical `x + p` re-encoding must be caught by the canonicity gate and nothing earlier"
    );
}

#[test]
fn lean_artifact_matches_committed_file() {
    let (builder, var_col) = build_gadget(SAMPLE);
    let forged = forged_builder();
    let emitted = emit_lean(&builder, var_col, forged.witness());

    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != emitted {
        panic!("frozen Lean reference differs: {path:?}");
    }
}
