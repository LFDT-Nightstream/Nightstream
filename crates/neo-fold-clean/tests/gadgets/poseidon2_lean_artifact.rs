//! Exact production Poseidon2 permutation SSA artifact and drift gate.

#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_fold_clean::engine::r1cs_circuit::builder::Poseidon2CompactPermutationAudit;
use neo_fold_clean::engine::r1cs_circuit::poseidon2::enforce_poseidon2_permutation;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/Poseidon2/Generated/Poseidon2PermutationArtifact.lean";

#[derive(Clone, Debug)]
enum Rhs {
    Linear(Vec<(usize, u64)>),
    Product(Vec<(usize, u64)>, Vec<(usize, u64)>),
}

#[derive(Clone, Debug)]
struct Definition {
    output: usize,
    rhs: Rhs,
}

struct BuiltPermutation {
    builder: R1csBuilder,
    outputs: [usize; 8],
    definitions: Vec<Definition>,
    compact: Poseidon2CompactPermutationAudit,
}

fn row_terms(trips: &[(usize, usize, F)], row: usize) -> Vec<(usize, u64)> {
    trips
        .iter()
        .filter(|&&(candidate, _, _)| candidate == row)
        .map(|&(_, column, coefficient)| (column, coefficient.as_canonical_u64()))
        .collect()
}

fn canonical_neg(coefficient: u64) -> u64 {
    if coefficient == 0 {
        0
    } else {
        F::ORDER_U64 - coefficient
    }
}

fn normalize(builder: &R1csBuilder, input_columns: &[usize]) -> Vec<Definition> {
    let (a, b, c) = builder.sparse_triplets();
    let mut known = vec![false; builder.cols()];
    for &column in input_columns {
        known[column] = true;
    }
    let mut definitions = Vec::with_capacity(builder.rows());

    for row in 0..builder.rows() {
        let a_terms = row_terms(a, row);
        let b_terms = row_terms(b, row);
        let c_terms = row_terms(c, row);
        let definition = if let [(output, coefficient)] = c_terms.as_slice() {
            assert_eq!(*coefficient, 1, "row {row} product output coefficient");
            assert!(!known[*output], "row {row} overwrites known column {output}");
            Definition {
                output: *output,
                rhs: Rhs::Product(a_terms, b_terms),
            }
        } else {
            assert!(c_terms.is_empty(), "row {row} has unsupported C terms");
            assert_eq!(b_terms, vec![(0, 1)], "row {row} is not a builder linear equality");
            let (&(output, coefficient), negated_rhs) = a_terms
                .split_first()
                .unwrap_or_else(|| panic!("row {row} has empty linear A"));
            assert_eq!(coefficient, 1, "row {row} linear output coefficient");
            assert!(!known[output], "row {row} overwrites known column {output}");
            let rhs = negated_rhs
                .iter()
                .map(|&(column, coefficient)| (column, canonical_neg(coefficient)))
                .collect();
            Definition {
                output,
                rhs: Rhs::Linear(rhs),
            }
        };

        let refs: Vec<usize> = match &definition.rhs {
            Rhs::Linear(terms) => terms.iter().map(|term| term.0).collect(),
            Rhs::Product(left, right) => left.iter().chain(right).map(|term| term.0).collect(),
        };
        for column in refs {
            assert!(known[column], "row {row} reads undefined column {column}");
        }
        known[definition.output] = true;
        definitions.push(definition);
    }
    assert!(
        known.into_iter().all(|value| value),
        "permutation left undefined columns"
    );
    definitions
}

fn build(inputs: [F; 8]) -> BuiltPermutation {
    let mut builder = R1csBuilder::new();
    let input_vars = inputs.map(|value| builder.alloc(value));
    let output_vars = enforce_poseidon2_permutation(&mut builder, &input_vars);
    let input_columns = std::iter::once(0)
        .chain(input_vars.into_iter().map(Var::col))
        .collect::<Vec<_>>();
    let definitions = normalize(&builder, &input_columns);
    let mut compact = builder.poseidon2_compact_permutation_audits();
    assert_eq!(compact.len(), 1, "isolated permutation must have one compact trace");
    BuiltPermutation {
        outputs: output_vars.map(Var::col),
        definitions,
        compact: compact.remove(0),
        builder,
    }
}

fn lean_terms(terms: &[(usize, u64)]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|&(column, coefficient)| format!("({column}, {coefficient})"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_rhs(rhs: &Rhs) -> String {
    match rhs {
        Rhs::Linear(terms) => format!(".linear {}", lean_terms(terms)),
        Rhs::Product(left, right) => format!(".product {} {}", lean_terms(left), lean_terms(right)),
    }
}

fn lean_rhses(definitions: &[Definition]) -> String {
    definitions
        .iter()
        .map(|definition| lean_rhs(&definition.rhs))
        .collect::<Vec<_>>()
        .join(",\n   ")
}

fn lc_terms(lc: &Lc) -> Vec<(usize, u64)> {
    let mut terms = lc
        .terms
        .iter()
        .map(|&(column, coefficient)| (column, coefficient.as_canonical_u64()))
        .collect::<Vec<_>>();
    if lc.constant != F::ZERO {
        terms.push((0, lc.constant.as_canonical_u64()));
    }
    terms
}

fn lean_lc(lc: &Lc) -> String {
    lean_terms(&lc_terms(lc))
}

fn lean_lcs<'a>(lcs: impl IntoIterator<Item = &'a Lc>) -> String {
    format!(
        "[{}]",
        lcs.into_iter()
            .map(lean_lc)
            .collect::<Vec<_>>()
            .join(",\n   ")
    )
}

fn artifact_hashes(honest: &BuiltPermutation, forged: &[F]) -> (String, String) {
    let row_payload = format!(
        "schema={SCHEMA_VERSION}\nkind=r1cs/poseidon2-goldilocks-width8-permutation\n\
         source=enforce_poseidon2_permutation\noutputs={}\ndefinitions={}\nrows={}\ncols={}\n\
         compact_inputs={}\ncompact_sbox_outputs={}\ncompact_outputs={}\n{}",
        lean_nat_list(honest.outputs),
        honest.definitions.len(),
        honest.builder.rows(),
        honest.builder.cols(),
        lean_lcs(honest.compact.sboxes.iter().map(|sbox| &sbox.input)),
        lean_nat_list(honest.compact.sboxes.iter().map(|sbox| sbox.output_col)),
        lean_lcs(&honest.compact.output_linear_forms),
        lean_rows(&honest.builder),
    );
    let witness_payload = format!(
        "{}\n{}",
        lean_witness("honestWitness", honest.builder.witness()),
        lean_witness("forgedWitness", forged),
    );
    (sha256_hex(&row_payload), sha256_hex(&witness_payload))
}

fn render_artifact(honest: &BuiltPermutation, row_hash: &str, witness_hash: &str) -> String {
    assert!(honest
        .definitions
        .iter()
        .enumerate()
        .all(|(index, definition)| definition.output == 9 + index));
    format!(
        "import Nightstream.Implementation.R1CS.Core.Program\n\n\
         /-! Generated exact SSA certificate for the production Goldilocks\n\
         Poseidon2 width-8 permutation. Regenerate only through the Rust drift gate. -/\n\n\
         namespace Nightstream.Implementation.R1CS.Poseidon2Permutation\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\n\
         set_option maxRecDepth 65536\n\
         set_option maxHeartbeats 5000000\n\n\
         def schemaVersion : Nat := {SCHEMA_VERSION}\n\
         def artifactKind : String := \"r1cs/poseidon2-goldilocks-width8-permutation\"\n\
         def sourceAnchor : String := \"enforce_poseidon2_permutation\"\n\
         def artifactSha256 : String := \"{row_hash}\"\n\
         def witnessSha256 : String := \"{witness_hash}\"\n\
         def rowCount : Nat := {}\n\
         def colCount : Nat := {}\n\
         def inputColumns : List Nat := [0, 1, 2, 3, 4, 5, 6, 7, 8]\n\
         def outputColumns : List Nat := {}\n\n\
         def compactSboxInputTerms : List (List (Nat × Nat)) :=\n  {}\n\n\
         def compactSboxOutputColumns : List Nat := {}\n\n\
         def compactOutputLinearForms : List (List (Nat × Nat)) :=\n  {}\n\n\
         def definitionOutputColumns : List Nat := List.range' 9 rowCount\n\n\
         def definitionRhs : List Rhs :=\n  [{}]\n\n\
         def definitions : List Definition :=\n  \
         (definitionOutputColumns.zip definitionRhs).map fun entry =>\n    \
         ⟨entry.1, entry.2⟩\n\n\
         def rows : List Row := definitions.map Definition.builderRow\n\n\
         theorem definition_output_mem {{definition : Definition}}\n\
             (member : definition ∈ definitions) :\n\
             definition.output ∈ definitionOutputColumns := by\n  \
         rw [definitions] at member\n  \
         rcases List.mem_map.mp member with ⟨entry, entryMember, rfl⟩\n  \
         exact (List.of_mem_zip entryMember).1\n\n\
         theorem definitions_length : definitions.length = rowCount := by decide\n\
         theorem rows_length : rows.length = rowCount := by decide\n\
         theorem definitions_canonical :\n\
             ∀ definition ∈ definitions, definition.Canonical := by decide\n\
         theorem definitions_wellFormed : WellFormed inputColumns definitions := by decide\n\n\
         theorem compact_trace_shape :\n\
             compactSboxInputTerms.length = 86 ∧\n\
             compactSboxOutputColumns.length = 86 ∧\n\
             compactSboxOutputColumns.Nodup ∧\n\
             compactOutputLinearForms.length = 8 := by decide\n\n\
         end Nightstream.Implementation.R1CS.Poseidon2Permutation\n",
        honest.builder.rows(),
        honest.builder.cols(),
        lean_nat_list(honest.outputs),
        lean_lcs(honest.compact.sboxes.iter().map(|sbox| &sbox.input)),
        lean_nat_list(honest.compact.sboxes.iter().map(|sbox| sbox.output_col)),
        lean_lcs(&honest.compact.output_linear_forms),
        lean_rhses(&honest.definitions),
    )
}

fn honest_inputs() -> [F; 8] {
    std::array::from_fn(|index| F::from_u64(17 + index as u64 * 13))
}

#[test]
fn poseidon2_permutation_normalizes_every_exact_row() {
    let built = build(honest_inputs());
    assert_eq!(built.definitions.len(), built.builder.rows());
    assert!(built.builder.unconstrained_columns().is_empty());
    assert!(built.builder.is_satisfied());
}

#[test]
fn poseidon2_permutation_rejects_input_tamper_without_recomputed_trace() {
    let mut built = build(honest_inputs());
    built
        .builder
        .tamper_witness(1, built.builder.witness()[1] + F::ONE);
    assert!(!built.builder.is_satisfied());
}

#[test]
fn lean_poseidon2_permutation_artifact_matches_committed_file() {
    let honest = build(honest_inputs());
    let mut forged = build(honest_inputs());
    forged
        .builder
        .tamper_witness(1, forged.builder.witness()[1] + F::ONE);
    let (row_hash, witness_hash) = artifact_hashes(&honest, forged.builder.witness());
    let rendered = render_artifact(&honest, &row_hash, &witness_hash);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        panic!("frozen Lean reference differs: {path:?}");
    }
}
