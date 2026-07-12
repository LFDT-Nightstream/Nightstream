//! Exact production Poseidon2 permutation SSA artifact and drift gate.

#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_fold_clean::engine::r1cs_circuit::poseidon2::enforce_poseidon2_permutation;
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
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
    BuiltPermutation {
        outputs: output_vars.map(Var::col),
        definitions,
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

fn lean_definition(definition: &Definition) -> String {
    match &definition.rhs {
        Rhs::Linear(terms) => format!("⟨{}, .linear {}⟩", definition.output, lean_terms(terms)),
        Rhs::Product(left, right) => format!(
            "⟨{}, .product {} {}⟩",
            definition.output,
            lean_terms(left),
            lean_terms(right)
        ),
    }
}

fn lean_definitions(definitions: &[Definition]) -> String {
    definitions
        .iter()
        .map(lean_definition)
        .collect::<Vec<_>>()
        .join(",\n   ")
}

fn artifact_hashes(honest: &BuiltPermutation, forged: &[F]) -> (String, String) {
    let row_payload = format!(
        "schema={SCHEMA_VERSION}\nkind=r1cs/poseidon2-goldilocks-width8-permutation\n\
         source=enforce_poseidon2_permutation\noutputs={}\ndefinitions={}\nrows={}\ncols={}\n{}",
        lean_nat_list(honest.outputs),
        honest.definitions.len(),
        honest.builder.rows(),
        honest.builder.cols(),
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
         def definitions : List Definition :=\n  [{}]\n\n\
         def rows : List Row := definitions.map Definition.builderRow\n\n\
         theorem definitions_length : definitions.length = rowCount := by decide\n\
         theorem rows_length : rows.length = rowCount := by decide\n\
         theorem definitions_canonical :\n\
             ∀ definition ∈ definitions, definition.Canonical := by decide\n\
         theorem definitions_wellFormed : WellFormed inputColumns definitions := by decide\n\n\
         end Nightstream.Implementation.R1CS.Poseidon2Permutation\n",
        honest.builder.rows(),
        honest.builder.cols(),
        lean_nat_list(honest.outputs),
        lean_definitions(&honest.definitions),
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
        let expected_path = format!("{path}.expected");
        std::fs::write(&expected_path, rendered).expect("write expected Poseidon2 Lean artifact");
        panic!("generated Lean Poseidon2 artifact drifted. Wrote {expected_path}; review and regenerate intentionally");
    }
}
