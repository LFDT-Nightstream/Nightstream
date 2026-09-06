//! Exact F' chunk-shape digest SSA artifact, conformance, and drift gate.

#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_ajtai::Commitment;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use neo_fold_clean::paper::digest::f_prime_chunk_public_digest;
use neo_fold_clean::paper::f_prime::digest_circuit::enforce_f_prime_chunk_public_digest_circuit;
use neo_fold_clean::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
use neo_fold_clean::paper::relations::CcsClaim;
use neo_math::{D, F};
use neo_params::goldilocks_paper_b2;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrime/FPrimeChunkDigestArtifact.lean";
const SHARD_REL_PREFIX: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrime/Generated/FPrimeChunkDigestDefinitions";
const START_INDEX: u64 = 9;
const FRESH_LEN: usize = 3;
const M_IN: usize = F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
const SHARD_SIZE: usize = 1_200;

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

struct BuiltChunkDigest {
    builder: R1csBuilder,
    claimed_cols: [usize; 4],
    computed_cols: [usize; 4],
    binding_row_start: usize,
    definitions: Vec<Definition>,
}

fn expected_digest() -> [F; 4] {
    let kappa = goldilocks_paper_b2::KAPPA as usize;
    let template = CcsClaim {
        c: Commitment::zeros(D, kappa),
        x: vec![F::ZERO; M_IN],
        m_in: M_IN,
        adv: None,
    };
    f_prime_chunk_public_digest(START_INDEX, &vec![template; FRESH_LEN])
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
            assert_eq!(b_terms, vec![(0, 1)], "row {row} is not a builder equality");
            let (&(output, coefficient), negated_rhs) = a_terms
                .split_first()
                .unwrap_or_else(|| panic!("row {row} has empty linear A"));
            assert_eq!(coefficient, 1, "row {row} linear output coefficient");
            assert!(!known[output], "row {row} overwrites known column {output}");
            Definition {
                output,
                rhs: Rhs::Linear(
                    negated_rhs
                        .iter()
                        .map(|&(column, coefficient)| (column, canonical_neg(coefficient)))
                        .collect(),
                ),
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
        "chunk digest left undefined columns"
    );
    definitions
}

fn build(claimed: [F; 4]) -> BuiltChunkDigest {
    let mut builder = R1csBuilder::new();
    let start = builder.alloc(F::from_u64(START_INDEX));
    let claimed_vars = claimed.map(|value| builder.alloc(value));
    let computed = enforce_f_prime_chunk_public_digest_circuit(
        &mut builder,
        start,
        FRESH_LEN,
        D,
        goldilocks_paper_b2::KAPPA as usize,
        M_IN,
    );
    let binding_row_start = builder.rows();
    for lane in 0..4 {
        builder.enforce_eq(&Lc::from_var(claimed_vars[lane]), &Lc::from_var(computed[lane]));
    }
    // Columns 2..5 are deliberately not inputs. The final four rows define
    // them from the computed digest, so the entire block is deterministic
    // from only constant-one and start-index columns.
    let definitions = normalize(&builder, &[0, start.col()]);
    BuiltChunkDigest {
        claimed_cols: claimed_vars.map(Var::col),
        computed_cols: computed.map(Var::col),
        binding_row_start,
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

fn artifact_hashes(honest: &BuiltChunkDigest, forged: &[F]) -> (String, String) {
    let row_payload = format!(
        "schema={SCHEMA_VERSION}\nkind=r1cs/f-prime-chunk-shape-digest\n\
         source=enforce_f_prime_chunk_public_digest_circuit\nclaimed_cols={}\ncomputed_cols={}\n\
         binding_row_start={}\nrows={}\ncols={}\n{}",
        lean_nat_list(honest.claimed_cols),
        lean_nat_list(honest.computed_cols),
        honest.binding_row_start,
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

fn shard_count(definitions: &[Definition]) -> usize {
    definitions.len().div_ceil(SHARD_SIZE)
}

fn render_shard(index: usize, definitions: &[Definition]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.Program\n\n\
         /-! Generated F' chunk-digest SSA shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeChunkDigest.Generated\n\n\
         open Nightstream.Implementation.R1CS.Program\n\n\
         set_option maxRecDepth 262144\n\n\
         def definitions{index} : List Definition :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.FPrimeChunkDigest.Generated\n",
        lean_definitions(definitions),
    )
}

fn render_main(honest: &BuiltChunkDigest, row_hash: &str, witness_hash: &str) -> String {
    let count = shard_count(&honest.definitions);
    let imports = (0..count)
        .map(|index| {
            format!(
                "import Nightstream.Implementation.R1CS.Artifacts.FPrime.Generated.FPrimeChunkDigestDefinitions{index}"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let definitions = (0..count)
        .map(|index| format!("Generated.definitions{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    format!(
        "{imports}\n\n\
         /-! Exact sharded SSA artifact for the production F' chunk-shape digest. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeChunkDigest\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\n\
         set_option maxRecDepth 262144\n\n\
         def schemaVersion : Nat := {SCHEMA_VERSION}\n\
         def artifactKind : String := \"r1cs/f-prime-chunk-shape-digest\"\n\
         def sourceAnchor : String := \"enforce_f_prime_chunk_public_digest_circuit\"\n\
         def artifactSha256 : String := \"{row_hash}\"\n\
         def witnessSha256 : String := \"{witness_hash}\"\n\n\
         def inputColumns : List Nat := [0, 1]\n\
         def claimedColumns : List Nat := {}\n\
         def computedColumns : List Nat := {}\n\
         def bindingRowStart : Nat := {}\n\
         def fullRowCount : Nat := {}\n\
         def fullColCount : Nat := {}\n\n\
         def definitions : List Definition :=\n    {definitions}\n\n\
         def rows : List Row := definitions.map Definition.builderRow\n\n\
         def columnPairs : List (Nat × Nat) := claimedColumns.zip computedColumns\n\
         def equalityRow (columns : Nat × Nat) : Row :=\n\
           ⟨[(columns.1, 1), (columns.2, goldilocksP - 1)], [(0, 1)], []⟩\n\
         def bindingRows : List Row := columnPairs.map equalityRow\n\n\
         theorem definitions_length : definitions.length = fullRowCount := by native_decide\n\
         theorem rows_length : rows.length = fullRowCount := by native_decide\n\
         theorem bindingRows_length : bindingRows.length = 4 := by native_decide\n\
         theorem definitions_canonical :\n\
             ∀ definition ∈ definitions, definition.Canonical := by native_decide\n\
         theorem definitions_wellFormed : WellFormed inputColumns definitions := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeChunkDigest\n",
        lean_nat_list(honest.claimed_cols),
        lean_nat_list(honest.computed_cols),
        honest.binding_row_start,
        honest.builder.rows(),
        honest.builder.cols(),
    )
}

fn artifact_path(relative: &str) -> String {
    format!("{}{}", env!("CARGO_MANIFEST_DIR"), relative)
}

#[test]
fn chunk_digest_binding_accepts_native_shape_digest() {
    let built = build(expected_digest());
    assert!(
        built.binding_row_start > 4_000,
        "Poseidon2 shape digest unexpectedly small"
    );
    assert_eq!(built.builder.rows(), built.binding_row_start + 4);
    assert_eq!(built.definitions.len(), built.builder.rows());
    assert!(built.builder.unconstrained_columns().is_empty());
    assert!(built.builder.is_satisfied());
}

#[test]
fn chunk_digest_binding_rejects_self_consistent_wrong_claim() {
    let mut wrong = expected_digest();
    wrong[0] += F::ONE;
    let built = build(wrong);
    assert_eq!(built.builder.first_unsatisfied_row(), Some(built.binding_row_start));
}

#[test]
fn lean_chunk_digest_artifact_matches_committed_files() {
    let honest = build(expected_digest());
    let mut wrong = expected_digest();
    wrong[0] += F::ONE;
    let forged = build(wrong);
    let (row_hash, witness_hash) = artifact_hashes(&honest, forged.builder.witness());

    let main_path = artifact_path(ARTIFACT_REL_PATH);
    let rendered_main = render_main(&honest, &row_hash, &witness_hash);
    let mut drifted = Vec::new();
    if std::fs::read_to_string(&main_path).unwrap_or_default() != rendered_main {
        drifted.push(main_path.clone());
    }

    for (index, shard) in honest.definitions.chunks(SHARD_SIZE).enumerate() {
        let path = artifact_path(&format!("{SHARD_REL_PREFIX}{index}.lean"));
        let rendered = render_shard(index, shard);
        if std::fs::read_to_string(&path).unwrap_or_default() != rendered {
            drifted.push(path);
        }
    }

    assert!(
        drifted.is_empty(),
        "generated Lean chunk-digest artifacts drifted: {drifted:?}"
    );
}
