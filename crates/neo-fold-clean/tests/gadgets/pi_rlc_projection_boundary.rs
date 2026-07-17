//! Exact boundary witness for production's one-point PiRLC projection check.

// Shared gadget-test support: each test binary uses a different subset.
#[allow(dead_code)]
mod checked_program_artifact_support;
#[allow(dead_code)]
mod lean_artifact_support;

use std::fs;
use std::path::{Path, PathBuf};

use checked_program_artifact_support::{lean_instructions, normalize_with_inputs};
use lean_artifact_support::lean_nat_list;
use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
use neo_fold_clean::engine::r1cs_circuit::ring_action::{
    enforce_beta_ladder, enforce_ring_action_projection_batch, projection_quotient, PROJECTION_QUOTIENT_LEN,
};
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_math::ring::D;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const MANIFEST_PATH: &str = "formal/nightstream-lean/assurance/pi-rlc-projection-boundary.json";
const LEAN_ARTIFACT_PATH: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/Projection/Generated/PiRLCProjectionArtifact.lean";
const BETA: u64 = 7;

struct ProjectionFixture {
    builder: R1csBuilder,
    rho: [F; D],
    input: [F; D],
    output: [F; D],
    rho_columns: [usize; D],
    input_columns: [usize; D],
    output_columns: [usize; D],
    quotient_columns: [usize; PROJECTION_QUOTIENT_LEN],
    beta_columns: [usize; 2],
    power_columns: Vec<[usize; 2]>,
}

#[derive(Clone, Copy)]
struct KColumns {
    c0: usize,
    c1: usize,
}

impl From<[usize; 2]> for KColumns {
    fn from(columns: [usize; 2]) -> Self {
        Self {
            c0: columns[0],
            c1: columns[1],
        }
    }
}

struct KMulLayout {
    left_c0: Vec<(usize, u64)>,
    left_c1: Vec<(usize, u64)>,
    right_c0: Vec<(usize, u64)>,
    right_c1: Vec<(usize, u64)>,
    sum_left: Vec<(usize, u64)>,
    sum_right: Vec<(usize, u64)>,
    start: usize,
    output: KColumns,
}

struct EvalLayout {
    coefficients: Vec<usize>,
    powers: Vec<KColumns>,
    products: Vec<KColumns>,
    output: KColumns,
}

struct ProjectionLayout {
    ladder: Vec<KMulLayout>,
    rho_eval: EvalLayout,
    input_eval: EvalLayout,
    pair_product: KMulLayout,
    output_eval: EvalLayout,
    quotient_eval: EvalLayout,
    quotient_phi_product: KMulLayout,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn source_hash(relative: &str) -> Value {
    let bytes = fs::read(repo_root().join(relative)).unwrap_or_else(|error| panic!("read {relative}: {error}"));
    json!({ "path": relative, "sha256": sha256_hex(&bytes) })
}

fn allocate<const N: usize>(builder: &mut R1csBuilder, values: [F; N]) -> [Var; N] {
    values.map(|value| builder.alloc(value))
}

fn build_projection(
    rho: [F; D],
    input: [F; D],
    output: [F; D],
    quotient: [F; PROJECTION_QUOTIENT_LEN],
) -> ProjectionFixture {
    let mut builder = R1csBuilder::new();
    let rho_wires = allocate(&mut builder, rho);
    let input_wires = allocate(&mut builder, input);
    let output_wires = allocate(&mut builder, output);
    let quotient_wires = allocate(&mut builder, quotient);

    // Production commits all operands and quotient advice before deriving beta.
    // This fixed-beta harness isolates the algebraic boundary after that step.
    let beta = KVar::new(builder.alloc(F::from_u64(BETA)), builder.alloc(F::ZERO));
    let powers = enforce_beta_ladder(&mut builder, beta, D);
    let pairs: [(&[Var; D], &[Var; D]); 1] = [(&rho_wires, &input_wires)];
    enforce_ring_action_projection_batch(&mut builder, &powers, &pairs, &output_wires, &quotient_wires);

    ProjectionFixture {
        builder,
        rho,
        input,
        output,
        rho_columns: rho_wires.map(Var::col),
        input_columns: input_wires.map(Var::col),
        output_columns: output_wires.map(Var::col),
        quotient_columns: quotient_wires.map(Var::col),
        beta_columns: [beta.c0.col(), beta.c1.col()],
        power_columns: powers
            .iter()
            .map(|power| [power.c0.col(), power.c1.col()])
            .collect(),
    }
}

fn var_terms(columns: KColumns) -> (Vec<(usize, u64)>, Vec<(usize, u64)>) {
    (vec![(columns.c0, 1)], vec![(columns.c1, 1)])
}

fn k_mul_layout(left: KColumns, right: KColumns, start: usize) -> KMulLayout {
    let (left_c0, left_c1) = var_terms(left);
    let (right_c0, right_c1) = var_terms(right);
    KMulLayout {
        sum_left: left_c0.iter().chain(&left_c1).copied().collect(),
        sum_right: right_c0.iter().chain(&right_c1).copied().collect(),
        left_c0,
        left_c1,
        right_c0,
        right_c1,
        start,
        output: KColumns {
            c0: start + 3,
            c1: start + 4,
        },
    }
}

fn eval_layout(coefficients: &[usize], powers: &[KColumns], start: usize) -> EvalLayout {
    assert!(!coefficients.is_empty(), "projection polynomials are nonempty");
    assert!(coefficients.len() <= powers.len(), "evaluation ladder width");
    let products = (0..coefficients.len() - 1)
        .map(|index| KColumns {
            c0: start + 2 * index,
            c1: start + 2 * index + 1,
        })
        .collect::<Vec<_>>();
    let output_start = start + 2 * (coefficients.len() - 1);
    EvalLayout {
        coefficients: coefficients.to_vec(),
        powers: powers[..coefficients.len()].to_vec(),
        products,
        output: KColumns {
            c0: output_start,
            c1: output_start + 1,
        },
    }
}

fn projection_layout(fixture: &ProjectionFixture) -> ProjectionLayout {
    let powers = fixture
        .power_columns
        .iter()
        .copied()
        .map(KColumns::from)
        .collect::<Vec<_>>();
    assert_eq!(powers.len(), D + 1);
    let beta = KColumns::from(fixture.beta_columns);
    let mut ladder = Vec::with_capacity(D);
    for index in 1..=D {
        let trace = k_mul_layout(powers[index - 1], beta, 220 + 5 * (index - 1));
        assert_eq!(trace.output.c0, powers[index].c0, "beta ladder c0 layout");
        assert_eq!(trace.output.c1, powers[index].c1, "beta ladder c1 layout");
        ladder.push(trace);
    }

    let rho_eval = eval_layout(&fixture.rho_columns, &powers, 490);
    let input_eval = eval_layout(&fixture.input_columns, &powers, 598);
    let pair_product = k_mul_layout(rho_eval.output, input_eval.output, 706);
    let output_eval = eval_layout(&fixture.output_columns, &powers, 711);
    let quotient_eval = eval_layout(&fixture.quotient_columns, &powers, 819);

    let phi = KMulLayout {
        left_c0: vec![(quotient_eval.output.c0, 1)],
        left_c1: vec![(quotient_eval.output.c1, 1)],
        right_c0: vec![(powers[D].c0, 1), (powers[D / 2].c0, 1), (0, 1)],
        right_c1: vec![(powers[D].c1, 1), (powers[D / 2].c1, 1)],
        sum_left: vec![(quotient_eval.output.c0, 1), (quotient_eval.output.c1, 1)],
        sum_right: vec![
            (powers[D].c0, 1),
            (powers[D / 2].c0, 1),
            (powers[D].c1, 1),
            (powers[D / 2].c1, 1),
            (0, 1),
        ],
        start: 925,
        output: KColumns { c0: 928, c1: 929 },
    };

    assert_eq!(fixture.builder.rows(), 714, "projection row layout changed");
    assert_eq!(fixture.builder.cols(), 930, "projection column layout changed");
    assert_eq!(rho_eval.output.c0, 596);
    assert_eq!(input_eval.output.c0, 704);
    assert_eq!(pair_product.output.c0, 709);
    assert_eq!(output_eval.output.c0, 817);
    assert_eq!(quotient_eval.output.c0, 923);

    ProjectionLayout {
        ladder,
        rho_eval,
        input_eval,
        pair_product,
        output_eval,
        quotient_eval,
        quotient_phi_product: phi,
    }
}

fn honest_fixture() -> ProjectionFixture {
    let mut rho = [F::ZERO; D];
    let mut input = [F::ZERO; D];
    rho[0] = F::from_u64(2);
    rho[1] = F::from_u64(3);
    input[0] = F::from_u64(5);
    input[1] = F::from_u64(11);
    let (output, quotient) = projection_quotient(&[(rho, input)]);
    build_projection(rho, input, output, quotient)
}

fn bad_root_fixture() -> ProjectionFixture {
    let rho = [F::ZERO; D];
    let input = [F::ZERO; D];
    let mut output = [F::ZERO; D];
    // E(X) = X - beta is nonzero coefficient-wise but E(beta) = 0.
    output[0] = -F::from_u64(BETA);
    output[1] = F::ONE;
    build_projection(rho, input, output, [F::ZERO; PROJECTION_QUOTIENT_LEN])
}

fn exact_mix(fixture: &ProjectionFixture) -> bool {
    projection_quotient(&[(fixture.rho, fixture.input)]).0 == fixture.output
}

fn row_hash(builder: &R1csBuilder) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"nightstream/pi-rlc-projection-boundary/v1");
    hasher.update(builder.rows().to_le_bytes());
    hasher.update(builder.cols().to_le_bytes());
    let (a, b, c) = builder.sparse_triplets();
    for (tag, trips) in [(b'A', a), (b'B', b), (b'C', c)] {
        for &(row, column, coefficient) in trips {
            hasher.update([tag]);
            hasher.update(row.to_le_bytes());
            hasher.update(column.to_le_bytes());
            hasher.update(coefficient.as_canonical_u64().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn witness_hash(builder: &R1csBuilder) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"nightstream/pi-rlc-projection-witness/v1");
    for value in builder.witness() {
        hasher.update(value.as_canonical_u64().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn nonzero_entries(builder: &R1csBuilder) -> usize {
    let (a, b, c) = builder.sparse_triplets();
    a.len() + b.len() + c.len()
}

fn lean_terms(terms: &[(usize, u64)]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|(column, coefficient)| format!("({column}, {coefficient})"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_k_columns(columns: KColumns) -> String {
    format!("⟨{}, {}⟩", columns.c0, columns.c1)
}

fn lean_k_mul(trace: &KMulLayout) -> String {
    format!(
        "⟨⟨{}, {}⟩, ⟨{}, {}⟩, {}, {}, {}, {}, {}, {}⟩",
        lean_terms(&trace.left_c0),
        lean_terms(&trace.left_c1),
        lean_terms(&trace.right_c0),
        lean_terms(&trace.right_c1),
        lean_terms(&trace.sum_left),
        lean_terms(&trace.sum_right),
        trace.start,
        trace.start + 1,
        trace.start + 2,
        lean_k_columns(trace.output),
    )
}

fn lean_eval(trace: &EvalLayout) -> String {
    format!(
        "⟨{}, [{}], [{}], {}⟩",
        lean_nat_list(trace.coefficients.iter().copied()),
        trace
            .powers
            .iter()
            .copied()
            .map(lean_k_columns)
            .collect::<Vec<_>>()
            .join(", "),
        trace
            .products
            .iter()
            .copied()
            .map(lean_k_columns)
            .collect::<Vec<_>>()
            .join(", "),
        lean_k_columns(trace.output),
    )
}

fn lean_k_mul_list(traces: &[KMulLayout]) -> String {
    format!(
        "[{}]",
        traces
            .iter()
            .map(lean_k_mul)
            .collect::<Vec<_>>()
            .join(",\n     ")
    )
}

fn render_lean_artifact(honest: &ProjectionFixture, bad: &ProjectionFixture) -> String {
    let layout = projection_layout(honest);
    let mut declared_inputs = vec![0];
    declared_inputs.extend(honest.rho_columns);
    declared_inputs.extend(honest.input_columns);
    declared_inputs.extend(honest.output_columns);
    declared_inputs.extend(honest.quotient_columns);
    declared_inputs.extend(honest.beta_columns);
    let program = normalize_with_inputs(&honest.builder, &declared_inputs);
    assert_eq!(program.input_columns, declared_inputs);
    assert_eq!(program.definition_count, 712);
    assert_eq!(program.check_count, 2);

    let powers = honest
        .power_columns
        .iter()
        .copied()
        .map(KColumns::from)
        .map(lean_k_columns)
        .collect::<Vec<_>>()
        .join(", ");
    let pair = format!(
        "⟨{}, {}, {}, {}, {}⟩",
        lean_nat_list(honest.rho_columns),
        lean_nat_list(honest.input_columns),
        lean_eval(&layout.rho_eval),
        lean_eval(&layout.input_eval),
        lean_k_mul(&layout.pair_product),
    );
    let projection_trace = format!(
        "⟨⟨{}, [{}], {}⟩,\n   [{}],\n   {}, {},\n   {},\n   {},\n   {},\n   106⟩",
        lean_k_columns(KColumns::from(honest.beta_columns)),
        powers,
        lean_k_mul_list(&layout.ladder),
        pair,
        lean_nat_list(honest.output_columns),
        lean_nat_list(honest.quotient_columns),
        lean_eval(&layout.output_eval),
        lean_eval(&layout.quotient_eval),
        lean_k_mul(&layout.quotient_phi_product),
    );
    let lean_witness = |name: &str, fixture: &ProjectionFixture| {
        format!(
            "def {name} : List Nat :=\n  [{}]\n",
            fixture
                .builder
                .witness()
                .iter()
                .map(|value| value.as_canonical_u64().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    format!(
        "import Nightstream.Implementation.R1CS.Core.ProjectionProgram\n\n\
         /-! Generated exact program and semantic layout for the production PiRLC projection primitive. -/\n\n\
         namespace Nightstream.Implementation.R1CS.PiRLCProjection\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\
         open Nightstream.Implementation.R1CS.ProjectionProgram\n\n\
         set_option maxRecDepth 262144\n\n\
         def artifactKind : String := \"r1cs/pi-rlc-projection-program\"\n\
         def sourceAnchor : String := \"enforce_ring_action_projection_batch\"\n\
         def rowSha256 : String := \"{}\"\n\
         def rowCount : Nat := {}\n\
         def colCount : Nat := {}\n\
         def definitionCount : Nat := {}\n\
         def checkCount : Nat := {}\n\n\
         def programInputColumns : List Nat := {}\n\
         def projectionTrace : ProjectionTrace :=\n  {}\n\n\
         def instructions : List Instruction :=\n  [{}]\n\n\
         def rows : List Row := CheckedProgram.rows instructions\n\n\
         {}\n\
         {}\n\
         theorem instructions_length : instructions.length = rowCount := by native_decide\n\
         theorem rows_length : rows.length = rowCount := by native_decide\n\
         theorem definitions_length : (definitions instructions).length = definitionCount := by native_decide\n\
         theorem checks_length : (checks instructions).length = checkCount := by native_decide\n\
         theorem definitions_canonical :\n\
             ∀ definition ∈ definitions instructions, definition.Canonical := by native_decide\n\
         theorem definitions_wellFormed :\n\
             WellFormed programInputColumns (definitions instructions) := by native_decide\n\
         theorem checks_reference :\n\
             ChecksReference\n\
               (knownAfter programInputColumns (definitions instructions)) instructions := by native_decide\n\
         theorem trace_definitions_are_exact_program_definitions :\n\
             ∀ definition ∈ projectionTrace.definitions,\n\
               definition ∈ definitions instructions := by native_decide\n\
         theorem trace_checks_are_exact_program_checks :\n\
             projectionTrace.checks = checks instructions := by native_decide\n\
         theorem honest_satisfies : Satisfies rows (assignmentOf honestWitness) := by native_decide\n\
         theorem badRoot_satisfies : Satisfies rows (assignmentOf badRootWitness) := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.PiRLCProjection\n",
        row_hash(&honest.builder),
        honest.builder.rows(),
        honest.builder.cols(),
        program.definition_count,
        program.check_count,
        lean_nat_list(program.input_columns.iter().copied()),
        projection_trace,
        lean_instructions(&program.instructions),
        lean_witness("honestWitness", honest),
        lean_witness("badRootWitness", bad),
    )
}

fn assert_generated_file(path: &Path, rendered: &str) {
    let committed = fs::read_to_string(path).unwrap_or_default();
    if committed != rendered {
        let expected = path.with_extension("lean.expected");
        fs::write(&expected, rendered).expect("write expected Lean projection artifact");
        panic!(
            "generated Lean projection artifact drifted; inspect {} and copy it to {}",
            expected.display(),
            path.display()
        );
    }
}

fn manifest(honest: &ProjectionFixture, bad: &ProjectionFixture) -> Value {
    json!({
        "schema": 1,
        "artifact_kind": "r1cs/pi-rlc-projection-boundary",
        "source_anchor": "enforce_ring_action_projection_batch",
        "profile": {
            "pairs": 1,
            "ring_degree": D,
            "error_degree_bound": 2 * D - 2,
            "beta_extension": "K",
            "fixed_beta_c0": BETA,
            "fixed_beta_c1": 0
        },
        "rows": honest.builder.rows(),
        "columns": honest.builder.cols(),
        "nonzero_entries": nonzero_entries(&honest.builder),
        "row_sha256": row_hash(&honest.builder),
        "honest": {
            "projected_accept": honest.builder.is_satisfied(),
            "exact_mix": exact_mix(honest),
            "witness_sha256": witness_hash(&honest.builder)
        },
        "bad_root": {
            "projected_accept": bad.builder.is_satisfied(),
            "exact_mix": exact_mix(bad),
            "error_polynomial_prefix": [F::ORDER_U64 - BETA, 1],
            "error_at_beta": 0,
            "witness_sha256": witness_hash(&bad.builder)
        },
        "source_hashes": [
            source_hash("crates/neo-fold-clean/src/engine/r1cs_circuit/ring_action.rs"),
            source_hash("crates/neo-fold-clean/src/paper/nifs/circuit/mod.rs"),
            source_hash("crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/consistency.rs"),
            source_hash("crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/fold_wires.rs"),
            source_hash("crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/mod.rs"),
            source_hash("crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/padding.rs"),
            source_hash("crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/binding.rs"),
            source_hash("crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/identities.rs"),
            source_hash("crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/mod.rs"),
            source_hash("crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/shared.rs"),
            source_hash("crates/neo-fold-clean/tests/gadgets/pi_rlc_projection_boundary.rs"),
            source_hash("formal/nightstream-lean/Nightstream/Implementation/R1CS/Core/ProjectionProgram.lean"),
            source_hash("formal/nightstream-lean/Nightstream/Implementation/R1CS/Core/ProjectionLengths.lean"),
            source_hash("formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/Projection/ProjectionSound.lean"),
            source_hash("formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/Projection/ProjectionBatchSound.lean"),
            source_hash("formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/Projection/PiRLCProjectionSound.lean"),
            source_hash("formal/nightstream-lean/Nightstream/SuperNeo/ProjectionCheck.lean"),
            source_hash("formal/nightstream-lean/Nightstream/Assurance/FPrimeRecursiveCircuit.lean")
        ]
    })
}

#[test]
fn production_projection_rows_expose_exact_or_bad_root_boundary() {
    let honest = honest_fixture();
    let bad = bad_root_fixture();

    assert!(honest.builder.is_satisfied(), "honest projection must satisfy");
    assert!(exact_mix(&honest), "honest projection must be coefficient-wise exact");
    assert!(
        bad.builder.is_satisfied(),
        "fixed-beta root collision must satisfy the one-point rows"
    );
    assert!(
        !exact_mix(&bad),
        "root-collision output must not equal the full ring-action mix"
    );
    assert_eq!(
        honest.builder.sparse_triplets(),
        bad.builder.sparse_triplets(),
        "honest and bad-root witnesses must exercise identical production rows"
    );

    let rendered = format!(
        "{}\n",
        serde_json::to_string_pretty(&manifest(&honest, &bad)).expect("render projection manifest")
    );
    let path = repo_root().join(MANIFEST_PATH);
    let committed = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}\nexpected manifest:\n{rendered}", path.display()));
    if committed != rendered {
        let expected = path.with_extension("json.expected");
        fs::write(&expected, &rendered).expect("write expected PiRLC projection manifest");
    }
    assert_eq!(
        committed, rendered,
        "PiRLC projection boundary drifted; reviewed output:\n{rendered}"
    );

    assert_generated_file(
        &repo_root().join(LEAN_ARTIFACT_PATH),
        &render_lean_artifact(&honest, &bad),
    );
}
