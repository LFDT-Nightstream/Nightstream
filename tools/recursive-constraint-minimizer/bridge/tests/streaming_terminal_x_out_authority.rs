//! Bounded cvc5 attacks for the exact terminal XOut authority families.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Range;
use std::path::{Path, PathBuf};

use neo_fold_clean::engine::r1cs_circuit::{enforce_poseidon2_permutation, R1csBuilder};
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{
    streaming_terminal_x_out_authority_audit, STREAMING_TERMINAL_R1CS_FAMILY_NAMES,
};
use neo_math::F;
use nightstream_constraint_exporter::{export_problem, ExportRequest};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use recursive_constraint_minimizer::{
    audit_complete_typed_candidate, row_is_satisfied, Conclusion, FieldModel, Scope, Selection, SolverConfig,
    SolverMode, SolverStatus, Term, TypedTarget, TypedTargetRow,
};
use sha2::{Digest, Sha256};

const LEAF_SCHEMA_VERSION: usize = 1;
const LEAF_PROFILE_ID: &str = "nightstream/goldilocks/streaming-terminal-x-out-context/v1";
const LEAF_ARTIFACT_PATH: &str = "../../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalXOutContext.lean";
const LEAF_SOURCE_IDENTITY: &str = "rust:streaming-terminal-x-out-context/v1";
const PHASE_LEAF_PROFILE_ID: &str = "nightstream/goldilocks/streaming-terminal-phase-semantic/v1";
const PHASE_LEAF_ARTIFACT_PATH: &str = "../../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.lean";
const PHASE_LEAF_SOURCE_IDENTITY: &str = "rust:streaming-terminal-phase-semantic/v1";
const NEBULA_LINK_LEAF_PROFILE_ID: &str = "nightstream/goldilocks/streaming-terminal-nebula-state-digest-link/v1";
const NEBULA_LINK_LEAF_ARTIFACT_PATH: &str =
    "../../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.lean";
const NEBULA_LINK_LEAF_SOURCE_IDENTITY: &str = "rust:streaming-terminal-nebula-state-digest-link/v1";
const PHASE_CONSTANT_FIELDS: usize = 11;
const PHASE_INPUT_FIELDS: usize = 19;
const POSEIDON2_ROWS: usize = 600;
const PHASE_FAMILY_ROWS: usize = 3_636;
const NEBULA_STATE_DIGEST_FAMILY_ROWS: usize = 19_353;

struct ContextLeafArtifact {
    source_rows_sha256: String,
    column_count: usize,
    x_out_columns: [usize; 32],
    vk_fs_source_columns: [usize; 4],
    pi_ccs_header_source_columns: [usize; 4],
    boundary_source_columns: [usize; 4],
    accumulator_source_columns: [usize; 4],
    domain_tag: u64,
    accepted_work_items: u64,
    nebula_marker: u64,
    baseline_changed_value: u64,
    mutated_changed_value: u64,
}

struct PhaseSemanticLeafArtifact {
    source_rows_sha256: String,
    row_count: usize,
    column_count: usize,
    source_row_start: usize,
    constant_values: [u64; PHASE_CONSTANT_FIELDS],
    constant_start_column: usize,
    local_columns: [usize; 4],
    payload_columns: [usize; 4],
    hash_output_columns: [usize; 4],
    x_out_semantic_columns: [usize; 4],
    baseline_digest_value: u64,
    equality_row_start: usize,
}

struct NebulaStateDigestLinkLeafArtifact {
    source_rows_sha256: String,
    row_count: usize,
    column_count: usize,
    source_row_start: usize,
    hash_output_columns: [usize; 4],
    x_out_state_columns: [usize; 4],
    baseline_digest_value: u64,
    equality_row_start: usize,
    selected_source_row: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExactRow {
    a: Vec<Term>,
    b: Vec<Term>,
    c: Vec<Term>,
}

fn normalized_terms(terms: impl IntoIterator<Item = (usize, F)>) -> Vec<Term> {
    let mut totals = BTreeMap::<usize, F>::new();
    for (column, coefficient) in terms {
        *totals.entry(column).or_insert(F::ZERO) += coefficient;
    }
    totals
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .map(|(column, coefficient)| Term {
            column,
            coefficient: coefficient.as_canonical_u64().to_string(),
        })
        .collect()
}

fn assert_linear_row(row: &recursive_constraint_minimizer::Row, output: usize, terms: &[(usize, F)]) {
    let expected_a = normalized_terms(
        std::iter::once((output, F::ONE)).chain(
            terms
                .iter()
                .map(|&(column, coefficient)| (column, -coefficient)),
        ),
    );
    assert_eq!(row.a, expected_a);
    assert_eq!(row.b, normalized_terms([(0, F::ONE)]));
    assert!(row.c.is_empty());
}

fn copied_source(row: &recursive_constraint_minimizer::Row, output: usize) -> usize {
    let source = row
        .a
        .iter()
        .find(|term| term.column != output)
        .expect("copy row source")
        .column;
    assert_linear_row(row, output, &[(source, F::ONE)]);
    source
}

fn exact_linear_row(output: usize, terms: &[(usize, F)]) -> ExactRow {
    ExactRow {
        a: normalized_terms(
            std::iter::once((output, F::ONE)).chain(
                terms
                    .iter()
                    .map(|&(column, coefficient)| (column, -coefficient)),
            ),
        ),
        b: normalized_terms([(0, F::ONE)]),
        c: Vec::new(),
    }
}

fn exact_exported_row(row: &recursive_constraint_minimizer::Row) -> ExactRow {
    ExactRow {
        a: row.a.clone(),
        b: row.b.clone(),
        c: row.c.clone(),
    }
}

fn exact_builder_rows(builder: &R1csBuilder) -> Vec<ExactRow> {
    let (a, b, c) = builder.sparse_triplets();
    let normalize_matrix = |trips: &[(usize, usize, F)]| {
        let mut rows = vec![Vec::new(); builder.rows()];
        for &(row, column, coefficient) in trips {
            rows[row].push((column, coefficient));
        }
        rows.into_iter().map(normalized_terms).collect::<Vec<_>>()
    };
    let a = normalize_matrix(a);
    let b = normalize_matrix(b);
    let c = normalize_matrix(c);
    (0..builder.rows())
        .map(|row| ExactRow {
            a: a[row].clone(),
            b: b[row].clone(),
            c: c[row].clone(),
        })
        .collect()
}

fn poseidon2_template() -> Vec<ExactRow> {
    let mut builder = R1csBuilder::new();
    let inputs = std::array::from_fn(|lane| builder.alloc(F::from_usize(lane + 1)));
    let _ = enforce_poseidon2_permutation(&mut builder, &inputs);
    assert_eq!(builder.rows(), POSEIDON2_ROWS);
    exact_builder_rows(&builder)
}

fn renamed_row(row: &ExactRow, column_map: &impl Fn(usize) -> usize) -> ExactRow {
    let rename = |terms: &[Term]| {
        normalized_terms(terms.iter().map(|term| {
            (
                column_map(term.column),
                F::from_u64(
                    term.coefficient
                        .parse()
                        .expect("canonical Goldilocks coefficient"),
                ),
            )
        }))
    };
    ExactRow {
        a: rename(&row.a),
        b: rename(&row.b),
        c: rename(&row.c),
    }
}

fn assert_poseidon2_rows(
    rows: &[ExactRow],
    row_range: Range<usize>,
    inputs: [usize; 8],
    first_allocated: usize,
    template: &[ExactRow],
) {
    assert_eq!(row_range.len(), POSEIDON2_ROWS);
    for (offset, source) in template.iter().enumerate() {
        let expected = renamed_row(source, &|column| match column {
            0 => 0,
            1..=8 => inputs[column - 1],
            _ => first_allocated + column - 9,
        });
        assert_eq!(rows[row_range.start + offset], expected);
    }
}

fn build_phase_semantic_leaf_artifact() -> PhaseSemanticLeafArtifact {
    let audit = streaming_terminal_x_out_authority_audit();
    let source = audit.source();
    let family = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[3];
    let mut complete_families = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[2..5]
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    complete_families.sort_unstable();
    let problem = export_problem(
        source,
        audit.row_families(),
        ExportRequest {
            profile: PHASE_LEAF_PROFILE_ID.to_owned(),
            scope: Scope::Lifecycle,
            public_input_count: 1,
            source_rows: (0..source.rows()).collect(),
            complete_families,
        },
    )
    .expect("exact terminal phase-semantic export");
    let selected = problem
        .rows
        .iter()
        .filter(|row| row.family == family)
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), PHASE_FAMILY_ROWS);
    assert_eq!(selected[0].source_index, 24);
    assert_eq!(selected[PHASE_FAMILY_ROWS - 1].source_index, 3_659);
    let rows = selected
        .iter()
        .map(|row| exact_exported_row(row))
        .collect::<Vec<_>>();

    let constant_columns = (0..PHASE_CONSTANT_FIELDS)
        .map(|row| {
            let output = rows[row]
                .a
                .iter()
                .find(|term| term.column != 0 && term.coefficient == "1")
                .expect("phase constant output")
                .column;
            assert_eq!(rows[row], exact_linear_row(output, &[(0, source.witness()[output])]),);
            output
        })
        .collect::<Vec<_>>();
    assert_eq!(
        constant_columns,
        (constant_columns[0]..constant_columns[0] + PHASE_CONSTANT_FIELDS).collect::<Vec<_>>(),
    );

    let zero_row = PHASE_CONSTANT_FIELDS;
    let zero_column = rows[zero_row]
        .a
        .iter()
        .find(|term| term.column != 0 && term.coefficient == "1")
        .expect("hash zero output")
        .column;
    assert_eq!(rows[zero_row], exact_linear_row(zero_column, &[]));

    let local_columns = [33, 34, 35, 36];
    let payload_columns = [37, 38, 39, 40];
    let input_columns = constant_columns
        .iter()
        .copied()
        .chain(local_columns)
        .chain(payload_columns)
        .collect::<Vec<_>>();
    assert_eq!(input_columns.len(), PHASE_INPUT_FIELDS);

    let template = poseidon2_template();
    let mut prior_outputs = [zero_column; 8];
    let mut row_start = zero_row + 1;
    let mut column_start = zero_column + 1;
    for chunk in input_columns.chunks(4) {
        for (lane, &input) in chunk.iter().enumerate() {
            assert_eq!(
                rows[row_start + lane],
                exact_linear_row(column_start + lane, &[(prior_outputs[lane], F::ONE), (input, F::ONE)]),
            );
        }
        let permutation_inputs = std::array::from_fn(|lane| {
            if lane < chunk.len() {
                column_start + lane
            } else {
                prior_outputs[lane]
            }
        });
        let first_allocated = column_start + chunk.len();
        assert_poseidon2_rows(
            &rows,
            row_start + chunk.len()..row_start + chunk.len() + POSEIDON2_ROWS,
            permutation_inputs,
            first_allocated,
            &template,
        );
        prior_outputs = std::array::from_fn(|lane| first_allocated + 592 + lane);
        row_start += chunk.len() + POSEIDON2_ROWS;
        column_start = first_allocated + POSEIDON2_ROWS;
    }

    assert_eq!(
        rows[row_start],
        exact_linear_row(column_start, &[(prior_outputs[0], F::ONE), (0, F::ONE)]),
    );
    let pad_inputs = std::array::from_fn(|lane| if lane == 0 { column_start } else { prior_outputs[lane] });
    let first_allocated = column_start + 1;
    assert_poseidon2_rows(
        &rows,
        row_start + 1..row_start + 1 + POSEIDON2_ROWS,
        pad_inputs,
        first_allocated,
        &template,
    );
    let digest_columns: [usize; 4] = std::array::from_fn(|lane| first_allocated + 592 + lane);
    row_start += 1 + POSEIDON2_ROWS;
    let x_out_semantic_columns = std::array::from_fn(|lane| audit.x_out_columns()[19 + lane]);
    let baseline_digest_value = source.witness()[digest_columns[0]].as_canonical_u64();
    assert_eq!(
        source.witness()[x_out_semantic_columns[0]].as_canonical_u64(),
        baseline_digest_value,
    );
    for lane in 0..4 {
        assert_eq!(
            rows[row_start + lane],
            exact_linear_row(x_out_semantic_columns[lane], &[(digest_columns[lane], F::ONE)]),
        );
    }
    assert_eq!(row_start + 4, PHASE_FAMILY_ROWS);

    PhaseSemanticLeafArtifact {
        source_rows_sha256: problem
            .source
            .artifact_digest
            .strip_prefix("sha256:")
            .expect("source artifact digest prefix")
            .to_owned(),
        row_count: selected.len(),
        column_count: problem.column_count,
        source_row_start: selected[0].source_index,
        constant_values: std::array::from_fn(|index| source.witness()[constant_columns[index]].as_canonical_u64()),
        constant_start_column: constant_columns[0],
        local_columns,
        payload_columns,
        hash_output_columns: digest_columns,
        x_out_semantic_columns,
        baseline_digest_value,
        equality_row_start: row_start,
    }
}

#[test]
fn terminal_x_out_phase_semantic_rows_match_structural_recipe() {
    let _ = build_phase_semantic_leaf_artifact();
}

fn build_nebula_state_digest_link_leaf_artifact() -> NebulaStateDigestLinkLeafArtifact {
    let audit = streaming_terminal_x_out_authority_audit();
    let source = audit.source();
    let family = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[4];
    let mut complete_families = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[2..5]
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    complete_families.sort_unstable();
    let problem = export_problem(
        source,
        audit.row_families(),
        ExportRequest {
            profile: NEBULA_LINK_LEAF_PROFILE_ID.to_owned(),
            scope: Scope::Lifecycle,
            public_input_count: 1,
            source_rows: (0..source.rows()).collect(),
            complete_families,
        },
    )
    .expect("exact terminal Nebula-state-digest export");
    let selected = problem
        .rows
        .iter()
        .filter(|row| row.family == family)
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), NEBULA_STATE_DIGEST_FAMILY_ROWS);
    assert_eq!(selected[0].source_index, 3_660);
    assert_eq!(selected[NEBULA_STATE_DIGEST_FAMILY_ROWS - 1].source_index, 23_012);

    let equality_row_start = selected.len() - 4;
    let x_out_state_columns = std::array::from_fn(|lane| audit.x_out_columns()[28 + lane]);
    let hash_output_columns =
        std::array::from_fn(|lane| copied_source(selected[equality_row_start + lane], x_out_state_columns[lane]));
    let selected_source_row = selected[equality_row_start].source_index;
    assert_eq!(selected_source_row, 23_009);
    let baseline_digest_value = source.witness()[hash_output_columns[0]].as_canonical_u64();
    assert_eq!(
        source.witness()[x_out_state_columns[0]].as_canonical_u64(),
        baseline_digest_value,
    );

    NebulaStateDigestLinkLeafArtifact {
        source_rows_sha256: problem
            .source
            .artifact_digest
            .strip_prefix("sha256:")
            .expect("source artifact digest prefix")
            .to_owned(),
        row_count: selected.len(),
        column_count: problem.column_count,
        source_row_start: selected[0].source_index,
        hash_output_columns,
        x_out_state_columns,
        baseline_digest_value,
        equality_row_start,
        selected_source_row,
    }
}

#[test]
fn terminal_x_out_nebula_state_digest_link_rows_match_structural_recipe() {
    let _ = build_nebula_state_digest_link_leaf_artifact();
}

fn build_context_leaf_artifact() -> ContextLeafArtifact {
    let audit = streaming_terminal_x_out_authority_audit();
    let source = audit.source();
    let family = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[2];
    let mut complete_families = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[2..5]
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    complete_families.sort_unstable();
    let problem = export_problem(
        source,
        audit.row_families(),
        ExportRequest {
            profile: LEAF_PROFILE_ID.to_owned(),
            scope: Scope::Lifecycle,
            public_input_count: 1,
            source_rows: (0..source.rows()).collect(),
            complete_families,
        },
    )
    .expect("exact terminal XOut leaf export");
    let context_rows = problem
        .rows
        .iter()
        .filter(|row| row.family == family)
        .collect::<Vec<_>>();
    assert_eq!(context_rows.len(), 24);
    assert_eq!(context_rows[0].source_index, 0);
    assert_eq!(context_rows[23].source_index, 23);
    let x_out_columns = audit.x_out_columns();
    let domain_tag = source.witness()[x_out_columns[0]].as_canonical_u64();
    let accepted_work_items = source.witness()[x_out_columns[9]].as_canonical_u64();
    let nebula_marker = source.witness()[x_out_columns[27]].as_canonical_u64();

    assert_linear_row(context_rows[0], x_out_columns[0], &[(0, F::from_u64(domain_tag))]);
    let vk_fs_source_columns =
        std::array::from_fn(|lane| copied_source(context_rows[1 + lane], x_out_columns[1 + lane]));
    let baseline_changed_value = source.witness()[vk_fs_source_columns[0]].as_canonical_u64();
    assert_eq!(
        source.witness()[x_out_columns[1]].as_canonical_u64(),
        baseline_changed_value,
    );
    let mutated_changed_value =
        (F::from_u64(baseline_changed_value) + F::ONE).as_canonical_u64();
    assert_ne!(baseline_changed_value, mutated_changed_value);
    let pi_ccs_header_source_columns =
        std::array::from_fn(|lane| copied_source(context_rows[5 + lane], x_out_columns[5 + lane]));
    assert_linear_row(
        context_rows[9],
        x_out_columns[9],
        &[(0, F::from_u64(accepted_work_items))],
    );
    assert_linear_row(context_rows[10], x_out_columns[10], &[]);
    assert_linear_row(
        context_rows[11],
        x_out_columns[11],
        &[(0, F::from_u64(accepted_work_items))],
    );
    assert_linear_row(context_rows[12], x_out_columns[12], &[]);
    assert_linear_row(context_rows[13], x_out_columns[13], &[(0, F::ONE)]);
    assert_linear_row(context_rows[14], x_out_columns[14], &[]);
    let boundary_source_columns =
        std::array::from_fn(|lane| copied_source(context_rows[15 + lane], x_out_columns[15 + lane]));
    let accumulator_source_columns =
        std::array::from_fn(|lane| copied_source(context_rows[19 + lane], x_out_columns[23 + lane]));
    assert_linear_row(context_rows[23], x_out_columns[27], &[(0, F::from_u64(nebula_marker))]);

    ContextLeafArtifact {
        source_rows_sha256: problem
            .source
            .artifact_digest
            .strip_prefix("sha256:")
            .expect("source artifact digest prefix")
            .to_owned(),
        column_count: problem.column_count,
        x_out_columns,
        vk_fs_source_columns,
        pi_ccs_header_source_columns,
        boundary_source_columns,
        accumulator_source_columns,
        domain_tag,
        accepted_work_items,
        nebula_marker,
        baseline_changed_value,
        mutated_changed_value,
    }
}

fn lean_nat_list(values: impl IntoIterator<Item = usize>) -> String {
    let values = values
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    format!("[{}]", values.join(", "))
}

fn sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn render_context_leaf_artifact() -> String {
    let artifact = build_context_leaf_artifact();
    let mut payload = String::new();
    writeln!(
        payload,
        "def rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := {LEAF_SCHEMA_VERSION}, profileId := \"{LEAF_PROFILE_ID}\",\n    \
            sourceIdentity := \"{LEAF_SOURCE_IDENTITY}\",\n    \
            sourceRowsSha256 := \"{}\", rowCount := 24, columnCount := {},\n    \
            domainTag := {}, acceptedWorkItems := {}, nebulaMarker := {},\n    \
            baselineChangedValue := {}, mutatedChangedValue := {},\n    \
            xOutColumns := {},\n    \
            vkFsSourceColumns := {}, piCcsHeaderSourceColumns := {},\n    \
            boundarySourceColumns := {}, accumulatorSourceColumns := {} }}",
        artifact.source_rows_sha256,
        artifact.column_count,
        artifact.domain_tag,
        artifact.accepted_work_items,
        artifact.nebula_marker,
        artifact.baseline_changed_value,
        artifact.mutated_changed_value,
        lean_nat_list(artifact.x_out_columns),
        lean_nat_list(artifact.vk_fs_source_columns),
        lean_nat_list(artifact.pi_ccs_header_source_columns),
        lean_nat_list(artifact.boundary_source_columns),
        lean_nat_list(artifact.accumulator_source_columns),
    )
    .unwrap();
    let artifact_hash = sha256_hex(&payload);
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutContextSchema\n\n\
         /-! Generated compact geometry for the exact Rust terminal XOut context family.\n\n\
         Rust compares all 24 source rows with the structural Lean recipe.\n\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutContext\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact\n\n\
         def artifactSha256 : String := \"{artifact_hash}\"\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutContext\n",
    )
}

fn context_leaf_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(LEAF_ARTIFACT_PATH)
}

fn render_phase_semantic_leaf_artifact() -> String {
    let artifact = build_phase_semantic_leaf_artifact();
    let mut payload = String::new();
    writeln!(
        payload,
        "def phaseConstantValues : List Nat := {}",
        lean_nat_list(artifact.constant_values.map(|value| value as usize)),
    )
    .unwrap();
    writeln!(
        payload,
        "\ndef rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := {LEAF_SCHEMA_VERSION}, profileId := \"{PHASE_LEAF_PROFILE_ID}\",\n    \
            sourceIdentity := \"{PHASE_LEAF_SOURCE_IDENTITY}\",\n    \
            sourceRowsSha256 := \"{}\", rowCount := {}, columnCount := {},\n    \
            sourceRowStart := {}, finalRowStart := {},\n    \
            constantValues := phaseConstantValues, constantStartColumn := {},\n    \
            localColumns := {}, payloadColumns := {},\n    \
            hashOutputColumns := {}, xOutSemanticColumns := {},\n    \
            baselineDigestValue := {}, equalityRowStart := {} }}",
        artifact.source_rows_sha256,
        artifact.row_count,
        artifact.column_count,
        artifact.source_row_start,
        artifact.source_row_start,
        artifact.constant_start_column,
        lean_nat_list(artifact.local_columns),
        lean_nat_list(artifact.payload_columns),
        lean_nat_list(artifact.hash_output_columns),
        lean_nat_list(artifact.x_out_semantic_columns),
        artifact.baseline_digest_value,
        artifact.equality_row_start,
    )
    .unwrap();
    let artifact_hash = sha256_hex(&payload);
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticSchema\n\n\
         /-! Generated compact geometry for the exact Rust terminal XOut phase-semantic family.\n\n\
         Rust compares all 3,636 source rows with the structural Lean recipe.\n\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact\n\n\
         def artifactSha256 : String := \"{artifact_hash}\"\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic\n",
    )
}

fn render_nebula_state_digest_link_leaf_artifact() -> String {
    let artifact = build_nebula_state_digest_link_leaf_artifact();
    let payload = format!(
        "def rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := {LEAF_SCHEMA_VERSION}, profileId := \"{NEBULA_LINK_LEAF_PROFILE_ID}\",\n    \
            sourceIdentity := \"{NEBULA_LINK_LEAF_SOURCE_IDENTITY}\",\n    \
            sourceRowsSha256 := \"{}\", rowCount := {}, columnCount := {},\n    \
            sourceRowStart := {}, finalRowStart := {},\n    \
            hashOutputColumns := {}, xOutStateColumns := {},\n    \
            baselineDigestValue := {}, equalityRowStart := {}, selectedSourceRow := {} }}",
        artifact.source_rows_sha256,
        artifact.row_count,
        artifact.column_count,
        artifact.source_row_start,
        artifact.source_row_start,
        lean_nat_list(artifact.hash_output_columns),
        lean_nat_list(artifact.x_out_state_columns),
        artifact.baseline_digest_value,
        artifact.equality_row_start,
        artifact.selected_source_row,
    );
    let artifact_hash = sha256_hex(&payload);
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestLinkSchema\n\n\
         /-! Generated link projection for the exact Rust terminal Nebula-state-digest family.\n\n\
         Rust checks the four final links and their ownership in the 19,353-row source family.\n\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact\n\n\
         def artifactSha256 : String := \"{artifact_hash}\"\n\n\
         {payload}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink\n",
    )
}

fn phase_semantic_leaf_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(PHASE_LEAF_ARTIFACT_PATH)
}

fn nebula_state_digest_link_leaf_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(NEBULA_LINK_LEAF_ARTIFACT_PATH)
}

#[test]
fn terminal_x_out_phase_semantic_leaf_artifact_is_current() {
    let path = phase_semantic_leaf_artifact_path();
    let rendered = render_phase_semantic_leaf_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected terminal XOut phase-semantic artifact");
        panic!(
            "terminal XOut phase-semantic Lean artifact drifted; inspect {}",
            expected.display()
        );
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_terminal_x_out_phase_semantic_leaf_artifact() {
    std::fs::write(
        phase_semantic_leaf_artifact_path(),
        render_phase_semantic_leaf_artifact(),
    )
    .expect("write generated terminal XOut phase-semantic artifact");
}

#[test]
fn terminal_x_out_nebula_state_digest_link_leaf_artifact_is_current() {
    let path = nebula_state_digest_link_leaf_artifact_path();
    let rendered = render_nebula_state_digest_link_leaf_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected terminal Nebula-state-digest link artifact");
        panic!(
            "terminal Nebula-state-digest link Lean artifact drifted; inspect {}",
            expected.display()
        );
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_terminal_x_out_nebula_state_digest_link_leaf_artifact() {
    std::fs::write(
        nebula_state_digest_link_leaf_artifact_path(),
        render_nebula_state_digest_link_leaf_artifact(),
    )
    .expect("write generated terminal Nebula-state-digest link artifact");
}

#[test]
fn terminal_x_out_context_leaf_artifact_is_current() {
    let path = context_leaf_artifact_path();
    let rendered = render_context_leaf_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected terminal XOut context artifact");
        panic!(
            "terminal XOut context Lean artifact drifted; inspect {}",
            expected.display()
        );
    }
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_terminal_x_out_context_leaf_artifact() {
    std::fs::write(context_leaf_artifact_path(), render_context_leaf_artifact())
        .expect("write generated terminal XOut context artifact");
}

#[test]
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
fn installed_cvc5_replays_terminal_x_out_context_counterexample() {
    audit_terminal_x_out_authority_family(2, 1);
}

#[test]
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
fn installed_cvc5_replays_terminal_phase_semantic_counterexample() {
    audit_terminal_x_out_authority_family(3, 19);
}

#[test]
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
fn installed_cvc5_replays_terminal_nebula_state_digest_counterexample() {
    audit_terminal_x_out_authority_family(4, 28);
}

fn audit_terminal_x_out_authority_family(family_index: usize, x_out_index: usize) {
    let audit = streaming_terminal_x_out_authority_audit();
    let source = audit.source();
    let family_names = audit
        .row_families()
        .iter()
        .map(|family| family.name.to_owned())
        .collect::<Vec<_>>();
    assert_eq!(family_names, STREAMING_TERMINAL_R1CS_FAMILY_NAMES[2..5]);
    let family = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[family_index];
    let lean_certificate = match family_index {
        2 => "FPrimeFullHistoryStreamingTerminalXOutContextNecessity.exact_removal_counterexample",
        3 => "FPrimeFullHistoryStreamingTerminalXOutPhaseSemanticNecessity.exact_removal_counterexample",
        4 => "FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLinkNecessity.exact_removal_counterexample",
        _ => unreachable!("terminal XOut authority audit supports families 2 through 4"),
    };
    let owned_source_runs = audit
        .row_families()
        .iter()
        .filter(|range| range.name == family)
        .map(|range| range.row_start..range.row_end)
        .collect::<Vec<_>>();
    assert!(!owned_source_runs.is_empty(), "selected family must own source rows");
    let mut complete_families = family_names;
    complete_families.sort_unstable();
    let problem = export_problem(
        source,
        audit.row_families(),
        ExportRequest {
            profile: "nightstream/goldilocks/streaming-terminal-x-out-authority/v1".to_owned(),
            scope: Scope::Lifecycle,
            public_input_count: 1,
            source_rows: (0..source.rows()).collect(),
            complete_families,
        },
    )
    .expect("complete exact terminal XOut authority export");
    assert!(owned_source_runs.iter().all(|run| {
        problem.rows[run.clone()]
            .iter()
            .all(|row| row.family == family)
    }));
    eprintln!(
        "streaming terminal XOut ownership: family={family} scope=lifecycle source_runs={owned_source_runs:?} final_audit_runs={owned_source_runs:?} row_map=identity",
    );
    let target = TypedTarget {
        id: "nightstream.streaming.terminal.x_out.authority".to_owned(),
        column_count: problem.column_count,
        rows: problem
            .rows
            .iter()
            .map(|row| TypedTargetRow {
                id: format!("target.{}", row.source_index),
                a: row.a.clone(),
                b: row.b.clone(),
                c: row.c.clone(),
            })
            .collect(),
    };

    let mut values = source
        .witness()
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect::<Vec<_>>();
    let changed_column = audit.x_out_columns()[x_out_index];
    let selected_row = problem
        .rows
        .iter()
        .find(|row| row.family == family && row.a.iter().any(|term| term.column == changed_column))
        .expect("selected row for changed XOut column");
    let source_column = copied_source(selected_row, changed_column);
    let baseline_changed_value = source.witness()[source_column].as_canonical_u64();
    assert_eq!(values[changed_column], baseline_changed_value);
    let mutated_changed_value =
        (F::from_u64(baseline_changed_value) + F::ONE).as_canonical_u64();
    values[changed_column] = mutated_changed_value;
    let candidate = FieldModel::from_canonical_values(values).expect("canonical terminal attack assignment");
    let violated_selected_rows = problem
        .rows
        .iter()
        .filter(|row| !row_is_satisfied(row, &candidate).expect("exact Rust row replay"))
        .map(|row| {
            assert_eq!(row.family, family, "candidate must satisfy every retained family");
            row.source_index
        })
        .collect::<Vec<_>>();
    assert!(!violated_selected_rows.is_empty());

    let report = match audit_complete_typed_candidate(
        &problem,
        &Selection::Family(family.to_owned()),
        &target,
        &candidate,
        &SolverConfig {
            executable: PathBuf::from("/Users/nijaar/.local/bin/cvc5"),
            mode: SolverMode::Split,
            timeout_ms: 30_000,
        },
    ) {
        Ok(report) => report,
        Err(error) => {
            eprintln!(
                "streaming terminal XOut audit: family={family} rows={} columns={} cvc5=Inconclusive replayed=0 violated_target=[] violated_selected={violated_selected_rows:?} decision=retain lean_certificate=missing error={error}",
                problem.rows.len(),
                problem.column_count,
            );
            return;
        }
    };
    eprintln!(
        "streaming terminal XOut audit: family={family} rows={} columns={} cvc5={:?} replayed={} violated_target={:?} violated_selected={violated_selected_rows:?} projection=({source_column}:{baseline_changed_value},{changed_column}:{mutated_changed_value}) decision=retain lean_certificate={lean_certificate}",
        problem.rows.len(),
        problem.column_count,
        report.solver_run.status,
        report.retained_rows_replayed.len(),
        report.violated_target_rows,
    );
    assert_eq!(report.solver_run.status, SolverStatus::Sat);
    assert_eq!(report.conclusion, Conclusion::CounterexampleCandidate);
    assert_eq!(
        report.retained_rows_replayed.len(),
        problem
            .rows
            .iter()
            .filter(|row| row.family != family)
            .count()
    );
    assert!(!report.violated_target_rows.is_empty());
    assert_eq!(report.model.expect("full cvc5 model"), candidate);
}
