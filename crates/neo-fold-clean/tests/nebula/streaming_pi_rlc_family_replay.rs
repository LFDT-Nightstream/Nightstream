//! Exact artifact and geometry checks for one phased PiRLC family replay.

#[path = "../gadgets/lean_artifact_support.rs"]
#[allow(dead_code)]
mod lean_artifact_support;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Range;
use std::path::{Path, PathBuf};

use lean_artifact_support::{lean_nat_list, sha256_hex};
use neo_fold_clean::engine::r1cs_circuit::{enforce_poseidon2_permutation, R1csBuilder};
use neo_fold_clean::frontends::nebula::f_prime::{
    NebulaFPrimePiRlcFamilyReplayArmKind, NebulaFPrimePiRlcFamilyReplaySynthesis,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const SCHEMA_VERSION: usize = 1;
const PROFILE_ID: &str = "nebula-f-prime-streaming-pi-rlc-family-replay-v1";
const SOURCE_COLUMNS: usize = 146_224;
const REPLAY_AUXILIARY_START: usize = SOURCE_COLUMNS + 16;
const INPUT_COLUMN_START: usize = 811;
const INPUT_FIELDS: usize = 810;
const OUTPUT_COLUMN_START: usize = 1_621;
const OUTPUT_FIELDS: usize = 54;
const POSEIDON2_ROWS: usize = 600;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct SparseRow {
    a: Vec<(usize, F)>,
    b: Vec<(usize, F)>,
    c: Vec<(usize, F)>,
}

#[derive(Clone, Debug)]
struct Poseidon2Call {
    rows: Range<usize>,
    inputs: [usize; 8],
    first_allocated: usize,
}

struct ArmArtifact {
    row_count: usize,
    column_count: usize,
    before_absorbed: usize,
    after_absorbed: usize,
    input_poseidon2_call_count: usize,
    output_poseidon2_call_count: usize,
    input_columns: Vec<usize>,
    output_columns: Vec<usize>,
    input_before_columns: [usize; 8],
    input_after_columns: [usize; 8],
    output_before_columns: [usize; 8],
    output_after_columns: [usize; 8],
    poseidon2_calls: Vec<Poseidon2Call>,
}

fn normalize_terms(terms: impl IntoIterator<Item = (usize, F)>) -> Vec<(usize, F)> {
    let mut totals = BTreeMap::<usize, F>::new();
    for (column, coefficient) in terms {
        *totals.entry(column).or_insert(F::ZERO) += coefficient;
    }
    totals
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect()
}

fn normalized_matrix(rows: usize, trips: &[(usize, usize, F)]) -> Vec<Vec<(usize, F)>> {
    let mut raw = vec![Vec::new(); rows];
    for &(row, column, coefficient) in trips {
        assert!(row < rows, "sparse triplet row is in range");
        raw[row].push((column, coefficient));
    }
    raw.into_iter().map(normalize_terms).collect()
}

fn normalized_rows(builder: &R1csBuilder) -> Vec<SparseRow> {
    let (a, b, c) = builder.sparse_triplets();
    let a = normalized_matrix(builder.rows(), a);
    let b = normalized_matrix(builder.rows(), b);
    let c = normalized_matrix(builder.rows(), c);
    (0..builder.rows())
        .map(|row| SparseRow {
            a: a[row].clone(),
            b: b[row].clone(),
            c: c[row].clone(),
        })
        .collect()
}

fn rename_terms(terms: &[(usize, F)], column_map: &impl Fn(usize) -> usize) -> Vec<(usize, F)> {
    normalize_terms(
        terms
            .iter()
            .map(|&(column, coefficient)| (column_map(column), coefficient)),
    )
}

fn rename_row(row: &SparseRow, column_map: &impl Fn(usize) -> usize) -> SparseRow {
    SparseRow {
        a: rename_terms(&row.a, column_map),
        b: rename_terms(&row.b, column_map),
        c: rename_terms(&row.c, column_map),
    }
}

fn poseidon2_template() -> Vec<SparseRow> {
    let mut builder = R1csBuilder::new();
    let inputs = std::array::from_fn(|lane| builder.alloc(F::from_usize(lane + 1)));
    let _ = enforce_poseidon2_permutation(&mut builder, &inputs);
    assert!(builder.is_satisfied());
    assert_eq!(builder.rows(), POSEIDON2_ROWS);
    assert_eq!(builder.cols(), 609);
    normalized_rows(&builder)
}

fn assert_renamed_poseidon2_calls(builder: &R1csBuilder, calls: &[Poseidon2Call]) {
    let actual = normalized_rows(builder);
    let template = poseidon2_template();
    for (index, call) in calls.iter().enumerate() {
        assert_eq!(call.rows.len(), POSEIDON2_ROWS);
        for (offset, expected) in template.iter().enumerate() {
            let expected = rename_row(expected, &|column| match column {
                0 => 0,
                1..=8 => call.inputs[column - 1],
                _ => call.first_allocated + column - 9,
            });
            assert_eq!(
                actual[call.rows.start + offset],
                expected,
                "Poseidon2 call {index}, row {offset}",
            );
        }
    }
}

fn build_arm(kind: NebulaFPrimePiRlcFamilyReplayArmKind) -> ArmArtifact {
    let synthesis = NebulaFPrimePiRlcFamilyReplaySynthesis::production(kind);
    let builder = synthesis.builder();
    let shape = synthesis.shape_audit();
    let calls = builder
        .poseidon2_permutation_audits()
        .into_iter()
        .enumerate()
        .map(|(index, audit)| {
            assert_eq!(audit.row_start, index * POSEIDON2_ROWS);
            assert_eq!(audit.row_end, audit.row_start + POSEIDON2_ROWS);
            assert_eq!(audit.allocated_col_count, POSEIDON2_ROWS);
            assert_eq!(
                audit.first_allocated_col,
                REPLAY_AUXILIARY_START + index * POSEIDON2_ROWS,
            );
            assert_eq!(
                audit.output_cols,
                std::array::from_fn(|lane| audit.first_allocated_col + 592 + lane),
            );
            Poseidon2Call {
                rows: audit.row_start..audit.row_end,
                inputs: audit.input_cols,
                first_allocated: audit.first_allocated_col,
            }
        })
        .collect::<Vec<_>>();

    assert_eq!(shape.source_columns, SOURCE_COLUMNS);
    assert_eq!(shape.rows, kind.rows());
    assert_eq!(shape.columns, kind.columns());
    assert_eq!(builder.rows(), shape.rows);
    assert_eq!(builder.cols(), shape.columns);
    assert_eq!(builder.first_unsatisfied_row(), None);
    assert_eq!(calls.len(), kind.poseidon2_calls());
    assert_eq!(
        synthesis.input_columns(),
        (INPUT_COLUMN_START..INPUT_COLUMN_START + INPUT_FIELDS).collect::<Vec<_>>(),
    );
    assert_eq!(
        synthesis.output_columns(),
        (OUTPUT_COLUMN_START..OUTPUT_COLUMN_START + OUTPUT_FIELDS).collect::<Vec<_>>(),
    );
    assert_eq!(
        synthesis.input_before_columns(),
        std::array::from_fn(|lane| SOURCE_COLUMNS + lane),
    );
    assert_eq!(
        synthesis.output_before_columns(),
        std::array::from_fn(|lane| SOURCE_COLUMNS + 8 + lane),
    );
    assert_renamed_poseidon2_calls(builder, &calls);

    ArmArtifact {
        row_count: shape.rows,
        column_count: shape.columns,
        before_absorbed: shape.before_absorbed,
        after_absorbed: shape.after_absorbed,
        input_poseidon2_call_count: shape.input_poseidon2_calls,
        output_poseidon2_call_count: shape.output_poseidon2_calls,
        input_columns: synthesis.input_columns().to_vec(),
        output_columns: synthesis.output_columns().to_vec(),
        input_before_columns: synthesis.input_before_columns(),
        input_after_columns: synthesis.input_after_columns(),
        output_before_columns: synthesis.output_before_columns(),
        output_after_columns: synthesis.output_after_columns(),
        poseidon2_calls: calls,
    }
}

fn render_calls(calls: &[Poseidon2Call]) -> String {
    let values = calls
        .iter()
        .map(|call| {
            format!(
                "{{ rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }}",
                call.rows.start,
                call.rows.end,
                lean_nat_list(call.inputs),
                call.first_allocated,
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", values.join(",\n      "))
}

fn render_arm(arm: &ArmArtifact) -> String {
    format!(
        "{{ rowCount := {}, columnCount := {},\n    \
         beforeAbsorbed := {}, afterAbsorbed := {},\n    \
         inputPoseidon2CallCount := {}, outputPoseidon2CallCount := {},\n    \
         inputColumns := {}, outputColumns := {},\n    \
         inputBeforeColumns := {}, inputAfterColumns := {},\n    \
         outputBeforeColumns := {}, outputAfterColumns := {},\n    \
         poseidon2Calls := {} }}",
        arm.row_count,
        arm.column_count,
        arm.before_absorbed,
        arm.after_absorbed,
        arm.input_poseidon2_call_count,
        arm.output_poseidon2_call_count,
        lean_nat_list(arm.input_columns.iter().copied()),
        lean_nat_list(arm.output_columns.iter().copied()),
        lean_nat_list(arm.input_before_columns),
        lean_nat_list(arm.input_after_columns),
        lean_nat_list(arm.output_before_columns),
        lean_nat_list(arm.output_after_columns),
        render_calls(&arm.poseidon2_calls),
    )
}

fn render_artifact() -> String {
    let even = build_arm(NebulaFPrimePiRlcFamilyReplayArmKind::Even);
    let odd = build_arm(NebulaFPrimePiRlcFamilyReplayArmKind::Odd);
    let mut payload = String::new();
    writeln!(payload, "def evenArm : RawArm :=\n  {}", render_arm(&even)).unwrap();
    writeln!(payload, "\ndef oddArm : RawArm :=\n  {}", render_arm(&odd)).unwrap();
    writeln!(
        payload,
        "\ndef rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := {SCHEMA_VERSION}, profileId := \"{PROFILE_ID}\",\n    \
            sourceColumns := {SOURCE_COLUMNS}, even := evenArm, odd := oddArm }}",
    )
    .unwrap();
    let hash = sha256_hex(&payload);
    let rendered = format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyReplaySchema\n\n\
         /-! Generated file: exact Rust metadata for both production PiRLC\n\
         family replay cursor shapes.\n\n\
         Owns: source-column placement and every emitted Poseidon2 call.\n\n\
         Does not own: PiRLC source rows, semantic family authority, or the\n\
         recursive lifecycle.\n\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         set_option maxRecDepth 524288\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplay.Artifact\n\n\
         def artifactSha256 : String := \"{hash}\"\n\n\
         {payload}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay\n",
    );
    assert!(
        rendered.lines().count() < 1_500,
        "generated Lean artifact must stay below 1,500 lines",
    );
    rendered
}

fn artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyReplay.lean",
    )
}

#[test]
fn production_pi_rlc_family_replay_artifact_is_current() {
    let path = artifact_path();
    let rendered = render_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected PiRLC family replay artifact");
        panic!(
            "PiRLC family replay Lean artifact drifted; inspect {}",
            expected.display(),
        );
    }
}

#[test]
fn production_pi_rlc_family_replay_shapes_are_exact() {
    let even = build_arm(NebulaFPrimePiRlcFamilyReplayArmKind::Even);
    let odd = build_arm(NebulaFPrimePiRlcFamilyReplayArmKind::Odd);
    assert_eq!((even.row_count, even.column_count), (129_000, 275_240));
    assert_eq!((odd.row_count, odd.column_count), (130_200, 276_440));
    assert_eq!(
        (even.input_poseidon2_call_count, even.output_poseidon2_call_count),
        (202, 13),
    );
    assert_eq!(
        (odd.input_poseidon2_call_count, odd.output_poseidon2_call_count),
        (203, 14),
    );
    assert_eq!((even.before_absorbed, even.after_absorbed), (0, 2));
    assert_eq!((odd.before_absorbed, odd.after_absorbed), (2, 0));
}

#[test]
#[ignore = "deliberately writes the reviewed generated Lean artifact"]
fn regenerate_production_pi_rlc_family_replay_artifact() {
    std::fs::write(artifact_path(), render_artifact()).expect("write PiRLC family replay artifact");
}
