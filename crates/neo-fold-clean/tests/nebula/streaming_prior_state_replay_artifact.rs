//! Exact compact Lean artifacts for both prior-state replay source arms.

use std::fmt::Write;

use neo_fold_clean::frontends::f_prime::gadget_native::audit_r1cs_gadget_native_source_manifest;
use neo_fold_clean::frontends::nebula::f_prime::{
    production_prior_state_replay_final_source_arm, production_prior_state_replay_full_source_arm,
    NebulaFPrimePriorStateReplayArmKind, NebulaFPrimePriorStateReplaySynthesis,
    PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMNS, PRIOR_STATE_REPLAY_FINAL_SOURCE_ROWS,
    PRIOR_STATE_REPLAY_FINAL_SOURCE_SHA256, PRIOR_STATE_REPLAY_FINAL_TARGET_BINDING_STATUS,
    PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMNS, PRIOR_STATE_REPLAY_FULL_SOURCE_ROWS, PRIOR_STATE_REPLAY_FULL_SOURCE_SHA256,
    PRIOR_STATE_REPLAY_LIFECYCLE_SCOPE, PRIOR_STATE_REPLAY_PROFILE_ID, PRIOR_STATE_REPLAY_SOURCE_HASH_SCHEMA,
    PRIOR_STATE_REPLAY_SOURCE_PUBLIC_COLUMNS,
};
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_math::F;
use p3_field::PrimeField64;
use sha2::{Digest, Sha256};

const GENERATED_REL_DIR: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated";
const MAIN_FILE: &str = "FPrimeFullHistoryStreamingPriorStateReplaySource.lean";
const RESIDUAL_ROWS_PER_SHARD: usize = 1_400;
const RESIDUAL_LIST_PART_SIZE: usize = 256;
const POSEIDON_LIST_PART_SIZE: usize = 64;
const BINDING_LIST_PART_SIZE: usize = 256;

#[derive(Clone)]
struct RecipeCall {
    row_start: usize,
    row_end: usize,
    input_columns: [usize; 8],
    first_allocated_column: usize,
}

#[derive(Clone)]
struct CanonicalCall {
    row_start: usize,
    row_end: usize,
    field_column: usize,
    bit_base: usize,
    high_flag_column: usize,
    inverse_column: usize,
}

#[derive(Clone)]
struct SourceRow {
    index: usize,
    a: Vec<(usize, u64)>,
    b: Vec<(usize, u64)>,
    c: Vec<(usize, u64)>,
}

#[derive(Clone)]
struct RangeData {
    start: usize,
    stop: usize,
}

#[derive(Clone)]
struct NamedRangeData {
    name: String,
    range: RangeData,
}

#[derive(Clone)]
struct StageData {
    path: String,
    rows: RangeData,
    columns: RangeData,
}

#[derive(Clone)]
struct ColumnBinding {
    source: usize,
    normalized: usize,
}

struct SemanticColumns {
    before_replay_state: Vec<ColumnBinding>,
    after_replay_state: Vec<ColumnBinding>,
    chunk: Vec<ColumnBinding>,
    target_digest: Vec<ColumnBinding>,
    before_local_state_digest: Vec<ColumnBinding>,
    after_local_state_digest: Vec<ColumnBinding>,
    before_program_cursor: ColumnBinding,
    after_program_cursor: ColumnBinding,
    after_x_out_bits: Vec<ColumnBinding>,
    before_x_out_bits: Vec<ColumnBinding>,
    before_program_cursor_bits: Vec<ColumnBinding>,
    after_program_cursor_bits: Vec<ColumnBinding>,
    before_x_out_preimage: Vec<ColumnBinding>,
    after_x_out_preimage: Vec<ColumnBinding>,
    before_boundary: Vec<ColumnBinding>,
    after_boundary: Vec<ColumnBinding>,
    before_accumulator: Vec<ColumnBinding>,
    after_accumulator: Vec<ColumnBinding>,
    delayed_nebula_payload: Vec<ColumnBinding>,
}

struct CompactArm {
    kind: NebulaFPrimePriorStateReplayArmKind,
    rows: usize,
    source_columns: usize,
    normalized_columns: usize,
    public_columns: usize,
    source_sha256: &'static str,
    public_bindings: Vec<ColumnBinding>,
    semantic_columns: SemanticColumns,
    stages: Vec<StageData>,
    row_families: Vec<NamedRangeData>,
    column_families: Vec<NamedRangeData>,
    poseidon_calls: Vec<RecipeCall>,
    canonical_calls: Vec<CanonicalCall>,
    residual_rows: Vec<SourceRow>,
}

struct RenderedArtifact {
    name: String,
    contents: String,
}

impl CompactArm {
    fn label(&self) -> &'static str {
        match self.kind {
            NebulaFPrimePriorStateReplayArmKind::Full => "full",
            NebulaFPrimePriorStateReplayArmKind::Final => "final",
        }
    }

    fn title(&self) -> &'static str {
        match self.kind {
            NebulaFPrimePriorStateReplayArmKind::Full => "Full",
            NebulaFPrimePriorStateReplayArmKind::Final => "Final",
        }
    }

    fn poseidon_stem(&self) -> String {
        format!(
            "FPrimeFullHistoryStreamingPriorStateReplay{}PoseidonCalls",
            self.title()
        )
    }

    fn canonical_stem(&self) -> String {
        format!(
            "FPrimeFullHistoryStreamingPriorStateReplay{}CanonicalCalls",
            self.title()
        )
    }

    fn residual_prefix(&self) -> String {
        format!("FPrimeFullHistoryStreamingPriorStateReplay{}ResidualRows", self.title())
    }
}

fn mark_recipe_rows(covered: &mut [bool], rows: std::ops::Range<usize>, recipe: &str) {
    assert!(rows.start < rows.end, "{recipe} row range must be nonempty");
    assert!(rows.end <= covered.len(), "{recipe} row range escapes the source");
    for row in rows {
        assert!(!covered[row], "{recipe} overlaps source row {row}");
        covered[row] = true;
    }
}

fn normalized_column(public_source_columns: &[usize], source_column: usize) -> usize {
    if source_column == 0 {
        return 0;
    }
    if let Some(position) = public_source_columns
        .iter()
        .position(|&column| column == source_column)
    {
        return position + 1;
    }
    let public_before = public_source_columns
        .iter()
        .filter(|&&column| column < source_column)
        .count();
    1 + public_source_columns.len() + (source_column - 1 - public_before)
}

fn bind_column(public_source_columns: &[usize], source: usize) -> ColumnBinding {
    ColumnBinding {
        source,
        normalized: normalized_column(public_source_columns, source),
    }
}

fn bind_columns(public_source_columns: &[usize], columns: impl IntoIterator<Item = usize>) -> Vec<ColumnBinding> {
    columns
        .into_iter()
        .map(|source| bind_column(public_source_columns, source))
        .collect()
}

fn source_terms(terms: &[(usize, F)]) -> Vec<(usize, u64)> {
    terms
        .iter()
        .map(|&(column, coefficient)| (column, coefficient.as_canonical_u64()))
        .collect()
}

fn source_rows_sha256(source: &SparseR1cs) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"nightstream-normalized-sparse-r1cs-csc-v1\0");
    hasher.update((source.n as u64).to_le_bytes());
    hasher.update((source.m as u64).to_le_bytes());
    hasher.update((source.m_in as u64).to_le_bytes());
    for (matrix_index, matrix) in [&source.a, &source.b, &source.c].into_iter().enumerate() {
        assert!(matrix.seeded_phi81_blocks().is_empty());
        assert!(matrix.geometric_runs().is_empty());
        let csc = matrix
            .sparse_component()
            .expect("prior-state replay source uses canonical CSC matrices");
        assert!(csc.is_canonical());
        hasher.update([matrix_index as u8]);
        hasher.update((csc.nrows as u64).to_le_bytes());
        hasher.update((csc.ncols as u64).to_le_bytes());
        hasher.update((csc.col_ptr.len() as u64).to_le_bytes());
        for &pointer in &csc.col_ptr {
            hasher.update(pointer.to_le_bytes());
        }
        hasher.update((csc.row_idx.len() as u64).to_le_bytes());
        for (&row, coefficient) in csc.row_idx.iter().zip(&csc.vals) {
            hasher.update(row.to_le_bytes());
            hasher.update(coefficient.as_canonical_u64().to_le_bytes());
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn build_compact_arm(kind: NebulaFPrimePriorStateReplayArmKind) -> CompactArm {
    let synthesis = match kind {
        NebulaFPrimePriorStateReplayArmKind::Full => NebulaFPrimePriorStateReplaySynthesis::production_full(),
        NebulaFPrimePriorStateReplayArmKind::Final => NebulaFPrimePriorStateReplaySynthesis::production_final(),
    };
    let builder = synthesis.builder_for_artifact();
    assert!(
        builder.is_satisfied(),
        "production prior-state replay source must be satisfied"
    );
    let snapshot = builder.snapshot();
    let trace = builder.encoding_trace();
    let public_source_columns = (0..synthesis.public_columns() - 1)
        .map(|index| {
            synthesis
                .public_output_column(index)
                .expect("prior-state replay public output index is complete")
        })
        .collect::<Vec<_>>();
    assert!(public_source_columns.iter().all(|&column| column != 0));
    let mut sorted_public_columns = public_source_columns.clone();
    sorted_public_columns.sort_unstable();
    sorted_public_columns.dedup();
    assert_eq!(sorted_public_columns.len(), public_source_columns.len());

    let manifest = audit_r1cs_gadget_native_source_manifest(&snapshot, trace, &public_source_columns)
        .expect("prior-state replay traced recipes exactly replay the source rows");
    assert_eq!(manifest.source_columns(), snapshot.cols());

    let mut covered = vec![false; snapshot.rows()];
    let poseidon_calls = trace
        .poseidon_permutations()
        .iter()
        .map(|permutation| {
            mark_recipe_rows(&mut covered, permutation.source_rows.clone(), "Poseidon2 permutation");
            assert_eq!(permutation.source_rows.end - permutation.source_rows.start, 600);
            assert_eq!(
                permutation.allocated_columns.end - permutation.allocated_columns.start,
                600
            );
            RecipeCall {
                row_start: permutation.source_rows.start,
                row_end: permutation.source_rows.end,
                input_columns: permutation.input_columns,
                first_allocated_column: permutation.allocated_columns.start,
            }
        })
        .collect::<Vec<_>>();
    let canonical_calls = trace
        .canonical_u64_decompositions()
        .iter()
        .map(|decomposition| {
            mark_recipe_rows(
                &mut covered,
                decomposition.source_rows.clone(),
                "canonical-u64 decomposition",
            );
            assert_eq!(decomposition.source_rows.end - decomposition.source_rows.start, 69);
            let bit_base = decomposition.bits[0].col();
            for (offset, bit) in decomposition.bits.iter().enumerate() {
                assert_eq!(bit.col(), bit_base + offset);
            }
            assert_eq!(decomposition.high_is_max.col(), bit_base + 64);
            assert_eq!(decomposition.inverse.col(), bit_base + 65);
            CanonicalCall {
                row_start: decomposition.source_rows.start,
                row_end: decomposition.source_rows.end,
                field_column: decomposition.field.col(),
                bit_base,
                high_flag_column: decomposition.high_is_max.col(),
                inverse_column: decomposition.inverse.col(),
            }
        })
        .collect::<Vec<_>>();
    let residual_rows = covered
        .iter()
        .enumerate()
        .filter_map(|(index, &is_recipe_row)| {
            (!is_recipe_row).then(|| SourceRow {
                index,
                a: source_terms(snapshot.a_row(index)),
                b: source_terms(snapshot.b_row(index)),
                c: source_terms(snapshot.c_row(index)),
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        poseidon_calls.len() * 600 + canonical_calls.len() * 69 + residual_rows.len(),
        snapshot.rows()
    );

    let lowered = match kind {
        NebulaFPrimePriorStateReplayArmKind::Full => {
            production_prior_state_replay_full_source_arm().expect("lower exact full source arm")
        }
        NebulaFPrimePriorStateReplayArmKind::Final => {
            production_prior_state_replay_final_source_arm().expect("lower exact final source arm")
        }
    };
    let (expected_rows, expected_columns, expected_sha256) = match kind {
        NebulaFPrimePriorStateReplayArmKind::Full => (
            PRIOR_STATE_REPLAY_FULL_SOURCE_ROWS,
            PRIOR_STATE_REPLAY_FULL_SOURCE_COLUMNS,
            PRIOR_STATE_REPLAY_FULL_SOURCE_SHA256,
        ),
        NebulaFPrimePriorStateReplayArmKind::Final => (
            PRIOR_STATE_REPLAY_FINAL_SOURCE_ROWS,
            PRIOR_STATE_REPLAY_FINAL_SOURCE_COLUMNS,
            PRIOR_STATE_REPLAY_FINAL_SOURCE_SHA256,
        ),
    };
    assert_eq!(snapshot.rows(), expected_rows);
    assert_eq!(snapshot.cols(), expected_columns);
    assert_eq!(lowered.n, expected_rows);
    assert_eq!(lowered.m, expected_columns);
    assert_eq!(lowered.m_in, PRIOR_STATE_REPLAY_SOURCE_PUBLIC_COLUMNS);
    assert_eq!(source_rows_sha256(&lowered), expected_sha256);

    let stages = lowered
        .physical_stage_ranges()
        .iter()
        .map(|stage| StageData {
            path: stage.path().to_owned(),
            rows: RangeData {
                start: stage.row_start(),
                stop: stage.row_end(),
            },
            columns: RangeData {
                start: stage.column_start(),
                stop: stage.column_end(),
            },
        })
        .collect();
    let row_families = lowered
        .row_family_ranges()
        .iter()
        .map(|family| NamedRangeData {
            name: family.name.to_owned(),
            range: RangeData {
                start: family.row_start,
                stop: family.row_end,
            },
        })
        .collect();
    let column_families = lowered
        .column_family_ranges()
        .iter()
        .map(|family| NamedRangeData {
            name: family.name.to_owned(),
            range: RangeData {
                start: family.column_start,
                stop: family.column_end,
            },
        })
        .collect();
    let public_bindings = bind_columns(&public_source_columns, public_source_columns.iter().copied());
    let target_digest = synthesis
        .target_digest_columns()
        .map(|columns| bind_columns(&public_source_columns, columns))
        .unwrap_or_default();
    let semantic_columns = SemanticColumns {
        before_replay_state: bind_columns(&public_source_columns, synthesis.before_state_columns()),
        after_replay_state: bind_columns(&public_source_columns, synthesis.after_state_columns()),
        chunk: bind_columns(&public_source_columns, synthesis.chunk_columns()),
        target_digest,
        before_local_state_digest: bind_columns(
            &public_source_columns,
            synthesis.before_phase_local_state_source_columns(),
        ),
        after_local_state_digest: bind_columns(
            &public_source_columns,
            synthesis.after_phase_local_state_source_columns(),
        ),
        before_program_cursor: bind_column(&public_source_columns, synthesis.before_program_cursor_column()),
        after_program_cursor: bind_column(&public_source_columns, synthesis.after_program_cursor_column()),
        after_x_out_bits: bind_columns(&public_source_columns, public_source_columns[0..256].iter().copied()),
        before_x_out_bits: bind_columns(&public_source_columns, public_source_columns[256..512].iter().copied()),
        before_program_cursor_bits: bind_columns(
            &public_source_columns,
            public_source_columns[512..576].iter().copied(),
        ),
        after_program_cursor_bits: bind_columns(
            &public_source_columns,
            public_source_columns[576..640].iter().copied(),
        ),
        before_x_out_preimage: bind_columns(&public_source_columns, synthesis.before_x_out_preimage_columns()),
        after_x_out_preimage: bind_columns(&public_source_columns, synthesis.after_x_out_preimage_columns()),
        before_boundary: bind_columns(&public_source_columns, synthesis.before_boundary_columns()),
        after_boundary: bind_columns(&public_source_columns, synthesis.after_boundary_columns()),
        before_accumulator: bind_columns(&public_source_columns, synthesis.before_accumulator_columns()),
        after_accumulator: bind_columns(&public_source_columns, synthesis.after_accumulator_columns()),
        delayed_nebula_payload: bind_columns(&public_source_columns, synthesis.phase_delayed_payload_columns()),
    };

    CompactArm {
        kind,
        rows: snapshot.rows(),
        source_columns: snapshot.cols(),
        normalized_columns: lowered.m,
        public_columns: lowered.m_in,
        source_sha256: expected_sha256,
        public_bindings,
        semantic_columns,
        stages,
        row_families,
        column_families,
        poseidon_calls,
        canonical_calls,
        residual_rows,
    }
}

fn lean_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('\"', "\\\""))
}

fn lean_nat_list(values: impl IntoIterator<Item = usize>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
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

fn lean_range(range: &RangeData) -> String {
    format!("{{ start := {}, stop := {} }}", range.start, range.stop)
}

fn lean_binding(binding: &ColumnBinding) -> String {
    format!(
        "{{ source := {}, normalized := {} }}",
        binding.source, binding.normalized
    )
}

fn lean_bindings(bindings: &[ColumnBinding]) -> String {
    format!(
        "[{}]",
        bindings
            .iter()
            .map(lean_binding)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_named_ranges(ranges: &[NamedRangeData]) -> String {
    format!(
        "[{}]",
        ranges
            .iter()
            .map(|range| format!(
                "{{ name := {}, range := {} }}",
                lean_string(&range.name),
                lean_range(&range.range)
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_stages(stages: &[StageData]) -> String {
    format!(
        "[{}]",
        stages
            .iter()
            .map(|stage| format!(
                "{{ path := {}, sourceRows := {}, normalizedPrivateColumns := {} }}",
                lean_string(&stage.path),
                lean_range(&stage.rows),
                lean_range(&stage.columns)
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn generated_header() -> &'static str {
    "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPriorStateReplaySourceSchema\n\n\
/-! GENERATED FILE. DO NOT EDIT. Exact compact Rust prior-state replay source data. -/\n\n\
set_option maxRecDepth 2048\n\n"
}

fn render_poseidon_calls(arm: &CompactArm) -> RenderedArtifact {
    let stem = arm.poseidon_stem();
    let mut out = String::from(generated_header());
    writeln!(
        out,
        "namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Generated.{stem}\n\n\
open Nightstream.Implementation.R1CS\n"
    )
    .expect("render Poseidon2 namespace");
    let parts = arm
        .poseidon_calls
        .chunks(POSEIDON_LIST_PART_SIZE)
        .collect::<Vec<_>>();
    for (part_index, part) in parts.iter().enumerate() {
        writeln!(out, "def callsPart{part_index} : List Poseidon2Call.Call :=\n[")
            .expect("render Poseidon2 part header");
        for (index, call) in part.iter().enumerate() {
            let separator = if index + 1 == part.len() { "" } else { "," };
            writeln!(
                out,
                "  {{ rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }}{}",
                call.row_start,
                call.row_end,
                lean_nat_list(call.input_columns),
                call.first_allocated_column,
                separator,
            )
            .expect("render Poseidon2 call");
        }
        out.push_str("]\n\n");
    }
    writeln!(
        out,
        "def calls : List Poseidon2Call.Call :=\n  {}\n",
        (0..parts.len())
            .map(|index| format!("callsPart{index}"))
            .collect::<Vec<_>>()
            .join(" ++ ")
    )
    .expect("render Poseidon2 parts");
    writeln!(
        out,
        "end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Generated.{stem}"
    )
    .expect("render Poseidon2 footer");
    RenderedArtifact {
        name: format!("{stem}.lean"),
        contents: out,
    }
}

fn render_canonical_calls(arm: &CompactArm) -> RenderedArtifact {
    let stem = arm.canonical_stem();
    let mut out = String::from(generated_header());
    writeln!(
        out,
        "namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Generated.{stem}\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact\n\n\
def calls : List CanonicalCall :=\n["
    )
    .expect("render canonical namespace");
    for (index, call) in arm.canonical_calls.iter().enumerate() {
        let separator = if index + 1 == arm.canonical_calls.len() {
            ""
        } else {
            ","
        };
        writeln!(
            out,
            "  {{ rowStart := {}, rowEnd := {}, fieldColumn := {}, bitBase := {}, highFlagColumn := {}, inverseColumn := {} }}{}",
            call.row_start,
            call.row_end,
            call.field_column,
            call.bit_base,
            call.high_flag_column,
            call.inverse_column,
            separator,
        )
        .expect("render canonical-u64 call");
    }
    writeln!(
        out,
        "]\n\nend Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Generated.{stem}"
    )
    .expect("render canonical footer");
    RenderedArtifact {
        name: format!("{stem}.lean"),
        contents: out,
    }
}

fn render_residual_shard(arm: &CompactArm, index: usize, rows: &[SourceRow]) -> RenderedArtifact {
    let stem = format!("{}{index}", arm.residual_prefix());
    let mut out = String::from(generated_header());
    writeln!(
        out,
        "namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Generated.{stem}\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact\n"
    )
    .expect("render residual namespace");
    let parts = rows.chunks(RESIDUAL_LIST_PART_SIZE).collect::<Vec<_>>();
    for (part_index, part) in parts.iter().enumerate() {
        writeln!(out, "\ndef rowsPart{part_index} : List IndexedRow :=\n[").expect("render residual part header");
        for (offset, row) in part.iter().enumerate() {
            let separator = if offset + 1 == part.len() { "" } else { "," };
            writeln!(
                out,
                "  {{ index := {}, row := ⟨{}, {}, {}⟩ }}{}",
                row.index,
                lean_terms(&row.a),
                lean_terms(&row.b),
                lean_terms(&row.c),
                separator,
            )
            .expect("render residual row");
        }
        out.push_str("]\n");
    }
    writeln!(
        out,
        "\ndef rows : List IndexedRow :=\n  {}\n\n\
end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Generated.{stem}",
        (0..parts.len())
            .map(|part_index| format!("rowsPart{part_index}"))
            .collect::<Vec<_>>()
            .join(" ++ ")
    )
    .expect("render residual footer");
    RenderedArtifact {
        name: format!("{stem}.lean"),
        contents: out,
    }
}

fn render_binding_definition(out: &mut String, name: &str, bindings: &[ColumnBinding]) {
    let parts = bindings.chunks(BINDING_LIST_PART_SIZE).collect::<Vec<_>>();
    if parts.is_empty() {
        writeln!(out, "def {name} : List ColumnBinding := []\n").expect("render empty binding list");
        return;
    }
    for (index, part) in parts.iter().enumerate() {
        writeln!(
            out,
            "def {name}Part{index} : List ColumnBinding :=\n  {}\n",
            lean_bindings(part)
        )
        .expect("render binding-list part");
    }
    writeln!(
        out,
        "def {name} : List ColumnBinding :=\n  {}\n",
        (0..parts.len())
            .map(|index| format!("{name}Part{index}"))
            .collect::<Vec<_>>()
            .join(" ++ ")
    )
    .expect("render binding-list aggregate");
}

fn render_semantic_column_definitions(out: &mut String, arm: &CompactArm) {
    let label = arm.label();
    let columns = &arm.semantic_columns;
    render_binding_definition(out, &format!("{label}BeforeReplayState"), &columns.before_replay_state);
    render_binding_definition(out, &format!("{label}AfterReplayState"), &columns.after_replay_state);
    render_binding_definition(out, &format!("{label}Chunk"), &columns.chunk);
    render_binding_definition(out, &format!("{label}TargetDigest"), &columns.target_digest);
    render_binding_definition(
        out,
        &format!("{label}BeforeLocalStateDigest"),
        &columns.before_local_state_digest,
    );
    render_binding_definition(
        out,
        &format!("{label}AfterLocalStateDigest"),
        &columns.after_local_state_digest,
    );
    render_binding_definition(out, &format!("{label}AfterXOutBits"), &columns.after_x_out_bits);
    render_binding_definition(out, &format!("{label}BeforeXOutBits"), &columns.before_x_out_bits);
    render_binding_definition(
        out,
        &format!("{label}BeforeProgramCursorBits"),
        &columns.before_program_cursor_bits,
    );
    render_binding_definition(
        out,
        &format!("{label}AfterProgramCursorBits"),
        &columns.after_program_cursor_bits,
    );
    render_binding_definition(
        out,
        &format!("{label}BeforeXOutPreimage"),
        &columns.before_x_out_preimage,
    );
    render_binding_definition(out, &format!("{label}AfterXOutPreimage"), &columns.after_x_out_preimage);
    render_binding_definition(out, &format!("{label}BeforeBoundary"), &columns.before_boundary);
    render_binding_definition(out, &format!("{label}AfterBoundary"), &columns.after_boundary);
    render_binding_definition(out, &format!("{label}BeforeAccumulator"), &columns.before_accumulator);
    render_binding_definition(out, &format!("{label}AfterAccumulator"), &columns.after_accumulator);
    render_binding_definition(
        out,
        &format!("{label}DelayedNebulaPayload"),
        &columns.delayed_nebula_payload,
    );
}

fn semantic_columns_reference(arm: &CompactArm) -> String {
    let label = arm.label();
    let columns = &arm.semantic_columns;
    format!(
        "{{ beforeReplayState := {label}BeforeReplayState, afterReplayState := {label}AfterReplayState, chunk := {label}Chunk, targetDigest := {label}TargetDigest, beforeLocalStateDigest := {label}BeforeLocalStateDigest, afterLocalStateDigest := {label}AfterLocalStateDigest, beforeProgramCursor := {}, afterProgramCursor := {}, afterXOutBits := {label}AfterXOutBits, beforeXOutBits := {label}BeforeXOutBits, beforeProgramCursorBits := {label}BeforeProgramCursorBits, afterProgramCursorBits := {label}AfterProgramCursorBits, beforeXOutPreimage := {label}BeforeXOutPreimage, afterXOutPreimage := {label}AfterXOutPreimage, beforeBoundary := {label}BeforeBoundary, afterBoundary := {label}AfterBoundary, beforeAccumulator := {label}BeforeAccumulator, afterAccumulator := {label}AfterAccumulator, delayedNebulaPayload := {label}DelayedNebulaPayload }}",
        lean_binding(&columns.before_program_cursor),
        lean_binding(&columns.after_program_cursor),
    )
}

fn render_arm_definition(out: &mut String, arm: &CompactArm, residual_shards: usize) {
    let label = arm.label();
    let title = arm.title();
    let residual_prefix = arm.residual_prefix();
    render_binding_definition(out, &format!("{label}PublicColumns"), &arm.public_bindings);
    render_semantic_column_definitions(out, arm);
    writeln!(
        out,
        "def {label}ResidualRows : List IndexedRow :=\n  {}\n",
        (0..residual_shards)
            .map(|index| format!("{residual_prefix}{index}.rows"))
            .collect::<Vec<_>>()
            .join(" ++ ")
    )
    .expect("render residual aggregate");
    writeln!(out, "def {label}Artifact : RawArm :=\n{{").expect("render arm header");
    writeln!(out, "  schemaVersion := 1").expect("render arm");
    writeln!(out, "  profileId := {}", lean_string(PRIOR_STATE_REPLAY_PROFILE_ID)).expect("render arm");
    writeln!(out, "  branchScope := \"recursive\"").expect("render arm");
    writeln!(
        out,
        "  lifecycleScope := {}",
        lean_string(PRIOR_STATE_REPLAY_LIFECYCLE_SCOPE)
    )
    .expect("render arm");
    writeln!(out, "  armKind := .{label}").expect("render arm");
    writeln!(out, "  sourcePath := \"crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_prior_state_replay_relation.rs\"").expect("render arm");
    writeln!(
        out,
        "  sourceHashSchema := {}",
        lean_string(PRIOR_STATE_REPLAY_SOURCE_HASH_SCHEMA)
    )
    .expect("render arm");
    writeln!(
        out,
        "  sourceArtifactIdentity := {}",
        lean_string(&format!("sha256:{}", arm.source_sha256))
    )
    .expect("render arm");
    let target_status = if arm.kind == NebulaFPrimePriorStateReplayArmKind::Final {
        PRIOR_STATE_REPLAY_FINAL_TARGET_BINDING_STATUS
    } else {
        "not applicable: the full arm has no final target digest"
    };
    writeln!(out, "  finalTargetBindingStatus := {}", lean_string(target_status)).expect("render arm");
    writeln!(out, "  sourceRowCount := {}", arm.rows).expect("render arm");
    writeln!(out, "  sourceColumnCount := {}", arm.source_columns).expect("render arm");
    writeln!(out, "  normalizedColumnCount := {}", arm.normalized_columns).expect("render arm");
    writeln!(out, "  publicColumnCount := {}", arm.public_columns).expect("render arm");
    writeln!(out, "  columnLayout := {{ constantOne := {{ source := 0, normalized := 0 }}, publicColumns := {label}PublicColumns, normalizedPrivateStart := {} }}", arm.public_columns).expect("render arm");
    writeln!(out, "  semanticColumns := {}", semantic_columns_reference(arm)).expect("render arm");
    writeln!(out, "  physicalStages := {}", render_stages(&arm.stages)).expect("render arm");
    writeln!(out, "  rowFamilies := {}", render_named_ranges(&arm.row_families)).expect("render arm");
    writeln!(out, "  columnFamilies := {}", render_named_ranges(&arm.column_families)).expect("render arm");
    writeln!(
        out,
        "  poseidon2Calls := FPrimeFullHistoryStreamingPriorStateReplay{title}PoseidonCalls.calls"
    )
    .expect("render arm");
    writeln!(
        out,
        "  canonicalU64Calls := FPrimeFullHistoryStreamingPriorStateReplay{title}CanonicalCalls.calls"
    )
    .expect("render arm");
    writeln!(out, "  residualRows := {label}ResidualRows\n}}\n").expect("render arm footer");
}

fn render_main(full: &CompactArm, final_arm: &CompactArm, full_shards: usize, final_shards: usize) -> RenderedArtifact {
    let mut out = String::new();
    out.push_str(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPriorStateReplaySourceSchema\n",
    );
    for arm in [full, final_arm] {
        writeln!(
            out,
            "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.{}",
            arm.poseidon_stem()
        )
        .expect("render main import");
        writeln!(
            out,
            "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.{}",
            arm.canonical_stem()
        )
        .expect("render main import");
        let shard_count = if arm.kind == NebulaFPrimePriorStateReplayArmKind::Full {
            full_shards
        } else {
            final_shards
        };
        for index in 0..shard_count {
            writeln!(
                out,
                "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.{}{index}",
                arm.residual_prefix()
            )
            .expect("render residual import");
        }
    }
    out.push_str("\n/-! GENERATED FILE. DO NOT EDIT. Compact exact Rust prior-state replay source artifacts. -/\n\n\
namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Generated.FPrimeFullHistoryStreamingPriorStateReplaySource\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact\n\n\
set_option maxRecDepth 2048\n\n");
    render_arm_definition(&mut out, full, full_shards);
    render_arm_definition(&mut out, final_arm, final_shards);
    out.push_str("end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Generated.FPrimeFullHistoryStreamingPriorStateReplaySource\n");
    RenderedArtifact {
        name: MAIN_FILE.to_owned(),
        contents: out,
    }
}

fn render_artifacts() -> Vec<RenderedArtifact> {
    let full = build_compact_arm(NebulaFPrimePriorStateReplayArmKind::Full);
    let final_arm = build_compact_arm(NebulaFPrimePriorStateReplayArmKind::Final);
    let mut artifacts = Vec::new();
    for arm in [&full, &final_arm] {
        artifacts.push(render_poseidon_calls(arm));
        artifacts.push(render_canonical_calls(arm));
        artifacts.extend(
            arm.residual_rows
                .chunks(RESIDUAL_ROWS_PER_SHARD)
                .enumerate()
                .map(|(index, rows)| render_residual_shard(arm, index, rows)),
        );
        let recipe_rows = arm.poseidon_calls.len() * 600 + arm.canonical_calls.len() * 69;
        assert_eq!(recipe_rows + arm.residual_rows.len(), arm.rows);
        println!(
            "prior-state {} compact source: rows={}, columns={}, Poseidon2 calls={}, canonical-u64 calls={}, recipe rows={}, residual rows={}",
            arm.label(),
            arm.rows,
            arm.source_columns,
            arm.poseidon_calls.len(),
            arm.canonical_calls.len(),
            recipe_rows,
            arm.residual_rows.len(),
        );
    }
    let full_shards = full.residual_rows.len().div_ceil(RESIDUAL_ROWS_PER_SHARD);
    let final_shards = final_arm
        .residual_rows
        .len()
        .div_ceil(RESIDUAL_ROWS_PER_SHARD);
    artifacts.push(render_main(&full, &final_arm, full_shards, final_shards));
    for artifact in &artifacts {
        assert!(
            artifact.contents.lines().count() <= 1_500,
            "generated {} exceeds the project source-file limit",
            artifact.name
        );
    }
    artifacts
}

fn artifact_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("{}{GENERATED_REL_DIR}/{name}", env!("CARGO_MANIFEST_DIR")))
}

fn compare_or_write_expected(artifact: &RenderedArtifact, drifted: &mut Vec<String>) {
    let path = artifact_path(&artifact.name);
    if std::fs::read_to_string(&path).ok().as_deref() != Some(artifact.contents.as_str()) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, &artifact.contents).expect("write expected prior-state replay artifact");
        drifted.push(expected.display().to_string());
    }
}

#[test]
fn production_prior_state_replay_compact_source_artifacts_match_committed_files() {
    let artifacts = render_artifacts();
    let mut drifted = Vec::new();
    for artifact in &artifacts {
        compare_or_write_expected(artifact, &mut drifted);
    }
    assert!(
        drifted.is_empty(),
        "compact prior-state replay source artifacts drifted; inspect and deliberately regenerate: {drifted:#?}"
    );
}

#[test]
#[ignore = "deliberately replaces reviewed generated prior-state replay source artifacts"]
fn regenerate_production_prior_state_replay_compact_source_artifacts() {
    for artifact in render_artifacts() {
        let path = artifact_path(&artifact.name);
        std::fs::write(&path, artifact.contents).expect("write reviewed prior-state replay artifact");
        let expected = path.with_extension("lean.expected");
        match std::fs::remove_file(expected) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("remove reviewed expected prior-state replay artifact: {error}"),
        }
    }
}
