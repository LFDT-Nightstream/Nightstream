use neo_fold_clean::engine::r1cs_circuit::{enforce_poseidon2_permutation, R1csBuilder};
use neo_fold_clean::frontends::f_prime::gadget_native::audit_r1cs_gadget_native_source_manifest;
use neo_fold_clean::frontends::nebula::f_prime::{
    production_streaming_prelude_source_arm, NebulaFPrimeStreamingPreludeSynthesis, NebulaFPrimeStreamingPublicLayout,
    STREAMING_PRELUDE_INITIAL_REPLAY_STATE_FAMILY, STREAMING_PRELUDE_INITIAL_REPLAY_STATE_ROWS_FAMILY,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};
use std::fmt::Write;

const PROFILE_ID: &str = "nightstream-goldilocks-b2-k16";
const GENERATED_REL_DIR: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated";
const MAIN_FILE: &str = "FPrimeFullHistoryStreamingPreludeSource.lean";
const COLLAPSED_DOMAIN_RECEIPT_FILE: &str = "FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt.lean";
const POSEIDON_FILE: &str = "FPrimeFullHistoryStreamingPreludePoseidonCalls.lean";
const CANONICAL_FILE: &str = "FPrimeFullHistoryStreamingPreludeCanonicalCalls.lean";
const RESIDUAL_PREFIX: &str = "FPrimeFullHistoryStreamingPreludeResidualRows";
// The 1,500-line source-file policy is authoritative. With this renderer,
// 1,400 data lines plus the fixed header and footer stay below that limit.
const RESIDUAL_ROWS_PER_SHARD: usize = 1_400;
// A 1,400-entry row literal exceeded maxRecDepth 2,048. This keeps each row
// literal below one quarter of that measured recursion budget.
const RESIDUAL_LIST_PART_SIZE: usize = 256;
// One Poseidon2 call contains an eight-column list. A 256-call part still
// exceeded depth 2,048; 64 calls keep that nested product at 512 entries.
const POSEIDON_LIST_PART_SIZE: usize = 64;

const COLLAPSED_DOMAIN_INPUT: [u64; 8] = [
    27_431_110_773_469_033,
    30_522_878_494_336_372,
    32_758_250_074_896_737,
    829_828_965,
    1_988_541_141_149_579_427,
    4_859_373_221_894_732_330,
    9_937_262_314_844_071_878,
    8_401_668_388_730_343_368,
];

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

struct CompactSource {
    rows: usize,
    source_columns: usize,
    normalized_columns: usize,
    public_columns: usize,
    public_bindings: Vec<ColumnBinding>,
    initial_replay_state: Vec<ColumnBinding>,
    before_local_state_digest: Vec<ColumnBinding>,
    after_local_state_digest: Vec<ColumnBinding>,
    before_program_cursor: ColumnBinding,
    after_program_cursor: ColumnBinding,
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

struct CollapsedDomainReceipt {
    output: [u64; 8],
    initial_states: Vec<[u64; 8]>,
    partial_states: Vec<[u64; 8]>,
    terminal_states: Vec<[u64; 8]>,
}

fn witness_state(builder: &R1csBuilder, start: usize) -> [u64; 8] {
    std::array::from_fn(|lane| builder.witness()[start + lane].as_canonical_u64())
}

fn build_collapsed_domain_receipt() -> CollapsedDomainReceipt {
    let mut builder = R1csBuilder::new();
    let inputs = COLLAPSED_DOMAIN_INPUT.map(|value| builder.alloc(F::from_u64(value)));
    let outputs = enforce_poseidon2_permutation(&mut builder, &inputs);
    assert_eq!(builder.rows(), 600);
    assert_eq!(builder.cols(), 609);
    assert!(builder.is_satisfied());

    let initial_states = [9, 49, 89, 129, 169]
        .into_iter()
        .map(|start| witness_state(&builder, start))
        .collect::<Vec<_>>();
    let partial_states = std::iter::once(witness_state(&builder, 169))
        .chain((0..22).map(|round| witness_state(&builder, 181 + 12 * round)))
        .collect::<Vec<_>>();
    let terminal_states = std::iter::once(witness_state(&builder, 433))
        .chain((0..4).map(|round| witness_state(&builder, 473 + 40 * round)))
        .collect::<Vec<_>>();
    let output = outputs.map(|column| builder.witness()[column.col()].as_canonical_u64());

    assert_eq!(initial_states.len(), 5);
    assert_eq!(partial_states.len(), 23);
    assert_eq!(terminal_states.len(), 5);
    assert_eq!(terminal_states.last(), Some(&output));

    CollapsedDomainReceipt {
        output,
        initial_states,
        partial_states,
        terminal_states,
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

fn source_terms(terms: &[(usize, F)]) -> Vec<(usize, u64)> {
    terms
        .iter()
        .map(|&(column, coefficient)| (column, coefficient.as_canonical_u64()))
        .collect()
}

fn build_compact_source() -> CompactSource {
    let synthesis = NebulaFPrimeStreamingPreludeSynthesis::production();
    let builder = synthesis.builder_for_test();
    assert!(builder.is_satisfied(), "production Prelude source must be satisfied");
    let snapshot = builder.snapshot();
    let trace = builder.encoding_trace();
    let public_source_columns = (0..synthesis.public_columns() - 1)
        .map(|index| {
            synthesis
                .public_output_column(index)
                .expect("Prelude public output index is complete")
        })
        .collect::<Vec<_>>();
    assert!(public_source_columns.iter().all(|&column| column != 0));
    let mut sorted_public_columns = public_source_columns.clone();
    sorted_public_columns.sort_unstable();
    sorted_public_columns.dedup();
    assert_eq!(sorted_public_columns.len(), public_source_columns.len());

    let manifest = audit_r1cs_gadget_native_source_manifest(&snapshot, trace, &public_source_columns)
        .expect("Prelude traced recipes exactly replay the source rows");
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

    let lowered = production_streaming_prelude_source_arm().expect("lower exact Prelude source arm");
    assert_eq!(lowered.n, snapshot.rows());
    assert_eq!(lowered.m, snapshot.cols());
    assert_eq!(lowered.m_in, 1 + public_source_columns.len());
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

    let public_bindings = public_source_columns
        .iter()
        .copied()
        .map(|source| bind_column(&public_source_columns, source))
        .collect();
    let initial_replay_state = synthesis
        .initial_replay_state_columns()
        .iter()
        .copied()
        .map(|source| bind_column(&public_source_columns, source))
        .collect();
    let before_local_state_digest = synthesis
        .before_local_state_digest_columns()
        .iter()
        .copied()
        .map(|source| bind_column(&public_source_columns, source))
        .collect();
    let after_local_state_digest = synthesis
        .after_local_state_digest_columns()
        .iter()
        .copied()
        .map(|source| bind_column(&public_source_columns, source))
        .collect();

    CompactSource {
        rows: snapshot.rows(),
        source_columns: snapshot.cols(),
        normalized_columns: lowered.m,
        public_columns: lowered.m_in,
        public_bindings,
        initial_replay_state,
        before_local_state_digest,
        after_local_state_digest,
        before_program_cursor: bind_column(&public_source_columns, synthesis.before_program_cursor_column()),
        after_program_cursor: bind_column(&public_source_columns, synthesis.after_program_cursor_column()),
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

fn lean_u64_list(values: impl IntoIterator<Item = u64>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_receipt_states(name: &str, states: &[[u64; 8]]) -> String {
    let rows = states
        .iter()
        .map(|state| format!("  {}", lean_u64_list(state.iter().copied())))
        .collect::<Vec<_>>()
        .join(",\n");
    format!("def {name} : List (List Nat) :=\n[\n{rows}\n]\n")
}

fn render_collapsed_domain_receipt() -> String {
    let receipt = build_collapsed_domain_receipt();
    let mut out = String::from(
        "import Nightstream.Implementation.R1CS.Core.Semantics\n\n\
/-! GENERATED FILE. DO NOT EDIT. Rust-emitted bounded Prelude domain-collapse receipt.\n\
This data is non-authoritative until the handwritten Lean leaf checks it. -/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt\n\n\
def schemaVersion : Nat := 1\n\n\
def artifactKind : String := \"nebula/f-prime/streaming-prelude-collapsed-domain-receipt\"\n\n\
def sourcePath : String :=\n\
  \"crates/neo-fold-clean/tests/nebula/streaming_prelude_relation.rs\"\n\n",
    );
    writeln!(
        out,
        "def inputValues : List Nat :=\n  {}\n",
        lean_u64_list(COLLAPSED_DOMAIN_INPUT)
    )
    .expect("render collapsed-domain input");
    writeln!(
        out,
        "def collapsedValues : List Nat :=\n  {}\n",
        lean_u64_list(receipt.output)
    )
    .expect("render collapsed-domain output");
    writeln!(
        out,
        "{}",
        render_receipt_states("initialStates", &receipt.initial_states)
    )
    .expect("render initial states");
    writeln!(
        out,
        "{}",
        render_receipt_states("partialStates", &receipt.partial_states)
    )
    .expect("render partial states");
    writeln!(
        out,
        "{}",
        render_receipt_states("terminalStates", &receipt.terminal_states)
    )
    .expect("render terminal states");
    out.push_str(
        "end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeCollapsedDomainReceipt\n",
    );
    out
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

fn generated_header() -> &'static str {
    "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeSourceSchema\n\n\
/-! GENERATED FILE. DO NOT EDIT. Exact compact Rust Prelude source data. -/\n\n\
set_option maxRecDepth 2048\n\n"
}

fn render_poseidon_calls(source: &CompactSource) -> String {
    let mut out = String::from(generated_header());
    out.push_str(
        "namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Generated.FPrimeFullHistoryStreamingPreludePoseidonCalls\n\n\
open Nightstream.Implementation.R1CS\n\n",
    );
    let parts = source
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
    out.push_str(
        "end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Generated.FPrimeFullHistoryStreamingPreludePoseidonCalls\n",
    );
    out
}

fn render_canonical_calls(source: &CompactSource) -> String {
    let mut out = String::from(generated_header());
    out.push_str(
        "namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Generated.FPrimeFullHistoryStreamingPreludeCanonicalCalls\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact\n\n\
def calls : List CanonicalCall :=\n[\n",
    );
    for (index, call) in source.canonical_calls.iter().enumerate() {
        let separator = if index + 1 == source.canonical_calls.len() {
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
    out.push_str(
        "]\n\nend Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Generated.FPrimeFullHistoryStreamingPreludeCanonicalCalls\n",
    );
    out
}

fn render_residual_shard(index: usize, rows: &[SourceRow]) -> String {
    let mut out = String::from(generated_header());
    writeln!(
        out,
        "namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Generated.{RESIDUAL_PREFIX}{index}\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact\n"
    )
    .expect("render residual header");
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
        "\ndef rows : List IndexedRow :=\n  {}\n\nend Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Generated.{RESIDUAL_PREFIX}{index}",
        (0..parts.len())
            .map(|part_index| format!("rowsPart{part_index}"))
            .collect::<Vec<_>>()
            .join(" ++ ")
    )
    .expect("render residual footer");
    out
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

fn render_main(source: &CompactSource, source_identity: &str, residual_shards: usize) -> String {
    let mut out = String::new();
    out.push_str("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeSourceSchema\n");
    writeln!(
        out,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.{POSEIDON_FILE_NO_EXTENSION}"
    )
    .expect("render main import");
    writeln!(
        out,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.{CANONICAL_FILE_NO_EXTENSION}"
    )
    .expect("render main import");
    for index in 0..residual_shards {
        writeln!(
            out,
            "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.{RESIDUAL_PREFIX}{index}"
        )
        .expect("render residual import");
    }
    out.push_str(
        "\n/-! GENERATED FILE. DO NOT EDIT. Compact exact Rust Prelude source artifact. -/\n\n\
namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Generated.FPrimeFullHistoryStreamingPreludeSource\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Artifact\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact\n\n\
set_option maxRecDepth 2048\n\n",
    );
    out.push_str("def residualRows : List IndexedRow :=\n  ");
    out.push_str(
        &(0..residual_shards)
            .map(|index| format!("{RESIDUAL_PREFIX}{index}.rows"))
            .collect::<Vec<_>>()
            .join(" ++ "),
    );
    out.push_str("\n\ndef artifact : RawArtifact :=\n{\n");
    writeln!(out, "  schemaVersion := 1").expect("render main");
    writeln!(out, "  profileId := {}", lean_string(PROFILE_ID)).expect("render main");
    writeln!(out, "  branchScope := \"base\"").expect("render main");
    writeln!(out, "  lifecycleScope := \"prelude\"").expect("render main");
    writeln!(
        out,
        "  sourcePath := \"crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_prelude_relation.rs\""
    )
    .expect("render main");
    writeln!(out, "  sourceArtifactIdentity := {}", lean_string(source_identity)).expect("render main");
    writeln!(out, "  sourceRowCount := {}", source.rows).expect("render main");
    writeln!(out, "  sourceColumnCount := {}", source.source_columns).expect("render main");
    writeln!(out, "  normalizedColumnCount := {}", source.normalized_columns).expect("render main");
    writeln!(out, "  publicColumnCount := {}", source.public_columns).expect("render main");
    writeln!(
        out,
        "  columnLayout := {{ constantOne := {{ source := 0, normalized := 0 }}, publicColumns := {}, normalizedPrivateStart := {} }}",
        lean_bindings(&source.public_bindings),
        source.public_columns,
    )
    .expect("render main");
    writeln!(
        out,
        "  semanticColumns := {{ initialReplayState := {}, beforeLocalStateDigest := {}, afterLocalStateDigest := {}, beforeProgramCursor := {}, afterProgramCursor := {} }}",
        lean_bindings(&source.initial_replay_state),
        lean_bindings(&source.before_local_state_digest),
        lean_bindings(&source.after_local_state_digest),
        lean_binding(&source.before_program_cursor),
        lean_binding(&source.after_program_cursor),
    )
    .expect("render main");
    writeln!(out, "  physicalStages := {}", render_stages(&source.stages)).expect("render main");
    writeln!(out, "  rowFamilies := {}", render_named_ranges(&source.row_families)).expect("render main");
    writeln!(
        out,
        "  columnFamilies := {}",
        render_named_ranges(&source.column_families)
    )
    .expect("render main");
    writeln!(
        out,
        "  poseidon2Calls := FPrimeFullHistoryStreamingPreludePoseidonCalls.calls"
    )
    .expect("render main");
    writeln!(
        out,
        "  canonicalU64Calls := FPrimeFullHistoryStreamingPreludeCanonicalCalls.calls"
    )
    .expect("render main");
    out.push_str(
        "  residualRows := residualRows\n}\n\nend Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Generated.FPrimeFullHistoryStreamingPreludeSource\n",
    );
    out
}

const POSEIDON_FILE_NO_EXTENSION: &str = "FPrimeFullHistoryStreamingPreludePoseidonCalls";
const CANONICAL_FILE_NO_EXTENSION: &str = "FPrimeFullHistoryStreamingPreludeCanonicalCalls";

fn sha256_hex(payload: &str) -> String {
    let digest = Sha256::digest(payload.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn render_artifacts() -> Vec<RenderedArtifact> {
    let source = build_compact_source();
    let poseidon = render_poseidon_calls(&source);
    let canonical = render_canonical_calls(&source);
    let residual = source
        .residual_rows
        .chunks(RESIDUAL_ROWS_PER_SHARD)
        .enumerate()
        .map(|(index, rows)| RenderedArtifact {
            name: format!("{RESIDUAL_PREFIX}{index}.lean"),
            contents: render_residual_shard(index, rows),
        })
        .collect::<Vec<_>>();
    let main_without_identity = render_main(&source, "", residual.len());
    let mut identity_payload = String::new();
    identity_payload.push_str(&poseidon);
    identity_payload.push_str(&canonical);
    for shard in &residual {
        identity_payload.push_str(&shard.contents);
    }
    identity_payload.push_str(&main_without_identity);
    let source_identity = format!("sha256:{}", sha256_hex(&identity_payload));
    let main = render_main(&source, &source_identity, residual.len());

    let mut artifacts = vec![
        RenderedArtifact {
            name: POSEIDON_FILE.to_owned(),
            contents: poseidon,
        },
        RenderedArtifact {
            name: CANONICAL_FILE.to_owned(),
            contents: canonical,
        },
    ];
    artifacts.extend(residual);
    artifacts.push(RenderedArtifact {
        name: MAIN_FILE.to_owned(),
        contents: main,
    });
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
        std::fs::write(&expected, &artifact.contents).expect("write expected Prelude source artifact");
        drifted.push(expected.display().to_string());
    }
}

#[test]
fn production_prelude_collapsed_domain_receipt_matches_committed_file() {
    let artifact = RenderedArtifact {
        name: COLLAPSED_DOMAIN_RECEIPT_FILE.to_owned(),
        contents: render_collapsed_domain_receipt(),
    };
    let mut drifted = Vec::new();
    compare_or_write_expected(&artifact, &mut drifted);
    assert!(
        drifted.is_empty(),
        "Prelude collapsed-domain receipt drifted; inspect and deliberately regenerate: {drifted:#?}"
    );
}

#[test]
fn production_prelude_owns_the_canonical_initial_replay_state() {
    let synthesis = NebulaFPrimeStreamingPreludeSynthesis::production();
    assert!(synthesis.builder_for_test().is_satisfied());
    assert_eq!(synthesis.initial_replay_state_columns().len(), 10);
    assert_eq!(synthesis.before_local_state_digest_columns().len(), 4);
    assert_eq!(synthesis.after_local_state_digest_columns().len(), 4);
    assert_eq!(
        synthesis.public_columns(),
        NebulaFPrimeStreamingPublicLayout::production().logical_columns()
    );

    let source = production_streaming_prelude_source_arm().expect("lower exact Prelude source arm");
    assert_eq!(
        source.m_in,
        NebulaFPrimeStreamingPublicLayout::production().logical_columns()
    );
    assert!(source
        .column_family_ranges()
        .iter()
        .any(|family| family.name == STREAMING_PRELUDE_INITIAL_REPLAY_STATE_FAMILY));
    assert!(source
        .row_family_ranges()
        .iter()
        .any(|family| family.name == STREAMING_PRELUDE_INITIAL_REPLAY_STATE_ROWS_FAMILY));
}

#[test]
fn production_prelude_rejects_initial_state_cursor_and_public_mutations() {
    let mut state_mutation = NebulaFPrimeStreamingPreludeSynthesis::production();
    let initial_state_column = state_mutation.initial_replay_state_columns()[0];
    state_mutation.tamper_witness_for_test(initial_state_column, F::ONE);
    assert!(!state_mutation.builder_for_test().is_satisfied());

    let mut cursor_mutation = NebulaFPrimeStreamingPreludeSynthesis::production();
    let after_cursor = cursor_mutation.after_program_cursor_column();
    cursor_mutation.tamper_witness_for_test(after_cursor, F::from_u64(2));
    assert!(!cursor_mutation.builder_for_test().is_satisfied());

    let mut public_mutation = NebulaFPrimeStreamingPreludeSynthesis::production();
    let public_column = public_mutation
        .public_output_column(0)
        .expect("Prelude has the after-XOut public image");
    let changed = if public_mutation.builder_for_test().witness()[public_column] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };
    public_mutation.tamper_witness_for_test(public_column, changed);
    assert!(!public_mutation.builder_for_test().is_satisfied());
}

#[test]
fn production_prelude_compact_source_coverage_is_exact() {
    let source = build_compact_source();
    let recipe_rows = source.poseidon_calls.len() * 600 + source.canonical_calls.len() * 69;
    assert_eq!(recipe_rows + source.residual_rows.len(), source.rows);
    println!(
        "Prelude compact source: rows={}, columns={}, Poseidon2 calls={}, canonical-u64 calls={}, recipe rows={}, residual rows={}",
        source.rows,
        source.source_columns,
        source.poseidon_calls.len(),
        source.canonical_calls.len(),
        recipe_rows,
        source.residual_rows.len(),
    );
}

#[test]
fn production_prelude_compact_source_artifacts_match_committed_files() {
    let artifacts = render_artifacts();
    let mut drifted = Vec::new();
    for artifact in &artifacts {
        compare_or_write_expected(artifact, &mut drifted);
    }
    assert!(
        drifted.is_empty(),
        "compact Prelude source artifacts drifted; inspect and deliberately regenerate: {drifted:#?}"
    );
}

#[test]
#[ignore = "deliberately replaces reviewed generated Prelude source artifacts"]
fn regenerate_production_prelude_compact_source_artifacts() {
    for artifact in render_artifacts() {
        let path = artifact_path(&artifact.name);
        std::fs::write(&path, artifact.contents).expect("write reviewed Prelude source artifact");
        let expected = path.with_extension("lean.expected");
        match std::fs::remove_file(expected) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("remove reviewed expected artifact: {error}"),
        }
    }
}

#[test]
#[ignore = "deliberately replaces the reviewed Prelude collapsed-domain receipt"]
fn regenerate_production_prelude_collapsed_domain_receipt() {
    let path = artifact_path(COLLAPSED_DOMAIN_RECEIPT_FILE);
    std::fs::write(&path, render_collapsed_domain_receipt()).expect("write reviewed Prelude collapsed-domain receipt");
    let expected = path.with_extension("lean.expected");
    match std::fs::remove_file(expected) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("remove reviewed expected Prelude collapsed-domain receipt: {error}"),
    }
}
