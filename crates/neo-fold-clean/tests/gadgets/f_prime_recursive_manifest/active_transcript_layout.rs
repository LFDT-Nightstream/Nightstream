//! Three-matrix diagnostic PiRLC transcript physical-layout artifact.
//!
//! Owns: exact source locations for the PiRLC transcript's constant pins,
//! Poseidon2 calls, emission order, state-column aliases, and digest-output
//! consumers in the satisfied fixed recursive relation.
//!
//! Does not own: transcript message meaning, cursor semantics, Poseidon2
//! correctness, sampler correctness, or permission to remove rows.
//!
//! Emits constraints: no.
//!
//! Authority boundary: stage labels and encoding traces are diagnostic
//! provenance. Generation independently partitions every selected source row
//! into one exact constant pin or one traced 600-row Poseidon2 call. The Lean
//! artifact remains artifact-checked layout, not protocol authority.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::ops::Range;

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::pi_rlc_challenge_stage;
use neo_fold_clean::engine::r1cs_circuit::{
    CanonicalU64TraceEntry, PoseidonPermutationTraceEntry, R1csEncodingTrace, R1csSnapshot,
};
use neo_fold_clean::paper::reductions::pi_ccs_circuit::stage as pi_ccs_stage;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{build_recursive_program, repo_root};

const LEAN_DATA_PATH: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcChallenge/Generated/TranscriptLayoutData.lean";
const PIN_COUNT: usize = 412;
const CALL_COUNT: usize = 137;
const CALL_ROWS: usize = 600;
const CALL_COLUMNS: usize = 600;
const CALL_OUTPUT_OFFSET: usize = 592;
const GROUP_COUNT: usize = 15;
const BLOCKS_PER_GROUP: usize = 4;
const LANES_PER_BLOCK: usize = 4;
const FIELD_OUTPUT_ALIAS_COUNT: usize = GROUP_COUNT * BLOCKS_PER_GROUP * LANES_PER_BLOCK;
const CANONICAL_U64_ROWS: usize = 69;
const BIND_INPUT_COUNT: usize = 4;
const BIND_FRAMING_COLUMN_COUNT: usize = 2;
const OWNED_RANGE_COUNT: usize = 1 + GROUP_COUNT + GROUP_COUNT * BLOCKS_PER_GROUP;
const OWNED_ROW_COUNT: usize = PIN_COUNT + CALL_COUNT * CALL_ROWS;

#[derive(Clone, Debug, PartialEq, Eq)]
struct OwnedRange {
    checkpoint_index: usize,
    row_start: usize,
    row_end: usize,
    emission_start: usize,
    emission_end: usize,
}

#[derive(Clone, Debug)]
struct SelectedRange {
    checkpoint_index: usize,
    label: &'static str,
    rows: Range<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ConstantPin {
    row: usize,
    column: usize,
    value: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompactCall {
    trace_index: usize,
    row_start: usize,
    row_end: usize,
    input_columns: [usize; 8],
    first_allocated_column: usize,
}

impl CompactCall {
    fn output_column(&self, lane: usize) -> usize {
        assert!(lane < 8, "Poseidon2 output lane");
        self.first_allocated_column + CALL_OUTPUT_OFFSET + lane
    }

    fn output_columns(&self) -> [usize; 8] {
        std::array::from_fn(|lane| self.output_column(lane))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmissionRef {
    Pin(usize),
    Call(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Boundary {
    state_columns: [usize; 8],
    cursor: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StateContinuity {
    from_call: usize,
    to_call: usize,
    lanes: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FieldOutputAlias {
    ordinal: usize,
    group_index: usize,
    block_index: usize,
    lane_index: usize,
    call_index: usize,
    output_lane: usize,
    field_column: usize,
    canonical_row_start: usize,
    canonical_row_end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Artifact {
    source_rows: usize,
    source_columns: usize,
    owned_row_count: usize,
    owned_ranges: Vec<OwnedRange>,
    constant_pins: Vec<ConstantPin>,
    calls: Vec<CompactCall>,
    emission_order: Vec<EmissionRef>,
    entry_producer_trace_index: usize,
    entry_boundary: Boundary,
    post_bind_boundary: Boundary,
    final_boundary: Boundary,
    entry_to_first_call_lanes: Vec<usize>,
    post_bind_to_first_rho_call_lanes: Vec<usize>,
    state_continuity: Vec<StateContinuity>,
    field_output_aliases: Vec<FieldOutputAlias>,
    bind_call_indices: Vec<usize>,
    first_rho_call_index: usize,
    bind_input_columns: Vec<usize>,
}

fn is_owned_stage(label: &str) -> bool {
    matches!(
        label,
        pi_rlc_challenge_stage::BIND_OUTPUTS_DIGEST
            | pi_rlc_challenge_stage::RHO_DOMAIN_SEPARATOR
            | pi_rlc_challenge_stage::TRANSCRIPT_DIGEST
    )
}

fn selected_ranges(trace: &R1csEncodingTrace) -> Vec<SelectedRange> {
    let ranges = trace
        .stages()
        .windows(2)
        .enumerate()
        .filter_map(|(checkpoint_index, pair)| {
            let start = &pair[0];
            let end = &pair[1];
            is_owned_stage(start.label).then(|| {
                assert!(start.row < end.row, "owned transcript stage must emit rows");
                SelectedRange {
                    checkpoint_index,
                    label: start.label,
                    rows: start.row..end.row,
                }
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(ranges.len(), OWNED_RANGE_COUNT, "owned transcript stage count");
    assert_eq!(
        ranges
            .iter()
            .filter(|range| range.label == pi_rlc_challenge_stage::BIND_OUTPUTS_DIGEST)
            .count(),
        1,
        "one bind-output stage"
    );
    assert_eq!(
        ranges
            .iter()
            .filter(|range| range.label == pi_rlc_challenge_stage::RHO_DOMAIN_SEPARATOR)
            .count(),
        GROUP_COUNT,
        "rho separator stage count"
    );
    assert_eq!(
        ranges
            .iter()
            .filter(|range| range.label == pi_rlc_challenge_stage::TRANSCRIPT_DIGEST)
            .count(),
        GROUP_COUNT * BLOCKS_PER_GROUP,
        "digest stage count"
    );
    assert_eq!(
        ranges.first().expect("owned transcript ranges").label,
        pi_rlc_challenge_stage::BIND_OUTPUTS_DIGEST,
        "bind-output stage is the physical entry"
    );
    for pair in ranges.windows(2) {
        assert!(pair[0].rows.end <= pair[1].rows.start, "owned ranges do not overlap");
    }
    ranges
}

fn exact_stage_range(trace: &R1csEncodingTrace, label: &str) -> Range<usize> {
    let matches = trace
        .stages()
        .windows(2)
        .filter(|pair| pair[0].label == label)
        .map(|pair| pair[0].row..pair[1].row)
        .collect::<Vec<_>>();
    let [rows] = matches.as_slice() else {
        panic!("expected one `{label}` stage, found {}", matches.len());
    };
    assert!(rows.start < rows.end, "`{label}` stage must emit rows");
    rows.clone()
}

fn pi_ccs_output_digest_columns(trace: &R1csEncodingTrace) -> [usize; 4] {
    let sis_rows = exact_stage_range(trace, pi_ccs_stage::OUTPUT_MESSAGE_SIS);
    let final_hashes = trace
        .poseidon_hashes()
        .iter()
        .filter(|hash| sis_rows.start <= hash.source_rows.start && hash.source_rows.end == sis_rows.end)
        .collect::<Vec<_>>();
    let [final_hash] = final_hashes.as_slice() else {
        panic!(
            "expected one final Poseidon2 hash in the PiCCS output-message SIS stage, found {}",
            final_hashes.len()
        );
    };
    final_hash.output_columns
}

fn constant_pin_at(source: &R1csSnapshot, row: usize) -> Option<ConstantPin> {
    if source.b_row(row) != [(0, F::ONE)] || !source.c_row(row).is_empty() {
        return None;
    }
    let outputs = source
        .a_row(row)
        .iter()
        .filter(|(column, coefficient)| *column != 0 && *coefficient == F::ONE)
        .map(|(column, _)| *column)
        .collect::<Vec<_>>();
    let [column] = outputs.as_slice() else {
        return None;
    };
    let value = source.witness()[*column];
    let expected = if value == F::ZERO {
        vec![(*column, F::ONE)]
    } else {
        vec![(0, -value), (*column, F::ONE)]
    };
    (source.a_row(row) == expected).then_some(ConstantPin {
        row,
        column: *column,
        value: value.as_canonical_u64(),
    })
}

fn compact_call(trace_index: usize, entry: &PoseidonPermutationTraceEntry) -> CompactCall {
    assert_eq!(entry.source_rows.len(), CALL_ROWS, "Poseidon2 source row ABI");
    assert_eq!(
        entry.allocated_columns.len(),
        CALL_COLUMNS,
        "Poseidon2 allocated-column ABI"
    );
    assert_eq!(
        entry.output_columns,
        std::array::from_fn(|lane| entry.allocated_columns.start + CALL_OUTPUT_OFFSET + lane),
        "Poseidon2 output-column ABI"
    );
    CompactCall {
        trace_index,
        row_start: entry.source_rows.start,
        row_end: entry.source_rows.end,
        input_columns: entry.input_columns,
        first_allocated_column: entry.allocated_columns.start,
    }
}

fn partition_owned_rows(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    selected: &[SelectedRange],
) -> (Vec<OwnedRange>, Vec<ConstantPin>, Vec<CompactCall>, Vec<EmissionRef>) {
    let calls_by_row = trace
        .poseidon_permutations()
        .iter()
        .enumerate()
        .map(|(index, call)| (call.source_rows.start, index))
        .collect::<HashMap<_, _>>();
    assert_eq!(
        calls_by_row.len(),
        trace.poseidon_permutations().len(),
        "Poseidon2 call starts are unique"
    );

    let mut ranges = Vec::with_capacity(selected.len());
    let mut pins = Vec::with_capacity(PIN_COUNT);
    let mut calls = Vec::with_capacity(CALL_COUNT);
    let mut emission_order = Vec::with_capacity(PIN_COUNT + CALL_COUNT);
    let mut expected_rows = BTreeSet::new();
    let mut owned_rows = BTreeSet::new();

    for selected_range in selected {
        for row in selected_range.rows.clone() {
            assert!(expected_rows.insert(row), "selected source ranges overlap at row {row}");
        }
        let emission_start = emission_order.len();
        let mut cursor = selected_range.rows.start;
        while cursor < selected_range.rows.end {
            if let Some(&trace_index) = calls_by_row.get(&cursor) {
                let entry = &trace.poseidon_permutations()[trace_index];
                assert!(
                    entry.source_rows.end <= selected_range.rows.end,
                    "Poseidon2 call crosses owned-stage boundary"
                );
                let local_index = calls.len();
                calls.push(compact_call(trace_index, entry));
                emission_order.push(EmissionRef::Call(local_index));
                for row in entry.source_rows.clone() {
                    assert!(owned_rows.insert(row), "duplicate Poseidon2 row ownership {row}");
                }
                cursor = entry.source_rows.end;
            } else {
                let pin = constant_pin_at(source, cursor)
                    .unwrap_or_else(|| panic!("owned transcript row {cursor} is neither a pin nor a Poseidon2 call"));
                let local_index = pins.len();
                pins.push(pin);
                emission_order.push(EmissionRef::Pin(local_index));
                assert!(
                    owned_rows.insert(cursor),
                    "duplicate constant-pin row ownership {cursor}"
                );
                cursor += 1;
            }
        }
        ranges.push(OwnedRange {
            checkpoint_index: selected_range.checkpoint_index,
            row_start: selected_range.rows.start,
            row_end: selected_range.rows.end,
            emission_start,
            emission_end: emission_order.len(),
        });
    }

    assert_eq!(pins.len(), PIN_COUNT, "constant-pin count");
    assert_eq!(calls.len(), CALL_COUNT, "Poseidon2 call count");
    assert_eq!(expected_rows, owned_rows, "exact selected-source-row coverage");
    assert_eq!(owned_rows.len(), OWNED_ROW_COUNT, "owned transcript source-row count");
    assert!(
        pins.iter().map(|pin| pin.row).collect::<HashSet<_>>().len() == pins.len(),
        "constant-pin rows are unique"
    );
    assert!(
        calls
            .windows(2)
            .all(|pair| pair[0].trace_index + 1 == pair[1].trace_index),
        "selected Poseidon2 calls form one physical trace interval"
    );
    (ranges, pins, calls, emission_order)
}

fn call_indices_in_range(calls: &[CompactCall], range: &SelectedRange) -> Vec<usize> {
    calls
        .iter()
        .enumerate()
        .filter(|(_, call)| range.rows.start <= call.row_start && call.row_end <= range.rows.end)
        .map(|(index, _)| index)
        .collect()
}

fn bind_input_columns(
    pins: &[ConstantPin],
    calls: &[CompactCall],
    entry_state: &[usize; 8],
    expected_bind_inputs: &[usize],
) -> Vec<usize> {
    let pin_columns = pins.iter().map(|pin| pin.column).collect::<HashSet<_>>();
    let allocated_columns = calls
        .iter()
        .flat_map(|call| call.first_allocated_column..call.first_allocated_column + CALL_COLUMNS)
        .collect::<HashSet<_>>();
    let entry_columns = entry_state.iter().copied().collect::<HashSet<_>>();
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    for column in calls.iter().flat_map(|call| call.input_columns) {
        if column != 0
            && !pin_columns.contains(&column)
            && !allocated_columns.contains(&column)
            && !entry_columns.contains(&column)
            && seen.insert(column)
        {
            found.push(column);
        }
    }
    assert_eq!(expected_bind_inputs.len(), BIND_INPUT_COUNT);
    assert_eq!(
        found.len(),
        BIND_INPUT_COUNT + BIND_FRAMING_COLUMN_COUNT,
        "four digest inputs plus two typed-framing columns"
    );
    assert!(
        expected_bind_inputs
            .iter()
            .all(|column| found.contains(column)),
        "every PiCCS output-digest column enters the typed bind stage"
    );
    expected_bind_inputs.to_vec()
}

fn matching_lanes(left: &[usize; 8], right: &[usize; 8]) -> Vec<usize> {
    (0..8).filter(|&lane| left[lane] == right[lane]).collect()
}

fn state_continuity(calls: &[CompactCall]) -> Vec<StateContinuity> {
    calls
        .windows(2)
        .enumerate()
        .map(|(from_call, pair)| {
            let lanes = matching_lanes(&pair[0].output_columns(), &pair[1].input_columns);
            assert!(!lanes.is_empty(), "adjacent transcript calls share state columns");
            StateContinuity {
                from_call,
                to_call: from_call + 1,
                lanes,
            }
        })
        .collect()
}

fn canonical_for_stage<'a>(canonical: &'a [CanonicalU64TraceEntry], row: usize) -> &'a CanonicalU64TraceEntry {
    let matches = canonical
        .iter()
        .filter(|entry| entry.source_rows.start == row)
        .collect::<Vec<_>>();
    let [entry] = matches.as_slice() else {
        panic!(
            "expected one canonical-u64 trace at lane stage row {row}, found {}",
            matches.len()
        );
    };
    entry
}

fn field_output_aliases(
    trace: &R1csEncodingTrace,
    calls: &[CompactCall],
    pins: &[ConstantPin],
) -> Vec<FieldOutputAlias> {
    let output_owners = calls
        .iter()
        .enumerate()
        .flat_map(|(call_index, call)| {
            (0..4).map(move |output_lane| (call.output_column(output_lane), (call_index, output_lane)))
        })
        .collect::<HashMap<_, _>>();
    let lane_stages = trace
        .stages()
        .iter()
        .filter(|stage| stage.label == pi_rlc_challenge_stage::LANE_BIT_DECOMPOSITION)
        .collect::<Vec<_>>();
    assert_eq!(lane_stages.len(), FIELD_OUTPUT_ALIAS_COUNT, "lane-stage count");

    let aliases = lane_stages
        .iter()
        .enumerate()
        .map(|(ordinal, stage)| {
            let canonical = canonical_for_stage(trace.canonical_u64_decompositions(), stage.row);
            assert_eq!(
                canonical.source_rows.len(),
                CANONICAL_U64_ROWS,
                "canonical-u64 source-row count"
            );
            let field_column = canonical.field.col();
            let &(call_index, output_lane) = output_owners
                .get(&field_column)
                .unwrap_or_else(|| panic!("canonical field column {field_column} is not a selected digest output"));
            assert!(
                calls[call_index].row_end <= stage.row,
                "digest-output call precedes its lane decomposition"
            );
            FieldOutputAlias {
                ordinal,
                group_index: ordinal / (BLOCKS_PER_GROUP * LANES_PER_BLOCK),
                block_index: ordinal / LANES_PER_BLOCK % BLOCKS_PER_GROUP,
                lane_index: ordinal % LANES_PER_BLOCK,
                call_index,
                output_lane,
                field_column,
                canonical_row_start: canonical.source_rows.start,
                canonical_row_end: canonical.source_rows.end,
            }
        })
        .collect::<Vec<_>>();

    assert_eq!(aliases.len(), FIELD_OUTPUT_ALIAS_COUNT, "field-output alias count");
    assert_eq!(
        aliases
            .iter()
            .map(|alias| alias.field_column)
            .collect::<HashSet<_>>()
            .len(),
        aliases.len(),
        "field-output columns are unique"
    );
    for block in aliases.chunks_exact(LANES_PER_BLOCK) {
        assert!(
            block
                .iter()
                .all(|alias| alias.call_index == block[0].call_index),
            "one digest call owns each four-lane block"
        );
        let call_end = calls[block[0].call_index].row_end;
        assert_eq!(
            block[0].canonical_row_start - call_end,
            2,
            "two typed challenge post-binding rows precede the block's first lane"
        );
        assert!(
            (call_end..block[0].canonical_row_start).all(|row| pins.iter().any(|pin| pin.row == row)),
            "typed challenge post-binding rows are constant pins"
        );
        assert!(
            block
                .iter()
                .enumerate()
                .all(|(lane, alias)| alias.output_lane == lane),
            "digest output lanes stay ordered"
        );
    }
    assert_eq!(
        aliases
            .iter()
            .map(|alias| alias.call_index)
            .collect::<HashSet<_>>()
            .len(),
        GROUP_COUNT * BLOCKS_PER_GROUP,
        "sixty digest-output calls"
    );
    aliases
}

fn extract() -> Artifact {
    let builder = build_recursive_program();
    assert!(builder.is_satisfied(), "fixed recursive source relation");
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    let selected = selected_ranges(trace);
    let (owned_ranges, constant_pins, calls, emission_order) = partition_owned_rows(&source, trace, &selected);

    // Other PiCCS digest gadgets can emit Poseidon2 calls after the transcript
    // cursor's last permutation. Recover the actual entry-state producer from
    // the unchanged capacity lanes of the first PiRLC call, not by taking the
    // globally preceding permutation.
    let first_call = calls.first().expect("selected Poseidon2 calls");
    let entry_producers = trace
        .poseidon_permutations()
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.source_rows.end <= selected[0].rows.start
                && candidate.output_columns[4..] == first_call.input_columns[4..]
        })
        .collect::<Vec<_>>();
    let [(entry_producer_trace_index, entry_producer)] = entry_producers.as_slice() else {
        panic!(
            "expected one prior Poseidon2 state matching the PiRLC entry capacity, found {}",
            entry_producers.len()
        );
    };
    let entry_producer_trace_index = *entry_producer_trace_index;
    let entry_state = entry_producer.output_columns;

    let bind_call_indices = call_indices_in_range(&calls, &selected[0]);
    assert_eq!(
        bind_call_indices,
        [0, 1, 2],
        "three typed bind-output calls at local indices zero through two"
    );
    let first_rho_range = selected
        .iter()
        .find(|range| range.label == pi_rlc_challenge_stage::TRANSCRIPT_DIGEST)
        .expect("first rho digest range");
    let first_rho_calls = call_indices_in_range(&calls, first_rho_range);
    let first_rho_call_index = *first_rho_calls.first().expect("rho-zero digest call");
    assert_eq!(first_rho_call_index, 3, "first rho call follows three typed bind calls");

    let expected_bind_inputs = pi_ccs_output_digest_columns(trace);
    let bind_input_columns = bind_input_columns(&constant_pins, &calls, &entry_state, &expected_bind_inputs);
    assert_eq!(
        bind_input_columns, expected_bind_inputs,
        "PiRLC bind inputs are the PiCCS output-message digest columns"
    );
    let bind_seen = bind_call_indices
        .iter()
        .flat_map(|&index| calls[index].input_columns)
        .filter(|column| bind_input_columns.contains(column))
        .collect::<HashSet<_>>();
    let trailing_bind_inputs = bind_input_columns
        .iter()
        .copied()
        .filter(|column| !bind_seen.contains(column))
        .collect::<Vec<_>>();
    assert_eq!(trailing_bind_inputs.len(), 1, "one post-bind buffered digest input");

    let last_bind_call = &calls[*bind_call_indices.last().expect("bind calls")];
    let mut post_bind_state = last_bind_call.output_columns();
    post_bind_state[..trailing_bind_inputs.len()].copy_from_slice(&trailing_bind_inputs);

    let entry_boundary = Boundary {
        state_columns: entry_state,
        cursor: 0,
    };
    let post_bind_boundary = Boundary {
        state_columns: post_bind_state,
        cursor: trailing_bind_inputs.len(),
    };
    let final_boundary = Boundary {
        state_columns: calls.last().expect("selected calls").output_columns(),
        cursor: 0,
    };
    let entry_to_first_call_lanes = matching_lanes(&entry_boundary.state_columns, &calls[0].input_columns);
    let post_bind_to_first_rho_call_lanes = matching_lanes(
        &post_bind_boundary.state_columns,
        &calls[first_rho_call_index].input_columns,
    );
    assert_eq!(entry_to_first_call_lanes, [4, 5, 6, 7], "entry capacity continuity");
    assert_eq!(
        post_bind_to_first_rho_call_lanes,
        [0, 4, 5, 6, 7],
        "mixed post-bind state continuity"
    );
    assert_eq!(post_bind_boundary.cursor, 1, "post-bind cursor metadata");
    let final_call_end = calls.last().expect("selected calls").row_end;
    let final_range_end = selected.last().expect("owned ranges").rows.end;
    assert_eq!(
        final_range_end - final_call_end,
        2,
        "two typed challenge post-binding pins close the final owned range"
    );
    assert!(
        (final_call_end..final_range_end).all(|row| constant_pins.iter().any(|pin| pin.row == row)),
        "the final typed challenge rows are constant pins"
    );

    let state_continuity = state_continuity(&calls);
    assert_eq!(state_continuity.len(), CALL_COUNT - 1, "adjacent-call continuity count");
    let field_output_aliases = field_output_aliases(trace, &calls, &constant_pins);

    Artifact {
        source_rows: source.rows(),
        source_columns: source.cols(),
        owned_row_count: owned_ranges
            .iter()
            .map(|range| range.row_end - range.row_start)
            .sum(),
        owned_ranges,
        constant_pins,
        calls,
        emission_order,
        entry_producer_trace_index,
        entry_boundary,
        post_bind_boundary,
        final_boundary,
        entry_to_first_call_lanes,
        post_bind_to_first_rho_call_lanes,
        state_continuity,
        field_output_aliases,
        bind_call_indices,
        first_rho_call_index,
        bind_input_columns,
    }
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

fn render(artifact: &Artifact) -> String {
    assert_eq!(artifact.owned_row_count, OWNED_ROW_COUNT, "owned row total");
    let mut out = String::from(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.TranscriptLayoutSchema\n\n\
/-! Generated by `active_pi_rlc_transcript_layout_matches_production_trace`; do not hand-edit.\n\n\
Owns: exact physical row/column locations for the active fixed recursive PiRLC\n\
transcript layout. Does not own message meaning, cursor semantics, Poseidon2\n\
correctness, sampler correctness, or authority for any digest.\n\n\
Assurance tier: artifact-checked layout after exact Rust source-row partitioning.\n\
The stage labels used for extraction are diagnostic provenance only.\n-/\n\n\
namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutData\n\n\
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutSchema\n\n",
    );

    out.push_str("def ownedRanges : List OwnedRange :=\n  [ ");
    for (index, range) in artifact.owned_ranges.iter().enumerate() {
        if index != 0 {
            out.push_str("  , ");
        }
        writeln!(
            out,
            "{{ checkpointIndex := {}, rowStart := {}, rowEnd := {}, emissionStart := {}, emissionEnd := {} }}",
            range.checkpoint_index, range.row_start, range.row_end, range.emission_start, range.emission_end,
        )
        .unwrap();
    }
    out.push_str("  ]\n\n");

    out.push_str("def constantPins : List ConstantPin :=\n  [ ");
    for (index, pin) in artifact.constant_pins.iter().enumerate() {
        if index != 0 {
            out.push_str("  , ");
        }
        writeln!(
            out,
            "{{ row := {}, column := {}, value := {} }}",
            pin.row, pin.column, pin.value,
        )
        .unwrap();
    }
    out.push_str("  ]\n\n");

    out.push_str("def calls : List CompactCall :=\n  [ ");
    for (index, call) in artifact.calls.iter().enumerate() {
        if index != 0 {
            out.push_str("  , ");
        }
        writeln!(
            out,
            "{{ traceIndex := {}, rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }}",
            call.trace_index,
            call.row_start,
            call.row_end,
            lean_nat_list(call.input_columns),
            call.first_allocated_column,
        )
        .unwrap();
    }
    out.push_str("  ]\n\n");

    out.push_str("def emissionOrder : List EmissionRef :=\n  [ ");
    for (index, emission) in artifact.emission_order.iter().enumerate() {
        if index != 0 {
            if index % 8 == 0 {
                out.push_str(",\n    ");
            } else {
                out.push_str(", ");
            }
        }
        match emission {
            EmissionRef::Pin(pin) => write!(out, ".pin {pin}").unwrap(),
            EmissionRef::Call(call) => write!(out, ".call {call}").unwrap(),
        }
    }
    out.push_str(" ]\n\n");

    out.push_str("def stateContinuity : List StateContinuity :=\n  [ ");
    for (index, continuity) in artifact.state_continuity.iter().enumerate() {
        if index != 0 {
            out.push_str("  , ");
        }
        writeln!(
            out,
            "{{ fromCall := {}, toCall := {}, lanes := {} }}",
            continuity.from_call,
            continuity.to_call,
            lean_nat_list(continuity.lanes.iter().copied()),
        )
        .unwrap();
    }
    out.push_str("  ]\n\n");

    out.push_str("def fieldOutputAliases : List FieldOutputAlias :=\n  [ ");
    for (index, alias) in artifact.field_output_aliases.iter().enumerate() {
        if index != 0 {
            out.push_str("  , ");
        }
        writeln!(
            out,
            "{{ ordinal := {}, groupIndex := {}, blockIndex := {}, laneIndex := {}, callIndex := {}, outputLane := {}, fieldColumn := {}, canonicalRowStart := {}, canonicalRowEnd := {} }}",
            alias.ordinal,
            alias.group_index,
            alias.block_index,
            alias.lane_index,
            alias.call_index,
            alias.output_lane,
            alias.field_column,
            alias.canonical_row_start,
            alias.canonical_row_end,
        )
        .unwrap();
    }
    out.push_str("  ]\n\n");

    writeln!(out, "def layout : TranscriptLayout :=").unwrap();
    writeln!(out, "  {{ sourceRows := {}", artifact.source_rows).unwrap();
    writeln!(out, "    sourceColumns := {}", artifact.source_columns).unwrap();
    writeln!(out, "    ownedRowCount := {}", artifact.owned_row_count).unwrap();
    writeln!(out, "    ownedRanges := ownedRanges").unwrap();
    writeln!(out, "    constantPins := constantPins").unwrap();
    writeln!(out, "    calls := calls").unwrap();
    writeln!(out, "    emissionOrder := emissionOrder").unwrap();
    writeln!(
        out,
        "    entryProducerTraceIndex := {}",
        artifact.entry_producer_trace_index,
    )
    .unwrap();
    for (name, boundary) in [
        ("entryBoundary", &artifact.entry_boundary),
        ("postBindBoundary", &artifact.post_bind_boundary),
        ("finalBoundary", &artifact.final_boundary),
    ] {
        writeln!(
            out,
            "    {name} := {{ stateColumns := {}, cursor := {} }}",
            lean_nat_list(boundary.state_columns),
            boundary.cursor,
        )
        .unwrap();
    }
    writeln!(
        out,
        "    entryToFirstCallLanes := {}",
        lean_nat_list(artifact.entry_to_first_call_lanes.iter().copied()),
    )
    .unwrap();
    writeln!(
        out,
        "    postBindToFirstRhoCallLanes := {}",
        lean_nat_list(artifact.post_bind_to_first_rho_call_lanes.iter().copied()),
    )
    .unwrap();
    writeln!(out, "    stateContinuity := stateContinuity").unwrap();
    writeln!(out, "    fieldOutputAliases := fieldOutputAliases").unwrap();
    writeln!(
        out,
        "    bindCallIndices := {}",
        lean_nat_list(artifact.bind_call_indices.iter().copied()),
    )
    .unwrap();
    writeln!(out, "    firstRhoCallIndex := {}", artifact.first_rho_call_index).unwrap();
    writeln!(
        out,
        "    bindInputColumns := {} }}",
        lean_nat_list(artifact.bind_input_columns.iter().copied()),
    )
    .unwrap();
    out.push_str("\nend Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutData\n");
    out
}

#[test]
fn active_pi_rlc_transcript_layout_matches_production_trace() {
    let artifact = extract();
    let rendered = render(&artifact);
    let path = repo_root().join(LEAN_DATA_PATH);
    let committed = fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = path.with_extension("lean.expected");
        fs::create_dir_all(expected.parent().expect("transcript artifact parent"))
            .expect("create transcript artifact directory");
        fs::write(&expected, &rendered).expect("write expected transcript artifact");
        panic!("active PiRLC transcript layout drifted; review {}", expected.display(),);
    }
}
