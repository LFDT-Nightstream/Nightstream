//! Fixed-recursive PiRLC transcript physical-layout artifact.
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

const LEAN_FACADE_PATH: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcChallenge/Generated/TranscriptLayoutData.lean";
const LEAN_SHARD_ROOT: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcChallenge/Generated/TranscriptLayoutData";
const LEAN_MODULE_ROOT: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.Generated.TranscriptLayoutData";
const LEAN_NAMESPACE_ROOT: &str = "Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutData";
const CALL_ROWS: usize = 600;
const CALL_COLUMNS: usize = 600;
const CALL_OUTPUT_OFFSET: usize = 592;
const GROUP_COUNT: usize = 15;
const BLOCKS_PER_GROUP: usize = neo_params::goldilocks_paper_b2::PI_RLC_SAMPLER_DIGEST_ROUNDS;
const LANES_PER_BLOCK: usize = 4;
const FIELD_OUTPUT_ALIAS_COUNT: usize = GROUP_COUNT * BLOCKS_PER_GROUP * LANES_PER_BLOCK;
const CANONICAL_U64_ROWS: usize = 69;
const BIND_INPUT_COUNT: usize = 4;
const BIND_FRAMING_COLUMN_COUNT: usize = 2;
const OWNED_RANGE_COUNT: usize = 1 + GROUP_COUNT + GROUP_COUNT * BLOCKS_PER_GROUP;

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
    let mut pins = Vec::new();
    let mut calls = Vec::new();
    let mut emission_order = Vec::new();
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

    assert_eq!(expected_rows, owned_rows, "exact selected-source-row coverage");
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

struct Phase<'a> {
    module_name: String,
    previous_module_name: Option<String>,
    label: String,
    pin_start: usize,
    call_start: usize,
    emission_start: usize,
    continuity_start: usize,
    alias_start: usize,
    owned_row_start: usize,
    owned_ranges: &'a [OwnedRange],
    constant_pins: Vec<&'a ConstantPin>,
    calls: Vec<&'a CompactCall>,
    emission_order: &'a [EmissionRef],
    state_continuity: Vec<&'a StateContinuity>,
    field_output_aliases: Vec<&'a FieldOutputAlias>,
}

struct RenderedFile {
    relative_path: String,
    contents: String,
}

fn span_is_in_ranges(row_start: usize, row_end: usize, ranges: &[OwnedRange]) -> bool {
    ranges
        .iter()
        .any(|range| range.row_start <= row_start && row_end <= range.row_end)
}

fn phase<'a>(
    artifact: &'a Artifact,
    module_name: String,
    previous_module_name: Option<String>,
    label: String,
    ranges: &'a [OwnedRange],
    group_index: Option<usize>,
    pin_start: usize,
    call_start: usize,
    emission_start: usize,
    continuity_start: usize,
    alias_start: usize,
    owned_row_start: usize,
) -> Phase<'a> {
    let first = ranges.first().expect("transcript phase owns a range");
    let last = ranges.last().expect("transcript phase owns a range");
    let constant_pins = artifact
        .constant_pins
        .iter()
        .filter(|pin| span_is_in_ranges(pin.row, pin.row + 1, ranges))
        .collect();
    let calls = artifact
        .calls
        .iter()
        .filter(|call| span_is_in_ranges(call.row_start, call.row_end, ranges))
        .collect();
    let state_continuity = artifact
        .state_continuity
        .iter()
        .filter(|edge| {
            let target = &artifact.calls[edge.to_call];
            span_is_in_ranges(target.row_start, target.row_end, ranges)
        })
        .collect();
    let field_output_aliases = artifact
        .field_output_aliases
        .iter()
        .filter(|alias| Some(alias.group_index) == group_index)
        .collect();
    Phase {
        module_name,
        previous_module_name,
        label,
        pin_start,
        call_start,
        emission_start,
        continuity_start,
        alias_start,
        owned_row_start,
        owned_ranges: ranges,
        constant_pins,
        calls,
        emission_order: &artifact.emission_order[first.emission_start..last.emission_end],
        state_continuity,
        field_output_aliases,
    }
}

fn phases(artifact: &Artifact) -> Vec<Phase<'_>> {
    let ranges_per_group = 1 + BLOCKS_PER_GROUP;
    assert_eq!(
        artifact.owned_ranges.len(),
        1 + GROUP_COUNT * ranges_per_group,
        "one transcript prelude range plus one sampler and eight digest ranges per group",
    );
    let mut phases = Vec::with_capacity(1 + GROUP_COUNT);
    let mut pin_start = 0;
    let mut call_start = 0;
    let mut emission_start = 0;
    let mut continuity_start = 0;
    let mut alias_start = 0;
    let mut owned_row_start = 0;

    let prelude = phase(
        artifact,
        "Prelude".into(),
        None,
        "the transcript bind prelude".into(),
        &artifact.owned_ranges[..1],
        None,
        pin_start,
        call_start,
        emission_start,
        continuity_start,
        alias_start,
        owned_row_start,
    );
    pin_start += prelude.constant_pins.len();
    call_start += prelude.calls.len();
    emission_start += prelude.emission_order.len();
    continuity_start += prelude.state_continuity.len();
    alias_start += prelude.field_output_aliases.len();
    owned_row_start += prelude
        .owned_ranges
        .iter()
        .map(|range| range.row_end - range.row_start)
        .sum::<usize>();
    phases.push(prelude);

    for group_index in 0..GROUP_COUNT {
        let start = 1 + group_index * ranges_per_group;
        let current = phase(
            artifact,
            format!("Group{group_index:02}"),
            Some(if group_index == 0 {
                "Prelude".into()
            } else {
                format!("Group{:02}", group_index - 1)
            }),
            format!("PiRLC sampler group {group_index}"),
            &artifact.owned_ranges[start..start + ranges_per_group],
            Some(group_index),
            pin_start,
            call_start,
            emission_start,
            continuity_start,
            alias_start,
            owned_row_start,
        );
        pin_start += current.constant_pins.len();
        call_start += current.calls.len();
        emission_start += current.emission_order.len();
        continuity_start += current.state_continuity.len();
        alias_start += current.field_output_aliases.len();
        owned_row_start += current
            .owned_ranges
            .iter()
            .map(|range| range.row_end - range.row_start)
            .sum::<usize>();
        phases.push(current);
    }

    assert_eq!(pin_start, artifact.constant_pins.len(), "phase pin cursor");
    assert_eq!(call_start, artifact.calls.len(), "phase call cursor");
    assert_eq!(emission_start, artifact.emission_order.len(), "phase emission cursor",);
    assert_eq!(
        continuity_start,
        artifact.state_continuity.len(),
        "phase continuity cursor",
    );
    assert_eq!(alias_start, artifact.field_output_aliases.len(), "phase alias cursor",);
    assert_eq!(owned_row_start, artifact.owned_row_count, "phase owned-row cursor",);

    assert_eq!(
        phases
            .iter()
            .flat_map(|phase| phase.owned_ranges.iter())
            .collect::<Vec<_>>(),
        artifact.owned_ranges.iter().collect::<Vec<_>>(),
        "phase ranges recompose the exact owned-range sequence",
    );
    assert_eq!(
        phases
            .iter()
            .flat_map(|phase| phase.constant_pins.iter().copied())
            .collect::<Vec<_>>(),
        artifact.constant_pins.iter().collect::<Vec<_>>(),
        "phase pins recompose the exact constant-pin sequence",
    );
    assert_eq!(
        phases
            .iter()
            .flat_map(|phase| phase.calls.iter().copied())
            .collect::<Vec<_>>(),
        artifact.calls.iter().collect::<Vec<_>>(),
        "phase calls recompose the exact Poseidon2-call sequence",
    );
    assert_eq!(
        phases
            .iter()
            .flat_map(|phase| phase.emission_order.iter())
            .collect::<Vec<_>>(),
        artifact.emission_order.iter().collect::<Vec<_>>(),
        "phase emissions recompose the exact emission sequence",
    );
    assert_eq!(
        phases
            .iter()
            .flat_map(|phase| phase.state_continuity.iter().copied())
            .collect::<Vec<_>>(),
        artifact.state_continuity.iter().collect::<Vec<_>>(),
        "phase continuity edges recompose the exact edge sequence",
    );
    assert_eq!(
        phases
            .iter()
            .flat_map(|phase| phase.field_output_aliases.iter().copied())
            .collect::<Vec<_>>(),
        artifact.field_output_aliases.iter().collect::<Vec<_>>(),
        "phase aliases recompose the exact field-output sequence",
    );
    phases
}

fn render_phase(artifact: &Artifact, phase: &Phase<'_>) -> String {
    let previous_import = phase
        .previous_module_name
        .as_ref()
        .map_or_else(String::new, |previous| {
            format!("import {LEAN_MODULE_ROOT}.{previous}\n")
        });
    let mut out = format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.TranscriptLayoutSchema\n{previous_import}\n\
/-! Generated by `active_pi_rlc_transcript_layout_matches_production_trace`; do not hand-edit.\n\n\
Owns: exact physical transcript-layout records for {}.\n\
Does not own message meaning, cursor semantics, Poseidon2 correctness, sampler\n\
correctness, or authority for any digest.\n\n\
Assurance tier: artifact-checked layout after exact Rust source-row partitioning.\n\
-/\n\n\
namespace {LEAN_NAMESPACE_ROOT}.{}\n\n\
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutSchema\n\n",
        phase.label, phase.module_name,
    );

    out.push_str("def ownedRanges : List OwnedRange :=\n  [ ");
    for (index, range) in phase.owned_ranges.iter().enumerate() {
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

    if phase.constant_pins.is_empty() {
        out.push_str("def constantPins : List ConstantPin := []\n\n");
    } else {
        out.push_str("def constantPins : List ConstantPin :=\n  [ ");
        for (index, pin) in phase.constant_pins.iter().enumerate() {
            if index != 0 {
                if index % 3 == 0 {
                    out.push_str("\n  , ");
                } else {
                    out.push_str(", ");
                }
            }
            write!(
                out,
                "{{ row := {}, column := {}, value := {} }}",
                pin.row, pin.column, pin.value,
            )
            .unwrap();
        }
        out.push_str("\n  ]\n\n");
    }

    if phase.calls.is_empty() {
        out.push_str("def calls : List CompactCall := []\n\n");
    } else {
        out.push_str("def calls : List CompactCall :=\n  [ ");
        for (index, call) in phase.calls.iter().enumerate() {
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
    }

    if phase.emission_order.is_empty() {
        out.push_str("def emissionOrder : List EmissionRef := []\n\n");
    } else {
        out.push_str("def emissionOrder : List EmissionRef :=\n  [ ");
        for (index, emission) in phase.emission_order.iter().enumerate() {
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
    }

    if phase.state_continuity.is_empty() {
        out.push_str("def stateContinuity : List StateContinuity := []\n\n");
    } else {
        out.push_str("def stateContinuity : List StateContinuity :=\n  [ ");
        for (index, continuity) in phase.state_continuity.iter().enumerate() {
            if index != 0 {
                if index % 2 == 0 {
                    out.push_str("\n  , ");
                } else {
                    out.push_str(", ");
                }
            }
            write!(
                out,
                "{{ fromCall := {}, toCall := {}, lanes := {} }}",
                continuity.from_call,
                continuity.to_call,
                lean_nat_list(continuity.lanes.iter().copied()),
            )
            .unwrap();
        }
        out.push_str("\n  ]\n\n");
    }

    if phase.field_output_aliases.is_empty() {
        out.push_str("def fieldOutputAliases : List FieldOutputAlias := []\n\n");
    } else {
        out.push_str("def fieldOutputAliases : List FieldOutputAlias :=\n  [ ");
        for (index, alias) in phase.field_output_aliases.iter().enumerate() {
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
    }

    writeln!(out, "def phase : Phase :=").unwrap();
    writeln!(out, "  {{ pinStart := {}", phase.pin_start).unwrap();
    writeln!(out, "    callStart := {}", phase.call_start).unwrap();
    writeln!(out, "    emissionStart := {}", phase.emission_start).unwrap();
    writeln!(out, "    continuityStart := {}", phase.continuity_start).unwrap();
    writeln!(out, "    aliasStart := {}", phase.alias_start).unwrap();
    writeln!(out, "    ownedRowStart := {}", phase.owned_row_start).unwrap();
    out.push_str("    ownedRanges := ownedRanges\n");
    out.push_str("    constantPins := constantPins\n");
    out.push_str("    calls := calls\n");
    out.push_str("    emissionOrder := emissionOrder\n");
    out.push_str("    stateContinuity := stateContinuity\n");
    out.push_str("    fieldOutputAliases := fieldOutputAliases }\n\n");

    let previous = phase
        .previous_module_name
        .as_ref()
        .map_or_else(|| "none".to_owned(), |name| format!("(some {name}.phase)"));
    writeln!(
        out,
        "theorem valid : Phase.ValidAfter {} {} {} phase := by\n  constructor <;> decide\n",
        artifact.source_rows, artifact.source_columns, previous,
    )
    .unwrap();
    for (name, count) in [
        ("ownedRanges", phase.owned_ranges.len()),
        ("constantPins", phase.constant_pins.len()),
        ("calls", phase.calls.len()),
        ("emissionOrder", phase.emission_order.len()),
        ("stateContinuity", phase.state_continuity.len()),
        ("fieldOutputAliases", phase.field_output_aliases.len()),
    ] {
        writeln!(
            out,
            "theorem {name}_length : phase.{name}.length = {count} := by decide",
        )
        .unwrap();
    }
    out.push_str("\ntheorem pinValuesCanonical :\n");
    out.push_str("    phase.constantPins.all\n");
    out.push_str("      (fun pin => decide (pin.value < 18446744069414584321)) = true := by\n");
    out.push_str("  simp only [List.all_eq_true, decide_eq_true_eq]\n");
    out.push_str("  intro pin member\n");
    out.push_str("  exact phase.pinValueCanonical valid.pinsValid member\n");
    writeln!(
        out,
        "\ntheorem pinEmissionIndices_eq :\n    pinEmissionIndices phase.emissionOrder = List.range' {} {} := by\n  simpa [phase] using valid.pinIndicesExact",
        phase.pin_start,
        phase.constant_pins.len(),
    )
    .unwrap();
    writeln!(
        out,
        "\ntheorem callEmissionIndices_eq :\n    callEmissionIndices phase.emissionOrder = List.range' {} {} := by\n  simpa [phase] using valid.callIndicesExact\n",
        phase.call_start,
        phase.calls.len(),
    )
    .unwrap();

    writeln!(out, "end {LEAN_NAMESPACE_ROOT}.{}", phase.module_name).unwrap();
    out
}

fn render_joined_definition(out: &mut String, name: &str, lean_type: &str, phases: &[Phase<'_>]) {
    writeln!(out, "def {name} : List {lean_type} :=").unwrap();
    for (index, phase) in phases.iter().enumerate() {
        let suffix = if index + 1 == phases.len() { "" } else { " ++" };
        writeln!(out, "  {}.phase.{}{suffix}", phase.module_name, name).unwrap();
    }
    out.push('\n');
}

fn render_facade(artifact: &Artifact, phases: &[Phase<'_>]) -> String {
    let mut out = String::new();
    for phase in phases {
        writeln!(out, "import {LEAN_MODULE_ROOT}.{}", phase.module_name).unwrap();
    }
    out.push_str(
        "\n/-! Generated by `active_pi_rlc_transcript_layout_matches_production_trace`; do not hand-edit.\n\n\
Owns: exact recomposition of the transcript prelude and 15 sampler-group\n\
layout shards. Does not own message meaning, cursor semantics, Poseidon2\n\
correctness, sampler correctness, or authority for any digest.\n\n\
Assurance tier: artifact-checked layout after exact Rust source-row partitioning.\n\
The shard boundary is the protocol's prelude/group ownership boundary.\n\
-/\n\n",
    );
    writeln!(out, "namespace {LEAN_NAMESPACE_ROOT}\n").unwrap();
    out.push_str("open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutSchema\n\n");

    render_joined_definition(&mut out, "ownedRanges", "OwnedRange", phases);
    render_joined_definition(&mut out, "constantPins", "ConstantPin", phases);
    render_joined_definition(&mut out, "calls", "CompactCall", phases);
    render_joined_definition(&mut out, "emissionOrder", "EmissionRef", phases);
    render_joined_definition(&mut out, "stateContinuity", "StateContinuity", phases);
    render_joined_definition(&mut out, "fieldOutputAliases", "FieldOutputAlias", phases);

    out.push_str("def phaseSequence : List Phase :=\n  [");
    for (index, phase) in phases.iter().enumerate() {
        if index != 0 {
            out.push_str(", ");
        }
        write!(out, "{}.phase", phase.module_name).unwrap();
    }
    out.push_str("]\n\n");

    for (name, expected) in [
        ("ownedRanges", artifact.owned_ranges.len()),
        ("constantPins", artifact.constant_pins.len()),
        ("calls", artifact.calls.len()),
        ("emissionOrder", artifact.emission_order.len()),
        ("stateContinuity", artifact.state_continuity.len()),
        ("fieldOutputAliases", artifact.field_output_aliases.len()),
    ] {
        writeln!(out, "theorem {name}_length : {name}.length = {expected} := by").unwrap();
        write!(out, "  simp only [{name}, List.length_append").unwrap();
        for phase in phases {
            write!(out, ", {}.{name}_length", phase.module_name).unwrap();
        }
        out.push_str("]\n\n");
    }

    out.push_str("theorem constantPinValuesCanonical :\n");
    out.push_str("    constantPins.all\n");
    out.push_str("      (fun pin => decide (pin.value < 18446744069414584321)) = true := by\n");
    out.push_str("  simp only [constantPins, List.all_append");
    for phase in phases {
        write!(out, ", {}.pinValuesCanonical", phase.module_name).unwrap();
    }
    out.push_str(", Bool.true_and]\n\n");

    for (kind, total) in [("pin", artifact.constant_pins.len()), ("call", artifact.calls.len())] {
        let function = format!("{kind}EmissionIndices");
        writeln!(
            out,
            "theorem {function}_eq :\n    {function} emissionOrder = List.range {total} := by",
        )
        .unwrap();
        write!(out, "  simp only [emissionOrder, {function}_append").unwrap();
        for phase in phases {
            write!(out, ", {}.{function}_eq", phase.module_name).unwrap();
        }
        out.push_str("]\n  repeat rw [List.range'_append]\n  rw [List.range_eq_range']\n\n");
    }

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
    writeln!(out, "\nend {LEAN_NAMESPACE_ROOT}").unwrap();
    out
}

fn render(artifact: &Artifact) -> Vec<RenderedFile> {
    let phases = phases(artifact);
    let mut files = vec![RenderedFile {
        relative_path: LEAN_FACADE_PATH.into(),
        contents: render_facade(artifact, &phases),
    }];
    files.extend(phases.iter().map(|phase| RenderedFile {
        relative_path: format!("{LEAN_SHARD_ROOT}/{}.lean", phase.module_name),
        contents: render_phase(artifact, phase),
    }));
    files
}

#[test]
fn active_pi_rlc_transcript_layout_matches_production_trace() {
    let artifact = extract();
    let rendered = render(&artifact);
    let root = repo_root();
    let mut drifted = Vec::new();
    for file in &rendered {
        let path = root.join(&file.relative_path);
        if fs::read_to_string(&path).unwrap_or_default() == file.contents {
            continue;
        }
        let expected = path.with_extension("lean.expected");
        fs::create_dir_all(expected.parent().expect("transcript artifact parent"))
            .expect("create transcript artifact directory");
        fs::write(&expected, &file.contents).expect("write expected transcript artifact");
        drifted.push(expected);
    }
    if !drifted.is_empty() {
        panic!(
            "active PiRLC transcript layout drifted; review {}",
            drifted
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
        );
    }

    let shard_prefix = format!("{LEAN_SHARD_ROOT}/");
    let expected_shards = rendered
        .iter()
        .filter(|file| file.relative_path.starts_with(&shard_prefix))
        .map(|file| {
            std::path::Path::new(&file.relative_path)
                .file_name()
                .expect("generated shard file name")
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    let committed_shards = fs::read_dir(root.join(LEAN_SHARD_ROOT))
        .expect("read committed transcript shard directory")
        .map(|entry| entry.expect("transcript shard directory entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "lean")
        })
        .map(|path| {
            path.file_name()
                .expect("committed shard file name")
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(committed_shards, expected_shards, "committed transcript shard set");
}
