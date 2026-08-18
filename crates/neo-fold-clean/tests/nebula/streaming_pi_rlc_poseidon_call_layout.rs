//! Exact semantic-to-physical call layout for production PiRLC replay.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::ops::Range;

use neo_fold_clean::engine::r1cs_circuit::builder::{Lc, Poseidon2CompactPermutationAudit};
use neo_fold_clean::frontends::nebula::f_prime::{
    production_pi_rlc_family_body_compact_layout_and_decoder_runs_for_ranges, NebulaFPrimePiRlcFamilyBodySynthesis,
    NebulaFPrimePiRlcFamilyReplayArmKind, NebulaFPrimePiRlcFamilyReplayCallAudit,
    NebulaFPrimePiRlcFamilyReplayCallClass, NebulaFPrimePiRlcFamilyReplayScope,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveCompactLayoutAudit, SelectiveProjectedDecoderRunProvenance, SelectiveProjectedSourceResolution,
    SelectiveRewriteKind,
};
use p3_field::PrimeField64;

const ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.lean";
const SOURCE_CALL_STRIDE: usize = 600;
const EMITTED_CALL_ROWS: usize = 86;
const SLOT_WIDTH: usize = 41;
const LOCAL_FINAL_STRIDE: usize = EMITTED_CALL_ROWS * SLOT_WIDTH;
const ROW_TEMPLATE_SOURCE: &str = "rust:nightstream/streaming-pi-rlc-family/poseidon2-normalized-row-template/v1";

#[derive(Clone, Debug, PartialEq, Eq)]
struct NormalizedLc {
    terms: Vec<(usize, u64)>,
    constant: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NormalizedPoseidonTrace {
    sboxes: Vec<(NormalizedLc, usize)>,
    outputs: [usize; 8],
    output_linear_forms: Vec<NormalizedLc>,
}

fn assert_lc_permutation(expected: &NormalizedLc, actual: &NormalizedLc, context: &str) {
    assert_eq!(actual.constant, expected.constant, "{context}: constant");
    let mut expected_terms = expected.terms.clone();
    let mut actual_terms = actual.terms.clone();
    expected_terms.sort_unstable();
    actual_terms.sort_unstable();
    assert_eq!(actual_terms, expected_terms, "{context}: term multiset");
}

fn assert_trace_permutation(
    expected: &NormalizedPoseidonTrace,
    actual: &NormalizedPoseidonTrace,
    arm: usize,
    call: usize,
) {
    assert_eq!(
        actual.sboxes.len(),
        expected.sboxes.len(),
        "arm {arm} call {call}: S-box count"
    );
    for (index, ((expected_input, expected_output), (actual_input, actual_output))) in
        expected.sboxes.iter().zip(&actual.sboxes).enumerate()
    {
        assert_eq!(
            actual_output, expected_output,
            "arm {arm} call {call} S-box {index}: output role"
        );
        assert_lc_permutation(
            expected_input,
            actual_input,
            &format!("arm {arm} call {call} S-box {index}: input"),
        );
    }
    assert_eq!(actual.outputs, expected.outputs, "arm {arm} call {call}: output roles");
    assert_eq!(
        actual.output_linear_forms.len(),
        expected.output_linear_forms.len(),
        "arm {arm} call {call}: output form count"
    );
    for (index, (expected_form, actual_form)) in expected
        .output_linear_forms
        .iter()
        .zip(&actual.output_linear_forms)
        .enumerate()
    {
        assert_lc_permutation(
            expected_form,
            actual_form,
            &format!("arm {arm} call {call} output form {index}"),
        );
    }
}

#[derive(Clone, Debug)]
struct NormalizedCall {
    scope: NebulaFPrimePiRlcFamilyReplayScope,
    index: usize,
    class: NebulaFPrimePiRlcFamilyReplayCallClass,
    row_start: usize,
    row_end: usize,
    state_before: [usize; 8],
    absorbed: Vec<usize>,
    permutation_input: [usize; 8],
    local_source_start: usize,
    output: [usize; 8],
}

#[derive(Clone, Debug)]
struct CallLayout {
    arm: usize,
    scope: NebulaFPrimePiRlcFamilyReplayScope,
    index: usize,
    class: NebulaFPrimePiRlcFamilyReplayCallClass,
    selector: usize,
    source_row_start: usize,
    emitted_row_start: usize,
    fresh_source_start: usize,
    fresh_count: usize,
    fresh_final_start: usize,
    initial_carried_source_start: Option<usize>,
    initial_carried_final_start: Option<usize>,
    initial_capacity_source_start: Option<usize>,
    initial_capacity_final_start: Option<usize>,
    local_source_start: usize,
    local_final_start: usize,
    previous_local_source_start: Option<usize>,
    previous_local_final_start: Option<usize>,
    previous_capacity_source_offset: Option<usize>,
}

#[derive(Clone, Debug)]
struct CallRun {
    arm: usize,
    scope: NebulaFPrimePiRlcFamilyReplayScope,
    call_count: usize,
    first_class: NebulaFPrimePiRlcFamilyReplayCallClass,
    selector: usize,
    source_row_start: usize,
    emitted_row_start: usize,
    first_fresh_count: usize,
    fresh_source_start: usize,
    fresh_final_start: usize,
    initial_carried_source_start: Option<usize>,
    initial_carried_final_start: Option<usize>,
    initial_capacity_source_start: usize,
    initial_capacity_final_start: usize,
    local_source_start: usize,
    local_final_start: usize,
    previous_capacity_source_offset: usize,
}

struct ArmLayout {
    source_rows: usize,
    source_columns: usize,
    final_rows: usize,
    final_columns: usize,
    runs: Vec<CallRun>,
}

struct PreparedArm {
    arm: usize,
    source_rows: usize,
    source_columns: usize,
    calls: Vec<NormalizedCall>,
    rewrite_owners: Vec<Range<usize>>,
    decoder_ranges: Vec<Range<usize>>,
    trace_template: NormalizedPoseidonTrace,
}

fn normalized_trace_column(call: &NormalizedCall, column: usize) -> usize {
    if let Some(position) = call
        .permutation_input
        .iter()
        .position(|&input| input == column)
    {
        return position;
    }
    assert!((call.local_source_start..call.local_source_start + SOURCE_CALL_STRIDE).contains(&column));
    8 + column - call.local_source_start
}

fn normalized_lc(synthesis: &NebulaFPrimePiRlcFamilyBodySynthesis, call: &NormalizedCall, value: &Lc) -> NormalizedLc {
    NormalizedLc {
        terms: value
            .terms
            .iter()
            .map(|&(column, coefficient)| {
                (
                    normalized_trace_column(call, normalize_column(synthesis, column)),
                    coefficient.as_canonical_u64(),
                )
            })
            .collect(),
        constant: value.constant.as_canonical_u64(),
    }
}

fn normalized_poseidon_trace(
    synthesis: &NebulaFPrimePiRlcFamilyBodySynthesis,
    call: &NormalizedCall,
    trace: &Poseidon2CompactPermutationAudit,
) -> NormalizedPoseidonTrace {
    assert_eq!(normalize_columns(synthesis, trace.input_cols), call.permutation_input);
    assert_eq!(trace.sboxes.len(), EMITTED_CALL_ROWS);
    assert_eq!(normalize_columns(synthesis, trace.output_cols), call.output);
    NormalizedPoseidonTrace {
        sboxes: trace
            .sboxes
            .iter()
            .map(|sbox| {
                (
                    normalized_lc(synthesis, call, &sbox.input),
                    normalized_trace_column(call, normalize_column(synthesis, sbox.output_col)),
                )
            })
            .collect(),
        outputs: trace
            .output_cols
            .map(|column| normalized_trace_column(call, normalize_column(synthesis, column))),
        output_linear_forms: trace
            .output_linear_forms
            .iter()
            .map(|value| normalized_lc(synthesis, call, value))
            .collect(),
    }
}

fn normalize_column(synthesis: &NebulaFPrimePiRlcFamilyBodySynthesis, column: usize) -> usize {
    synthesis
        .normalized_field_column_for_artifact(column)
        .expect("PiRLC replay source column is in the normalized assignment")
}

fn normalize_columns<const N: usize>(
    synthesis: &NebulaFPrimePiRlcFamilyBodySynthesis,
    columns: [usize; N],
) -> [usize; N] {
    columns.map(|column| normalize_column(synthesis, column))
}

fn normalize_call(
    synthesis: &NebulaFPrimePiRlcFamilyBodySynthesis,
    call: &NebulaFPrimePiRlcFamilyReplayCallAudit,
) -> NormalizedCall {
    let local_source_start = normalize_column(synthesis, call.first_allocated_column());
    for offset in 0..call.allocated_column_count() {
        assert_eq!(
            normalize_column(synthesis, call.first_allocated_column() + offset),
            local_source_start + offset,
        );
    }
    NormalizedCall {
        scope: call.scope(),
        index: call.index(),
        class: call.class(),
        row_start: call.row_start(),
        row_end: call.row_end(),
        state_before: normalize_columns(synthesis, call.state_before_columns()),
        absorbed: call
            .absorbed_columns()
            .iter()
            .map(|&column| normalize_column(synthesis, column))
            .collect(),
        permutation_input: normalize_columns(synthesis, call.permutation_input_columns()),
        local_source_start,
        output: normalize_columns(synthesis, call.output_columns()),
    }
}

fn replay_decoder_ranges(calls: &[NormalizedCall]) -> Vec<Range<usize>> {
    let mut required = Vec::with_capacity(calls.len() * 9);
    for call in calls {
        required.push(call.local_source_start..call.local_source_start + SOURCE_CALL_STRIDE);
        required.extend(
            call.permutation_input
                .iter()
                .map(|&column| column..column + 1),
        );
    }
    required.sort_by_key(|range| (range.start, range.end));
    let mut merged: Vec<Range<usize>> = Vec::new();
    for range in required.iter().cloned() {
        if let Some(previous) = merged.last_mut() {
            if range.start <= previous.end {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        merged.push(range);
    }
    for range in required {
        assert_eq!(
            merged
                .iter()
                .filter(|owner| owner.start <= range.start && range.end <= owner.end)
                .count(),
            1,
        );
    }
    merged
}

fn contiguous_start(values: &[usize], stride: usize) -> usize {
    let start = *values.first().expect("nonempty contiguous layout slice");
    assert!(values
        .iter()
        .enumerate()
        .all(|(index, value)| *value == start + index * stride));
    start
}

fn source_resolution(
    decoders: &[&SelectiveProjectedDecoderRunProvenance],
    column: usize,
) -> SelectiveProjectedSourceResolution {
    let resolutions = decoders
        .iter()
        .filter_map(|decoder| {
            decoder
                .runs()
                .iter()
                .find_map(|run| run.resolution_at(column))
        })
        .collect::<Vec<_>>();
    assert_eq!(resolutions.len(), 1);
    resolutions
        .into_iter()
        .next()
        .expect("source column has one complete decoder owner")
}

fn direct_slot_start(decoders: &[&SelectiveProjectedDecoderRunProvenance], column: usize) -> usize {
    match source_resolution(decoders, column) {
        SelectiveProjectedSourceResolution::Direct { start, width, .. }
        | SelectiveProjectedSourceResolution::EqualityAlias { start, width, .. } => {
            assert_eq!(width, SLOT_WIDTH);
            start
        }
        resolution => panic!("source column {column} is not a field slot: {resolution:?}"),
    }
}

fn contiguous_slot_start(decoders: &[&SelectiveProjectedDecoderRunProvenance], columns: &[usize]) -> usize {
    let starts = columns
        .iter()
        .map(|&column| direct_slot_start(decoders, column))
        .collect::<Vec<_>>();
    contiguous_start(&starts, SLOT_WIDTH)
}

fn poseidon_instance_starts(decoders: &[&SelectiveProjectedDecoderRunProvenance]) -> BTreeMap<usize, usize> {
    let templates = decoders
        .iter()
        .flat_map(|decoder| decoder.repeated_templates())
        .filter(|template| template.source_width() == SOURCE_CALL_STRIDE)
        .collect::<Vec<_>>();
    assert_eq!(templates.len(), 1);
    let instances = templates[0]
        .instances()
        .iter()
        .flat_map(|instances| (0..instances.count()).map(|index| instances.instance(index).unwrap()))
        .map(|(source_start, final_start, _, _)| (source_start, final_start))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        instances.len(),
        templates[0]
            .instances()
            .iter()
            .map(|instances| instances.count())
            .sum::<usize>()
    );
    instances
}

fn compress_run(arm: usize, scope: NebulaFPrimePiRlcFamilyReplayScope, calls: &[CallLayout]) -> CallRun {
    let calls = calls
        .iter()
        .filter(|call| call.scope == scope)
        .collect::<Vec<_>>();
    let first = calls
        .first()
        .copied()
        .expect("replay scope has a first call");
    assert_eq!(first.arm, arm);
    assert_eq!(first.index, 0);
    assert!(matches!(
        first.class,
        NebulaFPrimePiRlcFamilyReplayCallClass::Direct | NebulaFPrimePiRlcFamilyReplayCallClass::PartialStart
    ));
    let initial_capacity_source_start = first
        .initial_capacity_source_start
        .expect("first call has initial capacity source");
    let initial_capacity_final_start = first
        .initial_capacity_final_start
        .expect("first call has initial capacity slot");
    let previous_capacity_source_offset = first.local_source_start + SOURCE_CALL_STRIDE - 4;
    let previous_capacity_source_offset = previous_capacity_source_offset - first.local_source_start;

    for (index, call) in calls.iter().enumerate() {
        assert_eq!(call.arm, arm);
        assert_eq!(call.scope, scope);
        assert_eq!(call.index, index);
        assert_eq!(call.selector, first.selector);
        assert_eq!(
            call.source_row_start,
            first.source_row_start + index * SOURCE_CALL_STRIDE
        );
        assert_eq!(
            call.emitted_row_start,
            first.emitted_row_start + index * EMITTED_CALL_ROWS
        );
        assert_eq!(
            call.local_source_start,
            first.local_source_start + index * SOURCE_CALL_STRIDE
        );
        assert_eq!(
            call.local_final_start,
            first.local_final_start + index * LOCAL_FINAL_STRIDE
        );
        let fresh_offset = if index == 0 {
            0
        } else {
            first.fresh_count + (index - 1) * 4
        };
        assert_eq!(call.fresh_source_start, first.fresh_source_start + fresh_offset);
        assert_eq!(
            call.fresh_final_start,
            first.fresh_final_start + fresh_offset * SLOT_WIDTH
        );
        if index == 0 {
            assert_eq!(call.class, first.class);
            assert_eq!(call.previous_local_source_start, None);
            assert_eq!(call.previous_local_final_start, None);
        } else {
            assert_eq!(call.class, NebulaFPrimePiRlcFamilyReplayCallClass::Chained);
            assert_eq!(call.fresh_count, 4);
            assert_eq!(call.initial_carried_source_start, None);
            assert_eq!(call.initial_carried_final_start, None);
            assert_eq!(call.initial_capacity_source_start, None);
            assert_eq!(call.initial_capacity_final_start, None);
            assert_eq!(
                call.previous_local_source_start,
                Some(call.local_source_start - SOURCE_CALL_STRIDE)
            );
            assert_eq!(
                call.previous_local_final_start,
                Some(call.local_final_start - LOCAL_FINAL_STRIDE)
            );
            assert_eq!(
                call.previous_capacity_source_offset,
                Some(previous_capacity_source_offset)
            );
        }
    }

    CallRun {
        arm,
        scope,
        call_count: calls.len(),
        first_class: first.class,
        selector: first.selector,
        source_row_start: first.source_row_start,
        emitted_row_start: first.emitted_row_start,
        first_fresh_count: first.fresh_count,
        fresh_source_start: first.fresh_source_start,
        fresh_final_start: first.fresh_final_start,
        initial_carried_source_start: first.initial_carried_source_start,
        initial_carried_final_start: first.initial_carried_final_start,
        initial_capacity_source_start,
        initial_capacity_final_start,
        local_source_start: first.local_source_start,
        local_final_start: first.local_final_start,
        previous_capacity_source_offset,
    }
}

fn prepare_arm(arm: usize, kind: NebulaFPrimePiRlcFamilyReplayArmKind) -> PreparedArm {
    let synthesis = NebulaFPrimePiRlcFamilyBodySynthesis::production(kind);
    let calls = synthesis
        .replay_call_audits()
        .iter()
        .map(|call| normalize_call(&synthesis, call))
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), kind.poseidon2_calls());
    let permutation_audits = synthesis
        .builder_for_artifact()
        .poseidon2_permutation_audits();
    let compact_audits = synthesis
        .builder_for_artifact()
        .poseidon2_compact_permutation_audits();
    assert_eq!(permutation_audits.len(), compact_audits.len());
    let normalized_traces = calls
        .iter()
        .map(|call| {
            let matches = permutation_audits
                .iter()
                .enumerate()
                .filter(|(_, audit)| audit.row_start == call.row_start && audit.row_end == call.row_end)
                .collect::<Vec<_>>();
            let [(trace_index, _)] = matches.as_slice() else {
                panic!("replay call must have one exact compact Poseidon2 trace");
            };
            normalized_poseidon_trace(&synthesis, call, &compact_audits[*trace_index])
        })
        .collect::<Vec<_>>();
    let trace_template = normalized_traces
        .first()
        .cloned()
        .expect("replay arm has a Poseidon2 trace");
    for (call, trace) in normalized_traces.iter().enumerate() {
        assert_trace_permutation(&trace_template, trace, arm, call);
    }
    let decoder_ranges = replay_decoder_ranges(&calls);
    PreparedArm {
        arm,
        source_rows: synthesis.rows(),
        source_columns: synthesis.columns(),
        calls,
        rewrite_owners: Vec::new(),
        decoder_ranges,
        trace_template,
    }
}

fn bind_rewrite_owners(prepared: &mut PreparedArm, layout: &SelectiveCompactLayoutAudit) {
    assert!(prepared.rewrite_owners.is_empty());
    let mut rewrite_owners = Vec::with_capacity(prepared.calls.len());
    let mut emitted_row_starts = BTreeSet::new();
    for call in &prepared.calls {
        let matches = layout
            .rows()
            .rewrites()
            .iter()
            .filter(|rewrite| {
                rewrite.arm() == prepared.arm
                    && rewrite.kind() == SelectiveRewriteKind::Poseidon2
                    && rewrite.source_rows().len() == 1
                    && rewrite.source_rows()[0] == (call.row_start..call.row_end)
            })
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1);
        let rewrite = matches[0];
        assert_eq!(rewrite.emitted_rows().len(), EMITTED_CALL_ROWS);
        assert!(emitted_row_starts.insert(rewrite.emitted_rows().start));
        rewrite_owners.push(rewrite.emitted_rows());
    }
    assert_eq!(emitted_row_starts.len(), prepared.calls.len());
    prepared.rewrite_owners = rewrite_owners;
}

fn audit_arm(
    prepared: &PreparedArm,
    layout: &SelectiveCompactLayoutAudit,
    decoders: &[&SelectiveProjectedDecoderRunProvenance],
) -> ArmLayout {
    let arm = prepared.arm;
    let calls = &prepared.calls;
    assert!(!decoders.is_empty());
    assert!(decoders.iter().all(|decoder| decoder.arm() == arm));
    let mut actual_ranges = decoders
        .iter()
        .map(|decoder| decoder.source_range())
        .collect::<Vec<_>>();
    actual_ranges.sort_by_key(|range| (range.start, range.end));
    assert_eq!(actual_ranges, prepared.decoder_ranges);
    let final_columns = decoders[0].final_columns();
    assert!(decoders
        .iter()
        .all(|decoder| decoder.final_columns() == final_columns));
    let poseidon_instances = poseidon_instance_starts(decoders);
    let selector = layout.selector_columns()[arm];
    let mut layouts = Vec::with_capacity(calls.len());
    for (call, emitted_rows) in calls.iter().zip(&prepared.rewrite_owners) {
        let local_final_start = *poseidon_instances
            .get(&call.local_source_start)
            .expect("replay call has one exact 600-column decoder-template instance");

        assert_eq!(call.permutation_input[4..8], call.state_before[4..8]);
        let fresh_source_start = contiguous_start(&call.absorbed, 1);
        let fresh_final_start = contiguous_slot_start(decoders, &call.absorbed);
        let mut initial_carried_source_start = None;
        let mut initial_carried_final_start = None;
        let mut initial_capacity_source_start = None;
        let mut initial_capacity_final_start = None;
        let mut previous_local_source_start = None;
        let mut previous_local_final_start = None;
        let mut previous_capacity_source_offset = None;

        match call.class {
            NebulaFPrimePiRlcFamilyReplayCallClass::Direct => {
                assert_eq!(call.absorbed.len(), 4);
                assert_eq!(call.permutation_input[..4], call.absorbed);
                initial_capacity_source_start = Some(contiguous_start(&call.state_before[4..8], 1));
                initial_capacity_final_start = Some(contiguous_slot_start(decoders, &call.state_before[4..8]));
            }
            NebulaFPrimePiRlcFamilyReplayCallClass::PartialStart => {
                assert_eq!(call.absorbed.len(), 2);
                assert_eq!(call.permutation_input[..2], call.state_before[..2]);
                assert_eq!(call.permutation_input[2..4], call.absorbed);
                initial_carried_source_start = Some(contiguous_start(&call.state_before[..2], 1));
                initial_carried_final_start = Some(contiguous_slot_start(decoders, &call.state_before[..2]));
                initial_capacity_source_start = Some(contiguous_start(&call.state_before[4..8], 1));
                initial_capacity_final_start = Some(contiguous_slot_start(decoders, &call.state_before[4..8]));
            }
            NebulaFPrimePiRlcFamilyReplayCallClass::Chained => {
                assert_eq!(call.absorbed.len(), 4);
                assert_eq!(call.permutation_input[..4], call.absorbed);
                let previous = layouts
                    .iter()
                    .rev()
                    .find(|layout: &&CallLayout| layout.scope == call.scope)
                    .expect("chained replay call has a previous call in the same scope");
                assert_eq!(
                    call.state_before,
                    calls
                        .iter()
                        .find(|candidate| { candidate.scope == call.scope && candidate.index + 1 == call.index })
                        .expect("previous normalized replay call")
                        .output
                );
                let capacity_source_start = contiguous_start(&call.state_before[4..8], 1);
                let capacity_offset = capacity_source_start - previous.local_source_start;
                previous_local_source_start = Some(previous.local_source_start);
                previous_local_final_start = Some(previous.local_final_start);
                previous_capacity_source_offset = Some(capacity_offset);
            }
        }

        layouts.push(CallLayout {
            arm,
            scope: call.scope,
            index: call.index,
            class: call.class,
            selector,
            source_row_start: call.row_start,
            emitted_row_start: emitted_rows.start,
            fresh_source_start,
            fresh_count: call.absorbed.len(),
            fresh_final_start,
            initial_carried_source_start,
            initial_carried_final_start,
            initial_capacity_source_start,
            initial_capacity_final_start,
            local_source_start: call.local_source_start,
            local_final_start,
            previous_local_source_start,
            previous_local_final_start,
            previous_capacity_source_offset,
        });
    }

    let runs = [
        compress_run(arm, NebulaFPrimePiRlcFamilyReplayScope::Input, &layouts),
        compress_run(arm, NebulaFPrimePiRlcFamilyReplayScope::Output, &layouts),
    ]
    .into_iter()
    .collect();
    ArmLayout {
        source_rows: prepared.source_rows,
        source_columns: prepared.source_columns,
        final_rows: layout.rows().total_rows(),
        final_columns,
        runs,
    }
}

fn lean_scope(scope: NebulaFPrimePiRlcFamilyReplayScope) -> &'static str {
    match scope {
        NebulaFPrimePiRlcFamilyReplayScope::Input => ".input",
        NebulaFPrimePiRlcFamilyReplayScope::Output => ".output",
    }
}

fn lean_first_class(class: NebulaFPrimePiRlcFamilyReplayCallClass) -> &'static str {
    match class {
        NebulaFPrimePiRlcFamilyReplayCallClass::Direct => ".direct",
        NebulaFPrimePiRlcFamilyReplayCallClass::PartialStart => ".partialStart",
        NebulaFPrimePiRlcFamilyReplayCallClass::Chained => panic!("run cannot start with a chained call"),
    }
}

fn lean_option(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| format!("some {value}"))
}

fn render_artifact(arms: &[ArmLayout; 2]) -> String {
    assert_eq!(arms[0].final_rows, arms[1].final_rows);
    assert_eq!(arms[0].final_columns, arms[1].final_columns);
    let runs = arms.iter().flat_map(|arm| &arm.runs).collect::<Vec<_>>();
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
/-! Generated file: exact semantic-to-physical layout of every production\n\
PiRLC replay Poseidon2 call.\n\n\
Owns: source-arm identity, input/output scope, first-call class, exact source\n\
and emitted row runs, selectors, and source-to-final slot placement.\n\n\
Owns also: the canonical normalized 86-row compact trace shared by all calls\n\
up to linear-combination operand permutation and bound through the exact\n\
selective rewrite and decoder layout.\n\n\
Does not own: Poseidon2 semantics, lifecycle\n\
soundness, or permission to remove constraints.\n\n\
Emits constraints: no.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout\n\n\
inductive RawScope where\n  | input\n  | output\n  deriving DecidableEq, Repr\n\n\
inductive RawFirstClass where\n  | direct\n  | partialStart\n  deriving DecidableEq, Repr\n\n\
structure RawRun where\n\
  \x20\x20arm : Nat\n\
  \x20\x20scope : RawScope\n\
  \x20\x20callCount : Nat\n\
  \x20\x20firstClass : RawFirstClass\n\
  \x20\x20selectorColumn : Nat\n\
  \x20\x20sourceRowStart : Nat\n\
  \x20\x20emittedRowStart : Nat\n\
  \x20\x20firstFreshCount : Nat\n\
  \x20\x20freshSourceStart : Nat\n\
  \x20\x20freshFinalStart : Nat\n\
  \x20\x20initialCarriedSourceStart : Option Nat\n\
  \x20\x20initialCarriedFinalStart : Option Nat\n\
  \x20\x20initialCapacitySourceStart : Nat\n\
  \x20\x20initialCapacityFinalStart : Nat\n\
  \x20\x20localSourceStart : Nat\n\
  \x20\x20localFinalStart : Nat\n\
  \x20\x20previousCapacitySourceOffset : Nat\n\
  \x20\x20deriving DecidableEq, Repr\n\n\
def schemaVersion : Nat := 2\n\
def rowTemplateSource : String := \"{ROW_TEMPLATE_SOURCE}\"\n\
def sourceCallStride : Nat := {SOURCE_CALL_STRIDE}\n\
def emittedCallRows : Nat := {EMITTED_CALL_ROWS}\n\
def slotWidth : Nat := {SLOT_WIDTH}\n\
def localFinalStride : Nat := {LOCAL_FINAL_STRIDE}\n\
def evenSourceRows : Nat := {}\n\
def evenSourceColumns : Nat := {}\n\
def oddSourceRows : Nat := {}\n\
def oddSourceColumns : Nat := {}\n\
def finalRows : Nat := {}\n\
def finalColumns : Nat := {}",
        arms[0].source_rows,
        arms[0].source_columns,
        arms[1].source_rows,
        arms[1].source_columns,
        arms[0].final_rows,
        arms[0].final_columns,
    )
    .expect("render PiRLC call-layout header");
    for (index, run) in runs.iter().enumerate() {
        writeln!(
            rendered,
            "\ndef rawRun{index} : RawRun where\n\
  \x20\x20arm := {}\n\
  \x20\x20scope := {}\n\
  \x20\x20callCount := {}\n\
  \x20\x20firstClass := {}\n\
  \x20\x20selectorColumn := {}\n\
  \x20\x20sourceRowStart := {}\n\
  \x20\x20emittedRowStart := {}\n\
  \x20\x20firstFreshCount := {}\n\
  \x20\x20freshSourceStart := {}\n\
  \x20\x20freshFinalStart := {}\n\
  \x20\x20initialCarriedSourceStart := {}\n\
  \x20\x20initialCarriedFinalStart := {}\n\
  \x20\x20initialCapacitySourceStart := {}\n\
  \x20\x20initialCapacityFinalStart := {}\n\
  \x20\x20localSourceStart := {}\n\
  \x20\x20localFinalStart := {}\n\
  \x20\x20previousCapacitySourceOffset := {}",
            run.arm,
            lean_scope(run.scope),
            run.call_count,
            lean_first_class(run.first_class),
            run.selector,
            run.source_row_start,
            run.emitted_row_start,
            run.first_fresh_count,
            run.fresh_source_start,
            run.fresh_final_start,
            lean_option(run.initial_carried_source_start),
            lean_option(run.initial_carried_final_start),
            run.initial_capacity_source_start,
            run.initial_capacity_final_start,
            run.local_source_start,
            run.local_final_start,
            run.previous_capacity_source_offset,
        )
        .expect("render PiRLC call run");
    }
    writeln!(
        rendered,
        "\ndef rawRuns : List RawRun := [rawRun0, rawRun1, rawRun2, rawRun3]"
    )
    .expect("render PiRLC call-run list");
    writeln!(
        rendered,
        "\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout"
    )
    .expect("render PiRLC call-layout footer");
    rendered
}

#[test]
fn production_replay_poseidon2_exports_exact_call_layout() {
    let mut prepared = [
        prepare_arm(0, NebulaFPrimePiRlcFamilyReplayArmKind::Even),
        prepare_arm(1, NebulaFPrimePiRlcFamilyReplayArmKind::Odd),
    ];
    assert_trace_permutation(&prepared[0].trace_template, &prepared[1].trace_template, 1, 0);
    let requests = prepared
        .iter()
        .flat_map(|prepared| {
            prepared
                .decoder_ranges
                .iter()
                .cloned()
                .map(move |range| (prepared.arm, range))
        })
        .collect::<Vec<_>>();
    let (layout, decoders) = production_pi_rlc_family_body_compact_layout_and_decoder_runs_for_ranges(&requests)
        .expect("one prepared production PiRLC compiler and decoder audit");
    for arm in &mut prepared {
        bind_rewrite_owners(arm, &layout);
    }
    let even_decoders = decoders
        .iter()
        .filter(|decoder| decoder.arm() == 0)
        .collect::<Vec<_>>();
    let odd_decoders = decoders
        .iter()
        .filter(|decoder| decoder.arm() == 1)
        .collect::<Vec<_>>();
    let even_poseidon = even_decoders
        .iter()
        .flat_map(|decoder| decoder.repeated_templates())
        .filter(|template| template.source_width() == SOURCE_CALL_STRIDE)
        .collect::<Vec<_>>();
    let odd_poseidon = odd_decoders
        .iter()
        .flat_map(|decoder| decoder.repeated_templates())
        .filter(|template| template.source_width() == SOURCE_CALL_STRIDE)
        .collect::<Vec<_>>();
    assert_eq!(even_poseidon.len(), 1);
    assert_eq!(odd_poseidon.len(), 1);
    assert_eq!(even_poseidon[0].relative_runs(), odd_poseidon[0].relative_runs());
    let arms = [
        audit_arm(&prepared[0], &layout, &even_decoders),
        audit_arm(&prepared[1], &layout, &odd_decoders),
    ];
    let artifact = render_artifact(&arms);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH);
    if artifact != std::fs::read_to_string(&path).unwrap_or_default() {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, artifact).expect("write reviewed PiRLC replay-call layout artifact");
        panic!("production PiRLC replay-call layout drifted; wrote {expected}");
    }
}
