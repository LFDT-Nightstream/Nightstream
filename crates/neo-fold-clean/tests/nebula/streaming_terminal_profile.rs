//! Exact source-to-final placement test for the streaming terminal profile.

#[path = "../support/streaming_terminal_finalizer_artifact.rs"]
mod streaming_terminal_finalizer_artifact;
#[path = "../support/streaming_terminal_fixture.rs"]
mod streaming_terminal_fixture;
#[path = "../support/streaming_terminal_program_binding_artifact.rs"]
mod streaming_terminal_program_binding_artifact;
#[path = "../support/streaming_terminal_x_out_hash_artifact.rs"]
mod streaming_terminal_x_out_hash_artifact;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use neo_fold_clean::frontends::nebula::f_prime::{
    NebulaFPrimeStreamingTerminalFieldBinding, NebulaFPrimeStreamingTerminalFieldDomain,
    NebulaFPrimeStreamingTerminalFinalRowRun, NebulaFPrimeStreamingTerminalProfile,
    NebulaFPrimeStreamingTerminalSourceStageBinding, STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS,
};
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{
    streaming_terminal_x_out_authority_audit, STREAMING_TERMINAL_R1CS_FAMILY_NAMES,
};
use neo_fold_clean::paper::digest::F_PRIME_STATE_X_OUT_DOMAIN;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use streaming_terminal_fixture::{build_streaming_terminal_audit_fixture, StreamingTerminalAuditFixture};

const TERMINAL_PROFILE_ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalProfile.lean";
const TERMINAL_PROFILE_SELECTION_ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalProfileSelection.lean";
const TERMINAL_SOURCE_BINDING_ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalSourceBinding.lean";
const TERMINAL_FULL_X_OUT_CONTEXT_ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalFullXOutContext.lean";
const TERMINAL_FULL_PHASE_SEMANTIC_ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalFullPhaseSemantic.lean";
const TERMINAL_FULL_NEBULA_STATE_DIGEST_ARTIFACT_PATH: &str =
    "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalFullNebulaStateDigest.lean";
const TERMINAL_NEBULA_PRESENT_MARKER: u64 = 0x4e42_4c41;

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompactFinalRun {
    family: String,
    rows: std::ops::Range<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompactDecoderBlock {
    owner: &'static str,
    source_fields: std::ops::Range<usize>,
    decoded_columns: std::ops::Range<usize>,
    final_columns: std::ops::Range<usize>,
    width: usize,
    radix: u64,
    scale: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompactDecoderSegment {
    final_columns: std::ops::Range<usize>,
    radix: u64,
    scale: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompactCompositeDecoder {
    owner: &'static str,
    source_field: usize,
    decoded_column: usize,
    segments: Vec<CompactDecoderSegment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CompactDecoderGroup {
    Block(CompactDecoderBlock),
    Composite(CompactCompositeDecoder),
}

fn lean_range(range: std::ops::Range<usize>) -> String {
    format!("{{ start := {}, stop := {} }}", range.start, range.end)
}

fn compact_final_runs(runs: &[NebulaFPrimeStreamingTerminalFinalRowRun]) -> Vec<CompactFinalRun> {
    let mut compact = Vec::<CompactFinalRun>::new();
    for run in runs {
        let family = format!("{:?}", run.family());
        let rows = run.rows();
        assert!(rows.start <= rows.end);
        let extends_previous = compact
            .last()
            .is_some_and(|previous| previous.family == family && previous.rows.end == rows.start);
        if extends_previous {
            let previous = compact.last_mut().expect("checked compact final run");
            previous.rows.end = rows.end;
        } else {
            compact.push(CompactFinalRun { family, rows });
        }
    }
    compact
}

fn compact_columns(columns: &[usize]) -> Vec<std::ops::Range<usize>> {
    let mut runs = Vec::<std::ops::Range<usize>>::new();
    for &column in columns {
        let extends_previous = runs.last().is_some_and(|previous| previous.end == column);
        if extends_previous {
            let previous = runs.last_mut().expect("checked selector column run");
            previous.end += 1;
        } else {
            if let Some(previous) = runs.last() {
                assert!(previous.end < column, "selector columns must be strictly ordered");
            }
            runs.push(column..column + 1);
        }
    }
    runs
}

fn render_ranges(ranges: &[std::ops::Range<usize>]) -> String {
    let values = ranges.iter().cloned().map(lean_range).collect::<Vec<_>>();
    format!("[{}]", values.join(", "))
}

fn render_final_runs(runs: &[CompactFinalRun]) -> String {
    let mut rendered = String::from("[\n");
    for run in runs {
        writeln!(
            rendered,
            "    {{ family := {:?}, rows := {} }},",
            run.family,
            lean_range(run.rows.clone()),
        )
        .expect("render terminal final row run");
    }
    rendered.push_str("  ]");
    rendered
}

fn final_runs_proof(runs: &[CompactFinalRun]) -> String {
    let mut proof = String::new();
    for _ in runs {
        proof.push_str("FinalRunsWithin.cons (by decide) (");
    }
    proof.push_str("FinalRunsWithin.nil");
    proof.extend(std::iter::repeat_n(')', runs.len()));
    proof
}

fn render_source_stage_bindings(bindings: &[NebulaFPrimeStreamingTerminalSourceStageBinding]) -> String {
    let mut rendered = String::from("[\n");
    for binding in bindings {
        let final_runs = compact_final_runs(binding.final_row_runs());
        writeln!(
            rendered,
            "    {{ occurrence := {}, path := {:?}, sourceRows := {}, sourceFieldRuns := {}, finalRuns := {} }},",
            binding.source_stage_occurrence(),
            binding.source_stage_path(),
            lean_range(binding.source_rows()),
            render_ranges(&compact_columns(binding.source_fields())),
            render_final_runs(&final_runs),
        )
        .expect("render terminal source-stage binding");
    }
    rendered.push_str("  ]");
    rendered
}

fn source_stage_bindings_proof(bindings: &[NebulaFPrimeStreamingTerminalSourceStageBinding]) -> String {
    let mut proof = String::new();
    for binding in bindings {
        let final_runs = compact_final_runs(binding.final_row_runs());
        proof.push_str("SourceStageBindingsWithin.cons (by decide) (");
        proof.push_str(&final_runs_proof(&final_runs));
        proof.push_str(") (");
    }
    proof.push_str("SourceStageBindingsWithin.nil");
    proof.extend(std::iter::repeat_n(')', bindings.len()));
    proof
}

fn source_domain(domain: NebulaFPrimeStreamingTerminalFieldDomain) -> &'static str {
    match domain {
        NebulaFPrimeStreamingTerminalFieldDomain::Goldilocks => "goldilocks",
        NebulaFPrimeStreamingTerminalFieldDomain::Boolean => "boolean",
    }
}

fn render_terminal_profile(profile: &NebulaFPrimeStreamingTerminalProfile) -> String {
    let columns = profile.column_layout();
    let local = profile.after_local_state_digest();
    let payload = profile.after_delayed_payload();
    let compact_final_stage_runs = compact_final_runs(profile.final_stage_row_runs());
    let final_stage_runs = render_final_runs(&compact_final_stage_runs);
    let source_stage_bindings = render_source_stage_bindings(profile.source_stage_bindings());
    let final_stage_proof = final_runs_proof(&compact_final_stage_runs);
    let source_stage_proof = source_stage_bindings_proof(profile.source_stage_bindings());
    let schedule_selector_runs = compact_columns(columns.schedule_selectors());
    let overlay_selector_runs = compact_columns(columns.overlay_selectors());
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalProfileSchema\n\n\
         /-! Generated compact ownership profile for the exact Rust streaming terminal reference slice.\n\n\
         The complete Rust rows remain authoritative. This file records their exact source-to-final placement.\n\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfile\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfile.Artifact\n\n\
         def finalStageRuns : List FinalRun := {final_stage_runs}\n\n\
         def sourceStageBindings : List SourceStageBinding := {source_stage_bindings}\n\n\
         def rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := 1,\n    \
            profileId := {:?}, lifecycleScope := {:?},\n    \
            sourceArtifactIdentity := {:?},\n    \
            finalArtifactIdentity := {:?},\n    \
            acceptedWorkItems := {}, terminalArm := {},\n    \
            lifecycleGroup := {}, phaseKind := {},\n    \
            scheduleSelectorColumn := {}, lifecycleSelectorColumn := {},\n    \
            phaseSelectorColumn := {},\n    \
            sourceRows := {}, sourceColumns := {}, sourcePublicColumns := {},\n    \
            finalRows := {}, finalColumns := {}, finalPublicColumns := {},\n    \
            columnLayout :=\n      \
            {{ publicColumns := {}, lifecyclePrivate := {}, phasePrivate := {},\n        \
               scheduleSelectorRuns := {}, scheduledRingPadding := {},\n        \
               overlayPrivate := {}, overlaySelectorRuns := {},\n        \
               finalRingPadding := {} }},\n    \
            sourceStageOccurrence := {}, sourceStagePath := {:?},\n    \
            sourceStageRows := {}, finalStageRuns := finalStageRuns,\n    \
            afterXOutSourceFields := {:?},\n    \
            afterNebulaLaneSourceFields := {:?},\n    \
            afterLocalStateDigest :=\n      \
            {{ sourceFields := {}, sourceDomain := {:?}, finalLinkRows := {} }},\n    \
            afterDelayedPayload :=\n      \
            {{ sourceFields := {}, sourceDomain := {:?}, finalLinkRows := {} }},\n    \
            sourceStageBindings := sourceStageBindings }}\n\n\
         theorem finalStageRunsWithin :\n    \
             FinalRunsWithin rawArtifact.finalRows finalStageRuns := by\n  \
           unfold finalStageRuns\n  \
           exact {final_stage_proof}\n\n\
         theorem sourceStageBindingsWithin :\n    \
             SourceStageBindingsWithin rawArtifact.sourceRows rawArtifact.finalRows\n      \
               sourceStageBindings := by\n  \
           unfold sourceStageBindings\n  \
           exact {source_stage_proof}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfile\n",
        profile.profile_id(),
        profile.lifecycle_scope(),
        profile.source_artifact_identity(),
        profile.final_artifact_identity(),
        profile.accepted_work_items(),
        profile.terminal_arm(),
        profile.lifecycle_group(),
        profile.phase_kind(),
        profile.schedule_selector_column(),
        profile.lifecycle_selector_column(),
        profile.phase_selector_column(),
        profile.source_rows(),
        profile.source_columns(),
        profile.source_public_columns(),
        profile.final_rows(),
        profile.final_columns(),
        profile.final_public_columns(),
        lean_range(columns.public()),
        lean_range(columns.lifecycle_private()),
        lean_range(columns.phase_private()),
        render_ranges(&schedule_selector_runs),
        lean_range(columns.scheduled_ring_padding()),
        lean_range(columns.overlay_private()),
        render_ranges(&overlay_selector_runs),
        lean_range(columns.final_ring_padding()),
        profile.source_stage_occurrence(),
        profile.source_stage_path(),
        lean_range(profile.source_stage_rows()),
        profile.after_x_out().source_fields(),
        profile.after_nebula_lane().source_fields(),
        lean_range(local.source_fields()),
        source_domain(local.source_domain()),
        lean_range(local.final_common_phase_link_rows()),
        lean_range(payload.source_fields()),
        source_domain(payload.source_domain()),
        lean_range(payload.final_common_phase_link_rows()),
    )
}

fn terminal_profile_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(TERMINAL_PROFILE_ARTIFACT_PATH)
}

fn source_bindings(
    profile: &NebulaFPrimeStreamingTerminalProfile,
) -> Vec<(&'static str, &NebulaFPrimeStreamingTerminalFieldBinding)> {
    [
        ("x_out", profile.after_x_out().fields()),
        ("nebula_lane", profile.after_nebula_lane().fields()),
        ("local_state", profile.after_local_state_digest().fields()),
        ("delayed_payload", profile.after_delayed_payload().fields()),
    ]
    .into_iter()
    .flat_map(|(owner, bindings)| bindings.iter().map(move |binding| (owner, binding)))
    .collect()
}

fn decoder_segments(
    binding: &NebulaFPrimeStreamingTerminalFieldBinding,
    final_witness_column_start: usize,
) -> Vec<CompactDecoderSegment> {
    let terms = binding.decoder_terms();
    assert!(!terms.is_empty(), "terminal source decoder is empty");
    let mut segments = Vec::new();
    let mut start = 0;
    while start < terms.len() {
        let first = terms[start];
        assert_ne!(first.coefficient(), F::ZERO, "terminal source decoder scale is zero");
        let radix = terms.get(start + 1).and_then(|next| {
            (next.final_column() == first.final_column() + 1)
                .then(|| {
                    [2, 3, 7]
                        .into_iter()
                        .find(|&candidate| next.coefficient() == first.coefficient() * F::from_u64(candidate))
                })
                .flatten()
        });
        let mut stop = start + 1;
        if let Some(radix) = radix {
            while stop < terms.len()
                && terms[stop].final_column() == terms[stop - 1].final_column() + 1
                && terms[stop].coefficient() == terms[stop - 1].coefficient() * F::from_u64(radix)
            {
                stop += 1;
            }
        }
        let radix = radix.unwrap_or(2);
        let final_start = final_witness_column_start + first.final_column();
        let segment = CompactDecoderSegment {
            final_columns: final_start..final_start + (stop - start),
            radix,
            scale: first.coefficient().as_canonical_u64(),
        };
        let mut coefficient = first.coefficient();
        for (offset, term) in terms[start..stop].iter().enumerate() {
            assert_eq!(term.final_column() + final_witness_column_start, final_start + offset);
            assert_eq!(term.coefficient(), coefficient);
            coefficient *= F::from_u64(radix);
        }
        segments.push(segment);
        start = stop;
    }
    segments
}

fn compact_decoder_groups(fixture: &StreamingTerminalAuditFixture) -> Vec<CompactDecoderGroup> {
    let mut groups = Vec::<CompactDecoderGroup>::new();
    for (index, (owner, binding)) in source_bindings(&fixture.profile).into_iter().enumerate() {
        let source_field = binding.source_field();
        let decoded_column = fixture.source_binding_decoded_column_start + index;
        let segments = decoder_segments(binding, fixture.final_witness_column_start);
        if let [segment] = segments.as_slice() {
            let width = segment.final_columns.len();
            let extends_previous = matches!(
                groups.last(),
                Some(CompactDecoderGroup::Block(previous))
                    if previous.owner == owner
                        && previous.source_fields.end == source_field
                        && previous.decoded_columns.end == decoded_column
                        && previous.final_columns.end == segment.final_columns.start
                        && previous.width == width
                        && previous.radix == segment.radix
                        && previous.scale == segment.scale
            );
            if extends_previous {
                let Some(CompactDecoderGroup::Block(previous)) = groups.last_mut() else {
                    unreachable!("checked terminal decoder block")
                };
                previous.source_fields.end += 1;
                previous.decoded_columns.end += 1;
                previous.final_columns.end += width;
            } else {
                groups.push(CompactDecoderGroup::Block(CompactDecoderBlock {
                    owner,
                    source_fields: source_field..source_field + 1,
                    decoded_columns: decoded_column..decoded_column + 1,
                    final_columns: segment.final_columns.clone(),
                    width,
                    radix: segment.radix,
                    scale: segment.scale,
                }));
            }
        } else {
            groups.push(CompactDecoderGroup::Composite(CompactCompositeDecoder {
                owner,
                source_field,
                decoded_column,
                segments,
            }));
        }
    }
    groups
}

fn decoder_group_len(group: &CompactDecoderGroup) -> usize {
    match group {
        CompactDecoderGroup::Block(block) => block.decoded_columns.len(),
        CompactDecoderGroup::Composite(_) => 1,
    }
}

fn render_decoder_segments(segments: &[CompactDecoderSegment]) -> String {
    let values = segments
        .iter()
        .map(|segment| {
            format!(
                "{{ finalColumns := {}, radix := {}, scale := {} }}",
                lean_range(segment.final_columns.clone()),
                segment.radix,
                segment.scale,
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", values.join(", "))
}

fn render_decoder_groups(groups: &[CompactDecoderGroup]) -> String {
    let mut rendered = String::from("[\n");
    for group in groups {
        match group {
            CompactDecoderGroup::Block(block) => writeln!(
                rendered,
                "    DecoderGroup.block {{ owner := {:?}, sourceFields := {}, decodedColumns := {}, finalColumns := {}, width := {}, radix := {}, scale := {} }},",
                block.owner,
                lean_range(block.source_fields.clone()),
                lean_range(block.decoded_columns.clone()),
                lean_range(block.final_columns.clone()),
                block.width,
                block.radix,
                block.scale,
            ),
            CompactDecoderGroup::Composite(decoder) => writeln!(
                rendered,
                "    DecoderGroup.composite {{ owner := {:?}, sourceField := {}, decodedColumn := {}, segments := {} }},",
                decoder.owner,
                decoder.source_field,
                decoder.decoded_column,
                render_decoder_segments(&decoder.segments),
            ),
        }
        .expect("render terminal decoder group");
    }
    rendered.push_str("  ]");
    rendered
}

fn expected_decoder_row(
    binding: &NebulaFPrimeStreamingTerminalFieldBinding,
    decoded_column: usize,
    final_witness_column_start: usize,
) -> Vec<(usize, F)> {
    let terms = binding
        .decoder_terms()
        .iter()
        .map(|term| (final_witness_column_start + term.final_column(), term.coefficient()))
        .collect::<Vec<_>>();
    expected_linear_row(decoded_column, &terms)
}

fn expected_linear_row(output: usize, right: &[(usize, F)]) -> Vec<(usize, F)> {
    let mut terms = BTreeMap::<usize, F>::from([(output, F::ONE)]);
    for &(column, coefficient) in right {
        *terms.entry(column).or_insert(F::ZERO) -= coefficient;
    }
    terms.retain(|_, coefficient| *coefficient != F::ZERO);
    terms.into_iter().collect()
}

fn render_terminal_source_binding(fixture: StreamingTerminalAuditFixture) -> String {
    let family_name = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[0];
    let ranges = fixture
        .terminal
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == family_name)
        .map(|range| range.row_start..range.row_end)
        .collect::<Vec<_>>();
    let [family_rows] = ranges.as_slice() else {
        panic!("terminal source-binding family must have one row range")
    };
    let bindings = source_bindings(&fixture.profile);
    assert_eq!(family_rows.len(), bindings.len());
    let decoder_groups = compact_decoder_groups(&fixture);
    assert_eq!(
        decoder_groups.iter().map(decoder_group_len).sum::<usize>(),
        bindings.len(),
    );
    let final_witness_column_start = fixture.final_witness_column_start;
    let decoded_column_start = fixture.source_binding_decoded_column_start;
    let final_assignment_columns =
        final_witness_column_start..final_witness_column_start + fixture.profile.final_columns();
    let decoded_columns = decoded_column_start..decoded_column_start + bindings.len();
    let profile_id = fixture.profile.profile_id();
    let final_artifact_identity = fixture.profile.final_artifact_identity();
    let lifecycle_scope = fixture.profile.lifecycle_scope();
    let source = fixture.terminal.into_snapshot();
    for (index, (row, (_, binding))) in family_rows.clone().zip(bindings).enumerate() {
        assert_eq!(
            source.a_row(row),
            expected_decoder_row(binding, decoded_column_start + index, final_witness_column_start),
        );
        assert_eq!(source.b_row(row), &[(0, F::ONE)]);
        assert!(source.c_row(row).is_empty());
    }
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalSourceBindingSchema\n\n\
         /-! Generated compact certificate for the exact Rust terminal source-binding rows.\n\n\
         Emits constraints: no. Rust emits the rows reconstructed by these decoder groups.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBinding.Artifact\n\n\
         def decoderGroups : List DecoderGroup := {}\n\n\
         def rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := 1,\n    \
            profileId := {:?},\n    \
            sourceArtifactIdentity := \"rust:nightstream/streaming-terminal-lifecycle/source-rows/v1\",\n    \
            finalArtifactIdentity := {:?}, lifecycleScope := {:?},\n    \
            rowFamily := {:?}, rowStart := {}, rowStop := {},\n    \
            columnCount := {}, finalAssignmentColumns := {}, decodedColumns := {},\n    \
            decoderGroups := decoderGroups }}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding\n",
        render_decoder_groups(&decoder_groups),
        profile_id,
        final_artifact_identity,
        lifecycle_scope,
        family_name,
        family_rows.start,
        family_rows.end,
        source.cols(),
        lean_range(final_assignment_columns),
        lean_range(decoded_columns),
    )
}

fn terminal_source_binding_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(TERMINAL_SOURCE_BINDING_ARTIFACT_PATH)
}

fn render_terminal_full_x_out_context(fixture: StreamingTerminalAuditFixture) -> String {
    let family_name = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[2];
    let ranges = fixture
        .terminal
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == family_name)
        .map(|range| range.row_start..range.row_end)
        .collect::<Vec<_>>();
    let [family_rows] = ranges.as_slice() else {
        panic!("terminal full XOut context family must have one row range")
    };
    let x_out_columns = std::array::from_fn::<_, 32, _>(|index| fixture.source_binding_decoded_column_start + index);
    let mut definitions = Vec::<(usize, Vec<(usize, F)>)>::new();
    definitions.push((x_out_columns[0], vec![(0, F::from_u64(F_PRIME_STATE_X_OUT_DOMAIN))]));
    definitions.extend(
        x_out_columns[1..5]
            .iter()
            .copied()
            .zip(fixture.vk_fs_columns)
            .map(|(output, input)| (output, vec![(input, F::ONE)])),
    );
    definitions.extend(
        x_out_columns[5..9]
            .iter()
            .copied()
            .zip(fixture.pi_ccs_header_columns)
            .map(|(output, input)| (output, vec![(input, F::ONE)])),
    );
    for (index, value) in [
        (9, STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS as u64),
        (10, 0),
        (11, STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS as u64),
        (12, 0),
        (13, 1),
        (14, 0),
    ] {
        let terms = (value != 0)
            .then(|| vec![(0, F::from_u64(value))])
            .unwrap_or_default();
        definitions.push((x_out_columns[index], terms));
    }
    definitions.extend(
        x_out_columns[15..19]
            .iter()
            .copied()
            .zip(fixture.boundary_columns)
            .map(|(output, input)| (output, vec![(input, F::ONE)])),
    );
    definitions.extend(
        x_out_columns[23..27]
            .iter()
            .copied()
            .zip(fixture.accumulator_columns)
            .map(|(output, input)| (output, vec![(input, F::ONE)])),
    );
    definitions.push((
        x_out_columns[27],
        vec![(0, F::from_u64(TERMINAL_NEBULA_PRESENT_MARKER))],
    ));
    assert_eq!(definitions.len(), 24);
    assert_eq!(family_rows.len(), definitions.len());
    let source = fixture.terminal.into_snapshot();
    for (row, (output, terms)) in family_rows.clone().zip(&definitions) {
        assert_eq!(source.a_row(row), expected_linear_row(*output, terms));
        assert_eq!(source.b_row(row), &[(0, F::ONE)]);
        assert!(source.c_row(row).is_empty());
    }
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutContextSchema\n\n\
         /-! Generated exact full-layout Rust terminal XOut context geometry.\n\n\
         The empty SHA field is legacy diagnostic structure and is not authority.\n\
         Emits constraints: no. Rust emits the checked rows.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullXOutContext\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact\n\n\
         def lifecycleScope : String := \"recursive-terminal-arm-435\"\n\n\
         def rowStart : Nat := {}\n\n\
         def rowStop : Nat := {}\n\n\
         def rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := 1,\n    \
            profileId := \"nightstream/goldilocks/streaming-terminal-full-x-out-context/v1\",\n    \
            sourceIdentity := \"rust:nightstream/streaming-terminal-lifecycle/source-rows/v1\",\n    \
            sourceRowsSha256 := \"\", rowCount := 24, columnCount := {},\n    \
            domainTag := {}, acceptedWorkItems := {}, nebulaMarker := {},\n    \
            baselineChangedValue := 0, mutatedChangedValue := 1,\n    \
            xOutColumns := {:?},\n    \
            vkFsSourceColumns := {:?}, piCcsHeaderSourceColumns := {:?},\n    \
            boundarySourceColumns := {:?}, accumulatorSourceColumns := {:?} }}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullXOutContext\n",
        family_rows.start,
        family_rows.end,
        source.cols(),
        F_PRIME_STATE_X_OUT_DOMAIN,
        STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS,
        TERMINAL_NEBULA_PRESENT_MARKER,
        x_out_columns,
        fixture.vk_fs_columns,
        fixture.pi_ccs_header_columns,
        fixture.boundary_columns,
        fixture.accumulator_columns,
    )
}

fn terminal_full_x_out_context_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(TERMINAL_FULL_X_OUT_CONTEXT_ARTIFACT_PATH)
}

fn relocated_terms(
    terms: &[(usize, F)],
    external_columns: &BTreeMap<usize, usize>,
    source_internal_start: usize,
    full_internal_start: usize,
) -> Vec<(usize, F)> {
    let mut relocated = BTreeMap::<usize, F>::new();
    for &(column, coefficient) in terms {
        let target = if let Some(&target) = external_columns.get(&column) {
            target
        } else {
            assert!(
                column >= source_internal_start,
                "source column {column} has no relocation owner",
            );
            full_internal_start + (column - source_internal_start)
        };
        *relocated.entry(target).or_insert(F::ZERO) += coefficient;
    }
    relocated.retain(|_, coefficient| *coefficient != F::ZERO);
    relocated.into_iter().collect()
}

fn constant_row_output(source: &neo_fold_clean::engine::r1cs_circuit::R1csSnapshot, row: usize) -> usize {
    let outputs = source
        .a_row(row)
        .iter()
        .filter(|&&(column, coefficient)| column != 0 && coefficient == F::ONE)
        .map(|&(column, _)| column)
        .collect::<Vec<_>>();
    let [output] = outputs.as_slice() else {
        panic!("constant row must have one allocated output")
    };
    *output
}

fn render_terminal_full_phase_semantic(fixture: StreamingTerminalAuditFixture) -> String {
    let family_name = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[3];
    let full_ranges = fixture
        .terminal
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == family_name)
        .map(|range| range.row_start..range.row_end)
        .collect::<Vec<_>>();
    let [full_rows] = full_ranges.as_slice() else {
        panic!("terminal full phase-semantic family must have one row range")
    };

    let audit = streaming_terminal_x_out_authority_audit();
    let source_ranges = audit
        .row_families()
        .iter()
        .filter(|range| range.name == family_name)
        .map(|range| range.row_start..range.row_end)
        .collect::<Vec<_>>();
    let [source_rows] = source_ranges.as_slice() else {
        panic!("terminal audit phase-semantic family must have one row range")
    };
    assert_eq!(full_rows.len(), source_rows.len());

    let full_x_out = std::array::from_fn::<_, 32, _>(|index| fixture.source_binding_decoded_column_start + index);
    let full_local =
        std::array::from_fn::<_, 4, _>(|index| fixture.source_binding_decoded_column_start + 32 + 50 + index);
    let full_payload_start = fixture.source_binding_decoded_column_start + 32 + 50 + 4;
    let full_payload =
        (full_payload_start..full_payload_start + audit.delayed_payload_columns().len()).collect::<Vec<_>>();
    let mut external_columns = BTreeMap::<usize, usize>::from([(0, 0)]);
    external_columns.extend(audit.x_out_columns().into_iter().zip(full_x_out));
    external_columns.extend(audit.local_state_columns().into_iter().zip(full_local));
    external_columns.extend(
        audit
            .delayed_payload_columns()
            .iter()
            .copied()
            .zip(&full_payload)
            .map(|(source, &target)| (source, target)),
    );

    let source_internal_start = constant_row_output(audit.source(), source_rows.start);
    let full_source = fixture.terminal.into_snapshot();
    let full_internal_start = constant_row_output(&full_source, full_rows.start);
    for (source_row, full_row) in source_rows.clone().zip(full_rows.clone()) {
        assert_eq!(
            relocated_terms(
                audit.source().a_row(source_row),
                &external_columns,
                source_internal_start,
                full_internal_start,
            ),
            full_source.a_row(full_row),
        );
        assert_eq!(
            relocated_terms(
                audit.source().b_row(source_row),
                &external_columns,
                source_internal_start,
                full_internal_start,
            ),
            full_source.b_row(full_row),
        );
        assert_eq!(
            relocated_terms(
                audit.source().c_row(source_row),
                &external_columns,
                source_internal_start,
                full_internal_start,
            ),
            full_source.c_row(full_row),
        );
    }

    let constant_values = (0..11)
        .map(|index| {
            let coefficient = full_source
                .a_row(full_rows.start + index)
                .iter()
                .find(|&&(column, _)| column == 0)
                .map(|&(_, coefficient)| coefficient)
                .expect("phase-semantic constant row has a constant term");
            (-coefficient).as_canonical_u64()
        })
        .collect::<Vec<_>>();
    let x_out_semantic = full_x_out[19..23].to_vec();
    let hash_output_columns = (0..4)
        .map(|lane| {
            let x_out = x_out_semantic[lane];
            let columns = full_source
                .a_row(full_rows.end - 4 + lane)
                .iter()
                .filter(|&&(column, _)| column != x_out)
                .map(|&(column, _)| column)
                .collect::<Vec<_>>();
            let [output] = columns.as_slice() else {
                panic!("phase-semantic equality row must have one hash output")
            };
            *output
        })
        .collect::<Vec<_>>();
    let baseline_digest_value = full_source.witness()[hash_output_columns[0]].as_canonical_u64();
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticSchema\n\n\
         /-! Generated exact full-layout Rust terminal phase-semantic recipe.\n\n\
         Rust compares all rows with the authoritative audit recipe under the emitted relocation.\n\
         The empty SHA field is legacy diagnostic structure and is not authority.\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullPhaseSemantic\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact\n\n\
         def lifecycleScope : String := \"recursive-terminal-arm-435\"\n\n\
         def phaseConstantValues : List Nat := {:?}\n\n\
         def rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := 2,\n    \
            profileId := \"nightstream/goldilocks/streaming-terminal-full-phase-semantic/v1\",\n    \
            sourceIdentity := \"rust:nightstream/streaming-terminal-lifecycle/source-rows/v1\",\n    \
            sourceRowsSha256 := \"\", rowCount := {}, columnCount := {},\n    \
            sourceRowStart := {}, finalRowStart := {},\n    \
            constantValues := phaseConstantValues, constantStartColumn := {},\n    \
            localColumns := {:?}, payloadColumns := List.range' {} {},\n    \
            hashOutputColumns := {:?}, xOutSemanticColumns := {:?},\n    \
            baselineDigestValue := {}, equalityRowStart := {} }}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullPhaseSemantic\n",
        constant_values,
        full_rows.len(),
        full_source.cols(),
        full_rows.start,
        full_rows.start,
        full_internal_start,
        full_local,
        full_payload_start,
        full_payload.len(),
        hash_output_columns,
        x_out_semantic,
        baseline_digest_value,
        full_rows.len() - 4,
    )
}

fn terminal_full_phase_semantic_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(TERMINAL_FULL_PHASE_SEMANTIC_ARTIFACT_PATH)
}

fn variable_hash_row_count(constant_fields: usize, input_fields: usize) -> usize {
    constant_fields + 1 + input_fields + ((input_fields + 3) / 4) * 600 + 1 + 600
}

fn branch_constant_values(
    source: &neo_fold_clean::engine::r1cs_circuit::R1csSnapshot,
    row_start: usize,
    constant_start: usize,
    count: usize,
) -> Vec<u64> {
    (0..count)
        .map(|index| {
            let row = row_start + index;
            let output = constant_start + index;
            assert!(source.a_row(row).contains(&(output, F::ONE)));
            assert_eq!(source.b_row(row), &[(0, F::ONE)]);
            assert!(source.c_row(row).is_empty());
            source
                .a_row(row)
                .iter()
                .find(|&&(column, _)| column == 0)
                .map(|&(_, coefficient)| (-coefficient).as_canonical_u64())
                .unwrap_or(0)
        })
        .collect()
}

fn render_terminal_full_nebula_state_digest(fixture: StreamingTerminalAuditFixture) -> String {
    let family_name = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[4];
    let full_ranges = fixture
        .terminal
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == family_name)
        .map(|range| range.row_start..range.row_end)
        .collect::<Vec<_>>();
    let [full_rows] = full_ranges.as_slice() else {
        panic!("terminal full Nebula-state family must have one row range")
    };

    let audit = streaming_terminal_x_out_authority_audit();
    let source_ranges = audit
        .row_families()
        .iter()
        .filter(|range| range.name == family_name)
        .map(|range| range.row_start..range.row_end)
        .collect::<Vec<_>>();
    let [source_rows] = source_ranges.as_slice() else {
        panic!("terminal audit Nebula-state family must have one row range")
    };
    assert_eq!(full_rows.len(), source_rows.len());

    let full_x_out = std::array::from_fn::<_, 32, _>(|index| fixture.source_binding_decoded_column_start + index);
    let full_lane = std::array::from_fn::<_, 50, _>(|index| fixture.source_binding_decoded_column_start + 32 + index);
    let mut external_columns = BTreeMap::<usize, usize>::from([(0, 0)]);
    external_columns.extend(audit.x_out_columns().into_iter().zip(full_x_out));
    external_columns.extend(audit.post_phase_lane_columns().into_iter().zip(full_lane));

    let source_internal_start = constant_row_output(audit.source(), source_rows.start + 1);
    let full_source = fixture.terminal.into_snapshot();
    let full_internal_start = constant_row_output(&full_source, full_rows.start + 1);
    for (source_row, full_row) in source_rows.clone().zip(full_rows.clone()) {
        assert_eq!(
            relocated_terms(
                audit.source().a_row(source_row),
                &external_columns,
                source_internal_start,
                full_internal_start,
            ),
            full_source.a_row(full_row),
        );
        assert_eq!(
            relocated_terms(
                audit.source().b_row(source_row),
                &external_columns,
                source_internal_start,
                full_internal_start,
            ),
            full_source.b_row(full_row),
        );
        assert_eq!(
            relocated_terms(
                audit.source().c_row(source_row),
                &external_columns,
                source_internal_start,
                full_internal_start,
            ),
            full_source.c_row(full_row),
        );
    }

    let absent_constant_fields = 13;
    let absent_input_fields = 58;
    let present_constant_fields = 10;
    let present_input_fields = 59;
    let absent_row_start = 1;
    let absent_hash_rows = variable_hash_row_count(absent_constant_fields, absent_input_fields);
    let present_row_start = absent_row_start + absent_hash_rows;
    let present_hash_rows = variable_hash_row_count(present_constant_fields, present_input_fields);
    let mux_row_start = present_row_start + present_hash_rows;
    let equality_row_start = mux_row_start + 4;
    assert_eq!(equality_row_start + 4, full_rows.len());

    let absent_constant_start = full_internal_start;
    let present_constant_start = constant_row_output(&full_source, full_rows.start + present_row_start);
    let absent_constant_values = branch_constant_values(
        &full_source,
        full_rows.start + absent_row_start,
        absent_constant_start,
        absent_constant_fields,
    );
    let present_constant_values = branch_constant_values(
        &full_source,
        full_rows.start + present_row_start,
        present_constant_start,
        present_constant_fields,
    );
    let absent_constants = (absent_constant_start..absent_constant_start + absent_constant_fields).collect::<Vec<_>>();
    let present_constants =
        (present_constant_start..present_constant_start + present_constant_fields).collect::<Vec<_>>();

    let mut absent_input_columns = Vec::with_capacity(absent_input_fields);
    absent_input_columns.extend_from_slice(&absent_constants[0..7]);
    absent_input_columns.extend_from_slice(&full_lane[0..4]);
    absent_input_columns.extend_from_slice(&full_lane[5..8]);
    absent_input_columns.extend_from_slice(&full_lane[20..22]);
    absent_input_columns.extend_from_slice(&absent_constants[7..13]);
    absent_input_columns.extend_from_slice(&full_lane[12..20]);
    absent_input_columns.extend_from_slice(&full_lane[22..50]);
    assert_eq!(absent_input_columns.len(), absent_input_fields);

    let mut present_input_columns = Vec::with_capacity(present_input_fields);
    present_input_columns.extend_from_slice(&present_constants[0..7]);
    present_input_columns.extend_from_slice(&full_lane[0..4]);
    present_input_columns.extend_from_slice(&full_lane[5..8]);
    present_input_columns.extend_from_slice(&full_lane[20..22]);
    present_input_columns.extend_from_slice(&present_constants[7..9]);
    present_input_columns.extend_from_slice(&full_lane[8..12]);
    present_input_columns.push(present_constants[9]);
    present_input_columns.extend_from_slice(&full_lane[12..20]);
    present_input_columns.extend_from_slice(&full_lane[22..50]);
    assert_eq!(present_input_columns.len(), present_input_fields);

    let mut absent_output_columns = Vec::with_capacity(4);
    let mut present_output_columns = Vec::with_capacity(4);
    let mut hash_output_columns = Vec::with_capacity(4);
    for lane in 0..4 {
        let row = full_rows.start + mux_row_start + lane;
        absent_output_columns.push(
            full_source
                .b_row(row)
                .iter()
                .find(|&&(_, coefficient)| coefficient == -F::ONE)
                .map(|&(column, _)| column)
                .expect("mux row absent output"),
        );
        present_output_columns.push(
            full_source
                .b_row(row)
                .iter()
                .find(|&&(_, coefficient)| coefficient == F::ONE)
                .map(|&(column, _)| column)
                .expect("mux row present output"),
        );
        hash_output_columns.push(
            full_source
                .c_row(row)
                .iter()
                .find(|&&(_, coefficient)| coefficient == F::ONE)
                .map(|&(column, _)| column)
                .expect("mux row selected output"),
        );
    }
    let x_out_state_columns = full_x_out[28..32].to_vec();
    let baseline_digest_value = full_source.witness()[hash_output_columns[0]].as_canonical_u64();
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestLinkSchema\n\n\
         /-! Generated exact full-layout Rust terminal Nebula-state-digest recipe.\n\n\
         Rust compares all rows with the authoritative audit recipe under the emitted relocation.\n\
         The empty SHA field is legacy diagnostic structure and is not authority.\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullNebulaStateDigest\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact\n\n\
         def lifecycleScope : String := \"recursive-terminal-arm-435\"\n\n\
         def absentConstantValues : List Nat := {:?}\n\n\
         def presentConstantValues : List Nat := {:?}\n\n\
         def rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := 2,\n    \
            profileId := \"nightstream/goldilocks/streaming-terminal-full-nebula-state-digest/v1\",\n    \
            sourceIdentity := \"rust:nightstream/streaming-terminal-lifecycle/source-rows/v1\",\n    \
            sourceRowsSha256 := \"\", rowCount := {}, columnCount := {},\n    \
            sourceRowStart := {}, finalRowStart := {}, openColumn := {},\n    \
            absentConstantValues := absentConstantValues, absentConstantStartColumn := {},\n    \
            absentInputColumns := {:?}, absentOutputColumns := {:?},\n    \
            presentConstantValues := presentConstantValues, presentConstantStartColumn := {},\n    \
            presentInputColumns := {:?}, presentOutputColumns := {:?},\n    \
            hashOutputColumns := {:?}, xOutStateColumns := {:?},\n    \
            baselineDigestValue := {}, absentRowStart := {}, presentRowStart := {},\n    \
            muxRowStart := {}, equalityRowStart := {}, selectedSourceRow := {} }}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullNebulaStateDigest\n",
        absent_constant_values,
        present_constant_values,
        full_rows.len(),
        full_source.cols(),
        full_rows.start,
        full_rows.start,
        full_lane[4],
        absent_constant_start,
        absent_input_columns,
        absent_output_columns,
        present_constant_start,
        present_input_columns,
        present_output_columns,
        hash_output_columns,
        x_out_state_columns,
        baseline_digest_value,
        absent_row_start,
        present_row_start,
        mux_row_start,
        equality_row_start,
        full_rows.start + equality_row_start,
    )
}

fn terminal_full_nebula_state_digest_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(TERMINAL_FULL_NEBULA_STATE_DIGEST_ARTIFACT_PATH)
}

fn render_terminal_profile_selection(fixture: StreamingTerminalAuditFixture) -> String {
    let family_name = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[1];
    let ranges = fixture
        .terminal
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == family_name)
        .map(|range| range.row_start..range.row_end)
        .collect::<Vec<_>>();
    let [family_rows] = ranges.as_slice() else {
        panic!("terminal profile-selection family must have one row range")
    };
    let selectors = [
        fixture.schedule_selector_column,
        fixture.lifecycle_selector_column,
        fixture.phase_selector_column,
    ];
    assert_eq!(family_rows.len(), selectors.len());
    let source = fixture.terminal.into_snapshot();
    for (row, selector) in family_rows.clone().zip(selectors) {
        assert_eq!(source.a_row(row), &[(0, -F::ONE), (selector, F::ONE)]);
        assert_eq!(source.b_row(row), &[(0, F::ONE)]);
        assert!(source.c_row(row).is_empty());
    }
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalProfileSelectionSchema\n\n\
         /-! Generated exact Rust rows for terminal profile selection.\n\n\
         Emits constraints: no. Rust emits the described rows.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfileSelection\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfileSelection.Artifact\n\n\
         def rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := 1,\n    \
            profileId := \"nightstream/goldilocks/streaming-terminal-lifecycle/v1\",\n    \
            sourceArtifactIdentity := \"rust:nightstream/streaming-terminal-lifecycle/source-rows/v1\",\n    \
            lifecycleScope := \"recursive-terminal-arm-435\",\n    \
            rowFamily := {:?}, rowStart := {}, rowStop := {},\n    \
            columnCount := {}, selectorColumns := {:?} }}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfileSelection\n",
        family_name,
        family_rows.start,
        family_rows.end,
        source.cols(),
        selectors,
    )
}

fn terminal_profile_selection_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(TERMINAL_PROFILE_SELECTION_ARTIFACT_PATH)
}

#[test]
#[ignore = "exact full-layout terminal XOut context rows"]
fn terminal_full_x_out_context_artifact_is_current() {
    let rendered = render_terminal_full_x_out_context(build_streaming_terminal_audit_fixture());
    let path = terminal_full_x_out_context_artifact_path();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!(
            "terminal full XOut context artifact drifted; inspect {}",
            path.display()
        );
    }
}

#[test]
#[ignore = "exact full-layout terminal phase-semantic rows"]
fn terminal_full_phase_semantic_artifact_is_current() {
    let rendered = render_terminal_full_phase_semantic(build_streaming_terminal_audit_fixture());
    let path = terminal_full_phase_semantic_artifact_path();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!(
            "terminal full phase-semantic artifact drifted; inspect {}",
            path.display()
        );
    }
}

#[test]
#[ignore = "exact full-layout terminal Nebula-state-digest rows"]
fn terminal_full_nebula_state_digest_artifact_is_current() {
    let rendered = render_terminal_full_nebula_state_digest(build_streaming_terminal_audit_fixture());
    let path = terminal_full_nebula_state_digest_artifact_path();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!(
            "terminal full Nebula-state-digest artifact drifted; inspect {}",
            path.display()
        );
    }
}

#[test]
#[ignore = "exact full-layout terminal Nebula program-binding rows"]
fn terminal_full_program_binding_artifact_is_current() {
    let rendered = streaming_terminal_program_binding_artifact::render(build_streaming_terminal_audit_fixture());
    let path = streaming_terminal_program_binding_artifact::artifact_path();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!(
            "terminal full program-binding artifact drifted; inspect {}",
            path.display()
        );
    }
}

#[test]
#[ignore = "exact full-layout terminal Nebula finalizer rows"]
fn terminal_full_finalizer_artifact_is_current() {
    let rendered = streaming_terminal_finalizer_artifact::render(build_streaming_terminal_audit_fixture());
    let path = streaming_terminal_finalizer_artifact::artifact_path();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!("terminal full finalizer artifact drifted; inspect {}", path.display());
    }
}

#[test]
#[ignore = "exact terminal source-binding rows"]
fn terminal_source_binding_artifact_is_current() {
    let rendered = render_terminal_source_binding(build_streaming_terminal_audit_fixture());
    let path = terminal_source_binding_artifact_path();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!("terminal source-binding artifact drifted; inspect {}", path.display());
    }
}

#[test]
#[ignore = "exact terminal profile-selection rows"]
fn terminal_profile_selection_artifact_is_current() {
    let rendered = render_terminal_profile_selection(build_streaming_terminal_audit_fixture());
    let path = terminal_profile_selection_artifact_path();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!(
            "terminal profile-selection artifact drifted; inspect {}",
            path.display()
        );
    }
}

#[test]
#[ignore = "exact terminal source-to-final profile"]
fn terminal_profile_artifact_is_current() {
    let fixture = build_streaming_terminal_audit_fixture();
    let rendered = render_terminal_profile(&fixture.profile);
    let path = terminal_profile_artifact_path();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!("terminal profile Lean artifact drifted; inspect {}", path.display());
    }
}

#[test]
#[ignore = "expensive exact lifecycle selective-lowering profile"]
fn terminal_profile_binds_the_trailing_fresh_after_state_to_final_rows() {
    let mut fixture = build_streaming_terminal_audit_fixture();
    let cases: [(usize, fn(F) -> F, &str); 6] = [
        (
            fixture.schedule_selector_column,
            |_| F::ZERO,
            "terminal schedule selector",
        ),
        (
            fixture.verifier_key_column,
            |value| value + F::ONE,
            "verifier-key digest",
        ),
        (
            fixture.program_binding_column,
            |value| value + F::ONE,
            "Nebula program binding",
        ),
        (
            fixture.delayed_payload_column,
            |value| F::ONE - value,
            "delayed payload",
        ),
        (fixture.fresh_adv_column, |value| value + F::ONE, "fresh adv opening"),
        (
            fixture.final_closed_lane_column,
            |_| F::from_u64(2),
            "final closed lane",
        ),
    ];
    for (column, change, label) in cases {
        let original = fixture.terminal.witness()[column];
        fixture.terminal.tamper_witness(column, change(original));
        assert!(!fixture.terminal.is_satisfied(), "changed {label} must reject");
        fixture.terminal.tamper_witness(column, original);
        assert!(fixture.terminal.is_satisfied(), "restored {label} must satisfy");
    }
}

#[test]
#[ignore = "exact terminal XOut hash source-to-final provenance"]
fn terminal_x_out_hash_source_to_final_provenance_is_exact() {
    let fixture = build_streaming_terminal_audit_fixture();
    let provenance = streaming_terminal_x_out_hash_artifact::audit(&fixture);
    assert!(!provenance.source_rows().is_empty());
    assert_eq!(
        provenance.emitted_rows().len(),
        provenance.projected().row_artifacts().len()
    );
}

#[test]
#[ignore = "bounded first terminal XOut Poseidon2 leaf shape"]
fn terminal_x_out_first_poseidon2_leaf_shape() {
    let fixture = build_streaming_terminal_audit_fixture();
    let provenance = streaming_terminal_x_out_hash_artifact::audit(&fixture);
    let projected = provenance.projected();
    let source = projected
        .source_provenance()
        .expect("terminal XOut hash complete source provenance");
    let first_step = source
        .poseidon2_sbox_steps()
        .iter()
        .min_by_key(|step| step.emitted_row())
        .expect("first terminal XOut Poseidon2 S-box step");
    let rewrite_id = first_step.rewrite_id();
    let steps = source
        .poseidon2_sbox_steps()
        .iter()
        .filter(|step| step.rewrite_id() == rewrite_id)
        .collect::<Vec<_>>();
    assert_eq!(steps.len(), 86);

    let first_round = &fixture
        .lifecycle
        .after_x_out_hash_audit(
            neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeStreamingLifecycleArm::Recursive,
        )
        .hash()
        .rounds[0];
    let local_columns = first_round.first_allocated_column..first_round.first_allocated_column + 600;
    let external_source_columns = steps
        .iter()
        .flat_map(|step| step.input().terms().iter().chain(step.output().terms()))
        .map(|term| term.column())
        .filter(|column| !local_columns.contains(column))
        .collect::<std::collections::BTreeSet<_>>();
    let emitted_rows = steps
        .iter()
        .map(|step| step.emitted_row())
        .collect::<std::collections::BTreeSet<_>>();
    let rows = projected
        .row_artifacts()
        .iter()
        .filter(|row| emitted_rows.contains(&row.emitted_row()))
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 86);
    assert!(rows.iter().flat_map(|row| row.ports()).all(|port| {
        port.seeded_blocks().is_empty()
            && port
                .geometric_runs()
                .iter()
                .all(|run| run.length() == 41 && run.ratio() == F::from_u64(3))
    }));
    let explicit_columns = rows
        .iter()
        .flat_map(|row| row.ports())
        .flat_map(|port| port.explicit())
        .map(|term| term.column())
        .collect::<std::collections::BTreeSet<_>>();
    let geometric_intervals = rows
        .iter()
        .flat_map(|row| row.ports())
        .flat_map(|port| port.geometric_runs())
        .map(|run| (run.column_start(), run.length()))
        .collect::<std::collections::BTreeSet<_>>();
    let external_images = source
        .requested_source_images()
        .iter()
        .filter(|image| external_source_columns.contains(&image.column()))
        .collect::<Vec<_>>();
    let expected_external_columns = [
        first_round.state_before_cols[0],
        first_round.permutation_input_cols[0],
        first_round.permutation_input_cols[1],
        first_round.permutation_input_cols[2],
        first_round.permutation_input_cols[3],
    ]
    .into_iter()
    .collect::<std::collections::BTreeSet<_>>();
    let expected_geometric_intervals = [766, 807, 848]
        .into_iter()
        .map(|start| (start, 41))
        .chain((0..86).map(|index| (22_023_158 + index * 41, 41)))
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(projected.selector_columns(), [648, 649]);
    assert_eq!(first_step.source_rows(), &[(30_658_245, 30_658_845)]);
    assert_eq!(local_columns, 30_382_570..30_383_170);
    assert_eq!(external_source_columns, expected_external_columns);
    assert_eq!(explicit_columns, [0, 649].into_iter().collect());
    assert_eq!(geometric_intervals, expected_geometric_intervals);
    assert_eq!(
        external_images
            .iter()
            .map(|image| image.column())
            .collect::<std::collections::BTreeSet<_>>(),
        expected_external_columns
    );
    assert!(external_images.iter().all(|image| {
        image.port().seeded_blocks().is_empty()
            && image
                .port()
                .geometric_runs()
                .iter()
                .all(|run| run.length() == 41 && run.ratio() == F::from_u64(3))
    }));
}

#[test]
#[ignore = "exact first terminal XOut Poseidon2 leaf artifacts"]
fn terminal_x_out_first_poseidon2_leaf_artifacts_are_current() {
    let fixture = build_streaming_terminal_audit_fixture();
    let mut drifted = Vec::new();
    for (path, rendered) in streaming_terminal_x_out_hash_artifact::first_leaf_artifacts(&fixture) {
        if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
            drifted.push(path);
        }
    }
    if !drifted.is_empty() {
        panic!(
            "first terminal XOut Poseidon2 leaf artifacts drifted; inspect {}",
            drifted
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

#[test]
#[ignore = "exact recursive-terminal XOut public-hash artifact"]
fn terminal_x_out_public_hash_artifact_is_current() {
    let fixture = build_streaming_terminal_audit_fixture();
    let rendered = streaming_terminal_x_out_hash_artifact::render(&fixture);
    let path = streaming_terminal_x_out_hash_artifact::artifact_path();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        panic!("terminal XOut public-hash artifact drifted; inspect {}", path.display());
    }
}
