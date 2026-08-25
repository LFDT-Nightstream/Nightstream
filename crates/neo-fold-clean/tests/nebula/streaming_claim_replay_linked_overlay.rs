//! Exact final-assignment checks for the production claim-replay leaf.

#[path = "streaming_claim_replay_linked_overlay/conformance.rs"]
mod conformance;
#[path = "streaming_claim_replay_linked_overlay/render.rs"]
mod render;
#[path = "../support/selective_decoder_run_lean.rs"]
mod selective_decoder_run_lean;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::ops::Range;
use std::path::{Path, PathBuf};

use neo_fold_clean::frontends::nebula::f_prime::{
    build_production_claim_replay_base_low_norm_r1cs, build_production_claim_replay_linked_overlay_low_norm_r1cs,
    production_claim_active_coordinate_overlay_base_kind_map,
    production_claim_active_coordinate_overlay_compact_layout_and_decoder_runs_for_ranges,
    production_claim_active_coordinate_overlay_links,
    production_claim_active_coordinate_overlay_nonseeded_row_projection,
    production_claim_active_coordinate_overlay_seeded_placements,
    production_claim_replay_base_compact_layout_and_decoder_runs_for_ranges, production_claim_replay_base_phase_kinds,
    production_claim_replay_base_retained_row_projection, production_claim_replay_base_semantic_row_projection,
    production_pi_rlc_family_body_projected_rows_with_source_provenance, NebulaFPrimeClaimCoordinateOverlaySynthesis,
    NebulaFPrimeClaimReplayArmKind, NebulaFPrimeClaimReplaySynthesis, NebulaFPrimeStreamingProgramAudit,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    LinkedOverlayLowNormR1cs, SelectiveEmittedRowFamily, SelectiveLinearDefinitionAudit,
    SelectiveProjectedDecoderRunProvenance, SelectiveProjectedPort, SelectiveProjectedRowsAudit,
    SelectiveProjectedSourceResolutionRun, SelectiveSourceRowDisposition,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use render::{
    canonical_poseidon_call, lean_compact_port, lean_decoder_instances, lean_linear_definition,
    lean_source_linear_combination, semantic_source_map, write_source_map,
};
use selective_decoder_run_lean::write_runs;
use sha2::{Digest, Sha256};

const ACTIVE_CHUNKS: usize = 98;
const FULL_CHUNKS: usize = 97;
const PROFILE_ID: &str = "nebula-f-prime-streaming-claim-replay-goldilocks-b2-k16-v6";
const ARTIFACT_ID: &str = "rust:nightstream/streaming-selective-ccs/claim-replay-linked-overlay/v1";
const BASE_ARTIFACT_SHA256: &str = "fc5e19007da5e0b496de5570bcf92f73c4d24b770f7c278b1bcf53384788641b";
const OVERLAY_ARTIFACT_SHA256: &str = "0c6025a481ce873cc186b03cfe19cd5a8968b3207cc2db8214b5ebd80e095bf0";
const REPLAY_INITIAL_CAPACITY_SOURCE_START: usize = 654;
const REPLAY_FRESH_RATE_SOURCE_START: usize = 1333;
const REPLAY_RATE_WIDTH: usize = 4;
const REPLAY_CHAINED_CAPACITY_OFFSET: usize = 596;
const FINAL_REPLAY_TAIL_SOURCE_START: usize = REPLAY_FRESH_RATE_SOURCE_START + 143 * REPLAY_RATE_WIDTH;
const FINAL_REPLAY_TAIL_FIELDS: usize = 3;

fn evaluate_decoder(terms: &[(usize, F)], assignment: &[F]) -> F {
    terms.iter().fold(F::ZERO, |sum, &(column, coefficient)| {
        sum + coefficient * assignment[column]
    })
}

fn evaluate_relation_row(relation: &LinkedOverlayLowNormR1cs, row: usize, assignment: &[F]) -> F {
    let point = relation
        .structure()
        .matrices
        .iter()
        .map(|matrix| {
            matrix
                .materialize_row(row)
                .expect("linked-overlay row is in range")
                .into_iter()
                .fold(F::ZERO, |sum, (column, coefficient)| {
                    sum + coefficient * assignment[column]
                })
        })
        .collect::<Vec<_>>();
    relation.structure().f.eval(&point)
}

fn direct_slot_terms(start: usize, width: usize) -> Vec<(usize, F)> {
    let radix = match width {
        41 => F::from_u64(3),
        23 => F::from_u64(7),
        1..=64 => F::from_u64(2),
        _ => panic!("unsupported direct decoder width {width}"),
    };
    let mut coefficient = F::ONE;
    (start..start + width)
        .map(|column| {
            let term = (column, coefficient);
            coefficient *= radix;
            term
        })
        .collect()
}

fn expand_projected_port(port: &SelectiveProjectedPort) -> Vec<(usize, F)> {
    assert!(
        port.seeded_blocks().is_empty(),
        "retained claim-replay rows must not contain seeded final ports"
    );
    let mut terms = BTreeMap::<usize, F>::new();
    for term in port.explicit() {
        *terms.entry(term.column()).or_insert(F::ZERO) += term.coefficient();
    }
    for run in port.geometric_runs() {
        let mut coefficient = run.initial();
        for column in run.column_start()..run.column_start() + run.length() {
            *terms.entry(column).or_insert(F::ZERO) += coefficient;
            coefficient *= run.ratio();
        }
    }
    terms.retain(|_, coefficient| *coefficient != F::ZERO);
    terms.into_iter().collect()
}

fn assert_retained_projection_matches_final_rows(
    relation: &LinkedOverlayLowNormR1cs,
    projection: &SelectiveProjectedRowsAudit,
) {
    assert_eq!(projection.rows(), relation.base_relation().structure().n);
    assert_eq!(projection.columns(), relation.base_relation().structure().m);
    let base_rows = relation.layout().base_rows();
    for compact in projection.row_artifacts() {
        let row = compact.emitted_row();
        assert!(base_rows.contains(&row));
        for (port, projected) in compact.ports().iter().enumerate() {
            let materialized = relation.structure().matrices[port]
                .materialize_row(row)
                .expect("retained row is present in the final linked relation");
            assert_eq!(
                expand_projected_port(projected),
                materialized,
                "retained base row {row}, port {port} must equal the exact final linked row"
            );
        }
    }
}

fn active_replay_decoders() -> Vec<SelectiveProjectedDecoderRunProvenance> {
    let requests = [(0, 2357..155957), (1, 2357..88157)];
    let (_, decoders) = production_claim_replay_base_compact_layout_and_decoder_runs_for_ranges(&requests)
        .expect("audit exact active replay-call decoder ranges");
    assert_eq!(decoders.len(), 2);
    let full_template = &decoders[0].repeated_templates()[0];
    let final_template = &decoders[1].repeated_templates()[0];
    assert_eq!(full_template.source_width(), 600);
    assert_eq!(final_template.source_width(), 600);
    assert_eq!(full_template.relative_runs(), final_template.relative_runs());
    assert_eq!(full_template.instances().len(), 1);
    assert_eq!(final_template.instances().len(), 1);
    assert!(decoders
        .iter()
        .all(|decoder| decoder.residual_strided_runs().is_empty()));
    decoders
}

fn active_replay_external_decoders(
    relation: &LinkedOverlayLowNormR1cs,
) -> (
    Vec<SelectiveProjectedDecoderRunProvenance>,
    Vec<SelectiveLinearDefinitionAudit>,
) {
    let requests = [
        (
            0,
            REPLAY_INITIAL_CAPACITY_SOURCE_START..REPLAY_INITIAL_CAPACITY_SOURCE_START + REPLAY_RATE_WIDTH,
        ),
        (
            0,
            REPLAY_FRESH_RATE_SOURCE_START..REPLAY_FRESH_RATE_SOURCE_START + 256 * REPLAY_RATE_WIDTH,
        ),
        (
            1,
            REPLAY_INITIAL_CAPACITY_SOURCE_START..REPLAY_INITIAL_CAPACITY_SOURCE_START + REPLAY_RATE_WIDTH,
        ),
        (
            1,
            REPLAY_FRESH_RATE_SOURCE_START..REPLAY_FRESH_RATE_SOURCE_START + 143 * REPLAY_RATE_WIDTH,
        ),
        (
            1,
            FINAL_REPLAY_TAIL_SOURCE_START..FINAL_REPLAY_TAIL_SOURCE_START + FINAL_REPLAY_TAIL_FIELDS,
        ),
    ];
    let (_, decoders) = production_claim_replay_base_compact_layout_and_decoder_runs_for_ranges(&requests)
        .expect("audit exact external replay-input decoder ranges");
    assert_eq!(decoders.len(), requests.len());
    for (index, (decoder, (arm, source_range))) in decoders.iter().zip(requests).enumerate() {
        assert_eq!(decoder.arm(), arm);
        assert_eq!(decoder.source_range(), source_range);
        assert!(decoder.repeated_templates().is_empty());
        if index + 1 == decoders.len() {
            assert_eq!(decoder.runs().len(), 1);
            let run = decoder.runs()[0];
            assert_eq!(run.source_start(), FINAL_REPLAY_TAIL_SOURCE_START);
            assert_eq!(run.length(), FINAL_REPLAY_TAIL_FIELDS);
            assert_eq!(
                run.resolution(),
                SelectiveProjectedSourceResolutionRun::LinearDefinition
            );
        } else {
            assert!(decoder.runs().iter().all(|run| matches!(
                run.resolution(),
                SelectiveProjectedSourceResolutionRun::Direct { width: 41, .. }
            )));
        }
    }
    let tail_definitions = (FINAL_REPLAY_TAIL_SOURCE_START..FINAL_REPLAY_TAIL_SOURCE_START + FINAL_REPLAY_TAIL_FIELDS)
        .map(|target| {
            relation
                .base_relation()
                .selective_compiler_audit()
                .expect("production base retains its exact compiler audit")
                .source_arm_linear_definition(1, target)
                .expect("final replay tail has one exact compiler definition")
                .clone()
        })
        .collect::<Vec<_>>();
    assert_eq!(tail_definitions.len(), FINAL_REPLAY_TAIL_FIELDS);
    for (offset, definition) in tail_definitions.iter().enumerate() {
        assert_eq!(definition.source_row(), Some(86405 + offset));
        assert_eq!(definition.target(), FINAL_REPLAY_TAIL_SOURCE_START + offset);
        assert_eq!(definition.constant(), F::ZERO);
        assert_eq!(definition.terms().len(), 1);
        assert_eq!(definition.terms()[0].column(), 996 + offset);
        assert_eq!(definition.terms()[0].coefficient(), F::ONE);
    }
    (decoders, tail_definitions)
}

fn assert_active_replay_input_schedule(
    projected: &SelectiveProjectedRowsAudit,
    instances: neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedSourceDecoderTemplateInstances,
) -> usize {
    let source = projected
        .source_provenance()
        .expect("semantic row projection has complete source provenance");
    let steps = source.poseidon2_sbox_steps();
    assert_eq!(steps.len(), instances.count() * 86);
    let rows = projected
        .row_artifacts()
        .iter()
        .filter(|row| row.family() == SelectiveEmittedRowFamily::Poseidon2)
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), steps.len());
    let final_row_start = rows
        .first()
        .expect("active replay has one Poseidon2 row")
        .emitted_row();
    for (offset, row) in rows.iter().enumerate() {
        assert_eq!(row.emitted_row(), final_row_start + offset);
    }
    for call_index in 0..instances.count() {
        let source_start = instances.source_start() + call_index * instances.source_stride();
        let source_stop = source_start + 600;
        let external = steps[call_index * 86..(call_index + 1) * 86]
            .iter()
            .flat_map(|step| step.input().terms().iter().chain(step.output().terms()))
            .map(|term| term.column())
            .filter(|column| !(source_start..source_stop).contains(column))
            .collect::<BTreeSet<_>>();
        let fresh_start = REPLAY_FRESH_RATE_SOURCE_START + call_index * REPLAY_RATE_WIDTH;
        let mut expected = (fresh_start..fresh_start + REPLAY_RATE_WIDTH).collect::<BTreeSet<_>>();
        let capacity_start = if call_index == 0 {
            REPLAY_INITIAL_CAPACITY_SOURCE_START
        } else {
            source_start - (600 - REPLAY_CHAINED_CAPACITY_OFFSET)
        };
        expected.extend(capacity_start..capacity_start + REPLAY_RATE_WIDTH);
        assert_eq!(
            external, expected,
            "claim-replay call {call_index} must use the exact fresh-rate and chained-capacity sources"
        );
    }
    final_row_start
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn lean_range(range: Range<usize>) -> String {
    format!("{{ start := {}, stop := {} }}", range.start, range.end)
}

fn lean_nat_list(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[derive(Clone, Debug)]
struct RetainedRowOwner {
    suffix: &'static str,
    rows: Range<usize>,
}

fn retained_row_owners(synthesis: NebulaFPrimeClaimReplaySynthesis) -> Vec<RetainedRowOwner> {
    let kind = synthesis.kind();
    let lowered = synthesis
        .into_lowered_for_artifact()
        .expect("lower production claim-replay base for retained ownership");
    let shape = lowered.shape();
    let mut owners = shape
        .row_family_ranges()
        .iter()
        .filter_map(|family| {
            let suffix = match family.name {
                "nebula.streaming.claim_replay.expected_carry" => "ExpectedCarry",
                "nebula.streaming.claim_replay.state_pins" => "StatePins",
                "nebula.streaming.claim_replay.cursors" => "Cursors",
                "nebula.streaming.claim_replay.replay_output" => "ReplayOutput",
                "nebula.streaming.claim_replay.final_readiness" => "FinalReadiness",
                "nebula.streaming.claim_replay.final_absorbed" => "FinalAbsorbed",
                _ => return None,
            };
            Some(RetainedRowOwner {
                suffix,
                rows: family.row_start..family.row_end,
            })
        })
        .collect::<Vec<_>>();
    let expected_family_count = match kind {
        NebulaFPrimeClaimReplayArmKind::Full => 4,
        NebulaFPrimeClaimReplayArmKind::Final => 6,
    };
    assert_eq!(
        owners.len(),
        expected_family_count,
        "claim-replay retained semantic family ownership"
    );
    owners.extend(shape.physical_stage_ranges().iter().map(|stage| {
        let suffix = match stage.path() {
            "nebula.streaming.claim_replay.state_words" => "StateWordsOther",
            "nebula.streaming.claim_replay.chunk" => "ChunkOther",
            "nebula.streaming.claim_replay.state" => "StateOther",
            "nebula.streaming.claim_replay.poseidon2" => "PoseidonOther",
            "nebula.streaming.claim_replay.ready" => "ReadyOther",
            "nebula.streaming.claim_replay.state_digest" => "StateDigestOther",
            path => panic!("unexpected claim-replay physical stage {path}"),
        };
        RetainedRowOwner {
            suffix,
            rows: stage.rows(),
        }
    }));
    owners
}

fn retained_row_partitions<'a>(
    projection: &SelectiveProjectedRowsAudit,
    owners: &'a [RetainedRowOwner],
) -> Vec<(&'a str, Vec<usize>)> {
    let source = projection
        .source_provenance()
        .expect("retained projection has complete source provenance");
    let mut partitions = Vec::<(&str, Vec<usize>)>::new();
    for (index, step) in source.retained_steps().iter().enumerate() {
        let suffix = owners
            .iter()
            .find(|owner| owner.rows.contains(&step.source_row()))
            .map(|owner| owner.suffix)
            .expect("retained source row has semantic-family or physical-stage ownership");
        if let Some((previous, indices)) = partitions.last_mut() {
            if *previous == suffix {
                indices.push(index);
                continue;
            }
        }
        assert!(
            partitions.iter().all(|(previous, _)| *previous != suffix),
            "retained source-row owner must be one contiguous partition"
        );
        partitions.push((suffix, vec![index]));
    }
    assert_eq!(
        partitions
            .iter()
            .flat_map(|(_, indices)| indices.iter().copied())
            .collect::<Vec<_>>(),
        (0..source.retained_steps().len()).collect::<Vec<_>>(),
        "retained source-row partitions must have exact ordered coverage"
    );
    partitions
}

fn write_retained_rows(
    rendered: &mut String,
    definition: &str,
    projection: &SelectiveProjectedRowsAudit,
    indices: &[usize],
    final_rows: usize,
    final_columns: usize,
) {
    let source = projection
        .source_provenance()
        .expect("retained projection has complete source provenance");
    writeln!(rendered, "def {definition} : List RawCompactRow :=\n  [").expect("render retained row header");
    for (position, &index) in indices.iter().enumerate() {
        let row = &projection.row_artifacts()[index];
        let step = &source.retained_steps()[index];
        assert_eq!(row.family(), SelectiveEmittedRowFamily::Retained);
        assert_eq!(row.arm(), Some(source.arm()));
        assert_eq!(row.emitted_row(), step.emitted_row());
        let separator = if position == 0 { "    " } else { "  , " };
        let ports = row
            .ports()
            .iter()
            .map(lean_compact_port)
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            rendered,
            "{separator}{{ schemaVersion := {}, rows := {}, columns := {}, emittedRow := {}, runIndex := {}, family := .retained, arm := some {}, ports := [{}] }}",
            row.schema_version(),
            final_rows,
            final_columns,
            row.emitted_row(),
            row.run_index(),
            source.arm(),
            ports
        )
        .expect("render retained compact row");
    }
    writeln!(rendered, "  ]\n").expect("render retained row footer");
}

fn write_retained_source_rows(
    rendered: &mut String,
    definition: &str,
    projection: &SelectiveProjectedRowsAudit,
    indices: &[usize],
) {
    let source = projection
        .source_provenance()
        .expect("retained projection has complete source provenance");
    writeln!(rendered, "def {definition} : List RawSourceR1csRow :=\n  [").expect("render retained source-row header");
    for (position, &index) in indices.iter().enumerate() {
        let step = &source.retained_steps()[index];
        let separator = if position == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ row := {}, a := {}, b := {}, c := {} }}",
            step.source_row(),
            lean_source_linear_combination(step.a()),
            lean_source_linear_combination(step.b()),
            lean_source_linear_combination(step.c())
        )
        .expect("render retained source row");
    }
    writeln!(rendered, "  ]\n").expect("render retained source-row footer");
}

fn write_retained_projection(
    rendered: &mut String,
    name: &str,
    projection: &SelectiveProjectedRowsAudit,
    owners: &[RetainedRowOwner],
    final_rows: usize,
    final_columns: usize,
) {
    let source = projection
        .source_provenance()
        .expect("retained projection has complete source provenance");
    assert_eq!(projection.row_artifacts().len(), source.retained_steps().len());
    let partitions = retained_row_partitions(projection, owners);
    for (suffix, indices) in &partitions {
        write_retained_rows(
            rendered,
            &format!("{name}{suffix}Rows"),
            projection,
            indices,
            final_rows,
            final_columns,
        );
    }
    writeln!(
        rendered,
        "def {name}Rows : List RawCompactRow :=\n  {}\n",
        partitions
            .iter()
            .map(|(suffix, _)| format!("{name}{suffix}Rows"))
            .collect::<Vec<_>>()
            .join(" ++ ")
    )
    .expect("render retained row composition");
    for (suffix, indices) in &partitions {
        write_retained_source_rows(rendered, &format!("{name}{suffix}SourceRows"), projection, indices);
    }
    writeln!(
        rendered,
        "def {name}SourceRows : List RawSourceR1csRow :=\n  {}\n",
        partitions
            .iter()
            .map(|(suffix, _)| format!("{name}{suffix}SourceRows"))
            .collect::<Vec<_>>()
            .join(" ++ ")
    )
    .expect("render retained source-row composition");

    write_source_map(rendered, name, source.retained_slots(), source.linear_definitions());
    let semantic_indices = partitions
        .iter()
        .filter(|(suffix, _)| {
            matches!(
                *suffix,
                "ExpectedCarry" | "StatePins" | "Cursors" | "ReplayOutput" | "FinalReadiness"
            )
        })
        .flat_map(|(_, indices)| indices.iter().copied())
        .collect::<Vec<_>>();
    let (semantic_slots, semantic_definitions) = semantic_source_map(source, &semantic_indices);
    write_source_map(
        rendered,
        &format!("{name}Semantic"),
        &semantic_slots,
        &semantic_definitions,
    );

    writeln!(
        rendered,
        "def {name} : RawRetainedProjection :=\n  {{ arm := {}, finalRows := {}, finalColumns := {}, rows := {name}Rows, sourceRows := {name}SourceRows, slots := {name}Slots, definitions := {name}Definitions }}\n",
        source.arm(),
        final_rows,
        final_columns
    )
    .expect("render retained projection");
}

fn render_retained_projection_artifact(
    module_name: &str,
    arm_scope: &str,
    projection: &SelectiveProjectedRowsAudit,
    owners: &[RetainedRowOwner],
    final_rows: usize,
    final_columns: usize,
) -> (String, String) {
    let mut payload = String::new();
    write_retained_projection(
        &mut payload,
        "projection",
        projection,
        owners,
        final_rows,
        final_columns,
    );
    let artifact_sha256 = sha256_hex(payload.as_bytes());
    let rendered = format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteSchema\n\n\
         /-! GENERATED FILE. DO NOT EDIT. Exact compact Rust retained-row\n\
         projection for one production claim-replay base arm. -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.{module_name}\n\n\
         abbrev RawCompactRow := Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire.RawCompactRow\n\
         abbrev RawSourceR1csRow := Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Wire.RawSourceR1csRow\n\
         abbrev RawSourceSlot := Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Wire.RawSourceSlot\n\
         abbrev RawSourceDefinition := Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Wire.RawSourceDefinition\n\
         abbrev RawRetainedProjection := Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Wire.RawRetainedProjection\n\n\
         def artifactSha256 : String := \"{artifact_sha256}\"\n\
         def schemaVersion : Nat := 2\n\
         def profileId : String := \"{PROFILE_ID}\"\n\
         def baseArtifactSha256 : String := \"{BASE_ARTIFACT_SHA256}\"\n\
         def armScope : String := \"{arm_scope}\"\n\n\
         {payload}\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.{module_name}\n"
    );
    (rendered, artifact_sha256)
}

struct RenderedLinkedOverlayArtifacts {
    parent: String,
    full_retained: String,
    final_retained: String,
}

fn render_linked_overlay_artifacts(relation: &LinkedOverlayLowNormR1cs) -> RenderedLinkedOverlayArtifacts {
    let layout = relation.layout();
    let links = production_claim_active_coordinate_overlay_links();
    let seeded_placements = production_claim_active_coordinate_overlay_seeded_placements()
        .expect("audit exact active overlay seeded placements");
    let decoders = active_replay_decoders();
    let (external_decoders, final_replay_tail_definitions) = active_replay_external_decoders(relation);
    let full_retained = production_claim_replay_base_retained_row_projection(NebulaFPrimeClaimReplayArmKind::Full)
        .expect("project exact retained full-arm rows");
    let final_retained = production_claim_replay_base_retained_row_projection(NebulaFPrimeClaimReplayArmKind::Final)
        .expect("project exact retained final-arm rows");
    assert_retained_projection_matches_final_rows(relation, &full_retained);
    assert_retained_projection_matches_final_rows(relation, &final_retained);
    let full_retained_owners = retained_row_owners(
        NebulaFPrimeClaimReplaySynthesis::production_base_full(0)
            .expect("build production full claim-replay base for retained ownership"),
    );
    let final_retained_owners = retained_row_owners(NebulaFPrimeClaimReplaySynthesis::production_base_final());
    let (full_retained_rendered, full_retained_sha256) = render_retained_projection_artifact(
        "FPrimeFullHistoryStreamingClaimReplayLinkedOverlayFullRetained",
        "full",
        &full_retained,
        &full_retained_owners,
        relation.structure().n,
        relation.structure().m,
    );
    let (final_retained_rendered, final_retained_sha256) = render_retained_projection_artifact(
        "FPrimeFullHistoryStreamingClaimReplayLinkedOverlayFinalRetained",
        "final",
        &final_retained,
        &final_retained_owners,
        relation.structure().n,
        relation.structure().m,
    );
    let full_template = &decoders[0].repeated_templates()[0];
    let final_template = &decoders[1].repeated_templates()[0];
    let full_semantic = production_claim_replay_base_semantic_row_projection(NebulaFPrimeClaimReplayArmKind::Full)
        .expect("project exact full-arm replay input schedule");
    let final_semantic = production_claim_replay_base_semantic_row_projection(NebulaFPrimeClaimReplayArmKind::Final)
        .expect("project exact final-arm replay input schedule");
    let full_replay_final_row_start = assert_active_replay_input_schedule(&full_semantic, full_template.instances()[0]);
    let final_replay_final_row_start =
        assert_active_replay_input_schedule(&final_semantic, final_template.instances()[0]);
    let absolute_link_rows = (0..ACTIVE_CHUNKS)
        .map(|kind| {
            layout
                .field_link_rows_for_kind(kind)
                .expect("absolute active-kind link rows")
        })
        .collect::<Vec<_>>();
    let absolute_pin_rows = (0..ACTIVE_CHUNKS)
        .map(|kind| {
            layout
                .base_field_pin_rows_for_kind(kind)
                .expect("absolute active-kind base-field pin rows")
        })
        .collect::<Vec<_>>();
    let link_field_counts = links
        .iter()
        .map(|contract| contract.fields.len())
        .collect::<Vec<_>>();
    let base_pin_fields = links
        .iter()
        .flat_map(|contract| contract.base_pins.iter().map(|pin| pin.phase_field))
        .collect::<Vec<_>>();
    let base_pin_values = links
        .iter()
        .flat_map(|contract| {
            contract
                .base_pins
                .iter()
                .map(|pin| pin.value.as_canonical_u64() as usize)
        })
        .collect::<Vec<_>>();

    let mut payload = String::new();
    writeln!(
        payload,
        "def layout : RawLayout :=\n  {{ publicColumns := {}, basePrivateColumns := {}, overlayPrivateColumns := {}, ringPaddingColumns := {},\n    baseRows := {}, overlayRows := {}, baseKindEqualityRows := {}, overlayActivationRows := {},\n    fieldLinkRows := {}, baseFieldPinRows := {}, ringPaddingRows := {},\n    baseSelectorColumns := {}, overlaySelectorColumns := {},\n    basePhaseKinds := {}, overlayBaseKinds := {},\n    absoluteLinkRows := [{}], linkFieldCounts := {},\n    absolutePinRows := [{}], basePinFields := {}, basePinValues := {} }}",
        lean_range(layout.public_columns()),
        lean_range(layout.base_private_columns()),
        lean_range(layout.overlay_private_columns()),
        lean_range(layout.ring_padding_columns()),
        lean_range(layout.base_rows()),
        lean_range(layout.overlay_rows()),
        lean_range(layout.base_kind_equality_rows()),
        lean_range(layout.overlay_activation_rows()),
        lean_range(layout.field_link_rows()),
        lean_range(layout.base_field_pin_rows()),
        lean_range(layout.ring_padding_rows()),
        lean_nat_list(layout.base_selector_columns()),
        lean_nat_list(layout.overlay_selector_columns()),
        lean_nat_list(layout.base_phase_kinds()),
        lean_nat_list(layout.overlay_base_kinds()),
        absolute_link_rows
            .into_iter()
            .map(lean_range)
            .collect::<Vec<_>>()
            .join(", "),
        lean_nat_list(&link_field_counts),
        absolute_pin_rows
            .into_iter()
            .map(lean_range)
            .collect::<Vec<_>>()
            .join(", "),
        lean_nat_list(&base_pin_fields),
        lean_nat_list(&base_pin_values),
    )
    .expect("render linked-overlay layout");
    let first_pin_rows = layout
        .base_field_pin_rows_for_kind(0)
        .expect("first active-kind base-field pin rows");
    let initial_runtime_pins = &links[0].base_pins[1..];
    assert_eq!(initial_runtime_pins.len(), 8);
    assert!(initial_runtime_pins.iter().all(|pin| pin.value == F::ZERO));
    writeln!(
        payload,
        "\ndef initialRuntimePinRows : Range := {}\n\
         def initialRuntimePinFields : List Nat := {}\n\
         def initialRuntimePinValues : List Nat := {}\n\
         def programCursorPinField : Nat := {}\n\
         def programCursorPinValueStart : Nat := {}\n\
         def programCursorPinCount : Nat := {}",
        lean_range(first_pin_rows.start + 1..first_pin_rows.end),
        lean_nat_list(
            &initial_runtime_pins
                .iter()
                .map(|pin| pin.phase_field)
                .collect::<Vec<_>>()
        ),
        lean_nat_list(
            &initial_runtime_pins
                .iter()
                .map(|pin| pin.value.as_canonical_u64() as usize)
                .collect::<Vec<_>>()
        ),
        links[0].base_pins[0].phase_field,
        links[0].base_pins[0].value.as_canonical_u64(),
        links.len(),
    )
    .expect("render compact base-field pin leaves");
    writeln!(
        payload,
        "\ndef activeOverlaySeededPlacements : List RawOverlaySeededPlacement :=\n  ["
    )
    .expect("render seeded placement header");
    for (index, placement) in seeded_placements.iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        let runs = placement
            .word_start_runs()
            .iter()
            .map(|run| {
                format!(
                    "{{ sourceStart := {}, finalStart := {}, count := {}, sourceStride := {}, finalStride := {} }}",
                    run.source_start(),
                    run.final_start(),
                    run.count(),
                    run.source_stride(),
                    run.final_stride(),
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            payload,
            "{separator}{{ arm := {}, selectorColumn := {}, sourceRowStart := {}, finalRowStart := {}, wordStartRuns := [{}], wordCount := {}, wordWidth := {}, kappa := {}, messageColumns := {} }}",
            placement.arm(),
            placement.selector_column(),
            placement.source_row_start(),
            placement.final_row_start(),
            runs,
            placement.word_count(),
            placement.word_width(),
            placement.kappa(),
            placement.message_columns(),
        )
        .expect("render seeded placement");
    }
    writeln!(payload, "  ]").expect("render seeded placement footer");
    write_runs(
        &mut payload,
        "activeReplayRelativeRuns",
        "RawDecoderRun",
        full_template.relative_runs(),
    );
    for (name, decoder) in [
        ("fullReplayInitialCapacityRuns", &external_decoders[0]),
        ("fullReplayFreshRateRuns", &external_decoders[1]),
        ("finalReplayInitialCapacityRuns", &external_decoders[2]),
        ("finalReplayFreshRateRuns", &external_decoders[3]),
        ("finalReplayTailRuns", &external_decoders[4]),
    ] {
        write_runs(&mut payload, name, "RawDecoderRun", decoder.runs());
    }
    writeln!(
        payload,
        "def finalReplayTailDefinitions : List RawSourceDefinition :=\n  [{}]",
        final_replay_tail_definitions
            .iter()
            .map(lean_linear_definition)
            .collect::<Vec<_>>()
            .join(", ")
    )
    .expect("render exact final replay-tail definitions");
    writeln!(
        payload,
        "def fullActiveReplayInstances : RawDecoderTemplateInstances :=\n  {}\n\ndef finalActiveReplayInstances : RawDecoderTemplateInstances :=\n  {}",
        lean_decoder_instances(full_template.instances()[0]),
        lean_decoder_instances(final_template.instances()[0]),
    )
    .expect("render active replay decoder instances");
    writeln!(
        payload,
        "def replayInitialCapacitySourceStart : Nat := {REPLAY_INITIAL_CAPACITY_SOURCE_START}\n\
         def replayFreshRateSourceStart : Nat := {REPLAY_FRESH_RATE_SOURCE_START}\n\
         def replayRateWidth : Nat := {REPLAY_RATE_WIDTH}\n\
         def replayChainedCapacityOffset : Nat := {REPLAY_CHAINED_CAPACITY_OFFSET}\n\
         def fullReplayFinalRowStart : Nat := {full_replay_final_row_start}\n\
         def finalReplayFinalRowStart : Nat := {final_replay_final_row_start}\n\
         def replayFinalRowStride : Nat := 86"
    )
    .expect("render active replay input schedule");
    writeln!(
        payload,
        "def fullRetainedArtifactSha256 : String := \"{full_retained_sha256}\"\n\
         def finalRetainedArtifactSha256 : String := \"{final_retained_sha256}\"\n\n\
         def poseidonLeafOwner : String := \"Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeaf\"\n\
         def poseidonRowsPerCall : Nat := 86\n\
         def firstCallExternalPermutation : List Nat := [4, 5, 6, 7, 0, 1, 2, 3]\n\n\
         abbrev fullRetained := Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayLinkedOverlayFullRetained.projection\n\
         abbrev finalRetained := Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayLinkedOverlayFinalRetained.projection"
    )
    .expect("render retained leaf identities");
    let artifact_sha256 = sha256_hex(payload.as_bytes());

    let parent = format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimCoordinateOverlay\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayBase\n\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayLinkedOverlayFinalRetained\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayLinkedOverlayFullRetained\n\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeaf\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteSchema\n\n\
         /-! GENERATED FILE. DO NOT EDIT. Exact compact Rust receipt for the\n\
         selected production claim-replay base plus active coordinate overlay. -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayLinkedOverlay\n\n\
         abbrev RawDecoderRun := Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema.RawRun\n\
         abbrev RawDecoderTemplateInstances := Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema.RawTemplateInstances\n\
         abbrev RawSourceDefinition := Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Wire.RawSourceDefinition\n\n\
         structure RawOverlayWordStartRun where\n  sourceStart : Nat\n  finalStart : Nat\n  count : Nat\n  sourceStride : Nat\n  finalStride : Nat\n\
         deriving DecidableEq, Repr\n\n\
         structure RawOverlaySeededPlacement where\n  arm : Nat\n  selectorColumn : Nat\n  sourceRowStart : Nat\n  finalRowStart : Nat\n  wordStartRuns : List RawOverlayWordStartRun\n  wordCount : Nat\n  wordWidth : Nat\n  kappa : Nat\n  messageColumns : Nat\n\
         deriving DecidableEq, Repr\n\n\
         structure Range where\n  start : Nat\n  stop : Nat\n\
         deriving DecidableEq, Repr\n\n\
         structure RawLayout where\n  publicColumns : Range\n  basePrivateColumns : Range\n  overlayPrivateColumns : Range\n  ringPaddingColumns : Range\n  baseRows : Range\n  overlayRows : Range\n  baseKindEqualityRows : Range\n  overlayActivationRows : Range\n  fieldLinkRows : Range\n  baseFieldPinRows : Range\n  ringPaddingRows : Range\n  baseSelectorColumns : List Nat\n  overlaySelectorColumns : List Nat\n  basePhaseKinds : List Nat\n  overlayBaseKinds : List Nat\n  absoluteLinkRows : List Range\n  linkFieldCounts : List Nat\n  absolutePinRows : List Range\n  basePinFields : List Nat\n  basePinValues : List Nat\n\
         deriving DecidableEq, Repr\n\n\
         def artifactSha256 : String := \"{artifact_sha256}\"\n\
         def schemaVersion : Nat := 11\n\
         def profileId : String := \"{PROFILE_ID}\"\n\
         def artifactIdentity : String := \"{ARTIFACT_ID}\"\n\
         def baseArtifactSha256 : String := \"{BASE_ARTIFACT_SHA256}\"\n\
         def overlayArtifactSha256 : String := \"{OVERLAY_ARTIFACT_SHA256}\"\n\
         def activeOverlaySourceKinds : Range := {{ start := 1, stop := 99 }}\n\n\
         {payload}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayLinkedOverlay\n"
    );
    RenderedLinkedOverlayArtifacts {
        parent,
        full_retained: full_retained_rendered,
        final_retained: final_retained_rendered,
    }
}

fn generated_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingClaimReplayLinkedOverlay.lean",
    )
}

fn generated_full_retained_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingClaimReplayLinkedOverlayFullRetained.lean",
    )
}

fn generated_final_retained_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingClaimReplayLinkedOverlayFinalRetained.lean",
    )
}

#[test]
fn production_claim_replay_base_compiler_disposition_census_is_exact() {
    let relation =
        build_production_claim_replay_base_low_norm_r1cs().expect("build exact production claim-replay base");
    let audit = relation
        .selective_compiler_audit()
        .expect("production base retains its exact compiler audit");

    for (arm_index, arm) in audit.rows().arms().iter().enumerate() {
        let stages = &audit.source_arm_physical_stages()[arm_index];
        let mut source_cursor = 0;
        let mut census = vec![[0usize; 7]; stages.len()];
        for run in arm.source_runs() {
            let source_rows = run.source_rows();
            assert_eq!(source_rows.start, source_cursor);
            source_cursor = source_rows.end;
            let stage = run
                .stage_occurrence()
                .expect("every production-base source run has one physical stage");
            assert!(stage < stages.len());
            assert!(stages[stage].rows().start <= source_rows.start);
            assert!(source_rows.end <= stages[stage].rows().end);
            let disposition = match run.disposition() {
                SelectiveSourceRowDisposition::Retained => 0,
                SelectiveSourceRowDisposition::Poseidon2(_) => 1,
                SelectiveSourceRowDisposition::CenteredUnit(_) => 2,
                SelectiveSourceRowDisposition::ShiftedTernaryCanonical(_) => 3,
                SelectiveSourceRowDisposition::PolynomialEvaluation(_) => 4,
                SelectiveSourceRowDisposition::ProductSum(_) => 5,
                SelectiveSourceRowDisposition::LinearDefinition(_) => 6,
            };
            census[stage][disposition] += source_rows.len();
        }
        assert_eq!(
            source_cursor,
            stages
                .last()
                .expect("production base has stages")
                .rows()
                .end
        );
        eprintln!("claim-replay base arm {arm_index} source-row disposition census: {census:?}");
    }

    let requests = [(0, 2357..155957), (1, 2357..88157)];
    let (compact, decoders) = production_claim_replay_base_compact_layout_and_decoder_runs_for_ranges(&requests)
        .expect("audit exact active replay-call decoder ranges");
    assert_eq!(compact.rows(), audit.rows());
    assert_eq!(compact.selector_columns(), relation.selector_cols());
    assert_eq!(compact.final_columns(), relation.structure().m);
    assert_eq!(decoders.len(), requests.len());
    for ((arm, source_range), decoder) in requests.iter().zip(&decoders) {
        assert_eq!(decoder.arm(), *arm);
        assert_eq!(decoder.source_range(), source_range.clone());
        assert_eq!(decoder.final_columns(), relation.structure().m);
        eprintln!(
            "claim-replay base arm {arm} active decoder: contiguous_runs={}, strided_runs={}, templates={}, residual_runs={}, families={}",
            decoder.runs().len(),
            decoder.strided_runs().len(),
            decoder.repeated_templates().len(),
            decoder.residual_strided_runs().len(),
            decoder.source_families().len(),
        );
    }
    let full_template = &decoders[0].repeated_templates()[0];
    let final_template = &decoders[1].repeated_templates()[0];
    assert_eq!(full_template.source_width(), 600);
    assert_eq!(final_template.source_width(), 600);
    assert_eq!(full_template.relative_runs(), final_template.relative_runs());
    assert_eq!(full_template.instances().len(), 1);
    assert_eq!(final_template.instances().len(), 1);
    assert_eq!(full_template.instances()[0].count(), 256);
    assert_eq!(final_template.instances()[0].count(), 143);
    eprintln!(
        "claim-replay shared decoder template: relative_runs={}, full_instances={:?}, final_instances={:?}",
        full_template.relative_runs().len(),
        full_template.instances(),
        final_template.instances(),
    );
}

#[test]
fn production_claim_replay_base_semantic_projection_is_exact() {
    let relation = build_production_claim_replay_linked_overlay_low_norm_r1cs()
        .expect("build exact production claim-replay linked overlay");
    let decoders = active_replay_decoders();
    let reference_selected_rows = (74_375..74_547).collect::<Vec<_>>();
    let reference =
        production_pi_rlc_family_body_projected_rows_with_source_provenance(&reference_selected_rows, 0, &[], &[])
            .expect("project the proved direct and chained PiRLC Poseidon2 leaves");
    let reference_source = reference
        .source_provenance()
        .expect("PiRLC leaf reference has complete source provenance");
    let reference_rows = reference.row_artifacts().iter().collect::<Vec<_>>();
    let reference_direct = canonical_poseidon_call(
        &reference_source.poseidon2_sbox_steps()[..86],
        &reference_rows[..86],
        166_320,
        2_218_425,
        648,
        false,
    );
    let reference_chained = canonical_poseidon_call(
        &reference_source.poseidon2_sbox_steps()[86..172],
        &reference_rows[86..172],
        166_920,
        2_221_951,
        648,
        false,
    );
    for (arm, kind) in [
        NebulaFPrimeClaimReplayArmKind::Full,
        NebulaFPrimeClaimReplayArmKind::Final,
    ]
    .into_iter()
    .enumerate()
    {
        let projected = production_claim_replay_base_semantic_row_projection(kind)
            .expect("project exact production-base semantic rows");
        let source = projected
            .source_provenance()
            .expect("semantic row projection has complete source provenance");
        let retained = production_claim_replay_base_retained_row_projection(kind)
            .expect("project exact production-base retained rows");
        let retained_source = retained
            .source_provenance()
            .expect("retained row projection has complete source provenance");
        assert_retained_projection_matches_final_rows(&relation, &retained);
        assert_eq!(source.arm(), arm);
        assert_eq!(retained_source.arm(), arm);
        assert!(!projected.row_artifacts().is_empty());
        assert!(!source.retained_steps().is_empty());
        assert!(!source.poseidon2_sbox_steps().is_empty());
        assert!(
            source.poseidon2_output_steps().is_empty(),
            "production Poseidon2 outputs are exact compiler linear definitions"
        );
        assert_eq!(retained.row_artifacts().len(), source.retained_steps().len());
        assert_eq!(retained_source.retained_steps().len(), source.retained_steps().len());
        assert!(retained_source.poseidon2_sbox_steps().is_empty());
        assert!(retained_source.poseidon2_output_steps().is_empty());
        let expected = if arm == 0 {
            (22_178, 24_236, 23_208, 1_028, 22_016, 162, 180, 172, 8)
        } else {
            (12_469, 13_611, 13_038, 573, 12_298, 171, 177, 172, 5)
        };
        assert_eq!(
            (
                projected.row_artifacts().len(),
                source.source_columns().len(),
                source.retained_slots().len(),
                source.linear_definitions().len(),
                source.poseidon2_sbox_steps().len(),
                source.retained_steps().len(),
                retained_source.source_columns().len(),
                retained_source.retained_slots().len(),
                retained_source.linear_definitions().len(),
            ),
            expected
        );
        let poseidon_rows = projected
            .row_artifacts()
            .iter()
            .filter(|row| row.family() == SelectiveEmittedRowFamily::Poseidon2)
            .collect::<Vec<_>>();
        assert_eq!(poseidon_rows.len(), source.poseidon2_sbox_steps().len());
        let instances = decoders[arm].repeated_templates()[0].instances()[0];
        assert_eq!(source.poseidon2_sbox_steps().len(), instances.count() * 86);
        for call_index in 0..instances.count() {
            let source_start = instances.source_start() + call_index * instances.source_stride();
            let final_start = instances.final_start() + call_index * instances.final_stride();
            let steps = &source.poseidon2_sbox_steps()[call_index * 86..(call_index + 1) * 86];
            let rows = &poseidon_rows[call_index * 86..(call_index + 1) * 86];
            let actual = canonical_poseidon_call(steps, rows, source_start, final_start, 648 + arm, call_index == 0);
            let expected = if call_index == 0 {
                &reference_direct
            } else {
                &reference_chained
            };
            if &actual != expected {
                let step = actual
                    .0
                    .iter()
                    .zip(&expected.0)
                    .position(|(left, right)| left != right);
                let row = actual
                    .1
                    .iter()
                    .zip(&expected.1)
                    .position(|(left, right)| left != right);
                panic!(
                    "claim-replay arm {arm} Poseidon2 call {call_index} differs from the proved leaf: first_step={step:?} actual_step={:?} expected_step={:?}, first_row={row:?} actual_row={:?} expected_row={:?}",
                    step.map(|index| &actual.0[index]),
                    step.map(|index| &expected.0[index]),
                    row.map(|index| &actual.1[index]),
                    row.map(|index| &expected.1[index]),
                );
            }
        }
        eprintln!(
            "claim-replay base arm {arm} semantic projection: final_rows={}, source_columns={}, retained_slots={}, definitions={}, trace_eliminated={}, sboxes={}, outputs={}, rewrites={}, retained={}; retained-only: source_columns={}, retained_slots={}, definitions={}",
            projected.row_artifacts().len(),
            source.source_columns().len(),
            source.retained_slots().len(),
            source.linear_definitions().len(),
            source.trace_eliminated_columns().len(),
            source.poseidon2_sbox_steps().len(),
            source.poseidon2_output_steps().len(),
            source.rewrite_steps().len(),
            source.retained_steps().len(),
            retained_source.source_columns().len(),
            retained_source.retained_slots().len(),
            retained_source.linear_definitions().len(),
        );
    }
}

#[test]
fn production_claim_replay_active_overlay_compiler_disposition_census_is_exact() {
    let relation = build_production_claim_replay_linked_overlay_low_norm_r1cs()
        .expect("build exact production claim-replay linked overlay");
    let audit = relation
        .overlay_relation()
        .selective_compiler_audit()
        .expect("active overlay retains its exact compiler audit");
    for (arm_index, arm) in audit.rows().arms().iter().enumerate() {
        let mut source = [0usize; 7];
        for run in arm.source_runs() {
            let disposition = match run.disposition() {
                SelectiveSourceRowDisposition::Retained => 0,
                SelectiveSourceRowDisposition::Poseidon2(_) => 1,
                SelectiveSourceRowDisposition::CenteredUnit(_) => 2,
                SelectiveSourceRowDisposition::ShiftedTernaryCanonical(_) => 3,
                SelectiveSourceRowDisposition::PolynomialEvaluation(_) => 4,
                SelectiveSourceRowDisposition::ProductSum(_) => 5,
                SelectiveSourceRowDisposition::LinearDefinition(_) => 6,
            };
            source[disposition] += run.source_rows().len();
        }
        assert_eq!(
            source,
            match arm_index {
                0 => [298, 0, 0, 79_484, 0, 0, 652],
                61 | 69 => [298, 0, 0, 126_976, 0, 0, 328],
                97 => [149, 0, 0, 71_300, 0, 0, 326],
                _ => [149, 0, 0, 126_976, 0, 0, 326],
            },
            "active overlay source census for arm {arm_index}"
        );

        let rewrites = audit
            .rows()
            .rewrites()
            .iter()
            .filter(|rewrite| rewrite.arm() == arm_index)
            .collect::<Vec<_>>();
        assert!(rewrites.iter().all(|rewrite| {
            matches!(
                rewrite.kind(),
                neo_fold_clean::frontends::r1cs_f_prime::SelectiveRewriteKind::ShiftedTernaryCanonical
                    | neo_fold_clean::frontends::r1cs_f_prime::SelectiveRewriteKind::LinearDefinition
            )
        }));
        let shifted_rows = rewrites
            .iter()
            .filter(|rewrite| {
                rewrite.kind() == neo_fold_clean::frontends::r1cs_f_prime::SelectiveRewriteKind::ShiftedTernaryCanonical
            })
            .map(|rewrite| rewrite.emitted_rows().len())
            .sum::<usize>();
        assert_eq!(
            shifted_rows,
            match arm_index {
                0 => 13_461,
                97 => 12_075,
                _ => 21_504,
            },
            "active overlay shifted-ternary rows for arm {arm_index}"
        );
    }

    let requests = [0usize, 1, 61, 69, 97].map(|arm| {
        let source = NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(arm + 1)
            .expect("representative active coordinate-overlay source arm");
        (arm, 1..source.columns())
    });
    let (compact, decoders) =
        production_claim_active_coordinate_overlay_compact_layout_and_decoder_runs_for_ranges(&requests)
            .expect("decode exact representative active overlay source arms");
    assert_eq!(compact.rows(), audit.rows());
    assert_eq!(compact.selector_columns(), relation.overlay_relation().selector_cols());
    assert_eq!(compact.final_columns(), relation.overlay_relation().structure().m);
    for decoder in decoders {
        eprintln!(
            "active overlay arm {} complete decoder {}..{}: runs={}, strided={}, templates={}, residual={}, families={}",
            decoder.arm(),
            decoder.source_range().start,
            decoder.source_range().end,
            decoder.runs().len(),
            decoder.strided_runs().len(),
            decoder.repeated_templates().len(),
            decoder.residual_strided_runs().len(),
            decoder.source_families().len(),
        );
    }
}

#[test]
fn production_claim_replay_first_active_overlay_nonseeded_row_projection_is_exact() {
    let projection = production_claim_active_coordinate_overlay_nonseeded_row_projection(0)
        .expect("project the exact non-seeded rows of the first active coordinate-overlay arm");
    let relation = build_production_claim_replay_linked_overlay_low_norm_r1cs()
        .expect("build exact production claim-replay linked overlay");
    assert_eq!(projection.rows(), relation.overlay_relation().structure().n);
    assert_eq!(projection.columns(), relation.overlay_relation().structure().m);
    assert_eq!(
        projection.selector_columns(),
        relation.overlay_relation().selector_cols()
    );
    assert!(projection.source_provenance().is_some());
    assert!(!projection.row_artifacts().is_empty());
    assert!(projection
        .row_artifacts()
        .iter()
        .flat_map(|row| row.ports())
        .all(|port| port.seeded_blocks().is_empty()));
}

#[test]
fn production_claim_replay_active_overlay_seeded_placements_are_exact() {
    let placements = production_claim_active_coordinate_overlay_seeded_placements()
        .expect("audit exact compact coordinate-overlay block placements");
    assert_eq!(placements.len(), 101);
    let mut counts = [0usize; ACTIVE_CHUNKS];
    let mut profile_counts = [0usize; 3];
    for placement in &placements {
        counts[placement.arm()] += 1;
        assert_eq!(placement.word_width(), 41);
        assert_eq!(placement.kappa(), 2);
        match (placement.word_count(), placement.message_columns()) {
            (28_672, 21_770) => profile_counts[0] += 1,
            (62_208, 47_232) => profile_counts[1] += 1,
            (8_640, 6_560) => profile_counts[2] += 1,
            profile => panic!("unexpected coordinate-overlay seeded profile {profile:?}"),
        }
        assert_eq!(
            placement
                .word_start_runs()
                .iter()
                .map(|run| run.count())
                .sum::<usize>(),
            placement.word_count()
        );
    }
    assert_eq!(profile_counts, [30, 62, 9]);
    assert_eq!(counts.iter().filter(|&&count| count == 2).count(), 3);
    assert_eq!(counts.iter().filter(|&&count| count == 1).count(), 95);
    eprintln!(
        "claim-replay active overlay compact seeded placements: blocks={}, runs={}, double_call_arms={:?}, first_selector={}, first_rows={}->{}, terminal_selector={}, terminal_rows={}->{}",
        placements.len(),
        placements
            .iter()
            .map(|placement| placement.word_start_runs().len())
            .sum::<usize>(),
        counts
            .iter()
            .enumerate()
            .filter_map(|(arm, &count)| (count == 2).then_some(arm))
            .collect::<Vec<_>>(),
        placements[0].selector_column(),
        placements[0].source_row_start(),
        placements[0].final_row_start(),
        placements[100].selector_column(),
        placements[100].source_row_start(),
        placements[100].final_row_start(),
    );
}

#[test]
fn production_claim_replay_linked_overlay_has_exact_assignments_and_links() {
    let relation = build_production_claim_replay_linked_overlay_low_norm_r1cs()
        .expect("build exact production claim-replay linked overlay");
    let base_phase_kinds = production_claim_replay_base_phase_kinds();
    let base_kind_map = production_claim_active_coordinate_overlay_base_kind_map();
    let links = production_claim_active_coordinate_overlay_links();
    conformance::assert_exact_final_row_embedding(&relation, &links);
    let layout = relation.layout();
    let retained = [
        production_claim_replay_base_retained_row_projection(NebulaFPrimeClaimReplayArmKind::Full)
            .expect("project exact retained full-arm rows"),
        production_claim_replay_base_retained_row_projection(NebulaFPrimeClaimReplayArmKind::Final)
            .expect("project exact retained final-arm rows"),
    ];
    let retained_decoders = retained
        .iter()
        .enumerate()
        .map(|(base_kind, projection)| {
            projection
                .source_provenance()
                .expect("retained projection has complete source provenance")
                .source_columns()
                .iter()
                .map(|&source_column| {
                    let terms = relation
                        .base_field_decoding_terms(base_kind, source_column)
                        .expect("retained base source field has an exact final decoder");
                    (source_column, terms)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert_eq!(base_phase_kinds, vec![3, 4]);
    assert_eq!(base_kind_map.len(), ACTIVE_CHUNKS);
    assert!(base_kind_map[..FULL_CHUNKS].iter().all(|&kind| kind == 0));
    assert_eq!(base_kind_map[FULL_CHUNKS], 1);
    assert_eq!(links.len(), ACTIVE_CHUNKS);
    assert_eq!(layout.base_phase_kinds(), base_phase_kinds);
    assert_eq!(layout.overlay_base_kinds(), base_kind_map);
    assert_eq!(layout.base_selector_columns().len(), 2);
    assert_eq!(layout.overlay_selector_columns().len(), ACTIVE_CHUNKS);
    assert_eq!(layout.base_rows(), 0..relation.base_relation().structure().n);
    assert_eq!(
        layout.overlay_rows(),
        layout.base_rows().end..layout.base_rows().end + relation.overlay_relation().structure().n
    );
    assert_eq!(layout.base_kind_equality_rows().len(), 2);
    assert_eq!(layout.overlay_activation_rows().len(), ACTIVE_CHUNKS);
    let base_pin_count = links
        .iter()
        .map(|contract| contract.base_pins.len())
        .sum::<usize>();
    assert_eq!(base_pin_count, 106);
    assert_eq!(layout.base_field_pin_rows().len(), base_pin_count);
    assert_eq!(
        layout.field_link_rows().len(),
        links
            .iter()
            .map(|contract| contract.fields.len())
            .sum::<usize>()
    );
    assert_eq!(layout.rows(), relation.structure().n);
    assert_eq!(layout.columns(), relation.structure().m);

    let mut final_assignment = None;
    let mut previous_link_end = layout.field_link_rows().start;
    let mut previous_pin_end = layout.base_field_pin_rows().start;
    let mut saw_affine_defined_link = false;
    let mut saw_zero_affine_link = false;
    for overlay_kind in 0..ACTIVE_CHUNKS {
        let base_kind = base_kind_map[overlay_kind];
        let base = if base_kind == 0 {
            NebulaFPrimeClaimReplaySynthesis::production_base_full(overlay_kind)
                .expect("active full production-base source")
        } else {
            NebulaFPrimeClaimReplaySynthesis::production_base_final()
        };
        let overlay = NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(overlay_kind + 1)
            .expect("active coordinate-overlay source");
        assert!(base.is_satisfied(), "base source arm {overlay_kind} must accept");
        assert!(overlay.is_satisfied(), "overlay source arm {overlay_kind} must accept");

        let base_source = base
            .normalized_field_assignment_for_artifact()
            .expect("normalize production-base assignment");
        let overlay_source = overlay.normalized_field_assignment_for_artifact();
        let assignment = relation
            .encode(overlay_kind, &base_source, &overlay_source)
            .expect("encode one selected final assignment");
        assert_eq!(assignment.len(), relation.structure().m);
        let base_assignment = relation
            .base_relation()
            .encode(base_kind, &base_source)
            .expect("encode the selected production-base assignment");
        assert_eq!(
            &assignment[..base_assignment.len()],
            base_assignment.as_slice(),
            "the final assignment must retain the exact selected base assignment prefix"
        );
        for (source_column, terms) in &retained_decoders[base_kind] {
            assert_eq!(
                evaluate_decoder(terms, &assignment),
                base_source[*source_column],
                "retained base source column {source_column} must decode from the same final assignment"
            );
        }

        for (kind, &selector) in layout.base_selector_columns().iter().enumerate() {
            assert_eq!(assignment[selector], if kind == base_kind { F::ONE } else { F::ZERO });
        }
        for (kind, &selector) in layout.overlay_selector_columns().iter().enumerate() {
            assert_eq!(
                assignment[selector],
                if kind == overlay_kind { F::ONE } else { F::ZERO }
            );
        }

        let contract = &links[overlay_kind];
        assert_eq!(contract.overlay_kind, overlay_kind);
        assert_eq!(contract.phase_kind, base_phase_kinds[base_kind]);
        let link_rows = layout
            .field_link_rows_for_kind(overlay_kind)
            .expect("absolute final link rows for active kind");
        assert_eq!(link_rows.start, previous_link_end);
        assert_eq!(link_rows.len(), contract.fields.len());
        previous_link_end = link_rows.end;
        let pin_rows = layout
            .base_field_pin_rows_for_kind(overlay_kind)
            .expect("absolute final base-field pin rows for active kind");
        assert_eq!(pin_rows.start, previous_pin_end);
        assert_eq!(pin_rows.len(), contract.base_pins.len());
        previous_pin_end = pin_rows.end;
        let (pin, runtime_pins) = contract
            .base_pins
            .split_first()
            .expect("each active claim kind owns one program-cursor pin");
        assert_eq!(runtime_pins.len(), if overlay_kind == 0 { 8 } else { 0 });
        assert_eq!(
            pin.phase_field,
            base.normalized_before_program_cursor_column()
                .expect("normalized before-program-cursor field")
        );
        assert_eq!(
            pin.value,
            F::from_usize(NebulaFPrimeStreamingProgramAudit::production().first_claim_program_cursor() + overlay_kind)
        );
        let pin_terms = relation
            .base_field_decoding_terms(base_kind, pin.phase_field)
            .expect("base program-cursor decoder in final columns");
        assert_eq!(evaluate_decoder(&pin_terms, &assignment), pin.value);
        for (lane, runtime_pin) in runtime_pins.iter().enumerate() {
            assert_eq!(
                runtime_pin.phase_field,
                base.normalized_before_runtime_column(lane)
                    .expect("normalized before-runtime field")
            );
            assert_eq!(runtime_pin.value, F::ZERO);
            let runtime_terms = relation
                .base_field_decoding_terms(base_kind, runtime_pin.phase_field)
                .expect("base initial-runtime decoder in final columns");
            assert_eq!(evaluate_decoder(&runtime_terms, &assignment), F::ZERO);
        }
        if overlay_kind == 0 {
            let mut tampered = assignment.clone();
            let &(column, _) = pin_terms
                .first()
                .expect("program cursor has one direct decoder");
            tampered[column] += F::ONE;
            assert_ne!(
                evaluate_relation_row(&relation, pin_rows.start, &tampered),
                F::ZERO,
                "the exact first program-cursor pin row must reject a changed decoded value"
            );
        }

        for link in &contract.fields {
            let base_terms = relation
                .base_field_decoding_terms(base_kind, link.phase_field)
                .expect("base source field decoder in final columns");
            let overlay_terms = relation
                .overlay_field_decoding_terms(overlay_kind, link.overlay_field)
                .expect("overlay source field decoder in final columns");
            if let Some((base_start, base_width)) = relation
                .base_relation()
                .field_slot(base_kind, link.phase_field)
            {
                assert_eq!(base_terms, direct_slot_terms(base_start, base_width));
            } else {
                saw_affine_defined_link = true;
                saw_zero_affine_link |= base_terms.is_empty();
            }
            if let Some((overlay_start, overlay_width)) = relation
                .overlay_relation()
                .field_slot(overlay_kind, link.overlay_field)
            {
                let embedded_overlay_start = if overlay_start == 0 {
                    0
                } else {
                    layout.overlay_private_columns().start + overlay_start - 1
                };
                assert_eq!(overlay_terms, direct_slot_terms(embedded_overlay_start, overlay_width));
            } else {
                saw_affine_defined_link = true;
                saw_zero_affine_link |= overlay_terms.is_empty();
            }
            let decoded_base = evaluate_decoder(&base_terms, &assignment);
            let decoded_overlay = evaluate_decoder(&overlay_terms, &assignment);
            assert_eq!(decoded_base, base_source[link.phase_field]);
            assert_eq!(decoded_overlay, overlay_source[link.overlay_field]);
            assert_eq!(decoded_base, decoded_overlay);
        }

        if overlay_kind + 1 == ACTIVE_CHUNKS {
            final_assignment = Some(assignment);
        }
    }
    assert_eq!(previous_link_end, layout.field_link_rows().end);
    assert_eq!(previous_pin_end, layout.base_field_pin_rows().end);
    assert!(
        saw_affine_defined_link,
        "the exact link contract must retain affine-definition decoder provenance"
    );
    assert!(
        saw_zero_affine_link,
        "the exact link contract must retain zero affine-definition decoder provenance"
    );

    let final_assignment = final_assignment.expect("terminal claim assignment");
    assert_eq!(
        relation.first_unsatisfied_row(&final_assignment),
        None,
        "the exact selected terminal assignment must satisfy every final row"
    );
}

#[test]
fn production_claim_replay_linked_overlay_artifact_is_current() {
    let relation = build_production_claim_replay_linked_overlay_low_norm_r1cs()
        .expect("build exact production claim-replay linked overlay");
    let rendered = render_linked_overlay_artifacts(&relation);
    let mut stale = Vec::new();
    for (path, contents) in [
        (generated_artifact_path(), rendered.parent),
        (generated_full_retained_artifact_path(), rendered.full_retained),
        (generated_final_retained_artifact_path(), rendered.final_retained),
    ] {
        if std::fs::read_to_string(&path).ok().as_deref() != Some(contents.as_str()) {
            let expected = PathBuf::from(format!("{}.expected", path.display()));
            std::fs::write(&expected, contents).expect("write expected linked-overlay Lean receipt");
            stale.push(expected);
        }
    }
    assert!(
        stale.is_empty(),
        "linked-overlay Lean receipts are stale; inspect {}",
        stale
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

#[test]
#[ignore = "writes the exact generated Lean linked-overlay receipt"]
fn regenerate_production_claim_replay_linked_overlay_artifact() {
    let relation = build_production_claim_replay_linked_overlay_low_norm_r1cs()
        .expect("build exact production claim-replay linked overlay");
    let rendered = render_linked_overlay_artifacts(&relation);
    for (path, contents) in [
        (generated_artifact_path(), rendered.parent),
        (generated_full_retained_artifact_path(), rendered.full_retained),
        (generated_final_retained_artifact_path(), rendered.final_retained),
    ] {
        std::fs::write(path, contents).expect("write generated claim-replay linked-overlay receipt");
    }
}
