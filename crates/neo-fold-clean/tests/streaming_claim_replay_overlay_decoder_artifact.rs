//! Exact compact source-decoder audit for the 98 claim-replay overlay arms.

#[path = "support/selective_decoder_lean.rs"]
#[allow(dead_code)]
mod selective_decoder_lean;
#[path = "support/selective_decoder_run_lean.rs"]
mod selective_decoder_run_lean;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};

use neo_fold_clean::frontends::nebula::f_prime::{
    build_production_claim_active_coordinate_overlay_low_norm_r1cs,
    production_claim_active_coordinate_overlay_compact_layout_and_decoder_runs_for_ranges,
    NebulaFPrimeClaimCoordinateOverlaySynthesis,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveArmRowMappingAudit, SelectiveCanonicalOpeningAudit, SelectiveLinearDefinitionAudit,
    SelectiveProjectedDecoderRunProvenance, SelectiveProjectedSourceDecoderRun, SelectiveRewriteAudit,
    SelectiveRewriteKind, SelectiveSourceRowDisposition,
};
use p3_field::PrimeField64;
use selective_decoder_lean::write_decoder_arm_inline;
use selective_decoder_run_lean::write_runs;
use sha2::{Digest, Sha256};

const ACTIVE_ARMS: usize = 98;
const FINAL_COLUMNS: usize = 84_834;
const PROFILE_ID: &str = "nebula-f-prime-streaming-claim-coordinate-overlay-goldilocks-b2-k16-v1";
const ARTIFACT_ID: &str = "rust:nightstream/streaming-claim-coordinate-overlay/source-decoders/v1";
const OVERLAY_ARTIFACT_SHA256: &str = "0c6025a481ce873cc186b03cfe19cd5a8968b3207cc2db8214b5ebd80e095bf0";

#[derive(Clone, Debug, PartialEq, Eq)]
struct DefinitionTermRun {
    column_start: usize,
    column_stride: usize,
    coefficient: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceDefinitionRun {
    source_row_start: usize,
    source_row_stride: usize,
    target_start: usize,
    target_stride: usize,
    count: usize,
    constant: u64,
    terms: Vec<DefinitionTermRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RetainedRowRun {
    source_row_start: usize,
    final_row_start: usize,
    length: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CanonicalOpeningPoint {
    source_field: usize,
    source_digit: usize,
    source_row: usize,
    final_digit: usize,
    final_borrow: usize,
    final_row: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CanonicalOpeningRun {
    source_field_start: usize,
    source_field_stride: usize,
    source_digit_start: usize,
    source_digit_stride: usize,
    source_row_start: usize,
    source_row_stride: usize,
    final_digit_start: usize,
    final_digit_stride: usize,
    final_borrow_start: usize,
    final_borrow_stride: usize,
    final_row_start: usize,
    final_row_stride: usize,
    count: usize,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn decoder_requests() -> Vec<(usize, Range<usize>)> {
    (0..ACTIVE_ARMS)
        .map(|arm| {
            let source = NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(arm + 1)
                .expect("build exact active coordinate-overlay source arm");
            (arm, 1..source.columns())
        })
        .collect()
}

fn definition_run_from_pair(
    first: &SelectiveLinearDefinitionAudit,
    second: &SelectiveLinearDefinitionAudit,
) -> Option<SourceDefinitionRun> {
    let first_row = first.source_row()?;
    let second_row = second.source_row()?;
    if first.constant() != second.constant() || first.terms().len() != second.terms().len() {
        return None;
    }
    let terms = first
        .terms()
        .iter()
        .zip(second.terms())
        .map(|(first, second)| {
            if first.coefficient() != second.coefficient() {
                return None;
            }
            Some(DefinitionTermRun {
                column_start: first.column(),
                column_stride: second.column().checked_sub(first.column())?,
                coefficient: first.coefficient().as_canonical_u64(),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(SourceDefinitionRun {
        source_row_start: first_row,
        source_row_stride: second_row.checked_sub(first_row)?,
        target_start: first.target(),
        target_stride: second.target().checked_sub(first.target())?,
        count: 2,
        constant: first.constant().as_canonical_u64(),
        terms,
    })
}

fn definition_matches_run(
    definition: &SelectiveLinearDefinitionAudit,
    run: &SourceDefinitionRun,
    offset: usize,
) -> bool {
    definition.source_row() == Some(run.source_row_start + run.source_row_stride * offset)
        && definition.target() == run.target_start + run.target_stride * offset
        && definition.constant().as_canonical_u64() == run.constant
        && definition.terms().len() == run.terms.len()
        && definition
            .terms()
            .iter()
            .zip(&run.terms)
            .all(|(term, expected)| {
                term.column() == expected.column_start + expected.column_stride * offset
                    && term.coefficient().as_canonical_u64() == expected.coefficient
            })
}

fn singleton_definition_run(definition: &SelectiveLinearDefinitionAudit) -> SourceDefinitionRun {
    SourceDefinitionRun {
        source_row_start: definition
            .source_row()
            .expect("overlay definitions have exact source-row ownership"),
        source_row_stride: 0,
        target_start: definition.target(),
        target_stride: 0,
        count: 1,
        constant: definition.constant().as_canonical_u64(),
        terms: definition
            .terms()
            .iter()
            .map(|term| DefinitionTermRun {
                column_start: term.column(),
                column_stride: 0,
                coefficient: term.coefficient().as_canonical_u64(),
            })
            .collect(),
    }
}

fn compress_source_definitions(definitions: &[SelectiveLinearDefinitionAudit]) -> Vec<SourceDefinitionRun> {
    let mut runs = Vec::new();
    let mut cursor = 0;
    while cursor < definitions.len() {
        let mut run = definitions
            .get(cursor + 1)
            .and_then(|second| definition_run_from_pair(&definitions[cursor], second))
            .unwrap_or_else(|| singleton_definition_run(&definitions[cursor]));
        while cursor + run.count < definitions.len()
            && definition_matches_run(&definitions[cursor + run.count], &run, run.count)
        {
            run.count += 1;
        }
        for offset in 0..run.count {
            assert!(definition_matches_run(&definitions[cursor + offset], &run, offset));
        }
        cursor += run.count;
        runs.push(run);
    }
    assert_eq!(runs.iter().map(|run| run.count).sum::<usize>(), definitions.len());
    runs
}

fn lean_definition_run(run: &SourceDefinitionRun) -> String {
    let terms = run
        .terms
        .iter()
        .map(|term| {
            format!(
                "{{ columnStart := {}, columnStride := {}, coefficient := {} }}",
                term.column_start, term.column_stride, term.coefficient
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{{ sourceRowStart := {}, sourceRowStride := {}, targetStart := {}, targetStride := {}, count := {}, constant := {}, terms := [{}] }}",
        run.source_row_start,
        run.source_row_stride,
        run.target_start,
        run.target_stride,
        run.count,
        run.constant,
        terms,
    )
}

fn retained_row_runs(mapping: &SelectiveArmRowMappingAudit) -> Vec<RetainedRowRun> {
    let runs = mapping
        .source_runs()
        .iter()
        .filter_map(|run| {
            if run.disposition() != SelectiveSourceRowDisposition::Retained {
                return None;
            }
            let source = run.source_rows();
            let final_row_start = run
                .emitted_start()
                .expect("retained source run has exact emitted-row ownership");
            Some(RetainedRowRun {
                source_row_start: source.start,
                final_row_start,
                length: source.len(),
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        runs.iter().map(|run| run.length).sum::<usize>(),
        mapping.retained_emitted_rows().len(),
        "retained source and emitted row counts must agree"
    );
    runs
}

fn lean_retained_row_run(run: &RetainedRowRun) -> String {
    format!(
        "{{ sourceRowStart := {}, finalRowStart := {}, length := {} }}",
        run.source_row_start, run.final_row_start, run.length
    )
}

fn canonical_opening_points(
    arm: usize,
    openings: &[SelectiveCanonicalOpeningAudit],
    decoder: &SelectiveProjectedDecoderRunProvenance,
    rewrites: &[SelectiveRewriteAudit],
) -> Vec<CanonicalOpeningPoint> {
    let mut decoder_points = BTreeMap::new();
    for template in decoder.repeated_templates() {
        assert_eq!(template.source_width(), 122);
        for instances in template.instances() {
            for index in 0..instances.count() {
                let (source_digit, final_borrow, source_field, final_digit) = instances
                    .instance(index)
                    .expect("bounded canonical decoder template instance");
                assert!(
                    decoder_points
                        .insert((source_field, final_digit, final_borrow), source_digit)
                        .is_none(),
                    "canonical decoder instance key must be unique"
                );
            }
        }
    }

    let mut rewrite_points = BTreeMap::new();
    for rewrite in rewrites
        .iter()
        .filter(|rewrite| rewrite.arm() == arm && rewrite.kind() == SelectiveRewriteKind::ShiftedTernaryCanonical)
    {
        let source_ranges = rewrite.source_rows();
        assert_eq!(source_ranges.len(), 1);
        assert_eq!(source_ranges[0].len(), 124);
        let emitted = rewrite.emitted_rows();
        assert_eq!(emitted.len(), 21);
        assert!(
            rewrite_points
                .insert((emitted.start, emitted.end), source_ranges[0].start)
                .is_none(),
            "canonical emitted-row range must be unique"
        );
    }

    let points = openings
        .iter()
        .map(|opening| {
            assert_eq!(opening.digit_coordinates().len(), 41);
            assert!(opening
                .digit_coordinates()
                .iter()
                .enumerate()
                .all(|(offset, column)| *column == opening.digit_coordinates()[0] + offset));
            assert_eq!(opening.borrow_coordinates().len(), 20);
            assert!(opening
                .borrow_coordinates()
                .iter()
                .enumerate()
                .all(|(offset, column)| *column == opening.borrow_coordinates()[0] + offset));
            let emitted = opening.emitted_rows();
            assert_eq!(emitted.len(), 21);
            let source_field = opening.source_field();
            let final_digit = opening.digit_coordinates()[0];
            let final_borrow = opening.borrow_coordinates()[0];
            CanonicalOpeningPoint {
                source_field,
                source_digit: decoder_points
                    .remove(&(source_field, final_digit, final_borrow))
                    .expect("canonical opening joins one decoder template instance"),
                source_row: rewrite_points
                    .remove(&(emitted.start, emitted.end))
                    .expect("canonical opening joins one source-row rewrite"),
                final_digit,
                final_borrow,
                final_row: emitted.start,
            }
        })
        .collect::<Vec<_>>();
    assert!(decoder_points.is_empty());
    assert!(rewrite_points.is_empty());
    points
}

fn opening_point(opening: &CanonicalOpeningPoint) -> (usize, usize, usize, usize, usize, usize) {
    (
        opening.source_field,
        opening.source_digit,
        opening.source_row,
        opening.final_digit,
        opening.final_borrow,
        opening.final_row,
    )
}

fn opening_run_from_pair(first: &CanonicalOpeningPoint, second: &CanonicalOpeningPoint) -> Option<CanonicalOpeningRun> {
    let first = opening_point(first);
    let second = opening_point(second);
    Some(CanonicalOpeningRun {
        source_field_start: first.0,
        source_field_stride: second.0.checked_sub(first.0)?,
        source_digit_start: first.1,
        source_digit_stride: second.1.checked_sub(first.1)?,
        source_row_start: first.2,
        source_row_stride: second.2.checked_sub(first.2)?,
        final_digit_start: first.3,
        final_digit_stride: second.3.checked_sub(first.3)?,
        final_borrow_start: first.4,
        final_borrow_stride: second.4.checked_sub(first.4)?,
        final_row_start: first.5,
        final_row_stride: second.5.checked_sub(first.5)?,
        count: 2,
    })
}

fn opening_matches_run(opening: &CanonicalOpeningPoint, run: &CanonicalOpeningRun, offset: usize) -> bool {
    let point = opening_point(opening);
    point.0 == run.source_field_start + run.source_field_stride * offset
        && point.1 == run.source_digit_start + run.source_digit_stride * offset
        && point.2 == run.source_row_start + run.source_row_stride * offset
        && point.3 == run.final_digit_start + run.final_digit_stride * offset
        && point.4 == run.final_borrow_start + run.final_borrow_stride * offset
        && point.5 == run.final_row_start + run.final_row_stride * offset
}

fn singleton_opening_run(opening: &CanonicalOpeningPoint) -> CanonicalOpeningRun {
    let point = opening_point(opening);
    CanonicalOpeningRun {
        source_field_start: point.0,
        source_field_stride: 0,
        source_digit_start: point.1,
        source_digit_stride: 0,
        source_row_start: point.2,
        source_row_stride: 0,
        final_digit_start: point.3,
        final_digit_stride: 0,
        final_borrow_start: point.4,
        final_borrow_stride: 0,
        final_row_start: point.5,
        final_row_stride: 0,
        count: 1,
    }
}

fn compress_canonical_openings(openings: &[CanonicalOpeningPoint]) -> Vec<CanonicalOpeningRun> {
    let mut runs = Vec::new();
    let mut cursor = 0;
    while cursor < openings.len() {
        let mut run = openings
            .get(cursor + 1)
            .and_then(|second| opening_run_from_pair(&openings[cursor], second))
            .unwrap_or_else(|| singleton_opening_run(&openings[cursor]));
        while cursor + run.count < openings.len() && opening_matches_run(&openings[cursor + run.count], &run, run.count)
        {
            run.count += 1;
        }
        for offset in 0..run.count {
            assert!(opening_matches_run(&openings[cursor + offset], &run, offset));
        }
        cursor += run.count;
        runs.push(run);
    }
    assert_eq!(runs.iter().map(|run| run.count).sum::<usize>(), openings.len());
    runs
}

fn lean_canonical_opening_run(run: &CanonicalOpeningRun) -> String {
    format!(
        "{{ sourceFieldStart := {}, sourceFieldStride := {}, sourceDigitStart := {}, sourceDigitStride := {}, sourceRowStart := {}, sourceRowStride := {}, finalDigitStart := {}, finalDigitStride := {}, finalBorrowStart := {}, finalBorrowStride := {}, finalRowStart := {}, finalRowStride := {}, count := {} }}",
        run.source_field_start,
        run.source_field_stride,
        run.source_digit_start,
        run.source_digit_stride,
        run.source_row_start,
        run.source_row_stride,
        run.final_digit_start,
        run.final_digit_stride,
        run.final_borrow_start,
        run.final_borrow_stride,
        run.final_row_start,
        run.final_row_stride,
        run.count,
    )
}

fn render_artifact() -> String {
    let requests = decoder_requests();
    let (layout, decoders) =
        production_claim_active_coordinate_overlay_compact_layout_and_decoder_runs_for_ranges(&requests)
            .expect("audit all active coordinate-overlay source decoders");
    assert_eq!(decoders.len(), ACTIVE_ARMS);
    assert_eq!(layout.selector_columns(), (1..=ACTIVE_ARMS).collect::<Vec<_>>());
    assert_eq!(layout.final_columns(), FINAL_COLUMNS);
    let relation = build_production_claim_active_coordinate_overlay_low_norm_r1cs()
        .expect("build exact active coordinate-overlay relation");
    let compiler_audit = relation
        .selective_compiler_audit()
        .expect("active coordinate-overlay compiler audit");
    let definitions = compiler_audit.source_arm_linear_definitions();
    assert_eq!(definitions.len(), ACTIVE_ARMS);
    let definition_runs = definitions
        .iter()
        .map(|arm| compress_source_definitions(arm))
        .collect::<Vec<_>>();
    let retained_row_runs = compiler_audit
        .rows()
        .arms()
        .iter()
        .map(retained_row_runs)
        .collect::<Vec<_>>();
    assert_eq!(retained_row_runs.len(), ACTIVE_ARMS);
    let canonical_openings = compiler_audit.canonical_openings();
    assert_eq!(canonical_openings.len(), ACTIVE_ARMS);
    let canonical_opening_runs = canonical_openings
        .iter()
        .enumerate()
        .map(|(arm, openings)| {
            compress_canonical_openings(&canonical_opening_points(
                arm,
                openings,
                &decoders[arm],
                compiler_audit.rows().rewrites(),
            ))
        })
        .collect::<Vec<_>>();

    let mut profiles = Vec::<Vec<SelectiveProjectedSourceDecoderRun>>::new();
    let mut profile_indices = Vec::with_capacity(ACTIVE_ARMS);
    for (decoder, (arm, source_range)) in decoders.iter().zip(&requests) {
        assert_eq!(decoder.arm(), *arm);
        assert_eq!(decoder.source_range(), source_range.clone());
        assert_eq!(decoder.final_columns(), layout.final_columns());
        assert_eq!(decoder.repeated_templates().len(), 1);
        let rules = decoder.repeated_templates()[0].relative_runs().to_vec();
        let profile = profiles
            .iter()
            .position(|candidate| candidate == &rules)
            .unwrap_or_else(|| {
                profiles.push(rules);
                profiles.len() - 1
            });
        profile_indices.push(profile);
    }

    let mut payload = String::new();
    for (profile, rules) in profiles.iter().enumerate() {
        write_runs(
            &mut payload,
            &format!("templateProfile{profile:02}Rules00"),
            "RawRun",
            rules,
        );
    }
    assert_eq!(profiles.len(), 1, "all active overlay arms share one exact template");
    for (arm, decoder) in decoders.iter().enumerate() {
        let source_definition_runs = format!(
            "[{}]",
            definition_runs[arm]
                .iter()
                .map(lean_definition_run)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let opening_runs = format!(
            "[{}]",
            canonical_opening_runs[arm]
                .iter()
                .map(lean_canonical_opening_run)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let retained_runs = format!(
            "[{}]",
            retained_row_runs[arm]
                .iter()
                .map(lean_retained_row_run)
                .collect::<Vec<_>>()
                .join(", ")
        );
        write_decoder_arm_inline(
            &mut payload,
            &format!("arm{arm:02}"),
            decoder,
            &format!("templateProfile{:02}Rules", profile_indices[arm]),
            &source_definition_runs,
            &retained_runs,
            &opening_runs,
        );
    }
    let arm_names = (0..ACTIVE_ARMS)
        .map(|arm| format!("arm{arm:02}"))
        .collect::<Vec<_>>()
        .join(", ");
    let artifact_sha256 = sha256_hex(payload.as_bytes());
    let rendered = format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema\n\n\
         /-! GENERATED FILE. DO NOT EDIT. Exact compact Rust source decoders\n\
         for all 98 active production claim-coordinate overlay arms. -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlayDecoder\n\n\
         open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema\n\n\
         def artifactSha256 : String := \"{artifact_sha256}\"\n\
         def schemaVersion : Nat := 4\n\
         def profileId : String := \"{PROFILE_ID}\"\n\
         def artifactIdentity : String := \"{ARTIFACT_ID}\"\n\
         def overlayArtifactSha256 : String := \"{OVERLAY_ARTIFACT_SHA256}\"\n\
         def activeArmCount : Nat := {ACTIVE_ARMS}\n\
         def finalColumnCount : Nat := {FINAL_COLUMNS}\n\n\
         {payload}\
         def activeArms : List RawArm := [{arm_names}]\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlayDecoder\n"
    );
    assert!(rendered.lines().count() < 1_500, "generated decoder artifact line cap");
    rendered
}

fn generated_artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingClaimReplayCoordinateOverlayDecoder.lean",
    )
}

#[test]
fn production_claim_replay_overlay_decoder_artifact_is_current() {
    let path = generated_artifact_path();
    let rendered = render_artifact();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected claim-replay overlay decoder artifact");
        panic!(
            "claim-replay overlay decoder artifact drifted; inspect {}",
            expected.display()
        );
    }
}

#[test]
#[ignore = "writes the exact generated Lean decoder artifact"]
fn regenerate_production_claim_replay_overlay_decoder_artifact() {
    std::fs::write(generated_artifact_path(), render_artifact())
        .expect("write generated claim-replay overlay decoder artifact");
}

#[test]
fn production_claim_replay_overlay_linear_definition_profiles_are_measured() {
    let relation = build_production_claim_active_coordinate_overlay_low_norm_r1cs()
        .expect("build exact active coordinate-overlay relation");
    let definitions = relation
        .selective_compiler_audit()
        .expect("active coordinate-overlay compiler audit")
        .source_arm_linear_definitions();
    assert_eq!(definitions.len(), ACTIVE_ARMS);

    let mut global_shapes = BTreeSet::new();
    let counts = definitions.iter().map(Vec::len).collect::<Vec<_>>();
    for arm_definitions in definitions {
        for definition in arm_definitions {
            let relative_terms = definition
                .terms()
                .iter()
                .map(|term| {
                    (
                        term.column() as isize - definition.target() as isize,
                        term.coefficient().as_canonical_u64(),
                    )
                })
                .collect::<Vec<_>>();
            global_shapes.insert((
                definition.source_row().is_some(),
                definition.constant().as_canonical_u64(),
                relative_terms,
            ));
        }
    }
    eprintln!(
        "claim-replay overlay definition audit: arms={}, definitions_min={}, definitions_max={}, relative_shapes={}",
        definitions.len(),
        counts.iter().min().expect("one arm"),
        counts.iter().max().expect("one arm"),
        global_shapes.len(),
    );

    for arm in [0, 1, 61, 69, 97] {
        let arm_definitions = &definitions[arm];
        let mut arities = BTreeMap::<usize, usize>::new();
        for definition in arm_definitions {
            *arities.entry(definition.terms().len()).or_default() += 1;
        }
        eprintln!(
            "claim-replay overlay arm {arm} definitions: count={}, arities={arities:?}, source_rows={}, trace_owned={}",
            arm_definitions.len(),
            arm_definitions
                .iter()
                .filter(|definition| definition.source_row().is_some())
                .count(),
            arm_definitions
                .iter()
                .filter(|definition| definition.source_row().is_none())
                .count(),
        );
        for definition in arm_definitions
            .iter()
            .take(6)
            .chain(arm_definitions.iter().rev().take(6).rev())
        {
            let terms = definition
                .terms()
                .iter()
                .map(|term| (term.column(), term.coefficient().as_canonical_u64()))
                .collect::<Vec<_>>();
            eprintln!(
                "  row={:?} target={} constant={} terms={terms:?}",
                definition.source_row(),
                definition.target(),
                definition.constant().as_canonical_u64(),
            );
        }
    }
}
