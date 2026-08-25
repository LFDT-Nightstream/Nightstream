//! Exact source-shape and link tests for the split PiRLC family relation.

#[path = "../support/selective_decoder_lean.rs"]
mod selective_decoder_lean;
#[path = "../support/selective_decoder_run_lean.rs"]
mod selective_decoder_run_lean;

use neo_fold_clean::frontends::nebula::f_prime::{
    build_production_pi_rlc_family_body_low_norm_r1cs, build_production_pi_rlc_family_overlay_low_norm_r1cs,
    production_pi_rlc_family_body_algebra_retained_audit, production_pi_rlc_family_body_carry_retained_audit,
    production_pi_rlc_family_body_compiler_audit, production_pi_rlc_family_body_decoder_runs,
    production_pi_rlc_family_body_opening_rows_audit, production_pi_rlc_family_body_residual_retained_audit,
    production_pi_rlc_family_body_row_ledger, production_pi_rlc_family_normalized_link_audit,
    production_pi_rlc_family_overlay_kind_map, production_pi_rlc_family_overlay_link_runs,
    production_pi_rlc_family_overlay_links, production_pi_rlc_family_overlay_retained_audit,
    streaming_phase_semantic_digest, NebulaFPrimePiRlcBodyFixedFamily, NebulaFPrimePiRlcBodyRewriteKind,
    NebulaFPrimePiRlcFamilyBodyRowLedger, NebulaFPrimePiRlcFamilyBodySynthesis,
    NebulaFPrimePiRlcFamilyNormalizedLinkAudit, NebulaFPrimePiRlcFamilyOverlaySynthesis,
    NebulaFPrimePiRlcFamilyReplayArmKind, PI_RLC_FAMILY_BODY_EVEN_COLUMNS, PI_RLC_FAMILY_BODY_EVEN_ROWS,
    PI_RLC_FAMILY_BODY_EVEN_SOURCE_ROWS, PI_RLC_FAMILY_BODY_ODD_COLUMNS, PI_RLC_FAMILY_BODY_ODD_ROWS,
    PI_RLC_FAMILY_BODY_ODD_SOURCE_ROWS, PI_RLC_FAMILY_BODY_SOURCE_ROWS, PI_RLC_FAMILY_COUNT, PI_RLC_FAMILY_LINK_FIELDS,
    PI_RLC_FAMILY_OVERLAY_COLUMNS, PI_RLC_FAMILY_OVERLAY_ROWS, PI_RLC_GLOBAL_INPUT_FIELDS, PI_RLC_MESSAGE_COLUMNS,
    STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS, STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY, STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveEmittedRowFamily, SelectiveProjectedDecoderRunProvenance, SelectiveSourceRowDisposition,
};
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use selective_decoder_lean::write_decoder_arm;
use selective_decoder_run_lean::write_runs;
use std::collections::BTreeMap;
use std::fmt::Write as _;

const CLAIM_OVERLAY_KINDS: usize = 26;
const STATE_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/nebula/f-prime/streaming-pirlc-state/v1";
const BODY_DECODER_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder.lean";
const BODY_ROW_LEDGER_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger.lean";
const BODY_OPENING_ROWS_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows.lean";
const BODY_ALGEBRA_RETAINED_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained.lean";
const BODY_CARRY_RETAINED_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.lean";
const BODY_RESIDUAL_RETAINED_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained.lean";
const OVERLAY_RETAINED_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.lean";
const NORMALIZED_LINK_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.lean";

fn assert_generated_artifact_matches_committed(rendered: &str, path: &str, description: &str) {
    let committed = std::fs::read_to_string(path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write expected PiRLC family artifact");
        panic!("{description}; wrote {expected}");
    }
}

fn render_body_decoder_artifact(decoders: &[SelectiveProjectedDecoderRunProvenance]) -> String {
    let [even, odd] = decoders else {
        panic!("PiRLC decoder artifact requires two parity arms")
    };
    assert_eq!(even.repeated_templates().len(), odd.repeated_templates().len());
    for (even_template, odd_template) in even
        .repeated_templates()
        .iter()
        .zip(odd.repeated_templates())
    {
        assert_eq!(even_template.source_width(), odd_template.source_width());
        assert_eq!(even_template.relative_runs(), odd_template.relative_runs());
    }

    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema\n\n\
/-! Generated file: exact compact source-to-final decoder for both production\n\
PiRLC parity bodies.\n\n\
Owns: the two source ranges, final normalized column bound, three shared\n\
decoder templates, exact affine template instances, and residual strided\n\
rule batches emitted from the supported b = 2 production selective layout.\n\n\
Does not own: source-row semantics, matrix soundness, selector authority,\n\
assignment values, or lifecycle soundness.\n\n\
Emits constraints: no. Rust checks every expanded rule against the prepared\n\
layout before it renders this inert data. Lean validates the compact cover.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema\n"
    )
    .expect("render PiRLC decoder header");

    for (index, template) in even.repeated_templates().iter().enumerate() {
        write_runs(
            &mut rendered,
            &format!("templateRules{index:02}"),
            "RawRun",
            template.relative_runs(),
        );
    }
    for (label, decoder) in [("even", even), ("odd", odd)] {
        write_decoder_arm(&mut rendered, label, decoder, "templateRules");
    }
    writeln!(
        rendered,
        "end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder"
    )
    .expect("render PiRLC decoder footer");
    rendered
}

fn assert_body_decoder_artifact_matches_committed(decoders: &[SelectiveProjectedDecoderRunProvenance]) {
    let rendered = render_body_decoder_artifact(decoders);
    if std::env::var_os("NIGHTSTREAM_PRINT_PI_RLC_BODY_DECODER_ARTIFACT").is_some() {
        println!("BEGIN_PI_RLC_BODY_DECODER_ARTIFACT\n{rendered}END_PI_RLC_BODY_DECODER_ARTIFACT");
    }
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), BODY_DECODER_ARTIFACT_PATH);
    assert_generated_artifact_matches_committed(
        &rendered,
        &path,
        "production PiRLC body decoder Lean artifact drifted",
    );
}

fn lean_fixed_family(family: NebulaFPrimePiRlcBodyFixedFamily) -> &'static str {
    match family {
        NebulaFPrimePiRlcBodyFixedFamily::SelectorDomain => ".selectorDomain",
        NebulaFPrimePiRlcBodyFixedFamily::SharedDomain => ".sharedDomain",
        NebulaFPrimePiRlcBodyFixedFamily::ArmDomain => ".armDomain",
        NebulaFPrimePiRlcBodyFixedFamily::OneHot => ".oneHot",
        NebulaFPrimePiRlcBodyFixedFamily::PublicPadding => ".publicPadding",
        NebulaFPrimePiRlcBodyFixedFamily::PrivatePadding => ".privatePadding",
        NebulaFPrimePiRlcBodyFixedFamily::RingPadding => ".ringPadding",
    }
}

fn lean_rewrite_kind(kind: NebulaFPrimePiRlcBodyRewriteKind) -> &'static str {
    match kind {
        NebulaFPrimePiRlcBodyRewriteKind::Poseidon2 => ".poseidon2",
        NebulaFPrimePiRlcBodyRewriteKind::ShiftedTernaryCanonical => ".shiftedTernaryCanonical",
        NebulaFPrimePiRlcBodyRewriteKind::LinearDefinition => ".linearDefinition",
    }
}

fn lean_arm(arm: Option<usize>) -> String {
    arm.map_or_else(|| "none".to_owned(), |arm| format!("some {arm}"))
}

fn render_body_row_ledger_artifact(ledger: &NebulaFPrimePiRlcFamilyBodyRowLedger) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema\n\n\
/-! Generated file: exact compact row-owner ledger for the normalized\n\
production PiRLC parity bodies.\n\n\
Owns: fixed emitted intervals, retained source-to-emitted intervals, and\n\
affine rewrite batches copied from the production selective compiler audit.\n\n\
Does not own: row semantics, port images, matrix actions, assignment values,\n\
selector authority, or lifecycle soundness.\n\n\
Emits constraints: no. Rust expands this data and checks exact equality with\n\
the compiler audit before rendering it. Lean checks all ownership covers.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema\n\n\
def ledger : RawLedger where"
    )
    .expect("render PiRLC row-ledger preamble");
    writeln!(
        rendered,
        "  schemaVersion := 1\n  rows := {}\n  columns := {}\n  evenSourceRows := {}\n  oddSourceRows := {}\n  rewriteCount := {}\n  evenLinearDefinitionCount := {}\n  oddLinearDefinitionCount := {}",
        ledger.rows(),
        ledger.columns(),
        ledger.source_rows()[0],
        ledger.source_rows()[1],
        ledger.rewrite_count(),
        ledger.linear_definition_counts()[0],
        ledger.linear_definition_counts()[1],
    )
    .expect("render PiRLC row-ledger header");
    writeln!(rendered, "  fixedRuns :=\n    [").expect("render fixed run header");
    for (index, run) in ledger.fixed_runs().iter().enumerate() {
        let separator = if index == 0 { "      " } else { "    , " };
        writeln!(
            rendered,
            "{separator}{{ start := {}, length := {}, family := {}, arm := {} }}",
            run.start(),
            run.length(),
            lean_fixed_family(run.family()),
            lean_arm(run.arm()),
        )
        .expect("render fixed row-owner run");
    }
    writeln!(rendered, "    ]\n  retainedRuns :=\n    [").expect("render retained run header");
    for (index, run) in ledger.retained_runs().iter().enumerate() {
        let separator = if index == 0 { "      " } else { "    , " };
        writeln!(
            rendered,
            "{separator}{{ arm := {}, sourceStart := {}, length := {}, emittedStart := {} }}",
            run.arm(),
            run.source_start(),
            run.length(),
            run.emitted_start(),
        )
        .expect("render retained row-owner run");
    }
    writeln!(rendered, "    ]\n  rewriteBatches :=\n    [").expect("render rewrite-batch header");
    for (index, batch) in ledger.rewrite_batches().iter().enumerate() {
        let separator = if index == 0 { "      " } else { "    , " };
        writeln!(
            rendered,
            "{separator}{{ rewriteStart := {}, count := {}, rewriteStride := {}, arm := {}, kind := {}, sourceStart := {}, sourceStride := {}, sourceWidth := {}, emittedStart := {}, emittedStride := {}, emittedWidth := {} }}",
            batch.rewrite_start(),
            batch.count(),
            batch.rewrite_stride(),
            batch.arm(),
            lean_rewrite_kind(batch.kind()),
            batch.source_start(),
            batch.source_stride(),
            batch.source_width(),
            batch.emitted_start(),
            batch.emitted_stride(),
            batch.emitted_width(),
        )
        .expect("render rewrite batch");
    }
    writeln!(
        rendered,
        "    ]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger"
    )
    .expect("render PiRLC row-ledger footer");
    rendered
}

fn assert_body_row_ledger_artifact_matches_committed(ledger: &NebulaFPrimePiRlcFamilyBodyRowLedger) {
    let rendered = render_body_row_ledger_artifact(ledger);
    if std::env::var_os("NIGHTSTREAM_PRINT_PI_RLC_BODY_ROW_LEDGER_ARTIFACT").is_some() {
        println!("BEGIN_PI_RLC_BODY_ROW_LEDGER_ARTIFACT\n{rendered}END_PI_RLC_BODY_ROW_LEDGER_ARTIFACT");
        return;
    }
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), BODY_ROW_LEDGER_ARTIFACT_PATH);
    assert_generated_artifact_matches_committed(
        &rendered,
        &path,
        "production PiRLC body row-ledger Lean artifact drifted",
    );
}

fn lean_nat_list(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn lean_u64_list(values: &[u64]) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn lean_bool_list(values: &[bool]) -> String {
    values
        .iter()
        .map(bool::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_body_opening_rows_artifact(
    audit: &neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimePiRlcBodyOpeningRowsAudit,
) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRowsSchema\n\n\
/-! Generated file: compact receipt for the exhaustive normalized production\n\
PiRLC opening-row scan.\n\n\
Owns: exact source trace geometry, active digit-domain rows, zero-word rows,\n\
two-trit canonical rows, final opening slots, chunk classes, and nonzero\n\
censuses for both parity arms.\n\n\
Does not own: assignment values, outer norm authority, semantic canonicality,\n\
recursive orchestration, or lifecycle soundness. Lean checks the arithmetic\n\
properties of this inert receipt.\n\n\
Emits constraints: no. Rust checks every selected source and final matrix row\n\
before it renders this data.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRowsSchema\n\n\
def audit : RawAudit where"
    )
    .expect("render PiRLC opening-row preamble");
    writeln!(
        rendered,
        "  schemaVersion := {}\n  armCount := {}\n  openingCount := {}\n  digitCount := {}\n  borrowCount := {}\n  chunkCount := {}\n  sourceZeroRowStart := {}\n  sourceZeroDigitStart := {}\n  sourceFieldStart := {}\n  sourceDigitStart := {}\n  sourceDigitStride := {}\n  sourceCanonicalRowStart := {}\n  sourceCanonicalRowStride := {}\n  centeredRowStart := {}\n  centeredRowCount := {}\n  zeroEmittedStarts := [{}]\n  canonicalEmittedStarts := [{}]\n  selectorColumns := [{}]\n  finalDigitStart := {}\n  finalDigitStride := {}\n  finalZeroStart := {}\n  finalBorrowStart := {}\n  finalBorrowStride := {}\n  finalRows := {}\n  finalColumns := {}\n  normalizedChunkBounds := [{}]\n  complementedChunks := [{}]\n  sourceZeroNnz := [{}]\n  finalPortNnz := [{}]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows",
        audit.schema_version(),
        audit.arm_count(),
        audit.opening_count(),
        audit.digit_count(),
        audit.borrow_count(),
        audit.chunk_count(),
        audit.source_zero_row_start(),
        audit.source_zero_digit_start(),
        audit.source_field_start(),
        audit.source_digit_start(),
        audit.source_digit_stride(),
        audit.source_canonical_row_start(),
        audit.source_canonical_row_stride(),
        audit.centered_row_start(),
        audit.centered_row_count(),
        lean_nat_list(&audit.zero_emitted_starts()),
        lean_nat_list(&audit.canonical_emitted_starts()),
        lean_nat_list(&audit.selector_columns()),
        audit.final_digit_start(),
        audit.final_digit_stride(),
        audit.final_zero_start(),
        audit.final_borrow_start(),
        audit.final_borrow_stride(),
        audit.final_rows(),
        audit.final_columns(),
        lean_nat_list(&audit.normalized_chunk_bounds()),
        lean_bool_list(&audit.complemented_chunks()),
        lean_nat_list(&audit.source_zero_nnz()),
        lean_nat_list(&audit.final_port_nnz()),
    )
    .expect("render PiRLC opening-row body");
    rendered
}

fn assert_body_opening_rows_artifact_matches_committed(
    audit: &neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimePiRlcBodyOpeningRowsAudit,
) {
    let rendered = render_body_opening_rows_artifact(audit);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), BODY_OPENING_ROWS_ARTIFACT_PATH);
    assert_generated_artifact_matches_committed(&rendered, &path, "production PiRLC opening-row Lean artifact drifted");
}

fn render_body_algebra_retained_artifact(
    audit: &neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimePiRlcBodyAlgebraRetainedAudit,
) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetainedSchema\n\n\
/-! Generated file: compact receipt for the exhaustive normalized production\n\
PiRLC algebra retained-row scan.\n\n\
Owns: dimensions, source and final slot starts, low-norm widths and radices,\n\
retained row starts, selector columns, and exact nonzero censuses observed by\n\
the Rust scan.\n\n\
Does not own: semantic truth, matrix authority, assignment values, selector\n\
authority, recursive orchestration, or lifecycle soundness. Lean recomputes\n\
the arithmetic properties of this inert receipt.\n\n\
Emits constraints: no. Rust checks every selected source and final matrix row\n\
before it renders this data.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetainedSchema\n\n\
def audit : RawAudit where"
    )
    .expect("render PiRLC algebra-retained preamble");
    writeln!(
        rendered,
        "  schemaVersion := {}\n  sourceRows := {}\n  localColumns := {}\n  sourceColumnShift := {}\n  finalRows := {}\n  finalColumns := {}\n  selectorColumns := [{}]\n  emittedStarts := [{}]\n  sourceStarts := [{}]\n  finalStarts := [{}]\n  widths := [{}]\n  radices := [{}]\n  sourceNnz := [{}]\n  finalPortNnz := [{}]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained",
        audit.schema_version(),
        audit.source_rows(),
        audit.local_columns(),
        audit.source_column_shift(),
        audit.final_rows(),
        audit.final_columns(),
        lean_nat_list(&audit.selector_columns()),
        lean_nat_list(&audit.emitted_starts()),
        lean_nat_list(&audit.source_starts()),
        lean_nat_list(&audit.final_starts()),
        lean_nat_list(&audit.widths()),
        lean_u64_list(&audit.radices()),
        lean_nat_list(&audit.source_nnz()),
        lean_nat_list(&audit.final_port_nnz()),
    )
    .expect("render PiRLC algebra-retained body");
    rendered
}

fn assert_body_algebra_retained_artifact_matches_committed(
    audit: &neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimePiRlcBodyAlgebraRetainedAudit,
) {
    let rendered = render_body_algebra_retained_artifact(audit);
    if std::env::var_os("NIGHTSTREAM_PRINT_PI_RLC_BODY_ALGEBRA_RETAINED_ARTIFACT").is_some() {
        println!("BEGIN_PI_RLC_BODY_ALGEBRA_RETAINED_ARTIFACT\n{rendered}END_PI_RLC_BODY_ALGEBRA_RETAINED_ARTIFACT");
        return;
    }
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), BODY_ALGEBRA_RETAINED_ARTIFACT_PATH);
    assert_generated_artifact_matches_committed(
        &rendered,
        &path,
        "production PiRLC body algebra-retained Lean artifact drifted",
    );
}

fn render_body_carry_retained_artifact(
    audit: &neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimePiRlcBodyCarryRetainedAudit,
) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetainedSchema\n\n\
/-! Generated file: compact receipt for the exhaustive normalized production\n\
PiRLC carry retained-row scan.\n\n\
Owns: dimensions, the source row interval, source and final slot starts,\n\
low-norm widths and radices, retained row starts, selector columns, and exact\n\
nonzero censuses observed by the Rust scan.\n\n\
Does not own: semantic truth, matrix authority, assignment values, selector\n\
authority, challenge range, recursive orchestration, or lifecycle soundness.\n\
Lean recomputes the arithmetic properties of this inert receipt.\n\n\
Emits constraints: no. Rust checks every selected source and final matrix row\n\
before it renders this data.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetainedSchema\n\n\
def audit : RawAudit where"
    )
    .expect("render PiRLC carry-retained preamble");
    writeln!(
        rendered,
        "  schemaVersion := {}\n  sourceRowStart := {}\n  sourceRows := {}\n  localColumns := {}\n  sourceColumnShift := {}\n  finalRows := {}\n  finalColumns := {}\n  selectorColumns := [{}]\n  emittedStarts := [{}]\n  sourceStarts := [{}]\n  finalStarts := [{}]\n  widths := [{}]\n  radices := [{}]\n  sourceNnz := [{}]\n  finalPortNnz := [{}]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained",
        audit.schema_version(),
        audit.source_row_start(),
        audit.source_rows(),
        audit.local_columns(),
        audit.source_column_shift(),
        audit.final_rows(),
        audit.final_columns(),
        lean_nat_list(&audit.selector_columns()),
        lean_nat_list(&audit.emitted_starts()),
        lean_nat_list(&audit.source_starts()),
        lean_nat_list(&audit.final_starts()),
        lean_nat_list(&audit.widths()),
        lean_u64_list(&audit.radices()),
        lean_nat_list(&audit.source_nnz()),
        lean_nat_list(&audit.final_port_nnz()),
    )
    .expect("render PiRLC carry-retained body");
    rendered
}

fn assert_body_carry_retained_artifact_matches_committed(
    audit: &neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimePiRlcBodyCarryRetainedAudit,
) {
    let rendered = render_body_carry_retained_artifact(audit);
    if std::env::var_os("NIGHTSTREAM_PRINT_PI_RLC_BODY_CARRY_RETAINED_ARTIFACT").is_some() {
        println!("BEGIN_PI_RLC_BODY_CARRY_RETAINED_ARTIFACT\n{rendered}END_PI_RLC_BODY_CARRY_RETAINED_ARTIFACT");
        return;
    }
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), BODY_CARRY_RETAINED_ARTIFACT_PATH);
    assert_generated_artifact_matches_committed(
        &rendered,
        &path,
        "production PiRLC body carry-retained Lean artifact drifted",
    );
}

fn render_body_residual_retained_artifact(
    audit: &neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimePiRlcBodyResidualRetainedAudit,
) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetainedSchema\n\n\
/-! Generated file: compact receipt for the exhaustive normalized production\n\
PiRLC residual retained-row scan.\n\n\
Owns: dimensions, the source row interval, source and final slot starts,\n\
low-norm widths and radices, retained row starts, selector columns, and exact\n\
nonzero censuses observed by the Rust scan.\n\n\
Does not own: semantic truth, matrix authority, assignment values, selector\n\
authority, the local commitment output, recursive orchestration, or lifecycle\n\
soundness. Lean recomputes the arithmetic properties of this inert receipt.\n\n\
Emits constraints: no. Rust checks every selected source and final matrix row\n\
before it renders this data.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetainedSchema\n\n\
def audit : RawAudit where"
    )
    .expect("render PiRLC residual-retained preamble");
    writeln!(
        rendered,
        "  schemaVersion := {}\n  sourceRowStart := {}\n  sourceRows := {}\n  localColumns := {}\n  sourceColumnShift := {}\n  finalRows := {}\n  finalColumns := {}\n  selectorColumns := [{}]\n  emittedStarts := [{}]\n  sourceStarts := [{}]\n  finalStarts := [{}]\n  widths := [{}]\n  radices := [{}]\n  sourceNnz := [{}]\n  finalPortNnz := [{}]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained",
        audit.schema_version(),
        audit.source_row_start(),
        audit.source_rows(),
        audit.local_columns(),
        audit.source_column_shift(),
        audit.final_rows(),
        audit.final_columns(),
        lean_nat_list(&audit.selector_columns()),
        lean_nat_list(&audit.emitted_starts()),
        lean_nat_list(&audit.source_starts()),
        lean_nat_list(&audit.final_starts()),
        lean_nat_list(&audit.widths()),
        lean_u64_list(&audit.radices()),
        lean_nat_list(&audit.source_nnz()),
        lean_nat_list(&audit.final_port_nnz()),
    )
    .expect("render PiRLC residual-retained body");
    rendered
}

fn assert_body_residual_retained_artifact_matches_committed(
    audit: &neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimePiRlcBodyResidualRetainedAudit,
) {
    let rendered = render_body_residual_retained_artifact(audit);
    if std::env::var_os("NIGHTSTREAM_PRINT_PI_RLC_BODY_RESIDUAL_RETAINED_ARTIFACT").is_some() {
        println!("BEGIN_PI_RLC_BODY_RESIDUAL_RETAINED_ARTIFACT\n{rendered}END_PI_RLC_BODY_RESIDUAL_RETAINED_ARTIFACT");
        return;
    }
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), BODY_RESIDUAL_RETAINED_ARTIFACT_PATH);
    assert_generated_artifact_matches_committed(
        &rendered,
        &path,
        "production PiRLC body residual-retained Lean artifact drifted",
    );
}

fn lean_seed_rows(values: &[Vec<[u8; 32]>]) -> String {
    values
        .iter()
        .map(|row| {
            let chunks = row
                .iter()
                .map(|seed| {
                    format!(
                        "[{}]",
                        seed.iter()
                            .map(u8::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{chunks}]")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_overlay_retained_artifact(
    audit: &neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimePiRlcFamilyOverlayRetainedAudit,
) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetainedSchema\n\n\
/-! Generated file: compact receipt for the exhaustive normalized production\n\
PiRLC family-overlay retained-row scan.\n\n\
Owns: dimensions, selector and retained-row affine geometry, source and final\n\
slot starts, low-norm widths and radices, the exact verifier seed chunks, and\n\
compact-block and explicit nonzero censuses observed by the Rust scan.\n\n\
Does not own: semantic truth, matrix authority in Lean, assignment values,\n\
body-to-overlay links, selector authority, recursive orchestration, or\n\
lifecycle soundness. Lean recomputes the arithmetic properties of this inert\n\
receipt.\n\n\
Emits constraints: no. Rust checks all 110 source blocks, all 110 final\n\
blocks, and every retained explicit final row before it renders this data.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetainedSchema\n\n\
def audit : RawAudit where"
    )
    .expect("render PiRLC overlay-retained preamble");
    writeln!(
        rendered,
        "  schemaVersion := {}\n  familyCount := {}\n  sourceRows := {}\n  sourceColumns := {}\n  finalRows := {}\n  finalColumns := {}\n  selectorStart := {}\n  selectorCount := {}\n  retainedStart := {}\n  retainedStride := {}\n  sourceStarts := [{}]\n  finalStarts := [{}]\n  widths := [{}]\n  radices := [{}]\n  chunkSize := {}\n  chunkSeedsByRow := [{}]\n  sourceExplicitNnz := [{}]\n  finalBlockCounts := [{}]\n  finalExplicitPortNnz := [{}]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained",
        audit.schema_version(),
        audit.family_count(),
        audit.source_rows(),
        audit.source_columns(),
        audit.final_rows(),
        audit.final_columns(),
        audit.selector_start(),
        audit.selector_count(),
        audit.retained_start(),
        audit.retained_stride(),
        lean_nat_list(&audit.source_starts()),
        lean_nat_list(&audit.final_starts()),
        lean_nat_list(&audit.widths()),
        lean_u64_list(&audit.radices()),
        audit.chunk_size(),
        lean_seed_rows(audit.chunk_seeds_by_row()),
        lean_nat_list(&audit.source_explicit_nnz()),
        lean_nat_list(&audit.final_block_counts()),
        lean_nat_list(&audit.final_explicit_port_nnz()),
    )
    .expect("render PiRLC overlay-retained body");
    rendered
}

fn assert_overlay_retained_artifact_matches_committed(
    audit: &neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimePiRlcFamilyOverlayRetainedAudit,
) {
    let rendered = render_overlay_retained_artifact(audit);
    if std::env::var_os("NIGHTSTREAM_PRINT_PI_RLC_OVERLAY_RETAINED_ARTIFACT").is_some() {
        println!("BEGIN_PI_RLC_OVERLAY_RETAINED_ARTIFACT\n{rendered}END_PI_RLC_OVERLAY_RETAINED_ARTIFACT");
        return;
    }
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), OVERLAY_RETAINED_ARTIFACT_PATH);
    assert_generated_artifact_matches_committed(
        &rendered,
        &path,
        "production PiRLC family-overlay retained Lean artifact drifted",
    );
}

fn render_normalized_link_artifact(audit: &NebulaFPrimePiRlcFamilyNormalizedLinkAudit) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLinkSchema\n\n\
/-! Generated file: compact receipt for the normalized production PiRLC\n\
body-overlay link audit.\n\n\
Owns: the public-prefix shift, both final column bounds, parity kind codes,\n\
the three source-field runs, and the exact final low-norm slots and radices.\n\n\
Does not own: semantic truth, selector authority, shifted-ternary\n\
canonicality, row satisfaction, recursive orchestration, or lifecycle\n\
soundness. Lean checks the arithmetic properties of this inert receipt.\n\n\
Emits constraints: no. Rust checks both parity body maps against the prepared\n\
production layout. The separate overlay receipt checks all 110 overlay maps.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink\n\n\
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLinkSchema\n\n\
def runs : List RawRun :=\n  ["
    )
    .expect("render normalized PiRLC link preamble");
    for (index, run) in audit.runs().into_iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ bodySourceStart := {}, overlaySourceStart := {}, outerCount := {}, bodySourceStride := {}, overlaySourceStride := {}, fieldCount := {}, bodyFinalStart := {}, overlayFinalStart := {}, finalOuterStride := {}, finalFieldStride := {}, width := {}, radix := {} }}",
            run.body_source_start(),
            run.overlay_source_start(),
            run.outer_count(),
            run.body_source_stride(),
            run.overlay_source_stride(),
            run.field_count(),
            run.body_final_start(),
            run.overlay_final_start(),
            run.final_outer_stride(),
            run.final_field_stride(),
            run.width(),
            run.radix(),
        )
        .expect("render normalized PiRLC link run");
    }
    writeln!(
        rendered,
        "  ]\n\n\
def audit : RawAudit where\n\
  schemaVersion := {}\n\
  familyCount := {}\n\
  parityCount := {}\n\
  publicOutputCount := {}\n\
  bodyFinalColumns := {}\n\
  overlayFinalColumns := {}\n\
  linkCountPerFamily := {}\n\
  totalLinkCount := {}\n\
  phaseKinds := [{}]\n\
  runs := runs\n\n\
end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink",
        audit.schema_version(),
        audit.family_count(),
        audit.parity_count(),
        audit.public_output_count(),
        audit.body_final_columns(),
        audit.overlay_final_columns(),
        audit.link_count_per_family(),
        audit.total_link_count(),
        lean_nat_list(&audit.phase_kinds()),
    )
    .expect("render normalized PiRLC link body");
    rendered
}

fn assert_normalized_link_artifact_matches_committed(audit: &NebulaFPrimePiRlcFamilyNormalizedLinkAudit) {
    let rendered = render_normalized_link_artifact(audit);
    if std::env::var_os("NIGHTSTREAM_PRINT_PI_RLC_NORMALIZED_LINK_ARTIFACT").is_some() {
        println!("BEGIN_PI_RLC_NORMALIZED_LINK_ARTIFACT\n{rendered}END_PI_RLC_NORMALIZED_LINK_ARTIFACT");
        return;
    }
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), NORMALIZED_LINK_ARTIFACT_PATH);
    assert_generated_artifact_matches_committed(
        &rendered,
        &path,
        "production PiRLC normalized body-overlay link Lean artifact drifted",
    );
}

#[test]
#[ignore = "expensive exhaustive normalized PiRLC carry retained-row audit"]
fn pi_rlc_family_body_carry_retained_rows_match_exact_recipe() {
    let carry =
        production_pi_rlc_family_body_carry_retained_audit().expect("exact normalized PiRLC carry retained-row audit");
    assert_eq!(carry.schema_version(), 1);
    assert_eq!(carry.source_row_start(), 163_609);
    assert_eq!(carry.source_rows(), 1_837);
    assert_eq!(carry.local_columns(), 165_664);
    assert_eq!(carry.source_column_shift(), 640);
    assert_eq!(carry.final_rows(), 491_046);
    assert_eq!(carry.final_columns(), 8_858_862);
    assert_eq!(carry.selector_columns(), [648, 649]);
    assert_eq!(carry.emitted_starts(), [69_607, 305_118]);
    assert_eq!(carry.source_starts(), [641, 164_466, 165_384, 166_302, 166_303]);
    assert_eq!(carry.final_starts(), [702, 2_142_411, 2_180_049, 2_217_687, 2_217_728]);
    assert_eq!(carry.widths(), [41; 5]);
    assert_eq!(carry.radices(), [3; 5]);
    assert_eq!(carry.source_nnz(), [4_593, 1_837, 0]);
    assert_eq!(
        carry.final_port_nnz(),
        [0, 3_674, 303_106, 3_674, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_body_carry_retained_artifact_matches_committed(&carry);
}

#[test]
#[ignore = "expensive exhaustive normalized PiRLC residual retained-row audit"]
fn pi_rlc_family_body_residual_retained_rows_match_exact_recipe() {
    let residual = production_pi_rlc_family_body_residual_retained_audit()
        .expect("exact normalized PiRLC residual retained-row audit");
    assert_eq!(residual.schema_version(), 1);
    assert_eq!(residual.source_row_start(), 163_501);
    assert_eq!(residual.source_rows(), 108);
    assert_eq!(residual.local_columns(), 165_664);
    assert_eq!(residual.source_column_shift(), 640);
    assert_eq!(residual.final_rows(), 491_046);
    assert_eq!(residual.final_columns(), 8_858_862);
    assert_eq!(residual.selector_columns(), [648, 649]);
    assert_eq!(residual.emitted_starts(), [69_499, 305_010]);
    assert_eq!(residual.source_starts(), [164_142, 164_250, 164_358]);
    assert_eq!(residual.final_starts(), [2_129_127, 2_133_555, 2_137_983]);
    assert_eq!(residual.widths(), [41; 3]);
    assert_eq!(residual.radices(), [3; 3]);
    assert_eq!(residual.source_nnz(), [324, 108, 0]);
    assert_eq!(
        residual.final_port_nnz(),
        [0, 216, 26_568, 216, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_body_residual_retained_artifact_matches_committed(&residual);
}

#[test]
#[ignore = "expensive exhaustive normalized PiRLC family-overlay retained-row audit"]
fn pi_rlc_family_overlay_retained_rows_match_exact_recipe() {
    let overlay = production_pi_rlc_family_overlay_retained_audit()
        .expect("exact normalized PiRLC family-overlay retained-row audit");
    assert_eq!(overlay.schema_version(), 1);
    assert_eq!(overlay.family_count(), 110);
    assert_eq!(overlay.source_rows(), 108);
    assert_eq!(overlay.source_columns(), 37_788);
    assert_eq!(overlay.final_rows(), 12_001);
    assert_eq!(overlay.final_columns(), 42_228);
    assert_eq!(overlay.selector_start(), 1);
    assert_eq!(overlay.selector_count(), 110);
    assert_eq!(overlay.retained_start(), 111);
    assert_eq!(overlay.retained_stride(), 108);
    assert_eq!(overlay.source_starts(), [1, 42, 37_680]);
    assert_eq!(overlay.final_starts(), [111, 152, 37_790]);
    assert_eq!(overlay.widths(), [1, 41]);
    assert_eq!(overlay.radices(), [2, 3]);
    assert_eq!(overlay.chunk_size(), 32_768);
    assert_eq!(
        overlay
            .chunk_seeds_by_row()
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        [3, 3]
    );
    assert_eq!(overlay.source_explicit_nnz(), [0, 11_880, 11_880]);
    assert_eq!(overlay.final_block_counts(), [0, 0, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        overlay.final_explicit_port_nnz(),
        [0, 11_880, 0, 11_880, 487_080, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_overlay_retained_artifact_matches_committed(&overlay);
}

#[test]
fn pi_rlc_family_bodies_have_two_exact_parity_shapes() {
    let even = NebulaFPrimePiRlcFamilyBodySynthesis::production(NebulaFPrimePiRlcFamilyReplayArmKind::Even);
    let odd = NebulaFPrimePiRlcFamilyBodySynthesis::production(NebulaFPrimePiRlcFamilyReplayArmKind::Odd);

    assert_eq!(PI_RLC_GLOBAL_INPUT_FIELDS, 100_980);
    assert_eq!(PI_RLC_MESSAGE_COLUMNS, 76_670);
    assert_eq!(PI_RLC_FAMILY_BODY_SOURCE_ROWS, 165_446);
    assert_eq!(PI_RLC_FAMILY_BODY_EVEN_SOURCE_ROWS, 310_646);
    assert_eq!(PI_RLC_FAMILY_BODY_ODD_SOURCE_ROWS, 311_846);
    assert_eq!(even.fixture_family(), 0);
    assert_eq!(odd.fixture_family(), 1);
    assert_eq!(even.rows(), PI_RLC_FAMILY_BODY_EVEN_ROWS);
    assert_eq!(odd.rows(), PI_RLC_FAMILY_BODY_ODD_ROWS);
    assert_eq!(even.columns(), PI_RLC_FAMILY_BODY_EVEN_COLUMNS);
    assert_eq!(odd.columns(), PI_RLC_FAMILY_BODY_ODD_COLUMNS);
    assert_eq!(even.public_columns(), 641);
    assert_eq!(odd.public_columns(), 641);
    assert_eq!(even.before_state_field_columns().len(), 1_045);
    assert_eq!(even.after_state_field_columns().len(), 1_045);
    assert_eq!(odd.before_state_field_columns().len(), 1_045);
    assert_eq!(odd.after_state_field_columns().len(), 1_045);
    assert_eq!(even.before_x_out_preimage_columns().len(), 32);
    assert_eq!(even.after_x_out_preimage_columns().len(), 32);
    assert_eq!(odd.before_x_out_preimage_columns().len(), 32);
    assert_eq!(odd.after_x_out_preimage_columns().len(), 32);
    assert_eq!(even.shape_audit().replay_poseidon2_permutations, 242);
    assert_eq!(odd.shape_audit().replay_poseidon2_permutations, 244);
    assert_eq!(even.shape_audit().poseidon2_permutations, 1_880);
    assert_eq!(odd.shape_audit().poseidon2_permutations, 1_882);
    assert_eq!(even.phase_delayed_payload_columns().len(), 2_169);
    assert_eq!(odd.phase_delayed_payload_columns().len(), 2_169);
    assert_eq!(STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS, 2_169);
    assert!(even.is_satisfied());
    assert!(odd.is_satisfied());
    assert!(even.unconstrained_columns().is_empty());
    assert!(odd.unconstrained_columns().is_empty());
}

fn decode_public_word(synthesis: &NebulaFPrimePiRlcFamilyBodySynthesis, word: usize) -> u64 {
    (0..64).fold(0u64, |value, bit| {
        let column = synthesis
            .public_output_column(word * 64 + bit)
            .expect("PiRLC public bit column");
        let bit_value = synthesis
            .witness_value(column)
            .expect("PiRLC public bit value")
            .as_canonical_u64();
        assert!(bit_value <= 1);
        value | (bit_value << bit)
    })
}

fn native_state_digest(synthesis: &NebulaFPrimePiRlcFamilyBodySynthesis, columns: &[usize]) -> [F; 4] {
    let fields = columns
        .iter()
        .map(|&column| synthesis.witness_value(column).expect("family-state field"))
        .collect::<Vec<_>>();
    let mut transcript = Poseidon2Transcript::new(STATE_DIGEST_DOMAIN);
    transcript.append_fields(b"state", &fields);
    let bytes = transcript.digest32();
    std::array::from_fn(|lane| {
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[lane * 8..(lane + 1) * 8]);
        F::from_u64(u64::from_le_bytes(word))
    })
}

fn witness_words(synthesis: &NebulaFPrimePiRlcFamilyBodySynthesis, columns: [usize; 4]) -> [u64; 4] {
    columns.map(|column| {
        synthesis
            .witness_value(column)
            .expect("PiRLC digest field")
            .as_canonical_u64()
    })
}

#[test]
fn pi_rlc_family_public_words_bind_full_state_then_global_cursor() {
    let even = NebulaFPrimePiRlcFamilyBodySynthesis::production(NebulaFPrimePiRlcFamilyReplayArmKind::Even);
    let odd = NebulaFPrimePiRlcFamilyBodySynthesis::production(NebulaFPrimePiRlcFamilyReplayArmKind::Odd);

    assert_eq!(decode_public_word(&even, 8), 223);
    assert_eq!(decode_public_word(&even, 9), 224);
    assert_eq!(decode_public_word(&odd, 8), 224);
    assert_eq!(decode_public_word(&odd, 9), 225);
    assert_eq!(
        (0..4)
            .map(|word| decode_public_word(&even, word))
            .collect::<Vec<_>>(),
        witness_words(&even, even.after_x_out_digest_columns())
    );
    assert_eq!(
        (4..8)
            .map(|word| decode_public_word(&even, word))
            .collect::<Vec<_>>(),
        witness_words(&even, even.before_x_out_digest_columns())
    );
    let payload = even
        .phase_delayed_payload_columns()
        .into_iter()
        .map(|column| even.witness_value(column).expect("delayed payload bit"))
        .collect::<Vec<_>>();
    let before_local = native_state_digest(&even, even.before_state_field_columns());
    let after_local = native_state_digest(&even, even.after_state_field_columns());
    assert_eq!(
        witness_words(&even, even.before_phase_local_state_columns()),
        before_local.map(|lane| lane.as_canonical_u64())
    );
    assert_eq!(
        witness_words(&even, even.after_phase_local_state_columns()),
        after_local.map(|lane| lane.as_canonical_u64())
    );
    assert_eq!(
        streaming_phase_semantic_digest(after_local, &payload).map(|lane| lane.as_canonical_u64()),
        std::array::from_fn(|lane| {
            let column = even.after_x_out_preimage_columns()[19 + lane];
            even.witness_value(column)
                .expect("after semantic digest")
                .as_canonical_u64()
        })
    );
    assert_eq!(
        streaming_phase_semantic_digest(before_local, &payload).map(|lane| lane.as_canonical_u64()),
        std::array::from_fn(|lane| {
            let column = even.before_x_out_preimage_columns()[19 + lane];
            even.witness_value(column)
                .expect("before semantic digest")
                .as_canonical_u64()
        })
    );
    assert_ne!(
        (0..4)
            .map(|word| decode_public_word(&even, word))
            .collect::<Vec<_>>(),
        after_local.map(|lane| lane.as_canonical_u64())
    );
}

#[test]
fn pi_rlc_family_phase_envelope_has_exact_shared_carry_sources() {
    let body = NebulaFPrimePiRlcFamilyBodySynthesis::production(NebulaFPrimePiRlcFamilyReplayArmKind::Even);
    let range = |name| {
        let family = body
            .builder_for_artifact()
            .column_family_ranges()
            .iter()
            .find(|family| family.name == name)
            .expect("exact PiRLC phase-envelope source family");
        family.column_start..family.column_end
    };

    assert_eq!(
        range(STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY),
        body.before_phase_local_state_columns()[0]..body.before_phase_local_state_columns()[0] + 4
    );
    assert_eq!(
        range(STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY),
        body.after_phase_local_state_columns()[0]..body.after_phase_local_state_columns()[0] + 4
    );
    assert_eq!(
        range(STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY),
        range(STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY),
        "PiRLC must carry one exact payload slice"
    );
    assert_eq!(
        range(STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY).len(),
        STREAMING_DELAYED_NEBULA_PAYLOAD_FIELDS
    );
}

#[test]
fn pi_rlc_family_rejects_tampered_delayed_payload() {
    let mut body = NebulaFPrimePiRlcFamilyBodySynthesis::production(NebulaFPrimePiRlcFamilyReplayArmKind::Even);
    let column = body.phase_delayed_payload_columns()[0];
    body.tamper_witness_for_test(column, F::ONE);
    assert!(!body.is_satisfied());
}

#[test]
fn pi_rlc_family_rejects_tampered_public_state_digest() {
    let mut body = NebulaFPrimePiRlcFamilyBodySynthesis::production(NebulaFPrimePiRlcFamilyReplayArmKind::Even);
    let column = body
        .public_output_column(0)
        .expect("after-state digest bit");
    let changed = F::ONE - body.witness_value(column).expect("digest bit value");
    body.tamper_witness_for_test(column, changed);
    assert!(!body.is_satisfied());
}

#[test]
fn pi_rlc_family_rejects_tampered_outer_state() {
    let mut body = NebulaFPrimePiRlcFamilyBodySynthesis::production(NebulaFPrimePiRlcFamilyReplayArmKind::Even);
    let column = body.after_x_out_preimage_columns()[15];
    let changed = body
        .witness_value(column)
        .expect("after current-boundary lane")
        + F::ONE;
    body.tamper_witness_for_test(column, changed);
    assert!(!body.is_satisfied());
}

#[test]
fn pi_rlc_family_overlays_are_only_the_family_dependent_seeded_rows() {
    for family in [0, 1, PI_RLC_FAMILY_COUNT - 1] {
        let overlay = NebulaFPrimePiRlcFamilyOverlaySynthesis::production(family).expect("bounded family");
        let audit = overlay.shape_audit();
        assert_eq!(audit.family, family);
        assert_eq!(audit.rows, PI_RLC_FAMILY_OVERLAY_ROWS);
        assert_eq!(audit.columns, PI_RLC_FAMILY_OVERLAY_COLUMNS);
        assert_eq!(audit.zero_digits, 1..42);
        assert_eq!(audit.active_digits, 42..33_252);
        assert_eq!(audit.outputs, 33_252..33_360);
        assert!(overlay.is_satisfied());
        assert!(overlay.unconstrained_columns().is_empty());
    }
}

#[test]
fn exact_links_join_each_family_overlay_to_its_parity_body() {
    let runs = production_pi_rlc_family_overlay_link_runs();
    assert_eq!(runs.map(|run| run.link_count()), [41, 33_210, 108]);
    assert_eq!(runs[0].phase_field_start(), 46_055);
    assert_eq!(runs[0].overlay_field_start(), 1);
    assert_eq!(runs[1].phase_field_start(), 46_096);
    assert_eq!(runs[1].overlay_field_start(), 42);
    assert_eq!(runs[1].outer_count(), 810);
    assert_eq!(runs[1].phase_stride(), 122);
    assert_eq!(runs[1].overlay_stride(), 41);
    assert_eq!(runs[2].phase_field_start(), 144_918);
    assert_eq!(runs[2].overlay_field_start(), 33_252);

    let links = production_pi_rlc_family_overlay_links(CLAIM_OVERLAY_KINDS);
    assert_eq!(links.len(), PI_RLC_FAMILY_COUNT);
    assert_eq!(
        links
            .iter()
            .map(|contract| contract.fields.len())
            .sum::<usize>(),
        3_669_490
    );
    for (family, contract) in links.iter().enumerate() {
        assert_eq!(contract.overlay_kind, CLAIM_OVERLAY_KINDS + family);
        assert_eq!(contract.phase_kind, 10 + family % 2);
        assert_eq!(contract.fields.len(), PI_RLC_FAMILY_LINK_FIELDS);
        assert_eq!(contract.fields[0].phase_field, 46_055);
        assert_eq!(contract.fields[0].overlay_field, 1);
        assert_eq!(contract.fields[40].phase_field, 46_095);
        assert_eq!(contract.fields[40].overlay_field, 41);
        assert_eq!(contract.fields[41].phase_field, 46_096);
        assert_eq!(contract.fields[41].overlay_field, 42);
        assert_eq!(contract.fields[33_250].phase_field, 144_834);
        assert_eq!(contract.fields[33_250].overlay_field, 33_251);
        assert_eq!(contract.fields[33_251].phase_field, 144_918);
        assert_eq!(contract.fields[33_251].overlay_field, 33_252);
        assert_eq!(contract.fields[33_358].phase_field, 145_025);
        assert_eq!(contract.fields[33_358].overlay_field, 33_359);
    }
}

#[test]
fn honest_body_and_family_overlay_agree_on_every_linked_field() {
    for (kind, family) in [
        (NebulaFPrimePiRlcFamilyReplayArmKind::Even, 0),
        (NebulaFPrimePiRlcFamilyReplayArmKind::Odd, 1),
    ] {
        let body = NebulaFPrimePiRlcFamilyBodySynthesis::production(kind);
        let body_assignment = body
            .normalized_field_assignment_for_artifact()
            .expect("normalized body assignment");
        let overlay = NebulaFPrimePiRlcFamilyOverlaySynthesis::production(family).expect("bounded family");
        let contract = production_pi_rlc_family_overlay_links(0).remove(family);
        for link in contract.fields {
            assert_eq!(
                body_assignment.get(link.phase_field).copied(),
                overlay.witness_value(link.overlay_field),
                "family {family}, phase field {}, overlay field {}",
                link.phase_field,
                link.overlay_field,
            );
        }
    }
}

#[test]
#[ignore = "expensive exhaustive normalized PiRLC body-overlay link audit"]
fn pi_rlc_family_normalized_links_match_exact_slots() {
    let audit =
        production_pi_rlc_family_normalized_link_audit().expect("exact normalized PiRLC body-overlay link audit");
    assert_eq!(audit.schema_version(), 1);
    assert_eq!(audit.family_count(), 110);
    assert_eq!(audit.parity_count(), 2);
    assert_eq!(audit.public_output_count(), 640);
    assert_eq!(audit.body_final_columns(), 8_858_862);
    assert_eq!(audit.overlay_final_columns(), 42_228);
    assert_eq!(audit.link_count_per_family(), 37_787);
    assert_eq!(audit.total_link_count(), 4_156_570);
    assert_eq!(audit.phase_kinds(), [10, 11]);
    let runs = audit.runs();
    assert_eq!(runs.map(|run| run.body_source_start()), [52_103, 52_144, 164_142]);
    assert_eq!(runs.map(|run| run.overlay_source_start()), [1, 42, 37_680]);
    assert_eq!(runs.map(|run| run.body_final_start()), [2_110_644, 38_340, 2_129_127]);
    assert_eq!(runs.map(|run| run.overlay_final_start()), [111, 152, 37_790]);
    assert_eq!(runs.map(|run| run.link_count()), [41, 37_638, 108]);
    assert_eq!(runs.map(|run| run.width()), [1, 1, 41]);
    assert_eq!(runs.map(|run| run.radix()), [2, 2, 3]);
    assert_normalized_link_artifact_matches_committed(&audit);
}

#[test]
#[ignore = "exhaustively scans the production normalized opening rows"]
fn pi_rlc_family_opening_rows_match_exact_images() {
    let audit = production_pi_rlc_family_body_opening_rows_audit().expect("exact normalized PiRLC opening-row audit");
    assert_eq!(audit.schema_version(), 1);
    assert_eq!(audit.arm_count(), 2);
    assert_eq!(audit.opening_count(), 918);
    assert_eq!(audit.digit_count(), 41);
    assert_eq!(audit.borrow_count(), 20);
    assert_eq!(audit.chunk_count(), 21);
    assert_eq!(audit.centered_row_count(), 0);
    assert_eq!(audit.zero_emitted_starts(), [69_456, 304_967]);
    assert_eq!(audit.canonical_emitted_starts(), [236_063, 471_746]);
    assert_eq!(audit.final_digit_start(), 38_340);
    assert_eq!(audit.final_zero_start(), 2_110_644);
    assert_eq!(audit.final_borrow_start(), 2_110_685);
    assert_body_opening_rows_artifact_matches_committed(&audit);
}

#[test]
fn family_position_changes_the_derived_commitment() {
    let zero = NebulaFPrimePiRlcFamilyOverlaySynthesis::production(0).expect("family zero");
    let one = NebulaFPrimePiRlcFamilyOverlaySynthesis::production(1).expect("family one");
    assert!(zero
        .output_columns()
        .iter()
        .zip(one.output_columns())
        .any(|(&left, &right)| zero.witness_value(left) != one.witness_value(right)));
}

#[test]
fn tampered_family_overlay_fails_its_seeded_rows() {
    let mut overlay = NebulaFPrimePiRlcFamilyOverlaySynthesis::production(109).expect("last family");
    let output = overlay.output_columns()[0];
    let value = overlay.witness_value(output).expect("output witness");
    overlay.tamper_witness_for_test(output, value + F::ONE);
    assert!(!overlay.is_satisfied());
}

#[test]
fn production_map_selects_all_110_family_overlays_once() {
    let map = production_pi_rlc_family_overlay_kind_map(0, CLAIM_OVERLAY_KINDS);
    let selected = map
        .into_iter()
        .filter(|&kind| kind >= CLAIM_OVERLAY_KINDS)
        .collect::<Vec<_>>();
    assert_eq!(
        selected,
        (CLAIM_OVERLAY_KINDS..CLAIM_OVERLAY_KINDS + PI_RLC_FAMILY_COUNT).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "expensive production low-norm body shape snapshot"]
fn pi_rlc_family_body_low_norm_shape_snapshot() {
    let relation = build_production_pi_rlc_family_body_low_norm_r1cs().expect("build two PiRLC parity bodies");
    eprintln!(
        "PiRLC body low-norm rows={}, columns={}, public={}, selectors={}",
        relation.structure().n,
        relation.structure().m,
        relation.public_input_len(),
        relation.selector_cols().len(),
    );
    assert_eq!(relation.selector_cols().len(), 2);
    assert_eq!(relation.public_input_len(), 648);
    assert_eq!(relation.structure().n, 491_046);
    assert_eq!(relation.structure().m, 8_858_862);
    assert!(relation.structure().n < 1 << 24);
    assert!(relation.structure().m < 1 << 24);

    let compiler_audit = production_pi_rlc_family_body_compiler_audit().expect("production compiler ledger");
    let row_audit = compiler_audit.rows();
    assert_eq!(row_audit.total_rows(), relation.structure().n);
    assert_eq!(row_audit.arms().len(), 2);
    let mut emitted = BTreeMap::<&str, (usize, usize)>::new();
    for run in row_audit.emitted_runs() {
        let name = match run.family() {
            SelectiveEmittedRowFamily::SelectorDomain => "selector_domain",
            SelectiveEmittedRowFamily::SharedDomain => "shared_domain",
            SelectiveEmittedRowFamily::ArmDomain => "arm_domain",
            SelectiveEmittedRowFamily::OneHot => "one_hot",
            SelectiveEmittedRowFamily::PublicPadding => "public_padding",
            SelectiveEmittedRowFamily::PrivatePadding => "private_padding",
            SelectiveEmittedRowFamily::Retained => "retained",
            SelectiveEmittedRowFamily::Poseidon2 => "poseidon2",
            SelectiveEmittedRowFamily::CenteredUnit => "centered_unit",
            SelectiveEmittedRowFamily::ShiftedTernaryCanonical => "shifted_ternary_canonical",
            SelectiveEmittedRowFamily::PolynomialEvaluation => "polynomial_evaluation",
            SelectiveEmittedRowFamily::ProductSum => "product_sum",
            SelectiveEmittedRowFamily::RingPadding => "ring_padding",
        };
        let entry = emitted.entry(name).or_default();
        entry.0 += 1;
        entry.1 += run.emitted_rows().len();
    }
    for (arm_index, arm) in row_audit.arms().iter().enumerate() {
        let mut source = BTreeMap::<&str, (usize, usize)>::new();
        for run in arm.source_runs() {
            let name = match run.disposition() {
                SelectiveSourceRowDisposition::Retained => "retained",
                SelectiveSourceRowDisposition::Poseidon2(_) => "poseidon2",
                SelectiveSourceRowDisposition::CenteredUnit(_) => "centered_unit",
                SelectiveSourceRowDisposition::ShiftedTernaryCanonical(_) => "shifted_ternary_canonical",
                SelectiveSourceRowDisposition::PolynomialEvaluation(_) => "polynomial_evaluation",
                SelectiveSourceRowDisposition::ProductSum(_) => "product_sum",
                SelectiveSourceRowDisposition::LinearDefinition(_) => "linear_definition",
            };
            let entry = source.entry(name).or_default();
            entry.0 += 1;
            entry.1 += run.source_rows().len();
        }
        eprintln!(
            "PiRLC body row ledger arm {arm_index}: source_runs={}, retained_emitted={:?}, emitted={:?}, source={source:?}",
            arm.source_runs().len(),
            arm.retained_emitted_rows(),
            arm.emitted_rows(),
        );
    }
    eprintln!(
        "PiRLC body emitted ledger: runs={}, rewrites={}, prefix={:?}, ring_padding={:?}, families={emitted:?}",
        row_audit.emitted_runs().len(),
        row_audit.rewrites().len(),
        row_audit.prefix_rows(),
        row_audit.ring_padding_rows(),
    );

    let compact = production_pi_rlc_family_body_row_ledger().expect("compact production row ledger");
    assert_eq!(compact.rows(), 491_046);
    assert_eq!(compact.columns(), 8_858_862);
    assert_eq!(compact.source_rows(), [1_300_897, 1_302_097]);
    assert_eq!(compact.rewrite_count(), 14_638);
    assert_eq!(compact.fixed_runs().len(), 8);
    assert_eq!(compact.retained_runs().len(), 22);
    assert_eq!(compact.linear_definition_counts(), [4_520, 4_520]);
    assert_eq!(compact.rewrite_batches().len(), 40);
    let mut compact_rewrites = BTreeMap::<&str, (usize, usize)>::new();
    for batch in compact.rewrite_batches() {
        let name = match batch.kind() {
            NebulaFPrimePiRlcBodyRewriteKind::Poseidon2 => "poseidon2",
            NebulaFPrimePiRlcBodyRewriteKind::ShiftedTernaryCanonical => "shifted_ternary_canonical",
            NebulaFPrimePiRlcBodyRewriteKind::LinearDefinition => {
                panic!("linear definitions must use complement ownership")
            }
        };
        let entry = compact_rewrites.entry(name).or_default();
        entry.0 += 1;
        entry.1 += batch.count();
    }
    assert_eq!(compact_rewrites.get("poseidon2"), Some(&(38, 3_762)));
    assert_eq!(compact_rewrites.get("shifted_ternary_canonical"), Some(&(2, 1_836)));
    eprintln!(
        "PiRLC compact row ledger: fixed_runs={}, retained_runs={}, rewrite_batches={}, rewrites={compact_rewrites:?}",
        compact.fixed_runs().len(),
        compact.retained_runs().len(),
        compact.rewrite_batches().len(),
    );
    assert_body_row_ledger_artifact_matches_committed(&compact);

    let algebra = production_pi_rlc_family_body_algebra_retained_audit()
        .expect("exact normalized PiRLC algebra retained-row audit");
    assert_eq!(algebra.schema_version(), 1);
    assert_eq!(algebra.source_rows(), 49_626);
    assert_eq!(algebra.local_columns(), 51_463);
    assert_eq!(algebra.source_column_shift(), 640);
    assert_eq!(algebra.final_rows(), 491_046);
    assert_eq!(algebra.final_columns(), 8_858_862);
    assert_eq!(algebra.selector_columns(), [648, 649]);
    assert_eq!(algebra.emitted_starts(), [19_830, 255_341]);
    assert_eq!(algebra.source_starts(), [641, 1_559, 2_477, 2_531]);
    assert_eq!(algebra.final_starts(), [702, 38_340, 75_978, 78_192]);
    assert_eq!(algebra.widths(), [41; 4]);
    assert_eq!(algebra.radices(), [3; 4]);
    assert_eq!(algebra.source_nnz(), [99_198, 117_504, 49_626]);
    assert_eq!(
        algebra.final_port_nnz(),
        [0, 99_252, 4_164_156, 9_635_328, 4_069_332, 0, 0, 0, 0, 0, 0, 0, 0,],
    );
    eprintln!(
        "PiRLC normalized algebra retained audit: source_nnz={:?}, final_port_nnz={:?}",
        algebra.source_nnz(),
        algebra.final_port_nnz(),
    );
    assert_body_algebra_retained_artifact_matches_committed(&algebra);
}

#[test]
#[ignore = "expensive complete production source-to-final decoder audit"]
fn pi_rlc_family_body_decoder_runs_cover_both_source_assignments() {
    let decoders = production_pi_rlc_family_body_decoder_runs().expect("build both PiRLC body decoders");
    assert_eq!(decoders.len(), 2);
    let expected_ranges = [1..PI_RLC_FAMILY_BODY_EVEN_COLUMNS, 1..PI_RLC_FAMILY_BODY_ODD_COLUMNS];
    for (arm, (decoder, expected_range)) in decoders.iter().zip(expected_ranges).enumerate() {
        assert_eq!(decoder.arm(), arm);
        assert_eq!(decoder.source_range(), expected_range);
        assert_eq!(decoder.final_columns(), 8_858_862);
        let mut cursor = expected_range.start;
        let mut exact = vec![None; expected_range.len()];
        for run in decoder.runs() {
            assert_eq!(run.source_start(), cursor);
            assert!(run.length() > 0);
            for column in run.source_start()..run.source_end() {
                exact[column - expected_range.start] = run.resolution_at(column);
            }
            cursor = run.source_end();
        }
        assert_eq!(cursor, expected_range.end);
        assert!(exact.iter().all(Option::is_some));
        let mut strided_owners = vec![false; expected_range.len()];
        for run in decoder.strided_runs() {
            assert!(run.count() > 0);
            assert!(run.source_stride() > 0);
            for index in 0..run.count() {
                let column = run
                    .source_column(index)
                    .expect("bounded strided source column");
                let offset = column - expected_range.start;
                assert!(!strided_owners[offset]);
                strided_owners[offset] = true;
                assert_eq!(run.resolution_at(column), exact[offset]);
            }
        }
        assert!(strided_owners.into_iter().all(|owned| owned));
        let template_instances = decoder
            .repeated_templates()
            .iter()
            .flat_map(|template| template.instances())
            .map(|instances| instances.count())
            .sum::<usize>();
        assert_eq!(template_instances, [2_798, 2_800][arm]);
        assert!(decoder
            .repeated_templates()
            .iter()
            .any(|template| template.source_width() == 600));
        assert!(decoder
            .repeated_templates()
            .iter()
            .any(|template| template.source_width() == 122));
        assert!(!decoder.repeated_templates().is_empty());
        assert!(!decoder.residual_strided_runs().is_empty());
        let mut kinds = BTreeMap::<&str, (usize, usize)>::new();
        let mut lengths = BTreeMap::<usize, usize>::new();
        for run in decoder.runs() {
            let kind = match run.resolution() {
                neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedSourceResolutionRun::Direct {
                    ..
                } => "direct",
                neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedSourceResolutionRun::DecompositionAlias {
                    ..
                } => "decomposition_alias",
                neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedSourceResolutionRun::EqualityAlias {
                    ..
                } => "equality_alias",
                neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedSourceResolutionRun::LinearDefinition => {
                    "linear_definition"
                }
                neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedSourceResolutionRun::TraceEliminated => {
                    "trace_eliminated"
                }
            };
            let entry = kinds.entry(kind).or_default();
            entry.0 += 1;
            entry.1 += run.length();
            *lengths.entry(run.length()).or_default() += 1;
        }
        eprintln!(
            "PiRLC body arm {arm}: source={}..{}, decoder_runs={}, strided_runs={}, templates={}, template_relative_runs={}, template_instance_runs={}, template_instances={}, residual_strided_runs={}, source_families={}, kinds={kinds:?}, lengths={lengths:?}",
            expected_range.start,
            expected_range.end,
            decoder.runs().len(),
            decoder.strided_runs().len(),
            decoder.repeated_templates().len(),
            decoder
                .repeated_templates()
                .iter()
                .map(|template| template.relative_runs().len())
                .sum::<usize>(),
            decoder
                .repeated_templates()
                .iter()
                .map(|template| template.instances().len())
                .sum::<usize>(),
            template_instances,
            decoder.residual_strided_runs().len(),
            decoder.source_families().len(),
        );
    }
    assert_body_decoder_artifact_matches_committed(&decoders);
}

#[test]
#[ignore = "expensive production low-norm overlay shape snapshot"]
fn pi_rlc_family_overlay_low_norm_shape_snapshot() {
    let relation = build_production_pi_rlc_family_overlay_low_norm_r1cs().expect("build 110 PiRLC family overlays");
    eprintln!(
        "PiRLC overlay low-norm rows={}, columns={}, public={}, selectors={}",
        relation.structure().n,
        relation.structure().m,
        relation.public_input_len(),
        relation.selector_cols().len(),
    );
    assert_eq!(relation.selector_cols().len(), PI_RLC_FAMILY_COUNT);
    assert_eq!(relation.public_input_len(), 1);
    assert_eq!(relation.structure().n, 12_001);
    assert_eq!(relation.structure().m, 37_800);
    assert!(relation.structure().n < 1 << 24);
    assert!(relation.structure().m < 1 << 24);
}
