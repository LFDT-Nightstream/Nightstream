//! Differential checks and reviewed Lean artifacts for the read-only selective
//! encoder snapshot.
//!
//! Owns: deterministic fixture construction, final-matrix materialization,
//! coefficient assertions, compact whole-fixture selector-support coverage,
//! and drift detection for generated Lean data.
//!
//! Does not own: selector soundness, branch-to-paper refinement, production
//! F' row counts, production-family coverage, or permission to remove rows.
//!
//! | Stage path | Fixture evidence | Semantic consumer |
//! |---|---|---|
//! | `f_prime.selective_ccs.branch.selector_domain` | three exact Boolean rows | Lean coefficient classifier |
//! | `f_prime.selective_ccs.branch.total` | exact `1 - sum(selector)` row | Lean selector-total refinement |
//! | `f_prime.selective_ccs.branch.gate[0]` | one representative retained source row | Lean product-gate refinement |
//! | `f_prime.selective_ccs.branch.gate[*]` | every fixture row has exactly one final selector port and column | compact physical-coverage audit |
//! | `f_prime.selective_ccs.padding.public` | all thirteen zero-pin rows | Lean carrier refinement |

#[path = "../support/selective_selector_coverage_lean.rs"]
mod selector_coverage_lean;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use neo_ccs::{CcsMatrix, CscMat};
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::field_ext::{enforce_k_dot_product, KVar};
use neo_fold_clean::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder};
use neo_fold_clean::frontends::r1cs_f_prime::lowering::{LowNormR1csError, SelectiveSnapshotError};
use neo_fold_clean::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_rows_with_alignment,
    audit_multi_branch_selective_rows_with_complete_source_provenance_with_alignment, build_multi_branch_low_norm_r1cs,
    build_multi_branch_selective_low_norm_r1cs_with_alignment, lower_field_r1cs, SelectiveEmittedRowFamily,
    SelectiveGatePort, SelectiveProjectedPort, SelectiveProjectedRewriteOutput, SelectiveProjectedRewriteStep,
    SelectiveProjectedSourceProvenance, SelectiveProjectedSourceTerm, SelectiveRewriteKind, SelectiveRowArtifact,
    SelectiveSelectorGateCoverage,
};
use neo_fold_clean::paper::f_prime::r1cs::{F_PRIME_PUBLIC_INPUT_LEN, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN};
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{enforce_commit_fields, CCS_CLAIM_SIS_CONFIG};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use selector_coverage_lean::{lean_family, write_raw_coverage};

const SELECTOR_ROW_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistorySelectiveCcsSelectorDomainRow.lean";
const CARRIER_270_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistorySelectiveCarrier270.lean";
const SELECTOR_COVERAGE_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistorySelectiveSelectorCoverageFixture.lean";
const GROUPED_PRODUCT_REWRITE_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistorySelectiveGroupedProductRewriteFixture.lean";

fn render_selector_coverage_artifact(coverage: &SelectiveSelectorGateCoverage) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveSelectorCoverageSchema\n\n\
/-! Generated file: run-compressed selector coverage for one deterministic\n\
three-arm selective-compiler fixture.\n\n\
Owns: the complete exclusive owner ledger and the complete final-matrix\n\
general/evaluation selector support after Rust has reconciled them exactly.\n\n\
Does not own: a production F-prime relation, branch semantics, source-row\n\
refinement, constraint necessity, a trusted production count, or row removal.\n\n\
Emits constraints: no. Empty owner runs remain visible; selector support is\n\
split at owner boundaries and never expanded to one record per row.\n\n\
| Artifact branch | Exact Rust source | Semantic status |\n\
|---|---|---|\n\
| owner runs | exclusive emitted-row ledger | provenance, reconciled in Lean |\n\
| gate runs | final selector-port CSC matrices | physical support, reconciled in Lean |\n\
| coefficient | checked final CSC value | must decode as field one |\n\
| polynomial | final ordered sparse terms | compared to independent Lean syntax |\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveSelectorCoverageFixture\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.Wire",
    )
    .expect("render selector coverage header");
    write_raw_coverage(&mut rendered, "rawCoverage", coverage).expect("render selector coverage");
    writeln!(
        rendered,
        "\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveSelectorCoverageFixture"
    )
    .expect("render selector coverage footer");
    rendered
}

fn assert_selector_coverage_artifact_matches_committed(coverage: &SelectiveSelectorGateCoverage) {
    let rendered = render_selector_coverage_artifact(coverage);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), SELECTOR_COVERAGE_ARTIFACT_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write reviewed selector-coverage artifact");
        panic!("selector-coverage Lean fixture drifted; wrote {expected}. Inspect it and promote it explicitly");
    }
}

fn write_raw_row(rendered: &mut String, name: &str, artifact: &SelectiveRowArtifact) -> std::fmt::Result {
    let row = artifact.matrix_row();
    writeln!(
        rendered,
        "\ndef {name} : RawRow where\n  schemaVersion := {}\n  rows := {}\n  columns := {}\n  emittedRow := {}\n  runIndex := {}\n  family := .{}\n  arm := {}\n  ports := [",
        artifact.schema_version(),
        row.rows(),
        row.columns(),
        row.emitted_row(),
        artifact.run_index(),
        lean_family(artifact.family()),
        artifact.arm().map_or_else(|| "none".to_owned(), |arm| format!("some {arm}")),
    )?;
    for (port_index, port) in row.ports().iter().enumerate() {
        let separator = if port_index == 0 { "    " } else { "  , " };
        let terms = port
            .iter()
            .map(|term| {
                format!(
                    "{{ column := {}, coefficient := {} }}",
                    term.column(),
                    term.coefficient().as_canonical_u64()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(rendered, "{separator}{{ terms := [{terms}] }}")?;
    }
    writeln!(rendered, "]")
}

fn lean_source_terms(terms: &[SelectiveProjectedSourceTerm]) -> String {
    terms
        .iter()
        .map(|term| {
            format!(
                "{{ column := {}, coefficient := {} }}",
                term.column(),
                term.coefficient().as_canonical_u64()
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn lean_source_lc(constant: F, terms: &[SelectiveProjectedSourceTerm]) -> String {
    format!(
        "{{ constant := {}, terms := [{}] }}",
        constant.as_canonical_u64(),
        lean_source_terms(terms)
    )
}

fn csc_row(matrix: &CscMat<F>, row: usize) -> Vec<(usize, F)> {
    let mut terms = Vec::new();
    for column in 0..matrix.ncols {
        for entry in matrix.column_range(column) {
            if matrix.row_index(entry) == row {
                terms.push((column, matrix.vals[entry]));
            }
        }
    }
    terms
}

fn source_matrix_row(matrix: &CcsMatrix<F>, row: usize) -> Vec<(usize, F)> {
    match matrix {
        CcsMatrix::Csc(matrix) => csc_row(matrix, row),
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            assert!(
                blocks.is_empty() && geometric_runs.is_empty(),
                "grouped-product fixture source rows must be ordinary CSC"
            );
            csc_row(csc, row)
        }
        CcsMatrix::Identity { .. } | CcsMatrix::VerifierArtifact { .. } => {
            panic!("grouped-product fixture source rows must use explicit CSC matrices")
        }
    }
}

fn lean_source_matrix_lc(terms: &[(usize, F)]) -> String {
    let mut constant = F::ZERO;
    let mut nonconstant = Vec::new();
    for &(column, coefficient) in terms {
        if column == 0 {
            constant += coefficient;
        } else {
            nonconstant.push(format!(
                "{{ column := {column}, coefficient := {} }}",
                coefficient.as_canonical_u64()
            ));
        }
    }
    format!(
        "{{ constant := {}, terms := [{}] }}",
        constant.as_canonical_u64(),
        nonconstant.join(", ")
    )
}

fn write_raw_source_rows(
    rendered: &mut String,
    source_arm: &neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs,
    source: &SelectiveProjectedSourceProvenance,
) -> std::fmt::Result {
    let source_rows = source
        .rewrite_steps()
        .iter()
        .flat_map(|step| {
            step.source_rows()
                .iter()
                .flat_map(|&(start, stop)| start..stop)
        })
        .collect::<BTreeSet<_>>();
    writeln!(rendered, "\ndef rawSourceRows : List RawSourceR1csRow := [")?;
    for (index, row) in source_rows.into_iter().enumerate() {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(
            rendered,
            "{separator}{{ row := {row}, a := {}, b := {}, c := {} }}",
            lean_source_matrix_lc(&source_matrix_row(&source_arm.a, row)),
            lean_source_matrix_lc(&source_matrix_row(&source_arm.b, row)),
            lean_source_matrix_lc(&source_matrix_row(&source_arm.c, row)),
        )?;
    }
    writeln!(rendered, "]")
}

fn write_raw_source_layout(rendered: &mut String, source: &SelectiveProjectedSourceProvenance) -> std::fmt::Result {
    let mut source_columns = BTreeSet::new();
    let mut derived_indices = BTreeSet::new();
    for step in source.rewrite_steps() {
        match step.output() {
            SelectiveProjectedRewriteOutput::Source { terms, .. } => {
                source_columns.extend(terms.iter().map(|term| term.column()));
            }
            SelectiveProjectedRewriteOutput::DerivedProductSum { compiler_index } => {
                derived_indices.insert(*compiler_index);
            }
        }
        source_columns.extend(step.base_terms().iter().map(|term| term.column()));
        if let Some(previous) = step.previous() {
            derived_indices.insert(previous);
        }
        for factor in step.factors() {
            source_columns.extend(factor.left_terms().iter().map(|term| term.column()));
            source_columns.extend(factor.right_terms().iter().map(|term| term.column()));
        }
    }
    loop {
        let before = source_columns.len();
        for definition in source.linear_definitions() {
            if source_columns.contains(&definition.target()) {
                source_columns.extend(definition.terms().iter().map(|term| term.column()));
            }
        }
        if source_columns.len() == before {
            break;
        }
    }

    writeln!(rendered, "\ndef rawSourceSlots : List RawSourceSlot := [")?;
    for (index, slot) in source
        .retained_slots()
        .iter()
        .filter(|slot| source_columns.contains(&slot.column()))
        .enumerate()
    {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(
            rendered,
            "{separator}{{ column := {}, start := {}, width := {} }}",
            slot.column(),
            slot.start(),
            slot.width(),
        )?;
    }
    writeln!(rendered, "]")?;

    writeln!(rendered, "\ndef rawSourceDefinitions : List RawSourceDefinition := [")?;
    for (index, definition) in source
        .linear_definitions()
        .iter()
        .filter(|definition| source_columns.contains(&definition.target()))
        .enumerate()
    {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(
            rendered,
            "{separator}{{ target := {}, value := {} }}",
            definition.target(),
            lean_source_lc(definition.constant(), definition.terms()),
        )?;
    }
    writeln!(rendered, "]")?;

    writeln!(rendered, "\ndef rawDerivedSlots : List RawDerivedSlot := [")?;
    for (index, derived) in source
        .derived_product_sums()
        .iter()
        .filter(|derived| derived_indices.contains(&derived.compiler_index()))
        .enumerate()
    {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(
            rendered,
            "{separator}{{ compilerIndex := {}, start := {}, width := {} }}",
            derived.compiler_index(),
            derived.start(),
            derived.width(),
        )?;
    }
    writeln!(rendered, "]")
}

fn write_raw_rewrite_step(rendered: &mut String, name: &str, step: &SelectiveProjectedRewriteStep) -> std::fmt::Result {
    let kind = match step.kind() {
        SelectiveRewriteKind::PolynomialEvaluation => "polynomialEvaluation",
        SelectiveRewriteKind::ProductSum => "productSum",
        other => panic!("unsupported grouped-product rewrite kind {other:?}"),
    };
    let source_rows = step
        .source_rows()
        .iter()
        .map(|&(start, stop)| format!("{{ start := {start}, stop := {stop} }}"))
        .collect::<Vec<_>>()
        .join(", ");
    let output = match step.output() {
        SelectiveProjectedRewriteOutput::Source { constant, terms } => {
            format!(".source {}", lean_source_lc(*constant, terms))
        }
        SelectiveProjectedRewriteOutput::DerivedProductSum { compiler_index } => {
            format!(".derivedProductSum {compiler_index}")
        }
    };
    let previous = step
        .previous()
        .map_or_else(|| "none".to_owned(), |index| format!("some {index}"));

    writeln!(
        rendered,
        "\ndef {name} : RawStep where\n  emittedRow := {}\n  rewriteId := {}\n  kind := .{kind}\n  sourceRows := [{source_rows}]\n  output := {output}\n  base := {}\n  previous := {previous}\n  factors := [",
        step.emitted_row(),
        step.rewrite_id(),
        lean_source_lc(step.base_constant(), step.base_terms()),
    )?;
    for (index, factor) in step.factors().iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ left := {}, right := {}, coefficient := {} }}",
            lean_source_lc(factor.left_constant(), factor.left_terms()),
            lean_source_lc(factor.right_constant(), factor.right_terms()),
            factor.coefficient().as_canonical_u64(),
        )?;
    }
    writeln!(rendered, "  ]")
}

fn render_grouped_product_rewrite_artifact(
    source_rows: usize,
    source_columns: usize,
    source_arm: &neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs,
    source: &SelectiveProjectedSourceProvenance,
    rows: &[SelectiveRowArtifact],
) -> String {
    assert_eq!(source.rewrite_steps().len(), rows.len());
    let final_columns = rows
        .first()
        .expect("grouped-product fixture has final rows")
        .matrix_row()
        .columns();
    assert!(rows
        .iter()
        .all(|row| row.matrix_row().columns() == final_columns));
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteSchema\n\n\
/-! Generated file: exact grouped-product rewrite data for one deterministic\n\
two-arm selective-compiler fixture.\n\n\
Owns: the complete Rust-projected source recurrence and every final materialized\n\
row for one nonempty product-sum rewrite.\n\n\
Does not own: production-family coverage, source-to-final assignment\n\
refinement, recursive or terminal conformance, constraint necessity, or\n\
permission to remove a production row or coordinate.\n\n\
Emits constraints: no. Rust emits this file only after its exact provenance\n\
audit reproduces every final row from the executable rewrite steps.\n\n\
| Artifact branch | Exact Rust source | Semantic status |\n\
|---|---|---|\n\
| source expressions and slots | checked executable rewrite plan | untrusted wire data |\n\
| row coefficients | final materialized selective matrices | untrusted wire data |\n\
| row/rewrite join | emitted-row ownership ledger | checked again in Lean |\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveGroupedProductRewriteFixture\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire\n\
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Wire\n\n\
def sourceRowCount : Nat := {source_rows}\n\
def sourceColumnCount : Nat := {source_columns}\n\
def finalColumnCount : Nat := {final_columns}\n\
def arm : Nat := {}\n",
        source.arm(),
    )
    .expect("render grouped-product artifact header");

    write_raw_source_layout(&mut rendered, source).expect("render grouped-product source layout");
    write_raw_source_rows(&mut rendered, source_arm, source).expect("render grouped-product source rows");

    for (index, step) in source.rewrite_steps().iter().enumerate() {
        write_raw_rewrite_step(&mut rendered, &format!("rawStep{index:02}"), step)
            .expect("render grouped-product rewrite step");
    }
    writeln!(rendered, "\ndef rawSteps : List RawStep := [").expect("render grouped-product step list");
    for index in 0..source.rewrite_steps().len() {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(rendered, "{separator}rawStep{index:02}").expect("render grouped-product step item");
    }
    writeln!(rendered, "]").expect("render grouped-product step list footer");

    for (index, row) in rows.iter().enumerate() {
        write_raw_row(&mut rendered, &format!("rawRow{index:02}"), row).expect("render grouped-product final row");
    }
    writeln!(rendered, "\ndef rawRows : List RawRow := [").expect("render grouped-product row list");
    for index in 0..rows.len() {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(rendered, "{separator}rawRow{index:02}").expect("render grouped-product row item");
    }
    writeln!(
        rendered,
        "]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveGroupedProductRewriteFixture"
    )
    .expect("render grouped-product artifact footer");
    rendered
}

fn assert_grouped_product_rewrite_artifact_matches_committed(
    source_rows: usize,
    source_columns: usize,
    source_arm: &neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs,
    source: &SelectiveProjectedSourceProvenance,
    rows: &[SelectiveRowArtifact],
) {
    let rendered = render_grouped_product_rewrite_artifact(source_rows, source_columns, source_arm, source, rows);
    let path = format!(
        "{}{}",
        env!("CARGO_MANIFEST_DIR"),
        GROUPED_PRODUCT_REWRITE_ARTIFACT_PATH
    );
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write reviewed grouped-product artifact");
        panic!("grouped-product Lean fixture drifted; wrote {expected}. Inspect it and promote it explicitly");
    }
}

fn render_selector_row_artifact(artifact: &SelectiveRowArtifact) -> String {
    let row = artifact.matrix_row();
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsRowSchema\n\n\
/-! Generated file: one deterministic selective-compiler row fixture.\n\n\
Owns: exact final-matrix coefficients and diagnostic row-ledger provenance for\n\
the first selector-domain row of the two-arm snapshot test fixture.\n\n\
Does not own: a production F-prime profile, row-family truth, semantic\n\
soundness, constraint necessity, or permission to remove rows.\n\n\
Emits constraints: no. Rust materializes the final compact matrices before\n\
rendering; Lean independently decodes and classifies their coefficients.\n\n\
| Artifact branch | Exact source | Lean consumer |\n\
|---|---|---|\n\
| dimensions and row | final selective structure | fail-closed row decoder |\n\
| thirteen sparse ports | final materialized matrices | coefficient semantics |\n\
| run/family/arm | exclusive emitted-row ledger | diagnostic only |\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCcsSelectorDomainRow\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire\n\n\
def rawRow : RawRow where\n  schemaVersion := {}\n  rows := {}\n  columns := {}\n  emittedRow := {}\n  runIndex := {}\n  family := .{}\n  arm := {}\n  ports := [",
        artifact.schema_version(),
        row.rows(),
        row.columns(),
        row.emitted_row(),
        artifact.run_index(),
        lean_family(artifact.family()),
        artifact.arm().map_or_else(|| "none".to_owned(), |arm| format!("some {arm}")),
    )
    .expect("render selective row header");
    for (port_index, port) in row.ports().iter().enumerate() {
        let separator = if port_index == 0 { "    " } else { "  , " };
        let terms = port
            .iter()
            .map(|term| {
                format!(
                    "{{ column := {}, coefficient := {} }}",
                    term.column(),
                    term.coefficient().as_canonical_u64()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(rendered, "{separator}{{ terms := [{terms}] }}").expect("render selective row port");
    }
    writeln!(
        rendered,
        "]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCcsSelectorDomainRow"
    )
    .expect("render selective row footer");
    rendered
}

fn assert_selector_row_artifact_matches_committed(artifact: &SelectiveRowArtifact) {
    let rendered = render_selector_row_artifact(artifact);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), SELECTOR_ROW_ARTIFACT_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write reviewed selective-row artifact");
        panic!("selective-row Lean fixture drifted; wrote {expected}. Inspect it and promote it explicitly");
    }
}

fn render_carrier_270_artifact(
    layout: &neo_fold_clean::frontends::r1cs_f_prime::SelectiveLayoutAudit,
    selector_rows: &[SelectiveRowArtifact],
    one_hot_row: &SelectiveRowArtifact,
    gated_row: &SelectiveRowArtifact,
    padding_rows: &[SelectiveRowArtifact],
) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsRowSchema\n\n\
/-! Generated file: exact selective-compiler public-carrier fixture.\n\n\
Owns: the compiler-produced 257/270 public widths, public-padding, selector,\n\
private-alignment and branch ranges; the exact selector-domain and sum rows;\n\
one representative gated source row; and every public-padding row of the\n\
three-arm F-prime-width fixture.\n\n\
Does not own: semantic truth of those rows, a full fixed-point F-prime relation,\n\
private branch rows, NIFS soundness, constraint necessity, or row removal.\n\n\
Emits constraints: no. This file is inert Rust-exported data.\n\n\
| Artifact family | Exact source | Multiplicity |\n\
|---|---|---:|\n\
| public layout | prepared layout consumed by the selective emitter | 1 |\n\
| selector domain | final matrices joined to the exclusive row ledger | 3 |\n\
| selector total | final matrices joined to the exclusive row ledger | 1 |\n\
| representative arm gate | first retained source row in arm zero | 1 |\n\
| public zero padding | final thirteen-port structure joined to the exclusive row ledger | 13 |\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCarrier270\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire\n\n\
def logicalPublicInputLen : Nat := {}\n\
def publicInputLen : Nat := {}\n\
def publicPaddingColumns : List Nat := {:?}\n\
def selectorColumns : List Nat := {:?}\n\
def privateAlignmentPaddingColumns : List Nat := {:?}\n\
def sharedPrivateStart : Nat := {}\n\
def sharedPrivateEnd : Nat := {}\n\
def branchStart : Nat := {}\n\
def branchEnd : Nat := {}\n\
def ringAlignmentPaddingStart : Nat := {}\n\
def ringAlignmentPaddingEnd : Nat := {}\n",
        layout.logical_public_input_len(),
        layout.public_input_len(),
        layout.public_padding_columns(),
        layout.selector_columns(),
        layout.private_alignment_padding_columns(),
        layout.shared_private_columns().start,
        layout.shared_private_columns().end,
        layout.branch_columns().start,
        layout.branch_columns().end,
        layout.ring_alignment_padding_columns().start,
        layout.ring_alignment_padding_columns().end,
    )
    .expect("render carrier layout");

    for (index, artifact) in selector_rows.iter().enumerate() {
        write_raw_row(&mut rendered, &format!("rawSelectorRow{index:02}"), artifact)
            .expect("render carrier selector row");
    }
    writeln!(rendered, "\ndef rawSelectorRows : List RawRow := [").expect("render selector-row list header");
    for index in 0..selector_rows.len() {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(rendered, "{separator}rawSelectorRow{index:02}").expect("render selector-row list item");
    }
    writeln!(rendered, "]").expect("render selector-row list footer");

    write_raw_row(&mut rendered, "rawOneHotRow", one_hot_row).expect("render carrier one-hot row");

    write_raw_row(&mut rendered, "rawGatedRow", gated_row).expect("render representative gated row");

    for (index, artifact) in padding_rows.iter().enumerate() {
        write_raw_row(&mut rendered, &format!("rawPaddingRow{index:02}"), artifact)
            .expect("render carrier padding row");
    }

    writeln!(rendered, "\ndef rawPaddingRows : List RawRow := [").expect("render carrier padding-row list header");
    for index in 0..padding_rows.len() {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(rendered, "{separator}rawPaddingRow{index:02}").expect("render carrier padding-row list item");
    }
    writeln!(
        rendered,
        "]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCarrier270"
    )
    .expect("render carrier artifact footer");
    rendered
}

fn assert_carrier_270_artifact_matches_committed(
    layout: &neo_fold_clean::frontends::r1cs_f_prime::SelectiveLayoutAudit,
    selector_rows: &[SelectiveRowArtifact],
    one_hot_row: &SelectiveRowArtifact,
    gated_row: &SelectiveRowArtifact,
    padding_rows: &[SelectiveRowArtifact],
) {
    let rendered = render_carrier_270_artifact(layout, selector_rows, one_hot_row, gated_row, padding_rows);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), CARRIER_270_ARTIFACT_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write reviewed carrier-270 artifact");
        panic!("selective carrier-270 Lean artifact drifted; wrote {expected}. Inspect it and promote it explicitly");
    }
}

fn snapshot_arm(seed: u64) -> (neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs, Vec<F>) {
    let mut builder = R1csBuilder::new();
    let public = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public);
    let public_copy = builder.alloc(F::ONE);
    enforce_bit(&mut builder, public_copy);
    builder.enforce_eq(&Lc::from_var(public), &Lc::from_var(public_copy));
    let mut lhs = (0..6)
        .map(|index| {
            KVar::alloc(
                &mut builder,
                F::from_u64(seed + 2 * index + 1),
                F::from_u64(seed + 2 * index + 2),
            )
        })
        .collect::<Vec<_>>();
    let affine_source = lhs[0].c0;
    let affine_value = builder.witness()[affine_source.col()] + F::from_u64(3);
    let affine = builder.alloc(affine_value);
    let mut affine_rhs = Lc::from_var(affine_source);
    affine_rhs.add_constant(F::from_u64(3));
    builder.enforce_eq(&Lc::from_var(affine), &affine_rhs);
    lhs[0].c0 = affine;
    let rhs = (0..6)
        .map(|index| {
            KVar::alloc(
                &mut builder,
                F::from_u64(seed + 3 * index + 3),
                F::from_u64(seed + 3 * index + 4),
            )
        })
        .collect::<Vec<_>>();
    let output = enforce_k_dot_product(&mut builder, &lhs, &rhs);

    let equal_copy = builder.alloc(builder.witness()[output.c1.col()]);
    builder.enforce_eq(&Lc::from_var(output.c1), &Lc::from_var(equal_copy));
    let canonical_copy = builder.alloc(builder.witness()[output.c0.col()]);
    builder.enforce_eq(&Lc::from_var(output.c0), &Lc::from_var(canonical_copy));
    let _output_bits = decompose_var_to_u64_bits(&mut builder, output.c0);
    let _copy_bits = decompose_var_to_u64_bits(&mut builder, canonical_copy);
    lower_field_r1cs(builder, &[public])
        .expect("lower snapshot fixture")
        .into_parts()
}

fn expand_projected_port(port: &SelectiveProjectedPort) -> Vec<(usize, F)> {
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

fn f_prime_public_carrier_fixture() -> Vec<(neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs, Vec<F>)> {
    (0..3)
        .map(|_| {
            let mut builder = R1csBuilder::new();
            let public_bits = (0..F_PRIME_PUBLIC_INPUT_LEN - 1)
                .map(|_| {
                    let bit = builder.alloc(F::ZERO);
                    enforce_bit(&mut builder, bit);
                    bit
                })
                .collect::<Vec<_>>();
            lower_field_r1cs(builder, &public_bits)
                .expect("lower F-prime-width carrier fixture")
                .into_parts()
        })
        .collect()
}

fn expected_gate_for_owner(
    family: SelectiveEmittedRowFamily,
    arm: Option<usize>,
    selector_columns: &[usize],
) -> (SelectiveGatePort, usize) {
    use SelectiveEmittedRowFamily as Family;
    match family {
        Family::SelectorDomain
        | Family::OneHot
        | Family::PublicPadding
        | Family::PrivatePadding
        | Family::RingPadding => {
            assert_eq!(arm, None, "common family {family:?} must not carry an arm");
            (SelectiveGatePort::General, 0)
        }
        Family::SharedDomain => {
            assert_eq!(arm, None, "shared domain must not carry an arm");
            (SelectiveGatePort::GeneralEvaluation, 0)
        }
        Family::ArmDomain => {
            let arm = arm.expect("combined-gated arm domain");
            (SelectiveGatePort::GeneralEvaluation, selector_columns[arm])
        }
        Family::Retained | Family::Poseidon2 | Family::CenteredUnit | Family::ShiftedTernaryCanonical => {
            let arm = arm.expect("general-gated arm family");
            (SelectiveGatePort::General, selector_columns[arm])
        }
        Family::PolynomialEvaluation | Family::ProductSum => {
            let arm = arm.expect("evaluation-gated arm family");
            (SelectiveGatePort::Evaluation, selector_columns[arm])
        }
    }
}

#[test]
fn projected_emitter_rows_expand_to_materialized_rows() {
    let fixtures = [snapshot_arm(7), snapshot_arm(19)];
    let shapes = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, 0)
        .expect("compile selective projection fixture");
    let snapshot = relation
        .selective_snapshot()
        .expect("checked selective projection snapshot");
    assert!(snapshot.structure().n < 10_000, "bounded differential fixture");
    let selected_rows = (0..snapshot.structure().n).collect::<Vec<_>>();
    let projected = audit_multi_branch_selective_rows_with_alignment(&shapes, 0, D, 0, &selected_rows)
        .expect("project exact emitter rows");

    assert_eq!(projected.rows(), snapshot.structure().n);
    assert_eq!(projected.columns(), snapshot.structure().m);
    assert_eq!(projected.selector_columns(), snapshot.selector_cols());
    assert_eq!(projected.compiler_audit(), snapshot.compiler_audit());
    assert_eq!(projected.row_artifacts().len(), selected_rows.len());

    for (row, compact) in selected_rows.into_iter().zip(projected.row_artifacts()) {
        let materialized = snapshot
            .materialize_row(row)
            .expect("materialize differential row");
        assert_eq!(compact.schema_version(), materialized.schema_version());
        assert_eq!(compact.rows(), materialized.matrix_row().rows());
        assert_eq!(compact.columns(), materialized.matrix_row().columns());
        assert_eq!(compact.emitted_row(), materialized.matrix_row().emitted_row());
        assert_eq!(compact.run_index(), materialized.run_index());
        assert_eq!(compact.family(), materialized.family());
        assert_eq!(compact.arm(), materialized.arm());
        for port in 0..13 {
            let expected = materialized
                .matrix_row()
                .port(port)
                .expect("materialized port")
                .iter()
                .map(|term| (term.column(), term.coefficient()))
                .collect::<Vec<_>>();
            assert_eq!(
                expand_projected_port(&compact.ports()[port]),
                expected,
                "row {row}, port {port}"
            );
        }
    }
}

#[test]
fn projected_product_sum_rows_include_exact_executable_rewrite_steps() {
    let fixtures = [snapshot_arm(7), snapshot_arm(19)];
    let shapes = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, 0)
        .expect("compile product-sum provenance fixture");
    let rewrite = relation
        .selective_compiler_audit()
        .expect("selective compiler audit")
        .rows()
        .rewrites()
        .iter()
        .find(|rewrite| {
            rewrite.arm() == 0
                && rewrite.kind() == SelectiveRewriteKind::ProductSum
                && !rewrite.emitted_rows().is_empty()
        })
        .expect("nonempty product-sum rewrite");
    let selected_rows = rewrite.emitted_rows().collect::<Vec<_>>();
    let source_columns = (0..shapes[0].m).collect::<Vec<_>>();
    let projected = audit_multi_branch_selective_rows_with_complete_source_provenance_with_alignment(
        &shapes,
        0,
        D,
        0,
        &selected_rows,
        0,
        &source_columns,
        &[],
    )
    .expect("project exact product-sum rewrite provenance");
    let source = projected
        .source_provenance()
        .expect("complete source provenance");

    assert_eq!(source.arm(), 0);
    assert_eq!(source.rewrite_steps().len(), selected_rows.len());
    assert!(source
        .rewrite_steps()
        .iter()
        .zip(&selected_rows)
        .all(|(step, row)| {
            step.emitted_row() == *row
                && step.rewrite_id() == rewrite.id().index()
                && step.kind() == SelectiveRewriteKind::ProductSum
                && step.factors().len() <= 5
        }));

    let snapshot = relation
        .selective_snapshot()
        .expect("checked grouped-product snapshot");
    let materialized_rows = selected_rows
        .iter()
        .map(|&row| {
            snapshot
                .materialize_row(row)
                .expect("materialize grouped-product row")
        })
        .collect::<Vec<_>>();
    assert_grouped_product_rewrite_artifact_matches_committed(
        shapes[0].n,
        shapes[0].m,
        &shapes[0],
        source,
        &materialized_rows,
    );
}

#[test]
fn selective_snapshot_interpreter_matches_live_encoder() {
    let fixtures = [snapshot_arm(7), snapshot_arm(19)];
    let shapes = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, 0).expect("compile selective fixture");
    let snapshot = relation
        .selective_snapshot()
        .expect("checked selective snapshot");

    assert!(core::ptr::eq(snapshot.structure(), relation.structure()));
    let layout = relation
        .selective_compiler_audit()
        .expect("selective compiler audit")
        .layout();
    assert!(core::ptr::eq(snapshot.layout(), layout));
    assert_eq!(snapshot.layout().total_columns(), snapshot.structure().m);
    assert_eq!(snapshot.layout().public_input_len(), snapshot.public_input_len());
    assert_eq!(snapshot.layout().selector_columns(), snapshot.selector_cols());
    assert_eq!(snapshot.public_input_len(), relation.public_input_len());
    assert_eq!(snapshot.selector_cols(), relation.selector_cols());
    assert_eq!(snapshot.arm_count(), fixtures.len());
    let first = snapshot.arm(0).expect("first snapshot arm");
    assert!(first.coordinate_aliases().any(|alias| alias.is_some()));
    assert!(first.equality_sources().any(|source| source.is_some()));
    assert_ne!(first.derived_product_sums().len(), 0);
    for (arm, plan) in snapshot.arms().enumerate() {
        for (field, source) in plan.equality_sources().enumerate() {
            let Some(source) = source else {
                continue;
            };
            assert!(source < field, "arm {arm} equality source must be earlier");
            assert_eq!(
                plan.slot(field),
                plan.slot(source),
                "arm {arm} equality alias must reuse its source slot"
            );
        }
    }

    for (arm, (_, source_assignment)) in fixtures.iter().enumerate() {
        let live = relation
            .encode(arm, source_assignment)
            .expect("live selective encoding");
        let packed = relation
            .encode_signed_unit(arm, source_assignment)
            .expect("packed selective encoding");
        assert_eq!(packed.to_dense(), live, "arm {arm} packed encoding drifted");
        let replayed = snapshot
            .encode(arm, source_assignment)
            .expect("snapshot plan encoding");
        assert_eq!(replayed, live, "arm {arm} snapshot plan drifted");
        assert!(relation.is_satisfied(&replayed));

        let mut wrong_width = source_assignment.clone();
        wrong_width[1] = F::from_u64(2);
        let live_error = relation
            .encode(arm, &wrong_width)
            .expect_err("live width rejection");
        let replay_error = snapshot
            .encode(arm, &wrong_width)
            .expect_err("snapshot width rejection");
        assert_eq!(format!("{replay_error:?}"), format!("{live_error:?}"));

        let plan = snapshot.arm(arm).expect("snapshot arm");
        let (equal_field, equal_source) = plan
            .equality_sources()
            .enumerate()
            .find_map(|(field, source)| source.map(|source| (field, source)))
            .expect("fixture equality alias");
        let mut wrong_equality = source_assignment.clone();
        wrong_equality[equal_field] = wrong_equality[equal_source] + F::ONE;
        let live_error = relation
            .encode(arm, &wrong_equality)
            .expect_err("live equality rejection");
        let replay_error = snapshot
            .encode(arm, &wrong_equality)
            .expect_err("snapshot equality rejection");
        assert_eq!(format!("{replay_error:?}"), format!("{live_error:?}"));

        let balanced_field = (snapshot.public_field_count()..plan.field_count())
            .find(|&field| {
                plan.slot(field).is_some_and(|slot| slot.len() == 41)
                    && plan.coordinate_alias(field).is_none()
                    && plan.equality_source(field).is_none()
                    && !plan
                        .coordinate_aliases()
                        .flatten()
                        .any(|alias| alias.source_field() == field)
            })
            .expect("fixture balanced field without decomposition children");
        let mut negative_balanced = source_assignment.clone();
        negative_balanced[balanced_field] = -F::ONE;
        for field in balanced_field + 1..plan.field_count() {
            if let Some(source) = plan.equality_source(field) {
                negative_balanced[field] = negative_balanced[source];
            }
        }
        let live = relation
            .encode(arm, &negative_balanced)
            .expect("live negative balanced encoding");
        let packed = relation
            .encode_signed_unit(arm, &negative_balanced)
            .expect("packed negative balanced encoding");
        assert_eq!(packed.to_dense(), live, "arm {arm} packed negative encoding drifted");
        let replayed = snapshot
            .encode(arm, &negative_balanced)
            .expect("snapshot negative balanced encoding");
        assert_eq!(replayed, live, "arm {arm} negative balanced replay drifted");
    }
}

#[test]
fn packed_selective_encoder_rejects_non_unit_centered_coordinate() {
    let fixtures = [3, 5].map(|value| {
        let mut builder = R1csBuilder::new();
        let field = builder.alloc(F::from_u64(value));
        enforce_commit_fields(&mut builder, CCS_CLAIM_SIS_CONFIG, &[field]).expect("synthesize centered digits");
        lower_field_r1cs(builder, &[])
            .expect("lower centered fixture")
            .into_parts()
    });
    let shapes = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, 0).expect("compile centered fixture");
    let snapshot = relation.selective_snapshot().expect("centered snapshot");
    let centered_field = snapshot
        .arm(0)
        .expect("first arm")
        .centered_columns()
        .iter()
        .position(|&centered| centered)
        .expect("centered coordinate");
    let mut non_unit = fixtures[0].1.clone();
    non_unit[centered_field] = F::from_u64(2);

    assert!(relation.encode(0, &non_unit).is_ok(), "dense behavior changed");
    assert!(matches!(
        relation.encode_signed_unit(0, &non_unit),
        Err(LowNormR1csError::PackedNonSignedUnit { .. })
    ));
}

#[test]
fn selective_snapshot_rejects_non_selective_relation() {
    let fixtures = [snapshot_arm(3), snapshot_arm(5)];
    let shapes = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation = build_multi_branch_low_norm_r1cs(&shapes, 0).expect("compile ordinary multi-branch fixture");

    assert!(matches!(
        relation.selective_snapshot(),
        Err(SelectiveSnapshotError::NotSelective)
    ));
}

#[test]
fn selective_snapshot_reconciles_shared_private_layout_without_copying_plans() {
    let fixtures = [snapshot_arm(11), snapshot_arm(23)];
    let shapes = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 1, D, 0).expect("compile shared fixture");
    let snapshot = relation
        .selective_snapshot()
        .expect("checked shared snapshot");
    let shared = snapshot.layout().shared_private_columns();
    assert!(!shared.is_empty());
    let shared_field = snapshot.public_field_count();
    let expected = snapshot.arm(0).expect("first arm").slot(shared_field);
    assert!(expected.is_some_and(|slot| slot.range() == shared));
    for arm in snapshot.arms().skip(1) {
        assert_eq!(arm.slot(shared_field), expected);
    }
}

#[test]
fn selective_snapshot_selector_gate_coverage_matches_final_matrices() {
    let fixtures = [snapshot_arm(13), snapshot_arm(29), snapshot_arm(47)];
    let shapes = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, 0).expect("compile gate fixture");
    let snapshot = relation
        .selective_snapshot()
        .expect("checked gate snapshot");
    let coverage = snapshot
        .selector_gate_coverage()
        .expect("exact selector-port coverage");

    assert_eq!(coverage.rows(), snapshot.structure().n);
    assert_eq!(coverage.columns(), snapshot.structure().m);
    assert_eq!(coverage.selector_columns(), snapshot.selector_cols());
    assert_eq!(
        coverage.owner_runs(),
        snapshot.compiler_audit().rows().emitted_runs(),
        "coverage must preserve the compiler's complete ownership ledger"
    );

    let mut saw_general_arm = false;
    let mut saw_evaluation_arm = false;
    let nonempty_owners = coverage
        .owner_runs()
        .iter()
        .filter(|owner| !owner.emitted_rows().is_empty())
        .collect::<Vec<_>>();
    assert_eq!(nonempty_owners.len(), coverage.gate_runs().len());
    let mut nonempty_families = Vec::new();
    for (owner, gate) in nonempty_owners.into_iter().zip(coverage.gate_runs()) {
        let expected = expected_gate_for_owner(owner.family(), owner.arm(), coverage.selector_columns());
        assert_eq!(owner.emitted_rows(), gate.emitted_rows());
        assert_eq!((gate.port(), gate.column()), expected);
        assert_eq!(gate.coefficient(), F::ONE);
        if owner.arm().is_some() {
            saw_general_arm |= matches!(
                expected.0,
                SelectiveGatePort::General | SelectiveGatePort::GeneralEvaluation
            );
            saw_evaluation_arm |= matches!(
                expected.0,
                SelectiveGatePort::Evaluation | SelectiveGatePort::GeneralEvaluation
            );
        }
        if !nonempty_families.contains(&owner.family()) {
            nonempty_families.push(owner.family());
        }
    }
    assert!(saw_general_arm, "fixture must exercise the general selector port");
    assert!(saw_evaluation_arm, "fixture must exercise the evaluation selector port");
    for family in [
        SelectiveEmittedRowFamily::SelectorDomain,
        SelectiveEmittedRowFamily::ArmDomain,
        SelectiveEmittedRowFamily::OneHot,
        SelectiveEmittedRowFamily::PublicPadding,
        SelectiveEmittedRowFamily::PrivatePadding,
        SelectiveEmittedRowFamily::Retained,
        SelectiveEmittedRowFamily::ProductSum,
        SelectiveEmittedRowFamily::RingPadding,
    ] {
        assert!(nonempty_families.contains(&family), "missing fixture family {family:?}");
    }
    for family in [
        SelectiveEmittedRowFamily::SharedDomain,
        SelectiveEmittedRowFamily::Poseidon2,
        SelectiveEmittedRowFamily::CenteredUnit,
        SelectiveEmittedRowFamily::ShiftedTernaryCanonical,
        SelectiveEmittedRowFamily::PolynomialEvaluation,
    ] {
        assert!(
            !nonempty_families.contains(&family),
            "fixture unexpectedly gained {family:?}; add explicit review coverage"
        );
    }

    let coalesced = coverage.coalesced_owner_gate_runs();
    assert!(!coalesced.is_empty());
    assert!(coalesced.len() <= coverage.gate_runs().len());
    let mut cursor = 0usize;
    for run in &coalesced {
        let rows = run.emitted_rows();
        assert_eq!(rows.start, cursor, "coalesced coverage must have no gap");
        assert!(rows.start < rows.end, "coalesced runs must be nonempty");
        cursor = rows.end;
    }
    assert_eq!(cursor, coverage.rows());
    for (owner, gate) in coverage
        .owner_runs()
        .iter()
        .filter(|owner| !owner.emitted_rows().is_empty())
        .zip(coverage.gate_runs())
    {
        let owner_rows = owner.emitted_rows();
        let run = coalesced
            .iter()
            .find(|run| {
                let rows = run.emitted_rows();
                rows.start <= owner_rows.start && owner_rows.end <= rows.end
            })
            .expect("each exact owner run has one coalesced owner/gate run");
        assert_eq!((run.family(), run.arm()), (owner.family(), owner.arm()));
        assert_eq!((run.port(), run.column()), (gate.port(), gate.column()));
        assert_eq!(run.coefficient(), gate.coefficient());
    }
    for adjacent in coalesced.windows(2) {
        assert!(
            adjacent[0].family() != adjacent[1].family()
                || adjacent[0].arm() != adjacent[1].arm()
                || adjacent[0].port() != adjacent[1].port()
                || adjacent[0].column() != adjacent[1].column()
                || adjacent[0].coefficient() != adjacent[1].coefficient(),
            "coalesced coverage must be maximal"
        );
    }
    assert_eq!(coverage.polynomial_arity(), 13);
    assert_eq!(coverage.polynomial_terms().len(), 74);
    assert_selector_coverage_artifact_matches_committed(&coverage);
}

#[test]
fn selective_carrier_270_lean_artifact_matches_compiler() {
    let fixtures = f_prime_public_carrier_fixture();
    let shapes = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, F_PRIME_PUBLIC_INPUT_LEN % D)
            .expect("compile F-prime-width carrier fixture");
    let snapshot = relation
        .selective_snapshot()
        .expect("checked F-prime-width carrier snapshot");
    let layout = snapshot.layout();

    assert_eq!(layout.logical_public_input_len(), F_PRIME_PUBLIC_INPUT_LEN);
    assert_eq!(layout.public_input_len(), F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
    assert!(layout.public_padding_columns().iter().copied().eq(257..270));
    assert_eq!(layout.selector_columns(), &[270, 271, 272]);
    assert!(layout
        .private_alignment_padding_columns()
        .iter()
        .copied()
        .eq(273..311));
    assert_eq!(layout.branch_columns().start, 311);

    let emitted_runs = snapshot.compiler_audit().rows().emitted_runs();
    let (selector_run_index, selector_run) = emitted_runs
        .iter()
        .enumerate()
        .find(|(_, run)| run.family() == SelectiveEmittedRowFamily::SelectorDomain)
        .expect("selector-domain run");
    assert_eq!(selector_run.emitted_rows(), 0..3);
    let selector_rows = selector_run
        .emitted_rows()
        .enumerate()
        .map(|(arm, row)| {
            let artifact = snapshot
                .materialize_row(row)
                .expect("materialize selector row");
            assert_eq!(artifact.run_index(), selector_run_index);
            assert_eq!(artifact.family(), SelectiveEmittedRowFamily::SelectorDomain);
            assert_eq!(artifact.arm(), None);
            let terms = |port| {
                artifact
                    .matrix_row()
                    .port(port)
                    .expect("selector row port")
                    .iter()
                    .map(|term| (term.column(), term.coefficient()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(terms(0), vec![(270 + arm, F::ONE)]);
            assert_eq!(terms(1), vec![(0, F::ONE)]);
            for port in 2..13 {
                assert!(terms(port).is_empty(), "unexpected selector row term at port {port}");
            }
            artifact
        })
        .collect::<Vec<_>>();

    let (one_hot_run_index, one_hot_run) = emitted_runs
        .iter()
        .enumerate()
        .find(|(_, run)| run.family() == SelectiveEmittedRowFamily::OneHot)
        .expect("selector-total run");
    assert_eq!(one_hot_run.emitted_rows(), 3..4);
    let one_hot_row = snapshot
        .materialize_row(one_hot_run.emitted_rows().start)
        .expect("materialize selector-total row");
    assert_eq!(one_hot_row.run_index(), one_hot_run_index);
    assert_eq!(one_hot_row.family(), SelectiveEmittedRowFamily::OneHot);
    assert_eq!(one_hot_row.arm(), None);
    let one_hot_terms = |port| {
        one_hot_row
            .matrix_row()
            .port(port)
            .expect("selector-total row port")
            .iter()
            .map(|term| (term.column(), term.coefficient()))
            .collect::<Vec<_>>()
    };
    assert_eq!(one_hot_terms(1), vec![(0, F::ONE)]);
    assert_eq!(
        one_hot_terms(4),
        vec![(0, -F::ONE), (270, F::ONE), (271, F::ONE), (272, F::ONE)]
    );
    for port in [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12] {
        assert!(
            one_hot_terms(port).is_empty(),
            "unexpected selector-total term at port {port}"
        );
    }

    let (gated_run_index, gated_run) = emitted_runs
        .iter()
        .enumerate()
        .find(|(_, run)| run.family() == SelectiveEmittedRowFamily::Retained && run.arm() == Some(0))
        .expect("arm-zero retained run");
    assert_eq!(gated_run.emitted_rows().start, 55);
    let gated_row = snapshot
        .materialize_row(gated_run.emitted_rows().start)
        .expect("materialize representative gated row");
    assert_eq!(gated_row.run_index(), gated_run_index);
    assert_eq!(gated_row.family(), SelectiveEmittedRowFamily::Retained);
    assert_eq!(gated_row.arm(), Some(0));
    let gated_terms = |port| {
        gated_row
            .matrix_row()
            .port(port)
            .expect("representative gated row port")
            .iter()
            .map(|term| (term.column(), term.coefficient()))
            .collect::<Vec<_>>()
    };
    assert_eq!(gated_terms(1), vec![(270, F::ONE)]);
    assert_eq!(gated_terms(2), vec![(1, F::ONE)]);
    assert_eq!(gated_terms(3), vec![(0, -F::ONE), (1, F::ONE)]);
    for port in [0, 4, 5, 6, 7, 8, 9, 10, 11, 12] {
        assert!(gated_terms(port).is_empty(), "unexpected gated-row term at port {port}");
    }

    let padding_runs = snapshot
        .compiler_audit()
        .rows()
        .emitted_runs()
        .iter()
        .enumerate()
        .filter(|(_, run)| run.family() == SelectiveEmittedRowFamily::PublicPadding)
        .collect::<Vec<_>>();
    assert_eq!(padding_runs.len(), 1, "public padding must have one exclusive run");
    let (run_index, padding_run) = padding_runs[0];
    assert_eq!(padding_run.arm(), None);
    assert_eq!(padding_run.emitted_rows().len(), 13);

    let padding_rows = padding_run
        .emitted_rows()
        .enumerate()
        .map(|(offset, row)| {
            let artifact = snapshot
                .materialize_row(row)
                .expect("materialize public-padding row");
            assert_eq!(artifact.run_index(), run_index);
            assert_eq!(artifact.family(), SelectiveEmittedRowFamily::PublicPadding);
            assert_eq!(artifact.arm(), None);
            let terms = |port| {
                artifact
                    .matrix_row()
                    .port(port)
                    .expect("selective port")
                    .iter()
                    .map(|term| (term.column(), term.coefficient()))
                    .collect::<Vec<_>>()
            };
            for port in 0..13 {
                let expected = match port {
                    1 => vec![(0, F::ONE)],
                    4 => vec![(257 + offset, F::ONE)],
                    _ => Vec::new(),
                };
                assert_eq!(terms(port), expected, "padding offset {offset}, port {port}");
            }
            artifact
        })
        .collect::<Vec<_>>();

    assert_carrier_270_artifact_matches_committed(layout, &selector_rows, &one_hot_row, &gated_row, &padding_rows);
}

#[test]
fn materialized_selector_domain_rows_are_classified_from_final_matrices() {
    let fixtures = [snapshot_arm(31), snapshot_arm(43)];
    let shapes = fixtures
        .iter()
        .map(|(shape, _)| shape.clone())
        .collect::<Vec<_>>();
    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&shapes, 0, D, 0).expect("compile selector fixture");
    let snapshot = relation
        .selective_snapshot()
        .expect("checked selective snapshot");
    let (run_index, run) = snapshot
        .compiler_audit()
        .rows()
        .emitted_runs()
        .iter()
        .enumerate()
        .find(|(_, run)| run.family() == SelectiveEmittedRowFamily::SelectorDomain)
        .expect("selector-domain run");

    let first_artifact = snapshot
        .materialize_row(run.emitted_rows().start)
        .expect("first materialized selector row");
    assert_selector_row_artifact_matches_committed(&first_artifact);

    for (offset, row) in run.emitted_rows().enumerate() {
        let artifact = snapshot
            .materialize_row(row)
            .expect("materialized selector row");
        assert_eq!(artifact.run_index(), run_index);
        assert_eq!(artifact.family(), SelectiveEmittedRowFamily::SelectorDomain);
        assert_eq!(artifact.arm(), None);
        assert_eq!(artifact.matrix_row().emitted_row(), row);
        assert_eq!(artifact.matrix_row().rows(), snapshot.structure().n);
        assert_eq!(artifact.matrix_row().columns(), snapshot.structure().m);

        let terms = |port| {
            artifact
                .matrix_row()
                .port(port)
                .expect("selective port")
                .iter()
                .map(|term| (term.column(), term.coefficient()))
                .collect::<Vec<_>>()
        };
        // Classification comes from the final coefficients. The ledger label
        // above is checked only as a separate provenance join.
        assert_eq!(terms(0), vec![(snapshot.selector_cols()[offset], F::ONE)]);
        assert_eq!(terms(1), vec![(0, F::ONE)]);
        for port in 2..13 {
            assert!(terms(port).is_empty(), "unexpected selector row term at port {port}");
        }
    }
}
