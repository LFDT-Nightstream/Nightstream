//! Exact final selective-CCS row renderer for the bounded projection slice.
//!
//! Owns: materialization of every selected emitted row from the compiled
//! thirteen-port structure, its join to the exclusive emitted-run owner, and
//! deterministic sharding into the existing fail-closed Lean wire schema.
//!
//! Does not own: row semantics, source-expression decoding, selector truth,
//! upstream `y_zcol` authority, security reductions, or row removal.
//!
//! Emits constraints: no; this test generator records existing compiler rows.
//!
//! | Artifact leaf | Mathematical obligation | Authority class |
//! |---|---|---|
//! | selected matrix row | exact thirteen-port final A/B/C contribution stream | computed |

use std::collections::BTreeMap;
use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::{
    PiRlcYZcolProjectionLoweringDisposition, PiRlcYZcolProjectionRowMappingAudit,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveEmittedRowFamily, SelectiveProjectedRowArtifact, SelectiveProjectedRowsAudit, SelectiveRewriteKind,
};
use p3_field::PrimeField64;

use super::GeneratedLeanFile;

const GENERATED_ROOT: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiRlcProjection/YZcol/Generated";
const IMPORT_ROOT: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Schema";
const NAMESPACE_ROOT: &str =
    "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.SelectiveMatrixRows";
const SHARD_IMPORT_ROOT: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.SelectiveMatrixRows";
const MATERIALIZED_NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized";
const ROWS_PER_SHARD: usize = 40;
const STEADY_ARM: usize = 2;

const GENERATED_HEADER: &str = r#"/-
Generated file: exact materialized selective rows; do not hand-edit.

Owns: the exact ordered compact-row payload stored in this generated module.

Does not own: decoding, row satisfaction, source semantics, selector truth,
security events, or permission to remove rows.

Emits constraints: no.

| Artifact leaf | Mathematical obligation | Authority class |
|---|---|---|
| generated rows | exact Rust-rendered thirteen-port row data and order | computed |
-/

"#;

fn lean_family(family: SelectiveEmittedRowFamily) -> &'static str {
    match family {
        SelectiveEmittedRowFamily::SelectorDomain => "selectorDomain",
        SelectiveEmittedRowFamily::SharedDomain => "sharedDomain",
        SelectiveEmittedRowFamily::ArmDomain => "armDomain",
        SelectiveEmittedRowFamily::OneHot => "oneHot",
        SelectiveEmittedRowFamily::PublicPadding => "publicPadding",
        SelectiveEmittedRowFamily::PrivatePadding => "privatePadding",
        SelectiveEmittedRowFamily::Retained => "retained",
        SelectiveEmittedRowFamily::Poseidon2 => "poseidon2",
        SelectiveEmittedRowFamily::CenteredUnit => "centeredUnit",
        SelectiveEmittedRowFamily::ShiftedTernaryCanonical => "shiftedTernaryCanonical",
        SelectiveEmittedRowFamily::PolynomialEvaluation => "polynomialEvaluation",
        SelectiveEmittedRowFamily::ProductSum => "productSum",
        SelectiveEmittedRowFamily::RingPadding => "ringPadding",
    }
}

fn expected_family(disposition: PiRlcYZcolProjectionLoweringDisposition) -> Option<SelectiveEmittedRowFamily> {
    match disposition {
        PiRlcYZcolProjectionLoweringDisposition::Retained => Some(SelectiveEmittedRowFamily::Retained),
        PiRlcYZcolProjectionLoweringDisposition::Rewrite { kind, .. } => match kind {
            SelectiveRewriteKind::PolynomialEvaluation => Some(SelectiveEmittedRowFamily::PolynomialEvaluation),
            SelectiveRewriteKind::ProductSum => Some(SelectiveEmittedRowFamily::ProductSum),
            SelectiveRewriteKind::LinearDefinition => None,
            other => panic!("unsupported focused projection rewrite {other:?}"),
        },
    }
}

fn selected_rows<'a>(
    audit: &PiRlcYZcolProjectionRowMappingAudit,
    projected: &'a SelectiveProjectedRowsAudit,
) -> Vec<&'a SelectiveProjectedRowArtifact> {
    assert_eq!(projected.rows(), audit.final_relation_row_count());
    assert_eq!(projected.selector_columns().len(), 3, "fixed-point branch count");

    let mut expected_rows = BTreeMap::new();
    for fragment in audit.leaves().iter().flat_map(|leaf| leaf.fragments()) {
        let expected = expected_family(fragment.disposition());
        if expected.is_none() {
            assert!(fragment.emitted_rows().is_empty(), "eliminated definition emitted rows");
            continue;
        }
        let expected = expected.expect("nonempty rewrite family");
        for row in fragment.emitted_rows() {
            assert!(
                expected_rows.insert(row, expected).is_none(),
                "selected emitted row {row} has multiple projection owners"
            );
        }
    }
    let mut rows = projected.row_artifacts().iter().collect::<Vec<_>>();
    rows.sort_by_key(|artifact| artifact.emitted_row());
    for artifact in &rows {
        let row = artifact.emitted_row();
        let expected = expected_rows
            .remove(&row)
            .unwrap_or_else(|| panic!("projected emitter returned unselected row {row}"));
        assert_eq!(artifact.family(), expected, "selected row {row} family");
        assert_eq!(artifact.arm(), Some(STEADY_ARM), "selected row {row} arm");
    }
    assert_eq!(rows.len(), 1_254, "exact selected final-row count");
    assert!(expected_rows.is_empty(), "projected emitter omitted selected rows");
    rows
}

pub(super) fn write_raw_row(rendered: &mut String, artifact: &SelectiveProjectedRowArtifact) {
    writeln!(rendered, "  {{ schemaVersion := {}", artifact.schema_version()).expect("render schema version");
    writeln!(rendered, "    rows := {}", artifact.rows()).expect("render rows");
    writeln!(rendered, "    columns := {}", artifact.columns()).expect("render columns");
    writeln!(rendered, "    emittedRow := {}", artifact.emitted_row()).expect("render emitted row");
    writeln!(rendered, "    runIndex := {}", artifact.run_index()).expect("render run index");
    writeln!(rendered, "    family := .{}", lean_family(artifact.family())).expect("render family");
    match artifact.arm() {
        Some(arm) => writeln!(rendered, "    arm := some {arm}").expect("render arm"),
        None => writeln!(rendered, "    arm := none").expect("render arm"),
    }
    writeln!(rendered, "    ports := [").expect("render ports header");
    for (port_index, port) in artifact.ports().iter().enumerate() {
        let separator = if port_index == 0 { "        " } else { "      , " };
        let explicit = port
            .explicit()
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
        let geometric = port
            .geometric_runs()
            .iter()
            .map(|run| {
                format!(
                    "{{ columnStart := {}, length := {}, initial := {}, ratio := {} }}",
                    run.column_start(),
                    run.length(),
                    run.initial().as_canonical_u64(),
                    run.ratio().as_canonical_u64()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            rendered,
            "{separator}{{ explicit := [{explicit}], geometric := [{geometric}] }}"
        )
        .expect("render port");
    }
    writeln!(rendered, "      ] }}").expect("render row footer");
}

fn render_shard(index: usize, rows: &[&SelectiveProjectedRowArtifact]) -> GeneratedLeanFile {
    let module = format!("Row{index}");
    let namespace = format!("{NAMESPACE_ROOT}.{module}");
    let mut contents = String::new();
    contents.push_str(GENERATED_HEADER);
    writeln!(contents, "import {IMPORT_ROOT}\n").expect("render shard import");
    writeln!(contents, "namespace {namespace}\n").expect("render shard namespace");
    writeln!(contents, "open {MATERIALIZED_NAMESPACE}\n").expect("render materialized namespace");
    contents.push_str("set_option maxRecDepth 100000 in\n");
    contents.push_str("def rawRows : List RawRow := [\n");
    for (offset, row) in rows.iter().enumerate() {
        if offset != 0 {
            contents.push_str(",\n");
        }
        write_raw_row(&mut contents, row);
    }
    writeln!(contents, "]\n\nend {namespace}").expect("render shard footer");
    assert!(
        contents.lines().count() < 1_500,
        "generated selective-row shard {index} exceeds the repository line limit"
    );
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/SelectiveMatrixRows/{module}.lean"),
        contents,
    }
}

fn render_index(shard_count: usize, projected: &SelectiveProjectedRowsAudit) -> GeneratedLeanFile {
    let mut contents = String::new();
    contents.push_str(GENERATED_HEADER);
    for index in 0..shard_count {
        writeln!(contents, "import {SHARD_IMPORT_ROOT}.Row{index}").expect("render index imports");
    }
    contents.push('\n');
    writeln!(contents, "namespace {NAMESPACE_ROOT}\n").expect("render index namespace");
    writeln!(contents, "open {MATERIALIZED_NAMESPACE}\n").expect("render materialized namespace");
    writeln!(contents, "def finalRelationRows : Nat := {}", projected.rows()).expect("render final rows");
    writeln!(contents, "def finalRelationColumns : Nat := {}", projected.columns()).expect("render final columns");
    contents.push_str("def constantOneColumn : Nat := 0\n");
    writeln!(
        contents,
        "def steadySelectorColumn : Nat := {}\n",
        projected.selector_columns()[STEADY_ARM]
    )
    .expect("render steady selector");
    contents.push_str("def rawRows : List RawRow :=\n  ");
    for index in 0..shard_count {
        if index != 0 {
            contents.push_str(" ++\n  ");
        }
        write!(contents, "Row{index}.rawRows").expect("render shard join");
    }
    writeln!(contents, "\n\nend {NAMESPACE_ROOT}").expect("render index footer");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/SelectiveMatrixRows.lean"),
        contents,
    }
}

pub(super) fn render(
    audit: &PiRlcYZcolProjectionRowMappingAudit,
    projected: &SelectiveProjectedRowsAudit,
) -> Vec<GeneratedLeanFile> {
    let rows = selected_rows(audit, projected);
    let mut files = rows
        .chunks(ROWS_PER_SHARD)
        .enumerate()
        .map(|(index, rows)| render_shard(index, rows))
        .collect::<Vec<_>>();
    files.push(render_index(files.len(), projected));
    files
}
