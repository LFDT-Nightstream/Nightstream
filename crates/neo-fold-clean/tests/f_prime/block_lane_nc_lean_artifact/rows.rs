//! Exact sparse source-row and compact thirteen-port emitted-row rendering.

use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::{
    R1csIvcBlockLaneNcSelectiveRowsAudit, R1csIvcBlockLaneNcSourceRowAudit,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveEmittedRowFamily, SelectiveProjectedRowArtifact, SelectiveProjectedTerm,
};
use neo_math::F;
use p3_field::PrimeField64;

use super::render::source_shape;
use super::{generated_header, GeneratedLeanFile, GENERATED_ROOT, IMPORT_ROOT, NAMESPACE_ROOT};

const SOURCE_ROWS_PER_SHARD: usize = 128;
const EMITTED_ROWS_PER_SHARD: usize = 64;
const MAX_NESTED_RECORDS: usize = 256;

fn lean_terms(terms: &[(usize, F)]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|&(column, coefficient)| format!(
                "{{ column := {column}, coefficient := {} }}",
                coefficient.as_canonical_u64()
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_projected_terms(terms: &[SelectiveProjectedTerm]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|term| format!(
                "{{ column := {}, coefficient := {} }}",
                term.column(),
                term.coefficient().as_canonical_u64()
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

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

fn render_source_row(
    contents: &mut String,
    row: &R1csIvcBlockLaneNcSourceRowAudit,
    source_rows: usize,
    source_columns: usize,
) {
    assert!(
        [row.a().len(), row.b().len(), row.c().len()]
            .into_iter()
            .all(|count| count <= MAX_NESTED_RECORDS),
        "source row {} has an oversized sparse port",
        row.index()
    );
    writeln!(contents, "  {{ schemaVersion := 1").expect("render source schema");
    writeln!(contents, "    rows := {source_rows}").expect("render source rows");
    writeln!(contents, "    columns := {source_columns}").expect("render source columns");
    writeln!(contents, "    sourceRow := {}", row.index()).expect("render source index");
    writeln!(contents, "    a := {}", lean_terms(row.a())).expect("render source A");
    writeln!(contents, "    b := {}", lean_terms(row.b())).expect("render source B");
    writeln!(contents, "    c := {} }}", lean_terms(row.c())).expect("render source C");
}

fn render_source_shard(
    audit: &R1csIvcBlockLaneNcSelectiveRowsAudit,
    index: usize,
    rows: &[R1csIvcBlockLaneNcSourceRowAudit],
) -> GeneratedLeanFile {
    assert!(!rows.is_empty() && rows.len() <= SOURCE_ROWS_PER_SHARD);
    let namespace = format!("{NAMESPACE_ROOT}.SourceRows.Chunk{index}");
    let (source_rows, source_columns) = source_shape(audit);
    let mut contents = generated_header("at most 128 exact normalized source A/B/C rows");
    writeln!(contents, "import {IMPORT_ROOT}\n").expect("render source import");
    writeln!(contents, "namespace {namespace}\n").expect("render source namespace");
    contents.push_str("set_option maxRecDepth 100000 in\n");
    contents.push_str("def values : List RawSourceRow := [\n");
    for (offset, row) in rows.iter().enumerate() {
        if offset != 0 {
            contents.push_str(",\n");
        }
        render_source_row(&mut contents, row, source_rows, source_columns);
    }
    writeln!(contents, "]\n\nend {namespace}").expect("render source namespace end");
    assert!(contents.lines().count() < 1_500, "source-row shard line limit");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/SourceRows/Chunk{index}.lean"),
        contents,
    }
}

pub(super) fn render_emitted_row(contents: &mut String, row: &SelectiveProjectedRowArtifact) {
    assert!(row.ports().iter().all(|port| {
        port.explicit().len() <= MAX_NESTED_RECORDS && port.geometric_runs().len() <= MAX_NESTED_RECORDS
    }));
    writeln!(contents, "  {{ schemaVersion := {}", row.schema_version()).expect("render emitted schema");
    writeln!(contents, "    rows := {}", row.rows()).expect("render emitted rows");
    writeln!(contents, "    columns := {}", row.columns()).expect("render emitted columns");
    writeln!(contents, "    emittedRow := {}", row.emitted_row()).expect("render emitted index");
    writeln!(contents, "    runIndex := {}", row.run_index()).expect("render emitted owner");
    writeln!(contents, "    family := .{}", lean_family(row.family())).expect("render emitted family");
    match row.arm() {
        Some(arm) => writeln!(contents, "    arm := some {arm}").expect("render emitted arm"),
        None => contents.push_str("    arm := none\n"),
    }
    contents.push_str("    ports := [\n");
    for (port_index, port) in row.ports().iter().enumerate() {
        let separator = if port_index == 0 { "        " } else { "      , " };
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
            contents,
            "{separator}{{ explicit := {}, geometric := [{geometric}] }}",
            lean_projected_terms(port.explicit())
        )
        .expect("render emitted port");
    }
    contents.push_str("      ] }\n");
}

fn render_emitted_shard(index: usize, rows: &[SelectiveProjectedRowArtifact]) -> GeneratedLeanFile {
    assert!(!rows.is_empty() && rows.len() <= EMITTED_ROWS_PER_SHARD);
    let namespace = format!("{NAMESPACE_ROOT}.EmittedRows.Chunk{index}");
    let mut contents = generated_header("at most 64 exact compact thirteen-port selective rows");
    writeln!(contents, "import {IMPORT_ROOT}\n").expect("render emitted import");
    writeln!(contents, "namespace {namespace}\n").expect("render emitted namespace");
    contents.push_str("set_option maxRecDepth 100000 in\n");
    contents.push_str("def values : List RawEmittedRow := [\n");
    for (offset, row) in rows.iter().enumerate() {
        if offset != 0 {
            contents.push_str(",\n");
        }
        render_emitted_row(&mut contents, row);
    }
    writeln!(contents, "]\n\nend {namespace}").expect("render emitted namespace end");
    assert!(contents.lines().count() < 1_500, "emitted-row shard line limit");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/EmittedRows/Chunk{index}.lean"),
        contents,
    }
}

fn render_root(family: &str, shard_count: usize, value_type: &str) -> GeneratedLeanFile {
    assert_ne!(shard_count, 0, "generated row family has at least one shard");
    let namespace = format!("{NAMESPACE_ROOT}.{family}");
    let mut contents = generated_header(&format!("the ordered concatenation of every {family} shard"));
    for index in 0..shard_count {
        writeln!(contents, "import {NAMESPACE_ROOT}.{family}.Chunk{index}").expect("render shard import");
    }
    writeln!(contents, "\nnamespace {namespace}\n").expect("render row root namespace");
    contents.push_str("set_option maxRecDepth 100000 in\n");
    writeln!(contents, "def values : List {value_type} :=").expect("render row root type");
    for index in 0..shard_count {
        let continuation = if index + 1 == shard_count { "" } else { " ++" };
        writeln!(contents, "  Chunk{index}.values{continuation}").expect("render row root shard");
    }
    writeln!(contents, "\nend {namespace}").expect("render row root namespace end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/{family}.lean"),
        contents,
    }
}

pub(super) fn source_rows(audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) -> Vec<GeneratedLeanFile> {
    let source_rows = audit.source_row_artifacts();
    assert!(
        source_rows
            .windows(2)
            .all(|pair| pair[0].index() < pair[1].index()),
        "source rows are strictly ordered"
    );
    let mut files = source_rows
        .chunks(SOURCE_ROWS_PER_SHARD)
        .enumerate()
        .map(|(index, rows)| render_source_shard(audit, index, rows))
        .collect::<Vec<_>>();
    files.push(render_root("SourceRows", files.len(), "RawSourceRow"));
    files
}

pub(super) fn emitted_rows(audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) -> Vec<GeneratedLeanFile> {
    let rows = audit.projected_rows().row_artifacts();
    assert!(
        rows.windows(2)
            .all(|pair| pair[0].emitted_row() < pair[1].emitted_row()),
        "selected emitted rows are strictly ordered"
    );
    let mut files = rows
        .chunks(EMITTED_ROWS_PER_SHARD)
        .enumerate()
        .map(|(index, rows)| render_emitted_shard(index, rows))
        .collect::<Vec<_>>();
    files.push(render_root("EmittedRows", files.len(), "RawEmittedRow"));
    files
}
