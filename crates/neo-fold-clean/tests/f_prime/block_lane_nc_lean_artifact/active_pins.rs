//! Exact active public-write, selector-value, and selector-row rendering.
//!
//! Owns the fixed post-PiDEC recursive profile's 270 prepared public-write
//! addresses, block/lane decomposition, constant/selector pins, and the exact
//! sparse rows enforcing selector domain and one-hotness.
//!
//! Does not own runtime source values, NC semantics, commitment binding, or
//! permission to remove rows.

use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcBlockLaneNcSelectiveRowsAudit;
use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveEmittedRowFamily, SelectiveProjectedPublicCoordinateSource, SelectiveProjectedRowArtifact,
    SelectiveProjectedTerm,
};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use super::render::source_shape;
use super::rows::render_emitted_row;
use super::{generated_header, lean_nat_list, GeneratedLeanFile, GENERATED_ROOT, IMPORT_ROOT, NAMESPACE_ROOT};

const PUBLIC_COORDINATES: usize = 270;
const LOGICAL_PUBLIC_COORDINATES: usize = 257;
const PACKED_BLOCKS: usize = 5;
const SELECTOR_COLUMNS: [usize; 3] = [270, 271, 272];
const RECURSIVE_SELECTOR_VALUES: [usize; 3] = [0, 0, 1];
const COORDINATES_PER_SHARD: usize = 128;

fn expected_public_source(column: usize) -> SelectiveProjectedPublicCoordinateSource {
    match column {
        0 => SelectiveProjectedPublicCoordinateSource::ConstantOne,
        1..LOGICAL_PUBLIC_COORDINATES => SelectiveProjectedPublicCoordinateSource::SourceField(column),
        LOGICAL_PUBLIC_COORDINATES..PUBLIC_COORDINATES => SelectiveProjectedPublicCoordinateSource::FixedZero,
        _ => unreachable!("validated 270-coordinate public carrier"),
    }
}

fn lean_public_source(source: SelectiveProjectedPublicCoordinateSource) -> String {
    match source {
        SelectiveProjectedPublicCoordinateSource::ConstantOne => ".constantOne".to_owned(),
        SelectiveProjectedPublicCoordinateSource::SourceField(column) => format!(".sourceField {column}"),
        SelectiveProjectedPublicCoordinateSource::FixedZero => ".fixedZero".to_owned(),
    }
}

fn port_is(port: &[SelectiveProjectedTerm], expected: &[(usize, F)]) -> bool {
    port.len() == expected.len()
        && port
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .all(|(term, (column, coefficient))| term.column() == column && term.coefficient() == coefficient)
}

fn assert_selector_domain_row(row: &SelectiveProjectedRowArtifact, selector: usize) {
    assert_eq!(row.family(), SelectiveEmittedRowFamily::SelectorDomain);
    assert_eq!(row.arm(), None);
    assert!(row
        .ports()
        .iter()
        .all(|port| port.geometric_runs().is_empty()));
    assert!(port_is(row.ports()[0].explicit(), &[(selector, F::ONE)]));
    assert!(port_is(row.ports()[1].explicit(), &[(0, F::ONE)]));
    assert!(row.ports()[2..]
        .iter()
        .all(|port| port.explicit().is_empty()));
}

fn assert_one_hot_row(row: &SelectiveProjectedRowArtifact) {
    assert_eq!(row.family(), SelectiveEmittedRowFamily::OneHot);
    assert_eq!(row.arm(), None);
    assert!(row
        .ports()
        .iter()
        .all(|port| port.geometric_runs().is_empty()));
    assert!(row.ports()[0].explicit().is_empty());
    assert!(port_is(row.ports()[1].explicit(), &[(0, F::ONE)]));
    assert!(row.ports()[2].explicit().is_empty());
    assert!(row.ports()[3].explicit().is_empty());
    assert!(port_is(
        row.ports()[4].explicit(),
        &[
            (0, -F::ONE),
            (SELECTOR_COLUMNS[0], F::ONE),
            (SELECTOR_COLUMNS[1], F::ONE),
            (SELECTOR_COLUMNS[2], F::ONE),
        ],
    ));
    assert!(row.ports()[5..]
        .iter()
        .all(|port| port.explicit().is_empty()));
}

fn assert_active_profile(audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) {
    let projected = audit.projected_rows();
    assert_eq!(D, 54, "production packing lane count");
    assert_eq!(PUBLIC_COORDINATES, PACKED_BLOCKS * D, "complete block/lane carrier");
    assert_eq!(projected.selector_columns(), SELECTOR_COLUMNS);
    assert_eq!(projected.public_coordinates().len(), PUBLIC_COORDINATES);
    for (column, coordinate) in projected.public_coordinates().iter().copied().enumerate() {
        assert_eq!(coordinate.column(), column, "public-write address order");
        assert_eq!(
            coordinate.source(),
            expected_public_source(column),
            "public-write owner"
        );
    }

    let selector_rows = projected.selector_domain_row_artifacts();
    assert_eq!(selector_rows.len(), SELECTOR_COLUMNS.len());
    for (row, selector) in selector_rows.iter().zip(SELECTOR_COLUMNS) {
        assert_eq!(row.rows(), projected.rows());
        assert_eq!(row.columns(), projected.columns());
        assert_selector_domain_row(row, selector);
    }
    assert!(selector_rows
        .windows(2)
        .all(|rows| { rows[0].emitted_row() < rows[1].emitted_row() && rows[0].run_index() == rows[1].run_index() }));

    let one_hot = projected.one_hot_row_artifact();
    assert_eq!(one_hot.rows(), projected.rows());
    assert_eq!(one_hot.columns(), projected.columns());
    assert_ne!(one_hot.run_index(), selector_rows[0].run_index());
    assert!(one_hot.emitted_row() > selector_rows[selector_rows.len() - 1].emitted_row());
    assert_one_hot_row(one_hot);
}

fn packed_coordinate_shard(
    audit: &R1csIvcBlockLaneNcSelectiveRowsAudit,
    chunk_index: usize,
    start: usize,
    stop: usize,
) -> GeneratedLeanFile {
    assert!(start < stop && stop <= PUBLIC_COORDINATES && stop - start <= COORDINATES_PER_SHARD);
    let namespace = format!("{NAMESPACE_ROOT}.ActivePins.PackedCoordinates.Chunk{chunk_index}");
    let mut contents = generated_header("at most 128 exact active public-write addresses and block/lane owners");
    writeln!(contents, "import {IMPORT_ROOT}\n").expect("render packed-coordinate import");
    writeln!(contents, "namespace {namespace}\n").expect("render packed-coordinate namespace");
    contents.push_str("def values : List RawPackedPublicCoordinate := [\n");
    for (offset, coordinate) in audit.projected_rows().public_coordinates()[start..stop]
        .iter()
        .copied()
        .enumerate()
    {
        let column = coordinate.column();
        if offset != 0 {
            contents.push_str(",\n");
        }
        write!(
            contents,
            "  {{ schemaVersion := 1, column := {column}, block := {}, lane := {}, source := {} }}",
            column / D,
            column % D,
            lean_public_source(coordinate.source()),
        )
        .expect("render packed public coordinate");
    }
    writeln!(contents, "\n]\n\nend {namespace}").expect("render packed-coordinate namespace end");
    assert!(contents.lines().count() < 1_500, "packed-coordinate shard line limit");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/ActivePins/PackedCoordinates/Chunk{chunk_index}.lean"),
        contents,
    }
}

fn packed_coordinate_root(shard_count: usize) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.ActivePins.PackedCoordinates");
    let mut contents = generated_header("the ordered 128/128/14 active public-write partition");
    for index in 0..shard_count {
        writeln!(contents, "import {namespace}.Chunk{index}").expect("render packed-coordinate shard import");
    }
    writeln!(contents, "\nnamespace {namespace}\n").expect("render packed-coordinate root namespace");
    contents.push_str("def values : List RawPackedPublicCoordinate :=\n");
    for index in 0..shard_count {
        let continuation = if index + 1 == shard_count { "" } else { " ++" };
        writeln!(contents, "  Chunk{index}.values{continuation}").expect("render packed-coordinate root");
    }
    writeln!(contents, "\nend {namespace}").expect("render packed-coordinate root namespace end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/ActivePins/PackedCoordinates.lean"),
        contents,
    }
}

fn metadata(audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.ActivePins");
    let projected = audit.projected_rows();
    let (source_rows, source_columns) = source_shape(audit);
    let selector_rows = projected.selector_domain_row_artifacts();
    let mut contents =
        generated_header("the active constant/selector pins and their exact selector-domain and one-hot row owners");
    writeln!(contents, "import {IMPORT_ROOT}").expect("render active-pin schema import");
    writeln!(contents, "import {namespace}.PackedCoordinates\n").expect("render public-coordinate import");
    writeln!(contents, "namespace {namespace}\n").expect("render active-pin namespace");
    contents.push_str("def raw : RawActivePins :=\n");
    writeln!(contents, "  {{ schemaVersion := 1").expect("render active-pin schema");
    writeln!(contents, "    sourceRows := {source_rows}").expect("render source rows");
    writeln!(contents, "    sourceColumns := {source_columns}").expect("render source columns");
    writeln!(contents, "    finalRows := {}", projected.rows()).expect("render final rows");
    writeln!(contents, "    finalColumns := {}", projected.columns()).expect("render final columns");
    contents.push_str("    constantOneColumn := 0\n");
    contents.push_str("    constantOneValue := 1\n");
    writeln!(contents, "    selectorColumns := {}", lean_nat_list(SELECTOR_COLUMNS)).expect("render selector columns");
    writeln!(
        contents,
        "    recursiveSelectorValues := {}",
        lean_nat_list(RECURSIVE_SELECTOR_VALUES)
    )
    .expect("render recursive selector values");
    writeln!(contents, "    packedLaneCount := {D}").expect("render packed lane count");
    writeln!(contents, "    packedBlockCount := {PACKED_BLOCKS}").expect("render packed block count");
    writeln!(contents, "    publicCoordinateCount := {PUBLIC_COORDINATES}").expect("render public width");
    contents.push_str("    selectorDomainRows := [\n");
    for (index, row) in selector_rows.iter().enumerate() {
        if index != 0 {
            contents.push_str(",\n");
        }
        render_emitted_row(&mut contents, row);
    }
    contents.push_str("    ]\n    oneHotRow :=\n");
    render_emitted_row(&mut contents, projected.one_hot_row_artifact());
    contents.push_str("  }\n\n");
    contents.push_str("def packedCoordinates : List RawPackedPublicCoordinate := PackedCoordinates.values\n");
    writeln!(contents, "\nend {namespace}").expect("render active-pin namespace end");
    assert!(contents.lines().count() < 1_500, "active-pin metadata file line limit");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/ActivePins.lean"),
        contents,
    }
}

pub(super) fn render(audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) -> Vec<GeneratedLeanFile> {
    assert_active_profile(audit);
    let ranges = [(0, 128), (128, 256), (256, 270)];
    assert_eq!(ranges.map(|(start, stop)| stop - start), [128, 128, 14]);
    let mut files = ranges
        .into_iter()
        .enumerate()
        .map(|(index, (start, stop))| packed_coordinate_shard(audit, index, start, stop))
        .collect::<Vec<_>>();
    files.push(packed_coordinate_root(files.len()));
    files.push(metadata(audit));
    files
}
