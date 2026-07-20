//! Exact fresh public-X source decoder from the stabilized steady recursive
//! arm.
//!
//! Owns: the coordinate order of `prior_link.fresh_public_inputs[0]`, its
//! normalized source-arm columns, and each column's exact selective-decoder
//! disposition.
//!
//! Does not own: the full private witness `Z`, per-coordinate binding-row
//! provenance, delayed-projection authority, commitment binding, or row
//! removal.
//!
//! Emits constraints: none; this file renders checked artifact data.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcYZcolSelectiveRowsAudit;
use neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedRowsAudit;

use super::GeneratedLeanFile;

const GENERATED_ROOT: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiCcsNc/DelayedProjection/FreshSourceDecoder/Generated";
const IMPORT_ROOT: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Schema";
const NAMESPACE_ROOT: &str = "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Generated";
const RECORD_NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder";
const SCHEMA_VERSION: usize = 1;
const SOURCE_ARM: usize = 2;
const SOURCE_COUNT: usize = 1;
const RECORD_COUNT: usize = 270;
const CHUNK_SIZES: [usize; 2] = [256, 14];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resolution {
    ConstantOne,
    Direct {
        start: usize,
        width: usize,
        centered: bool,
    },
    DecompositionAlias {
        source: usize,
        digit: usize,
        start: usize,
        centered: bool,
    },
    EqualityAlias {
        source: usize,
        start: usize,
        width: usize,
        centered: bool,
    },
    LinearDefinition,
    TraceEliminated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreshRecord {
    logical_column: usize,
    source_arm_column: usize,
    resolution: Resolution,
}

fn records(audit: &R1csIvcYZcolSelectiveRowsAudit) -> Result<Vec<FreshRecord>, String> {
    audit
        .fresh_source_assignments()
        .map(
            |(logical_column, source_arm_column, tag, source, digit, start, width, centered)| {
                let resolution = match (tag, source, digit, start, width, centered) {
                    (0, None, None, None, None, None) => Resolution::ConstantOne,
                    (1, None, None, Some(start), Some(width), Some(centered)) => {
                        Resolution::Direct { start, width, centered }
                    }
                    (2, Some(source), Some(digit), Some(start), Some(1), Some(centered)) => {
                        Resolution::DecompositionAlias {
                            source,
                            digit,
                            start,
                            centered,
                        }
                    }
                    (3, Some(source), None, Some(start), Some(width), Some(centered)) => Resolution::EqualityAlias {
                        source,
                        start,
                        width,
                        centered,
                    },
                    (4, None, None, None, None, None) => Resolution::LinearDefinition,
                    (5, None, None, None, None, None) => Resolution::TraceEliminated,
                    fields => {
                        return Err(format!(
                            "fresh public-X coordinate {logical_column} has malformed decoder fields {fields:?}"
                        ));
                    }
                };
                Ok(FreshRecord {
                    logical_column,
                    source_arm_column,
                    resolution,
                })
            },
        )
        .collect()
}

fn validate(records: &[FreshRecord], final_column_count: usize) -> Result<(), String> {
    if records.len() != RECORD_COUNT {
        return Err(format!(
            "fresh public-X decoder has {} records, expected {RECORD_COUNT}",
            records.len()
        ));
    }
    let mut source_columns = BTreeSet::new();
    for (expected_coordinate, record) in records.iter().copied().enumerate() {
        if record.logical_column != expected_coordinate {
            return Err(format!(
                "fresh public-X record {expected_coordinate} names coordinate {}",
                record.logical_column
            ));
        }
        if !source_columns.insert(record.source_arm_column) {
            return Err(format!(
                "fresh public-X coordinate {expected_coordinate} repeats source-arm column {}",
                record.source_arm_column
            ));
        }
        let final_range = match record.resolution {
            Resolution::ConstantOne => {
                if record.source_arm_column != 0 {
                    return Err(format!(
                        "fresh public-X coordinate {expected_coordinate} claims constant-one resolution for source column {}",
                        record.source_arm_column
                    ));
                }
                continue;
            }
            Resolution::LinearDefinition | Resolution::TraceEliminated => continue,
            Resolution::Direct { start, width, .. } | Resolution::EqualityAlias { start, width, .. } => {
                if width == 0 {
                    return Err(format!(
                        "fresh public-X coordinate {expected_coordinate} has a zero-width decoder"
                    ));
                }
                (start, width)
            }
            Resolution::DecompositionAlias { start, .. } => (start, 1),
        };
        if let Some(end) = final_range.0.checked_add(final_range.1) {
            if end > final_column_count {
                return Err(format!(
                    "fresh public-X coordinate {expected_coordinate} final range {}..{end} exceeds width {final_column_count}",
                    final_range.0
                ));
            }
        } else {
            return Err(format!(
                "fresh public-X coordinate {expected_coordinate} final range overflows"
            ));
        }
    }
    Ok(())
}

fn resolution_syntax(resolution: Resolution) -> String {
    match resolution {
        Resolution::ConstantOne => ".constantOne".into(),
        Resolution::Direct { start, width, centered } => format!(".direct {start} {width} {centered}"),
        Resolution::DecompositionAlias {
            source,
            digit,
            start,
            centered,
        } => format!(".decompositionAlias {source} {digit} {start} {centered}"),
        Resolution::EqualityAlias {
            source,
            start,
            width,
            centered,
        } => format!(".equalityAlias {source} {start} {width} {centered}"),
        Resolution::LinearDefinition => ".linearDefinition".into(),
        Resolution::TraceEliminated => ".traceEliminated".into(),
    }
}

pub(super) fn render(
    projected: &SelectiveProjectedRowsAudit,
    audit: &R1csIvcYZcolSelectiveRowsAudit,
) -> Vec<GeneratedLeanFile> {
    let records = records(audit).expect("well-formed fresh public-X decoder fields");
    validate(&records, projected.columns()).expect("exact fresh public-X decoder");
    assert_eq!(
        CHUNK_SIZES.into_iter().sum::<usize>(),
        records.len(),
        "fresh public-X decoder shard sizes cover exactly 270 records"
    );

    let mut offset = 0;
    CHUNK_SIZES
        .into_iter()
        .enumerate()
        .map(|(chunk_index, chunk_size)| {
            let end = offset + chunk_size;
            let chunk = &records[offset..end];
            offset = end;
            let mut contents = String::new();
            writeln!(
                contents,
                "import {IMPORT_ROOT}

/-!
Generated file: exact fresh public-X source decoder chunk; do not hand-edit.

The records describe only `prior_link.fresh_public_inputs[0]`, coordinates
0 through 269. This is the public-X source prefix consumed by the recursive
step, not the full private witness `Z` and not commitment authority.

The current Rust wire surface does not identify the exact binding row owned by
each coordinate. Consequently this artifact records normalized column and
selective-decoder provenance only; the row-level prior-link bridge remains
open.

Owns: one exact {chunk_size}-record proof-free decoder shard.

Does not own: source values, full-witness coordinates, per-coordinate binding
rows, commitment binding, or permission to remove constraints.

Emits constraints: none; generated certificate data only.

| Stage path | Mathematical obligation | Authority class | Artifact owner |
|---|---|---|---|
| `pi_ccs.nc.fresh_x.generated.chunk{chunk_index}` | exact ordered source column and fail-closed selective disposition | generated/checked | `fresh_source.rs` |
-/

namespace {NAMESPACE_ROOT}.Chunk{chunk_index}

open {RECORD_NAMESPACE}

def schemaVersion : Nat := {SCHEMA_VERSION}
def sourceArm : Nat := {SOURCE_ARM}
def sourceCount : Nat := {SOURCE_COUNT}
def logicalColumnCount : Nat := {RECORD_COUNT}
def finalColumnCount : Nat := {}
def records : List SourceColumnRecord := [",
                projected.columns(),
            )
            .expect("render fresh decoder header");
            for (index, record) in chunk.iter().copied().enumerate() {
                let separator = if index == 0 { "  " } else { ", " };
                writeln!(
                    contents,
                    "{separator}{{ logicalColumn := {}, sourceArmColumn := {}, resolution := {} }}",
                    record.logical_column,
                    record.source_arm_column,
                    resolution_syntax(record.resolution),
                )
                .expect("render fresh decoder record");
            }
            writeln!(contents, "]\n\nend {NAMESPACE_ROOT}.Chunk{chunk_index}").expect("render fresh decoder footer");
            assert!(
                contents.lines().count() < 1_500,
                "generated fresh decoder chunk exceeds the repository line limit"
            );
            GeneratedLeanFile {
                relative_path: format!("{GENERATED_ROOT}/Chunk{chunk_index}.lean"),
                contents,
            }
        })
        .collect()
}

#[test]
fn fresh_source_validation_rejects_coordinate_ownership_and_range_mutations() {
    let mut records = (0..RECORD_COUNT)
        .map(|logical_column| FreshRecord {
            logical_column,
            source_arm_column: logical_column + 1,
            resolution: Resolution::Direct {
                start: logical_column,
                width: 1,
                centered: false,
            },
        })
        .collect::<Vec<_>>();
    assert!(validate(&records, RECORD_COUNT).is_ok());

    records[0].logical_column = 1;
    assert!(validate(&records, RECORD_COUNT).is_err());
    records[0].logical_column = 0;

    records[1].source_arm_column = records[0].source_arm_column;
    assert!(validate(&records, RECORD_COUNT).is_err());
    records[1].source_arm_column = 2;

    records[2].resolution = Resolution::Direct {
        start: RECORD_COUNT,
        width: 1,
        centered: false,
    };
    assert!(validate(&records, RECORD_COUNT).is_err());
}
