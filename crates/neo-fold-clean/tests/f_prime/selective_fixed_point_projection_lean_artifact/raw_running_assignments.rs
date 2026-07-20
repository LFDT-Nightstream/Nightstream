//! Exact raw-running assignment decoder exported from the stabilized steady
//! recursive arm.
//!
//! Owns: child/logical-coordinate order, normalized source-arm columns, and
//! the corresponding direct one-coordinate slots in the final selective
//! assignment.
//!
//! Does not own: raw-child semantic authority, delayed-projection algebra,
//! transcript sampling, commitment binding, or permission to remove rows.
//!
//! Emits constraints: none; this file renders checked artifact data.
//!
//! | Stable stage path | Obligation | Authority |
//! |---|---|---|
//! | `pi_ccs_nc.delayed_projection.raw_running_decoder` | Exact physical-column provenance for `14 × 270` raw running coordinates | computed artifact |

use std::collections::BTreeSet;
use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcRawRunningAssignmentAudit;
use neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedRowsAudit;

use super::GeneratedLeanFile;

const GENERATED_ROOT: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiCcsNc/DelayedProjection/RawRunningDecoder/Generated";
const IMPORT_ROOT: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Schema";
const NAMESPACE_ROOT: &str = "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated";
const RECORD_NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder";
const SCHEMA_VERSION: usize = 1;
const SOURCE_ARM: usize = 2;
const CHILD_COUNT: usize = 14;
const LOGICAL_COLUMN_COUNT: usize = 270;
const RECORD_COUNT: usize = CHILD_COUNT * LOGICAL_COLUMN_COUNT;
const CHUNK_SIZE: usize = 252;
const CHUNK_COUNT: usize = 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RawRecord {
    child: usize,
    logical_column: usize,
    source_arm_column: usize,
    final_column: usize,
}

fn records(audit: &[R1csIvcRawRunningAssignmentAudit]) -> Vec<RawRecord> {
    audit
        .iter()
        .copied()
        .map(|entry| RawRecord {
            child: entry.child(),
            logical_column: entry.logical_column(),
            source_arm_column: entry.source_column(),
            final_column: entry.final_column(),
        })
        .collect()
}

fn validate(records: &[RawRecord], final_column_count: usize) -> Result<(), String> {
    if records.len() != RECORD_COUNT {
        return Err(format!(
            "raw running decoder has {} records, expected {RECORD_COUNT}",
            records.len()
        ));
    }
    for (index, record) in records.iter().copied().enumerate() {
        let expected_child = index / LOGICAL_COLUMN_COUNT;
        let expected_logical_column = index % LOGICAL_COLUMN_COUNT;
        if record.child != expected_child || record.logical_column != expected_logical_column {
            return Err(format!(
                "raw running decoder record {index} is ({}, {}), expected ({expected_child}, {expected_logical_column})",
                record.child, record.logical_column
            ));
        }
        if record.final_column >= final_column_count {
            return Err(format!(
                "raw running decoder record {index} final column {} exceeds final width {final_column_count}",
                record.final_column
            ));
        }
    }
    let source_columns = records
        .iter()
        .map(|record| record.source_arm_column)
        .collect::<BTreeSet<_>>();
    if source_columns.len() != RECORD_COUNT {
        return Err("raw running decoder source-arm columns are not uniquely owned".into());
    }
    let final_columns = records
        .iter()
        .map(|record| record.final_column)
        .collect::<BTreeSet<_>>();
    if final_columns.len() != RECORD_COUNT {
        return Err("raw running decoder final selective columns are not uniquely owned".into());
    }
    Ok(())
}

pub(super) fn render(
    projected: &SelectiveProjectedRowsAudit,
    audit: &[R1csIvcRawRunningAssignmentAudit],
) -> Vec<GeneratedLeanFile> {
    let records = records(audit);
    validate(&records, projected.columns()).expect("exact raw running-assignment decoder");
    let chunks = records.chunks(CHUNK_SIZE).collect::<Vec<_>>();
    assert_eq!(chunks.len(), CHUNK_COUNT, "exact raw decoder chunk count");
    assert!(
        chunks.iter().all(|chunk| chunk.len() == CHUNK_SIZE),
        "every raw decoder chunk has exactly 252 proof-free records"
    );

    chunks
        .into_iter()
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let mut contents = String::new();
            writeln!(
                contents,
                "import {IMPORT_ROOT}

/-!
Generated file: authoritative raw-running assignment decoder chunk; do not
hand-edit.

Each provenance record carries both the normalized source-arm column and its
final selective-assignment column. The generator fails closed unless the final
column is the exact direct, centered, width-one selective slot for the record's
actual
`running[child].x[(logicalColumn % 54) * x_cols + logicalColumn / 54]` wire.

This data does not establish delayed-projection acceptance, raw-child semantic
authority, commitment binding, or row-removal permission.

Owns: one exact 252-record raw-running physical-column provenance shard.

Does not own: assignment values, combined-NC acceptance, transcript scheduling,
commitment binding, or permission to remove rows.

Emits constraints: none; generated data only.

| Stable stage path | Obligation | Authority |
|---|---|---|
| `pi_ccs_nc.delayed_projection.raw_running_decoder.generated.chunk` | Exact generated coordinate-to-column records | computed artifact |
-/

namespace {NAMESPACE_ROOT}.Chunk{chunk_index}

open {RECORD_NAMESPACE}

def schemaVersion : Nat := {SCHEMA_VERSION}
def sourceArm : Nat := {SOURCE_ARM}
def childCount : Nat := {CHILD_COUNT}
def logicalColumnCount : Nat := {LOGICAL_COLUMN_COUNT}
def finalColumnCount : Nat := {}
def allocationRecords : List AllocationRecord := [",
                projected.columns(),
            )
            .expect("render raw decoder header");
            for (index, record) in chunk.iter().copied().enumerate() {
                let separator = if index == 0 { "  " } else { ", " };
                writeln!(
                    contents,
                    "{separator}{{ child := {}, logicalColumn := {}, sourceArmColumn := {}, finalColumn := {} }}",
                    record.child, record.logical_column, record.source_arm_column, record.final_column,
                )
                .expect("render raw decoder record");
            }
            writeln!(
                contents,
                "]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end {NAMESPACE_ROOT}.Chunk{chunk_index}"
            )
            .expect("render raw decoder footer");
            assert!(
                contents.lines().count() < 1_500,
                "generated raw decoder chunk exceeds the repository line limit"
            );
            GeneratedLeanFile {
                relative_path: format!("{GENERATED_ROOT}/Chunk{chunk_index}.lean"),
                contents,
            }
        })
        .collect()
}

#[test]
fn raw_running_assignment_validation_rejects_coordinate_and_ownership_mutations() {
    let mut records = (0..RECORD_COUNT)
        .map(|index| RawRecord {
            child: index / LOGICAL_COLUMN_COUNT,
            logical_column: index % LOGICAL_COLUMN_COUNT,
            source_arm_column: index,
            final_column: index,
        })
        .collect::<Vec<_>>();
    assert!(validate(&records, RECORD_COUNT).is_ok());

    records[0].logical_column = 1;
    assert!(validate(&records, RECORD_COUNT).is_err());
    records[0].logical_column = 0;

    records[1].source_arm_column = records[0].source_arm_column;
    assert!(validate(&records, RECORD_COUNT).is_err());
    records[1].source_arm_column = 1;

    records[1].final_column = records[0].final_column;
    assert!(validate(&records, RECORD_COUNT).is_err());
}
