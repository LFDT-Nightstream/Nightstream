//! Compact full-`Z` decoder layout for the stabilized production relation.
//!
//! Owns: the generated production width, exact `54 × 265,535` packed
//! witness geometry, fourteen-child multiplicity, and the 54-live/10-zero
//! Boolean-lane classification.
//!
//! Does not own: witness values, commitment binding, combined-NC acceptance,
//! delayed-projection rows, transcript scheduling, or row removal.
//!
//! Emits constraints: none; this file renders direct-dataflow evidence.

use std::fmt::Write as _;

use neo_ajtai::{has_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcRawRunningAssignmentAudit;
use neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedRowsAudit;
use neo_fold_clean::{config, paper::params::Params, CcsInstance};
use neo_math::{D, F, K};
use neo_reductions::common::decode_superneo_coeffs_from_witness_mat;
use p3_field::PrimeCharacteristicRing;

use super::GeneratedLeanFile;

const GENERATED_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiCcsNc/DelayedProjection/PackedWitnessDecoder/Generated/Layout.lean";
const IMPORT_ROOT: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder.Schema";
const NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder.Generated.Layout";
const SCHEMA_VERSION: usize = 2;
const PUBLIC_LOGICAL_COLUMNS_PER_CHILD: usize = 270;
const COMMITMENT_PROBE_BLOCKS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LaneSourceRecord {
    boolean_lane: usize,
    witness_lane: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    relation_rows: usize,
    logical_width: usize,
    child_count: usize,
    matrix_rows: usize,
    matrix_columns: usize,
    boolean_lane_count: usize,
    lane_sources: Vec<LaneSourceRecord>,
    fixture_commitment_width: usize,
    fixture_commitment_data_length: usize,
    production_commitment_width: usize,
    production_commitment_data_length: usize,
    commitment_probe_blocks: usize,
    commitment_probe_columns: Vec<usize>,
}

fn validate_runtime_decoder_convention() {
    const PROBE_BLOCKS: usize = 2;
    let logical_width = PROBE_BLOCKS * D;
    let mut matrix = Mat::zero(D, PROBE_BLOCKS, F::ZERO);
    for block in 0..PROBE_BLOCKS {
        for lane in 0..D {
            matrix[(lane, block)] = F::from_u64((block * D + lane + 1) as u64);
        }
    }
    let decoded = decode_superneo_coeffs_from_witness_mat(&matrix, logical_width)
        .expect("two-block packed witness decoder probe");
    assert_eq!(decoded.len(), logical_width);
    for logical_column in 0..logical_width {
        let block = logical_column / D;
        let lane = logical_column % D;
        assert_eq!(decoded[logical_column], K::from(matrix[(lane, block)]));
    }
}

/// Exercise the authoritative fresh-instance path, not merely a parallel
/// reimplementation of its index arithmetic.  One-hot probes distinguish
/// every cell of two complete Phi81 blocks while keeping setup and commitment
/// work tiny.  The production-width theorem remains symbolic/generated.
fn validate_runtime_packing_and_commitment_convention(params: &Params) -> usize {
    let logical_width = COMMITMENT_PROBE_BLOCKS * D;
    let structure = CcsStructure::new(
        vec![Mat::zero(1, logical_width, F::ZERO)],
        SparsePoly::new(1, Vec::new()),
    )
    .expect("two-block packed-witness probe structure");
    if !has_global_pp_for_dims(D, COMMITMENT_PROBE_BLOCKS) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x5057_4445_434f_4445_u64.to_le_bytes());
        set_global_pp_seeded(D, params.kappa() as usize, COMMITMENT_PROBE_BLOCKS, seed)
            .expect("two-block packed-witness probe Ajtai setup");
    }
    let log = AjtaiSModule::from_global_for_dims(D, COMMITMENT_PROBE_BLOCKS)
        .expect("two-block packed-witness probe Ajtai module");
    assert_eq!(log.dims(), (D, COMMITMENT_PROBE_BLOCKS));
    let commitment_width = log.kappa();
    let pp = log
        .verification_pp()
        .expect("materialize the tiny two-block verifier key");
    assert_eq!((pp.d, pp.kappa, pp.m), (D, commitment_width, COMMITMENT_PROBE_BLOCKS));
    assert_eq!(pp.m_rows.len(), commitment_width);
    assert!(pp
        .m_rows
        .iter()
        .all(|row| row.len() == COMMITMENT_PROBE_BLOCKS));

    for logical_column in 0..logical_width {
        let mut assignment = vec![F::ZERO; logical_width];
        assignment[logical_column] = F::ONE;
        let instance = CcsInstance::from_low_norm_assignment(params, &log, &structure, &assignment, 0)
            .expect("one-hot packed-witness probe instance");
        let expected_block = logical_column / D;
        let expected_lane = logical_column % D;
        for block in 0..COMMITMENT_PROBE_BLOCKS {
            for lane in 0..D {
                let expected = if (lane, block) == (expected_lane, expected_block) {
                    F::ONE
                } else {
                    F::ZERO
                };
                assert_eq!(instance.witness.Z[(lane, block)], expected);
            }
        }
        assert_eq!(log.commit(&instance.witness.Z), instance.claim.c);
        assert_eq!((instance.claim.c.d, instance.claim.c.kappa), (D, commitment_width));
        assert_eq!(instance.claim.c.data.len(), D * commitment_width);
        for commitment_row in 0..commitment_width {
            for lane in 0..D {
                assert_eq!(
                    instance.claim.c.data[commitment_row * D + lane],
                    instance.claim.c.col(commitment_row)[lane]
                );
            }
        }
        assert_eq!(
            decode_superneo_coeffs_from_witness_mat(&instance.witness.Z, logical_width)
                .expect("decode one-hot constructor witness"),
            assignment.into_iter().map(K::from).collect::<Vec<_>>()
        );
    }
    commitment_width
}

fn layout(
    projected: &SelectiveProjectedRowsAudit,
    raw_running: &[R1csIvcRawRunningAssignmentAudit],
    params: &Params,
) -> Result<Layout, String> {
    validate_runtime_decoder_convention();
    let probe_commitment_width = validate_runtime_packing_and_commitment_convention(params);

    let logical_width = projected.columns();
    if logical_width == 0 || !logical_width.is_multiple_of(D) {
        return Err(format!(
            "production full-Z width {logical_width} is not a nonzero whole number of {D}-lane blocks"
        ));
    }
    if raw_running.is_empty()
        || !raw_running
            .len()
            .is_multiple_of(PUBLIC_LOGICAL_COLUMNS_PER_CHILD)
    {
        return Err("raw-running audit does not determine a whole child family".into());
    }
    let child_count = raw_running.len() / PUBLIC_LOGICAL_COLUMNS_PER_CHILD;
    if raw_running.iter().enumerate().any(|(index, record)| {
        record.child() != index / PUBLIC_LOGICAL_COLUMNS_PER_CHILD
            || record.logical_column() != index % PUBLIC_LOGICAL_COLUMNS_PER_CHILD
    }) {
        return Err("raw-running audit is not in exact child-major order".into());
    }

    let boolean_lane_count = D.next_power_of_two();
    let lane_sources = (0..boolean_lane_count)
        .map(|boolean_lane| LaneSourceRecord {
            boolean_lane,
            witness_lane: (boolean_lane < D).then_some(boolean_lane),
        })
        .collect::<Vec<_>>();
    let result = Layout {
        relation_rows: projected.rows(),
        logical_width,
        child_count,
        matrix_rows: D,
        matrix_columns: logical_width.div_ceil(D),
        boolean_lane_count,
        lane_sources,
        fixture_commitment_width: params.kappa() as usize,
        fixture_commitment_data_length: params.kappa() as usize * D,
        production_commitment_width: config::KAPPA as usize,
        production_commitment_data_length: config::KAPPA as usize * D,
        commitment_probe_blocks: COMMITMENT_PROBE_BLOCKS,
        commitment_probe_columns: (0..COMMITMENT_PROBE_BLOCKS * D).collect(),
    };
    if probe_commitment_width != result.fixture_commitment_width {
        return Err(format!(
            "two-block commitment probe width {probe_commitment_width} differs from fixture width {}",
            result.fixture_commitment_width
        ));
    }
    validate(&result)?;
    Ok(result)
}

fn validate(layout: &Layout) -> Result<(), String> {
    if layout.child_count != 14 {
        return Err(format!(
            "production full-Z child count is {}, expected 14",
            layout.child_count
        ));
    }
    if layout.matrix_rows != D || layout.matrix_columns.checked_mul(layout.matrix_rows) != Some(layout.logical_width) {
        return Err("full-Z matrix geometry does not cover the exact logical width".into());
    }
    if layout.boolean_lane_count != D.next_power_of_two() || layout.lane_sources.len() != layout.boolean_lane_count {
        return Err("full-Z Boolean lane domain is not the exact next-power-of-two completion".into());
    }
    for (index, record) in layout.lane_sources.iter().copied().enumerate() {
        let expected = (index < D).then_some(index);
        if record.boolean_lane != index || record.witness_lane != expected {
            return Err(format!(
                "full-Z Boolean lane record {index} is {record:?}, expected witness lane {expected:?}"
            ));
        }
    }
    if layout.fixture_commitment_width == 0
        || layout.fixture_commitment_data_length != layout.fixture_commitment_width * layout.matrix_rows
        || layout.production_commitment_width != config::KAPPA as usize
        || layout.production_commitment_data_length != layout.production_commitment_width * layout.matrix_rows
        || layout.commitment_probe_blocks != COMMITMENT_PROBE_BLOCKS
        || layout.commitment_probe_columns.len() != COMMITMENT_PROBE_BLOCKS * D
        || layout
            .commitment_probe_columns
            .iter()
            .copied()
            .enumerate()
            .any(|(index, column)| index != column)
    {
        return Err("packed-witness commitment probe layout is not exact".into());
    }
    Ok(())
}

pub(super) fn render(
    projected: &SelectiveProjectedRowsAudit,
    raw_running: &[R1csIvcRawRunningAssignmentAudit],
    params: &Params,
) -> GeneratedLeanFile {
    let layout = layout(projected, raw_running, params).expect("exact production full-Z decoder layout");
    let mut contents = String::new();
    writeln!(
        contents,
        "import {IMPORT_ROOT}

/-!
Generated file: exact production full-`Z` decoder layout; do not hand-edit.

The artifact records the stabilized relation width and the same column-major
`(lane, block)` convention exercised through Rust's actual
`decode_superneo_coeffs_from_witness_mat` implementation. It is compact:
the full logical-coordinate map is represented by one affine block stride
and 64 proof-free lane records, not by enumerating every witness cell.

Owns: production packed-witness dimensions and live/virtual lane provenance.

The bounded constructor probe additionally records every one-hot logical
column exercised through `CcsInstance::from_low_norm_assignment`, the actual
Ajtai verifier-key dimensions, commitment recomputation, and column-major
commitment-data indexing.

Does not own: witness values, commitment binding, NC acceptance, generated
delayed-projection rows, transcript scheduling, or row-removal permission.

Emits constraints: none; generated direct-dataflow evidence only.

| Stable stage path | Obligation | Authority |
|---|---|---|
| `pi_ccs_nc.delayed_projection.full_z_decoder.generated` | exact full-width block/lane decoder and 54+10 lane partition | computed artifact |
-/

namespace {NAMESPACE}

def schemaVersion : Nat := {SCHEMA_VERSION}
def relationRows : Nat := {}
def logicalWidth : Nat := {}
def childCount : Nat := {}
def matrixRows : Nat := {}
def matrixColumns : Nat := {}
def booleanLaneCount : Nat := {}
def fixtureCommitmentWidth : Nat := {}
def fixtureCommitmentDataLength : Nat := {}
def productionCommitmentWidth : Nat := {}
def productionCommitmentDataLength : Nat := {}
def commitmentProbeBlocks : Nat := {}
def laneSources : List LaneSourceRecord := [",
        layout.relation_rows,
        layout.logical_width,
        layout.child_count,
        layout.matrix_rows,
        layout.matrix_columns,
        layout.boolean_lane_count,
        layout.fixture_commitment_width,
        layout.fixture_commitment_data_length,
        layout.production_commitment_width,
        layout.production_commitment_data_length,
        layout.commitment_probe_blocks,
    )
    .expect("render full-Z layout header");
    for (index, record) in layout.lane_sources.iter().copied().enumerate() {
        let separator = if index == 0 { "  " } else { ", " };
        let witness_lane = match record.witness_lane {
            Some(lane) => format!("some {lane}"),
            None => "none".to_owned(),
        };
        writeln!(
            contents,
            "{separator}{{ booleanLane := {}, witnessLane := {witness_lane} }}",
            record.boolean_lane,
        )
        .expect("render full-Z lane source");
    }
    writeln!(contents, "]\ndef commitmentProbeColumns : List Nat := [").expect("render commitment probe header");
    for (index, column) in layout.commitment_probe_columns.iter().copied().enumerate() {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(contents, "{separator}{column}").expect("render commitment probe column");
    }
    writeln!(contents, "]\n\nend {NAMESPACE}").expect("render full-Z layout footer");
    assert!(
        contents.lines().count() < 1_500,
        "generated full-Z decoder layout exceeds the repository line limit"
    );
    GeneratedLeanFile {
        relative_path: GENERATED_PATH.to_owned(),
        contents,
    }
}

#[test]
fn full_z_layout_validation_rejects_dimension_and_lane_mutations() {
    let mut layout = Layout {
        relation_rows: 1,
        logical_width: 2 * D,
        child_count: 14,
        matrix_rows: D,
        matrix_columns: 2,
        boolean_lane_count: D.next_power_of_two(),
        lane_sources: (0..D.next_power_of_two())
            .map(|boolean_lane| LaneSourceRecord {
                boolean_lane,
                witness_lane: (boolean_lane < D).then_some(boolean_lane),
            })
            .collect(),
        fixture_commitment_width: 4,
        fixture_commitment_data_length: 4 * D,
        production_commitment_width: config::KAPPA as usize,
        production_commitment_data_length: config::KAPPA as usize * D,
        commitment_probe_blocks: COMMITMENT_PROBE_BLOCKS,
        commitment_probe_columns: (0..COMMITMENT_PROBE_BLOCKS * D).collect(),
    };
    assert!(validate(&layout).is_ok());

    layout.matrix_columns += 1;
    assert!(validate(&layout).is_err());
    layout.matrix_columns -= 1;

    layout.lane_sources[D].witness_lane = Some(D - 1);
    assert!(validate(&layout).is_err());
    layout.lane_sources[D].witness_lane = None;

    layout.lane_sources[0].boolean_lane = 1;
    assert!(validate(&layout).is_err());
    layout.lane_sources[0].boolean_lane = 0;

    layout.fixture_commitment_data_length += 1;
    assert!(validate(&layout).is_err());
}
