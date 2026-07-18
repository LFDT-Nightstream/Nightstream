//! Exact PiCCS-output to PiRLC `y_zcol` consumer ownership.
//!
//! Owns: the cross-phase column identity from every typed active
//! `PiCcsOutput.YZcolLimb` source to the corresponding coefficient of the two
//! PiRLC input-evaluation families in the 13-matrix fixed-point arm.
//!
//! Does not own: PiCCS output truth, transcript challenge authority, the
//! projection quotient identity, the returned parent, encoded lowering,
//! security bounds, costs, or row removal.
//!
//! Emits constraints: no.
//!
//! Authority boundary: exact column identity proves dataflow only. It does
//! not prove that the PiCCS `y_zcol` values are derived from authoritative CCS
//! assignments; that remains a separately named Lean premise.
//!
//! | Protocol → phase → family | Mathematical obligation | Physical evidence | Lean owner |
//! |---|---|---|---|
//! | `nifs.pi_ccs.output_message_hashes.digest.preimage.y_zcol` | 15 × 54 × 2 typed source limbs | primary SIS input-column map | `ActiveSourceLayout.SourceRole.yZcolLimb` |
//! | `nifs.pi_rlc.verify.identities.y_zcol.evaluations.inputs.limb0` | source-major low-limb coefficients | 15 exact evaluation traces | `YZcolConsumer.decodedInputs` |
//! | `nifs.pi_rlc.verify.identities.y_zcol.evaluations.inputs.limb1` | source-major high-limb coefficients | 15 exact evaluation traces | `YZcolConsumer.decodedInputs` |
//! | cross-phase binding | producer column equals consumer column coordinate-for-coordinate | direct vector equality | `YZcolConsumer.ConsumerMatches` |

use std::collections::HashMap;
use std::ops::Range;

use neo_math::ring::D;

use super::{invalid, PiCcsOutputSisPhysicalAudit};
use crate::engine::r1cs_circuit::ring_action::PROJECTION_QUOTIENT_LEN;
use crate::engine::r1cs_circuit::PiRlcYZcolBoundaryAudit;
use crate::frontends::r1cs_f_prime::ivc::R1csIvcError;
use crate::frontends::r1cs_f_prime::SparseR1cs;
use crate::paper::reductions::pi_ccs_output_message::{FieldPath, KLimb, Profile};
use crate::paper::reductions::pi_rlc_circuit::stage;

mod certificate;
mod identity;
mod rows;
mod selective_rows;

pub use certificate::{
    PiRlcYZcolKMulAudit, PiRlcYZcolLinearCombinationAudit, PiRlcYZcolPolynomialEvaluationAudit,
    PiRlcYZcolProductFactorAudit, PiRlcYZcolProductIdentityAudit,
};
pub use identity::{PiRlcYZcolProjectionIdentityAudit, PiRlcYZcolProjectionLimbAudit, PiRlcYZcolProjectionSharedAudit};
pub use rows::PiRlcYZcolProjectionRowAudit;
pub use selective_rows::{
    PiRlcYZcolProjectionLeafRowMappingAudit, PiRlcYZcolProjectionLoweredFragmentAudit,
    PiRlcYZcolProjectionLoweringDisposition, PiRlcYZcolProjectionRowMappingAudit,
};

const LIMBS: usize = 2;

/// Fixed protocol scope attached to one active projection certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolProjectionProfileAudit {
    source_count: usize,
    matrix_count: usize,
    field_count: usize,
    source_arm_row_count: usize,
    source_arm_column_count: usize,
    lane_count: usize,
    limb_count: usize,
}

impl PiRlcYZcolProjectionProfileAudit {
    pub fn source_count(self) -> usize {
        self.source_count
    }

    pub fn matrix_count(self) -> usize {
        self.matrix_count
    }

    pub fn field_count(self) -> usize {
        self.field_count
    }

    pub fn source_arm_row_count(self) -> usize {
        self.source_arm_row_count
    }

    pub fn source_arm_column_count(self) -> usize {
        self.source_arm_column_count
    }

    pub fn lane_count(self) -> usize {
        self.lane_count
    }

    pub fn limb_count(self) -> usize {
        self.limb_count
    }
}

/// Serializer provenance for one PiCCS-produced `y_zcol` column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiCcsOutputYZcolProducerEntryAudit {
    field_index: usize,
    column: usize,
}

impl PiCcsOutputYZcolProducerEntryAudit {
    pub fn field_index(self) -> usize {
        self.field_index
    }

    pub fn column(self) -> usize {
        self.column
    }
}

/// One source-specific polynomial-evaluation consumer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsOutputYZcolProjectionInputAudit {
    source: usize,
    limb: usize,
    rows: Range<usize>,
    producer_entries: Vec<PiCcsOutputYZcolProducerEntryAudit>,
    producer_columns: Vec<usize>,
    coefficient_columns: Vec<usize>,
}

impl PiCcsOutputYZcolProjectionInputAudit {
    pub fn source(&self) -> usize {
        self.source
    }

    pub fn limb(&self) -> usize {
        self.limb
    }

    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }

    pub fn producer_columns(&self) -> &[usize] {
        &self.producer_columns
    }

    pub fn producer_entries(&self) -> &[PiCcsOutputYZcolProducerEntryAudit] {
        &self.producer_entries
    }

    pub fn coefficient_columns(&self) -> &[usize] {
        &self.coefficient_columns
    }

    pub fn consumer_columns(&self) -> &[usize] {
        &self.coefficient_columns
    }
}

/// Complete source-major consumer tree for both extension-field limbs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiCcsOutputYZcolProjectionAudit {
    profile: PiRlcYZcolProjectionProfileAudit,
    limbs: [Vec<PiCcsOutputYZcolProjectionInputAudit>; LIMBS],
    boundary: PiRlcYZcolBoundaryAudit,
    identity: PiRlcYZcolProjectionIdentityAudit,
    selective_rows: PiRlcYZcolProjectionRowMappingAudit,
}

impl PiCcsOutputYZcolProjectionAudit {
    pub fn profile(&self) -> PiRlcYZcolProjectionProfileAudit {
        self.profile
    }

    pub fn limb(&self, limb: usize) -> &[PiCcsOutputYZcolProjectionInputAudit] {
        &self.limbs[limb]
    }

    pub fn input_count(&self) -> usize {
        self.limbs.iter().map(Vec::len).sum()
    }

    pub fn coefficient_count(&self) -> usize {
        self.limbs
            .iter()
            .flat_map(|inputs| inputs.iter())
            .map(|input| input.coefficient_columns.len())
            .sum()
    }

    pub fn boundary(&self) -> &PiRlcYZcolBoundaryAudit {
        &self.boundary
    }

    pub fn identity(&self) -> &PiRlcYZcolProjectionIdentityAudit {
        &self.identity
    }

    /// Exact planned source-to-selective interval ownership for this
    /// cross-branch certificate. Semantic rewrite refinement remains open.
    pub fn selective_rows(&self) -> &PiRlcYZcolProjectionRowMappingAudit {
        &self.selective_rows
    }
}

pub(super) fn recover(
    arm: &SparseR1cs,
    profile: Profile,
    sis: &PiCcsOutputSisPhysicalAudit,
    row_mapping: &crate::frontends::r1cs_f_prime::SelectiveRowMappingAudit,
    arm_index: usize,
) -> Result<PiCcsOutputYZcolProjectionAudit, R1csIvcError> {
    if profile != Profile::active_f_prime() {
        return Err(invalid(format!(
            "PiRLC y_zcol consumer audit requires the active 15-source/13-matrix profile, found {profile:?}"
        )));
    }
    if sis.primary().input_columns().len() != profile.field_count() {
        return Err(invalid(
            "primary SIS input-column map differs from the active typed profile",
        ));
    }
    let [boundary] = arm.pi_rlc_y_zcol_boundary_audits() else {
        return Err(invalid(format!(
            "active arm contains {} PiRLC y_zcol semantic boundaries, expected one",
            arm.pi_rlc_y_zcol_boundary_audits().len()
        )));
    };
    for limb in 0..LIMBS {
        if boundary.parent_columns(limb).len() != D || boundary.quotient_columns(limb).len() != PROJECTION_QUOTIENT_LEN
        {
            return Err(invalid(format!(
                "PiRLC y_zcol limb {limb} boundary has {} parent and {} quotient columns, expected {D} and {PROJECTION_QUOTIENT_LEN}",
                boundary.parent_columns(limb).len(),
                boundary.quotient_columns(limb).len()
            )));
        }
    }

    let typed_columns = (0..profile.field_count())
        .map(|index| {
            let path = profile
                .decode(index)
                .expect("index is bounded by the profile field count");
            (path, (index, sis.primary().input_columns()[index]))
        })
        .collect::<HashMap<_, _>>();
    if typed_columns.len() != profile.field_count() {
        return Err(invalid("active PiCCS field paths are not one-to-one"));
    }

    let paths = [
        stage::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS_LIMB0,
        stage::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS_LIMB1,
    ];
    let mut limbs: [Vec<PiCcsOutputYZcolProjectionInputAudit>; LIMBS] = std::array::from_fn(|_| Vec::new());

    for (limb, stage_path) in paths.into_iter().enumerate() {
        let mut ranges = arm
            .physical_stage_ranges()
            .iter()
            .filter(|range| range.path() == stage_path && range.row_start() < range.row_end())
            .map(|range| range.rows())
            .collect::<Vec<_>>();
        ranges.sort_by_key(|range| range.start);
        if ranges.len() != profile.source_count() {
            return Err(invalid(format!(
                "`{stage_path}` has {} nonempty source intervals, expected {}",
                ranges.len(),
                profile.source_count()
            )));
        }

        for (source, rows) in ranges.into_iter().enumerate() {
            let matching = arm
                .polynomial_evaluation_traces()
                .iter()
                .filter(|trace| trace.row_start == rows.start && trace.row_end == rows.end)
                .collect::<Vec<_>>();
            let [evaluation] = matching.as_slice() else {
                return Err(invalid(format!(
                    "`{stage_path}` source {source} rows {rows:?} match {} polynomial evaluations",
                    matching.len()
                )));
            };
            if evaluation.coefficient_cols.len() != D || evaluation.power_cols.len() != D {
                return Err(invalid(format!(
                    "`{stage_path}` source {source} has {} coefficients and {} powers, expected {D} each",
                    evaluation.coefficient_cols.len(),
                    evaluation.power_cols.len()
                )));
            }

            let producer_entries = (0..D)
                .map(|lane| {
                    let path = FieldPath::YZcolLimb {
                        source,
                        lane,
                        limb: match limb {
                            0 => KLimb::C0,
                            1 => KLimb::C1,
                            _ => unreachable!("two-limb audit"),
                        },
                    };
                    typed_columns
                        .get(&path)
                        .copied()
                        .map(|(field_index, column)| PiCcsOutputYZcolProducerEntryAudit { field_index, column })
                        .ok_or_else(|| invalid(format!("active PiCCS source map omits {path:?}")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let expected = producer_entries
                .iter()
                .map(|entry| entry.column)
                .collect::<Vec<_>>();
            if evaluation.coefficient_cols != expected {
                return Err(invalid(format!(
                    "`{stage_path}` source {source} does not consume the exact typed PiCCS y_zcol columns"
                )));
            }

            limbs[limb].push(PiCcsOutputYZcolProjectionInputAudit {
                source,
                limb,
                rows,
                producer_entries,
                producer_columns: expected,
                coefficient_columns: evaluation.coefficient_cols.clone(),
            });
        }
    }

    let identity = identity::recover(arm, profile, &limbs, boundary)?;
    let selective_rows = selective_rows::recover(&identity, row_mapping, arm.physical_stage_ranges(), arm_index)?;
    Ok(PiCcsOutputYZcolProjectionAudit {
        profile: PiRlcYZcolProjectionProfileAudit {
            source_count: profile.source_count(),
            matrix_count: profile.matrix_count(),
            field_count: profile.field_count(),
            source_arm_row_count: arm.n,
            source_arm_column_count: arm.m,
            lane_count: profile.lane_count(),
            limb_count: LIMBS,
        },
        limbs,
        boundary: boundary.clone(),
        identity,
        selective_rows,
    })
}
