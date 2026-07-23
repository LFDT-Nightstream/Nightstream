//! Authoritative raw-witness projection for the delayed old-block audit.
//!
//! Owns: capture and fail-closed replay of the fourteen ordered
//! `RunningInstance.witnesses` matrices at the pending 19-coordinate block
//! point, including 54 active and ten verifier-computed zero lanes.
//!
//! Does not own: child `CeClaim.y_zcol` sidecars, digest authority, transcript
//! sampling, commitment binding, or R1CS rows for the projected lane values.
//!
//! Emits constraints: no; this is native execution evidence.
//!
//! | Stable stage path | Obligation | Authority class |
//! |---|---|---|
//! | `f_prime.pi_ccs_nc.raw_old_block.running_witnesses` | Project the exact ordered full witness matrices at `pending.old_block` | direct dataflow |
//! | `f_prime.pi_ccs_nc.raw_old_block.padding` | Append exactly ten computed-zero lanes to each child | computed |
//! | `f_prime.pi_ccs_nc.raw_old_block.recomposition` | Recompose fourteen projected children in radix order and compare with the pending parent | checked |

use neo_math::{D, K};
use neo_reductions::block_projection::{project_raw_witnesses_at_block_point, BLOCK_PROJECTION_POINT_LEN};
use p3_field::PrimeCharacteristicRing;

use super::R1csIvcRawAssignmentAuthority;
use crate::engine::r1cs_circuit::RawOldBlockProjectionPlan;
use crate::paper::relations::WitnessMat;

pub const RAW_OLD_BLOCK_CHILD_COUNT: usize = 14;
pub const RAW_OLD_BLOCK_ACTIVE_LANES: usize = D;
pub const RAW_OLD_BLOCK_PADDED_LANES: usize = 64;
pub const RAW_OLD_BLOCK_ZERO_PADDING_LANES: usize = RAW_OLD_BLOCK_PADDED_LANES - RAW_OLD_BLOCK_ACTIVE_LANES;
pub const RAW_OLD_BLOCK_LOGICAL_COLUMNS: usize = 11_437_038;
pub const RAW_OLD_BLOCK_PACKED_COLUMNS: usize = 211_797;

/// Fixed production profile whose raw-witness dataflow is exported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R1csIvcRawOldBlockProfile {
    ActiveFPrimeCombinedNcDelayedV1,
}

/// Exact base-field-to-quadratic-extension decoding used by the native loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R1csIvcRawOldBlockFieldDecoding {
    BaseFieldEmbedding,
}

/// One ordered raw child's projection at the pending old block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csIvcRawOldBlockChildAudit {
    child: usize,
    authority: R1csIvcRawAssignmentAuthority,
    active_lanes: [K; RAW_OLD_BLOCK_ACTIVE_LANES],
    zero_padding: [K; RAW_OLD_BLOCK_ZERO_PADDING_LANES],
}

impl R1csIvcRawOldBlockChildAudit {
    pub fn child(&self) -> usize {
        self.child
    }

    pub fn authority(&self) -> R1csIvcRawAssignmentAuthority {
        self.authority
    }

    pub fn active_lanes(&self) -> &[K; RAW_OLD_BLOCK_ACTIVE_LANES] {
        &self.active_lanes
    }

    pub fn zero_padding(&self) -> &[K; RAW_OLD_BLOCK_ZERO_PADDING_LANES] {
        &self.zero_padding
    }
}

/// Proof-free execution record produced only from the full raw witness family.
///
/// The projected lane values are native execution evidence, not physical R1CS
/// columns. The pending old block and parent are separately joined to exact
/// generated source/normalized columns by `R1csIvcGeneratedKBindingAudit`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csIvcRawOldBlockExecutionAudit {
    profile: R1csIvcRawOldBlockProfile,
    field_decoding: R1csIvcRawOldBlockFieldDecoding,
    logical_columns: usize,
    packed_rows: usize,
    packed_columns: usize,
    old_block: [K; BLOCK_PROJECTION_POINT_LEN],
    children: Vec<R1csIvcRawOldBlockChildAudit>,
    radix: K,
    recomposed_parent_y_zcol: [K; RAW_OLD_BLOCK_ACTIVE_LANES],
}

impl R1csIvcRawOldBlockExecutionAudit {
    pub fn profile(&self) -> R1csIvcRawOldBlockProfile {
        self.profile
    }

    pub fn field_decoding(&self) -> R1csIvcRawOldBlockFieldDecoding {
        self.field_decoding
    }

    pub fn logical_columns(&self) -> usize {
        self.logical_columns
    }

    pub fn packed_shape(&self) -> (usize, usize) {
        (self.packed_rows, self.packed_columns)
    }

    pub fn old_block(&self) -> &[K; BLOCK_PROJECTION_POINT_LEN] {
        &self.old_block
    }

    pub fn children(&self) -> &[R1csIvcRawOldBlockChildAudit] {
        &self.children
    }

    pub fn radix(&self) -> K {
        self.radix
    }

    pub fn recomposed_parent_y_zcol(&self) -> &[K; RAW_OLD_BLOCK_ACTIVE_LANES] {
        &self.recomposed_parent_y_zcol
    }

    /// Compact parametric schedule used by the terminal R1CS emitter for
    /// these exact ordered raw witnesses.
    pub fn projection_plan(&self) -> RawOldBlockProjectionPlan {
        RawOldBlockProjectionPlan::new(self.logical_columns, self.children.len())
            .expect("validated raw-old-block profile fits the fixed domain")
    }
}

/// Capture the exact native projection and immediately replay it against the
/// same raw matrices and pending state. Construction fails closed on any
/// profile, child order, lane, padding, old-point, or parent mismatch.
pub(super) fn capture_and_validate_raw_old_block_execution(
    witnesses: &[WitnessMat],
    logical_columns: usize,
    old_block: &[K; BLOCK_PROJECTION_POINT_LEN],
    pending_parent_y_zcol: &[K; RAW_OLD_BLOCK_ACTIVE_LANES],
    radix: K,
) -> Result<R1csIvcRawOldBlockExecutionAudit, String> {
    validate_profile_geometry(witnesses, logical_columns)?;
    let projected = project_raw_witnesses_at_block_point(witnesses, logical_columns, old_block)
        .map_err(|error| format!("raw old-block projection failed: {error}"))?;
    let children = projected
        .into_iter()
        .enumerate()
        .map(|(child, active_lanes)| R1csIvcRawOldBlockChildAudit {
            child,
            authority: R1csIvcRawAssignmentAuthority::RunningWitnessMat,
            active_lanes,
            zero_padding: [K::ZERO; RAW_OLD_BLOCK_ZERO_PADDING_LANES],
        })
        .collect::<Vec<_>>();
    let recomposed_parent_y_zcol = recompose_children(&children, radix)
        .ok_or_else(|| "raw old-block child family does not match the fixed profile".to_string())?;
    let audit = R1csIvcRawOldBlockExecutionAudit {
        profile: R1csIvcRawOldBlockProfile::ActiveFPrimeCombinedNcDelayedV1,
        field_decoding: R1csIvcRawOldBlockFieldDecoding::BaseFieldEmbedding,
        logical_columns,
        packed_rows: D,
        packed_columns: logical_columns.div_ceil(D),
        old_block: *old_block,
        children,
        radix,
        recomposed_parent_y_zcol,
    };
    validate_raw_old_block_execution(
        &audit,
        witnesses,
        logical_columns,
        old_block,
        pending_parent_y_zcol,
        radix,
    )?;
    Ok(audit)
}

/// Fail-closed replay used by the active execution path and mutation tests.
///
/// This deliberately reprojects the supplied `WitnessMat`s. It does not trust
/// the captured lane values, authority tag, child indices, or recomposition.
pub fn validate_raw_old_block_execution(
    audit: &R1csIvcRawOldBlockExecutionAudit,
    witnesses: &[WitnessMat],
    logical_columns: usize,
    pending_old_block: &[K; BLOCK_PROJECTION_POINT_LEN],
    pending_parent_y_zcol: &[K; RAW_OLD_BLOCK_ACTIVE_LANES],
    radix: K,
) -> Result<(), String> {
    validate_profile_geometry(witnesses, logical_columns)?;
    if audit.profile != R1csIvcRawOldBlockProfile::ActiveFPrimeCombinedNcDelayedV1
        || audit.field_decoding != R1csIvcRawOldBlockFieldDecoding::BaseFieldEmbedding
        || audit.logical_columns != logical_columns
        || audit.packed_rows != D
        || audit.packed_columns != logical_columns.div_ceil(D)
        || audit.old_block != *pending_old_block
        || audit.radix != radix
        || audit.children.len() != RAW_OLD_BLOCK_CHILD_COUNT
    {
        return Err("raw old-block execution header or pending-point association drift".into());
    }

    let expected = project_raw_witnesses_at_block_point(witnesses, logical_columns, pending_old_block)
        .map_err(|error| format!("raw old-block replay failed: {error}"))?;
    for (child, (record, expected_lanes)) in audit.children.iter().zip(&expected).enumerate() {
        if record.child != child
            || record.authority != R1csIvcRawAssignmentAuthority::RunningWitnessMat
            || record.active_lanes != *expected_lanes
            || record.zero_padding != [K::ZERO; RAW_OLD_BLOCK_ZERO_PADDING_LANES]
        {
            return Err(format!(
                "raw old-block child {child} order, authority, lane, or padding mismatch"
            ));
        }
    }
    let recomposed = recompose_children(&audit.children, radix)
        .ok_or_else(|| "raw old-block child family is incomplete".to_string())?;
    if audit.recomposed_parent_y_zcol != recomposed || recomposed != *pending_parent_y_zcol {
        return Err("raw old-block radix recomposition disagrees with the pending parent".into());
    }
    Ok(())
}

fn validate_profile_geometry(witnesses: &[WitnessMat], logical_columns: usize) -> Result<(), String> {
    if D != 54
        || RAW_OLD_BLOCK_ZERO_PADDING_LANES != 10
        || logical_columns != RAW_OLD_BLOCK_LOGICAL_COLUMNS
        || logical_columns.div_ceil(D) != RAW_OLD_BLOCK_PACKED_COLUMNS
        || witnesses.len() != RAW_OLD_BLOCK_CHILD_COUNT
    {
        return Err(format!(
            "raw old-block fixed profile drift: children={} lanes={D}+{} logical_columns={logical_columns} packed_columns={}",
            witnesses.len(),
            RAW_OLD_BLOCK_ZERO_PADDING_LANES,
            logical_columns.div_ceil(D),
        ));
    }
    Ok(())
}

fn recompose_children(children: &[R1csIvcRawOldBlockChildAudit], radix: K) -> Option<[K; RAW_OLD_BLOCK_ACTIVE_LANES]> {
    if children.len() != RAW_OLD_BLOCK_CHILD_COUNT {
        return None;
    }
    let mut power = K::ONE;
    let mut recomposed = [K::ZERO; RAW_OLD_BLOCK_ACTIVE_LANES];
    for (child, record) in children.iter().enumerate() {
        if record.child != child {
            return None;
        }
        for (target, value) in recomposed.iter_mut().zip(record.active_lanes) {
            *target += value * power;
        }
        power *= radix;
    }
    Some(recomposed)
}
