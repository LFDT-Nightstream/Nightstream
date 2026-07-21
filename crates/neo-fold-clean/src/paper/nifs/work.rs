//! Witness threading and checked cross-phase handoffs used by NIFS.
//!
//! NIFS.P needs to assemble a parallel `(claims, witnesses)` array of
//! length K+k for Π_RLC after Π_CCS hands it back the K+k output claims.
//! These helpers are pure data-movement — no math.

use neo_reductions::optimized_engine::PiCcsProofVariant;

use crate::paper::construction2::running::{PendingProjectionState, RunningInstanceError};
use crate::paper::relations::{CcsClaim, CcsInstance, CcsWitness, CeClaim, WitnessMat};

/// Split fresh CCS instances by moving their public claims and private
/// witnesses into parallel arrays. No witness matrix is cloned here.
pub(super) fn split_fresh_instances(fresh: Vec<CcsInstance>) -> (Vec<CcsClaim>, Vec<CcsWitness>) {
    let mut claims = Vec::with_capacity(fresh.len());
    let mut witnesses = Vec::with_capacity(fresh.len());
    for instance in fresh {
        claims.push(instance.claim);
        witnesses.push(instance.witness);
    }
    (claims, witnesses)
}

/// Build the borrowed K+k witness array Π_RLC expects, parallel to the
/// Π_CCS output claims. The fresh witnesses live in the split arrays
/// above; the carried witnesses remain borrowed from the running
/// accumulator.
pub(super) fn chain_witness_refs<'a>(fresh: &'a [CcsWitness], running: &'a [WitnessMat]) -> Vec<&'a WitnessMat> {
    let mut out = Vec::with_capacity(fresh.len() + running.len());
    out.extend(fresh.iter().map(|w| &w.Z));
    out.extend(running.iter());
    out
}

/// Construct the next one-fold state from an accepted block-lane Π_CCS point
/// and the independently recomputed Π_RLC parent. The legacy proof format has
/// no delayed state.
pub(super) fn outgoing_pending_projection(
    variant: PiCcsProofVariant,
    outputs: &[CeClaim],
    parent: &CeClaim,
) -> Result<Option<PendingProjectionState>, RunningInstanceError> {
    if variant == PiCcsProofVariant::SplitNcV1 {
        return Ok(None);
    }
    let first = outputs
        .first()
        .ok_or(RunningInstanceError::PendingOldBlockLength {
            expected: crate::paper::construction2::running::PENDING_PROJECTION_OLD_BLOCK_LEN,
            got: 0,
        })?;
    PendingProjectionState::try_from_block_and_parent(&first.s_col, &parent.y_zcol).map(Some)
}
