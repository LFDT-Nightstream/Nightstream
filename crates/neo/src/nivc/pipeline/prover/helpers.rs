//! Helper functions for NIVC proving
//!
//! This module provides utilities for constructing inputs to the IVC folding
//! operation from NIVC lane state.

use crate::F;
use crate::ivc::Accumulator;
use super::super::super::api::LaneRunningState;
use p3_field::PrimeCharacteristicRing;

/// Build a lane-scoped accumulator view from NIVC lane state.
///
/// Constructs an [`Accumulator`] that represents the previous state of a single
/// lane, suitable for passing to the IVC prover. The accumulator uses the lane's
/// commitment digest and coordinates, paired with the global output `y_i` from
/// the shared NIVC state.
///
/// # Arguments
/// - `lane`: The running state of a specific lane (one step function `F_j`)
/// - `global_y`: The current compact global output vector `y_i`
/// - `step`: The current global step counter
pub fn build_prev_acc_lane(lane: &LaneRunningState, global_y: &[F], step: u64) -> Accumulator {
    Accumulator {
        c_z_digest: lane.c_digest,
        c_coords: lane.c_coords.clone(),
        y_compact: global_y.to_vec(),
        step,
    }
}

/// Build application public inputs for the step.
///
/// Constructs the public input vector that binds the lane identifier, step-specific
/// I/O, and the Merkle root of all lane accumulators into the Fiat-Shamir transcript.
///
/// Format: `[lane_idx || step_io || lanes_root]`
///
/// # Arguments
/// - `lane_idx`: The lane index (selects which step function `F_j` is being proved)
/// - `step_io`: Application-specific public inputs for this step
/// - `lanes_root`: The Merkle root of all lane accumulators (binding integrity)
///
/// # Returns
/// A field element vector ready for transcript binding.
pub fn make_app_inputs(lane_idx: usize, step_io: &[F], lanes_root: &[F]) -> Vec<F> {
    let mut app_inputs = Vec::with_capacity(1 + step_io.len() + lanes_root.len());
    app_inputs.push(F::from_u64(lane_idx as u64));
    app_inputs.extend_from_slice(step_io);
    app_inputs.extend_from_slice(lanes_root);
    app_inputs
}

