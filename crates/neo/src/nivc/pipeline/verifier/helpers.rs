//! Helper functions for NIVC verification

use crate::F;
use crate::ivc::Accumulator;
use super::super::super::api::{NivcAccumulators, LaneRunningState};
use super::super::super::internal::lanes_root_fields;
use p3_field::PrimeCharacteristicRing;

/// Build expected application inputs: [which || step_io || lanes_root]
pub fn expected_app_inputs(which: usize, step_io: &[F], acc: &NivcAccumulators) -> Vec<F> {
    let lanes_root = lanes_root_fields(acc);
    let mut expected = Vec::with_capacity(1 + step_io.len() + lanes_root.len());
    expected.push(F::from_u64(which as u64));
    expected.extend_from_slice(step_io);
    expected.extend_from_slice(&lanes_root);
    expected
}

/// Build a lane-scoped accumulator for verification
pub fn build_prev_acc_lane(lane: &LaneRunningState, global_y: &[F], step: u64) -> Accumulator {
    Accumulator {
        c_z_digest: lane.c_digest,
        c_coords: lane.c_coords.clone(),
        y_compact: global_y.to_vec(),
        step,
    }
}

/// Check that step_x has the correct prefix and suffix structure
pub fn check_step_x_prefix_suffix(
    step_x: &[F],
    acc_prefix: &[F],
    expected_app_inputs: &[F],
) -> bool {
    let digest_len = acc_prefix.len();
    if step_x.len() != digest_len + expected_app_inputs.len() {
        return false;
    }
    if &step_x[..digest_len] != acc_prefix {
        return false;
    }
    if &step_x[digest_len..] != expected_app_inputs {
        return false;
    }
    true
}

