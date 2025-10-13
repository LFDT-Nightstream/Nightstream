//! NIVC proving pipeline
//!
//! This module runs one **NIVC step** (as in the HyperNova paper) for a single
//! lane. Each lane represents one step function `F_j`. Calling [`step`] folds
//! that lane's fresh step into its running accumulator and updates the global
//! state.
//!
//! Paper mapping (HyperNova, 2024):
//! - lane index `lane_idx` → program counter φ selecting `F_j`
//! - `state.acc.lanes[lane_idx]` → running instance `U_i[j]`
//! - `step_io`, `step_witness` → fresh instance `(u_i, w_i)`
//! - `prove_ivc_step_chained` → non-interactive folding operation
//! - `state.acc.global_y` → compact global output `y_i`
//! - `NivcStepProof` → exported step proof `Π_i`
//!
//! ```text
//! 1) Select lane: j = φ(z_i, ω_i) = lane_idx
//! 2) (U_i[j]) + (u_i, w_i) --prove_ivc_step_chained--> U_{i+1}[j]   [cache prev_aug_X[j]]
//! 3) Global accumulator: y_i → y_{i+1} ; step counter: i → i+1
//! ```

use crate::F;
use crate::ivc::{IvcStepInput, prove_ivc_step_chained};
use super::super::api::{NivcState, NivcStepProof};
use super::super::internal::{IndexExtractor, lanes_root_fields};

mod helpers;
use helpers::{build_prev_acc_lane, make_app_inputs};

/// Run one NIVC step for lane `lane_idx`.
///
/// Builds lane-scoped accumulator input, binds public data into the
/// transcript, folds the new step via `prove_ivc_step_chained`, and
/// updates both the lane and global accumulators.
///
/// Returns the per-step proof; updates `state` in place.
///
/// # Errors
/// - Returns `Err` if `lane_idx` is out of range or if the IVC fold fails.
pub fn step(
    state: &mut NivcState,
    lane_idx: usize,
    step_io: &[F],
    step_witness: &[F],
) -> anyhow::Result<NivcStepProof> {
    if lane_idx >= state.program.steps.len() {
        anyhow::bail!("lane_idx out of bounds");
    }
    let spec = &state.program.steps[lane_idx];

    // Previous accumulator for this lane
    let lane = &state.acc.lanes[lane_idx];
    let prev_acc_lane = build_prev_acc_lane(lane, &state.acc.global_y, state.acc.step);

    // Public input binding: lane index + lanes root
    let lanes_root = lanes_root_fields(&state.acc);
    let app_inputs = make_app_inputs(lane_idx, step_io, &lanes_root);

    // Extract y_step subset per binding spec
    let extractor = IndexExtractor {
        indices: spec.binding.y_step_offsets.clone(),
    };
    let y_step = extractor.extract_y_step(step_witness);

    // Carry over lane metadata
    let prev_me = state.acc.lanes[lane_idx].me.clone();
    let prev_wit = state.acc.lanes[lane_idx].wit.clone();
    let prev_mcs = state.acc.lanes[lane_idx]
        .lhs_mcs
        .clone()
        .zip(state.acc.lanes[lane_idx].lhs_mcs_wit.clone());

    // Perform chained IVC fold
    let input = IvcStepInput {
        params: &state.params,
        step_ccs: &spec.ccs,
        step_witness,
        prev_accumulator: &prev_acc_lane,
        step: state.acc.step,
        public_input: Some(&app_inputs),
        y_step: &y_step,
        binding_spec: &spec.binding,
        transcript_only_app_inputs: true,
        prev_augmented_x: state.prev_aug_x_by_lane[lane_idx].as_deref(),
    };
    let (res, me_out, wit_out, lhs_next) =
        prove_ivc_step_chained(input, prev_me, prev_wit, prev_mcs)
            .map_err(|e| anyhow::anyhow!("prove_ivc_step_chained failed: {}", e))?;

    // Update lane state
    let lane_mut = &mut state.acc.lanes[lane_idx];
    lane_mut.me = Some(me_out);
    lane_mut.wit = Some(wit_out);
    lane_mut.c_coords = res.proof.next_accumulator.c_coords.clone();
    lane_mut.c_digest = res.proof.next_accumulator.c_z_digest;
    lane_mut.lhs_mcs = Some(lhs_next.0);
    lane_mut.lhs_mcs_wit = Some(lhs_next.1);

    // Update global accumulator and step counter
    state.acc.global_y = res.proof.next_accumulator.y_compact.clone();
    state.acc.step += 1;

    // Cache augmented public input for next use of this lane
    state.prev_aug_x_by_lane[lane_idx] =
        Some(res.proof.public_inputs.step_augmented_public_input().to_vec());

    // Return step proof
    let sp = NivcStepProof {
        lane_idx,
        step_io: step_io.to_vec(),
        inner: res.proof,
    };
    state.steps.push(sp.clone());
    Ok(sp)
}

