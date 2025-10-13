//! NIVC verification pipeline

use crate::{F, NeoParams};
use super::super::api::{NivcProgram, NivcChainProof, NivcAccumulators};
use p3_field::PrimeField64;

mod helpers;
use helpers::{expected_app_inputs, build_prev_acc_lane, check_step_x_prefix_suffix};

/// Verify an NIVC chain given the program, initial y, and parameter set.
pub fn verify_chain(
    program: &NivcProgram,
    params: &NeoParams,
    chain: &NivcChainProof,
    initial_y: &[F],
) -> anyhow::Result<bool> {
    if program.is_empty() { 
        return Ok(false); 
    }

    // Initialize verifier‑side accumulators (no ME state needed; we rely on inner proofs)
    let mut acc = NivcAccumulators::new(program.len(), initial_y.to_vec());
    acc.step = 0;
    for lane in &mut acc.lanes {
        lane.c_coords.clear();
        lane.c_digest = [0u8; 32];
    }

    // Maintain lane-local previous augmented X to enforce LHS linking on repeated lane usage
    let mut prev_aug_x_by_lane: Vec<Option<Vec<F>>> = vec![None; program.len()];

    for sp in &chain.steps {
        let j = sp.lane_idx;
        if j >= program.len() { 
            return Ok(false); 
        }

        // Lane‑scoped accumulator to feed the existing IVC verifier
        let lane = &acc.lanes[j];
        let prev_acc_lane = build_prev_acc_lane(lane, &acc.global_y, acc.step);

        // Build expected step_x = [H(prev_acc_lane) || which || step_io || lanes_root]
        let acc_prefix = crate::ivc::compute_accumulator_digest_fields(&prev_acc_lane)
            .map_err(|e| anyhow::anyhow!("compute_accumulator_digest_fields failed: {}", e))?;
        let expected_app = expected_app_inputs(j, &sp.step_io, &acc);
        
        // Enforce prefix/suffix equality
        let step_x = sp.inner.public_inputs.wrapper_public_input_x();
        if !check_step_x_prefix_suffix(step_x, &acc_prefix, &expected_app) {
            return Ok(false);
        }
        
        // Redundant but explicit: selector in suffix must match `which`
        let digest_len = acc_prefix.len();
        let which_in_x = step_x[digest_len].as_canonical_u64() as usize;
        if which_in_x != j { 
            return Ok(false); 
        }

        let ok = crate::ivc::verify_ivc_step(
            &program.steps[j].ccs,
            &sp.inner,
            &prev_acc_lane,
            &program.steps[j].binding,
            params,
            prev_aug_x_by_lane[j].as_deref(),
        ).map_err(|e| anyhow::anyhow!("verify_ivc_step failed: {}", e))?;
        
        if !ok { 
            return Ok(false); 
        }

        // Update lane commitment and global y from the proof
        let lane_mut = &mut acc.lanes[j];
        lane_mut.c_coords = sp.inner.next_accumulator.c_coords.clone();
        lane_mut.c_digest = sp.inner.next_accumulator.c_z_digest;
        acc.global_y = sp.inner.next_accumulator.y_compact.clone();
        acc.step += 1;

        // Update lane-local previous augmented X for linking next time this lane is used
        prev_aug_x_by_lane[j] = Some(sp.inner.public_inputs.step_augmented_public_input().to_vec());
    }

    // Final snapshot minimal check (global y and step)
    Ok(acc.global_y == chain.final_acc.global_y && acc.step == chain.final_acc.step)
}

