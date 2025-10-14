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
    #[cfg(feature = "neo-logs")]
    {
        println!("🔎 NIVC VERIFY: Starting chain verification");
        println!("   program.steps.len() = {}", program.len());
        println!("   chain.steps.len() = {}", chain.steps.len());
        println!("   chain.final_acc.step = {}", chain.final_acc.step);
    }
    
    if program.is_empty() { 
        return Ok(false); 
    }
    
    // SECURITY: Reject programs with CCS structures that are too small for sumcheck security
    // ℓ = ceil(log2(n_padded)) must be ≥ 2, so minimum n=3 (→ padded to 4 → ℓ=2)
    for (_lane, spec) in program.steps.iter().enumerate() {
        if spec.ccs.n < 3 {
            #[cfg(feature = "neo-logs")]
            println!("   ❌ SECURITY: Lane {} has n={} < 3 (ℓ must be ≥ 2 post-padding)", _lane, spec.ccs.n);
            return Ok(false);
        }
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

    for (_step_idx, sp) in chain.steps.iter().enumerate() {
        #[cfg(feature = "neo-logs")]
        println!("🔎 NIVC VERIFY: Step {}/{} (lane {})", _step_idx + 1, chain.steps.len(), sp.lane_idx);
        
        let j = sp.lane_idx;
        if j >= program.len() { 
            #[cfg(feature = "neo-logs")]
            println!("   ❌ Lane index {} >= program.len() {}", j, program.len());
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
            #[cfg(feature = "neo-logs")]
            println!("   ❌ Prefix/suffix check failed");
            return Ok(false);
        }
        
        // Redundant but explicit: selector in suffix must match `which`
        let digest_len = acc_prefix.len();
        let which_in_x = step_x[digest_len].as_canonical_u64() as usize;
        if which_in_x != j { 
            #[cfg(feature = "neo-logs")]
            println!("   ❌ Lane selector mismatch: expected {}, got {}", j, which_in_x);
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
            #[cfg(feature = "neo-logs")]
            println!("   ❌ IVC step verification failed");
            return Ok(false); 
        }
        
        #[cfg(feature = "neo-logs")]
        println!("   ✅ Step verified");

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
    let y_matches = acc.global_y == chain.final_acc.global_y;
    let step_matches = acc.step == chain.final_acc.step;
    
    #[cfg(feature = "neo-logs")]
    {
        println!("🔎 NIVC VERIFY: Final check");
        println!("   Computed acc.step: {}", acc.step);
        println!("   Expected chain.final_acc.step: {}", chain.final_acc.step);
        println!("   Step matches: {}", step_matches);
        println!("   Y matches: {}", y_matches);
        println!("   Overall result: {}", y_matches && step_matches);
    }
    
    Ok(y_matches && step_matches)
}

