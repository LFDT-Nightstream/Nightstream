//! Helper functions for NIVC finalization

use crate::F;
use super::super::super::api::{NivcChainProof, NivcStepProof};
use neo_ccs::CcsStructure;
use neo_spartan_bridge::pi_ccs_embed as piccs;
use p3_field::PrimeCharacteristicRing;

/// Pick the best ME instance and witness to use for the final SNARK
pub fn pick_me_and_witness<'a>(
    last: &'a NivcStepProof,
    chain: &'a NivcChainProof,
    j: usize,
) -> anyhow::Result<(&'a neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>, &'a neo_ccs::MeWitness<F>)> {
    // Extract ME and witness for final SNARK:
    // Prefer the RHS step ME/witness whose commitment equals c_step, so Ajtai@step binds correctly.
    // Fall back to the most recent available pair; else use running lane.
    if let (Some(meis), Some(wits)) = (&last.inner.me_instances, &last.inner.digit_witnesses) {
        if !meis.is_empty() && !wits.is_empty() {
            // Try to locate the RHS (step) instance by exact commitment equality.
            let mut idx = core::cmp::min(meis.len(), wits.len()) - 1; // default: last
            for i in 0..core::cmp::min(meis.len(), wits.len()) {
                if meis[i].c.data.len() == last.inner.c_step_coords.len()
                    && meis[i].c.data == last.inner.c_step_coords
                {
                    idx = i;
                    break;
                }
            }
            return Ok((&meis[idx], &wits[idx]));
        }
    }
    
    // Fallback to running lane
    let lane = &chain.final_acc.lanes[j];
    match (&lane.me, &lane.wit) {
        (Some(me), Some(wit)) => Ok((me, wit)),
        _ => anyhow::bail!("No running ME instance available on the chosen lane for final proof"),
    }
}

/// Build augmented CCS for the final SNARK
pub fn build_augmented_ccs(
    ccs: &CcsStructure<F>,
    step_x_len: usize,
    y_step_offsets: &[usize],
    y_prev_witness_indices: &[usize],
    step_program_input_witness_indices: &[usize],
    y_len: usize,
    const1_witness_index: usize,
) -> anyhow::Result<CcsStructure<F>> {
    crate::ivc::build_augmented_ccs_linked_with_rlc(
        ccs,
        step_x_len,
        y_step_offsets,
        y_prev_witness_indices,
        step_program_input_witness_indices,
        y_len,
        const1_witness_index,
        None,
    ).map_err(|e| anyhow::anyhow!("Failed to build augmented CCS: {}", e))
}

/// Build Pi-CCS embed from augmented CCS matrices
pub fn build_pi_ccs_embed(augmented_ccs: &CcsStructure<F>) -> Option<piccs::PiCcsEmbed> {
    let mut mats = Vec::with_capacity(augmented_ccs.matrices.len());
    for mj in &augmented_ccs.matrices {
        let rows = mj.rows();
        let cols = mj.cols();
        let mut entries = Vec::new();
        for r in 0..rows { 
            for c in 0..cols {
                let a = mj[(r, c)];
                if a != F::ZERO { 
                    entries.push((r as u32, c as u32, a)); 
                }
            }
        }
        mats.push(piccs::CcsCsr { rows, cols, entries });
    }
    Some(piccs::PiCcsEmbed { matrices: mats })
}
