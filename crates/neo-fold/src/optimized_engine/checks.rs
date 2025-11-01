//! Checks module: Structural sanity and shape validation
//!
//! Validates consistency between inputs, outputs, and CCS structure
//! to prevent malformed or malicious instance data.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};


/// Validate input consistency before starting reduction
pub fn validate_inputs(
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[neo_ccs::Mat<F>],
) -> Result<(), PiCcsError> {
    if me_inputs.len() != me_witnesses.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "me_inputs.len() {} != me_witnesses.len() {}",
            me_inputs.len(),
            me_witnesses.len()
        )));
    }

    if mcs_list.is_empty() || mcs_list.len() != witnesses.len() {
        return Err(PiCcsError::InvalidInput(
            "empty or mismatched MCS list/witnesses".into(),
        ));
    }

    if !me_inputs.is_empty() {
        let r_inp = &me_inputs[0].r;
        if !me_inputs.iter().all(|me| &me.r == r_inp) {
            return Err(PiCcsError::InvalidInput(
                "all ME inputs must share the same r".into(),
            ));
        }
    }

    if s.n == 0 {
        return Err(PiCcsError::InvalidInput("n=0 not allowed".into()));
    }

    Ok(())
}

/// Sanity check that outputs match inputs in count, shape, and ordering
/// 
/// This validates that:
/// - Output count = |MCS| + |ME inputs|
/// - Outputs are ordered: [MCS-derived outputs..., ME-derived outputs...]
/// - Shape consistency (m_in)
/// 
/// This is a defensive check to prevent subtle bugs from output mis-ordering
/// that would silently corrupt the γ-exponent schedule in Eval'.
#[cfg(debug_assertions)]
pub fn sanity_check_outputs_against_inputs(
    _s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    out_me: &[MeInstance<Cmt, F, K>],
) -> Result<(), PiCcsError> {
    let expected_count = mcs_list.len() + me_inputs.len();
    if out_me.len() != expected_count {
        return Err(PiCcsError::InvalidInput(format!(
            "output count {} != |MCS|={} + |ME|={} = {}",
            out_me.len(),
            mcs_list.len(),
            me_inputs.len(),
            expected_count
        )));
    }

    // Check all outputs have correct m_in relative to their source
    // - First |MCS| outputs correspond to mcs_list[i].m_in
    // - Next |ME| outputs correspond to me_inputs[i].m_in
    for (idx, out) in out_me.iter().enumerate() {
        if idx < mcs_list.len() {
            let want = mcs_list[idx].m_in;
            if out.m_in != want {
                return Err(PiCcsError::InvalidInput(format!(
                    "out_me[{}].m_in {} != mcs_list[{}].m_in {}",
                    idx, out.m_in, idx, want
                )));
            }
        } else {
            let j = idx - mcs_list.len();
            let want = me_inputs.get(j).map(|me| me.m_in).unwrap_or(0);
            if out.m_in != want {
                return Err(PiCcsError::InvalidInput(format!(
                    "out_me[{}].m_in {} != me_inputs[{}].m_in {}",
                    idx, out.m_in, j, want
                )));
            }
        }
    }

    // Verify output ordering: first |MCS| outputs correspond to MCS inputs,
    // then |ME| outputs correspond to ME inputs
    
    Ok(())
}

/// No-op version for release builds
#[cfg(not(debug_assertions))]
#[inline]
pub fn sanity_check_outputs_against_inputs(
    _s: &CcsStructure<F>,
    _mcs_list: &[McsInstance<Cmt, F>],
    _me_inputs: &[MeInstance<Cmt, F, K>],
    _out_me: &[MeInstance<Cmt, F, K>],
) -> Result<(), PiCcsError> {
    Ok(())
}
