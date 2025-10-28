//! Checks module: Structural sanity and shape validation
//!
//! Validates consistency between inputs, outputs, and CCS structure
//! to prevent malformed or malicious instance data.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K, D};

/// Validate ME outputs against inputs for structural consistency
///
/// Checks:
/// - Output count matches MCS + ME input count
/// - Commitments and m_in values match
/// - X matrices have correct shape (d×m_in)
/// - y vectors have correct shape (t vectors, each d elements)
/// - y_scalars length matches t
pub fn sanity_check_outputs_against_inputs(
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    out_me: &[MeInstance<Cmt, F, K>],
) -> Result<(), PiCcsError> {
    if out_me.len() != mcs_list.len() + me_inputs.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "output count {} != MCS {} + ME {} = {}",
            out_me.len(),
            mcs_list.len(),
            me_inputs.len(),
            mcs_list.len() + me_inputs.len()
        )));
    }

    for (i, (out, inp)) in out_me
        .iter()
        .take(mcs_list.len())
        .zip(mcs_list.iter())
        .enumerate()
    {
        if out.c != inp.c {
            return Err(PiCcsError::InvalidInput(format!(
                "output[{i}].c mismatch with MCS instance"
            )));
        }
        if out.m_in != inp.m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "output[{i}].m_in mismatch"
            )));
        }
        if out.X.rows() != D || out.X.cols() != inp.m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "output[{i}].X shape ({},{}), expected ({},{})",
                out.X.rows(),
                out.X.cols(),
                D,
                inp.m_in
            )));
        }
        if out.y.len() != s.t() {
            return Err(PiCcsError::InvalidInput(format!(
                "output[{i}].y length {} != t={}",
                out.y.len(),
                s.t()
            )));
        }
        for (j, yj) in out.y.iter().enumerate() {
            if yj.len() != D {
                return Err(PiCcsError::InvalidInput(format!(
                    "output[{i}].y[{j}] length {} != d={}",
                    yj.len(),
                    D
                )));
            }
        }
        if out.y_scalars.len() != s.t() {
            return Err(PiCcsError::InvalidInput(format!(
                "output[{i}].y_scalars length {} != t={}",
                out.y_scalars.len(),
                s.t()
            )));
        }
    }

    for (idx, (out, inp)) in out_me
        .iter()
        .skip(mcs_list.len())
        .zip(me_inputs.iter())
        .enumerate()
    {
        if out.c != inp.c {
            return Err(PiCcsError::InvalidInput(format!(
                "me_output[{idx}].c mismatch"
            )));
        }
        if out.m_in != inp.m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "me_output[{idx}].m_in mismatch"
            )));
        }
        if out.y.len() != s.t() {
            return Err(PiCcsError::InvalidInput(format!(
                "me_output[{idx}].y length {} != t={}",
                out.y.len(),
                s.t()
            )));
        }
        for (j, yj) in out.y.iter().enumerate() {
            if yj.len() != D {
                return Err(PiCcsError::InvalidInput(format!(
                    "me_output[{idx}].y[{j}] length {} != d={}",
                    yj.len(),
                    D
                )));
            }
        }
    }

    Ok(())
}

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

