//! Owns native direct-CCS state validation predicates.

use neo_ajtai::get_global_pp_for_dims;
use neo_ccs::CcsStructure;
use neo_math::{D, F};
use neo_params::NeoParams;

use super::*;
use crate::ivc::SuperNeoIvcState;

pub(crate) fn validate_direct_ajtai_context(
    params: &NeoParams,
    structure: &CcsStructure<F>,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let witness_cols = structure.m.div_ceil(D);
    let pp = get_global_pp_for_dims(D, witness_cols).map_err(|err| {
        DirectCcsFPrimeSnarkError::Input(format!(
            "direct CCS program requires a registered Ajtai PP for (d,m)=({D},{witness_cols}): {err}"
        ))
    })?;
    if pp.kappa != params.kappa as usize {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct CCS Ajtai PP kappa mismatch for (d,m)=({D},{witness_cols}): registered {}, params {}",
            pp.kappa, params.kappa
        )));
    }
    Ok(())
}

pub(crate) fn superneo_ivc_states_match(left: &SuperNeoIvcState, right: &SuperNeoIvcState) -> bool {
    left.chunk_count == right.chunk_count
        && left.step_count == right.step_count
        && left.transcript == right.transcript
        && left.carry.claims == right.carry.claims
        && left.carry.witnesses == right.carry.witnesses
}

impl DirectCcsIvcState {
    pub(super) fn validate_current_surface(&self) -> Result<(), DirectCcsFPrimeSnarkError> {
        let expected_accumulator_digest = direct_accumulator_digest_from_claims(&self.params, &self.state.carry.claims);
        if self.accumulator_digest != expected_accumulator_digest {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct CCS IVC accumulator digest does not match carried CE state".into(),
            ));
        }
        let expected_x = direct_state_x_out(
            self.vk_fs_digest,
            &self.mat_digest,
            self.state.chunk_count,
            self.state.step_count,
            self.initial_boundary_digest,
            self.current_boundary_digest,
            DIRECT_CCS_TRIVIAL_PC,
            self.accumulator_digest,
            self.construction2_accumulator_digest,
            self.public_trace_digest,
        );
        if self.x_i != expected_x || self.construction2_u_i.x_i() != &self.x_i {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct CCS IVC Construction-2 current instance does not bind to carried x_i".into(),
            ));
        }
        if self.state.chunk_count == 0 {
            if self.state.step_count != 0 || self.last_step.is_some() {
                return Err(DirectCcsFPrimeSnarkError::Input(
                    "direct CCS IVC base state cannot carry non-zero progress".into(),
                ));
            }
            if !self
                .construction2_u_i
                .is_canonical_zero_for(self.params.kappa as usize, &self.x_i)
            {
                return Err(DirectCcsFPrimeSnarkError::Input(
                    "direct CCS IVC base state must carry a canonical Construction-2 default instance".into(),
                ));
            }
        } else {
            let boundary = Construction2PublicBoundary::from_fresh_instance(&self.construction2_u_i);
            if boundary.commitment_digest != boundary.expected_commitment_digest()
                || boundary.fresh_instance_digest != boundary.expected_fresh_instance_digest()
                || !boundary.has_canonical_commitment_shape()
            {
                return Err(DirectCcsFPrimeSnarkError::Input(
                    "direct CCS IVC carried Construction-2 boundary is not canonical".into(),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn validate_chunk_shape(&self, chunk: &ChunkInput) -> Result<(), DirectCcsFPrimeSnarkError> {
        let expected_cols = self.structure.m.div_ceil(D);
        for step in &chunk.steps {
            if let Some(expected_m_in) = self.public_input_len {
                if step.mcs.m_in != expected_m_in {
                    return Err(DirectCcsFPrimeSnarkError::Input(format!(
                        "direct CCS step {} has m_in={}, expected fixed program public input len {}",
                        step.label, step.mcs.m_in, expected_m_in
                    )));
                }
            }
            if step.mcs.m_in != step.mcs.x.len() {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct CCS step {} has m_in={} but {} public inputs",
                    step.label,
                    step.mcs.m_in,
                    step.mcs.x.len()
                )));
            }
            if step.mcs.m_in > self.structure.m {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct CCS step {} has m_in={} beyond CCS columns {}",
                    step.label, step.mcs.m_in, self.structure.m
                )));
            }
            let expected_w = self.structure.m - step.mcs.m_in;
            if step.witness.w.len() != expected_w {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct CCS step {} witness tail has len {}, expected {}",
                    step.label,
                    step.witness.w.len(),
                    expected_w
                )));
            }
            if step.witness.Z.rows() != D || step.witness.Z.cols() != expected_cols {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct CCS step {} packed witness shape is {}x{}, expected {}x{}",
                    step.label,
                    step.witness.Z.rows(),
                    step.witness.Z.cols(),
                    D,
                    expected_cols
                )));
            }
        }
        Ok(())
    }
}
