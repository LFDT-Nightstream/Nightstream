//! Native shape summary for the latest SuperNeo NIFS payload.

use serde::{Deserialize, Serialize};

use super::super::super::state::{DirectCcsFPrimeSnarkError, DirectCcsIvcState};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsFPrimeNifsPayloadShape {
    pub chunk_index: u64,
    pub fresh_claims: usize,
    pub incoming_ce_claims: usize,
    pub pi_ccs_outputs: usize,
    pub final_ce_claims: usize,
    pub fe_sumcheck_rounds: usize,
    pub fe_sumcheck_messages: usize,
    pub nc_sumcheck_rounds: usize,
    pub nc_sumcheck_messages: usize,
    pub transcript_absorbed_in: usize,
    pub transcript_absorbed_out: usize,
}

impl DirectCcsFPrimeNifsPayloadShape {
    pub fn from_latest_state(state: &DirectCcsIvcState) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let last = state.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input("direct F' NIFS payload shape requires an appended step".into())
        })?;
        Ok(Self {
            chunk_index: last.relation.chunk_index,
            fresh_claims: last.relation.chunk.steps.len(),
            incoming_ce_claims: last.relation.state_in.carry.claims.len(),
            pi_ccs_outputs: last.relation.replay_witness.ccs_outputs.len(),
            final_ce_claims: last.relation.state_out.carry.claims.len(),
            fe_sumcheck_rounds: last
                .relation
                .replay_witness
                .ccs_replay_proof
                .sumcheck_rounds
                .len(),
            fe_sumcheck_messages: last
                .relation
                .replay_witness
                .ccs_replay_proof
                .sumcheck_rounds
                .iter()
                .map(Vec::len)
                .sum(),
            nc_sumcheck_rounds: last
                .relation
                .replay_witness
                .ccs_replay_proof
                .sumcheck_rounds_nc
                .len(),
            nc_sumcheck_messages: last
                .relation
                .replay_witness
                .ccs_replay_proof
                .sumcheck_rounds_nc
                .iter()
                .map(Vec::len)
                .sum(),
            transcript_absorbed_in: last.relation.state_in.transcript.absorbed,
            transcript_absorbed_out: last.relation.state_out.transcript.absorbed,
        })
    }
}
