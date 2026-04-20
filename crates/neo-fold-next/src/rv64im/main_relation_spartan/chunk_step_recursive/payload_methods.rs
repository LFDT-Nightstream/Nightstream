use super::Rv64imMainRecursionFPrimePayload;
use crate::rv64im::final_relation::Rv64imChunkFoldTranscriptSnapshot;
use crate::rv64im::main_recursion::{Rv64imMainRecursionFPrimeAdvice, RV64IM_MAIN_RECURSION_TRIVIAL_PC};
use crate::rv64im::main_relation_trace::{
    build_rv64im_main_circuit_chunk_replay_surface, build_rv64im_main_circuit_pi_ccs_replay_surface,
    Rv64imMainCircuitChunkReplaySurface,
};
use crate::rv64im::SimpleKernelError;

impl Rv64imMainRecursionFPrimePayload {
    pub fn phi_side_commitment_words(&self) -> &[Vec<u64>] {
        &self.phi_side_commitment_words
    }

    pub fn z_0(&self) -> &[u8; 32] {
        &self.z_0
    }

    pub fn z_i(&self) -> &[u8; 32] {
        &self.z_i
    }

    pub fn z_next(&self) -> &[u8; 32] {
        &self.z_next
    }

    pub fn pc_i(&self) -> u64 {
        self.pc_i
    }

    pub fn pc_next(&self) -> u64 {
        self.pc_next
    }

    pub fn fixed_transcript_out(&self) -> &Rv64imChunkFoldTranscriptSnapshot {
        &self.fixed_transcript_out
    }

    pub fn padded_fresh_claim_count(&self) -> usize {
        self.chunk_cover.fresh_claim_count as usize
    }

    pub fn effective_fresh_claim_count(&self) -> usize {
        self.step_shape.fresh_claim_count as usize
    }

    pub(crate) fn effective_chunk_replay_surface(
        &self,
    ) -> Result<Rv64imMainCircuitChunkReplaySurface, SimpleKernelError> {
        let mut replay = self.pi_ccs.replay.clone();
        replay
            .sumcheck_rounds
            .truncate(self.step_shape.fe_round_lengths.len());
        for (round, live_len) in replay
            .sumcheck_rounds
            .iter_mut()
            .zip(self.step_shape.fe_round_lengths.iter())
        {
            if round.len() < *live_len as usize {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM recursive-step payload cannot truncate a padded FE round to the live coefficient count"
                        .into(),
                ));
            }
            round.truncate(*live_len as usize);
        }
        replay
            .sumcheck_rounds_nc
            .truncate(self.step_shape.nc_round_lengths.len());
        for (round, live_len) in replay
            .sumcheck_rounds_nc
            .iter_mut()
            .zip(self.step_shape.nc_round_lengths.iter())
        {
            if round.len() < *live_len as usize {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM recursive-step payload cannot truncate a padded NC round to the live coefficient count"
                        .into(),
                ));
            }
            round.truncate(*live_len as usize);
        }
        build_rv64im_main_circuit_chunk_replay_surface(
            &self.handoff,
            &self.fresh_claims[..self.step_shape.fresh_claim_count as usize],
            build_rv64im_main_circuit_pi_ccs_replay_surface(
                self.pi_ccs.ccs_outputs[..self.step_shape.ccs_output_count as usize].to_vec(),
                replay,
                self.pi_ccs.public_challenges.clone(),
                self.pi_ccs.row_chals.clone(),
                self.pi_ccs.alpha_prime.clone(),
                self.pi_ccs.s_col.clone(),
                self.pi_ccs.alpha_prime_nc.clone(),
            ),
            self.pi_rlc.parent.clone(),
            self.pi_dec.children[..self.step_shape.child_count as usize].to_vec(),
        )
    }

    pub fn matches_cover_shape(&self) -> bool {
        self.state_in_claims.len() == self.cover_shape.state_in_claim_count as usize
            && self.state_out_claims.len() == self.cover_shape.state_out_claim_count as usize
            && self.chunk_cover.fresh_claim_count == self.cover_shape.fresh_claim_count
            && self.chunk_cover.fresh_witness_count == self.cover_shape.fresh_witness_count
            && self.chunk_cover.fresh_claim_shapes.len() == self.cover_shape.fresh_claim_count as usize
            && self.chunk_cover.fresh_witness_shapes.len() == self.cover_shape.fresh_witness_count as usize
            && self
                .fresh_claims
                .iter()
                .enumerate()
                .all(|(idx, claim)| self.chunk_cover.fresh_claim_shapes[idx].covers_claim(claim))
            && self.fresh_witnesses.len() == self.cover_shape.fresh_witness_count as usize
            && self
                .fresh_witnesses
                .iter()
                .enumerate()
                .all(|(idx, witness)| self.chunk_cover.fresh_witness_shapes[idx].covers_witness(witness))
            && self
                .chunk_cover
                .parent_claim_shape
                .covers_claim(&self.pi_rlc.parent)
            && self.chunk_cover.ccs_output_count == self.cover_shape.ccs_output_count
            && self.chunk_cover.child_count == self.cover_shape.child_count
            && self.chunk_cover.ccs_output_shapes.len() == self.cover_shape.ccs_output_count as usize
            && self.chunk_cover.child_claim_shapes.len() == self.cover_shape.child_count as usize
            && self.pi_ccs.ccs_outputs.len() == self.cover_shape.ccs_output_count as usize
            && self
                .pi_ccs
                .ccs_outputs
                .iter()
                .enumerate()
                .all(|(idx, claim)| self.chunk_cover.ccs_output_shapes[idx].covers_claim(claim))
            && self.pi_dec.children.len() == self.cover_shape.child_count as usize
            && self
                .pi_dec
                .children
                .iter()
                .enumerate()
                .all(|(idx, claim)| self.chunk_cover.child_claim_shapes[idx].covers_claim(claim))
            && self.chunk_cover.fe_round_lengths == self.cover_shape.fe_round_lengths
            && self.chunk_cover.nc_round_lengths == self.cover_shape.nc_round_lengths
            && self.pi_ccs.replay.sumcheck_rounds.len() == self.cover_shape.fe_round_lengths.len()
            && self.pi_ccs.replay.sumcheck_rounds_nc.len() == self.cover_shape.nc_round_lengths.len()
            && self
                .pi_ccs
                .replay
                .sumcheck_rounds
                .iter()
                .zip(self.cover_shape.fe_round_lengths.iter())
                .all(|(round, live_len)| round.len() == *live_len as usize)
            && self
                .pi_ccs
                .replay
                .sumcheck_rounds_nc
                .iter()
                .zip(self.cover_shape.nc_round_lengths.iter())
                .all(|(round, live_len)| round.len() == *live_len as usize)
    }

    pub fn matches_explicit_semantics(&self, advice: &Rv64imMainRecursionFPrimeAdvice) -> bool {
        self.z_0 == *advice.z_0()
            && self.z_i == *advice.z_i()
            && self.z_next == advice.fresh_state_out().carry.terminal_handle.0
            && self.pc_i == advice.pc_i()
            && self.pc_next == RV64IM_MAIN_RECURSION_TRIVIAL_PC
            && self.phi_side_commitment_words == advice.phi_side().commitment_words()
    }
}
