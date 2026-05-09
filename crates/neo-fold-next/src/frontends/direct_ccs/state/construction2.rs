//! Construction-2 instance derivation for the direct-CCS append path.

use super::*;

impl DirectCcsIvcState {
    pub(super) fn derive_next_construction2_u_i(
        &self,
        relation: &SuperNeoIvcStepRelation,
        surface: &DirectCcsChunkCircuitSurface,
        chunk_count_out: u64,
        step_count_out: u64,
        x_out: &Construction2EncodedPublicInput,
        accumulator_out_digest: [u8; 32],
        construction2_accumulator_out_digest: [u8; 32],
        current_boundary_out_digest: [u8; 32],
        final_claims: &[CeClaim<Commitment, F, K>],
        final_witnesses: &[Mat<F>],
    ) -> Result<Construction2FreshInstance, DirectCcsFPrimeSnarkError> {
        let circuit = DirectCcsFPrimeCircuit {
            params: self.params.clone(),
            structure: self.structure.clone(),
            dims: self.dims,
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            chunks: vec![surface.clone()],
            initial_claims: relation.state_in.carry.claims.clone(),
            initial_transcript: Some(relation.state_in.transcript.clone()),
            chunk_count_in: relation.state_in.chunk_count,
            step_count_in: relation.state_in.step_count,
            x_in: self.x_i.clone(),
            construction2_input_u_i: self.construction2_u_i.clone(),
            accumulator_in_digest: self.accumulator_digest,
            construction2_accumulator_in_digest: self.construction2_accumulator_digest,
            public_trace_in_digest: self.public_trace_digest,
            current_boundary_in_digest: self.current_boundary_digest,
            chunk_count_out,
            step_count_out,
            x_out: x_out.clone(),
            accumulator_out_digest,
            construction2_accumulator_out_digest,
            public_trace_out_digest: direct_public_trace_update_digest(
                self.public_trace_digest,
                surface.replay.handoff.public_chunk_instance_digest,
            ),
            current_boundary_out_digest,
            construction2_fold: None,
            final_claims: final_claims.to_vec(),
            final_witnesses: final_carry_witnesses(final_witnesses)?,
        };
        let relation = DirectCcsTerminalCommittedRelation::from_terminal_circuit(circuit.terminal_circuit(false))
            .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        Construction2FreshInstance::from_public_boundary(relation.public_boundary())
            .map_err(DirectCcsFPrimeSnarkError::Input)
    }
}
