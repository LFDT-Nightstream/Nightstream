//! Read-only state views and latest F' relation materialization.

use super::*;

impl DirectCcsIvcState {
    pub fn final_state(&self) -> &SuperNeoIvcState {
        &self.state
    }
    pub fn params(&self) -> &NeoParams {
        &self.params
    }
    pub fn structure(&self) -> &CcsStructure<F> {
        &self.structure
    }
    pub fn construction2_public_boundary(&self) -> Construction2PublicBoundary {
        Construction2PublicBoundary::from_fresh_instance(&self.construction2_u_i)
    }

    pub fn latest_relation_and_advice(&self) -> Result<DirectCcsLatestFPrimeSummary, DirectCcsFPrimeSnarkError> {
        let last = self.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input(
                "direct CCS folded compression requires at least one appended SuperNeo relation".into(),
            )
        })?;
        Ok(DirectCcsLatestFPrimeSummary {
            chunk_index: last.relation.chunk_index,
            fresh_claims: last.relation.chunk.steps.len(),
            incoming_ce_claims: last.relation.state_in.carry.claims.len(),
            output_ce_claims: last.relation.replay_witness.ccs_outputs.len(),
            final_ce_claims: self.state.carry.claims.len(),
            construction2_x_in: last.x_i.clone(),
            construction2_x_out: last.x_out.clone(),
        })
    }

    pub(crate) fn latest_construction2_fold_context(
        &self,
    ) -> Result<DirectCcsConstruction2FoldContext, DirectCcsFPrimeSnarkError> {
        let last = self.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input("direct CCS Construction-2 fold context requires a latest step".into())
        })?;
        Ok(DirectCcsConstruction2FoldContext {
            params: self.params.clone(),
            structure: self.structure.clone(),
            dims: self.dims,
            mat_digest: self.mat_digest,
            initial_claims: last.relation.state_in.carry.claims.clone(),
            initial_transcript: Some(last.relation.state_in.transcript.clone()),
            surface: last.surface.clone(),
            accumulator_in_digest: last.accumulator_in_digest,
            accumulator_out_digest: last.accumulator_out_digest,
        })
    }

    pub(crate) fn with_latest_construction2_fold_context(
        mut self,
        context: Option<DirectCcsConstruction2FoldContext>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let Some(context) = context else {
            return Ok(self);
        };
        let last = self.last_step.as_mut().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input("direct CCS Construction-2 fold context requires a latest step".into())
        })?;
        context.validate_digest_linkage(
            last.construction2_accumulator_in_digest,
            last.construction2_accumulator_out_digest,
        )?;
        last.construction2_fold = Some(context);
        let relation =
            DirectCcsTerminalCommittedRelation::from_terminal_circuit(self.latest_circuit()?.terminal_circuit(false))
                .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(err.to_string()))?;
        let construction2_u_i = Construction2FreshInstance::from_public_boundary(relation.public_boundary())
            .map_err(DirectCcsFPrimeSnarkError::Input)?;
        if construction2_u_i.x_i() != &self.x_i {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct CCS Construction-2 folded output u_i does not match current x_i".into(),
            ));
        }
        self.construction2_u_i = construction2_u_i;
        Ok(self)
    }

    pub(crate) fn latest_circuit(&self) -> Result<DirectCcsFPrimeCircuit, DirectCcsFPrimeSnarkError> {
        let last = self.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input(
                "direct CCS folded compression requires at least one appended SuperNeo relation".into(),
            )
        })?;
        Ok(DirectCcsFPrimeCircuit {
            params: self.params.clone(),
            structure: self.structure.clone(),
            dims: self.dims,
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            chunks: vec![last.surface.clone()],
            initial_claims: last.relation.state_in.carry.claims.clone(),
            initial_transcript: Some(last.relation.state_in.transcript.clone()),
            chunk_count_in: last.relation.state_in.chunk_count,
            step_count_in: last.relation.state_in.step_count,
            x_in: last.x_i.clone(),
            construction2_input_u_i: last.construction2_u_i.clone(),
            accumulator_in_digest: last.accumulator_in_digest,
            construction2_accumulator_in_digest: last.construction2_accumulator_in_digest,
            public_trace_in_digest: last.public_trace_in_digest,
            current_boundary_in_digest: last.current_boundary_in_digest,
            chunk_count_out: self.state.chunk_count,
            step_count_out: self.state.step_count,
            x_out: last.x_out.clone(),
            accumulator_out_digest: last.accumulator_out_digest,
            construction2_accumulator_out_digest: last.construction2_accumulator_out_digest,
            public_trace_out_digest: last.public_trace_out_digest,
            current_boundary_out_digest: last.current_boundary_out_digest,
            construction2_fold: last.construction2_fold.clone(),
            final_claims: self.state.carry.claims.clone(),
            final_witnesses: final_carry_witnesses(&self.state.carry.witnesses)?,
        })
    }
}
