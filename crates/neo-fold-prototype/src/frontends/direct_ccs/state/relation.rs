//! Relation construction and verification for the direct-CCS append path.

use super::*;

impl DirectCcsIvcState {
    pub(super) fn fold_chunk_with_superneo<L, MR, MB>(
        &self,
        chunk: ChunkInput,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<(SuperNeoIvcState, SuperNeoIvcStepRelation), DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        self.validate_current_surface()?;
        self.validate_chunk_shape(&chunk)?;
        // The shared SuperNeo chunk prover owns the reduction sequence:
        // prepare fresh/carry claims -> Pi_CCS -> Pi_RLC -> Pi_DEC.
        let (next_state, relation) = self
            .state
            .append_chunk_with_perf_and_accumulator_handle(
                &self.params,
                &self.structure,
                chunk,
                log,
                mixers,
                &self.optimized_cache,
            )
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        self.verify_superneo_relation_accumulator_handle(&relation, log, mixers)?;
        Ok((next_state, relation))
    }

    pub(super) fn verify_carried_superneo_relation<L, MR, MB>(
        &self,
        relation: &SuperNeoIvcStepRelation,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<(), DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        self.ensure_relation_starts_from_current_state(relation)?;
        self.validate_current_surface()?;
        self.validate_chunk_shape(&relation.chunk)?;
        self.verify_superneo_relation_accumulator_handle(relation, log, mixers)
    }

    fn ensure_relation_starts_from_current_state(
        &self,
        relation: &SuperNeoIvcStepRelation,
    ) -> Result<(), DirectCcsFPrimeSnarkError> {
        if !superneo_ivc_states_match(&self.state, &relation.state_in) {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct CCS IVC append relation does not start from the carried state".into(),
            ));
        }
        Ok(())
    }

    fn verify_superneo_relation_accumulator_handle<L, MR, MB>(
        &self,
        relation: &SuperNeoIvcStepRelation,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<(), DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        relation
            .verify_with_accumulator_handle(&self.params, &self.structure, log, mixers, &self.optimized_cache)
            .map(|_| ())
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))
    }
}
