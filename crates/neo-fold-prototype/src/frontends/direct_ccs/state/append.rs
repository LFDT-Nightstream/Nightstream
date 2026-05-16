//! Append transitions for direct-CCS IVC state.
//!
//! This file owns the hot state transition: validate the carried state, append
//! one SuperNeo chunk/relation, derive the next Construction-2 instance, and
//! record the latest relation used by terminal compression.

use super::*;

impl DirectCcsIvcState {
    pub fn append_step<L, MR, MB>(
        &self,
        step: DirectCcsStep,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let chunk = self.chunk_for_step(step);
        self.append_chunk(chunk, log, mixers)
    }

    pub(crate) fn append_step_with_f_prime_accumulator<L, MR, MB>(
        &self,
        step: DirectCcsStep,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
        f_prime_accumulator_digest: [u8; 32],
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let chunk = self.chunk_for_step(step);
        let (next_state, relation) = self.fold_chunk_with_superneo(chunk, log, mixers)?;
        self.advance_construction2_after_superneo_step(next_state, &relation, log, mixers, f_prime_accumulator_digest)
    }

    pub fn append_chunk<L, MR, MB>(
        &self,
        chunk: ChunkInput,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let (next_state, relation) = self.fold_chunk_with_superneo(chunk, log, mixers)?;
        self.append_verified_relation_with_state(next_state, &relation, log, mixers)
    }

    pub fn append_relation<L, MR, MB>(
        &self,
        relation: &SuperNeoIvcStepRelation,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        self.verify_carried_superneo_relation(relation, log, mixers)?;
        self.append_verified_relation_with_state(relation.state_out.clone(), relation, log, mixers)
    }

    pub(crate) fn append_relation_with_f_prime_accumulator<L, MR, MB>(
        &self,
        relation: &SuperNeoIvcStepRelation,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
        f_prime_accumulator_digest: [u8; 32],
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        self.verify_carried_superneo_relation(relation, log, mixers)?;
        self.advance_construction2_after_superneo_step(
            relation.state_out.clone(),
            relation,
            log,
            mixers,
            f_prime_accumulator_digest,
        )
    }

    pub fn append_all<L, MR, MB>(
        params: &NeoParams,
        structure: &CcsStructure<F>,
        relations: &[SuperNeoIvcStepRelation],
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let mut state = Self::from_parts(params, structure)?;
        for relation in relations {
            state = state.append_relation(relation, log, mixers)?;
        }
        Ok(state)
    }

    fn append_verified_relation_with_state<L, MR, MB>(
        &self,
        state_out: SuperNeoIvcState,
        relation: &SuperNeoIvcStepRelation,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let accumulator_digest = direct_accumulator_digest_from_claims(&self.params, &state_out.carry.claims);
        self.advance_construction2_after_superneo_step(state_out, relation, log, mixers, accumulator_digest)
    }

    fn advance_construction2_after_superneo_step<L, MR, MB>(
        &self,
        state_out: SuperNeoIvcState,
        relation: &SuperNeoIvcStepRelation,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
        construction2_accumulator_digest_out: [u8; 32],
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let terminal_replay = build_direct_ccs_chunk_surface_from_ivc_relation(
            &self.params,
            &self.structure,
            self.dims,
            relation,
            log,
            mixers,
            &self.optimized_cache,
        )?;
        let construction2 = self.derive_next_construction2_state(
            &state_out,
            relation,
            &terminal_replay,
            construction2_accumulator_digest_out,
        )?;
        Ok(self.advance_with_verified_superneo_step(state_out, relation, terminal_replay, construction2))
    }

    fn chunk_for_step(&self, step: DirectCcsStep) -> ChunkInput {
        ChunkInput {
            start_index: self.state.step_count as usize,
            steps: vec![step.into_step_input()],
        }
    }

    fn derive_next_construction2_state(
        &self,
        state_out: &SuperNeoIvcState,
        relation: &SuperNeoIvcStepRelation,
        terminal_replay: &DirectCcsChunkCircuitSurface,
        construction2_accumulator_digest_out: [u8; 32],
    ) -> Result<DirectCcsNextConstruction2State, DirectCcsFPrimeSnarkError> {
        let accumulator_digest = direct_accumulator_digest_from_claims(&self.params, &state_out.carry.claims);
        let public_trace_digest = direct_public_trace_update_digest(
            self.public_trace_digest,
            terminal_replay.replay.handoff.public_chunk_instance_digest,
        );
        let current_boundary_digest = direct_boundary_update_digest(
            self.current_boundary_digest,
            terminal_replay.replay.handoff.public_chunk_instance_digest,
        );
        let x_out = direct_state_x_out(
            self.vk_fs_digest,
            &self.mat_digest,
            state_out.chunk_count,
            state_out.step_count,
            self.initial_boundary_digest,
            current_boundary_digest,
            DIRECT_CCS_TRIVIAL_PC,
            accumulator_digest,
            construction2_accumulator_digest_out,
            public_trace_digest,
        );
        let construction2_u_i = self.derive_next_construction2_u_i(
            relation,
            terminal_replay,
            state_out.chunk_count,
            state_out.step_count,
            &x_out,
            accumulator_digest,
            construction2_accumulator_digest_out,
            current_boundary_digest,
            &state_out.carry.claims,
            &state_out.carry.witnesses,
        )?;
        Ok(DirectCcsNextConstruction2State {
            accumulator_digest,
            construction2_accumulator_digest: construction2_accumulator_digest_out,
            public_trace_digest,
            current_boundary_digest,
            x_out,
            construction2_u_i,
        })
    }

    fn advance_with_verified_superneo_step(
        &self,
        state_out: SuperNeoIvcState,
        relation: &SuperNeoIvcStepRelation,
        terminal_replay: DirectCcsChunkCircuitSurface,
        construction2: DirectCcsNextConstruction2State,
    ) -> Self {
        let DirectCcsNextConstruction2State {
            accumulator_digest,
            construction2_accumulator_digest,
            public_trace_digest,
            current_boundary_digest,
            x_out,
            construction2_u_i,
        } = construction2;
        Self {
            params: self.params.clone(),
            structure: self.structure.clone(),
            public_input_len: self.public_input_len,
            dims: self.dims,
            mat_digest: self.mat_digest,
            vk_fs_digest: self.vk_fs_digest,
            initial_boundary_digest: self.initial_boundary_digest,
            current_boundary_digest,
            optimized_cache: self.optimized_cache.clone(),
            state: state_out,
            accumulator_digest,
            construction2_accumulator_digest,
            public_trace_digest,
            x_i: x_out.clone(),
            construction2_u_i,
            last_step: Some(DirectCcsIvcStepRecord {
                relation: relation.clone(),
                surface: terminal_replay,
                x_i: self.x_i.clone(),
                construction2_u_i: self.construction2_u_i.clone(),
                x_out,
                accumulator_in_digest: self.accumulator_digest,
                accumulator_out_digest: accumulator_digest,
                construction2_accumulator_in_digest: self.construction2_accumulator_digest,
                construction2_accumulator_out_digest: construction2_accumulator_digest,
                public_trace_in_digest: self.public_trace_digest,
                public_trace_out_digest: public_trace_digest,
                current_boundary_in_digest: self.current_boundary_digest,
                current_boundary_out_digest: current_boundary_digest,
                construction2_fold: None,
            }),
        }
    }
}

struct DirectCcsNextConstruction2State {
    accumulator_digest: [u8; 32],
    construction2_accumulator_digest: [u8; 32],
    public_trace_digest: [u8; 32],
    current_boundary_digest: [u8; 32],
    x_out: Construction2EncodedPublicInput,
    construction2_u_i: Construction2FreshInstance,
}
