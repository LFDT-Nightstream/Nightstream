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
        let chunk = ChunkInput {
            start_index: self.state.step_count as usize,
            steps: vec![step.into_step_input()],
        };
        self.append_chunk(chunk, log, mixers)
    }

    pub(crate) fn append_step_with_construction2_accumulator_digest<L, MR, MB>(
        &self,
        step: DirectCcsStep,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
        construction2_accumulator_digest_out: [u8; 32],
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let chunk = ChunkInput {
            start_index: self.state.step_count as usize,
            steps: vec![step.into_step_input()],
        };
        self.append_chunk_with_construction2_accumulator_digest(
            chunk,
            log,
            mixers,
            construction2_accumulator_digest_out,
        )
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
        let (next_state, relation) = self.build_and_verify_relation_from_chunk(chunk, log, mixers)?;
        self.append_verified_relation_with_state(next_state, &relation, log, mixers)
    }

    fn append_chunk_with_construction2_accumulator_digest<L, MR, MB>(
        &self,
        chunk: ChunkInput,
        log: &L,
        mixers: crate::prover::CommitmentMixers<MR, MB>,
        construction2_accumulator_digest_out: [u8; 32],
    ) -> Result<Self, DirectCcsFPrimeSnarkError>
    where
        L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let (next_state, relation) = self.build_and_verify_relation_from_chunk(chunk, log, mixers)?;
        self.append_verified_relation_with_state_with_construction2_accumulator_digest(
            next_state,
            &relation,
            log,
            mixers,
            construction2_accumulator_digest_out,
        )
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
        self.validate_and_verify_carried_relation(relation, log, mixers)?;
        self.append_verified_relation_with_state(relation.state_out.clone(), relation, log, mixers)
    }

    pub(crate) fn append_relation_with_construction2_accumulator_digest<L, MR, MB>(
        &self,
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
        self.validate_and_verify_carried_relation(relation, log, mixers)?;
        self.append_verified_relation_with_state_with_construction2_accumulator_digest(
            relation.state_out.clone(),
            relation,
            log,
            mixers,
            construction2_accumulator_digest_out,
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
        self.append_verified_relation_with_state_with_construction2_accumulator_digest(
            state_out,
            relation,
            log,
            mixers,
            accumulator_digest,
        )
    }

    fn append_verified_relation_with_state_with_construction2_accumulator_digest<L, MR, MB>(
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
        let surface = build_direct_ccs_chunk_surface_from_ivc_relation(
            &self.params,
            &self.structure,
            self.dims,
            relation,
            log,
            mixers,
            &self.optimized_cache,
        )?;
        let accumulator_digest = direct_accumulator_digest_from_claims(&self.params, &state_out.carry.claims);
        let public_trace_digest = direct_public_trace_update_digest(
            self.public_trace_digest,
            surface.replay.handoff.public_chunk_instance_digest,
        );
        let current_boundary_digest = direct_boundary_update_digest(
            self.current_boundary_digest,
            surface.replay.handoff.public_chunk_instance_digest,
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
            &surface,
            state_out.chunk_count,
            state_out.step_count,
            &x_out,
            accumulator_digest,
            construction2_accumulator_digest_out,
            current_boundary_digest,
            &state_out.carry.claims,
            &state_out.carry.witnesses,
        )?;
        Ok(Self {
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
            construction2_accumulator_digest: construction2_accumulator_digest_out,
            public_trace_digest,
            x_i: x_out.clone(),
            construction2_u_i,
            last_step: Some(DirectCcsIvcStepRecord {
                relation: relation.clone(),
                surface,
                x_i: self.x_i.clone(),
                construction2_u_i: self.construction2_u_i.clone(),
                x_out,
                accumulator_in_digest: self.accumulator_digest,
                accumulator_out_digest: accumulator_digest,
                construction2_accumulator_in_digest: self.construction2_accumulator_digest,
                construction2_accumulator_out_digest: construction2_accumulator_digest_out,
                public_trace_in_digest: self.public_trace_digest,
                public_trace_out_digest: public_trace_digest,
                current_boundary_in_digest: self.current_boundary_digest,
                current_boundary_out_digest: current_boundary_digest,
                construction2_fold: None,
            }),
        })
    }
}
