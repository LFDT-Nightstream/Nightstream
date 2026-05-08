//! Owns direct-CCS IVC state transitions.
//!
//! Public methods stay in append/compress order at the top. Helper methods
//! below validate carried state, derive the next Construction-2 instance, and
//! build the latest terminal F' circuit consumed by the prover.

use super::*;

impl DirectCcsIvcState {
    pub fn new(program: DirectCcsProgram) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let mut state = Self::from_parts(program.params(), program.structure())?;
        state.public_input_len = program.public_input_len();
        state.reset_base_public_image();
        Ok(state)
    }

    pub fn new_with_canonical_zero_carry(program: DirectCcsProgram) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let carry = program.canonical_zero_carry()?;
        let mut state = Self::from_parts(program.params(), program.structure())?;
        state.public_input_len = program.public_input_len();
        state.state = SuperNeoIvcState::seed_with_carry(carry);
        state.accumulator_digest = direct_accumulator_digest_from_claims(&state.params, &state.state.carry.claims);
        state.construction2_accumulator_digest = state.accumulator_digest;
        state.reset_base_public_image();
        Ok(state)
    }

    pub fn from_parts(params: &NeoParams, structure: &CcsStructure<F>) -> Result<Self, DirectCcsFPrimeSnarkError> {
        validate_direct_ajtai_context(params, structure)?;
        let dims = build_dims_and_policy(params, structure)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        let optimized_cache = OptimizedStructureCache::build(structure)
            .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
        let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, None)
            .try_into()
            .map_err(|digest: Vec<Goldilocks>| {
                DirectCcsFPrimeSnarkError::Input(format!("expected 4 matrix digest limbs, got {}", digest.len()))
            })?;
        let state = SuperNeoIvcState::seed();
        let accumulator_digest = direct_accumulator_digest_from_claims(params, &state.carry.claims);
        let construction2_accumulator_digest = accumulator_digest;
        let vk_fs_digest = direct_vk_fs_digest(params, &mat_digest, None);
        let initial_boundary_digest = direct_initial_boundary_digest(&mat_digest, None);
        let current_boundary_digest = initial_boundary_digest;
        let public_trace_digest = direct_public_trace_seed_digest(&mat_digest);
        let x_i = direct_state_x_out(
            vk_fs_digest,
            &mat_digest,
            state.chunk_count,
            state.step_count,
            initial_boundary_digest,
            current_boundary_digest,
            DIRECT_CCS_TRIVIAL_PC,
            accumulator_digest,
            construction2_accumulator_digest,
            public_trace_digest,
        );
        let construction2_u_i = Construction2FreshInstance::canonical_zero(params.kappa as usize, x_i.clone());
        Ok(Self {
            params: params.clone(),
            structure: structure.clone(),
            public_input_len: None,
            dims,
            mat_digest,
            vk_fs_digest,
            initial_boundary_digest,
            current_boundary_digest,
            optimized_cache,
            state,
            accumulator_digest,
            construction2_accumulator_digest,
            public_trace_digest,
            x_i,
            construction2_u_i,
            last_step: None,
        })
    }

    pub fn compress_with_trace(
        &self,
        emit: &mut dyn FnMut(&str),
    ) -> Result<(DirectCcsFPrimeSnarkProof, DirectCcsFPrimeSnarkPerf), DirectCcsFPrimeSnarkError> {
        let (snark, _vk, perf) = self.compress_snark_with_trace(emit)?;
        Ok((snark.proof().clone(), perf))
    }

    pub fn compress_snark_with_trace(
        &self,
        emit: &mut dyn FnMut(&str),
    ) -> Result<
        (
            DirectCcsIvcSnark,
            DirectCcsIvcSnarkVerifierKey,
            DirectCcsFPrimeSnarkPerf,
        ),
        DirectCcsFPrimeSnarkError,
    > {
        self.ensure_terminal_compression_is_proof_complete()?;
        emit("direct_ccs_ivc.phase=latest_relation_and_advice.start");
        let circuit = self.latest_circuit()?;
        emit("direct_ccs_ivc.phase=latest_relation_and_advice.done");
        emit("direct_ccs_ivc.phase=spartan_terminal_prove.start");
        let proved = prove_direct_ccs_f_prime_circuit(circuit, emit);
        emit("direct_ccs_ivc.phase=spartan_terminal_prove.done");
        proved
    }

    pub fn compress(&self) -> Result<(DirectCcsFPrimeSnarkProof, DirectCcsFPrimeSnarkPerf), DirectCcsFPrimeSnarkError> {
        let mut emit = |_message: &str| {};
        self.compress_with_trace(&mut emit)
    }

    pub fn compress_snark(
        &self,
    ) -> Result<
        (
            DirectCcsIvcSnark,
            DirectCcsIvcSnarkVerifierKey,
            DirectCcsFPrimeSnarkPerf,
        ),
        DirectCcsFPrimeSnarkError,
    > {
        let mut emit = |_message: &str| {};
        self.compress_snark_with_trace(&mut emit)
    }

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

    fn build_and_verify_relation_from_chunk<L, MR, MB>(
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
        self.verify_relation_accumulator_handle(&relation, log, mixers)?;
        Ok((next_state, relation))
    }

    fn validate_and_verify_carried_relation<L, MR, MB>(
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
        self.verify_relation_accumulator_handle(relation, log, mixers)
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

    fn verify_relation_accumulator_handle<L, MR, MB>(
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

    fn derive_next_construction2_u_i(
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

    fn reset_base_public_image(&mut self) {
        self.vk_fs_digest = direct_vk_fs_digest(&self.params, &self.mat_digest, self.public_input_len);
        self.initial_boundary_digest = direct_initial_boundary_digest(&self.mat_digest, self.public_input_len);
        self.current_boundary_digest = self.initial_boundary_digest;
        self.public_trace_digest = direct_public_trace_seed_digest(&self.mat_digest);
        self.x_i = direct_state_x_out(
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
        self.construction2_u_i =
            Construction2FreshInstance::canonical_zero(self.params.kappa as usize, self.x_i.clone());
    }

    fn ensure_terminal_compression_is_proof_complete(&self) -> Result<(), DirectCcsFPrimeSnarkError> {
        let last = self.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input(
                "direct CCS folded compression requires at least one appended SuperNeo relation".into(),
            )
        })?;
        if self.state.chunk_count > 1 && last.construction2_fold.is_none() {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "plain direct CCS terminal compression is latest-only and disabled for multi-step runs".into(),
            ));
        }
        Ok(())
    }

    fn validate_current_surface(&self) -> Result<(), DirectCcsFPrimeSnarkError> {
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

    fn validate_chunk_shape(&self, chunk: &ChunkInput) -> Result<(), DirectCcsFPrimeSnarkError> {
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
