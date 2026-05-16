//! Construction and base public-image initialization for direct-CCS state.

use super::*;

impl DirectCcsIvcState {
    pub fn new(program: DirectCcsProgram) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let mut state = Self::from_parts(program.params(), program.structure())?;
        state.public_input_len = program.public_input_len();
        state.reset_base_public_image();
        Ok(state)
    }

    pub fn start(program: DirectCcsProgram) -> Result<Self, DirectCcsFPrimeSnarkError> {
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
}
