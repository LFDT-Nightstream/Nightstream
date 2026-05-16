//! Verifier-side proof-session performance counters.

#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkVerifyPerf {
    pub start_index: usize,
    pub fresh_steps: usize,
    pub incoming_main_claims: usize,
    pub ccs_outputs: usize,
    pub dec_children: usize,
    pub prepare_inputs_ms: f64,
    pub ccs_bind_ms: f64,
    pub ccs_bind_header_instances_ms: f64,
    pub ccs_bind_header_prefix_ms: f64,
    pub ccs_bind_header_poly_ms: f64,
    pub ccs_bind_header_public_instances_ms: f64,
    pub ccs_bind_me_inputs_ms: f64,
    pub ccs_bind_sample_challenges_ms: f64,
    pub ccs_fe_sumcheck_ms: f64,
    pub ccs_nc_sumcheck_ms: f64,
    pub ccs_output_checks_ms: f64,
    pub ccs_terminal_ms: f64,
    pub ccs_ms: f64,
    pub digest_checks_ms: f64,
    pub dims_ms: f64,
    pub rlc_challenge_ms: f64,
    pub rlc_rho_mats_ms: f64,
    pub rlc_rho_k_lift_ms: f64,
    pub rlc_x_ms: f64,
    pub rlc_y_ms: f64,
    pub rlc_y_zcol_ms: f64,
    pub rlc_aux_ms: f64,
    pub rlc_commitment_collect_ms: f64,
    pub rlc_commitment_mix_ms: f64,
    pub rlc_commitment_ms: f64,
    pub rlc_ms: f64,
    pub dec_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RunVerifyPerf {
    pub chunks: Vec<ChunkVerifyPerf>,
    pub total_ms: f64,
}

impl RunVerifyPerf {
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn fresh_steps(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.fresh_steps).sum()
    }

    pub fn incoming_main_claims(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.incoming_main_claims)
            .sum()
    }

    pub fn ccs_outputs(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.ccs_outputs).sum()
    }

    pub fn dec_children(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.dec_children).sum()
    }

    pub fn prepare_inputs_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.prepare_inputs_ms)
            .sum()
    }

    pub fn ccs_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.ccs_ms).sum()
    }

    pub fn ccs_bind_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.ccs_bind_ms).sum()
    }

    pub fn ccs_bind_header_instances_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_header_instances_ms)
            .sum()
    }

    pub fn ccs_bind_header_prefix_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_header_prefix_ms)
            .sum()
    }

    pub fn ccs_bind_header_poly_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_header_poly_ms)
            .sum()
    }

    pub fn ccs_bind_header_public_instances_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_header_public_instances_ms)
            .sum()
    }

    pub fn ccs_bind_me_inputs_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_me_inputs_ms)
            .sum()
    }

    pub fn ccs_bind_sample_challenges_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_bind_sample_challenges_ms)
            .sum()
    }

    pub fn ccs_fe_sumcheck_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_fe_sumcheck_ms)
            .sum()
    }

    pub fn ccs_nc_sumcheck_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_nc_sumcheck_ms)
            .sum()
    }

    pub fn ccs_output_checks_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_output_checks_ms)
            .sum()
    }

    pub fn ccs_terminal_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.ccs_terminal_ms).sum()
    }

    pub fn digest_checks_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.digest_checks_ms).sum()
    }

    pub fn dims_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dims_ms).sum()
    }

    pub fn rlc_challenge_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_challenge_ms).sum()
    }

    pub fn rlc_x_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_x_ms).sum()
    }

    pub fn rlc_rho_mats_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_rho_mats_ms).sum()
    }

    pub fn rlc_rho_k_lift_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.rlc_rho_k_lift_ms)
            .sum()
    }

    pub fn rlc_y_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_y_ms).sum()
    }

    pub fn rlc_y_zcol_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_y_zcol_ms).sum()
    }

    pub fn rlc_aux_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_aux_ms).sum()
    }

    pub fn rlc_commitment_collect_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.rlc_commitment_collect_ms)
            .sum()
    }

    pub fn rlc_commitment_mix_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.rlc_commitment_mix_ms)
            .sum()
    }

    pub fn rlc_commitment_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.rlc_commitment_ms)
            .sum()
    }

    pub fn rlc_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_ms).sum()
    }

    pub fn dec_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dec_ms).sum()
    }
}
