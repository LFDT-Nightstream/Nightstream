//! Prover-side proof-session performance counters.

#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkProvePerf {
    pub start_index: usize,
    pub fresh_steps: usize,
    pub incoming_main_claims: usize,
    pub ccs_outputs: usize,
    pub dec_children: usize,
    pub prepare_inputs_ms: f64,
    pub ccs_bind_ms: f64,
    pub ccs_sample_challenges_ms: f64,
    pub ccs_fe_sumcheck_ms: f64,
    pub ccs_nc_sumcheck_ms: f64,
    pub ccs_output_materialize_ms: f64,
    pub ccs_ms: f64,
    pub dims_ms: f64,
    pub rlc_prepare_ms: f64,
    pub rlc_ms: f64,
    pub dec_split_ms: f64,
    pub dec_commit_ms: f64,
    pub dec_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RunProvePerf {
    pub chunks: Vec<ChunkProvePerf>,
    pub total_ms: f64,
}

impl RunProvePerf {
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

    pub fn ccs_sample_challenges_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_sample_challenges_ms)
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

    pub fn ccs_output_materialize_ms(&self) -> f64 {
        self.chunks
            .iter()
            .map(|chunk| chunk.ccs_output_materialize_ms)
            .sum()
    }

    pub fn dims_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dims_ms).sum()
    }

    pub fn rlc_prepare_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_prepare_ms).sum()
    }

    pub fn rlc_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.rlc_ms).sum()
    }

    pub fn dec_split_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dec_split_ms).sum()
    }

    pub fn dec_commit_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dec_commit_ms).sum()
    }

    pub fn dec_ms(&self) -> f64 {
        self.chunks.iter().map(|chunk| chunk.dec_ms).sum()
    }
}
