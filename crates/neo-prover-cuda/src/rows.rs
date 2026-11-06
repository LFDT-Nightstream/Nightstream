//! Device evaluation of circuit-owned rows, without a lifecycle-crate
//! dependency; the CUDA port of `neo-prover-metal`'s `rows.rs`.

use neo_ccs::Mat;
use neo_math::F;
use neo_reductions::PiCcsError;

use crate::{oracle_error, CudaActivity, CudaError, CudaSession};

pub struct CudaRowProver {
    session: CudaSession,
}

impl CudaRowProver {
    /// A session that expands the needed commitment key rows in every call.
    pub fn new() -> Result<Self, CudaError> {
        Ok(Self {
            session: CudaSession::new()?,
        })
    }

    /// Keep the commitment key rows on the device after the first commitment.
    pub fn keep_commitment_key(&mut self) {
        self.session.keep_production_key();
    }

    /// Free the kept commitment key rows until the next commitment.
    pub fn release_commitment_key_rows(&self) {
        self.session.release_production_key_rows();
    }

    /// The key weights `Σ_i λ_i·a_i` over the commitment key rows; see
    /// `CudaSession::production_key_weights`.
    pub fn production_key_weights(&self, powers: &[[u64; 3]], columns: usize) -> Result<Vec<[u64; 3]>, CudaError> {
        self.session.production_key_weights(powers, columns)
    }

    /// The device view of one GKR layer over the caller's tables; see
    /// `CudaSession::gkr_layer`.
    pub fn gkr_layer<'a>(
        &self,
        trees: Vec<Vec<&'a mut [u64]>>,
        eq: &'a mut [u64],
    ) -> Result<crate::CudaGkrLayer<'a>, CudaError> {
        self.session.gkr_layer(trees, eq)
    }

    pub fn gkr_message(
        &self,
        layer: &crate::CudaGkrLayer<'_>,
        weights: &[[u64; 3]],
        mu: [u64; 3],
        degree: usize,
    ) -> Result<Vec<[u64; 3]>, CudaError> {
        self.session.gkr_message(layer, weights, mu, degree)
    }

    pub fn gkr_fold(&self, layer: &mut crate::CudaGkrLayer<'_>, r: [u64; 3]) -> Result<(), CudaError> {
        self.session.gkr_fold(layer, r)
    }

    /// The DFT of every column of a row-major matrix; see
    /// `CudaSession::dft_columns`.
    pub fn dft_columns(&self, values: &mut [u64], width: usize, generator: u64) -> Result<(), CudaError> {
        self.session.dft_columns(values, width, generator)
    }

    /// Every digest layer of a Merkle tree; see
    /// `CudaSession::poseidon2_merkle_layers`.
    pub fn merkle_layers(
        &self,
        rows: usize,
        width: usize,
        fill: impl FnOnce(&mut [u64]),
    ) -> Result<Vec<Vec<[u64; 4]>>, CudaError> {
        self.session.poseidon2_merkle_layers(rows, width, fill)
    }

    /// The leaf digests of rows; see `CudaSession::poseidon2_merkle_leaves`.
    pub fn merkle_leaves(&self, values: &[u64], width: usize) -> Result<Vec<[u64; 4]>, CudaError> {
        self.session.poseidon2_merkle_leaves(values, width)
    }

    /// Device bytes a kept commitment key holds over `columns` commitment
    /// columns, at 8 bytes per key coefficient.
    pub fn commitment_key_bytes(columns: usize) -> usize {
        neo_ajtai::nightstream_fprime_setup::PRODUCTION_VERIFIER_ROWS as usize
            * columns
            * neo_math::D
            * size_of::<u64>()
    }

    pub fn activity(&self) -> CudaActivity {
        self.session.activity()
    }

    pub fn commit_production_prefixes<W: std::borrow::Borrow<Mat<F>>>(
        &self,
        witnesses: &[W],
    ) -> Result<Vec<neo_ajtai::Commitment>, PiCcsError> {
        self.session
            .commit_production_prefixes(witnesses)
            .map_err(oracle_error)
    }
}
