//! Device evaluation of circuit-owned rows, without a lifecycle-crate dependency.

use neo_ccs::{CcsStructure, Mat, V1_2Evaluations};
use neo_math::{F, K};
use neo_reductions::{
    optimized_engine::{PaperJointOracleBackend, PaperJointOracleInput, PaperJointRoundOracle},
    superneo_eval::MatrixRows,
    PiCcsError,
};

#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
use crate::session::MetalPaperJointOracle;
use crate::{oracle_error, MetalActivity, MetalError, MetalSession};

pub struct MetalRowProver {
    session: MetalSession,
}

impl MetalRowProver {
    /// A session that expands the needed commitment key rows in every call.
    pub fn new() -> Result<Self, MetalError> {
        Ok(Self {
            session: MetalSession::new()?,
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
    /// `MetalSession::production_key_weights`.
    pub fn production_key_weights(&self, powers: &[[u64; 3]], columns: usize) -> Result<Vec<[u64; 3]>, MetalError> {
        #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
        {
            self.session.production_key_weights(powers, columns)
        }
        #[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
        {
            let _ = (powers, columns);
            Err(MetalError::Unavailable)
        }
    }

    /// The device view of one GKR layer over the caller's tables; see
    /// `MetalSession::gkr_layer`.
    pub fn gkr_layer<'a>(
        &self,
        trees: Vec<Vec<&'a mut [u64]>>,
        eq: &'a mut [u64],
    ) -> Result<crate::MetalGkrLayer<'a>, MetalError> {
        self.session.gkr_layer(trees, eq)
    }

    pub fn gkr_message(
        &self,
        layer: &crate::MetalGkrLayer<'_>,
        weights: &[[u64; 3]],
        mu: [u64; 3],
        degree: usize,
    ) -> Result<Vec<[u64; 3]>, MetalError> {
        self.session.gkr_message(layer, weights, mu, degree)
    }

    pub fn gkr_fold(&self, layer: &mut crate::MetalGkrLayer<'_>, r: [u64; 3]) -> Result<(), MetalError> {
        self.session.gkr_fold(layer, r)
    }

    /// The DFT of every column of a row-major matrix; see
    /// `MetalSession::dft_columns`.
    pub fn dft_columns(&self, values: &mut [u64], width: usize, generator: u64) -> Result<(), MetalError> {
        self.session.dft_columns(values, width, generator)
    }

    /// Every digest layer of a Merkle tree; see
    /// `MetalSession::poseidon2_merkle_layers`.
    pub fn merkle_layers(
        &self,
        rows: usize,
        width: usize,
        fill: impl FnOnce(&mut [u64]),
    ) -> Result<Vec<Vec<[u64; 4]>>, MetalError> {
        self.session.poseidon2_merkle_layers(rows, width, fill)
    }

    /// The leaf digests of rows; see `MetalSession::poseidon2_merkle_leaves`.
    pub fn merkle_leaves(&self, values: &[u64], width: usize) -> Result<Vec<[u64; 4]>, MetalError> {
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

    pub fn activity(&self) -> MetalActivity {
        self.session.activity()
    }

    pub fn commit_production_prefixes<W: std::borrow::Borrow<Mat<F>>>(
        &self,
        witnesses: &[W],
    ) -> Result<Vec<neo_ajtai::Commitment>, PiCcsError> {
        #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
        {
            self.session
                .commit_production_prefixes(witnesses)
                .map_err(oracle_error)
        }
        #[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
        {
            let _ = witnesses;
            Err(oracle_error(MetalError::Unavailable))
        }
    }

    pub fn split_rlc_witnesses(
        &self,
        rhos: &[Mat<F>],
        witnesses: &[&Mat<F>],
        digits: usize,
        base: u32,
    ) -> Result<neo_reductions::common::SplitDigits, PiCcsError> {
        #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
        {
            self.session
                .split_rlc_witnesses(rhos, witnesses, digits, base)
                .map_err(oracle_error)
        }
        #[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
        {
            let _ = (rhos, witnesses, digits, base);
            Err(oracle_error(MetalError::Unavailable))
        }
    }

    pub fn child_openings(
        &mut self,
        rows: &dyn MatrixRows,
        workspace_bytes: usize,
        witnesses: &[Mat<F>],
        point: &[K],
        assignment_width: usize,
    ) -> Result<Vec<V1_2Evaluations<K>>, PiCcsError> {
        #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
        {
            let plan = self
                .session
                .prepare_joint_matrix_plan(rows, workspace_bytes)
                .map_err(oracle_error)?;
            self.session
                .eval_joint_dec_openings(&plan, witnesses, point, assignment_width)
                .map_err(oracle_error)?
                .ok_or_else(|| {
                    oracle_error(MetalError::Shape(
                        "device child openings are unavailable for this shape",
                    ))
                })
        }
        #[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
        {
            let _ = (rows, workspace_bytes, witnesses, point, assignment_width);
            Err(oracle_error(MetalError::Unavailable))
        }
    }

    pub fn first_unsatisfied_row(
        &mut self,
        rows: &dyn MatrixRows,
        workspace_bytes: usize,
        structure: &CcsStructure<F>,
        witness: &Mat<F>,
    ) -> Result<Option<usize>, PiCcsError> {
        #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
        {
            let plan = self
                .session
                .prepare_joint_matrix_plan(rows, workspace_bytes)
                .map_err(oracle_error)?;
            self.session
                .first_unsatisfied_row(&plan, structure, witness)
                .map_err(oracle_error)
        }
        #[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
        {
            let _ = (rows, workspace_bytes, structure, witness);
            Err(oracle_error(MetalError::Unavailable))
        }
    }

    pub fn evaluate_terminal_rows(
        &mut self,
        rows: &dyn MatrixRows,
        workspace_bytes: usize,
        structure: &CcsStructure<F>,
        running: &[Mat<F>],
        point: &[K],
        fresh: &Mat<F>,
    ) -> Result<neo_reductions::superneo_eval::TerminalEvaluations, PiCcsError> {
        #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
        {
            let plan = self
                .session
                .prepare_joint_matrix_plan(rows, workspace_bytes)
                .map_err(oracle_error)?;
            self.session
                .evaluate_terminal_rows(&plan, structure, running, point, fresh)
                .map_err(oracle_error)
        }
        #[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
        {
            let _ = (rows, workspace_bytes, structure, running, point, fresh);
            Err(oracle_error(MetalError::Unavailable))
        }
    }
}

impl PaperJointOracleBackend for MetalRowProver {
    fn create<'a>(
        &'a mut self,
        input: PaperJointOracleInput<'a>,
    ) -> Result<Box<dyn PaperJointRoundOracle + 'a>, PiCcsError> {
        #[cfg(all(target_vendor = "apple", neo_metal_shaders))]
        {
            let plan = self
                .session
                .prepare_joint_matrix_plan(input.rows, input.workspace_bytes)
                .map_err(oracle_error)?;
            Ok(Box::new(MetalPaperJointOracle::new(&self.session, plan, input)?))
        }
        #[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
        {
            let _ = input;
            Err(oracle_error(MetalError::Unavailable))
        }
    }
}
