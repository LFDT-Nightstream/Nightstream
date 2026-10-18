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
    pub fn new() -> Result<Self, MetalError> {
        Ok(Self {
            session: MetalSession::new()?,
        })
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
