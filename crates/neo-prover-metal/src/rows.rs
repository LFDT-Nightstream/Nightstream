//! Device evaluation of circuit-owned rows, without a lifecycle-crate dependency.

use neo_ccs::{CcsStructure, Mat, V1_2Evaluations};
use neo_math::{F, K};
use neo_reductions::{
    optimized_engine::{PaperJointOracleBackend, PaperJointOracleInput, PaperJointRoundOracle},
    superneo_eval::MatrixRows,
    PiCcsError,
};

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
        self.session
            .commit_production_prefixes(witnesses)
            .map_err(oracle_error)
    }

    /// Π_RLC's parent witness `Σ ρ_i·Z_i`; see
    /// `MetalSession::mix_rlc_witnesses`.
    pub fn mix_rlc_witnesses(&self, rhos: &[Mat<F>], witnesses: &[&Mat<F>]) -> Result<Mat<F>, PiCcsError> {
        self.session
            .mix_rlc_witnesses(rhos, witnesses)
            .map_err(oracle_error)
    }

    pub fn split_rlc_witnesses(
        &self,
        rhos: &[Mat<F>],
        witnesses: &[&Mat<F>],
        digits: usize,
        base: u32,
    ) -> Result<neo_reductions::common::SplitDigits, PiCcsError> {
        self.session
            .split_rlc_witnesses(rhos, witnesses, digits, base)
            .map_err(oracle_error)
    }

    pub fn child_openings(
        &mut self,
        rows: &dyn MatrixRows,
        workspace_bytes: usize,
        witnesses: &[Mat<F>],
        point: &[K],
        assignment_width: usize,
    ) -> Result<Vec<V1_2Evaluations<K>>, PiCcsError> {
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

    /// The first row of `structure` over `rows` that the signed-unit
    /// `witness` fails, or `None` when every row holds.
    pub fn first_unsatisfied_row(
        &self,
        rows: &dyn MatrixRows,
        workspace_bytes: usize,
        structure: &CcsStructure<F>,
        witness: &Mat<F>,
    ) -> Result<Option<usize>, PiCcsError> {
        let plan = self
            .session
            .prepare_joint_matrix_plan(rows, workspace_bytes)
            .map_err(oracle_error)?;
        self.session
            .first_unsatisfied_row(&plan, structure, witness)
            .map_err(oracle_error)
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
        let plan = self
            .session
            .prepare_joint_matrix_plan(rows, workspace_bytes)
            .map_err(oracle_error)?;
        self.session
            .evaluate_terminal_rows(&plan, structure, running, point, fresh)
            .map_err(oracle_error)
    }
}

impl PaperJointOracleBackend for MetalRowProver {
    fn create<'a>(
        &'a mut self,
        input: PaperJointOracleInput<'a>,
    ) -> Result<Box<dyn PaperJointRoundOracle + 'a>, PiCcsError> {
        let plan = self
            .session
            .prepare_joint_matrix_plan(input.rows, input.workspace_bytes)
            .map_err(oracle_error)?;
        Ok(Box::new(MetalPaperJointOracle::new(&self.session, plan, input)?))
    }
}
