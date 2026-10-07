//! Device evaluation of circuit-owned rows, without a lifecycle-crate dependency.

use neo_ccs::{CcsStructure, Mat, V1_1Evaluations};
use neo_math::{F, K};
use neo_reductions::{
    optimized_engine::{PaperJointOracleBackend, PaperJointOracleInput, PaperJointRoundOracle},
    superneo_eval::MatrixRows,
    PiCcsError,
};

#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
use crate::session::MetalPaperJointOracle;
use crate::{oracle_error, MetalActivity, MetalError, MetalSession};

/// Retains the device session; each operation loads bounded original-row windows.
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

    /// A prover session: the commitment key rows stay on the device after the
    /// first commitment, at 8 bytes per key coefficient.
    pub fn with_resident_commitment_key() -> Result<Self, MetalError> {
        let mut session = MetalSession::new()?;
        session.keep_production_key();
        Ok(Self { session })
    }

    /// Device bytes the kept commitment key holds over `columns` commitment
    /// columns; zero for a session that streams key rows.
    pub fn kept_commitment_key_bytes(&self, columns: usize) -> usize {
        self.session.kept_production_key_bytes(columns)
    }

    pub fn activity(&self) -> MetalActivity {
        self.session.activity()
    }

    /// Commit complete signed-unit witnesses with the verifier-owned indexed key.
    /// Key generation, ring products, and reduction run on the device.
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

    /// Mix the Π_RLC witnesses `Σ ρ_i·Z_i` and split the parent into `digits`
    /// balanced base-`base` planes on the device; the parent stays there.
    /// Returns the planes and their nonzero flags. Unsupported inputs return
    /// an error; callers do not receive an implicit host fallback.
    pub fn split_rlc_witnesses(
        &self,
        rhos: &[Mat<F>],
        witnesses: &[&Mat<F>],
        digits: usize,
        base: u32,
    ) -> Result<(Vec<Mat<F>>, Vec<bool>), PiCcsError> {
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

    /// Compute every supplied child opening on the device. Unsupported shapes
    /// return an error; callers do not receive an implicit host fallback.
    pub fn child_openings(
        &mut self,
        rows: &dyn MatrixRows,
        workspace_bytes: usize,
        witnesses: &[Mat<F>],
        point: &[K],
        assignment_width: usize,
    ) -> Result<Vec<V1_1Evaluations<K>>, PiCcsError> {
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

    /// Check every row against a complete signed-unit witness on the device.
    /// `None` means all rows satisfy the polynomial; errors do not trigger a host fallback.
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

    /// Compute running openings and check the fresh relation while each
    /// original matrix window remains loaded. Both operations run on Metal.
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
