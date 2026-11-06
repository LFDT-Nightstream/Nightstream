//! Exact row satisfaction from verifier-owned matrices and a complete signed-unit witness.

use neo_ccs::{CcsStructure, Mat};
use neo_math::{D, F};
use neo_reductions::superneo_eval::SuperneoZBlocks;

use super::{joint_term_metadata, nonempty, Buffer, CudaJointMatrixPlan, CudaSession, CudaWitnessMasks, Grid};
use crate::CudaError;

pub(super) struct TerminalRowCheck<'a> {
    structure: &'a CcsStructure<F>,
    masks: CudaWitnessMasks,
    headers: Buffer,
    variables: Buffer,
    first_failure: Buffer,
    rows: u32,
}

impl TerminalRowCheck<'_> {
    pub(super) fn workspace_bytes(&self, rows: usize) -> Result<usize, CudaError> {
        rows.checked_mul(self.structure.t())
            .and_then(|n| n.checked_mul(size_of::<F>()))
            .and_then(|n| n.checked_add(3 * size_of::<u64>()))
            .ok_or(CudaError::Shape("terminal row workspace overflow"))
    }

    pub(super) fn check_window(
        &self,
        session: &CudaSession,
        plan: &CudaJointMatrixPlan,
        window: &super::matrix_window::CudaMatrixWindow,
    ) -> Result<Option<usize>, CudaError> {
        let tables = session.build_application_from_matrix_window(plan, &self.masks, window)?;
        self.check_tables(session, &tables, window.rows.start, window.rows.len())
    }

    fn check_tables(
        &self,
        session: &CudaSession,
        tables: &Buffer,
        row_start: usize,
        count: usize,
    ) -> Result<Option<usize>, CudaError> {
        let shape = [count as u64, self.structure.f().terms().len() as u64, row_start as u64];
        let kernels = &session.kernels.relation;
        session.launch(Grid::Threads(count), |stream, config| {
            kernels.ccs_first_unsatisfied_row(
                stream,
                config,
                tables.ptr(),
                self.headers.ptr(),
                self.variables.ptr(),
                shape,
                self.first_failure.ptr(),
            )
        })?;
        session.finish()?;
        let first = session.read_buffer::<u32>(&self.first_failure, 1)?[0];
        if first > self.rows {
            return Err(CudaError::Execution("terminal row result exceeds the relation".into()));
        }
        Ok((first < self.rows).then_some(first as usize))
    }
}

impl CudaSession {
    /// The first row of `structure` that the signed-unit `witness` fails.
    pub(crate) fn first_unsatisfied_row(
        &self,
        plan: &CudaJointMatrixPlan,
        structure: &CcsStructure<F>,
        witness: &Mat<F>,
    ) -> Result<Option<usize>, CudaError> {
        self.check_relation_windows(plan, structure, witness, None)
    }

    #[cfg(test)]
    pub(crate) fn first_unsatisfied_row_with_application_workspace(
        &self,
        plan: &CudaJointMatrixPlan,
        structure: &CcsStructure<F>,
        witness: &Mat<F>,
        workspace_bytes: usize,
    ) -> Result<Option<usize>, CudaError> {
        self.check_relation_windows(plan, structure, witness, Some(workspace_bytes))
    }

    fn check_relation_windows(
        &self,
        plan: &CudaJointMatrixPlan,
        structure: &CcsStructure<F>,
        witness: &Mat<F>,
        workspace: Option<usize>,
    ) -> Result<Option<usize>, CudaError> {
        let check = self.prepare_terminal_row_check(plan, structure, witness)?;
        let reserved = super::ApplicationTables::scratch_bytes(plan, 1)? + 3 * size_of::<u64>();
        let budget = super::application::available_workspace(self, reserved).min(workspace.unwrap_or(usize::MAX));
        let row_bytes = plan
            .matrix_count
            .checked_mul(size_of::<F>())
            .filter(|&n| n > 0)
            .ok_or(CudaError::Shape("terminal matrix row size overflow"))?;
        let capacity = (budget / row_bytes).min(plan.rows);
        if capacity == 0 {
            return Err(CudaError::MemoryLimit {
                requested: row_bytes,
                allocated: 0,
                limit: budget,
            });
        }
        for row_start in (0..plan.rows).step_by(capacity) {
            let count = capacity.min(plan.rows - row_start);
            let tables = self.build_joint_application_window(plan, &check.masks, 1, count, row_start)?;
            if let Some(row) = check.check_tables(self, &tables, row_start, count)? {
                return Ok(Some(row));
            }
        }
        Ok(None)
    }
    pub(super) fn prepare_terminal_row_check<'a>(
        &self,
        plan: &CudaJointMatrixPlan,
        structure: &'a CcsStructure<F>,
        witness: &Mat<F>,
    ) -> Result<TerminalRowCheck<'a>, CudaError> {
        if structure.n() != plan.rows
            || structure.m().div_ceil(D) != plan.blocks
            || structure.t() != plan.matrix_count
            || structure.f().arity() != plan.matrix_count
            || structure
                .f()
                .terms()
                .iter()
                .any(|term| term.exps.len() != plan.matrix_count)
        {
            return Err(CudaError::Shape("terminal relation dimensions or polynomial arity"));
        }
        let rows = u32::try_from(plan.rows).map_err(|_| CudaError::Shape("terminal row index exceeds u32"))?;
        let blocks = SuperneoZBlocks::from_witness_mat(witness, structure.m())
            .map_err(|_| CudaError::Shape("terminal witness shape"))?;
        let words = blocks
            .signed_digit_masks(2)
            .ok_or(CudaError::Shape("terminal witness is not signed-unit"))?;
        let masks = self.prepare_witness_digit_masks(&words, 1, plan.blocks, 1, structure.m())?;
        drop(words);
        drop(blocks);
        let (headers, variables) =
            joint_term_metadata(structure).map_err(|_| CudaError::Shape("terminal polynomial metadata"))?;
        let headers = self.buffer_from_slice(nonempty(&headers))?;
        let variables = self.buffer_from_slice(nonempty(&variables))?;
        let first_failure = self.buffer_from_slice(&[rows])?;
        Ok(TerminalRowCheck {
            structure,
            masks,
            headers,
            variables,
            first_failure,
            rows,
        })
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/relation.rs"]
mod tests;
