//! Exact row satisfaction from verifier-owned matrices and a complete signed-unit witness.

use neo_ccs::{CcsStructure, Mat};
use neo_math::{D, F};
use neo_reductions::superneo_eval::SuperneoZBlocks;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{
    application, joint_term_metadata, nonempty, ApplicationTables, Buffer, MetalJointMatrixPlan, MetalSession,
    MetalWitnessMasks,
};
use crate::MetalError;

pub(super) struct TerminalRowCheck<'a> {
    structure: &'a CcsStructure<F>,
    masks: MetalWitnessMasks,
    headers: Buffer,
    variables: Buffer,
    first_failure: Buffer,
    rows: u32,
}

impl TerminalRowCheck<'_> {
    pub(super) fn workspace_bytes(&self, rows: usize) -> Result<usize, MetalError> {
        rows.checked_mul(self.structure.t())
            .and_then(|n| n.checked_mul(size_of::<F>()))
            .and_then(|n| n.checked_add(3 * size_of::<u64>()))
            .ok_or(MetalError::Shape("terminal row workspace overflow"))
    }

    pub(super) fn check_window(
        &self,
        session: &MetalSession,
        plan: &MetalJointMatrixPlan,
        window: &super::matrix_window::MetalMatrixWindow,
    ) -> Result<Option<usize>, MetalError> {
        let tables = session.build_application_from_matrix_window(plan, &self.masks, window)?;
        self.check_tables(session, &tables, window.rows.start, window.rows.len())
    }

    fn check_tables(
        &self,
        session: &MetalSession,
        tables: &Buffer,
        row_start: usize,
        count: usize,
    ) -> Result<Option<usize>, MetalError> {
        let shape =
            session.buffer_from_slice(&[count as u64, self.structure.f.terms().len() as u64, row_start as u64])?;
        let command = session.command_buffer("nightstream.terminal.relation")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&session.ccs_first_unsatisfied_row);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(tables), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.headers), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.variables), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&self.first_failure), 0, 4);
        }
        session.dispatch(&encoder, &session.ccs_first_unsatisfied_row, count);
        encoder.endEncoding();
        session.finish(&command)?;
        let first = session.read_buffer::<u32>(&self.first_failure, 1)[0];
        if first > self.rows {
            return Err(MetalError::Execution("terminal row result exceeds the relation".into()));
        }
        Ok((first < self.rows).then_some(first as usize))
    }
}

impl MetalSession {
    pub(crate) fn first_unsatisfied_row(
        &self,
        plan: &MetalJointMatrixPlan,
        structure: &CcsStructure<F>,
        witness: &Mat<F>,
    ) -> Result<Option<usize>, MetalError> {
        self.check_relation_windows(plan, structure, witness, None)
    }

    #[cfg(test)]
    pub(crate) fn first_unsatisfied_row_with_application_workspace(
        &self,
        plan: &MetalJointMatrixPlan,
        structure: &CcsStructure<F>,
        witness: &Mat<F>,
        workspace_bytes: usize,
    ) -> Result<Option<usize>, MetalError> {
        self.check_relation_windows(plan, structure, witness, Some(workspace_bytes))
    }

    fn check_relation_windows(
        &self,
        plan: &MetalJointMatrixPlan,
        structure: &CcsStructure<F>,
        witness: &Mat<F>,
        workspace: Option<usize>,
    ) -> Result<Option<usize>, MetalError> {
        let check = self.prepare_terminal_row_check(plan, structure, witness)?;
        let reserved = ApplicationTables::scratch_bytes(plan, 1)? + 3 * size_of::<u64>();
        let budget = application::available_workspace(self, reserved).min(workspace.unwrap_or(usize::MAX));
        let row_bytes = plan
            .matrix_count
            .checked_mul(size_of::<F>())
            .filter(|&n| n > 0)
            .ok_or(MetalError::Shape("terminal matrix row size overflow"))?;
        let capacity = (budget / row_bytes).min(plan.rows);
        if capacity == 0 {
            return Err(MetalError::MemoryLimit {
                requested: row_bytes,
                allocated: 0,
                limit: budget,
            });
        }
        for row_start in (0..plan.rows).step_by(capacity) {
            let count = capacity.min(plan.rows - row_start);
            // Exact verification never uses the prover's satisfied-row substitution.
            let tables = self.build_joint_application_window(plan, &check.masks, 1, count, row_start)?;
            if let Some(row) = check.check_tables(self, &tables, row_start, count)? {
                return Ok(Some(row));
            }
        }
        Ok(None)
    }
    pub(super) fn prepare_terminal_row_check<'a>(
        &self,
        plan: &MetalJointMatrixPlan,
        structure: &'a CcsStructure<F>,
        witness: &Mat<F>,
    ) -> Result<TerminalRowCheck<'a>, MetalError> {
        if structure.n != plan.rows
            || structure.m.div_ceil(D) != plan.blocks
            || structure.t() != plan.matrix_count
            || structure.f.arity() != plan.matrix_count
            || structure
                .f
                .terms()
                .iter()
                .any(|term| term.exps.len() != plan.matrix_count)
        {
            return Err(MetalError::Shape("terminal relation dimensions or polynomial arity"));
        }
        let rows = u32::try_from(plan.rows).map_err(|_| MetalError::Shape("terminal row index exceeds u32"))?;
        let blocks = SuperneoZBlocks::from_witness_mat(witness, structure.m)
            .map_err(|_| MetalError::Shape("terminal witness shape"))?;
        let words = blocks
            .signed_digit_masks(2)
            .ok_or(MetalError::Shape("terminal witness is not signed-unit"))?;
        let masks = self.prepare_witness_digit_masks(&words, 1, plan.blocks, 1, structure.m)?;
        drop(words);
        drop(blocks);
        let (headers, variables) =
            joint_term_metadata(structure).map_err(|_| MetalError::Shape("terminal polynomial metadata"))?;
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
