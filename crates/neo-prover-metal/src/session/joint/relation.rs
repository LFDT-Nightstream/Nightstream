//! Exact row satisfaction from verifier-owned matrices and a complete signed-unit witness.

use neo_ccs::{CcsStructure, Mat};
use neo_math::{D, F};
use neo_reductions::superneo_eval::SuperneoZBlocks;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{application, joint_term_metadata, nonempty, ApplicationTables, MetalJointMatrixPlan, MetalSession};
use crate::MetalError;

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
            let tables = self.build_joint_application_window(plan, &masks, 1, count, row_start)?;
            let shape = self.buffer_from_slice(&[count as u64, structure.f.terms().len() as u64, row_start as u64])?;
            let command = self.command_buffer("nightstream.terminal.relation")?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.ccs_first_unsatisfied_row);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&tables), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&headers), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&variables), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&first_failure), 0, 4);
            }
            self.dispatch(&encoder, &self.ccs_first_unsatisfied_row, count);
            encoder.endEncoding();
            self.finish(&command)?;
            let first = self.read_buffer::<u32>(&first_failure, 1)[0];
            if first > rows {
                return Err(MetalError::Execution("terminal row result exceeds the relation".into()));
            }
            if first < rows {
                return Ok(Some(first as usize));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/relation.rs"]
mod tests;
