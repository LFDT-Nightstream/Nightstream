//! Shape measurement for the committed terminal direct-CCS circuit.

use super::*;

impl DirectCcsTerminalCommittedCircuit {
    pub(super) fn measure_with_breakdown(
        &self,
        cs: &mut ShapeCS<NeoFoldDeciderEngine>,
    ) -> Result<DirectCcsTerminalCommittedConstraintBreakdown, SynthesisError> {
        let start = shape_point(cs);
        let mut out = DirectCcsTerminalCommittedConstraintBreakdown::default();

        let before = shape_point(cs);
        let terminal_public_inputs = self
            .assignment
            .terminal_public_values
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                AllocatedNum::alloc_input(cs.namespace(|| format!("terminal_public_{idx}")), || {
                    Ok(native_to_spartan(value))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.public_input_alloc_shape = shape_delta(before, cs);
        out.public_input_alloc = out.public_input_alloc_shape.rows;

        let before = shape_point(cs);
        let boundary = self.alloc_public_boundary_inputs(cs)?;
        out.boundary_input_alloc_shape = shape_delta(before, cs);
        out.boundary_input_alloc = out.boundary_input_alloc_shape.rows;

        let before = shape_point(cs);
        let (committed_width, packed_z) = self.allocate_committed_packed_z(cs)?;
        out.packed_witness_alloc_shape = shape_delta(before, cs);
        out.packed_witness_alloc = out.packed_witness_alloc_shape.rows;

        out.public_boundary = self.measure_public_boundary(cs, &terminal_public_inputs, &boundary)?;

        let before = shape_point(cs);
        self.enforce_public_commitment_shape(cs, &packed_z, &boundary)?;
        out.public_commitment_shape_shape = shape_delta(before, cs);
        out.public_commitment_shape = out.public_commitment_shape_shape.rows;

        out.committed_image = self.measure_committed_image(cs, &terminal_public_inputs, &packed_z, committed_width)?;

        let before = shape_point(cs);
        out.terminal_body_source_links =
            self.synthesize_terminal_with_committed_sources(cs, &terminal_public_inputs, &packed_z, committed_width)?;
        out.terminal_body_shape = shape_delta(before, cs);
        out.terminal_body_with_sources = out.terminal_body_shape.rows;
        out.terminal_body_without_source_links = out
            .terminal_body_with_sources
            .saturating_sub(out.terminal_body_source_links);

        let before = shape_point(cs);
        self.enforce_terminal_commitment(cs, &packed_z, &boundary.commitment_data)?;
        out.terminal_ajtai_commitment_shape = shape_delta(before, cs);
        out.terminal_ajtai_commitment = out.terminal_ajtai_commitment_shape.rows;
        out.total_shape = shape_delta(start, cs);
        out.total = out.total_shape.rows;
        Ok(out)
    }

    fn measure_public_boundary(
        &self,
        cs: &mut ShapeCS<NeoFoldDeciderEngine>,
        terminal_public_inputs: &[AllocatedNum<SpartanF>],
        boundary: &Construction2TerminalBoundaryInputs,
    ) -> Result<DirectCcsPublicBoundaryConstraintBreakdown, SynthesisError> {
        let start = shape_point(cs);
        let mut out = DirectCcsPublicBoundaryConstraintBreakdown::default();

        let before = shape_point(cs);
        enforce_terminal_boundary_digests(
            cs,
            boundary,
            CONSTRUCTION2_COMMITMENT_RAW_TAG,
            CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG,
            "direct_terminal_boundary",
        )?;
        out.digest_checks_shape = shape_delta(before, cs);
        out.digest_checks = out.digest_checks_shape.rows;

        let x_range = self.assignment.terminal_circuit.construction2_x_bit_range();
        if x_range.len() != CONSTRUCTION2_ENC_INST_BITS || x_range.end > terminal_public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for limb_idx in 0usize..4 {
            let mut packed = LinearCombination::<SpartanF>::zero();
            for bit_idx in 0usize..64 {
                let bit_offset = limb_idx
                    .checked_mul(64)
                    .and_then(|value| value.checked_add(bit_idx))
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let public_idx = x_range
                    .start
                    .checked_add(bit_offset)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let before = shape_point(cs);
                enforce_boolean_allocated(
                    &mut cs.namespace(|| format!("direct_terminal_x_i_public_bit_{public_idx}")),
                    &terminal_public_inputs[public_idx],
                    &format!("direct_terminal_x_i_public_bit_{public_idx}"),
                );
                let delta = shape_delta(before, cs);
                out.x_i_bit_checks += delta.rows;
                out.x_i_bit_checks_shape.rows += delta.rows;
                out.x_i_bit_checks_shape.public_cols += delta.public_cols;
                out.x_i_bit_checks_shape.aux_cols += delta.aux_cols;
                packed = packed
                    + (
                        SpartanF::from_canonical_u64(1u64 << bit_idx),
                        terminal_public_inputs[public_idx].get_variable(),
                    );
            }
            let before = shape_point(cs);
            cs.enforce(
                || format!("direct_terminal_boundary_x_i_limb_{limb_idx}_eq"),
                |_| packed,
                |lc| lc + <ShapeCS<NeoFoldDeciderEngine> as ConstraintSystem<SpartanF>>::one(),
                |lc| lc + boundary.x_i[limb_idx].get_variable(),
            );
            let delta = shape_delta(before, cs);
            out.x_i_limb_links += delta.rows;
            out.x_i_limb_links_shape.rows += delta.rows;
            out.x_i_limb_links_shape.public_cols += delta.public_cols;
            out.x_i_limb_links_shape.aux_cols += delta.aux_cols;
        }
        out.total_shape = shape_delta(start, cs);
        out.total = out.total_shape.rows;
        Ok(out)
    }

    fn measure_committed_image(
        &self,
        cs: &mut ShapeCS<NeoFoldDeciderEngine>,
        terminal_public_inputs: &[AllocatedNum<SpartanF>],
        packed_z: &PackedWitnessVar,
        committed_width: usize,
    ) -> Result<DirectCcsCommittedImageConstraintBreakdown, SynthesisError> {
        let start = shape_point(cs);
        let mut out = DirectCcsCommittedImageConstraintBreakdown::default();

        let x_range = self.assignment.terminal_circuit.construction2_x_bit_range();
        if x_range.len() != self.assignment.r2_public_values.len() || x_range.end > terminal_public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for public_idx in 0..self.assignment.r2_public_values.len() {
            let packed_entry = packed_z.logical_entry(committed_width, public_idx)?;
            let before = shape_point(cs);
            cs.enforce(
                || format!("direct_terminal_r2_public_z_link_{public_idx}"),
                |lc| lc + packed_entry.get_variable(),
                |lc| lc + <ShapeCS<NeoFoldDeciderEngine> as ConstraintSystem<SpartanF>>::one(),
                |lc| lc + terminal_public_inputs[x_range.start + public_idx].get_variable(),
            );
            let delta = shape_delta(before, cs);
            out.public_z_links += delta.rows;
            out.public_z_links_shape.rows += delta.rows;
            out.public_z_links_shape.public_cols += delta.public_cols;
            out.public_z_links_shape.aux_cols += delta.aux_cols;
        }
        let constant_one_col = committed_width
            .checked_sub(1)
            .ok_or(SynthesisError::Unsatisfiable)?;
        let constant_one = packed_z.logical_entry(committed_width, constant_one_col)?;
        let before = shape_point(cs);
        cs.enforce(
            || "direct_terminal_r2_constant_one_link",
            |lc| lc + constant_one.get_variable(),
            |lc| lc + <ShapeCS<NeoFoldDeciderEngine> as ConstraintSystem<SpartanF>>::one(),
            |lc| lc + <ShapeCS<NeoFoldDeciderEngine> as ConstraintSystem<SpartanF>>::one(),
        );
        out.constant_one_link_shape = shape_delta(before, cs);
        out.constant_one_link = out.constant_one_link_shape.rows;

        for logical_col in 0..committed_width {
            let value = packed_z.logical_entry(committed_width, logical_col)?;
            let before = shape_point(cs);
            enforce_boolean_allocated(
                &mut cs.namespace(|| format!("direct_terminal_r2_low_norm_bit_{logical_col}")),
                &value,
                &format!("direct_terminal_r2_low_norm_bit_{logical_col}"),
            );
            let delta = shape_delta(before, cs);
            out.low_norm_bit_checks += delta.rows;
            out.low_norm_bit_checks_shape.rows += delta.rows;
            out.low_norm_bit_checks_shape.public_cols += delta.public_cols;
            out.low_norm_bit_checks_shape.aux_cols += delta.aux_cols;
        }
        let before = shape_point(cs);
        enforce_packed_padding_zero(cs, packed_z, committed_width, "direct_terminal_r2_padding_zero")?;
        out.padding_zero_checks_shape = shape_delta(before, cs);
        out.padding_zero_checks = out.padding_zero_checks_shape.rows;
        out.total_shape = shape_delta(start, cs);
        out.total = out.total_shape.rows;
        Ok(out)
    }
}
