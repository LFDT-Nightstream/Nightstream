//! Spartan circuit synthesis for the committed terminal direct-CCS step.

use super::*;

impl SpartanCircuit<NeoFoldDeciderEngine> for DirectCcsTerminalCommittedCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        let mut values = self
            .assignment
            .terminal_public_values
            .iter()
            .map(native_to_spartan)
            .collect::<Vec<_>>();
        values.extend(terminal_committed_boundary_public_values(&self.public_boundary));
        Ok(values)
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
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
        let boundary = self.alloc_public_boundary_inputs(cs)?;
        let (committed_width, packed_z) = self.allocate_committed_packed_z(cs)?;
        self.enforce_public_boundary(cs, &terminal_public_inputs, &boundary)?;
        self.enforce_public_commitment_shape(cs, &packed_z, &boundary)?;
        self.enforce_committed_image(cs, &terminal_public_inputs, &packed_z, committed_width)?;
        self.synthesize_terminal_with_committed_sources(cs, &terminal_public_inputs, &packed_z, committed_width)?;
        self.enforce_terminal_commitment(cs, &packed_z, &boundary.commitment_data)?;
        Ok(())
    }
}

impl DirectCcsTerminalCommittedCircuit {
    pub(super) fn alloc_public_boundary_inputs<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
    ) -> Result<Construction2TerminalBoundaryInputs, SynthesisError> {
        alloc_terminal_boundary_public_inputs(
            cs,
            "direct_terminal_boundary",
            &direct_terminal_boundary_view(&self.public_boundary),
        )
    }

    fn enforce_public_boundary<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        terminal_public_inputs: &[AllocatedNum<SpartanF>],
        boundary: &Construction2TerminalBoundaryInputs,
    ) -> Result<(), SynthesisError> {
        enforce_terminal_boundary_digests(
            cs,
            boundary,
            CONSTRUCTION2_COMMITMENT_RAW_TAG,
            CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG,
            "direct_terminal_boundary",
        )?;

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
                enforce_boolean_allocated(
                    &mut cs.namespace(|| format!("direct_terminal_x_i_public_bit_{public_idx}")),
                    &terminal_public_inputs[public_idx],
                    &format!("direct_terminal_x_i_public_bit_{public_idx}"),
                );
                packed = packed
                    + (
                        SpartanF::from_canonical_u64(1u64 << bit_idx),
                        terminal_public_inputs[public_idx].get_variable(),
                    );
            }
            cs.enforce(
                || format!("direct_terminal_boundary_x_i_limb_{limb_idx}_eq"),
                |_| packed,
                |lc| lc + CS::one(),
                |lc| lc + boundary.x_i[limb_idx].get_variable(),
            );
        }
        Ok(())
    }

    pub(super) fn enforce_public_commitment_shape<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        packed_z: &PackedWitnessVar,
        boundary: &Construction2TerminalBoundaryInputs,
    ) -> Result<(), SynthesisError> {
        enforce_public_commitment_shape(cs, packed_z, boundary, "direct_terminal_boundary")
    }

    pub(super) fn allocate_committed_packed_z<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
    ) -> Result<(usize, PackedWitnessVar), SynthesisError> {
        let full_width = self
            .assignment
            .committed_width()
            .map_err(|_| SynthesisError::Unsatisfiable)?;
        let packed_native = self
            .assignment
            .committed_packed_witness()
            .map_err(|_| SynthesisError::Unsatisfiable)?;
        let packed_cols = commit_cols_for_full_width(full_width);
        if packed_native.rows() != D || packed_native.cols() != packed_cols {
            return Err(SynthesisError::Unsatisfiable);
        }
        let packed_z = alloc_packed_mat_witness(
            &mut cs.namespace(|| "direct_terminal_r2_packed_z"),
            &packed_native,
            "direct_terminal_r2_packed_z",
        )?;
        Ok((full_width, packed_z))
    }

    fn enforce_committed_image<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        terminal_public_inputs: &[AllocatedNum<SpartanF>],
        packed_z: &PackedWitnessVar,
        committed_width: usize,
    ) -> Result<(), SynthesisError> {
        let x_range = self.assignment.terminal_circuit.construction2_x_bit_range();
        if x_range.len() != self.assignment.r2_public_values.len() || x_range.end > terminal_public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for public_idx in 0..self.assignment.r2_public_values.len() {
            let packed_entry = packed_z.logical_entry(committed_width, public_idx)?;
            cs.enforce(
                || format!("direct_terminal_r2_public_z_link_{public_idx}"),
                |lc| lc + packed_entry.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + terminal_public_inputs[x_range.start + public_idx].get_variable(),
            );
        }
        let constant_one_col = committed_width
            .checked_sub(1)
            .ok_or(SynthesisError::Unsatisfiable)?;
        let constant_one = packed_z.logical_entry(committed_width, constant_one_col)?;
        cs.enforce(
            || "direct_terminal_r2_constant_one_link",
            |lc| lc + constant_one.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + CS::one(),
        );

        for logical_col in 0..committed_width {
            let value = packed_z.logical_entry(committed_width, logical_col)?;
            enforce_boolean_allocated(
                &mut cs.namespace(|| format!("direct_terminal_r2_low_norm_bit_{logical_col}")),
                &value,
                &format!("direct_terminal_r2_low_norm_bit_{logical_col}"),
            );
        }
        enforce_packed_padding_zero(cs, packed_z, committed_width, "direct_terminal_r2_padding_zero")
    }

    pub(super) fn synthesize_terminal_with_committed_sources<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        public_inputs: &[AllocatedNum<SpartanF>],
        packed_z: &PackedWitnessVar,
        committed_width: usize,
    ) -> Result<usize, SynthesisError> {
        let mut linking_cs = DirectSourceWitnessLinkingCs::new(
            cs,
            &self.assignment.layout,
            packed_z,
            committed_width,
            self.assignment.r2_public_values.len(),
        );
        self.assignment
            .terminal_circuit
            .synthesize_body_with_public_inputs(&mut linking_cs, public_inputs)?;
        Ok(linking_cs.source_link_constraints)
    }

    pub(super) fn enforce_terminal_commitment<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        packed_z: &PackedWitnessVar,
        commitment_inputs: &[AllocatedNum<SpartanF>],
    ) -> Result<(), SynthesisError> {
        enforce_terminal_ajtai_commitment(
            &mut cs.namespace(|| "direct_terminal_r2_ajtai_commitment"),
            packed_z,
            commitment_inputs,
            "direct_terminal_r2_ajtai_commitment",
        )
    }
}
