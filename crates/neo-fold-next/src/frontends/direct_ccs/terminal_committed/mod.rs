//! Owns the direct-CCS terminal Construction-2 committed-step proof.
//!
//! This is the non-VM analogue of the RV32IM terminal `F'` committed-step
//! boundary: the public output is `u_i = (C_i, x_i)`, where `C_i` opens to a
//! low-norm SuperNeo-packed source image linked to the latest direct-CCS F'
//! terminal circuit.

mod assignment;
mod commitment;
mod perf;
mod proof;
mod source_linking;
mod types;

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError};
use neo_math::D;

use super::ivc::DirectCcsTerminalFPrimeCircuit;
use crate::construction2::{
    Construction2Commitment, Construction2FreshInstance, Construction2PublicBoundary, CONSTRUCTION2_COMMITMENT_RAW_TAG,
    CONSTRUCTION2_ENC_INST_BITS, CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG,
};
use crate::construction2_terminal::{
    alloc_terminal_boundary_public_inputs, enforce_boolean_allocated, enforce_packed_padding_zero,
    enforce_public_commitment_shape, enforce_terminal_ajtai_commitment, enforce_terminal_boundary_digests,
    native_to_spartan, Construction2TerminalBoundaryInputs, TerminalPrivateColumnEncoding,
};
use crate::spartan_backend::{NeoFoldDeciderEngine, ShapeCS, SpartanCircuit, SpartanF};
use crate::superneo_circuit::witness::{alloc_packed_mat_witness, PackedWitnessVar};
use crate::witness_layout::commit_cols_for_full_width;

use commitment::{direct_terminal_boundary_view, terminal_committed_boundary_public_values};
use perf::{shape_delta, shape_point};
pub(crate) use proof::{
    prove_direct_ccs_terminal_committed_relation, setup_direct_ccs_terminal_committed_relation_cached,
    verify_direct_ccs_terminal_committed_relation,
};
use source_linking::DirectSourceWitnessLinkingCs;
pub use types::DirectCcsTerminalCommittedConstraintBreakdown;
use types::{
    DirectCcsCommittedImageConstraintBreakdown, DirectCcsPublicBoundaryConstraintBreakdown,
    DirectCcsTerminalCommittedCircuit, DirectCcsTerminalR2Assignment, SimpleKernelError,
};
pub(crate) use types::{
    DirectCcsTerminalCommittedKeyPair, DirectCcsTerminalCommittedPerf, DirectCcsTerminalCommittedProof,
    DirectCcsTerminalCommittedRelation, DirectCcsTerminalError,
};

impl DirectCcsTerminalCommittedRelation {
    pub(crate) fn from_terminal_circuit(circuit: DirectCcsTerminalFPrimeCircuit) -> Result<Self, SimpleKernelError> {
        let assignment = DirectCcsTerminalR2Assignment::from_terminal_circuit(circuit)?;
        let commitment = assignment.commitment()?;
        let fresh_instance = Construction2FreshInstance::from_parts(
            Construction2Commitment::from_commitment(commitment),
            assignment.terminal_circuit.construction2_x_i()?,
        );
        let public_boundary = Construction2PublicBoundary::from_fresh_instance(&fresh_instance);
        Ok(Self {
            public_boundary,
            assignment,
        })
    }

    pub(crate) fn public_boundary(&self) -> &Construction2PublicBoundary {
        &self.public_boundary
    }

    pub(crate) fn committed_circuit(&self) -> DirectCcsTerminalCommittedCircuit {
        DirectCcsTerminalCommittedCircuit {
            public_boundary: self.public_boundary.clone(),
            assignment: self.assignment.clone(),
        }
    }

    pub(crate) fn measure(&self) -> Result<DirectCcsTerminalCommittedPerf, SimpleKernelError> {
        let circuit = self.committed_circuit();
        let public_inputs = circuit
            .public_values()
            .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed public IO failed: {err}")))?
            .len();
        let mut cs = ShapeCS::<NeoFoldDeciderEngine>::new();
        let breakdown = circuit
            .measure_with_breakdown(&mut cs)
            .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed shape failed: {err}")))?;
        Ok(DirectCcsTerminalCommittedPerf {
            constraints: cs.num_constraints(),
            public_inputs,
            committed_width: self.assignment.committed_width()?,
            commitment_words: self.public_boundary.commitment_data.len(),
            source_values: self.assignment.layout.source_labels.len(),
            source_bit_values: self
                .assignment
                .layout
                .source_encoding_count(TerminalPrivateColumnEncoding::Bit),
            source_u32_values: self
                .assignment
                .layout
                .source_encoding_count(TerminalPrivateColumnEncoding::U32),
            source_u64_values: self
                .assignment
                .layout
                .source_encoding_count(TerminalPrivateColumnEncoding::U64),
            unclassified_private_values: 0,
            breakdown,
            sizes: [0; 10],
            nnz: 0,
        })
    }
}

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
    fn measure_with_breakdown(
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

    fn alloc_public_boundary_inputs<CS: ConstraintSystem<SpartanF>>(
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

    fn enforce_public_commitment_shape<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        packed_z: &PackedWitnessVar,
        boundary: &Construction2TerminalBoundaryInputs,
    ) -> Result<(), SynthesisError> {
        enforce_public_commitment_shape(cs, packed_z, boundary, "direct_terminal_boundary")
    }

    fn allocate_committed_packed_z<CS: ConstraintSystem<SpartanF>>(
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

    fn synthesize_terminal_with_committed_sources<CS: ConstraintSystem<SpartanF>>(
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

    fn enforce_terminal_commitment<CS: ConstraintSystem<SpartanF>>(
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
