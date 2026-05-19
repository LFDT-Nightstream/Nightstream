//! Owns the Spartan circuit for the terminal committed `F'` R2 proof.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError, Variable};
use neo_math::D;

use crate::construction2::terminal::{
    alloc_terminal_boundary_public_inputs, enforce_boolean_allocated, enforce_packed_padding_zero,
    enforce_public_commitment_shape, enforce_terminal_ajtai_commitment, enforce_terminal_boundary_digests,
    native_to_spartan, terminal_boundary_public_values, Construction2TerminalBoundaryInputs,
    Construction2TerminalBoundaryView,
};
use crate::rv32im::construction2::Rv32imMainRecursionConstruction2PublicBoundary;
use crate::rv32im::construction2::{
    RV32IM_CONSTRUCTION2_COMMITMENT_RAW_TAG, RV32IM_CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG,
};
use crate::rv32im::main_relation_spartan::{
    enforce_digest_eq, synthesize_rv32im_main_recursion_step_body, Rv32imMainRecursionStepSpartanPublishedTarget,
};
use crate::superneo_circuit::witness::{alloc_packed_mat_witness, PackedWitnessVar};
use crate::witness_layout::commit_cols_for_full_width;

use super::{
    Rv32imDeciderEngine, Rv32imTerminalFPrimeCommittedStepCircuit, Rv32imTerminalFPrimePrivateColumnEncoding,
    Rv32imTerminalFPrimeR2ColumnLayout, SpartanCircuit, SpartanF,
};

impl SpartanCircuit<Rv32imDeciderEngine> for Rv32imTerminalFPrimeCommittedStepCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        let boundary_values = terminal_f_prime_committed_step_boundary_public_values(&self.public_boundary);
        let mut values = Vec::with_capacity(
            self.assignment
                .terminal_public_values
                .len()
                .checked_add(boundary_values.len())
                .ok_or(SynthesisError::Unsatisfiable)?,
        );
        values.extend(
            self.assignment
                .terminal_public_values
                .iter()
                .map(native_to_spartan),
        );
        values.extend(boundary_values);
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
        let public_inputs = self
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
        let boundary_inputs = self.alloc_public_boundary_inputs(cs)?;
        let (committed_width, packed_z) = self.allocate_committed_packed_z(cs)?;
        self.enforce_public_boundary(cs, &public_inputs, &boundary_inputs)?;
        self.enforce_public_commitment_shape(cs, &packed_z, &boundary_inputs)?;
        self.enforce_committed_superneo_image(cs, &public_inputs, &packed_z, committed_width)?;
        self.synthesize_terminal_f_prime_with_committed_sources(cs, &public_inputs, &packed_z, committed_width)?;
        self.enforce_terminal_commitment(cs, &packed_z, &boundary_inputs.commitment_data)?;
        Ok(())
    }
}

impl Rv32imTerminalFPrimeCommittedStepCircuit {
    fn alloc_public_boundary_inputs<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
    ) -> Result<Construction2TerminalBoundaryInputs, SynthesisError> {
        alloc_terminal_boundary_public_inputs(
            cs,
            "terminal_boundary",
            &rv32im_terminal_boundary_view(&self.public_boundary),
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
            RV32IM_CONSTRUCTION2_COMMITMENT_RAW_TAG,
            RV32IM_CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG,
            "terminal_boundary",
        )?;

        let x_out_digest_start = Rv32imMainRecursionStepSpartanPublishedTarget::terminal_r2_public_value_range_static()
            .start
            .checked_sub(4)
            .ok_or(SynthesisError::Unsatisfiable)?;
        if x_out_digest_start + 4 > terminal_public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        let x_out_digest = [
            terminal_public_inputs[x_out_digest_start].clone(),
            terminal_public_inputs[x_out_digest_start + 1].clone(),
            terminal_public_inputs[x_out_digest_start + 2].clone(),
            terminal_public_inputs[x_out_digest_start + 3].clone(),
        ];
        enforce_digest_eq(
            &mut cs.namespace(|| "terminal_boundary_x_i_eq"),
            &boundary.x_i,
            &x_out_digest,
            "terminal_boundary_x_i_eq",
        )
    }

    fn enforce_public_commitment_shape<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        packed_z: &PackedWitnessVar,
        boundary: &Construction2TerminalBoundaryInputs,
    ) -> Result<(), SynthesisError> {
        enforce_public_commitment_shape(cs, packed_z, boundary, "terminal_boundary")
    }

    fn allocate_committed_packed_z<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
    ) -> Result<(usize, PackedWitnessVar), SynthesisError> {
        let full_width = self
            .assignment
            .committed_full_width()
            .map_err(|_| SynthesisError::Unsatisfiable)?;
        if full_width == 0 {
            return Err(SynthesisError::Unsatisfiable);
        }
        let (_params, packed_native) = self
            .assignment
            .committed_packed_witness()
            .map_err(|_| SynthesisError::Unsatisfiable)?;
        let packed_cols = commit_cols_for_full_width(full_width);
        if packed_native.rows() != D || packed_native.cols() != packed_cols {
            return Err(SynthesisError::Unsatisfiable);
        }

        let packed_z = alloc_packed_mat_witness(
            &mut cs.namespace(|| "terminal_r2_packed_z"),
            &packed_native,
            "terminal_r2_packed_z",
        )?;
        Ok((full_width, packed_z))
    }

    fn enforce_committed_superneo_image<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        public_inputs: &[AllocatedNum<SpartanF>],
        packed_z: &PackedWitnessVar,
        committed_width: usize,
    ) -> Result<(), SynthesisError> {
        let public_len = self.assignment.r2_public_values.len();
        let committed_witness_len = self.assignment.committed_witness_values().len();
        if committed_width
            != public_len
                .checked_add(committed_witness_len)
                .ok_or(SynthesisError::Unsatisfiable)?
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        let r2_public_range = Rv32imMainRecursionStepSpartanPublishedTarget::terminal_r2_public_value_range_static();
        if r2_public_range.end > public_inputs.len() || r2_public_range.len() != public_len {
            return Err(SynthesisError::Unsatisfiable);
        }
        for public_idx in 0..public_len {
            let packed_entry = packed_z.logical_entry(committed_width, public_idx)?;
            let expected = public_inputs[r2_public_range.start + public_idx].get_variable();
            cs.enforce(
                || format!("terminal_r2_superneo_public_z_link_{public_idx}"),
                |lc| lc + packed_entry.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + expected,
            );
        }
        let constant_one_col = committed_width
            .checked_sub(1)
            .ok_or(SynthesisError::Unsatisfiable)?;
        let constant_one = packed_z.logical_entry(committed_width, constant_one_col)?;
        cs.enforce(
            || "terminal_r2_superneo_constant_one_link",
            |lc| lc + constant_one.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + CS::one(),
        );
        self.enforce_committed_low_norm_bounds(
            &mut cs.namespace(|| "terminal_r2_superneo_low_norm_bound"),
            packed_z,
            committed_width,
        )?;
        enforce_packed_padding_zero(cs, packed_z, committed_width, "terminal_r2_superneo_padding_zero")
    }

    fn synthesize_terminal_f_prime_with_committed_sources<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        public_inputs: &[AllocatedNum<SpartanF>],
        packed_z: &PackedWitnessVar,
        committed_width: usize,
    ) -> Result<(), SynthesisError> {
        let mut linking_cs = SourceWitnessLinkingCs::new(
            cs,
            &self.assignment.relation().layout,
            packed_z,
            committed_width,
            self.assignment.r2_public_values.len(),
        );
        let mut public_cursor = 0usize;
        synthesize_rv32im_main_recursion_step_body(
            &self.f_prime_circuit,
            &mut linking_cs,
            public_inputs,
            &mut public_cursor,
            None,
        )?;
        if public_cursor != public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(())
    }

    fn enforce_committed_low_norm_bounds<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        packed_z: &PackedWitnessVar,
        committed_width: usize,
    ) -> Result<(), SynthesisError> {
        let public_len = self.assignment.r2_public_values.len();
        for public_idx in 0..public_len {
            let bit = packed_z.logical_entry(committed_width, public_idx)?;
            enforce_boolean_allocated(
                &mut cs.namespace(|| format!("terminal_r2_public_bit_bound_{public_idx}")),
                &bit,
                &format!("terminal_r2_public_bit_bound_{public_idx}"),
            );
        }

        for witness_idx in 0..self.assignment.relation.layout.source_encodings.len() {
            let encoding = self
                .assignment
                .relation
                .layout
                .witness_encoding(witness_idx)
                .map_err(|_| SynthesisError::Unsatisfiable)?;
            let committed_start = public_len
                .checked_add(self.assignment.relation.layout.source_offsets[witness_idx])
                .ok_or(SynthesisError::Unsatisfiable)?;
            for limb_idx in 0..encoding.limb_count() {
                let logical_col = committed_start
                    .checked_add(limb_idx)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let limb = packed_z.logical_entry(committed_width, logical_col)?;
                enforce_boolean_allocated(
                    &mut cs.namespace(|| format!("terminal_r2_witness_bit_bound_{witness_idx}_{limb_idx}")),
                    &limb,
                    &format!("terminal_r2_witness_bit_bound_{witness_idx}_{limb_idx}"),
                );
            }
        }
        Ok(())
    }

    fn enforce_terminal_commitment<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        packed_z: &PackedWitnessVar,
        commitment_inputs: &[AllocatedNum<SpartanF>],
    ) -> Result<(), SynthesisError> {
        enforce_terminal_ajtai_commitment(
            &mut cs.namespace(|| "terminal_r2_ajtai_commitment"),
            packed_z,
            commitment_inputs,
            "terminal_r2_ajtai_commitment",
        )
    }
}

struct SourceWitnessLinkingCs<'a, 'b, CS: ConstraintSystem<SpartanF>> {
    inner: &'a mut CS,
    layout: &'b Rv32imTerminalFPrimeR2ColumnLayout,
    packed_z: &'b PackedWitnessVar,
    committed_width: usize,
    public_len: usize,
    current_namespace: Vec<String>,
}

impl<'a, 'b, CS: ConstraintSystem<SpartanF>> SourceWitnessLinkingCs<'a, 'b, CS> {
    fn new(
        inner: &'a mut CS,
        layout: &'b Rv32imTerminalFPrimeR2ColumnLayout,
        packed_z: &'b PackedWitnessVar,
        committed_width: usize,
        public_len: usize,
    ) -> Self {
        Self {
            inner,
            layout,
            packed_z,
            committed_width,
            public_len,
            current_namespace: Vec::new(),
        }
    }

    fn alloc_path(&self, annotation: &str) -> String {
        if self.current_namespace.is_empty() {
            return annotation.to_owned();
        }
        let mut path = self.current_namespace.join("/");
        path.push('/');
        path.push_str(annotation);
        path
    }

    fn source_lc(
        &self,
        offset: usize,
        encoding: Rv32imTerminalFPrimePrivateColumnEncoding,
    ) -> Result<LinearCombination<SpartanF>, SynthesisError> {
        let mut lc = LinearCombination::<SpartanF>::zero();
        for limb_idx in 0..encoding.limb_count() {
            let logical_col = self
                .public_len
                .checked_add(offset)
                .and_then(|value| value.checked_add(limb_idx))
                .ok_or(SynthesisError::Unsatisfiable)?;
            let limb = self
                .packed_z
                .logical_entry(self.committed_width, logical_col)?;
            lc = lc + (SpartanF::from_canonical_u64(1u64 << limb_idx), limb.get_variable());
        }
        Ok(lc)
    }
}

impl<CS: ConstraintSystem<SpartanF>> ConstraintSystem<SpartanF> for SourceWitnessLinkingCs<'_, '_, CS> {
    type Root = Self;

    fn alloc<FN, A, AR>(&mut self, annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let annotation = annotation().into();
        let label = self.alloc_path(&annotation);
        let var = self.inner.alloc(|| annotation.clone(), f)?;
        if let Some((offset, encoding)) = self.layout.source_binding(&label) {
            let source_lc = self.source_lc(offset, encoding)?;
            self.inner.enforce(
                || format!("terminal_r2_source_link_{label}"),
                |lc| lc + var,
                |lc| lc + CS::one(),
                |_| source_lc,
            );
        }
        Ok(var)
    }

    fn alloc_input<FN, A, AR>(&mut self, annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.inner.alloc_input(annotation, f)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LB: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LC: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
    {
        self.inner.enforce(annotation, a, b, c);
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        let name = name_fn().into();
        self.current_namespace.push(name.clone());
        self.inner.push_namespace(|| name);
    }

    fn pop_namespace(&mut self) {
        assert!(self.current_namespace.pop().is_some());
        self.inner.pop_namespace();
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

pub(crate) fn terminal_f_prime_committed_step_boundary_public_values(
    boundary: &Rv32imMainRecursionConstruction2PublicBoundary,
) -> Vec<SpartanF> {
    terminal_boundary_public_values(&rv32im_terminal_boundary_view(boundary))
}

fn rv32im_terminal_boundary_view(
    boundary: &Rv32imMainRecursionConstruction2PublicBoundary,
) -> Construction2TerminalBoundaryView<'_> {
    Construction2TerminalBoundaryView {
        fresh_instance_digest: boundary.fresh_instance_digest,
        commitment_digest: boundary.commitment_digest,
        commitment_d: boundary.commitment_d,
        commitment_kappa: boundary.commitment_kappa,
        commitment_data: &boundary.commitment_data,
        x_i_bytes: boundary.x_i.bytes(),
    }
}
