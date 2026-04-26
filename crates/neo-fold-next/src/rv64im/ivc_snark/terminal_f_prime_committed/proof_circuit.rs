//! Owns the Spartan circuit for the terminal committed `F'` R2 proof.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError};
use neo_math::D;

use crate::rv64im::construction2::Rv64imMainRecursionConstruction2PublicBoundary;
use crate::rv64im::main_relation_circuit::ce_consistency::enforce_ajtai_commitment_linear_consistency;
use crate::rv64im::main_relation_circuit::witness::{alloc_packed_mat_witness, PackedWitnessVar};
use crate::rv64im::main_relation_spartan::{
    construction2_commitment_digest_circuit, construction2_public_boundary_digest_circuit, digest32_as_spartan_fields,
    enforce_digest_eq, Rv64imMainRecursionStepSpartanPublishedTarget,
};
use crate::witness_layout::commit_cols_for_full_width;

use super::circuit::{enforce_boolean_allocated, matrix_row_linear_combinations, native_to_spartan, set_z_entry};
use super::{Rv64imDeciderEngine, Rv64imTerminalFPrimeCommittedStepCircuit, SpartanCircuit, SpartanF};

struct Rv64imTerminalFPrimeBoundaryInputs {
    fresh_instance_digest: [AllocatedNum<SpartanF>; 4],
    commitment_digest: [AllocatedNum<SpartanF>; 4],
    commitment_d: AllocatedNum<SpartanF>,
    commitment_kappa: AllocatedNum<SpartanF>,
    commitment_data: Vec<AllocatedNum<SpartanF>>,
    x_i: [AllocatedNum<SpartanF>; 4],
}

impl SpartanCircuit<Rv64imDeciderEngine> for Rv64imTerminalFPrimeCommittedStepCircuit {
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
        let z_entries = self.synthesize_z_entries(cs, &public_inputs, &packed_z, committed_width)?;
        self.enforce_committed_superneo_image(cs, &packed_z, committed_width, &z_entries)?;
        self.enforce_rowwise_terminal_r2(cs, &z_entries)?;
        self.enforce_terminal_commitment(cs, &packed_z, &boundary_inputs.commitment_data)?;
        Ok(())
    }
}

impl Rv64imTerminalFPrimeCommittedStepCircuit {
    fn alloc_public_boundary_inputs<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
    ) -> Result<Rv64imTerminalFPrimeBoundaryInputs, SynthesisError> {
        let fresh_instance_digest = alloc_digest_public_inputs(
            cs,
            "terminal_boundary_fresh_instance_digest",
            self.public_boundary.fresh_instance_digest,
        )?;
        let commitment_digest = alloc_digest_public_inputs(
            cs,
            "terminal_boundary_commitment_digest",
            self.public_boundary.commitment_digest,
        )?;
        let commitment_d = AllocatedNum::alloc_input(cs.namespace(|| "terminal_boundary_commitment_d"), || {
            Ok(SpartanF::from_canonical_u64(self.public_boundary.commitment_d))
        })?;
        let commitment_kappa =
            AllocatedNum::alloc_input(cs.namespace(|| "terminal_boundary_commitment_kappa"), || {
                Ok(SpartanF::from_canonical_u64(self.public_boundary.commitment_kappa))
            })?;
        let commitment_data = self
            .public_boundary
            .commitment_data
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                AllocatedNum::alloc_input(
                    cs.namespace(|| format!("terminal_boundary_commitment_data_{idx}")),
                    || Ok(native_to_spartan(value)),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let x_i = alloc_digest_public_inputs(cs, "terminal_boundary_x_i", self.public_boundary.x_i.bytes())?;
        Ok(Rv64imTerminalFPrimeBoundaryInputs {
            fresh_instance_digest,
            commitment_digest,
            commitment_d,
            commitment_kappa,
            commitment_data,
            x_i,
        })
    }

    fn enforce_public_boundary<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        terminal_public_inputs: &[AllocatedNum<SpartanF>],
        boundary: &Rv64imTerminalFPrimeBoundaryInputs,
    ) -> Result<(), SynthesisError> {
        enforce_allocated_num_eq_constant(
            &mut cs.namespace(|| "terminal_boundary_commitment_d_eq"),
            &boundary.commitment_d,
            SpartanF::from_canonical_u64(D as u64),
            "terminal_boundary_commitment_d_eq",
        );
        let expected_commitment_digest = construction2_commitment_digest_circuit(
            &mut cs.namespace(|| "terminal_boundary_expected_commitment_digest"),
            &boundary.commitment_d,
            &boundary.commitment_kappa,
            &boundary.commitment_data,
            "terminal_boundary_expected_commitment_digest",
        )?;
        enforce_digest_eq(
            &mut cs.namespace(|| "terminal_boundary_commitment_digest_eq"),
            &boundary.commitment_digest,
            &expected_commitment_digest,
            "terminal_boundary_commitment_digest_eq",
        )?;
        let expected_fresh_instance_digest = construction2_public_boundary_digest_circuit(
            &mut cs.namespace(|| "terminal_boundary_expected_fresh_instance_digest"),
            &boundary.commitment_digest,
            &boundary.x_i,
            "terminal_boundary_expected_fresh_instance_digest",
        )?;
        enforce_digest_eq(
            &mut cs.namespace(|| "terminal_boundary_fresh_instance_digest_eq"),
            &boundary.fresh_instance_digest,
            &expected_fresh_instance_digest,
            "terminal_boundary_fresh_instance_digest_eq",
        )?;

        let x_out_digest_start = Rv64imMainRecursionStepSpartanPublishedTarget::terminal_r2_public_value_range_static()
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
        boundary: &Rv64imTerminalFPrimeBoundaryInputs,
    ) -> Result<(), SynthesisError> {
        if packed_z.rows() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
        if boundary.commitment_data.len() % D != 0 {
            return Err(SynthesisError::Unsatisfiable);
        }
        let expected_kappa = boundary.commitment_data.len() / D;
        enforce_allocated_num_eq_constant(
            &mut cs.namespace(|| "terminal_boundary_commitment_kappa_matches_data_len"),
            &boundary.commitment_kappa,
            SpartanF::from_canonical_u64(expected_kappa as u64),
            "terminal_boundary_commitment_kappa_matches_data_len",
        );
        Ok(())
    }

    fn synthesize_z_entries<CS: ConstraintSystem<SpartanF>>(
        &self,
        _cs: &mut CS,
        public_inputs: &[AllocatedNum<SpartanF>],
        packed_z: &PackedWitnessVar,
        committed_width: usize,
    ) -> Result<Vec<LinearCombination<SpartanF>>, SynthesisError> {
        let relation = self.assignment.relation();
        let num_io = self.assignment.terminal_public_values.len();
        if public_inputs.len() != num_io || relation.structure().m != self.assignment.raw_full_width() {
            return Err(SynthesisError::Unsatisfiable);
        }

        let mut entries = vec![None; relation.structure().m];
        for (public_idx, public_input) in public_inputs.iter().enumerate() {
            let col = relation
                .layout
                .public_col(public_idx)
                .map_err(|_| SynthesisError::Unsatisfiable)?;
            set_z_entry(
                &mut entries,
                col,
                LinearCombination::<SpartanF>::zero() + public_input.get_variable(),
            )?;
        }

        let public_len = self.assignment.r2_public_values.len();
        for witness_idx in 0..relation.num_variables() {
            let start_col = relation
                .layout
                .witness_col_start(witness_idx)
                .map_err(|_| SynthesisError::Unsatisfiable)?;
            let limb_count = relation
                .layout
                .witness_encoding(witness_idx)
                .map_err(|_| SynthesisError::Unsatisfiable)?
                .limb_count();
            for limb_idx in 0..limb_count {
                let packed_logical_col = public_len
                    .checked_add(relation.layout.private_offsets[witness_idx])
                    .and_then(|value| value.checked_add(limb_idx))
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let limb = packed_z.logical_entry(committed_width, packed_logical_col)?;
                set_z_entry(
                    &mut entries,
                    start_col + limb_idx,
                    LinearCombination::<SpartanF>::zero() + limb.get_variable(),
                )?;
            }
        }

        let one_col = relation.layout.one_col();
        set_z_entry(&mut entries, one_col, LinearCombination::<SpartanF>::zero() + CS::one())?;

        entries
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(SynthesisError::Unsatisfiable)
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
        packed_z: &PackedWitnessVar,
        committed_width: usize,
        z_entries: &[LinearCombination<SpartanF>],
    ) -> Result<(), SynthesisError> {
        let public_len = self.assignment.r2_public_values.len();
        let committed_witness_len = self.assignment.committed_witness_values().len();
        if committed_width
            != public_len
                .checked_add(committed_witness_len)
                .ok_or(SynthesisError::Unsatisfiable)?
            || z_entries.len() != self.assignment.raw_full_width()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        for public_idx in 0..public_len {
            let packed_entry = packed_z.logical_entry(committed_width, public_idx)?;
            let expected = z_entries
                .get(public_idx)
                .cloned()
                .ok_or(SynthesisError::Unsatisfiable)?;
            cs.enforce(
                || format!("terminal_r2_superneo_public_z_link_{public_idx}"),
                |lc| lc + packed_entry.get_variable(),
                |lc| lc + CS::one(),
                |_| expected,
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
        for row in 0..packed_z.rows() {
            for col in 0..packed_z.cols() {
                let logical_col = col
                    .checked_mul(D)
                    .and_then(|base| base.checked_add(row))
                    .ok_or(SynthesisError::Unsatisfiable)?;
                if logical_col < committed_width {
                    continue;
                }
                let padding = packed_z.entry(row, col)?;
                cs.enforce(
                    || format!("terminal_r2_superneo_padding_zero_{row}_{col}"),
                    |lc| lc + padding.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc,
                );
            }
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

        for witness_idx in 0..self.assignment.relation.num_variables() {
            let encoding = self
                .assignment
                .relation
                .layout
                .witness_encoding(witness_idx)
                .map_err(|_| SynthesisError::Unsatisfiable)?;
            let committed_start = public_len
                .checked_add(self.assignment.relation.layout.private_offsets[witness_idx])
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

    fn enforce_rowwise_terminal_r2<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        z_entries: &[LinearCombination<SpartanF>],
    ) -> Result<(), SynthesisError> {
        let structure = self.assignment.relation().structure();
        if structure.matrices.len() != 3 {
            return Err(SynthesisError::Unsatisfiable);
        }
        let a_rows = matrix_row_linear_combinations(&structure.matrices[0], z_entries)?;
        let b_rows = matrix_row_linear_combinations(&structure.matrices[1], z_entries)?;
        let c_rows = matrix_row_linear_combinations(&structure.matrices[2], z_entries)?;
        if a_rows.len() != structure.n || b_rows.len() != structure.n || c_rows.len() != structure.n {
            return Err(SynthesisError::Unsatisfiable);
        }
        for row in 0..structure.n {
            cs.enforce(
                || format!("terminal_r2_row_{row}"),
                |_| a_rows[row].clone(),
                |_| b_rows[row].clone(),
                |_| c_rows[row].clone(),
            );
        }
        Ok(())
    }

    fn enforce_terminal_commitment<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        packed_z: &PackedWitnessVar,
        commitment_inputs: &[AllocatedNum<SpartanF>],
    ) -> Result<(), SynthesisError> {
        if packed_z.rows() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
        let packed_entries = packed_z
            .row_major_values()
            .iter()
            .map(|entry| LinearCombination::<SpartanF>::zero() + entry.get_variable())
            .collect::<Vec<_>>();
        enforce_ajtai_commitment_linear_consistency(
            &mut cs.namespace(|| "terminal_r2_ajtai_commitment"),
            packed_z.rows(),
            packed_z.cols(),
            &packed_entries,
            commitment_inputs,
            "terminal_r2_ajtai_commitment",
        )
    }
}

pub(crate) fn terminal_f_prime_committed_step_boundary_public_values(
    boundary: &Rv64imMainRecursionConstruction2PublicBoundary,
) -> Vec<SpartanF> {
    let mut values = Vec::with_capacity(14 + boundary.commitment_data.len());
    values.extend(digest32_as_spartan_fields(boundary.fresh_instance_digest));
    values.extend(digest32_as_spartan_fields(boundary.commitment_digest));
    values.push(SpartanF::from_canonical_u64(boundary.commitment_d));
    values.push(SpartanF::from_canonical_u64(boundary.commitment_kappa));
    values.extend(boundary.commitment_data.iter().map(native_to_spartan));
    values.extend(digest32_as_spartan_fields(boundary.x_i.bytes()));
    values
}

fn alloc_digest_public_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    label: &str,
    digest: [u8; 32],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let fields = digest32_as_spartan_fields(digest);
    let values = fields
        .into_iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("{label}_{idx}")), || Ok(value)))
        .collect::<Result<Vec<_>, _>>()?;
    values.try_into().map_err(|_| SynthesisError::Unsatisfiable)
}

fn enforce_allocated_num_eq_constant<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: &AllocatedNum<SpartanF>,
    expected: SpartanF,
    label: &str,
) {
    cs.enforce(
        || label,
        |lc| lc + value.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (expected, CS::one()),
    );
}
