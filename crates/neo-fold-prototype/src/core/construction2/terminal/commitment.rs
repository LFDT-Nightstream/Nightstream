//! Owns committed witness shape and Ajtai checks for terminal Construction-2.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError};
use neo_math::D;

use super::constraints::enforce_allocated_num_eq_constant;
use super::types::Construction2TerminalBoundaryInputs;
use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::ce_consistency::enforce_ajtai_commitment_linear_consistency;
use crate::superneo_circuit::witness::PackedWitnessVar;

pub(crate) fn enforce_public_commitment_shape<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    packed_z: &PackedWitnessVar,
    boundary: &Construction2TerminalBoundaryInputs,
    label_prefix: &str,
) -> Result<(), SynthesisError> {
    if packed_z.rows() != D || boundary.commitment_data.len() % D != 0 {
        return Err(SynthesisError::Unsatisfiable);
    }
    let expected_kappa = boundary.commitment_data.len() / D;
    enforce_allocated_num_eq_constant(
        &mut cs.namespace(|| format!("{label_prefix}_commitment_kappa_matches_data_len")),
        &boundary.commitment_kappa,
        SpartanF::from_canonical_u64(expected_kappa as u64),
        &format!("{label_prefix}_commitment_kappa_matches_data_len"),
    );
    Ok(())
}

pub(crate) fn enforce_packed_padding_zero<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    packed_z: &PackedWitnessVar,
    committed_width: usize,
    label_prefix: &str,
) -> Result<(), SynthesisError> {
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
                || format!("{label_prefix}_{row}_{col}"),
                |lc| lc + padding.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
        }
    }
    Ok(())
}

pub(crate) fn enforce_terminal_ajtai_commitment<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    packed_z: &PackedWitnessVar,
    commitment_inputs: &[AllocatedNum<SpartanF>],
    label: &str,
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
        cs,
        packed_z.rows(),
        packed_z.cols(),
        &packed_entries,
        commitment_inputs,
        label,
    )
}
