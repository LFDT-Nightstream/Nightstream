//! Owns Ajtai commitment consistency for CE witnesses.

use super::*;

pub fn enforce_ajtai_commitment_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    claim: &CircuitCeClaim,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_ajtai_commitment_data_consistency(cs, witness, &claim.commitment.data, label)
}

pub fn enforce_ajtai_commitment_data_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    c_data: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_ajtai_commitment(cs, witness, c_data, label)
}

pub fn enforce_ajtai_commitment_linear_consistency<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    rows: usize,
    cols: usize,
    witness_entries: &[LinearCombination<SpartanF>],
    c_data: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_ajtai_commitment_linear(cs, rows, cols, witness_entries, c_data, label)
}

fn enforce_ajtai_commitment<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    c_data: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    let witness_entries = witness
        .row_major_values()
        .iter()
        .map(|value| LinearCombination::<SpartanF>::zero() + value.get_variable())
        .collect::<Vec<_>>();
    enforce_ajtai_commitment_linear(cs, witness.rows(), witness.cols(), &witness_entries, c_data, label)
}

fn enforce_ajtai_commitment_linear<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness_rows: usize,
    witness_cols: usize,
    witness_entries: &[LinearCombination<SpartanF>],
    c_data: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    let rows = ajtai_commitment_rows(witness_rows, witness_cols)?;
    if rows.len() != c_data.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (coord_idx, (coeffs, actual)) in rows.iter().zip(c_data.iter()).enumerate() {
        if coeffs.len() != witness_entries.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        cs.enforce(
            || format!("{label}_{coord_idx}"),
            |lc| {
                let mut acc = lc;
                for (coeff, entry) in coeffs.iter().zip(witness_entries.iter()) {
                    if *coeff != F::ZERO {
                        acc = acc + (SpartanF::from_canonical_u64(coeff.as_canonical_u64()), entry);
                    }
                }
                acc
            },
            |lc| lc + CS::one(),
            |lc| lc + actual.get_variable(),
        );
    }
    Ok(())
}

fn ajtai_commitment_rows(rows: usize, cols: usize) -> Result<Vec<Vec<F>>, SynthesisError> {
    let pp = get_global_pp_for_dims(rows, cols).map_err(|_| SynthesisError::Unsatisfiable)?;
    let coord_count = rows
        .checked_mul(pp.kappa)
        .ok_or(SynthesisError::Unsatisfiable)?;
    let witness_len = rows
        .checked_mul(cols)
        .ok_or(SynthesisError::Unsatisfiable)?;
    let mut out = vec![vec![F::ZERO; witness_len]; coord_count];

    for (commit_col, pp_row) in pp.m_rows.iter().enumerate() {
        for (witness_col, ring_el) in pp_row.iter().copied().enumerate() {
            let mut rots = [[F::ZERO; D]; D];
            precompute_rot_columns(ring_el, &mut rots);
            for witness_row in 0..rows {
                let base = witness_row
                    .checked_mul(cols)
                    .and_then(|start| start.checked_add(witness_col))
                    .ok_or(SynthesisError::Unsatisfiable)?;
                for coord_row in 0..rows {
                    let coord = commit_col
                        .checked_mul(rows)
                        .and_then(|start| start.checked_add(coord_row))
                        .ok_or(SynthesisError::Unsatisfiable)?;
                    out[coord][base] = rots[witness_row][coord_row];
                }
            }
        }
    }

    Ok(out)
}
