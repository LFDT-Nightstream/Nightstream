//! Owns witness-side Π_RLC / Π_DEC constraints over packed SuperNeo `Z` matrices.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

use super::rho_sampling::RotRhoMatrixVar;
use super::witness::{alloc_packed_mat_witness, PackedWitnessVar};

pub fn mix_packed_witnesses_with_rho_mats<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witnesses: &[PackedWitnessVar],
    rho_mats: &[Mat<F>],
    label: &str,
) -> Result<PackedWitnessVar, SynthesisError> {
    if witnesses.is_empty() || witnesses.len() != rho_mats.len() || witnesses[0].rows() != D {
        return Err(SynthesisError::Unsatisfiable);
    }
    let cols = witnesses[0].cols();
    for (witness, rho) in witnesses.iter().zip(rho_mats.iter()) {
        if witness.rows() != D || witness.cols() != cols || rho.rows() != D || rho.cols() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    let mut native = vec![F::ZERO; D * cols];
    let mut values = Vec::with_capacity(D * cols);
    for row in 0..D {
        for col in 0..cols {
            let mut native_value = F::ZERO;
            for (witness, rho) in witnesses.iter().zip(rho_mats.iter()) {
                for src in 0..D {
                    let coeff = rho[(row, src)];
                    native_value += coeff * witness.entry_value(src, col)?;
                }
            }
            native[row * cols + col] = native_value;
            let mixed = AllocatedNum::alloc(cs.namespace(|| format!("{label}_alloc_{row}_{col}")), || {
                Ok(SpartanF::from_canonical_u64(native_value.as_canonical_u64()))
            })?;
            cs.enforce(
                || format!("{label}_{row}_{col}"),
                |lc| {
                    let mut acc = lc;
                    for (witness, rho) in witnesses.iter().zip(rho_mats.iter()) {
                        for src in 0..D {
                            let coeff = SpartanF::from_canonical_u64(rho[(row, src)].as_canonical_u64());
                            acc = acc
                                + (
                                    coeff,
                                    witness
                                        .entry(src, col)
                                        .expect("checked dims")
                                        .get_variable(),
                                );
                        }
                    }
                    acc
                },
                |lc| lc + CS::one(),
                |lc| lc + mixed.get_variable(),
            );
            values.push(mixed);
        }
    }

    Ok(PackedWitnessVar::from_parts(D, cols, values, native))
}

pub fn mix_packed_witnesses_with_rho_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witnesses: &[PackedWitnessVar],
    rho_mats: &[RotRhoMatrixVar],
    label: &str,
) -> Result<PackedWitnessVar, SynthesisError> {
    if witnesses.is_empty() || witnesses.len() != rho_mats.len() || witnesses[0].rows() != D {
        return Err(SynthesisError::Unsatisfiable);
    }
    let cols = witnesses[0].cols();
    for witness in witnesses {
        if witness.rows() != D || witness.cols() != cols {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    let mut native = vec![F::ZERO; D * cols];
    let mut values = Vec::with_capacity(D * cols);
    for row in 0..D {
        for col in 0..cols {
            let mut acc = alloc_constant_field(
                cs.namespace(|| format!("{label}_zero_{row}_{col}")),
                SpartanF::ZERO,
                &format!("{label}_zero_{row}_{col}"),
            )?;
            let mut acc_value = F::ZERO;
            for (wit_idx, (witness, rho)) in witnesses.iter().zip(rho_mats.iter()).enumerate() {
                for src in 0..D {
                    let coeff_var = rho.entry(row, src)?;
                    let coeff_value = rho.entry_value(row, src)?;
                    let product = coeff_var.mul(
                        cs.namespace(|| format!("{label}_mul_{row}_{col}_{wit_idx}_{src}")),
                        &witness.entry(src, col)?,
                    )?;
                    let product_value = coeff_value * witness.entry_value(src, col)?;
                    acc_value += product_value;
                    acc = add_field_vars(
                        cs.namespace(|| format!("{label}_acc_{row}_{col}_{wit_idx}_{src}")),
                        &acc,
                        &product,
                        SpartanF::from_canonical_u64(acc_value.as_canonical_u64()),
                    )?;
                }
            }
            native[row * cols + col] = acc_value;
            values.push(acc);
        }
    }

    Ok(PackedWitnessVar::from_parts(D, cols, values, native))
}

pub fn enforce_packed_dec_split<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &PackedWitnessVar,
    children: &[PackedWitnessVar],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty() || parent.rows() != D {
        return Err(SynthesisError::Unsatisfiable);
    }
    let cols = parent.cols();
    for child in children {
        if child.rows() != D || child.cols() != cols {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    let b = SpartanF::from_canonical_u64(base_b as u64);
    for row in 0..D {
        for col in 0..cols {
            cs.enforce(
                || format!("{label}_{row}_{col}"),
                |lc| {
                    let mut acc = lc;
                    let mut pow = SpartanF::ONE;
                    for child in children {
                        acc = acc + (pow, child.entry(row, col).expect("checked dims").get_variable());
                        pow *= b;
                    }
                    acc
                },
                |lc| lc + CS::one(),
                |lc| lc + parent.entry(row, col).expect("checked dims").get_variable(),
            );
        }
    }
    Ok(())
}

pub fn enforce_packed_witness_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    left: &PackedWitnessVar,
    right: &PackedWitnessVar,
    label: &str,
) -> Result<(), SynthesisError> {
    if left.rows() != right.rows() || left.cols() != right.cols() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for row in 0..left.rows() {
        for col in 0..left.cols() {
            cs.enforce(
                || format!("{label}_{row}_{col}"),
                |lc| lc + left.entry(row, col).expect("checked dims").get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + right.entry(row, col).expect("checked dims").get_variable(),
            );
        }
    }
    Ok(())
}

pub fn alloc_split_children_from_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    children: &[neo_ccs::Mat<F>],
    label: &str,
) -> Result<Vec<PackedWitnessVar>, SynthesisError> {
    let mut out = Vec::with_capacity(children.len());
    for (idx, child) in children.iter().enumerate() {
        out.push(alloc_packed_mat_witness(cs, child, &format!("{label}_{idx}"))?);
    }
    Ok(out)
}

fn alloc_constant_field<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    value: SpartanF,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(value))?;
    cs.enforce(
        || format!("{label}_constant"),
        |lc| lc + out.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (value, CS::one()),
    );
    Ok(out)
}

fn add_field_vars<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    left: &AllocatedNum<SpartanF>,
    right: &AllocatedNum<SpartanF>,
    sum_value: SpartanF,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let sum = AllocatedNum::alloc(cs.namespace(|| "sum_alloc"), || Ok(sum_value))?;
    cs.enforce(
        || "sum_eq",
        |lc| lc + left.get_variable() + right.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + sum.get_variable(),
    );
    Ok(sum)
}
