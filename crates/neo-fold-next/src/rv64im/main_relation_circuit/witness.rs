//! Owns witness-side CE checks over SuperNeo packed `Z` for the RV64IM main relation circuit.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ccs::CcsWitness;
use neo_ccs::Mat;
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::common::decompose_balanced_fixed_d_digits_k;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

use super::claim::CeClaimVar;
use super::k_field::{alloc_k, enforce_k_eq, enforce_k_eq_weighted_base_linear_combination, KNum, KNumVar};
#[derive(Clone)]
pub struct PackedWitnessVar {
    rows: usize,
    cols: usize,
    values: Vec<AllocatedNum<SpartanF>>,
    native_values: Vec<F>,
}

pub fn alloc_packed_witness<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &CcsWitness<F>,
    label: &str,
) -> Result<PackedWitnessVar, SynthesisError> {
    alloc_packed_mat_witness(cs, &witness.Z, label)
}

pub fn alloc_packed_mat_witness<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &Mat<F>,
    label: &str,
) -> Result<PackedWitnessVar, SynthesisError> {
    let rows = witness.rows();
    let cols = witness.cols();
    let values = witness
        .as_slice()
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PackedWitnessVar {
        rows,
        cols,
        values,
        native_values: witness.as_slice().to_vec(),
    })
}

impl PackedWitnessVar {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn logical_entry(
        &self,
        expected_m: usize,
        logical_col: usize,
    ) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
        if self.rows != D || expected_m == 0 || logical_col >= expected_m {
            return Err(SynthesisError::Unsatisfiable);
        }
        let block = logical_col / D;
        let off = logical_col % D;
        let index = off
            .checked_mul(self.cols)
            .and_then(|start| start.checked_add(block))
            .ok_or(SynthesisError::Unsatisfiable)?;
        self.values
            .get(index)
            .cloned()
            .ok_or(SynthesisError::Unsatisfiable)
    }

    pub fn logical_value(&self, expected_m: usize, logical_col: usize) -> Result<F, SynthesisError> {
        if self.rows != D || expected_m == 0 || logical_col >= expected_m {
            return Err(SynthesisError::Unsatisfiable);
        }
        let block = logical_col / D;
        let off = logical_col % D;
        let index = off
            .checked_mul(self.cols)
            .and_then(|start| start.checked_add(block))
            .ok_or(SynthesisError::Unsatisfiable)?;
        self.native_values
            .get(index)
            .copied()
            .ok_or(SynthesisError::Unsatisfiable)
    }

    pub fn row_major_values(&self) -> &[AllocatedNum<SpartanF>] {
        &self.values
    }

    pub fn row_major_native_values(&self) -> &[F] {
        &self.native_values
    }

    pub fn entry(&self, row: usize, col: usize) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
        if row >= self.rows || col >= self.cols {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(self.values[row * self.cols + col].clone())
    }

    pub fn entry_value(&self, row: usize, col: usize) -> Result<F, SynthesisError> {
        if row >= self.rows || col >= self.cols {
            return Err(SynthesisError::Unsatisfiable);
        }
        Ok(self.native_values[row * self.cols + col])
    }

    pub(crate) fn from_parts(
        rows: usize,
        cols: usize,
        values: Vec<AllocatedNum<SpartanF>>,
        native_values: Vec<F>,
    ) -> Self {
        Self {
            rows,
            cols,
            values,
            native_values,
        }
    }
}

#[derive(Clone)]
pub struct BalancedDigitWitnessVar {
    digits: Vec<Vec<AllocatedNum<SpartanF>>>,
    digit_values: Vec<Vec<F>>,
}

impl BalancedDigitWitnessVar {
    pub(crate) fn logical_cols(&self) -> usize {
        self.digits.len()
    }

    pub(crate) fn digit_vars(&self, logical_col: usize) -> Result<&[AllocatedNum<SpartanF>], SynthesisError> {
        self.digits
            .get(logical_col)
            .map(Vec::as_slice)
            .ok_or(SynthesisError::Unsatisfiable)
    }

    pub(crate) fn digit_values(&self, logical_col: usize) -> Result<&[F], SynthesisError> {
        self.digit_values
            .get(logical_col)
            .map(Vec::as_slice)
            .ok_or(SynthesisError::Unsatisfiable)
    }
}

pub fn enforce_x_projection<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    claim: &CeClaimVar,
    expected_m: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if witness.rows != D || claim.x_rows != D || claim.x_cols != claim.m_in || expected_m == 0 {
        return Err(SynthesisError::Unsatisfiable);
    }
    for col in 0..claim.m_in {
        let active_row = col % D;
        let want = witness.logical_entry(expected_m, col)?;
        for row in 0..D {
            let claim_idx = row
                .checked_mul(claim.x_cols)
                .and_then(|start| start.checked_add(col))
                .ok_or(SynthesisError::Unsatisfiable)?;
            let actual = claim
                .x
                .get(claim_idx)
                .ok_or(SynthesisError::Unsatisfiable)?;
            if row == active_row {
                cs.enforce(
                    || format!("{label}_x_match_{row}_{col}"),
                    |lc| lc + actual.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + want.get_variable(),
                );
            } else {
                cs.enforce(
                    || format!("{label}_x_zero_{row}_{col}"),
                    |lc| lc + actual.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc,
                );
            }
        }
    }
    Ok(())
}

pub fn alloc_balanced_digit_witness<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    expected_m: usize,
    params: &NeoParams,
    _delta: SpartanF,
    label: &str,
) -> Result<BalancedDigitWitnessVar, SynthesisError> {
    if witness.rows != D || expected_m == 0 {
        return Err(SynthesisError::Unsatisfiable);
    }
    let base_b = SpartanF::from_canonical_u64(params.b as u64);
    let mut digits = Vec::with_capacity(expected_m);
    let mut digit_values = Vec::with_capacity(expected_m);

    for logical_col in 0..expected_m {
        let z_var = witness.logical_entry(expected_m, logical_col)?;
        let z_value = witness.logical_value(expected_m, logical_col)?;
        let native_digits =
            decompose_balanced_fixed_d_digits_k(z_value, params.b).map_err(|_| SynthesisError::Unsatisfiable)?;
        let mut digit_row = Vec::with_capacity(D);
        let mut digit_row_values = Vec::with_capacity(D);

        for (rho, native_digit) in native_digits.iter().enumerate() {
            let coeffs = native_digit.as_coeffs();
            if coeffs[1] != F::ZERO {
                return Err(SynthesisError::Unsatisfiable);
            }
            let digit_field = AllocatedNum::alloc(
                cs.namespace(|| format!("{label}_digit_field_{logical_col}_{rho}")),
                || Ok(SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64())),
            )?;
            if params.b == 2 {
                enforce_balanced_base2_digit(
                    cs,
                    &digit_field,
                    SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                    &format!("{label}_digit_range_{logical_col}_{rho}"),
                )?;
            } else {
                let range_eval = range_product_f(
                    cs,
                    &digit_field,
                    coeffs[0],
                    params.b,
                    &format!("{label}_digit_range_{logical_col}_{rho}"),
                )?;
                enforce_field_is_zero(
                    cs,
                    &range_eval,
                    &format!("{label}_digit_range_zero_{logical_col}_{rho}"),
                );
            }
            digit_row.push(digit_field);
            digit_row_values.push(coeffs[0]);
        }

        cs.enforce(
            || format!("{label}_digit_recompose_{logical_col}"),
            |lc| {
                let mut acc = lc;
                let mut pow = SpartanF::ONE;
                for digit in &digit_row {
                    acc = acc + (pow, digit.get_variable());
                    pow *= base_b;
                }
                acc
            },
            |lc| lc + CS::one(),
            |lc| lc + z_var.get_variable(),
        );

        digits.push(digit_row);
        digit_values.push(digit_row_values);
    }

    Ok(BalancedDigitWitnessVar { digits, digit_values })
}

pub fn enforce_balanced_digit_alphabet<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    expected_m: usize,
    params: &NeoParams,
    label: &str,
) -> Result<(), SynthesisError> {
    if witness.rows != D || expected_m == 0 {
        return Err(SynthesisError::Unsatisfiable);
    }
    for logical_col in 0..expected_m {
        let z_var = witness.logical_entry(expected_m, logical_col)?;
        let z_value = witness.logical_value(expected_m, logical_col)?;
        if params.b == 2 {
            enforce_balanced_base2_digit(
                cs,
                &z_var,
                SpartanF::from_canonical_u64(z_value.as_canonical_u64()),
                &format!("{label}_{logical_col}"),
            )?;
        } else {
            let range_eval = range_product_f(cs, &z_var, z_value, params.b, &format!("{label}_{logical_col}"))?;
            enforce_field_is_zero(cs, &range_eval, &format!("{label}_{logical_col}_zero"));
        }
    }
    Ok(())
}

fn alloc_constant_f<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
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

fn enforce_field_is_zero<CS: ConstraintSystem<SpartanF>>(cs: &mut CS, value: &AllocatedNum<SpartanF>, label: &str) {
    cs.enforce(
        || format!("{label}_eq"),
        |lc| lc + value.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
}

fn range_product_f<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: &AllocatedNum<SpartanF>,
    value_native: F,
    base_b: u32,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let mut acc = alloc_constant_f(cs, SpartanF::ONE, &format!("{label}_one"))?;
    let mut acc_value = SpartanF::ONE;
    let value_native = SpartanF::from_canonical_u64(value_native.as_canonical_u64());

    for t in -((base_b as i64) - 1)..=((base_b as i64) - 1) {
        let t_field = SpartanF::from_canonical_u64(F::from_i64(t).as_canonical_u64());
        let diff_value = value_native - t_field;
        let diff = AllocatedNum::alloc(cs.namespace(|| format!("{label}_diff_{t}")), || Ok(diff_value))?;
        cs.enforce(
            || format!("{label}_diff_eq_{t}"),
            |lc| lc + value.get_variable() + (-t_field, CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + diff.get_variable(),
        );

        let next_acc_value = acc_value * diff_value;
        let next_acc = AllocatedNum::alloc(cs.namespace(|| format!("{label}_acc_{t}")), || Ok(next_acc_value))?;
        cs.enforce(
            || format!("{label}_acc_eq_{t}"),
            |lc| lc + acc.get_variable(),
            |lc| lc + diff.get_variable(),
            |lc| lc + next_acc.get_variable(),
        );
        acc = next_acc;
        acc_value = next_acc_value;
    }

    Ok(acc)
}

fn enforce_balanced_base2_digit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: &AllocatedNum<SpartanF>,
    value_native: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    let square_value = value_native.square();
    let square = AllocatedNum::alloc(cs.namespace(|| format!("{label}_square")), || Ok(square_value))?;
    cs.enforce(
        || format!("{label}_square_eq"),
        |lc| lc + value.get_variable(),
        |lc| lc + value.get_variable(),
        |lc| lc + square.get_variable(),
    );

    cs.enforce(
        || format!("{label}_balanced_eq"),
        |lc| lc + square.get_variable(),
        |lc| lc + value.get_variable(),
        |lc| lc + value.get_variable(),
    );
    Ok(())
}

pub fn compute_digit_y_zcol<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digits: &BalancedDigitWitnessVar,
    expected_m: usize,
    chi_s: &[K],
    d_pad: usize,
    delta: SpartanF,
    label: &str,
) -> Result<Vec<KNumVar>, SynthesisError> {
    if digits.logical_cols() != expected_m || d_pad < D {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut yz = Vec::with_capacity(d_pad);

    for rho in 0..d_pad {
        let mut terms = Vec::new();
        let mut yz_value = K::ZERO;
        if rho < D {
            for logical_col in 0..expected_m {
                let weight = chi_s.get(logical_col).copied().unwrap_or(K::ZERO);
                let digit_var = digits
                    .digit_vars(logical_col)?
                    .get(rho)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let digit_value = *digits
                    .digit_values(logical_col)?
                    .get(rho)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                yz_value += weight * K::from(digit_value);
                let coeffs = weight.as_coeffs();
                terms.push((
                    SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                    SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                    digit_var.get_variable(),
                ));
            }
        }
        let yz_var = alloc_k(cs, Some(KNum::from_neo_k(yz_value)), &format!("{label}_{rho}"))?;
        enforce_k_eq_weighted_base_linear_combination(cs, &yz_var, &terms, &format!("{label}_{rho}_eq"));
        yz.push(yz_var);
    }

    let _ = delta;
    Ok(yz)
}

pub fn compute_linear_y_zcol<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    expected_m: usize,
    chi_s: &[K],
    d_pad: usize,
    label: &str,
) -> Result<Vec<KNumVar>, SynthesisError> {
    if witness.rows != D || d_pad < D {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut yz = Vec::with_capacity(d_pad);
    for rho in 0..d_pad {
        let mut terms = Vec::new();
        let mut yz_value = K::ZERO;
        if rho < D {
            for logical_col in 0..expected_m {
                if logical_col % D != rho {
                    continue;
                }
                let weight = chi_s.get(logical_col).copied().unwrap_or(K::ZERO);
                let z_var = witness.logical_entry(expected_m, logical_col)?;
                let z_value = witness.logical_value(expected_m, logical_col)?;
                yz_value += weight * K::from(z_value);
                let coeffs = weight.as_coeffs();
                terms.push((
                    SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                    SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                    z_var.get_variable(),
                ));
            }
        }
        let yz_var = alloc_k(cs, Some(KNum::from_neo_k(yz_value)), &format!("{label}_{rho}"))?;
        enforce_k_eq_weighted_base_linear_combination(cs, &yz_var, &terms, &format!("{label}_{rho}_eq"));
        yz.push(yz_var);
    }
    Ok(yz)
}

pub fn enforce_claim_y_zcol<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    computed: &[KNumVar],
    claim: &CeClaimVar,
    label: &str,
) -> Result<(), SynthesisError> {
    if computed.len() != claim.y_zcol.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (computed_var, claim_var)) in computed.iter().zip(claim.y_zcol.iter()).enumerate() {
        enforce_k_eq(cs, computed_var, claim_var, &format!("{label}_{idx}"));
    }
    Ok(())
}
