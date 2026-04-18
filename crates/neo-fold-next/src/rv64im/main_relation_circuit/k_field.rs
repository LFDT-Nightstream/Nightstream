//! Owns K-field bellpepper gadgets for the RV64IM main relation circuit.

use bellpepper_core::{ConstraintSystem, SynthesisError, Variable};
use ff::PrimeField;
use neo_math::{KExtensions, K as NeoK};
use p3_field::PrimeField64;

#[derive(Clone, Debug)]
pub struct KNum<F: PrimeField> {
    pub c0: F,
    pub c1: F,
}

#[derive(Clone, Debug)]
pub struct KNumVar {
    pub c0: Variable,
    pub c1: Variable,
}

impl<F: PrimeField> KNum<F> {
    pub fn new(c0: F, c1: F) -> Self {
        Self { c0, c1 }
    }

    pub fn from_f(c0: F) -> Self {
        Self { c0, c1: F::ZERO }
    }

    pub fn from_neo_k(k: NeoK) -> Self {
        let coeffs = k.as_coeffs();
        Self {
            c0: F::from(coeffs[0].as_canonical_u64()),
            c1: F::from(coeffs[1].as_canonical_u64()),
        }
    }
}

pub fn alloc_k<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    value: Option<KNum<F>>,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let c0 = cs.alloc(
        || format!("{label}_c0"),
        || {
            value
                .as_ref()
                .map(|value| value.c0)
                .ok_or(SynthesisError::AssignmentMissing)
        },
    )?;
    let c1 = cs.alloc(
        || format!("{label}_c1"),
        || {
            value
                .as_ref()
                .map(|value| value.c1)
                .ok_or(SynthesisError::AssignmentMissing)
        },
    )?;
    Ok(KNumVar { c0, c1 })
}

pub fn alloc_constant_k<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    value: KNum<F>,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let out = alloc_k(cs, Some(value.clone()), label)?;
    cs.enforce(
        || format!("{label}_c0_constant"),
        |lc| lc + out.c0,
        |lc| lc + CS::one(),
        |lc| lc + (value.c0, CS::one()),
    );
    cs.enforce(
        || format!("{label}_c1_constant"),
        |lc| lc + out.c1,
        |lc| lc + CS::one(),
        |lc| lc + (value.c1, CS::one()),
    );
    Ok(out)
}

pub fn k_lift_from_f<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    f_var: Variable,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let c1 = cs.alloc(|| format!("{label}_c1_zero"), || Ok(F::ZERO))?;
    cs.enforce(
        || format!("{label}_c1_is_zero"),
        |lc| lc + c1,
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(KNumVar { c0: f_var, c1 })
}

pub fn enforce_k_eq<F: PrimeField, CS: ConstraintSystem<F>>(cs: &mut CS, left: &KNumVar, right: &KNumVar, label: &str) {
    if left.c0 == right.c0 && left.c1 == right.c1 {
        return;
    }
    cs.enforce(
        || format!("{label}_c0_eq"),
        |lc| lc + left.c0,
        |lc| lc + CS::one(),
        |lc| lc + right.c0,
    );
    cs.enforce(
        || format!("{label}_c1_eq"),
        |lc| lc + left.c1,
        |lc| lc + CS::one(),
        |lc| lc + right.c1,
    );
}

pub fn enforce_k_eq_native<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    actual: &KNumVar,
    expected: KNum<F>,
    label: &str,
) {
    cs.enforce(
        || format!("{label}_c0_eq"),
        |lc| lc + actual.c0,
        |lc| lc + CS::one(),
        |lc| lc + (expected.c0, CS::one()),
    );
    cs.enforce(
        || format!("{label}_c1_eq"),
        |lc| lc + actual.c1,
        |lc| lc + CS::one(),
        |lc| lc + (expected.c1, CS::one()),
    );
}

pub fn enforce_k_eq_weighted_base_linear_combination<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    target: &KNumVar,
    terms: &[(F, F, Variable)],
    label: &str,
) {
    cs.enforce(
        || format!("{label}_c0"),
        |lc| {
            let mut acc = lc;
            for (coeff_c0, _, variable) in terms {
                acc = acc + (*coeff_c0, *variable);
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.c0,
    );
    cs.enforce(
        || format!("{label}_c1"),
        |lc| {
            let mut acc = lc;
            for (_, coeff_c1, variable) in terms {
                acc = acc + (*coeff_c1, *variable);
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.c1,
    );
}

pub fn enforce_k_eq_constant_f_linear_combination<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    target: &KNumVar,
    terms: &[(F, Variable, Variable)],
    label: &str,
) {
    cs.enforce(
        || format!("{label}_c0"),
        |lc| {
            let mut acc = lc;
            for (coeff, c0, _) in terms {
                acc = acc + (*coeff, *c0);
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.c0,
    );
    cs.enforce(
        || format!("{label}_c1"),
        |lc| {
            let mut acc = lc;
            for (coeff, _, c1) in terms {
                acc = acc + (*coeff, *c1);
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.c1,
    );
}

pub fn alloc_k_constant_k_linear_combination<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    terms: &[(KNum<F>, Variable, Variable)],
    value_hint: KNum<F>,
    delta: F,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let out_c0 = cs.alloc(|| format!("{label}_c0"), || Ok(value_hint.c0))?;
    let out_c1 = cs.alloc(|| format!("{label}_c1"), || Ok(value_hint.c1))?;

    cs.enforce(
        || format!("{label}_c0_eq"),
        |lc| {
            let mut acc = lc;
            for (coeff, c0, c1) in terms {
                acc = acc + (coeff.c0, *c0);
                acc = acc + (delta * coeff.c1, *c1);
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + out_c0,
    );
    cs.enforce(
        || format!("{label}_c1_eq"),
        |lc| {
            let mut acc = lc;
            for (coeff, c0, c1) in terms {
                acc = acc + (coeff.c1, *c0);
                acc = acc + (coeff.c0, *c1);
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + out_c1,
    );

    Ok(KNumVar { c0: out_c0, c1: out_c1 })
}

pub fn k_add<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    left: &KNumVar,
    right: &KNumVar,
    value_hint: Option<KNum<F>>,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let hint = value_hint.ok_or(SynthesisError::AssignmentMissing)?;
    let c0 = cs.alloc(|| format!("{label}_sum_c0"), || Ok(hint.c0))?;
    cs.enforce(
        || format!("{label}_c0"),
        |lc| lc + left.c0 + right.c0,
        |lc| lc + CS::one(),
        |lc| lc + c0,
    );

    let c1 = cs.alloc(|| format!("{label}_sum_c1"), || Ok(hint.c1))?;
    cs.enforce(
        || format!("{label}_c1"),
        |lc| lc + left.c1 + right.c1,
        |lc| lc + CS::one(),
        |lc| lc + c1,
    );

    Ok(KNumVar { c0, c1 })
}

pub fn k_mul<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    left: &KNumVar,
    right: &KNumVar,
    left_hint: KNum<F>,
    right_hint: KNum<F>,
    product_hint: KNum<F>,
    delta: F,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let sum_left = left_hint.c0 + left_hint.c1;
    let sum_right = right_hint.c0 + right_hint.c1;
    let sum_cross = sum_left * sum_right;
    let a0b0 = left_hint.c0 * right_hint.c0;
    let a1b1 = left_hint.c1 * right_hint.c1;

    let out_c0 = cs.alloc(|| format!("{label}_c0"), || Ok(product_hint.c0))?;
    let out_c1 = cs.alloc(|| format!("{label}_c1"), || Ok(product_hint.c1))?;

    let sum_cross_var = cs.alloc(|| format!("{label}_sum_cross"), || Ok(sum_cross))?;
    let a0b0_var = cs.alloc(|| format!("{label}_a0b0"), || Ok(a0b0))?;
    let a1b1_var = cs.alloc(|| format!("{label}_a1b1"), || Ok(a1b1))?;

    cs.enforce(
        || format!("{label}_sum_cross_eq"),
        |lc| lc + left.c0 + left.c1,
        |lc| lc + right.c0 + right.c1,
        |lc| lc + sum_cross_var,
    );

    cs.enforce(
        || format!("{label}_a0b0_eq"),
        |lc| lc + left.c0,
        |lc| lc + right.c0,
        |lc| lc + a0b0_var,
    );
    cs.enforce(
        || format!("{label}_a1b1_eq"),
        |lc| lc + left.c1,
        |lc| lc + right.c1,
        |lc| lc + a1b1_var,
    );

    cs.enforce(
        || format!("{label}_out_c0_eq"),
        |lc| lc + a0b0_var + (delta, a1b1_var),
        |lc| lc + CS::one(),
        |lc| lc + out_c0,
    );
    cs.enforce(
        || format!("{label}_out_c1_eq"),
        |lc| lc + sum_cross_var - a0b0_var - a1b1_var,
        |lc| lc + CS::one(),
        |lc| lc + out_c1,
    );

    Ok(KNumVar { c0: out_c0, c1: out_c1 })
}

pub fn k_scalar_mul<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    scalar: F,
    value: &KNumVar,
    value_hint: Option<KNum<F>>,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let hint = value_hint.ok_or(SynthesisError::AssignmentMissing)?;
    let c0 = cs.alloc(|| format!("{label}_scaled_c0"), || Ok(hint.c0))?;
    let c1 = cs.alloc(|| format!("{label}_scaled_c1"), || Ok(hint.c1))?;

    cs.enforce(
        || format!("{label}_c0"),
        |lc| lc + (scalar, value.c0),
        |lc| lc + CS::one(),
        |lc| lc + c0,
    );
    cs.enforce(
        || format!("{label}_c1"),
        |lc| lc + (scalar, value.c1),
        |lc| lc + CS::one(),
        |lc| lc + c1,
    );

    Ok(KNumVar { c0, c1 })
}

pub fn k_base_mul_var<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    value: &KNumVar,
    base_var: Variable,
    value_hint: KNum<F>,
    base_hint: F,
    product_hint: KNum<F>,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let out_c0 = cs.alloc(|| format!("{label}_c0"), || Ok(product_hint.c0))?;
    let out_c1 = cs.alloc(|| format!("{label}_c1"), || Ok(product_hint.c1))?;

    let _ = value_hint;
    let _ = base_hint;

    cs.enforce(
        || format!("{label}_c0_eq"),
        |lc| lc + value.c0,
        |lc| lc + base_var,
        |lc| lc + out_c0,
    );
    cs.enforce(
        || format!("{label}_c1_eq"),
        |lc| lc + value.c1,
        |lc| lc + base_var,
        |lc| lc + out_c1,
    );

    Ok(KNumVar { c0: out_c0, c1: out_c1 })
}
