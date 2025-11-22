//! K-field gadgets: 2-limb representation over F
//!
//! K is a degree-2 extension field over F (Goldilocks).
//! We represent K = c0 + c1 * u where u^2 = δ for some constant δ.
//!
//! This module implements:
//! - Addition: (a0, a1) + (b0, b1) = (a0+b0, a1+b1)
//! - Multiplication: (a0, a1) * (b0, b1) = (a0*b0 + a1*b1*δ, a0*b1 + a1*b0)
//! - Lifting from F to K: x -> (x, 0)

use bellpepper_core::{
    ConstraintSystem, SynthesisError, Variable,
};
use ff::PrimeField;
use neo_math::K as NeoK;

/// A K-field element represented as 2 limbs over F
#[derive(Clone, Debug)]
pub struct KNum<F: PrimeField> {
    pub c0: F,
    pub c1: F,
}

/// A K-field element in the circuit (as allocated variables)
#[derive(Clone, Debug)]
pub struct KNumVar {
    pub c0: Variable,
    pub c1: Variable,
}

impl<F: PrimeField> KNum<F> {
    /// Create a K number from two F elements
    pub fn new(c0: F, c1: F) -> Self {
        Self { c0, c1 }
    }

    /// Lift an F element to K (as c0 + 0*u)
    pub fn from_f(c0: F) -> Self {
        Self { c0, c1: F::ZERO }
    }

    /// Convert from Neo's K type to circuit K type
    /// Extracts the base field coefficients from K = c0 + c1*u
    pub fn from_neo_k(k: NeoK) -> Self {
        use neo_math::KExtensions;
        use p3_field::PrimeField64;
        
        // Extract actual base field coefficients (not arbitrary limbs)
        let coeffs = k.as_coeffs(); // Returns [Fq; 2]
        Self {
            c0: F::from(coeffs[0].as_canonical_u64()),
            c1: F::from(coeffs[1].as_canonical_u64()),
        }
    }
}

/// Allocate a K number in the circuit
pub fn alloc_k<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    value: Option<KNum<F>>,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let c0 = cs.alloc(
        || format!("{}_c0", label),
        || value.as_ref().map(|v| v.c0).ok_or(SynthesisError::AssignmentMissing),
    )?;
    let c1 = cs.alloc(
        || format!("{}_c1", label),
        || value.as_ref().map(|v| v.c1).ok_or(SynthesisError::AssignmentMissing),
    )?;
    Ok(KNumVar { c0, c1 })
}

/// Lift an F field element to K (as allocated variables)
/// Creates K = (f_var, 0) where the second component is constrained to be zero
pub fn k_lift_from_f<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    f_var: Variable,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    // c0 = f_var, c1 = 0 (witness variable, not public input)
    let c1 = cs.alloc(
        || format!("{}_c1_zero", label),
        || Ok(F::ZERO),
    )?;
    
    // Enforce c1 = 0
    cs.enforce(
        || format!("{}_c1_is_zero", label),
        |lc| lc + c1,
        |lc| lc + CS::one(),
        |lc| lc,
    );

    Ok(KNumVar { c0: f_var, c1 })
}

/// K-field addition: (a0, a1) + (b0, b1) = (a0+b0, a1+b1)
/// 
/// value_hint: Optional precomputed result for witness generation
pub fn k_add<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    a: &KNumVar,
    b: &KNumVar,
    value_hint: Option<KNum<F>>,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    // c0 = a.c0 + b.c0
    let c0_val = value_hint.as_ref().map(|v| v.c0);
    let c0 = cs.alloc(
        || format!("{}_sum_c0", label),
        || Ok(c0_val.unwrap_or(F::ZERO)),
    )?;

    // Enforce c0 = a.c0 + b.c0
    cs.enforce(
        || format!("{}_c0_constraint", label),
        |lc| lc + a.c0 + b.c0,
        |lc| lc + CS::one(),
        |lc| lc + c0,
    );

    // c1 = a.c1 + b.c1
    let c1_val = value_hint.as_ref().map(|v| v.c1);
    let c1 = cs.alloc(
        || format!("{}_sum_c1", label),
        || Ok(c1_val.unwrap_or(F::ZERO)),
    )?;

    // Enforce c1 = a.c1 + b.c1
    cs.enforce(
        || format!("{}_c1_constraint", label),
        |lc| lc + a.c1 + b.c1,
        |lc| lc + CS::one(),
        |lc| lc + c1,
    );

    Ok(KNumVar { c0, c1 })
}

/// K-field multiplication: (a0, a1) * (b0, b1) = (a0*b0 + δ*a1*b1, a0*b1 + a1*b0)
///
/// Where δ is the constant such that u^2 = δ in the extension field.
/// For Goldilocks K, δ = 7 (since u^2 = 7).
///
/// value_hint: Optional precomputed result for witness generation
pub fn k_mul<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    a: &KNumVar,
    b: &KNumVar,
    delta: F,
    value_hint: Option<KNum<F>>,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    // Allocate result components using hint
    let c0_val = value_hint.as_ref().map(|v| v.c0);
    let c1_val = value_hint.as_ref().map(|v| v.c1);
    
    let c0 = cs.alloc(
        || format!("{}_prod_c0", label),
        || Ok(c0_val.unwrap_or(F::ZERO)),
    )?;
    
    let c1 = cs.alloc(
        || format!("{}_prod_c1", label),
        || Ok(c1_val.unwrap_or(F::ZERO)),
    )?;

    // Allocate intermediate products (witness only, no hint needed)
    let a0_b0 = cs.alloc(
        || format!("{}_a0b0", label),
        || Ok(F::ZERO),
    )?;

    let a1_b1 = cs.alloc(
        || format!("{}_a1b1", label),
        || Ok(F::ZERO),
    )?;

    let a0_b1 = cs.alloc(
        || format!("{}_a0b1", label),
        || Ok(F::ZERO),
    )?;

    let a1_b0 = cs.alloc(
        || format!("{}_a1b0", label),
        || Ok(F::ZERO),
    )?;

    // Enforce intermediate products
    cs.enforce(
        || format!("{}_a0b0_constraint", label),
        |lc| lc + a.c0,
        |lc| lc + b.c0,
        |lc| lc + a0_b0,
    );

    cs.enforce(
        || format!("{}_a1b1_constraint", label),
        |lc| lc + a.c1,
        |lc| lc + b.c1,
        |lc| lc + a1_b1,
    );

    cs.enforce(
        || format!("{}_a0b1_constraint", label),
        |lc| lc + a.c0,
        |lc| lc + b.c1,
        |lc| lc + a0_b1,
    );

    cs.enforce(
        || format!("{}_a1b0_constraint", label),
        |lc| lc + a.c1,
        |lc| lc + b.c0,
        |lc| lc + a1_b0,
    );

    // Enforce c0 = a0_b0 + δ * a1_b1
    cs.enforce(
        || format!("{}_c0_constraint", label),
        |lc| lc + a0_b0 + (delta, a1_b1),
        |lc| lc + CS::one(),
        |lc| lc + c0,
    );

    // Enforce c1 = a0_b1 + a1_b0
    cs.enforce(
        || format!("{}_c1_constraint", label),
        |lc| lc + a0_b1 + a1_b0,
        |lc| lc + CS::one(),
        |lc| lc + c1,
    );

    Ok(KNumVar { c0, c1 })
}

/// K-field scalar multiplication by F constant: k * (c0, c1) = (k*c0, k*c1)
///
/// value_hint: Optional precomputed result for witness generation
pub fn k_scalar_mul<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    k: F,
    a: &KNumVar,
    value_hint: Option<KNum<F>>,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    // c0 = k * a.c0
    let c0_val = value_hint.as_ref().map(|v| v.c0);
    let c0 = cs.alloc(
        || format!("{}_scaled_c0", label),
        || Ok(c0_val.unwrap_or(F::ZERO)),
    )?;

    cs.enforce(
        || format!("{}_c0_constraint", label),
        |lc| lc + (k, a.c0),
        |lc| lc + CS::one(),
        |lc| lc + c0,
    );

    // c1 = k * a.c1
    let c1_val = value_hint.as_ref().map(|v| v.c1);
    let c1 = cs.alloc(
        || format!("{}_scaled_c1", label),
        || Ok(c1_val.unwrap_or(F::ZERO)),
    )?;

    cs.enforce(
        || format!("{}_c1_constraint", label),
        |lc| lc + (k, a.c1),
        |lc| lc + CS::one(),
        |lc| lc + c1,
    );

    Ok(KNumVar { c0, c1 })
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    use bellpepper_core::test_cs::TestConstraintSystem;
    use p3_goldilocks::Goldilocks;

    // Note: Full tests would require proper PrimeField implementation for Goldilocks
    // This is a placeholder showing the structure
}
