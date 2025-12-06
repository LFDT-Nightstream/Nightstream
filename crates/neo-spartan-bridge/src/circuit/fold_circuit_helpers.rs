//! Helper functions for FoldRunCircuit
//!
//! This module contains utility functions used by the FoldRunCircuit
//! for field conversions, K-field operations, and allocation helpers.

use crate::error::{Result, SpartanBridgeError};
use crate::gadgets::k_field::{
    alloc_k, k_add as k_add_raw, k_mul as k_mul_raw, k_scalar_mul as k_scalar_mul_raw, KNum, KNumVar,
};
use crate::CircuitF;
use bellpepper_core::{ConstraintSystem, Variable};
use neo_ccs::Mat;
use neo_math::F as NeoF;
use p3_field::PrimeField64;

/// Helper: convert Neo base-field element to circuit field
pub fn neo_f_to_circuit(f: &NeoF) -> CircuitF {
    CircuitF::from(f.as_canonical_u64())
}

/// Helper: allocate a K element from neo_math::K
pub fn alloc_k_from_neo<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, k: neo_math::K, label: &str) -> Result<KNumVar> {
    let k_num = KNum::<CircuitF>::from_neo_k(k);
    alloc_k(cs, Some(k_num), label).map_err(SpartanBridgeError::BellpepperError)
}

/// Helper: allocate a dense matrix of NeoF as circuit variables.
pub fn alloc_matrix_from_neo<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    mat: &Mat<NeoF>,
    label: &str,
) -> Result<Vec<Vec<Variable>>> {
    let rows = mat.rows();
    let cols = mat.cols();
    let mut vars = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row_vars = Vec::with_capacity(cols);
        for c in 0..cols {
            let value = neo_f_to_circuit(&mat[(r, c)]);
            let var = cs.alloc(|| format!("{}_{}_{}", label, r, c), || Ok(value))?;
            row_vars.push(var);
        }
        vars.push(row_vars);
    }
    Ok(vars)
}

/// Helper: allocate a table of K elements (y-vectors) from neo_math::K.
pub fn alloc_y_table_from_neo<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    y: &[Vec<neo_math::K>],
    label: &str,
) -> Result<Vec<Vec<KNumVar>>> {
    let mut table = Vec::with_capacity(y.len());
    for (j, row) in y.iter().enumerate() {
        let mut row_vars = Vec::with_capacity(row.len());
        for (idx, k_val) in row.iter().enumerate() {
            let var = alloc_k_from_neo(cs, *k_val, &format!("{}_{}_{}", label, j, idx))?;
            row_vars.push(var);
        }
        table.push(row_vars);
    }
    Ok(table)
}

/// Helper: enforce equality of two KNumVars.
pub fn enforce_k_eq<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, a: &KNumVar, b: &KNumVar, label: &str) {
    cs.enforce(
        || format!("{}_c0", label),
        |lc| lc + a.c0,
        |lc| lc + CS::one(),
        |lc| lc + b.c0,
    );
    cs.enforce(
        || format!("{}_c1", label),
        |lc| lc + a.c1,
        |lc| lc + CS::one(),
        |lc| lc + b.c1,
    );
}

/// Helper: allocate a constant K element from a base-field value.
pub fn k_const<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, c0: CircuitF, label: &str) -> Result<KNumVar> {
    let k_num = KNum::<CircuitF>::from_f(c0);
    alloc_k(cs, Some(k_num), label).map_err(SpartanBridgeError::BellpepperError)
}

/// Helper: K zero.
pub fn k_zero<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, label: &str) -> Result<KNumVar> {
    k_const(cs, CircuitF::from(0u64), label)
}

/// Helper: K one.
pub fn k_one<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, label: &str) -> Result<KNumVar> {
    k_const(cs, CircuitF::from(1u64), label)
}

/// Helper: K addition: r = a + b.
pub fn k_add<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, a: &KNumVar, b: &KNumVar, label: &str) -> Result<KNumVar> {
    k_add_raw(cs, a, b, None, label).map_err(SpartanBridgeError::BellpepperError)
}

/// Helper: K multiplication: r = a * b.
pub fn k_mul<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    a: &KNumVar,
    b: &KNumVar,
    delta: CircuitF,
    label: &str,
) -> Result<KNumVar> {
    k_mul_raw(cs, a, b, delta, None, label).map_err(SpartanBridgeError::BellpepperError)
}

/// Helper: K scalar multiplication: r = k * a.
pub fn k_scalar_mul<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    k: CircuitF,
    a: &KNumVar,
    label: &str,
) -> Result<KNumVar> {
    k_scalar_mul_raw(cs, k, a, None, label).map_err(SpartanBridgeError::BellpepperError)
}

/// Helper: K subtraction: r = a - b.
pub fn k_sub<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, a: &KNumVar, b: &KNumVar, label: &str) -> Result<KNumVar> {
    // Compute -b via scalar multiplication by -1, then add.
    let minus_one = CircuitF::from(0u64) - CircuitF::from(1u64);
    let neg_b = k_scalar_mul(cs, minus_one, b, &format!("{}_neg_b", label))?;
    k_add(cs, a, &neg_b, label)
}

/// Helper: 1 - a.
pub fn k_one_minus<CS: ConstraintSystem<CircuitF>>(cs: &mut CS, a: &KNumVar, label: &str) -> Result<KNumVar> {
    let one = k_one(cs, &format!("{}_one", label))?;
    k_sub(cs, &one, a, &format!("{}_1_minus", label))
}

/// Helper: K multiplication with native K hints for operands and product.
///
/// Returns both the resulting KNumVar and its native K value.
pub fn k_mul_with_hint<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    a: &KNumVar,
    a_val: neo_math::K,
    b: &KNumVar,
    b_val: neo_math::K,
    delta: CircuitF,
    label: &str,
) -> Result<(KNumVar, neo_math::K)> {
    use crate::gadgets::k_field::KNum as KNumField;

    let a_hint = KNumField::<CircuitF>::from_neo_k(a_val);
    let b_hint = KNumField::<CircuitF>::from_neo_k(b_val);
    let prod_val = a_val * b_val;
    let prod_hint = KNumField::<CircuitF>::from_neo_k(prod_val);

    // Allocate result components with known product hint.
    let c0 = cs.alloc(|| format!("{}_prod_c0", label), || Ok(prod_hint.c0))?;

    let c1 = cs.alloc(|| format!("{}_prod_c1", label), || Ok(prod_hint.c1))?;

    // Intermediate products in the base field.
    let a0b0_val = a_hint.c0 * b_hint.c0;
    let a1b1_val = a_hint.c1 * b_hint.c1;
    let a0b1_val = a_hint.c0 * b_hint.c1;
    let a1b0_val = a_hint.c1 * b_hint.c0;

    let a0_b0 = cs.alloc(|| format!("{}_a0b0", label), || Ok(a0b0_val))?;

    let a1_b1 = cs.alloc(|| format!("{}_a1b1", label), || Ok(a1b1_val))?;

    let a0_b1 = cs.alloc(|| format!("{}_a0b1", label), || Ok(a0b1_val))?;

    let a1_b0 = cs.alloc(|| format!("{}_a1b0", label), || Ok(a1b0_val))?;

    // Enforce intermediate products.
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

    Ok((KNumVar { c0, c1 }, prod_val))
}

/// Helper: K exponentiation by small integer exponent: base^exp.
#[allow(dead_code)]
pub fn k_pow<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    base: &KNumVar,
    exp: u32,
    delta: CircuitF,
    label: &str,
) -> Result<KNumVar> {
    if exp == 0 {
        return k_one(cs, &format!("{}_pow0", label));
    }
    let mut acc = base.clone();
    for i in 1..exp {
        acc = k_mul(cs, &acc, base, delta, &format!("{}_pow_step_{}", label, i))?;
    }
    Ok(acc)
}
