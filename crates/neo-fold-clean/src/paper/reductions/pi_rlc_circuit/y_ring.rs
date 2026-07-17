//! Unpadded `y_ring` row branch of Π_RLC.V.
//!
//! Owns: exact unpadded extension-limb combination for one `y_ring` row.
//!
//! Does not own: padding, projection binding, or row iteration.
//!
//! Emits constraints: yes; allocation helpers emit none.
//!
//! Authority boundary: every input and output has exact active length `D`.
//!
//! | Stage path | Function | Equation | Multiplicity | Emitted rows/formula | Lowered gate | Lean theorem |
//! |---|---|---|---:|---|---|---|
//! | y-ring combination | `alloc_rlc_y_row_inputs*` | typed active row view | once per row | none | none | `YRingCombination` shape |
//! | y-ring combination | `enforce_rlc_y_row_combination` | `y_j=sum_i rho_i*y_(i,j)` on both limbs | inputs × 2 limbs | Toom-3 products plus equalities | ring product | `yRingCombinationWithIntermediates_iff_direct` |

use neo_math::ring::D;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

use super::Error;
use crate::engine::r1cs_circuit::ring_action::enforce_ring_mul_toom3;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};

/// Wires for one input's `y_ring[j]` row and matching rho polynomial.
#[derive(Clone, Debug)]
pub struct RlcYRowPairWires {
    pub rho_coeffs: [Var; D],
    pub y_c0: [Var; D],
    pub y_c1: [Var; D],
}

/// Wires for all input rows and the expected combined row.
#[derive(Clone, Debug)]
pub struct RlcYRowWires {
    pub inputs: Vec<RlcYRowPairWires>,
    pub combined_c0: [Var; D],
    pub combined_c1: [Var; D],
}

/// Allocate one `y_ring` row combination without emitting constraints.
pub fn alloc_rlc_y_row_inputs(
    builder: &mut R1csBuilder,
    rhos_first_col: &[[F; D]],
    inputs_y: &[Vec<K>],
    combined_y: &[K],
) -> Result<RlcYRowWires, Error> {
    validate_y_rows(inputs_y, combined_y, rhos_first_col.len())?;

    let mut input_wires = Vec::with_capacity(inputs_y.len());
    for (rho_col, y_i) in rhos_first_col.iter().zip(inputs_y.iter()) {
        let mut rho_coeffs = [Var::ONE; D];
        for (slot, &value) in rho_coeffs.iter_mut().zip(rho_col.iter()) {
            *slot = builder.alloc(value);
        }
        let (y_c0, y_c1) = alloc_k_limbs(builder, y_i);
        input_wires.push(RlcYRowPairWires { rho_coeffs, y_c0, y_c1 });
    }
    let (combined_c0, combined_c1) = alloc_k_limbs(builder, combined_y);
    Ok(RlcYRowWires {
        inputs: input_wires,
        combined_c0,
        combined_c1,
    })
}

/// Allocate one row combination while reusing transcript-derived rho wires.
pub fn alloc_rlc_y_row_inputs_with_rhos(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    inputs_y: &[Vec<K>],
    combined_y: &[K],
) -> Result<RlcYRowWires, Error> {
    validate_y_rows(inputs_y, combined_y, rho_wires.len())?;

    let mut input_wires = Vec::with_capacity(inputs_y.len());
    for (rho, y_i) in rho_wires.iter().zip(inputs_y.iter()) {
        let (y_c0, y_c1) = alloc_k_limbs(builder, y_i);
        input_wires.push(RlcYRowPairWires {
            rho_coeffs: *rho,
            y_c0,
            y_c1,
        });
    }
    let (combined_c0, combined_c1) = alloc_k_limbs(builder, combined_y);
    Ok(RlcYRowWires {
        inputs: input_wires,
        combined_c0,
        combined_c1,
    })
}

fn validate_y_rows(inputs_y: &[Vec<K>], combined_y: &[K], rho_count: usize) -> Result<(), Error> {
    if inputs_y.is_empty() {
        return Err(Error::Empty);
    }
    if rho_count != inputs_y.len() {
        return Err(Error::PairCountMismatch {
            rhos: rho_count,
            inputs: inputs_y.len(),
        });
    }
    for (idx, y) in inputs_y.iter().enumerate() {
        if y.len() != D {
            return Err(Error::ShapeMismatch {
                what: "y_ring row length",
                expected: format!("{D}"),
                got: format!("{} at idx {idx}", y.len()),
            });
        }
    }
    if combined_y.len() != D {
        return Err(Error::ShapeMismatch {
            what: "combined y_ring row length",
            expected: format!("{D}"),
            got: format!("{}", combined_y.len()),
        });
    }
    Ok(())
}

fn alloc_k_limbs(builder: &mut R1csBuilder, values: &[K]) -> ([Var; D], [Var; D]) {
    let mut c0 = [Var::ONE; D];
    let mut c1 = [Var::ONE; D];
    for (index, value) in values.iter().enumerate() {
        let [value_c0, value_c1] = value.as_coeffs();
        c0[index] = builder.alloc(value_c0);
        c1[index] = builder.alloc(value_c1);
    }
    (c0, c1)
}

/// Enforce one `y_ring` row as `combined.y = sum_i rho_i * input_i.y`.
pub fn enforce_rlc_y_row_combination(builder: &mut R1csBuilder, wires: &RlcYRowWires) {
    let mut per_pair_c0: Vec<[Var; D]> = Vec::with_capacity(wires.inputs.len());
    let mut per_pair_c1: Vec<[Var; D]> = Vec::with_capacity(wires.inputs.len());
    for pair in &wires.inputs {
        per_pair_c0.push(enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &pair.y_c0));
        per_pair_c1.push(enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &pair.y_c1));
    }
    for row in 0..D {
        let mut combination_c0 = Lc::zero();
        let mut combination_c1 = Lc::zero();
        for (product_c0, product_c1) in per_pair_c0.iter().zip(per_pair_c1.iter()) {
            combination_c0.add_term(product_c0[row], F::ONE);
            combination_c1.add_term(product_c1[row], F::ONE);
        }
        builder.enforce_eq(&Lc::from_var(wires.combined_c0[row]), &combination_c0);
        builder.enforce_eq(&Lc::from_var(wires.combined_c1[row]), &combination_c1);
    }
}
