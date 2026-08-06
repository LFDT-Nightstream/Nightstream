//! Packed public-input matrix branch of Π_RLC.V.
//!
//! Owns: public-X ring combination over the compact coefficient embedding.
//!
//! Does not own: active-width derivation or transcript binding.
//!
//! Emits constraints: yes; allocation helpers emit none.
//!
//! Authority boundary: the verifier requires the exact coefficient-embedding
//! width, so every stored X column is active.
//! For the fixed profile, the five active rings contain 270 coefficients while
//! CCS exposes 257 scalar fields. This module proves implementation arithmetic
//! only; the paper `L_in` representation/refinement remains open.
//!
//! | Stage path | Function | Equation | Multiplicity | Emitted rows/formula | Lowered gate | Lean theorem |
//! |---|---|---|---:|---|---|---|
//! | `fold_wires.x` | `alloc_rlc_x_inputs*` | typed `(rho_i,X_i,parent)` view | once | none | none | parameter shape |
//! | full X combination | `enforce_rlc_x_combination` | `X=sum_i rho_i*X_i` on active columns | active columns × inputs | ring products plus equalities | ring product | `xCombinationWithIntermediates_iff_direct` |
//! | `identities.x` | projection helper | active aggregate identity at beta | one per active column | polynomial rows | product-sum | exact-or-bad-root bridge open |
//! | `padding.x` | `enforce_rlc_x_padding_glue` | inactive input/output columns are zero | inactive cells | one equality each | linear | `PaddingZero` analogue |

use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::Error;
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::ring_action::{
    enforce_ring_action_projection_batch_with_rho_evaluations_and_stages, enforce_ring_mul_toom3,
    PolynomialEvaluationsAtBeta, ProjectionIdentityStageLabels, PROJECTION_QUOTIENT_LEN,
};
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};

/// Wires for one input matrix and its rho polynomial.
#[derive(Clone, Debug)]
pub struct RlcXPairWires {
    pub rho_coeffs: [Var; D],
    pub x_flat: Vec<Var>,
    pub x_cols: usize,
}

/// Wires for all input matrices and the expected combined matrix.
#[derive(Clone, Debug)]
pub struct RlcXWires {
    pub inputs: Vec<RlcXPairWires>,
    pub combined_x_flat: Vec<Var>,
    pub x_cols: usize,
}

/// Allocate X-combination witnesses without emitting constraints.
pub fn alloc_rlc_x_inputs(
    builder: &mut R1csBuilder,
    rhos_first_col: &[[F; D]],
    inputs_x: &[neo_ccs::Mat<F>],
    combined_x: &neo_ccs::Mat<F>,
) -> Result<RlcXWires, Error> {
    if inputs_x.is_empty() {
        return Err(Error::Empty);
    }
    if rhos_first_col.len() != inputs_x.len() {
        return Err(Error::PairCountMismatch {
            rhos: rhos_first_col.len(),
            inputs: inputs_x.len(),
        });
    }
    let x_cols = inputs_x[0].cols();
    validate_combined_shape(combined_x, x_cols)?;

    let mut input_wires = Vec::with_capacity(inputs_x.len());
    for (idx, (rho_col, x_i)) in rhos_first_col.iter().zip(inputs_x.iter()).enumerate() {
        validate_input_shape(x_i, x_cols, idx)?;
        let mut rho_coeffs = [Var::ONE; D];
        for (slot, &value) in rho_coeffs.iter_mut().zip(rho_col.iter()) {
            *slot = builder.alloc(value);
        }
        input_wires.push(RlcXPairWires {
            rho_coeffs,
            x_flat: builder.alloc_vec(x_i.as_slice()),
            x_cols,
        });
    }
    Ok(RlcXWires {
        inputs: input_wires,
        combined_x_flat: builder.alloc_vec(combined_x.as_slice()),
        x_cols,
    })
}

/// Allocate X-combination witnesses while reusing transcript-derived rho wires.
pub fn alloc_rlc_x_inputs_with_rhos(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    inputs_x: &[neo_ccs::Mat<F>],
    combined_x: &neo_ccs::Mat<F>,
) -> Result<RlcXWires, Error> {
    if inputs_x.is_empty() {
        return Err(Error::Empty);
    }
    if rho_wires.len() != inputs_x.len() {
        return Err(Error::PairCountMismatch {
            rhos: rho_wires.len(),
            inputs: inputs_x.len(),
        });
    }
    let x_cols = inputs_x[0].cols();
    validate_combined_shape(combined_x, x_cols)?;

    let mut input_wires = Vec::with_capacity(inputs_x.len());
    for (idx, (rho, x_i)) in rho_wires.iter().zip(inputs_x.iter()).enumerate() {
        validate_input_shape(x_i, x_cols, idx)?;
        input_wires.push(RlcXPairWires {
            rho_coeffs: *rho,
            x_flat: builder.alloc_vec(x_i.as_slice()),
            x_cols,
        });
    }
    Ok(RlcXWires {
        inputs: input_wires,
        combined_x_flat: builder.alloc_vec(combined_x.as_slice()),
        x_cols,
    })
}

fn validate_combined_shape(combined_x: &neo_ccs::Mat<F>, x_cols: usize) -> Result<(), Error> {
    if combined_x.rows() != D || combined_x.cols() != x_cols {
        return Err(Error::ShapeMismatch {
            what: "combined X shape",
            expected: format!("(rows=D, cols={x_cols})"),
            got: format!("(rows={}, cols={})", combined_x.rows(), combined_x.cols()),
        });
    }
    Ok(())
}

fn validate_input_shape(x_i: &neo_ccs::Mat<F>, x_cols: usize, idx: usize) -> Result<(), Error> {
    if x_i.rows() != D || x_i.cols() != x_cols {
        return Err(Error::ShapeMismatch {
            what: "input X shape",
            expected: format!("(rows=D, cols={x_cols})"),
            got: format!("(rows={}, cols={}) at idx {idx}", x_i.rows(), x_i.cols()),
        });
    }
    Ok(())
}

/// Enforce `combined.X = sum_i rho_i * X_i` column-by-column.
pub fn enforce_rlc_x_combination(builder: &mut R1csBuilder, wires: &RlcXWires) {
    let x_cols = wires.x_cols;

    let mut per_pair_per_col: Vec<Vec<[Var; D]>> = Vec::with_capacity(wires.inputs.len());
    for pair in &wires.inputs {
        let mut per_col = Vec::with_capacity(x_cols);
        for column in 0..x_cols {
            let mut x_col = [Var::ONE; D];
            for (row, slot) in x_col.iter_mut().enumerate() {
                *slot = pair.x_flat[row * x_cols + column];
            }
            per_col.push(enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &x_col));
        }
        per_pair_per_col.push(per_col);
    }

    for row in 0..D {
        for column in 0..x_cols {
            let mut combination = Lc::zero();
            for per_col in &per_pair_per_col {
                combination.add_term(per_col[column][row], F::ONE);
            }
            let target = wires.combined_x_flat[row * x_cols + column];
            builder.enforce_eq(&Lc::from_var(target), &combination);
        }
    }
}

/// Projection-checked `X` combination. Kept as the compact public operation;
/// the NIFS profiler calls the two phases separately to expose exact owners.
pub fn enforce_rlc_x_combination_projection_with_quotient_wires(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    rho_evaluations: &PolynomialEvaluationsAtBeta,
    wires: &RlcXWires,
    quotient_wires: &[[Var; PROJECTION_QUOTIENT_LEN]],
) -> Result<(), Error> {
    enforce_rlc_x_projection_identities_with_quotient_wires(builder, powers, rho_evaluations, wires, quotient_wires)?;
    enforce_rlc_x_padding_glue(builder, wires)
}

/// Emit one projection identity per active `X` ring column.
pub fn enforce_rlc_x_projection_identities_with_quotient_wires(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    rho_evaluations: &PolynomialEvaluationsAtBeta,
    wires: &RlcXWires,
    quotient_wires: &[[Var; PROJECTION_QUOTIENT_LEN]],
) -> Result<(), Error> {
    enforce_rlc_x_projection_identities_with_quotient_wires_and_stages(
        builder,
        powers,
        rho_evaluations,
        wires,
        quotient_wires,
        None,
    )
}

/// Active X identities with diagnostic phase labels.
pub fn enforce_rlc_x_projection_identities_with_quotient_wires_and_stages(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    rho_evaluations: &PolynomialEvaluationsAtBeta,
    wires: &RlcXWires,
    quotient_wires: &[[Var; PROJECTION_QUOTIENT_LEN]],
    stages: Option<ProjectionIdentityStageLabels>,
) -> Result<(), Error> {
    let active_cols = validate_rlc_x_projection_shape(wires)?;
    if quotient_wires.len() != active_cols {
        return Err(Error::ShapeMismatch {
            what: "X projection quotient count",
            expected: format!("{active_cols}"),
            got: format!("{}", quotient_wires.len()),
        });
    }

    for column in 0..active_cols {
        let inputs: Vec<[Var; D]> = wires
            .inputs
            .iter()
            .map(|pair| core::array::from_fn(|row| pair.x_flat[row * wires.x_cols + column]))
            .collect();
        let pair_refs: Vec<(&[Var; D], &[Var; D])> = wires
            .inputs
            .iter()
            .zip(inputs.iter())
            .map(|(pair, input)| (&pair.rho_coeffs, input))
            .collect();
        let output = core::array::from_fn(|row| wires.combined_x_flat[row * wires.x_cols + column]);
        enforce_ring_action_projection_batch_with_rho_evaluations_and_stages(
            builder,
            powers,
            rho_evaluations,
            &pair_refs,
            &output,
            &quotient_wires[column],
            stages,
        );
    }
    Ok(())
}

/// Validate that no noncanonical padding columns exist.
pub fn enforce_rlc_x_padding_glue(_builder: &mut R1csBuilder, wires: &RlcXWires) -> Result<(), Error> {
    validate_rlc_x_projection_shape(wires)?;
    Ok(())
}

fn validate_rlc_x_projection_shape(wires: &RlcXWires) -> Result<usize, Error> {
    if wires.inputs.is_empty() {
        return Err(Error::Empty);
    }
    let active_cols = wires.x_cols;
    if wires.combined_x_flat.len() != D * wires.x_cols {
        return Err(Error::ShapeMismatch {
            what: "combined X projection shape",
            expected: format!("{} coefficients", D * wires.x_cols),
            got: format!("{}", wires.combined_x_flat.len()),
        });
    }
    for (idx, pair) in wires.inputs.iter().enumerate() {
        if pair.x_cols != wires.x_cols || pair.x_flat.len() != D * wires.x_cols {
            return Err(Error::ShapeMismatch {
                what: "input X projection shape",
                expected: format!("(x_cols={}, data={})", wires.x_cols, D * wires.x_cols),
                got: format!("(x_cols={}, data={}) at idx {idx}", pair.x_cols, pair.x_flat.len()),
            });
        }
    }
    Ok(active_cols)
}
