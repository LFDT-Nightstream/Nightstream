//! Packed public-input matrix branch of Π_RLC.V.
//!
//! Owns: active public-X ring combination and inactive-column zero padding.
//!
//! Does not own: active-width derivation or transcript binding.
//!
//! Emits constraints: yes; allocation helpers emit none.
//!
//! Authority boundary: `active_cols` is verifier-derived from `m_in`; inactive
//! witness columns are constrained to zero and cannot carry hidden authority.
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
    pub m_in: usize,
}

/// Wires for all input matrices and the expected combined matrix.
#[derive(Clone, Debug)]
pub struct RlcXWires {
    pub inputs: Vec<RlcXPairWires>,
    pub combined_x_flat: Vec<Var>,
    pub m_in: usize,
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
    let m_in = inputs_x[0].cols();
    validate_combined_shape(combined_x, m_in)?;

    let mut input_wires = Vec::with_capacity(inputs_x.len());
    for (idx, (rho_col, x_i)) in rhos_first_col.iter().zip(inputs_x.iter()).enumerate() {
        validate_input_shape(x_i, m_in, idx)?;
        let mut rho_coeffs = [Var::ONE; D];
        for (slot, &value) in rho_coeffs.iter_mut().zip(rho_col.iter()) {
            *slot = builder.alloc(value);
        }
        input_wires.push(RlcXPairWires {
            rho_coeffs,
            x_flat: builder.alloc_vec(x_i.as_slice()),
            m_in,
        });
    }
    Ok(RlcXWires {
        inputs: input_wires,
        combined_x_flat: builder.alloc_vec(combined_x.as_slice()),
        m_in,
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
    let m_in = inputs_x[0].cols();
    validate_combined_shape(combined_x, m_in)?;

    let mut input_wires = Vec::with_capacity(inputs_x.len());
    for (idx, (rho, x_i)) in rho_wires.iter().zip(inputs_x.iter()).enumerate() {
        validate_input_shape(x_i, m_in, idx)?;
        input_wires.push(RlcXPairWires {
            rho_coeffs: *rho,
            x_flat: builder.alloc_vec(x_i.as_slice()),
            m_in,
        });
    }
    Ok(RlcXWires {
        inputs: input_wires,
        combined_x_flat: builder.alloc_vec(combined_x.as_slice()),
        m_in,
    })
}

fn validate_combined_shape(combined_x: &neo_ccs::Mat<F>, m_in: usize) -> Result<(), Error> {
    if combined_x.rows() != D || combined_x.cols() != m_in {
        return Err(Error::ShapeMismatch {
            what: "combined X shape",
            expected: format!("(rows=D, cols={m_in})"),
            got: format!("(rows={}, cols={})", combined_x.rows(), combined_x.cols()),
        });
    }
    Ok(())
}

fn validate_input_shape(x_i: &neo_ccs::Mat<F>, m_in: usize, idx: usize) -> Result<(), Error> {
    if x_i.rows() != D || x_i.cols() != m_in {
        return Err(Error::ShapeMismatch {
            what: "input X shape",
            expected: format!("(rows=D, cols={m_in})"),
            got: format!("(rows={}, cols={}) at idx {idx}", x_i.rows(), x_i.cols()),
        });
    }
    Ok(())
}

/// Enforce `combined.X = sum_i rho_i * X_i` column-by-column.
pub fn enforce_rlc_x_combination(builder: &mut R1csBuilder, wires: &RlcXWires) {
    let m_in = wires.m_in;
    let active_cols = crate::paper::relations::superneo_public_x_cols(m_in);

    for pair in &wires.inputs {
        for row in 0..D {
            for column in active_cols..m_in {
                builder.enforce_eq(&Lc::from_var(pair.x_flat[row * m_in + column]), &Lc::zero());
            }
        }
    }

    let mut per_pair_per_col: Vec<Vec<[Var; D]>> = Vec::with_capacity(wires.inputs.len());
    for pair in &wires.inputs {
        let mut per_col = Vec::with_capacity(active_cols);
        for column in 0..active_cols {
            let mut x_col = [Var::ONE; D];
            for (row, slot) in x_col.iter_mut().enumerate() {
                *slot = pair.x_flat[row * m_in + column];
            }
            per_col.push(enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &x_col));
        }
        per_pair_per_col.push(per_col);
    }

    for row in 0..D {
        for column in 0..active_cols {
            let mut combination = Lc::zero();
            for per_col in &per_pair_per_col {
                combination.add_term(per_col[column][row], F::ONE);
            }
            let target = wires.combined_x_flat[row * m_in + column];
            builder.enforce_eq(&Lc::from_var(target), &combination);
        }
    }

    for row in 0..D {
        for column in active_cols..m_in {
            builder.enforce_eq(&Lc::from_var(wires.combined_x_flat[row * m_in + column]), &Lc::zero());
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
            .map(|pair| core::array::from_fn(|row| pair.x_flat[row * wires.m_in + column]))
            .collect();
        let pair_refs: Vec<(&[Var; D], &[Var; D])> = wires
            .inputs
            .iter()
            .zip(inputs.iter())
            .map(|(pair, input)| (&pair.rho_coeffs, input))
            .collect();
        let output = core::array::from_fn(|row| wires.combined_x_flat[row * wires.m_in + column]);
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

/// Pin every inactive input and output `X` column to zero.
pub fn enforce_rlc_x_padding_glue(builder: &mut R1csBuilder, wires: &RlcXWires) -> Result<(), Error> {
    let active_cols = validate_rlc_x_projection_shape(wires)?;
    let inactive_inputs = wires.inputs.iter().flat_map(|pair| {
        (0..D).flat_map(move |row| (active_cols..wires.m_in).map(move |column| pair.x_flat[row * wires.m_in + column]))
    });
    let inactive_output = (0..D)
        .flat_map(|row| (active_cols..wires.m_in).map(move |column| wires.combined_x_flat[row * wires.m_in + column]));
    enforce_unique_zero_wires(builder, inactive_inputs.chain(inactive_output));
    Ok(())
}

fn validate_rlc_x_projection_shape(wires: &RlcXWires) -> Result<usize, Error> {
    if wires.inputs.is_empty() {
        return Err(Error::Empty);
    }
    let active_cols = crate::paper::relations::superneo_public_x_cols(wires.m_in);
    if wires.combined_x_flat.len() != D * wires.m_in {
        return Err(Error::ShapeMismatch {
            what: "combined X projection shape",
            expected: format!("{} coefficients", D * wires.m_in),
            got: format!("{}", wires.combined_x_flat.len()),
        });
    }
    for (idx, pair) in wires.inputs.iter().enumerate() {
        if pair.m_in != wires.m_in || pair.x_flat.len() != D * wires.m_in {
            return Err(Error::ShapeMismatch {
                what: "input X projection shape",
                expected: format!("(m_in={}, data={})", wires.m_in, D * wires.m_in),
                got: format!("(m_in={}, data={}) at idx {idx}", pair.m_in, pair.x_flat.len()),
            });
        }
    }
    Ok(active_cols)
}

fn enforce_unique_zero_wires(builder: &mut R1csBuilder, wires: impl Iterator<Item = Var>) {
    let mut constrained = std::collections::HashSet::new();
    for wire in wires {
        if constrained.insert(wire.col()) {
            builder.enforce_eq(&Lc::from_var(wire), &Lc::zero());
        }
    }
}
