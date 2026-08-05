//! Padded ring-vector arithmetic for the in-circuit Π_RLC verifier.
//!
//! The active `D` coefficients use the ring action. Every encoding tail
//! coefficient is constrained to zero. Transcript binding and claim iteration
//! are owned by the NIFS circuit.
//!
//! Owns: padded ring-vector allocation, active ring action, and zero tails.
//!
//! Does not own: transcript binding, claim iteration, or challenge sampling.
//!
//! Emits constraints: ring multiplication, projection identities, and
//! canonical zero padding.
//!
//! | Region | Constraint family |
//! | --- | --- |
//! | active coefficients | Phi81 ring action |
//! | encoding tail | equality to zero |

use neo_math::ring::D;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

use super::Error;
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::ring_action::{
    enforce_ring_action_projection_batch_with_rho_evaluations_and_stages, enforce_ring_mul_toom3,
    PolynomialEvaluationsAtBeta, ProjectionIdentityStageLabels, PROJECTION_QUOTIENT_LEN,
};
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};

/// One padded input K-vector and its rho polynomial.
#[derive(Clone, Debug)]
pub struct RlcPaddedKVectorPairWires {
    pub rho_coeffs: [Var; D],
    pub y_c0: Vec<Var>,
    pub y_c1: Vec<Var>,
}

/// All padded input vectors and the expected combined vector.
#[derive(Clone, Debug)]
pub struct RlcPaddedKVectorWires {
    pub inputs: Vec<RlcPaddedKVectorPairWires>,
    pub combined_c0: Vec<Var>,
    pub combined_c1: Vec<Var>,
    pub d_pad: usize,
}

fn alloc_padded_inputs_inner(
    builder: &mut R1csBuilder,
    inputs_y: &[Vec<K>],
    combined_y: &[K],
    d_pad: usize,
    rho_provider: impl Fn(&mut R1csBuilder, usize) -> [Var; D],
) -> Result<RlcPaddedKVectorWires, Error> {
    if inputs_y.is_empty() {
        return Err(Error::Empty);
    }
    if d_pad < D {
        return Err(Error::ShapeMismatch {
            what: "d_pad < D",
            expected: format!(">= {D}"),
            got: format!("{d_pad}"),
        });
    }
    for (idx, y) in inputs_y.iter().enumerate() {
        if y.len() != d_pad {
            return Err(Error::ShapeMismatch {
                what: "padded y length",
                expected: format!("{d_pad}"),
                got: format!("{} at idx {idx}", y.len()),
            });
        }
    }
    if combined_y.len() != d_pad {
        return Err(Error::ShapeMismatch {
            what: "combined padded y length",
            expected: format!("{d_pad}"),
            got: format!("{}", combined_y.len()),
        });
    }

    let mut input_wires = Vec::with_capacity(inputs_y.len());
    for (idx, y_i) in inputs_y.iter().enumerate() {
        let rho_coeffs = rho_provider(builder, idx);
        let mut y_c0 = Vec::with_capacity(d_pad);
        let mut y_c1 = Vec::with_capacity(d_pad);
        for value in y_i {
            let [c0, c1] = value.as_coeffs();
            y_c0.push(builder.alloc(c0));
            y_c1.push(builder.alloc(c1));
        }
        input_wires.push(RlcPaddedKVectorPairWires { rho_coeffs, y_c0, y_c1 });
    }
    let mut combined_c0 = Vec::with_capacity(d_pad);
    let mut combined_c1 = Vec::with_capacity(d_pad);
    for value in combined_y {
        let [c0, c1] = value.as_coeffs();
        combined_c0.push(builder.alloc(c0));
        combined_c1.push(builder.alloc(c1));
    }
    Ok(RlcPaddedKVectorWires {
        inputs: input_wires,
        combined_c0,
        combined_c1,
        d_pad,
    })
}

/// Allocate padded K-vector witnesses, including rho witnesses.
pub fn alloc_rlc_padded_k_vector_inputs(
    builder: &mut R1csBuilder,
    rhos_first_col: &[[F; D]],
    inputs_y: &[Vec<K>],
    combined_y: &[K],
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    if rhos_first_col.len() != inputs_y.len() {
        return Err(Error::PairCountMismatch {
            rhos: rhos_first_col.len(),
            inputs: inputs_y.len(),
        });
    }
    alloc_padded_inputs_inner(builder, inputs_y, combined_y, d_pad, |b, idx| {
        let mut rho = [Var::ONE; D];
        for (slot, &value) in rho.iter_mut().zip(rhos_first_col[idx].iter()) {
            *slot = b.alloc(value);
        }
        rho
    })
}

/// Allocate padded K-vector witnesses while reusing transcript-derived rho wires.
pub fn alloc_rlc_padded_k_vector_inputs_with_rhos(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    inputs_y: &[Vec<K>],
    combined_y: &[K],
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    if rho_wires.len() != inputs_y.len() {
        return Err(Error::PairCountMismatch {
            rhos: rho_wires.len(),
            inputs: inputs_y.len(),
        });
    }
    alloc_padded_inputs_inner(builder, inputs_y, combined_y, d_pad, |_b, idx| rho_wires[idx])
}

/// Enforce the active ring fold and canonical zero padding.
pub fn enforce_rlc_padded_k_vector_combination(builder: &mut R1csBuilder, wires: &RlcPaddedKVectorWires) {
    let mut per_pair_c0: Vec<[Var; D]> = Vec::with_capacity(wires.inputs.len());
    let mut per_pair_c1: Vec<[Var; D]> = Vec::with_capacity(wires.inputs.len());
    for pair in &wires.inputs {
        let mut y_c0 = [Var::ONE; D];
        let mut y_c1 = [Var::ONE; D];
        for index in 0..D {
            y_c0[index] = pair.y_c0[index];
            y_c1[index] = pair.y_c1[index];
        }
        per_pair_c0.push(enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &y_c0));
        per_pair_c1.push(enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &y_c1));
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

    for pair in &wires.inputs {
        for row in D..wires.d_pad {
            builder.enforce_eq(&Lc::from_var(pair.y_c0[row]), &Lc::zero());
            builder.enforce_eq(&Lc::from_var(pair.y_c1[row]), &Lc::zero());
        }
    }
    for row in D..wires.d_pad {
        builder.enforce_eq(&Lc::from_var(wires.combined_c0[row]), &Lc::zero());
        builder.enforce_eq(&Lc::from_var(wires.combined_c1[row]), &Lc::zero());
    }
}

/// Projection-checked padded K-vector combination. Kept as the compact public
/// operation; the NIFS profiler calls the two phases separately.
pub fn enforce_rlc_padded_k_vector_combination_projection_with_quotient_wires(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    rho_evaluations: &PolynomialEvaluationsAtBeta,
    wires: &RlcPaddedKVectorWires,
    quotient_c0: &[Var; PROJECTION_QUOTIENT_LEN],
    quotient_c1: &[Var; PROJECTION_QUOTIENT_LEN],
) -> Result<(), Error> {
    enforce_rlc_padded_k_projection_identities_with_quotient_wires(
        builder,
        powers,
        rho_evaluations,
        wires,
        quotient_c0,
        quotient_c1,
    )?;
    enforce_rlc_padded_k_padding_glue(builder, wires)
}

/// Emit the two base-field-limb projection identities.
pub fn enforce_rlc_padded_k_projection_identities_with_quotient_wires(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    rho_evaluations: &PolynomialEvaluationsAtBeta,
    wires: &RlcPaddedKVectorWires,
    quotient_c0: &[Var; PROJECTION_QUOTIENT_LEN],
    quotient_c1: &[Var; PROJECTION_QUOTIENT_LEN],
) -> Result<(), Error> {
    enforce_rlc_padded_k_projection_identities_with_quotient_wires_and_stages(
        builder,
        powers,
        rho_evaluations,
        wires,
        quotient_c0,
        quotient_c1,
        None,
    )
}

/// Padded K-vector identities with one diagnostic phase-label set per limb.
pub fn enforce_rlc_padded_k_projection_identities_with_quotient_wires_and_stages(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    rho_evaluations: &PolynomialEvaluationsAtBeta,
    wires: &RlcPaddedKVectorWires,
    quotient_c0: &[Var; PROJECTION_QUOTIENT_LEN],
    quotient_c1: &[Var; PROJECTION_QUOTIENT_LEN],
    stages: Option<[ProjectionIdentityStageLabels; 2]>,
) -> Result<(), Error> {
    validate_rlc_padded_k_projection_shape(wires)?;
    let (c0_stages, c1_stages) = match stages {
        Some([c0, c1]) => (Some(c0), Some(c1)),
        None => (None, None),
    };
    let inputs_c0: Vec<[Var; D]> = wires
        .inputs
        .iter()
        .map(|pair| core::array::from_fn(|index| pair.y_c0[index]))
        .collect();
    let inputs_c1: Vec<[Var; D]> = wires
        .inputs
        .iter()
        .map(|pair| core::array::from_fn(|index| pair.y_c1[index]))
        .collect();
    let pairs_c0: Vec<(&[Var; D], &[Var; D])> = wires
        .inputs
        .iter()
        .zip(inputs_c0.iter())
        .map(|(pair, input)| (&pair.rho_coeffs, input))
        .collect();
    let pairs_c1: Vec<(&[Var; D], &[Var; D])> = wires
        .inputs
        .iter()
        .zip(inputs_c1.iter())
        .map(|(pair, input)| (&pair.rho_coeffs, input))
        .collect();
    let output_c0 = core::array::from_fn(|index| wires.combined_c0[index]);
    let output_c1 = core::array::from_fn(|index| wires.combined_c1[index]);
    enforce_ring_action_projection_batch_with_rho_evaluations_and_stages(
        builder,
        powers,
        rho_evaluations,
        &pairs_c0,
        &output_c0,
        quotient_c0,
        c0_stages,
    );
    enforce_ring_action_projection_batch_with_rho_evaluations_and_stages(
        builder,
        powers,
        rho_evaluations,
        &pairs_c1,
        &output_c1,
        quotient_c1,
        c1_stages,
    );
    Ok(())
}

/// Pin the padded tail of every input and output K-vector to zero.
pub fn enforce_rlc_padded_k_padding_glue(
    builder: &mut R1csBuilder,
    wires: &RlcPaddedKVectorWires,
) -> Result<(), Error> {
    validate_rlc_padded_k_projection_shape(wires)?;
    for pair in &wires.inputs {
        for lane in D..wires.d_pad {
            builder.enforce_eq(&Lc::from_var(pair.y_c0[lane]), &Lc::zero());
            builder.enforce_eq(&Lc::from_var(pair.y_c1[lane]), &Lc::zero());
        }
    }
    for lane in D..wires.d_pad {
        builder.enforce_eq(&Lc::from_var(wires.combined_c0[lane]), &Lc::zero());
        builder.enforce_eq(&Lc::from_var(wires.combined_c1[lane]), &Lc::zero());
    }
    Ok(())
}

fn validate_rlc_padded_k_projection_shape(wires: &RlcPaddedKVectorWires) -> Result<(), Error> {
    if wires.inputs.is_empty() {
        return Err(Error::Empty);
    }
    if wires.d_pad < D || wires.combined_c0.len() != wires.d_pad || wires.combined_c1.len() != wires.d_pad {
        return Err(Error::ShapeMismatch {
            what: "combined padded projection shape",
            expected: format!("two limbs of length d_pad >= {D}"),
            got: format!(
                "d_pad={}, c0={}, c1={}",
                wires.d_pad,
                wires.combined_c0.len(),
                wires.combined_c1.len()
            ),
        });
    }
    for (idx, pair) in wires.inputs.iter().enumerate() {
        if pair.y_c0.len() != wires.d_pad || pair.y_c1.len() != wires.d_pad {
            return Err(Error::ShapeMismatch {
                what: "input padded projection shape",
                expected: format!("two limbs of length {}", wires.d_pad),
                got: format!("c0={}, c1={} at idx {idx}", pair.y_c0.len(), pair.y_c1.len()),
            });
        }
    }
    Ok(())
}
