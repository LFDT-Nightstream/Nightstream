//! Commitment branch of Π_RLC.V.
//!
//! Owns: paper commitment ring-combination witnesses and exact projection leaf
//! equations, plus the mechanically identical Nebula `adv` extension leaves.
//!
//! Does not own: transcript binding, beta sampling, or claim allocation order.
//!
//! Emits constraints: yes; allocation helpers emit none.
//!
//! Authority boundary: projection helpers consume the exact quotient and beta
//! wires bound by NIFS orchestration.
//!
//! | Stage path | Function | Equation | Multiplicity | Emitted rows/formula | Lowered gate | Lean theorem |
//! |---|---|---|---:|---|---|---|
//! | `fold_wires.commitment/adv` | `alloc_rlc_commitment_inputs*` | typed `(rho_i,c_i,parent)` view; `adv` is a Nebula extension, not paper CE | once per claim family | none | none | parameter shape |
//! | full combination | `enforce_rlc_commitment_combination` | `c = sum_i rho_i*c_i` coefficientwise | lanes × inputs | Toom-3 products plus equalities | ring product | `commitmentCombinationWithIntermediates_iff_direct` |
//! | `identities.commitment/adv` | projection helper | aggregate identity at beta | one per lane | polynomial evaluation rows | product-sum | exact-or-bad-root bridge open |

use neo_ajtai::Commitment;
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::Error;
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::ring_action::{
    enforce_polynomial_evaluations_at_beta, enforce_ring_action_projection_batch_with_rho_evaluations_and_stages,
    enforce_ring_mul_toom3, projection_quotient, PolynomialEvaluationsAtBeta, ProjectionIdentityStageLabels,
    PROJECTION_QUOTIENT_LEN,
};
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};

/// Wires for one commitment + the matching rho polynomial coefficients.
///
/// `c_data` is column-major over `kappa` columns of `D` rows. `rho_coeffs`
/// follows the first column of the native rotation matrix `RotRho`.
#[derive(Clone, Debug)]
pub struct RlcPairWires {
    pub rho_coeffs: [Var; D],
    pub c_data: Vec<Var>,
    pub kappa: usize,
}

/// Wires for all input pairs and the expected combined commitment.
#[derive(Clone, Debug)]
pub struct RlcCommitmentWires {
    pub inputs: Vec<RlcPairWires>,
    pub combined_c_data: Vec<Var>,
    pub kappa: usize,
}

fn commitment_data_len(kappa: usize) -> Result<usize, Error> {
    if kappa == 0 {
        return Err(Error::ShapeMismatch {
            what: "commitment rank",
            expected: "a positive kappa".into(),
            got: "kappa=0".into(),
        });
    }
    kappa.checked_mul(D).ok_or_else(|| Error::ShapeMismatch {
        what: "commitment data length",
        expected: "D*kappa to fit in usize".into(),
        got: format!("D={D}, kappa={kappa}"),
    })
}

/// Allocate commitment-combination witnesses without emitting constraints.
pub fn alloc_rlc_commitment_inputs(
    builder: &mut R1csBuilder,
    rhos_first_col: &[[F; D]],
    inputs: &[Commitment],
    combined: &Commitment,
) -> Result<RlcCommitmentWires, Error> {
    if inputs.is_empty() {
        return Err(Error::Empty);
    }
    if rhos_first_col.len() != inputs.len() {
        return Err(Error::PairCountMismatch {
            rhos: rhos_first_col.len(),
            inputs: inputs.len(),
        });
    }
    let kappa = inputs[0].kappa;
    let expected_data_len = commitment_data_len(kappa)?;
    if combined.kappa != kappa || combined.d != D || combined.data.len() != expected_data_len {
        return Err(Error::ShapeMismatch {
            what: "combined commitment shape",
            expected: format!("(d={D}, kappa={kappa}, data={expected_data_len})"),
            got: format!(
                "(d={}, kappa={}, data={})",
                combined.d,
                combined.kappa,
                combined.data.len()
            ),
        });
    }

    let mut input_wires = Vec::with_capacity(inputs.len());
    for (idx, (rho_col, c)) in rhos_first_col.iter().zip(inputs.iter()).enumerate() {
        if c.kappa != kappa || c.d != D || c.data.len() != expected_data_len {
            return Err(Error::ShapeMismatch {
                what: "input commitment shape",
                expected: format!("(d={D}, kappa={kappa}, data={expected_data_len})"),
                got: format!("(d={}, kappa={}, data={}) at idx {idx}", c.d, c.kappa, c.data.len()),
            });
        }
        let mut rho_coeffs = [Var::ONE; D];
        for (slot, &value) in rho_coeffs.iter_mut().zip(rho_col.iter()) {
            *slot = builder.alloc(value);
        }
        input_wires.push(RlcPairWires {
            rho_coeffs,
            c_data: builder.alloc_vec(&c.data),
            kappa,
        });
    }
    Ok(RlcCommitmentWires {
        inputs: input_wires,
        combined_c_data: builder.alloc_vec(&combined.data),
        kappa,
    })
}

/// Allocate commitment witnesses while reusing transcript-derived rho wires.
pub fn alloc_rlc_commitment_inputs_with_rhos(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    inputs: &[Commitment],
    combined: &Commitment,
) -> Result<RlcCommitmentWires, Error> {
    if inputs.is_empty() {
        return Err(Error::Empty);
    }
    if rho_wires.len() != inputs.len() {
        return Err(Error::PairCountMismatch {
            rhos: rho_wires.len(),
            inputs: inputs.len(),
        });
    }
    let kappa = inputs[0].kappa;
    let expected_data_len = commitment_data_len(kappa)?;
    if combined.kappa != kappa || combined.d != D || combined.data.len() != expected_data_len {
        return Err(Error::ShapeMismatch {
            what: "combined commitment shape",
            expected: format!("(d={D}, kappa={kappa}, data={expected_data_len})"),
            got: format!(
                "(d={}, kappa={}, data={})",
                combined.d,
                combined.kappa,
                combined.data.len()
            ),
        });
    }

    let mut input_wires = Vec::with_capacity(inputs.len());
    for (idx, (rho, c)) in rho_wires.iter().zip(inputs.iter()).enumerate() {
        if c.kappa != kappa || c.d != D || c.data.len() != expected_data_len {
            return Err(Error::ShapeMismatch {
                what: "input commitment shape",
                expected: format!("(d={D}, kappa={kappa}, data={expected_data_len})"),
                got: format!("(d={}, kappa={}, data={}) at idx {idx}", c.d, c.kappa, c.data.len()),
            });
        }
        input_wires.push(RlcPairWires {
            rho_coeffs: *rho,
            c_data: builder.alloc_vec(&c.data),
            kappa,
        });
    }
    Ok(RlcCommitmentWires {
        inputs: input_wires,
        combined_c_data: builder.alloc_vec(&combined.data),
        kappa,
    })
}

/// Enforce `combined.c = sum_i rho_i * c_i` lane-by-lane.
pub fn enforce_rlc_commitment_combination(builder: &mut R1csBuilder, wires: &RlcCommitmentWires) {
    for lane in 0..wires.kappa {
        let mut per_pair_out: Vec<[Var; D]> = Vec::with_capacity(wires.inputs.len());
        for pair in &wires.inputs {
            let mut c_lane = [Var::ONE; D];
            for (slot, src) in c_lane
                .iter_mut()
                .zip(pair.c_data[lane * D..(lane + 1) * D].iter())
            {
                *slot = *src;
            }
            per_pair_out.push(enforce_ring_mul_toom3(builder, &pair.rho_coeffs, &c_lane));
        }

        for coefficient in 0..D {
            let mut combination = Lc::zero();
            for pair_out in &per_pair_out {
                combination.add_term(pair_out[coefficient], F::ONE);
            }
            let target = wires.combined_c_data[lane * D + coefficient];
            builder.enforce_eq(&Lc::from_var(target), &combination);
        }
    }
}

/// One commitment lane of a projection-checked mix and its division quotient.
#[derive(Clone, Debug)]
pub struct RlcLaneProjection {
    pub out: [F; D],
    pub q: [F; PROJECTION_QUOTIENT_LEN],
}

/// Recompute each commitment lane and its quotient before the transcript
/// squeezes the projection challenge.
pub fn rlc_projection_quotients(
    rhos_first_col: &[[F; D]],
    inputs: &[Commitment],
) -> Result<Vec<RlcLaneProjection>, Error> {
    if inputs.is_empty() {
        return Err(Error::Empty);
    }
    if rhos_first_col.len() != inputs.len() {
        return Err(Error::PairCountMismatch {
            rhos: rhos_first_col.len(),
            inputs: inputs.len(),
        });
    }
    let kappa = inputs[0].kappa;
    let expected_data_len = commitment_data_len(kappa)?;
    for (idx, commitment) in inputs.iter().enumerate() {
        if commitment.kappa != kappa || commitment.d != D || commitment.data.len() != expected_data_len {
            return Err(Error::ShapeMismatch {
                what: "projection input commitment shape",
                expected: format!("(d={D}, kappa={kappa}, data={expected_data_len})"),
                got: format!(
                    "(d={}, kappa={}, data={}) at idx {idx}",
                    commitment.d,
                    commitment.kappa,
                    commitment.data.len()
                ),
            });
        }
    }

    let mut per_lane = Vec::with_capacity(kappa);
    for lane in 0..kappa {
        let pairs: Vec<([F; D], [F; D])> = rhos_first_col
            .iter()
            .zip(inputs.iter())
            .map(|(rho, commitment)| {
                let mut lane_coefficients = [F::ZERO; D];
                lane_coefficients.copy_from_slice(&commitment.data[lane * D..(lane + 1) * D]);
                (*rho, lane_coefficients)
            })
            .collect();
        let (out, q) = projection_quotient(&pairs);
        per_lane.push(RlcLaneProjection { out, q });
    }
    Ok(per_lane)
}

/// Allocate quotient advice from already allocated rho and operand wires.
/// The caller must absorb these exact wires before squeezing beta.
pub fn alloc_rlc_projection_quotient_advice(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    input_wires: &[[Var; D]],
) -> Result<[Var; PROJECTION_QUOTIENT_LEN], Error> {
    if rho_wires.is_empty() {
        return Err(Error::Empty);
    }
    if rho_wires.len() != input_wires.len() {
        return Err(Error::PairCountMismatch {
            rhos: rho_wires.len(),
            inputs: input_wires.len(),
        });
    }
    let pairs: Vec<([F; D], [F; D])> = rho_wires
        .iter()
        .zip(input_wires.iter())
        .map(|(rho, input)| {
            (
                core::array::from_fn(|index| builder.witness()[rho[index].col()]),
                core::array::from_fn(|index| builder.witness()[input[index].col()]),
            )
        })
        .collect();
    let (_, quotient) = projection_quotient(&pairs);
    Ok(quotient.map(|value| builder.alloc(value)))
}

/// Projection-checked commitment combination using native quotient advice.
/// The transcript must bind the returned quotient wires before beta is sampled.
pub fn enforce_rlc_commitment_combination_projection(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    wires: &RlcCommitmentWires,
    quotients: &[[F; PROJECTION_QUOTIENT_LEN]],
) -> Result<Vec<[Var; PROJECTION_QUOTIENT_LEN]>, Error> {
    let kappa = wires.kappa;
    if quotients.len() != kappa {
        return Err(Error::ShapeMismatch {
            what: "projection quotient count",
            expected: format!("kappa = {kappa}"),
            got: format!("{}", quotients.len()),
        });
    }
    let quotient_wires = quotients
        .iter()
        .map(|quotient| quotient.map(|value| builder.alloc(value)))
        .collect::<Vec<_>>();
    let rho_polynomials = wires
        .inputs
        .iter()
        .map(|pair| pair.rho_coeffs)
        .collect::<Vec<_>>();
    let rho_evaluations = enforce_polynomial_evaluations_at_beta(builder, &rho_polynomials, powers);
    enforce_rlc_commitment_combination_projection_with_quotient_wires(
        builder,
        powers,
        &rho_evaluations,
        wires,
        &quotient_wires,
    )?;
    Ok(quotient_wires)
}

/// Enforce a projection-checked commitment combination using the exact
/// quotient wires already bound into the transcript.
pub fn enforce_rlc_commitment_combination_projection_with_quotient_wires(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    rho_evaluations: &PolynomialEvaluationsAtBeta,
    wires: &RlcCommitmentWires,
    quotient_wires: &[[Var; PROJECTION_QUOTIENT_LEN]],
) -> Result<(), Error> {
    enforce_rlc_commitment_combination_projection_with_quotient_wires_and_stages(
        builder,
        powers,
        rho_evaluations,
        wires,
        quotient_wires,
        None,
    )
}

/// Projection-checked commitment combination with diagnostic phase labels.
pub fn enforce_rlc_commitment_combination_projection_with_quotient_wires_and_stages(
    builder: &mut R1csBuilder,
    powers: &[KVar],
    rho_evaluations: &PolynomialEvaluationsAtBeta,
    wires: &RlcCommitmentWires,
    quotient_wires: &[[Var; PROJECTION_QUOTIENT_LEN]],
    stages: Option<ProjectionIdentityStageLabels>,
) -> Result<(), Error> {
    let kappa = wires.kappa;
    let expected_data_len = commitment_data_len(kappa)?;
    if quotient_wires.len() != kappa {
        return Err(Error::ShapeMismatch {
            what: "projection quotient wire count",
            expected: format!("kappa = {kappa}"),
            got: format!("{}", quotient_wires.len()),
        });
    }
    if wires.inputs.is_empty() {
        return Err(Error::Empty);
    }
    if wires.combined_c_data.len() != expected_data_len {
        return Err(Error::ShapeMismatch {
            what: "projection combined commitment wires",
            expected: format!("{expected_data_len} coefficients"),
            got: format!("{}", wires.combined_c_data.len()),
        });
    }
    for (idx, pair) in wires.inputs.iter().enumerate() {
        if pair.kappa != kappa || pair.c_data.len() != expected_data_len {
            return Err(Error::ShapeMismatch {
                what: "projection input commitment wires",
                expected: format!("(kappa={kappa}, data={expected_data_len})"),
                got: format!("(kappa={}, data={}) at idx {idx}", pair.kappa, pair.c_data.len()),
            });
        }
    }

    for lane in 0..kappa {
        let pair_arrays: Vec<([Var; D], [Var; D])> = wires
            .inputs
            .iter()
            .map(|pair| {
                let mut commitment_lane = [Var::ONE; D];
                for (slot, source) in commitment_lane
                    .iter_mut()
                    .zip(pair.c_data[lane * D..(lane + 1) * D].iter())
                {
                    *slot = *source;
                }
                (pair.rho_coeffs, commitment_lane)
            })
            .collect();
        let pair_refs = pair_arrays
            .iter()
            .map(|(rho, commitment)| (rho, commitment))
            .collect::<Vec<_>>();

        let mut output_lane = [Var::ONE; D];
        for (slot, source) in output_lane
            .iter_mut()
            .zip(wires.combined_c_data[lane * D..(lane + 1) * D].iter())
        {
            *slot = *source;
        }

        enforce_ring_action_projection_batch_with_rho_evaluations_and_stages(
            builder,
            powers,
            rho_evaluations,
            &pair_refs,
            &output_lane,
            &quotient_wires[lane],
            stages,
        );
    }
    Ok(())
}
