//! Independent NIFS binding and PiRLC projection reference.
//!
//! This file intentionally repeats the selected protocol tags, field order,
//! and schoolbook quotient arithmetic. PaperExact must not call the optimized
//! digest or projection scheduler, because that would hide shared protocol
//! drift from the complete NIFS crosscheck.

use neo_ajtai::Commitment;
use neo_ccs::LaneCommitments;
use neo_math::ring::PHI_MID_DEGREE;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::common::RotRho;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::ring_action::PROJECTION_QUOTIENT_LEN;
use crate::engine::transcript::Transcript;
use crate::paper::reductions::accumulator_sis_circuit::{accumulator_digest, SisAccumulatorConfig};
use crate::paper::relations::{CeClaim, RlcMixer};

use super::pi_rlc::{Error, ProjectionSchedule};

const PROJECTION_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC4; 32],
    kappa: 2,
    domain: 0x5049_524C_435F_5052,
};
const NEBULA_LEAF_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC5; 32],
    kappa: 2,
    domain: 0x4E42_4C41_5F4C_4546,
};

#[derive(Clone)]
struct LaneProjection {
    out: [F; D],
    q: [F; PROJECTION_QUOTIENT_LEN],
}

pub(crate) fn mixed_adv(
    mix: RlcMixer,
    rhos: &[RotRho],
    inputs: &[CeClaim],
) -> Result<Option<LaneCommitments<Commitment>>, Error> {
    let present = inputs.iter().filter(|claim| claim.adv.is_some()).count();
    if present == 0 {
        return Ok(None);
    }
    if present != inputs.len() {
        return Err(Error::AdvPresence {
            present,
            total: inputs.len(),
        });
    }
    let rho_matrices = rhos
        .iter()
        .map(|rho| rho.as_mat().clone())
        .collect::<Vec<_>>();
    let component = |select: fn(&LaneCommitments<Commitment>) -> &Commitment| {
        let commitments = inputs
            .iter()
            .map(|claim| {
                select(
                    claim
                        .adv
                        .as_ref()
                        .expect("all PaperExact adv inputs are present"),
                )
                .clone()
            })
            .collect::<Vec<_>>();
        mix(&rho_matrices, &commitments)
    };
    Ok(Some(LaneCommitments {
        ops: component(|adv| &adv.ops),
        is: component(|adv| &adv.is),
        fs: component(|adv| &adv.fs),
    }))
}

pub(crate) fn projection_schedule(
    tr: &mut Transcript,
    rhos: &[RotRho],
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<ProjectionSchedule, Error> {
    if rhos.len() != inputs.len() || inputs.is_empty() {
        return Err(Error::Shape);
    }
    let mut binding = pack_bytes(b"neo.fold.clean/pi_rlc/projection_binding/v1");
    let rho_coeffs = rhos
        .iter()
        .map(|rho| core::array::from_fn(|row| rho.as_mat()[(row, 0)]))
        .collect::<Vec<[F; D]>>();

    let input_commitments = inputs
        .iter()
        .map(|claim| claim.c.clone())
        .collect::<Vec<_>>();
    let commitment_lanes = checked_commitment_lanes(&rho_coeffs, &input_commitments, &combined.c, None)?;
    append_binding(&mut binding, b"pi_rlc/projection_combined_c", &combined.c.data);
    for lane in &commitment_lanes {
        append_binding(&mut binding, b"pi_rlc/projection_quotients", &lane.q);
    }

    let present = inputs.iter().filter(|claim| claim.adv.is_some()).count();
    let adv_lanes = match (&combined.adv, present) {
        (None, 0) => None,
        (Some(_), 0) | (None, _) => return Err(Error::AdvConsistency),
        (Some(combined_adv), count) if count == inputs.len() => {
            let coordinate = |select: fn(&LaneCommitments<Commitment>) -> &Commitment,
                              target: &Commitment,
                              name: &'static str|
             -> Result<Vec<LaneProjection>, Error> {
                let commitments = inputs
                    .iter()
                    .map(|claim| {
                        select(
                            claim
                                .adv
                                .as_ref()
                                .expect("all PaperExact adv inputs are present"),
                        )
                        .clone()
                    })
                    .collect::<Vec<_>>();
                checked_commitment_lanes(&rho_coeffs, &commitments, target, Some(name))
            };
            let ops = coordinate(|adv| &adv.ops, &combined_adv.ops, "ops")?;
            let is = coordinate(|adv| &adv.is, &combined_adv.is, "is")?;
            let fs = coordinate(|adv| &adv.fs, &combined_adv.fs, "fs")?;

            for leaf in nebula_leaves(combined_adv) {
                append_binding(&mut binding, b"pi_rlc/projection_combined_adv", &leaf);
            }
            for lane in ops.iter().chain(&is).chain(&fs) {
                append_binding(&mut binding, b"pi_rlc/projection_adv_quotients", &lane.q);
            }
            Some(LaneCommitments { ops, is, fs })
        }
        (Some(_), count) => {
            return Err(Error::AdvPresence {
                present: count,
                total: inputs.len(),
            });
        }
    };

    let active_x_columns = crate::paper::relations::superneo_public_x_cols(combined.m_in);
    if combined.m_in % D != 0
        || combined.X.rows() != D
        || combined.X.cols() != active_x_columns
        || inputs
            .iter()
            .any(|claim| claim.m_in != combined.m_in || claim.X.rows() != D || claim.X.cols() != active_x_columns)
    {
        return Err(Error::Shape);
    }
    let mut x_lanes = Vec::with_capacity(active_x_columns);
    for column in 0..active_x_columns {
        let values = inputs
            .iter()
            .map(|claim| core::array::from_fn(|row| claim.X[(row, column)]))
            .collect::<Vec<[F; D]>>();
        let target = core::array::from_fn(|row| combined.X[(row, column)]);
        let lane = checked_auxiliary(&rho_coeffs, &values, target, "X", column)?;
        append_binding(&mut binding, b"pi_rlc/projection_combined_x", &target);
        append_binding(&mut binding, b"pi_rlc/projection_x_quotients", &lane.q);
        x_lanes.push(lane);
    }

    if inputs
        .iter()
        .any(|claim| claim.eval_k.len() != combined.eval_k.len() || claim.eval_a.len() != combined.eval_a.len())
    {
        return Err(Error::Shape);
    }
    let mut evaluation_lanes = Vec::with_capacity(combined.eval_a.len() + 1);
    let eval_k_values = inputs
        .iter()
        .map(|claim| claim.eval_k.as_slice())
        .collect::<Vec<_>>();
    let eval_k_lanes = checked_k_vector(&rho_coeffs, &eval_k_values, &combined.eval_k, "Eval_K", 0)?;
    for lane in &eval_k_lanes {
        append_binding(&mut binding, b"pi_rlc/projection_combined_evaluation", &lane.out);
        append_binding(&mut binding, b"pi_rlc/projection_evaluation_quotients", &lane.q);
    }
    evaluation_lanes.push(eval_k_lanes);
    for matrix in 0..combined.eval_a.len() {
        let values = inputs
            .iter()
            .map(|claim| claim.eval_a[matrix].as_slice())
            .collect::<Vec<_>>();
        let lanes = checked_k_vector(
            &rho_coeffs,
            &values,
            &combined.eval_a[matrix],
            "Eval_A",
            2 * (matrix + 1),
        )?;
        for lane in &lanes {
            append_binding(&mut binding, b"pi_rlc/projection_combined_evaluation", &lane.out);
            append_binding(&mut binding, b"pi_rlc/projection_evaluation_quotients", &lane.q);
        }
        evaluation_lanes.push(lanes);
    }

    let digest = accumulator_digest(PROJECTION_CONFIG, &binding)?;
    tr.append_fields(b"pi_rlc/projection_binding_digest", &digest);
    let beta = tr.challenge_fields(b"pi_rlc/projection_beta", 2);

    Ok(ProjectionSchedule {
        rhos: rho_coeffs,
        q_lanes: commitment_lanes.into_iter().map(|lane| lane.q).collect(),
        adv_q_lanes: adv_lanes.map(|lanes| LaneCommitments {
            ops: lanes.ops.into_iter().map(|lane| lane.q).collect(),
            is: lanes.is.into_iter().map(|lane| lane.q).collect(),
            fs: lanes.fs.into_iter().map(|lane| lane.q).collect(),
        }),
        x_q_lanes: x_lanes.into_iter().map(|lane| lane.q).collect(),
        evaluation_q_lanes: evaluation_lanes
            .into_iter()
            .map(|lanes| lanes.map(|lane| lane.q))
            .collect(),
        beta: K::from_coeffs([beta[0], beta[1]]),
    })
}

fn nebula_leaves(adv: &LaneCommitments<Commitment>) -> [[F; 4]; 3] {
    [
        nebula_leaf(b"neo.fold.clean/nebula/leaf/ops/v4", &adv.ops),
        nebula_leaf(b"neo.fold.clean/nebula/leaf/mem/v4", &adv.is),
        nebula_leaf(b"neo.fold.clean/nebula/leaf/mem/v4", &adv.fs),
    ]
}

fn nebula_leaf(tag: &[u8], commitment: &Commitment) -> [F; 4] {
    let mut preimage = pack_bytes(tag);
    push_commitment(&mut preimage, commitment);
    accumulator_digest(NEBULA_LEAF_CONFIG, &preimage).expect("PaperExact Nebula leaf binding is nonempty")
}

fn checked_commitment_lanes(
    rhos: &[[F; D]],
    inputs: &[Commitment],
    combined: &Commitment,
    coordinate: Option<&'static str>,
) -> Result<Vec<LaneProjection>, Error> {
    if inputs.is_empty()
        || rhos.len() != inputs.len()
        || combined.d != D
        || combined.data.len() != combined.kappa * D
        || inputs.iter().any(|commitment| {
            commitment.d != D || commitment.kappa != combined.kappa || commitment.data.len() != combined.kappa * D
        })
    {
        return Err(Error::Shape);
    }

    let mut output = Vec::with_capacity(combined.kappa);
    for lane in 0..combined.kappa {
        let pairs = rhos
            .iter()
            .zip(inputs)
            .map(|(rho, commitment)| {
                let mut value = [F::ZERO; D];
                value.copy_from_slice(&commitment.data[lane * D..(lane + 1) * D]);
                (*rho, value)
            })
            .collect::<Vec<_>>();
        let projection = quotient(&pairs);
        if combined.data[lane * D..(lane + 1) * D] != projection.out {
            return match coordinate {
                None => Err(Error::ProjectionMixDrift { lane }),
                Some(coordinate) => Err(Error::AdvProjectionMixDrift { coordinate, lane }),
            };
        }
        output.push(projection);
    }
    Ok(output)
}

fn checked_auxiliary(
    rhos: &[[F; D]],
    inputs: &[[F; D]],
    combined: [F; D],
    client: &'static str,
    identity: usize,
) -> Result<LaneProjection, Error> {
    if rhos.len() != inputs.len() {
        return Err(Error::Shape);
    }
    let pairs = rhos
        .iter()
        .copied()
        .zip(inputs.iter().copied())
        .collect::<Vec<_>>();
    let projection = quotient(&pairs);
    if projection.out != combined {
        return Err(Error::AuxiliaryProjectionMixDrift { client, identity });
    }
    Ok(projection)
}

fn checked_k_vector(
    rhos: &[[F; D]],
    inputs: &[&[K]],
    combined: &[K],
    client: &'static str,
    identity: usize,
) -> Result<[LaneProjection; 2], Error> {
    if combined.len() < D || inputs.iter().any(|input| input.len() < D) {
        return Err(Error::Shape);
    }
    let input_c0 = inputs
        .iter()
        .map(|input| core::array::from_fn(|lane| input[lane].as_coeffs()[0]))
        .collect::<Vec<[F; D]>>();
    let input_c1 = inputs
        .iter()
        .map(|input| core::array::from_fn(|lane| input[lane].as_coeffs()[1]))
        .collect::<Vec<[F; D]>>();
    let combined_c0 = core::array::from_fn(|lane| combined[lane].as_coeffs()[0]);
    let combined_c1 = core::array::from_fn(|lane| combined[lane].as_coeffs()[1]);
    Ok([
        checked_auxiliary(rhos, &input_c0, combined_c0, client, identity)?,
        checked_auxiliary(rhos, &input_c1, combined_c1, client, identity + 1)?,
    ])
}

fn quotient(pairs: &[([F; D], [F; D])]) -> LaneProjection {
    let mut product = [F::ZERO; 2 * D - 1];
    for (rho, value) in pairs {
        for (left, &rho_coefficient) in rho.iter().enumerate() {
            for (right, &value_coefficient) in value.iter().enumerate() {
                product[left + right] += rho_coefficient * value_coefficient;
            }
        }
    }

    let mut q = [F::ZERO; PROJECTION_QUOTIENT_LEN];
    for degree in (D..2 * D - 1).rev() {
        let coefficient = product[degree];
        q[degree - D] = coefficient;
        product[degree] = F::ZERO;
        product[degree - PHI_MID_DEGREE] -= coefficient;
        product[degree - D] -= coefficient;
    }
    let mut out = [F::ZERO; D];
    out.copy_from_slice(&product[..D]);
    LaneProjection { out, q }
}

fn append_binding(preimage: &mut Vec<F>, label: &[u8], fields: &[F]) {
    preimage.extend(pack_bytes(label));
    preimage.push(F::from_u64(fields.len() as u64));
    preimage.extend_from_slice(fields);
}

fn push_commitment(preimage: &mut Vec<F>, commitment: &Commitment) {
    preimage.push(F::from_u64(commitment.d as u64));
    preimage.push(F::from_u64(commitment.kappa as u64));
    preimage.push(F::from_u64(commitment.data.len() as u64));
    preimage.extend_from_slice(&commitment.data);
}

fn pack_bytes(bytes: &[u8]) -> Vec<F> {
    let mut output = Vec::with_capacity(1 + bytes.len().div_ceil(7));
    output.push(F::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(7) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        output.push(F::from_u64(u64::from_le_bytes(limb)));
    }
    output
}
