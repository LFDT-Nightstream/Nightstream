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
use crate::paper::relations::{CcsClaim, CeClaim, RlcMixer};

use super::pi_rlc::{Error, ProjectionSchedule};

const CCS_CLAIM_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC1; 32],
    kappa: 2,
    domain: 0x4343_535F_434C_4149,
};
const OUTPUTS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC3; 32],
    kappa: 2,
    domain: 0x5049_4343_535F_4F55,
};
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
const ACCUMULATOR_CLAIM_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC7; 32],
    kappa: 2,
    domain: 0x4143_4345_5F43_4C4D,
};

const NEBULA_ADV_PRESENT_MARKER: u64 = 0x4e42_4c41;
const PROTOCOL_BINDING_MAX_FIELDS: usize = 50_371 * D / 41;

#[derive(Clone)]
struct LaneProjection {
    out: [F; D],
    q: [F; PROJECTION_QUOTIENT_LEN],
}

pub(crate) fn pi_ccs_instance_digest(
    fresh: &[CcsClaim],
    running_count: usize,
    running_parent: Option<&CeClaim>,
) -> [F; 4] {
    let mut preimage = pack_bytes(b"neo.fold.clean/pi_ccs_instance_digest/parent_authority/v1");
    preimage.push(F::from_u64(fresh.len() as u64));
    for claim in fresh {
        preimage.extend_from_slice(&ccs_claim_digest(claim));
    }
    preimage.push(F::from_u64(running_count as u64));
    match (running_count, running_parent) {
        (0, None) => preimage.push(F::ZERO),
        (_, Some(parent)) => {
            preimage.push(F::ONE);
            preimage.extend_from_slice(&accumulator_claim_digest(parent));
        }
        (_, None) => preimage.push(F::from_u64(u64::MAX)),
    }
    poseidon(&preimage)
}

pub(crate) fn accumulator_handle(claims: &[CeClaim], parent: Option<&CeClaim>) -> [F; 4] {
    let shape_is_valid = (claims.is_empty() && parent.is_none()) || (!claims.is_empty() && parent.is_some());
    if claims.is_empty() && parent.is_none() {
        return exact_child_accumulator_handle(claims);
    }
    if let Some(parent) = parent {
        if let Some(digest) = strict_binary_accumulator_family_digest(claims, parent) {
            return digest;
        }
    }
    if shape_is_valid {
        return exact_child_accumulator_handle(claims);
    }

    let mut preimage = pack_bytes(b"neo.fold.clean/accumulator/malformed/v3");
    preimage.push(F::from_u64(claims.len() as u64));
    preimage.extend_from_slice(&exact_child_accumulator_handle(claims));
    match parent {
        Some(parent) => {
            preimage.push(F::ONE);
            preimage.extend_from_slice(&accumulator_claim_digest(parent));
        }
        None => preimage.push(F::ZERO),
    }
    poseidon(&preimage)
}

fn exact_child_accumulator_handle(claims: &[CeClaim]) -> [F; 4] {
    let mut preimage = pack_bytes(b"neo.fold.clean/accumulator/children/v4");
    preimage.push(F::from_u64(claims.len() as u64));
    for claim in claims {
        preimage.extend_from_slice(&accumulator_claim_digest(claim));
    }
    poseidon(&preimage)
}

fn strict_binary_accumulator_family_digest(claims: &[CeClaim], parent: &CeClaim) -> Option<[F; 4]> {
    let preimage = strict_binary_accumulator_family_preimage(claims, parent)?;
    let chunk_digests = preimage
        .chunks(PROTOCOL_BINDING_MAX_FIELDS)
        .map(|chunk| {
            accumulator_digest(ACCUMULATOR_CLAIM_CONFIG, chunk).expect("PaperExact strict-family SIS chunk is nonempty")
        })
        .collect::<Vec<_>>();

    let mut aggregate = pack_bytes(b"neo.fold.clean/accumulator/strict_binary_family/aggregate/v1");
    aggregate.push(F::from_u64(preimage.len() as u64));
    aggregate.push(F::from_u64(chunk_digests.len() as u64));
    for digest in chunk_digests {
        aggregate.extend_from_slice(&digest);
    }
    Some(poseidon(&aggregate))
}

fn strict_binary_accumulator_family_preimage(claims: &[CeClaim], parent: &CeClaim) -> Option<Vec<F>> {
    let first = claims.first()?;
    let active_x_columns = crate::paper::relations::superneo_public_x_cols(first.m_in);
    if first.m_in % D != 0
        || first.c.d != D
        || first.c.kappa == 0
        || first.c.data.len() != first.c.d * first.c.kappa
        || first.X.rows() != D
        || first.X.cols() != active_x_columns
        || first.y_ring.is_empty()
        || first.ct.len() != first.y_ring.len()
        || first
            .y_ring
            .iter()
            .any(|row| row.len() != D.next_power_of_two())
        || !claim_has_canonical_derived_fields(first)
        || !adv_has_shape(&first.adv, first.c.d, first.c.kappa)
    {
        return None;
    }

    let expected_child_x = neo_reductions::common::split_b_matrix_k(&parent.X, claims.len(), 2).ok()?;
    if parent.m_in != first.m_in
        || parent.X.rows() != first.X.rows()
        || parent.X.cols() != first.X.cols()
        || expected_child_x.len() != claims.len()
    {
        return None;
    }

    let has_adv = first.adv.is_some();
    for (claim, expected_x) in claims.iter().zip(&expected_child_x) {
        if claim.c.d != first.c.d
            || claim.c.kappa != first.c.kappa
            || claim.c.data.len() != first.c.data.len()
            || claim.X.rows() != first.X.rows()
            || claim.X.cols() != first.X.cols()
            || claim.X != *expected_x
            || claim.r != first.r
            || claim.y_ring.len() != first.y_ring.len()
            || claim
                .y_ring
                .iter()
                .zip(&first.y_ring)
                .any(|(row, first_row)| row.len() != first_row.len())
            || claim.ct.len() != claim.y_ring.len()
            || claim.m_in != first.m_in
            || claim.fold_digest != first.fold_digest
            || claim.adv.is_some() != has_adv
            || !claim_has_canonical_derived_fields(claim)
            || !adv_has_shape(&claim.adv, first.c.d, first.c.kappa)
        {
            return None;
        }
    }

    let mut preimage = pack_bytes(b"neo.fold.clean/accumulator/strict_binary_family/v5");
    preimage.push(F::from_u64(claims.len() as u64));
    preimage.push(F::from_u64(first.c.d as u64));
    preimage.push(F::from_u64(first.c.kappa as u64));
    preimage.push(F::from_u64(first.c.data.len() as u64));
    preimage.push(F::from_u64(first.X.rows() as u64));
    preimage.push(F::from_u64(first.X.cols() as u64));
    preimage.push(F::from_u64(active_x_columns as u64));
    for row in 0..parent.X.rows() {
        for column in 0..active_x_columns {
            preimage.push(parent.X[(row, column)]);
        }
    }
    push_k_slice(&mut preimage, &first.r);
    preimage.push(F::from_u64(first.y_ring.len() as u64));
    preimage.push(F::from_u64(D as u64));
    preimage.push(F::from_u64(first.m_in as u64));
    for chunk in first.fold_digest.chunks_exact(8) {
        preimage.push(F::from_u64(u64::from_le_bytes(
            chunk.try_into().expect("eight-byte digest limb"),
        )));
    }
    preimage.push(if has_adv { F::ONE } else { F::ZERO });

    for (index, claim) in claims.iter().enumerate() {
        preimage.push(F::from_u64(index as u64));
        preimage.extend_from_slice(&claim.c.data);
        for row in &claim.y_ring {
            for value in row.iter().take(D) {
                preimage.extend_from_slice(&value.as_coeffs());
            }
        }
        if let Some(adv) = &claim.adv {
            preimage.extend_from_slice(&adv.ops.data);
            preimage.extend_from_slice(&adv.is.data);
            preimage.extend_from_slice(&adv.fs.data);
        }
    }
    Some(preimage)
}

fn claim_has_canonical_derived_fields(claim: &CeClaim) -> bool {
    claim
        .ct
        .iter()
        .zip(&claim.y_ring)
        .all(|(ct, row)| row.first() == Some(ct))
        && claim
            .y_ring
            .iter()
            .all(|row| row.iter().skip(D).all(|value| *value == K::ZERO))
}

fn adv_has_shape(adv: &Option<LaneCommitments<Commitment>>, d: usize, kappa: usize) -> bool {
    let Some(adv) = adv else {
        return true;
    };
    [&adv.ops, &adv.is, &adv.fs]
        .into_iter()
        .all(|commitment| commitment.d == d && commitment.kappa == kappa && commitment.data.len() == d * kappa)
}

pub(crate) fn pi_ccs_outputs_digest(claims: &[CeClaim]) -> [F; 4] {
    let mut preimage = pack_bytes(b"neo.fold.clean/pi_ccs_outputs_digest/v3");
    preimage.push(F::from_u64(claims.len() as u64));
    for claim in claims {
        preimage.extend(pack_bytes(b"neo.fold.clean/pi_ccs_output_message_digest/v3"));
        preimage.push(F::from_u64(claim.y_ring.len() as u64));
        for row in &claim.y_ring {
            let active = &row[..row.len().min(D)];
            push_k_slice(&mut preimage, active);
        }
    }
    accumulator_digest(OUTPUTS_CONFIG, &preimage).expect("PaperExact PiCCS output binding is nonempty")
}

pub(crate) fn bind_pi_rlc_inputs(tr: &mut Transcript, claims: &[CeClaim]) -> Result<(), Error> {
    if claims.is_empty() {
        return Err(Error::Shape);
    }
    tr.append_fields(b"pi_rlc/input_claims_digest", &pi_ccs_outputs_digest(claims));
    Ok(())
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
        .any(|claim| claim.y_ring.len() != combined.y_ring.len())
    {
        return Err(Error::Shape);
    }
    let mut y_ring_lanes = Vec::with_capacity(combined.y_ring.len());
    for row in 0..combined.y_ring.len() {
        let values = inputs
            .iter()
            .map(|claim| claim.y_ring[row].as_slice())
            .collect::<Vec<_>>();
        let lanes = checked_k_vector(&rho_coeffs, &values, &combined.y_ring[row], "y_ring", 2 * row)?;
        for lane in &lanes {
            append_binding(&mut binding, b"pi_rlc/projection_combined_y_ring", &lane.out);
            append_binding(&mut binding, b"pi_rlc/projection_y_ring_quotients", &lane.q);
        }
        y_ring_lanes.push(lanes);
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
        y_ring_q_lanes: y_ring_lanes
            .into_iter()
            .map(|lanes| lanes.map(|lane| lane.q))
            .collect(),
        beta: K::from_coeffs([beta[0], beta[1]]),
    })
}

fn ccs_claim_digest(claim: &CcsClaim) -> [F; 4] {
    let mut preimage = pack_bytes(b"neo.fold.clean/ccs_claim_digest/v1");
    push_commitment(&mut preimage, &claim.c);
    preimage.push(F::from_u64(claim.x.len() as u64));
    preimage.extend_from_slice(&claim.x);
    preimage.push(F::from_u64(claim.m_in as u64));
    append_adv(&mut preimage, &claim.adv);
    accumulator_digest(CCS_CLAIM_CONFIG, &preimage).expect("PaperExact CCS claim binding is nonempty")
}

fn accumulator_claim_digest(claim: &CeClaim) -> [F; 4] {
    let mut preimage = pack_bytes(b"neo.fold.clean/accumulator_ce_claim_digest/v3");
    push_commitment(&mut preimage, &claim.c);

    let active_x_columns = crate::paper::relations::superneo_public_x_cols(claim.m_in);
    preimage.push(F::from_u64(claim.X.rows() as u64));
    preimage.push(F::from_u64(claim.X.cols() as u64));
    preimage.push(F::from_u64(active_x_columns as u64));
    for row in 0..claim.X.rows() {
        for column in 0..claim.X.cols() {
            preimage.push(claim.X[(row, column)]);
        }
    }

    push_k_slice(&mut preimage, &claim.r);
    preimage.push(F::from_u64(claim.y_ring.len() as u64));
    for row in &claim.y_ring {
        push_k_slice(&mut preimage, row);
    }
    push_k_slice(&mut preimage, &claim.ct);
    preimage.push(F::from_u64(claim.m_in as u64));
    for chunk in claim.fold_digest.chunks_exact(8) {
        preimage.push(F::from_u64(u64::from_le_bytes(
            chunk.try_into().expect("eight-byte digest limb"),
        )));
    }
    append_adv(&mut preimage, &claim.adv);
    accumulator_digest(ACCUMULATOR_CLAIM_CONFIG, &preimage).expect("PaperExact accumulator claim binding is nonempty")
}

fn append_adv(preimage: &mut Vec<F>, adv: &Option<LaneCommitments<Commitment>>) {
    if let Some(adv) = adv {
        preimage.push(F::from_u64(NEBULA_ADV_PRESENT_MARKER));
        for leaf in nebula_leaves(adv) {
            preimage.extend_from_slice(&leaf);
        }
    }
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

fn push_k_slice(preimage: &mut Vec<F>, values: &[K]) {
    preimage.push(F::from_u64(values.len() as u64));
    for value in values {
        preimage.extend_from_slice(&value.as_coeffs());
    }
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

fn poseidon(fields: &[F]) -> [F; 4] {
    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(fields)
}
