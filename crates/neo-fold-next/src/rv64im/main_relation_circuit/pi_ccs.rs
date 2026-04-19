//! Owns transcript-binding and challenge-sampling gadgets for Π_CCS replay.
//!
//! This module mirrors the native `neo_reductions::engines::utils` transcript
//! flow under RV64IM-owned circuit code. It does not own terminal identities,
//! RLC/DEC algebra, or Ajtai witness checks.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_ccs::SparsePoly;
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::{
    pi_ccs_header_bundle_digest_fields_from_parts, Dims, PI_CCS_HEADER_BUNDLE_RAW_TAG, PI_CCS_INSTANCE_DIGEST_RAW_TAG,
    PI_CCS_ME_COUNT_RAW_TAG, PI_CCS_ME_DIGEST_RAW_TAG, PI_CCS_ME_INPUTS_RAW_DOMAIN_TAG,
};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use spartan2::provider::goldi::F as SpartanF;

use super::claim::{
    me_digest_poseidon, me_digest_poseidon_values, me_digest_poseidon_values_from_native_claim,
    me_digest_poseidon_with_native_claim, CeClaimVar,
};
use super::k_field::KNumVar;
use super::transcript::Poseidon2TranscriptCircuit;

#[derive(Clone)]
pub struct PiCcsChallengeVars {
    pub alpha: Vec<KNumVar>,
    pub beta_a: Vec<KNumVar>,
    pub beta_r: Vec<KNumVar>,
    pub beta_m: Vec<KNumVar>,
    pub gamma: KNumVar,
}

pub fn bind_header_and_instance_digest<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptCircuit,
    params: &NeoParams,
    n: usize,
    m: usize,
    t: usize,
    poly: &SparsePoly<F>,
    dims: Dims,
    mat_digest: &[Goldilocks],
    instance_digest_fields: &[SpartanF; 4],
) -> Result<(), SynthesisError> {
    let header_bundle = pi_ccs_header_bundle_digest_fields_from_parts(params, n, m, t, poly, dims, mat_digest)
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let header_bundle_fields = header_bundle.map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    tr.append_const_fields_raw(
        cs.namespace(|| "header_bundle"),
        &[
            SpartanF::from_canonical_u64(PI_CCS_HEADER_BUNDLE_RAW_TAG),
            header_bundle_fields[0],
            header_bundle_fields[1],
            header_bundle_fields[2],
            header_bundle_fields[3],
        ],
    )?;
    let instance_digest_vars = instance_digest_fields
        .iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc(cs.namespace(|| format!("instance_digest_{idx}")), || Ok(*value)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut field_terms = Vec::with_capacity(1 + instance_digest_vars.len());
    let mut field_constants = Vec::with_capacity(1 + instance_digest_vars.len());
    let mut field_values = Vec::with_capacity(1 + instance_digest_vars.len());
    field_terms.push(Vec::new());
    field_constants.push(SpartanF::from_canonical_u64(PI_CCS_INSTANCE_DIGEST_RAW_TAG));
    field_values.push(SpartanF::from_canonical_u64(PI_CCS_INSTANCE_DIGEST_RAW_TAG));
    for (num, value) in instance_digest_vars
        .iter()
        .zip(instance_digest_fields.iter())
    {
        field_terms.push(vec![(num.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_values.push(*value);
    }
    tr.append_field_linear_combinations_raw(
        cs.namespace(|| "instance_digest"),
        &field_terms,
        &field_constants,
        &field_values,
    )?;
    Ok(())
}

pub fn bind_me_input_digests<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptCircuit,
    me_input_digests: &[[AllocatedNum<SpartanF>; 4]],
    me_input_digest_values: &[[SpartanF; 4]],
) -> Result<(), SynthesisError> {
    if me_input_digests.len() != me_input_digest_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    tr.append_const_fields_raw(
        cs.namespace(|| "me_inputs_domain"),
        &[SpartanF::from_canonical_u64(PI_CCS_ME_INPUTS_RAW_DOMAIN_TAG)],
    )?;
    tr.append_const_fields_raw(
        cs.namespace(|| "me_count"),
        &[
            SpartanF::from_canonical_u64(PI_CCS_ME_COUNT_RAW_TAG),
            SpartanF::from_canonical_u64(me_input_digests.len() as u64),
        ],
    )?;
    let flattened = me_input_digests
        .iter()
        .flat_map(|digest| digest.iter().cloned())
        .collect::<Vec<_>>();
    let flattened_values = me_input_digest_values
        .iter()
        .flat_map(|digest| digest.iter().copied())
        .collect::<Vec<_>>();
    let mut field_terms = Vec::with_capacity(1 + flattened.len());
    let mut field_constants = Vec::with_capacity(1 + flattened.len());
    let mut field_values = Vec::with_capacity(1 + flattened_values.len());
    field_terms.push(Vec::new());
    field_constants.push(SpartanF::from_canonical_u64(PI_CCS_ME_DIGEST_RAW_TAG));
    field_values.push(SpartanF::from_canonical_u64(PI_CCS_ME_DIGEST_RAW_TAG));
    for (num, value) in flattened.iter().zip(flattened_values.iter()) {
        field_terms.push(vec![(num.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_values.push(*value);
    }
    tr.append_field_linear_combinations_raw(
        cs.namespace(|| "me_digest"),
        &field_terms,
        &field_constants,
        &field_values,
    )?;
    Ok(())
}

pub fn bind_me_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptCircuit,
    me_inputs: &[CeClaimVar],
) -> Result<Vec<[AllocatedNum<SpartanF>; 4]>, SynthesisError> {
    let mut digests = Vec::with_capacity(me_inputs.len());
    let mut digest_values = Vec::with_capacity(me_inputs.len());
    for (idx, claim) in me_inputs.iter().enumerate() {
        digests.push(me_digest_poseidon(
            &mut cs.namespace(|| format!("me_input_digest_{idx}")),
            claim,
            &format!("me_input_digest_{idx}"),
        )?);
        digest_values.push(me_digest_poseidon_values(claim));
    }
    bind_me_input_digests(cs, tr, &digests, &digest_values)?;
    Ok(digests)
}

pub fn bind_me_inputs_with_native_claims<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptCircuit,
    me_inputs: &[CeClaimVar],
    native_claims: &[CeClaim<Commitment, F, K>],
) -> Result<Vec<[AllocatedNum<SpartanF>; 4]>, SynthesisError> {
    if me_inputs.len() != native_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut digests = Vec::with_capacity(me_inputs.len());
    let mut digest_values = Vec::with_capacity(me_inputs.len());
    for (idx, (claim, native_claim)) in me_inputs.iter().zip(native_claims.iter()).enumerate() {
        digests.push(me_digest_poseidon_with_native_claim(
            &mut cs.namespace(|| format!("me_input_digest_{idx}")),
            claim,
            native_claim,
            &format!("me_input_digest_{idx}"),
        )?);
        digest_values.push(me_digest_poseidon_values_from_native_claim(native_claim));
    }
    bind_me_input_digests(cs, tr, &digests, &digest_values)?;
    Ok(digests)
}

pub fn sample_challenges<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptCircuit,
    dims: Dims,
) -> Result<PiCcsChallengeVars, SynthesisError> {
    tr.append_const_fields_raw(cs.namespace(|| "challenge_domain"), &[SpartanF::from_canonical_u64(2)])?;

    let alpha_beta_gamma = sample_k_vec_batched(
        cs.namespace(|| "alpha_beta_gamma"),
        tr,
        dims.ell_d + dims.ell + 1,
        "alpha_beta_gamma",
    )?;
    if alpha_beta_gamma.len() != dims.ell_d + dims.ell + 1 {
        return Err(SynthesisError::Unsatisfiable);
    }
    let alpha = alpha_beta_gamma[..dims.ell_d].to_vec();
    let beta = &alpha_beta_gamma[dims.ell_d..dims.ell_d + dims.ell];
    let gamma = alpha_beta_gamma[dims.ell_d + dims.ell].clone();

    tr.append_const_fields_raw(cs.namespace(|| "nc_beta_m_domain"), &[SpartanF::from_canonical_u64(3)])?;
    let beta_m = sample_k_vec_batched(cs.namespace(|| "beta_m"), tr, dims.ell_m, "beta_m")?;

    let beta_a = beta[..dims.ell_d].to_vec();
    let beta_r = beta[dims.ell_d..].to_vec();

    Ok(PiCcsChallengeVars {
        alpha,
        beta_a,
        beta_r,
        beta_m,
        gamma,
    })
}

fn sample_k_vec_batched<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    tr: &mut Poseidon2TranscriptCircuit,
    n: usize,
    label: &str,
) -> Result<Vec<KNumVar>, SynthesisError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let fields = tr.challenge_fields_raw(cs.namespace(|| format!("{label}_batch")), n.saturating_mul(2))?;
    if fields.len() != n.saturating_mul(2) {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut out = Vec::with_capacity(n);
    for idx in 0..n {
        out.push(KNumVar {
            c0: fields[idx * 2].get_variable(),
            c1: fields[idx * 2 + 1].get_variable(),
        });
    }
    Ok(out)
}
