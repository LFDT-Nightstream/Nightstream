//! Owns public-step and public-chunk digest gadgets for the RV64IM main relation circuit.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::CcsClaim;
use neo_math::F;
use p3_field::PrimeField64;
use spartan2::{bellpepper::poseidon2::hash_packed_goldilocks_fields, provider::goldi::F as SpartanF};

use super::claim::packed_bytes_field_values;

#[derive(Clone)]
pub struct CcsClaimVar {
    pub c_data: Vec<AllocatedNum<SpartanF>>,
    pub x: Vec<AllocatedNum<SpartanF>>,
    pub m_in: usize,
    pub commitment_d: usize,
    pub commitment_kappa: usize,
}

#[derive(Clone)]
pub struct PublicStepVar {
    pub claim: CcsClaimVar,
    pub label_encoding: Vec<AllocatedNum<SpartanF>>,
}

pub fn alloc_public_step<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    step: &crate::proof::PublicStep,
    label: &str,
) -> Result<PublicStepVar, SynthesisError> {
    Ok(PublicStepVar {
        claim: alloc_ccs_claim(cs, &step.mcs, &format!("{label}_claim"))?,
        label_encoding: alloc_packed_bytes_as_witness_fields(cs, step.label.as_bytes(), &format!("{label}_label"))?,
    })
}

pub fn public_step_digest<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    step: &PublicStepVar,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = Vec::new();
    preimage.extend(alloc_const_packed_bytes(
        cs,
        b"neo.fold.next/finalize/public_step_digest/v1",
        &format!("{label}_domain"),
    )?);
    preimage.extend(step.label_encoding.iter().cloned());
    preimage.extend(ccs_claim_digest(cs, &step.claim, &format!("{label}_claim_digest"))?);
    hash_packed_goldilocks_fields(cs.namespace(|| format!("{label}_hash")), &preimage)
}

pub fn public_chunk_instance_digest<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    start_index: &AllocatedNum<SpartanF>,
    public_step_count: &AllocatedNum<SpartanF>,
    steps: &[PublicStepVar],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = Vec::new();
    preimage.extend(alloc_const_packed_bytes(
        cs,
        b"neo.fold.next/finalize/public_chunk_digest/v1",
        &format!("{label}_domain"),
    )?);
    preimage.push(start_index.clone());
    preimage.push(public_step_count.clone());
    for (step_idx, step) in steps.iter().enumerate() {
        preimage.extend(public_step_digest(
            &mut cs.namespace(|| format!("{label}_step_digest_{step_idx}")),
            step,
            &format!("{label}_step_digest_{step_idx}"),
        )?);
    }
    hash_packed_goldilocks_fields(cs.namespace(|| format!("{label}_hash")), &preimage)
}

fn alloc_ccs_claim<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CcsClaim<Commitment, F>,
    label: &str,
) -> Result<CcsClaimVar, SynthesisError> {
    Ok(CcsClaimVar {
        c_data: alloc_f_slice(cs, &claim.c.data, &format!("{label}_c_data"))?,
        x: alloc_f_slice(cs, &claim.x, &format!("{label}_x"))?,
        m_in: claim.m_in,
        commitment_d: claim.c.d,
        commitment_kappa: claim.c.kappa,
    })
}

fn ccs_claim_digest<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CcsClaimVar,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = Vec::new();
    preimage.extend(alloc_const_packed_bytes(
        cs,
        b"neo.fold.next/finalize/ccs_claim_digest/v1",
        &format!("{label}_domain"),
    )?);
    preimage.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(claim.commitment_d as u64),
        &format!("{label}_commitment_d"),
    )?);
    preimage.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(claim.commitment_kappa as u64),
        &format!("{label}_commitment_kappa"),
    )?);
    preimage.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(claim.c_data.len() as u64),
        &format!("{label}_c_len"),
    )?);
    preimage.extend(claim.c_data.iter().cloned());
    preimage.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(claim.x.len() as u64),
        &format!("{label}_x_len"),
    )?);
    preimage.extend(claim.x.iter().cloned());
    preimage.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(claim.m_in as u64),
        &format!("{label}_m_in"),
    )?);
    hash_packed_goldilocks_fields(cs.namespace(|| format!("{label}_hash")), &preimage)
}

fn alloc_f_slice<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    values: &[F],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })
        })
        .collect()
}

fn alloc_packed_bytes_as_witness_fields<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    bytes: &[u8],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    packed_bytes_field_values(bytes)
        .into_iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(value)))
        .collect()
}

fn alloc_const_packed_bytes<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    bytes: &[u8],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    packed_bytes_field_values(bytes)
        .into_iter()
        .enumerate()
        .map(|(idx, value)| alloc_constant(cs, value, &format!("{label}_{idx}")))
        .collect()
}

fn alloc_constant<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: SpartanF,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(value))?;
    cs.enforce(
        || format!("{label}_constant"),
        |lc| lc + out.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (value, CS::one()),
    );
    Ok(out)
}
