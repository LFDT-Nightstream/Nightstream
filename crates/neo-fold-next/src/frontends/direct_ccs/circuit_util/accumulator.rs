//! Owns direct-CCS accumulator digest construction, both native and in-circuit.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::fields::{
    direct_domain_fields, direct_domain_spartan_fields, push_constant_spartan_fields, spartan_one, spartan_zero,
};
use crate::finalize::digest_fields_as_digest32;
use crate::spartan_backend::SpartanF;
use crate::superneo_circuit::claim::CircuitCeClaim;
use crate::superneo_circuit::transcript::hash_field_linear_combinations_raw;

pub(crate) fn direct_accumulator_digest_from_claims(
    params: &NeoParams,
    claims: &[CeClaim<Commitment, F, K>],
) -> [u8; 32] {
    direct_accumulator_digest_from_claims_with_base(params.b, claims)
}

pub(crate) fn direct_accumulator_digest_from_claims_with_base(
    base: u32,
    claims: &[CeClaim<Commitment, F, K>],
) -> [u8; 32] {
    let mut preimage = direct_domain_fields(b"neo.fold.next/direct_ccs/accumulator_phi_dec_parent/v1");
    preimage.push(F::from_u64(claims.len() as u64));
    if let Some(first) = claims.first() {
        let parent_len = first.c.data.len();
        preimage.push(F::from_u64(parent_len as u64));
        let base = F::from_u64(base as u64);
        let mut powers = Vec::with_capacity(claims.len());
        let mut pow = F::ONE;
        for claim in claims {
            if claim.c.data.len() != parent_len {
                preimage.push(F::from_u64(u64::MAX));
                return digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage));
            }
            powers.push(pow);
            pow *= base;
        }
        for lane_idx in 0..parent_len {
            let mut value = F::ZERO;
            for (claim, pow) in claims.iter().zip(powers.iter().copied()) {
                value += claim.c.data[lane_idx] * pow;
            }
            preimage.push(value);
        }
    }
    digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub(crate) fn direct_accumulator_digest_circuit_from_claims<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    claims: &[CircuitCeClaim],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut field_terms = Vec::new();
    let mut field_constants = Vec::new();
    let mut field_values = Vec::new();
    push_constant_spartan_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        direct_domain_spartan_fields(b"neo.fold.next/direct_ccs/accumulator_phi_dec_parent/v1"),
    );
    field_terms.push(Vec::new());
    field_constants.push(SpartanF::from_canonical_u64(claims.len() as u64));
    field_values.push(SpartanF::from_canonical_u64(claims.len() as u64));
    if let Some(first) = claims.first() {
        let parent_len = first.commitment.data.len();
        field_terms.push(Vec::new());
        field_constants.push(SpartanF::from_canonical_u64(parent_len as u64));
        field_values.push(SpartanF::from_canonical_u64(parent_len as u64));
        let base = SpartanF::from_canonical_u64(params.b as u64);
        let mut powers = Vec::with_capacity(claims.len());
        let mut pow = spartan_one();
        for claim in claims {
            if claim.commitment.data.len() != parent_len
                || claim.commitment.data.len() != claim.commitment.data_values.len()
            {
                return Err(SynthesisError::Unsatisfiable);
            }
            powers.push(pow);
            pow *= base;
        }
        for lane_idx in 0..parent_len {
            let mut terms = Vec::with_capacity(claims.len());
            let mut value = spartan_zero();
            for (claim, pow) in claims.iter().zip(powers.iter().copied()) {
                terms.push((claim.commitment.data[lane_idx].get_variable(), pow));
                value += SpartanF::from_canonical_u64(claim.commitment.data_values[lane_idx].as_canonical_u64()) * pow;
            }
            field_terms.push(terms);
            field_constants.push(spartan_zero());
            field_values.push(value);
        }
    }
    hash_field_linear_combinations_raw(
        cs.namespace(|| "direct_accumulator_digest_hash"),
        &field_terms,
        &field_constants,
        &field_values,
    )
}
