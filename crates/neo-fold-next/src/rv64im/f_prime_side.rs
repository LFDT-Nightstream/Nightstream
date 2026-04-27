//! Owns RV64IM F' side-lane public data normalization.

use std::collections::BTreeMap;

use neo_math::{KExtensions, K};
use p3_field::PrimeField64;
use serde::{Deserialize, Serialize};

use crate::nightstream::rv64im::{Rv64imEvalPublic, Rv64imOpenedObjectPublic, Rv64imSideOpeningPublic};
use crate::rv64im::kernel::{FamilyEvalSchemaId, PackedColumnEval};
use crate::rv64im::SimpleKernelError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionSideClaim {
    pub schema: FamilyEvalSchemaId,
    pub slot: u32,
    pub point_words: Vec<u64>,
    pub payload_words: Vec<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionSideLaneWitness {
    pub(crate) claims: Vec<Rv64imMainRecursionSideClaim>,
}

impl Rv64imMainRecursionSideLaneWitness {
    pub fn zero() -> Self {
        Self { claims: Vec::new() }
    }

    pub fn claims(&self) -> &[Rv64imMainRecursionSideClaim] {
        &self.claims
    }

    pub fn claim_count(&self) -> u64 {
        self.claims.len() as u64
    }

    pub fn is_zero(&self) -> bool {
        self.claims.is_empty()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionPhiSide {
    pub(crate) commitment_words: Vec<Vec<u64>>,
}

impl Rv64imMainRecursionPhiSide {
    pub fn zero() -> Self {
        Self {
            commitment_words: Vec::new(),
        }
    }

    pub fn commitment_words(&self) -> &[Vec<u64>] {
        &self.commitment_words
    }

    pub fn commitment_count(&self) -> u64 {
        self.commitment_words.len() as u64
    }

    pub fn is_zero(&self) -> bool {
        self.commitment_words.is_empty()
    }
}

fn digest32_as_u64_words(digest: [u8; 32]) -> [u64; 4] {
    core::array::from_fn(|limb| {
        let start = limb * 8;
        u64::from_le_bytes(digest[start..start + 8].try_into().expect("digest limb"))
    })
}

fn k_slice_as_u64_words(values: &[K]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|&value| value.as_coeffs().map(|coeff| coeff.as_canonical_u64()))
        .collect()
}

fn packed_column_evals_as_u64_words(values: &[PackedColumnEval]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|column_eval| k_slice_as_u64_words(&column_eval.coeffs))
        .collect()
}

fn build_rv64im_main_recursion_side_claim_from_eval_public(
    eval: &Rv64imEvalPublic,
) -> Result<Rv64imMainRecursionSideClaim, SimpleKernelError> {
    eval.claim.validate().map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM main recursion side-lane adapter carries an internally inconsistent {:?}/{} eval: {err}",
            eval.claim.payload.schema, eval.claim.id.slot
        ))
    })?;
    if eval.digest != eval.expected_digest() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main recursion side-lane adapter carries a stale {:?}/{} eval digest",
            eval.claim.payload.schema, eval.claim.id.slot
        )));
    }
    Ok(Rv64imMainRecursionSideClaim {
        schema: eval.claim.payload.schema,
        slot: eval.claim.id.slot,
        point_words: k_slice_as_u64_words(&eval.claim.point),
        payload_words: packed_column_evals_as_u64_words(&eval.claim.payload.column_evals),
    })
}

fn build_rv64im_main_recursion_phi_side_commitment_words(
    opened_object: &Rv64imOpenedObjectPublic,
) -> Result<Vec<u64>, SimpleKernelError> {
    let Some(expected_schema) = FamilyEvalSchemaId::from_family(opened_object.opened_object.family) else {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main recursion side-lane adapter carries unsupported opened-object family {:?}",
            opened_object.opened_object.family
        )));
    };
    if expected_schema != opened_object.schema {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main recursion side-lane adapter opened-object schema mismatch: expected {:?}, got {:?}",
            expected_schema, opened_object.schema
        )));
    }
    if opened_object.digest != opened_object.expected_digest() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM main recursion side-lane adapter carries a stale {:?} opened-object digest",
            opened_object.schema
        )));
    }

    let mut words = Vec::with_capacity(15);
    words.push(opened_object.schema.tag());
    words.push(opened_object.opened_object.layout_version);
    words.push(opened_object.opened_object.row_domain_log_size as u64);
    words.extend(digest32_as_u64_words(
        opened_object.opened_object.commitment_root_digest,
    ));
    words.extend(digest32_as_u64_words(opened_object.commitment_context.pp_seed_digest));
    words.extend(digest32_as_u64_words(
        opened_object.commitment_context.module_shape_digest,
    ));
    Ok(words)
}

pub fn build_rv64im_main_recursion_side_lane_from_side_opening_public(
    public: &Rv64imSideOpeningPublic,
) -> Result<(Rv64imMainRecursionSideLaneWitness, Rv64imMainRecursionPhiSide), SimpleKernelError> {
    if public.digest != public.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion side-lane adapter carries a stale side-opening public digest".into(),
        ));
    }

    let mut opened_object_by_schema = BTreeMap::<FamilyEvalSchemaId, &Rv64imOpenedObjectPublic>::new();
    let mut previous_schema = None;
    let mut commitment_words = Vec::with_capacity(public.opened_objects.len());
    for opened_object in &public.opened_objects {
        if let Some(previous_schema) = previous_schema {
            if previous_schema >= opened_object.schema {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM main recursion side-lane adapter requires strict canonical opened-object schema order"
                        .into(),
                ));
            }
        }
        if opened_object_by_schema
            .insert(opened_object.schema, opened_object)
            .is_some()
        {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main recursion side-lane adapter carries duplicate {:?} opened objects",
                opened_object.schema
            )));
        }
        commitment_words.push(build_rv64im_main_recursion_phi_side_commitment_words(opened_object)?);
        previous_schema = Some(opened_object.schema);
    }

    let mut previous_key = None;
    let mut claims = Vec::with_capacity(public.evals.len());
    for eval in &public.evals {
        let key = (eval.claim.payload.schema, eval.claim.id.slot);
        if let Some(previous_key) = previous_key {
            if previous_key >= key {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM main recursion side-lane adapter requires strict canonical eval order".into(),
                ));
            }
        }
        let Some(opened_object) = opened_object_by_schema.get(&eval.claim.payload.schema) else {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main recursion side-lane adapter is missing the {:?} opened object for slot {}",
                eval.claim.payload.schema, eval.claim.id.slot
            )));
        };
        if eval.claim.opened_object != opened_object.opened_object
            || eval.claim.commitment_context != opened_object.commitment_context
        {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM main recursion side-lane adapter {:?}/{} eval does not match the opened-object public",
                eval.claim.payload.schema, eval.claim.id.slot
            )));
        }
        claims.push(build_rv64im_main_recursion_side_claim_from_eval_public(eval)?);
        previous_key = Some(key);
    }

    Ok((
        Rv64imMainRecursionSideLaneWitness { claims },
        Rv64imMainRecursionPhiSide { commitment_words },
    ))
}
