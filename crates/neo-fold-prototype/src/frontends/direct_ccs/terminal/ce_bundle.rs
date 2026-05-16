//! Owns the direct-CCS CE-bundle proof used for folded F' accumulators.
//!
//! This is a direct wrapper over the shared SuperNeo CE relation circuit. It
//! keeps RV32IM naming out of the generic direct API while preserving the same
//! terminal boundary shape: the final folded accumulator claims are proof
//! material, and the Spartan proof checks their CE witnesses.

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use super::super::state::DirectCcsFPrimeSnarkError;
use crate::superneo_circuit::ce_spartan::{
    debug_measure_rv32im_ce_bundle_relation_constraints, prove_rv32im_ce_bundle_relation,
    setup_rv32im_ce_bundle_relation, verify_rv32im_ce_bundle_relation, Rv32imCeBundleConstraintBreakdown,
    Rv32imCeBundleProof, Rv32imCeRelationProverKey, Rv32imCeRelationVerifierKey,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsCeBundleProof {
    pub snark_data: Vec<u8>,
}

pub(crate) type DirectCcsCeBundleProverKey = Rv32imCeRelationProverKey;
pub(crate) type DirectCcsCeBundleVerifierKey = Rv32imCeRelationVerifierKey;
pub(crate) type DirectCcsCeBundleConstraintBreakdown = Rv32imCeBundleConstraintBreakdown;

impl DirectCcsCeBundleProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }

    fn as_shared(&self) -> Rv32imCeBundleProof {
        Rv32imCeBundleProof {
            snark_data: self.snark_data.clone(),
        }
    }
}

pub(crate) fn direct_ce_bundle_witnesses(zs: &[Mat<F>]) -> Result<Vec<CcsWitness<F>>, DirectCcsFPrimeSnarkError> {
    zs.iter()
        .enumerate()
        .map(|(idx, z)| {
            if z.rows() != D {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct CE bundle witness {idx} has {} rows, expected {D}",
                    z.rows()
                )));
            }
            Ok(CcsWitness {
                w: Vec::new(),
                Z: z.clone(),
            })
        })
        .collect()
}

pub(crate) fn canonical_direct_ce_claim(claim: &CeClaim<Commitment, F, K>) -> CeClaim<Commitment, F, K> {
    CeClaim {
        c: claim.c.clone(),
        X: claim.X.clone(),
        r: claim.r.clone(),
        s_col: Vec::new(),
        y_ring: claim
            .y_ring
            .iter()
            .map(|row| row.iter().copied().take(D).collect())
            .collect(),
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol: Vec::new(),
        m_in: claim.m_in,
        fold_digest: [0; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

pub(crate) fn canonical_direct_ce_claims(claims: &[CeClaim<Commitment, F, K>]) -> Vec<CeClaim<Commitment, F, K>> {
    claims.iter().map(canonical_direct_ce_claim).collect()
}

pub(crate) fn ensure_direct_ce_claims_are_canonical(
    claims: &[CeClaim<Commitment, F, K>],
) -> Result<(), DirectCcsFPrimeSnarkError> {
    for (idx, claim) in claims.iter().enumerate() {
        if !claim.s_col.is_empty()
            || !claim.ct.is_empty()
            || !claim.aux_openings.is_empty()
            || !claim.y_zcol.is_empty()
            || claim.fold_digest != [0; 32]
            || !claim.c_step_coords.is_empty()
            || claim.u_offset != 0
            || claim.u_len != 0
        {
            return Err(DirectCcsFPrimeSnarkError::Verify(format!(
                "direct CE bundle claim {idx} carries non-authoritative transport fields"
            )));
        }
        let expected_commitment_words = claim.c.kappa.checked_mul(D).ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Verify(format!("direct CE bundle claim {idx} commitment shape overflows"))
        })?;
        if claim.c.d != D || claim.c.kappa == 0 || claim.c.data.len() != expected_commitment_words {
            return Err(DirectCcsFPrimeSnarkError::Verify(format!(
                "direct CE bundle claim {idx} commitment shape is not canonical D x kappa"
            )));
        }
        if claim.X.rows() != D || claim.X.cols() != claim.m_in {
            return Err(DirectCcsFPrimeSnarkError::Verify(format!(
                "direct CE bundle claim {idx} X shape is not D x m_in"
            )));
        }
        for (matrix_idx, row) in claim.y_ring.iter().enumerate() {
            if row.len() != D {
                return Err(DirectCcsFPrimeSnarkError::Verify(format!(
                    "direct CE bundle claim {idx} y_ring[{matrix_idx}] must carry exactly D coefficients"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn setup_direct_ce_bundle_relation(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[CeClaim<Commitment, F, K>],
    witnesses: &[CcsWitness<F>],
) -> Result<(DirectCcsCeBundleProverKey, DirectCcsCeBundleVerifierKey), DirectCcsFPrimeSnarkError> {
    if claims.len() != witnesses.len() {
        return Err(DirectCcsFPrimeSnarkError::Setup(
            "direct CE bundle setup requires one witness per claim".into(),
        ));
    }
    let claims = canonical_direct_ce_claims(claims);
    setup_rv32im_ce_bundle_relation(params, structure, &claims, witnesses, F::from_u64(7))
        .map_err(|err| DirectCcsFPrimeSnarkError::Setup(format!("direct CE bundle setup failed: {err}")))
}

pub(crate) fn measure_direct_ce_bundle_relation(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[CeClaim<Commitment, F, K>],
    witnesses: &[CcsWitness<F>],
) -> Result<DirectCcsCeBundleConstraintBreakdown, DirectCcsFPrimeSnarkError> {
    let claims = canonical_direct_ce_claims(claims);
    debug_measure_rv32im_ce_bundle_relation_constraints(params, structure, &claims, witnesses, F::from_u64(7))
        .map_err(|err| DirectCcsFPrimeSnarkError::Synthesis(format!("direct CE bundle measure failed: {err}")))
}

pub(crate) fn prove_direct_ce_bundle_relation(
    pk: &DirectCcsCeBundleProverKey,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[CeClaim<Commitment, F, K>],
    witnesses: &[CcsWitness<F>],
) -> Result<DirectCcsCeBundleProof, DirectCcsFPrimeSnarkError> {
    let claims = canonical_direct_ce_claims(claims);
    let proof = prove_rv32im_ce_bundle_relation(pk, params, structure, &claims, witnesses, F::from_u64(7))
        .map_err(|err| DirectCcsFPrimeSnarkError::Prove(format!("direct CE bundle prove failed: {err}")))?;
    Ok(DirectCcsCeBundleProof {
        snark_data: proof.snark_data,
    })
}

pub(crate) fn verify_direct_ce_bundle_relation(
    vk: &DirectCcsCeBundleVerifierKey,
    claims: &[CeClaim<Commitment, F, K>],
    proof: &DirectCcsCeBundleProof,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    ensure_direct_ce_claims_are_canonical(claims)?;
    verify_rv32im_ce_bundle_relation(vk, claims, &proof.as_shared())
        .map_err(|err| DirectCcsFPrimeSnarkError::Verify(format!("direct CE bundle verify failed: {err}")))
}
