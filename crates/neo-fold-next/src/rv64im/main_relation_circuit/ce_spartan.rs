//! Owns a direct Spartan proof surface for one CE claim and witness.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CcsWitness, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::PrimeField64;
use serde::{Deserialize, Serialize};
use spartan2::{
    provider::{goldi::F as SpartanF, GoldilocksP3MerkleMleEngine},
    spartan::R1CSSNARK,
    traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use thiserror::Error;

use super::ce_consistency::enforce_ce_consistency;
use super::claim::{alloc_ce_claim, me_digest_poseidon};
use super::witness::alloc_packed_witness;

pub type Rv64imCeRelationEngine = GoldilocksP3MerkleMleEngine;
pub type Rv64imCeRelationSnark = R1CSSNARK<Rv64imCeRelationEngine>;
pub type Rv64imCeRelationProverKey = spartan2::spartan::SpartanProverKey<Rv64imCeRelationEngine>;
pub type Rv64imCeRelationVerifierKey = spartan2::spartan::SpartanVerifierKey<Rv64imCeRelationEngine>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imCeRelationProof {
    pub snark_data: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum Rv64imCeRelationError {
    #[error("rv64im ce relation setup failed: {0}")]
    Setup(String),
    #[error("rv64im ce relation prepare failed: {0}")]
    Prepare(String),
    #[error("rv64im ce relation prove failed: {0}")]
    Prove(String),
    #[error("rv64im ce relation verify failed: {0}")]
    Verify(String),
    #[error("rv64im ce relation proof encoding failed: {0}")]
    Encode(String),
    #[error("rv64im ce relation proof decoding failed: {0}")]
    Decode(String),
    #[error("rv64im ce relation public IO mismatch")]
    PublicIoMismatch,
}

#[derive(Clone)]
struct Rv64imCeRelationCircuit {
    params: NeoParams,
    structure: CcsStructure<F>,
    claim: CeClaim<Commitment, F, K>,
    witness: CcsWitness<F>,
    delta: SpartanF,
}

impl Rv64imCeRelationCircuit {
    fn expected_public_values(&self) -> [SpartanF; 4] {
        native_claim_digest_fields(&self.claim)
    }
}

impl SpartanCircuit<Rv64imCeRelationEngine> for Rv64imCeRelationCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.expected_public_values().to_vec())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let public_inputs = self
            .expected_public_values()
            .into_iter()
            .enumerate()
            .map(|(idx, value)| {
                AllocatedNum::alloc_input(cs.namespace(|| format!("claim_digest_input_{idx}")), || Ok(value))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let claim = alloc_ce_claim(&mut cs.namespace(|| "claim"), &self.claim, "claim")?;
        let witness = alloc_packed_witness(&mut cs.namespace(|| "witness"), &self.witness, "witness")?;
        let digest = me_digest_poseidon(&mut cs.namespace(|| "claim_digest"), &claim, "claim_digest")?;
        for (idx, (actual, expected)) in digest.iter().zip(public_inputs.iter()).enumerate() {
            cs.enforce(
                || format!("claim_digest_match_{idx}"),
                |lc| lc + actual.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + expected.get_variable(),
            );
        }

        enforce_ce_consistency(
            &mut cs.namespace(|| "ce_consistency"),
            &self.params,
            &self.structure,
            &witness,
            &claim,
            self.delta,
            "ce",
        )?;
        Ok(())
    }
}

pub fn setup_rv64im_ce_relation(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claim: &CeClaim<Commitment, F, K>,
    witness: &CcsWitness<F>,
    delta: F,
) -> Result<(Rv64imCeRelationProverKey, Rv64imCeRelationVerifierKey), Rv64imCeRelationError> {
    let circuit = Rv64imCeRelationCircuit {
        params: params.clone(),
        structure: structure.clone(),
        claim: claim.clone(),
        witness: witness.clone(),
        delta: SpartanF::from_canonical_u64(delta.as_canonical_u64()),
    };
    Rv64imCeRelationSnark::setup(circuit).map_err(|err| Rv64imCeRelationError::Setup(err.to_string()))
}

pub fn prove_rv64im_ce_relation(
    pk: &Rv64imCeRelationProverKey,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claim: &CeClaim<Commitment, F, K>,
    witness: &CcsWitness<F>,
    delta: F,
) -> Result<Rv64imCeRelationProof, Rv64imCeRelationError> {
    let circuit = Rv64imCeRelationCircuit {
        params: params.clone(),
        structure: structure.clone(),
        claim: claim.clone(),
        witness: witness.clone(),
        delta: SpartanF::from_canonical_u64(delta.as_canonical_u64()),
    };
    let prep = Rv64imCeRelationSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| Rv64imCeRelationError::Prepare(err.to_string()))?;
    let proof = Rv64imCeRelationSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| Rv64imCeRelationError::Prove(err.to_string()))?;
    let snark_data = bincode::serialize(&proof).map_err(|err| Rv64imCeRelationError::Encode(err.to_string()))?;
    Ok(Rv64imCeRelationProof { snark_data })
}

pub fn verify_rv64im_ce_relation(
    vk: &Rv64imCeRelationVerifierKey,
    claim: &CeClaim<Commitment, F, K>,
    proof: &Rv64imCeRelationProof,
) -> Result<(), Rv64imCeRelationError> {
    let proof: Rv64imCeRelationSnark =
        bincode::deserialize(&proof.snark_data).map_err(|err| Rv64imCeRelationError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Rv64imCeRelationError::Verify(err.to_string()))?;
    let expected = native_claim_digest_fields(claim);
    if public_values != expected.to_vec() {
        return Err(Rv64imCeRelationError::PublicIoMismatch);
    }
    Ok(())
}

fn native_claim_digest_fields(claim: &CeClaim<Commitment, F, K>) -> [SpartanF; 4] {
    neo_reductions::engines::utils::me_digest_poseidon(claim)
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
}
