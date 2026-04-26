//! Owns a direct Spartan proof surface for one CE claim and witness.

use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CcsWitness, CeClaim};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::ce_consistency::{debug_paper_dec_child_y_ring_formula_mismatch, enforce_paper_ce_claim_consistency};
use super::claim::{
    alloc_ce_claim, me_input_projection_digest_poseidon, me_input_projection_digest_poseidon_values_from_native_claim,
};
use super::witness::alloc_packed_witness;
use crate::rv64im::ivc_snark::{
    GoldilocksP3MerkleMleEngine, R1CSSNARKTrait, SpartanCircuit, SpartanF, SpartanProverKey, SpartanVerifierKey,
    R1CSSNARK,
};

pub type Rv64imCeRelationEngine = GoldilocksP3MerkleMleEngine;
pub type Rv64imCeRelationSnark = R1CSSNARK<Rv64imCeRelationEngine>;
pub type Rv64imCeRelationProverKey = SpartanProverKey<Rv64imCeRelationEngine>;
pub type Rv64imCeRelationVerifierKey = SpartanVerifierKey<Rv64imCeRelationEngine>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imCeRelationProof {
    pub snark_data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imCeBundleProof {
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

#[derive(Clone)]
struct Rv64imCeBundleCircuit {
    params: NeoParams,
    structure: CcsStructure<F>,
    claims: Vec<CeClaim<Commitment, F, K>>,
    witnesses: Vec<CcsWitness<F>>,
    delta: SpartanF,
}

impl Rv64imCeRelationCircuit {
    fn expected_public_values(&self) -> Result<[SpartanF; 4], SynthesisError> {
        me_input_projection_digest_poseidon_values_from_native_claim(&self.claim)
    }
}

impl Rv64imCeBundleCircuit {
    fn expected_public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        let mut out = Vec::with_capacity(self.claims.len() * 4);
        for claim in &self.claims {
            out.extend(me_input_projection_digest_poseidon_values_from_native_claim(claim)?);
        }
        Ok(out)
    }
}

impl SpartanCircuit<Rv64imCeRelationEngine> for Rv64imCeRelationCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.expected_public_values()?.to_vec())
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
            .expected_public_values()?
            .into_iter()
            .enumerate()
            .map(|(idx, value)| {
                AllocatedNum::alloc_input(cs.namespace(|| format!("claim_digest_input_{idx}")), || Ok(value))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let claim = alloc_ce_claim(&mut cs.namespace(|| "claim"), &self.claim, "claim")?;
        let witness = alloc_packed_witness(&mut cs.namespace(|| "witness"), &self.witness, "witness")?;
        let digest = me_input_projection_digest_poseidon(&mut cs.namespace(|| "claim_digest"), &claim, "claim_digest")?;
        for (idx, (actual, expected)) in digest.iter().zip(public_inputs.iter()).enumerate() {
            cs.enforce(
                || format!("claim_digest_match_{idx}"),
                |lc| lc + actual.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + expected.get_variable(),
            );
        }

        enforce_paper_ce_claim_consistency(
            &mut cs.namespace(|| "ce_consistency"),
            &self.params,
            &self.structure,
            &self.structure,
            &witness,
            &claim,
            self.delta,
            "ce",
        )?;
        Ok(())
    }
}

impl SpartanCircuit<Rv64imCeRelationEngine> for Rv64imCeBundleCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        self.expected_public_values()
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
        if self.claims.len() != self.witnesses.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        let public_values = self.expected_public_values()?;
        let public_inputs = public_values
            .into_iter()
            .enumerate()
            .map(|(idx, value)| {
                AllocatedNum::alloc_input(cs.namespace(|| format!("claim_digest_input_{idx}")), || Ok(value))
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (claim_idx, (claim, witness)) in self.claims.iter().zip(self.witnesses.iter()).enumerate() {
            let claim_var = alloc_ce_claim(
                &mut cs.namespace(|| format!("claim_{claim_idx}")),
                claim,
                &format!("claim_{claim_idx}"),
            )?;
            let witness_var = alloc_packed_witness(
                &mut cs.namespace(|| format!("witness_{claim_idx}")),
                witness,
                &format!("witness_{claim_idx}"),
            )?;
            let digest = me_input_projection_digest_poseidon(
                &mut cs.namespace(|| format!("claim_digest_{claim_idx}")),
                &claim_var,
                &format!("claim_digest_{claim_idx}"),
            )?;
            let input_offset = claim_idx
                .checked_mul(4)
                .ok_or(SynthesisError::Unsatisfiable)?;
            for (idx, (actual, expected)) in digest
                .iter()
                .zip(public_inputs[input_offset..input_offset + 4].iter())
                .enumerate()
            {
                cs.enforce(
                    || format!("claim_digest_match_{claim_idx}_{idx}"),
                    |lc| lc + actual.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + expected.get_variable(),
                );
            }

            enforce_paper_ce_claim_consistency(
                &mut cs.namespace(|| format!("ce_consistency_{claim_idx}")),
                &self.params,
                &self.structure,
                &self.structure,
                &witness_var,
                &claim_var,
                self.delta,
                &format!("ce_{claim_idx}"),
            )?;
        }
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

pub fn setup_rv64im_ce_bundle_relation(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[CeClaim<Commitment, F, K>],
    witnesses: &[CcsWitness<F>],
    delta: F,
) -> Result<(Rv64imCeRelationProverKey, Rv64imCeRelationVerifierKey), Rv64imCeRelationError> {
    let circuit = Rv64imCeBundleCircuit {
        params: params.clone(),
        structure: structure.clone(),
        claims: claims.to_vec(),
        witnesses: witnesses.to_vec(),
        delta: SpartanF::from_canonical_u64(delta.as_canonical_u64()),
    };
    Rv64imCeRelationSnark::setup(circuit).map_err(|err| Rv64imCeRelationError::Setup(err.to_string()))
}

pub fn debug_check_rv64im_ce_relation(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claim: &CeClaim<Commitment, F, K>,
    witness: &CcsWitness<F>,
    delta: F,
) -> Result<(), Rv64imCeRelationError> {
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let native_y = neo_reductions::common::compute_y_from_Z_and_r(structure, &witness.Z, &claim.r, ell_d, params.b);
    for (matrix_idx, (want_row, got_row)) in claim.y_ring.iter().zip(native_y.0.iter()).enumerate() {
        for rho in 0..D {
            let want = want_row.get(rho).copied().unwrap_or(K::ZERO);
            let got = got_row.get(rho).copied().unwrap_or(K::ZERO);
            if want != got {
                return Err(Rv64imCeRelationError::Prepare(format!(
                    "native CE y_ring mismatch before circuit synthesis: matrix={matrix_idx}, rho={rho}, claim={}, native={}",
                    format_k(want),
                    format_k(got),
                )));
            }
        }
    }
    let forms = neo_ccs::build_superneo_ring_forms::<F, K>(structure, &claim.r)
        .map_err(|err| Rv64imCeRelationError::Prepare(err.to_string()))?;
    let z_coeffs = neo_reductions::common::decode_superneo_coeffs_from_witness_mat(&witness.Z, structure.m)
        .map_err(|err| Rv64imCeRelationError::Prepare(err.to_string()))?;
    for (matrix_idx, matrix_forms) in forms.iter().enumerate() {
        for rho in 0..D {
            let mut acc = K::ZERO;
            for (coeffs, z) in matrix_forms.iter().zip(z_coeffs.iter()) {
                acc += coeffs[rho] * *z;
            }
            let want = claim
                .y_ring
                .get(matrix_idx)
                .and_then(|row| row.get(rho))
                .copied()
                .unwrap_or(K::ZERO);
            if acc != want {
                return Err(Rv64imCeRelationError::Prepare(format!(
                    "build_superneo_ring_forms mismatch before circuit synthesis: matrix={matrix_idx}, rho={rho}, claim={}, forms={}",
                    format_k(want),
                    format_k(acc),
                )));
            }
        }
    }
    if let Some(mismatch) = debug_paper_dec_child_y_ring_formula_mismatch(structure, witness, claim)
        .map_err(Rv64imCeRelationError::Prepare)?
    {
        return Err(Rv64imCeRelationError::Prepare(mismatch));
    }

    let circuit = Rv64imCeRelationCircuit {
        params: params.clone(),
        structure: structure.clone(),
        claim: claim.clone(),
        witness: witness.clone(),
        delta: SpartanF::from_canonical_u64(delta.as_canonical_u64()),
    };
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| Rv64imCeRelationError::Prepare(err.to_string()))?;
    if !cs.is_satisfied() {
        return Err(Rv64imCeRelationError::Prepare(
            cs.which_is_unsatisfied()
                .unwrap_or("unknown unsatisfied final CE constraint")
                .to_string(),
        ));
    }
    Ok(())
}

fn format_k(value: K) -> String {
    let [re, im] = value.as_coeffs();
    format!("({}, {})", re.as_canonical_u64(), im.as_canonical_u64())
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

pub fn prove_rv64im_ce_bundle_relation(
    pk: &Rv64imCeRelationProverKey,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[CeClaim<Commitment, F, K>],
    witnesses: &[CcsWitness<F>],
    delta: F,
) -> Result<Rv64imCeBundleProof, Rv64imCeRelationError> {
    let circuit = Rv64imCeBundleCircuit {
        params: params.clone(),
        structure: structure.clone(),
        claims: claims.to_vec(),
        witnesses: witnesses.to_vec(),
        delta: SpartanF::from_canonical_u64(delta.as_canonical_u64()),
    };
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| Rv64imCeRelationError::Prepare(err.to_string()))?;
    if !cs.is_satisfied() {
        return Err(Rv64imCeRelationError::Prepare(
            cs.which_is_unsatisfied()
                .unwrap_or("unknown unsatisfied final CE bundle constraint")
                .to_string(),
        ));
    }
    let prep = Rv64imCeRelationSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| Rv64imCeRelationError::Prepare(err.to_string()))?;
    let proof = Rv64imCeRelationSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| Rv64imCeRelationError::Prove(err.to_string()))?;
    let snark_data = bincode::serialize(&proof).map_err(|err| Rv64imCeRelationError::Encode(err.to_string()))?;
    Ok(Rv64imCeBundleProof { snark_data })
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
    let expected = me_input_projection_digest_poseidon_values_from_native_claim(claim)
        .map_err(|_| Rv64imCeRelationError::PublicIoMismatch)?;
    if public_values != expected.to_vec() {
        return Err(Rv64imCeRelationError::PublicIoMismatch);
    }
    Ok(())
}

pub fn verify_rv64im_ce_bundle_relation(
    vk: &Rv64imCeRelationVerifierKey,
    claims: &[CeClaim<Commitment, F, K>],
    proof: &Rv64imCeBundleProof,
) -> Result<(), Rv64imCeRelationError> {
    let proof: Rv64imCeRelationSnark =
        bincode::deserialize(&proof.snark_data).map_err(|err| Rv64imCeRelationError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Rv64imCeRelationError::Verify(err.to_string()))?;
    let mut expected = Vec::with_capacity(claims.len() * 4);
    for claim in claims {
        expected.extend(
            me_input_projection_digest_poseidon_values_from_native_claim(claim)
                .map_err(|_| Rv64imCeRelationError::PublicIoMismatch)?,
        );
    }
    if public_values != expected {
        return Err(Rv64imCeRelationError::PublicIoMismatch);
    }
    Ok(())
}
