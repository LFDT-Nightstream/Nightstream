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
use crate::spartan_backend::{
    GoldilocksP3MerkleMleEngine, R1CSSNARKTrait, ShapeCS, SpartanCircuit, SpartanF, SpartanProverKey,
    SpartanVerifierKey, R1CSSNARK,
};

pub type Rv32imCeRelationEngine = GoldilocksP3MerkleMleEngine;
pub type Rv32imCeRelationSnark = R1CSSNARK<Rv32imCeRelationEngine>;
pub type Rv32imCeRelationProverKey = SpartanProverKey<Rv32imCeRelationEngine>;
pub type Rv32imCeRelationVerifierKey = SpartanVerifierKey<Rv32imCeRelationEngine>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imCeRelationProof {
    pub snark_data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imCeBundleProof {
    pub snark_data: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rv32imCeBundleConstraintBreakdown {
    pub public_input_count: usize,
    pub claim_count: usize,
    pub total_constraints: usize,
    pub digest_constraints: usize,
    pub digest_match_constraints: usize,
    pub ce_relation_constraints: usize,
}

#[derive(Debug, Error)]
pub enum Rv32imCeRelationError {
    #[error("rv32im ce relation setup failed: {0}")]
    Setup(String),
    #[error("rv32im ce relation prepare failed: {0}")]
    Prepare(String),
    #[error("rv32im ce relation prove failed: {0}")]
    Prove(String),
    #[error("rv32im ce relation verify failed: {0}")]
    Verify(String),
    #[error("rv32im ce relation proof encoding failed: {0}")]
    Encode(String),
    #[error("rv32im ce relation proof decoding failed: {0}")]
    Decode(String),
    #[error("rv32im ce relation public IO mismatch")]
    PublicIoMismatch,
}

#[derive(Clone)]
struct Rv32imCeRelationCircuit {
    params: NeoParams,
    structure: CcsStructure<F>,
    claim: CeClaim<Commitment, F, K>,
    witness: CcsWitness<F>,
    delta: SpartanF,
}

#[derive(Clone)]
struct Rv32imCeBundleCircuit {
    params: NeoParams,
    structure: CcsStructure<F>,
    claims: Vec<CeClaim<Commitment, F, K>>,
    witnesses: Vec<CcsWitness<F>>,
    delta: SpartanF,
}

impl Rv32imCeRelationCircuit {
    fn expected_public_values(&self) -> Result<[SpartanF; 4], SynthesisError> {
        me_input_projection_digest_poseidon_values_from_native_claim(&self.claim)
    }
}

impl Rv32imCeBundleCircuit {
    fn expected_public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        let mut out = Vec::with_capacity(self.claims.len() * 4);
        for claim in &self.claims {
            out.extend(me_input_projection_digest_poseidon_values_from_native_claim(claim)?);
        }
        Ok(out)
    }
}

impl SpartanCircuit<Rv32imCeRelationEngine> for Rv32imCeRelationCircuit {
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

impl SpartanCircuit<Rv32imCeRelationEngine> for Rv32imCeBundleCircuit {
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

pub fn setup_rv32im_ce_relation(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claim: &CeClaim<Commitment, F, K>,
    witness: &CcsWitness<F>,
    delta: F,
) -> Result<(Rv32imCeRelationProverKey, Rv32imCeRelationVerifierKey), Rv32imCeRelationError> {
    let circuit = Rv32imCeRelationCircuit {
        params: params.clone(),
        structure: structure.clone(),
        claim: claim.clone(),
        witness: witness.clone(),
        delta: SpartanF::from_canonical_u64(delta.as_canonical_u64()),
    };
    Rv32imCeRelationSnark::setup(circuit).map_err(|err| Rv32imCeRelationError::Setup(err.to_string()))
}

pub fn setup_rv32im_ce_bundle_relation(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[CeClaim<Commitment, F, K>],
    witnesses: &[CcsWitness<F>],
    delta: F,
) -> Result<(Rv32imCeRelationProverKey, Rv32imCeRelationVerifierKey), Rv32imCeRelationError> {
    let circuit = Rv32imCeBundleCircuit {
        params: params.clone(),
        structure: structure.clone(),
        claims: claims.to_vec(),
        witnesses: witnesses.to_vec(),
        delta: SpartanF::from_canonical_u64(delta.as_canonical_u64()),
    };
    Rv32imCeRelationSnark::setup(circuit).map_err(|err| Rv32imCeRelationError::Setup(err.to_string()))
}

pub fn debug_measure_rv32im_ce_bundle_relation_constraints(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[CeClaim<Commitment, F, K>],
    witnesses: &[CcsWitness<F>],
    delta: F,
) -> Result<Rv32imCeBundleConstraintBreakdown, Rv32imCeRelationError> {
    if claims.len() != witnesses.len() {
        return Err(Rv32imCeRelationError::Prepare(
            "final CE bundle requires one witness per claim".into(),
        ));
    }

    let circuit = Rv32imCeBundleCircuit {
        params: params.clone(),
        structure: structure.clone(),
        claims: claims.to_vec(),
        witnesses: witnesses.to_vec(),
        delta: SpartanF::from_canonical_u64(delta.as_canonical_u64()),
    };
    let public_values = circuit
        .expected_public_values()
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;
    let mut cs = ShapeCS::<Rv32imCeRelationEngine>::new();
    let public_inputs = public_values
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc_input(cs.namespace(|| format!("claim_digest_input_{idx}")), || Ok(value))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;

    let mut out = Rv32imCeBundleConstraintBreakdown {
        public_input_count: public_inputs.len(),
        claim_count: claims.len(),
        ..Rv32imCeBundleConstraintBreakdown::default()
    };

    for (claim_idx, (claim, witness)) in claims.iter().zip(witnesses.iter()).enumerate() {
        let claim_var = alloc_ce_claim(
            &mut cs.namespace(|| format!("claim_{claim_idx}")),
            claim,
            &format!("claim_{claim_idx}"),
        )
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;
        let witness_var = alloc_packed_witness(
            &mut cs.namespace(|| format!("witness_{claim_idx}")),
            witness,
            &format!("witness_{claim_idx}"),
        )
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;

        let before_digest = cs.num_constraints();
        let digest = me_input_projection_digest_poseidon(
            &mut cs.namespace(|| format!("claim_digest_{claim_idx}")),
            &claim_var,
            &format!("claim_digest_{claim_idx}"),
        )
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;
        out.digest_constraints += cs.num_constraints() - before_digest;

        let before_match = cs.num_constraints();
        let one = <ShapeCS<Rv32imCeRelationEngine> as ConstraintSystem<SpartanF>>::one();
        let input_offset = claim_idx
            .checked_mul(4)
            .ok_or_else(|| Rv32imCeRelationError::Prepare("claim digest input offset overflow".into()))?;
        for (idx, (actual, expected)) in digest
            .iter()
            .zip(public_inputs[input_offset..input_offset + 4].iter())
            .enumerate()
        {
            cs.enforce(
                || format!("claim_digest_match_{claim_idx}_{idx}"),
                |lc| lc + actual.get_variable(),
                |lc| lc + one,
                |lc| lc + expected.get_variable(),
            );
        }
        out.digest_match_constraints += cs.num_constraints() - before_match;

        let before_relation = cs.num_constraints();
        enforce_paper_ce_claim_consistency(
            &mut cs.namespace(|| format!("ce_consistency_{claim_idx}")),
            params,
            structure,
            structure,
            &witness_var,
            &claim_var,
            SpartanF::from_canonical_u64(delta.as_canonical_u64()),
            &format!("ce_{claim_idx}"),
        )
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;
        out.ce_relation_constraints += cs.num_constraints() - before_relation;
    }

    out.total_constraints = cs.num_constraints();
    Ok(out)
}

pub fn debug_check_rv32im_ce_relation(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claim: &CeClaim<Commitment, F, K>,
    witness: &CcsWitness<F>,
    delta: F,
) -> Result<(), Rv32imCeRelationError> {
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let native_y = neo_reductions::common::compute_y_from_Z_and_r(structure, &witness.Z, &claim.r, ell_d, params.b);
    for (matrix_idx, (want_row, got_row)) in claim.y_ring.iter().zip(native_y.0.iter()).enumerate() {
        for rho in 0..D {
            let want = want_row.get(rho).copied().unwrap_or(K::ZERO);
            let got = got_row.get(rho).copied().unwrap_or(K::ZERO);
            if want != got {
                return Err(Rv32imCeRelationError::Prepare(format!(
                    "native CE y_ring mismatch before circuit synthesis: matrix={matrix_idx}, rho={rho}, claim={}, native={}",
                    format_k(want),
                    format_k(got),
                )));
            }
        }
    }
    let forms = neo_ccs::build_superneo_ring_forms::<F, K>(structure, &claim.r)
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;
    let z_coeffs = neo_reductions::common::decode_superneo_coeffs_from_witness_mat(&witness.Z, structure.m)
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;
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
                return Err(Rv32imCeRelationError::Prepare(format!(
                    "build_superneo_ring_forms mismatch before circuit synthesis: matrix={matrix_idx}, rho={rho}, claim={}, forms={}",
                    format_k(want),
                    format_k(acc),
                )));
            }
        }
    }
    if let Some(mismatch) = debug_paper_dec_child_y_ring_formula_mismatch(structure, witness, claim)
        .map_err(Rv32imCeRelationError::Prepare)?
    {
        return Err(Rv32imCeRelationError::Prepare(mismatch));
    }

    let circuit = Rv32imCeRelationCircuit {
        params: params.clone(),
        structure: structure.clone(),
        claim: claim.clone(),
        witness: witness.clone(),
        delta: SpartanF::from_canonical_u64(delta.as_canonical_u64()),
    };
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;
    if !cs.is_satisfied() {
        return Err(Rv32imCeRelationError::Prepare(
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

pub fn prove_rv32im_ce_relation(
    pk: &Rv32imCeRelationProverKey,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claim: &CeClaim<Commitment, F, K>,
    witness: &CcsWitness<F>,
    delta: F,
) -> Result<Rv32imCeRelationProof, Rv32imCeRelationError> {
    let circuit = Rv32imCeRelationCircuit {
        params: params.clone(),
        structure: structure.clone(),
        claim: claim.clone(),
        witness: witness.clone(),
        delta: SpartanF::from_canonical_u64(delta.as_canonical_u64()),
    };
    let prep = Rv32imCeRelationSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;
    let proof = Rv32imCeRelationSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| Rv32imCeRelationError::Prove(err.to_string()))?;
    let snark_data = bincode::serialize(&proof).map_err(|err| Rv32imCeRelationError::Encode(err.to_string()))?;
    Ok(Rv32imCeRelationProof { snark_data })
}

pub fn prove_rv32im_ce_bundle_relation(
    pk: &Rv32imCeRelationProverKey,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    claims: &[CeClaim<Commitment, F, K>],
    witnesses: &[CcsWitness<F>],
    delta: F,
) -> Result<Rv32imCeBundleProof, Rv32imCeRelationError> {
    let circuit = Rv32imCeBundleCircuit {
        params: params.clone(),
        structure: structure.clone(),
        claims: claims.to_vec(),
        witnesses: witnesses.to_vec(),
        delta: SpartanF::from_canonical_u64(delta.as_canonical_u64()),
    };
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;
    if !cs.is_satisfied() {
        return Err(Rv32imCeRelationError::Prepare(
            cs.which_is_unsatisfied()
                .unwrap_or("unknown unsatisfied final CE bundle constraint")
                .to_string(),
        ));
    }
    let prep = Rv32imCeRelationSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| Rv32imCeRelationError::Prepare(err.to_string()))?;
    let proof = Rv32imCeRelationSnark::prove(pk, circuit, &prep, false)
        .map_err(|err| Rv32imCeRelationError::Prove(err.to_string()))?;
    let snark_data = bincode::serialize(&proof).map_err(|err| Rv32imCeRelationError::Encode(err.to_string()))?;
    Ok(Rv32imCeBundleProof { snark_data })
}

pub fn verify_rv32im_ce_relation(
    vk: &Rv32imCeRelationVerifierKey,
    claim: &CeClaim<Commitment, F, K>,
    proof: &Rv32imCeRelationProof,
) -> Result<(), Rv32imCeRelationError> {
    let proof: Rv32imCeRelationSnark =
        bincode::deserialize(&proof.snark_data).map_err(|err| Rv32imCeRelationError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Rv32imCeRelationError::Verify(err.to_string()))?;
    let expected = me_input_projection_digest_poseidon_values_from_native_claim(claim)
        .map_err(|_| Rv32imCeRelationError::PublicIoMismatch)?;
    if public_values != expected.to_vec() {
        return Err(Rv32imCeRelationError::PublicIoMismatch);
    }
    Ok(())
}

pub fn verify_rv32im_ce_bundle_relation(
    vk: &Rv32imCeRelationVerifierKey,
    claims: &[CeClaim<Commitment, F, K>],
    proof: &Rv32imCeBundleProof,
) -> Result<(), Rv32imCeRelationError> {
    let proof: Rv32imCeRelationSnark =
        bincode::deserialize(&proof.snark_data).map_err(|err| Rv32imCeRelationError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Rv32imCeRelationError::Verify(err.to_string()))?;
    let mut expected = Vec::with_capacity(claims.len() * 4);
    for claim in claims {
        expected.extend(
            me_input_projection_digest_poseidon_values_from_native_claim(claim)
                .map_err(|_| Rv32imCeRelationError::PublicIoMismatch)?,
        );
    }
    if public_values != expected {
        return Err(Rv32imCeRelationError::PublicIoMismatch);
    }
    Ok(())
}
