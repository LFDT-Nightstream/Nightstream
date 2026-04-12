//! Owns the direct Spartan proof for the compact RV64IM Nightstream side statement.
//!
//! This circuit no longer replays raw side witnesses. It binds only the
//! Nightstream core digest and the authoritative side public-instance digest.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem, SynthesisError};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeField64;
use serde::{Deserialize, Serialize};
use spartan2::{
    provider::{goldi::F as SpartanF, GoldilocksP3MerkleMleEngine},
    spartan::R1CSSNARK,
    traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};

use super::authoritative_side::{
    build_rv64im_authoritative_side_public_instance, build_rv64im_authoritative_side_statement,
    Rv64imAuthoritativeSidePublicInstance, Rv64imAuthoritativeSideStatement,
};
use super::opening_artifact::build_rv64im_opening_artifact_from_accepted_artifact;
use super::Rv64imSideProofBundle;
use crate::finalize::digest32_as_fields;
use crate::nightstream::NightstreamStatement;
use crate::rv64im::kernel::{Rv64imAcceptedProofArtifact, SimpleKernelError};

pub type Rv64imSideSpartanEngine = GoldilocksP3MerkleMleEngine;
pub type Rv64imSideSpartanSnark = R1CSSNARK<Rv64imSideSpartanEngine>;
pub type Rv64imSideSpartanProverKey = spartan2::spartan::SpartanProverKey<Rv64imSideSpartanEngine>;
pub type Rv64imSideSpartanVerifierKey = spartan2::spartan::SpartanVerifierKey<Rv64imSideSpartanEngine>;

type Rv64imSideSpartanKeyPair = Arc<(Rv64imSideSpartanProverKey, Rv64imSideSpartanVerifierKey)>;

static RV64IM_SIDE_SPARTAN_SETUP_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Rv64imSideSpartanKeyPair>>> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imSideSpartanProof {
    pub snark_data: Vec<u8>,
}

impl Rv64imSideSpartanProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }
}

#[derive(Clone)]
struct Rv64imSideRelationCircuit {
    statement: Rv64imAuthoritativeSideStatement,
}

impl Rv64imSideRelationCircuit {
    fn expected_public_values(&self) -> Vec<SpartanF> {
        digest32_as_fields(self.statement.digest())
            .into_iter()
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
            .collect()
    }
}

impl SpartanCircuit<Rv64imSideSpartanEngine> for Rv64imSideRelationCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.expected_public_values())
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
        for (idx, value) in self.expected_public_values().into_iter().enumerate() {
            let _ = AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value))?;
        }
        Ok(())
    }
}

pub fn setup_rv64im_side_spartan(
    statement: &Rv64imAuthoritativeSideStatement,
    witness: &Rv64imAuthoritativeSidePublicInstance,
) -> Result<(Rv64imSideSpartanProverKey, Rv64imSideSpartanVerifierKey), SimpleKernelError> {
    let _ = witness;
    Rv64imSideSpartanSnark::setup(Rv64imSideRelationCircuit {
        statement: statement.clone(),
    })
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side relation setup failed: {err}")))
}

pub fn setup_rv64im_side_spartan_cached(
    statement: &Rv64imAuthoritativeSideStatement,
    witness: &Rv64imAuthoritativeSidePublicInstance,
) -> Result<Rv64imSideSpartanKeyPair, SimpleKernelError> {
    let shape_digest = rv64im_side_relation_shape_digest(statement, witness);
    let cache = RV64IM_SIDE_SPARTAN_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM side relation setup cache poisoned".into()))?
        .get(&shape_digest)
        .cloned()
    {
        return Ok(keys);
    }

    let keys = Arc::new(setup_rv64im_side_spartan(statement, witness)?);
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM side relation setup cache poisoned".into()))?
        .insert(shape_digest, keys.clone());
    Ok(keys)
}

pub fn prove_rv64im_side_spartan(
    pk: &Rv64imSideSpartanProverKey,
    statement: &Rv64imAuthoritativeSideStatement,
    witness: &Rv64imAuthoritativeSidePublicInstance,
) -> Result<Rv64imSideSpartanProof, SimpleKernelError> {
    let _ = witness;
    let circuit = Rv64imSideRelationCircuit {
        statement: statement.clone(),
    };
    let prep = Rv64imSideSpartanSnark::prep_prove(pk, circuit.clone(), true)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side relation prepare failed: {err}")))?;
    let proof = Rv64imSideSpartanSnark::prove(pk, circuit, &prep, true)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side relation prove failed: {err}")))?;
    let snark_data = bincode::serialize(&proof)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side relation encode failed: {err}")))?;
    Ok(Rv64imSideSpartanProof { snark_data })
}

pub fn build_rv64im_side_spartan_from_accepted_artifact(
    nightstream_statement: &NightstreamStatement,
    side_bundle: &Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(Rv64imAuthoritativeSideStatement, Rv64imAuthoritativeSidePublicInstance), SimpleKernelError> {
    let opening_artifact = build_rv64im_opening_artifact_from_accepted_artifact(
        &accepted_artifact.statement,
        side_bundle,
        accepted_artifact,
    )?;
    let public_instance = build_rv64im_authoritative_side_public_instance(
        nightstream_statement.core_digest(),
        side_bundle,
        &opening_artifact,
    )?;
    let statement = build_rv64im_authoritative_side_statement(nightstream_statement, &public_instance)?;
    Ok((statement, public_instance))
}

pub fn verify_rv64im_side_spartan(
    vk: &Rv64imSideSpartanVerifierKey,
    statement: &Rv64imAuthoritativeSideStatement,
    proof: &Rv64imSideSpartanProof,
) -> Result<(), SimpleKernelError> {
    let snark: Rv64imSideSpartanSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side relation decode failed: {err}")))?;
    let public_values = snark
        .verify(vk)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side relation verify failed: {err}")))?;
    verify_side_public_io(statement, &public_values)
}

pub fn debug_check_rv64im_side_spartan_circuit(
    statement: &Rv64imAuthoritativeSideStatement,
    witness: &Rv64imAuthoritativeSidePublicInstance,
) -> Result<(), SimpleKernelError> {
    let _ = witness;
    let circuit = Rv64imSideRelationCircuit {
        statement: statement.clone(),
    };
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side relation debug synthesis failed: {err}")))?;
    if !cs.is_satisfied() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM side relation circuit unsatisfied: {}",
            cs.which_is_unsatisfied()
                .unwrap_or_else(|| "unknown".into())
        )));
    }
    Ok(())
}

pub fn measure_rv64im_side_spartan_circuit_constraints(
    statement: &Rv64imAuthoritativeSideStatement,
    witness: &Rv64imAuthoritativeSidePublicInstance,
) -> Result<usize, SimpleKernelError> {
    let _ = witness;
    let circuit = Rv64imSideRelationCircuit {
        statement: statement.clone(),
    };
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side relation counting synthesis failed: {err}")))?;
    if !cs.is_satisfied() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM side relation counting circuit unsatisfied: {}",
            cs.which_is_unsatisfied()
                .unwrap_or_else(|| "unknown".into())
        )));
    }
    Ok(cs.num_constraints())
}

fn verify_side_public_io(
    statement: &Rv64imAuthoritativeSideStatement,
    public_values: &[SpartanF],
) -> Result<(), SimpleKernelError> {
    let expected = digest32_as_fields(statement.digest())
        .into_iter()
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
        .collect::<Vec<_>>();
    if expected != public_values {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side relation public IO mismatch".into(),
        ));
    }
    Ok(())
}

fn rv64im_side_relation_shape_digest(
    statement: &Rv64imAuthoritativeSideStatement,
    witness: &Rv64imAuthoritativeSidePublicInstance,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/side_relation_shape");
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/side_relation_shape/nightstream_statement_core_digest",
        &statement.nightstream_statement_core_digest,
    );
    tr.append_u64s(
        b"neo.fold.next/nightstream/rv64im/side_relation_shape/counts",
        &[
            witness.side_surface_public.targets.len() as u64,
            witness.opened_objects.len() as u64,
            witness.evals.len() as u64,
        ],
    );
    tr.digest32()
}

pub fn setup_rv64im_side_spartan_from_accepted_artifact(
    nightstream_statement: &NightstreamStatement,
    side_bundle: &Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(Rv64imSideSpartanProverKey, Rv64imSideSpartanVerifierKey), SimpleKernelError> {
    let (statement, witness) =
        build_rv64im_side_spartan_from_accepted_artifact(nightstream_statement, side_bundle, accepted_artifact)?;
    setup_rv64im_side_spartan(&statement, &witness)
}
