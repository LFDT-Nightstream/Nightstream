//! Owns optional Spartan compression for the native RV64IM IVC carrier.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};

mod chunk_step_circuit;
mod chunk_step_spartan;
mod spartan_support;

use self::chunk_step_spartan::{
    debug_check_rv64im_chunk_step_ivc_spartan_circuit, prove_rv64im_chunk_step_ivc_spartan,
    setup_rv64im_chunk_step_ivc_spartan, setup_rv64im_chunk_step_ivc_spartan_cached,
    verify_rv64im_chunk_step_ivc_spartan,
};
pub(crate) use self::spartan_support::{
    hash_packed_goldilocks_fields, GoldilocksP3MerkleMleEngine, R1CSSNARKTrait, Rv64imDeciderEngine,
    Rv64imDeciderProverKey, Rv64imDeciderSnark, Rv64imDeciderVerifierKey, ShapeCS, SpartanCircuit, SpartanF,
    SpartanProverKey, SpartanShape, SpartanVerifierKey, SplitR1CSShape, R1CSSNARK,
};
use crate::rv64im::chunk_step_ivc::{build_rv64im_chunk_step_ivc_relations, Rv64imChunkStepIvcRelation};
use crate::rv64im::final_relation::{Rv64imFinalBuildProof, Rv64imFinalStatement};
use crate::rv64im::ivc::{build_rv64im_ivc_state_from_relations, Rv64imIvcPublicImage, Rv64imIvcState};
use crate::rv64im::main_relation_spartan::{build_rv64im_chunk_step_ivc_shape, Rv64imChunkStepIvcShape};
use crate::rv64im::SimpleKernelError;

pub type Rv64imIvcSnarkProverKey = Rv64imDeciderProverKey;
pub type Rv64imIvcSnarkVerifierKey = Rv64imDeciderVerifierKey;
pub type Rv64imIvcSnarkKeyPair = Arc<(Rv64imIvcSnarkProverKey, Rv64imIvcSnarkVerifierKey)>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imTerminalDeciderSetupShape {
    terminal_step_shape: Rv64imChunkStepIvcShape,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imIvcSnarkProof {
    pub snark_data: Vec<u8>,
}

impl Rv64imIvcSnarkProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imIvcSnark {
    proof: Rv64imIvcSnarkProof,
    public_image: Rv64imIvcPublicImage,
}

static RV64IM_IVC_SNARK_PROOF_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Arc<Rv64imIvcSnarkProof>>>> = OnceLock::new();
static RV64IM_IVC_SNARK_RELATION_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Arc<Rv64imChunkStepIvcRelation>>>> =
    OnceLock::new();

fn rv64im_ivc_snark_cache_key(statement: &Rv64imFinalStatement, proof: &Rv64imFinalBuildProof) -> [u8; 32] {
    let mut digest = [0u8; 32];
    for ((dst, lhs), rhs) in digest
        .iter_mut()
        .zip(statement.digest.iter())
        .zip(proof.proof_digest.iter())
    {
        *dst = *lhs ^ *rhs;
    }
    digest
}

fn build_rv64im_terminal_step_relation(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Rv64imChunkStepIvcRelation, SimpleKernelError> {
    let relations = build_rv64im_chunk_step_ivc_relations(statement, proof)?;
    let ivc_state = build_rv64im_ivc_state_from_relations(&relations)?;
    ivc_state.build_terminal_relation()
}

fn build_rv64im_terminal_step_relation_cached(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Arc<Rv64imChunkStepIvcRelation>, SimpleKernelError> {
    let cache_key = rv64im_ivc_snark_cache_key(statement, proof);
    let cache = RV64IM_IVC_SNARK_RELATION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(relation) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM IVC SNARK relation cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(relation);
    }

    let relation = Arc::new(build_rv64im_terminal_step_relation(statement, proof)?);
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM IVC SNARK relation cache poisoned".into()))?
        .insert(cache_key, relation.clone());
    Ok(relation)
}

fn setup_rv64im_ivc_snark_from_terminal_relation_cached(
    terminal_relation: &Rv64imChunkStepIvcRelation,
) -> Result<Rv64imIvcSnarkKeyPair, SimpleKernelError> {
    setup_rv64im_chunk_step_ivc_spartan_cached(&terminal_relation.statement, &terminal_relation.witness)
}

fn prove_rv64im_ivc_snark_on_terminal_relation(
    pk: &Rv64imIvcSnarkProverKey,
    terminal_relation: &Rv64imChunkStepIvcRelation,
) -> Result<Rv64imIvcSnarkProof, SimpleKernelError> {
    let snark_data = prove_rv64im_chunk_step_ivc_spartan(pk, &terminal_relation.statement, &terminal_relation.witness)?;
    Ok(Rv64imIvcSnarkProof { snark_data })
}

impl Rv64imIvcSnark {
    pub(crate) fn from_parts(proof: Rv64imIvcSnarkProof, public_image: Rv64imIvcPublicImage) -> Self {
        Self { proof, public_image }
    }

    pub fn proof(&self) -> &Rv64imIvcSnarkProof {
        &self.proof
    }

    pub fn proof_mut(&mut self) -> &mut Rv64imIvcSnarkProof {
        &mut self.proof
    }

    pub fn public_image(&self) -> &Rv64imIvcPublicImage {
        &self.public_image
    }

    pub fn public_image_mut(&mut self) -> &mut Rv64imIvcPublicImage {
        &mut self.public_image
    }

    pub fn verify(
        &self,
        vk: &Rv64imIvcSnarkVerifierKey,
        expected_public_image: &Rv64imIvcPublicImage,
    ) -> Result<(), SimpleKernelError> {
        if &self.public_image != expected_public_image {
            return Err(SimpleKernelError::Bridge(
                "RV64IM IVC SNARK public image does not match the caller-supplied public image".into(),
            ));
        }
        let terminal_statement = self
            .public_image
            .terminal_statement
            .as_ref()
            .ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV64IM IVC SNARK verify requires a terminal statement in the public image".into(),
                )
            })?;
        verify_rv64im_chunk_step_ivc_spartan(vk, terminal_statement, &self.proof.snark_data)
    }
}

pub fn build_rv64im_terminal_decider_setup_shape_from_components(
    statement: &Rv64imFinalStatement,
    proof_digest: [u8; 32],
    kernel_export: &crate::rv64im::kernel::Rv64imKernelExportProof,
    chunk_summaries: &[crate::finalize::FixedShapeChunkSummary],
    steps: &[crate::rv64im::final_relation::Rv64imChunkTransitionWitness],
) -> Result<Rv64imTerminalDeciderSetupShape, SimpleKernelError> {
    let proof = Rv64imFinalBuildProof {
        proof_digest,
        kernel_export: kernel_export.clone(),
        chunk_summaries: chunk_summaries.to_vec(),
        steps: steps.to_vec(),
    };
    let terminal_relation = build_rv64im_terminal_step_relation_cached(statement, &proof)?;
    let terminal_step_shape =
        build_rv64im_chunk_step_ivc_shape(&terminal_relation.statement, &terminal_relation.witness)
            .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM terminal decider shape build failed: {err}")))?;
    Ok(Rv64imTerminalDeciderSetupShape { terminal_step_shape })
}

pub fn debug_check_rv64im_terminal_decider_circuit(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<(), SimpleKernelError> {
    let terminal_relation = build_rv64im_terminal_step_relation_cached(statement, proof)?;
    debug_check_rv64im_chunk_step_ivc_spartan_circuit(&terminal_relation.statement, &terminal_relation.witness)?;
    let keys = setup_rv64im_ivc_snark_from_terminal_relation_cached(&terminal_relation)?;
    let (pk, vk) = &*keys;
    let proof = prove_rv64im_chunk_step_ivc_spartan(pk, &terminal_relation.statement, &terminal_relation.witness)?;
    verify_rv64im_chunk_step_ivc_spartan(vk, &terminal_relation.statement, &proof)?;
    Ok(())
}

pub fn setup_rv64im_ivc_snark_from_final(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<(Rv64imIvcSnarkProverKey, Rv64imIvcSnarkVerifierKey), SimpleKernelError> {
    let terminal_relation = build_rv64im_terminal_step_relation(statement, proof)?;
    setup_rv64im_chunk_step_ivc_spartan(&terminal_relation.statement, &terminal_relation.witness)
}

pub fn setup_rv64im_ivc_snark_from_final_cached(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Rv64imIvcSnarkKeyPair, SimpleKernelError> {
    let terminal_relation = build_rv64im_terminal_step_relation_cached(statement, proof)?;
    setup_rv64im_ivc_snark_from_terminal_relation_cached(&terminal_relation)
}

pub fn prove_rv64im_ivc_snark_from_final(
    pk: &Rv64imIvcSnarkProverKey,
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
    public_image: Rv64imIvcPublicImage,
) -> Result<Rv64imIvcSnark, SimpleKernelError> {
    let terminal_relation = build_rv64im_terminal_step_relation(statement, proof)?;
    let proof = prove_rv64im_ivc_snark_on_terminal_relation(pk, &terminal_relation)?;
    Ok(Rv64imIvcSnark::from_parts(proof, public_image))
}

pub fn prove_rv64im_ivc_snark_from_final_cached(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
    public_image: Rv64imIvcPublicImage,
) -> Result<Rv64imIvcSnark, SimpleKernelError> {
    let cache_key = rv64im_ivc_snark_cache_key(statement, proof);
    let cache = RV64IM_IVC_SNARK_PROOF_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(proof) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM IVC SNARK proof cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(Rv64imIvcSnark::from_parts((*proof).clone(), public_image));
    }

    let terminal_relation = build_rv64im_terminal_step_relation_cached(statement, proof)?;
    let keys = setup_rv64im_ivc_snark_from_terminal_relation_cached(&terminal_relation)?;
    let proof = Arc::new(prove_rv64im_ivc_snark_on_terminal_relation(
        &keys.as_ref().0,
        &terminal_relation,
    )?);
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM IVC SNARK proof cache poisoned".into()))?
        .insert(cache_key, proof.clone());
    Ok(Rv64imIvcSnark::from_parts((*proof).clone(), public_image))
}

pub fn verify_rv64im_ivc_snark_against_final(
    vk: &Rv64imIvcSnarkVerifierKey,
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
    snark: &Rv64imIvcSnark,
) -> Result<(), SimpleKernelError> {
    let terminal_relation = build_rv64im_terminal_step_relation(statement, proof)?;
    verify_rv64im_chunk_step_ivc_spartan(vk, &terminal_relation.statement, &snark.proof.snark_data)
}

impl Rv64imIvcState {
    pub fn compress(&self) -> Result<Rv64imIvcSnark, SimpleKernelError> {
        self.verify()?;
        let terminal_relation = self.build_terminal_relation()?;
        let keys = setup_rv64im_ivc_snark_from_terminal_relation_cached(&terminal_relation)?;
        let proof = prove_rv64im_chunk_step_ivc_spartan(
            &keys.as_ref().0,
            &terminal_relation.statement,
            &terminal_relation.witness,
        )
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM IVC compression prove failed: {err}")))?;
        Ok(Rv64imIvcSnark {
            proof: Rv64imIvcSnarkProof { snark_data: proof },
            public_image: self.public_image(),
        })
    }
}

pub fn setup_rv64im_ivc_snark_cached(state: &Rv64imIvcState) -> Result<Rv64imIvcSnarkKeyPair, SimpleKernelError> {
    state.verify()?;
    let terminal_relation = state.build_terminal_relation()?;
    setup_rv64im_ivc_snark_from_terminal_relation_cached(&terminal_relation)
}
