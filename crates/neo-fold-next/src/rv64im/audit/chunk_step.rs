//! Owns audit helpers for chunk-step replay, IVC, and compressed-chain wrappers.

use serde::{Deserialize, Serialize};

use crate::rv64im::main_relation_spartan::{
    prove_rv64im_chunk_step_ivc_spartan_compressed_chain, verify_rv64im_chunk_step_ivc_spartan_compressed_chain,
    Rv64imChunkStepIvcSpartanCompressedChainProof,
};
use crate::rv64im::SimpleKernelError;

pub use crate::rv64im::chunk_step_ivc::{
    build_rv64im_chunk_step_ivc_published_target, build_rv64im_chunk_step_ivc_relations,
    rv64im_chunk_step_ivc_initial_state, validate_rv64im_chunk_step_ivc_published_statement,
    validate_rv64im_chunk_step_ivc_surface, verify_rv64im_chunk_step_ivc, verify_rv64im_chunk_step_ivc_chain,
    Rv64imChunkStepIvcPublishedTarget, Rv64imChunkStepIvcRelation, Rv64imChunkStepIvcStatement,
    Rv64imChunkStepIvcWitness,
};
pub use crate::rv64im::chunk_step_relation::{
    build_rv64im_chunk_step_relations, validate_rv64im_chunk_step_relation_surface, verify_rv64im_chunk_step_relation,
    Rv64imChunkStepRelation, Rv64imChunkStepRelationStatement, Rv64imChunkStepRelationWitness,
};
pub use crate::rv64im::main_relation_spartan::{
    build_rv64im_chunk_step_ivc_recursive_step_cover_shape, build_rv64im_chunk_step_ivc_recursive_step_padding,
    build_rv64im_chunk_step_ivc_recursive_step_padding_from_shape, build_rv64im_chunk_step_ivc_shape,
    prove_rv64im_chunk_step_ivc_spartan, prove_rv64im_chunk_step_ivc_spartan_chain,
    setup_rv64im_chunk_step_ivc_spartan, setup_rv64im_chunk_step_ivc_spartan_cached,
    verify_rv64im_chunk_step_ivc_spartan, verify_rv64im_chunk_step_ivc_spartan_chain,
    Rv64imChunkStepIvcRecursiveStepPadding, Rv64imChunkStepIvcShape, Rv64imChunkStepIvcSpartanChainProof,
    Rv64imChunkStepIvcSpartanError, Rv64imChunkStepIvcSpartanKeyPair, Rv64imChunkStepIvcSpartanProof,
    Rv64imChunkStepIvcSpartanProverKey, Rv64imChunkStepIvcSpartanVerifierKey,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imAuditChunkStepIvcCompressedChainProof {
    pub snark_data: Vec<u8>,
}

pub fn prove_rv64im_chunk_step_ivc_spartan_compressed_chain_audit(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<Rv64imAuditChunkStepIvcCompressedChainProof, SimpleKernelError> {
    let proof = prove_rv64im_chunk_step_ivc_spartan_compressed_chain(relations)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM audit compressed chain prove failed: {err}")))?;
    Ok(Rv64imAuditChunkStepIvcCompressedChainProof {
        snark_data: proof.snark_data,
    })
}

pub fn verify_rv64im_chunk_step_ivc_spartan_compressed_chain_audit(
    relations: &[Rv64imChunkStepIvcRelation],
    proof: &Rv64imAuditChunkStepIvcCompressedChainProof,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_chunk_step_ivc_spartan_compressed_chain(
        relations,
        &Rv64imChunkStepIvcSpartanCompressedChainProof {
            snark_data: proof.snark_data.clone(),
        },
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM audit compressed chain verify failed: {err}")))
}

pub fn rv64im_step_statement_chain_digest(relations: &[Rv64imChunkStepIvcRelation]) -> [u8; 32] {
    crate::rv64im::chunk_step_ivc::rv64im_step_statement_chain_digest(relations)
}

pub fn rv64im_recursion_step_statement_chain_digest(relations: &[Rv64imChunkStepIvcRelation]) -> [u8; 32] {
    crate::rv64im::chunk_step_ivc::rv64im_recursion_step_statement_chain_digest(relations)
}

pub fn rv64im_step_statement_chain_digest_init() -> [u8; 32] {
    crate::rv64im::chunk_step_ivc::rv64im_step_statement_chain_digest_init()
}

pub fn rv64im_step_statement_chain_digest_step(current: [u8; 32], digest: [u8; 32]) -> [u8; 32] {
    crate::rv64im::chunk_step_ivc::rv64im_step_statement_chain_digest_step(current, digest)
}

pub fn rv64im_bridge_handoff_chain_digest(relations: &[Rv64imChunkStepIvcRelation]) -> [u8; 32] {
    crate::rv64im::chunk_step_ivc::rv64im_bridge_handoff_chain_digest(relations)
}

pub fn rv64im_bridge_handoff_chain_digest_init() -> [u8; 32] {
    crate::rv64im::chunk_step_ivc::rv64im_bridge_handoff_chain_digest_init()
}

pub fn rv64im_bridge_handoff_chain_digest_step(current: [u8; 32], digest: [u8; 32]) -> [u8; 32] {
    crate::rv64im::chunk_step_ivc::rv64im_bridge_handoff_chain_digest_step(current, digest)
}
