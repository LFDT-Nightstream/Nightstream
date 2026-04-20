//! Owns audit-only chunk-step replay helpers and native shape builders.

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
    Rv64imChunkStepIvcRecursiveStepPadding, Rv64imChunkStepIvcShape, Rv64imChunkStepIvcSpartanError,
};

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
