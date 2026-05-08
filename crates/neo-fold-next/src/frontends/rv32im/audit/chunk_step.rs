//! Owns audit-only chunk-step replay helpers and native shape builders.

pub use crate::rv32im::chunk_step_ivc::{
    audit_check_rv32im_chunk_step_ivc_chain, build_rv32im_chunk_step_ivc_published_target,
    build_rv32im_chunk_step_ivc_relations, rv32im_chunk_step_ivc_initial_state,
    validate_rv32im_chunk_step_ivc_published_statement, validate_rv32im_chunk_step_ivc_surface,
    verify_rv32im_chunk_step_ivc, Rv32imChunkStepIvcPublishedTarget, Rv32imChunkStepIvcRelation,
    Rv32imChunkStepIvcStatement, Rv32imChunkStepIvcWitness,
};
pub use crate::rv32im::chunk_step_relation::{
    build_rv32im_chunk_step_relations, validate_rv32im_chunk_step_relation_surface, verify_rv32im_chunk_step_relation,
    Rv32imChunkStepRelation, Rv32imChunkStepRelationStatement, Rv32imChunkStepRelationWitness,
};
pub use crate::rv32im::main_relation_spartan::{
    build_rv32im_chunk_step_ivc_recursive_step_cover_shape, build_rv32im_chunk_step_ivc_recursive_step_padding,
    build_rv32im_chunk_step_ivc_recursive_step_padding_from_shape, build_rv32im_chunk_step_ivc_shape,
    Rv32imChunkStepIvcRecursiveStepPadding, Rv32imChunkStepIvcShape, Rv32imChunkStepIvcSpartanError,
};

pub fn rv32im_step_statement_chain_digest(relations: &[Rv32imChunkStepIvcRelation]) -> [u8; 32] {
    crate::rv32im::chunk_step_ivc::rv32im_step_statement_chain_digest(relations)
}

pub fn rv32im_recursion_step_statement_chain_digest(relations: &[Rv32imChunkStepIvcRelation]) -> [u8; 32] {
    crate::rv32im::chunk_step_ivc::rv32im_recursion_step_statement_chain_digest(relations)
}

pub fn rv32im_step_statement_chain_digest_init() -> [u8; 32] {
    crate::rv32im::chunk_step_ivc::rv32im_step_statement_chain_digest_init()
}

pub fn rv32im_step_statement_chain_digest_step(current: [u8; 32], digest: [u8; 32]) -> [u8; 32] {
    crate::rv32im::chunk_step_ivc::rv32im_step_statement_chain_digest_step(current, digest)
}

pub fn rv32im_bridge_handoff_chain_digest(relations: &[Rv32imChunkStepIvcRelation]) -> [u8; 32] {
    crate::rv32im::chunk_step_ivc::rv32im_bridge_handoff_chain_digest(relations)
}

pub fn rv32im_bridge_handoff_chain_digest_init() -> [u8; 32] {
    crate::rv32im::chunk_step_ivc::rv32im_bridge_handoff_chain_digest_init()
}

pub fn rv32im_bridge_handoff_chain_digest_step(current: [u8; 32], digest: [u8; 32]) -> [u8; 32] {
    crate::rv32im::chunk_step_ivc::rv32im_bridge_handoff_chain_digest_step(current, digest)
}
