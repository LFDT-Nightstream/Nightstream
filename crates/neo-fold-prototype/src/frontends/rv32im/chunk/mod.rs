//! Owns RV32IM chunk-level folding and one-step recursion surfaces.

pub mod fold;
pub(crate) mod step_ivc;
pub(crate) mod step_relation;
pub(crate) mod transition;

pub use fold::{
    adapt_rv32im_chunk_to_fresh_ccs, rv32im_chunk_fold_seed, Rv32imAccumulatorHandle, Rv32imChunkFoldCarry,
    Rv32imChunkFoldFresh, Rv32imChunkStepPublic,
};
pub use step_ivc::{
    build_rv32im_chunk_step_ivc_relations, rv32im_chunk_step_ivc_initial_state, Rv32imChunkStepIvcRelation,
    Rv32imChunkStepIvcStatement, Rv32imChunkStepIvcWitness,
};
