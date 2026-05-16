//! Owns direct-CCS program, step, live IVC state, and terminal circuit data.
//!
//! This is the native SuperNeo state layer for arbitrary CCS/R1CS frontends.
//! Terminal proving, frontend lowering, and recursive carrier logic live in
//! sibling modules.

mod append;
mod compress;
mod construction2;
mod init;
mod relation;
mod summary;
pub(crate) mod surface;
mod types;
mod validation;
mod zero_carry;

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::OptimizedStructureCache;
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices_with_sparse_cache, Dims};
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use validation::{superneo_ivc_states_match, validate_direct_ajtai_context};

use super::public_image::{
    direct_boundary_update_digest, direct_initial_boundary_digest, direct_public_trace_seed_digest,
    direct_public_trace_update_digest, direct_state_x_out, direct_vk_fs_digest, DIRECT_CCS_TRIVIAL_PC,
};
use super::snark::{DirectCcsIvcSnark, DirectCcsIvcSnarkVerifierKey};
use super::step::DirectCcsStep;
use super::terminal::committed::{
    DirectCcsTerminalCommittedConstraintBreakdown, DirectCcsTerminalCommittedProof, DirectCcsTerminalCommittedRelation,
};
use super::terminal::construction2_fold::DirectCcsConstruction2FoldContext;
use super::terminal::final_ce::final_carry_witnesses;
use super::terminal::gadgets::direct_accumulator_digest_from_claims;
use super::terminal::prove_direct_ccs_terminal_snark;
use crate::construction2::{Construction2EncodedPublicInput, Construction2FreshInstance, Construction2PublicBoundary};
use crate::ivc::{SuperNeoIvcState, SuperNeoIvcStepRelation, SuperNeoIvcTranscriptSnapshot};
use crate::proof::{Carry, ChunkInput};
use crate::superneo_circuit::ce_consistency::PaperCeRelationConstraintBreakdown;
use crate::superneo_nifs_circuit::{SuperNeoChunkCover, SuperNeoChunkReplaySurface};
use surface::build_direct_ccs_chunk_surface_from_ivc_relation;
pub(crate) use types::{
    DirectCcsChunkCircuitSurface, DirectCcsFPrimeCircuit, DirectCcsIvcStepRecord, DirectCcsTerminalFPrimeCircuit,
};
pub use types::{
    DirectCcsFPrimeChunkPerf, DirectCcsFPrimeCommittedPerf, DirectCcsFPrimeCommittedSourcePerf,
    DirectCcsFPrimeConstraintPerf, DirectCcsFPrimeFinalCePerf, DirectCcsFPrimeProofSizePerf, DirectCcsFPrimeR1csPerf,
    DirectCcsFPrimeSnarkError, DirectCcsFPrimeSnarkPerf, DirectCcsFPrimeSnarkProof, DirectCcsFPrimeTimingPerf,
    DirectCcsIvcState, DirectCcsLatestFPrimeSummary, DirectCcsProgram,
};
use zero_carry::build_direct_canonical_zero_carry;
