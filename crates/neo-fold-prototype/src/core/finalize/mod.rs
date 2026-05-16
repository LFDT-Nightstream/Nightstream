//! Owns the packaged proof boundary for the active main-lane path.
//!
//! Ownership:
//! - packages the verified session spine into one final proof/public statement pair
//! - binds the package with Poseidon2 digests
//! - verifies finalized packages by replaying the session verifier
//! - does not redefine `Π_CCS -> Π_RLC -> Π_DEC`

mod digest;
mod fixed_shape;
mod package;
mod verify;

pub(crate) use digest::{
    digest32_as_fields, digest_fields_as_digest32, digest_public_statement_from_digests, final_main_claim_digests,
    public_chunk_digest, FIXED_SHAPE_DIGEST_FIELD_LEN,
};
pub use fixed_shape::FixedShapeChunkSummary;
pub(crate) use fixed_shape::{
    digest_fixed_shape_final_proof, fixed_shape_recursive_seed, fixed_shape_recursive_step_handle,
    fixed_shape_terminal_handle_digest_fields, validate_fixed_shape_chunk_layout,
};
pub use package::{package_proof, package_session_proof};
pub use verify::{
    verify_finalized_session, verify_finalized_session_with_perf, verify_finalized_session_with_perf_and_cache,
    verify_packaged_proof,
};
pub(crate) use verify::{verify_finalized_session_with_detailed_perf_and_cache, PackagedVerifyPerf};
