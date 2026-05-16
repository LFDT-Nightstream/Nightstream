//! Owns the generic proof-session SuperNeo driver.
//!
//! This is the main controller for the shared core path. It partitions steps
//! into chunks, threads the main carry and transcript across those chunks,
//! calls the chunk prover/verifier facades, and hands completed sessions to
//! final packaging.

mod cache;
mod layout;
mod package;
mod prove;
mod verify;

pub(crate) use package::verify_packaged_with_detailed_perf_and_cache;
pub use package::{
    prove_and_package, prove_and_package_with_final_carry_perf, prove_and_package_with_perf, verify_packaged,
    verify_packaged_with_perf, verify_packaged_with_perf_and_cache,
};
pub(crate) use prove::prove_chunks_from_slice_with_perf_and_cache;
pub use prove::{
    prove_chunks, prove_chunks_with_cache, prove_chunks_with_perf, prove_chunks_with_perf_and_cache, prove_run,
    prove_run_with_perf,
};
pub use verify::{
    verify_chunks, verify_chunks_with_cache, verify_chunks_with_perf, verify_chunks_with_perf_and_cache, verify_run,
    verify_run_with_perf,
};
