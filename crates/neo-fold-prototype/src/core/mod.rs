//! Shared SuperNeo and Construction-2 proof plumbing.
//!
//! Main flow:
//! - `session` is the generic proof-session controller. Start there for
//!   top-level prove/verify orchestration.
//! - `prover` and `verifier` own one-chunk facade objects used by `session`.
//! - `chunk_folding` owns the one-chunk SuperNeo `Π_CCS -> Π_RLC -> Π_DEC`
//!   transition.
//! - `proof` owns shared session/chunk/carry data types.
//! - `finalize` owns packaged proof/public digest boundaries.
//! - `ivc` owns native IVC relation construction for recursive/direct callers.
//! - `construction2` owns public-image primitives plus terminal committed-step
//!   helpers.
//! - `opening` owns opening claim types plus the shared time-opening reduction.
//!
//! Support modules: `witness_layout` and `multilinear`.

pub mod chunk_folding;
pub mod construction2;
pub mod finalize;
pub mod ivc;
pub(crate) mod multilinear;
pub mod opening;
pub mod proof;
pub mod prover;
pub mod session;
pub mod step_build;
pub mod verifier;
pub mod witness_layout;
