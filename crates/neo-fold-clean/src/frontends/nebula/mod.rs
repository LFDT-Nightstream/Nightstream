//! Nebula memory-checking frontend — offline memory checking (RAM + public
//! ROM) for the SuperNeo folding chain.
//!
//! Owns the frontend-side building blocks, one concern per file:
//!
//! - [`layout`] — plan parameters and every bit-level encoding (lanes, step
//!   public input).
//! - [`fingerprint`] — the packed public-coin fingerprint and running
//!   products over `K`.
//! - [`trace`] — the native memory machine: sequential-consistency
//!   semantics, RS/WS tuple emission, IS/FS snapshots. This is the first pass
//!   of the two-pass prover and the test oracle; it is never verifier
//!   authority.
//! - [`circuit`] — the uniform `S_mem` step circuit: one CCS structure per
//!   plan, plus the step witness builder.
//! - [`plan`] — the validated plan artifact: constants, structure, lane scheme,
//!   `D_init` (the verifier's ROM handle), plan digest.
//! - [`prove`] — the two-pass segment prover: one
//!   [`trace::SegmentTrace`] in, `N` folded `S_mem` steps out.
//! - `f_prime` — the crate-internal Stage 2 fixed-relation prototype. It is not
//!   a public lifecycle until its PiCCS relation consumes the sealed v1.1
//!   package.
//!
//! Does not own: the fold pipeline's `adv` mirroring
//! (`paper/relations`, `paper/reductions`) or the F′ `NebulaLane` carry
//! (`paper/construction2/nebula_lane.rs`).

pub mod application;
pub mod circuit;
// Keep the Stage 2 assembly private; expose only the existing WASM integration types.
#[allow(dead_code, unused_imports)]
pub(crate) mod f_prime;
pub use f_prime::{
    NebulaFPrimeChainBuilder, NebulaFPrimeChainError, NebulaFPrimePreparedProfile, NebulaFPrimePreprocessing,
    VerifiedNebulaFPrimeEncoderArtifact,
};
pub mod fingerprint;
pub mod layout;
pub mod plan;
pub mod prove;
pub mod trace;
