//! Frontends — turn user computation into foldable CCS instances.
//!
//! Each frontend lives in a sibling submodule and owns the translation
//! from a user-friendly representation (R1CS, RV32IM trace, …) to the
//! `paper::relations::CcsInstance` the IVC core consumes.
//!
//! ## Soundness boundary
//!
//! A frontend produces `CcsInstance`s; the IVC core folds them. The proof
//! contract depends on the frontend:
//!
//! - `r1cs_f_prime` loads the verifier-owned Lean Stage 1 package. `nebula`
//!   owns its separate memory-checking relation.
//! - `direct_ccs` folds caller-supplied application relations. It proves CCS
//!   satisfaction and NIFS continuity, but it does not assert that each input
//!   is an encoded F' step. Multi-chunk direct-CCS verification therefore
//!   keeps and replays the audit trail.
//!
//! ## Adding a frontend
//!
//! New frontends should expose:
//! - A user-facing relation type (e.g., `direct_ccs::R1cs`).
//! - A `preprocess` entry that takes caller-supplied protocol params and
//!   reads Ajtai setup from the verifier-owned global protocol config.
//!   Frontend convenience for *test* setup (e.g., `preprocess_seeded`) is
//!   fine; proof/prover-supplied setup is not.
//! - A `build_instance` entry that validates user input, packs the
//!   witness, and commits via Ajtai.

pub mod bellpepper;
pub mod direct_ccs;
pub mod f_prime;
pub mod nebula;
pub mod r1cs_f_prime;
