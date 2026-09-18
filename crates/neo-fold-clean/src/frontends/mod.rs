//! Frontends — turn user computation into foldable CCS instances.
//!
//! Each frontend lives in a sibling submodule and owns the translation
//! from a user-friendly representation (R1CS, RV32IM trace, …) to the
//! `paper::relations::CcsInstance` the IVC core consumes.
//!
//! ## Soundness boundary
//!
//! A frontend produces `CcsInstance`s; the IVC core folds them and the
//! chain proof attests:
//!
//! 1. Each `CcsInstance` satisfies its CCS relation (witness check inside
//!    Π_CCS sumcheck).
//! 2. The K+k claims at each step folded correctly via NIFS.
//! 3. The chain hash binds consecutive states.
//!
//! What the chain proof does **not** attest, until PR5's decider lands:
//!
//! - That each instance is the encoding of "F'_i ran" — the
//!   Construction-2 augmented step. Without that, there is no in-circuit
//!   proof tying the CCS instance to the previous recursive step's
//!   computation. The IVC core trusts the frontend to supply CCS
//!   instances that match whatever F' the protocol expects.
//!
//! For self-prover use cases (you produced the witnesses, you trust them)
//! this is sufficient. For verifying a third party's computation,
//! Spartan terminal compression (PR5) provides the missing F' binding.
//!
//! ## Adding a frontend
//!
//! New frontends should expose at minimum:
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
