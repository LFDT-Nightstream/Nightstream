//! Claim-critical code that Aeneas translates to Lean.
//!
//! Owns: the functions that `formal/compression-lean/CompressionLeanRust` proves equal to
//! their Lean copies, and the circuit core that they run on (`Backend`, `Native`, `Recorder`).
//! Does not own their callers or the prover's fast paths. `scripts/aeneas_extract.sh` enforces
//! the code rules in `.audit/aeneas-plan.md`.
#![forbid(unsafe_code)]
#![deny(clippy::arithmetic_side_effects)]

pub mod algebra;
pub mod backend;
pub mod block;
pub mod circuit;
pub mod degrees;
pub mod diagonal;
pub mod field;
pub mod hash;
pub mod layout;
pub mod native;
pub mod paths;
pub mod poseidon2;
pub mod r1cs;
pub mod record;
pub mod sumcheck;
