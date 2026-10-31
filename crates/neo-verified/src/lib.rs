//! Claim-critical code that Aeneas translates to Lean.
//!
//! Owns: the functions that `formal/compression-lean/CompressionLeanRust` proves equal to
//! their Lean copies. Does not own their callers. `scripts/aeneas_extract.sh` enforces the
//! code rules in `.audit/aeneas-plan.md`.
#![forbid(unsafe_code)]
#![deny(clippy::arithmetic_side_effects)]

pub mod circuit;
pub mod degrees;
pub mod layout;
