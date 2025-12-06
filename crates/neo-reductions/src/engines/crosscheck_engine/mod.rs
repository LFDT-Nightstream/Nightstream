//! Cross-check engine wrapper for validation during development.
//!
//! This module provides the CrossCheckEngine, which runs the optimized engine
//! and validates key identities against paper-exact helpers. This is useful
//! for debugging and ensuring correctness.

mod crosscheck;
mod logging;

pub use crosscheck::{crosscheck_prove, crosscheck_verify, CrossCheckEngine, CrosscheckCfg};
