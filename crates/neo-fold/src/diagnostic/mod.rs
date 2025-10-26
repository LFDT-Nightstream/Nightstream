//! Constraint diagnostic system for Neo
//!
//! Captures detailed information about failing constraints for debugging,
//! replay, and regression test generation.
//!
//! # Features
//!
//! - **CCS-native**: Works with arbitrary constraint polynomials
//! - **Phase-aware**: Tracks which reduction phase (Π_CCS, Π_RLC, Π_DEC)
//! - **Gradient-based blame**: Mathematically correct sensitivity analysis
//! - **Transcript capture**: Full reproducibility
//! - **Leak-aware witness policies**: Minimal disclosure by default
//! - **Stable hashing**: Canonical serialization for cross-build stability

pub mod types;
pub mod capture;
pub mod blame;
pub mod hash;
pub mod witness;
pub mod export;

#[cfg(feature = "prove-diagnostics")]
pub mod simple_printer;
#[cfg(feature = "prove-diagnostics")]
pub mod ccs_check;

pub use types::*;
pub use capture::capture_diagnostic;
pub use witness::WitnessPolicy;

#[cfg(feature = "prove-diagnostics")]
pub use export::{export_diagnostic, load_diagnostic, DiagnosticFormat};
#[cfg(feature = "prove-diagnostics")]
pub use simple_printer::{print_simple_diagnostic, save_and_print_diagnostic};
#[cfg(feature = "prove-diagnostics")]
pub use ccs_check::check_ccs_with_diagnostics;

/// Schema version for evolution
pub const SCHEMA_VERSION: &str = "neo.constraint.diagnostic@1";


