//! Engine modules for orchestrating the two-phase sum-check oracle
//!
//! This module contains the phase implementations and the main oracle delegator.

pub mod row_phase;
pub mod oracle;

pub use row_phase::RowPhase;
pub use oracle::GenericCcsOracle;