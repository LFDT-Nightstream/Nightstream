//! Phase management for two-axis sum-check decomposition
//!
//! This module defines the phase enum and associated types for tracking
//! whether we're in the row phase (rounds 0..ell_n-1) or Ajtai phase
//! (rounds ell_n..ell_n+ell_d-1).

use neo_math::K;
use p3_field::Field;

/// Oracle phase during sum-check protocol
pub enum Phase<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// Row phase: processing X_r bits (rounds 0..ell_n-1)
    Row(super::engine::RowPhase<'a, F>),
    /// Ajtai phase: processing X_a bits (rounds ell_n..ell_n+ell_d-1)
    Ajtai(super::engine::AjtaiPhase<'a, F>),
}

impl<'a, F> Phase<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// Check if we're in row phase
    pub fn is_row(&self) -> bool {
        matches!(self, Phase::Row(_))
    }
    
    /// Check if we're in Ajtai phase
    pub fn is_ajtai(&self) -> bool {
        matches!(self, Phase::Ajtai(_))
    }
}