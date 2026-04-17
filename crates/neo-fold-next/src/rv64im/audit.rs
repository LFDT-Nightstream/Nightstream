//! Owns audit-only RV64IM escape hatches grouped by responsibility.
//!
//! `chunk_step` owns one-step and IVC replay helpers.
//! `decider` owns the direct main-relation Spartan compatibility surface.
//! `main_recursion` owns native F', NIFS, and recursive-step Spartan audit helpers.

pub mod chunk_step;
pub mod decider;
pub mod main_recursion;

pub use chunk_step::*;
pub use decider::*;
pub use main_recursion::*;
