//! Owns route-specific bellpepper gadgets for the RV64IM main Spartan relation.
//!
//! This module exists so RV64IM can compile its theorem relation directly,
//! instead of routing theorem meaning through generic digest shells.

pub mod carrier;
pub mod ce_consistency;
pub mod ce_spartan;
pub mod claim;
pub mod initial_sum;
pub mod k_field;
pub mod output_binding;
pub mod pi_ccs;
pub mod pi_dec;
pub mod pi_rlc;
pub mod public_chunk;
pub mod rho_sampling;
pub mod structure;
pub mod sumcheck;
pub mod sumcheck_replay;
pub mod terminal_common;
pub mod terminal_identity;
pub mod transcript;
pub mod witness;
pub mod witness_transition;
