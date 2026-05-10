//! Engine layer: implementation backing the paper layer.
//!
//! Auditor: nothing here is a paper claim. If you trust `neo-reductions` /
//! `neo-ccs` / `neo-ajtai` / `spartan2`, you can trust this. The job of this
//! module is to keep transcript discipline and convert paper-layer values
//! to the engine's wire format.
//!
//! Each submodule is named after the paper section it serves:
//! - `transcript` — Poseidon2 wrapper with one absorb-label namespace.
//! - (`pi_ccs`, `pi_rlc`, `pi_dec`, `decider` — added as wiring lands.)

pub mod optimized;
pub mod transcript;
