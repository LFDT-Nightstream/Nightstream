//! Engine layer: implementation backing the paper layer.
//!
//! Auditor: nothing here is a paper claim. If you trust `neo-reductions` /
//! `neo-ccs` / `neo-ajtai`, you can trust this. The job of this
//! module is to keep transcript discipline and convert paper-layer values
//! to the engine's wire format.
//!
//! Each submodule is named after the paper section it serves:
//! - `transcript` — Poseidon2 wrapper with one absorb-label namespace.
//! - `r1cs_circuit` — low-level R1CS-builder primitives used by the
//!   in-circuit verifier gadgets in `paper/reductions/*_circuit.rs`.
//! - (`decider` — Spartan terminal compression, added as wiring lands.)

pub mod ccs_native;
pub mod decider;
pub mod optimized;
pub mod paper_exact;
pub mod r1cs_circuit;
pub mod transcript;
