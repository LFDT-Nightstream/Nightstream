//! Circuit synthesis for FoldRun verification
//!
//! This module converts a FoldRun into R1CS constraints that verify:
//! 1. Each step's Π-CCS proof (terminal identity + sumcheck)
//! 2. RLC equalities
//! 3. DEC equalities
//! 4. Accumulator chaining between steps

pub mod fold_circuit;
pub mod fold_circuit_helpers;
pub mod route_a_memory_terminal;
pub mod route_a_time;
pub mod witness;

pub use fold_circuit::FoldRunCircuit;
pub use witness::{FoldRunInstance, FoldRunWitness};
