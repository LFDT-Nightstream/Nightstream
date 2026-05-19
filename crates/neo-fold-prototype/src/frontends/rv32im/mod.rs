//! Owns the RV32IM frontend.
//!
//! Start here for the RV32IM public API. Implementation ownership is split
//! below by machine execution, chunk folding, F', Construction-2, kernel
//! artifacts, public proof, and stage summaries.

pub mod api;
pub mod audit;
pub mod builder;
pub mod ccs;
pub mod chunk;
pub mod claim_tree;
pub mod construction2;
mod decider;
mod encoded_public_input;
pub mod execute;
pub mod f_prime;
pub mod final_relation;
pub mod isa;
pub mod ivc;
pub(crate) mod ivc_snark;
pub mod kernel;
pub mod layout;
pub mod lower;
pub mod main_proof;
pub(crate) mod main_relation_spartan;
pub(crate) mod main_relation_trace;
mod nifs;
mod perf_case;
pub mod recursion_shape;
mod recursion_spartan;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod tables;
mod trace_expand;

pub use api::*;
