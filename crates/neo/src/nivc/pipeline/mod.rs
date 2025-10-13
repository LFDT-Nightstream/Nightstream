//! NIVC pipeline modules: prover, verifier, finalizer

pub mod prover;
pub mod verifier;
pub mod finalizer;

// Re-export main entry points
pub use prover::step as prove_step;
pub use verifier::verify_chain;
pub use finalizer::{finalize, finalize_with_options, NivcFinalizeOptions};

