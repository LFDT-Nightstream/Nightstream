//! Recursive proving of an application's steps over circuits that Lean emits,
//! and compression of the final proof.

pub mod application;
pub mod error;

mod artifacts;
mod circuit;
mod engine;
mod folding;
mod lifecycle;
pub mod nebula;
pub mod relation;

#[cfg(test)]
pub(crate) use circuit::References;
pub use circuit::{
    ApplicationSize, Circuit, CompressionKey, CompressionSetup, Error, FinalProof, Proof, Prover, State, Verifier,
};
pub use engine::{Engine, EngineError};
/// The Goldilocks field of every state, input and output word, and the
/// Plonky3 traits that build and read its elements.
pub use p3_field::{PrimeCharacteristicRing, PrimeField64};
pub use p3_goldilocks::Goldilocks;
