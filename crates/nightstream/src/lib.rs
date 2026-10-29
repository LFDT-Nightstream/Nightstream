//! Native circuit assembly and recursive proving from Lean-exported verifier components.

pub mod application;

mod assembly;
mod circuit;
mod engine;
mod folding;
mod lifecycle;

pub use circuit::{Circuit, CompressionKey, CompressionSetup, Error, Prover, Verifier};
pub use engine::{Engine, EngineError};
pub use lifecycle::{FinalProof, Stage1Envelope as Proof, Stage1State as State};
/// The Goldilocks field of every state, input and output word, and the
/// Plonky3 traits that build and read its elements.
pub use p3_field::{PrimeCharacteristicRing, PrimeField64};
pub use p3_goldilocks::Goldilocks;
