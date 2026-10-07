//! Native circuit assembly and recursive proving from Lean-exported verifier components.

pub mod application;
pub mod assembly;
pub use nightstream_fprime::components;

mod circuit;
mod engine;
mod folding;
mod lifecycle;

pub use circuit::{Circuit, Error, Prover, Verifiable, Verifier};
pub use engine::{Engine, EngineError};
pub use lifecycle::{FinalProof, Stage1Envelope as Proof, Stage1State as State};
