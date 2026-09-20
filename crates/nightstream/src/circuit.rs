//! Prepared application ownership and the public recursive proof lifecycle.
//! The expected circuit comes from local preparation, never from proof data.

use neo_math::F;
use p3_field::PrimeField64;

use crate::application::{ApplicationCircuit, ApplicationError};
use crate::assembly::{self, AssemblyError};
use crate::engine::{Engine, EngineError, Prover};
use crate::lifecycle::{ExtendError, PreparedLifecycle, Stage1Envelope, Stage1State, VerifyError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Assembly(#[from] AssemblyError),
    #[error(transparent)]
    Application(#[from] ApplicationError),
    #[error(transparent)]
    Package(#[from] nightstream_fprime::PackageError),
    #[error(transparent)]
    Extend(#[from] ExtendError),
    #[error(transparent)]
    Verify(#[from] VerifyError),
}

/// A Rust application connected to the unchanged exported recursive verifier.
/// Preparation fixes the complete relation and its verifier-owned identity.
pub struct Circuit {
    application: ApplicationCircuit,
    lifecycle: PreparedLifecycle,
}

impl Circuit {
    /// Prepare from the pinned shared reference and a locally defined application.
    /// The assembler applies only the relocation rules exported by Lean.
    pub fn prepare(reference_bytes: &[u8], application: ApplicationCircuit) -> Result<Self, Error> {
        Self::prepare_with_engine(reference_bytes, application, Engine::Optimized)
    }

    /// Prepare one circuit with an explicit prover engine. Unavailable engines
    /// fail before circuit loading; they never select a CPU fallback.
    pub fn prepare_with_engine(
        reference_bytes: &[u8],
        application: ApplicationCircuit,
        engine: Engine,
    ) -> Result<Self, Error> {
        let prover = Prover::new(engine)?;
        let (package, binding) = assembly::prepare(reference_bytes, &application)?;
        let lifecycle = PreparedLifecycle::from_package(package, binding, prover)?;
        Ok(Self { application, lifecycle })
    }

    pub fn engine(&self) -> Engine {
        self.lifecycle.engine()
    }

    pub fn identity(&self) -> [u64; 4] {
        self.lifecycle.package_identity()
    }

    /// Prove the first application step and retain the openings for later steps.
    pub fn prove(&self, initial_state: [F; 4], private_inputs: &[F]) -> Result<Stage1Envelope, Error> {
        self.extend(Stage1Envelope::initial(initial_state), private_inputs)
    }

    pub fn extend(&self, proof: Stage1Envelope, private_inputs: &[F]) -> Result<Stage1Envelope, Error> {
        let witness = self
            .application
            .execute(proof.state().current(), private_inputs)?;
        let words: Vec<_> = private_inputs
            .iter()
            .map(PrimeField64::as_canonical_u64)
            .collect();
        Ok(self
            .lifecycle
            .extend_with_output(proof, &words, witness.output_state())?)
    }

    /// Check every remaining claim against this circuit and the expected state.
    /// This is terminal verification of openings, not a compression backend.
    pub fn verify(&self, expected_state: &Stage1State, proof: &Stage1Envelope) -> Result<(), Error> {
        Ok(self.lifecycle.verify(expected_state, proof)?)
    }
}
