//! Compiled circuit ownership and separate proving and verifier capabilities.
//! The caller selects verifier configuration; proof data cannot replace it.

use std::{path::Path, sync::Arc};

use neo_math::F;
use nightstream_fprime::{LoadedPerApplicationPackage, Stage1VerifierBinding};
use p3_field::PrimeField64;

use crate::application::{ApplicationCircuit, ApplicationError};
use crate::assembly::{self, AssemblyError};
use crate::engine::{Backend, Engine, EngineError};
use crate::lifecycle::{ExtendError, PreparedLifecycle, Stage1Envelope, Stage1State, VerifyError};

mod storage;

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
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Extend(#[from] ExtendError),
    #[error(transparent)]
    Verify(#[from] VerifyError),
    #[error(transparent)]
    Parameters(#[from] neo_params::ParamsError),
}

struct CompiledCircuit {
    application: ApplicationCircuit,
    package: Arc<LoadedPerApplicationPackage>,
    binding: Stage1VerifierBinding,
}

impl CompiledCircuit {
    fn lifecycle(&self, backend: Backend, minimum_security_bits: u32) -> Result<PreparedLifecycle, Error> {
        Ok(PreparedLifecycle::from_package(
            Arc::clone(&self.package),
            self.binding.clone(),
            backend,
            minimum_security_bits,
        )?)
    }
}

/// Reusable circuit data and bindings, independent of the execution engine.
/// The caller chooses whether this package is its expected verifier configuration.
pub struct Circuit {
    compiled: Arc<CompiledCircuit>,
}

impl Circuit {
    pub fn compile(reference_bytes: &[u8], application: ApplicationCircuit) -> Result<Self, Error> {
        let (package, binding) = assembly::prepare(reference_bytes, &application)?;
        crate::lifecycle::validate_key_prefix(
            package.logical_column_count(),
            binding.verifier_context().commitment_key_words(),
        )?;
        Ok(Self {
            compiled: Arc::new(CompiledCircuit {
                application,
                package: Arc::new(package),
                binding,
            }),
        })
    }

    /// Save validated execution data and bindings for use in another process.
    /// The destination must not exist.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        storage::write(path.as_ref(), &self.compiled)
    }

    /// Decode saved circuit data without repeating whole-circuit identity hashing.
    /// Format and execution structure are checked. The caller is responsible
    /// for package provenance and for selecting its expected verifier configuration.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Self {
            compiled: Arc::new(storage::read(path.as_ref())?),
        })
    }

    /// Return the package's claimed circuit identifier.
    /// Compilation computes it from the circuit. Loading preserves the saved
    /// value without recomputing it, so equality does not authenticate a loaded
    /// package. The caller selects the expected verifier configuration.
    pub fn identity(&self) -> [u64; 4] {
        self.compiled.binding.package_identity()
    }

    /// Require the selected profile to meet the caller's positive statistical minimum.
    pub fn prover(&self, engine: Engine, minimum_security_bits: u32) -> Result<Prover, Error> {
        Prover::new(Arc::clone(&self.compiled), Backend::new(engine)?, minimum_security_bits)
    }
}

/// Proof generation from reusable execution data. This does not configure a verifier.
pub struct Prover {
    compiled: Arc<CompiledCircuit>,
    lifecycle: PreparedLifecycle,
}

impl Prover {
    fn new(compiled: Arc<CompiledCircuit>, backend: Backend, minimum_security_bits: u32) -> Result<Self, Error> {
        let lifecycle = compiled.lifecycle(backend, minimum_security_bits)?;
        Ok(Self { compiled, lifecycle })
    }

    /// Load proving data without repeating compilation or whole-circuit hashing.
    /// Saved identities are claims; verification uses an independently configured circuit.
    pub fn load(path: impl AsRef<Path>, engine: Engine, minimum_security_bits: u32) -> Result<Self, Error> {
        let backend = Backend::new(engine)?;
        Self::new(Arc::new(storage::read(path.as_ref())?), backend, minimum_security_bits)
    }

    pub fn engine(&self) -> Engine {
        self.lifecycle.engine()
    }

    pub fn prove(&self, initial_state: [F; 4], private_inputs: &[F]) -> Result<Stage1Envelope, Error> {
        self.extend(&Stage1Envelope::initial(initial_state), private_inputs)
    }

    /// Construct the next proof without changing the supplied proof, including
    /// on input, device, or I/O failure. Packed witnesses share immutable storage.
    pub fn extend(&self, proof: &Stage1Envelope, private_inputs: &[F]) -> Result<Stage1Envelope, Error> {
        let witness = self
            .compiled
            .application
            .execute(proof.state().current(), private_inputs)?;
        let words: Vec<_> = private_inputs
            .iter()
            .map(PrimeField64::as_canonical_u64)
            .collect();
        Ok(self.lifecycle.extend_with_output(
            proof.snapshot(),
            &words,
            witness.output_state(),
            Some(witness.values()),
        )?)
    }
}

/// Terminal verification against a circuit chosen independently of the proof.
pub struct Verifier {
    lifecycle: PreparedLifecycle,
}

impl Verifier {
    /// Use the caller-selected package as the expected circuit configuration.
    /// This does not establish the package's provenance.
    /// Reject a profile below the caller's positive statistical minimum.
    pub fn from_package(package: &Circuit, engine: Engine, minimum_security_bits: u32) -> Result<Self, Error> {
        Ok(Self {
            lifecycle: package
                .compiled
                .lifecycle(Backend::new(engine)?, minimum_security_bits)?,
        })
    }

    /// Compile a local application and use it as the expected configuration.
    pub fn compile(
        reference_bytes: &[u8],
        application: ApplicationCircuit,
        engine: Engine,
        minimum_security_bits: u32,
    ) -> Result<Self, Error> {
        Self::from_package(
            &Circuit::compile(reference_bytes, application)?,
            engine,
            minimum_security_bits,
        )
    }

    pub fn engine(&self) -> Engine {
        self.lifecycle.engine()
    }

    /// Check every remaining claim and opening against the configured circuit.
    pub fn verify(&self, expected_state: &Stage1State, proof: &Stage1Envelope) -> Result<(), Error> {
        Ok(self.lifecycle.verify(expected_state, proof)?)
    }
}

#[cfg(test)]
#[path = "../tests/circuit/compiled_circuit.rs"]
mod compiled_circuit;

#[cfg(test)]
#[path = "../tests/circuit/hash_chain_vector.rs"]
mod hash_chain_vector;
