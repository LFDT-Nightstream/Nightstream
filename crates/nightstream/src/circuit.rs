//! Compiled circuit ownership and separate proving and verifier capabilities.
//! The caller selects verifier configuration; proof data cannot replace it.

use std::{path::Path, sync::Arc};

use neo_math::F;
use neo_reductions::superneo_eval::RetainedMatrixWindow;
use nightstream_fprime::{LoadedPerApplicationPackage, Stage1VerifierBinding};
use p3_field::PrimeField64;

use crate::application::{ApplicationCircuit, ApplicationError};
use crate::assembly::{self, AssemblyError};
use crate::engine::{Backend, Engine, EngineError};
use crate::lifecycle::{
    ExtendError, FinalProof, FinishError, PreparedLifecycle, ProofCodecError, Stage1Envelope, Stage1State, VerifyError,
};

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
    Finish(#[from] FinishError),
    #[error(transparent)]
    ProofBytes(#[from] ProofCodecError),
    #[error(transparent)]
    Parameters(#[from] neo_params::ParamsError),
    #[error(transparent)]
    Compression(#[from] neo_spartan::Error),
}

struct CompiledCircuit {
    application: ApplicationCircuit,
    package: Arc<LoadedPerApplicationPackage>,
    binding: Stage1VerifierBinding,
    /// The package's complete matrix window, shared by every lifecycle.
    matrix_window: Arc<RetainedMatrixWindow>,
}

impl CompiledCircuit {
    fn lifecycle(&self, backend: Backend, minimum_security_bits: u32) -> Result<PreparedLifecycle, Error> {
        Ok(PreparedLifecycle::from_package(
            Arc::clone(&self.package),
            self.binding.clone(),
            backend,
            minimum_security_bits,
        )?
        .sharing_matrix_window(Arc::clone(&self.matrix_window)))
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
                matrix_window: Arc::default(),
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
        Prover::new(
            Arc::clone(&self.compiled),
            Backend::new_prover(engine)?,
            minimum_security_bits,
        )
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
        let backend = Backend::new_prover(engine)?;
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

    /// The circuit's strict byte encoding of `proof`, for `Verifier::decode_proof`.
    pub fn encode_proof(&self, proof: &Stage1Envelope) -> Result<Vec<u8>, Error> {
        Ok(self.lifecycle.encode_proof(proof)?)
    }

    /// Write this circuit's compression setup files into the existing
    /// directory `dir` (about 29 GB at the production size) and return the
    /// setup. Its key is what verifiers need.
    pub fn compression_setup(&self, dir: impl AsRef<Path>) -> Result<CompressionSetup, Error> {
        Ok(self.lifecycle.compression_setup(dir.as_ref())?)
    }

    /// Reopen setup files written for `key`. The files gain no authority.
    pub fn open_compression_setup(
        &self,
        dir: impl AsRef<Path>,
        key: &CompressionKey,
    ) -> Result<CompressionSetup, Error> {
        Ok(self.lifecycle.open_compression_setup(dir.as_ref(), key)?)
    }

    /// Compress `proof`: one more PiCCS + PiRLC fold without PiDEC, then a
    /// sum-check and WHIR argument for the folded claim. The result carries
    /// no witness and cannot be extended; `proof` does not change.
    pub fn finish_with_spartan(&self, proof: &Stage1Envelope, setup: &CompressionSetup) -> Result<FinalProof, Error> {
        Ok(self.lifecycle.finish_with_spartan(proof, setup)?)
    }

    /// The circuit's byte encoding of `proof`, for `Verifier::decode_final_proof`.
    pub fn encode_final_proof(&self, proof: &FinalProof) -> Result<Vec<u8>, Error> {
        Ok(self.lifecycle.encode_final_proof(proof)?)
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

    /// Decode untrusted proof bytes. The configured circuit fixes the exact
    /// length, which is checked before allocation. Acceptance still needs `verify`.
    pub fn decode_proof(&self, bytes: &[u8]) -> Result<Stage1Envelope, Error> {
        Ok(self.lifecycle.decode_proof(bytes)?)
    }

    /// Decode untrusted finished-proof bytes. Acceptance still needs `verify_final`.
    pub fn decode_final_proof(&self, bytes: &[u8]) -> Result<FinalProof, Error> {
        Ok(self.lifecycle.decode_final_proof(bytes)?)
    }

    /// Check `proof` against the configured circuit and the expected state,
    /// with its witnesses.
    pub fn verify(&self, expected_state: &Stage1State, proof: &Stage1Envelope) -> Result<(), Error> {
        Ok(self.lifecycle.verify(expected_state, proof)?)
    }

    /// Derive the compression key from the configured circuit, without files.
    /// This key, or a copy of it you pinned, is the authority for
    /// `verify_final`. Never take a key from a prover.
    pub fn compression_key(&self) -> Result<CompressionKey, Error> {
        Ok(self.lifecycle.compression_key()?)
    }

    /// Check a finished proof against the configured circuit, the expected
    /// state and a trusted compression key: the layer-0 replay, then the
    /// layer-1 argument, which reads only the key.
    pub fn verify_final(
        &self,
        expected_state: &Stage1State,
        key: &CompressionKey,
        proof: &FinalProof,
    ) -> Result<(), Error> {
        Ok(self.lifecycle.verify_final(expected_state, key, proof)?)
    }
}

/// The verifier's trusted constants of one circuit's compression.
pub type CompressionKey = neo_spartan::Key;
/// The prover's compression setup: the key and the setup files.
pub type CompressionSetup = neo_spartan::Setup;

#[cfg(test)]
#[path = "../tests/circuit/compiled_circuit.rs"]
mod compiled_circuit;

#[cfg(test)]
#[path = "../tests/circuit/hash_chain_vector.rs"]
mod hash_chain_vector;
