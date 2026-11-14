//! The prover and verifier: a light and a refresh step circuit that Lean
//! emits, with the Rust application in place of Lean's, and two slots.
//!
//! The crate embeds the two packages (`artifacts`); the caller selects the
//! application, and the verifier derives everything else from them. Nothing
//! a proof carries is authority.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use neo_math::F;
use nightstream_fprime::{BoundedDepthPackages, Linear, NativeBlock, NativeShape};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::application::{ApplicationCircuit, ApplicationError};
use crate::engine::{Backend, Engine, EngineError};
use crate::lifecycle::bounded_depth as lifecycle;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The selected engine is unavailable.
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Application(#[from] ApplicationError),
    #[error(transparent)]
    Package(#[from] nightstream_fprime::PackageError),
    #[error("invalid compiled circuit: {0}")]
    Circuit(&'static str),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    ProofBytes(#[from] crate::lifecycle::ProofCodecError),
    #[error(transparent)]
    Parameters(#[from] neo_params::ParamsError),
    #[error(transparent)]
    BoundedDepth(crate::lifecycle::bounded_depth::BoundedDepthError),
}

impl From<lifecycle::BoundedDepthError> for Error {
    /// A device failure judged nothing, so it is an engine failure.
    fn from(error: lifecycle::BoundedDepthError) -> Self {
        match error.engine_failure() {
            Some(engine) => Self::Engine(engine.clone()),
            None => Self::BoundedDepth(error),
        }
    }
}

/// The size of an application: its witness words, its local words and its
/// rows. They set the cost of a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApplicationSize {
    pub witness_words: usize,
    pub local_words: usize,
    pub rows: usize,
}

/// Caller-supplied state coordinates. Construction assigns no authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    iteration: u64,
    z0: [F; 4],
    current: [F; 4],
}

impl State {
    pub fn new(iteration: u64, z0: [F; 4], current: [F; 4]) -> Self {
        Self { iteration, z0, current }
    }

    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    pub fn z0(&self) -> [F; 4] {
        self.z0
    }

    pub fn current(&self) -> [F; 4] {
        self.current
    }
}

const MAGIC: &[u8; 8] = b"NSOPTB\0\x01";
const MEMORY_MAGIC: &[u8; 8] = b"NSOPTM\0\x01";

fn read_word(input: &mut impl Read) -> Result<usize, Error> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes)?;
    usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| Error::Circuit("saved word exceeds usize"))
}

fn read_field(input: &mut impl Read) -> Result<F, Error> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes)?;
    let word = u64::from_le_bytes(bytes);
    if word >= F::ORDER_U64 {
        return Err(Error::Circuit("saved field word is not canonical"));
    }
    Ok(F::from_u64(word))
}

/// The shape after the records: the blocks, each its first local and its
/// lanes (constant, term count, terms).
fn write_shape(output: &mut impl Write, shape: &NativeShape) -> Result<(), Error> {
    let mut word = |value: u64| output.write_all(&value.to_le_bytes());
    word(shape.blocks.len() as u64)?;
    for block in &shape.blocks {
        word(block.first_local as u64)?;
        for lane in &block.lanes {
            word(lane.constant.as_canonical_u64())?;
            word(lane.terms.len() as u64)?;
            for &(variable, coefficient) in &lane.terms {
                word(variable as u64)?;
                word(coefficient.as_canonical_u64())?;
            }
        }
    }
    Ok(())
}

fn read_shape(input: &mut impl Read) -> Result<NativeShape, Error> {
    let mut shape = NativeShape::default();
    for _ in 0..read_word(input)? {
        let first_local = read_word(input)?;
        let mut lanes: [Linear; 16] = Default::default();
        for lane in &mut lanes {
            lane.constant = read_field(input)?;
            for _ in 0..read_word(input)? {
                lane.terms.push((read_word(input)?, read_field(input)?));
            }
        }
        shape.blocks.push(NativeBlock { first_local, lanes });
    }
    Ok(shape)
}

/// The saved form of an application, after the file's magic (`Circuit::load`).
fn write_application(application: &ApplicationCircuit, output: &mut impl Write) -> Result<(), Error> {
    output.write_all(&(application.private_input_count() as u64).to_le_bytes())?;
    output.write_all(&application.prepared_output_forms()?)?;
    for row in application.prepared_output_rows() {
        output.write_all(&(row as u64).to_le_bytes())?;
    }
    let records = application.records();
    output.write_all(&(records.row_count() as u64).to_le_bytes())?;
    output.write_all(&(records.recipe_count() as u64).to_le_bytes())?;
    records.write_to(output)?;
    write_shape(output, application.shape())
}

fn encode_application(application: &ApplicationCircuit) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    write_application(application, &mut bytes)?;
    Ok(bytes)
}

/// The two Lean-emitted packages: the light and refresh step circuits
/// (`formal/nightstream-fprime/artifacts`), for the tests that load Lean's
/// own packages (`load_packages`).
#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) struct References<'a> {
    pub(crate) light: &'a [u8],
    pub(crate) refresh: &'a [u8],
}

/// The packages and their application: a Rust application in place of
/// Lean's (`compile`), or Lean's own (`load_packages`), whose witness program
/// fills its locals from the witness words and the output state.
pub struct Circuit {
    packages: BoundedDepthPackages,
    application: Option<ApplicationCircuit>,
    /// A memory relation: only `nebula::Circuit::verifier` verifies it,
    /// because this verifier checks no SPEC.md Algorithm 8 terminal
    /// condition.
    memory: bool,
}

impl Circuit {
    /// Put `application` in place of Lean's application in the embedded
    /// packages, whose identities are their pins (`artifacts`), each
    /// permutation as a block. The reference hash chain keeps Lean's
    /// packages: it computes the step of Lean's application region, and the
    /// Lean checks of Rust's files read those packages.
    pub fn compile(application: ApplicationCircuit) -> Result<Self, Error> {
        crate::application::check_blocks(&application)?;
        let embedded = crate::artifacts::embedded();
        let witness_words = application.private_input_count();
        let packages = if encode_application(&application)?
            == encode_application(&crate::application::poseidon2_hash_chain_v1()?)?
        {
            embedded.packages.clone()
        } else {
            embedded
                .packages
                .with_application(application.records().clone(), witness_words, application.shape())?
        };
        Ok(Self {
            packages,
            application: Some(application),
            memory: false,
        })
    }

    /// Check the pinned identity of Lean-emitted light and refresh packages
    /// whose application is Lean's (`expected`). Only the tests load Lean
    /// packages.
    #[cfg(test)]
    pub(crate) fn load_packages(references: References<'_>, expected: [u64; 4]) -> Result<Self, Error> {
        let packages = nightstream_fprime::load_bounded_depth_packages(references.light, references.refresh, expected)?;
        Ok(Self {
            packages,
            application: None,
            memory: false,
        })
    }

    /// This circuit as a memory relation (`nebula::Circuit`).
    pub(crate) fn into_memory(self) -> Self {
        Self { memory: true, ..self }
    }

    /// The Rust application, for tests that execute a loaded circuit.
    #[cfg(test)]
    pub(crate) fn application(&self) -> Option<&ApplicationCircuit> {
        self.application.as_ref()
    }

    /// The size of the Rust application, or `None` for Lean's.
    pub fn application_size(&self) -> Option<ApplicationSize> {
        self.application
            .as_ref()
            .map(|application| ApplicationSize {
                witness_words: application.private_input_count(),
                local_words: application.generated_range().len(),
                rows: application.row_count(),
            })
    }

    /// The packages' identity with the application.
    pub fn identity(&self) -> [u64; 4] {
        self.packages.identity()
    }

    /// Save the application for a prover in another process
    /// (`Prover::load`). The destination must not exist. A verifier compiles
    /// its own circuit and never loads this file.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let mut output = std::io::BufWriter::new(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)?,
        );
        output.write_all(if self.memory { MEMORY_MAGIC } else { MAGIC })?;
        let application = self
            .application
            .as_ref()
            .ok_or(ApplicationError::NoApplication)?;
        write_application(application, &mut output)?;
        output.flush()?;
        Ok(())
    }

    /// Read a saved application and compile it against the embedded packages
    /// again; their pinned identities are checked again.
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let mut input = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0; 8];
        input.read_exact(&mut magic)?;
        let memory = match &magic {
            magic if magic == MAGIC => false,
            magic if magic == MEMORY_MAGIC => true,
            _ => return Err(Error::Circuit("not a saved bounded-depth application")),
        };
        let private_count = read_word(&mut input)?;
        let mut forms = [0; 4];
        input.read_exact(&mut forms)?;
        let mut rows = [0; 4];
        for row in &mut rows {
            *row = read_word(&mut input)?;
        }
        let (row_count, recipe_count) = (read_word(&mut input)?, read_word(&mut input)?);
        let records = nightstream_fprime::ApplicationRecords::read_from(&mut input, row_count, recipe_count)?;
        let shape = read_shape(&mut input)?;
        let application = ApplicationCircuit::from_records(Arc::new(records), private_count, forms, rows, shape)?;
        let circuit = Self::compile(application)?;
        Ok(if memory { circuit.into_memory() } else { circuit })
    }

    /// Require one fold of the selected profile to meet the caller's positive
    /// local minimum. This is not an end-to-end bound over recursive depth or
    /// multiple attempts.
    pub fn prover(&self, engine: Engine, minimum_security_bits: u32) -> Result<Prover, Error> {
        self.prover_on(Backend::new(engine)?, minimum_security_bits)
    }

    fn prover_on(&self, backend: Backend, minimum_security_bits: u32) -> Result<Prover, Error> {
        Ok(Prover {
            lifecycle: lifecycle::BoundedDepth::new(self.packages.clone(), minimum_security_bits, backend)?,
            application: self.application.clone(),
        })
    }

    /// A verifier of this circuit; the caller selects it independently of any
    /// proof. Reject a profile whose one-fold security level is below the
    /// caller's positive local minimum, and a memory relation, whose verifier
    /// is `nebula::Circuit::verifier`. The caller separately budgets recursive
    /// depth, queries, retries, and concurrent attempts.
    pub fn verifier(&self, engine: Engine, minimum_security_bits: u32) -> Result<Verifier, Error> {
        if self.memory {
            return Err(Error::Circuit("a memory relation needs nebula::Circuit::verifier"));
        }
        self.stage1_verifier(engine, minimum_security_bits)
    }

    /// The step verifier inside `nebula::Verifier`.
    pub(crate) fn stage1_verifier(&self, engine: Engine, minimum_security_bits: u32) -> Result<Verifier, Error> {
        Ok(Verifier {
            lifecycle: lifecycle::BoundedDepth::new(
                self.packages.clone(),
                minimum_security_bits,
                Backend::new(engine)?,
            )?,
        })
    }
}

/// A proof of some steps from an initial state, with its witnesses: it can
/// be extended.
#[derive(Clone)]
pub struct Proof(lifecycle::Envelope);

impl Proof {
    pub fn state(&self) -> State {
        self.0.state()
    }
}

/// A finished proof: the statement and an argument that some proof with
/// witnesses verifies for it (SPEC.md section 6). It carries no witness and
/// cannot be extended.
#[derive(Clone)]
pub struct FinalProof(lifecycle::FinalProof);

impl FinalProof {
    pub fn state(&self) -> State {
        self.0.state()
    }

    /// The proof's strict bytes, for `FinalProof::from_bytes`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        Ok(self.0.to_bytes()?)
    }

    /// Decode untrusted bytes. A proof has one encoding: each field word is
    /// below the modulus, and no byte follows the proof. Acceptance still
    /// needs `Verifier::verify_final`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        Ok(Self(lifecycle::FinalProof::from_bytes(bytes)?))
    }
}

/// The prover's key of the compression: the rows of the Ajtai key that the
/// circuit reads and their commitment. It holds about 10 GB for the
/// Poseidon2 hash chain. `Prover::compression_setup` makes it once from the
/// prover's circuit, and each `Prover::compress` reads it. It gives the
/// verifier nothing: a verifier takes its own `CompressionKey`.
pub struct CompressionSetup(neo_compress::ProverKey);

/// The verifier's key of the compression: the commitment of the rows of the
/// Ajtai key, in place of the rows (SPEC.md section 6). It is a function of
/// the public seeds and of the size of the verifier's circuit. No proof
/// carries it.
///
/// The key of each size is pinned in this library, and a test computes each
/// pinned key again from the public seeds. `Verifier::compression_key` gives
/// the pinned key of the verifier's circuit and expands no key element.
/// `Verifier::decode_compression_key` accepts exactly the bytes of that
/// pinned key, so a decoded key is checked against the pin and is not
/// trusted input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressionKey(neo_compress::VerificationKey);

impl CompressionKey {
    /// The key's strict bytes, for `Verifier::decode_compression_key`.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.to_bytes()
    }
}

pub struct Prover {
    lifecycle: lifecycle::BoundedDepth,
    application: Option<ApplicationCircuit>,
}

impl Prover {
    /// Load the prover of a circuit that `Circuit::write` saved. A verifier
    /// compiles its own circuit and never loads this file. An unavailable
    /// engine fails before the circuit loads.
    /// `minimum_security_bits` gates one fold, not the cumulative error of a
    /// recursive proof or multiple attempts.
    pub fn load(path: impl AsRef<Path>, engine: Engine, minimum_security_bits: u32) -> Result<Self, Error> {
        let backend = Backend::new(engine)?;
        Circuit::load(path)?.prover_on(backend, minimum_security_bits)
    }

    pub fn prove(&self, initial_state: [F; 4], private_inputs: &[F]) -> Result<Proof, Error> {
        self.extend(&Proof(self.lifecycle.initial(initial_state)?), private_inputs)
    }

    /// The next proof; `proof` does not change. The Rust application
    /// computes the output state.
    pub fn extend(&self, proof: &Proof, private_inputs: &[F]) -> Result<Proof, Error> {
        let application = self
            .application
            .as_ref()
            .ok_or(ApplicationError::NoApplication)?;
        // The application runs on the CPU while the engine folds the last
        // step; neither needs the other's result.
        let (witness, folded) = std::thread::scope(|scope| {
            let run = scope.spawn(|| application.execute(proof.state().current(), private_inputs));
            let folded = self.lifecycle.fold(&proof.0);
            (
                run.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
                folded,
            )
        });
        let witness = witness?;
        Ok(Proof(self.lifecycle.step(
            &proof.0,
            folded?,
            private_inputs,
            witness.output_state(),
            Some(witness.locals()),
        )?))
    }

    /// The proof of no step from `z0`, which a memory run extends.
    pub(crate) fn initial(&self, z0: [F; 4]) -> Result<Proof, Error> {
        Ok(Proof(self.lifecycle.initial(z0)?))
    }

    /// The first proof with the output state that the caller supplies.
    #[cfg(test)]
    pub(crate) fn prove_with_output(
        &self,
        initial_state: [F; 4],
        private_inputs: &[F],
        output: [F; 4],
    ) -> Result<Proof, Error> {
        self.extend_with_output(&Proof(self.lifecycle.initial(initial_state)?), private_inputs, output)
    }

    /// The next proof with the output state that the caller supplies; the
    /// application's rows check it.
    pub(crate) fn extend_with_output(
        &self,
        proof: &Proof,
        private_inputs: &[F],
        output: [F; 4],
    ) -> Result<Proof, Error> {
        Ok(Proof(
            self.lifecycle
                .extend(&proof.0, private_inputs, output, None)?,
        ))
    }

    /// The next proof of a memory invocation whose native output state is
    /// `output`. A Rust application executes and must reach `output`; Lean's
    /// application takes it (`extend_with_output`).
    pub(crate) fn extend_invocation(
        &self,
        proof: &Proof,
        private_inputs: &[F],
        output: [F; 4],
    ) -> Result<Proof, Error> {
        if self.application.is_none() {
            return self.extend_with_output(proof, private_inputs, output);
        }
        let next = self.extend(proof, private_inputs)?;
        if next.state().current() != output {
            return Err(Error::Circuit("the application's output differs from the native step"));
        }
        Ok(next)
    }

    /// The proof's strict bytes, for `Verifier::decode_proof`.
    pub fn encode_proof(&self, proof: &Proof) -> Result<Vec<u8>, Error> {
        Ok(self.lifecycle.encode_proof(&proof.0)?)
    }

    /// Expand the rows of the Ajtai key that the prover's circuit reads and
    /// commit them, once for each prover. A verifier of the same circuit
    /// takes the key of these proofs itself (`Verifier::compression_key`).
    pub fn compression_setup(&self) -> Result<CompressionSetup, Error> {
        Ok(CompressionSetup(self.lifecycle.compression_setup()?))
    }

    /// Compress `proof`, which does not change, for `Verifier::verify_final`
    /// with the key of `setup`. The result is not checked here: it verifies
    /// when `proof` does.
    pub fn compress(&self, proof: &Proof, setup: &CompressionSetup) -> Result<FinalProof, Error> {
        Ok(FinalProof(self.lifecycle.compress(&setup.0, &proof.0)?.0))
    }
}

pub struct Verifier {
    lifecycle: lifecycle::BoundedDepth,
}

impl Verifier {
    /// Decode untrusted proof bytes; the relations fix the exact length,
    /// which is checked before allocation. Acceptance still needs `verify`.
    pub fn decode_proof(&self, bytes: &[u8]) -> Result<Proof, Error> {
        Ok(Proof(self.lifecycle.decode_proof(bytes)?))
    }

    /// Check `proof` and its witnesses against the expected state
    /// (`Terminal.HoldsFor`).
    pub fn verify(&self, expected: &State, proof: &Proof) -> Result<(), Error> {
        Ok(self
            .lifecycle
            .verify(expected.iteration(), expected.z0(), expected.current(), &proof.0)?)
    }

    /// The compression key of this verifier's circuit: the pinned key of
    /// its size (`CompressionKey`). A prover of the same circuit makes
    /// proofs for this key (`Prover::compression_setup`). Make the key once
    /// and use it for each proof: it keeps the Akita verifier setup that its
    /// first proof makes.
    pub fn compression_key(&self) -> Result<CompressionKey, Error> {
        Ok(CompressionKey(self.lifecycle.compression_key()?))
    }

    /// Decode the bytes of the compression key of this verifier's circuit.
    /// Only the bytes of the pinned key of the circuit's size decode
    /// (`CompressionKey`).
    pub fn decode_compression_key(&self, bytes: &[u8]) -> Result<CompressionKey, Error> {
        Ok(CompressionKey(self.lifecycle.decode_compression_key(bytes)?))
    }

    /// Check a finished proof against the expected state under the key of
    /// this verifier (`compression_key`). The verifier derives the relations
    /// from its circuit and reads no row of the Ajtai key. Nothing that the
    /// proof carries is authority, and the key is no part of the proof.
    pub fn verify_final(&self, expected: &State, key: &CompressionKey, proof: &FinalProof) -> Result<(), Error> {
        self.lifecycle.verify_final(
            &key.0,
            expected.iteration(),
            expected.z0(),
            expected.current(),
            &proof.0,
        )?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/bounded_depth_lifecycle/native_field_census.rs"]
mod native_field_census;
