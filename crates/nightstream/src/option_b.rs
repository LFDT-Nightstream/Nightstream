//! The bounded-depth prover and verifier (option B): a light and a refresh
//! step circuit that Lean emits, with the Rust application in place of
//! Lean's, two slots, and a final proof that the shrink layer compresses. It
//! replaces the one-circuit `Circuit`, `Prover` and `Verifier` (CW-8).
//!
//! The caller selects the packages and the application; the verifier derives
//! everything else from them. Nothing a proof carries is authority.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use neo_math::F;
use nightstream_fprime::{
    load_option_b_finish, load_option_b_packages, OptionBPackages, StepPackage, OPTION_B_FINISH_IDENTITY,
    OPTION_B_PACKAGES_IDENTITY,
};

use crate::application::{ApplicationCircuit, ApplicationError};
use crate::engine::{Engine, EngineError};
use crate::lifecycle::option_b::{self as lifecycle, Finish};
use crate::{Error, State};

const MAGIC: &[u8; 8] = b"NSOPTB\0\x01";
const MEMORY_MAGIC: &[u8; 8] = b"NSOPTM\0\x01";

fn read_word(input: &mut impl Read) -> Result<usize, Error> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes)?;
    usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| Error::Circuit("saved word exceeds usize"))
}

fn cpu_only(engine: Engine) -> Result<(), Error> {
    if engine != Engine::Optimized {
        return Err(EngineError::Unavailable {
            engine,
            reason: "option B proves and verifies on the CPU path",
        }
        .into());
    }
    Ok(())
}

/// The three Lean-emitted packages: the light and refresh step circuits and
/// the finish circuit (`formal/nightstream-fprime/artifacts`).
#[derive(Clone, Copy)]
pub struct References<'a> {
    pub light: &'a [u8],
    pub refresh: &'a [u8],
    pub finish: &'a [u8],
}

/// The packages and their application: a Rust application in place of
/// Lean's (`compile`), or Lean's own (`load_packages`), whose witness program
/// fills its locals from the witness words and the output state.
pub struct Circuit {
    packages: OptionBPackages,
    finish: StepPackage,
    application: Option<ApplicationCircuit>,
    /// A memory relation: only `nebula::Verifier` verifies it, because this
    /// verifier checks no SPEC.md Algorithm 10 terminal condition.
    memory: bool,
}

impl Circuit {
    /// Check the packages' pinned identities, then put `application` in place
    /// of Lean's application.
    pub fn compile(references: References<'_>, application: ApplicationCircuit) -> Result<Self, Error> {
        let witness_words = application.private_input_count();
        let packages = load_option_b_packages(references.light, references.refresh, OPTION_B_PACKAGES_IDENTITY)?
            .with_application(application.records().clone(), witness_words)?;
        let finish =
            load_option_b_finish(references.finish, OPTION_B_FINISH_IDENTITY)?.with_witness_words(witness_words)?;
        Ok(Self {
            packages,
            finish,
            application: Some(application),
            memory: false,
        })
    }

    /// Check the pinned identity of Lean-emitted light and refresh packages
    /// whose application is Lean's (`expected`), and of the finish package,
    /// which moves to their witness words.
    pub fn load_packages(references: References<'_>, expected: [u64; 4]) -> Result<Self, Error> {
        let packages = load_option_b_packages(references.light, references.refresh, expected)?;
        let finish = load_option_b_finish(references.finish, OPTION_B_FINISH_IDENTITY)?
            .with_witness_words(packages.light().witness_words())?;
        Ok(Self {
            packages,
            finish,
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
    pub fn application_size(&self) -> Option<crate::ApplicationSize> {
        self.application
            .as_ref()
            .map(|application| crate::ApplicationSize {
                witness_words: application.private_input_count(),
                local_words: application.generated_range().len(),
                rows: application.row_count(),
            })
    }

    /// The packages' identity with the application.
    pub fn identity(&self) -> [u64; 4] {
        self.packages.identity()
    }

    /// Save the application for a prover in another process; the packages
    /// stay the caller's (`load`). The destination must not exist.
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
        output.write_all(&(application.private_input_count() as u64).to_le_bytes())?;
        output.write_all(&application.prepared_output_forms()?)?;
        for row in application.prepared_output_rows() {
            output.write_all(&(row as u64).to_le_bytes())?;
        }
        let records = application.records();
        output.write_all(&(records.row_count() as u64).to_le_bytes())?;
        output.write_all(&(records.recipe_count() as u64).to_le_bytes())?;
        records.write_to(&mut output)?;
        output.flush()?;
        Ok(())
    }

    /// Read a saved application and compile it against the caller's
    /// packages again; their pinned identities are checked again.
    pub fn load(path: impl AsRef<Path>, references: References<'_>) -> Result<Self, Error> {
        let mut input = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0; 8];
        input.read_exact(&mut magic)?;
        let memory = match &magic {
            magic if magic == MAGIC => false,
            magic if magic == MEMORY_MAGIC => true,
            _ => return Err(Error::Circuit("not a saved option-B application")),
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
        let application = ApplicationCircuit::from_records(Arc::new(records), private_count, forms, rows)?;
        let circuit = Self::compile(references, application)?;
        Ok(if memory { circuit.into_memory() } else { circuit })
    }

    /// Require the profile to meet the caller's positive statistical minimum.
    /// Option B proves and verifies on the CPU path (`Engine::Optimized`).
    pub fn prover(&self, engine: Engine, minimum_security_bits: u32) -> Result<Prover, Error> {
        cpu_only(engine)?;
        Ok(Prover {
            lifecycle: lifecycle::OptionB::new(self.packages.clone(), minimum_security_bits)?,
            finish: Finish::new(self.finish.clone())?,
            application: self.application.clone(),
        })
    }

    /// A verifier of this circuit; the caller selects it independently of any
    /// proof.
    pub fn verifier(&self, engine: Engine, minimum_security_bits: u32) -> Result<Verifier, Error> {
        if self.memory {
            return Err(Error::Circuit("a memory relation needs nebula::Verifier"));
        }
        self.stage1_verifier(engine, minimum_security_bits)
    }

    /// The Stage 1 verifier inside `nebula::Verifier`.
    pub(crate) fn stage1_verifier(&self, engine: Engine, minimum_security_bits: u32) -> Result<Verifier, Error> {
        cpu_only(engine)?;
        Ok(Verifier {
            lifecycle: lifecycle::OptionB::new(self.packages.clone(), minimum_security_bits)?,
            finish: Finish::new(self.finish.clone())?,
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

/// A finished proof: the statement and the shrink proof. It carries no
/// witness and cannot be extended.
#[derive(Clone)]
pub struct FinalProof(lifecycle::FinalProof);

/// The verifier's compression constants, derived from the relations.
#[derive(Clone)]
pub struct CompressionKey(lifecycle::Keys);

/// The prover's layer-1 setup files and their key.
pub struct CompressionSetup(lifecycle::Setups);

pub struct Prover {
    lifecycle: lifecycle::OptionB,
    finish: Finish,
    application: Option<ApplicationCircuit>,
}

impl Prover {
    pub fn prove(&self, initial_state: [F; 4], private_inputs: &[F]) -> Result<Proof, Error> {
        self.extend(&Proof(self.lifecycle.initial(initial_state)?), private_inputs)
    }

    /// The next proof; `proof` does not change. The Rust application
    /// computes the output state.
    pub fn extend(&self, proof: &Proof, private_inputs: &[F]) -> Result<Proof, Error> {
        let next = self
            .application
            .as_ref()
            .ok_or(ApplicationError::NoApplication)?
            .execute(proof.state().current(), private_inputs)?
            .output_state();
        self.extend_with_output(proof, private_inputs, next)
    }

    /// The first proof with the output state that the caller supplies.
    pub fn prove_with_output(
        &self,
        initial_state: [F; 4],
        private_inputs: &[F],
        output: [F; 4],
    ) -> Result<Proof, Error> {
        self.extend_with_output(&Proof(self.lifecycle.initial(initial_state)?), private_inputs, output)
    }

    /// The next proof with the output state that the caller supplies; the
    /// application's rows check it.
    pub fn extend_with_output(&self, proof: &Proof, private_inputs: &[F], output: [F; 4]) -> Result<Proof, Error> {
        Ok(Proof(self.lifecycle.extend(&proof.0, private_inputs, output)?))
    }

    /// The proof's strict bytes, for `Verifier::decode_proof`.
    pub fn encode_proof(&self, proof: &Proof) -> Result<Vec<u8>, Error> {
        Ok(self.lifecycle.encode_proof(&proof.0)?)
    }

    /// Write both layer-1 setups into the existing directory `dir`.
    pub fn compression_setup(&self, dir: impl AsRef<Path>) -> Result<CompressionSetup, Error> {
        Ok(CompressionSetup(self.lifecycle.compression_setups(dir.as_ref())?))
    }

    /// Compress `proof`; `proof` does not change.
    pub fn finish(&self, proof: &Proof, setup: &CompressionSetup) -> Result<FinalProof, Error> {
        let witness = self
            .lifecycle
            .final_witness(&proof.0, &setup.0, &self.finish)?;
        let state = proof.state();
        Ok(FinalProof(self.lifecycle.prove_final(
            setup.0.keys(),
            &self.finish,
            (state.iteration(), state.z0(), state.current()),
            &witness,
        )?))
    }
}

pub struct Verifier {
    lifecycle: lifecycle::OptionB,
    finish: Finish,
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

    /// Derive the compression key from the relations, without files. This
    /// key, or a copy you pinned, is the authority for `verify_final`.
    pub fn compression_key(&self) -> Result<CompressionKey, Error> {
        Ok(CompressionKey(self.lifecycle.compression_keys()?))
    }

    /// Check a finished proof against the expected state and a trusted key.
    pub fn verify_final(&self, expected: &State, key: &CompressionKey, proof: &FinalProof) -> Result<(), Error> {
        Ok(self.lifecycle.verify_final(
            &key.0,
            &self.finish,
            expected.iteration(),
            expected.z0(),
            expected.current(),
            &proof.0,
        )?)
    }
}
