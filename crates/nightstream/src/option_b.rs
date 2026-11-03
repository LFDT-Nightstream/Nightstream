//! The bounded-depth prover and verifier (option B): a light and a refresh
//! step circuit that Lean emits, with the Rust application in place of
//! Lean's, two slots, and a final proof that the shrink layer compresses. It
//! replaces the one-circuit `Circuit`, `Prover` and `Verifier` (CW-8).
//!
//! The caller selects the packages and the application; the verifier derives
//! everything else from them. Nothing a proof carries is authority.

use std::path::Path;

use neo_math::F;
use nightstream_fprime::{
    load_option_b_finish, load_option_b_packages, OptionBPackages, StepPackage, OPTION_B_FINISH_IDENTITY,
    OPTION_B_PACKAGES_IDENTITY,
};

use crate::application::ApplicationCircuit;
use crate::lifecycle::option_b::{self as lifecycle, Finish};
use crate::{Error, State};

/// The three Lean-emitted packages: the light and refresh step circuits and
/// the finish circuit (`formal/nightstream-fprime/artifacts`).
#[derive(Clone, Copy)]
pub struct References<'a> {
    pub light: &'a [u8],
    pub refresh: &'a [u8],
    pub finish: &'a [u8],
}

/// The packages with a Rust application in place of Lean's.
pub struct Circuit {
    packages: OptionBPackages,
    finish: StepPackage,
    application: ApplicationCircuit,
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
            application,
        })
    }

    /// The packages' identity with the application.
    pub fn identity(&self) -> [u64; 4] {
        self.packages.identity()
    }

    /// Require the profile to meet the caller's positive statistical minimum.
    pub fn prover(&self, minimum_security_bits: u32) -> Result<Prover, Error> {
        Ok(Prover {
            lifecycle: lifecycle::OptionB::new(self.packages.clone(), minimum_security_bits)?,
            finish: Finish::new(self.finish.clone())?,
            application: self.application.clone(),
        })
    }

    /// A verifier of this circuit; the caller selects it independently of any
    /// proof.
    pub fn verifier(&self, minimum_security_bits: u32) -> Result<Verifier, Error> {
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
    application: ApplicationCircuit,
}

impl Prover {
    pub fn prove(&self, initial_state: [F; 4], private_inputs: &[F]) -> Result<Proof, Error> {
        self.extend(&Proof(self.lifecycle.initial(initial_state)?), private_inputs)
    }

    /// The next proof; `proof` does not change.
    pub fn extend(&self, proof: &Proof, private_inputs: &[F]) -> Result<Proof, Error> {
        let next = self
            .application
            .execute(proof.state().current(), private_inputs)?
            .output_state();
        Ok(Proof(self.lifecycle.extend(&proof.0, private_inputs, next)?))
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
