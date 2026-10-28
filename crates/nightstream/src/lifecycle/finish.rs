//! Terminal compression. Layer 0 runs PiCCS and PiRLC once more on the final
//! running and fresh claims and stops before PiDEC (paper Lemma 1: a reduction
//! of knowledge to one CE(B) claim). Layer 1 proves that claim with the
//! `neo-spartan` argument of knowledge. The shrink layer then proves that the
//! final program (`terminal/`) accepts both, so the final proof is small.
//!
//! Owns the meaning of `FinalProof`, `CompressionKey` and `CompressionSetup`
//! and their flows. Does not own the reductions, the layer-1 protocol, the
//! shrink argument or the final program. Nothing carried in a proof is
//! authority: the key is, and the caller must trust it.

use neo_math::{D, F};
use neo_spartan::shrink::{self, Shape, Shrink, ShrinkProof};
use neo_spartan::Relation;
use nightstream_fprime::{
    PackageError, PiCcsV1_2EncodedInputs, PiCcsV1_2PackageInputs, PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS,
    PI_CCS_V1_2_ROUND_COUNT, PI_CCS_V1_2_SOURCE_COUNT,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};

use super::terminal::{self, FinalProgram, Package, TerminalPackageError};
use super::{
    encode_pi_ccs_v1_2_public_input, pi_ccs_v1_2_prior_children, PiCcsV1_2ProofInputs, PreparedLifecycle,
    ProofCodecError, ProveError, Stage1Envelope, Stage1State, StepInputError, VerifyError,
};
use crate::folding::{self as nifs, pi_ccs, pi_rlc, transcript::Transcript, CcsClaim, RunningInstance};

/// The terminal proof the shrink layer proves: the inputs of the terminal
/// package and the layer-1 argument for one state. It never leaves the
/// prover.
pub(crate) struct TerminalProof {
    pub(crate) inputs: PiCcsV1_2EncodedInputs,
    pub(crate) layer1: neo_spartan::Proof,
}

/// A finished proof: the state and a shrink proof that the final program
/// accepts it. It carries no witness and cannot be extended.
#[derive(Clone)]
pub struct FinalProof {
    state: Stage1State,
    shrink: ShrinkProof,
}

const FINAL_MAGIC: &[u8; 16] = b"NS-FINAL-PROOF02";
const STATE_BYTES: usize = 9 * 8;

impl FinalProof {
    pub fn state(&self) -> &Stage1State {
        &self.state
    }

    /// The tag, the iteration, `z0` and the current state as little-endian
    /// words, then the shrink proof.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = FINAL_MAGIC.to_vec();
        bytes.extend_from_slice(&self.state.iteration().to_le_bytes());
        for word in self.state.z0().iter().chain(&self.state.current()) {
            bytes.extend_from_slice(&word.as_canonical_u64().to_le_bytes());
        }
        bytes.extend(self.shrink.to_bytes());
        bytes
    }

    /// Strict: the bytes must be the canonical encoding of the decoded proof.
    /// Acceptance still needs `Verifier::verify_final`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProofCodecError> {
        let header = FINAL_MAGIC.len() + STATE_BYTES;
        if bytes.len() < header || &bytes[..FINAL_MAGIC.len()] != FINAL_MAGIC {
            return Err(ProofCodecError("final proof tag"));
        }
        let mut words = bytes[FINAL_MAGIC.len()..header]
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight bytes")));
        let iteration = words.next().expect("nine words");
        let mut field = || -> Result<F, ProofCodecError> {
            let word = words.next().expect("nine words");
            (word < F::ORDER_U64)
                .then(|| F::from_u64(word))
                .ok_or(ProofCodecError("non-canonical state word"))
        };
        let z0 = [field()?, field()?, field()?, field()?];
        let current = [field()?, field()?, field()?, field()?];
        let shrink = ShrinkProof::from_bytes(&bytes[header..]).map_err(|_| ProofCodecError("shrink proof"))?;
        Ok(Self {
            state: Stage1State::new(iteration, z0, current),
            shrink,
        })
    }
}

/// The verifier's trusted constants of one circuit's compression: the
/// layer-1 key, the constants the final program replays, and the shrink
/// shape. `Verifier::verify_final` checks it against the circuit before it
/// checks a proof with it. Authority: derived from the circuit
/// (`Verifier::compression_key`) or a pinned copy. Never take a key from a
/// prover.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompressionKey {
    layer1: neo_spartan::Key,
    /// The circuit's verifier context digest.
    context: [u64; 4],
    /// PiRLC's guard bound `sources · T · (b - 1)`, which every honest
    /// parent obeys.
    bound: u32,
    /// `-log2` of the error each compression layer (layer 1, shrink) may add.
    layer_bits: f64,
    shape: Shape,
}

impl CompressionKey {
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("an in-memory key always encodes")
    }

    /// Strict: the bytes must be the canonical encoding of a well-formed
    /// key.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProofCodecError> {
        let key: Self = bincode::deserialize(bytes).map_err(|_| ProofCodecError("compression key"))?;
        if key.to_bytes() != bytes {
            return Err(ProofCodecError("non-canonical compression key"));
        }
        key.layer1
            .check()
            .map_err(|_| ProofCodecError("compression key structure"))?;
        Ok(key)
    }

    /// Check `proof` for `expected_state` with this key alone.
    /// `Verifier::verify_final` checks the key first, then calls this.
    pub(crate) fn verify(&self, expected_state: &Stage1State, proof: &FinalProof) -> Result<(), VerifyError> {
        if proof.state != *expected_state {
            return Err(VerifyError::Statement("final proof differs from the external state"));
        }
        if expected_state.iteration() == 0 || expected_state.iteration() >= F::ORDER_U64 {
            return Err(VerifyError::Statement(
                "an active proof requires a positive canonical iteration",
            ));
        }
        let relation = self.relation().map_err(VerifyError::Final)?;
        let package = terminal::package()?;
        let program = self.program(&relation, package, expected_state, None);
        let shrink = Shrink::new(self.shape, self.layer_bits).map_err(VerifyError::Final)?;
        shrink::verify(&shrink, &program, &proof.shrink).map_err(VerifyError::Final)
    }

    /// The layer-1 relation of the parent claim.
    pub(crate) fn relation(&self) -> Result<Relation<'_>, neo_spartan::Error> {
        Relation::new(
            &self.layer1,
            PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS / D,
            PI_CCS_V1_2_ROUND_COUNT,
            self.bound,
            self.layer_bits,
        )
    }

    pub(crate) fn program<'a>(
        &'a self,
        relation: &'a Relation<'a>,
        package: &'a Package,
        state: &'a Stage1State,
        proof: Option<&'a TerminalProof>,
    ) -> FinalProgram<'a> {
        FinalProgram {
            context: self.context,
            package,
            relation,
            state,
            proof,
        }
    }
}

/// `exp2` and `log2` come from the platform's libm, so one security
/// requirement can differ in its last bits between two platforms (about
/// 1e-14 bits here), and a key pinned on one must pass on the other. 1e-9
/// bits is far above that noise and raises the error bound by a factor below
/// 1 + 1e-9.
const SECURITY_SLACK_BITS: f64 = 1e-9;

/// The prover's compression setup: the setup files and their key.
pub struct CompressionSetup {
    layer1: neo_spartan::Setup,
    key: CompressionKey,
}

impl CompressionSetup {
    pub fn key(&self) -> &CompressionKey {
        &self.key
    }
}

/// A compression call failed: the setup, the key or a finish.
#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    #[error("compression input: {0}")]
    Input(&'static str),
    #[error("compression key: {0}")]
    Key(&'static str),
    #[error(transparent)]
    StepInputs(#[from] StepInputError),
    #[error("prior running claims are not a canonical PiDEC child family: {0}")]
    PriorFamily(#[source] nifs::Error),
    #[error(transparent)]
    Prove(#[from] ProveError),
    #[error(transparent)]
    Package(#[from] PackageError),
    #[error(transparent)]
    TerminalPackage(#[from] TerminalPackageError),
    #[error(transparent)]
    Fold(#[from] nifs::Error),
    #[error(transparent)]
    Layer1(#[from] neo_spartan::Error),
}

/// The last fold of an active proof, stopped before Π_DEC.
pub(crate) struct LastFold {
    pub(crate) state: Stage1State,
    /// The running claims as the state hash reads them, without witnesses.
    pub(crate) running: RunningInstance,
    pub(crate) fresh: CcsClaim,
    pub(crate) pi_ccs: pi_ccs::Proof,
    pub(crate) parent: pi_rlc::Output,
    /// The fold transcript after PiRLC.
    pub(crate) transcript: Transcript,
}

impl PreparedLifecycle {
    /// Prove the final accumulator of `envelope` without its witnesses.
    /// `envelope` does not change and can still be extended.
    pub(crate) fn finish_with_spartan(
        &self,
        envelope: &Stage1Envelope,
        setup: &CompressionSetup,
    ) -> Result<FinalProof, CompressionError> {
        self.check_compression_key(&setup.key)
            .map_err(CompressionError::Key)?;
        // The package loads while the last fold and layer 1 run.
        terminal::preload();
        let last = self.last_fold(envelope)?;
        // Layer 1 and the shrink make no Ajtai commitment, so a kept key
        // would only add to their peak.
        self.backend
            .release_commitment_key_rows()
            .map_err(ProveError::from)?;
        let terminal_inputs = self.terminal_inputs(&last)?;
        let key = &setup.key;
        let relation = key.relation()?;
        let accelerator = self.backend.accelerator();
        let layer1 = neo_spartan::prove(
            &relation,
            &setup.layer1,
            &accelerator,
            last.transcript.into_inner(),
            &last.parent.claim,
            &last.parent.witness,
        )
        .map_err(|error| self.backend.accelerator_error(error))?;
        let package = terminal::package()?;
        let inputs = package.loaded().encode_terminal_inputs(&terminal_inputs)?;
        let terminal = TerminalProof { inputs, layer1 };
        let program = key.program(&relation, package, &last.state, Some(&terminal));
        let layer = Shrink::new(key.shape, key.layer_bits)?;
        let shrink =
            shrink::prove(&layer, &program, &accelerator).map_err(|error| self.backend.accelerator_error(error))?;
        // A device fault fails the call here, instead of a returned proof
        // that does not verify.
        shrink::verify(&layer, &key.program(&relation, package, &last.state, None), &shrink)?;
        Ok(FinalProof {
            state: last.state,
            shrink,
        })
    }

    /// Run PiCCS and PiRLC on the running and fresh claims of `envelope`.
    pub(crate) fn last_fold(&self, envelope: &Stage1Envelope) -> Result<LastFold, CompressionError> {
        let state = *envelope.state();
        let Some((running, fresh)) = envelope.active_parts() else {
            return Err(CompressionError::Input("an initial proof has no fold to finish"));
        };
        let (running, fresh) = (running.clone(), fresh.clone());
        let (bound, _, _) = self.checked_prior_state(&state, &running, &fresh.claim)?;
        nifs::validate_running_children(&self.params, &self.structure, bound).map_err(CompressionError::PriorFamily)?;
        self.validate_prover_sources(&fresh, &running)?;
        // The state hash reads only the claims; the witnesses go into the fold.
        let semantic = running.claims_only();
        let fresh_claim = fresh.claim.clone();

        let rows = self.matrix_rows();
        let context = self.fold_context(&rows).map_err(ProveError::from)?;
        let mut transcript = Transcript::session();
        let (pi_ccs, parent, _) = self
            .backend
            .prove_parent(&mut transcript, context, fresh, running)?;
        Ok(LastFold {
            state,
            running: semantic,
            fresh: fresh_claim,
            pi_ccs,
            parent,
            transcript,
        })
    }

    /// The caller inputs of the Lean terminal package for `last`. The package
    /// has no output state: the prior preimage fills its output-preimage
    /// segment, so the canonical and context rows of PiCCS hold for it too.
    pub(crate) fn terminal_inputs(&self, last: &LastFold) -> Result<PiCcsV1_2PackageInputs, StepInputError> {
        let (_, prior_preimage, prior_digest) = self.checked_prior_state(&last.state, &last.running, &last.fresh)?;
        let prior_children = pi_ccs_v1_2_prior_children(&last.running.claims)?;
        let prior_public_input = encode_pi_ccs_v1_2_public_input(prior_digest)?;
        Ok(
            PiCcsV1_2ProofInputs::from_proof(&last.fresh, &last.pi_ccs)?.into_package_inputs(
                prior_preimage.clone(),
                prior_preimage,
                prior_children,
                prior_public_input,
                [0; 4],
                self.binding.verifier_context().clone(),
            )?,
        )
    }

    /// The cheap part of a compression key, checked before any setup work:
    /// the security budget and the terminal package against this circuit.
    fn compression_constants(&self) -> Result<(f64, &'static Package), CompressionError> {
        let layer_bits = self.compression_layer_bits().ok_or(CompressionError::Key(
            "the fold error leaves no room under the security minimum",
        ))?;
        let package = terminal::package()?;
        if !package.checks(self.structure.f()) {
            return Err(CompressionError::Key(
                "the terminal package checks another CCS polynomial",
            ));
        }
        Ok((layer_bits, package))
    }

    /// The key of this circuit for the layer-1 key `layer1`: the final
    /// program's constants and the shape of one run of it.
    fn compression_key_for(
        &self,
        layer1: neo_spartan::Key,
        layer_bits: f64,
        package: &Package,
    ) -> Result<CompressionKey, CompressionError> {
        let mut key = CompressionKey {
            layer1,
            context: self.binding.verifier_context().digest(),
            bound: self.compression_bound(),
            layer_bits,
            shape: Shape::default(),
        };
        let shape = {
            let relation = key.relation()?;
            let state = Stage1State::new(1, [F::ZERO; 4], [F::ZERO; 4]);
            Shape::derive(&key.program(&relation, package, &state, None))?
        };
        key.shape = shape;
        Ok(key)
    }

    /// Write the compression setup files of this circuit into `dir`.
    pub(crate) fn compression_setup(&self, dir: &std::path::Path) -> Result<CompressionSetup, CompressionError> {
        let (layer_bits, package) = self.compression_constants()?;
        let layer1 = neo_spartan::Setup::build(&self.matrix_rows(), dir)?;
        let key = self.compression_key_for(layer1.key().clone(), layer_bits, package)?;
        Ok(CompressionSetup { layer1, key })
    }

    /// Reopen setup files written for `key`, which must be this circuit's.
    pub(crate) fn open_compression_setup(
        &self,
        dir: &std::path::Path,
        key: &CompressionKey,
    ) -> Result<CompressionSetup, CompressionError> {
        self.check_compression_key(key)
            .map_err(CompressionError::Key)?;
        let layer1 = neo_spartan::Setup::open(&self.matrix_rows(), dir, &key.layer1)?;
        Ok(CompressionSetup {
            layer1,
            key: key.clone(),
        })
    }

    /// Derive the compression key from the package, without files.
    pub(crate) fn compression_key(&self) -> Result<CompressionKey, CompressionError> {
        let (layer_bits, package) = self.compression_constants()?;
        let layer1 = neo_spartan::Key::derive(&self.matrix_rows())?;
        self.compression_key_for(layer1, layer_bits, package)
    }

    /// Refuse a compression key of another circuit, guard bound or matrix
    /// size, or one below this circuit's security minimum. The key's setup
    /// root is the caller's trust: a key it derived (`compression_key`) or
    /// pinned.
    pub(crate) fn check_compression_key(&self, key: &CompressionKey) -> Result<(), &'static str> {
        if key.context != self.binding.verifier_context().digest()
            || key.bound != self.compression_bound()
            || !key.layer1.fits(&self.matrix_rows())
        {
            return Err("the compression key belongs to another circuit");
        }
        if self
            .compression_layer_bits()
            .is_none_or(|bits| key.layer_bits < bits - SECURITY_SLACK_BITS)
        {
            return Err("the compression key is below this circuit's security minimum");
        }
        Ok(())
    }

    /// PiRLC's guard bound `sources · T · (b - 1)`.
    fn compression_bound(&self) -> u32 {
        PI_CCS_V1_2_SOURCE_COUNT as u32 * self.params.inner().T() * (self.params.b() - 1)
    }

    /// The bits each compression layer must reach: layer 1 and the shrink
    /// layer each take half of the error the fold leaves under the minimum.
    fn compression_layer_bits(&self) -> Option<f64> {
        self.params
            .compression_security_bits()
            .map(|bits| bits + 1.0)
    }
}
