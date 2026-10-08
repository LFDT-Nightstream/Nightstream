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
    PackageError, PiCcsV1_1EncodedInputs, PiCcsV1_1PackageInputs, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS,
    PI_CCS_V1_1_ROUND_COUNT, PI_CCS_V1_1_SOURCE_COUNT,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};

use super::terminal::{self, FinalProgram, Package};
use super::{
    encode_pi_ccs_v1_1_public_input, extend::prepare_running, pi_ccs_v1_1_prior_children, PiCcsV1_1ProofInputs,
    PreparedLifecycle, ProofCodecError, ProveError, Stage1Envelope, Stage1State, StepInputError, VerifyError,
};
use crate::folding::{
    self as nifs, ajtai_dec_mixer, pi_ccs, pi_rlc, transcript::Transcript, CcsClaim, RunningInstance,
};

/// The terminal proof the shrink layer proves: the inputs of the terminal
/// package and the layer-1 argument for one state. It never leaves the
/// prover.
pub(crate) struct TerminalProof {
    pub(crate) inputs: PiCcsV1_1EncodedInputs,
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
    /// Acceptance still needs `CompressionKey::verify`.
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
/// shape. A verifier needs nothing else. Authority: derived from the circuit
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

    /// Strict: the bytes must be the canonical encoding of the decoded key.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProofCodecError> {
        let key: Self = bincode::deserialize(bytes).map_err(|_| ProofCodecError("compression key"))?;
        if key.to_bytes() != bytes {
            return Err(ProofCodecError("non-canonical compression key"));
        }
        Ok(key)
    }

    /// Check `proof` for `expected_state` with this key alone.
    pub fn verify(&self, expected_state: &Stage1State, proof: &FinalProof) -> Result<(), VerifyError> {
        if proof.state != *expected_state {
            return Err(VerifyError::Statement("final proof differs from the external state"));
        }
        if expected_state.iteration() == 0 || expected_state.iteration() >= F::ORDER_U64 {
            return Err(VerifyError::Statement(
                "an active proof requires a positive canonical iteration",
            ));
        }
        let relation = self.relation().map_err(VerifyError::Final)?;
        let package = terminal::package().map_err(VerifyError::Final)?;
        let program = self.program(&relation, package, expected_state, None);
        let shrink = Shrink::new(self.shape, self.layer_bits).map_err(VerifyError::Final)?;
        shrink::verify(&shrink, &program, &proof.shrink).map_err(VerifyError::Final)
    }

    /// The layer-1 relation of the parent claim.
    fn relation(&self) -> Result<Relation<'_>, neo_spartan::Error> {
        Relation::new(
            &self.layer1,
            PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS / D,
            PI_CCS_V1_1_ROUND_COUNT,
            self.bound,
            self.layer_bits,
        )
    }

    fn program<'a>(
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

#[derive(Debug, thiserror::Error)]
pub enum FinishError {
    #[error("compression input: {0}")]
    Input(&'static str),
    #[error(transparent)]
    StepInputs(#[from] StepInputError),
    #[error("prior running claims are not a canonical PiDEC child family: {0}")]
    PriorFamily(#[source] nifs::Error),
    #[error(transparent)]
    Prove(#[from] ProveError),
    #[error(transparent)]
    Package(#[from] PackageError),
    #[error(transparent)]
    Fold(#[from] nifs::Error),
    #[error(transparent)]
    Compression(#[from] neo_spartan::Error),
}

/// The last fold of an active proof, stopped before Π_DEC.
pub(crate) struct LastFold {
    pub(crate) state: Stage1State,
    /// The running instance as the state hash reads it.
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
    ) -> Result<FinalProof, FinishError> {
        let last = self.last_fold(envelope)?;
        let package = terminal::package()?;
        let inputs = package
            .loaded()
            .encode_terminal_inputs(&self.terminal_inputs(&last)?)?;
        let key = &setup.key;
        let relation = key.relation()?;
        let layer1 = neo_spartan::prove(
            &relation,
            &setup.layer1,
            last.transcript.into_inner(),
            &last.parent.claim,
            &last.parent.witness,
        )?;
        let terminal = TerminalProof { inputs, layer1 };
        let program = key.program(&relation, package, &last.state, Some(&terminal));
        let shrink = shrink::prove(&Shrink::new(key.shape, key.layer_bits)?, &program)?;
        Ok(FinalProof {
            state: last.state,
            shrink,
        })
    }

    /// Run PiCCS and PiRLC on the running and fresh claims of `envelope`.
    pub(crate) fn last_fold(&self, envelope: &Stage1Envelope) -> Result<LastFold, FinishError> {
        let state = envelope.state().clone();
        let (semantic, mut fresh) = match (envelope.running(), envelope.fresh()) {
            (Some(running), Some(fresh)) => (running.clone(), fresh.clone()),
            _ => return Err(FinishError::Input("an initial proof has no fold to finish")),
        };
        let (_, digest) = self.checked_prior_state(&state, &semantic, &fresh.claim)?;
        let mut running = semantic.clone();
        prepare_running(&mut running, &self.params, digest);
        nifs::validate_running_parent_authority(&self.params, &self.structure, ajtai_dec_mixer, &running)
            .map_err(FinishError::PriorFamily)?;
        // The complete Z opening is the source; w is a redundant cache.
        fresh.witness.w.clear();
        self.validate_prover_sources(std::slice::from_ref(&fresh), &running)?;
        let fresh_claim = fresh.claim.clone();

        let rows = self.matrix_rows();
        let workspace_bytes = self.matrix_workspace_bytes()?;
        let mut transcript = Transcript::session();
        let (pi_ccs, parent, _) = nifs::prove_parent_with_rows(
            &mut transcript,
            &self.params,
            &self.structure,
            &rows,
            workspace_bytes,
            vec![fresh],
            running,
        )?;
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
    pub(crate) fn terminal_inputs(&self, last: &LastFold) -> Result<PiCcsV1_1PackageInputs, StepInputError> {
        let (prior_preimage, prior_digest) = self.checked_prior_state(&last.state, &last.running, &last.fresh)?;
        let prior_children = pi_ccs_v1_1_prior_children(&last.running.claims)?;
        let prior_public_input = encode_pi_ccs_v1_1_public_input(prior_digest)?;
        Ok(
            PiCcsV1_1ProofInputs::from_proof(std::slice::from_ref(&last.fresh), &last.pi_ccs)?.into_package_inputs(
                prior_preimage.clone(),
                prior_preimage,
                prior_children,
                prior_public_input,
                [0; 4],
                self.binding.verifier_context().clone(),
            )?,
        )
    }

    /// The key of this circuit for the layer-1 key `layer1`: the final
    /// program's constants and the shape of one run of it.
    fn compression_key_for(&self, layer1: neo_spartan::Key) -> Result<CompressionKey, neo_spartan::Error> {
        let bits = self
            .params
            .compression_security_bits()
            .ok_or(neo_spartan::Error::Setup(
                "the fold error leaves no room under the security minimum",
            ))?;
        let package = terminal::package()?;
        if !package.checks(&self.structure.f) {
            return Err(neo_spartan::Error::Setup(
                "the terminal package checks another CCS polynomial",
            ));
        }
        let mut key = CompressionKey {
            layer1,
            context: self.binding.verifier_context().digest(),
            bound: PI_CCS_V1_1_SOURCE_COUNT as u32 * self.params.T() * (self.params.b() - 1),
            // Layer 1 and the shrink layer each take half of the remainder.
            layer_bits: bits + 1.0,
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
    pub(crate) fn compression_setup(&self, dir: &std::path::Path) -> Result<CompressionSetup, neo_spartan::Error> {
        let layer1 = neo_spartan::Setup::build(&self.matrix_rows(), dir)?;
        let key = self.compression_key_for(layer1.key().clone())?;
        Ok(CompressionSetup { layer1, key })
    }

    /// Reopen setup files written for `key`.
    pub(crate) fn open_compression_setup(
        &self,
        dir: &std::path::Path,
        key: &CompressionKey,
    ) -> Result<CompressionSetup, neo_spartan::Error> {
        let layer1 = neo_spartan::Setup::open(&self.matrix_rows(), dir, &key.layer1)?;
        Ok(CompressionSetup {
            layer1,
            key: key.clone(),
        })
    }

    /// Derive the compression key from the package, without files.
    pub(crate) fn compression_key(&self) -> Result<CompressionKey, neo_spartan::Error> {
        self.compression_key_for(neo_spartan::Key::derive(&self.matrix_rows())?)
    }
}
