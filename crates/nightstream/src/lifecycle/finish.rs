//! Terminal compression. Layer 0 runs PiCCS and PiRLC once more on the final
//! running and fresh claims and stops before PiDEC (paper Lemma 1: a reduction
//! of knowledge to one CE(B) claim). Layer 1 proves that claim with the
//! `neo-spartan` argument of knowledge, so the proof carries no witness.
//!
//! Owns the meaning of `FinalProof` and its prover and verifier flows. Does
//! not own the reductions, the layer-1 protocol, or the byte layout
//! (`encoding.rs`). The verifier recomputes the state digest, the running
//! frames and the parent claim; nothing carried in the proof is authority.
//! Layer 1 takes its constants from a compression key the caller trusts.

use neo_math::D;
use neo_math::F;
use nightstream_fprime::{
    PackageError, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COUNT, PI_CCS_V1_1_SOURCE_COUNT,
};
use p3_field::PrimeField64;

use super::terminal::FinalProgram;
use super::{
    extend::prepare_running, PreparedLifecycle, ProveError, Stage1Envelope, Stage1State, StepInputError, VerifyError,
};
use crate::folding::{self as nifs, ajtai_dec_mixer, pi_ccs, pi_rlc, transcript::Transcript, CcsClaim, CeClaim};

/// A finished proof: the statement, the layer-0 messages and the layer-1
/// argument. A finished proof cannot be extended.
#[derive(Clone)]
pub struct FinalProof {
    pub(super) state: Stage1State,
    /// The 16 semantic running claims of the state-hash preimage.
    pub(super) running: Vec<CeClaim>,
    /// The latest fresh claim: commitment and public input.
    pub(super) fresh: CcsClaim,
    pub(super) pi_ccs: pi_ccs::Proof,
    pub(super) pi_rlc: pi_rlc::Proof,
    pub(super) layer1: neo_spartan::Proof,
}

impl FinalProof {
    pub fn state(&self) -> &Stage1State {
        &self.state
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
    Layer1(#[from] neo_spartan::Error),
}

impl PreparedLifecycle {
    /// Prove the final accumulator of `envelope` without its witnesses.
    /// `envelope` does not change and can still be extended.
    pub(crate) fn finish_with_spartan(
        &self,
        envelope: &Stage1Envelope,
        setup: &neo_spartan::Setup,
    ) -> Result<FinalProof, FinishError> {
        let state = envelope.state().clone();
        let (mut running, mut fresh) = match (envelope.running(), envelope.fresh()) {
            (Some(running), Some(fresh)) => (running.clone(), fresh.clone()),
            _ => return Err(FinishError::Input("an initial proof has no fold to finish")),
        };
        let semantic = running.claims.clone();
        let (_, digest) = self.checked_prior_state(&state, &running, &fresh.claim)?;
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
        let (pi_ccs, parent, pi_rlc) = nifs::prove_parent_with_rows(
            &mut transcript,
            &self.params,
            &self.structure,
            &rows,
            workspace_bytes,
            vec![fresh],
            running,
        )?;
        let relation = self.compression_relation(setup.key())?;
        let layer1 = neo_spartan::prove(
            &relation,
            setup,
            transcript.into_inner(),
            &parent.claim,
            &parent.witness,
        )?;
        Ok(FinalProof {
            state,
            running: semantic,
            fresh: fresh_claim,
            pi_ccs,
            pi_rlc,
            layer1,
        })
    }

    /// Check a finished proof against the caller's expected state and key,
    /// by the final program on plain values.
    pub(crate) fn verify_final(
        &self,
        expected_state: &Stage1State,
        key: &neo_spartan::Key,
        proof: &FinalProof,
    ) -> Result<(), VerifyError> {
        if proof.state != *expected_state {
            return Err(VerifyError::Statement("final proof differs from the external state"));
        }
        if expected_state.iteration() == 0 || expected_state.iteration() >= F::ORDER_U64 {
            return Err(VerifyError::Statement(
                "an active proof requires a positive canonical iteration",
            ));
        }
        let relation = self.compression_relation(key).map_err(VerifyError::Final)?;
        self.final_program(&relation, expected_state, Some(proof))
            .run(&mut neo_spartan::Native)
            .map_err(VerifyError::Final)
    }

    /// The final verifier of this circuit for `state`.
    pub(crate) fn final_program<'a>(
        &'a self,
        relation: &'a neo_spartan::Relation<'a>,
        state: &'a Stage1State,
        proof: Option<&'a FinalProof>,
    ) -> FinalProgram<'a> {
        FinalProgram {
            context: self.binding.verifier_context().digest(),
            f: &self.structure.f,
            base: self.params.b(),
            relation,
            state,
            proof,
        }
    }

    /// The layer-1 relation for this circuit. Every PiRLC source is a
    /// signed-unit vector, so the honest parent obeys PiRLC's guard bound
    /// `sources · T · (b - 1)`. The security share is what the caller's
    /// minimum leaves after the fold's exact error.
    fn compression_relation<'k>(
        &self,
        key: &'k neo_spartan::Key,
    ) -> Result<neo_spartan::Relation<'k>, neo_spartan::Error> {
        let bits = self
            .params
            .compression_security_bits()
            .ok_or(neo_spartan::Error::Setup(
                "the fold error leaves no room under the security minimum",
            ))?;
        let bound = PI_CCS_V1_1_SOURCE_COUNT as u32 * self.params.T() * (self.params.b() - 1);
        neo_spartan::Relation::new(
            key,
            PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS / D,
            PI_CCS_V1_1_ROUND_COUNT,
            bound,
            bits,
        )
    }

    /// Write the compression setup files of this circuit into `dir`.
    pub(crate) fn compression_setup(&self, dir: &std::path::Path) -> Result<neo_spartan::Setup, neo_spartan::Error> {
        neo_spartan::Setup::build(&self.matrix_rows(), dir)
    }

    /// Reopen setup files written for `key`.
    pub(crate) fn open_compression_setup(
        &self,
        dir: &std::path::Path,
        key: &neo_spartan::Key,
    ) -> Result<neo_spartan::Setup, neo_spartan::Error> {
        neo_spartan::Setup::open(&self.matrix_rows(), dir, key)
    }

    /// Derive the compression key from the package, without files.
    pub(crate) fn compression_key(&self) -> Result<neo_spartan::Key, neo_spartan::Error> {
        neo_spartan::Key::derive(&self.matrix_rows())
    }
}
