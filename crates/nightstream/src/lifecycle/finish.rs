//! Terminal compression. Layer 0 runs PiCCS and PiRLC once more on the final
//! running and fresh claims and stops before PiDEC (paper Lemma 1: a reduction
//! of knowledge to one CE(B) claim). Layer 1 proves that claim with the
//! `neo-spartan` argument of knowledge, so the proof carries no witness.
//!
//! Owns the meaning of `FinalProof` and its prover and verifier flows. Does
//! not own the reductions, the layer-1 protocol, or the byte layout
//! (`encoding.rs`). The verifier recomputes the state digest, the running
//! frames and the parent claim; nothing carried in the proof is authority.

use neo_math::D;
use neo_reductions::superneo_eval::MatrixRows;
use nightstream_fprime::{PackageError, PI_CCS_V1_1_SOURCE_COUNT};

use super::{
    extend::prepare_running, PreparedLifecycle, ProveError, Stage1Envelope, Stage1State, StepInputError, VerifyError,
};
use crate::folding::{
    self as nifs, ajtai_dec_mixer, ajtai_rlc_mixer, pi_ccs, pi_rlc, transcript::Transcript, CcsClaim, CeClaim,
    RunningInstance,
};

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
    pub(crate) fn finish_with_spartan(&self, envelope: &Stage1Envelope) -> Result<FinalProof, FinishError> {
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
        let relation = self.compression_relation(&rows, &parent.claim)?;
        let layer1 = neo_spartan::prove(&relation, transcript.into_inner(), &parent.claim, &parent.witness)?;
        Ok(FinalProof {
            state,
            running: semantic,
            fresh: fresh_claim,
            pi_ccs,
            pi_rlc,
            layer1,
        })
    }

    /// Check a finished proof against the caller's expected state.
    pub(crate) fn verify_final(&self, expected_state: &Stage1State, proof: &FinalProof) -> Result<(), VerifyError> {
        if proof.state != *expected_state {
            return Err(VerifyError::Statement("final proof differs from the external state"));
        }
        let digest = self.check_statement(expected_state, &proof.running, &proof.fresh)?;
        let mut running = RunningInstance::new(proof.running.clone(), Vec::new(), None);
        prepare_running(&mut running, &self.params, digest);
        let mut transcript = Transcript::session();
        let parent = nifs::verify_parent(
            &mut transcript,
            &self.params,
            &self.structure,
            ajtai_rlc_mixer,
            ajtai_dec_mixer,
            std::slice::from_ref(&proof.fresh),
            &running,
            &proof.pi_ccs,
            &proof.pi_rlc,
        )
        .map_err(VerifyError::Layer0)?;
        let rows = self.matrix_rows();
        let relation = self
            .compression_relation(&rows, &parent)
            .map_err(VerifyError::Layer1)?;
        neo_spartan::verify(&relation, transcript.into_inner(), &parent, &proof.layer1).map_err(VerifyError::Layer1)
    }

    /// The layer-1 relation for this circuit. Every PiRLC source is a
    /// signed-unit vector, so the honest parent obeys PiRLC's guard bound
    /// `sources · T · (b - 1)`. The security share is what the caller's
    /// minimum leaves after the fold's exact error.
    fn compression_relation<'r>(
        &self,
        rows: &'r dyn MatrixRows,
        parent: &CeClaim,
    ) -> Result<neo_spartan::Relation<'r>, neo_spartan::Error> {
        let bits = self
            .params
            .compression_security_bits()
            .ok_or(neo_spartan::Error::Setup(
                "the fold error leaves no room under the security minimum",
            ))?;
        let bound = PI_CCS_V1_1_SOURCE_COUNT as u32 * self.params.T() * (self.params.b() - 1);
        neo_spartan::Relation::new(rows, parent.m_in / D, parent.r.len(), bound, bits)
    }
}
