//! Selected application lifecycle. The state and the semantic running claims
//! are the prior authority; retained witness matrices enter the existing
//! prover, checked caller packet and emitted fresh-assignment construction.

use neo_math::F;
use neo_reductions::PiCcsError;
use nightstream_fprime::PackageError;
use p3_field::PrimeField64;

use super::ProofState;
use super::{
    CompleteStepError, PiCcsV1_2PackageBridgeError, PreparedLifecycle, ProveError, Stage1Envelope, Stage1State,
    StepInputError,
};
use crate::folding::{self as nifs, CcsClaim, CcsInstance, RunningInstance};

#[derive(Debug, thiserror::Error)]
pub enum ExtendError {
    #[error("selected extension input: {0}")]
    Input(&'static str),
    #[error(transparent)]
    Bridge(#[from] PiCcsV1_2PackageBridgeError),
    #[error(transparent)]
    Package(#[from] PackageError),
    #[error("selected base sampler: {0}")]
    Sampling(#[from] crate::folding::kernels::Error),
    #[error("selected base public reduction: {0}")]
    BaseReduction(#[from] PiCcsError),
    #[error("selected prior running claims are not a canonical PiDEC child family: {0}")]
    PriorFamily(#[source] nifs::Error),
    #[error(transparent)]
    Prove(#[from] ProveError),
    #[error(transparent)]
    StepInputs(#[from] StepInputError),
    #[error(transparent)]
    Complete(#[from] CompleteStepError),
}

impl PreparedLifecycle {
    /// Apply one message to an initial or active envelope. Active inputs use
    /// the canonical PiDEC child families produced by this lifecycle; this
    /// native family check runs before proving. The supplied frame digests and
    /// redundant `w` values are not read.
    /// The returned envelope retains the actual witnesses and requires final
    /// verification against the caller's expected state.
    pub(crate) fn extend_with_output(
        &self,
        envelope: Stage1Envelope,
        application_witness: &[u64],
        output: [F; 4],
        application_values: Option<&[F]>,
    ) -> Result<Stage1Envelope, ExtendError> {
        let (state, proof) = envelope.into_parts();
        if state.iteration() >= F::ORDER_U64 - 1 {
            return Err(ExtendError::Input("iteration has no canonical successor"));
        }
        let params = &self.params;
        match proof {
            ProofState::Initial => {
                if state.iteration() != 0 || state.current() != state.z0() {
                    return Err(ExtendError::Input(
                        "bottom requires zero iterations and equal endpoints",
                    ));
                }
                let (inputs, witnesses) = self.base_inputs(params, state.z0(), application_witness, output)?;
                Ok(self.complete_step(inputs, witnesses, application_values)?)
            }
            ProofState::Active { running, fresh } => {
                let fold = self.prove_active(state, running, fresh)?;
                self.complete_fold(fold, application_witness, output, application_values)
            }
        }
    }

    /// Prove the NIFS fold of an active envelope after the prior state check
    /// and the child-family check. The frame digests and `w` are not read.
    pub(super) fn prove_active(
        &self,
        state: Stage1State,
        running: RunningInstance,
        mut fresh: CcsInstance,
    ) -> Result<ProvedFold, ExtendError> {
        self.checked_prior_state(&state, &running, &fresh.claim)?;
        nifs::validate_running_children(&self.params, &self.structure, &running).map_err(ExtendError::PriorFamily)?;
        // The complete Z opening is the source; w is a redundant cache.
        fresh.witness.w.clear();
        let prior = running.claims_only();
        let fresh_claim = fresh.claim.clone();
        let (next, proof) = self.prove(fresh, running)?;
        Ok(ProvedFold {
            state,
            prior,
            fresh: fresh_claim,
            next,
            proof,
        })
    }

    /// Replay the fold's NIFS verifier into the caller packet, then complete
    /// the envelope from the proved children.
    pub(super) fn complete_fold(
        &self,
        fold: ProvedFold,
        application_witness: &[u64],
        output: [F; 4],
        application_values: Option<&[F]>,
    ) -> Result<Stage1Envelope, ExtendError> {
        let inputs = self.step_inputs(
            &fold.state,
            &fold.prior,
            &fold.fresh,
            &fold.proof,
            application_witness,
            output,
        )?;
        Ok(self.complete_proved_step(inputs, fold.next, application_values)?)
    }
}

/// One proved active fold before its caller packet and fresh witness exist.
#[cfg_attr(test, derive(Clone))]
pub(super) struct ProvedFold {
    pub(super) state: Stage1State,
    pub(super) prior: RunningInstance,
    pub(super) fresh: CcsClaim,
    pub(super) next: RunningInstance,
    pub(super) proof: nifs::NifsProof,
}
