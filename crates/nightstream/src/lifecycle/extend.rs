//! Selected application lifecycle. State and semantic claims determine the
//! native frame and parent cache; retained witness matrices enter the existing
//! prover, checked caller packet and emitted fresh-assignment construction.

use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_reductions::PiCcsError;
use nightstream_fprime::{PackageError, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::ProofState;
use super::{
    step_inputs::digest_bytes, CompleteStepError, PiCcsV1_1PackageBridgeError, PreparedLifecycle, ProveError,
    Stage1Envelope, Stage1State, StepInputError,
};
use crate::folding::{
    self as nifs, ajtai_dec_mixer, CcsClaim, CcsInstance, CeClaim, Params, RunningInstance, EVALUATION_WIDTH,
};

#[derive(Debug, thiserror::Error)]
pub enum ExtendError {
    #[error("selected extension input: {0}")]
    Input(&'static str),
    #[error(transparent)]
    Bridge(#[from] PiCcsV1_1PackageBridgeError),
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
    /// native family check runs before proving. The supplied parent cache,
    /// frame digests and redundant `w` values do not establish authority.
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

    /// Prove the NIFS fold of an active envelope from its normalized prior
    /// claims. The supplied parent cache, frame digests and `w` are not used.
    pub(super) fn prove_active(
        &self,
        state: Stage1State,
        mut running: RunningInstance,
        mut fresh: CcsInstance,
    ) -> Result<ProvedFold, ExtendError> {
        let (_, digest) = self.checked_prior_state(&state, &running, &fresh.claim)?;
        prepare_running(&mut running, &self.params, digest);
        nifs::validate_running_parent_authority(&self.params, &self.structure, ajtai_dec_mixer, &running)
            .map_err(ExtendError::PriorFamily)?;
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

/// The state serializer has checked all semantic shapes, the exact padded
/// width and zero surplus before this function reads any evaluation coordinate.
pub(super) fn prepare_running(running: &mut RunningInstance, params: &Params, digest: [u64; 4]) {
    let frame = digest_bytes(digest);
    let padded = EVALUATION_WIDTH;
    for claim in &mut running.claims {
        claim.fold_digest = frame;
    }
    let commitments = running
        .claims
        .iter()
        .map(|claim| claim.c.clone())
        .collect::<Vec<_>>();
    let mut parent = CeClaim {
        c: ajtai_dec_mixer(&commitments, params.b()),
        X: Mat::zero(D, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS / D, F::ZERO),
        r: running.claims[0].r.clone(),
        eval_k: vec![K::ZERO; padded],
        eval_a: vec![vec![K::ZERO; padded]; running.claims[0].eval_a.len()],
        m_in: PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS,
        fold_digest: frame,
        adv: None,
    };
    let mut weight = F::ONE;
    for claim in &running.claims {
        for column in 0..parent.X.cols() {
            for lane in 0..D {
                parent.X[(lane, column)] += weight * claim.X[(lane, column)];
            }
        }
        let extension_weight = K::from(weight);
        for lane in 0..D {
            parent.eval_k[lane] += extension_weight * claim.eval_k[lane];
            for (sum, values) in parent.eval_a.iter_mut().zip(&claim.eval_a) {
                sum[lane] += extension_weight * values[lane];
            }
        }
        weight *= F::from_u64(u64::from(params.b()));
    }
    running.parent_authority = Some(parent);
}
