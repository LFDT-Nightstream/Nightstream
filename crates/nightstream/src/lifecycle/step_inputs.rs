//! Native NIFS handoff to the selected Stage 1 caller packet.
//! Package-owned verification fixes the returned children. State hashes are
//! recomputed; packet data still requires the emitted witness program and its
//! relation checks.

use neo_math::{KExtensions, F};
use nightstream_fprime::{
    PackageError, PiCcsV1_2PackageInputs, PiDecV1_2PackageInputs, PI_CCS_V1_2_COEFFICIENT_COUNT,
    PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS, PI_DEC_V1_2_PUBLIC_INPUT_WORDS_PER_CHILD,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{
    encode_pi_ccs_v1_2_public_input, pi_ccs_v1_2_state_hash, serialize_pi_ccs_v1_2_state_preimage,
    PiCcsV1_2PackageBridgeError, PiCcsV1_2ProofInputs, PreparedLifecycle,
};
use crate::folding::transcript::Transcript;
use crate::folding::{self as nifs, ajtai_dec_mixer, ajtai_rlc_mixer, CcsClaim, RunningInstance};

/// Caller-supplied state coordinates. Construction assigns no authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1State {
    iteration: u64,
    z0: [F; 4],
    current: [F; 4],
}

impl Stage1State {
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

#[derive(Debug, thiserror::Error)]
pub enum StepInputError {
    #[error("selected Stage 1 caller input: {0}")]
    Input(&'static str),
    #[error(transparent)]
    Nifs(#[from] nifs::Error),
    #[error(transparent)]
    Bridge(#[from] PiCcsV1_2PackageBridgeError),
    #[error(transparent)]
    Package(#[from] PackageError),
}

/// Derived caller data for the emitted witness program. This packet is not
/// an accepted envelope; `next_running` contains no prover witnesses.
#[derive(Debug)]
pub struct Stage1StepInputs {
    pub(super) pi_ccs: PiCcsV1_2PackageInputs,
    pub(super) pi_dec: PiDecV1_2PackageInputs,
    pub(super) application_witness: Vec<u64>,
    #[cfg(test)]
    pub(super) output_preimage: Vec<u64>,
    #[cfg(test)]
    pub(super) output_digest: [u64; 4],
    pub(super) next_public_input: Vec<u64>,
    pub(super) next_state: Stage1State,
    pub(super) next_running: RunningInstance,
}

impl Stage1StepInputs {
    pub fn pi_ccs(&self) -> &PiCcsV1_2PackageInputs {
        &self.pi_ccs
    }

    pub fn pi_dec(&self) -> &PiDecV1_2PackageInputs {
        &self.pi_dec
    }

    pub fn application_witness(&self) -> &[u64] {
        &self.application_witness
    }

    #[cfg(test)]
    pub fn output_preimage(&self) -> &[u64] {
        &self.output_preimage
    }

    #[cfg(test)]
    pub fn output_digest(&self) -> [u64; 4] {
        self.output_digest
    }

    pub fn next_public_input(&self) -> &[u64] {
        &self.next_public_input
    }

    pub fn next_state(&self) -> Stage1State {
        self.next_state
    }

    /// The verified PiDEC children: the next running claims.
    pub fn next_running(&self) -> &RunningInstance {
        &self.next_running
    }

    pub(super) fn into_running(self) -> RunningInstance {
        self.next_running
    }
}

impl PreparedLifecycle {
    /// Replay the native NIFS verifier and construct the exact recursive
    /// caller packet. The package owns the context, parameters and fixed pc.
    /// `execute_step_witness` executes the selected relation on this data.
    pub fn step_inputs(
        &self,
        state: &Stage1State,
        running: &RunningInstance,
        fresh: &CcsClaim,
        proof: &nifs::NifsProof,
        application_witness: &[u64],
        output: [F; 4],
    ) -> Result<Stage1StepInputs, StepInputError> {
        let (prior_preimage, prior_digest) = self.checked_prior_state(state, running, fresh)?;
        let context = self.binding.verifier_context().digest().map(F::from_u64);
        let prior_public_input = encode_pi_ccs_v1_2_public_input(prior_digest)?;
        let params = &self.params;
        let mut transcript = Transcript::session();
        let next_running = nifs::verify(
            &mut transcript,
            params,
            &self.structure,
            ajtai_rlc_mixer,
            ajtai_dec_mixer,
            fresh,
            running,
            proof,
        )?;
        if next_running.claims.iter().any(|claim| claim.adv.is_some()) {
            return Err(StepInputError::Input(
                "selected returned claims cannot carry auxiliary commitments",
            ));
        }

        let output_preimage = serialize_pi_ccs_v1_2_state_preimage(
            context,
            state.iteration + 1,
            state.z0,
            output,
            &next_running.claims,
            1,
        )?;
        let output_digest = pi_ccs_v1_2_state_hash(&output_preimage)?;
        let next_public_input = encode_pi_ccs_v1_2_public_input(output_digest)?;

        // The checked output serializer has validated all child shapes and
        // zero padding before these exact coefficient prefixes are read.
        let coefficients = PI_CCS_V1_2_COEFFICIENT_COUNT;
        let pi_dec = PiDecV1_2PackageInputs::new(
            next_running
                .claims
                .iter()
                .map(|child| {
                    child
                        .c
                        .data
                        .iter()
                        .map(|field| field.as_canonical_u64())
                        .collect()
                })
                .collect(),
            next_running
                .claims
                .iter()
                .map(|child| {
                    child.eval_k[..coefficients]
                        .iter()
                        .map(|value| value.as_coeffs().map(|field| field.as_canonical_u64()))
                        .collect()
                })
                .collect(),
            next_running
                .claims
                .iter()
                .map(|child| {
                    child
                        .eval_a
                        .iter()
                        .map(|matrix| {
                            matrix[..coefficients]
                                .iter()
                                .map(|value| value.as_coeffs().map(|field| field.as_canonical_u64()))
                                .collect()
                        })
                        .collect()
                })
                .collect(),
            next_running
                .claims
                .iter()
                .map(|child| {
                    (0..PI_DEC_V1_2_PUBLIC_INPUT_WORDS_PER_CHILD)
                        .map(|column| child.X[(column % coefficients, column / coefficients)].as_canonical_u64())
                        .collect()
                })
                .collect(),
        )?;
        #[cfg(test)]
        let recorded_output_preimage = output_preimage.clone();
        let pi_ccs = PiCcsV1_2ProofInputs::from_proof(fresh, &proof.pi_ccs)?.into_package_inputs(
            prior_preimage,
            output_preimage,
            prior_public_input,
            output_digest,
            self.binding.verifier_context().clone(),
        )?;

        Ok(Stage1StepInputs {
            pi_ccs,
            pi_dec,
            application_witness: application_witness.to_vec(),
            #[cfg(test)]
            output_preimage: recorded_output_preimage,
            #[cfg(test)]
            output_digest,
            next_public_input,
            next_state: Stage1State::new(state.iteration + 1, state.z0, output),
            next_running,
        })
    }

    /// Check semantic prior data before proving. Frame metadata does not enter
    /// the state preimage.
    pub(super) fn checked_prior_state(
        &self,
        state: &Stage1State,
        running: &RunningInstance,
        fresh: &CcsClaim,
    ) -> Result<(Vec<u64>, [u64; 4]), StepInputError> {
        if state.iteration == 0 || state.iteration >= F::ORDER_U64 - 1 {
            return Err(StepInputError::Input(
                "recursive counter must be positive and have a canonical successor",
            ));
        }
        if fresh.adv.is_some() || running.claims.iter().any(|claim| claim.adv.is_some()) {
            return Err(StepInputError::Input(
                "selected plain claims cannot carry auxiliary commitments",
            ));
        }
        let preimage = serialize_pi_ccs_v1_2_state_preimage(
            self.binding.verifier_context().digest().map(F::from_u64),
            state.iteration,
            state.z0,
            state.current,
            &running.claims,
            1,
        )?;
        let digest = pi_ccs_v1_2_state_hash(&preimage)?;
        let public = encode_pi_ccs_v1_2_public_input(digest)?;
        if fresh.m_in != PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS
            || fresh.x.len() != public.len()
            || fresh
                .x
                .iter()
                .zip(&public)
                .any(|(field, word)| field.as_canonical_u64() != *word)
        {
            return Err(StepInputError::Input(
                "fresh public input differs from the recomputed prior state hash",
            ));
        }
        Ok((preimage, digest))
    }
}
