//! Native NIFS handoff to the selected Stage 1 caller packet.
//! Package-owned verification fixes the returned children. State hashes and
//! frame metadata are recomputed; packet data still requires the emitted
//! witness program and its relation checks.

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::{KExtensions, F};
use nightstream_fprime::{
    PackageError, PiCcsV1_1PackageInputs, PiDecV1_1PackageInputs, PI_CCS_V1_1_COEFFICIENT_COUNT,
    PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_DEC_V1_1_PUBLIC_INPUT_WORDS_PER_CHILD,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{
    encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage,
    PiCcsV1_1PackageBridgeError, PiCcsV1_1ProofInputs, Poseidon2HashChainV1Package,
};
use crate::engine::transcript::Transcript;
use crate::paper::{
    construction2::RunningInstance,
    nifs,
    params::Params,
    relations::{ajtai_dec_mixer, ajtai_rlc_mixer, CcsClaim},
};

// Lifecycle.Stage1.Poseidon2HashChainV1.preimage: domain ++ current ++ message.
const APPLICATION_TAG: &[u8; 40] = b"Nightstream/Stage1/Poseidon2HashChain/v1";

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
    Parameters(#[from] neo_params::ParamsError),
    #[error(transparent)]
    Nifs(#[from] nifs::Error),
    #[error(transparent)]
    Bridge(#[from] PiCcsV1_1PackageBridgeError),
    #[error(transparent)]
    Package(#[from] PackageError),
}

/// Derived caller data for the emitted witness program. This packet is not
/// an accepted envelope; `next_running` contains no prover witnesses.
#[derive(Debug)]
pub struct Stage1StepInputs {
    pi_ccs: PiCcsV1_1PackageInputs,
    pi_dec: PiDecV1_1PackageInputs,
    application_witness: [u64; 4],
    output_preimage: Vec<u64>,
    output_digest: [u64; 4],
    next_public_input: Vec<u64>,
    next_state: Stage1State,
    next_running: RunningInstance,
}

impl Stage1StepInputs {
    pub fn pi_ccs(&self) -> &PiCcsV1_1PackageInputs {
        &self.pi_ccs
    }

    pub fn pi_dec(&self) -> &PiDecV1_1PackageInputs {
        &self.pi_dec
    }

    pub fn application_witness(&self) -> &[u64; 4] {
        &self.application_witness
    }

    pub fn output_preimage(&self) -> &[u64] {
        &self.output_preimage
    }

    pub fn output_digest(&self) -> [u64; 4] {
        self.output_digest
    }

    pub fn next_public_input(&self) -> &[u64] {
        &self.next_public_input
    }

    pub fn next_state(&self) -> Stage1State {
        self.next_state
    }

    /// Claims and checked parent cache, framed for the next state digest.
    /// The original NIFS proof retains its own transcript-frame metadata.
    pub fn next_running(&self) -> &RunningInstance {
        &self.next_running
    }

    pub(super) fn into_running(self) -> RunningInstance {
        self.next_running
    }
}

impl Poseidon2HashChainV1Package {
    /// Replay the native NIFS verifier and construct the exact recursive
    /// caller packet. The package owns the context, parameters and fixed pc.
    /// `execute_step_witness` executes the selected relation on this data.
    pub fn step_inputs(
        &self,
        state: &Stage1State,
        running: &RunningInstance,
        fresh: &CcsClaim,
        proof: &nifs::NifsProof,
        message: [F; 4],
    ) -> Result<Stage1StepInputs, StepInputError> {
        if state.iteration == 0 || state.iteration >= F::ORDER_U64 - 1 {
            return Err(StepInputError::Input(
                "recursive counter must be positive and have a canonical successor",
            ));
        }
        if fresh.adv.is_some()
            || running
                .claims
                .iter()
                .chain(running.parent_authority.iter())
                .any(|claim| claim.adv.is_some())
        {
            return Err(StepInputError::Input(
                "selected plain claims cannot carry auxiliary commitments",
            ));
        }
        let context = self.binding.verifier_context().digest().map(F::from_u64);
        let prior_preimage = serialize_pi_ccs_v1_1_state_preimage(
            context,
            state.iteration,
            state.z0,
            state.current,
            &running.claims,
            1,
        )?;
        let prior_digest = pi_ccs_v1_1_state_hash(&prior_preimage)?;
        let prior_public_input = encode_pi_ccs_v1_1_public_input(prior_digest)?;
        if fresh.m_in != PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS
            || fresh.x.len() != prior_public_input.len()
            || fresh
                .x
                .iter()
                .zip(&prior_public_input)
                .any(|(field, word)| field.as_canonical_u64() != *word)
        {
            return Err(StepInputError::Input(
                "fresh public input differs from the recomputed prior state hash",
            ));
        }
        let prior_frame = digest_bytes(prior_digest);
        if running
            .claims
            .iter()
            .chain(running.parent_authority.iter())
            .any(|claim| claim.fold_digest != prior_frame)
        {
            return Err(StepInputError::Input(
                "running child or parent frame differs from the prior state hash",
            ));
        }
        let params = Params::for_ccs_shape(
            self.structure.n,
            self.structure.m,
            self.structure.t(),
            self.structure.max_degree(),
        )?;
        let mut transcript = Transcript::session();
        let mut next_running = nifs::verify(
            &mut transcript,
            &params,
            &self.structure,
            ajtai_rlc_mixer,
            ajtai_dec_mixer,
            std::slice::from_ref(fresh),
            running,
            proof,
        )?;
        if next_running
            .claims
            .iter()
            .chain(next_running.parent_authority.iter())
            .any(|claim| claim.adv.is_some())
        {
            return Err(StepInputError::Input(
                "selected returned claims cannot carry auxiliary commitments",
            ));
        }

        let mut application_input = Vec::with_capacity(APPLICATION_TAG.len() + state.current.len() + message.len());
        application_input.extend(
            APPLICATION_TAG
                .iter()
                .map(|&byte| F::from_u64(u64::from(byte))),
        );
        application_input.extend_from_slice(&state.current);
        application_input.extend_from_slice(&message);
        let output = poseidon2_hash(&application_input);
        let output_preimage = serialize_pi_ccs_v1_1_state_preimage(
            context,
            state.iteration + 1,
            state.z0,
            output,
            &next_running.claims,
            1,
        )?;
        let output_digest = pi_ccs_v1_1_state_hash(&output_preimage)?;
        let next_public_input = encode_pi_ccs_v1_1_public_input(output_digest)?;

        // The checked output serializer has validated all child shapes and
        // zero padding before these exact coefficient prefixes are read.
        let coefficients = PI_CCS_V1_1_COEFFICIENT_COUNT;
        let pi_dec = PiDecV1_1PackageInputs::new(
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
                    (0..PI_DEC_V1_1_PUBLIC_INPUT_WORDS_PER_CHILD)
                        .map(|column| child.X[(column % coefficients, column / coefficients)].as_canonical_u64())
                        .collect()
                })
                .collect(),
        )?;
        let pi_ccs = PiCcsV1_1ProofInputs::from_proof(std::slice::from_ref(fresh), &proof.pi_ccs)?
            .into_package_inputs(
                prior_preimage,
                output_preimage.clone(),
                prior_public_input,
                output_digest,
                self.binding.verifier_context().clone(),
            )?;

        // The next PiCCS call absorbs its state hash as the prior frame.
        // Rebind only this returned carrier after verifying the original proof.
        let next_frame = digest_bytes(output_digest);
        for claim in next_running
            .claims
            .iter_mut()
            .chain(next_running.parent_authority.iter_mut())
        {
            claim.fold_digest = next_frame;
        }
        Ok(Stage1StepInputs {
            pi_ccs,
            pi_dec,
            application_witness: message.map(|field| field.as_canonical_u64()),
            output_preimage,
            output_digest,
            next_public_input,
            next_state: Stage1State::new(state.iteration + 1, state.z0, output),
            next_running,
        })
    }
}

fn digest_bytes(words: [u64; 4]) -> [u8; 32] {
    let mut bytes = [0; 32];
    for (lane, word) in words.into_iter().enumerate() {
        bytes[lane * 8..(lane + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}
