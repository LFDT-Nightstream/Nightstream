//! Direct-CCS proof lifecycle.

use crate::frontends::direct_ccs::{
    start_direct_ccs_proof_state, DirectCcsFPrimeSnarkError, DirectCcsProgram, DirectCcsProofState,
    DirectCcsRecursiveIvcPublicImage, DirectCcsRecursiveIvcSnark, DirectCcsRecursiveIvcSnarkPerf,
    DirectCcsRecursiveIvcSnarkVerifierKey, DirectCcsRecursiveIvcSummary, DirectCcsStep,
};

use super::{IncrementalProofSystem, SpartanProofSystem};

pub struct DirectCcs;

pub type DirectCcsFinishedProof = DirectCcsRecursiveIvcSnark;
pub type DirectCcsFinishedVerifierKey = DirectCcsRecursiveIvcSnarkVerifierKey;
pub type DirectCcsFinishedProofPerf = DirectCcsRecursiveIvcSnarkPerf;
pub type DirectCcsFinishedPublicImage = DirectCcsRecursiveIvcPublicImage;
pub type DirectCcsProofSummary = DirectCcsRecursiveIvcSummary;

pub type DirectCcsRlcCommitmentMixer =
    fn(&[neo_ccs::Mat<neo_math::F>], &[neo_ajtai::Commitment]) -> neo_ajtai::Commitment;
pub type DirectCcsDecCommitmentMixer = fn(&[neo_ajtai::Commitment], u32) -> neo_ajtai::Commitment;
type DirectCcsCoreCommitmentOps =
    crate::core::prover::CommitmentMixers<DirectCcsRlcCommitmentMixer, DirectCcsDecCommitmentMixer>;

/// Commitment operations used by the Direct-CCS folding path.
#[derive(Clone, Copy)]
pub struct DirectCcsCommitmentOps {
    inner: DirectCcsCoreCommitmentOps,
}

impl DirectCcsCommitmentOps {
    pub fn new(mix_rhos_commits: DirectCcsRlcCommitmentMixer, combine_b_pows: DirectCcsDecCommitmentMixer) -> Self {
        Self {
            inner: crate::core::prover::CommitmentMixers {
                mix_rhos_commits,
                combine_b_pows,
            },
        }
    }

    fn as_core_ops(&self) -> DirectCcsCoreCommitmentOps {
        self.inner
    }
}

/// Spartan-compressed Direct-CCS proof artifacts.
pub struct DirectCcsFinishedProofBundle {
    proof: DirectCcsFinishedProof,
    verifier_key: DirectCcsFinishedVerifierKey,
    perf: DirectCcsFinishedProofPerf,
}

impl DirectCcsFinishedProofBundle {
    pub fn proof(&self) -> &DirectCcsFinishedProof {
        &self.proof
    }

    pub fn verifier_key(&self) -> &DirectCcsFinishedVerifierKey {
        &self.verifier_key
    }

    pub fn perf(&self) -> &DirectCcsFinishedProofPerf {
        &self.perf
    }
}

/// Native Direct-CCS proof state before Spartan finishing.
///
/// This is a replayable prover-side artifact. It keeps the private step inputs
/// so `verify_direct_ccs` can rebuild the same Direct CCS/F' carrier without
/// paying for Spartan. The public proof boundary is `finish_direct_ccs_with_spartan`.
#[derive(Clone)]
pub struct DirectCcsProof {
    steps: Vec<DirectCcsStep>,
    state: DirectCcsProofState,
}

impl DirectCcsProof {
    fn start(preprocessing: &DirectCcsProverPreprocessing) -> Result<Self, DirectCcsFPrimeSnarkError> {
        Ok(Self {
            steps: Vec::new(),
            state: start_direct_ccs_proof_state(preprocessing.program.clone())?,
        })
    }

    pub fn steps(&self) -> &[DirectCcsStep] {
        &self.steps
    }

    pub fn summary(&self) -> DirectCcsProofSummary {
        self.state.summary()
    }
}

/// Prover-side preprocessing for Direct CCS.
#[derive(Clone)]
pub struct DirectCcsProverPreprocessing {
    program: DirectCcsProgram,
    commitment_module: neo_ajtai::AjtaiSModule,
    commitment_ops: DirectCcsCommitmentOps,
}

/// Prepare a Direct-CCS prover for repeated proofs of the same CCS program.
pub fn preprocess_direct_ccs(
    program: DirectCcsProgram,
    commitment_module: neo_ajtai::AjtaiSModule,
    commitment_ops: DirectCcsCommitmentOps,
) -> DirectCcsProverPreprocessing {
    DirectCcsProverPreprocessing {
        program,
        commitment_module,
        commitment_ops,
    }
}

impl IncrementalProofSystem for DirectCcs {
    type Preprocessing = DirectCcsProverPreprocessing;
    type Step = DirectCcsStep;
    type Proof = DirectCcsProof;
    type Error = DirectCcsFPrimeSnarkError;

    fn prove<Steps>(preprocessing: &Self::Preprocessing, steps: Steps) -> Result<Self::Proof, Self::Error>
    where
        Steps: IntoIterator<Item = Self::Step>,
    {
        prove_direct_ccs(preprocessing, steps)
    }

    fn extend(
        preprocessing: &Self::Preprocessing,
        proof: Self::Proof,
        step: Self::Step,
    ) -> Result<Self::Proof, Self::Error> {
        extend_direct_ccs(preprocessing, proof, step)
    }

    fn verify(preprocessing: &Self::Preprocessing, proof: &Self::Proof) -> Result<(), Self::Error> {
        verify_direct_ccs(preprocessing, proof)
    }
}

impl SpartanProofSystem for DirectCcs {
    type FinishedProof = DirectCcsFinishedProof;
    type FinishedVerifierKey = DirectCcsFinishedVerifierKey;
    type FinishedPublicImage = DirectCcsFinishedPublicImage;
    type FinishedProofBundle = DirectCcsFinishedProofBundle;

    fn finish_with_spartan(proof: &Self::Proof) -> Result<Self::FinishedProofBundle, Self::Error> {
        finish_direct_ccs_with_spartan(proof)
    }

    fn verify_finished_with_spartan(
        verifier_key: &Self::FinishedVerifierKey,
        expected_public_image: &Self::FinishedPublicImage,
        proof: &Self::FinishedProof,
    ) -> Result<(), Self::Error> {
        verify_finished_direct_ccs_with_spartan(verifier_key, expected_public_image, proof)
    }
}

/// Prove a Direct-CCS run without Spartan finishing.
pub fn prove_direct_ccs(
    preprocessing: &DirectCcsProverPreprocessing,
    steps: impl IntoIterator<Item = DirectCcsStep>,
) -> Result<DirectCcsProof, DirectCcsFPrimeSnarkError> {
    let mut proof = DirectCcsProof::start(preprocessing)?;
    for step in steps {
        proof = extend_direct_ccs(preprocessing, proof, step)?;
    }
    Ok(proof)
}

/// Extend a native Direct-CCS proof by one CCS step.
pub fn extend_direct_ccs(
    preprocessing: &DirectCcsProverPreprocessing,
    proof: DirectCcsProof,
    step: DirectCcsStep,
) -> Result<DirectCcsProof, DirectCcsFPrimeSnarkError> {
    let mut proof = proof;
    let next_state = proof.state.append_step(
        step.clone(),
        &preprocessing.commitment_module,
        preprocessing.commitment_ops.as_core_ops(),
    )?;
    proof.state = next_state;
    proof.steps.push(step);
    Ok(proof)
}

/// Finish a native Direct-CCS proof with Spartan.
pub fn finish_direct_ccs_with_spartan(
    proof: &DirectCcsProof,
) -> Result<DirectCcsFinishedProofBundle, DirectCcsFPrimeSnarkError> {
    let (finished_proof, verifier_key, perf) = proof.state.compress_recursive_snark()?;
    Ok(DirectCcsFinishedProofBundle {
        proof: finished_proof,
        verifier_key,
        perf,
    })
}

/// Prove a Direct-CCS run and immediately finish it with Spartan.
pub fn prove_and_finish_direct_ccs_with_spartan(
    preprocessing: &DirectCcsProverPreprocessing,
    steps: impl IntoIterator<Item = DirectCcsStep>,
) -> Result<DirectCcsFinishedProofBundle, DirectCcsFPrimeSnarkError> {
    let proof = prove_direct_ccs(preprocessing, steps)?;
    finish_direct_ccs_with_spartan(&proof)
}

/// Verify a native Direct-CCS proof without Spartan.
pub fn verify_direct_ccs(
    preprocessing: &DirectCcsProverPreprocessing,
    proof: &DirectCcsProof,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let rebuilt = prove_direct_ccs(preprocessing, proof.steps.iter().cloned())?;
    verify_direct_ccs_replay_matches(proof, &rebuilt)
}

/// Verify a Spartan-finished Direct-CCS proof against the caller's public image.
pub fn verify_finished_direct_ccs_with_spartan(
    vk: &DirectCcsFinishedVerifierKey,
    expected_public_image: &DirectCcsFinishedPublicImage,
    proof: &DirectCcsFinishedProof,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    proof.verify(vk, expected_public_image)
}

fn verify_direct_ccs_replay_matches(
    recorded: &DirectCcsProof,
    replayed: &DirectCcsProof,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    if direct_ccs_native_proofs_match(recorded, replayed) {
        return Ok(());
    }
    Err(DirectCcsFPrimeSnarkError::Verify(
        "native Direct CCS proof does not replay to the same carried state".into(),
    ))
}

fn direct_ccs_native_proofs_match(left: &DirectCcsProof, right: &DirectCcsProof) -> bool {
    let left_state = left.state.direct_state().final_state();
    let right_state = right.state.direct_state().final_state();
    left.summary() == right.summary()
        && left_state.chunk_count == right_state.chunk_count
        && left_state.step_count == right_state.step_count
        && left_state.transcript == right_state.transcript
        && left_state.carry.claims == right_state.carry.claims
        && left_state.carry.witnesses == right_state.carry.witnesses
        && left.state.direct_state().construction2_public_boundary()
            == right.state.direct_state().construction2_public_boundary()
}
