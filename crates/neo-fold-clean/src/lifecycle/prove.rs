//! Prover-side lifecycle: `prove` (top-level loop) + `extend` (one step) +
//! `start_proof` (base-case `Uncompressed` constructor).

use crate::engine::transcript::Transcript;
use crate::lifecycle::{Error, Preprocessing, Uncompressed};
use crate::paper::construction2::{self, State};
use crate::paper::relations::{CcsClaim, CcsInstance};

/// Drive the IVC over a sequence of batches, top-down. Each batch is
/// `Vec<CcsInstance>` — typically produced by
/// [`crate::lifecycle::FoldSchedule::partition`].
pub fn prove<I>(prep: &Preprocessing, batches: I) -> Result<Uncompressed, Error>
where
    I: IntoIterator<Item = Vec<CcsInstance>>,
{
    let mut in_flight = start_proof(prep);
    for batch in batches {
        in_flight = extend(prep, in_flight, batch)?;
    }
    Ok(in_flight)
}

/// Extend an in-flight proof by one step. The batch is the K instances the
/// next step will fold into running (i.e., what becomes `state.proof.latest`).
pub fn extend(prep: &Preprocessing, mut proof: Uncompressed, batch: Vec<CcsInstance>) -> Result<Uncompressed, Error> {
    if proof.final_fold.is_some() {
        return Err(Error::AlreadyFinalized);
    }
    let public_batch: Vec<CcsClaim> = batch.iter().map(|i| i.claim.clone()).collect();
    super::validate_public_input_len(prep, &public_batch)?;
    let (next_state, step_proof) = construction2::step(
        &mut proof.transcript,
        &prep.params,
        &prep.structure,
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        proof.state,
        batch,
    )?;
    proof.state = next_state;
    proof.steps.push(step_proof);
    proof.public_batches.push(public_batch);
    Ok(proof)
}

/// Base-case `Uncompressed`: empty steps, empty `public_batches`,
/// fresh transcript, base `State`.
pub(super) fn start_proof(prep: &Preprocessing) -> Uncompressed {
    let structure = crate::paper::digest::structure_digest(&prep.structure);
    let z_0 = crate::paper::digest::initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = crate::paper::digest::public_trace_seed_digest(&structure);
    let acc_digest = crate::paper::digest::accumulator_digest_from_claims(prep.params.b(), &[]);
    Uncompressed {
        state: State::base(z_0, public_trace, acc_digest),
        steps: Vec::new(),
        public_batches: Vec::new(),
        final_fold: None,
        transcript: Transcript::session(),
    }
}
