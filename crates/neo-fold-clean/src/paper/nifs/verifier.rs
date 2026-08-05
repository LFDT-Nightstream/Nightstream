//! Verifier-side NIFS composition `Pi_CCS -> Pi_RLC -> Pi_DEC`.
//!
//! Owns: checked-parent-cache validation, ordered verifier replay, and
//! construction of the exact claims-only output accumulator.
//!
//! Does not own: reduction arithmetic, transcript implementation, or in-circuit
//! lowering.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the ordered child vector is the Construction-2
//! accumulator; its Pi_RLC parent cache is independently revalidated through
//! Pi_DEC before each supplied proof feeds the next phase.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Running parent cache | `validate_running_parent_authority` | no | Pi_DEC recomposition |
//! | Reduction replay | [`verify`] | no | Pi_CCS, Pi_RLC, and Pi_DEC verifiers |
//! | Output accumulator | [`verify`] | no | Exact ordered child claims |

use neo_reductions::optimized_engine::OptimizedStructureCache;

use crate::engine::transcript::Transcript;
use crate::paper::construction2::RunningInstance;
use crate::paper::nifs::{Error, NifsProof};
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, DecMixer, RlcMixer, Structure};
use crate::paper::{pi_ccs, pi_dec, pi_rlc};

/// Run the three verifier checks in order on the recorded proofs. Returns
/// the verifier-side k-claim accumulator (claims only — witnesses are
/// prover-only and never cross this boundary).
pub fn verify(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    proof: &NifsProof,
) -> Result<RunningInstance, Error> {
    #[cfg(feature = "perf-timers")]
    let total_started = std::time::Instant::now();
    #[cfg(feature = "perf-timers")]
    let parent_started = std::time::Instant::now();
    validate_running_parent_authority(pp, s, combine_b_pows, running)?;
    #[cfg(feature = "perf-timers")]
    let parent_elapsed = parent_started.elapsed();

    // 1. Π_CCS — re-run the sumcheck and terminal identity check; the K+k
    //    output claims live inside `proof.pi_ccs.outputs`, so the verifier
    //    sees them on the wire (no placeholder, no replay).
    #[cfg(feature = "perf-timers")]
    let pi_ccs_started = std::time::Instant::now();
    let ccs_out_claims = pi_ccs::verify(tr, pp, s, cache, fresh_claims, running, &proof.pi_ccs)?;
    #[cfg(feature = "perf-timers")]
    let pi_ccs_elapsed = pi_ccs_started.elapsed();
    #[cfg(feature = "perf-timers")]
    let pi_rlc_started = std::time::Instant::now();
    let combined = pi_rlc::verify(tr, pp, s, mix_rhos_commits, &ccs_out_claims, &proof.pi_rlc)?;
    #[cfg(feature = "perf-timers")]
    let pi_rlc_elapsed = pi_rlc_started.elapsed();
    #[cfg(feature = "perf-timers")]
    let pi_dec_started = std::time::Instant::now();
    let children = pi_dec::verify(pp, s, combine_b_pows, &combined, &proof.pi_dec)?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[nifs-verify] parent={:.3}s pi_ccs={:.3}s pi_rlc={:.3}s pi_dec={:.3}s total={:.3}s fresh={} running={} outputs={} children={}",
        parent_elapsed.as_secs_f64(),
        pi_ccs_elapsed.as_secs_f64(),
        pi_rlc_elapsed.as_secs_f64(),
        pi_dec_started.elapsed().as_secs_f64(),
        total_started.elapsed().as_secs_f64(),
        fresh_claims.len(),
        running.claims.len(),
        ccs_out_claims.len(),
        children.len(),
    );
    Ok(RunningInstance::new(children, Vec::new(), Some(combined)))
}

/// Independent PaperExact verifier replay for the same NIFS proof bytes.
pub fn verify_paper_exact(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    proof: &NifsProof,
) -> Result<RunningInstance, Error> {
    validate_running_parent_authority_paper_exact(pp, s, combine_b_pows, running)?;
    let ccs_out_claims = pi_ccs::verify_paper_exact(tr, pp, s, fresh_claims, running, &proof.pi_ccs)?;
    let combined = pi_rlc::verify_paper_exact(tr, pp, s, mix_rhos_commits, &ccs_out_claims, &proof.pi_rlc)?;
    let children = pi_dec::verify_paper_exact(pp, s, combine_b_pows, &combined, &proof.pi_dec)?;
    Ok(RunningInstance::new(children, Vec::new(), Some(combined)))
}

fn validate_running_parent_authority_paper_exact(
    pp: &Params,
    s: &Structure,
    combine: DecMixer,
    running: &RunningInstance,
) -> Result<(), Error> {
    match (running.claims.is_empty(), running.parent_authority.as_ref()) {
        (true, None) => Ok(()),
        (true, Some(_)) | (false, None) => Err(pi_dec::Error::VerifyRejected.into()),
        (false, Some(parent)) => {
            let proof = pi_dec::Proof {
                children: running.claims.clone(),
            };
            pi_dec::verify_paper_exact(pp, s, combine, parent, &proof)?;
            Ok(())
        }
    }
}

fn validate_running_parent_authority(
    pp: &Params,
    s: &Structure,
    combine: DecMixer,
    running: &RunningInstance,
) -> Result<(), Error> {
    match (running.claims.is_empty(), running.parent_authority.as_ref()) {
        (true, None) => Ok(()),
        (true, Some(_)) => Err(pi_dec::Error::VerifyRejected.into()),
        (false, None) => Err(pi_dec::Error::VerifyRejected.into()),
        (false, Some(parent)) => {
            let proof = pi_dec::Proof {
                children: running.claims.clone(),
            };
            pi_dec::verify(pp, s, combine, parent, &proof)?;
            Ok(())
        }
    }
}
