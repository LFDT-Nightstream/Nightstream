//! NIFS.V — verifier-side composition. What F' re-runs in-circuit.

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
    // 1. Π_CCS — re-run the sumcheck and terminal identity check; the K+k
    //    output claims live inside `proof.pi_ccs.outputs`, so the verifier
    //    sees them on the wire (no placeholder, no replay).
    let ccs_out_claims = pi_ccs::verify(tr, pp, s, cache, fresh_claims, running, &proof.pi_ccs)?;
    let combined = pi_rlc::verify(tr, pp, s, mix_rhos_commits, &ccs_out_claims, &proof.pi_rlc)?;
    let children = pi_dec::verify(pp, s, combine_b_pows, &combined, &proof.pi_dec)?;
    Ok(RunningInstance {
        claims: children,
        witnesses: Vec::new(),
        parent_authority: Some(combined),
    })
}
