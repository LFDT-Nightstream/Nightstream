//! NIFS.V — verifier-side composition. What F' re-runs in-circuit.

use crate::engine::transcript::Transcript;
use crate::paper::nifs::{Error, NifsProof};
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, CeClaim, DecMixer, RlcMixer, Structure};
use crate::paper::{pi_ccs, pi_dec, pi_rlc};

/// Run the three verifier checks in order on the recorded proofs. Returns
/// the verifier-side k-claim accumulator (claims only — witnesses are
/// prover-only and never cross this boundary).
pub fn verify(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    fresh_claims: &[CcsClaim],
    running_claims: &[CeClaim],
    proof: &NifsProof,
) -> Result<Vec<CeClaim>, Error> {
    // 1. Π_CCS — re-run the sumcheck and terminal identity check; the K+k
    //    output claims live inside `proof.pi_ccs.outputs`, so the verifier
    //    sees them on the wire (no placeholder, no replay).
    let ccs_out_claims = pi_ccs::verify(tr, pp, s, fresh_claims, running_claims, &proof.pi_ccs)?;
    let combined = pi_rlc::verify(tr, pp, s, mix_rhos_commits, &ccs_out_claims, &proof.pi_rlc)?;
    let children = pi_dec::verify(pp, s, combine_b_pows, &combined, &proof.pi_dec)?;
    Ok(children)
}
