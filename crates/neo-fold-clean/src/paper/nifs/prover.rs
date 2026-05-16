//! NIFS.P — prover-side composition `Π_CCS → Π_RLC → Π_DEC`.

use neo_ajtai::AjtaiSModule;

use crate::engine::transcript::Transcript;
use crate::paper::construction2::RunningInstance;
use crate::paper::nifs::work::{chain_witnesses, collect_fresh_witness_mats};
use crate::paper::nifs::{Error, NifsProof};
use crate::paper::params::Params;
use crate::paper::relations::{CcsInstance, DecMixer, RlcMixer, Structure};
use crate::paper::{pi_ccs, pi_dec, pi_rlc};

/// Run Π_CCS → Π_RLC → Π_DEC in order. Returns the new k-claim
/// `RunningInstance` (with prover-side witness matrices) plus the
/// `NifsProof` the verifier will replay.
pub fn prove(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    fresh: Vec<CcsInstance>,
    running: &RunningInstance,
) -> Result<(RunningInstance, NifsProof), Error> {
    let fresh_witness_mats = collect_fresh_witness_mats(&fresh);
    let prior_running_witnesses = running.witnesses.clone();

    // 1. Π_CCS — fold K fresh CCS into K+k CE claims at r'.
    let pi_ccs_proof = pi_ccs::prove(tr, pp, s, log, fresh, running)?;
    let all_witnesses = chain_witnesses(fresh_witness_mats, prior_running_witnesses);

    // 2. Π_RLC — combine into one CE claim of norm B.
    let (rlc_out, pi_rlc_proof) = pi_rlc::prove(tr, pp, s, mix_rhos_commits, &pi_ccs_proof.outputs, &all_witnesses)?;

    // 3. Π_DEC — split_b back to k CE claims of norm b.
    let (dec_out, pi_dec_proof) = pi_dec::prove(pp, s, log, combine_b_pows, &rlc_out.claim, &rlc_out.witness)?;

    let next_running = RunningInstance {
        claims: dec_out.claims,
        witnesses: dec_out.witnesses,
        parent_authority: Some(rlc_out.claim),
    };
    Ok((
        next_running,
        NifsProof {
            pi_ccs: pi_ccs_proof,
            pi_rlc: pi_rlc_proof,
            pi_dec: pi_dec_proof,
        },
    ))
}
