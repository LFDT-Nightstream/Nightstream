//! NIFS.P — prover-side composition `Π_CCS → Π_RLC → Π_DEC`.

use neo_ajtai::AjtaiSModule;
use neo_reductions::optimized_engine::OptimizedStructureCache;

use crate::engine::transcript::Transcript;
use crate::paper::construction2::RunningInstance;
use crate::paper::nifs::work::{chain_witness_refs, split_fresh_instances};
use crate::paper::nifs::{Error, NifsProof};
use crate::paper::params::Params;
use crate::paper::relations::{CcsInstance, DecMixer, LaneScheme, RlcMixer, Structure};
use crate::paper::{pi_ccs, pi_dec, pi_rlc};

/// Run Π_CCS → Π_RLC → Π_DEC in order. Returns the new k-claim
/// `RunningInstance` (with prover-side witness matrices) plus the
/// `NifsProof` the verifier will replay.
///
/// `lanes` is the Nebula lane-commitment context (spec §5.2 R2); `None`
/// for plain chains. It is prover-only plumbing — Π_DEC needs it to
/// commit child lane slices; NIFS.V's adv checks are public arithmetic.
pub fn prove(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &AjtaiSModule,
    lanes: Option<&LaneScheme>,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    fresh: Vec<CcsInstance>,
    running: &RunningInstance,
) -> Result<(RunningInstance, NifsProof), Error> {
    prove_owned(
        tr,
        pp,
        s,
        cache,
        log,
        lanes,
        mix_rhos_commits,
        combine_b_pows,
        fresh,
        running.clone(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_owned(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    log: &AjtaiSModule,
    lanes: Option<&LaneScheme>,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    fresh: Vec<CcsInstance>,
    running: RunningInstance,
) -> Result<(RunningInstance, NifsProof), Error> {
    crate::heap::release_unused_pages();
    #[cfg(feature = "perf-timers")]
    let t_witnesses = std::time::Instant::now();
    let (fresh_claims, fresh_witnesses) = split_fresh_instances(fresh);
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[nifs-prove] split fresh instances          {:>7.2}s",
        t_witnesses.elapsed().as_secs_f64()
    );

    // 1. Π_CCS — fold K fresh CCS into K+k CE claims at r'.
    #[cfg(feature = "perf-timers")]
    let t_ccs = std::time::Instant::now();
    let (pi_ccs_proof, pi_dec_precompute) =
        pi_ccs::prove_from_parts(tr, pp, s, cache, log, &fresh_claims, &fresh_witnesses, &running)?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[nifs-prove] pi_ccs                         {:>7.2}s",
        t_ccs.elapsed().as_secs_f64()
    );

    #[cfg(feature = "perf-timers")]
    let t_chain = std::time::Instant::now();
    let all_witnesses = chain_witness_refs(&fresh_witnesses, &running.witnesses);
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[nifs-prove] chain_witnesses                {:>7.2}s",
        t_chain.elapsed().as_secs_f64()
    );

    // 2. Π_RLC — combine into one CE claim of norm B.
    #[cfg(feature = "perf-timers")]
    let t_rlc = std::time::Instant::now();
    let (rlc_out, pi_rlc_proof) =
        pi_rlc::prove_refs(tr, pp, s, mix_rhos_commits, &pi_ccs_proof.outputs, &all_witnesses)?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[nifs-prove] pi_rlc                         {:>7.2}s",
        t_rlc.elapsed().as_secs_f64()
    );
    drop(all_witnesses);
    drop(fresh_witnesses);
    drop(running);
    crate::heap::release_unused_pages();

    // 3. Π_DEC — split_b back to k CE claims of norm b.
    #[cfg(feature = "perf-timers")]
    let t_dec = std::time::Instant::now();
    let (dec_out, pi_dec_proof) = pi_dec::prove_with_precompute(
        pp,
        s,
        cache,
        log,
        lanes,
        combine_b_pows,
        &rlc_out.claim,
        &rlc_out.witness,
        &pi_dec_precompute,
    )?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[nifs-prove] pi_dec                         {:>7.2}s",
        t_dec.elapsed().as_secs_f64()
    );

    let next_running = RunningInstance {
        claims: dec_out.claims,
        witnesses: dec_out.witnesses,
        parent_authority: Some(rlc_out.claim),
    };
    let out = (
        next_running,
        NifsProof {
            pi_ccs: pi_ccs_proof,
            pi_rlc: pi_rlc_proof,
            pi_dec: pi_dec_proof,
        },
    );
    crate::heap::release_unused_pages();
    Ok(out)
}
