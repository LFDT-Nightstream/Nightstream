//! Prover-side NIFS composition `Pi_CCS -> Pi_RLC -> Pi_DEC`.
//!
//! Owns: ordered prover orchestration, witness handoff between reductions, and
//! backend/adapter entrypoints.
//!
//! Does not own: reduction arithmetic, verifier checks, or backend kernels.
//!
//! Emits constraints: no.
//!
//! Authority boundary: returned proofs, accumulators, and backend summaries are
//! prover outputs; they acquire authority only through NIFS.V.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | CPU composition | [`prove`] | no | Prover witness and protocol transcript |
//! | Reduction order | `prove_owned` | no | Pi_CCS, Pi_RLC, then Pi_DEC outputs |
//! | Adapter dispatch | Adapter entrypoints | no | Materialized proof verified by NIFS.V |

use neo_ajtai::AjtaiSModule;
use neo_reductions::optimized_engine::{OptimizedStructureCache, PaperJointOracleBackend};
use neo_reductions::superneo_eval::SuperneoEvalCache;

use crate::engine::transcript::Transcript;
use crate::paper::construction2::RunningInstance;
use crate::paper::nifs::work::{chain_witness_refs, split_fresh_instances};
use crate::paper::nifs::{
    Error, NifsProof, NifsProverAdapter, NifsProverRequest, OptimizedCpuNifsProver, OptimizedNifsProverAdapter,
};
use crate::paper::params::Params;
use crate::paper::relations::{
    ajtai_dec_mixer, ajtai_rlc_mixer, CcsInstance, DecMixer, LaneScheme, RlcMixer, Structure,
};
use crate::paper::{pi_ccs, pi_dec, pi_rlc};

#[cfg(test)]
#[path = "../../../tests/nifs/selected_rows.rs"]
mod selected_tests;

enum ProverResources<'a> {
    General {
        cache: &'a OptimizedStructureCache,
        log: &'a AjtaiSModule,
        lanes: Option<&'a LaneScheme>,
        backend: Option<&'a mut dyn PaperJointOracleBackend>,
    },
    SelectedRows {
        cache: &'a SuperneoEvalCache,
    },
}

/// Run Π_CCS → Π_RLC → Π_DEC in order. Returns the new k-claim
/// `RunningInstance` (with prover-side witness matrices) plus the
/// `NifsProof` the verifier will replay.
///
/// `lanes` is the Nebula lane-commitment context (the auxiliary-commitment flow); `None`
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
    prove_owned_inner(
        tr,
        pp,
        s,
        ProverResources::General {
            cache,
            log,
            lanes,
            backend: None,
        },
        mix_rhos_commits,
        combine_b_pows,
        fresh,
        running,
    )
}

/// The closed package owns this row cache and the fixed commitment key.
pub(crate) fn prove_owned_with_rows(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    cache: &SuperneoEvalCache,
    fresh: Vec<CcsInstance>,
    running: RunningInstance,
) -> Result<(RunningInstance, NifsProof), Error> {
    prove_owned_inner(
        tr,
        pp,
        s,
        ProverResources::SelectedRows { cache },
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        fresh,
        running,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_owned_inner(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mut resources: ProverResources<'_>,
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
    let (pi_ccs_proof, pi_dec_precompute) = match &mut resources {
        ProverResources::General {
            cache, log, backend, ..
        } => match backend.as_mut() {
            Some(backend) => pi_ccs::prove_from_parts_with_backend(
                tr,
                pp,
                s,
                cache,
                log,
                &fresh_claims,
                &fresh_witnesses,
                &running,
                *backend,
            ),
            None => pi_ccs::prove_from_parts(tr, pp, s, cache, log, &fresh_claims, &fresh_witnesses, &running),
        },
        ProverResources::SelectedRows { cache } => {
            pi_ccs::prove_from_parts_with_rows(tr, pp, s, cache, &fresh_claims, &fresh_witnesses, &running)
        }
    }?;
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
    let (dec_out, pi_dec_proof) = match &mut resources {
        ProverResources::General {
            cache,
            log,
            lanes,
            backend,
        } => match backend.as_mut() {
            Some(backend) => pi_dec::prove_with_precompute_and_backend(
                pp,
                s,
                cache,
                log,
                *lanes,
                combine_b_pows,
                &rlc_out.claim,
                &rlc_out.witness,
                &pi_dec_precompute,
                *backend,
            ),
            None => pi_dec::prove_with_precompute(
                pp,
                s,
                cache,
                log,
                *lanes,
                combine_b_pows,
                &rlc_out.claim,
                &rlc_out.witness,
                &pi_dec_precompute,
            ),
        },
        ProverResources::SelectedRows { cache } => {
            pi_dec::prove_with_production_key(pp, s, cache, &rlc_out.claim, &rlc_out.witness)
        }
    }?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[nifs-prove] pi_dec                         {:>7.2}s",
        t_dec.elapsed().as_secs_f64()
    );

    let next_running = RunningInstance::new(dec_out.claims, dec_out.witnesses, Some(rlc_out.claim));
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

pub fn prove_with_adapter(
    adapter: &mut dyn NifsProverAdapter,
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
    adapter.prove(NifsProverRequest {
        tr,
        pp,
        s,
        cache,
        log,
        lanes,
        mix_rhos_commits,
        combine_b_pows,
        fresh,
        running,
    })
}

impl NifsProverAdapter for OptimizedCpuNifsProver {
    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<(RunningInstance, NifsProof), Error> {
        prove(
            request.tr,
            request.pp,
            request.s,
            request.cache,
            request.log,
            request.lanes,
            request.mix_rhos_commits,
            request.combine_b_pows,
            request.fresh,
            request.running,
        )
    }
}

/// Run the normal NIFS composition with an accelerated one-joint oracle.
///
/// PiRLC, PiDEC, transcript ownership, and the ordinary proof boundary stay
/// in this canonical implementation.
#[doc(hidden)]
pub fn prove_with_joint_oracle_backend(
    request: NifsProverRequest<'_>,
    backend: &mut dyn PaperJointOracleBackend,
) -> Result<(RunningInstance, NifsProof), Error> {
    let (running, proof) = prove_owned_inner(
        request.tr,
        request.pp,
        request.s,
        ProverResources::General {
            cache: request.cache,
            log: request.log,
            lanes: request.lanes,
            backend: Some(backend),
        },
        request.mix_rhos_commits,
        request.combine_b_pows,
        request.fresh,
        request.running.clone(),
    )?;
    Ok((running, proof))
}

impl OptimizedNifsProverAdapter for OptimizedCpuNifsProver {}
