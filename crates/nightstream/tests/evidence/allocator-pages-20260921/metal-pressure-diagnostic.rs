//! Metal round evaluation and child openings for the selected C/R/D fold.
//! The host keeps transcript order and public checks; device arithmetic also
//! generates the fixed key and computes child commitments.

use crate::folding::{
    self, ajtai_dec_mixer, ajtai_rlc_mixer, pi_ccs, pi_dec, pi_rlc, transcript::Transcript, CcsInstance, NifsProof,
    Params, RunningInstance, Structure,
};
use neo_math::D;
use neo_prover_metal::MetalRowProver;
use neo_reductions::{
    common::split_b_matrix_k_with_nonzero_flags, optimized_engine::optimized_prove_with_row_cache_and_backend,
    superneo_eval::SuperneoEvalCache,
};
use std::sync::Arc;

pub(crate) fn prove(
    device: &mut MetalRowProver,
    transcript: &mut Transcript,
    params: &Params,
    structure: &Structure,
    cache: &Arc<SuperneoEvalCache>,
    fresh: Vec<CcsInstance>,
    running: RunningInstance,
) -> Result<(RunningInstance, NifsProof), folding::Error> {
    let (claims, witnesses): (Vec<_>, Vec<_>) = fresh
        .into_iter()
        .map(|source| (source.claim, source.witness))
        .unzip();
    memory_snapshot(device, "before_ccs");
    let (outputs, sumcheck, _, _) = optimized_prove_with_row_cache_and_backend(
        transcript.inner_mut(),
        params.inner(),
        structure,
        &claims,
        &witnesses,
        &running.claims,
        &running.witnesses,
        cache,
        device,
    )
    .map_err(folding::kernels::Error::from)
    .map_err(pi_ccs::Error::from)?;
    memory_snapshot(device, "after_ccs");
    crate::memory::release_unused_pages();
    memory_snapshot(device, "after_ccs_relief");
    let c = pi_ccs::Proof { outputs, sumcheck };
    let sources: Vec<_> = witnesses
        .iter()
        .map(|witness| &witness.Z)
        .chain(running.witnesses.iter())
        .collect();
    let (parent, r) = pi_rlc::prove_refs(transcript, params, structure, ajtai_rlc_mixer, &c.outputs, &sources)?;
    memory_snapshot(device, "after_rlc");
    drop(sources);
    drop(witnesses);
    drop(running);
    let (digits, flags) = split_b_matrix_k_with_nonzero_flags(&parent.witness, params.k_rho() as usize, params.b())
        .map_err(folding::kernels::Error::from)
        .map_err(pi_dec::Error::from)?;
    drop(parent.witness);
    memory_snapshot(device, "after_split");
    let commitments = device
        .commit_production_prefixes(&digits)
        .map_err(folding::kernels::Error::from)
        .map_err(pi_dec::Error::from)?;
    let openings = device
        .child_openings(Arc::clone(cache), &digits, &parent.claim.r, structure.m)
        .map_err(folding::kernels::Error::from)
        .map_err(pi_dec::Error::from)?;
    let (children, ok_y, ok_x, ok_c) =
        neo_reductions::api::dec_children_with_commit_superneo_cached_from_trusted_split_digits(
            neo_reductions::api::FoldingMode::Optimized,
            structure,
            params.inner(),
            &parent.claim,
            &digits,
            &flags,
            D.next_power_of_two().trailing_zeros() as usize,
            &commitments,
            ajtai_dec_mixer,
            None,
            None,
            Some(&openings),
        );
    if !(ok_y && ok_x && ok_c) {
        return Err(pi_dec::Error::Engine(folding::kernels::Error::PiDecPublicCheckFailed { ok_y, ok_x, ok_c }).into());
    }
    let d = pi_dec::Proof { children };
    let children = pi_dec::verify(params, structure, ajtai_dec_mixer, &parent.claim, &d)?;
    Ok((
        RunningInstance::new(children, digits, Some(parent.claim)),
        NifsProof {
            pi_ccs: c,
            pi_rlc: r,
            pi_dec: d,
        },
    ))
}

fn memory_snapshot(device: &MetalRowProver, phase: &str) {
    let rss = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    eprintln!(
        "memory phase={phase} device_bytes={} rss_kib={}",
        device.activity().current_allocated_bytes,
        String::from_utf8_lossy(&rss.stdout).trim()
    );
}
