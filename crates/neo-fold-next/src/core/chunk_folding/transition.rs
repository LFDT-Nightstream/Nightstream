use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::api::{
    dec_children_with_commit, rlc_with_commit, rlc_with_commit_refs, sample_rot_rhos_n_typed,
    split_b_matrix_k_with_nonzero_flags, FoldingMode, RotRing,
};
use neo_reductions::engines::utils;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::{OptimizedStructureCache, PiCcsProvePerf};
use neo_reductions::pi_rlc_dec::OptimizedRlcDec;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use std::time::Instant;

use super::types::{CcsTransitionState, ChunkTransitionCore, CommitmentMixers};
use crate::proof::{Carry, ChunkProvePerf};

#[allow(clippy::too_many_arguments)]
pub(super) fn finish_chunk_transition_core_with_perf<L, MR, MB>(
    total_started: Instant,
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk_start_index: usize,
    fresh_step_count: usize,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: Option<&OptimizedStructureCache>,
    prepare_inputs_ms: f64,
    fresh_witnesses: &[CcsWitness<F>],
    ccs_outputs: &[CeClaim<Commitment, F, K>],
    fold_digest: [u8; 32],
    ccs_perf: PiCcsProvePerf,
    ccs_ms: f64,
) -> Result<(ChunkTransitionCore, ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    validate_ccs_outputs(
        chunk_start_index,
        fresh_step_count,
        incoming_main,
        ccs_outputs,
        fold_digest,
    )?;

    let dims_started = Instant::now();
    let dims = utils::build_dims_and_policy(params, s)?;
    let dims_ms = dims_started.elapsed().as_secs_f64() * 1_000.0;
    let rlc_rhos = sample_rlc_rhos(tr, params, ccs_outputs.len())?;
    let is_optimized = matches!(mode, FoldingMode::Optimized);

    let rlc_prepare_started = Instant::now();
    let mut rlc_inputs_wit = Vec::with_capacity(fresh_step_count + incoming_main.witnesses.len());
    rlc_inputs_wit.extend(chunk_fresh_witness_mats(fresh_witnesses));
    rlc_inputs_wit.extend(incoming_main.witnesses.iter());
    let rlc_prepare_ms = rlc_prepare_started.elapsed().as_secs_f64() * 1_000.0;

    let rlc_started = Instant::now();
    let (parent, z_mix) = if is_optimized {
        rlc_with_commit_refs(
            mode.clone(),
            s,
            params,
            &rlc_rhos,
            ccs_outputs,
            &rlc_inputs_wit,
            dims.ell_d,
            mixers.mix_rhos_commits,
        )?
    } else {
        let owned_rlc_inputs_wit: Vec<Mat<F>> = rlc_inputs_wit.iter().map(|z| (*z).clone()).collect();
        rlc_with_commit(
            mode.clone(),
            s,
            params,
            &rlc_rhos,
            ccs_outputs,
            &owned_rlc_inputs_wit,
            dims.ell_d,
            mixers.mix_rhos_commits,
        )?
    };
    let rlc_ms = rlc_started.elapsed().as_secs_f64() * 1_000.0;

    let k_dec = params.k_rho as usize;
    let dec_split_started = Instant::now();
    let (z_split, digit_nonzero) = split_b_matrix_k_with_nonzero_flags(&z_mix, k_dec, params.b)?;
    let dec_split_ms = dec_split_started.elapsed().as_secs_f64() * 1_000.0;
    let dec_commit_started = Instant::now();
    let child_commitments = commit_split_children(log, &z_split, &digit_nonzero)?;
    let dec_commit_ms = dec_commit_started.elapsed().as_secs_f64() * 1_000.0;
    let dec_started = Instant::now();
    let (children, ok_y, ok_x, ok_c) = if is_optimized {
        let cache = optimized_cache
            .ok_or_else(|| PiCcsError::InvalidInput("missing optimized structure cache for optimized DEC".into()))?;
        OptimizedRlcDec::dec_children_with_commit_cached(
            s,
            params,
            &parent,
            &z_split,
            dims.ell_d,
            &child_commitments,
            mixers.combine_b_pows,
            Some(cache.sparse()),
        )
    } else {
        dec_children_with_commit(
            mode,
            s,
            params,
            &parent,
            &z_split,
            dims.ell_d,
            &child_commitments,
            mixers.combine_b_pows,
        )
    };
    let dec_ms = dec_started.elapsed().as_secs_f64() * 1_000.0;
    if !(ok_y && ok_x && ok_c) {
        return Err(PiCcsError::ProtocolError(format!(
            "Π_DEC public checks failed for chunk starting at {}: y={}, X={}, c={}",
            chunk_start_index, ok_y, ok_x, ok_c
        )));
    }

    let ccs_output_count = ccs_outputs.len();
    let dec_children = children.len();
    let transition = ChunkTransitionCore {
        parent,
        children,
        z_split,
    };
    let perf = ChunkProvePerf {
        start_index: chunk_start_index,
        fresh_steps: fresh_step_count,
        incoming_main_claims: incoming_main.claims.len(),
        ccs_outputs: ccs_output_count,
        dec_children,
        prepare_inputs_ms,
        ccs_bind_ms: ccs_perf.bind_ms,
        ccs_sample_challenges_ms: ccs_perf.sample_challenges_ms,
        ccs_fe_sumcheck_ms: ccs_perf.fe_sumcheck_ms,
        ccs_nc_sumcheck_ms: ccs_perf.nc_sumcheck_ms,
        ccs_output_materialize_ms: ccs_perf.output_materialize_ms,
        ccs_ms,
        dims_ms,
        rlc_prepare_ms,
        rlc_ms,
        dec_split_ms,
        dec_commit_ms,
        dec_ms,
        total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
    };
    Ok((transition, perf))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn finish_chunk_transition_with_perf<L, MR, MB>(
    total_started: Instant,
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk_start_index: usize,
    fresh_step_count: usize,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: Option<&OptimizedStructureCache>,
    prepare_inputs_ms: f64,
    fresh_witnesses: &[CcsWitness<F>],
    ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    fold_digest: [u8; 32],
    ccs_perf: PiCcsProvePerf,
    ccs_ms: f64,
) -> Result<(CcsTransitionState, ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let (transition, perf) = finish_chunk_transition_core_with_perf(
        total_started,
        mode,
        tr,
        params,
        s,
        chunk_start_index,
        fresh_step_count,
        incoming_main,
        log,
        mixers,
        optimized_cache,
        prepare_inputs_ms,
        fresh_witnesses,
        &ccs_outputs,
        fold_digest,
        ccs_perf,
        ccs_ms,
    )?;
    Ok((
        CcsTransitionState {
            ccs_outputs,
            parent: transition.parent,
            children: transition.children,
            z_split: transition.z_split,
        },
        perf,
    ))
}

fn chunk_fresh_witness_mats<'a>(fresh_witnesses: &'a [CcsWitness<F>]) -> impl Iterator<Item = &'a Mat<F>> + 'a {
    fresh_witnesses.iter().map(|witness| &witness.Z)
}

fn validate_ccs_outputs(
    chunk_start_index: usize,
    fresh_step_count: usize,
    incoming_main: &Carry,
    ccs_outputs: &[CeClaim<Commitment, F, K>],
    fold_digest: [u8; 32],
) -> Result<(), PiCcsError> {
    let expected = fresh_step_count
        .checked_add(incoming_main.claims.len())
        .ok_or_else(|| PiCcsError::InvalidInput("Π_CCS output count overflow".into()))?;
    if ccs_outputs.len() != expected {
        return Err(PiCcsError::ProtocolError(format!(
            "Π_CCS returned {} outputs for chunk starting at {}, expected {}",
            ccs_outputs.len(),
            chunk_start_index,
            expected
        )));
    }
    for (idx, out) in ccs_outputs.iter().enumerate() {
        if out.fold_digest != fold_digest {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_CCS output[{idx}] fold_digest mismatch for chunk starting at {}",
                chunk_start_index
            )));
        }
    }
    Ok(())
}

fn sample_rlc_rhos(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    input_count: usize,
) -> Result<Vec<neo_reductions::api::RotRho>, PiCcsError> {
    let ring = RotRing::goldilocks();
    sample_rot_rhos_n_typed(tr, params, &ring, input_count)
}

fn commit_split_children<L>(log: &L, z_split: &[Mat<F>], digit_nonzero: &[bool]) -> Result<Vec<Commitment>, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
{
    if z_split.len() != digit_nonzero.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "DEC split mismatch: |Z_split|={} != |digit_nonzero|={}",
            z_split.len(),
            digit_nonzero.len()
        )));
    }
    if z_split.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "DEC requires at least one child witness".into(),
        ));
    }

    let zero = log.commit(&Mat::zero(z_split[0].rows(), z_split[0].cols(), F::ZERO));
    let mut child_commitments = vec![zero.clone(); z_split.len()];
    let nonzero_idx: Vec<usize> = digit_nonzero
        .iter()
        .enumerate()
        .filter_map(|(idx, &nz)| nz.then_some(idx))
        .collect();
    if nonzero_idx.is_empty() {
        return Ok(child_commitments);
    }

    let mats: Vec<&Mat<F>> = nonzero_idx.iter().map(|&idx| &z_split[idx]).collect();
    let commits = log.commit_many(&mats);
    if commits.len() != mats.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "DEC commit_many returned {} commitments for {} matrices",
            commits.len(),
            mats.len()
        )));
    }
    for (pos, &idx) in nonzero_idx.iter().enumerate() {
        child_commitments[idx] = commits[pos].clone();
    }
    Ok(child_commitments)
}
