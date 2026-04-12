//! Optimized prove implementation for PiCcsEngine.
//!
//! This module contains the prove logic for the optimized engine, using
//! sparse/oracle optimizations while preserving paper-equivalent semantics.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::optimized_engine::{
    PiCcsProof, PiCcsProofVariant, PiCcsProvePerf, PiCcsReplayOutputs, PiCcsReplayProofWitness,
    PiCcsReplayTerminalState, PiCcsReplayWitnessOutputs,
};
use crate::sumcheck::RoundOracle;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::KExtensions;
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;

use super::OptimizedStructureCache;
use crate::engines::utils;

#[derive(Clone, Copy, Eq, PartialEq)]
enum ReplayTraceMode {
    Prove,
    TerminalState,
}

impl ReplayTraceMode {
    fn captures_rounds(self) -> bool {
        matches!(self, Self::Prove)
    }
}

struct OptimizedProofRounds {
    sumcheck_rounds: Vec<Vec<K>>,
    initial_sum: K,
    sumcheck_rounds_nc: Vec<Vec<K>>,
    initial_sum_nc: K,
}

/// Optimized prove implementation.
pub fn optimized_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    let cache = OptimizedStructureCache::build(s)?;
    optimized_prove_with_cache(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        &cache,
    )
}

pub fn optimized_prove_with_cache<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    let (me_outputs, proof, _perf) = optimized_prove_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
    )?;
    Ok((me_outputs, proof))
}

pub fn optimized_prove_with_cache_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, PiCcsProvePerf), PiCcsError> {
    let (terminal_state, rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        None,
        ReplayTraceMode::Prove,
    )?;
    let rounds = rounds.expect("optimized prove trace must capture proof rounds");

    let mut proof = PiCcsProof::new(rounds.sumcheck_rounds, Some(rounds.initial_sum));
    proof.variant = PiCcsProofVariant::SplitNcV1;
    proof.sumcheck_challenges = [terminal_state.row_chals.clone(), terminal_state.alpha_prime.clone()].concat();
    proof.sumcheck_rounds_nc = rounds.sumcheck_rounds_nc;
    proof.sc_initial_sum_nc = Some(rounds.initial_sum_nc);
    proof.sumcheck_challenges_nc = [terminal_state.s_col.clone(), terminal_state.alpha_prime_nc.clone()].concat();
    proof.challenges_public = terminal_state.challenges_public.clone();
    proof.sumcheck_final = terminal_state.sumcheck_final;
    proof.sumcheck_final_nc = terminal_state.sumcheck_final_nc;
    proof.header_digest = terminal_state.fold_digest.to_vec();

    Ok((terminal_state.me_outputs, proof, terminal_state.perf))
}

pub fn optimized_prove_with_cache_and_instance_digest_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof, PiCcsProvePerf), PiCcsError> {
    let (terminal_state, rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        Some(public_instance_digest),
        ReplayTraceMode::Prove,
    )?;
    let rounds = rounds.expect("optimized prove trace must capture proof rounds");

    let mut proof = PiCcsProof::new(rounds.sumcheck_rounds, Some(rounds.initial_sum));
    proof.variant = PiCcsProofVariant::SplitNcV1;
    proof.sumcheck_challenges = [terminal_state.row_chals.clone(), terminal_state.alpha_prime.clone()].concat();
    proof.sumcheck_rounds_nc = rounds.sumcheck_rounds_nc;
    proof.sc_initial_sum_nc = Some(rounds.initial_sum_nc);
    proof.sumcheck_challenges_nc = [terminal_state.s_col.clone(), terminal_state.alpha_prime_nc.clone()].concat();
    proof.challenges_public = terminal_state.challenges_public.clone();
    proof.sumcheck_final = terminal_state.sumcheck_final;
    proof.sumcheck_final_nc = terminal_state.sumcheck_final_nc;
    proof.header_digest = terminal_state.fold_digest.to_vec();

    Ok((terminal_state.me_outputs, proof, terminal_state.perf))
}

pub fn optimized_replay_terminal_state_with_cache_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayTerminalState, PiCcsError> {
    let (terminal_state, _rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        None,
        ReplayTraceMode::TerminalState,
    )?;
    validate_replay_terminal_state(params, s, mcs_list, me_inputs, &terminal_state)?;
    Ok(terminal_state)
}

pub fn optimized_replay_terminal_state_with_cache_and_instance_digest_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayTerminalState, PiCcsError> {
    let (terminal_state, _rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        Some(public_instance_digest),
        ReplayTraceMode::TerminalState,
    )?;
    validate_replay_terminal_state(params, s, mcs_list, me_inputs, &terminal_state)?;
    Ok(terminal_state)
}

pub fn optimized_replay_outputs_with_cache_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayOutputs, PiCcsError> {
    let terminal_state = optimized_replay_terminal_state_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
    )?;
    Ok(PiCcsReplayOutputs {
        me_outputs: terminal_state.me_outputs,
        fold_digest: terminal_state.fold_digest,
        perf: terminal_state.perf,
    })
}

pub fn optimized_replay_outputs_with_cache_and_instance_digest_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayOutputs, PiCcsError> {
    let (terminal_state, _rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        Some(public_instance_digest),
        ReplayTraceMode::TerminalState,
    )?;
    Ok(PiCcsReplayOutputs {
        me_outputs: terminal_state.me_outputs,
        fold_digest: terminal_state.fold_digest,
        perf: terminal_state.perf,
    })
}

pub fn optimized_replay_witness_with_cache_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayWitnessOutputs, PiCcsError> {
    let (terminal_state, rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        None,
        ReplayTraceMode::Prove,
    )?;
    let rounds = rounds.expect("optimized replay-witness trace must capture proof rounds");
    Ok(PiCcsReplayWitnessOutputs {
        me_outputs: terminal_state.me_outputs,
        replay_proof: PiCcsReplayProofWitness {
            sumcheck_rounds: rounds.sumcheck_rounds,
            sumcheck_rounds_nc: rounds.sumcheck_rounds_nc,
            header_digest: terminal_state.fold_digest,
        },
        perf: terminal_state.perf,
    })
}

pub fn optimized_replay_witness_with_cache_and_instance_digest_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<PiCcsReplayWitnessOutputs, PiCcsError> {
    let (terminal_state, rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        Some(public_instance_digest),
        ReplayTraceMode::Prove,
    )?;
    let rounds = rounds.expect("optimized replay-witness trace must capture proof rounds");
    Ok(PiCcsReplayWitnessOutputs {
        me_outputs: terminal_state.me_outputs,
        replay_proof: PiCcsReplayProofWitness {
            sumcheck_rounds: rounds.sumcheck_rounds,
            sumcheck_rounds_nc: rounds.sumcheck_rounds_nc,
            header_digest: terminal_state.fold_digest,
        },
        perf: terminal_state.perf,
    })
}

pub fn optimized_replay_trace_with_cache_and_instance_digest_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<(PiCcsReplayTerminalState, PiCcsReplayProofWitness), PiCcsError> {
    let (terminal_state, rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        cache,
        Some(public_instance_digest),
        ReplayTraceMode::Prove,
    )?;
    validate_replay_terminal_state(params, s, mcs_list, me_inputs, &terminal_state)?;
    let rounds = rounds.expect("optimized replay trace must capture proof rounds");
    Ok((
        terminal_state.clone(),
        PiCcsReplayProofWitness {
            sumcheck_rounds: rounds.sumcheck_rounds,
            sumcheck_rounds_nc: rounds.sumcheck_rounds_nc,
            header_digest: terminal_state.fold_digest,
        },
    ))
}

fn run_optimized_replay_with_cache_and_perf<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    cache: &OptimizedStructureCache,
    public_instance_digest: Option<[F; 4]>,
    mode: ReplayTraceMode,
) -> Result<(PiCcsReplayTerminalState, Option<OptimizedProofRounds>), PiCcsError> {
    let total_started = std::time::Instant::now();
    if mcs_list.is_empty() {
        return Err(PiCcsError::InvalidInput("optimized_prove: empty mcs_list".into()));
    }
    if mcs_list.len() != mcs_witnesses.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "optimized_prove: |mcs_list| mismatch (expected {}, got {})",
            mcs_list.len(),
            mcs_witnesses.len()
        )));
    }
    if me_inputs.len() != me_witnesses.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "optimized_prove: |me_inputs| mismatch (expected {}, got {})",
            me_inputs.len(),
            me_witnesses.len()
        )));
    }

    // Dims + transcript binding
    let bind_started = std::time::Instant::now();
    let dims = utils::build_dims_and_policy(params, s)?;
    if let Some(public_instance_digest) = public_instance_digest {
        utils::bind_header_and_instance_digest_with_digest(
            tr,
            params,
            s,
            dims,
            cache.mat_digest(),
            &public_instance_digest,
        )?;
    } else {
        utils::bind_header_and_instances_with_digest(tr, params, s, mcs_list, dims, cache.mat_digest())?;
    }
    utils::bind_me_inputs(tr, me_inputs)?;
    let bind_ms = bind_started.elapsed().as_secs_f64() * 1_000.0;

    // Sample challenges
    let sample_started = std::time::Instant::now();
    let mut ch = utils::sample_challenges(tr, dims.ell_d, dims.ell)?;
    ch.beta_m = utils::sample_beta_m(tr, dims.ell_m)?;
    let sample_challenges_ms = sample_started.elapsed().as_secs_f64() * 1_000.0;

    let r_inputs = utils::shared_me_input_r(me_inputs, dims.ell_n)?;

    // Initial sum: use the public T computed from ME inputs and α
    // (not the full hypercube sum Q, which includes MCS/NC terms).
    // This ensures invalid witnesses fail the first sumcheck invariant.
    let initial_sum = super::claimed_initial_sum_from_inputs_with_k_mcs(s, &ch, mcs_witnesses.len(), me_inputs);

    #[cfg(feature = "debug-logs")]
    {
        eprintln!("\n========== OPTIMIZED PROVE ==========");
        eprintln!(
            "[prove] k_total = {} (mcs_witnesses={}, me_witnesses={}, me_inputs={})",
            mcs_witnesses.len() + me_witnesses.len(),
            mcs_witnesses.len(),
            me_witnesses.len(),
            me_inputs.len()
        );
        eprintln!(
            "[prove] dims: ell_d={}, ell_n={}, d_sc={}",
            dims.ell_d, dims.ell_n, dims.d_sc
        );
        eprintln!("[prove] gamma = {:?}", ch.gamma);
        eprintln!("[prove] initial_sum (public T) = {:?}", initial_sum);

        // For debugging: compute the full hypercube sum to compare
        let full_sum = super::sum_q_over_hypercube_paper_exact(
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            &ch,
            dims.ell_d,
            dims.ell_n,
            r_inputs,
        );
        let diff = full_sum - initial_sum;
        eprintln!("[prove] full Q sum = {:?}", full_sum);
        eprintln!("[prove] difference (full - T) = {:?}", diff);
        eprintln!("[prove] breakdown:");
        eprintln!("[prove]   T (Eval block) = {:?}", initial_sum);
        eprintln!("[prove]   eq(X,β)·(F+NC) = {:?}", diff);
        if full_sum != initial_sum {
            eprintln!("[prove] WARNING: Full sum != T! This means eq(X,β)·(F+NC) ≠ 0");
            eprintln!("[prove]   For valid witnesses, this should be zero!");
            eprintln!("[prove]   Either:");
            eprintln!("[prove]     - F(CCS constraints) doesn't hold → circuit witness is invalid");
            eprintln!("[prove]     - NC(norm constraints) doesn't hold → X doesn't match Z columns");
        }
    }

    // Optimized oracles with cached sparse formats and factored algebra
    let mut oracle = super::oracle::OptimizedOracle::new_with_sparse_and_superneo_cache(
        s,
        params,
        mcs_witnesses,
        me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        r_inputs,
        cache.sparse_arc(),
        cache.superneo_arc(),
    );

    // ---------------------------------------------------------------------
    // FE sumcheck channel (SplitNcV1).
    // ---------------------------------------------------------------------
    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)]);
    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    tr.append_fields_raw(&initial_sum.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(crate::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);

    let mut running_sum = initial_sum;
    let mut sumcheck_rounds = mode
        .captures_rounds()
        .then(|| Vec::with_capacity(oracle.num_rounds()));
    let mut sumcheck_chals: Vec<K> = Vec::with_capacity(oracle.num_rounds());

    let fe_sumcheck_started = std::time::Instant::now();
    for round_idx in 0..oracle.num_rounds() {
        let deg = oracle.degree_bound();
        let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
        let ys = oracle.evals_at(&xs);

        #[cfg(feature = "debug-logs")]
        if round_idx == 0 {
            eprintln!("\n[prove] === Round 0 ===");
            eprintln!("[prove] p(0) = {:?}", ys[0]);
            eprintln!("[prove] p(1) = {:?}", ys[1]);
            eprintln!("[prove] p(0) + p(1) = {:?}", ys[0] + ys[1]);
            eprintln!("[prove] running_sum (should equal T) = {:?}", running_sum);
            if ys[0] + ys[1] != running_sum {
                eprintln!("[prove] ERROR: Sumcheck invariant violated!");
                eprintln!("[prove]   This means the witness is invalid or T is computed incorrectly");
            } else {
                eprintln!("[prove] OK: p(0) + p(1) == running_sum");
            }
        }

        if ys[0] + ys[1] != running_sum {
            #[cfg(feature = "debug-logs")]
            {
                eprintln!("\n[prove] SUMCHECK FAILED at round {}", round_idx);
                eprintln!("[prove] p(0)+p(1) = {:?}", ys[0] + ys[1]);
                eprintln!("[prove] running_sum = {:?}", running_sum);
                eprintln!("[prove] difference = {:?}", (ys[0] + ys[1]) - running_sum);
            }
            return Err(PiCcsError::SumcheckError(format!(
                "round {} invariant failed: p(0)+p(1) ≠ running_sum (paper-exact)",
                round_idx
            )));
        }
        // Sumcheck requires coefficients in low→high order (c0, c1, ..., cn) so that
        // poly_eval_k(coeffs, ·) reproduces ys at x=0,1 and the verifier invariant
        // p(0)+p(1) == running_sum holds.
        let coeffs = crate::sumcheck::interpolate_from_evals(&xs, &ys);

        debug_assert_eq!(crate::sumcheck::poly_eval_k(&coeffs, K::ZERO), ys[0]);
        debug_assert_eq!(crate::sumcheck::poly_eval_k(&coeffs, K::ONE), ys[1]);

        let coeff_fields = crate::sumcheck::round_coeff_fields(&coeffs);
        tr.append_fields_raw(&coeff_fields);
        let c = tr.challenge_fields_raw(2);
        let r_i = neo_math::from_complex(c[0], c[1]);
        sumcheck_chals.push(r_i);

        // Evaluate at challenge using poly_eval_k (low→high) for consistency.
        running_sum = crate::sumcheck::poly_eval_k(&coeffs, r_i);

        oracle.fold(r_i);
        if let Some(rounds) = sumcheck_rounds.as_mut() {
            rounds.push(coeffs);
        }
    }
    let fe_sumcheck_ms = fe_sumcheck_started.elapsed().as_secs_f64() * 1_000.0;

    // ---------------------------------------------------------------------
    // NC-only sumcheck (split-NC scaffolding; claimed sum is 0)
    // ---------------------------------------------------------------------
    let mut oracle_nc = super::oracle::NcOracle::new(
        s,
        params,
        mcs_witnesses,
        me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_m,
        dims.d_sc,
    );

    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)]);
    let initial_sum_nc = K::ZERO;
    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    tr.append_fields_raw(&initial_sum_nc.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(crate::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);

    let mut running_sum_nc = initial_sum_nc;
    let mut sumcheck_rounds_nc = mode
        .captures_rounds()
        .then(|| Vec::with_capacity(oracle_nc.num_rounds()));
    let mut sumcheck_chals_nc: Vec<K> = Vec::with_capacity(oracle_nc.num_rounds());

    let nc_sumcheck_started = std::time::Instant::now();
    for _round_idx in 0..oracle_nc.num_rounds() {
        let coeffs = if let Some(coeffs) = oracle_nc.optimized_col_phase_round_coeffs() {
            coeffs
        } else {
            let deg = oracle_nc.degree_bound();
            let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
            let ys = oracle_nc.evals_at(&xs);
            crate::sumcheck::interpolate_from_evals(&xs, &ys)
        };

        let p0 = coeffs[0];
        let p1 = crate::sumcheck::poly_eval_k_base(&coeffs, F::ONE);
        if p0 + p1 != running_sum_nc {
            return Err(PiCcsError::SumcheckError(
                "NC sumcheck invariant failed: p(0)+p(1) ≠ running_sum".into(),
            ));
        }

        let coeff_fields = crate::sumcheck::round_coeff_fields(&coeffs);
        tr.append_fields_raw(&coeff_fields);
        let c = tr.challenge_fields_raw(2);
        let r_i = neo_math::from_complex(c[0], c[1]);
        sumcheck_chals_nc.push(r_i);

        running_sum_nc = crate::sumcheck::poly_eval_k(&coeffs, r_i);
        oracle_nc.fold(r_i);
        if let Some(rounds) = sumcheck_rounds_nc.as_mut() {
            rounds.push(coeffs);
        }
    }
    let nc_sumcheck_ms = nc_sumcheck_started.elapsed().as_secs_f64() * 1_000.0;

    // Build outputs at r′ using the oracle's r′-only precomputation (no dense scan).
    let output_started = std::time::Instant::now();
    let fold_digest = tr.digest32();
    let (s_col, _alpha_nc) = sumcheck_chals_nc.split_at(dims.ell_m);
    let y_zcol_digits = (!s_col.is_empty()).then(|| oracle_nc.finalized_y_zcol_digits());
    let out_me = oracle.build_me_outputs_from_ajtai_precomp(
        mcs_list,
        me_inputs,
        s_col,
        y_zcol_digits.as_deref(),
        fold_digest,
        log,
    );
    let output_materialize_ms = output_started.elapsed().as_secs_f64() * 1_000.0;

    let perf = PiCcsProvePerf {
        bind_ms,
        sample_challenges_ms,
        fe_sumcheck_ms,
        nc_sumcheck_ms,
        output_materialize_ms,
        total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
    };

    let terminal_state = PiCcsReplayTerminalState {
        me_outputs: out_me,
        challenges_public: ch,
        row_chals: sumcheck_chals[..dims.ell_n].to_vec(),
        alpha_prime: sumcheck_chals[dims.ell_n..].to_vec(),
        s_col: sumcheck_chals_nc[..dims.ell_m].to_vec(),
        alpha_prime_nc: sumcheck_chals_nc[dims.ell_m..].to_vec(),
        sumcheck_final: running_sum,
        sumcheck_final_nc: running_sum_nc,
        fold_digest,
        perf,
    };
    let rounds = mode.captures_rounds().then(|| OptimizedProofRounds {
        sumcheck_rounds: sumcheck_rounds.expect("prove mode must capture FE rounds"),
        initial_sum,
        sumcheck_rounds_nc: sumcheck_rounds_nc.expect("prove mode must capture NC rounds"),
        initial_sum_nc,
    });
    Ok((terminal_state, rounds))
}

fn validate_replay_terminal_state(
    params: &NeoParams,
    s: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    replay: &PiCcsReplayTerminalState,
) -> Result<(), PiCcsError> {
    utils::validate_me_outputs_against_inputs(
        s,
        params,
        fresh_claims,
        me_inputs,
        &replay.me_outputs,
        &replay.row_chals,
        &replay.s_col,
    )?;
    let r_inputs = utils::shared_me_input_r(me_inputs, replay.row_chals.len())?;
    let rhs_fe = super::rhs_terminal_identity_fe_with_k_mcs(
        s,
        params,
        &replay.challenges_public,
        &replay.row_chals,
        &replay.alpha_prime,
        &replay.me_outputs,
        fresh_claims.len(),
        r_inputs,
    );
    if replay.sumcheck_final != rhs_fe {
        return Err(PiCcsError::ProtocolError(
            "optimized replay FE terminal state does not match relation identity".into(),
        ));
    }

    let rhs_nc = super::rhs_terminal_identity_nc(
        params,
        &replay.challenges_public,
        &replay.s_col,
        &replay.alpha_prime_nc,
        &replay.me_outputs,
    );
    if replay.sumcheck_final_nc != rhs_nc {
        return Err(PiCcsError::ProtocolError(
            "optimized replay NC terminal state does not match relation identity".into(),
        ));
    }

    Ok(())
}

/// Simple wrapper for k=1 case (no ME inputs)
pub fn optimized_prove_simple<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    log: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    optimized_prove(tr, params, s, mcs_list, mcs_witnesses, &[], &[], log)
}
