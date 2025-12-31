//! Paper-exact prove implementation for PiCcsEngine.
//!
//! This module contains the prove logic for the paper-exact engine,
//! which runs the sumcheck protocol using the paper-exact oracle
//! to evaluate true per-round polynomials.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::optimized_engine::PiCcsProof;
use crate::sumcheck::RoundOracle;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, Mat, McsInstance, McsWitness, MeInstance};
use neo_math::KExtensions;
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;
use std::sync::Arc;

use crate::engines::utils;

/// Paper-exact prove implementation.
///
/// This function runs the sumcheck protocol using the paper-exact oracle,
/// which directly evaluates the polynomial Q over the Boolean hypercube
/// without optimizations.
pub fn optimized_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    // Dims + transcript binding
    let dims = utils::build_dims_and_policy(params, s)?;
    utils::bind_header_and_instances(tr, params, s, mcs_list, dims.ell, dims.d_sc, 0)?;
    utils::bind_me_inputs(tr, me_inputs)?;

    // Sample challenges
    let ch = utils::sample_challenges(tr, dims.ell_d, dims.ell)?;

    // Validate ME input r (if provided)
    for (idx, me) in me_inputs.iter().enumerate() {
        if me.r.len() != dims.ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "ME input r length mismatch at accumulator #{}: expected ell_n = {}, got {}",
                idx,
                dims.ell_n,
                me.r.len()
            )));
        }
    }

    // Initial sum: use the public T computed from ME inputs and α
    // (not the full hypercube sum Q, which includes MCS/NC terms).
    // This ensures invalid witnesses fail the first sumcheck invariant.
    let initial_sum = crate::paper_exact_engine::claimed_initial_sum_from_inputs(s, &ch, me_inputs);

    #[cfg(feature = "debug-logs")]
    {
        eprintln!("\n========== PAPER-EXACT PROVE ==========");
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
        let full_sum = crate::paper_exact_engine::sum_q_over_hypercube_paper_exact(
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            &ch,
            dims.ell_d,
            dims.ell_n,
            me_inputs.first().map(|mi| mi.r.as_slice()),
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

    // Bind initial sum to transcript
    tr.append_fields(b"sumcheck/initial_sum", &initial_sum.as_coeffs());

    // Optimized oracle with cached sparse formats and factored algebra
    let sparse = Arc::new(super::oracle::SparseCache::build(s));
    let mut oracle = super::oracle::OptimizedOracle::new_with_sparse(
        s,
        params,
        mcs_witnesses,
        me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        me_inputs.first().map(|mi| mi.r.as_slice()),
        sparse,
    );

    let mut running_sum = initial_sum;
    let mut sumcheck_rounds: Vec<Vec<K>> = Vec::with_capacity(oracle.num_rounds());
    let mut sumcheck_chals: Vec<K> = Vec::with_capacity(oracle.num_rounds());

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

        for &c in &coeffs {
            tr.append_fields(b"sumcheck/round/coeff", &c.as_coeffs());
        }
        let c0 = tr.challenge_field(b"sumcheck/challenge/0");
        let c1 = tr.challenge_field(b"sumcheck/challenge/1");
        let r_i = neo_math::from_complex(c0, c1);
        sumcheck_chals.push(r_i);

        // Evaluate at challenge using poly_eval_k (low→high) for consistency.
        running_sum = crate::sumcheck::poly_eval_k(&coeffs, r_i);

        oracle.fold(r_i);
        sumcheck_rounds.push(coeffs);
    }

    // Build outputs at r′ using the oracle's r′-only precomputation (no dense scan).
    let fold_digest = tr.digest32();
    let out_me = oracle.build_me_outputs_from_ajtai_precomp(mcs_list, me_inputs, fold_digest, log);

    let mut proof = PiCcsProof::new(sumcheck_rounds, Some(initial_sum));
    proof.sumcheck_challenges = sumcheck_chals;
    proof.challenges_public = ch;
    proof.sumcheck_final = running_sum;
    proof.header_digest = fold_digest.to_vec();

    Ok((out_me, proof))
}

/// Simple wrapper for k=1 case (no ME inputs)
pub fn optimized_prove_simple<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    optimized_prove(tr, params, s, mcs_list, mcs_witnesses, &[], &[], log)
}

// ============================================================================
// Route A: Split CCS prover for batched sum-check
// ============================================================================

/// Prepared CCS context for Route A batched sum-check.
///
/// Contains all setup data needed for the CCS oracle, allowing the caller
/// to run time-rounds in a batched sum-check with Twist/Shout.
pub struct CcsBatchContext<'a, F>
where
    F: p3_field::Field + p3_field::PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    /// The CCS oracle (owns mutable state for folding)
    pub oracle: super::oracle::OptimizedOracle<'a, F>,
    /// Public challenges
    pub challenges: super::common::Challenges,
    /// Dimensions
    pub ell_n: usize,
    pub ell_d: usize,
    pub d_sc: usize,
    /// Initial sum (claimed sum for CCS)
    pub initial_sum: K,
}

/// Prepare CCS for Route A batched sum-check.
///
/// This function performs CCS setup without running sum-check rounds:
/// 1. Build dimensions and bind header
/// 2. Sample challenges (α, β, γ)
/// 3. Create the CCS oracle
/// 4. Compute initial sum
///
/// The caller can then use `context.oracle` in a batched sum-check with Twist/Shout.
/// After the batched sum-check completes, call `finalize_ccs_proof` to build ME outputs.
///
/// # Returns
/// - `CcsBatchContext` containing the oracle and all data needed for finalization
pub fn prepare_ccs_for_batch<'a>(
    tr: &mut Poseidon2Transcript,
    params: &'a NeoParams,
    s: &'a CcsStructure<F>,
    mcs_witnesses: &'a [McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &'a [neo_ccs::Mat<F>],
) -> Result<CcsBatchContext<'a, F>, PiCcsError> {
    // Dims + transcript binding
    let dims = utils::build_dims_and_policy(params, s)?;

    // Note: We skip bind_header_and_instances here as the caller may need to
    // bind additional Twist/Shout commitments first. The caller is responsible
    // for proper transcript binding order.

    // Sample challenges
    let ch = utils::sample_challenges(tr, dims.ell_d, dims.ell)?;

    // Validate ME input r (if provided)
    for (idx, me) in me_inputs.iter().enumerate() {
        if me.r.len() != dims.ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "ME input r length mismatch at accumulator #{}: expected ell_n = {}, got {}",
                idx,
                dims.ell_n,
                me.r.len()
            )));
        }
    }

    // Initial sum: use the public T computed from ME inputs and α
    let initial_sum = crate::paper_exact_engine::claimed_initial_sum_from_inputs(s, &ch, me_inputs);

    // Create the oracle
    let sparse = Arc::new(super::oracle::SparseCache::build(s));
    let oracle = super::oracle::OptimizedOracle::new_with_sparse(
        s,
        params,
        mcs_witnesses,
        me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        me_inputs.first().map(|mi| mi.r.as_slice()),
        sparse,
    );

    Ok(CcsBatchContext {
        oracle,
        challenges: ch,
        ell_n: dims.ell_n,
        ell_d: dims.ell_d,
        d_sc: dims.d_sc,
        initial_sum,
    })
}

/// Finalize CCS proof after batched sum-check.
///
/// This function completes the CCS proof after time-rounds have been executed
/// via batched sum-check:
/// 1. Run remaining Ajtai rounds (if not already done in batch)
/// 2. Build ME outputs using the time-round challenges (r_prime)
/// 3. Construct the proof structure
///
/// # Arguments
/// - `tr`: Transcript (should be in same state as after batched sum-check)
/// - `context`: The batch context from `prepare_ccs_for_batch`
/// - `time_challenges`: Challenges from the batched sum-check (length = ell_n)
/// - `time_rounds`: Round polynomials from the batched sum-check for CCS
/// - `running_sum`: The running sum after time-rounds
/// - Additional arguments for ME output construction
pub fn finalize_ccs_after_batch<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[neo_ccs::Mat<F>],
    mut context: CcsBatchContext<'_, F>,
    time_challenges: &[K],
    time_rounds: Vec<Vec<K>>,
    running_sum: K,
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, super::PiCcsProof), PiCcsError> {
    // Validate time challenges length
    if time_challenges.len() != context.ell_n {
        return Err(PiCcsError::InvalidInput(format!(
            "time_challenges length mismatch: expected {}, got {}",
            context.ell_n,
            time_challenges.len()
        )));
    }

    // The oracle should already have time-rounds folded in.
    // Now run remaining Ajtai rounds.
    let mut sumcheck_rounds = time_rounds;
    let mut sumcheck_chals: Vec<K> = time_challenges.to_vec();
    let mut current_sum = running_sum;

    // Continue with Ajtai rounds
    for _round_idx in 0..context.ell_d {
        let deg = context.oracle.degree_bound();
        let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
        let ys = context.oracle.evals_at(&xs);

        // Check invariant
        if ys[0] + ys[1] != current_sum {
            return Err(PiCcsError::ProtocolError(
                "Ajtai round sumcheck invariant failed".into(),
            ));
        }

        // Interpolate to coefficients
        let coeffs = crate::sumcheck::interpolate_from_evals(&xs, &ys);

        // Append to transcript and get challenge
        for c in coeffs.iter() {
            tr.append_fields(b"sumcheck/round", &c.as_coeffs());
        }
        let c0 = tr.challenge_field(b"sumcheck/challenge/0");
        let c1 = tr.challenge_field(b"sumcheck/challenge/1");
        let r_i = neo_math::from_complex(c0, c1);
        sumcheck_chals.push(r_i);

        // Fold oracle
        current_sum = crate::sumcheck::poly_eval_k(&coeffs, r_i);
        context.oracle.fold(r_i);
        sumcheck_rounds.push(coeffs);
    }

    // Build ME outputs
    let fold_digest = tr.digest32();
    let _ = (params, s, mcs_witnesses, me_witnesses);
    let out_me = context
        .oracle
        .build_me_outputs_from_ajtai_precomp(mcs_list, me_inputs, fold_digest, log);

    let mut proof = super::PiCcsProof::new(sumcheck_rounds, Some(context.initial_sum));
    proof.sumcheck_challenges = sumcheck_chals;
    proof.challenges_public = context.challenges;
    proof.sumcheck_final = current_sum;
    proof.header_digest = fold_digest.to_vec();

    Ok((out_me, proof))
}
