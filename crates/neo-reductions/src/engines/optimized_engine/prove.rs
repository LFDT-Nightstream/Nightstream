//! Optimized prove implementation for PiCcsEngine.
//!
//! This module contains the prove logic for the optimized engine, using
//! sparse/oracle optimizations while preserving paper-equivalent semantics.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::optimized_engine::{PiCcsProof, PiCcsProofVariant};
use crate::sumcheck::RoundOracle;
use crate::{accelerator, accelerator::SplitNcOptimizedOracle};
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_gpu::ProverComputeBackend;
use neo_math::KExtensions;
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;
use std::sync::Arc;

use crate::engines::utils;

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
    optimized_prove_with_backend(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        &ProverComputeBackend::Cpu,
    )
}

/// Optimized prove implementation with a backend-aware Split-NC oracle seam.
pub fn optimized_prove_with_backend<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    compute_backend: &ProverComputeBackend,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    let backend_ctx = accelerator::BackendContext::new(compute_backend)?;
    optimized_prove_with_context(
        tr,
        params,
        s,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        log,
        &backend_ctx,
    )
}

/// Optimized prove implementation with a reusable backend context.
pub fn optimized_prove_with_context<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
    backend_ctx: &accelerator::BackendContext,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
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
    let dims = utils::build_dims_and_policy(params, s)?;
    utils::bind_header_and_instances(tr, params, s, mcs_list, dims)?;
    utils::bind_me_inputs_with_context(tr, me_inputs, &backend_ctx)?;

    // Sample challenges
    let mut ch = utils::sample_challenges(tr, dims.ell_d, dims.ell)?;
    ch.beta_m = utils::sample_beta_m(tr, dims.ell_m)?;

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
    let sparse = Arc::new(super::oracle::SparseCache::build(s));
    let mut oracle = SplitNcOptimizedOracle::new_with_sparse(
        s,
        params,
        mcs_witnesses,
        me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        r_inputs,
        sparse,
        &backend_ctx,
    )?;

    // ---------------------------------------------------------------------
    // FE sumcheck channel (SplitNcV1).
    // ---------------------------------------------------------------------
    tr.append_message(b"sumcheck/fe", b"");
    tr.append_fields(b"sumcheck/initial_sum", &initial_sum.as_coeffs());

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

    // ---------------------------------------------------------------------
    // NC-only sumcheck (split-NC scaffolding; claimed sum is 0)
    // ---------------------------------------------------------------------
    let mut oracle_nc = accelerator::SplitNcNcOracle::new(
        s,
        params,
        mcs_witnesses,
        me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_m,
        dims.d_sc,
        &backend_ctx,
    )?;

    tr.append_message(b"sumcheck/nc", b"");
    let initial_sum_nc = K::ZERO;
    tr.append_fields(b"sumcheck/initial_sum", &initial_sum_nc.as_coeffs());

    let mut running_sum_nc = initial_sum_nc;
    let mut sumcheck_rounds_nc: Vec<Vec<K>> = Vec::with_capacity(oracle_nc.num_rounds());
    let mut sumcheck_chals_nc: Vec<K> = Vec::with_capacity(oracle_nc.num_rounds());

    for _round_idx in 0..oracle_nc.num_rounds() {
        let deg = oracle_nc.degree_bound();
        let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
        let ys = oracle_nc.evals_at(&xs);

        if ys[0] + ys[1] != running_sum_nc {
            return Err(PiCcsError::SumcheckError(
                "NC sumcheck invariant failed: p(0)+p(1) ≠ running_sum".into(),
            ));
        }

        let coeffs = crate::sumcheck::interpolate_from_evals(&xs, &ys);
        debug_assert_eq!(crate::sumcheck::poly_eval_k(&coeffs, K::ZERO), ys[0]);
        debug_assert_eq!(crate::sumcheck::poly_eval_k(&coeffs, K::ONE), ys[1]);

        for &c in &coeffs {
            tr.append_fields(b"sumcheck/round/coeff", &c.as_coeffs());
        }
        let c0 = tr.challenge_field(b"sumcheck/challenge/0");
        let c1 = tr.challenge_field(b"sumcheck/challenge/1");
        let r_i = neo_math::from_complex(c0, c1);
        sumcheck_chals_nc.push(r_i);

        running_sum_nc = crate::sumcheck::poly_eval_k(&coeffs, r_i);
        oracle_nc.fold(r_i);
        sumcheck_rounds_nc.push(coeffs);
    }

    // Build outputs at r′ using the oracle's r′-only precomputation (no dense scan).
    let fold_digest = tr.digest32();
    let (s_col, _alpha_nc) = sumcheck_chals_nc.split_at(dims.ell_m);
    let out_me = oracle.build_me_outputs_from_ajtai_precomp(mcs_list, me_inputs, s_col, fold_digest, log);

    let mut proof = PiCcsProof::new(sumcheck_rounds, Some(initial_sum));
    proof.variant = PiCcsProofVariant::SplitNcV1;
    proof.sumcheck_challenges = sumcheck_chals;
    proof.sumcheck_rounds_nc = sumcheck_rounds_nc;
    proof.sc_initial_sum_nc = Some(initial_sum_nc);
    proof.sumcheck_challenges_nc = sumcheck_chals_nc;
    proof.challenges_public = ch;
    proof.sumcheck_final = running_sum;
    proof.sumcheck_final_nc = running_sum_nc;
    proof.header_digest = fold_digest.to_vec();

    Ok((out_me, proof))
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
