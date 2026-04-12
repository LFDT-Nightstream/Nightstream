//! Paper-exact prove implementation for PiCcsEngine.
//!
//! This module contains the prove logic for the paper-exact engine,
//! which runs the sumcheck protocol using the paper-exact oracle
//! to evaluate true per-round polynomials.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use crate::optimized_engine::{PiCcsProof, PiCcsProofVariant};
use crate::sumcheck::RoundOracle;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::KExtensions;
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;

/// Paper-exact prove implementation.
///
/// This function runs the sumcheck protocol using the paper-exact oracle,
/// which directly evaluates the polynomial Q over the Boolean hypercube
/// without optimizations.
pub fn paper_exact_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    if mcs_list.is_empty() {
        return Err(PiCcsError::InvalidInput("paper_exact_prove: empty mcs_list".into()));
    }
    if mcs_list.len() != mcs_witnesses.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "paper_exact_prove: |mcs_list| mismatch (expected {}, got {})",
            mcs_list.len(),
            mcs_witnesses.len()
        )));
    }
    if me_inputs.len() != me_witnesses.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "paper_exact_prove: |me_inputs| mismatch (expected {}, got {})",
            me_inputs.len(),
            me_witnesses.len()
        )));
    }

    // Dims + transcript binding
    let dims = crate::engines::utils::build_dims_and_policy(params, s)?;
    crate::engines::utils::bind_header_and_instances(tr, params, s, mcs_list, dims)?;
    crate::engines::utils::bind_me_inputs(tr, me_inputs)?;

    // Sample challenges
    let mut ch = crate::engines::utils::sample_challenges(tr, dims.ell_d, dims.ell)?;
    ch.beta_m = crate::engines::utils::sample_beta_m(tr, dims.ell_m)?;

    let r_inputs = crate::engines::utils::shared_me_input_r(me_inputs, dims.ell_n)?;

    // Initial sum: use the public T computed from ME inputs and α
    // (not the full hypercube sum Q, which includes MCS/NC terms).
    // This ensures invalid witnesses fail the first sumcheck invariant.
    let initial_sum =
        crate::paper_exact_engine::claimed_initial_sum_from_inputs_with_k_mcs(s, &ch, mcs_witnesses.len(), me_inputs);

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

    // ---------------------------------------------------------------------
    // FE sumcheck channel (SplitNcV1).
    // ---------------------------------------------------------------------
    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)]);
    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    tr.append_fields_raw(&initial_sum.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(crate::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);

    // Paper-exact oracle to evaluate true per-round polynomials
    let mut oracle = crate::paper_exact_engine::oracle::PaperExactOracle::new(
        s,
        params,
        mcs_witnesses,
        me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        r_inputs,
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

        let coeff_fields = crate::sumcheck::round_coeff_fields(&coeffs);
        tr.append_fields_raw(&coeff_fields);
        let c = tr.challenge_fields_raw(2);
        let r_i = neo_math::from_complex(c[0], c[1]);
        sumcheck_chals.push(r_i);

        // Evaluate at challenge using poly_eval_k (low→high) for consistency.
        running_sum = crate::sumcheck::poly_eval_k(&coeffs, r_i);

        oracle.fold(r_i);
        sumcheck_rounds.push(coeffs);
    }

    // ---------------------------------------------------------------------
    // NC-only sumcheck (split-NC scaffolding; claimed sum is 0)
    // ---------------------------------------------------------------------
    let mut oracle_nc = crate::optimized_engine::oracle::NcOracle::new(
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

        let coeff_fields = crate::sumcheck::round_coeff_fields(&coeffs);
        tr.append_fields_raw(&coeff_fields);
        let c = tr.challenge_fields_raw(2);
        let r_i = neo_math::from_complex(c[0], c[1]);
        sumcheck_chals_nc.push(r_i);

        running_sum_nc = crate::sumcheck::poly_eval_k(&coeffs, r_i);
        oracle_nc.fold(r_i);
        sumcheck_rounds_nc.push(coeffs);
    }

    // Build outputs literally at r′ using paper-exact helper
    let fold_digest = tr.digest32();
    let (r_prime, _alpha_prime) = sumcheck_chals.split_at(dims.ell_n);
    let (s_col, _alpha_nc) = sumcheck_chals_nc.split_at(dims.ell_m);
    let out_me = crate::paper_exact_engine::build_me_outputs_paper_exact(
        s,
        params,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        r_prime,
        s_col,
        dims.ell_d,
        fold_digest,
        log,
    );

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
