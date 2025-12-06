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

/// Paper-exact prove implementation.
///
/// This function runs the sumcheck protocol using the paper-exact oracle,
/// which directly evaluates the polynomial Q over the Boolean hypercube
/// without optimizations.
pub fn paper_exact_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
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
    let dims = crate::engines::utils::build_dims_and_policy(params, s)?;
    crate::engines::utils::bind_header_and_instances(tr, params, s, mcs_list, dims.ell, dims.d_sc, 0)?;
    crate::engines::utils::bind_me_inputs(tr, me_inputs)?;

    // Sample challenges
    let ch = crate::engines::utils::sample_challenges(tr, dims.ell_d, dims.ell)?;

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
        me_inputs.first().map(|mi| mi.r.as_slice()),
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
        // Interpolation may return coefficients in an unspecified order.
        // We MUST store them in low→high order (c0, c1, ..., cn) so that
        // poly_eval_k(coeffs, ·) reproduces ys at x=0,1 and the verifier's
        // invariant p(0)+p(1) == running_sum holds.
        let mut coeffs = crate::optimized_engine::interpolate_univariate(&xs, &ys);

        // If evaluating at 0/1 doesn't recreate ys, flip the order.
        // After this normalization, `coeffs` is guaranteed low→high.
        let ok_at_01 = crate::sumcheck::poly_eval_k(&coeffs, K::ZERO) == ys[0]
            && crate::sumcheck::poly_eval_k(&coeffs, K::ONE) == ys[1];
        if !ok_at_01 {
            coeffs.reverse();
        }

        debug_assert_eq!(
            crate::sumcheck::poly_eval_k(&coeffs, K::ZERO),
            ys[0],
            "interpolate_univariate returned coefficients in unexpected order (x=0)"
        );
        debug_assert_eq!(
            crate::sumcheck::poly_eval_k(&coeffs, K::ONE),
            ys[1],
            "interpolate_univariate returned coefficients in unexpected order (x=1)"
        );

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

    // Build outputs literally at r′ using paper-exact helper
    let fold_digest = tr.digest32();
    let (r_prime, _alpha_prime) = sumcheck_chals.split_at(dims.ell_n);
    let out_me = crate::paper_exact_engine::build_me_outputs_paper_exact(
        s,
        params,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        r_prime,
        dims.ell_d,
        fold_digest,
        log,
    );

    let mut proof = PiCcsProof::new(sumcheck_rounds, Some(initial_sum));
    proof.sumcheck_challenges = sumcheck_chals;
    proof.challenges_public = ch;
    proof.sumcheck_final = running_sum;
    proof.header_digest = fold_digest.to_vec();

    Ok((out_me, proof))
}
