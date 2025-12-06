//! Cross-check engine wrapper for validation during development.
//!
//! This module provides the CrossCheckEngine, which runs the optimized engine
//! and validates key identities against paper-exact helpers. This is useful
//! for debugging and ensuring correctness.

#![allow(non_snake_case)]

use crate::engines::optimized_engine::PiCcsProof;
use crate::engines::PiCcsEngine;
use crate::error::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, Mat, McsInstance, McsWitness, MeInstance};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use super::logging::*;

/// Runtime cross-check configuration.
#[derive(Clone, Debug, Default)]
pub struct CrosscheckCfg {
    pub fail_fast: bool,
    pub initial_sum: bool,
    pub per_round: bool,
    pub terminal: bool,
    pub outputs: bool,
}

/// Cross-check engine wrapper that runs the optimized engine
/// and validates key identities against paper-exact helpers.
#[derive(Clone, Debug)]
pub struct CrossCheckEngine<I, R> {
    pub inner: I,      // Optimized engine
    pub ref_oracle: R, // PaperExact engine (reference)
    pub cfg: CrosscheckCfg,
}

/// Implementation of prove logic for cross-checking.
///
/// # How Crosscheck Works
///
/// 1. **Runs the optimized engine** to produce a proof
///    - Any transcript dumps you see are from the optimized prover
///    - The proof contains sumcheck rounds and a final sumcheck value
///
/// 2. **Extracts challenges** by replaying the optimized engine's transcript
///    - This gives us the random points (r', α') used in the proof
///
/// 3. **Validates specific computations** by comparing:
///    - Optimized engine's formulas (efficient, from public outputs)
///    - Paper-exact engine's formulas (direct, from witnesses)
///
/// The paper-exact engine never produces a full proof - it only computes
/// specific values for comparison against the optimized engine's proof.
pub fn crosscheck_prove<I, L>(
    inner: &I,
    cfg: &CrosscheckCfg,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError>
where
    I: PiCcsEngine,
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
{
    // 1) Run optimized path
    let (out_me_opt, proof) = inner.prove(tr, params, s, mcs_list, mcs_witnesses, me_inputs, me_witnesses, log)?;

    // 2) Use the prover-recorded challenges as the single source of truth.
    //    This avoids tiny transcript replays from drifting and ensures that
    //    the r we embed into outputs matches the one used to build them.
    let dims = crate::engines::utils::build_dims_and_policy(params, s)?;
    assert_eq!(
        proof.sumcheck_challenges.len(),
        dims.ell_n + dims.ell_d,
        "sumcheck_challenges length mismatch"
    );
    let (r_prime, alpha_prime) = proof.sumcheck_challenges.split_at(dims.ell_n);

    // 3) Optional cross-checks
    if cfg.initial_sum {
        let lhs_exact = crate::engines::paper_exact_engine::sum_q_over_hypercube_paper_exact(
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            &proof.challenges_public,
            dims.ell_d,
            dims.ell_n,
            me_inputs.get(0).map(|mi| mi.r.as_slice()),
        );
        if let Some(initial_sum_prover) = proof.sc_initial_sum {
            if lhs_exact != initial_sum_prover {
                return Err(PiCcsError::ProtocolError(
                    "crosscheck: initial sum mismatch (optimized vs paper-exact)".into(),
                ));
            }
        }
    }

    if cfg.per_round {
        #[cfg(feature = "paper-exact")]
        {
            use crate::sumcheck::RoundOracle;

            let detailed_log = std::env::var("NEO_CROSSCHECK_DETAIL").is_ok();

            if detailed_log {
                log_per_round_header(proof.sumcheck_rounds.len());
            }

            let mut paper_oracle = crate::engines::paper_exact_engine::oracle::PaperExactOracle::new(
                s,
                params,
                mcs_witnesses,
                me_witnesses,
                proof.challenges_public.clone(),
                dims.ell_d,
                dims.ell_n,
                dims.d_sc,
                me_inputs.first().map(|mi| mi.r.as_slice()),
            );

            for (round_idx, (opt_coeffs, &challenge)) in proof
                .sumcheck_rounds
                .iter()
                .zip(proof.sumcheck_challenges.iter())
                .enumerate()
            {
                let deg = paper_oracle.degree_bound();
                let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
                let paper_evals = paper_oracle.evals_at(&xs);

                if detailed_log {
                    log_per_round_progress(round_idx, proof.sumcheck_rounds.len(), paper_evals.len());
                }

                for (i, (&x, &expected)) in xs.iter().zip(paper_evals.iter()).enumerate() {
                    let actual = crate::sumcheck::poly_eval_k(opt_coeffs, x);
                    if actual != expected {
                        log_per_round_mismatch(round_idx, proof.sumcheck_rounds.len(), i, actual, expected);
                        if cfg.fail_fast {
                            return Err(PiCcsError::ProtocolError(format!(
                                "crosscheck: round {} polynomial mismatch at x={}",
                                round_idx, i
                            )));
                        }
                    }
                }

                paper_oracle.fold(challenge);
            }

            if detailed_log {
                log_per_round_success(proof.sumcheck_rounds.len());
            }
        }

        #[cfg(not(feature = "paper-exact"))]
        {
            log_paper_exact_feature_warning();
        }
    }

    if cfg.terminal {
        let detailed_log = std::env::var("NEO_CROSSCHECK_DETAIL").is_ok();

        if detailed_log {
            log_terminal_header();
        }

        let running_sum_prover = if let Some(initial) = proof.sc_initial_sum {
            let mut running = initial;
            for (coeffs, &ri) in proof
                .sumcheck_rounds
                .iter()
                .zip(proof.sumcheck_challenges.iter())
            {
                running = crate::sumcheck::poly_eval_k(coeffs, ri);
            }
            running
        } else {
            use crate::sumcheck::poly_eval_k;
            proof
                .sumcheck_rounds
                .get(0)
                .map(|p0| poly_eval_k(p0, K::ZERO) + poly_eval_k(p0, K::ONE))
                .unwrap_or(K::ZERO)
        };

        if detailed_log {
            log_terminal_optimized_header();
        }
        let rhs_opt = crate::paper_exact_engine::rhs_terminal_identity_paper_exact(
            s,
            params,
            &proof.challenges_public,
            r_prime,
            alpha_prime,
            &out_me_opt,
            me_inputs.first().map(|mi| mi.r.as_slice()),
        );
        if detailed_log {
            log_terminal_optimized_result(rhs_opt);
        }

        if detailed_log {
            log_terminal_paper_exact_header();
        }

        let r_inputs = me_inputs.get(0).map(|mi| mi.r.as_slice());
        let (lhs_exact, _rhs_unused) = crate::engines::paper_exact_engine::q_eval_at_ext_point_paper_exact_with_inputs(
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            alpha_prime,
            r_prime,
            &proof.challenges_public,
            r_inputs,
        );
        if detailed_log {
            log_terminal_paper_exact_result(lhs_exact);
            log_terminal_comparison(running_sum_prover, rhs_opt, lhs_exact);
        }

        if rhs_opt != lhs_exact || rhs_opt != running_sum_prover {
            log_terminal_mismatch(rhs_opt, lhs_exact, running_sum_prover);
            return Err(PiCcsError::ProtocolError(
                "crosscheck: terminal evaluation claim mismatch".into(),
            ));
        }
    }

    if cfg.outputs {
        let fold_digest: [u8; 32] = proof
            .header_digest
            .as_slice()
            .try_into()
            .unwrap_or([0u8; 32]);

        let out_me_ref = crate::engines::paper_exact_engine::build_me_outputs_paper_exact(
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

        if out_me_ref.len() != out_me_opt.len() {
            log_outputs_length_mismatch(out_me_ref.len(), out_me_opt.len());
            return Err(PiCcsError::ProtocolError("crosscheck: outputs length mismatch".into()));
        }

        for (idx, (a, b)) in out_me_ref.iter().zip(out_me_opt.iter()).enumerate() {
            let mut mismatches = Vec::new();

            if a.m_in != b.m_in {
                mismatches.push(format!("m_in: paper={}, optimized={}", a.m_in, b.m_in));
            }

            if a.r != b.r {
                let r_match_count = a.r.iter().zip(b.r.iter()).filter(|(x, y)| x == y).count();
                mismatches.push(format!(
                    "r: length paper={}, optimized={}, matching elements={}/{}",
                    a.r.len(),
                    b.r.len(),
                    r_match_count,
                    a.r.len().min(b.r.len())
                ));
                if a.r.len() == b.r.len() && !a.r.is_empty() {
                    mismatches.push(format!("  first paper r[0]={:?}", a.r.get(0)));
                    mismatches.push(format!("  first opt   r[0]={:?}", b.r.get(0)));
                }
            }

            if a.c.data != b.c.data {
                mismatches.push(format!(
                    "c.data: paper len={}, opt len={}, data_match={}",
                    a.c.data.len(),
                    b.c.data.len(),
                    a.c.data == b.c.data
                ));
                // Show first few elements for debugging
                let show_len = 4.min(a.c.data.len()).min(b.c.data.len());
                if show_len > 0 {
                    mismatches.push(format!("  paper c.data[0..{}]={:?}", show_len, &a.c.data[..show_len]));
                    mismatches.push(format!("  opt   c.data[0..{}]={:?}", show_len, &b.c.data[..show_len]));
                }
            }

            if !mismatches.is_empty() {
                log_outputs_metadata_mismatch(
                    idx,
                    out_me_ref.len(),
                    &mismatches,
                    a.m_in == b.m_in,
                    a.r == b.r,
                    a.c.data == b.c.data,
                    a.m_in,
                    a.r.len(),
                    a.c.data.len(),
                    &fold_digest,
                    mcs_list.len(),
                );

                return Err(PiCcsError::ProtocolError(format!(
                    "crosscheck: output metadata mismatch at index {}",
                    idx
                )));
            }

            for (j, (ya, yb)) in a.y.iter().zip(b.y.iter()).enumerate() {
                if ya.len() != yb.len() {
                    log_outputs_y_row_length_mismatch(idx, j, ya.len(), yb.len());
                    return Err(PiCcsError::ProtocolError(format!(
                        "crosscheck: y row {} length mismatch at instance {}",
                        j, idx
                    )));
                }
                if ya != yb {
                    let match_count = ya.iter().zip(yb.iter()).filter(|(x, y)| x == y).count();
                    let mut first_mismatch_info = None;
                    for (k, (a_val, b_val)) in ya.iter().zip(yb.iter()).enumerate() {
                        if a_val != b_val {
                            first_mismatch_info = Some((k, a_val, b_val));
                            break;
                        }
                    }
                    if let Some((k, a_val, b_val)) = first_mismatch_info {
                        log_outputs_y_row_content_mismatch(idx, j, match_count, ya.len(), k, a_val, b_val);
                    }
                    return Err(PiCcsError::ProtocolError(format!(
                        "crosscheck: y row {} content mismatch at instance {}",
                        j, idx
                    )));
                }
            }

            if a.y_scalars != b.y_scalars {
                let match_count = a
                    .y_scalars
                    .iter()
                    .zip(b.y_scalars.iter())
                    .filter(|(x, y)| x == y)
                    .count();
                let mismatches: Vec<(usize, K, K)> = a
                    .y_scalars
                    .iter()
                    .zip(b.y_scalars.iter())
                    .enumerate()
                    .filter_map(|(k, (a_val, b_val))| {
                        if a_val != b_val {
                            Some((k, *a_val, *b_val))
                        } else {
                            None
                        }
                    })
                    .collect();
                log_outputs_y_scalars_mismatch(idx, match_count, a.y_scalars.len(), &mismatches);
                return Err(PiCcsError::ProtocolError(format!(
                    "crosscheck: y_scalars mismatch at instance {}",
                    idx
                )));
            }

            // X matrix equality (dense, small D)
            if a.X.rows() != b.X.rows() || a.X.cols() != b.X.cols() {
                log_outputs_x_dimension_mismatch(idx, a.X.rows(), a.X.cols(), b.X.rows(), b.X.cols());
                return Err(PiCcsError::ProtocolError(format!(
                    "crosscheck: X dims mismatch at instance {}",
                    idx
                )));
            }
            for r in 0..a.X.rows() {
                for c in 0..a.X.cols() {
                    if a.X[(r, c)] != b.X[(r, c)] {
                        log_outputs_x_element_mismatch(idx, r, c, &a.X[(r, c)], &b.X[(r, c)]);
                        return Err(PiCcsError::ProtocolError(format!(
                            "crosscheck: X mismatch at ({},{}) in instance {}",
                            r, c, idx
                        )));
                    }
                }
            }
        }
    }

    Ok((out_me_opt, proof))
}

/// Implementation of verify logic for cross-checking.
/// Verification remains the optimized path; cross-checking is a prover-side aid.
pub fn crosscheck_verify<I>(
    inner: &I,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_outputs: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError>
where
    I: PiCcsEngine,
{
    inner.verify(tr, params, s, mcs_list, me_inputs, me_outputs, proof)
}
