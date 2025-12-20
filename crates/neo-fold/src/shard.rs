//! Shard-level folding: CPU (CCS) + Memory sidecar (Twist/Shout).
//!
//! Architecture (per integration-summary.md):
//!
//! At each folding step, aggregate:
//! 1. (k) running ME instances
//! 2. fresh ME instances from Π_CCS
//! 3. fresh ME instances from Π_Twist and Π_Shout
//! 4. fresh ME instances from Index→OneHot adapter
//!
//! Then: (all ME claims) → Π_RLC → Π_DEC → k_rho children
//!
//! CURRENT IMPLEMENTATION (simplified):
//! - CPU folding runs first (Π_CCS → Π_RLC → Π_DEC per step)
//! - Memory sidecar proved after CPU folding
//! - Final merge combines CPU output + memory ME via Π_RLC → Π_DEC
//!
//! TODO: True per-step integration requires per-step memory instances.

#![allow(non_snake_case)]

use crate::folding::{CommitMixers, FoldStep};
use crate::memory_sidecar::utils::RoundOraclePrefix;
use crate::pi_ccs::{self as ccs, FoldingMode};
pub use crate::shard_proof_types::{
    BatchedTimeProof, MemOrLutProof, MemSidecarProof, RlcDecProof, ShardProof, ShoutProofK, StepProof, TwistProofK,
};
use crate::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{from_complex, KExtensions, D, F, K};
use neo_memory::ts_common as ts;
use neo_memory::twist_oracle::table_mle_eval;
use neo_memory::witness::StepWitnessBundle;
use neo_params::NeoParams;
use neo_reductions::engines::utils;
use neo_reductions::paper_exact_engine::{
    build_me_outputs_paper_exact, claimed_initial_sum_from_inputs, rhs_terminal_identity_paper_exact,
};
use neo_reductions::sumcheck::{
    interpolate_from_evals, poly_eval_k, verify_batched_sumcheck_rounds, verify_sumcheck_rounds, BatchedClaim,
    RoundOracle,
};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

// ============================================================================
// Utilities
// ============================================================================

pub use crate::memory_sidecar::memory::absorb_step_memory_commitments;

pub fn normalize_me_claims(
    me_claims: &mut [MeInstance<Cmt, F, K>],
    ell_n: usize,
    ell_d: usize,
    t: usize,
) -> Result<(), PiCcsError> {
    let y_pad = 1usize << ell_d;
    for (i, me) in me_claims.iter_mut().enumerate() {
        if me.r.len() != ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] r.len()={}, expected ell_n={}",
                i,
                me.r.len(),
                ell_n
            )));
        }
        if me.y.len() > t {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] y.len()={}, expected <= t={}",
                i,
                me.y.len(),
                t
            )));
        }
        for (j, row) in me.y.iter_mut().enumerate() {
            if row.len() > y_pad {
                return Err(PiCcsError::InvalidInput(format!(
                    "ME[{}] y[{}].len()={}, expected <= {}",
                    i,
                    j,
                    row.len(),
                    y_pad
                )));
            }
            row.resize(y_pad, K::ZERO);
        }
        me.y.resize_with(t, || vec![K::ZERO; y_pad]);
        if me.y_scalars.len() > t {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] y_scalars.len()={}, expected <= t={}",
                i,
                me.y_scalars.len(),
                t
            )));
        }
        me.y_scalars.resize(t, K::ZERO);
    }
    Ok(())
}

fn bind_batched_dynamic_claims(
    tr: &mut Poseidon2Transcript,
    claimed_sums: &[K],
    labels: &[&[u8]],
    claim_is_dynamic: &[bool],
) {
    debug_assert_eq!(claimed_sums.len(), labels.len());
    debug_assert_eq!(claimed_sums.len(), claim_is_dynamic.len());

    for (idx, ((sum, label), dyn_ok)) in claimed_sums
        .iter()
        .zip(labels.iter())
        .zip(claim_is_dynamic.iter())
        .enumerate()
    {
        if !*dyn_ok {
            continue;
        }
        tr.append_message(b"batched/claim_label", label);
        tr.append_message(b"batched/claim_idx", &(idx as u64).to_le_bytes());
        tr.append_fields(b"batched/claimed_sum", &sum.as_coeffs());
    }
}

fn validate_me_batch_invariants(batch: &[MeInstance<Cmt, F, K>], context: &str) -> Result<(), PiCcsError> {
    if batch.is_empty() {
        return Ok(());
    }
    let me0 = &batch[0];
    let r0 = &me0.r;
    let m_in0 = me0.m_in;
    let y_len0 = me0.y.len();
    let y_row_len0 = me0.y.first().map(|r| r.len()).unwrap_or(0);
    let y_scalars_len0 = me0.y_scalars.len();

    if me0.X.rows() != D {
        return Err(PiCcsError::ProtocolError(format!(
            "{}: ME claim 0 has X.rows()={}, expected D={}",
            context,
            me0.X.rows(),
            D
        )));
    }
    if me0.X.cols() != m_in0 {
        return Err(PiCcsError::ProtocolError(format!(
            "{}: ME claim 0 has X.cols()={}, expected m_in={}",
            context,
            me0.X.cols(),
            m_in0
        )));
    }

    for (i, me) in batch.iter().enumerate().skip(1) {
        if me.r != *r0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has different r than claim 0 (r-alignment required for RLC)",
                context, i
            )));
        }
        if me.m_in != m_in0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has m_in={}, expected {}",
                context, i, me.m_in, m_in0
            )));
        }
        if me.X.rows() != D || me.X.cols() != m_in0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has X shape {}x{}, expected {}x{}",
                context,
                i,
                me.X.rows(),
                me.X.cols(),
                D,
                m_in0
            )));
        }
        if me.y.len() != y_len0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has y.len()={}, expected {}",
                context,
                i,
                me.y.len(),
                y_len0
            )));
        }
        for (j, row) in me.y.iter().enumerate() {
            if row.len() != y_row_len0 {
                return Err(PiCcsError::ProtocolError(format!(
                    "{}: ME claim {} has y[{}].len()={}, expected {}",
                    context,
                    i,
                    j,
                    row.len(),
                    y_row_len0
                )));
            }
        }
        if me.y_scalars.len() != y_scalars_len0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has y_scalars.len()={}, expected {}",
                context,
                i,
                me.y_scalars.len(),
                y_scalars_len0
            )));
        }
    }
    Ok(())
}

// ============================================================================
// Shard Proving
// ============================================================================

pub fn fold_shard_prove<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let s = s_me
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    // Route A terminal checks interpret `ME.y_scalars[0]` as MLE(column)(r_time), which requires M₀ = I.
    s.assert_m0_is_identity_for_nc()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first (M₀=I) required: {e:?}")))?;
    let utils::Dims {
        ell_d,
        ell_n,
        ell,
        d_sc,
    } = utils::build_dims_and_policy(params, &s)?;
    let k_dec = params.k_rho as usize;
    let ring = ccs::RotRing::goldilocks();

    // Initialize accumulator
    let mut accumulator = acc_init.to_vec();
    let mut accumulator_wit = acc_wit_init.to_vec();

    let mut step_proofs = Vec::with_capacity(steps.len());

    for (idx, step) in steps.iter().enumerate() {
        absorb_step_memory_commitments(tr, step);

        let (mcs_inst, mcs_wit) = &step.mcs;

        // k = accumulator.len() + 1
        let k = accumulator.len() + 1;

        // --------------------------------------------------------------------
        // Route A: Shared-challenge batched sum-check for time/row rounds.
        // --------------------------------------------------------------------
        //
        // 1) Bind CCS header + ME inputs
        // 2) Sample CCS challenges (α, β, γ) and bind initial sum
        // 3) Build CCS oracle + lazy Twist/Shout oracles
        // 4) Run ONE batched sum-check for the first ell_n rounds (row/time)
        // 5) Finish CCS alone for remaining ell_d Ajtai rounds
        // 6) Emit CCS + memory ME claims at the shared r_time and fold via RLC/DEC

        utils::bind_header_and_instances(tr, params, &s, core::slice::from_ref(mcs_inst), ell, d_sc, 0)?;
        utils::bind_me_inputs(tr, &accumulator)?;
        let ch = utils::sample_challenges(tr, ell_d, ell)?;
        let ccs_initial_sum = claimed_initial_sum_from_inputs(&s, &ch, &accumulator);
        tr.append_fields(b"sumcheck/initial_sum", &ccs_initial_sum.as_coeffs());

        // Route A memory checks use a separate transcript-derived cycle point `r_cycle`
        // to form χ_{r_cycle}(t) weights inside their sum-check polynomials.
        let r_cycle: Vec<K> =
            ts::sample_ext_point(tr, b"route_a/r_cycle", b"route_a/cycle/0", b"route_a/cycle/1", ell_n);

        // CCS oracle (engine-selected)
        let mut ccs_oracle: Box<dyn RoundOracle> = match mode.clone() {
            FoldingMode::Optimized => {
                Box::new(neo_reductions::engines::optimized_engine::oracle::OptimizedOracle::new(
                    &s,
                    params,
                    core::slice::from_ref(mcs_wit),
                    &accumulator_wit,
                    ch.clone(),
                    ell_d,
                    ell_n,
                    d_sc,
                    accumulator.first().map(|mi| mi.r.as_slice()),
                ))
            }
            #[cfg(feature = "paper-exact")]
            FoldingMode::PaperExact => Box::new(
                neo_reductions::engines::paper_exact_engine::oracle::PaperExactOracle::new(
                    &s,
                    params,
                    core::slice::from_ref(mcs_wit),
                    &accumulator_wit,
                    ch.clone(),
                    ell_d,
                    ell_n,
                    d_sc,
                    accumulator.first().map(|mi| mi.r.as_slice()),
                ),
            ),
            #[cfg(feature = "paper-exact")]
            FoldingMode::OptimizedWithCrosscheck(_) => {
                Box::new(neo_reductions::engines::optimized_engine::oracle::OptimizedOracle::new(
                    &s,
                    params,
                    core::slice::from_ref(mcs_wit),
                    &accumulator_wit,
                    ch.clone(),
                    ell_d,
                    ell_n,
                    d_sc,
                    accumulator.first().map(|mi| mi.r.as_slice()),
                ))
            }
        };

        let shout_pre = crate::memory_sidecar::memory::prove_shout_addr_pre_time(tr, params, step, ell_n, &r_cycle)?;
        let twist_decoded = crate::memory_sidecar::memory::decode_twist_pre_time(params, step, ell_n)?;
        let mut mem_oracles = crate::memory_sidecar::memory::build_route_a_memory_oracles(
            params, step, ell_n, &r_cycle, &shout_pre, &twist_decoded,
        )?;

        // Run the batched time/row sumcheck in a tight scope to satisfy Rust borrow rules.
        let (r_time, per_claim_results, batched_time) = {
            let mut claimed_sums: Vec<K> = Vec::new();
            let mut degree_bounds: Vec<usize> = Vec::new();
            let mut labels: Vec<&'static [u8]> = Vec::new();
            let mut claim_is_dynamic: Vec<bool> = Vec::new();
            let mut claims: Vec<BatchedClaim<'_>> = Vec::new();

            // CCS claim (time/row rounds only)
            let mut ccs_time = RoundOraclePrefix::new(ccs_oracle.as_mut(), ell_n);
            claimed_sums.push(ccs_initial_sum);
            degree_bounds.push(ccs_time.degree_bound());
            labels.push(b"ccs/time");
            // Keep CCS/time claimed sum in the dynamic-claim registry for transcript consistency.
            claim_is_dynamic.push(true);
            claims.push(BatchedClaim {
                oracle: &mut ccs_time,
                claimed_sum: ccs_initial_sum,
                label: b"ccs/time",
            });

            let mut shout_guard =
                crate::memory_sidecar::memory::build_route_a_shout_time_claims_guard(&mut mem_oracles.shout, ell_n);
            crate::memory_sidecar::memory::append_route_a_shout_time_claims(
                &mut shout_guard,
                &mut claimed_sums,
                &mut degree_bounds,
                &mut labels,
                &mut claim_is_dynamic,
                &mut claims,
            );
            let mut twist_guard =
                crate::memory_sidecar::memory::build_route_a_twist_time_claims_guard_v2(&mut mem_oracles.twist, ell_n);
            crate::memory_sidecar::memory::append_route_a_twist_time_claims_v2(
                &mut twist_guard,
                &mut claimed_sums,
                &mut degree_bounds,
                &mut labels,
                &mut claim_is_dynamic,
                &mut claims,
            );

            #[cfg(debug_assertions)]
            {
                let mut exp_degree_bounds: Vec<usize> = Vec::new();
                let mut exp_labels: Vec<&[u8]> = Vec::new();
                let mut exp_dynamic: Vec<bool> = Vec::new();

                exp_degree_bounds.push(d_sc);
                exp_labels.push(b"ccs/time" as &[u8]);
                exp_dynamic.push(true);

                crate::memory_sidecar::memory::append_expected_batched_time_metadata_for_memory(
                    step,
                    &mut exp_degree_bounds,
                    &mut exp_labels,
                    &mut exp_dynamic,
                );

                debug_assert_eq!(degree_bounds, exp_degree_bounds, "batched time degree bounds drift");
                debug_assert_eq!(labels.len(), exp_labels.len(), "batched time labels length drift");
                for (i, (got, exp)) in labels.iter().zip(exp_labels.iter()).enumerate() {
                    debug_assert_eq!(*got, *exp, "batched time label drift at claim {}", i);
                }
                debug_assert_eq!(claim_is_dynamic, exp_dynamic, "batched time dynamic-flag drift");
            }

            // Run batched sum-check prover (shared r_time challenges)
            bind_batched_dynamic_claims(tr, &claimed_sums, &labels, &claim_is_dynamic);
            let (r_time, per_claim_results) =
                neo_reductions::sumcheck::run_batched_sumcheck_prover(tr, claims.as_mut_slice())
                    .map_err(|e| PiCcsError::SumcheckError(format!("batched time sumcheck: {e}")))?;

            if r_time.len() != ell_n {
                return Err(PiCcsError::ProtocolError(format!(
                    "batched sumcheck returned r_time.len()={}, expected ell_n={ell_n}",
                    r_time.len()
                )));
            }

            let batched_time = BatchedTimeProof {
                claimed_sums: claimed_sums.clone(),
                degree_bounds: degree_bounds.clone(),
                labels: labels.clone(),
                round_polys: per_claim_results
                    .iter()
                    .map(|r| r.round_polys.clone())
                    .collect(),
            };

            Ok::<_, PiCcsError>((r_time, per_claim_results, batched_time))
        }?;

        // Finish CCS Ajtai rounds alone, continuing from the CCS oracle state after ell_n folds.
        let ccs_time_rounds = per_claim_results
            .first()
            .map(|r| r.round_polys.clone())
            .unwrap_or_default();
        let mut sumcheck_rounds = ccs_time_rounds;
        let mut sumcheck_chals = r_time.clone();
        let mut running_sum = per_claim_results
            .first()
            .map(|r| r.final_value)
            .unwrap_or(ccs_initial_sum);

        for ajtai_round in 0..ell_d {
            let deg = ccs_oracle.degree_bound();
            let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
            let ys = ccs_oracle.evals_at(&xs);

            let sum_at_01 = ys[0] + ys[1];
            if sum_at_01 != running_sum {
                return Err(PiCcsError::ProtocolError(format!(
                    "CCS Ajtai round {} invariant failed at step {}: expected {}, got {}",
                    ajtai_round, idx, running_sum, sum_at_01
                )));
            }

            let coeffs = interpolate_from_evals(&xs, &ys);
            for &c in coeffs.iter() {
                tr.append_fields(b"sumcheck/round/coeff", &c.as_coeffs());
            }
            let c0 = tr.challenge_field(b"sumcheck/challenge/0");
            let c1 = tr.challenge_field(b"sumcheck/challenge/1");
            let r_i = from_complex(c0, c1);
            sumcheck_chals.push(r_i);

            running_sum = poly_eval_k(&coeffs, r_i);
            ccs_oracle.fold(r_i);
            sumcheck_rounds.push(coeffs);
        }

        // CCS oracle borrows accumulator_wit; drop before updating accumulator_wit at the end.
        drop(ccs_oracle);

        // Build CCS ME outputs at r_time
        let fold_digest = tr.digest32();
        let ccs_out = build_me_outputs_paper_exact(
            &s,
            params,
            core::slice::from_ref(mcs_inst),
            core::slice::from_ref(mcs_wit),
            &accumulator,
            &accumulator_wit,
            &r_time,
            ell_d,
            fold_digest,
            l,
        );

        if ccs_out.len() != k {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_CCS returned {} outputs; expected k={k}",
                ccs_out.len()
            )));
        }

        let mut ccs_proof = crate::PiCcsProof::new(sumcheck_rounds, Some(ccs_initial_sum));
        ccs_proof.sumcheck_challenges = sumcheck_chals;
        ccs_proof.challenges_public = ch;
        ccs_proof.sumcheck_final = running_sum;
        ccs_proof.header_digest = fold_digest.to_vec();

        // Witnesses for CCS outputs: [Z_mcs, Z_seed...]
        let mut outs_Z: Vec<Mat<F>> = Vec::with_capacity(k);
        outs_Z.push(mcs_wit.Z.clone());
        outs_Z.extend(accumulator_wit.iter().cloned());

        // Memory sidecar: emit ME claims at the shared r_time (no fixed-challenge sumcheck).
        let time_final_values: Vec<K> = per_claim_results.iter().map(|r| r.final_value).collect();
        let mut mem_out = crate::memory_sidecar::memory::finalize_route_a_memory_prover(
            tr,
            params,
            &s,
            step,
            &mut mem_oracles,
            &shout_pre,
            &twist_decoded,
            &r_time,
            &time_final_values,
            1, // claim 0 is CCS/time
            mcs_inst.m_in,
        )?;

        normalize_me_claims(&mut mem_out.mem.me_claims_time, ell_n, ell_d, s.t())?;
        normalize_me_claims(&mut mem_out.mem.me_claims_val, ell_n, ell_d, s.t())?;

        // Build RLC inputs: CCS outputs + memory ME
        let mut rlc_inputs = ccs_out.clone();
        rlc_inputs.extend(mem_out.mem.me_claims_time.clone());

        let mut rlc_wits = outs_Z.clone();
        for w in mem_out.me_wits_time.iter() {
            rlc_wits.push(ts::pad_mat_to_ccs_width(w, s.m)?);
        }

        // RLC
        let rhos = ccs::sample_rot_rhos_n(tr, params, &ring, rlc_inputs.len())?;
        let (parent_pub, Z_mix) = ccs::rlc_with_commit(
            mode.clone(),
            &s,
            params,
            &rhos,
            &rlc_inputs,
            &rlc_wits,
            ell_d,
            mixers.mix_rhos_commits,
        );

        // DEC
        let Z_split = ccs::split_b_matrix_k(&Z_mix, k_dec, params.b)?;
        let child_cs: Vec<Cmt> = Z_split.iter().map(|Zi| l.commit(Zi)).collect();
        let (children, ok_y, ok_X, ok_c) = ccs::dec_children_with_commit(
            mode.clone(),
            &s,
            params,
            &parent_pub,
            &Z_split,
            ell_d,
            &child_cs,
            mixers.combine_b_pows,
        );
        if !(ok_y && ok_X && ok_c) {
            return Err(PiCcsError::ProtocolError(format!(
                "DEC public check failed at step {} (y={}, X={}, c={})",
                idx, ok_y, ok_X, ok_c
            )));
        }

        // --------------------------------------------------------------------
        // Phase 2: Second folding lane for Twist val-eval ME claims at r_val.
        // --------------------------------------------------------------------
        let val_fold = if mem_out.mem.me_claims_val.is_empty() {
            None
        } else {
            validate_me_batch_invariants(&mem_out.mem.me_claims_val, "prove step memory val outputs")?;
            if mem_out.me_wits_val.len() != mem_out.mem.me_claims_val.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "Twist(val) witness count mismatch (have {}, need {})",
                    mem_out.me_wits_val.len(),
                    mem_out.mem.me_claims_val.len()
                )));
            }

            tr.append_message(b"fold/val_lane_start", &(idx as u64).to_le_bytes());

            let val_rhos = ccs::sample_rot_rhos_n(tr, params, &ring, mem_out.mem.me_claims_val.len())?;
            let mut val_wits: Vec<Mat<F>> = Vec::with_capacity(mem_out.me_wits_val.len());
            for w in mem_out.me_wits_val.iter() {
                val_wits.push(ts::pad_mat_to_ccs_width(w, s.m)?);
            }

            let (val_parent_pub, Z_mix) = ccs::rlc_with_commit(
                mode.clone(),
                &s,
                params,
                &val_rhos,
                &mem_out.mem.me_claims_val,
                &val_wits,
                ell_d,
                mixers.mix_rhos_commits,
            );

            let Z_split = ccs::split_b_matrix_k(&Z_mix, k_dec, params.b)?;
            let child_cs: Vec<Cmt> = Z_split.iter().map(|Zi| l.commit(Zi)).collect();
            let (val_children, ok_y, ok_X, ok_c) = ccs::dec_children_with_commit(
                mode.clone(),
                &s,
                params,
                &val_parent_pub,
                &Z_split,
                ell_d,
                &child_cs,
                mixers.combine_b_pows,
            );
            if !(ok_y && ok_X && ok_c) {
                return Err(PiCcsError::ProtocolError(format!(
                    "DEC(val) public check failed at step {} (y={}, X={}, c={})",
                    idx, ok_y, ok_X, ok_c
                )));
            }

            Some(RlcDecProof {
                rlc_rhos: val_rhos,
                rlc_parent: val_parent_pub,
                dec_children: val_children,
            })
        };

        accumulator = children.clone();
        accumulator_wit = Z_split;

        step_proofs.push(StepProof {
            fold: FoldStep {
                ccs_out,
                ccs_proof,
                rlc_rhos: rhos,
                rlc_parent: parent_pub,
                dec_children: children,
            },
            mem: mem_out.mem,
            batched_time,
            val_fold,
        });

        tr.append_message(b"fold/step_done", &(idx as u64).to_le_bytes());
    }

    Ok(ShardProof { steps: step_proofs })
}

// ============================================================================
// Shard Verification
// ============================================================================

pub fn fold_shard_verify<L, MR, MB>(
    _mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    _l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let s = s_me
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    // Route A terminal checks interpret `ME.y_scalars[0]` as MLE(column)(r_time), which requires M₀ = I.
    s.assert_m0_is_identity_for_nc()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first (M₀=I) required: {e:?}")))?;
    let utils::Dims {
        ell_d,
        ell_n,
        ell,
        d_sc,
    } = utils::build_dims_and_policy(params, &s)?;
    let ring = ccs::RotRing::goldilocks();

    if steps.len() != proof.steps.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "step count mismatch: public {} vs proof {}",
            steps.len(),
            proof.steps.len()
        )));
    }

    let mut accumulator = acc_init.to_vec();

    for (idx, (step, step_proof)) in steps.iter().zip(proof.steps.iter()).enumerate() {
        absorb_step_memory_commitments(tr, step);

        let (mcs_inst, _mcs_wit) = &step.mcs;

        // --------------------------------------------------------------------
        // Route A: Verify shared-challenge batched sum-check (time/row rounds),
        // then finish CCS Ajtai rounds, then proceed with RLC→DEC as before.
        // --------------------------------------------------------------------

        // Bind CCS header + ME inputs and sample public challenges.
        utils::bind_header_and_instances(tr, params, &s, core::slice::from_ref(mcs_inst), ell, d_sc, 0)?;
        utils::bind_me_inputs(tr, &accumulator)?;
        let ch = utils::sample_challenges(tr, ell_d, ell)?;

        // Public initial sum T for CCS sumcheck.
        let claimed_initial = claimed_initial_sum_from_inputs(&s, &ch, &accumulator);
        if let Some(x) = step_proof.fold.ccs_proof.sc_initial_sum {
            if x != claimed_initial {
                return Err(PiCcsError::SumcheckError(
                    "initial sum mismatch: proof claims different value than public T".into(),
                ));
            }
        }
        tr.append_fields(b"sumcheck/initial_sum", &claimed_initial.as_coeffs());

        // Route A memory checks use a separate transcript-derived cycle point `r_cycle`
        // to form χ_{r_cycle}(t) weights inside their sum-check polynomials.
        let r_cycle: Vec<K> =
            ts::sample_ext_point(tr, b"route_a/r_cycle", b"route_a/cycle/0", b"route_a/cycle/1", ell_n);

        // Expected batched claims metadata (do not trust proof for degree bounds/labels).
        //
        // Claimed sums are mostly fixed (0) except CCS/time (=public T).
        let mut expected_degree_bounds: Vec<usize> = Vec::new();
        let mut expected_labels: Vec<&[u8]> = Vec::new();
        let mut claim_is_dynamic: Vec<bool> = Vec::new();

        expected_degree_bounds.push(d_sc);
        expected_labels.push(b"ccs/time");
        claim_is_dynamic.push(true); // fixed to public T below
        crate::memory_sidecar::memory::append_expected_batched_time_metadata_for_memory(
            step,
            &mut expected_degree_bounds,
            &mut expected_labels,
            &mut claim_is_dynamic,
        );

        let expected_claims = claim_is_dynamic.len();
        if step_proof.batched_time.round_polys.len() != expected_claims {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: batched_time claim count mismatch (expected {}, got {})",
                idx,
                expected_claims,
                step_proof.batched_time.round_polys.len()
            )));
        }
        if step_proof.batched_time.claimed_sums.len() != expected_claims {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: batched_time claimed_sums.len() mismatch (expected {}, got {})",
                idx,
                expected_claims,
                step_proof.batched_time.claimed_sums.len()
            )));
        }
        if step_proof.batched_time.claimed_sums.is_empty() || step_proof.batched_time.claimed_sums[0] != claimed_initial
        {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: batched_time claimed_sums[0] (CCS/time) != public initial sum",
                idx
            )));
        }
        for (i, (&sum, &dyn_ok)) in step_proof
            .batched_time
            .claimed_sums
            .iter()
            .zip(claim_is_dynamic.iter())
            .enumerate()
        {
            if i == 0 {
                continue;
            }
            if !dyn_ok && sum != K::ZERO {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: batched_time claimed_sums[{}] must be 0 (label {:?})",
                    idx, i, expected_labels[i]
                )));
            }
        }
        if step_proof.batched_time.degree_bounds != expected_degree_bounds {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: batched_time degree_bounds mismatch",
                idx
            )));
        }
        if step_proof.batched_time.labels.len() != expected_labels.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: batched_time labels length mismatch",
                idx
            )));
        }
        for (i, (got, exp)) in step_proof
            .batched_time
            .labels
            .iter()
            .zip(expected_labels.iter())
            .enumerate()
        {
            if (*got as &[u8]) != *exp {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: batched_time label mismatch at claim {}",
                    idx, i
                )));
            }
        }

        let shout_pre = crate::memory_sidecar::memory::verify_shout_addr_pre_time(tr, step, &step_proof.mem)?;
        // Verify the batched time/row sumcheck rounds (derives shared r_time).
        bind_batched_dynamic_claims(
            tr,
            &step_proof.batched_time.claimed_sums,
            &expected_labels,
            &claim_is_dynamic,
        );
        let (r_time, final_values, ok) = verify_batched_sumcheck_rounds(
            tr,
            &step_proof.batched_time.round_polys,
            &step_proof.batched_time.claimed_sums,
            &expected_labels,
            &expected_degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "batched time sumcheck verification failed".into(),
            ));
        }
        if r_time.len() != ell_n {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: r_time length mismatch (got {}, expected ell_n={})",
                idx,
                r_time.len(),
                ell_n
            )));
        }
        if final_values.len() != expected_claims {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: batched final_values length mismatch",
                idx
            )));
        }

        // CCS proof structure consistency with batched time proof.
        let want_rounds_total = ell_n + ell_d;
        if step_proof.fold.ccs_proof.sumcheck_rounds.len() != want_rounds_total {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: CCS sumcheck_rounds.len()={}, expected {}",
                idx,
                step_proof.fold.ccs_proof.sumcheck_rounds.len(),
                want_rounds_total
            )));
        }
        if step_proof.fold.ccs_proof.sumcheck_challenges.len() != want_rounds_total {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: CCS sumcheck_challenges.len()={}, expected {}",
                idx,
                step_proof.fold.ccs_proof.sumcheck_challenges.len(),
                want_rounds_total
            )));
        }
        for (round_idx, (a, b)) in step_proof
            .fold
            .ccs_proof
            .sumcheck_rounds
            .iter()
            .take(ell_n)
            .zip(step_proof.batched_time.round_polys[0].iter())
            .enumerate()
        {
            if a != b {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: CCS time round poly mismatch at round {}",
                    idx, round_idx
                )));
            }
        }

        if step_proof.fold.ccs_proof.sumcheck_challenges[..ell_n] != r_time {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: CCS time challenges mismatch with r_time",
                idx
            )));
        }

        let expected_k = accumulator.len() + 1;
        if step_proof.fold.ccs_out.len() != expected_k {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS returned {} outputs; expected k={}",
                idx,
                step_proof.fold.ccs_out.len(),
                expected_k
            )));
        }
        if step_proof.fold.ccs_out.is_empty() {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS produced empty ccs_out",
                idx
            )));
        }
        if step_proof.fold.ccs_out[0].r != r_time {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS output r != r_time (Route A requires shared r)",
                idx
            )));
        }

        // Finish CCS Ajtai rounds alone (continuing transcript state after batched rounds).
        let ajtai_rounds = &step_proof.fold.ccs_proof.sumcheck_rounds[ell_n..];
        let (ajtai_chals, running_sum, ok) = verify_sumcheck_rounds(tr, d_sc, final_values[0], ajtai_rounds);
        if !ok {
            return Err(PiCcsError::SumcheckError("Π_CCS Ajtai rounds invalid".into()));
        }

        // Verify stored sumcheck challenges/final match transcript-derived values.
        let mut r_all = r_time.clone();
        r_all.extend_from_slice(&ajtai_chals);
        if r_all != step_proof.fold.ccs_proof.sumcheck_challenges {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS sumcheck challenges mismatch",
                idx
            )));
        }
        if running_sum != step_proof.fold.ccs_proof.sumcheck_final {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS sumcheck_final mismatch",
                idx
            )));
        }

        // Validate ME input r length (required by RHS assembly if k>1).
        for (i, me) in accumulator.iter().enumerate() {
            if me.r.len() != ell_n {
                return Err(PiCcsError::InvalidInput(format!(
                    "step {}: ME input r length mismatch at accumulator #{}: expected {}, got {}",
                    idx,
                    i,
                    ell_n,
                    me.r.len()
                )));
            }
        }

        // Paper-exact RHS assembly for CCS terminal identity.
        let rhs = rhs_terminal_identity_paper_exact(
            &s,
            params,
            &ch,
            &r_time,
            &ajtai_chals,
            &step_proof.fold.ccs_out,
            accumulator.first().map(|mi| mi.r.as_slice()),
        );
        if running_sum != rhs {
            return Err(PiCcsError::SumcheckError("Π_CCS terminal identity check failed".into()));
        }

        let observed_digest = tr.digest32();
        if observed_digest != step_proof.fold.ccs_proof.header_digest.as_slice() {
            return Err(PiCcsError::ProtocolError("Π_CCS header digest mismatch".into()));
        }

        // Verify mem proofs and collect ME claims.
        let mem_out = crate::memory_sidecar::memory::verify_route_a_memory_step(
            tr,
            params,
            step,
            &r_time,
            &r_cycle,
            &final_values,
            &step_proof.batched_time.claimed_sums,
            1, // claim 0 is CCS/time
            &step_proof.mem,
            &shout_pre,
        )?;
        let crate::memory_sidecar::memory::RouteAMemoryVerifyOutput {
            collected_me_time,
            collected_me_val,
            claim_idx_end: claim_idx,
            twist_rollover,
        } = mem_out;

        // Enforce Twist rollover between consecutive steps:
        // init_{i+1}(r_addr) == end_i(r_addr) for each memory instance.
        if idx + 1 < steps.len() {
            let next_step = steps
                .get(idx + 1)
                .ok_or_else(|| PiCcsError::ProtocolError("missing next step".into()))?;
            if next_step.mem_instances.len() != step.mem_instances.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "step {}: Twist rollover requires identical mem instance count in next step (current {}, next {})",
                    idx,
                    step.mem_instances.len(),
                    next_step.mem_instances.len()
                )));
            }
            if twist_rollover.len() != step.mem_instances.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Twist rollover output len()={}, expected {}",
                    idx,
                    twist_rollover.len(),
                    step.mem_instances.len()
                )));
            }

            for (i_mem, ((inst, _), (next_inst, _))) in step
                .mem_instances
                .iter()
                .zip(next_step.mem_instances.iter())
                .enumerate()
            {
                if next_inst.k != inst.k
                    || next_inst.d != inst.d
                    || next_inst.ell != inst.ell
                    || next_inst.n_side != inst.n_side
                {
                    return Err(PiCcsError::InvalidInput(format!(
                        "step {}: Twist rollover shape mismatch at mem {} (current k/d/ell/n_side = {}/{}/{}/{}, next = {}/{}/{}/{})",
                        idx,
                        i_mem,
                        inst.k,
                        inst.d,
                        inst.ell,
                        inst.n_side,
                        next_inst.k,
                        next_inst.d,
                        next_inst.ell,
                        next_inst.n_side
                    )));
                }

                let (r_addr, end_at_r_addr) = twist_rollover
                    .get(i_mem)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing Twist rollover entry".into()))?;
                let next_init_k: Vec<K> = next_inst.init_vals.iter().map(|&v| v.into()).collect();
                let init_next_at_r = table_mle_eval(&next_init_k, r_addr);
                if init_next_at_r != *end_at_r_addr {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: Twist rollover mismatch at mem {}",
                        idx, i_mem
                    )));
                }
            }
        }

        if claim_idx != final_values.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: batched claim index mismatch (consumed {}, have {})",
                idx,
                claim_idx,
                final_values.len()
            )));
        }

        validate_me_batch_invariants(&step_proof.fold.ccs_out, "verify step ccs outputs")?;
        validate_me_batch_invariants(&collected_me_time, "verify step memory outputs")?;
        validate_me_batch_invariants(&collected_me_val, "verify step memory val outputs")?;

        // Enforce r-alignment between CCS outputs and memory ME claims (needed for RLC).
        let r_ccs = &step_proof.fold.ccs_out[0].r;
        for (i, me) in collected_me_time.iter().enumerate() {
            if &me.r != r_ccs {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: memory ME[{}] r != CCS r (cannot fold via Π_RLC)",
                    idx, i
                )));
            }
        }

        // Sample rhos and check (after memory proofs, matching prover order)
        let mut rlc_inputs = step_proof.fold.ccs_out.clone();
        rlc_inputs.extend_from_slice(&collected_me_time);

        if step_proof.fold.rlc_rhos.len() != rlc_inputs.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: RLC ρ count mismatch (expected {}, got {})",
                idx,
                rlc_inputs.len(),
                step_proof.fold.rlc_rhos.len()
            )));
        }

        let rhos_from_tr = ccs::sample_rot_rhos_n(tr, params, &ring, rlc_inputs.len())?;
        for (j, (sampled, stored)) in rhos_from_tr
            .iter()
            .zip(step_proof.fold.rlc_rhos.iter())
            .enumerate()
        {
            if sampled.as_slice() != stored.as_slice() {
                return Err(PiCcsError::ProtocolError(format!(
                    "RLC ρ #{} mismatch: transcript vs proof",
                    j
                )));
            }
        }

        // Recompute RLC parent
        let parent_pub = ccs::rlc_public(
            &s,
            params,
            &step_proof.fold.rlc_rhos,
            &rlc_inputs,
            mixers.mix_rhos_commits,
            ell_d,
        );

        if parent_pub.X.as_slice() != step_proof.fold.rlc_parent.X.as_slice() {
            return Err(PiCcsError::ProtocolError("RLC X mismatch".into()));
        }
        if parent_pub.c != step_proof.fold.rlc_parent.c {
            return Err(PiCcsError::ProtocolError("RLC commitment mismatch".into()));
        }
        if parent_pub.r != step_proof.fold.rlc_parent.r {
            return Err(PiCcsError::ProtocolError("RLC r mismatch".into()));
        }
        if parent_pub.y != step_proof.fold.rlc_parent.y {
            return Err(PiCcsError::ProtocolError("RLC y mismatch".into()));
        }
        if parent_pub.y_scalars != step_proof.fold.rlc_parent.y_scalars {
            return Err(PiCcsError::ProtocolError("RLC y_scalars mismatch".into()));
        }

        if !ccs::verify_dec_public(
            &s,
            params,
            &step_proof.fold.rlc_parent,
            &step_proof.fold.dec_children,
            mixers.combine_b_pows,
            ell_d,
        ) {
            return Err(PiCcsError::ProtocolError("DEC public check failed".into()));
        }

        accumulator = step_proof.fold.dec_children.clone();

        // Phase 2: Verify the r_val folding lane for Twist val-eval ME claims.
        match (collected_me_val.is_empty(), step_proof.val_fold.as_ref()) {
            (true, None) => {}
            (true, Some(_)) => {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: unexpected val_fold proof (no r_val ME claims)",
                    idx
                )));
            }
            (false, None) => {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: missing val_fold proof (have r_val ME claims)",
                    idx
                )));
            }
            (false, Some(val_fold)) => {
                tr.append_message(b"fold/val_lane_start", &(idx as u64).to_le_bytes());

                if val_fold.rlc_rhos.len() != collected_me_val.len() {
                    return Err(PiCcsError::InvalidInput(format!(
                        "step {}: val-lane RLC ρ count mismatch (expected {}, got {})",
                        idx,
                        collected_me_val.len(),
                        val_fold.rlc_rhos.len()
                    )));
                }

                let rhos_from_tr = ccs::sample_rot_rhos_n(tr, params, &ring, collected_me_val.len())?;
                for (j, (sampled, stored)) in rhos_from_tr.iter().zip(val_fold.rlc_rhos.iter()).enumerate() {
                    if sampled.as_slice() != stored.as_slice() {
                        return Err(PiCcsError::ProtocolError(format!(
                            "step {}: val-lane RLC ρ #{} mismatch: transcript vs proof",
                            idx, j
                        )));
                    }
                }

                let parent_pub = ccs::rlc_public(
                    &s,
                    params,
                    &val_fold.rlc_rhos,
                    &collected_me_val,
                    mixers.mix_rhos_commits,
                    ell_d,
                );

                if parent_pub.X.as_slice() != val_fold.rlc_parent.X.as_slice() {
                    return Err(PiCcsError::ProtocolError("val-lane RLC X mismatch".into()));
                }
                if parent_pub.c != val_fold.rlc_parent.c {
                    return Err(PiCcsError::ProtocolError(
                        "val-lane RLC commitment mismatch".into(),
                    ));
                }
                if parent_pub.r != val_fold.rlc_parent.r {
                    return Err(PiCcsError::ProtocolError("val-lane RLC r mismatch".into()));
                }
                if parent_pub.y != val_fold.rlc_parent.y {
                    return Err(PiCcsError::ProtocolError("val-lane RLC y mismatch".into()));
                }
                if parent_pub.y_scalars != val_fold.rlc_parent.y_scalars {
                    return Err(PiCcsError::ProtocolError(
                        "val-lane RLC y_scalars mismatch".into(),
                    ));
                }

                if !ccs::verify_dec_public(
                    &s,
                    params,
                    &val_fold.rlc_parent,
                    &val_fold.dec_children,
                    mixers.combine_b_pows,
                    ell_d,
                ) {
                    return Err(PiCcsError::ProtocolError(
                        "val-lane DEC public check failed".into(),
                    ));
                }
            }
        }

        tr.append_message(b"fold/step_done", &(idx as u64).to_le_bytes());
    }

    Ok(())
}
