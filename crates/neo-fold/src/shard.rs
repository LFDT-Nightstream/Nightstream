//! Shard-level folding: CPU (Π_CCS) + memory sidecar (Twist/Shout) via Route A.
//!
//! High-level flow (per step):
//! 1. Bind CCS header + carried ME inputs.
//! 2. Prove/verify a *batched* time/row sum-check that shares `r_time` across CCS + Twist/Shout time oracles.
//! 3. Finish CCS Ajtai rounds using the CCS oracle state after the batched rounds.
//! 4. Finalize the memory sidecar at the shared `r_time` (and optionally produce Twist `r_val` claims).
//! 5. Fold all `r_time` ME claims (CCS outputs + memory claims) via Π_RLC → Π_DEC into `k_rho` children.
//! 6. If Twist produces `r_val` ME claims, fold them in a separate Π_RLC → Π_DEC lane.
//!
//! Notes:
//! - CCS-only folding is supported by passing steps with empty LUT/MEM vectors.
//! - Index→OneHot adapter is integrated via the Shout address-domain proving flow.

#![allow(non_snake_case)]

use crate::finalize::ObligationFinalizer;
use crate::memory_sidecar::sumcheck_ds::{run_sumcheck_prover_ds, verify_sumcheck_rounds_ds};
use crate::memory_sidecar::utils::RoundOraclePrefix;
use crate::pi_ccs::{self as ccs, FoldingMode};
pub use crate::shard_proof_types::{
    BatchedTimeProof, FoldStep, MemOrLutProof, MemSidecarProof, RlcDecProof, ShardFoldOutputs, ShardFoldWitnesses,
    ShardObligations, ShardProof, ShoutProofK, StepProof, TwistProofK,
};
use crate::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{KExtensions, D, F, K};
use neo_memory::ts_common as ts;
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_reductions::engines::utils;
use neo_reductions::paper_exact_engine::{
    build_me_outputs_paper_exact, claimed_initial_sum_from_inputs,
};
use neo_reductions::sumcheck::{poly_eval_k, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

// ============================================================================
// Utilities
// ============================================================================

pub use crate::memory_sidecar::memory::absorb_step_memory_commitments;

/// Commitment mixers so the coordinator stays scheme-agnostic.
/// - `mix_rhos_commits(ρ, cs)` returns Σ ρ_i · c_i  (S-action).
/// - `combine_b_pows(cs, b)` returns Σ \bar b^{i-1} c_i  (DEC check).
#[derive(Clone, Copy)]
pub struct CommitMixers<MR, MB>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    MB: Fn(&[Cmt], u32) -> Cmt,
{
    pub mix_rhos_commits: MR,
    pub combine_b_pows: MB,
}

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

fn pad_mats_to_ccs_width(mats: &[Mat<F>], target_cols: usize) -> Result<Vec<Mat<F>>, PiCcsError> {
    mats.iter()
        .map(|m| ts::pad_mat_to_ccs_width(m, target_cols))
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum RlcLane {
    Main,
    Val,
}

fn bind_rlc_inputs(
    tr: &mut Poseidon2Transcript,
    lane: RlcLane,
    step_idx: usize,
    me_inputs: &[MeInstance<Cmt, F, K>],
) -> Result<(), PiCcsError> {
    let lane_scope: &'static [u8] = match lane {
        RlcLane::Main => b"main",
        RlcLane::Val => b"val",
    };

    tr.append_message(b"fold/rlc_inputs/v1", lane_scope);
    tr.append_u64s(b"step_idx", &[step_idx as u64]);
    tr.append_u64s(b"me_count", &[me_inputs.len() as u64]);

    for me in me_inputs {
        tr.append_fields(b"c_data", &me.c.data);
        tr.append_u64s(b"m_in", &[me.m_in as u64]);
        tr.append_message(b"me_fold_digest", &me.fold_digest);

        for limb in &me.r {
            tr.append_fields(b"r_limb", &limb.as_coeffs());
        }

        tr.append_fields(b"X", me.X.as_slice());

        for yj in &me.y {
            for &y_elem in yj {
                tr.append_fields(b"y_elem", &y_elem.as_coeffs());
            }
        }

        for ysc in &me.y_scalars {
            tr.append_fields(b"y_scalar", &ysc.as_coeffs());
        }

        tr.append_u64s(b"c_step_coords_len", &[me.c_step_coords.len() as u64]);
        tr.append_fields(b"c_step_coords", &me.c_step_coords);
        tr.append_u64s(b"u_offset", &[me.u_offset as u64]);
        tr.append_u64s(b"u_len", &[me.u_len as u64]);
    }

    Ok(())
}

fn prove_rlc_dec_lane<L, MR, MB>(
    mode: &FoldingMode,
    lane: RlcLane,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    ring: &ccs::RotRing,
    ell_d: usize,
    k_dec: usize,
    step_idx: usize,
    me_inputs: &[MeInstance<Cmt, F, K>],
    wit_inputs: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(RlcDecProof, Vec<Mat<F>>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    bind_rlc_inputs(tr, lane, step_idx, me_inputs)?;
    let rlc_rhos = ccs::sample_rot_rhos_n(tr, params, ring, me_inputs.len())?;
    let (rlc_parent, Z_mix) = ccs::rlc_with_commit(
        mode.clone(),
        s,
        params,
        &rlc_rhos,
        me_inputs,
        wit_inputs,
        ell_d,
        mixers.mix_rhos_commits,
    );

    let Z_split = ccs::split_b_matrix_k(&Z_mix, k_dec, params.b)?;
    let child_cs: Vec<Cmt> = Z_split.iter().map(|Zi| l.commit(Zi)).collect();
    let (dec_children, ok_y, ok_X, ok_c) = ccs::dec_children_with_commit(
        mode.clone(),
        s,
        params,
        &rlc_parent,
        &Z_split,
        ell_d,
        &child_cs,
        mixers.combine_b_pows,
    );
    if !(ok_y && ok_X && ok_c) {
        let lane_label = match lane {
            RlcLane::Main => "DEC",
            RlcLane::Val => "DEC(val)",
        };
        return Err(PiCcsError::ProtocolError(format!(
            "{} public check failed at step {} (y={}, X={}, c={})",
            lane_label, step_idx, ok_y, ok_X, ok_c
        )));
    }

    Ok((
        RlcDecProof {
            rlc_rhos,
            rlc_parent,
            dec_children,
        },
        Z_split,
    ))
}

fn verify_rlc_dec_lane<MR, MB>(
    lane: RlcLane,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    ring: &ccs::RotRing,
    ell_d: usize,
    mixers: CommitMixers<MR, MB>,
    step_idx: usize,
    rlc_inputs: &[MeInstance<Cmt, F, K>],
    rlc_rhos: &[Mat<F>],
    rlc_parent: &MeInstance<Cmt, F, K>,
    dec_children: &[MeInstance<Cmt, F, K>],
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    bind_rlc_inputs(tr, lane, step_idx, rlc_inputs)?;

    if rlc_rhos.len() != rlc_inputs.len() {
        let prefix = match lane {
            RlcLane::Main => "",
            RlcLane::Val => "val-lane ",
        };
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: {}RLC ρ count mismatch (expected {}, got {})",
            step_idx,
            prefix,
            rlc_inputs.len(),
            rlc_rhos.len()
        )));
    }

    let rhos_from_tr = ccs::sample_rot_rhos_n(tr, params, ring, rlc_inputs.len())?;
    for (j, (sampled, stored)) in rhos_from_tr.iter().zip(rlc_rhos.iter()).enumerate() {
        if sampled.as_slice() != stored.as_slice() {
            return Err(PiCcsError::ProtocolError(match lane {
                RlcLane::Main => format!("step {}: RLC ρ #{} mismatch: transcript vs proof", step_idx, j),
                RlcLane::Val => format!("step {}: val-lane RLC ρ #{} mismatch: transcript vs proof", step_idx, j),
            }));
        }
    }

    let parent_pub = ccs::rlc_public(s, params, rlc_rhos, rlc_inputs, mixers.mix_rhos_commits, ell_d);

    let prefix = match lane {
        RlcLane::Main => "",
        RlcLane::Val => "val-lane ",
    };
    if parent_pub.X.as_slice() != rlc_parent.X.as_slice() {
        return Err(PiCcsError::ProtocolError(format!("step {}: {prefix}RLC X mismatch", step_idx)));
    }
    if parent_pub.c != rlc_parent.c {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC commitment mismatch",
            step_idx
        )));
    }
    if parent_pub.r != rlc_parent.r {
        return Err(PiCcsError::ProtocolError(format!("step {}: {prefix}RLC r mismatch", step_idx)));
    }
    if parent_pub.y != rlc_parent.y {
        return Err(PiCcsError::ProtocolError(format!("step {}: {prefix}RLC y mismatch", step_idx)));
    }
    if parent_pub.y_scalars != rlc_parent.y_scalars {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC y_scalars mismatch",
            step_idx
        )));
    }

    if !ccs::verify_dec_public(s, params, rlc_parent, dec_children, mixers.combine_b_pows, ell_d) {
        return Err(PiCcsError::ProtocolError(match lane {
            RlcLane::Main => format!("step {}: DEC public check failed", step_idx),
            RlcLane::Val => format!("step {}: val-lane DEC public check failed", step_idx),
        }));
    }

    Ok(())
}

// ============================================================================
// Shard Proving
// ============================================================================

fn fold_shard_prove_impl<L, MR, MB>(
    collect_val_lane_wits: bool,
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(ShardProof, Vec<Mat<F>>, Vec<Mat<F>>), PiCcsError>
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
    let mut val_lane_wits: Vec<Mat<F>> = Vec::new();
    let mut prev_twist_decoded: Option<Vec<neo_memory::twist::TwistDecodedCols>> = None;

    for (idx, step) in steps.iter().enumerate() {
        crate::memory_sidecar::memory::absorb_step_memory_commitments_witness(tr, step);

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
        let twist_pre = crate::memory_sidecar::memory::prove_twist_addr_pre_time(tr, params, step, ell_n, &r_cycle)?;
        let twist_read_claims: Vec<K> = twist_pre.iter().map(|p| p.read_check_claim_sum).collect();
        let twist_write_claims: Vec<K> = twist_pre.iter().map(|p| p.write_check_claim_sum).collect();
        let mut mem_oracles = crate::memory_sidecar::memory::build_route_a_memory_oracles(
            params, step, ell_n, &r_cycle, &shout_pre, &twist_pre,
        )?;

        let crate::memory_sidecar::route_a_time::RouteABatchedTimeProverOutput {
            r_time,
            per_claim_results,
            proof: batched_time,
        } = crate::memory_sidecar::route_a_time::prove_route_a_batched_time(
            tr,
            idx,
            ell_n,
            d_sc,
            ccs_initial_sum,
            ccs_oracle.as_mut(),
            &mut mem_oracles,
            step,
            twist_read_claims,
            twist_write_claims,
        )?;

        // Finish CCS Ajtai rounds alone, continuing from the CCS oracle state after ell_n folds.
        let ccs_time_rounds = per_claim_results
            .first()
            .map(|r| r.round_polys.clone())
            .unwrap_or_default();
        let mut sumcheck_rounds = ccs_time_rounds;
        let mut sumcheck_chals = r_time.clone();
        let ajtai_initial_sum = per_claim_results
            .first()
            .map(|r| r.final_value)
            .unwrap_or(ccs_initial_sum);

        let mut ccs_ajtai = RoundOraclePrefix::new(ccs_oracle.as_mut(), ell_d);
        let (ajtai_rounds, ajtai_chals) =
            run_sumcheck_prover_ds(tr, b"ccs/ajtai", idx, &mut ccs_ajtai, ajtai_initial_sum)?;
        let mut running_sum = ajtai_initial_sum;
        for (round_poly, &r_i) in ajtai_rounds.iter().zip(ajtai_chals.iter()) {
            running_sum = poly_eval_k(round_poly, r_i);
        }
        sumcheck_rounds.extend_from_slice(&ajtai_rounds);
        sumcheck_chals.extend_from_slice(&ajtai_chals);

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
        let prev_step = (idx > 0).then(|| &steps[idx - 1]);
        let prev_twist_decoded_ref = prev_twist_decoded.as_deref();
        let mut mem_out = crate::memory_sidecar::memory::finalize_route_a_memory_prover(
            tr,
            params,
            &s,
            step,
            prev_step,
            prev_twist_decoded_ref,
            &mut mem_oracles,
            &shout_pre,
            &twist_pre,
            &r_time,
            mcs_inst.m_in,
            idx,
        )?;
        prev_twist_decoded = Some(twist_pre.into_iter().map(|p| p.decoded).collect());

        normalize_me_claims(&mut mem_out.mem.me_claims_time, ell_n, ell_d, s.t())?;
        normalize_me_claims(&mut mem_out.mem.me_claims_val, ell_n, ell_d, s.t())?;

        validate_me_batch_invariants(&ccs_out, "prove step ccs outputs")?;
        validate_me_batch_invariants(&mem_out.mem.me_claims_time, "prove step memory outputs")?;

        // Build RLC inputs: CCS outputs + memory ME
        let mut rlc_inputs = ccs_out.clone();
        rlc_inputs.extend(mem_out.mem.me_claims_time.clone());
        validate_me_batch_invariants(&rlc_inputs, "prove step RLC inputs (ccs_out + mem_time)")?;

        let mut rlc_wits = outs_Z.clone();
        rlc_wits.extend(pad_mats_to_ccs_width(&mem_out.me_wits_time, s.m)?);

        let (main_fold, Z_split) = prove_rlc_dec_lane(
            &mode,
            RlcLane::Main,
            tr,
            params,
            &s,
            &ring,
            ell_d,
            k_dec,
            idx,
            &rlc_inputs,
            &rlc_wits,
            l,
            mixers,
        )?;
        let RlcDecProof {
            rlc_rhos: rhos,
            rlc_parent: parent_pub,
            dec_children: children,
        } = main_fold;

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
            let val_wits = pad_mats_to_ccs_width(&mem_out.me_wits_val, s.m)?;
            let (val_fold, mut Z_split_val) = prove_rlc_dec_lane(
                &mode,
                RlcLane::Val,
                tr,
                params,
                &s,
                &ring,
                ell_d,
                k_dec,
                idx,
                &mem_out.mem.me_claims_val,
                &val_wits,
                l,
                mixers,
            )?;

            if collect_val_lane_wits {
                val_lane_wits.extend(Z_split_val.drain(..));
            }

            Some(val_fold)
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

    Ok((ShardProof { steps: step_proofs }, accumulator_wit, val_lane_wits))
}

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
    let (proof, _final_main_wits, _val_lane_wits) =
        fold_shard_prove_impl(false, mode, tr, params, s_me, steps, acc_init, acc_wit_init, l, mixers)?;
    Ok(proof)
}

pub fn fold_shard_prove_with_witnesses<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(ShardProof, ShardFoldOutputs<Cmt, F, K>, ShardFoldWitnesses<F>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, final_main_wits, val_lane_wits) =
        fold_shard_prove_impl(true, mode, tr, params, s_me, steps, acc_init, acc_wit_init, l, mixers)?;
    let outputs = proof.compute_fold_outputs(acc_init);
    if outputs.obligations.main.len() != final_main_wits.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "final main witness count mismatch (have {}, need {})",
            final_main_wits.len(),
            outputs.obligations.main.len()
        )));
    }
    if outputs.obligations.val.len() != val_lane_wits.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "val-lane witness count mismatch (have {}, need {})",
            val_lane_wits.len(),
            outputs.obligations.val.len()
        )));
    }
    Ok((
        proof,
        outputs,
        ShardFoldWitnesses {
            final_main_wits,
            val_lane_wits,
        },
    ))
}

// ============================================================================
// Shard Verification
// ============================================================================

pub fn fold_shard_verify<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
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
    let mut val_lane_obligations: Vec<MeInstance<Cmt, F, K>> = Vec::new();

    for (idx, (step, step_proof)) in steps.iter().zip(proof.steps.iter()).enumerate() {
        absorb_step_memory_commitments(tr, step);

        let mcs_inst = &step.mcs_inst;

        // --------------------------------------------------------------------
        // Route A: Verify shared-challenge batched sum-check (time/row rounds),
        // then finish CCS Ajtai rounds, then proceed with RLC→DEC as before.
        // --------------------------------------------------------------------

        // Bind CCS header + ME inputs and sample public challenges.
        utils::bind_header_and_instances(tr, params, &s, core::slice::from_ref(mcs_inst), ell, d_sc, 0)?;
        utils::bind_me_inputs(tr, &accumulator)?;
        let ch = utils::sample_challenges(tr, ell_d, ell)?;
        let expected_ch = &step_proof.fold.ccs_proof.challenges_public;
        if expected_ch.alpha != ch.alpha
            || expected_ch.beta_a != ch.beta_a
            || expected_ch.beta_r != ch.beta_r
            || expected_ch.gamma != ch.gamma
        {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS challenges_public mismatch",
                idx
            )));
        }

        // Public initial sum T for CCS sumcheck (engine-selected).
        let claimed_initial = match &mode {
            FoldingMode::Optimized => crate::optimized_engine::claimed_initial_sum_from_inputs(&s, &ch, &accumulator),
            #[cfg(feature = "paper-exact")]
            FoldingMode::PaperExact => crate::paper_exact_engine::claimed_initial_sum_from_inputs(&s, &ch, &accumulator),
            #[cfg(feature = "paper-exact")]
            FoldingMode::OptimizedWithCrosscheck(_) => {
                crate::optimized_engine::claimed_initial_sum_from_inputs(&s, &ch, &accumulator)
            }
        };
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

        let shout_pre = crate::memory_sidecar::memory::verify_shout_addr_pre_time(tr, step, &step_proof.mem)?;
        let twist_pre = crate::memory_sidecar::memory::verify_twist_addr_pre_time(tr, step, &step_proof.mem)?;
        let crate::memory_sidecar::route_a_time::RouteABatchedTimeVerifyOutput { r_time, final_values } =
            crate::memory_sidecar::route_a_time::verify_route_a_batched_time(
                tr,
                idx,
                ell_n,
                d_sc,
                claimed_initial,
                step,
                &step_proof.batched_time,
            )?;

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
        let (ajtai_chals, running_sum, ok) =
            verify_sumcheck_rounds_ds(tr, b"ccs/ajtai", idx, d_sc, final_values[0], ajtai_rounds);
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

        // Engine-selected RHS assembly for CCS terminal identity.
        let rhs = match &mode {
            FoldingMode::Optimized => crate::optimized_engine::rhs_terminal_identity_paper_exact(
                &s,
                params,
                &ch,
                &r_time,
                &ajtai_chals,
                &step_proof.fold.ccs_out,
                accumulator.first().map(|mi| mi.r.as_slice()),
            ),
            #[cfg(feature = "paper-exact")]
            FoldingMode::PaperExact => crate::paper_exact_engine::rhs_terminal_identity_paper_exact(
                &s,
                params,
                &ch,
                &r_time,
                &ajtai_chals,
                &step_proof.fold.ccs_out,
                accumulator.first().map(|mi| mi.r.as_slice()),
            ),
            #[cfg(feature = "paper-exact")]
            FoldingMode::OptimizedWithCrosscheck(_) => crate::optimized_engine::rhs_terminal_identity_paper_exact(
                &s,
                params,
                &ch,
                &r_time,
                &ajtai_chals,
                &step_proof.fold.ccs_out,
                accumulator.first().map(|mi| mi.r.as_slice()),
            ),
        };
        if running_sum != rhs {
            return Err(PiCcsError::SumcheckError("Π_CCS terminal identity check failed".into()));
        }

        let observed_digest = tr.digest32();
        if observed_digest != step_proof.fold.ccs_proof.header_digest.as_slice() {
            return Err(PiCcsError::ProtocolError("Π_CCS header digest mismatch".into()));
        }

        // Verify mem proofs and collect ME claims.
        let prev_step = (idx > 0).then(|| &steps[idx - 1]);
        let mem_out = crate::memory_sidecar::memory::verify_route_a_memory_step(
            tr,
            params,
            step,
            prev_step,
            &r_time,
            &r_cycle,
            &final_values,
            &step_proof.batched_time.claimed_sums,
            1, // claim 0 is CCS/time
            &step_proof.mem,
            &shout_pre,
            &twist_pre,
            idx,
        )?;
        let crate::memory_sidecar::memory::RouteAMemoryVerifyOutput {
            collected_me_time,
            collected_me_val,
            claim_idx_end: claim_idx,
            twist_total_inc_sums: _twist_total_inc_sums,
        } = mem_out;

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

        let mut rlc_inputs = step_proof.fold.ccs_out.clone();
        rlc_inputs.extend_from_slice(&collected_me_time);
        validate_me_batch_invariants(&rlc_inputs, "verify step RLC inputs (ccs_out + mem_time)")?;
        verify_rlc_dec_lane(
            RlcLane::Main,
            tr,
            params,
            &s,
            &ring,
            ell_d,
            mixers,
            idx,
            &rlc_inputs,
            &step_proof.fold.rlc_rhos,
            &step_proof.fold.rlc_parent,
            &step_proof.fold.dec_children,
        )?;

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
                verify_rlc_dec_lane(
                    RlcLane::Val,
                    tr,
                    params,
                    &s,
                    &ring,
                    ell_d,
                    mixers,
                    idx,
                    &collected_me_val,
                    &val_fold.rlc_rhos,
                    &val_fold.rlc_parent,
                    &val_fold.dec_children,
                )?;

                val_lane_obligations.extend_from_slice(&val_fold.dec_children);
            }
        }

        tr.append_message(b"fold/step_done", &(idx as u64).to_le_bytes());
    }

    Ok(ShardFoldOutputs {
        obligations: ShardObligations {
            main: accumulator,
            val: val_lane_obligations,
        },
    })
}

pub fn fold_shard_verify_and_finalize<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    let outputs = fold_shard_verify(mode, tr, params, s_me, steps, acc_init, proof, mixers)?;
    finalizer.finalize(&outputs.obligations)?;
    Ok(())
}
