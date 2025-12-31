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
use neo_reductions::paper_exact_engine::{build_me_outputs_paper_exact, claimed_initial_sum_from_inputs};
use neo_reductions::sumcheck::{poly_eval_k, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use std::sync::Arc;

// ============================================================================
// Utilities
// ============================================================================

pub use crate::memory_sidecar::memory::absorb_step_memory;

// ============================================================================
// Optional step-to-step (cross-chunk) linking
// ============================================================================

/// Optional verifier-side linking constraints across adjacent shard steps.
///
/// This is intended for chunked CPU circuits that expose boundary state as part of the public
/// input vector `x` per step, and need the verifier to enforce that the state chains across steps.
#[derive(Clone, Debug)]
pub struct StepLinkingConfig {
    /// Equalities on adjacent steps: require `steps[i].x[prev_idx] == steps[i+1].x[next_idx]`.
    pub prev_next_equalities: Vec<(usize, usize)>,
}

impl StepLinkingConfig {
    pub fn new(prev_next_equalities: Vec<(usize, usize)>) -> Self {
        Self { prev_next_equalities }
    }
}

pub fn check_step_linking(
    steps: &[StepInstanceBundle<Cmt, F, K>],
    cfg: &StepLinkingConfig,
) -> Result<(), PiCcsError> {
    if steps.len() <= 1 || cfg.prev_next_equalities.is_empty() {
        return Ok(());
    }
    for (i, (prev, next)) in steps.iter().zip(steps.iter().skip(1)).enumerate() {
        let prev_x = &prev.mcs_inst.x;
        let next_x = &next.mcs_inst.x;
        for &(prev_idx, next_idx) in &cfg.prev_next_equalities {
            if prev_idx >= prev_x.len() || next_idx >= next_x.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "step linking index out of range at boundary {i}: prev_x.len()={}, next_x.len()={}, pair=({prev_idx},{next_idx})",
                    prev_x.len(),
                    next_x.len(),
                )));
            }
            if prev_x[prev_idx] != next_x[next_idx] {
                return Err(PiCcsError::ProtocolError(format!(
                    "step linking failed at boundary {i}: prev_x[{prev_idx}] != next_x[{next_idx}]",
                )));
            }
        }
    }
    Ok(())
}

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
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC X mismatch",
            step_idx
        )));
    }
    if parent_pub.c != rlc_parent.c {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC commitment mismatch",
            step_idx
        )));
    }
    if parent_pub.r != rlc_parent.r {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC r mismatch",
            step_idx
        )));
    }
    if parent_pub.y != rlc_parent.y {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC y mismatch",
            step_idx
        )));
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
    ob: Option<(&crate::output_binding::OutputBindingConfig, &[F])>,
) -> Result<(ShardProof, Vec<Mat<F>>, Vec<Mat<F>>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let s0 = s_me
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    let (s, cpu_bus) = crate::memory_sidecar::cpu_bus::prepare_ccs_for_shared_cpu_bus_steps(&s0, steps)?;
    // Route A terminal checks interpret `ME.y_scalars[0]` as MLE(column)(r_time), which requires M₀ = I.
    s.assert_m0_is_identity_for_nc()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first (M₀=I) required: {e:?}")))?;
    let utils::Dims {
        ell_d,
        ell_n,
        ell,
        d_sc,
    } = utils::build_dims_and_policy(params, &s)?;
    let ccs_mat_digest = utils::digest_ccs_matrices(&s);
    let ccs_sparse_cache = Arc::new(
        neo_reductions::engines::optimized_engine::oracle::SparseCache::build(&s),
    );
    let k_dec = params.k_rho as usize;
    let ring = ccs::RotRing::goldilocks();

    // Initialize accumulator
    let mut accumulator = acc_init.to_vec();
    let mut accumulator_wit = acc_wit_init.to_vec();

    let mut step_proofs = Vec::with_capacity(steps.len());
    let mut val_lane_wits: Vec<Mat<F>> = Vec::new();
    let mut prev_twist_decoded: Option<Vec<crate::memory_sidecar::memory::TwistDecodedColsSparse>> = None;
    let mut output_proof: Option<neo_memory::output_check::OutputBindingProof> = None;

    if ob.is_some() && steps.is_empty() {
        return Err(PiCcsError::InvalidInput("output binding requires >= 1 step".into()));
    }

    for (idx, step) in steps.iter().enumerate() {
        crate::memory_sidecar::memory::absorb_step_memory_witness(tr, step);

        let include_ob = ob.is_some() && (idx + 1 == steps.len());
        let mut ob_time_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut ob_r_prime: Option<Vec<K>> = None;

        // Output binding is injected only on the final step, and must run before sampling Route-A `r_time`.
        if include_ob {
            let (cfg, final_memory_state) =
                ob.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;

            if output_proof.is_some() {
                return Err(PiCcsError::ProtocolError(
                    "output binding already attached (internal error)".into(),
                ));
            }

            if cfg.mem_idx >= step.mem_instances.len() {
                return Err(PiCcsError::InvalidInput("output binding mem_idx out of range".into()));
            }
            let expected_k = 1usize
                .checked_shl(cfg.num_bits as u32)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
            if final_memory_state.len() != expected_k {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: final_memory_state.len()={} != 2^num_bits={}",
                    final_memory_state.len(),
                    expected_k
                )));
            }
            let mem_inst = &step.mem_instances[cfg.mem_idx].0;
            if mem_inst.k != expected_k {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits implies k={}, but mem_inst.k={}",
                    expected_k, mem_inst.k
                )));
            }
            let ell_addr = mem_inst.twist_layout().ell_addr;
            if ell_addr != cfg.num_bits {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits={}, but twist_layout.ell_addr={}",
                    cfg.num_bits, ell_addr
                )));
            }

            tr.append_message(b"shard/output_binding_start", &(idx as u64).to_le_bytes());
            tr.append_u64s(b"output_binding/mem_idx", &[cfg.mem_idx as u64]);
            tr.append_u64s(b"output_binding/num_bits", &[cfg.num_bits as u64]);

            let (output_sc, r_prime) = neo_memory::output_check::generate_output_sumcheck_proof_and_challenges(
                tr,
                cfg.num_bits,
                cfg.program_io.clone(),
                final_memory_state,
            )
            .map_err(|e| PiCcsError::ProtocolError(format!("output sumcheck failed: {e:?}")))?;

            output_proof = Some(neo_memory::output_check::OutputBindingProof { output_sc });
            ob_r_prime = Some(r_prime);
        }

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

        utils::bind_header_and_instances_with_digest(
            tr,
            params,
            &s,
            core::slice::from_ref(mcs_inst),
            ell,
            d_sc,
            &ccs_mat_digest,
        )?;
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
                Box::new(neo_reductions::engines::optimized_engine::oracle::OptimizedOracle::new_with_sparse(
                    &s,
                    params,
                    core::slice::from_ref(mcs_wit),
                    &accumulator_wit,
                    ch.clone(),
                    ell_d,
                    ell_n,
                    d_sc,
                    accumulator.first().map(|mi| mi.r.as_slice()),
                    ccs_sparse_cache.clone(),
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
                Box::new(neo_reductions::engines::optimized_engine::oracle::OptimizedOracle::new_with_sparse(
                    &s,
                    params,
                    core::slice::from_ref(mcs_wit),
                    &accumulator_wit,
                    ch.clone(),
                    ell_d,
                    ell_n,
                    d_sc,
                    accumulator.first().map(|mi| mi.r.as_slice()),
                    ccs_sparse_cache.clone(),
                ))
            }
        };

        let shout_pre = crate::memory_sidecar::memory::prove_shout_addr_pre_time(
            tr,
            params,
            step,
            Some(&cpu_bus),
            ell_n,
            &r_cycle,
            idx,
        )?;
        let twist_pre = crate::memory_sidecar::memory::prove_twist_addr_pre_time(
            tr,
            params,
            step,
            Some(&cpu_bus),
            ell_n,
            &r_cycle,
        )?;
        let twist_read_claims: Vec<K> = twist_pre.iter().map(|p| p.read_check_claim_sum).collect();
        let twist_write_claims: Vec<K> = twist_pre.iter().map(|p| p.write_check_claim_sum).collect();
        let mut mem_oracles = crate::memory_sidecar::memory::build_route_a_memory_oracles(
            params, step, ell_n, &r_cycle, &shout_pre, &twist_pre,
        )?;

        if include_ob {
            let (cfg, _final_memory_state) =
                ob.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;
            let r_prime = ob_r_prime
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("output binding r_prime missing".into()))?;
            let pre = twist_pre
                .get(cfg.mem_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("output binding mem_idx out of range for twist_pre".into()))?;

            let (oracle, claimed_sum) = neo_memory::twist_oracle::TwistTotalIncOracleSparseTime::new(
                pre.decoded.wa_bits.clone(),
                pre.decoded.has_write.clone(),
                pre.decoded.inc_at_write_addr.clone(),
                r_prime,
            );

            ob_time_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle: Box::new(oracle),
                claimed_sum,
                label: crate::output_binding::OB_INC_TOTAL_LABEL,
            });
        }

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
            ob_time_claim,
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
        let mut mem_proof = crate::memory_sidecar::memory::finalize_route_a_memory_prover(
            tr,
            params,
            &s,
            step,
            prev_step,
            prev_twist_decoded_ref,
            &mut mem_oracles,
            &shout_pre.addr_pre,
            &twist_pre,
            &r_time,
            mcs_inst.m_in,
            idx,
        )?;
        prev_twist_decoded = Some(twist_pre.into_iter().map(|p| p.decoded).collect());

        normalize_me_claims(&mut mem_proof.cpu_me_claims_val, ell_n, ell_d, s.t())?;

        validate_me_batch_invariants(&ccs_out, "prove step ccs outputs")?;

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
            &ccs_out,
            &outs_Z,
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
        let val_fold = if mem_proof.cpu_me_claims_val.is_empty() {
            None
        } else {
            validate_me_batch_invariants(&mem_proof.cpu_me_claims_val, "prove step memory val outputs")?;

            tr.append_message(b"fold/val_lane_start", &(idx as u64).to_le_bytes());
            let mut val_wits = Vec::with_capacity(mem_proof.cpu_me_claims_val.len());
            val_wits.push(mcs_wit.Z.clone());
            if let Some(prev) = prev_step {
                val_wits.push(prev.mcs.1.Z.clone());
            }
            if val_wits.len() != mem_proof.cpu_me_claims_val.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "Twist(val) witness count mismatch (have {}, need {})",
                    val_wits.len(),
                    mem_proof.cpu_me_claims_val.len()
                )));
            }
            let val_wits = pad_mats_to_ccs_width(&val_wits, s.m)?;
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
                &mem_proof.cpu_me_claims_val,
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
            mem: mem_proof,
            batched_time,
            val_fold,
        });

        tr.append_message(b"fold/step_done", &(idx as u64).to_le_bytes());
    }

    Ok((
        ShardProof {
            steps: step_proofs,
            output_proof,
        },
        accumulator_wit,
        val_lane_wits,
    ))
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
    let (proof, _final_main_wits, _val_lane_wits) = fold_shard_prove_impl(
        false,
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        None,
    )?;
    Ok(proof)
}

pub fn fold_shard_prove_with_output_binding<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    final_memory_state: &[F],
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, _final_main_wits, _val_lane_wits) = fold_shard_prove_impl(
        false,
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        Some((ob_cfg, final_memory_state)),
    )?;
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
    let (proof, final_main_wits, val_lane_wits) = fold_shard_prove_impl(
        true,
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        None,
    )?;
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

fn fold_shard_verify_impl<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: Option<&crate::output_binding::OutputBindingConfig>,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let s0 = s_me
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    let (s, cpu_bus) = crate::memory_sidecar::cpu_bus::prepare_ccs_for_shared_cpu_bus_steps(&s0, steps)?;
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
    if ob_cfg.is_some() && steps.is_empty() {
        return Err(PiCcsError::InvalidInput("output binding requires >= 1 step".into()));
    }
    if ob_cfg.is_none() && proof.output_proof.is_some() {
        return Err(PiCcsError::InvalidInput(
            "shard proof contains output binding, but verifier did not supply OutputBindingConfig".into(),
        ));
    }
    if ob_cfg.is_some() && proof.output_proof.is_none() {
        return Err(PiCcsError::InvalidInput(
            "verifier supplied OutputBindingConfig, but shard proof has no output binding".into(),
        ));
    }

    let mut accumulator = acc_init.to_vec();
    let mut val_lane_obligations: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let ccs_mat_digest = utils::digest_ccs_matrices(&s);

    for (idx, (step, step_proof)) in steps.iter().zip(proof.steps.iter()).enumerate() {
        absorb_step_memory(tr, step);

        let include_ob = ob_cfg.is_some() && (idx + 1 == steps.len());
        let mut ob_state: Option<neo_memory::output_check::OutputSumcheckState> = None;
        let mut ob_inc_total_degree_bound: Option<usize> = None;

        if include_ob {
            let cfg =
                ob_cfg.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;
            let ob_proof = proof
                .output_proof
                .as_ref()
                .ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but proof missing".into()))?;

            if cfg.mem_idx >= step.mem_insts.len() {
                return Err(PiCcsError::InvalidInput("output binding mem_idx out of range".into()));
            }
            let mem_inst = step
                .mem_insts
                .get(cfg.mem_idx)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding mem_idx out of range".into()))?;
            let expected_k = 1usize
                .checked_shl(cfg.num_bits as u32)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
            if mem_inst.k != expected_k {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits implies k={}, but mem_inst.k={}",
                    expected_k, mem_inst.k
                )));
            }
            let ell_addr = mem_inst.twist_layout().ell_addr;
            if ell_addr != cfg.num_bits {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits={}, but twist_layout.ell_addr={}",
                    cfg.num_bits, ell_addr
                )));
            }

            tr.append_message(b"shard/output_binding_start", &(idx as u64).to_le_bytes());
            tr.append_u64s(b"output_binding/mem_idx", &[cfg.mem_idx as u64]);
            tr.append_u64s(b"output_binding/num_bits", &[cfg.num_bits as u64]);

            let state = neo_memory::output_check::verify_output_sumcheck_rounds_get_state(
                tr,
                cfg.num_bits,
                cfg.program_io.clone(),
                &ob_proof.output_sc,
            )
            .map_err(|e| PiCcsError::ProtocolError(format!("output sumcheck failed: {e:?}")))?;
            ob_inc_total_degree_bound = Some(2 + ell_addr);
            ob_state = Some(state);
        }

        let mcs_inst = &step.mcs_inst;

        // --------------------------------------------------------------------
        // Route A: Verify shared-challenge batched sum-check (time/row rounds),
        // then finish CCS Ajtai rounds, then proceed with RLC→DEC as before.
        // --------------------------------------------------------------------

        // Bind CCS header + ME inputs and sample public challenges.
        utils::bind_header_and_instances_with_digest(
            tr,
            params,
            &s,
            core::slice::from_ref(mcs_inst),
            ell,
            d_sc,
            &ccs_mat_digest,
        )?;
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
            FoldingMode::PaperExact => {
                crate::paper_exact_engine::claimed_initial_sum_from_inputs(&s, &ch, &accumulator)
            }
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

        let shout_pre = crate::memory_sidecar::memory::verify_shout_addr_pre_time(tr, step, &step_proof.mem, idx)?;
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
                ob_inc_total_degree_bound,
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

        // Verify mem proofs (shared CPU bus only).
        let prev_step = (idx > 0).then(|| &steps[idx - 1]);
        let mem_out = crate::memory_sidecar::memory::verify_route_a_memory_step(
            tr,
            &cpu_bus,
            step,
            prev_step,
            &step_proof.fold.ccs_out[0],
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

        let expected_consumed = if include_ob {
            final_values
                .len()
                .checked_sub(1)
                .ok_or_else(|| PiCcsError::ProtocolError("missing output binding claim".into()))?
        } else {
            final_values.len()
        };
        if mem_out.claim_idx_end != expected_consumed {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: batched claim index mismatch (consumed {}, expected {})",
                idx, mem_out.claim_idx_end, expected_consumed
            )));
        }

        if include_ob {
            let cfg =
                ob_cfg.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;
            let ob_state = ob_state
                .take()
                .ok_or_else(|| PiCcsError::ProtocolError("output sumcheck state missing".into()))?;

            let inc_idx = final_values
                .len()
                .checked_sub(1)
                .ok_or_else(|| PiCcsError::ProtocolError("missing inc_total claim".into()))?;
            if step_proof.batched_time.labels.get(inc_idx).copied() != Some(crate::output_binding::OB_INC_TOTAL_LABEL) {
                return Err(PiCcsError::ProtocolError("output binding claim not last".into()));
            }

            let inc_total_claim = *step_proof
                .batched_time
                .claimed_sums
                .get(inc_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("missing inc_total claimed_sum".into()))?;
            let inc_total_final = *final_values
                .get(inc_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("missing inc_total final_value".into()))?;

            let twist_open = mem_out
                .twist_time_openings
                .get(cfg.mem_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("missing twist_time_openings for mem_idx".into()))?;
            let inc_terminal = crate::output_binding::inc_terminal_from_time_openings(twist_open, &ob_state.r_prime)
                .map_err(|e| PiCcsError::ProtocolError(format!("inc_total terminal mismatch: {e:?}")))?;
            if inc_total_final != inc_terminal {
                return Err(PiCcsError::ProtocolError("inc_total terminal mismatch".into()));
            }

            let mem_inst = step
                .mem_insts
                .get(cfg.mem_idx)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding mem_idx out of range".into()))?;
            let expected_k = 1usize
                .checked_shl(cfg.num_bits as u32)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
            if mem_inst.k != expected_k {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits implies k={}, but mem_inst.k={}",
                    expected_k, mem_inst.k
                )));
            }
            let ell_addr = mem_inst.twist_layout().ell_addr;
            if ell_addr != cfg.num_bits {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits={}, but twist_layout.ell_addr={}",
                    cfg.num_bits, ell_addr
                )));
            }
            let val_init = crate::output_binding::val_init_from_mem_init(&mem_inst.init, mem_inst.k, &ob_state.r_prime)
                .map_err(|e| PiCcsError::ProtocolError(format!("MemInit eval failed: {e:?}")))?;

            let val_final_at_r_prime = val_init + inc_total_claim;
            let expected_out = ob_state.eq_eval * ob_state.io_mask_eval * (val_final_at_r_prime - ob_state.val_io_eval);
            if expected_out != ob_state.output_final {
                return Err(PiCcsError::ProtocolError("output binding final check failed".into()));
            }
        }

        validate_me_batch_invariants(&step_proof.fold.ccs_out, "verify step ccs outputs")?;
        validate_me_batch_invariants(&step_proof.mem.cpu_me_claims_val, "verify step memory val outputs")?;
        verify_rlc_dec_lane(
            RlcLane::Main,
            tr,
            params,
            &s,
            &ring,
            ell_d,
            mixers,
            idx,
            &step_proof.fold.ccs_out,
            &step_proof.fold.rlc_rhos,
            &step_proof.fold.rlc_parent,
            &step_proof.fold.dec_children,
        )?;

        accumulator = step_proof.fold.dec_children.clone();

        // Phase 2: Verify the r_val folding lane for Twist val-eval ME claims.
        match (
            step_proof.mem.cpu_me_claims_val.is_empty(),
            step_proof.val_fold.as_ref(),
        ) {
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
                    &step_proof.mem.cpu_me_claims_val,
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
    fold_shard_verify_impl(mode, tr, params, s_me, steps, acc_init, proof, mixers, None)
}

pub fn fold_shard_verify_with_step_linking<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    step_linking: &StepLinkingConfig,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify(mode, tr, params, s_me, steps, acc_init, proof, mixers)
}

pub fn fold_shard_verify_with_output_binding<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_impl(mode, tr, params, s_me, steps, acc_init, proof, mixers, Some(ob_cfg))
}

pub fn fold_shard_verify_with_output_binding_and_step_linking<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    step_linking: &StepLinkingConfig,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_with_output_binding(mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg)
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
    let report = finalizer.finalize(&outputs.obligations)?;
    outputs
        .obligations
        .require_all_finalized(report.did_finalize_main, report.did_finalize_val)?;
    Ok(())
}

pub fn fold_shard_verify_and_finalize_with_step_linking<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    step_linking: &StepLinkingConfig,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_and_finalize(mode, tr, params, s_me, steps, acc_init, proof, mixers, finalizer)
}

pub fn fold_shard_verify_and_finalize_with_output_binding<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    let outputs =
        fold_shard_verify_with_output_binding(mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg)?;
    let report = finalizer.finalize(&outputs.obligations)?;
    outputs
        .obligations
        .require_all_finalized(report.did_finalize_main, report.did_finalize_val)?;
    Ok(())
}

pub fn fold_shard_verify_and_finalize_with_output_binding_and_step_linking<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    step_linking: &StepLinkingConfig,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_and_finalize_with_output_binding(
        mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg, finalizer,
    )
}
