//! The segment prover — spec §1's two-pass flow, per segment window.
//!
//! Owns: turning one [`SegmentTrace`] (the native pass) into `N` folded
//! `S_mem` instances: lane encodings → lane commitments → `D_pre` → γ →
//! step witnesses with running products → `extend` loop. Also the γ
//! pre-derivation ([`derive_segment_gamma`]) the instances need before
//! the lifecycle's own `open_segment` runs, and the §6.4 prover resume
//! ([`resume_segment`]): the carried lane is the checkpoint, the trace
//! is the "remaining witness plan", and recompute-vs-`D_pre` is what
//! authenticates the pair.
//!
//! Does not own: memory semantics ([`super::trace`] produced the trace),
//! the plan ([`super::plan`]), the lane transition (the lifecycle runs
//! `NebulaLane::advance_for_batch` inside every extend — this module
//! only *prepares* consistent inputs; nothing here is verifier
//! authority).

use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use crate::frontends::nebula::circuit::StepData;
use crate::frontends::nebula::fingerprint::Gammas;
use crate::frontends::nebula::layout::LayoutError;
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::nebula::trace::SegmentTrace;
use crate::lifecycle::{self, Preprocessing, UncompressedAudit};
use crate::paper::digest;
use crate::paper::relations::{CcsInstance, LaneSchemeError, RelationError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SegmentError {
    #[error("segment prover: trace was produced under different plan constants")]
    PlanMismatch,
    #[error("segment prover: preprocessing structure is not this plan's S_mem structure")]
    StructureMismatch,
    #[error("segment prover: chain carries no Nebula lane (preprocess with the plan's config)")]
    LaneMissing,
    #[error("segment prover: trace opens segment {trace_seg} at ts {trace_ts}, chain lane is at segment {lane_seg}, ts {lane_ts}")]
    ChainPositionMismatch {
        trace_seg: u64,
        trace_ts: u64,
        lane_seg: u64,
        lane_ts: u64,
    },
    #[error("segment prover: resume requires a mid-segment lane (segment open, steps remaining); this chain is at a segment boundary")]
    NotMidSegment,
    #[error("segment prover: trace does not reproduce the open segment's pre-committed lane chains (D_pre mismatch — wrong or mutated trace)")]
    ResumeTraceMismatch,
    #[error("segment prover: {0}")]
    Layout(#[from] LayoutError),
    #[error("segment prover: {0}")]
    Lanes(#[from] LaneSchemeError),
    #[error("segment prover: {0}")]
    Instance(#[from] RelationError),
    #[error(transparent)]
    Lifecycle(#[from] lifecycle::Error),
    #[error(transparent)]
    Nebula(#[from] crate::paper::construction2::Error),
    #[error("segment prover: {0}")]
    Lane(#[from] crate::paper::construction2::NebulaError),
}

/// γ exactly as the lifecycle's `open_segment` will squeeze it at this
/// chain position (spec §6.2) — the prover needs it *before* building
/// instances, because the step witnesses embed γ and the running
/// products in `x`. Runs the identical transcript on a scratch lane.
pub fn derive_segment_gamma(
    prep: &Preprocessing,
    audit: &UncompressedAudit,
    d_pre: [[F; 4]; 3],
) -> Result<[K; 2], SegmentError> {
    let state = &audit.proof.state;
    let mut lane = state.nebula.clone().ok_or(SegmentError::LaneMissing)?;
    let cfg = prep.nebula().ok_or(SegmentError::LaneMissing)?;
    lane.open_segment(cfg, prep.vk.digest(), state.z_i, state.acc_digest, d_pre)?;
    Ok(lane.gamma.expect("open_segment squeezed γ"))
}

/// Prove one segment window: fold the trace's `N` steps into the chain,
/// opening the segment on the first extend. On return the lane has
/// closed (product equation, `D_seen == D_pre`, boundary continuity all
/// checked by the lifecycle's own transition) and the chain is ready for
/// the next segment or finalization.
pub fn prove_segment(
    prep: &Preprocessing,
    plan: &NebulaPlan,
    audit: UncompressedAudit,
    trace: &SegmentTrace,
) -> Result<UncompressedAudit, SegmentError> {
    let params = plan.params();
    if trace.params() != params {
        return Err(SegmentError::PlanMismatch);
    }
    if prep.structure().m != plan.circuit().cols() || prep.structure().n != plan.circuit().rows() {
        return Err(SegmentError::StructureMismatch);
    }
    let lane = audit
        .proof
        .state
        .nebula
        .as_ref()
        .ok_or(SegmentError::LaneMissing)?;
    if trace.seg_idx != lane.seg_idx || trace.ts_in != lane.ts {
        return Err(SegmentError::ChainPositionMismatch {
            trace_seg: trace.seg_idx,
            trace_ts: trace.ts_in,
            lane_seg: lane.seg_idx,
            lane_ts: lane.ts,
        });
    }

    // Pass 1 (spec §1): lane encodings and MSIS-binding commitments for
    // every step, before γ exists anywhere.
    let (advs, d_pre) = commit_segment_lanes(plan, trace)?;

    // Commit-then-challenge (spec §6.2): γ from the chain transcript over
    // the claimed D_pre — the same squeeze the lifecycle will replay.
    let gamma = derive_segment_gamma(prep, &audit, d_pre)?;

    // Pass 2 + deposit: the whole segment, opening on the first chunk.
    let carry = StepCarry {
        next_step: 0,
        ts_in: trace.ts_in,
        h_in: [K::ONE; 4],
        sp_in: [0; 2],
    };
    fold_steps(prep, plan, trace, gamma, &advs, carry, audit, Some(d_pre))
}

/// Resume a segment whose chain paused mid-segment (spec §6.4's prover
/// resume): some chunks already folded, γ squeezed, segment not yet
/// closed. The carried lane **is** the checkpoint — γ, `D_pre`, the step
/// index, and the `ts`/`h`/`sp` carry all live in `state.nebula` — so
/// the only thing the caller must re-supply is the segment's trace, and
/// the only thing to validate is that this trace reproduces the open
/// segment's pre-committed chains (recompute-vs-`D_pre`; a wrong or
/// mutated trace cannot pass, because γ was squeezed over `D_pre`).
///
/// Completes the segment: on return the lane has closed, exactly as if
/// `prove_segment` had run uninterrupted.
pub fn resume_segment(
    prep: &Preprocessing,
    plan: &NebulaPlan,
    audit: UncompressedAudit,
    trace: &SegmentTrace,
) -> Result<UncompressedAudit, SegmentError> {
    let params = plan.params();
    if trace.params() != params {
        return Err(SegmentError::PlanMismatch);
    }
    if prep.structure().m != plan.circuit().cols() || prep.structure().n != plan.circuit().rows() {
        return Err(SegmentError::StructureMismatch);
    }
    let lane = audit
        .proof
        .state
        .nebula
        .as_ref()
        .ok_or(SegmentError::LaneMissing)?;
    let Some(gamma) = lane.gamma else {
        return Err(SegmentError::NotMidSegment);
    };
    if lane.seg_idx != trace.seg_idx || lane.idx == 0 || lane.idx as usize >= params.steps_per_segment() {
        return Err(SegmentError::NotMidSegment);
    }

    // The trace must be the segment the chain opened: its lane
    // commitments must chain to the exact `D_pre` γ was squeezed over.
    let (advs, d_pre) = commit_segment_lanes(plan, trace)?;
    if d_pre != lane.d_pre {
        return Err(SegmentError::ResumeTraceMismatch);
    }

    let carry = StepCarry {
        next_step: lane.idx as usize,
        ts_in: lane.ts,
        h_in: lane.h,
        sp_in: lane.sp,
    };
    fold_steps(prep, plan, trace, gamma, &advs, carry, audit, None)
}

/// Pass 1 (spec §1): per-step lane encodings and their MSIS-binding
/// commitments for the whole segment, plus the `D_pre` chains — all
/// γ-independent.
fn commit_segment_lanes(
    plan: &NebulaPlan,
    trace: &SegmentTrace,
) -> Result<(Vec<neo_ccs::LaneCommitments<neo_ajtai::Commitment>>, [[F; 4]; 3]), SegmentError> {
    let params = plan.params();
    let n = params.steps_per_segment();
    let mut advs = Vec::with_capacity(n);
    for i in 0..n {
        let ops_bits = params.encode_ops_lane(trace.step_ops(i))?;
        let is_bits = params.encode_scan_lane(&trace.is_cells[i * params.b_scan..(i + 1) * params.b_scan])?;
        let fs_bits = params.encode_scan_lane(&trace.fs_cells[i * params.b_scan..(i + 1) * params.b_scan])?;
        advs.push(plan.scheme().commit_bits(&ops_bits, &is_bits, &fs_bits)?);
    }
    let d_pre = digest::nebula_lane_chains(advs.iter());
    Ok((advs, d_pre))
}

/// The carry entering the next step to build — `x`-threading state
/// (spec §4.4). At segment open everything is at its reset value; on
/// resume it is read straight off the carried lane.
struct StepCarry {
    next_step: usize,
    ts_in: u64,
    h_in: [K; 4],
    sp_in: [u64; 2],
}

/// Pass 2 + deposit (spec §1): build the step witnesses from
/// `carry.next_step` to the segment's end and fold them in chunks of
/// the fold arity — SuperNeo multi-folding (Theorem 1's `CCS(b)^K`
/// arity, spec §6.3 chunking note): one recursion step covers
/// up to `max_fresh_count` S_mem steps (61 at the Goldilocks preset).
/// The lane transition is chunk-agnostic (`advance_for_batch` walks the
/// deposited claims in order); `open_d_pre` rides the first chunk when
/// this call opens the segment (`None` on resume — γ is already
/// squeezed).
#[allow(clippy::too_many_arguments)]
fn fold_steps(
    prep: &Preprocessing,
    plan: &NebulaPlan,
    trace: &SegmentTrace,
    gamma: [K; 2],
    advs: &[neo_ccs::LaneCommitments<neo_ajtai::Commitment>],
    carry: StepCarry,
    audit: UncompressedAudit,
    open_d_pre: Option<[[F; 4]; 3]>,
) -> Result<UncompressedAudit, SegmentError> {
    let params = plan.params();
    let gammas = Gammas {
        gamma1: gamma[0],
        gamma2: gamma[1],
    };

    let n = params.steps_per_segment();
    let mut instances = Vec::with_capacity(n - carry.next_step);
    let mut ts_in = carry.ts_in;
    let mut h_in = carry.h_in;
    let mut sp_in = carry.sp_in;
    for i in carry.next_step..n {
        let data = StepData {
            seg_idx: trace.seg_idx,
            idx: i as u64,
            ts_in,
            h_in,
            sp_in,
            ops: trace.step_ops(i),
            is_cells: &trace.is_cells[i * params.b_scan..(i + 1) * params.b_scan],
            fs_cells: &trace.fs_cells[i * params.b_scan..(i + 1) * params.b_scan],
        };
        let (z, x) = plan.circuit().witness(&gammas, &data)?;
        ts_in = x.ts_out;
        h_in = x.h_out;
        sp_in = x.sp_out;

        let mut instance = CcsInstance::from_low_norm_assignment(
            &prep.params,
            &prep.log,
            prep.structure(),
            &z,
            plan.circuit().m_in(),
        )?;
        instance.claim.adv = Some(advs[i].clone());
        debug_assert_eq!(
            plan.scheme()
                .commit(&instance.witness.Z)
                .expect("witness lanes commit"),
            advs[i],
            "bit-level and witness-slice lane commits must agree"
        );
        instances.push(instance);
    }

    let max_batch = prep.params.max_fresh_count().max(1);
    let mut audit = audit;
    let mut instances = instances.into_iter();
    let mut open_d_pre = open_d_pre;
    loop {
        let batch: Vec<CcsInstance> = instances.by_ref().take(max_batch).collect();
        if batch.is_empty() {
            break;
        }
        audit = match open_d_pre.take() {
            Some(d_pre) => lifecycle::extend_nebula_open(prep, audit, batch, d_pre)?,
            None => lifecycle::extend(prep, audit, batch)?,
        };
    }
    Ok(audit)
}
