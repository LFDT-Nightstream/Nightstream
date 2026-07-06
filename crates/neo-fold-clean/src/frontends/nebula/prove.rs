//! The segment prover — spec §1's two-pass flow, per segment window.
//!
//! Owns: turning one [`SegmentTrace`] (the native pass) into `N` folded
//! `S_mem` instances: lane encodings → lane commitments → `D_pre` → γ →
//! step witnesses with running products → `extend` loop. Also the γ
//! pre-derivation ([`derive_segment_gamma`]) the instances need before
//! the lifecycle's own `open_segment` runs.
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
    let n = params.steps_per_segment();
    let mut advs = Vec::with_capacity(n);
    for i in 0..n {
        let ops_bits = params.encode_ops_lane(trace.step_ops(i))?;
        let is_bits = params.encode_scan_lane(&trace.is_cells[i * params.b_scan..(i + 1) * params.b_scan])?;
        let fs_bits = params.encode_scan_lane(&trace.fs_cells[i * params.b_scan..(i + 1) * params.b_scan])?;
        advs.push(plan.scheme().commit_bits(&ops_bits, &is_bits, &fs_bits)?);
    }
    let d_pre = digest::nebula_lane_chains(advs.iter());

    // Commit-then-challenge (spec §6.2): γ from the chain transcript over
    // the claimed D_pre — the same squeeze the lifecycle will replay.
    let gamma = derive_segment_gamma(prep, &audit, d_pre)?;
    let gammas = Gammas {
        gamma1: gamma[0],
        gamma2: gamma[1],
    };

    // Pass 2: step witnesses with γ-dependent running products, built in
    // step order (`ts`/`h`/`sp` chain through consecutive x's; stacks
    // open at 0 — they are segment-local, spec §3.1).
    let mut instances = Vec::with_capacity(n);
    let mut ts_in = trace.ts_in;
    let mut h_in = [K::ONE; 4];
    let mut sp_in = [0u64; 2];
    for i in 0..n {
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

    // Deposit in chunks of the fold arity — Nebula §5's amortization: one
    // recursion step covers up to `max_fresh_count` S_mem steps (61 at
    // the Goldilocks preset), so a segment costs ⌈N / max_fresh⌉ F′ steps
    // instead of N. The lane transition is chunk-agnostic
    // (`advance_for_batch` walks the deposited claims in order); the
    // first chunk carries the segment-open payload.
    let max_batch = prep.params.max_fresh_count().max(1);
    let mut audit = audit;
    let mut instances = instances.into_iter();
    let mut first = true;
    loop {
        let batch: Vec<CcsInstance> = instances.by_ref().take(max_batch).collect();
        if batch.is_empty() {
            break;
        }
        audit = if first {
            first = false;
            lifecycle::extend_nebula_open(prep, audit, batch, d_pre)?
        } else {
            lifecycle::extend(prep, audit, batch)?
        };
    }
    Ok(audit)
}
