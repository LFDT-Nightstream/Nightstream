//! App-agnostic recursive-step enforcement plan.
//!
//! Given a base image config (region sizes), produces the full
//! [`FPrimeImageConfig`] for a real F' recursive step: every one-shot
//! Poseidon hash is enforced with preimage sources read from the
//! committed F' image regions, and the resulting digest is bound to the
//! corresponding state-out lane. App frontends
//! ([`crate::frontends::fibonacci_f_prime`],
//! [`crate::frontends::r1cs_f_prime`]) supply the per-app
//! [`RecursiveStepImagePlan`] and reuse this builder.
//!
//! Currently emits enforcements for the three state-advance hashes
//! whose preimages route fully through F' image regions plus the
//! protocol's domain-tag constants:
//!
//! 1. `boundary_update`: `H(tag, prev_z_i, chunk_digest) → new_z_i`
//! 2. `public_trace_update`: `H(tag, prev_public_trace, chunk_digest) → new_public_trace`
//! 3. `accumulator_from_parent_c_data`: `H(tag, child_count, c_data_len, c_data...) → new_acc_digest`
//!
//! Not yet emitted (lane sources need additional plumbing):
//! - `state_x_out` (uses `u64_halves` packing for counters; needs a
//!   `U64HalfLanes` source variant before binding).
//!
//! ## Invariants this module relies on
//!
//! Lane indices within `state_lanes` (24 state-in digest lanes + 14
//! state-out lanes + 4 chunk-digest lanes = 42 total):
//!
//! - state-in digests: vk_fs (0..4), structure (4..8), z_0 (8..12),
//!   z_i_in (12..16), acc_digest_in (16..20), public_trace_in (20..24)
//! - state-out: new_chunk_count (24), new_step_count (25), new_z_i
//!   (26..30), new_public_trace (30..34), new_acc_digest (34..38)
//! - chunk_digest: (38..42)
//!
//! These match `fibonacci_structure::collect_state_lane_slots`.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::frontends::f_prime_shell::image::{
    FPrimeImageConfig, NifsPayloadShape, OneShotDigestToPublicXOutBinding, OneShotDigestToStateOutBinding,
    PoseidonPreimageLaneSource, PoseidonTransitionEnforcement, StateOutDigestTarget,
};
use crate::paper::digest::pack_bytes_as_fields;
use crate::paper::f_prime::ring_action_trace::RingActionTraceLayout;

/// Lane index of `vk_fs_digest[0]` in `state_lanes`.
pub const STATE_LANE_VK_FS_BASE: usize = 0;
/// Lane index of `structure_digest[0]` in `state_lanes`.
pub const STATE_LANE_STRUCTURE_BASE: usize = 4;
/// Lane index of `z_0[0]` in `state_lanes` (initial boundary).
pub const STATE_LANE_Z_0_BASE: usize = 8;
/// Lane index of `z_i_in[0]` in `state_lanes`.
pub const STATE_LANE_Z_I_IN_BASE: usize = 12;
/// Lane index of `public_trace_in[0]` in `state_lanes`.
pub const STATE_LANE_PUBLIC_TRACE_IN_BASE: usize = 20;
/// Lane index of `new_chunk_count` in `state_lanes` (start of state-out).
pub const STATE_LANE_NEW_CHUNK_COUNT: usize = 24;
/// Lane index of `new_step_count` in `state_lanes`.
pub const STATE_LANE_NEW_STEP_COUNT: usize = 25;
/// Lane index of `new_z_i[0]` in `state_lanes`.
pub const STATE_LANE_NEW_Z_I_BASE: usize = 26;
/// Lane index of `new_public_trace[0]` in `state_lanes`.
pub const STATE_LANE_NEW_PUBLIC_TRACE_BASE: usize = 30;
/// Lane index of `new_acc_digest[0]` in `state_lanes`.
pub const STATE_LANE_NEW_ACC_DIGEST_BASE: usize = 34;
/// Lane index of `chunk_digest[0]` in `state_lanes` (start of the
/// chunk-digest sub-region).
pub const STATE_LANE_CHUNK_DIGEST_BASE: usize = 38;
/// Four lanes per Goldilocks digest.
pub const DIGEST_LANE_COUNT: usize = 4;

/// Domain-separation tag for the boundary-update hash.
const BOUNDARY_UPDATE_TAG: &[u8] = b"neo.fold.clean/boundary_update/v1";
/// Domain-separation tag for the public-trace-update hash.
const PUBLIC_TRACE_UPDATE_TAG: &[u8] = b"neo.fold.clean/public_trace_update/v1";
/// Domain-separation tag for the parent-authority accumulator hash.
const ACCUMULATOR_TAG: &[u8] = b"neo.fold.next/direct_ccs/accumulator_phi_dec_parent/v1";
/// Domain-separation tag for the recursive `state_x_out` hash.
const STATE_X_OUT_TAG: &[u8] = b"neo.fold.clean/state_x_out/v1";

/// Build the preimage-source list for the `boundary_update` hash.
///
/// Preimage layout:
///   `pack_bytes_as_fields(tag)` ‖ `prev_z_i (4)` ‖ `chunk_digest (4)`
pub fn boundary_update_preimage_sources() -> Vec<PoseidonPreimageLaneSource> {
    let header = pack_bytes_as_fields(BOUNDARY_UPDATE_TAG);
    let mut sources: Vec<PoseidonPreimageLaneSource> = header
        .iter()
        .map(|&v| PoseidonPreimageLaneSource::Constant(v))
        .collect();
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(STATE_LANE_Z_I_IN_BASE + i));
    }
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(STATE_LANE_CHUNK_DIGEST_BASE + i));
    }
    sources
}

/// Build the preimage-source list for the `public_trace_update` hash.
///
/// Preimage layout:
///   `pack_bytes_as_fields(tag)` ‖ `prev_public_trace (4)` ‖ `chunk_digest (4)`
pub fn public_trace_update_preimage_sources() -> Vec<PoseidonPreimageLaneSource> {
    let header = pack_bytes_as_fields(PUBLIC_TRACE_UPDATE_TAG);
    let mut sources: Vec<PoseidonPreimageLaneSource> = header
        .iter()
        .map(|&v| PoseidonPreimageLaneSource::Constant(v))
        .collect();
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(
            STATE_LANE_PUBLIC_TRACE_IN_BASE + i,
        ));
    }
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(STATE_LANE_CHUNK_DIGEST_BASE + i));
    }
    sources
}

/// Build the preimage-source list for `accumulator_from_parent_c_data`.
///
/// Per `paper::digest::accumulator_from_parent_c_data` (and mirrored in
/// the Phase 1.3d test fixture):
///   `pack_bytes_as_fields(tag)` ‖ `child_count` ‖
///   if `child_count > 0`: `c_data.len()` ‖ `c_data[..]`
///
/// `child_count` and `c_data.len()` are baked as `Constant` for now —
/// proper plumbing of `running.claims.len()` and the ce-claim shape
/// header into image regions is a follow-up slice. The c_data entries
/// themselves source from `nifs_payload_lanes`, so the variable part
/// of the hash is bound to committed image state.
/// Preimage source list for the **base** accumulator hash — the
/// empty-accumulator case where `child_count = 0` and no c_data is
/// absorbed. Matches `accumulator_digest_from_claims(_, &[])`'s
/// preimage exactly.
pub fn base_accumulator_preimage_sources() -> Vec<PoseidonPreimageLaneSource> {
    let header = pack_bytes_as_fields(ACCUMULATOR_TAG);
    let mut sources: Vec<PoseidonPreimageLaneSource> = header
        .iter()
        .map(|&v| PoseidonPreimageLaneSource::Constant(v))
        .collect();
    sources.push(PoseidonPreimageLaneSource::Constant(F::from_u64(0)));
    sources
}

pub fn accumulator_preimage_sources(
    child_count: u64,
    ce_claim_payload_index: usize,
    c_data_entries: usize,
) -> Vec<PoseidonPreimageLaneSource> {
    let header = pack_bytes_as_fields(ACCUMULATOR_TAG);
    let mut sources: Vec<PoseidonPreimageLaneSource> = header
        .iter()
        .map(|&v| PoseidonPreimageLaneSource::Constant(v))
        .collect();
    sources.push(PoseidonPreimageLaneSource::Constant(F::from_u64(child_count)));
    if child_count > 0 {
        sources.push(PoseidonPreimageLaneSource::Constant(F::from_u64(c_data_entries as u64)));
        // Both CcsClaim and CeClaim NIFS-payload layouts start with
        // `d, kappa, c_data_len` (3 u64 lanes), then `c_data` entries.
        const C_DATA_FIELD_OFFSET_IN_PAYLOAD: usize = 3;
        for i in 0..c_data_entries {
            sources.push(PoseidonPreimageLaneSource::NifsPayloadLane {
                payload_index: ce_claim_payload_index,
                lane_index: C_DATA_FIELD_OFFSET_IN_PAYLOAD + i,
            });
        }
    }
    sources
}

/// Build the preimage-source list for the recursive `state_x_out` hash.
///
/// Preimage layout (matches `paper::digest::state_x_out_digest`):
/// ```text
///   pack_bytes_as_fields(tag)
///   ‖ vk_fs (4)              [state_lanes 0..4]
///   ‖ structure (4)          [state_lanes 4..8]
///   ‖ chunk_count halves (2) [state_lanes[24] split into low_32, high_32]
///   ‖ step_count halves (2)  [state_lanes[25] split]
///   ‖ z_0 (4)                [state_lanes 8..12]    initial_boundary
///   ‖ new_z_i (4)            [state_lanes 26..30]   current_boundary
///   ‖ pc halves (2)          [Constant fixture/per-step values]
///   ‖ new_acc_digest (4)     [state_lanes 34..38]   semantic_acc
///   ‖ new_acc_digest (4)     [state_lanes 34..38]   construction2_acc (repeated)
///   ‖ new_public_trace (4)   [state_lanes 30..34]
/// ```
///
/// `pc` is baked as `Constant` because the recursive image stores it in
/// boundary (the source-image boundary region) and we don't yet have a boundary
/// lane source variant — proper plumbing is a follow-up.
pub fn state_x_out_preimage_sources(pc: u64) -> Vec<PoseidonPreimageLaneSource> {
    state_x_out_preimage_sources_with_app_x(pc, &[])
}

/// Variant of [`state_x_out_preimage_sources`] that appends app-level
/// public-input lanes to the preimage.
///
/// `app_public_input_var_indices` lists the app-assignment variable
/// indices `j` whose 64-bit canonical-u64 lane should be absorbed into
/// the `state_x_out` Poseidon hash after the chain-coordinate prefix.
/// Each index resolves through
/// [`PoseidonPreimageLaneSource::AppAssignmentLane`] to
/// `lane_slots.app_assignment_lanes[j]`, i.e. the 64 committed bits at
/// `layout.app_private.offset + j * 64`.
///
/// This is how an R1CS frontend binds its public input `x = z[..m_in]`
/// to the verifier-visible `public_output_digest` (without that
/// binding, two assignments with different `x` but the same R1CS
/// shape produce the same `state_x_out` digest — a soundness gap).
pub fn state_x_out_preimage_sources_with_app_x(
    pc: u64,
    app_public_input_var_indices: &[usize],
) -> Vec<PoseidonPreimageLaneSource> {
    let header = pack_bytes_as_fields(STATE_X_OUT_TAG);
    let mut sources: Vec<PoseidonPreimageLaneSource> = header
        .iter()
        .map(|&v| PoseidonPreimageLaneSource::Constant(v))
        .collect();
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(STATE_LANE_VK_FS_BASE + i));
    }
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(STATE_LANE_STRUCTURE_BASE + i));
    }
    sources.push(PoseidonPreimageLaneSource::StateLaneLowHalf(STATE_LANE_NEW_CHUNK_COUNT));
    sources.push(PoseidonPreimageLaneSource::StateLaneHighHalf(
        STATE_LANE_NEW_CHUNK_COUNT,
    ));
    sources.push(PoseidonPreimageLaneSource::StateLaneLowHalf(STATE_LANE_NEW_STEP_COUNT));
    sources.push(PoseidonPreimageLaneSource::StateLaneHighHalf(STATE_LANE_NEW_STEP_COUNT));
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(STATE_LANE_Z_0_BASE + i));
    }
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(STATE_LANE_NEW_Z_I_BASE + i));
    }
    sources.push(PoseidonPreimageLaneSource::Constant(F::from_u64(pc & 0xffff_ffff)));
    sources.push(PoseidonPreimageLaneSource::Constant(F::from_u64(pc >> 32)));
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(
            STATE_LANE_NEW_ACC_DIGEST_BASE + i,
        ));
    }
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(
            STATE_LANE_NEW_ACC_DIGEST_BASE + i,
        ));
    }
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(
            STATE_LANE_NEW_PUBLIC_TRACE_BASE + i,
        ));
    }
    for &var_idx in app_public_input_var_indices {
        sources.push(PoseidonPreimageLaneSource::AppAssignmentLane(var_idx));
    }
    sources
}

/// Caller-supplied parameters that the recursive plan can't synthesize.
#[derive(Clone, Debug)]
pub struct RecursiveStepImagePlan {
    pub limbs: usize,
    pub boundary_bits: usize,
    pub kmul_count: usize,
    pub ring_action_pair_count: usize,
    pub ring_action_pair_layout: RingActionTraceLayout,
    pub sponge_transcript_permutes: usize,
    /// NIFS-payload shapes (in fill order). If `accumulator` is
    /// `Some`, the payload at its `ce_claim_payload_index` must be a
    /// CcsClaim or CeClaim whose `c_data_entries` matches the
    /// accumulator's.
    pub nifs_payload_shapes: Vec<NifsPayloadShape>,
    /// If `Some`, the plan emits the accumulator enforcement.
    pub accumulator: Option<AccumulatorPlanOptions>,
    /// If `Some`, the plan emits the `state_x_out` enforcement and
    /// the public-x_out digest binding.
    pub state_x_out: Option<StateXOutPlanOptions>,
}

/// Accumulator-enforcement parameters.
#[derive(Clone, Debug)]
pub struct AccumulatorPlanOptions {
    /// Index into `nifs_payload_shapes` of the ce-claim that holds the
    /// `c_data` referenced by this hash's preimage.
    pub ce_claim_payload_index: usize,
    /// Number of c_data entries to read from that payload.
    pub c_data_entries: usize,
    /// `running.claims.len()` at the step being enforced.
    pub child_count: u64,
    /// Unified-mode flag. When `true`, the recursive-step image config
    /// emits **two** accumulator Poseidon enforcements (one with the
    /// empty-preimage `H(tag, 0)`, one with the recursive preimage
    /// `H(tag, child_count, c_data_entries, c_data ...)`) and pushes a
    /// [`crate::frontends::f_prime_shell::image::UnifiedAccumulatorSelector`]
    /// onto the resulting config so the structure builder emits the
    /// selector product rows over the `is_base` lane. When `false`,
    /// the legacy single-accumulator path applies.
    pub unified: bool,
}

/// `state_x_out` enforcement parameters.
#[derive(Clone, Debug)]
pub struct StateXOutPlanOptions {
    /// Program counter at the step being enforced. Baked as `Constant`
    /// in the source list until a boundary lane source variant lands.
    pub pc: u64,
    /// Bit offsets in the image's `values` where the 4 public-x_out
    /// digest lanes live. Must lie inside the boundary region.
    pub public_x_out_lane_bit_starts: [usize; 4],
    /// Indices of app-assignment variables to append to the
    /// `state_x_out` Poseidon preimage as canonical-u64 lanes. Each
    /// index `j` binds the 64 committed bits at
    /// `layout.app_private.offset + j * 64` into the verifier-visible
    /// `public_output_digest`.
    ///
    /// Used by app frontends with a "verifier-supplied public input"
    /// semantics (R1CS). Defaults to empty — no extra binding.
    pub app_public_input_var_indices: Vec<usize>,
}

/// Assemble the recursive-step `FPrimeImageConfig` from the
/// plan. Produces enforcements for boundary_update, public_trace_update,
/// and (if requested) accumulator_from_parent_c_data, plus matching
/// state-out digest bindings.
pub fn build_recursive_step_image_config(plan: &RecursiveStepImagePlan) -> FPrimeImageConfig {
    let boundary_sources = boundary_update_preimage_sources();
    let public_trace_sources = public_trace_update_preimage_sources();

    let mut preimage_lens = vec![boundary_sources.len(), public_trace_sources.len()];
    let mut enforcements = vec![
        PoseidonTransitionEnforcement {
            one_shot_index: 0,
            preimage_lanes: boundary_sources,
        },
        PoseidonTransitionEnforcement {
            one_shot_index: 1,
            preimage_lanes: public_trace_sources,
        },
    ];
    let mut state_out_bindings = vec![
        OneShotDigestToStateOutBinding {
            one_shot_index: 0,
            state_out_target: StateOutDigestTarget::NewZI,
        },
        OneShotDigestToStateOutBinding {
            one_shot_index: 1,
            state_out_target: StateOutDigestTarget::NewPublicTrace,
        },
    ];

    let mut unified_selector: Option<crate::frontends::f_prime_shell::image::UnifiedAccumulatorSelector> = None;
    if let Some(acc) = &plan.accumulator {
        if acc.unified {
            // Unified mode: emit two accumulator Poseidon enforcements.
            // The structure builder picks between their digests via the
            // `is_base` selector, so we DO NOT push a direct
            // NewAccDigest state-out binding.
            let base_sources = base_accumulator_preimage_sources();
            preimage_lens.push(base_sources.len());
            enforcements.push(PoseidonTransitionEnforcement {
                one_shot_index: 2,
                preimage_lanes: base_sources,
            });
            let rec_sources =
                accumulator_preimage_sources(acc.child_count, acc.ce_claim_payload_index, acc.c_data_entries);
            preimage_lens.push(rec_sources.len());
            enforcements.push(PoseidonTransitionEnforcement {
                one_shot_index: 3,
                preimage_lanes: rec_sources,
            });
            unified_selector = Some(crate::frontends::f_prime_shell::image::UnifiedAccumulatorSelector {
                base_trace_index: 2,
                recursive_trace_index: 3,
            });
        } else {
            // Legacy single-accumulator path.
            let acc_sources =
                accumulator_preimage_sources(acc.child_count, acc.ce_claim_payload_index, acc.c_data_entries);
            preimage_lens.push(acc_sources.len());
            enforcements.push(PoseidonTransitionEnforcement {
                one_shot_index: 2,
                preimage_lanes: acc_sources,
            });
            state_out_bindings.push(OneShotDigestToStateOutBinding {
                one_shot_index: 2,
                state_out_target: StateOutDigestTarget::NewAccDigest,
            });
        }
    }

    let mut public_x_out_bindings: Vec<OneShotDigestToPublicXOutBinding> = Vec::new();
    if let Some(sxo) = &plan.state_x_out {
        assert!(
            plan.accumulator.is_some(),
            "state_x_out enforcement requires accumulator enforcement to be configured first"
        );
        // Legacy plan: state_x_out at one_shot index 3; unified plan
        // pushes it to index 4 because the second accumulator
        // enforcement consumed index 3.
        let sxo_index = if plan
            .accumulator
            .as_ref()
            .map(|a| a.unified)
            .unwrap_or(false)
        {
            4
        } else {
            3
        };
        let sxo_sources = state_x_out_preimage_sources_with_app_x(sxo.pc, &sxo.app_public_input_var_indices);
        preimage_lens.push(sxo_sources.len());
        enforcements.push(PoseidonTransitionEnforcement {
            one_shot_index: sxo_index,
            preimage_lanes: sxo_sources,
        });
        public_x_out_bindings.push(OneShotDigestToPublicXOutBinding {
            one_shot_index: sxo_index,
            public_x_out_lane_bit_starts: sxo.public_x_out_lane_bit_starts,
        });
    }

    FPrimeImageConfig {
        limbs: plan.limbs,
        boundary_bits: plan.boundary_bits,
        nifs_payload_shapes: plan.nifs_payload_shapes.clone(),
        kmul_count: plan.kmul_count,
        ring_action_pair_count: plan.ring_action_pair_count,
        ring_action_pair_layout: plan.ring_action_pair_layout,
        poseidon_one_shot_preimage_lens: preimage_lens,
        sponge_transcript_permutes: plan.sponge_transcript_permutes,
        one_shot_digest_to_state_out_bindings: state_out_bindings,
        one_shot_digest_to_public_x_out_bindings: public_x_out_bindings,
        poseidon_transition_enforcements: enforcements,
        unified_accumulator_selector: unified_selector,
    }
}

// ── Preimage builders (for tests and downstream consumers) ───────────────

/// Compute the boundary-update preimage with the given `prev_z_i` and
/// `chunk_digest`. Mirrors `paper::digest::boundary_update_digest`'s
/// preimage construction so callers can encode a Poseidon trace whose
/// witness will satisfy the source-bound absorb rows.
pub fn build_boundary_update_preimage_fields(prev_z_i: [F; 4], chunk_digest: [F; 4]) -> Vec<F> {
    let mut p = pack_bytes_as_fields(BOUNDARY_UPDATE_TAG);
    p.extend(prev_z_i);
    p.extend(chunk_digest);
    p
}

/// Compute the public-trace-update preimage.
pub fn build_public_trace_update_preimage_fields(prev_public_trace: [F; 4], chunk_digest: [F; 4]) -> Vec<F> {
    let mut p = pack_bytes_as_fields(PUBLIC_TRACE_UPDATE_TAG);
    p.extend(prev_public_trace);
    p.extend(chunk_digest);
    p
}

/// Compute the accumulator preimage. Returns the exact field-vector
/// `paper::digest::accumulator_from_parent_c_data` hashes.
pub fn build_accumulator_preimage_fields(child_count: u64, c_data: &[F]) -> Vec<F> {
    let mut p = pack_bytes_as_fields(ACCUMULATOR_TAG);
    p.push(F::from_u64(child_count));
    if child_count > 0 {
        p.push(F::from_u64(c_data.len() as u64));
        p.extend_from_slice(c_data);
    }
    p
}

/// Compute the `state_x_out` preimage. Returns the exact field-vector
/// `paper::digest::state_x_out_digest` hashes. Callers pass the *actual*
/// values committed in image regions (state-in, state-out, etc.) so the
/// trace encoded from this preimage matches the source-bound absorb
/// rows the planner emits.
#[allow(clippy::too_many_arguments)]
pub fn build_state_x_out_preimage_fields(
    vk_fs_digest: [F; 4],
    structure_digest: [F; 4],
    new_chunk_count: u64,
    new_step_count: u64,
    z_0: [F; 4],
    new_z_i: [F; 4],
    pc: u64,
    new_acc_digest: [F; 4],
    new_public_trace: [F; 4],
) -> Vec<F> {
    build_state_x_out_preimage_fields_with_app_x(
        vk_fs_digest,
        structure_digest,
        new_chunk_count,
        new_step_count,
        z_0,
        new_z_i,
        pc,
        new_acc_digest,
        new_public_trace,
        &[],
    )
}

/// Variant of [`build_state_x_out_preimage_fields`] that appends the
/// canonical-u64 values of `app_public_input` to the preimage in fill
/// order. Mirrors [`state_x_out_preimage_sources_with_app_x`]'s shape
/// so the trace encoded from this preimage matches the absorb rows the
/// planner emits when `StateXOutPlanOptions::app_public_input_var_indices`
/// is non-empty.
///
/// Each entry must be a canonical Goldilocks element whose value fits
/// in 64 bits unsigned (the structure binds it through the 64 bits
/// stored in `app_private`).
#[allow(clippy::too_many_arguments)]
pub fn build_state_x_out_preimage_fields_with_app_x(
    vk_fs_digest: [F; 4],
    structure_digest: [F; 4],
    new_chunk_count: u64,
    new_step_count: u64,
    z_0: [F; 4],
    new_z_i: [F; 4],
    pc: u64,
    new_acc_digest: [F; 4],
    new_public_trace: [F; 4],
    app_public_input: &[F],
) -> Vec<F> {
    let mut p = pack_bytes_as_fields(STATE_X_OUT_TAG);
    p.extend(vk_fs_digest);
    p.extend(structure_digest);
    p.push(F::from_u64(new_chunk_count & 0xffff_ffff));
    p.push(F::from_u64(new_chunk_count >> 32));
    p.push(F::from_u64(new_step_count & 0xffff_ffff));
    p.push(F::from_u64(new_step_count >> 32));
    p.extend(z_0);
    p.extend(new_z_i);
    p.push(F::from_u64(pc & 0xffff_ffff));
    p.push(F::from_u64(pc >> 32));
    p.extend(new_acc_digest);
    p.extend(new_acc_digest);
    p.extend(new_public_trace);
    p.extend_from_slice(app_public_input);
    p
}
