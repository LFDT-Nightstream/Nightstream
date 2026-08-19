//! App-agnostic recursive-step enforcement plan.
//!
//! Given a base image config (region sizes), produces the full
//! [`FPrimeImageConfig`] for a real F' recursive step: every one-shot
//! Poseidon hash is enforced with preimage sources read from the
//! committed F' image regions, and the resulting digest is bound to the
//! corresponding state-out lane. App frontends
//! ([`fibonacci_f_prime`],
//! [`crate::frontends::r1cs_f_prime`]) supply the per-app
//! [`RecursiveStepImagePlan`] and reuse this builder.
//!
//! Currently emits enforcements for the state-output hash whose
//! preimage routes fully through F' image regions plus the protocol's
//! compact domain constant:
//!
//! 1. `state_x_out`: `H(vk_fs, state, new_acc_digest, …) → public_x_out`
//!
//! The local chunk-shape coordinate `new_z_i` is constrained linearly to
//! `chunk_digest` by the structure builder. Content authority lives on
//! the NIFS accumulator path (`new_acc_digest`), so canonical unified
//! mode does not spend a producer-side Poseidon trace on
//! `H(prev_z_i, chunk_digest)`.
//!
//! The recursive accumulator handle (`new_acc_digest`) is deliberately not
//! recomputed by a producer-side `H(parent.c_data...)` trace in this image.
//! The composed decider consumes it at the next recursive step (or terminal
//! fold) and checks it against the actual NIFS.V running accumulator.
//!
//! ## Invariants this module relies on
//!
//! Lane indices within `state_lanes` (28 state-in digest lanes + 18
//! state-out lanes + 4 chunk-digest lanes = 50 total):
//!
//! - state-in digests: vk_fs (0..4), structure (4..8), z_0 (8..12),
//!   z_i_in (12..16), semantic_state_digest_in (16..20),
//!   acc_digest_in (20..24), public_trace_in (24..28)
//! - state-out: new_chunk_count (28), new_step_count (29), new_z_i
//!   (30..34), new_public_trace (34..38), new_semantic_state_digest
//!   (38..42), new_acc_digest (42..46)
//! - chunk_digest: (46..50)
//!
//! These match `fibonacci_structure::collect_state_lane_slots`.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use crate::frontends::f_prime::image::{
    FPrimeImageConfig, NifsPayloadShape, OneShotDigestToPublicXOutBinding, OneShotDigestToStateInBinding,
    OneShotDigestToStateOutBinding, PoseidonPreimageLaneSource, PoseidonTransitionEnforcement, StateInDigestTarget,
    StateOutDigestTarget,
};
use crate::paper::digest::{
    pack_bytes_as_fields, StateXOutDigestMode, F_PRIME_BOUNDARY_UPDATE_DOMAIN, F_PRIME_STATE_X_OUT_DOMAIN,
};
use crate::paper::f_prime::ring_action_trace::RingActionTraceLayout;

/// Lane index of `vk_fs_digest[0]` in `state_lanes`.
pub const STATE_LANE_VK_FS_BASE: usize = 0;
/// Lane index of `structure_digest[0]` in `state_lanes`.
pub const STATE_LANE_STRUCTURE_BASE: usize = 4;
/// Lane index of `z_0[0]` in `state_lanes` (initial boundary).
pub const STATE_LANE_Z_0_BASE: usize = 8;
/// Lane index of `z_i_in[0]` in `state_lanes`.
pub const STATE_LANE_Z_I_IN_BASE: usize = 12;
/// Lane index of `semantic_state_digest_in[0]` in `state_lanes`.
pub const STATE_LANE_SEMANTIC_STATE_IN_BASE: usize = 16;
/// Lane index of `acc_digest_in[0]` in `state_lanes`.
pub const STATE_LANE_ACC_DIGEST_IN_BASE: usize = 20;
/// Lane index of `public_trace_in[0]` in `state_lanes`.
pub const STATE_LANE_PUBLIC_TRACE_IN_BASE: usize = 24;
/// Lane index of `new_chunk_count` in `state_lanes` (start of state-out).
pub const STATE_LANE_NEW_CHUNK_COUNT: usize = 28;
/// Lane index of `new_step_count` in `state_lanes`.
pub const STATE_LANE_NEW_STEP_COUNT: usize = 29;
/// Lane index of `new_z_i[0]` in `state_lanes`.
pub const STATE_LANE_NEW_Z_I_BASE: usize = 30;
/// Lane index of `new_public_trace[0]` in `state_lanes`.
pub const STATE_LANE_NEW_PUBLIC_TRACE_BASE: usize = 34;
/// Lane index of `new_semantic_state_digest[0]` in `state_lanes`.
pub const STATE_LANE_NEW_SEMANTIC_STATE_BASE: usize = 38;
/// Lane index of `new_acc_digest[0]` in `state_lanes`.
pub const STATE_LANE_NEW_ACC_DIGEST_BASE: usize = 42;
/// Lane index of `chunk_digest[0]` in `state_lanes` (start of the
/// chunk-digest sub-region).
pub const STATE_LANE_CHUNK_DIGEST_BASE: usize = 46;
/// Four lanes per Goldilocks digest.
pub const DIGEST_LANE_COUNT: usize = 4;

/// Domain-separation tag for the public-trace-update hash.
///
/// Kept for native legacy helpers and parity tests. The canonical F'
/// source image no longer emits this as a separate one-shot trace:
/// `public_trace` follows `z_i`, so we do not spend a second bit-backed
/// Poseidon2 chain over the same `chunk_digest`.
const PUBLIC_TRACE_UPDATE_TAG: &[u8] = b"neo.fold.clean/public_trace_update/v1";
/// Domain-separation tag for the legacy parent-commitment accumulator hash.
const ACCUMULATOR_TAG: &[u8] = b"neo.fold.next/direct_ccs/accumulator_phi_dec_parent/v1";
/// Domain-separation tag for app semantic state digests.
const SEMANTIC_STATE_TAG: &[u8] = b"neo.fold.clean/semantic_state/v1";

/// Build the preimage-source list for the `boundary_update` hash.
///
/// Preimage layout:
///   `domain_id` ‖ `prev_z_i (4)` ‖ `chunk_digest (4)`
pub fn boundary_update_preimage_sources() -> Vec<PoseidonPreimageLaneSource> {
    let mut sources: Vec<PoseidonPreimageLaneSource> = vec![PoseidonPreimageLaneSource::Constant(F::from_u64(
        F_PRIME_BOUNDARY_UPDATE_DOMAIN,
    ))];
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

/// Build the preimage-source list for an app semantic-state hash over
/// R1CS assignment variables.
///
/// Preimage layout:
///   `pack_bytes_as_fields(tag)` ‖ app assignment lanes in caller order
pub fn semantic_state_preimage_sources(var_indices: &[usize]) -> Vec<PoseidonPreimageLaneSource> {
    let header = pack_bytes_as_fields(SEMANTIC_STATE_TAG);
    let mut sources: Vec<PoseidonPreimageLaneSource> = header
        .iter()
        .map(|&v| PoseidonPreimageLaneSource::Constant(v))
        .collect();
    for &var_idx in var_indices {
        sources.push(PoseidonPreimageLaneSource::AppAssignmentLane(var_idx));
    }
    sources
}

/// Build the preimage-source list for a semantic-state digest over app
/// public input bindings. Full-field bindings are appended first, followed
/// by little-endian packed Boolean chunks.
pub fn semantic_state_preimage_sources_with_app_public(
    app_public_input_var_indices: &[usize],
    app_public_input_bit_var_indices: &[usize],
) -> Vec<PoseidonPreimageLaneSource> {
    let header = pack_bytes_as_fields(SEMANTIC_STATE_TAG);
    let mut sources: Vec<PoseidonPreimageLaneSource> = header
        .iter()
        .map(|&v| PoseidonPreimageLaneSource::Constant(v))
        .collect();
    for &var_idx in app_public_input_var_indices {
        sources.push(PoseidonPreimageLaneSource::AppAssignmentLane(var_idx));
    }
    for chunk in app_public_input_bit_var_indices.chunks(POSEIDON2_GOLDILOCKS_BITS) {
        sources.push(PoseidonPreimageLaneSource::AppAssignmentBitPack(chunk.to_vec()));
    }
    sources
}

/// Native preimage for the semantic-state digest over app assignment
/// variables. Mirrors [`semantic_state_preimage_sources`].
pub fn build_semantic_state_preimage_fields(app_state: &[F]) -> Vec<F> {
    let mut p = pack_bytes_as_fields(SEMANTIC_STATE_TAG);
    p.extend_from_slice(app_state);
    p
}

/// Build the preimage-source list for the legacy parent-commitment
/// accumulator digest.
///
/// Legacy non-unified accumulator plans use this producer-side hash.
/// Canonical unified plans carry only an accumulator handle in
/// `state_out` and bind that handle when the next recursive step or
/// terminal fold consumes the actual NIFS.V running accumulator.
///
/// Legacy preimage (mirrored in the Phase 1.3d test fixture):
///   `pack_bytes_as_fields(tag)` ‖ `child_count` ‖
///   if `child_count > 0`: `c_data.len()` ‖ `c_data[..]`
///
/// `child_count` and `c_data.len()` are baked as `Constant` for now —
/// proper plumbing of `running.claims.len()` and the ce-claim shape
/// header into image regions is a follow-up slice. The c_data entries
/// themselves source from `nifs_payload_lanes`, so the variable part
/// of the hash is bound to committed image state.
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
///   domain_id
///   ‖ vk_fs (4)                     [state_lanes 0..4]
///   ‖ pi_ccs_header_bundle (4)       [state_lanes 4..8]
///   ‖ chunk_count halves (2)        [state_lanes[28] split]
///   ‖ step_count halves (2)         [state_lanes[29] split]
///   ‖ pc halves (2)                 [constant pc]
///   ‖ new_z_i (4)                   [state_lanes 30..34]   current_boundary
///   ‖ new_semantic_state_digest (4) [state_lanes 38..42]   semantic_acc
///   ‖ new_acc_digest (4)            [state_lanes 42..46]   construction2_acc
/// ```
///
/// `chunk_count` and `step_count` are both carried by `state_x_out`.
/// `step_count` is the paper-level iteration counter; `chunk_count` is
/// local scheduling state, but it is not verifier-derivable in the
/// non-replay verifier and therefore must be authenticated.
///
/// `pc` is absorbed directly as two u64 halves, matching HyperNova
/// Construction 2's recursive-link preimage. This build still uses a
/// constant `TRIVIAL_PC = 1`, but the source image keeps the binding
/// explicit.
///
/// `z_0` is intentionally not absorbed directly either. It is the
/// verifier-derived `initial_boundary_digest(structure_digest,
/// public_input_len)`, and both inputs are already absorbed into
/// `vk_fs_digest`.
pub fn state_x_out_preimage_sources(pc: u64) -> Vec<PoseidonPreimageLaneSource> {
    state_x_out_preimage_sources_with_mode(pc, StateXOutDigestMode::Stateful)
}

/// Variant retained for older call sites. App public inputs are no
/// longer appended here; R1CS-F' binds them through the semantic-state
/// digest so native `compute_x_out` and the F' CCS hash the same public
/// recursive link.
pub fn state_x_out_preimage_sources_with_app_x(
    pc: u64,
    _app_public_input_var_indices: &[usize],
    _app_public_input_bit_var_indices: &[usize],
) -> Vec<PoseidonPreimageLaneSource> {
    state_x_out_preimage_sources_with_mode(pc, StateXOutDigestMode::Stateful)
}

pub fn state_x_out_preimage_sources_with_mode(pc: u64, mode: StateXOutDigestMode) -> Vec<PoseidonPreimageLaneSource> {
    let mut sources: Vec<PoseidonPreimageLaneSource> = vec![PoseidonPreimageLaneSource::Constant(F::from_u64(
        F_PRIME_STATE_X_OUT_DOMAIN,
    ))];
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
    sources.push(PoseidonPreimageLaneSource::Constant(F::from_u64(pc & 0xffff_ffff)));
    sources.push(PoseidonPreimageLaneSource::Constant(F::from_u64(pc >> 32)));
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(STATE_LANE_NEW_Z_I_BASE + i));
    }
    if matches!(mode, StateXOutDigestMode::Stateful) {
        for i in 0..DIGEST_LANE_COUNT {
            sources.push(PoseidonPreimageLaneSource::StateLane(
                STATE_LANE_NEW_SEMANTIC_STATE_BASE + i,
            ));
        }
    }
    for i in 0..DIGEST_LANE_COUNT {
        sources.push(PoseidonPreimageLaneSource::StateLane(
            STATE_LANE_NEW_ACC_DIGEST_BASE + i,
        ));
    }
    sources
}

pub fn state_x_out_digest_mode_for_options(sxo: &StateXOutPlanOptions) -> StateXOutDigestMode {
    if sxo.semantic_state_in_var_indices.is_empty()
        && sxo.semantic_state_out_var_indices.is_empty()
        && sxo.app_public_input_var_indices.is_empty()
        && sxo.app_public_input_bit_var_indices.is_empty()
    {
        StateXOutDigestMode::Stateless
    } else {
        StateXOutDigestMode::Stateful
    }
}

/// Whether the low-norm F' source image must carry the plan's
/// `nifs_payload_shapes`.
///
/// Legacy non-unified accumulator plans hash `parent.c_data` from the
/// source-image payload, so they still need the payload lanes. Unified
/// plans use delayed accumulator-handle binding: NIFS authority lives in
/// the in-circuit NIFS verifier messages, and the source image only
/// carries the compact state handles consumed by the next step or
/// terminal fold.
pub fn source_image_emits_nifs_payloads(plan: &RecursiveStepImagePlan) -> bool {
    !matches!(plan.accumulator.as_ref(), Some(acc) if acc.unified)
}

/// Caller-supplied parameters that the recursive plan can't synthesize.
#[derive(Clone, Debug)]
pub struct RecursiveStepImagePlan {
    pub limbs: usize,
    /// Optional app-private variable widths. Empty preserves legacy
    /// app-private semantics. R1CS-F' may set this to one entry per
    /// R1CS variable so explicit Boolean variables occupy one bit and
    /// field variables occupy 64 bits.
    pub app_private_var_widths: Vec<usize>,
    pub boundary_bits: usize,
    pub kmul_count: usize,
    pub ring_action_pair_count: usize,
    /// Projection-checked ring action (Road A): per-identity pair
    /// consumption; forwarded to the image config verbatim.
    pub projection_batches: Vec<usize>,
    pub ring_action_pair_layout: RingActionTraceLayout,
    pub sponge_transcript_permutes: usize,
    /// NIFS-payload shapes (in fill order). Legacy non-unified plans
    /// materialise these in the source image. Canonical unified plans
    /// keep the metadata for prior-fold shape validation but elide the
    /// source-image payload columns via [`source_image_emits_nifs_payloads`].
    ///
    /// If `accumulator` is `Some`, the payload at its
    /// `ce_claim_payload_index` must be a CcsClaim or CeClaim whose
    /// `c_data_entries` matches the accumulator's.
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
    /// Unified-mode flag. When `true`, the image carries `new_acc_digest`
    /// as an explicit state-out value and relies on the next recursive step
    /// or terminal fold to recompute/bind it against the actual NIFS.V
    /// running accumulator. When `false`, the legacy single-accumulator path
    /// still emits the producer-side `H(parent.c_data...)` trace.
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
    /// Indices of app-assignment variables to absorb into the outgoing
    /// semantic-state digest as canonical-u64 lanes. Each index `j`
    /// binds the 64 committed bits at
    /// `layout.app_private.offset + j * 64` into a Construction-2
    /// state coordinate that is then absorbed by `state_x_out`.
    ///
    /// Used by app frontends with a "verifier-supplied public input"
    /// semantics (R1CS). Defaults to empty — no extra binding.
    pub app_public_input_var_indices: Vec<usize>,
    /// Public-input variables known to be Boolean and packed
    /// little-endian into 64-bit outgoing semantic-state preimage lanes.
    ///
    /// This is an opt-in width cut for Boolean-heavy R1CS frontends:
    /// the app R1CS must constrain each listed variable to `{0,1}`.
    /// The structure binds each packed lane to the actual committed
    /// app-assignment lanes; the compiler additionally rejects honest
    /// assignments where a packed variable is not `0` or `1`.
    pub app_public_input_bit_var_indices: Vec<usize>,
    /// App-assignment variables whose Poseidon2 digest must equal the
    /// incoming carried semantic-state digest.
    pub semantic_state_in_var_indices: Vec<usize>,
    /// App-assignment variables whose Poseidon2 digest becomes the
    /// outgoing carried semantic-state digest.
    pub semantic_state_out_var_indices: Vec<usize>,
    /// **Verifier-owned initial semantic-state anchor**. When `Some`,
    /// the F' image's CCS structure emits a base-gated constraint
    ///   `is_base * (state_in.semantic_state_digest_in_lane[k] - anchor[k]) == 0`
    /// for each of the 4 digest lanes. Without this constraint, a
    /// hand-crafted prover could submit a base step whose
    /// `state_in.semantic_state_digest_in_lane` (and thus
    /// `H(state_in_app_vars)`) disagrees with the verifier's anchor —
    /// the protocol would have no proven binding from the claimed
    /// initial app state to the actual first-step witness.
    ///
    /// `None` (the default) means stateless seed semantics — no
    /// anchor constraint is emitted.
    pub initial_semantic_state_digest_anchor: Option<[u8; 32]>,
}

/// Assemble the recursive-step `FPrimeImageConfig` from the plan.
///
/// Produces enforcements for optional semantic-state hashes and
/// `state_x_out`, plus matching digest bindings. Unified accumulator
/// plans deliberately do not emit producer-side Poseidon traces for
/// either `new_acc_digest` or the local chunk-shape coordinate
/// `new_z_i`; those are carried in state_out and checked when consumed.
pub fn build_recursive_step_image_config(plan: &RecursiveStepImagePlan) -> FPrimeImageConfig {
    let mut preimage_lens = Vec::new();
    let mut enforcements = Vec::new();
    let mut state_out_bindings = Vec::new();
    let mut state_in_bindings: Vec<OneShotDigestToStateInBinding> = Vec::new();

    if let Some(acc) = &plan.accumulator {
        if acc.unified {
            // Unified mode uses delayed accumulator-handle binding: the
            // outgoing handle is absorbed into state_x_out, then recomputed
            // by the next recursive step's `acc_digest_in == digest(running)`
            // check or by the terminal fold. Emitting the producer-side
            // `H(parent.c_data...)` trace here costs millions of columns and
            // is redundant in the composed lifecycle relation.
            //
            // This does NOT make the source image a complete recursive F'
            // relation by itself. In particular, stateful app chains still
            // need proof-side rows that bind the previous latest/public
            // recursive link to `state_in`; otherwise a hand-rolled image can
            // satisfy its local app transition while starting from the wrong
            // private semantic state. The red-team acceptance gate is
            // `r1cs_stateful_redteam_folded_f_prime_rejects_disconnected_semantic_input`.
            let _ = (acc.child_count, acc.ce_claim_payload_index, acc.c_data_entries);
        } else {
            // Legacy single-accumulator path.
            let acc_sources =
                accumulator_preimage_sources(acc.child_count, acc.ce_claim_payload_index, acc.c_data_entries);
            let acc_index = preimage_lens.len();
            preimage_lens.push(acc_sources.len());
            enforcements.push(PoseidonTransitionEnforcement {
                one_shot_index: acc_index,
                preimage_lanes: acc_sources,
            });
            state_out_bindings.push(OneShotDigestToStateOutBinding {
                one_shot_index: acc_index,
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
        let mut next_index = preimage_lens.len();
        if !sxo.semantic_state_in_var_indices.is_empty() {
            let semantic_in_sources = semantic_state_preimage_sources(&sxo.semantic_state_in_var_indices);
            preimage_lens.push(semantic_in_sources.len());
            enforcements.push(PoseidonTransitionEnforcement {
                one_shot_index: next_index,
                preimage_lanes: semantic_in_sources,
            });
            state_in_bindings.push(OneShotDigestToStateInBinding {
                one_shot_index: next_index,
                state_in_target: StateInDigestTarget::SemanticStateDigestIn,
            });
            next_index += 1;
        }
        let app_public_semantic_output = sxo.semantic_state_out_var_indices.is_empty()
            && (!sxo.app_public_input_var_indices.is_empty() || !sxo.app_public_input_bit_var_indices.is_empty());
        if !sxo.semantic_state_out_var_indices.is_empty() || app_public_semantic_output {
            let semantic_out_sources = if app_public_semantic_output {
                semantic_state_preimage_sources_with_app_public(
                    &sxo.app_public_input_var_indices,
                    &sxo.app_public_input_bit_var_indices,
                )
            } else {
                semantic_state_preimage_sources(&sxo.semantic_state_out_var_indices)
            };
            preimage_lens.push(semantic_out_sources.len());
            enforcements.push(PoseidonTransitionEnforcement {
                one_shot_index: next_index,
                preimage_lanes: semantic_out_sources,
            });
            state_out_bindings.push(OneShotDigestToStateOutBinding {
                one_shot_index: next_index,
                state_out_target: StateOutDigestTarget::NewSemanticStateDigest,
            });
            next_index += 1;
        }
        let sxo_index = next_index;
        let sxo_sources = state_x_out_preimage_sources_with_mode(sxo.pc, state_x_out_digest_mode_for_options(sxo));
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
        app_private_var_widths: plan.app_private_var_widths.clone(),
        boundary_bits: plan.boundary_bits,
        nifs_payload_shapes: if source_image_emits_nifs_payloads(plan) {
            plan.nifs_payload_shapes.clone()
        } else {
            Vec::new()
        },
        kmul_count: plan.kmul_count,
        ring_action_pair_count: plan.ring_action_pair_count,
        projection_batches: plan.projection_batches.clone(),
        ring_action_pair_layout: plan.ring_action_pair_layout,
        poseidon_one_shot_preimage_lens: preimage_lens,
        sponge_transcript_permutes: plan.sponge_transcript_permutes,
        one_shot_digest_to_state_out_bindings: state_out_bindings,
        one_shot_digest_to_state_in_bindings: state_in_bindings,
        one_shot_digest_to_public_x_out_bindings: public_x_out_bindings,
        poseidon_transition_enforcements: enforcements,
        unified_accumulator_selector: None,
        initial_semantic_state_digest_anchor: plan
            .state_x_out
            .as_ref()
            .and_then(|sxo| sxo.initial_semantic_state_digest_anchor),
    }
}

// ── Preimage builders (for tests and downstream consumers) ───────────────

/// Compute the boundary-update preimage with the given `prev_z_i` and
/// `chunk_digest`. Mirrors `paper::digest::boundary_update_digest`'s
/// preimage construction so callers can encode a Poseidon trace whose
/// witness will satisfy the source-bound absorb rows.
pub fn build_boundary_update_preimage_fields(prev_z_i: [F; 4], chunk_digest: [F; 4]) -> Vec<F> {
    let mut p = vec![F::from_u64(F_PRIME_BOUNDARY_UPDATE_DOMAIN)];
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

/// Compute the legacy parent-commitment accumulator preimage used only by the
/// non-unified trace-budget path.
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
    mode: StateXOutDigestMode,
    vk_fs_digest: [F; 4],
    pi_ccs_header_bundle: [F; 4],
    new_chunk_count: u64,
    new_step_count: u64,
    z_0: [F; 4],
    new_z_i: [F; 4],
    pc: u64,
    new_semantic_state_digest: [F; 4],
    new_acc_digest: [F; 4],
    new_public_trace: [F; 4],
) -> Vec<F> {
    build_state_x_out_preimage_fields_with_app_x(
        mode,
        vk_fs_digest,
        pi_ccs_header_bundle,
        new_chunk_count,
        new_step_count,
        z_0,
        new_z_i,
        pc,
        new_semantic_state_digest,
        new_acc_digest,
        new_public_trace,
        &[],
    )
}

/// Variant kept for call-site compatibility with older frontend code.
///
/// App public bindings are not appended directly to `state_x_out`.
/// R1CS-F' binds them through `state_out.semantic_state_digest`, which
/// is already an authoritative Construction-2 state coordinate absorbed
/// by `state_x_out`.
#[allow(clippy::too_many_arguments)]
pub fn build_state_x_out_preimage_fields_with_app_x(
    mode: StateXOutDigestMode,
    vk_fs_digest: [F; 4],
    pi_ccs_header_bundle: [F; 4],
    new_chunk_count: u64,
    new_step_count: u64,
    _z_0: [F; 4],
    new_z_i: [F; 4],
    pc: u64,
    new_semantic_state_digest: [F; 4],
    new_acc_digest: [F; 4],
    _new_public_trace: [F; 4],
    _app_public_input: &[F],
) -> Vec<F> {
    let mut p = vec![F::from_u64(F_PRIME_STATE_X_OUT_DOMAIN)];
    p.extend(vk_fs_digest);
    p.extend(pi_ccs_header_bundle);
    p.push(F::from_u64(new_chunk_count & 0xffff_ffff));
    p.push(F::from_u64(new_chunk_count >> 32));
    p.push(F::from_u64(new_step_count & 0xffff_ffff));
    p.push(F::from_u64(new_step_count >> 32));
    p.push(F::from_u64(pc & 0xffff_ffff));
    p.push(F::from_u64(pc >> 32));
    p.extend(new_z_i);
    if matches!(mode, StateXOutDigestMode::Stateful) {
        p.extend(new_semantic_state_digest);
    }
    p.extend(new_acc_digest);
    p
}
