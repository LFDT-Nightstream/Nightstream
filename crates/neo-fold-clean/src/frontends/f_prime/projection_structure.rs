//! Projection-region shell configs and (Phase B) row emission for the
//! Road A folded F' image — split out of `structure.rs` per the repo's
//! 1,500-line file cap; `structure.rs` re-exports this surface.
//!
//! Owns: the pinned production shell configs (Road A projection shell +
//! the D2 reference shell) and the production batch partition. Phase B
//! adds the projection region's semantic CCS row emission here
//! (`ivc_invariants::projection_shell_semantic_rows_must_be_enforced`
//! is its red gate).

use crate::frontends::f_prime::image::FPrimeImageConfig;
use crate::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};

/// Production batch partition: `P_total = 465` pairs over `J = 72`
/// identities (33 batches of 7 + 39 of 6). **Placeholder partition** —
/// region widths depend only on the totals; the real per-identity
/// consumption comes from the Lemma 5 adoption census (audit item 4)
/// when the encoder fills real folds.
pub fn production_projection_batches() -> Vec<usize> {
    let mut batches = vec![7usize; 33];
    batches.extend(std::iter::repeat_n(6usize, 39));
    debug_assert_eq!(batches.iter().sum::<usize>(), PRODUCTION_RING_ACTION_PAIR_COUNT);
    debug_assert_eq!(batches.len(), PRODUCTION_PROJECTION_IDENTITY_COUNT);
    batches
}

/// kmul K-mul invocations per F' recursive step. Pinned by the Phase 1.3d
/// coverage gate's measurement of the actual `enforce_k_mul_with_intermediates`
/// call count.
pub const PRODUCTION_KMUL_COUNT: usize = 7100;

/// ring_action ring-action pair invocations per F' recursive step. Pinned by the
/// Phase 1.3d coverage gate's measurement.
pub const PRODUCTION_RING_ACTION_PAIR_COUNT: usize = 465;

/// Projection identities for the production shell — the Lemma 5 J
/// census's known-clients figure (`4κ = 72`; the adoption census, audit
/// item 4, refines this; the soundness bound conservatively uses
/// `J ≤ 465` until reviewed).
pub const PRODUCTION_PROJECTION_IDENTITY_COUNT: usize = 72;

/// The production F' shell (Road A): kmuls unchanged, and the ring
/// action carried as **projection regions** (candidate E) instead of
/// D²-materialized pairs — the committed-width flip the
/// `folded_f_prime_shell_must_adopt_projection_budget` gate pins.
/// Semantic rows for the projection region are the tracked next phase
/// (`projection_shell_semantic_rows_must_be_enforced`).
pub fn production_kmul_ring_action_shell_image_config() -> FPrimeImageConfig {
    FPrimeImageConfig {
        projection_batches: production_projection_batches(),
        ring_action_pair_count: 0,
        ..production_kmul_d2_ring_action_shell_image_config()
    }
}

/// The pre-Road-A D² reference shell (kept for wire-parity tests and
/// as the measured baseline the projection numbers are compared to).
pub fn production_kmul_d2_ring_action_shell_image_config() -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3,
        app_private_var_widths: Vec::new(),
        boundary_bits: 0,
        nifs_payload_shapes: vec![],
        kmul_count: PRODUCTION_KMUL_COUNT,
        ring_action_pair_count: PRODUCTION_RING_ACTION_PAIR_COUNT,
        projection_batches: Vec::new(),
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        poseidon_one_shot_preimage_lens: vec![],
        sponge_transcript_permutes: 0,
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_state_in_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
        initial_semantic_state_digest_anchor: None,
    }
}
