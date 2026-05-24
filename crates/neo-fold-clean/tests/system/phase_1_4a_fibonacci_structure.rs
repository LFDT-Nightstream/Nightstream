//! Phase 1.4a — `enc(F')` CCS structure shape + bit-validity parity.
//!
//! Scope:
//! - **Shape gate**: the structure builder produces a CCS whose width
//!   matches the image layout end plus decoded state_in/state_out/chunk_digest, kmul, and ring_action
//!   lane columns, with one boolean row per committed bit, one decode
//!   row per canonical-u64 lane, and one ring_action product row per `(ρ, c)`
//!   pair cell. Production-target counts (the Phase 1.3d coverage
//!   gate's measurements) are pinned in [`production_kmul_ring_action_shell_image_config`].
//! - **Satisfiability gate**: an honestly-filled Phase 1.3d-style image
//!   satisfies the structure; tampering one bit to a non-`{0,1}` value
//!   trips it.
//!
//! Out of scope:
//! - Remaining Phase 1.4c functional constraints (ring_action output lanes,
//!   counters, hashes, and other decoded values ↔ F' relations).
//! - Phase 1.5 lifecycle migration. `preprocess_seeded` is unchanged;
//!   the failing `per_step_ccs_structure_must_encode_f_prime` invariant
//!   still measures the bit-carrier R1CS, not this structure.

use neo_fold_clean::frontends::f_prime::image::{FPrimeImage, FPrimeImageConfig, FPrimeImageLayout, KMulView};
use neo_fold_clean::frontends::f_prime::structure::{
    build_f_prime_structure, production_kmul_ring_action_shell_image_config, PRODUCTION_KMUL_COUNT,
    PRODUCTION_RING_ACTION_PAIR_COUNT,
};
use neo_fold_clean::paper::f_prime::poseidon_trace::assert_committed_coords_are_bits;
use neo_fold_clean::paper::f_prime::ring_action_trace::{
    encode_ring_action_trace, LowNormEncoding, RingActionTraceLayout,
};
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

// ── Helpers ──────────────────────────────────────────────────────────────

/// Small but representative test config: two kmul K-mul slots, two ring_action
/// ring-action pairs under U64 encoding. End-result image is large
/// enough to exercise both gadget regions but small enough that the
/// CCS structure materializes in ~30 MB of triplets.
fn small_test_image_config() -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3,
        boundary_bits: 0,
        nifs_payload_shapes: vec![],
        kmul_count: 2,
        ring_action_pair_count: 2,
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
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

/// The production-target image config pins the gadget invocation
/// counts the Phase 1.3d coverage gate measured, and pins U64 as the
/// only encoding choice safe for in-circuit wires.
#[test]
fn phase_1_4a_production_config_pins_emitter_counts() {
    let config = production_kmul_ring_action_shell_image_config();

    assert_eq!(
        config.kmul_count, PRODUCTION_KMUL_COUNT,
        "production kmul K-mul count must match the Phase 1.3d coverage measurement"
    );
    assert_eq!(
        config.ring_action_pair_count, PRODUCTION_RING_ACTION_PAIR_COUNT,
        "production ring_action ring-action pair count must match the Phase 1.3d coverage measurement"
    );

    // Every ring_action subregion must use the canonical-u64 encoding. SignedDigit{n}
    // would panic on out-of-range production wires.
    let pair = config.ring_action_pair_layout;
    assert_eq!(pair.rho_enc, LowNormEncoding::U64, "ring_action ρ encoding must be U64");
    assert_eq!(pair.c_enc, LowNormEncoding::U64, "ring_action c encoding must be U64");
    assert_eq!(
        pair.prod_enc,
        LowNormEncoding::U64,
        "ring_action prod encoding must be U64"
    );
    assert_eq!(
        pair.out_enc,
        LowNormEncoding::U64,
        "ring_action out encoding must be U64"
    );

    // The layout-end must dwarf the `per_step_ccs_structure_must_encode_f_prime`
    // floor (50 000): this is the first sign the F' frontend is shaped
    // like a real per-step CCS, not the trivial bit-carrier stand-in.
    let layout = FPrimeImageLayout::new(config);
    assert!(
        layout.end >= 50_000,
        "Phase 1.4a production layout.end = {} must exceed the ivc_invariants floor (50_000)",
        layout.end,
    );
    eprintln!(
        "phase_1_4a production: layout.end = {} bits ({} K-muls, {} ring-mul pairs, all U64)",
        layout.end, PRODUCTION_KMUL_COUNT, PRODUCTION_RING_ACTION_PAIR_COUNT,
    );
}

/// The strict-low-norm structure shape contract (Phase 1.5b-0):
/// `m == layout.end` (every committed coordinate is a single image bit;
/// no decoded lane columns are appended). `n` is `(layout.end − 1)` bit
/// validity rows plus the ring_action product / output rows.
#[test]
fn phase_1_4a_structure_shape_matches_image_layout() {
    let layout = FPrimeImageLayout::new(small_test_image_config());
    let structure = build_f_prime_structure(layout.clone());

    assert_eq!(
        structure.ccs.m, layout.end,
        "structure.m must equal layout.end — no decoded lane columns in the strict low-norm structure"
    );
    assert_eq!(
        structure.ccs.n,
        (layout.end - 1) + structure.ring_action_product_row_count() + structure.ring_action_output_row_count(),
        "structure.n must equal (layout.end − 1) + ring_action product rows + ring_action output rows"
    );
    // Mixed-gate F' CCS: 8 matrices for (bit, prod_l, prod_r, prod_out,
    // sbox_in, sbox_out, lin_l, lin_r); polynomial arity matches.
    assert_eq!(structure.ccs.matrices.len(), 8, "mixed-gate F' CCS has 8 matrices");
    assert_eq!(structure.ccs.f.arity(), 8, "f's arity must match matrix count");

    // Layout metadata round-trips so downstream consumers see the same shape.
    assert_eq!(structure.layout.config.kmul_count, 2);
    assert_eq!(structure.layout.config.ring_action_pair_count, 2);
    assert_eq!(
        structure.layout.config.ring_action_pair_layout.rho_enc,
        LowNormEncoding::U64,
    );
    eprintln!(
        "phase_1_4a small layout: m = {}, n = {} (state_in/state_out/chunk_digest = {}, kmul = {}, ring_action = {})",
        structure.ccs.m,
        structure.ccs.n,
        structure.lane_slots.state_lanes.len(),
        structure.lane_slots.kmul_lanes.len(),
        structure.lane_slots.ring_action_lanes.len(),
    );
}

/// An honestly-filled image (kmul / ring_action slots populated via the
/// production fill paths) satisfies every constraint of the structure
/// once extended with the canonical state_in/state_out/chunk_digest lane decoding.
#[test]
fn phase_1_4a_structure_satisfies_honest_image() {
    let layout = FPrimeImageLayout::new(small_test_image_config());
    let structure = build_f_prime_structure(layout.clone());
    let mut image = FPrimeImage::new(layout);

    // kmul: fill both slots with non-zero K-mul views.
    for i in 0..image.layout.config.kmul_count {
        let view = KMulView {
            p: [F::from_u64(42 + i as u64), F::ZERO],
            q: [F::from_u64(7 * (i as u64 + 1)), F::ZERO],
            r: [F::from_u64(0xdead_beef + i as u64), F::ZERO],
        };
        image.fill_kmul_at(i, &view);
    }

    // ring_action: splice both pairs with non-trivial ρ / c values.
    for i in 0..image.layout.config.ring_action_pair_count {
        let rho_vals: [F; D] = std::array::from_fn(|k| F::from_u64(((i as u64 + 1) * 17 + k as u64) % 31));
        let c_vals: [F; D] = std::array::from_fn(|k| F::from_u64(((i as u64 + 1) * 23 + k as u64) % 53));
        let trace = encode_ring_action_trace(&rho_vals, &c_vals, image.layout.config.ring_action_pair_layout);
        image.splice_ring_action_pair(i, &trace);
    }

    // Image's own bit-validity invariant (the pre-existing check).
    assert_committed_coords_are_bits(&image.values);

    // Extend image bits with canonical state_in/state_out/chunk_digest lane decoding, then
    // the same witness must satisfy every constraint row.
    let z = structure.extend_witness_from_image(&image);
    assert!(
        structure.is_satisfied(&z),
        "Phase 1.4 structure must be satisfied by an honestly-filled, lane-decoded image \
         (first failing row: {:?})",
        structure.first_unsatisfied_row(&z),
    );
}

/// Strict-low-norm invariant: the witness `extend_witness_from_image`
/// returns is exactly the committed image bits. Every coordinate
/// satisfies SuperNeo's `‖z‖_∞ < b` for `b = 2`: `z[0] = 1` and every
/// other entry is in `{0, 1}`. This is the gate Phase 1.5b's manual
/// folding smoke test relies on.
#[test]
fn phase_1_4a_strict_structure_witness_is_low_norm() {
    let layout = FPrimeImageLayout::new(small_test_image_config());
    let structure = build_f_prime_structure(layout.clone());
    let mut image = FPrimeImage::new(layout.clone());

    // Fill non-trivial kmul / ring_action content so we exercise lanes that used to
    // hold high-norm decoded columns.
    for i in 0..image.layout.config.kmul_count {
        let view = KMulView {
            p: [F::from_u64(42 + i as u64), F::ZERO],
            q: [F::from_u64(7 * (i as u64 + 1)), F::ZERO],
            r: [F::from_u64(0xdead_beef + i as u64), F::ZERO],
        };
        image.fill_kmul_at(i, &view);
    }
    for i in 0..image.layout.config.ring_action_pair_count {
        let rho_vals: [F; D] = std::array::from_fn(|k| F::from_u64(((i as u64 + 1) * 17 + k as u64) % 31));
        let c_vals: [F; D] = std::array::from_fn(|k| F::from_u64(((i as u64 + 1) * 23 + k as u64) % 53));
        let trace = encode_ring_action_trace(&rho_vals, &c_vals, image.layout.config.ring_action_pair_layout);
        image.splice_ring_action_pair(i, &trace);
    }

    let z = structure.extend_witness_from_image(&image);
    assert_eq!(z.len(), layout.end, "witness length must equal layout.end");
    assert_eq!(z.len(), structure.ccs.m, "witness length must equal ccs.m");
    assert_eq!(z[0], F::ONE, "z[0] must be the CCS constant slot = 1");
    for (i, &v) in z.iter().enumerate().skip(1) {
        assert!(
            v == F::ZERO || v == F::ONE,
            "z[{i}] = {v:?} violates the strict low-norm invariant; every committed coord must be in {{0, 1}}"
        );
    }
}

/// Tampering one bit to a non-`{0, 1}` value flips exactly one
/// constraint row. Confirms the structure actually constrains bits
/// (i.e., the boolean rows are not vacuously satisfied for any z).
#[test]
fn phase_1_4a_structure_rejects_non_bit_witness() {
    let layout = FPrimeImageLayout::new(small_test_image_config());
    let structure = build_f_prime_structure(layout.clone());
    let image = FPrimeImage::new(layout);

    // Baseline: all-zero bits + zero decoded lanes trivially satisfies.
    let mut z = structure.extend_witness_from_image(&image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    // Pick a bit inside the kmul region and corrupt it to 2.
    let target_col = structure.layout.kmul.offset + 1;
    z[target_col] = F::from_u64(2);

    let failing = structure.first_unsatisfied_row(&z);
    assert_eq!(
        failing,
        Some(target_col - 1),
        "tampered bit at col {target_col} must violate exactly bit-validity row {}",
        target_col - 1,
    );
    assert!(
        !structure.is_satisfied(&z),
        "tampered witness must not satisfy structure"
    );
}
