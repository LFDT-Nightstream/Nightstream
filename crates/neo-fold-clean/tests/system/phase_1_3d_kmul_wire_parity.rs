//! Phase 1.3d-mini-3 — kmul K-mul intermediate wire parity.
//!
//! Couples the bit-backed `KMulView` slot against the K-mul gadget's
//! actual Karatsuba intermediates `(p, q, r)`. Uses the new
//! `enforce_k_mul_with_intermediates` production variant — same
//! constraints as `enforce_k_mul`, plus the three intermediate wires
//! returned for audit/parity.
//!
//! Three-way parity per K-mul case:
//!
//! - **Native**: `p = a.c0·b.c0`, `q = a.c1·b.c1`,
//!   `r = (a.c0+a.c1)·(b.c0+b.c1)` computed directly in F.
//! - **Wire**: witness values at `KMulIntermediates::{p, q, r}` after
//!   `enforce_k_mul_with_intermediates` runs.
//! - **Image**: bits decoded from a kmul slot filled from those F values.
//!
//! Out of scope:
//! - ring_action ring-action product parity (Phase 1.3d-mini-4).
//! - CCS structure / lifecycle / Spartan.
//! - Anything that turns `ivc_invariants` green.

use neo_fold_clean::engine::r1cs_circuit::field_ext::{enforce_k_mul_with_intermediates, KLc, KMulIntermediates, KVar};
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImage, FPrimeImageConfig, FPrimeImageLayout, KMulView};
use neo_fold_clean::paper::f_prime::poseidon_trace::assert_committed_coords_are_bits;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

// ── Helpers ──────────────────────────────────────────────────────────────

fn alloc_k(builder: &mut R1csBuilder, value: K) -> KVar {
    let [c0, c1] = value.as_coeffs();
    KVar::alloc(builder, c0, c1)
}

/// Image config with a single kmul slot — all other regions empty.
fn kmul_only_image_config() -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3,
        app_private_var_widths: Vec::new(),
        boundary_bits: 0,
        nifs_payload_shapes: vec![],
        kmul_count: 1,
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::SignedDigit { bits: 5 },
            LowNormEncoding::SignedDigit { bits: 8 },
            LowNormEncoding::SignedDigit { bits: 12 },
            LowNormEncoding::SignedDigit { bits: 20 },
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

/// Wire-side view: read the three intermediates' witness values into a
/// `KMulView`. High limb of each K-word slot is set to ZERO because
/// the Karatsuba intermediates are single F values (the slot's two-wide
/// shape is reserved for future extensions).
fn intermediates_to_view(intermediates: KMulIntermediates, witness: &[F]) -> KMulView {
    KMulView {
        p: [witness[intermediates.p.col()], F::ZERO],
        q: [witness[intermediates.q.col()], F::ZERO],
        r: [witness[intermediates.r.col()], F::ZERO],
    }
}

/// Native-side view: compute the three intermediates from `(a, b)` directly.
fn native_view(a: K, b: K) -> KMulView {
    let [a0, a1] = a.as_coeffs();
    let [b0, b1] = b.as_coeffs();
    let p = a0 * b0;
    let q = a1 * b1;
    let r = (a0 + a1) * (b0 + b1);
    KMulView {
        p: [p, F::ZERO],
        q: [q, F::ZERO],
        r: [r, F::ZERO],
    }
}

fn run_k_mul_and_collect_views(a: K, b: K) -> (KMulView, KMulView, KVar, R1csBuilder) {
    let native = native_view(a, b);
    let mut builder = R1csBuilder::new();
    let a_var = alloc_k(&mut builder, a);
    let b_var = alloc_k(&mut builder, b);
    let (out, intermediates) =
        enforce_k_mul_with_intermediates(&mut builder, &KLc::from_var(a_var), &KLc::from_var(b_var));
    assert!(
        builder.is_satisfied(),
        "K-mul gadget must be satisfied by the honest witness (first bad row {:?})",
        builder.first_unsatisfied_row()
    );
    let wire = intermediates_to_view(intermediates, builder.witness());
    (native, wire, out, builder)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn phase_1_3d_kmul_three_way_parity_small_k_values() {
    let cases = [
        (
            K::from_coeffs([F::from_u64(3), F::from_u64(5)]),
            K::from_coeffs([F::from_u64(7), F::from_u64(11)]),
        ),
        (K::ONE, K::from_coeffs([F::from_u64(17), F::from_u64(0)])),
        (
            K::from_coeffs([F::from_u64(0xdead_beef), F::from_u64(0xfeed_face)]),
            K::from_coeffs([F::from_u64(0x1234_5678), F::from_u64(0xabcd_ef01)]),
        ),
    ];

    for (idx, (a, b)) in cases.iter().enumerate() {
        let (native, wire, _out, _builder) = run_k_mul_and_collect_views(*a, *b);
        assert_eq!(native, wire, "case {idx}: native ↔ wire");

        let layout = FPrimeImageLayout::new(kmul_only_image_config());
        let mut image = FPrimeImage::new(layout);
        image.fill_kmul_at(0, &native);
        let decoded = image.decode_kmul_at(0);

        assert_committed_coords_are_bits(&image.values);
        assert_eq!(decoded, native, "case {idx}: image ↔ native");
        assert_eq!(decoded, wire, "case {idx}: image ↔ wire");
    }
}

#[test]
fn phase_1_3d_kmul_intermediates_match_product_definitions() {
    // Specific signed-integer-shaped case so we can check the math by hand.
    let a = K::from_coeffs([F::from_u64(2), F::from_u64(3)]);
    let b = K::from_coeffs([F::from_u64(5), F::from_u64(7)]);
    let (native, wire, _out, _builder) = run_k_mul_and_collect_views(a, b);

    // p = 2*5 = 10; q = 3*7 = 21; r = (2+3)*(5+7) = 5*12 = 60.
    assert_eq!(native.p[0], F::from_u64(10), "p = a0·b0");
    assert_eq!(native.q[0], F::from_u64(21), "q = a1·b1");
    assert_eq!(native.r[0], F::from_u64(60), "r = (a0+a1)(b0+b1)");
    assert_eq!(
        wire, native,
        "wire view must match hand-computed Karatsuba intermediates"
    );
}

#[test]
fn phase_1_3d_kmul_negative_k_value_round_trips_through_signed_repr() {
    // Verify the SignedDigit-via-canonical-u64 path inside kmul doesn't
    // get confused by a "large positive" canonical Goldilocks repr of a
    // small negative — the limb is just decomposed canonically.
    let a = K::from_coeffs([-F::ONE, F::from_u64(2)]);
    let b = K::from_coeffs([F::from_u64(4), -F::ONE]);
    let (native, wire, _out, _builder) = run_k_mul_and_collect_views(a, b);
    assert_eq!(native, wire, "wire ↔ native for K elements containing -F::ONE");

    let layout = FPrimeImageLayout::new(kmul_only_image_config());
    let mut image = FPrimeImage::new(layout);
    image.fill_kmul_at(0, &native);
    assert_committed_coords_are_bits(&image.values);
    let decoded = image.decode_kmul_at(0);
    assert_eq!(decoded, native, "image ↔ native after canonical-u64 round-trip");
}

#[test]
fn phase_1_3d_kmul_output_lanes_consistent_with_intermediates() {
    // The K-mul output limbs are functions of (p, q, r). Verify the
    // gadget's reported output matches the spec's K-product formula
    // when the intermediates come from the gadget's own wire values.
    let a = K::from_coeffs([F::from_u64(13), F::from_u64(17)]);
    let b = K::from_coeffs([F::from_u64(19), F::from_u64(23)]);
    let (_native, wire, out, builder) = run_k_mul_and_collect_views(a, b);

    let w = <neo_math::Fq as p3_field::extension::BinomiallyExtendable<2>>::W;
    let out_c0_val = builder.witness()[out.c0.col()];
    let out_c1_val = builder.witness()[out.c1.col()];

    // Gadget invariants: out_c0 = p + W·q; out_c1 = r − p − q.
    assert_eq!(out_c0_val, wire.p[0] + w * wire.q[0], "out_c0 = p + W·q (wire values)");
    assert_eq!(
        out_c1_val,
        wire.r[0] - wire.p[0] - wire.q[0],
        "out_c1 = r − p − q (wire values)"
    );

    // Cross-check: out as K equals native a·b.
    let out_k = K::from_coeffs([out_c0_val, out_c1_val]);
    assert_eq!(out_k, a * b, "K-mul output ↔ native a·b");
}
