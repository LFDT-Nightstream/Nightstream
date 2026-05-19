//! Phase 1.1-mini-4 — ring-action (ring_action) layout + parity surface.
//!
//! Promotes the Phase 0D `ring_action_low_norm_prototype` scaffold's
//! reusable layout primitives into the new production
//! `paper::f_prime::ring_action_trace` module, then locks both supported
//! encodings (`SignedDigit{5/8/12/20}` and `U64`) with parity tests
//! against `neo_math::ring::Rq::mul`.
//!
//! Out of scope (council instructions):
//! - lifecycle migration,
//! - Spartan,
//! - generic `AppStep` trait,
//! - any change that turns an `ivc_invariants` test green,
//! - re-running the gadget-side R1CS satisfaction measurement (that
//!   stays in `tests/perf/ring_action_low_norm_prototype.rs`).

use neo_fold_clean::paper::f_prime::poseidon_trace::assert_committed_coords_are_bits;
use neo_fold_clean::paper::f_prime::ring_action_trace::{
    decode_ring_action_output, encode_ring_action_trace, LowNormEncoding, RingActionTraceLayout,
};
use neo_math::ring::{Rq, D};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

// ── Deterministic test values (same regime as Phase 0D) ──────────────────

fn signed_to_field(x: i64) -> F {
    if x >= 0 {
        F::from_u64(x as u64)
    } else {
        F::ZERO - F::from_u64((-x) as u64)
    }
}

/// ρ values in `[-3, 3]` — small alphabet-sampling regime.
fn make_rho_values() -> [F; D] {
    let raw: [i64; D] = std::array::from_fn(|i| ((i as i64 * 7 + 3) % 7) - 3);
    raw.map(signed_to_field)
}

/// c values in `[-50, 50]` — commitment-data regime under small norm.
fn make_c_values() -> [F; D] {
    let raw: [i64; D] = std::array::from_fn(|i| ((i as i64 * 13 + 1).rem_euclid(101)) - 50);
    raw.map(signed_to_field)
}

fn signed_digit_layout() -> RingActionTraceLayout {
    RingActionTraceLayout::new(
        LowNormEncoding::SignedDigit { bits: 5 },
        LowNormEncoding::SignedDigit { bits: 8 },
        LowNormEncoding::SignedDigit { bits: 12 },
        LowNormEncoding::SignedDigit { bits: 20 },
    )
}

fn u64_layout() -> RingActionTraceLayout {
    RingActionTraceLayout::new(
        LowNormEncoding::U64,
        LowNormEncoding::U64,
        LowNormEncoding::U64,
        LowNormEncoding::U64,
    )
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn phase_1_mini_4_signed_digit_layout_offsets_monotonic() {
    let layout = signed_digit_layout();
    assert_eq!(layout.constant_slot, 0);
    assert_eq!(layout.rho_offset, 1);
    assert_eq!(layout.c_offset, 1 + D * 5);
    assert_eq!(layout.prod_offset, layout.c_offset + D * 8);
    assert_eq!(layout.out_offset, layout.prod_offset + D * D * 12);
    assert_eq!(layout.end, layout.out_offset + D * 20);

    // Per-lane offsets monotonic + contiguous within each subregion.
    for i in 1..D {
        assert_eq!(layout.rho_limb_start(i), layout.rho_limb_start(i - 1) + 5);
        assert_eq!(layout.c_limb_start(i), layout.c_limb_start(i - 1) + 8);
        assert_eq!(layout.out_lane_start(i), layout.out_lane_start(i - 1) + 20);
    }
    for i in 0..D {
        for j in 1..D {
            assert_eq!(layout.prod_limb_start(i, j), layout.prod_limb_start(i, j - 1) + 12);
        }
    }
}

#[test]
fn phase_1_mini_4_u64_layout_offsets_monotonic() {
    let layout = u64_layout();
    assert_eq!(layout.constant_slot, 0);
    assert_eq!(layout.rho_offset, 1);
    assert_eq!(layout.c_offset, 1 + D * 64);
    assert_eq!(layout.prod_offset, layout.c_offset + D * 64);
    assert_eq!(layout.out_offset, layout.prod_offset + D * D * 64);
    assert_eq!(layout.end, layout.out_offset + D * 64);
}

#[test]
fn phase_1_mini_4_signed_digit_committed_coords_are_bits() {
    let rho = make_rho_values();
    let c = make_c_values();
    let image = encode_ring_action_trace(&rho, &c, signed_digit_layout());
    assert_committed_coords_are_bits(&image.values);
}

#[test]
fn phase_1_mini_4_u64_committed_coords_are_bits() {
    let rho = make_rho_values();
    let c = make_c_values();
    let image = encode_ring_action_trace(&rho, &c, u64_layout());
    assert_committed_coords_are_bits(&image.values);
}

#[test]
fn phase_1_mini_4_signed_digit_decoded_output_matches_rq_mul() {
    let rho = make_rho_values();
    let c = make_c_values();
    let image = encode_ring_action_trace(&rho, &c, signed_digit_layout());
    let decoded = decode_ring_action_output(&image);
    let reference = Rq(rho).mul(&Rq(c)).0;

    assert_eq!(decoded, image.output_native, "decode ↔ image.output_native");
    assert_eq!(decoded, reference, "decode ↔ Rq::mul reference (SignedDigit)");
    eprintln!(
        "mini-4 SignedDigit{{5/8/12/20}}: layout.end = {} bits (~{:.2} KiB)",
        image.layout.end,
        image.layout.end as f64 / 8.0 / 1024.0,
    );
}

#[test]
fn phase_1_mini_4_u64_decoded_output_matches_rq_mul() {
    let rho = make_rho_values();
    let c = make_c_values();
    let image = encode_ring_action_trace(&rho, &c, u64_layout());
    let decoded = decode_ring_action_output(&image);
    let reference = Rq(rho).mul(&Rq(c)).0;

    assert_eq!(decoded, image.output_native, "decode ↔ image.output_native");
    assert_eq!(decoded, reference, "decode ↔ Rq::mul reference (U64)");
    eprintln!(
        "mini-4 U64: layout.end = {} bits (~{:.2} KiB)",
        image.layout.end,
        image.layout.end as f64 / 8.0 / 1024.0,
    );
}
