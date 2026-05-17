//! Phase 1.3c — fill kmul K-mul intermediates.
//!
//! Adds typed `KMulView` round-trip coverage to the Phase 1.2 image.
//! Each K-mul slot stores three K-word pairs `(p, q, r)` (the
//! Karatsuba intermediates) totaling `KMUL_SLOT_BITS = 384` bits.
//! Tests check:
//!
//! - Round-trip parity for one slot at any index, and for the full kmul
//!   batch fill.
//! - Low-norm bit invariant after kmul is filled.
//! - Disjointness: kmul fill leaves boundary..nifs_payloads and ring_action..poseidon zero.
//! - Wrong-shape inputs panic (out-of-range index, wrong batch count).
//! - Karatsuba-shaped fixture: `p = a0·b0`, `q = a1·b1`,
//!   `r = (a0+a1)·(b0+b1)` round-trips bit-for-bit.
//!
//! Out of scope (council instructions):
//! - Per-step parity vs the F' R1CS emitter (Phase 1.3d).
//! - CCS structure / lifecycle / Spartan / generic AppStep.
//! - Touching nifs_payloads, ring_action, poseidon, or `ivc_invariants` behaviour.

use neo_fold_clean::frontends::f_prime::image::{FPrimeImage, FPrimeImageConfig, FPrimeImageLayout, KMulView};
use neo_fold_clean::paper::f_prime::poseidon_trace::assert_committed_coords_are_bits;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

// ── Config + helpers ─────────────────────────────────────────────────────

const KMUL_COUNT: usize = 6;

fn skeleton_config() -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3,
        boundary_bits: 704,
        nifs_payload_shapes: vec![],
        kmul_count: KMUL_COUNT,
        ring_action_pair_count: 2,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::SignedDigit { bits: 5 },
            LowNormEncoding::SignedDigit { bits: 8 },
            LowNormEncoding::SignedDigit { bits: 12 },
            LowNormEncoding::SignedDigit { bits: 20 },
        ),
        poseidon_one_shot_preimage_lens: vec![13, 40],
        sponge_transcript_permutes: 16,
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
    }
}

fn fresh_image() -> FPrimeImage {
    FPrimeImage::new(FPrimeImageLayout::new(skeleton_config()))
}

/// Deterministic kmul views — arbitrary F values per slot. Sized to
/// match `kmul_count`.
fn deterministic_views(count: usize) -> Vec<KMulView> {
    (0..count)
        .map(|i| {
            let seed = (i as u64).wrapping_mul(1_009) + 7;
            KMulView {
                p: [F::from_u64(seed), F::from_u64(seed.wrapping_add(1))],
                q: [F::from_u64(seed.wrapping_add(100)), F::from_u64(seed.wrapping_add(101))],
                r: [F::from_u64(seed.wrapping_add(200)), F::from_u64(seed.wrapping_add(201))],
            }
        })
        .collect()
}

/// Karatsuba-shaped fixture: pick deterministic `a0, a1, b0, b1` in F,
/// compute `p = a0·b0`, `q = a1·b1`, `r = (a0+a1)·(b0+b1)`. Returns
/// the K-mul intermediates with the upper limb of each pair set to
/// zero (the F-valued K-mul case).
fn karatsuba_view(seed: u64) -> KMulView {
    let a0 = F::from_u64(seed.wrapping_mul(3) + 11);
    let a1 = F::from_u64(seed.wrapping_mul(5) + 13);
    let b0 = F::from_u64(seed.wrapping_mul(7) + 17);
    let b1 = F::from_u64(seed.wrapping_mul(11) + 19);
    let p = a0 * b0;
    let q = a1 * b1;
    let r = (a0 + a1) * (b0 + b1);
    KMulView {
        p: [p, F::ZERO],
        q: [q, F::ZERO],
        r: [r, F::ZERO],
    }
}

// ── Round-trip tests ─────────────────────────────────────────────────────

#[test]
fn phase_1_3c_single_kmul_round_trips() {
    let mut image = fresh_image();
    let view = KMulView {
        p: [F::from_u64(42), F::from_u64(43)],
        q: [F::from_u64(100), F::from_u64(101)],
        r: [F::from_u64(7_777), F::from_u64(8_888)],
    };
    // Fill at a non-zero index to exercise the offset arithmetic.
    image.fill_kmul_at(3, &view);
    let decoded = image.decode_kmul_at(3);
    assert_eq!(decoded, view);
    assert_committed_coords_are_bits(&image.values);
}

#[test]
fn phase_1_3c_all_kmuls_round_trip() {
    let mut image = fresh_image();
    let views = deterministic_views(KMUL_COUNT);
    image.fill_all_kmul(&views);
    let decoded = image.decode_kmul_all();
    assert_eq!(decoded, views);
    assert_committed_coords_are_bits(&image.values);
    eprintln!(
        "phase_1_3c kmul: {} K-muls × 384 bits = {} bits total",
        KMUL_COUNT,
        KMUL_COUNT * 384,
    );
}

#[test]
fn phase_1_3c_karatsuba_intermediates_round_trip() {
    let mut image = fresh_image();
    let views: Vec<KMulView> = (0..KMUL_COUNT)
        .map(|i| karatsuba_view(i as u64 + 1))
        .collect();
    image.fill_all_kmul(&views);
    let decoded = image.decode_kmul_all();
    assert_eq!(decoded, views, "Karatsuba-shaped intermediates round-trip bit-for-bit");
    assert_committed_coords_are_bits(&image.values);
    // The high half of each pair should still be zero after round-trip.
    for view in &decoded {
        assert_eq!(view.p[1], F::ZERO);
        assert_eq!(view.q[1], F::ZERO);
        assert_eq!(view.r[1], F::ZERO);
    }
}

// ── Disjointness ─────────────────────────────────────────────────────────

#[test]
fn phase_1_3c_kmul_fill_leaves_other_regions_zero() {
    let mut image = fresh_image();
    image.fill_all_kmul(&deterministic_views(KMUL_COUNT));

    for region in [
        image.layout.boundary,
        image.layout.state_in,
        image.layout.state_out,
        image.layout.chunk_digest,
        image.layout.app_private,
        image.layout.nifs_payloads,
        image.layout.ring_action,
        image.layout.poseidon,
    ] {
        for v in &image.values[region.offset..region.end()] {
            assert_eq!(*v, F::ZERO, "kmul fill must not perturb non-kmul region {region:?}");
        }
    }
}

// ── Wrong-shape panics ───────────────────────────────────────────────────

#[test]
#[should_panic(expected = "kmul K-mul index")]
fn phase_1_3c_index_out_of_range_panics() {
    let mut image = fresh_image();
    image.fill_kmul_at(
        KMUL_COUNT, // one past the end
        &KMulView {
            p: [F::ZERO, F::ZERO],
            q: [F::ZERO, F::ZERO],
            r: [F::ZERO, F::ZERO],
        },
    );
}

#[test]
#[should_panic(expected = "kmul K-mul view count must equal kmul_count")]
fn phase_1_3c_all_wrong_count_panics() {
    let mut image = fresh_image();
    image.fill_all_kmul(&deterministic_views(KMUL_COUNT - 1));
}

#[test]
#[should_panic(expected = "kmul K-mul index")]
fn phase_1_3c_decode_out_of_range_panics() {
    let image = fresh_image();
    let _ = image.decode_kmul_at(KMUL_COUNT);
}
