//! Phase 1.2 — Fibonacci F' source-image skeleton composition tests.
//!
//! Validates that the new `frontends::f_prime::image` module
//! composes mini-1 through mini-4 primitives into one coherent layout
//! and image for a Fibonacci F' recursive step. Tests cover:
//!
//! - Layout invariants: boundary..poseidon regions are non-overlapping, contiguous,
//!   and their union covers `[1, end)` (skipping the constant slot).
//! - Bit invariant on an empty image (only `z[0] = ONE`).
//! - Splice + decode round-trips for the regions we have primitives for
//!   (one-shot Poseidon traces from mini-1/3a, ring-action pairs from
//!   mini-4, sponge transcripts from mini-3b).
//!
//! Out of scope: lifecycle migration, Spartan, generic AppStep, anything
//! that turns an `ivc_invariants` test green.

use neo_fold_clean::engine::ccs_native::poseidon2_transcript::SpongeTraceBuilder;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImage, FPrimeImageConfig, FPrimeImageLayout};
use neo_fold_clean::paper::f_prime::poseidon_trace::{assert_committed_coords_are_bits, encode_poseidon_trace};
use neo_fold_clean::paper::f_prime::ring_action_trace::{
    encode_ring_action_trace, LowNormEncoding, RingActionTraceLayout,
};
use neo_math::ring::{Rq, D};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

// ── Helpers ──────────────────────────────────────────────────────────────

fn signed_to_field(x: i64) -> F {
    if x >= 0 {
        F::from_u64(x as u64)
    } else {
        F::ZERO - F::from_u64((-x) as u64)
    }
}

fn signed_digit_ring_layout() -> RingActionTraceLayout {
    RingActionTraceLayout::new(
        LowNormEncoding::SignedDigit { bits: 5 },
        LowNormEncoding::SignedDigit { bits: 8 },
        LowNormEncoding::SignedDigit { bits: 12 },
        LowNormEncoding::SignedDigit { bits: 20 },
    )
}

fn make_rho_values() -> [F; D] {
    let raw: [i64; D] = std::array::from_fn(|i| ((i as i64 * 7 + 3) % 7) - 3);
    raw.map(signed_to_field)
}

fn make_c_values() -> [F; D] {
    let raw: [i64; D] = std::array::from_fn(|i| ((i as i64 * 13 + 1).rem_euclid(101)) - 50);
    raw.map(signed_to_field)
}

/// A small but realistic Fibonacci-F'-shape config. Values are
/// deliberately compact so the test stays fast while covering every
/// region group.
fn skeleton_config() -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3, // LIMBS=3 → 2 carry bits
        app_private_var_widths: Vec::new(),
        boundary_bits: 704, // recursive: enc_inst(x_out) + enc_inst(prior_x_out) + 3 u64 counters
        nifs_payload_shapes: vec![], // placeholder size — Phase 1.3 fills nifs_payloads properly
        kmul_count: 8,      // small handful of K-muls
        ring_action_pair_count: 4, // far below production's 288; enough to test splicing
        ring_action_pair_layout: signed_digit_ring_layout(),
        poseidon_one_shot_preimage_lens: vec![13, 13, 40, 40, 1235, 1300, 977, 977],
        // 8 one-shots covering boundary_update + public_trace_update +
        // state_x_out + prior_x_out + fresh_ccs_digest + parent_authority +
        // acc_handle + acc_output. Sizes are catalog §B.3 ballparks.
        sponge_transcript_permutes: 64, // F' transcript session permute count (toy)
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_state_in_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
        initial_semantic_state_digest_anchor: None,
    }
}

fn skeleton_layout() -> FPrimeImageLayout {
    FPrimeImageLayout::new(skeleton_config())
}

// ── Layout invariants ────────────────────────────────────────────────────

#[test]
fn phase_1_2_top_level_regions_are_contiguous_and_non_overlapping() {
    let layout = skeleton_layout();
    let regions = layout.top_level_regions();

    // First region starts at offset 1 (after the constant slot).
    assert_eq!(regions[0].offset, 1, "boundary must start right after constant slot");

    // Each region's end equals the next region's start (contiguous, no gaps).
    for w in regions.windows(2) {
        assert_eq!(
            w[0].end(),
            w[1].offset,
            "regions must be contiguous; gap between {:?} and {:?}",
            w[0],
            w[1]
        );
    }

    // Final region's end equals layout.end.
    assert_eq!(regions.last().unwrap().end(), layout.end);

    // No region overlaps any other (implied by contiguous + non-decreasing,
    // but assert it as a strict invariant in case the iteration order
    // changes in the future).
    for (i, a) in regions.iter().enumerate() {
        for (j, b) in regions.iter().enumerate() {
            if i == j {
                continue;
            }
            let disjoint = a.end() <= b.offset || b.end() <= a.offset;
            assert!(disjoint, "regions {a:?} and {b:?} overlap");
        }
    }
}

#[test]
fn phase_1_2_sub_region_splices_lie_within_their_parents() {
    let layout = skeleton_layout();

    // ring_action pair splices each fit within ring_action.
    let pair_bits = layout.config.ring_action_pair_layout.end - 1;
    for (i, &splice) in layout.ring_action_pair_splices.iter().enumerate() {
        assert!(splice >= layout.ring_action.offset, "pair {i} splice below ring_action");
        assert!(
            splice + pair_bits <= layout.ring_action.end(),
            "pair {i} splice above ring_action"
        );
    }

    // poseidon one-shot splices each fit within poseidon.
    for (i, (&splice, layout_i)) in layout
        .one_shot_poseidon_splices
        .iter()
        .zip(layout.one_shot_poseidon_layouts.iter())
        .enumerate()
    {
        assert!(splice >= layout.poseidon.offset, "one-shot {i} below poseidon");
        assert!(
            splice + layout_i.trace_len <= layout.poseidon.end(),
            "one-shot {i} above poseidon"
        );
    }

    // Sponge transcript splice fits within poseidon.
    assert!(layout.sponge_transcript_splice >= layout.poseidon.offset);
    assert!(
        layout.sponge_transcript_splice + layout.sponge_transcript_bits <= layout.poseidon.end(),
        "sponge above poseidon"
    );
}

// ── Bit invariants ───────────────────────────────────────────────────────

#[test]
fn phase_1_2_empty_image_satisfies_low_norm_invariant() {
    let image = FPrimeImage::new(skeleton_layout());
    assert_eq!(image.values[0], F::ONE, "constant slot");
    // A fresh image pre-fills the is_base counter zero-test inverse lane
    // so it satisfies the `is_base ↔ new_chunk_count` link rows with
    // `new_chunk_count = 0`: the lane holds the bits of
    // `(0 - 1)^{-1} = -1`. Every other coordinate is ZERO.
    let inv_lane = (image.layout.is_base.offset + 1)..image.layout.is_base.end();
    let expected_inv = (F::ZERO - F::ONE).as_canonical_u64();
    for (bit, i) in inv_lane.clone().enumerate() {
        let expected = F::from_u64((expected_inv >> bit) & 1);
        assert_eq!(image.values[i], expected, "inverse lane bit {bit} mismatch");
    }
    for (i, v) in image.values.iter().enumerate().skip(1) {
        if inv_lane.contains(&i) {
            continue;
        }
        assert_eq!(*v, F::ZERO, "empty image z[{i}] must be ZERO");
    }
    // The shared bit invariant from the poseidon_trace module accepts {0,1}.
    assert_committed_coords_are_bits(&image.values);
}

#[test]
fn phase_1_2_image_after_splicing_one_shot_still_satisfies_bit_invariant() {
    let mut image = FPrimeImage::new(skeleton_layout());

    // Splice a real Poseidon trace into the smallest one-shot slot (the
    // first one, preimage_len = 13 — boundary-update shape).
    let preimage_len = image.layout.config.poseidon_one_shot_preimage_lens[0];
    let preimage: Vec<F> = (0..preimage_len)
        .map(|i| F::from_u64(7 * i as u64 + 3))
        .collect();
    let trace = encode_poseidon_trace(&preimage);
    image.splice_one_shot_poseidon(0, &trace);

    assert_committed_coords_are_bits(&image.values);
}

// ── Round-trip parity tests ──────────────────────────────────────────────

#[test]
fn phase_1_2_one_shot_poseidon_splice_round_trips() {
    let mut image = FPrimeImage::new(skeleton_layout());

    // Splice into one-shot index 2 (preimage_len = 40, state_x_out shape).
    let index = 2;
    let preimage_len = image.layout.config.poseidon_one_shot_preimage_lens[index];
    let preimage: Vec<F> = (0..preimage_len)
        .map(|i| F::from_u64(11 * i as u64 + 17))
        .collect();
    let trace = encode_poseidon_trace(&preimage);
    let expected_digest = trace.digest_native;

    image.splice_one_shot_poseidon(index, &trace);

    let decoded = image.decode_one_shot_poseidon_digest(index);
    assert_eq!(decoded, expected_digest, "spliced digest must round-trip");
    assert_committed_coords_are_bits(&image.values);
}

#[test]
fn phase_1_2_ring_action_pair_splice_round_trips() {
    let mut image = FPrimeImage::new(skeleton_layout());

    let rho = make_rho_values();
    let c = make_c_values();
    let layout = signed_digit_ring_layout();
    let trace = encode_ring_action_trace(&rho, &c, layout);
    let reference = Rq(rho).mul(&Rq(c)).0;

    // Splice into pair index 1 to exercise the non-trivial offset path.
    image.splice_ring_action_pair(1, &trace);

    let decoded = image.decode_ring_action_pair_output(1);
    assert_eq!(decoded, trace.output_native, "decode ↔ trace.output_native");
    assert_eq!(decoded, reference, "decode ↔ Rq::mul reference");
    assert_committed_coords_are_bits(&image.values);
}

#[test]
fn phase_1_2_sponge_transcript_splice_preserves_bit_invariant() {
    let mut image = FPrimeImage::new(skeleton_layout());

    // Build a sponge transcript with exactly the configured permute count.
    // Each `challenge_fields_raw(4)` triggers exactly one permute (one
    // `squeeze_once(4)` internally — padded permute, no follow-on).
    let permutes_target = image.layout.config.sponge_transcript_permutes;
    let mut builder = SpongeTraceBuilder::new(b"phase_1_2_test_app");
    for _ in 0..permutes_target {
        let _ = builder.challenge_fields_raw(4);
    }
    let trace = builder.finish();
    assert_eq!(
        trace.layout.permute_offsets.len(),
        permutes_target,
        "test must produce exactly the configured permute count"
    );

    // The skeleton config sized poseidon's sponge slot to `permutes_target *
    // BITS_PER_PERMUTATION`. The trace's bit count must match.
    assert_eq!(
        trace.values.len() - 1,
        image.layout.sponge_transcript_bits,
        "sponge trace bit count must match layout config"
    );

    image.splice_sponge_transcript(&trace);
    assert_committed_coords_are_bits(&image.values);

    eprintln!(
        "phase_1_2: image.end = {} bits (~{:.2} KiB); ring_action {} pairs, poseidon {} one-shots + {} sponge permutes",
        image.layout.end,
        image.layout.end as f64 / 8.0 / 1024.0,
        image.layout.config.ring_action_pair_count,
        image.layout.config.poseidon_one_shot_preimage_lens.len(),
        image.layout.config.sponge_transcript_permutes,
    );
}
