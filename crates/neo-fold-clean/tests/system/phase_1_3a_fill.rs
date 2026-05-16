//! Phase 1.3a — fill boundary–app_private only.
//!
//! Promotes the Phase 1.2 skeleton from "layout + splices" into the
//! start of a real encoder by filling boundary (boundary), state_in (state-in
//! lanes), state_out (state-out lanes), chunk_digest (chunk_digest), and app_private (Fibonacci
//! app-private carries) from typed inputs. Tests check that:
//!
//! - Each fill writes only into its own region and leaves all other
//!   bits as zero (skeleton invariant preserved).
//! - After all five fills, the b=2 low-norm bit invariant holds.
//! - Each fill round-trips via the matching decode helper.
//!
//! Out of scope:
//! - nifs_payloads NIFS payload fill (Phase 1.3b).
//! - kmul K-mul fill (Phase 1.3c).
//! - ring_action/poseidon fills (already covered by Phase 1.2 splice tests).
//! - Lifecycle migration, Spartan, generic AppStep.
//! - Any change that turns an `ivc_invariants` test green.

use neo_fold_clean::frontends::fibonacci_f_prime::image::{
    FibonacciFPrimeImage, FibonacciFPrimeImageConfig, FibonacciFPrimeImageLayout, StateIn, StateOut,
};
use neo_fold_clean::paper::f_prime::poseidon_trace::assert_committed_coords_are_bits;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

// ── Fixtures ─────────────────────────────────────────────────────────────

fn skeleton_config() -> FibonacciFPrimeImageConfig {
    FibonacciFPrimeImageConfig {
        limbs: 3,
        boundary_bits: 704,
        nifs_payload_shapes: vec![],
        kmul_count: 8,
        ring_action_pair_count: 4,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::SignedDigit { bits: 5 },
            LowNormEncoding::SignedDigit { bits: 8 },
            LowNormEncoding::SignedDigit { bits: 12 },
            LowNormEncoding::SignedDigit { bits: 20 },
        ),
        poseidon_one_shot_preimage_lens: vec![13, 13, 40, 40, 1235, 1300, 977, 977],
        sponge_transcript_permutes: 64,
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
    }
}

fn fresh_image() -> FibonacciFPrimeImage {
    FibonacciFPrimeImage::new(FibonacciFPrimeImageLayout::new(skeleton_config()))
}

fn mk_digest(seed: u64) -> [F; 4] {
    std::array::from_fn(|i| F::from_u64(seed.wrapping_add(101 * i as u64 + 7)))
}

fn deterministic_state_in() -> StateIn {
    StateIn {
        vk_fs_digest: mk_digest(1_000),
        structure_digest: mk_digest(2_000),
        z_0: mk_digest(3_000),
        z_i_in: mk_digest(4_000),
        acc_digest_in: mk_digest(5_000),
        public_trace_in: mk_digest(6_000),
    }
}

fn deterministic_state_out() -> StateOut {
    StateOut {
        new_chunk_count: 7,
        new_step_count: 42,
        new_z_i: mk_digest(7_000),
        new_public_trace: mk_digest(8_000),
        new_acc_digest: mk_digest(9_000),
    }
}

fn deterministic_chunk_digest() -> [F; 4] {
    mk_digest(10_000)
}

fn deterministic_boundary_bits() -> Vec<F> {
    // 704 deterministic {0,1} bits. Use the low bits of (3·i + 1).
    (0..704)
        .map(|i| if (3 * i + 1) & 1 == 1 { F::ONE } else { F::ZERO })
        .collect()
}

fn deterministic_app_private_carries(limbs: usize) -> Vec<F> {
    // LIMBS - 1 bits.
    (0..limbs.saturating_sub(1))
        .map(|i| if i % 2 == 0 { F::ONE } else { F::ZERO })
        .collect()
}

// ── Bit invariant per fill (regions stay disjoint) ───────────────────────

#[test]
fn phase_1_3a_each_fill_writes_only_its_region() {
    // Filling state_in with a non-zero state must leave boundary, state_out..poseidon zero.
    let mut image = fresh_image();
    image.fill_state_in(&deterministic_state_in());

    // boundary should be all zero (we didn't fill it).
    for v in &image.values[image.layout.boundary.offset..image.layout.boundary.end()] {
        assert_eq!(*v, F::ZERO, "state_in fill must not perturb boundary");
    }
    // state_out, chunk_digest, app_private, nifs_payloads, kmul, ring_action, poseidon should also be all zero.
    let post_state_in_offsets = [
        image.layout.state_out,
        image.layout.chunk_digest,
        image.layout.app_private,
        image.layout.nifs_payloads,
        image.layout.kmul,
        image.layout.ring_action,
        image.layout.poseidon,
    ];
    for region in post_state_in_offsets {
        for v in &image.values[region.offset..region.end()] {
            assert_eq!(*v, F::ZERO, "state_in fill must not perturb later regions");
        }
    }
}

#[test]
fn phase_1_3a_all_fills_preserve_low_norm_bit_invariant() {
    let mut image = fresh_image();
    image.fill_boundary(&deterministic_boundary_bits());
    image.fill_state_in(&deterministic_state_in());
    image.fill_state_out(&deterministic_state_out());
    image.fill_chunk_digest(deterministic_chunk_digest());
    image.fill_app_private(&deterministic_app_private_carries(image.layout.config.limbs));
    assert_committed_coords_are_bits(&image.values);
}

// ── Round-trip parity per region ─────────────────────────────────────────

#[test]
fn phase_1_3a_boundary_round_trips() {
    let mut image = fresh_image();
    let bits = deterministic_boundary_bits();
    image.fill_boundary(&bits);
    let decoded = image.decode_boundary();
    assert_eq!(decoded, bits, "boundary round-trip");
}

#[test]
fn phase_1_3a_state_in_round_trips() {
    let mut image = fresh_image();
    let state = deterministic_state_in();
    image.fill_state_in(&state);
    let decoded = image.decode_state_in();
    assert_eq!(decoded, state, "state_in state-in round-trip");
}

#[test]
fn phase_1_3a_state_out_round_trips() {
    let mut image = fresh_image();
    let state = deterministic_state_out();
    image.fill_state_out(&state);
    let decoded = image.decode_state_out();
    assert_eq!(decoded, state, "state_out state-out round-trip");
}

#[test]
fn phase_1_3a_chunk_digest_round_trips() {
    let mut image = fresh_image();
    let digest = deterministic_chunk_digest();
    image.fill_chunk_digest(digest);
    let decoded = image.decode_chunk_digest();
    assert_eq!(decoded, digest, "chunk_digest round-trip");
}

#[test]
fn phase_1_3a_app_private_round_trips() {
    let mut image = fresh_image();
    let carries = deterministic_app_private_carries(image.layout.config.limbs);
    image.fill_app_private(&carries);
    let decoded = image.decode_app_private();
    assert_eq!(decoded, carries, "app_private app-private round-trip");
}

// ── Negative / shape tests ───────────────────────────────────────────────

#[test]
#[should_panic(expected = "boundary bit count must match layout")]
fn phase_1_3a_boundary_wrong_length_panics() {
    let mut image = fresh_image();
    image.fill_boundary(&[F::ZERO]); // wrong length, expects 704
}

#[test]
#[should_panic(expected = "app_private app-private bit count must match LIMBS - 1")]
fn phase_1_3a_app_private_wrong_length_panics() {
    let mut image = fresh_image();
    image.fill_app_private(&[F::ZERO; 7]); // wrong length, expects LIMBS - 1 = 2
}

#[test]
#[should_panic(expected = "must be in {0,1}")]
fn phase_1_3a_boundary_non_bit_value_panics() {
    let mut image = fresh_image();
    let mut bits = deterministic_boundary_bits();
    bits[5] = F::from_u64(2); // not a bit
    image.fill_boundary(&bits);
}
