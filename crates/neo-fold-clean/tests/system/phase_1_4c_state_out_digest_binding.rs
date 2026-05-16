//! Phase 1.4c-c — poseidon one-shot digest ↔ state_out digest binding.
//!
//! First cross-region functional constraint: each configured binding
//! asserts that a poseidon one-shot Poseidon trace's 4 decoded digest output
//! lanes equal the corresponding state_out digest lanes. Per binding:
//!
//! ```text
//!     z[trace_digest_lane_m] · 1 = z[state_out_digest_lane_m]   for m in 0..4
//! ```
//!
//! Out of scope:
//! - Internal Poseidon transition constraints (the trace's degree-7
//!   polynomial relation is owned by `engine::ccs_native::poseidon2`).
//! - Bindings to boundary (state_x_out) or other regions.

use neo_fold_clean::engine::ccs_native::poseidon2::{POSEIDON2_DIGEST_LEN, POSEIDON2_GOLDILOCKS_BITS, POSEIDON2_WIDTH};
use neo_fold_clean::frontends::fibonacci_f_prime::image::{
    FibonacciFPrimeImage, FibonacciFPrimeImageConfig, FibonacciFPrimeImageLayout, OneShotDigestToStateOutBinding,
    StateOut, StateOutDigestTarget,
};
use neo_fold_clean::frontends::fibonacci_f_prime::structure::build_fibonacci_f_prime_structure;
use neo_fold_clean::paper::f_prime::poseidon_trace::{encode_poseidon_trace, PoseidonTraceImage};
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn binding_config(bindings: Vec<OneShotDigestToStateOutBinding>) -> FibonacciFPrimeImageConfig {
    FibonacciFPrimeImageConfig {
        limbs: 3,
        boundary_bits: 0,
        nifs_payload_shapes: vec![],
        kmul_count: 0,
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        poseidon_one_shot_preimage_lens: vec![3],
        sponge_transcript_permutes: 0,
        one_shot_digest_to_state_out_bindings: bindings,
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
    }
}

fn binding_config_with_ring_action(bindings: Vec<OneShotDigestToStateOutBinding>) -> FibonacciFPrimeImageConfig {
    FibonacciFPrimeImageConfig {
        ring_action_pair_count: 1,
        ..binding_config(bindings)
    }
}

fn build_trace() -> PoseidonTraceImage {
    let preimage = vec![F::from_u64(0x1111), F::from_u64(0x2222), F::from_u64(0x3333)];
    encode_poseidon_trace(&preimage)
}

/// Build a layout + image with one Poseidon trace and state_out.new_z_i set
/// to the trace's native digest. The chosen target is configurable.
fn honest_image_with_binding(
    target: StateOutDigestTarget,
) -> (
    FibonacciFPrimeImageLayout,
    FibonacciFPrimeImage,
    [F; POSEIDON2_DIGEST_LEN],
) {
    let trace = build_trace();
    let digest = trace.digest_native;
    let layout = FibonacciFPrimeImageLayout::new(binding_config(vec![OneShotDigestToStateOutBinding {
        one_shot_index: 0,
        state_out_target: target,
    }]));
    let mut image = FibonacciFPrimeImage::new(layout.clone());
    image.splice_one_shot_poseidon(0, &trace);

    // Set the chosen state_out digest target to match the trace's digest.
    let state = match target {
        StateOutDigestTarget::NewZI => StateOut {
            new_chunk_count: 0,
            new_step_count: 0,
            new_z_i: digest,
            new_public_trace: [F::ZERO; 4],
            new_acc_digest: [F::ZERO; 4],
        },
        StateOutDigestTarget::NewPublicTrace => StateOut {
            new_chunk_count: 0,
            new_step_count: 0,
            new_z_i: [F::ZERO; 4],
            new_public_trace: digest,
            new_acc_digest: [F::ZERO; 4],
        },
        StateOutDigestTarget::NewAccDigest => StateOut {
            new_chunk_count: 0,
            new_step_count: 0,
            new_z_i: [F::ZERO; 4],
            new_public_trace: [F::ZERO; 4],
            new_acc_digest: digest,
        },
    };
    image.fill_state_out(&state);

    (layout, image, digest)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn phase_1_4c_poseidon_state_out_binding_row_shape() {
    let layout =
        FibonacciFPrimeImageLayout::new(binding_config_with_ring_action(vec![OneShotDigestToStateOutBinding {
            one_shot_index: 0,
            state_out_target: StateOutDigestTarget::NewZI,
        }]));
    let structure = build_fibonacci_f_prime_structure(layout);

    assert_eq!(structure.state_out_digest_binding_row_count(), POSEIDON2_DIGEST_LEN);
    assert_eq!(
        structure.state_out_digest_binding_row_start(),
        structure.ring_action_output_row_start() + structure.ring_action_output_row_count(),
        "binding rows come immediately after the ring_action output rows"
    );
    assert_eq!(
        structure.state_out_digest_binding_row(0, 0),
        structure.state_out_digest_binding_row_start()
    );
    assert_eq!(
        structure.state_out_digest_binding_row(0, POSEIDON2_DIGEST_LEN - 1),
        structure.state_out_digest_binding_row_start() + POSEIDON2_DIGEST_LEN - 1
    );
    assert_eq!(
        structure.ccs.n,
        structure.state_out_digest_binding_row_start() + POSEIDON2_DIGEST_LEN,
        "total n must include binding rows"
    );
}

#[test]
fn phase_1_4c_honest_poseidon_state_out_binding_satisfies() {
    for target in [
        StateOutDigestTarget::NewZI,
        StateOutDigestTarget::NewPublicTrace,
        StateOutDigestTarget::NewAccDigest,
    ] {
        let (layout, image, _) = honest_image_with_binding(target);
        let structure = build_fibonacci_f_prime_structure(layout);
        let z = structure.extend_witness_from_image(&image);
        assert!(
            structure.is_satisfied(&z),
            "honest binding ({:?}) must satisfy structure (first failing row: {:?})",
            target,
            structure.first_unsatisfied_row(&z),
        );
    }
}

#[test]
fn phase_1_4c_tampered_state_out_digest_lane_trips_binding_row() {
    let (layout, image, _) = honest_image_with_binding(StateOutDigestTarget::NewZI);
    let structure = build_fibonacci_f_prime_structure(layout);
    let mut z = structure.extend_witness_from_image(&image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    // Tamper the state_out new_z_i lane 0 by rewriting its 64 bits to encode a
    // different canonical-u64 value. Bit validity still holds; only the
    // poseidon↔state_out binding row at lane 0 sees the mismatch.
    let state_out_z_i_lane0 = structure.lane_slots.state_lanes[26];
    let new_value = decode_lane_f(&z, state_out_z_i_lane0.bit_start) + F::ONE;
    write_u64_bits(&mut z, state_out_z_i_lane0.bit_start, new_value.as_canonical_u64());

    assert_eq!(
        structure.first_unsatisfied_row(&z),
        Some(structure.state_out_digest_binding_row(0, 0)),
        "tampering state_out z_i[0] with coherent bits must trip only the poseidon↔state_out binding row at lane 0"
    );
}

#[test]
fn phase_1_4c_tampered_trace_digest_bit_trips_binding_row() {
    let (layout, image, _) = honest_image_with_binding(StateOutDigestTarget::NewAccDigest);
    let structure = build_fibonacci_f_prime_structure(layout);
    let mut z = structure.extend_witness_from_image(&image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    // Flip a zero-bit in trace lane 0 of digest position (the trace's
    // first digest-output lane). Bit validity still holds; the binding
    // row trips because the trace's lane 0 now decodes differently from
    // the state_out digest target.
    let trace_slots = &structure.lane_slots.poseidon_trace_lanes[0];
    let digest_lane_base = trace_slots.len() - POSEIDON2_WIDTH;
    let target_lane = trace_slots[digest_lane_base];

    let mut flipped = None;
    for bit in 0..POSEIDON2_GOLDILOCKS_BITS {
        let pos = target_lane.bit_start + bit;
        if z[pos] == F::ZERO {
            z[pos] = F::ONE;
            flipped = Some(bit);
            break;
        }
    }
    let _flipped = flipped.expect("digest lane must have at least one zero bit");

    assert_eq!(
        structure.first_unsatisfied_row(&z),
        Some(structure.state_out_digest_binding_row(0, 0)),
        "flipping a trace digest lane 0 bit keeps bit validity satisfied; poseidon↔state_out binding fails"
    );
}

fn write_u64_bits(z: &mut [F], bit_start: usize, value: u64) {
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        z[bit_start + i] = if ((value >> i) & 1) == 1 { F::ONE } else { F::ZERO };
    }
}

/// Recompose the 64-bit lane starting at `bit_start` from `z` into a
/// field element.
fn decode_lane_f(z: &[F], bit_start: usize) -> F {
    let mut acc = F::ZERO;
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        if z[bit_start + i] == F::ONE {
            acc += F::from_u64(1u64 << i);
        }
    }
    acc
}

// Bring `as_canonical_u64` into scope for the tamper test.
use p3_field::PrimeField64;
