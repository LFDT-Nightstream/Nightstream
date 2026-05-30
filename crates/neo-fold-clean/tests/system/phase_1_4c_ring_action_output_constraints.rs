//! Phase 1.4c-b — ring_action output constraints.
//!
//! Product rows prove each decoded product cell is `ρ[i] · c[j]`.
//! This file pins the next semantic layer: each decoded output lane
//! must equal the cyclotomic reduction of those product cells,
//! `out[m] = Σ Φ_TABLE[i+j][m] · prod[i][j]`.
//!
//! Out of scope:
//! - kmul K-mul functional constraints.
//! - Lifecycle migration; `ivc_invariants` must remain red until the
//!   chain folds `enc(F'_i)`.

use neo_fold_clean::engine::r1cs_circuit::ring_action::phi_reduction_coeff;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImage, FPrimeImageConfig, FPrimeImageLayout};
use neo_fold_clean::frontends::f_prime::structure::build_f_prime_structure;
use neo_fold_clean::paper::f_prime::ring_action_trace::{
    encode_ring_action_trace, LowNormEncoding, RingActionTraceLayout,
};
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn small_ring_action_config() -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3,
        boundary_bits: 0,
        nifs_payload_shapes: vec![],
        kmul_count: 0,
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
        initial_semantic_state_digest_anchor: None,
    }
}

fn honest_rho_c(pair_idx: usize) -> ([F; D], [F; D]) {
    let rho: [F; D] = std::array::from_fn(|k| F::from_u64(((pair_idx as u64 + 1) * 43 + k as u64 * 7) % 1009));
    let c: [F; D] = std::array::from_fn(|k| F::from_u64(((pair_idx as u64 + 1) * 59 + k as u64 * 11) % 1013));
    (rho, c)
}

fn honest_ring_action_image() -> (FPrimeImageLayout, FPrimeImage) {
    let layout = FPrimeImageLayout::new(small_ring_action_config());
    let mut image = FPrimeImage::new(layout.clone());
    for pair_idx in 0..image.layout.config.ring_action_pair_count {
        let (rho, c) = honest_rho_c(pair_idx);
        let trace = encode_ring_action_trace(&rho, &c, image.layout.config.ring_action_pair_layout);
        image.splice_ring_action_pair(pair_idx, &trace);
    }
    (layout, image)
}

fn write_u64_bits(bits: &mut [F], bit_start: usize, value: u64) {
    for i in 0..64 {
        bits[bit_start + i] = if ((value >> i) & 1) == 1 { F::ONE } else { F::ZERO };
    }
}

/// Recompose the 64-bit lane starting at `bit_start` from `z` into a u64.
fn decode_lane_u64(z: &[F], bit_start: usize) -> u64 {
    let mut acc: u64 = 0;
    for i in 0..64 {
        if z[bit_start + i] == F::ONE {
            acc |= 1u64 << i;
        }
    }
    acc
}

/// Recompose the 64-bit lane starting at `bit_start` from `z` into a field
/// element.
fn decode_lane_f(z: &[F], bit_start: usize) -> F {
    let mut acc = F::ZERO;
    for i in 0..64 {
        if z[bit_start + i] == F::ONE {
            acc += F::from_u64(1u64 << i);
        }
    }
    acc
}

fn ring_action_out_slot_index(pair_idx: usize, m: usize) -> usize {
    let lanes_per_pair = 3 * D + D * D;
    pair_idx * lanes_per_pair + 2 * D + D * D + m
}

#[test]
fn phase_1_4c_ring_action_output_row_count_shape() {
    let layout = FPrimeImageLayout::new(small_ring_action_config());
    let structure = build_f_prime_structure(layout.clone());
    let output_rows = layout.config.ring_action_pair_count * D;

    assert_eq!(
        structure.ring_action_output_row_count(),
        output_rows,
        "ring_action output row count must be D per ring-action pair"
    );
    assert_eq!(
        structure.ring_action_output_row_start(),
        structure.ring_action_product_row_start() + structure.ring_action_product_row_count(),
        "output rows must immediately follow product rows"
    );
    assert_eq!(
        structure.ring_action_output_row(0, 0),
        structure.ring_action_output_row_start()
    );
    assert_eq!(
        structure.ring_action_output_row(1, D - 1),
        structure.ccs.n - 1,
        "last output lane of the last pair must occupy the final structure row"
    );
}

#[test]
fn phase_1_4c_honest_ring_action_image_satisfies_output_rows() {
    let (layout, image) = honest_ring_action_image();
    let structure = build_f_prime_structure(layout);

    let z = structure.extend_witness_from_image(&image);
    assert!(
        structure.is_satisfied(&z),
        "honest ring_action trace must satisfy bit, decode, product, and output rows (first failing row: {:?})",
        structure.first_unsatisfied_row(&z),
    );
}

#[test]
fn phase_1_4c_tampered_ring_action_output_lane_with_matching_bits_trips_output_row() {
    let (layout, mut image) = honest_ring_action_image();
    let structure = build_f_prime_structure(layout);

    let baseline = structure.extend_witness_from_image(&image);
    assert!(structure.is_satisfied(&baseline), "baseline must satisfy");

    // Tamper out[9] in pair 0 at the bit level. Bit-validity and the
    // product rows still hold; the output semantic row catches it.
    let pair_idx = 0usize;
    let out_m = 9usize;
    let target_ring_action_idx = ring_action_out_slot_index(pair_idx, out_m);
    let slot = structure.lane_slots.ring_action_lanes[target_ring_action_idx];

    let old_value = decode_lane_u64(&baseline, slot.bit_start);
    let tampered_value = old_value
        .checked_add(1)
        .expect("test fixture output must fit u64 + 1");
    write_u64_bits(&mut image.values, slot.bit_start, tampered_value);

    let z = structure.extend_witness_from_image(&image);
    let expected_row = structure.ring_action_output_row(pair_idx, out_m);
    assert_eq!(
        structure.first_unsatisfied_row(&z),
        Some(expected_row),
        "tampering out[{out_m}] with matching bits must trip only the output row"
    );
}

#[test]
fn phase_1_4c_output_row_matches_phi_reduction_coefficients() {
    let (layout, image) = honest_ring_action_image();
    let structure = build_f_prime_structure(layout);
    let z = structure.extend_witness_from_image(&image);

    let pair_idx = 1usize;
    let out_m = 17usize;
    let lanes_per_pair = 3 * D + D * D;
    let pair_base = pair_idx * lanes_per_pair;
    let out_slot = structure.lane_slots.ring_action_lanes[ring_action_out_slot_index(pair_idx, out_m)];
    let out_value = decode_lane_f(&z, out_slot.bit_start);

    let mut expected = F::ZERO;
    for i in 0..D {
        for j in 0..D {
            let coeff = phi_reduction_coeff(i + j, out_m);
            let prod_slot = structure.lane_slots.ring_action_lanes[pair_base + 2 * D + i * D + j];
            expected += coeff * decode_lane_f(&z, prod_slot.bit_start);
        }
    }
    assert_eq!(
        out_value, expected,
        "decoded output lane must equal the explicit Φ reduction of decoded product cells"
    );
    assert_eq!(
        structure.first_unsatisfied_row(&z),
        None,
        "explicit coefficient check should agree with structure satisfaction"
    );
}
