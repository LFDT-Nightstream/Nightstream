//! Phase 1.4c-a — ring_action product constraints.
//!
//! Phase 1.4b-b proved that ring_action trace bits decode into canonical-u64
//! lane values. This file pins the first semantic layer on top: every
//! decoded product cell must equal its decoded ring-action inputs,
//! `ρ[i] · c[j] = prod[i][j]`.
//!
//! Out of scope:
//! - ring_action output-lane linear reduction from product cells, covered by
//!   the sibling 1.4c-b test.
//! - kmul K-mul functional constraints.
//! - Lifecycle migration; `ivc_invariants` must remain red until the
//!   chain folds `enc(F'_i)`.

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
        app_private_var_widths: Vec::new(),
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
    let rho: [F; D] = std::array::from_fn(|k| F::from_u64(((pair_idx as u64 + 1) * 17 + k as u64 * 3) % 257));
    let c: [F; D] = std::array::from_fn(|k| F::from_u64(((pair_idx as u64 + 1) * 29 + k as u64 * 5) % 263));
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

/// Recompose the 64-bit lane starting at `bit_start` into a u64.
fn decode_lane_u64(z: &[F], bit_start: usize) -> u64 {
    let mut acc: u64 = 0;
    for i in 0..64 {
        if z[bit_start + i] == F::ONE {
            acc |= 1u64 << i;
        }
    }
    acc
}

#[test]
fn phase_1_4c_ring_action_product_row_count_shape() {
    let layout = FPrimeImageLayout::new(small_ring_action_config());
    let structure = build_f_prime_structure(layout.clone());
    let product_rows = layout.config.ring_action_pair_count * D * D;
    let output_rows = layout.config.ring_action_pair_count * D;

    assert_eq!(
        structure.ring_action_product_row_count(),
        product_rows,
        "ring_action product row count must be D² per ring-action pair"
    );
    assert_eq!(
        structure.ccs.n,
        structure.semantic_boolean_row_count()
            + structure.is_base_counter_link_row_count()
            + product_rows
            + output_rows,
        "structure.n must include semantic Boolean rows + is_base↔counter link rows + ring_action product rows + ring_action output rows (no decode rows in the strict low-norm structure)"
    );

    let start = structure.semantic_boolean_row_count() + structure.is_base_counter_link_row_count();
    assert_eq!(structure.ring_action_product_row_start(), start);
    assert_eq!(structure.ring_action_product_row(0, 0, 0), start);
    assert_eq!(
        structure.ring_action_product_row(1, D - 1, D - 1),
        start + product_rows - 1,
        "last product cell of the last pair must occupy the final product row"
    );
}

#[test]
fn phase_1_4c_honest_ring_action_image_satisfies_product_rows() {
    let (layout, image) = honest_ring_action_image();
    let structure = build_f_prime_structure(layout);

    let z = structure.extend_witness_from_image(&image);
    assert!(
        structure.is_satisfied(&z),
        "honest ring_action trace must satisfy bit, decode, and product rows (first failing row: {:?})",
        structure.first_unsatisfied_row(&z),
    );
}

#[test]
fn phase_1_4c_tampered_ring_action_product_cell_with_matching_bits_trips_product_row() {
    let (layout, mut image) = honest_ring_action_image();
    let structure = build_f_prime_structure(layout);

    let baseline = structure.extend_witness_from_image(&image);
    assert!(structure.is_satisfied(&baseline), "baseline must satisfy");

    // Tamper prod[3][7] in pair 0 at the bit level. Bit-validity still
    // holds; only the semantic product row should fail.
    let pair_idx = 0usize;
    let prod_i = 3usize;
    let prod_j = 7usize;
    let lanes_per_pair = 3 * D + D * D;
    let target_ring_action_idx = pair_idx * lanes_per_pair + 2 * D + prod_i * D + prod_j;
    let slot = structure.lane_slots.ring_action_lanes[target_ring_action_idx];

    let old_value = decode_lane_u64(&baseline, slot.bit_start);
    let tampered_value = old_value
        .checked_add(1)
        .expect("test fixture product must fit u64 + 1");
    write_u64_bits(&mut image.values, slot.bit_start, tampered_value);

    let z = structure.extend_witness_from_image(&image);
    let expected_row = structure.ring_action_product_row(pair_idx, prod_i, prod_j);
    assert_eq!(
        structure.first_unsatisfied_row(&z),
        Some(expected_row),
        "tampering prod[{prod_i}][{prod_j}] with matching bits must trip only the product row"
    );
}

#[test]
fn phase_1_4c_tampered_ring_action_input_cell_with_matching_bits_trips_first_affected_product_row() {
    let (layout, mut image) = honest_ring_action_image();
    let structure = build_f_prime_structure(layout);

    let baseline = structure.extend_witness_from_image(&image);
    assert!(structure.is_satisfied(&baseline), "baseline must satisfy");

    // Tamper ρ[5] in pair 1 with matching bits. The first affected
    // product row is j=0 for that pair.
    let pair_idx = 1usize;
    let rho_i = 5usize;
    let lanes_per_pair = 3 * D + D * D;
    let target_ring_action_idx = pair_idx * lanes_per_pair + rho_i;
    let slot = structure.lane_slots.ring_action_lanes[target_ring_action_idx];

    let old_value = decode_lane_u64(&baseline, slot.bit_start);
    let tampered_value = old_value
        .checked_add(1)
        .expect("test fixture rho must fit u64 + 1");
    write_u64_bits(&mut image.values, slot.bit_start, tampered_value);

    let z = structure.extend_witness_from_image(&image);
    let expected_row = structure.ring_action_product_row(pair_idx, rho_i, 0);
    assert_eq!(
        structure.first_unsatisfied_row(&z),
        Some(expected_row),
        "tampering rho[{rho_i}] in pair {pair_idx} must first trip product row j=0"
    );
}
