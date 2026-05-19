//! Phase 1.4c-d — poseidon one-shot digest ↔ boundary public-output binding.
//!
//! Mirrors 1.4c-c (poseidon↔state_out binding) but for the public `x_out` lanes
//! committed inside boundary. Each binding identifies a one-shot Poseidon
//! trace and an explicit 4-tuple of bit offsets inside boundary — usually
//! the four 64-bit lanes of `enc_inst(x_out)`. Per binding, the
//! structure emits:
//!
//! - 4 fresh decoded columns, one per boundary lane (after `sponge_transcript_lanes`).
//! - 4 decode rows binding each new column to its 64-bit window.
//! - 4 equality rows linking the trace's digest output lanes to the
//!   newly-decoded boundary lanes.
//!
//! Out of scope:
//! - Internal Poseidon transition constraints.
//! - Other boundary layout assumptions (only the four lane offsets matter).

use neo_fold_clean::engine::ccs_native::poseidon2::{POSEIDON2_DIGEST_LEN, POSEIDON2_GOLDILOCKS_BITS, POSEIDON2_WIDTH};
use neo_fold_clean::frontends::f_prime::image::{
    FPrimeImage, FPrimeImageConfig, FPrimeImageLayout, OneShotDigestToPublicXOutBinding,
};
use neo_fold_clean::frontends::f_prime::structure::build_f_prime_structure;
use neo_fold_clean::paper::f_prime::poseidon_trace::{encode_poseidon_trace, PoseidonTraceImage};
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// boundary sized to hold exactly one digest (256 bits). The four
/// 64-bit digest lanes are at offsets boundary_start, +64, +128, +192.
const BOUNDARY_BITS: usize = 4 * POSEIDON2_GOLDILOCKS_BITS;

fn public_x_out_lane_bit_starts(layout: &FPrimeImageLayout) -> [usize; 4] {
    let boundary_start = layout.boundary.offset;
    std::array::from_fn(|m| boundary_start + m * POSEIDON2_GOLDILOCKS_BITS)
}

fn binding_config(boundary_bindings: Vec<OneShotDigestToPublicXOutBinding>) -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3,
        boundary_bits: BOUNDARY_BITS,
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
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: boundary_bindings,
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
    }
}

fn build_trace() -> PoseidonTraceImage {
    let preimage = vec![F::from_u64(0x5151), F::from_u64(0x6262), F::from_u64(0x7373)];
    encode_poseidon_trace(&preimage)
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

/// Build an honest image: trace spliced into poseidon; trace's digest bits
/// written into the four boundary lanes addressed by the binding.
fn honest_boundary_image() -> (FPrimeImageLayout, FPrimeImage) {
    // First, build the layout with a placeholder so we can read
    // boundary.offset, then construct the real bindings.
    let scratch_layout = FPrimeImageLayout::new(binding_config(vec![]));
    let boundary_lanes = public_x_out_lane_bit_starts(&scratch_layout);
    let layout = FPrimeImageLayout::new(binding_config(vec![OneShotDigestToPublicXOutBinding {
        one_shot_index: 0,
        public_x_out_lane_bit_starts: boundary_lanes,
    }]));
    assert_eq!(layout.end, scratch_layout.end, "binding addition must not move regions");

    let trace = build_trace();
    let mut image = FPrimeImage::new(layout.clone());
    image.splice_one_shot_poseidon(0, &trace);

    // Write trace.digest_native into the four boundary lanes.
    for (m, lane_start) in boundary_lanes.iter().enumerate() {
        write_u64_bits(
            &mut image.values,
            *lane_start,
            trace.digest_native[m].as_canonical_u64(),
        );
    }

    (layout, image)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn phase_1_4c_poseidon_boundary_binding_row_shape() {
    let (layout, _) = honest_boundary_image();
    let structure = build_f_prime_structure(layout);

    assert_eq!(structure.public_x_out_binding_row_count(), POSEIDON2_DIGEST_LEN);
    assert_eq!(
        structure.public_x_out_binding_row_start(),
        structure.state_out_digest_binding_row_start() + structure.state_out_digest_binding_row_count(),
        "poseidon↔boundary binding rows come after poseidon↔state_out binding rows"
    );
    assert_eq!(
        structure.public_x_out_binding_row(0, 0),
        structure.public_x_out_binding_row_start()
    );
    assert_eq!(
        structure.public_x_out_binding_row(0, POSEIDON2_DIGEST_LEN - 1),
        structure.public_x_out_binding_row_start() + POSEIDON2_DIGEST_LEN - 1,
        "last poseidon↔boundary binding row is the last row of the structure"
    );
    assert_eq!(
        structure.ccs.n,
        structure.public_x_out_binding_row_start() + POSEIDON2_DIGEST_LEN,
        "total n must include poseidon↔boundary binding rows"
    );
    // boundary decoded lanes are allocated after all other decode lanes.
    assert_eq!(structure.lane_slots.public_x_out_binding_lanes.len(), 1);
    assert_eq!(structure.lane_slots.public_x_out_binding_lanes[0].len(), 4);
}

#[test]
fn phase_1_4c_honest_poseidon_boundary_binding_satisfies() {
    let (layout, image) = honest_boundary_image();
    let structure = build_f_prime_structure(layout);
    let z = structure.extend_witness_from_image(&image);
    assert!(
        structure.is_satisfied(&z),
        "honest poseidon↔boundary binding must satisfy structure (first failing row: {:?})",
        structure.first_unsatisfied_row(&z),
    );
}

#[test]
fn phase_1_4c_tampered_boundary_lane_trips_binding_row() {
    let (layout, image) = honest_boundary_image();
    let structure = build_f_prime_structure(layout);
    let mut z = structure.extend_witness_from_image(&image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    // Tamper boundary lane 1: rewrite its 64 bits to a different
    // canonical-u64 value. Bit validity still holds; only the poseidon↔boundary
    // binding row sees the mismatch.
    let lane = 1usize;
    let boundary_slot = structure.lane_slots.public_x_out_binding_lanes[0][lane];
    let new_value = decode_lane_f(&z, boundary_slot.bit_start) + F::ONE;
    write_u64_bits(&mut z, boundary_slot.bit_start, new_value.as_canonical_u64());

    assert_eq!(
        structure.first_unsatisfied_row(&z),
        Some(structure.public_x_out_binding_row(0, lane)),
        "tampering boundary lane {lane} with coherent bits must trip only that poseidon↔boundary binding row"
    );
}

#[test]
fn phase_1_4c_tampered_trace_digest_bit_trips_boundary_binding_row() {
    let (layout, image) = honest_boundary_image();
    let structure = build_f_prime_structure(layout);
    let mut z = structure.extend_witness_from_image(&image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    // Flip a zero-bit in the trace's first digest output lane. Bit
    // validity still holds; the binding row trips because the trace's
    // lane 0 ≠ boundary lane 0.
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
    let _flipped = flipped.expect("trace digest lane 0 must have at least one zero bit");

    assert_eq!(
        structure.first_unsatisfied_row(&z),
        Some(structure.public_x_out_binding_row(0, 0)),
        "flipping a trace digest lane 0 bit keeps bit validity satisfied; poseidon↔boundary binding fails"
    );
}
