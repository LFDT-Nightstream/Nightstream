//! Phase 1.4e — Recursive-step enforcement plan integration test.
//!
//! Builds a Fibonacci F' recursive-step fixture with three state-advance
//! Poseidon hashes (boundary_update, public_trace_update, accumulator)
//! and verifies that:
//!
//! 1. The planner's config + the honest image satisfies the full Phase
//!    1.4 structure.
//! 2. Tampering `z_i_in` (coherent bit+decoded) leaves bit/decode rows
//!    green but trips a `boundary_update` absorb-binding row.
//! 3. Tampering `chunk_digest` coherently leaves bit/decode green but
//!    trips at least one absorb-binding row (both boundary_update and
//!    public_trace_update consume `chunk_digest`).
//! 4. Tampering `public_trace_in` coherently leaves bit/decode green
//!    but trips a `public_trace_update` absorb-binding row.
//! 5. Tampering a ce-claim `c_data` lane (in `nifs_payload_lanes`)
//!    coherently leaves bit/decode
//!    green but trips an accumulator absorb-binding row.
//!
//! No test-local hard-coded preimage `Vec<F>` is used as authority —
//! preimages come from `fibonacci_recursive_plan`'s builders, which
//! mirror `paper::digest::*` exactly.

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::fibonacci_f_prime::image::{
    FibonacciFPrimeImage, FibonacciFPrimeImageLayout, NifsCeClaimShape, NifsCeClaimView, NifsPayloadShape, StateIn,
    StateOut,
};
use neo_fold_clean::frontends::fibonacci_f_prime::recursive_plan::{
    build_accumulator_preimage_fields, build_boundary_update_preimage_fields,
    build_public_trace_update_preimage_fields, build_recursive_step_image_config, AccumulatorPlanOptions,
    RecursiveStepImagePlan,
};
use neo_fold_clean::frontends::fibonacci_f_prime::structure::build_fibonacci_f_prime_structure;
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Number of `c_data` F-values the ce-claim payload carries. Two is
/// enough to exercise the c_data lane wiring without bloating the
/// NIFS-payload region.
const C_DATA_ENTRIES: usize = 2;

/// Lane indices in `state_lanes` for the data we tamper. Mirrors the
/// constants in `fibonacci_recursive_plan` (verified by inspection).
const STATE_LANE_Z_I_IN_BASE: usize = 12;
const STATE_LANE_PUBLIC_TRACE_IN_BASE: usize = 20;
const STATE_LANE_CHUNK_DIGEST_BASE: usize = 38;

fn make_plan() -> RecursiveStepImagePlan {
    let ce_shape = NifsCeClaimShape {
        c_data_entries: C_DATA_ENTRIES,
        x_rows: 0,
        x_active_cols: 0,
        r_len: 0,
        y_ring_inner_lens: vec![],
        y_zcol_len: 0,
        s_col_len: 0,
    };
    RecursiveStepImagePlan {
        limbs: 3,
        boundary_bits: 0,
        kmul_count: 0,
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: vec![NifsPayloadShape::CeClaim(ce_shape)],
        accumulator: Some(AccumulatorPlanOptions {
            ce_claim_payload_index: 0,
            c_data_entries: C_DATA_ENTRIES,
            child_count: 1,
            unified: false,
        }),
        state_x_out: None,
    }
}

struct Fixture {
    layout: FibonacciFPrimeImageLayout,
    image: FibonacciFPrimeImage,
}

fn build_honest_fixture() -> Fixture {
    let plan = make_plan();
    let config = build_recursive_step_image_config(&plan);
    let layout = FibonacciFPrimeImageLayout::new(config);
    let mut image = FibonacciFPrimeImage::new(layout.clone());

    let z_i_in: [F; 4] = [
        F::from_u64(0x111),
        F::from_u64(0x222),
        F::from_u64(0x333),
        F::from_u64(0x444),
    ];
    let public_trace_in: [F; 4] = [
        F::from_u64(0xaaa),
        F::from_u64(0xbbb),
        F::from_u64(0xccc),
        F::from_u64(0xddd),
    ];
    let chunk_digest: [F; 4] = [
        F::from_u64(0x10001),
        F::from_u64(0x20002),
        F::from_u64(0x30003),
        F::from_u64(0x40004),
    ];
    let c_data: Vec<F> = vec![F::from_u64(0xabcd), F::from_u64(0xef01)];

    image.fill_state_in(&StateIn {
        vk_fs_digest: [F::ZERO; 4],
        structure_digest: [F::ZERO; 4],
        z_0: [F::ZERO; 4],
        z_i_in,
        acc_digest_in: [F::ZERO; 4],
        public_trace_in,
    });
    image.fill_chunk_digest(chunk_digest);

    let ce_view = NifsCeClaimView {
        d: 0,
        kappa: 0,
        c_data: c_data.clone(),
        x_rows: 0,
        x_cols: 0,
        x_active_cols: 0,
        x_active_flat: vec![],
        r: vec![],
        y_ring: vec![],
        y_zcol: vec![],
        s_col: vec![],
        m_in: 0,
        fold_digest_fields: [F::ZERO; 4],
    };
    image.fill_nifs_ce_claim_at(0, &ce_view);

    let boundary_preimage = build_boundary_update_preimage_fields(z_i_in, chunk_digest);
    let public_trace_preimage = build_public_trace_update_preimage_fields(public_trace_in, chunk_digest);
    let accumulator_preimage = build_accumulator_preimage_fields(1, &c_data);

    let boundary_trace = encode_poseidon_trace(&boundary_preimage);
    let public_trace_trace = encode_poseidon_trace(&public_trace_preimage);
    let accumulator_trace = encode_poseidon_trace(&accumulator_preimage);

    image.splice_one_shot_poseidon(0, &boundary_trace);
    image.splice_one_shot_poseidon(1, &public_trace_trace);
    image.splice_one_shot_poseidon(2, &accumulator_trace);

    image.fill_state_out(&StateOut {
        new_chunk_count: 0,
        new_step_count: 0,
        new_z_i: boundary_trace.digest_native,
        new_public_trace: public_trace_trace.digest_native,
        new_acc_digest: accumulator_trace.digest_native,
    });

    Fixture { layout, image }
}

/// Recompose a 64-bit lane to its canonical-u64 F value from the
/// committed bits.
fn decode_lane(z: &[F], lane_bit_start: usize) -> F {
    let mut acc = F::ZERO;
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        if z[lane_bit_start + i] == F::ONE {
            acc += F::from_u64(1u64 << i);
        }
    }
    acc
}

/// Coherently rewrite a state lane's bits to encode `new_value`. Bit
/// validity remains satisfied; only constraints that USE the source
/// value see a mismatch.
fn flip_lane_bits_to(z: &mut [F], lane_bit_start: usize, new_value: F) {
    let v = new_value.as_canonical_u64();
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        z[lane_bit_start + i] = if ((v >> i) & 1) == 1 { F::ONE } else { F::ZERO };
    }
}

#[test]
fn phase_1_4e_honest_recursive_step_satisfies_structure() {
    let fix = build_honest_fixture();
    let structure = build_fibonacci_f_prime_structure(fix.layout);
    let z = structure.extend_witness_from_image(&fix.image);
    assert!(
        structure.is_satisfied(&z),
        "honest recursive step must satisfy full Phase 1.4 structure (first failing row: {:?})",
        structure.first_unsatisfied_row(&z),
    );
}

#[test]
fn phase_1_4e_tampered_z_i_in_trips_boundary_absorb() {
    let fix = build_honest_fixture();
    let structure = build_fibonacci_f_prime_structure(fix.layout);
    let mut z = structure.extend_witness_from_image(&fix.image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    let lane = structure.lane_slots.state_lanes[STATE_LANE_Z_I_IN_BASE];
    let new_value = decode_lane(&z, lane.bit_start) + F::ONE;
    flip_lane_bits_to(&mut z, lane.bit_start, new_value);

    assert!(
        !structure.is_satisfied(&z),
        "coherent z_i_in tamper must trip a boundary_update absorb-binding row"
    );
}

#[test]
fn phase_1_4e_tampered_chunk_digest_trips_some_binding() {
    let fix = build_honest_fixture();
    let structure = build_fibonacci_f_prime_structure(fix.layout);
    let mut z = structure.extend_witness_from_image(&fix.image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    let lane = structure.lane_slots.state_lanes[STATE_LANE_CHUNK_DIGEST_BASE];
    let new_value = decode_lane(&z, lane.bit_start) + F::ONE;
    flip_lane_bits_to(&mut z, lane.bit_start, new_value);

    assert!(
        !structure.is_satisfied(&z),
        "coherent chunk_digest tamper must trip boundary_update and/or public_trace_update absorb-binding rows"
    );
}

#[test]
fn phase_1_4e_tampered_public_trace_in_trips_public_trace_absorb() {
    let fix = build_honest_fixture();
    let structure = build_fibonacci_f_prime_structure(fix.layout);
    let mut z = structure.extend_witness_from_image(&fix.image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    let lane = structure.lane_slots.state_lanes[STATE_LANE_PUBLIC_TRACE_IN_BASE];
    let new_value = decode_lane(&z, lane.bit_start) + F::ONE;
    flip_lane_bits_to(&mut z, lane.bit_start, new_value);

    assert!(
        !structure.is_satisfied(&z),
        "coherent public_trace_in tamper must trip a public_trace_update absorb-binding row"
    );
}

#[test]
fn phase_1_4e_tampered_c_data_lane_trips_accumulator_absorb() {
    let fix = build_honest_fixture();
    let structure = build_fibonacci_f_prime_structure(fix.layout);
    let mut z = structure.extend_witness_from_image(&fix.image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    // CeClaim payload lane 3 is the first c_data entry (after d, kappa,
    // c_data_len). Tampering it coherently keeps bit/decode rows green
    // but the accumulator absorb binding sees a different value than the
    // trace was encoded for.
    let lane = structure.lane_slots.nifs_payload_lanes[0][3];
    let new_value = decode_lane(&z, lane.bit_start) + F::ONE;
    flip_lane_bits_to(&mut z, lane.bit_start, new_value);

    assert!(
        !structure.is_satisfied(&z),
        "coherent c_data[0] tamper must trip an accumulator absorb-binding row"
    );
}
