//! Phase 1.4e — Recursive-step enforcement plan integration test.
//!
//! Builds a legacy non-unified Fibonacci F' recursive-step fixture with
//! the accumulator Poseidon hash and verifies that:
//!
//! 1. The planner's config + the honest image satisfies the full Phase
//!    1.4 structure.
//! 2. Tampering a ce-claim `c_data` lane (in `nifs_payload_lanes`)
//!    coherently leaves bit/decode
//!    green but trips an accumulator absorb-binding row.
//!
//! No test-local hard-coded preimage `Vec<F>` is used as authority —
//! preimages come from `fibonacci_recursive_plan`'s builders, which
//! mirror `paper::digest::*` exactly.

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::f_prime::image::{
    FPrimeImage, FPrimeImageLayout, NifsCeClaimShape, NifsCeClaimView, NifsPayloadShape, StateIn, StateOut,
};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_accumulator_preimage_fields, build_recursive_step_image_config, AccumulatorPlanOptions,
    RecursiveStepImagePlan,
};
use neo_fold_clean::frontends::f_prime::structure::build_f_prime_structure;
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Number of `c_data` F-values the ce-claim payload carries. Two is
/// enough to exercise the c_data lane wiring without bloating the
/// NIFS-payload region.
const C_DATA_ENTRIES: usize = 2;

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
        app_private_var_widths: Vec::new(),
        boundary_bits: 0,
        kmul_count: 0,
        ring_action_pair_count: 0,
        projection_pair_count: 0,
        projection_identity_count: 0,
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
    layout: FPrimeImageLayout,
    image: FPrimeImage,
}

fn build_honest_fixture() -> Fixture {
    let plan = make_plan();
    let config = build_recursive_step_image_config(&plan);
    let layout = FPrimeImageLayout::new(config);
    let mut image = FPrimeImage::new(layout.clone());

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
        semantic_state_digest_in: [F::ZERO; 4],
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

    let accumulator_preimage = build_accumulator_preimage_fields(1, &c_data);

    let accumulator_trace = encode_poseidon_trace(&accumulator_preimage);

    image.splice_one_shot_poseidon(0, &accumulator_trace);

    image.fill_state_out(&StateOut {
        new_chunk_count: 0,
        new_step_count: 0,
        new_z_i: chunk_digest,
        new_public_trace: chunk_digest,
        new_acc_digest: accumulator_trace.digest_native,
        new_semantic_state_digest: accumulator_trace.digest_native,
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
    let structure = build_f_prime_structure(fix.layout);
    let z = structure.extend_witness_from_image(&fix.image);
    assert!(
        structure.is_satisfied(&z),
        "honest recursive step must satisfy full Phase 1.4 structure (first failing row: {:?})",
        structure.first_unsatisfied_row(&z),
    );
}

#[test]
fn phase_1_4e_tampered_c_data_lane_trips_accumulator_absorb() {
    let fix = build_honest_fixture();
    let structure = build_f_prime_structure(fix.layout);
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
