//! Phase 1.4f — `state_x_out` enforcement.
//!
//! Closes the IVC public-output loop: the recursive step's `state_x_out`
//! Poseidon hash is enforced end-to-end. Its preimage sources read from
//! committed state-in/state-out/chunk-digest lanes (including counters
//! as low/high halves), and its digest output is bound to the public
//! `x_out` lanes carved out of the boundary region.
//!
//! Tests:
//! 1. Honest: with the accumulator trace, `state_x_out` trace, and the
//!    direct `chunk_digest -> new_z_i` mirror, the fixture image
//!    satisfies the structure.
//! 2. Coherent tamper of `new_chunk_count`'s low half (a counter half
//!    that feeds state_x_out's preimage) fails.
//! 3. Coherent tamper of `new_z_i`'s lane (which feeds state_x_out and
//!    is itself mirrored from `chunk_digest`) fails.
//! 4. Coherent tamper of the `chunk_digest` lane fails the mirror.
//! 5. Coherent tamper of `new_acc_digest`'s lane fails.
//! 6. Coherent tamper of the public `x_out` lane in boundary fails the
//!    state_x_out → boundary digest binding.

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::f_prime::image::{
    FPrimeImage, FPrimeImageLayout, NifsCeClaimShape, NifsCeClaimView, NifsPayloadShape, StateIn, StateOut,
};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_accumulator_preimage_fields, build_recursive_step_image_config, build_state_x_out_preimage_fields,
    AccumulatorPlanOptions, RecursiveStepImagePlan, StateXOutPlanOptions, STATE_LANE_CHUNK_DIGEST_BASE,
    STATE_LANE_NEW_ACC_DIGEST_BASE, STATE_LANE_NEW_STEP_COUNT, STATE_LANE_NEW_Z_I_BASE,
};
use neo_fold_clean::frontends::f_prime::structure::build_f_prime_structure;
use neo_fold_clean::paper::digest::StateXOutDigestMode;
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const C_DATA_ENTRIES: usize = 2;
const PC: u64 = 1;
const NEW_CHUNK_COUNT: u64 = 7;
const NEW_STEP_COUNT: u64 = 13;
const PUBLIC_X_OUT_LANE_COUNT: usize = 4;
const BOUNDARY_BITS: usize = PUBLIC_X_OUT_LANE_COUNT * POSEIDON2_GOLDILOCKS_BITS;

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
        app_private_widths_are_range_constraints: false,
        boundary_bits: BOUNDARY_BITS,
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
        state_x_out: None, // Filled in by the fixture builder once we
                           // know boundary lane positions from a layout probe.
    }
}

struct Fixture {
    layout: FPrimeImageLayout,
    image: FPrimeImage,
}

fn build_honest_fixture() -> Fixture {
    // Probe-build the layout once to learn boundary's offset, then rebuild
    // the plan with concrete public-x_out lane bit starts.
    let probe_plan = make_plan();
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let boundary_start = probe_layout.boundary.offset;
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|m| boundary_start + m * POSEIDON2_GOLDILOCKS_BITS);

    let mut plan = make_plan();
    plan.state_x_out = Some(StateXOutPlanOptions {
        pc: PC,
        public_x_out_lane_bit_starts,
        app_public_input_var_indices: Vec::new(),
        app_public_input_bit_var_indices: Vec::new(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
        initial_semantic_state_digest_anchor: None,
    });
    let layout = FPrimeImageLayout::new(build_recursive_step_image_config(&plan));
    let mut image = FPrimeImage::new(layout.clone());

    let vk_fs_digest: [F; 4] = [
        F::from_u64(0x101),
        F::from_u64(0x202),
        F::from_u64(0x303),
        F::from_u64(0x404),
    ];
    let structure_digest: [F; 4] = [
        F::from_u64(0x505),
        F::from_u64(0x606),
        F::from_u64(0x707),
        F::from_u64(0x808),
    ];
    let z_0: [F; 4] = [
        F::from_u64(0x900),
        F::from_u64(0xa00),
        F::from_u64(0xb00),
        F::from_u64(0xc00),
    ];
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
        vk_fs_digest,
        structure_digest,
        z_0,
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

    // Encode the accumulator and state_x_out Poseidon traces. The old
    // boundary_update trace is no longer part of the canonical F' image:
    // `new_z_i` directly mirrors `chunk_digest`.
    let accumulator_preimage = build_accumulator_preimage_fields(1, &c_data);
    let accumulator_trace = encode_poseidon_trace(&accumulator_preimage);
    image.splice_one_shot_poseidon(0, &accumulator_trace);

    let new_z_i = chunk_digest;
    let new_public_trace = new_z_i;
    let new_acc_digest = accumulator_trace.digest_native;

    image.fill_state_out(&StateOut {
        new_chunk_count: NEW_CHUNK_COUNT,
        new_step_count: NEW_STEP_COUNT,
        new_z_i,
        new_public_trace,
        new_semantic_state_digest: new_acc_digest,
        new_acc_digest,
    });

    // Now build the state_x_out preimage using the real post-step
    // values and splice its trace.
    let state_x_out_preimage = build_state_x_out_preimage_fields(
        StateXOutDigestMode::Stateless,
        vk_fs_digest,
        structure_digest,
        NEW_CHUNK_COUNT,
        NEW_STEP_COUNT,
        z_0,
        new_z_i,
        PC,
        new_acc_digest,
        new_acc_digest,
        new_public_trace,
    );
    let state_x_out_trace = encode_poseidon_trace(&state_x_out_preimage);
    image.splice_one_shot_poseidon(1, &state_x_out_trace);

    // Write state_x_out's digest into boundary's public-x_out lanes.
    for (m, &lane_bit_start) in public_x_out_lane_bit_starts.iter().enumerate() {
        let value = state_x_out_trace.digest_native[m].as_canonical_u64();
        for j in 0..POSEIDON2_GOLDILOCKS_BITS {
            image.values[lane_bit_start + j] = if ((value >> j) & 1) == 1 { F::ONE } else { F::ZERO };
        }
    }

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

/// Coherently rewrite a 64-bit lane's bits to encode `new_value`. Bit
/// validity remains satisfied (each bit is still in `{0, 1}`); only
/// constraints that USE the source value see a mismatch.
fn flip_lane_bits_to(z: &mut [F], lane_bit_start: usize, new_value: F) {
    let v = new_value.as_canonical_u64();
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        z[lane_bit_start + i] = if ((v >> i) & 1) == 1 { F::ONE } else { F::ZERO };
    }
}

#[test]
fn phase_1_4f_honest_recursive_step_with_state_x_out_satisfies() {
    let fix = build_honest_fixture();
    let structure = build_f_prime_structure(fix.layout);
    let z = structure.extend_witness_from_image(&fix.image);
    assert!(
        structure.is_satisfied(&z),
        "honest recursive step with accumulator/state_x_out enforcements must satisfy structure (first failing row: {:?})",
        structure.first_unsatisfied_row(&z),
    );
}

#[test]
fn phase_1_4f_tampered_new_step_count_trips_state_x_out_absorb() {
    let fix = build_honest_fixture();
    let structure = build_f_prime_structure(fix.layout);
    let mut z = structure.extend_witness_from_image(&fix.image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    // new_step_count is one state_lane. Its low half feeds the
    // state_x_out preimage absorb. Coherent tamper still satisfies
    // bit/decode, but the state_x_out absorb sees a different
    // low-half value.
    let lane = structure.lane_slots.state_lanes[STATE_LANE_NEW_STEP_COUNT];
    let new_value = decode_lane(&z, lane.bit_start) + F::ONE;
    flip_lane_bits_to(&mut z, lane.bit_start, new_value);

    assert!(
        !structure.is_satisfied(&z),
        "coherent new_step_count tamper must trip a state_x_out absorb row (counter-half source mismatch)"
    );
}

#[test]
fn phase_1_4f_tampered_new_z_i_trips_some_state_x_out_row() {
    let fix = build_honest_fixture();
    let structure = build_f_prime_structure(fix.layout);
    let mut z = structure.extend_witness_from_image(&fix.image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    // new_z_i lane 0 is both an absorb source for state_x_out and the
    // target of the direct chunk_digest mirror. Tampering it coherently
    // breaks at least one of those.
    let lane = structure.lane_slots.state_lanes[STATE_LANE_NEW_Z_I_BASE];
    let new_value = decode_lane(&z, lane.bit_start) + F::ONE;
    flip_lane_bits_to(&mut z, lane.bit_start, new_value);

    assert!(
        !structure.is_satisfied(&z),
        "coherent new_z_i tamper must trip a binding involving that lane"
    );
}

#[test]
fn phase_1_4f_tampered_chunk_digest_trips_chunk_boundary_mirror() {
    let fix = build_honest_fixture();
    let structure = build_f_prime_structure(fix.layout);
    let mut z = structure.extend_witness_from_image(&fix.image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    let lane = structure.lane_slots.state_lanes[STATE_LANE_CHUNK_DIGEST_BASE];
    let new_value = decode_lane(&z, lane.bit_start) + F::ONE;
    flip_lane_bits_to(&mut z, lane.bit_start, new_value);

    assert!(
        !structure.is_satisfied(&z),
        "coherent chunk_digest tamper must trip the chunk_digest -> new_z_i mirror"
    );
}

#[test]
fn phase_1_4f_tampered_new_acc_digest_trips_some_state_x_out_row() {
    let fix = build_honest_fixture();
    let structure = build_f_prime_structure(fix.layout);
    let mut z = structure.extend_witness_from_image(&fix.image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    let lane = structure.lane_slots.state_lanes[STATE_LANE_NEW_ACC_DIGEST_BASE];
    let new_value = decode_lane(&z, lane.bit_start) + F::ONE;
    flip_lane_bits_to(&mut z, lane.bit_start, new_value);

    assert!(
        !structure.is_satisfied(&z),
        "coherent new_acc_digest tamper must trip a binding involving that lane"
    );
}

#[test]
fn phase_1_4f_tampered_public_x_out_lane_trips_state_x_out_digest_binding() {
    let fix = build_honest_fixture();
    let structure = build_f_prime_structure(fix.layout);
    let mut z = structure.extend_witness_from_image(&fix.image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    // The public-x_out lane is allocated by the planner via the boundary
    // binding. Tamper one lane coherently.
    let lane = structure.lane_slots.public_x_out_binding_lanes[0][0];
    let new_value = decode_lane(&z, lane.bit_start) + F::ONE;
    flip_lane_bits_to(&mut z, lane.bit_start, new_value);

    assert!(
        !structure.is_satisfied(&z),
        "coherent public-x_out lane tamper must trip the state_x_out → boundary digest binding"
    );
}
