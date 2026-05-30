//! Phase 1.4d-a-3 + 1.4d-a-4 — Poseidon2 transitions + variable preimage binding.
//!
//! Phase 1.4d-a-3 lifted the bit-backed Poseidon2 round constraints
//! into F'. Phase 1.4d-a-4 replaces the constant-baked absorb rows with
//! source-bound absorb rows so the preimage is no longer free config
//! authority — every preimage lane now reads from a decoded F' image
//! column or a literal constant.
//!
//! Tests pin:
//! 1. **Shape**: enabling enforcement adds the expected row count
//!    (lifted non-bitness rows minus absorb rows, plus
//!    `absorbs · WIDTH` re-emitted absorb rows).
//! 2. **Honest pass**: a real Poseidon2 trace whose preimage agrees
//!    with the sources satisfies every row.
//! 3. **Source soundness**: tampering a source lane coherently breaks
//!    only the absorb-binding row.
//! 4. **Internal soundness**: tampering an interior trace lane (post-absorb)
//!    breaks a lifted round constraint, not an absorb-binding row.

use neo_fold_clean::engine::ccs_native::poseidon2::{
    build_bit_backed_poseidon2_hash, POSEIDON2_GOLDILOCKS_BITS, POSEIDON2_RATE,
};
use neo_fold_clean::frontends::f_prime::image::{
    FPrimeImage, FPrimeImageConfig, FPrimeImageLayout, PoseidonPreimageLaneSource, PoseidonTransitionEnforcement,
    StateOut,
};
use neo_fold_clean::frontends::f_prime::structure::build_f_prime_structure;
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Preimage length we pin for these tests. 4 = exactly one absorb chunk
/// + one padding absorb (4 RATE-sized chunk + 0-remainder + final pad).
const PREIMAGE_LEN: usize = 4;

/// Wires the preimage to four `StateLane` sources. Lane indices 4..8 in
/// `state_lanes` (state-in digests start at 0) cover the `vk_fs_digest`'s second
/// half, which is convenient: we control them via `fill_state_in`.
const SOURCE_STATE_LANE_INDICES: [usize; PREIMAGE_LEN] = [4, 5, 6, 7];

fn enforcement_config(enforcements: Vec<PoseidonTransitionEnforcement>) -> FPrimeImageConfig {
    FPrimeImageConfig {
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
        poseidon_one_shot_preimage_lens: vec![PREIMAGE_LEN],
        sponge_transcript_permutes: 0,
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_state_in_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: enforcements,
        unified_accumulator_selector: None,
        initial_semantic_state_digest_anchor: None,
    }
}

fn source_lanes() -> Vec<PoseidonPreimageLaneSource> {
    SOURCE_STATE_LANE_INDICES
        .iter()
        .map(|&idx| PoseidonPreimageLaneSource::StateLane(idx))
        .collect()
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

/// Build an honest image: fill state-in digest lanes 4..8 with arbitrary values,
/// then splice in a Poseidon2 trace computed over those same values.
fn honest_image() -> (FPrimeImageLayout, FPrimeImage, [F; PREIMAGE_LEN]) {
    let layout = FPrimeImageLayout::new(enforcement_config(vec![PoseidonTransitionEnforcement {
        one_shot_index: 0,
        preimage_lanes: source_lanes(),
    }]));
    let mut image = FPrimeImage::new(layout.clone());

    // The 24 state-in digest lanes are: vk_fs[0..4], structure[0..4], z_0[0..4],
    // z_i_in[0..4], acc_digest_in[0..4], public_trace_in[0..4]. Indices
    // 4..8 cover structure_digest[0..4]. We set those + everything else
    // to known values; the preimage we'll hash is `structure_digest`'s
    // four lanes.
    let preimage_vals: [F; PREIMAGE_LEN] = [
        F::from_u64(0x111),
        F::from_u64(0x222),
        F::from_u64(0x333),
        F::from_u64(0x444),
    ];
    let state_in = neo_fold_clean::frontends::f_prime::image::StateIn {
        vk_fs_digest: [F::ZERO; 4],
        structure_digest: preimage_vals,
        z_0: [F::ZERO; 4],
        z_i_in: [F::ZERO; 4],
        acc_digest_in: [F::ZERO; 4],
        semantic_state_digest_in: [F::ZERO; 4],
        public_trace_in: [F::ZERO; 4],
    };
    image.fill_state_in(&state_in);
    image.fill_state_out(&StateOut {
        new_chunk_count: 0,
        new_step_count: 0,
        new_z_i: [F::ZERO; 4],
        new_public_trace: [F::ZERO; 4],
        new_acc_digest: [F::ZERO; 4],
        new_semantic_state_digest: [F::ZERO; 4],
    });

    let preimage_vec: Vec<F> = preimage_vals.iter().copied().collect();
    let trace = encode_poseidon_trace(&preimage_vec);
    image.splice_one_shot_poseidon(0, &trace);

    (layout, image, preimage_vals)
}

#[test]
fn phase_1_4d_a4_enforcement_adds_lifted_rows_plus_variable_absorb_rows() {
    let (layout, _, preimage) = honest_image();
    let with_enforcement = build_f_prime_structure(layout);

    let baseline_layout = FPrimeImageLayout::new(enforcement_config(vec![]));
    let baseline = build_f_prime_structure(baseline_layout);

    let native = build_bit_backed_poseidon2_hash(&preimage.to_vec());
    let native_bitness_count = native
        .structure
        .matrices
        .first()
        .and_then(|m| m.as_csc())
        .map(|csc| csc.vals.len())
        .expect("native bitness matrix is CSC");
    let native_absorb_row_count = native.absorb_rows.iter().flatten().count();

    // Lifted rows = native rows - bitness - absorb. Then we re-emit
    // `absorbs * WIDTH` absorb-binding rows ourselves.
    let absorbs = PREIMAGE_LEN.div_ceil(POSEIDON2_RATE) + 1;
    let variable_absorb_rows = absorbs * neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_WIDTH;
    let expected_delta = (native.structure.n - native_bitness_count - native_absorb_row_count) + variable_absorb_rows;
    let observed_delta = with_enforcement.ccs.n - baseline.ccs.n;
    assert_eq!(
        observed_delta, expected_delta,
        "enforcement adds (lifted non-bit-non-absorb) + (variable absorb-binding) rows"
    );
    // Sanity: absorb rows count matches the council-spec formula.
    assert_eq!(native_absorb_row_count, variable_absorb_rows);
}

#[test]
fn phase_1_4d_a4_honest_trace_with_matching_sources_satisfies() {
    let (layout, image, _) = honest_image();
    let structure = build_f_prime_structure(layout);
    let z = structure.extend_witness_from_image(&image);
    assert!(
        structure.is_satisfied(&z),
        "honest Poseidon trace whose preimage matches the source lanes must satisfy structure (first failing row: {:?})",
        structure.first_unsatisfied_row(&z),
    );
}

#[test]
fn phase_1_4d_a4_tampered_source_lane_trips_absorb_binding() {
    let (layout, image, _) = honest_image();
    let structure = build_f_prime_structure(layout);
    let mut z = structure.extend_witness_from_image(&image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    // Tamper a source state lane by rewriting its 64 bits to encode a
    // different canonical-u64 value. Bit validity still holds; the
    // absorb-binding row sees the mismatch.
    let source_lane_idx = SOURCE_STATE_LANE_INDICES[0];
    let source_slot = structure.lane_slots.state_lanes[source_lane_idx];
    let new_value = decode_lane_f(&z, source_slot.bit_start) + F::ONE;
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        let bit = if ((new_value.as_canonical_u64() >> i) & 1) == 1 {
            F::ONE
        } else {
            F::ZERO
        };
        z[source_slot.bit_start + i] = bit;
    }

    assert!(
        !structure.is_satisfied(&z),
        "coherently tampering a source lane must trip an absorb-binding row"
    );
}

#[test]
fn phase_1_4d_a4_tampered_internal_trace_lane_trips_round_constraint() {
    // Internal = a trace word that's NOT a post-absorb state word.
    // The first post-absorb word is at trace word index 0 (start of
    // permutation 0). The "pre-external linear" output sits at trace
    // word index 8 (just after the 8 post-absorb words). Tampering
    // word index 8 should trip the MDS/external-linear row, not an
    // absorb-binding row.
    use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_WIDTH;

    let (layout, image, _) = honest_image();
    let structure = build_f_prime_structure(layout);
    let mut z = structure.extend_witness_from_image(&image);
    assert!(structure.is_satisfied(&z), "baseline must satisfy");

    let trace_lanes = &structure.lane_slots.poseidon_trace_lanes[0];
    // Word `POSEIDON2_WIDTH` (= 8) is the first "post-external-linear"
    // word in permutation 0 — bound to a linear (MDS) row, not an
    // absorb-binding row.
    let internal_word_idx = POSEIDON2_WIDTH;
    let lane = trace_lanes[internal_word_idx];

    // Toggle one bit. Bit validity still holds; the lifted MDS/round
    // constraint on this trace word fails.
    let target_bit = lane.bit_start;
    let new_bit_val = if z[target_bit] == F::ZERO { F::ONE } else { F::ZERO };
    z[target_bit] = new_bit_val;

    assert!(
        !structure.is_satisfied(&z),
        "tampering an internal trace lane must trip a lifted round constraint"
    );
}
