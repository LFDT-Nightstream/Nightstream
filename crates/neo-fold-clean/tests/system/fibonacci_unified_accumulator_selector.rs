//! Delayed accumulator-handle binding for the canonical Fibonacci F' plan.
//!
//! This target used to test the producer-side unified accumulator selector.
//! The canonical plan no longer emits that selector: it carries the outgoing
//! accumulator handle in `state_out`, absorbs it into `state_x_out`, and relies
//! on the next recursive step or terminal fold to recompute the handle from
//! the consumed running accumulator.

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::f_prime::encoder::encode_f_prime_step;
use neo_fold_clean::frontends::f_prime::image::{PoseidonPreimageLaneSource, StateOutDigestTarget};
use neo_fold_clean::frontends::f_prime::recursive_plan::build_recursive_step_image_config;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use support::fibonacci_f_prime::{build_honest_step_input, canonical_threaded_plan};

#[test]
fn unified_plan_uses_delayed_accumulator_binding() {
    let plan = canonical_threaded_plan();
    let config = build_recursive_step_image_config(&plan);

    assert!(
        config.unified_accumulator_selector.is_none(),
        "canonical unified mode no longer uses a producer-side selector",
    );
    assert_eq!(
        config.poseidon_one_shot_preimage_lens.len(),
        1,
        "canonical unified mode emits only the state_x_out hash",
    );
    assert!(
        !config
            .one_shot_digest_to_state_out_bindings
            .iter()
            .any(|binding| binding.state_out_target == StateOutDigestTarget::NewAccDigest),
        "producer side must not bind new_acc_digest to a local accumulator hash trace",
    );
    assert_eq!(
        config.one_shot_digest_to_public_x_out_bindings[0].one_shot_index, 0,
        "state_x_out is one-shot index 0 after removing accumulator, public_trace, and boundary hash traces",
    );
}

#[test]
fn unified_plan_does_not_hash_parent_c_data_inside_producer_image() {
    let plan = canonical_threaded_plan();
    let config = build_recursive_step_image_config(&plan);

    let parent_c_data_lanes = config
        .poseidon_transition_enforcements
        .iter()
        .flat_map(|enforcement| enforcement.preimage_lanes.iter())
        .filter(|lane| matches!(lane, PoseidonPreimageLaneSource::NifsPayloadLane { .. }))
        .count();

    assert_eq!(
        parent_c_data_lanes, 0,
        "no producer-side Poseidon hash may absorb parent.c_data lanes from the NIFS payload",
    );
}

#[test]
fn unified_honest_fixture_satisfies_without_accumulator_trace() {
    let (input, _) = build_honest_step_input();
    let encoded = encode_f_prime_step(input);

    assert_eq!(
        encoded.image.layout.one_shot_poseidon_layouts.len(),
        1,
        "fixture should match the canonical delayed-handle compact layout",
    );
    assert!(
        encoded.structure.is_satisfied(&encoded.witness),
        "honest delayed-handle image must satisfy the F' structure",
    );
}

#[test]
fn unified_state_x_out_still_absorbs_outgoing_accumulator_handle() {
    let (input, _) = build_honest_step_input();
    let encoded = encode_f_prime_step(input);
    let mut witness = encoded.witness.clone();

    // state_out layout: two u64 counters, then new_z_i, new_public_trace,
    // new_semantic_state_digest, new_acc_digest. Flip one committed bit in
    // new_acc_digest while leaving the state_x_out Poseidon trace untouched.
    let acc_digest_bit =
        encoded.image.layout.state_out.offset + 2 * POSEIDON2_GOLDILOCKS_BITS + 3 * 4 * POSEIDON2_GOLDILOCKS_BITS;
    witness[acc_digest_bit] = if witness[acc_digest_bit] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };

    assert!(
        !encoded.structure.is_satisfied(&witness),
        "tampering state_out.new_acc_digest without rebuilding state_x_out must make the image unsatisfied",
    );
}

#[test]
fn unified_stateless_semantic_digest_must_equal_accumulator_handle() {
    let (input, _) = build_honest_step_input();
    let encoded = encode_f_prime_step(input);
    let mut witness = encoded.witness.clone();

    // The stateless state_x_out preimage omits new_semantic_state_digest
    // because the CCS structure separately enforces
    // new_semantic_state_digest == new_acc_digest. Flip only the semantic
    // lane; this must fail even though the state_x_out trace itself does not
    // absorb the semantic lanes.
    let semantic_digest_bit =
        encoded.image.layout.state_out.offset + 2 * POSEIDON2_GOLDILOCKS_BITS + 2 * 4 * POSEIDON2_GOLDILOCKS_BITS;
    witness[semantic_digest_bit] = if witness[semantic_digest_bit] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };

    assert!(
        !encoded.structure.is_satisfied(&witness),
        "stateless F' image accepted semantic_state_digest != acc_digest",
    );
}

#[test]
fn unified_public_x_out_binding_row_helper_counts_stateless_rows() {
    let (input, _) = build_honest_step_input();
    let encoded = encode_f_prime_step(input);
    let mut witness = encoded.witness.clone();
    assert!(encoded.structure.is_satisfied(&witness), "baseline must satisfy");

    let binding = &encoded
        .image
        .layout
        .config
        .one_shot_digest_to_public_x_out_bindings[0];
    let target_bit = binding.public_x_out_lane_bit_starts[0];
    witness[target_bit] = if witness[target_bit] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };

    assert_eq!(
        encoded.structure.first_unsatisfied_row(&witness),
        Some(encoded.structure.public_x_out_binding_row(0, 0)),
        "public_x_out row helper must include the stateless semantic/acc rows emitted before it",
    );
}
