//! Owns the in-circuit Poseidon2 permutation used by the SuperNeo transcript.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ccs::crypto::poseidon2_goldilocks;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, MATRIX_DIAG_8_GOLDILOCKS};
use p3_poseidon2::{poseidon2_round_numbers_128, ExternalLayerConstants};
use rand_chacha_p3::rand_core::{Rng, SeedableRng};
use rand_chacha_p3::ChaCha8Rng;
use std::sync::LazyLock;

use super::lane::{combine_scaled_lanes, TranscriptLane};
use super::{GOLDILOCKS_S_BOX_DEGREE, WIDTH};
use crate::spartan_backend::SpartanF;

struct Poseidon2RoundConstants {
    initial: Vec<[SpartanF; WIDTH]>,
    terminal: Vec<[SpartanF; WIDTH]>,
    internal: Vec<SpartanF>,
    internal_diag_m_1: [SpartanF; WIDTH],
}

static POSEIDON2_CONSTANTS: LazyLock<Poseidon2RoundConstants> = LazyLock::new(build_poseidon2_constants);

fn build_poseidon2_constants() -> Poseidon2RoundConstants {
    let mut rng = ChaCha8Rng::from_seed(neo_params::poseidon2_goldilocks::SEED);
    let (rounds_f, rounds_p) =
        poseidon2_round_numbers_128::<Goldilocks>(WIDTH, GOLDILOCKS_S_BOX_DEGREE).expect("Poseidon2 width 8 rounds");
    let external = ExternalLayerConstants::<Goldilocks, WIDTH>::new_from_rng(rounds_f, &mut rng);
    let internal = (0..rounds_p)
        .map(|_| convert_goldilocks(Goldilocks::from_u64(rng.next_u64())))
        .collect::<Vec<_>>();

    Poseidon2RoundConstants {
        initial: external
            .get_initial_constants()
            .iter()
            .copied()
            .map(convert_goldilocks_array)
            .collect(),
        terminal: external
            .get_terminal_constants()
            .iter()
            .copied()
            .map(convert_goldilocks_array)
            .collect(),
        internal,
        internal_diag_m_1: convert_goldilocks_array(MATRIX_DIAG_8_GOLDILOCKS),
    }
}

pub(super) fn permute_state<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    state: &[TranscriptLane; WIDTH],
) -> Result<[TranscriptLane; WIDTH], SynthesisError> {
    if state.iter().all(TranscriptLane::is_constant) {
        let mut permuted = core::array::from_fn(|idx| Goldilocks::from_u64(state[idx].value.to_canonical_u64()));
        permuted = poseidon2_goldilocks::permute_state(permuted);
        return Ok(permuted.map(|value| TranscriptLane::from_constant(convert_goldilocks(value))));
    }

    let constants = &*POSEIDON2_CONSTANTS;

    let mut state = external_linear_layer(cs.namespace(|| "initial_external_layer"), state)?;

    for (round_idx, round_constants) in constants.initial.iter().enumerate() {
        let mut next = state.clone();
        for i in 0..WIDTH {
            next[i] = sbox_with_round_constant(
                cs.namespace(|| format!("initial_round_{round_idx}_{i}")),
                &state[i],
                round_constants[i],
            )?;
        }
        state = external_linear_layer(cs.namespace(|| format!("initial_round_{round_idx}_linear")), &next)?;
    }

    for (round_idx, round_constant) in constants.internal.iter().copied().enumerate() {
        let mut next = state.clone();
        next[0] = sbox_with_round_constant(
            cs.namespace(|| format!("internal_round_{round_idx}_0")),
            &state[0],
            round_constant,
        )?;
        state = internal_linear_layer(
            cs.namespace(|| format!("internal_round_{round_idx}_linear")),
            &next,
            constants.internal_diag_m_1,
        )?;
    }

    for (round_idx, round_constants) in constants.terminal.iter().enumerate() {
        let mut next = state.clone();
        for i in 0..WIDTH {
            next[i] = sbox_with_round_constant(
                cs.namespace(|| format!("terminal_round_{round_idx}_{i}")),
                &state[i],
                round_constants[i],
            )?;
        }
        state = external_linear_layer(cs.namespace(|| format!("terminal_round_{round_idx}_linear")), &next)?;
    }

    Ok(state)
}

fn sbox_with_round_constant<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    input: &TranscriptLane,
    round_constant: SpartanF,
) -> Result<TranscriptLane, SynthesisError> {
    let shifted_value = input.value + round_constant;

    let shifted_sq_value = shifted_value.square();
    let shifted_sq = AllocatedNum::alloc(cs.namespace(|| "shift_sq"), || Ok(shifted_sq_value))?;
    cs.enforce(
        || "shift_sq_enforce",
        |_| input.lc::<CS>() + (round_constant, CS::one()),
        |_| input.lc::<CS>() + (round_constant, CS::one()),
        |lc| lc + shifted_sq.get_variable(),
    );
    let shifted_sq_lane = TranscriptLane::from_allocated(shifted_sq, shifted_sq_value);

    let shifted_4 = square_lane(cs.namespace(|| "shift_4"), &shifted_sq_lane)?;
    let shifted_6 = mul_lanes(cs.namespace(|| "shift_6"), &shifted_4, &shifted_sq_lane)?;

    let out_value = shifted_6.value * shifted_value;
    let out = AllocatedNum::alloc(cs.namespace(|| "out"), || Ok(out_value))?;
    cs.enforce(
        || "out_enforce",
        |_| shifted_6.lc::<CS>(),
        |_| input.lc::<CS>() + (round_constant, CS::one()),
        |lc| lc + out.get_variable(),
    );

    Ok(TranscriptLane::from_allocated(out, out_value))
}

fn external_linear_layer<CS: ConstraintSystem<SpartanF>>(
    _cs: CS,
    state: &[TranscriptLane; WIDTH],
) -> Result<[TranscriptLane; WIDTH], SynthesisError> {
    let left = apply_mat4(&state[0..4]);
    let right = apply_mat4(&state[4..8]);

    let two = SpartanF::from_canonical_u64(2);
    let mut out = core::array::from_fn(|i| left[i % 4].clone());
    for i in 0..4 {
        out[i] = combine_scaled_lanes(&[(&left[i], two), (&right[i], SpartanF::ONE)]);
        out[i + 4] = combine_scaled_lanes(&[(&left[i], SpartanF::ONE), (&right[i], two)]);
    }
    Ok(out)
}

fn apply_mat4(state: &[TranscriptLane]) -> [TranscriptLane; 4] {
    let two = SpartanF::from_canonical_u64(2);
    let three = SpartanF::from_canonical_u64(3);

    let row_0 = combine_scaled_lanes(&[
        (&state[0], two),
        (&state[1], three),
        (&state[2], SpartanF::ONE),
        (&state[3], SpartanF::ONE),
    ]);
    let row_1 = combine_scaled_lanes(&[
        (&state[0], SpartanF::ONE),
        (&state[1], two),
        (&state[2], three),
        (&state[3], SpartanF::ONE),
    ]);
    let row_2 = combine_scaled_lanes(&[
        (&state[0], SpartanF::ONE),
        (&state[1], SpartanF::ONE),
        (&state[2], two),
        (&state[3], three),
    ]);
    let row_3 = combine_scaled_lanes(&[
        (&state[0], three),
        (&state[1], SpartanF::ONE),
        (&state[2], SpartanF::ONE),
        (&state[3], two),
    ]);

    [row_0, row_1, row_2, row_3]
}

fn internal_linear_layer<CS: ConstraintSystem<SpartanF>>(
    _cs: CS,
    state: &[TranscriptLane; WIDTH],
    diag_m_1: [SpartanF; WIDTH],
) -> Result<[TranscriptLane; WIDTH], SynthesisError> {
    let sum_inputs = state
        .iter()
        .map(|lane| (lane, SpartanF::ONE))
        .collect::<Vec<_>>();
    let sum = combine_scaled_lanes(&sum_inputs);

    let out = core::array::from_fn(|i| combine_scaled_lanes(&[(&sum, SpartanF::ONE), (&state[i], diag_m_1[i])]));
    Ok(out)
}

fn square_lane<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    input: &TranscriptLane,
) -> Result<TranscriptLane, SynthesisError> {
    let out_value = input.value.square();
    let out = AllocatedNum::alloc(cs.namespace(|| "value"), || Ok(out_value))?;
    cs.enforce(
        || "square",
        |_| input.lc::<CS>(),
        |_| input.lc::<CS>(),
        |lc| lc + out.get_variable(),
    );
    Ok(TranscriptLane::from_allocated(out, out_value))
}

fn mul_lanes<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    left: &TranscriptLane,
    right: &TranscriptLane,
) -> Result<TranscriptLane, SynthesisError> {
    let out_value = left.value * right.value;
    let out = AllocatedNum::alloc(cs.namespace(|| "value"), || Ok(out_value))?;
    cs.enforce(
        || "mul",
        |_| left.lc::<CS>(),
        |_| right.lc::<CS>(),
        |lc| lc + out.get_variable(),
    );
    Ok(TranscriptLane::from_allocated(out, out_value))
}

fn convert_goldilocks(value: Goldilocks) -> SpartanF {
    SpartanF::from_canonical_u64(value.as_canonical_u64())
}

fn convert_goldilocks_array<const N: usize>(values: [Goldilocks; N]) -> [SpartanF; N] {
    values.map(convert_goldilocks)
}
