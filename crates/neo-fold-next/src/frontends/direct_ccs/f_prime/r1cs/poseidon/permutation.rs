//! Owns the Poseidon2 permutation expansion used by the direct F' low-norm R1CS shell.

use std::sync::LazyLock;

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, MATRIX_DIAG_8_GOLDILOCKS};
use p3_poseidon2::{poseidon2_round_numbers_128, ExternalLayerConstants};
use rand_chacha_p3::rand_core::{Rng, SeedableRng};
use rand_chacha_p3::ChaCha8Rng;

use super::*;

struct LowNormPoseidon2RoundConstants {
    initial: Vec<[F; POSEIDON2_WIDTH]>,
    terminal: Vec<[F; POSEIDON2_WIDTH]>,
    internal: Vec<F>,
    internal_diag_m_1: [F; POSEIDON2_WIDTH],
}

static LOW_NORM_POSEIDON2_CONSTANTS: LazyLock<LowNormPoseidon2RoundConstants> =
    LazyLock::new(build_low_norm_poseidon2_constants);

pub(super) fn poseidon2_permutation_alloc_rows() -> usize {
    let constants = &*LOW_NORM_POSEIDON2_CONSTANTS;
    let external_layer_rows = 2 * 4 + POSEIDON2_WIDTH;
    let full_round_rows = POSEIDON2_WIDTH * 4 + external_layer_rows;
    let partial_round_rows = 4 + POSEIDON2_WIDTH;
    external_layer_rows
        + (constants.initial.len() + constants.terminal.len()) * full_round_rows
        + constants.internal.len() * partial_round_rows
}

pub(super) fn poseidon2_hash_lcs(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    input: &[FieldLc],
) -> Result<[FieldLc; POSEIDON2_DIGEST_LEN], DirectCcsFPrimeSnarkError> {
    let mut state: [FieldLc; POSEIDON2_WIDTH] = core::array::from_fn(|_| FieldLc::constant(F::ZERO));
    for chunk in input.chunks(POSEIDON2_RATE) {
        for (idx, value) in chunk.iter().enumerate() {
            state[idx] = state[idx].add_scaled(value, F::ONE);
        }
        state = poseidon2_permute_lcs(a_trips, b_trips, c_trips, row, witness, state)?;
    }
    state[0] = state[0].add_constant(F::ONE);
    state = poseidon2_permute_lcs(a_trips, b_trips, c_trips, row, witness, state)?;
    Ok(core::array::from_fn(|idx| state[idx].clone()))
}

fn poseidon2_permute_lcs(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    mut state: [FieldLc; POSEIDON2_WIDTH],
) -> Result<[FieldLc; POSEIDON2_WIDTH], DirectCcsFPrimeSnarkError> {
    let constants = &*LOW_NORM_POSEIDON2_CONSTANTS;
    state = external_linear_layer_lcs(a_trips, b_trips, row, witness, &state)?;
    for round_constants in &constants.initial {
        let mut next = state.clone();
        for idx in 0..POSEIDON2_WIDTH {
            next[idx] = sbox_with_round_constant_lc(
                a_trips,
                b_trips,
                c_trips,
                row,
                witness,
                &state[idx],
                round_constants[idx],
            )?;
        }
        state = external_linear_layer_lcs(a_trips, b_trips, row, witness, &next)?;
    }
    for round_constant in constants.internal.iter().copied() {
        let mut next = state.clone();
        next[0] = sbox_with_round_constant_lc(a_trips, b_trips, c_trips, row, witness, &state[0], round_constant)?;
        state = internal_linear_layer_lcs(a_trips, b_trips, row, witness, &next, constants.internal_diag_m_1)?;
    }
    for round_constants in &constants.terminal {
        let mut next = state.clone();
        for idx in 0..POSEIDON2_WIDTH {
            next[idx] = sbox_with_round_constant_lc(
                a_trips,
                b_trips,
                c_trips,
                row,
                witness,
                &state[idx],
                round_constants[idx],
            )?;
        }
        state = external_linear_layer_lcs(a_trips, b_trips, row, witness, &next)?;
    }
    Ok(state)
}

fn sbox_with_round_constant_lc(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    input: &FieldLc,
    round_constant: F,
) -> Result<FieldLc, DirectCcsFPrimeSnarkError> {
    let shifted = input.add_constant(round_constant);
    let shifted_sq = alloc_mul_lane(a_trips, b_trips, c_trips, row, witness, &shifted, &shifted)?;
    let shifted_4 = alloc_mul_lane(a_trips, b_trips, c_trips, row, witness, &shifted_sq, &shifted_sq)?;
    let shifted_6 = alloc_mul_lane(a_trips, b_trips, c_trips, row, witness, &shifted_4, &shifted_sq)?;
    alloc_mul_lane(a_trips, b_trips, c_trips, row, witness, &shifted_6, &shifted)
}

fn external_linear_layer_lcs(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    state: &[FieldLc; POSEIDON2_WIDTH],
) -> Result<[FieldLc; POSEIDON2_WIDTH], DirectCcsFPrimeSnarkError> {
    let left = apply_mat4_lcs(
        a_trips,
        b_trips,
        row,
        witness,
        core::array::from_fn(|i| state[i].clone()),
    )?;
    let right = apply_mat4_lcs(
        a_trips,
        b_trips,
        row,
        witness,
        core::array::from_fn(|i| state[i + 4].clone()),
    )?;
    let two = F::from_u64(2);
    let mut out = core::array::from_fn(|idx| left[idx % 4].clone());
    for idx in 0..4 {
        out[idx] = alloc_affine_lane(
            a_trips,
            b_trips,
            row,
            witness,
            left[idx]
                .add_scaled(&right[idx], F::ONE)
                .add_scaled(&left[idx], F::ONE),
        )?;
        out[idx + 4] = alloc_affine_lane(a_trips, b_trips, row, witness, left[idx].add_scaled(&right[idx], two))?;
    }
    Ok(out)
}

fn apply_mat4_lcs(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    state: [FieldLc; 4],
) -> Result<[FieldLc; 4], DirectCcsFPrimeSnarkError> {
    let two = F::from_u64(2);
    let three = F::from_u64(3);
    Ok([
        alloc_affine_lane(
            a_trips,
            b_trips,
            row,
            witness,
            state[0]
                .add_scaled(&state[1], three)
                .add_scaled(&state[2], F::ONE)
                .add_scaled(&state[3], F::ONE)
                .add_scaled(&state[0], F::ONE),
        )?,
        alloc_affine_lane(
            a_trips,
            b_trips,
            row,
            witness,
            state[0]
                .add_scaled(&state[1], two)
                .add_scaled(&state[2], three)
                .add_scaled(&state[3], F::ONE),
        )?,
        alloc_affine_lane(
            a_trips,
            b_trips,
            row,
            witness,
            state[0]
                .add_scaled(&state[1], F::ONE)
                .add_scaled(&state[2], two)
                .add_scaled(&state[3], three),
        )?,
        alloc_affine_lane(
            a_trips,
            b_trips,
            row,
            witness,
            state[0]
                .add_scaled(&state[1], F::ONE)
                .add_scaled(&state[2], F::ONE)
                .add_scaled(&state[3], two)
                .add_scaled(&state[0], two),
        )?,
    ])
}

fn internal_linear_layer_lcs(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    state: &[FieldLc; POSEIDON2_WIDTH],
    diag_m_1: [F; POSEIDON2_WIDTH],
) -> Result<[FieldLc; POSEIDON2_WIDTH], DirectCcsFPrimeSnarkError> {
    let mut out = state.clone();
    for idx in 0..POSEIDON2_WIDTH {
        let mut lc = FieldLc::constant(F::ZERO);
        for (j, lane) in state.iter().enumerate() {
            let coeff = if idx == j { diag_m_1[idx] + F::ONE } else { F::ONE };
            lc = lc.add_scaled(lane, coeff);
        }
        out[idx] = alloc_affine_lane(a_trips, b_trips, row, witness, lc)?;
    }
    Ok(out)
}

fn alloc_mul_lane(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    left: &FieldLc,
    right: &FieldLc,
) -> Result<FieldLc, DirectCcsFPrimeSnarkError> {
    let out = append_field_lane_bits(witness, left.value * right.value);
    let out_lc = FieldLc::lane(witness, out)?;
    push_lc_trips(a_trips, *row, left);
    push_lc_trips(b_trips, *row, right);
    push_lc_trips(c_trips, *row, &out_lc);
    *row += 1;
    Ok(out_lc)
}

fn alloc_affine_lane(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    value: FieldLc,
) -> Result<FieldLc, DirectCcsFPrimeSnarkError> {
    let out = append_field_lane_bits(witness, value.value);
    let out_lc = FieldLc::lane(witness, out)?;
    let diff = value.add_scaled(&out_lc, -F::ONE);
    push_lc_trips(a_trips, *row, &diff);
    b_trips.push((*row, ONE_COL, F::ONE));
    *row += 1;
    Ok(out_lc)
}

fn build_low_norm_poseidon2_constants() -> LowNormPoseidon2RoundConstants {
    let mut rng = ChaCha8Rng::from_seed(neo_params::poseidon2_goldilocks::SEED);
    let (rounds_f, rounds_p) = poseidon2_round_numbers_128::<Goldilocks>(POSEIDON2_WIDTH, POSEIDON2_SBOX_DEGREE)
        .expect("Poseidon2 width 8 round numbers");
    let external = ExternalLayerConstants::<Goldilocks, POSEIDON2_WIDTH>::new_from_rng(rounds_f, &mut rng);
    let internal = (0..rounds_p)
        .map(|_| goldilocks_to_f(Goldilocks::from_u64(rng.next_u64())))
        .collect::<Vec<_>>();
    LowNormPoseidon2RoundConstants {
        initial: external
            .get_initial_constants()
            .iter()
            .copied()
            .map(goldilocks_array_to_f)
            .collect(),
        terminal: external
            .get_terminal_constants()
            .iter()
            .copied()
            .map(goldilocks_array_to_f)
            .collect(),
        internal,
        internal_diag_m_1: goldilocks_array_to_f(MATRIX_DIAG_8_GOLDILOCKS),
    }
}

fn goldilocks_to_f(value: Goldilocks) -> F {
    F::from_u64(value.as_canonical_u64())
}

fn goldilocks_array_to_f<const N: usize>(values: [Goldilocks; N]) -> [F; N] {
    core::array::from_fn(|idx| goldilocks_to_f(values[idx]))
}
