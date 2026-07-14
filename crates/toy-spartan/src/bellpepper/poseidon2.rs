//! Owns a minimal bellpepper Poseidon2 gadget over Toy Spartan's Goldilocks field.

use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::Field;
use once_cell::sync::Lazy;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, MATRIX_DIAG_8_GOLDILOCKS};
use p3_poseidon2::{ExternalLayerConstants, poseidon2_round_numbers_128};
use rand_chacha_p3::{
  ChaCha8Rng,
  rand_core::{Rng, SeedableRng},
};

use crate::provider::goldi::F;

/// Poseidon2 state width used by the backend-binding shell.
pub const POSEIDON2_WIDTH: usize = neo_params::poseidon2_goldilocks::WIDTH;
/// Poseidon2 sponge rate used by the backend-binding shell.
pub const POSEIDON2_RATE: usize = neo_params::poseidon2_goldilocks::RATE;
/// Poseidon2 digest length in field elements used by the backend-binding shell.
pub const POSEIDON2_DIGEST_LEN: usize = neo_params::poseidon2_goldilocks::DIGEST_LEN;

const GOLDILOCKS_S_BOX_DEGREE: u64 = 7;

struct Poseidon2RoundConstants {
  initial: Vec<[F; POSEIDON2_WIDTH]>,
  terminal: Vec<[F; POSEIDON2_WIDTH]>,
  internal: Vec<F>,
  internal_diag_m_1: [F; POSEIDON2_WIDTH],
}

static POSEIDON2_CONSTANTS: Lazy<Poseidon2RoundConstants> = Lazy::new(build_poseidon2_constants);

/// Hash packed Goldilocks witness fields with the canonical width-8 Poseidon2 sponge.
pub fn hash_packed_goldilocks_fields<CS: ConstraintSystem<F>>(
  mut cs: CS,
  input: &[AllocatedNum<F>],
) -> Result<[AllocatedNum<F>; POSEIDON2_DIGEST_LEN], SynthesisError> {
  let mut state = core::array::from_fn(|i| {
    alloc_affine(cs.namespace(|| format!("zero_state_{i}")), &[], F::ZERO)
      .expect("zero state allocation must succeed")
  });

  for (chunk_idx, chunk) in input.chunks(POSEIDON2_RATE).enumerate() {
    for (i, value) in chunk.iter().enumerate() {
      state[i] = alloc_affine(
        cs.namespace(|| format!("absorb_chunk_{chunk_idx}_{i}")),
        &[(state[i].clone(), F::ONE), (value.clone(), F::ONE)],
        F::ZERO,
      )?;
    }
    state = permute_state(
      cs.namespace(|| format!("permute_after_chunk_{chunk_idx}")),
      &state,
    )?;
  }

  state[0] = alloc_affine(
    cs.namespace(|| "padding_one"),
    &[(state[0].clone(), F::ONE)],
    F::ONE,
  )?;
  state = permute_state(cs.namespace(|| "permute_after_padding"), &state)?;

  Ok(core::array::from_fn(|i| state[i].clone()))
}

fn build_poseidon2_constants() -> Poseidon2RoundConstants {
  let mut rng = ChaCha8Rng::from_seed(neo_params::poseidon2_goldilocks::SEED);
  let (rounds_f, rounds_p) =
    poseidon2_round_numbers_128::<Goldilocks>(POSEIDON2_WIDTH, GOLDILOCKS_S_BOX_DEGREE)
      .expect("Poseidon2 width 8 round numbers must exist");
  let external =
    ExternalLayerConstants::<Goldilocks, POSEIDON2_WIDTH>::new_from_rng(rounds_f, &mut rng);
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

fn permute_state<CS: ConstraintSystem<F>>(
  mut cs: CS,
  state: &[AllocatedNum<F>; POSEIDON2_WIDTH],
) -> Result<[AllocatedNum<F>; POSEIDON2_WIDTH], SynthesisError> {
  let constants = &*POSEIDON2_CONSTANTS;

  let mut state = external_linear_layer(cs.namespace(|| "initial_external_layer"), state)?;

  for (round_idx, round_constants) in constants.initial.iter().enumerate() {
    let mut next = core::array::from_fn(|i| state[i].clone());
    for i in 0..POSEIDON2_WIDTH {
      next[i] = sbox_with_round_constant(
        cs.namespace(|| format!("initial_round_{round_idx}_state_{i}")),
        &state[i],
        round_constants[i],
      )?;
    }
    state = external_linear_layer(
      cs.namespace(|| format!("initial_round_{round_idx}_external_layer")),
      &next,
    )?;
  }

  for (round_idx, round_constant) in constants.internal.iter().copied().enumerate() {
    let mut next = core::array::from_fn(|i| state[i].clone());
    next[0] = sbox_with_round_constant(
      cs.namespace(|| format!("internal_round_{round_idx}_state_0")),
      &state[0],
      round_constant,
    )?;
    state = internal_linear_layer(
      cs.namespace(|| format!("internal_round_{round_idx}_linear_layer")),
      &next,
      constants.internal_diag_m_1,
    )?;
  }

  for (round_idx, round_constants) in constants.terminal.iter().enumerate() {
    let mut next = core::array::from_fn(|i| state[i].clone());
    for i in 0..POSEIDON2_WIDTH {
      next[i] = sbox_with_round_constant(
        cs.namespace(|| format!("terminal_round_{round_idx}_state_{i}")),
        &state[i],
        round_constants[i],
      )?;
    }
    state = external_linear_layer(
      cs.namespace(|| format!("terminal_round_{round_idx}_external_layer")),
      &next,
    )?;
  }

  Ok(state)
}

fn sbox_with_round_constant<CS: ConstraintSystem<F>>(
  mut cs: CS,
  input: &AllocatedNum<F>,
  round_constant: F,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let shifted = alloc_affine(
    cs.namespace(|| "shift"),
    &[(input.clone(), F::ONE)],
    round_constant,
  )?;
  let shifted_sq = shifted.square(cs.namespace(|| "shift_sq"))?;
  let shifted_4 = shifted_sq.square(cs.namespace(|| "shift_4"))?;
  let shifted_6 = shifted_4.mul(cs.namespace(|| "shift_6"), &shifted_sq)?;
  shifted_6.mul(cs.namespace(|| "shift_7"), &shifted)
}

fn external_linear_layer<CS: ConstraintSystem<F>>(
  mut cs: CS,
  state: &[AllocatedNum<F>; POSEIDON2_WIDTH],
) -> Result<[AllocatedNum<F>; POSEIDON2_WIDTH], SynthesisError> {
  let left = apply_mat4(
    cs.namespace(|| "left_mat4"),
    &core::array::from_fn(|i| state[i].clone()),
  )?;
  let right = apply_mat4(
    cs.namespace(|| "right_mat4"),
    &core::array::from_fn(|i| state[i + 4].clone()),
  )?;

  let two = F::from_canonical_u64(2);
  let mut out = core::array::from_fn(|i| left[i % 4].clone());
  for i in 0..4 {
    out[i] = alloc_affine(
      cs.namespace(|| format!("outer_left_{i}")),
      &[(left[i].clone(), two), (right[i].clone(), F::ONE)],
      F::ZERO,
    )?;
    out[i + 4] = alloc_affine(
      cs.namespace(|| format!("outer_right_{i}")),
      &[(left[i].clone(), F::ONE), (right[i].clone(), two)],
      F::ZERO,
    )?;
  }
  Ok(out)
}

fn apply_mat4<CS: ConstraintSystem<F>>(
  mut cs: CS,
  state: &[AllocatedNum<F>; 4],
) -> Result<[AllocatedNum<F>; 4], SynthesisError> {
  let two = F::from_canonical_u64(2);
  let three = F::from_canonical_u64(3);

  Ok([
    alloc_affine(
      cs.namespace(|| "row_0"),
      &[
        (state[0].clone(), two),
        (state[1].clone(), three),
        (state[2].clone(), F::ONE),
        (state[3].clone(), F::ONE),
      ],
      F::ZERO,
    )?,
    alloc_affine(
      cs.namespace(|| "row_1"),
      &[
        (state[0].clone(), F::ONE),
        (state[1].clone(), two),
        (state[2].clone(), three),
        (state[3].clone(), F::ONE),
      ],
      F::ZERO,
    )?,
    alloc_affine(
      cs.namespace(|| "row_2"),
      &[
        (state[0].clone(), F::ONE),
        (state[1].clone(), F::ONE),
        (state[2].clone(), two),
        (state[3].clone(), three),
      ],
      F::ZERO,
    )?,
    alloc_affine(
      cs.namespace(|| "row_3"),
      &[
        (state[0].clone(), three),
        (state[1].clone(), F::ONE),
        (state[2].clone(), F::ONE),
        (state[3].clone(), two),
      ],
      F::ZERO,
    )?,
  ])
}

fn internal_linear_layer<CS: ConstraintSystem<F>>(
  mut cs: CS,
  state: &[AllocatedNum<F>; POSEIDON2_WIDTH],
  diag_m_1: [F; POSEIDON2_WIDTH],
) -> Result<[AllocatedNum<F>; POSEIDON2_WIDTH], SynthesisError> {
  let mut out = core::array::from_fn(|i| state[i].clone());
  for i in 0..POSEIDON2_WIDTH {
    let terms = (0..POSEIDON2_WIDTH)
      .map(|j| {
        let coeff = if i == j { diag_m_1[i] + F::ONE } else { F::ONE };
        (state[j].clone(), coeff)
      })
      .collect::<Vec<_>>();
    out[i] = alloc_affine(cs.namespace(|| format!("row_{i}")), &terms, F::ZERO)?;
  }
  Ok(out)
}

fn alloc_affine<CS: ConstraintSystem<F>>(
  mut cs: CS,
  terms: &[(AllocatedNum<F>, F)],
  constant: F,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let out = AllocatedNum::alloc(cs.namespace(|| "value"), || {
    let mut acc = constant;
    for (value, coeff) in terms {
      let term = value.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      acc += term * *coeff;
    }
    Ok(acc)
  })?;

  cs.enforce(
    || "affine",
    |lc| {
      let mut lc = lc + (constant, CS::one());
      for (value, coeff) in terms {
        lc = lc + (*coeff, value.get_variable());
      }
      lc
    },
    |lc| lc + CS::one(),
    |lc| lc + out.get_variable(),
  );

  Ok(out)
}

fn convert_goldilocks(value: Goldilocks) -> F {
  F::from_canonical_u64(value.as_canonical_u64())
}

fn convert_goldilocks_array<const N: usize>(values: [Goldilocks; N]) -> [F; N] {
  core::array::from_fn(|i| convert_goldilocks(values[i]))
}
