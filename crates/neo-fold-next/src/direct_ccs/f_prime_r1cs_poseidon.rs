//! Owns Poseidon2 linkage constraints for the compact direct F' source shell.

use std::sync::LazyLock;

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, MATRIX_DIAG_8_GOLDILOCKS};
use p3_poseidon2::{poseidon2_round_numbers_128, ExternalLayerConstants};
use rand_chacha_p3::rand_core::{Rng, SeedableRng};
use rand_chacha_p3::ChaCha8Rng;

use super::f_prime::DirectCcsFPrimeLowNormSourceImage;
use super::ivc::DirectCcsFPrimeSnarkError;
use crate::construction2::CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG;

const ONE_COL: usize = 0;
const U64_BITS: usize = 64;
const POSEIDON2_WIDTH: usize = neo_params::poseidon2_goldilocks::WIDTH;
const POSEIDON2_RATE: usize = neo_params::poseidon2_goldilocks::RATE;
const POSEIDON2_DIGEST_LEN: usize = neo_params::poseidon2_goldilocks::DIGEST_LEN;
const POSEIDON2_SBOX_DEGREE: u64 = 7;

struct LowNormPoseidon2RoundConstants {
    initial: Vec<[F; POSEIDON2_WIDTH]>,
    terminal: Vec<[F; POSEIDON2_WIDTH]>,
    internal: Vec<F>,
    internal_diag_m_1: [F; POSEIDON2_WIDTH],
}

static LOW_NORM_POSEIDON2_CONSTANTS: LazyLock<LowNormPoseidon2RoundConstants> =
    LazyLock::new(build_low_norm_poseidon2_constants);

#[derive(Clone)]
struct FieldLc {
    terms: Vec<(usize, F)>,
    constant: F,
    value: F,
}

impl FieldLc {
    fn constant(value: F) -> Self {
        Self {
            terms: Vec::new(),
            constant: value,
            value,
        }
    }

    fn lane(witness: &[F], start_col: usize) -> Result<Self, DirectCcsFPrimeSnarkError> {
        Self::bits(witness, start_col, U64_BITS)
    }

    fn bits(witness: &[F], start_col: usize, bit_len: usize) -> Result<Self, DirectCcsFPrimeSnarkError> {
        let value = field_bits_value(witness, start_col, bit_len)?;
        let mut coeff = F::ONE;
        let mut terms = Vec::with_capacity(bit_len);
        for bit_index in 0..bit_len {
            terms.push((start_col + bit_index, coeff));
            coeff += coeff;
        }
        Ok(Self {
            terms,
            constant: F::ZERO,
            value,
        })
    }

    fn add_scaled(&self, other: &Self, scalar: F) -> Self {
        let mut terms = self.terms.clone();
        terms.extend(
            other
                .terms
                .iter()
                .filter(|(_, coeff)| *coeff != F::ZERO)
                .map(|(col, coeff)| (*col, *coeff * scalar)),
        );
        Self {
            terms,
            constant: self.constant + other.constant * scalar,
            value: self.value + other.value * scalar,
        }
    }

    fn add_constant(&self, constant: F) -> Self {
        Self {
            terms: self.terms.clone(),
            constant: self.constant + constant,
            value: self.value + constant,
        }
    }
}

pub(crate) fn add_poseidon_linkage_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    source: &DirectCcsFPrimeLowNormSourceImage,
    source_start_col: usize,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    add_direct_state_image_digest_constraints(
        a_trips,
        b_trips,
        c_trips,
        row,
        witness,
        source,
        source_start_col,
        source.chunk_count_in_bit_offset(),
        source.step_count_in_bit_offset(),
        source.current_boundary_in_digest_bit_offset(),
        source.semantic_accumulator_in_digest_bit_offset(),
        source.f_prime_accumulator_in_digest_bit_offset(),
        source.public_trace_in_digest_bit_offset(),
        source.compact_x_in_bit_offset(),
    )?;
    add_direct_state_image_digest_constraints(
        a_trips,
        b_trips,
        c_trips,
        row,
        witness,
        source,
        source_start_col,
        source.chunk_count_out_bit_offset(),
        source.step_count_out_bit_offset(),
        source.current_boundary_out_digest_bit_offset(),
        source.semantic_accumulator_out_digest_bit_offset(),
        source.f_prime_accumulator_out_digest_bit_offset(),
        source.public_trace_out_digest_bit_offset(),
        source.compact_x_out_bit_offset(),
    )?;
    add_digest_update_poseidon_constraints(
        a_trips,
        b_trips,
        c_trips,
        row,
        witness,
        source,
        source_start_col,
        b"neo.fold.next/direct_ccs/current_boundary_update/v1",
        source.current_boundary_in_digest_bit_offset(),
        source.current_boundary_out_digest_bit_offset(),
    )?;
    add_digest_update_poseidon_constraints(
        a_trips,
        b_trips,
        c_trips,
        row,
        witness,
        source,
        source_start_col,
        b"neo.fold.next/direct_ccs/public_trace_update/v1",
        source.public_trace_in_digest_bit_offset(),
        source.public_trace_out_digest_bit_offset(),
    )?;
    add_construction2_fresh_instance_digest_constraints(
        a_trips,
        b_trips,
        c_trips,
        row,
        witness,
        source_start_col,
        source.construction2_u_in_commitment_digest_bit_offset(),
        source.construction2_u_in_x_i_bit_offset(),
        source.construction2_u_in_fresh_digest_bit_offset(),
    )
}

pub(crate) fn estimated_poseidon_digest_recomputation_cost() -> (usize, usize) {
    let state_image_fields = direct_domain_field_lcs(b"neo.fold.next/direct_ccs/f_prime_x_out/v2").len()
        + 4
        + 4
        + 2
        + 2
        + 4
        + 4
        + 2
        + 4
        + 4
        + 4;
    let digest_update_fields = direct_domain_field_lcs(b"neo.fold.next/direct_ccs/current_boundary_update/v1").len()
        + 2 * POSEIDON2_DIGEST_LEN;
    let public_trace_fields =
        direct_domain_field_lcs(b"neo.fold.next/direct_ccs/public_trace_update/v1").len() + 2 * POSEIDON2_DIGEST_LEN;
    let construction2_boundary_fields = 1 + 2 * POSEIDON2_DIGEST_LEN;
    let input_lengths = [
        state_image_fields,
        state_image_fields,
        digest_update_fields,
        public_trace_fields,
        construction2_boundary_fields,
    ];
    let permutation_alloc_rows = poseidon2_permutation_alloc_rows();
    input_lengths
        .into_iter()
        .map(|input_len| {
            let permutations = input_len.div_ceil(POSEIDON2_RATE) + 1;
            let aux_bits = permutations * permutation_alloc_rows * U64_BITS;
            let rows = permutations * permutation_alloc_rows + POSEIDON2_DIGEST_LEN;
            (aux_bits, rows)
        })
        .fold((0, 0), |(aux_acc, row_acc), (aux, rows)| {
            (aux_acc + aux, row_acc + rows)
        })
}

fn add_direct_state_image_digest_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    source: &DirectCcsFPrimeLowNormSourceImage,
    source_start_col: usize,
    chunk_count_offset: usize,
    step_count_offset: usize,
    current_boundary_digest_offset: usize,
    semantic_accumulator_digest_offset: usize,
    f_prime_accumulator_digest_offset: usize,
    public_trace_digest_offset: usize,
    output_x_offset: usize,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let mut preimage = direct_domain_field_lcs(b"neo.fold.next/direct_ccs/f_prime_x_out/v2");
    push_digest_lanes(
        &mut preimage,
        witness,
        source_start_col + source.vk_fs_digest_bit_offset(),
    )?;
    push_field_lanes(
        &mut preimage,
        witness,
        source_start_col + source.mat_digest_bit_offset(),
        4,
    )?;
    push_u64_halves_lanes(&mut preimage, witness, source_start_col + chunk_count_offset)?;
    push_u64_halves_lanes(&mut preimage, witness, source_start_col + step_count_offset)?;
    push_digest_lanes(
        &mut preimage,
        witness,
        source_start_col + source.initial_boundary_digest_bit_offset(),
    )?;
    push_digest_lanes(
        &mut preimage,
        witness,
        source_start_col + current_boundary_digest_offset,
    )?;
    push_u64_halves_lanes(&mut preimage, witness, source_start_col + source.pc_bit_offset())?;
    push_digest_lanes(
        &mut preimage,
        witness,
        source_start_col + semantic_accumulator_digest_offset,
    )?;
    push_digest_lanes(
        &mut preimage,
        witness,
        source_start_col + f_prime_accumulator_digest_offset,
    )?;
    push_digest_lanes(&mut preimage, witness, source_start_col + public_trace_digest_offset)?;
    let digest = poseidon2_hash_lcs(a_trips, b_trips, c_trips, row, witness, &preimage)?;
    enforce_digest_lanes_equal_source(
        a_trips,
        b_trips,
        row,
        witness,
        &digest,
        source_start_col + output_x_offset,
    )
}

fn add_digest_update_poseidon_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    source: &DirectCcsFPrimeLowNormSourceImage,
    source_start_col: usize,
    domain: &[u8],
    input_digest_offset: usize,
    output_digest_offset: usize,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let mut preimage = direct_domain_field_lcs(domain);
    push_digest_lanes(&mut preimage, witness, source_start_col + input_digest_offset)?;
    push_digest_lanes(
        &mut preimage,
        witness,
        source_start_col + source.latest_chunk_digest_bit_offset(),
    )?;
    let digest = poseidon2_hash_lcs(a_trips, b_trips, c_trips, row, witness, &preimage)?;
    enforce_digest_lanes_equal_source(
        a_trips,
        b_trips,
        row,
        witness,
        &digest,
        source_start_col + output_digest_offset,
    )
}

fn add_construction2_fresh_instance_digest_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &mut Vec<F>,
    source_start_col: usize,
    commitment_digest_offset: usize,
    x_i_offset: usize,
    fresh_digest_offset: usize,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let mut preimage = vec![FieldLc::constant(F::from_u64(CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG))];
    push_digest_lanes(&mut preimage, witness, source_start_col + commitment_digest_offset)?;
    push_digest_lanes(&mut preimage, witness, source_start_col + x_i_offset)?;
    let digest = poseidon2_hash_lcs(a_trips, b_trips, c_trips, row, witness, &preimage)?;
    enforce_digest_lanes_equal_source(
        a_trips,
        b_trips,
        row,
        witness,
        &digest,
        source_start_col + fresh_digest_offset,
    )
}

fn poseidon2_permutation_alloc_rows() -> usize {
    let constants = &*LOW_NORM_POSEIDON2_CONSTANTS;
    let external_layer_rows = 2 * 4 + POSEIDON2_WIDTH;
    let full_round_rows = POSEIDON2_WIDTH * 4 + external_layer_rows;
    let partial_round_rows = 4 + POSEIDON2_WIDTH;
    external_layer_rows
        + (constants.initial.len() + constants.terminal.len()) * full_round_rows
        + constants.internal.len() * partial_round_rows
}

fn direct_domain_field_lcs(domain: &[u8]) -> Vec<FieldLc> {
    crate::superneo_circuit::claim::packed_bytes_field_values(domain)
        .into_iter()
        .map(|value| FieldLc::constant(F::from_u64(value.to_canonical_u64())))
        .collect()
}

fn push_digest_lanes(out: &mut Vec<FieldLc>, witness: &[F], start_col: usize) -> Result<(), DirectCcsFPrimeSnarkError> {
    push_field_lanes(out, witness, start_col, POSEIDON2_DIGEST_LEN)
}

fn push_field_lanes(
    out: &mut Vec<FieldLc>,
    witness: &[F],
    start_col: usize,
    count: usize,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    for idx in 0..count {
        out.push(FieldLc::lane(witness, start_col + idx * U64_BITS)?);
    }
    Ok(())
}

fn push_u64_halves_lanes(
    out: &mut Vec<FieldLc>,
    witness: &[F],
    start_col: usize,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    out.push(FieldLc::bits(witness, start_col, 32)?);
    out.push(FieldLc::bits(witness, start_col + 32, 32)?);
    Ok(())
}

fn poseidon2_hash_lcs(
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

fn enforce_digest_lanes_equal_source(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    witness: &[F],
    digest: &[FieldLc; POSEIDON2_DIGEST_LEN],
    source_digest_start_col: usize,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    for (idx, lane) in digest.iter().enumerate() {
        let source_lane = FieldLc::lane(witness, source_digest_start_col + idx * U64_BITS)?;
        let diff = lane.add_scaled(&source_lane, -F::ONE);
        push_lc_trips(a_trips, *row, &diff);
        b_trips.push((*row, ONE_COL, F::ONE));
        *row += 1;
    }
    Ok(())
}

fn push_lc_trips(trips: &mut Vec<(usize, usize, F)>, row: usize, lc: &FieldLc) {
    if lc.constant != F::ZERO {
        trips.push((row, ONE_COL, lc.constant));
    }
    for (col, coeff) in &lc.terms {
        if *coeff != F::ZERO {
            trips.push((row, *col, *coeff));
        }
    }
}

fn append_field_lane_bits(witness: &mut Vec<F>, value: F) -> usize {
    let start = witness.len();
    let value = value.as_canonical_u64();
    for bit_index in 0..U64_BITS {
        witness.push(F::from_u64((value >> bit_index) & 1));
    }
    start
}

fn field_bits_value(witness: &[F], start_col: usize, bit_len: usize) -> Result<F, DirectCcsFPrimeSnarkError> {
    if start_col
        .checked_add(bit_len)
        .is_none_or(|end| end > witness.len())
    {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "direct F' field lane is outside the witness".into(),
        ));
    }
    let mut value = 0u64;
    for bit_index in 0..bit_len {
        let bit = witness[start_col + bit_index];
        if bit == F::ONE {
            value |= 1u64 << bit_index;
        } else if bit != F::ZERO {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct F' field lane contains a non-binary bit".into(),
            ));
        }
    }
    Ok(F::from_u64(value))
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
