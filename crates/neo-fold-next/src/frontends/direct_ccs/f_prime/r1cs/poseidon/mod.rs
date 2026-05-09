//! Owns Poseidon2 linkage constraints for the compact direct F' source shell.

mod permutation;

use permutation::{poseidon2_hash_lcs, poseidon2_permutation_alloc_rows};

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::super::super::state::DirectCcsFPrimeSnarkError;
use super::super::DirectCcsFPrimeLowNormSourceImage;
use crate::construction2::CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG;

const ONE_COL: usize = 0;
const U64_BITS: usize = 64;
const POSEIDON2_WIDTH: usize = neo_params::poseidon2_goldilocks::WIDTH;
const POSEIDON2_RATE: usize = neo_params::poseidon2_goldilocks::RATE;
const POSEIDON2_DIGEST_LEN: usize = neo_params::poseidon2_goldilocks::DIGEST_LEN;
const POSEIDON2_SBOX_DEGREE: u64 = 7;

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
    )?;
    add_construction2_fresh_instance_digest_constraints(
        a_trips,
        b_trips,
        c_trips,
        row,
        witness,
        source_start_col,
        source.construction2_u_out_commitment_digest_bit_offset(),
        source.construction2_u_out_x_i_bit_offset(),
        source.construction2_u_out_fresh_digest_bit_offset(),
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
