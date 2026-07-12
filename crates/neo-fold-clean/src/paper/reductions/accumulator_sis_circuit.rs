//! Domain-separated SIS/Ajtai binding compression for fixed field sequences.
//!
//! Inputs reuse the relation's 41-digit balanced-ternary encoding, pack it
//! row-major into an Ajtai message matrix, and commit under a domain-specific
//! rank-2 seeded map. An independent rank-1 map compresses that short
//! commitment envelope before one Poseidon2 digest enters Fiat–Shamir.
//! Carried accumulator chains remain outside this module.

use neo_ajtai::{commit_row_major_seeded, seeded_pp_chunk_seeds, Commitment};
use neo_ccs::{Mat, SeededPhi81LinearBlock};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::r1cs_circuit::builder::BALANCED_TERNARY_DIGITS;
use crate::engine::r1cs_circuit::{enforce_poseidon2_hash, Lc, R1csBuilder, Var};
use crate::paper::digest::pack_bytes_as_fields;
use crate::paper::relations::product_commitment_circuit::{alloc_commitment, CommitmentWires};

const SIS_ACCUMULATOR_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/accumulator/sis/digest/v4";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SisAccumulatorConfig {
    pub seed: [u8; 32],
    pub kappa: usize,
    pub domain: u64,
}

/// Minimum rank for protocol-binding compression maps. At the R7 maximum
/// message width, rank 1 estimates at 59.9 rough bits; rank 2 estimates at
/// 167.0 bits. See `scripts/estimate_nebula_sis.sage` for the pinned model.
pub const PROTOCOL_BINDING_KAPPA: usize = 2;

/// Independent short-message map used between the rank-2 binding and
/// Poseidon2. Its input is at most one rank-2 commitment plus metadata, not
/// the original witness-proportional sequence.
pub const SIS_DIGEST_COMPRESSION_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC6; 32],
    kappa: 1,
    domain: 0x5349_535F_4447_5354,
};

pub const CCS_CLAIM_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC1; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x4343_535F_434C_4149,
};

pub const CE_CLAIM_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC2; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x4345_5F43_4C41_494D,
};

pub const PI_CCS_OUTPUTS_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC3; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x5049_4343_535F_4F55,
};

pub const PI_RLC_PROJECTION_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC4; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x5049_524C_435F_5052,
};

pub const NEBULA_LEAF_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC5; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x4E42_4C41_5F4C_4546,
};

#[derive(Debug, Error)]
pub enum SisAccumulatorError {
    #[error("SIS accumulator requires at least one input field")]
    EmptyInput,
    #[error("SIS accumulator kappa must be nonzero")]
    ZeroKappa,
}

pub struct SisAccumulatorWires {
    pub commitment: CommitmentWires,
    pub digest_compression: CommitmentWires,
    pub digest: [Var; 4],
}

pub fn accumulator_digest(config: SisAccumulatorConfig, fields: &[F]) -> Result<[F; 4], SisAccumulatorError> {
    let binding = commit_fields(config, fields)?;
    let digest_compression = commit_fields(SIS_DIGEST_COMPRESSION_CONFIG, &binding.data)?;
    Ok(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&digest_envelope(
        config,
        fields.len(),
        &binding,
        &digest_compression,
    )))
}

pub fn commit_fields(config: SisAccumulatorConfig, fields: &[F]) -> Result<Commitment, SisAccumulatorError> {
    validate(config, fields.len())?;
    let message = balanced_ternary_message(fields);
    Ok(commit_row_major_seeded(
        config.seed,
        D,
        config.kappa,
        message.cols(),
        &message,
    ))
}

pub fn enforce_commit_fields(
    builder: &mut R1csBuilder,
    config: SisAccumulatorConfig,
    fields: &[Var],
) -> Result<CommitmentWires, SisAccumulatorError> {
    validate(config, fields.len())?;
    let values: Vec<F> = fields
        .iter()
        .map(|field| builder.witness()[field.col()])
        .collect();
    let native = commit_fields(config, &values)?;
    let commitment = alloc_commitment(builder, &native);
    let digit_words: Vec<[Var; BALANCED_TERNARY_DIGITS]> = fields
        .iter()
        .map(|&field| decompose_var_to_balanced_ternary(builder, field))
        .collect();
    let word_starts = digit_words
        .iter()
        .map(|digits| {
            let start = digits[0].col();
            assert!(digits
                .iter()
                .enumerate()
                .all(|(digit, var)| var.col() == start + digit));
            start
        })
        .collect::<Vec<_>>();
    let message_cols = (fields.len() * BALANCED_TERNARY_DIGITS).div_ceil(D);
    let (chunk_size, chunk_seeds) = seeded_pp_chunk_seeds(config.seed, config.kappa, message_cols);
    let block = SeededPhi81LinearBlock::new_with_word_width(
        builder.rows(),
        word_starts,
        BALANCED_TERNARY_DIGITS,
        config.kappa,
        message_cols,
        chunk_size,
        chunk_seeds,
    )
    .expect("fixed seeded SIS geometry");
    builder.enforce_seeded_phi81_a_block(block, &commitment.data);
    Ok(commitment)
}

pub fn enforce_accumulator_digest(
    builder: &mut R1csBuilder,
    config: SisAccumulatorConfig,
    fields: &[Var],
) -> Result<SisAccumulatorWires, SisAccumulatorError> {
    let commitment = enforce_commit_fields(builder, config, fields)?;
    let digest_compression = enforce_commit_fields(builder, SIS_DIGEST_COMPRESSION_CONFIG, &commitment.data)?;
    let digest_preimage = digest_envelope_wires(builder, config, fields.len(), &commitment, &digest_compression);
    let digest = enforce_poseidon2_hash(builder, &digest_preimage);
    Ok(SisAccumulatorWires {
        commitment,
        digest_compression,
        digest,
    })
}

fn digest_envelope(
    config: SisAccumulatorConfig,
    field_count: usize,
    binding: &Commitment,
    digest_compression: &Commitment,
) -> Vec<F> {
    let mut envelope = pack_bytes_as_fields(SIS_ACCUMULATOR_DIGEST_DOMAIN);
    envelope.push(F::from_u64(config.domain));
    envelope.push(F::from_u64(field_count as u64));
    envelope.push(F::from_u64(binding.kappa as u64));
    envelope.extend_from_slice(&digest_compression.data);
    envelope
}

fn digest_envelope_wires(
    builder: &mut R1csBuilder,
    config: SisAccumulatorConfig,
    field_count: usize,
    binding: &CommitmentWires,
    digest_compression: &CommitmentWires,
) -> Vec<Var> {
    let mut envelope = alloc_constant_fields(builder, SIS_ACCUMULATOR_DIGEST_DOMAIN);
    envelope.push(alloc_constant(builder, F::from_u64(config.domain)));
    envelope.push(alloc_constant(builder, F::from_u64(field_count as u64)));
    envelope.push(alloc_constant(builder, F::from_u64(binding.kappa as u64)));
    envelope.extend_from_slice(&digest_compression.data);
    envelope
}

fn alloc_constant_fields(builder: &mut R1csBuilder, domain: &[u8]) -> Vec<Var> {
    pack_bytes_as_fields(domain)
        .into_iter()
        .map(|value| alloc_constant(builder, value))
        .collect()
}

fn balanced_ternary_message(fields: &[F]) -> Mat<F> {
    let message_cols = (fields.len() * BALANCED_TERNARY_DIGITS).div_ceil(D);
    let mut message = Mat::zero(D, message_cols, F::ZERO);
    for (field_index, field) in fields.iter().enumerate() {
        for (digit, value) in balanced_ternary_digits(*field).into_iter().enumerate() {
            let index = field_index * BALANCED_TERNARY_DIGITS + digit;
            message[(index / message_cols, index % message_cols)] = value;
        }
    }
    message
}

fn decompose_var_to_balanced_ternary(builder: &mut R1csBuilder, field: Var) -> [Var; BALANCED_TERNARY_DIGITS] {
    if let Some(digits) = builder.balanced_ternary_decomposition(field) {
        return digits;
    }
    let values = balanced_ternary_digits(builder.witness()[field.col()]);
    let digits = values.map(|value| builder.alloc(value));
    let negative_indicators = digits.map(|digit| {
        let value = builder.witness()[digit.col()];
        let is_negative = value == -F::ONE;
        debug_assert!(is_negative || value == F::ZERO || value == F::ONE);
        let negative = builder.alloc(if is_negative { F::ONE } else { F::ZERO });
        builder.record_boolean(negative);

        // `q = d(d-1)/2` is the exact indicator for `d = -1` on the
        // centered alphabet. The second row completes `d(d-1)(d+1) = 0`.
        let mut minus_one = Lc::from_var(digit);
        minus_one.add_constant(-F::ONE);
        let twice_negative = Lc::zero().add_scaled(&Lc::from_var(negative), F::from_u64(2));
        builder.enforce(&Lc::from_var(digit), &minus_one, &twice_negative);
        let mut plus_one = Lc::from_var(digit);
        plus_one.add_constant(F::ONE);
        builder.enforce(&Lc::from_var(negative), &plus_one, &Lc::zero());
        builder.record_centered_unit(digit);
        negative
    });
    let mut reconstruction = Lc::zero();
    let mut power = F::ONE;
    for digit in digits {
        reconstruction.add_term(digit, power);
        power *= F::from_u64(3);
    }
    builder.enforce_eq(&Lc::from_var(field), &reconstruction);
    enforce_shifted_base3_canonical(builder, &digits, &negative_indicators);
    builder.record_balanced_ternary_decomposition(field, digits);
    digits
}

/// Select one low-norm opening for every field residue without a 64-bit
/// comparator. Let `M = (3^41-1)/2` and `t_i = d_i+1 ∈ {0,1,2}`. Native
/// encoding chooses the ordinary base-3 expansion
///
/// `N = (x + M) mod p = sum_i t_i 3^i`, with `0 <= N < p`.
///
/// Reconstruction already proves `x = sum_i d_i 3^i (mod p)`. The borrow
/// chain below proves `N <= p-1`, so ordinary base-3 uniqueness rules out the
/// alternative `N+p` opening. Each recurrence also inductively pins its next
/// borrow to a bit; the recorded Boolean width is therefore derived, not
/// trusted as a protocol conclusion.
fn enforce_shifted_base3_canonical(
    builder: &mut R1csBuilder,
    digits: &[Var; BALANCED_TERNARY_DIGITS],
    negative_indicators: &[Var; BALANCED_TERNARY_DIGITS],
) {
    let mut bound = F::ORDER_U64 - 1;
    let mut borrow_value = false;
    let mut borrow = Lc::zero();

    for index in 0..BALANCED_TERNARY_DIGITS {
        let bound_digit = bound % 3;
        bound /= 3;

        let digit_value = builder.witness()[digits[index].col()];
        let trit = if digit_value == -F::ONE {
            0
        } else if digit_value == F::ZERO {
            1
        } else {
            debug_assert_eq!(digit_value, F::ONE);
            2
        };
        let next_borrow_value = trit + u64::from(borrow_value) > bound_digit;
        let next_borrow = if index + 1 == BALANCED_TERNARY_DIGITS {
            debug_assert!(!next_borrow_value, "honest shifted-base-3 opening must be below p");
            Lc::zero()
        } else {
            let next = builder.alloc(if next_borrow_value { F::ONE } else { F::ZERO });
            builder.record_boolean(next);
            Lc::from_var(next)
        };

        let negative = Lc::from_var(negative_indicators[index]);
        let positive = Lc::from_var(digits[index]).add_scaled(&negative, F::ONE);
        let zero = Lc::from_const(F::ONE)
            .add_scaled(&Lc::from_var(digits[index]), -F::ONE)
            .add_scaled(&negative, -F::from_u64(2));

        match bound_digit {
            // b' = 1 unless t=0 and b=0.
            0 => {
                let one_minus_borrow = Lc::from_const(F::ONE).add_scaled(&borrow, -F::ONE);
                let one_minus_next = Lc::from_const(F::ONE).add_scaled(&next_borrow, -F::ONE);
                builder.enforce(&negative, &one_minus_borrow, &one_minus_next);
            }
            // b' = isPositive(t) + isZero(t) * b.
            1 => {
                let rhs = next_borrow.clone().add_scaled(&positive, -F::ONE);
                builder.enforce(&zero, &borrow, &rhs);
            }
            // b' = isPositive(t) * b.
            2 => builder.enforce(&positive, &borrow, &next_borrow),
            _ => unreachable!("base-3 digit"),
        }

        borrow = next_borrow;
        borrow_value = next_borrow_value;
    }
    debug_assert_eq!(bound, 0, "41 base-3 digits must cover p-1");
}

fn balanced_ternary_digits(value: F) -> [F; BALANCED_TERNARY_DIGITS] {
    let modulus = F::ORDER_U64 as u128;
    let shift = (3u128.pow(BALANCED_TERNARY_DIGITS as u32) - 1) / 2;
    let mut remaining = (value.as_canonical_u64() as u128 + shift) % modulus;
    let digits = core::array::from_fn(|_| {
        let trit = remaining % 3;
        remaining /= 3;
        match trit {
            0 => -F::ONE,
            1 => F::ZERO,
            2 => F::ONE,
            _ => unreachable!("base-3 digit"),
        }
    });
    assert_eq!(remaining, 0, "shifted Goldilocks residue must fit in 41 base-3 digits");
    digits
}

fn validate(config: SisAccumulatorConfig, field_count: usize) -> Result<(), SisAccumulatorError> {
    if field_count == 0 {
        return Err(SisAccumulatorError::EmptyInput);
    }
    if config.kappa == 0 {
        return Err(SisAccumulatorError::ZeroKappa);
    }
    Ok(())
}

fn alloc_constant(builder: &mut R1csBuilder, value: F) -> Var {
    let wire = builder.alloc(value);
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(value));
    wire
}
