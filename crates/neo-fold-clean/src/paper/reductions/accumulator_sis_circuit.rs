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

use crate::engine::r1cs_circuit::builder::{CenteredUnitTrace, BALANCED_TERNARY_DIGITS};
use crate::engine::r1cs_circuit::{enforce_poseidon2_hash, Lc, R1csBuilder, Var};
use crate::paper::digest::pack_bytes_as_fields;
use crate::paper::relations::product_commitment_circuit::{alloc_commitment, CommitmentWires};

const SIS_ACCUMULATOR_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/accumulator/sis/digest/v3";

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
    for digit in digits {
        let row_start = builder.rows();
        let column_start = builder.cols();
        let mut minus_one = Lc::from_var(digit);
        minus_one.add_constant(-F::ONE);
        let product = builder.alloc_mul(&Lc::from_var(digit), &minus_one);
        let mut plus_one = Lc::from_var(digit);
        plus_one.add_constant(F::ONE);
        builder.enforce(&Lc::from_var(product), &plus_one, &Lc::zero());
        builder.record_centered_unit_trace(CenteredUnitTrace {
            row_start,
            row_end: builder.rows(),
            allocated_columns: (column_start..builder.cols()).collect(),
            value_col: digit.col(),
        });
    }
    let mut reconstruction = Lc::zero();
    let mut power = F::ONE;
    for digit in digits {
        reconstruction.add_term(digit, power);
        power *= F::from_u64(3);
    }
    builder.enforce_eq(&Lc::from_var(field), &reconstruction);
    builder.record_balanced_ternary_decomposition(field, digits);
    digits
}

fn balanced_ternary_digits(value: F) -> [F; BALANCED_TERNARY_DIGITS] {
    let modulus = F::ORDER_U64 as i128;
    let canonical = value.as_canonical_u64() as i128;
    let mut remaining = if canonical <= modulus / 2 {
        canonical
    } else {
        canonical - modulus
    };
    let digits = core::array::from_fn(|_| {
        let residue = remaining.rem_euclid(3);
        let digit = if residue == 2 { -1i128 } else { residue };
        remaining = (remaining - digit) / 3;
        match digit {
            -1 => -F::ONE,
            0 => F::ZERO,
            1 => F::ONE,
            _ => unreachable!("balanced ternary digit is centered"),
        }
    });
    assert_eq!(remaining, 0, "Goldilocks centered representative fits in 41 trits");
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
