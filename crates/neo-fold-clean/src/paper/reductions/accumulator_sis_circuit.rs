//! Candidate C14 binding compression for a fixed field sequence.
//!
//! Inputs are canonically decomposed into bits, packed row-major into an
//! Ajtai message matrix, and committed under a domain-specific seeded map.
//! This module does not replace the running-accumulator handle yet; it owns
//! only the native/circuit primitive needed to measure that proposal.

use neo_ajtai::{commit_row_major, precompute_rot_columns, setup_par, Commitment, PP};
use neo_ccs::Mat;
use neo_math::ring::Rq;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::engine::r1cs_circuit::{enforce_poseidon2_hash, Lc, R1csBuilder, Var};
use crate::paper::digest::pack_bytes_as_fields;
use crate::paper::relations::product_commitment_circuit::{alloc_commitment, CommitmentWires};

const SIS_ACCUMULATOR_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/accumulator/sis/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SisAccumulatorConfig {
    pub seed: [u8; 32],
    pub kappa: usize,
}

#[derive(Debug, Error)]
pub enum SisAccumulatorError {
    #[error("SIS accumulator requires at least one input field")]
    EmptyInput,
    #[error("SIS accumulator kappa must be nonzero")]
    ZeroKappa,
    #[error("SIS accumulator setup failed: {0}")]
    Setup(String),
}

pub struct SisAccumulatorWires {
    pub commitment: CommitmentWires,
    pub digest: [Var; 4],
}

pub fn accumulator_digest(config: SisAccumulatorConfig, fields: &[F]) -> Result<[F; 4], SisAccumulatorError> {
    let commitment = commit_fields(config, fields)?;
    let mut preimage = pack_bytes_as_fields(SIS_ACCUMULATOR_DIGEST_DOMAIN);
    preimage.push(F::from_u64(commitment.d as u64));
    preimage.push(F::from_u64(commitment.kappa as u64));
    preimage.push(F::from_u64(commitment.data.len() as u64));
    preimage.extend_from_slice(&commitment.data);
    Ok(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub fn commit_fields(config: SisAccumulatorConfig, fields: &[F]) -> Result<Commitment, SisAccumulatorError> {
    let (message, pp) = message_and_pp(config, fields)?;
    Ok(commit_row_major(&pp, &message))
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
    let (message, pp) = message_and_pp(config, &values)?;
    let native = commit_row_major(&pp, &message);
    let commitment = alloc_commitment(builder, &native);
    let bits: Vec<Var> = fields
        .iter()
        .flat_map(|&field| decompose_var_to_u64_bits(builder, field))
        .collect();
    let message_cols = bits.len().div_ceil(D);

    for (commit_col, pp_row) in pp.m_rows.iter().enumerate() {
        let mut outputs = vec![Lc::zero(); D];
        for (message_col, &ring_element) in pp_row.iter().enumerate() {
            let mut rotations = [[F::ZERO; D]; D];
            precompute_rot_columns(ring_element, &mut rotations);
            for message_row in 0..D {
                let bit_index = message_row * message_cols + message_col;
                if let Some(&bit) = bits.get(bit_index) {
                    for coord_row in 0..D {
                        let coefficient = rotations[message_row][coord_row];
                        if coefficient != F::ZERO {
                            outputs[coord_row].add_term(bit, coefficient);
                        }
                    }
                }
            }
        }
        for (coord_row, lhs) in outputs.iter().enumerate() {
            builder.enforce_eq(&lhs, &Lc::from_var(commitment.data[commit_col * D + coord_row]));
        }
    }
    Ok(commitment)
}

pub fn enforce_accumulator_digest(
    builder: &mut R1csBuilder,
    config: SisAccumulatorConfig,
    fields: &[Var],
) -> Result<SisAccumulatorWires, SisAccumulatorError> {
    let commitment = enforce_commit_fields(builder, config, fields)?;
    let mut preimage = pack_bytes_as_fields(SIS_ACCUMULATOR_DIGEST_DOMAIN)
        .into_iter()
        .map(|value| alloc_constant(builder, value))
        .collect::<Vec<_>>();
    preimage.push(alloc_constant(builder, F::from_u64(commitment.d as u64)));
    preimage.push(alloc_constant(builder, F::from_u64(commitment.kappa as u64)));
    preimage.push(alloc_constant(builder, F::from_u64(commitment.data.len() as u64)));
    preimage.extend_from_slice(&commitment.data);
    let digest = enforce_poseidon2_hash(builder, &preimage);
    Ok(SisAccumulatorWires { commitment, digest })
}

fn message_and_pp(config: SisAccumulatorConfig, fields: &[F]) -> Result<(Mat<F>, PP<Rq>), SisAccumulatorError> {
    validate(config, fields.len())?;
    let bit_count = fields.len() * u64::BITS as usize;
    let message_cols = bit_count.div_ceil(D);
    let mut message = Mat::zero(D, message_cols, F::ZERO);
    for (field_index, field) in fields.iter().enumerate() {
        let value = field.as_canonical_u64();
        for bit in 0..u64::BITS as usize {
            let index = field_index * u64::BITS as usize + bit;
            message[(index / message_cols, index % message_cols)] = F::from_u64((value >> bit) & 1);
        }
    }
    let pp = seeded_pp(config, message_cols)?;
    Ok((message, pp))
}

fn seeded_pp(config: SisAccumulatorConfig, message_cols: usize) -> Result<PP<Rq>, SisAccumulatorError> {
    let mut rng = ChaCha8Rng::from_seed(config.seed);
    setup_par(&mut rng, D, config.kappa, message_cols).map_err(|error| SisAccumulatorError::Setup(error.to_string()))
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
