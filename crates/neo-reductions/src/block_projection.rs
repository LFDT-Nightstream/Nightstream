//! Raw packed-witness projection at the production 19-bit block point.
//!
//! This module reads `Mat<D×ceil(m/D)>` entries directly. It does not consume
//! `CeClaim::y_zcol`, a digest, or any other prover-carried projection value.

use neo_ccs::utils::tensor_point_parallel;
use neo_ccs::Mat;
use neo_math::{D, K};
use p3_field::{Field, PrimeCharacteristicRing};

use crate::common::validate_superneo_witness_mat;
use crate::error::PiCcsError;

/// The fixed production block domain has `2^19` leaves.
pub const BLOCK_PROJECTION_POINT_LEN: usize = 19;

/// Number of leaves in the fixed production block domain.
pub const BLOCK_PROJECTION_DOMAIN_SIZE: usize = 1usize << BLOCK_PROJECTION_POINT_LEN;

/// Project one raw packed witness to its 54 active lane values at a fixed
/// 19-coordinate block point.
///
/// For lane `rho`, the result is
/// `Σ_block χ_point[block] · K::from(witness[(rho, block)])`.
/// Packed blocks outside `ceil(expected_m / D)` are verifier-computed zeros.
pub fn project_raw_witness_at_block_point<Ff>(
    witness: &Mat<Ff>,
    expected_m: usize,
    block_point: &[K; BLOCK_PROJECTION_POINT_LEN],
) -> Result<[K; D], PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let mut projected = project_raw_witnesses_at_block_point(core::slice::from_ref(witness), expected_m, block_point)?;
    Ok(projected
        .pop()
        .expect("one input witness produces one projected row"))
}

/// Batch form of [`project_raw_witness_at_block_point`].
///
/// The equality weights are expanded once and shared across every witness,
/// which is the terminal verifier's natural ordered-child input shape.
pub fn project_raw_witnesses_at_block_point<Ff>(
    witnesses: &[Mat<Ff>],
    expected_m: usize,
    block_point: &[K; BLOCK_PROJECTION_POINT_LEN],
) -> Result<Vec<[K; D]>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let active_blocks = expected_m.div_ceil(D);
    if active_blocks > BLOCK_PROJECTION_DOMAIN_SIZE {
        return Err(PiCcsError::InvalidInput(format!(
            "block projection needs {active_blocks} packed blocks, exceeding the fixed {BLOCK_PROJECTION_DOMAIN_SIZE}-block domain"
        )));
    }
    for witness in witnesses {
        validate_superneo_witness_mat(witness, expected_m)?;
    }

    let weights = tensor_point_parallel::<K>(block_point);
    let projected = witnesses
        .iter()
        .map(|witness| {
            let mut lanes = [K::ZERO; D];
            for lane in 0..D {
                let mut value = K::ZERO;
                for block in 0..active_blocks {
                    value += K::from(witness[(lane, block)]) * weights[block];
                }
                lanes[lane] = value;
            }
            lanes
        })
        .collect();
    Ok(projected)
}

/// Project ordered raw children and recombine them with powers of `radix`.
///
/// Child order is semantic: child `i` owns coefficient `radix^i`. This is the
/// exact terminal form of the one-fold delayed Π_DEC projection check.
pub fn radix_recompose_raw_witnesses_at_block_point<Ff>(
    witnesses: &[Mat<Ff>],
    expected_m: usize,
    block_point: &[K; BLOCK_PROJECTION_POINT_LEN],
    radix: K,
) -> Result<[K; D], PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let projected = project_raw_witnesses_at_block_point(witnesses, expected_m, block_point)?;
    let mut power = K::ONE;
    let mut recomposed = [K::ZERO; D];
    for child in projected {
        for lane in 0..D {
            recomposed[lane] += child[lane] * power;
        }
        power *= radix;
    }
    Ok(recomposed)
}
