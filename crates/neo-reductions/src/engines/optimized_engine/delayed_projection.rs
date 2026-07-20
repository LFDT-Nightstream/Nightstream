//! Delayed old-point `y_zcol` authority for Split-NC Π_CCS.
//!
//! Owns the compact public parent view, its two domain-separated challenges,
//! and the exact initial/terminal algebra added to the existing NC SumCheck.
//! It does not own transcript scheduling, witness-table folding, or primitive
//! security.

use crate::error::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::CeClaim;
use neo_math::{D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};

/// Public old-point data fixed before the delayed-projection challenges.
#[derive(Clone, Copy, Debug)]
pub struct DelayedProjectionInput<'a> {
    pub s_col: &'a [K],
    pub y_zcol: &'a [K],
}

/// Verifier-derived challenges for the delayed NC summand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelayedProjectionChallenges {
    pub producer_beta: K,
    pub batch_weight: K,
}

/// Fully sampled configuration consumed by the prover oracle.
#[derive(Clone, Copy, Debug)]
pub struct DelayedProjectionConfig<'a> {
    pub input: DelayedProjectionInput<'a>,
    pub challenges: DelayedProjectionChallenges,
}

pub fn validate_input(input: DelayedProjectionInput<'_>, ell_m: usize, d_pad: usize) -> Result<(), PiCcsError> {
    if input.s_col.len() != ell_m {
        return Err(PiCcsError::InvalidInput(format!(
            "delayed projection s_col length mismatch: expected {ell_m}, got {}",
            input.s_col.len()
        )));
    }
    if input.y_zcol.len() != d_pad {
        return Err(PiCcsError::InvalidInput(format!(
            "delayed projection y_zcol length mismatch: expected {d_pad}, got {}",
            input.y_zcol.len()
        )));
    }
    if input.y_zcol.iter().skip(D).any(|&value| value != K::ZERO) {
        return Err(PiCcsError::InvalidInput(
            "delayed projection y_zcol padding lanes must be zero".into(),
        ));
    }
    Ok(())
}

/// Constant-first evaluation of the 54 active parent coefficients.
pub fn parent_evaluation(input: DelayedProjectionInput<'_>, producer_beta: K) -> K {
    input
        .y_zcol
        .iter()
        .take(D)
        .rev()
        .fold(K::ZERO, |acc, &coefficient| acc * producer_beta + coefficient)
}

pub fn claimed_initial_sum(config: DelayedProjectionConfig<'_>) -> K {
    config.challenges.batch_weight * parent_evaluation(config.input, config.challenges.producer_beta)
}

/// Multilinear extension of the Boolean table `lane ↦ beta^lane`.
pub fn beta_power_selector(producer_beta: K, alpha: &[K]) -> K {
    let mut beta_pow = producer_beta;
    let mut result = K::ONE;
    for &coordinate in alpha {
        result *= (K::ONE - coordinate) + coordinate * beta_pow;
        beta_pow *= beta_pow;
    }
    result
}

pub fn equality_evaluation(left: &[K], right: &[K]) -> Result<K, PiCcsError> {
    if left.len() != right.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "delayed projection equality-point length mismatch: {} != {}",
            left.len(),
            right.len()
        )));
    }
    Ok(left
        .iter()
        .zip(right)
        .fold(K::ONE, |acc, (&l, &r)| acc * (l * r + (K::ONE - l) * (K::ONE - r))))
}

pub fn multilinear_evaluation(values: &[K], point: &[K]) -> Result<K, PiCcsError> {
    let domain = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("delayed projection MLE domain overflow".into()))?;
    if values.len() < domain {
        return Err(PiCcsError::InvalidInput(format!(
            "delayed projection coefficient vector too short: expected at least {domain}, got {}",
            values.len()
        )));
    }
    let mut result = K::ZERO;
    for (index, &value) in values.iter().take(domain).enumerate() {
        let mut weight = K::ONE;
        for (bit, &coordinate) in point.iter().enumerate() {
            weight *= if ((index >> bit) & 1) == 1 {
                coordinate
            } else {
                K::ONE - coordinate
            };
        }
        result += value * weight;
    }
    Ok(result)
}

/// Terminal evaluation of the radix-recomposed running-child assignments.
///
/// Π_CCS outputs are ordered `fresh || running`; only the running suffix is
/// part of the prior Π_DEC decomposition.
pub fn running_output_evaluation<Ff>(
    outputs: &[CeClaim<Cmt, Ff, K>],
    k_mcs: usize,
    alpha: &[K],
    radix: u32,
) -> Result<K, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if k_mcs > outputs.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "delayed projection fresh-output count {k_mcs} exceeds total {}",
            outputs.len()
        )));
    }
    let radix = K::from(F::from_u64(radix as u64));
    let mut radix_power = K::ONE;
    let mut result = K::ZERO;
    for output in outputs.iter().skip(k_mcs) {
        result += radix_power * multilinear_evaluation(&output.y_zcol, alpha)?;
        radix_power *= radix;
    }
    Ok(result)
}

pub fn terminal_rhs<Ff>(
    config: DelayedProjectionConfig<'_>,
    terminal_s: &[K],
    terminal_alpha: &[K],
    outputs: &[CeClaim<Cmt, Ff, K>],
    k_mcs: usize,
    radix: u32,
) -> Result<K, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let point_equality = equality_evaluation(terminal_s, config.input.s_col)?;
    let lane_selector = beta_power_selector(config.challenges.producer_beta, terminal_alpha);
    let raw_evaluation = running_output_evaluation(outputs, k_mcs, terminal_alpha, radix)?;
    Ok(config.challenges.batch_weight * point_equality * lane_selector * raw_evaluation)
}
