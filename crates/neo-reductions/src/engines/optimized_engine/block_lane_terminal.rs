//! Verifier terminal formula for the production block-by-lane NC channel.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::CeClaim;
use neo_math::{F, K};
use p3_field::{Field, PrimeCharacteristicRing};

use crate::error::PiCcsError;

use super::common::eq_points;
use super::delayed_projection::{beta_power_selector, multilinear_evaluation};
use super::oracle::{BlockLaneNcChallenges, BlockLaneNcPending};

pub(super) fn claimed_initial(challenges: &BlockLaneNcChallenges, pending: Option<&BlockLaneNcPending>) -> K {
    let Some(pending) = pending else {
        return K::ZERO;
    };
    let parent = pending
        .parent_y
        .iter()
        .rev()
        .fold(K::ZERO, |value, coefficient| {
            value * challenges.producer_beta + *coefficient
        });
    challenges.batch_weight * parent
}

pub(super) fn rhs<Ff>(
    challenges: &BlockLaneNcChallenges,
    block_point: &[K],
    lane_point: &[K],
    outputs: &[CeClaim<Cmt, Ff, K>],
    fresh_count: usize,
    pending: Option<&BlockLaneNcPending>,
) -> Result<K, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if fresh_count > outputs.len() {
        return Err(PiCcsError::InvalidInput(
            "block-lane fresh-output count exceeds total outputs".into(),
        ));
    }

    let mut ordinary = K::ZERO;
    let mut gamma_power = K::ONE;
    let mut running = K::ZERO;
    let mut radix_power = K::ONE;
    let radix = K::from(F::from_u64(2));
    for (source, output) in outputs.iter().enumerate() {
        let value = multilinear_evaluation(&output.y_zcol, lane_point)?;
        ordinary += gamma_power * (value * value * value - value);
        gamma_power *= challenges.gamma;
        if source >= fresh_count {
            running += radix_power * value;
            radix_power *= radix;
        }
    }
    ordinary *= eq_points(block_point, &challenges.beta_block) * eq_points(lane_point, &challenges.beta_lane);

    let delayed = pending.map_or(K::ZERO, |pending| {
        challenges.batch_weight
            * eq_points(block_point, &pending.old_block)
            * beta_power_selector(challenges.producer_beta, lane_point)
            * running
    });
    Ok(ordinary + delayed)
}
