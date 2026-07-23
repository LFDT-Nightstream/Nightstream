//! Value-level helpers for the delayed combined-NC execution replay.
//!
//! Owns: fixed output-table decoding, SumCheck prolog binding, and the exact
//! combined-NC terminal right-hand side.
//!
//! Does not own: transcript scheduling, raw old-block witness authority,
//! generated columns, commitment binding, or semantic acceptance.
//!
//! Emits constraints: no; these helpers replay native values only.
//!
//! | Stable stage path | Obligation | Authority class |
//! |---|---|---|
//! | `f_prime.pi_ccs_nc.output_tables` | Split 54 active and ten checked-zero output lanes | checked |
//! | `f_prime.pi_ccs_nc.terminal_rhs` | Evaluate the ordinary and delayed terminal summands | computed |

use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;

use super::{ACTIVE_RECURSIVE_FRESH_OUTPUTS, OUTPUT_Y_ZCOL_PADDED_LANES, OUTPUT_Y_ZCOL_ZERO_PADDING_LANES};
use crate::paper::reductions::pi_ccs_output_message::Profile as PiCcsOutputProfile;
use crate::paper::relations::CeClaim;

pub(super) fn capture_output_y_zcol_tables(
    outputs: &[CeClaim],
    fresh_count: usize,
    running_count: usize,
    profile: PiCcsOutputProfile,
) -> Result<(Vec<[K; D]>, Vec<[K; OUTPUT_Y_ZCOL_ZERO_PADDING_LANES]>), String> {
    if profile.lane_count() != D
        || D + OUTPUT_Y_ZCOL_ZERO_PADDING_LANES != OUTPUT_Y_ZCOL_PADDED_LANES
        || fresh_count != ACTIVE_RECURSIVE_FRESH_OUTPUTS
        || outputs.len() != profile.source_count()
        || fresh_count + running_count != outputs.len()
    {
        return Err(format!(
            "active output-y_zcol profile drift: profile_sources={} profile_lanes={} fresh={fresh_count} running={running_count} outputs={} padded_lanes={OUTPUT_Y_ZCOL_PADDED_LANES}",
            profile.source_count(),
            profile.lane_count(),
            outputs.len(),
        ));
    }

    let mut active = Vec::with_capacity(outputs.len());
    let mut zero_padding = Vec::with_capacity(outputs.len());
    for (source, output) in outputs.iter().enumerate() {
        if output.y_zcol.len() != OUTPUT_Y_ZCOL_PADDED_LANES {
            return Err(format!(
                "PiCCS output {source} y_zcol has {} lanes, expected exactly {D} active plus {OUTPUT_Y_ZCOL_ZERO_PADDING_LANES} zero padding",
                output.y_zcol.len(),
            ));
        }
        let active_table = core::array::from_fn(|lane| output.y_zcol[lane]);
        let padding_table = core::array::from_fn(|lane| output.y_zcol[D + lane]);
        if let Some((lane, _)) = padding_table
            .iter()
            .enumerate()
            .find(|(_, value)| **value != K::ZERO)
        {
            return Err(format!(
                "PiCCS output {source} y_zcol padding lane {} is nonzero",
                D + lane
            ));
        }
        active.push(active_table);
        zero_padding.push(padding_table);
    }
    Ok((active, zero_padding))
}

pub(super) fn append_sumcheck_prolog(
    transcript: &mut neo_transcript::Poseidon2Transcript,
    channel_tag: u64,
    initial_tag: u64,
    version_tag: u64,
    initial: K,
) {
    transcript.append_fields_raw(&[F::from_u64(channel_tag)]);
    transcript.append_fields_raw(&[F::from_u64(initial_tag)]);
    transcript.append_fields_raw(&initial.as_coeffs());
    transcript.append_fields_raw(&[F::from_u64(version_tag)]);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn combined_nc_terminal_rhs(
    outputs: &[CeClaim],
    fresh_count: usize,
    gamma: K,
    beta_lane: &[K],
    beta_block: &[K],
    producer_beta: K,
    batch_weight: K,
    pending_old_block: &[K],
    block_point: &[K],
    lane_point: &[K],
    eq_points: fn(&[K], &[K]) -> K,
    beta_power_selector: fn(K, &[K]) -> K,
) -> Result<K, String> {
    if fresh_count > outputs.len() {
        return Err("combined-NC output list is shorter than the fresh prefix".into());
    }
    if block_point.len() != beta_block.len()
        || block_point.len() != pending_old_block.len()
        || lane_point.len() != beta_lane.len()
    {
        return Err("combined-NC terminal point shape drift".into());
    }

    let mut ordinary = K::ZERO;
    let mut gamma_power = K::ONE;
    let mut running_evaluation = K::ZERO;
    let mut radix_power = K::ONE;
    let radix = K::from(F::from_u64(2));
    for (source, output) in outputs.iter().enumerate() {
        let value = multilinear_evaluation(&output.y_zcol, lane_point)?;
        ordinary += gamma_power * (value * value * value - value);
        gamma_power *= gamma;
        if source >= fresh_count {
            running_evaluation += radix_power * value;
            radix_power *= radix;
        }
    }
    ordinary *= eq_points(block_point, beta_block) * eq_points(lane_point, beta_lane);
    let delayed = batch_weight
        * eq_points(block_point, pending_old_block)
        * beta_power_selector(producer_beta, lane_point)
        * running_evaluation;
    Ok(ordinary + delayed)
}

fn multilinear_evaluation(values: &[K], point: &[K]) -> Result<K, String> {
    let domain = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| "combined-NC lane domain overflow".to_string())?;
    if values.len() != domain {
        return Err(format!(
            "combined-NC y_zcol has {} values, expected exactly {domain}",
            values.len()
        ));
    }
    let mut result = K::ZERO;
    for (index, &value) in values.iter().take(domain).enumerate() {
        let mut weight = K::ONE;
        for (bit, &coordinate) in point.iter().enumerate() {
            weight *= if (index >> bit) & 1 == 1 {
                coordinate
            } else {
                K::ONE - coordinate
            };
        }
        result += value * weight;
    }
    Ok(result)
}
