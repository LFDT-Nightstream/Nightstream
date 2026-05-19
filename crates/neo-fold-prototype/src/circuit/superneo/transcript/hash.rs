//! One-shot raw transcript hash helpers.

use super::permutation::permute_state;
use super::*;

pub(crate) fn hash_field_linear_combinations_raw<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    field_terms: &[Vec<(Variable, SpartanF)>],
    field_constants: &[SpartanF],
    field_values: &[SpartanF],
) -> Result<[AllocatedNum<SpartanF>; DIGEST_LEN], SynthesisError> {
    if field_terms.len() != field_constants.len() || field_terms.len() != field_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let lanes = field_terms
        .iter()
        .zip(field_constants.iter())
        .zip(field_values.iter())
        .map(|((terms, constant), value)| TranscriptLane::from_terms(terms.clone(), *constant, *value))
        .collect::<Vec<_>>();
    hash_lane_slice_raw(cs.namespace(|| "hash_field_linear_combinations"), &lanes)
}

fn hash_lane_slice_raw<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    lanes: &[TranscriptLane],
) -> Result<[AllocatedNum<SpartanF>; DIGEST_LEN], SynthesisError> {
    let mut state = core::array::from_fn(|_| TranscriptLane::from_constant(SpartanF::ZERO));

    for (chunk_idx, chunk) in lanes.chunks(RATE).enumerate() {
        for (lane_idx, lane) in chunk.iter().enumerate() {
            state[lane_idx] = state[lane_idx].add(lane);
        }
        state = permute_state(cs.namespace(|| format!("permute_after_chunk_{chunk_idx}")), &state)?;
    }

    state[0] = state[0].add(&TranscriptLane::from_constant(SpartanF::ONE));
    state = permute_state(cs.namespace(|| "permute_after_padding"), &state)?;

    let mut out = Vec::with_capacity(DIGEST_LEN);
    for digest_idx in 0..DIGEST_LEN {
        out.push(state[digest_idx].allocate_canonical(cs.namespace(|| format!("digest_{digest_idx}")))?);
    }
    out.try_into().map_err(|_| SynthesisError::Unsatisfiable)
}
