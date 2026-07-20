//! Native transcript replay for the production 19-block/6-lane NC channel.

use neo_ccs::{CcsStructure, CcsWitness, Mat};
use neo_math::{D, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::engines::utils::{self, Dims};
use crate::error::PiCcsError;
use crate::sumcheck::RoundOracle;

use super::common::Challenges;
use super::oracle::{
    BlockLaneNcChallenges, BlockLaneNcOracle, BlockLaneNcPending, BLOCK_LANE_NC_BLOCK_VARIABLES,
    BLOCK_LANE_NC_LANE_VARIABLES,
};
use super::transcript_segments::append_nc_sumcheck_prolog;

pub(super) struct BlockLaneNcTrace {
    pub rounds: Option<Vec<Vec<K>>>,
    pub challenges: Vec<K>,
    pub final_sum: K,
    pub initial_sum: K,
    pub block_rows: Vec<[K; D]>,
}

pub(super) fn sample_challenges(
    transcript: &mut Poseidon2Transcript,
    dims: Dims,
    public: &mut Challenges,
) -> Result<BlockLaneNcChallenges, PiCcsError> {
    if dims.ell_block != BLOCK_LANE_NC_BLOCK_VARIABLES || dims.ell_d != BLOCK_LANE_NC_LANE_VARIABLES {
        return Err(PiCcsError::InvalidInput(format!(
            "block-lane delayed variant requires 19 block and 6 lane variables, got {} and {}",
            dims.ell_block, dims.ell_d
        )));
    }
    let beta_block = utils::sample_beta_block(transcript, BLOCK_LANE_NC_BLOCK_VARIABLES)?;
    let (producer_beta, batch_weight) = utils::sample_delayed_projection_challenges(transcript)?;
    public.beta_m = beta_block.clone();
    Ok(BlockLaneNcChallenges {
        beta_block: beta_block
            .try_into()
            .map_err(|_| PiCcsError::ProtocolError("block challenge arity changed".into()))?,
        beta_lane: public
            .beta_a
            .clone()
            .try_into()
            .map_err(|_| PiCcsError::ProtocolError("lane challenge arity changed".into()))?,
        gamma: public.gamma,
        producer_beta,
        batch_weight,
    })
}

pub(super) fn run<'a>(
    transcript: &mut Poseidon2Transcript,
    structure: &CcsStructure<F>,
    fresh: &'a [CcsWitness<F>],
    running: &'a [Mat<F>],
    challenges: BlockLaneNcChallenges,
    pending: Option<BlockLaneNcPending>,
    capture_rounds: bool,
) -> Result<BlockLaneNcTrace, PiCcsError> {
    let mut oracle = BlockLaneNcOracle::new(structure, fresh, running, challenges, pending)?;
    let initial_sum = oracle.initial_sum();
    append_nc_sumcheck_prolog(transcript, initial_sum);
    let mut claimed = initial_sum;
    let mut round_challenges = Vec::with_capacity(oracle.num_rounds());
    let mut rounds = capture_rounds.then(|| Vec::with_capacity(oracle.num_rounds()));

    for round in 0..oracle.num_rounds() {
        let coefficients = oracle.round_coefficients();
        if coefficients[0] + crate::sumcheck::poly_eval_k_base(&coefficients, F::ONE) != claimed {
            return Err(PiCcsError::SumcheckError(format!(
                "block-lane NC sumcheck invariant failed at round {round}"
            )));
        }
        transcript.append_fields_raw(&crate::sumcheck::round_coeff_fields(&coefficients));
        let sampled = transcript.challenge_fields_raw(2);
        let challenge = neo_math::from_complex(sampled[0], sampled[1]);
        claimed = crate::sumcheck::poly_eval_k(&coefficients, challenge);
        round_challenges.push(challenge);
        oracle.fold(challenge);
        if let Some(rounds) = rounds.as_mut() {
            rounds.push(coefficients.to_vec());
        }
    }

    if claimed != oracle.finalized_value() {
        return Err(PiCcsError::SumcheckError(
            "block-lane NC terminal disagrees with the folded raw oracle".into(),
        ));
    }
    let block_rows = oracle
        .block_projected_source_rows()
        .iter()
        .map(|row| std::array::from_fn(|lane| row[lane]))
        .collect();
    Ok(BlockLaneNcTrace {
        rounds,
        challenges: round_challenges,
        final_sum: claimed,
        initial_sum,
        block_rows,
    })
}
