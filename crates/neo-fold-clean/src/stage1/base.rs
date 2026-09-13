//! Base caller data for the selected emitted witness program. The zero inner
//! proof is a placeholder, while its sampler and public split remain exact.
//! The retained base accumulator is the separate canonical zero instance.

use neo_ccs::Mat;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::{api::rlc_public, common::split_b_matrix_k};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::{
    derive_pi_ccs_v1_1_transcript, PiCcsV1_1OutputEvaluations, PiCcsV1_1PackageInputs, PiDecV1_1PackageInputs,
    PI_CCS_V1_1_FRESH_COMMITMENT_WORDS, PI_CCS_V1_1_MATRIX_COUNT, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS,
    PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT, PI_CCS_V1_1_SOURCE_COUNT, PI_DEC_V1_1_CHILD_COUNT,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{
    encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage,
    step_inputs::{application_output, digest_bytes},
    ExtendError, Poseidon2HashChainV1Package, Stage1State, Stage1StepInputs,
};
use crate::{
    engine::optimized,
    paper::{
        construction2::{LaneCommitmentMode, RunningInstance},
        params::Params,
        relations::ajtai_rlc_mixer,
    },
};

impl Poseidon2HashChainV1Package {
    pub(super) fn base_inputs(
        &self,
        params: &Params,
        z0: [F; 4],
        message: [F; 4],
    ) -> Result<(Stage1StepInputs, Vec<Mat<F>>), ExtendError> {
        let mut running = RunningInstance::canonical_zero(
            params,
            &self.structure,
            PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS,
            LaneCommitmentMode::Plain,
        )
        .map_err(|_| ExtendError::Input("canonical zero running shape"))?;
        let context = self.binding.verifier_context().digest().map(F::from_u64);
        let prior_preimage = serialize_pi_ccs_v1_1_state_preimage(context, 0, z0, z0, &running.claims, 1)?;
        let prior_digest = pi_ccs_v1_1_state_hash(&prior_preimage)?;
        let prior_public = encode_pi_ccs_v1_1_public_input(prior_digest)?;
        let rounds = vec![vec![[0; 2]; PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT]; PI_CCS_V1_1_ROUND_COUNT];
        let evaluation_words = (PI_CCS_V1_1_MATRIX_COUNT + 1) * D * 2;
        let transcript = derive_pi_ccs_v1_1_transcript(
            &[
                prior_digest.to_vec(),
                vec![0; PI_CCS_V1_1_FRESH_COMMITMENT_WORDS],
                prior_public.clone(),
            ],
            &[
                vec![0; PI_CCS_V1_1_ROUND_COUNT * 2],
                vec![0; PI_DEC_V1_1_CHILD_COUNT * evaluation_words],
            ],
            &rounds,
            &vec![0; PI_CCS_V1_1_SOURCE_COUNT * evaluation_words],
        )?;
        let point = transcript
            .round_point()
            .iter()
            .map(|value| K::from_coeffs(value.map(F::from_u64)))
            .collect::<Vec<_>>();
        let mut sampler = Poseidon2Transcript::from_state_and_absorbed(transcript.outgoing_state().map(F::from_u64), 0);
        let rhos = optimized::sample_rho_n(&mut sampler, params, PI_CCS_V1_1_SOURCE_COUNT)?;

        // Only the first dummy source has the fresh public input. All
        // commitments and evaluations are zero, at the derived round point.
        let mut sources = vec![running.claims[0].clone(); PI_CCS_V1_1_SOURCE_COUNT];
        for source in &mut sources {
            source.r = point.clone();
            source.fold_digest = digest_bytes(prior_digest);
        }
        for (column, value) in prior_public.iter().copied().enumerate() {
            sources[0].X[(column % D, column / D)] = F::from_u64(value);
        }
        let parent = rlc_public(
            &self.structure,
            params.inner(),
            &rhos,
            &sources,
            ajtai_rlc_mixer,
            D.next_power_of_two().trailing_zeros() as usize,
        )?;
        let child_public = split_b_matrix_k(&parent.X, PI_DEC_V1_1_CHILD_COUNT, params.b())?;
        let pi_dec = PiDecV1_1PackageInputs::new(
            vec![vec![0; PI_CCS_V1_1_FRESH_COMMITMENT_WORDS]; PI_DEC_V1_1_CHILD_COUNT],
            vec![vec![[0; 2]; D]; PI_DEC_V1_1_CHILD_COUNT],
            vec![vec![vec![[0; 2]; D]; PI_CCS_V1_1_MATRIX_COUNT]; PI_DEC_V1_1_CHILD_COUNT],
            child_public
                .iter()
                .map(|child| {
                    (0..PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS)
                        .map(|column| child[(column % D, column / D)].as_canonical_u64())
                        .collect()
                })
                .collect(),
        )?;

        // Dummy child public digits fill the always-present IR input; the
        // formal base branch retains the canonical zero running instance.
        let output = application_output(z0, message);
        let output_preimage = serialize_pi_ccs_v1_1_state_preimage(context, 1, z0, output, &running.claims, 1)?;
        let output_digest = pi_ccs_v1_1_state_hash(&output_preimage)?;
        let next_public_input = encode_pi_ccs_v1_1_public_input(output_digest)?;
        let pi_ccs = PiCcsV1_1PackageInputs::new(
            prior_preimage,
            output_preimage.clone(),
            vec![0; PI_CCS_V1_1_FRESH_COMMITMENT_WORDS],
            rounds,
            PiCcsV1_1OutputEvaluations::new(
                vec![vec![[0; 2]; D]; PI_CCS_V1_1_SOURCE_COUNT],
                vec![vec![vec![[0; 2]; D]; PI_CCS_V1_1_MATRIX_COUNT]; PI_CCS_V1_1_SOURCE_COUNT],
            )?,
            prior_public,
            output_digest,
            self.binding.verifier_context().clone(),
        )?;
        let frame = digest_bytes(output_digest);
        for claim in running
            .claims
            .iter_mut()
            .chain(running.parent_authority.iter_mut())
        {
            claim.fold_digest = frame;
        }
        let witnesses = std::mem::take(&mut running.witnesses);
        Ok((
            Stage1StepInputs {
                pi_ccs,
                pi_dec,
                application_witness: message.map(|field| field.as_canonical_u64()),
                output_preimage,
                output_digest,
                next_public_input,
                next_state: Stage1State::new(1, z0, output),
                next_running: running,
            },
            witnesses,
        ))
    }
}
