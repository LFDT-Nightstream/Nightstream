//! Owns the explicit SuperNeo shard verify script.
//!
//! This mirrors the real `Π_CCS -> Π_RLC -> Π_DEC` spine.

use neo_ajtai::Commitment;
use neo_ccs::CcsStructure;
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::api::{
    rlc_public_matches_verified_inputs_with_perf, sample_rot_rhos_n_typed, verify, verify_dec_public, FoldingMode,
    RotRing,
};
use neo_reductions::engines::utils;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::{
    optimized_verify_with_cache_and_instance_digest_and_perf, OptimizedStructureCache,
};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use std::time::Instant;

use crate::finalize::public_chunk_digest;
use crate::proof::{ChunkProof, ChunkVerifyPerf, PiDecArtifact, PiRlcArtifact, PublicChunk};
use crate::prover::CommitmentMixers;

pub struct ShardVerifier;

impl ShardVerifier {
    pub fn verify_chunk_artifacts<'a, MR, MB>(
        mode: FoldingMode,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        chunk: &PublicChunk,
        incoming_main: &[neo_ccs::CeClaim<Commitment, F, K>],
        ccs_outputs: &'a [neo_ccs::CeClaim<Commitment, F, K>],
        ccs_proof: &neo_reductions::api::PiCcsProof,
        rlc: &PiRlcArtifact,
        dec: &'a PiDecArtifact,
        mixers: CommitmentMixers<MR, MB>,
        optimized_cache: Option<&OptimizedStructureCache>,
    ) -> Result<&'a [neo_ccs::CeClaim<Commitment, F, K>], PiCcsError>
    where
        MR: Fn(&[neo_ccs::Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        Ok(Self::verify_chunk_artifacts_with_perf(
            mode,
            tr,
            params,
            s,
            chunk,
            incoming_main,
            ccs_outputs,
            ccs_proof,
            rlc,
            dec,
            mixers,
            optimized_cache,
        )?
        .0)
    }

    pub fn verify_chunk_artifacts_with_perf<'a, MR, MB>(
        mode: FoldingMode,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        chunk: &PublicChunk,
        incoming_main: &[neo_ccs::CeClaim<Commitment, F, K>],
        ccs_outputs: &'a [neo_ccs::CeClaim<Commitment, F, K>],
        ccs_proof: &neo_reductions::api::PiCcsProof,
        rlc: &PiRlcArtifact,
        dec: &'a PiDecArtifact,
        mixers: CommitmentMixers<MR, MB>,
        optimized_cache: Option<&OptimizedStructureCache>,
    ) -> Result<(&'a [neo_ccs::CeClaim<Commitment, F, K>], ChunkVerifyPerf), PiCcsError>
    where
        MR: Fn(&[neo_ccs::Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let public_instance_digest = public_chunk_digest(chunk);
        Self::verify_chunk_artifacts_with_public_instance_digest_and_perf(
            mode,
            tr,
            params,
            s,
            chunk,
            incoming_main,
            ccs_outputs,
            ccs_proof,
            rlc,
            dec,
            mixers,
            optimized_cache,
            public_instance_digest,
        )
    }

    fn verify_chunk_artifacts_with_public_instance_digest_and_perf<'a, MR, MB>(
        mode: FoldingMode,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        chunk: &PublicChunk,
        incoming_main: &[neo_ccs::CeClaim<Commitment, F, K>],
        ccs_outputs: &'a [neo_ccs::CeClaim<Commitment, F, K>],
        ccs_proof: &neo_reductions::api::PiCcsProof,
        rlc: &PiRlcArtifact,
        dec: &'a PiDecArtifact,
        mixers: CommitmentMixers<MR, MB>,
        optimized_cache: Option<&OptimizedStructureCache>,
        public_instance_digest: [F; 4],
    ) -> Result<(&'a [neo_ccs::CeClaim<Commitment, F, K>], ChunkVerifyPerf), PiCcsError>
    where
        MR: Fn(&[neo_ccs::Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let total_started = Instant::now();
        validate_chunk_artifacts(chunk, ccs_outputs)?;
        append_chunk_transcript(tr, chunk);

        let prepare_inputs_started = Instant::now();
        let fresh_claims = chunk
            .steps
            .iter()
            .map(|step| step.mcs.clone())
            .collect::<Vec<_>>();
        let prepare_inputs_ms = prepare_inputs_started.elapsed().as_secs_f64() * 1_000.0;

        let ccs_started = Instant::now();
        let (ok_ccs, ccs_perf) = if matches!(mode, FoldingMode::Optimized) {
            let cache = optimized_cache.ok_or_else(|| {
                PiCcsError::InvalidInput("missing optimized structure cache for optimized verify_chunk".into())
            })?;
            optimized_verify_with_cache_and_instance_digest_and_perf(
                tr,
                params,
                s,
                &fresh_claims,
                incoming_main,
                ccs_outputs,
                ccs_proof,
                cache,
                public_instance_digest,
            )?
        } else {
            (
                verify(
                    mode,
                    tr,
                    params,
                    s,
                    &fresh_claims,
                    incoming_main,
                    ccs_outputs,
                    ccs_proof,
                )?,
                neo_reductions::optimized_engine::PiCcsVerifyPerf::default(),
            )
        };
        let ccs_ms = ccs_started.elapsed().as_secs_f64() * 1_000.0;
        if !ok_ccs {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_CCS verification failed for chunk starting at {}",
                chunk.start_index
            )));
        }

        let digest_checks_started = Instant::now();
        let observed_digest = tr.digest32();
        if ccs_proof.header_digest.as_slice() != observed_digest {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_CCS header digest mismatch for chunk starting at {}",
                chunk.start_index
            )));
        }
        for (idx, out) in ccs_outputs.iter().enumerate() {
            if out.fold_digest != observed_digest {
                return Err(PiCcsError::ProtocolError(format!(
                    "Π_CCS output[{idx}] fold_digest mismatch for chunk starting at {}",
                    chunk.start_index
                )));
            }
        }
        let digest_checks_ms = digest_checks_started.elapsed().as_secs_f64() * 1_000.0;

        let dims_started = Instant::now();
        let dims = utils::build_dims_and_policy(params, s)?;
        let dims_ms = dims_started.elapsed().as_secs_f64() * 1_000.0;
        let rlc_challenge_started = Instant::now();
        let expected_rhos = sample_rlc_rhos(tr, params, ccs_outputs.len())?;
        let rlc_challenge_ms = rlc_challenge_started.elapsed().as_secs_f64() * 1_000.0;

        let rlc_started = Instant::now();
        let (parent_matches, rlc_perf) = rlc_public_matches_verified_inputs_with_perf(
            s,
            params,
            &expected_rhos,
            ccs_outputs,
            &rlc.parent,
            mixers.mix_rhos_commits,
            dims.ell_d,
        )?;
        if !parent_matches {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_RLC public recompute mismatch for chunk starting at {}",
                chunk.start_index
            )));
        }
        let rlc_ms = rlc_started.elapsed().as_secs_f64() * 1_000.0;

        let dec_started = Instant::now();
        if !verify_dec_public(s, params, &rlc.parent, &dec.children, mixers.combine_b_pows, dims.ell_d) {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_DEC public verification failed for chunk starting at {}",
                chunk.start_index
            )));
        }
        let dec_ms = dec_started.elapsed().as_secs_f64() * 1_000.0;

        let perf = ChunkVerifyPerf {
            start_index: chunk.start_index,
            fresh_steps: chunk.steps.len(),
            incoming_main_claims: incoming_main.len(),
            ccs_outputs: ccs_outputs.len(),
            dec_children: dec.children.len(),
            prepare_inputs_ms,
            ccs_bind_ms: ccs_perf.bind_ms,
            ccs_bind_header_instances_ms: ccs_perf.bind_header_instances_ms,
            ccs_bind_header_prefix_ms: ccs_perf.bind_header_prefix_ms,
            ccs_bind_header_poly_ms: ccs_perf.bind_header_poly_ms,
            ccs_bind_header_public_instances_ms: ccs_perf.bind_header_public_instances_ms,
            ccs_bind_me_inputs_ms: ccs_perf.bind_me_inputs_ms,
            ccs_bind_sample_challenges_ms: ccs_perf.bind_sample_challenges_ms,
            ccs_fe_sumcheck_ms: ccs_perf.fe_sumcheck_ms,
            ccs_nc_sumcheck_ms: ccs_perf.nc_sumcheck_ms,
            ccs_output_checks_ms: ccs_perf.output_checks_ms,
            ccs_terminal_ms: ccs_perf.terminal_ms,
            ccs_ms,
            digest_checks_ms,
            dims_ms,
            rlc_challenge_ms,
            rlc_rho_mats_ms: rlc_perf.rho_mats_ms,
            rlc_rho_k_lift_ms: rlc_perf.rho_k_lift_ms,
            rlc_x_ms: rlc_perf.x_ms,
            rlc_y_ms: rlc_perf.y_ms,
            rlc_y_zcol_ms: rlc_perf.y_zcol_ms,
            rlc_aux_ms: rlc_perf.aux_ms,
            rlc_commitment_collect_ms: rlc_perf.commitment_collect_ms,
            rlc_commitment_mix_ms: rlc_perf.commitment_mix_ms,
            rlc_commitment_ms: rlc_perf.commitment_ms,
            rlc_ms,
            dec_ms,
            total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
        };

        Ok((&dec.children, perf))
    }

    pub fn verify_chunk<'a, MR, MB>(
        mode: FoldingMode,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        chunk: &PublicChunk,
        incoming_main: &[neo_ccs::CeClaim<Commitment, F, K>],
        proof: &'a ChunkProof,
        mixers: CommitmentMixers<MR, MB>,
        optimized_cache: Option<&OptimizedStructureCache>,
    ) -> Result<&'a [neo_ccs::CeClaim<Commitment, F, K>], PiCcsError>
    where
        MR: Fn(&[neo_ccs::Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        Ok(Self::verify_chunk_with_perf(
            mode,
            tr,
            params,
            s,
            chunk,
            incoming_main,
            proof,
            mixers,
            optimized_cache,
        )?
        .0)
    }

    pub fn verify_chunk_with_perf<'a, MR, MB>(
        mode: FoldingMode,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        chunk: &PublicChunk,
        incoming_main: &[neo_ccs::CeClaim<Commitment, F, K>],
        proof: &'a ChunkProof,
        mixers: CommitmentMixers<MR, MB>,
        optimized_cache: Option<&OptimizedStructureCache>,
    ) -> Result<(&'a [neo_ccs::CeClaim<Commitment, F, K>], ChunkVerifyPerf), PiCcsError>
    where
        MR: Fn(&[neo_ccs::Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        validate_chunk_metadata(chunk, proof)?;
        Self::verify_chunk_artifacts_with_perf(
            mode,
            tr,
            params,
            s,
            chunk,
            incoming_main,
            &proof.ccs_outputs,
            &proof.ccs_proof,
            &proof.rlc,
            &proof.dec,
            mixers,
            optimized_cache,
        )
    }

    pub fn verify_chunk_with_precomputed_digest_with_perf<'a, MR, MB>(
        mode: FoldingMode,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        chunk: &PublicChunk,
        incoming_main: &[neo_ccs::CeClaim<Commitment, F, K>],
        proof: &'a ChunkProof,
        mixers: CommitmentMixers<MR, MB>,
        optimized_cache: Option<&OptimizedStructureCache>,
        public_instance_digest: [F; 4],
    ) -> Result<(&'a [neo_ccs::CeClaim<Commitment, F, K>], ChunkVerifyPerf), PiCcsError>
    where
        MR: Fn(&[neo_ccs::Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        validate_chunk_metadata(chunk, proof)?;
        Self::verify_chunk_artifacts_with_public_instance_digest_and_perf(
            mode,
            tr,
            params,
            s,
            chunk,
            incoming_main,
            &proof.ccs_outputs,
            &proof.ccs_proof,
            &proof.rlc,
            &proof.dec,
            mixers,
            optimized_cache,
            public_instance_digest,
        )
    }
}

const CHUNK_META_RAW_TAG: u64 = 14;
const STEP_INDEX_RAW_TAG: u64 = 15;

fn append_chunk_transcript(tr: &mut Poseidon2Transcript, chunk: &PublicChunk) {
    if chunk.steps.len() == 1 {
        tr.append_fields_raw(&[F::from_u64(STEP_INDEX_RAW_TAG), F::from_u64(chunk.start_index as u64)]);
        return;
    }

    tr.append_fields_raw(&[
        F::from_u64(CHUNK_META_RAW_TAG),
        F::from_u64(chunk.start_index as u64),
        F::from_u64(chunk.steps.len() as u64),
    ]);
}

fn validate_chunk_metadata(chunk: &PublicChunk, proof: &ChunkProof) -> Result<(), PiCcsError> {
    if proof.chunk.start_index != chunk.start_index {
        return Err(PiCcsError::InvalidInput(format!(
            "proof chunk start mismatch: expected {}, got {}",
            chunk.start_index, proof.chunk.start_index
        )));
    }
    if proof.chunk.steps.len() != chunk.steps.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "proof chunk length mismatch: expected {}, got {}",
            chunk.steps.len(),
            proof.chunk.steps.len()
        )));
    }
    for (idx, (expected, actual)) in chunk.steps.iter().zip(proof.chunk.steps.iter()).enumerate() {
        if actual.label != expected.label {
            return Err(PiCcsError::InvalidInput(format!(
                "proof chunk step[{idx}] label mismatch: expected '{}', got '{}'",
                expected.label, actual.label
            )));
        }
        if actual.mcs.m_in != expected.mcs.m_in || actual.mcs.x != expected.mcs.x || actual.mcs.c != expected.mcs.c {
            return Err(PiCcsError::InvalidInput(format!(
                "public MCS mismatch for chunk step[{}] '{}'",
                idx, expected.label
            )));
        }
    }
    if proof.ccs_outputs.is_empty() {
        return Err(PiCcsError::InvalidInput("missing Π_CCS outputs for chunk".into()));
    }
    Ok(())
}

fn validate_chunk_artifacts(
    chunk: &PublicChunk,
    ccs_outputs: &[neo_ccs::CeClaim<Commitment, F, K>],
) -> Result<(), PiCcsError> {
    if chunk.steps.is_empty() {
        return Err(PiCcsError::InvalidInput("missing public steps for chunk".into()));
    }
    if ccs_outputs.is_empty() {
        return Err(PiCcsError::InvalidInput("missing Π_CCS outputs for chunk".into()));
    }
    Ok(())
}

fn sample_rlc_rhos(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    input_count: usize,
) -> Result<Vec<neo_reductions::api::RotRho>, PiCcsError> {
    let ring = RotRing::goldilocks();
    sample_rot_rhos_n_typed(tr, params, &ring, input_count)
}
