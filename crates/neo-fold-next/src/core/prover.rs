//! Owns the explicit SuperNeo shard prove script.
//!
//! Ownership:
//! - sequences `Π_CCS -> Π_RLC -> Π_DEC`
//! - does not build VM/frontend step relations
//! - does not own sibling-family proofs

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::CcsStructure;
use neo_math::F;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::Poseidon2Transcript;

use crate::chunk_relation::{compute_chunk_relation_for_prover_chunk_with_perf, compute_chunk_relation_with_perf};
use crate::proof::{Carry, ChunkInput, ChunkProvePerf, ChunkResult, ProverChunkInput};

pub use crate::chunk_relation::{ChunkRelationArtifacts, CommitmentMixers};

pub struct ShardProver;

impl ShardProver {
    pub fn prove_chunk<L, MR, MB>(
        mode: FoldingMode,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        chunk: &ChunkInput,
        incoming_main: &Carry,
        log: &L,
        mixers: CommitmentMixers<MR, MB>,
        optimized_cache: Option<&OptimizedStructureCache>,
    ) -> Result<ChunkResult, PiCcsError>
    where
        L: SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[neo_ccs::Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        Ok(Self::prove_chunk_with_perf(mode, tr, params, s, chunk, incoming_main, log, mixers, optimized_cache)?.0)
    }

    pub fn prove_chunk_with_perf<L, MR, MB>(
        mode: FoldingMode,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        chunk: &ChunkInput,
        incoming_main: &Carry,
        log: &L,
        mixers: CommitmentMixers<MR, MB>,
        optimized_cache: Option<&OptimizedStructureCache>,
    ) -> Result<(ChunkResult, ChunkProvePerf), PiCcsError>
    where
        L: SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[neo_ccs::Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let (computation, perf) =
            compute_chunk_relation_with_perf(mode, tr, params, s, chunk, incoming_main, log, mixers, optimized_cache)?;
        Ok((computation.into_chunk_result(chunk), perf))
    }

    pub(crate) fn prove_prepared_chunk_with_perf<L, MR, MB>(
        mode: FoldingMode,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        chunk: &ProverChunkInput,
        incoming_main: &Carry,
        log: &L,
        mixers: CommitmentMixers<MR, MB>,
        optimized_cache: Option<&OptimizedStructureCache>,
    ) -> Result<(ChunkResult, ChunkProvePerf), PiCcsError>
    where
        L: SModuleHomomorphism<F, Commitment> + Sync,
        MR: Fn(&[neo_ccs::Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
        MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
    {
        let (computation, perf) = compute_chunk_relation_for_prover_chunk_with_perf(
            mode,
            tr,
            params,
            s,
            chunk,
            incoming_main,
            log,
            mixers,
            optimized_cache,
        )?;
        Ok((
            computation.into_chunk_result_with_public_chunk(chunk.public_chunk.clone()),
            perf,
        ))
    }
}
