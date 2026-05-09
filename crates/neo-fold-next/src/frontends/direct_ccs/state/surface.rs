//! Builds direct-CCS circuit surfaces from native SuperNeo step relations.

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, Mat};
use neo_math::F;
use neo_reductions::engines::optimized_engine::OptimizedStructureCache;
use neo_reductions::engines::utils::Dims;
use neo_transcript::Poseidon2Transcript;

use super::super::terminal::gadgets::direct_accumulator_digest_from_claims;
use super::{DirectCcsChunkCircuitSurface, DirectCcsFPrimeSnarkError};
use crate::chunk_folding::trace_chunk_relation_with_witness_and_instance_digest_and_me_input_handle;
use crate::finalize::{digest32_as_fields, digest_fields_as_digest32, public_chunk_digest};
use crate::ivc::SuperNeoIvcStepRelation;
use crate::superneo_nifs_circuit::{
    build_superneo_chunk_replay_surface, build_superneo_pi_ccs_replay_surface, SuperNeoChunkCover,
    SuperNeoChunkHandoff, SuperNeoPublicInputLayout,
};

pub(crate) fn build_direct_ccs_chunk_surface_from_ivc_relation<L, MR, MB>(
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    _dims: Dims,
    relation: &SuperNeoIvcStepRelation,
    log: &L,
    mixers: crate::prover::CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
) -> Result<DirectCcsChunkCircuitSurface, DirectCcsFPrimeSnarkError>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let public_chunk = relation.chunk.public();
    let public_chunk_instance_digest = public_chunk_digest(&public_chunk);
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(
        relation.state_in.transcript.state,
        relation.state_in.transcript.absorbed,
    );
    let me_input_accumulator_handle = digest32_as_fields(direct_accumulator_digest_from_claims(
        params,
        &relation.state_in.carry.claims,
    ));
    let trace = trace_chunk_relation_with_witness_and_instance_digest_and_me_input_handle(
        &mut transcript,
        params,
        structure,
        &relation.chunk,
        &relation.state_in.carry,
        &relation.replay_witness,
        log,
        mixers,
        optimized_cache,
        public_chunk_instance_digest,
        me_input_accumulator_handle,
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
    let handoff = SuperNeoChunkHandoff {
        public_chunk,
        public_chunk_instance_digest,
        public_chunk_digest: digest_fields_as_digest32(public_chunk_instance_digest),
        bridge_handoff_digest: [0u8; 32],
        chunk_relation_digest: relation.chunk_relation_digest,
        public_input_layout: SuperNeoPublicInputLayout::PackedPrefix,
    };
    let pi_ccs = build_superneo_pi_ccs_replay_surface(
        trace.ccs_outputs,
        trace.ccs_replay_proof,
        trace.terminal_state.challenges_public,
        trace.terminal_state.row_chals,
        trace.terminal_state.alpha_prime,
        trace.terminal_state.s_col,
        trace.terminal_state.alpha_prime_nc,
    );
    let fresh_claims = relation
        .chunk
        .steps
        .iter()
        .map(|step| step.mcs.clone())
        .collect::<Vec<_>>();
    let fresh_witnesses = relation
        .chunk
        .steps
        .iter()
        .map(|step| step.witness.clone())
        .collect::<Vec<_>>();
    let replay = build_superneo_chunk_replay_surface(
        &handoff,
        &fresh_claims,
        &fresh_witnesses,
        pi_ccs,
        trace.parent,
        trace.children,
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Input(format!("failed to build latest direct chunk surface: {err}")))?;
    let cover = SuperNeoChunkCover::from_replay_surface(&replay);
    Ok(DirectCcsChunkCircuitSurface { cover, replay })
}
