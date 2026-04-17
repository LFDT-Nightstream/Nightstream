//! Owns a compact native chunk-authority surface for the future recursive-step inner verifier.
//!
//! This is not a verifier boundary by itself. Every field here is derived from
//! authoritative native inputs so the current recursive-step payload can be
//! checked against the real chunk replay without carrying the full replay
//! theorem forever.

use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{F, K};
use neo_reductions::engines::utils::me_digest_poseidon_into;
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use super::super::chunk_step_recursive::rv64im_chunk_step_recursive_carry_state_digest;
use super::super::Rv64imMainRecursionFPrimeBackendRelation;
use crate::finalize::digest_fields_as_digest32;
use crate::rv64im::final_relation::{
    rv64im_chunk_fold_transcript_snapshot_digest, rv64im_recursive_accumulator_instance_digest_from_parts,
    Rv64imChunkFoldTranscriptSnapshot,
};
use crate::rv64im::main_relation_trace::Rv64imMainCircuitChunkReplaySurface;
use crate::rv64im::SimpleKernelError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainRecursionStepAuthoritativeChunkSurface {
    pub chunk_index: u64,
    pub vk_fs_digest: [u8; 32],
    pub transcript_in_digest: [u8; 32],
    pub transcript_out_digest: [u8; 32],
    pub state_in_claim_count: u64,
    pub state_in_claims_digest: [u8; 32],
    pub state_out_claim_count: u64,
    pub state_out_claims_digest: [u8; 32],
    pub ccs_output_count: u64,
    pub ccs_outputs_digest: [u8; 32],
    pub child_claim_count: u64,
    pub child_claims_digest: [u8; 32],
    pub parent_claim_digest: [u8; 32],
    pub public_chunk_instance_digest: [u8; 32],
    pub public_chunk_digest: [u8; 32],
    pub bridge_handoff_digest: [u8; 32],
    pub chunk_relation_digest: [u8; 32],
    pub carry_state_in_digest: [u8; 32],
    pub carry_state_out_digest: [u8; 32],
    pub folded_accumulator_digest: [u8; 32],
    pub terminal_handle_digest: [u8; 32],
}

impl Rv64imMainRecursionStepAuthoritativeChunkSurface {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/version",
            b"v1",
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/chunk_index",
            &[self.chunk_index],
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/vk_fs_digest",
            &self.vk_fs_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/transcript_in_digest",
            &self.transcript_in_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/transcript_out_digest",
            &self.transcript_out_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/state_in_claim_count",
            &[self.state_in_claim_count],
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/state_in_claims_digest",
            &self.state_in_claims_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/state_out_claim_count",
            &[self.state_out_claim_count],
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/state_out_claims_digest",
            &self.state_out_claims_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/ccs_output_count",
            &[self.ccs_output_count],
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/ccs_outputs_digest",
            &self.ccs_outputs_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/child_claim_count",
            &[self.child_claim_count],
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/child_claims_digest",
            &self.child_claims_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/parent_claim_digest",
            &self.parent_claim_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/public_chunk_instance_digest",
            &self.public_chunk_instance_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/public_chunk_digest",
            &self.public_chunk_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/bridge_handoff_digest",
            &self.bridge_handoff_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/chunk_relation_digest",
            &self.chunk_relation_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/carry_state_in_digest",
            &self.carry_state_in_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/carry_state_out_digest",
            &self.carry_state_out_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/folded_accumulator_digest",
            &self.folded_accumulator_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/terminal_handle_digest",
            &self.terminal_handle_digest,
        );
        tr.digest32()
    }
}

fn digest_me_claim_sequence(domain: &'static [u8], claims: &[CeClaim<Commitment, F, K>]) -> [u8; 32] {
    let mut scratch = Vec::<F>::with_capacity(2048);
    let mut tr = Poseidon2Transcript::new(domain);
    tr.append_message(b"version", b"v1");
    tr.append_u64s(b"count", &[claims.len() as u64]);
    for claim in claims {
        tr.append_message(
            b"claim_digest",
            &digest_fields_as_digest32(me_digest_poseidon_into(&mut scratch, claim)),
        );
    }
    tr.digest32()
}

fn build_surface_from_parts(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
    transcript_in: &Rv64imChunkFoldTranscriptSnapshot,
    transcript_out: &Rv64imChunkFoldTranscriptSnapshot,
    state_in_claims: &[CeClaim<Commitment, F, K>],
    state_out_claims: &[CeClaim<Commitment, F, K>],
    replay_surface: &Rv64imMainCircuitChunkReplaySurface,
) -> Rv64imMainRecursionStepAuthoritativeChunkSurface {
    let advice = &backend_relation.f_prime_advice;
    let terminal_handle_digest = *backend_relation.payload.z_next();
    Rv64imMainRecursionStepAuthoritativeChunkSurface {
        chunk_index: advice.chunk_index(),
        vk_fs_digest: advice.verifier_key_fs().expected_digest(),
        transcript_in_digest: rv64im_chunk_fold_transcript_snapshot_digest(transcript_in),
        transcript_out_digest: rv64im_chunk_fold_transcript_snapshot_digest(transcript_out),
        state_in_claim_count: state_in_claims.len() as u64,
        state_in_claims_digest: digest_me_claim_sequence(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/state_in_claims",
            state_in_claims,
        ),
        state_out_claim_count: state_out_claims.len() as u64,
        state_out_claims_digest: digest_me_claim_sequence(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/state_out_claims",
            state_out_claims,
        ),
        ccs_output_count: replay_surface.pi_ccs.ccs_outputs.len() as u64,
        ccs_outputs_digest: digest_me_claim_sequence(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/ccs_outputs",
            &replay_surface.pi_ccs.ccs_outputs,
        ),
        child_claim_count: replay_surface.pi_dec.children.len() as u64,
        child_claims_digest: digest_me_claim_sequence(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/children",
            &replay_surface.pi_dec.children,
        ),
        parent_claim_digest: digest_me_claim_sequence(
            b"neo.fold.next/rv64im/main_recursion_step_authoritative_chunk_surface/parent",
            std::slice::from_ref(&replay_surface.pi_rlc.parent),
        ),
        public_chunk_instance_digest: digest_fields_as_digest32(replay_surface.handoff.public_chunk_instance_digest),
        public_chunk_digest: replay_surface.handoff.public_chunk_digest,
        bridge_handoff_digest: replay_surface.handoff.bridge_handoff_digest,
        chunk_relation_digest: replay_surface.handoff.chunk_relation_digest,
        carry_state_in_digest: rv64im_chunk_step_recursive_carry_state_digest(
            state_in_claims,
            transcript_in,
            *backend_relation.payload.z_i(),
        ),
        carry_state_out_digest: rv64im_chunk_step_recursive_carry_state_digest(
            state_out_claims,
            transcript_out,
            terminal_handle_digest,
        ),
        folded_accumulator_digest: rv64im_recursive_accumulator_instance_digest_from_parts(
            state_out_claims,
            terminal_handle_digest,
        ),
        terminal_handle_digest,
    }
}

fn build_native_surface(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepAuthoritativeChunkSurface, SimpleKernelError> {
    let native_trace = backend_relation.f_prime_advice.main_circuit_chunk_trace();
    let native_replay_surface = native_trace.replay_surface()?;
    Ok(build_surface_from_parts(
        backend_relation,
        &backend_relation.f_prime_advice.running_state().transcript,
        &backend_relation.f_prime_advice.fresh_state_out().transcript,
        &backend_relation
            .f_prime_advice
            .running_state()
            .carry
            .main
            .claims,
        &backend_relation
            .f_prime_advice
            .fresh_state_out()
            .carry
            .main
            .claims,
        &native_replay_surface,
    ))
}

pub fn build_rv64im_main_recursion_step_authoritative_chunk_surface(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepAuthoritativeChunkSurface, SimpleKernelError> {
    let payload = &backend_relation.payload;
    let effective_replay_surface = payload.effective_chunk_replay_surface(
        &backend_relation.f_prime_advice.running_state().transcript,
        &backend_relation
            .f_prime_advice
            .running_state()
            .carry
            .main
            .claims,
    )?;
    let state_in_claim_count = payload.step_shape.state_in_claim_count as usize;
    let state_out_claim_count = payload.step_shape.state_out_claim_count as usize;
    if payload.state_in_claims.len() < state_in_claim_count || payload.state_out_claims.len() < state_out_claim_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM recursive-step authoritative chunk surface cannot truncate carried claims past the live step counts"
                .into(),
        ));
    }
    Ok(build_surface_from_parts(
        backend_relation,
        &backend_relation.f_prime_advice.running_state().transcript,
        payload.fixed_transcript_out(),
        &payload.state_in_claims[..state_in_claim_count],
        &payload.state_out_claims[..state_out_claim_count],
        &effective_replay_surface,
    ))
}

pub fn debug_check_rv64im_main_recursion_step_authoritative_chunk_surface_matches_native(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), SimpleKernelError> {
    let carried_surface = build_rv64im_main_recursion_step_authoritative_chunk_surface(backend_relation)?;
    let native_surface = build_native_surface(backend_relation)?;
    if carried_surface != native_surface {
        return Err(SimpleKernelError::Bridge(
            "RV64IM recursive-step authoritative chunk surface recovered from the carried payload does not match the native chunk replay theorem surface"
                .into(),
        ));
    }
    Ok(())
}
