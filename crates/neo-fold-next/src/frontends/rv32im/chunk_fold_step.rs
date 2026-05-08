//! Owns the explicit RV32IM one-chunk fold-verifier step reused by recursion and decider tracing.

use neo_ajtai::{AjtaiSModule, Commitment};
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::me_input_projection_digest_poseidon_into;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::chunk_relation::ChunkReplayWitness;
use crate::finalize::fixed_shape_recursive_seed;
use crate::proof::{Carry, ChunkProvePerf, PublicChunk};
use crate::rv32im::chunk_relation::{
    prove_rv32im_chunk_transition_with_perf, rv32im_step_handle, verify_rv32im_chunk_relation_with_replay,
};
use crate::rv32im::kernel::{Rv32imVerifiedKernelChunkHandoff, SimpleKernelError};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imAccumulatorHandle(pub [u8; 32]);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Rv32imChunkFoldCarry {
    pub main: Carry,
    pub main_projection_digests: Vec<[F; 4]>,
    pub terminal_handle: Rv32imAccumulatorHandle,
}

impl Rv32imChunkFoldCarry {
    pub fn from_main(main: Carry, terminal_handle: Rv32imAccumulatorHandle) -> Self {
        let main_projection_digests = rv32im_main_claim_projection_digests(&main.claims);
        Self {
            main,
            main_projection_digests,
            terminal_handle,
        }
    }

    pub fn seed() -> Self {
        Self::seed_for_step_cap(1)
    }

    pub fn seed_for_claim_count(claim_count: usize) -> Self {
        Self::from_main(
            crate::rv32im::construction2_default::build_rv32im_main_recursion_canonical_zero_carry_for_claim_count(
                claim_count,
            )
            .expect("canonical RV32IM chunk-fold seed carry must build"),
            Rv32imAccumulatorHandle(rv32im_chunk_fold_seed()),
        )
    }

    pub fn seed_for_step_cap(step_cap: usize) -> Self {
        let claim_count = crate::rv32im::kernel::rv32im_simple_root_params_for_step_cap(step_cap).k_rho as usize;
        Self::seed_for_claim_count(claim_count)
    }

    pub fn validate_projection_digests(&self, label: &str) -> Result<(), SimpleKernelError> {
        let expected = try_rv32im_main_claim_projection_digests(&self.main.claims)?;
        if self.main_projection_digests != expected {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM chunk-fold carry {label} projection digests drifted from the authoritative carried CE claims"
            )));
        }
        Ok(())
    }
}

pub(crate) fn rv32im_main_claim_projection_digests(claims: &[CeClaim<Commitment, F, K>]) -> Vec<[F; 4]> {
    try_rv32im_main_claim_projection_digests(claims)
        .expect("RV32IM carried CE projection digest requires SuperNeo X shape")
}

pub(crate) fn try_rv32im_main_claim_projection_digests(
    claims: &[CeClaim<Commitment, F, K>],
) -> Result<Vec<[F; 4]>, SimpleKernelError> {
    let mut scratch = Vec::<F>::with_capacity(2048);
    claims
        .iter()
        .enumerate()
        .map(|(idx, claim)| {
            me_input_projection_digest_poseidon_into(&mut scratch, claim).map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV32IM carried CE projection digest {idx} requires SuperNeo X = D x m_in shape: {err}"
                ))
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imChunkStepPublic {
    pub program_digest: [u8; 32],
    pub chunk_index: u64,
    pub step_lo: u64,
    pub step_hi: u64,
    pub state_in: [u8; 32],
    pub state_out: [u8; 32],
    pub halted_out: bool,
}

impl Rv32imChunkStepPublic {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/chunk_fold_step_public");
        tr.append_message(
            b"neo.fold.next/rv32im/chunk_fold_step_public/program_digest",
            &self.program_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv32im/chunk_fold_step_public/meta",
            &[self.chunk_index, self.step_lo, self.step_hi, self.halted_out as u64],
        );
        tr.append_message(b"neo.fold.next/rv32im/chunk_fold_step_public/state_in", &self.state_in);
        tr.append_message(
            b"neo.fold.next/rv32im/chunk_fold_step_public/state_out",
            &self.state_out,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug)]
pub struct Rv32imChunkFoldFresh {
    pub public_chunk: PublicChunk,
    pub public_chunk_instance_digest: [F; 4],
    pub public_chunk_digest: [u8; 32],
    pub bridge_handoff_digest: [u8; 32],
    pub fresh_claims: Vec<CcsClaim<Commitment, F>>,
    pub fresh_witnesses: Vec<CcsWitness<F>>,
}

#[derive(Clone, Debug)]
pub struct Rv32imChunkFoldVerifierStepOutput {
    pub next_carry: Rv32imChunkFoldCarry,
    pub public_chunk_digest: [u8; 32],
    pub chunk_relation_digest: [u8; 32],
    pub step_public: Rv32imChunkStepPublic,
}

pub fn rv32im_chunk_fold_seed() -> [u8; 32] {
    fixed_shape_recursive_seed(b"neo.fold.next/rv32im/recursive_seed")
}

pub fn adapt_rv32im_chunk_to_fresh_ccs(handoff: &Rv32imVerifiedKernelChunkHandoff) -> Rv32imChunkFoldFresh {
    Rv32imChunkFoldFresh {
        public_chunk: handoff.public_chunk.clone(),
        public_chunk_instance_digest: handoff.public_chunk_instance_digest,
        public_chunk_digest: handoff.public_chunk_digest,
        bridge_handoff_digest: handoff.bridge_handoff.digest,
        fresh_claims: handoff
            .chunk_input
            .steps
            .iter()
            .map(|step| step.mcs.clone())
            .collect(),
        fresh_witnesses: handoff
            .chunk_input
            .steps
            .iter()
            .map(|step| step.witness.clone())
            .collect(),
    }
}

pub(crate) fn build_rv32im_chunk_step_public(
    program_digest: [u8; 32],
    chunk_index: usize,
    fresh: &Rv32imChunkFoldFresh,
    carry_in: &Rv32imChunkFoldCarry,
    carry_out: &Rv32imChunkFoldCarry,
    halted_out: bool,
) -> Rv32imChunkStepPublic {
    let step_lo = fresh.public_chunk.start_index as u64;
    let step_hi = step_lo + fresh.public_chunk.steps.len() as u64;
    Rv32imChunkStepPublic {
        program_digest,
        chunk_index: chunk_index as u64,
        step_lo,
        step_hi,
        state_in: carry_in.terminal_handle.0,
        state_out: carry_out.terminal_handle.0,
        halted_out,
    }
}

pub(crate) fn verify_rv32im_chunk_fold_verifier_step(
    program_digest: [u8; 32],
    chunk_index: usize,
    halted_out: bool,
    handoff: &Rv32imVerifiedKernelChunkHandoff,
    carry_in: &Rv32imChunkFoldCarry,
    replay_witness: &ChunkReplayWitness,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    log: &AjtaiSModule,
    optimized_cache: &OptimizedStructureCache,
) -> Result<Rv32imChunkFoldVerifierStepOutput, SimpleKernelError> {
    let fresh = adapt_rv32im_chunk_to_fresh_ccs(handoff);
    let me_input_accumulator_handle = crate::finalize::digest32_as_fields(
        crate::rv32im::final_relation::rv32im_chunk_fold_carry_recursive_accumulator_digest(carry_in),
    );
    let (next_main, public_chunk_digest, chunk_relation_digest) = verify_rv32im_chunk_relation_with_replay(
        chunk_index,
        handoff,
        &carry_in.main,
        me_input_accumulator_handle,
        replay_witness,
        transcript,
        params,
        structure,
        log,
        optimized_cache,
    )?;
    let next_carry = Rv32imChunkFoldCarry::from_main(
        next_main,
        Rv32imAccumulatorHandle(rv32im_step_handle(
            carry_in.terminal_handle.0,
            chunk_index,
            fresh.public_chunk.start_index,
            fresh.public_chunk.steps.len(),
            chunk_relation_digest,
        )),
    );
    let step_public =
        build_rv32im_chunk_step_public(program_digest, chunk_index, &fresh, carry_in, &next_carry, halted_out);
    Ok(Rv32imChunkFoldVerifierStepOutput {
        next_carry,
        public_chunk_digest,
        chunk_relation_digest,
        step_public,
    })
}

pub(crate) fn prove_rv32im_chunk_fold_verifier_step_with_perf(
    program_digest: [u8; 32],
    chunk_index: usize,
    halted_out: bool,
    handoff: &Rv32imVerifiedKernelChunkHandoff,
    carry_in: &Rv32imChunkFoldCarry,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    log: &AjtaiSModule,
    optimized_cache: &OptimizedStructureCache,
) -> Result<((ChunkReplayWitness, Rv32imChunkFoldVerifierStepOutput), ChunkProvePerf), SimpleKernelError> {
    let fresh = adapt_rv32im_chunk_to_fresh_ccs(handoff);
    let me_input_accumulator_handle = crate::finalize::digest32_as_fields(
        crate::rv32im::final_relation::rv32im_chunk_fold_carry_recursive_accumulator_digest(carry_in),
    );
    let ((replay_witness, next_main, public_chunk_digest, chunk_relation_digest), perf) =
        prove_rv32im_chunk_transition_with_perf(
            chunk_index,
            handoff,
            &carry_in.main,
            me_input_accumulator_handle,
            transcript,
            params,
            structure,
            log,
            optimized_cache,
        )?;
    let next_carry = Rv32imChunkFoldCarry::from_main(
        next_main,
        Rv32imAccumulatorHandle(rv32im_step_handle(
            carry_in.terminal_handle.0,
            chunk_index,
            fresh.public_chunk.start_index,
            fresh.public_chunk.steps.len(),
            chunk_relation_digest,
        )),
    );
    let step_public =
        build_rv32im_chunk_step_public(program_digest, chunk_index, &fresh, carry_in, &next_carry, halted_out);
    Ok((
        (
            replay_witness,
            Rv32imChunkFoldVerifierStepOutput {
                next_carry,
                public_chunk_digest,
                chunk_relation_digest,
                step_public,
            },
        ),
        perf,
    ))
}
