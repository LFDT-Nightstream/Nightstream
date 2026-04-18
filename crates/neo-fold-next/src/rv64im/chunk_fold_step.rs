//! Owns the explicit RV64IM one-chunk fold-verifier step reused by recursion and decider tracing.

use neo_ajtai::{AjtaiSModule, Commitment};
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness};
use neo_math::F;
use neo_params::NeoParams;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::chunk_relation::ChunkReplayWitness;
use crate::finalize::fixed_shape_recursive_seed;
use crate::proof::{Carry, ChunkProvePerf, PublicChunk};
use crate::rv64im::chunk_relation::{
    prove_rv64im_chunk_transition_with_perf, rv64im_step_handle, verify_rv64im_chunk_relation_with_replay,
};
use crate::rv64im::kernel::{Rv64imVerifiedKernelChunkHandoff, SimpleKernelError};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imAccumulatorHandle(pub [u8; 32]);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Rv64imChunkFoldCarry {
    pub main: Carry,
    pub terminal_handle: Rv64imAccumulatorHandle,
}

impl Rv64imChunkFoldCarry {
    pub fn seed() -> Self {
        Self {
            main: crate::rv64im::construction2_default::build_rv64im_main_recursion_canonical_zero_carry()
                .expect("canonical RV64IM chunk-fold seed carry must build"),
            terminal_handle: Rv64imAccumulatorHandle(rv64im_chunk_fold_seed()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imChunkStepPublic {
    pub program_digest: [u8; 32],
    pub chunk_index: u64,
    pub step_lo: u64,
    pub step_hi: u64,
    pub state_in: [u8; 32],
    pub state_out: [u8; 32],
    pub halted_out: bool,
}

impl Rv64imChunkStepPublic {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/chunk_fold_step_public");
        tr.append_message(
            b"neo.fold.next/rv64im/chunk_fold_step_public/program_digest",
            &self.program_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/chunk_fold_step_public/meta",
            &[self.chunk_index, self.step_lo, self.step_hi, self.halted_out as u64],
        );
        tr.append_message(b"neo.fold.next/rv64im/chunk_fold_step_public/state_in", &self.state_in);
        tr.append_message(
            b"neo.fold.next/rv64im/chunk_fold_step_public/state_out",
            &self.state_out,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug)]
pub struct Rv64imChunkFoldFresh {
    pub public_chunk: PublicChunk,
    pub public_chunk_instance_digest: [F; 4],
    pub public_chunk_digest: [u8; 32],
    pub bridge_handoff_digest: [u8; 32],
    pub fresh_claims: Vec<CcsClaim<Commitment, F>>,
    pub fresh_witnesses: Vec<CcsWitness<F>>,
}

#[derive(Clone, Debug)]
pub struct Rv64imChunkFoldVerifierStepOutput {
    pub next_carry: Rv64imChunkFoldCarry,
    pub public_chunk_digest: [u8; 32],
    pub chunk_relation_digest: [u8; 32],
    pub step_public: Rv64imChunkStepPublic,
}

pub fn rv64im_chunk_fold_seed() -> [u8; 32] {
    fixed_shape_recursive_seed(b"neo.fold.next/rv64im/recursive_seed")
}

pub fn adapt_rv64im_chunk_to_fresh_ccs(handoff: &Rv64imVerifiedKernelChunkHandoff) -> Rv64imChunkFoldFresh {
    Rv64imChunkFoldFresh {
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

pub(crate) fn build_rv64im_chunk_step_public(
    program_digest: [u8; 32],
    chunk_index: usize,
    fresh: &Rv64imChunkFoldFresh,
    carry_in: &Rv64imChunkFoldCarry,
    carry_out: &Rv64imChunkFoldCarry,
    halted_out: bool,
) -> Rv64imChunkStepPublic {
    let step_lo = fresh.public_chunk.start_index as u64;
    let step_hi = step_lo + fresh.public_chunk.steps.len() as u64;
    Rv64imChunkStepPublic {
        program_digest,
        chunk_index: chunk_index as u64,
        step_lo,
        step_hi,
        state_in: carry_in.terminal_handle.0,
        state_out: carry_out.terminal_handle.0,
        halted_out,
    }
}

pub(crate) fn verify_rv64im_chunk_fold_verifier_step(
    program_digest: [u8; 32],
    chunk_index: usize,
    halted_out: bool,
    handoff: &Rv64imVerifiedKernelChunkHandoff,
    carry_in: &Rv64imChunkFoldCarry,
    replay_witness: &ChunkReplayWitness,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    log: &AjtaiSModule,
    optimized_cache: &OptimizedStructureCache,
) -> Result<Rv64imChunkFoldVerifierStepOutput, SimpleKernelError> {
    let fresh = adapt_rv64im_chunk_to_fresh_ccs(handoff);
    let (next_main, public_chunk_digest, chunk_relation_digest) = verify_rv64im_chunk_relation_with_replay(
        chunk_index,
        handoff,
        &carry_in.main,
        replay_witness,
        transcript,
        params,
        structure,
        log,
        optimized_cache,
    )?;
    let next_carry = Rv64imChunkFoldCarry {
        main: next_main,
        terminal_handle: Rv64imAccumulatorHandle(rv64im_step_handle(
            carry_in.terminal_handle.0,
            chunk_index,
            fresh.public_chunk.start_index,
            fresh.public_chunk.steps.len(),
            chunk_relation_digest,
        )),
    };
    let step_public =
        build_rv64im_chunk_step_public(program_digest, chunk_index, &fresh, carry_in, &next_carry, halted_out);
    Ok(Rv64imChunkFoldVerifierStepOutput {
        next_carry,
        public_chunk_digest,
        chunk_relation_digest,
        step_public,
    })
}

pub(crate) fn prove_rv64im_chunk_fold_verifier_step_with_perf(
    program_digest: [u8; 32],
    chunk_index: usize,
    halted_out: bool,
    handoff: &Rv64imVerifiedKernelChunkHandoff,
    carry_in: &Rv64imChunkFoldCarry,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    log: &AjtaiSModule,
    optimized_cache: &OptimizedStructureCache,
) -> Result<((ChunkReplayWitness, Rv64imChunkFoldVerifierStepOutput), ChunkProvePerf), SimpleKernelError> {
    let fresh = adapt_rv64im_chunk_to_fresh_ccs(handoff);
    let ((replay_witness, next_main, public_chunk_digest, chunk_relation_digest), perf) =
        prove_rv64im_chunk_transition_with_perf(
            chunk_index,
            handoff,
            &carry_in.main,
            transcript,
            params,
            structure,
            log,
            optimized_cache,
        )?;
    let next_carry = Rv64imChunkFoldCarry {
        main: next_main,
        terminal_handle: Rv64imAccumulatorHandle(rv64im_step_handle(
            carry_in.terminal_handle.0,
            chunk_index,
            fresh.public_chunk.start_index,
            fresh.public_chunk.steps.len(),
            chunk_relation_digest,
        )),
    };
    let step_public =
        build_rv64im_chunk_step_public(program_digest, chunk_index, &fresh, carry_in, &next_carry, halted_out);
    Ok((
        (
            replay_witness,
            Rv64imChunkFoldVerifierStepOutput {
                next_carry,
                public_chunk_digest,
                chunk_relation_digest,
                step_public,
            },
        ),
        perf,
    ))
}
