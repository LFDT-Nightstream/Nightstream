//! Owns the current terminal recursion relation above the recursive-step proof-chain backend.
//!
//! This module owns the theorem-facing closure checks that bind the published
//! RV64IM accumulator statement to the current recursion proof carrier. It does
//! not own recursion-proof packaging or the recursive-step backend circuit.

use crate::chunk_relation::ChunkReplayWitness;
use crate::rv64im::chunk_fold_step::{adapt_rv64im_chunk_to_fresh_ccs, Rv64imChunkFoldCarry, Rv64imChunkStepPublic};
use crate::rv64im::chunk_relation::rv64im_chunk_replay_witness_digest;
use crate::rv64im::final_relation::{
    rv64im_chunk_fold_carried_transcript_snapshot, rv64im_chunk_fold_carry_recursive_accumulator_digest,
    rv64im_chunk_fold_state_instance_digest, rv64im_chunk_fold_transcript_snapshot_digest, Rv64imChunkFoldState,
    Rv64imChunkFoldTranscriptSnapshot, Rv64imRecursiveAccumulator, Rv64imTerminalChunkFoldWitness,
};
use crate::rv64im::kernel::Rv64imVerifiedKernelChunkHandoff;
use crate::rv64im::main_proof::Rv64imAccumulatorPublicStatement;
use crate::rv64im::main_recursion::{
    build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs, Rv64imEncodedPublicInput,
    Rv64imMainRecursionBackendStepStatement, Rv64imVerifierKeyFs,
};
use crate::rv64im::SimpleKernelError;
use neo_transcript::{Poseidon2Transcript, Transcript};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Rv64imMainRecursionFinalRelationPublicImage {
    x_last: Rv64imEncodedPublicInput,
    folded_accumulator_digest: [u8; 32],
    terminal_handle_digest: [u8; 32],
}

impl Rv64imMainRecursionFinalRelationPublicImage {
    fn from_accumulator_witness_with_vk_fs(
        vk_fs: &Rv64imVerifierKeyFs,
        accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
    ) -> Result<Self, SimpleKernelError> {
        let backend_statement = accumulator_witness.backend_statement_with_vk_fs(vk_fs)?;
        Ok(Self {
            x_last: backend_statement.x_out,
            folded_accumulator_digest: backend_statement.folded_accumulator_digest,
            terminal_handle_digest: accumulator_witness.running_final().terminal_handle.0,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Rv64imMainRecursionFinalRelationStatement {
    shape_digest: [u8; 32],
    vk_fs: Rv64imVerifierKeyFs,
    chunk_count: u64,
    pc_final: u64,
    accumulator_final: Rv64imRecursiveAccumulator,
    x_last: Rv64imEncodedPublicInput,
}

impl Rv64imMainRecursionFinalRelationStatement {
    fn from_published_statement(
        published_statement: &Rv64imAccumulatorPublicStatement,
    ) -> Result<Self, SimpleKernelError> {
        published_statement.validate()?;
        Ok(Self {
            shape_digest: published_statement.shape_digest(),
            vk_fs: published_statement.vk_fs().clone(),
            chunk_count: published_statement.expected_chunk_count()?,
            pc_final: published_statement.pc_final(),
            accumulator_final: published_statement.accumulator_final().clone(),
            x_last: published_statement.x_last().clone(),
        })
    }

    fn canonical_public_image(&self) -> Rv64imMainRecursionFinalRelationPublicImage {
        Rv64imMainRecursionFinalRelationPublicImage {
            x_last: self.x_last.clone(),
            folded_accumulator_digest: rv64im_chunk_fold_carry_recursive_accumulator_digest(
                &Rv64imChunkFoldCarry::from_main(
                    crate::proof::Carry {
                        claims: self.accumulator_final.final_main_claims.clone(),
                        witnesses: Vec::new(),
                    },
                    self.accumulator_final.terminal_handle,
                ),
            ),
            terminal_handle_digest: self.accumulator_final.terminal_handle.0,
        }
    }

    pub fn vk_fs(&self) -> &Rv64imVerifierKeyFs {
        &self.vk_fs
    }

    pub fn chunk_count(&self) -> u64 {
        self.chunk_count
    }

    pub fn pc_final(&self) -> u64 {
        self.pc_final
    }

    pub fn accumulator_final(&self) -> &Rv64imRecursiveAccumulator {
        &self.accumulator_final
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Rv64imMainRecursionAccumulatorWitness {
    public_statement_digest: [u8; 32],
    handoff: Rv64imVerifiedKernelChunkHandoff,
    running_last: Rv64imChunkFoldCarry,
    transcript_in: Rv64imChunkFoldTranscriptSnapshot,
    final_fold_witness: ChunkReplayWitness,
    running_final: Rv64imChunkFoldCarry,
    transcript_out: Rv64imChunkFoldTranscriptSnapshot,
    step_public: Rv64imChunkStepPublic,
    halted_out: bool,
    chunk_count: u64,
    pc_final: u64,
}

impl Rv64imMainRecursionAccumulatorWitness {
    fn terminal_chunk_fold_witness(&self) -> Rv64imTerminalChunkFoldWitness {
        Rv64imTerminalChunkFoldWitness {
            public_statement_digest: self.public_statement_digest,
            handoff: self.handoff.clone(),
            running_last: self.running_last.clone(),
            transcript_in: self.transcript_in.clone(),
            fresh_last: adapt_rv64im_chunk_to_fresh_ccs(&self.handoff),
            final_fold_witness: self.final_fold_witness.clone(),
            running_final: self.running_final.clone(),
            transcript_out: self.transcript_out.clone(),
            step_public: self.step_public.clone(),
            halted_out: self.halted_out,
        }
    }

    pub fn public_statement_digest(&self) -> [u8; 32] {
        self.public_statement_digest
    }

    pub fn handoff(&self) -> &Rv64imVerifiedKernelChunkHandoff {
        &self.handoff
    }

    pub fn running_last(&self) -> &Rv64imChunkFoldCarry {
        &self.running_last
    }

    pub fn transcript_in(&self) -> &Rv64imChunkFoldTranscriptSnapshot {
        &self.transcript_in
    }

    pub fn final_fold_witness(&self) -> &ChunkReplayWitness {
        &self.final_fold_witness
    }

    pub fn running_final(&self) -> &Rv64imChunkFoldCarry {
        &self.running_final
    }

    pub fn transcript_out(&self) -> &Rv64imChunkFoldTranscriptSnapshot {
        &self.transcript_out
    }

    pub fn step_public(&self) -> &Rv64imChunkStepPublic {
        &self.step_public
    }

    pub fn step_public_mut(&mut self) -> &mut Rv64imChunkStepPublic {
        &mut self.step_public
    }

    pub fn halted_out(&self) -> bool {
        self.halted_out
    }

    pub fn chunk_count(&self) -> u64 {
        self.chunk_count
    }

    pub fn pc_final(&self) -> u64 {
        self.pc_final
    }

    pub fn accumulator_final(&self) -> Rv64imRecursiveAccumulator {
        self.terminal_chunk_fold_witness().accumulator_final()
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_recursion_accumulator_witness");
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_accumulator_witness/version",
            b"v1",
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_accumulator_witness/public_statement_digest",
            &self.public_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_accumulator_witness/handoff_digest",
            &rv64im_verified_kernel_chunk_handoff_digest(&self.handoff),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_accumulator_witness/state_in_digest",
            &rv64im_chunk_fold_state_instance_digest(&Rv64imChunkFoldState {
                carry: self.running_last.clone(),
                transcript: self.transcript_in.clone(),
            }),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_accumulator_witness/final_fold_witness_digest",
            &rv64im_chunk_replay_witness_digest(&self.final_fold_witness),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_accumulator_witness/state_out_digest",
            &rv64im_chunk_fold_state_instance_digest(&Rv64imChunkFoldState {
                carry: self.running_final.clone(),
                transcript: rv64im_chunk_fold_carried_transcript_snapshot(&self.transcript_out),
            }),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_accumulator_witness/transcript_out_digest",
            &rv64im_chunk_fold_transcript_snapshot_digest(&self.transcript_out),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_recursion_accumulator_witness/step_public_digest",
            &self.step_public.expected_digest(),
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/main_recursion_accumulator_witness/meta",
            &[self.chunk_count, self.pc_final, self.halted_out as u64],
        );
        tr.digest32()
    }

    fn backend_statement_with_vk_fs(
        &self,
        vk_fs: &Rv64imVerifierKeyFs,
    ) -> Result<Rv64imMainRecursionBackendStepStatement, SimpleKernelError> {
        Ok(build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
            vk_fs,
            self.chunk_count,
            rv64im_chunk_fold_carry_recursive_accumulator_digest(self.running_final()),
            self.running_final().terminal_handle.0,
        ))
    }
}

fn rv64im_verified_kernel_chunk_handoff_digest(handoff: &Rv64imVerifiedKernelChunkHandoff) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/verified_kernel_chunk_handoff");
    tr.append_message(b"neo.fold.next/rv64im/verified_kernel_chunk_handoff/version", b"v1");
    tr.append_u64s(
        b"neo.fold.next/rv64im/verified_kernel_chunk_handoff/meta",
        &[
            handoff.chunk_input.start_index as u64,
            handoff.chunk_input.steps.len() as u64,
            handoff.public_chunk.start_index as u64,
            handoff.public_chunk.steps.len() as u64,
        ],
    );
    tr.append_message(
        b"neo.fold.next/rv64im/verified_kernel_chunk_handoff/public_chunk_digest",
        &handoff.public_chunk_digest,
    );
    tr.append_message(
        b"neo.fold.next/rv64im/verified_kernel_chunk_handoff/bridge_handoff_digest",
        &handoff.bridge_handoff.digest,
    );
    for digest in &handoff.prepared_step_digests {
        tr.append_message(
            b"neo.fold.next/rv64im/verified_kernel_chunk_handoff/prepared_step_digest",
            digest,
        );
    }
    tr.append_fields_raw(&handoff.public_chunk_instance_digest);
    tr.digest32()
}

pub(crate) fn build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs(
    vk_fs: &Rv64imVerifierKeyFs,
    chunk_count: u64,
    accumulator_final: &Rv64imRecursiveAccumulator,
) -> Result<Rv64imEncodedPublicInput, SimpleKernelError> {
    let folded_accumulator_digest =
        rv64im_chunk_fold_carry_recursive_accumulator_digest(&Rv64imChunkFoldCarry::from_main(
            crate::proof::Carry {
                claims: accumulator_final.final_main_claims.clone(),
                witnesses: Vec::new(),
            },
            accumulator_final.terminal_handle,
        ));
    Ok(build_rv64im_main_recursion_backend_statement_from_parts_with_vk_fs(
        vk_fs,
        chunk_count,
        folded_accumulator_digest,
        accumulator_final.terminal_handle.0,
    )
    .x_out)
}

fn build_rv64im_main_recursion_final_relation_statement(
    published_statement: &Rv64imAccumulatorPublicStatement,
) -> Result<Rv64imMainRecursionFinalRelationStatement, SimpleKernelError> {
    Rv64imMainRecursionFinalRelationStatement::from_published_statement(published_statement)
}

fn validate_rv64im_main_recursion_accumulator_witness_against_statement(
    final_relation_statement: &Rv64imMainRecursionFinalRelationStatement,
    accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
) -> Result<(), SimpleKernelError> {
    // This surface only binds the carried terminal accumulator witness back to
    // the published final-relation statement. The decider relation builder
    // owns the verifier-path terminal fold replay needed to recover the
    // terminal chunk relation digest for the live Goal 3 decider.
    if accumulator_witness.chunk_count() != final_relation_statement.chunk_count() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness chunk count does not match the published statement schedule"
                .into(),
        ));
    }
    if accumulator_witness.pc_final() != final_relation_statement.pc_final() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness pc_final does not match the published statement".into(),
        ));
    }
    if accumulator_witness.step_public().chunk_index + 1 != final_relation_statement.chunk_count() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion final relation terminal chunk index does not close the published statement chunk schedule"
                .into(),
        ));
    }
    if !accumulator_witness.halted_out() || !accumulator_witness.step_public().halted_out {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness does not close on a terminal halted chunk".into(),
        ));
    }
    if accumulator_witness.accumulator_final() != *final_relation_statement.accumulator_final() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness final accumulator does not match the published statement".into(),
        ));
    }
    if Rv64imMainRecursionFinalRelationPublicImage::from_accumulator_witness_with_vk_fs(
        final_relation_statement.vk_fs(),
        accumulator_witness,
    )? != final_relation_statement.canonical_public_image()
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness does not match the canonical final-relation statement public image"
                .into(),
        ));
    }
    if accumulator_witness.step_public().halted_out != accumulator_witness.halted_out() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main recursion accumulator witness halted flag does not match the terminal step public".into(),
        ));
    }
    Ok(())
}

pub fn validate_rv64im_main_recursion_accumulator_witness_against_published_statement(
    published_statement: &Rv64imAccumulatorPublicStatement,
    accumulator_witness: &Rv64imMainRecursionAccumulatorWitness,
) -> Result<(), SimpleKernelError> {
    let final_relation_statement = build_rv64im_main_recursion_final_relation_statement(published_statement)?;
    validate_rv64im_main_recursion_accumulator_witness_against_statement(&final_relation_statement, accumulator_witness)
}
