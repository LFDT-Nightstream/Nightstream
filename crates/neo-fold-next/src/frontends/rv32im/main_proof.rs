//! Owns the compact RV32IM published boundary: the published accumulator
//! statement, compressed main proof, and the local final-seam cache used by
//! build support.

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use crate::finalize::{digest32_as_fields, digest_fields_as_digest32, FixedShapeChunkSummary};
use crate::proof::FoldSchedule;
use crate::rv32im::chunk::step_ivc::{
    build_rv32im_chunk_step_ivc_relations, rv32im_chunk_step_ivc_initial_state_for_step_cap,
    Rv32imChunkStepIvcStatement,
};
use crate::rv32im::construction2::{
    build_rv32im_main_recursion_construction2_verified_step_statement_digest_from_step_statement,
    Rv32imMainRecursionConstruction2PublicBoundary,
};
use crate::rv32im::encoded_public_input::encoded_public_input_has_canonical_field_limb_bytes;
use crate::rv32im::f_prime::{
    build_rv32im_main_recursion_verifier_key_fs_for_step_cap, Rv32imEncodedPublicInput, Rv32imVerifierKeyFs,
    RV32IM_MAIN_RECURSION_TRIVIAL_PC,
};
use crate::rv32im::final_relation::{
    reconstruct_rv32im_final_statement_from_export_and_replay, rv32im_recursive_accumulator_instance_digest_from_parts,
    Rv32imChunkTransitionWitness, Rv32imFinalBuildProof, Rv32imFinalStatement, Rv32imRecursiveAccumulator,
};
use crate::rv32im::ivc::Rv32imIvcPublicImage;
use crate::rv32im::ivc::{build_rv32im_ivc_prover_state_from_relations, derive_rv32im_ivc_step_cap};
use crate::rv32im::ivc_snark::{prove_rv32im_ivc_snark_from_final_cached, Rv32imIvcSnark, Rv32imIvcSnarkProof};
use crate::rv32im::kernel::{Rv32imKernelExportProof, SimpleKernelError};
use crate::rv32im::recursion_spartan::build_rv32im_main_recursion_x_last_from_accumulator_with_vk_fs;

const RV32IM_CHUNK_SUMMARY_CHAIN_RAW_TAG: u64 = 0x7276_3634_6373756d;

fn rv32im_digest_chain_initial(raw_tag: u64) -> [u8; 32] {
    digest_fields_as_digest32(poseidon2_hash(&[F::from_u64(raw_tag)]))
}

fn rv32im_digest_chain_step(raw_tag: u64, current: [u8; 32], item: [u8; 32]) -> [u8; 32] {
    let mut preimage = Vec::with_capacity(1 + 4 + 4);
    preimage.push(F::from_u64(raw_tag));
    preimage.extend(digest32_as_fields(current));
    preimage.extend(digest32_as_fields(item));
    digest_fields_as_digest32(poseidon2_hash(&preimage))
}

fn rv32im_chunk_summary_chain_digest_from_summaries(chunk_summaries: &[FixedShapeChunkSummary]) -> [u8; 32] {
    let mut current = rv32im_digest_chain_initial(RV32IM_CHUNK_SUMMARY_CHAIN_RAW_TAG);
    for summary in chunk_summaries {
        current = rv32im_digest_chain_step(RV32IM_CHUNK_SUMMARY_CHAIN_RAW_TAG, current, summary.digest());
    }
    current
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32imMainFinalProofSurface {
    fold_schedule: FoldSchedule,
    semantic_step_count: u64,
    chunk_summary_count: u64,
    final_pc: u64,
    chunk_summary_chain_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv32imCompressedMainProof {
    published_statement: Rv32imPublishedStatement,
    ivc_snark: Rv32imIvcSnark,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Rv32imLocalFinalSeam {
    proof_digest: [u8; 32],
    kernel_export: Rv32imKernelExportProof,
    steps: Vec<Rv32imChunkTransitionWitness>,
}

impl Rv32imLocalFinalSeam {
    pub(crate) fn new(
        proof_digest: [u8; 32],
        kernel_export: Rv32imKernelExportProof,
        steps: Vec<Rv32imChunkTransitionWitness>,
    ) -> Self {
        Self {
            proof_digest,
            kernel_export,
            steps,
        }
    }

    pub(crate) fn kernel_export(&self) -> &Rv32imKernelExportProof {
        &self.kernel_export
    }

    pub(crate) fn rebuild(&self) -> Result<(Rv32imFinalStatement, Rv32imFinalBuildProof), SimpleKernelError> {
        reconstruct_rv32im_final_statement_from_export_and_replay(
            self.kernel_export.public_statement_digest(),
            &self.kernel_export,
            &self.steps,
        )
    }

    pub(crate) fn rebuild_final_statement(&self) -> Result<Rv32imFinalStatement, SimpleKernelError> {
        self.rebuild().map(|(statement, _)| statement)
    }

    pub(crate) fn rebuild_final_proof(&self) -> Result<Rv32imFinalBuildProof, SimpleKernelError> {
        self.rebuild().map(|(_, proof)| proof)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv32imAccumulatorPublicStatement {
    shape_digest: [u8; 32],
    vk_fs: Rv32imVerifierKeyFs,
    fold_schedule: FoldSchedule,
    step_count: u64,
    pc_final: u64,
    accumulator_final: Rv32imRecursiveAccumulator,
    x_last: Rv32imEncodedPublicInput,
    construction2_u_i: Rv32imMainRecursionConstruction2PublicBoundary,
    terminal_bridge_handoff_digest: [u8; 32],
    terminal_verified_step_statement_digest: [u8; 32],
    terminal_step_statement: Rv32imChunkStepIvcStatement,
}

pub type Rv32imPublishedStatement = Rv32imAccumulatorPublicStatement;

impl Rv32imMainFinalProofSurface {
    pub fn from_final_proof(statement: &Rv32imFinalStatement, proof: &Rv32imFinalBuildProof, final_pc: u64) -> Self {
        Self {
            fold_schedule: statement.folded.fold_schedule,
            semantic_step_count: statement.folded.semantic_step_count,
            chunk_summary_count: proof.chunk_summaries.len() as u64,
            final_pc,
            chunk_summary_chain_digest: rv32im_chunk_summary_chain_digest_from_summaries(&proof.chunk_summaries),
        }
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/main_final_surface");
        tr.append_message(b"neo.fold.next/nightstream/rv32im/main_final_surface/version", b"v9");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/main_final_surface/counts",
            &[self.semantic_step_count, self.chunk_summary_count, self.final_pc],
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/main_final_surface/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/main_final_surface/chunk_summary_chain_digest",
            &self.chunk_summary_chain_digest,
        );
        tr.digest32()
    }

    pub fn chunk_summary_count(&self) -> u64 {
        self.chunk_summary_count
    }

    pub fn chunk_summary_chain_digest(&self) -> [u8; 32] {
        self.chunk_summary_chain_digest
    }

    pub fn chunk_summary_chain_digest_mut(&mut self) -> &mut [u8; 32] {
        &mut self.chunk_summary_chain_digest
    }

    pub fn fold_schedule(&self) -> FoldSchedule {
        self.fold_schedule
    }

    pub fn semantic_step_count(&self) -> u64 {
        self.semantic_step_count
    }

    pub fn final_pc(&self) -> u64 {
        self.final_pc
    }
}

impl PartialEq for Rv32imMainFinalProofSurface {
    fn eq(&self, other: &Self) -> bool {
        self.fold_schedule == other.fold_schedule
            && self.semantic_step_count == other.semantic_step_count
            && self.chunk_summary_count == other.chunk_summary_count
            && self.final_pc == other.final_pc
            && self.chunk_summary_chain_digest == other.chunk_summary_chain_digest
    }
}

impl Rv32imAccumulatorPublicStatement {
    fn expected_chunk_count_from_parts(fold_schedule: FoldSchedule, step_count: u64) -> Result<u64, SimpleKernelError> {
        let step_count = usize::try_from(step_count).map_err(|_| {
            SimpleKernelError::Bridge(
                "RV32IM published accumulator statement step_count does not fit into the local chunk scheduler".into(),
            )
        })?;
        fold_schedule
            .chunk_count(step_count)
            .map(|count| count as u64)
            .map_err(|err| {
                SimpleKernelError::Bridge(
                    format!(
                        "RV32IM published accumulator statement fold schedule is inconsistent with step_count: {err}"
                    )
                    .into(),
                )
            })
    }

    fn from_final_surface_with_terminal_step_statement(
        final_statement: &Rv32imFinalStatement,
        final_surface: &Rv32imMainFinalProofSurface,
        construction2_u_i: Rv32imMainRecursionConstruction2PublicBoundary,
        terminal_bridge_handoff_digest: [u8; 32],
        terminal_step_statement: Rv32imChunkStepIvcStatement,
    ) -> Result<Self, SimpleKernelError> {
        let fold_schedule = final_surface.fold_schedule();
        let step_count = final_surface.semantic_step_count();
        let step_cap = derive_rv32im_ivc_step_cap(
            fold_schedule,
            usize::try_from(step_count).map_err(|_| {
                SimpleKernelError::Bridge(
                    "RV32IM published accumulator statement step_count does not fit into the native step-cap model"
                        .into(),
                )
            })?,
        )?;
        let vk_fs = build_rv32im_main_recursion_verifier_key_fs_for_step_cap(step_cap)?;
        let accumulator_final = final_statement.folded.final_accumulator.clone();
        let chunk_count = Self::expected_chunk_count_from_parts(fold_schedule, step_count)?;
        let x_last =
            build_rv32im_main_recursion_x_last_from_accumulator_with_vk_fs(&vk_fs, chunk_count, &accumulator_final)?;
        let terminal_verified_step_statement_digest =
            build_rv32im_main_recursion_construction2_verified_step_statement_digest_from_step_statement(
                &terminal_step_statement,
            )?;
        Ok(Self {
            shape_digest: vk_fs.main_lane_shape_digest,
            vk_fs,
            fold_schedule,
            step_count,
            pc_final: final_surface.final_pc(),
            accumulator_final,
            x_last,
            construction2_u_i,
            terminal_bridge_handoff_digest,
            terminal_verified_step_statement_digest,
            terminal_step_statement,
        })
    }

    /// Builds the published accumulator from final build artifacts.
    ///
    /// This is prover/build support. Final acceptance is the compressed proof
    /// verifier, not this local artifact construction path.
    pub fn from_final_artifacts(
        final_statement: &Rv32imFinalStatement,
        final_proof: &Rv32imFinalBuildProof,
        final_pc: u64,
    ) -> Result<Self, SimpleKernelError> {
        let final_surface = Rv32imMainFinalProofSurface::from_final_proof(final_statement, final_proof, final_pc);
        let relations = build_rv32im_chunk_step_ivc_relations(final_statement, final_proof)?;
        let step_cap = derive_rv32im_ivc_step_cap(
            final_statement.folded.fold_schedule,
            usize::try_from(final_statement.folded.semantic_step_count).map_err(|_| {
                SimpleKernelError::Bridge(
                    "RV32IM published accumulator statement step_count does not fit into the native IVC step-cap model"
                        .into(),
                )
            })?,
        )?;
        let ivc_state = build_rv32im_ivc_prover_state_from_relations(&relations, step_cap)?;
        let construction2_u_i =
            Rv32imMainRecursionConstruction2PublicBoundary::from_fresh_instance(ivc_state.construction2_u_i());
        let terminal_relation = relations
            .last()
            .ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV32IM published accumulator statement requires terminal chunk statement metadata".into(),
                )
            })?
            .clone();
        Self::from_final_surface_with_terminal_step_statement(
            final_statement,
            &final_surface,
            construction2_u_i,
            terminal_relation.witness.handoff.bridge_handoff.digest,
            terminal_relation.statement,
        )
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/accumulator_public_statement");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/accumulator_public_statement/version",
            b"v13",
        );
        let canonical_folded_accumulator_digest = self.canonical_folded_accumulator_digest();
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/accumulator_public_statement/shape_digest",
            &self.shape_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/accumulator_public_statement/vk_fs_digest",
            &self.vk_fs.expected_digest(),
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/accumulator_public_statement/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/accumulator_public_statement/counts",
            &[self.step_count, self.pc_final],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/accumulator_public_statement/accumulator_final_digest",
            &canonical_folded_accumulator_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/accumulator_public_statement/x_last",
            &self.x_last.bytes(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/accumulator_public_statement/construction2_u_i",
            &self.construction2_u_i.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/accumulator_public_statement/terminal_bridge_handoff_digest",
            &self.terminal_bridge_handoff_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/accumulator_public_statement/terminal_verified_step_statement_digest",
            &self.terminal_verified_step_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/accumulator_public_statement/terminal_step_statement",
            &self.terminal_step_statement.expected_digest(),
        );
        tr.digest32()
    }

    pub fn expected_chunk_count(&self) -> Result<u64, SimpleKernelError> {
        Self::expected_chunk_count_from_parts(self.fold_schedule, self.step_count)
    }

    pub fn shape_digest(&self) -> [u8; 32] {
        self.shape_digest
    }

    pub fn shape_digest_mut(&mut self) -> &mut [u8; 32] {
        &mut self.shape_digest
    }

    pub fn vk_fs(&self) -> &Rv32imVerifierKeyFs {
        &self.vk_fs
    }

    pub fn vk_fs_mut(&mut self) -> &mut Rv32imVerifierKeyFs {
        &mut self.vk_fs
    }

    pub fn fold_schedule(&self) -> FoldSchedule {
        self.fold_schedule
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn pc_final(&self) -> u64 {
        self.pc_final
    }

    pub fn pc_final_mut(&mut self) -> &mut u64 {
        &mut self.pc_final
    }

    pub fn accumulator_final(&self) -> &Rv32imRecursiveAccumulator {
        &self.accumulator_final
    }

    pub fn accumulator_final_mut(&mut self) -> &mut Rv32imRecursiveAccumulator {
        &mut self.accumulator_final
    }

    pub fn canonical_terminal_handle_digest(&self) -> [u8; 32] {
        self.accumulator_final.terminal_handle.0
    }

    pub fn canonical_folded_accumulator_digest(&self) -> [u8; 32] {
        rv32im_recursive_accumulator_instance_digest_from_parts(
            &self.accumulator_final.final_main_claims,
            self.accumulator_final.terminal_handle.0,
        )
    }

    pub fn x_last(&self) -> &Rv32imEncodedPublicInput {
        &self.x_last
    }

    pub fn x_last_mut(&mut self) -> &mut Rv32imEncodedPublicInput {
        &mut self.x_last
    }

    pub fn construction2_u_i(&self) -> &Rv32imMainRecursionConstruction2PublicBoundary {
        &self.construction2_u_i
    }

    pub fn construction2_u_i_mut(&mut self) -> &mut Rv32imMainRecursionConstruction2PublicBoundary {
        &mut self.construction2_u_i
    }

    pub fn terminal_step_statement(&self) -> &Rv32imChunkStepIvcStatement {
        &self.terminal_step_statement
    }

    pub fn terminal_step_statement_mut(&mut self) -> &mut Rv32imChunkStepIvcStatement {
        &mut self.terminal_step_statement
    }

    pub fn terminal_bridge_handoff_digest(&self) -> [u8; 32] {
        self.terminal_bridge_handoff_digest
    }

    pub fn terminal_bridge_handoff_digest_mut(&mut self) -> &mut [u8; 32] {
        &mut self.terminal_bridge_handoff_digest
    }

    pub fn terminal_verified_step_statement_digest(&self) -> [u8; 32] {
        self.terminal_verified_step_statement_digest
    }

    pub fn validate(&self) -> Result<(), SimpleKernelError> {
        let expected_vk_fs = build_rv32im_main_recursion_verifier_key_fs_for_step_cap(derive_rv32im_ivc_step_cap(
            self.fold_schedule,
            usize::try_from(self.step_count).map_err(|_| {
                SimpleKernelError::Bridge(
                    "RV32IM published accumulator statement step_count does not fit into the native step-cap model"
                        .into(),
                )
            })?,
        )?)?;
        if self.vk_fs != expected_vk_fs {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement verifier key fs does not match the canonical recursion verifier key"
                    .into(),
            ));
        }
        if self.shape_digest != self.vk_fs.main_lane_shape_digest {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement shape_digest does not match the carried recursion verifier key fs"
                    .into(),
            ));
        }
        let expected_chunk_count = self.expected_chunk_count()?;
        let expected_x_last = build_rv32im_main_recursion_x_last_from_accumulator_with_vk_fs(
            &self.vk_fs,
            expected_chunk_count,
            &self.accumulator_final,
        )?;
        if self.x_last != expected_x_last {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement x_last does not match the Construction-2 final instance hash"
                    .into(),
            ));
        }
        if self.construction2_u_i.x_i != self.x_last {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement Construction-2 u_i.x_i does not match x_last".into(),
            ));
        }
        if !encoded_public_input_has_canonical_field_limb_bytes(&self.x_last) {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement x_last is not a canonical four-limb field encoding".into(),
            ));
        }
        if !self.construction2_u_i.has_canonical_commitment_shape() {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement Construction-2 u_i commitment shape is not canonical".into(),
            ));
        }
        if self.construction2_u_i.commitment_digest != self.construction2_u_i.expected_commitment_digest() {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement Construction-2 u_i commitment digest does not bind commitment data"
                    .into(),
            ));
        }
        if self.construction2_u_i.fresh_instance_digest != self.construction2_u_i.expected_fresh_instance_digest() {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement Construction-2 u_i digest does not bind commitment and x_last"
                    .into(),
            ));
        }
        if self
            .terminal_step_statement
            .step_public
            .chunk_index
            .checked_add(1)
            != Some(expected_chunk_count)
        {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement terminal chunk index does not close the published chunk schedule"
                    .into(),
            ));
        }
        if self.terminal_step_statement.step_public.step_hi != self.step_count {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement terminal step_hi does not close the published step_count"
                    .into(),
            ));
        }
        if self.terminal_step_statement.step_public.step_lo != self.terminal_step_statement.chunk_summary.start_index {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement terminal step_lo does not match the terminal chunk summary"
                    .into(),
            ));
        }
        if self.terminal_step_statement.chunk_summary.public_step_count == 0 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement terminal chunk must carry at least one public step".into(),
            ));
        }
        let Some(summary_step_hi) = self
            .terminal_step_statement
            .chunk_summary
            .start_index
            .checked_add(self.terminal_step_statement.chunk_summary.public_step_count)
        else {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement terminal chunk summary overflows the step domain".into(),
            ));
        };
        if summary_step_hi != self.terminal_step_statement.step_public.step_hi {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement terminal chunk summary does not match the terminal step span"
                    .into(),
            ));
        }
        if !self.terminal_step_statement.step_public.halted_out {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement terminal step must close on a halted chunk".into(),
            ));
        }
        if self.terminal_step_statement.step_public.state_out != self.accumulator_final.terminal_handle.0 {
            return Err(SimpleKernelError::Bridge(
                "RV32IM published accumulator statement terminal state_out does not match the authoritative final accumulator terminal handle"
                    .into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn build_rv32im_ivc_public_image_from_published_statement(
    published_statement: &Rv32imAccumulatorPublicStatement,
) -> Result<Rv32imIvcPublicImage, SimpleKernelError> {
    published_statement.validate()?;
    Ok(Rv32imIvcPublicImage {
        vk_fs_digest: published_statement.vk_fs().expected_digest(),
        chunk_count: published_statement.expected_chunk_count()?,
        step_count: published_statement.step_count(),
        z_0: rv32im_chunk_step_ivc_initial_state_for_step_cap(published_statement.vk_fs().step_cap()?)
            .carry
            .terminal_handle
            .0,
        z_i: published_statement.canonical_terminal_handle_digest(),
        // The native IVC carrier exposes the recursion control-lane PC, not the
        // architectural final program counter. The published statement still
        // binds the authoritative architectural final PC separately via
        // `pc_final`.
        pc: RV32IM_MAIN_RECURSION_TRIVIAL_PC,
        x_i: published_statement.x_last().clone(),
        construction2_u_i: published_statement.construction2_u_i().clone(),
        folded_accumulator_digest: published_statement.canonical_folded_accumulator_digest(),
        terminal_bridge_handoff_digest: published_statement.terminal_bridge_handoff_digest(),
        terminal_verified_step_statement_digest: published_statement.terminal_verified_step_statement_digest(),
        terminal_statement: Some(published_statement.terminal_step_statement().clone()),
    })
}

impl Rv32imCompressedMainProof {
    /// Builds a compressed proof from final build artifacts.
    ///
    /// This constructs prover-side inputs for the Spartan proof. Consumers
    /// must derive the public image and call `Rv32imIvcSnark::verify`; this
    /// constructor is not a verifier.
    pub fn from_final_artifacts(
        statement: &Rv32imFinalStatement,
        proof: &Rv32imFinalBuildProof,
        final_pc: u64,
    ) -> Result<Self, SimpleKernelError> {
        let mut published_statement =
            Rv32imAccumulatorPublicStatement::from_final_artifacts(statement, proof, final_pc)?;
        let ivc_snark = prove_rv32im_ivc_snark_from_final_cached(statement, proof)?;
        *published_statement.construction2_u_i_mut() = ivc_snark.public_image().construction2_u_i.clone();
        published_statement.terminal_verified_step_statement_digest = ivc_snark
            .public_image()
            .terminal_verified_step_statement_digest;
        Ok(Self {
            published_statement,
            ivc_snark,
        })
    }

    pub fn published_statement(&self) -> &Rv32imPublishedStatement {
        &self.published_statement
    }

    pub fn published_statement_mut(&mut self) -> &mut Rv32imPublishedStatement {
        &mut self.published_statement
    }

    pub fn ivc_snark(&self) -> &Rv32imIvcSnark {
        &self.ivc_snark
    }

    pub fn ivc_snark_mut(&mut self) -> &mut Rv32imIvcSnark {
        &mut self.ivc_snark
    }

    pub fn ivc_recursion_snark_proof(&self) -> &Rv32imIvcSnarkProof {
        self.ivc_snark.proof()
    }

    pub fn ivc_recursion_snark_proof_mut(&mut self) -> &mut Rv32imIvcSnarkProof {
        self.ivc_snark.proof_mut()
    }

    pub fn expected_ivc_public_image(&self) -> Result<Rv32imIvcPublicImage, SimpleKernelError> {
        build_rv32im_ivc_public_image_from_published_statement(&self.published_statement)
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/compressed_main_proof");
        tr.append_message(b"neo.fold.next/nightstream/rv32im/compressed_main_proof/version", b"v3");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/compressed_main_proof/published_statement_digest",
            &self.published_statement.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/compressed_main_proof/public_image_digest",
            &self.ivc_snark.public_image().expected_digest(),
        );
        let proof_bytes = bincode::serialize(self.ivc_snark.proof())
            .expect("RV32IM compressed main proof digest requires serializable recursion SNARK proof");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/compressed_main_proof/ivc_recursion_snark_proof",
            &proof_bytes,
        );
        tr.digest32()
    }

    pub fn binding_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/compressed_main_proof_binding");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/compressed_main_proof_binding/version",
            b"v3",
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/compressed_main_proof_binding/published_statement_digest",
            &self.published_statement.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/compressed_main_proof_binding/public_image_digest",
            &self.ivc_snark.public_image().expected_digest(),
        );
        let proof_bytes = bincode::serialize(self.ivc_snark.proof())
            .expect("RV32IM compressed main proof binding requires serializable recursion SNARK proof");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/compressed_main_proof_binding/ivc_recursion_snark_proof",
            &proof_bytes,
        );
        tr.digest32()
    }
}
