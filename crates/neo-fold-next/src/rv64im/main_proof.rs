//! Owns the compact RV64IM published boundary: the published accumulator
//! statement, compressed main proof, and the local final-seam cache used by
//! build support.

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use crate::finalize::{digest32_as_fields, digest_fields_as_digest32, FixedShapeChunkSummary};
use crate::proof::FoldSchedule;
use crate::rv64im::chunk_step_ivc::{
    build_rv64im_chunk_step_ivc_relations, rv64im_chunk_step_ivc_initial_state, Rv64imChunkStepIvcStatement,
};
use crate::rv64im::final_relation::{
    reconstruct_rv64im_final_statement_from_export_and_replay, rv64im_recursive_accumulator_instance_digest_from_parts,
    Rv64imChunkTransitionWitness, Rv64imFinalBuildProof, Rv64imFinalStatement, Rv64imRecursiveAccumulator,
};
use crate::rv64im::ivc::derive_rv64im_ivc_step_cap;
use crate::rv64im::ivc::Rv64imIvcPublicImage;
use crate::rv64im::ivc_snark::{
    prove_rv64im_ivc_snark_from_final_cached, Rv64imIvcSnark, Rv64imIvcSnarkProof, Rv64imIvcSnarkVerifierKey,
};
use crate::rv64im::kernel::{Rv64imKernelExportProof, SimpleKernelError};
use crate::rv64im::main_recursion::{
    build_rv64im_main_recursion_verifier_key_fs_for_step_cap, Rv64imEncodedPublicInput, Rv64imVerifierKeyFs,
};
use crate::rv64im::recursion_spartan::build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs;

const RV64IM_CHUNK_SUMMARY_CHAIN_RAW_TAG: u64 = 0x7276_3634_6373756d;

fn rv64im_digest_chain_initial(raw_tag: u64) -> [u8; 32] {
    digest_fields_as_digest32(poseidon2_hash(&[F::from_u64(raw_tag)]))
}

fn rv64im_digest_chain_step(raw_tag: u64, current: [u8; 32], item: [u8; 32]) -> [u8; 32] {
    let mut preimage = Vec::with_capacity(1 + 4 + 4);
    preimage.push(F::from_u64(raw_tag));
    preimage.extend(digest32_as_fields(current));
    preimage.extend(digest32_as_fields(item));
    digest_fields_as_digest32(poseidon2_hash(&preimage))
}

fn rv64im_chunk_summary_chain_digest_from_summaries(chunk_summaries: &[FixedShapeChunkSummary]) -> [u8; 32] {
    let mut current = rv64im_digest_chain_initial(RV64IM_CHUNK_SUMMARY_CHAIN_RAW_TAG);
    for summary in chunk_summaries {
        current = rv64im_digest_chain_step(RV64IM_CHUNK_SUMMARY_CHAIN_RAW_TAG, current, summary.digest());
    }
    current
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imMainFinalProofSurface {
    fold_schedule: FoldSchedule,
    semantic_step_count: u64,
    chunk_summary_count: u64,
    final_pc: u64,
    chunk_summary_chain_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imCompressedMainProof {
    published_statement: Rv64imPublishedStatement,
    ivc_snark: Rv64imIvcSnark,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Rv64imLocalFinalSeam {
    proof_digest: [u8; 32],
    kernel_export: Rv64imKernelExportProof,
    steps: Vec<Rv64imChunkTransitionWitness>,
}

impl Rv64imLocalFinalSeam {
    pub(crate) fn new(
        proof_digest: [u8; 32],
        kernel_export: Rv64imKernelExportProof,
        steps: Vec<Rv64imChunkTransitionWitness>,
    ) -> Self {
        Self {
            proof_digest,
            kernel_export,
            steps,
        }
    }

    pub(crate) fn kernel_export(&self) -> &Rv64imKernelExportProof {
        &self.kernel_export
    }

    pub(crate) fn rebuild(&self) -> Result<(Rv64imFinalStatement, Rv64imFinalBuildProof), SimpleKernelError> {
        reconstruct_rv64im_final_statement_from_export_and_replay(
            self.kernel_export.public_statement_digest(),
            &self.kernel_export,
            &self.steps,
        )
    }

    pub(crate) fn rebuild_final_statement(&self) -> Result<Rv64imFinalStatement, SimpleKernelError> {
        self.rebuild().map(|(statement, _)| statement)
    }

    pub(crate) fn rebuild_final_proof(&self) -> Result<Rv64imFinalBuildProof, SimpleKernelError> {
        self.rebuild().map(|(_, proof)| proof)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imAccumulatorPublicStatement {
    shape_digest: [u8; 32],
    vk_fs: Rv64imVerifierKeyFs,
    fold_schedule: FoldSchedule,
    step_count: u64,
    pc_final: u64,
    accumulator_final: Rv64imRecursiveAccumulator,
    x_last: Rv64imEncodedPublicInput,
    terminal_step_statement: Rv64imChunkStepIvcStatement,
}

pub type Rv64imPublishedStatement = Rv64imAccumulatorPublicStatement;

impl Rv64imMainFinalProofSurface {
    pub fn from_final_proof(statement: &Rv64imFinalStatement, proof: &Rv64imFinalBuildProof, final_pc: u64) -> Self {
        Self {
            fold_schedule: statement.folded.fold_schedule,
            semantic_step_count: statement.folded.semantic_step_count,
            chunk_summary_count: proof.chunk_summaries.len() as u64,
            final_pc,
            chunk_summary_chain_digest: rv64im_chunk_summary_chain_digest_from_summaries(&proof.chunk_summaries),
        }
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/main_final_surface");
        tr.append_message(b"neo.fold.next/nightstream/rv64im/main_final_surface/version", b"v9");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/main_final_surface/counts",
            &[self.semantic_step_count, self.chunk_summary_count, self.final_pc],
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/main_final_surface/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/main_final_surface/chunk_summary_chain_digest",
            &self.chunk_summary_chain_digest,
        );
        tr.digest32()
    }

    pub fn validate_against_final_statement(
        &self,
        final_statement: &Rv64imFinalStatement,
    ) -> Result<(), SimpleKernelError> {
        if final_statement.folded.fold_schedule != self.fold_schedule {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Nightstream main proof fold schedule does not match the carried final statement".into(),
            ));
        }
        if final_statement.folded.semantic_step_count != self.semantic_step_count {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Nightstream main proof semantic step count does not match the carried final statement".into(),
            ));
        }
        if final_statement.folded.chunk_count != self.chunk_summary_count {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Nightstream main proof chunk-summary count does not match the carried final statement".into(),
            ));
        }
        Ok(())
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

impl PartialEq for Rv64imMainFinalProofSurface {
    fn eq(&self, other: &Self) -> bool {
        self.fold_schedule == other.fold_schedule
            && self.semantic_step_count == other.semantic_step_count
            && self.chunk_summary_count == other.chunk_summary_count
            && self.final_pc == other.final_pc
            && self.chunk_summary_chain_digest == other.chunk_summary_chain_digest
    }
}

impl Rv64imAccumulatorPublicStatement {
    fn expected_chunk_count_from_parts(fold_schedule: FoldSchedule, step_count: u64) -> Result<u64, SimpleKernelError> {
        let step_count = usize::try_from(step_count).map_err(|_| {
            SimpleKernelError::Bridge(
                "RV64IM published accumulator statement step_count does not fit into the local chunk scheduler".into(),
            )
        })?;
        fold_schedule
            .chunk_count(step_count)
            .map(|count| count as u64)
            .map_err(|err| {
                SimpleKernelError::Bridge(
                    format!(
                        "RV64IM published accumulator statement fold schedule is inconsistent with step_count: {err}"
                    )
                    .into(),
                )
            })
    }

    fn from_final_surface_with_terminal_step_statement(
        final_statement: &Rv64imFinalStatement,
        final_surface: &Rv64imMainFinalProofSurface,
        terminal_step_statement: Rv64imChunkStepIvcStatement,
    ) -> Result<Self, SimpleKernelError> {
        let fold_schedule = final_surface.fold_schedule();
        let step_count = final_surface.semantic_step_count();
        let step_cap = derive_rv64im_ivc_step_cap(
            fold_schedule,
            usize::try_from(step_count).map_err(|_| {
                SimpleKernelError::Bridge(
                    "RV64IM published accumulator statement step_count does not fit into the native step-cap model"
                        .into(),
                )
            })?,
        )?;
        let vk_fs = build_rv64im_main_recursion_verifier_key_fs_for_step_cap(step_cap)?;
        let accumulator_final = final_statement.folded.final_accumulator.clone();
        let chunk_count = Self::expected_chunk_count_from_parts(fold_schedule, step_count)?;
        let x_last =
            build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs(&vk_fs, chunk_count, &accumulator_final)?;
        Ok(Self {
            shape_digest: vk_fs.main_lane_shape_digest,
            vk_fs,
            fold_schedule,
            step_count,
            pc_final: final_surface.final_pc(),
            accumulator_final,
            x_last,
            terminal_step_statement,
        })
    }

    pub fn from_verified_final_seam(
        final_statement: &Rv64imFinalStatement,
        final_proof: &Rv64imFinalBuildProof,
        final_pc: u64,
    ) -> Result<Self, SimpleKernelError> {
        let final_surface = Rv64imMainFinalProofSurface::from_final_proof(final_statement, final_proof, final_pc);
        let terminal_step_statement = build_rv64im_chunk_step_ivc_relations(final_statement, final_proof)?
            .last()
            .ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV64IM published accumulator statement requires a terminal chunk-step relation".into(),
                )
            })?
            .statement
            .clone();
        Self::from_final_surface_with_terminal_step_statement(final_statement, &final_surface, terminal_step_statement)
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/accumulator_public_statement");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/accumulator_public_statement/version",
            b"v10",
        );
        let canonical_folded_accumulator_digest = self.canonical_folded_accumulator_digest();
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/accumulator_public_statement/shape_digest",
            &self.shape_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/accumulator_public_statement/vk_fs_digest",
            &self.vk_fs.expected_digest(),
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/accumulator_public_statement/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/accumulator_public_statement/counts",
            &[self.step_count, self.pc_final],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/accumulator_public_statement/accumulator_final_digest",
            &canonical_folded_accumulator_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/accumulator_public_statement/x_last",
            &self.x_last.bytes(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/accumulator_public_statement/terminal_step_statement",
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

    pub fn vk_fs(&self) -> &Rv64imVerifierKeyFs {
        &self.vk_fs
    }

    pub fn vk_fs_mut(&mut self) -> &mut Rv64imVerifierKeyFs {
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

    pub fn accumulator_final(&self) -> &Rv64imRecursiveAccumulator {
        &self.accumulator_final
    }

    pub fn accumulator_final_mut(&mut self) -> &mut Rv64imRecursiveAccumulator {
        &mut self.accumulator_final
    }

    pub fn canonical_terminal_handle_digest(&self) -> [u8; 32] {
        self.accumulator_final.terminal_handle.0
    }

    pub fn canonical_folded_accumulator_digest(&self) -> [u8; 32] {
        rv64im_recursive_accumulator_instance_digest_from_parts(
            &self.accumulator_final.final_main_claims,
            self.accumulator_final.terminal_handle.0,
        )
    }

    pub fn x_last(&self) -> &Rv64imEncodedPublicInput {
        &self.x_last
    }

    pub fn x_last_mut(&mut self) -> &mut Rv64imEncodedPublicInput {
        &mut self.x_last
    }

    pub fn terminal_step_statement(&self) -> &Rv64imChunkStepIvcStatement {
        &self.terminal_step_statement
    }

    pub fn terminal_step_statement_mut(&mut self) -> &mut Rv64imChunkStepIvcStatement {
        &mut self.terminal_step_statement
    }

    pub fn validate(&self) -> Result<(), SimpleKernelError> {
        let expected_vk_fs = build_rv64im_main_recursion_verifier_key_fs_for_step_cap(derive_rv64im_ivc_step_cap(
            self.fold_schedule,
            usize::try_from(self.step_count).map_err(|_| {
                SimpleKernelError::Bridge(
                    "RV64IM published accumulator statement step_count does not fit into the native step-cap model"
                        .into(),
                )
            })?,
        )?)?;
        if self.vk_fs != expected_vk_fs {
            return Err(SimpleKernelError::Bridge(
                "RV64IM published accumulator statement verifier key fs does not match the canonical recursion verifier key"
                    .into(),
            ));
        }
        if self.shape_digest != self.vk_fs.main_lane_shape_digest {
            return Err(SimpleKernelError::Bridge(
                "RV64IM published accumulator statement shape_digest does not match the carried recursion verifier key fs"
                    .into(),
            ));
        }
        let expected_chunk_count = self.expected_chunk_count()?;
        if self
            .terminal_step_statement
            .step_public
            .chunk_index
            .checked_add(1)
            != Some(expected_chunk_count)
        {
            return Err(SimpleKernelError::Bridge(
                "RV64IM published accumulator statement terminal chunk index does not close the published chunk schedule"
                    .into(),
            ));
        }
        if self.terminal_step_statement.step_public.step_hi != self.step_count {
            return Err(SimpleKernelError::Bridge(
                "RV64IM published accumulator statement terminal step_hi does not close the published step_count"
                    .into(),
            ));
        }
        if self.terminal_step_statement.step_public.step_lo != self.terminal_step_statement.chunk_summary.start_index {
            return Err(SimpleKernelError::Bridge(
                "RV64IM published accumulator statement terminal step_lo does not match the terminal chunk summary"
                    .into(),
            ));
        }
        let Some(summary_step_hi) = self
            .terminal_step_statement
            .chunk_summary
            .start_index
            .checked_add(self.terminal_step_statement.chunk_summary.public_step_count)
        else {
            return Err(SimpleKernelError::Bridge(
                "RV64IM published accumulator statement terminal chunk summary overflows the step domain".into(),
            ));
        };
        if summary_step_hi != self.terminal_step_statement.step_public.step_hi {
            return Err(SimpleKernelError::Bridge(
                "RV64IM published accumulator statement terminal chunk summary does not match the terminal step span"
                    .into(),
            ));
        }
        if !self.terminal_step_statement.step_public.halted_out {
            return Err(SimpleKernelError::Bridge(
                "RV64IM published accumulator statement terminal step must close on a halted chunk".into(),
            ));
        }
        if self.terminal_step_statement.step_public.state_out != self.accumulator_final.terminal_handle.0 {
            return Err(SimpleKernelError::Bridge(
                "RV64IM published accumulator statement terminal state_out does not match the authoritative final accumulator terminal handle"
                    .into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn build_rv64im_ivc_public_image_from_published_statement(
    published_statement: &Rv64imAccumulatorPublicStatement,
) -> Result<Rv64imIvcPublicImage, SimpleKernelError> {
    published_statement.validate()?;
    Ok(Rv64imIvcPublicImage {
        vk_fs_digest: published_statement.vk_fs().expected_digest(),
        chunk_count: published_statement.expected_chunk_count()?,
        step_count: published_statement.step_count(),
        z_0: rv64im_chunk_step_ivc_initial_state()
            .carry
            .terminal_handle
            .0,
        z_i: published_statement.canonical_terminal_handle_digest(),
        pc: published_statement.pc_final(),
        x_i: published_statement.x_last().clone(),
        folded_accumulator_digest: published_statement.canonical_folded_accumulator_digest(),
        terminal_statement: Some(published_statement.terminal_step_statement().clone()),
    })
}

pub(crate) fn validate_rv64im_ivc_public_image_against_published_statement(
    published_statement: &Rv64imAccumulatorPublicStatement,
    public_image: &Rv64imIvcPublicImage,
) -> Result<(), SimpleKernelError> {
    let expected_public_image = build_rv64im_ivc_public_image_from_published_statement(published_statement)?;
    if public_image != &expected_public_image {
        return Err(SimpleKernelError::Bridge(
            "RV64IM IVC public image does not match the carried published statement".into(),
        ));
    }
    Ok(())
}

impl Rv64imCompressedMainProof {
    pub fn from_verified_final_seam(
        statement: &Rv64imFinalStatement,
        proof: &Rv64imFinalBuildProof,
        final_pc: u64,
    ) -> Result<Self, SimpleKernelError> {
        let published_statement =
            Rv64imAccumulatorPublicStatement::from_verified_final_seam(statement, proof, final_pc)?;
        let public_image = build_rv64im_ivc_public_image_from_published_statement(&published_statement)?;
        Ok(Self {
            published_statement,
            ivc_snark: prove_rv64im_ivc_snark_from_final_cached(statement, proof, public_image)?,
        })
    }

    pub fn published_statement(&self) -> &Rv64imPublishedStatement {
        &self.published_statement
    }

    pub fn published_statement_mut(&mut self) -> &mut Rv64imPublishedStatement {
        &mut self.published_statement
    }

    pub fn ivc_snark(&self) -> &Rv64imIvcSnark {
        &self.ivc_snark
    }

    pub fn ivc_snark_mut(&mut self) -> &mut Rv64imIvcSnark {
        &mut self.ivc_snark
    }

    pub fn terminal_decider_proof(&self) -> &Rv64imIvcSnarkProof {
        self.ivc_snark.proof()
    }

    pub fn terminal_decider_proof_mut(&mut self) -> &mut Rv64imIvcSnarkProof {
        self.ivc_snark.proof_mut()
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/compressed_main_proof");
        tr.append_message(b"neo.fold.next/nightstream/rv64im/compressed_main_proof/version", b"v2");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/compressed_main_proof/published_statement_digest",
            &self.published_statement.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/compressed_main_proof/public_image_digest",
            &self.ivc_snark.public_image().expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/compressed_main_proof/terminal_decider_proof",
            &self.ivc_snark.proof().snark_data,
        );
        tr.digest32()
    }

    pub fn binding_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/compressed_main_proof_binding");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/compressed_main_proof_binding/version",
            b"v2",
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/compressed_main_proof_binding/published_statement_digest",
            &self.published_statement.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/compressed_main_proof_binding/public_image_digest",
            &self.ivc_snark.public_image().expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/compressed_main_proof_binding/terminal_decider_proof",
            &self.ivc_snark.proof().snark_data,
        );
        tr.digest32()
    }

    pub fn verify(&self, terminal_decider_vk: &Rv64imIvcSnarkVerifierKey) -> Result<(), SimpleKernelError> {
        let expected_public_image = build_rv64im_ivc_public_image_from_published_statement(&self.published_statement)?;
        validate_rv64im_ivc_public_image_against_published_statement(
            &self.published_statement,
            self.ivc_snark.public_image(),
        )?;
        self.ivc_snark
            .verify(terminal_decider_vk, &expected_public_image)
    }
}
