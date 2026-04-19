//! Owns the RV64IM published main-proof carrier.
//!
//! This module owns the published main-proof boundary above the current final
//! seam. It separates the theorem-facing final surface from the carried
//! recursion proof and verifies that proof only through the surface-bound
//! compressed-chain verifier.
//!
//! Local build caches such as the kernel-export proof and chunk summaries may
//! be carried for nearby builder code, but they are not authoritative and are
//! never part of published verification.

use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use crate::finalize::{digest32_as_fields, digest_fields_as_digest32, FixedShapeChunkSummary};
use crate::nightstream::rv64im::Rv64imSideOpeningPublic;
use crate::proof::FoldSchedule;
use crate::rv64im::chunk_step_ivc::build_rv64im_chunk_step_ivc_relations;
use crate::rv64im::final_relation::{
    rv64im_recursive_accumulator_instance_digest_from_parts, verify_rv64im_final_statement_with_output,
    Rv64imFinalBuildProof, Rv64imFinalStatement, Rv64imRecursiveAccumulator,
};
use crate::rv64im::kernel::{Rv64imKernelExportProof, SimpleKernelError};
use crate::rv64im::main_recursion::{
    build_rv64im_main_recursion_f_prime_advices, build_rv64im_main_recursion_f_prime_advices_with_side_opening_public,
    build_rv64im_main_recursion_verifier_key_fs, Rv64imEncodedPublicInput, Rv64imMainRecursionFPrimeAdvice,
    Rv64imVerifierKeyFs,
};
use crate::rv64im::recursion_spartan::{
    build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs, prove_rv64im_recursion_proof_from_advices,
    setup_rv64im_recursion, validate_rv64im_main_recursion_public_surface_against_published_statement,
    validate_rv64im_recursion_verifier_key_against_published_statement, verify_rv64im_recursion, Rv64imRecursionProof,
    Rv64imRecursionVerifierKey,
};

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imMainProof {
    linkage_anchor_digest: [u8; 32],
    published_statement: Rv64imPublishedStatement,
    #[serde(skip, default)]
    final_statement: Option<Rv64imFinalStatement>,
    #[serde(skip, default)]
    final_surface: Option<Rv64imMainFinalProofSurface>,
    #[serde(skip, default)]
    chunk_summaries: Vec<FixedShapeChunkSummary>,
    #[serde(skip, default)]
    kernel_export: Option<Rv64imKernelExportProof>,
    recursion_proof: Rv64imRecursionProof,
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
}

pub type Rv64imPublishedStatement = Rv64imAccumulatorPublicStatement;
pub type Rv64imPublishedProof = Rv64imRecursionProof;

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

    pub fn from_final_surface(
        final_statement: &Rv64imFinalStatement,
        final_surface: &Rv64imMainFinalProofSurface,
    ) -> Result<Self, SimpleKernelError> {
        let vk_fs = build_rv64im_main_recursion_verifier_key_fs()?;
        let accumulator_final = final_statement.folded.final_accumulator.clone();
        let fold_schedule = final_surface.fold_schedule();
        let step_count = final_surface.semantic_step_count();
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
        })
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

    pub fn validate(&self) -> Result<(), SimpleKernelError> {
        let expected_vk_fs = build_rv64im_main_recursion_verifier_key_fs()?;
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
        let _ = self.expected_chunk_count()?;
        Ok(())
    }
}

impl Rv64imMainProof {
    pub fn from_final(
        statement: &Rv64imFinalStatement,
        proof: &Rv64imFinalBuildProof,
    ) -> Result<Self, SimpleKernelError> {
        let verified_kernel = verify_rv64im_final_statement_with_output(statement, proof)?;
        let relations = build_rv64im_chunk_step_ivc_relations(statement, proof)?;
        let advices = build_rv64im_main_recursion_f_prime_advices(&relations)?;
        Self::from_final_with_relations_and_advices(statement, proof, verified_kernel.final_pc, &relations, &advices)
    }

    pub fn from_final_with_side_opening_public(
        statement: &Rv64imFinalStatement,
        proof: &Rv64imFinalBuildProof,
        side_opening_public: &Rv64imSideOpeningPublic,
    ) -> Result<Self, SimpleKernelError> {
        let verified_kernel = verify_rv64im_final_statement_with_output(statement, proof)?;
        let relations = build_rv64im_chunk_step_ivc_relations(statement, proof)?;
        let advices =
            build_rv64im_main_recursion_f_prime_advices_with_side_opening_public(&relations, side_opening_public)?;
        Self::from_final_with_relations_and_advices(statement, proof, verified_kernel.final_pc, &relations, &advices)
    }

    pub fn final_statement_cache(&self) -> Option<&Rv64imFinalStatement> {
        self.final_statement.as_ref()
    }

    pub fn final_statement_cache_mut(&mut self) -> Option<&mut Rv64imFinalStatement> {
        self.final_statement.as_mut()
    }

    pub fn final_surface_cache(&self) -> Option<&Rv64imMainFinalProofSurface> {
        self.final_surface.as_ref()
    }

    pub fn final_surface_cache_mut(&mut self) -> Option<&mut Rv64imMainFinalProofSurface> {
        self.final_surface.as_mut()
    }

    pub fn kernel_export_cache(&self) -> Option<&Rv64imKernelExportProof> {
        self.kernel_export.as_ref()
    }

    pub fn kernel_export_cache_mut(&mut self) -> Option<&mut Rv64imKernelExportProof> {
        self.kernel_export.as_mut()
    }

    pub fn published_statement(&self) -> &Rv64imPublishedStatement {
        &self.published_statement
    }

    pub fn published_statement_mut(&mut self) -> &mut Rv64imPublishedStatement {
        &mut self.published_statement
    }

    pub fn published_proof(&self) -> &Rv64imPublishedProof {
        &self.recursion_proof
    }

    pub fn linkage_anchor_digest(&self) -> [u8; 32] {
        self.linkage_anchor_digest
    }

    pub fn linkage_anchor_digest_mut(&mut self) -> &mut [u8; 32] {
        &mut self.linkage_anchor_digest
    }

    pub fn chunk_summaries(&self) -> &[FixedShapeChunkSummary] {
        &self.chunk_summaries
    }

    pub fn chunk_summary_count(&self) -> u64 {
        self.final_surface
            .as_ref()
            .expect("main-proof chunk summary count requires a local final-surface cache")
            .chunk_summary_count()
    }

    pub fn chunk_summary_chain_digest(&self) -> [u8; 32] {
        self.final_surface
            .as_ref()
            .expect("main-proof chunk summary chain digest requires a local final-surface cache")
            .chunk_summary_chain_digest()
    }

    pub fn validate_final_surface(&self) -> Result<(), SimpleKernelError> {
        match (&self.final_statement, &self.final_surface) {
            (Some(final_statement), Some(final_surface)) => {
                final_surface.validate_against_final_statement(final_statement)
            }
            (None, None) => Ok(()),
            _ => Err(SimpleKernelError::Bridge(
                "RV64IM main proof local final caches are partially present".into(),
            )),
        }
    }

    pub fn validate_local_build_caches(&self) -> Result<(), SimpleKernelError> {
        self.validate_final_surface()?;
        if let (Some(final_statement), Some(final_surface)) = (&self.final_statement, &self.final_surface) {
            let expected_published_statement =
                Rv64imAccumulatorPublicStatement::from_final_surface(final_statement, final_surface)?;
            if self.published_statement != expected_published_statement {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM Nightstream main proof local final caches do not reconstruct the carried published statement"
                        .into(),
                ));
            }
            if self.linkage_anchor_digest != final_statement.public_statement_digest {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM Nightstream main proof linkage anchor digest does not match the carried final-statement public digest"
                        .into(),
                ));
            }
        }
        if let Some(kernel_export) = &self.kernel_export {
            if self.linkage_anchor_digest != kernel_export.public_statement_digest() {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM Nightstream main proof local kernel-export cache does not match the carried final-statement public digest"
                        .into(),
                ));
            }
        }
        if !self.chunk_summaries.is_empty() {
            let final_surface = self.final_surface.as_ref().ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV64IM Nightstream main proof chunk-summary cache requires a local final-surface cache".into(),
                )
            })?;
            if final_surface.chunk_summary_count() != self.chunk_summaries.len() as u64 {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM Nightstream main proof chunk-summary count does not match the local chunk-summary cache"
                        .into(),
                ));
            }
            let expected_chain_digest = rv64im_chunk_summary_chain_digest_from_summaries(&self.chunk_summaries);
            if final_surface.chunk_summary_chain_digest() != expected_chain_digest {
                return Err(SimpleKernelError::Bridge(
                    "RV64IM Nightstream main proof chunk-summary chain digest does not match the local chunk-summary cache"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    pub fn recursion_proof(&self) -> &Rv64imRecursionProof {
        &self.recursion_proof
    }

    pub fn recursion_proof_mut(&mut self) -> &mut Rv64imRecursionProof {
        &mut self.recursion_proof
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/main_proof");
        tr.append_message(b"neo.fold.next/nightstream/rv64im/main_proof/version", b"v9");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/main_proof/published_statement_digest",
            &self.published_statement.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/main_proof/recursion_proof_digest",
            &self.recursion_proof.expected_digest(),
        );
        tr.digest32()
    }

    pub fn binding_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/main_proof_binding");
        tr.append_message(b"neo.fold.next/nightstream/rv64im/main_proof_binding/version", b"v6");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/main_proof_binding/published_statement_digest",
            &self.published_statement.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/main_proof_binding/recursion_final_public_image_digest",
            &self.recursion_proof.final_public_image_digest(),
        );
        tr.digest32()
    }

    fn from_final_with_relations_and_advices(
        statement: &Rv64imFinalStatement,
        proof: &Rv64imFinalBuildProof,
        final_pc: u64,
        relations: &[crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcRelation],
        advices: &[Rv64imMainRecursionFPrimeAdvice],
    ) -> Result<Self, SimpleKernelError> {
        let recursion_proof = prove_rv64im_recursion_proof_from_advices(relations, advices)?;
        let final_surface = Rv64imMainFinalProofSurface::from_final_proof(statement, proof, final_pc);
        let published_statement = Rv64imAccumulatorPublicStatement::from_final_surface(statement, &final_surface)?;
        Ok(Self {
            linkage_anchor_digest: statement.public_statement_digest,
            published_statement,
            final_statement: Some(statement.clone()),
            final_surface: Some(final_surface),
            chunk_summaries: proof.chunk_summaries.clone(),
            kernel_export: Some(proof.kernel_export.clone()),
            recursion_proof,
        })
    }
}

impl PartialEq for Rv64imMainProof {
    fn eq(&self, other: &Self) -> bool {
        self.linkage_anchor_digest == other.linkage_anchor_digest
            && self.published_statement == other.published_statement
            && self.recursion_proof == other.recursion_proof
    }
}

pub fn build_rv64im_main_proof(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Rv64imMainProof, SimpleKernelError> {
    Rv64imMainProof::from_final(statement, proof)
}

pub fn build_rv64im_main_proof_with_side_opening_public(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
    side_opening_public: &Rv64imSideOpeningPublic,
) -> Result<Rv64imMainProof, SimpleKernelError> {
    Rv64imMainProof::from_final_with_side_opening_public(statement, proof, side_opening_public)
}

pub fn verify_rv64im_published_main_proof(
    published_statement: &Rv64imPublishedStatement,
    published_proof: &Rv64imPublishedProof,
) -> Result<(), SimpleKernelError> {
    let (_, recursion_vk) = setup_rv64im_recursion()?;
    verify_rv64im_published_main_proof_with_vk(&recursion_vk, published_statement, published_proof)
}

pub fn verify_rv64im_published_main_proof_with_vk(
    recursion_vk: &Rv64imRecursionVerifierKey,
    published_statement: &Rv64imPublishedStatement,
    published_proof: &Rv64imPublishedProof,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_recursion(recursion_vk, published_statement, published_proof)
}

pub fn verify_rv64im_main_proof(main_proof: &Rv64imMainProof) -> Result<(), SimpleKernelError> {
    main_proof.validate_final_surface()?;
    let (_, recursion_vk) = setup_rv64im_recursion()?;
    validate_rv64im_recursion_verifier_key_against_published_statement(
        &recursion_vk,
        main_proof.published_statement(),
    )?;
    validate_rv64im_main_recursion_public_surface_against_published_statement(
        main_proof.published_statement(),
        main_proof.published_proof(),
    )?;
    verify_rv64im_recursion(
        &recursion_vk,
        main_proof.published_statement(),
        main_proof.published_proof(),
    )
}
