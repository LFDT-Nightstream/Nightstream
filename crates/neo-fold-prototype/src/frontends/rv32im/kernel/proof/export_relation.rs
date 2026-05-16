//! Owns the RV32IM kernel export relation above the accepted-proof artifact.
//!
//! It verifies the rich accepted artifact once, then collapses it into:
//! - canonical per-chunk public surfaces,
//! - canonical per-chunk prepared-step bridge bindings,
//! - one fixed export relation digest for recursion/decider work.

use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::finalize::{digest32_as_fields, public_chunk_digest};
use crate::proof::{partition_step_inputs, ChunkInput, FoldSchedule, PublicChunk};
use crate::rv32im::stage1::{build_stage1_summary, Stage1Summary};
use crate::rv32im::stage2::{build_stage2_summary, Stage2Summary};
use crate::rv32im::stage3::{build_stage3_summary, Stage3Summary};

use super::artifacts::digest_rows;
use super::proof_accepted::{accepted_proof_artifact_from_public_proof, Rv32imAcceptedProofArtifact};
use super::proof_api::{Rv32imMainLaneProofBinding, Rv32imMainLaneProofBundle, Rv32imProof, Rv32imProofStatement};
use super::proof_staged_verify::{
    derive_stage1_export_proof, derive_stage2_export_proof, derive_stage3_export_proof,
    verify_accepted_proof_artifact_export_core_with_perf, verify_accepted_proof_core_with_transcript_surface_with_perf,
    Rv32imAcceptedProofCoreInputs,
};
use super::proof_witness::{
    kernel_export_claim_proof_from_bundle, kernel_opening_proof_bundle_from_opening,
    stage_claim_proof_bundle_from_claims, stage_package_proof_bundle_from_packages, Rv32imKernelExportClaimProof,
};
use super::root_lane_commitment::build_root_lane_commitment_summary_artifact_from_public_witness;
use super::root_lane_witness::RootExecutionBundle;
use super::simple::{
    build_prepared_steps_from_execution_rows, build_public_root_lane_witness_and_binding_summary,
    rv32im_root_step_cap_for_schedule, rv32im_simple_root_context_id_for_schedule,
    rv32im_simple_root_params_for_step_cap, SimpleKernelError,
};
use super::stage_artifacts::{
    build_public_kernel_opening_bundle_from_export_parts_with_perf, build_stage_claim_bundle_from_export_parts,
};
use super::stage_package_perf::build_public_stage_package_bundle_with_perf;
use super::transcript::verify_transcript_record;
use super::{
    build_main_lane_surface, build_root_lane_columns, prepared_step_digest, prepared_step_digests, public_step_digests,
    VerifiedTranscriptSurface,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imPreparedStepBridgeBinding {
    pub logical_index: u64,
    pub trace_index: u64,
    pub row_binding_digest: [u8; 32],
    pub prepared_step_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imChunkBridgeHandoff {
    pub chunk_index: u64,
    pub chunk_start_index: u64,
    pub public_step_count: u64,
    pub step_bindings: Vec<Rv32imPreparedStepBridgeBinding>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32imVerifiedKernelChunkHandoff {
    pub chunk_input: ChunkInput,
    pub public_chunk: PublicChunk,
    pub public_chunk_digest: [u8; 32],
    pub public_chunk_instance_digest: [F; 4],
    pub prepared_step_digests: Vec<[u8; 32]>,
    pub bridge_handoff: Rv32imChunkBridgeHandoff,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32imKernelChunkExportWitness {
    pub chunk_input: ChunkInput,
    pub prepared_step_digests: Vec<[u8; 32]>,
    pub bridge_handoff: Rv32imChunkBridgeHandoff,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32imKernelExportWitness {
    pub chunk_handoffs: Vec<Rv32imKernelChunkExportWitness>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32imKernelExportMainLaneProof {
    pub packaged: crate::proof::PackagedProof,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32imKernelExportSource {
    pub kernel_claims: Rv32imKernelExportClaimProof,
    pub main_lane: Rv32imKernelExportMainLaneProof,
    pub transcript: VerifiedTranscriptSurface,
    pub root_execution: RootExecutionBundle,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32imKernelExportProof {
    pub source: Rv32imKernelExportSource,
    pub witness: Rv32imKernelExportWitness,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub(crate) struct Rv32imKernelExportBuildOutput {
    pub relation: Rv32imKernelExportRelation,
    pub proof: Rv32imKernelExportProof,
    pub result: Rv32imKernelExportRelationResult,
}

#[derive(Clone, Debug)]
pub struct Rv32imKernelExportRelationResult {
    pub fold_schedule: FoldSchedule,
    pub chunk_handoffs: Vec<Rv32imVerifiedKernelChunkHandoff>,
    pub final_state_digest: [u8; 32],
    pub final_pc: u64,
    pub halted: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv32imVerifiedKernelExportCore {
    fold_schedule: FoldSchedule,
    public_chunks: Vec<PublicChunk>,
    root_execution: RootExecutionBundle,
    final_state_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imChunkExportSurface {
    pub public_chunk_digest: [u8; 32],
    pub bridge_handoff_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imKernelExportRelation {
    pub fold_schedule: FoldSchedule,
    pub chunk_count: u64,
    pub public_step_count: u64,
    pub final_state_digest: [u8; 32],
    pub final_pc: u64,
    pub halted: bool,
    pub chunk_surfaces: Vec<Rv32imChunkExportSurface>,
    pub digest: [u8; 32],
}

impl Rv32imChunkBridgeHandoff {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/chunk_bridge_handoff");
        tr.append_u64s(
            b"rv32im/chunk_bridge_handoff/meta",
            &[self.chunk_index, self.chunk_start_index, self.public_step_count],
        );
        append_step_bindings(&mut tr, &self.step_bindings);
        tr.digest32()
    }
}

impl Rv32imPreparedStepBridgeBinding {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/prepared_step_bridge_binding");
        tr.append_u64s(
            b"rv32im/prepared_step_bridge_binding/meta",
            &[self.logical_index, self.trace_index],
        );
        tr.append_message(
            b"rv32im/prepared_step_bridge_binding/row_binding_digest",
            &self.row_binding_digest,
        );
        tr.append_message(
            b"rv32im/prepared_step_bridge_binding/prepared_step_digest",
            &self.prepared_step_digest,
        );
        tr.digest32()
    }
}

impl Rv32imKernelChunkExportWitness {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/kernel_chunk_export_witness");
        tr.append_u64s(
            b"rv32im/kernel_chunk_export_witness/meta",
            &[self.chunk_input.start_index as u64, self.chunk_input.steps.len() as u64],
        );
        for digest in &self.prepared_step_digests {
            tr.append_message(b"rv32im/kernel_chunk_export_witness/prepared_step_digest", digest);
        }
        tr.append_message(
            b"rv32im/kernel_chunk_export_witness/bridge_handoff_digest",
            &self.bridge_handoff.digest,
        );
        tr.digest32()
    }
}

impl Rv32imKernelExportWitness {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/kernel_export_witness");
        tr.append_u64s(
            b"rv32im/kernel_export_witness/chunk_count",
            &[self.chunk_handoffs.len() as u64],
        );
        for handoff in &self.chunk_handoffs {
            tr.append_message(b"rv32im/kernel_export_witness/chunk_handoff_digest", &handoff.digest);
        }
        tr.digest32()
    }
}

impl Rv32imKernelExportMainLaneProof {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/kernel_export_main_lane_proof");
        tr.append_message(
            b"rv32im/kernel_export_main_lane_proof/statement_digest",
            &self.packaged.statement.digest,
        );
        tr.append_message(
            b"rv32im/kernel_export_main_lane_proof/proof_digest",
            &self.packaged.proof.proof_digest,
        );
        tr.digest32()
    }

    fn to_public_bundle(
        &self,
        root_lane_columns_digest: [u8; 32],
        root_lane_commitment_digest: [u8; 32],
    ) -> Result<Rv32imMainLaneProofBundle, SimpleKernelError> {
        let public_step_count = self.public_step_count();
        let chunk_count = self.chunk_count();
        let binding = Rv32imMainLaneProofBinding {
            root_lane_columns_digest,
            root_lane_commitment_digest,
            fold_schedule: self.fold_schedule(),
            chunk_count,
            public_step_count,
            digest: [0; 32],
        };
        let binding = Rv32imMainLaneProofBinding {
            digest: binding.expected_digest(),
            ..binding
        };
        if binding
            .fold_schedule
            .chunk_count(public_step_count as usize)
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))? as u64
            != chunk_count
        {
            return Err(SimpleKernelError::Bridge(
                "RV32IM kernel export main-lane packaged statement chunk layout is inconsistent".into(),
            ));
        }
        let bundle = Rv32imMainLaneProofBundle {
            binding,
            packaged: self.packaged.clone(),
            digest: [0; 32],
        };
        Ok(Rv32imMainLaneProofBundle {
            digest: bundle.expected_digest(),
            ..bundle
        })
    }

    pub fn public_step_count(&self) -> u64 {
        self.packaged
            .statement
            .chunks
            .iter()
            .map(|chunk| chunk.steps.len() as u64)
            .sum()
    }

    pub fn fold_schedule(&self) -> FoldSchedule {
        self.packaged.statement.fold_schedule
    }

    pub fn chunk_count(&self) -> u64 {
        self.packaged.statement.chunks.len() as u64
    }
}

impl Rv32imKernelExportSource {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/kernel_export_source");
        tr.append_message(b"rv32im/kernel_export_source/kernel_claims", &self.kernel_claims.digest);
        tr.append_message(b"rv32im/kernel_export_source/main_lane", &self.main_lane.digest);
        tr.append_message(
            b"rv32im/kernel_export_source/transcript_digest",
            &self.transcript.expected_digest(),
        );
        tr.append_message(
            b"rv32im/kernel_export_source/root_execution",
            &self.root_execution.digest,
        );
        tr.digest32()
    }

    pub fn public_statement_digest(&self) -> [u8; 32] {
        kernel_export_statement_from_source(self).digest
    }
}

impl Rv32imKernelExportProof {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/kernel_export_proof");
        tr.append_message(b"rv32im/kernel_export_proof/source_digest", &self.source.digest);
        tr.append_message(b"rv32im/kernel_export_proof/witness_digest", &self.witness.digest);
        tr.digest32()
    }

    pub fn public_statement_digest(&self) -> [u8; 32] {
        self.source.public_statement_digest()
    }
}

impl Rv32imChunkExportSurface {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/chunk_export_surface");
        tr.append_message(
            b"rv32im/chunk_export_surface/public_chunk_digest",
            &self.public_chunk_digest,
        );
        tr.append_message(
            b"rv32im/chunk_export_surface/bridge_handoff_digest",
            &self.bridge_handoff_digest,
        );
        tr.digest32()
    }
}

impl Rv32imKernelExportRelation {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/kernel_export_relation");
        tr.append_u64s(
            b"rv32im/kernel_export_relation/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_u64s(
            b"rv32im/kernel_export_relation/meta",
            &[
                self.chunk_count,
                self.public_step_count,
                self.final_pc,
                self.halted as u64,
            ],
        );
        tr.append_message(
            b"rv32im/kernel_export_relation/final_state_digest",
            &self.final_state_digest,
        );
        append_chunk_surfaces(&mut tr, &self.chunk_surfaces);
        tr.digest32()
    }
}

pub fn build_rv32im_kernel_export_witness(proof: &Rv32imProof) -> Result<Rv32imKernelExportWitness, SimpleKernelError> {
    build_rv32im_kernel_export_seam(proof).map(|(_, witness)| witness)
}

pub fn verify_rv32im_kernel_export_witness(
    relation: &Rv32imKernelExportRelation,
    witness: &Rv32imKernelExportWitness,
) -> Result<(), SimpleKernelError> {
    verify_rv32im_kernel_export_witness_with_output(relation, witness).map(|_| ())
}

pub fn build_rv32im_kernel_export_relation(
    proof: &Rv32imProof,
) -> Result<Rv32imKernelExportRelation, SimpleKernelError> {
    build_rv32im_kernel_export_seam(proof).map(|(relation, _)| relation)
}

pub fn verify_rv32im_kernel_export_relation(
    relation: &Rv32imKernelExportRelation,
    proof: &Rv32imProof,
) -> Result<(), SimpleKernelError> {
    verify_rv32im_kernel_export_relation_with_output(relation, proof).map(|_| ())
}

pub(crate) fn verify_rv32im_kernel_export_relation_with_output(
    relation: &Rv32imKernelExportRelation,
    proof: &Rv32imProof,
) -> Result<Rv32imKernelExportRelationResult, SimpleKernelError> {
    let artifact = accepted_proof_artifact_from_public_proof(proof)?;
    let (expected, result) = build_rv32im_kernel_export_relation_from_artifact(&artifact)?;
    if relation != &expected {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export relation does not match the verified accepted artifact".into(),
        ));
    }
    Ok(result)
}

pub(crate) fn build_rv32im_kernel_export_seam(
    proof: &Rv32imProof,
) -> Result<(Rv32imKernelExportRelation, Rv32imKernelExportWitness), SimpleKernelError> {
    let artifact = accepted_proof_artifact_from_public_proof(proof)?;
    build_rv32im_kernel_export_seam_from_accepted_artifact(&artifact).map(|(relation, witness, _)| (relation, witness))
}

pub(crate) fn build_rv32im_kernel_export_seam_from_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<
    (
        Rv32imKernelExportRelation,
        Rv32imKernelExportWitness,
        Rv32imKernelExportRelationResult,
    ),
    SimpleKernelError,
> {
    let (relation, result) = build_rv32im_kernel_export_relation_from_artifact(artifact)?;
    let witness = kernel_export_witness_from_result(&result);
    Ok((relation, witness, result))
}

pub(crate) fn build_rv32im_kernel_export_proof_from_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<
    (
        Rv32imKernelExportRelation,
        Rv32imKernelExportProof,
        Rv32imKernelExportRelationResult,
    ),
    SimpleKernelError,
> {
    let (relation, witness, result) = build_rv32im_kernel_export_seam_from_accepted_artifact(artifact)?;
    let source = build_rv32im_kernel_export_source_from_accepted_artifact(artifact)?;
    let mut proof = Rv32imKernelExportProof {
        source,
        witness,
        digest: [0; 32],
    };
    proof.digest = proof.expected_digest();
    Ok((relation, proof, result))
}

pub(crate) fn build_rv32im_kernel_export_proof_from_carried_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<
    (
        Rv32imKernelExportRelation,
        Rv32imKernelExportProof,
        Rv32imKernelExportRelationResult,
    ),
    SimpleKernelError,
> {
    let source = build_rv32im_kernel_export_source_from_accepted_artifact(artifact)?;
    let built = build_rv32im_kernel_export_build_output_from_carried_accepted_artifact_with_source(artifact, source)?;
    Ok((built.relation, built.proof, built.result))
}

pub(crate) fn build_rv32im_kernel_export_build_output_from_carried_accepted_artifact_with_source(
    artifact: &Rv32imAcceptedProofArtifact,
    source: Rv32imKernelExportSource,
) -> Result<Rv32imKernelExportBuildOutput, SimpleKernelError> {
    build_rv32im_kernel_export_build_output_from_carried_accepted_artifact_with_source_and_chunk_inputs(
        artifact, source, None,
    )
}

pub(crate) fn build_rv32im_kernel_export_build_output_from_carried_accepted_artifact_with_source_and_chunk_inputs(
    artifact: &Rv32imAcceptedProofArtifact,
    source: Rv32imKernelExportSource,
    chunk_inputs: Option<Vec<ChunkInput>>,
) -> Result<Rv32imKernelExportBuildOutput, SimpleKernelError> {
    if artifact.digest != artifact.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM accepted proof artifact digest mismatch".into(),
        ));
    }
    if artifact.claim.digest != artifact.claim.expected_digest()
        || artifact.statement.digest != artifact.statement.expected_digest()
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM accepted proof public claim digest mismatch".into(),
        ));
    }
    if source.digest != source.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM carried kernel export source digest mismatch".into(),
        ));
    }
    let (final_pc, halted) = kernel_export_terminal_row(&source)?;
    if source.root_execution.digest != artifact.root_execution.digest
        || source.main_lane.packaged.statement.digest != artifact.main_lane.packaged.statement.digest
        || source.kernel_claims.final_state_digest() != artifact.statement.final_state_digest
        || final_pc != artifact.statement.final_pc
        || halted != artifact.statement.halted
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM carried kernel export source does not match the accepted artifact".into(),
        ));
    }
    let (relation, result) = match chunk_inputs {
        Some(chunk_inputs) => build_rv32im_kernel_export_relation_from_verified_parts_and_chunk_inputs(
            source.main_lane.packaged.statement.fold_schedule,
            &source.main_lane.packaged.statement.chunks,
            chunk_inputs,
            &source.root_execution,
            artifact.statement.final_state_digest,
            artifact.statement.final_pc,
            artifact.statement.halted,
        )?,
        None => build_rv32im_kernel_export_relation_from_verified_parts(
            source.main_lane.packaged.statement.fold_schedule,
            &source.main_lane.packaged.statement.chunks,
            &source.root_execution,
            artifact.statement.final_state_digest,
            artifact.statement.final_pc,
            artifact.statement.halted,
        )?,
    };
    let witness = kernel_export_witness_from_result(&result);
    let (relation, proof, result) =
        build_rv32im_kernel_export_proof_from_carried_parts(source, relation, witness, result)?;
    Ok(Rv32imKernelExportBuildOutput {
        relation,
        proof,
        result,
    })
}

fn build_rv32im_kernel_export_proof_from_carried_parts(
    source: Rv32imKernelExportSource,
    relation: Rv32imKernelExportRelation,
    witness: Rv32imKernelExportWitness,
    result: Rv32imKernelExportRelationResult,
) -> Result<
    (
        Rv32imKernelExportRelation,
        Rv32imKernelExportProof,
        Rv32imKernelExportRelationResult,
    ),
    SimpleKernelError,
> {
    let mut proof = Rv32imKernelExportProof {
        source,
        witness,
        digest: [0; 32],
    };
    proof.digest = proof.expected_digest();
    Ok((relation, proof, result))
}

pub fn build_rv32im_kernel_export_source_from_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<Rv32imKernelExportSource, SimpleKernelError> {
    let source = Rv32imKernelExportSource {
        kernel_claims: kernel_export_claim_proof_from_bundle(&artifact.kernel_claims)?,
        main_lane: kernel_export_main_lane_proof_from_bundle(&artifact.main_lane),
        transcript: verify_transcript_record(&artifact.transcript)?,
        root_execution: artifact.root_execution.clone(),
        digest: [0; 32],
    };
    Ok(Rv32imKernelExportSource {
        digest: source.expected_digest(),
        ..source
    })
}

fn kernel_export_terminal_row(source: &Rv32imKernelExportSource) -> Result<(u64, bool), SimpleKernelError> {
    let Some(last_row) = source.root_execution.execution_rows.last() else {
        return Err(SimpleKernelError::Bridge(
            "RV32IM export root execution must carry at least one row".into(),
        ));
    };
    Ok((u64::from(last_row.next_pc), last_row.halted))
}

fn kernel_export_statement_from_source(source: &Rv32imKernelExportSource) -> Rv32imProofStatement {
    let (final_pc, halted) = kernel_export_terminal_row(source)
        .expect("RV32IM export statement rebuild should derive terminal control flow from root execution");
    let execution_digest = digest_rows(&source.root_execution.execution_rows);
    let root_lane_columns = build_root_lane_columns(&source.root_execution.execution_rows);
    let main_lane_surface = build_main_lane_surface(&root_lane_columns);
    let stage1 = stage1_summary_from_export_source(source);
    let stage2 = stage2_summary_from_export_source(source);
    let stage3 = stage3_summary_from_export_source(source);
    let stage_claims =
        build_stage_claim_bundle_from_export_parts(&stage1, &stage2, &stage3, &source.transcript, execution_digest)
            .expect("RV32IM export statement rebuild should derive stage claims from carried stage summaries");
    let stage_claims = stage_claim_proof_bundle_from_claims(&stage_claims).expect(
        "RV32IM export statement rebuild should derive the stage-claim proof bundle from carried stage summaries",
    );
    let (stage_packages, _) =
        build_public_stage_package_bundle_with_perf(&stage1, &stage2, &stage3, &stage_claims.claims)
            .expect("RV32IM export statement rebuild should derive stage packages from carried stage summaries");
    let stage_packages = stage_package_proof_bundle_from_packages(&stage_packages);
    let root_lane_commitment = kernel_export_root_lane_commitment_from_source(source)
        .expect("RV32IM export statement rebuild should derive the root-lane commitment from root execution");
    let (kernel_opening, _) = build_public_kernel_opening_bundle_from_export_parts_with_perf(
        &stage_claims.claims,
        &stage_packages.packages,
        source.root_execution.prepared_step_bindings.digest,
        source.root_execution.prepared_step_bindings.binding_count,
        source
            .root_execution
            .prepared_step_bindings
            .first_binding_digest,
        source
            .root_execution
            .prepared_step_bindings
            .last_binding_digest,
        execution_digest,
        source.kernel_claims.final_state_digest(),
        source.transcript.final_digest,
        final_pc,
        halted,
        &root_lane_commitment,
    )
    .expect("RV32IM export statement rebuild should derive the kernel opening from carried export inputs");
    let kernel_opening = kernel_opening_proof_bundle_from_opening(&kernel_opening);
    let initial_pc = source
        .root_execution
        .execution_rows
        .first()
        .map(|row| row.pc)
        .expect("RV32IM export statement rebuild should derive initial pc from root execution");
    let statement = Rv32imProofStatement {
        root_params_id: rv32im_simple_root_context_id_for_schedule(
            source.main_lane.fold_schedule(),
            root_lane_columns.time_len as usize,
        )
        .expect("RV32IM export statement rebuild should derive the root params id from the carried main-lane schedule"),
        fold_schedule: source.main_lane.fold_schedule(),
        chunk_count: source.main_lane.chunk_count(),
        stage_claims_digest: stage_claims.digest,
        stage_packages_digest: stage_packages.digest,
        kernel_opening_digest: kernel_opening.digest,
        prepared_step_bindings_digest: source.root_execution.prepared_step_bindings.digest,
        execution_digest,
        final_state_digest: source.kernel_claims.final_state_digest(),
        transcript_final_digest: source.transcript.final_digest,
        main_lane_surface_digest: main_lane_surface.digest,
        root_lane_columns_digest: root_lane_columns.digest,
        public_step_count: root_lane_columns.time_len,
        initial_pc: u64::from(initial_pc),
        final_pc,
        halted,
        digest: [0; 32],
    };
    Rv32imProofStatement {
        digest: statement.expected_digest(),
        ..statement
    }
}

fn stage1_summary_from_export_source(source: &Rv32imKernelExportSource) -> Stage1Summary {
    build_stage1_summary(&source.root_execution.execution_rows)
}

fn stage2_summary_from_export_source(source: &Rv32imKernelExportSource) -> Stage2Summary {
    build_stage2_summary(&source.root_execution.execution_rows)
}

fn stage3_summary_from_export_source(source: &Rv32imKernelExportSource) -> Stage3Summary {
    build_stage3_summary(&source.root_execution.execution_rows)
}

fn kernel_export_root_lane_commitment_from_source(
    source: &Rv32imKernelExportSource,
) -> Result<super::RootLaneCommitmentSummaryArtifact, SimpleKernelError> {
    let params = rv32im_simple_root_params_for_step_cap(rv32im_root_step_cap_for_schedule(
        source.main_lane.fold_schedule(),
        source.main_lane.public_step_count() as usize,
    )?);
    let (root_lane_witness, _) =
        build_public_root_lane_witness_and_binding_summary(&source.root_execution.execution_rows);
    build_root_lane_commitment_summary_artifact_from_public_witness(&params, &root_lane_witness)
}

pub fn verify_rv32im_kernel_export_source(source: &Rv32imKernelExportSource) -> Result<(), SimpleKernelError> {
    verify_rv32im_kernel_export_source_with_output(source).map(|_| ())
}

pub(crate) fn verify_rv32im_kernel_export_source_with_output(
    source: &Rv32imKernelExportSource,
) -> Result<Rv32imVerifiedKernelExportCore, SimpleKernelError> {
    if source.digest != source.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export source digest mismatch".into(),
        ));
    }
    if source.main_lane.digest != source.main_lane.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export main-lane surface digest mismatch".into(),
        ));
    }
    let execution_digest = digest_rows(&source.root_execution.execution_rows);
    let (final_pc, halted) = kernel_export_terminal_row(source)?;
    let root_lane_columns = build_root_lane_columns(&source.root_execution.execution_rows);
    let root_lane_commitment = kernel_export_root_lane_commitment_from_source(source)?;
    let main_lane = source
        .main_lane
        .to_public_bundle(root_lane_columns.digest, root_lane_commitment.digest)?;
    let statement = kernel_export_statement_from_source(source);
    let stage1 = stage1_summary_from_export_source(source);
    let stage2 = stage2_summary_from_export_source(source);
    let stage3 = stage3_summary_from_export_source(source);
    let stage_claims =
        build_stage_claim_bundle_from_export_parts(&stage1, &stage2, &stage3, &source.transcript, execution_digest)?;
    let stage_claims = stage_claim_proof_bundle_from_claims(&stage_claims)?;
    let (stage_packages, _) =
        build_public_stage_package_bundle_with_perf(&stage1, &stage2, &stage3, &stage_claims.claims)?;
    let stage_packages = stage_package_proof_bundle_from_packages(&stage_packages);
    let (kernel_opening, _) = build_public_kernel_opening_bundle_from_export_parts_with_perf(
        &stage_claims.claims,
        &stage_packages.packages,
        source.root_execution.prepared_step_bindings.digest,
        source.root_execution.prepared_step_bindings.binding_count,
        source
            .root_execution
            .prepared_step_bindings
            .first_binding_digest,
        source
            .root_execution
            .prepared_step_bindings
            .last_binding_digest,
        execution_digest,
        source.kernel_claims.final_state_digest(),
        source.transcript.final_digest,
        final_pc,
        halted,
        &root_lane_commitment,
    )?;
    let kernel_opening = kernel_opening_proof_bundle_from_opening(&kernel_opening);
    let stage1 = derive_stage1_export_proof(&source.root_execution.execution_rows, &stage_packages);
    let stage2 = derive_stage2_export_proof(&source.root_execution.execution_rows, &stage_packages);
    let stage3 = derive_stage3_export_proof(
        &source.root_execution.execution_rows,
        &source.root_execution,
        &stage_packages,
        statement.initial_pc,
        statement.final_pc,
        stage2.temporal_digest,
    );
    let inputs = Rv32imAcceptedProofCoreInputs {
        statement: &statement,
        stage_claims: &stage_claims,
        stage_packages: &stage_packages,
        kernel_opening: &kernel_opening,
        kernel_claims: source.kernel_claims.clone(),
        root_lane_columns: &root_lane_columns,
        root_lane_commitment: &root_lane_commitment,
        main_lane: &main_lane,
        stage1,
        stage2,
        stage3,
        root_execution: &source.root_execution,
    };
    verify_accepted_proof_core_with_transcript_surface_with_perf(&inputs, &source.transcript, None)?;
    Ok(Rv32imVerifiedKernelExportCore {
        fold_schedule: source.main_lane.packaged.statement.fold_schedule,
        public_chunks: source.main_lane.packaged.statement.chunks.clone(),
        root_execution: source.root_execution.clone(),
        final_state_digest: statement.final_state_digest,
        final_pc,
        halted,
    })
}

pub(crate) fn verify_rv32im_kernel_export_witness_with_output(
    relation: &Rv32imKernelExportRelation,
    witness: &Rv32imKernelExportWitness,
) -> Result<Rv32imKernelExportRelationResult, SimpleKernelError> {
    if relation.digest != relation.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export relation digest mismatch".into(),
        ));
    }
    if witness.digest != witness.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export witness digest mismatch".into(),
        ));
    }
    if relation.chunk_count as usize != relation.chunk_surfaces.len() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export relation chunk-count surface mismatch".into(),
        ));
    }
    if relation.chunk_count as usize != witness.chunk_handoffs.len() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export witness chunk count does not match relation".into(),
        ));
    }

    let mut chunk_handoffs = Vec::with_capacity(witness.chunk_handoffs.len());
    let mut public_step_count = 0u64;
    for (chunk_index, (handoff, surface)) in witness
        .chunk_handoffs
        .iter()
        .zip(relation.chunk_surfaces.iter())
        .enumerate()
    {
        if handoff.digest != handoff.expected_digest() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM kernel export witness chunk {chunk_index} digest mismatch"
            )));
        }
        if handoff.bridge_handoff.digest != handoff.bridge_handoff.expected_digest() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM kernel export bridge handoff chunk {chunk_index} digest mismatch"
            )));
        }

        let public_chunk = handoff.chunk_input.public();
        let public_chunk_surface_digest = rv32im_public_chunk_digest(&public_chunk);
        let public_chunk_instance_digest = public_chunk_digest(&public_chunk);
        if handoff.bridge_handoff.chunk_index != chunk_index as u64 {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM kernel export bridge handoff chunk {chunk_index} index mismatch"
            )));
        }
        if handoff.bridge_handoff.chunk_start_index != public_chunk.start_index as u64 {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM kernel export bridge handoff chunk {chunk_index} start-index mismatch"
            )));
        }
        if handoff.bridge_handoff.public_step_count != public_chunk.steps.len() as u64 {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM kernel export bridge handoff chunk {chunk_index} step-count mismatch"
            )));
        }
        if handoff.prepared_step_digests.len() != handoff.chunk_input.steps.len() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM kernel export witness chunk {chunk_index} prepared-step digest count does not match the chunk input"
            )));
        }
        if handoff.bridge_handoff.step_bindings.len() != handoff.chunk_input.steps.len() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM kernel export bridge handoff chunk {chunk_index} binding count does not match the chunk input"
            )));
        }
        for (chunk_local_index, ((binding, step), cached_prepared_step_digest)) in handoff
            .bridge_handoff
            .step_bindings
            .iter()
            .zip(handoff.chunk_input.steps.iter())
            .zip(handoff.prepared_step_digests.iter())
            .enumerate()
        {
            if binding.digest != binding.expected_digest() {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV32IM prepared-step bridge binding {chunk_index}:{chunk_local_index} digest mismatch"
                )));
            }
            if binding.logical_index != (public_chunk.start_index + chunk_local_index) as u64 {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV32IM prepared-step bridge binding {chunk_index}:{chunk_local_index} lost logical row order"
                )));
            }
            let expected_prepared_step_digest = prepared_step_digest(step);
            if *cached_prepared_step_digest != expected_prepared_step_digest {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV32IM prepared-step bridge binding {chunk_index}:{chunk_local_index} cached digest does not match the carried chunk input"
                )));
            }
            if binding.prepared_step_digest != *cached_prepared_step_digest {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV32IM prepared-step bridge binding {chunk_index}:{chunk_local_index} does not match the carried chunk input"
                )));
            }
        }
        if surface.digest != surface.expected_digest() {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM kernel export surface chunk {chunk_index} digest mismatch"
            )));
        }
        if public_chunk_surface_digest != surface.public_chunk_digest {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM kernel export witness chunk {chunk_index} public chunk does not match relation"
            )));
        }
        if handoff.bridge_handoff.digest != surface.bridge_handoff_digest {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM kernel export witness chunk {chunk_index} bridge handoff does not match relation"
            )));
        }

        public_step_count += public_chunk.steps.len() as u64;
        chunk_handoffs.push(Rv32imVerifiedKernelChunkHandoff {
            chunk_input: handoff.chunk_input.clone(),
            public_chunk,
            public_chunk_digest: public_chunk_surface_digest,
            public_chunk_instance_digest,
            prepared_step_digests: handoff.prepared_step_digests.clone(),
            bridge_handoff: handoff.bridge_handoff.clone(),
        });
    }

    if public_step_count != relation.public_step_count {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export witness public-step count does not match relation".into(),
        ));
    }

    Ok(Rv32imKernelExportRelationResult {
        fold_schedule: relation.fold_schedule,
        chunk_handoffs,
        final_state_digest: relation.final_state_digest,
        final_pc: relation.final_pc,
        halted: relation.halted,
    })
}

pub(crate) fn verify_rv32im_kernel_export_proof_with_output(
    expected_relation_digest: [u8; 32],
    proof: &Rv32imKernelExportProof,
) -> Result<Rv32imKernelExportRelationResult, SimpleKernelError> {
    let (relation, verified_relation_result) = verify_rv32im_kernel_export_proof_with_relation_output(proof)?;
    if relation.digest != expected_relation_digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export proof relation digest does not match the carried folded statement".into(),
        ));
    }
    Ok(verified_relation_result)
}

pub fn verify_rv32im_kernel_export_proof_with_relation_output(
    proof: &Rv32imKernelExportProof,
) -> Result<(Rv32imKernelExportRelation, Rv32imKernelExportRelationResult), SimpleKernelError> {
    if proof.digest != proof.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export proof digest mismatch".into(),
        ));
    }
    let verified_source = verify_rv32im_kernel_export_source_with_output(&proof.source)?;
    let (relation, verified_relation_result) =
        build_rv32im_kernel_export_relation_from_verified_core(&verified_source)?;
    let verified_witness = verify_rv32im_kernel_export_witness_with_output(&relation, &proof.witness)?;
    if verified_witness.fold_schedule != verified_relation_result.fold_schedule
        || verified_witness.final_state_digest != verified_relation_result.final_state_digest
        || verified_witness.final_pc != verified_relation_result.final_pc
        || verified_witness.halted != verified_relation_result.halted
        || verified_witness.chunk_handoffs.len() != verified_relation_result.chunk_handoffs.len()
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export proof witness does not match the verified export core".into(),
        ));
    }
    for (witness_handoff, expected_handoff) in verified_witness
        .chunk_handoffs
        .iter()
        .zip(verified_relation_result.chunk_handoffs.iter())
    {
        if witness_handoff.public_chunk_digest != expected_handoff.public_chunk_digest
            || witness_handoff.prepared_step_digests != expected_handoff.prepared_step_digests
            || witness_handoff.bridge_handoff != expected_handoff.bridge_handoff
        {
            return Err(SimpleKernelError::Bridge(
                "RV32IM kernel export proof handoff surface does not match the verified export core".into(),
            ));
        }
    }
    Ok((relation, verified_relation_result))
}

pub(crate) fn build_rv32im_kernel_export_relation_from_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<(Rv32imKernelExportRelation, Rv32imKernelExportRelationResult), SimpleKernelError> {
    verify_accepted_proof_artifact_export_core_with_perf(artifact)?;
    build_rv32im_kernel_export_relation_from_verified_artifact(artifact)
}

fn build_rv32im_kernel_export_relation_from_verified_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<(Rv32imKernelExportRelation, Rv32imKernelExportRelationResult), SimpleKernelError> {
    build_rv32im_kernel_export_relation_from_verified_core(&kernel_export_core_from_accepted_artifact(artifact))
}

fn build_rv32im_kernel_export_relation_from_verified_core(
    core: &Rv32imVerifiedKernelExportCore,
) -> Result<(Rv32imKernelExportRelation, Rv32imKernelExportRelationResult), SimpleKernelError> {
    build_rv32im_kernel_export_relation_from_verified_parts(
        core.fold_schedule,
        &core.public_chunks,
        &core.root_execution,
        core.final_state_digest,
        core.final_pc,
        core.halted,
    )
}

fn build_rv32im_kernel_export_relation_from_verified_parts(
    fold_schedule: FoldSchedule,
    public_chunks: &[PublicChunk],
    root_execution: &RootExecutionBundle,
    final_state_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
) -> Result<(Rv32imKernelExportRelation, Rv32imKernelExportRelationResult), SimpleKernelError> {
    let result = build_export_relation_result_from_parts(
        fold_schedule,
        public_chunks,
        root_execution,
        final_state_digest,
        final_pc,
        halted,
    )?;
    let mut relation = Rv32imKernelExportRelation {
        fold_schedule: result.fold_schedule,
        chunk_count: result.chunk_handoffs.len() as u64,
        public_step_count: result
            .chunk_handoffs
            .iter()
            .map(|handoff| handoff.public_chunk.steps.len() as u64)
            .sum(),
        final_state_digest: result.final_state_digest,
        final_pc: result.final_pc,
        halted: result.halted,
        chunk_surfaces: result
            .chunk_handoffs
            .iter()
            .map(chunk_export_surface)
            .collect(),
        digest: [0; 32],
    };
    relation.digest = relation.expected_digest();
    Ok((relation, result))
}

fn build_rv32im_kernel_export_relation_from_verified_parts_and_chunk_inputs(
    fold_schedule: FoldSchedule,
    public_chunks: &[PublicChunk],
    chunk_inputs: Vec<ChunkInput>,
    root_execution: &RootExecutionBundle,
    final_state_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
) -> Result<(Rv32imKernelExportRelation, Rv32imKernelExportRelationResult), SimpleKernelError> {
    let result = build_export_relation_result_from_partitioned_chunks(
        fold_schedule,
        public_chunks,
        chunk_inputs,
        root_execution,
        final_state_digest,
        final_pc,
        halted,
    )?;
    let mut relation = Rv32imKernelExportRelation {
        fold_schedule: result.fold_schedule,
        chunk_count: result.chunk_handoffs.len() as u64,
        public_step_count: result
            .chunk_handoffs
            .iter()
            .map(|handoff| handoff.public_chunk.steps.len() as u64)
            .sum(),
        final_state_digest: result.final_state_digest,
        final_pc: result.final_pc,
        halted: result.halted,
        chunk_surfaces: result
            .chunk_handoffs
            .iter()
            .map(chunk_export_surface)
            .collect(),
        digest: [0; 32],
    };
    relation.digest = relation.expected_digest();
    Ok((relation, result))
}

fn chunk_input_matches_public_chunk(chunk_input: &ChunkInput, public_chunk: &PublicChunk) -> bool {
    chunk_input.start_index == public_chunk.start_index
        && chunk_input.steps.len() == public_chunk.steps.len()
        && chunk_input
            .steps
            .iter()
            .zip(public_chunk.steps.iter())
            .all(|(step, public_step)| {
                step.label == public_step.label
                    && step.mcs.m_in == public_step.mcs.m_in
                    && step.mcs.x == public_step.mcs.x
                    && step.mcs.c.d == public_step.mcs.c.d
                    && step.mcs.c.kappa == public_step.mcs.c.kappa
                    && step.mcs.c.data == public_step.mcs.c.data
            })
}

fn kernel_export_core_from_accepted_artifact(artifact: &Rv32imAcceptedProofArtifact) -> Rv32imVerifiedKernelExportCore {
    Rv32imVerifiedKernelExportCore {
        fold_schedule: artifact.main_lane.packaged.statement.fold_schedule,
        public_chunks: artifact.main_lane.packaged.statement.chunks.clone(),
        root_execution: artifact.root_execution.clone(),
        final_state_digest: artifact.statement.final_state_digest,
        final_pc: artifact.statement.final_pc,
        halted: artifact.statement.halted,
    }
}

fn build_export_relation_result_from_parts(
    fold_schedule: FoldSchedule,
    public_chunks: &[PublicChunk],
    root_execution: &RootExecutionBundle,
    final_state_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
) -> Result<Rv32imKernelExportRelationResult, SimpleKernelError> {
    let prepared_steps = build_prepared_steps_from_execution_rows(&root_execution.execution_rows)?;
    let chunk_inputs = partition_step_inputs(fold_schedule, prepared_steps)?;
    build_export_relation_result_from_partitioned_chunks(
        fold_schedule,
        public_chunks,
        chunk_inputs,
        root_execution,
        final_state_digest,
        final_pc,
        halted,
    )
}

fn build_export_relation_result_from_partitioned_chunks(
    fold_schedule: FoldSchedule,
    public_chunks: &[PublicChunk],
    chunk_inputs: Vec<ChunkInput>,
    root_execution: &RootExecutionBundle,
    final_state_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
) -> Result<Rv32imKernelExportRelationResult, SimpleKernelError> {
    if chunk_inputs.len() != public_chunks.len() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export chunk partition does not match the verified main-lane statement".into(),
        ));
    }

    let total_public_steps: usize = public_chunks.iter().map(|chunk| chunk.steps.len()).sum();
    if total_public_steps != root_execution.prepared_step_bindings.bindings.len() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM kernel export prepared-step count does not match root-execution bindings".into(),
        ));
    }
    let mut chunk_handoffs = Vec::with_capacity(public_chunks.len());
    for (chunk_index, (chunk_input, public_chunk)) in chunk_inputs
        .into_iter()
        .zip(public_chunks.iter().cloned())
        .enumerate()
    {
        let public_chunk_surface_digest = rv32im_public_chunk_digest(&public_chunk);
        let public_chunk_instance_digest = public_chunk_digest(&public_chunk);
        if !chunk_input_matches_public_chunk(&chunk_input, &public_chunk) {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM kernel export public chunk {chunk_index} does not match the verified main-lane statement"
            )));
        }
        let prepared_step_digests = prepared_step_digests(&chunk_input.steps);
        let bridge_handoff = build_chunk_bridge_handoff(
            root_execution,
            chunk_index as u64,
            &chunk_input,
            &public_chunk,
            &prepared_step_digests,
        )?;
        chunk_handoffs.push(Rv32imVerifiedKernelChunkHandoff {
            chunk_input,
            public_chunk,
            public_chunk_digest: public_chunk_surface_digest,
            public_chunk_instance_digest,
            prepared_step_digests,
            bridge_handoff,
        });
    }

    Ok(Rv32imKernelExportRelationResult {
        fold_schedule,
        chunk_handoffs,
        final_state_digest,
        final_pc,
        halted,
    })
}

fn build_chunk_bridge_handoff(
    root_execution: &RootExecutionBundle,
    chunk_index: u64,
    chunk_input: &ChunkInput,
    public_chunk: &PublicChunk,
    prepared_step_digests: &[[u8; 32]],
) -> Result<Rv32imChunkBridgeHandoff, SimpleKernelError> {
    let start = public_chunk.start_index;
    let end = start + public_chunk.steps.len();
    let bindings = root_execution
        .prepared_step_bindings
        .bindings
        .get(start..end)
        .ok_or_else(|| SimpleKernelError::Bridge("RV32IM chunk bridge binding range exceeds root execution".into()))?;
    if chunk_input.steps.len() != bindings.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk bridge handoff {chunk_index} prepared-step count does not match the carried chunk input"
        )));
    }
    let routes = root_execution
        .row_chunk_routes
        .get(start..end)
        .ok_or_else(|| SimpleKernelError::Bridge("RV32IM chunk bridge routing range exceeds root execution".into()))?;
    if chunk_input.steps.len() != routes.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk bridge handoff {chunk_index} route count does not match the carried chunk input"
        )));
    }
    if chunk_input.steps.len() != prepared_step_digests.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM chunk bridge handoff {chunk_index} prepared-step digest count does not match the carried chunk input"
        )));
    }
    let mut handoff = Rv32imChunkBridgeHandoff {
        chunk_index,
        chunk_start_index: start as u64,
        public_step_count: public_chunk.steps.len() as u64,
        step_bindings: bindings
            .iter()
            .zip(routes.iter())
            .zip(prepared_step_digests.into_iter())
            .enumerate()
            .map(|(chunk_local_index, ((binding, route), prepared_step_digest))| {
                if route.logical_index != (start + chunk_local_index) as u64
                    || route.chunk_index != chunk_index
                    || route.chunk_start_index != start as u64
                    || route.chunk_local_index != chunk_local_index as u64
                {
                    return Err(SimpleKernelError::Bridge(format!(
                        "RV32IM chunk bridge handoff {chunk_index}:{chunk_local_index} route alignment mismatch"
                    )));
                }
                let binding = Rv32imPreparedStepBridgeBinding {
                    logical_index: (start + chunk_local_index) as u64,
                    trace_index: binding.trace_index as u64,
                    row_binding_digest: binding.digest,
                    prepared_step_digest: *prepared_step_digest,
                    digest: [0; 32],
                };
                Ok(Rv32imPreparedStepBridgeBinding {
                    digest: binding.expected_digest(),
                    ..binding
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        digest: [0; 32],
    };
    handoff.digest = handoff.expected_digest();
    Ok(handoff)
}

fn chunk_export_surface(handoff: &Rv32imVerifiedKernelChunkHandoff) -> Rv32imChunkExportSurface {
    let bridge_handoff_digest = handoff.bridge_handoff.digest;
    let mut surface = Rv32imChunkExportSurface {
        public_chunk_digest: handoff.public_chunk_digest,
        bridge_handoff_digest,
        digest: [0; 32],
    };
    surface.digest = surface.expected_digest();
    surface
}

fn kernel_export_witness_from_result(result: &Rv32imKernelExportRelationResult) -> Rv32imKernelExportWitness {
    let chunk_handoffs = result
        .chunk_handoffs
        .iter()
        .map(kernel_chunk_export_witness)
        .collect();
    let mut witness = Rv32imKernelExportWitness {
        chunk_handoffs,
        digest: [0; 32],
    };
    witness.digest = witness.expected_digest();
    witness
}

fn kernel_chunk_export_witness(handoff: &Rv32imVerifiedKernelChunkHandoff) -> Rv32imKernelChunkExportWitness {
    let mut witness = Rv32imKernelChunkExportWitness {
        chunk_input: handoff.chunk_input.clone(),
        prepared_step_digests: handoff.prepared_step_digests.clone(),
        bridge_handoff: handoff.bridge_handoff.clone(),
        digest: [0; 32],
    };
    witness.digest = witness.expected_digest();
    witness
}

fn kernel_export_main_lane_proof_from_bundle(bundle: &Rv32imMainLaneProofBundle) -> Rv32imKernelExportMainLaneProof {
    let proof = Rv32imKernelExportMainLaneProof {
        packaged: bundle.packaged.clone(),
        digest: [0; 32],
    };
    Rv32imKernelExportMainLaneProof {
        digest: proof.expected_digest(),
        ..proof
    }
}

pub(crate) fn rv32im_public_chunk_digest(chunk: &PublicChunk) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/public_chunk");
    tr.append_u64s(
        b"rv32im/public_chunk/meta",
        &[chunk.start_index as u64, chunk.steps.len() as u64],
    );
    for digest in public_step_digests(&chunk.steps) {
        tr.append_fields(b"rv32im/public_chunk/step", &digest32_as_fields(digest));
    }
    tr.digest32()
}

fn append_step_bindings(tr: &mut Poseidon2Transcript, bindings: &[Rv32imPreparedStepBridgeBinding]) {
    tr.append_u64s(
        b"rv32im/chunk_bridge_handoff/step_binding_count",
        &[bindings.len() as u64],
    );
    for binding in bindings {
        tr.append_message(b"rv32im/chunk_bridge_handoff/step_binding_digest", &binding.digest);
    }
}

fn append_chunk_surfaces(tr: &mut Poseidon2Transcript, surfaces: &[Rv32imChunkExportSurface]) {
    tr.append_u64s(
        b"rv32im/kernel_export_relation/chunk_surface_count",
        &[surfaces.len() as u64],
    );
    for surface in surfaces {
        tr.append_message(b"rv32im/kernel_export_relation/chunk_surface_digest", &surface.digest);
    }
}
