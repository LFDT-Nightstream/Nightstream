//! Owns the final public RV64IM proof API above the simple-kernel path.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};
use std::{thread, time::Instant};

use crate::proof::{ChunkInput, FoldSchedule, PackagedProof};

use super::main_lane_artifact::build_simple_kernel_main_lane_artifact_from_summary;
use super::proof_accepted::{
    accepted_proof_artifact_from_legacy_proof, accepted_proof_artifact_from_prover_materials,
    audit_bundle_from_legacy_proof, Rv64imAcceptedProofArtifact, Rv64imAuditBundle,
};
use super::proof_bridge::{main_lane_proof_bundle_from_artifact, proof_from_public_kernel_and_main_lane_bundle};
use super::proof_staged_verify::verify_accepted_proof_artifact_with_perf;
use super::proof_verify::{
    validate_public_proof_against_input_with_perf, verify_kernel_output_from_public_proof_with_perf,
};
use super::proof_witness::{
    proof_witness_bundle_from_public_kernel_and_trace_stages, Rv64imKernelClaimProofBundle,
    Rv64imKernelOpeningProofBundle, Rv64imProofWitnessBundle, Rv64imStageClaimProofBundle,
    Rv64imStagePackageProofBundle, Rv64imStageWitnessProjectionBundle, Rv64imTraceProjectionBundle,
};
use super::simple::{
    build_public_simple_kernel_output_and_witness_from_derived_with_perf,
    build_public_simple_kernel_output_and_witness_with_perf, prove_root_main_lane_packaged_proof_with_inputs_and_perf,
};
use super::{
    build_parity_case_from_source, RootLaneColumns, RootLaneCommitmentSummaryArtifact, Rv64imProofProvePerf,
    Rv64imPublicProofVerifyPerf, SimpleKernelError, SimpleKernelPublicInput,
};

pub type Rv64imProofInput = SimpleKernelPublicInput;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imProofStatement {
    pub root_params_id: [u8; 32],
    pub fold_schedule: FoldSchedule,
    pub chunk_count: u64,
    pub stage_claims_digest: [u8; 32],
    pub stage_packages_digest: [u8; 32],
    pub kernel_opening_digest: [u8; 32],
    pub prepared_step_bindings_digest: [u8; 32],
    pub execution_digest: [u8; 32],
    pub final_state_digest: [u8; 32],
    pub transcript_final_digest: [u8; 32],
    pub main_lane_surface_digest: [u8; 32],
    pub root_lane_columns_digest: [u8; 32],
    pub public_step_count: u64,
    pub initial_pc: u64,
    pub final_pc: u64,
    pub halted: bool,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imAcceptedProofStatementBinding {
    pub proof_statement_digest: [u8; 32],
    pub kernel_opening_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imAcceptedProofMainLaneBinding {
    pub main_lane_bundle_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imAcceptedProofTerminalBinding {
    pub final_state_digest: [u8; 32],
    pub final_pc: u64,
    pub halted: bool,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imAcceptedProofClaim {
    pub root_params_id: [u8; 32],
    pub statement: Rv64imAcceptedProofStatementBinding,
    pub main_lane: Rv64imAcceptedProofMainLaneBinding,
    pub terminal: Rv64imAcceptedProofTerminalBinding,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imMainLaneClaimBinding {
    pub main_lane_bundle_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imMainLaneClaim {
    pub root_params_id: [u8; 32],
    pub binding: Rv64imMainLaneClaimBinding,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imKernelOpeningStageClaimBinding {
    pub stage_claims_digest: [u8; 32],
    pub stage_packages_digest: [u8; 32],
    pub kernel_opening_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imKernelOpeningTerminalClaimBinding {
    pub prepared_step_bindings_digest: [u8; 32],
    pub execution_digest: [u8; 32],
    pub transcript_final_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imKernelOpeningClaim {
    pub root_params_id: [u8; 32],
    pub stages: Rv64imKernelOpeningStageClaimBinding,
    pub terminal: Rv64imKernelOpeningTerminalClaimBinding,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imJointOpeningClaimBinding {
    pub proof_statement_digest: [u8; 32],
    pub main_lane_claim_digest: [u8; 32],
    pub kernel_opening_claim_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imJointOpeningClaim {
    pub root_params_id: [u8; 32],
    pub binding: Rv64imJointOpeningClaimBinding,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imRoot0StageClaimBinding {
    pub stage1_digest: [u8; 32],
    pub stage2_digest: [u8; 32],
    pub stage3_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imRoot0TerminalClaimBinding {
    pub root0_digest: [u8; 32],
    pub execution_digest: [u8; 32],
    pub final_state_digest: [u8; 32],
    pub transcript_final_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imRoot0Claim {
    pub root_params_id: [u8; 32],
    pub stages: Rv64imRoot0StageClaimBinding,
    pub terminal: Rv64imRoot0TerminalClaimBinding,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imKernelClaimBundle {
    pub accepted: Rv64imAcceptedProofClaim,
    pub main_lane: Rv64imMainLaneClaim,
    pub opening: Rv64imKernelOpeningClaim,
    pub joint_opening: Rv64imJointOpeningClaim,
    pub root0: Rv64imRoot0Claim,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imMainLaneProofBinding {
    pub root_lane_columns_digest: [u8; 32],
    pub root_lane_commitment_digest: [u8; 32],
    pub fold_schedule: FoldSchedule,
    pub chunk_count: u64,
    pub public_step_count: u64,
    pub digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imPublicProofOptions {
    pub root_fold_schedule: FoldSchedule,
}

impl Default for Rv64imPublicProofOptions {
    fn default() -> Self {
        Self {
            root_fold_schedule: FoldSchedule::WholeTrace,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imMainLaneProofBundle {
    pub binding: Rv64imMainLaneProofBinding,
    pub packaged: PackagedProof,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imMainLaneProofSummaryBundle {
    pub binding: Rv64imMainLaneProofBinding,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imKernelProofBundle {
    pub root_params_id: [u8; 32],
    pub trace: Rv64imTraceProjectionBundle,
    pub stages: Rv64imStageWitnessProjectionBundle,
    pub stage_claims: Rv64imStageClaimProofBundle,
    pub stage_packages: Rv64imStagePackageProofBundle,
    pub kernel_opening: Rv64imKernelOpeningProofBundle,
    pub kernel_claims: Rv64imKernelClaimProofBundle,
    pub root_lane_columns: RootLaneColumns,
    pub root_lane_commitment: RootLaneCommitmentSummaryArtifact,
    pub main_lane: Rv64imMainLaneProofBundle,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct Rv64imProof {
    pub claim: Rv64imKernelClaimBundle,
    pub statement: Rv64imProofStatement,
    pub kernel: Rv64imKernelProofBundle,
    pub witness: Rv64imProofWitnessBundle,
}

#[derive(Clone, Debug)]
pub(crate) struct Rv64imPublicProofProverSeam {
    pub proof: Rv64imProof,
    pub kernel: super::simple::PublicSimpleKernelOutput,
    pub sidecar: super::simple::PublicSimpleKernelWitnessSidecar,
    pub main_lane_inputs: Vec<ChunkInput>,
}

fn allow_public_proof_parallel_branches() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        thread::available_parallelism()
            .map(|parallelism| parallelism.get() > 1)
            .unwrap_or(false)
    }

    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

impl Rv64imProofStatement {
    pub fn recompute_digest(&self) -> [u8; 32] {
        self.expected_digest()
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/proof_statement");
        tr.append_message(b"rv64im/proof_statement/root_params_id", &self.root_params_id);
        tr.append_u64s(
            b"rv64im/proof_statement/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_message(b"rv64im/proof_statement/stage_claims_digest", &self.stage_claims_digest);
        tr.append_message(
            b"rv64im/proof_statement/stage_packages_digest",
            &self.stage_packages_digest,
        );
        tr.append_message(
            b"rv64im/proof_statement/kernel_opening_digest",
            &self.kernel_opening_digest,
        );
        tr.append_message(
            b"rv64im/proof_statement/prepared_step_bindings_digest",
            &self.prepared_step_bindings_digest,
        );
        tr.append_message(b"rv64im/proof_statement/execution_digest", &self.execution_digest);
        tr.append_message(b"rv64im/proof_statement/final_state_digest", &self.final_state_digest);
        tr.append_message(
            b"rv64im/proof_statement/transcript_final_digest",
            &self.transcript_final_digest,
        );
        tr.append_message(
            b"rv64im/proof_statement/main_lane_surface_digest",
            &self.main_lane_surface_digest,
        );
        tr.append_message(
            b"rv64im/proof_statement/root_lane_columns_digest",
            &self.root_lane_columns_digest,
        );
        tr.append_u64s(
            b"rv64im/proof_statement/meta",
            &[
                self.chunk_count,
                self.public_step_count,
                self.initial_pc,
                self.final_pc,
                self.halted as u64,
            ],
        );
        tr.digest32()
    }
}

impl Rv64imAcceptedProofClaim {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/accepted_proof_claim");
        tr.append_message(b"rv64im/accepted_proof/root_params_id", &self.root_params_id);
        tr.append_message(b"rv64im/accepted_proof/statement_digest", &self.statement.digest);
        tr.append_message(b"rv64im/accepted_proof/main_lane_digest", &self.main_lane.digest);
        tr.append_message(b"rv64im/accepted_proof/terminal_digest", &self.terminal.digest);
        tr.digest32()
    }
}

impl Rv64imAcceptedProofStatementBinding {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/accepted_proof_statement_binding");
        tr.append_message(
            b"rv64im/accepted_proof_statement_binding/proof_statement_digest",
            &self.proof_statement_digest,
        );
        tr.append_message(
            b"rv64im/accepted_proof_statement_binding/kernel_opening_digest",
            &self.kernel_opening_digest,
        );
        tr.digest32()
    }
}

impl Rv64imAcceptedProofMainLaneBinding {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/accepted_proof_main_lane_binding");
        tr.append_message(
            b"rv64im/accepted_proof_main_lane_binding/main_lane_bundle_digest",
            &self.main_lane_bundle_digest,
        );
        tr.digest32()
    }
}

impl Rv64imAcceptedProofTerminalBinding {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/accepted_proof_terminal_binding");
        tr.append_message(
            b"rv64im/accepted_proof_terminal_binding/final_state_digest",
            &self.final_state_digest,
        );
        tr.append_u64s(
            b"rv64im/accepted_proof_terminal_binding/meta",
            &[self.final_pc, self.halted as u64],
        );
        tr.digest32()
    }
}

impl Rv64imMainLaneClaim {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_lane_claim");
        tr.append_message(b"rv64im/main_lane_claim/root_params_id", &self.root_params_id);
        tr.append_message(b"rv64im/main_lane_claim/binding_digest", &self.binding.digest);
        tr.digest32()
    }
}

impl Rv64imMainLaneClaimBinding {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_lane_claim_binding");
        tr.append_message(
            b"rv64im/main_lane_claim_binding/main_lane_bundle_digest",
            &self.main_lane_bundle_digest,
        );
        tr.digest32()
    }
}

impl Rv64imKernelOpeningClaim {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_opening_claim");
        tr.append_message(b"rv64im/kernel_opening_claim/root_params_id", &self.root_params_id);
        tr.append_message(b"rv64im/kernel_opening_claim/stages_digest", &self.stages.digest);
        tr.append_message(b"rv64im/kernel_opening_claim/terminal_digest", &self.terminal.digest);
        tr.digest32()
    }
}

impl Rv64imKernelOpeningStageClaimBinding {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_opening_stage_claim_binding");
        tr.append_message(
            b"rv64im/kernel_opening_stage_claim_binding/stage_claims_digest",
            &self.stage_claims_digest,
        );
        tr.append_message(
            b"rv64im/kernel_opening_stage_claim_binding/stage_packages_digest",
            &self.stage_packages_digest,
        );
        tr.append_message(
            b"rv64im/kernel_opening_stage_claim_binding/kernel_opening_digest",
            &self.kernel_opening_digest,
        );
        tr.digest32()
    }
}

impl Rv64imKernelOpeningTerminalClaimBinding {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_opening_terminal_claim_binding");
        tr.append_message(
            b"rv64im/kernel_opening_terminal_claim_binding/prepared_step_bindings_digest",
            &self.prepared_step_bindings_digest,
        );
        tr.append_message(
            b"rv64im/kernel_opening_terminal_claim_binding/execution_digest",
            &self.execution_digest,
        );
        tr.append_message(
            b"rv64im/kernel_opening_terminal_claim_binding/transcript_final_digest",
            &self.transcript_final_digest,
        );
        tr.digest32()
    }
}

impl Rv64imJointOpeningClaim {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/joint_opening_claim");
        tr.append_message(b"rv64im/joint_opening_claim/root_params_id", &self.root_params_id);
        tr.append_message(b"rv64im/joint_opening_claim/binding_digest", &self.binding.digest);
        tr.digest32()
    }
}

impl Rv64imJointOpeningClaimBinding {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/joint_opening_claim_binding");
        tr.append_message(
            b"rv64im/joint_opening_claim_binding/proof_statement_digest",
            &self.proof_statement_digest,
        );
        tr.append_message(
            b"rv64im/joint_opening_claim_binding/main_lane_claim_digest",
            &self.main_lane_claim_digest,
        );
        tr.append_message(
            b"rv64im/joint_opening_claim_binding/kernel_opening_claim_digest",
            &self.kernel_opening_claim_digest,
        );
        tr.digest32()
    }
}

impl Rv64imRoot0Claim {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/root0_claim");
        tr.append_message(b"rv64im/root0_claim/root_params_id", &self.root_params_id);
        tr.append_message(b"rv64im/root0_claim/stages_digest", &self.stages.digest);
        tr.append_message(b"rv64im/root0_claim/terminal_digest", &self.terminal.digest);
        tr.digest32()
    }
}

impl Rv64imRoot0StageClaimBinding {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/root0_stage_claim_binding");
        tr.append_message(b"rv64im/root0_stage_claim_binding/stage1_digest", &self.stage1_digest);
        tr.append_message(b"rv64im/root0_stage_claim_binding/stage2_digest", &self.stage2_digest);
        tr.append_message(b"rv64im/root0_stage_claim_binding/stage3_digest", &self.stage3_digest);
        tr.digest32()
    }
}

impl Rv64imRoot0TerminalClaimBinding {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/root0_terminal_claim_binding");
        tr.append_message(b"rv64im/root0_terminal_claim_binding/root0_digest", &self.root0_digest);
        tr.append_message(
            b"rv64im/root0_terminal_claim_binding/execution_digest",
            &self.execution_digest,
        );
        tr.append_message(
            b"rv64im/root0_terminal_claim_binding/final_state_digest",
            &self.final_state_digest,
        );
        tr.append_message(
            b"rv64im/root0_terminal_claim_binding/transcript_final_digest",
            &self.transcript_final_digest,
        );
        tr.digest32()
    }
}

impl Rv64imKernelClaimBundle {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_claim_bundle");
        tr.append_message(b"rv64im/kernel_claim_bundle/accepted_digest", &self.accepted.digest);
        tr.append_message(b"rv64im/kernel_claim_bundle/main_lane_digest", &self.main_lane.digest);
        tr.append_message(b"rv64im/kernel_claim_bundle/opening_digest", &self.opening.digest);
        tr.append_message(
            b"rv64im/kernel_claim_bundle/joint_opening_digest",
            &self.joint_opening.digest,
        );
        tr.append_message(b"rv64im/kernel_claim_bundle/root0_digest", &self.root0.digest);
        tr.digest32()
    }
}

impl Rv64imMainLaneProofBinding {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_lane_proof_binding");
        tr.append_message(
            b"rv64im/main_lane_proof_binding/root_lane_columns_digest",
            &self.root_lane_columns_digest,
        );
        tr.append_message(
            b"rv64im/main_lane_proof_binding/root_lane_commitment_digest",
            &self.root_lane_commitment_digest,
        );
        tr.append_u64s(
            b"rv64im/main_lane_proof_binding/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_u64s(
            b"rv64im/main_lane_proof_binding/meta",
            &[self.chunk_count, self.public_step_count],
        );
        tr.digest32()
    }
}

impl Rv64imMainLaneProofBundle {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_lane_proof_bundle");
        tr.append_message(b"rv64im/main_lane_proof_bundle/binding_digest", &self.binding.digest);
        tr.append_message(
            b"rv64im/main_lane_proof_bundle/statement_digest",
            &self.packaged.statement.digest,
        );
        tr.append_message(
            b"rv64im/main_lane_proof_bundle/proof_digest",
            &self.packaged.proof.proof_digest,
        );
        tr.digest32()
    }

    pub fn root_lane_columns_digest(&self) -> [u8; 32] {
        self.binding.root_lane_columns_digest
    }

    pub fn root_lane_commitment_digest(&self) -> [u8; 32] {
        self.binding.root_lane_commitment_digest
    }

    pub fn public_step_count(&self) -> u64 {
        self.binding.public_step_count
    }

    pub fn fold_schedule(&self) -> FoldSchedule {
        self.binding.fold_schedule
    }

    pub fn chunk_count(&self) -> u64 {
        self.binding.chunk_count
    }

    pub fn statement_digest(&self) -> [u8; 32] {
        self.packaged.statement.digest
    }

    pub fn proof_digest(&self) -> [u8; 32] {
        self.packaged.proof.proof_digest
    }

    pub fn summary(&self) -> Rv64imMainLaneProofSummaryBundle {
        let summary = Rv64imMainLaneProofSummaryBundle {
            binding: self.binding.clone(),
            digest: [0; 32],
        };
        Rv64imMainLaneProofSummaryBundle {
            digest: summary.expected_digest(),
            ..summary
        }
    }
}

impl Rv64imMainLaneProofSummaryBundle {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_lane_proof_summary_bundle");
        tr.append_message(
            b"rv64im/main_lane_proof_summary_bundle/binding_digest",
            &self.binding.digest,
        );
        tr.digest32()
    }
}

impl Rv64imKernelProofBundle {
    pub(super) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_proof_bundle");
        tr.append_message(b"rv64im/kernel_proof_bundle/root_params_id", &self.root_params_id);
        tr.append_message(b"rv64im/kernel_proof_bundle/trace_digest", &self.trace.digest);
        tr.append_message(b"rv64im/kernel_proof_bundle/stages_digest", &self.stages.digest);
        tr.append_message(
            b"rv64im/kernel_proof_bundle/stage_claims_digest",
            &self.stage_claims.digest,
        );
        tr.append_message(
            b"rv64im/kernel_proof_bundle/stage_packages_digest",
            &self.stage_packages.digest,
        );
        tr.append_message(
            b"rv64im/kernel_proof_bundle/kernel_opening_digest",
            &self.kernel_opening.digest,
        );
        tr.append_message(
            b"rv64im/kernel_proof_bundle/kernel_claims_digest",
            &self.kernel_claims.digest,
        );
        tr.append_message(
            b"rv64im/kernel_proof_bundle/root_lane_columns_digest",
            &self.root_lane_columns.digest,
        );
        tr.append_message(
            b"rv64im/kernel_proof_bundle/root_lane_commitment_digest",
            &self.root_lane_commitment.digest,
        );
        tr.append_message(b"rv64im/kernel_proof_bundle/main_lane_digest", &self.main_lane.digest);
        tr.digest32()
    }
}

pub fn build_rv64im_audit_witness_bundle(
    input: &Rv64imProofInput,
) -> Result<Rv64imProofWitnessBundle, SimpleKernelError> {
    let ((public, sidecar), _) = build_public_simple_kernel_output_and_witness_with_perf(input)?;
    proof_witness_bundle_from_public_kernel_and_trace_stages(&public, &sidecar.trace, &sidecar.stages)
}

pub fn build_rv64im_accepted_proof_artifact(
    proof: &Rv64imProof,
) -> Result<Rv64imAcceptedProofArtifact, SimpleKernelError> {
    accepted_proof_artifact_from_legacy_proof(proof)
}

pub fn build_rv64im_audit_bundle(proof: &Rv64imProof) -> Rv64imAuditBundle {
    audit_bundle_from_legacy_proof(proof)
}

fn prove_rv64im_public_proof_and_sidecar_with_perf(
    input: &Rv64imProofInput,
    options: Rv64imPublicProofOptions,
) -> Result<
    (
        (
            super::simple::PublicSimpleKernelOutput,
            super::simple::PublicSimpleKernelWitnessSidecar,
        ),
        Rv64imProof,
        Rv64imProofProvePerf,
    ),
    SimpleKernelError,
> {
    let (built, perf) = prove_rv64im_public_proof_prover_seam_with_perf(input, options)?;
    let Rv64imPublicProofProverSeam {
        proof, kernel, sidecar, ..
    } = built;
    Ok(((kernel, sidecar), proof, perf))
}

pub(crate) fn prove_rv64im_public_proof_prover_seam_with_perf(
    input: &Rv64imProofInput,
    options: Rv64imPublicProofOptions,
) -> Result<(Rv64imPublicProofProverSeam, Rv64imProofProvePerf), SimpleKernelError> {
    let total_started = Instant::now();
    let shared_trace_started = Instant::now();
    let (_, derived) = build_parity_case_from_source(input.source.clone(), input.max_steps)?;
    let shared_trace_ms = shared_trace_started.elapsed().as_secs_f64() * 1_000.0;

    let ((kernel, sidecar), simple_kernel, root_main_lane, main_lane_inputs, root_main_lane_perf) =
        if allow_public_proof_parallel_branches() {
            thread::scope(|scope| {
                let root_rows = &derived.execution_rows;
                let schedule = options.root_fold_schedule;
                let root_handle =
                    scope.spawn(move || prove_root_main_lane_packaged_proof_with_inputs_and_perf(root_rows, schedule));
                let ((kernel, sidecar), simple_kernel) =
                    build_public_simple_kernel_output_and_witness_from_derived_with_perf(&derived)?;
                let (root_main_lane, main_lane_inputs, root_main_lane_perf) = root_handle
                    .join()
                    .map_err(|_| SimpleKernelError::Proof("RV64IM root main-lane worker panicked".into()))??;
                Ok::<_, SimpleKernelError>((
                    (kernel, sidecar),
                    simple_kernel,
                    root_main_lane,
                    main_lane_inputs,
                    root_main_lane_perf,
                ))
            })?
        } else {
            let ((kernel, sidecar), simple_kernel) =
                build_public_simple_kernel_output_and_witness_from_derived_with_perf(&derived)?;
            let (root_main_lane, main_lane_inputs, root_main_lane_perf) =
                prove_root_main_lane_packaged_proof_with_inputs_and_perf(
                    &sidecar.trace.execution_rows,
                    options.root_fold_schedule,
                )?;
            (
                (kernel, sidecar),
                simple_kernel,
                root_main_lane,
                main_lane_inputs,
                root_main_lane_perf,
            )
        };
    let parallel_overlap_ms = simple_kernel.total_ms.min(root_main_lane_perf.total_ms);
    let main_lane_started = Instant::now();
    let main_lane = build_simple_kernel_main_lane_artifact_from_summary(
        &kernel.root_lane_columns,
        &kernel.root_lane_commitment,
        options.root_fold_schedule,
    )?;
    let main_lane_ms = main_lane_started.elapsed().as_secs_f64() * 1_000.0;
    let export_started = Instant::now();
    let witness = proof_witness_bundle_from_public_kernel_and_trace_stages(&kernel, &sidecar.trace, &sidecar.stages)?;
    let main_lane_bundle = main_lane_proof_bundle_from_artifact(&main_lane, root_main_lane);
    let proof = proof_from_public_kernel_and_main_lane_bundle(&kernel, main_lane_bundle, witness)?;
    let public_export_ms = export_started.elapsed().as_secs_f64() * 1_000.0;
    let perf = Rv64imProofProvePerf {
        shared_trace_ms,
        simple_kernel,
        parallel_overlap_ms,
        main_lane_ms,
        root_main_lane: root_main_lane_perf,
        public_export_ms,
        total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
    };
    Ok((
        Rv64imPublicProofProverSeam {
            proof,
            kernel,
            sidecar,
            main_lane_inputs,
        },
        perf,
    ))
}

pub fn prove_rv64im_public_proof(input: &Rv64imProofInput) -> Result<Rv64imProof, SimpleKernelError> {
    let (proof, _) = prove_rv64im_public_proof_with_perf(input)?;
    Ok(proof)
}

pub fn prove_rv64im_accepted_proof(
    input: &Rv64imProofInput,
) -> Result<(Rv64imAcceptedProofArtifact, Rv64imAuditBundle), SimpleKernelError> {
    let ((artifact, audit), _) = prove_rv64im_accepted_proof_with_perf(input)?;
    Ok((artifact, audit))
}

pub fn prove_rv64im_public_proof_with_options(
    input: &Rv64imProofInput,
    options: Rv64imPublicProofOptions,
) -> Result<Rv64imProof, SimpleKernelError> {
    let (proof, _) = prove_rv64im_public_proof_with_options_and_perf(input, options)?;
    Ok(proof)
}

pub fn prove_rv64im_accepted_proof_with_options(
    input: &Rv64imProofInput,
    options: Rv64imPublicProofOptions,
) -> Result<(Rv64imAcceptedProofArtifact, Rv64imAuditBundle), SimpleKernelError> {
    let ((artifact, audit), _) = prove_rv64im_accepted_proof_with_options_and_perf(input, options)?;
    Ok((artifact, audit))
}

pub fn prove_rv64im_public_proof_with_perf(
    input: &Rv64imProofInput,
) -> Result<(Rv64imProof, Rv64imProofProvePerf), SimpleKernelError> {
    prove_rv64im_public_proof_with_options_and_perf(input, Rv64imPublicProofOptions::default())
}

pub fn prove_rv64im_accepted_proof_with_perf(
    input: &Rv64imProofInput,
) -> Result<((Rv64imAcceptedProofArtifact, Rv64imAuditBundle), Rv64imProofProvePerf), SimpleKernelError> {
    prove_rv64im_accepted_proof_with_options_and_perf(input, Rv64imPublicProofOptions::default())
}

pub fn prove_rv64im_public_proof_with_options_and_perf(
    input: &Rv64imProofInput,
    options: Rv64imPublicProofOptions,
) -> Result<(Rv64imProof, Rv64imProofProvePerf), SimpleKernelError> {
    let ((_, _), proof, perf) = prove_rv64im_public_proof_and_sidecar_with_perf(input, options)?;
    Ok((proof, perf))
}

pub fn prove_rv64im_accepted_proof_with_options_and_perf(
    input: &Rv64imProofInput,
    options: Rv64imPublicProofOptions,
) -> Result<((Rv64imAcceptedProofArtifact, Rv64imAuditBundle), Rv64imProofProvePerf), SimpleKernelError> {
    let (built, perf) = prove_rv64im_public_proof_prover_seam_with_perf(input, options)?;
    let artifact = accepted_proof_artifact_from_prover_materials(
        &built.proof.claim,
        &built.proof.statement,
        &built.kernel,
        &built.sidecar,
        &built.proof.kernel.main_lane,
        &built.proof.kernel.stage_claims,
        &built.proof.kernel.stage_packages,
        &built.proof.kernel.kernel_opening,
        &built.proof.kernel.kernel_claims,
        &built.proof.kernel.root_lane_columns,
        &built.proof.kernel.root_lane_commitment,
    )?;
    let audit = build_rv64im_audit_bundle(&built.proof);
    Ok(((artifact, audit), perf))
}

pub fn prove_rv64im_audit_proof(
    input: &Rv64imProofInput,
) -> Result<(Rv64imProofWitnessBundle, Rv64imProof), SimpleKernelError> {
    let (witness, proof, _) = prove_rv64im_audit_proof_with_perf(input)?;
    Ok((witness, proof))
}

pub fn prove_rv64im_audit_proof_with_perf(
    input: &Rv64imProofInput,
) -> Result<(Rv64imProofWitnessBundle, Rv64imProof, Rv64imProofProvePerf), SimpleKernelError> {
    let ((_, _), proof, perf) =
        prove_rv64im_public_proof_and_sidecar_with_perf(input, Rv64imPublicProofOptions::default())?;
    Ok((proof.witness.clone(), proof, perf))
}

pub fn verify_rv64im_public_proof(proof: &Rv64imProof) -> Result<(), SimpleKernelError> {
    verify_rv64im_public_proof_with_perf(proof).map(|_| ())
}

pub fn verify_rv64im_accepted_proof(artifact: &Rv64imAcceptedProofArtifact) -> Result<(), SimpleKernelError> {
    verify_rv64im_accepted_proof_with_perf(artifact).map(|_| ())
}

pub fn verify_rv64im_public_proof_with_perf(
    proof: &Rv64imProof,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    let artifact = build_rv64im_accepted_proof_artifact(proof)?;
    verify_rv64im_accepted_proof_with_perf(&artifact)
}

pub fn verify_rv64im_accepted_proof_with_perf(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    verify_accepted_proof_artifact_with_perf(artifact)
}

pub fn validate_rv64im_public_proof_against_input(
    input: &Rv64imProofInput,
    proof: &Rv64imProof,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_public_proof_against_input_with_perf(input, proof).map(|_| ())
}

fn legacy_proof_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
    audit: &Rv64imAuditBundle,
) -> Rv64imProof {
    let kernel = Rv64imKernelProofBundle {
        root_params_id: artifact.statement.root_params_id,
        trace: audit.witness.trace.projection(),
        stages: audit.witness.stages.projection_bundle(),
        stage_claims: artifact.stage_claims.clone(),
        stage_packages: artifact.stage_packages.clone(),
        kernel_opening: artifact.kernel_opening.clone(),
        kernel_claims: artifact.kernel_claims.clone(),
        root_lane_columns: artifact.root_lane_columns.clone(),
        root_lane_commitment: artifact.root_lane_commitment.clone(),
        main_lane: artifact.main_lane.clone(),
        digest: [0; 32],
    };
    let kernel = Rv64imKernelProofBundle {
        digest: kernel.expected_digest(),
        ..kernel
    };
    Rv64imProof {
        claim: artifact.claim.clone(),
        statement: artifact.statement.clone(),
        kernel,
        witness: audit.witness.clone(),
    }
}

pub fn audit_rv64im_accepted_proof_against_input(
    input: &Rv64imProofInput,
    artifact: &Rv64imAcceptedProofArtifact,
    audit: &Rv64imAuditBundle,
) -> Result<(), SimpleKernelError> {
    audit_rv64im_accepted_proof_against_input_with_perf(input, artifact, audit).map(|_| ())
}

pub fn validate_rv64im_public_proof_against_input_with_perf(
    input: &Rv64imProofInput,
    proof: &Rv64imProof,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    validate_public_proof_against_input_with_perf(input, proof)
}

pub fn audit_rv64im_accepted_proof_against_input_with_perf(
    input: &Rv64imProofInput,
    artifact: &Rv64imAcceptedProofArtifact,
    audit: &Rv64imAuditBundle,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    verify_rv64im_accepted_proof(artifact)?;
    let proof = legacy_proof_from_accepted_artifact(artifact, audit);
    validate_public_proof_against_input_with_perf(input, &proof)
}

pub fn verify_rv64im_audit_proof(proof: &Rv64imProof) -> Result<Rv64imProofWitnessBundle, SimpleKernelError> {
    let (witness, _) = verify_rv64im_audit_proof_with_perf(proof)?;
    Ok(witness)
}

pub fn verify_rv64im_audit_proof_with_perf(
    proof: &Rv64imProof,
) -> Result<(Rv64imProofWitnessBundle, Rv64imPublicProofVerifyPerf), SimpleKernelError> {
    let (_, perf) = verify_kernel_output_from_public_proof_with_perf(proof)?;
    Ok((proof.witness.clone(), perf))
}
