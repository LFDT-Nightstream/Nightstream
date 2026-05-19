//! Owns Simple Kernel public data shapes and error reporting.

use std::ops::Deref;

use neo_reductions::error::PiCcsError;
use serde::{Deserialize, Serialize};

use crate::rv32im::isa::Rv32BuildError;
use crate::rv32im::lower::Rv32ExpandedRow;
use crate::rv32im::stage1::Stage1Summary;
use crate::rv32im::stage2::Stage2Summary;
use crate::rv32im::stage3::Stage3Summary;

use super::{
    RootLaneColumns, RootLaneCommitmentArtifact, Rv32imKernelSummary, Rv32imParityCaseManifest, Rv32imParitySourceCase,
    TranscriptRecord,
};
use super::{
    RootLaneCommitmentSummaryArtifact, RootLaneWitness, Rv32imStageWitnessProjectionBundle, Rv32imTraceProjectionBundle,
};
use super::{
    SimpleKernelMainLaneArtifact, SimpleKernelOpeningBundle, SimpleKernelStageClaimBundle,
    SimpleKernelStagePackageBundle,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelPublicInput {
    pub source: Rv32imParitySourceCase,
    pub max_steps: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelProverInput {
    pub public: SimpleKernelPublicInput,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelVerifierInput {
    pub public: SimpleKernelPublicInput,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparedStepBinding {
    pub trace_index: usize,
    pub row_digest: [u8; 32],
    pub row_opening_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparedStepBindingSummary {
    pub bindings: Vec<PreparedStepBinding>,
    pub binding_count: u64,
    pub first_binding_digest: Option<[u8; 32]>,
    pub last_binding_digest: Option<[u8; 32]>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelTraceWitness {
    pub manifest: Rv32imParityCaseManifest,
    pub execution_rows: Vec<Rv32ExpandedRow>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelStageWitnessBundle {
    pub stage1: Stage1Summary,
    pub stage2: Stage2Summary,
    pub stage3: Stage3Summary,
    pub transcript: TranscriptRecord,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelKernelClaimBundle {
    pub kernel: Rv32imKernelSummary,
    pub prepared_step_bindings: PreparedStepBindingSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleKernelOutput {
    pub trace: SimpleKernelTraceWitness,
    pub stages: SimpleKernelStageWitnessBundle,
    pub stage_claims: SimpleKernelStageClaimBundle,
    pub stage_packages: SimpleKernelStagePackageBundle,
    pub kernel_opening: SimpleKernelOpeningBundle,
    pub kernel_claims: SimpleKernelKernelClaimBundle,
    pub root_lane_columns: RootLaneColumns,
    pub root_lane_commitment: RootLaneCommitmentArtifact,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleKernelAuditOutput {
    pub kernel: SimpleKernelOutput,
    pub prepared_steps: Vec<crate::proof::StepInput>,
}

impl Deref for SimpleKernelAuditOutput {
    type Target = SimpleKernelOutput;

    fn deref(&self) -> &Self::Target {
        &self.kernel
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleKernelProof {
    pub root_params_id: [u8; 32],
    pub trace: SimpleKernelTraceWitness,
    pub stages: SimpleKernelStageWitnessBundle,
    pub stage_claims: SimpleKernelStageClaimBundle,
    pub stage_packages: SimpleKernelStagePackageBundle,
    pub kernel_opening: SimpleKernelOpeningBundle,
    pub kernel_claims: SimpleKernelKernelClaimBundle,
    pub root_lane_columns: RootLaneColumns,
    pub root_lane_commitment: RootLaneCommitmentArtifact,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleKernelPackagedProof {
    pub kernel: SimpleKernelProof,
    pub main_lane: SimpleKernelMainLaneArtifact,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct PublicSimpleKernelOutput {
    pub trace: Rv32imTraceProjectionBundle,
    pub stages: Rv32imStageWitnessProjectionBundle,
    pub stage_claims: SimpleKernelStageClaimBundle,
    pub stage_packages: SimpleKernelStagePackageBundle,
    pub kernel_opening: SimpleKernelOpeningBundle,
    pub kernel_claims: SimpleKernelKernelClaimBundle,
    pub root_lane_columns: RootLaneColumns,
    pub root_lane_commitment: RootLaneCommitmentSummaryArtifact,
}

pub(crate) struct SimpleKernelExpectedSeed {
    pub(super) trace: SimpleKernelTraceWitness,
    pub(super) stages: SimpleKernelStageWitnessBundle,
    pub(super) stage_claims: SimpleKernelStageClaimBundle,
    pub(super) kernel_claims: SimpleKernelKernelClaimBundle,
    pub(super) root_lane_columns: RootLaneColumns,
    pub(super) root_lane_commitment: RootLaneCommitmentArtifact,
    pub(super) root_lane_witness: RootLaneWitness,
}

pub(crate) struct SimpleKernelBuildSeed {
    pub(super) trace: SimpleKernelTraceWitness,
    pub(super) stages: SimpleKernelStageWitnessBundle,
    pub(super) stage_claims: SimpleKernelStageClaimBundle,
    pub(super) stage_packages: SimpleKernelStagePackageBundle,
    pub(super) kernel_opening: SimpleKernelOpeningBundle,
    pub(super) kernel_claims: SimpleKernelKernelClaimBundle,
    pub(super) root_lane_columns: RootLaneColumns,
    pub(super) root_lane_commitment: RootLaneCommitmentArtifact,
}

pub(crate) struct PublicSimpleKernelBuildSeed {
    pub(super) trace: Rv32imTraceProjectionBundle,
    pub(super) stages: Rv32imStageWitnessProjectionBundle,
    pub(super) stage_claims: SimpleKernelStageClaimBundle,
    pub(super) stage_packages: SimpleKernelStagePackageBundle,
    pub(super) kernel_opening: SimpleKernelOpeningBundle,
    pub(super) kernel_claims: SimpleKernelKernelClaimBundle,
    pub(super) root_lane_columns: RootLaneColumns,
    pub(super) root_lane_commitment: RootLaneCommitmentSummaryArtifact,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct PublicSimpleKernelWitnessSidecar {
    pub trace: SimpleKernelTraceWitness,
    pub stages: SimpleKernelStageWitnessBundle,
}

#[derive(Debug)]
pub enum SimpleKernelError {
    Build(String),
    Bridge(String),
    Proof(String),
}

impl core::fmt::Display for SimpleKernelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Build(s) => write!(f, "build failed: {s}"),
            Self::Bridge(s) => write!(f, "bridge failed: {s}"),
            Self::Proof(s) => write!(f, "proof failed: {s}"),
        }
    }
}

impl std::error::Error for SimpleKernelError {}

impl From<Rv32BuildError> for SimpleKernelError {
    fn from(value: Rv32BuildError) -> Self {
        Self::Build(value.to_string())
    }
}

impl From<PiCcsError> for SimpleKernelError {
    fn from(value: PiCcsError) -> Self {
        Self::Proof(value.to_string())
    }
}
