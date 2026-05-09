//! Owns side-bundle bridge data shapes and their digest contracts.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::kernel::{
    RootLaneCommitmentSummaryArtifact, Stage1VerifiedClaims, Stage2VerifiedClaims, Stage3VerifiedClaims,
    VerifiedTranscriptSurface,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imSideProofBundle {
    pub statement_core_digest: [u8; 32],
    pub transcript: VerifiedTranscriptSurface,
    pub stage1: Stage1VerifiedClaims,
    pub stage2: Stage2VerifiedClaims,
    pub stage3: Stage3VerifiedClaims,
    pub stage_claim_proof_bridge: Rv32imStageClaimProofBridge,
    pub kernel_opening_bridge: Rv32imKernelOpeningBridge,
    pub kernel_claim_bridge: Rv32imKernelClaimBridge,
    pub kernel_claim_proof_bridge: Rv32imKernelClaimProofBridge,
    pub main_lane_bridge: Rv32imMainLaneProofBridge,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imKernelOpeningBridge {
    pub prepared_step_bindings: Rv32imPreparedStepBindingSummaryBridge,
    pub root_lane_commitment: RootLaneCommitmentSummaryArtifact,
    pub bindings_opening_statement_digest: [u8; 32],
    pub bindings_opening_digest: [u8; 32],
    pub prepared_steps_opening_statement_digest: [u8; 32],
    pub prepared_steps_opening_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imPreparedStepBindingSummaryBridge {
    pub binding_count: u64,
    pub first_binding_digest: Option<[u8; 32]>,
    pub last_binding_digest: Option<[u8; 32]>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imStageClaimProofBridge {
    pub packaged_statement_digest: [u8; 32],
    pub packaged_proof_digest: [u8; 32],
    pub stage_claim_proof_bundle_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imKernelClaimBridge {
    pub stage1_digest: [u8; 32],
    pub stage2_digest: [u8; 32],
    pub stage3_digest: [u8; 32],
    pub root0_digest: [u8; 32],
    pub kernel_claim_bundle_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imKernelClaimProofBridge {
    pub packaged_statement_digest: [u8; 32],
    pub packaged_proof_digest: [u8; 32],
    pub kernel_claim_proof_bundle_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imMainLaneProofBridge {
    pub main_lane_statement_digest: [u8; 32],
    pub main_lane_proof_digest: [u8; 32],
    pub digest: [u8; 32],
}

impl Rv32imSideProofBundle {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/side_proof_bundle");
        tr.append_message(b"neo.fold.next/nightstream/rv32im/side_proof_bundle/version", b"v1");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/statement_core_digest",
            &self.statement_core_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/transcript",
            &self.transcript.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/stage1",
            &self.stage1.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/stage2",
            &self.stage2.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/stage3",
            &self.stage3.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/stage_claim_proof_bridge",
            &self.stage_claim_proof_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/kernel_opening_bridge",
            &self.kernel_opening_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/kernel_claim_bridge",
            &self.kernel_claim_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/kernel_claim_proof_bridge",
            &self.kernel_claim_proof_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/main_lane_bridge",
            &self.main_lane_bridge.digest,
        );
        tr.digest32()
    }
}

impl Rv32imKernelOpeningBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/prepared_step_bindings",
            &self.prepared_step_bindings.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/root_lane_commitment",
            &self.root_lane_commitment.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/bindings_opening_statement_digest",
            &self.bindings_opening_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/bindings_opening_digest",
            &self.bindings_opening_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/prepared_steps_opening_statement_digest",
            &self.prepared_steps_opening_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/prepared_steps_opening_digest",
            &self.prepared_steps_opening_digest,
        );
        tr.digest32()
    }
}

impl Rv32imPreparedStepBindingSummaryBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge/binding_count",
            &[self.binding_count],
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge/first_present",
            &[self.first_binding_digest.is_some() as u64],
        );
        if let Some(digest) = self.first_binding_digest {
            tr.append_message(
                b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge/first_binding_digest",
                &digest,
            );
        }
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge/last_present",
            &[self.last_binding_digest.is_some() as u64],
        );
        if let Some(digest) = self.last_binding_digest {
            tr.append_message(
                b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge/last_binding_digest",
                &digest,
            );
        }
        tr.digest32()
    }
}

impl Rv32imStageClaimProofBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/stage_claim_proof_bridge");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/stage_claim_proof_bridge/packaged_statement_digest",
            &self.packaged_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/stage_claim_proof_bridge/packaged_proof_digest",
            &self.packaged_proof_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/stage_claim_proof_bridge/stage_claim_proof_bundle_digest",
            &self.stage_claim_proof_bundle_digest,
        );
        tr.digest32()
    }
}

impl Rv32imKernelClaimBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge/stage1_digest",
            &self.stage1_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge/stage2_digest",
            &self.stage2_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge/stage3_digest",
            &self.stage3_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge/root0_digest",
            &self.root0_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge/kernel_claim_bundle_digest",
            &self.kernel_claim_bundle_digest,
        );
        tr.digest32()
    }
}

impl Rv32imKernelClaimProofBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/kernel_claim_proof_bridge");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_proof_bridge/packaged_statement_digest",
            &self.packaged_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_proof_bridge/packaged_proof_digest",
            &self.packaged_proof_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_proof_bridge/kernel_claim_proof_bundle_digest",
            &self.kernel_claim_proof_bundle_digest,
        );
        tr.digest32()
    }
}

impl Rv32imMainLaneProofBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/main_lane_proof_bridge");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/main_lane_proof_bridge/main_lane_statement_digest",
            &self.main_lane_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/main_lane_proof_bridge/main_lane_proof_digest",
            &self.main_lane_proof_digest,
        );
        tr.digest32()
    }
}
