//! Owns the RV32IM packaged main-lane artifact bound to the committed root lane exports.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::proof::FoldSchedule;

use super::{RootLaneColumns, RootLaneCommitmentArtifact, RootLaneCommitmentSummaryArtifact, SimpleKernelError};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelMainLaneBinding {
    pub root_lane_columns_digest: [u8; 32],
    pub root_lane_commitment_digest: [u8; 32],
    pub fold_schedule: FoldSchedule,
    pub chunk_count: u64,
    pub public_step_count: u64,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleKernelMainLaneArtifact {
    pub binding: SimpleKernelMainLaneBinding,
    pub digest: [u8; 32],
}

impl SimpleKernelMainLaneBinding {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/simple_kernel_main_lane_binding");
        tr.append_message(
            b"rv32im/simple_kernel_main_lane_binding/root_lane_columns_digest",
            &self.root_lane_columns_digest,
        );
        tr.append_message(
            b"rv32im/simple_kernel_main_lane_binding/root_lane_commitment_digest",
            &self.root_lane_commitment_digest,
        );
        tr.append_u64s(
            b"rv32im/simple_kernel_main_lane_binding/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_u64s(
            b"rv32im/simple_kernel_main_lane_binding/meta",
            &[self.chunk_count, self.public_step_count],
        );
        tr.digest32()
    }
}

impl SimpleKernelMainLaneArtifact {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/simple_kernel_main_lane_artifact");
        tr.append_message(
            b"rv32im/simple_kernel_main_lane_artifact/binding_digest",
            &self.binding.digest,
        );
        tr.digest32()
    }
}

fn build_simple_kernel_main_lane_artifact_from_binding(
    root_lane_columns_digest: [u8; 32],
    root_lane_commitment_digest: [u8; 32],
    fold_schedule: FoldSchedule,
    chunk_count: u64,
    public_step_count: u64,
) -> SimpleKernelMainLaneArtifact {
    let binding = SimpleKernelMainLaneBinding {
        root_lane_columns_digest,
        root_lane_commitment_digest,
        fold_schedule,
        chunk_count,
        public_step_count,
        digest: [0; 32],
    };
    let binding = SimpleKernelMainLaneBinding {
        digest: binding.expected_digest(),
        ..binding
    };
    let artifact = SimpleKernelMainLaneArtifact {
        binding,
        digest: [0; 32],
    };
    SimpleKernelMainLaneArtifact {
        digest: artifact.expected_digest(),
        ..artifact
    }
}

pub fn build_simple_kernel_main_lane_artifact(
    root_lane_columns: &RootLaneColumns,
    root_lane_commitment: &RootLaneCommitmentArtifact,
    fold_schedule: FoldSchedule,
) -> Result<SimpleKernelMainLaneArtifact, SimpleKernelError> {
    Ok(build_simple_kernel_main_lane_artifact_from_binding(
        root_lane_columns.digest,
        root_lane_commitment.digest,
        fold_schedule,
        fold_schedule
            .chunk_count(root_lane_columns.time_len as usize)
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))? as u64,
        root_lane_columns.time_len,
    ))
}

pub fn build_simple_kernel_main_lane_artifact_from_summary(
    root_lane_columns: &RootLaneColumns,
    root_lane_commitment: &RootLaneCommitmentSummaryArtifact,
    fold_schedule: FoldSchedule,
) -> Result<SimpleKernelMainLaneArtifact, SimpleKernelError> {
    Ok(build_simple_kernel_main_lane_artifact_from_binding(
        root_lane_columns.digest,
        root_lane_commitment.digest,
        fold_schedule,
        fold_schedule
            .chunk_count(root_lane_columns.time_len as usize)
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))? as u64,
        root_lane_columns.time_len,
    ))
}

pub fn validate_simple_kernel_main_lane_artifact(
    root_lane_columns: &RootLaneColumns,
    root_lane_commitment: &RootLaneCommitmentArtifact,
    artifact: &SimpleKernelMainLaneArtifact,
) -> Result<(), SimpleKernelError> {
    if artifact.binding.digest != artifact.binding.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM packaged main-lane binding digest mismatch".into(),
        ));
    }
    if artifact.binding.root_lane_columns_digest != root_lane_columns.digest
        || artifact.binding.root_lane_commitment_digest != root_lane_commitment.digest
        || artifact.binding.public_step_count != root_lane_columns.time_len
        || artifact.binding.chunk_count
            != artifact
                .binding
                .fold_schedule
                .chunk_count(root_lane_columns.time_len as usize)
                .map_err(|err| SimpleKernelError::Bridge(err.to_string()))? as u64
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM packaged main-lane binding does not match the root lane exports".into(),
        ));
    }
    if artifact.digest != artifact.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM packaged main-lane artifact digest mismatch".into(),
        ));
    }
    Ok(())
}
