//! Owns accepted-artifact surfaces required for theorem-facing completeness checks.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::kernel::RootExecutionBundle;
use crate::rv32im::stage1::Stage1ProofBundle;
use crate::rv32im::stage2::Stage2ProofBundle;
use crate::rv32im::stage3::Stage3ProofBundle;

const STAGE1_SHOUT_CHANNELS: [&str; 3] = ["bytecode", "alu", "branch"];
const STAGE1_ADDRESS_FAMILIES: [&str; 3] = ["bytecode", "alu", "branch"];
const STAGE2_ADDRESS_FAMILIES: [&str; 2] = ["reg", "ram"];
const TWIST_MEMORY_FAMILIES: [&str; 2] = ["reg", "ram"];
const SCALAR_SOUNDNESS_TERMS: [&str; 9] = [
    "ram_raf",
    "stage1_linkage",
    "stage2_linkage",
    "continuity",
    "opening_provenance",
    "program_binding",
    "pcs",
    "fs",
    "outer",
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StepCompositionSurface {
    pub stage1_semantics_digest: [u8; 32],
    pub stage2_semantics_digest: [u8; 32],
    pub stage2_temporal_digest: [u8; 32],
    pub stage3_semantics_digest: [u8; 32],
    pub root_execution_digest: [u8; 32],
    pub prepared_step_bindings_digest: [u8; 32],
    pub row_chunk_routes_digest: [u8; 32],
    pub real_row_count: u64,
    pub prepared_step_count: u64,
    pub first_real_step_index: u64,
    pub last_real_step_index: u64,
    pub initial_pc: u64,
    pub final_pc: u64,
    pub halted: bool,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KernelSoundnessAccountingSurface {
    pub schema_version: u64,
    pub stage1_shout_channels: Vec<String>,
    pub stage1_address_families: Vec<String>,
    pub stage2_address_families: Vec<String>,
    pub twist_memory_families: Vec<String>,
    pub scalar_terms: Vec<String>,
    pub schema_digest: [u8; 32],
    pub digest: [u8; 32],
}

fn append_labels(tr: &mut Poseidon2Transcript, label: &'static [u8], values: &[String]) {
    tr.append_u64s(label, &[values.len() as u64]);
    for value in values {
        tr.append_message(label, value.as_bytes());
    }
}

impl StepCompositionSurface {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/step_composition_surface");
        tr.append_message(
            b"rv32im/step_composition_surface/stage1_semantics_digest",
            &self.stage1_semantics_digest,
        );
        tr.append_message(
            b"rv32im/step_composition_surface/stage2_semantics_digest",
            &self.stage2_semantics_digest,
        );
        tr.append_message(
            b"rv32im/step_composition_surface/stage2_temporal_digest",
            &self.stage2_temporal_digest,
        );
        tr.append_message(
            b"rv32im/step_composition_surface/stage3_semantics_digest",
            &self.stage3_semantics_digest,
        );
        tr.append_message(
            b"rv32im/step_composition_surface/root_execution_digest",
            &self.root_execution_digest,
        );
        tr.append_message(
            b"rv32im/step_composition_surface/prepared_step_bindings_digest",
            &self.prepared_step_bindings_digest,
        );
        tr.append_message(
            b"rv32im/step_composition_surface/row_chunk_routes_digest",
            &self.row_chunk_routes_digest,
        );
        tr.append_u64s(
            b"rv32im/step_composition_surface/meta",
            &[
                self.real_row_count,
                self.prepared_step_count,
                self.first_real_step_index,
                self.last_real_step_index,
                self.initial_pc,
                self.final_pc,
                self.halted as u64,
            ],
        );
        tr.digest32()
    }
}

impl KernelSoundnessAccountingSurface {
    pub(crate) fn expected_schema_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/kernel_soundness_accounting_schema");
        tr.append_u64s(
            b"rv32im/kernel_soundness_accounting_schema/version",
            &[self.schema_version],
        );
        append_labels(
            &mut tr,
            b"rv32im/kernel_soundness_accounting_schema/stage1_shout_channel",
            &self.stage1_shout_channels,
        );
        append_labels(
            &mut tr,
            b"rv32im/kernel_soundness_accounting_schema/stage1_address_family",
            &self.stage1_address_families,
        );
        append_labels(
            &mut tr,
            b"rv32im/kernel_soundness_accounting_schema/stage2_address_family",
            &self.stage2_address_families,
        );
        append_labels(
            &mut tr,
            b"rv32im/kernel_soundness_accounting_schema/twist_memory_family",
            &self.twist_memory_families,
        );
        append_labels(
            &mut tr,
            b"rv32im/kernel_soundness_accounting_schema/scalar_term",
            &self.scalar_terms,
        );
        tr.digest32()
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/kernel_soundness_accounting_surface");
        tr.append_u64s(
            b"rv32im/kernel_soundness_accounting_surface/meta",
            &[
                self.schema_version,
                self.stage1_shout_channels.len() as u64,
                self.stage1_address_families.len() as u64,
                self.stage2_address_families.len() as u64,
                self.twist_memory_families.len() as u64,
                self.scalar_terms.len() as u64,
            ],
        );
        tr.append_message(
            b"rv32im/kernel_soundness_accounting_surface/schema_digest",
            &self.schema_digest,
        );
        tr.digest32()
    }
}

pub(crate) fn build_step_composition_surface(
    stage1: &Stage1ProofBundle,
    stage2: &Stage2ProofBundle,
    stage3: &Stage3ProofBundle,
    root_execution: &RootExecutionBundle,
    initial_pc: u64,
    final_pc: u64,
) -> StepCompositionSurface {
    let mut real_rows = root_execution
        .execution_rows
        .iter()
        .filter(|row| row.is_real);
    let first_real_step_index = real_rows
        .next()
        .map(|row| row.step_index as u64)
        .unwrap_or(0);
    let last_real_step_index = root_execution
        .execution_rows
        .iter()
        .rev()
        .find(|row| row.is_real)
        .map(|row| row.step_index as u64)
        .unwrap_or(0);
    let surface = StepCompositionSurface {
        stage1_semantics_digest: stage1.semantics.digest,
        stage2_semantics_digest: stage2.semantics.digest,
        stage2_temporal_digest: stage2.temporal.digest,
        stage3_semantics_digest: stage3.semantics.digest,
        root_execution_digest: root_execution.digest,
        prepared_step_bindings_digest: root_execution.prepared_step_bindings.digest,
        row_chunk_routes_digest: root_execution.row_chunk_routes_digest,
        real_row_count: root_execution
            .execution_rows
            .iter()
            .filter(|row| row.is_real)
            .count() as u64,
        prepared_step_count: root_execution.prepared_step_bindings.binding_count,
        first_real_step_index,
        last_real_step_index,
        initial_pc,
        final_pc,
        halted: stage3.bridge.halted,
        digest: [0; 32],
    };
    StepCompositionSurface {
        digest: surface.expected_digest(),
        ..surface
    }
}

pub(crate) fn canonical_kernel_soundness_accounting_surface() -> KernelSoundnessAccountingSurface {
    let surface = KernelSoundnessAccountingSurface {
        schema_version: 1,
        stage1_shout_channels: STAGE1_SHOUT_CHANNELS
            .iter()
            .map(ToString::to_string)
            .collect(),
        stage1_address_families: STAGE1_ADDRESS_FAMILIES
            .iter()
            .map(ToString::to_string)
            .collect(),
        stage2_address_families: STAGE2_ADDRESS_FAMILIES
            .iter()
            .map(ToString::to_string)
            .collect(),
        twist_memory_families: TWIST_MEMORY_FAMILIES
            .iter()
            .map(ToString::to_string)
            .collect(),
        scalar_terms: SCALAR_SOUNDNESS_TERMS
            .iter()
            .map(ToString::to_string)
            .collect(),
        schema_digest: [0; 32],
        digest: [0; 32],
    };
    let schema_digest = surface.expected_schema_digest();
    let surface = KernelSoundnessAccountingSurface {
        schema_digest,
        ..surface
    };
    KernelSoundnessAccountingSurface {
        digest: surface.expected_digest(),
        ..surface
    }
}
