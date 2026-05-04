#![allow(dead_code)]

use std::sync::OnceLock;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv32im::layout::{RV32IM_PARITY_LOWERING_VERSION_ID, RV32IM_PARITY_PROTOCOL_VERSION_ID};
use neo_fold_next::rv32im::tables::Rv32FamilyTag;
use neo_fold_next::rv32im::{
    build_rv32im_accepted_proof_artifact, encode_addi, encode_ecall, encode_lw, encode_sw,
    prove_rv32im_public_proof_with_options, MemoryWord, Rv32imAcceptedProofArtifact, Rv32imParityCaseManifest,
    Rv32imParitySourceCase, Rv32imProofInput, Rv32imPublicProofOptions,
};

const START_PC: u32 = 0x1000;
const MAX_STEPS: usize = 32;

fn manifest(name: &str, fixture_id: &str, family_tags: Vec<Rv32FamilyTag>) -> Rv32imParityCaseManifest {
    Rv32imParityCaseManifest {
        name: name.into(),
        fixture_id: fixture_id.into(),
        protocol_version_id: RV32IM_PARITY_PROTOCOL_VERSION_ID,
        lowering_version_id: RV32IM_PARITY_LOWERING_VERSION_ID,
        family_tags,
    }
}

pub fn phase0_memory_input() -> Rv32imProofInput {
    let mut initial_registers = [0u32; 32];
    initial_registers[10] = 0x80;
    let program_words = vec![
        encode_addi(1, 0, 42),
        encode_sw(1, 10, 0),
        encode_lw(2, 10, 0),
        encode_ecall(),
    ];
    Rv32imProofInput {
        source: Rv32imParitySourceCase {
            manifest: manifest(
                "rv32im_phase0_memory_surface",
                "rv32im_phase0_memory_surface_v1",
                vec![Rv32FamilyTag::AlignedMemory, Rv32FamilyTag::NativeAlu],
            ),
            start_pc: START_PC,
            program_words,
            initial_registers,
            initial_memory: Vec::<MemoryWord>::new(),
            transcript_seed: b"rv32im-phase0-memory-surface-v1".to_vec(),
        },
        max_steps: MAX_STEPS,
    }
}

pub fn phase0_memory_artifact() -> &'static Rv32imAcceptedProofArtifact {
    static ARTIFACT: OnceLock<Rv32imAcceptedProofArtifact> = OnceLock::new();
    ARTIFACT.get_or_init(|| {
        let proof = prove_rv32im_public_proof_with_options(
            &phase0_memory_input(),
            Rv32imPublicProofOptions {
                root_fold_schedule: FoldSchedule::RowsPerChunk(1),
            },
        )
        .expect("prove phase0 memory public proof");
        build_rv32im_accepted_proof_artifact(&proof).expect("build phase0 memory accepted artifact")
    })
}
