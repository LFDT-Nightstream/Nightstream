#![allow(dead_code)]

use std::sync::OnceLock;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::layout::{RV64IM_PARITY_LOWERING_VERSION_ID, RV64IM_PARITY_PROTOCOL_VERSION_ID};
use neo_fold_next::rv64im::tables::Rv64FamilyTag;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, encode_addi, encode_ecall, encode_ld, encode_sd,
    prove_rv64im_public_proof_with_options, MemoryWord, Rv64imAcceptedProofArtifact, Rv64imParityCaseManifest,
    Rv64imParitySourceCase, Rv64imProofInput, Rv64imPublicProofOptions,
};

const START_PC: u64 = 0x1000;
const MAX_STEPS: usize = 32;

fn manifest(name: &str, fixture_id: &str, family_tags: Vec<Rv64FamilyTag>) -> Rv64imParityCaseManifest {
    Rv64imParityCaseManifest {
        name: name.into(),
        fixture_id: fixture_id.into(),
        protocol_version_id: RV64IM_PARITY_PROTOCOL_VERSION_ID,
        lowering_version_id: RV64IM_PARITY_LOWERING_VERSION_ID,
        family_tags,
    }
}

pub fn phase0_memory_input() -> Rv64imProofInput {
    let mut initial_registers = [0u64; 32];
    initial_registers[10] = 0x80;
    let program_words = vec![
        encode_addi(1, 0, 42),
        encode_sd(1, 10, 0),
        encode_ld(2, 10, 0),
        encode_ecall(),
    ];
    Rv64imProofInput {
        source: Rv64imParitySourceCase {
            manifest: manifest(
                "rv64im_phase0_memory_surface",
                "rv64im_phase0_memory_surface_v1",
                vec![Rv64FamilyTag::AlignedMemory, Rv64FamilyTag::NativeAlu],
            ),
            start_pc: START_PC,
            program_words,
            initial_registers,
            initial_memory: Vec::<MemoryWord>::new(),
            transcript_seed: b"rv64im-phase0-memory-surface-v1".to_vec(),
        },
        max_steps: MAX_STEPS,
    }
}

pub fn phase0_memory_artifact() -> &'static Rv64imAcceptedProofArtifact {
    static ARTIFACT: OnceLock<Rv64imAcceptedProofArtifact> = OnceLock::new();
    ARTIFACT.get_or_init(|| {
        let proof = prove_rv64im_public_proof_with_options(
            &phase0_memory_input(),
            Rv64imPublicProofOptions {
                root_fold_schedule: FoldSchedule::RowsPerChunk(1),
            },
        )
        .expect("prove phase0 memory public proof");
        build_rv64im_accepted_proof_artifact(&proof).expect("build phase0 memory accepted artifact")
    })
}
