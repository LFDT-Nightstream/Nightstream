use std::sync::{LazyLock, Mutex, MutexGuard};

use neo_transcript::{Poseidon2Transcript, Transcript};

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::audit_rv64im_accepted_proof;
use neo_fold_next::rv64im::layout::{RV64IM_PARITY_LOWERING_VERSION_ID, RV64IM_PARITY_PROTOCOL_VERSION_ID};
use neo_fold_next::rv64im::tables::Rv64FamilyTag;
use neo_fold_next::rv64im::{
    encode_add, encode_addi, encode_beq, encode_divu, encode_ecall, encode_ld, encode_sd, parity_source_cases,
    prove_rv64im_accepted_proof, prove_rv64im_accepted_proof_with_options, sem_inputs_digest, MemoryWord,
    Rv64imAcceptedProofArtifact, Rv64imAuditBundle, Rv64imParityCaseManifest, Rv64imParitySourceCase, Rv64imProofInput,
    Rv64imPublicProofOptions,
};

const START_PC: u64 = 0x1000;
const MAX_STEPS: usize = 32;
type AcceptedFixture = (Rv64imAcceptedProofArtifact, Rv64imAuditBundle);

fn manifest(name: &str, fixture_id: &str, family_tags: Vec<Rv64FamilyTag>) -> Rv64imParityCaseManifest {
    Rv64imParityCaseManifest {
        name: name.into(),
        fixture_id: fixture_id.into(),
        protocol_version_id: RV64IM_PARITY_PROTOCOL_VERSION_ID,
        lowering_version_id: RV64IM_PARITY_LOWERING_VERSION_ID,
        family_tags,
    }
}

#[allow(dead_code)]
pub fn alu_input() -> Rv64imProofInput {
    let program_words = vec![
        encode_addi(1, 0, 5),
        encode_addi(2, 0, 7),
        encode_add(3, 1, 2),
        encode_ecall(),
    ];
    Rv64imProofInput {
        source: Rv64imParitySourceCase {
            manifest: manifest(
                "rv64im_stage_test_alu",
                "rv64im_stage_test_alu_v1",
                vec![Rv64FamilyTag::NativeAlu],
            ),
            start_pc: START_PC,
            program_words,
            initial_registers: [0; 32],
            initial_memory: Vec::<MemoryWord>::new(),
            transcript_seed: b"rv64im-stage-test-alu-v1".to_vec(),
        },
        max_steps: MAX_STEPS,
    }
}

#[allow(dead_code)]
pub fn branch_input() -> Rv64imProofInput {
    let program_words = vec![
        encode_addi(1, 0, 1),
        encode_addi(2, 0, 1),
        encode_beq(1, 2, 8),
        encode_addi(3, 0, 7),
        encode_addi(3, 0, 9),
        encode_ecall(),
    ];
    Rv64imProofInput {
        source: Rv64imParitySourceCase {
            manifest: manifest(
                "rv64im_stage_test_branch",
                "rv64im_stage_test_branch_v1",
                vec![Rv64FamilyTag::ControlFlow, Rv64FamilyTag::NativeAlu],
            ),
            start_pc: START_PC,
            program_words,
            initial_registers: [0; 32],
            initial_memory: Vec::<MemoryWord>::new(),
            transcript_seed: b"rv64im-stage-test-branch-v1".to_vec(),
        },
        max_steps: MAX_STEPS,
    }
}

#[allow(dead_code)]
pub fn memory_input() -> Rv64imProofInput {
    let mut initial_registers = [0; 32];
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
                "rv64im_stage_test_memory",
                "rv64im_stage_test_memory_v1",
                vec![Rv64FamilyTag::AlignedMemory, Rv64FamilyTag::NativeAlu],
            ),
            start_pc: START_PC,
            program_words,
            initial_registers,
            initial_memory: Vec::<MemoryWord>::new(),
            transcript_seed: b"rv64im-stage-test-memory-v1".to_vec(),
        },
        max_steps: MAX_STEPS,
    }
}

#[allow(dead_code)]
pub fn divu_input() -> Rv64imProofInput {
    let mut initial_registers = [0; 32];
    initial_registers[1] = 100;
    initial_registers[2] = 7;
    let program_words = vec![encode_divu(5, 1, 2), encode_ecall()];
    Rv64imProofInput {
        source: Rv64imParitySourceCase {
            manifest: manifest(
                "rv64im_stage_test_divu",
                "rv64im_stage_test_divu_v1",
                vec![Rv64FamilyTag::UnsignedDivRem],
            ),
            start_pc: START_PC,
            program_words,
            initial_registers,
            initial_memory: Vec::<MemoryWord>::new(),
            transcript_seed: b"rv64im-stage-test-divu-v1".to_vec(),
        },
        max_steps: MAX_STEPS,
    }
}

#[allow(dead_code)]
pub fn parity_input(name: &str) -> Rv64imProofInput {
    let source = parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"));
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

#[allow(dead_code)]
pub fn accepted_test_guard() -> MutexGuard<'static, ()> {
    static ACCEPTED_TEST_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
    ACCEPTED_TEST_MUTEX
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

pub fn prove_accepted(input: &Rv64imProofInput) -> (Rv64imAcceptedProofArtifact, Rv64imAuditBundle) {
    prove_rv64im_accepted_proof(input).expect("prove accepted rv64im proof")
}

fn prove_accepted_rows_per_chunk_1(input: &Rv64imProofInput) -> (Rv64imAcceptedProofArtifact, Rv64imAuditBundle) {
    prove_rv64im_accepted_proof_with_options(
        input,
        Rv64imPublicProofOptions {
            root_fold_schedule: FoldSchedule::RowsPerChunk(1),
        },
    )
    .expect("prove accepted rv64im proof with RowsPerChunk(1)")
}

#[allow(dead_code)]
pub fn accepted_alu() -> AcceptedFixture {
    static FIXTURE: LazyLock<AcceptedFixture> = LazyLock::new(|| {
        let input = alu_input();
        prove_accepted(&input)
    });
    FIXTURE.clone()
}

#[allow(dead_code)]
pub fn accepted_branch() -> AcceptedFixture {
    static FIXTURE: LazyLock<AcceptedFixture> = LazyLock::new(|| {
        let input = branch_input();
        prove_accepted(&input)
    });
    FIXTURE.clone()
}

#[allow(dead_code)]
pub fn accepted_memory() -> AcceptedFixture {
    static FIXTURE: LazyLock<AcceptedFixture> = LazyLock::new(|| {
        let input = memory_input();
        prove_accepted(&input)
    });
    FIXTURE.clone()
}

#[allow(dead_code)]
pub fn accepted_divu() -> AcceptedFixture {
    static FIXTURE: LazyLock<AcceptedFixture> = LazyLock::new(|| {
        let input = divu_input();
        prove_accepted(&input)
    });
    FIXTURE.clone()
}

#[allow(dead_code)]
pub fn accepted_multiply_high() -> AcceptedFixture {
    static FIXTURE: LazyLock<AcceptedFixture> = LazyLock::new(|| {
        let input = parity_input("multiply_high_mulh_mulhu_mulhsu_ecall");
        prove_accepted_rows_per_chunk_1(&input)
    });
    FIXTURE.clone()
}

pub fn expect_accepted_audit_failure(artifact: &Rv64imAcceptedProofArtifact, needle: &str) {
    let err = audit_rv64im_accepted_proof(artifact).expect_err("tampered accepted proof must fail");
    assert!(
        format!("{err}").contains(needle),
        "expected error to contain `{needle}`, got `{err}`"
    );
}

pub fn refresh_accepted_artifact_digest(artifact: &mut Rv64imAcceptedProofArtifact) {
    let mut artifact_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/accepted_proof_artifact");
    artifact_tr.append_message(b"rv64im/accepted_proof_artifact/claim", &artifact.claim.digest);
    artifact_tr.append_message(b"rv64im/accepted_proof_artifact/statement", &artifact.statement.digest);
    artifact_tr.append_message(
        b"rv64im/accepted_proof_artifact/stage_claims",
        &artifact.stage_claims.digest,
    );
    artifact_tr.append_message(
        b"rv64im/accepted_proof_artifact/stage_packages",
        &artifact.stage_packages.digest,
    );
    artifact_tr.append_message(
        b"rv64im/accepted_proof_artifact/kernel_opening",
        &artifact.kernel_opening.digest,
    );
    artifact_tr.append_message(
        b"rv64im/accepted_proof_artifact/kernel_claims",
        &artifact.kernel_claims.digest,
    );
    artifact_tr.append_message(
        b"rv64im/accepted_proof_artifact/root_lane_columns",
        &artifact.root_lane_columns.digest,
    );
    artifact_tr.append_message(
        b"rv64im/accepted_proof_artifact/root_lane_commitment",
        &artifact.root_lane_commitment.digest,
    );
    artifact_tr.append_message(b"rv64im/accepted_proof_artifact/main_lane", &artifact.main_lane.digest);
    artifact_tr.append_message(b"rv64im/accepted_proof_artifact/stage1", &artifact.stage1.digest);
    artifact_tr.append_message(b"rv64im/accepted_proof_artifact/stage2", &artifact.stage2.digest);
    artifact_tr.append_message(b"rv64im/accepted_proof_artifact/stage3", &artifact.stage3.digest);
    artifact_tr.append_message(
        b"rv64im/accepted_proof_artifact/root_execution",
        &artifact.root_execution.digest,
    );
    artifact_tr.append_message(
        b"rv64im/accepted_proof_artifact/step_composition",
        &artifact.step_composition.digest,
    );
    artifact_tr.append_message(
        b"rv64im/accepted_proof_artifact/soundness_accounting",
        &artifact.soundness_accounting.digest,
    );
    artifact_tr.append_message(
        b"rv64im/accepted_proof_artifact/transcript_digest",
        &artifact.transcript.expected_digest(),
    );
    artifact.digest = artifact_tr.digest32();
}

#[allow(dead_code)]
pub fn refresh_stage1_semantic_digests(artifact: &mut Rv64imAcceptedProofArtifact) {
    let sem_inputs_digest = sem_inputs_digest(&artifact.stage1.sem_inputs);
    let row_bindings_digest = artifact.stage1.semantics.row_bindings_digest;

    let mut alu_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage1_alu_shout_proof");
    alu_tr.append_message(b"rv64im/stage1_alu_shout_proof/sem_inputs_digest", &sem_inputs_digest);
    alu_tr.append_u64s(
        b"rv64im/stage1_alu_shout_proof/meta",
        &[
            artifact.stage1.alu.effect_trace_index,
            artifact.stage1.alu.commit_trace_index,
        ],
    );
    artifact.stage1.alu.sem_inputs_digest = sem_inputs_digest;
    artifact.stage1.alu.digest = alu_tr.digest32();

    let mut branch_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage1_branch_shout_proof");
    branch_tr.append_message(
        b"rv64im/stage1_branch_shout_proof/sem_inputs_digest",
        &sem_inputs_digest,
    );
    branch_tr.append_u64s(
        b"rv64im/stage1_branch_shout_proof/meta",
        &[
            artifact.stage1.branch.first_trace_index,
            artifact.stage1.branch.last_trace_index,
        ],
    );
    artifact.stage1.branch.sem_inputs_digest = sem_inputs_digest;
    artifact.stage1.branch.digest = branch_tr.digest32();

    let mut linkage_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage1_linkage_proof");
    linkage_tr.append_message(
        b"rv64im/stage1_linkage_proof/rows_digest",
        &artifact.stage1.linkage.rows_digest,
    );
    linkage_tr.append_message(b"rv64im/stage1_linkage_proof/sem_inputs_digest", &sem_inputs_digest);
    linkage_tr.append_u64s(
        b"rv64im/stage1_linkage_proof/meta",
        &[
            artifact.stage1.linkage.mix,
            artifact.stage1.linkage.first_trace_index,
            artifact.stage1.linkage.effect_trace_index,
            artifact.stage1.linkage.commit_trace_index,
            artifact.stage1.linkage.last_trace_index,
        ],
    );
    artifact.stage1.linkage.sem_inputs_digest = sem_inputs_digest;
    artifact.stage1.linkage.digest = linkage_tr.digest32();

    let mut semantics_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage1_semantics_proof");
    semantics_tr.append_message(b"rv64im/stage1_semantics_proof/sem_inputs_digest", &sem_inputs_digest);
    semantics_tr.append_message(
        b"rv64im/stage1_semantics_proof/row_bindings_digest",
        &row_bindings_digest,
    );
    semantics_tr.append_u64s(
        b"rv64im/stage1_semantics_proof/meta",
        &[
            artifact.stage1.semantics.sequence_count,
            artifact.stage1.semantics.helper_row_count,
        ],
    );
    artifact.stage1.semantics.sem_inputs_digest = sem_inputs_digest;
    artifact.stage1.semantics.digest = semantics_tr.digest32();

    let mut stage1_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage1_proof_bundle");
    stage1_tr.append_message(b"rv64im/stage1_proof_bundle/bytecode", &artifact.stage1.bytecode.digest);
    stage1_tr.append_message(b"rv64im/stage1_proof_bundle/alu", &artifact.stage1.alu.digest);
    stage1_tr.append_message(b"rv64im/stage1_proof_bundle/branch", &artifact.stage1.branch.digest);
    stage1_tr.append_message(
        b"rv64im/stage1_proof_bundle/semantics",
        &artifact.stage1.semantics.digest,
    );
    stage1_tr.append_message(
        b"rv64im/stage1_proof_bundle/address_correctness",
        &artifact.stage1.address_correctness.digest,
    );
    stage1_tr.append_message(b"rv64im/stage1_proof_bundle/linkage", &artifact.stage1.linkage.digest);
    stage1_tr.append_message(
        b"rv64im/stage1_proof_bundle/selected_opening",
        &artifact.stage1.selected_opening.digest,
    );
    stage1_tr.append_u64s(
        b"rv64im/stage1_proof_bundle/meta",
        &[
            artifact.stage1.sem_inputs.len() as u64,
            artifact.stage1.row_bindings.len() as u64,
        ],
    );
    artifact.stage1.digest = stage1_tr.digest32();

    refresh_accepted_artifact_digest(artifact);
}

#[allow(dead_code)]
pub fn refresh_stage3_semantic_digests(artifact: &mut Rv64imAcceptedProofArtifact) {
    let mut semantics_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage3_semantics_proof");
    semantics_tr.append_message(
        b"rv64im/stage3_semantics_proof/continuity_digest",
        &artifact.stage3.semantics.continuity_digest,
    );
    semantics_tr.append_message(
        b"rv64im/stage3_semantics_proof/root_semantic_rows_digest",
        &artifact.stage3.semantics.root_semantic_rows_digest,
    );
    semantics_tr.append_message(
        b"rv64im/stage3_semantics_proof/row_chunk_routes_digest",
        &artifact.stage3.semantics.row_chunk_routes_digest,
    );
    semantics_tr.append_message(
        b"rv64im/stage3_semantics_proof/prepared_step_bindings_digest",
        &artifact.stage3.semantics.prepared_step_bindings_digest,
    );
    semantics_tr.append_message(
        b"rv64im/stage3_semantics_proof/stage2_temporal_digest",
        &artifact.stage3.semantics.stage2_temporal_digest,
    );
    semantics_tr.append_u64s(
        b"rv64im/stage3_semantics_proof/meta",
        &[
            artifact.stage3.semantics.initial_pc,
            artifact.stage3.semantics.final_pc,
            artifact.stage3.semantics.real_row_count,
            artifact.stage3.semantics.first_real_step_index,
            artifact.stage3.semantics.last_real_step_index,
        ],
    );
    artifact.stage3.semantics.digest = semantics_tr.digest32();

    let mut stage3_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage3_proof_bundle");
    stage3_tr.append_message(b"rv64im/stage3_proof_bundle/bridge", &artifact.stage3.bridge.digest);
    stage3_tr.append_message(
        b"rv64im/stage3_proof_bundle/semantics",
        &artifact.stage3.semantics.digest,
    );
    stage3_tr.append_message(b"rv64im/stage3_proof_bundle/linkage", &artifact.stage3.linkage.digest);
    stage3_tr.append_message(
        b"rv64im/stage3_proof_bundle/selected_opening",
        &artifact.stage3.selected_opening.digest,
    );
    artifact.stage3.digest = stage3_tr.digest32();
    refresh_accepted_artifact_digest(artifact);
}

#[allow(dead_code)]
pub fn refresh_step_composition_surface_digest(artifact: &mut Rv64imAcceptedProofArtifact) {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/step_composition_surface");
    tr.append_message(
        b"rv64im/step_composition_surface/stage1_semantics_digest",
        &artifact.step_composition.stage1_semantics_digest,
    );
    tr.append_message(
        b"rv64im/step_composition_surface/stage2_semantics_digest",
        &artifact.step_composition.stage2_semantics_digest,
    );
    tr.append_message(
        b"rv64im/step_composition_surface/stage2_temporal_digest",
        &artifact.step_composition.stage2_temporal_digest,
    );
    tr.append_message(
        b"rv64im/step_composition_surface/stage3_semantics_digest",
        &artifact.step_composition.stage3_semantics_digest,
    );
    tr.append_message(
        b"rv64im/step_composition_surface/root_execution_digest",
        &artifact.step_composition.root_execution_digest,
    );
    tr.append_message(
        b"rv64im/step_composition_surface/prepared_step_bindings_digest",
        &artifact.step_composition.prepared_step_bindings_digest,
    );
    tr.append_message(
        b"rv64im/step_composition_surface/row_chunk_routes_digest",
        &artifact.step_composition.row_chunk_routes_digest,
    );
    tr.append_u64s(
        b"rv64im/step_composition_surface/meta",
        &[
            artifact.step_composition.real_row_count,
            artifact.step_composition.prepared_step_count,
            artifact.step_composition.first_real_step_index,
            artifact.step_composition.last_real_step_index,
            artifact.step_composition.initial_pc,
            artifact.step_composition.final_pc,
            artifact.step_composition.halted as u64,
        ],
    );
    artifact.step_composition.digest = tr.digest32();
    refresh_accepted_artifact_digest(artifact);
}

#[allow(dead_code)]
pub fn refresh_soundness_accounting_surface_digest(artifact: &mut Rv64imAcceptedProofArtifact) {
    let mut schema_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_soundness_accounting_schema");
    schema_tr.append_u64s(
        b"rv64im/kernel_soundness_accounting_schema/version",
        &[artifact.soundness_accounting.schema_version],
    );
    schema_tr.append_u64s(
        b"rv64im/kernel_soundness_accounting_schema/stage1_shout_channel",
        &[artifact.soundness_accounting.stage1_shout_channels.len() as u64],
    );
    for value in &artifact.soundness_accounting.stage1_shout_channels {
        schema_tr.append_message(
            b"rv64im/kernel_soundness_accounting_schema/stage1_shout_channel",
            value.as_bytes(),
        );
    }
    schema_tr.append_u64s(
        b"rv64im/kernel_soundness_accounting_schema/stage1_address_family",
        &[artifact.soundness_accounting.stage1_address_families.len() as u64],
    );
    for value in &artifact.soundness_accounting.stage1_address_families {
        schema_tr.append_message(
            b"rv64im/kernel_soundness_accounting_schema/stage1_address_family",
            value.as_bytes(),
        );
    }
    schema_tr.append_u64s(
        b"rv64im/kernel_soundness_accounting_schema/stage2_address_family",
        &[artifact.soundness_accounting.stage2_address_families.len() as u64],
    );
    for value in &artifact.soundness_accounting.stage2_address_families {
        schema_tr.append_message(
            b"rv64im/kernel_soundness_accounting_schema/stage2_address_family",
            value.as_bytes(),
        );
    }
    schema_tr.append_u64s(
        b"rv64im/kernel_soundness_accounting_schema/twist_memory_family",
        &[artifact.soundness_accounting.twist_memory_families.len() as u64],
    );
    for value in &artifact.soundness_accounting.twist_memory_families {
        schema_tr.append_message(
            b"rv64im/kernel_soundness_accounting_schema/twist_memory_family",
            value.as_bytes(),
        );
    }
    schema_tr.append_u64s(
        b"rv64im/kernel_soundness_accounting_schema/scalar_term",
        &[artifact.soundness_accounting.scalar_terms.len() as u64],
    );
    for value in &artifact.soundness_accounting.scalar_terms {
        schema_tr.append_message(
            b"rv64im/kernel_soundness_accounting_schema/scalar_term",
            value.as_bytes(),
        );
    }
    artifact.soundness_accounting.schema_digest = schema_tr.digest32();

    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_soundness_accounting_surface");
    tr.append_u64s(
        b"rv64im/kernel_soundness_accounting_surface/meta",
        &[
            artifact.soundness_accounting.schema_version,
            artifact.soundness_accounting.stage1_shout_channels.len() as u64,
            artifact.soundness_accounting.stage1_address_families.len() as u64,
            artifact.soundness_accounting.stage2_address_families.len() as u64,
            artifact.soundness_accounting.twist_memory_families.len() as u64,
            artifact.soundness_accounting.scalar_terms.len() as u64,
        ],
    );
    tr.append_message(
        b"rv64im/kernel_soundness_accounting_surface/schema_digest",
        &artifact.soundness_accounting.schema_digest,
    );
    artifact.soundness_accounting.digest = tr.digest32();
    refresh_accepted_artifact_digest(artifact);
}
