#![allow(non_snake_case)]

use neo_fold::pi_ccs::FoldingMode;
use neo_fold::rv64_trace_shard::Rv64TraceWiring;
use neo_math::F;
use neo_memory::riscv::lookups::{RiscvInstruction, RiscvMemOp, RiscvOpcode};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

#[path = "support/rv64_elf.rs"]
mod rv64_elf;

use rv64_elf::build_text_elf64;

#[test]
fn test_rv64_trace_wiring_base_subset_prove_verify() {
    let elf = build_text_elf64(
        0x2000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 8,
            },
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_output_claim_exact_u64(3, 15)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
    assert_eq!(
        run.proof().riscv_profile.as_ref(),
        Some(run.profile_config()),
        "RV64 proof must carry the exact validated profile config"
    );
    assert_eq!(
        run.proof().riscv_memory_layout.as_ref(),
        Some(run.memory_layout()),
        "RV64 proof must carry the exact validated memory layout"
    );
}

#[cfg(feature = "poseidon-precompile")]
#[test]
fn test_rv64_trace_wiring_poseidon_precompile_prove_verify() {
    let elf = build_text_elf64(
        0x2000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 10,
                rs1: 0,
                imm: 1,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 11,
                rs1: 0,
                imm: 0,
            },
            RiscvInstruction::Poseidon2AbsorbElem { rs1: 10, rs2: 11 },
            RiscvInstruction::Poseidon2Finalize,
            RiscvInstruction::Poseidon2SqueezeWord { rd: 12, idx: 0 },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 10,
                rs1: 0,
                imm: 0,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 11,
                rs1: 0,
                imm: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(8)
        .max_steps(32)
        .prove()
        .expect("poseidon prove");

    let step = &run.proof().steps[0];
    assert!(!step.mem.poseidon_cycle_me_claims.is_empty());
    assert!(!step.mem.poseidon_local_me_claims.is_empty());
    assert!(!step.poseidon_cycle_fold.is_empty());
    assert!(step.poseidon_local_time.is_some());
    assert!(!step.poseidon_local_fold.is_empty());

    run.verify().expect("poseidon verify");
}

#[cfg(feature = "poseidon-precompile")]
#[test]
fn test_rv64_trace_wiring_poseidon_precompile_uses_low32_transport_words() {
    let elf = build_text_elf64(
        0x2000,
        &[
            RiscvInstruction::Lui { rd: 10, imm: 0x80000 },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 10,
                rs1: 10,
                imm: 0x123,
            },
            RiscvInstruction::Lui { rd: 11, imm: 0x90000 },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 11,
                rs1: 11,
                imm: 0x456,
            },
            RiscvInstruction::Poseidon2AbsorbElem { rs1: 10, rs2: 11 },
            RiscvInstruction::Poseidon2Finalize,
            RiscvInstruction::Poseidon2SqueezeWord { rd: 12, idx: 0 },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(8)
        .max_steps(32)
        .prove()
        .expect("poseidon prove with dirty high halves");

    run.verify()
        .expect("poseidon verify with dirty high halves");
}

#[test]
fn test_rv64_trace_wiring_proof_metadata_roundtrip_verify() {
    let elf = build_text_elf64(
        0x2000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 8,
            },
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            RiscvInstruction::Halt,
        ],
    );

    let run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_output_claim_exact_u64(3, 15)
        .prove()
        .expect("prove");

    let bytes = bincode::serialize(run.proof()).expect("serialize");
    let proof = bincode::deserialize(&bytes).expect("deserialize");
    run.verify_proof(&proof).expect("verify roundtrip");
}

#[test]
fn test_rv64_trace_wiring_rejects_missing_proof_metadata() {
    let elf = build_text_elf64(
        0x2000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::Halt,
        ],
    );

    let run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_output_claim_exact_u64(1, 7)
        .prove()
        .expect("prove");

    let mut proof = run.proof().clone();
    proof.riscv_profile = None;
    proof.riscv_memory_layout = None;
    let err = run
        .verify_proof(&proof)
        .expect_err("missing RV64 proof metadata must fail");
    assert!(
        err.to_string().contains("missing riscv_profile metadata"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rv64_trace_wiring_rejects_tampered_proof_profile_metadata() {
    let elf = build_text_elf64(
        0x2000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::Halt,
        ],
    );

    let run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_output_claim_exact_u64(1, 7)
        .prove()
        .expect("prove");

    let mut proof = run.proof().clone();
    let profile = proof.riscv_profile.as_mut().expect("profile metadata");
    profile.lowering_version += 1;
    let err = run
        .verify_proof(&proof)
        .expect_err("tampered RV64 profile metadata must fail");
    assert!(
        err.to_string().contains("invalid lowering_version") || err.to_string().contains("proof profile mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rv64_trace_wiring_ld_sd_prove_verify() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 16,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 42,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sd,
                rs1: 1,
                rs2: 2,
                imm: 0,
            },
            RiscvInstruction::Load {
                op: RiscvMemOp::Ld,
                rd: 3,
                rs1: 1,
                imm: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(5)
        .max_steps(16)
        .reg_output_claim_exact_u64(3, 42)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_unsigned_narrow_width_subset_prove_verify() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 64,
            },
            RiscvInstruction::Lui { rd: 2, imm: 0x12345 },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 2,
                imm: 0x678,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sw,
                rs1: 1,
                rs2: 2,
                imm: 0,
            },
            RiscvInstruction::Load {
                op: RiscvMemOp::Lwu,
                rd: 3,
                rs1: 1,
                imm: 0,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sh,
                rs1: 1,
                rs2: 2,
                imm: 8,
            },
            RiscvInstruction::Load {
                op: RiscvMemOp::Lhu,
                rd: 4,
                rs1: 1,
                imm: 8,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sb,
                rs1: 1,
                rs2: 2,
                imm: 16,
            },
            RiscvInstruction::Load {
                op: RiscvMemOp::Lbu,
                rd: 5,
                rs1: 1,
                imm: 16,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(10)
        .max_steps(32)
        .reg_output_claim_exact_u64(3, 0x1234_5678)
        .reg_output_claim_exact_u64(4, 0x5678)
        .reg_output_claim_exact_u64(5, 0x78)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_unsigned_narrow_store_ram_output_binding() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 64,
            },
            RiscvInstruction::Lui { rd: 2, imm: 0x12345 },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 2,
                imm: 0x678,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sw,
                rs1: 1,
                rs2: 2,
                imm: 0,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sh,
                rs1: 1,
                rs2: 2,
                imm: 8,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sb,
                rs1: 1,
                rs2: 2,
                imm: 16,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(7)
        .max_steps(24)
        .output_claim(64, F::from_u64(0x1234_5678))
        .output_claim(72, F::from_u64(0x5678))
        .output_claim(80, F::from_u64(0x78))
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_supported_w_subset_prove_verify() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAluw {
                op: RiscvOpcode::Addw,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::IAluw {
                op: RiscvOpcode::Sllw,
                rd: 2,
                rs1: 1,
                imm: 3,
            },
            RiscvInstruction::IAluw {
                op: RiscvOpcode::Srlw,
                rd: 3,
                rs1: 2,
                imm: 1,
            },
            RiscvInstruction::IAluw {
                op: RiscvOpcode::Sraw,
                rd: 4,
                rs1: 2,
                imm: 1,
            },
            RiscvInstruction::RAluw {
                op: RiscvOpcode::Subw,
                rd: 5,
                rs1: 2,
                rs2: 1,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(6)
        .max_steps(16)
        .reg_output_claim_exact_u64(4, 28)
        .reg_output_claim_exact_u64(5, 49)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_signed_narrow_positive_load_subset_prove_verify() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 64,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 0x7f,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sb,
                rs1: 1,
                rs2: 2,
                imm: 0,
            },
            RiscvInstruction::Load {
                op: RiscvMemOp::Lb,
                rd: 3,
                rs1: 1,
                imm: 0,
            },
            RiscvInstruction::Lui { rd: 4, imm: 0x1 },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 4,
                rs1: 4,
                imm: 0x234,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sh,
                rs1: 1,
                rs2: 4,
                imm: 8,
            },
            RiscvInstruction::Load {
                op: RiscvMemOp::Lh,
                rd: 5,
                rs1: 1,
                imm: 8,
            },
            RiscvInstruction::Lui { rd: 6, imm: 0x12345 },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 6,
                rs1: 6,
                imm: 0x678,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sw,
                rs1: 1,
                rs2: 6,
                imm: 16,
            },
            RiscvInstruction::Load {
                op: RiscvMemOp::Lw,
                rd: 7,
                rs1: 1,
                imm: 16,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(13)
        .max_steps(32)
        .reg_output_claim_exact_u64(3, 0x7f)
        .reg_output_claim_exact_u64(5, 0x1234)
        .reg_output_claim_exact_u64(7, 0x1234_5678)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_allows_noninjective_signed_narrow_load_final_result() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 64,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 0x80,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sb,
                rs1: 1,
                rs2: 2,
                imm: 0,
            },
            RiscvInstruction::Load {
                op: RiscvMemOp::Lb,
                rd: 3,
                rs1: 1,
                imm: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(5)
        .max_steps(16)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_allows_consumed_noninjective_signed_narrow_load_result() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 64,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 0x80,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sb,
                rs1: 1,
                rs2: 2,
                imm: 0,
            },
            RiscvInstruction::Load {
                op: RiscvMemOp::Lb,
                rd: 3,
                rs1: 1,
                imm: 0,
            },
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 4,
                rs1: 3,
                rs2: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(6)
        .max_steps(16)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_mulw_positive_subset_prove_verify() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 9,
            },
            RiscvInstruction::RAluw {
                op: RiscvOpcode::Mulw,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_output_claim_exact_u64(3, 63)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_allows_noninjective_mulw_final_result() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::Lui { rd: 1, imm: 0x40000 },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 2,
            },
            RiscvInstruction::RAluw {
                op: RiscvOpcode::Mulw,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_allows_consumed_noninjective_mulw_result() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::Lui { rd: 1, imm: 0x40000 },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 2,
            },
            RiscvInstruction::RAluw {
                op: RiscvOpcode::Mulw,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 4,
                rs1: 3,
                rs2: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(5)
        .max_steps(16)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_divuw_positive_subset_prove_verify() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 21,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 5,
            },
            RiscvInstruction::RAluw {
                op: RiscvOpcode::Divuw,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_output_claim_exact_u64(3, 4)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_remuw_positive_subset_prove_verify() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 21,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 5,
            },
            RiscvInstruction::RAluw {
                op: RiscvOpcode::Remuw,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_output_claim_exact_u64(3, 1)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_divw_positive_subset_prove_verify() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 3,
            },
            RiscvInstruction::RAluw {
                op: RiscvOpcode::Divw,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_output_claim_exact_u64(3, 2)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_remw_positive_subset_prove_verify() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 3,
            },
            RiscvInstruction::RAluw {
                op: RiscvOpcode::Remw,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_output_claim_exact_u64(3, 1)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_allows_noninjective_divw_final_result() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::Lui { rd: 1, imm: 0x80000 },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 1,
            },
            RiscvInstruction::RAluw {
                op: RiscvOpcode::Divw,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_allows_consumed_noninjective_divw_result() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::Lui { rd: 1, imm: 0x80000 },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 1,
            },
            RiscvInstruction::RAluw {
                op: RiscvOpcode::Divw,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 4,
                rs1: 3,
                rs2: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(5)
        .max_steps(16)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_ram_output_binding_prove_verify() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 16,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 42,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sd,
                rs1: 1,
                rs2: 2,
                imm: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .output_claim(16, F::from_u64(42))
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_rejects_tampered_ram_output_proof() {
    let elf = build_text_elf64(
        0x3000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 16,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 0,
                imm: 42,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sd,
                rs1: 1,
                rs2: 2,
                imm: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .output_claim(16, F::from_u64(42))
        .prove()
        .expect("prove");

    let mut proof = run.proof().clone();
    let output_proof = proof.output_proof.as_mut().expect("output proof");
    let _ = output_proof.output_sc.round_polys.pop();
    run.verify_proof(&proof)
        .expect_err("tampered RV64 RAM output proof must fail verification");
}

#[test]
fn test_rv64_trace_wiring_allows_non_injective_reg_init_when_only_consumed_internally() {
    let elf = build_text_elf64(
        0x4000,
        &[
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 1,
                rs2: 0,
            },
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 3,
                rs1: 2,
                rs2: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_init_u64(1, <Goldilocks as PrimeField64>::ORDER_U64)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_still_rejects_non_injective_reg_output_binding() {
    let elf = build_text_elf64(
        0x4000,
        &[
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 1,
                rs2: 0,
            },
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 3,
                rs1: 2,
                rs2: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let err = match Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_init_u64(1, <Goldilocks as PrimeField64>::ORDER_U64)
        .reg_output_claim(3, F::ZERO)
        .prove()
    {
        Ok(_) => panic!("non-injective RV64 register outputs must stay rejected until exact public transport exists"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("Goldilocks modulus"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rv64_trace_wiring_rejects_out_of_range_reg_init() {
    let elf = build_text_elf64(
        0x2000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::Halt,
        ],
    );

    let err = match Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_init_u64(32, 1)
        .prove()
    {
        Ok(_) => panic!("out-of-range RV64 reg init must fail"),
        Err(err) => err,
    };

    assert!(
        err.to_string()
            .contains("reg_init_u64: register index out of range"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rv64_trace_wiring_rejects_non_zero_x0_init() {
    let elf = build_text_elf64(
        0x2000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::Halt,
        ],
    );

    let err = match Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_init_u64(0, 1)
        .prove()
    {
        Ok(_) => panic!("non-zero x0 init must fail"),
        Err(err) => err,
    };

    assert!(
        err.to_string().contains("reg_init_u64: x0 must be 0"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rv64_trace_wiring_rejects_out_of_range_exact_reg_output_claim() {
    let elf = build_text_elf64(
        0x2000,
        &[
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::Halt,
        ],
    );

    let err = match Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_output_claim_exact_u64(32, 7)
        .prove()
    {
        Ok(_) => panic!("out-of-range exact reg output claim must fail"),
        Err(err) => err,
    };

    assert!(
        err.to_string()
            .contains("reg_output_claim_exact_u64: register index out of range"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rv64_trace_wiring_exact_reg_output_binding_allows_non_injective_public_value() {
    let elf = build_text_elf64(
        0x4000,
        &[
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 1,
                rs2: 0,
            },
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 3,
                rs1: 2,
                rs2: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let mut run = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_init_u64(1, <Goldilocks as PrimeField64>::ORDER_U64)
        .reg_output_claim_exact_u64(3, <Goldilocks as PrimeField64>::ORDER_U64)
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
fn test_rv64_trace_wiring_exact_reg_output_binding_wrong_high_limb_fails_verification() {
    let elf = build_text_elf64(
        0x4000,
        &[
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 1,
                rs2: 0,
            },
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Add,
                rd: 3,
                rs1: 2,
                rs2: 0,
            },
            RiscvInstruction::Halt,
        ],
    );

    let value = <Goldilocks as PrimeField64>::ORDER_U64;
    let run_ok = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_init_u64(1, value)
        .reg_output_claim_exact_u64(3, value)
        .prove()
        .expect("prove");

    let run_wrong = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .chunk_rows(4)
        .max_steps(16)
        .reg_init_u64(1, value)
        .reg_output_claim_exact_u64(3, value ^ (1u64 << 32))
        .prove()
        .expect("prove with wrong exact-reg claim");

    let err = run_wrong
        .verify_proof(run_ok.proof())
        .expect_err("wrong exact-reg high limb must fail verification");
    assert!(
        err.to_string().contains("verification failed")
            || err.to_string().contains("output binding")
            || err.to_string().contains("claim")
            || err.to_string().contains("output sumcheck failed")
            || err.to_string().contains("RoundCheckFailed"),
        "unexpected error: {err}"
    );
}
