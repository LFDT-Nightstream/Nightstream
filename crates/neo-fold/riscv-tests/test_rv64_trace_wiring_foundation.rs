use neo_fold::pi_ccs::FoldingMode;
use neo_fold::rv64_trace_shard::Rv64TraceWiring;
use neo_math::F;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

#[path = "support/rv64_elf.rs"]
mod rv64_elf;

use rv64_elf::{build_elf64, TestSegment};

#[test]
fn rv64_trace_wiring_prepare_remaps_outputs_into_small_logical_domain() {
    let text = encode_program(&[
        RiscvInstruction::IAluw {
            op: RiscvOpcode::Addw,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::Halt,
    ]);
    let elf = build_elf64(
        0x4000,
        &[
            TestSegment {
                vaddr: 0x4000,
                flags: 0x5,
                mem_size: text.len() as u64,
                data: text,
            },
            TestSegment {
                vaddr: 0x8000_0100,
                flags: 0x6,
                mem_size: 8,
                data: vec![0x11, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00],
            },
        ],
    );

    let prepared = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .ram_init_u64(0x9000_0000, 0x33)
        .output_claim(0x9000_0000, F::from_u64(0x33))
        .prepare()
        .expect("prepare");

    assert_eq!(prepared.entry_segment.vaddr, 0x4000);
    assert!(prepared.output_num_bits <= 4);
    assert_eq!(
        prepared.guest_output_claims.get_claim(0x9000_0000),
        Some(F::from_u64(0x33))
    );

    let logical_addr = prepared
        .logical_output_claims
        .claimed_addresses()
        .next()
        .expect("logical output claim");
    assert_ne!(logical_addr, 0x9000_0000);
    assert_eq!(
        prepared.logical_output_claims.get_claim(logical_addr),
        Some(F::from_u64(0x33))
    );
    assert_eq!(prepared.ram_init_words.get(&0x8000_0100), Some(&0x0000_0022_0000_0011));
    assert_eq!(prepared.ram_init_words.get(&0x8000_0104), Some(&0x22));
    assert_eq!(prepared.ram_init_words.get(&0x9000_0000), Some(&0x33));
}

#[test]
fn rv64_trace_wiring_prepare_rejects_compressed_code() {
    let elf = build_elf64(
        0x1000,
        &[TestSegment {
            vaddr: 0x1000,
            flags: 0x5,
            mem_size: 4,
            data: vec![0x01, 0x00, 0x00, 0x00],
        }],
    );

    let err = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .prepare()
        .expect_err("compressed code must be rejected");
    assert!(err.to_string().contains("compressed"), "unexpected error: {err}");
}

#[test]
fn rv64_trace_wiring_simulate_runs_supported_elf_program() {
    let text = encode_program(&[
        RiscvInstruction::IAluw {
            op: RiscvOpcode::Addw,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::Halt,
    ]);
    let elf = build_elf64(
        0x2000,
        &[TestSegment {
            vaddr: 0x2000,
            flags: 0x5,
            mem_size: text.len() as u64,
            data: text,
        }],
    );

    let trace = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .simulate()
        .expect("simulate");
    assert!(trace.did_halt(), "supported RV64 ELF should halt under simulation");
    assert!(!trace.steps.is_empty(), "simulation trace should not be empty");
}

#[test]
fn rv64_trace_wiring_rejects_multiple_executable_segments() {
    let text = encode_program(&[RiscvInstruction::Halt]);
    let elf = build_elf64(
        0x2000,
        &[
            TestSegment {
                vaddr: 0x2000,
                flags: 0x5,
                mem_size: text.len() as u64,
                data: text.clone(),
            },
            TestSegment {
                vaddr: 0x3000,
                flags: 0x5,
                mem_size: text.len() as u64,
                data: text,
            },
        ],
    );

    let err = Rv64TraceWiring::from_elf(&elf)
        .expect("from_elf")
        .mode(FoldingMode::Optimized)
        .prepare()
        .expect_err("multiple executable segments must be rejected in phase 1");
    assert!(
        err.to_string()
            .contains("exactly one executable PT_LOAD segment"),
        "unexpected error: {err}"
    );
}
