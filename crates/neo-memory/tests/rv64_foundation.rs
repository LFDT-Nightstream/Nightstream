use neo_memory::output_check::ProgramIO;
use neo_memory::riscv::elf_loader::load_elf;
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_instruction_with_xlen, decode_program, encode_instruction, RiscvCpu, RiscvMemory, RiscvShoutTables, PROG_ID,
};
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};
use neo_memory::riscv::trace::{Rv64TraceLayout, Rv64TraceWitness};
use neo_memory::{
    lower_loaded_program, RiscvGuestMemoryLayout, RiscvProofProfile, RiscvProofProfileConfig, RiscvProofProfileError,
    RiscvTraceProfileKind,
};
use neo_vm_trace::{trace_program, Twist, VmCpu};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

const ELF_HDR64_SIZE: usize = 64;
const ELF_PHDR64_SIZE: usize = 56;
const PT_LOAD: u32 = 1;

#[derive(Clone)]
struct TestSegment {
    vaddr: u64,
    flags: u32,
    mem_size: u64,
    data: Vec<u8>,
}

fn build_elf64(entry: u64, segments: &[TestSegment]) -> Vec<u8> {
    let phoff = ELF_HDR64_SIZE as u64;
    let mut offset = ELF_HDR64_SIZE + ELF_PHDR64_SIZE * segments.len();
    let mut file = vec![0u8; offset];

    file[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    file[4] = 2;
    file[5] = 1;
    file[6] = 1;
    file[7] = 0;
    file[16..18].copy_from_slice(&2u16.to_le_bytes());
    file[18..20].copy_from_slice(&0xF3u16.to_le_bytes());
    file[20..24].copy_from_slice(&1u32.to_le_bytes());
    file[24..32].copy_from_slice(&entry.to_le_bytes());
    file[32..40].copy_from_slice(&phoff.to_le_bytes());
    file[52..54].copy_from_slice(&(ELF_HDR64_SIZE as u16).to_le_bytes());
    file[54..56].copy_from_slice(&(ELF_PHDR64_SIZE as u16).to_le_bytes());
    file[56..58].copy_from_slice(&(segments.len() as u16).to_le_bytes());

    for (idx, segment) in segments.iter().enumerate() {
        let ph = ELF_HDR64_SIZE + idx * ELF_PHDR64_SIZE;
        file[ph..ph + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
        file[ph + 4..ph + 8].copy_from_slice(&segment.flags.to_le_bytes());
        file[ph + 8..ph + 16].copy_from_slice(&(offset as u64).to_le_bytes());
        file[ph + 16..ph + 24].copy_from_slice(&segment.vaddr.to_le_bytes());
        file[ph + 24..ph + 32].copy_from_slice(&segment.vaddr.to_le_bytes());
        file[ph + 32..ph + 40].copy_from_slice(&(segment.data.len() as u64).to_le_bytes());
        file[ph + 40..ph + 48].copy_from_slice(&segment.mem_size.to_le_bytes());
        file[ph + 48..ph + 56].copy_from_slice(&8u64.to_le_bytes());

        file.extend_from_slice(&segment.data);
        offset += segment.data.len();
    }

    file
}

#[test]
fn rv64_profile_rejects_compressed_elf_code() {
    let elf = build_elf64(
        0x1000,
        &[TestSegment {
            vaddr: 0x1000,
            flags: 0x5,
            mem_size: 4,
            data: vec![0x01, 0x00, 0x00, 0x00],
        }],
    );
    let loaded = load_elf(&elf).expect("load elf");
    let profile = RiscvProofProfile::rv64_note_circuits_phase1();

    let err = profile
        .validate_loaded_program(&loaded)
        .expect_err("compressed code must be rejected");
    assert_eq!(err, RiscvProofProfileError::CompressedProgramNotSupported);
}

#[test]
fn guest_memory_layout_remaps_sparse_guest_addresses_into_dense_domain() {
    let instructions = encode_program(&[
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
                mem_size: instructions.len() as u64,
                data: instructions,
            },
            TestSegment {
                vaddr: 0x8000_0100,
                flags: 0x6,
                mem_size: 16,
                data: vec![1, 0, 0, 0, 2, 0, 0, 0],
            },
        ],
    );
    let loaded = load_elf(&elf).expect("load elf");
    let layout = RiscvGuestMemoryLayout::from_loaded_program(&loaded, 64).expect("layout");

    let code_addr = layout.remap_address(0x4000).expect("code address");
    let data_addr = layout.remap_address(0x8000_0100).expect("data address");
    let data_addr_2 = layout
        .remap_address(0x8000_0104)
        .expect("second data address");

    assert_eq!(code_addr, 0);
    assert_eq!(data_addr, 2);
    assert_eq!(data_addr_2, 3);
    assert!(layout.required_num_bits() <= 3);

    let guest_io = ProgramIO::new().with_output(0x8000_0104, Goldilocks::from_u64(9));
    let logical_io = layout.remap_program_io(&guest_io).expect("remap claims");
    assert_eq!(logical_io.get_claim(3), Some(Goldilocks::from_u64(9)));
}

#[test]
fn guest_memory_layout_is_stable_even_if_elf_segments_are_out_of_order() {
    let instructions = encode_program(&[
        RiscvInstruction::IAluw {
            op: RiscvOpcode::Addw,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ]);
    let elf = build_elf64(
        0x4000,
        &[
            TestSegment {
                vaddr: 0x8000_0100,
                flags: 0x6,
                mem_size: 8,
                data: vec![1, 0, 0, 0, 2, 0, 0, 0],
            },
            TestSegment {
                vaddr: 0x4000,
                flags: 0x5,
                mem_size: instructions.len() as u64,
                data: instructions,
            },
        ],
    );
    let loaded = load_elf(&elf).expect("load elf");
    let layout = RiscvGuestMemoryLayout::from_loaded_program(&loaded, 64).expect("layout");

    assert_eq!(layout.remap_address(0x4000).expect("text address"), 0);
    assert_eq!(layout.remap_address(0x8000_0100).expect("data address"), 2);
}

#[test]
fn rv64_note_lowering_accepts_supported_program_and_rejects_atomics_profile() {
    let instructions = encode_program(&[
        RiscvInstruction::Load {
            op: RiscvMemOp::Ld,
            rd: 1,
            rs1: 2,
            imm: 0,
        },
        RiscvInstruction::IAluw {
            op: RiscvOpcode::Addw,
            rd: 3,
            rs1: 1,
            imm: 4,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 1,
            rs2: 3,
        },
        RiscvInstruction::Halt,
    ]);
    let elf = build_elf64(
        0x2000,
        &[TestSegment {
            vaddr: 0x2000,
            flags: 0x5,
            mem_size: instructions.len() as u64,
            data: instructions,
        }],
    );
    let loaded = load_elf(&elf).expect("load elf");
    let profile = RiscvProofProfile::rv64_note_circuits_phase1();
    let lowered = lower_loaded_program(&loaded, &profile).expect("lower program");

    assert_eq!(lowered.instructions.len(), 4);
    assert!(matches!(
        &lowered.instructions[1].1,
        neo_memory::LoweredInstruction::Passthrough(RiscvInstruction::IAluw {
            op: RiscvOpcode::Addw,
            ..
        })
    ));

    let atomics_profile = RiscvProofProfile::new(RiscvProofProfileConfig {
        xlen: 64,
        compressed: false,
        atomics: true,
        poseidon_precompile: cfg!(feature = "poseidon-precompile"),
        lowering_version: RiscvProofProfile::RV64_NOTE_CIRCUITS_LOWERING_VERSION,
        memory_layout_kind: neo_memory::ProofAddressRemapKind::SegmentedWordAddress,
        trace_profile: RiscvTraceProfileKind::Rv64NoteCircuitsPhase1,
    });
    assert_eq!(
        atomics_profile.expect_err("atomics must be rejected"),
        RiscvProofProfileError::AtomicsNotSupported
    );
}

#[test]
fn rv64_trace_layout_adds_exact_value_transport_columns() {
    let layout = Rv64TraceLayout::new();
    assert_eq!(layout.cols, 49, "RV64 trace width regression");
    assert_eq!(layout.rs1_val_lo32 + 1, layout.rs1_val_hi32);
    assert_eq!(layout.rs2_val_lo32 + 1, layout.rs2_val_hi32);
    assert_eq!(layout.rd_val_lo32 + 1, layout.rd_val_hi32);
    assert!(
        layout.shout_rhs_hi32 > layout.virtual_commit_from_prev,
        "RV64 exact transport columns must be appended after base trace cols"
    );
}

#[test]
fn rv64_trace_witness_populates_exact_low_high_limbs() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 33,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sll,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(64);
    cpu.load_program(0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(64, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(64);
    let trace = trace_program(cpu, twist, shout, 16).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_with_xlen(&trace, trace.steps.len(), /*machine_xlen=*/ 64)
        .expect("exec table");

    let layout = Rv64TraceLayout::new();
    let wit = Rv64TraceWitness::from_exec_table(&layout, &exec).expect("rv64 trace witness");
    let row = 2usize;
    assert_eq!(
        wit.cols[layout.rd_val][row],
        Goldilocks::from_u64(0x0000_0002_0000_0000u64)
    );
    assert_eq!(wit.cols[layout.rd_val_lo32][row], Goldilocks::ZERO);
    assert_eq!(wit.cols[layout.rd_val_hi32][row], Goldilocks::from_u64(2));
}

#[test]
fn rv64_auipc_negative_upper_immediate_sign_extends_before_jalr() {
    let sparse_program = vec![
        (0x2eb0, RiscvInstruction::Auipc { rd: 1, imm: 0xffffd }),
        (
            0x2eb4,
            RiscvInstruction::Jalr {
                rd: 1,
                rs1: 1,
                imm: 360,
            },
        ),
        (0x18, RiscvInstruction::Halt),
    ];

    let mut cpu = RiscvCpu::new(64);
    cpu.load_program_sparse(0x2eb0, sparse_program.clone());

    let mut memory = RiscvMemory::new(64);
    for (pc, inst) in &sparse_program {
        let word = encode_instruction(inst).to_le_bytes();
        for (i, byte) in word.iter().enumerate() {
            memory.store(PROG_ID, pc + i as u64, *byte as u64);
        }
    }

    let mut shout = RiscvShoutTables::new(64);
    let meta0 = cpu.step(&mut memory, &mut shout).expect("auipc step");
    assert_eq!(meta0.pc_after, 0x2eb4);
    assert_eq!(
        cpu.snapshot_regs()[1],
        0xffff_ffff_ffff_feb0,
        "AUIPC must sign-extend the upper immediate on RV64"
    );

    let meta1 = cpu.step(&mut memory, &mut shout).expect("jalr step");
    assert_eq!(meta1.pc_after, 0x18, "JALR target must land back in low text");
}

#[test]
fn rv64_decoder_accepts_6bit_shift_immediates() {
    let decoded = decode_instruction_with_xlen(0x0209_1913, 64).expect("rv64 decode slli shamt=32");
    assert!(matches!(
        decoded,
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 18,
            rs1: 18,
            imm: 32,
        }
    ));
}
