//! Full Twist + Shout + Output Binding for a tiny RV32 ELF program.
//!
//! This is an end-to-end integration test that exercises:
//! - ELF loading (`neo_memory::elf_loader::load_elf`)
//! - VM execution tracing (`neo_vm_trace::trace_program`)
//! - Twist encoding (RW memory)
//! - Shout encoding using an **implicit** RV32 opcode table (no 2^64 table materialization)
//! - Output binding against the final memory state

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::CcsStructure;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::output_binding::OutputBindingConfig;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{FoldingSession, ProveInput, SidecarLinkPolicy};
use neo_fold::PiCcsError;
use neo_math::F;
use neo_memory::elf_loader::load_elf;
use neo_memory::output_check::ProgramIO;
use neo_memory::plain::{PlainLutTrace, PlainMemLayout};
use neo_memory::riscv_lookups::{encode_program, trace_to_plain_mem_trace, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode, RiscvShoutTables};
use neo_memory::MemInit;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use neo_vm_trace::trace_program;

#[derive(Clone, Copy, Default)]
struct DummyCommit;

impl SModuleHomomorphism<F, Cmt> for DummyCommit {
    fn commit(&self, z: &Mat<F>) -> Cmt {
        Cmt::zeros(z.rows(), 1)
    }

    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let mut out = Mat::zero(rows, m_in, F::ZERO);
        for r in 0..rows {
            for c in 0..m_in.min(z.cols()) {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

fn create_identity_ccs(n: usize) -> CcsStructure<F> {
    let mat = Mat::identity(n);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

fn build_minimal_elf32_riscv(code: &[u8], entry: u32) -> Vec<u8> {
    // ELF32 header (52) + one program header (32) + code.
    let ehsize = 52usize;
    let phentsize = 32usize;
    let phoff = ehsize as u32;
    let p_offset = (ehsize + phentsize) as u32;

    let mut elf = vec![0u8; ehsize + phentsize];

    // e_ident
    elf[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    elf[4] = 1; // ELFCLASS32
    elf[5] = 1; // little-endian
    elf[6] = 1; // version

    // e_type = ET_EXEC (2)
    elf[16..18].copy_from_slice(&2u16.to_le_bytes());
    // e_machine = EM_RISCV (0xF3)
    elf[18..20].copy_from_slice(&0xF3u16.to_le_bytes());
    // e_version
    elf[20..24].copy_from_slice(&1u32.to_le_bytes());
    // e_entry
    elf[24..28].copy_from_slice(&entry.to_le_bytes());
    // e_phoff
    elf[28..32].copy_from_slice(&phoff.to_le_bytes());
    // e_ehsize
    elf[40..42].copy_from_slice(&(ehsize as u16).to_le_bytes());
    // e_phentsize
    elf[42..44].copy_from_slice(&(phentsize as u16).to_le_bytes());
    // e_phnum
    elf[44..46].copy_from_slice(&1u16.to_le_bytes());

    // Program header (PT_LOAD)
    let ph = ehsize;
    elf[ph + 0..ph + 4].copy_from_slice(&1u32.to_le_bytes()); // p_type
    elf[ph + 4..ph + 8].copy_from_slice(&p_offset.to_le_bytes()); // p_offset
    elf[ph + 8..ph + 12].copy_from_slice(&entry.to_le_bytes()); // p_vaddr
    elf[ph + 12..ph + 16].copy_from_slice(&entry.to_le_bytes()); // p_paddr
    elf[ph + 16..ph + 20].copy_from_slice(&(code.len() as u32).to_le_bytes()); // p_filesz
    elf[ph + 20..ph + 24].copy_from_slice(&(code.len() as u32).to_le_bytes()); // p_memsz
    elf[ph + 24..ph + 28].copy_from_slice(&5u32.to_le_bytes()); // p_flags = R|X
    elf[ph + 28..ph + 32].copy_from_slice(&4u32.to_le_bytes()); // p_align

    elf.extend_from_slice(code);
    elf
}

#[test]
fn test_rv32_elf_twist_shout_output_binding_end_to_end() -> Result<(), PiCcsError> {
    // Program:
    //   x1 = 7
    //   x2 = 35
    //   x3 = x1 + x2 = 42
    //   mem[0] = x3
    //   halt
    let program = vec![
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
            imm: 35,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 3,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];

    let code = encode_program(&program);
    let elf = build_minimal_elf32_riscv(&code, 0);
    let loaded = load_elf(&elf).expect("load_elf");
    assert!(!loaded.is_64bit, "expected ELF32");
    assert_eq!(loaded.entry, 0);

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(loaded.entry, loaded.get_instructions());

    // Important: keep data memory empty/zero; the CPU fetches instructions from its internal program vector.
    let memory = RiscvMemory::new(32);
    let shout = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout, 64).expect("trace_program");
    assert!(trace.did_halt(), "program should halt");

    // Check the expected store value appears in the trace.
    let store_step = trace.steps.iter().find(|s| !s.twist_events.is_empty()).expect("store step");
    let store_event = store_step.twist_events.first().expect("store event");
    assert_eq!(store_event.addr, 0);
    assert_eq!(store_event.value, 42);

    // Convert trace → plain traces.
    let mem_trace = trace_to_plain_mem_trace::<F>(&trace);

    let add_shout_id = RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Add);
    let steps = trace.len();
    let mut lut_trace = PlainLutTrace {
        has_lookup: vec![F::ZERO; steps],
        addr: vec![0u64; steps],
        val: vec![F::ZERO; steps],
    };
    for (j, step) in trace.steps.iter().enumerate() {
        if let Some(event) = step.shout_events.first() {
            if event.shout_id == add_shout_id {
                lut_trace.has_lookup[j] = F::ONE;
                lut_trace.addr[j] = event.key;
                lut_trace.val[j] = F::from_u64(event.value);
            }
        }
    }

    // Minimal CCS + dummy CPU witness (unconstrained).
    let n = 8usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;

    let l = DummyCommit::default();
    let mut session = FoldingSession::new(FoldingMode::PaperExact, params, l)
        .with_sidecar_link_policy(SidecarLinkPolicy::NoLinking);

    let cpu_witness: Vec<F> = vec![F::ZERO; ccs.m];
    let cpu_input = ProveInput {
        ccs: &ccs,
        public_input: &[],
        witness: &cpu_witness,
        output_claims: &[],
    };

    // One Twist memory instance with k=4 (2-bit addresses), initialized to zero.
    let mem_layout = PlainMemLayout { k: 4, d: 2, n_side: 2 };
    let mem_init = MemInit::Zero;
    session.add_step_from_io_with_sidecars(&cpu_input, |sc| {
        sc.add_twist(&mem_layout, &mem_init, &mem_trace);
        sc.add_riscv_shout(RiscvOpcode::Add, 32, &lut_trace);
        Ok(())
    })?;

    // Output binding: claim memory[0] == 42 in a 4-cell memory.
    let final_memory_state = vec![F::from_u64(42), F::ZERO, F::ZERO, F::ZERO];
    let ob_cfg = OutputBindingConfig::new(2, ProgramIO::new().with_output(0, F::from_u64(42)));
    let run = session.fold_and_prove_with_output_binding_simple(&ccs, &ob_cfg, &final_memory_state)?;
    let ok = session.verify_with_output_binding_collected_simple(&ccs, &run, &ob_cfg)?;
    assert!(ok);

    Ok(())
}
