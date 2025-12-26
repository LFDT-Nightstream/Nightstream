//! Full Twist + Shout + Output Binding for a toolchain-built RV32 ELF program.
//!
//! This test uses `crates/neo-fold/riscv-tests/guest-programs/build.sh fibonacci` to produce
//! `crates/neo-fold/riscv-tests/guest-programs/fibonacci.elf`
//! and then runs:
//! - ELF loading
//! - VM execution tracing
//! - Twist encoding over a mapped RAM domain (64KiB / 4-byte words)
//! - Shout encoding using implicit RV32 opcode tables (sparse address-domain oracle)
//! - Output binding against the final memory state

#![allow(non_snake_case)]

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::process::Command;

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
use neo_memory::plain::{PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::riscv_lookups::{RiscvCpu, RiscvMemory, RiscvOpcode, RiscvShoutTables};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_vm_trace::{trace_program, TwistOpKind, VmTrace};
use p3_field::PrimeCharacteristicRing;

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

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn guest_programs_dir(repo_root: &PathBuf) -> PathBuf {
    repo_root.join("crates/neo-fold/riscv-tests/guest-programs")
}

fn try_load_or_build_guest_elf(guest: &str) -> Result<Vec<u8>, PiCcsError> {
    let root = repo_root();
    let guest_dir = guest_programs_dir(&root);
    let elf_path = guest_dir.join(format!("{guest}.elf"));
    if let Ok(bytes) = std::fs::read(&elf_path) {
        return Ok(bytes);
    }

    if std::env::var("NEO_BUILD_GUEST_ELF").is_err() {
        eprintln!(
            "skipping: missing `{}` (build with `bash crates/neo-fold/riscv-tests/guest-programs/build.sh {}` or set NEO_BUILD_GUEST_ELF=1)",
            elf_path.display(),
            guest
        );
        return Ok(Vec::new());
    }

    let build_script = guest_dir.join("build.sh");
    let status = Command::new("bash")
        .arg(build_script)
        .arg(guest)
        .current_dir(&root)
        .status()
        .map_err(|e| PiCcsError::InvalidInput(format!("failed to run guest build script: {e}")))?;
    if !status.success() {
        return Err(PiCcsError::InvalidInput(format!(
            "guest build script failed with status {status}"
        )));
    }

    std::fs::read(&elf_path)
        .map_err(|e| PiCcsError::InvalidInput(format!("failed to read `{}`: {e}", elf_path.display())))
}

fn is_supported_implicit_rv32_opcode(op: RiscvOpcode) -> bool {
    matches!(
        op,
        RiscvOpcode::And
            | RiscvOpcode::Xor
            | RiscvOpcode::Or
            | RiscvOpcode::Add
            | RiscvOpcode::Sub
            | RiscvOpcode::Eq
            | RiscvOpcode::Neq
            | RiscvOpcode::Slt
            | RiscvOpcode::Sltu
    )
}

fn map_word_addr(addr: u64, mem_base: u64, mem_size_bytes: u64, word_bytes: u64) -> Result<u64, PiCcsError> {
    if addr < mem_base || addr >= mem_base + mem_size_bytes {
        return Err(PiCcsError::InvalidInput(format!(
            "address out of mapped RAM range: addr={addr:#x}, range=[{mem_base:#x}, {:#x})",
            mem_base + mem_size_bytes
        )));
    }
    let off = addr - mem_base;
    if off % word_bytes != 0 {
        return Err(PiCcsError::InvalidInput(format!(
            "unaligned word access: addr={addr:#x} (word_bytes={word_bytes})"
        )));
    }
    Ok(off / word_bytes)
}

fn trace_to_plain_mem_trace_word_mapped(
    trace: &VmTrace<u64, u64>,
    mem_base: u64,
    mem_size_bytes: u64,
    word_bytes: u64,
) -> Result<PlainMemTrace<F>, PiCcsError> {
    let steps = trace.len();

    let mut has_read = vec![F::ZERO; steps];
    let mut has_write = vec![F::ZERO; steps];
    let mut read_addr = vec![0u64; steps];
    let mut write_addr = vec![0u64; steps];
    let mut read_val = vec![F::ZERO; steps];
    let mut write_val = vec![F::ZERO; steps];
    let mut inc_at_write_addr = vec![F::ZERO; steps];

    let mut mem_state: HashMap<u64, F> = HashMap::new();

    for (j, step) in trace.steps.iter().enumerate() {
        let mut saw_read = false;
        let mut saw_write = false;

        for event in &step.twist_events {
            match event.kind {
                TwistOpKind::Read => {
                    if saw_read {
                        return Err(PiCcsError::InvalidInput(format!(
                            "multiple reads in a single step (step={j})"
                        )));
                    }
                    saw_read = true;
                    has_read[j] = F::ONE;
                    read_addr[j] = map_word_addr(event.addr, mem_base, mem_size_bytes, word_bytes)?;
                    read_val[j] = F::from_u64(event.value);
                }
                TwistOpKind::Write => {
                    if saw_write {
                        return Err(PiCcsError::InvalidInput(format!(
                            "multiple writes in a single step (step={j})"
                        )));
                    }
                    saw_write = true;
                    has_write[j] = F::ONE;
                    let addr = map_word_addr(event.addr, mem_base, mem_size_bytes, word_bytes)?;
                    write_addr[j] = addr;
                    let new_val = F::from_u64(event.value);
                    write_val[j] = new_val;

                    let old_val = mem_state.get(&addr).copied().unwrap_or(F::ZERO);
                    inc_at_write_addr[j] = new_val - old_val;
                    mem_state.insert(addr, new_val);
                }
            }
        }
    }

    Ok(PlainMemTrace {
        steps,
        has_read,
        has_write,
        read_addr,
        write_addr,
        read_val,
        write_val,
        inc_at_write_addr,
    })
}

fn mem_init_from_segments_word_mapped(
    segments: &[(u64, Vec<u8>)],
    mem_base: u64,
    mem_size_bytes: u64,
    word_bytes: u64,
    k_words: u64,
) -> Result<MemInit<F>, PiCcsError> {
    let mut words: BTreeMap<u64, u32> = BTreeMap::new();

    for (seg_base, seg_bytes) in segments.iter() {
        for (i, &b) in seg_bytes.iter().enumerate() {
            let addr = seg_base.wrapping_add(i as u64);
            if addr < mem_base || addr >= mem_base + mem_size_bytes {
                continue;
            }
            let off = addr - mem_base;
            let word_off = off & !(word_bytes - 1);
            let byte_in_word = (off - word_off) as usize;
            let word_idx = word_off / word_bytes;
            if word_idx >= k_words {
                continue;
            }

            let cur = words.get(&word_idx).copied().unwrap_or(0);
            let mut bytes = cur.to_le_bytes();
            bytes[byte_in_word] = b;
            words.insert(word_idx, u32::from_le_bytes(bytes));
        }
    }

    let mut pairs: Vec<(u64, F)> = Vec::new();
    for (word_idx, word) in words.into_iter() {
        if word != 0 {
            pairs.push((word_idx, F::from_u64(word as u64)));
        }
    }
    Ok(MemInit::Sparse(pairs))
}

fn final_memory_state_from_init_and_trace(
    init: &MemInit<F>,
    k_words: usize,
    mem_trace: &PlainMemTrace<F>,
) -> Result<Vec<F>, PiCcsError> {
    let mut state = vec![F::ZERO; k_words];

    if let MemInit::Sparse(pairs) = init {
        for &(addr, val) in pairs.iter() {
            let idx = usize::try_from(addr).map_err(|_| {
                PiCcsError::InvalidInput(format!("MemInit address doesn't fit usize: addr={addr}"))
            })?;
            if idx >= k_words {
                return Err(PiCcsError::InvalidInput(format!(
                    "MemInit address out of range: addr={addr} >= k_words={k_words}"
                )));
            }
            state[idx] = val;
        }
    }

    for j in 0..mem_trace.steps {
        if mem_trace.has_write[j] == F::ONE {
            let idx = usize::try_from(mem_trace.write_addr[j]).map_err(|_| {
                PiCcsError::InvalidInput(format!(
                    "write_addr doesn't fit usize: addr={}",
                    mem_trace.write_addr[j]
                ))
            })?;
            state[idx] = mem_trace.write_val[j];
        }
    }

    Ok(state)
}

#[test]
fn test_riscv_toolchain_elf_fibonacci_twist_shout_output_binding() -> Result<(), PiCcsError> {
    const XLEN: usize = 32;

    // Match `crates/neo-fold/riscv-tests/guest-programs/fibonacci/link.ld`.
    const MEM_BASE: u64 = 0x8000_0000;
    const MEM_SIZE_BYTES: u64 = 64 * 1024;
    const WORD_BYTES: u64 = 4;
    const K_WORDS: u64 = MEM_SIZE_BYTES / WORD_BYTES; // 16384
    const NUM_BITS: usize = 14; // log2(16384)

    const OUTPUT_ADDR_PHYS: u64 = 0x8000_1000;
    let output_idx = map_word_addr(OUTPUT_ADDR_PHYS, MEM_BASE, MEM_SIZE_BYTES, WORD_BYTES)?;

    let elf = try_load_or_build_guest_elf("fibonacci")?;
    if elf.is_empty() {
        return Ok(());
    }

    let loaded = load_elf(&elf).map_err(|e| PiCcsError::InvalidInput(format!("load_elf failed: {e}")))?;
    assert!(!loaded.is_64bit, "expected ELF32");

    // Initialize memory from PT_LOAD segments so toolchain programs can read .rodata/.data.
    let mut memory = RiscvMemory::new(XLEN);
    for (vaddr, seg) in loaded.segments.iter() {
        for (i, &byte) in seg.iter().enumerate() {
            memory.write_byte(vaddr + i as u64, byte);
        }
    }

    // CPU init: set stack pointer to the end of RAM.
    let mut cpu = RiscvCpu::new(XLEN);
    cpu.load_program(loaded.entry, loaded.get_instructions());
    cpu.set_reg(2, MEM_BASE + MEM_SIZE_BYTES - WORD_BYTES);

    let shout_tables = RiscvShoutTables::new(XLEN);
    let trace = trace_program(cpu, memory, shout_tables, 20_000).expect("trace_program");
    assert!(trace.did_halt(), "program should halt");

    // Sanity: ensure the program writes the expected output.
    let mut last_out: Option<u64> = None;
    for step in trace.steps.iter() {
        for ev in step.twist_events.iter() {
            if ev.kind == TwistOpKind::Write && ev.addr == OUTPUT_ADDR_PHYS {
                last_out = Some(ev.value);
            }
        }
    }
    assert_eq!(last_out, Some(55), "expected Fibonacci(10)=55 at {OUTPUT_ADDR_PHYS:#x}");

    // Map the RAM byte-addressed trace to a word-addressed Twist domain.
    let mem_trace = trace_to_plain_mem_trace_word_mapped(&trace, MEM_BASE, MEM_SIZE_BYTES, WORD_BYTES)?;
    let mem_init = mem_init_from_segments_word_mapped(
        &loaded.segments,
        MEM_BASE,
        MEM_SIZE_BYTES,
        WORD_BYTES,
        K_WORDS,
    )?;
    let final_memory_state = final_memory_state_from_init_and_trace(&mem_init, K_WORDS as usize, &mem_trace)?;
    assert_eq!(final_memory_state[output_idx as usize], F::from_u64(55));

    // Build opcode-separated Shout traces from VM trace, proving only opcodes with implicit RV32 MLEs.
    let mut used_by_id: BTreeMap<u32, RiscvOpcode> = BTreeMap::new();
    let op_map = RiscvShoutTables::new(XLEN);
    for (j, step) in trace.steps.iter().enumerate() {
        if step.shout_events.len() > 1 {
            return Err(PiCcsError::InvalidInput(format!(
                "multiple Shout events in a single step (step={j})"
            )));
        }
        if let Some(ev) = step.shout_events.first() {
            let op = op_map
                .id_to_opcode(ev.shout_id)
                .ok_or_else(|| PiCcsError::InvalidInput(format!("unknown shout_id={}", ev.shout_id.0)))?;
            if !is_supported_implicit_rv32_opcode(op) {
                return Err(PiCcsError::InvalidInput(format!(
                    "implicit RV32 Shout MLE not implemented for opcode {op:?} (shout_id={})",
                    ev.shout_id.0
                )));
            }
            used_by_id.insert(ev.shout_id.0, op);
        }
    }

    let steps = trace.len();
    let mut traces_by_id: BTreeMap<u32, (RiscvOpcode, PlainLutTrace<F>)> = BTreeMap::new();
    for (id, op) in used_by_id.iter() {
        traces_by_id.insert(
            *id,
            (
                *op,
                PlainLutTrace {
                    has_lookup: vec![F::ZERO; steps],
                    addr: vec![0u64; steps],
                    val: vec![F::ZERO; steps],
                },
            ),
        );
    }
    for (j, step) in trace.steps.iter().enumerate() {
        if let Some(ev) = step.shout_events.first() {
            let entry = traces_by_id
                .get_mut(&ev.shout_id.0)
                .expect("shout_id should have been collected");
            entry.1.has_lookup[j] = F::ONE;
            entry.1.addr[j] = ev.key;
            entry.1.val[j] = F::from_u64(ev.value);
        }
    }

    // Minimal CCS + dummy CPU witness (unconstrained).
    // Route-A's time/row domain size is derived from `ccs.n`, so ensure `2^ell_n >= steps`.
    let ccs_n = steps.next_power_of_two().max(2);
    let ccs = create_identity_ccs(ccs_n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs_n).expect("params");
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

    let mem_layout = PlainMemLayout {
        k: K_WORDS as usize,
        d: NUM_BITS,
        n_side: 2,
    };

    session.add_step_from_io_with_sidecars(&cpu_input, |sc| {
        sc.add_twist(&mem_layout, &mem_init, &mem_trace);
        for (_id, (op, lut_trace)) in traces_by_id.iter() {
            sc.add_riscv_shout(*op, XLEN, lut_trace);
        }
        Ok(())
    })?;

    // Output binding: claim the mapped output word equals 55.
    let ob_cfg = OutputBindingConfig::new(NUM_BITS, ProgramIO::new().with_output(output_idx, F::from_u64(55)));
    let run = session.fold_and_prove_with_output_binding_simple(&ccs, &ob_cfg, &final_memory_state)?;
    let ok = session.verify_with_output_binding_collected_simple(&ccs, &run, &ob_cfg)?;
    assert!(ok);

    Ok(())
}
