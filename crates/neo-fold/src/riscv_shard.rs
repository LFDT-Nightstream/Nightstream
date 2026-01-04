//! Convenience wrappers for verifying RISC-V shard proofs safely.
//!
//! These helpers are intentionally small: they standardize the step-linking configuration
//! for RV32 B1 chunked execution so callers don't accidentally verify a "bag of chunks".

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::time::{Duration, Instant};

use crate::pi_ccs::FoldingMode;
use crate::shard::{
    fold_shard_verify_with_output_binding_and_step_linking, fold_shard_verify_with_step_linking, CommitMixers,
    ShardFoldOutputs, ShardProof, StepLinkingConfig,
};
use crate::output_binding::simple_output_config;
use crate::session::FoldingSession;
use crate::PiCcsError;
use neo_ajtai::{AjtaiSModule, Commitment as Cmt};
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{F, K};
use neo_memory::mem_init_from_initial_mem;
use neo_memory::plain::LutTable;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{
    build_rv32_b1_step_ccs, rv32_b1_chunk_to_witness, rv32_b1_shared_cpu_bus_config, rv32_b1_step_linking_pairs,
    Rv32B1Layout,
};
use neo_memory::riscv::lookups::{decode_program, RiscvCpu, RiscvInstruction, RiscvOpcode, RiscvShoutTables, PROG_ID};
use neo_memory::riscv::shard::{extract_boundary_state, Rv32BoundaryState};
use neo_memory::witness::StepInstanceBundle;
use neo_memory::witness::LutTableSpec;
use neo_memory::R1csCpu;
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;

pub fn rv32_b1_step_linking_config(layout: &Rv32B1Layout) -> StepLinkingConfig {
    StepLinkingConfig::new(rv32_b1_step_linking_pairs(layout))
}

/// Enforce that the *public statement* initial memory matches chunk 0's `MemInstance.init`.
///
/// This lets later chunk `init` snapshots remain proof-internal rollover data (Twist needs them),
/// while keeping the user-facing statement independent of `chunk_size`.
pub fn rv32_b1_enforce_chunk0_mem_init_matches_statement<Cmt2, K2>(
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    statement_initial_mem: &HashMap<(u32, u64), F>,
    steps: &[StepInstanceBundle<Cmt2, F, K2>],
) -> Result<(), PiCcsError> {
    let chunk0 = steps
        .first()
        .ok_or_else(|| PiCcsError::InvalidInput("no steps provided".into()))?;

    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();

    if chunk0.mem_insts.len() != mem_ids.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "mem instance count mismatch: chunk0 has {}, but mem_layouts has {}",
            chunk0.mem_insts.len(),
            mem_ids.len()
        )));
    }

    for (idx, mem_id) in mem_ids.into_iter().enumerate() {
        let layout = mem_layouts.get(&mem_id).ok_or_else(|| {
            PiCcsError::InvalidInput(format!("missing PlainMemLayout for mem_id={mem_id}"))
        })?;
        let expected = mem_init_from_initial_mem(mem_id, layout.k, statement_initial_mem)?;
        let got = &chunk0.mem_insts[idx].init;
        if *got != expected {
            return Err(PiCcsError::InvalidInput(format!(
                "chunk0 MemInstance.init mismatch for mem_id={mem_id}"
            )));
        }
    }

    Ok(())
}

pub fn fold_shard_verify_rv32_b1<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    layout: &Rv32B1Layout,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let step_linking = rv32_b1_step_linking_config(layout);
    fold_shard_verify_with_step_linking(mode, tr, params, s_me, steps, acc_init, proof, mixers, &step_linking)
}

pub fn fold_shard_verify_rv32_b1_with_statement_mem_init<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    statement_initial_mem: &HashMap<(u32, u64), F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    layout: &Rv32B1Layout,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    rv32_b1_enforce_chunk0_mem_init_matches_statement(mem_layouts, statement_initial_mem, steps)?;
    fold_shard_verify_rv32_b1(mode, tr, params, s_me, steps, acc_init, proof, mixers, layout)
}

pub fn fold_shard_verify_rv32_b1_with_output_binding<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    layout: &Rv32B1Layout,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let step_linking = rv32_b1_step_linking_config(layout);
    fold_shard_verify_with_output_binding_and_step_linking(
        mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg, &step_linking,
    )
}

fn pow2_ceil_k(min_k: usize) -> (usize, usize) {
    // RV32 B1 alignment constraints require bit-addressed memories with d>=2.
    let k = min_k.next_power_of_two().max(4);
    let d = k.trailing_zeros() as usize;
    (k, d)
}

fn infer_required_shout_opcodes(program: &[RiscvInstruction]) -> HashSet<RiscvOpcode> {
    let mut ops = HashSet::new();

    // The ADD table is required because the step circuit uses it for address/PC wiring in multiple
    // instructions (LW/SW/AUIPC/JALR), even if the program has no explicit ADD/ADDI.
    ops.insert(RiscvOpcode::Add);

    for instr in program {
        match instr {
            RiscvInstruction::RAlu { op, .. } => {
                ops.insert(*op);
            }
            RiscvInstruction::IAlu { op, .. } => {
                ops.insert(*op);
            }
            RiscvInstruction::Branch { cond, .. } => {
                ops.insert(cond.to_shout_opcode());
            }
            RiscvInstruction::Load { .. } => {
                ops.insert(RiscvOpcode::Add);
            }
            RiscvInstruction::Store { .. } => {
                ops.insert(RiscvOpcode::Add);
            }
            RiscvInstruction::Jalr { .. } => {
                ops.insert(RiscvOpcode::Add);
            }
            RiscvInstruction::Auipc { .. } => {
                ops.insert(RiscvOpcode::Add);
            }
            RiscvInstruction::Amo { op, .. } => match op {
                neo_memory::riscv::lookups::RiscvMemOp::AmoaddW
                | neo_memory::riscv::lookups::RiscvMemOp::AmoaddD => {
                    ops.insert(RiscvOpcode::Add);
                }
                neo_memory::riscv::lookups::RiscvMemOp::AmoxorW
                | neo_memory::riscv::lookups::RiscvMemOp::AmoxorD => {
                    ops.insert(RiscvOpcode::Xor);
                }
                neo_memory::riscv::lookups::RiscvMemOp::AmoandW
                | neo_memory::riscv::lookups::RiscvMemOp::AmoandD => {
                    ops.insert(RiscvOpcode::And);
                }
                neo_memory::riscv::lookups::RiscvMemOp::AmoorW
                | neo_memory::riscv::lookups::RiscvMemOp::AmoorD => {
                    ops.insert(RiscvOpcode::Or);
                }
                _ => {}
            },
            _ => {}
        }
    }

    ops
}

fn all_shout_opcodes() -> HashSet<RiscvOpcode> {
    use RiscvOpcode::*;
    HashSet::from([
        And, Xor, Or, Sub, Add, Mul, Mulh, Mulhu, Mulhsu, Div, Divu, Rem, Remu, Sltu, Slt, Eq, Neq, Sll, Srl, Sra,
        Addw, Subw, Sllw, Srlw, Sraw, Mulw, Divw, Divuw, Remw, Remuw, Andn,
    ])
}

/// High-level “few lines” builder for proving/verifying an RV32 program using the B1 shared-bus step circuit.
///
/// This:
/// - chooses parameters + Ajtai committer automatically,
/// - infers the minimal Shout table set from the program (unless overridden),
/// - enforces RV32 B1 step linking, and
/// - (optionally) proves output binding against RAM.
#[derive(Clone, Debug)]
pub struct Rv32B1 {
    program_base: u64,
    program_bytes: Vec<u8>,
    xlen: usize,
    ram_bytes: usize,
    chunk_size: usize,
    max_steps: Option<usize>,
    mode: FoldingMode,
    shout_auto_minimal: bool,
    shout_ops: Option<HashSet<RiscvOpcode>>,
    output_claim: Option<(u64, F)>,
}

impl Rv32B1 {
    /// Create a runner from ROM bytes (must be a valid RV32 program encoding).
    pub fn from_rom(program_base: u64, program_bytes: &[u8]) -> Self {
        Self {
            program_base,
            program_bytes: program_bytes.to_vec(),
            xlen: 32,
            ram_bytes: 0x200,
            chunk_size: 1,
            max_steps: None,
            mode: FoldingMode::Optimized,
            shout_auto_minimal: true,
            shout_ops: None,
            output_claim: None,
        }
    }

    pub fn xlen(mut self, xlen: usize) -> Self {
        self.xlen = xlen;
        self
    }

    pub fn ram_bytes(mut self, ram_bytes: usize) -> Self {
        self.ram_bytes = ram_bytes;
        self
    }

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Limit the number of instructions executed from the decoded program.
    ///
    /// This is primarily for tests/benchmarks that want a tiny trace.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    pub fn mode(mut self, mode: FoldingMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn shout_auto_minimal(mut self) -> Self {
        self.shout_ops = None;
        self.shout_auto_minimal = true;
        self
    }

    pub fn shout_all(mut self) -> Self {
        self.shout_ops = None;
        self.shout_auto_minimal = false;
        self
    }

    pub fn shout_ops(mut self, ops: impl IntoIterator<Item = RiscvOpcode>) -> Self {
        self.shout_ops = Some(ops.into_iter().collect());
        self.shout_auto_minimal = false;
        self
    }

    pub fn output(mut self, output_addr: u64, expected_output: F) -> Self {
        self.output_claim = Some((output_addr, expected_output));
        self
    }

    pub fn prove(self) -> Result<Rv32B1Run, PiCcsError> {
        if self.xlen != 32 {
            return Err(PiCcsError::InvalidInput(format!(
                "RV32 B1 MVP requires xlen == 32 (got {})",
                self.xlen
            )));
        }
        if self.program_bytes.is_empty() {
            return Err(PiCcsError::InvalidInput("program_bytes must be non-empty".into()));
        }
        if self.chunk_size == 0 {
            return Err(PiCcsError::InvalidInput("chunk_size must be non-zero".into()));
        }
        if self.ram_bytes == 0 {
            return Err(PiCcsError::InvalidInput("ram_bytes must be non-zero".into()));
        }
        if self.program_base != 0 {
            return Err(PiCcsError::InvalidInput(
                "RV32 B1 MVP requires program_base == 0 (addresses are indices into PROG/RAM layouts)".into(),
            ));
        }
        if self.program_bytes.len() % 4 != 0 {
            return Err(PiCcsError::InvalidInput(
                "program_bytes must be 4-byte aligned (RV32 B1 runner does not support RVC)".into(),
            ));
        }
        for (i, chunk) in self.program_bytes.chunks_exact(4).enumerate() {
            let first_half = u16::from_le_bytes([chunk[0], chunk[1]]);
            if (first_half & 0b11) != 0b11 {
                return Err(PiCcsError::InvalidInput(format!(
                    "RV32 B1 runner does not support compressed instructions (RVC): found compressed encoding at word index {i}"
                )));
            }
        }

        let program = decode_program(&self.program_bytes)
            .map_err(|e| PiCcsError::InvalidInput(format!("decode_program failed: {e}")))?;
        let max_steps = match self.max_steps {
            Some(n) => {
                if n == 0 {
                    return Err(PiCcsError::InvalidInput("max_steps must be non-zero".into()));
                }
                n
            }
            None => program.len(),
        };
        let twist = neo_memory::riscv::lookups::RiscvMemory::with_program_in_twist(
            self.xlen,
            PROG_ID,
            /*base_addr=*/ 0,
            &self.program_bytes,
        );
        let shout = RiscvShoutTables::new(self.xlen);

        let (prog_layout, initial_mem) =
            neo_memory::riscv::rom_init::prog_rom_layout_and_init_words(PROG_ID, /*base_addr=*/ 0, &self.program_bytes)
                .map_err(|e| PiCcsError::InvalidInput(format!("prog_rom_layout_and_init_words failed: {e}")))?;

        let (k_ram, d_ram) = pow2_ceil_k(self.ram_bytes.max(4));
        let mem_layouts = HashMap::from([
            (neo_memory::riscv::lookups::RAM_ID.0, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
            (PROG_ID.0, prog_layout),
        ]);

        // Shout tables (either inferred, all, or explicitly provided).
        let mut shout_ops = match &self.shout_ops {
            Some(ops) => ops.clone(),
            None if self.shout_auto_minimal => infer_required_shout_opcodes(&program),
            None => all_shout_opcodes(),
        };
        // The ADD table is required even for programs without explicit ADD/ADDI due to address/PC wiring.
        shout_ops.insert(RiscvOpcode::Add);

        let mut table_specs: HashMap<u32, LutTableSpec> = HashMap::new();
        for op in shout_ops {
            let table_id = shout.opcode_to_id(op).0;
            table_specs.insert(table_id, LutTableSpec::RiscvOpcode { opcode: op, xlen: self.xlen });
        }
        let mut shout_table_ids: Vec<u32> = table_specs.keys().copied().collect();
        shout_table_ids.sort_unstable();

        let (ccs_base, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, self.chunk_size)
            .map_err(|e| PiCcsError::InvalidInput(format!("build_rv32_b1_step_ccs failed: {e}")))?;

        // Session + Ajtai committer + params (auto-picked for this CCS).
        let mut session = FoldingSession::<AjtaiSModule>::new_ajtai(self.mode.clone(), &ccs_base)?;
        let params = session.params().clone();
        let committer = session.committer().clone();

        let mut vm = RiscvCpu::new(self.xlen);
        vm.load_program(/*base=*/ 0, program);

        let empty_tables: HashMap<u32, LutTable<F>> = HashMap::new();

        // CPU arithmetization (builds chunk witnesses and commits them).
        let mut cpu = R1csCpu::new(
            ccs_base,
            params,
            committer,
            layout.m_in,
            &empty_tables,
            &table_specs,
            rv32_b1_chunk_to_witness(layout.clone()),
        );
        cpu = cpu
            .with_shared_cpu_bus(
                rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts.clone(), initial_mem.clone())
                    .map_err(|e| PiCcsError::InvalidInput(format!("rv32_b1_shared_cpu_bus_config failed: {e}")))?,
                self.chunk_size,
            )
            .map_err(|e| PiCcsError::InvalidInput(format!("shared bus inject failed: {e}")))?;

        // Always enforce step-to-step chunk chaining for RV32 B1.
        session.set_step_linking(rv32_b1_step_linking_config(&layout));

        // Execute + collect step bundles (and aux for output binding).
        session.execute_shard_shared_cpu_bus(
            vm,
            twist,
            shout,
            /*max_steps=*/ max_steps,
            self.chunk_size,
            &mem_layouts,
            &empty_tables,
            &table_specs,
            &initial_mem,
            &cpu,
        )?;

        // Enforce that the *statement* initial memory matches chunk 0's public MemInit.
        let steps_public = session.steps_public();
        rv32_b1_enforce_chunk0_mem_init_matches_statement(&mem_layouts, &initial_mem, &steps_public)?;

        let ccs = cpu.ccs.clone();

        // Prove phase (timed)
        let prove_start = Instant::now();
        let proof = match self.output_claim {
            Some((addr, expected)) => {
                let ob_cfg = simple_output_config(d_ram, addr, expected);
                session.fold_and_prove_with_output_binding_auto_simple(&ccs, &ob_cfg)?
            }
            None => session.fold_and_prove(&ccs)?,
        };
        let prove_duration = prove_start.elapsed();

        Ok(Rv32B1Run {
            session,
            proof,
            ccs,
            layout,
            mem_layouts,
            initial_mem,
            ram_num_bits: d_ram,
            output_claim: self.output_claim,
            prove_duration,
            verify_duration: None,
        })
    }
}

pub struct Rv32B1Run {
    session: FoldingSession<AjtaiSModule>,
    proof: ShardProof,
    ccs: CcsStructure<F>,
    layout: Rv32B1Layout,
    mem_layouts: HashMap<u32, PlainMemLayout>,
    initial_mem: HashMap<(u32, u64), F>,
    ram_num_bits: usize,
    output_claim: Option<(u64, F)>,
    prove_duration: Duration,
    verify_duration: Option<Duration>,
}

impl Rv32B1Run {
    pub fn verify(&mut self) -> Result<(), PiCcsError> {
        let verify_start = Instant::now();
        let ok = match self.output_claim {
            Some((addr, expected)) => {
                let ob_cfg = simple_output_config(self.ram_num_bits, addr, expected);
                self.session
                    .verify_with_output_binding_collected_simple(&self.ccs, &self.proof, &ob_cfg)?
            }
            None => self.session.verify_collected(&self.ccs, &self.proof)?,
        };
        self.verify_duration = Some(verify_start.elapsed());

        if !ok {
            return Err(PiCcsError::ProtocolError("verification failed".into()));
        }
        Ok(())
    }

    pub fn proof(&self) -> &ShardProof {
        &self.proof
    }

    pub fn steps_public(&self) -> Vec<StepInstanceBundle<Cmt, F, K>> {
        self.session.steps_public()
    }

    pub fn final_boundary_state(&self) -> Result<Rv32BoundaryState, PiCcsError> {
        let steps_public = self.steps_public();
        let last = steps_public
            .last()
            .ok_or_else(|| PiCcsError::InvalidInput("no steps collected".into()))?;
        extract_boundary_state(&self.layout, &last.mcs_inst.x)
            .map_err(|e| PiCcsError::InvalidInput(format!("extract_boundary_state failed: {e}")))
    }

    pub fn verify_output_claim(&self, output_addr: u64, expected_output: F) -> Result<bool, PiCcsError> {
        let ob_cfg = simple_output_config(self.ram_num_bits, output_addr, expected_output);
        self.session
            .verify_with_output_binding_collected_simple(&self.ccs, &self.proof, &ob_cfg)
    }

    pub fn verify_default_output_claim(&self) -> Result<bool, PiCcsError> {
        let Some((addr, expected)) = self.output_claim else {
            return Err(PiCcsError::InvalidInput("no output claim configured".into()));
        };
        self.verify_output_claim(addr, expected)
    }

    pub fn mem_layouts(&self) -> &HashMap<u32, PlainMemLayout> {
        &self.mem_layouts
    }

    pub fn initial_mem(&self) -> &HashMap<(u32, u64), F> {
        &self.initial_mem
    }

    pub fn prove_duration(&self) -> Duration {
        self.prove_duration
    }

    pub fn verify_duration(&self) -> Option<Duration> {
        self.verify_duration
    }
}
