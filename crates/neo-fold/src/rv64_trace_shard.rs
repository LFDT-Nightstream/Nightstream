//! Canonical RV64IM real-ELF trace proving path built on the shared-bus shard prover.
//!
//! This is the maintained note-circuit entrypoint:
//! - build guest code to RV64IM ELF,
//! - load it with `Rv64TraceWiring::from_elf(...)`,
//! - bind public RAM or exact register outputs,
//! - then prove/verify through the shared-bus shard flow.
//!
//! The path is intentionally conservative and only enables the subset that is
//! already sound under the current Goldilocks-backed trace arithmetization.
//! Supported product contract:
//! - ISA: RV64IM
//! - not supported: compressed instructions (`C`) and atomics (`A`)
//! - broader arbitrary-RV64IM program coverage remains an explicit follow-on
//!   expansion beyond the maintained note repro path.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use crate::output_binding::OutputBindingConfig;
use crate::pi_ccs::FoldingMode;
use crate::rv64_ram_bridge::{derive_rv64_ram_bridge, extend_layout_with_guest_addresses, segment_backed_ram_words};
use crate::session::FoldingSession;
use crate::shard::{ShardProof, StepLinkingConfig};
use crate::PiCcsError;
use neo_ajtai::AjtaiSModule;
use neo_ccs::CcsStructure;
use neo_math::{D, F};
use neo_memory::cpu::bus_layout::{
    build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes, ShoutInstanceShape,
};
use neo_memory::output_check::ProgramIO;
use neo_memory::plain::{LutTable, PlainMemLayout};
use neo_memory::riscv::ccs::{
    build_rv64_trace_wiring_ccs, rv64_trace_ccs_witness_from_exec_table, Rv64TraceCcsLayout, TraceShoutBusSpec,
};
use neo_memory::riscv::elf_loader::{load_elf, ElfLoadSegment, LoadedProgram};
use neo_memory::riscv::exec_table::{RiscvExecRow, RiscvExecTable};
use neo_memory::riscv::lookups::{
    RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID, REG_EXACT_ID,
    REG_ID,
};
use neo_memory::riscv::rom_init::prog_rom_layout_and_init_words;
use neo_memory::riscv::trace::{
    riscv_decode_lookup_backed_row_from_instr_word, riscv_decode_lookup_table_id_for_col,
    riscv_decode_lookup_transport_cols, riscv_trace_lookup_addr_group_for_table_id,
    riscv_trace_lookup_n_vals_for_table_id, riscv_trace_lookup_selector_group_for_table_id,
    rv64_width_lookup_backed_cols, rv64_width_lookup_table_id_for_col, rv64_width_sidecar_witness_from_exec_table,
    Rv32DecodeSidecarLayout, Rv64WidthSidecarLayout,
};
use neo_memory::{
    lower_loaded_program, LoweredInstruction, LoweredProgram, LutTableSpec, R1csCpu, RiscvGuestMemoryLayout,
    RiscvProofProfile, RiscvProofProfileConfig, RiscvProofProfileError,
};
use neo_params::NeoParams;
use neo_vm_trace::{trace_program, ShoutId, StepTrace, Twist as _, VmTrace};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand_chacha::rand_core::SeedableRng;

mod helpers;
use helpers::*;

#[cfg(target_arch = "wasm32")]
use js_sys::Date;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

const DEFAULT_RV64_TRACE_MAX_STEPS: usize = 1 << 20;
const DEFAULT_RV64_TRACE_CHUNK_ROWS: usize = 1 << 16;
const RV64_OPCODE_ELL_ADDR: usize = 128;
const RV64_PACKED_MUL_ADDR_GROUP: u64 = 0x5256_4110;

#[cfg(target_arch = "wasm32")]
type TimePoint = f64;
#[cfg(not(target_arch = "wasm32"))]
type TimePoint = Instant;

#[inline]
fn time_now() -> TimePoint {
    #[cfg(target_arch = "wasm32")]
    {
        Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Instant::now()
    }
}

#[inline]
fn elapsed_duration(start: TimePoint) -> Duration {
    #[cfg(target_arch = "wasm32")]
    {
        let elapsed_ms = Date::now() - start;
        Duration::from_secs_f64(elapsed_ms / 1_000.0)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        start.elapsed()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum OutputTarget {
    #[default]
    Ram,
    Reg,
    RegExact,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv64TraceProvePhaseDurations {
    pub setup: Duration,
    pub chunk_build_commit: Duration,
    pub fold_and_prove: Duration,
}

#[derive(Clone, Debug)]
pub struct Rv64TraceWiring {
    elf_bytes: Vec<u8>,
    loaded_program: LoadedProgram,
    profile: RiscvProofProfile,
    max_steps: Option<usize>,
    min_trace_len: usize,
    chunk_rows: Option<usize>,
    mode: FoldingMode,
    ram_init: HashMap<u64, u64>,
    reg_init: HashMap<u64, u64>,
    output_claims: ProgramIO<F>,
    exact_reg_output_words: BTreeMap<u64, u64>,
    output_target: OutputTarget,
}

#[derive(Clone, Debug)]
pub struct Rv64PreparedProgram {
    pub profile: RiscvProofProfile,
    pub memory_layout: RiscvGuestMemoryLayout,
    pub lowered_program: LoweredProgram,
    pub entry_segment: ElfLoadSegment,
    pub ram_init_words: HashMap<u64, u64>,
    pub reg_init_words: HashMap<u64, u64>,
    pub guest_output_claims: ProgramIO<F>,
    pub logical_output_claims: ProgramIO<F>,
    pub output_num_bits: usize,
    output_target: OutputTarget,
}

pub struct Rv64TraceWiringRun {
    session: FoldingSession<AjtaiSModule>,
    ccs: CcsStructure<F>,
    layout: Rv64TraceCcsLayout,
    exec: RiscvExecTable,
    proof: ShardProof,
    used_mem_ids: Vec<u32>,
    used_shout_table_ids: Vec<u32>,
    output_binding_cfg: Option<OutputBindingConfig>,
    profile_config: RiscvProofProfileConfig,
    memory_layout: RiscvGuestMemoryLayout,
    prove_duration: Duration,
    prove_phase_durations: Rv64TraceProvePhaseDurations,
    verify_duration: Option<Duration>,
}

impl Rv64TraceWiring {
    pub fn from_elf(elf_bytes: &[u8]) -> Result<Self, PiCcsError> {
        let loaded_program =
            load_elf(elf_bytes).map_err(|e| PiCcsError::InvalidInput(format!("load_elf failed: {e}")))?;
        Ok(Self {
            elf_bytes: elf_bytes.to_vec(),
            loaded_program,
            profile: RiscvProofProfile::rv64im(),
            max_steps: None,
            min_trace_len: 4,
            chunk_rows: None,
            mode: FoldingMode::Optimized,
            ram_init: HashMap::new(),
            reg_init: HashMap::new(),
            output_claims: ProgramIO::new(),
            exact_reg_output_words: BTreeMap::new(),
            output_target: OutputTarget::Ram,
        })
    }

    pub fn profile(mut self, profile: RiscvProofProfile) -> Self {
        self.profile = profile;
        self
    }

    pub fn min_trace_len(mut self, min_trace_len: usize) -> Self {
        self.min_trace_len = min_trace_len.max(1);
        self
    }

    pub fn chunk_rows(mut self, chunk_rows: usize) -> Self {
        self.chunk_rows = Some(chunk_rows);
        self
    }

    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    pub fn mode(mut self, mode: FoldingMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn ram_init_u32(mut self, addr: u64, value: u32) -> Self {
        self.ram_init.insert(addr, value as u64);
        self
    }

    pub fn ram_init_u64(mut self, addr: u64, value: u64) -> Self {
        self.ram_init.insert(addr, value);
        self
    }

    pub fn reg_init_u64(mut self, reg: u64, value: u64) -> Self {
        self.reg_init.insert(reg, value);
        self
    }

    pub fn output_claim(mut self, addr: u64, value: F) -> Self {
        if !matches!(self.output_target, OutputTarget::Ram) {
            self.output_target = OutputTarget::Ram;
            self.output_claims = ProgramIO::new();
            self.exact_reg_output_words.clear();
        }
        self.output_claims = self.output_claims.with_output(addr, value);
        self
    }

    pub fn reg_output_claim(mut self, reg: u64, expected: F) -> Self {
        if !matches!(self.output_target, OutputTarget::Reg) {
            self.output_target = OutputTarget::Reg;
            self.output_claims = ProgramIO::new();
            self.exact_reg_output_words.clear();
        }
        self.output_claims = self.output_claims.with_output(reg, expected);
        self
    }

    pub fn reg_output_claim_exact_u64(mut self, reg: u64, expected: u64) -> Self {
        if !matches!(self.output_target, OutputTarget::RegExact) {
            self.output_target = OutputTarget::RegExact;
            self.output_claims = ProgramIO::new();
            self.exact_reg_output_words.clear();
        }
        self.exact_reg_output_words.insert(reg, expected);
        self
    }

    pub fn loaded_program(&self) -> &LoadedProgram {
        &self.loaded_program
    }

    pub fn elf_bytes(&self) -> &[u8] {
        &self.elf_bytes
    }

    pub fn prepare(&self) -> Result<Rv64PreparedProgram, PiCcsError> {
        self.profile
            .validate_loaded_program(&self.loaded_program)
            .map_err(profile_err_to_piccs)?;
        validate_rv64_reg_init_words(&self.reg_init)?;
        match self.output_target {
            OutputTarget::Ram => {}
            OutputTarget::Reg => validate_rv64_reg_output_claims(&self.output_claims, "reg_output_claim")?,
            OutputTarget::RegExact => validate_rv64_exact_reg_output_words(&self.exact_reg_output_words)?,
        }

        let lowered_program = lower_loaded_program(&self.loaded_program, &self.profile)
            .map_err(|e| PiCcsError::InvalidInput(e.to_string()))?;
        let exec_segment_count = self
            .loaded_program
            .segments
            .iter()
            .filter(|segment| segment.flags.execute)
            .count();
        if exec_segment_count != 1 {
            return Err(PiCcsError::InvalidInput(format!(
                "RV64 foundation currently requires exactly one executable PT_LOAD segment (got {exec_segment_count})"
            )));
        }
        let entry_segment = self
            .loaded_program
            .entry_segment()
            .cloned()
            .ok_or_else(|| {
                PiCcsError::InvalidInput("ELF has no executable PT_LOAD segment containing entrypoint".into())
            })?;

        let mut memory_layout = RiscvGuestMemoryLayout::from_loaded_program(&self.loaded_program, self.profile.xlen())
            .map_err(|e| PiCcsError::InvalidInput(format!("memory layout derivation failed: {e}")))?;
        if matches!(self.output_target, OutputTarget::Ram) {
            memory_layout = extend_layout_with_guest_addresses(memory_layout, self.ram_init.keys().copied())?;
            memory_layout = extend_layout_with_guest_addresses(memory_layout, self.output_claims.claimed_addresses())?;
        }

        let logical_output_claims = match self.output_target {
            OutputTarget::Ram => memory_layout
                .remap_program_io(&self.output_claims)
                .map_err(|e| PiCcsError::InvalidInput(format!("output claim remap failed: {e}")))?,
            OutputTarget::Reg => self.output_claims.clone(),
            OutputTarget::RegExact => {
                let mut claims = ProgramIO::new();
                for (&reg, &value) in &self.exact_reg_output_words {
                    let lo = value & 0xffff_ffff;
                    let hi = value >> 32;
                    claims = claims.with_output(reg, F::from_u64(lo));
                    claims = claims.with_output(
                        reg.checked_add(32).ok_or_else(|| {
                            PiCcsError::InvalidInput(format!("exact reg output address overflow: reg={reg}"))
                        })?,
                        F::from_u64(hi),
                    );
                }
                claims
            }
        };
        let output_num_bits = match self.output_target {
            OutputTarget::Ram => memory_layout.required_num_bits(),
            OutputTarget::Reg => 5,
            OutputTarget::RegExact => 6,
        };

        let mut ram_init_words = segment_backed_ram_words(&self.loaded_program, self.profile.xlen());
        for (&addr, &value) in &self.ram_init {
            ram_init_words.insert(addr, value);
        }

        Ok(Rv64PreparedProgram {
            profile: self.profile.clone(),
            memory_layout,
            lowered_program,
            entry_segment,
            ram_init_words,
            reg_init_words: self.reg_init.clone(),
            guest_output_claims: self.output_claims.clone(),
            logical_output_claims,
            output_num_bits,
            output_target: self.output_target,
        })
    }

    pub fn simulate(&self) -> Result<VmTrace<u64, u64, u128>, PiCcsError> {
        let prepared = self.prepare()?;
        prepared.simulate(self.max_steps.unwrap_or(DEFAULT_RV64_TRACE_MAX_STEPS))
    }

    pub fn prove(self) -> Result<Rv64TraceWiringRun, PiCcsError> {
        let prepared = self.prepare()?;
        let program = prepared.program_instructions()?;
        let shout_ops = validate_rv64_trace_proving_subset(&program)?;
        let max_steps = self.max_steps.unwrap_or(DEFAULT_RV64_TRACE_MAX_STEPS);
        if max_steps == 0 {
            return Err(PiCcsError::InvalidInput("max_steps must be non-zero".into()));
        }

        let prove_start = time_now();
        let setup_start = prove_start;
        let mut trace = prepared.simulate(max_steps)?;
        validate_trace_opcode_lookup_one_hot(&trace, prepared.profile.xlen())?;
        let target_len = trace.steps.len().max(self.min_trace_len);
        let (prog_layout, prog_init_words) =
            prog_rom_layout_and_init_words::<F>(PROG_ID, prepared.entry_segment.vaddr, &prepared.entry_segment.data)
                .map_err(|e| PiCcsError::InvalidInput(format!("prog_rom_layout_and_init_words failed: {e}")))?;
        inject_decode_lookup_events_into_trace(&mut trace, &prog_layout, &prog_init_words)?;

        let exec = RiscvExecTable::from_trace_padded_with_xlen(&trace, target_len, /*machine_xlen=*/ 64)
            .map_err(|e| PiCcsError::InvalidInput(format!("RiscvExecTable::from_trace_padded failed: {e}")))?;
        exec.validate_cycle_chain()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_cycle_chain failed: {e}")))?;
        exec.validate_pc_chain()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_pc_chain failed: {e}")))?;
        exec.validate_halted_tail()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_halted_tail failed: {e}")))?;
        exec.validate_inactive_rows_are_empty()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_inactive_rows_are_empty failed: {e}")))?;
        validate_rv64_trace_field_injectivity(&exec, &prepared)?;
        let width_layout = Rv64WidthSidecarLayout::new();
        let include_width_lookup = rv64_program_requires_width_lookup(&program);
        let (rv64_width_lookup_tables, rv64_width_lookup_addr_d) = if include_width_lookup {
            let (tables, addr_d) = build_rv64_width_lookup_tables(&width_layout, &exec, trace.steps.len())?;
            inject_rv64_width_lookup_events_into_trace(&mut trace, &exec, &width_layout)?;
            (tables, addr_d)
        } else {
            (HashMap::new(), 0usize)
        };
        let rv64_ram_bridge = if rv64_program_requires_ram_sidecar(&program) || !prepared.ram_init_words.is_empty() {
            derive_rv64_ram_bridge(
                &trace,
                prepared.profile.xlen(),
                &prepared.guest_output_claims,
                &prepared.ram_init_words,
            )?
        } else {
            None
        };

        let requested_chunk_rows = self
            .chunk_rows
            .unwrap_or(DEFAULT_RV64_TRACE_CHUNK_ROWS)
            .max(max_consecutive_pc_run(&exec));
        let base_step_rows = requested_chunk_rows.min(exec.rows.len().max(1));
        let mut step_rows = base_step_rows;
        while step_rows < exec.rows.len() && boundary_splits_virtual_sequence(&exec, step_rows) {
            step_rows = step_rows
                .checked_add(base_step_rows)
                .ok_or_else(|| PiCcsError::InvalidInput(format!("trace chunk_rows overflow: {step_rows}")))?;
        }
        step_rows = step_rows.min(exec.rows.len().max(1));
        let exec_chunks = split_exec_into_fixed_chunks(&exec, step_rows)?;

        let mut layout = Rv64TraceCcsLayout::new_uniform(step_rows)
            .map_err(|e| PiCcsError::InvalidInput(format!("Rv64TraceCcsLayout::new_uniform failed: {e}")))?;

        let mut max_reg_addr = trace
            .steps
            .iter()
            .flat_map(|step| step.twist_events.iter())
            .filter(|event| event.twist_id == REG_ID)
            .map(|event| event.addr)
            .max()
            .unwrap_or(31);
        if let Some(max_init_reg_addr) = prepared.reg_init_words.keys().copied().max() {
            max_reg_addr = max_reg_addr.max(max_init_reg_addr);
        }
        let reg_d = required_bits_for_max_addr(max_reg_addr).max(5);
        let reg_k = 1usize
            .checked_shl(reg_d as u32)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("RV64 REG address width too large: d={reg_d}")))?;
        let reg_exact_d = 6usize;
        let reg_exact_k = 1usize << reg_exact_d;
        let mut mem_layouts: HashMap<u32, PlainMemLayout> = HashMap::from([
            (
                REG_ID.0,
                PlainMemLayout {
                    k: reg_k,
                    d: reg_d,
                    n_side: 2,
                    lanes: 2,
                },
            ),
            (PROG_ID.0, prog_layout.clone()),
        ]);
        if prepared.output_target == OutputTarget::RegExact && !prepared.logical_output_claims.is_empty() {
            mem_layouts.insert(
                REG_EXACT_ID.0,
                PlainMemLayout {
                    k: reg_exact_k,
                    d: reg_exact_d,
                    n_side: 2,
                    lanes: 2,
                },
            );
        }
        let include_ram_sidecar = rv64_program_requires_ram_sidecar(&program) || !prepared.ram_init_words.is_empty();
        if include_ram_sidecar {
            let bridge = rv64_ram_bridge.as_ref().ok_or_else(|| {
                PiCcsError::InvalidInput("RV64 RAM sidecar enabled but no affine RAM bridge was derived".into())
            })?;
            let ram_d = bridge.output_num_bits.max(2);
            let ram_k = 1usize
                .checked_shl(ram_d as u32)
                .ok_or_else(|| PiCcsError::InvalidInput(format!("RAM address width too large: d={ram_d}")))?;
            mem_layouts.insert(
                RAM_ID.0,
                PlainMemLayout {
                    k: ram_k,
                    d: ram_d,
                    n_side: 2,
                    lanes: 1,
                },
            );
        }

        let decode_lookup_tables = build_decode_lookup_tables(&prog_layout, &prog_init_words);
        let decode_layout = Rv32DecodeSidecarLayout::new();
        let decode_lookup_cols = riscv_decode_lookup_transport_cols(&decode_layout);
        let mut decode_table_ids: Vec<u32> = decode_lookup_cols
            .iter()
            .map(|&col_id| riscv_decode_lookup_table_id_for_col(col_id))
            .collect();
        decode_table_ids.sort_unstable();
        decode_table_ids.dedup();
        let mut shout_bus_specs = Vec::new();
        for &table_id in rv64_trace_table_specs(&shout_ops).keys() {
            shout_bus_specs.push(TraceShoutBusSpec {
                table_id,
                ell_addr: RV64_OPCODE_ELL_ADDR,
                n_vals: 1,
            });
        }
        if decode_table_ids.len() == 1 {
            shout_bus_specs.push(TraceShoutBusSpec {
                table_id: decode_table_ids[0],
                ell_addr: prog_layout.d,
                n_vals: decode_lookup_cols.len().max(1),
            });
        } else {
            for &col_id in decode_lookup_cols.iter() {
                shout_bus_specs.push(TraceShoutBusSpec {
                    table_id: riscv_decode_lookup_table_id_for_col(col_id),
                    ell_addr: prog_layout.d,
                    n_vals: 1,
                });
            }
        }
        if include_width_lookup {
            let width_lookup_cols = rv64_width_lookup_backed_cols(&width_layout);
            let mut width_table_ids: Vec<u32> = width_lookup_cols
                .iter()
                .map(|&col_id| rv64_width_lookup_table_id_for_col(col_id))
                .collect();
            width_table_ids.sort_unstable();
            width_table_ids.dedup();
            if width_table_ids.len() == 1 {
                shout_bus_specs.push(TraceShoutBusSpec {
                    table_id: width_table_ids[0],
                    ell_addr: rv64_width_lookup_addr_d,
                    n_vals: width_lookup_cols.len().max(1),
                });
            } else {
                for &col_id in &width_lookup_cols {
                    shout_bus_specs.push(TraceShoutBusSpec {
                        table_id: rv64_width_lookup_table_id_for_col(col_id),
                        ell_addr: rv64_width_lookup_addr_d,
                        n_vals: 1,
                    });
                }
            }
        }

        let table_specs = rv64_trace_table_specs(&shout_ops);
        let mut lut_tables = decode_lookup_tables.clone();
        lut_tables.extend(rv64_width_lookup_tables.clone());
        let lut_lanes: HashMap<u32, usize> = HashMap::new();
        let bus_cols = estimate_route_a_bus_cols(layout.t, &table_specs, &lut_tables, &mem_layouts, &lut_lanes)?;
        layout.m = layout
            .m_in
            .checked_add(layout.trace.cols)
            .and_then(|v| v.checked_add(bus_cols))
            .ok_or_else(|| PiCcsError::InvalidInput("uniform m overflow".into()))?;

        let ccs = build_rv64_trace_wiring_ccs(&layout)
            .map_err(|e| PiCcsError::InvalidInput(format!("build_rv64_trace_wiring_ccs failed: {e}")))?;
        let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n.max(ccs.m))
            .map_err(|e| PiCcsError::InvalidInput(format!("NeoParams::goldilocks_auto_r1cs_ccs failed: {e}")))?;
        let m_commit = neo_memory::ajtai::commit_cols_for_ccs_m(ccs.m);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let pp = neo_ajtai::setup_par(&mut rng, D, params.kappa as usize, m_commit)
            .map_err(|e| PiCcsError::InvalidInput(format!("Ajtai setup failed: {e}")))?;
        let mut session = FoldingSession::new(self.mode.clone(), params, AjtaiSModule::new(Arc::new(pp)));
        session.set_step_linking(StepLinkingConfig::new(vec![(layout.pc_final, layout.pc0)]));

        let mut initial_mem: HashMap<(u32, u64), F> = HashMap::new();
        for ((mem_id, addr), value) in &prog_init_words {
            if *value != F::ZERO {
                initial_mem.insert((*mem_id, *addr), *value);
            }
        }
        for (&reg, &value) in &prepared.reg_init_words {
            let fv = field_from_u64_exact_transport(value);
            if fv != F::ZERO {
                initial_mem.insert((REG_ID.0, reg), fv);
            }
        }
        if prepared.output_target == OutputTarget::RegExact && !prepared.logical_output_claims.is_empty() {
            for (&reg, &value) in &prepared.reg_init_words {
                let lo = value & 0xffff_ffff;
                let hi = value >> 32;
                if lo != 0 {
                    initial_mem.insert((REG_EXACT_ID.0, reg), F::from_u64(lo));
                }
                if hi != 0 {
                    let hi_addr = reg.checked_add(32).ok_or_else(|| {
                        PiCcsError::InvalidInput(format!("exact reg init address overflow: reg={reg}"))
                    })?;
                    initial_mem.insert((REG_EXACT_ID.0, hi_addr), F::from_u64(hi));
                }
            }
        }
        let ram_init_words = rv64_ram_bridge
            .as_ref()
            .map(|bridge| &bridge.logical_ram_init_words)
            .unwrap_or(&prepared.ram_init_words);
        for (&addr, &value) in ram_init_words {
            let fv = field_from_u64_exact_transport(value);
            if fv != F::ZERO {
                initial_mem.insert((RAM_ID.0, addr), fv);
            }
        }

        let setup_duration = elapsed_duration(setup_start);
        let chunk_start = time_now();
        let mut lookup_addr_groups = HashMap::<u32, u64>::new();
        let mut lookup_selector_groups = HashMap::<u32, u64>::new();
        let all_table_ids: HashSet<u32> = table_specs
            .keys()
            .copied()
            .chain(lut_tables.keys().copied())
            .collect();
        for table_id in all_table_ids {
            let ell_addr = table_ell_addr_for_shared_bus(table_id, &table_specs, &lut_tables)?;
            if let Some(group) = trace_lookup_addr_group_for_table_shape(table_id, ell_addr) {
                lookup_addr_groups.insert(table_id, group);
            }
            if let Some(group) = riscv_trace_lookup_selector_group_for_table_id(table_id) {
                lookup_selector_groups.insert(table_id, group as u64);
            }
        }
        let mut chunk_build_commit_duration = elapsed_duration(chunk_start);

        let cpu = R1csCpu::new(
            ccs.clone(),
            session.params().clone(),
            session.committer().clone(),
            layout.m_in,
            &lut_tables,
            &table_specs,
            rv64_trace_chunk_to_witness(layout.clone()),
        )
        .map_err(|e| PiCcsError::InvalidInput(format!("R1csCpu::new failed: {e}")))?
        .with_lookup_sharing_groups(lookup_addr_groups, lookup_selector_groups);

        let mut mem_addr_remaps = HashMap::new();
        if let Some(bridge) = &rv64_ram_bridge {
            mem_addr_remaps.insert(RAM_ID.0, bridge.remap.clone());
        }
        let mut trace_for_session = trace.clone();
        if prepared.output_target == OutputTarget::RegExact && !prepared.logical_output_claims.is_empty() {
            inject_exact_reg_writes_into_trace(&mut trace_for_session)?;
        }
        session.execute_shard_shared_cpu_bus_from_trace_with_mem_remaps(
            &trace_for_session,
            max_steps,
            layout.t,
            &mem_layouts,
            &lut_tables,
            &table_specs,
            &lut_lanes,
            &initial_mem,
            &mem_addr_remaps,
            &cpu,
        )?;
        if session.steps_witness().len() != exec_chunks.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "shared trace build drift: step bundle count {} != exec chunk count {}",
                session.steps_witness().len(),
                exec_chunks.len()
            )));
        }
        chunk_build_commit_duration += elapsed_duration(chunk_start);

        let mem_order = session
            .steps_public()
            .first()
            .map(|s| {
                s.mem_insts
                    .iter()
                    .map(|inst| inst.mem_id)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let reg_ob_mem_idx = mem_order.iter().position(|&id| id == REG_ID.0);
        let reg_exact_ob_mem_idx = mem_order.iter().position(|&id| id == REG_EXACT_ID.0);
        let ram_ob_mem_idx = mem_order.iter().position(|&id| id == RAM_ID.0);

        let fold_start = time_now();
        let (mut proof, output_binding_cfg, proof_memory_layout) = if prepared.output_target == OutputTarget::Ram
            && !prepared.logical_output_claims.is_empty()
        {
            let bridge = rv64_ram_bridge.as_ref().ok_or_else(|| {
                PiCcsError::InvalidInput("RV64 RAM output binding requires an affine RAM bridge".into())
            })?;
            let ram_mem_idx = ram_ob_mem_idx.ok_or_else(|| {
                PiCcsError::ProtocolError("missing RAM mem instance for RV64 RAM output binding".into())
            })?;
            let ram_output_num_bits = mem_layouts
                .get(&RAM_ID.0)
                .map(|layout| layout.d)
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("missing RAM mem layout for RV64 RAM output binding".into())
                })?;
            bridge
                .logical_output_claims
                .validate(ram_output_num_bits)
                .map_err(|e| {
                    PiCcsError::InvalidInput(format!(
                        "RV64 RAM output binding invalid (num_bits={}): {e}",
                        ram_output_num_bits
                    ))
                })?;
            let final_ram_state = final_shared_mem_state_dense(
                session.shared_bus_aux(),
                RAM_ID.0,
                1usize << ram_output_num_bits,
                ram_output_num_bits,
            )?;
            for (addr, expected) in bridge.logical_output_claims.claims() {
                let got = final_ram_state.get(addr as usize).copied().ok_or_else(|| {
                    PiCcsError::ProtocolError(format!(
                        "RV64 RAM output binding final state missing logical addr={addr}"
                    ))
                })?;
                if got != expected {
                    return Err(PiCcsError::ProtocolError(format!(
                        "RV64 RAM output binding mismatch at logical addr={addr}: final_state={got:?}, expected={expected:?}"
                    )));
                }
            }
            let ob_cfg = OutputBindingConfig::new(ram_output_num_bits, bridge.logical_output_claims.clone())
                .with_mem_idx(ram_mem_idx);
            let proof = session.fold_and_prove_with_output_binding_simple(&ccs, &ob_cfg, &final_ram_state)?;
            (proof, Some(ob_cfg), bridge.memory_layout.clone())
        } else if prepared.logical_output_claims.is_empty() {
            (session.fold_and_prove(&ccs)?, None, prepared.memory_layout.clone())
        } else if prepared.output_target == OutputTarget::Reg {
            prepared
                .logical_output_claims
                .validate(reg_d)
                .map_err(|e| {
                    PiCcsError::InvalidInput(format!("RV64 register output binding invalid (num_bits={reg_d}): {e}"))
                })?;
            let final_reg_state = final_reg_state_dense_injective(&exec, &prepared.reg_init_words, reg_k)?;
            let ob_cfg = OutputBindingConfig::new(reg_d, prepared.logical_output_claims.clone()).with_mem_idx(
                reg_ob_mem_idx
                    .ok_or_else(|| PiCcsError::ProtocolError("missing REG mem instance for output binding".into()))?,
            );
            let proof = session.fold_and_prove_with_output_binding_simple(&ccs, &ob_cfg, &final_reg_state)?;
            (proof, Some(ob_cfg), prepared.memory_layout.clone())
        } else {
            prepared
                .logical_output_claims
                .validate(reg_exact_d)
                .map_err(|e| {
                    PiCcsError::InvalidInput(format!(
                        "RV64 exact register output binding invalid (num_bits={reg_exact_d}): {e}"
                    ))
                })?;
            let final_reg_exact_state =
                final_shared_mem_state_dense(session.shared_bus_aux(), REG_EXACT_ID.0, reg_exact_k, reg_exact_d)?;
            let ob_cfg = OutputBindingConfig::new(reg_exact_d, prepared.logical_output_claims.clone()).with_mem_idx(
                reg_exact_ob_mem_idx.ok_or_else(|| {
                    PiCcsError::ProtocolError("missing REG_EXACT mem instance for output binding".into())
                })?,
            );
            let proof = session.fold_and_prove_with_output_binding_simple(&ccs, &ob_cfg, &final_reg_exact_state)?;
            (proof, Some(ob_cfg), prepared.memory_layout.clone())
        };
        proof.riscv_profile = Some(prepared.profile.config().clone());
        proof.riscv_memory_layout = Some(proof_memory_layout.clone());
        let fold_and_prove_duration = elapsed_duration(fold_start);
        let prove_duration = elapsed_duration(prove_start);

        let mut used_mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
        used_mem_ids.sort_unstable();
        let mut used_shout_table_ids: Vec<u32> = table_specs.keys().copied().collect();
        for spec in &shout_bus_specs {
            if !used_shout_table_ids.contains(&spec.table_id) {
                used_shout_table_ids.push(spec.table_id);
            }
        }
        used_shout_table_ids.sort_unstable();

        Ok(Rv64TraceWiringRun {
            session,
            ccs,
            layout,
            exec,
            proof,
            used_mem_ids,
            used_shout_table_ids,
            output_binding_cfg,
            profile_config: prepared.profile.config().clone(),
            memory_layout: proof_memory_layout,
            prove_duration,
            prove_phase_durations: Rv64TraceProvePhaseDurations {
                setup: setup_duration,
                chunk_build_commit: chunk_build_commit_duration,
                fold_and_prove: fold_and_prove_duration,
            },
            verify_duration: None,
        })
    }
}

impl Rv64PreparedProgram {
    pub fn program_instruction_pairs(&self) -> Result<Vec<(u64, RiscvInstruction)>, PiCcsError> {
        let end = self
            .entry_segment
            .vaddr
            .saturating_add(self.entry_segment.data.len() as u64);
        let mut instructions = Vec::new();
        for (addr, lowered) in &self.lowered_program.instructions {
            if *addr < self.entry_segment.vaddr || *addr >= end {
                return Err(PiCcsError::InvalidInput(format!(
                    "RV64 foundation only supports lowered instructions inside the single entry segment (saw addr={addr:#x}, entry_segment=[{:#x}, {:#x}))",
                    self.entry_segment.vaddr,
                    end
                )));
            }
            let instruction = match lowered {
                LoweredInstruction::Passthrough(instruction) => instruction.clone(),
            };
            instructions.push((*addr, instruction));
        }
        if instructions.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "RV64 prepared program has no lowered instructions in the entry segment".into(),
            ));
        }
        Ok(instructions)
    }

    pub fn program_instructions(&self) -> Result<Vec<RiscvInstruction>, PiCcsError> {
        Ok(self
            .program_instruction_pairs()?
            .into_iter()
            .map(|(_, instruction)| instruction)
            .collect())
    }

    pub fn simulate(&self, max_steps: usize) -> Result<VmTrace<u64, u64, u128>, PiCcsError> {
        if max_steps == 0 {
            return Err(PiCcsError::InvalidInput(
                "RV64 simulate max_steps must be non-zero".into(),
            ));
        }
        let mut cpu = RiscvCpu::new(self.profile.xlen());
        cpu.load_program_sparse(self.entry_segment.vaddr, self.program_instruction_pairs()?);
        cpu.set_runtime_decomposition_enabled(true);

        let mut memory = RiscvMemory::with_program_in_twist(
            self.profile.xlen(),
            PROG_ID,
            self.entry_segment.vaddr,
            &self.entry_segment.data,
        );
        for (&addr, &value) in &self.ram_init_words {
            memory.store(RAM_ID, addr, value);
        }
        for (&reg, &value) in &self.reg_init_words {
            memory.store(REG_ID, reg, value);
        }
        let shout = RiscvShoutTables::new(self.profile.xlen());
        trace_program(cpu, memory, shout, max_steps)
            .map_err(|e| PiCcsError::InvalidInput(format!("RV64 trace_program failed: {e}")))
    }
}

impl Rv64TraceWiringRun {
    fn validate_profile_and_layout(&self) -> Result<(), PiCcsError> {
        let profile = RiscvProofProfile::new(self.profile_config.clone()).map_err(profile_err_to_piccs)?;
        if self.memory_layout.xlen != profile.xlen() {
            return Err(PiCcsError::ProtocolError(format!(
                "RV64 run memory layout xlen mismatch: layout.xlen={} profile.xlen={}",
                self.memory_layout.xlen,
                profile.xlen()
            )));
        }
        if self.memory_layout.remap_kind != self.profile_config.memory_layout_kind {
            return Err(PiCcsError::ProtocolError(format!(
                "RV64 run remap kind mismatch: layout={:?} profile={:?}",
                self.memory_layout.remap_kind, self.profile_config.memory_layout_kind
            )));
        }
        Ok(())
    }

    fn validate_proof_metadata(&self, proof: &ShardProof) -> Result<(), PiCcsError> {
        self.validate_profile_and_layout()?;

        let proof_profile = proof
            .riscv_profile
            .as_ref()
            .ok_or_else(|| PiCcsError::ProtocolError("RV64 proof missing riscv_profile metadata".into()))?;
        let proof_layout = proof
            .riscv_memory_layout
            .as_ref()
            .ok_or_else(|| PiCcsError::ProtocolError("RV64 proof missing riscv_memory_layout metadata".into()))?;

        let validated_profile = RiscvProofProfile::new(proof_profile.clone()).map_err(profile_err_to_piccs)?;
        if proof_profile != &self.profile_config {
            return Err(PiCcsError::ProtocolError(format!(
                "RV64 proof profile mismatch: proof={proof_profile:?} run={:?}",
                self.profile_config
            )));
        }
        if proof_layout != &self.memory_layout {
            return Err(PiCcsError::ProtocolError(format!(
                "RV64 proof memory layout mismatch: proof={proof_layout:?} run={:?}",
                self.memory_layout
            )));
        }
        if proof_layout.xlen != validated_profile.xlen() {
            return Err(PiCcsError::ProtocolError(format!(
                "RV64 proof memory layout xlen mismatch: layout.xlen={} profile.xlen={}",
                proof_layout.xlen,
                validated_profile.xlen()
            )));
        }
        if proof_layout.remap_kind != proof_profile.memory_layout_kind {
            return Err(PiCcsError::ProtocolError(format!(
                "RV64 proof remap kind mismatch: layout={:?} profile={:?}",
                proof_layout.remap_kind, proof_profile.memory_layout_kind
            )));
        }
        Ok(())
    }

    pub fn params(&self) -> &NeoParams {
        self.session.params()
    }

    pub fn committer(&self) -> &AjtaiSModule {
        self.session.committer()
    }

    pub fn ccs(&self) -> &CcsStructure<F> {
        &self.ccs
    }

    pub fn layout(&self) -> &Rv64TraceCcsLayout {
        &self.layout
    }

    pub fn exec_table(&self) -> &RiscvExecTable {
        &self.exec
    }

    pub fn proof(&self) -> &ShardProof {
        &self.proof
    }

    pub fn used_memory_ids(&self) -> &[u32] {
        &self.used_mem_ids
    }

    pub fn used_shout_table_ids(&self) -> &[u32] {
        &self.used_shout_table_ids
    }

    pub fn profile_config(&self) -> &RiscvProofProfileConfig {
        &self.profile_config
    }

    pub fn memory_layout(&self) -> &RiscvGuestMemoryLayout {
        &self.memory_layout
    }

    pub fn verify_proof(&self, proof: &ShardProof) -> Result<(), PiCcsError> {
        self.validate_proof_metadata(proof)?;
        let ok = match &self.output_binding_cfg {
            None => self.session.verify_collected(&self.ccs, proof)?,
            Some(cfg) => self
                .session
                .verify_with_output_binding_collected_simple(&self.ccs, proof, cfg)?,
        };
        if !ok {
            return Err(PiCcsError::ProtocolError("verification failed".into()));
        }
        Ok(())
    }

    pub fn verify(&mut self) -> Result<(), PiCcsError> {
        let verify_start = time_now();
        self.verify_proof(&self.proof)?;
        self.verify_duration = Some(elapsed_duration(verify_start));
        Ok(())
    }

    pub fn ccs_num_constraints(&self) -> usize {
        self.ccs.n
    }

    pub fn ccs_num_variables(&self) -> usize {
        self.ccs.m
    }

    pub fn trace_len(&self) -> usize {
        self.exec.rows.iter().filter(|r| r.active).count()
    }

    pub fn fold_count(&self) -> usize {
        self.proof
            .steps
            .iter()
            .map(|step| {
                step.compressed_substeps
                    .as_ref()
                    .map_or(1, |sub| sub.len() + 1)
            })
            .sum()
    }

    pub fn prove_duration(&self) -> Duration {
        self.prove_duration
    }

    pub fn prove_phase_durations(&self) -> Rv64TraceProvePhaseDurations {
        self.prove_phase_durations
    }

    pub fn verify_duration(&self) -> Option<Duration> {
        self.verify_duration
    }
}
