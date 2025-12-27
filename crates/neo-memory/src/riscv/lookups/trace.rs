use std::collections::{HashMap, HashSet};

use neo_vm_trace::VmTrace;
use p3_field::PrimeField64;

use crate::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};

use super::isa::RiscvOpcode;
use super::tables::RiscvLookupTable;

/// Configuration for trace-to-proof conversion.
#[derive(Clone, Debug)]
pub struct TraceToProofConfig {
    /// Word size in bits (32 or 64)
    pub xlen: usize,
    /// Memory layout parameters
    pub mem_layout: PlainMemLayout,
    /// Shout table for each opcode
    pub opcode_tables: HashMap<RiscvOpcode, LutTable<p3_goldilocks::Goldilocks>>,
}

impl Default for TraceToProofConfig {
    fn default() -> Self {
        Self {
            xlen: 32,
            mem_layout: PlainMemLayout {
                k: 16,
                d: 1,
                n_side: 256,
            },
            opcode_tables: HashMap::new(),
        }
    }
}

/// Convert a VmTrace to PlainMemTrace for Twist encoding.
///
/// This extracts all memory read/write events from the trace and formats them
/// for Neo's Twist (read/write memory) argument.
pub fn trace_to_plain_mem_trace<F: PrimeField64>(trace: &VmTrace<u64, u64>) -> PlainMemTrace<F> {
    let steps = trace.len();

    let mut has_read = vec![F::ZERO; steps];
    let mut has_write = vec![F::ZERO; steps];
    let mut read_addr = vec![0u64; steps];
    let mut write_addr = vec![0u64; steps];
    let mut read_val = vec![F::ZERO; steps];
    let mut write_val = vec![F::ZERO; steps];
    let mut inc_at_write_addr = vec![F::ZERO; steps];

    // Track memory state for increment calculation
    let mut mem_state: HashMap<u64, F> = HashMap::new();

    for (j, step) in trace.steps.iter().enumerate() {
        for event in &step.twist_events {
            match event.kind {
                neo_vm_trace::TwistOpKind::Read => {
                    has_read[j] = F::ONE;
                    read_addr[j] = event.addr;
                    read_val[j] = F::from_u64(event.value);
                }
                neo_vm_trace::TwistOpKind::Write => {
                    has_write[j] = F::ONE;
                    write_addr[j] = event.addr;
                    write_val[j] = F::from_u64(event.value);

                    // Calculate increment
                    let old_val = mem_state.get(&event.addr).copied().unwrap_or(F::ZERO);
                    let new_val = F::from_u64(event.value);
                    inc_at_write_addr[j] = new_val - old_val;
                    mem_state.insert(event.addr, new_val);
                }
            }
        }
    }

    PlainMemTrace {
        steps,
        has_read,
        has_write,
        read_addr,
        write_addr,
        read_val,
        write_val,
        inc_at_write_addr,
    }
}

/// Convert a VmTrace to PlainLutTrace for Shout encoding.
///
/// This extracts all lookup events from the trace and formats them
/// for Neo's Shout (read-only lookup) argument.
///
/// # Note
/// Currently assumes a single unified lookup table. For multiple opcode-specific
/// tables, use `trace_to_plain_lut_traces_by_opcode`.
pub fn trace_to_plain_lut_trace<F: PrimeField64>(trace: &VmTrace<u64, u64>) -> PlainLutTrace<F> {
    let steps = trace.len();

    let mut has_lookup = vec![F::ZERO; steps];
    let mut addr = vec![0u64; steps];
    let mut val = vec![F::ZERO; steps];

    for (j, step) in trace.steps.iter().enumerate() {
        // Take the first Shout event if any
        if let Some(event) = step.shout_events.first() {
            has_lookup[j] = F::ONE;
            addr[j] = event.key;
            val[j] = F::from_u64(event.value);
        }
    }

    PlainLutTrace { has_lookup, addr, val }
}

/// Convert a VmTrace to multiple PlainLutTraces, one per opcode/table.
///
/// This separates lookup events by their ShoutId, allowing different
/// opcodes to use different lookup tables.
pub fn trace_to_plain_lut_traces_by_opcode<F: PrimeField64>(
    trace: &VmTrace<u64, u64>,
    num_tables: usize,
) -> Vec<PlainLutTrace<F>> {
    let steps = trace.len();

    // Initialize a trace for each table
    let mut traces: Vec<PlainLutTrace<F>> = (0..num_tables)
        .map(|_| PlainLutTrace {
            has_lookup: vec![F::ZERO; steps],
            addr: vec![0u64; steps],
            val: vec![F::ZERO; steps],
        })
        .collect();

    for (j, step) in trace.steps.iter().enumerate() {
        for event in &step.shout_events {
            let table_id = event.shout_id.0 as usize;
            if table_id < num_tables {
                traces[table_id].has_lookup[j] = F::ONE;
                traces[table_id].addr[j] = event.key;
                traces[table_id].val[j] = F::from_u64(event.value);
            }
        }
    }

    traces
}

/// Build a lookup table for a specific RISC-V opcode.
///
/// This creates a `LutTable` that can be used with Neo's Shout encoding.
pub fn build_opcode_lut_table<F: PrimeField64>(table_id: u32, opcode: RiscvOpcode, xlen: usize) -> LutTable<F> {
    let table: RiscvLookupTable<F> = RiscvLookupTable::new(opcode, xlen);
    let size = table.size();
    let k = (size as f64).log2().ceil() as usize;

    LutTable {
        table_id,
        k,
        d: 1,
        n_side: size,
        content: table.content(),
    }
}

/// Summary of a trace conversion.
#[derive(Clone, Debug)]
pub struct TraceConversionSummary {
    /// Total steps in the trace
    pub total_steps: usize,
    /// Number of memory read operations
    pub num_reads: usize,
    /// Number of memory write operations
    pub num_writes: usize,
    /// Number of lookup operations
    pub num_lookups: usize,
    /// Unique memory addresses accessed
    pub unique_addresses: usize,
    /// Unique lookup keys used
    pub unique_lookup_keys: usize,
}

/// Extract final register values from a trace as a ProgramIO structure.
///
/// This creates a ProgramIO suitable for the Output Sumcheck, using RISC-V
/// register conventions (x10-x17 as return value registers).
pub fn extract_program_io<F: p3_field::PrimeField64>(
    trace: &VmTrace<u64, u64>,
    output_regs: &[usize],
) -> crate::output_check::ProgramIO<F> {
    // RISC-V ABI: x10-x17 (a0-a7) are argument/return registers
    // We map register x_i to virtual address i

    let mut program_io = crate::output_check::ProgramIO::new();

    if let Some(last_step) = trace.steps.last() {
        for &reg in output_regs {
            if reg < 32 {
                let val = last_step.regs_after[reg];
                program_io = program_io.with_output(reg as u64, F::from_u64(val));
            }
        }
    }

    program_io
}

/// Build a final memory state vector suitable for the Output Sumcheck.
///
/// This creates a sparse representation of the final register file state,
/// where virtual address i contains the value of register x_i.
pub fn build_final_memory_state<F: p3_field::PrimeField64>(trace: &VmTrace<u64, u64>, num_bits: usize) -> Vec<F> {
    let size = 1usize << num_bits;
    let mut state = vec![F::ZERO; size];

    if let Some(last_step) = trace.steps.last() {
        // Map registers to virtual addresses 0-31
        for (i, &val) in last_step.regs_after.iter().enumerate() {
            if i < size {
                state[i] = F::from_u64(val);
            }
        }
    }

    state
}

/// Analyze a trace and return a summary.
pub fn analyze_trace(trace: &VmTrace<u64, u64>) -> TraceConversionSummary {
    let mut num_reads = 0;
    let mut num_writes = 0;
    let mut num_lookups = 0;
    let mut addresses = HashSet::new();
    let mut lookup_keys = HashSet::new();

    for step in &trace.steps {
        for event in &step.twist_events {
            match event.kind {
                neo_vm_trace::TwistOpKind::Read => {
                    num_reads += 1;
                    addresses.insert(event.addr);
                }
                neo_vm_trace::TwistOpKind::Write => {
                    num_writes += 1;
                    addresses.insert(event.addr);
                }
            }
        }
        for event in &step.shout_events {
            num_lookups += 1;
            lookup_keys.insert(event.key);
        }
    }

    TraceConversionSummary {
        total_steps: trace.len(),
        num_reads,
        num_writes,
        num_lookups,
        unique_addresses: addresses.len(),
        unique_lookup_keys: lookup_keys.len(),
    }
}
