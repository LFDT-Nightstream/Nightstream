use super::ir::{LinearMemoryAccess, LinearMemoryWordLane, StackLaneAccess, WasmParamInitState, WasmStepTrace};
use super::isa::{WasmOpcode, WasmShoutOpcode};
use super::tables::{lookup_payload, WasmLookupArity};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WasmMemoryKind {
    Stack,
    LinearMemory,
    Locals,
    Globals,
    Tables,
    TableSizes,
    FunctionTypes,
    ModuleTypes,
    FunctionEntries,
    PcEdgeKinds,
    CallStack,
    PcRom,
}

impl WasmMemoryKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Stack => "stack",
            Self::LinearMemory => "linear_memory",
            Self::Locals => "locals",
            Self::Globals => "globals",
            Self::Tables => "tables",
            Self::TableSizes => "table_sizes",
            Self::FunctionTypes => "function_types",
            Self::ModuleTypes => "module_types",
            Self::FunctionEntries => "function_entries",
            Self::PcEdgeKinds => "pc_edge_kinds",
            Self::CallStack => "call_stack",
            Self::PcRom => "pc_rom",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmLookupRow {
    pub trace_index: usize,
    pub cycle: u64,
    pub pc_before: u64,
    pub shout_opcode: WasmShoutOpcode,
    pub shout_id: u32,
    pub arity: WasmLookupArity,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmMemoryEventKind {
    Read,
    Write,
    Init,
    Push,
    Pop,
    Rom,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmMemoryEvent {
    pub family: WasmMemoryKind,
    pub kind: WasmMemoryEventKind,
    pub trace_index: Option<usize>,
    pub cycle: u64,
    pub addr0: u64,
    pub addr1: u64,
    pub value0: u64,
    pub value1: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmBoundaryRow {
    pub trace_index: usize,
    pub cycle: u64,
    pub pc_before: u64,
    pub pc_after: u64,
    pub sp_before: u64,
    pub sp_after: u64,
    pub memory_pages_before: Option<u32>,
    pub memory_pages_after: Option<u32>,
    pub locals_fbp_before: u64,
    pub locals_fbp_after: u64,
    pub halted: bool,
    pub param_init_before: WasmParamInitState,
    pub param_init_after: WasmParamInitState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmRelationProof {
    pub lookup_rows: Vec<WasmLookupRow>,
    pub memory_events: Vec<WasmMemoryEvent>,
    pub boundary_rows: Vec<WasmBoundaryRow>,
    pub final_stack_slots: Vec<(u64, u32)>,
    pub final_local_slots: Vec<(u64, u32)>,
}

pub fn prove_wasm_relation(
    trace: &[WasmStepTrace],
    initial_locals: &[u32],
    pc_rom: &[(u64, u64, u64)],
    pc_edge_kinds: &[(u64, u64)],
    function_entries: &[(u64, u64)],
    _transcript_seed: &[u8],
) -> Result<WasmRelationProof, String> {
    validate_relation_trace(trace, pc_rom, pc_edge_kinds, function_entries)?;

    let lookup_rows = relation_lookup_rows(trace);
    let memory_events = relation_memory_events(trace, initial_locals, pc_rom, pc_edge_kinds, function_entries);
    let boundary_rows = relation_boundary_rows(trace);
    let final_stack_slots = final_stack_slots(trace);
    let final_local_slots = final_local_slots(trace, initial_locals);

    Ok(WasmRelationProof {
        lookup_rows,
        memory_events,
        boundary_rows,
        final_stack_slots,
        final_local_slots,
    })
}

pub fn verify_wasm_relation(
    trace: &[WasmStepTrace],
    initial_locals: &[u32],
    pc_rom: &[(u64, u64, u64)],
    pc_edge_kinds: &[(u64, u64)],
    function_entries: &[(u64, u64)],
    transcript_seed: &[u8],
    proof: &WasmRelationProof,
) -> Result<(), String> {
    let expected = prove_wasm_relation(
        trace,
        initial_locals,
        pc_rom,
        pc_edge_kinds,
        function_entries,
        transcript_seed,
    )?;
    if expected != *proof {
        return Err("wasm relation proof mismatch".into());
    }
    Ok(())
}

fn validate_relation_trace(
    trace: &[WasmStepTrace],
    pc_rom: &[(u64, u64, u64)],
    pc_edge_kinds: &[(u64, u64)],
    function_entries: &[(u64, u64)],
) -> Result<(), String> {
    validate_boundary_links(trace)?;
    let _ = pc_rom;
    validate_pc_edge_kinds(trace, pc_edge_kinds)?;
    validate_call_stack(trace)?;
    validate_call_indirect_targets(trace, function_entries)?;
    Ok(())
}

fn validate_boundary_links(trace: &[WasmStepTrace]) -> Result<(), String> {
    if trace.is_empty() {
        return Ok(());
    }
    if trace[0].param_init_before.active || trace[0].param_init_before.remaining != 0 {
        return Err("initial param-init state must be inactive".into());
    }
    for pair in trace.windows(2) {
        let prev = &pair[0];
        let next = &pair[1];
        if prev.halted {
            return Err(format!(
                "halted row at cycle {} is not the final boundary row",
                prev.cycle
            ));
        }
        if prev.pc_after != next.pc_before {
            return Err(format!(
                "pc continuity failed at cycles {} -> {}: {} != {}",
                prev.cycle, next.cycle, prev.pc_after, next.pc_before
            ));
        }
        if prev.sp_after != next.sp_before {
            return Err(format!(
                "sp continuity failed at cycles {} -> {}: {} != {}",
                prev.cycle, next.cycle, prev.sp_after, next.sp_before
            ));
        }
        if prev.memory_pages_after != next.memory_pages_before {
            return Err(format!(
                "memory page continuity failed at cycles {} -> {}: {:?} != {:?}",
                prev.cycle, next.cycle, prev.memory_pages_after, next.memory_pages_before
            ));
        }
        if prev.locals_fbp_after != next.locals_fbp {
            return Err(format!(
                "locals_fbp continuity failed at cycles {} -> {}: {} != {}",
                prev.cycle, next.cycle, prev.locals_fbp_after, next.locals_fbp
            ));
        }
        if prev.param_init_after != next.param_init_before {
            return Err(format!(
                "param-init continuity failed at cycles {} -> {}: {:?} != {:?}",
                prev.cycle, next.cycle, prev.param_init_after, next.param_init_before
            ));
        }
    }
    if !trace.last().is_some_and(|row| row.halted) {
        return Err("final boundary row must be halted".into());
    }
    if trace
        .last()
        .is_some_and(|row| row.param_init_after.active || row.param_init_after.remaining != 0)
    {
        return Err("final param-init state must be inactive".into());
    }
    Ok(())
}

fn validate_call_stack(trace: &[WasmStepTrace]) -> Result<(), String> {
    let mut stack = Vec::new();
    for row in trace {
        if let Some(push) = row.call_stack_push {
            stack.push(push);
        }
        if let Some(pop) = row.call_stack_pop {
            let expected = stack
                .pop()
                .ok_or_else(|| format!("call stack underflow at cycle {}", row.cycle))?;
            if expected != pop {
                return Err(format!(
                    "call stack mismatch at cycle {}: expected ({}, {}), got ({}, {})",
                    row.cycle, expected.0, expected.1, pop.0, pop.1
                ));
            }
            if row.pc_after != pop.0 {
                return Err(format!(
                    "return pc mismatch at cycle {}: expected pc_after {}, got {}",
                    row.cycle, pop.0, row.pc_after
                ));
            }
        }
    }
    if !stack.is_empty() {
        return Err("call stack not empty at end of relation".into());
    }
    Ok(())
}

fn validate_pc_edge_kinds(trace: &[WasmStepTrace], pc_edge_kinds: &[(u64, u64)]) -> Result<(), String> {
    let edge_kinds = pc_edge_kinds.iter().copied().collect::<BTreeMap<_, _>>();
    for row in trace {
        if !row.row_kind.is_program() {
            continue;
        }
        let expected = edge_kinds.get(&row.pc_before).copied().ok_or_else(|| {
            format!(
                "pc_edge_kinds missing entry for pc {} at cycle {}",
                row.pc_before, row.cycle
            )
        })?;
        let actual = u64::from(row.pc_edge_kind.as_u32());
        if actual != expected {
            return Err(format!(
                "pc_edge_kind mismatch at cycle {}: expected {}, got {}",
                row.cycle, expected, actual
            ));
        }
    }
    Ok(())
}

fn validate_call_indirect_targets(trace: &[WasmStepTrace], function_entries: &[(u64, u64)]) -> Result<(), String> {
    let entries = function_entries.iter().copied().collect::<BTreeMap<_, _>>();
    for row in trace {
        if row.opcode != WasmOpcode::CallIndirect {
            continue;
        }
        let function_ref = row
            .table_value
            .ok_or_else(|| format!("call_indirect missing table_value at cycle {}", row.cycle))?;
        let expected_pc = entries
            .get(&u64::from(function_ref))
            .copied()
            .ok_or_else(|| {
                format!(
                    "function_entries missing target for call_indirect ref {} at cycle {}",
                    function_ref, row.cycle
                )
            })?;
        if row.pc_after != expected_pc {
            return Err(format!(
                "call_indirect target mismatch at cycle {}: expected pc_after {}, got {}",
                row.cycle, expected_pc, row.pc_after
            ));
        }
    }
    Ok(())
}

fn relation_lookup_rows(trace: &[WasmStepTrace]) -> Vec<WasmLookupRow> {
    trace
        .iter()
        .enumerate()
        .filter_map(|(trace_index, step)| {
            let shout_opcode = step.info.shout_opcode?;
            let payload = lookup_payload(step)?;
            Some(WasmLookupRow {
                trace_index,
                cycle: step.cycle,
                pc_before: step.pc_before,
                shout_opcode,
                shout_id: payload.shout_id,
                arity: payload.arity,
                inputs: payload.inputs,
                outputs: payload.outputs,
            })
        })
        .collect()
}

fn relation_boundary_rows(trace: &[WasmStepTrace]) -> Vec<WasmBoundaryRow> {
    trace
        .iter()
        .enumerate()
        .map(|(trace_index, step)| WasmBoundaryRow {
            trace_index,
            cycle: step.cycle,
            pc_before: step.pc_before,
            pc_after: step.pc_after,
            sp_before: step.sp_before,
            sp_after: step.sp_after,
            memory_pages_before: step.memory_pages_before,
            memory_pages_after: step.memory_pages_after,
            locals_fbp_before: step.locals_fbp,
            locals_fbp_after: step.locals_fbp_after,
            halted: step.halted,
            param_init_before: step.param_init_before,
            param_init_after: step.param_init_after,
        })
        .collect()
}

fn local_access(step: &WasmStepTrace, value: Option<u32>) -> Option<StackLaneAccess> {
    match (step.local_index, value) {
        (Some(idx), Some(value)) => Some(StackLaneAccess {
            addr: step.locals_fbp + u64::from(idx),
            value,
        }),
        _ => None,
    }
}

fn global_access(step: &WasmStepTrace, value: Option<u32>) -> Option<StackLaneAccess> {
    match (step.global_index, value) {
        (Some(idx), Some(value)) => Some(StackLaneAccess {
            addr: u64::from(idx),
            value,
        }),
        _ => None,
    }
}

pub(crate) fn relation_memory_events(
    trace: &[WasmStepTrace],
    initial_locals: &[u32],
    pc_rom: &[(u64, u64, u64)],
    pc_edge_kinds: &[(u64, u64)],
    function_entries: &[(u64, u64)],
) -> Vec<WasmMemoryEvent> {
    let mut events = Vec::new();
    let function_entry_map = function_entries.iter().copied().collect::<BTreeMap<_, _>>();

    for (addr, &value) in initial_locals.iter().enumerate() {
        events.push(WasmMemoryEvent {
            family: WasmMemoryKind::Locals,
            kind: WasmMemoryEventKind::Init,
            trace_index: None,
            cycle: 0,
            addr0: addr as u64,
            addr1: 0,
            value0: u64::from(value),
            value1: 0,
        });
    }

    for &(pc_before, control_choice, pc_after) in pc_rom {
        events.push(WasmMemoryEvent {
            family: WasmMemoryKind::PcRom,
            kind: WasmMemoryEventKind::Rom,
            trace_index: None,
            cycle: 0,
            addr0: pc_before,
            addr1: control_choice,
            value0: pc_after,
            value1: 0,
        });
    }
    for &(pc_before, edge_kind) in pc_edge_kinds {
        events.push(WasmMemoryEvent {
            family: WasmMemoryKind::PcEdgeKinds,
            kind: WasmMemoryEventKind::Rom,
            trace_index: None,
            cycle: 0,
            addr0: pc_before,
            addr1: 0,
            value0: edge_kind,
            value1: 0,
        });
    }

    for (idx, step) in trace.iter().enumerate() {
        let trace_index = Some(idx);
        let cycle = step.cycle;
        append_stack_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Read,
            step.stack_read0,
        );
        append_stack_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Read,
            step.stack_read1,
        );
        append_stack_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Read,
            step.stack_read2,
        );
        append_stack_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Write,
            step.stack_write0,
        );
        append_linear_memory_event(
            &mut events,
            trace_index,
            cycle,
            step.linear_memory,
            step.stack_write0.is_some(),
        );
        append_locals_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Read,
            local_access(step, step.local_read_value),
        );
        append_locals_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Write,
            local_access(step, step.local_write_value),
        );
        append_globals_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Read,
            global_access(step, step.global_read_value),
        );
        append_globals_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Write,
            global_access(step, step.global_write_value),
        );
        append_table_size_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Read,
            step.table_id,
            step.table_size,
        );
        append_table_size_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Write,
            step.table_id,
            step.table_size,
        );
        append_function_type_event(&mut events, trace_index, cycle, step.table_value, step.function_type_id);
        append_module_type_event(
            &mut events,
            trace_index,
            cycle,
            step.call_indirect_type_index,
            step.expected_type_id,
        );
        append_function_entry_event(
            &mut events,
            trace_index,
            cycle,
            step.opcode,
            step.table_value,
            &function_entry_map,
        );
        append_table_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Read,
            step.table_id,
            step.table_index,
            step.table_value,
        );
        append_table_event(
            &mut events,
            trace_index,
            cycle,
            WasmMemoryEventKind::Write,
            step.table_id,
            step.table_index,
            step.table_value,
        );

        if let Some((return_pc, caller_fbp)) = step.call_stack_push {
            events.push(WasmMemoryEvent {
                family: WasmMemoryKind::CallStack,
                kind: WasmMemoryEventKind::Push,
                trace_index,
                cycle,
                addr0: cycle,
                addr1: 0,
                value0: return_pc,
                value1: caller_fbp,
            });
        }
        if let Some((return_pc, caller_fbp)) = step.call_stack_pop {
            events.push(WasmMemoryEvent {
                family: WasmMemoryKind::CallStack,
                kind: WasmMemoryEventKind::Pop,
                trace_index,
                cycle,
                addr0: cycle,
                addr1: 0,
                value0: return_pc,
                value1: caller_fbp,
            });
        }
    }

    events
}

fn append_linear_memory_event(
    events: &mut Vec<WasmMemoryEvent>,
    trace_index: Option<usize>,
    cycle: u64,
    access: Option<LinearMemoryAccess>,
    is_load: bool,
) {
    if let Some(access) = access {
        append_linear_memory_lane_event(events, trace_index, cycle, access.lane0, is_load);
        if let Some(lane1) = access.lane1 {
            append_linear_memory_lane_event(events, trace_index, cycle, lane1, is_load);
        }
        if let Some(lane2) = access.lane2 {
            append_linear_memory_lane_event(events, trace_index, cycle, lane2, is_load);
        }
    }
}

fn append_linear_memory_lane_event(
    events: &mut Vec<WasmMemoryEvent>,
    trace_index: Option<usize>,
    cycle: u64,
    lane: LinearMemoryWordLane,
    is_load: bool,
) {
    events.push(WasmMemoryEvent {
        family: WasmMemoryKind::LinearMemory,
        kind: if is_load {
            WasmMemoryEventKind::Read
        } else {
            WasmMemoryEventKind::Write
        },
        trace_index,
        cycle,
        addr0: lane.word_addr,
        addr1: 0,
        value0: u64::from(lane.value_before),
        value1: u64::from(lane.value_after),
    });
}

fn append_stack_event(
    events: &mut Vec<WasmMemoryEvent>,
    trace_index: Option<usize>,
    cycle: u64,
    kind: WasmMemoryEventKind,
    lane: Option<StackLaneAccess>,
) {
    if let Some(lane) = lane {
        events.push(WasmMemoryEvent {
            family: WasmMemoryKind::Stack,
            kind,
            trace_index,
            cycle,
            addr0: lane.addr,
            addr1: 0,
            value0: u64::from(lane.value),
            value1: 0,
        });
    }
}

fn append_locals_event(
    events: &mut Vec<WasmMemoryEvent>,
    trace_index: Option<usize>,
    cycle: u64,
    kind: WasmMemoryEventKind,
    lane: Option<StackLaneAccess>,
) {
    if let Some(lane) = lane {
        events.push(WasmMemoryEvent {
            family: WasmMemoryKind::Locals,
            kind,
            trace_index,
            cycle,
            addr0: lane.addr,
            addr1: 0,
            value0: u64::from(lane.value),
            value1: 0,
        });
    }
}

fn append_globals_event(
    events: &mut Vec<WasmMemoryEvent>,
    trace_index: Option<usize>,
    cycle: u64,
    kind: WasmMemoryEventKind,
    lane: Option<StackLaneAccess>,
) {
    if let Some(lane) = lane {
        events.push(WasmMemoryEvent {
            family: WasmMemoryKind::Globals,
            kind,
            trace_index,
            cycle,
            addr0: lane.addr,
            addr1: 0,
            value0: u64::from(lane.value),
            value1: 0,
        });
    }
}

fn append_table_event(
    events: &mut Vec<WasmMemoryEvent>,
    trace_index: Option<usize>,
    cycle: u64,
    kind: WasmMemoryEventKind,
    table_id: Option<u32>,
    table_index: Option<u32>,
    table_value: Option<u32>,
) {
    if let (Some(table_id), Some(table_index), Some(table_value)) = (table_id, table_index, table_value) {
        events.push(WasmMemoryEvent {
            family: WasmMemoryKind::Tables,
            kind,
            trace_index,
            cycle,
            addr0: u64::from(table_id),
            addr1: u64::from(table_index),
            value0: u64::from(table_value),
            value1: 0,
        });
    }
}

fn append_table_size_event(
    events: &mut Vec<WasmMemoryEvent>,
    trace_index: Option<usize>,
    cycle: u64,
    kind: WasmMemoryEventKind,
    table_id: Option<u32>,
    table_size: Option<u32>,
) {
    if let (Some(table_id), Some(table_size)) = (table_id, table_size) {
        events.push(WasmMemoryEvent {
            family: WasmMemoryKind::TableSizes,
            kind,
            trace_index,
            cycle,
            addr0: u64::from(table_id),
            addr1: 0,
            value0: u64::from(table_size),
            value1: 0,
        });
    }
}

fn append_function_type_event(
    events: &mut Vec<WasmMemoryEvent>,
    trace_index: Option<usize>,
    cycle: u64,
    function_ref: Option<u32>,
    function_type_id: Option<u32>,
) {
    if let (Some(function_ref), Some(function_type_id)) = (function_ref, function_type_id) {
        events.push(WasmMemoryEvent {
            family: WasmMemoryKind::FunctionTypes,
            kind: WasmMemoryEventKind::Rom,
            trace_index,
            cycle,
            addr0: u64::from(function_ref),
            addr1: 0,
            value0: u64::from(function_type_id),
            value1: 0,
        });
    }
}

fn append_module_type_event(
    events: &mut Vec<WasmMemoryEvent>,
    trace_index: Option<usize>,
    cycle: u64,
    raw_type_index: Option<u32>,
    expected_type_id: Option<u32>,
) {
    if let (Some(raw_type_index), Some(expected_type_id)) = (raw_type_index, expected_type_id) {
        events.push(WasmMemoryEvent {
            family: WasmMemoryKind::ModuleTypes,
            kind: WasmMemoryEventKind::Rom,
            trace_index,
            cycle,
            addr0: u64::from(raw_type_index),
            addr1: 0,
            value0: u64::from(expected_type_id),
            value1: 0,
        });
    }
}

fn append_function_entry_event(
    events: &mut Vec<WasmMemoryEvent>,
    trace_index: Option<usize>,
    cycle: u64,
    opcode: WasmOpcode,
    function_ref: Option<u32>,
    function_entries: &BTreeMap<u64, u64>,
) {
    if opcode != WasmOpcode::CallIndirect {
        return;
    }
    if let Some(function_ref) = function_ref {
        let Some(entry_pc) = function_entries.get(&u64::from(function_ref)).copied() else {
            return;
        };
        events.push(WasmMemoryEvent {
            family: WasmMemoryKind::FunctionEntries,
            kind: WasmMemoryEventKind::Rom,
            trace_index,
            cycle,
            addr0: u64::from(function_ref),
            addr1: 0,
            value0: entry_pc,
            value1: 0,
        });
    }
}

fn final_stack_slots(trace: &[WasmStepTrace]) -> Vec<(u64, u32)> {
    let mut slots = BTreeMap::new();
    for step in trace {
        if let Some(write) = step.stack_write0 {
            slots.insert(write.addr, write.value);
        }
    }
    slots.into_iter().collect()
}

fn final_local_slots(trace: &[WasmStepTrace], initial_locals: &[u32]) -> Vec<(u64, u32)> {
    let mut slots = BTreeMap::new();
    for (addr, &value) in initial_locals.iter().enumerate() {
        slots.insert(addr as u64, value);
    }
    for step in trace {
        if let Some(access) = local_access(step, step.local_write_value) {
            slots.insert(access.addr, access.value);
        }
    }
    slots.into_iter().collect()
}
