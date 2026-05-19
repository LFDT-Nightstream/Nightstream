use super::isa::{WasmOpcode, WasmOpcodeInfo};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmPcEdgeKind {
    Static = 0,
    ReturnLike = 1,
    DynamicCallIndirect = 2,
    Terminal = 3,
}

impl WasmPcEdgeKind {
    pub const STATIC_U64: u64 = Self::Static as u64;

    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StackLaneAccess {
    pub addr: u64,
    pub value: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinearMemoryWordLane {
    pub word_addr: u64,
    pub value_before: u32,
    pub value_after: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinearMemoryAccess {
    pub width_bytes: u8,
    pub byte_offset: u8,
    pub lane0: LinearMemoryWordLane,
    pub lane1: Option<LinearMemoryWordLane>,
    pub lane2: Option<LinearMemoryWordLane>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmParamInitState {
    pub active: bool,
    pub remaining: u32,
}

impl WasmParamInitState {
    pub const ZERO: Self = Self {
        active: false,
        remaining: 0,
    };
}

impl Default for WasmParamInitState {
    fn default() -> Self {
        Self::ZERO
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmBoundaryState {
    pub pc: u64,
    pub sp: u64,
    pub memory_pages: Option<u32>,
    pub locals_fbp: u64,
    pub halted: bool,
    pub param_init: WasmParamInitState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmAuxOpcode {
    CallParamInit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmRowKind {
    Program,
    Aux(WasmAuxOpcode),
}

impl WasmRowKind {
    pub fn is_program(self) -> bool {
        matches!(self, Self::Program)
    }

    pub fn is_call_param_init(self) -> bool {
        matches!(self, Self::Aux(WasmAuxOpcode::CallParamInit))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WasmStepTrace {
    pub cycle: u64,
    pub row_kind: WasmRowKind,
    pub pc_before: u64,
    pub pc_after: u64,
    pub control_choice: u32,
    pub pc_edge_kind: WasmPcEdgeKind,
    pub param_init_before: WasmParamInitState,
    pub param_init_after: WasmParamInitState,
    pub wide_values_enabled: bool,
    pub opcode_code: u16,
    pub opcode: WasmOpcode,
    pub info: WasmOpcodeInfo,
    pub stack_reads_override: Option<u8>,
    pub stack_writes_override: Option<u8>,
    pub sp_before: u64,
    pub sp_after: u64,
    /// Normalized function reference for the currently executing frame.
    /// `function_ref` below is opcode-target metadata, not this frame identity.
    pub current_function_ref: u32,
    pub current_function_num_locals: u32,
    pub stack_read0: Option<StackLaneAccess>,
    pub stack_read0_hi: Option<u32>,
    pub stack_read1: Option<StackLaneAccess>,
    pub stack_read1_hi: Option<u32>,
    pub stack_read2: Option<StackLaneAccess>,
    pub stack_read2_hi: Option<u32>,
    pub stack_write0: Option<StackLaneAccess>,
    pub stack_write0_hi: Option<u32>,
    pub linear_memory: Option<LinearMemoryAccess>,
    pub linear_memory_offset: u64,
    pub memory_pages_before: Option<u32>,
    pub memory_pages_after: Option<u32>,
    pub halted: bool,
    /// Frame base pointer for locals before this row. Absolute local address is
    /// `locals_fbp + local_index`.
    pub locals_fbp: u64,
    /// Frame base pointer after this row. Equal to `locals_fbp` except across
    /// traced guest calls and non-final returns.
    pub locals_fbp_after: u64,
    /// Index of the local variable accessed (for local.get / local.set / local.tee).
    pub local_index: Option<u32>,
    /// Value of the local before this step (populated for local.get: the value pushed).
    pub local_read_value: Option<u32>,
    pub local_read_value_hi: Option<u32>,
    /// Value written into the local this step (populated for local.set / local.tee).
    pub local_write_value: Option<u32>,
    pub local_write_value_hi: Option<u32>,
    /// Index of the global variable accessed (for global.get / global.set).
    pub global_index: Option<u32>,
    /// Value of the global before this step (populated for global.get).
    pub global_read_value: Option<u32>,
    pub global_read_value_hi: Option<u32>,
    /// Value written into the global this step (populated for global.set).
    pub global_write_value: Option<u32>,
    pub global_write_value_hi: Option<u32>,
    /// Referenced table id for table-state opcodes and indirect table consumers.
    pub table_id: Option<u32>,
    /// Referenced element index within the table contents namespace.
    pub table_index: Option<u32>,
    /// Normalized table element value observed by this step. For `call_indirect`, the Wasmtime
    /// adapter is expected to populate this with the selected funcref id so the relation layer can
    /// validate `pc_after` against the static `function_entries` ROM.
    pub table_value: Option<u32>,
    /// Normalized deduplicated type id for the observed function reference.
    pub function_type_id: Option<u32>,
    /// Raw module type-section index from the `call_indirect` immediate.
    pub call_indirect_type_index: Option<u32>,
    /// Normalized deduplicated type id expected by the current opcode.
    pub expected_type_id: Option<u32>,
    /// Size of the referenced table observed by this step.
    pub table_size: Option<u32>,
    /// Normalized function reference selected by call-like opcodes.
    pub function_ref: Option<u32>,
    /// True when `function_ref` names a guest-defined function rather than a host import.
    pub target_function_is_guest: bool,
    /// Parameter/result arity for the selected call target.
    pub call_param_count: Option<u8>,
    pub call_result_count: Option<u8>,
    /// Pushed to the runtime call stack at this step (populated for `call` instructions).
    /// Contains (return_pc, caller_fbp) — the return context saved before entering the callee.
    pub call_stack_push: Option<(u64, u64)>,
    /// Popped from the runtime call stack at this step (populated for non-final `return`).
    /// Contains (return_pc, caller_fbp) — restored when returning to the caller.
    pub call_stack_pop: Option<(u64, u64)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WasmBuildError {
    Trace(String),
    Unsupported(String),
    StateMismatch(String),
}

impl core::fmt::Display for WasmBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Trace(msg) | Self::Unsupported(msg) | Self::StateMismatch(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for WasmBuildError {}

pub fn boundary_states(trace: &[WasmStepTrace]) -> Vec<(WasmBoundaryState, WasmBoundaryState)> {
    trace
        .iter()
        .map(|row| {
            (
                WasmBoundaryState {
                    pc: row.pc_before,
                    sp: row.sp_before,
                    memory_pages: row.memory_pages_before,
                    locals_fbp: row.locals_fbp,
                    halted: false,
                    param_init: row.param_init_before,
                },
                WasmBoundaryState {
                    pc: row.pc_after,
                    sp: row.sp_after,
                    memory_pages: row.memory_pages_after,
                    locals_fbp: row.locals_fbp_after,
                    halted: row.halted,
                    param_init: row.param_init_after,
                },
            )
        })
        .collect()
}
