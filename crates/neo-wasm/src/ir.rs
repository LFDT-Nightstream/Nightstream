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
pub struct StackValueAccess {
    pub addr_lo: u64,
    pub value_lo: u32,
    pub value_hi: Option<u32>,
}

impl StackValueAccess {
    pub const fn new(addr_lo: u64, value_lo: u32) -> Self {
        Self {
            addr_lo,
            value_lo,
            value_hi: None,
        }
    }

    pub const fn with_hi(addr_lo: u64, value_lo: u32, value_hi: u32) -> Self {
        Self {
            addr_lo,
            value_lo,
            value_hi: Some(value_hi),
        }
    }

    pub const fn with_optional_hi(mut self, value_hi: Option<u32>) -> Self {
        self.value_hi = value_hi;
        self
    }
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

/// Carried state of an aux-row countdown mode: `active` while `remaining`
/// counts down to zero, one aux row per tick. Used by both call-argument
/// modes — param-init (guest) and host-arg (host) — which differ only in
/// what each popped value feeds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmCountdownState {
    pub active: bool,
    pub remaining: u32,
}

impl WasmCountdownState {
    pub const ZERO: Self = Self {
        active: false,
        remaining: 0,
    };
}

impl Default for WasmCountdownState {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Carried state of the in-circuit host-event absorb machinery.
///
/// Host-call rows stream the event's words (header, popped args, result)
/// into `evbuf`; when the 8-word block fills — or the event's stream ends —
/// `perm_pending` is raised and a group of `HostEventPerm` aux rows runs the
/// width-12 permutation one round-row at a time (`perm_round` is the position
/// inside that group, 0 when idle). The group's last row folds the block into
/// `WasmStepState::comm_chain`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmEventAbsorbState {
    /// The 8-word block currently being filled (already-absorbed slots are
    /// zeroed by the group's first perm row).
    pub evbuf: [u64; 8],
    /// Next buffer pair slot (0..=3) an event word pair lands in.
    pub evbuf_slot: u8,
    /// A filled block (or completed event stream) awaits its perm rows.
    pub perm_pending: bool,
    /// Row position inside the current perm group (0 when idle).
    pub perm_round: u8,
    /// Running permutation state. Meaningful only from the absorb (the row
    /// raising `perm_pending` premixes `[chain | evbuf]` with the initial
    /// external linear layer) through the group's rows; carried junk
    /// in between and never cleared.
    pub perm_state: [u64; 12],
}

impl WasmEventAbsorbState {
    pub const ZERO: Self = Self {
        evbuf: [0; 8],
        evbuf_slot: 0,
        perm_pending: false,
        perm_round: 0,
        perm_state: [0; 12],
    };
}

impl Default for WasmEventAbsorbState {
    fn default() -> Self {
        Self::ZERO
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmBoundaryState {
    pub pc: u64,
    pub sp: u64,
    pub output_enabled: bool,
    pub output_value_lo: u32,
    pub output_value_hi: u32,
    pub memory_pages: Option<u32>,
    pub locals_fbp: u64,
    pub halted: bool,
    pub trapped: bool,
    pub param_init: WasmCountdownState,
    pub host_args: WasmCountdownState,
    pub host_result_pending: bool,
    pub host_callee_fref: u32,
    pub comm_chain: [u64; 4],
    pub event_absorb: WasmEventAbsorbState,
}

/// Carry state for binding the whole execution's claimed output.
///
/// This is not the result produced by this row's opcode. It is protocol-side
/// state maintained by the tracer/normalizer so the final top-level function
/// result can be carried into the proof/public-input boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmOutputState {
    pub enabled: bool,
    pub value_lo: u32,
    pub value_hi: u32,
}

impl WasmOutputState {
    pub const ZERO: Self = Self {
        enabled: false,
        value_lo: 0,
        value_hi: 0,
    };
}

impl Default for WasmOutputState {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Carried VM/IVC state at one side of a normalized trace row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmStepState {
    pub pc: u64,
    pub sp: u64,
    pub output: WasmOutputState,
    pub call_stack_depth: u64,
    pub memory_pages: Option<u32>,
    /// Maximum linear-memory page count (module's declared memory maximum,
    /// capped at the wasm32 limit of 65536). Constant for the whole execution
    /// and carried in the semantic-state digest so it is verifier-authoritative
    /// — `memory.grow` and the OOB bound check depend on it not being forgeable.
    pub max_memory_pages: Option<u32>,
    pub locals_fbp: u64,
    pub halted: bool,
    /// Execution ended in a wasm trap. Terminal and mutually exclusive with
    /// a captured output; carried into the semantic-state digest so a
    /// verifier can assert "this execution trapped".
    pub trapped: bool,
    pub param_init: WasmCountdownState,
    /// Host-call argument-pop mode: each `HostCallArg` aux row pops one
    /// pre-call operand while `remaining` counts down to zero.
    pub host_args: WasmCountdownState,
    /// A host call with `result_count = 1` still owes its result push; the
    /// `HostCallResult` aux row consumes this flag.
    pub host_result_pending: bool,
    /// Callee attribution for host-call events: set from the call row's
    /// (ROM/table-bound) `COL_FUNCTION_REF` on every host call and preserved
    /// on all other rows until the next host call overwrites it. Consumers
    /// (the event absorb) read it only on rows of the event that set it, so
    /// the stale value between events is inert.
    pub host_callee_fref: u32,
    /// Host-event commitment chain state (canonical Goldilocks limbs; see
    /// [`crate::comm_chain`]). Genesis is all-zero; the last row of each
    /// absorbed block's `HostEventPerm` group folds the block in
    /// (feed-forward); every other row carries it unchanged.
    pub comm_chain: [u64; 4],
    /// In-circuit host-event absorb machinery (block buffer + perm rows).
    pub event_absorb: WasmEventAbsorbState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmAuxOpcode {
    CallParamInit,
    /// Pops one host-call argument off the operand stack. The program call row
    /// pops only the indirect table index; argument arity is handled by these
    /// aux rows.
    HostCallArg,
    /// Pushes the host call's single result at the post-pop stack top.
    /// Emitted after the `HostCallArg` rows iff `result_count = 1` (the
    /// canonical ABI caps flat results at 1).
    HostCallResult,
    /// One row of the host-event chain permutation group: a full-round row or
    /// a partial-pair row of the width-12 Poseidon2 block absorb (see
    /// [`crate::comm_chain::COMM_CHAIN_PERM_ROWS`]). Scheduled whenever
    /// `WasmEventAbsorbState::perm_pending` is raised.
    HostEventPerm,
    /// Synthetic state-preserving row used to pad a trace up to a
    /// multiple of `batch_size`. Not a real wasm opcode — the CCS gates
    /// these rows so that `_after == _before` for every state column.
    Padding,
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

    pub fn is_host_call_arg(self) -> bool {
        matches!(self, Self::Aux(WasmAuxOpcode::HostCallArg))
    }

    pub fn is_host_call_result(self) -> bool {
        matches!(self, Self::Aux(WasmAuxOpcode::HostCallResult))
    }

    pub fn is_host_event_perm(self) -> bool {
        matches!(self, Self::Aux(WasmAuxOpcode::HostEventPerm))
    }

    pub fn is_padding(self) -> bool {
        matches!(self, Self::Aux(WasmAuxOpcode::Padding))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WasmVmStep {
    pub cycle: u64,
    pub row_kind: WasmRowKind,
    pub state_before: WasmStepState,
    pub state_after: WasmStepState,
    /// Branch/discriminant used when one opcode has multiple valid next-PC edges.
    /// For example, `br_if` chooses taken vs fallthrough, and `br_table`
    /// chooses one table arm/default. Static fallthrough rows use 0.
    pub control_choice: u32,
    pub pc_edge_kind: WasmPcEdgeKind,
    pub wide_values_enabled: bool,
    pub opcode: WasmOpcode,
    pub info: WasmOpcodeInfo,
    /// Dynamic stack read count when opcode metadata is not enough.
    /// Call rows read only the `call_indirect` table index; args are popped
    /// by param-init (guest) or host-arg (host) aux rows.
    pub stack_reads_override: Option<u8>,
    /// Dynamic stack write count when opcode metadata is not enough.
    /// Guest calls produce their return values in later guest rows; host
    /// calls push their result on a trailing host-result aux row.
    pub stack_writes_override: Option<u8>,
    pub output_captured: bool,
    /// Normalized function reference for the currently executing frame.
    /// `function_ref` below is opcode-target metadata, not this frame identity.
    pub current_function_ref: u32,
    pub current_function_num_locals: u32,
    pub stack_read0: Option<StackValueAccess>,
    pub stack_read1: Option<StackValueAccess>,
    pub stack_read2: Option<StackValueAccess>,
    pub stack_write0: Option<StackValueAccess>,
    pub linear_memory: Option<LinearMemoryAccess>,
    pub linear_memory_offset: u64,
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
    /// adapter is expected to populate this with the selected funcref id so witness-level memory
    /// semantics (and the per-row CCS constraints) can validate `pc_after` against the static
    /// `function_entries` ROM.
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

pub fn boundary_states(trace: &[WasmVmStep]) -> Vec<(WasmBoundaryState, WasmBoundaryState)> {
    trace
        .iter()
        .map(|row| {
            (
                WasmBoundaryState {
                    pc: row.state_before.pc,
                    sp: row.state_before.sp,
                    output_enabled: row.state_before.output.enabled,
                    output_value_lo: row.state_before.output.value_lo,
                    output_value_hi: row.state_before.output.value_hi,
                    memory_pages: row.state_before.memory_pages,
                    locals_fbp: row.state_before.locals_fbp,
                    halted: row.state_before.halted,
                    trapped: row.state_before.trapped,
                    param_init: row.state_before.param_init,
                    host_args: row.state_before.host_args,
                    host_result_pending: row.state_before.host_result_pending,
                    host_callee_fref: row.state_before.host_callee_fref,
                    comm_chain: row.state_before.comm_chain,
                    event_absorb: row.state_before.event_absorb,
                },
                WasmBoundaryState {
                    pc: row.state_after.pc,
                    sp: row.state_after.sp,
                    output_enabled: row.state_after.output.enabled,
                    output_value_lo: row.state_after.output.value_lo,
                    output_value_hi: row.state_after.output.value_hi,
                    memory_pages: row.state_after.memory_pages,
                    locals_fbp: row.state_after.locals_fbp,
                    halted: row.state_after.halted,
                    trapped: row.state_after.trapped,
                    param_init: row.state_after.param_init,
                    host_args: row.state_after.host_args,
                    host_result_pending: row.state_after.host_result_pending,
                    host_callee_fref: row.state_after.host_callee_fref,
                    comm_chain: row.state_after.comm_chain,
                    event_absorb: row.state_after.event_absorb,
                },
            )
        })
        .collect()
}
