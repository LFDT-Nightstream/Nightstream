use super::isa::{WasmOpcode, WasmOpcodeInfo};

// Proof-visible `function_call_metadata` ROM encoding: parameter count in
// bits 0..7, result count in bits 8..15, and the guest flag in bit 16.
pub(crate) const FUNCTION_CALL_METADATA_RESULT_FACTOR: u64 = 1 << 8;
pub(crate) const FUNCTION_CALL_METADATA_GUEST_FACTOR: u64 = 1 << 16;

pub(crate) const fn pack_function_call_metadata(param_count: u8, result_count: u8, is_guest: bool) -> u64 {
    param_count as u64
        + (result_count as u64) * FUNCTION_CALL_METADATA_RESULT_FACTOR
        + (is_guest as u64) * FUNCTION_CALL_METADATA_GUEST_FACTOR
}

pub(crate) const fn function_call_metadata_is_guest(metadata: u64) -> bool {
    metadata & FUNCTION_CALL_METADATA_GUEST_FACTOR != 0
}

pub(crate) const fn function_call_metadata_shape(metadata: u64) -> (u8, u8, bool) {
    (
        metadata as u8,
        (metadata / FUNCTION_CALL_METADATA_RESULT_FACTOR) as u8,
        function_call_metadata_is_guest(metadata),
    )
}

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
/// Gather rows stage event words into `evbuf`; when the 8-word block fills,
/// `perm_pending` is raised and a group of `HostEventPerm` aux rows runs the
/// width-12 permutation one round-row at a time (`perm_round` is the position
/// inside that group, 0 when idle). The group's last row folds the block into
/// `WasmStepState::comm_chain`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmEventAbsorbState {
    /// The 8-word block currently being filled (already-absorbed slots are
    /// zeroed by the group's first perm row).
    pub evbuf: [u64; 8],
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

/// Carried state of the host-event gather machinery: the per-call event
/// schedule, the argument-region base for
/// addressed slot reads, and the slot cursor inside the block being
/// staged (see [`crate::host_event_bindings`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmHostEventState {
    /// Export whose invocation owns this turn. Stable across import calls
    /// and guest tail calls; changed only by a turn boundary.
    pub turn_export_fref: u32,
    /// Events still owed in the current phase; loaded from the event-count
    /// ROM on the call row (pre) and the result row (post), decremented as
    /// each block's last slot row stages it. Program rows require zero.
    pub events_remaining: u32,
    /// Current event's index within the template (the ROM key component);
    /// zeroed on the call row, incremented per completed block.
    pub event_index: u32,
    /// Stack slot index of the call's first argument:
    /// `sp_at_call - index_pops - param_count`; latched on the call row.
    pub args_base: u64,
    /// Next block word a slot row stages (0..=7).
    pub slot_cursor: u8,
}

impl WasmHostEventState {
    pub const ZERO: Self = Self {
        turn_export_fref: 0,
        events_remaining: 0,
        event_index: 0,
        args_base: 0,
        slot_cursor: 0,
    };
}

impl Default for WasmHostEventState {
    fn default() -> Self {
        Self::ZERO
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmBoundaryState {
    pub pc: u64,
    pub sp: u64,
    pub stack_frame_base: u64,
    pub output_enabled: bool,
    pub output_value_lo: u32,
    pub output_value_hi: u32,
    pub memory_pages: Option<u32>,
    pub locals_fbp: u64,
    pub halted: bool,
    pub trapped: bool,
    pub param_init: WasmCountdownState,
    pub tail_call_pending: bool,
    pub host_callee_fref: u32,
    pub comm_chain: [u64; 4],
    pub event_absorb: WasmEventAbsorbState,
    pub host_events: WasmHostEventState,
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
    /// Global operand-stack address at which the current function frame starts.
    pub stack_frame_base: u64,
    pub output: WasmOutputState,
    pub call_stack_depth: u64,
    pub memory_pages: Option<u32>,
    /// Maximum linear-memory page count (module's declared memory maximum,
    /// capped at the wasm32 limit of 65536). Constant for the whole execution
    /// and carried in the semantic-state digest so it is verifier-authoritative
    /// — `memory.grow` and the OOB bound check depend on it not being forgeable.
    pub max_memory_pages: Option<u32>,
    pub locals_fbp: u64,
    /// The current turn has halted: raised by the halting row, carried (and
    /// digest-bound) until a turn boundary clears it. Program rows require
    /// `false`, so halt is terminal except for exit-event draining, padding,
    /// and an explicit re-entry.
    pub halted: bool,
    /// Execution ended in a wasm trap. Terminal and mutually exclusive with
    /// a captured output; carried into the semantic-state digest so a
    /// verifier can assert "this execution trapped".
    pub trapped: bool,
    pub param_init: WasmCountdownState,
    /// A tail call has initialized its replacement frame but still needs to
    /// discard the replaced frame's residual operand stack.
    pub tail_call_pending: bool,
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
    /// Host-event gather machinery state.
    pub host_events: WasmHostEventState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmAuxOpcode {
    CallParamInit,
    /// Drops residual operands from the replaced frame after tail-call
    /// parameters have been copied into the callee's locals.
    TailEnter,
    /// One row of the host-event chain permutation group: a full-round row or
    /// a partial-pair row of the width-12 Poseidon2 block absorb (see
    /// [`crate::comm_chain::COMM_CHAIN_PERM_ROWS`]). Scheduled whenever
    /// `WasmEventAbsorbState::perm_pending` is raised.
    HostEventPerm,
    /// Eight rows stage an expanded event block into the absorb buffer, then
    /// raise `perm_pending` for its permutation group.
    HostEventGather,
    /// Re-entry between export invocations. Requires the previous turn to be
    /// halted and drained, then loads the next export's entry PC and schedule.
    TurnBoundary,
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

    pub fn is_tail_enter(self) -> bool {
        matches!(self, Self::Aux(WasmAuxOpcode::TailEnter))
    }

    pub fn is_host_event_perm(self) -> bool {
        matches!(self, Self::Aux(WasmAuxOpcode::HostEventPerm))
    }

    pub fn is_host_event_gather(self) -> bool {
        matches!(self, Self::Aux(WasmAuxOpcode::HostEventGather))
    }

    pub fn is_turn_boundary(self) -> bool {
        matches!(self, Self::Aux(WasmAuxOpcode::TurnBoundary))
    }

    pub fn is_padding(self) -> bool {
        matches!(self, Self::Aux(WasmAuxOpcode::Padding))
    }
}

/// The source binding selected by one host-event gather slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum WasmHostEventSlotKind {
    Const,
    Arg,
    Result,
    InputLocal,
    Output,
    MemoryRead,
    MemoryWrite,
}

impl WasmHostEventSlotKind {
    pub const COUNT: usize = 7;

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// Native meaning of the host-event ROM's kind-dependent variant column.
///
/// [`WasmHostEventRomVariant::encoded`] is the only conversion to the compact
/// field representation used by the ROM and circuit.
/// Width of a host-event linear-memory access.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmHostEventMemoryWidth {
    Byte,
    Half,
    Word,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmHostEventMemoryBase {
    Argument,
    Local,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmHostEventRomVariant {
    None,
    LowLimb,
    HighLimb,
    Memory {
        base: WasmHostEventMemoryBase,
        width: WasmHostEventMemoryWidth,
    },
}

impl WasmHostEventRomVariant {
    pub(crate) const MEMORY_BYTE_ENCODING_FACTOR: u8 = 2;
    pub(crate) const MEMORY_HALF_ENCODING_FACTOR: u8 = 4;
    pub(crate) const MEMORY_OUTPUT_ENCODING_FACTOR: u8 = 8;

    pub const fn encoded(self) -> u8 {
        match self {
            Self::None | Self::LowLimb => 0,
            Self::HighLimb => 1,
            Self::Memory { base, width } => {
                let width = match width {
                    WasmHostEventMemoryWidth::Word => 0,
                    WasmHostEventMemoryWidth::Byte => Self::MEMORY_BYTE_ENCODING_FACTOR,
                    WasmHostEventMemoryWidth::Half => Self::MEMORY_HALF_ENCODING_FACTOR,
                };
                let base = match base {
                    WasmHostEventMemoryBase::Argument => 0,
                    WasmHostEventMemoryBase::Local => 1,
                    WasmHostEventMemoryBase::Output => Self::MEMORY_OUTPUT_ENCODING_FACTOR,
                };
                base + width
            }
        }
    }

    pub const fn is_low_limb(self) -> bool {
        matches!(self, Self::LowLimb)
    }

    pub const fn is_high_limb(self) -> bool {
        matches!(self, Self::HighLimb)
    }

    pub const fn uses_local_memory_base(self) -> bool {
        matches!(
            self,
            Self::Memory {
                base: WasmHostEventMemoryBase::Local,
                ..
            }
        )
    }

    pub const fn uses_output_memory_base(self) -> bool {
        matches!(
            self,
            Self::Memory {
                base: WasmHostEventMemoryBase::Output,
                ..
            }
        )
    }

    pub const fn uses_byte_memory_width(self) -> bool {
        matches!(
            self,
            Self::Memory {
                width: WasmHostEventMemoryWidth::Byte,
                ..
            }
        )
    }

    pub const fn uses_half_memory_width(self) -> bool {
        matches!(
            self,
            Self::Memory {
                width: WasmHostEventMemoryWidth::Half,
                ..
            }
        )
    }
}

/// The host-event ROM entry expected by a gather row (bound by the internal `host_event_slot_*`
/// families at key `(fref, event_index, slot_cursor)`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmHostEventRomEntry {
    pub kind: WasmHostEventSlotKind,
    /// Argument, local, or runtime-input index, depending on `kind`.
    pub arg: u8,
    /// Kind-dependent native variant; encoded only at the ROM/circuit boundary.
    pub variant: WasmHostEventRomVariant,
    /// `Const` low limb; memory byte offset; zero for other slot kinds.
    pub immediate0: u32,
    /// `Const` high limb; `MemoryWrite*` input index; zero for other slot kinds.
    pub immediate1: u32,
    /// Whether this slot belongs to an unabsorbed event.
    pub advice: bool,
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
    /// Call-like rows read only an indirect table index; args are popped by
    /// guest param-init or host-arg aux rows.
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
    /// Pushed by ordinary guest calls: return pc, caller locals base, and
    /// caller operand-stack base.
    pub call_stack_push: Option<(u64, u64, u64)>,
    /// Popped by non-final returns, restoring the same three caller fields.
    pub call_stack_pop: Option<(u64, u64, u64)>,
    /// Host-event ROM slot entry for `HostEventGather` rows.
    pub host_event_rom_slot: Option<WasmHostEventRomEntry>,
    /// Biased initial-schedule count read when a host-event schedule starts.
    pub host_event_initial_schedule_count: Option<u32>,
    /// Exit-schedule count read when a clean export halt starts its exit events.
    pub host_event_exit_schedule_count: Option<u32>,
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
                    stack_frame_base: row.state_before.stack_frame_base,
                    output_enabled: row.state_before.output.enabled,
                    output_value_lo: row.state_before.output.value_lo,
                    output_value_hi: row.state_before.output.value_hi,
                    memory_pages: row.state_before.memory_pages,
                    locals_fbp: row.state_before.locals_fbp,
                    halted: row.state_before.halted,
                    trapped: row.state_before.trapped,
                    param_init: row.state_before.param_init,
                    tail_call_pending: row.state_before.tail_call_pending,
                    host_callee_fref: row.state_before.host_callee_fref,
                    comm_chain: row.state_before.comm_chain,
                    event_absorb: row.state_before.event_absorb,
                    host_events: row.state_before.host_events,
                },
                WasmBoundaryState {
                    pc: row.state_after.pc,
                    sp: row.state_after.sp,
                    stack_frame_base: row.state_after.stack_frame_base,
                    output_enabled: row.state_after.output.enabled,
                    output_value_lo: row.state_after.output.value_lo,
                    output_value_hi: row.state_after.output.value_hi,
                    memory_pages: row.state_after.memory_pages,
                    locals_fbp: row.state_after.locals_fbp,
                    halted: row.state_after.halted,
                    trapped: row.state_after.trapped,
                    param_init: row.state_after.param_init,
                    tail_call_pending: row.state_after.tail_call_pending,
                    host_callee_fref: row.state_after.host_callee_fref,
                    comm_chain: row.state_after.comm_chain,
                    event_absorb: row.state_after.event_absorb,
                    host_events: row.state_after.host_events,
                },
            )
        })
        .collect()
}
