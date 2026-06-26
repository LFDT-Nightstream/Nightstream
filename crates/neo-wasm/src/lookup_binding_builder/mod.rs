use super::isa::WasmOpTable;
use super::ivc_state::{build_ivc_state_continuity_links, StateColumns, WasmCrossStepLinkSpec};
use super::layout::{
    selector_col, COL_CALL_INDIRECT_IS_NOT_TRAP, COL_CALL_INDIRECT_TYPE_INDEX, COL_CALL_PARAM_COUNT,
    COL_CALL_RESULT_COUNT, COL_CALL_STACK_ADDR, COL_CALL_STACK_DEPTH_AFTER, COL_CALL_STACK_DEPTH_BEFORE,
    COL_CALL_STACK_POP_CALLER_FBP, COL_CALL_STACK_POP_PRESENT, COL_CALL_STACK_POP_RETURN_PC,
    COL_CALL_STACK_PUSH_PRESENT, COL_CALL_STACK_RETURN_PC_CHOICE, COL_CONTROL_CHOICE, COL_CURRENT_FUNCTION_NUM_LOCALS,
    COL_CURRENT_FUNCTION_REF, COL_EXPECTED_TYPE_ID, COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, COL_FUNCTION_REF,
    COL_FUNCTION_TYPE_ID, COL_GLOBAL_INDEX, COL_GLOBAL_VALUE, COL_GLOBAL_VALUE_HI, COL_HALTED, COL_IS_PROGRAM_ROW,
    COL_LINEAR_MEM_ACCESS_BYTE0, COL_LINEAR_MEM_ACCESS_BYTE1, COL_LINEAR_MEM_ACCESS_BYTE2, COL_LINEAR_MEM_ACCESS_BYTE3,
    COL_LINEAR_MEM_ACCESS_BYTE4, COL_LINEAR_MEM_ACCESS_BYTE5, COL_LINEAR_MEM_ACCESS_BYTE6, COL_LINEAR_MEM_ACCESS_BYTE7,
    COL_LINEAR_MEM_BYTE_OFFSET, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1,
    COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_3,
    COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_0, COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_1,
    COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_2, COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_3,
    COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_0, COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_1,
    COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_2, COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_3,
    COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_0, COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_1,
    COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_2, COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_3, COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_0,
    COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_1, COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_2, COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_3,
    COL_LINEAR_MEM_I64_STORE_OFFSET_IS_0, COL_LINEAR_MEM_I64_STORE_OFFSET_IS_1, COL_LINEAR_MEM_I64_STORE_OFFSET_IS_2,
    COL_LINEAR_MEM_I64_STORE_OFFSET_IS_3, COL_LINEAR_MEM_IMM_OFFSET, COL_LINEAR_MEM_IS_BYTE_WIDTH,
    COL_LINEAR_MEM_IS_DOUBLE_WIDTH, COL_LINEAR_MEM_IS_FULL_WIDTH, COL_LINEAR_MEM_IS_HALF_WIDTH,
    COL_LINEAR_MEM_LANE0_ADDR, COL_LINEAR_MEM_LANE0_BYTE0, COL_LINEAR_MEM_LANE0_BYTE0_BEFORE,
    COL_LINEAR_MEM_LANE0_BYTE1, COL_LINEAR_MEM_LANE0_BYTE1_BEFORE, COL_LINEAR_MEM_LANE0_BYTE2,
    COL_LINEAR_MEM_LANE0_BYTE2_BEFORE, COL_LINEAR_MEM_LANE0_BYTE3, COL_LINEAR_MEM_LANE0_BYTE3_BEFORE,
    COL_LINEAR_MEM_LANE0_LOAD_ACTIVE, COL_LINEAR_MEM_LANE0_STORE_ACTIVE, COL_LINEAR_MEM_LANE0_VALUE,
    COL_LINEAR_MEM_LANE0_VALUE_BEFORE, COL_LINEAR_MEM_LANE1_ADDR, COL_LINEAR_MEM_LANE1_BYTE0,
    COL_LINEAR_MEM_LANE1_BYTE0_BEFORE, COL_LINEAR_MEM_LANE1_BYTE1, COL_LINEAR_MEM_LANE1_BYTE1_BEFORE,
    COL_LINEAR_MEM_LANE1_BYTE2, COL_LINEAR_MEM_LANE1_BYTE2_BEFORE, COL_LINEAR_MEM_LANE1_BYTE3,
    COL_LINEAR_MEM_LANE1_BYTE3_BEFORE, COL_LINEAR_MEM_LANE1_LOAD_ACTIVE, COL_LINEAR_MEM_LANE1_STORE_ACTIVE,
    COL_LINEAR_MEM_LANE1_VALUE, COL_LINEAR_MEM_LANE1_VALUE_BEFORE, COL_LINEAR_MEM_LANE2_ADDR,
    COL_LINEAR_MEM_LANE2_BYTE0, COL_LINEAR_MEM_LANE2_BYTE0_BEFORE, COL_LINEAR_MEM_LANE2_BYTE1,
    COL_LINEAR_MEM_LANE2_BYTE1_BEFORE, COL_LINEAR_MEM_LANE2_BYTE2, COL_LINEAR_MEM_LANE2_BYTE2_BEFORE,
    COL_LINEAR_MEM_LANE2_BYTE3, COL_LINEAR_MEM_LANE2_BYTE3_BEFORE, COL_LINEAR_MEM_LANE2_LOAD_ACTIVE,
    COL_LINEAR_MEM_LANE2_STORE_ACTIVE, COL_LINEAR_MEM_LANE2_VALUE, COL_LINEAR_MEM_LANE2_VALUE_BEFORE,
    COL_LINEAR_MEM_OFFSET_IS_0, COL_LINEAR_MEM_OFFSET_IS_1, COL_LINEAR_MEM_OFFSET_IS_2, COL_LINEAR_MEM_OFFSET_IS_3,
    COL_LINEAR_MEM_USE_LANE0, COL_LINEAR_MEM_USE_LANE1, COL_LINEAR_MEM_USE_LANE2, COL_LOCALS_FBP_AFTER,
    COL_LOCALS_FBP_BEFORE, COL_LOCAL_INDEX, COL_LOCAL_VALUE, COL_LOCAL_VALUE_HI, COL_LOCAL_WRITE_ENABLED,
    COL_OPCODE_CODE, COL_OP_TABLE_ENABLED, COL_OP_TABLE_ID, COL_OUTPUT_CAPTURED, COL_PADDING_ACTIVE,
    COL_PARAM_INIT_ACTIVE_AFTER, COL_PARAM_INIT_ACTIVE_BEFORE, COL_PARAM_INIT_REMAINING_AFTER,
    COL_PARAM_INIT_REMAINING_AFTER_INV, COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO, COL_PARAM_INIT_REMAINING_BEFORE,
    COL_PC_AFTER, COL_PC_BEFORE, COL_PC_EDGE_KIND, COL_PC_EDGE_KIND_INV, COL_PC_EDGE_KIND_IS_STATIC, COL_PC_ROM_ACTIVE,
    COL_SIGN_EXT_BIT, COL_SIGN_EXT_LOW7, COL_SP_AFTER, COL_SP_BEFORE, COL_STACK_READ0_ACTIVE, COL_STACK_READ0_ADDR_HI,
    COL_STACK_READ0_ADDR_LO, COL_STACK_READ0_VALUE_HI, COL_STACK_READ0_VALUE_LO, COL_STACK_READ1_ACTIVE,
    COL_STACK_READ1_ADDR_HI, COL_STACK_READ1_ADDR_LO, COL_STACK_READ1_VALUE_HI, COL_STACK_READ1_VALUE_LO,
    COL_STACK_READ2_ACTIVE, COL_STACK_READ2_ADDR_HI, COL_STACK_READ2_ADDR_LO, COL_STACK_READ2_VALUE_HI,
    COL_STACK_READ2_VALUE_LO, COL_STACK_READS, COL_STACK_WRITE0_ACTIVE, COL_STACK_WRITE0_ADDR_HI,
    COL_STACK_WRITE0_ADDR_LO, COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO, COL_STACK_WRITES, COL_TABLE_ID,
    COL_TABLE_INDEX, COL_TABLE_READ_ENABLED, COL_TABLE_SIZE, COL_TABLE_SIZE_READ_ENABLED, COL_TABLE_VALUE,
    COL_TARGET_FUNCTION_IS_GUEST, COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE, COL_WIDE_VALUES_ENABLED, WITNESS_WIDTH,
};
use super::lookup_semantics::{semantics_for_lookup_family, LookupSemantics};
use super::tables::WasmLookupArity;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Column(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmLookupFamilySpec {
    pub name: &'static str,
    pub arity: WasmLookupArity,
    pub kind: WasmLookupFamilyKind,
    pub semantics: LookupSemantics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmLookupFamilyKind {
    OpTable(WasmOpTable),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmLookupBindingSpec {
    pub name: &'static str,
    pub family: &'static str,
    pub columns: Vec<Column>,
    pub gate: Option<Column>,
    pub role: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmMemorySpec {
    pub name: &'static str,
    pub columns: Vec<WasmMemoryColumnSpec>,
    pub is_rom: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmMemoryColumnSpec {
    pub address_columns: Vec<Column>,
    pub value_column: Column,
    pub kind: WasmMemoryColumnKind,
    pub activation: WasmMemoryActivation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmMemoryColumnKind {
    Read,
    Write {
        /// The column witnessing the prior memory state at `address_columns`.
        /// `None` means the wasm CCS does not constrain the prior value beyond what
        /// the memory argument enforces (the read witness still participates in the
        /// mcc). `Some(c)` means row-local CCS constraints reference `c` — e.g.,
        /// subword stores' byte-preservation rows compare bytes of `c` against
        /// bytes of `value_column`. So the MCC needs to either emit an equality
        /// constraint to this column, or use it directly.
        value_before_column: Option<Column>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmMemoryActivation {
    Always,
    BooleanGate(Column),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearMemoryColumns {
    pub imm_offset: Column,
    pub byte_offset: Column,
    pub use_lane0: Column,
    pub use_lane1: Column,
    pub use_lane2: Column,
    /// Per-lane direction gates: `laneN_load_active = use_laneN · is_load`,
    /// `laneN_store_active = use_laneN · is_store`. Bound by the CCS via a
    /// quadratic row each. Used as the activation column for the lane's
    /// memory spec entry — Read on load, Write+RMW on store — so loads emit
    /// only a Read tuple into the memory log (no write-modify-memory channel
    /// available to a malicious prover on a load row).
    pub lane0_load_active: Column,
    pub lane1_load_active: Column,
    pub lane2_load_active: Column,
    pub lane0_store_active: Column,
    pub lane1_store_active: Column,
    pub lane2_store_active: Column,
    pub is_byte_width: Column,
    pub is_half_width: Column,
    pub is_full_width: Column,
    pub is_double_width: Column,
    pub lane0_addr: Column,
    pub lane0_value: Column,
    pub lane1_addr: Column,
    pub lane1_value: Column,
    pub lane2_addr: Column,
    pub lane2_value: Column,
    // one hot selector for possible byte offsets
    pub offset_is: [Column; 4],
    pub byte_width_offset_is: [Column; 4],
    pub half_width_offset_is: [Column; 4],
    pub full_width_offset_is: [Column; 4],
    pub double_width_offset_is: [Column; 4],
    pub i64_load_offset_is: [Column; 4],
    pub i64_store_offset_is: [Column; 4],
    pub lane0_bytes: [Column; 4],
    pub lane1_bytes: [Column; 4],
    pub lane2_bytes: [Column; 4],
    /// Lane values **before** this row's access. Together with `_bytes_before`
    /// they let row-local CCS constraints relate the prior state to the
    /// post-state — required for sound byte preservation in subword stores.
    pub lane0_value_before: Column,
    pub lane1_value_before: Column,
    pub lane2_value_before: Column,
    pub lane0_bytes_before: [Column; 4],
    pub lane1_bytes_before: [Column; 4],
    pub lane2_bytes_before: [Column; 4],
    /// High byte view of the value being read or written by i64 load/store
    /// rows. The low byte view lives in `SignExtensionColumns`, because it is
    /// also useful for non-memory sign-extension opcodes.
    pub access_bytes_hi: [Column; 4],
}

/// Shared scratch columns for low 32-bit byte decomposition and sign extension.
///
/// Linear-memory constraints use `bytes` as the low access-byte view. Non-memory
/// sign-extension opcodes also reuse this group; those opcodes are not
/// linear-memory rows, so only one constraint family is active on a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignExtensionColumns {
    pub bytes: [Column; 4],
    pub low7: Column,
    pub bit: Column,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlColumns {
    pub opcode_code: Column,
    // 1 if this is an opcode in the real wasm program (meaning we need to check
    // that is in fact in the checked binary)
    //
    // 0 if this is a helper/aux/micro-opcode
    pub is_program_row: Column,
    /// Synthetic state-preserving row marker. Mutually exclusive with
    /// `is_program_row` and `param_init.param_init_active_before` via the
    /// row-kind one-hot.
    pub padding_active: Column,
    pub pc_rom_active: Column,
    pub pc_edge_kind_is_static: Column,
    pub pc_edge_kind_inv: Column,
    pub control_choice: Column,
    pub pc_edge_kind: Column,
    pub wide_values_enabled: Column,
    pub halted: Column,
    pub stack_reads: Column,
    pub stack_writes: Column,
    pub stack_read0_active: Column,
    pub stack_read1_active: Column,
    pub stack_read2_active: Column,
    pub stack_write0_active: Column,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamInitColumns {
    pub param_init_active_before: Column,
    pub param_init_active_after: Column,
    pub param_init_remaining_before: Column,
    pub param_init_remaining_after: Column,
    pub param_init_remaining_after_is_zero: Column,
    pub param_init_remaining_after_inv: Column,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallColumns {
    pub call_stack_push_present: Column,
    pub call_stack_pop_present: Column,
    pub call_stack_access_return_pc: Column,
    pub call_stack_access_caller_fbp: Column,
    pub call_stack_depth_before: Column,
    pub call_stack_depth_after: Column,
    pub call_stack_addr: Column,
    pub call_stack_return_pc_choice: Column,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameColumns {
    pub current_function_ref: Column,
    pub current_function_num_locals: Column,
    pub locals_fbp_before: Column,
    pub locals_fbp_after: Column,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmLookupBindingLayout {
    pub witness_width: usize,
    pub lookup_families: Vec<WasmLookupFamilySpec>,
    pub lookup_bindings: Vec<WasmLookupBindingSpec>,
    pub memories: Vec<WasmMemorySpec>,
    pub cross_step_links: Vec<WasmCrossStepLinkSpec>,
    pub control: ControlColumns,
    pub state: StateColumns,
    pub param_init: ParamInitColumns,
    pub call: CallColumns,
    pub frame: FrameColumns,
    pub linear_memory: LinearMemoryColumns,
    pub sign_extension: SignExtensionColumns,
}

fn rom_read_spec(
    name: &'static str,
    address_columns: Vec<Column>,
    value_column: Column,
    activation: WasmMemoryActivation,
) -> WasmMemorySpec {
    WasmMemorySpec {
        name,
        columns: vec![WasmMemoryColumnSpec {
            address_columns,
            value_column,
            kind: WasmMemoryColumnKind::Read,
            activation,
        }],
        is_rom: true,
    }
}

pub fn build_wasm_lookup_binding_layout() -> &'static WasmLookupBindingLayout {
    static LAYOUT: OnceLock<WasmLookupBindingLayout> = OnceLock::new();
    LAYOUT.get_or_init(build_wasm_lookup_binding_layout_uncached)
}

fn build_wasm_lookup_binding_layout_uncached() -> WasmLookupBindingLayout {
    let control = ControlColumns {
        opcode_code: Column(COL_OPCODE_CODE),
        is_program_row: Column(COL_IS_PROGRAM_ROW),
        padding_active: Column(COL_PADDING_ACTIVE),
        pc_rom_active: Column(COL_PC_ROM_ACTIVE),
        pc_edge_kind_is_static: Column(COL_PC_EDGE_KIND_IS_STATIC),
        pc_edge_kind_inv: Column(COL_PC_EDGE_KIND_INV),
        control_choice: Column(COL_CONTROL_CHOICE),
        pc_edge_kind: Column(COL_PC_EDGE_KIND),
        wide_values_enabled: Column(COL_WIDE_VALUES_ENABLED),
        halted: Column(COL_HALTED),
        stack_reads: Column(COL_STACK_READS),
        stack_writes: Column(COL_STACK_WRITES),
        stack_read0_active: Column(COL_STACK_READ0_ACTIVE),
        stack_read1_active: Column(COL_STACK_READ1_ACTIVE),
        stack_read2_active: Column(COL_STACK_READ2_ACTIVE),
        stack_write0_active: Column(COL_STACK_WRITE0_ACTIVE),
    };
    let state = StateColumns {
        pc_before: Column(COL_PC_BEFORE),
        pc_after: Column(COL_PC_AFTER),
        sp_before: Column(COL_SP_BEFORE),
        sp_after: Column(COL_SP_AFTER),
        trapped_before: Column(COL_TRAPPED_BEFORE),
        trapped_after: Column(COL_TRAPPED_AFTER),
    };
    let param_init = ParamInitColumns {
        param_init_active_before: Column(COL_PARAM_INIT_ACTIVE_BEFORE),
        param_init_active_after: Column(COL_PARAM_INIT_ACTIVE_AFTER),
        param_init_remaining_before: Column(COL_PARAM_INIT_REMAINING_BEFORE),
        param_init_remaining_after: Column(COL_PARAM_INIT_REMAINING_AFTER),
        param_init_remaining_after_is_zero: Column(COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO),
        param_init_remaining_after_inv: Column(COL_PARAM_INIT_REMAINING_AFTER_INV),
    };
    let call = CallColumns {
        call_stack_push_present: Column(COL_CALL_STACK_PUSH_PRESENT),
        call_stack_pop_present: Column(COL_CALL_STACK_POP_PRESENT),
        call_stack_access_return_pc: Column(COL_CALL_STACK_POP_RETURN_PC),
        call_stack_access_caller_fbp: Column(COL_CALL_STACK_POP_CALLER_FBP),
        call_stack_depth_before: Column(COL_CALL_STACK_DEPTH_BEFORE),
        call_stack_depth_after: Column(COL_CALL_STACK_DEPTH_AFTER),
        call_stack_addr: Column(COL_CALL_STACK_ADDR),
        call_stack_return_pc_choice: Column(COL_CALL_STACK_RETURN_PC_CHOICE),
    };
    let frame = FrameColumns {
        current_function_ref: Column(COL_CURRENT_FUNCTION_REF),
        current_function_num_locals: Column(COL_CURRENT_FUNCTION_NUM_LOCALS),
        locals_fbp_before: Column(COL_LOCALS_FBP_BEFORE),
        locals_fbp_after: Column(COL_LOCALS_FBP_AFTER),
    };
    let linear_memory = LinearMemoryColumns {
        imm_offset: Column(COL_LINEAR_MEM_IMM_OFFSET),
        byte_offset: Column(COL_LINEAR_MEM_BYTE_OFFSET),
        use_lane0: Column(COL_LINEAR_MEM_USE_LANE0),
        use_lane1: Column(COL_LINEAR_MEM_USE_LANE1),
        use_lane2: Column(COL_LINEAR_MEM_USE_LANE2),
        lane0_load_active: Column(COL_LINEAR_MEM_LANE0_LOAD_ACTIVE),
        lane1_load_active: Column(COL_LINEAR_MEM_LANE1_LOAD_ACTIVE),
        lane2_load_active: Column(COL_LINEAR_MEM_LANE2_LOAD_ACTIVE),
        lane0_store_active: Column(COL_LINEAR_MEM_LANE0_STORE_ACTIVE),
        lane1_store_active: Column(COL_LINEAR_MEM_LANE1_STORE_ACTIVE),
        lane2_store_active: Column(COL_LINEAR_MEM_LANE2_STORE_ACTIVE),
        is_byte_width: Column(COL_LINEAR_MEM_IS_BYTE_WIDTH),
        is_half_width: Column(COL_LINEAR_MEM_IS_HALF_WIDTH),
        is_full_width: Column(COL_LINEAR_MEM_IS_FULL_WIDTH),
        is_double_width: Column(COL_LINEAR_MEM_IS_DOUBLE_WIDTH),
        lane0_addr: Column(COL_LINEAR_MEM_LANE0_ADDR),
        lane0_value: Column(COL_LINEAR_MEM_LANE0_VALUE),
        lane1_addr: Column(COL_LINEAR_MEM_LANE1_ADDR),
        lane1_value: Column(COL_LINEAR_MEM_LANE1_VALUE),
        lane2_addr: Column(COL_LINEAR_MEM_LANE2_ADDR),
        lane2_value: Column(COL_LINEAR_MEM_LANE2_VALUE),
        offset_is: [
            Column(COL_LINEAR_MEM_OFFSET_IS_0),
            Column(COL_LINEAR_MEM_OFFSET_IS_1),
            Column(COL_LINEAR_MEM_OFFSET_IS_2),
            Column(COL_LINEAR_MEM_OFFSET_IS_3),
        ],
        byte_width_offset_is: [
            Column(COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0),
            Column(COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1),
            Column(COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2),
            Column(COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_3),
        ],
        half_width_offset_is: [
            Column(COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_0),
            Column(COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_1),
            Column(COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_2),
            Column(COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_3),
        ],
        full_width_offset_is: [
            Column(COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_0),
            Column(COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_1),
            Column(COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_2),
            Column(COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_3),
        ],
        double_width_offset_is: [
            Column(COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_0),
            Column(COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_1),
            Column(COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_2),
            Column(COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_3),
        ],
        i64_load_offset_is: [
            Column(COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_0),
            Column(COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_1),
            Column(COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_2),
            Column(COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_3),
        ],
        i64_store_offset_is: [
            Column(COL_LINEAR_MEM_I64_STORE_OFFSET_IS_0),
            Column(COL_LINEAR_MEM_I64_STORE_OFFSET_IS_1),
            Column(COL_LINEAR_MEM_I64_STORE_OFFSET_IS_2),
            Column(COL_LINEAR_MEM_I64_STORE_OFFSET_IS_3),
        ],
        lane0_bytes: [
            Column(COL_LINEAR_MEM_LANE0_BYTE0),
            Column(COL_LINEAR_MEM_LANE0_BYTE1),
            Column(COL_LINEAR_MEM_LANE0_BYTE2),
            Column(COL_LINEAR_MEM_LANE0_BYTE3),
        ],
        lane1_bytes: [
            Column(COL_LINEAR_MEM_LANE1_BYTE0),
            Column(COL_LINEAR_MEM_LANE1_BYTE1),
            Column(COL_LINEAR_MEM_LANE1_BYTE2),
            Column(COL_LINEAR_MEM_LANE1_BYTE3),
        ],
        lane2_bytes: [
            Column(COL_LINEAR_MEM_LANE2_BYTE0),
            Column(COL_LINEAR_MEM_LANE2_BYTE1),
            Column(COL_LINEAR_MEM_LANE2_BYTE2),
            Column(COL_LINEAR_MEM_LANE2_BYTE3),
        ],
        lane0_value_before: Column(COL_LINEAR_MEM_LANE0_VALUE_BEFORE),
        lane1_value_before: Column(COL_LINEAR_MEM_LANE1_VALUE_BEFORE),
        lane2_value_before: Column(COL_LINEAR_MEM_LANE2_VALUE_BEFORE),
        lane0_bytes_before: [
            Column(COL_LINEAR_MEM_LANE0_BYTE0_BEFORE),
            Column(COL_LINEAR_MEM_LANE0_BYTE1_BEFORE),
            Column(COL_LINEAR_MEM_LANE0_BYTE2_BEFORE),
            Column(COL_LINEAR_MEM_LANE0_BYTE3_BEFORE),
        ],
        lane1_bytes_before: [
            Column(COL_LINEAR_MEM_LANE1_BYTE0_BEFORE),
            Column(COL_LINEAR_MEM_LANE1_BYTE1_BEFORE),
            Column(COL_LINEAR_MEM_LANE1_BYTE2_BEFORE),
            Column(COL_LINEAR_MEM_LANE1_BYTE3_BEFORE),
        ],
        lane2_bytes_before: [
            Column(COL_LINEAR_MEM_LANE2_BYTE0_BEFORE),
            Column(COL_LINEAR_MEM_LANE2_BYTE1_BEFORE),
            Column(COL_LINEAR_MEM_LANE2_BYTE2_BEFORE),
            Column(COL_LINEAR_MEM_LANE2_BYTE3_BEFORE),
        ],
        access_bytes_hi: [
            Column(COL_LINEAR_MEM_ACCESS_BYTE4),
            Column(COL_LINEAR_MEM_ACCESS_BYTE5),
            Column(COL_LINEAR_MEM_ACCESS_BYTE6),
            Column(COL_LINEAR_MEM_ACCESS_BYTE7),
        ],
    };
    let sign_extension = SignExtensionColumns {
        bytes: [
            Column(COL_LINEAR_MEM_ACCESS_BYTE0),
            Column(COL_LINEAR_MEM_ACCESS_BYTE1),
            Column(COL_LINEAR_MEM_ACCESS_BYTE2),
            Column(COL_LINEAR_MEM_ACCESS_BYTE3),
        ],
        low7: Column(COL_SIGN_EXT_LOW7),
        bit: Column(COL_SIGN_EXT_BIT),
    };

    let lookup_families: Vec<WasmLookupFamilySpec> = WasmOpTable::all()
        .into_iter()
        .map(|op_table| {
            let family = WasmLookupFamilySpec {
                name: op_table.name(),
                arity: match op_table {
                    WasmOpTable::I32Clz | WasmOpTable::I32Ctz | WasmOpTable::I32Popcnt => WasmLookupArity::Unary,
                    op if op.is_i64_binary() => WasmLookupArity::Tuple(4),
                    op if op.is_i64_unary() => WasmLookupArity::Tuple(2),
                    _ => WasmLookupArity::Binary,
                },
                kind: WasmLookupFamilyKind::OpTable(op_table),
                semantics: LookupSemantics {
                    predicate: super::lookup_semantics::LookupPredicate::And(Vec::new()),
                },
            };
            WasmLookupFamilySpec {
                semantics: semantics_for_lookup_family(&family),
                ..family
            }
        })
        .collect();

    let lookup_bindings: Vec<WasmLookupBindingSpec> = WasmOpTable::all()
        .into_iter()
        .map(|table| WasmLookupBindingSpec {
            name: table.name(),
            family: table.name(),
            columns: match table {
                op if op.is_i64_binary() => vec![
                    Column(COL_OP_TABLE_ID),
                    Column(COL_STACK_READ0_VALUE_LO),
                    Column(COL_STACK_READ0_VALUE_HI),
                    Column(COL_STACK_READ1_VALUE_LO),
                    Column(COL_STACK_READ1_VALUE_HI),
                    Column(COL_STACK_WRITE0_VALUE_LO),
                    Column(COL_STACK_WRITE0_VALUE_HI),
                ],
                op if op.is_i64_unary() => vec![
                    Column(COL_OP_TABLE_ID),
                    Column(COL_STACK_READ0_VALUE_LO),
                    Column(COL_STACK_READ0_VALUE_HI),
                    Column(COL_STACK_WRITE0_VALUE_LO),
                    Column(COL_STACK_WRITE0_VALUE_HI),
                ],
                _ => vec![
                    Column(COL_OP_TABLE_ID),
                    Column(COL_STACK_READ0_VALUE_LO),
                    Column(COL_STACK_READ1_VALUE_LO),
                    Column(COL_STACK_WRITE0_VALUE_LO),
                ],
            },
            gate: Some(Column(COL_OP_TABLE_ENABLED)),
            role: "op-table row binding",
        })
        .collect();

    let memories = vec![
        WasmMemorySpec {
            name: "stack",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ0_ADDR_LO)],
                    value_column: Column(COL_STACK_READ0_VALUE_LO),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(control.stack_read0_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ0_ADDR_HI)],
                    value_column: Column(COL_STACK_READ0_VALUE_HI),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(control.stack_read0_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ0_ADDR_LO)],
                    value_column: Column(COL_STACK_READ0_VALUE_LO),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_OUTPUT_CAPTURED)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ0_ADDR_HI)],
                    value_column: Column(COL_STACK_READ0_VALUE_HI),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_OUTPUT_CAPTURED)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ1_ADDR_LO)],
                    value_column: Column(COL_STACK_READ1_VALUE_LO),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(control.stack_read1_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ1_ADDR_HI)],
                    value_column: Column(COL_STACK_READ1_VALUE_HI),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(control.stack_read1_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ2_ADDR_LO)],
                    value_column: Column(COL_STACK_READ2_VALUE_LO),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(control.stack_read2_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ2_ADDR_HI)],
                    value_column: Column(COL_STACK_READ2_VALUE_HI),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(control.stack_read2_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_WRITE0_ADDR_LO)],
                    value_column: Column(COL_STACK_WRITE0_VALUE_LO),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(control.stack_write0_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_WRITE0_ADDR_HI)],
                    value_column: Column(COL_STACK_WRITE0_VALUE_HI),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(control.stack_write0_active),
                },
            ],
            is_rom: false,
        },
        WasmMemorySpec {
            name: "call_stack_return_pcs",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![call.call_stack_addr],
                    value_column: call.call_stack_access_return_pc,
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(call.call_stack_push_present),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![call.call_stack_addr],
                    value_column: call.call_stack_access_return_pc,
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(call.call_stack_pop_present),
                },
            ],
            is_rom: false,
        },
        WasmMemorySpec {
            name: "call_stack_caller_fbps",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![call.call_stack_addr],
                    value_column: call.call_stack_access_caller_fbp,
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(call.call_stack_push_present),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![call.call_stack_addr],
                    value_column: call.call_stack_access_caller_fbp,
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(call.call_stack_pop_present),
                },
            ],
            is_rom: false,
        },
        WasmMemorySpec {
            name: "linear_memory",
            columns: vec![
                // Linear-memory rows are split into pure-Read (loads) and
                // RMW Write (stores) per the Nebula-style memory argument:
                // a load emits one Read tuple `(addr, lane*_value, t_r)`
                // — the memory argument enforces it matches the latest
                // write at `addr`, with no write tuple from the load. A
                // store emits a paired Read at `lane*_value_before` and
                // Write at `lane*_value`, so the cells log records the
                // new state only on real stores. This is what stops a
                // malicious prover from corrupting memory via a load row.
                // See `i32_store8_memory_check_rejects_tampered_consistent_prior_state`
                // for the test guarding the store side.
                WasmMemoryColumnSpec {
                    address_columns: vec![linear_memory.lane0_addr],
                    value_column: linear_memory.lane0_value,
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(linear_memory.lane0_load_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![linear_memory.lane1_addr],
                    value_column: linear_memory.lane1_value,
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(linear_memory.lane1_load_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![linear_memory.lane2_addr],
                    value_column: linear_memory.lane2_value,
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(linear_memory.lane2_load_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![linear_memory.lane0_addr],
                    value_column: linear_memory.lane0_value,
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: Some(linear_memory.lane0_value_before),
                    },
                    activation: WasmMemoryActivation::BooleanGate(linear_memory.lane0_store_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![linear_memory.lane1_addr],
                    value_column: linear_memory.lane1_value,
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: Some(linear_memory.lane1_value_before),
                    },
                    activation: WasmMemoryActivation::BooleanGate(linear_memory.lane1_store_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![linear_memory.lane2_addr],
                    value_column: linear_memory.lane2_value,
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: Some(linear_memory.lane2_value_before),
                    },
                    activation: WasmMemoryActivation::BooleanGate(linear_memory.lane2_store_active),
                },
            ],
            is_rom: false,
        },
        WasmMemorySpec {
            name: "locals",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![frame.locals_fbp_before, Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::LocalGet).unwrap(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![frame.locals_fbp_before, Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_LOCAL_WRITE_ENABLED)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![frame.locals_fbp_before, Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(param_init.param_init_active_before),
                },
            ],
            is_rom: false,
        },
        // Parallel high-limb cells log for locals, keyed like `locals`.
        WasmMemorySpec {
            name: "locals_hi",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![frame.locals_fbp_before, Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE_HI),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::LocalGet).unwrap(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![frame.locals_fbp_before, Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE_HI),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_LOCAL_WRITE_ENABLED)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![frame.locals_fbp_before, Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE_HI),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(param_init.param_init_active_before),
                },
            ],
            is_rom: false,
        },
        WasmMemorySpec {
            name: "globals",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_GLOBAL_INDEX)],
                    value_column: Column(COL_GLOBAL_VALUE),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::GlobalGet).unwrap(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_GLOBAL_INDEX)],
                    value_column: Column(COL_GLOBAL_VALUE),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::GlobalSet).unwrap(),
                    )),
                },
            ],
            is_rom: false,
        },
        // Parallel high-limb cells log for globals.
        WasmMemorySpec {
            name: "globals_hi",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_GLOBAL_INDEX)],
                    value_column: Column(COL_GLOBAL_VALUE_HI),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::GlobalGet).unwrap(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_GLOBAL_INDEX)],
                    value_column: Column(COL_GLOBAL_VALUE_HI),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::GlobalSet).unwrap(),
                    )),
                },
            ],
            is_rom: false,
        },
        WasmMemorySpec {
            name: "tables",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_TABLE_ID), Column(COL_TABLE_INDEX)],
                    value_column: Column(COL_TABLE_VALUE),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_TABLE_READ_ENABLED)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_TABLE_ID), Column(COL_TABLE_INDEX)],
                    value_column: Column(COL_TABLE_VALUE),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::TableSet).unwrap(),
                    )),
                },
            ],
            is_rom: false,
        },
        WasmMemorySpec {
            name: "table_sizes",
            // Read by table.size and by call_indirect: the latter binds the
            // authoritative table size for the OOB comparison, so the gate
            // stays on even on a trapping call_indirect row.
            columns: vec![WasmMemoryColumnSpec {
                address_columns: vec![Column(COL_TABLE_ID)],
                value_column: Column(COL_TABLE_SIZE),
                kind: WasmMemoryColumnKind::Read,
                activation: WasmMemoryActivation::BooleanGate(Column(COL_TABLE_SIZE_READ_ENABLED)),
            }],
            is_rom: false,
        },
        rom_read_spec(
            "program_opcodes",
            vec![state.pc_before],
            control.opcode_code,
            WasmMemoryActivation::BooleanGate(control.is_program_row),
        ),
        rom_read_spec(
            "program_local_indices",
            vec![state.pc_before],
            Column(COL_LOCAL_INDEX),
            WasmMemoryActivation::BooleanGate(control.is_program_row),
        ),
        rom_read_spec(
            "program_global_indices",
            vec![state.pc_before],
            Column(COL_GLOBAL_INDEX),
            WasmMemoryActivation::BooleanGate(control.is_program_row),
        ),
        rom_read_spec(
            "program_table_ids",
            vec![state.pc_before],
            Column(COL_TABLE_ID),
            WasmMemoryActivation::BooleanGate(control.is_program_row),
        ),
        rom_read_spec(
            "program_memory_offsets",
            vec![state.pc_before],
            linear_memory.imm_offset,
            WasmMemoryActivation::BooleanGate(control.is_program_row),
        ),
        rom_read_spec(
            "program_call_indirect_type_indices",
            vec![state.pc_before],
            Column(COL_CALL_INDIRECT_TYPE_INDEX),
            WasmMemoryActivation::BooleanGate(control.is_program_row),
        ),
        rom_read_spec(
            "program_call_indirect_expected_type_ids",
            vec![state.pc_before],
            Column(COL_EXPECTED_TYPE_ID),
            WasmMemoryActivation::BooleanGate(control.is_program_row),
        ),
        rom_read_spec(
            "program_i32_const_values",
            vec![state.pc_before],
            Column(COL_STACK_WRITE0_VALUE_LO),
            WasmMemoryActivation::BooleanGate(Column(selector_col(super::isa::WasmOpcode::I32Const).unwrap())),
        ),
        rom_read_spec(
            "program_i64_const_values_lo",
            vec![state.pc_before],
            Column(COL_STACK_WRITE0_VALUE_LO),
            WasmMemoryActivation::BooleanGate(Column(selector_col(super::isa::WasmOpcode::I64Const).unwrap())),
        ),
        rom_read_spec(
            "program_i64_const_values_hi",
            vec![state.pc_before],
            Column(COL_STACK_WRITE0_VALUE_HI),
            WasmMemoryActivation::BooleanGate(Column(selector_col(super::isa::WasmOpcode::I64Const).unwrap())),
        ),
        rom_read_spec(
            "program_ref_func_refs",
            vec![state.pc_before],
            Column(COL_STACK_WRITE0_VALUE_LO),
            WasmMemoryActivation::BooleanGate(Column(selector_col(super::isa::WasmOpcode::RefFunc).unwrap())),
        ),
        rom_read_spec(
            "function_types",
            vec![Column(COL_FUNCTION_REF)],
            Column(COL_FUNCTION_TYPE_ID),
            // NOTE: we don't read the type on direct `call` opcodes because
            // validated wasm guarantees that.
            WasmMemoryActivation::BooleanGate(Column(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE)),
        ),
        rom_read_spec(
            "function_local_counts",
            vec![frame.current_function_ref],
            frame.current_function_num_locals,
            WasmMemoryActivation::BooleanGate(control.is_program_row),
        ),
        rom_read_spec(
            "pc_function_refs",
            vec![state.pc_before],
            frame.current_function_ref,
            WasmMemoryActivation::Always,
        ),
        WasmMemorySpec {
            name: "function_guest_flags",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_FUNCTION_REF)],
                    value_column: Column(COL_TARGET_FUNCTION_IS_GUEST),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::Call).unwrap(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_FUNCTION_REF)],
                    value_column: Column(COL_TARGET_FUNCTION_IS_GUEST),
                    kind: WasmMemoryColumnKind::Read,
                    // De-gated on call_indirect trap rows: no call happens,
                    // so the callee metadata is unread and unconstrained.
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CALL_INDIRECT_IS_NOT_TRAP)),
                },
            ],
            is_rom: true,
        },
        WasmMemorySpec {
            name: "function_param_counts",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_FUNCTION_REF)],
                    value_column: Column(COL_CALL_PARAM_COUNT),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::Call).unwrap(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_FUNCTION_REF)],
                    value_column: Column(COL_CALL_PARAM_COUNT),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CALL_INDIRECT_IS_NOT_TRAP)),
                },
            ],
            is_rom: true,
        },
        WasmMemorySpec {
            name: "function_result_counts",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_FUNCTION_REF)],
                    value_column: Column(COL_CALL_RESULT_COUNT),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::Call).unwrap(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_FUNCTION_REF)],
                    value_column: Column(COL_CALL_RESULT_COUNT),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CALL_INDIRECT_IS_NOT_TRAP)),
                },
            ],
            is_rom: true,
        },
        rom_read_spec(
            "module_types",
            vec![Column(COL_CALL_INDIRECT_TYPE_INDEX)],
            Column(COL_EXPECTED_TYPE_ID),
            WasmMemoryActivation::BooleanGate(Column(selector_col(super::isa::WasmOpcode::CallIndirect).unwrap())),
        ),
        rom_read_spec(
            "call_targets",
            vec![state.pc_before],
            Column(COL_FUNCTION_REF),
            WasmMemoryActivation::BooleanGate(Column(selector_col(super::isa::WasmOpcode::Call).unwrap())),
        ),
        rom_read_spec(
            "function_entries",
            vec![Column(COL_FUNCTION_REF)],
            state.pc_after,
            // De-gated on call_indirect trap rows: a trapping row is
            // terminal and never binds a callee entry pc.
            WasmMemoryActivation::BooleanGate(Column(COL_CALL_INDIRECT_IS_NOT_TRAP)),
        ),
        rom_read_spec(
            "pc_edge_kinds",
            vec![state.pc_before],
            control.pc_edge_kind,
            WasmMemoryActivation::BooleanGate(control.is_program_row),
        ),
        WasmMemorySpec {
            name: "pc_rom",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![state.pc_before, control.control_choice],
                    value_column: state.pc_after,
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(control.pc_rom_active),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![state.pc_before, call.call_stack_return_pc_choice],
                    value_column: call.call_stack_access_return_pc,
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(call.call_stack_push_present),
                },
            ],
            is_rom: true,
        },
    ];

    let cross_step_links = build_ivc_state_continuity_links(&state, &param_init, &call, &frame);

    WasmLookupBindingLayout {
        witness_width: WITNESS_WIDTH,
        lookup_families,
        lookup_bindings,
        memories,
        cross_step_links,
        control,
        state,
        param_init,
        call,
        frame,
        linear_memory,
        sign_extension,
    }
}
