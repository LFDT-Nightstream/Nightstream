use super::isa::{WasmOpTable, WasmOpcode};
use super::ivc_state::{build_ivc_state_continuity_links, WasmCrossStepLinkSpec};
use super::layout::{
    selector_col, Column, COL_CALL_INDIRECT_IS_NOT_TRAP, COL_CALL_INDIRECT_TYPE_INDEX, COL_CALL_STACK_ADDR,
    COL_CALL_STACK_CALLER_FBP_VALUE, COL_CALL_STACK_CALLER_SP_BASE_VALUE, COL_CALL_STACK_POP_PRESENT,
    COL_CALL_STACK_PUSH_PRESENT, COL_CALL_STACK_RETURN_PC_VALUE, COL_CALL_TARGET_METADATA, COL_CI_HOST_CALL,
    COL_CONTROL_CHOICE, COL_CURRENT_FUNCTION_NUM_LOCALS, COL_CURRENT_FUNCTION_REF, COL_EXPECTED_TYPE_ID,
    COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, COL_FUNCTION_REF, COL_FUNCTION_TYPE_ID, COL_GATHER_ACTIVE,
    COL_GATHER_LOCAL_WRITE, COL_GATHER_LOCAL_WRITE_LO, COL_GLOBAL_INDEX, COL_GLOBAL_VALUE, COL_GLOBAL_VALUE_HI,
    COL_GRAMMAR_EVIDX_BEFORE, COL_GRAMMAR_EXIT_LATCH, COL_GRAMMAR_POST_COUNT, COL_GRAMMAR_PRE_COUNT,
    COL_GRAMMAR_SLOT_ARG, COL_GRAMMAR_SLOT_CONST_HI, COL_GRAMMAR_SLOT_CONST_LO, COL_GRAMMAR_SLOT_CURSOR_BEFORE,
    COL_GRAMMAR_SLOT_KIND, COL_GRAMMAR_SLOT_VARIANT, COL_GUEST_ENTRY_ACTIVE, COL_HOST_CALLEE_FREF_AFTER,
    COL_HOST_CALLEE_FREF_BEFORE, COL_HOST_CALL_ACTIVE, COL_IS_PROGRAM_ROW, COL_LINEAR_MEM_ACCESS_BYTE0,
    COL_LINEAR_MEM_ACCESS_BYTE1, COL_LINEAR_MEM_ACCESS_BYTE2, COL_LINEAR_MEM_ACCESS_BYTE3, COL_LINEAR_MEM_ACCESS_BYTE4,
    COL_LINEAR_MEM_ACCESS_BYTE5, COL_LINEAR_MEM_ACCESS_BYTE6, COL_LINEAR_MEM_ACCESS_BYTE7, COL_LINEAR_MEM_BYTE_OFFSET,
    COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0, COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1,
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
    COL_LINEAR_MEM_LANE0_BYTE0, COL_LINEAR_MEM_LANE0_BYTE0_BEFORE, COL_LINEAR_MEM_LANE0_BYTE1,
    COL_LINEAR_MEM_LANE0_BYTE1_BEFORE, COL_LINEAR_MEM_LANE0_BYTE2, COL_LINEAR_MEM_LANE0_BYTE2_BEFORE,
    COL_LINEAR_MEM_LANE0_BYTE3, COL_LINEAR_MEM_LANE0_BYTE3_BEFORE, COL_LINEAR_MEM_LANE1_BYTE0,
    COL_LINEAR_MEM_LANE1_BYTE0_BEFORE, COL_LINEAR_MEM_LANE1_BYTE1, COL_LINEAR_MEM_LANE1_BYTE1_BEFORE,
    COL_LINEAR_MEM_LANE1_BYTE2, COL_LINEAR_MEM_LANE1_BYTE2_BEFORE, COL_LINEAR_MEM_LANE1_BYTE3,
    COL_LINEAR_MEM_LANE1_BYTE3_BEFORE, COL_LINEAR_MEM_LANE2_BYTE0, COL_LINEAR_MEM_LANE2_BYTE0_BEFORE,
    COL_LINEAR_MEM_LANE2_BYTE1, COL_LINEAR_MEM_LANE2_BYTE1_BEFORE, COL_LINEAR_MEM_LANE2_BYTE2,
    COL_LINEAR_MEM_LANE2_BYTE2_BEFORE, COL_LINEAR_MEM_LANE2_BYTE3, COL_LINEAR_MEM_LANE2_BYTE3_BEFORE,
    COL_LINEAR_MEM_LANE_ADDR, COL_LINEAR_MEM_LANE_LOAD_ACTIVE, COL_LINEAR_MEM_LANE_STORE_ACTIVE,
    COL_LINEAR_MEM_LANE_VALUE, COL_LINEAR_MEM_LANE_VALUE_BEFORE, COL_LINEAR_MEM_OFFSET_IS_0,
    COL_LINEAR_MEM_OFFSET_IS_1, COL_LINEAR_MEM_OFFSET_IS_2, COL_LINEAR_MEM_OFFSET_IS_3, COL_LINEAR_MEM_USE_LANE0,
    COL_LINEAR_MEM_USE_LANE1, COL_LINEAR_MEM_USE_LANE2, COL_LOCALS_FBP_BEFORE, COL_LOCAL_INDEX, COL_LOCAL_VALUE,
    COL_LOCAL_VALUE_HI, COL_LOCAL_WRITE_ENABLED, COL_OPCODE_CODE, COL_OP_TABLE_ENABLED, COL_OP_TABLE_ID,
    COL_OUTPUT_CAPTURED, COL_PARAM_INIT_ACTIVE_BEFORE, COL_PC_AFTER, COL_PC_BEFORE, COL_PC_EDGE_KIND,
    COL_PC_FREF_ACTIVE, COL_PC_ROM_ACTIVE, COL_PC_ROM_CALL_RETURN_CHOICE, COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE,
    COL_PROGRAM_GLOBAL_INDEX_ACTIVE, COL_PROGRAM_LOCAL_INDEX_ACTIVE, COL_PROGRAM_TABLE_ID_ACTIVE, COL_SIGN_EXT_BIT,
    COL_SIGN_EXT_LOW7, COL_STACK_READ_ACTIVE, COL_STACK_READ_ADDR_HI, COL_STACK_READ_ADDR_LO, COL_STACK_READ_VALUE_HI,
    COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_ACTIVE, COL_STACK_WRITE0_ADDR_HI, COL_STACK_WRITE0_ADDR_LO,
    COL_STACK_WRITE0_HI_ACTIVE, COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO, COL_TABLE_ID, COL_TABLE_INDEX,
    COL_TABLE_READ_ENABLED, COL_TABLE_SIZE, COL_TABLE_SIZE_READ_ENABLED, COL_TABLE_VALUE, COL_TURN_BOUNDARY,
    COL_TURN_EXPORT_FREF_BEFORE,
};
use super::lookup_semantics::{semantics_for_lookup_family, LookupSemantics};
use super::tables::WasmLookupArity;
use std::sync::OnceLock;

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
    // TODO: Review removing this variant together with its Nebula
    // `MemoryPortActivation::UnlessColumn` lowering; the current WASM layout
    // declares no always-active memory ports.
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmAuxiliaryRelations {
    pub lookup_families: Vec<WasmLookupFamilySpec>,
    pub lookup_bindings: Vec<WasmLookupBindingSpec>,
    pub memories: Vec<WasmMemorySpec>,
    pub ivc_state_links: Vec<WasmCrossStepLinkSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmRelationLayout {
    pub auxiliary: WasmAuxiliaryRelations,
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

pub fn build_wasm_relation_layout() -> &'static WasmRelationLayout {
    static LAYOUT: OnceLock<WasmRelationLayout> = OnceLock::new();
    LAYOUT.get_or_init(build_wasm_relation_layout_uncached)
}

fn build_wasm_relation_layout_uncached() -> WasmRelationLayout {
    let linear_memory = LinearMemoryColumns {
        imm_offset: Column(COL_LINEAR_MEM_IMM_OFFSET),
        byte_offset: Column(COL_LINEAR_MEM_BYTE_OFFSET),
        use_lane0: Column(COL_LINEAR_MEM_USE_LANE0),
        use_lane1: Column(COL_LINEAR_MEM_USE_LANE1),
        use_lane2: Column(COL_LINEAR_MEM_USE_LANE2),
        lane0_load_active: Column(COL_LINEAR_MEM_LANE_LOAD_ACTIVE[0]),
        lane1_load_active: Column(COL_LINEAR_MEM_LANE_LOAD_ACTIVE[1]),
        lane2_load_active: Column(COL_LINEAR_MEM_LANE_LOAD_ACTIVE[2]),
        lane0_store_active: Column(COL_LINEAR_MEM_LANE_STORE_ACTIVE[0]),
        lane1_store_active: Column(COL_LINEAR_MEM_LANE_STORE_ACTIVE[1]),
        lane2_store_active: Column(COL_LINEAR_MEM_LANE_STORE_ACTIVE[2]),
        is_byte_width: Column(COL_LINEAR_MEM_IS_BYTE_WIDTH),
        is_half_width: Column(COL_LINEAR_MEM_IS_HALF_WIDTH),
        is_full_width: Column(COL_LINEAR_MEM_IS_FULL_WIDTH),
        is_double_width: Column(COL_LINEAR_MEM_IS_DOUBLE_WIDTH),
        lane0_addr: Column(COL_LINEAR_MEM_LANE_ADDR[0]),
        lane0_value: Column(COL_LINEAR_MEM_LANE_VALUE[0]),
        lane1_addr: Column(COL_LINEAR_MEM_LANE_ADDR[1]),
        lane1_value: Column(COL_LINEAR_MEM_LANE_VALUE[1]),
        lane2_addr: Column(COL_LINEAR_MEM_LANE_ADDR[2]),
        lane2_value: Column(COL_LINEAR_MEM_LANE_VALUE[2]),
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
        lane0_value_before: Column(COL_LINEAR_MEM_LANE_VALUE_BEFORE[0]),
        lane1_value_before: Column(COL_LINEAR_MEM_LANE_VALUE_BEFORE[1]),
        lane2_value_before: Column(COL_LINEAR_MEM_LANE_VALUE_BEFORE[2]),
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
                    Column(COL_STACK_READ_VALUE_LO[0]),
                    Column(COL_STACK_READ_VALUE_HI[0]),
                    Column(COL_STACK_READ_VALUE_LO[1]),
                    Column(COL_STACK_READ_VALUE_HI[1]),
                    Column(COL_STACK_WRITE0_VALUE_LO),
                    Column(COL_STACK_WRITE0_VALUE_HI),
                ],
                op if op.is_i64_unary() => vec![
                    Column(COL_OP_TABLE_ID),
                    Column(COL_STACK_READ_VALUE_LO[0]),
                    Column(COL_STACK_READ_VALUE_HI[0]),
                    Column(COL_STACK_WRITE0_VALUE_LO),
                    Column(COL_STACK_WRITE0_VALUE_HI),
                ],
                _ => vec![
                    Column(COL_OP_TABLE_ID),
                    Column(COL_STACK_READ_VALUE_LO[0]),
                    Column(COL_STACK_READ_VALUE_LO[1]),
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
                    address_columns: vec![Column(COL_STACK_READ_ADDR_LO[0])],
                    value_column: Column(COL_STACK_READ_VALUE_LO[0]),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_STACK_READ_ACTIVE[0])),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ_ADDR_HI[0])],
                    value_column: Column(COL_STACK_READ_VALUE_HI[0]),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_STACK_READ_ACTIVE[0])),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ_ADDR_LO[0])],
                    value_column: Column(COL_STACK_READ_VALUE_LO[0]),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_OUTPUT_CAPTURED)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ_ADDR_HI[0])],
                    value_column: Column(COL_STACK_READ_VALUE_HI[0]),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_OUTPUT_CAPTURED)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ_ADDR_LO[1])],
                    value_column: Column(COL_STACK_READ_VALUE_LO[1]),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_STACK_READ_ACTIVE[1])),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ_ADDR_HI[1])],
                    value_column: Column(COL_STACK_READ_VALUE_HI[1]),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_STACK_READ_ACTIVE[1])),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ_ADDR_LO[2])],
                    value_column: Column(COL_STACK_READ_VALUE_LO[2]),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_STACK_READ_ACTIVE[2])),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_READ_ADDR_HI[2])],
                    value_column: Column(COL_STACK_READ_VALUE_HI[2]),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_STACK_READ_ACTIVE[2])),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_WRITE0_ADDR_LO)],
                    value_column: Column(COL_STACK_WRITE0_VALUE_LO),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_STACK_WRITE0_ACTIVE)),
                },
                // The hi-word port has its own gate: it fires with the lo
                // port on ordinary writes, and ALONE on result-hi gather
                // rows (which write only the pushed cell's hi lane).
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_STACK_WRITE0_ADDR_HI)],
                    value_column: Column(COL_STACK_WRITE0_VALUE_HI),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_STACK_WRITE0_HI_ACTIVE)),
                },
            ],
            is_rom: false,
        },
        WasmMemorySpec {
            name: "call_stack_return_pcs",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_CALL_STACK_ADDR)],
                    value_column: Column(COL_CALL_STACK_RETURN_PC_VALUE),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CALL_STACK_PUSH_PRESENT)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_CALL_STACK_ADDR)],
                    value_column: Column(COL_CALL_STACK_RETURN_PC_VALUE),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CALL_STACK_POP_PRESENT)),
                },
            ],
            is_rom: false,
        },
        WasmMemorySpec {
            name: "call_stack_caller_fbps",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_CALL_STACK_ADDR)],
                    value_column: Column(COL_CALL_STACK_CALLER_FBP_VALUE),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CALL_STACK_PUSH_PRESENT)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_CALL_STACK_ADDR)],
                    value_column: Column(COL_CALL_STACK_CALLER_FBP_VALUE),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CALL_STACK_POP_PRESENT)),
                },
            ],
            is_rom: false,
        },
        WasmMemorySpec {
            name: "call_stack_caller_sp_bases",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_CALL_STACK_ADDR)],
                    value_column: Column(COL_CALL_STACK_CALLER_SP_BASE_VALUE),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CALL_STACK_PUSH_PRESENT)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_CALL_STACK_ADDR)],
                    value_column: Column(COL_CALL_STACK_CALLER_SP_BASE_VALUE),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CALL_STACK_POP_PRESENT)),
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
                WasmMemoryColumnSpec {
                    address_columns: vec![linear_memory.lane0_addr],
                    value_column: linear_memory.lane0_value,
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        crate::ccs::host_event_chain::gather_memory_read_kind_col(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![linear_memory.lane0_addr],
                    value_column: linear_memory.lane0_value,
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: Some(linear_memory.lane0_value_before),
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        crate::ccs::host_event_chain::gather_memory_write_kind_col(),
                    )),
                },
            ],
            is_rom: false,
        },
        WasmMemorySpec {
            name: "locals",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_LOCALS_FBP_BEFORE), Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::LocalGet).unwrap(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_LOCALS_FBP_BEFORE), Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        crate::ccs::host_event_chain::gather_memory_local_base_col(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_LOCALS_FBP_BEFORE), Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_LOCAL_WRITE_ENABLED)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_LOCALS_FBP_BEFORE), Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_PARAM_INIT_ACTIVE_BEFORE)),
                },
                // Input bootstrap: lo-lane entry gather rows write the
                // claim-input word into the entry frame's locals (kind 4).
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_LOCALS_FBP_BEFORE), Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_GATHER_LOCAL_WRITE_LO)),
                },
            ],
            is_rom: false,
        },
        // Parallel high-limb cells log for locals, keyed like `locals`.
        WasmMemorySpec {
            name: "locals_hi",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_LOCALS_FBP_BEFORE), Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE_HI),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::LocalGet).unwrap(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_LOCALS_FBP_BEFORE), Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE_HI),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_LOCAL_WRITE_ENABLED)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_LOCALS_FBP_BEFORE), Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE_HI),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_PARAM_INIT_ACTIVE_BEFORE)),
                },
                // Input bootstrap: every input-local row writes the hi
                // lane — zero on lo rows (total write), the claim word on
                // hi rows (the CCS pins the value column either way).
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_LOCALS_FBP_BEFORE), Column(COL_LOCAL_INDEX)],
                    value_column: Column(COL_LOCAL_VALUE_HI),
                    kind: WasmMemoryColumnKind::Write {
                        value_before_column: None,
                    },
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_GATHER_LOCAL_WRITE)),
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
            vec![Column(COL_PC_BEFORE)],
            Column(COL_OPCODE_CODE),
            WasmMemoryActivation::BooleanGate(Column(COL_IS_PROGRAM_ROW)),
        ),
        rom_read_spec(
            "program_local_indices",
            vec![Column(COL_PC_BEFORE)],
            Column(COL_LOCAL_INDEX),
            WasmMemoryActivation::BooleanGate(Column(COL_PROGRAM_LOCAL_INDEX_ACTIVE)),
        ),
        rom_read_spec(
            "program_global_indices",
            vec![Column(COL_PC_BEFORE)],
            Column(COL_GLOBAL_INDEX),
            WasmMemoryActivation::BooleanGate(Column(COL_PROGRAM_GLOBAL_INDEX_ACTIVE)),
        ),
        rom_read_spec(
            "program_table_ids",
            vec![Column(COL_PC_BEFORE)],
            Column(COL_TABLE_ID),
            WasmMemoryActivation::BooleanGate(Column(COL_PROGRAM_TABLE_ID_ACTIVE)),
        ),
        rom_read_spec(
            "program_memory_offsets",
            vec![Column(COL_PC_BEFORE)],
            linear_memory.imm_offset,
            WasmMemoryActivation::BooleanGate(linear_memory.use_lane0),
        ),
        rom_read_spec(
            "program_call_indirect_type_indices",
            vec![Column(COL_PC_BEFORE)],
            Column(COL_CALL_INDIRECT_TYPE_INDEX),
            WasmMemoryActivation::BooleanGate(Column(COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE)),
        ),
        rom_read_spec(
            "program_call_indirect_expected_type_ids",
            vec![Column(COL_PC_BEFORE)],
            Column(COL_EXPECTED_TYPE_ID),
            WasmMemoryActivation::BooleanGate(Column(COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE)),
        ),
        rom_read_spec(
            "program_i32_const_values",
            vec![Column(COL_PC_BEFORE)],
            Column(COL_STACK_WRITE0_VALUE_LO),
            WasmMemoryActivation::BooleanGate(Column(selector_col(super::isa::WasmOpcode::I32Const).unwrap())),
        ),
        rom_read_spec(
            "program_i64_const_values_lo",
            vec![Column(COL_PC_BEFORE)],
            Column(COL_STACK_WRITE0_VALUE_LO),
            WasmMemoryActivation::BooleanGate(Column(selector_col(super::isa::WasmOpcode::I64Const).unwrap())),
        ),
        rom_read_spec(
            "program_i64_const_values_hi",
            vec![Column(COL_PC_BEFORE)],
            Column(COL_STACK_WRITE0_VALUE_HI),
            WasmMemoryActivation::BooleanGate(Column(selector_col(super::isa::WasmOpcode::I64Const).unwrap())),
        ),
        rom_read_spec(
            "program_ref_func_refs",
            vec![Column(COL_PC_BEFORE)],
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
            vec![Column(COL_CURRENT_FUNCTION_REF)],
            Column(COL_CURRENT_FUNCTION_NUM_LOCALS),
            WasmMemoryActivation::BooleanGate(Column(COL_IS_PROGRAM_ROW)),
        ),
        rom_read_spec(
            "pc_function_refs",
            vec![Column(COL_PC_BEFORE)],
            Column(COL_CURRENT_FUNCTION_REF),
            // Program and frame-transition rows only. Gather rows carry the
            // one-past-the-end pc, while permutation, turn-boundary, and
            // padding rows do not consume frame identity either.
            WasmMemoryActivation::BooleanGate(Column(COL_PC_FREF_ACTIVE)),
        ),
        WasmMemorySpec {
            name: "function_call_metadata",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_FUNCTION_REF)],
                    value_column: Column(COL_CALL_TARGET_METADATA),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::Call).unwrap(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_FUNCTION_REF)],
                    value_column: Column(COL_CALL_TARGET_METADATA),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(
                        selector_col(super::isa::WasmOpcode::ReturnCall).unwrap(),
                    )),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_FUNCTION_REF)],
                    value_column: Column(COL_CALL_TARGET_METADATA),
                    kind: WasmMemoryColumnKind::Read,
                    // De-gated on call_indirect trap rows: no call happens,
                    // so the callee metadata is unread and unconstrained.
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CALL_INDIRECT_IS_NOT_TRAP)),
                },
            ],
            is_rom: true,
        },
        WasmMemorySpec {
            name: "module_types",
            columns: [WasmOpcode::CallIndirect, WasmOpcode::ReturnCallIndirect]
                .into_iter()
                .map(|opcode| WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_CALL_INDIRECT_TYPE_INDEX)],
                    value_column: Column(COL_EXPECTED_TYPE_ID),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(selector_col(opcode).unwrap())),
                })
                .collect(),
            is_rom: true,
        },
        WasmMemorySpec {
            name: "call_targets",
            columns: [WasmOpcode::Call, WasmOpcode::ReturnCall]
                .into_iter()
                .map(|opcode| WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_PC_BEFORE)],
                    value_column: Column(COL_FUNCTION_REF),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(selector_col(opcode).unwrap())),
                })
                .collect(),
            is_rom: true,
        },
        WasmMemorySpec {
            name: "function_entries",
            columns: vec![
                // Gated on guest-call rows only: host imports have no entry
                // pc (host calls fall through to pc+1, pinned by a CCS row),
                // and a trapping call_indirect row is terminal and never
                // binds a callee entry pc.
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_FUNCTION_REF)],
                    value_column: Column(COL_PC_AFTER),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_GUEST_ENTRY_ACTIVE)),
                },
                // Turn boundary: the next turn's pc jump is bound to the
                // entered export's entry pc, keyed by the repointed
                // attribution.
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_HOST_CALLEE_FREF_AFTER)],
                    value_column: Column(COL_PC_AFTER),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_TURN_BOUNDARY)),
                },
            ],
            is_rom: true,
        },
        rom_read_spec(
            "pc_edge_kinds",
            vec![Column(COL_PC_BEFORE)],
            Column(COL_PC_EDGE_KIND),
            WasmMemoryActivation::BooleanGate(Column(COL_IS_PROGRAM_ROW)),
        ),
        // Event-template ROMs (see `docs/host-event-grammar-tables.md` §3.4):
        // per-slot source descriptors keyed by (fref, event index, slot
        // cursor) read on gather rows, and per-import event counts read on
        // grammar call/result rows. Content is generated from the embedder's
        // `HostEventGrammar` (see `event_grammar::preload_grammar_tables`).
        rom_read_spec(
            "grammar_slot_kind",
            vec![
                Column(COL_HOST_CALLEE_FREF_BEFORE),
                Column(COL_GRAMMAR_EVIDX_BEFORE),
                Column(COL_GRAMMAR_SLOT_CURSOR_BEFORE),
            ],
            Column(COL_GRAMMAR_SLOT_KIND),
            WasmMemoryActivation::BooleanGate(Column(COL_GATHER_ACTIVE)),
        ),
        rom_read_spec(
            "grammar_slot_arg",
            vec![
                Column(COL_HOST_CALLEE_FREF_BEFORE),
                Column(COL_GRAMMAR_EVIDX_BEFORE),
                Column(COL_GRAMMAR_SLOT_CURSOR_BEFORE),
            ],
            Column(COL_GRAMMAR_SLOT_ARG),
            WasmMemoryActivation::BooleanGate(Column(COL_GATHER_ACTIVE)),
        ),
        rom_read_spec(
            "grammar_slot_variant",
            vec![
                Column(COL_HOST_CALLEE_FREF_BEFORE),
                Column(COL_GRAMMAR_EVIDX_BEFORE),
                Column(COL_GRAMMAR_SLOT_CURSOR_BEFORE),
            ],
            Column(COL_GRAMMAR_SLOT_VARIANT),
            WasmMemoryActivation::BooleanGate(Column(COL_GATHER_ACTIVE)),
        ),
        rom_read_spec(
            "grammar_slot_const_lo",
            vec![
                Column(COL_HOST_CALLEE_FREF_BEFORE),
                Column(COL_GRAMMAR_EVIDX_BEFORE),
                Column(COL_GRAMMAR_SLOT_CURSOR_BEFORE),
            ],
            Column(COL_GRAMMAR_SLOT_CONST_LO),
            WasmMemoryActivation::BooleanGate(Column(COL_GATHER_ACTIVE)),
        ),
        rom_read_spec(
            "grammar_slot_const_hi",
            vec![
                Column(COL_HOST_CALLEE_FREF_BEFORE),
                Column(COL_GRAMMAR_EVIDX_BEFORE),
                Column(COL_GRAMMAR_SLOT_CURSOR_BEFORE),
            ],
            Column(COL_GRAMMAR_SLOT_CONST_HI),
            WasmMemoryActivation::BooleanGate(Column(COL_GATHER_ACTIVE)),
        ),
        // Import pre-counts and export entry-counts are SEPARATE families so
        // a PRE_COUNT read can never land on the other kind's cell: turn
        // boundaries and exit latches see only export entry cells, while
        // host-call rows see only import pre-count cells. These cells store
        // count + 1 ("presence bias"): an undeclared fref
        // reads 0 and the load rows subtract 1, poisoning the schedule to
        // EVREM = -1 = p-1 (EVREM is field-width; the row itself stays
        // satisfiable). Each completed event block decrements EVREM and
        // increments EVIDX. While EVREM is nonzero, program, result, and
        // boundary rows cannot execute, so EVIDX cannot reset. Every gather
        // uses EVIDX as an active grammar-ROM address component; the Nebula
        // memory binding range-proves address components to at most 32 bits
        // (and to the family's narrower configured width). Field addition
        // does not wrap at 2^32, so after at most 2^32 blocks the next gather
        // is unsatisfiable while EVREM remains nonzero. The trace therefore
        // cannot halt, enforcing template presence in the composed circuit
        // without preprocessing validation.
        WasmMemorySpec {
            name: "grammar_import_pre_counts",
            columns: vec![WasmMemoryColumnSpec {
                address_columns: vec![Column(COL_FUNCTION_REF)],
                value_column: Column(COL_GRAMMAR_PRE_COUNT),
                kind: WasmMemoryColumnKind::Read,
                activation: WasmMemoryActivation::BooleanGate(Column(COL_HOST_CALL_ACTIVE)),
            }],
            is_rom: true,
        },
        WasmMemorySpec {
            name: "grammar_export_entry_counts",
            columns: vec![
                // Exit latch: re-reads the export's entry count to continue
                // the event numbering for exit events.
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_TURN_EXPORT_FREF_BEFORE)],
                    value_column: Column(COL_GRAMMAR_PRE_COUNT),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_GRAMMAR_EXIT_LATCH)),
                },
                // Turn boundary: loads the entered export's entry-event
                // count as the next turn's owed schedule.
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_HOST_CALLEE_FREF_AFTER)],
                    value_column: Column(COL_GRAMMAR_PRE_COUNT),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_TURN_BOUNDARY)),
                },
            ],
            is_rom: true,
        },
        // Exit latch: the export's exit-event count. Raw (no presence
        // bias): the turn's export fref was bound at entry.
        rom_read_spec(
            "grammar_export_exit_counts",
            vec![Column(COL_TURN_EXPORT_FREF_BEFORE)],
            Column(COL_GRAMMAR_POST_COUNT),
            WasmMemoryActivation::BooleanGate(Column(COL_GRAMMAR_EXIT_LATCH)),
        ),
        WasmMemorySpec {
            name: "pc_rom",
            columns: vec![
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_PC_BEFORE), Column(COL_CONTROL_CHOICE)],
                    value_column: Column(COL_PC_AFTER),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_PC_ROM_ACTIVE)),
                },
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_PC_BEFORE), Column(COL_PC_ROM_CALL_RETURN_CHOICE)],
                    value_column: Column(COL_CALL_STACK_RETURN_PC_VALUE),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CALL_STACK_PUSH_PRESENT)),
                },
                // An indirect host call binds its fall-through pc_after to
                // the call site's return-pc slot: host imports have no
                // `function_entries` entry, and the DynamicCallIndirect edge
                // kind bypasses the static pc ROM read.
                WasmMemoryColumnSpec {
                    address_columns: vec![Column(COL_PC_BEFORE), Column(COL_PC_ROM_CALL_RETURN_CHOICE)],
                    value_column: Column(COL_PC_AFTER),
                    kind: WasmMemoryColumnKind::Read,
                    activation: WasmMemoryActivation::BooleanGate(Column(COL_CI_HOST_CALL)),
                },
            ],
            is_rom: true,
        },
    ];

    let ivc_state_links = build_ivc_state_continuity_links();

    WasmRelationLayout {
        auxiliary: WasmAuxiliaryRelations {
            lookup_families,
            lookup_bindings,
            memories,
            ivc_state_links,
        },
        linear_memory,
        sign_extension,
    }
}
