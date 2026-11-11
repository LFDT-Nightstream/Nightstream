use super::isa::WasmOpcode;
use super::ivc_state::build_ivc_state_continuity_links;
use super::layout::{
    selector_col, Column, COL_CALL_STACK_ADDR, COL_CALL_STACK_CALLER_FBP_VALUE, COL_CALL_STACK_CALLER_SP_BASE_VALUE,
    COL_CALL_STACK_POP_PRESENT, COL_CALL_STACK_PUSH_PRESENT, COL_CALL_STACK_RETURN_PC_VALUE, COL_CALL_TARGET_METADATA,
    COL_CURRENT_FUNCTION_REF, COL_EDGE_ADDR, COL_ENTERED_FUNCTION_NUM_LOCALS, COL_EXIT_SLIDE_DISTANCE,
    COL_EXPECTED_TYPE_ID, COL_FRAME_MOVE_DISTANCE_AFTER, COL_FRAME_MOVE_REMAINING_AFTER, COL_FRAME_SLIDE_ACTIVE,
    COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, COL_FUNCTION_REF, COL_FUNCTION_TYPE_ID, COL_GATHER_ACTIVE,
    COL_GATHER_LOCAL_WRITE, COL_GATHER_LOCAL_WRITE_LO, COL_GLOBAL_INDEX, COL_GLOBAL_VALUE, COL_GLOBAL_VALUE_HI,
    COL_GUEST_ENTRY_ACTIVE, COL_HOST_CALLEE_FREF_AFTER, COL_HOST_CALL_ACTIVE, COL_HOST_EVENT_EXIT_LATCH,
    COL_HOST_EVENT_INDEX_BEFORE, COL_HOST_EVENT_SLOT_ARG, COL_HOST_EVENT_SLOT_CURSOR_BEFORE,
    COL_HOST_EVENT_SLOT_IMMEDIATE0, COL_HOST_EVENT_SLOT_IMMEDIATE1, COL_HOST_EVENT_SLOT_KIND,
    COL_HOST_EVENT_SLOT_VARIANT, COL_IS_PROGRAM_ROW, COL_LINEAR_MEM_ACCESS_BYTE0, COL_LINEAR_MEM_ACCESS_BYTE1,
    COL_LINEAR_MEM_ACCESS_BYTE2, COL_LINEAR_MEM_ACCESS_BYTE3, COL_LINEAR_MEM_ACCESS_BYTE4, COL_LINEAR_MEM_ACCESS_BYTE5,
    COL_LINEAR_MEM_ACCESS_BYTE6, COL_LINEAR_MEM_ACCESS_BYTE7, COL_LINEAR_MEM_BYTE_OFFSET,
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
    COL_LINEAR_MEM_USE_LANE1, COL_LINEAR_MEM_USE_LANE2, COL_LOCAL_ADDR_HI, COL_LOCAL_ADDR_LO, COL_LOCAL_INDEX,
    COL_LOCAL_VALUE, COL_LOCAL_VALUE_HI, COL_LOCAL_WRITE_ENABLED, COL_OPCODE_CODE, COL_OUTPUT_CAPTURED, COL_PC_AFTER,
    COL_PC_BEFORE, COL_PC_EDGE_KIND, COL_PC_FREF_ACTIVE, COL_PC_ROM_ACTIVE,
    COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE, COL_PROGRAM_GLOBAL_INDEX_ACTIVE, COL_PROGRAM_LOCAL_INDEX_ACTIVE,
    COL_PROGRAM_TABLE_ID_ACTIVE, COL_SIGN_EXT_BIT, COL_SIGN_EXT_LOW7, COL_STACK_READ_ACTIVE, COL_STACK_READ_ADDR_HI,
    COL_STACK_READ_ADDR_LO, COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_ADDR_HI,
    COL_STACK_WRITE0_ADDR_LO, COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO, COL_TABLE_ID, COL_TABLE_INDEX,
    COL_TABLE_READ_ENABLED, COL_TABLE_SET_LIVE, COL_TABLE_SIZE, COL_TABLE_SIZE_READ_ENABLED, COL_TABLE_VALUE,
    COL_TURN_BOUNDARY, COL_TURN_EXPORT_FREF_BEFORE,
};
use super::WasmMemoryId;
use neo_application::{
    ContinuityCatalog, MemoryCatalog, MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec,
};
use std::sync::OnceLock;

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
    /// The tables: each memory's ports under `WasmMemoryId::table`.
    pub memory: MemoryCatalog<WasmMemoryId>,
    /// The memories before the overlay, one for each declaration.
    pub logical_memory: Vec<MemorySpec<WasmMemoryId>>,
    pub continuity: ContinuityCatalog,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmRelationLayout {
    pub auxiliary: WasmAuxiliaryRelations,
    pub linear_memory: LinearMemoryColumns,
    pub sign_extension: SignExtensionColumns,
}

fn rom_read_spec(
    id: WasmMemoryId,
    address_columns: Vec<Column>,
    value_column: Column,
    activation: MemoryPortActivation,
) -> MemorySpec<WasmMemoryId> {
    MemorySpec {
        id,
        kind: MemoryKind::Rom,
        ports: vec![MemoryPortSpec {
            address_columns: address_columns.into_iter().map(|column| column.0).collect(),
            value_column: value_column.0,
            kind: MemoryPortKind::Read,
            activation,
        }],
    }
}

fn memory_columns<const N: usize>(columns: [Column; N]) -> Vec<usize> {
    columns.into_iter().map(|column| column.0).collect()
}

const fn active_when(column: Column) -> MemoryPortActivation {
    MemoryPortActivation::When(column.0)
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

    let mut memories = vec![
        MemorySpec {
            id: WasmMemoryId::Stack,
            kind: MemoryKind::Ram,
            ports: vec![
                // SPEC.md section 7.6 binds slots to this event order. Keep
                // each limb's order, with low-limb ports before high-limb ports.
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_STACK_READ_ADDR_LO[0])]),
                    value_column: COL_STACK_READ_VALUE_LO[0],
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(crate::layout::COL_STACK_SEPARATE[0])),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_STACK_READ_ADDR_LO[0])]),
                    value_column: COL_STACK_READ_VALUE_LO[0],
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_OUTPUT_CAPTURED)),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_STACK_READ_ADDR_LO[1])]),
                    value_column: COL_STACK_READ_VALUE_LO[1],
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_STACK_READ_ACTIVE[1])),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_STACK_READ_ADDR_LO[2])]),
                    value_column: COL_STACK_READ_VALUE_LO[2],
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_STACK_READ_ACTIVE[2])),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_STACK_WRITE0_ADDR_LO)]),
                    value_column: COL_STACK_WRITE0_VALUE_LO,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(crate::layout::COL_STACK_SEPARATE[1])),
                },
                // SPEC.md section 7.6: keep replacement before the local write.
                MemoryPortSpec {
                    address_columns: vec![COL_STACK_WRITE0_ADDR_LO],
                    value_column: COL_STACK_WRITE0_VALUE_LO,
                    kind: MemoryPortKind::Write {
                        value_before_column: Some(COL_STACK_READ_VALUE_LO[0]),
                    },
                    activation: active_when(Column(crate::layout::COL_STACK_RMW)),
                },
                // Locals live in their frame at `2 * (fbp + index)`.
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_LOCAL_ADDR_LO)]),
                    value_column: COL_LOCAL_VALUE,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(selector_col(super::isa::WasmOpcode::LocalGet).unwrap())),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_LOCAL_ADDR_LO)]),
                    value_column: COL_LOCAL_VALUE,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(crate::ccs::host_event_chain::gather_memory_local_base_col())),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_LOCAL_ADDR_LO)]),
                    value_column: COL_LOCAL_VALUE,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(COL_LOCAL_WRITE_ENABLED)),
                },
                // Input bootstrap: lo-lane entry gather rows write the
                // entry-input word into the entry frame's locals.
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_LOCAL_ADDR_LO)]),
                    value_column: COL_LOCAL_VALUE,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(COL_GATHER_LOCAL_WRITE_LO)),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_STACK_READ_ADDR_HI[0])]),
                    value_column: COL_STACK_READ_VALUE_HI[0],
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(crate::layout::COL_STACK_SEPARATE[0])),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_STACK_READ_ADDR_HI[0])]),
                    value_column: COL_STACK_READ_VALUE_HI[0],
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_OUTPUT_CAPTURED)),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_STACK_READ_ADDR_HI[1])]),
                    value_column: COL_STACK_READ_VALUE_HI[1],
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_STACK_READ_ACTIVE[1])),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_STACK_READ_ADDR_HI[2])]),
                    value_column: COL_STACK_READ_VALUE_HI[2],
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_STACK_READ_ACTIVE[2])),
                },
                // The hi-word port has its own gate: it fires with the lo
                // port on ordinary writes, and ALONE on result-hi gather
                // rows (which write only the pushed cell's hi lane).
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_STACK_WRITE0_ADDR_HI)]),
                    value_column: COL_STACK_WRITE0_VALUE_HI,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(crate::layout::COL_STACK_SEPARATE[2])),
                },
                // SPEC.md section 7.6: keep replacement before the local write.
                MemoryPortSpec {
                    address_columns: vec![COL_STACK_WRITE0_ADDR_HI],
                    value_column: COL_STACK_WRITE0_VALUE_HI,
                    kind: MemoryPortKind::Write {
                        value_before_column: Some(COL_STACK_READ_VALUE_HI[0]),
                    },
                    activation: active_when(Column(crate::layout::COL_STACK_RMW)),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_LOCAL_ADDR_HI)]),
                    value_column: COL_LOCAL_VALUE_HI,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(crate::ccs::host_event_chain::gather_memory_local_base_col())),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_LOCAL_ADDR_HI)]),
                    value_column: COL_LOCAL_VALUE_HI,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(selector_col(super::isa::WasmOpcode::LocalGet).unwrap())),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_LOCAL_ADDR_HI)]),
                    value_column: COL_LOCAL_VALUE_HI,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(COL_LOCAL_WRITE_ENABLED)),
                },
                // Input bootstrap: every input-local row writes the hi
                // lane — zero on lo rows (total write), the input word on
                // hi rows (the CCS pins the value column either way).
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_LOCAL_ADDR_HI)]),
                    value_column: COL_LOCAL_VALUE_HI,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(COL_GATHER_LOCAL_WRITE)),
                },
            ],
        },
        MemorySpec {
            id: WasmMemoryId::CallStackReturnPc,
            kind: MemoryKind::Ram,
            ports: vec![
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_CALL_STACK_ADDR)]),
                    value_column: COL_CALL_STACK_RETURN_PC_VALUE,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(COL_CALL_STACK_PUSH_PRESENT)),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_CALL_STACK_ADDR)]),
                    value_column: COL_CALL_STACK_RETURN_PC_VALUE,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_CALL_STACK_POP_PRESENT)),
                },
            ],
        },
        MemorySpec {
            id: WasmMemoryId::CallStackCallerFbp,
            kind: MemoryKind::Ram,
            ports: vec![
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_CALL_STACK_ADDR)]),
                    value_column: COL_CALL_STACK_CALLER_FBP_VALUE,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(COL_CALL_STACK_PUSH_PRESENT)),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_CALL_STACK_ADDR)]),
                    value_column: COL_CALL_STACK_CALLER_FBP_VALUE,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_CALL_STACK_POP_PRESENT)),
                },
            ],
        },
        MemorySpec {
            id: WasmMemoryId::CallStackCallerSpBase,
            kind: MemoryKind::Ram,
            ports: vec![
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_CALL_STACK_ADDR)]),
                    value_column: COL_CALL_STACK_CALLER_SP_BASE_VALUE,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(COL_CALL_STACK_PUSH_PRESENT)),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_CALL_STACK_ADDR)]),
                    value_column: COL_CALL_STACK_CALLER_SP_BASE_VALUE,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_CALL_STACK_POP_PRESENT)),
                },
            ],
        },
        MemorySpec {
            id: WasmMemoryId::LinearMemory,
            kind: MemoryKind::Ram,
            ports: vec![
                // Linear-memory rows are split into pure-Read (loads) and
                // RMW Write (stores) for the memory-consistency argument:
                // a load emits one Read tuple `(addr, lane*_value, t_r)`
                // — the memory argument enforces it matches the latest
                // write at `addr`, with no write tuple from the load. A
                // store emits a paired Read at `lane*_value_before` and
                // Write at `lane*_value`, so the cells log records the
                // new state only on real stores. This is what stops a
                // malicious prover from corrupting memory via a load row.
                // See `i32_store8_memory_check_rejects_tampered_consistent_prior_state`
                // for the test guarding the store side.
                MemoryPortSpec {
                    address_columns: memory_columns([linear_memory.lane0_addr]),
                    value_column: linear_memory.lane0_value.0,
                    kind: MemoryPortKind::Read,
                    activation: active_when(linear_memory.lane0_load_active),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([linear_memory.lane1_addr]),
                    value_column: linear_memory.lane1_value.0,
                    kind: MemoryPortKind::Read,
                    activation: active_when(linear_memory.lane1_load_active),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([linear_memory.lane2_addr]),
                    value_column: linear_memory.lane2_value.0,
                    kind: MemoryPortKind::Read,
                    activation: active_when(linear_memory.lane2_load_active),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([linear_memory.lane0_addr]),
                    value_column: linear_memory.lane0_value.0,
                    kind: MemoryPortKind::Write {
                        value_before_column: Some(linear_memory.lane0_value_before.0),
                    },
                    activation: active_when(linear_memory.lane0_store_active),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([linear_memory.lane1_addr]),
                    value_column: linear_memory.lane1_value.0,
                    kind: MemoryPortKind::Write {
                        value_before_column: Some(linear_memory.lane1_value_before.0),
                    },
                    activation: active_when(linear_memory.lane1_store_active),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([linear_memory.lane2_addr]),
                    value_column: linear_memory.lane2_value.0,
                    kind: MemoryPortKind::Write {
                        value_before_column: Some(linear_memory.lane2_value_before.0),
                    },
                    activation: active_when(linear_memory.lane2_store_active),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([linear_memory.lane0_addr]),
                    value_column: linear_memory.lane0_value.0,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(crate::ccs::host_event_chain::gather_memory_read_kind_col())),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([linear_memory.lane0_addr]),
                    value_column: linear_memory.lane0_value.0,
                    kind: MemoryPortKind::Write {
                        value_before_column: Some(linear_memory.lane0_value_before.0),
                    },
                    activation: active_when(Column(crate::ccs::host_event_chain::gather_memory_write_kind_col())),
                },
            ],
        },
        MemorySpec {
            id: WasmMemoryId::GlobalLo,
            kind: MemoryKind::Ram,
            ports: vec![
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_GLOBAL_INDEX)]),
                    value_column: COL_GLOBAL_VALUE,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(selector_col(super::isa::WasmOpcode::GlobalGet).unwrap())),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_GLOBAL_INDEX)]),
                    value_column: COL_GLOBAL_VALUE,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(selector_col(super::isa::WasmOpcode::GlobalSet).unwrap())),
                },
            ],
        },
        // Parallel high-limb cells log for globals.
        MemorySpec {
            id: WasmMemoryId::GlobalHi,
            kind: MemoryKind::Ram,
            ports: vec![
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_GLOBAL_INDEX)]),
                    value_column: COL_GLOBAL_VALUE_HI,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(selector_col(super::isa::WasmOpcode::GlobalGet).unwrap())),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_GLOBAL_INDEX)]),
                    value_column: COL_GLOBAL_VALUE_HI,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(selector_col(super::isa::WasmOpcode::GlobalSet).unwrap())),
                },
            ],
        },
        MemorySpec {
            id: WasmMemoryId::TableElement,
            kind: MemoryKind::Ram,
            ports: vec![
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_TABLE_ID), Column(COL_TABLE_INDEX)]),
                    value_column: COL_TABLE_VALUE,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_TABLE_READ_ENABLED)),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_TABLE_ID), Column(COL_TABLE_INDEX)]),
                    value_column: COL_TABLE_VALUE,
                    kind: MemoryPortKind::Write {
                        value_before_column: None,
                    },
                    activation: active_when(Column(COL_TABLE_SET_LIVE)),
                },
            ],
        },
        MemorySpec {
            id: WasmMemoryId::TableSize,
            kind: MemoryKind::Ram,
            // Read by table.size and by call_indirect's resolve step: the latter
            // binds the authoritative table size for the OOB comparison, so the
            // gate stays on even on a trapping resolve row.
            ports: vec![MemoryPortSpec {
                address_columns: memory_columns([Column(COL_TABLE_ID)]),
                value_column: COL_TABLE_SIZE,
                kind: MemoryPortKind::Read,
                activation: active_when(Column(COL_TABLE_SIZE_READ_ENABLED)),
            }],
        },
        rom_read_spec(
            WasmMemoryId::ProgramOpcode,
            vec![Column(COL_PC_BEFORE)],
            Column(COL_OPCODE_CODE),
            active_when(Column(COL_IS_PROGRAM_ROW)),
        ),
        rom_read_spec(
            WasmMemoryId::ProgramLocalIndex,
            vec![Column(COL_PC_BEFORE)],
            Column(COL_LOCAL_INDEX),
            active_when(Column(COL_PROGRAM_LOCAL_INDEX_ACTIVE)),
        ),
        rom_read_spec(
            WasmMemoryId::ProgramGlobalIndex,
            vec![Column(COL_PC_BEFORE)],
            Column(COL_GLOBAL_INDEX),
            active_when(Column(COL_PROGRAM_GLOBAL_INDEX_ACTIVE)),
        ),
        rom_read_spec(
            WasmMemoryId::ProgramTableId,
            vec![Column(COL_PC_BEFORE)],
            Column(COL_TABLE_ID),
            active_when(Column(COL_PROGRAM_TABLE_ID_ACTIVE)),
        ),
        rom_read_spec(
            WasmMemoryId::ProgramMemoryOffset,
            vec![Column(COL_PC_BEFORE)],
            linear_memory.imm_offset,
            active_when(linear_memory.use_lane0),
        ),
        rom_read_spec(
            WasmMemoryId::ProgramCallIndirectExpectedTypeId,
            vec![Column(COL_PC_BEFORE)],
            Column(COL_EXPECTED_TYPE_ID),
            active_when(Column(COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE)),
        ),
        rom_read_spec(
            WasmMemoryId::ProgramI32ConstValue,
            vec![Column(COL_PC_BEFORE)],
            Column(COL_STACK_WRITE0_VALUE_LO),
            active_when(Column(selector_col(super::isa::WasmOpcode::I32Const).unwrap())),
        ),
        rom_read_spec(
            WasmMemoryId::ProgramI64ConstValueLo,
            vec![Column(COL_PC_BEFORE)],
            Column(COL_STACK_WRITE0_VALUE_LO),
            active_when(Column(selector_col(super::isa::WasmOpcode::I64Const).unwrap())),
        ),
        rom_read_spec(
            WasmMemoryId::ProgramI64ConstValueHi,
            vec![Column(COL_PC_BEFORE)],
            Column(COL_STACK_WRITE0_VALUE_HI),
            active_when(Column(selector_col(super::isa::WasmOpcode::I64Const).unwrap())),
        ),
        rom_read_spec(
            WasmMemoryId::ProgramRefFuncRef,
            vec![Column(COL_PC_BEFORE)],
            Column(COL_STACK_WRITE0_VALUE_LO),
            active_when(Column(selector_col(super::isa::WasmOpcode::RefFunc).unwrap())),
        ),
        rom_read_spec(
            WasmMemoryId::FunctionType,
            vec![Column(COL_FUNCTION_REF)],
            Column(COL_FUNCTION_TYPE_ID),
            // NOTE: we don't read the type on direct `call` opcodes because
            // validated wasm guarantees that.
            active_when(Column(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE)),
        ),
        MemorySpec {
            id: WasmMemoryId::FunctionLocalCount,
            kind: MemoryKind::Rom,
            ports: vec![
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_FUNCTION_REF)]),
                    value_column: COL_ENTERED_FUNCTION_NUM_LOCALS,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_GUEST_ENTRY_ACTIVE)),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_HOST_CALLEE_FREF_AFTER)]),
                    value_column: COL_ENTERED_FUNCTION_NUM_LOCALS,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_TURN_BOUNDARY)),
                },
            ],
        },
        rom_read_spec(
            WasmMemoryId::PcFunctionRef,
            vec![Column(COL_PC_BEFORE)],
            Column(COL_CURRENT_FUNCTION_REF),
            // Program and frame-transition rows only. Gather rows carry the
            // one-past-the-end pc, while permutation, turn-boundary, and
            // padding rows do not consume frame identity either.
            active_when(Column(COL_PC_FREF_ACTIVE)),
        ),
        MemorySpec {
            id: WasmMemoryId::FunctionCallMetadata,
            kind: MemoryKind::Rom,
            ports: vec![
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_FUNCTION_REF)]),
                    value_column: COL_CALL_TARGET_METADATA,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(selector_col(super::isa::WasmOpcode::Call).unwrap())),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_FUNCTION_REF)]),
                    value_column: COL_CALL_TARGET_METADATA,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(selector_col(super::isa::WasmOpcode::ReturnCall).unwrap())),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_FUNCTION_REF)]),
                    value_column: COL_CALL_TARGET_METADATA,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(selector_col(super::isa::WasmOpcode::CallIndirect).unwrap())),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_FUNCTION_REF)]),
                    value_column: COL_CALL_TARGET_METADATA,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(
                        selector_col(super::isa::WasmOpcode::ReturnCallIndirect).unwrap(),
                    )),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_HOST_CALLEE_FREF_AFTER)]),
                    value_column: COL_CALL_TARGET_METADATA,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_TURN_BOUNDARY)),
                },
            ],
        },
        MemorySpec {
            id: WasmMemoryId::CallTarget,
            kind: MemoryKind::Rom,
            ports: [WasmOpcode::Call, WasmOpcode::ReturnCall]
                .into_iter()
                .map(|opcode| MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_PC_BEFORE)]),
                    value_column: COL_FUNCTION_REF,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(selector_col(opcode).unwrap())),
                })
                .collect(),
        },
        MemorySpec {
            id: WasmMemoryId::FunctionEntry,
            kind: MemoryKind::Rom,
            ports: vec![
                // Gated on guest-call rows only: host imports have no entry
                // pc (host calls fall through to pc+1, pinned by a CCS row).
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_FUNCTION_REF)]),
                    value_column: COL_PC_AFTER,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_GUEST_ENTRY_ACTIVE)),
                },
                // Turn boundary: the next turn's pc jump is bound to the
                // entered export's entry pc, keyed by the repointed
                // attribution.
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_HOST_CALLEE_FREF_AFTER)]),
                    value_column: COL_PC_AFTER,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_TURN_BOUNDARY)),
                },
            ],
        },
        rom_read_spec(
            WasmMemoryId::PcEdgeKind,
            vec![Column(COL_PC_BEFORE)],
            Column(COL_PC_EDGE_KIND),
            active_when(Column(COL_IS_PROGRAM_ROW)),
        ),
        // Event-template ROMs (see `docs/host-event-grammar-tables.md` §3.4):
        // per-slot source descriptors keyed by (event index, slot cursor)
        // read on gather rows, where the event index counts from the first
        // event row of the template (`HostEventBase`), and per-import event counts read on
        // host-call/result rows. Content is generated from the embedder's
        // `HostEventBindings` (see `host_event_bindings::preload_host_event_tables`).
        rom_read_spec(
            WasmMemoryId::HostEventSlotKind,
            vec![
                Column(COL_HOST_EVENT_INDEX_BEFORE),
                Column(COL_HOST_EVENT_SLOT_CURSOR_BEFORE),
            ],
            Column(COL_HOST_EVENT_SLOT_KIND),
            active_when(Column(COL_GATHER_ACTIVE)),
        ),
        rom_read_spec(
            WasmMemoryId::HostEventSlotArg,
            vec![
                Column(COL_HOST_EVENT_INDEX_BEFORE),
                Column(COL_HOST_EVENT_SLOT_CURSOR_BEFORE),
            ],
            Column(COL_HOST_EVENT_SLOT_ARG),
            active_when(Column(COL_GATHER_ACTIVE)),
        ),
        rom_read_spec(
            WasmMemoryId::HostEventSlotVariant,
            vec![
                Column(COL_HOST_EVENT_INDEX_BEFORE),
                Column(COL_HOST_EVENT_SLOT_CURSOR_BEFORE),
            ],
            Column(COL_HOST_EVENT_SLOT_VARIANT),
            active_when(Column(COL_GATHER_ACTIVE)),
        ),
        rom_read_spec(
            WasmMemoryId::HostEventSlotImmediate0,
            vec![
                Column(COL_HOST_EVENT_INDEX_BEFORE),
                Column(COL_HOST_EVENT_SLOT_CURSOR_BEFORE),
            ],
            Column(COL_HOST_EVENT_SLOT_IMMEDIATE0),
            active_when(Column(COL_GATHER_ACTIVE)),
        ),
        rom_read_spec(
            WasmMemoryId::HostEventSlotImmediate1,
            vec![
                Column(COL_HOST_EVENT_INDEX_BEFORE),
                Column(COL_HOST_EVENT_SLOT_CURSOR_BEFORE),
            ],
            Column(COL_HOST_EVENT_SLOT_IMMEDIATE1),
            active_when(Column(COL_GATHER_ACTIVE)),
        ),
        // Import schedule counts and export entry-schedule counts are
        // SEPARATE families so an initial-schedule count read can never land
        // on the other kind's cell: turn
        // boundaries and exit latches see only export entry cells, while
        // host-call rows see only import schedule cells. These cells store
        // count + 1 ("presence bias"): an undeclared fref
        // reads 0 and the load rows subtract 1, poisoning the schedule to
        // events_remaining = -1 = p-1 (the countdown is field-width; the row
        // itself stays satisfiable). Each completed event block decrements
        // events_remaining and increments event_index. While events_remaining
        // is nonzero, program, result, and boundary rows cannot execute, so
        // event_index cannot reset. Every gather uses event_index as an active
        // host-event ROM address component. A composed memory argument must
        // range-prove address components to at most 32 bits (and to the
        // family's narrower configured width). With those bounds, field
        // addition cannot wrap at 2^32: after at most 2^32 blocks the next
        // gather is unsatisfiable while events_remaining remains nonzero.
        // An undeclared fref also reads the base 0, so its gathers read the
        // event rows of the declared templates first; the bound holds the
        // same. This enforces template presence in the composed circuit. The bare
        // frontend relation does not establish this memory-binding guarantee.
        MemorySpec {
            id: WasmMemoryId::HostEventImportScheduleCount,
            kind: MemoryKind::Rom,
            ports: vec![
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_FUNCTION_REF)]),
                    value_column: crate::layout::COL_HOST_IMPORT_SCHEDULE_COUNTS,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_HOST_CALL_ACTIVE)),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_HOST_CALLEE_FREF_AFTER)]),
                    value_column: crate::layout::COL_HOST_IMPORT_SCHEDULE_COUNTS,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(crate::layout::COL_HOST_RESUME)),
                },
            ],
        },
        // The first event row of a template. Each row that loads a schedule
        // reads it with the key of that schedule, so the event index of the
        // gather rows starts in the template of the bound function.
        MemorySpec {
            id: WasmMemoryId::HostEventBase,
            kind: MemoryKind::Rom,
            ports: [
                (COL_FUNCTION_REF, COL_HOST_CALL_ACTIVE),
                (COL_HOST_CALLEE_FREF_AFTER, crate::layout::COL_HOST_RESUME),
                (COL_TURN_EXPORT_FREF_BEFORE, COL_HOST_EVENT_EXIT_LATCH),
                (COL_HOST_CALLEE_FREF_AFTER, COL_TURN_BOUNDARY),
            ]
            .into_iter()
            .map(|(address, gate)| MemoryPortSpec {
                address_columns: memory_columns([Column(address)]),
                value_column: crate::layout::COL_HOST_EVENT_BASE,
                kind: MemoryPortKind::Read,
                activation: active_when(Column(gate)),
            })
            .collect(),
        },
        // One word for each export: the entry count plus one in the low
        // half and the exit count in the high half, decoded as the import
        // word is. The exit latch re-reads it to continue the event
        // numbering, and the turn boundary loads it as the next turn's owed
        // schedule (the exit count lets the nonempty-template guard accept
        // an empty entry schedule).
        MemorySpec {
            id: WasmMemoryId::HostEventExportScheduleCount,
            kind: MemoryKind::Rom,
            ports: [
                (COL_TURN_EXPORT_FREF_BEFORE, COL_HOST_EVENT_EXIT_LATCH),
                (COL_HOST_CALLEE_FREF_AFTER, COL_TURN_BOUNDARY),
            ]
            .into_iter()
            .map(|(address, gate)| MemoryPortSpec {
                address_columns: memory_columns([Column(address)]),
                value_column: crate::layout::COL_HOST_IMPORT_SCHEDULE_COUNTS,
                kind: MemoryPortKind::Read,
                activation: active_when(Column(gate)),
            })
            .collect(),
        },
        MemorySpec {
            id: WasmMemoryId::PcRom,
            kind: MemoryKind::Rom,
            ports: vec![MemoryPortSpec {
                address_columns: memory_columns([Column(COL_EDGE_ADDR)]),
                value_column: COL_PC_AFTER,
                kind: MemoryPortKind::Read,
                activation: active_when(Column(COL_PC_ROM_ACTIVE)),
            }],
        },
        // A slide's countdown and distance come from the taken edge; a clean
        // activation return (the exit latch) reads its distance to bind its
        // result count, without seeding a move.
        rom_read_spec(
            WasmMemoryId::FrameSlideRemaining,
            vec![Column(COL_EDGE_ADDR)],
            Column(COL_FRAME_MOVE_REMAINING_AFTER),
            active_when(Column(COL_FRAME_SLIDE_ACTIVE)),
        ),
        MemorySpec {
            id: WasmMemoryId::FrameSlideDistance,
            kind: MemoryKind::Rom,
            ports: vec![
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_EDGE_ADDR)]),
                    value_column: COL_FRAME_MOVE_DISTANCE_AFTER,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_FRAME_SLIDE_ACTIVE)),
                },
                MemoryPortSpec {
                    address_columns: memory_columns([Column(COL_EDGE_ADDR)]),
                    value_column: COL_EXIT_SLIDE_DISTANCE,
                    kind: MemoryPortKind::Read,
                    activation: active_when(Column(COL_HOST_EVENT_EXIT_LATCH)),
                },
            ],
        },
        rom_read_spec(
            WasmMemoryId::ProgramBrTableLen,
            vec![Column(COL_PC_BEFORE)],
            Column(super::layout::COL_BR_TABLE_LEN),
            active_when(Column(selector_col(WasmOpcode::BrTable).unwrap())),
        ),
    ];

    use crate::layout::{COL_HOST_CONTEXT, COL_HOST_CONTEXT_ADDR, COL_HOST_CONTEXT_READ, COL_HOST_SUSPEND};
    let context_memories = [
        WasmMemoryId::CallStackReturnPc,
        WasmMemoryId::CallStackCallerFbp,
        WasmMemoryId::CallStackCallerSpBase,
        WasmMemoryId::HostContextExport,
        WasmMemoryId::HostContextImport,
        WasmMemoryId::HostContextArgsBase,
        WasmMemoryId::HostContextParentDepth,
        WasmMemoryId::HostContextSp,
    ];
    for (id, value_column) in context_memories.into_iter().zip(COL_HOST_CONTEXT) {
        let ports = vec![
            MemoryPortSpec {
                address_columns: memory_columns([Column(COL_HOST_CONTEXT_ADDR)]),
                value_column,
                kind: MemoryPortKind::Write {
                    value_before_column: None,
                },
                activation: active_when(Column(COL_HOST_SUSPEND)),
            },
            MemoryPortSpec {
                address_columns: memory_columns([Column(COL_HOST_CONTEXT_ADDR)]),
                value_column,
                kind: MemoryPortKind::Read,
                activation: active_when(Column(COL_HOST_CONTEXT_READ)),
            },
        ];
        if let Some(memory) = memories.iter_mut().find(|memory| memory.id == id) {
            memory.ports.extend(ports);
        } else {
            memories.push(MemorySpec {
                id,
                kind: MemoryKind::Ram,
                ports,
            });
        }
    }
    let columns = crate::witness_layout::range_checked_column_registry();
    let logical_memory = memories;
    let mut memories: Vec<MemorySpec<WasmMemoryId>> = Vec::new();
    for memory in &logical_memory {
        let id = memory.id.table();
        match memories.iter_mut().find(|table| table.id == id) {
            Some(table) => {
                assert_eq!(table.kind, memory.kind, "table {id} has one kind");
                table.ports.extend(memory.ports.iter().cloned());
            }
            None => memories.push(MemorySpec {
                id,
                kind: memory.kind,
                ports: memory.ports.clone(),
            }),
        }
    }
    let memory = MemoryCatalog::new(memories, &columns).expect("valid WASM memory declarations");
    let continuity = ContinuityCatalog::new(build_ivc_state_continuity_links(), &columns)
        .expect("valid WASM continuity declarations");

    WasmRelationLayout {
        auxiliary: WasmAuxiliaryRelations {
            memory,
            logical_memory,
            continuity,
        },
        linear_memory,
        sign_extension,
    }
}
