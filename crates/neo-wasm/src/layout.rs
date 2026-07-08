//! Owns the static WASM row layout.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::isa::{opcode_code, WasmOpcode};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Column(pub usize);

pub const PUBLIC_INPUTS: usize = 7;
/// Reserved `control_choice` value under which `pc_rom` stores a call site's
/// continuation pc (the instruction after the call): the return address for
/// guest calls, the fall-through target for indirect host calls. Shares the
/// choice axis with the per-opcode branch discriminants (`br_if`, `if`,
/// `br_table` arms), which use 0/1/arm-index at their own pcs; the
/// reservation is per call-site pc, not global.
pub const PC_ROM_CALL_RETURN_CHOICE: u64 = 1;

/// Declared intrinsic range for a witness column.
///
/// These declarations are meant to be enforced; otherwise the proof is not
/// sound. Enforcement can happen in the wasm CCS itself, as part of a lookup
/// argument. The selected approach is not supposed to change the semantics, but
/// may affect performance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnWidth {
    /// Constrained to {0, 1}.
    Boolean,
    /// Constrained to [0, 256).
    Byte,
    /// Constrained to [0, 2^32).
    U32,
    /// No declared bound: the value is treated as a full field element.
    /// Use for columns whose intrinsic range has not been audited yet, or
    /// whose width depends on a row gate (e.g. wide limbs that are 64-bit
    /// only when `wide_values_enabled = 1`).
    Field,
}

/// Static metadata about a witness column. `name` is the `UPPER_SNAKE_CASE`
/// Rust identifier (e.g. `"COL_OPCODE_CODE"`) as produced by `stringify!`;
/// consumers that want a lowercased / display label should strip the `COL_`
/// prefix and lowercase. `role` is a free-form human-readable description (may
/// be empty if the column has not been documented). `width` declares the
/// intrinsic range; see [`ColumnWidth`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmColumnSpec {
    pub index: usize,
    pub name: &'static str,
    pub role: &'static str,
    pub width: ColumnWidth,
}

/// Used to define named columns directly related to wasm/vm semantics. More
/// generic constraints like range-checks and lookups are supposed to be added
/// modularly on top of the base CSS.
macro_rules! define_columns {
    ($( ( $name:ident, $role:literal $(, $width:expr)? ) ),+ $(,)?) => {
        define_columns!(@assign 0usize; $($name),+);

        /// Macro-generated table of column metadata.i
        pub const COLUMN_SPECS: &[WasmColumnSpec] = &[
            $(WasmColumnSpec {
                index: $name,
                name: stringify!($name),
                role: $role,
                width: define_columns!(@maybe_width $($width)?),
            }),+
        ];
    };
    (@maybe_width $width:expr) => { $width };
    (@maybe_width) => { ColumnWidth::Field };
    (@assign $idx:expr; $name:ident, $($rest:ident),+) => {
        pub const $name: usize = $idx;
        define_columns!(@assign $idx + 1usize; $($rest),+);
    };
    (@assign $idx:expr; $name:ident) => {
        pub const $name: usize = $idx;
        /// Number of macro-declared named columns. NOT the final witness
        /// width. Range constraints may be added, plus the F' transformation,
        /// lookup/mcc related constraints derived from the specs.
        pub const NAMED_COLUMN_COUNT: usize = $idx + 1usize;
    };
}

define_columns!(
    (COL_ONE, ""),
    (COL_OUTPUT_ENABLED_BEFORE, "carried simple-output flag before this row"),
    (COL_OUTPUT_ENABLED_AFTER, "carried simple-output flag after this row"),
    (
        COL_OUTPUT_VALUE_LO_BEFORE,
        "carried simple-output low limb before this row"
    ),
    (
        COL_OUTPUT_VALUE_LO_AFTER,
        "carried simple-output low limb after this row"
    ),
    (
        COL_OUTPUT_VALUE_HI_BEFORE,
        "carried simple-output high limb before this row"
    ),
    (
        COL_OUTPUT_VALUE_HI_AFTER,
        "carried simple-output high limb after this row"
    ),
    (COL_OPCODE_CODE, "opcode decode selector source", ColumnWidth::U32),
    (COL_PC_BEFORE, "transition source pc", ColumnWidth::U32),
    (COL_PC_AFTER, "transition destination pc", ColumnWidth::U32),
    (
        COL_CONTROL_CHOICE,
        "normalized control-edge selector where 0 is default/fallthrough",
        ColumnWidth::U32
    ),
    (
        COL_PC_EDGE_KIND,
        "static next-pc kind: static, return-like, dynamic call_indirect, or terminal",
        ColumnWidth::U32
    ),
    (
        COL_WIDE_VALUES_ENABLED,
        "row flag enabling high limbs for i64-shaped values",
        ColumnWidth::Boolean
    ),
    (
        COL_OUTPUT_CAPTURED,
        "one-row gate when the simple-output carry captures a halted result",
        ColumnWidth::Boolean
    ),
    (COL_SP_BEFORE, "transition source stack pointer", ColumnWidth::U32),
    (COL_SP_AFTER, "transition destination stack pointer", ColumnWidth::U32),
    (COL_HALTED, "terminal row flag", ColumnWidth::Boolean),
    (
        COL_IS_PROGRAM_ROW,
        "real decoded wasm program row",
        ColumnWidth::Boolean
    ),
    (
        COL_PC_ROM_ACTIVE,
        "program static edge ROM read gate",
        ColumnWidth::Boolean
    ),
    (
        COL_PC_EDGE_KIND_IS_STATIC,
        "zero-test flag for static pc-edge rows",
        ColumnWidth::Boolean
    ),
    (COL_PC_EDGE_KIND_INV, "inverse witness for pc edge-kind zero test"),
    (
        COL_PARAM_INIT_ACTIVE_BEFORE,
        "call-parameter initialization mode before this row",
        ColumnWidth::Boolean
    ),
    (
        COL_PARAM_INIT_ACTIVE_AFTER,
        "call-parameter initialization mode after this row",
        ColumnWidth::Boolean
    ),
    (
        COL_PADDING_ACTIVE,
        "synthetic state-preserving padding row flag",
        ColumnWidth::Boolean
    ),
    (
        COL_PARAM_INIT_REMAINING_BEFORE,
        "remaining call parameters to initialize before this row",
        ColumnWidth::U32
    ),
    (
        COL_PARAM_INIT_REMAINING_AFTER,
        "remaining call parameters to initialize after this row",
        ColumnWidth::U32
    ),
    (
        COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO,
        "zero-test flag for remaining call parameters after this row",
        ColumnWidth::Boolean
    ),
    (
        COL_PARAM_INIT_REMAINING_AFTER_INV,
        "inverse witness for remaining call parameters after this row"
    ),
    (
        COL_HOST_ARGS_ACTIVE_BEFORE,
        "host-call argument-pop mode before this row",
        ColumnWidth::Boolean
    ),
    (
        COL_HOST_ARGS_ACTIVE_AFTER,
        "host-call argument-pop mode after this row",
        ColumnWidth::Boolean
    ),
    (
        COL_HOST_ARGS_REMAINING_BEFORE,
        "remaining host-call arguments to pop before this row",
        ColumnWidth::U32
    ),
    (
        COL_HOST_ARGS_REMAINING_AFTER,
        "remaining host-call arguments to pop after this row",
        ColumnWidth::U32
    ),
    (
        COL_HOST_ARGS_REMAINING_AFTER_IS_ZERO,
        "zero-test flag for remaining host-call arguments after this row",
        ColumnWidth::Boolean
    ),
    (
        COL_HOST_ARGS_REMAINING_AFTER_INV,
        "inverse witness for remaining host-call arguments after this row"
    ),
    (
        COL_HOST_RESULT_PENDING_BEFORE,
        "host-call result push still owed before this row",
        ColumnWidth::Boolean
    ),
    (
        COL_HOST_RESULT_PENDING_AFTER,
        "host-call result push still owed after this row",
        ColumnWidth::Boolean
    ),
    (
        COL_HOST_CALLEE_FREF_BEFORE,
        "callee function ref of the most recent host call before this row (event attribution carry)",
        ColumnWidth::U32
    ),
    (
        COL_HOST_CALLEE_FREF_AFTER,
        "callee function ref of the most recent host call after this row (event attribution carry)",
        ColumnWidth::U32
    ),
    (
        COL_HOST_RESULT_ACTIVE,
        "this row pushes the pending host-call result",
        ColumnWidth::Boolean
    ),
    (
        COL_CI_HOST_CALL,
        "non-trapping call_indirect row targeting a host import",
        ColumnWidth::Boolean
    ),
    (
        COL_GUEST_CALL_ACTIVE,
        "guest-call row flag: this row enters a traced guest callee (and pushes its return context)",
        ColumnWidth::Boolean
    ),
    (
        COL_CALL_STACK_POP_PRESENT,
        "flag indicating that this row restores a saved caller return context",
        ColumnWidth::Boolean
    ),
    (
        COL_CALL_STACK_RETURN_PC_VALUE,
        "call-stack return-pc cell value: written on guest-call push, read back on pop into pc_after",
        ColumnWidth::U32
    ),
    (
        COL_CALL_STACK_CALLER_FBP_VALUE,
        "call-stack caller-fbp cell value: written on guest-call push, read back on pop into locals_fbp_after",
        ColumnWidth::U32
    ),
    (
        COL_CALL_STACK_DEPTH_BEFORE,
        "call return-context stack depth before this row",
        ColumnWidth::U32
    ),
    (
        COL_CALL_STACK_DEPTH_AFTER,
        "call return-context stack depth after this row",
        ColumnWidth::U32
    ),
    (
        COL_CALL_STACK_ADDR,
        "call return-context stack address read or written this row",
        ColumnWidth::U32
    ),
    (
        COL_PC_ROM_CALL_RETURN_CHOICE,
        "pc-rom control-choice coordinate for the call-site continuation-pc read (guest pushes and indirect host fall-through)",
        ColumnWidth::U32
    ),
    (
        COL_CURRENT_FUNCTION_REF,
        "normalized function reference for the currently executing frame",
        ColumnWidth::U32
    ),
    (
        COL_CURRENT_FUNCTION_NUM_LOCALS,
        "number of locals in the current function frame",
        ColumnWidth::U32
    ),
    (COL_STACK_READS, "stack delta source", ColumnWidth::U32),
    (COL_STACK_WRITES, "stack delta destination", ColumnWidth::U32),
    (
        COL_STACK_READ0_ACTIVE,
        "stack lane 0 read activity flag",
        ColumnWidth::Boolean
    ),
    (
        COL_STACK_READ1_ACTIVE,
        "stack lane 1 read activity flag",
        ColumnWidth::Boolean
    ),
    (
        COL_STACK_READ2_ACTIVE,
        "stack lane 2 read activity flag",
        ColumnWidth::Boolean
    ),
    (
        COL_STACK_WRITE0_ACTIVE,
        "stack lane 0 write activity flag",
        ColumnWidth::Boolean
    ),
    (COL_OP_TABLE_ENABLED, "lookup gate", ColumnWidth::Boolean),
    (COL_SEL_NOP, "", ColumnWidth::Boolean),
    (COL_SEL_I32_CONST, "", ColumnWidth::Boolean),
    (COL_SEL_I64_CONST, "", ColumnWidth::Boolean),
    (COL_SEL_REF_FUNC, "", ColumnWidth::Boolean),
    (COL_SEL_I32_ADD, "", ColumnWidth::Boolean),
    (COL_SEL_I64_ADD, "", ColumnWidth::Boolean),
    (COL_SEL_I32_SUB, "", ColumnWidth::Boolean),
    (COL_SEL_I64_SUB, "", ColumnWidth::Boolean),
    (COL_SEL_I32_LOAD, "", ColumnWidth::Boolean),
    (COL_SEL_I32_LOAD8_S, "", ColumnWidth::Boolean),
    (COL_SEL_I32_LOAD8_U, "", ColumnWidth::Boolean),
    (COL_SEL_I32_LOAD16_S, "", ColumnWidth::Boolean),
    (COL_SEL_I32_LOAD16_U, "", ColumnWidth::Boolean),
    (COL_SEL_I64_LOAD, "", ColumnWidth::Boolean),
    (COL_SEL_I32_STORE, "", ColumnWidth::Boolean),
    (COL_SEL_I32_STORE8, "", ColumnWidth::Boolean),
    (COL_SEL_I32_STORE16, "", ColumnWidth::Boolean),
    (COL_SEL_I64_STORE, "", ColumnWidth::Boolean),
    (COL_SEL_MEMORY_SIZE, "", ColumnWidth::Boolean),
    (COL_SEL_MEMORY_GROW, "", ColumnWidth::Boolean),
    (COL_SEL_TABLE_SIZE, "", ColumnWidth::Boolean),
    (COL_SEL_TABLE_GET, "", ColumnWidth::Boolean),
    (COL_SEL_TABLE_SET, "", ColumnWidth::Boolean),
    (COL_SEL_DROP, "", ColumnWidth::Boolean),
    (COL_SEL_BR, "", ColumnWidth::Boolean),
    (COL_SEL_BLOCK, "", ColumnWidth::Boolean),
    (COL_SEL_LOOP, "", ColumnWidth::Boolean),
    (COL_SEL_IF, "", ColumnWidth::Boolean),
    (COL_SEL_ELSE, "", ColumnWidth::Boolean),
    (COL_SEL_END, "", ColumnWidth::Boolean),
    (COL_SEL_UNREACHABLE, "", ColumnWidth::Boolean),
    (COL_SEL_I32_CLZ, "", ColumnWidth::Boolean),
    (COL_SEL_I32_CTZ, "", ColumnWidth::Boolean),
    (COL_SEL_I32_POPCNT, "", ColumnWidth::Boolean),
    (COL_SEL_I32_EQZ, "", ColumnWidth::Boolean),
    (COL_SEL_I64_EQZ, "", ColumnWidth::Boolean),
    (COL_SEL_I32_EQ, "", ColumnWidth::Boolean),
    (COL_SEL_I32_NE, "", ColumnWidth::Boolean),
    (COL_SEL_I32_LTS, "", ColumnWidth::Boolean),
    (COL_SEL_I32_LTU, "", ColumnWidth::Boolean),
    (COL_SEL_I32_GTS, "", ColumnWidth::Boolean),
    (COL_SEL_I32_GTU, "", ColumnWidth::Boolean),
    (COL_SEL_I32_LES, "", ColumnWidth::Boolean),
    (COL_SEL_I32_LEU, "", ColumnWidth::Boolean),
    (COL_SEL_I32_GES, "", ColumnWidth::Boolean),
    (COL_SEL_I32_GEU, "", ColumnWidth::Boolean),
    (COL_SEL_I32_AND, "", ColumnWidth::Boolean),
    (COL_SEL_I32_OR, "", ColumnWidth::Boolean),
    (COL_SEL_I32_XOR, "", ColumnWidth::Boolean),
    (COL_SEL_I32_MUL, "", ColumnWidth::Boolean),
    (COL_SEL_I64_AND, "", ColumnWidth::Boolean),
    (COL_SEL_I64_OR, "", ColumnWidth::Boolean),
    (COL_SEL_I64_XOR, "", ColumnWidth::Boolean),
    (COL_SEL_I64_MUL, "", ColumnWidth::Boolean),
    (COL_SEL_I32_SHL, "", ColumnWidth::Boolean),
    (COL_SEL_I32_SHR_U, "", ColumnWidth::Boolean),
    (COL_SEL_I32_SHR_S, "", ColumnWidth::Boolean),
    (COL_SEL_I32_ROTL, "", ColumnWidth::Boolean),
    (COL_SEL_I32_ROTR, "", ColumnWidth::Boolean),
    (COL_SEL_I32_DIV_U, "", ColumnWidth::Boolean),
    (COL_SEL_I32_DIV_S, "", ColumnWidth::Boolean),
    (COL_SEL_I32_REM_U, "", ColumnWidth::Boolean),
    (COL_SEL_I32_REM_S, "", ColumnWidth::Boolean),
    (COL_SEL_SELECT, "", ColumnWidth::Boolean),
    (COL_SEL_BR_IF_EQZ, "", ColumnWidth::Boolean),
    (COL_SEL_BR_TABLE, "", ColumnWidth::Boolean),
    (COL_SEL_CALL, "", ColumnWidth::Boolean),
    (COL_SEL_CALL_INDIRECT, "", ColumnWidth::Boolean),
    (COL_SEL_RETURN, "", ColumnWidth::Boolean),
    (COL_SEL_LOCAL_GET, "", ColumnWidth::Boolean),
    (COL_SEL_LOCAL_SET, "", ColumnWidth::Boolean),
    (COL_SEL_LOCAL_TEE, "", ColumnWidth::Boolean),
    (COL_SEL_GLOBAL_GET, "", ColumnWidth::Boolean),
    (COL_SEL_GLOBAL_SET, "", ColumnWidth::Boolean),
    (COL_SEL_I64_EQ, "", ColumnWidth::Boolean),
    (COL_SEL_I64_NE, "", ColumnWidth::Boolean),
    (COL_SEL_I64_STORE8, "", ColumnWidth::Boolean),
    (COL_SEL_I64_STORE16, "", ColumnWidth::Boolean),
    (COL_SEL_I64_STORE32, "", ColumnWidth::Boolean),
    (COL_SEL_I64_LOAD8_U, "", ColumnWidth::Boolean),
    (COL_SEL_I64_LOAD16_U, "", ColumnWidth::Boolean),
    (COL_SEL_I64_LOAD32_U, "", ColumnWidth::Boolean),
    (COL_SEL_I64_LOAD8_S, "", ColumnWidth::Boolean),
    (COL_SEL_I64_LOAD16_S, "", ColumnWidth::Boolean),
    (COL_SEL_I64_LOAD32_S, "", ColumnWidth::Boolean),
    (
        COL_LOCAL_WRITE_ENABLED,
        "locals memory write gate for local.set/local.tee",
        ColumnWidth::Boolean
    ),
    (
        COL_TABLE_READ_ENABLED,
        "table memory read gate for table.get/call_indirect",
        ColumnWidth::Boolean
    ),
    (
        COL_LOCALS_FBP_BEFORE,
        "locals memory frame base before this row",
        ColumnWidth::U32
    ),
    (
        COL_LOCALS_FBP_AFTER,
        "locals memory frame base after this row",
        ColumnWidth::U32
    ),
    (COL_LOCAL_INDEX, "locals memory offset", ColumnWidth::U32),
    (COL_LOCAL_VALUE, "locals memory value", ColumnWidth::U32),
    (
        COL_LOCAL_VALUE_HI,
        "locals memory high limb for future i64 support",
        ColumnWidth::U32
    ),
    (COL_GLOBAL_INDEX, "globals memory index", ColumnWidth::U32),
    (COL_GLOBAL_VALUE, "globals memory value", ColumnWidth::U32),
    (
        COL_GLOBAL_VALUE_HI,
        "globals memory high limb for future i64 support",
        ColumnWidth::U32
    ),
    (COL_TABLE_ID, "table state namespace selector", ColumnWidth::U32),
    (COL_TABLE_INDEX, "table element index", ColumnWidth::U32),
    (
        COL_TABLE_VALUE,
        "normalized table element value observed by this step",
        ColumnWidth::U32
    ),
    (
        COL_TABLE_SIZE,
        "size of the referenced table observed by this step",
        ColumnWidth::U32
    ),
    (
        COL_FUNCTION_REF,
        "normalized function reference selected by call-like opcodes",
        ColumnWidth::U32
    ),
    (
        COL_CALL_PARAM_COUNT,
        "parameter count for the selected call target",
        ColumnWidth::U32
    ),
    (
        COL_CALL_RESULT_COUNT,
        "result count for the selected call target",
        ColumnWidth::U32
    ),
    (
        COL_TARGET_FUNCTION_IS_GUEST,
        "true when the selected call target is a guest-defined function",
        ColumnWidth::Boolean
    ),
    (
        COL_CALL_INDIRECT_TYPE_INDEX,
        "raw module type-section index from the call_indirect instruction immediate",
        ColumnWidth::U32
    ),
    (
        COL_FUNCTION_TYPE_ID,
        "normalized deduplicated type id for the observed function reference",
        ColumnWidth::U32
    ),
    (
        COL_EXPECTED_TYPE_ID,
        "normalized deduplicated type id expected by the current opcode",
        ColumnWidth::U32
    ),
    (
        COL_MEMORY_PAGES_BEFORE,
        "linear memory page count before this step",
        ColumnWidth::U32
    ),
    (
        COL_MEMORY_PAGES_AFTER,
        "linear memory page count after this step",
        ColumnWidth::U32
    ),
    (
        COL_MAX_MEMORY_PAGES_BEFORE,
        "verifier-authoritative max linear-memory page count before this step (carried constant)",
        ColumnWidth::U32
    ),
    (
        COL_MAX_MEMORY_PAGES_AFTER,
        "verifier-authoritative max linear-memory page count after this step (carried constant)",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ0_ADDR_LO,
        "operand-stack read lane 0 low-limb physical address",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ0_ADDR_HI,
        "operand-stack read lane 0 high-limb physical address",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ0_VALUE_LO,
        "operand-stack read lane 0 value",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ0_VALUE_HI,
        "operand-stack read lane 0 high limb for future i64 support",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ1_ADDR_LO,
        "operand-stack read lane 1 low-limb physical address",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ1_ADDR_HI,
        "operand-stack read lane 1 high-limb physical address",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ1_VALUE_LO,
        "operand-stack read lane 1 value",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ1_VALUE_HI,
        "operand-stack read lane 1 high limb for future i64 support",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ2_ADDR_LO,
        "operand-stack read lane 2 low-limb physical address",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ2_ADDR_HI,
        "operand-stack read lane 2 high-limb physical address",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ2_VALUE_LO,
        "operand-stack read lane 2 value",
        ColumnWidth::U32
    ),
    (
        COL_STACK_READ2_VALUE_HI,
        "operand-stack read lane 2 high limb for future i64 support",
        ColumnWidth::U32
    ),
    (
        COL_STACK_WRITE0_ADDR_LO,
        "operand-stack write lane 0 low-limb physical address",
        ColumnWidth::U32
    ),
    (
        COL_STACK_WRITE0_ADDR_HI,
        "operand-stack write lane 0 high-limb physical address",
        ColumnWidth::U32
    ),
    (
        COL_STACK_WRITE0_VALUE_LO,
        "operand-stack write lane 0 value",
        ColumnWidth::U32
    ),
    (
        COL_STACK_WRITE0_VALUE_HI,
        "operand-stack write lane 0 high limb for future i64 support",
        ColumnWidth::U32
    ),
    (COL_WIDE_AUX0, "", ColumnWidth::Boolean),
    (COL_WIDE_AUX1, "", ColumnWidth::Boolean),
    (
        COL_LINEAR_MEM_IMM_OFFSET,
        "linear-memory immediate offset in bytes",
        ColumnWidth::U32
    ),
    (
        COL_LINEAR_MEM_BYTE_OFFSET,
        "linear-memory byte offset within the first word lane",
        ColumnWidth::U32
    ),
    (
        COL_LINEAR_MEM_USE_LANE1,
        "linear-memory second-lane flag for unaligned accesses",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_USE_LANE2,
        "linear-memory third-lane flag for wide unaligned accesses",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_USE_LANE0,
        "linear-memory first-lane activity gate",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_LANE0_LOAD_ACTIVE,
        "linear-memory lane0 load gate (use_lane0 AND opcode is a load)",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_LANE1_LOAD_ACTIVE,
        "linear-memory lane1 load gate (use_lane1 AND opcode is a load)",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_LANE2_LOAD_ACTIVE,
        "linear-memory lane2 load gate (use_lane2 AND opcode is a load)",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_LANE0_STORE_ACTIVE,
        "linear-memory lane0 store gate (use_lane0 AND opcode is a store)",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_LANE1_STORE_ACTIVE,
        "linear-memory lane1 store gate (use_lane1 AND opcode is a store)",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_LANE2_STORE_ACTIVE,
        "linear-memory lane2 store gate (use_lane2 AND opcode is a store)",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_LANE0_ADDR,
        "linear-memory first word-lane address",
        ColumnWidth::U32
    ),
    (
        COL_LINEAR_MEM_LANE0_VALUE,
        "linear-memory first word-lane accessed value"
    ),
    (
        COL_LINEAR_MEM_LANE1_ADDR,
        "linear-memory second word-lane address",
        ColumnWidth::U32
    ),
    (
        COL_LINEAR_MEM_LANE1_VALUE,
        "linear-memory second word-lane accessed value"
    ),
    (
        COL_LINEAR_MEM_LANE2_ADDR,
        "linear-memory third word-lane address",
        ColumnWidth::U32
    ),
    (
        COL_LINEAR_MEM_LANE2_VALUE,
        "linear-memory third word-lane accessed value"
    ),
    (COL_OP_TABLE_ID, "lookup table row selector", ColumnWidth::U32),
    (COL_OP_TABLE_VALUE, "lookup payload witness"),
    // `COL_SELECT_COND_IS_ZERO` is forced to {0, 1} by the zero-test rows
    // emitted by `push_select_constraints`. Declared `Boolean` so the
    // spec reflects its actual range; whichever path eventually enforces
    // `ColumnWidth::Boolean` will overlap with the gadget's constraint and
    // one of the two becomes redundant; an optimization to revisit later.
    // We deliberately do not introduce a dedicated `ImpliedBoolean` width
    // for this case because that would be premature complexity.
    (
        COL_SELECT_COND_IS_ZERO,
        "scratch column for push_select_constraints for select opcode",
        ColumnWidth::Boolean
    ),
    (
        COL_SELECT_SCRATCH_INV,
        "scratch column for push_select_constraints for select opcode"
    ),
    (
        COL_SELECT_OUT_DELTA_LO,
        "scratch column for push_select_constraints low-limb mux"
    ),
    (
        COL_SELECT_OUT_DELTA_HI,
        "scratch column for push_select_constraints high-limb mux"
    ),
    // Shared zero-test scratch columns for the CCS-native comparators:
    // i32.eqz / i64.eqz / i32.eq / i32.ne / i64.eq / i64.ne. The active
    // opcode's selector pins COL_CMP_LO_DIFF to the lo-limb input
    // (read0, read0_lo, or read0[_lo] - read1[_lo]); the zero-test gadget
    // forces COL_CMP_LO_IS_ZERO = (cmp_lo_diff == 0).
    //
    // The i64 comparators (i64.eqz / i64.eq / i64.ne) need a hi-limb
    // zero-test too: the full u64 value does not embed injectively into
    // Goldilocks (q = 2^64 - 2^32 + 1 has a nontrivial preimage at
    // lo=1, hi=0xffffffff). We pin COL_CMP_HI_DIFF to the hi-limb input,
    // zero-test it independently, and AND the two flags into COL_CMP_AND.
    (COL_CMP_LO_DIFF, "comparator zero-test input (lo limb / full i32 input)"),
    (COL_CMP_LO_INV, "comparator zero-test inverse witness"),
    (COL_CMP_LO_IS_ZERO, "comparator zero-test result", ColumnWidth::Boolean),
    (
        COL_CMP_HI_DIFF,
        "comparator hi-limb zero-test input (i64.eqz / i64.eq / i64.ne)"
    ),
    (COL_CMP_HI_INV, "comparator hi-limb zero-test inverse witness"),
    (
        COL_CMP_HI_IS_ZERO,
        "comparator hi-limb zero-test result",
        ColumnWidth::Boolean
    ),
    (
        COL_CMP_AND,
        "AND of COL_CMP_LO_IS_ZERO and COL_CMP_HI_IS_ZERO; i64.eqz / i64.eq result",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_OFFSET_IS_0,
        "linear-memory offset case selector for byte offset 0",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_OFFSET_IS_1,
        "linear-memory offset case selector for byte offset 1",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_OFFSET_IS_2,
        "linear-memory offset case selector for byte offset 2",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_OFFSET_IS_3,
        "linear-memory offset case selector for byte offset 3",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_IS_BYTE_WIDTH,
        "linear-memory selector for 8-bit byte-width accesses",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0,
        "linear-memory byte-width offset case selector for byte offset 0",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1,
        "linear-memory byte-width offset case selector for byte offset 1",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2,
        "linear-memory byte-width offset case selector for byte offset 2",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_3,
        "linear-memory byte-width offset case selector for byte offset 3",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_IS_HALF_WIDTH,
        "linear-memory selector for 16-bit half-width accesses",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_0,
        "linear-memory half-width offset case selector for byte offset 0",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_1,
        "linear-memory half-width offset case selector for byte offset 1",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_2,
        "linear-memory half-width offset case selector for byte offset 2",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_3,
        "linear-memory half-width offset case selector for byte offset 3",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_IS_FULL_WIDTH,
        "linear-memory selector for 32-bit full-width accesses",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_0,
        "linear-memory full-width offset case selector for byte offset 0",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_1,
        "linear-memory full-width offset case selector for byte offset 1",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_2,
        "linear-memory full-width offset case selector for byte offset 2",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_3,
        "linear-memory full-width offset case selector for byte offset 3",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_IS_DOUBLE_WIDTH,
        "linear-memory selector for 64-bit double-width accesses",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_0,
        "linear-memory double-width offset case selector for byte offset 0",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_1,
        "linear-memory double-width offset case selector for byte offset 1",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_2,
        "linear-memory double-width offset case selector for byte offset 2",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_3,
        "linear-memory double-width offset case selector for byte offset 3",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_0,
        "i64.load offset case selector for byte offset 0",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_1,
        "i64.load offset case selector for byte offset 1",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_2,
        "i64.load offset case selector for byte offset 2",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_3,
        "i64.load offset case selector for byte offset 3",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_0,
        "i64.store offset case selector for byte offset 0",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_1,
        "i64.store offset case selector for byte offset 1",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_2,
        "i64.store offset case selector for byte offset 2",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_3,
        "i64.store offset case selector for byte offset 3",
        ColumnWidth::Boolean
    ),
    (
        COL_LINEAR_MEM_LANE0_BYTE0,
        "linear-memory first word lane byte 0",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE0_BYTE1,
        "linear-memory first word lane byte 1",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE0_BYTE2,
        "linear-memory first word lane byte 2",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE0_BYTE3,
        "linear-memory first word lane byte 3",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE1_BYTE0,
        "linear-memory second word lane byte 0",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE1_BYTE1,
        "linear-memory second word lane byte 1",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE1_BYTE2,
        "linear-memory second word lane byte 2",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE1_BYTE3,
        "linear-memory second word lane byte 3",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE2_BYTE0,
        "linear-memory third word lane byte 0",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE2_BYTE1,
        "linear-memory third word lane byte 1",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE2_BYTE2,
        "linear-memory third word lane byte 2",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE2_BYTE3,
        "linear-memory third word lane byte 3",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE0_VALUE_BEFORE,
        "linear-memory first word-lane value before this row"
    ),
    (
        COL_LINEAR_MEM_LANE1_VALUE_BEFORE,
        "linear-memory second word-lane value before this row"
    ),
    (
        COL_LINEAR_MEM_LANE2_VALUE_BEFORE,
        "linear-memory third word-lane value before this row"
    ),
    (
        COL_LINEAR_MEM_LANE0_BYTE0_BEFORE,
        "linear-memory first word lane byte 0 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE0_BYTE1_BEFORE,
        "linear-memory first word lane byte 1 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE0_BYTE2_BEFORE,
        "linear-memory first word lane byte 2 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE0_BYTE3_BEFORE,
        "linear-memory first word lane byte 3 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE1_BYTE0_BEFORE,
        "linear-memory second word lane byte 0 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE1_BYTE1_BEFORE,
        "linear-memory second word lane byte 1 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE1_BYTE2_BEFORE,
        "linear-memory second word lane byte 2 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE1_BYTE3_BEFORE,
        "linear-memory second word lane byte 3 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE2_BYTE0_BEFORE,
        "linear-memory third word lane byte 0 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE2_BYTE1_BEFORE,
        "linear-memory third word lane byte 1 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE2_BYTE2_BEFORE,
        "linear-memory third word lane byte 2 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_LANE2_BYTE3_BEFORE,
        "linear-memory third word lane byte 3 before this row",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_ACCESS_BYTE0,
        "linear-memory access value lo byte 0",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_ACCESS_BYTE1,
        "linear-memory access value lo byte 1",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_ACCESS_BYTE2,
        "linear-memory access value lo byte 2",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_ACCESS_BYTE3,
        "linear-memory access value lo byte 3",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_ACCESS_BYTE4,
        "linear-memory access value hi byte 0 (i64 only)",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_ACCESS_BYTE5,
        "linear-memory access value hi byte 1 (i64 only)",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_ACCESS_BYTE6,
        "linear-memory access value hi byte 2 (i64 only)",
        ColumnWidth::Byte
    ),
    (
        COL_LINEAR_MEM_ACCESS_BYTE7,
        "linear-memory access value hi byte 3 (i64 only)",
        ColumnWidth::Byte
    ),
    // Genuine range is 7 bits, [0, 128). Annotated `Byte` as a conservative
    // (over-)approximation so it gets the same enforcement as other byte
    // columns. Tighten to a 7-bit declaration when a `Bits(N)` variant lands.
    (
        COL_SIGN_EXT_LOW7,
        "sign-extension scratch lower 7 bits of the sign source byte",
        ColumnWidth::Byte
    ),
    (
        COL_SIGN_EXT_BIT,
        "sign-extension scratch sign bit",
        ColumnWidth::Boolean
    ),
    (COL_SEL_I32_WRAP_I64, "selector for i32.wrap_i64", ColumnWidth::Boolean),
    (
        COL_SEL_I64_EXTEND_I32_U,
        "selector for i64.extend_i32_u",
        ColumnWidth::Boolean
    ),
    (
        COL_SEL_I64_EXTEND_I32_S,
        "selector for i64.extend_i32_s",
        ColumnWidth::Boolean
    ),
    (
        COL_SEL_I32_EXTEND8_S,
        "selector for i32.extend8_s",
        ColumnWidth::Boolean
    ),
    (
        COL_SEL_I32_EXTEND16_S,
        "selector for i32.extend16_s",
        ColumnWidth::Boolean
    ),
    (
        COL_SEL_I64_EXTEND8_S,
        "selector for i64.extend8_s",
        ColumnWidth::Boolean
    ),
    (
        COL_SEL_I64_EXTEND16_S,
        "selector for i64.extend16_s",
        ColumnWidth::Boolean
    ),
    (
        COL_SEL_I64_EXTEND32_S,
        "selector for i64.extend32_s",
        ColumnWidth::Boolean
    ),
    (COL_SEL_I64_LTS, "selector for i64.lt_s", ColumnWidth::Boolean),
    (COL_SEL_I64_LTU, "selector for i64.lt_u", ColumnWidth::Boolean),
    (COL_SEL_I64_GTS, "selector for i64.gt_s", ColumnWidth::Boolean),
    (COL_SEL_I64_GTU, "selector for i64.gt_u", ColumnWidth::Boolean),
    (COL_SEL_I64_LES, "selector for i64.le_s", ColumnWidth::Boolean),
    (COL_SEL_I64_LEU, "selector for i64.le_u", ColumnWidth::Boolean),
    (COL_SEL_I64_GES, "selector for i64.ge_s", ColumnWidth::Boolean),
    (COL_SEL_I64_GEU, "selector for i64.ge_u", ColumnWidth::Boolean),
    (COL_SEL_I64_SHL, "selector for i64.shl", ColumnWidth::Boolean),
    (COL_SEL_I64_SHR_S, "selector for i64.shr_s", ColumnWidth::Boolean),
    (COL_SEL_I64_SHR_U, "selector for i64.shr_u", ColumnWidth::Boolean),
    (COL_SEL_I64_ROTL, "selector for i64.rotl", ColumnWidth::Boolean),
    (COL_SEL_I64_ROTR, "selector for i64.rotr", ColumnWidth::Boolean),
    (COL_SEL_I64_DIV_S, "selector for i64.div_s", ColumnWidth::Boolean),
    (COL_SEL_I64_DIV_U, "selector for i64.div_u", ColumnWidth::Boolean),
    (COL_SEL_I64_REM_S, "selector for i64.rem_s", ColumnWidth::Boolean),
    (COL_SEL_I64_REM_U, "selector for i64.rem_u", ColumnWidth::Boolean),
    (COL_SEL_I64_CLZ, "selector for i64.clz", ColumnWidth::Boolean),
    (COL_SEL_I64_CTZ, "selector for i64.ctz", ColumnWidth::Boolean),
    (COL_SEL_I64_POPCNT, "selector for i64.popcnt", ColumnWidth::Boolean),
    (
        COL_TRAPPED_BEFORE,
        "carried trapped-execution flag before this row",
        ColumnWidth::Boolean
    ),
    (
        COL_TRAPPED_AFTER,
        "carried trapped-execution flag after this row",
        ColumnWidth::Boolean
    ),
    // Div/rem trap scratch; see the `trap transition` constraints in ccs.rs.
    (
        COL_DIV_DIVISOR_IS_ZERO,
        "zero-test flag for the divisor (stack read1) on this row",
        ColumnWidth::Boolean
    ),
    (COL_DIV_DIVISOR_INV, "inverse witness for the divisor zero test"),
    (
        COL_DIV_TRAP,
        "this row is a div/rem op trapping on a zero divisor or signed overflow",
        ColumnWidth::Boolean
    ),
    (
        COL_DIV_DIVIDEND_IS_MIN,
        "zero-test flag: the dividend (stack read0) equals the active signed div/rem width's MIN",
        ColumnWidth::Boolean
    ),
    (COL_DIV_DIVIDEND_MIN_INV, "inverse witness for the dividend MIN test"),
    (
        COL_DIV_DIVISOR_IS_NEG1,
        "zero-test flag: the divisor (stack read1) equals the active signed div/rem width's -1",
        ColumnWidth::Boolean
    ),
    (COL_DIV_DIVISOR_NEG1_INV, "inverse witness for the divisor -1 test"),
    (
        COL_DIV_OVERFLOW_COND,
        "product of the dividend-is-MIN and divisor-is--1 flags",
        ColumnWidth::Boolean
    ),
    (
        COL_DIV_OVERFLOW,
        "this row is a signed div op trapping on MIN / -1 overflow",
        ColumnWidth::Boolean
    ),
    // call_indirect trap scratch; see `call indirect trap` constraints in
    // ccs/call.rs.
    (
        COL_CI_ENTRY_IS_NULL,
        "zero-test flag: the table entry (table value) read by this row is a null funcref",
        ColumnWidth::Boolean
    ),
    (COL_CI_ENTRY_NULL_INV, "inverse witness for the null-funcref zero test"),
    (
        COL_CI_TYPE_EQ,
        "zero-test flag: callee type id equals the call_indirect expected type id",
        ColumnWidth::Boolean
    ),
    (COL_CI_TYPE_EQ_INV, "inverse witness for the callee-type equality test"),
    (
        COL_CALL_INDIRECT_IS_TRAP,
        "this row is a call_indirect trapping on OOB index, null entry, or callee type mismatch",
        ColumnWidth::Boolean
    ),
    (
        COL_CALL_INDIRECT_IS_NOT_TRAP,
        "non-trapping call_indirect row: gates callee metadata and entry-pc reads",
        ColumnWidth::Boolean
    ),
    (
        COL_FUNCTION_CALL_TYPE_LOOKUP_GATE,
        "call_indirect row with an (in-bounds) non-null table entry: gates the function_types read",
        ColumnWidth::Boolean
    ),
    // Shared unsigned-comparison scratch for the bounds traps (see
    // `push_unsigned_ge_gadget`). `low` is the range-checked borrow-bit
    // remainder; `ge` is `a >= b` for whichever mutually-exclusive comparison
    // the row's opcode selects.
    (
        COL_CMP_LOW,
        "borrow-bit remainder of the active unsigned comparison",
        ColumnWidth::U32
    ),
    (
        COL_CMP_GE,
        "result a >= b of the active unsigned comparison",
        ColumnWidth::Boolean
    ),
    (
        COL_CI_OOB,
        "whether call_indirect traps because the table index is >= the table size",
        ColumnWidth::Boolean
    ),
    (
        COL_TABLE_SIZE_READ_ENABLED,
        "table_sizes read gate: table.size, or call_indirect (binds the size for the OOB check)",
        ColumnWidth::Boolean
    ),
    (
        COL_MEM_OOB,
        "whether this load/store traps because the access is past the end of linear memory",
        ColumnWidth::Boolean
    ),
    (
        COL_MEM_LOAD_LIVE,
        "load lane gate factor: a load row that is not OOB (de-gates lane reads on an OOB trap)",
        ColumnWidth::Boolean
    ),
    (
        COL_MEM_STORE_LIVE,
        "store lane gate factor: a store row that is not OOB (de-gates lane writes on an OOB trap)",
        ColumnWidth::Boolean
    ),
    (
        COL_GROW_SUCCESS,
        "memory.grow row: the growth fits under max pages (before + delta <= max)",
        ColumnWidth::Boolean
    ),
    (
        COL_HALTED_BEFORE,
        "carried terminal flag before this row",
        ColumnWidth::Boolean
    ),
);

pub const SELECTOR_COLS: [usize; 113] = [
    COL_SEL_NOP,
    COL_SEL_I32_CONST,
    COL_SEL_I64_CONST,
    COL_SEL_REF_FUNC,
    COL_SEL_I32_ADD,
    COL_SEL_I64_ADD,
    COL_SEL_I32_SUB,
    COL_SEL_I64_SUB,
    COL_SEL_I32_LOAD,
    COL_SEL_I32_LOAD8_S,
    COL_SEL_I32_LOAD8_U,
    COL_SEL_I32_LOAD16_S,
    COL_SEL_I32_LOAD16_U,
    COL_SEL_I64_LOAD,
    COL_SEL_I32_STORE,
    COL_SEL_I32_STORE8,
    COL_SEL_I32_STORE16,
    COL_SEL_I64_STORE,
    COL_SEL_MEMORY_SIZE,
    COL_SEL_MEMORY_GROW,
    COL_SEL_TABLE_SIZE,
    COL_SEL_TABLE_GET,
    COL_SEL_TABLE_SET,
    COL_SEL_DROP,
    COL_SEL_BR,
    COL_SEL_BLOCK,
    COL_SEL_LOOP,
    COL_SEL_IF,
    COL_SEL_ELSE,
    COL_SEL_END,
    COL_SEL_UNREACHABLE,
    COL_SEL_I32_CLZ,
    COL_SEL_I32_CTZ,
    COL_SEL_I32_POPCNT,
    COL_SEL_I32_EQZ,
    COL_SEL_I64_EQZ,
    COL_SEL_I32_EQ,
    COL_SEL_I32_NE,
    COL_SEL_I32_LTS,
    COL_SEL_I32_LTU,
    COL_SEL_I32_GTS,
    COL_SEL_I32_GTU,
    COL_SEL_I32_LES,
    COL_SEL_I32_LEU,
    COL_SEL_I32_GES,
    COL_SEL_I32_GEU,
    COL_SEL_I32_AND,
    COL_SEL_I32_OR,
    COL_SEL_I32_XOR,
    COL_SEL_I32_MUL,
    COL_SEL_I64_AND,
    COL_SEL_I64_OR,
    COL_SEL_I64_XOR,
    COL_SEL_I64_MUL,
    COL_SEL_I32_SHL,
    COL_SEL_I32_SHR_U,
    COL_SEL_I32_SHR_S,
    COL_SEL_I32_ROTL,
    COL_SEL_I32_ROTR,
    COL_SEL_I32_DIV_U,
    COL_SEL_I32_DIV_S,
    COL_SEL_I32_REM_U,
    COL_SEL_I32_REM_S,
    COL_SEL_SELECT,
    COL_SEL_BR_IF_EQZ,
    COL_SEL_BR_TABLE,
    COL_SEL_CALL,
    COL_SEL_CALL_INDIRECT,
    COL_SEL_RETURN,
    COL_SEL_LOCAL_GET,
    COL_SEL_LOCAL_SET,
    COL_SEL_LOCAL_TEE,
    COL_SEL_GLOBAL_GET,
    COL_SEL_GLOBAL_SET,
    COL_SEL_I64_EQ,
    COL_SEL_I64_NE,
    COL_SEL_I64_STORE8,
    COL_SEL_I64_STORE16,
    COL_SEL_I64_STORE32,
    COL_SEL_I64_LOAD8_U,
    COL_SEL_I64_LOAD16_U,
    COL_SEL_I64_LOAD32_U,
    COL_SEL_I64_LOAD8_S,
    COL_SEL_I64_LOAD16_S,
    COL_SEL_I64_LOAD32_S,
    COL_SEL_I32_WRAP_I64,
    COL_SEL_I64_EXTEND_I32_U,
    COL_SEL_I64_EXTEND_I32_S,
    COL_SEL_I32_EXTEND8_S,
    COL_SEL_I32_EXTEND16_S,
    COL_SEL_I64_EXTEND8_S,
    COL_SEL_I64_EXTEND16_S,
    COL_SEL_I64_EXTEND32_S,
    COL_SEL_I64_LTS,
    COL_SEL_I64_LTU,
    COL_SEL_I64_GTS,
    COL_SEL_I64_GTU,
    COL_SEL_I64_LES,
    COL_SEL_I64_LEU,
    COL_SEL_I64_GES,
    COL_SEL_I64_GEU,
    COL_SEL_I64_SHL,
    COL_SEL_I64_SHR_S,
    COL_SEL_I64_SHR_U,
    COL_SEL_I64_ROTL,
    COL_SEL_I64_ROTR,
    COL_SEL_I64_DIV_S,
    COL_SEL_I64_DIV_U,
    COL_SEL_I64_REM_S,
    COL_SEL_I64_REM_U,
    COL_SEL_I64_CLZ,
    COL_SEL_I64_CTZ,
    COL_SEL_I64_POPCNT,
];

pub fn selector_col(op: WasmOpcode) -> Option<usize> {
    match op {
        WasmOpcode::Nop => Some(COL_SEL_NOP),
        WasmOpcode::I32Const => Some(COL_SEL_I32_CONST),
        WasmOpcode::I64Const => Some(COL_SEL_I64_CONST),
        WasmOpcode::RefFunc => Some(COL_SEL_REF_FUNC),
        WasmOpcode::I32Add => Some(COL_SEL_I32_ADD),
        WasmOpcode::I64Add => Some(COL_SEL_I64_ADD),
        WasmOpcode::I32Sub => Some(COL_SEL_I32_SUB),
        WasmOpcode::I64Sub => Some(COL_SEL_I64_SUB),
        WasmOpcode::I32Load => Some(COL_SEL_I32_LOAD),
        WasmOpcode::I32Load8S => Some(COL_SEL_I32_LOAD8_S),
        WasmOpcode::I32Load8U => Some(COL_SEL_I32_LOAD8_U),
        WasmOpcode::I32Load16S => Some(COL_SEL_I32_LOAD16_S),
        WasmOpcode::I32Load16U => Some(COL_SEL_I32_LOAD16_U),
        WasmOpcode::I64Load => Some(COL_SEL_I64_LOAD),
        WasmOpcode::I32Store => Some(COL_SEL_I32_STORE),
        WasmOpcode::I32Store8 => Some(COL_SEL_I32_STORE8),
        WasmOpcode::I32Store16 => Some(COL_SEL_I32_STORE16),
        WasmOpcode::I64Store => Some(COL_SEL_I64_STORE),
        WasmOpcode::MemorySize => Some(COL_SEL_MEMORY_SIZE),
        WasmOpcode::MemoryGrow => Some(COL_SEL_MEMORY_GROW),
        WasmOpcode::TableSize => Some(COL_SEL_TABLE_SIZE),
        WasmOpcode::TableGet => Some(COL_SEL_TABLE_GET),
        WasmOpcode::TableSet => Some(COL_SEL_TABLE_SET),
        WasmOpcode::Drop => Some(COL_SEL_DROP),
        WasmOpcode::Br => Some(COL_SEL_BR),
        WasmOpcode::Block => Some(COL_SEL_BLOCK),
        WasmOpcode::Loop => Some(COL_SEL_LOOP),
        WasmOpcode::If => Some(COL_SEL_IF),
        WasmOpcode::Else => Some(COL_SEL_ELSE),
        WasmOpcode::End => Some(COL_SEL_END),
        WasmOpcode::Unreachable => Some(COL_SEL_UNREACHABLE),
        WasmOpcode::I32Clz => Some(COL_SEL_I32_CLZ),
        WasmOpcode::I32Ctz => Some(COL_SEL_I32_CTZ),
        WasmOpcode::I32Popcnt => Some(COL_SEL_I32_POPCNT),
        WasmOpcode::I32Eqz => Some(COL_SEL_I32_EQZ),
        WasmOpcode::I64Eqz => Some(COL_SEL_I64_EQZ),
        WasmOpcode::I32Eq => Some(COL_SEL_I32_EQ),
        WasmOpcode::I32Ne => Some(COL_SEL_I32_NE),
        WasmOpcode::I64Eq => Some(COL_SEL_I64_EQ),
        WasmOpcode::I64Ne => Some(COL_SEL_I64_NE),
        WasmOpcode::I64Store8 => Some(COL_SEL_I64_STORE8),
        WasmOpcode::I64Store16 => Some(COL_SEL_I64_STORE16),
        WasmOpcode::I64Store32 => Some(COL_SEL_I64_STORE32),
        WasmOpcode::I64Load8U => Some(COL_SEL_I64_LOAD8_U),
        WasmOpcode::I64Load16U => Some(COL_SEL_I64_LOAD16_U),
        WasmOpcode::I64Load32U => Some(COL_SEL_I64_LOAD32_U),
        WasmOpcode::I64Load8S => Some(COL_SEL_I64_LOAD8_S),
        WasmOpcode::I64Load16S => Some(COL_SEL_I64_LOAD16_S),
        WasmOpcode::I64Load32S => Some(COL_SEL_I64_LOAD32_S),
        WasmOpcode::I32WrapI64 => Some(COL_SEL_I32_WRAP_I64),
        WasmOpcode::I64ExtendI32U => Some(COL_SEL_I64_EXTEND_I32_U),
        WasmOpcode::I64ExtendI32S => Some(COL_SEL_I64_EXTEND_I32_S),
        WasmOpcode::I32Extend8S => Some(COL_SEL_I32_EXTEND8_S),
        WasmOpcode::I32Extend16S => Some(COL_SEL_I32_EXTEND16_S),
        WasmOpcode::I64Extend8S => Some(COL_SEL_I64_EXTEND8_S),
        WasmOpcode::I64Extend16S => Some(COL_SEL_I64_EXTEND16_S),
        WasmOpcode::I64Extend32S => Some(COL_SEL_I64_EXTEND32_S),
        WasmOpcode::I32LtS => Some(COL_SEL_I32_LTS),
        WasmOpcode::I32LtU => Some(COL_SEL_I32_LTU),
        WasmOpcode::I32GtS => Some(COL_SEL_I32_GTS),
        WasmOpcode::I32GtU => Some(COL_SEL_I32_GTU),
        WasmOpcode::I32LeS => Some(COL_SEL_I32_LES),
        WasmOpcode::I32LeU => Some(COL_SEL_I32_LEU),
        WasmOpcode::I32GeS => Some(COL_SEL_I32_GES),
        WasmOpcode::I32GeU => Some(COL_SEL_I32_GEU),
        WasmOpcode::I32And => Some(COL_SEL_I32_AND),
        WasmOpcode::I32Or => Some(COL_SEL_I32_OR),
        WasmOpcode::I32Xor => Some(COL_SEL_I32_XOR),
        WasmOpcode::I32Mul => Some(COL_SEL_I32_MUL),
        WasmOpcode::I64And => Some(COL_SEL_I64_AND),
        WasmOpcode::I64Or => Some(COL_SEL_I64_OR),
        WasmOpcode::I64Xor => Some(COL_SEL_I64_XOR),
        WasmOpcode::I64Mul => Some(COL_SEL_I64_MUL),
        WasmOpcode::I32Shl => Some(COL_SEL_I32_SHL),
        WasmOpcode::I32ShrU => Some(COL_SEL_I32_SHR_U),
        WasmOpcode::I32ShrS => Some(COL_SEL_I32_SHR_S),
        WasmOpcode::I32Rotl => Some(COL_SEL_I32_ROTL),
        WasmOpcode::I32Rotr => Some(COL_SEL_I32_ROTR),
        WasmOpcode::I32DivU => Some(COL_SEL_I32_DIV_U),
        WasmOpcode::I32DivS => Some(COL_SEL_I32_DIV_S),
        WasmOpcode::I32RemU => Some(COL_SEL_I32_REM_U),
        WasmOpcode::I32RemS => Some(COL_SEL_I32_REM_S),
        WasmOpcode::I64LtS => Some(COL_SEL_I64_LTS),
        WasmOpcode::I64LtU => Some(COL_SEL_I64_LTU),
        WasmOpcode::I64GtS => Some(COL_SEL_I64_GTS),
        WasmOpcode::I64GtU => Some(COL_SEL_I64_GTU),
        WasmOpcode::I64LeS => Some(COL_SEL_I64_LES),
        WasmOpcode::I64LeU => Some(COL_SEL_I64_LEU),
        WasmOpcode::I64GeS => Some(COL_SEL_I64_GES),
        WasmOpcode::I64GeU => Some(COL_SEL_I64_GEU),
        WasmOpcode::I64Shl => Some(COL_SEL_I64_SHL),
        WasmOpcode::I64ShrS => Some(COL_SEL_I64_SHR_S),
        WasmOpcode::I64ShrU => Some(COL_SEL_I64_SHR_U),
        WasmOpcode::I64Rotl => Some(COL_SEL_I64_ROTL),
        WasmOpcode::I64Rotr => Some(COL_SEL_I64_ROTR),
        WasmOpcode::I64DivS => Some(COL_SEL_I64_DIV_S),
        WasmOpcode::I64DivU => Some(COL_SEL_I64_DIV_U),
        WasmOpcode::I64RemS => Some(COL_SEL_I64_REM_S),
        WasmOpcode::I64RemU => Some(COL_SEL_I64_REM_U),
        WasmOpcode::I64Clz => Some(COL_SEL_I64_CLZ),
        WasmOpcode::I64Ctz => Some(COL_SEL_I64_CTZ),
        WasmOpcode::I64Popcnt => Some(COL_SEL_I64_POPCNT),
        WasmOpcode::Select => Some(COL_SEL_SELECT),
        WasmOpcode::BrIf => Some(COL_SEL_BR_IF_EQZ),
        WasmOpcode::BrTable => Some(COL_SEL_BR_TABLE),
        WasmOpcode::Call => Some(COL_SEL_CALL),
        WasmOpcode::CallIndirect => Some(COL_SEL_CALL_INDIRECT),
        WasmOpcode::Return => Some(COL_SEL_RETURN),
        WasmOpcode::LocalGet => Some(COL_SEL_LOCAL_GET),
        WasmOpcode::LocalSet => Some(COL_SEL_LOCAL_SET),
        WasmOpcode::LocalTee => Some(COL_SEL_LOCAL_TEE),
        WasmOpcode::GlobalGet => Some(COL_SEL_GLOBAL_GET),
        WasmOpcode::GlobalSet => Some(COL_SEL_GLOBAL_SET),
        // Call correctness is enforced by Stage 2 (call stack) and Stage 3 (continuity),
        // not by a CCS row constraint. No selector column needed.
        WasmOpcode::Trap | WasmOpcode::Unsupported => None,
    }
}

pub fn build_pad_row() -> [F; NAMED_COLUMN_COUNT] {
    let mut row = [F::ZERO; NAMED_COLUMN_COUNT];
    row[COL_ONE] = F::ONE;
    row[COL_OPCODE_CODE] = F::from_u64(u64::from(opcode_code(WasmOpcode::Return)));
    row[COL_IS_PROGRAM_ROW] = F::ONE;
    row[COL_PC_EDGE_KIND] = F::ONE;
    row[COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO] = F::ONE;
    row[COL_HOST_ARGS_REMAINING_AFTER_IS_ZERO] = F::ONE;
    row[COL_HALTED] = F::ONE;
    row[COL_SEL_RETURN] = F::ONE;
    row
}
