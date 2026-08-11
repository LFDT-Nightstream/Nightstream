//! Owns the static WASM row layout.

use super::isa::WasmOpcode;
use crate::column_registry::define_column_region;
pub use crate::column_registry::{ColumnWidth, WasmColumnSpec};
pub use crate::host_event_layout::*;

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

define_column_region! {
    region: "wasm_named",
    start: 0usize,
    width: pub WASM_COLUMN_COUNT,
    specs: pub WASM_COLUMN_SPECS,
    indices: pub,
    columns: [
        COL_ONE: Field => "",
        COL_OUTPUT_ENABLED_BEFORE: Field => "carried simple-output flag before this row",
        COL_OUTPUT_ENABLED_AFTER: Field => "carried simple-output flag after this row",
        COL_OUTPUT_VALUE_LO_BEFORE: Field => "carried output low limb before",
        COL_OUTPUT_VALUE_LO_AFTER: Field => "carried output low limb after",
        COL_OUTPUT_VALUE_HI_BEFORE: Field => "carried output high limb before",
        COL_OUTPUT_VALUE_HI_AFTER: Field => "carried output high limb after",
        COL_OPCODE_CODE: U32 => "opcode decode selector source",
        COL_PC_BEFORE: U32 => "transition source pc",
        COL_PC_AFTER: U32 => "transition destination pc",
        COL_CONTROL_CHOICE: U32 => "control-edge choice",
        COL_PC_EDGE_KIND: U32 => "pc edge kind",
        COL_WIDE_VALUES_ENABLED: Boolean => "wide-value flag",
        COL_OUTPUT_CAPTURED: Boolean => "output-capture gate",
        COL_SP_BEFORE: U32 => "transition source stack pointer",
        COL_SP_AFTER: U32 => "transition destination stack pointer",
        COL_STACK_FRAME_BASE_BEFORE: U32 => "operand-stack frame base before",
        COL_STACK_FRAME_BASE_AFTER: U32 => "operand-stack frame base after",
        COL_HALTED: Boolean => "terminal row flag",
        COL_IS_PROGRAM_ROW: Boolean => "decoded wasm program row",
        COL_PC_ROM_ACTIVE: Boolean => "static pc-edge ROM gate",
        COL_PC_EDGE_KIND_IS_STATIC: Boolean => "static pc-edge flag",
        COL_PC_EDGE_KIND_INV: Field => "inverse witness for pc edge-kind zero test",
        COL_PARAM_INIT_ACTIVE_BEFORE: Boolean => "parameter-init mode before",
        COL_PARAM_INIT_ACTIVE_AFTER: Boolean => "parameter-init mode after",
        COL_TAIL_CALL_PENDING_BEFORE: Boolean => "tail-entry pending before",
        COL_TAIL_CALL_PENDING_AFTER: Boolean => "tail-entry pending after",
        COL_TAIL_ENTER_ACTIVE: Boolean => "tail-enter aux row",
        COL_TAIL_DISCARD_COUNT: U32 => "tail-enter discard count",
        COL_PADDING_ACTIVE: Boolean => "padding row",
        COL_PARAM_INIT_REMAINING_BEFORE: U32 => "remaining call parameters to initialize before this row",
        COL_PARAM_INIT_REMAINING_AFTER: U32 => "remaining call parameters to initialize after this row",
        COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO: Boolean =>
            "zero-test flag for remaining call parameters after this row",
        COL_PARAM_INIT_REMAINING_AFTER_INV: Field =>
            "inverse witness for remaining call parameters after this row",
        COL_GUEST_ENTRY_ACTIVE: Boolean => "guest-entry row flag: this row enters a traced guest callee",
        COL_CALL_STACK_PUSH_PRESENT: Boolean => "flag indicating that this row saves a caller return context",
        COL_CALL_STACK_POP_PRESENT: Boolean =>
            "flag indicating that this row restores a saved caller return context",
        COL_CALL_STACK_RETURN_PC_VALUE: U32 =>
            "call-stack return-pc cell value: written on guest-call push, read back on pop into pc_after",
        COL_CALL_STACK_CALLER_FBP_VALUE: U32 =>
            "call-stack caller-fbp cell value: written on guest-call push, read back on pop into locals_fbp_after",
        COL_CALL_STACK_CALLER_SP_BASE_VALUE: U32 =>
            "call-stack caller operand-stack base: written on push and restored on pop",
        COL_CALL_STACK_DEPTH_BEFORE: U32 => "call return-context stack depth before this row",
        COL_CALL_STACK_DEPTH_AFTER: U32 => "call return-context stack depth after this row",
        COL_CALL_STACK_ADDR: U32 => "call return-context stack address read or written this row",
        COL_PC_ROM_CALL_RETURN_CHOICE: U32 =>
            "pc-rom control-choice coordinate for the call-site continuation-pc read (guest pushes and indirect host fall-through)",
        COL_CURRENT_FUNCTION_REF: U32 => "normalized function reference for the currently executing frame",
        COL_CURRENT_FUNCTION_NUM_LOCALS: U32 => "number of locals in the current function frame",
        COL_STACK_READS: U32 => "stack delta source",
        COL_STACK_WRITES: U32 => "stack delta destination",
        COL_STACK_READ_ACTIVE: [Boolean; 3] => "operand-stack read activity flags",
        COL_STACK_WRITE0_ACTIVE: Boolean => "stack lane 0 write activity flag",
        COL_STACK_WRITE0_HI_ACTIVE: Boolean =>
            "stack write hi-word port gate: write0_active, plus result-hi gather rows writing only the hi lane",
        COL_OP_TABLE_ENABLED: Boolean => "lookup gate",
        COL_SEL_NOP: Boolean => "",
        COL_SEL_I32_CONST: Boolean => "",
        COL_SEL_I64_CONST: Boolean => "",
        COL_SEL_REF_FUNC: Boolean => "",
        COL_SEL_I32_ADD: Boolean => "",
        COL_SEL_I64_ADD: Boolean => "",
        COL_SEL_I32_SUB: Boolean => "",
        COL_SEL_I64_SUB: Boolean => "",
        COL_SEL_I32_LOAD: Boolean => "",
        COL_SEL_I32_LOAD8_S: Boolean => "",
        COL_SEL_I32_LOAD8_U: Boolean => "",
        COL_SEL_I32_LOAD16_S: Boolean => "",
        COL_SEL_I32_LOAD16_U: Boolean => "",
        COL_SEL_I64_LOAD: Boolean => "",
        COL_SEL_I32_STORE: Boolean => "",
        COL_SEL_I32_STORE8: Boolean => "",
        COL_SEL_I32_STORE16: Boolean => "",
        COL_SEL_I64_STORE: Boolean => "",
        COL_SEL_MEMORY_SIZE: Boolean => "",
        COL_SEL_MEMORY_GROW: Boolean => "",
        COL_SEL_TABLE_SIZE: Boolean => "",
        COL_SEL_TABLE_GET: Boolean => "",
        COL_SEL_TABLE_SET: Boolean => "",
        COL_SEL_DROP: Boolean => "",
        COL_SEL_BR: Boolean => "",
        COL_SEL_BLOCK: Boolean => "",
        COL_SEL_LOOP: Boolean => "",
        COL_SEL_IF: Boolean => "",
        COL_SEL_ELSE: Boolean => "",
        COL_SEL_END: Boolean => "",
        COL_SEL_UNREACHABLE: Boolean => "",
        COL_SEL_I32_CLZ: Boolean => "",
        COL_SEL_I32_CTZ: Boolean => "",
        COL_SEL_I32_POPCNT: Boolean => "",
        COL_SEL_I32_EQZ: Boolean => "",
        COL_SEL_I64_EQZ: Boolean => "",
        COL_SEL_I32_EQ: Boolean => "",
        COL_SEL_I32_NE: Boolean => "",
        COL_SEL_I32_LTS: Boolean => "",
        COL_SEL_I32_LTU: Boolean => "",
        COL_SEL_I32_GTS: Boolean => "",
        COL_SEL_I32_GTU: Boolean => "",
        COL_SEL_I32_LES: Boolean => "",
        COL_SEL_I32_LEU: Boolean => "",
        COL_SEL_I32_GES: Boolean => "",
        COL_SEL_I32_GEU: Boolean => "",
        COL_SEL_I32_AND: Boolean => "",
        COL_SEL_I32_OR: Boolean => "",
        COL_SEL_I32_XOR: Boolean => "",
        COL_SEL_I32_MUL: Boolean => "",
        COL_SEL_I64_AND: Boolean => "",
        COL_SEL_I64_OR: Boolean => "",
        COL_SEL_I64_XOR: Boolean => "",
        COL_SEL_I64_MUL: Boolean => "",
        COL_SEL_I32_SHL: Boolean => "",
        COL_SEL_I32_SHR_U: Boolean => "",
        COL_SEL_I32_SHR_S: Boolean => "",
        COL_SEL_I32_ROTL: Boolean => "",
        COL_SEL_I32_ROTR: Boolean => "",
        COL_SEL_I32_DIV_U: Boolean => "",
        COL_SEL_I32_DIV_S: Boolean => "",
        COL_SEL_I32_REM_U: Boolean => "",
        COL_SEL_I32_REM_S: Boolean => "",
        COL_SEL_SELECT: Boolean => "",
        COL_SEL_BR_IF_EQZ: Boolean => "",
        COL_SEL_BR_TABLE: Boolean => "",
        COL_SEL_CALL: Boolean => "",
        COL_SEL_CALL_INDIRECT: Boolean => "",
        COL_SEL_RETURN: Boolean => "",
        COL_SEL_LOCAL_GET: Boolean => "",
        COL_SEL_LOCAL_SET: Boolean => "",
        COL_SEL_LOCAL_TEE: Boolean => "",
        COL_SEL_GLOBAL_GET: Boolean => "",
        COL_SEL_GLOBAL_SET: Boolean => "",
        COL_SEL_I64_EQ: Boolean => "",
        COL_SEL_I64_NE: Boolean => "",
        COL_SEL_I64_STORE8: Boolean => "",
        COL_SEL_I64_STORE16: Boolean => "",
        COL_SEL_I64_STORE32: Boolean => "",
        COL_SEL_I64_LOAD8_U: Boolean => "",
        COL_SEL_I64_LOAD16_U: Boolean => "",
        COL_SEL_I64_LOAD32_U: Boolean => "",
        COL_SEL_I64_LOAD8_S: Boolean => "",
        COL_SEL_I64_LOAD16_S: Boolean => "",
        COL_SEL_I64_LOAD32_S: Boolean => "",
        COL_LOCAL_WRITE_ENABLED: Boolean => "locals memory write gate for local.set/local.tee",
        COL_PROGRAM_LOCAL_INDEX_ACTIVE: Boolean => "PC-indexed local immediate ROM gate",
        COL_PROGRAM_GLOBAL_INDEX_ACTIVE: Boolean => "PC-indexed global immediate ROM gate",
        COL_PROGRAM_TABLE_ID_ACTIVE: Boolean => "PC-indexed table-id immediate ROM gate",
        COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE: Boolean => "PC-indexed call-indirect immediate ROM gate",
        COL_TABLE_READ_ENABLED: Boolean => "table memory read gate for table.get and indirect calls",
        COL_LOCALS_FBP_BEFORE: U32 => "locals memory frame base before this row",
        COL_LOCALS_FBP_AFTER: U32 => "locals memory frame base after this row",
        COL_LOCAL_INDEX: U32 => "locals memory offset",
        COL_LOCAL_VALUE: U32 => "locals memory value",
        COL_LOCAL_VALUE_HI: U32 => "locals memory high limb for future i64 support",
        COL_GLOBAL_INDEX: U32 => "globals memory index",
        COL_GLOBAL_VALUE: U32 => "globals memory value",
        COL_GLOBAL_VALUE_HI: U32 => "globals memory high limb for future i64 support",
        COL_TABLE_ID: U32 => "table state namespace selector",
        COL_TABLE_INDEX: U32 => "table element index",
        COL_TABLE_VALUE: U32 => "normalized table element value observed by this step",
        COL_TABLE_SIZE: U32 => "size of the referenced table observed by this step",
        COL_FUNCTION_REF: U32 => "normalized function reference selected by call-like opcodes",
        COL_CALL_PARAM_COUNT: Byte => "parameter count for the selected call target",
        COL_CALL_RESULT_COUNT: Byte => "result count for the selected call target",
        COL_TARGET_FUNCTION_IS_GUEST: Boolean => "true when the selected call target is a guest-defined function",
        COL_CALL_TARGET_METADATA: Field =>
            "packed call-target arity and guest flag; range follows from unpacking",
        COL_CALL_INDIRECT_TYPE_INDEX: U32 =>
            "raw module type-section index from the call_indirect instruction immediate",
        COL_FUNCTION_TYPE_ID: U32 => "normalized deduplicated type id for the observed function reference",
        COL_EXPECTED_TYPE_ID: U32 => "normalized deduplicated type id expected by the current opcode",
        COL_MEMORY_PAGES_BEFORE: U32 => "linear memory page count before this step",
        COL_MEMORY_PAGES_AFTER: U32 => "linear memory page count after this step",
        COL_MAX_MEMORY_PAGES_BEFORE: U32 =>
            "verifier-authoritative max linear-memory page count before this step (carried constant)",
        COL_MAX_MEMORY_PAGES_AFTER: U32 =>
            "verifier-authoritative max linear-memory page count after this step (carried constant)",
        COL_STACK_READ_ADDR_LO: [U32; 3] => "operand-stack read low-limb physical addresses",
        COL_STACK_READ_ADDR_HI: [U32; 3] => "operand-stack read high-limb physical addresses",
        COL_STACK_READ_VALUE_LO: [U32; 3] => "operand-stack read low-limb values",
        COL_STACK_READ_VALUE_HI: [U32; 3] => "operand-stack read high-limb values",
        COL_STACK_WRITE0_ADDR_LO: U32 => "operand-stack write lane 0 low-limb physical address",
        COL_STACK_WRITE0_ADDR_HI: U32 => "operand-stack write lane 0 high-limb physical address",
        COL_STACK_WRITE0_VALUE_LO: U32 => "operand-stack write lane 0 value",
        COL_STACK_WRITE0_VALUE_HI: U32 => "operand-stack write lane 0 high limb for future i64 support",
        COL_WIDE_AUX0: Boolean => "",
        COL_WIDE_AUX1: Boolean => "",
        COL_LINEAR_MEM_IMM_OFFSET: U32 => "linear-memory immediate offset in bytes",
        COL_LINEAR_MEM_BYTE_OFFSET: U32 => "linear-memory byte offset within the first word lane",
        COL_LINEAR_MEM_USE_LANE1: Boolean => "linear-memory second-lane flag for unaligned accesses",
        COL_LINEAR_MEM_USE_LANE2: Boolean => "linear-memory third-lane flag for wide unaligned accesses",
        COL_LINEAR_MEM_USE_LANE0: Boolean => "linear-memory first-lane activity gate",
        COL_LINEAR_MEM_LANE_LOAD_ACTIVE: [Boolean; 3] =>
            "linear-memory word-lane load gates (lane is used AND opcode is a load)",
        COL_LINEAR_MEM_LANE_STORE_ACTIVE: [Boolean; 3] =>
            "linear-memory word-lane store gates (lane is used AND opcode is a store)",
        COL_LINEAR_MEM_LANE_ADDR: [U32; 3] => "linear-memory word-lane addresses",
        COL_LINEAR_MEM_LANE_VALUE: [Field; 3] => "linear-memory word-lane accessed values",
        COL_OP_TABLE_ID: U32 => "lookup table row selector",
        COL_OP_TABLE_VALUE: Field => "lookup payload witness",
        // `COL_SELECT_COND_IS_ZERO` is forced to {0, 1} by the zero-test rows
        // emitted by `push_select_constraints`. Declared `Boolean` so the
        // spec reflects its actual range; whichever path eventually enforces
        // `ColumnWidth::Boolean` will overlap with the gadget's constraint and
        // one of the two becomes redundant; an optimization to revisit later.
        // We deliberately do not introduce a dedicated `ImpliedBoolean` width
        // for this case because that would be premature complexity.
        COL_SELECT_COND_IS_ZERO: Boolean => "scratch column for push_select_constraints for select opcode",
        COL_SELECT_SCRATCH_INV: Field => "scratch column for push_select_constraints for select opcode",
        COL_SELECT_OUT_DELTA_LO: Field => "scratch column for push_select_constraints low-limb mux",
        COL_SELECT_OUT_DELTA_HI: Field => "scratch column for push_select_constraints high-limb mux",
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
        COL_CMP_LO_DIFF: Field => "comparator zero-test input (lo limb / full i32 input)",
        COL_CMP_LO_INV: Field => "comparator zero-test inverse witness",
        COL_CMP_LO_IS_ZERO: Boolean => "comparator zero-test result",
        COL_CMP_HI_DIFF: Field => "comparator hi-limb zero-test input (i64.eqz / i64.eq / i64.ne)",
        COL_CMP_HI_INV: Field => "comparator hi-limb zero-test inverse witness",
        COL_CMP_HI_IS_ZERO: Boolean => "comparator hi-limb zero-test result",
        COL_CMP_AND: Boolean => "AND of COL_CMP_LO_IS_ZERO and COL_CMP_HI_IS_ZERO; i64.eqz / i64.eq result",
        COL_LINEAR_MEM_OFFSET_IS_0: Boolean => "linear-memory offset case selector for byte offset 0",
        COL_LINEAR_MEM_OFFSET_IS_1: Boolean => "linear-memory offset case selector for byte offset 1",
        COL_LINEAR_MEM_OFFSET_IS_2: Boolean => "linear-memory offset case selector for byte offset 2",
        COL_LINEAR_MEM_OFFSET_IS_3: Boolean => "linear-memory offset case selector for byte offset 3",
        COL_LINEAR_MEM_IS_BYTE_WIDTH: Boolean => "linear-memory selector for 8-bit byte-width accesses",
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0: Boolean =>
            "linear-memory byte-width offset case selector for byte offset 0",
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1: Boolean =>
            "linear-memory byte-width offset case selector for byte offset 1",
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2: Boolean =>
            "linear-memory byte-width offset case selector for byte offset 2",
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_3: Boolean =>
            "linear-memory byte-width offset case selector for byte offset 3",
        COL_LINEAR_MEM_IS_HALF_WIDTH: Boolean => "linear-memory selector for 16-bit half-width accesses",
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_0: Boolean =>
            "linear-memory half-width offset case selector for byte offset 0",
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_1: Boolean =>
            "linear-memory half-width offset case selector for byte offset 1",
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_2: Boolean =>
            "linear-memory half-width offset case selector for byte offset 2",
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_3: Boolean =>
            "linear-memory half-width offset case selector for byte offset 3",
        COL_LINEAR_MEM_IS_FULL_WIDTH: Boolean => "linear-memory selector for 32-bit full-width accesses",
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_0: Boolean =>
            "linear-memory full-width offset case selector for byte offset 0",
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_1: Boolean =>
            "linear-memory full-width offset case selector for byte offset 1",
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_2: Boolean =>
            "linear-memory full-width offset case selector for byte offset 2",
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_3: Boolean =>
            "linear-memory full-width offset case selector for byte offset 3",
        COL_LINEAR_MEM_IS_DOUBLE_WIDTH: Boolean => "linear-memory selector for 64-bit double-width accesses",
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_0: Boolean =>
            "linear-memory double-width offset case selector for byte offset 0",
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_1: Boolean =>
            "linear-memory double-width offset case selector for byte offset 1",
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_2: Boolean =>
            "linear-memory double-width offset case selector for byte offset 2",
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_3: Boolean =>
            "linear-memory double-width offset case selector for byte offset 3",
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_0: Boolean => "i64.load offset case selector for byte offset 0",
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_1: Boolean => "i64.load offset case selector for byte offset 1",
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_2: Boolean => "i64.load offset case selector for byte offset 2",
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_3: Boolean => "i64.load offset case selector for byte offset 3",
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_0: Boolean => "i64.store offset case selector for byte offset 0",
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_1: Boolean => "i64.store offset case selector for byte offset 1",
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_2: Boolean => "i64.store offset case selector for byte offset 2",
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_3: Boolean => "i64.store offset case selector for byte offset 3",
        COL_LINEAR_MEM_LANE0_BYTE0: Byte => "linear-memory first word lane byte 0",
        COL_LINEAR_MEM_LANE0_BYTE1: Byte => "linear-memory first word lane byte 1",
        COL_LINEAR_MEM_LANE0_BYTE2: Byte => "linear-memory first word lane byte 2",
        COL_LINEAR_MEM_LANE0_BYTE3: Byte => "linear-memory first word lane byte 3",
        COL_LINEAR_MEM_LANE1_BYTE0: Byte => "linear-memory second word lane byte 0",
        COL_LINEAR_MEM_LANE1_BYTE1: Byte => "linear-memory second word lane byte 1",
        COL_LINEAR_MEM_LANE1_BYTE2: Byte => "linear-memory second word lane byte 2",
        COL_LINEAR_MEM_LANE1_BYTE3: Byte => "linear-memory second word lane byte 3",
        COL_LINEAR_MEM_LANE2_BYTE0: Byte => "linear-memory third word lane byte 0",
        COL_LINEAR_MEM_LANE2_BYTE1: Byte => "linear-memory third word lane byte 1",
        COL_LINEAR_MEM_LANE2_BYTE2: Byte => "linear-memory third word lane byte 2",
        COL_LINEAR_MEM_LANE2_BYTE3: Byte => "linear-memory third word lane byte 3",
        COL_LINEAR_MEM_LANE_VALUE_BEFORE: [Field; 3] => "linear-memory word-lane values before this row",
        COL_LINEAR_MEM_LANE0_BYTE0_BEFORE: Byte => "linear-memory first word lane byte 0 before this row",
        COL_LINEAR_MEM_LANE0_BYTE1_BEFORE: Byte => "linear-memory first word lane byte 1 before this row",
        COL_LINEAR_MEM_LANE0_BYTE2_BEFORE: Byte => "linear-memory first word lane byte 2 before this row",
        COL_LINEAR_MEM_LANE0_BYTE3_BEFORE: Byte => "linear-memory first word lane byte 3 before this row",
        COL_LINEAR_MEM_LANE1_BYTE0_BEFORE: Byte => "linear-memory second word lane byte 0 before this row",
        COL_LINEAR_MEM_LANE1_BYTE1_BEFORE: Byte => "linear-memory second word lane byte 1 before this row",
        COL_LINEAR_MEM_LANE1_BYTE2_BEFORE: Byte => "linear-memory second word lane byte 2 before this row",
        COL_LINEAR_MEM_LANE1_BYTE3_BEFORE: Byte => "linear-memory second word lane byte 3 before this row",
        COL_LINEAR_MEM_LANE2_BYTE0_BEFORE: Byte => "linear-memory third word lane byte 0 before this row",
        COL_LINEAR_MEM_LANE2_BYTE1_BEFORE: Byte => "linear-memory third word lane byte 1 before this row",
        COL_LINEAR_MEM_LANE2_BYTE2_BEFORE: Byte => "linear-memory third word lane byte 2 before this row",
        COL_LINEAR_MEM_LANE2_BYTE3_BEFORE: Byte => "linear-memory third word lane byte 3 before this row",
        COL_LINEAR_MEM_ACCESS_BYTE0: Byte => "linear-memory access value lo byte 0",
        COL_LINEAR_MEM_ACCESS_BYTE1: Byte => "linear-memory access value lo byte 1",
        COL_LINEAR_MEM_ACCESS_BYTE2: Byte => "linear-memory access value lo byte 2",
        COL_LINEAR_MEM_ACCESS_BYTE3: Byte => "linear-memory access value lo byte 3",
        COL_LINEAR_MEM_ACCESS_BYTE4: Byte => "linear-memory access value hi byte 0 (i64 only)",
        COL_LINEAR_MEM_ACCESS_BYTE5: Byte => "linear-memory access value hi byte 1 (i64 only)",
        COL_LINEAR_MEM_ACCESS_BYTE6: Byte => "linear-memory access value hi byte 2 (i64 only)",
        COL_LINEAR_MEM_ACCESS_BYTE7: Byte => "linear-memory access value hi byte 3 (i64 only)",
        // Genuine range is 7 bits, [0, 128). Annotated `Byte` as a conservative
        // (over-)approximation so it gets the same enforcement as other byte
        // columns. Tighten to a 7-bit declaration when a `Bits(N)` variant lands.
        COL_SIGN_EXT_LOW7: Byte => "sign-extension scratch lower 7 bits of the sign source byte",
        COL_SIGN_EXT_BIT: Boolean => "sign-extension scratch sign bit",
        COL_SEL_I32_WRAP_I64: Boolean => "selector for i32.wrap_i64",
        COL_SEL_I64_EXTEND_I32_U: Boolean => "selector for i64.extend_i32_u",
        COL_SEL_I64_EXTEND_I32_S: Boolean => "selector for i64.extend_i32_s",
        COL_SEL_I32_EXTEND8_S: Boolean => "selector for i32.extend8_s",
        COL_SEL_I32_EXTEND16_S: Boolean => "selector for i32.extend16_s",
        COL_SEL_I64_EXTEND8_S: Boolean => "selector for i64.extend8_s",
        COL_SEL_I64_EXTEND16_S: Boolean => "selector for i64.extend16_s",
        COL_SEL_I64_EXTEND32_S: Boolean => "selector for i64.extend32_s",
        COL_SEL_I64_LTS: Boolean => "selector for i64.lt_s",
        COL_SEL_I64_LTU: Boolean => "selector for i64.lt_u",
        COL_SEL_I64_GTS: Boolean => "selector for i64.gt_s",
        COL_SEL_I64_GTU: Boolean => "selector for i64.gt_u",
        COL_SEL_I64_LES: Boolean => "selector for i64.le_s",
        COL_SEL_I64_LEU: Boolean => "selector for i64.le_u",
        COL_SEL_I64_GES: Boolean => "selector for i64.ge_s",
        COL_SEL_I64_GEU: Boolean => "selector for i64.ge_u",
        COL_SEL_I64_SHL: Boolean => "selector for i64.shl",
        COL_SEL_I64_SHR_S: Boolean => "selector for i64.shr_s",
        COL_SEL_I64_SHR_U: Boolean => "selector for i64.shr_u",
        COL_SEL_I64_ROTL: Boolean => "selector for i64.rotl",
        COL_SEL_I64_ROTR: Boolean => "selector for i64.rotr",
        COL_SEL_I64_DIV_S: Boolean => "selector for i64.div_s",
        COL_SEL_I64_DIV_U: Boolean => "selector for i64.div_u",
        COL_SEL_I64_REM_S: Boolean => "selector for i64.rem_s",
        COL_SEL_I64_REM_U: Boolean => "selector for i64.rem_u",
        COL_SEL_I64_CLZ: Boolean => "selector for i64.clz",
        COL_SEL_I64_CTZ: Boolean => "selector for i64.ctz",
        COL_SEL_I64_POPCNT: Boolean => "selector for i64.popcnt",
        COL_SEL_RETURN_CALL: Boolean => "selector for return_call",
        COL_SEL_RETURN_CALL_INDIRECT: Boolean => "selector for return_call_indirect",
        COL_TRAPPED_BEFORE: Boolean => "carried trapped-execution flag before this row",
        COL_TRAPPED_AFTER: Boolean => "carried trapped-execution flag after this row",
        // Div/rem trap scratch; see the `trap transition` constraints in ccs.rs.
        COL_DIV_DIVISOR_IS_ZERO: Boolean => "zero-test flag for the divisor (stack read1) on this row",
        COL_DIV_DIVISOR_INV: Field => "inverse witness for the divisor zero test",
        COL_DIV_TRAP: Boolean => "this row is a div/rem op trapping on a zero divisor or signed overflow",
        COL_DIV_DIVIDEND_IS_MIN: Boolean =>
            "zero-test flag: the dividend (stack read0) equals the active signed div/rem width's MIN",
        COL_DIV_DIVIDEND_MIN_INV: Field => "inverse witness for the dividend MIN test",
        COL_DIV_DIVISOR_IS_NEG1: Boolean =>
            "zero-test flag: the divisor (stack read1) equals the active signed div/rem width's -1",
        COL_DIV_DIVISOR_NEG1_INV: Field => "inverse witness for the divisor -1 test",
        COL_DIV_OVERFLOW_COND: Boolean => "product of the dividend-is-MIN and divisor-is--1 flags",
        COL_DIV_OVERFLOW: Boolean => "this row is a signed div op trapping on MIN / -1 overflow",
        // Indirect-call trap scratch; see the trap constraints in
        // ccs/call.rs.
        COL_CI_ENTRY_IS_NULL: Boolean =>
            "zero-test flag: the table entry (table value) read by this row is a null funcref",
        COL_CI_ENTRY_NULL_INV: Field => "inverse witness for the null-funcref zero test",
        COL_CI_TYPE_EQ: Boolean => "zero-test flag: callee type id equals the indirect call's expected type id",
        COL_CI_TYPE_EQ_INV: Field => "inverse witness for the callee-type equality test",
        COL_CALL_INDIRECT_IS_TRAP: Boolean =>
            "this row is an indirect call trapping on OOB index, null entry, or callee type mismatch",
        COL_CALL_INDIRECT_IS_NOT_TRAP: Boolean =>
            "non-trapping indirect-call row: gates callee metadata and entry-pc reads",
        COL_FUNCTION_CALL_TYPE_LOOKUP_GATE: Boolean =>
            "indirect-call row with an in-bounds non-null entry: gates the function-types read",
        // Shared unsigned-comparison scratch for the bounds traps (see
        // `push_unsigned_ge_gadget`). `low` is the range-checked borrow-bit
        // remainder; `ge` is `a >= b` for whichever mutually-exclusive comparison
        // the row's opcode selects.
        COL_CMP_LOW: U32 => "borrow-bit remainder of the active unsigned comparison",
        COL_CMP_GE: Boolean => "result a >= b of the active unsigned comparison",
        COL_CI_OOB: Boolean => "whether an indirect call traps because the table index is >= the table size",
        COL_TABLE_SIZE_READ_ENABLED: Boolean => "table_sizes read gate: table.size or an indirect call",
        COL_MEM_OOB: Boolean =>
            "whether this load/store traps because the access is past the end of linear memory",
        COL_MEM_LOAD_LIVE: Boolean =>
            "load lane gate factor: a load row that is not OOB (de-gates lane reads on an OOB trap)",
        COL_MEM_STORE_LIVE: Boolean =>
            "store lane gate factor: a store row that is not OOB (de-gates lane writes on an OOB trap)",
        COL_GROW_SUCCESS: Boolean => "memory.grow row: the growth fits under max pages (before + delta <= max)",
        COL_HALTED_BEFORE: Boolean => "carried terminal flag before this row",
    ]
}

pub const NAMED_COLUMN_COUNT: usize = WASM_COLUMN_COUNT + HOST_EVENT_COLUMN_COUNT;

pub const COLUMN_SPEC_REGIONS: &[&[WasmColumnSpec]] = &[WASM_COLUMN_SPECS, HOST_EVENT_COLUMN_SPECS];

pub fn column_specs() -> impl Iterator<Item = &'static WasmColumnSpec> {
    COLUMN_SPEC_REGIONS.iter().flat_map(|specs| specs.iter())
}

pub fn column_spec(column: usize) -> Option<&'static WasmColumnSpec> {
    column_specs().find(|spec| spec.contains(column))
}

mod opcode_selectors;
pub use opcode_selectors::{selector_col, SELECTOR_COLS};
