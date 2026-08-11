//! Owns the static WASM row layout.

use super::isa::WasmOpcode;
use crate::column_registry::define_column_region;
pub use crate::column_registry::{ColumnWidth, WasmColumnSpec};

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
    width: pub NAMED_COLUMN_COUNT,
    specs: pub COLUMN_SPECS,
    indices: pub,
    columns: [
        COL_ONE: [Field; 1] => "",
        COL_OUTPUT_ENABLED_BEFORE: [Field; 1] => "carried simple-output flag before this row",
        COL_OUTPUT_ENABLED_AFTER: [Field; 1] => "carried simple-output flag after this row",
        COL_OUTPUT_VALUE_LO_BEFORE: [Field; 1] => "carried output low limb before",
        COL_OUTPUT_VALUE_LO_AFTER: [Field; 1] => "carried output low limb after",
        COL_OUTPUT_VALUE_HI_BEFORE: [Field; 1] => "carried output high limb before",
        COL_OUTPUT_VALUE_HI_AFTER: [Field; 1] => "carried output high limb after",
        COL_OPCODE_CODE: [U32; 1] => "opcode decode selector source",
        COL_PC_BEFORE: [U32; 1] => "transition source pc",
        COL_PC_AFTER: [U32; 1] => "transition destination pc",
        COL_CONTROL_CHOICE: [U32; 1] => "control-edge choice",
        COL_PC_EDGE_KIND: [U32; 1] => "pc edge kind",
        COL_WIDE_VALUES_ENABLED: [Boolean; 1] => "wide-value flag",
        COL_OUTPUT_CAPTURED: [Boolean; 1] => "output-capture gate",
        COL_SP_BEFORE: [U32; 1] => "transition source stack pointer",
        COL_SP_AFTER: [U32; 1] => "transition destination stack pointer",
        COL_STACK_FRAME_BASE_BEFORE: [U32; 1] => "operand-stack frame base before",
        COL_STACK_FRAME_BASE_AFTER: [U32; 1] => "operand-stack frame base after",
        COL_HALTED: [Boolean; 1] => "terminal row flag",
        COL_IS_PROGRAM_ROW: [Boolean; 1] => "decoded wasm program row",
        COL_PC_ROM_ACTIVE: [Boolean; 1] => "static pc-edge ROM gate",
        COL_PC_EDGE_KIND_IS_STATIC: [Boolean; 1] => "static pc-edge flag",
        COL_PC_EDGE_KIND_INV: [Field; 1] => "inverse witness for pc edge-kind zero test",
        COL_PARAM_INIT_ACTIVE_BEFORE: [Boolean; 1] => "parameter-init mode before",
        COL_PARAM_INIT_ACTIVE_AFTER: [Boolean; 1] => "parameter-init mode after",
        COL_TAIL_CALL_PENDING_BEFORE: [Boolean; 1] => "tail-entry pending before",
        COL_TAIL_CALL_PENDING_AFTER: [Boolean; 1] => "tail-entry pending after",
        COL_TAIL_ENTER_ACTIVE: [Boolean; 1] => "tail-enter aux row",
        COL_TAIL_DISCARD_COUNT: [U32; 1] => "tail-enter discard count",
        COL_PADDING_ACTIVE: [Boolean; 1] => "padding row",
        COL_PARAM_INIT_REMAINING_BEFORE: [U32; 1] => "remaining call parameters to initialize before this row",
        COL_PARAM_INIT_REMAINING_AFTER: [U32; 1] => "remaining call parameters to initialize after this row",
        COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO: [Boolean; 1] =>
            "zero-test flag for remaining call parameters after this row",
        COL_PARAM_INIT_REMAINING_AFTER_INV: [Field; 1] =>
            "inverse witness for remaining call parameters after this row",
        COL_HOST_ARGS_ACTIVE_BEFORE: [Boolean; 1] => "host-call argument-pop mode before this row",
        COL_HOST_ARGS_ACTIVE_AFTER: [Boolean; 1] => "host-call argument-pop mode after this row",
        COL_HOST_ARGS_REMAINING_BEFORE: [U32; 1] => "remaining host-call arguments to pop before this row",
        COL_HOST_ARGS_REMAINING_AFTER: [U32; 1] => "remaining host-call arguments to pop after this row",
        COL_HOST_ARGS_REMAINING_AFTER_IS_ZERO: [Boolean; 1] =>
            "zero-test flag for remaining host-call arguments after this row",
        COL_HOST_ARGS_REMAINING_AFTER_INV: [Field; 1] =>
            "inverse witness for remaining host-call arguments after this row",
        COL_HOST_RESULT_PENDING_BEFORE: [Boolean; 1] => "host-call result push still owed before this row",
        COL_HOST_RESULT_PENDING_AFTER: [Boolean; 1] => "host-call result push still owed after this row",
        COL_HOST_CALLEE_FREF_BEFORE: [U32; 1] =>
            "callee function ref of the most recent host call before this row (event attribution carry)",
        COL_HOST_CALLEE_FREF_AFTER: [U32; 1] =>
            "callee function ref of the most recent host call after this row (event attribution carry)",
        COL_TURN_EXPORT_FREF_BEFORE: [U32; 1] => "export function ref owning the current grammar turn before this row",
        COL_TURN_EXPORT_FREF_AFTER: [U32; 1] => "export function ref owning the current grammar turn after this row",
        COL_COMM_CHAIN0_BEFORE: [Field; 1] => "host-event commitment chain limb 0 before this row",
        COL_COMM_CHAIN1_BEFORE: [Field; 1] => "host-event commitment chain limb 1 before this row",
        COL_COMM_CHAIN2_BEFORE: [Field; 1] => "host-event commitment chain limb 2 before this row",
        COL_COMM_CHAIN3_BEFORE: [Field; 1] => "host-event commitment chain limb 3 before this row",
        COL_COMM_CHAIN0_AFTER: [Field; 1] => "host-event commitment chain limb 0 after this row",
        COL_COMM_CHAIN1_AFTER: [Field; 1] => "host-event commitment chain limb 1 after this row",
        COL_COMM_CHAIN2_AFTER: [Field; 1] => "host-event commitment chain limb 2 after this row",
        COL_COMM_CHAIN3_AFTER: [Field; 1] => "host-event commitment chain limb 3 after this row",
        // Host-event absorb machinery: the 8-word block buffer host-call rows
        // stream event words into, the one-hot pair-slot cursor, the
        // pending-permutation flag, and the perm-row group state (round cursor +
        // 12-lane running permutation state). See `ir::WasmEventAbsorbState`.
        COL_EVBUF0_BEFORE: [Field; 1] => "host-event block buffer word 0 before this row",
        COL_EVBUF1_BEFORE: [Field; 1] => "host-event block buffer word 1 before this row",
        COL_EVBUF2_BEFORE: [Field; 1] => "host-event block buffer word 2 before this row",
        COL_EVBUF3_BEFORE: [Field; 1] => "host-event block buffer word 3 before this row",
        COL_EVBUF4_BEFORE: [Field; 1] => "host-event block buffer word 4 before this row",
        COL_EVBUF5_BEFORE: [Field; 1] => "host-event block buffer word 5 before this row",
        COL_EVBUF6_BEFORE: [Field; 1] => "host-event block buffer word 6 before this row",
        COL_EVBUF7_BEFORE: [Field; 1] => "host-event block buffer word 7 before this row",
        COL_EVBUF0_AFTER: [Field; 1] => "host-event block buffer word 0 after this row",
        COL_EVBUF1_AFTER: [Field; 1] => "host-event block buffer word 1 after this row",
        COL_EVBUF2_AFTER: [Field; 1] => "host-event block buffer word 2 after this row",
        COL_EVBUF3_AFTER: [Field; 1] => "host-event block buffer word 3 after this row",
        COL_EVBUF4_AFTER: [Field; 1] => "host-event block buffer word 4 after this row",
        COL_EVBUF5_AFTER: [Field; 1] => "host-event block buffer word 5 after this row",
        COL_EVBUF6_AFTER: [Field; 1] => "host-event block buffer word 6 after this row",
        COL_EVBUF7_AFTER: [Field; 1] => "host-event block buffer word 7 after this row",
        COL_EVBUF_SLOT0_BEFORE: [Boolean; 1] => "one-hot next event-word pair slot 0 before this row",
        COL_EVBUF_SLOT1_BEFORE: [Boolean; 1] => "one-hot next event-word pair slot 1 before this row",
        COL_EVBUF_SLOT2_BEFORE: [Boolean; 1] => "one-hot next event-word pair slot 2 before this row",
        COL_EVBUF_SLOT3_BEFORE: [Boolean; 1] => "one-hot next event-word pair slot 3 before this row",
        COL_EVBUF_SLOT0_AFTER: [Boolean; 1] => "one-hot next event-word pair slot 0 after this row",
        COL_EVBUF_SLOT1_AFTER: [Boolean; 1] => "one-hot next event-word pair slot 1 after this row",
        COL_EVBUF_SLOT2_AFTER: [Boolean; 1] => "one-hot next event-word pair slot 2 after this row",
        COL_EVBUF_SLOT3_AFTER: [Boolean; 1] => "one-hot next event-word pair slot 3 after this row",
        COL_PERM_PENDING_BEFORE: [Boolean; 1] => "a filled host-event block awaits its perm rows before this row",
        COL_PERM_PENDING_AFTER: [Boolean; 1] => "a filled host-event block awaits its perm rows after this row",
        COL_PERM_ROUND_BEFORE: [Field; 1] =>
            "perm-group row position before this row (0 when idle; bounded by the position one-hot)",
        COL_PERM_ROUND_AFTER: [Field; 1] =>
            "perm-group row position after this row (0 when idle; bounded by the position one-hot)",
        COL_PERM_ROUND_BEFORE_IS_ZERO: [Boolean; 1] =>
            "zero-test flag for the perm-group row position before this row",
        COL_PERM_ROUND_BEFORE_INV: [Field; 1] => "inverse witness for the perm-group row position before this row",
        COL_PERM_STATE0_BEFORE: [Field; 1] => "chain permutation lane 0 before this row",
        COL_PERM_STATE1_BEFORE: [Field; 1] => "chain permutation lane 1 before this row",
        COL_PERM_STATE2_BEFORE: [Field; 1] => "chain permutation lane 2 before this row",
        COL_PERM_STATE3_BEFORE: [Field; 1] => "chain permutation lane 3 before this row",
        COL_PERM_STATE4_BEFORE: [Field; 1] => "chain permutation lane 4 before this row",
        COL_PERM_STATE5_BEFORE: [Field; 1] => "chain permutation lane 5 before this row",
        COL_PERM_STATE6_BEFORE: [Field; 1] => "chain permutation lane 6 before this row",
        COL_PERM_STATE7_BEFORE: [Field; 1] => "chain permutation lane 7 before this row",
        COL_PERM_STATE8_BEFORE: [Field; 1] => "chain permutation lane 8 before this row",
        COL_PERM_STATE9_BEFORE: [Field; 1] => "chain permutation lane 9 before this row",
        COL_PERM_STATE10_BEFORE: [Field; 1] => "chain permutation lane 10 before this row",
        COL_PERM_STATE11_BEFORE: [Field; 1] => "chain permutation lane 11 before this row",
        COL_PERM_STATE0_AFTER: [Field; 1] => "chain permutation lane 0 after this row",
        COL_PERM_STATE1_AFTER: [Field; 1] => "chain permutation lane 1 after this row",
        COL_PERM_STATE2_AFTER: [Field; 1] => "chain permutation lane 2 after this row",
        COL_PERM_STATE3_AFTER: [Field; 1] => "chain permutation lane 3 after this row",
        COL_PERM_STATE4_AFTER: [Field; 1] => "chain permutation lane 4 after this row",
        COL_PERM_STATE5_AFTER: [Field; 1] => "chain permutation lane 5 after this row",
        COL_PERM_STATE6_AFTER: [Field; 1] => "chain permutation lane 6 after this row",
        COL_PERM_STATE7_AFTER: [Field; 1] => "chain permutation lane 7 after this row",
        COL_PERM_STATE8_AFTER: [Field; 1] => "chain permutation lane 8 after this row",
        COL_PERM_STATE9_AFTER: [Field; 1] => "chain permutation lane 9 after this row",
        COL_PERM_STATE10_AFTER: [Field; 1] => "chain permutation lane 10 after this row",
        COL_PERM_STATE11_AFTER: [Field; 1] => "chain permutation lane 11 after this row",
        COL_GRAMMAR_MODE_BEFORE: [Boolean; 1] =>
            "per-program constant: chain absorbs embedder grammar events (1) or raw host-call records (0)",
        COL_GRAMMAR_MODE_AFTER: [Boolean; 1] =>
            "per-program constant: chain absorbs embedder grammar events (1) or raw host-call records (0)",
        COL_GATHER_ACTIVE: [Boolean; 1] => "grammar-mode row staging one expanded event block into the absorb buffer",
        COL_RAW_HOST_CALL: [Boolean; 1] =>
            "host-call program row with the raw absorb machinery active: host_call_gate · (1 - grammar_mode)",
        COL_RAW_ARGS_ACTIVE: [Boolean; 1] =>
            "host-arg row with the raw absorb machinery active: host_args_active · (1 - grammar_mode)",
        COL_RAW_RESULT_ACTIVE: [Boolean; 1] =>
            "host-result row with the raw absorb machinery active: host_result_active · (1 - grammar_mode)",
        // Grammar-mode gather machinery: carried schedule/cursor/oracle state
        // plus the per-row grammar-ROM interface columns (bound on gather rows
        // by the `grammar_slot_*` families, on call/result rows by the
        // `grammar_event_counts_*` families). See `ir::WasmGrammarState` and
        // `docs/host-event-grammar-tables.md`.
        COL_GRAMMAR_EVREM_BEFORE: [Field; 1] => "grammar events still owed in the current phase, before this row",
        COL_GRAMMAR_EVREM_AFTER: [Field; 1] => "grammar events still owed in the current phase, after this row",
        COL_GRAMMAR_EVREM_BEFORE_IS_ZERO: [Boolean; 1] => "zero-test flag for the owed grammar events before this row",
        COL_GRAMMAR_EVREM_BEFORE_INV: [Field; 1] => "inverse witness for the owed grammar events before this row",
        COL_GRAMMAR_EVIDX_BEFORE: [Field; 1] => "current grammar event index within the template, before this row",
        COL_GRAMMAR_EVIDX_AFTER: [Field; 1] => "current grammar event index within the template, after this row",
        COL_GRAMMAR_ARGS_BASE_BEFORE: [Field; 1] =>
            "stack slot index of the current call's first argument, before this row",
        COL_GRAMMAR_ARGS_BASE_AFTER: [Field; 1] =>
            "stack slot index of the current call's first argument, after this row",
        COL_GRAMMAR_SLOT_CURSOR_BEFORE: [Field; 1] => "next block word a gather row stages (0..=7), before this row",
        COL_GRAMMAR_SLOT_CURSOR_AFTER: [Field; 1] => "next block word a gather row stages (0..=7), after this row",
        COL_GRAMMAR_SLOT_KIND: [Field; 1] =>
            "grammar-ROM slot source kind (0 const, 1 arg, 2 result, 3 claim, 4 input-local, 5 output, 6 memory-read, 7 memory-write)",
        COL_GRAMMAR_SLOT_ARG: [Field; 1] => "grammar-ROM slot arg/oracle index",
        COL_GRAMMAR_SLOT_VARIANT: [Field; 1] =>
            "encoded grammar-ROM slot variant: value kinds use 0 lo / 1 hi; memory kinds use bit 0 for local base, bit 1 for byte width, and bit 2 for half width",
        COL_GRAMMAR_SLOT_CONST_LO: [Field; 1] => "grammar-ROM slot constant, low 32 bits",
        COL_GRAMMAR_SLOT_CONST_HI: [Field; 1] => "grammar-ROM slot constant, high 32 bits",
        COL_GRAMMAR_PRE_COUNT: [Field; 1] =>
            "grammar-ROM event count for the called import / entered export (biased +1)",
        COL_GRAMMAR_POST_COUNT: [Field; 1] => "grammar-ROM exit-event count for the halting export",
        COL_GRAMMAR_HOST_CALL: [Field; 1] => "host-call program row in grammar mode: host_call_gate · grammar_mode",
        COL_GATHER_LOCAL_WRITE: [Boolean; 1] =>
            "gather row writing a claim-input word into an entry-frame locals lane (slot kind 4); gates the hi-lane write (zero on lo rows)",
        COL_GATHER_LOCAL_WRITE_LO: [Boolean; 1] =>
            "input-local gather row targeting the lo lane: gather_local_write · (1 - slot_variant)",
        COL_GRAMMAR_EXIT_LATCH: [Boolean; 1] =>
            "clean export-halt row in grammar mode: loads the export's exit-event schedule",
        COL_TURN_BOUNDARY: [Boolean; 1] =>
            "multi-turn re-entry row: re-arms the output, loads the next export's entry schedule, jumps to its entry pc",
        COL_PC_FREF_ACTIVE: [Boolean; 1] =>
            "pc -> function-ref ROM gate: every row except gather rows (post-halt exit gathers sit past the last pc)",
        COL_HOST_RESULT_ACTIVE: [Boolean; 1] => "this row pushes the pending host-call result",
        COL_CI_HOST_CALL: [Boolean; 1] => "non-trapping call_indirect row targeting a host import",
        COL_GUEST_ENTRY_ACTIVE: [Boolean; 1] => "guest-entry row flag: this row enters a traced guest callee",
        COL_CALL_STACK_PUSH_PRESENT: [Boolean; 1] => "flag indicating that this row saves a caller return context",
        COL_CALL_STACK_POP_PRESENT: [Boolean; 1] =>
            "flag indicating that this row restores a saved caller return context",
        COL_CALL_STACK_RETURN_PC_VALUE: [U32; 1] =>
            "call-stack return-pc cell value: written on guest-call push, read back on pop into pc_after",
        COL_CALL_STACK_CALLER_FBP_VALUE: [U32; 1] =>
            "call-stack caller-fbp cell value: written on guest-call push, read back on pop into locals_fbp_after",
        COL_CALL_STACK_CALLER_SP_BASE_VALUE: [U32; 1] =>
            "call-stack caller operand-stack base: written on push and restored on pop",
        COL_CALL_STACK_DEPTH_BEFORE: [U32; 1] => "call return-context stack depth before this row",
        COL_CALL_STACK_DEPTH_AFTER: [U32; 1] => "call return-context stack depth after this row",
        COL_CALL_STACK_ADDR: [U32; 1] => "call return-context stack address read or written this row",
        COL_PC_ROM_CALL_RETURN_CHOICE: [U32; 1] =>
            "pc-rom control-choice coordinate for the call-site continuation-pc read (guest pushes and indirect host fall-through)",
        COL_CURRENT_FUNCTION_REF: [U32; 1] => "normalized function reference for the currently executing frame",
        COL_CURRENT_FUNCTION_NUM_LOCALS: [U32; 1] => "number of locals in the current function frame",
        COL_STACK_READS: [U32; 1] => "stack delta source",
        COL_STACK_WRITES: [U32; 1] => "stack delta destination",
        COL_STACK_READ0_ACTIVE: [Boolean; 1] => "stack lane 0 read activity flag",
        COL_STACK_READ1_ACTIVE: [Boolean; 1] => "stack lane 1 read activity flag",
        COL_STACK_READ2_ACTIVE: [Boolean; 1] => "stack lane 2 read activity flag",
        COL_STACK_WRITE0_ACTIVE: [Boolean; 1] => "stack lane 0 write activity flag",
        COL_STACK_WRITE0_HI_ACTIVE: [Boolean; 1] =>
            "stack write hi-word port gate: write0_active, plus result-hi gather rows writing only the hi lane",
        COL_OP_TABLE_ENABLED: [Boolean; 1] => "lookup gate",
        COL_SEL_NOP: [Boolean; 1] => "",
        COL_SEL_I32_CONST: [Boolean; 1] => "",
        COL_SEL_I64_CONST: [Boolean; 1] => "",
        COL_SEL_REF_FUNC: [Boolean; 1] => "",
        COL_SEL_I32_ADD: [Boolean; 1] => "",
        COL_SEL_I64_ADD: [Boolean; 1] => "",
        COL_SEL_I32_SUB: [Boolean; 1] => "",
        COL_SEL_I64_SUB: [Boolean; 1] => "",
        COL_SEL_I32_LOAD: [Boolean; 1] => "",
        COL_SEL_I32_LOAD8_S: [Boolean; 1] => "",
        COL_SEL_I32_LOAD8_U: [Boolean; 1] => "",
        COL_SEL_I32_LOAD16_S: [Boolean; 1] => "",
        COL_SEL_I32_LOAD16_U: [Boolean; 1] => "",
        COL_SEL_I64_LOAD: [Boolean; 1] => "",
        COL_SEL_I32_STORE: [Boolean; 1] => "",
        COL_SEL_I32_STORE8: [Boolean; 1] => "",
        COL_SEL_I32_STORE16: [Boolean; 1] => "",
        COL_SEL_I64_STORE: [Boolean; 1] => "",
        COL_SEL_MEMORY_SIZE: [Boolean; 1] => "",
        COL_SEL_MEMORY_GROW: [Boolean; 1] => "",
        COL_SEL_TABLE_SIZE: [Boolean; 1] => "",
        COL_SEL_TABLE_GET: [Boolean; 1] => "",
        COL_SEL_TABLE_SET: [Boolean; 1] => "",
        COL_SEL_DROP: [Boolean; 1] => "",
        COL_SEL_BR: [Boolean; 1] => "",
        COL_SEL_BLOCK: [Boolean; 1] => "",
        COL_SEL_LOOP: [Boolean; 1] => "",
        COL_SEL_IF: [Boolean; 1] => "",
        COL_SEL_ELSE: [Boolean; 1] => "",
        COL_SEL_END: [Boolean; 1] => "",
        COL_SEL_UNREACHABLE: [Boolean; 1] => "",
        COL_SEL_I32_CLZ: [Boolean; 1] => "",
        COL_SEL_I32_CTZ: [Boolean; 1] => "",
        COL_SEL_I32_POPCNT: [Boolean; 1] => "",
        COL_SEL_I32_EQZ: [Boolean; 1] => "",
        COL_SEL_I64_EQZ: [Boolean; 1] => "",
        COL_SEL_I32_EQ: [Boolean; 1] => "",
        COL_SEL_I32_NE: [Boolean; 1] => "",
        COL_SEL_I32_LTS: [Boolean; 1] => "",
        COL_SEL_I32_LTU: [Boolean; 1] => "",
        COL_SEL_I32_GTS: [Boolean; 1] => "",
        COL_SEL_I32_GTU: [Boolean; 1] => "",
        COL_SEL_I32_LES: [Boolean; 1] => "",
        COL_SEL_I32_LEU: [Boolean; 1] => "",
        COL_SEL_I32_GES: [Boolean; 1] => "",
        COL_SEL_I32_GEU: [Boolean; 1] => "",
        COL_SEL_I32_AND: [Boolean; 1] => "",
        COL_SEL_I32_OR: [Boolean; 1] => "",
        COL_SEL_I32_XOR: [Boolean; 1] => "",
        COL_SEL_I32_MUL: [Boolean; 1] => "",
        COL_SEL_I64_AND: [Boolean; 1] => "",
        COL_SEL_I64_OR: [Boolean; 1] => "",
        COL_SEL_I64_XOR: [Boolean; 1] => "",
        COL_SEL_I64_MUL: [Boolean; 1] => "",
        COL_SEL_I32_SHL: [Boolean; 1] => "",
        COL_SEL_I32_SHR_U: [Boolean; 1] => "",
        COL_SEL_I32_SHR_S: [Boolean; 1] => "",
        COL_SEL_I32_ROTL: [Boolean; 1] => "",
        COL_SEL_I32_ROTR: [Boolean; 1] => "",
        COL_SEL_I32_DIV_U: [Boolean; 1] => "",
        COL_SEL_I32_DIV_S: [Boolean; 1] => "",
        COL_SEL_I32_REM_U: [Boolean; 1] => "",
        COL_SEL_I32_REM_S: [Boolean; 1] => "",
        COL_SEL_SELECT: [Boolean; 1] => "",
        COL_SEL_BR_IF_EQZ: [Boolean; 1] => "",
        COL_SEL_BR_TABLE: [Boolean; 1] => "",
        COL_SEL_CALL: [Boolean; 1] => "",
        COL_SEL_CALL_INDIRECT: [Boolean; 1] => "",
        COL_SEL_RETURN: [Boolean; 1] => "",
        COL_SEL_LOCAL_GET: [Boolean; 1] => "",
        COL_SEL_LOCAL_SET: [Boolean; 1] => "",
        COL_SEL_LOCAL_TEE: [Boolean; 1] => "",
        COL_SEL_GLOBAL_GET: [Boolean; 1] => "",
        COL_SEL_GLOBAL_SET: [Boolean; 1] => "",
        COL_SEL_I64_EQ: [Boolean; 1] => "",
        COL_SEL_I64_NE: [Boolean; 1] => "",
        COL_SEL_I64_STORE8: [Boolean; 1] => "",
        COL_SEL_I64_STORE16: [Boolean; 1] => "",
        COL_SEL_I64_STORE32: [Boolean; 1] => "",
        COL_SEL_I64_LOAD8_U: [Boolean; 1] => "",
        COL_SEL_I64_LOAD16_U: [Boolean; 1] => "",
        COL_SEL_I64_LOAD32_U: [Boolean; 1] => "",
        COL_SEL_I64_LOAD8_S: [Boolean; 1] => "",
        COL_SEL_I64_LOAD16_S: [Boolean; 1] => "",
        COL_SEL_I64_LOAD32_S: [Boolean; 1] => "",
        COL_LOCAL_WRITE_ENABLED: [Boolean; 1] => "locals memory write gate for local.set/local.tee",
        COL_PROGRAM_LOCAL_INDEX_ACTIVE: [Boolean; 1] => "PC-indexed local immediate ROM gate",
        COL_PROGRAM_GLOBAL_INDEX_ACTIVE: [Boolean; 1] => "PC-indexed global immediate ROM gate",
        COL_PROGRAM_TABLE_ID_ACTIVE: [Boolean; 1] => "PC-indexed table-id immediate ROM gate",
        COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE: [Boolean; 1] => "PC-indexed call-indirect immediate ROM gate",
        COL_TABLE_READ_ENABLED: [Boolean; 1] => "table memory read gate for table.get and indirect calls",
        COL_LOCALS_FBP_BEFORE: [U32; 1] => "locals memory frame base before this row",
        COL_LOCALS_FBP_AFTER: [U32; 1] => "locals memory frame base after this row",
        COL_LOCAL_INDEX: [U32; 1] => "locals memory offset",
        COL_LOCAL_VALUE: [U32; 1] => "locals memory value",
        COL_LOCAL_VALUE_HI: [U32; 1] => "locals memory high limb for future i64 support",
        COL_GLOBAL_INDEX: [U32; 1] => "globals memory index",
        COL_GLOBAL_VALUE: [U32; 1] => "globals memory value",
        COL_GLOBAL_VALUE_HI: [U32; 1] => "globals memory high limb for future i64 support",
        COL_TABLE_ID: [U32; 1] => "table state namespace selector",
        COL_TABLE_INDEX: [U32; 1] => "table element index",
        COL_TABLE_VALUE: [U32; 1] => "normalized table element value observed by this step",
        COL_TABLE_SIZE: [U32; 1] => "size of the referenced table observed by this step",
        COL_FUNCTION_REF: [U32; 1] => "normalized function reference selected by call-like opcodes",
        COL_CALL_PARAM_COUNT: [Byte; 1] => "parameter count for the selected call target",
        COL_CALL_RESULT_COUNT: [Byte; 1] => "result count for the selected call target",
        COL_TARGET_FUNCTION_IS_GUEST: [Boolean; 1] => "true when the selected call target is a guest-defined function",
        COL_CALL_TARGET_METADATA: [Field; 1] =>
            "packed call-target arity and guest flag; range follows from unpacking",
        COL_CALL_INDIRECT_TYPE_INDEX: [U32; 1] =>
            "raw module type-section index from the call_indirect instruction immediate",
        COL_FUNCTION_TYPE_ID: [U32; 1] => "normalized deduplicated type id for the observed function reference",
        COL_EXPECTED_TYPE_ID: [U32; 1] => "normalized deduplicated type id expected by the current opcode",
        COL_MEMORY_PAGES_BEFORE: [U32; 1] => "linear memory page count before this step",
        COL_MEMORY_PAGES_AFTER: [U32; 1] => "linear memory page count after this step",
        COL_MAX_MEMORY_PAGES_BEFORE: [U32; 1] =>
            "verifier-authoritative max linear-memory page count before this step (carried constant)",
        COL_MAX_MEMORY_PAGES_AFTER: [U32; 1] =>
            "verifier-authoritative max linear-memory page count after this step (carried constant)",
        COL_STACK_READ0_ADDR_LO: [U32; 1] => "operand-stack read lane 0 low-limb physical address",
        COL_STACK_READ0_ADDR_HI: [U32; 1] => "operand-stack read lane 0 high-limb physical address",
        COL_STACK_READ0_VALUE_LO: [U32; 1] => "operand-stack read lane 0 value",
        COL_STACK_READ0_VALUE_HI: [U32; 1] => "operand-stack read lane 0 high limb for future i64 support",
        COL_STACK_READ1_ADDR_LO: [U32; 1] => "operand-stack read lane 1 low-limb physical address",
        COL_STACK_READ1_ADDR_HI: [U32; 1] => "operand-stack read lane 1 high-limb physical address",
        COL_STACK_READ1_VALUE_LO: [U32; 1] => "operand-stack read lane 1 value",
        COL_STACK_READ1_VALUE_HI: [U32; 1] => "operand-stack read lane 1 high limb for future i64 support",
        COL_STACK_READ2_ADDR_LO: [U32; 1] => "operand-stack read lane 2 low-limb physical address",
        COL_STACK_READ2_ADDR_HI: [U32; 1] => "operand-stack read lane 2 high-limb physical address",
        COL_STACK_READ2_VALUE_LO: [U32; 1] => "operand-stack read lane 2 value",
        COL_STACK_READ2_VALUE_HI: [U32; 1] => "operand-stack read lane 2 high limb for future i64 support",
        COL_STACK_WRITE0_ADDR_LO: [U32; 1] => "operand-stack write lane 0 low-limb physical address",
        COL_STACK_WRITE0_ADDR_HI: [U32; 1] => "operand-stack write lane 0 high-limb physical address",
        COL_STACK_WRITE0_VALUE_LO: [U32; 1] => "operand-stack write lane 0 value",
        COL_STACK_WRITE0_VALUE_HI: [U32; 1] => "operand-stack write lane 0 high limb for future i64 support",
        COL_WIDE_AUX0: [Boolean; 1] => "",
        COL_WIDE_AUX1: [Boolean; 1] => "",
        COL_LINEAR_MEM_IMM_OFFSET: [U32; 1] => "linear-memory immediate offset in bytes",
        COL_LINEAR_MEM_BYTE_OFFSET: [U32; 1] => "linear-memory byte offset within the first word lane",
        COL_LINEAR_MEM_USE_LANE1: [Boolean; 1] => "linear-memory second-lane flag for unaligned accesses",
        COL_LINEAR_MEM_USE_LANE2: [Boolean; 1] => "linear-memory third-lane flag for wide unaligned accesses",
        COL_LINEAR_MEM_USE_LANE0: [Boolean; 1] => "linear-memory first-lane activity gate",
        COL_LINEAR_MEM_LANE0_LOAD_ACTIVE: [Boolean; 1] =>
            "linear-memory lane0 load gate (use_lane0 AND opcode is a load)",
        COL_LINEAR_MEM_LANE1_LOAD_ACTIVE: [Boolean; 1] =>
            "linear-memory lane1 load gate (use_lane1 AND opcode is a load)",
        COL_LINEAR_MEM_LANE2_LOAD_ACTIVE: [Boolean; 1] =>
            "linear-memory lane2 load gate (use_lane2 AND opcode is a load)",
        COL_LINEAR_MEM_LANE0_STORE_ACTIVE: [Boolean; 1] =>
            "linear-memory lane0 store gate (use_lane0 AND opcode is a store)",
        COL_LINEAR_MEM_LANE1_STORE_ACTIVE: [Boolean; 1] =>
            "linear-memory lane1 store gate (use_lane1 AND opcode is a store)",
        COL_LINEAR_MEM_LANE2_STORE_ACTIVE: [Boolean; 1] =>
            "linear-memory lane2 store gate (use_lane2 AND opcode is a store)",
        COL_LINEAR_MEM_LANE0_ADDR: [U32; 1] => "linear-memory first word-lane address",
        COL_LINEAR_MEM_LANE0_VALUE: [Field; 1] => "linear-memory first word-lane accessed value",
        COL_LINEAR_MEM_LANE1_ADDR: [U32; 1] => "linear-memory second word-lane address",
        COL_LINEAR_MEM_LANE1_VALUE: [Field; 1] => "linear-memory second word-lane accessed value",
        COL_LINEAR_MEM_LANE2_ADDR: [U32; 1] => "linear-memory third word-lane address",
        COL_LINEAR_MEM_LANE2_VALUE: [Field; 1] => "linear-memory third word-lane accessed value",
        COL_OP_TABLE_ID: [U32; 1] => "lookup table row selector",
        COL_OP_TABLE_VALUE: [Field; 1] => "lookup payload witness",
        // `COL_SELECT_COND_IS_ZERO` is forced to {0, 1} by the zero-test rows
        // emitted by `push_select_constraints`. Declared `Boolean` so the
        // spec reflects its actual range; whichever path eventually enforces
        // `ColumnWidth::Boolean` will overlap with the gadget's constraint and
        // one of the two becomes redundant; an optimization to revisit later.
        // We deliberately do not introduce a dedicated `ImpliedBoolean` width
        // for this case because that would be premature complexity.
        COL_SELECT_COND_IS_ZERO: [Boolean; 1] => "scratch column for push_select_constraints for select opcode",
        COL_SELECT_SCRATCH_INV: [Field; 1] => "scratch column for push_select_constraints for select opcode",
        COL_SELECT_OUT_DELTA_LO: [Field; 1] => "scratch column for push_select_constraints low-limb mux",
        COL_SELECT_OUT_DELTA_HI: [Field; 1] => "scratch column for push_select_constraints high-limb mux",
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
        COL_CMP_LO_DIFF: [Field; 1] => "comparator zero-test input (lo limb / full i32 input)",
        COL_CMP_LO_INV: [Field; 1] => "comparator zero-test inverse witness",
        COL_CMP_LO_IS_ZERO: [Boolean; 1] => "comparator zero-test result",
        COL_CMP_HI_DIFF: [Field; 1] => "comparator hi-limb zero-test input (i64.eqz / i64.eq / i64.ne)",
        COL_CMP_HI_INV: [Field; 1] => "comparator hi-limb zero-test inverse witness",
        COL_CMP_HI_IS_ZERO: [Boolean; 1] => "comparator hi-limb zero-test result",
        COL_CMP_AND: [Boolean; 1] => "AND of COL_CMP_LO_IS_ZERO and COL_CMP_HI_IS_ZERO; i64.eqz / i64.eq result",
        COL_LINEAR_MEM_OFFSET_IS_0: [Boolean; 1] => "linear-memory offset case selector for byte offset 0",
        COL_LINEAR_MEM_OFFSET_IS_1: [Boolean; 1] => "linear-memory offset case selector for byte offset 1",
        COL_LINEAR_MEM_OFFSET_IS_2: [Boolean; 1] => "linear-memory offset case selector for byte offset 2",
        COL_LINEAR_MEM_OFFSET_IS_3: [Boolean; 1] => "linear-memory offset case selector for byte offset 3",
        COL_LINEAR_MEM_IS_BYTE_WIDTH: [Boolean; 1] => "linear-memory selector for 8-bit byte-width accesses",
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_0: [Boolean; 1] =>
            "linear-memory byte-width offset case selector for byte offset 0",
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1: [Boolean; 1] =>
            "linear-memory byte-width offset case selector for byte offset 1",
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2: [Boolean; 1] =>
            "linear-memory byte-width offset case selector for byte offset 2",
        COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_3: [Boolean; 1] =>
            "linear-memory byte-width offset case selector for byte offset 3",
        COL_LINEAR_MEM_IS_HALF_WIDTH: [Boolean; 1] => "linear-memory selector for 16-bit half-width accesses",
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_0: [Boolean; 1] =>
            "linear-memory half-width offset case selector for byte offset 0",
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_1: [Boolean; 1] =>
            "linear-memory half-width offset case selector for byte offset 1",
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_2: [Boolean; 1] =>
            "linear-memory half-width offset case selector for byte offset 2",
        COL_LINEAR_MEM_HALF_WIDTH_OFFSET_IS_3: [Boolean; 1] =>
            "linear-memory half-width offset case selector for byte offset 3",
        COL_LINEAR_MEM_IS_FULL_WIDTH: [Boolean; 1] => "linear-memory selector for 32-bit full-width accesses",
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_0: [Boolean; 1] =>
            "linear-memory full-width offset case selector for byte offset 0",
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_1: [Boolean; 1] =>
            "linear-memory full-width offset case selector for byte offset 1",
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_2: [Boolean; 1] =>
            "linear-memory full-width offset case selector for byte offset 2",
        COL_LINEAR_MEM_FULL_WIDTH_OFFSET_IS_3: [Boolean; 1] =>
            "linear-memory full-width offset case selector for byte offset 3",
        COL_LINEAR_MEM_IS_DOUBLE_WIDTH: [Boolean; 1] => "linear-memory selector for 64-bit double-width accesses",
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_0: [Boolean; 1] =>
            "linear-memory double-width offset case selector for byte offset 0",
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_1: [Boolean; 1] =>
            "linear-memory double-width offset case selector for byte offset 1",
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_2: [Boolean; 1] =>
            "linear-memory double-width offset case selector for byte offset 2",
        COL_LINEAR_MEM_DOUBLE_WIDTH_OFFSET_IS_3: [Boolean; 1] =>
            "linear-memory double-width offset case selector for byte offset 3",
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_0: [Boolean; 1] => "i64.load offset case selector for byte offset 0",
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_1: [Boolean; 1] => "i64.load offset case selector for byte offset 1",
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_2: [Boolean; 1] => "i64.load offset case selector for byte offset 2",
        COL_LINEAR_MEM_I64_LOAD_OFFSET_IS_3: [Boolean; 1] => "i64.load offset case selector for byte offset 3",
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_0: [Boolean; 1] => "i64.store offset case selector for byte offset 0",
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_1: [Boolean; 1] => "i64.store offset case selector for byte offset 1",
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_2: [Boolean; 1] => "i64.store offset case selector for byte offset 2",
        COL_LINEAR_MEM_I64_STORE_OFFSET_IS_3: [Boolean; 1] => "i64.store offset case selector for byte offset 3",
        COL_LINEAR_MEM_LANE0_BYTE0: [Byte; 1] => "linear-memory first word lane byte 0",
        COL_LINEAR_MEM_LANE0_BYTE1: [Byte; 1] => "linear-memory first word lane byte 1",
        COL_LINEAR_MEM_LANE0_BYTE2: [Byte; 1] => "linear-memory first word lane byte 2",
        COL_LINEAR_MEM_LANE0_BYTE3: [Byte; 1] => "linear-memory first word lane byte 3",
        COL_LINEAR_MEM_LANE1_BYTE0: [Byte; 1] => "linear-memory second word lane byte 0",
        COL_LINEAR_MEM_LANE1_BYTE1: [Byte; 1] => "linear-memory second word lane byte 1",
        COL_LINEAR_MEM_LANE1_BYTE2: [Byte; 1] => "linear-memory second word lane byte 2",
        COL_LINEAR_MEM_LANE1_BYTE3: [Byte; 1] => "linear-memory second word lane byte 3",
        COL_LINEAR_MEM_LANE2_BYTE0: [Byte; 1] => "linear-memory third word lane byte 0",
        COL_LINEAR_MEM_LANE2_BYTE1: [Byte; 1] => "linear-memory third word lane byte 1",
        COL_LINEAR_MEM_LANE2_BYTE2: [Byte; 1] => "linear-memory third word lane byte 2",
        COL_LINEAR_MEM_LANE2_BYTE3: [Byte; 1] => "linear-memory third word lane byte 3",
        COL_LINEAR_MEM_LANE0_VALUE_BEFORE: [Field; 1] => "linear-memory first word-lane value before this row",
        COL_LINEAR_MEM_LANE1_VALUE_BEFORE: [Field; 1] => "linear-memory second word-lane value before this row",
        COL_LINEAR_MEM_LANE2_VALUE_BEFORE: [Field; 1] => "linear-memory third word-lane value before this row",
        COL_LINEAR_MEM_LANE0_BYTE0_BEFORE: [Byte; 1] => "linear-memory first word lane byte 0 before this row",
        COL_LINEAR_MEM_LANE0_BYTE1_BEFORE: [Byte; 1] => "linear-memory first word lane byte 1 before this row",
        COL_LINEAR_MEM_LANE0_BYTE2_BEFORE: [Byte; 1] => "linear-memory first word lane byte 2 before this row",
        COL_LINEAR_MEM_LANE0_BYTE3_BEFORE: [Byte; 1] => "linear-memory first word lane byte 3 before this row",
        COL_LINEAR_MEM_LANE1_BYTE0_BEFORE: [Byte; 1] => "linear-memory second word lane byte 0 before this row",
        COL_LINEAR_MEM_LANE1_BYTE1_BEFORE: [Byte; 1] => "linear-memory second word lane byte 1 before this row",
        COL_LINEAR_MEM_LANE1_BYTE2_BEFORE: [Byte; 1] => "linear-memory second word lane byte 2 before this row",
        COL_LINEAR_MEM_LANE1_BYTE3_BEFORE: [Byte; 1] => "linear-memory second word lane byte 3 before this row",
        COL_LINEAR_MEM_LANE2_BYTE0_BEFORE: [Byte; 1] => "linear-memory third word lane byte 0 before this row",
        COL_LINEAR_MEM_LANE2_BYTE1_BEFORE: [Byte; 1] => "linear-memory third word lane byte 1 before this row",
        COL_LINEAR_MEM_LANE2_BYTE2_BEFORE: [Byte; 1] => "linear-memory third word lane byte 2 before this row",
        COL_LINEAR_MEM_LANE2_BYTE3_BEFORE: [Byte; 1] => "linear-memory third word lane byte 3 before this row",
        COL_LINEAR_MEM_ACCESS_BYTE0: [Byte; 1] => "linear-memory access value lo byte 0",
        COL_LINEAR_MEM_ACCESS_BYTE1: [Byte; 1] => "linear-memory access value lo byte 1",
        COL_LINEAR_MEM_ACCESS_BYTE2: [Byte; 1] => "linear-memory access value lo byte 2",
        COL_LINEAR_MEM_ACCESS_BYTE3: [Byte; 1] => "linear-memory access value lo byte 3",
        COL_LINEAR_MEM_ACCESS_BYTE4: [Byte; 1] => "linear-memory access value hi byte 0 (i64 only)",
        COL_LINEAR_MEM_ACCESS_BYTE5: [Byte; 1] => "linear-memory access value hi byte 1 (i64 only)",
        COL_LINEAR_MEM_ACCESS_BYTE6: [Byte; 1] => "linear-memory access value hi byte 2 (i64 only)",
        COL_LINEAR_MEM_ACCESS_BYTE7: [Byte; 1] => "linear-memory access value hi byte 3 (i64 only)",
        // Genuine range is 7 bits, [0, 128). Annotated `Byte` as a conservative
        // (over-)approximation so it gets the same enforcement as other byte
        // columns. Tighten to a 7-bit declaration when a `Bits(N)` variant lands.
        COL_SIGN_EXT_LOW7: [Byte; 1] => "sign-extension scratch lower 7 bits of the sign source byte",
        COL_SIGN_EXT_BIT: [Boolean; 1] => "sign-extension scratch sign bit",
        COL_SEL_I32_WRAP_I64: [Boolean; 1] => "selector for i32.wrap_i64",
        COL_SEL_I64_EXTEND_I32_U: [Boolean; 1] => "selector for i64.extend_i32_u",
        COL_SEL_I64_EXTEND_I32_S: [Boolean; 1] => "selector for i64.extend_i32_s",
        COL_SEL_I32_EXTEND8_S: [Boolean; 1] => "selector for i32.extend8_s",
        COL_SEL_I32_EXTEND16_S: [Boolean; 1] => "selector for i32.extend16_s",
        COL_SEL_I64_EXTEND8_S: [Boolean; 1] => "selector for i64.extend8_s",
        COL_SEL_I64_EXTEND16_S: [Boolean; 1] => "selector for i64.extend16_s",
        COL_SEL_I64_EXTEND32_S: [Boolean; 1] => "selector for i64.extend32_s",
        COL_SEL_I64_LTS: [Boolean; 1] => "selector for i64.lt_s",
        COL_SEL_I64_LTU: [Boolean; 1] => "selector for i64.lt_u",
        COL_SEL_I64_GTS: [Boolean; 1] => "selector for i64.gt_s",
        COL_SEL_I64_GTU: [Boolean; 1] => "selector for i64.gt_u",
        COL_SEL_I64_LES: [Boolean; 1] => "selector for i64.le_s",
        COL_SEL_I64_LEU: [Boolean; 1] => "selector for i64.le_u",
        COL_SEL_I64_GES: [Boolean; 1] => "selector for i64.ge_s",
        COL_SEL_I64_GEU: [Boolean; 1] => "selector for i64.ge_u",
        COL_SEL_I64_SHL: [Boolean; 1] => "selector for i64.shl",
        COL_SEL_I64_SHR_S: [Boolean; 1] => "selector for i64.shr_s",
        COL_SEL_I64_SHR_U: [Boolean; 1] => "selector for i64.shr_u",
        COL_SEL_I64_ROTL: [Boolean; 1] => "selector for i64.rotl",
        COL_SEL_I64_ROTR: [Boolean; 1] => "selector for i64.rotr",
        COL_SEL_I64_DIV_S: [Boolean; 1] => "selector for i64.div_s",
        COL_SEL_I64_DIV_U: [Boolean; 1] => "selector for i64.div_u",
        COL_SEL_I64_REM_S: [Boolean; 1] => "selector for i64.rem_s",
        COL_SEL_I64_REM_U: [Boolean; 1] => "selector for i64.rem_u",
        COL_SEL_I64_CLZ: [Boolean; 1] => "selector for i64.clz",
        COL_SEL_I64_CTZ: [Boolean; 1] => "selector for i64.ctz",
        COL_SEL_I64_POPCNT: [Boolean; 1] => "selector for i64.popcnt",
        COL_SEL_RETURN_CALL: [Boolean; 1] => "selector for return_call",
        COL_SEL_RETURN_CALL_INDIRECT: [Boolean; 1] => "selector for return_call_indirect",
        COL_TRAPPED_BEFORE: [Boolean; 1] => "carried trapped-execution flag before this row",
        COL_TRAPPED_AFTER: [Boolean; 1] => "carried trapped-execution flag after this row",
        // Div/rem trap scratch; see the `trap transition` constraints in ccs.rs.
        COL_DIV_DIVISOR_IS_ZERO: [Boolean; 1] => "zero-test flag for the divisor (stack read1) on this row",
        COL_DIV_DIVISOR_INV: [Field; 1] => "inverse witness for the divisor zero test",
        COL_DIV_TRAP: [Boolean; 1] => "this row is a div/rem op trapping on a zero divisor or signed overflow",
        COL_DIV_DIVIDEND_IS_MIN: [Boolean; 1] =>
            "zero-test flag: the dividend (stack read0) equals the active signed div/rem width's MIN",
        COL_DIV_DIVIDEND_MIN_INV: [Field; 1] => "inverse witness for the dividend MIN test",
        COL_DIV_DIVISOR_IS_NEG1: [Boolean; 1] =>
            "zero-test flag: the divisor (stack read1) equals the active signed div/rem width's -1",
        COL_DIV_DIVISOR_NEG1_INV: [Field; 1] => "inverse witness for the divisor -1 test",
        COL_DIV_OVERFLOW_COND: [Boolean; 1] => "product of the dividend-is-MIN and divisor-is--1 flags",
        COL_DIV_OVERFLOW: [Boolean; 1] => "this row is a signed div op trapping on MIN / -1 overflow",
        // Indirect-call trap scratch; see the trap constraints in
        // ccs/call.rs.
        COL_CI_ENTRY_IS_NULL: [Boolean; 1] =>
            "zero-test flag: the table entry (table value) read by this row is a null funcref",
        COL_CI_ENTRY_NULL_INV: [Field; 1] => "inverse witness for the null-funcref zero test",
        COL_CI_TYPE_EQ: [Boolean; 1] => "zero-test flag: callee type id equals the indirect call's expected type id",
        COL_CI_TYPE_EQ_INV: [Field; 1] => "inverse witness for the callee-type equality test",
        COL_CALL_INDIRECT_IS_TRAP: [Boolean; 1] =>
            "this row is an indirect call trapping on OOB index, null entry, or callee type mismatch",
        COL_CALL_INDIRECT_IS_NOT_TRAP: [Boolean; 1] =>
            "non-trapping indirect-call row: gates callee metadata and entry-pc reads",
        COL_FUNCTION_CALL_TYPE_LOOKUP_GATE: [Boolean; 1] =>
            "indirect-call row with an in-bounds non-null entry: gates the function-types read",
        // Shared unsigned-comparison scratch for the bounds traps (see
        // `push_unsigned_ge_gadget`). `low` is the range-checked borrow-bit
        // remainder; `ge` is `a >= b` for whichever mutually-exclusive comparison
        // the row's opcode selects.
        COL_CMP_LOW: [U32; 1] => "borrow-bit remainder of the active unsigned comparison",
        COL_CMP_GE: [Boolean; 1] => "result a >= b of the active unsigned comparison",
        COL_CI_OOB: [Boolean; 1] => "whether an indirect call traps because the table index is >= the table size",
        COL_TABLE_SIZE_READ_ENABLED: [Boolean; 1] => "table_sizes read gate: table.size or an indirect call",
        COL_MEM_OOB: [Boolean; 1] =>
            "whether this load/store traps because the access is past the end of linear memory",
        COL_MEM_LOAD_LIVE: [Boolean; 1] =>
            "load lane gate factor: a load row that is not OOB (de-gates lane reads on an OOB trap)",
        COL_MEM_STORE_LIVE: [Boolean; 1] =>
            "store lane gate factor: a store row that is not OOB (de-gates lane writes on an OOB trap)",
        COL_GROW_SUCCESS: [Boolean; 1] => "memory.grow row: the growth fits under max pages (before + delta <= max)",
        COL_HALTED_BEFORE: [Boolean; 1] => "carried terminal flag before this row",
    ]
}

mod opcode_selectors;
pub use opcode_selectors::{selector_col, SELECTOR_COLS};

// The host-event absorb column groups are addressed arithmetically from
// their first member; pin the declaration order so a layout reshuffle is a
// compile error instead of silent column aliasing.
const _: () = {
    assert!(COL_EVBUF7_BEFORE == COL_EVBUF0_BEFORE + 7);
    assert!(COL_EVBUF7_AFTER == COL_EVBUF0_AFTER + 7);
    assert!(COL_EVBUF_SLOT3_BEFORE == COL_EVBUF_SLOT0_BEFORE + 3);
    assert!(COL_EVBUF_SLOT3_AFTER == COL_EVBUF_SLOT0_AFTER + 3);
    assert!(COL_PERM_STATE11_BEFORE == COL_PERM_STATE0_BEFORE + 11);
    assert!(COL_PERM_STATE11_AFTER == COL_PERM_STATE0_AFTER + 11);
};
