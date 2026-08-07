//! Owns the static WASM row layout.

use super::isa::WasmOpcode;

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
    (COL_OUTPUT_VALUE_LO_BEFORE, "carried output low limb before"),
    (COL_OUTPUT_VALUE_LO_AFTER, "carried output low limb after"),
    (COL_OUTPUT_VALUE_HI_BEFORE, "carried output high limb before"),
    (COL_OUTPUT_VALUE_HI_AFTER, "carried output high limb after"),
    (COL_OPCODE_CODE, "opcode decode selector source", ColumnWidth::U32),
    (COL_PC_BEFORE, "transition source pc", ColumnWidth::U32),
    (COL_PC_AFTER, "transition destination pc", ColumnWidth::U32),
    (COL_CONTROL_CHOICE, "control-edge choice", ColumnWidth::U32),
    (COL_PC_EDGE_KIND, "pc edge kind", ColumnWidth::U32),
    (COL_WIDE_VALUES_ENABLED, "wide-value flag", ColumnWidth::Boolean),
    (COL_OUTPUT_CAPTURED, "output-capture gate", ColumnWidth::Boolean),
    (COL_SP_BEFORE, "transition source stack pointer", ColumnWidth::U32),
    (COL_SP_AFTER, "transition destination stack pointer", ColumnWidth::U32),
    (COL_STACK_FRAME_BASE_BEFORE, "operand-stack frame base before", ColumnWidth::U32),
    (COL_STACK_FRAME_BASE_AFTER, "operand-stack frame base after", ColumnWidth::U32),
    (COL_HALTED, "terminal row flag", ColumnWidth::Boolean),
    (COL_IS_PROGRAM_ROW, "decoded wasm program row", ColumnWidth::Boolean),
    (COL_PC_ROM_ACTIVE, "static pc-edge ROM gate", ColumnWidth::Boolean),
    (COL_PC_EDGE_KIND_IS_STATIC, "static pc-edge flag", ColumnWidth::Boolean),
    (COL_PC_EDGE_KIND_INV, "inverse witness for pc edge-kind zero test"),
    (COL_PARAM_INIT_ACTIVE_BEFORE, "parameter-init mode before", ColumnWidth::Boolean),
    (COL_PARAM_INIT_ACTIVE_AFTER, "parameter-init mode after", ColumnWidth::Boolean),
    (COL_TAIL_CALL_PENDING_BEFORE, "tail-entry pending before", ColumnWidth::Boolean),
    (COL_TAIL_CALL_PENDING_AFTER, "tail-entry pending after", ColumnWidth::Boolean),
    (COL_TAIL_ENTER_ACTIVE, "tail-enter aux row", ColumnWidth::Boolean),
    (COL_TAIL_DISCARD_COUNT, "tail-enter discard count", ColumnWidth::U32),
    (COL_PADDING_ACTIVE, "padding row", ColumnWidth::Boolean),
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
        COL_TURN_EXPORT_FREF_BEFORE,
        "export function ref owning the current grammar turn before this row",
        ColumnWidth::U32
    ),
    (
        COL_TURN_EXPORT_FREF_AFTER,
        "export function ref owning the current grammar turn after this row",
        ColumnWidth::U32
    ),
    (
        COL_COMM_CHAIN0_BEFORE,
        "host-event commitment chain limb 0 before this row"
    ),
    (
        COL_COMM_CHAIN1_BEFORE,
        "host-event commitment chain limb 1 before this row"
    ),
    (
        COL_COMM_CHAIN2_BEFORE,
        "host-event commitment chain limb 2 before this row"
    ),
    (
        COL_COMM_CHAIN3_BEFORE,
        "host-event commitment chain limb 3 before this row"
    ),
    (
        COL_COMM_CHAIN0_AFTER,
        "host-event commitment chain limb 0 after this row"
    ),
    (
        COL_COMM_CHAIN1_AFTER,
        "host-event commitment chain limb 1 after this row"
    ),
    (
        COL_COMM_CHAIN2_AFTER,
        "host-event commitment chain limb 2 after this row"
    ),
    (
        COL_COMM_CHAIN3_AFTER,
        "host-event commitment chain limb 3 after this row"
    ),
    // Host-event absorb machinery: the 8-word block buffer host-call rows
    // stream event words into, the one-hot pair-slot cursor, the
    // pending-permutation flag, and the perm-row group state (round cursor +
    // 12-lane running permutation state). See `ir::WasmEventAbsorbState`.
    (COL_EVBUF0_BEFORE, "host-event block buffer word 0 before this row"),
    (COL_EVBUF1_BEFORE, "host-event block buffer word 1 before this row"),
    (COL_EVBUF2_BEFORE, "host-event block buffer word 2 before this row"),
    (COL_EVBUF3_BEFORE, "host-event block buffer word 3 before this row"),
    (COL_EVBUF4_BEFORE, "host-event block buffer word 4 before this row"),
    (COL_EVBUF5_BEFORE, "host-event block buffer word 5 before this row"),
    (COL_EVBUF6_BEFORE, "host-event block buffer word 6 before this row"),
    (COL_EVBUF7_BEFORE, "host-event block buffer word 7 before this row"),
    (COL_EVBUF0_AFTER, "host-event block buffer word 0 after this row"),
    (COL_EVBUF1_AFTER, "host-event block buffer word 1 after this row"),
    (COL_EVBUF2_AFTER, "host-event block buffer word 2 after this row"),
    (COL_EVBUF3_AFTER, "host-event block buffer word 3 after this row"),
    (COL_EVBUF4_AFTER, "host-event block buffer word 4 after this row"),
    (COL_EVBUF5_AFTER, "host-event block buffer word 5 after this row"),
    (COL_EVBUF6_AFTER, "host-event block buffer word 6 after this row"),
    (COL_EVBUF7_AFTER, "host-event block buffer word 7 after this row"),
    (
        COL_EVBUF_SLOT0_BEFORE,
        "one-hot next event-word pair slot 0 before this row",
        ColumnWidth::Boolean
    ),
    (
        COL_EVBUF_SLOT1_BEFORE,
        "one-hot next event-word pair slot 1 before this row",
        ColumnWidth::Boolean
    ),
    (
        COL_EVBUF_SLOT2_BEFORE,
        "one-hot next event-word pair slot 2 before this row",
        ColumnWidth::Boolean
    ),
    (
        COL_EVBUF_SLOT3_BEFORE,
        "one-hot next event-word pair slot 3 before this row",
        ColumnWidth::Boolean
    ),
    (
        COL_EVBUF_SLOT0_AFTER,
        "one-hot next event-word pair slot 0 after this row",
        ColumnWidth::Boolean
    ),
    (
        COL_EVBUF_SLOT1_AFTER,
        "one-hot next event-word pair slot 1 after this row",
        ColumnWidth::Boolean
    ),
    (
        COL_EVBUF_SLOT2_AFTER,
        "one-hot next event-word pair slot 2 after this row",
        ColumnWidth::Boolean
    ),
    (
        COL_EVBUF_SLOT3_AFTER,
        "one-hot next event-word pair slot 3 after this row",
        ColumnWidth::Boolean
    ),
    (
        COL_PERM_PENDING_BEFORE,
        "a filled host-event block awaits its perm rows before this row",
        ColumnWidth::Boolean
    ),
    (
        COL_PERM_PENDING_AFTER,
        "a filled host-event block awaits its perm rows after this row",
        ColumnWidth::Boolean
    ),
    (
        COL_PERM_ROUND_BEFORE,
        "perm-group row position before this row (0 when idle; bounded by the position one-hot)"
    ),
    (
        COL_PERM_ROUND_AFTER,
        "perm-group row position after this row (0 when idle; bounded by the position one-hot)"
    ),
    (
        COL_PERM_ROUND_BEFORE_IS_ZERO,
        "zero-test flag for the perm-group row position before this row",
        ColumnWidth::Boolean
    ),
    (
        COL_PERM_ROUND_BEFORE_INV,
        "inverse witness for the perm-group row position before this row"
    ),
    (COL_PERM_STATE0_BEFORE, "chain permutation lane 0 before this row"),
    (COL_PERM_STATE1_BEFORE, "chain permutation lane 1 before this row"),
    (COL_PERM_STATE2_BEFORE, "chain permutation lane 2 before this row"),
    (COL_PERM_STATE3_BEFORE, "chain permutation lane 3 before this row"),
    (COL_PERM_STATE4_BEFORE, "chain permutation lane 4 before this row"),
    (COL_PERM_STATE5_BEFORE, "chain permutation lane 5 before this row"),
    (COL_PERM_STATE6_BEFORE, "chain permutation lane 6 before this row"),
    (COL_PERM_STATE7_BEFORE, "chain permutation lane 7 before this row"),
    (COL_PERM_STATE8_BEFORE, "chain permutation lane 8 before this row"),
    (COL_PERM_STATE9_BEFORE, "chain permutation lane 9 before this row"),
    (COL_PERM_STATE10_BEFORE, "chain permutation lane 10 before this row"),
    (COL_PERM_STATE11_BEFORE, "chain permutation lane 11 before this row"),
    (COL_PERM_STATE0_AFTER, "chain permutation lane 0 after this row"),
    (COL_PERM_STATE1_AFTER, "chain permutation lane 1 after this row"),
    (COL_PERM_STATE2_AFTER, "chain permutation lane 2 after this row"),
    (COL_PERM_STATE3_AFTER, "chain permutation lane 3 after this row"),
    (COL_PERM_STATE4_AFTER, "chain permutation lane 4 after this row"),
    (COL_PERM_STATE5_AFTER, "chain permutation lane 5 after this row"),
    (COL_PERM_STATE6_AFTER, "chain permutation lane 6 after this row"),
    (COL_PERM_STATE7_AFTER, "chain permutation lane 7 after this row"),
    (COL_PERM_STATE8_AFTER, "chain permutation lane 8 after this row"),
    (COL_PERM_STATE9_AFTER, "chain permutation lane 9 after this row"),
    (COL_PERM_STATE10_AFTER, "chain permutation lane 10 after this row"),
    (COL_PERM_STATE11_AFTER, "chain permutation lane 11 after this row"),
    (
        COL_GRAMMAR_MODE_BEFORE,
        "per-program constant: chain absorbs embedder grammar events (1) or raw host-call records (0)",
        ColumnWidth::Boolean
    ),
    (
        COL_GRAMMAR_MODE_AFTER,
        "per-program constant: chain absorbs embedder grammar events (1) or raw host-call records (0)",
        ColumnWidth::Boolean
    ),
    (
        COL_GATHER_ACTIVE,
        "grammar-mode row staging one expanded event block into the absorb buffer",
        ColumnWidth::Boolean
    ),
    (
        COL_RAW_HOST_CALL,
        "host-call program row with the raw absorb machinery active: host_call_gate · (1 - grammar_mode)",
        ColumnWidth::Boolean
    ),
    (
        COL_RAW_ARGS_ACTIVE,
        "host-arg row with the raw absorb machinery active: host_args_active · (1 - grammar_mode)",
        ColumnWidth::Boolean
    ),
    (
        COL_RAW_RESULT_ACTIVE,
        "host-result row with the raw absorb machinery active: host_result_active · (1 - grammar_mode)",
        ColumnWidth::Boolean
    ),
    // Grammar-mode gather machinery: carried schedule/cursor/oracle state
    // plus the per-row grammar-ROM interface columns (bound on gather rows
    // by the `grammar_slot_*` families, on call/result rows by the
    // `grammar_event_counts_*` families). See `ir::WasmGrammarState` and
    // `docs/host-event-grammar-tables.md`.
    (COL_GRAMMAR_EVREM_BEFORE, "grammar events still owed in the current phase, before this row"),
    (COL_GRAMMAR_EVREM_AFTER, "grammar events still owed in the current phase, after this row"),
    (
        COL_GRAMMAR_EVREM_BEFORE_IS_ZERO,
        "zero-test flag for the owed grammar events before this row",
        ColumnWidth::Boolean
    ),
    (
        COL_GRAMMAR_EVREM_BEFORE_INV,
        "inverse witness for the owed grammar events before this row"
    ),
    (COL_GRAMMAR_EVIDX_BEFORE, "current grammar event index within the template, before this row"),
    (COL_GRAMMAR_EVIDX_AFTER, "current grammar event index within the template, after this row"),
    (COL_GRAMMAR_ARGS_BASE_BEFORE, "stack slot index of the current call's first argument, before this row"),
    (COL_GRAMMAR_ARGS_BASE_AFTER, "stack slot index of the current call's first argument, after this row"),
    (COL_GRAMMAR_SLOT_CURSOR_BEFORE, "next block word a gather row stages (0..=7), before this row"),
    (COL_GRAMMAR_SLOT_CURSOR_AFTER, "next block word a gather row stages (0..=7), after this row"),
    (COL_GRAMMAR_SLOT_KIND, "grammar-ROM slot source kind (0 const, 1 arg, 2 result, 3 claim, 4 input-local, 5 output, 6 memory-read, 7 memory-write)"),
    (COL_GRAMMAR_SLOT_ARG, "grammar-ROM slot arg/oracle index"),
    (
        COL_GRAMMAR_SLOT_VARIANT,
        "encoded grammar-ROM slot variant: value kinds use 0 lo / 1 hi; memory kinds use bit 0 for local base and bit 1 for byte width"
    ),
    (COL_GRAMMAR_SLOT_CONST_LO, "grammar-ROM slot constant, low 32 bits"),
    (COL_GRAMMAR_SLOT_CONST_HI, "grammar-ROM slot constant, high 32 bits"),
    (COL_GRAMMAR_PRE_COUNT, "grammar-ROM event count for the called import / entered export (biased +1)"),
    (COL_GRAMMAR_POST_COUNT, "grammar-ROM exit-event count for the halting export"),
    (
        COL_GRAMMAR_HOST_CALL,
        "host-call program row in grammar mode: host_call_gate · grammar_mode"
    ),
    (
        COL_GATHER_LOCAL_WRITE,
        "gather row writing a claim-input word into an entry-frame locals lane (slot kind 4); gates the hi-lane write (zero on lo rows)",
        ColumnWidth::Boolean
    ),
    (
        COL_GATHER_LOCAL_WRITE_LO,
        "input-local gather row targeting the lo lane: gather_local_write · (1 - slot_variant)",
        ColumnWidth::Boolean
    ),
    (
        COL_GRAMMAR_EXIT_LATCH,
        "clean export-halt row in grammar mode: loads the export's exit-event schedule",
        ColumnWidth::Boolean
    ),
    (
        COL_TURN_BOUNDARY,
        "multi-turn re-entry row: re-arms the output, loads the next export's entry schedule, jumps to its entry pc",
        ColumnWidth::Boolean
    ),
    (
        COL_PC_FREF_ACTIVE,
        "pc -> function-ref ROM gate: every row except gather rows (post-halt exit gathers sit past the last pc)",
        ColumnWidth::Boolean
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
        COL_GUEST_ENTRY_ACTIVE,
        "guest-entry row flag: this row enters a traced guest callee",
        ColumnWidth::Boolean
    ),
    (
        COL_CALL_STACK_PUSH_PRESENT,
        "flag indicating that this row saves a caller return context",
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
        COL_CALL_STACK_CALLER_SP_BASE_VALUE,
        "call-stack caller operand-stack base: written on push and restored on pop",
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
    (
        COL_STACK_WRITE0_HI_ACTIVE,
        "stack write hi-word port gate: write0_active, plus result-hi gather rows writing only the hi lane",
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
        COL_PROGRAM_LOCAL_INDEX_ACTIVE,
        "PC-indexed local immediate ROM gate",
        ColumnWidth::Boolean
    ),
    (
        COL_PROGRAM_GLOBAL_INDEX_ACTIVE,
        "PC-indexed global immediate ROM gate",
        ColumnWidth::Boolean
    ),
    (
        COL_PROGRAM_TABLE_ID_ACTIVE,
        "PC-indexed table-id immediate ROM gate",
        ColumnWidth::Boolean
    ),
    (
        COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE,
        "PC-indexed call-indirect immediate ROM gate",
        ColumnWidth::Boolean
    ),
    (
        COL_TABLE_READ_ENABLED,
        "table memory read gate for table.get and indirect calls",
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
        ColumnWidth::Byte
    ),
    (
        COL_CALL_RESULT_COUNT,
        "result count for the selected call target",
        ColumnWidth::Byte
    ),
    (
        COL_TARGET_FUNCTION_IS_GUEST,
        "true when the selected call target is a guest-defined function",
        ColumnWidth::Boolean
    ),
    (
        COL_CALL_TARGET_METADATA,
        "packed call-target arity and guest flag; range follows from unpacking"
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
        COL_SEL_RETURN_CALL,
        "selector for return_call",
        ColumnWidth::Boolean
    ),
    (
        COL_SEL_RETURN_CALL_INDIRECT,
        "selector for return_call_indirect",
        ColumnWidth::Boolean
    ),
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
    // Indirect-call trap scratch; see the trap constraints in
    // ccs/call.rs.
    (
        COL_CI_ENTRY_IS_NULL,
        "zero-test flag: the table entry (table value) read by this row is a null funcref",
        ColumnWidth::Boolean
    ),
    (COL_CI_ENTRY_NULL_INV, "inverse witness for the null-funcref zero test"),
    (
        COL_CI_TYPE_EQ,
        "zero-test flag: callee type id equals the indirect call's expected type id",
        ColumnWidth::Boolean
    ),
    (COL_CI_TYPE_EQ_INV, "inverse witness for the callee-type equality test"),
    (
        COL_CALL_INDIRECT_IS_TRAP,
        "this row is an indirect call trapping on OOB index, null entry, or callee type mismatch",
        ColumnWidth::Boolean
    ),
    (
        COL_CALL_INDIRECT_IS_NOT_TRAP,
        "non-trapping indirect-call row: gates callee metadata and entry-pc reads",
        ColumnWidth::Boolean
    ),
    (
        COL_FUNCTION_CALL_TYPE_LOOKUP_GATE,
        "indirect-call row with an in-bounds non-null entry: gates the function-types read",
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
        "whether an indirect call traps because the table index is >= the table size",
        ColumnWidth::Boolean
    ),
    (
        COL_TABLE_SIZE_READ_ENABLED,
        "table_sizes read gate: table.size or an indirect call",
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
