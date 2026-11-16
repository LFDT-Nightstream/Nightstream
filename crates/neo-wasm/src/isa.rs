//! Owns WASM opcode taxonomy and stable operation ids.

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WasmOpcodeClass {
    System,
    ControlFlow,
    Numeric,
    Compare,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WasmMemoryAccessKind {
    Load,
    Store,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WasmMemoryExtension {
    Unextended,
    Zero,
    Sign,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WasmMemoryAccessInfo {
    pub kind: WasmMemoryAccessKind,
    pub width_bytes: u8,
    pub result_bits: u8,
    pub extension: WasmMemoryExtension,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WasmOperation {
    I32Clz,
    I32Ctz,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,
    I32And,
    I32Or,
    I32Xor,
    I32Mul,
    I64And,
    I64Or,
    I64Xor,
    I64Mul,
    I32Shl,
    I32ShrU,
    I32ShrS,
    I32Rotl,
    I32Rotr,
    I32DivU,
    I32DivS,
    I32RemU,
    I32RemS,
    I32Popcnt,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rotl,
    I64Rotr,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64Clz,
    I64Ctz,
    I64Popcnt,
}

impl WasmOperation {
    pub fn opcode(self) -> WasmOpcode {
        OPCODE_FACTS
            .iter()
            .find(|facts| facts.operation == Some(self))
            .expect("every WasmOperation has an OPCODE_FACTS row")
            .opcode
    }

    pub fn all() -> [Self; 48] {
        [
            Self::I32Clz,
            Self::I32Ctz,
            Self::I32LtS,
            Self::I32LtU,
            Self::I32GtS,
            Self::I32GtU,
            Self::I32LeS,
            Self::I32LeU,
            Self::I32GeS,
            Self::I32GeU,
            Self::I32And,
            Self::I32Or,
            Self::I32Xor,
            Self::I32Mul,
            Self::I64And,
            Self::I64Or,
            Self::I64Xor,
            Self::I64Mul,
            Self::I32Shl,
            Self::I32ShrU,
            Self::I32ShrS,
            Self::I32Rotl,
            Self::I32Rotr,
            Self::I32DivU,
            Self::I32DivS,
            Self::I32RemU,
            Self::I32RemS,
            Self::I32Popcnt,
            Self::I64LtS,
            Self::I64LtU,
            Self::I64GtS,
            Self::I64GtU,
            Self::I64LeS,
            Self::I64LeU,
            Self::I64GeS,
            Self::I64GeU,
            Self::I64Shl,
            Self::I64ShrS,
            Self::I64ShrU,
            Self::I64Rotl,
            Self::I64Rotr,
            Self::I64DivS,
            Self::I64DivU,
            Self::I64RemS,
            Self::I64RemU,
            Self::I64Clz,
            Self::I64Ctz,
            Self::I64Popcnt,
        ]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum WasmOpcode {
    Nop,
    I32Const,
    I64Const,
    RefFunc,
    I32Add,
    I64Add,
    I32Sub,
    I64Sub,
    I32Load,
    I32Load8S,
    I32Load8U,
    I32Load16S,
    I32Load16U,
    I64Load,
    I32Store,
    I32Store8,
    I32Store16,
    I64Store,
    I64Store8,
    I64Store16,
    I64Store32,
    I64Load8U,
    I64Load16U,
    I64Load32U,
    I64Load8S,
    I64Load16S,
    I64Load32S,
    I32WrapI64,
    I64ExtendI32U,
    I64ExtendI32S,
    I32Extend8S,
    I32Extend16S,
    I64Extend8S,
    I64Extend16S,
    I64Extend32S,
    MemorySize,
    MemoryGrow,
    TableSize,
    TableGet,
    TableSet,
    Drop,
    Br,
    Block,
    Loop,
    If,
    Else,
    End,
    Unreachable,
    I32Clz,
    I32Ctz,
    I32Popcnt,
    I32Eqz,
    I64Eqz,
    I32Eq,
    I32Ne,
    I64Eq,
    I64Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,
    I32And,
    I32Or,
    I32Xor,
    I32Mul,
    I64And,
    I64Or,
    I64Xor,
    I64Mul,
    I32Shl,
    I32ShrU,
    I32ShrS,
    I32Rotl,
    I32Rotr,
    I32DivU,
    I32DivS,
    I32RemU,
    I32RemS,
    Select,
    BrIf,
    BrTable,
    Call,
    CallIndirect,
    ReturnCall,
    ReturnCallIndirect,
    /// Synthetic first half of `call_indirect` / `return_call_indirect`: checks
    /// the table entry and replaces the index on the stack with the function
    /// reference the following invocation calls. It has its own pc, just before
    /// the invocation's.
    CallIndirectResolve,
    Return,
    LocalGet,
    LocalSet,
    LocalTee,
    GlobalGet,
    GlobalSet,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rotl,
    I64Rotr,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64Clz,
    I64Ctz,
    I64Popcnt,
    Trap,
    Unsupported,
}

impl WasmOpcode {
    pub const fn uses_pc_rom(self) -> bool {
        matches!(
            self,
            Self::Block
                | Self::Loop
                | Self::If
                | Self::Else
                | Self::End
                | Self::Br
                | Self::BrIf
                | Self::BrTable
                | Self::CallIndirectResolve
        )
    }

    pub const fn falls_through(self) -> bool {
        !self.uses_pc_rom()
            && !matches!(
                self,
                Self::Return
                    | Self::Call
                    | Self::ReturnCall
                    | Self::Unreachable
                    | Self::CallIndirect
                    | Self::ReturnCallIndirect
                    | Self::Trap
                    | Self::Unsupported
            )
    }

    pub fn supported() -> [Self; 116] {
        std::array::from_fn(|i| OPCODE_FACTS[i].opcode)
    }

    /// True for opcodes that access linear memory. Single source of truth
    /// for the linear-memory selector list — see
    /// [`crate::ccs::linear_memory`].
    pub fn uses_linear_memory(self) -> bool {
        self.memory_access_info().is_some()
    }

    /// SPEC.md section 7.6: fixed-arity replacements write over operand zero.
    pub(crate) fn uses_stack_rmw(self) -> bool {
        let facts = self.facts();
        (1..=3).contains(&facts.stack_reads) && facts.stack_writes == 1
    }

    /// Rocq `StackPorts.full_stack_read0_hi_bound`: stores use a wasm32 address; indirect invocations
    /// use the resolved u32 function reference. Neither consumes the high limb.
    pub(crate) fn omits_stack_read0_hi(self) -> bool {
        matches!(self, Self::CallIndirect | Self::ReturnCallIndirect)
            || self
                .memory_access_info()
                .is_some_and(|access| access.kind == WasmMemoryAccessKind::Store)
    }

    /// True when the instruction consumes the PC-indexed local immediate.
    pub fn uses_local_index_immediate(self) -> bool {
        matches!(self, Self::LocalGet | Self::LocalSet | Self::LocalTee)
    }

    /// True when the instruction consumes the PC-indexed global immediate.
    pub fn uses_global_index_immediate(self) -> bool {
        matches!(self, Self::GlobalGet | Self::GlobalSet)
    }

    /// True when the instruction consumes the PC-indexed table namespace.
    pub fn uses_table_id_immediate(self) -> bool {
        matches!(
            self,
            Self::TableSize | Self::TableGet | Self::TableSet | Self::CallIndirectResolve
        )
    }

    /// True for opcodes whose row carries a meaningful `wide_values_enabled`
    /// (= 1) flag: rows that consume or produce an i64 stack value, plus the
    /// polymorphic ops (`Drop`, `Select`, `Call`, `CallIndirect`,
    /// `local.*`, `global.*`) which can act on i64 values and whose CCS rows
    /// gate the wide-stack-lane machinery accordingly.
    ///
    /// Single source of truth for the wide-value gating row in
    /// [`crate::ccs::build_wasm_relation`], the `wide_values_enabled`
    /// determination in `adapters::wasmtime::normalize`, and the test-row
    /// builders. The constraint `is_program_row · (wide_values_enabled −
    /// Σ uses_wide_values selectors) = 0` forces these flags into agreement.
    pub fn uses_wide_values(self) -> bool {
        matches!(
            self,
            Self::I64Const
                | Self::I64Add
                | Self::I64Sub
                | Self::I64Load
                | Self::I64Store
                | Self::I64Store8
                | Self::I64Store16
                | Self::I64Store32
                | Self::I64Load8U
                | Self::I64Load16U
                | Self::I64Load32U
                | Self::I64Load8S
                | Self::I64Load16S
                | Self::I64Load32S
                | Self::I32WrapI64
                | Self::I64ExtendI32U
                | Self::I64ExtendI32S
                | Self::I64Extend8S
                | Self::I64Extend16S
                | Self::I64Extend32S
                | Self::I64Eqz
                | Self::I64Eq
                | Self::I64Ne
                | Self::I64And
                | Self::I64Or
                | Self::I64Xor
                | Self::I64Mul
                | Self::I64LtS
                | Self::I64LtU
                | Self::I64GtS
                | Self::I64GtU
                | Self::I64LeS
                | Self::I64LeU
                | Self::I64GeS
                | Self::I64GeU
                | Self::I64Shl
                | Self::I64ShrS
                | Self::I64ShrU
                | Self::I64Rotl
                | Self::I64Rotr
                | Self::I64DivS
                | Self::I64DivU
                | Self::I64RemS
                | Self::I64RemU
                | Self::I64Clz
                | Self::I64Ctz
                | Self::I64Popcnt
                | Self::Drop
                | Self::Select
                | Self::Call
                | Self::CallIndirect
                | Self::ReturnCall
                | Self::ReturnCallIndirect
                | Self::LocalGet
                | Self::LocalSet
                | Self::LocalTee
                | Self::GlobalGet
                | Self::GlobalSet
        )
    }

    /// Div/rem opcodes that trap on a zero divisor (stack read1). Single
    /// source of truth for the CCS `div_trap` gate, the witness builder, and
    /// the trace normalizer's trap-row synthesis.
    pub fn traps_on_zero_divisor(self) -> bool {
        matches!(
            self,
            Self::I32DivS
                | Self::I32DivU
                | Self::I32RemS
                | Self::I32RemU
                | Self::I64DivS
                | Self::I64DivU
                | Self::I64RemS
                | Self::I64RemU
        )
    }

    /// Signed div opcodes that trap on overflow (MIN_value / -1). Single
    /// source of truth for the CCS `div_overflow` gate, the witness builder,
    /// and the trace normalizer's trap-row synthesis. `rem_s` is excluded:
    /// wasm defines `MIN rem -1 = 0` without trapping.
    pub fn traps_on_signed_overflow(self) -> bool {
        matches!(self, Self::I32DivS | Self::I64DivS)
    }

    pub fn memory_access_info(self) -> Option<WasmMemoryAccessInfo> {
        use WasmMemoryAccessKind::{Load, Store};
        use WasmMemoryExtension::{Sign, Zero};
        match self {
            Self::I32Load8S => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 1,
                result_bits: 32,
                extension: Sign,
            }),
            Self::I32Load8U => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 1,
                result_bits: 32,
                extension: Zero,
            }),
            Self::I64Load8U => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 1,
                result_bits: 64,
                extension: Zero,
            }),
            Self::I64Load8S => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 1,
                result_bits: 64,
                extension: Sign,
            }),
            Self::I32Store8 | Self::I64Store8 => Some(WasmMemoryAccessInfo {
                kind: Store,
                width_bytes: 1,
                result_bits: 0,
                extension: WasmMemoryExtension::Unextended,
            }),
            Self::I32Load16S => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 2,
                result_bits: 32,
                extension: Sign,
            }),
            Self::I32Load16U => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 2,
                result_bits: 32,
                extension: Zero,
            }),
            Self::I64Load16U => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 2,
                result_bits: 64,
                extension: Zero,
            }),
            Self::I64Load16S => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 2,
                result_bits: 64,
                extension: Sign,
            }),
            Self::I32Store16 | Self::I64Store16 => Some(WasmMemoryAccessInfo {
                kind: Store,
                width_bytes: 2,
                result_bits: 0,
                extension: WasmMemoryExtension::Unextended,
            }),
            Self::I32Load => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 4,
                result_bits: 32,
                extension: WasmMemoryExtension::Unextended,
            }),
            Self::I64Load32U => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 4,
                result_bits: 64,
                extension: Zero,
            }),
            Self::I64Load32S => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 4,
                result_bits: 64,
                extension: Sign,
            }),
            Self::I32Store | Self::I64Store32 => Some(WasmMemoryAccessInfo {
                kind: Store,
                width_bytes: 4,
                result_bits: 0,
                extension: WasmMemoryExtension::Unextended,
            }),
            Self::I64Load => Some(WasmMemoryAccessInfo {
                kind: Load,
                width_bytes: 8,
                result_bits: 64,
                extension: WasmMemoryExtension::Unextended,
            }),
            Self::I64Store => Some(WasmMemoryAccessInfo {
                kind: Store,
                width_bytes: 8,
                result_bits: 0,
                extension: WasmMemoryExtension::Unextended,
            }),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        self.facts().name
    }

    fn facts(self) -> OpcodeFacts {
        OPCODE_FACTS[self as usize]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmOpcodeInfo {
    pub opcode: WasmOpcode,
    pub code: u16,
    pub name: &'static str,
    pub class: WasmOpcodeClass,
    pub stack_reads: u8,
    pub stack_writes: u8,
    pub memory_access: Option<WasmMemoryAccessInfo>,
    pub operation: Option<WasmOperation>,
}

pub fn opcode_info_from_code(code: u16) -> WasmOpcodeInfo {
    let opcode = OPCODE_BY_CODE[usize::from(code)];
    let facts = opcode.facts();
    WasmOpcodeInfo {
        opcode,
        code,
        name: facts.name,
        class: facts.class,
        stack_reads: facts.stack_reads,
        stack_writes: facts.stack_writes,
        memory_access: opcode.memory_access_info(),
        operation: facts.operation,
    }
}

pub fn opcode_code(op: WasmOpcode) -> u16 {
    op.facts().code
}

#[derive(Clone, Copy)]
struct OpcodeFacts {
    opcode: WasmOpcode,
    code: u16,
    name: &'static str,
    class: WasmOpcodeClass,
    stack_reads: u8,
    stack_writes: u8,
    operation: Option<WasmOperation>,
}

// Canonical WebAssembly binary opcode bytes:
// https://webassembly.github.io/spec/core/binary/instructions.html
#[rustfmt::skip]
const OPCODE_FACTS: [OpcodeFacts; 118] = {
    use WasmOpcode as Op;
    use WasmOpcodeClass::{Compare, ControlFlow, Numeric, System, Unknown};
    use WasmOperation as Operation;

    const fn facts(
        opcode: WasmOpcode,
        code: u16,
        name: &'static str,
        class: WasmOpcodeClass,
        stack_reads: u8,
        stack_writes: u8,
        operation: Option<WasmOperation>,
    ) -> OpcodeFacts {
        OpcodeFacts { opcode, code, name, class, stack_reads, stack_writes, operation }
    }

    [
        facts(Op::Nop,                 0x01,   "nop",                   System,      0, 0, None),
        facts(Op::I32Const,            0x41,   "i32_const",             Numeric,     0, 1, None),
        facts(Op::I64Const,            0x42,   "i64_const",             Numeric,     0, 1, None),
        facts(Op::RefFunc,             0xD2,   "ref_func",              System,      0, 1, None),
        facts(Op::I32Add,              0x6A,   "i32_add",               Numeric,     2, 1, None),
        facts(Op::I64Add,              0x7C,   "i64_add",               Numeric,     2, 1, None),
        facts(Op::I32Sub,              0x6B,   "i32_sub",               Numeric,     2, 1, None),
        facts(Op::I64Sub,              0x7D,   "i64_sub",               Numeric,     2, 1, None),
        facts(Op::I32Load,             0x28,   "i32_load",              System,      1, 1, None),
        facts(Op::I32Load8S,           0x2C,   "i32_load8_s",           System,      1, 1, None),
        facts(Op::I32Load8U,           0x2D,   "i32_load8_u",           System,      1, 1, None),
        facts(Op::I32Load16S,          0x2E,   "i32_load16_s",          System,      1, 1, None),
        facts(Op::I32Load16U,          0x2F,   "i32_load16_u",          System,      1, 1, None),
        facts(Op::I64Load,             0x29,   "i64_load",              System,      1, 1, None),
        facts(Op::I32Store,            0x36,   "i32_store",             System,      2, 0, None),
        facts(Op::I32Store8,           0x3A,   "i32_store8",            System,      2, 0, None),
        facts(Op::I32Store16,          0x3B,   "i32_store16",           System,      2, 0, None),
        facts(Op::I64Store,            0x37,   "i64_store",             System,      2, 0, None),
        facts(Op::I64Store8,           0x3C,   "i64_store8",            System,      2, 0, None),
        facts(Op::I64Store16,          0x3D,   "i64_store16",           System,      2, 0, None),
        facts(Op::I64Store32,          0x3E,   "i64_store32",           System,      2, 0, None),
        facts(Op::I64Load8U,           0x31,   "i64_load8_u",           System,      1, 1, None),
        facts(Op::I64Load16U,          0x33,   "i64_load16_u",          System,      1, 1, None),
        facts(Op::I64Load32U,          0x35,   "i64_load32_u",          System,      1, 1, None),
        facts(Op::I64Load8S,           0x30,   "i64_load8_s",           System,      1, 1, None),
        facts(Op::I64Load16S,          0x32,   "i64_load16_s",          System,      1, 1, None),
        facts(Op::I64Load32S,          0x34,   "i64_load32_s",          System,      1, 1, None),
        facts(Op::I32WrapI64,          0xA7,   "i32_wrap_i64",          Numeric,     1, 1, None),
        facts(Op::I64ExtendI32U,       0xAD,   "i64_extend_i32_u",      Numeric,     1, 1, None),
        facts(Op::I64ExtendI32S,       0xAC,   "i64_extend_i32_s",      Numeric,     1, 1, None),
        facts(Op::I32Extend8S,         0xC0,   "i32_extend8_s",         Numeric,     1, 1, None),
        facts(Op::I32Extend16S,        0xC1,   "i32_extend16_s",        Numeric,     1, 1, None),
        facts(Op::I64Extend8S,         0xC2,   "i64_extend8_s",         Numeric,     1, 1, None),
        facts(Op::I64Extend16S,        0xC3,   "i64_extend16_s",        Numeric,     1, 1, None),
        facts(Op::I64Extend32S,        0xC4,   "i64_extend32_s",        Numeric,     1, 1, None),
        facts(Op::MemorySize,          0x3F,   "memory_size",           System,      0, 1, None),
        facts(Op::MemoryGrow,          0x40,   "memory_grow",           System,      1, 1, None),
        facts(Op::TableSize,           0xFC10, "table_size",            System,      0, 1, None),
        facts(Op::TableGet,            0x25,   "table_get",             System,      1, 1, None),
        facts(Op::TableSet,            0x26,   "table_set",             System,      2, 0, None),
        facts(Op::Drop,                0x1A,   "drop",                  System,      1, 0, None),
        facts(Op::Br,                  0x0C,   "br",                    ControlFlow, 0, 0, None),
        facts(Op::Block,               0x02,   "block",                 ControlFlow, 0, 0, None),
        facts(Op::Loop,                0x03,   "loop",                  ControlFlow, 0, 0, None),
        facts(Op::If,                  0x04,   "if",                    ControlFlow, 1, 0, None),
        facts(Op::Else,                0x05,   "else",                  ControlFlow, 0, 0, None),
        facts(Op::End,                 0x0B,   "end",                   ControlFlow, 0, 0, None),
        facts(Op::Unreachable,         0x00,   "unreachable",           System,      0, 0, None),
        facts(Op::I32Clz,              0x67,   "i32_clz",               Numeric,     1, 1, Some(Operation::I32Clz)),
        facts(Op::I32Ctz,              0x68,   "i32_ctz",               Numeric,     1, 1, Some(Operation::I32Ctz)),
        facts(Op::I32Popcnt,           0x69,   "i32_popcnt",            Numeric,     1, 1, Some(Operation::I32Popcnt)),
        facts(Op::I32Eqz,              0x45,   "i32_eqz",               Compare,     1, 1, None),
        facts(Op::I64Eqz,              0x50,   "i64_eqz",               Compare,     1, 1, None),
        facts(Op::I32Eq,               0x46,   "i32_eq",                Compare,     2, 1, None),
        facts(Op::I32Ne,               0x47,   "i32_ne",                Compare,     2, 1, None),
        facts(Op::I64Eq,               0x51,   "i64_eq",                Compare,     2, 1, None),
        facts(Op::I64Ne,               0x52,   "i64_ne",                Compare,     2, 1, None),
        facts(Op::I32LtS,              0x48,   "i32_lt_s",              Compare,     2, 1, Some(Operation::I32LtS)),
        facts(Op::I32LtU,              0x49,   "i32_lt_u",              Compare,     2, 1, Some(Operation::I32LtU)),
        facts(Op::I32GtS,              0x4A,   "i32_gt_s",              Compare,     2, 1, Some(Operation::I32GtS)),
        facts(Op::I32GtU,              0x4B,   "i32_gt_u",              Compare,     2, 1, Some(Operation::I32GtU)),
        facts(Op::I32LeS,              0x4C,   "i32_le_s",              Compare,     2, 1, Some(Operation::I32LeS)),
        facts(Op::I32LeU,              0x4D,   "i32_le_u",              Compare,     2, 1, Some(Operation::I32LeU)),
        facts(Op::I32GeS,              0x4E,   "i32_ge_s",              Compare,     2, 1, Some(Operation::I32GeS)),
        facts(Op::I32GeU,              0x4F,   "i32_ge_u",              Compare,     2, 1, Some(Operation::I32GeU)),
        facts(Op::I32And,              0x71,   "i32_and",               Numeric,     2, 1, Some(Operation::I32And)),
        facts(Op::I32Or,               0x72,   "i32_or",                Numeric,     2, 1, Some(Operation::I32Or)),
        facts(Op::I32Xor,              0x73,   "i32_xor",               Numeric,     2, 1, Some(Operation::I32Xor)),
        facts(Op::I32Mul,              0x6C,   "i32_mul",               Numeric,     2, 1, Some(Operation::I32Mul)),
        facts(Op::I64And,              0x83,   "i64_and",               Numeric,     2, 1, Some(Operation::I64And)),
        facts(Op::I64Or,               0x84,   "i64_or",                Numeric,     2, 1, Some(Operation::I64Or)),
        facts(Op::I64Xor,              0x85,   "i64_xor",               Numeric,     2, 1, Some(Operation::I64Xor)),
        facts(Op::I64Mul,              0x7E,   "i64_mul",               Numeric,     2, 1, Some(Operation::I64Mul)),
        facts(Op::I32Shl,              0x74,   "i32_shl",               Numeric,     2, 1, Some(Operation::I32Shl)),
        facts(Op::I32ShrU,             0x76,   "i32_shr_u",             Numeric,     2, 1, Some(Operation::I32ShrU)),
        facts(Op::I32ShrS,             0x75,   "i32_shr_s",             Numeric,     2, 1, Some(Operation::I32ShrS)),
        facts(Op::I32Rotl,             0x77,   "i32_rotl",              Numeric,     2, 1, Some(Operation::I32Rotl)),
        facts(Op::I32Rotr,             0x78,   "i32_rotr",              Numeric,     2, 1, Some(Operation::I32Rotr)),
        facts(Op::I32DivU,             0x6E,   "i32_div_u",             Numeric,     2, 1, Some(Operation::I32DivU)),
        facts(Op::I32DivS,             0x6D,   "i32_div_s",             Numeric,     2, 1, Some(Operation::I32DivS)),
        facts(Op::I32RemU,             0x70,   "i32_rem_u",             Numeric,     2, 1, Some(Operation::I32RemU)),
        facts(Op::I32RemS,             0x6F,   "i32_rem_s",             Numeric,     2, 1, Some(Operation::I32RemS)),
        facts(Op::Select,              0x1B,   "select",                ControlFlow, 3, 1, None),
        facts(Op::BrIf,                0x0D,   "br_if",                 ControlFlow, 1, 0, None),
        facts(Op::BrTable,             0x0E,   "br_table",              ControlFlow, 1, 0, None),
        // Call-like arity is dynamic: direct rows read 0, indirect rows read
        // the checked function reference, and aux rows handle arguments and
        // host results.
        facts(Op::Call,                0x10,   "call",                  ControlFlow, 0, 0, None),
        facts(Op::CallIndirect,        0x11,   "call_indirect",         ControlFlow, 0, 0, None),
        facts(Op::ReturnCall,          0x12,   "return_call",           ControlFlow, 0, 0, None),
        facts(Op::ReturnCallIndirect,  0x13,   "return_call_indirect",  ControlFlow, 0, 0, None),
        // Reads the table index, writes the checked function reference in its place.
        facts(Op::CallIndirectResolve, 0xFFF0, "call_indirect_resolve", ControlFlow, 1, 1, None),
        facts(Op::Return,              0x0F,   "return",                System,      0, 0, None),
        // stack_reads/writes are the operand stack effects only; local memory accesses
        // are tracked separately via local_read_value / local_write_value in the IR.
        facts(Op::LocalGet,            0x20,   "local_get",             System,      0, 1, None),
        facts(Op::LocalSet,            0x21,   "local_set",             System,      1, 0, None),
        facts(Op::LocalTee,            0x22,   "local_tee",             System,      1, 1, None),
        // stack_reads/writes are the operand stack effects only; global memory accesses
        // are tracked separately via global_read_value / global_write_value in the IR.
        facts(Op::GlobalGet,           0x23,   "global_get",            System,      0, 1, None),
        facts(Op::GlobalSet,           0x24,   "global_set",            System,      1, 0, None),
        facts(Op::I64LtS,              0x53,   "i64_lt_s",              Compare,     2, 1, Some(Operation::I64LtS)),
        facts(Op::I64LtU,              0x54,   "i64_lt_u",              Compare,     2, 1, Some(Operation::I64LtU)),
        facts(Op::I64GtS,              0x55,   "i64_gt_s",              Compare,     2, 1, Some(Operation::I64GtS)),
        facts(Op::I64GtU,              0x56,   "i64_gt_u",              Compare,     2, 1, Some(Operation::I64GtU)),
        facts(Op::I64LeS,              0x57,   "i64_le_s",              Compare,     2, 1, Some(Operation::I64LeS)),
        facts(Op::I64LeU,              0x58,   "i64_le_u",              Compare,     2, 1, Some(Operation::I64LeU)),
        facts(Op::I64GeS,              0x59,   "i64_ge_s",              Compare,     2, 1, Some(Operation::I64GeS)),
        facts(Op::I64GeU,              0x5A,   "i64_ge_u",              Compare,     2, 1, Some(Operation::I64GeU)),
        facts(Op::I64Shl,              0x86,   "i64_shl",               Numeric,     2, 1, Some(Operation::I64Shl)),
        facts(Op::I64ShrS,             0x87,   "i64_shr_s",             Numeric,     2, 1, Some(Operation::I64ShrS)),
        facts(Op::I64ShrU,             0x88,   "i64_shr_u",             Numeric,     2, 1, Some(Operation::I64ShrU)),
        facts(Op::I64Rotl,             0x89,   "i64_rotl",              Numeric,     2, 1, Some(Operation::I64Rotl)),
        facts(Op::I64Rotr,             0x8A,   "i64_rotr",              Numeric,     2, 1, Some(Operation::I64Rotr)),
        facts(Op::I64DivS,             0x7F,   "i64_div_s",             Numeric,     2, 1, Some(Operation::I64DivS)),
        facts(Op::I64DivU,             0x80,   "i64_div_u",             Numeric,     2, 1, Some(Operation::I64DivU)),
        facts(Op::I64RemS,             0x81,   "i64_rem_s",             Numeric,     2, 1, Some(Operation::I64RemS)),
        facts(Op::I64RemU,             0x82,   "i64_rem_u",             Numeric,     2, 1, Some(Operation::I64RemU)),
        facts(Op::I64Clz,              0x79,   "i64_clz",               Numeric,     1, 1, Some(Operation::I64Clz)),
        facts(Op::I64Ctz,              0x7A,   "i64_ctz",               Numeric,     1, 1, Some(Operation::I64Ctz)),
        facts(Op::I64Popcnt,           0x7B,   "i64_popcnt",            Numeric,     1, 1, Some(Operation::I64Popcnt)),
        facts(Op::Trap,                0xFFFE, "trap",                  System,      0, 0, None),
        facts(Op::Unsupported,         0xFF,   "unsupported",           Unknown,     0, 0, None),
    ]
};

const _: () = {
    let mut i = 0;
    while i < OPCODE_FACTS.len() {
        assert!(
            OPCODE_FACTS[i].opcode as usize == i,
            "OPCODE_FACTS rows must follow WasmOpcode order"
        );
        i += 1;
    }
    assert!(
        WasmOpcode::Trap as usize == 116,
        "supported() is the 116 rows before Trap"
    );
};

static OPCODE_BY_CODE: [WasmOpcode; 1 << 16] = {
    let mut by_code = [WasmOpcode::Unsupported; 1 << 16];
    let mut i = 0;
    while i < OPCODE_FACTS.len() {
        let code = OPCODE_FACTS[i].code as usize;
        assert!(
            by_code[code] as usize == WasmOpcode::Unsupported as usize,
            "duplicate opcode code"
        );
        by_code[code] = OPCODE_FACTS[i].opcode;
        i += 1;
    }
    by_code
};
