//! Structural decoding of `wasmparser::Operator`s into the adapter's
//! intermediate opcode / memory / control descriptors.
//!
//! Pure classification: given a parsed operator, produce the metadata the
//! parent module records in its static opcode map. Owns no Wasmtime `Store`
//! access and builds no trace rows — that lives in the parent module and
//! `normalize`.

use crate::ir::WasmPcEdgeKind;
use crate::isa::WasmOpcode;

#[derive(Clone, Debug)]
pub(crate) struct DecodedOpcode {
    pub(crate) text: String,
    pub(crate) memory: Option<DecodedMemoryOpcode>,
    /// Structurally decoded from `wasmparser::Operator` at map-build time.
    pub(crate) decoded: Option<(WasmOpcode, Option<u32>)>,
    pub(crate) control: Option<DecodedControlOpcode>,
    pub(crate) pc_edge_kind: WasmPcEdgeKind,
    pub(crate) call_indirect_type_index: Option<u32>,
    pub(crate) expected_type_id: Option<u32>,
    /// For `call` instructions: binary offset of the instruction after the call = return address.
    pub(crate) call_return_pc: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DecodedMemoryOpcode {
    pub(crate) kind: DecodedMemoryAccessKind,
    pub(crate) memory_index: u32,
    pub(crate) offset: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DecodedMemoryAccessKind {
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
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DecodedControlOpcode {
    BrTable { len: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ControlFrameKind {
    Block,
    Loop,
    If,
}

#[derive(Clone, Debug)]
pub(crate) struct ControlFrame {
    pub(crate) kind: ControlFrameKind,
    pub(crate) branch_target: Option<u64>,
    pub(crate) pending_to_end: Vec<u64>,
    pub(crate) pending_if_false: Option<u64>,
}

impl ControlFrame {
    pub(crate) fn new(kind: ControlFrameKind, branch_target: Option<u64>) -> Self {
        Self {
            kind,
            branch_target,
            pending_to_end: Vec::new(),
            pending_if_false: None,
        }
    }
}

impl DecodedMemoryAccessKind {
    pub(crate) fn width_bytes(self) -> u8 {
        match self {
            Self::I32Load | Self::I32Store | Self::I64Store32 | Self::I64Load32U | Self::I64Load32S => 4,
            Self::I64Load | Self::I64Store => 8,
            Self::I32Load8S
            | Self::I32Load8U
            | Self::I32Store8
            | Self::I64Store8
            | Self::I64Load8U
            | Self::I64Load8S => 1,
            Self::I32Load16S
            | Self::I32Load16U
            | Self::I32Store16
            | Self::I64Store16
            | Self::I64Load16U
            | Self::I64Load16S => 2,
        }
    }

    pub(crate) fn is_store(self) -> bool {
        matches!(
            self,
            Self::I32Store
                | Self::I32Store8
                | Self::I32Store16
                | Self::I64Store
                | Self::I64Store8
                | Self::I64Store16
                | Self::I64Store32
        )
    }
}

pub(crate) fn decode_opcode(operator: &wasmparser::Operator<'_>) -> Option<(WasmOpcode, Option<u32>)> {
    match operator {
        wasmparser::Operator::Nop => Some((WasmOpcode::Nop, None)),
        wasmparser::Operator::I32Const { value } => Some((WasmOpcode::I32Const, Some(*value as u32))),
        wasmparser::Operator::I64Const { .. } => Some((WasmOpcode::I64Const, None)),
        wasmparser::Operator::RefFunc { function_index } => {
            Some((WasmOpcode::RefFunc, Some(function_index.saturating_add(1))))
        }
        wasmparser::Operator::I32Add => Some((WasmOpcode::I32Add, None)),
        wasmparser::Operator::I64Add => Some((WasmOpcode::I64Add, None)),
        wasmparser::Operator::I32Sub => Some((WasmOpcode::I32Sub, None)),
        wasmparser::Operator::I64Sub => Some((WasmOpcode::I64Sub, None)),
        wasmparser::Operator::I64Load { .. } => Some((WasmOpcode::I64Load, None)),
        wasmparser::Operator::I64Store { .. } => Some((WasmOpcode::I64Store, None)),
        wasmparser::Operator::I64Store8 { .. } => Some((WasmOpcode::I64Store8, None)),
        wasmparser::Operator::I64Store16 { .. } => Some((WasmOpcode::I64Store16, None)),
        wasmparser::Operator::I64Store32 { .. } => Some((WasmOpcode::I64Store32, None)),
        wasmparser::Operator::I64Load8U { .. } => Some((WasmOpcode::I64Load8U, None)),
        wasmparser::Operator::I64Load16U { .. } => Some((WasmOpcode::I64Load16U, None)),
        wasmparser::Operator::I64Load32U { .. } => Some((WasmOpcode::I64Load32U, None)),
        wasmparser::Operator::I64Load8S { .. } => Some((WasmOpcode::I64Load8S, None)),
        wasmparser::Operator::I64Load16S { .. } => Some((WasmOpcode::I64Load16S, None)),
        wasmparser::Operator::I64Load32S { .. } => Some((WasmOpcode::I64Load32S, None)),
        wasmparser::Operator::I32WrapI64 => Some((WasmOpcode::I32WrapI64, None)),
        wasmparser::Operator::I64And => Some((WasmOpcode::I64And, None)),
        wasmparser::Operator::I64Or => Some((WasmOpcode::I64Or, None)),
        wasmparser::Operator::I64Xor => Some((WasmOpcode::I64Xor, None)),
        wasmparser::Operator::I64Mul => Some((WasmOpcode::I64Mul, None)),
        wasmparser::Operator::I32Load { .. } => Some((WasmOpcode::I32Load, None)),
        wasmparser::Operator::I32Load8S { .. } => Some((WasmOpcode::I32Load8S, None)),
        wasmparser::Operator::I32Load8U { .. } => Some((WasmOpcode::I32Load8U, None)),
        wasmparser::Operator::I32Load16S { .. } => Some((WasmOpcode::I32Load16S, None)),
        wasmparser::Operator::I32Load16U { .. } => Some((WasmOpcode::I32Load16U, None)),
        wasmparser::Operator::I32Store { .. } => Some((WasmOpcode::I32Store, None)),
        wasmparser::Operator::I32Store8 { .. } => Some((WasmOpcode::I32Store8, None)),
        wasmparser::Operator::I32Store16 { .. } => Some((WasmOpcode::I32Store16, None)),
        wasmparser::Operator::MemorySize { .. } => Some((WasmOpcode::MemorySize, None)),
        wasmparser::Operator::MemoryGrow { .. } => Some((WasmOpcode::MemoryGrow, None)),
        wasmparser::Operator::TableSize { table } => Some((WasmOpcode::TableSize, Some(*table))),
        wasmparser::Operator::TableGet { table } => Some((WasmOpcode::TableGet, Some(*table))),
        wasmparser::Operator::TableSet { table } => Some((WasmOpcode::TableSet, Some(*table))),
        wasmparser::Operator::Drop => Some((WasmOpcode::Drop, None)),
        wasmparser::Operator::Br { .. } => Some((WasmOpcode::Br, None)),
        wasmparser::Operator::Block { .. } => Some((WasmOpcode::Block, None)),
        wasmparser::Operator::Loop { .. } => Some((WasmOpcode::Loop, None)),
        wasmparser::Operator::If { .. } => Some((WasmOpcode::If, None)),
        wasmparser::Operator::Else => Some((WasmOpcode::Else, None)),
        wasmparser::Operator::Unreachable => Some((WasmOpcode::Unreachable, None)),
        wasmparser::Operator::I32Clz => Some((WasmOpcode::I32Clz, None)),
        wasmparser::Operator::I32Ctz => Some((WasmOpcode::I32Ctz, None)),
        wasmparser::Operator::I32Popcnt => Some((WasmOpcode::I32Popcnt, None)),
        wasmparser::Operator::I32Eqz => Some((WasmOpcode::I32Eqz, None)),
        wasmparser::Operator::I64Eqz => Some((WasmOpcode::I64Eqz, None)),
        wasmparser::Operator::I32Eq => Some((WasmOpcode::I32Eq, None)),
        wasmparser::Operator::I32Ne => Some((WasmOpcode::I32Ne, None)),
        wasmparser::Operator::I64Eq => Some((WasmOpcode::I64Eq, None)),
        wasmparser::Operator::I64Ne => Some((WasmOpcode::I64Ne, None)),
        wasmparser::Operator::I32LtS => Some((WasmOpcode::I32LtS, None)),
        wasmparser::Operator::I32LtU => Some((WasmOpcode::I32LtU, None)),
        wasmparser::Operator::I32GtS => Some((WasmOpcode::I32GtS, None)),
        wasmparser::Operator::I32GtU => Some((WasmOpcode::I32GtU, None)),
        wasmparser::Operator::I32LeS => Some((WasmOpcode::I32LeS, None)),
        wasmparser::Operator::I32LeU => Some((WasmOpcode::I32LeU, None)),
        wasmparser::Operator::I32GeS => Some((WasmOpcode::I32GeS, None)),
        wasmparser::Operator::I32GeU => Some((WasmOpcode::I32GeU, None)),
        wasmparser::Operator::I32And => Some((WasmOpcode::I32And, None)),
        wasmparser::Operator::I32Or => Some((WasmOpcode::I32Or, None)),
        wasmparser::Operator::I32Xor => Some((WasmOpcode::I32Xor, None)),
        wasmparser::Operator::I32Mul => Some((WasmOpcode::I32Mul, None)),
        wasmparser::Operator::I32Shl => Some((WasmOpcode::I32Shl, None)),
        wasmparser::Operator::I32ShrU => Some((WasmOpcode::I32ShrU, None)),
        wasmparser::Operator::I32ShrS => Some((WasmOpcode::I32ShrS, None)),
        wasmparser::Operator::I32Rotl => Some((WasmOpcode::I32Rotl, None)),
        wasmparser::Operator::I32Rotr => Some((WasmOpcode::I32Rotr, None)),
        wasmparser::Operator::I32DivU => Some((WasmOpcode::I32DivU, None)),
        wasmparser::Operator::I32DivS => Some((WasmOpcode::I32DivS, None)),
        wasmparser::Operator::I32RemU => Some((WasmOpcode::I32RemU, None)),
        wasmparser::Operator::I32RemS => Some((WasmOpcode::I32RemS, None)),
        wasmparser::Operator::Select => Some((WasmOpcode::Select, None)),
        wasmparser::Operator::TypedSelect { .. } => Some((WasmOpcode::Select, None)),
        wasmparser::Operator::BrIf { .. } => Some((WasmOpcode::BrIf, None)),
        wasmparser::Operator::BrTable { .. } => Some((WasmOpcode::BrTable, None)),
        wasmparser::Operator::Return => Some((WasmOpcode::Return, None)),
        // Local index stored as immediate; locals value captured at runtime from frame snapshot.
        wasmparser::Operator::LocalGet { local_index } => Some((WasmOpcode::LocalGet, Some(*local_index))),
        wasmparser::Operator::LocalSet { local_index } => Some((WasmOpcode::LocalSet, Some(*local_index))),
        wasmparser::Operator::LocalTee { local_index } => Some((WasmOpcode::LocalTee, Some(*local_index))),
        wasmparser::Operator::GlobalGet { global_index } => Some((WasmOpcode::GlobalGet, Some(*global_index))),
        wasmparser::Operator::GlobalSet { global_index } => Some((WasmOpcode::GlobalSet, Some(*global_index))),
        wasmparser::Operator::Call { function_index } => Some((WasmOpcode::Call, Some(*function_index))),
        wasmparser::Operator::CallIndirect { table_index, .. } => Some((WasmOpcode::CallIndirect, Some(*table_index))),
        _ => None,
    }
}

pub(crate) fn decode_control_opcode(operator: &wasmparser::Operator<'_>) -> Option<DecodedControlOpcode> {
    match operator {
        wasmparser::Operator::BrTable { targets } => Some(DecodedControlOpcode::BrTable { len: targets.len() }),
        _ => None,
    }
}

pub(crate) fn decode_memory_opcode(operator: &wasmparser::Operator<'_>) -> Option<DecodedMemoryOpcode> {
    match operator {
        wasmparser::Operator::I32Load { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Load,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Load8U { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Load8U,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Load8S { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Load8S,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Load16S { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Load16S,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Load16U { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Load16U,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Load { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Load,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Store { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Store,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Store8 { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Store8,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Store16 { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Store16,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Store { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Store,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Store8 { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Store8,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Store16 { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Store16,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Store32 { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Store32,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Load8U { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Load8U,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Load16U { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Load16U,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Load32U { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Load32U,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Load8S { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Load8S,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Load16S { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Load16S,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Load32S { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Load32S,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        _ => None,
    }
}
