//! WASM operation relations, independent of their proof implementation.
pub mod r1cs;
pub mod semantics;

pub const OPERATIONS_REGION: &str = "operations";

use crate::isa::WasmOperation;
use crate::layout::{
    Column, COL_OPERATION_ENABLED, COL_OPERATION_ID, COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO,
    COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO,
};
use semantics::OperationSemantics;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmOperationFamilySpec {
    pub name: &'static str,
    pub arity: WasmOperationArity,
    pub operation: WasmOperation,
    pub semantics: OperationSemantics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmOperationBindingSpec {
    pub name: &'static str,
    pub family: &'static str,
    pub columns: Vec<Column>,
    pub gate: Option<Column>,
    pub role: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmOperationArity {
    Unary,
    Binary,
    Tuple(u8),
}

impl WasmOperationArity {
    pub fn width(self) -> u8 {
        match self {
            Self::Unary => 1,
            Self::Binary => 2,
            Self::Tuple(width) => width,
        }
    }
}

pub(crate) fn families_and_bindings() -> (Vec<WasmOperationFamilySpec>, Vec<WasmOperationBindingSpec>) {
    let operation_families: Vec<WasmOperationFamilySpec> = WasmOperation::all()
        .into_iter()
        .map(|operation| WasmOperationFamilySpec {
            name: operation.name(),
            arity: match operation {
                WasmOperation::I32Clz | WasmOperation::I32Ctz | WasmOperation::I32Popcnt => WasmOperationArity::Unary,
                op if op.is_i64_binary() => WasmOperationArity::Tuple(4),
                op if op.is_i64_unary() => WasmOperationArity::Tuple(2),
                _ => WasmOperationArity::Binary,
            },
            operation,
            semantics: semantics::semantics_for_operation(operation),
        })
        .collect();

    let operation_bindings: Vec<WasmOperationBindingSpec> = WasmOperation::all()
        .into_iter()
        .map(|operation| WasmOperationBindingSpec {
            name: operation.name(),
            family: operation.name(),
            columns: match operation {
                op if op.is_i64_binary() => vec![
                    Column(COL_OPERATION_ID),
                    Column(COL_STACK_READ_VALUE_LO[0]),
                    Column(COL_STACK_READ_VALUE_HI[0]),
                    Column(COL_STACK_READ_VALUE_LO[1]),
                    Column(COL_STACK_READ_VALUE_HI[1]),
                    Column(COL_STACK_WRITE0_VALUE_LO),
                    Column(COL_STACK_WRITE0_VALUE_HI),
                ],
                op if op.is_i64_unary() => vec![
                    Column(COL_OPERATION_ID),
                    Column(COL_STACK_READ_VALUE_LO[0]),
                    Column(COL_STACK_READ_VALUE_HI[0]),
                    Column(COL_STACK_WRITE0_VALUE_LO),
                    Column(COL_STACK_WRITE0_VALUE_HI),
                ],
                _ => vec![
                    Column(COL_OPERATION_ID),
                    Column(COL_STACK_READ_VALUE_LO[0]),
                    Column(COL_STACK_READ_VALUE_LO[1]),
                    Column(COL_STACK_WRITE0_VALUE_LO),
                ],
            },
            gate: Some(Column(COL_OPERATION_ENABLED)),
            role: "operation row binding",
        })
        .collect();

    (operation_families, operation_bindings)
}
