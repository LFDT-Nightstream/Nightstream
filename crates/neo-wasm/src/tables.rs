use super::ir::WasmStepTrace;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmLookupArity {
    Unary,
    Binary,
    Tuple(u8),
}

impl WasmLookupArity {
    pub fn width(self) -> u8 {
        match self {
            Self::Unary => 1,
            Self::Binary => 2,
            Self::Tuple(width) => width,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WasmLookupPayload {
    pub arity: WasmLookupArity,
    pub op_table_id: u32,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
}

pub fn lookup_payload(trace: &WasmStepTrace) -> Option<WasmLookupPayload> {
    let op_table = trace.info.op_table?;
    Some(match op_table {
        super::isa::WasmOpTable::I64And
        | super::isa::WasmOpTable::I64Or
        | super::isa::WasmOpTable::I64Xor
        | super::isa::WasmOpTable::I64Mul => WasmLookupPayload {
            arity: WasmLookupArity::Tuple(4),
            op_table_id: op_table.op_table_id(),
            inputs: vec![
                trace.stack_read0.map(|lane| lane.value_lo).unwrap_or(0),
                trace
                    .stack_read0
                    .and_then(|lane| lane.value_hi)
                    .unwrap_or(0),
                trace.stack_read1.map(|lane| lane.value_lo).unwrap_or(0),
                trace
                    .stack_read1
                    .and_then(|lane| lane.value_hi)
                    .unwrap_or(0),
            ],
            outputs: vec![
                trace.stack_write0.map(|lane| lane.value_lo).unwrap_or(0),
                trace
                    .stack_write0
                    .and_then(|lane| lane.value_hi)
                    .unwrap_or(0),
            ],
        },
        _ => match trace.info.stack_reads {
            1 => WasmLookupPayload {
                arity: WasmLookupArity::Unary,
                op_table_id: op_table.op_table_id(),
                inputs: vec![trace.stack_read0.map(|lane| lane.value_lo).unwrap_or(0)],
                outputs: vec![trace.stack_write0.map(|lane| lane.value_lo).unwrap_or(0)],
            },
            2 => WasmLookupPayload {
                arity: WasmLookupArity::Binary,
                op_table_id: op_table.op_table_id(),
                inputs: vec![
                    trace.stack_read0.map(|lane| lane.value_lo).unwrap_or(0),
                    trace.stack_read1.map(|lane| lane.value_lo).unwrap_or(0),
                ],
                outputs: vec![trace.stack_write0.map(|lane| lane.value_lo).unwrap_or(0)],
            },
            _ => return None,
        },
    })
}
