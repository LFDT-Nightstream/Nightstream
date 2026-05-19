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
    pub shout_id: u32,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
}

pub fn lookup_payload(trace: &WasmStepTrace) -> Option<WasmLookupPayload> {
    let shout = trace.info.shout_opcode?;
    Some(match shout {
        super::isa::WasmShoutOpcode::I64Eqz => WasmLookupPayload {
            arity: WasmLookupArity::Binary,
            shout_id: shout.to_shout_id(),
            inputs: vec![
                trace.stack_read0.map(|lane| lane.value).unwrap_or(0),
                trace.stack_read0_hi.unwrap_or(0),
            ],
            outputs: vec![trace.stack_write0.map(|lane| lane.value).unwrap_or(0)],
        },
        super::isa::WasmShoutOpcode::I64And
        | super::isa::WasmShoutOpcode::I64Or
        | super::isa::WasmShoutOpcode::I64Xor
        | super::isa::WasmShoutOpcode::I64Mul => WasmLookupPayload {
            arity: WasmLookupArity::Tuple(4),
            shout_id: shout.to_shout_id(),
            inputs: vec![
                trace.stack_read0.map(|lane| lane.value).unwrap_or(0),
                trace.stack_read0_hi.unwrap_or(0),
                trace.stack_read1.map(|lane| lane.value).unwrap_or(0),
                trace.stack_read1_hi.unwrap_or(0),
            ],
            outputs: vec![
                trace.stack_write0.map(|lane| lane.value).unwrap_or(0),
                trace.stack_write0_hi.unwrap_or(0),
            ],
        },
        _ => match trace.info.stack_reads {
            1 => WasmLookupPayload {
                arity: WasmLookupArity::Unary,
                shout_id: shout.to_shout_id(),
                inputs: vec![trace.stack_read0.map(|lane| lane.value).unwrap_or(0)],
                outputs: vec![trace.stack_write0.map(|lane| lane.value).unwrap_or(0)],
            },
            2 => WasmLookupPayload {
                arity: WasmLookupArity::Binary,
                shout_id: shout.to_shout_id(),
                inputs: vec![
                    trace.stack_read0.map(|lane| lane.value).unwrap_or(0),
                    trace.stack_read1.map(|lane| lane.value).unwrap_or(0),
                ],
                outputs: vec![trace.stack_write0.map(|lane| lane.value).unwrap_or(0)],
            },
            _ => return None,
        },
    })
}
