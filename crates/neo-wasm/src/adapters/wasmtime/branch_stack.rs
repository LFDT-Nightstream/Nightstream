use super::decode::OperandLane;
use crate::ir::WasmProgramError;
use std::collections::BTreeMap;
use wasmparser::{
    FrameKind, FuncValidator, ModuleArity, Operator, Parser, ValidPayload, Validator, WasmModuleResources,
};

pub(super) type FrameSlideEdges = Vec<(u64, u32, u32)>;

pub(super) struct OperatorStack {
    pub(super) height: u32,
    pub(super) frame_operands: Vec<Option<OperandLane>>,
    pub(super) slides: FrameSlideEdges,
}

fn canonical_slide(kept: u32, distance: u32) -> (u32, u32) {
    if distance == 0 {
        (0, 0)
    } else {
        (kept, distance)
    }
}

fn arity<T: WasmModuleResources>(
    validator: &FuncValidator<T>,
    depth: usize,
    offset: usize,
) -> Result<(u32, usize), WasmProgramError> {
    let target = validator
        .get_control_frame(depth)
        .ok_or_else(|| WasmProgramError::Invalid(format!("invalid branch depth at byte {offset}")))?;
    let (params, results) = validator
        .block_type_arity(target.block_type)
        .ok_or_else(|| WasmProgramError::Invalid(format!("invalid branch target type at byte {offset}")))?;
    let kept = if target.kind == FrameKind::Loop {
        params
    } else {
        results
    };
    Ok((kept, target.height))
}

fn branch<T: WasmModuleResources>(
    validator: &FuncValidator<T>,
    depth: u32,
    condition_operands: usize,
    offset: usize,
) -> Result<(u32, u32), WasmProgramError> {
    let (kept, height) = arity(validator, depth as usize, offset)?;
    let dropped = (validator.operand_stack_height() as usize)
        .checked_sub(condition_operands)
        .and_then(|available| available.checked_sub(height))
        .and_then(|available| available.checked_sub(kept as usize))
        .ok_or_else(|| WasmProgramError::Invalid(format!("branch operand stack underflow at byte {offset}")))?;
    Ok(canonical_slide(kept, narrow(dropped, offset)?))
}

fn function_return<T: WasmModuleResources>(
    validator: &FuncValidator<T>,
    height: usize,
    offset: usize,
) -> Result<(u32, u32), WasmProgramError> {
    let outermost = validator.control_stack_height() as usize - 1;
    let (results, _) = arity(validator, outermost, offset)?;
    let frame = validator.len_locals() as usize + height;
    let dropped = frame
        .checked_sub(results as usize)
        .ok_or_else(|| WasmProgramError::Invalid(format!("return operand stack underflow at byte {offset}")))?;
    Ok(canonical_slide(results, narrow(dropped, offset)?))
}

fn narrow(value: usize, offset: usize) -> Result<u32, WasmProgramError> {
    u32::try_from(value).map_err(|_| WasmProgramError::Unsupported(format!("frame slide too large at byte {offset}")))
}

pub(super) fn operator_stacks_by_offset(bytes: &[u8]) -> Result<BTreeMap<u32, OperatorStack>, WasmProgramError> {
    let invalid = |error| WasmProgramError::Invalid(format!("invalid wasm program: {error}"));
    let mut stacks = BTreeMap::new();
    let mut validator = Validator::new();
    let mut allocations = Default::default();
    for payload in Parser::new(0).parse_all(bytes) {
        let payload = payload.map_err(invalid)?;
        if let ValidPayload::Func(function, body) = validator.payload(&payload).map_err(invalid)? {
            let mut function = function.into_validator(allocations);
            function
                .read_locals(&mut body.get_binary_reader())
                .map_err(invalid)?;
            let mut operators = body.get_operators_reader().map_err(invalid)?;
            while !operators.eof() {
                let offset = operators.original_position();
                let operator = operators.read().map_err(invalid)?;
                let frame = function.get_control_frame(0).ok_or_else(|| {
                    WasmProgramError::Invalid(format!("operator outside a function at byte {offset}"))
                })?;
                let reachable = !frame.unreachable;
                let height = function.operand_stack_height() as usize;
                let frame_operands = (0..height.checked_sub(frame.height).ok_or_else(|| {
                    WasmProgramError::Invalid(format!("operand stack below its control frame at byte {offset}"))
                })?)
                    .rev()
                    .map(|depth| {
                        function
                            .get_operand_type(depth)
                            .map(|ty| ty.map(OperandLane::of_type))
                            .ok_or_else(|| WasmProgramError::Invalid(format!("missing operand type at byte {offset}")))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let edges = match &operator {
                    Operator::End if function.control_stack_height() == 1 => {
                        let (results, _) = arity(&function, 0, offset)?;
                        let (kept, distance) = function_return(&function, results as usize, offset)?;
                        vec![(0, kept, distance)]
                    }
                    _ if !reachable => vec![],
                    // A `br` to the body's own label is a `return`.
                    Operator::Br { relative_depth }
                        if *relative_depth as usize == function.control_stack_height() as usize - 1 =>
                    {
                        let (kept, distance) = function_return(&function, height, offset)?;
                        vec![(0, kept, distance)]
                    }
                    Operator::Br { relative_depth } => {
                        let (kept, distance) = branch(&function, *relative_depth, 0, offset)?;
                        vec![(0, kept, distance)]
                    }
                    Operator::BrIf { relative_depth } => {
                        let (kept, distance) = branch(&function, *relative_depth, 1, offset)?;
                        vec![(0, 0, 0), (1, kept, distance)]
                    }
                    Operator::BrTable { targets } => {
                        let mut edges = Vec::new();
                        for (arm, target) in targets.targets().enumerate() {
                            let (kept, distance) = branch(&function, target.map_err(invalid)?, 1, offset)?;
                            edges.push((arm as u64 + 1, kept, distance));
                        }
                        let (kept, distance) = branch(&function, targets.default(), 1, offset)?;
                        edges.push((0, kept, distance));
                        edges
                    }
                    Operator::Return => {
                        let (kept, distance) = function_return(&function, height, offset)?;
                        vec![(0, kept, distance)]
                    }
                    _ => vec![],
                };
                stacks.insert(
                    narrow(offset, offset)?,
                    OperatorStack {
                        height: narrow(height, offset)?,
                        frame_operands,
                        slides: edges,
                    },
                );
                function.op(offset, &operator).map_err(invalid)?;
            }
            let end_offset = operators.original_position();
            operators
                .get_binary_reader()
                .finish_expression(&function.visitor(end_offset))
                .map_err(invalid)?;
            allocations = function.into_allocations();
        }
    }
    Ok(stacks)
}
