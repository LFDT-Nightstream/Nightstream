use super::decode::OperandLane;
use crate::ir::WasmProgramError;
use std::collections::BTreeMap;
use wasmparser::{FrameKind, FuncValidator, FunctionBody, ModuleArity, Operator, WasmModuleResources};

pub(super) type FrameSlideEdges = Vec<(u64, u32, u32)>;

pub(super) struct OperatorStack {
    pub(super) height: u32,
    pub(super) frame_operands: FrameOperands,
    pub(super) slides: FrameSlideEdges,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameOperands {
    top: usize,
    len: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OperandTypes {
    nodes: Vec<(usize, Option<OperandLane>)>,
}

impl OperandTypes {
    fn pop(&self, top: &mut usize) -> Option<OperandLane> {
        let (parent, lane) = self.nodes[*top - 1];
        *top = parent;
        lane
    }

    pub(super) fn lanes(&self, frame: FrameOperands) -> Vec<Option<OperandLane>> {
        let mut top = frame.top;
        let mut lanes: Vec<_> = (0..frame.len).map(|_| self.pop(&mut top)).collect();
        lanes.reverse();
        lanes
    }

    fn update<T: WasmModuleResources>(
        &mut self,
        mut top: usize,
        height: usize,
        outputs: u32,
        validator: &FuncValidator<T>,
        offset: usize,
    ) -> Result<usize, WasmProgramError> {
        let next_height = validator.operand_stack_height() as usize;
        let prefix = next_height.saturating_sub(outputs as usize);
        for _ in prefix..height {
            self.pop(&mut top);
        }
        for depth in (0..next_height - prefix).rev() {
            let ty = validator
                .get_operand_type(depth)
                .ok_or_else(|| WasmProgramError::Invalid(format!("missing operand type at byte {offset}")))?;
            self.nodes.push((top, ty.map(OperandLane::of_type)));
            top = self.nodes.len();
        }
        Ok(top)
    }
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

pub(super) fn operator_stacks_by_offset<T: WasmModuleResources>(
    function: &mut FuncValidator<T>,
    body: &FunctionBody<'_>,
    operand_types: &mut OperandTypes,
) -> Result<BTreeMap<u32, OperatorStack>, WasmProgramError> {
    let invalid = |error| WasmProgramError::Invalid(format!("invalid wasm program: {error}"));
    let mut stacks = BTreeMap::new();
    function
        .read_locals(&mut body.get_binary_reader())
        .map_err(invalid)?;
    let mut operators = body.get_operators_reader().map_err(invalid)?;
    let mut top = 0;
    while !operators.eof() {
        let offset = operators.original_position();
        let operator = operators.read().map_err(invalid)?;
        let frame = function
            .get_control_frame(0)
            .ok_or_else(|| WasmProgramError::Invalid(format!("operator outside a function at byte {offset}")))?;
        let reachable = !frame.unreachable;
        let height = function.operand_stack_height() as usize;
        let frame_operands = FrameOperands {
            top,
            len: height.checked_sub(frame.height).ok_or_else(|| {
                WasmProgramError::Invalid(format!("operand stack below its control frame at byte {offset}"))
            })?,
        };
        let edges = match &operator {
            Operator::End if function.control_stack_height() == 1 => {
                let (results, _) = arity(function, 0, offset)?;
                let (kept, distance) = function_return(function, results as usize, offset)?;
                vec![(0, kept, distance)]
            }
            _ if !reachable => vec![],
            // A `br` to the body's own label is a `return`.
            Operator::Br { relative_depth }
                if *relative_depth as usize == function.control_stack_height() as usize - 1 =>
            {
                let (kept, distance) = function_return(function, height, offset)?;
                vec![(0, kept, distance)]
            }
            Operator::Br { relative_depth } => {
                let (kept, distance) = branch(function, *relative_depth, 0, offset)?;
                vec![(0, kept, distance)]
            }
            Operator::BrIf { relative_depth } => {
                let (kept, distance) = branch(function, *relative_depth, 1, offset)?;
                vec![(0, 0, 0), (1, kept, distance)]
            }
            Operator::BrTable { targets } => {
                let mut edges = Vec::new();
                for (arm, target) in targets.targets().enumerate() {
                    let (kept, distance) = branch(function, target.map_err(invalid)?, 1, offset)?;
                    edges.push((arm as u64 + 1, kept, distance));
                }
                let (kept, distance) = branch(function, targets.default(), 1, offset)?;
                edges.push((0, kept, distance));
                edges
            }
            Operator::Return => {
                let (kept, distance) = function_return(function, height, offset)?;
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
        let arity = operator.operator_arity(function);
        function.op(offset, &operator).map_err(invalid)?;
        let (_, outputs) =
            arity.ok_or_else(|| WasmProgramError::Unsupported(format!("unknown operator arity at byte {offset}")))?;
        top = operand_types.update(top, height, outputs, function, offset)?;
    }
    let end_offset = operators.original_position();
    operators
        .get_binary_reader()
        .finish_expression(&function.visitor(end_offset))
        .map_err(invalid)?;
    Ok(stacks)
}

#[cfg(test)]
#[path = "../../../tests/unit/operand_types.rs"]
mod tests;
