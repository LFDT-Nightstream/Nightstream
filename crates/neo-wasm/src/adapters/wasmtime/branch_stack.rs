//! Static frame slides: for each branch edge, `return`, and function `end`, the operands
//! a taken edge keeps (`remaining`) and how far down it moves them (`distance`). Keys are
//! the same `(pc, control choice)` edges as the pc ROM; the parser assigns pcs.

use crate::ir::WasmProgramError;
use std::collections::BTreeMap;
use wasmparser::{
    FrameKind, FuncValidator, ModuleArity, Operator, Parser, ValidPayload, Validator, WasmModuleResources,
};

/// `(control choice, remaining, distance)` per edge.
pub(super) type FrameSlideEdges = Vec<(u64, u32, u32)>;

/// A slide that moves nothing needs no move rows.
fn slide(kept: u32, distance: u32) -> (u32, u32) {
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

/// A branch to the label at `depth` keeps the label's arity and drops the operands
/// between it and the label's height. `condition` operands are consumed by the row.
fn branch<T: WasmModuleResources>(
    validator: &FuncValidator<T>,
    depth: u32,
    condition: usize,
    offset: usize,
) -> Result<(u32, u32), WasmProgramError> {
    let (kept, height) = arity(validator, depth as usize, offset)?;
    let dropped = validator.operand_stack_height() as usize - condition - height - kept as usize;
    Ok(slide(kept, narrow(dropped, offset)?))
}

/// A return keeps the results and drops the rest of the frame, locals included.
fn function_return<T: WasmModuleResources>(
    validator: &FuncValidator<T>,
    height: usize,
    offset: usize,
) -> Result<(u32, u32), WasmProgramError> {
    let outermost = validator.control_stack_height() as usize - 1;
    let (results, _) = arity(validator, outermost, offset)?;
    let frame = validator.len_locals() as usize + height;
    Ok(slide(results, narrow(frame - results as usize, offset)?))
}

fn narrow(value: usize, offset: usize) -> Result<u32, WasmProgramError> {
    u32::try_from(value).map_err(|_| WasmProgramError::Unsupported(format!("frame slide too large at byte {offset}")))
}

/// Slides keyed by the operator's byte offset in `bytes`.
pub(super) fn frame_slides(bytes: &[u8]) -> Result<BTreeMap<u32, FrameSlideEdges>, WasmProgramError> {
    let invalid = |error| WasmProgramError::Invalid(format!("invalid wasm program: {error}"));
    let mut slides = BTreeMap::new();
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
                // Unreachable code executes no rows; a function's `end` always can.
                let reachable = function
                    .get_control_frame(0)
                    .is_some_and(|frame| !frame.unreachable);
                let height = function.operand_stack_height() as usize;
                let edges = match &operator {
                    Operator::End if function.control_stack_height() == 1 => {
                        let (results, _) = arity(&function, 0, offset)?;
                        let (kept, distance) = function_return(&function, results as usize, offset)?;
                        vec![(0, kept, distance)]
                    }
                    _ if !reachable => vec![],
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
                if !edges.is_empty() {
                    slides.insert(narrow(offset, offset)?, edges);
                }
                function.op(offset, &operator).map_err(invalid)?;
            }
            allocations = function.into_allocations();
        }
    }
    Ok(slides)
}
