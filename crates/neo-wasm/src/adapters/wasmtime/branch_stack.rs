//! Reject branch stack unwinding until the circuit binds the discarded count
//! and relocates carried results. PC-ROM edges alone only constrain the jump.

use crate::ir::WasmBuildError;
use wasmparser::{
    FrameKind, FuncValidator, ModuleArity, Operator, Parser, ValidPayload, Validator, WasmModuleResources,
};

fn validate_target<T: WasmModuleResources>(
    validator: &FuncValidator<T>,
    depth: u32,
    condition_operands: usize,
    offset: usize,
) -> Result<(), WasmBuildError> {
    // Polymorphic stacks in unreachable code do not describe an executed row.
    if validator
        .get_control_frame(0)
        .is_some_and(|frame| frame.unreachable)
    {
        return Ok(());
    }
    let target = validator
        .get_control_frame(depth as usize)
        .ok_or_else(|| WasmBuildError::Trace(format!("invalid branch depth at byte {offset}")))?;
    let (params, results) = validator
        .block_type_arity(target.block_type)
        .ok_or_else(|| WasmBuildError::Trace(format!("invalid branch target type at byte {offset}")))?;
    let carried = if target.kind == FrameKind::Loop {
        params
    } else {
        results
    };
    let kept = target.height + carried as usize + condition_operands;
    if validator.operand_stack_height() as usize > kept {
        return Err(WasmBuildError::Unsupported(format!(
            "branch stack unwinding at byte {offset}: discarding temporary operands is not constrained by the circuit"
        )));
    }
    Ok(())
}

pub(super) fn validate_branch_stack_support(bytes: &[u8]) -> Result<(), WasmBuildError> {
    let invalid = |error| WasmBuildError::Trace(format!("invalid wasm program: {error}"));
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
                match &operator {
                    Operator::Br { relative_depth } => validate_target(&function, *relative_depth, 0, offset)?,
                    Operator::BrIf { relative_depth } => validate_target(&function, *relative_depth, 1, offset)?,
                    Operator::BrTable { targets } => {
                        for target in targets.targets() {
                            validate_target(&function, target.map_err(invalid)?, 1, offset)?;
                        }
                        validate_target(&function, targets.default(), 1, offset)?;
                    }
                    _ => {}
                }
                function.op(offset, &operator).map_err(invalid)?;
            }
            allocations = function.into_allocations();
        }
    }
    Ok(())
}
