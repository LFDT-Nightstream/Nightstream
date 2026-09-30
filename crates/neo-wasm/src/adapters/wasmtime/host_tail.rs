//! Expands direct imported tail calls into the parser's proof-program Call/Return
//! pair. Wasmtime executes the original bytes; the synthetic return carries an
//! unobserved zero-advice result. Indirect host tails and operand discards are
//! unsupported.

use super::WasmtimeTraceStep;
use crate::host_event_bindings::{ExportTemplate, ImportTemplate, MemoryBase, SlotBinding};
use crate::{WasmBuildError, WasmOpcode, WasmPcEdgeKind};
use std::borrow::Cow;

/// The existing Return relation does not discard residual operands. Establish
/// the lowering's precondition from module validation, not the captured stack.
/// Only modules containing a direct imported tail call need this extra pass.
pub(super) fn validate_lowering(bytes: &[u8], import_params: &[u8]) -> Result<(), WasmBuildError> {
    let invalid = |error| WasmBuildError::Trace(format!("invalid host-tail module: {error}"));
    let mut validator = wasmparser::Validator::new();
    let mut allocations = wasmparser::FuncValidatorAllocations::default();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        let payload = payload.map_err(invalid)?;
        if let wasmparser::ValidPayload::Func(function, body) = validator.payload(&payload).map_err(invalid)? {
            let mut function = function.into_validator(allocations);
            function
                .read_locals(&mut body.get_binary_reader())
                .map_err(invalid)?;
            let mut operators = body.get_operators_reader().map_err(invalid)?;
            while !operators.eof() {
                let (operator, offset) = operators.read_with_offset().map_err(invalid)?;
                if let wasmparser::Operator::ReturnCall { function_index } = operator {
                    if let Some(&params) = import_params.get(function_index as usize) {
                        // Unreachable code has a polymorphic stack, not a runtime
                        // operand count. Ordinary validation below still applies.
                        let unreachable = function
                            .get_control_frame(0)
                            .is_some_and(|frame| frame.unreachable);
                        if !unreachable && function.operand_stack_height() != u32::from(params) {
                            return Err(WasmBuildError::Unsupported(
                                "direct host tail calls with residual operands are not supported".into(),
                            ));
                        }
                    }
                }
                function.op(offset, &operator).map_err(invalid)?;
            }
            allocations = function.into_allocations();
        }
    }
    Ok(())
}

pub(super) fn is_direct_host_tail(row: &WasmtimeTraceStep) -> bool {
    row.opcode_decoded == Some(WasmOpcode::ReturnCall) && !row.target_function_is_guest
}

pub(super) fn expand(rows: &[WasmtimeTraceStep]) -> Result<Cow<'_, [WasmtimeTraceStep]>, WasmBuildError> {
    if !rows.iter().any(is_direct_host_tail) {
        return Ok(Cow::Borrowed(rows));
    }
    let mut expanded = Vec::with_capacity(rows.len());
    for row in rows {
        expanded.push(row.clone());
        if !is_direct_host_tail(row) {
            continue;
        }
        if row.call_param_count.map(usize::from) != Some(row.operand_stack_words.len()) {
            return Err(WasmBuildError::Unsupported(
                "direct host tail calls with residual operands are not supported".into(),
            ));
        }
        let result_count = match row.call_result_count {
            Some(count @ 0..=1) => usize::from(count),
            _ => {
                return Err(WasmBuildError::Trace(
                    "host tail call requires zero or one result".into(),
                ))
            }
        };
        let synthetic_return_pc = row
            .call_return_pc
            .and_then(|pc| u32::try_from(pc).ok())
            .ok_or_else(|| WasmBuildError::Trace("host tail call is missing its parsed return PC".into()))?;
        // Wasmtime executes the original return_call, so no Return row is captured;
        // it is inserted right after the call. Same-instance reentry while the host
        // call is still running would therefore normalize as a later turn, ordered
        // after this turn's exit. Such reentry is not detected, including in
        // components. Callers must not normalize a failed invocation (see
        // `WasmtimeTraceRegistry`).
        expanded.push(WasmtimeTraceStep {
            step: row.step,
            function: row.function.clone(),
            function_index: row.function_index,
            current_function_ref: row.current_function_ref,
            pc: Some(synthetic_return_pc),
            pc_after_instruction: Some(u64::from(synthetic_return_pc) + 1),
            opcode_decoded: Some(WasmOpcode::Return),
            opcode: Some("Return (lowered host return_call)".into()),
            pc_edge_kind: Some(WasmPcEdgeKind::ReturnLike),
            operand_stack_words: vec![0; result_count],
            operand_stack_words_hi: vec![0; result_count],
            locals_words: row.locals_words.clone(),
            num_locals: row.num_locals,
            memory_pages_before: row.memory_pages_after,
            memory_pages_after: row.memory_pages_after,
            memory_max_pages: row.memory_max_pages,
            ..Default::default()
        });
    }
    Ok(Cow::Owned(expanded))
}

pub(super) fn validate_advice(import: &ImportTemplate, export: &ExportTemplate) -> Result<(), WasmBuildError> {
    if import.events.iter().any(|event| {
        event.absorb
            && event
                .block
                .iter()
                .any(|slot| matches!(slot, SlotBinding::ResultElem { .. }))
    }) {
        return Err(WasmBuildError::Unsupported(
            "terminal host-tail results cannot appear in absorbing events".into(),
        ));
    }
    if export
        .exit
        .iter()
        .flat_map(|event| &event.block)
        .any(|slot| {
            matches!(
                slot,
                SlotBinding::OutputElem { .. }
                    | SlotBinding::MemoryRead8 {
                        base: MemoryBase::Output,
                        ..
                    }
                    | SlotBinding::MemoryRead16 {
                        base: MemoryBase::Output,
                        ..
                    }
                    | SlotBinding::MemoryRead32 {
                        base: MemoryBase::Output,
                        ..
                    }
            )
        })
    {
        return Err(WasmBuildError::Unsupported(
            "terminal host-tail advice cannot supply an observed export output or output pointer".into(),
        ));
    }
    Ok(())
}
