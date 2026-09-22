//! Captures host-written buffers when the calling Wasm activation resumes.
//! Addresses come from pre-call arguments; bytes come from the return boundary.

use super::memory_address::memory_pointer;
use super::memory_inputs::{finish_inputs, memory_input_writes, recover_memory_inputs, MemoryInputWrite};
use super::{LoweringTables, WasmtimeTraceStep};
use crate::host_event_bindings::{ImportTemplate, MemoryBase};
use crate::{WasmBuildError, WasmOpcode};
use std::collections::BTreeMap;

#[derive(Debug)]
pub(super) struct PendingImport {
    pub(super) row: usize,
    // Host-to-guest activations, not guest call depth. Reentry adds an activation.
    pub(super) activation_depth: usize,
    pub(super) function_index: u32,
    pub(super) return_pc: u64,
    pub(super) memory_pages: Option<u32>,
    pub(super) writes: Result<Vec<MemoryInputWrite>, WasmBuildError>,
}

impl PendingImport {
    pub(super) fn from_call(
        row: &WasmtimeTraceStep,
        tables: &LoweringTables,
        activation_depth: usize,
    ) -> Result<Option<Self>, WasmBuildError> {
        if !matches!(row.opcode_decoded, Some(WasmOpcode::Call | WasmOpcode::CallIndirect))
            || row.target_function_is_guest
        {
            return Ok(None);
        }
        let Some(template) = row
            .function_ref
            .and_then(|fref| tables.artifacts.host_event_bindings.imports.get(&fref))
        else {
            return Ok(None);
        };
        if template.input_count == 0 {
            return Ok(None);
        }
        let function_index = row.function_index.ok_or_else(|| {
            WasmBuildError::Trace(format!(
                "missing caller function index for host call at step {}",
                row.step
            ))
        })?;
        let return_pc = row
            .call_return_pc
            .ok_or_else(|| WasmBuildError::Trace(format!("missing return PC for host call at step {}", row.step)))?;
        let writes = (|| {
            let count = row
                .call_param_count
                .ok_or_else(|| WasmBuildError::Trace("missing host call parameter count".into()))?;
            let index = usize::from(row.opcode_decoded == Some(WasmOpcode::CallIndirect));
            let start = row
                .operand_stack_words
                .len()
                .checked_sub(usize::from(count) + index)
                .ok_or_else(|| WasmBuildError::Trace("operand stack underflow collecting host call args".into()))?;
            let args: Vec<_> = (start..start + usize::from(count))
                .map(|i| {
                    (
                        row.operand_stack_words[i],
                        row.operand_stack_words_hi.get(i).copied().unwrap_or(0),
                    )
                })
                .collect();
            import_memory_writes(template, &args, row.memory_pages_before)
        })();
        Ok(Some(Self {
            row: row.step as usize,
            activation_depth,
            function_index,
            return_pc,
            memory_pages: row.memory_pages_before,
            writes,
        }))
    }
}

fn import_memory_writes(
    template: &ImportTemplate,
    args: &[(u32, u32)],
    pages: Option<u32>,
) -> Result<Vec<MemoryInputWrite>, WasmBuildError> {
    memory_input_writes(
        &template.events,
        |base| {
            let MemoryBase::Arg(arg) = base else {
                return Err(WasmBuildError::Trace(
                    "import memory writes require an argument base".into(),
                ));
            };
            memory_pointer(args, arg, "import argument")
        },
        pages,
        "import return",
    )
}

pub(super) fn recover_import_inputs(
    template: &ImportTemplate,
    args: &[(u32, u32)],
    pages: Option<u32>,
    captured: Option<&Result<BTreeMap<u32, u8>, String>>,
) -> Result<Vec<u64>, WasmBuildError> {
    let writes = import_memory_writes(template, args, pages)?;
    let mut inputs = vec![None; usize::from(template.input_count)];
    recover_memory_inputs(&mut inputs, &writes, captured, "import return")?;
    finish_inputs(inputs, "import return")
}
