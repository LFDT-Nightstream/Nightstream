use super::super::super::entry_inputs::recover_entry_inputs;
use super::super::host_event_emit::{
    apply_export_entry_memory, plan_export_blocks, read_export_exit_memory, EventBlockPlan,
};
use super::super::memory::LinearMemoryImage;
use super::super::NormalizedStep;
use crate::host_event_bindings::{HostEventBindings, MemoryBase, SlotBinding};
use crate::ir::{function_call_metadata_shape, WasmBuildError};

pub(super) struct TurnSetup<'g> {
    pub(super) fref: u32,
    pub(super) param_count: u8,
    pub(super) result_count: u8,
    pub(super) template: &'g crate::host_event_bindings::ExportTemplate,
    pub(super) entry_plans: Vec<EventBlockPlan>,
}

pub(super) fn setup_turn<'g>(
    bindings: &'g HostEventBindings,
    first: &NormalizedStep,
    program: Option<&super::super::super::WasmProgramTables>,
    re_entered: bool,
    memory: &mut LinearMemoryImage,
) -> Result<TurnSetup<'g>, WasmBuildError> {
    let fref = first.current_function_ref.unwrap_or(0);
    let template = bindings.exports.get(&fref).ok_or_else(|| {
        WasmBuildError::Trace(format!(
            "host-event bindings require an export template for the invoked export (fref {fref})"
        ))
    })?;
    let (param_count, result_count) = if let Some(program) = program {
        let metadata = program
            .function_call_metadata
            .iter()
            .find_map(|&(function_ref, metadata)| (function_ref == u64::from(fref)).then_some(metadata))
            .ok_or_else(|| WasmBuildError::Trace(format!("missing call metadata for export fref {fref}")))?;
        let (params, results, is_guest) = function_call_metadata_shape(metadata);
        if !is_guest {
            return Err(WasmBuildError::Trace(format!(
                "export fref {fref} is not a guest function"
            )));
        }
        (params, results)
    } else {
        (0, 0)
    };
    if let Some(program) = program {
        let entry_pc = program
            .function_entries
            .iter()
            .find_map(|&(function_ref, pc)| (function_ref == u64::from(fref)).then_some(pc))
            .ok_or_else(|| WasmBuildError::Trace(format!("missing entry pc for export fref {fref}")))?;
        if u64::from(first.pc) != entry_pc {
            return Err(WasmBuildError::Trace(format!(
                "export fref {fref} starts at pc {}, expected entry pc {entry_pc}; entry rows are missing",
                first.pc
            )));
        }
    }
    validate_runtime_entry_locals(template, first)?;
    let inputs = recover_entry_inputs(
        template,
        &first.locals_snapshot,
        first.memory_pages_before,
        first.entry_memory.as_ref(),
    )
    .map_err(|err| WasmBuildError::Trace(format!("export fref {fref} entry recovery: {err}")))?;
    let entry_blocks = crate::host_event_bindings::expand_export_entry(template, &inputs)
        .map_err(|err| WasmBuildError::Trace(format!("export entry expansion: {err}")))?;
    let memory_accesses = apply_export_entry_memory(
        &template.entry,
        &entry_blocks,
        &first.locals_snapshot,
        first.memory_pages_before,
        memory,
    )?;
    let entry_plans = plan_export_blocks(&template.entry, &entry_blocks, &first.locals_snapshot, &memory_accesses)?;
    if re_entered && entry_plans.is_empty() && template.exit.is_empty() {
        return Err(WasmBuildError::Trace(format!(
            "re-entered export fref {fref} requires at least one entry or exit event"
        )));
    }

    let mut expected_locals = vec![(false, 0u32, 0u32); first.locals_snapshot.len()];
    for plan in &entry_plans {
        for row in &plan.rows {
            if let Some((local, limb, value)) = row.local_write {
                let lanes = &mut expected_locals[local as usize];
                if limb == 0 {
                    *lanes = (true, value, 0);
                } else {
                    lanes.2 = value;
                }
            }
        }
    }
    // Zero-init rows clear non-parameter locals at re-entry; the entry
    // template writes every parameter. Any unwritten local must be zero in
    // the captured frame, just as on the first turn.
    for (local, &(_, lo, hi)) in expected_locals.iter().enumerate() {
        let (ran_lo, ran_hi) = first.locals_snapshot[local];
        if (lo, hi) != (ran_lo, ran_hi) {
            return Err(WasmBuildError::Trace(format!(
                "entry bootstrap does not reproduce the entry frame's locals: local {local} \
                 is ({lo}, {hi}) after the bootstrap writes but wasmtime ran with ({ran_lo}, {ran_hi})"
            )));
        }
    }
    Ok(TurnSetup {
        fref,
        param_count,
        result_count,
        template,
        entry_plans,
    })
}

pub(super) fn plan_turn_exit(
    template: &crate::host_event_bindings::ExportTemplate,
    last: &NormalizedStep,
    output: Option<(u32, u32)>,
    memory: &LinearMemoryImage,
) -> Result<Vec<EventBlockPlan>, WasmBuildError> {
    let resolved_memory = read_export_exit_memory(&template.exit, output, last.memory_pages_after, memory)?;
    let blocks = crate::host_event_bindings::expand_export_exit(template, output, &resolved_memory.reads)
        .map_err(|err| WasmBuildError::Trace(format!("export exit expansion: {err}")))?;
    plan_export_blocks(
        &template.exit,
        &blocks,
        &last.locals_snapshot,
        &resolved_memory.accesses,
    )
}

fn validate_runtime_entry_locals(
    template: &crate::host_event_bindings::ExportTemplate,
    first: &NormalizedStep,
) -> Result<(), WasmBuildError> {
    let local_count = usize::try_from(first.num_locals)
        .map_err(|_| WasmBuildError::Trace("runtime local count does not fit usize".to_string()))?;

    if first.locals_snapshot.len() != local_count {
        return Err(WasmBuildError::Trace(format!(
            "runtime local count {} does not match locals snapshot length {}",
            first.num_locals,
            first.locals_snapshot.len()
        )));
    }

    for source in template.entry.iter().flat_map(|event| &event.block) {
        let local = match *source {
            SlotBinding::InputLocal { local, .. }
            | SlotBinding::MemoryWrite32 {
                base: MemoryBase::Local(local),
                ..
            }
            | SlotBinding::MemoryWrite16 {
                base: MemoryBase::Local(local),
                ..
            }
            | SlotBinding::MemoryWrite8 {
                base: MemoryBase::Local(local),
                ..
            } => local,
            _ => continue,
        };

        if usize::from(local) >= local_count {
            return Err(WasmBuildError::Trace(format!(
                "export entry local {local} is missing from the runtime locals snapshot"
            )));
        }
    }

    Ok(())
}
