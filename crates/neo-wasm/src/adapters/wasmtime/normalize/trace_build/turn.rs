use super::super::grammar_emit::{
    apply_export_entry_memory, plan_export_blocks, read_export_exit_memory, GrammarBlockPlan,
};
use super::super::memory::LinearMemoryImage;
use super::super::NormalizedStep;
use crate::event_grammar::{HostEventGrammar, TurnClaims};
use crate::ir::WasmBuildError;

pub(super) struct TurnSetup<'g> {
    pub(super) fref: u32,
    pub(super) template: &'g crate::event_grammar::ExportTemplate,
    pub(super) entry_plans: Vec<GrammarBlockPlan>,
}

pub(super) fn setup_turn<'g>(
    grammar: &'g HostEventGrammar,
    first: &NormalizedStep,
    claims: &TurnClaims,
    re_entered: bool,
    memory: &mut LinearMemoryImage,
) -> Result<TurnSetup<'g>, WasmBuildError> {
    let fref = first.current_function_ref.unwrap_or(0);
    let template = grammar.exports.get(&fref).ok_or_else(|| {
        WasmBuildError::Trace(format!(
            "event grammar requires an export template for the invoked export (fref {fref})"
        ))
    })?;
    let local_bound = u8::try_from(first.num_locals.min(255)).expect("bounded");
    template.validate(local_bound)?;
    let entry_blocks = crate::event_grammar::expand_export_entry(template, &claims.entry)
        .map_err(|err| WasmBuildError::Trace(format!("export entry expansion: {err}")))?;
    let memory_accesses = apply_export_entry_memory(&template.entry, &entry_blocks, &first.locals_snapshot, memory)?;
    let entry_plans = plan_export_blocks(&template.entry, &entry_blocks, &first.locals_snapshot, &memory_accesses)?;
    if re_entered && entry_plans.is_empty() {
        return Err(WasmBuildError::Trace(format!(
            "re-entered export fref {fref} requires at least one entry event"
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
    for (local, &(lo_written, lo, hi)) in expected_locals.iter().enumerate() {
        if !lo_written {
            if re_entered {
                return Err(WasmBuildError::Trace(format!(
                    "re-entered turn must bootstrap-write every local: local {local} has no lo-lane write \
                     (the locals RAM still holds the previous turn's values)"
                )));
            }
            // First-turn locals that the entry template leaves alone are
            // bound by the locals-RAM preload (the verifier's statement).
            continue;
        }
        let ran_lo = first.locals_snapshot[local];
        let ran_hi = first.locals_snapshot_hi.get(local).copied().unwrap_or(0);
        if (lo, hi) != (ran_lo, ran_hi) {
            return Err(WasmBuildError::Trace(format!(
                "entry bootstrap does not reproduce the entry frame's locals: local {local} \
                 is ({lo}, {hi}) after the bootstrap writes but wasmtime ran with ({ran_lo}, {ran_hi})"
            )));
        }
    }
    Ok(TurnSetup {
        fref,
        template,
        entry_plans,
    })
}

pub(super) fn plan_turn_exit(
    template: &crate::event_grammar::ExportTemplate,
    last: &NormalizedStep,
    claims: &TurnClaims,
    output: Option<(u32, u32)>,
    memory: &LinearMemoryImage,
) -> Result<Vec<GrammarBlockPlan>, WasmBuildError> {
    let resolved_memory = read_export_exit_memory(&template.exit, &last.locals_snapshot, memory)?;
    let blocks = crate::event_grammar::expand_export_exit(template, output, &claims.exit, &resolved_memory.reads)
        .map_err(|err| WasmBuildError::Trace(format!("export exit expansion: {err}")))?;
    plan_export_blocks(
        &template.exit,
        &blocks,
        &last.locals_snapshot,
        &resolved_memory.accesses,
    )
}
