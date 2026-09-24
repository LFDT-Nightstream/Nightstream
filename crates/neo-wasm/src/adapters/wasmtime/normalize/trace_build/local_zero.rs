//! Emit the exhaustive zero writes that give each new local frame Wasm's fresh values.

use super::super::host_event_emit::HostEventRowContext;
use crate::ir::{WasmAuxOpcode, WasmStepState, WasmVmStep};

pub(super) fn emit_local_zero_rows(
    out: &mut Vec<WasmVmStep>,
    ctx: &HostEventRowContext,
    mut state: WasmStepState,
) -> WasmStepState {
    while state.local_zero.active {
        let local = ctx.current_function_num_locals - state.local_zero.remaining;
        let mut after = state;
        after.local_zero.remaining -= 1;
        after.local_zero.active = after.local_zero.remaining != 0;
        after.param_init.active = !after.local_zero.active && after.param_init.remaining != 0;
        out.push(WasmVmStep {
            local_index: Some(local),
            local_write_value: Some(0),
            local_write_value_hi: Some(0),
            ..ctx.row(out.len() as u64, WasmAuxOpcode::LocalZero, state, after)
        });
        state = after;
    }
    state
}
