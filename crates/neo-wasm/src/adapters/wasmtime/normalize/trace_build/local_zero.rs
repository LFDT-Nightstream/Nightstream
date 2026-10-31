//! Emit the zero pushes that give a new frame's declared locals Wasm's fresh values.

use super::super::host_event_emit::HostEventRowContext;
use crate::ir::{StackValueAccess, WasmAuxOpcode, WasmStepState, WasmVmStep};

pub(super) fn emit_local_zero_rows(
    out: &mut Vec<WasmVmStep>,
    ctx: &HostEventRowContext,
    mut state: WasmStepState,
) -> WasmStepState {
    while state.local_zero.active {
        let mut after = state;
        after.sp += 1;
        after.frame_move.distance = 0;
        after.local_zero.remaining -= 1;
        after.local_zero.active = after.local_zero.remaining != 0;
        out.push(WasmVmStep {
            stack_writes_override: Some(1),
            stack_write0: Some(StackValueAccess::new(2 * state.sp, 0).with_optional_hi(Some(0))),
            ..ctx.row(out.len() as u64, WasmAuxOpcode::LocalZero, state, after)
        });
        state = after;
    }
    state
}
