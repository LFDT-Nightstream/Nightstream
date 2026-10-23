//! Emit the rows that copy a returning or tail-called frame's kept cells
//! (results or tail-call arguments) down to their new position.

use super::super::host_event_emit::HostEventRowContext;
use crate::ir::{StackValueAccess, WasmAuxOpcode, WasmStepState, WasmVmStep};

/// `cells` are the kept `(lo, hi)` values, bottom-up.
pub(super) fn emit_frame_move_rows(
    out: &mut Vec<WasmVmStep>,
    ctx: &HostEventRowContext,
    mut state: WasmStepState,
    cells: &[(u32, u32)],
) -> WasmStepState {
    debug_assert_eq!(cells.len(), state.frame_move.remaining as usize);
    for &(lo, hi) in cells {
        let dst = state.sp - u64::from(state.frame_move.remaining);
        let src = dst + state.frame_move.distance;
        let mut after = state;
        after.frame_move.remaining -= 1;
        after.frame_move.active = after.frame_move.remaining != 0;
        after.local_zero.active = !after.frame_move.active && after.local_zero.remaining != 0;
        out.push(WasmVmStep {
            wide_values_enabled: hi != 0,
            stack_reads_override: Some(1),
            stack_writes_override: Some(1),
            stack_read0: Some(StackValueAccess::new(2 * src, lo).with_optional_hi(Some(hi))),
            stack_write0: Some(StackValueAccess::new(2 * dst, lo).with_optional_hi(Some(hi))),
            ..ctx.row(out.len() as u64, WasmAuxOpcode::FrameMove, state, after)
        });
        state = after;
    }
    state
}
