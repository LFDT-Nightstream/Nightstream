//! Emit the rows that copy a slide's or tail call's kept cells (branch-carried
//! operands, results, or tail-call arguments) down to their new position.

use super::super::host_event_emit::HostEventRowContext;
use super::super::NormalizedStep;
use crate::ir::{StackValueAccess, WasmAuxOpcode, WasmBuildError, WasmFrameMoveState, WasmStepState, WasmVmStep};

/// A branch or popping return keeps its edge's static `remaining` operands below the
/// row's reads and drops `distance` cells under them; it must land at the traced sp.
pub(super) fn frame_slide(
    current: &NormalizedStep,
    sp_before: u64,
    sp_after: u64,
) -> Result<(Vec<(u32, u32)>, WasmFrameMoveState), WasmBuildError> {
    let (remaining, distance) = current.frame_slide.ok_or_else(|| {
        WasmBuildError::Trace(format!(
            "{} at cycle {} has no static frame slide",
            current.info.name, current.cycle
        ))
    })?;
    let reads = current
        .stack_reads_override
        .unwrap_or(current.info.stack_reads);
    if sp_before.checked_sub(u64::from(reads) + u64::from(distance)) != Some(sp_after) {
        return Err(WasmBuildError::Trace(format!(
            "{} at cycle {} slides by {distance} from sp {sp_before}, but execution resumes at sp {sp_after}",
            current.info.name, current.cycle
        )));
    }
    let kept_end = current.operand_stack.len() - usize::from(reads);
    let kept_start = kept_end
        .checked_sub(remaining as usize)
        .ok_or_else(|| WasmBuildError::Trace(format!("frame slide underflow at cycle {}", current.cycle)))?;
    let cells = (kept_start..kept_end)
        .map(|pos| {
            (
                current.operand_stack[pos],
                current.operand_stack_hi.get(pos).copied().unwrap_or(0),
            )
        })
        .collect();
    let state = WasmFrameMoveState {
        active: remaining != 0,
        remaining,
        distance: u64::from(distance),
    };
    Ok((cells, state))
}

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
