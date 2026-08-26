//! Owns the columns carried from one wasm IVC step to the next.
//!
//! These links are not lookup or memory bindings. They define which `_after`
//! columns from row `i` must equal `_before` columns in row `i + 1`.

use neo_application::{ContinuityGroup, ContinuityLink};

use super::layout::{
    COL_CALL_STACK_DEPTH_AFTER, COL_CALL_STACK_DEPTH_BEFORE, COL_COMM_CHAIN_AFTER, COL_COMM_CHAIN_BEFORE,
    COL_EVBUF_AFTER, COL_EVBUF_BEFORE, COL_HALTED, COL_HALTED_BEFORE, COL_HOST_CALLEE_FREF_AFTER,
    COL_HOST_CALLEE_FREF_BEFORE, COL_HOST_EVENTS_REMAINING_AFTER, COL_HOST_EVENTS_REMAINING_BEFORE,
    COL_HOST_EVENT_ARGS_BASE_AFTER, COL_HOST_EVENT_ARGS_BASE_BEFORE, COL_HOST_EVENT_INDEX_AFTER,
    COL_HOST_EVENT_INDEX_BEFORE, COL_HOST_EVENT_SLOT_CURSOR_AFTER, COL_HOST_EVENT_SLOT_CURSOR_BEFORE,
    COL_LOCALS_FBP_AFTER, COL_LOCALS_FBP_BEFORE, COL_MAX_MEMORY_PAGES_AFTER, COL_MAX_MEMORY_PAGES_BEFORE,
    COL_MEMORY_PAGES_AFTER, COL_MEMORY_PAGES_BEFORE, COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE,
    COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE, COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE,
    COL_PARAM_INIT_ACTIVE_AFTER, COL_PARAM_INIT_ACTIVE_BEFORE, COL_PARAM_INIT_REMAINING_AFTER,
    COL_PARAM_INIT_REMAINING_BEFORE, COL_PC_AFTER, COL_PC_BEFORE, COL_PERM_PENDING_AFTER, COL_PERM_PENDING_BEFORE,
    COL_PERM_ROUND_AFTER, COL_PERM_ROUND_BEFORE, COL_PERM_STATE_AFTER, COL_PERM_STATE_BEFORE, COL_SP_AFTER,
    COL_SP_BEFORE, COL_STACK_FRAME_BASE_AFTER, COL_STACK_FRAME_BASE_BEFORE, COL_TAIL_CALL_PENDING_AFTER,
    COL_TAIL_CALL_PENDING_BEFORE, COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE, COL_TURN_EXPORT_FREF_AFTER,
    COL_TURN_EXPORT_FREF_BEFORE,
};

const fn link(previous_step_column: usize, next_step_column: usize) -> ContinuityLink {
    ContinuityLink {
        previous_step_column,
        next_step_column,
    }
}

pub(crate) fn build_ivc_state_continuity_links() -> Vec<ContinuityGroup> {
    vec![
        ContinuityGroup {
            name: "halted_continuity",
            role: "row[i].halted_after must match row[i+1].halted_before",
            links: vec![link(COL_HALTED, COL_HALTED_BEFORE)],
        },
        ContinuityGroup {
            name: "pc_continuity",
            role: "row[i].pc_after must match row[i+1].pc_before",
            links: vec![link(COL_PC_AFTER, COL_PC_BEFORE)],
        },
        ContinuityGroup {
            name: "sp_continuity",
            role: "row[i].sp_after must match row[i+1].sp_before",
            links: vec![link(COL_SP_AFTER, COL_SP_BEFORE)],
        },
        ContinuityGroup {
            name: "stack_frame_base_continuity",
            role: "row[i].stack_frame_base_after must match row[i+1].stack_frame_base_before",
            links: vec![link(COL_STACK_FRAME_BASE_AFTER, COL_STACK_FRAME_BASE_BEFORE)],
        },
        ContinuityGroup {
            name: "output_continuity",
            role: "row[i].simple output carry must match row[i+1].simple output carry",
            links: vec![
                link(COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE),
                link(COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE),
                link(COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE),
            ],
        },
        ContinuityGroup {
            name: "call_stack_depth_continuity",
            role: "row[i].call_stack_depth_after must match row[i+1].call_stack_depth_before",
            links: vec![link(COL_CALL_STACK_DEPTH_AFTER, COL_CALL_STACK_DEPTH_BEFORE)],
        },
        ContinuityGroup {
            name: "memory_pages_continuity",
            role: "row[i].memory_pages_after must match row[i+1].memory_pages_before",
            links: vec![link(COL_MEMORY_PAGES_AFTER, COL_MEMORY_PAGES_BEFORE)],
        },
        ContinuityGroup {
            name: "max_memory_pages_continuity",
            role: "row[i].max_memory_pages_after must match row[i+1].max_memory_pages_before (carried constant)",
            links: vec![link(COL_MAX_MEMORY_PAGES_AFTER, COL_MAX_MEMORY_PAGES_BEFORE)],
        },
        ContinuityGroup {
            name: "locals_fbp_continuity",
            role: "row[i].locals_fbp_after must match row[i+1].locals_fbp_before",
            links: vec![link(COL_LOCALS_FBP_AFTER, COL_LOCALS_FBP_BEFORE)],
        },
        ContinuityGroup {
            name: "trapped_continuity",
            role: "row[i].trapped_after must match row[i+1].trapped_before",
            links: vec![link(COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE)],
        },
        ContinuityGroup {
            name: "param_init_continuity",
            role: "row[i].param_init_after state must match row[i+1].param_init_before state",
            links: vec![
                link(COL_PARAM_INIT_ACTIVE_AFTER, COL_PARAM_INIT_ACTIVE_BEFORE),
                link(COL_PARAM_INIT_REMAINING_AFTER, COL_PARAM_INIT_REMAINING_BEFORE),
            ],
        },
        ContinuityGroup {
            name: "tail_call_continuity",
            role: "row[i].tail_call_pending_after must match row[i+1].tail_call_pending_before",
            links: vec![link(COL_TAIL_CALL_PENDING_AFTER, COL_TAIL_CALL_PENDING_BEFORE)],
        },
        ContinuityGroup {
            name: "host_call_attribution_continuity",
            role: "row[i].host callee fref must match row[i+1]",
            links: vec![link(COL_HOST_CALLEE_FREF_AFTER, COL_HOST_CALLEE_FREF_BEFORE)],
        },
        ContinuityGroup {
            name: "turn_export_fref_continuity",
            role: "row[i].turn export fref must match row[i+1]",
            links: vec![link(COL_TURN_EXPORT_FREF_AFTER, COL_TURN_EXPORT_FREF_BEFORE)],
        },
        ContinuityGroup {
            name: "comm_chain_continuity",
            role: "row[i].host-event commitment chain must match row[i+1].host-event commitment chain",
            links: COL_COMM_CHAIN_AFTER
                .into_iter()
                .zip(COL_COMM_CHAIN_BEFORE)
                .map(|(after, before)| link(after, before))
                .collect(),
        },
        ContinuityGroup {
            name: "host_event_gather_continuity",
            role: "row[i].host_events gather machinery (schedule, args base, cursor) must match row[i+1]",
            links: vec![
                link(COL_HOST_EVENTS_REMAINING_AFTER, COL_HOST_EVENTS_REMAINING_BEFORE),
                link(COL_HOST_EVENT_INDEX_AFTER, COL_HOST_EVENT_INDEX_BEFORE),
                link(COL_HOST_EVENT_ARGS_BASE_AFTER, COL_HOST_EVENT_ARGS_BASE_BEFORE),
                link(COL_HOST_EVENT_SLOT_CURSOR_AFTER, COL_HOST_EVENT_SLOT_CURSOR_BEFORE),
            ],
        },
        ContinuityGroup {
            name: "event_absorb_continuity",
            role: "row[i].host-event absorb state (block buffer and perm group) must match row[i+1]",
            links: {
                let mut links = vec![
                    link(COL_PERM_PENDING_AFTER, COL_PERM_PENDING_BEFORE),
                    link(COL_PERM_ROUND_AFTER, COL_PERM_ROUND_BEFORE),
                ];
                links.extend(
                    COL_EVBUF_AFTER
                        .into_iter()
                        .zip(COL_EVBUF_BEFORE)
                        .chain(COL_PERM_STATE_AFTER.into_iter().zip(COL_PERM_STATE_BEFORE))
                        .map(|(after, before)| link(after, before)),
                );
                links
            },
        },
    ]
}
