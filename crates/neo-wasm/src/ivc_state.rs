//! Owns the columns carried from one wasm IVC step to the next.
//!
//! These links are not lookup or memory bindings. They define which `_after`
//! columns from row `i` must equal `_before` columns in row `i + 1`.

use super::layout::{
    COL_MAX_MEMORY_PAGES_AFTER, COL_MAX_MEMORY_PAGES_BEFORE, COL_MEMORY_PAGES_AFTER, COL_MEMORY_PAGES_BEFORE,
    COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE,
    COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE,
};
use super::lookup_binding_builder::{CallColumns, Column, FrameColumns, ParamInitColumns};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateColumns {
    pub pc_before: Column,
    pub pc_after: Column,
    pub sp_before: Column,
    pub sp_after: Column,
    pub trapped_before: Column,
    pub trapped_after: Column,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmCrossStepLinkSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub column_pairs: Vec<WasmCrossStepColumnPair>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WasmCrossStepColumnPair {
    pub prev_after: Column,
    pub next_before: Column,
}

pub(crate) fn build_ivc_state_continuity_links(
    state: &StateColumns,
    param_init: &ParamInitColumns,
    call: &CallColumns,
    frame: &FrameColumns,
) -> Vec<WasmCrossStepLinkSpec> {
    vec![
        WasmCrossStepLinkSpec {
            name: "pc_continuity",
            description: "row[i].pc_after must match row[i+1].pc_before",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: state.pc_after,
                next_before: state.pc_before,
            }],
        },
        WasmCrossStepLinkSpec {
            name: "sp_continuity",
            description: "row[i].sp_after must match row[i+1].sp_before",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: state.sp_after,
                next_before: state.sp_before,
            }],
        },
        WasmCrossStepLinkSpec {
            name: "output_continuity",
            description: "row[i].simple output carry must match row[i+1].simple output carry",
            column_pairs: vec![
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_OUTPUT_ENABLED_AFTER),
                    next_before: Column(COL_OUTPUT_ENABLED_BEFORE),
                },
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_OUTPUT_VALUE_LO_AFTER),
                    next_before: Column(COL_OUTPUT_VALUE_LO_BEFORE),
                },
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_OUTPUT_VALUE_HI_AFTER),
                    next_before: Column(COL_OUTPUT_VALUE_HI_BEFORE),
                },
            ],
        },
        WasmCrossStepLinkSpec {
            name: "call_stack_depth_continuity",
            description: "row[i].call_stack_depth_after must match row[i+1].call_stack_depth_before",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: call.call_stack_depth_after,
                next_before: call.call_stack_depth_before,
            }],
        },
        WasmCrossStepLinkSpec {
            name: "memory_pages_continuity",
            description: "row[i].memory_pages_after must match row[i+1].memory_pages_before",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: Column(COL_MEMORY_PAGES_AFTER),
                next_before: Column(COL_MEMORY_PAGES_BEFORE),
            }],
        },
        WasmCrossStepLinkSpec {
            name: "max_memory_pages_continuity",
            description: "row[i].max_memory_pages_after must match row[i+1].max_memory_pages_before (carried constant)",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: Column(COL_MAX_MEMORY_PAGES_AFTER),
                next_before: Column(COL_MAX_MEMORY_PAGES_BEFORE),
            }],
        },
        WasmCrossStepLinkSpec {
            name: "locals_fbp_continuity",
            description: "row[i].locals_fbp_after must match row[i+1].locals_fbp_before",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: frame.locals_fbp_after,
                next_before: frame.locals_fbp_before,
            }],
        },
        WasmCrossStepLinkSpec {
            name: "trapped_continuity",
            description: "row[i].trapped_after must match row[i+1].trapped_before",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: state.trapped_after,
                next_before: state.trapped_before,
            }],
        },
        WasmCrossStepLinkSpec {
            name: "param_init_continuity",
            description: "row[i].param_init_after state must match row[i+1].param_init_before state",
            column_pairs: vec![
                WasmCrossStepColumnPair {
                    prev_after: param_init.param_init_active_after,
                    next_before: param_init.param_init_active_before,
                },
                WasmCrossStepColumnPair {
                    prev_after: param_init.param_init_remaining_after,
                    next_before: param_init.param_init_remaining_before,
                },
            ],
        },
    ]
}
