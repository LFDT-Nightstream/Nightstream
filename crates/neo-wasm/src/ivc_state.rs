//! Owns the columns carried from one wasm IVC step to the next.
//!
//! These links are not lookup or memory bindings. They define which `_after`
//! columns from row `i` must equal `_before` columns in row `i + 1`.

use super::layout::{
    Column, COL_CALL_STACK_DEPTH_AFTER, COL_CALL_STACK_DEPTH_BEFORE, COL_COMM_CHAIN0_AFTER, COL_COMM_CHAIN0_BEFORE,
    COL_COMM_CHAIN1_AFTER, COL_COMM_CHAIN1_BEFORE, COL_COMM_CHAIN2_AFTER, COL_COMM_CHAIN2_BEFORE,
    COL_COMM_CHAIN3_AFTER, COL_COMM_CHAIN3_BEFORE, COL_EVBUF0_AFTER, COL_EVBUF0_BEFORE, COL_EVBUF7_AFTER,
    COL_EVBUF7_BEFORE, COL_EVBUF_SLOT0_AFTER, COL_EVBUF_SLOT0_BEFORE, COL_EVBUF_SLOT3_AFTER, COL_EVBUF_SLOT3_BEFORE,
    COL_GRAMMAR_MODE_AFTER, COL_GRAMMAR_MODE_BEFORE, COL_HALTED, COL_HALTED_BEFORE,
    COL_HOST_ARGS_ACTIVE_AFTER, COL_HOST_ARGS_ACTIVE_BEFORE, COL_HOST_ARGS_REMAINING_AFTER,
    COL_HOST_ARGS_REMAINING_BEFORE, COL_HOST_CALLEE_FREF_AFTER, COL_HOST_CALLEE_FREF_BEFORE,
    COL_HOST_RESULT_PENDING_AFTER, COL_HOST_RESULT_PENDING_BEFORE, COL_LOCALS_FBP_AFTER, COL_LOCALS_FBP_BEFORE,
    COL_MAX_MEMORY_PAGES_AFTER, COL_MAX_MEMORY_PAGES_BEFORE, COL_MEMORY_PAGES_AFTER, COL_MEMORY_PAGES_BEFORE,
    COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE,
    COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE, COL_PARAM_INIT_ACTIVE_AFTER, COL_PARAM_INIT_ACTIVE_BEFORE,
    COL_PARAM_INIT_REMAINING_AFTER, COL_PARAM_INIT_REMAINING_BEFORE, COL_PC_AFTER, COL_PC_BEFORE,
    COL_PERM_PENDING_AFTER, COL_PERM_PENDING_BEFORE, COL_PERM_ROUND_AFTER, COL_PERM_ROUND_BEFORE,
    COL_PERM_STATE0_AFTER, COL_PERM_STATE0_BEFORE, COL_PERM_STATE11_AFTER, COL_PERM_STATE11_BEFORE, COL_SP_AFTER,
    COL_SP_BEFORE, COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE,
};

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

pub(crate) fn build_ivc_state_continuity_links() -> Vec<WasmCrossStepLinkSpec> {
    vec![
        WasmCrossStepLinkSpec {
            name: "halted_continuity",
            description: "row[i].halted_after must match row[i+1].halted_before",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: Column(COL_HALTED),
                next_before: Column(COL_HALTED_BEFORE),
            }],
        },
        WasmCrossStepLinkSpec {
            name: "pc_continuity",
            description: "row[i].pc_after must match row[i+1].pc_before",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: Column(COL_PC_AFTER),
                next_before: Column(COL_PC_BEFORE),
            }],
        },
        WasmCrossStepLinkSpec {
            name: "sp_continuity",
            description: "row[i].sp_after must match row[i+1].sp_before",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: Column(COL_SP_AFTER),
                next_before: Column(COL_SP_BEFORE),
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
                prev_after: Column(COL_CALL_STACK_DEPTH_AFTER),
                next_before: Column(COL_CALL_STACK_DEPTH_BEFORE),
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
                prev_after: Column(COL_LOCALS_FBP_AFTER),
                next_before: Column(COL_LOCALS_FBP_BEFORE),
            }],
        },
        WasmCrossStepLinkSpec {
            name: "trapped_continuity",
            description: "row[i].trapped_after must match row[i+1].trapped_before",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: Column(COL_TRAPPED_AFTER),
                next_before: Column(COL_TRAPPED_BEFORE),
            }],
        },
        WasmCrossStepLinkSpec {
            name: "param_init_continuity",
            description: "row[i].param_init_after state must match row[i+1].param_init_before state",
            column_pairs: vec![
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_PARAM_INIT_ACTIVE_AFTER),
                    next_before: Column(COL_PARAM_INIT_ACTIVE_BEFORE),
                },
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_PARAM_INIT_REMAINING_AFTER),
                    next_before: Column(COL_PARAM_INIT_REMAINING_BEFORE),
                },
            ],
        },
        WasmCrossStepLinkSpec {
            name: "host_call_continuity",
            description: "row[i].host-call arg/result state must match row[i+1].host-call arg/result state",
            column_pairs: vec![
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_HOST_ARGS_ACTIVE_AFTER),
                    next_before: Column(COL_HOST_ARGS_ACTIVE_BEFORE),
                },
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_HOST_ARGS_REMAINING_AFTER),
                    next_before: Column(COL_HOST_ARGS_REMAINING_BEFORE),
                },
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_HOST_RESULT_PENDING_AFTER),
                    next_before: Column(COL_HOST_RESULT_PENDING_BEFORE),
                },
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_HOST_CALLEE_FREF_AFTER),
                    next_before: Column(COL_HOST_CALLEE_FREF_BEFORE),
                },
            ],
        },
        WasmCrossStepLinkSpec {
            name: "comm_chain_continuity",
            description: "row[i].host-event commitment chain must match row[i+1].host-event commitment chain",
            column_pairs: vec![
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_COMM_CHAIN0_AFTER),
                    next_before: Column(COL_COMM_CHAIN0_BEFORE),
                },
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_COMM_CHAIN1_AFTER),
                    next_before: Column(COL_COMM_CHAIN1_BEFORE),
                },
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_COMM_CHAIN2_AFTER),
                    next_before: Column(COL_COMM_CHAIN2_BEFORE),
                },
                WasmCrossStepColumnPair {
                    prev_after: Column(COL_COMM_CHAIN3_AFTER),
                    next_before: Column(COL_COMM_CHAIN3_BEFORE),
                },
            ],
        },
        WasmCrossStepLinkSpec {
            name: "grammar_mode_continuity",
            description: "row[i].grammar_mode (per-program constant) must match row[i+1]",
            column_pairs: vec![WasmCrossStepColumnPair {
                prev_after: Column(COL_GRAMMAR_MODE_AFTER),
                next_before: Column(COL_GRAMMAR_MODE_BEFORE),
            }],
        },
        WasmCrossStepLinkSpec {
            name: "event_absorb_continuity",
            description: "row[i].host-event absorb state (block buffer, slot cursor, perm group) must match row[i+1]",
            column_pairs: {
                let mut pairs = vec![
                    WasmCrossStepColumnPair {
                        prev_after: Column(COL_PERM_PENDING_AFTER),
                        next_before: Column(COL_PERM_PENDING_BEFORE),
                    },
                    WasmCrossStepColumnPair {
                        prev_after: Column(COL_PERM_ROUND_AFTER),
                        next_before: Column(COL_PERM_ROUND_BEFORE),
                    },
                ];
                pairs.extend(
                    (COL_EVBUF0_AFTER..=COL_EVBUF7_AFTER)
                        .zip(COL_EVBUF0_BEFORE..=COL_EVBUF7_BEFORE)
                        .chain(
                            (COL_EVBUF_SLOT0_AFTER..=COL_EVBUF_SLOT3_AFTER)
                                .zip(COL_EVBUF_SLOT0_BEFORE..=COL_EVBUF_SLOT3_BEFORE),
                        )
                        .chain(
                            (COL_PERM_STATE0_AFTER..=COL_PERM_STATE11_AFTER)
                                .zip(COL_PERM_STATE0_BEFORE..=COL_PERM_STATE11_BEFORE),
                        )
                        .map(|(after, before)| WasmCrossStepColumnPair {
                            prev_after: Column(after),
                            next_before: Column(before),
                        }),
                );
                pairs
            },
        },
    ]
}
