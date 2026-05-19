//! Diagnostic and measurement API for RV32IM.

pub use super::super::main_relation_spartan::debug_measure_rv32im_main_recursion_step_chunk_replay_fingerprint;
pub use super::super::main_relation_spartan::debug_measure_rv32im_main_relation_state_in_prefix_fingerprints;
pub use super::super::perf_case::{
    build_mixed_opcode_perf_source_case, mixed_opcode_perf_expected_x1, RV32IM_MIXED_OPCODE_PERF_BLOCK_LEN,
    RV32IM_MIXED_OPCODE_PERF_DEFAULT_N,
};
