//! RV32 trace-wiring CCS (shared-bus compatible).
//!
//! This module exposes the trace-mode CCS layout/witness/builders and shared CPU-bus
//! requirements/configuration used by the `Rv32TraceWiring` proving flow.

mod bus_bindings;
mod constants;
mod constraint_builder;
mod rv64_trace;
mod trace;

pub use bus_bindings::{
    rv32_trace_shared_bus_extraction, rv32_trace_shared_bus_extraction_with_specs, rv32_trace_shared_bus_requirements,
    rv32_trace_shared_bus_requirements_with_specs, rv32_trace_shared_cpu_bus_config,
    rv32_trace_shared_cpu_bus_config_with_specs, TraceSharedBusExtraction, TraceShoutBusSpec,
};
pub use rv64_trace::{
    build_rv64_trace_wiring_ccs, rv64_trace_ccs_witness_from_exec_table, rv64_trace_ccs_witness_from_trace_witness,
    Rv64TraceCcsLayout,
};
pub use trace::{
    build_rv32_trace_wiring_ccs, build_rv32_trace_wiring_ccs_with_reserved_rows, build_rv32_uniform_constraint_key,
    build_rv32_uniform_constraint_key_with_m_in, rv32_trace_ccs_witness_from_exec_table,
    rv32_trace_ccs_witness_from_trace_witness, Rv32TraceCcsLayout,
};

use constants::{
    ADD_TABLE_ID, AND_TABLE_ID, DIVU_TABLE_ID, DIV_TABLE_ID, EQ_TABLE_ID, MULHSU_TABLE_ID, MULHU_TABLE_ID,
    MULH_TABLE_ID, MUL_TABLE_ID, NEQ_TABLE_ID, OR_TABLE_ID, REMU_TABLE_ID, REM_TABLE_ID, SLL_TABLE_ID, SLTU_TABLE_ID,
    SLT_TABLE_ID, SRA_TABLE_ID, SRL_TABLE_ID, SUB_TABLE_ID, XOR_TABLE_ID,
};

/// Minimal trace-mode Shout profile for tiny RV32 programs.
pub const RV32_TRACE_SHOUT_PROFILE_MIN3: &[u32] = &[ADD_TABLE_ID, EQ_TABLE_ID, NEQ_TABLE_ID];

/// Full RV32I trace-mode Shout profile.
pub const RV32_TRACE_SHOUT_PROFILE_FULL12: &[u32] = &[
    AND_TABLE_ID,
    XOR_TABLE_ID,
    OR_TABLE_ID,
    ADD_TABLE_ID,
    SUB_TABLE_ID,
    SLT_TABLE_ID,
    SLTU_TABLE_ID,
    SLL_TABLE_ID,
    SRL_TABLE_ID,
    SRA_TABLE_ID,
    EQ_TABLE_ID,
    NEQ_TABLE_ID,
];

/// Full RV32IM trace-mode Shout profile.
pub const RV32_TRACE_SHOUT_PROFILE_FULL20: &[u32] = &[
    AND_TABLE_ID,
    XOR_TABLE_ID,
    OR_TABLE_ID,
    ADD_TABLE_ID,
    SUB_TABLE_ID,
    SLT_TABLE_ID,
    SLTU_TABLE_ID,
    SLL_TABLE_ID,
    SRL_TABLE_ID,
    SRA_TABLE_ID,
    EQ_TABLE_ID,
    NEQ_TABLE_ID,
    MUL_TABLE_ID,
    MULH_TABLE_ID,
    MULHU_TABLE_ID,
    MULHSU_TABLE_ID,
    DIV_TABLE_ID,
    DIVU_TABLE_ID,
    REM_TABLE_ID,
    REMU_TABLE_ID,
];
