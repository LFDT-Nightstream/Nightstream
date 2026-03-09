use crate::riscv::instruction::operand_mode_keys_enabled;

pub mod air;
pub mod decode_lookup;
pub mod layout;
pub mod rv32;
pub mod rv64;
pub mod rv64_width_sidecar;
pub mod sidecar_extract;
pub mod width_sidecar;
pub mod witness;

pub use air::{RiscvTraceAir, Rv32TraceAir};
pub use rv32::{
    rv32_decode_lookup_addr_group_for_table_id as riscv_decode_lookup_addr_group_for_table_id,
    rv32_decode_lookup_backed_cols as riscv_decode_lookup_backed_cols,
    rv32_decode_lookup_backed_row_from_instr_word as riscv_decode_lookup_backed_row_from_instr_word,
    rv32_decode_lookup_table_id_for_col as riscv_decode_lookup_table_id_for_col,
    rv32_decode_lookup_transport_cols as riscv_decode_lookup_transport_cols,
    rv32_decode_lookup_transport_n_vals as riscv_decode_lookup_transport_n_vals,
    rv32_decode_lookup_val_slot_for_col as riscv_decode_lookup_val_slot_for_col,
    rv32_is_decode_lookup_grouped_table_id as riscv_is_decode_lookup_grouped_table_id,
    rv32_is_decode_lookup_table_id as riscv_is_decode_lookup_table_id,
};
pub use rv32::{
    rv32_decode_lookup_addr_group_for_table_id, rv32_decode_lookup_backed_cols,
    rv32_decode_lookup_backed_row_from_instr_word, rv32_decode_lookup_table_id_for_col,
    rv32_decode_lookup_transport_cols, rv32_decode_lookup_transport_n_vals, rv32_decode_lookup_val_slot_for_col,
    rv32_is_decode_lookup_grouped_table_id, rv32_is_decode_lookup_table_id, Rv32DecodeSidecarLayout, Rv32TraceLayout,
    RV32_TRACE_DECODE_LOOKUP_GROUPED_TABLE_ID, RV32_TRACE_DECODE_LOOKUP_TABLE_BASE,
};
pub use rv32::{
    rv32_is_width_lookup_grouped_table_id, rv32_is_width_lookup_table_id, rv32_width_lookup_addr_group_for_table_id,
    rv32_width_lookup_backed_cols, rv32_width_lookup_table_id_for_col, rv32_width_lookup_transport_n_vals,
    rv32_width_lookup_val_slot_for_col, rv32_width_sidecar_witness_from_exec_table, Rv32TraceWitness,
    Rv32WidthSidecarLayout, Rv32WidthSidecarWitness, RV32_TRACE_WIDTH_LOOKUP_GROUPED_TABLE_ID,
    RV32_TRACE_WIDTH_LOOKUP_TABLE_BASE,
};
pub use rv64::{Rv64TraceLayout, Rv64TraceWitness};
pub use rv64_width_sidecar::{
    rv64_is_width_lookup_grouped_table_id, rv64_is_width_lookup_table_id, rv64_width_lookup_addr_group_for_table_id,
    rv64_width_lookup_backed_cols, rv64_width_lookup_table_id_for_col, rv64_width_lookup_transport_n_vals,
    rv64_width_lookup_val_slot_for_col, rv64_width_sidecar_witness_from_exec_table, Rv64WidthSidecarLayout,
    Rv64WidthSidecarWitness, RV64_TRACE_WIDTH_LOOKUP_GROUPED_TABLE_ID, RV64_TRACE_WIDTH_LOOKUP_TABLE_BASE,
};
pub use sidecar_extract::{
    extract_shout_lanes_over_time, extract_twist_lanes_over_time, ShoutLaneOverTime, TraceTwistLanesOverTime,
    TwistLaneOverTime,
};

#[inline]
pub fn rv32_trace_cpu_cols() -> usize {
    Rv32TraceLayout::new().cols
}

#[inline]
pub fn rv64_trace_cpu_cols() -> usize {
    Rv64TraceLayout::new().cols
}

#[inline]
pub fn infer_riscv_trace_machine_xlen(cpu_cols_len: usize) -> Option<usize> {
    if cpu_cols_len == rv32_trace_cpu_cols() {
        Some(32)
    } else if cpu_cols_len >= rv64_trace_cpu_cols() {
        Some(64)
    } else {
        None
    }
}

/// Shared-address group id for canonical RV32 opcode Shout tables (table_id 0..=19).
///
/// These families all use the same interleaved `(lhs,rhs)` key width (`ell_addr=64`),
/// so in RV32 trace shared-bus mode they can share one addr-bit range.
pub const RV32_TRACE_OPCODE_ADDR_GROUP: u32 = 0x5256_4100;
/// Shared-address group id for RV32 combined-key opcode Shout tables in operand-mode keying.
///
/// When operand-mode keys are enabled, these opcodes no longer use interleaved `(lhs,rhs)` keys.
/// They must therefore not share opcode `addr_bits` with interleaved-key tables.
pub const RV32_TRACE_OPCODE_COMBINED_ADDR_GROUP: u32 = 0x5256_4101;
/// Shared selector-group id for decode lookup families (table_id range at `RV32_TRACE_DECODE_LOOKUP_TABLE_BASE`).
pub const RV32_TRACE_DECODE_SELECTOR_GROUP: u32 = 0x5256_4B00;
/// Shared selector-group id for width lookup families (table_id range at `RV32_TRACE_WIDTH_LOOKUP_TABLE_BASE`).
pub const RV32_TRACE_WIDTH_SELECTOR_GROUP: u32 = 0x5256_5B00;

#[inline]
pub fn rv32_trace_uses_combined_operand_key_table_id(table_id: u32) -> bool {
    operand_mode_keys_enabled() && matches!(table_id, 3 | 4)
}

#[inline]
pub fn rv32_trace_lookup_addr_group_for_table_id(table_id: u32) -> Option<u32> {
    if table_id <= 19 {
        if rv32_trace_uses_combined_operand_key_table_id(table_id) {
            Some(RV32_TRACE_OPCODE_COMBINED_ADDR_GROUP)
        } else {
            Some(RV32_TRACE_OPCODE_ADDR_GROUP)
        }
    } else {
        rv32_decode_lookup_addr_group_for_table_id(table_id)
            .or_else(|| riscv_trace_shared_width_lookup_addr_group_for_table_id(table_id))
    }
}

#[inline]
pub fn rv32_trace_lookup_selector_group_for_table_id(table_id: u32) -> Option<u32> {
    if rv32_is_decode_lookup_table_id(table_id) {
        Some(RV32_TRACE_DECODE_SELECTOR_GROUP)
    } else if riscv_trace_is_width_lookup_table_id(table_id) {
        Some(RV32_TRACE_WIDTH_SELECTOR_GROUP)
    } else {
        None
    }
}

#[inline]
pub fn rv32_trace_lookup_n_vals_for_table_id(table_id: u32) -> usize {
    let n_vals = if rv32_is_decode_lookup_grouped_table_id(table_id) {
        rv32_decode_lookup_transport_n_vals()
    } else if riscv_trace_is_width_lookup_grouped_table_id(table_id) {
        if rv32_is_width_lookup_grouped_table_id(table_id) {
            rv32_width_lookup_transport_n_vals()
        } else {
            rv64_width_lookup_transport_n_vals()
        }
    } else {
        1
    };
    n_vals.max(1)
}

#[inline]
pub fn riscv_trace_lookup_addr_group_for_table_id(table_id: u32) -> Option<u32> {
    rv32_trace_lookup_addr_group_for_table_id(table_id)
}

#[inline]
pub fn riscv_trace_lookup_selector_group_for_table_id(table_id: u32) -> Option<u32> {
    rv32_trace_lookup_selector_group_for_table_id(table_id)
}

#[inline]
pub fn riscv_trace_lookup_n_vals_for_table_id(table_id: u32) -> usize {
    rv32_trace_lookup_n_vals_for_table_id(table_id)
}

#[inline]
pub fn riscv_trace_shared_width_lookup_backed_cols(layout: &Rv32WidthSidecarLayout) -> Vec<usize> {
    rv32_width_lookup_backed_cols(layout)
}

#[inline]
pub const fn riscv_trace_shared_width_lookup_table_id_for_col(col: usize) -> u32 {
    rv32_width_lookup_table_id_for_col(col)
}

#[inline]
pub fn riscv_trace_shared_width_lookup_transport_n_vals() -> usize {
    rv32_width_lookup_transport_n_vals()
}

#[inline]
pub fn riscv_trace_shared_width_lookup_val_slot_for_col(col: usize) -> Option<usize> {
    rv32_width_lookup_val_slot_for_col(col)
}

#[inline]
pub fn riscv_trace_shared_width_lookup_addr_group_for_table_id(table_id: u32) -> Option<u32> {
    rv32_width_lookup_addr_group_for_table_id(table_id).or_else(|| rv64_width_lookup_addr_group_for_table_id(table_id))
}

#[inline]
pub const fn riscv_trace_uses_shared_width_lookup_table_id(table_id: u32) -> bool {
    rv32_is_width_lookup_table_id(table_id)
}

#[inline]
pub const fn riscv_trace_uses_shared_width_lookup_grouped_table_id(table_id: u32) -> bool {
    rv32_is_width_lookup_grouped_table_id(table_id)
}

#[inline]
pub const fn riscv_trace_is_width_lookup_table_id(table_id: u32) -> bool {
    riscv_trace_uses_shared_width_lookup_table_id(table_id) || rv64_is_width_lookup_table_id(table_id)
}

#[inline]
pub const fn riscv_trace_is_width_lookup_grouped_table_id(table_id: u32) -> bool {
    riscv_trace_uses_shared_width_lookup_grouped_table_id(table_id) || rv64_is_width_lookup_grouped_table_id(table_id)
}
