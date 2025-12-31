use std::collections::HashMap;

use crate::plain::PlainMemLayout;

use super::constants::{ADD_TABLE_ID, REMU_TABLE_ID, RV32_XLEN};

pub(super) fn derive_mem_ids_and_ell_addrs(mem_layouts: &HashMap<u32, PlainMemLayout>) -> Result<(Vec<u32>, Vec<usize>), String> {
    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();

    let mut twist_ell_addrs = Vec::with_capacity(mem_ids.len());
    for mem_id in &mem_ids {
        let layout = mem_layouts
            .get(mem_id)
            .ok_or_else(|| format!("missing mem_layout for mem_id={mem_id}"))?;
        if layout.n_side == 0 || !layout.n_side.is_power_of_two() {
            return Err(format!("mem_id={mem_id}: n_side={} must be power of two", layout.n_side));
        }
        let ell = layout.n_side.trailing_zeros() as usize;
        twist_ell_addrs.push(layout.d * ell);
    }

    Ok((mem_ids, twist_ell_addrs))
}

pub(super) fn derive_shout_ids_and_ell_addrs(shout_table_ids: &[u32]) -> Result<(Vec<u32>, Vec<usize>), String> {
    let mut table_ids: Vec<u32> = shout_table_ids.to_vec();
    table_ids.sort_unstable();
    table_ids.dedup();
    if table_ids.is_empty() {
        return Err("RV32 B1: shout_table_ids must be non-empty".into());
    }
    if !table_ids.contains(&ADD_TABLE_ID) {
        return Err(format!("RV32 B1: shout_table_ids must include ADD table_id={ADD_TABLE_ID}"));
    }
    // This circuit supports RV32IM via the 20 base+M opcode tables (ids 0..=19). Callers may pass any
    // subset, as long as it covers the opcodes that will actually appear in the VM trace.
    // (Missing table specs are rejected by `build_shard_witness_shared_cpu_bus` when the trace contains
    // a Shout event for an unlisted `table_id`.)
    for &table_id in &table_ids {
        if table_id > REMU_TABLE_ID {
            return Err(format!(
                "RV32 B1: unsupported table_id={table_id} (expected RISC-V opcode table ids 0..={REMU_TABLE_ID})"
            ));
        }
    }
    // MVP: every Shout table in this circuit is a RISC-V opcode table with d=2*xlen, n_side=2 => ell_addr=2*xlen.
    let shout_ell_addrs = vec![2 * RV32_XLEN; table_ids.len()];
    Ok((table_ids, shout_ell_addrs))
}
