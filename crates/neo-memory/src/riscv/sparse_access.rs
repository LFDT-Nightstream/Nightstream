//! Sparse access representations for RISC-V sidecars.
//!
//! This module does **not** implement Jolt's full instruction-lookup protocol.
//! It only provides small, reusable building blocks inspired by Jolt's approach:
//! represent read-access patterns as sparse matrices over (address, cycle).
//!
//! These helpers are intended for future Tier-2.1+ work where Shout/ALU sidecars
//! move from packed "bus slices" toward true sparse read matrices (InstructionRa-like).

use neo_math::K;
use p3_field::PrimeCharacteristicRing;

use crate::mle::build_chi_table;
use crate::sparse_matrix::{SparseMat, SparseMatEntry};

use super::exec_table::Rv32ShoutEventTable;

/// Build sparse `(addr, cycle)` matrices for RV32 Shout events.
///
/// - `ra(addr, cycle)` is 1 for each executed lookup event.
/// - `val(addr, cycle)` is the lookup value for each executed event.
///
/// Dimensions:
/// - address domain: 64 bits (`ell_addr = 64`), with `addr = event.key` (interleaved RV32 operands)
/// - cycle domain: `ell_cycle` bits, with `cycle = event.row_idx` (exec-table row index)
pub fn rv32_shout_event_table_to_sparse_ra_and_val(
    events: &Rv32ShoutEventTable,
    ell_cycle: usize,
) -> Result<(SparseMat<K>, SparseMat<K>), String> {
    if ell_cycle >= 64 {
        return Err(format!(
            "rv32_shout_event_table_to_sparse_ra_and_val: ell_cycle={ell_cycle} too large for u64 cycle indices"
        ));
    }
    let max_cycle = 1u64
        .checked_shl(ell_cycle as u32)
        .ok_or_else(|| "rv32_shout_event_table_to_sparse_ra_and_val: 2^ell_cycle overflow".to_string())?;

    let mut ra_entries: Vec<SparseMatEntry<K>> = Vec::with_capacity(events.rows.len());
    let mut val_entries: Vec<SparseMatEntry<K>> = Vec::with_capacity(events.rows.len());

    for row in events.rows.iter() {
        let cycle = u64::try_from(row.row_idx)
            .map_err(|_| "rv32_shout_event_table_to_sparse_ra_and_val: row_idx does not fit u64".to_string())?;
        if cycle >= max_cycle {
            return Err(format!(
                "rv32_shout_event_table_to_sparse_ra_and_val: event row_idx {} out of range for ell_cycle={ell_cycle}",
                row.row_idx
            ));
        }

        let addr = u64::try_from(row.key)
            .map_err(|_| "rv32_shout_event_table_to_sparse_ra_and_val: key does not fit u64".to_string())?;
        ra_entries.push(SparseMatEntry {
            row: addr,
            col: cycle,
            value: K::ONE,
        });
        val_entries.push(SparseMatEntry {
            row: addr,
            col: cycle,
            value: K::from_u64(row.value),
        });
    }

    let ra = SparseMat::from_entries(/*ell_row=*/ 64, /*ell_col=*/ ell_cycle, ra_entries);
    let val = SparseMat::from_entries(/*ell_row=*/ 64, /*ell_col=*/ ell_cycle, val_entries);
    Ok((ra, val))
}

fn chi_at_u64_index(r: &[K], idx: u64) -> K {
    let mut acc = K::ONE;
    for (bit, &ri) in r.iter().enumerate() {
        let is_one = ((idx >> bit) & 1) == 1;
        acc *= if is_one { ri } else { K::ONE - ri };
    }
    acc
}

/// Evaluate the RV32 Shout event-table sparse matrices (RA and VAL) at `(r_addr, r_cycle)`
/// using a Jolt-style chunked address equality table.
///
/// This is purely a helper for future "InstructionRa-like" protocols: it shows how to compute
/// `χ_{r_addr}(key)` without committing to `ell_addr=64` addr-bit columns.
pub fn rv32_shout_event_table_ra_val_mle_eval_chunked(
    events: &Rv32ShoutEventTable,
    r_addr: &[K],
    r_cycle: &[K],
    log_k_chunk: usize,
) -> Result<(K, K), String> {
    if r_addr.len() != 64 {
        return Err(format!(
            "rv32_shout_event_table_ra_val_mle_eval_chunked: expected r_addr.len()=64, got {}",
            r_addr.len()
        ));
    }
    // This helper builds a dense χ table of length 2^log_k_chunk, so keep chunk sizes small.
    // (Jolt commonly uses 8 or 16.)
    if log_k_chunk == 0 || log_k_chunk > 16 {
        return Err(format!(
            "rv32_shout_event_table_ra_val_mle_eval_chunked: log_k_chunk must be in [1,16], got {log_k_chunk}"
        ));
    }
    if 64 % log_k_chunk != 0 {
        return Err(format!(
            "rv32_shout_event_table_ra_val_mle_eval_chunked: log_k_chunk={log_k_chunk} must divide 64"
        ));
    }
    if r_cycle.len() >= 64 {
        return Err(format!(
            "rv32_shout_event_table_ra_val_mle_eval_chunked: r_cycle.len()={} too large for u64 cycle indices",
            r_cycle.len()
        ));
    }

    let mask: u128 = (1u128 << log_k_chunk) - 1;
    let n_chunks = 64 / log_k_chunk;

    let mut eq_evals_by_chunk: Vec<Vec<K>> = Vec::with_capacity(n_chunks);
    for chunk in 0..n_chunks {
        let start = chunk * log_k_chunk;
        let end = start + log_k_chunk;
        eq_evals_by_chunk.push(build_chi_table(&r_addr[start..end]));
    }

    let mut ra_acc = K::ZERO;
    let mut val_acc = K::ZERO;
    for row in events.rows.iter() {
        let cycle = u64::try_from(row.row_idx)
            .map_err(|_| "rv32_shout_event_table_ra_val_mle_eval_chunked: row_idx does not fit u64".to_string())?;
        let w_cycle = chi_at_u64_index(r_cycle, cycle);

        let mut w_addr = K::ONE;
        for (chunk, eq_evals) in eq_evals_by_chunk.iter().enumerate() {
            let shift = chunk * log_k_chunk;
            let idx = ((row.key >> shift) & mask) as usize;
            let w = eq_evals
                .get(idx)
                .copied()
                .ok_or_else(|| "rv32_shout_event_table_ra_val_mle_eval_chunked: chunk idx out of range".to_string())?;
            w_addr *= w;
        }

        let w = w_cycle * w_addr;
        ra_acc += w;
        val_acc += K::from_u64(row.value) * w;
    }

    Ok((ra_acc, val_acc))
}
