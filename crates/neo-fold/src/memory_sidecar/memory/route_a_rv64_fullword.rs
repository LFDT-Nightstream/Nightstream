use std::collections::BTreeMap;

use neo_memory::riscv::lookups::RAM_ID;
use neo_memory::riscv::trace::{
    rv64_width_lookup_backed_cols, rv64_width_lookup_table_id_for_col, rv64_width_lookup_val_slot_for_col,
    Rv64TraceLayout, Rv64WidthSidecarLayout,
};

use super::*;

#[inline]
fn step_has_rv64_width_mem_ops(time_columns: &neo_memory::witness::TimeColumns<F>) -> bool {
    let trace = Rv64TraceLayout::new();
    if time_columns.cpu_cols.len() != trace.cols {
        return false;
    }
    let Some(abs_idx) = time_columns
        .col_ids
        .iter()
        .position(|&id| id == trace.instr_word)
    else {
        return false;
    };
    if abs_idx >= time_columns.cpu_cols.len() {
        return false;
    }
    time_columns.cpu_cols[abs_idx].iter().any(|word_f| {
        let word_u64 = word_f.as_canonical_u64();
        if word_u64 > u32::MAX as u64 {
            return false;
        }
        let word = word_u64 as u32;
        let opcode = word & 0x7f;
        let funct3 = (word >> 12) & 0x7;
        match opcode {
            0x03 => matches!(funct3, 0b000 | 0b001 | 0b010 | 0b011 | 0b100 | 0b101 | 0b110),
            0x23 => matches!(funct3, 0b000 | 0b001 | 0b010 | 0b011),
            _ => false,
        }
    })
}

#[inline]
pub(crate) fn rv64_fullword_width_stage_required_for_step_instance(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    wb_wp_required_for_step_instance(step)
        && step.mem_insts.iter().any(|inst| inst.mem_id == RAM_ID.0)
        && step_has_rv64_width_mem_ops(&step.time_columns)
}

#[inline]
pub(crate) fn rv64_fullword_width_stage_required_for_step_witness(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    wb_wp_required_for_step_witness(step)
        && step
            .mem_instances
            .iter()
            .any(|(inst, _)| inst.mem_id == RAM_ID.0)
        && step_has_rv64_width_mem_ops(&step.time_columns)
}

#[inline]
pub(crate) fn rv64_fullword_width_stage_required_from_proof(
    step: &StepInstanceBundle<Cmt, F, K>,
    batched_time: &crate::shard_proof_types::BatchedTimeProof,
) -> bool {
    step.mem_insts.iter().any(|inst| inst.mem_id == RAM_ID.0)
        && !step
            .lut_insts
            .iter()
            .any(|inst| riscv_trace_uses_shared_width_lookup_table_id(inst.table_id))
        && batched_time
            .labels
            .iter()
            .any(|label| label.as_slice().starts_with(b"width/"))
}

#[inline]
pub(crate) fn rv64_fullword_wp_opening_columns() -> Vec<usize> {
    let trace32 = Rv32TraceLayout::new();
    let trace64 = Rv64TraceLayout::new();
    let mut wp_cols = riscv_trace_wp_opening_columns(&trace32);
    wp_cols.extend([
        trace64.rd_val_lo32,
        trace64.rd_val_hi32,
        trace64.ram_rv_lo32,
        trace64.ram_rv_hi32,
        trace64.ram_wv_lo32,
        trace64.ram_wv_hi32,
        trace64.rs2_val_lo32,
        trace64.rs2_val_hi32,
    ]);
    wp_cols
}

#[inline]
fn rv64_pack_low_bits(bits: &[K], width_bits: usize) -> K {
    let mut coeff = K::ONE;
    let mut out = K::ZERO;
    for &bit in bits.iter().take(width_bits) {
        out += coeff * bit;
        coeff += coeff;
    }
    out
}

#[inline]
fn rv64_sign_extend_to_lo_hi(bits: &[K], width_bits: usize) -> (K, K) {
    let low = rv64_pack_low_bits(bits, width_bits);
    let sign = bits
        .get(width_bits.saturating_sub(1))
        .copied()
        .unwrap_or(K::ZERO);
    let low_sign_mask = if width_bits >= 32 {
        0u64
    } else {
        (u32::MAX << width_bits) as u64
    };
    let hi_sign_mask = u32::MAX as u64;
    (
        low + sign * K::from_u64(low_sign_mask),
        sign * K::from_u64(hi_sign_mask),
    )
}

#[inline]
fn rv64_load_residuals_exact(
    rd_lo: K,
    rd_hi: K,
    ram_rv_lo: K,
    ram_rv_hi: K,
    rd_has_write: K,
    ram_has_read: K,
    load_lb: K,
    load_lh: K,
    load_lw: K,
    load_lbu: K,
    load_lhu: K,
    load_lwu: K,
    load_ld: K,
    ram_rv_low_bits: [K; 32],
) -> [K; 17] {
    let low8 = rv64_pack_low_bits(&ram_rv_low_bits, 8);
    let low16 = rv64_pack_low_bits(&ram_rv_low_bits, 16);
    let low32 = rv64_pack_low_bits(&ram_rv_low_bits, 32);
    let (signext8_lo, signext8_hi) = rv64_sign_extend_to_lo_hi(&ram_rv_low_bits, 8);
    let (signext16_lo, signext16_hi) = rv64_sign_extend_to_lo_hi(&ram_rv_low_bits, 16);
    let (signext32_lo, signext32_hi) = rv64_sign_extend_to_lo_hi(&ram_rv_low_bits, 32);
    let load_gate = load_lb + load_lh + load_lw + load_lbu + load_lhu + load_lwu + load_ld;
    [
        load_gate * (ram_rv_lo - low32),
        load_lb * (rd_lo - signext8_lo),
        load_lb * (rd_hi - signext8_hi),
        load_lh * (rd_lo - signext16_lo),
        load_lh * (rd_hi - signext16_hi),
        load_lw * (rd_lo - signext32_lo),
        load_lw * (rd_hi - signext32_hi),
        load_lbu * (rd_lo - low8),
        load_lbu * rd_hi,
        load_lhu * (rd_lo - low16),
        load_lhu * rd_hi,
        load_lwu * (rd_lo - low32),
        load_lwu * rd_hi,
        load_ld * (rd_lo - ram_rv_lo),
        load_ld * (rd_hi - ram_rv_hi),
        load_gate * (rd_has_write - K::ONE),
        load_gate * (ram_has_read - K::ONE),
    ]
}

#[inline]
fn rv64_store_residuals_exact(
    ram_wv_lo: K,
    ram_wv_hi: K,
    ram_rv_lo: K,
    ram_rv_hi: K,
    rs2_lo: K,
    rs2_hi: K,
    rd_has_write: K,
    ram_has_read: K,
    ram_has_write: K,
    store_sb: K,
    store_sh: K,
    store_sw: K,
    store_sd: K,
    ram_rv_low_bits: [K; 32],
    rs2_low_bits: [K; 32],
) -> [K; 13] {
    let rv_low8 = rv64_pack_low_bits(&ram_rv_low_bits, 8);
    let rv_low16 = rv64_pack_low_bits(&ram_rv_low_bits, 16);
    let rv_low32 = rv64_pack_low_bits(&ram_rv_low_bits, 32);
    let rs2_low8 = rv64_pack_low_bits(&rs2_low_bits, 8);
    let rs2_low16 = rv64_pack_low_bits(&rs2_low_bits, 16);
    let rs2_low32 = rv64_pack_low_bits(&rs2_low_bits, 32);
    let narrow_store_gate = store_sb + store_sh + store_sw;
    let store_gate = narrow_store_gate + store_sd;
    [
        store_gate * (ram_rv_lo - rv_low32),
        store_gate * (rs2_lo - rs2_low32),
        store_sb * (ram_wv_lo - (ram_rv_lo - rv_low8 + rs2_low8)),
        store_sb * (ram_wv_hi - ram_rv_hi),
        store_sh * (ram_wv_lo - (ram_rv_lo - rv_low16 + rs2_low16)),
        store_sh * (ram_wv_hi - ram_rv_hi),
        store_sw * (ram_wv_lo - rs2_low32),
        store_sw * (ram_wv_hi - ram_rv_hi),
        store_sd * (ram_wv_lo - rs2_lo),
        store_sd * (ram_wv_hi - rs2_hi),
        store_gate * rd_has_write,
        narrow_store_gate * (ram_has_read - K::ONE) + store_sd * ram_has_read,
        store_gate * (ram_has_write - K::ONE),
    ]
}

#[inline]
fn rv64_addr_from_bits(bits: &[K]) -> K {
    let mut coeff = K::ONE;
    let mut out = K::ZERO;
    for &bit in bits {
        out += coeff * bit;
        coeff += coeff;
    }
    out
}

#[inline]
fn rv64_selector_linkage_residual(
    cpu_ram_addr: K,
    read_addr_bits: &[K],
    has_read: K,
    write_addr_bits: &[K],
    has_write: K,
    guest_base: u64,
    logical_base: u64,
    cell_bytes: u64,
) -> K {
    let guest_base_k = K::from_u64(guest_base);
    let logical_base_k = K::from_u64(logical_base);
    let cell_bytes_k = K::from_u64(cell_bytes);
    let read_addr = rv64_addr_from_bits(read_addr_bits);
    let write_addr = rv64_addr_from_bits(write_addr_bits);
    let read_residual = cpu_ram_addr - guest_base_k - cell_bytes_k * (read_addr - logical_base_k);
    let write_residual = cpu_ram_addr - guest_base_k - cell_bytes_k * (write_addr - logical_base_k);
    has_read * read_residual + has_write * write_residual
}

struct Rv64RamTwistLaneWitnessView {
    ra_bits: Vec<usize>,
    wa_bits: Vec<usize>,
    has_read: usize,
    has_write: usize,
    decoded: BTreeMap<usize, Vec<K>>,
}

struct Rv64RamTwistLaneOpenings {
    ra_bits: Vec<K>,
    wa_bits: Vec<K>,
    has_read: K,
    has_write: K,
}

fn decode_rv64_ram_twist_lane_witness(
    step: &StepWitnessBundle<Cmt, F, K>,
    t_len: usize,
    label: &str,
) -> Result<Rv64RamTwistLaneWitnessView, PiCcsError> {
    let (ram_mem_idx, _) = step
        .mem_instances
        .iter()
        .enumerate()
        .find(|(_, (inst, _))| inst.mem_id == RAM_ID.0)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("{label}: missing RAM mem instance")))?;
    let bus = build_bus_layout_for_step_witness(step, t_len)?;
    let twist_inst_cols = bus
        .twist_cols
        .get(ram_mem_idx)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("{label}: missing RAM twist bus columns")))?;
    let lane0 = twist_inst_cols
        .lanes
        .first()
        .ok_or_else(|| PiCcsError::ProtocolError(format!("{label}: expected one RAM twist lane")))?;
    let ell_addr = lane0.ra_bits.end - lane0.ra_bits.start;
    if lane0.wa_bits.end - lane0.wa_bits.start != ell_addr {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: RAM twist lane addr widths drift"
        )));
    }

    let mut mem_bus_col_ids = Vec::with_capacity(2 * ell_addr + 2);
    mem_bus_col_ids.extend(lane0.ra_bits.clone());
    mem_bus_col_ids.extend(lane0.wa_bits.clone());
    mem_bus_col_ids.push(lane0.has_read);
    mem_bus_col_ids.push(lane0.has_write);
    let decoded = decode_lookup_backed_col_values_batch(
        t_len,
        bus.bus_cols,
        Some(&step.time_columns.mem_cols),
        &mem_bus_col_ids,
    )?;

    Ok(Rv64RamTwistLaneWitnessView {
        ra_bits: lane0.ra_bits.clone().collect(),
        wa_bits: lane0.wa_bits.clone().collect(),
        has_read: lane0.has_read,
        has_write: lane0.has_write,
        decoded,
    })
}

fn open_rv64_ram_twist_lane(
    cpu_bus: &BusLayout,
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
    label: &str,
) -> Result<Rv64RamTwistLaneOpenings, PiCcsError> {
    let (ram_mem_idx, _) = step
        .mem_insts
        .iter()
        .enumerate()
        .find(|(_, inst)| inst.mem_id == RAM_ID.0)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("{label}: missing RAM mem instance")))?;
    let bus_col_ids = bus_logical_col_ids_for_step_instance(step, cpu_bus, label)?;
    let opening_entry = require_time_opening_entry_for_point(step_time_openings, r_time, &bus_col_ids, label)?;
    if opening_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: requires CommittedOpening source (got {:?})",
            opening_entry.source
        )));
    }
    let logical_map = require_time_openings_for_point(step_time_openings, r_time, &bus_col_ids, label)?;
    let mut bus_time_open_map = BTreeMap::new();
    for (mem_local_col, &logical_col_id) in bus_col_ids.iter().enumerate() {
        let v = logical_map.get(&logical_col_id).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: missing logical opening value for mem_local_col={mem_local_col} logical_col_id={logical_col_id}"
            ))
        })?;
        bus_time_open_map.insert(mem_local_col, v);
    }
    let twist_inst_cols = cpu_bus
        .twist_cols
        .get(ram_mem_idx)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("{label}: missing RAM twist bus columns")))?;
    let lane0 = twist_inst_cols
        .lanes
        .first()
        .ok_or_else(|| PiCcsError::ProtocolError(format!("{label}: expected one RAM twist lane")))?;

    let mut ra_bits = Vec::with_capacity(lane0.ra_bits.end - lane0.ra_bits.start);
    let mut wa_bits = Vec::with_capacity(lane0.wa_bits.end - lane0.wa_bits.start);
    for col_id in lane0.ra_bits.clone() {
        ra_bits.push(named_opening(&bus_time_open_map, col_id, label)?);
    }
    for col_id in lane0.wa_bits.clone() {
        wa_bits.push(named_opening(&bus_time_open_map, col_id, label)?);
    }
    let has_read = named_opening(&bus_time_open_map, lane0.has_read, label)?;
    let has_write = named_opening(&bus_time_open_map, lane0.has_write, label)?;
    Ok(Rv64RamTwistLaneOpenings {
        ra_bits,
        wa_bits,
        has_read,
        has_write,
    })
}

#[inline]
fn time_mem_local_col_for_step(
    step: &StepWitnessBundle<Cmt, F, K>,
    logical_col_id: usize,
    label: &str,
) -> Result<usize, PiCcsError> {
    let cpu_cols_len = step.time_columns.cpu_cols.len();
    let mem_cols_len = step.time_columns.mem_cols.len();
    let total_cols = cpu_cols_len
        .checked_add(mem_cols_len)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: cpu_cols + mem_cols overflow")))?;
    if step.time_columns.col_ids.len() != total_cols {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: time column id table mismatch (col_ids={}, cpu_cols={}, mem_cols={})",
            step.time_columns.col_ids.len(),
            cpu_cols_len,
            mem_cols_len
        )));
    }
    let abs_pos = step
        .time_columns
        .col_ids
        .iter()
        .position(|&id| id == logical_col_id)
        .ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: logical col_id={} is not present in step.time_columns.col_ids",
                logical_col_id
            ))
        })?;
    if abs_pos < cpu_cols_len {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: logical col_id={} resolved to CPU column position {} (expected mem column)",
            logical_col_id, abs_pos
        )));
    }
    let mem_local = abs_pos - cpu_cols_len;
    if mem_local >= mem_cols_len {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: logical col_id={} resolved out of mem column range (mem_local={}, mem_cols={})",
            logical_col_id, mem_local, mem_cols_len
        )));
    }
    Ok(mem_local)
}

#[inline]
fn time_mem_logical_col_id_for_step(
    step: &StepWitnessBundle<Cmt, F, K>,
    mem_local_col: usize,
    label: &str,
) -> Result<usize, PiCcsError> {
    let cpu_cols_len = step.time_columns.cpu_cols.len();
    let mem_cols_len = step.time_columns.mem_cols.len();
    let total_cols = cpu_cols_len
        .checked_add(mem_cols_len)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: cpu_cols + mem_cols overflow")))?;
    if step.time_columns.col_ids.len() != total_cols {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: time column id table mismatch (col_ids={}, cpu_cols={}, mem_cols={})",
            step.time_columns.col_ids.len(),
            cpu_cols_len,
            mem_cols_len
        )));
    }
    let idx = cpu_cols_len
        .checked_add(mem_local_col)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: cpu_cols + mem_local_col overflow")))?;
    step.time_columns.col_ids.get(idx).copied().ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "{label}: missing logical id for mem local col {} (cpu_cols={}, mem_cols={})",
            mem_local_col, cpu_cols_len, mem_cols_len
        ))
    })
}

fn resolve_shared_rv64_width_lookup_lut_indices(
    step: &StepWitnessBundle<Cmt, F, K>,
    width_layout: &Rv64WidthSidecarLayout,
) -> Result<(Vec<usize>, Vec<(usize, usize)>), PiCcsError> {
    let width_open_cols = rv64_width_lookup_backed_cols(width_layout);
    let mut width_lut_slots = Vec::with_capacity(width_open_cols.len());
    for &col_id in &width_open_cols {
        let table_id = rv64_width_lookup_table_id_for_col(col_id);
        let lut_idx = step
            .lut_instances
            .iter()
            .position(|(inst, _)| inst.table_id == table_id)
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W3(rv64): missing RV64 width lookup table_id={table_id} for col_id={col_id}"
                ))
            })?;
        let val_slot = rv64_width_lookup_val_slot_for_col(col_id).ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "W3(rv64): RV64 width col_id={col_id} is not part of width lookup transport slot map"
            ))
        })?;
        width_lut_slots.push((lut_idx, val_slot));
    }
    Ok((width_open_cols, width_lut_slots))
}

fn rv64_width_lookup_bus_val_cols_witness(
    step: &StepWitnessBundle<Cmt, F, K>,
    t_len: usize,
) -> Result<Vec<usize>, PiCcsError> {
    let width = Rv64WidthSidecarLayout::new();
    let (width_cols, width_lut_slots) = resolve_shared_rv64_width_lookup_lut_indices(step, &width)?;
    let mut width_bus_col_by_col = BTreeMap::<usize, usize>::new();
    if step.time_columns.t != t_len || step.time_columns.cpu_cols.is_empty() {
        return Err(PiCcsError::ProtocolError(format!(
            "W3(rv64): canonical time columns required for RV64 width lookup openings (time_t={}, cpu_cols={}, expected_t={t_len})",
            step.time_columns.t,
            step.time_columns.cpu_cols.len()
        )));
    }
    let bus = build_bus_layout_for_step_witness(step, t_len)?;
    if bus.shout_cols.len() != step.lut_instances.len() {
        return Err(PiCcsError::ProtocolError(
            "W3(rv64): bus shout lane count drift while resolving RV64 width lookup columns".into(),
        ));
    }
    for (&width_col_id, &(lut_idx, val_slot)) in width_cols.iter().zip(width_lut_slots.iter()) {
        let inst_cols = bus.shout_cols.get(lut_idx).ok_or_else(|| {
            PiCcsError::ProtocolError("W3(rv64): missing shout cols for RV64 width lookup table".into())
        })?;
        let lane0 = inst_cols.lanes.first().ok_or_else(|| {
            PiCcsError::ProtocolError("W3(rv64): expected one shout lane for RV64 width lookup table".into())
        })?;
        let val_col = lane0.vals.get(val_slot).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "W3(rv64): RV64 width val_slot={} out of range for lut_idx={} (n_vals={})",
                val_slot,
                lut_idx,
                lane0.vals.len()
            ))
        })?;
        let logical_bus_col = time_mem_logical_col_id_for_step(step, val_col, "W3(rv64)")?;
        width_bus_col_by_col.insert(width_col_id, logical_bus_col);
    }
    let mut out = Vec::with_capacity(width_cols.len());
    for &col_id in &width_cols {
        let bus_col = width_bus_col_by_col.get(&col_id).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "W3(rv64): missing RV64 width lookup bus val column for width col_id={col_id}"
            ))
        })?;
        out.push(bus_col);
    }
    Ok(out)
}

pub(crate) fn build_route_a_rv64_fullword_time_claims(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
) -> Result<W3TimeClaims, PiCcsError> {
    if !rv64_fullword_width_stage_required_for_step_witness(step) {
        return Ok((None, None, None, None, None));
    }

    let trace = Rv64TraceLayout::new();
    let width = Rv64WidthSidecarLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &Rv32TraceLayout::new())?;
    let m_in = step.mcs.0.m_in;
    let ell_n = r_cycle.len();

    let main_col_ids = [
        trace.active,
        trace.instr_word,
        trace.ram_addr,
        trace.rd_val_lo32,
        trace.rd_val_hi32,
        trace.ram_rv_lo32,
        trace.ram_rv_hi32,
        trace.ram_wv_lo32,
        trace.ram_wv_hi32,
        trace.rs2_val_lo32,
        trace.rs2_val_hi32,
    ];
    let main_decoded = decode_trace_col_values_batch(params, step, t_len, &main_col_ids)?;

    let width_col_ids = rv64_width_lookup_backed_cols(&width);
    let width_decoded = {
        let width_bus_abs_cols = rv64_width_lookup_bus_val_cols_witness(step, t_len)?;
        let bus = build_bus_layout_for_step_witness(step, t_len)?;
        let mut width_bus_val_cols = Vec::with_capacity(width_bus_abs_cols.len());
        for abs_col in width_bus_abs_cols.iter().copied() {
            let local_col = time_mem_local_col_for_step(step, abs_col, "W3(rv64)")?;
            if local_col >= bus.bus_cols {
                return Err(PiCcsError::ProtocolError(format!(
                    "W3(rv64): RV64 width lookup bus column out of range (local_col={local_col}, bus_cols={})",
                    bus.bus_cols
                )));
            }
            width_bus_val_cols.push(local_col);
        }
        let lookup_vals = decode_lookup_backed_col_values_batch(
            t_len,
            bus.bus_cols,
            Some(&step.time_columns.mem_cols),
            &width_bus_val_cols,
        )?;
        let mut by_col = BTreeMap::<usize, Vec<K>>::new();
        for (idx, &col_id) in width_col_ids.iter().enumerate() {
            let bus_col_id = width_bus_val_cols[idx];
            let vals = lookup_vals.get(&bus_col_id).ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W3(rv64): missing decoded RV64 width lookup values for bus_col={bus_col_id}"
                ))
            })?;
            by_col.insert(col_id, vals.clone());
        }
        by_col
    };

    let ram_twist = decode_rv64_ram_twist_lane_witness(step, t_len, "W3(rv64)")?;
    let ram_twist_has_read_vals = ram_twist
        .decoded
        .get(&ram_twist.has_read)
        .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing RAM twist has_read column".into()))?;
    let ram_twist_has_write_vals = ram_twist
        .decoded
        .get(&ram_twist.has_write)
        .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing RAM twist has_write column".into()))?;

    let decode_col_ids: Vec<usize> = core::iter::once(decode.op_load)
        .chain(core::iter::once(decode.op_store))
        .chain(core::iter::once(decode.rd_has_write))
        .chain(decode.funct3_is.iter().take(7).copied())
        .collect();
    let decode_decoded = {
        let instr_vals = main_decoded
            .get(&trace.instr_word)
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing instr_word decode column".into()))?;
        let active_vals = main_decoded
            .get(&trace.active)
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing active decode column".into()))?;
        if instr_vals.len() != t_len || active_vals.len() != t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "W3(rv64): decoded CPU column lengths drift (instr={}, active={}, t_len={t_len})",
                instr_vals.len(),
                active_vals.len()
            )));
        }
        let mut decoded = BTreeMap::<usize, Vec<K>>::new();
        for &col_id in &decode_col_ids {
            decoded.insert(col_id, Vec::with_capacity(t_len));
        }
        for j in 0..t_len {
            let instr_word = decode_k_to_u32(instr_vals[j], "W3(rv64)/instr_word")?;
            let active = active_vals[j] != K::ZERO;
            let mut row = riscv_decode_lookup_backed_row_from_instr_word(&decode, instr_word, active);
            if !active {
                row.fill(F::ZERO);
            }
            for &col_id in &decode_col_ids {
                decoded
                    .get_mut(&col_id)
                    .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): decode map build failed".into()))?
                    .push(K::from(row[col_id]));
            }
        }
        decoded
    };

    #[cfg(debug_assertions)]
    for j in 0..t_len {
        let rd_lo = *main_decoded
            .get(&trace.rd_val_lo32)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing rd_val_lo32 row while validating".into()))?;
        let rd_hi = *main_decoded
            .get(&trace.rd_val_hi32)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing rd_val_hi32 row while validating".into()))?;
        let ram_rv_lo = *main_decoded
            .get(&trace.ram_rv_lo32)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing ram_rv_lo32 row while validating".into()))?;
        let ram_rv_hi = *main_decoded
            .get(&trace.ram_rv_hi32)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing ram_rv_hi32 row while validating".into()))?;
        let ram_wv_lo = *main_decoded
            .get(&trace.ram_wv_lo32)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing ram_wv_lo32 row while validating".into()))?;
        let ram_wv_hi = *main_decoded
            .get(&trace.ram_wv_hi32)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing ram_wv_hi32 row while validating".into()))?;
        let rs2_lo = *main_decoded
            .get(&trace.rs2_val_lo32)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing rs2_val_lo32 row while validating".into()))?;
        let rs2_hi = *main_decoded
            .get(&trace.rs2_val_hi32)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing rs2_val_hi32 row while validating".into()))?;
        let op_load = *decode_decoded
            .get(&decode.op_load)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing op_load row while validating".into()))?;
        let op_store = *decode_decoded
            .get(&decode.op_store)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing op_store row while validating".into()))?;
        let rd_has_write = *decode_decoded
            .get(&decode.rd_has_write)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing rd_has_write row while validating".into()))?;
        let ram_has_read = *ram_twist_has_read_vals.get(j).ok_or_else(|| {
            PiCcsError::ProtocolError("W3(rv64): missing RAM twist has_read row while validating".into())
        })?;
        let ram_has_write = *ram_twist_has_write_vals.get(j).ok_or_else(|| {
            PiCcsError::ProtocolError("W3(rv64): missing RAM twist has_write row while validating".into())
        })?;
        let funct3_is = [
            *decode_decoded
                .get(&decode.funct3_is[0])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(rv64): missing funct3_is[0] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[1])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(rv64): missing funct3_is[1] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[2])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(rv64): missing funct3_is[2] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[3])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(rv64): missing funct3_is[3] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[4])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(rv64): missing funct3_is[4] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[5])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(rv64): missing funct3_is[5] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[6])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(rv64): missing funct3_is[6] row while validating".into())
                })?,
        ];
        let mut ram_rv_low_bits = [K::ZERO; 32];
        let mut rs2_low_bits = [K::ZERO; 32];
        for bit in 0..32usize {
            ram_rv_low_bits[bit] = *width_decoded
                .get(&width.ram_rv_low_bit[bit])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError(format!("W3(rv64): missing ram_rv_low_bit[{bit}] row while validating"))
                })?;
            rs2_low_bits[bit] = *width_decoded
                .get(&width.rs2_low_bit[bit])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError(format!("W3(rv64): missing rs2_low_bit[{bit}] row while validating"))
                })?;
        }

        let load_residuals = rv64_load_residuals_exact(
            rd_lo,
            rd_hi,
            ram_rv_lo,
            ram_rv_hi,
            rd_has_write,
            ram_has_read,
            op_load * funct3_is[0],
            op_load * funct3_is[1],
            op_load * funct3_is[2],
            op_load * funct3_is[4],
            op_load * funct3_is[5],
            op_load * funct3_is[6],
            op_load * funct3_is[3],
            ram_rv_low_bits,
        );
        if let Some((idx, _)) = load_residuals
            .iter()
            .enumerate()
            .find(|(_, r)| **r != K::ZERO)
        {
            return Err(PiCcsError::ProtocolError(format!(
                "w3(rv64)/load_semantics residual non-zero at row={j}, idx={idx}, op_load={op_load}, funct3_is={:?}, rd_has_write={rd_has_write}, ram_has_read={ram_has_read}",
                funct3_is
            )));
        }

        let store_residuals = rv64_store_residuals_exact(
            ram_wv_lo,
            ram_wv_hi,
            ram_rv_lo,
            ram_rv_hi,
            rs2_lo,
            rs2_hi,
            rd_has_write,
            ram_has_read,
            ram_has_write,
            op_store * funct3_is[0],
            op_store * funct3_is[1],
            op_store * funct3_is[2],
            op_store * funct3_is[3],
            ram_rv_low_bits,
            rs2_low_bits,
        );
        if let Some((idx, _)) = store_residuals
            .iter()
            .enumerate()
            .find(|(_, r)| **r != K::ZERO)
        {
            return Err(PiCcsError::ProtocolError(format!(
                "w3(rv64)/store_semantics residual non-zero at row={j}, idx={idx}, op_store={op_store}, funct3_is={:?}, rd_has_write={rd_has_write}, ram_has_read={ram_has_read}, ram_has_write={ram_has_write}",
                funct3_is
            )));
        }
    }

    let mut main_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in &main_col_ids {
        let vals = main_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3(rv64): missing main decoded column {col_id}")))?;
        main_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let mut width_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in &width_col_ids {
        let vals = width_decoded.get(&col_id).ok_or_else(|| {
            PiCcsError::ProtocolError(format!("W3(rv64): missing RV64 width decoded column {col_id}"))
        })?;
        width_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let mut decode_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in &decode_col_ids {
        let vals = decode_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3(rv64): missing decode decoded column {col_id}")))?;
        decode_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }

    let main_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        main_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3(rv64): missing main sparse column {col_id}")))
    };
    let width_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        width_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3(rv64): missing RV64 width sparse column {col_id}")))
    };
    let decode_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        decode_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3(rv64): missing decode sparse column {col_id}")))
    };
    let ram_twist_has_read_sparse = sparse_trace_col_from_values(m_in, ell_n, ram_twist_has_read_vals)?;
    let ram_twist_has_write_sparse = sparse_trace_col_from_values(m_in, ell_n, ram_twist_has_write_vals)?;

    let bitness_cols: Vec<usize> = width
        .ram_rv_low_bit
        .iter()
        .chain(width.rs2_low_bit.iter())
        .copied()
        .collect();
    let mut bitness_sparse = Vec::with_capacity(bitness_cols.len());
    for &col_id in &bitness_cols {
        bitness_sparse.push(width_col(col_id)?);
    }
    let bitness_weights = w3_bitness_weight_vector(r_cycle, bitness_cols.len());
    let bitness_oracle = FormulaOracleSparseTime::new(bitness_sparse, 3, r_cycle, move |vals: &[K]| {
        let mut weighted = K::ZERO;
        for (bit, weight) in vals.iter().zip(bitness_weights.iter()) {
            weighted += *weight * *bit * (*bit - K::ONE);
        }
        weighted
    });

    let mut quiescence_sparse = Vec::with_capacity(1 + width.cols);
    quiescence_sparse.push(main_col(trace.active)?);
    for &col_id in &width_col_ids {
        quiescence_sparse.push(width_col(col_id)?);
    }
    let quiescence_weights = w3_quiescence_weight_vector(r_cycle, width.cols);
    let quiescence_oracle = FormulaOracleSparseTime::new(quiescence_sparse, 3, r_cycle, move |vals: &[K]| {
        let active = vals[0];
        let mut weighted = K::ZERO;
        for (idx, weight) in quiescence_weights.iter().enumerate() {
            weighted += *weight * vals[1 + idx];
        }
        (K::ONE - active) * weighted
    });

    let selector_linkage_oracle = {
        let (ram_mem_idx_dbg, (ram_inst, _)) = step
            .mem_instances
            .iter()
            .enumerate()
            .find(|(_, (inst, _))| inst.mem_id == RAM_ID.0)
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing RAM mem instance".into()))?;
        #[cfg(not(debug_assertions))]
        let _ = ram_mem_idx_dbg;
        match ram_inst.guest_addr_remap.as_ref() {
            None => None,
            Some(remap) => {
                let ell_addr = ram_twist.ra_bits.len();

                #[cfg(debug_assertions)]
                {
                    let cpu_ram_addr_vals = main_decoded.get(&trace.ram_addr).ok_or_else(|| {
                        PiCcsError::ProtocolError("W3(rv64): missing ram_addr rows while validating".into())
                    })?;
                    for j in 0..t_len {
                        let mut ra_bits = Vec::with_capacity(ell_addr);
                        let mut wa_bits = Vec::with_capacity(ell_addr);
                        for &col_id in &ram_twist.ra_bits {
                            ra_bits.push(
                                *ram_twist
                                    .decoded
                                    .get(&col_id)
                                    .and_then(|v| v.get(j))
                                    .ok_or_else(|| {
                                        PiCcsError::ProtocolError(format!(
                                            "W3(rv64): missing ra_bit column {col_id} row while validating"
                                        ))
                                    })?,
                            );
                        }
                        for &col_id in &ram_twist.wa_bits {
                            wa_bits.push(
                                *ram_twist
                                    .decoded
                                    .get(&col_id)
                                    .and_then(|v| v.get(j))
                                    .ok_or_else(|| {
                                        PiCcsError::ProtocolError(format!(
                                            "W3(rv64): missing wa_bit column {col_id} row while validating"
                                        ))
                                    })?,
                            );
                        }
                        let residual = rv64_selector_linkage_residual(
                            cpu_ram_addr_vals[j],
                            &ra_bits,
                            ram_twist_has_read_vals[j],
                            &wa_bits,
                            ram_twist_has_write_vals[j],
                            remap.guest_base,
                            remap.logical_base,
                            remap.cell_bytes,
                        );
                        if residual != K::ZERO {
                            return Err(PiCcsError::ProtocolError(format!(
                                "w3(rv64)/selector_linkage residual non-zero at row={j}, ram_mem_idx={ram_mem_idx_dbg}"
                            )));
                        }
                    }
                }

                let mut ra_sparse = Vec::with_capacity(ell_addr);
                let mut wa_sparse = Vec::with_capacity(ell_addr);
                for &col_id in &ram_twist.ra_bits {
                    let vals = ram_twist.decoded.get(&col_id).ok_or_else(|| {
                        PiCcsError::ProtocolError(format!("W3(rv64): missing ra_bit column {col_id}"))
                    })?;
                    ra_sparse.push(sparse_trace_col_from_values(m_in, ell_n, vals)?);
                }
                for &col_id in &ram_twist.wa_bits {
                    let vals = ram_twist.decoded.get(&col_id).ok_or_else(|| {
                        PiCcsError::ProtocolError(format!("W3(rv64): missing wa_bit column {col_id}"))
                    })?;
                    wa_sparse.push(sparse_trace_col_from_values(m_in, ell_n, vals)?);
                }

                let mut cols = Vec::with_capacity(3 + 2 * ell_addr);
                cols.push(main_col(trace.ram_addr)?);
                cols.push(ram_twist_has_read_sparse.clone());
                cols.push(ram_twist_has_write_sparse.clone());
                cols.extend(ra_sparse);
                cols.extend(wa_sparse);
                let guest_base = remap.guest_base;
                let logical_base = remap.logical_base;
                let cell_bytes = remap.cell_bytes;
                Some((
                    Box::new(FormulaOracleSparseTime::new(cols, 3, r_cycle, move |vals: &[K]| {
                        let has_read = vals[1];
                        let has_write = vals[2];
                        let ra_bits = &vals[3..3 + ell_addr];
                        let wa_bits = &vals[3 + ell_addr..3 + 2 * ell_addr];
                        rv64_selector_linkage_residual(
                            vals[0],
                            ra_bits,
                            has_read,
                            wa_bits,
                            has_write,
                            guest_base,
                            logical_base,
                            cell_bytes,
                        )
                    })) as Box<dyn RoundOracle + Send>,
                    K::ZERO,
                ))
            }
        }
    };

    let mut load_sparse = Vec::with_capacity(46);
    load_sparse.push(main_col(trace.rd_val_lo32)?);
    load_sparse.push(main_col(trace.rd_val_hi32)?);
    load_sparse.push(main_col(trace.ram_rv_lo32)?);
    load_sparse.push(main_col(trace.ram_rv_hi32)?);
    load_sparse.push(decode_col(decode.rd_has_write)?);
    load_sparse.push(ram_twist_has_read_sparse.clone());
    load_sparse.push(decode_col(decode.op_load)?);
    load_sparse.push(decode_col(decode.funct3_is[0])?);
    load_sparse.push(decode_col(decode.funct3_is[1])?);
    load_sparse.push(decode_col(decode.funct3_is[2])?);
    load_sparse.push(decode_col(decode.funct3_is[3])?);
    load_sparse.push(decode_col(decode.funct3_is[4])?);
    load_sparse.push(decode_col(decode.funct3_is[5])?);
    load_sparse.push(decode_col(decode.funct3_is[6])?);
    for &col_id in &width.ram_rv_low_bit {
        load_sparse.push(width_col(col_id)?);
    }
    let load_weights = w3_load_weight_vector(r_cycle, 17);
    let load_oracle = FormulaOracleSparseTime::new(load_sparse, 5, r_cycle, move |vals: &[K]| {
        let mut ram_rv_low_bits = [K::ZERO; 32];
        ram_rv_low_bits.copy_from_slice(&vals[14..46]);
        let residuals = rv64_load_residuals_exact(
            vals[0],
            vals[1],
            vals[2],
            vals[3],
            vals[4],
            vals[5],
            vals[6] * vals[7],
            vals[6] * vals[8],
            vals[6] * vals[9],
            vals[6] * vals[11],
            vals[6] * vals[12],
            vals[6] * vals[13],
            vals[6] * vals[10],
            ram_rv_low_bits,
        );
        residuals
            .iter()
            .zip(load_weights.iter())
            .fold(K::ZERO, |acc, (residual, weight)| acc + *weight * *residual)
    });

    let mut store_sparse = Vec::with_capacity(78);
    store_sparse.push(main_col(trace.ram_wv_lo32)?);
    store_sparse.push(main_col(trace.ram_wv_hi32)?);
    store_sparse.push(main_col(trace.ram_rv_lo32)?);
    store_sparse.push(main_col(trace.ram_rv_hi32)?);
    store_sparse.push(main_col(trace.rs2_val_lo32)?);
    store_sparse.push(main_col(trace.rs2_val_hi32)?);
    store_sparse.push(decode_col(decode.rd_has_write)?);
    store_sparse.push(ram_twist_has_read_sparse);
    store_sparse.push(ram_twist_has_write_sparse);
    store_sparse.push(decode_col(decode.op_store)?);
    store_sparse.push(decode_col(decode.funct3_is[0])?);
    store_sparse.push(decode_col(decode.funct3_is[1])?);
    store_sparse.push(decode_col(decode.funct3_is[2])?);
    store_sparse.push(decode_col(decode.funct3_is[3])?);
    for &col_id in &width.ram_rv_low_bit {
        store_sparse.push(width_col(col_id)?);
    }
    for &col_id in &width.rs2_low_bit {
        store_sparse.push(width_col(col_id)?);
    }
    let store_weights = w3_store_weight_vector(r_cycle, 13);
    let store_oracle = FormulaOracleSparseTime::new(store_sparse, 4, r_cycle, move |vals: &[K]| {
        let mut ram_rv_low_bits = [K::ZERO; 32];
        ram_rv_low_bits.copy_from_slice(&vals[14..46]);
        let mut rs2_low_bits = [K::ZERO; 32];
        rs2_low_bits.copy_from_slice(&vals[46..78]);
        let residuals = rv64_store_residuals_exact(
            vals[0],
            vals[1],
            vals[2],
            vals[3],
            vals[4],
            vals[5],
            vals[6],
            vals[7],
            vals[8],
            vals[9] * vals[10],
            vals[9] * vals[11],
            vals[9] * vals[12],
            vals[9] * vals[13],
            ram_rv_low_bits,
            rs2_low_bits,
        );
        residuals
            .iter()
            .zip(store_weights.iter())
            .fold(K::ZERO, |acc, (residual, weight)| acc + *weight * *residual)
    });

    Ok((
        Some((Box::new(bitness_oracle), K::ZERO)),
        Some((Box::new(quiescence_oracle), K::ZERO)),
        selector_linkage_oracle,
        Some((Box::new(load_oracle), K::ZERO)),
        Some((Box::new(store_oracle), K::ZERO)),
    ))
}

fn rv64_width_lookup_open_map_from_committed_openings(
    step: &StepInstanceBundle<Cmt, F, K>,
    cpu_bus: &BusLayout,
    r_time: &[K],
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
    label: &str,
) -> Result<BTreeMap<usize, K>, PiCcsError> {
    let width = Rv64WidthSidecarLayout::new();
    let width_open_cols = rv64_width_lookup_backed_cols(&width);
    let mut width_lut_slots = Vec::with_capacity(width_open_cols.len());
    for &col_id in &width_open_cols {
        let table_id = rv64_width_lookup_table_id_for_col(col_id);
        let lut_idx = step
            .lut_insts
            .iter()
            .position(|inst| inst.table_id == table_id)
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "{label}: missing RV64 width lookup table_id={table_id} for col_id={col_id}"
                ))
            })?;
        let val_slot = rv64_width_lookup_val_slot_for_col(col_id).ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: RV64 width col_id={col_id} is not part of width lookup transport slot map"
            ))
        })?;
        width_lut_slots.push((lut_idx, val_slot));
    }
    let bus_col_ids = bus_logical_col_ids_for_step_instance(step, cpu_bus, label)?;
    let logical_map = require_time_openings_for_point(step_time_openings, r_time, &bus_col_ids, label)?;
    let mut out = BTreeMap::<usize, K>::new();
    for (&col_id, &(lut_idx, val_slot)) in width_open_cols.iter().zip(width_lut_slots.iter()) {
        let inst_cols = cpu_bus
            .shout_cols
            .get(lut_idx)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("{label}: missing shout cols for lut_idx={lut_idx}")))?;
        let lane0 = inst_cols.lanes.first().ok_or_else(|| {
            PiCcsError::ProtocolError(format!("{label}: expected one shout lane for lut_idx={lut_idx}"))
        })?;
        let mem_local_col = lane0.vals.get(val_slot).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: RV64 width val_slot={} out of range for lut_idx={} (n_vals={})",
                val_slot,
                lut_idx,
                lane0.vals.len()
            ))
        })?;
        let logical_col_id = bus_col_ids.get(mem_local_col).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: missing logical bus col for mem_local_col={mem_local_col}"
            ))
        })?;
        let value = logical_map.get(&logical_col_id).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: missing opening value for logical_col_id={logical_col_id}"
            ))
        })?;
        out.insert(col_id, value);
    }
    Ok(out)
}

pub(crate) fn verify_route_a_rv64_fullword_terminals(
    cpu_bus: &BusLayout,
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    claim_plan: &RouteATimeClaimPlan,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
) -> Result<(), PiCcsError> {
    if mem_proof.wp_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "W3(rv64) requires WP ME openings for shared main-trace terminals".into(),
        ));
    }

    let trace = Rv64TraceLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();
    let width = Rv64WidthSidecarLayout::new();
    let wp_me = &mem_proof.wp_me_claims[0];
    if wp_me.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "W3(rv64) WP ME claim r mismatch (expected r_time)".into(),
        ));
    }
    if wp_me.c != step.mcs_inst.c {
        return Err(PiCcsError::ProtocolError(
            "W3(rv64) WP ME claim commitment mismatch".into(),
        ));
    }
    if wp_me.m_in != step.mcs_inst.m_in {
        return Err(PiCcsError::ProtocolError("W3(rv64) WP ME claim m_in mismatch".into()));
    }

    let wp_cols = rv64_fullword_wp_opening_columns();
    let (_wp_entry, wp_open_map) =
        require_time_openings_covering_point(step_time_openings, wp_me.r.as_slice(), &wp_cols, "W3(rv64) WP")?;
    let wp_open_col = |col_id: usize| -> Result<K, PiCcsError> { named_opening(&wp_open_map, col_id, "W3(rv64) WP") };
    let decode_open_map =
        decode_lookup_open_map_from_committed_openings(step, cpu_bus, r_time, step_time_openings, "W3(rv64) decode")?;
    let decode_open_col =
        |col_id: usize| -> Result<K, PiCcsError> { named_opening(&decode_open_map, col_id, "W3(rv64) decode") };
    let width_open_map = rv64_width_lookup_open_map_from_committed_openings(
        step,
        cpu_bus,
        r_time,
        step_time_openings,
        "W3(rv64) width",
    )?;
    let width_open_col =
        |col_id: usize| -> Result<K, PiCcsError> { named_opening(&width_open_map, col_id, "W3(rv64) width") };
    let mut ram_rv_low_bits = [K::ZERO; 32];
    let mut rs2_low_bits = [K::ZERO; 32];
    for bit in 0..32usize {
        ram_rv_low_bits[bit] = width_open_col(width.ram_rv_low_bit[bit])?;
        rs2_low_bits[bit] = width_open_col(width.rs2_low_bit[bit])?;
    }

    if let Some(claim_idx) = claim_plan.width_bitness {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w3(rv64)/bitness claim index out of range".into(),
            ));
        }
        let mut bitness_open = Vec::with_capacity(64);
        bitness_open.extend_from_slice(&ram_rv_low_bits);
        bitness_open.extend_from_slice(&rs2_low_bits);
        let weights = w3_bitness_weight_vector(r_cycle, bitness_open.len());
        let mut weighted = K::ZERO;
        for (bit, weight) in bitness_open.iter().zip(weights.iter()) {
            weighted += *weight * *bit * (*bit - K::ONE);
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w3(rv64)/bitness terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.width_quiescence {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w3(rv64)/quiescence claim index out of range".into(),
            ));
        }
        let active = wp_open_col(trace.active)?;
        let mut quiescence_open = Vec::with_capacity(width.cols);
        for &col_id in rv64_width_lookup_backed_cols(&width).iter() {
            quiescence_open.push(width_open_col(col_id)?);
        }
        let weights = w3_quiescence_weight_vector(r_cycle, quiescence_open.len());
        let mut weighted = K::ZERO;
        for (value, weight) in quiescence_open.iter().zip(weights.iter()) {
            weighted += *weight * *value;
        }
        let expected = eq_points(r_time, r_cycle) * (K::ONE - active) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w3(rv64)/quiescence terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.width_selector_linkage {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w3(rv64)/selector_linkage claim index out of range".into(),
            ));
        }
        let (_ram_mem_idx, ram_inst) = step
            .mem_insts
            .iter()
            .enumerate()
            .find(|(_, inst)| inst.mem_id == RAM_ID.0)
            .ok_or_else(|| PiCcsError::ProtocolError("W3(rv64): missing RAM mem instance".into()))?;
        let remap = ram_inst.guest_addr_remap.as_ref().ok_or_else(|| {
            PiCcsError::ProtocolError(
                "w3(rv64)/selector_linkage scheduled without RAM guest_addr_remap metadata".into(),
            )
        })?;
        let twist_open =
            open_rv64_ram_twist_lane(cpu_bus, step, r_time, step_time_openings, "W3(rv64)/selector_linkage")?;
        let residual = rv64_selector_linkage_residual(
            wp_open_col(trace.ram_addr)?,
            &twist_open.ra_bits,
            twist_open.has_read,
            &twist_open.wa_bits,
            twist_open.has_write,
            remap.guest_base,
            remap.logical_base,
            remap.cell_bytes,
        );
        let expected = eq_points(r_time, r_cycle) * residual;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w3(rv64)/selector_linkage terminal value mismatch".into(),
            ));
        }
    }

    let rd_lo = wp_open_col(trace.rd_val_lo32)?;
    let rd_hi = wp_open_col(trace.rd_val_hi32)?;
    let ram_rv_lo = wp_open_col(trace.ram_rv_lo32)?;
    let ram_rv_hi = wp_open_col(trace.ram_rv_hi32)?;
    let ram_wv_lo = wp_open_col(trace.ram_wv_lo32)?;
    let ram_wv_hi = wp_open_col(trace.ram_wv_hi32)?;
    let rs2_lo = wp_open_col(trace.rs2_val_lo32)?;
    let rs2_hi = wp_open_col(trace.rs2_val_hi32)?;
    let rd_has_write = decode_open_col(decode.rd_has_write)?;
    let twist_open = open_rv64_ram_twist_lane(cpu_bus, step, r_time, step_time_openings, "W3(rv64)/mem_semantics")?;
    let op_load = decode_open_col(decode.op_load)?;
    let op_store = decode_open_col(decode.op_store)?;
    if let Some(claim_idx) = claim_plan.width_load_semantics {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w3(rv64)/load_semantics claim index out of range".into(),
            ));
        }
        let residuals = rv64_load_residuals_exact(
            rd_lo,
            rd_hi,
            ram_rv_lo,
            ram_rv_hi,
            rd_has_write,
            twist_open.has_read,
            op_load * decode_open_col(decode.funct3_is[0])?,
            op_load * decode_open_col(decode.funct3_is[1])?,
            op_load * decode_open_col(decode.funct3_is[2])?,
            op_load * decode_open_col(decode.funct3_is[4])?,
            op_load * decode_open_col(decode.funct3_is[5])?,
            op_load * decode_open_col(decode.funct3_is[6])?,
            op_load * decode_open_col(decode.funct3_is[3])?,
            ram_rv_low_bits,
        );
        let weights = w3_load_weight_vector(r_cycle, residuals.len());
        let weighted = residuals
            .iter()
            .zip(weights.iter())
            .fold(K::ZERO, |acc, (residual, weight)| acc + *weight * *residual);
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w3(rv64)/load_semantics terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.width_store_semantics {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w3(rv64)/store_semantics claim index out of range".into(),
            ));
        }
        let residuals = rv64_store_residuals_exact(
            ram_wv_lo,
            ram_wv_hi,
            ram_rv_lo,
            ram_rv_hi,
            rs2_lo,
            rs2_hi,
            rd_has_write,
            twist_open.has_read,
            twist_open.has_write,
            op_store * decode_open_col(decode.funct3_is[0])?,
            op_store * decode_open_col(decode.funct3_is[1])?,
            op_store * decode_open_col(decode.funct3_is[2])?,
            op_store * decode_open_col(decode.funct3_is[3])?,
            ram_rv_low_bits,
            rs2_low_bits,
        );
        let weights = w3_store_weight_vector(r_cycle, residuals.len());
        let weighted = residuals
            .iter()
            .zip(weights.iter())
            .fold(K::ZERO, |acc, (residual, weight)| acc + *weight * *residual);
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w3(rv64)/store_semantics terminal value mismatch".into(),
            ));
        }
    }

    Ok(())
}
