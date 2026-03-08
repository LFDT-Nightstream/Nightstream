use neo_memory::riscv::lookups::REG_EXACT_ID;
use neo_memory::riscv::trace::Rv64TraceLayout;
use neo_memory::sparse_time::SparseIdxVec;
use neo_reductions::sumcheck::RoundOracle;

use super::*;

pub(crate) const RV64_REG_EXACT_LINKAGE_DEGREE_BOUND: usize = 3;

#[inline]
fn reg_exact_addr_from_bits(bits: &[K]) -> K {
    let mut coeff = K::ONE;
    let mut out = K::ZERO;
    for &bit in bits {
        out += coeff * bit;
        coeff += coeff;
    }
    out
}

#[inline]
fn reg_exact_linkage_residual(
    rd_addr: K,
    rd_has_write: K,
    is_virtual: K,
    rd_lo: K,
    rd_hi: K,
    lane0_has_read: K,
    lane0_has_write: K,
    lane0_wv: K,
    lane0_wa_bits: &[K],
    lane1_has_read: K,
    lane1_has_write: K,
    lane1_wv: K,
    lane1_wa_bits: &[K],
) -> K {
    let arch_has_write = rd_has_write * (K::ONE - is_virtual);
    let lane0_addr = reg_exact_addr_from_bits(lane0_wa_bits);
    let lane1_addr = reg_exact_addr_from_bits(lane1_wa_bits);
    lane0_has_read
        + lane1_has_read
        + (lane0_has_write - arch_has_write)
        + (lane1_has_write - arch_has_write)
        + lane0_has_write * (lane0_addr - rd_addr)
        + lane1_has_write * (lane1_addr - (rd_addr + K::from_u64(32)))
        + lane0_has_write * (lane0_wv - rd_lo)
        + lane1_has_write * (lane1_wv - rd_hi)
}

pub(crate) fn build_rv64_reg_exact_output_linkage_claim(
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
) -> Result<Option<(Box<dyn RoundOracle + Send>, K)>, PiCcsError> {
    let (mem_idx, _) = match step
        .mem_instances
        .iter()
        .enumerate()
        .find(|(_, (inst, _))| inst.mem_id == REG_EXACT_ID.0)
    {
        Some(found) => found,
        None => return Ok(None),
    };

    let trace = Rv64TraceLayout::new();
    if step.time_columns.cpu_cols.len() != trace.cols {
        return Err(PiCcsError::ProtocolError(
            "RV64 exact reg output linkage requires RV64 trace CPU columns".into(),
        ));
    }
    let t_len = step.time_columns.t;
    let m_in = step.mcs.0.m_in;
    let ell_n = r_cycle.len();

    let main_decoded = {
        let mut decoded = BTreeMap::<usize, Vec<K>>::new();
        for &col_id in &[
            trace.rd_addr,
            trace.rd_has_write,
            trace.is_virtual,
            trace.rd_val_lo32,
            trace.rd_val_hi32,
        ] {
            let vals = step.time_columns.cpu_cols.get(col_id).ok_or_else(|| {
                PiCcsError::ProtocolError(format!("RV64 exact reg linkage missing cpu column {col_id}"))
            })?;
            if vals.len() != t_len {
                return Err(PiCcsError::ProtocolError(format!(
                    "RV64 exact reg linkage cpu column length mismatch for col_id={col_id} (len={}, t_len={t_len})",
                    vals.len()
                )));
            }
            decoded.insert(col_id, vals.iter().copied().map(K::from).collect());
        }
        decoded
    };
    let rd_addr_vals = main_decoded
        .get(&trace.rd_addr)
        .ok_or_else(|| PiCcsError::ProtocolError("RV64 exact reg linkage missing rd_addr".into()))?;
    let rd_has_write_vals = main_decoded
        .get(&trace.rd_has_write)
        .ok_or_else(|| PiCcsError::ProtocolError("RV64 exact reg linkage missing rd_has_write".into()))?;
    let is_virtual_vals = main_decoded
        .get(&trace.is_virtual)
        .ok_or_else(|| PiCcsError::ProtocolError("RV64 exact reg linkage missing is_virtual".into()))?;
    let rd_lo_vals = main_decoded
        .get(&trace.rd_val_lo32)
        .ok_or_else(|| PiCcsError::ProtocolError("RV64 exact reg linkage missing rd_val_lo32".into()))?;
    let rd_hi_vals = main_decoded
        .get(&trace.rd_val_hi32)
        .ok_or_else(|| PiCcsError::ProtocolError("RV64 exact reg linkage missing rd_val_hi32".into()))?;

    let bus = build_bus_layout_for_step_witness(step, t_len)?;
    let twist_inst_cols = bus
        .twist_cols
        .get(mem_idx)
        .ok_or_else(|| PiCcsError::ProtocolError("RV64 exact reg linkage missing twist bus columns".into()))?;
    if twist_inst_cols.lanes.len() < 2 {
        return Err(PiCcsError::ProtocolError(
            "RV64 exact reg linkage requires two twist lanes".into(),
        ));
    }
    let lane0 = &twist_inst_cols.lanes[0];
    let lane1 = &twist_inst_cols.lanes[1];
    let ell_addr = lane0.wa_bits.end - lane0.wa_bits.start;
    if ell_addr != 6 || lane1.wa_bits.end - lane1.wa_bits.start != ell_addr {
        return Err(PiCcsError::ProtocolError(
            "RV64 exact reg linkage requires 6-bit addresses on both lanes".into(),
        ));
    }

    let mem_col_values = |col_id: usize| -> Result<&Vec<F>, PiCcsError> {
        step.time_columns
            .mem_cols
            .get(col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("RV64 exact reg linkage missing mem column {col_id}")))
    };
    let sparse_from_mem = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        let vals = mem_col_values(col_id)?;
        let vals_k: Vec<K> = vals.iter().map(|v| K::from(*v)).collect();
        sparse_trace_col_from_values(m_in, ell_n, &vals_k)
    };

    #[cfg(debug_assertions)]
    {
        let row_mem_value = |col_id: usize, row: usize| -> Result<K, PiCcsError> {
            let col = mem_col_values(col_id)?;
            let value = col.get(row).ok_or_else(|| {
                PiCcsError::ProtocolError(format!("RV64 exact reg linkage mem column {col_id} missing row {row}"))
            })?;
            Ok(K::from(*value))
        };
        for j in 0..t_len {
            let lane0_bits: Vec<K> = lane0
                .wa_bits
                .clone()
                .map(|col_id| row_mem_value(col_id, j))
                .collect::<Result<Vec<_>, _>>()?;
            let lane1_bits: Vec<K> = lane1
                .wa_bits
                .clone()
                .map(|col_id| row_mem_value(col_id, j))
                .collect::<Result<Vec<_>, _>>()?;
            let residual = reg_exact_linkage_residual(
                rd_addr_vals[j],
                rd_has_write_vals[j],
                is_virtual_vals[j],
                rd_lo_vals[j],
                rd_hi_vals[j],
                row_mem_value(lane0.has_read, j)?,
                row_mem_value(lane0.has_write, j)?,
                row_mem_value(lane0.wv, j)?,
                &lane0_bits,
                row_mem_value(lane1.has_read, j)?,
                row_mem_value(lane1.has_write, j)?,
                row_mem_value(lane1.wv, j)?,
                &lane1_bits,
            );
            if residual != K::ZERO {
                return Err(PiCcsError::ProtocolError(format!(
                    "RV64 exact reg linkage residual non-zero at row={j}"
                )));
            }
        }
    }

    let mut cols = Vec::with_capacity(10 + 2 * ell_addr);
    cols.push(sparse_trace_col_from_values(m_in, ell_n, rd_addr_vals)?);
    cols.push(sparse_trace_col_from_values(m_in, ell_n, rd_has_write_vals)?);
    cols.push(sparse_trace_col_from_values(m_in, ell_n, is_virtual_vals)?);
    cols.push(sparse_trace_col_from_values(m_in, ell_n, rd_lo_vals)?);
    cols.push(sparse_trace_col_from_values(m_in, ell_n, rd_hi_vals)?);
    cols.push(sparse_from_mem(lane0.has_read)?);
    cols.push(sparse_from_mem(lane0.has_write)?);
    cols.push(sparse_from_mem(lane0.wv)?);
    for col_id in lane0.wa_bits.clone() {
        cols.push(sparse_from_mem(col_id)?);
    }
    cols.push(sparse_from_mem(lane1.has_read)?);
    cols.push(sparse_from_mem(lane1.has_write)?);
    cols.push(sparse_from_mem(lane1.wv)?);
    for col_id in lane1.wa_bits.clone() {
        cols.push(sparse_from_mem(col_id)?);
    }

    let oracle = FormulaOracleSparseTime::new(
        cols,
        RV64_REG_EXACT_LINKAGE_DEGREE_BOUND,
        r_cycle,
        move |vals: &[K]| {
            let rd_addr = vals[0];
            let rd_has_write = vals[1];
            let is_virtual = vals[2];
            let rd_lo = vals[3];
            let rd_hi = vals[4];
            let lane0_has_read = vals[5];
            let lane0_has_write = vals[6];
            let lane0_wv = vals[7];
            let lane0_bits = &vals[8..8 + ell_addr];
            let lane1_base = 8 + ell_addr;
            let lane1_has_read = vals[lane1_base];
            let lane1_has_write = vals[lane1_base + 1];
            let lane1_wv = vals[lane1_base + 2];
            let lane1_bits = &vals[lane1_base + 3..lane1_base + 3 + ell_addr];
            reg_exact_linkage_residual(
                rd_addr,
                rd_has_write,
                is_virtual,
                rd_lo,
                rd_hi,
                lane0_has_read,
                lane0_has_write,
                lane0_wv,
                lane0_bits,
                lane1_has_read,
                lane1_has_write,
                lane1_wv,
                lane1_bits,
            )
        },
    );
    Ok(Some((Box::new(oracle), K::ZERO)))
}

pub(crate) fn verify_rv64_reg_exact_output_linkage_terminal(
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    claim_idx: usize,
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
    twist_openings: &[TwistTimeLaneOpenings],
    exact_reg_mem_idx: usize,
) -> Result<(), PiCcsError> {
    let trace = Rv64TraceLayout::new();
    let required_cols = [
        trace.rd_addr,
        trace.rd_has_write,
        trace.is_virtual,
        trace.rd_val_lo32,
        trace.rd_val_hi32,
    ];
    let mut matched_maps = Vec::new();
    for opening in step_time_openings.iter() {
        if opening.point.as_slice() != r_time {
            continue;
        }
        if !required_cols
            .iter()
            .all(|col_id| opening.col_ids.contains(col_id))
        {
            continue;
        }
        if opening.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "output_binding/reg_exact_linkage requires CommittedOpening source (got {:?})",
                opening.source
            )));
        }
        if opening.col_ids.len() != opening.evals.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "output_binding/reg_exact_linkage malformed opening (col_ids={}, evals={})",
                opening.col_ids.len(),
                opening.evals.len()
            )));
        }
        let map: BTreeMap<usize, K> = opening
            .col_ids
            .iter()
            .copied()
            .zip(opening.evals.iter().copied())
            .collect();
        let vals = [
            named_opening(&map, trace.rd_addr, "output_binding/reg_exact_linkage")?,
            named_opening(&map, trace.rd_has_write, "output_binding/reg_exact_linkage")?,
            named_opening(&map, trace.is_virtual, "output_binding/reg_exact_linkage")?,
            named_opening(&map, trace.rd_val_lo32, "output_binding/reg_exact_linkage")?,
            named_opening(&map, trace.rd_val_hi32, "output_binding/reg_exact_linkage")?,
        ];
        matched_maps.push((map, vals));
    }
    if matched_maps.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "output_binding/reg_exact_linkage missing covering time opening".into(),
        ));
    }
    let (_, expected_vals) = &matched_maps[0];
    for (_, vals) in matched_maps.iter().skip(1) {
        if vals != expected_vals {
            return Err(PiCcsError::ProtocolError(
                "output_binding/reg_exact_linkage found inconsistent covering time openings".into(),
            ));
        }
    }
    let opening_map = &matched_maps[0].0;

    let twist_open = twist_openings
        .get(exact_reg_mem_idx)
        .ok_or_else(|| PiCcsError::ProtocolError("output_binding/reg_exact_linkage missing twist openings".into()))?;
    if twist_open.lanes.len() < 2 {
        return Err(PiCcsError::ProtocolError(
            "output_binding/reg_exact_linkage requires two exact-reg lanes".into(),
        ));
    }
    let lane0 = &twist_open.lanes[0];
    let lane1 = &twist_open.lanes[1];
    let residual = reg_exact_linkage_residual(
        named_opening(&opening_map, trace.rd_addr, "output_binding/reg_exact_linkage")?,
        named_opening(&opening_map, trace.rd_has_write, "output_binding/reg_exact_linkage")?,
        named_opening(&opening_map, trace.is_virtual, "output_binding/reg_exact_linkage")?,
        named_opening(&opening_map, trace.rd_val_lo32, "output_binding/reg_exact_linkage")?,
        named_opening(&opening_map, trace.rd_val_hi32, "output_binding/reg_exact_linkage")?,
        lane0.has_read,
        lane0.has_write,
        lane0.wv,
        &lane0.wa_bits,
        lane1.has_read,
        lane1.has_write,
        lane1.wv,
        &lane1.wa_bits,
    );
    let expected = eq_points(r_time, r_cycle) * residual;
    if batched_final_values.get(claim_idx).copied() != Some(expected) {
        return Err(PiCcsError::ProtocolError(
            "output_binding/reg_exact_linkage terminal value mismatch".into(),
        ));
    }
    if step
        .mem_insts
        .get(exact_reg_mem_idx)
        .map(|inst| inst.mem_id)
        != Some(REG_EXACT_ID.0)
    {
        return Err(PiCcsError::ProtocolError(
            "output_binding/reg_exact_linkage mem_idx is not REG_EXACT_ID".into(),
        ));
    }
    Ok(())
}
