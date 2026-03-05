use super::*;

struct DenseCols<T> {
    cols: Vec<Option<Vec<T>>>,
}

impl<T> DenseCols<T> {
    fn from_cols(cols: Vec<Vec<T>>) -> Self {
        Self {
            cols: cols.into_iter().map(Some).collect(),
        }
    }

    fn get(&self, col_id: &usize) -> Option<&Vec<T>> {
        self.cols.get(*col_id).and_then(|v| v.as_ref())
    }

    fn get_mut(&mut self, col_id: &usize) -> Option<&mut Vec<T>> {
        self.cols.get_mut(*col_id).and_then(|v| v.as_mut())
    }

    fn insert(&mut self, col_id: usize, vals: Vec<T>) {
        if col_id >= self.cols.len() {
            self.cols.resize_with(col_id + 1, || None);
        }
        self.cols[col_id] = Some(vals);
    }
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

pub(crate) fn width_lookup_bus_val_cols_witness(
    step: &StepWitnessBundle<Cmt, F, K>,
    t_len: usize,
) -> Result<Vec<usize>, PiCcsError> {
    let width = Rv32WidthSidecarLayout::new();
    let (width_cols, width_lut_slots) = resolve_shared_width_lookup_lut_indices(step, &width)?;
    let mut width_bus_col_by_col: BTreeMap<usize, usize> = BTreeMap::new();
    if step.time_columns.t != t_len || step.time_columns.cpu_cols.is_empty() {
        return Err(PiCcsError::ProtocolError(format!(
            "W3(shared): canonical time columns required for width lookup openings (time_t={}, cpu_cols={}, expected_t={t_len})",
            step.time_columns.t,
            step.time_columns.cpu_cols.len()
        )));
    }
    let bus = build_bus_layout_for_step_witness(step, t_len)?;
    if bus.shout_cols.len() != step.lut_instances.len() {
        return Err(PiCcsError::ProtocolError(
            "W3(shared): bus shout lane count drift while resolving width lookup columns".into(),
        ));
    }
    for (&width_col_id, &(lut_idx, val_slot)) in width_cols.iter().zip(width_lut_slots.iter()) {
        let inst_cols = bus
            .shout_cols
            .get(lut_idx)
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing shout cols for width lookup table".into()))?;
        let lane0 = inst_cols.lanes.get(0).ok_or_else(|| {
            PiCcsError::ProtocolError("W3(shared): expected one shout lane for width lookup table".into())
        })?;
        let val_col = lane0.vals.get(val_slot).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "W3(shared): width val_slot={} out of range for lut_idx={} (n_vals={})",
                val_slot,
                lut_idx,
                lane0.vals.len()
            ))
        })?;
        let logical_bus_col = time_mem_logical_col_id_for_step(step, val_col, "W3(shared)")?;
        width_bus_col_by_col.insert(width_col_id, logical_bus_col);
    }
    let mut out = Vec::with_capacity(width_cols.len());
    for &col_id in width_cols.iter() {
        let bus_col = width_bus_col_by_col.get(&col_id).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "W3(shared): missing width lookup bus val column for width col_id={col_id}"
            ))
        })?;
        out.push(bus_col);
    }
    Ok(out)
}

pub(crate) fn build_route_a_width_time_claims(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
) -> Result<W3TimeClaims, PiCcsError> {
    if !width_stage_required_for_step_witness(step) {
        return Ok((None, None, None, None, None));
    }
    let trace = Rv32TraceLayout::new();
    let width = Rv32WidthSidecarLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();
    let m_in = step.mcs.0.m_in;
    let ell_n = r_cycle.len();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("W3: t_len must be >= 1".into()));
    }

    let main_col_ids = [
        trace.active,
        trace.instr_word,
        trace.rd_val,
        trace.ram_rv,
        trace.ram_wv,
        trace.rs2_val,
    ];
    let main_decoded = decode_trace_col_values_batch(params, step, t_len, &main_col_ids)?;
    let width_col_ids = rv32_width_lookup_backed_cols(&width);
    let width_decoded: DenseCols<K> = {
        let width_bus_abs_cols = width_lookup_bus_val_cols_witness(step, t_len)?;
        let bus = build_bus_layout_for_step_witness(step, t_len)?;
        let mut width_bus_val_cols = Vec::with_capacity(width_bus_abs_cols.len());
        for abs_col in width_bus_abs_cols.iter().copied() {
            let local_col = time_mem_local_col_for_step(step, abs_col, "W3(shared)")?;
            if local_col >= bus.bus_cols {
                return Err(PiCcsError::ProtocolError(format!(
                    "W3(shared): width lookup bus column out of range (local_col={local_col}, bus_cols={})",
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
        let mut by_col = DenseCols::from_cols(Vec::new());
        for (idx, &col_id) in width_col_ids.iter().enumerate() {
            let bus_col_id = width_bus_val_cols[idx];
            let vals = lookup_vals.get(&bus_col_id).ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W3(shared): missing decoded lookup values for bus_col={bus_col_id}"
                ))
            })?;
            by_col.insert(col_id, vals.clone());
        }
        by_col
    };
    let decode_col_ids: Vec<usize> = core::iter::once(decode.op_load)
        .chain(core::iter::once(decode.op_store))
        .chain(core::iter::once(decode.rd_has_write))
        .chain(core::iter::once(decode.ram_has_read))
        .chain(core::iter::once(decode.ram_has_write))
        .chain(decode.funct3_is.iter().copied())
        .collect();
    let decode_decoded = {
        let instr_vals = main_decoded
            .get(&trace.instr_word)
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing instr_word decode column".into()))?;
        let active_vals = main_decoded
            .get(&trace.active)
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing active decode column".into()))?;
        if instr_vals.len() != t_len || active_vals.len() != t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "W3(shared): decoded CPU column lengths drift (instr={}, active={}, t_len={t_len})",
                instr_vals.len(),
                active_vals.len()
            )));
        }
        let mut decoded = DenseCols::from_cols(Vec::new());
        for &col_id in decode_col_ids.iter() {
            decoded.insert(col_id, Vec::with_capacity(t_len));
        }
        for j in 0..t_len {
            let instr_word = decode_k_to_u32(instr_vals[j], "W3(shared)/instr_word")?;
            let active = active_vals[j] != K::ZERO;
            let mut row = rv32_decode_lookup_backed_row_from_instr_word(&decode, instr_word, active);
            if !active {
                row.fill(F::ZERO);
            }
            for &col_id in decode_col_ids.iter() {
                decoded
                    .get_mut(&col_id)
                    .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): decode map build failed".into()))?
                    .push(K::from(row[col_id]));
            }
        }
        decoded
    };

    // Defensive row-wise validation: width-sidecar residual families must vanish before
    // entering batched sumcheck. This turns opaque aggregate failures into actionable row/idx errors.
    #[cfg(debug_assertions)]
    for j in 0..t_len {
        let rd_val = *main_decoded
            .get(&trace.rd_val)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing rd_val row while validating".into()))?;
        let ram_rv = *main_decoded
            .get(&trace.ram_rv)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing ram_rv row while validating".into()))?;
        let ram_wv = *main_decoded
            .get(&trace.ram_wv)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing ram_wv row while validating".into()))?;
        let rs2_val = *main_decoded
            .get(&trace.rs2_val)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing rs2_val row while validating".into()))?;
        let active = *main_decoded
            .get(&trace.active)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing active row while validating".into()))?;

        let rd_has_write = *decode_decoded
            .get(&decode.rd_has_write)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing rd_has_write row while validating".into()))?;
        let ram_has_read = *decode_decoded
            .get(&decode.ram_has_read)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing ram_has_read row while validating".into()))?;
        let ram_has_write = *decode_decoded
            .get(&decode.ram_has_write)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("W3(shared): missing ram_has_write row while validating".into())
            })?;
        let op_load = *decode_decoded
            .get(&decode.op_load)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing op_load row while validating".into()))?;
        let op_store = *decode_decoded
            .get(&decode.op_store)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing op_store row while validating".into()))?;
        let funct3_is = [
            *decode_decoded
                .get(&decode.funct3_is[0])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(shared): missing funct3_is[0] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[1])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(shared): missing funct3_is[1] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[2])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(shared): missing funct3_is[2] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[3])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(shared): missing funct3_is[3] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[4])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(shared): missing funct3_is[4] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[5])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(shared): missing funct3_is[5] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[6])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(shared): missing funct3_is[6] row while validating".into())
                })?,
            *decode_decoded
                .get(&decode.funct3_is[7])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError("W3(shared): missing funct3_is[7] row while validating".into())
                })?,
        ];
        let ram_rv_q16 = *width_decoded
            .get(&width.ram_rv_q16)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing ram_rv_q16 row while validating".into()))?;
        let rs2_q16 = *width_decoded
            .get(&width.rs2_q16)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing rs2_q16 row while validating".into()))?;
        let mut ram_rv_low_bits = [K::ZERO; 16];
        let mut rs2_low_bits = [K::ZERO; 16];
        for bit in 0..16usize {
            ram_rv_low_bits[bit] = *width_decoded
                .get(&width.ram_rv_low_bit[bit])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError(format!(
                        "W3(shared): missing ram_rv_low_bit[{bit}] row while validating"
                    ))
                })?;
            rs2_low_bits[bit] = *width_decoded
                .get(&width.rs2_low_bit[bit])
                .and_then(|v| v.get(j))
                .ok_or_else(|| {
                    PiCcsError::ProtocolError(format!("W3(shared): missing rs2_low_bit[{bit}] row while validating"))
                })?;
        }

        let load_flags = [
            op_load * funct3_is[0],
            op_load * funct3_is[4],
            op_load * funct3_is[1],
            op_load * funct3_is[5],
            op_load * funct3_is[2],
        ];
        let load_residuals = w3_load_semantics_residuals(
            rd_val,
            ram_rv,
            rd_has_write,
            ram_has_read,
            load_flags,
            ram_rv_q16,
            ram_rv_low_bits,
        );
        if let Some((idx, _)) = load_residuals
            .iter()
            .enumerate()
            .find(|(_, r)| **r != K::ZERO)
        {
            return Err(PiCcsError::ProtocolError(format!(
                "w3/load_semantics residual non-zero at row={j}, idx={idx}, active={active}, op_load={op_load}, funct3_is={:?}, rd_has_write={rd_has_write}, ram_has_read={ram_has_read}",
                funct3_is
            )));
        }

        let store_flags = [
            op_store * funct3_is[0],
            op_store * funct3_is[1],
            op_store * funct3_is[2],
        ];
        let store_residuals = w3_store_semantics_residuals(
            ram_wv,
            ram_rv,
            rs2_val,
            rd_has_write,
            ram_has_read,
            ram_has_write,
            store_flags,
            rs2_q16,
            ram_rv_low_bits,
            rs2_low_bits,
        );
        if let Some((idx, _)) = store_residuals
            .iter()
            .enumerate()
            .find(|(_, r)| **r != K::ZERO)
        {
            return Err(PiCcsError::ProtocolError(format!(
                "w3/store_semantics residual non-zero at row={j}, idx={idx}, active={active}, op_store={op_store}, funct3_is={:?}, ram_has_read={ram_has_read}, ram_has_write={ram_has_write}",
                funct3_is
            )));
        }
    }

    let mut main_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in main_col_ids.iter() {
        let vals = main_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing main decoded column {col_id}")))?;
        main_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let mut width_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in width_col_ids.iter() {
        let vals = width_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing width decoded column {col_id}")))?;
        width_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let mut decode_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in decode_col_ids.iter() {
        let vals = decode_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing decode decoded column {col_id}")))?;
        decode_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }

    let main_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        main_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing main sparse column {col_id}")))
    };
    let width_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        width_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing width sparse column {col_id}")))
    };
    let decode_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        decode_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing decode sparse column {col_id}")))
    };

    let bitness_cols: Vec<usize> = width
        .ram_rv_low_bit
        .iter()
        .chain(width.rs2_low_bit.iter())
        .copied()
        .collect();
    let mut bitness_sparse = Vec::with_capacity(bitness_cols.len());
    for &col_id in bitness_cols.iter() {
        bitness_sparse.push(width_col(col_id)?);
    }
    let bitness_weights = w3_bitness_weight_vector(r_cycle, bitness_cols.len());
    let bitness_oracle = FormulaOracleSparseTime::new(bitness_sparse, 3, r_cycle, move |vals: &[K]| {
        let mut weighted = K::ZERO;
        for (b, w) in vals.iter().zip(bitness_weights.iter()) {
            weighted += *w * *b * (*b - K::ONE);
        }
        weighted
    });

    let mut quiescence_sparse = Vec::with_capacity(1 + width.cols);
    quiescence_sparse.push(main_col(trace.active)?);
    for &col_id in width_col_ids.iter() {
        quiescence_sparse.push(width_col(col_id)?);
    }
    let quiescence_weights = w3_quiescence_weight_vector(r_cycle, width.cols);
    let quiescence_oracle = FormulaOracleSparseTime::new(quiescence_sparse, 3, r_cycle, move |vals: &[K]| {
        let active = vals[0];
        let mut weighted = K::ZERO;
        for (i, w) in quiescence_weights.iter().enumerate() {
            weighted += *w * vals[1 + i];
        }
        (K::ONE - active) * weighted
    });

    let mut load_sparse = Vec::with_capacity(31);
    load_sparse.push(main_col(trace.rd_val)?);
    load_sparse.push(main_col(trace.ram_rv)?);
    load_sparse.push(decode_col(decode.rd_has_write)?);
    load_sparse.push(decode_col(decode.ram_has_read)?);
    load_sparse.push(decode_col(decode.op_load)?);
    load_sparse.push(decode_col(decode.funct3_is[0])?);
    load_sparse.push(decode_col(decode.funct3_is[1])?);
    load_sparse.push(decode_col(decode.funct3_is[2])?);
    load_sparse.push(decode_col(decode.funct3_is[4])?);
    load_sparse.push(decode_col(decode.funct3_is[5])?);
    load_sparse.push(width_col(width.ram_rv_q16)?);
    for &col_id in width.ram_rv_low_bit.iter() {
        load_sparse.push(width_col(col_id)?);
    }
    let load_weights = w3_load_weight_vector(r_cycle, 16);
    let load_oracle = FormulaOracleSparseTime::new(load_sparse, 5, r_cycle, move |vals: &[K]| {
        let rd_val = vals[0];
        let ram_rv = vals[1];
        let rd_has_write = vals[2];
        let ram_has_read = vals[3];
        let op_load = vals[4];
        let funct3_is_0 = vals[5];
        let funct3_is_1 = vals[6];
        let funct3_is_2 = vals[7];
        let funct3_is_4 = vals[8];
        let funct3_is_5 = vals[9];
        let ram_rv_q16 = vals[10];
        let load_flags = [
            op_load * funct3_is_0,
            op_load * funct3_is_4,
            op_load * funct3_is_1,
            op_load * funct3_is_5,
            op_load * funct3_is_2,
        ];
        let mut ram_rv_low_bits = [K::ZERO; 16];
        ram_rv_low_bits.copy_from_slice(&vals[11..27]);
        let residuals = w3_load_semantics_residuals(
            rd_val,
            ram_rv,
            rd_has_write,
            ram_has_read,
            load_flags,
            ram_rv_q16,
            ram_rv_low_bits,
        );
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(load_weights.iter()) {
            weighted += *w * *r;
        }
        weighted
    });

    let mut store_sparse = Vec::with_capacity(45);
    store_sparse.push(main_col(trace.ram_wv)?);
    store_sparse.push(main_col(trace.ram_rv)?);
    store_sparse.push(main_col(trace.rs2_val)?);
    store_sparse.push(decode_col(decode.rd_has_write)?);
    store_sparse.push(decode_col(decode.ram_has_read)?);
    store_sparse.push(decode_col(decode.ram_has_write)?);
    store_sparse.push(decode_col(decode.op_store)?);
    store_sparse.push(decode_col(decode.funct3_is[0])?);
    store_sparse.push(decode_col(decode.funct3_is[1])?);
    store_sparse.push(decode_col(decode.funct3_is[2])?);
    store_sparse.push(width_col(width.rs2_q16)?);
    for &col_id in width.ram_rv_low_bit.iter() {
        store_sparse.push(width_col(col_id)?);
    }
    for &col_id in width.rs2_low_bit.iter() {
        store_sparse.push(width_col(col_id)?);
    }
    let store_weights = w3_store_weight_vector(r_cycle, 12);
    let store_oracle = FormulaOracleSparseTime::new(store_sparse, 4, r_cycle, move |vals: &[K]| {
        let ram_wv = vals[0];
        let ram_rv = vals[1];
        let rs2_val = vals[2];
        let rd_has_write = vals[3];
        let ram_has_read = vals[4];
        let ram_has_write = vals[5];
        let op_store = vals[6];
        let funct3_is_0 = vals[7];
        let funct3_is_1 = vals[8];
        let funct3_is_2 = vals[9];
        let rs2_q16 = vals[10];
        let store_flags = [op_store * funct3_is_0, op_store * funct3_is_1, op_store * funct3_is_2];
        let mut ram_rv_low_bits = [K::ZERO; 16];
        ram_rv_low_bits.copy_from_slice(&vals[11..27]);
        let mut rs2_low_bits = [K::ZERO; 16];
        rs2_low_bits.copy_from_slice(&vals[27..43]);
        let residuals = w3_store_semantics_residuals(
            ram_wv,
            ram_rv,
            rs2_val,
            rd_has_write,
            ram_has_read,
            ram_has_write,
            store_flags,
            rs2_q16,
            ram_rv_low_bits,
            rs2_low_bits,
        );
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(store_weights.iter()) {
            weighted += *w * *r;
        }
        weighted
    });

    Ok((
        Some((Box::new(bitness_oracle), K::ZERO)),
        Some((Box::new(quiescence_oracle), K::ZERO)),
        None,
        Some((Box::new(load_oracle), K::ZERO)),
        Some((Box::new(store_oracle), K::ZERO)),
    ))
}

type ControlTimeClaims = (
    Option<(Box<dyn RoundOracle + Send>, K)>,
    Option<(Box<dyn RoundOracle + Send>, K)>,
    Option<(Box<dyn RoundOracle + Send>, K)>,
    Option<(Box<dyn RoundOracle + Send>, K)>,
);

pub(crate) fn build_route_a_control_time_claims(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
) -> Result<ControlTimeClaims, PiCcsError> {
    if !control_stage_required_for_step_witness(step) {
        return Ok((None, None, None, None));
    }
    let trace = Rv32TraceLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();
    let m_in = step.mcs.0.m_in;
    let ell_n = r_cycle.len();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("control stage: t_len must be >= 1".into()));
    }

    let main_col_ids = vec![
        trace.active,
        trace.is_virtual,
        trace.instr_word,
        trace.pc_before,
        trace.pc_after,
        trace.rs1_val,
        trace.rd_val,
        trace.shout_val,
        trace.jalr_drop_bit,
    ];
    let decode_col_ids = vec![
        decode.op_lui,
        decode.op_auipc,
        decode.op_jal,
        decode.op_jalr,
        decode.op_branch,
        decode.op_load,
        decode.op_store,
        decode.op_alu_imm,
        decode.op_alu_reg,
        decode.op_misc_mem,
        decode.op_system,
        decode.op_amo,
        decode.op_custom,
        decode.op_lui_write,
        decode.op_auipc_write,
        decode.op_jal_write,
        decode.op_jalr_write,
        decode.rd_is_zero,
        decode.imm_i,
        decode.imm_b,
        decode.imm_j,
        decode.funct3_is[6],
        decode.funct3_is[7],
        decode.funct3_bit[0],
        decode.funct3_bit[1],
        decode.funct3_bit[2],
        decode.rs1_bit[0],
        decode.rs1_bit[1],
        decode.rs1_bit[2],
        decode.rs1_bit[3],
        decode.rs1_bit[4],
        decode.rs2_bit[0],
        decode.rs2_bit[1],
        decode.rs2_bit[2],
        decode.rs2_bit[3],
        decode.rs2_bit[4],
        decode.funct7_bit[0],
        decode.funct7_bit[1],
        decode.funct7_bit[2],
        decode.funct7_bit[3],
        decode.funct7_bit[4],
        decode.funct7_bit[5],
        decode.funct7_bit[6],
    ];

    let main_decoded = decode_trace_col_values_batch(params, step, t_len, &main_col_ids)?;
    let decode_decoded = {
        let instr_vals = main_decoded
            .get(&trace.instr_word)
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing instr_word decode column".into()))?;
        let active_vals = main_decoded
            .get(&trace.active)
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing active decode column".into()))?;
        if instr_vals.len() != t_len || active_vals.len() != t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "control(shared): decoded CPU column lengths drift (instr={}, active={}, t_len={t_len})",
                instr_vals.len(),
                active_vals.len()
            )));
        }
        let mut decoded = BTreeMap::<usize, Vec<K>>::new();
        for &col_id in decode_col_ids.iter() {
            decoded.insert(col_id, Vec::with_capacity(t_len));
        }
        for j in 0..t_len {
            let instr_word = decode_k_to_u32(instr_vals[j], "control(shared)/instr_word")?;
            let active = active_vals[j] != K::ZERO;
            let mut row = rv32_decode_lookup_backed_row_from_instr_word(&decode, instr_word, active);
            if !active {
                row.fill(F::ZERO);
            }
            let rd_has_write = if active {
                K::ONE - K::from(row[decode.rd_is_zero])
            } else {
                K::ZERO
            };
            let op_lui = K::from(row[decode.op_lui]);
            let op_auipc = K::from(row[decode.op_auipc]);
            let op_jal = K::from(row[decode.op_jal]);
            let op_jalr = K::from(row[decode.op_jalr]);
            for &col_id in decode_col_ids.iter() {
                let val = match col_id {
                    c if c == decode.op_lui_write => op_lui * rd_has_write,
                    c if c == decode.op_auipc_write => op_auipc * rd_has_write,
                    c if c == decode.op_jal_write => op_jal * rd_has_write,
                    c if c == decode.op_jalr_write => op_jalr * rd_has_write,
                    _ => K::from(row[col_id]),
                };
                decoded
                    .get_mut(&col_id)
                    .ok_or_else(|| PiCcsError::ProtocolError("control(shared): decode map build failed".into()))?
                    .push(val);
            }
        }
        decoded
    };

    #[cfg(debug_assertions)]
    for j in 0..t_len {
        let is_virtual = *main_decoded
            .get(&trace.is_virtual)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing is_virtual row while validating".into())
            })?;
        let pc_before = *main_decoded
            .get(&trace.pc_before)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing pc_before row while validating".into())
            })?;
        let pc_after = *main_decoded
            .get(&trace.pc_after)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing pc_after row while validating".into())
            })?;
        let op_lui = *decode_decoded
            .get(&decode.op_lui)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing op_lui row while validating".into()))?;
        let op_auipc = *decode_decoded
            .get(&decode.op_auipc)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing op_auipc row while validating".into())
            })?;
        let op_load = *decode_decoded
            .get(&decode.op_load)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing op_load row while validating".into()))?;
        let op_store = *decode_decoded
            .get(&decode.op_store)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing op_store row while validating".into())
            })?;
        let op_alu_imm = *decode_decoded
            .get(&decode.op_alu_imm)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing op_alu_imm row while validating".into())
            })?;
        let op_alu_reg = *decode_decoded
            .get(&decode.op_alu_reg)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing op_alu_reg row while validating".into())
            })?;
        let op_misc_mem = *decode_decoded
            .get(&decode.op_misc_mem)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing op_misc_mem row while validating".into())
            })?;
        let op_system = *decode_decoded
            .get(&decode.op_system)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing op_system row while validating".into())
            })?;
        let op_amo = *decode_decoded
            .get(&decode.op_amo)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing op_amo row while validating".into()))?;
        let op_custom = *decode_decoded
            .get(&decode.op_custom)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing op_custom row while validating".into())
            })?;
        let residual = control_next_pc_linear_residual(
            pc_before,
            pc_after,
            is_virtual,
            op_lui,
            op_auipc,
            op_load,
            op_store,
            op_alu_imm,
            op_alu_reg,
            op_misc_mem,
            op_system,
            op_amo,
            op_custom,
        );
        if residual != K::ZERO {
            return Err(PiCcsError::ProtocolError(format!(
                "control/next_pc_linear residual non-zero at row={j}, residual={residual}, is_virtual={is_virtual}, pc_before={pc_before}, pc_after={pc_after}, op_lui={op_lui}, op_auipc={op_auipc}, op_load={op_load}, op_store={op_store}, op_alu_imm={op_alu_imm}, op_alu_reg={op_alu_reg}, op_misc_mem={op_misc_mem}, op_system={op_system}, op_amo={op_amo}, op_custom={op_custom}"
            )));
        }
    }

    #[cfg(debug_assertions)]
    for j in 0..t_len {
        let active = *main_decoded
            .get(&trace.active)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing active row while validating".into()))?;
        let pc_before = *main_decoded
            .get(&trace.pc_before)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing pc_before row while validating".into())
            })?;
        let pc_after = *main_decoded
            .get(&trace.pc_after)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing pc_after row while validating".into())
            })?;
        let rs1_val = *main_decoded
            .get(&trace.rs1_val)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rs1_val row while validating".into()))?;
        let jalr_drop_bit = *main_decoded
            .get(&trace.jalr_drop_bit)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing jalr_drop_bit row while validating".into())
            })?;
        let shout_val = *main_decoded
            .get(&trace.shout_val)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing shout_val row while validating".into())
            })?;
        let imm_i = *decode_decoded
            .get(&decode.imm_i)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing imm_i row while validating".into()))?;
        let imm_b = *decode_decoded
            .get(&decode.imm_b)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing imm_b row while validating".into()))?;
        let imm_j = *decode_decoded
            .get(&decode.imm_j)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing imm_j row while validating".into()))?;
        let imm_sign_bit = *decode_decoded
            .get(&decode.funct7_bit[6])
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing funct7_bit[6] row while validating".into())
            })?;
        let op_jal = *decode_decoded
            .get(&decode.op_jal)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing op_jal row while validating".into()))?;
        let op_jalr = *decode_decoded
            .get(&decode.op_jalr)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing op_jalr row while validating".into()))?;
        let op_branch = *decode_decoded
            .get(&decode.op_branch)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing op_branch row while validating".into())
            })?;
        let funct3_bit0 = *decode_decoded
            .get(&decode.funct3_bit[0])
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing funct3_bit[0] row while validating".into())
            })?;
        let residuals = control_next_pc_control_residuals(
            active,
            pc_before,
            pc_after,
            rs1_val,
            jalr_drop_bit,
            imm_i,
            imm_b,
            imm_j,
            imm_sign_bit,
            op_jal,
            op_jalr,
            op_branch,
            shout_val,
            funct3_bit0,
        );
        if let Some((idx, _)) = residuals.iter().enumerate().find(|(_, r)| **r != K::ZERO) {
            return Err(PiCcsError::ProtocolError(format!(
                "control/next_pc_control residual non-zero at row={j}, idx={idx}, active={active}, op_jal={op_jal}, op_jalr={op_jalr}, op_branch={op_branch}, jalr_drop_bit={jalr_drop_bit}, pc_before={pc_before}, pc_after={pc_after}, rs1_val={rs1_val}, imm_i={imm_i}, imm_b={imm_b}, imm_j={imm_j}, imm_sign_bit={imm_sign_bit}, shout_val={shout_val}, funct3_bit0={funct3_bit0}"
            )));
        }
    }

    #[cfg(debug_assertions)]
    for j in 0..t_len {
        let rd_val = *main_decoded
            .get(&trace.rd_val)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rd_val row while validating".into()))?;
        let pc_before = *main_decoded
            .get(&trace.pc_before)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing pc_before row while validating".into())
            })?;
        let op_lui = *decode_decoded
            .get(&decode.op_lui)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing op_lui row while validating".into()))?;
        let op_auipc = *decode_decoded
            .get(&decode.op_auipc)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing op_auipc row while validating".into())
            })?;
        let op_jal = *decode_decoded
            .get(&decode.op_jal)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing op_jal row while validating".into()))?;
        let op_jalr = *decode_decoded
            .get(&decode.op_jalr)
            .and_then(|v| v.get(j))
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing op_jalr row while validating".into()))?;
        let rd_is_zero = *decode_decoded
            .get(&decode.rd_is_zero)
            .and_then(|v| v.get(j))
            .ok_or_else(|| {
                PiCcsError::ProtocolError("control(shared): missing rd_is_zero row while validating".into())
            })?;
        let funct3_bits = [
            *decode_decoded
                .get(&decode.funct3_bit[0])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing funct3_bit[0] row".into()))?,
            *decode_decoded
                .get(&decode.funct3_bit[1])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing funct3_bit[1] row".into()))?,
            *decode_decoded
                .get(&decode.funct3_bit[2])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing funct3_bit[2] row".into()))?,
        ];
        let rs1_bits = [
            *decode_decoded
                .get(&decode.rs1_bit[0])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rs1_bit[0] row".into()))?,
            *decode_decoded
                .get(&decode.rs1_bit[1])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rs1_bit[1] row".into()))?,
            *decode_decoded
                .get(&decode.rs1_bit[2])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rs1_bit[2] row".into()))?,
            *decode_decoded
                .get(&decode.rs1_bit[3])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rs1_bit[3] row".into()))?,
            *decode_decoded
                .get(&decode.rs1_bit[4])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rs1_bit[4] row".into()))?,
        ];
        let rs2_bits = [
            *decode_decoded
                .get(&decode.rs2_bit[0])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rs2_bit[0] row".into()))?,
            *decode_decoded
                .get(&decode.rs2_bit[1])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rs2_bit[1] row".into()))?,
            *decode_decoded
                .get(&decode.rs2_bit[2])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rs2_bit[2] row".into()))?,
            *decode_decoded
                .get(&decode.rs2_bit[3])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rs2_bit[3] row".into()))?,
            *decode_decoded
                .get(&decode.rs2_bit[4])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing rs2_bit[4] row".into()))?,
        ];
        let funct7_bits = [
            *decode_decoded
                .get(&decode.funct7_bit[0])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing funct7_bit[0] row".into()))?,
            *decode_decoded
                .get(&decode.funct7_bit[1])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing funct7_bit[1] row".into()))?,
            *decode_decoded
                .get(&decode.funct7_bit[2])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing funct7_bit[2] row".into()))?,
            *decode_decoded
                .get(&decode.funct7_bit[3])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing funct7_bit[3] row".into()))?,
            *decode_decoded
                .get(&decode.funct7_bit[4])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing funct7_bit[4] row".into()))?,
            *decode_decoded
                .get(&decode.funct7_bit[5])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing funct7_bit[5] row".into()))?,
            *decode_decoded
                .get(&decode.funct7_bit[6])
                .and_then(|v| v.get(j))
                .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing funct7_bit[6] row".into()))?,
        ];
        let imm_u = control_imm_u_from_bits(funct3_bits, rs1_bits, rs2_bits, funct7_bits);
        let op_lui_write = op_lui * (K::ONE - rd_is_zero);
        let op_auipc_write = op_auipc * (K::ONE - rd_is_zero);
        let op_jal_write = op_jal * (K::ONE - rd_is_zero);
        let op_jalr_write = op_jalr * (K::ONE - rd_is_zero);
        let residuals = control_writeback_residuals(
            rd_val,
            pc_before,
            imm_u,
            op_lui_write,
            op_auipc_write,
            op_jal_write,
            op_jalr_write,
        );
        if let Some((idx, _)) = residuals.iter().enumerate().find(|(_, r)| **r != K::ZERO) {
            return Err(PiCcsError::ProtocolError(format!(
                "control/writeback residual non-zero at row={j}, idx={idx}, rd_val={rd_val}, pc_before={pc_before}, imm_u={imm_u}, op_lui={op_lui}, op_auipc={op_auipc}, op_jal={op_jal}, op_jalr={op_jalr}, rd_is_zero={rd_is_zero}"
            )));
        }
    }

    let mut main_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in main_col_ids.iter() {
        let vals = main_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("control stage missing main decoded column {col_id}")))?;
        main_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let mut decode_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in decode_col_ids.iter() {
        let vals = decode_decoded.get(&col_id).ok_or_else(|| {
            PiCcsError::ProtocolError(format!("control stage missing decode decoded column {col_id}"))
        })?;
        decode_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }

    let main_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        main_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("control stage missing main sparse col {col_id}")))
    };
    let decode_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        decode_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("control stage missing decode sparse col {col_id}")))
    };

    let linear_sparse = vec![
        main_col(trace.pc_before)?,
        main_col(trace.pc_after)?,
        main_col(trace.is_virtual)?,
        decode_col(decode.op_lui)?,
        decode_col(decode.op_auipc)?,
        decode_col(decode.op_load)?,
        decode_col(decode.op_store)?,
        decode_col(decode.op_alu_imm)?,
        decode_col(decode.op_alu_reg)?,
        decode_col(decode.op_misc_mem)?,
        decode_col(decode.op_system)?,
        decode_col(decode.op_amo)?,
        decode_col(decode.op_custom)?,
    ];
    let linear_weights = control_next_pc_linear_weight_vector(r_cycle, 1);
    let linear_oracle = FormulaOracleSparseTime::new(linear_sparse, 4, r_cycle, move |vals: &[K]| {
        let residual = control_next_pc_linear_residual(
            vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8], vals[9], vals[10],
            vals[11], vals[12],
        );
        linear_weights[0] * residual
    });

    let control_sparse = vec![
        main_col(trace.active)?,
        main_col(trace.pc_before)?,
        main_col(trace.pc_after)?,
        main_col(trace.rs1_val)?,
        main_col(trace.jalr_drop_bit)?,
        main_col(trace.shout_val)?,
        decode_col(decode.funct3_bit[0])?,
        decode_col(decode.op_jal)?,
        decode_col(decode.op_jalr)?,
        decode_col(decode.op_branch)?,
        decode_col(decode.imm_i)?,
        decode_col(decode.imm_b)?,
        decode_col(decode.imm_j)?,
        decode_col(decode.funct7_bit[6])?,
    ];
    let control_weights = control_next_pc_control_weight_vector(r_cycle, 5);
    let control_oracle = FormulaOracleSparseTime::new(control_sparse, 5, r_cycle, move |vals: &[K]| {
        let residuals = control_next_pc_control_residuals(
            vals[0], vals[1], vals[2], vals[3], vals[4], vals[10], vals[11], vals[12], vals[13], vals[7], vals[8],
            vals[9], vals[5], vals[6],
        );
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(control_weights.iter()) {
            weighted += *w * *r;
        }
        weighted
    });

    let branch_sparse = vec![
        decode_col(decode.op_branch)?,
        main_col(trace.shout_val)?,
        decode_col(decode.funct3_bit[0])?,
        decode_col(decode.funct3_bit[1])?,
        decode_col(decode.funct3_bit[2])?,
        decode_col(decode.funct3_is[6])?,
        decode_col(decode.funct3_is[7])?,
    ];
    let branch_weights = control_branch_semantics_weight_vector(r_cycle, 3);
    let branch_oracle = FormulaOracleSparseTime::new(branch_sparse, 4, r_cycle, move |vals: &[K]| {
        let residuals =
            control_branch_semantics_residuals(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]);
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(branch_weights.iter()) {
            weighted += *w * *r;
        }
        weighted
    });

    let mut write_sparse = vec![
        main_col(trace.rd_val)?,
        main_col(trace.pc_before)?,
        decode_col(decode.op_lui)?,
        decode_col(decode.op_auipc)?,
        decode_col(decode.op_jal)?,
        decode_col(decode.op_jalr)?,
        decode_col(decode.rd_is_zero)?,
        decode_col(decode.funct3_bit[0])?,
        decode_col(decode.funct3_bit[1])?,
        decode_col(decode.funct3_bit[2])?,
    ];
    for &col_id in decode.rs1_bit.iter() {
        write_sparse.push(decode_col(col_id)?);
    }
    for &col_id in decode.rs2_bit.iter() {
        write_sparse.push(decode_col(col_id)?);
    }
    for &col_id in decode.funct7_bit.iter() {
        write_sparse.push(decode_col(col_id)?);
    }
    let write_weights = control_writeback_weight_vector(r_cycle, 4);
    let write_oracle = FormulaOracleSparseTime::new(write_sparse, 5, r_cycle, move |vals: &[K]| {
        let rd_val = vals[0];
        let pc_before = vals[1];
        let op_lui = vals[2];
        let op_auipc = vals[3];
        let op_jal = vals[4];
        let op_jalr = vals[5];
        let rd_is_zero = vals[6];
        let op_lui_write = op_lui * (K::ONE - rd_is_zero);
        let op_auipc_write = op_auipc * (K::ONE - rd_is_zero);
        let op_jal_write = op_jal * (K::ONE - rd_is_zero);
        let op_jalr_write = op_jalr * (K::ONE - rd_is_zero);
        let funct3_bits = [vals[7], vals[8], vals[9]];
        let rs1_bits = [vals[10], vals[11], vals[12], vals[13], vals[14]];
        let rs2_bits = [vals[15], vals[16], vals[17], vals[18], vals[19]];
        let funct7_bits = [vals[20], vals[21], vals[22], vals[23], vals[24], vals[25], vals[26]];
        let imm_u = control_imm_u_from_bits(funct3_bits, rs1_bits, rs2_bits, funct7_bits);
        let residuals = control_writeback_residuals(
            rd_val,
            pc_before,
            imm_u,
            op_lui_write,
            op_auipc_write,
            op_jal_write,
            op_jalr_write,
        );
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(write_weights.iter()) {
            weighted += *w * *r;
        }
        weighted
    });

    Ok((
        Some((Box::new(linear_oracle), K::ZERO)),
        Some((Box::new(control_oracle), K::ZERO)),
        Some((Box::new(branch_oracle), K::ZERO)),
        Some((Box::new(write_oracle), K::ZERO)),
    ))
}
pub(crate) fn emit_route_a_wb_wp_me_claims(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_time: &[K],
) -> Result<(Vec<CeClaim<Cmt, F, K>>, Vec<CeClaim<Cmt, F, K>>), PiCcsError> {
    if !wb_wp_required_for_step_witness(step) {
        return Ok((Vec::new(), Vec::new()));
    }

    let trace = Rv32TraceLayout::new();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    let m_in = step.mcs.0.m_in;
    let core_t = s.t();
    let (mcs_inst, mcs_wit) = &step.mcs;

    let wb_cols = rv32_trace_wb_columns(&trace);
    let mut wb_claims = ts::emit_me_claims_for_mats(
        tr,
        b"cpu/me_digest_wb_time",
        params,
        s,
        core::slice::from_ref(&mcs_inst.c),
        core::slice::from_ref(&mcs_wit.Z),
        r_time,
        m_in,
    )?;
    if wb_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(format!(
            "WB expects exactly one CPU ME claim at r_time, got {}",
            wb_claims.len()
        )));
    }
    let mut wp_claims = wb_claims.clone();
    if wp_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(format!(
            "WP expects exactly one CPU ME claim at r_time, got {}",
            wp_claims.len()
        )));
    }
    wp_claims[0].fold_digest = {
        let mut fork = tr.fork(b"cpu/me_digest_wp_time");
        fork.digest32()
    };

    let wb_use_time_cols = step.time_columns.t == t_len
        && !step.time_columns.cpu_cols.is_empty()
        && wb_cols
            .iter()
            .all(|&col_id| col_id < step.time_columns.cpu_cols.len());
    if !wb_use_time_cols {
        return Err(PiCcsError::ProtocolError(format!(
            "WB(shared): canonical time CPU columns are required (time_t={}, cpu_cols={}, expected_t={t_len})",
            step.time_columns.t,
            step.time_columns.cpu_cols.len()
        )));
    }
    crate::memory_sidecar::cpu_bus::append_time_columns_openings_to_me_instance(
        params,
        m_in,
        t_len,
        &step.time_columns.cpu_cols,
        &wb_cols,
        core_t,
        &mut wb_claims[0],
    )?;

    let mut wp_cols = rv32_trace_wp_opening_columns(&trace);
    if control_stage_required_for_step_witness(step) {
        wp_cols.extend(rv32_trace_control_extra_opening_columns(&trace));
    }
    if decode_stage_required_for_step_witness(step) {
        let decode_layout = Rv32DecodeSidecarLayout::new();
        let (_decode_open_cols, decode_lut_slots) = resolve_shared_decode_lookup_lut_indices(step, &decode_layout)?;
        let bus = build_bus_layout_for_step_witness(step, t_len)?;
        if bus.shout_cols.len() != step.lut_instances.len() {
            return Err(PiCcsError::ProtocolError(
                "W2(shared): bus layout shout lane count drift".into(),
            ));
        }
        for &(lut_idx, val_slot) in decode_lut_slots.iter() {
            let inst_cols = bus.shout_cols.get(lut_idx).ok_or_else(|| {
                PiCcsError::ProtocolError("W2(shared): missing shout cols for decode lookup table".into())
            })?;
            let lane0 = inst_cols.lanes.get(0).ok_or_else(|| {
                PiCcsError::ProtocolError("W2(shared): expected one shout lane for decode lookup table".into())
            })?;
            let val_col = lane0.vals.get(val_slot).copied().ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W2(shared): decode val_slot={} out of range for lut_idx={} (n_vals={})",
                    val_slot,
                    lut_idx,
                    lane0.vals.len()
                ))
            })?;
            let logical_bus_col = time_mem_logical_col_id_for_step(step, val_col, "W2(shared)")?;
            wp_cols.push(logical_bus_col);
        }
    }
    if width_stage_required_for_step_witness(step) {
        wp_cols.extend(width_lookup_bus_val_cols_witness(step, t_len)?);
    }
    let wp_use_time_cols = step.time_columns.t == t_len
        && !step.time_columns.cpu_cols.is_empty()
        && !step.time_columns.mem_cols.is_empty();
    if !wp_use_time_cols {
        return Err(PiCcsError::ProtocolError(format!(
            "WP(shared): canonical time CPU/MEM columns are required (time_t={}, cpu_cols={}, mem_cols={}, expected_t={t_len})",
            step.time_columns.t,
            step.time_columns.cpu_cols.len(),
            step.time_columns.mem_cols.len()
        )));
    }
    crate::memory_sidecar::cpu_bus::append_mixed_time_columns_openings_to_me_instance(
        params,
        m_in,
        t_len,
        &step.time_columns.cpu_cols,
        &step.time_columns.mem_cols,
        &step.time_columns.col_ids,
        &wp_cols,
        core_t,
        &mut wp_claims[0],
    )?;
    Ok((wb_claims, wp_claims))
}

pub(crate) fn verify_route_a_wb_wp_terminals(
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    claim_plan: &RouteATimeClaimPlan,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
) -> Result<(), PiCcsError> {
    // WB/WP terminal checks are RV32 trace-layout specific.
    if step.mcs_inst.m_in != 5 {
        return Ok(());
    }

    let trace = Rv32TraceLayout::new();
    let requires_wb_wp = claim_plan.wb_bool.is_some()
        || claim_plan.wp_quiescence.is_some()
        || !mem_proof.wb_me_claims.is_empty()
        || !mem_proof.wp_me_claims.is_empty();
    if !requires_wb_wp {
        return Ok(());
    }
    let cpu_cols_len = step.time_columns.cpu_cols.len();
    let mem_cols_len = step.time_columns.mem_cols.len();
    let expected_logical_cols = cpu_cols_len.saturating_add(mem_cols_len);
    let strict_committed_mode = step.mcs_inst.m_in == 5
        && step.time_columns.t > 0
        && cpu_cols_len > 0
        && step.time_columns.col_ids.len() == expected_logical_cols;
    if !strict_committed_mode {
        return Err(PiCcsError::ProtocolError(
            "WB/WP terminals require canonical committed time-column mode".into(),
        ));
    }

    if let Some(claim_idx) = claim_plan.wb_bool {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "wb/booleanity claim index out of range".into(),
            ));
        }
        if mem_proof.wb_me_claims.len() != 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "WB expects exactly one ME claim at r_time (got {})",
                mem_proof.wb_me_claims.len()
            )));
        }
        let me = &mem_proof.wb_me_claims[0];
        if me.r.as_slice() != r_time {
            return Err(PiCcsError::ProtocolError(
                "WB ME claim r mismatch (expected r_time)".into(),
            ));
        }
        if me.c != step.mcs_inst.c {
            return Err(PiCcsError::ProtocolError("WB ME claim commitment mismatch".into()));
        }
        if me.m_in != step.mcs_inst.m_in {
            return Err(PiCcsError::ProtocolError("WB ME claim m_in mismatch".into()));
        }

        let wb_bool_cols = rv32_trace_wb_columns(&trace);
        let (wb_open_entry, wb_open_map) =
            require_time_openings_covering_point(step_time_openings, r_time, &wb_bool_cols, "WB")?;
        if wb_open_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "WB requires CommittedOpening source (got {:?})",
                wb_open_entry.source
            )));
        }
        let wb_weights = wb_weight_vector(r_cycle, wb_bool_cols.len());
        let mut wb_weighted_bitness = K::ZERO;
        for (&col_id, &w) in wb_bool_cols.iter().zip(wb_weights.iter()) {
            let b = named_opening(&wb_open_map, col_id, "WB")?;
            wb_weighted_bitness += w * b * (b - K::ONE);
        }

        let expected_terminal = eq_points(r_time, r_cycle) * wb_weighted_bitness;
        let observed_terminal = batched_final_values[claim_idx];
        if observed_terminal != expected_terminal {
            return Err(PiCcsError::ProtocolError(
                "wb/booleanity terminal value mismatch".into(),
            ));
        }
    } else if !mem_proof.wb_me_claims.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "unexpected WB ME claims: wb/booleanity stage is not enabled".into(),
        ));
    }

    if let Some(claim_idx) = claim_plan.wp_quiescence {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "wp/quiescence claim index out of range".into(),
            ));
        }
        if mem_proof.wp_me_claims.len() != 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "WP expects exactly one ME claim at r_time (got {})",
                mem_proof.wp_me_claims.len()
            )));
        }
        let me = &mem_proof.wp_me_claims[0];
        if me.r.as_slice() != r_time {
            return Err(PiCcsError::ProtocolError(
                "WP ME claim r mismatch (expected r_time)".into(),
            ));
        }
        if me.c != step.mcs_inst.c {
            return Err(PiCcsError::ProtocolError("WP ME claim commitment mismatch".into()));
        }
        if me.m_in != step.mcs_inst.m_in {
            return Err(PiCcsError::ProtocolError("WP ME claim m_in mismatch".into()));
        }

        let wp_open_cols = rv32_trace_wp_opening_columns(&trace);
        let (wp_open_entry, wp_open_map) =
            require_time_openings_covering_point(step_time_openings, r_time, &wp_open_cols, "WP")?;
        if wp_open_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "WP requires CommittedOpening source (got {:?})",
                wp_open_entry.source
            )));
        }
        let active_open = named_opening(&wp_open_map, trace.active, "WP")?;
        let wp_cols_no_active = &wp_open_cols[1..];
        let wp_weights = wp_weight_vector(r_cycle, wp_cols_no_active.len());
        let mut wp_weighted_sum = K::ZERO;
        for (&col_id, &w) in wp_cols_no_active.iter().zip(wp_weights.iter()) {
            let v = named_opening(&wp_open_map, col_id, "WP")?;
            wp_weighted_sum += w * v;
        }
        let expected_terminal = eq_points(r_time, r_cycle) * (K::ONE - active_open) * wp_weighted_sum;
        let observed_terminal = batched_final_values[claim_idx];
        if observed_terminal != expected_terminal {
            return Err(PiCcsError::ProtocolError(
                "wp/quiescence terminal value mismatch".into(),
            ));
        }
    } else if !mem_proof.wp_me_claims.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "unexpected WP ME claims: wp/quiescence stage is not enabled".into(),
        ));
    }

    Ok(())
}
