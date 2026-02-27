use super::*;

pub(crate) fn decode_lookup_open_map_from_committed_openings(
    step: &StepInstanceBundle<Cmt, F, K>,
    cpu_bus: &BusLayout,
    point: &[K],
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
    label: &str,
) -> Result<BTreeMap<usize, K>, PiCcsError> {
    let decode_layout = Rv32DecodeSidecarLayout::new();
    let decode_open_cols = rv32_decode_lookup_transport_cols(&decode_layout);
    let bus_logical_cols = bus_logical_col_ids_for_step_instance(step, cpu_bus, label)?;
    let mut decode_col_to_logical = Vec::with_capacity(decode_open_cols.len());
    for &col_id in decode_open_cols.iter() {
        let table_id = rv32_decode_lookup_table_id_for_col(col_id);
        let val_slot = rv32_decode_lookup_val_slot_for_col(col_id).ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: decode col_id={col_id} is not part of decode lookup transport slot map"
            ))
        })?;
        let lut_idx = step
            .lut_insts
            .iter()
            .position(|inst| inst.table_id == table_id)
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "{label}: missing decode lookup table_id={table_id} for col_id={col_id}"
                ))
            })?;
        let inst_cols = cpu_bus
            .shout_cols
            .get(lut_idx)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("{label}: missing shout cols for lut_idx={lut_idx}")))?;
        let lane0 = inst_cols.lanes.first().ok_or_else(|| {
            PiCcsError::ProtocolError(format!("{label}: expected one shout lane for lut_idx={lut_idx}"))
        })?;
        let mem_local_col = lane0.vals.get(val_slot).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: decode val_slot={} out of range for lut_idx={} (n_vals={})",
                val_slot,
                lut_idx,
                lane0.vals.len()
            ))
        })?;
        let logical_col = bus_logical_cols
            .get(mem_local_col)
            .copied()
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "{label}: bus local col {} out of range (bus_cols={})",
                    mem_local_col,
                    bus_logical_cols.len()
                ))
            })?;
        decode_col_to_logical.push((col_id, logical_col));
    }
    let mut required_logical: Vec<usize> = decode_col_to_logical
        .iter()
        .map(|(_, logical)| *logical)
        .collect();
    required_logical.sort_unstable();
    required_logical.dedup();
    let (_entry, logical_map) =
        require_time_openings_covering_point(step_time_openings, point, &required_logical, label)?;

    let mut decode_open_map = BTreeMap::new();
    for (col_id, logical_col) in decode_col_to_logical {
        let v = logical_map.get(&logical_col).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: missing logical opening value for logical_col_id={logical_col}"
            ))
        })?;
        decode_open_map.insert(col_id, v);
    }
    Ok(decode_open_map)
}

fn decode_open_map_from_instr_and_transport(
    step: &StepInstanceBundle<Cmt, F, K>,
    cpu_bus: &BusLayout,
    point: &[K],
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
    _instr_word: K,
    _active: K,
    label: &str,
) -> Result<BTreeMap<usize, K>, PiCcsError> {
    decode_lookup_open_map_from_committed_openings(step, cpu_bus, point, step_time_openings, label)
}

fn width_lookup_open_map_from_committed_openings(
    step: &StepInstanceBundle<Cmt, F, K>,
    cpu_bus: &BusLayout,
    point: &[K],
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
    label: &str,
) -> Result<BTreeMap<usize, K>, PiCcsError> {
    let width = Rv32WidthSidecarLayout::new();
    let width_open_cols = rv32_width_lookup_backed_cols(&width);
    let bus_logical_cols = bus_logical_col_ids_for_step_instance(step, cpu_bus, label)?;
    let mut width_col_to_logical = Vec::with_capacity(width_open_cols.len());
    for &col_id in width_open_cols.iter() {
        let table_id = rv32_width_lookup_table_id_for_col(col_id);
        let val_slot = rv32_width_lookup_val_slot_for_col(col_id).ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: width col_id={col_id} is not part of width lookup transport slot map"
            ))
        })?;
        let lut_idx = step
            .lut_insts
            .iter()
            .position(|inst| inst.table_id == table_id)
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "{label}: missing width lookup table_id={table_id} for col_id={col_id}"
                ))
            })?;
        let inst_cols = cpu_bus
            .shout_cols
            .get(lut_idx)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("{label}: missing shout cols for lut_idx={lut_idx}")))?;
        let lane0 = inst_cols.lanes.first().ok_or_else(|| {
            PiCcsError::ProtocolError(format!("{label}: expected one shout lane for lut_idx={lut_idx}"))
        })?;
        let mem_local_col = lane0.vals.get(val_slot).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: width val_slot={} out of range for lut_idx={} (n_vals={})",
                val_slot,
                lut_idx,
                lane0.vals.len()
            ))
        })?;
        let logical_col = bus_logical_cols
            .get(mem_local_col)
            .copied()
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "{label}: bus local col {} out of range (bus_cols={})",
                    mem_local_col,
                    bus_logical_cols.len()
                ))
            })?;
        width_col_to_logical.push((col_id, logical_col));
    }
    let mut required_logical: Vec<usize> = width_col_to_logical
        .iter()
        .map(|(_, logical)| *logical)
        .collect();
    required_logical.sort_unstable();
    required_logical.dedup();
    let (_entry, logical_map) =
        require_time_openings_covering_point(step_time_openings, point, &required_logical, label)?;

    let mut width_open_map = BTreeMap::new();
    for (col_id, logical_col) in width_col_to_logical {
        let v = logical_map.get(&logical_col).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "{label}: missing logical opening value for logical_col_id={logical_col}"
            ))
        })?;
        width_open_map.insert(col_id, v);
    }
    Ok(width_open_map)
}

pub(crate) fn verify_route_a_decode_terminals(
    cpu_bus: &BusLayout,
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    claim_plan: &RouteATimeClaimPlan,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
) -> Result<(), PiCcsError> {
    if claim_plan.decode_fields.is_none() && claim_plan.decode_immediates.is_none() {
        return Ok(());
    }

    if mem_proof.wb_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "W2 requires WB ME openings for shared active/bit terminals".into(),
        ));
    }

    let decode_layout = Rv32DecodeSidecarLayout::new();
    if mem_proof.wp_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "W2 requires WP ME openings for shared main-trace/decode terminals".into(),
        ));
    }
    let wp_me = &mem_proof.wp_me_claims[0];
    if wp_me.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "W2 WP ME claim r mismatch (expected r_time)".into(),
        ));
    }
    if wp_me.c != step.mcs_inst.c {
        return Err(PiCcsError::ProtocolError("W2 WP ME claim commitment mismatch".into()));
    }
    if wp_me.m_in != step.mcs_inst.m_in {
        return Err(PiCcsError::ProtocolError("W2 WP ME claim m_in mismatch".into()));
    }
    let trace = Rv32TraceLayout::new();
    let wb_me = &mem_proof.wb_me_claims[0];
    let wb_cols = rv32_trace_wb_columns(&trace);
    let wb_open_map = require_time_openings_for_point(step_time_openings, wb_me.r.as_slice(), &wb_cols, "W2 WB")?;
    let wb_open_col = |col_id: usize| -> Result<K, PiCcsError> { named_opening(&wb_open_map, col_id, "W2 WB") };

    let wp_cols = rv32_trace_wp_opening_columns(&trace);
    let (_wp_entry, wp_open_map) =
        require_time_openings_covering_point(step_time_openings, wp_me.r.as_slice(), &wp_cols, "W2 WP")?;
    let wp_open_col = |col_id: usize| -> Result<K, PiCcsError> { named_opening(&wp_open_map, col_id, "W2 WP") };
    let decode_open_map = decode_open_map_from_instr_and_transport(
        step,
        cpu_bus,
        r_time,
        step_time_openings,
        wp_open_col(trace.instr_word)?,
        wp_open_col(trace.active)?,
        "W2 decode",
    )?;
    let decode_open_col =
        |col_id: usize| -> Result<K, PiCcsError> { named_opening(&decode_open_map, col_id, "W2 decode") };

    if let Some(claim_idx) = claim_plan.decode_fields {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w2/decode_fields claim index out of range".into(),
            ));
        }
        let opcode_flags = [
            decode_open_col(decode_layout.op_lui)?,
            decode_open_col(decode_layout.op_auipc)?,
            decode_open_col(decode_layout.op_jal)?,
            decode_open_col(decode_layout.op_jalr)?,
            decode_open_col(decode_layout.op_branch)?,
            decode_open_col(decode_layout.op_load)?,
            decode_open_col(decode_layout.op_store)?,
            decode_open_col(decode_layout.op_alu_imm)?,
            decode_open_col(decode_layout.op_alu_reg)?,
            decode_open_col(decode_layout.op_misc_mem)?,
            decode_open_col(decode_layout.op_system)?,
            decode_open_col(decode_layout.op_amo)?,
        ];
        let op_custom = decode_open_col(decode_layout.op_custom)?;
        let funct3_is = [
            decode_open_col(decode_layout.funct3_is[0])?,
            decode_open_col(decode_layout.funct3_is[1])?,
            decode_open_col(decode_layout.funct3_is[2])?,
            decode_open_col(decode_layout.funct3_is[3])?,
            decode_open_col(decode_layout.funct3_is[4])?,
            decode_open_col(decode_layout.funct3_is[5])?,
            decode_open_col(decode_layout.funct3_is[6])?,
            decode_open_col(decode_layout.funct3_is[7])?,
        ];
        let funct3_bits = [
            decode_open_col(decode_layout.funct3_bit[0])?,
            decode_open_col(decode_layout.funct3_bit[1])?,
            decode_open_col(decode_layout.funct3_bit[2])?,
        ];
        let funct7_bits = [
            decode_open_col(decode_layout.funct7_bit[0])?,
            decode_open_col(decode_layout.funct7_bit[1])?,
            decode_open_col(decode_layout.funct7_bit[2])?,
            decode_open_col(decode_layout.funct7_bit[3])?,
            decode_open_col(decode_layout.funct7_bit[4])?,
            decode_open_col(decode_layout.funct7_bit[5])?,
            decode_open_col(decode_layout.funct7_bit[6])?,
        ];
        let rs1_bits = [
            decode_open_col(decode_layout.rs1_bit[0])?,
            decode_open_col(decode_layout.rs1_bit[1])?,
            decode_open_col(decode_layout.rs1_bit[2])?,
            decode_open_col(decode_layout.rs1_bit[3])?,
            decode_open_col(decode_layout.rs1_bit[4])?,
        ];
        let rd_bits = [
            decode_open_col(decode_layout.rd_bit[0])?,
            decode_open_col(decode_layout.rd_bit[1])?,
            decode_open_col(decode_layout.rd_bit[2])?,
            decode_open_col(decode_layout.rd_bit[3])?,
            decode_open_col(decode_layout.rd_bit[4])?,
        ];
        let decode_rs1_addr = w2_reg_addr_from_bits(rs1_bits);
        let decode_rs2_addr = decode_open_col(decode_layout.rs2)?;
        let decode_rd_addr = w2_reg_addr_from_bits(rd_bits);
        let rd_is_zero = decode_open_col(decode_layout.rd_is_zero)?;
        let decode_rd_has_write = decode_open_col(decode_layout.rd_has_write)?;
        let imm_i = decode_open_col(decode_layout.imm_i)?;
        let decode_inputs = W2DecodeFieldsOpenings {
            active: wp_open_col(trace.active)?,
            halted: wb_open_col(trace.halted)?,
            is_virtual: wp_open_col(trace.is_virtual)?,
            virtual_sequence_remaining: wp_open_col(trace.virtual_sequence_remaining)?,
            trace_rs1_addr: wp_open_col(trace.rs1_addr)?,
            trace_rs2_addr: wp_open_col(trace.rs2_addr)?,
            trace_rd_addr: wp_open_col(trace.rd_addr)?,
            rs1_val: wp_open_col(trace.rs1_val)?,
            rs2_val: wp_open_col(trace.rs2_val)?,
            rd_val: wp_open_col(trace.rd_val)?,
            trace_rd_has_write: wp_open_col(trace.rd_has_write)?,
            ram_addr: wp_open_col(trace.ram_addr)?,
            shout_has_lookup: wp_open_col(trace.shout_has_lookup)?,
            shout_table_id: wp_open_col(trace.shout_table_id)?,
            shout_val: wp_open_col(trace.shout_val)?,
            shout_lhs: wp_open_col(trace.shout_lhs)?,
            shout_rhs: wp_open_col(trace.shout_rhs)?,
            shout_add_sub_key: wp_open_col(trace.shout_add_sub_key)?,
            decode_opcode: decode_open_col(decode_layout.opcode)?,
            decode_rs1_addr,
            decode_rs2_addr,
            decode_rd_addr,
            rd_is_zero,
            decode_rd_has_write,
            ram_has_read: decode_open_col(decode_layout.ram_has_read)?,
            ram_has_write: decode_open_col(decode_layout.ram_has_write)?,
            opcode_flags,
            op_custom,
            funct3_is,
            funct3_bits,
            funct7_bits,
            imm_i,
            imm_s: decode_open_col(decode_layout.imm_s)?,
        };
        let weights = w2_decode_pack_weight_vector(r_cycle, W2_FIELDS_RESIDUAL_COUNT);
        let weighted = w2_decode_fields_weighted_residual(&decode_inputs, &weights);
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w2/decode_fields terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.decode_immediates {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w2/decode_immediates claim index out of range".into(),
            ));
        }
        let residuals = w2_decode_immediate_residuals(
            decode_open_col(decode_layout.imm_i)?,
            decode_open_col(decode_layout.imm_s)?,
            decode_open_col(decode_layout.imm_b)?,
            decode_open_col(decode_layout.imm_j)?,
            [
                decode_open_col(decode_layout.rd_bit[0])?,
                decode_open_col(decode_layout.rd_bit[1])?,
                decode_open_col(decode_layout.rd_bit[2])?,
                decode_open_col(decode_layout.rd_bit[3])?,
                decode_open_col(decode_layout.rd_bit[4])?,
            ],
            [
                decode_open_col(decode_layout.funct3_bit[0])?,
                decode_open_col(decode_layout.funct3_bit[1])?,
                decode_open_col(decode_layout.funct3_bit[2])?,
            ],
            [
                decode_open_col(decode_layout.rs1_bit[0])?,
                decode_open_col(decode_layout.rs1_bit[1])?,
                decode_open_col(decode_layout.rs1_bit[2])?,
                decode_open_col(decode_layout.rs1_bit[3])?,
                decode_open_col(decode_layout.rs1_bit[4])?,
            ],
            [
                decode_open_col(decode_layout.rs2_bit[0])?,
                decode_open_col(decode_layout.rs2_bit[1])?,
                decode_open_col(decode_layout.rs2_bit[2])?,
                decode_open_col(decode_layout.rs2_bit[3])?,
                decode_open_col(decode_layout.rs2_bit[4])?,
            ],
            [
                decode_open_col(decode_layout.funct7_bit[0])?,
                decode_open_col(decode_layout.funct7_bit[1])?,
                decode_open_col(decode_layout.funct7_bit[2])?,
                decode_open_col(decode_layout.funct7_bit[3])?,
                decode_open_col(decode_layout.funct7_bit[4])?,
                decode_open_col(decode_layout.funct7_bit[5])?,
                decode_open_col(decode_layout.funct7_bit[6])?,
            ],
        );
        let mut weighted = K::ZERO;
        let weights = w2_decode_imm_weight_vector(r_cycle, residuals.len());
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w2/decode_immediates terminal value mismatch".into(),
            ));
        }
    }

    Ok(())
}

pub(crate) fn verify_route_a_width_terminals(
    cpu_bus: &BusLayout,
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    claim_plan: &RouteATimeClaimPlan,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
) -> Result<(), PiCcsError> {
    let any_w3_claim = claim_plan.width_bitness.is_some()
        || claim_plan.width_quiescence.is_some()
        || claim_plan.width_selector_linkage.is_some()
        || claim_plan.width_load_semantics.is_some()
        || claim_plan.width_store_semantics.is_some();
    if !any_w3_claim {
        return Ok(());
    }

    if mem_proof.wp_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "W3 requires WP ME openings for shared main-trace terminals".into(),
        ));
    }

    let trace = Rv32TraceLayout::new();
    let width = Rv32WidthSidecarLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();

    let wp_me = &mem_proof.wp_me_claims[0];
    if wp_me.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "W3 WP ME claim r mismatch (expected r_time)".into(),
        ));
    }
    if wp_me.c != step.mcs_inst.c {
        return Err(PiCcsError::ProtocolError("W3 WP ME claim commitment mismatch".into()));
    }
    if wp_me.m_in != step.mcs_inst.m_in {
        return Err(PiCcsError::ProtocolError("W3 WP ME claim m_in mismatch".into()));
    }
    let wp_cols = rv32_trace_wp_opening_columns(&trace);
    let (_wp_entry, wp_open_map) =
        require_time_openings_covering_point(step_time_openings, wp_me.r.as_slice(), &wp_cols, "W3 WP")?;
    let wp_open_col = |col_id: usize| -> Result<K, PiCcsError> { named_opening(&wp_open_map, col_id, "W3 WP") };

    let decode_open_map = decode_open_map_from_instr_and_transport(
        step,
        cpu_bus,
        r_time,
        step_time_openings,
        wp_open_col(trace.instr_word)?,
        wp_open_col(trace.active)?,
        "W3 decode",
    )?;
    let decode_open_col =
        |col_id: usize| -> Result<K, PiCcsError> { named_opening(&decode_open_map, col_id, "W3 decode") };
    let width_open_map =
        width_lookup_open_map_from_committed_openings(step, cpu_bus, r_time, step_time_openings, "W3 width")?;
    let width_open_col =
        |col_id: usize| -> Result<K, PiCcsError> { named_opening(&width_open_map, col_id, "W3 width") };

    let active = wp_open_col(trace.active)?;
    let rd_has_write = decode_open_col(decode.rd_has_write)?;
    let rd_val = wp_open_col(trace.rd_val)?;
    let ram_has_read = decode_open_col(decode.ram_has_read)?;
    let ram_has_write = decode_open_col(decode.ram_has_write)?;
    let ram_rv = wp_open_col(trace.ram_rv)?;
    let ram_wv = wp_open_col(trace.ram_wv)?;
    let rs2_val = wp_open_col(trace.rs2_val)?;

    let mut ram_rv_low_bits = [K::ZERO; 16];
    let mut rs2_low_bits = [K::ZERO; 16];
    for k in 0..16 {
        ram_rv_low_bits[k] = width_open_col(width.ram_rv_low_bit[k])?;
        rs2_low_bits[k] = width_open_col(width.rs2_low_bit[k])?;
    }
    let ram_rv_q16 = width_open_col(width.ram_rv_q16)?;
    let rs2_q16 = width_open_col(width.rs2_q16)?;
    let funct3_is = [
        decode_open_col(decode.funct3_is[0])?,
        decode_open_col(decode.funct3_is[1])?,
        decode_open_col(decode.funct3_is[2])?,
        decode_open_col(decode.funct3_is[3])?,
        decode_open_col(decode.funct3_is[4])?,
        decode_open_col(decode.funct3_is[5])?,
        decode_open_col(decode.funct3_is[6])?,
        decode_open_col(decode.funct3_is[7])?,
    ];
    let op_load = decode_open_col(decode.op_load)?;
    let op_store = decode_open_col(decode.op_store)?;
    let load_flags = [
        op_load * funct3_is[0],
        op_load * funct3_is[4],
        op_load * funct3_is[1],
        op_load * funct3_is[5],
        op_load * funct3_is[2],
    ];
    let store_flags = [
        op_store * funct3_is[0],
        op_store * funct3_is[1],
        op_store * funct3_is[2],
    ];

    if let Some(claim_idx) = claim_plan.width_bitness {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError("w3/bitness claim index out of range".into()));
        }
        let mut bitness_open = Vec::with_capacity(32);
        bitness_open.extend_from_slice(&ram_rv_low_bits);
        bitness_open.extend_from_slice(&rs2_low_bits);
        let weights = w3_bitness_weight_vector(r_cycle, bitness_open.len());
        let mut weighted = K::ZERO;
        for (b, w) in bitness_open.iter().zip(weights.iter()) {
            weighted += *w * *b * (*b - K::ONE);
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError("w3/bitness terminal value mismatch".into()));
        }
    }

    if let Some(claim_idx) = claim_plan.width_quiescence {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w3/quiescence claim index out of range".into(),
            ));
        }
        let mut quiescence_open = vec![ram_rv_q16, rs2_q16];
        quiescence_open.extend_from_slice(&ram_rv_low_bits);
        quiescence_open.extend_from_slice(&rs2_low_bits);
        let weights = w3_quiescence_weight_vector(r_cycle, quiescence_open.len());
        let mut weighted = K::ZERO;
        for (v, w) in quiescence_open.iter().zip(weights.iter()) {
            weighted += *w * *v;
        }
        let expected = eq_points(r_time, r_cycle) * (K::ONE - active) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w3/quiescence terminal value mismatch".into(),
            ));
        }
    }

    if claim_plan.width_selector_linkage.is_some() {
        return Err(PiCcsError::ProtocolError(
            "w3/selector_linkage must be disabled in reduced width-sidecar mode".into(),
        ));
    }

    if let Some(claim_idx) = claim_plan.width_load_semantics {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w3/load_semantics claim index out of range".into(),
            ));
        }
        let residuals = w3_load_semantics_residuals(
            rd_val,
            ram_rv,
            rd_has_write,
            ram_has_read,
            load_flags,
            ram_rv_q16,
            ram_rv_low_bits,
        );
        let weights = w3_load_weight_vector(r_cycle, residuals.len());
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w3/load_semantics terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.width_store_semantics {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w3/store_semantics claim index out of range".into(),
            ));
        }
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
        let weights = w3_store_weight_vector(r_cycle, residuals.len());
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w3/store_semantics terminal value mismatch".into(),
            ));
        }
    }

    Ok(())
}

pub(crate) fn verify_route_a_control_terminals(
    cpu_bus: &BusLayout,
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    claim_plan: &RouteATimeClaimPlan,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
) -> Result<(), PiCcsError> {
    let any_control_claim = claim_plan.control_next_pc_linear.is_some()
        || claim_plan.control_next_pc_control.is_some()
        || claim_plan.control_branch_semantics.is_some()
        || claim_plan.control_writeback.is_some();
    if !any_control_claim {
        return Ok(());
    }

    if mem_proof.wp_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "control stage requires WP ME openings for main-trace terminals".into(),
        ));
    }
    let trace = Rv32TraceLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();

    let wp_me = &mem_proof.wp_me_claims[0];
    if wp_me.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "control stage WP ME claim r mismatch (expected r_time)".into(),
        ));
    }
    if wp_me.c != step.mcs_inst.c {
        return Err(PiCcsError::ProtocolError(
            "control stage WP ME claim commitment mismatch".into(),
        ));
    }
    if wp_me.m_in != step.mcs_inst.m_in {
        return Err(PiCcsError::ProtocolError(
            "control stage WP ME claim m_in mismatch".into(),
        ));
    }
    let wp_base_cols = rv32_trace_wp_opening_columns(&trace);
    let control_extra_cols = rv32_trace_control_extra_opening_columns(&trace);
    let mut wp_all_cols = wp_base_cols.clone();
    wp_all_cols.extend(control_extra_cols.iter().copied());
    let wp_open_map =
        require_time_openings_for_point(step_time_openings, wp_me.r.as_slice(), &wp_all_cols, "control stage WP")?;
    let wp_open_col =
        |col_id: usize| -> Result<K, PiCcsError> { named_opening(&wp_open_map, col_id, "control stage WP") };
    let decode_open_map = decode_lookup_open_map_from_committed_openings(
        step,
        cpu_bus,
        r_time,
        step_time_openings,
        "control stage decode",
    )?;
    let decode_open_col =
        |col_id: usize| -> Result<K, PiCcsError> { named_opening(&decode_open_map, col_id, "control stage decode") };

    let active = wp_open_col(trace.active)?;
    let is_virtual = wp_open_col(trace.is_virtual)?;
    let pc_before = wp_open_col(trace.pc_before)?;
    let pc_after = wp_open_col(trace.pc_after)?;
    let rs1_val = wp_open_col(trace.rs1_val)?;
    let rd_val = wp_open_col(trace.rd_val)?;
    let jalr_drop_bit = wp_open_col(trace.jalr_drop_bit)?;
    let shout_val = wp_open_col(trace.shout_val)?;
    let funct3_bits = [
        decode_open_col(decode.funct3_bit[0])?,
        decode_open_col(decode.funct3_bit[1])?,
        decode_open_col(decode.funct3_bit[2])?,
    ];
    let rs1_bits = [
        decode_open_col(decode.rs1_bit[0])?,
        decode_open_col(decode.rs1_bit[1])?,
        decode_open_col(decode.rs1_bit[2])?,
        decode_open_col(decode.rs1_bit[3])?,
        decode_open_col(decode.rs1_bit[4])?,
    ];
    let rs2_bits = [
        decode_open_col(decode.rs2_bit[0])?,
        decode_open_col(decode.rs2_bit[1])?,
        decode_open_col(decode.rs2_bit[2])?,
        decode_open_col(decode.rs2_bit[3])?,
        decode_open_col(decode.rs2_bit[4])?,
    ];
    let funct7_bits = [
        decode_open_col(decode.funct7_bit[0])?,
        decode_open_col(decode.funct7_bit[1])?,
        decode_open_col(decode.funct7_bit[2])?,
        decode_open_col(decode.funct7_bit[3])?,
        decode_open_col(decode.funct7_bit[4])?,
        decode_open_col(decode.funct7_bit[5])?,
        decode_open_col(decode.funct7_bit[6])?,
    ];

    let op_lui = decode_open_col(decode.op_lui)?;
    let op_auipc = decode_open_col(decode.op_auipc)?;
    let op_jal = decode_open_col(decode.op_jal)?;
    let op_jalr = decode_open_col(decode.op_jalr)?;
    let op_branch = decode_open_col(decode.op_branch)?;
    let op_load = decode_open_col(decode.op_load)?;
    let op_store = decode_open_col(decode.op_store)?;
    let op_alu_imm = decode_open_col(decode.op_alu_imm)?;
    let op_alu_reg = decode_open_col(decode.op_alu_reg)?;
    let op_misc_mem = decode_open_col(decode.op_misc_mem)?;
    let op_system = decode_open_col(decode.op_system)?;
    let op_amo = decode_open_col(decode.op_amo)?;
    let op_custom = decode_open_col(decode.op_custom)?;
    let rd_is_zero = decode_open_col(decode.rd_is_zero)?;
    let op_lui_write = op_lui * (K::ONE - rd_is_zero);
    let op_auipc_write = op_auipc * (K::ONE - rd_is_zero);
    let op_jal_write = op_jal * (K::ONE - rd_is_zero);
    let op_jalr_write = op_jalr * (K::ONE - rd_is_zero);
    let imm_i = decode_open_col(decode.imm_i)?;
    let imm_b = decode_open_col(decode.imm_b)?;
    let imm_j = decode_open_col(decode.imm_j)?;
    let funct3_is6 = decode_open_col(decode.funct3_is[6])?;
    let funct3_is7 = decode_open_col(decode.funct3_is[7])?;

    if let Some(claim_idx) = claim_plan.control_next_pc_linear {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "control/next_pc_linear claim index out of range".into(),
            ));
        }
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
        let weights = control_next_pc_linear_weight_vector(r_cycle, 1);
        let expected = eq_points(r_time, r_cycle) * weights[0] * residual;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "control/next_pc_linear terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.control_next_pc_control {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "control/next_pc_control claim index out of range".into(),
            ));
        }
        let residuals = control_next_pc_control_residuals(
            active,
            pc_before,
            pc_after,
            rs1_val,
            jalr_drop_bit,
            imm_i,
            imm_b,
            imm_j,
            funct7_bits[6],
            op_jal,
            op_jalr,
            op_branch,
            shout_val,
            funct3_bits[0],
        );
        let weights = control_next_pc_control_weight_vector(r_cycle, residuals.len());
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "control/next_pc_control terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.control_branch_semantics {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "control/branch_semantics claim index out of range".into(),
            ));
        }
        let residuals = control_branch_semantics_residuals(
            op_branch,
            shout_val,
            funct3_bits[0],
            funct3_bits[1],
            funct3_bits[2],
            funct3_is6,
            funct3_is7,
        );
        let weights = control_branch_semantics_weight_vector(r_cycle, residuals.len());
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "control/branch_semantics terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.control_writeback {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "control/writeback claim index out of range".into(),
            ));
        }
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
        let weights = control_writeback_weight_vector(r_cycle, residuals.len());
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "control/writeback terminal value mismatch".into(),
            ));
        }
    }

    Ok(())
}
