use super::*;

#[inline]
fn packed_trace_link_operands(op: Rv32PackedShoutOp, addr_bits: &[K]) -> Result<(K, K), PiCcsError> {
    let lhs = *addr_bits
        .first()
        .ok_or_else(|| PiCcsError::InvalidInput("packed Shout trace linkage requires lhs in addr_bits[0]".into()))?;
    let rhs = match op {
        Rv32PackedShoutOp::Sll | Rv32PackedShoutOp::Srl | Rv32PackedShoutOp::Sra => {
            if addr_bits.len() < 6 {
                return Err(PiCcsError::InvalidInput(
                    "packed shift trace linkage requires 5 shamt bits after lhs".into(),
                ));
            }
            let two = K::from(F::from_u64(2));
            let mut pow = K::ONE;
            let mut acc = K::ZERO;
            for bit in addr_bits.iter().skip(1).take(5) {
                acc += pow * *bit;
                pow *= two;
            }
            acc
        }
        _ => *addr_bits.get(1).ok_or_else(|| {
            PiCcsError::InvalidInput("packed Shout trace linkage requires rhs in addr_bits[1]".into())
        })?,
    };
    Ok((lhs, rhs))
}

pub fn verify_route_a_memory_step(
    tr: &mut Poseidon2Transcript,
    cpu_bus: &BusLayout,
    m: usize,
    core_t: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    prev_step: Option<&StepInstanceBundle<Cmt, F, K>>,
    ccs_out0: &CeClaim<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    batched_claimed_sums: &[K],
    rv64_fullword_width_stage_from_proof: bool,
    claim_idx_start: usize,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
    prev_step_time_openings: Option<&[crate::shard_proof_types::TimePointOpening]>,
    shout_pre: &[ShoutAddrPreVerifyData],
    twist_pre: &[TwistAddrPreVerifyData],
    step_idx: usize,
    poseidon_link_chals: Option<&PoseidonLinkChallenges>,
    poseidon_cont_chals: Option<&PoseidonContinuityChallenges>,
) -> Result<RouteAMemoryVerifyOutput, PiCcsError> {
    let chi_cycle_at_r_time = eq_points(r_time, r_cycle);
    let trace_mode = wb_wp_required_for_step_instance(step);
    let cpu_link = if trace_mode {
        extract_trace_cpu_link_openings(m, core_t, cpu_bus.bus_cols, step, ccs_out0, step_time_openings, r_time)?
    } else {
        None
    };
    let enforce_trace_shout_linkage = trace_mode && !step.lut_insts.is_empty();
    if enforce_trace_shout_linkage && cpu_link.is_none() {
        return Err(PiCcsError::ProtocolError(
            "missing CPU trace linkage openings in shared-bus mode".into(),
        ));
    }
    let has_prev = prev_step.is_some();
    if let Some(prev) = prev_step {
        if prev.mem_insts.len() != step.mem_insts.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist rollover requires stable mem instance count: prev has {}, current has {}",
                prev.mem_insts.len(),
                step.mem_insts.len()
            )));
        }
        for (idx, (prev_inst, inst)) in prev.mem_insts.iter().zip(step.mem_insts.iter()).enumerate() {
            if prev_inst.d != inst.d
                || prev_inst.ell != inst.ell
                || prev_inst.k != inst.k
                || prev_inst.lanes != inst.lanes
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist rollover requires stable geometry at mem_idx={}: prev (k={}, d={}, ell={}, lanes={}) vs cur (k={}, d={}, ell={}, lanes={})",
                    idx,
                    prev_inst.k,
                    prev_inst.d,
                    prev_inst.ell,
                    prev_inst.lanes,
                    inst.k,
                    inst.d,
                    inst.ell,
                    inst.lanes
                )));
            }
        }
    }

    for (idx, inst) in step.lut_insts.iter().enumerate() {
        if !inst.comms.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Shout instances (comms must be empty, lut_idx={idx})"
            )));
        }
    }
    for (idx, inst) in step.mem_insts.iter().enumerate() {
        if !inst.comms.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Twist instances (comms must be empty, mem_idx={idx})"
            )));
        }
    }
    if let Some(prev) = prev_step {
        for (idx, inst) in prev.lut_insts.iter().enumerate() {
            if !inst.comms.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared CPU bus requires metadata-only Shout instances (comms must be empty, prev lut_idx={idx})"
                )));
            }
        }
        for (idx, inst) in prev.mem_insts.iter().enumerate() {
            if !inst.comms.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared CPU bus requires metadata-only Twist instances (comms must be empty, prev mem_idx={idx})"
                )));
            }
        }
    }

    let proofs_mem = &mem_proof.proofs;

    if cpu_bus.shout_cols.len() != step.lut_insts.len() || cpu_bus.twist_cols.len() != step.mem_insts.len() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus layout mismatch for step (instance counts)".into(),
        ));
    }
    let strict_committed_mode_for_step = |st: &StepInstanceBundle<Cmt, F, K>| -> bool {
        let cpu_cols_len = st.time_columns.cpu_cols.len();
        let mem_cols_len = st.time_columns.mem_cols.len();
        let expected_logical_cols = cpu_cols_len.saturating_add(mem_cols_len);
        st.time_columns.t == cpu_bus.chunk_size
            && mem_cols_len == cpu_bus.bus_cols
            && st.time_columns.col_ids.len() == expected_logical_cols
            && expected_logical_cols > 0
    };
    let strict_committed_mode = strict_committed_mode_for_step(step);
    if cpu_bus.bus_cols > 0 && !strict_committed_mode {
        return Err(PiCcsError::ProtocolError(format!(
            "route-a: canonical committed mode required for bus openings (time_t={}, chunk_size={}, mem_cols={}, bus_cols={}, logical_col_ids={})",
            step.time_columns.t,
            cpu_bus.chunk_size,
            step.time_columns.mem_cols.len(),
            cpu_bus.bus_cols,
            step.time_columns.col_ids.len()
        )));
    }

    let bus_time_open_map = if cpu_bus.bus_cols > 0 {
        let bus_col_ids = bus_logical_col_ids_for_step_instance(step, cpu_bus, "route-a/time")?;
        let opening_entry =
            require_time_opening_entry_for_point(step_time_openings, r_time, &bus_col_ids, "route-a/time")?;
        if opening_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "route-a/time requires CommittedOpening source (got {:?})",
                opening_entry.source
            )));
        }
        let logical_map = require_time_openings_for_point(step_time_openings, r_time, &bus_col_ids, "route-a/time")?;
        let mut local_map = BTreeMap::new();
        for (mem_local_col, &logical_col_id) in bus_col_ids.iter().enumerate() {
            let v = logical_map.get(&logical_col_id).copied().ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "route-a/time: missing logical opening value for mem_local_col={mem_local_col} logical_col_id={logical_col_id}"
                ))
            })?;
            local_map.insert(mem_local_col, v);
        }
        local_map
    } else {
        BTreeMap::new()
    };
    let wb_enabled = wb_wp_required_for_step_instance(step);
    let wp_enabled = wb_wp_required_for_step_instance(step);
    let w2_enabled = decode_stage_required_for_step_instance(step);
    let w3_enabled = width_stage_required_for_step_instance(step) || rv64_fullword_width_stage_from_proof;
    let control_enabled = control_stage_required_for_step_instance(step);
    let poseidon_cycle_enabled = RouteATimeClaimPlan::poseidon_stage_required_for_step_instance(step)?;
    let claim_plan = RouteATimeClaimPlan::build(
        step,
        claim_idx_start,
        wb_enabled,
        wp_enabled,
        w2_enabled,
        w3_enabled,
        control_enabled,
        poseidon_cycle_enabled,
    )?;
    if claim_plan.claim_idx_end > batched_final_values.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "batched_final_values too short (need at least {}, have {})",
            claim_plan.claim_idx_end,
            batched_final_values.len()
        )));
    }
    if claim_plan.claim_idx_end > batched_claimed_sums.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "batched_claimed_sums too short (need at least {}, have {})",
            claim_plan.claim_idx_end,
            batched_claimed_sums.len()
        )));
    }
    let is_virtual_open = if trace_mode {
        if mem_proof.wp_me_claims.len() != 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "virtual-domain check requires one WP ME claim (got {})",
                mem_proof.wp_me_claims.len()
            )));
        }
        let wp_me = &mem_proof.wp_me_claims[0];
        if wp_me.r.as_slice() != r_time {
            return Err(PiCcsError::ProtocolError(
                "virtual-domain check: WP ME r mismatch".into(),
            ));
        }
        if wp_me.c != step.mcs_inst.c {
            return Err(PiCcsError::ProtocolError(
                "virtual-domain check: WP ME commitment mismatch".into(),
            ));
        }
        if wp_me.m_in != step.mcs_inst.m_in {
            return Err(PiCcsError::ProtocolError(
                "virtual-domain check: WP ME m_in mismatch".into(),
            ));
        }
        let trace_layout = Rv32TraceLayout::new();
        let wp_cols = rv32_trace_wp_opening_columns(&trace_layout);
        let (wp_entry, wp_openings) =
            require_time_openings_covering_point(step_time_openings, r_time, &wp_cols, "virtual-domain check/WP")?;
        if wp_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "virtual-domain check/WP requires CommittedOpening source (got {:?})",
                wp_entry.source
            )));
        }
        named_opening(
            &wp_openings,
            trace_layout.is_virtual,
            "virtual-domain check: is_virtual",
        )?
    } else {
        K::ZERO
    };

    let expected_proofs = step.lut_insts.len() + step.mem_insts.len();
    if proofs_mem.len() != expected_proofs {
        return Err(PiCcsError::InvalidInput(format!(
            "mem proof count mismatch (expected {}, got {})",
            expected_proofs,
            proofs_mem.len()
        )));
    }
    let total_shout_lanes: usize = step.lut_insts.iter().map(|inst| inst.lanes.max(1)).sum();
    if shout_pre.len() != total_shout_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected total_lanes={}, got {})",
            total_shout_lanes,
            shout_pre.len()
        )));
    }
    if twist_pre.len() != step.mem_insts.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time count mismatch (expected {}, got {})",
            step.mem_insts.len(),
            twist_pre.len()
        )));
    }

    let mut twist_time_openings: Vec<TwistTimeLaneOpenings> = Vec::with_capacity(step.mem_insts.len());

    // Shout instances first.
    let mut shout_lane_base: usize = 0;
    let mut shout_trace_sums = ShoutTraceLinkSums::default();
    #[derive(Clone)]
    struct ShoutGammaLaneVerifyData {
        has_lookup: K,
        val: K,
        addr_bits: Vec<K>,
        pre: ShoutAddrPreVerifyData,
    }
    let mut shout_addr_range_counts_interleaved = std::collections::HashMap::<(usize, usize), usize>::new();
    let mut shout_addr_range_counts_combined = std::collections::HashMap::<(usize, usize), usize>::new();
    if enforce_trace_shout_linkage {
        for (inst, inst_cols) in step.lut_insts.iter().zip(cpu_bus.shout_cols.iter()) {
            let table_id = rv32_trace_link_table_id_from_spec(&inst.table_spec)?;
            let is_combined_lane = table_id
                .map(neo_memory::riscv::trace::rv32_trace_uses_combined_operand_key_table_id)
                .unwrap_or(false);
            for lane_cols in inst_cols.lanes.iter() {
                let key = (lane_cols.addr_bits.start, lane_cols.addr_bits.end);
                if table_id.is_some() {
                    if is_combined_lane {
                        *shout_addr_range_counts_combined.entry(key).or_insert(0) += 1;
                    } else {
                        *shout_addr_range_counts_interleaved.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    let mut shout_gamma_lane_data: Vec<Option<ShoutGammaLaneVerifyData>> = vec![None; total_shout_lanes];
    for (proof_idx, inst) in step.lut_insts.iter().enumerate() {
        match &proofs_mem[proof_idx] {
            MemOrLutProof::Shout(_proof) => {}
            _ => return Err(PiCcsError::InvalidInput("expected Shout proof".into())),
        }
        let packed_layout = rv32_packed_shout_layout(&inst.table_spec)?;
        if matches!(packed_layout, Some((_op, _xlen, time_bits)) if time_bits != 0) {
            return Err(PiCcsError::InvalidInput(
                "RiscvOpcodeEventTablePacked is not supported in shared-bus Route-A verification".into(),
            ));
        }
        let packed_xlen = packed_layout.map(|(_op, xlen, _)| xlen).unwrap_or(0);
        let packed_opcode = match &inst.table_spec {
            Some(LutTableSpec::RiscvOpcodePacked { opcode, xlen }) => {
                if !neo_memory::riscv::packed::rv_packed_supported_opcode(*opcode, *xlen) {
                    return Err(PiCcsError::InvalidInput(format!(
                        "unsupported packed RISC-V shout spec in Route-A verification: opcode={opcode:?}, xlen={xlen}"
                    )));
                }
                Some(*opcode)
            }
            _ => None,
        };
        let packed_trace_op = packed_layout.map(|(op, _, _)| op);

        let ell_addr = inst.d * inst.ell;
        let expected_lanes = inst.lanes.max(1);
        let lane_table_id_u32 = if enforce_trace_shout_linkage {
            rv32_trace_link_table_id_from_spec(&inst.table_spec)?
        } else {
            None
        };
        let lane_table_id = if let Some(table_id) = lane_table_id_u32 {
            Some(K::from(F::from_u64(table_id as u64)))
        } else {
            None
        };

        let inst_cols = cpu_bus
            .shout_cols
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (shout)".into()))?;
        if inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at lut_idx={proof_idx}: bus shout lanes={} but instance expects {expected_lanes}",
                inst_cols.lanes.len()
            )));
        }

        struct ShoutLaneOpen {
            addr_bits: Vec<K>,
            has_lookup: K,
            val: K,
            addr_key: (usize, usize),
        }
        let mut lane_opens: Vec<ShoutLaneOpen> = Vec::with_capacity(expected_lanes);
        for (lane_idx, shout_cols) in inst_cols.lanes.iter().enumerate() {
            if shout_cols.addr_bits.end - shout_cols.addr_bits.start != ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at lut_idx={proof_idx}, lane_idx={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut addr_bits_open = Vec::with_capacity(ell_addr);
            for (_j, col_id) in shout_cols.addr_bits.clone().enumerate() {
                addr_bits_open.push(named_opening(
                    &bus_time_open_map,
                    col_id,
                    "route-a/time shout addr_bits",
                )?);
            }
            let has_lookup_open = named_opening(
                &bus_time_open_map,
                shout_cols.has_lookup,
                "route-a/time shout has_lookup",
            )?;
            let val_open = named_opening(&bus_time_open_map, shout_cols.primary_val(), "route-a/time shout val")?;
            let key = (shout_cols.addr_bits.start, shout_cols.addr_bits.end);
            lane_opens.push(ShoutLaneOpen {
                addr_bits: addr_bits_open,
                has_lookup: has_lookup_open,
                val: val_open,
                addr_key: key,
            });
        }

        let shout_claims = claim_plan
            .shout
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing Shout claim schedule at index {}", proof_idx)))?;
        if shout_claims.lanes.len() != expected_lanes {
            return Err(PiCcsError::ProtocolError(format!(
                "Shout claim schedule lane count mismatch at lut_idx={proof_idx}: expected {expected_lanes}, got {}",
                shout_claims.lanes.len()
            )));
        }
        if shout_claims.transport_only && shout_claims.bitness.is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "transport-only shout table unexpectedly has bitness claim at lut_idx={proof_idx}"
            )));
        }
        if shout_lane_base
            .checked_add(expected_lanes)
            .ok_or_else(|| PiCcsError::ProtocolError("shout lane index overflow".into()))?
            > shout_pre.len()
        {
            return Err(PiCcsError::ProtocolError("Shout pre-time lane indexing drift".into()));
        }

        // Route A Shout ordering in batched_time:
        // - value (time rounds only) per lane
        // - adapter (time rounds only) per lane
        // - aggregated bitness for ungrouped lanes only (grouped lanes are checked in gamma groups)
        if !shout_claims.transport_only {
            for lane_claim in shout_claims.lanes.iter() {
                if lane_claim.transport_only {
                    continue;
                }
                if lane_claim.gamma_group.is_none() {
                    if lane_claim.value.is_none() || lane_claim.adapter.is_none() {
                        return Err(PiCcsError::ProtocolError(
                            "missing shout lane claim indices for non-grouped lane".into(),
                        ));
                    }
                } else if lane_claim.value.is_some() || lane_claim.adapter.is_some() {
                    return Err(PiCcsError::ProtocolError(
                        "grouped shout lane must not carry direct value/adapter indices".into(),
                    ));
                }
            }
        } else {
            for lane_claim in shout_claims.lanes.iter() {
                if lane_claim.value.is_some()
                    || lane_claim.adapter.is_some()
                    || lane_claim.event_table_hash.is_some()
                    || lane_claim.gamma_group.is_some()
                    || !lane_claim.transport_only
                {
                    return Err(PiCcsError::ProtocolError(format!(
                        "transport-only shout lane schedule drift at lut_idx={proof_idx}"
                    )));
                }
            }
        }
        if let Some(bitness_idx) = shout_claims.bitness {
            let mut opens: Vec<K> = Vec::new();
            if let Some(op) = packed_opcode {
                for (lane_idx, lane) in lane_opens.iter().enumerate() {
                    let lane_claims = shout_claims
                        .lanes
                        .get(lane_idx)
                        .ok_or_else(|| PiCcsError::ProtocolError("shout claim schedule lane idx drift".into()))?;
                    if lane_claims.gamma_group.is_some() {
                        continue;
                    }
                    let mut lane_terms = neo_memory::riscv::packed::rv_collect_packed_bitness_terms(
                        op,
                        packed_xlen,
                        lane.addr_bits.as_slice(),
                        lane.has_lookup,
                        lane.val,
                    )?;
                    opens.append(&mut lane_terms);
                }
            } else {
                opens.reserve(expected_lanes * (ell_addr + 1));
                for (lane_idx, lane) in lane_opens.iter().enumerate() {
                    let lane_claims = shout_claims
                        .lanes
                        .get(lane_idx)
                        .ok_or_else(|| PiCcsError::ProtocolError("shout claim schedule lane idx drift".into()))?;
                    if lane_claims.gamma_group.is_some() {
                        continue;
                    }
                    opens.extend_from_slice(&lane.addr_bits);
                    opens.push(lane.has_lookup);
                }
            }
            let weights = bitness_weights(r_cycle, opens.len(), 0x5348_4F55_54u64 + proof_idx as u64);
            let mut acc = K::ZERO;
            for (w, b) in weights.iter().zip(opens.iter()) {
                acc += *w * *b * (*b - K::ONE);
            }
            let expected = chi_cycle_at_r_time * acc;
            if expected != batched_final_values[bitness_idx] {
                return Err(PiCcsError::ProtocolError(
                    "shout/bitness terminal value mismatch".into(),
                ));
            }
        } else if !shout_claims.transport_only {
            let has_non_grouped_lane = shout_claims
                .lanes
                .iter()
                .any(|lane| lane.gamma_group.is_none());
            if has_non_grouped_lane {
                return Err(PiCcsError::ProtocolError(format!(
                    "missing shout bitness claim for non-grouped lanes at lut_idx={proof_idx}"
                )));
            }
        }

        for (lane_idx, lane) in lane_opens.iter().enumerate() {
            if let Some(lane_table_id) = lane_table_id {
                shout_trace_sums.has_lookup += lane.has_lookup;
                shout_trace_sums.val += lane.val;
                shout_trace_sums.table_id += lane.has_lookup * lane_table_id;
                let is_combined_lane = lane_table_id_u32
                    .map(neo_memory::riscv::trace::rv32_trace_uses_combined_operand_key_table_id)
                    .unwrap_or(false);
                if is_combined_lane {
                    let bits_to_scalar = |bits: &[K]| {
                        let two = K::from(F::from_u64(2));
                        let mut pow = K::ONE;
                        let mut acc = K::ZERO;
                        for bit in bits.iter() {
                            acc += pow * *bit;
                            pow *= two;
                        }
                        acc
                    };
                    let key = bits_to_scalar(lane.addr_bits.as_slice());
                    let group_size = shout_addr_range_counts_combined
                        .get(&lane.addr_key)
                        .copied()
                        .unwrap_or(1);
                    if group_size > 1 {
                        let inv_count = K::from_u64(group_size as u64).inverse();
                        shout_trace_sums.add_sub_key += key * inv_count;
                    } else {
                        shout_trace_sums.add_sub_key += key;
                    }
                } else {
                    let (lhs, rhs) = if let Some(op) = packed_trace_op {
                        packed_trace_link_operands(op, lane.addr_bits.as_slice())?
                    } else {
                        unpack_interleaved_halves_lsb(&lane.addr_bits)?
                    };
                    let group_size = shout_addr_range_counts_interleaved
                        .get(&lane.addr_key)
                        .copied()
                        .unwrap_or(1);
                    if group_size > 1 {
                        let inv_count = K::from_u64(group_size as u64).inverse();
                        shout_trace_sums.link_lhs += lhs * inv_count;
                        shout_trace_sums.link_rhs += rhs * inv_count;
                    } else {
                        shout_trace_sums.link_lhs += lhs;
                        shout_trace_sums.link_rhs += rhs;
                    }
                }
            }

            let pre = shout_pre.get(shout_lane_base + lane_idx).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "missing pre-time Shout lane data at index {}",
                    shout_lane_base + lane_idx
                ))
            })?;
            let lane_claims = shout_claims
                .lanes
                .get(lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout claim schedule lane idx drift".into()))?;
            if shout_claims.transport_only {
                continue;
            }

            if lane_claims.gamma_group.is_some() {
                if packed_opcode.is_some() {
                    return Err(PiCcsError::ProtocolError(
                        "packed shout lane unexpectedly assigned to gamma group".into(),
                    ));
                }
                if !pre.is_active {
                    if pre.addr_claim_sum != K::ZERO || pre.addr_final != K::ZERO || lane.has_lookup != K::ZERO {
                        return Err(PiCcsError::ProtocolError(
                            "shout gamma lane inactive-row invariants violated".into(),
                        ));
                    }
                }
                shout_gamma_lane_data[shout_lane_base + lane_idx] = Some(ShoutGammaLaneVerifyData {
                    has_lookup: lane.has_lookup,
                    val: lane.val,
                    addr_bits: lane.addr_bits.clone(),
                    pre: pre.clone(),
                });
            } else {
                let value_idx = lane_claims
                    .value
                    .ok_or_else(|| PiCcsError::ProtocolError("missing shout value claim idx".into()))?;
                let adapter_idx = lane_claims
                    .adapter
                    .ok_or_else(|| PiCcsError::ProtocolError("missing shout adapter claim idx".into()))?;
                let value_claim = batched_claimed_sums[value_idx];
                let value_final = batched_final_values[value_idx];
                let adapter_claim = batched_claimed_sums[adapter_idx];
                let adapter_final = batched_final_values[adapter_idx];

                if packed_opcode.is_some() {
                    // Packed Route-A lanes are verified as zero-sum constraints. The claimed sums
                    // must be zero, but terminal evaluations at the sampled random point are not
                    // required to be zero in general.
                    if value_claim != K::ZERO || adapter_claim != K::ZERO {
                        return Err(PiCcsError::ProtocolError(format!(
                            "packed shout lane zero-claim invariant mismatch at lut_idx={proof_idx}, lane_idx={lane_idx}: value_claim={value_claim:?}, adapter_claim={adapter_claim:?}"
                        )));
                    }
                    if pre.is_active
                        || pre.addr_claim_sum != K::ZERO
                        || pre.addr_final != K::ZERO
                        || pre.table_eval_at_r_addr != K::ZERO
                    {
                        return Err(PiCcsError::ProtocolError(
                            "packed shout lane addr-pre invariants mismatch".into(),
                        ));
                    }
                    continue;
                }

                let expected_value_final = chi_cycle_at_r_time * lane.has_lookup * lane.val;
                if expected_value_final != value_final {
                    return Err(PiCcsError::ProtocolError("shout value terminal value mismatch".into()));
                }

                let eq_addr = eq_bits_prod(&lane.addr_bits, &pre.r_addr)?;
                let expected_adapter_final = chi_cycle_at_r_time * lane.has_lookup * eq_addr;
                if expected_adapter_final != adapter_final {
                    return Err(PiCcsError::ProtocolError(
                        "shout adapter terminal value mismatch".into(),
                    ));
                }

                if value_claim != pre.addr_claim_sum {
                    return Err(PiCcsError::ProtocolError(
                        "shout value claimed sum != addr claimed sum".into(),
                    ));
                }

                if pre.is_active {
                    let expected_addr_final = pre.table_eval_at_r_addr * adapter_claim;
                    if expected_addr_final != pre.addr_final {
                        return Err(PiCcsError::ProtocolError("shout addr terminal value mismatch".into()));
                    }
                } else {
                    // If we skipped the addr-pre sumcheck, the only sound case is "no lookups".
                    // Enforce this by requiring the addr claim + adapter claim to be zero.
                    if pre.addr_claim_sum != K::ZERO {
                        return Err(PiCcsError::ProtocolError(
                            "shout addr-pre skipped but addr claim is nonzero".into(),
                        ));
                    }
                    if adapter_claim != K::ZERO {
                        return Err(PiCcsError::ProtocolError(
                            "shout addr-pre skipped but adapter claim is nonzero".into(),
                        ));
                    }
                    if pre.addr_final != K::ZERO {
                        return Err(PiCcsError::ProtocolError(
                            "shout addr-pre skipped but addr_final is nonzero".into(),
                        ));
                    }
                }
            }
        }

        shout_lane_base += expected_lanes;
    }
    if shout_lane_base != shout_pre.len() {
        return Err(PiCcsError::ProtocolError(
            "shout pre-time lanes not fully consumed".into(),
        ));
    }
    if !step.lut_insts.is_empty() && enforce_trace_shout_linkage {
        let cpu = cpu_link
            .ok_or_else(|| PiCcsError::ProtocolError("missing CPU trace linkage openings in shared-bus mode".into()))?;
        let expected_table_id =
            expected_trace_shout_table_id_from_openings(step, cpu_bus, mem_proof, step_time_openings, r_time)?;
        verify_non_event_trace_shout_linkage(cpu, shout_trace_sums, expected_table_id)?;
    }

    for group in claim_plan.shout_gamma_groups.iter() {
        let weights = bitness_weights(r_cycle, group.lanes.len(), 0x5348_5F47_414D_4Du64 ^ group.key);
        let value_claim = batched_claimed_sums[group.value];
        let value_final = batched_final_values[group.value];
        let adapter_claim = batched_claimed_sums[group.adapter];
        let adapter_final = batched_final_values[group.adapter];
        let bitness_final = batched_final_values[group.bitness];

        let mut expected_value_claim = K::ZERO;
        let mut expected_value_final = K::ZERO;
        let mut expected_adapter_claim = K::ZERO;
        let mut expected_adapter_final = K::ZERO;
        let mut expected_bitness_final = K::ZERO;
        for (slot, lane_ref) in group.lanes.iter().enumerate() {
            let lane = shout_gamma_lane_data
                .get(lane_ref.flat_lane_idx)
                .and_then(|x| x.as_ref())
                .ok_or_else(|| PiCcsError::ProtocolError("missing shout gamma lane verify data".into()))?;
            let w = weights[slot];
            let eq_addr = eq_bits_prod(&lane.addr_bits, &lane.pre.r_addr)?;
            expected_value_claim += w * lane.pre.addr_claim_sum;
            expected_value_final += w * lane.has_lookup * lane.val;
            expected_adapter_claim += w * lane.pre.addr_final;
            expected_adapter_final += w * lane.pre.table_eval_at_r_addr * lane.has_lookup * eq_addr;
            for b in lane.addr_bits.iter() {
                expected_bitness_final += w * *b * (*b - K::ONE);
            }
            expected_bitness_final += w * lane.has_lookup * (lane.has_lookup - K::ONE);
        }
        expected_value_final *= chi_cycle_at_r_time;
        expected_adapter_final *= chi_cycle_at_r_time;
        expected_bitness_final *= chi_cycle_at_r_time;

        if value_claim != expected_value_claim {
            return Err(PiCcsError::ProtocolError(
                "shout gamma value claimed sum mismatch".into(),
            ));
        }
        if value_final != expected_value_final {
            return Err(PiCcsError::ProtocolError("shout gamma value terminal mismatch".into()));
        }
        if adapter_claim != expected_adapter_claim {
            return Err(PiCcsError::ProtocolError(
                "shout gamma adapter claimed sum mismatch".into(),
            ));
        }
        if adapter_final != expected_adapter_final {
            return Err(PiCcsError::ProtocolError(
                "shout gamma adapter terminal mismatch".into(),
            ));
        }
        if bitness_final != expected_bitness_final {
            return Err(PiCcsError::ProtocolError(
                "shout gamma bitness terminal mismatch".into(),
            ));
        }
    }

    // Twist instances next.
    let proof_mem_offset = step.lut_insts.len();

    // --------------------------------------------------------------------
    // Twist time checks at addr-pre `r_addr`.
    // --------------------------------------------------------------------
    for (i_mem, inst) in step.mem_insts.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };
        let layout = inst.twist_layout();
        let ell_addr = layout
            .lanes
            .get(0)
            .ok_or_else(|| PiCcsError::InvalidInput("TwistWitnessLayout has no lanes".into()))?
            .ell_addr;

        let twist_inst_cols = cpu_bus
            .twist_cols
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (twist)".into()))?;
        let expected_lanes = inst.lanes.max(1);
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={i_mem}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        struct TwistLaneTimeOpen {
            ra_bits: Vec<K>,
            wa_bits: Vec<K>,
            has_read: K,
            has_write: K,
            wv: K,
            rv: K,
            inc: K,
        }

        let mut lane_opens: Vec<TwistLaneTimeOpen> = Vec::with_capacity(twist_inst_cols.lanes.len());
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.ra_bits.end - twist_cols.ra_bits.start != ell_addr
                || twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut ra_bits_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.ra_bits.clone() {
                ra_bits_open.push(named_opening(&bus_time_open_map, col_id, "route-a/time twist ra_bits")?);
            }
            let mut wa_bits_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits_open.push(named_opening(&bus_time_open_map, col_id, "route-a/time twist wa_bits")?);
            }

            let has_read_open = named_opening(&bus_time_open_map, twist_cols.has_read, "route-a/time twist has_read")?;
            let has_write_open =
                named_opening(&bus_time_open_map, twist_cols.has_write, "route-a/time twist has_write")?;
            let wv_open = named_opening(&bus_time_open_map, twist_cols.wv, "route-a/time twist wv")?;
            let rv_open = named_opening(&bus_time_open_map, twist_cols.rv, "route-a/time twist rv")?;
            let inc_write_open = named_opening(&bus_time_open_map, twist_cols.inc, "route-a/time twist inc")?;

            lane_opens.push(TwistLaneTimeOpen {
                ra_bits: ra_bits_open,
                wa_bits: wa_bits_open,
                has_read: has_read_open,
                has_write: has_write_open,
                wv: wv_open,
                rv: rv_open,
                inc: inc_write_open,
            });
        }
        let pre = twist_pre
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing Twist pre-time data at index {}", i_mem)))?;
        let r_addr = &pre.r_addr;
        if r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist r_addr.len()={}, expected ell_addr={}",
                r_addr.len(),
                ell_addr
            )));
        }

        let twist_claims = claim_plan
            .twist
            .get(i_mem)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing Twist claim schedule at index {}", i_mem)))?;

        // Route A Twist ordering in batched_time:
        // - read_check (time rounds only)
        // - write_check (time rounds only)
        // - bitness for ra_bits then wa_bits then has_read then has_write (time-only)
        let read_check_claim = batched_claimed_sums[twist_claims.read_check];
        let read_check_final = batched_final_values[twist_claims.read_check];
        let write_check_claim = batched_claimed_sums[twist_claims.write_check];
        let write_check_final = batched_final_values[twist_claims.write_check];

        if read_check_claim != pre.read_check_claim_sum {
            return Err(PiCcsError::ProtocolError(
                "twist read_check claimed sum != addr-pre final".into(),
            ));
        }
        if write_check_claim != pre.write_check_claim_sum {
            return Err(PiCcsError::ProtocolError(
                "twist write_check claimed sum != addr-pre final".into(),
            ));
        }

        // Aggregated bitness terminal check (ra_bits, wa_bits, has_read, has_write).
        {
            let mut opens: Vec<K> = Vec::with_capacity(expected_lanes * (2 * ell_addr + 2));
            for lane in lane_opens.iter() {
                opens.extend_from_slice(&lane.ra_bits);
                opens.extend_from_slice(&lane.wa_bits);
                opens.push(lane.has_read);
                opens.push(lane.has_write);
            }
            let weights = bitness_weights(r_cycle, opens.len(), 0x5457_4953_54u64 + i_mem as u64);
            let mut acc = K::ZERO;
            for (w, b) in weights.iter().zip(opens.iter()) {
                acc += *w * *b * (*b - K::ONE);
            }
            let expected = chi_cycle_at_r_time * acc;
            if expected != batched_final_values[twist_claims.bitness] {
                return Err(PiCcsError::ProtocolError(
                    "twist/bitness terminal value mismatch".into(),
                ));
            }
        }

        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

        let init_at_r_addr = eval_init_at_r_addr(&inst.init, inst.k, r_addr)?;
        let claimed_val = init_at_r_addr + val_eval.claimed_inc_sum_lt;

        // Terminal checks for read_check / write_check at (r_time, r_addr).
        let mut expected_read_check_final = K::ZERO;
        let mut expected_write_check_final = K::ZERO;
        for lane in lane_opens.iter() {
            let read_eq_addr = eq_bits_prod(&lane.ra_bits, r_addr)?;
            expected_read_check_final += chi_cycle_at_r_time * lane.has_read * (claimed_val - lane.rv) * read_eq_addr;

            let write_eq_addr = eq_bits_prod(&lane.wa_bits, r_addr)?;
            expected_write_check_final +=
                chi_cycle_at_r_time * lane.has_write * (lane.wv - claimed_val - lane.inc) * write_eq_addr;
        }
        if expected_read_check_final != read_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/read_check terminal value mismatch".into(),
            ));
        }

        if expected_write_check_final != write_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/write_check terminal value mismatch".into(),
            ));
        }
        if let Some(claim_idx) = twist_claims.virtual_write_domain {
            if claim_idx >= batched_final_values.len() {
                return Err(PiCcsError::ProtocolError(
                    "twist/virtual_write_domain claim index out of range".into(),
                ));
            }
            let mut residual = K::ZERO;
            for lane in lane_opens.iter() {
                let wa_bit5 = lane.wa_bits.get(5).copied().unwrap_or(K::ZERO);
                residual += is_virtual_open * lane.has_write * (K::ONE - wa_bit5);
            }
            let expected = chi_cycle_at_r_time * residual;
            if expected != batched_final_values[claim_idx] {
                return Err(PiCcsError::ProtocolError(
                    "twist/virtual_write_domain terminal value mismatch".into(),
                ));
            }
        }
        if let Some(claim_idx) = twist_claims.nonvirtual_arch_domain {
            if claim_idx >= batched_final_values.len() {
                return Err(PiCcsError::ProtocolError(
                    "twist/nonvirtual_arch_domain claim index out of range".into(),
                ));
            }
            let mut residual = K::ZERO;
            for lane in lane_opens.iter() {
                let ra_bit5 = lane.ra_bits.get(5).copied().unwrap_or(K::ZERO);
                let wa_bit5 = lane.wa_bits.get(5).copied().unwrap_or(K::ZERO);
                residual += (K::ONE - is_virtual_open) * lane.has_read * ra_bit5;
                residual += (K::ONE - is_virtual_open) * lane.has_write * wa_bit5;
            }
            let expected = chi_cycle_at_r_time * residual;
            if expected != batched_final_values[claim_idx] {
                return Err(PiCcsError::ProtocolError(
                    "twist/nonvirtual_arch_domain terminal value mismatch".into(),
                ));
            }
        }

        twist_time_openings.push(TwistTimeLaneOpenings {
            lanes: lane_opens
                .into_iter()
                .map(|lane| TwistTimeLaneOpeningsLane {
                    has_read: lane.has_read,
                    wa_bits: lane.wa_bits,
                    has_write: lane.has_write,
                    wv: lane.wv,
                    inc_at_write_addr: lane.inc,
                })
                .collect(),
        });
    }

    // --------------------------------------------------------------------
    // Phase 2: Verify batched Twist val-eval sum-check, deriving shared r_val.
    // --------------------------------------------------------------------
    let mut r_val: Vec<K> = Vec::new();
    let mut val_eval_finals: Vec<K> = Vec::new();
    if !step.mem_insts.is_empty() {
        let plan = crate::memory_sidecar::claim_plan::TwistValEvalClaimPlan::build(step.mem_insts.iter(), has_prev);
        let claim_count = plan.claim_count;

        let mut per_claim_rounds: Vec<Vec<Vec<K>>> = Vec::with_capacity(claim_count);
        let mut per_claim_sums: Vec<K> = Vec::with_capacity(claim_count);
        let mut bind_claims: Vec<(u8, K)> = Vec::with_capacity(claim_count);
        let mut claim_idx = 0usize;

        for (i_mem, _inst) in step.mem_insts.iter().enumerate() {
            let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
                MemOrLutProof::Twist(proof) => proof,
                _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
            };
            let val = twist_proof
                .val_eval
                .as_ref()
                .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

            per_claim_rounds.push(val.rounds_lt.clone());
            per_claim_sums.push(val.claimed_inc_sum_lt);
            bind_claims.push((plan.bind_tags[claim_idx], val.claimed_inc_sum_lt));
            claim_idx += 1;

            per_claim_rounds.push(val.rounds_total.clone());
            per_claim_sums.push(val.claimed_inc_sum_total);
            bind_claims.push((plan.bind_tags[claim_idx], val.claimed_inc_sum_total));
            claim_idx += 1;

            if has_prev {
                let prev_total = val.claimed_prev_inc_sum_total.ok_or_else(|| {
                    PiCcsError::InvalidInput("Twist(Route A): missing claimed_prev_inc_sum_total".into())
                })?;
                let prev_rounds = val
                    .rounds_prev_total
                    .clone()
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing rounds_prev_total".into()))?;
                per_claim_rounds.push(prev_rounds);
                per_claim_sums.push(prev_total);
                bind_claims.push((plan.bind_tags[claim_idx], prev_total));
                claim_idx += 1;
            } else if val.claimed_prev_inc_sum_total.is_some() || val.rounds_prev_total.is_some() {
                return Err(PiCcsError::InvalidInput(
                    "Twist(Route A): rollover fields present but prev_step is None".into(),
                ));
            }
        }

        tr.append_message(
            b"twist/val_eval/batch_start",
            &(step.mem_insts.len() as u64).to_le_bytes(),
        );
        tr.append_message(b"twist/val_eval/step_idx", &(step_idx as u64).to_le_bytes());
        bind_twist_val_eval_claim_sums(tr, &bind_claims);

        let (r_val_out, finals_out, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"twist/val_eval_batch",
            step_idx,
            &per_claim_rounds,
            &per_claim_sums,
            &plan.labels,
            &plan.degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "twist val-eval batched sumcheck invalid".into(),
            ));
        }
        if r_val_out.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val_out.len(),
                r_time.len()
            )));
        }
        if finals_out.len() != claim_count {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval finals.len()={}, expected {}",
                finals_out.len(),
                claim_count
            )));
        }
        r_val = r_val_out;
        val_eval_finals = finals_out;

        tr.append_message(b"twist/val_eval/batch_done", &[]);
    }

    // Verify val-eval terminal identity against CPU ME openings at r_val.
    let lt = if step.mem_insts.is_empty() {
        if !r_val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-eval produced r_val but no mem instances are present".into(),
            ));
        }
        K::ZERO
    } else {
        if r_val.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val.len(),
                r_time.len()
            )));
        }
        lt_eval(&r_val, r_time)
    };

    let (bus_val_open_cur, bus_val_open_prev, has_prev_val_claim) = if step.mem_insts.is_empty() {
        if !mem_proof.val_me_claims.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "proof contains val-lane CPU ME claims with no Twist instances".into(),
            ));
        }
        (BTreeMap::new(), BTreeMap::new(), false)
    } else {
        let expected = 1usize + usize::from(has_prev);
        if mem_proof.val_me_claims.len() != expected {
            return Err(PiCcsError::InvalidInput(format!(
                "shared bus expects {} CPU ME claim(s) at r_val, got {}",
                expected,
                mem_proof.val_me_claims.len()
            )));
        }

        let cpu_me_cur = mem_proof
            .val_me_claims
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("missing CPU ME claim at r_val".into()))?;
        if cpu_me_cur.r.as_slice() != r_val {
            return Err(PiCcsError::ProtocolError(
                "CPU ME(val) r mismatch (expected r_val)".into(),
            ));
        }
        if cpu_me_cur.c != step.mcs_inst.c {
            return Err(PiCcsError::ProtocolError(
                "CPU ME(val) commitment mismatch (current step)".into(),
            ));
        }
        let bus_col_ids = bus_logical_col_ids_for_step_instance(step, cpu_bus, "route-a/val cur")?;
        let named_cur = require_time_openings_for_point(
            step_time_openings,
            r_val.as_slice(),
            bus_col_ids.as_slice(),
            "route-a/val cur",
        )?;
        let named_cur_entry = require_time_opening_entry_for_point(
            step_time_openings,
            r_val.as_slice(),
            bus_col_ids.as_slice(),
            "route-a/val cur",
        )?;
        if named_cur_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "route-a/val cur requires CommittedOpening source (got {:?})",
                named_cur_entry.source
            )));
        }
        let logical_to_local_bus_map = |logical_map: &BTreeMap<usize, K>,
                                        local_bus_col_ids: &[usize],
                                        label: &str|
         -> Result<BTreeMap<usize, K>, PiCcsError> {
            let mut local_map = BTreeMap::new();
            for (mem_local_col, &logical_col_id) in local_bus_col_ids.iter().enumerate() {
                let v = logical_map.get(&logical_col_id).copied().ok_or_else(|| {
                        PiCcsError::ProtocolError(format!(
                            "{label}: missing logical opening value for mem_local_col={mem_local_col} logical_col_id={logical_col_id}"
                        ))
                    })?;
                local_map.insert(mem_local_col, v);
            }
            Ok(local_map)
        };
        let bus_open_cur = logical_to_local_bus_map(&named_cur, bus_col_ids.as_slice(), "route-a/val cur")?;

        let enforce_prev_val_openings = has_prev && step.mcs_inst.m_in == 5;
        let cpu_me_prev = if enforce_prev_val_openings {
            let prev_inst =
                prev_step.ok_or_else(|| PiCcsError::ProtocolError("prev_step missing with has_prev=true".into()))?;
            let cpu_me_prev = mem_proof
                .val_me_claims
                .get(1)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev CPU ME claim at r_val".into()))?;
            if cpu_me_prev.r.as_slice() != r_val {
                return Err(PiCcsError::ProtocolError(
                    "CPU ME(val/prev) r mismatch (expected r_val)".into(),
                ));
            }
            if cpu_me_prev.c != prev_inst.mcs_inst.c {
                return Err(PiCcsError::ProtocolError("CPU ME(val/prev) commitment mismatch".into()));
            }
            Some(cpu_me_prev)
        } else {
            None
        };
        let bus_open_prev = if let Some(cpu_me_prev) = cpu_me_prev {
            // Canonical committed path: previous-step val openings at r_val must come from
            // previous-step committed time openings, never ME tails.
            let prev_step_ref = prev_step.ok_or_else(|| {
                PiCcsError::ProtocolError("route-a/val prev: missing prev_step with has_prev=true".into())
            })?;
            let prev_bus_col_ids = bus_logical_col_ids_for_step_instance(prev_step_ref, cpu_bus, "route-a/val prev")?;
            let named_prev = if let Some(prev_openings) = prev_step_time_openings {
                if let Some(prev_entry) = time_opening_entry_for_point(
                    prev_openings,
                    r_val.as_slice(),
                    prev_bus_col_ids.as_slice(),
                    "route-a/val prev",
                )? {
                    if prev_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
                        return Err(PiCcsError::ProtocolError(format!(
                            "route-a/val prev requires CommittedOpening source (got {:?})",
                            prev_entry.source
                        )));
                    }
                    Some(require_time_openings_for_point(
                        prev_openings,
                        r_val.as_slice(),
                        prev_bus_col_ids.as_slice(),
                        "route-a/val prev",
                    )?)
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(logical_prev) = named_prev {
                logical_to_local_bus_map(&logical_prev, prev_bus_col_ids.as_slice(), "route-a/val prev")?
            } else {
                // Compatibility fallback for chunked-IVC continuity: if previous-step named openings
                // at current r_val are absent, read the already-bound prev CPU ME tail openings.
                let need = core_t
                    .checked_add(cpu_bus.bus_cols)
                    .ok_or_else(|| PiCcsError::ProtocolError("route-a/val prev: core_t + bus_cols overflow".into()))?;
                if cpu_me_prev.ct.len() < need {
                    return Err(PiCcsError::ProtocolError(format!(
                        "route-a/val prev: missing bus tail openings on prev CPU ME claim (ct.len()={}, need >= {})",
                        cpu_me_prev.ct.len(),
                        need
                    )));
                }
                let mut local = BTreeMap::new();
                for mem_local_col in 0..cpu_bus.bus_cols {
                    local.insert(mem_local_col, cpu_me_prev.ct[core_t + mem_local_col]);
                }
                local
            }
        } else {
            BTreeMap::new()
        };

        (bus_open_cur, bus_open_prev, cpu_me_prev.is_some())
    };

    for (i_mem, inst) in step.mem_insts.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };
        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;
        let layout = inst.twist_layout();
        let ell_addr = layout
            .lanes
            .get(0)
            .ok_or_else(|| PiCcsError::InvalidInput("TwistWitnessLayout has no lanes".into()))?
            .ell_addr;

        let twist_inst_cols = cpu_bus
            .twist_cols
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (twist)".into()))?;
        let expected_lanes = inst.lanes.max(1);
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={i_mem}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        let r_addr = twist_pre
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing Twist pre-time data at index {}", i_mem)))?
            .r_addr
            .as_slice();

        let mut inc_at_r_addr_val = K::ZERO;
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut wa_bits_val_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits_val_open.push(named_opening(&bus_val_open_cur, col_id, "route-a/val cur wa_bits")?);
            }
            let has_write_val_open =
                named_opening(&bus_val_open_cur, twist_cols.has_write, "route-a/val cur has_write")?;
            let inc_at_write_addr_val_open = named_opening(&bus_val_open_cur, twist_cols.inc, "route-a/val cur inc")?;

            let eq_wa_val = eq_bits_prod(&wa_bits_val_open, r_addr)?;
            inc_at_r_addr_val += has_write_val_open * inc_at_write_addr_val_open * eq_wa_val;
        }

        let expected_lt_final = inc_at_r_addr_val * lt;
        let claims_per_mem = if has_prev { 3 } else { 2 };
        let base = claims_per_mem * i_mem;
        if expected_lt_final != val_eval_finals[base] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_lt terminal value mismatch".into(),
            ));
        }
        let expected_total_final = inc_at_r_addr_val;
        if expected_total_final != val_eval_finals[base + 1] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_total terminal value mismatch".into(),
            ));
        }

        if has_prev_val_claim {
            let prev =
                prev_step.ok_or_else(|| PiCcsError::ProtocolError("prev_step missing with has_prev=true".into()))?;
            let prev_inst = prev
                .mem_insts
                .get(i_mem)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev mem instance".into()))?;

            // Terminal check for prev-total: uses previous-step openings at current r_val.
            let mut inc_at_r_addr_prev = K::ZERO;
            for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
                if twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                    )));
                }

                let mut wa_bits_prev_open = Vec::with_capacity(ell_addr);
                for col_id in twist_cols.wa_bits.clone() {
                    wa_bits_prev_open.push(named_opening(&bus_val_open_prev, col_id, "route-a/val prev wa_bits")?);
                }
                let has_write_prev_open =
                    named_opening(&bus_val_open_prev, twist_cols.has_write, "route-a/val prev has_write")?;
                let inc_prev_open = named_opening(&bus_val_open_prev, twist_cols.inc, "route-a/val prev inc")?;

                let eq_wa_prev = eq_bits_prod(&wa_bits_prev_open, r_addr)?;
                inc_at_r_addr_prev += has_write_prev_open * inc_prev_open * eq_wa_prev;
            }
            if inc_at_r_addr_prev != val_eval_finals[base + 2] {
                return Err(PiCcsError::ProtocolError(
                    "twist/rollover_prev_total terminal value mismatch".into(),
                ));
            }

            // Enforce rollover equation: Init_i(r_addr) == Init_{i-1}(r_addr) + PrevTotal(i).
            let claimed_prev_total = val_eval
                .claimed_prev_inc_sum_total
                .ok_or_else(|| PiCcsError::ProtocolError("twist rollover missing claimed_prev_inc_sum_total".into()))?;
            let init_prev_at_r_addr = eval_init_at_r_addr(&prev_inst.init, prev_inst.k, r_addr)?;
            let init_cur_at_r_addr = eval_init_at_r_addr(&inst.init, inst.k, r_addr)?;
            if init_cur_at_r_addr != init_prev_at_r_addr + claimed_prev_total {
                return Err(PiCcsError::ProtocolError("twist rollover init check failed".into()));
            }
        }
    }

    verify_route_a_wb_wp_terminals(
        step,
        r_time,
        r_cycle,
        batched_final_values,
        &claim_plan,
        mem_proof,
        step_time_openings,
    )?;
    verify_route_a_decode_terminals(
        cpu_bus,
        step,
        r_time,
        r_cycle,
        batched_final_values,
        &claim_plan,
        mem_proof,
        step_time_openings,
    )?;
    verify_route_a_width_terminals(
        cpu_bus,
        step,
        r_time,
        r_cycle,
        batched_final_values,
        &claim_plan,
        mem_proof,
        step_time_openings,
        rv64_fullword_width_stage_from_proof,
    )?;
    verify_route_a_control_terminals(
        cpu_bus,
        step,
        r_time,
        r_cycle,
        batched_final_values,
        &claim_plan,
        mem_proof,
        step_time_openings,
    )?;
    verify_route_a_poseidon_cycle_terminals(
        core_t,
        cpu_bus,
        step,
        r_time,
        r_cycle,
        batched_final_values,
        &claim_plan,
        mem_proof,
        step_time_openings,
        poseidon_link_chals,
        poseidon_cont_chals,
    )?;
    Ok(RouteAMemoryVerifyOutput {
        claim_idx_end: claim_plan.claim_idx_end,
        twist_time_openings,
    })
}
