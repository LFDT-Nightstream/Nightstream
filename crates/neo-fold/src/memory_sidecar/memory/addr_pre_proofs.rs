use super::*;

pub(crate) fn prove_shout_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    cpu_bus: &BusLayout,
    ell_n: usize,
    r_cycle: &[K],
    step_idx: usize,
) -> Result<ShoutAddrPreBatchProverData, PiCcsError> {
    if step.lut_instances.is_empty() {
        return Ok(ShoutAddrPreBatchProverData {
            addr_pre: ShoutAddrPreProof::default(),
            decoded: Vec::new(),
        });
    }

    let pow2_cycle = 1usize << ell_n;
    let n_lut = step.lut_instances.len();
    let total_lanes: usize = step
        .lut_instances
        .iter()
        .map(|(inst, _)| inst.lanes.max(1))
        .sum();

    let mut decoded_cols: Vec<ShoutDecodedColsSparse> = Vec::with_capacity(n_lut);
    let mut claimed_sums: Vec<K> = vec![K::ZERO; total_lanes];

    struct AddrPreGroupBuilder {
        active_lanes: Vec<u32>,
        active_claimed_sums: Vec<K>,
        addr_oracles: Vec<Box<dyn RoundOracle + Send>>,
    }

    // Group Shout addr-pre claims by `ell_addr` so we can run one batched sumcheck per group.
    let mut groups: std::collections::BTreeMap<u32, AddrPreGroupBuilder> = std::collections::BTreeMap::new();

    let mut flat_lane_idx: usize = 0;
    let bus = cpu_bus;
    let use_time_mem_cols = step.time_columns.t == bus.chunk_size && step.time_columns.mem_cols.len() == bus.bus_cols;
    let expected_m = step
        .mcs
        .0
        .m_in
        .checked_add(step.mcs.1.w.len())
        .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus witness width overflow".into()))?;
    let cpu_z_k = if use_time_mem_cols {
        Vec::new()
    } else {
        crate::memory_sidecar::cpu_bus::decode_cpu_z_to_k(params, &step.mcs.1.Z, expected_m)?
    };
    if bus.shout_cols.len() != step.lut_instances.len() || bus.twist_cols.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus layout mismatch for step (instance counts)".into(),
        ));
    }
    let mut addr_range_counts = std::collections::HashMap::<(usize, usize), usize>::new();
    for inst_cols in bus.shout_cols.iter() {
        for lane_cols in inst_cols.lanes.iter() {
            let key = (lane_cols.addr_bits.start, lane_cols.addr_bits.end);
            *addr_range_counts.entry(key).or_insert(0) += 1;
        }
    }
    // Shared-bus trace mode can have many lookup families reusing the same bus columns
    // (e.g. decode/width selector+addr groups and opcode addr groups). Cache sparse
    // decodes by (col_id, steps) to avoid rebuilding identical SparseIdxVec values.
    let mut full_col_sparse_cache: std::collections::HashMap<(usize, usize), SparseIdxVec<K>> =
        std::collections::HashMap::new();
    let mut has_lookup_cache: std::collections::HashMap<(usize, usize), (SparseIdxVec<K>, Vec<usize>, bool)> =
        std::collections::HashMap::new();

    let mut decode_full_col = |col_id: usize, steps: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        if let Some(cached) = full_col_sparse_cache.get(&(col_id, steps)) {
            return Ok(cached.clone());
        }
        let decoded = if use_time_mem_cols {
            crate::memory_sidecar::cpu_bus::build_time_sparse_from_mem_cols(
                &step.time_columns.mem_cols,
                bus,
                col_id,
                steps,
                pow2_cycle,
            )?
        } else {
            crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(&cpu_z_k, bus, col_id, steps, pow2_cycle)?
        };
        full_col_sparse_cache.insert((col_id, steps), decoded.clone());
        Ok(decoded)
    };

    for (idx, (lut_inst, _lut_wit)) in step.lut_instances.iter().enumerate() {
        neo_memory::addr::validate_shout_bit_addressing(lut_inst)?;
        if lut_inst.steps > pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout(Route A): steps={} exceeds 2^ell_cycle={pow2_cycle}",
                lut_inst.steps
            )));
        }

        let z = &cpu_z_k;
        let inst_ell_addr = lut_inst.d * lut_inst.ell;
        let is_packed_spec = matches!(
            lut_inst.table_spec,
            Some(LutTableSpec::RiscvOpcodePacked { .. } | LutTableSpec::RiscvOpcodeEventTablePacked { .. })
        );
        let inst_ell_addr_u32 = u32::try_from(inst_ell_addr)
            .map_err(|_| PiCcsError::InvalidInput("Shout(Route A): ell_addr overflows u32".into()))?;
        groups
            .entry(inst_ell_addr_u32)
            .or_insert_with(|| AddrPreGroupBuilder {
                active_lanes: Vec::new(),
                active_claimed_sums: Vec::new(),
                addr_oracles: Vec::new(),
            });
        let inst_cols = bus.shout_cols.get(idx).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch: missing shout_cols for lut_idx={idx}"
            ))
        })?;
        let expected_lanes = lut_inst.lanes.max(1);
        if inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at lut_idx={idx}: shout lanes={} but instance expects {}",
                inst_cols.lanes.len(),
                expected_lanes
            )));
        }

        let mut lanes: Vec<ShoutLaneSparseCols> = Vec::with_capacity(expected_lanes);

        for (lane_idx, shout_cols) in inst_cols.lanes.iter().enumerate() {
            if shout_cols.addr_bits.end - shout_cols.addr_bits.start != inst_ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                        "shared_cpu_bus layout mismatch at lut_idx={idx}, lane_idx={lane_idx}: expected ell_addr={inst_ell_addr}"
                    )));
            }
            let addr_key = (shout_cols.addr_bits.start, shout_cols.addr_bits.end);
            let shared_addr_group = addr_range_counts.get(&addr_key).copied().unwrap_or(0) > 1;

            let cache_key = (shout_cols.has_lookup, lut_inst.steps);
            if !has_lookup_cache.contains_key(&cache_key) {
                let has_lookup = decode_full_col(shout_cols.has_lookup, lut_inst.steps)?;
                let has_any_lookup = has_lookup
                    .entries()
                    .iter()
                    .any(|&(_t, gate)| gate != K::ZERO);
                let active_js: Vec<usize> = if has_any_lookup {
                    let m_in = bus.m_in;
                    let mut out: Vec<usize> = Vec::with_capacity(has_lookup.entries().len());
                    for &(t, gate) in has_lookup.entries() {
                        if gate == K::ZERO {
                            continue;
                        }
                        let j = t.checked_sub(m_in).ok_or_else(|| {
                            PiCcsError::InvalidInput(format!(
                                "Shout(Route A): has_lookup time index underflow: t={t} < m_in={m_in}"
                            ))
                        })?;
                        if j >= lut_inst.steps {
                            return Err(PiCcsError::ProtocolError(format!(
                                "Shout(Route A): has_lookup time index out of range: j={j} >= steps={}",
                                lut_inst.steps
                            )));
                        }
                        out.push(j);
                    }
                    out
                } else {
                    Vec::new()
                };
                has_lookup_cache.insert(cache_key, (has_lookup, active_js, has_any_lookup));
            }
            let (has_lookup, active_js, has_any_lookup) = has_lookup_cache
                .get(&cache_key)
                .map(|(cached_has, cached_js, cached_any)| (cached_has.clone(), cached_js.as_slice(), *cached_any))
                .ok_or_else(|| PiCcsError::ProtocolError("Shout(Route A): missing has_lookup cache entry".into()))?;

            let addr_bits: Vec<SparseIdxVec<K>> = if shared_addr_group {
                let mut out = Vec::with_capacity(inst_ell_addr);
                for col_id in shout_cols.addr_bits.clone() {
                    out.push(decode_full_col(col_id, lut_inst.steps)?);
                }
                out
            } else if has_any_lookup {
                let mut out = Vec::with_capacity(inst_ell_addr);
                for col_id in shout_cols.addr_bits.clone() {
                    out.push(if use_time_mem_cols {
                        crate::memory_sidecar::cpu_bus::build_time_sparse_from_mem_cols_at_js(
                            &step.time_columns.mem_cols,
                            bus,
                            col_id,
                            active_js,
                            pow2_cycle,
                        )?
                    } else {
                        crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col_at_js(
                            z, bus, col_id, active_js, pow2_cycle,
                        )?
                    });
                }
                out
            } else {
                vec![SparseIdxVec::new(pow2_cycle); inst_ell_addr]
            };

            let val = if has_any_lookup {
                if use_time_mem_cols {
                    crate::memory_sidecar::cpu_bus::build_time_sparse_from_mem_cols_at_js(
                        &step.time_columns.mem_cols,
                        bus,
                        shout_cols.primary_val(),
                        active_js,
                        pow2_cycle,
                    )?
                } else {
                    crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col_at_js(
                        z,
                        bus,
                        shout_cols.primary_val(),
                        active_js,
                        pow2_cycle,
                    )?
                }
            } else {
                SparseIdxVec::new(pow2_cycle)
            };

            if has_any_lookup && !is_packed_spec {
                let (addr_oracle, lane_sum): (Box<dyn RoundOracle + Send>, K) = match &lut_inst.table_spec {
                    None => {
                        let table_k: Vec<K> = lut_inst.table.iter().map(|&v| v.into()).collect();
                        let (o, sum) =
                            AddressLookupOracle::new(&addr_bits, &has_lookup, &table_k, r_cycle, inst_ell_addr);
                        (Box::new(o), sum)
                    }
                    Some(LutTableSpec::RiscvOpcode { opcode, xlen }) => {
                        let (o, sum) = RiscvAddressLookupOracleSparse::new_sparse_time(
                            *opcode,
                            *xlen,
                            &addr_bits,
                            &has_lookup,
                            r_cycle,
                        )?;
                        (Box::new(o), sum)
                    }
                    Some(LutTableSpec::RiscvOpcodePacked { .. })
                    | Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. }) => {
                        return Err(PiCcsError::ProtocolError(
                            "packed shout lane unexpectedly reached implicit addr-pre oracle path".into(),
                        ));
                    }
                    Some(LutTableSpec::IdentityU32) => {
                        let (o, sum) = IdentityAddressLookupOracleSparse::new_sparse_time(
                            inst_ell_addr,
                            &addr_bits,
                            &has_lookup,
                            r_cycle,
                        )?;
                        (Box::new(o), sum)
                    }
                };

                claimed_sums[flat_lane_idx] = lane_sum;
                let lane_idx_u32 = u32::try_from(flat_lane_idx)
                    .map_err(|_| PiCcsError::InvalidInput("Shout(Route A): lane index overflow".into()))?;
                let group = groups
                    .get_mut(&inst_ell_addr_u32)
                    .ok_or_else(|| PiCcsError::ProtocolError("Shout(Route A): missing ell_addr group".into()))?;
                group.active_lanes.push(lane_idx_u32);
                group.active_claimed_sums.push(lane_sum);
                group.addr_oracles.push(addr_oracle);
            }

            lanes.push(ShoutLaneSparseCols {
                addr_bits,
                has_lookup,
                val,
            });
            flat_lane_idx += 1;
        }

        let decoded = ShoutDecodedColsSparse { lanes };

        decoded_cols.push(decoded);
    }
    if flat_lane_idx != total_lanes {
        return Err(PiCcsError::ProtocolError(format!(
            "Shout(Route A): flat lane indexing drift (got {flat_lane_idx}, expected {total_lanes})"
        )));
    }

    let labels_all: Vec<&'static [u8]> = vec![b"shout/addr_pre".as_slice(); total_lanes];
    tr.append_message(b"shout/addr_pre_time/step_idx", &(step_idx as u64).to_le_bytes());
    bind_batched_claim_sums(tr, b"shout/addr_pre_time/claimed_sums", &claimed_sums, &labels_all);

    let mut group_proofs: Vec<ShoutAddrPreGroupProof<K>> = Vec::with_capacity(groups.len());
    for (group_idx, (ell_addr, mut group)) in groups.into_iter().enumerate() {
        tr.append_message(b"shout/addr_pre_time/group_idx", &(group_idx as u64).to_le_bytes());
        tr.append_message(b"shout/addr_pre_time/group_ell_addr", &(ell_addr as u64).to_le_bytes());

        let (r_addr, round_polys) = if group.active_lanes.is_empty() {
            // No active lanes in this `ell_addr` group; sample an arbitrary `r_addr` without running sumcheck.
            tr.append_message(b"shout/addr_pre_time/no_sumcheck", &(step_idx as u64).to_le_bytes());
            tr.append_message(
                b"shout/addr_pre_time/no_sumcheck/ell_addr",
                &(ell_addr as u64).to_le_bytes(),
            );
            (
                ts::sample_ext_point(
                    tr,
                    b"shout/addr_pre_time/no_sumcheck/r_addr",
                    b"shout/addr_pre_time/no_sumcheck/r_addr/0",
                    b"shout/addr_pre_time/no_sumcheck/r_addr/1",
                    ell_addr as usize,
                ),
                Vec::new(),
            )
        } else {
            let labels_active: Vec<&'static [u8]> = vec![b"shout/addr_pre".as_slice(); group.addr_oracles.len()];
            let mut claims: Vec<BatchedClaim<'_>> = group
                .addr_oracles
                .iter_mut()
                .zip(group.active_claimed_sums.iter())
                .zip(labels_active.iter())
                .map(|((oracle, sum), label)| BatchedClaim {
                    oracle: oracle.as_mut(),
                    claimed_sum: *sum,
                    label: *label,
                })
                .collect();

            let (r_addr, per_claim_results) =
                run_batched_sumcheck_prover_ds(tr, b"shout/addr_pre_time", step_idx, claims.as_mut_slice())?;
            let round_polys = per_claim_results
                .into_iter()
                .map(|r| r.round_polys)
                .collect::<Vec<_>>();
            (r_addr, round_polys)
        };

        group_proofs.push(ShoutAddrPreGroupProof {
            ell_addr,
            active_lanes: group.active_lanes,
            round_polys,
            r_addr,
        });
    }

    Ok(ShoutAddrPreBatchProverData {
        addr_pre: ShoutAddrPreProof {
            claimed_sums,
            groups: group_proofs,
        },
        decoded: decoded_cols,
    })
}

pub fn verify_shout_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    step: &StepInstanceBundle<Cmt, F, K>,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    step_idx: usize,
) -> Result<Vec<ShoutAddrPreVerifyData>, PiCcsError> {
    let proof = &mem_proof.shout_addr_pre;

    if step.lut_insts.is_empty() {
        if !proof.claimed_sums.is_empty() || !proof.groups.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "shout_addr_pre must be empty when there are no Shout instances".into(),
            ));
        }
        return Ok(Vec::new());
    }

    let total_lanes: usize = step.lut_insts.iter().map(|inst| inst.lanes.max(1)).sum();
    if proof.claimed_sums.len() != total_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout_addr_pre claimed_sums.len()={}, expected total_lanes={}",
            proof.claimed_sums.len(),
            total_lanes
        )));
    }

    // Flatten lane->ell_addr mapping in canonical order so we can validate group membership and
    // attach the correct `r_addr` per lane.
    let mut lane_ell_addr: Vec<u32> = Vec::with_capacity(total_lanes);
    let mut required_ell_addrs: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for lut_inst in step.lut_insts.iter() {
        neo_memory::addr::validate_shout_bit_addressing(lut_inst)?;
        let inst_ell_addr = lut_inst.d * lut_inst.ell;
        let inst_ell_addr_u32 = u32::try_from(inst_ell_addr)
            .map_err(|_| PiCcsError::InvalidInput("Shout: ell_addr overflows u32".into()))?;
        required_ell_addrs.insert(inst_ell_addr_u32);
        for _lane_idx in 0..lut_inst.lanes.max(1) {
            lane_ell_addr.push(inst_ell_addr_u32);
        }
    }
    if lane_ell_addr.len() != total_lanes {
        return Err(PiCcsError::ProtocolError(
            "shout addr-pre lane indexing drift (lane_ell_addr)".into(),
        ));
    }

    // Groups must match the step's required `ell_addr` set and be sorted/unique.
    if proof.groups.len() != required_ell_addrs.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "shout_addr_pre groups.len()={}, expected {} (distinct ell_addr values in step)",
            proof.groups.len(),
            required_ell_addrs.len()
        )));
    }
    let required_list: Vec<u32> = required_ell_addrs.into_iter().collect();
    for (idx, group) in proof.groups.iter().enumerate() {
        let expected_ell_addr = required_list[idx];
        if group.ell_addr != expected_ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "shout_addr_pre groups not sorted or mismatched: groups[{idx}].ell_addr={} but expected {expected_ell_addr}",
                group.ell_addr
            )));
        }
        if group.r_addr.len() != group.ell_addr as usize {
            return Err(PiCcsError::InvalidInput(format!(
                "shout_addr_pre group ell_addr={} has r_addr.len()={}, expected {}",
                group.ell_addr,
                group.r_addr.len(),
                group.ell_addr
            )));
        }
        if group.round_polys.len() != group.active_lanes.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "shout_addr_pre group ell_addr={} round_polys.len()={}, expected active_lanes.len()={}",
                group.ell_addr,
                group.round_polys.len(),
                group.active_lanes.len()
            )));
        }

        for (pos, &lane_idx) in group.active_lanes.iter().enumerate() {
            let lane_idx_usize = lane_idx as usize;
            if lane_idx_usize >= total_lanes {
                return Err(PiCcsError::InvalidInput(
                    "shout_addr_pre active_lanes has index out of range".into(),
                ));
            }
            if lane_ell_addr[lane_idx_usize] != group.ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shout_addr_pre active_lanes contains lane_idx={} with ell_addr={}, but group ell_addr={}",
                    lane_idx, lane_ell_addr[lane_idx_usize], group.ell_addr
                )));
            }
            if pos > 0 && group.active_lanes[pos - 1] >= lane_idx {
                return Err(PiCcsError::InvalidInput(
                    "shout_addr_pre active_lanes must be strictly increasing".into(),
                ));
            }
        }
        for (pos, rounds) in group.round_polys.iter().enumerate() {
            if rounds.len() != group.ell_addr as usize {
                return Err(PiCcsError::InvalidInput(format!(
                    "shout_addr_pre group ell_addr={} round_polys[{pos}].len()={}, expected {}",
                    group.ell_addr,
                    rounds.len(),
                    group.ell_addr
                )));
            }
        }
    }

    // Bind all claimed sums (all lanes) once.
    let labels_all: Vec<&'static [u8]> = vec![b"shout/addr_pre".as_slice(); total_lanes];
    tr.append_message(b"shout/addr_pre_time/step_idx", &(step_idx as u64).to_le_bytes());
    bind_batched_claim_sums(
        tr,
        b"shout/addr_pre_time/claimed_sums",
        &proof.claimed_sums,
        &labels_all,
    );

    // Verify each `ell_addr` group independently, collecting per-lane addr-pre finals and
    // recording the shared `r_addr` for that group.
    let mut lane_is_active = vec![false; total_lanes];
    let mut lane_addr_final = vec![K::ZERO; total_lanes];
    let mut r_addr_by_ell: std::collections::BTreeMap<u32, Vec<K>> = std::collections::BTreeMap::new();
    let mut seen_active: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for (group_idx, group) in proof.groups.iter().enumerate() {
        tr.append_message(b"shout/addr_pre_time/group_idx", &(group_idx as u64).to_le_bytes());
        tr.append_message(
            b"shout/addr_pre_time/group_ell_addr",
            &(group.ell_addr as u64).to_le_bytes(),
        );

        if group.active_lanes.is_empty() {
            // No active lanes in this group: match prover's deterministic fallback sampling.
            tr.append_message(b"shout/addr_pre_time/no_sumcheck", &(step_idx as u64).to_le_bytes());
            tr.append_message(
                b"shout/addr_pre_time/no_sumcheck/ell_addr",
                &(group.ell_addr as u64).to_le_bytes(),
            );
            let r_addr = ts::sample_ext_point(
                tr,
                b"shout/addr_pre_time/no_sumcheck/r_addr",
                b"shout/addr_pre_time/no_sumcheck/r_addr/0",
                b"shout/addr_pre_time/no_sumcheck/r_addr/1",
                group.ell_addr as usize,
            );
            if r_addr != group.r_addr {
                return Err(PiCcsError::ProtocolError(
                    "shout_addr_pre r_addr mismatch: transcript-derived vs proof".into(),
                ));
            }
            r_addr_by_ell.insert(group.ell_addr, r_addr);
            continue;
        }

        let active_count = group.active_lanes.len();
        let mut active_claimed_sums: Vec<K> = Vec::with_capacity(active_count);
        for &lane_idx in group.active_lanes.iter() {
            if !seen_active.insert(lane_idx) {
                return Err(PiCcsError::InvalidInput(
                    "shout_addr_pre active_lanes contains duplicates across groups".into(),
                ));
            }
            active_claimed_sums.push(
                *proof
                    .claimed_sums
                    .get(lane_idx as usize)
                    .ok_or_else(|| PiCcsError::ProtocolError("shout addr-pre active lane idx drift".into()))?,
            );
        }
        let labels_active: Vec<&'static [u8]> = vec![b"shout/addr_pre".as_slice(); active_count];
        let degree_bounds = vec![2usize; active_count];
        let (r_addr, finals, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"shout/addr_pre_time",
            step_idx,
            &group.round_polys,
            &active_claimed_sums,
            &labels_active,
            &degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "shout addr-pre batched sumcheck invalid".into(),
            ));
        }
        if r_addr != group.r_addr {
            return Err(PiCcsError::ProtocolError(
                "shout_addr_pre r_addr mismatch: transcript-derived vs proof".into(),
            ));
        }
        if finals.len() != active_count {
            return Err(PiCcsError::ProtocolError(format!(
                "shout addr-pre finals.len()={}, expected active_count={active_count}",
                finals.len()
            )));
        }

        for (pos, &lane_idx) in group.active_lanes.iter().enumerate() {
            let lane_idx_usize = lane_idx as usize;
            lane_is_active[lane_idx_usize] = true;
            lane_addr_final[lane_idx_usize] = finals[pos];
        }
        r_addr_by_ell.insert(group.ell_addr, r_addr);
    }

    // Build per-lane verify data in canonical order.
    let mut out = Vec::with_capacity(total_lanes);
    for (lut_inst, inst_ell_addr) in step.lut_insts.iter().map(|inst| (inst, inst.d * inst.ell)) {
        let expected_lanes = lut_inst.lanes.max(1);
        let inst_ell_addr_u32 = u32::try_from(inst_ell_addr)
            .map_err(|_| PiCcsError::InvalidInput("Shout: ell_addr overflows u32".into()))?;
        let r_addr = r_addr_by_ell
            .get(&inst_ell_addr_u32)
            .ok_or_else(|| PiCcsError::ProtocolError("missing shout addr-pre group r_addr".into()))?;

        for _lane_idx in 0..expected_lanes {
            let flat_lane_idx = out.len();
            let addr_claim_sum = *proof
                .claimed_sums
                .get(flat_lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout addr-pre lane index drift".into()))?;
            let is_active = *lane_is_active
                .get(flat_lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout addr-pre lane idx drift".into()))?;
            let addr_final = *lane_addr_final
                .get(flat_lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout addr-pre lane idx drift".into()))?;

            let table_eval_at_r_addr = if is_active {
                match &lut_inst.table_spec {
                    None => {
                        let pow2 = 1usize
                            .checked_shl(r_addr.len() as u32)
                            .ok_or_else(|| PiCcsError::InvalidInput("Shout: 2^ell_addr overflow".into()))?;
                        let mut acc = K::ZERO;
                        for (i, &v) in lut_inst.table.iter().enumerate().take(pow2) {
                            let w = neo_memory::mle::chi_at_index(r_addr, i);
                            acc += K::from(v) * w;
                        }
                        acc
                    }
                    Some(spec) => spec.eval_table_mle(r_addr)?,
                }
            } else {
                K::ZERO
            };

            out.push(ShoutAddrPreVerifyData {
                is_active,
                addr_claim_sum,
                addr_final: if is_active { addr_final } else { K::ZERO },
                r_addr: r_addr.clone(),
                table_eval_at_r_addr,
            });
        }
    }
    if out.len() != total_lanes {
        return Err(PiCcsError::ProtocolError("shout addr-pre lane count mismatch".into()));
    }

    Ok(out)
}

pub fn verify_twist_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    step: &StepInstanceBundle<Cmt, F, K>,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
) -> Result<Vec<TwistAddrPreVerifyData>, PiCcsError> {
    let mut out = Vec::with_capacity(step.mem_insts.len());
    let proof_offset = step.lut_insts.len();

    for (idx, mem_inst) in step.mem_insts.iter().enumerate() {
        let proof = match mem_proof.proofs.get(proof_offset + idx) {
            Some(MemOrLutProof::Twist(p)) => p,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };

        if proof.addr_pre.claimed_sums.len() != 2 {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_pre claimed_sums.len()={}, expected 2",
                proof.addr_pre.claimed_sums.len()
            )));
        }
        if proof.addr_pre.round_polys.len() != 2 {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_pre round_polys.len()={}, expected 2",
                proof.addr_pre.round_polys.len()
            )));
        }
        if proof.addr_pre.claimed_sums[0] != K::ZERO || proof.addr_pre.claimed_sums[1] != K::ZERO {
            return Err(PiCcsError::ProtocolError(
                "twist addr_pre claimed_sums mismatch (expected both 0)".into(),
            ));
        }

        let labels: [&[u8]; 2] = [b"twist/read_addr_pre".as_slice(), b"twist/write_addr_pre".as_slice()];
        let degree_bounds = vec![2usize, 2usize];
        tr.append_message(b"twist/addr_pre_time/claim_idx", &(idx as u64).to_le_bytes());
        bind_batched_claim_sums(
            tr,
            b"twist/addr_pre_time/claimed_sums",
            &proof.addr_pre.claimed_sums,
            &labels,
        );

        let (r_addr, finals, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"twist/addr_pre_time",
            idx,
            &proof.addr_pre.round_polys,
            &proof.addr_pre.claimed_sums,
            &labels,
            &degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "twist addr-pre batched sumcheck invalid".into(),
            ));
        }
        if r_addr != proof.addr_pre.r_addr {
            return Err(PiCcsError::ProtocolError(
                "twist addr_pre r_addr mismatch: transcript-derived vs proof".into(),
            ));
        }

        let ell_addr = mem_inst.d * mem_inst.ell;
        if r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_pre r_addr.len()={}, expected ell_addr={}",
                r_addr.len(),
                ell_addr
            )));
        }
        if finals.len() != 2 {
            return Err(PiCcsError::ProtocolError(format!(
                "twist addr-pre finals.len()={}, expected 2",
                finals.len()
            )));
        }

        out.push(TwistAddrPreVerifyData {
            r_addr,
            read_check_claim_sum: finals[0],
            write_check_claim_sum: finals[1],
        });
    }

    Ok(out)
}
