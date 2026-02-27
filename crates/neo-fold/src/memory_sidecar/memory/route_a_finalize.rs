use super::*;

pub(crate) fn finalize_route_a_memory_prover(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    cpu_bus: &BusLayout,
    s: &CcsStructure<F>,
    step: &StepWitnessBundle<Cmt, F, K>,
    prev_step: Option<&StepWitnessBundle<Cmt, F, K>>,
    prev_twist_decoded: Option<&[TwistDecodedColsSparse]>,
    oracles: &mut RouteAMemoryOracles,
    shout_addr_pre: &ShoutAddrPreProof<K>,
    twist_pre: &[TwistAddrPreProverData],
    r_time: &[K],
    m_in: usize,
    step_idx: usize,
) -> Result<MemSidecarProof<Cmt, F, K>, PiCcsError> {
    let has_prev = prev_step.is_some();
    if has_prev != prev_twist_decoded.is_some() {
        return Err(PiCcsError::InvalidInput(format!(
            "Twist rollover decoded cache mismatch: prev_step.is_some()={} but prev_twist_decoded.is_some()={}",
            has_prev,
            prev_twist_decoded.is_some()
        )));
    }
    let total_lanes: usize = step
        .lut_instances
        .iter()
        .map(|(inst, _)| inst.lanes.max(1))
        .sum();
    if shout_addr_pre.claimed_sums.len() != total_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout addr-pre proof count mismatch (expected claimed_sums.len()=total_lanes={}, got {})",
            total_lanes,
            shout_addr_pre.claimed_sums.len(),
        )));
    }
    {
        let mut lane_ell_addr: Vec<u32> = Vec::with_capacity(total_lanes);
        let mut required_ell_addrs: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for (lut_inst, _lut_wit) in step.lut_instances.iter().map(|(inst, wit)| (inst, wit)) {
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

        if shout_addr_pre.groups.len() != required_ell_addrs.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "shout addr-pre group count mismatch (expected {}, got {})",
                required_ell_addrs.len(),
                shout_addr_pre.groups.len()
            )));
        }
        let required_list: Vec<u32> = required_ell_addrs.into_iter().collect();
        let mut seen_active: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for (idx, group) in shout_addr_pre.groups.iter().enumerate() {
            let expected_ell_addr = required_list[idx];
            if group.ell_addr != expected_ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shout addr-pre groups not sorted or mismatched: groups[{idx}].ell_addr={} but expected {expected_ell_addr}",
                    group.ell_addr
                )));
            }
            if group.r_addr.len() != group.ell_addr as usize {
                return Err(PiCcsError::InvalidInput(format!(
                    "shout addr-pre group ell_addr={} has r_addr.len()={}, expected {}",
                    group.ell_addr,
                    group.r_addr.len(),
                    group.ell_addr
                )));
            }
            if group.round_polys.len() != group.active_lanes.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shout addr-pre group ell_addr={} round_polys.len()={}, expected active_lanes.len()={}",
                    group.ell_addr,
                    group.round_polys.len(),
                    group.active_lanes.len()
                )));
            }
            for (pos, &lane_idx) in group.active_lanes.iter().enumerate() {
                let lane_idx_usize = lane_idx as usize;
                if lane_idx_usize >= total_lanes {
                    return Err(PiCcsError::InvalidInput(
                        "shout addr-pre active_lanes has index out of range".into(),
                    ));
                }
                if lane_ell_addr[lane_idx_usize] != group.ell_addr {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shout addr-pre active_lanes contains lane_idx={} with ell_addr={}, but group ell_addr={}",
                        lane_idx, lane_ell_addr[lane_idx_usize], group.ell_addr
                    )));
                }
                if pos > 0 && group.active_lanes[pos - 1] >= lane_idx {
                    return Err(PiCcsError::InvalidInput(
                        "shout addr-pre active_lanes must be strictly increasing".into(),
                    ));
                }
                if !seen_active.insert(lane_idx) {
                    return Err(PiCcsError::InvalidInput(
                        "shout addr-pre active_lanes contains duplicates across groups".into(),
                    ));
                }
            }
            for (pos, rounds) in group.round_polys.iter().enumerate() {
                if rounds.len() != group.ell_addr as usize {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shout addr-pre group ell_addr={} round_polys[{pos}].len()={}, expected {}",
                        group.ell_addr,
                        rounds.len(),
                        group.ell_addr
                    )));
                }
            }
        }
    }
    if twist_pre.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time count mismatch (expected {}, got {})",
            step.mem_instances.len(),
            twist_pre.len()
        )));
    }
    if oracles.twist.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist oracle count mismatch (expected {}, got {})",
            step.mem_instances.len(),
            oracles.twist.len()
        )));
    }

    for (idx, (lut_inst, lut_wit)) in step.lut_instances.iter().enumerate() {
        if !lut_inst.comms.is_empty() || !lut_wit.mats.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Shout instances (comms/mats must be empty, lut_idx={idx})"
            )));
        }
    }
    for (idx, (mem_inst, mem_wit)) in step.mem_instances.iter().enumerate() {
        if !mem_inst.comms.is_empty() || !mem_wit.mats.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Twist instances (comms/mats must be empty, mem_idx={idx})"
            )));
        }
    }
    if let Some(prev) = prev_step {
        if prev.mem_instances.len() != step.mem_instances.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist rollover requires stable mem instance count: prev has {}, current has {}",
                prev.mem_instances.len(),
                step.mem_instances.len()
            )));
        }
        for (idx, (mem_inst, mem_wit)) in prev.mem_instances.iter().enumerate() {
            if !mem_inst.comms.is_empty() || !mem_wit.mats.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared CPU bus requires metadata-only Twist instances (comms/mats must be empty, prev mem_idx={idx})"
                )));
            }
        }
    }

    let mut val_me_claims: Vec<CeClaim<Cmt, F, K>> = Vec::new();
    let mut wb_me_claims: Vec<CeClaim<Cmt, F, K>> = Vec::new();
    let mut wp_me_claims: Vec<CeClaim<Cmt, F, K>> = Vec::new();
    let mut proofs: Vec<MemOrLutProof> = Vec::new();

    // --------------------------------------------------------------------
    // Phase 2: Twist val-eval sum-check (batched across mem instances).
    // --------------------------------------------------------------------
    let mut twist_val_eval_proofs: Vec<twist::TwistValEvalProof<K>> = Vec::new();
    let mut r_val: Vec<K> = Vec::new();
    if !step.mem_instances.is_empty() {
        let plan = crate::memory_sidecar::claim_plan::TwistValEvalClaimPlan::build(
            step.mem_instances.iter().map(|(inst, _)| inst),
            has_prev,
        );
        let n_mem = step.mem_instances.len();
        let claims_per_mem = plan.claims_per_mem;
        let claim_count = plan.claim_count;

        let mut val_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(claim_count);
        let mut bind_claims: Vec<(u8, K)> = Vec::with_capacity(claim_count);
        let mut claimed_sums: Vec<K> = Vec::with_capacity(claim_count);

        let mut claimed_inc_sums_lt: Vec<K> = Vec::with_capacity(n_mem);
        let mut claimed_inc_sums_total: Vec<K> = Vec::with_capacity(n_mem);
        let mut claimed_prev_inc_sums_total: Vec<Option<K>> = Vec::with_capacity(n_mem);

        let mut claim_idx = 0usize;
        for (i_mem, (mem_inst, _mem_wit)) in step.mem_instances.iter().enumerate() {
            let pre = twist_pre
                .get(i_mem)
                .ok_or_else(|| PiCcsError::ProtocolError("missing Twist pre-time data".into()))?;
            let decoded = &pre.decoded;
            let r_addr = &pre.addr_pre.r_addr;
            if decoded.lanes.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist(Route A): decoded lanes empty at mem_idx={i_mem}"
                )));
            }

            let mut lt_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(decoded.lanes.len());
            let mut claimed_inc_sum_lt = K::ZERO;
            for lane in decoded.lanes.iter() {
                let (oracle, claim) = TwistValEvalOracleSparseTime::new(
                    lane.wa_bits.clone(),
                    lane.has_write.clone(),
                    lane.inc_at_write_addr.clone(),
                    r_addr,
                    r_time,
                );
                lt_oracles.push(Box::new(oracle));
                claimed_inc_sum_lt += claim;
            }
            let oracle_lt: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(lt_oracles)?);

            let mut total_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(decoded.lanes.len());
            let mut claimed_inc_sum_total = K::ZERO;
            for lane in decoded.lanes.iter() {
                let (oracle, claim) = TwistTotalIncOracleSparseTime::new(
                    lane.wa_bits.clone(),
                    lane.has_write.clone(),
                    lane.inc_at_write_addr.clone(),
                    r_addr,
                );
                total_oracles.push(Box::new(oracle));
                claimed_inc_sum_total += claim;
            }
            let oracle_total: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(total_oracles)?);

            val_oracles.push(oracle_lt);
            bind_claims.push((plan.bind_tags[claim_idx], claimed_inc_sum_lt));
            claimed_sums.push(claimed_inc_sum_lt);
            claim_idx += 1;

            val_oracles.push(oracle_total);
            bind_claims.push((plan.bind_tags[claim_idx], claimed_inc_sum_total));
            claimed_sums.push(claimed_inc_sum_total);
            claim_idx += 1;

            claimed_inc_sums_lt.push(claimed_inc_sum_lt);
            claimed_inc_sums_total.push(claimed_inc_sum_total);

            if let Some(prev) = prev_step {
                let (prev_inst, _prev_wit) = prev
                    .mem_instances
                    .get(i_mem)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing prev mem instance".into()))?;
                if prev_inst.d != mem_inst.d
                    || prev_inst.ell != mem_inst.ell
                    || prev_inst.k != mem_inst.k
                    || prev_inst.lanes != mem_inst.lanes
                {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Twist rollover requires stable geometry at mem_idx={}: prev (k={}, d={}, ell={}, lanes={}) vs cur (k={}, d={}, ell={}, lanes={})",
                        i_mem,
                        prev_inst.k,
                        prev_inst.d,
                        prev_inst.ell,
                        prev_inst.lanes,
                        mem_inst.k,
                        mem_inst.d,
                        mem_inst.ell,
                        mem_inst.lanes
                    )));
                }
                let prev_decoded = prev_twist_decoded
                    .ok_or_else(|| PiCcsError::ProtocolError("missing prev Twist decoded cols".into()))?
                    .get(i_mem)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing prev Twist decoded cols at mem_idx".into()))?;
                if prev_decoded.lanes.is_empty() {
                    return Err(PiCcsError::ProtocolError(
                        "missing prev Twist decoded cols lanes".into(),
                    ));
                }

                let mut prev_total_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(prev_decoded.lanes.len());
                let mut claimed_prev_total = K::ZERO;
                for lane in prev_decoded.lanes.iter() {
                    let (oracle, claim) = TwistTotalIncOracleSparseTime::new(
                        lane.wa_bits.clone(),
                        lane.has_write.clone(),
                        lane.inc_at_write_addr.clone(),
                        r_addr,
                    );
                    prev_total_oracles.push(Box::new(oracle));
                    claimed_prev_total += claim;
                }
                let oracle_prev_total: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(prev_total_oracles)?);

                val_oracles.push(oracle_prev_total);
                bind_claims.push((plan.bind_tags[claim_idx], claimed_prev_total));
                claimed_sums.push(claimed_prev_total);
                claim_idx += 1;

                claimed_prev_inc_sums_total.push(Some(claimed_prev_total));
            } else {
                claimed_prev_inc_sums_total.push(None);
            }
        }

        tr.append_message(
            b"twist/val_eval/batch_start",
            &(step.mem_instances.len() as u64).to_le_bytes(),
        );
        tr.append_message(b"twist/val_eval/step_idx", &(step_idx as u64).to_le_bytes());
        bind_twist_val_eval_claim_sums(tr, &bind_claims);

        let mut claims: Vec<BatchedClaim<'_>> = val_oracles
            .iter_mut()
            .zip(claimed_sums.iter())
            .zip(plan.labels.iter())
            .map(|((oracle, sum), label)| BatchedClaim {
                oracle: oracle.as_mut(),
                claimed_sum: *sum,
                label: *label,
            })
            .collect();

        let (r_val_out, mut per_claim_results) =
            run_batched_sumcheck_prover_ds(tr, b"twist/val_eval_batch", step_idx, claims.as_mut_slice())?;

        if per_claim_results.len() != claim_count {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval results count mismatch (expected {}, got {})",
                claim_count,
                per_claim_results.len()
            )));
        }
        if r_val_out.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val_out.len(),
                r_time.len()
            )));
        }
        r_val = r_val_out;

        for i in 0..n_mem {
            let base = claims_per_mem * i;
            twist_val_eval_proofs.push(twist::TwistValEvalProof {
                claimed_inc_sum_lt: claimed_inc_sums_lt[i],
                rounds_lt: std::mem::take(&mut per_claim_results[base].round_polys),
                claimed_inc_sum_total: claimed_inc_sums_total[i],
                rounds_total: std::mem::take(&mut per_claim_results[base + 1].round_polys),
                claimed_prev_inc_sum_total: claimed_prev_inc_sums_total[i],
                rounds_prev_total: has_prev.then(|| std::mem::take(&mut per_claim_results[base + 2].round_polys)),
            });
        }

        tr.append_message(b"twist/val_eval/batch_done", &[]);
    }

    if step.lut_instances.is_empty() {
        if !shout_addr_pre.claimed_sums.is_empty() || !shout_addr_pre.groups.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "shout_addr_pre must be empty when there are no Shout instances".into(),
            ));
        }
    }

    for _ in 0..step.lut_instances.len() {
        proofs.push(MemOrLutProof::Shout(ShoutProofK::default()));
    }

    for idx in 0..step.mem_instances.len() {
        let mut proof = TwistProofK::default();
        proof.addr_pre = twist_pre
            .get(idx)
            .ok_or_else(|| PiCcsError::ProtocolError("missing Twist addr_pre".into()))?
            .addr_pre
            .clone();
        proof.val_eval = twist_val_eval_proofs.get(idx).cloned();

        proofs.push(MemOrLutProof::Twist(proof));
    }

    if !step.mem_instances.is_empty() {
        if r_val.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val.len(),
                r_time.len()
            )));
        }

        let core_t = s.t();

        // Shared-bus mode: val-lane checks read bus openings from CPU ME claims at r_val.
        // Emit CPU ME at r_val for current step (and previous step for rollover).
        let (mcs_inst, mcs_wit) = &step.mcs;
        let cpu_claims_cur = ts::emit_me_claims_for_mats(
            tr,
            b"cpu_bus/me_digest_val",
            params,
            s,
            core::slice::from_ref(&mcs_inst.c),
            core::slice::from_ref(&mcs_wit.Z),
            &r_val,
            m_in,
        )?;
        if cpu_claims_cur.len() != 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "expected exactly 1 CPU ME claim at r_val, got {}",
                cpu_claims_cur.len()
            )));
        }
        let mut cpu_claims_cur = cpu_claims_cur;
        if step.time_columns.t != cpu_bus.chunk_size || step.time_columns.mem_cols.len() != cpu_bus.bus_cols {
            return Err(PiCcsError::InvalidInput(format!(
                "uniform Route-A requires canonical time mem columns for val-lane openings (t={}, chunk_size={}, mem_cols={}, bus_cols={})",
                step.time_columns.t,
                cpu_bus.chunk_size,
                step.time_columns.mem_cols.len(),
                cpu_bus.bus_cols
            )));
        }
        crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance_from_time_columns(
            params,
            cpu_bus,
            core_t,
            &step.time_columns.mem_cols,
            &mut cpu_claims_cur[0],
        )?;
        val_me_claims.extend(cpu_claims_cur);

        if let Some(prev) = prev_step {
            let (prev_mcs_inst, prev_mcs_wit) = &prev.mcs;
            let cpu_claims_prev = ts::emit_me_claims_for_mats(
                tr,
                b"cpu_bus/me_digest_val",
                params,
                s,
                core::slice::from_ref(&prev_mcs_inst.c),
                core::slice::from_ref(&prev_mcs_wit.Z),
                &r_val,
                m_in,
            )?;
            if cpu_claims_prev.len() != 1 {
                return Err(PiCcsError::ProtocolError(format!(
                    "expected exactly 1 prev CPU ME claim at r_val, got {}",
                    cpu_claims_prev.len()
                )));
            }
            let mut cpu_claims_prev = cpu_claims_prev;
            // Val-lane continuity consumes previous-step bus openings at r_val.
            // In canonical mode those openings are sourced from prev step time mem columns,
            // so the shape must match the current bus layout exactly.
            if prev.time_columns.t != cpu_bus.chunk_size || prev.time_columns.mem_cols.len() != cpu_bus.bus_cols {
                return Err(PiCcsError::InvalidInput(format!(
                    "uniform Route-A requires canonical prev time mem columns for val-lane openings (t={}, chunk_size={}, mem_cols={}, bus_cols={})",
                    prev.time_columns.t,
                    cpu_bus.chunk_size,
                    prev.time_columns.mem_cols.len(),
                    cpu_bus.bus_cols
                )));
            }
            crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance_from_time_columns(
                params,
                cpu_bus,
                core_t,
                &prev.time_columns.mem_cols,
                &mut cpu_claims_prev[0],
            )?;
            val_me_claims.extend(cpu_claims_prev);
        }
    }

    if step.mem_instances.is_empty() {
        if !twist_val_eval_proofs.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-eval proofs must be empty when no mem instances are present".into(),
            ));
        }
        if !r_val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist r_val must be empty when no mem instances are present".into(),
            ));
        }
        if !val_me_claims.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-lane ME claims must be empty when no mem instances are present".into(),
            ));
        }
    } else if val_me_claims.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "twist val-eval requires non-empty val-lane ME claims".into(),
        ));
    }

    let (wb_claims, wp_claims) = emit_route_a_wb_wp_me_claims(tr, params, s, step, r_time)?;
    wb_me_claims.extend(wb_claims);
    wp_me_claims.extend(wp_claims);

    Ok(MemSidecarProof {
        val_me_claims,
        wb_me_claims,
        wp_me_claims,
        poseidon_cycle_me_claims: Vec::new(),
        poseidon_local_me_claims: Vec::new(),
        shout_addr_pre: shout_addr_pre.clone(),
        proofs,
    })
}

// ============================================================================
// ============================================================================
