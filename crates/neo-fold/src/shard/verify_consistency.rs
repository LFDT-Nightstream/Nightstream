use super::*;
use p3_field::PrimeField64;

#[inline]
fn point_key_words(point: &[K]) -> Vec<u64> {
    let coeffs_per_elem = point.first().map(|k| k.as_coeffs().len()).unwrap_or(0);
    let mut out = Vec::with_capacity(point.len().saturating_mul(coeffs_per_elem));
    for k in point.iter() {
        out.extend(k.as_coeffs().iter().map(|f| f.as_canonical_u64()));
    }
    out
}

fn sorted_col_eval_pairs(col_ids: &[usize], evals: &[K], label: &str) -> Result<Vec<(usize, K)>, PiCcsError> {
    if col_ids.len() != evals.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: malformed opening col_ids/evals length mismatch ({} vs {})",
            col_ids.len(),
            evals.len()
        )));
    }
    let mut pairs: Vec<(usize, K)> = col_ids.iter().copied().zip(evals.iter().copied()).collect();
    pairs.sort_unstable_by_key(|(col_id, _)| *col_id);
    if pairs.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: opening contains duplicate col_ids"
        )));
    }
    Ok(pairs)
}

#[inline]
fn bus_logical_col_ids_for_step(
    step_proof: &StepProof,
    bus: &neo_memory::cpu::BusLayout,
    label: &str,
) -> Result<Vec<usize>, PiCcsError> {
    let cpu_cols_len = step_proof.fold.time_cpu_commitments.len();
    let mem_cols_len = step_proof.fold.time_mem_commitments.len();
    if mem_cols_len != bus.bus_cols {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: proof mem commitment count {} != bus.bus_cols {}",
            mem_cols_len, bus.bus_cols
        )));
    }
    let total_cols = cpu_cols_len
        .checked_add(mem_cols_len)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: cpu+mem commitment count overflow")))?;
    if step_proof.fold.time_col_ids.len() != total_cols {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: proof logical col_id count {} != cpu+mem count {}",
            step_proof.fold.time_col_ids.len(),
            total_cols
        )));
    }
    Ok(step_proof.fold.time_col_ids[cpu_cols_len..].to_vec())
}

pub(crate) fn validate_step_time_opening_batches_with_transcript(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step_idx: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    step_proof: &StepProof,
    cpu_bus: &neo_memory::cpu::BusLayout,
) -> Result<(), PiCcsError> {
    let fold = &step_proof.fold;
    let t = fold.time_t;
    let manifest = crate::time_opening::manifest::build_opening_claim_manifest(
        &fold.openings,
        &fold.opening_proofs,
        &fold.time_col_ids,
        fold.time_cpu_commitments.len(),
    )?;
    if manifest != fold.opening_manifest {
        return Err(PiCcsError::ProtocolError(
            "verify/time-opening stage8: opening manifest mismatch".into(),
        ));
    }
    crate::time_opening::manifest::bind_opening_claim_manifest(tr, step_idx, &manifest);
    let all_coeffs = bind_time_opening_batches_and_sample_coeffs(tr, params, step_idx, &fold.opening_proofs)?;
    if all_coeffs.len() != fold.opening_proofs.len() {
        return Err(PiCcsError::ProtocolError(
            "verify/time-opening stage8: opening-batch coefficient/proof length mismatch".into(),
        ));
    }
    let reduction = crate::time_opening::reduction::build_opening_reduction(&manifest)?;
    if reduction.groups != fold.opening_reduction.groups {
        return Err(PiCcsError::ProtocolError(
            "verify/time-opening stage8: reduction proof mismatch".into(),
        ));
    }
    crate::time_opening::reduction::verify_opening_unification_sumcheck(
        tr,
        step_idx,
        &fold.opening_reduction,
        &fold.opening_unification,
    )?;
    crate::time_opening::joint_lane::verify_joint_opening_lane(
        tr,
        params,
        step_idx,
        step,
        cpu_bus,
        t,
        &fold.time_cpu_commitments,
        &fold.time_mem_commitments,
        &fold.time_col_ids,
        &fold.opening_proofs,
        &fold.opening_manifest.digest,
        &fold.opening_reduction,
        &fold.opening_unification,
        &fold.joint_opening_lane,
        &all_coeffs,
    )
}

pub(crate) fn validate_step_time_openings_consistency(
    step: &StepInstanceBundle<Cmt, F, K>,
    step_proof: &StepProof,
    cpu_bus: &neo_memory::cpu::BusLayout,
    r_time: &[K],
) -> Result<(), PiCcsError> {
    let openings = &step_proof.fold.openings;
    let expected_logical_cols = step_proof
        .fold
        .time_cpu_commitments
        .len()
        .saturating_add(step_proof.fold.time_mem_commitments.len());
    if step_proof.fold.time_t == 0 {
        return Err(PiCcsError::ProtocolError(
            "verify/openings: canonical committed mode requires time_t > 0".into(),
        ));
    }
    if expected_logical_cols == 0 || step_proof.fold.time_col_ids.len() != expected_logical_cols {
        return Err(PiCcsError::ProtocolError(format!(
            "verify/openings: canonical committed mode requires logical col ids (time_col_ids={}, expected={expected_logical_cols})",
            step_proof.fold.time_col_ids.len()
        )));
    }
    for (idx, &col_id) in step_proof.fold.time_col_ids.iter().enumerate() {
        if col_id != idx {
            return Err(PiCcsError::ProtocolError(format!(
                "verify/openings: canonical logical col_id mapping required (time_col_ids[{idx}]={col_id}, expected {idx})"
            )));
        }
    }
    if openings.is_empty() != step_proof.fold.opening_proofs.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "verify/openings: fold.openings and fold.opening_proofs must be both empty or both non-empty".into(),
        ));
    }
    for (idx, opening) in openings.iter().enumerate() {
        if opening.point.is_empty() {
            return Err(PiCcsError::ProtocolError(format!(
                "step opening[{idx}] has empty point"
            )));
        }
        if opening.col_ids.len() != opening.evals.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "step opening[{idx}] malformed: col_ids.len()={} != evals.len()={}",
                opening.col_ids.len(),
                opening.evals.len()
            )));
        }
        let unique: std::collections::BTreeSet<usize> = opening.col_ids.iter().copied().collect();
        if unique.len() != opening.col_ids.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "step opening[{idx}] has duplicate col_ids"
            )));
        }
        if opening.source == crate::shard_proof_types::TimeOpeningSource::Unknown {
            return Err(PiCcsError::ProtocolError(format!(
                "step opening[{idx}] has unknown source"
            )));
        }
        if opening.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "step opening[{idx}] has non-committed source {:?}",
                opening.source
            )));
        }
    }

    let mut openings_map = std::collections::BTreeMap::<(Vec<u64>, Vec<usize>), Vec<K>>::new();
    for (idx, opening) in openings.iter().enumerate() {
        let pairs = sorted_col_eval_pairs(&opening.col_ids, &opening.evals, "verify/openings")?;
        let key_cols: Vec<usize> = pairs.iter().map(|(col_id, _)| *col_id).collect();
        let key_point = point_key_words(&opening.point);
        let key = (key_point, key_cols);
        let key_evals: Vec<K> = pairs.into_iter().map(|(_, eval)| eval).collect();
        if openings_map.insert(key, key_evals).is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "step openings contain duplicate (point,col_ids) entry at index {idx}"
            )));
        }
    }

    let allowed_col_ids: std::collections::BTreeSet<usize> = step_proof.fold.time_col_ids.iter().copied().collect();
    let mut proofs_map = std::collections::BTreeMap::<(Vec<u64>, Vec<usize>), Vec<K>>::new();
    for (idx, pf) in step_proof.fold.opening_proofs.iter().enumerate() {
        if pf.point.is_empty() {
            return Err(PiCcsError::ProtocolError(format!(
                "step opening_proofs[{idx}] has empty point"
            )));
        }
        if pf.col_ids.is_empty() || pf.col_ids.len() != pf.evals.len() || pf.col_ids.len() != pf.digit_evals.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "step opening_proofs[{idx}] malformed lengths (col_ids={}, evals={}, digit_evals={})",
                pf.col_ids.len(),
                pf.evals.len(),
                pf.digit_evals.len()
            )));
        }
        if !pf.col_ids.windows(2).all(|w| w[0] < w[1]) {
            return Err(PiCcsError::ProtocolError(format!(
                "step opening_proofs[{idx}] col_ids must be strictly sorted"
            )));
        }
        for &col_id in pf.col_ids.iter() {
            if !allowed_col_ids.contains(&col_id) {
                return Err(PiCcsError::ProtocolError(format!(
                    "step opening_proofs[{idx}] col_id={} not in time_col_ids",
                    col_id
                )));
            }
        }
        for (d_idx, row) in pf.digit_evals.iter().enumerate() {
            if row.len() != neo_math::D {
                return Err(PiCcsError::ProtocolError(format!(
                    "step opening_proofs[{idx}] digit_evals[{d_idx}] len {} != D={}",
                    row.len(),
                    neo_math::D
                )));
            }
        }
        let key_point = point_key_words(&pf.point);
        let key = (key_point, pf.col_ids.clone());
        if proofs_map.insert(key, pf.evals.clone()).is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "step opening_proofs contains duplicate (point,col_ids) entry at index {idx}"
            )));
        }
    }
    if openings_map != proofs_map {
        return Err(PiCcsError::ProtocolError(
            "verify/openings: fold.openings must exactly match fold.opening_proofs (point,col_ids,evals)".into(),
        ));
    }

    if cpu_bus.bus_cols > 0 {
        let col_ids = bus_logical_col_ids_for_step(step_proof, cpu_bus, "verify/openings bus")?;
        let opening_entry = crate::memory_sidecar::memory::require_time_opening_entry_for_point(
            openings,
            r_time,
            &col_ids,
            "verify/openings bus",
        )?;
        if opening_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "verify/openings bus requires CommittedOpening source (got {:?})",
                opening_entry.source
            )));
        }
    }

    if cpu_bus.bus_cols > 0 && !step_proof.mem.val_me_claims.is_empty() {
        let cpu_me_val_cur = step_proof
            .mem
            .val_me_claims
            .first()
            .ok_or_else(|| PiCcsError::ProtocolError("verify/openings val: missing val_me_claims[0]".into()))?;
        let col_ids = bus_logical_col_ids_for_step(step_proof, cpu_bus, "verify/openings val")?;
        let opening_entry = crate::memory_sidecar::memory::require_time_opening_entry_for_point(
            openings,
            cpu_me_val_cur.r.as_slice(),
            col_ids.as_slice(),
            "verify/openings val",
        )?;
        if opening_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "verify/openings val requires CommittedOpening source (got {:?})",
                opening_entry.source
            )));
        }
    }

    if crate::memory_sidecar::memory::wb_wp_required_for_step_instance(step) && step.mcs_inst.m_in == 5 {
        let trace = neo_memory::riscv::trace::Rv32TraceLayout::new();
        let trace_cols: Vec<usize> = vec![
            trace.active,
            trace.cycle,
            trace.pc_before,
            trace.instr_word,
            trace.rs1_addr,
            trace.rs1_val,
            trace.rs2_addr,
            trace.rs2_val,
            trace.rd_addr,
            trace.rd_val,
            trace.ram_addr,
            trace.ram_rv,
            trace.ram_wv,
            trace.shout_has_lookup,
            trace.shout_val,
            trace.shout_link_lhs,
            trace.shout_link_rhs,
            trace.shout_add_sub_key,
        ];
        let opening_entry = crate::memory_sidecar::memory::require_time_opening_entry_for_point(
            openings,
            r_time,
            &trace_cols,
            "verify/openings trace",
        )?;
        if opening_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "verify/openings trace requires CommittedOpening source (got {:?})",
                opening_entry.source
            )));
        }
    }

    if let Some(wb_me) = step_proof.mem.wb_me_claims.first() {
        let trace = neo_memory::riscv::trace::Rv32TraceLayout::new();
        let wb_cols = crate::memory_sidecar::memory::rv32_trace_wb_columns(&trace);
        let opening_entry = crate::memory_sidecar::memory::require_time_opening_entry_for_point(
            openings,
            wb_me.r.as_slice(),
            &wb_cols,
            "verify/openings wb",
        )?;
        if opening_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "verify/openings wb requires CommittedOpening source (got {:?})",
                opening_entry.source
            )));
        }
    }

    if let Some(wp_me) = step_proof.mem.wp_me_claims.first() {
        let trace = neo_memory::riscv::trace::Rv32TraceLayout::new();
        let mut wp_cols = crate::memory_sidecar::memory::rv32_trace_wp_opening_columns(&trace);
        if crate::memory_sidecar::memory::control_stage_required_for_step_instance(step) {
            wp_cols.extend(crate::memory_sidecar::memory::rv32_trace_control_extra_opening_columns(
                &trace,
            ));
        }
        let opening_entry = crate::memory_sidecar::memory::require_time_opening_entry_for_point(
            openings,
            wp_me.r.as_slice(),
            &wp_cols,
            "verify/openings wp",
        )?;
        if opening_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "verify/openings wp requires CommittedOpening source (got {:?})",
                opening_entry.source
            )));
        }
    }

    Ok(())
}

pub(crate) fn validate_time_sumcheck_metadata(
    step_idx: usize,
    step_proof: &StepProof,
    ccs_r_time: &[K],
    route_r_time: &[K],
    control_required: bool,
) -> Result<(), PiCcsError> {
    let labels = &step_proof.batched_time.labels;
    let claimed_sums = &step_proof.batched_time.claimed_sums;
    let round_polys = &step_proof.batched_time.round_polys;

    let ccs_ell_n = ccs_r_time.len();
    if step_proof.fold.ccs_proof.sumcheck_rounds.len() < ccs_ell_n {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: ccs_proof has too few rounds for cpu_sumcheck metadata",
            step_idx
        )));
    }
    let expected_cpu_sum = step_proof
        .fold
        .ccs_proof
        .sc_initial_sum
        .ok_or_else(|| PiCcsError::ProtocolError(format!("step {}: missing ccs_proof.sc_initial_sum", step_idx)))?;
    let expected_cpu_rounds = &step_proof.fold.ccs_proof.sumcheck_rounds[..ccs_ell_n];
    if step_proof.fold.cpu_sumcheck.claimed_sum != expected_cpu_sum {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: cpu_sumcheck claimed_sum mismatch",
            step_idx
        )));
    }
    if step_proof.fold.cpu_sumcheck.round_polys.as_slice() != expected_cpu_rounds {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: cpu_sumcheck round_polys mismatch",
            step_idx
        )));
    }
    if step_proof.fold.cpu_sumcheck.r_time.as_slice() != ccs_r_time {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: cpu_sumcheck r_time mismatch",
            step_idx
        )));
    }

    let control_idx = labels
        .iter()
        .position(|label| label.as_slice() == b"control/next_pc_linear");
    match control_idx {
        Some(expected_shift_idx) => {
            let expected_shift_sum = *claimed_sums.get(expected_shift_idx).ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "step {}: missing batched_time claimed_sum for shift index {}",
                    step_idx, expected_shift_idx
                ))
            })?;
            let expected_shift_rounds = round_polys.get(expected_shift_idx).ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "step {}: missing batched_time rounds for shift index {}",
                    step_idx, expected_shift_idx
                ))
            })?;
            if step_proof.fold.shift_sumcheck.claimed_sum != expected_shift_sum {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: shift_sumcheck claimed_sum mismatch",
                    step_idx
                )));
            }
            if step_proof.fold.shift_sumcheck.round_polys != *expected_shift_rounds {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: shift_sumcheck round_polys mismatch",
                    step_idx
                )));
            }
        }
        None => {
            if control_required {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: missing batched_time label control/next_pc_linear",
                    step_idx
                )));
            }
            if step_proof.fold.shift_sumcheck.claimed_sum != K::ZERO {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: shift_sumcheck must be zero-valued when control stage is disabled",
                    step_idx
                )));
            }
            if !step_proof.fold.shift_sumcheck.round_polys.is_empty() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: shift_sumcheck rounds must be empty when control stage is disabled",
                    step_idx
                )));
            }
        }
    }
    if step_proof.fold.shift_sumcheck.r_time.as_slice() != route_r_time {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: shift_sumcheck r_time mismatch",
            step_idx
        )));
    }

    Ok(())
}
