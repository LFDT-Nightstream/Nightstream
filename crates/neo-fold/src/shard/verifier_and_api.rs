use super::*;

pub fn fold_shard_prove<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_prove_mixed_ccs_batched(mode, tr, params, s_me, steps, acc_init, acc_wit_init, l, mixers, None)
}

pub(crate) fn fold_shard_prove_with_context<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    ctx: &ShardProverContext,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_prove_mixed_ccs_batched(
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        Some(ctx),
    )
}

pub(crate) fn fold_shard_prove_with_context_and_step_timings<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    ctx: &ShardProverContext,
) -> Result<(ShardProof, Vec<f64>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let start = time_now();
    let proof = fold_shard_prove_mixed_ccs_batched(
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        Some(ctx),
    )?;
    let total_ms = elapsed_ms(start);
    let step_count = proof.steps.len();
    let step_prove_ms = if step_count == 0 {
        Vec::new()
    } else {
        vec![total_ms / (step_count as f64); step_count]
    };
    Ok((proof, step_prove_ms))
}

pub fn fold_shard_prove_with_output_binding<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    final_memory_state: &[F],
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_prove_mixed_ccs_batched_with_output_binding(
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        ob_cfg,
        final_memory_state,
        None,
    )
}

pub(crate) fn fold_shard_prove_with_output_binding_with_context<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    final_memory_state: &[F],
    ctx: &ShardProverContext,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_prove_mixed_ccs_batched_with_output_binding(
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        ob_cfg,
        final_memory_state,
        Some(ctx),
    )
}

pub fn fold_shard_prove_with_witnesses<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(ShardProof, ShardFoldOutputs<Cmt, F, K>, ShardFoldWitnesses<F>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, final_main_wits, val_lane_wits) = fold_shard_prove_mixed_ccs_batched_with_witnesses(
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        0,
        None,
    )?;
    let outputs = proof.compute_fold_outputs(acc_init);
    if outputs.obligations.main.len() != final_main_wits.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "final main witness count mismatch (have {}, need {})",
            final_main_wits.len(),
            outputs.obligations.main.len()
        )));
    }
    if outputs.obligations.val.len() != val_lane_wits.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "val-lane witness count mismatch (have {}, need {})",
            val_lane_wits.len(),
            outputs.obligations.val.len()
        )));
    }
    Ok((
        proof,
        outputs,
        ShardFoldWitnesses {
            final_main_wits,
            val_lane_wits,
        },
    ))
}

/// Same as `fold_shard_prove_with_witnesses`, but offsets the per-step transcript index by `step_idx_offset`.
///
/// This is useful for "continuation" style proving across multiple calls while preserving a globally
/// increasing step index for transcript domain separation.
pub fn fold_shard_prove_with_witnesses_with_step_offset<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    step_idx_offset: usize,
) -> Result<(ShardProof, ShardFoldOutputs<Cmt, F, K>, ShardFoldWitnesses<F>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, final_main_wits, val_lane_wits) = fold_shard_prove_mixed_ccs_batched_with_witnesses(
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        step_idx_offset,
        None,
    )?;
    let outputs = proof.compute_fold_outputs(acc_init);
    if outputs.obligations.main.len() != final_main_wits.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "final main witness count mismatch (have {}, need {})",
            final_main_wits.len(),
            outputs.obligations.main.len()
        )));
    }
    if outputs.obligations.val.len() != val_lane_wits.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "val-lane witness count mismatch (have {}, need {})",
            val_lane_wits.len(),
            outputs.obligations.val.len()
        )));
    }
    Ok((
        proof,
        outputs,
        ShardFoldWitnesses {
            final_main_wits,
            val_lane_wits,
        },
    ))
}

// ============================================================================
// Shard Verification
// ============================================================================

fn merge_step_time_columns_from_proof(
    step: &StepInstanceBundle<Cmt, F, K>,
    step_proof: &StepProof,
) -> Result<StepInstanceBundle<Cmt, F, K>, PiCcsError> {
    let mut out = step.clone();
    let proof_t = step_proof.fold.time_t;
    let proof_col_ids = &step_proof.fold.time_col_ids;
    let cpu_cols_len = step_proof.fold.time_cpu_commitments.len();
    let mem_cols_len = step_proof.fold.time_mem_commitments.len();
    let expected_col_ids_len = cpu_cols_len
        .checked_add(mem_cols_len)
        .ok_or_else(|| PiCcsError::InvalidInput("step proof time col_id length overflow".into()))?;
    if proof_col_ids.len() != expected_col_ids_len {
        return Err(PiCcsError::ProtocolError(format!(
            "step proof time_col_ids length mismatch: got {}, expected {} (=cpu_commitments+mem_commitments)",
            proof_col_ids.len(),
            expected_col_ids_len
        )));
    }
    if proof_col_ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != proof_col_ids.len()
    {
        return Err(PiCcsError::ProtocolError(
            "step proof time_col_ids contains duplicates".into(),
        ));
    }
    for (idx, &col_id) in proof_col_ids.iter().enumerate() {
        if col_id != idx {
            return Err(PiCcsError::ProtocolError(format!(
                "step proof time_col_ids must be canonical contiguous ids (time_col_ids[{idx}]={col_id}, expected {idx})"
            )));
        }
    }

    if proof_t == 0 && expected_col_ids_len > 0 {
        return Err(PiCcsError::ProtocolError(
            "step proof time columns are malformed: time_t=0 with non-empty time commitments".into(),
        ));
    }

    // Verifier-side acceptance must not depend on statement-local time column payload.
    // Replace statement payload with proof metadata only (t + logical ids + committed counts).
    out.time_columns.t = proof_t;
    out.time_columns.cpu_cols = vec![Vec::new(); cpu_cols_len];
    out.time_columns.mem_cols = vec![Vec::new(); mem_cols_len];
    out.time_columns.col_ids = proof_col_ids.clone();

    Ok(out)
}

pub(crate) fn fold_shard_verify_impl<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    step_idx_offset: usize,
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: Option<&crate::output_binding::OutputBindingConfig>,
    prover_ctx: Option<&ShardProverContext>,
    initial_prev_step: Option<&StepInstanceBundle<Cmt, F, K>>,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if steps.len() != proof.steps.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "step count mismatch: public {} vs proof {}",
            steps.len(),
            proof.steps.len()
        )));
    }

    let mut effective_steps = Vec::with_capacity(steps.len());
    for (step, step_proof) in steps.iter().zip(proof.steps.iter()) {
        effective_steps.push(merge_step_time_columns_from_proof(step, step_proof)?);
    }

    for (step_idx, step) in effective_steps.iter().enumerate() {
        if step.lut_insts.is_empty() && step.mem_insts.is_empty() {
            continue;
        }
        let is_shared_step = step.lut_insts.iter().all(|inst| inst.comms.is_empty())
            && step.mem_insts.iter().all(|inst| inst.comms.is_empty());
        if !is_shared_step {
            return Err(PiCcsError::InvalidInput(format!(
                "legacy no-shared CPU bus mode was removed; step_idx={step_idx} must use shared-bus statement format"
            )));
        }
    }
    tr.append_message(b"shard/cpu_bus_mode", &[1u8]);
    let (s, cpu_bus) = crate::memory_sidecar::cpu_bus::prepare_ccs_for_shared_cpu_bus_steps(s_me, &effective_steps)?;
    let dims = utils::build_dims_and_policy(params, s)?;
    let utils::Dims {
        ell_d,
        ell_n,
        ell_m,
        ell,
        d_sc,
        ..
    } = dims;
    let ring = ccs::RotRing::goldilocks();

    if ob_cfg.is_some() && steps.is_empty() {
        return Err(PiCcsError::InvalidInput("output binding requires >= 1 step".into()));
    }
    if ob_cfg.is_none() && proof.output_proof.is_some() {
        return Err(PiCcsError::InvalidInput(
            "shard proof contains output binding, but verifier did not supply OutputBindingConfig".into(),
        ));
    }
    if ob_cfg.is_some() && proof.output_proof.is_none() {
        return Err(PiCcsError::InvalidInput(
            "verifier supplied OutputBindingConfig, but shard proof has no output binding".into(),
        ));
    }

    let mut accumulator = acc_init.to_vec();
    let mut val_lane_obligations: Vec<CeClaim<Cmt, F, K>> = Vec::new();
    let ccs_sparse_cache: Option<Arc<SparseCache<F>>> = if mode_uses_sparse_cache(&mode) {
        Some(
            prover_ctx
                .and_then(|ctx| ctx.ccs_sparse_cache.clone())
                .unwrap_or_else(|| Arc::new(SparseCache::build(s))),
        )
    } else {
        None
    };
    let ccs_mat_digest = prover_ctx
        .map(|ctx| ctx.ccs_mat_digest.clone())
        .unwrap_or_else(|| utils::digest_ccs_matrices_with_sparse_cache(s, ccs_sparse_cache.as_deref()));

    for (idx, (step, step_proof)) in effective_steps.iter().zip(proof.steps.iter()).enumerate() {
        let step_idx = step_idx_offset
            .checked_add(idx)
            .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;
        let has_prev = idx > 0 || initial_prev_step.is_some();
        absorb_step_memory(tr, step);

        let include_ob = ob_cfg.is_some() && (idx + 1 == steps.len());
        let mut ob_state: Option<neo_memory::output_check::OutputSumcheckState> = None;
        let mut ob_sparse_addr_weights: Option<Vec<(Vec<K>, K)>> = None;
        let mut ob_sparse_val_offset: Option<K> = None;
        let mut ob_reg_exact_linkage_degree_bound: Option<usize> = None;
        let mut ob_inc_total_degree_bound: Option<usize> = None;

        if include_ob {
            let cfg =
                ob_cfg.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;
            let ob_proof = proof
                .output_proof
                .as_ref()
                .ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but proof missing".into()))?;

            if cfg.mem_idx >= step.mem_insts.len() {
                return Err(PiCcsError::InvalidInput("output binding mem_idx out of range".into()));
            }
            let mem_inst = step
                .mem_insts
                .get(cfg.mem_idx)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding mem_idx out of range".into()))?;
            let expected_k = 1usize
                .checked_shl(cfg.num_bits as u32)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
            if mem_inst.k != expected_k {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits implies k={}, but mem_inst.k={}",
                    expected_k, mem_inst.k
                )));
            }
            let ell_addr = mem_inst.twist_layout().lanes[0].ell_addr;
            if ell_addr != cfg.num_bits {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits={}, but twist_layout.ell_addr={}",
                    cfg.num_bits, ell_addr
                )));
            }
            if mem_inst.mem_id == neo_memory::riscv::lookups::REG_EXACT_ID.0 {
                ob_reg_exact_linkage_degree_bound =
                    Some(crate::memory_sidecar::memory::RV64_REG_EXACT_LINKAGE_DEGREE_BOUND);
            }

            tr.append_message(b"shard/output_binding_start", &(step_idx as u64).to_le_bytes());
            tr.append_u64s(b"output_binding/mem_idx", &[cfg.mem_idx as u64]);
            tr.append_u64s(b"output_binding/num_bits", &[cfg.num_bits as u64]);

            let use_dense_output_sumcheck = cfg.num_bits <= neo_memory::output_check::OUTPUT_SUMCHECK_MAX_NUM_BITS;
            if use_dense_output_sumcheck {
                let state = neo_memory::output_check::verify_output_sumcheck_rounds_get_state(
                    tr,
                    cfg.num_bits,
                    cfg.program_io.clone(),
                    &ob_proof.output_sc,
                )
                .map_err(|e| PiCcsError::ProtocolError(format!("output sumcheck failed: {e:?}")))?;
                ob_state = Some(state);
            } else {
                if !ob_proof.output_sc.round_polys.is_empty() {
                    return Err(PiCcsError::ProtocolError(
                        "output sparse binding expects empty output sumcheck rounds".into(),
                    ));
                }
                let sampled = crate::output_binding::sample_output_lincomb_weights(tr, &cfg.program_io);
                let mut addr_weights = Vec::with_capacity(sampled.len());
                let mut val_offset = K::ZERO;
                for (addr, claimed_value, alpha) in sampled {
                    let addr_bits = crate::output_binding::addr_bits_as_k(addr, cfg.num_bits);
                    let val_init =
                        crate::output_binding::val_init_from_mem_init(&mem_inst.init, mem_inst.k, &addr_bits)
                            .map_err(|e| PiCcsError::ProtocolError(format!("MemInit eval failed: {e:?}")))?;
                    addr_weights.push((addr_bits, alpha));
                    let claimed_k: K = claimed_value.into();
                    val_offset += alpha * (val_init - claimed_k);
                }
                ob_sparse_addr_weights = Some(addr_weights);
                ob_sparse_val_offset = Some(val_offset);
            }
            ob_inc_total_degree_bound = Some(2 + ell_addr);
        }

        let mcs_inst = &step.mcs_inst;

        // --------------------------------------------------------------------
        // Route A: Verify shared-challenge batched sum-check (time/row rounds),
        // then finish CCS Ajtai rounds, then proceed with RLC→DEC as before.
        // --------------------------------------------------------------------

        // Bind CCS header + ME inputs and sample public challenges.
        utils::bind_header_and_instances_with_digest(
            tr,
            params,
            &s,
            core::slice::from_ref(mcs_inst),
            dims,
            &ccs_mat_digest,
        )?;
        utils::bind_me_inputs(tr, &accumulator)?;
        let expected_time_col_ids = step_proof
            .fold
            .time_cpu_commitments
            .len()
            .checked_add(step_proof.fold.time_mem_commitments.len())
            .ok_or_else(|| PiCcsError::InvalidInput("verify/time_columns: commitment count overflow".into()))?;
        if step_proof.fold.time_col_ids.len() != expected_time_col_ids {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: verify/time_columns col_ids mismatch: proof has {}, commitments imply {}",
                idx,
                step_proof.fold.time_col_ids.len(),
                expected_time_col_ids
            )));
        }
        if step_proof.fold.time_declared_len > step_proof.fold.time_t {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: verify/time_columns declared len {} exceeds time_t {}",
                idx, step_proof.fold.time_declared_len, step_proof.fold.time_t
            )));
        }
        let has_stage8_artifacts = !step_proof.fold.openings.is_empty()
            || !step_proof.fold.opening_proofs.is_empty()
            || !step_proof.fold.opening_unification.round_polys.is_empty()
            || !step_proof.fold.joint_opening_lane.groups.is_empty()
            || step_proof.fold.joint_opening_lane.unified_fold.is_some()
            || !step_proof.stage8_fold.is_empty()
            || !step_proof.fold.opening_manifest.entries.is_empty()
            || !step_proof.fold.opening_reduction.groups.is_empty();
        let requires_stage8_openings = cpu_bus.bus_cols > 0
            || !step.mem_insts.is_empty()
            || !step.lut_insts.is_empty()
            || !step_proof.mem.wb_me_claims.is_empty()
            || !step_proof.mem.wp_me_claims.is_empty();
        if requires_stage8_openings && !has_stage8_artifacts {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: missing Stage-8 artifacts for load-bearing named openings",
                idx
            )));
        }
        if has_stage8_artifacts {
            if step_proof.fold.openings.is_empty()
                || step_proof.fold.opening_proofs.is_empty()
                || step_proof.fold.opening_manifest.entries.is_empty()
                || step_proof.fold.opening_reduction.groups.is_empty()
                || step_proof.fold.opening_unification.round_polys.is_empty()
                || step_proof.fold.joint_opening_lane.groups.is_empty()
            {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: malformed Stage-8 artifact set (canonical mode requires openings/proofs/manifest/reduction/unification/groups)",
                    idx
                )));
            }
            let expected_stage8_fold_len = if step_proof.fold.joint_opening_lane.groups.is_empty() {
                0usize
            } else {
                1usize
            };
            if step_proof.stage8_fold.len() != expected_stage8_fold_len {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: malformed Stage-8 artifact set (stage8_fold proofs={}, expected {})",
                    idx,
                    step_proof.stage8_fold.len(),
                    expected_stage8_fold_len
                )));
            }
        }
        if has_stage8_artifacts {
            if step_proof.fold.time_t == 0 {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: verify/time_columns time_t must be > 0 in Stage-8 committed mode",
                    idx
                )));
            }
            let observed_declared_len = validate_time_active_mask_and_count(
                step.time_columns.active_col.as_slice(),
                step_proof.fold.time_t,
                "verify/time_columns",
            )?;
            if observed_declared_len != step_proof.fold.time_declared_len {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: verify/time_columns declared len mismatch (proof={}, observed={})",
                    idx, step_proof.fold.time_declared_len, observed_declared_len
                )));
            }
        }
        bind_time_column_commitments(
            tr,
            step_idx,
            step_proof.fold.time_t,
            step_proof.fold.time_declared_len,
            &step_proof.fold.time_col_ids,
            &step_proof.fold.time_cpu_commitments,
            &step_proof.fold.time_mem_commitments,
        );
        let mut ch = utils::sample_challenges(tr, ell_d, ell)?;
        if step_proof.fold.ccs_proof.variant == crate::optimized_engine::PiCcsProofVariant::SplitNcV1 {
            ch.beta_m = utils::sample_beta_m(tr, ell_m)?;
        }
        let expected_ch = &step_proof.fold.ccs_proof.challenges_public;
        if expected_ch.alpha != ch.alpha
            || expected_ch.beta_a != ch.beta_a
            || expected_ch.beta_r != ch.beta_r
            || expected_ch.beta_m != ch.beta_m
            || expected_ch.gamma != ch.gamma
        {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS challenges_public mismatch",
                idx
            )));
        }

        // Public initial sum T for CCS sumcheck (engine-selected).
        let claimed_initial = match &mode {
            FoldingMode::Optimized => {
                crate::optimized_engine::claimed_initial_sum_from_inputs_with_k_mcs(&s, &ch, 1, &accumulator)
            }
            #[cfg(feature = "paper-exact")]
            FoldingMode::PaperExact => {
                crate::paper_exact_engine::claimed_initial_sum_from_inputs_with_k_mcs(&s, &ch, 1, &accumulator)
            }
            #[cfg(feature = "paper-exact")]
            FoldingMode::OptimizedWithCrosscheck(_) => {
                crate::optimized_engine::claimed_initial_sum_from_inputs_with_k_mcs(&s, &ch, 1, &accumulator)
            }
        };
        if let Some(x) = step_proof.fold.ccs_proof.sc_initial_sum {
            if x != claimed_initial {
                return Err(PiCcsError::SumcheckError(
                    "initial sum mismatch: proof claims different value than public T".into(),
                ));
            }
        }
        tr.append_fields(b"sumcheck/initial_sum", &claimed_initial.as_coeffs());

        let poseidon_runtime = prepare_poseidon_verifier_runtime(tr, step, step_proof, step_idx)?;
        let poseidon_cycle_enabled = poseidon_runtime.cycle_enabled;
        let poseidon_verify_setup = poseidon_runtime.verify_setup;
        let poseidon_link_chals = poseidon_runtime.link_chals;
        let poseidon_cont_chals = poseidon_runtime.cont_chals;

        // Route A memory checks use a separate transcript-derived cycle point `r_cycle`
        // to form χ_{r_cycle}(t) weights inside their sum-check polynomials.
        let route_steps = {
            let inst_steps = step
                .lut_insts
                .iter()
                .map(|inst| inst.steps)
                .chain(step.mem_insts.iter().map(|inst| inst.steps))
                .max()
                .unwrap_or(0);
            core::cmp::max(step_proof.fold.time_t, inst_steps)
        };
        let route_domain = step
            .mcs_inst
            .m_in
            .checked_add(route_steps)
            .ok_or_else(|| PiCcsError::ProtocolError("verify/route_a: route domain overflow".into()))?;
        // Keep Route-A row challenge dimension aligned with Π_RLC/Π_DEC validators,
        // which use at least one row bit even for n=1 domains.
        let route_pow2 = route_domain.max(2).next_power_of_two();
        let ell_t = route_pow2.trailing_zeros() as usize;
        let r_cycle: Vec<K> =
            ts::sample_ext_point(tr, b"route_a/r_cycle", b"route_a/cycle/0", b"route_a/cycle/1", ell_t);

        let shout_pre = crate::memory_sidecar::memory::verify_shout_addr_pre_time(tr, step, &step_proof.mem, step_idx)?;
        let twist_pre = crate::memory_sidecar::memory::verify_twist_addr_pre_time(tr, step, &step_proof.mem)?;
        let wb_enabled = crate::memory_sidecar::memory::wb_wp_required_for_step_instance(step);
        let wp_enabled = crate::memory_sidecar::memory::wb_wp_required_for_step_instance(step);
        let decode_stage_enabled = crate::memory_sidecar::memory::decode_stage_required_for_step_instance(step);
        let width_stage_enabled = crate::memory_sidecar::memory::width_stage_required_for_step_instance(step)
            || crate::memory_sidecar::memory::rv64_fullword_width_stage_required_from_proof(
                step,
                &step_proof.batched_time,
            );
        let control_stage_enabled = crate::memory_sidecar::memory::control_stage_required_for_step_instance(step);
        let crate::memory_sidecar::route_a_time::RouteABatchedTimeVerifyOutput {
            r_time: route_r_time,
            final_values,
        } = crate::memory_sidecar::route_a_time::verify_route_a_batched_time(
            tr,
            step_idx,
            ell_t,
            step,
            &step_proof.batched_time,
            wb_enabled,
            wp_enabled,
            decode_stage_enabled,
            width_stage_enabled,
            control_stage_enabled,
            poseidon_cycle_enabled,
            ob_reg_exact_linkage_degree_bound,
            ob_inc_total_degree_bound,
        )?;
        if route_r_time.len() != ell_t {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Route-A r_time length mismatch (got {}, expected ell_t={ell_t})",
                idx,
                route_r_time.len()
            )));
        }

        let poseidon_local_checked =
            verify_poseidon_local_time_from_setup(tr, step_proof, step_idx, &poseidon_verify_setup)?;
        ensure_poseidon_link_sums_match_verify(
            poseidon_cycle_enabled,
            &step_proof.batched_time,
            step_proof.poseidon_local_time.as_ref(),
        )?;

        // CCS proof structure consistency.
        let want_rounds_total = ell_n + ell_d;
        if step_proof.fold.ccs_proof.sumcheck_rounds.len() != want_rounds_total {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: CCS sumcheck_rounds.len()={}, expected {}",
                idx,
                step_proof.fold.ccs_proof.sumcheck_rounds.len(),
                want_rounds_total
            )));
        }
        if step_proof.fold.ccs_proof.sumcheck_challenges.len() != want_rounds_total {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: CCS sumcheck_challenges.len()={}, expected {}",
                idx,
                step_proof.fold.ccs_proof.sumcheck_challenges.len(),
                want_rounds_total
            )));
        }
        let ccs_time_rounds = &step_proof.fold.ccs_proof.sumcheck_rounds[..ell_n];
        let ajtai_rounds = &step_proof.fold.ccs_proof.sumcheck_rounds[ell_n..];
        let (ccs_r_time, ccs_time_final, ok_time) =
            verify_sumcheck_rounds_ds(tr, b"ccs/time", step_idx, d_sc, claimed_initial, ccs_time_rounds);
        if !ok_time {
            return Err(PiCcsError::SumcheckError("Π_CCS row rounds invalid".into()));
        }
        if ccs_r_time != step_proof.fold.ccs_proof.sumcheck_challenges[..ell_n] {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS row challenges mismatch",
                idx
            )));
        }

        validate_time_sumcheck_metadata(step_idx, step_proof, &ccs_r_time, &route_r_time, control_stage_enabled)?;

        let expected_k = accumulator.len() + 1;
        if step_proof.fold.ccs_out.len() != expected_k {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS returned {} outputs; expected k={}",
                idx,
                step_proof.fold.ccs_out.len(),
                expected_k
            )));
        }
        if step_proof.fold.ccs_out.is_empty() {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS produced empty ccs_out",
                idx
            )));
        }
        if step_proof.fold.ccs_out[0].r != ccs_r_time {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS output r != ccs row point",
                idx
            )));
        }

        // Bind Π_CCS outputs to the public MCS instance and carried ME inputs.
        //
        // - Commitments must match (Π_CCS does not change commitments).
        // - `X` must match the digit-decomposition of public `x` for the MCS output.
        // - `X` must match the carried ME inputs for subsequent outputs.
        {
            let out0 = &step_proof.fold.ccs_out[0];
            if out0.c != mcs_inst.c {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[0].c does not match mcs_inst.c",
                    idx
                )));
            }
            if out0.m_in != mcs_inst.m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[0].m_in={}, expected {}",
                    idx, out0.m_in, mcs_inst.m_in
                )));
            }
            if out0.X.rows() != D || out0.X.cols() != mcs_inst.m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[0].X has shape {}×{}, expected {}×{}",
                    idx,
                    out0.X.rows(),
                    out0.X.cols(),
                    D,
                    mcs_inst.m_in
                )));
            }

            for (i, inp) in accumulator.iter().enumerate() {
                let out = &step_proof.fold.ccs_out[i + 1];
                if out.c != inp.c {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: Π_CCS output[{}].c does not match accumulator[{}].c",
                        idx,
                        i + 1,
                        i
                    )));
                }
                if out.m_in != inp.m_in {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: Π_CCS output[{}].m_in={}, expected {}",
                        idx,
                        i + 1,
                        out.m_in,
                        inp.m_in
                    )));
                }
                if out.X != inp.X {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: Π_CCS output[{}].X does not match accumulator[{}].X",
                        idx,
                        i + 1,
                        i
                    )));
                }
            }
        }

        // Finish CCS Ajtai rounds alone (continuing transcript state after CCS row rounds).
        let (ajtai_chals, running_sum, ok) =
            verify_sumcheck_rounds_ds(tr, b"ccs/ajtai", step_idx, d_sc, ccs_time_final, ajtai_rounds);
        if !ok {
            return Err(PiCcsError::SumcheckError("Π_CCS Ajtai rounds invalid".into()));
        }

        // Verify stored sumcheck challenges/final match transcript-derived values.
        let mut r_all = ccs_r_time.clone();
        r_all.extend_from_slice(&ajtai_chals);
        if r_all != step_proof.fold.ccs_proof.sumcheck_challenges {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS sumcheck challenges mismatch",
                idx
            )));
        }
        if running_sum != step_proof.fold.ccs_proof.sumcheck_final {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS sumcheck_final mismatch",
                idx
            )));
        }

        // Validate ME input r length (required by RHS assembly if k>1).
        for (i, me) in accumulator.iter().enumerate() {
            if me.r.len() != ell_n {
                return Err(PiCcsError::InvalidInput(format!(
                    "step {}: ME input r length mismatch at accumulator #{}: expected {}, got {}",
                    idx,
                    i,
                    ell_n,
                    me.r.len()
                )));
            }
        }

        if step_proof.fold.ccs_proof.variant != crate::optimized_engine::PiCcsProofVariant::SplitNcV1 {
            return Err(PiCcsError::ProtocolError("unsupported Π_CCS proof variant".into()));
        }

        // FE-only terminal identity.
        let rhs_fe = crate::paper_exact_engine::rhs_terminal_identity_fe_paper_exact(
            &s,
            params,
            &ch,
            &ccs_r_time,
            &ajtai_chals,
            &step_proof.fold.ccs_out,
            accumulator.first().map(|mi| mi.r.as_slice()),
        );
        if running_sum != rhs_fe {
            return Err(PiCcsError::SumcheckError(
                "Π_CCS FE-only terminal identity check failed".into(),
            ));
        }

        // NC-only sumcheck + terminal identity.
        if step_proof.fold.ccs_proof.sumcheck_rounds_nc.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "Π_CCS SplitNcV1 requires non-empty sumcheck_rounds_nc".into(),
            ));
        }
        if let Some(x) = step_proof.fold.ccs_proof.sc_initial_sum_nc {
            if x != K::ZERO {
                return Err(PiCcsError::InvalidInput(
                    "Π_CCS SplitNcV1 requires sc_initial_sum_nc == 0".into(),
                ));
            }
        }
        let want_nc_rounds_total = ell_m
            .checked_add(ell_d)
            .ok_or_else(|| PiCcsError::ProtocolError("ell_m + ell_d overflow".into()))?;
        if step_proof.fold.ccs_proof.sumcheck_rounds_nc.len() != want_nc_rounds_total {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: Π_CCS NC sumcheck_rounds_nc.len()={}, expected {}",
                idx,
                step_proof.fold.ccs_proof.sumcheck_rounds_nc.len(),
                want_nc_rounds_total
            )));
        }
        if step_proof.fold.ccs_proof.sumcheck_challenges_nc.len() != want_nc_rounds_total {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: Π_CCS NC sumcheck_challenges_nc.len()={}, expected {}",
                idx,
                step_proof.fold.ccs_proof.sumcheck_challenges_nc.len(),
                want_nc_rounds_total
            )));
        }

        let (nc_chals, running_sum_nc, ok_nc) = verify_sumcheck_rounds_ds(
            tr,
            b"ccs/nc",
            step_idx,
            d_sc,
            K::ZERO,
            &step_proof.fold.ccs_proof.sumcheck_rounds_nc,
        );
        if !ok_nc {
            return Err(PiCcsError::SumcheckError("Π_CCS NC rounds invalid".into()));
        }

        if nc_chals != step_proof.fold.ccs_proof.sumcheck_challenges_nc {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS NC sumcheck challenges mismatch",
                idx
            )));
        }
        if running_sum_nc != step_proof.fold.ccs_proof.sumcheck_final_nc {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS sumcheck_final_nc mismatch",
                idx
            )));
        }

        let (s_col_prime, alpha_prime_nc) = nc_chals.split_at(ell_m);
        let d_pad = 1usize
            .checked_shl(ell_d as u32)
            .ok_or_else(|| PiCcsError::ProtocolError("2^ell_d overflow".into()))?;
        for (out_idx, out) in step_proof.fold.ccs_out.iter().enumerate() {
            if out.r != ccs_r_time {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[{out_idx}] r != ccs_r_time",
                    idx
                )));
            }
            if out.s_col.as_slice() != s_col_prime {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[{out_idx}] s_col mismatch",
                    idx
                )));
            }
            if out.y_zcol.len() != d_pad {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[{out_idx}] y_zcol.len()={}, expected {}",
                    idx,
                    out.y_zcol.len(),
                    d_pad
                )));
            }
        }

        let rhs_nc = crate::paper_exact_engine::rhs_terminal_identity_nc_paper_exact(
            params,
            &ch,
            s_col_prime,
            alpha_prime_nc,
            &step_proof.fold.ccs_out,
        );
        if running_sum_nc != rhs_nc {
            return Err(PiCcsError::SumcheckError(
                "Π_CCS NC terminal identity check failed".into(),
            ));
        }

        let observed_digest = tr.digest32();
        if observed_digest != step_proof.fold.ccs_proof.header_digest.as_slice() {
            return Err(PiCcsError::ProtocolError("Π_CCS header digest mismatch".into()));
        }
        let expected_digest: [u8; 32] = step_proof
            .fold
            .ccs_proof
            .header_digest
            .as_slice()
            .try_into()
            .map_err(|_| PiCcsError::ProtocolError("Π_CCS header digest must be 32 bytes".into()))?;
        for (out_idx, out) in step_proof.fold.ccs_out.iter().enumerate() {
            if out.fold_digest != expected_digest {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[{out_idx}] fold_digest mismatch",
                    idx
                )));
            }
        }

        let has_stage8_artifacts = !step_proof.fold.openings.is_empty()
            || !step_proof.fold.opening_proofs.is_empty()
            || !step_proof.fold.opening_unification.round_polys.is_empty()
            || !step_proof.fold.joint_opening_lane.groups.is_empty()
            || step_proof.fold.joint_opening_lane.unified_fold.is_some()
            || !step_proof.stage8_fold.is_empty();
        // Full-column commitment replay is intentionally skipped in verifier hot path.
        // Soundness for load-bearing values is enforced via committed named openings
        // (`validate_step_time_opening_proofs` + batched transcript checks).
        // Commitment binding is enforced via transcript-bound batched opening checks below.
        if has_stage8_artifacts || requires_stage8_openings {
            validate_step_time_openings_consistency(step, step_proof, &cpu_bus, &route_r_time)?;
        }

        // Verify mem proofs (shared CPU bus only).
        let prev_step = if idx > 0 {
            Some(&effective_steps[idx - 1])
        } else {
            initial_prev_step
        };
        let prev_step_openings = if idx > 0 {
            Some(proof.steps[idx - 1].fold.openings.as_slice())
        } else {
            None
        };
        let mem_out = crate::memory_sidecar::memory::verify_route_a_memory_step(
            tr,
            &cpu_bus,
            s.m,
            s.t(),
            step,
            prev_step,
            &step_proof.fold.ccs_out[0],
            &route_r_time,
            &r_cycle,
            &final_values,
            &step_proof.batched_time.claimed_sums,
            crate::memory_sidecar::memory::rv64_fullword_width_stage_required_from_proof(
                step,
                &step_proof.batched_time,
            ),
            0,
            &step_proof.mem,
            &step_proof.fold.openings,
            prev_step_openings,
            &shout_pre,
            &twist_pre,
            step_idx,
            poseidon_link_chals.as_ref(),
            poseidon_cont_chals.as_ref(),
        )?;

        let exact_reg_output_binding_active = include_ob
            && ob_cfg
                .map(|cfg| step.mem_insts[cfg.mem_idx].mem_id == neo_memory::riscv::lookups::REG_EXACT_ID.0)
                .unwrap_or(false);
        let expected_consumed = if exact_reg_output_binding_active {
            final_values
                .len()
                .checked_sub(2)
                .ok_or_else(|| PiCcsError::ProtocolError("missing output binding claims".into()))?
        } else if include_ob {
            final_values
                .len()
                .checked_sub(1)
                .ok_or_else(|| PiCcsError::ProtocolError("missing output binding claim".into()))?
        } else {
            final_values.len()
        };
        if mem_out.claim_idx_end != expected_consumed {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: batched claim index mismatch (consumed {}, expected {})",
                idx, mem_out.claim_idx_end, expected_consumed
            )));
        }
        if let Some((r_local, local_final_values, local_anchor)) = poseidon_local_checked.as_ref() {
            crate::memory_sidecar::memory::verify_route_a_poseidon_local_terminals(
                s.t(),
                ell_n,
                r_local,
                local_anchor,
                local_final_values,
                &step_proof.mem,
                poseidon_link_chals.as_ref(),
            )?;
        }

        if include_ob {
            let cfg =
                ob_cfg.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;
            let use_dense_output_sumcheck = cfg.num_bits <= neo_memory::output_check::OUTPUT_SUMCHECK_MAX_NUM_BITS;

            let inc_idx = final_values
                .len()
                .checked_sub(1)
                .ok_or_else(|| PiCcsError::ProtocolError("missing inc_total claim".into()))?;
            if exact_reg_output_binding_active {
                let exact_idx = inc_idx
                    .checked_sub(1)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing reg_exact linkage claim".into()))?;
                if step_proof
                    .batched_time
                    .labels
                    .get(exact_idx)
                    .map(|l| l.as_slice())
                    != Some(crate::output_binding::OB_REG_EXACT_LINKAGE_LABEL)
                {
                    return Err(PiCcsError::ProtocolError(
                        "output binding exact-register linkage claim must be penultimate".into(),
                    ));
                }
                crate::memory_sidecar::memory::verify_rv64_reg_exact_output_linkage_terminal(
                    step,
                    &route_r_time,
                    &r_cycle,
                    &final_values,
                    exact_idx,
                    &step_proof.fold.openings,
                    &mem_out.twist_time_openings,
                    cfg.mem_idx,
                )?;
            }
            if step_proof
                .batched_time
                .labels
                .get(inc_idx)
                .map(|l| l.as_slice())
                != Some(crate::output_binding::OB_INC_TOTAL_LABEL)
            {
                return Err(PiCcsError::ProtocolError("output binding claim not last".into()));
            }

            let inc_total_claim = *step_proof
                .batched_time
                .claimed_sums
                .get(inc_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("missing inc_total claimed_sum".into()))?;
            let inc_total_final = *final_values
                .get(inc_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("missing inc_total final_value".into()))?;

            let twist_open = mem_out
                .twist_time_openings
                .get(cfg.mem_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("missing twist_time_openings for mem_idx".into()))?;
            if use_dense_output_sumcheck {
                let ob_state = ob_state
                    .take()
                    .ok_or_else(|| PiCcsError::ProtocolError("output sumcheck state missing".into()))?;
                let inc_terminal =
                    crate::output_binding::inc_terminal_from_time_openings(twist_open, &ob_state.r_prime)
                        .map_err(|e| PiCcsError::ProtocolError(format!("inc_total terminal mismatch: {e:?}")))?;
                if inc_total_final != inc_terminal {
                    return Err(PiCcsError::ProtocolError("inc_total terminal mismatch".into()));
                }

                let mem_inst = step
                    .mem_insts
                    .get(cfg.mem_idx)
                    .ok_or_else(|| PiCcsError::InvalidInput("output binding mem_idx out of range".into()))?;
                let expected_k = 1usize
                    .checked_shl(cfg.num_bits as u32)
                    .ok_or_else(|| PiCcsError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
                if mem_inst.k != expected_k {
                    return Err(PiCcsError::InvalidInput(format!(
                        "output binding: cfg.num_bits implies k={}, but mem_inst.k={}",
                        expected_k, mem_inst.k
                    )));
                }
                let ell_addr = mem_inst.twist_layout().lanes[0].ell_addr;
                if ell_addr != cfg.num_bits {
                    return Err(PiCcsError::InvalidInput(format!(
                        "output binding: cfg.num_bits={}, but twist_layout.ell_addr={}",
                        cfg.num_bits, ell_addr
                    )));
                }
                let val_init =
                    crate::output_binding::val_init_from_mem_init(&mem_inst.init, mem_inst.k, &ob_state.r_prime)
                        .map_err(|e| PiCcsError::ProtocolError(format!("MemInit eval failed: {e:?}")))?;
                let val_final_at_r_prime = val_init + inc_total_claim;
                let expected_out =
                    ob_state.eq_eval * ob_state.io_mask_eval * (val_final_at_r_prime - ob_state.val_io_eval);
                if expected_out != ob_state.output_final {
                    return Err(PiCcsError::ProtocolError("output binding final check failed".into()));
                }
            } else {
                let addr_weights = ob_sparse_addr_weights
                    .take()
                    .ok_or_else(|| PiCcsError::ProtocolError("output sparse addr/weight set missing".into()))?;
                let val_offset = ob_sparse_val_offset
                    .take()
                    .ok_or_else(|| PiCcsError::ProtocolError("output sparse value offset missing".into()))?;
                let inc_terminal =
                    crate::output_binding::weighted_inc_terminal_from_time_openings(twist_open, &addr_weights)
                        .map_err(|e| PiCcsError::ProtocolError(format!("inc_total terminal mismatch: {e:?}")))?;
                if inc_total_final != inc_terminal {
                    return Err(PiCcsError::ProtocolError("inc_total terminal mismatch".into()));
                }
                if val_offset + inc_total_claim != K::ZERO {
                    return Err(PiCcsError::ProtocolError("output sparse final check failed".into()));
                }
            }
        }

        validate_me_batch_invariants(&step_proof.fold.ccs_out, "verify step ccs outputs")?;
        verify_rlc_dec_lane(
            RlcLane::Main,
            tr,
            params,
            &s,
            &ring,
            ell_d,
            mixers,
            step_idx,
            &step_proof.fold.ccs_out,
            &step_proof.fold.rlc_rhos,
            &step_proof.fold.rlc_parent,
            &step_proof.fold.dec_children,
        )?;

        accumulator = step_proof.fold.dec_children.clone();

        // Phase 2: Verify folding lanes for ME claims evaluated at r_val.
        if step_proof.mem.val_me_claims.is_empty() {
            if !step_proof.val_fold.is_empty() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: unexpected val_fold proof(s) (no r_val ME claims)",
                    idx
                )));
            }
        } else {
            tr.append_message(b"fold/val_lane_start", &(step_idx as u64).to_le_bytes());
            let expected = 1usize + usize::from(has_prev);
            if step_proof.mem.val_me_claims.len() != expected {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: val_me_claims count mismatch in shared-bus mode (have {}, expected {})",
                    idx,
                    step_proof.mem.val_me_claims.len(),
                    expected
                )));
            }
            if step_proof.val_fold.len() != expected {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: val_fold count mismatch in shared-bus mode (have {}, expected {})",
                    idx,
                    step_proof.val_fold.len(),
                    expected
                )));
            }

            for (claim_idx, (me, proof)) in step_proof
                .mem
                .val_me_claims
                .iter()
                .zip(step_proof.val_fold.iter())
                .enumerate()
            {
                let ctx = match claim_idx {
                    0 => "cpu",
                    1 => "cpu_prev",
                    _ => {
                        return Err(PiCcsError::ProtocolError(
                            "unexpected extra r_val ME claim in shared-bus mode".into(),
                        ));
                    }
                };
                tr.append_message(b"fold/val_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                tr.append_message(b"fold/val_lane_claim_ctx", ctx.as_bytes());
                verify_rlc_dec_lane(
                    RlcLane::Val,
                    tr,
                    params,
                    &s,
                    &ring,
                    ell_d,
                    mixers,
                    step_idx,
                    core::slice::from_ref(me),
                    &proof.rlc_rhos,
                    &proof.rlc_parent,
                    &proof.dec_children,
                )
                .map_err(|e| {
                    PiCcsError::ProtocolError(format!(
                        "step {} val_fold(shared) claim {} ({ctx}) verify failed: {e:?}",
                        idx, claim_idx
                    ))
                })?;
                val_lane_obligations.extend_from_slice(&proof.dec_children);
            }
        }

        if step_proof.mem.wb_me_claims.is_empty() {
            if !step_proof.wb_fold.is_empty() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: unexpected wb_fold proof(s) (no WB ME claims)",
                    idx
                )));
            }
        } else {
            if step_proof.wb_fold.len() != step_proof.mem.wb_me_claims.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: wb_fold count mismatch (have {}, expected {})",
                    idx,
                    step_proof.wb_fold.len(),
                    step_proof.mem.wb_me_claims.len()
                )));
            }
            tr.append_message(b"fold/wb_lane_start", &(step_idx as u64).to_le_bytes());
            for (claim_idx, (me, proof)) in step_proof
                .mem
                .wb_me_claims
                .iter()
                .zip(step_proof.wb_fold.iter())
                .enumerate()
            {
                tr.append_message(b"fold/wb_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                verify_rlc_dec_lane(
                    RlcLane::Val,
                    tr,
                    params,
                    &s,
                    &ring,
                    ell_d,
                    mixers,
                    step_idx,
                    core::slice::from_ref(me),
                    &proof.rlc_rhos,
                    &proof.rlc_parent,
                    &proof.dec_children,
                )
                .map_err(|e| {
                    PiCcsError::ProtocolError(format!("step {} wb_fold claim {} verify failed: {e:?}", idx, claim_idx))
                })?;
                val_lane_obligations.extend_from_slice(&proof.dec_children);
            }
        }

        if step_proof.mem.wp_me_claims.is_empty() {
            if !step_proof.wp_fold.is_empty() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: unexpected wp_fold proof(s) (no WP ME claims)",
                    idx
                )));
            }
        } else {
            if step_proof.wp_fold.len() != step_proof.mem.wp_me_claims.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: wp_fold count mismatch (have {}, expected {})",
                    idx,
                    step_proof.wp_fold.len(),
                    step_proof.mem.wp_me_claims.len()
                )));
            }
            tr.append_message(b"fold/wp_lane_start", &(step_idx as u64).to_le_bytes());
            for (claim_idx, (me, proof)) in step_proof
                .mem
                .wp_me_claims
                .iter()
                .zip(step_proof.wp_fold.iter())
                .enumerate()
            {
                tr.append_message(b"fold/wp_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                verify_rlc_dec_lane(
                    RlcLane::Val,
                    tr,
                    params,
                    &s,
                    &ring,
                    ell_d,
                    mixers,
                    step_idx,
                    core::slice::from_ref(me),
                    &proof.rlc_rhos,
                    &proof.rlc_parent,
                    &proof.dec_children,
                )
                .map_err(|e| {
                    PiCcsError::ProtocolError(format!("step {} wp_fold claim {} verify failed: {e:?}", idx, claim_idx))
                })?;
                val_lane_obligations.extend_from_slice(&proof.dec_children);
            }
        }

        verify_poseidon_fold_lanes(
            tr,
            params,
            &s,
            &ring,
            ell_d,
            mixers,
            step_idx,
            idx,
            step_proof,
            &mut val_lane_obligations,
        )?;

        if has_stage8_artifacts || requires_stage8_openings {
            validate_step_time_opening_batches_with_transcript(tr, params, step_idx, step, step_proof, &cpu_bus)?;
        }
        let stage8_plan = crate::time_opening::joint_lane::build_stage8_fold_lane_plan(
            &step_proof.fold.joint_opening_lane,
            &step_proof.fold.opening_unification,
            step_proof.fold.time_t,
        )?;
        let expected_stage8_proofs = if stage8_plan.is_some() { 1usize } else { 0usize };
        if step_proof.stage8_fold.len() != expected_stage8_proofs {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: expected stage8_fold proofs to match Stage-8 lane plan (proofs={}, expected={})",
                idx,
                step_proof.stage8_fold.len(),
                expected_stage8_proofs
            )));
        }
        if let Some(plan) = stage8_plan {
            let stage8_params = stage8_time_decomp_params(params)?;
            tr.append_message(b"fold/stage8_lane_start", &(step_idx as u64).to_le_bytes());
            tr.append_message(b"fold/stage8_lane_group_idx", &0u64.to_le_bytes());
            let proof_stage8 = step_proof
                .stage8_fold
                .first()
                .ok_or_else(|| PiCcsError::ProtocolError(format!("step {}: missing Stage-8 fold proof", idx)))?;
            verify_rlc_dec_lane(
                RlcLane::Val,
                tr,
                &stage8_params,
                &plan.ccs,
                &ring,
                ell_d,
                mixers,
                step_idx,
                plan.claims.as_slice(),
                &proof_stage8.rlc_rhos,
                &proof_stage8.rlc_parent,
                &proof_stage8.dec_children,
            )
            .map_err(|e| PiCcsError::ProtocolError(format!("step {} stage8_fold verify failed: {e:?}", idx)))?;
            val_lane_obligations.extend_from_slice(&proof_stage8.dec_children);
        }

        tr.append_message(b"fold/step_done", &(step_idx as u64).to_le_bytes());
    }

    Ok(ShardFoldOutputs {
        obligations: ShardObligations {
            main: accumulator,
            val: val_lane_obligations,
        },
    })
}

pub fn fold_shard_verify<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_mixed_ccs_batched(mode, tr, params, s_me, steps, 0, acc_init, proof, mixers, None)
}

/// Same as `fold_shard_verify`, but offsets the per-step transcript index by `step_idx_offset`.
pub fn fold_shard_verify_with_step_offset<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    step_idx_offset: usize,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_mixed_ccs_batched(
        mode,
        tr,
        params,
        s_me,
        steps,
        step_idx_offset,
        acc_init,
        proof,
        mixers,
        None,
    )
}

pub fn fold_shard_verify_with_step_linking<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    step_linking: &StepLinkingConfig,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify(mode, tr, params, s_me, steps, acc_init, proof, mixers)
}

pub fn fold_shard_verify_with_output_binding<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_mixed_ccs_batched_with_output_binding(
        mode, tr, params, s_me, steps, 0, acc_init, proof, mixers, ob_cfg, None,
    )
}

pub(crate) fn fold_shard_verify_with_context<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    prover_ctx: &ShardProverContext,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_mixed_ccs_batched(
        mode,
        tr,
        params,
        s_me,
        steps,
        0,
        acc_init,
        proof,
        mixers,
        Some(prover_ctx),
    )
}

pub(crate) fn fold_shard_verify_with_step_linking_with_context<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    step_linking: &StepLinkingConfig,
    prover_ctx: &ShardProverContext,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_with_context(mode, tr, params, s_me, steps, acc_init, proof, mixers, prover_ctx)
}

pub(crate) fn fold_shard_verify_with_output_binding_with_context<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    prover_ctx: &ShardProverContext,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_mixed_ccs_batched_with_output_binding(
        mode,
        tr,
        params,
        s_me,
        steps,
        0,
        acc_init,
        proof,
        mixers,
        ob_cfg,
        Some(prover_ctx),
    )
}

pub(crate) fn fold_shard_verify_with_output_binding_and_step_linking_with_context<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    step_linking: &StepLinkingConfig,
    prover_ctx: &ShardProverContext,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_with_output_binding_with_context(
        mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg, prover_ctx,
    )
}

pub fn fold_shard_verify_with_output_binding_and_step_linking<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    step_linking: &StepLinkingConfig,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_with_output_binding(mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg)
}

pub fn fold_shard_verify_and_finalize<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    let outputs = fold_shard_verify(mode, tr, params, s_me, steps, acc_init, proof, mixers)?;
    let report = finalizer.finalize(&outputs.obligations)?;
    outputs
        .obligations
        .require_all_finalized(report.did_finalize_main, report.did_finalize_val)?;
    Ok(())
}

pub fn fold_shard_verify_and_finalize_with_step_linking<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    step_linking: &StepLinkingConfig,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_and_finalize(mode, tr, params, s_me, steps, acc_init, proof, mixers, finalizer)
}

pub fn fold_shard_verify_and_finalize_with_output_binding<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    let outputs =
        fold_shard_verify_with_output_binding(mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg)?;
    let report = finalizer.finalize(&outputs.obligations)?;
    outputs
        .obligations
        .require_all_finalized(report.did_finalize_main, report.did_finalize_val)?;
    Ok(())
}

pub fn fold_shard_verify_and_finalize_with_output_binding_and_step_linking<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    step_linking: &StepLinkingConfig,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_and_finalize_with_output_binding(
        mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg, finalizer,
    )
}
