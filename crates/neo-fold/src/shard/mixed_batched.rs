use super::*;

#[inline]
fn is_ccs_only_witness_step(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    step.lut_instances.is_empty() && step.mem_instances.is_empty()
}

#[inline]
fn is_ccs_only_instance_step(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    step.lut_insts.is_empty() && step.mem_insts.is_empty()
}

fn contiguous_segment_end_witness(steps: &[StepWitnessBundle<Cmt, F, K>], start: usize) -> usize {
    let is_ccs_only = is_ccs_only_witness_step(&steps[start]);
    let mut end = start + 1;
    while end < steps.len() && is_ccs_only_witness_step(&steps[end]) == is_ccs_only {
        end += 1;
    }
    end
}

fn ccs_only_segment_batch_supported_witness(
    s_me: &CcsStructure<F>,
    segment: &[StepWitnessBundle<Cmt, F, K>],
) -> Result<bool, PiCcsError> {
    let (_s, cpu_bus) = crate::memory_sidecar::cpu_bus::prepare_ccs_for_shared_cpu_bus_steps(s_me, segment)?;
    Ok(cpu_bus.bus_cols == 0)
}

fn ccs_only_segment_batch_supported_instance(
    s_me: &CcsStructure<F>,
    segment: &[StepInstanceBundle<Cmt, F, K>],
) -> Result<bool, PiCcsError> {
    let (_s, cpu_bus) = crate::memory_sidecar::cpu_bus::prepare_ccs_for_shared_cpu_bus_steps(s_me, segment)?;
    Ok(cpu_bus.bus_cols == 0)
}

#[inline]
fn ccs_only_batch_size_for_mode(mode: &FoldingMode, params: &NeoParams, acc_len: usize, step_count: usize) -> usize {
    match mode {
        FoldingMode::Optimized => ccs_only_batched::auto_mcs_batch_size(params, acc_len, step_count),
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck(_) => ccs_only_batched::auto_mcs_batch_size(params, acc_len, step_count),
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => 1,
    }
}

pub(crate) fn fold_shard_prove_mixed_ccs_batched<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    prover_ctx: Option<&ShardProverContext>,
    compute_backend: &ProverComputeBackend,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, _final_main_wits, _val_lane_wits) = fold_shard_prove_mixed_ccs_batched_with_witnesses_internal(
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
        prover_ctx,
        compute_backend,
    )?;
    Ok(proof)
}

pub(crate) fn fold_shard_prove_mixed_ccs_batched_with_output_binding<L, MR, MB>(
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
    prover_ctx: Option<&ShardProverContext>,
    compute_backend: &ProverComputeBackend,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, _final_main_wits, _val_lane_wits) = fold_shard_prove_mixed_ccs_batched_with_witnesses_internal(
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
        Some((ob_cfg, final_memory_state)),
        prover_ctx,
        compute_backend,
    )?;
    Ok(proof)
}

pub(crate) fn fold_shard_prove_mixed_ccs_batched_with_witnesses<L, MR, MB>(
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
    prover_ctx: Option<&ShardProverContext>,
    compute_backend: &ProverComputeBackend,
) -> Result<(ShardProof, Vec<Mat<F>>, Vec<Mat<F>>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_prove_mixed_ccs_batched_with_witnesses_internal(
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
        prover_ctx,
        compute_backend,
    )
}

fn fold_shard_prove_mixed_ccs_batched_with_witnesses_internal<L, MR, MB>(
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
    ob: Option<(&crate::output_binding::OutputBindingConfig, &[F])>,
    prover_ctx: Option<&ShardProverContext>,
    compute_backend: &ProverComputeBackend,
) -> Result<(ShardProof, Vec<Mat<F>>, Vec<Mat<F>>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if acc_init.len() != acc_wit_init.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "acc_init.len()={} != acc_wit_init.len()={}",
            acc_init.len(),
            acc_wit_init.len()
        )));
    }
    let backend_ctx = neo_reductions::accelerator::BackendContext::new(compute_backend)?;

    let mut accumulator = acc_init.to_vec();
    let mut accumulator_wit = acc_wit_init.to_vec();
    let mut val_lane_wits: Vec<Mat<F>> = Vec::new();
    let mut merged_steps: Vec<StepProof> = Vec::new();
    let mut output_proof: Option<neo_memory::output_check::OutputBindingProof> = None;
    let mut segment_meta: Vec<ShardSegmentMeta> = Vec::new();

    let mut cursor = 0usize;
    while cursor < steps.len() {
        let end = contiguous_segment_end_witness(steps, cursor);
        let segment = &steps[cursor..end];
        let segment_step_idx_offset = step_idx_offset
            .checked_add(cursor)
            .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;
        let is_ccs_only = is_ccs_only_witness_step(&steps[cursor]);
        let is_final_segment = end == steps.len();
        let segment_ob = if is_final_segment { ob } else { None };

        if is_ccs_only {
            if segment_ob.is_some() {
                return Err(PiCcsError::InvalidInput(
                    "output binding requires final segment to include Route-A sidecars".into(),
                ));
            }
            if !ccs_only_segment_batch_supported_witness(s_me, segment)? {
                return Err(PiCcsError::InvalidInput(format!(
                    "ccs-only segment at step {} is incompatible with ccs-only batched path (shared-bus columns present)",
                    segment_step_idx_offset
                )));
            }

            let batch_size = ccs_only_batch_size_for_mode(&mode, params, accumulator.len(), segment.len());
            let (segment_proof, next_acc, next_wits) =
                ccs_only_batched::fold_shard_prove_ccs_only_batched_with_outputs_and_offset_and_context(
                    mode.clone(),
                    tr,
                    params,
                    s_me,
                    segment,
                    &accumulator,
                    &accumulator_wit,
                    l,
                    mixers,
                    batch_size,
                    segment_step_idx_offset,
                    &backend_ctx,
                )?;
            accumulator = next_acc;
            accumulator_wit = next_wits;
            if segment_proof.output_proof.is_some() {
                return Err(PiCcsError::ProtocolError(
                    "ccs-only batched segment unexpectedly produced output binding proof".into(),
                ));
            }
            segment_meta.push(ShardSegmentMeta {
                kind: ShardSegmentKind::CcsOnly,
                public_steps: segment.len(),
                proof_steps: segment_proof.steps.len(),
            });
            merged_steps.extend(segment_proof.steps);
            cursor = end;
            continue;
        }

        let (mut segment_proof, next_main_wits, segment_val_lane_wits) =
            fold_shard_prove_route_a_segment_with_witnesses(
                mode.clone(),
                tr,
                params,
                s_me,
                segment,
                segment_step_idx_offset,
                &accumulator,
                &accumulator_wit,
                l,
                mixers,
                segment_ob,
                prover_ctx,
                compute_backend,
                Some(&backend_ctx),
            )?;
        let route_meta_entries = segment_proof.segment_meta.clone().ok_or_else(|| {
            PiCcsError::ProtocolError(
                "route-a segment proof missing segment_meta (legacy unsegmented route-a proofs are not supported)"
                    .into(),
            )
        })?;
        let (meta_public_steps, meta_proof_steps) = route_meta_entries
            .iter()
            .fold((0usize, 0usize), |(a, b), e| (a + e.public_steps, b + e.proof_steps));
        if meta_public_steps != segment.len() || meta_proof_steps != segment_proof.steps.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "route-a segment metadata does not match produced proof (meta_public={}, segment_public={}, meta_proof={}, proof_steps={})",
                meta_public_steps,
                segment.len(),
                meta_proof_steps,
                segment_proof.steps.len()
            )));
        }
        if route_meta_entries
            .iter()
            .any(|e| e.kind != ShardSegmentKind::RouteA)
        {
            return Err(PiCcsError::ProtocolError(
                "route-a segment metadata contains non-RouteA entry".into(),
            ));
        }
        if let Some(ob_pf) = segment_proof.output_proof.take() {
            if output_proof.is_some() {
                return Err(PiCcsError::ProtocolError(
                    "multiple output binding proofs produced across segments".into(),
                ));
            }
            output_proof = Some(ob_pf);
        }
        let segment_outputs = segment_proof.compute_fold_outputs(&accumulator);
        accumulator = segment_outputs.obligations.main;
        accumulator_wit = next_main_wits;
        val_lane_wits.extend(segment_val_lane_wits);
        segment_meta.extend(route_meta_entries);
        merged_steps.extend(segment_proof.steps);
        cursor = end;
    }

    if ob.is_some() && output_proof.is_none() {
        return Err(PiCcsError::ProtocolError(
            "output binding requested but no segment produced output binding proof".into(),
        ));
    }
    if ob.is_none() && output_proof.is_some() {
        return Err(PiCcsError::ProtocolError(
            "unexpected output binding proof in mixed batched proving".into(),
        ));
    }

    Ok((
        ShardProof {
            steps: merged_steps,
            output_proof,
            riscv_profile: None,
            riscv_memory_layout: None,
            segment_meta: Some(segment_meta),
        },
        accumulator_wit,
        val_lane_wits,
    ))
}

pub(crate) fn fold_shard_verify_mixed_ccs_batched<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    step_idx_offset: usize,
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    prover_ctx: Option<&ShardProverContext>,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_mixed_ccs_batched_internal(
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
        prover_ctx,
        &ProverComputeBackend::Cpu,
    )
}

pub(crate) fn fold_shard_verify_mixed_ccs_batched_with_output_binding<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    step_idx_offset: usize,
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    prover_ctx: Option<&ShardProverContext>,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_mixed_ccs_batched_internal(
        mode,
        tr,
        params,
        s_me,
        steps,
        step_idx_offset,
        acc_init,
        proof,
        mixers,
        Some(ob_cfg),
        prover_ctx,
        &ProverComputeBackend::Cpu,
    )
}

pub(crate) fn fold_shard_verify_mixed_ccs_batched_with_backend<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    step_idx_offset: usize,
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    prover_ctx: Option<&ShardProverContext>,
    compute_backend: &ProverComputeBackend,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_mixed_ccs_batched_internal(
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
        prover_ctx,
        compute_backend,
    )
}

pub(crate) fn fold_shard_verify_mixed_ccs_batched_with_output_binding_and_backend<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    step_idx_offset: usize,
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    prover_ctx: Option<&ShardProverContext>,
    compute_backend: &ProverComputeBackend,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_mixed_ccs_batched_internal(
        mode,
        tr,
        params,
        s_me,
        steps,
        step_idx_offset,
        acc_init,
        proof,
        mixers,
        Some(ob_cfg),
        prover_ctx,
        compute_backend,
    )
}

fn fold_shard_verify_mixed_ccs_batched_internal<MR, MB>(
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
    compute_backend: &ProverComputeBackend,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if ob_cfg.is_none() && proof.output_proof.is_some() {
        return Err(PiCcsError::InvalidInput(
            "mixed batched verification requires proof.output_proof = None".into(),
        ));
    }
    if ob_cfg.is_some() && proof.output_proof.is_none() {
        return Err(PiCcsError::InvalidInput(
            "verifier supplied OutputBindingConfig, but shard proof has no output binding".into(),
        ));
    }

    let mut accumulator = acc_init.to_vec();
    let mut val_lane_obligations: Vec<CeClaim<Cmt, F, K>> = Vec::new();
    let segment_meta = proof.segment_meta.as_deref().ok_or_else(|| {
        PiCcsError::InvalidInput(
            "mixed batched verification requires segment_meta (legacy unsegmented proofs are not supported)".into(),
        )
    })?;
    let mut step_cursor = 0usize;
    let mut proof_cursor = 0usize;
    let mut prev_route_a_step_ctx: Option<&StepInstanceBundle<Cmt, F, K>> = None;
    let backend_ctx = neo_reductions::accelerator::BackendContext::new(compute_backend)?;
    for (meta_idx, meta_entry) in segment_meta.iter().enumerate() {
        if meta_entry.public_steps == 0 || meta_entry.proof_steps == 0 {
            return Err(PiCcsError::InvalidInput(format!(
                "segment_meta entry {} must have public_steps>=1 and proof_steps>=1",
                meta_idx
            )));
        }
        if step_cursor + meta_entry.public_steps > steps.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "segment_meta entry {} overflows public steps (cursor={}, public_steps={}, total={})",
                meta_idx,
                step_cursor,
                meta_entry.public_steps,
                steps.len()
            )));
        }
        if proof_cursor + meta_entry.proof_steps > proof.steps.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "proof too short for segment_meta entry {} (need {}, remaining {})",
                meta_idx,
                meta_entry.proof_steps,
                proof.steps.len().saturating_sub(proof_cursor)
            )));
        }

        let segment = &steps[step_cursor..step_cursor + meta_entry.public_steps];
        let global_step_offset = step_idx_offset
            .checked_add(step_cursor)
            .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;
        let is_ccs_only = matches!(meta_entry.kind, ShardSegmentKind::CcsOnly);
        let is_final_segment = meta_idx + 1 == segment_meta.len();
        let segment_ob_cfg = if is_final_segment { ob_cfg } else { None };

        let segment_proof_steps = meta_entry.proof_steps;

        if is_ccs_only {
            if segment.iter().any(|step| !is_ccs_only_instance_step(step)) {
                return Err(PiCcsError::InvalidInput(format!(
                    "segment_meta entry {} marked CcsOnly but contains Route-A statement step(s)",
                    meta_idx
                )));
            }
            if segment_ob_cfg.is_some() {
                return Err(PiCcsError::InvalidInput(
                    "output binding requires final segment to include Route-A sidecars".into(),
                ));
            }
            if !ccs_only_segment_batch_supported_instance(s_me, segment)? {
                return Err(PiCcsError::InvalidInput(format!(
                    "ccs-only segment at step {} is incompatible with ccs-only batched path (shared-bus columns present)",
                    global_step_offset
                )));
            }
            let batch_size = ccs_only_batch_size_for_mode(&mode, params, accumulator.len(), segment.len());
            let expected_steps = (segment.len() + batch_size - 1) / batch_size;
            if segment_proof_steps != expected_steps {
                return Err(PiCcsError::InvalidInput(format!(
                    "ccs-only segment proof step count mismatch at step {} (proof_steps={}, expected={})",
                    global_step_offset, segment_proof_steps, expected_steps
                )));
            }
        } else {
            if segment.iter().any(is_ccs_only_instance_step) {
                return Err(PiCcsError::InvalidInput(format!(
                    "segment_meta entry {} marked RouteA but contains CCS-only statement step(s)",
                    meta_idx
                )));
            }
        }

        let segment_proof = ShardProof {
            steps: proof.steps[proof_cursor..proof_cursor + segment_proof_steps].to_vec(),
            output_proof: if segment_ob_cfg.is_some() {
                proof.output_proof.clone()
            } else {
                None
            },
            riscv_profile: None,
            riscv_memory_layout: None,
            segment_meta: if is_ccs_only {
                None
            } else {
                Some(vec![meta_entry.clone()])
            },
        };
        proof_cursor += segment_proof_steps;

        let segment_outputs = if is_ccs_only {
            let batch_size = ccs_only_batch_size_for_mode(&mode, params, accumulator.len(), segment.len());
            ccs_only_batched::fold_shard_verify_ccs_only_batched_with_offset_and_context(
                mode.clone(),
                tr,
                params,
                s_me,
                segment,
                &accumulator,
                &segment_proof,
                mixers,
                batch_size,
                global_step_offset,
                &backend_ctx,
            )?
        } else {
            fold_shard_verify_route_a_segment(
                mode.clone(),
                tr,
                params,
                s_me,
                segment,
                global_step_offset,
                &accumulator,
                &segment_proof,
                mixers,
                segment_ob_cfg,
                prover_ctx,
                compute_backend,
                Some(&backend_ctx),
                prev_route_a_step_ctx,
            )?
        };

        accumulator = segment_outputs.obligations.main;
        val_lane_obligations.extend(segment_outputs.obligations.val);
        if is_ccs_only {
            prev_route_a_step_ctx = None;
        } else {
            prev_route_a_step_ctx = segment.last();
        }
        step_cursor += meta_entry.public_steps;
    }

    if step_cursor != steps.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "segment_meta consumed {} public steps, expected {}",
            step_cursor,
            steps.len()
        )));
    }
    if proof_cursor != proof.steps.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "proof has {} extra step(s) after mixed-segment verification",
            proof.steps.len() - proof_cursor
        )));
    }

    Ok(ShardFoldOutputs {
        obligations: ShardObligations {
            main: accumulator,
            val: val_lane_obligations,
        },
    })
}
