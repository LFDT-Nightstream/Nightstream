use super::*;

const ROUTE_A_SEGMENT_BATCH_TARGET_STEPS: usize = 8;

#[inline]
fn is_ccs_only_witness_step(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    step.lut_instances.is_empty() && step.mem_instances.is_empty()
}

#[inline]
fn is_ccs_only_instance_step(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    step.lut_insts.is_empty() && step.mem_insts.is_empty()
}

fn ensure_route_a_segment_witness(
    steps: &[StepWitnessBundle<Cmt, F, K>],
    step_idx_offset: usize,
) -> Result<(), PiCcsError> {
    for (local_idx, step) in steps.iter().enumerate() {
        if is_ccs_only_witness_step(step) {
            let step_idx = step_idx_offset
                .checked_add(local_idx)
                .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;
            return Err(PiCcsError::InvalidInput(format!(
                "route-a segment contains ccs-only step (step_idx={step_idx})"
            )));
        }
    }
    Ok(())
}

fn ensure_route_a_segment_instance(
    steps: &[StepInstanceBundle<Cmt, F, K>],
    step_idx_offset: usize,
) -> Result<(), PiCcsError> {
    for (local_idx, step) in steps.iter().enumerate() {
        if is_ccs_only_instance_step(step) {
            let step_idx = step_idx_offset
                .checked_add(local_idx)
                .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;
            return Err(PiCcsError::InvalidInput(format!(
                "route-a segment contains ccs-only statement step (step_idx={step_idx})"
            )));
        }
    }
    Ok(())
}

#[inline]
fn auto_route_a_segment_batch_size(public_steps: usize) -> usize {
    core::cmp::max(1, core::cmp::min(ROUTE_A_SEGMENT_BATCH_TARGET_STEPS, public_steps))
}

fn compress_route_a_chunk_steps(chunk_steps: Vec<StepProof>) -> Result<StepProof, PiCcsError> {
    if chunk_steps.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "route-a chunk produced zero proof steps".into(),
        ));
    }
    let mut chunk_steps = chunk_steps;
    let mut terminal = chunk_steps
        .pop()
        .ok_or_else(|| PiCcsError::ProtocolError("route-a chunk missing terminal step proof".into()))?;
    if terminal.compressed_substeps.is_some() {
        return Err(PiCcsError::ProtocolError(
            "route-a chunk terminal proof must not already be compressed".into(),
        ));
    }
    if chunk_steps
        .iter()
        .any(|step| step.compressed_substeps.is_some())
    {
        return Err(PiCcsError::ProtocolError(
            "route-a chunk substeps must not be pre-compressed".into(),
        ));
    }
    // Store only prefix steps in `compressed_substeps`; the container itself is
    // the terminal step. This avoids storing the terminal step twice.
    if !chunk_steps.is_empty() {
        terminal.compressed_substeps = Some(chunk_steps);
    }
    Ok(terminal)
}

fn expand_route_a_chunk_steps(
    container: &StepProof,
    public_steps: usize,
    meta_idx: usize,
) -> Result<Vec<StepProof>, PiCcsError> {
    if public_steps == 0 {
        return Err(PiCcsError::InvalidInput(format!(
            "route-a segment metadata entry {} must have public_steps>=1",
            meta_idx
        )));
    }

    let mut out: Vec<StepProof> = Vec::with_capacity(public_steps);
    if let Some(prefix) = container.compressed_substeps.as_ref() {
        if prefix.iter().any(|step| step.compressed_substeps.is_some()) {
            return Err(PiCcsError::InvalidInput(format!(
                "route-a compressed chunk at entry {} must not contain nested compressed_substeps",
                meta_idx
            )));
        }
        out.extend(prefix.iter().cloned());
    } else if public_steps > 1 {
        return Err(PiCcsError::InvalidInput(format!(
            "route-a segment metadata entry {} expects {} public steps but container has no compressed_substeps",
            meta_idx, public_steps
        )));
    }

    let mut terminal = container.clone();
    terminal.compressed_substeps = None;
    out.push(terminal);

    if out.len() != public_steps {
        return Err(PiCcsError::InvalidInput(format!(
            "route-a compressed chunk length mismatch at entry {} (materialized_steps={}, public_steps={})",
            meta_idx,
            out.len(),
            public_steps
        )));
    }
    Ok(out)
}

pub(crate) fn fold_shard_prove_route_a_segment_with_witnesses<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    step_idx_offset: usize,
    acc_init: &[CeClaim<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    ob: Option<(&crate::output_binding::OutputBindingConfig, &[F])>,
    prover_ctx: Option<&ShardProverContext>,
) -> Result<(ShardProof, Vec<Mat<F>>, Vec<Mat<F>>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    ensure_route_a_segment_witness(steps, step_idx_offset)?;
    if steps.is_empty() {
        if ob.is_some() {
            return Err(PiCcsError::InvalidInput("output binding requires >= 1 step".into()));
        }
        return Ok((
            ShardProof {
                steps: Vec::new(),
                output_proof: None,
                segment_meta: None,
            },
            acc_wit_init.to_vec(),
            Vec::new(),
        ));
    }

    let batch_size = auto_route_a_segment_batch_size(steps.len());
    let mut accumulator = acc_init.to_vec();
    let mut accumulator_wit = acc_wit_init.to_vec();
    let mut merged_steps: Vec<StepProof> = Vec::new();
    let mut merged_val_lane_wits: Vec<Mat<F>> = Vec::new();
    let mut merged_output_proof: Option<neo_memory::output_check::OutputBindingProof> = None;
    let mut prev_step_ctx: Option<&StepWitnessBundle<Cmt, F, K>> = None;
    let mut prev_twist_decoded: Option<Vec<crate::memory_sidecar::memory::TwistDecodedColsSparse>> = None;
    let mut poseidon_carry = crate::memory_sidecar::memory::PoseidonSidecarCarryState::new();
    let mut route_chunk_meta: Vec<ShardSegmentMeta> = Vec::new();

    let mut cursor = 0usize;
    while cursor < steps.len() {
        let end = core::cmp::min(cursor + batch_size, steps.len());
        let chunk = &steps[cursor..end];
        let chunk_step_offset = step_idx_offset
            .checked_add(cursor)
            .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;
        let chunk_ob = if end == steps.len() { ob } else { None };

        let (chunk_proof, next_main_wits, mut chunk_val_lane_wits, next_prev_twist_decoded, next_poseidon_carry) =
            fold_shard_prove_impl(
                true,
                mode.clone(),
                tr,
                params,
                s_me,
                chunk,
                chunk_step_offset,
                &accumulator,
                &accumulator_wit,
                l,
                mixers,
                chunk_ob,
                prover_ctx,
                None,
                prev_step_ctx,
                prev_twist_decoded.take(),
                Some(poseidon_carry),
            )?;
        let next_accumulator = chunk_proof.compute_final_main_children(&accumulator);
        let ShardProof {
            steps: chunk_steps,
            output_proof: chunk_output_proof,
            segment_meta: _,
        } = chunk_proof;
        if chunk_steps.len() != chunk.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "route-a chunk proof step count mismatch (proof_steps={}, public_steps={})",
                chunk_steps.len(),
                chunk.len()
            )));
        }
        let compressed_step = compress_route_a_chunk_steps(chunk_steps)?;

        if chunk_ob.is_none() && chunk_output_proof.is_some() {
            return Err(PiCcsError::ProtocolError(
                "route-a segment chunk unexpectedly produced output binding proof".into(),
            ));
        }
        if let Some(ob_pf) = chunk_output_proof {
            if merged_output_proof.is_some() {
                return Err(PiCcsError::ProtocolError(
                    "route-a segment produced multiple output binding proofs".into(),
                ));
            }
            merged_output_proof = Some(ob_pf);
        }

        accumulator = next_accumulator;
        accumulator_wit = next_main_wits;
        route_chunk_meta.push(ShardSegmentMeta {
            kind: ShardSegmentKind::RouteA,
            public_steps: chunk.len(),
            proof_steps: 1,
        });
        merged_val_lane_wits.append(&mut chunk_val_lane_wits);
        merged_steps.push(compressed_step);
        prev_step_ctx = chunk.last();
        prev_twist_decoded = next_prev_twist_decoded;
        poseidon_carry = next_poseidon_carry;
        cursor = end;
    }

    let proof = ShardProof {
        steps: merged_steps,
        output_proof: merged_output_proof,
        segment_meta: Some(route_chunk_meta),
    };
    if ob.is_some() != proof.output_proof.is_some() {
        return Err(PiCcsError::ProtocolError(format!(
            "route-a segment output binding mismatch (requested={}, proof_has={})",
            ob.is_some(),
            proof.output_proof.is_some()
        )));
    }
    let (meta_public, meta_proof) = proof
        .segment_meta
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .fold((0usize, 0usize), |(a, b), e| (a + e.public_steps, b + e.proof_steps));
    if meta_public != steps.len() || meta_proof != proof.steps.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "route-a segment metadata mismatch (meta_public={}, segment_public={}, meta_proof={}, proof_steps={})",
            meta_public,
            steps.len(),
            meta_proof,
            proof.steps.len()
        )));
    }
    Ok((proof, accumulator_wit, merged_val_lane_wits))
}

pub(crate) fn fold_shard_verify_route_a_segment<MR, MB>(
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
    if ob_cfg.is_some() != proof.output_proof.is_some() {
        return Err(PiCcsError::InvalidInput(
            "route-a segment verification output binding mismatch".into(),
        ));
    }
    ensure_route_a_segment_instance(steps, step_idx_offset)?;

    if steps.is_empty() {
        return Ok(ShardFoldOutputs {
            obligations: ShardObligations {
                main: acc_init.to_vec(),
                val: Vec::new(),
            },
        });
    }

    let mut accumulator = acc_init.to_vec();
    let mut val_lane_obligations: Vec<CeClaim<Cmt, F, K>> = Vec::new();
    let mut prev_step_ctx: Option<&StepInstanceBundle<Cmt, F, K>> = initial_prev_step;

    let route_chunk_meta = proof.segment_meta.as_deref().ok_or_else(|| {
        PiCcsError::InvalidInput(
            "route-a segment verification requires segment_meta (legacy unsegmented proofs are not supported)".into(),
        )
    })?;
    if route_chunk_meta.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "route-a segment proof has empty segment_meta".into(),
        ));
    }

    let mut step_cursor = 0usize;
    let mut proof_cursor = 0usize;
    for (meta_idx, entry) in route_chunk_meta.iter().enumerate() {
        if entry.kind != ShardSegmentKind::RouteA {
            return Err(PiCcsError::InvalidInput(format!(
                "route-a segment metadata kind mismatch at entry {}: {:?}",
                meta_idx, entry.kind
            )));
        }
        if entry.public_steps == 0 || entry.proof_steps == 0 {
            return Err(PiCcsError::InvalidInput(format!(
                "route-a segment metadata entry {} must have public_steps>=1 and proof_steps>=1",
                meta_idx
            )));
        }
        if step_cursor + entry.public_steps > steps.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "route-a segment metadata overflows public steps at entry {}",
                meta_idx
            )));
        }
        if proof_cursor + entry.proof_steps > proof.steps.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "route-a segment proof too short for metadata entry {} (need {}, remaining {})",
                meta_idx,
                entry.proof_steps,
                proof.steps.len().saturating_sub(proof_cursor)
            )));
        }
        if entry.proof_steps != 1 {
            return Err(PiCcsError::InvalidInput(format!(
                "route-a segment metadata entry {} must have proof_steps=1 (got {}, public_steps={})",
                meta_idx, entry.proof_steps, entry.public_steps
            )));
        }

        let chunk = &steps[step_cursor..step_cursor + entry.public_steps];
        let chunk_step_offset = step_idx_offset
            .checked_add(step_cursor)
            .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;
        let chunk_is_final = step_cursor + entry.public_steps == steps.len();
        let container = &proof.steps[proof_cursor];
        let chunk_steps = expand_route_a_chunk_steps(container, entry.public_steps, meta_idx)?;
        let chunk_proof = ShardProof {
            steps: chunk_steps,
            output_proof: if chunk_is_final {
                proof.output_proof.clone()
            } else {
                None
            },
            segment_meta: None,
        };
        let chunk_ob_cfg = if chunk_is_final { ob_cfg } else { None };
        let chunk_outputs = fold_shard_verify_impl(
            mode.clone(),
            tr,
            params,
            s_me,
            chunk,
            chunk_step_offset,
            &accumulator,
            &chunk_proof,
            mixers,
            chunk_ob_cfg,
            prover_ctx,
            prev_step_ctx,
        )?;

        accumulator = chunk_outputs.obligations.main;
        val_lane_obligations.extend(chunk_outputs.obligations.val);
        prev_step_ctx = chunk.last();
        step_cursor += entry.public_steps;
        proof_cursor += 1;
    }

    if step_cursor != steps.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "route-a segment metadata consumed {} public steps, expected {}",
            step_cursor,
            steps.len()
        )));
    }
    if proof_cursor != proof.steps.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "route-a segment proof has {} extra step(s) after chunked verification",
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
