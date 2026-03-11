use super::*;

/// Deterministic safe batch size for CCS-only K>1 folding.
///
/// Uses the Π_RLC bound `count * T * (b - 1) < B` with a conservative carried-ME count:
/// `max(acc_len, k_rho)` so repeated batches remain sound.
pub(crate) fn auto_mcs_batch_size(params: &NeoParams, acc_len: usize, step_count: usize) -> usize {
    if step_count <= 1 {
        return 1;
    }
    let denom = (params.T as u128).saturating_mul((params.b as u128).saturating_sub(1));
    if denom == 0 {
        return 1;
    }
    let count_limit = (params.B as u128).saturating_sub(1) / denom;
    let carried = core::cmp::max(acc_len, params.k_rho as usize) as u128;
    if count_limit <= carried {
        return 1;
    }
    let max_mcs = (count_limit - carried) as usize;
    core::cmp::min(step_count, core::cmp::max(1, max_mcs))
}

fn ensure_ccs_only_steps_witness(steps: &[StepWitnessBundle<Cmt, F, K>]) -> Result<(), PiCcsError> {
    for (idx, step) in steps.iter().enumerate() {
        if !step.lut_instances.is_empty() || !step.mem_instances.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "ccs-only batching does not support mem/lut sidecars (step_idx={idx})"
            )));
        }
    }
    Ok(())
}

fn ensure_ccs_only_steps_instance(steps: &[StepInstanceBundle<Cmt, F, K>]) -> Result<(), PiCcsError> {
    for (idx, step) in steps.iter().enumerate() {
        if !step.lut_insts.is_empty() || !step.mem_insts.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "ccs-only batching does not support mem/lut sidecars (step_idx={idx})"
            )));
        }
    }
    Ok(())
}

fn empty_mem_sidecar_proof() -> MemSidecarProof<Cmt, F, K> {
    MemSidecarProof {
        val_me_claims: Vec::new(),
        wb_me_claims: Vec::new(),
        wp_me_claims: Vec::new(),
        poseidon_cycle_me_claims: Vec::new(),
        poseidon_local_me_claims: Vec::new(),
        shout_addr_pre: Default::default(),
        proofs: Vec::new(),
    }
}

fn empty_batched_time_proof() -> BatchedTimeProof {
    BatchedTimeProof {
        claimed_sums: Vec::new(),
        degree_bounds: Vec::new(),
        labels: Vec::new(),
        round_polys: Vec::new(),
    }
}

fn ensure_empty_sidecars_in_step_proof(step_idx: usize, step_proof: &StepProof) -> Result<(), PiCcsError> {
    if !step_proof.mem.val_me_claims.is_empty()
        || !step_proof.mem.wb_me_claims.is_empty()
        || !step_proof.mem.wp_me_claims.is_empty()
        || !step_proof.mem.poseidon_cycle_me_claims.is_empty()
        || !step_proof.mem.poseidon_local_me_claims.is_empty()
        || !step_proof.mem.shout_addr_pre.claimed_sums.is_empty()
        || !step_proof.mem.shout_addr_pre.groups.is_empty()
        || !step_proof.mem.proofs.is_empty()
    {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: expected empty mem sidecar proof for ccs-only batching",
            step_idx
        )));
    }
    if !step_proof.batched_time.claimed_sums.is_empty()
        || !step_proof.batched_time.degree_bounds.is_empty()
        || !step_proof.batched_time.labels.is_empty()
        || !step_proof.batched_time.round_polys.is_empty()
    {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: expected empty batched_time proof for ccs-only batching",
            step_idx
        )));
    }
    if !step_proof.val_fold.is_empty() || !step_proof.wb_fold.is_empty() || !step_proof.wp_fold.is_empty() {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: expected no auxiliary folding lanes for ccs-only batching",
            step_idx
        )));
    }
    if step_proof.poseidon_local_time.is_some()
        || !step_proof.poseidon_cycle_fold.is_empty()
        || !step_proof.poseidon_local_fold.is_empty()
    {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: expected no poseidon artifacts for ccs-only batching",
            step_idx
        )));
    }
    Ok(())
}

/// CCS-only shard folding with batched MCS slots.
///
/// This path batches up to `mcs_batch_size` consecutive CCS-only steps into one Π_CCS call,
/// then runs the standard main-lane Π_RLC→Π_DEC fold.
///
/// Constraints:
/// - only CCS-only steps are supported (`lut_instances` and `mem_instances` must be empty),
/// - output binding / Route-A sidecars are not used on this path.
pub fn fold_shard_prove_ccs_only_batched<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    mcs_batch_size: usize,
    compute_backend: &ProverComputeBackend,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, _next_acc, _next_wits) = fold_shard_prove_ccs_only_batched_with_outputs_and_offset(
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        mcs_batch_size,
        0,
        compute_backend,
    )?;
    Ok(proof)
}

pub(crate) fn fold_shard_prove_ccs_only_batched_with_outputs_and_offset<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    mcs_batch_size: usize,
    step_idx_offset: usize,
    compute_backend: &ProverComputeBackend,
) -> Result<(ShardProof, Vec<CeClaim<Cmt, F, K>>, Vec<Mat<F>>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if mcs_batch_size == 0 {
        return Err(PiCcsError::InvalidInput("mcs_batch_size must be >= 1".into()));
    }
    ensure_ccs_only_steps_witness(steps)?;

    if acc_init.len() != acc_wit_init.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "acc_init.len()={} != acc_wit_init.len()={}",
            acc_init.len(),
            acc_wit_init.len()
        )));
    }

    tr.append_message(b"shard/cpu_bus_mode", &[1u8]);
    let (s, cpu_bus) = crate::memory_sidecar::cpu_bus::prepare_ccs_for_shared_cpu_bus_steps(s_me, steps)?;
    if cpu_bus.bus_cols != 0 {
        return Err(PiCcsError::InvalidInput(
            "ccs-only batching requires zero shared-bus columns".into(),
        ));
    }

    let ell_d = utils::build_dims_and_policy(params, s)?.ell_d;
    let ring = ccs::RotRing::goldilocks();
    let k_dec = params.k_rho as usize;

    let ccs_sparse_cache: Option<Arc<SparseCache<F>>> = if mode_uses_sparse_cache(&mode) {
        Some(Arc::new(SparseCache::build(s)))
    } else {
        None
    };
    let backend_ctx = neo_reductions::accelerator::BackendContext::new(compute_backend)?;

    let mut accumulator = acc_init.to_vec();
    let mut accumulator_wit = acc_wit_init.to_vec();
    let mut step_proofs: Vec<StepProof> = Vec::new();

    let mut cursor = 0usize;
    while cursor < steps.len() {
        let end = core::cmp::min(cursor + mcs_batch_size, steps.len());
        let batch = &steps[cursor..end];
        let step_idx = step_idx_offset
            .checked_add(cursor)
            .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;

        for step in batch {
            crate::memory_sidecar::memory::absorb_step_memory_witness(tr, step);
        }

        let m_in = batch[0].mcs.0.m_in;
        if batch.iter().any(|step| step.mcs.0.m_in != m_in) {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: batched MCS instances must share m_in",
                step_idx
            )));
        }
        if let Some(acc0) = accumulator.first() {
            if acc0.m_in != m_in {
                return Err(PiCcsError::InvalidInput(format!(
                    "step {}: batched MCS m_in={} does not match accumulator m_in={}",
                    step_idx, m_in, acc0.m_in
                )));
            }
        }

        let mcs_list: Vec<neo_ccs::CcsClaim<Cmt, F>> = batch.iter().map(|step| step.mcs.0.clone()).collect();
        let mcs_wits: Vec<neo_ccs::CcsWitness<F>> = batch.iter().map(|step| step.mcs.1.clone()).collect();

        let (ccs_out, ccs_proof) = ccs::prove_with_context(
            mode.clone(),
            tr,
            params,
            s,
            &mcs_list,
            &mcs_wits,
            &accumulator,
            &accumulator_wit,
            l,
            &backend_ctx,
        )?;

        let expected_k = mcs_list.len() + accumulator.len();
        if ccs_out.len() != expected_k {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS returned {} outputs; expected {}",
                step_idx,
                ccs_out.len(),
                expected_k
            )));
        }
        validate_me_batch_invariants(&ccs_out, "prove ccs-only batched ccs outputs")?;

        let mut outs_Z: Vec<&Mat<F>> = Vec::with_capacity(mcs_wits.len() + accumulator_wit.len());
        outs_Z.extend(mcs_wits.iter().map(|w| &w.Z));
        outs_Z.extend(accumulator_wit.iter());

        let (main_fold, Z_split) = prove_rlc_dec_lane(
            &mode,
            RlcLane::Main,
            tr,
            params,
            s,
            ccs_sparse_cache.as_deref(),
            None,
            &ring,
            ell_d,
            k_dec,
            step_idx,
            &backend_ctx,
            None,
            &ccs_out,
            &outs_Z,
            true,
            l,
            mixers,
        )?;
        let RlcDecProof {
            rlc_rhos,
            rlc_parent,
            dec_children,
        } = main_fold;

        accumulator = dec_children.clone();
        accumulator_wit = Z_split;

        step_proofs.push(StepProof {
            fold: FoldStep {
                ccs_out,
                ccs_proof,
                rlc_rhos,
                rlc_parent,
                dec_children,
                cpu_sumcheck: crate::shard_proof_types::CpuTimeSumcheckProof::default(),
                shift_sumcheck: crate::shard_proof_types::ShiftTimeSumcheckProof::default(),
                time_cpu_commitments: Vec::new(),
                time_mem_commitments: Vec::new(),
                time_t: 0,
                time_declared_len: 0,
                time_col_ids: Vec::new(),
                memory_time_proofs: Vec::new(),
                openings: Vec::new(),
                opening_proofs: Vec::new(),
                opening_manifest: crate::shard_proof_types::OpeningClaimManifest::default(),
                opening_reduction: crate::shard_proof_types::OpeningReductionProof::default(),
                opening_unification: crate::shard_proof_types::OpeningUnificationProof::default(),
                joint_opening_lane: crate::shard_proof_types::JointOpeningLaneProof::default(),
                folding_lanes: crate::shard_proof_types::FoldingLanes::default(),
            },
            mem: empty_mem_sidecar_proof(),
            batched_time: empty_batched_time_proof(),
            poseidon_local_time: None,
            poseidon_cycle_fold: Vec::new(),
            poseidon_local_fold: Vec::new(),
            val_fold: Vec::new(),
            wb_fold: Vec::new(),
            wp_fold: Vec::new(),
            compressed_substeps: None,
            stage8_fold: Vec::new(),
        });

        tr.append_message(b"fold/step_done", &(step_idx as u64).to_le_bytes());
        cursor = end;
    }

    Ok((
        ShardProof {
            steps: step_proofs,
            output_proof: None,
            riscv_profile: None,
            riscv_memory_layout: None,
            segment_meta: None,
        },
        accumulator,
        accumulator_wit,
    ))
}

/// Verify proofs produced by [`fold_shard_prove_ccs_only_batched`].
pub fn fold_shard_verify_ccs_only_batched<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    mcs_batch_size: usize,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_ccs_only_batched_with_offset(
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        proof,
        mixers,
        mcs_batch_size,
        0,
        &ProverComputeBackend::Cpu,
    )
}

pub(crate) fn fold_shard_verify_ccs_only_batched_with_offset<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[CeClaim<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    mcs_batch_size: usize,
    step_idx_offset: usize,
    compute_backend: &ProverComputeBackend,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if mcs_batch_size == 0 {
        return Err(PiCcsError::InvalidInput("mcs_batch_size must be >= 1".into()));
    }
    ensure_ccs_only_steps_instance(steps)?;

    if proof.output_proof.is_some() {
        return Err(PiCcsError::InvalidInput(
            "ccs-only batched verification does not accept output binding proofs".into(),
        ));
    }

    tr.append_message(b"shard/cpu_bus_mode", &[1u8]);
    let (s, cpu_bus) = crate::memory_sidecar::cpu_bus::prepare_ccs_for_shared_cpu_bus_steps(s_me, steps)?;
    if cpu_bus.bus_cols != 0 {
        return Err(PiCcsError::InvalidInput(
            "ccs-only batching requires zero shared-bus columns".into(),
        ));
    }

    let dims = utils::build_dims_and_policy(params, s)?;
    let ell_d = dims.ell_d;
    let ring = ccs::RotRing::goldilocks();
    let backend_ctx = neo_reductions::accelerator::BackendContext::new(compute_backend)?;

    let expected_proof_steps = if steps.is_empty() {
        0
    } else {
        (steps.len() + mcs_batch_size - 1) / mcs_batch_size
    };
    if proof.steps.len() != expected_proof_steps {
        return Err(PiCcsError::InvalidInput(format!(
            "batched proof step count mismatch (public batches={}, proof batches={})",
            expected_proof_steps,
            proof.steps.len()
        )));
    }

    let mut accumulator = acc_init.to_vec();
    let mut cursor = 0usize;
    for step_proof in &proof.steps {
        let end = core::cmp::min(cursor + mcs_batch_size, steps.len());
        let batch = &steps[cursor..end];
        let step_idx = step_idx_offset
            .checked_add(cursor)
            .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;

        for step in batch {
            absorb_step_memory(tr, step);
        }
        ensure_empty_sidecars_in_step_proof(step_idx, step_proof)?;

        let m_in = batch[0].mcs_inst.m_in;
        if batch.iter().any(|step| step.mcs_inst.m_in != m_in) {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: batched MCS instances must share m_in",
                step_idx
            )));
        }
        if let Some(acc0) = accumulator.first() {
            if acc0.m_in != m_in {
                return Err(PiCcsError::InvalidInput(format!(
                    "step {}: batched MCS m_in={} does not match accumulator m_in={}",
                    step_idx, m_in, acc0.m_in
                )));
            }
        }

        let mcs_list: Vec<neo_ccs::CcsClaim<Cmt, F>> = batch.iter().map(|step| step.mcs_inst.clone()).collect();
        let expected_k = mcs_list.len() + accumulator.len();
        if step_proof.fold.ccs_out.len() != expected_k {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS output length mismatch (have {}, expected {})",
                step_idx,
                step_proof.fold.ccs_out.len(),
                expected_k
            )));
        }

        let ok_ccs = ccs::verify_with_context(
            mode.clone(),
            tr,
            params,
            s,
            &mcs_list,
            &accumulator,
            &step_proof.fold.ccs_out,
            &step_proof.fold.ccs_proof,
            &backend_ctx,
        )?;
        if !ok_ccs {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS verification failed in ccs-only batched mode",
                step_idx
            )));
        }
        let observed_digest = tr.digest32();
        if observed_digest != step_proof.fold.ccs_proof.header_digest.as_slice() {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS header digest mismatch in ccs-only batched mode",
                step_idx
            )));
        }
        let expected_digest: [u8; 32] = step_proof
            .fold
            .ccs_proof
            .header_digest
            .as_slice()
            .try_into()
            .map_err(|_| {
                PiCcsError::ProtocolError(format!("step {}: Π_CCS header digest must be 32 bytes", step_idx))
            })?;
        for (out_idx, out) in step_proof.fold.ccs_out.iter().enumerate() {
            if out.fold_digest != expected_digest {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[{out_idx}] fold_digest mismatch in ccs-only batched mode",
                    step_idx
                )));
            }
        }

        validate_me_batch_invariants(&step_proof.fold.ccs_out, "verify ccs-only batched ccs outputs")?;
        verify_rlc_dec_lane(
            RlcLane::Main,
            tr,
            params,
            s,
            &ring,
            ell_d,
            mixers,
            step_idx,
            &backend_ctx,
            &step_proof.fold.ccs_out,
            &step_proof.fold.rlc_rhos,
            &step_proof.fold.rlc_parent,
            &step_proof.fold.dec_children,
        )?;

        accumulator = step_proof.fold.dec_children.clone();
        tr.append_message(b"fold/step_done", &(step_idx as u64).to_le_bytes());
        cursor = end;
    }

    if cursor != steps.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "batched verifier consumed {} public steps, expected {}",
            cursor,
            steps.len()
        )));
    }

    Ok(ShardFoldOutputs {
        obligations: ShardObligations {
            main: accumulator,
            val: Vec::new(),
        },
    })
}
