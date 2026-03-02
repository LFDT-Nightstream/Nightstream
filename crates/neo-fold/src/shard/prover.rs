use super::*;

#[derive(Clone)]
pub(crate) struct ShardProverContext {
    pub ccs_mat_digest: Vec<F>,
    pub ccs_sparse_cache: Option<Arc<SparseCache<F>>>,
}

#[inline]
pub(crate) fn mode_uses_sparse_cache(mode: &FoldingMode) -> bool {
    match mode {
        FoldingMode::Optimized => true,
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck(_) => true,
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => false,
    }
}

fn cpu_sumcheck_from_ccs(
    claimed_sum: K,
    round_polys: Vec<Vec<K>>,
    r_time: &[K],
) -> crate::shard_proof_types::CpuTimeSumcheckProof {
    crate::shard_proof_types::CpuTimeSumcheckProof {
        claimed_sum,
        round_polys,
        r_time: r_time.to_vec(),
    }
}

fn shift_sumcheck_from_batched_time(
    batched_time: &crate::shard_proof_types::BatchedTimeProof,
    r_time: &[K],
    control_required: bool,
) -> Result<crate::shard_proof_types::ShiftTimeSumcheckProof, PiCcsError> {
    let control_idx = batched_time
        .labels
        .iter()
        .position(|label| label.as_slice() == b"control/next_pc_linear");
    let idx = match (control_required, control_idx) {
        (true, Some(i)) | (false, Some(i)) => i,
        (true, None) => {
            return Err(PiCcsError::ProtocolError(
                "missing batched-time control/next_pc_linear label".into(),
            ))
        }
        (false, None) => {
            return Ok(crate::shard_proof_types::ShiftTimeSumcheckProof {
                claimed_sum: K::ZERO,
                round_polys: Vec::new(),
                r_time: r_time.to_vec(),
            })
        }
    };
    let claimed_sum = *batched_time
        .claimed_sums
        .get(idx)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("missing batched-time claimed_sum at index {idx}")))?;
    let round_polys = batched_time
        .round_polys
        .get(idx)
        .cloned()
        .ok_or_else(|| PiCcsError::ProtocolError(format!("missing batched-time rounds at index {idx}")))?;
    Ok(crate::shard_proof_types::ShiftTimeSumcheckProof {
        claimed_sum,
        round_polys,
        r_time: r_time.to_vec(),
    })
}

pub(crate) fn fold_shard_prove_impl<L, MR, MB>(
    collect_val_lane_wits: bool,
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
    mut step_prove_ms_out: Option<&mut Vec<f64>>,
    initial_prev_step: Option<&StepWitnessBundle<Cmt, F, K>>,
    initial_prev_twist_decoded: Option<Vec<crate::memory_sidecar::memory::TwistDecodedColsSparse>>,
    initial_poseidon_carry: Option<crate::memory_sidecar::memory::PoseidonSidecarCarryState>,
) -> Result<
    (
        ShardProof,
        Vec<Mat<F>>,
        Vec<Mat<F>>,
        Option<Vec<crate::memory_sidecar::memory::TwistDecodedColsSparse>>,
        crate::memory_sidecar::memory::PoseidonSidecarCarryState,
    ),
    PiCcsError,
>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    for (step_idx, step) in steps.iter().enumerate() {
        if step.lut_instances.is_empty() && step.mem_instances.is_empty() {
            continue;
        }
        let is_shared_step = step
            .lut_instances
            .iter()
            .all(|(inst, wit)| inst.comms.is_empty() && wit.mats.is_empty())
            && step
                .mem_instances
                .iter()
                .all(|(inst, wit)| inst.comms.is_empty() && wit.mats.is_empty());
        if !is_shared_step {
            return Err(PiCcsError::InvalidInput(format!(
                "legacy no-shared CPU bus mode was removed; step_idx={step_idx} must use shared-bus witness format"
            )));
        }
    }
    tr.append_message(b"shard/cpu_bus_mode", &[1u8]);
    let (s, cpu_bus) = crate::memory_sidecar::cpu_bus::prepare_ccs_for_shared_cpu_bus_steps(s_me, steps)?;
    let dims = utils::build_dims_and_policy(params, s)?;
    let utils::Dims {
        ell_d,
        ell_n,
        ell_m,
        ell,
        d_sc,
        ..
    } = dims;
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
    if mode_uses_sparse_cache(&mode) && ccs_sparse_cache.is_none() {
        return Err(PiCcsError::ProtocolError(
            "missing SparseCache for optimized mode".into(),
        ));
    }
    let k_dec = params.k_rho as usize;
    let ring = ccs::RotRing::goldilocks();

    if acc_init.len() != acc_wit_init.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "acc_init.len()={} != acc_wit_init.len()={}",
            acc_init.len(),
            acc_wit_init.len()
        )));
    }

    // Initialize accumulator
    let mut accumulator = acc_init.to_vec();
    let mut accumulator_wit = acc_wit_init.to_vec();

    let mut step_proofs = Vec::with_capacity(steps.len());
    let mut val_lane_wits: Vec<Mat<F>> = Vec::new();
    let mut prev_twist_decoded = initial_prev_twist_decoded;
    let mut poseidon_carry =
        initial_poseidon_carry.unwrap_or_else(crate::memory_sidecar::memory::PoseidonSidecarCarryState::new);
    let mut output_proof: Option<neo_memory::output_check::OutputBindingProof> = None;
    if ob.is_some() && steps.is_empty() {
        return Err(PiCcsError::InvalidInput("output binding requires >= 1 step".into()));
    }

    for (idx, step) in steps.iter().enumerate() {
        let step_idx = step_idx_offset
            .checked_add(idx)
            .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;
        let step_start = time_now();
        crate::memory_sidecar::memory::absorb_step_memory_witness(tr, step);

        let include_ob = ob.is_some() && (idx + 1 == steps.len());
        let mut wb_time_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut wp_time_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut decode_decode_fields_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut decode_decode_immediates_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> =
            None;
        let mut width_bitness_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut width_quiescence_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut width_load_semantics_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut width_store_semantics_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut control_next_pc_linear_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut control_next_pc_control_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> =
            None;
        let mut control_branch_semantics_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> =
            None;
        let mut control_control_writeback_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> =
            None;
        let mut ob_time_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut ob_r_prime: Option<Vec<K>> = None;
        let mut ob_sparse_addr_weights: Option<Vec<(Vec<K>, K)>> = None;

        // Output binding is injected only on the final step, and must run before sampling Route-A `r_time`.
        if include_ob {
            let (cfg, final_memory_state) =
                ob.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;

            if output_proof.is_some() {
                return Err(PiCcsError::ProtocolError(
                    "output binding already attached (internal error)".into(),
                ));
            }

            if cfg.mem_idx >= step.mem_instances.len() {
                return Err(PiCcsError::InvalidInput("output binding mem_idx out of range".into()));
            }
            let expected_k = 1usize
                .checked_shl(cfg.num_bits as u32)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
            let mem_inst = &step.mem_instances[cfg.mem_idx].0;
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

            tr.append_message(b"shard/output_binding_start", &(step_idx as u64).to_le_bytes());
            tr.append_u64s(b"output_binding/mem_idx", &[cfg.mem_idx as u64]);
            tr.append_u64s(b"output_binding/num_bits", &[cfg.num_bits as u64]);

            let use_dense_output_sumcheck = cfg.num_bits <= neo_memory::output_check::OUTPUT_SUMCHECK_MAX_NUM_BITS;
            if use_dense_output_sumcheck {
                if final_memory_state.len() != expected_k {
                    return Err(PiCcsError::InvalidInput(format!(
                        "output binding: final_memory_state.len()={} != 2^num_bits={}",
                        final_memory_state.len(),
                        expected_k
                    )));
                }
                let (output_sc, r_prime) = neo_memory::output_check::generate_output_sumcheck_proof_and_challenges(
                    tr,
                    cfg.num_bits,
                    cfg.program_io.clone(),
                    final_memory_state,
                )
                .map_err(|e| PiCcsError::ProtocolError(format!("output sumcheck failed: {e:?}")))?;
                output_proof = Some(neo_memory::output_check::OutputBindingProof { output_sc });
                ob_r_prime = Some(r_prime);
            } else {
                let sampled = crate::output_binding::sample_output_lincomb_weights(tr, &cfg.program_io);
                let addr_weights = sampled
                    .into_iter()
                    .map(|(addr, _claim_value, alpha)| {
                        (crate::output_binding::addr_bits_as_k(addr, cfg.num_bits), alpha)
                    })
                    .collect::<Vec<_>>();
                output_proof = Some(neo_memory::output_check::OutputBindingProof {
                    output_sc: neo_memory::output_check::OutputSumcheckProof::default(),
                });
                ob_sparse_addr_weights = Some(addr_weights);
            }
        }

        let (mcs_inst, mcs_wit) = &step.mcs;
        let route_steps = {
            let inst_steps = step
                .lut_instances
                .iter()
                .map(|(inst, _)| inst.steps)
                .chain(step.mem_instances.iter().map(|(inst, _)| inst.steps))
                .max()
                .unwrap_or(0);
            core::cmp::max(step.time_columns.t, inst_steps)
        };
        let route_domain = mcs_inst
            .m_in
            .checked_add(route_steps)
            .ok_or_else(|| PiCcsError::InvalidInput("prove/route_a: route domain overflow".into()))?;
        // Keep Route-A row challenge dimension aligned with Π_RLC/Π_DEC validators,
        // which use at least one row bit even for n=1 domains.
        let route_pow2 = route_domain.max(2).next_power_of_two();
        let ell_t = route_pow2.trailing_zeros() as usize;
        let time_declared_len = validate_time_active_mask_and_count(
            step.time_columns.active_col.as_slice(),
            step.time_columns.t,
            "prove/time_columns",
        )?;
        let (time_cpu_commitments, time_mem_commitments) = commit_time_column_sets(
            params,
            step.time_columns.t,
            &step.time_columns.cpu_cols,
            &step.time_columns.mem_cols,
            "prove/time_columns",
        )?;
        let expected_time_col_ids = time_cpu_commitments
            .len()
            .checked_add(time_mem_commitments.len())
            .ok_or_else(|| PiCcsError::InvalidInput("time commitment count overflow".into()))?;
        if step.time_columns.col_ids.len() != expected_time_col_ids {
            return Err(PiCcsError::ProtocolError(format!(
                "time column metadata mismatch: col_ids.len()={} != cpu_commitments+mem_commitments={expected_time_col_ids}",
                step.time_columns.col_ids.len()
            )));
        }
        for (idx, &col_id) in step.time_columns.col_ids.iter().enumerate() {
            if col_id != idx {
                return Err(PiCcsError::ProtocolError(format!(
                    "time column metadata mismatch: col_ids must be canonical contiguous ids (col_ids[{idx}]={col_id}, expected {idx})"
                )));
            }
        }
        let has_committed_time_cpu = step.time_columns.t > 0 && !time_cpu_commitments.is_empty();
        let has_committed_time_mem = step.time_columns.t > 0 && !time_mem_commitments.is_empty();

        // k = accumulator.len() + 1
        let k = accumulator.len() + 1;

        // --------------------------------------------------------------------
        // Route A: Shared-challenge batched sum-check for time/row rounds.
        // --------------------------------------------------------------------
        utils::bind_header_and_instances_with_digest(
            tr,
            params,
            &s,
            core::slice::from_ref(mcs_inst),
            dims,
            &ccs_mat_digest,
        )?;
        utils::bind_me_inputs(tr, &accumulator)?;
        bind_time_column_commitments(
            tr,
            step_idx,
            step.time_columns.t,
            time_declared_len,
            &step.time_columns.col_ids,
            &time_cpu_commitments,
            &time_mem_commitments,
        );
        let mut ch = utils::sample_challenges(tr, ell_d, ell)?;
        ch.beta_m = utils::sample_beta_m(tr, ell_m)?;
        let ccs_initial_sum = claimed_initial_sum_from_inputs_with_k_mcs(&s, &ch, 1, &accumulator);
        tr.append_fields(b"sumcheck/initial_sum", &ccs_initial_sum.as_coeffs());

        // Build Poseidon lanes and bind their commitments *before* sampling route_a/r_cycle.
        let poseidon_setup = build_poseidon_prover_setup(tr, params, step, step_idx, ell_n, &mut poseidon_carry)?;
        let poseidon_cycle_enabled = poseidon_setup.cycle_enabled;
        let poseidon_sidecar = poseidon_setup.sidecar;
        let mut poseidon_cycle_wit = poseidon_setup.cycle_wit;
        let poseidon_cycle_open_spec = poseidon_setup.cycle_open_spec;
        let mut poseidon_cycle_wits: Option<Vec<Mat<F>>> = None;
        let mut poseidon_cycle_open_specs: Option<Vec<(usize, usize, Vec<usize>)>> = None;
        let mut poseidon_local_wit_full = poseidon_setup.local_wit_full;
        let mut poseidon_local_wits = poseidon_setup.local_wits;
        let mut poseidon_local_open_specs = poseidon_setup.local_open_specs;
        let poseidon_local_t_len = poseidon_setup.local_t_len;
        let poseidon_local_layout = poseidon_setup.local_layout;
        let poseidon_local_ell = poseidon_setup.local_ell;
        let mut poseidon_link_chals: Option<crate::memory_sidecar::memory::PoseidonLinkChallenges> = None;
        let mut poseidon_cont_chals: Option<crate::memory_sidecar::memory::PoseidonContinuityChallenges> = None;
        if poseidon_cycle_enabled {
            let sidecar_ref = poseidon_sidecar
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon sidecar table".into()))?;
            let cycle_open_spec = poseidon_cycle_open_spec
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon cycle opening spec".into()))?;
            let local_t_len =
                poseidon_local_t_len.ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local t_len".into()))?;
            let local_layout = poseidon_local_layout
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local layout".into()))?;
            let mcs_logical = neo_memory::ajtai::decode_vector_for_ccs_m(params, s.m, &mcs_wit.Z).map_err(|e| {
                PiCcsError::ProtocolError(format!(
                    "failed to decode packed main witness for poseidon lane prefix (m={}): {e}",
                    s.m
                ))
            })?;
            let link_chals = crate::memory_sidecar::memory::sample_poseidon_link_challenges(tr);
            let cont_chals = crate::memory_sidecar::memory::sample_poseidon_continuity_challenges(tr);

            let cycle_layout = crate::memory_sidecar::memory::PoseidonCycleTraceLayout::new();
            let cycle_wit = poseidon_cycle_wit
                .as_mut()
                .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon cycle witness".into()))?;
            crate::memory_sidecar::memory::populate_poseidon_cycle_link_aux_columns(
                cycle_open_spec.1,
                cycle_wit,
                &cycle_layout,
                sidecar_ref,
                &link_chals,
                &cont_chals,
            )?;

            let local_wit = poseidon_local_wit_full
                .as_mut()
                .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local witness".into()))?;
            crate::memory_sidecar::memory::populate_poseidon_local_link_aux_column(
                local_t_len,
                local_wit,
                local_layout,
                sidecar_ref,
                &link_chals,
            )?;

            let cycle_wit_ro = poseidon_cycle_wit
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon cycle witness".into()))?;
            let (cycle_wits_built, cycle_cols_per_chunk) = split_poseidon_lane_wit_by_time_cols(
                params,
                cycle_wit_ro,
                cycle_open_spec.2.len(),
                cycle_open_spec.1,
                cycle_open_spec.0,
                Some(mcs_logical.as_slice()),
                s.m,
            )?;
            let cycle_open_specs_built: Vec<(usize, usize, Vec<usize>)> = cycle_cols_per_chunk
                .into_iter()
                .map(|chunk_cols| (cycle_open_spec.0, cycle_open_spec.1, (0..chunk_cols).collect()))
                .collect();
            poseidon_cycle_wits = Some(cycle_wits_built);
            poseidon_cycle_open_specs = Some(cycle_open_specs_built);

            let (local_wits_built, local_cols_per_chunk) = split_poseidon_lane_wit_by_time_cols(
                params,
                local_wit,
                local_layout.cols(),
                local_t_len,
                0,
                None,
                s.m,
            )?;
            let local_open_specs_built: Vec<(usize, usize, Vec<usize>)> = local_cols_per_chunk
                .into_iter()
                .map(|chunk_cols| (0, local_t_len, (0..chunk_cols).collect()))
                .collect();
            poseidon_local_wits = Some(local_wits_built);
            poseidon_local_open_specs = Some(local_open_specs_built);
            poseidon_link_chals = Some(link_chals);
            poseidon_cont_chals = Some(cont_chals);
        }

        let (poseidon_cycle_commits, poseidon_local_commits) = if poseidon_cycle_enabled {
            let cycle_wits_ref = poseidon_cycle_wits
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon cycle witness chunks".into()))?;
            let local_wits_ref = poseidon_local_wits
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local witness chunks".into()))?;
            let mut cycle_cs = Vec::with_capacity(cycle_wits_ref.len());
            for z in cycle_wits_ref.iter() {
                let cycle_committer = crate::shard::poseidon_lane_helpers::poseidon_lane_committer(
                    params,
                    z.cols(),
                    "poseidon cycle commit",
                )?;
                cycle_cs.push(cycle_committer.commit(z));
            }
            let mut local_cs = Vec::with_capacity(local_wits_ref.len());
            for z in local_wits_ref.iter() {
                let local_committer = crate::shard::poseidon_lane_helpers::poseidon_lane_committer(
                    params,
                    z.cols(),
                    "poseidon local commit",
                )?;
                local_cs.push(local_committer.commit(z));
            }
            absorb_poseidon_lane_commitments_prover(tr, &cycle_cs, &local_cs);
            (Some(cycle_cs), Some(local_cs))
        } else {
            (None, None)
        };

        // Route A memory checks use a separate transcript-derived cycle point `r_cycle`
        // to form χ_{r_cycle}(t) weights inside their sum-check polynomials.
        let r_cycle: Vec<K> =
            ts::sample_ext_point(tr, b"route_a/r_cycle", b"route_a/cycle/0", b"route_a/cycle/1", ell_t);

        // CCS oracle (engine-selected).
        //
        // Keep the optimized oracle concrete so we can build outputs from its Ajtai precompute.
        let mut ccs_oracle: CcsOracleDispatch<'_> = match mode.clone() {
            FoldingMode::Optimized => {
                let sparse = ccs_sparse_cache
                    .as_ref()
                    .ok_or_else(|| PiCcsError::ProtocolError("missing SparseCache for optimized mode".into()))?;
                CcsOracleDispatch::Optimized(
                    neo_reductions::engines::optimized_engine::oracle::OptimizedOracle::new_with_sparse(
                        &s,
                        params,
                        core::slice::from_ref(mcs_wit),
                        &accumulator_wit,
                        ch.clone(),
                        ell_d,
                        ell_n,
                        d_sc,
                        accumulator.first().map(|mi| mi.r.as_slice()),
                        sparse.clone(),
                    ),
                )
            }
            #[cfg(feature = "paper-exact")]
            FoldingMode::PaperExact => CcsOracleDispatch::PaperExact(
                neo_reductions::engines::paper_exact_engine::oracle::PaperExactOracle::new(
                    &s,
                    params,
                    core::slice::from_ref(mcs_wit),
                    &accumulator_wit,
                    ch.clone(),
                    ell_d,
                    ell_n,
                    d_sc,
                    accumulator.first().map(|mi| mi.r.as_slice()),
                ),
            ),
            #[cfg(feature = "paper-exact")]
            FoldingMode::OptimizedWithCrosscheck(_) => {
                let sparse = ccs_sparse_cache
                    .as_ref()
                    .ok_or_else(|| PiCcsError::ProtocolError("missing SparseCache for optimized mode".into()))?;
                CcsOracleDispatch::Optimized(
                    neo_reductions::engines::optimized_engine::oracle::OptimizedOracle::new_with_sparse(
                        &s,
                        params,
                        core::slice::from_ref(mcs_wit),
                        &accumulator_wit,
                        ch.clone(),
                        ell_d,
                        ell_n,
                        d_sc,
                        accumulator.first().map(|mi| mi.r.as_slice()),
                        sparse.clone(),
                    ),
                )
            }
        };

        let shout_pre = crate::memory_sidecar::memory::prove_shout_addr_pre_time(
            tr, params, step, &cpu_bus, ell_t, &r_cycle, step_idx,
        )?;

        let twist_pre =
            crate::memory_sidecar::memory::prove_twist_addr_pre_time(tr, params, step, &cpu_bus, ell_t, &r_cycle)
                .map_err(|e| PiCcsError::ProtocolError(format!("twist addr-pre failed at step_idx={step_idx}: {e}")))?;
        let twist_read_claims: Vec<K> = twist_pre.iter().map(|p| p.read_check_claim_sum).collect();
        let twist_write_claims: Vec<K> = twist_pre.iter().map(|p| p.write_check_claim_sum).collect();
        let mut mem_oracles = crate::memory_sidecar::memory::build_route_a_memory_oracles(
            params, step, ell_t, &r_cycle, &shout_pre, &twist_pre,
        )?;

        let (wb_time_claim_built, wp_time_claim_built) =
            crate::memory_sidecar::memory::build_route_a_wb_wp_time_claims(params, step, &r_cycle)?;
        let wb_wp_required = crate::memory_sidecar::memory::wb_wp_required_for_step_witness(step);
        if wb_wp_required && (wb_time_claim_built.is_none() || wp_time_claim_built.is_none()) {
            return Err(PiCcsError::ProtocolError(
                "WB/WP claims are required in RV32 trace mode but were not built".into(),
            ));
        }
        if let Some((oracle, _claimed_sum)) = wb_time_claim_built {
            wb_time_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"wb/booleanity",
            });
        }
        if let Some((oracle, _claimed_sum)) = wp_time_claim_built {
            wp_time_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"wp/quiescence",
            });
        }
        let (decode_decode_fields_built, decode_decode_immediates_built) =
            crate::memory_sidecar::memory::build_route_a_decode_time_claims(params, step, &r_cycle)?;
        let decode_required = crate::memory_sidecar::memory::decode_stage_required_for_step_witness(step);
        if decode_required && (decode_decode_fields_built.is_none() || decode_decode_immediates_built.is_none()) {
            return Err(PiCcsError::ProtocolError(
                "decode stage claims are required in RV32 trace mode but were not built".into(),
            ));
        }
        if let Some((oracle, _claimed_sum)) = decode_decode_fields_built {
            decode_decode_fields_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"decode/fields",
            });
        }
        if let Some((oracle, _claimed_sum)) = decode_decode_immediates_built {
            decode_decode_immediates_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"decode/immediates",
            });
        }
        let (
            width_bitness_built,
            width_quiescence_built,
            _width_selector_linkage_built,
            width_load_semantics_built,
            width_store_semantics_built,
        ) = crate::memory_sidecar::memory::build_route_a_width_time_claims(params, step, &r_cycle)?;
        let width_required = crate::memory_sidecar::memory::width_stage_required_for_step_witness(step);
        if width_required
            && (width_bitness_built.is_none()
                || width_quiescence_built.is_none()
                || width_load_semantics_built.is_none()
                || width_store_semantics_built.is_none())
        {
            return Err(PiCcsError::ProtocolError(
                "width stage claims are required in RV32 trace mode but were not built".into(),
            ));
        }
        if let Some((oracle, _claimed_sum)) = width_bitness_built {
            width_bitness_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"width/bitness",
            });
        }
        if let Some((oracle, _claimed_sum)) = width_quiescence_built {
            width_quiescence_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"width/quiescence",
            });
        }
        if let Some((oracle, _claimed_sum)) = width_load_semantics_built {
            width_load_semantics_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"width/load_semantics",
            });
        }
        if let Some((oracle, _claimed_sum)) = width_store_semantics_built {
            width_store_semantics_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"width/store_semantics",
            });
        }
        let (
            control_next_pc_linear_built,
            control_next_pc_control_built,
            control_branch_semantics_built,
            control_control_writeback_built,
        ) = crate::memory_sidecar::memory::build_route_a_control_time_claims(params, step, &r_cycle)?;
        let control_required = crate::memory_sidecar::memory::control_stage_required_for_step_witness(step);
        if control_required
            && (control_next_pc_linear_built.is_none()
                || control_next_pc_control_built.is_none()
                || control_branch_semantics_built.is_none()
                || control_control_writeback_built.is_none())
        {
            return Err(PiCcsError::ProtocolError(
                "control stage claims are required in RV32 trace mode but were not built".into(),
            ));
        }
        if let Some((oracle, _claimed_sum)) = control_next_pc_linear_built {
            control_next_pc_linear_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"control/next_pc_linear",
            });
        }
        if let Some((oracle, _claimed_sum)) = control_next_pc_control_built {
            control_next_pc_control_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"control/next_pc_control",
            });
        }
        if let Some((oracle, _claimed_sum)) = control_branch_semantics_built {
            control_branch_semantics_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"control/branch_semantics",
            });
        }
        if let Some((oracle, _claimed_sum)) = control_control_writeback_built {
            control_control_writeback_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"control/writeback",
            });
        }
        let poseidon_cycle_claims = build_poseidon_cycle_time_claims(
            params,
            step,
            &r_cycle,
            ell_n,
            poseidon_cycle_enabled,
            poseidon_sidecar.as_ref(),
            poseidon_cycle_wit.as_ref(),
            poseidon_cycle_open_spec.as_ref(),
            poseidon_link_chals.as_ref(),
            poseidon_cont_chals.as_ref(),
        )?;
        let poseidon_io_link_claim = poseidon_cycle_claims.io_link;
        let poseidon_bitness_claim = poseidon_cycle_claims.bitness;
        let poseidon_canonical_u64_claim = poseidon_cycle_claims.canonical_u64;
        let poseidon_sidecar_link_claim = poseidon_cycle_claims.sidecar_link;
        let poseidon_mode_claim = poseidon_cycle_claims.mode;
        let poseidon_link_cycle_inv_claim = poseidon_cycle_claims.link_cycle_inv;
        let poseidon_link_cycle_sum_claim = poseidon_cycle_claims.link_cycle_sum;
        let poseidon_cont_inv_claim = poseidon_cycle_claims.cont_inv;
        let poseidon_cont_sum_claim = poseidon_cycle_claims.cont_sum;

        if include_ob {
            let (cfg, _final_memory_state) =
                ob.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;
            let pre = twist_pre
                .get(cfg.mem_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("output binding mem_idx out of range for twist_pre".into()))?;

            if pre.decoded.lanes.is_empty() {
                return Err(PiCcsError::ProtocolError(
                    "output binding: Twist decoded lanes empty".into(),
                ));
            }

            let mut oracles: Vec<Box<dyn RoundOracle>> = Vec::new();
            let mut claimed_sum = K::ZERO;
            let use_dense_output_sumcheck = cfg.num_bits <= neo_memory::output_check::OUTPUT_SUMCHECK_MAX_NUM_BITS;
            if use_dense_output_sumcheck {
                let r_prime = ob_r_prime
                    .as_ref()
                    .ok_or_else(|| PiCcsError::ProtocolError("output binding r_prime missing".into()))?;
                oracles.reserve(pre.decoded.lanes.len());
                for lane in pre.decoded.lanes.iter() {
                    let (oracle, claim) = neo_memory::twist_oracle::TwistTotalIncOracleSparseTime::new(
                        lane.wa_bits.clone(),
                        lane.has_write.clone(),
                        lane.inc_at_write_addr.clone(),
                        r_prime,
                    );
                    oracles.push(Box::new(oracle));
                    claimed_sum += claim;
                }
            } else {
                let addr_weights = ob_sparse_addr_weights
                    .as_ref()
                    .ok_or_else(|| PiCcsError::ProtocolError("output binding sparse addr/weight set missing".into()))?;
                oracles.reserve(pre.decoded.lanes.len().saturating_mul(addr_weights.len()));
                for lane in pre.decoded.lanes.iter() {
                    for (r_addr, alpha) in addr_weights.iter() {
                        if *alpha == K::ZERO {
                            continue;
                        }
                        let scaled_inc = neo_memory::sparse_time::SparseIdxVec::from_entries(
                            lane.inc_at_write_addr.len(),
                            lane.inc_at_write_addr
                                .entries()
                                .iter()
                                .map(|(idx, val)| (*idx, *val * *alpha))
                                .collect(),
                        );
                        let (oracle, claim) = neo_memory::twist_oracle::TwistTotalIncOracleSparseTime::new(
                            lane.wa_bits.clone(),
                            lane.has_write.clone(),
                            scaled_inc,
                            r_addr,
                        );
                        oracles.push(Box::new(oracle));
                        claimed_sum += claim;
                    }
                }
            }
            if oracles.is_empty() {
                return Err(PiCcsError::ProtocolError(
                    "output binding produced zero active Twist increment oracles".into(),
                ));
            }
            let oracle = crate::memory_sidecar::memory::SumRoundOracle::new(oracles)?;

            ob_time_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle: Box::new(oracle),
                claimed_sum,
                label: crate::output_binding::OB_INC_TOTAL_LABEL,
            });
        }

        let crate::memory_sidecar::route_a_time::RouteABatchedTimeProverOutput {
            r_time,
            proof: batched_time,
        } = crate::memory_sidecar::route_a_time::prove_route_a_batched_time(
            tr,
            step_idx,
            ell_t,
            &mut mem_oracles,
            step,
            twist_read_claims,
            twist_write_claims,
            wb_time_claim,
            wp_time_claim,
            decode_decode_fields_claim,
            decode_decode_immediates_claim,
            width_bitness_claim,
            width_quiescence_claim,
            None,
            width_load_semantics_claim,
            width_store_semantics_claim,
            control_next_pc_linear_claim,
            control_next_pc_control_claim,
            control_branch_semantics_claim,
            control_control_writeback_claim,
            poseidon_io_link_claim,
            poseidon_bitness_claim,
            poseidon_canonical_u64_claim,
            poseidon_sidecar_link_claim,
            poseidon_mode_claim,
            poseidon_link_cycle_inv_claim,
            poseidon_link_cycle_sum_claim,
            poseidon_cont_inv_claim,
            poseidon_cont_sum_claim,
            ob_time_claim,
        )?;

        let poseidon_local_artifacts = prove_poseidon_local_time_artifacts(
            tr,
            step_idx,
            poseidon_cycle_enabled,
            poseidon_local_ell,
            poseidon_local_open_specs.as_ref(),
            poseidon_local_t_len,
            poseidon_local_layout,
            poseidon_local_wit_full.as_ref(),
            poseidon_link_chals.as_ref(),
        )?;
        let poseidon_local_time = poseidon_local_artifacts.local_time;
        let poseidon_r_local = poseidon_local_artifacts.r_local;
        ensure_poseidon_link_sums_match(poseidon_cycle_enabled, &batched_time, poseidon_local_time.as_ref())?;

        // Run CCS row rounds independently from Route-A batching.
        let mut ccs_time = RoundOraclePrefix::new(&mut ccs_oracle, ell_n);
        let (ccs_time_rounds, ccs_time_chals) =
            run_sumcheck_prover_ds(tr, b"ccs/time", step_idx, &mut ccs_time, ccs_initial_sum)?;
        let mut ajtai_initial_sum = ccs_initial_sum;
        for (round_poly, &r_i) in ccs_time_rounds.iter().zip(ccs_time_chals.iter()) {
            ajtai_initial_sum = poly_eval_k(round_poly, r_i);
        }
        let ccs_time_rounds_meta = ccs_time_rounds.clone();
        let ccs_time_chals_meta = ccs_time_chals.clone();
        let mut sumcheck_rounds = ccs_time_rounds;
        let mut sumcheck_chals = ccs_time_chals;

        let mut ccs_ajtai = RoundOraclePrefix::new(&mut ccs_oracle, ell_d);
        let (ajtai_rounds, ajtai_chals) =
            run_sumcheck_prover_ds(tr, b"ccs/ajtai", step_idx, &mut ccs_ajtai, ajtai_initial_sum)?;
        let mut running_sum = ajtai_initial_sum;
        for (round_poly, &r_i) in ajtai_rounds.iter().zip(ajtai_chals.iter()) {
            running_sum = poly_eval_k(round_poly, r_i);
        }
        sumcheck_rounds.extend_from_slice(&ajtai_rounds);
        sumcheck_chals.extend_from_slice(&ajtai_chals);

        // --------------------------------------------------------------------
        // NC-only sumcheck (digit-range / norm-check) over {0,1}^{ell_m + ell_d}.
        // --------------------------------------------------------------------
        let mut ccs_nc_oracle = neo_reductions::engines::optimized_engine::oracle::NcOracle::new(
            &s,
            params,
            core::slice::from_ref(mcs_wit),
            &accumulator_wit,
            ch.clone(),
            ell_d,
            ell_m,
            d_sc,
        );
        let (sumcheck_rounds_nc, sumcheck_chals_nc) =
            run_sumcheck_prover_ds(tr, b"ccs/nc", step_idx, &mut ccs_nc_oracle, K::ZERO)?;
        let mut running_sum_nc = K::ZERO;
        for (round_poly, &r_i) in sumcheck_rounds_nc.iter().zip(sumcheck_chals_nc.iter()) {
            running_sum_nc = poly_eval_k(round_poly, r_i);
        }
        let (s_col, _alpha_prime_nc) = sumcheck_chals_nc.split_at(ell_m);

        // Build CCS ME outputs at r_time.
        let fold_digest = tr.digest32();
        let mut ccs_out = match &mut ccs_oracle {
            CcsOracleDispatch::Optimized(oracle) => oracle.build_me_outputs_from_ajtai_precomp(
                core::slice::from_ref(mcs_inst),
                &accumulator,
                s_col,
                fold_digest,
                l,
            ),
            #[cfg(feature = "paper-exact")]
            CcsOracleDispatch::PaperExact(_) => build_me_outputs_paper_exact(
                &s,
                params,
                core::slice::from_ref(mcs_inst),
                core::slice::from_ref(mcs_wit),
                &accumulator,
                &accumulator_wit,
                &ccs_time_chals_meta,
                s_col,
                ell_d,
                fold_digest,
                l,
            ),
        };

        // CCS oracle borrows accumulator_wit; drop before updating accumulator_wit at the end.
        drop(ccs_oracle);

        let mut trace_linkage_t_len: Option<usize> = None;
        let mut named_trace_col_ids: Vec<usize> = Vec::new();
        let core_t = s.t();

        // Shared CPU bus: append "implicit openings" for all bus columns without materializing
        // bus copyout matrices into the CCS.
        if cpu_bus.bus_cols > 0 {
            if ccs_out.len() != 1 + accumulator_wit.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "CCS output count mismatch for bus openings (ccs_out.len()={}, expected {})",
                    ccs_out.len(),
                    1 + accumulator_wit.len()
                )));
            }

            let can_use_time_mem_cols =
                step.time_columns.t == cpu_bus.chunk_size && step.time_columns.mem_cols.len() == cpu_bus.bus_cols;
            if !can_use_time_mem_cols {
                return Err(PiCcsError::ProtocolError(format!(
                    "shared bus openings require canonical time mem columns (time_t={}, mem_cols={}, chunk_size={}, bus_cols={})",
                    step.time_columns.t,
                    step.time_columns.mem_cols.len(),
                    cpu_bus.chunk_size,
                    cpu_bus.bus_cols
                )));
            }
            let out0_supports_bus_point =
                crate::memory_sidecar::cpu_bus::point_covers_bus_time_rows(&cpu_bus, ccs_out[0].r.as_slice())?;
            if out0_supports_bus_point {
                crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance_from_time_columns(
                    params,
                    &cpu_bus,
                    core_t,
                    &step.time_columns.mem_cols,
                    &mut ccs_out[0],
                )?;
            } else {
                crate::memory_sidecar::cpu_bus::append_zero_bus_openings_to_me_instance(
                    params,
                    &cpu_bus,
                    core_t,
                    &mut ccs_out[0],
                )?;
            }
            for (out, Z) in ccs_out.iter_mut().skip(1).zip(accumulator_wit.iter()) {
                let out_supports_bus_point =
                    crate::memory_sidecar::cpu_bus::point_covers_bus_time_rows(&cpu_bus, out.r.as_slice())?;
                if Z.cols() == cpu_bus.m && out_supports_bus_point {
                    crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                        params, &cpu_bus, core_t, Z, out,
                    )?;
                } else {
                    crate::memory_sidecar::cpu_bus::append_zero_bus_openings_to_me_instance(
                        params, &cpu_bus, core_t, out,
                    )?;
                }
            }
        }

        // For RV32 trace wiring CCS, append time-combined openings for trace columns needed to
        // link Twist/Shout sidecars at r_time. In shared-bus mode this is appended after bus openings.
        if crate::memory_sidecar::memory::wb_wp_required_for_step_witness(step) && mcs_inst.m_in == 5 {
            // Infer that the CPU witness is the RV32 trace column-major layout:
            // z = [x (m_in) | trace_cols * t_len]
            let m_in = mcs_inst.m_in;
            let t_len = (step.time_columns.t > 0 && !step.time_columns.cpu_cols.is_empty())
                .then_some(step.time_columns.t)
                .or_else(|| step.mem_instances.first().map(|(inst, _wit)| inst.steps))
                .or_else(|| {
                    // Shout event-table instances may have `steps != t_len`; prefer a non-event-table
                    // instance if present, otherwise fall back to inferring from the trace layout.
                    step.lut_instances
                        .iter()
                        .find(|(inst, _wit)| {
                            !matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. }))
                        })
                        .map(|(inst, _wit)| inst.steps)
                })
                .or_else(|| {
                    // Trace CCS layout inference: z = [x (m_in) | trace_cols * t_len]
                    let trace = Rv32TraceLayout::new();
                    let w = s.m.checked_sub(m_in)?;
                    if trace.cols == 0 || w % trace.cols != 0 {
                        return None;
                    }
                    Some(w / trace.cols)
                })
                .ok_or_else(|| PiCcsError::InvalidInput("missing mem/lut instances".into()))?;
            if t_len == 0 {
                return Err(PiCcsError::InvalidInput("trace linkage requires steps>=1".into()));
            }
            for (i, (inst, _wit)) in step.mem_instances.iter().enumerate() {
                if inst.steps != t_len {
                    return Err(PiCcsError::InvalidInput(format!(
                        "trace linkage requires stable steps across mem instances (mem_idx={i} has steps={}, expected {t_len})",
                        inst.steps
                    )));
                }
            }

            let trace = Rv32TraceLayout::new();

            let trace_cols_to_open_dense: Vec<usize> = vec![
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
            ];
            let trace_cols_to_open_shout: Vec<usize> = vec![
                trace.shout_has_lookup,
                trace.shout_val,
                trace.shout_link_lhs,
                trace.shout_link_rhs,
                trace.shout_add_sub_key,
            ];
            let trace_cols_to_open_all: Vec<usize> = trace_cols_to_open_dense
                .iter()
                .chain(trace_cols_to_open_shout.iter())
                .copied()
                .collect();
            let trace_open_base = core_t + cpu_bus.bus_cols;
            let can_use_time_cpu_cols = step.time_columns.t == t_len
                && !step.time_columns.cpu_cols.is_empty()
                && trace_cols_to_open_all
                    .iter()
                    .all(|&col_id| col_id < step.time_columns.cpu_cols.len());
            if !can_use_time_cpu_cols {
                return Err(PiCcsError::ProtocolError(format!(
                    "route-a trace openings require canonical time CPU columns (time_t={}, cpu_cols={}, expected_t={t_len}, max_trace_col={})",
                    step.time_columns.t,
                    step.time_columns.cpu_cols.len(),
                    trace_cols_to_open_all.iter().copied().max().unwrap_or(0)
                )));
            }

            // Event-table style micro-optimization: Shout trace columns are constrained to be 0
            // whenever `shout_has_lookup == 0`, so we can compute their openings by summing only
            // over the active lookup rows.
            let active_shout_js: Vec<usize> = step.time_columns.cpu_cols[trace.shout_has_lookup]
                .iter()
                .enumerate()
                .filter_map(|(j, v)| (*v != F::ZERO).then_some(j))
                .collect();

            crate::memory_sidecar::cpu_bus::append_time_columns_openings_to_me_instance(
                params,
                m_in,
                t_len,
                &step.time_columns.cpu_cols,
                &trace_cols_to_open_dense,
                trace_open_base,
                &mut ccs_out[0],
            )?;
            crate::memory_sidecar::cpu_bus::append_time_columns_openings_to_me_instance_at_js(
                params,
                m_in,
                t_len,
                &step.time_columns.cpu_cols,
                &trace_cols_to_open_shout,
                trace_open_base + trace_cols_to_open_dense.len(),
                &mut ccs_out[0],
                &active_shout_js,
            )?;

            for (out, Z) in ccs_out.iter_mut().skip(1).zip(accumulator_wit.iter()) {
                let _ = Z; // only child 0 carries canonical non-physical trace openings.
                crate::memory_sidecar::cpu_bus::append_zero_time_openings_to_me_instance(
                    params,
                    trace_cols_to_open_all.len(),
                    trace_open_base,
                    out,
                )?;
            }
            named_trace_col_ids.extend(trace_cols_to_open_all.iter().copied());
            trace_linkage_t_len = Some(t_len);
        }

        if ccs_out.len() != k {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_CCS returned {} outputs; expected k={k}",
                ccs_out.len()
            )));
        }

        let mut ccs_proof = crate::PiCcsProof::new(sumcheck_rounds, Some(ccs_initial_sum));
        ccs_proof.variant = crate::optimized_engine::PiCcsProofVariant::SplitNcV1;
        ccs_proof.sumcheck_challenges = sumcheck_chals;
        ccs_proof.sumcheck_rounds_nc = sumcheck_rounds_nc;
        ccs_proof.sc_initial_sum_nc = Some(K::ZERO);
        ccs_proof.sumcheck_challenges_nc = sumcheck_chals_nc;
        ccs_proof.challenges_public = ch;
        ccs_proof.sumcheck_final = running_sum;
        ccs_proof.sumcheck_final_nc = running_sum_nc;
        ccs_proof.header_digest = fold_digest.to_vec();

        #[cfg(feature = "paper-exact")]
        if let FoldingMode::OptimizedWithCrosscheck(cfg) = &mode {
            crosscheck_route_a_ccs_step(
                cfg,
                step_idx,
                params,
                &s,
                &cpu_bus,
                mcs_inst,
                mcs_wit,
                &accumulator,
                &accumulator_wit,
                &ccs_out,
                &ccs_proof,
                ell_d,
                ell_n,
                ell_m,
                d_sc,
                fold_digest,
                l,
            )?;
        }

        // Witnesses for CCS outputs: [Z_mcs, Z_seed...] (borrow; avoid multi-GB clones)
        let mut outs_Z: Vec<&Mat<F>> = Vec::with_capacity(k);
        outs_Z.push(&mcs_wit.Z);
        outs_Z.extend(accumulator_wit.iter());

        // Memory sidecar: emit ME claims at the shared r_time (no fixed-challenge sumcheck).
        let prev_step = if idx > 0 {
            Some(&steps[idx - 1])
        } else {
            initial_prev_step
        };
        let prev_twist_decoded_ref = prev_twist_decoded.as_deref();
        let mut mem_proof = crate::memory_sidecar::memory::finalize_route_a_memory_prover(
            tr,
            params,
            &cpu_bus,
            &s,
            step,
            prev_step,
            prev_twist_decoded_ref,
            &mut mem_oracles,
            &shout_pre.addr_pre,
            &twist_pre,
            &r_time,
            mcs_inst.m_in,
            step_idx,
        )?;
        prev_twist_decoded = Some(twist_pre.into_iter().map(|p| p.decoded).collect());

        // Normalize ME claim shapes for per-claim folding lanes.
        for me in mem_proof.val_me_claims.iter_mut() {
            let t = me.y_ring.len();
            normalize_me_claims(core::slice::from_mut(me), ell_t, ell_d, t)?;
        }
        for me in mem_proof.wb_me_claims.iter_mut() {
            let t = me.y_ring.len();
            normalize_me_claims(core::slice::from_mut(me), ell_t, ell_d, t)?;
        }
        for me in mem_proof.wp_me_claims.iter_mut() {
            let t = me.y_ring.len();
            normalize_me_claims(core::slice::from_mut(me), ell_t, ell_d, t)?;
        }
        emit_poseidon_me_claims(
            tr,
            params,
            &s,
            &r_time,
            ell_t,
            ell_d,
            &mut mem_proof,
            PoseidonMeClaimsInputs {
                poseidon_cycle_enabled,
                poseidon_cycle_wits: poseidon_cycle_wits.as_ref(),
                poseidon_cycle_commits: poseidon_cycle_commits.as_ref(),
                poseidon_cycle_open_specs: poseidon_cycle_open_specs.as_ref(),
                poseidon_local_wits: poseidon_local_wits.as_ref(),
                poseidon_local_commits: poseidon_local_commits.as_ref(),
                poseidon_local_open_specs: poseidon_local_open_specs.as_ref(),
                poseidon_local_t_len,
                poseidon_local_layout,
                poseidon_r_local: poseidon_r_local.as_ref(),
            },
        )?;
        validate_me_batch_invariants(&ccs_out, "prove step ccs outputs")?;

        let want_main_wits = collect_val_lane_wits || idx + 1 < steps.len();
        let (main_fold, Z_split) = prove_rlc_dec_lane(
            &mode,
            RlcLane::Main,
            tr,
            params,
            &s,
            ccs_sparse_cache.as_deref(),
            Some(&cpu_bus),
            &ring,
            ell_d,
            k_dec,
            step_idx,
            trace_linkage_t_len,
            &ccs_out,
            &outs_Z,
            want_main_wits,
            l,
            mixers,
        )?;
        let RlcDecProof {
            rlc_rhos: rhos,
            rlc_parent: parent_pub,
            dec_children: children,
        } = main_fold;

        let has_prev = prev_step.is_some();

        // --------------------------------------------------------------------
        // Phase 2: Second folding lane for Twist val-eval ME claims at r_val.
        // --------------------------------------------------------------------
        let mut val_fold: Vec<RlcDecProof> = Vec::new();
        if !mem_proof.val_me_claims.is_empty() {
            tr.append_message(b"fold/val_lane_start", &(step_idx as u64).to_le_bytes());
            let expected = 1usize + usize::from(has_prev);
            if mem_proof.val_me_claims.len() != expected {
                return Err(PiCcsError::ProtocolError(format!(
                    "Twist(val) claim count mismatch (have {}, expected {})",
                    mem_proof.val_me_claims.len(),
                    expected
                )));
            }
            // Disabled: once Π_RLC(k=1) applies sampled ρ (non-identity),
            // val-lane parents are no longer compatible with main-lane Z_split reuse.
            let shared_val_lane_child_cs: Option<Vec<Cmt>> = None;

            for (claim_idx, me) in mem_proof.val_me_claims.iter().enumerate() {
                let (wit, ctx) = match claim_idx {
                    0 => (&mcs_wit.Z, "cpu"),
                    1 => {
                        let prev = prev_step
                            .ok_or_else(|| PiCcsError::ProtocolError("missing prev_step for r_val claim".into()))?;
                        (&prev.mcs.1.Z, "cpu_prev")
                    }
                    _ => {
                        return Err(PiCcsError::ProtocolError(
                            "unexpected extra r_val ME claim in shared-bus mode".into(),
                        ));
                    }
                };
                tr.append_message(b"fold/val_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                tr.append_message(b"fold/val_lane_claim_ctx", ctx.as_bytes());

                // Reuse main-lane split/commit artifacts for the current-step shared-bus
                // val lane so we don't pay an extra full split+commit.
                if claim_idx == 0 {
                    if let Some(child_cs) = shared_val_lane_child_cs.as_ref() {
                        let n_lane = 1usize
                            .checked_shl(me.r.len() as u32)
                            .ok_or_else(|| PiCcsError::InvalidInput("val-lane r dimension overflow".into()))?;
                        let mut s_lane = s.clone();
                        s_lane.n = n_lane;
                        bind_rlc_inputs(tr, RlcLane::Val, step_idx, core::slice::from_ref(me))?;
                        let rlc_rhos = ccs::sample_rot_rhos_n_typed(tr, params, &ring, 1)?;
                        let mut rlc_parent = ccs::rlc_public(
                            &s_lane,
                            params,
                            &rlc_rhos,
                            core::slice::from_ref(me),
                            mixers.mix_rhos_commits,
                            ell_d,
                        )?;
                        let (mut dec_children, ok_y, ok_x, ok_c) = ccs::dec_children_with_commit_cached(
                            mode.clone(),
                            &s_lane,
                            params,
                            &rlc_parent,
                            &Z_split,
                            ell_d,
                            child_cs,
                            mixers.combine_b_pows,
                            ccs_sparse_cache.as_deref(),
                        );
                        if !(ok_y && ok_x && ok_c) {
                            return Err(PiCcsError::ProtocolError(format!(
                                "DEC(val fast-path) public check failed at step {} claim_idx={} (y={}, X={}, c={}, me.r.len()={}, parent.r.len()={}, s_lane.n={})",
                                step_idx,
                                claim_idx,
                                ok_y,
                                ok_x,
                                ok_c,
                                me.r.len(),
                                rlc_parent.r.len(),
                                s_lane.n
                            )));
                        }
                        if cpu_bus.bus_cols > 0 {
                            let core_t = s.t();
                            let want_len = core_t
                                .checked_add(cpu_bus.bus_cols)
                                .ok_or_else(|| PiCcsError::InvalidInput("core_t + bus_cols overflow".into()))?;
                            let parent_has_prefilled_bus =
                                rlc_parent.y_ring.len() > core_t || rlc_parent.ct.len() > core_t;
                            let parent_supports_bus_point = crate::memory_sidecar::cpu_bus::point_covers_bus_time_rows(
                                &cpu_bus,
                                rlc_parent.r.as_slice(),
                            )?;
                            if !parent_has_prefilled_bus && !parent_supports_bus_point {
                                crate::memory_sidecar::cpu_bus::append_zero_bus_openings_to_me_instance(
                                    params,
                                    &cpu_bus,
                                    core_t,
                                    &mut rlc_parent,
                                )?;
                                for child in dec_children.iter_mut() {
                                    crate::memory_sidecar::cpu_bus::append_zero_bus_openings_to_me_instance(
                                        params, &cpu_bus, core_t, child,
                                    )?;
                                }
                            } else if wit.cols() == cpu_bus.m && !parent_has_prefilled_bus {
                                crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                                    params,
                                    &cpu_bus,
                                    core_t,
                                    wit,
                                    &mut rlc_parent,
                                )?;
                                for (child, zi) in dec_children.iter_mut().zip(Z_split.iter()) {
                                    let child_supports_bus_point =
                                        crate::memory_sidecar::cpu_bus::point_covers_bus_time_rows(
                                            &cpu_bus,
                                            child.r.as_slice(),
                                        )?;
                                    if child_supports_bus_point {
                                        crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                                            params, &cpu_bus, core_t, zi, child,
                                        )?;
                                    } else {
                                        crate::memory_sidecar::cpu_bus::append_zero_bus_openings_to_me_instance(
                                            params, &cpu_bus, core_t, child,
                                        )?;
                                    }
                                }
                            } else {
                                if rlc_parent.y_ring.len() == core_t && rlc_parent.ct.len() == core_t {
                                    crate::memory_sidecar::cpu_bus::append_zero_bus_openings_to_me_instance(
                                        params,
                                        &cpu_bus,
                                        core_t,
                                        &mut rlc_parent,
                                    )?;
                                } else if rlc_parent.y_ring.len() != want_len || rlc_parent.ct.len() != want_len {
                                    return Err(PiCcsError::ProtocolError(format!(
                                        "step {}: val-lane non-physical bus path expects exact parent y/ct len {} (got y.len()={}, ct.len()={})",
                                        step_idx,
                                        want_len,
                                        rlc_parent.y_ring.len(),
                                        rlc_parent.ct.len()
                                    )));
                                }
                                let y_pad = (params.d as usize).next_power_of_two();
                                for (child_idx, child) in dec_children.iter_mut().enumerate() {
                                    if child.y_ring.len() < core_t || child.ct.len() < core_t {
                                        return Err(PiCcsError::ProtocolError(format!(
                                            "step {}: val-lane non-physical bus path expects child y/ct len >= core_t={} (got y.len()={}, ct.len()={})",
                                            step_idx,
                                            core_t,
                                            child.y_ring.len(),
                                            child.ct.len()
                                        )));
                                    }
                                    child.y_ring.truncate(core_t);
                                    child.ct.truncate(core_t);
                                    // As in the main lane, propagated non-physical bus openings are
                                    // metadata-only: carry them on child 0 and keep sibling children zero.
                                    for col_id in 0..cpu_bus.bus_cols {
                                        if child_idx == 0 {
                                            child
                                                .y_ring
                                                .push(rlc_parent.y_ring[core_t + col_id].clone());
                                            child.ct.push(rlc_parent.ct[core_t + col_id]);
                                        } else {
                                            child.y_ring.push(vec![K::ZERO; y_pad]);
                                            child.ct.push(K::ZERO);
                                        }
                                    }
                                    if child_idx > 0 {
                                        debug_assert!(
                                            child.y_ring[core_t..]
                                                .iter()
                                                .all(|row| row.iter().all(|v| *v == K::ZERO))
                                                && child.ct[core_t..].iter().all(|v| *v == K::ZERO),
                                            "non-primary val-lane DEC children must keep propagated metadata openings at zero"
                                        );
                                    }
                                    if child.y_ring.len() != want_len || child.ct.len() != want_len {
                                        return Err(PiCcsError::ProtocolError(format!(
                                            "step {}: val-lane child suffix-length drift (child y/ct={}/{}, expected={})",
                                            step_idx,
                                            child.y_ring.len(),
                                            child.ct.len(),
                                            want_len
                                        )));
                                    }
                                }
                            }
                        }
                        if collect_val_lane_wits {
                            val_lane_wits.extend(Z_split.iter().cloned());
                        }
                        val_fold.push(RlcDecProof {
                            rlc_rhos,
                            rlc_parent,
                            dec_children,
                        });
                        continue;
                    }
                }

                let (proof, mut Z_split_val) = prove_rlc_dec_lane(
                    &mode,
                    RlcLane::Val,
                    tr,
                    params,
                    &s,
                    ccs_sparse_cache.as_deref(),
                    Some(&cpu_bus),
                    &ring,
                    ell_d,
                    k_dec,
                    step_idx,
                    None,
                    core::slice::from_ref(me),
                    core::slice::from_ref(&wit),
                    collect_val_lane_wits,
                    l,
                    mixers,
                )?;
                if collect_val_lane_wits {
                    val_lane_wits.extend(Z_split_val.drain(..));
                }
                val_fold.push(proof);
            }
        }

        // Additional WB folding lane(s): CPU ME openings used by wb/booleanity stage.
        let mut wb_fold: Vec<RlcDecProof> = Vec::new();
        if !mem_proof.wb_me_claims.is_empty() {
            let trace = Rv32TraceLayout::new();
            let wb_cols = crate::memory_sidecar::memory::rv32_trace_wb_columns(&trace);
            let core_t = s.t();
            tr.append_message(b"fold/wb_lane_start", &(step_idx as u64).to_le_bytes());
            for (claim_idx, me) in mem_proof.wb_me_claims.iter().enumerate() {
                let n_lane = 1usize
                    .checked_shl(me.r.len() as u32)
                    .ok_or_else(|| PiCcsError::InvalidInput("wb-lane r dimension overflow".into()))?;
                let mut s_lane = s.clone();
                s_lane.n = n_lane;
                tr.append_message(b"fold/wb_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                bind_rlc_inputs(tr, RlcLane::Val, step_idx, core::slice::from_ref(me))?;
                let rlc_rhos = ccs::sample_rot_rhos_n_typed(tr, params, &ring, 1)?;
                let rlc_parent = ccs::rlc_public(
                    &s_lane,
                    params,
                    &rlc_rhos,
                    core::slice::from_ref(me),
                    mixers.mix_rhos_commits,
                    ell_d,
                )?;
                let rlc_rho_mats = ccs::rot_rhos_to_mats(&rlc_rhos);
                let (_, z_mix) = neo_reductions::optimized_engine::rlc_reduction_optimized_with_commit_mix(
                    &s_lane,
                    params,
                    &rlc_rho_mats,
                    core::slice::from_ref(me),
                    &[&mcs_wit.Z],
                    ell_d,
                    mixers.mix_rhos_commits,
                );
                let k_dec_lane = core::cmp::max(k_dec, required_dec_digits_for_matrix(params, &z_mix)?);
                let (dec_wits, digit_nonzero) = ccs::split_b_matrix_k_with_nonzero_flags(&z_mix, k_dec_lane, params.b)?;
                let zero_c = Cmt::zeros(mcs_inst.c.d, mcs_inst.c.kappa);
                let mut child_cs: Vec<Cmt> = vec![zero_c.clone(); dec_wits.len()];
                let nonzero_idx: Vec<usize> = digit_nonzero
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &nz)| nz.then_some(idx))
                    .collect();
                if !nonzero_idx.is_empty() {
                    let mats: Vec<&Mat<F>> = nonzero_idx.iter().map(|&idx| &dec_wits[idx]).collect();
                    let commits = l.commit_many(&mats);
                    if commits.len() != mats.len() {
                        return Err(PiCcsError::ProtocolError(format!(
                            "WB DEC commit_many returned {} commitments for {} matrices",
                            commits.len(),
                            mats.len()
                        )));
                    }
                    for (pos, &idx) in nonzero_idx.iter().enumerate() {
                        child_cs[idx] = commits[pos].clone();
                    }
                }
                let (mut dec_children, ok_y, ok_x, ok_c) = ccs::dec_children_with_commit_cached(
                    mode.clone(),
                    &s_lane,
                    params,
                    &rlc_parent,
                    &dec_wits,
                    ell_d,
                    &child_cs,
                    mixers.combine_b_pows,
                    ccs_sparse_cache.as_deref(),
                );
                if !(ok_y && ok_x && ok_c) {
                    return Err(PiCcsError::ProtocolError(format!(
                        "DEC(wb lane) public check failed at step {} claim_idx={} (y={}, X={}, c={}, me.r.len()={}, parent.r.len()={}, s_lane.n={})",
                        step_idx,
                        claim_idx,
                        ok_y,
                        ok_x,
                        ok_c,
                        me.r.len(),
                        rlc_parent.r.len(),
                        s_lane.n
                    )));
                }
                if dec_children.len() != dec_wits.len() {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: WB fold requires materialized DEC witnesses (children={}, wits={})",
                        step_idx,
                        dec_children.len(),
                        dec_wits.len()
                    )));
                }
                let want_len = core_t
                    .checked_add(wb_cols.len())
                    .ok_or_else(|| PiCcsError::InvalidInput("core_t + wb_cols overflow".into()))?;
                if rlc_parent.y_ring.len() != want_len || rlc_parent.ct.len() != want_len {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: WB fold expects exact parent y/ct len {} (got y.len()={}, ct.len()={})",
                        step_idx,
                        want_len,
                        rlc_parent.y_ring.len(),
                        rlc_parent.ct.len()
                    )));
                }
                let y_pad = (params.d as usize).next_power_of_two();
                for (child_idx, child) in dec_children.iter_mut().enumerate() {
                    if child.y_ring.len() < core_t || child.ct.len() < core_t {
                        return Err(PiCcsError::ProtocolError(format!(
                            "step {}: WB fold expects child y/ct len >= core_t={} (got y.len()={}, ct.len()={})",
                            step_idx,
                            core_t,
                            child.y_ring.len(),
                            child.ct.len()
                        )));
                    }
                    child.y_ring.truncate(core_t);
                    child.ct.truncate(core_t);
                    for open_idx in 0..wb_cols.len() {
                        if child_idx == 0 {
                            child
                                .y_ring
                                .push(rlc_parent.y_ring[core_t + open_idx].clone());
                            child.ct.push(rlc_parent.ct[core_t + open_idx]);
                        } else {
                            child.y_ring.push(vec![K::ZERO; y_pad]);
                            child.ct.push(K::ZERO);
                        }
                    }
                    if child.y_ring.len() != want_len || child.ct.len() != want_len {
                        return Err(PiCcsError::ProtocolError(format!(
                            "step {}: WB fold child suffix-length drift (child y/ct={}/{}, expected={})",
                            step_idx,
                            child.y_ring.len(),
                            child.ct.len(),
                            want_len
                        )));
                    }
                }
                if collect_val_lane_wits {
                    val_lane_wits.extend(dec_wits.iter().cloned());
                }
                wb_fold.push(RlcDecProof {
                    rlc_rhos,
                    rlc_parent,
                    dec_children,
                });
            }
        }

        // Additional WP folding lane(s): CPU ME openings used by wp/quiescence stage.
        let mut wp_fold: Vec<RlcDecProof> = Vec::new();
        if !mem_proof.wp_me_claims.is_empty() {
            let trace = Rv32TraceLayout::new();
            let t_len = crate::memory_sidecar::memory::infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
            let mut wp_open_cols = crate::memory_sidecar::memory::rv32_trace_wp_opening_columns(&trace);
            if control_required {
                wp_open_cols.extend(crate::memory_sidecar::memory::rv32_trace_control_extra_opening_columns(
                    &trace,
                ));
            }
            if decode_required {
                let decode_layout = Rv32DecodeSidecarLayout::new();
                let (_decode_open_cols, decode_lut_slots) =
                    crate::memory_sidecar::memory::resolve_shared_decode_lookup_lut_indices(step, &decode_layout)?;
                let bus = crate::memory_sidecar::memory::build_bus_layout_for_step_witness(step, t_len)?;
                if bus.shout_cols.len() != step.lut_instances.len() {
                    return Err(PiCcsError::ProtocolError(
                        "W2(shared): bus layout shout lane count drift in WP fold".into(),
                    ));
                }
                if step.time_columns.t != t_len || step.time_columns.cpu_cols.is_empty() {
                    return Err(PiCcsError::ProtocolError(format!(
                        "W2(shared): canonical time CPU columns required in WP fold (time_t={}, cpu_cols={}, expected_t={t_len})",
                        step.time_columns.t,
                        step.time_columns.cpu_cols.len()
                    )));
                }
                let cpu_cols_len = step.time_columns.cpu_cols.len();
                let mem_cols_len = step.time_columns.mem_cols.len();
                let expected_logical_cols = cpu_cols_len.checked_add(mem_cols_len).ok_or_else(|| {
                    PiCcsError::InvalidInput("W2(shared): cpu_cols + mem_cols overflow in WP fold".into())
                })?;
                if step.time_columns.col_ids.len() != expected_logical_cols {
                    return Err(PiCcsError::ProtocolError(format!(
                        "W2(shared): time column id table mismatch in WP fold (col_ids={}, cpu_cols={}, mem_cols={})",
                        step.time_columns.col_ids.len(),
                        cpu_cols_len,
                        mem_cols_len
                    )));
                }
                for &(lut_idx, val_slot) in decode_lut_slots.iter() {
                    let inst_cols = bus.shout_cols.get(lut_idx).ok_or_else(|| {
                        PiCcsError::ProtocolError(
                            "W2(shared): missing shout cols for decode lookup table in WP fold".into(),
                        )
                    })?;
                    let lane0 = inst_cols.lanes.get(0).ok_or_else(|| {
                        PiCcsError::ProtocolError(
                            "W2(shared): expected one shout lane for decode lookup table in WP fold".into(),
                        )
                    })?;
                    let val_col = lane0.vals.get(val_slot).copied().ok_or_else(|| {
                        PiCcsError::ProtocolError(format!(
                            "W2(shared): decode val_slot={} out of range for lut_idx={} in WP fold (n_vals={})",
                            val_slot,
                            lut_idx,
                            lane0.vals.len()
                        ))
                    })?;
                    let logical_idx = cpu_cols_len.checked_add(val_col).ok_or_else(|| {
                        PiCcsError::InvalidInput("W2(shared): cpu_cols + lane primary value overflow in WP fold".into())
                    })?;
                    let logical_col = step
                        .time_columns
                        .col_ids
                        .get(logical_idx)
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError(format!(
                                "W2(shared): missing logical id for mem local col {} in WP fold",
                                val_col
                            ))
                        })?;
                    wp_open_cols.push(logical_col);
                }
            }
            if width_required {
                wp_open_cols.extend(crate::memory_sidecar::memory::width_lookup_bus_val_cols_witness(
                    step, t_len,
                )?);
            }
            let core_t = s.t();
            tr.append_message(b"fold/wp_lane_start", &(step_idx as u64).to_le_bytes());
            for (claim_idx, me) in mem_proof.wp_me_claims.iter().enumerate() {
                let n_lane = 1usize
                    .checked_shl(me.r.len() as u32)
                    .ok_or_else(|| PiCcsError::InvalidInput("wp-lane r dimension overflow".into()))?;
                let mut s_lane = s.clone();
                s_lane.n = n_lane;
                tr.append_message(b"fold/wp_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                bind_rlc_inputs(tr, RlcLane::Val, step_idx, core::slice::from_ref(me))?;
                let rlc_rhos = ccs::sample_rot_rhos_n_typed(tr, params, &ring, 1)?;
                let rlc_parent = ccs::rlc_public(
                    &s_lane,
                    params,
                    &rlc_rhos,
                    core::slice::from_ref(me),
                    mixers.mix_rhos_commits,
                    ell_d,
                )?;
                let rlc_rho_mats = ccs::rot_rhos_to_mats(&rlc_rhos);
                let (_, z_mix) = neo_reductions::optimized_engine::rlc_reduction_optimized_with_commit_mix(
                    &s_lane,
                    params,
                    &rlc_rho_mats,
                    core::slice::from_ref(me),
                    &[&mcs_wit.Z],
                    ell_d,
                    mixers.mix_rhos_commits,
                );
                let k_dec_lane = core::cmp::max(k_dec, required_dec_digits_for_matrix(params, &z_mix)?);
                let (dec_wits, digit_nonzero) = ccs::split_b_matrix_k_with_nonzero_flags(&z_mix, k_dec_lane, params.b)?;
                let zero_c = Cmt::zeros(mcs_inst.c.d, mcs_inst.c.kappa);
                let mut child_cs: Vec<Cmt> = vec![zero_c.clone(); dec_wits.len()];
                let nonzero_idx: Vec<usize> = digit_nonzero
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &nz)| nz.then_some(idx))
                    .collect();
                if !nonzero_idx.is_empty() {
                    let mats: Vec<&Mat<F>> = nonzero_idx.iter().map(|&idx| &dec_wits[idx]).collect();
                    let commits = l.commit_many(&mats);
                    if commits.len() != mats.len() {
                        return Err(PiCcsError::ProtocolError(format!(
                            "WP DEC commit_many returned {} commitments for {} matrices",
                            commits.len(),
                            mats.len()
                        )));
                    }
                    for (pos, &idx) in nonzero_idx.iter().enumerate() {
                        child_cs[idx] = commits[pos].clone();
                    }
                }
                let (mut dec_children, ok_y, ok_x, ok_c) = ccs::dec_children_with_commit_cached(
                    mode.clone(),
                    &s_lane,
                    params,
                    &rlc_parent,
                    &dec_wits,
                    ell_d,
                    &child_cs,
                    mixers.combine_b_pows,
                    ccs_sparse_cache.as_deref(),
                );
                if !(ok_y && ok_x && ok_c) {
                    return Err(PiCcsError::ProtocolError(format!(
                        "DEC(wp lane) public check failed at step {} claim_idx={} (y={}, X={}, c={}, me.r.len()={}, parent.r.len()={}, s_lane.n={})",
                        step_idx,
                        claim_idx,
                        ok_y,
                        ok_x,
                        ok_c,
                        me.r.len(),
                        rlc_parent.r.len(),
                        s_lane.n
                    )));
                }
                if dec_children.len() != dec_wits.len() {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: WP fold requires materialized DEC witnesses (children={}, wits={})",
                        step_idx,
                        dec_children.len(),
                        dec_wits.len()
                    )));
                }
                let want_len = core_t
                    .checked_add(wp_open_cols.len())
                    .ok_or_else(|| PiCcsError::InvalidInput("core_t + wp_open_cols overflow".into()))?;
                if rlc_parent.y_ring.len() != want_len || rlc_parent.ct.len() != want_len {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: WP fold expects exact parent y/ct len {} (got y.len()={}, ct.len()={})",
                        step_idx,
                        want_len,
                        rlc_parent.y_ring.len(),
                        rlc_parent.ct.len()
                    )));
                }
                let y_pad = (params.d as usize).next_power_of_two();
                for (child_idx, child) in dec_children.iter_mut().enumerate() {
                    if child.y_ring.len() < core_t || child.ct.len() < core_t {
                        return Err(PiCcsError::ProtocolError(format!(
                            "step {}: WP fold expects child y/ct len >= core_t={} (got y.len()={}, ct.len()={})",
                            step_idx,
                            core_t,
                            child.y_ring.len(),
                            child.ct.len()
                        )));
                    }
                    child.y_ring.truncate(core_t);
                    child.ct.truncate(core_t);
                    for open_idx in 0..wp_open_cols.len() {
                        if child_idx == 0 {
                            child
                                .y_ring
                                .push(rlc_parent.y_ring[core_t + open_idx].clone());
                            child.ct.push(rlc_parent.ct[core_t + open_idx]);
                        } else {
                            child.y_ring.push(vec![K::ZERO; y_pad]);
                            child.ct.push(K::ZERO);
                        }
                    }
                    if child.y_ring.len() != want_len || child.ct.len() != want_len {
                        return Err(PiCcsError::ProtocolError(format!(
                            "step {}: WP fold child suffix-length drift (child y/ct={}/{}, expected={})",
                            step_idx,
                            child.y_ring.len(),
                            child.ct.len(),
                            want_len
                        )));
                    }
                }
                if collect_val_lane_wits {
                    val_lane_wits.extend(dec_wits.iter().cloned());
                }
                wp_fold.push(RlcDecProof {
                    rlc_rhos,
                    rlc_parent,
                    dec_children,
                });
            }
        }

        let poseidon_fold_lanes = prove_poseidon_fold_lanes(
            &mode,
            tr,
            params,
            &s,
            ccs_sparse_cache.as_deref(),
            &ring,
            ell_d,
            step_idx,
            &mem_proof,
            poseidon_cycle_wits.as_ref(),
            poseidon_cycle_open_specs.as_ref(),
            poseidon_local_wits.as_ref(),
            poseidon_local_open_specs.as_ref(),
            l,
            mixers,
        )?;
        let poseidon_cycle_fold = poseidon_fold_lanes.cycle_fold;
        let poseidon_local_fold = poseidon_fold_lanes.local_fold;

        accumulator = children.clone();
        accumulator_wit = if want_main_wits { Z_split } else { Vec::new() };

        let fold_openings = {
            let mut out = Vec::new();
            if cpu_bus.bus_cols > 0 {
                let cpu_cols_len = step.time_columns.cpu_cols.len();
                let mem_cols_len = step.time_columns.mem_cols.len();
                let expected_logical_cols = cpu_cols_len.checked_add(mem_cols_len).ok_or_else(|| {
                    PiCcsError::InvalidInput("named openings bus: cpu_cols + mem_cols overflow".into())
                })?;
                let has_logical_bus_ids =
                    mem_cols_len == cpu_bus.bus_cols && step.time_columns.col_ids.len() == expected_logical_cols;
                if !has_logical_bus_ids {
                    return Err(PiCcsError::ProtocolError(format!(
                        "named openings bus: canonical committed mode requires logical bus ids (col_ids={}, cpu_cols={}, mem_cols={}, bus_cols={})",
                        step.time_columns.col_ids.len(),
                        cpu_cols_len,
                        mem_cols_len,
                        cpu_bus.bus_cols
                    )));
                }
                let col_ids: Vec<usize> = step.time_columns.col_ids[cpu_cols_len..].to_vec();
                let can_use_time_mem_cols =
                    step.time_columns.t == cpu_bus.chunk_size && step.time_columns.mem_cols.len() == cpu_bus.bus_cols;
                if !can_use_time_mem_cols {
                    return Err(PiCcsError::ProtocolError(format!(
                        "named openings bus: canonical time mem columns are required (time_t={}, mem_cols={}, expected chunk_size={}, bus_cols={})",
                        step.time_columns.t,
                        step.time_columns.mem_cols.len(),
                        cpu_bus.chunk_size,
                        cpu_bus.bus_cols
                    )));
                }
                if !has_committed_time_mem {
                    return Err(PiCcsError::ProtocolError(
                        "named openings bus: canonical time-column path requires committed time mem columns".into(),
                    ));
                }
                let open_map = crate::memory_sidecar::cpu_bus::shared_bus_openings_from_time_columns_at_point(
                    &cpu_bus,
                    &step.time_columns.mem_cols,
                    &r_time,
                    "named openings bus",
                )?;
                let mut evals = Vec::with_capacity(col_ids.len());
                for (mem_local_col, _) in col_ids.iter().enumerate() {
                    let v = open_map.get(&mem_local_col).copied().ok_or_else(|| {
                        PiCcsError::ProtocolError(format!(
                            "named openings bus: missing mem local col_id={mem_local_col}"
                        ))
                    })?;
                    evals.push(v);
                }
                out.push(crate::shard_proof_types::TimePointOpening {
                    point: r_time.clone(),
                    col_ids: col_ids.clone(),
                    evals,
                    source: crate::shard_proof_types::TimeOpeningSource::CommittedOpening,
                });

                // Export current-step r_val bus openings as named openings so the verifier can
                // bind Twist val-lane checks to committed openings instead of ME tail offsets.
                if let Some(cpu_me_val_cur) = mem_proof.val_me_claims.first() {
                    if cpu_me_val_cur.r.as_slice() != r_time.as_slice() {
                        if !has_committed_time_mem {
                            return Err(PiCcsError::ProtocolError(
                                "named openings bus/val: canonical time-column path requires committed time mem columns"
                                    .into(),
                            ));
                        }
                        let open_map = crate::memory_sidecar::cpu_bus::shared_bus_openings_from_time_columns_at_point(
                            &cpu_bus,
                            &step.time_columns.mem_cols,
                            cpu_me_val_cur.r.as_slice(),
                            "named openings bus/val",
                        )?;
                        let mut evals = Vec::with_capacity(col_ids.len());
                        for (mem_local_col, _) in col_ids.iter().enumerate() {
                            let v = open_map.get(&mem_local_col).copied().ok_or_else(|| {
                                PiCcsError::ProtocolError(format!(
                                    "named openings bus/val: missing mem local col_id={mem_local_col}"
                                ))
                            })?;
                            evals.push(v);
                        }
                        out.push(crate::shard_proof_types::TimePointOpening {
                            point: cpu_me_val_cur.r.clone(),
                            col_ids,
                            evals,
                            source: crate::shard_proof_types::TimeOpeningSource::CommittedOpening,
                        });
                    }
                }
            }
            if !named_trace_col_ids.is_empty() {
                let can_use_time_cpu_cols = step.time_columns.t > 0
                    && !step.time_columns.cpu_cols.is_empty()
                    && named_trace_col_ids
                        .iter()
                        .all(|&col_id| col_id < step.time_columns.cpu_cols.len());
                if !can_use_time_cpu_cols {
                    return Err(PiCcsError::ProtocolError(format!(
                        "named openings trace: canonical time cpu columns are required (time_t={}, cpu_cols={}, trace_cols={})",
                        step.time_columns.t,
                        step.time_columns.cpu_cols.len(),
                        named_trace_col_ids.len()
                    )));
                }
                if !has_committed_time_cpu {
                    return Err(PiCcsError::ProtocolError(
                        "named openings trace: canonical time-column path requires committed time cpu columns".into(),
                    ));
                }
                let trace_map = crate::memory_sidecar::cpu_bus::time_columns_openings_from_time_columns_at_point(
                    mcs_inst.m_in,
                    step.time_columns.t,
                    &step.time_columns.cpu_cols,
                    &named_trace_col_ids,
                    &r_time,
                    "named openings trace",
                )?;
                let mut evals = Vec::with_capacity(named_trace_col_ids.len());
                for &col_id in named_trace_col_ids.iter() {
                    let v = trace_map.get(&col_id).copied().ok_or_else(|| {
                        PiCcsError::ProtocolError(format!("named openings trace: missing col_id={col_id}"))
                    })?;
                    evals.push(v);
                }
                out.push(crate::shard_proof_types::TimePointOpening {
                    point: r_time.clone(),
                    col_ids: named_trace_col_ids.clone(),
                    evals,
                    source: crate::shard_proof_types::TimeOpeningSource::CommittedOpening,
                });
            }
            if let Some(wb_me) = mem_proof.wb_me_claims.first() {
                let trace = Rv32TraceLayout::new();
                let wb_cols = crate::memory_sidecar::memory::rv32_trace_wb_columns(&trace);
                let can_use_time_cpu_cols = step.time_columns.t > 0
                    && !step.time_columns.cpu_cols.is_empty()
                    && wb_cols
                        .iter()
                        .all(|&col_id| col_id < step.time_columns.cpu_cols.len());
                if mcs_inst.m_in == 5 && !can_use_time_cpu_cols {
                    return Err(PiCcsError::ProtocolError(format!(
                        "named openings wb: canonical Route-A requires time cpu columns (time_t={}, cpu_cols={}, wb_cols={})",
                        step.time_columns.t,
                        step.time_columns.cpu_cols.len(),
                        wb_cols.len()
                    )));
                }
                if !can_use_time_cpu_cols {
                    return Err(PiCcsError::ProtocolError(format!(
                        "named openings wb: canonical time cpu columns are required (time_t={}, cpu_cols={}, wb_cols={})",
                        step.time_columns.t,
                        step.time_columns.cpu_cols.len(),
                        wb_cols.len()
                    )));
                }
                if !has_committed_time_cpu {
                    return Err(PiCcsError::ProtocolError(
                        "named openings wb: canonical time-column path requires committed time cpu columns".into(),
                    ));
                }
                let trace_map = crate::memory_sidecar::cpu_bus::time_columns_openings_from_time_columns_at_point(
                    mcs_inst.m_in,
                    step.time_columns.t,
                    &step.time_columns.cpu_cols,
                    &wb_cols,
                    wb_me.r.as_slice(),
                    "named openings wb",
                )?;
                let mut evals = Vec::with_capacity(wb_cols.len());
                for &col_id in wb_cols.iter() {
                    let v = trace_map.get(&col_id).copied().ok_or_else(|| {
                        PiCcsError::ProtocolError(format!("named openings wb: missing col_id={col_id}"))
                    })?;
                    evals.push(v);
                }
                out.push(crate::shard_proof_types::TimePointOpening {
                    point: wb_me.r.clone(),
                    col_ids: wb_cols,
                    evals,
                    source: crate::shard_proof_types::TimeOpeningSource::CommittedOpening,
                });
            }
            if let Some(wp_me) = mem_proof.wp_me_claims.first() {
                let trace = Rv32TraceLayout::new();
                let mut wp_cols = crate::memory_sidecar::memory::rv32_trace_wp_opening_columns(&trace);
                if control_required {
                    wp_cols.extend(crate::memory_sidecar::memory::rv32_trace_control_extra_opening_columns(
                        &trace,
                    ));
                }
                let mut seen_wp_cols = std::collections::BTreeSet::new();
                wp_cols.retain(|col_id| seen_wp_cols.insert(*col_id));
                let can_use_time_cpu_cols = step.time_columns.t > 0
                    && !step.time_columns.cpu_cols.is_empty()
                    && wp_cols
                        .iter()
                        .all(|&col_id| col_id < step.time_columns.cpu_cols.len());
                if mcs_inst.m_in == 5 && !can_use_time_cpu_cols {
                    return Err(PiCcsError::ProtocolError(format!(
                        "named openings wp: canonical Route-A requires time cpu columns (time_t={}, cpu_cols={}, wp_cols={})",
                        step.time_columns.t,
                        step.time_columns.cpu_cols.len(),
                        wp_cols.len()
                    )));
                }
                if !can_use_time_cpu_cols {
                    return Err(PiCcsError::ProtocolError(format!(
                        "named openings wp: canonical time cpu columns are required (time_t={}, cpu_cols={}, wp_cols={})",
                        step.time_columns.t,
                        step.time_columns.cpu_cols.len(),
                        wp_cols.len()
                    )));
                }
                if !has_committed_time_cpu {
                    return Err(PiCcsError::ProtocolError(
                        "named openings wp: canonical time-column path requires committed time cpu columns".into(),
                    ));
                }
                let trace_map = crate::memory_sidecar::cpu_bus::time_columns_openings_from_time_columns_at_point(
                    mcs_inst.m_in,
                    step.time_columns.t,
                    &step.time_columns.cpu_cols,
                    &wp_cols,
                    wp_me.r.as_slice(),
                    "named openings wp",
                )?;
                let mut evals = Vec::with_capacity(wp_cols.len());
                for &col_id in wp_cols.iter() {
                    let v = trace_map.get(&col_id).copied().ok_or_else(|| {
                        PiCcsError::ProtocolError(format!("named openings wp: missing col_id={col_id}"))
                    })?;
                    evals.push(v);
                }
                out.push(crate::shard_proof_types::TimePointOpening {
                    point: wp_me.r.clone(),
                    col_ids: wp_cols,
                    evals,
                    source: crate::shard_proof_types::TimeOpeningSource::CommittedOpening,
                });
            }
            out
        };
        let opening_proofs = {
            let mut out = Vec::new();
            let logical_col_pos = crate::time_opening::me_adapter::build_logical_col_pos(&step.time_columns.col_ids)?;
            let cpu_cols_len = time_cpu_commitments.len();
            struct EncodedTimeColCache {
                z_col: neo_ccs::Mat<F>,
                row_nz: Vec<Vec<(usize, F)>>,
            }
            let mut z_col_cache = std::collections::BTreeMap::<usize, EncodedTimeColCache>::new();
            let mut point_weight_cache: Vec<(crate::shard_proof_types::OpeningDomain, Vec<K>, Vec<F>, Vec<F>)> =
                Vec::new();
            for opening in fold_openings.iter() {
                if opening.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
                    continue;
                }
                if opening.col_ids.len() != opening.evals.len() {
                    return Err(PiCcsError::ProtocolError(
                        "time/opening proof build: malformed opening col_ids/evals length mismatch".into(),
                    ));
                }
                let mut pairs: Vec<(usize, K)> = opening
                    .col_ids
                    .iter()
                    .copied()
                    .zip(opening.evals.iter().copied())
                    .collect();
                pairs.sort_unstable_by_key(|(col_id, _)| *col_id);
                if pairs.windows(2).any(|w| w[0].0 == w[1].0) {
                    return Err(PiCcsError::ProtocolError(
                        "time/opening proof build: duplicate col_ids in committed opening".into(),
                    ));
                }
                let mut col_ids = Vec::with_capacity(pairs.len());
                let mut evals = Vec::with_capacity(pairs.len());
                let sorted_col_ids: Vec<usize> = pairs.iter().map(|(col_id, _)| *col_id).collect();
                let domain = crate::time_opening::me_adapter::domain_for_col_ids(
                    sorted_col_ids.as_slice(),
                    &logical_col_pos,
                    cpu_cols_len,
                )?;
                let cache_idx = if let Some(idx) = point_weight_cache
                    .iter()
                    .position(|(d, p, _, _)| *d == domain && p.as_slice() == opening.point.as_slice())
                {
                    idx
                } else {
                    let point_chi = crate::time_opening::me_adapter::build_small_chi_table(opening.point.as_slice())?;
                    let point_row_weights = match domain {
                        crate::shard_proof_types::OpeningDomain::Cpu => {
                            crate::time_opening::me_adapter::cpu_time_row_weights(
                                opening.point.as_slice(),
                                step.mcs.0.m_in,
                                step.time_columns.t,
                                point_chi.as_deref(),
                            )?
                        }
                        crate::shard_proof_types::OpeningDomain::Mem => {
                            crate::time_opening::me_adapter::mem_time_row_weights(
                                opening.point.as_slice(),
                                &cpu_bus,
                                point_chi.as_deref(),
                            )?
                        }
                    };
                    let (point_row_weights_re, point_row_weights_im) =
                        crate::time_opening::me_adapter::split_row_weight_coeffs(point_row_weights.as_slice());
                    point_weight_cache.push((
                        domain,
                        opening.point.clone(),
                        point_row_weights_re,
                        point_row_weights_im,
                    ));
                    point_weight_cache.len() - 1
                };
                let (_, _, point_row_weights_re, point_row_weights_im) = &point_weight_cache[cache_idx];
                let mut digit_evals = Vec::with_capacity(pairs.len());
                for (col_id, eval) in pairs.into_iter() {
                    col_ids.push(col_id);
                    evals.push(eval);
                    if !z_col_cache.contains_key(&col_id) {
                        let abs_pos = logical_col_pos.get(&col_id).copied().ok_or_else(|| {
                            PiCcsError::ProtocolError(format!(
                                "time/opening proof build: logical col_id={} missing",
                                col_id
                            ))
                        })?;
                        let col = if abs_pos < cpu_cols_len {
                            step.time_columns.cpu_cols.get(abs_pos).ok_or_else(|| {
                                PiCcsError::ProtocolError(format!(
                                    "time/opening proof build: cpu column index {} out of range",
                                    abs_pos
                                ))
                            })?
                        } else {
                            let mem_idx = abs_pos - cpu_cols_len;
                            step.time_columns.mem_cols.get(mem_idx).ok_or_else(|| {
                                PiCcsError::ProtocolError(format!(
                                    "time/opening proof build: mem column index {} out of range",
                                    mem_idx
                                ))
                            })?
                        };
                        let z_col = neo_memory::ajtai::encode_vector_balanced_to_mat_with_base(
                            params,
                            col,
                            crate::time_opening::STAGE8_TIME_DECOMP_BASE,
                        );
                        let row_nz = crate::time_opening::me_adapter::mat_row_nonzero_entries(&z_col);
                        z_col_cache.insert(col_id, EncodedTimeColCache { z_col, row_nz });
                    }
                    let z_col = z_col_cache.get(&col_id).ok_or_else(|| {
                        PiCcsError::ProtocolError(format!(
                            "time/opening proof build: cached column missing for col_id={col_id}",
                        ))
                    })?;
                    let digits = crate::time_opening::me_adapter::eval_mat_digits_from_sparse_row_weight_coeffs(
                        point_row_weights_re.as_slice(),
                        point_row_weights_im.as_slice(),
                        z_col.row_nz.as_slice(),
                        z_col.z_col.cols(),
                    )?;
                    let recomposed = crate::time_opening::me_adapter::recompose_digits_to_scalar(
                        digits.as_slice(),
                        crate::time_opening::STAGE8_TIME_DECOMP_BASE,
                    );
                    if recomposed != eval {
                        return Err(PiCcsError::ProtocolError(format!(
                            "time/opening proof build: digit recomposition mismatch for col_id={col_id} (domain={domain:?}, eval={eval:?}, recomposed={recomposed:?})"
                        )));
                    }
                    digit_evals.push(digits);
                }
                out.push(crate::shard_proof_types::TimeOpeningProof {
                    point: opening.point.clone(),
                    col_ids,
                    evals,
                    digit_evals,
                });
            }
            out
        };
        let (opening_manifest, opening_reduction, opening_unification, joint_opening_lane, stage8_fold) =
            if opening_proofs.is_empty() {
                if !fold_openings.is_empty() {
                    return Err(PiCcsError::ProtocolError(
                        "time/opening: missing opening proofs for non-empty named openings".into(),
                    ));
                }
                (
                    crate::shard_proof_types::OpeningClaimManifest::default(),
                    crate::shard_proof_types::OpeningReductionProof::default(),
                    crate::shard_proof_types::OpeningUnificationProof::default(),
                    crate::shard_proof_types::JointOpeningLaneProof::default(),
                    Vec::new(),
                )
            } else {
                let opening_manifest = crate::time_opening::manifest::build_opening_claim_manifest(
                    &fold_openings,
                    &opening_proofs,
                    &step.time_columns.col_ids,
                    time_cpu_commitments.len(),
                )?;
                crate::time_opening::manifest::bind_opening_claim_manifest(tr, step_idx, &opening_manifest);
                let opening_batch_coeffs =
                    bind_time_opening_batches_and_sample_coeffs(tr, params, step_idx, &opening_proofs)?;
                let opening_reduction = crate::time_opening::reduction::build_opening_reduction(&opening_manifest)?;
                let opening_unification = crate::time_opening::reduction::prove_opening_unification_sumcheck(
                    tr,
                    step_idx,
                    &opening_reduction,
                )?;
                let (joint_opening_lane, stage8_joint_wits) =
                    crate::time_opening::joint_lane::prove_joint_opening_lane_with_witnesses(
                        tr,
                        params,
                        step_idx,
                        step,
                        &cpu_bus,
                        &time_cpu_commitments,
                        &time_mem_commitments,
                        &step.time_columns.col_ids,
                        &opening_proofs,
                        &opening_manifest.digest,
                        &opening_reduction,
                        &opening_unification,
                        &opening_batch_coeffs,
                    )?;
                let mut stage8_fold: Vec<RlcDecProof> = Vec::with_capacity(1);
                let stage8_params = stage8_time_decomp_params(params)?;
                let stage8_plan = crate::time_opening::joint_lane::build_stage8_fold_lane_plan(
                    &joint_opening_lane,
                    &opening_unification,
                    step.time_columns.t,
                )?;
                if let Some(plan) = stage8_plan {
                    if stage8_joint_wits.len() != plan.claims.len() {
                        return Err(PiCcsError::ProtocolError(format!(
                            "stage8 fold: witness/claim count mismatch (wits={}, claims={})",
                            stage8_joint_wits.len(),
                            plan.claims.len()
                        )));
                    }
                    if !has_global_pp_for_dims(D, plan.ccs.m) {
                        return Err(PiCcsError::InvalidInput(format!(
                        "stage8 fold: missing global PP for (D,m)=({D},{}); PP must be pre-registered with canonical seed",
                        plan.ccs.m
                    )));
                    }
                    let stage8_committer =
                        neo_ajtai::AjtaiSModule::from_global_for_dims(D, plan.ccs.m).map_err(|e| {
                            PiCcsError::InvalidInput(format!(
                                "stage8 fold: missing global committer for (D,m)=({D},{}): {e}",
                                plan.ccs.m
                            ))
                        })?;
                    tr.append_message(b"fold/stage8_lane_start", &(step_idx as u64).to_le_bytes());
                    tr.append_message(b"fold/stage8_lane_group_idx", &0u64.to_le_bytes());
                    let wit_refs: Vec<&Mat<F>> = stage8_joint_wits.iter().collect();
                    let (stage8_proof, _stage8_wits) = prove_rlc_dec_lane(
                        &mode,
                        RlcLane::Val,
                        tr,
                        &stage8_params,
                        &plan.ccs,
                        None,
                        None,
                        &ring,
                        ell_d,
                        k_dec,
                        step_idx,
                        None,
                        plan.claims.as_slice(),
                        wit_refs.as_slice(),
                        false,
                        &stage8_committer,
                        mixers,
                    )?;
                    stage8_fold.push(stage8_proof);
                } else if !stage8_joint_wits.is_empty() {
                    return Err(PiCcsError::ProtocolError(
                        "stage8 fold: missing lane plan for non-empty stage8 witnesses".into(),
                    ));
                }
                (
                    opening_manifest,
                    opening_reduction,
                    opening_unification,
                    joint_opening_lane,
                    stage8_fold,
                )
            };
        let cpu_sumcheck = cpu_sumcheck_from_ccs(ccs_initial_sum, ccs_time_rounds_meta, &ccs_time_chals_meta);
        let shift_sumcheck = shift_sumcheck_from_batched_time(&batched_time, &r_time, control_required)?;

        step_proofs.push(StepProof {
            fold: FoldStep {
                ccs_out,
                ccs_proof,
                rlc_rhos: rhos,
                rlc_parent: parent_pub,
                dec_children: children,
                cpu_sumcheck,
                shift_sumcheck,
                time_cpu_commitments,
                time_mem_commitments,
                time_t: step.time_columns.t,
                time_declared_len,
                time_col_ids: step.time_columns.col_ids.clone(),
                memory_time_proofs: batched_time.labels.clone(),
                openings: fold_openings,
                opening_proofs,
                opening_manifest,
                opening_reduction,
                opening_unification,
                joint_opening_lane,
                folding_lanes: crate::shard_proof_types::FoldingLanes {
                    main_children: accumulator.len(),
                    val_children: val_fold.iter().map(|p| p.dec_children.len()).sum(),
                    wb_children: wb_fold.iter().map(|p| p.dec_children.len()).sum(),
                    wp_children: wp_fold.iter().map(|p| p.dec_children.len()).sum(),
                    stage8_children: stage8_fold.iter().map(|p| p.dec_children.len()).sum(),
                },
            },
            mem: mem_proof,
            batched_time,
            poseidon_local_time,
            poseidon_cycle_fold,
            poseidon_local_fold,
            val_fold,
            wb_fold,
            wp_fold,
            compressed_substeps: None,
            stage8_fold,
        });

        tr.append_message(b"fold/step_done", &(step_idx as u64).to_le_bytes());
        if let Some(out) = step_prove_ms_out.as_deref_mut() {
            out.push(elapsed_ms(step_start));
        }
    }

    Ok((
        ShardProof {
            steps: step_proofs,
            output_proof,
            segment_meta: None,
        },
        accumulator_wit,
        val_lane_wits,
        prev_twist_decoded,
        poseidon_carry,
    ))
}
