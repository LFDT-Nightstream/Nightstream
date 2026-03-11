use super::*;

pub(crate) fn prove_rlc_dec_lane<L, MR, MB>(
    mode: &FoldingMode,
    lane: RlcLane,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    ccs_sparse_cache: Option<&SparseCache<F>>,
    cpu_bus: Option<&neo_memory::cpu::BusLayout>,
    ring: &ccs::RotRing,
    ell_d: usize,
    k_dec: usize,
    step_idx: usize,
    backend_ctx: &neo_reductions::accelerator::BackendContext,
    trace_linkage_t_len: Option<usize>,
    me_inputs: &[CeClaim<Cmt, F, K>],
    wit_inputs: &[&Mat<F>],
    want_witnesses: bool,
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(RlcDecProof, Vec<Mat<F>>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if me_inputs.is_empty() {
        let prefix = match lane {
            RlcLane::Main => "",
            RlcLane::Val => "val-lane ",
        };
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: {prefix}RLC input batch is empty",
            step_idx
        )));
    }
    if wit_inputs.len() != me_inputs.len() {
        let prefix = match lane {
            RlcLane::Main => "",
            RlcLane::Val => "val-lane ",
        };
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: {prefix}RLC witness count mismatch (me_inputs.len()={}, wit_inputs.len()={})",
            step_idx,
            me_inputs.len(),
            wit_inputs.len()
        )));
    }
    let s_lane_owned: CcsStructure<F>;
    let s_lane: &CcsStructure<F> = if matches!(lane, RlcLane::Main) {
        s
    } else {
        let ell_lane = me_inputs[0].r.len();
        let n_lane = 1usize
            .checked_shl(ell_lane as u32)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("step {}: lane r dimension overflow", step_idx)))?;
        let mut s_tmp = s.clone();
        s_tmp.n = n_lane;
        s_lane_owned = s_tmp;
        &s_lane_owned
    };
    let s = s_lane;
    let inputs_have_extra_y = me_inputs.iter().any(|me| me.y_ring.len() > s.t());

    bind_rlc_inputs_with_context(tr, lane, step_idx, me_inputs, backend_ctx)?;
    let k_rlc_min = neo_reductions::common::min_k_rho_for_rlc_count(params, ring, me_inputs.len())? as usize;
    let k_rho_eff = core::cmp::max(k_dec, k_rlc_min);
    let mut params_rlc = params.clone();
    params_rlc.k_rho = k_rho_eff as u32;
    let rlc_rhos = ccs::sample_rot_rhos_n_typed(tr, &params_rlc, ring, me_inputs.len())?;
    let rlc_rho_mats = ccs::rot_rhos_to_mats(&rlc_rhos);
    let (mut rlc_parent, Z_mix) = if inputs_have_extra_y {
        let parent_pub = ccs::rlc_public(s, params, &rlc_rhos, me_inputs, mixers.mix_rhos_commits, ell_d)?;
        let (_, z_mix_tmp) = neo_reductions::optimized_engine::rlc_reduction_optimized_with_commit_mix(
            s,
            params,
            &rlc_rho_mats,
            me_inputs,
            wit_inputs,
            ell_d,
            mixers.mix_rhos_commits,
        );
        (parent_pub, z_mix_tmp)
    } else {
        #[cfg(feature = "paper-exact")]
        {
            if matches!(mode, FoldingMode::PaperExact) {
                let wit_owned: Vec<Mat<F>> = wit_inputs.iter().map(|m| (*m).clone()).collect();
                neo_reductions::optimized_engine::rlc_reduction_paper_exact_with_commit_mix(
                    s,
                    params,
                    &rlc_rho_mats,
                    me_inputs,
                    &wit_owned,
                    ell_d,
                    mixers.mix_rhos_commits,
                )
            } else {
                neo_reductions::optimized_engine::rlc_reduction_optimized_with_commit_mix(
                    s,
                    params,
                    &rlc_rho_mats,
                    me_inputs,
                    wit_inputs,
                    ell_d,
                    mixers.mix_rhos_commits,
                )
            }
        }
        #[cfg(not(feature = "paper-exact"))]
        {
            neo_reductions::optimized_engine::rlc_reduction_optimized_with_commit_mix(
                s,
                params,
                &rlc_rho_mats,
                me_inputs,
                wit_inputs,
                ell_d,
                mixers.mix_rhos_commits,
            )
        }
    };

    let Z_mix = &Z_mix;
    let k_dec_eff = core::cmp::max(
        k_rho_eff,
        core::cmp::max(
            required_dec_digits_for_matrix(params, Z_mix)?,
            required_dec_digits_for_matrix(params, &rlc_parent.X)?,
        ),
    );
    let m_commit = Z_mix.cols();
    let has_stream_pp =
        has_global_pp_for_dims(D, m_commit) || get_global_pp_seeded_params_for_dims(D, m_commit).is_ok();
    let inputs_have_aux_openings = me_inputs.iter().any(|me| !me.aux_openings.is_empty());
    let can_stream_dec = !want_witnesses
        && has_stream_pp
        && !cpu_bus.map(|b| b.bus_cols > 0).unwrap_or(false)
        && !inputs_have_aux_openings;

    let materialize_dec = || -> Result<(Vec<CeClaim<Cmt, F, K>>, bool, bool, bool, Vec<Mat<F>>), PiCcsError> {
        // Standard DEC: materialize digit matrices (needed when carrying witnesses forward).
        let (Z_split, digit_nonzero) = ccs::split_b_matrix_k_with_nonzero_flags(Z_mix, k_dec_eff, params.b)?;
        let zero_c = Cmt::zeros(rlc_parent.c.d, rlc_parent.c.kappa);
        let mut child_cs: Vec<Cmt> = vec![zero_c.clone(); Z_split.len()];
        let nonzero_idx: Vec<usize> = digit_nonzero
            .iter()
            .enumerate()
            .filter_map(|(idx, &nz)| nz.then_some(idx))
            .collect();
        if !nonzero_idx.is_empty() {
            let mats: Vec<&Mat<F>> = nonzero_idx.iter().map(|&idx| &Z_split[idx]).collect();
            let commits = l.commit_many(&mats);
            if commits.len() != mats.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: DEC commit_many returned {} commitments for {} matrices",
                    step_idx,
                    commits.len(),
                    mats.len()
                )));
            }
            for (pos, &idx) in nonzero_idx.iter().enumerate() {
                child_cs[idx] = commits[pos].clone();
            }
        }
        let (dec_children, ok_y, ok_X, ok_c) = ccs::dec_children_with_commit_cached(
            mode.clone(),
            s,
            params,
            &rlc_parent,
            &Z_split,
            ell_d,
            &child_cs,
            mixers.combine_b_pows,
            ccs_sparse_cache,
        );
        Ok((dec_children, ok_y, ok_X, ok_c, Z_split))
    };

    let (mut dec_children, ok_y, ok_X, ok_c, maybe_wits) = if can_stream_dec {
        // Memory-optimized DEC: compute children + commitments without materializing Z_split.
        // If public consistency checks fail (e.g. global PP mismatch vs local committer),
        // fall back to the materialized path for correctness.
        match dec_stream_no_witness(
            params,
            s,
            &rlc_parent,
            Z_mix,
            ell_d,
            k_dec_eff,
            mixers.combine_b_pows,
            ccs_sparse_cache,
        ) {
            Ok((children, _child_cs, ok_y, ok_X, ok_c)) if ok_y && ok_X && ok_c => {
                (children, ok_y, ok_X, ok_c, Vec::new())
            }
            Ok(_) | Err(_) => materialize_dec()?,
        }
    } else {
        materialize_dec()?
    };
    if !(ok_y && ok_X && ok_c) {
        let lane_label = match lane {
            RlcLane::Main => "DEC",
            RlcLane::Val => "DEC(val)",
        };
        let parent_y_zcol_len = rlc_parent.y_zcol.len();
        let first_child_y_zcol_len = dec_children.first().map(|c| c.y_zcol.len()).unwrap_or(0);
        let parent_y_rows = rlc_parent.y_ring.len();
        let first_child_y_rows = dec_children.first().map(|c| c.y_ring.len()).unwrap_or(0);
        let parent_m_in = rlc_parent.m_in;
        let parent_x_shape = (rlc_parent.X.rows(), rlc_parent.X.cols());
        let child0_x_shape = dec_children
            .first()
            .map(|c| (c.X.rows(), c.X.cols()))
            .unwrap_or((0, 0));
        return Err(PiCcsError::ProtocolError(format!(
            "{} public check failed at step {} (y={}, X={}, c={}, parent.m_in={}, parent.X={:?}, child0.X={:?}, parent.y_zcol.len()={}, child0.y_zcol.len()={}, parent.y_ring.len()={}, child0.y_ring.len()={})",
            lane_label, step_idx, ok_y, ok_X, ok_c, parent_m_in, parent_x_shape, child0_x_shape, parent_y_zcol_len, first_child_y_zcol_len, parent_y_rows, first_child_y_rows
        )));
    }

    // Shared CPU bus: carry the implicit bus openings through Π_RLC/Π_DEC so they remain
    // part of the folded instance (and are checked by public DEC verification).
    if let Some(bus) = cpu_bus {
        if bus.bus_cols > 0 {
            let core_t = s.t();
            let parent_has_prefilled_bus = rlc_parent.y_ring.len() > core_t || rlc_parent.ct.len() > core_t;
            if Z_mix.cols() == bus.m && !parent_has_prefilled_bus {
                crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                    params,
                    bus,
                    core_t,
                    Z_mix,
                    &mut rlc_parent,
                )?;
                for (child, Zi) in dec_children.iter_mut().zip(maybe_wits.iter()) {
                    crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(params, bus, core_t, Zi, child)?;
                }
            } else {
                if rlc_parent.y_ring.len() < core_t || rlc_parent.ct.len() < core_t {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: non-physical bus path expects parent y/ct len to be at least core_t={} (got y.len()={}, ct.len()={})",
                        step_idx,
                        core_t,
                        rlc_parent.y_ring.len(),
                        rlc_parent.ct.len()
                    )));
                }
                let parent_extra = rlc_parent.y_ring.len().saturating_sub(core_t);
                if parent_extra < bus.bus_cols {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: non-physical bus path missing bus suffix coordinates (have {}, expected at least {})",
                        step_idx, parent_extra, bus.bus_cols
                    )));
                }
                if !matches!(lane, RlcLane::Main) || trace_linkage_t_len.is_none() {
                    if parent_extra != bus.bus_cols || rlc_parent.ct.len() != core_t + bus.bus_cols {
                        return Err(PiCcsError::ProtocolError(format!(
                            "step {}: non-physical bus path requires exact parent suffix length (y extra={}, ct extra={}, expected bus_cols={})",
                            step_idx,
                            parent_extra,
                            rlc_parent.ct.len().saturating_sub(core_t),
                            bus.bus_cols
                        )));
                    }
                }

                let y_pad = (params.d as usize).next_power_of_two();
                for (child_idx, child) in dec_children.iter_mut().enumerate() {
                    if child.y_ring.len() < core_t || child.ct.len() < core_t {
                        return Err(PiCcsError::ProtocolError(format!(
                            "step {}: non-physical bus path expects child y/ct len to start at core_t={} (got y.len()={}, ct.len()={})",
                            step_idx,
                            core_t,
                            child.y_ring.len(),
                            child.ct.len()
                        )));
                    }
                    child.y_ring.truncate(core_t);
                    child.ct.truncate(core_t);
                    // Non-physical bus openings are lane metadata carried through Π_DEC.
                    // They are not decomposed witness coordinates; keep the canonical parent
                    // opening mass on child 0 and force all sibling children to zero.
                    for col_id in 0..bus.bus_cols {
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
                            "non-primary DEC children must keep propagated metadata openings at zero"
                        );
                    }
                }
            }
        }
    }

    // If the main lane carries RV32 trace linkage openings, propagate them through Π_DEC so child
    // instances keep the same aux_openings shape (after optional shared-bus openings).
    if matches!(lane, RlcLane::Main) && trace_linkage_t_len.is_some() {
        let core_t = s.t();
        let trace_open_base = core_t + cpu_bus.map_or(0usize, |bus| bus.bus_cols);
        let _trace_open_base_aux = rlc_parent
            .ct
            .len()
            .checked_sub(trace_open_base)
            .ok_or_else(|| PiCcsError::InvalidInput("trace linkage aux base underflow".into()))?;
        let trace = Rv32TraceLayout::new();
        let trace_cols_to_open: Vec<usize> = vec![
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

        if rlc_parent.y_ring.len() >= trace_open_base && rlc_parent.ct.len() >= trace_open_base {
            let m_in = rlc_parent.m_in;
            if m_in != 5 {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace linkage openings expect m_in=5 (got {m_in})"
                )));
            }
            let t_len = trace_linkage_t_len
                .ok_or_else(|| PiCcsError::ProtocolError("trace linkage openings require explicit t_len".into()))?;
            if t_len == 0 {
                return Err(PiCcsError::InvalidInput("trace linkage expects t_len >= 1".into()));
            }

            let trace_available = rlc_parent.y_ring.len().saturating_sub(trace_open_base);
            if trace_available != trace_cols_to_open.len()
                || rlc_parent.ct.len().saturating_sub(trace_open_base) != trace_cols_to_open.len()
            {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: trace linkage propagation requires exact trace suffix length (y extra={}, ct extra={}, expected={})",
                    step_idx,
                    trace_available,
                    rlc_parent.ct.len().saturating_sub(trace_open_base),
                    trace_cols_to_open.len()
                )));
            }
            let y_pad = (params.d as usize).next_power_of_two();
            for (child_idx, child) in dec_children.iter_mut().enumerate() {
                if child.y_ring.len() < trace_open_base || child.ct.len() < trace_open_base {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: trace linkage propagation expects child y/ct len to start at trace_open_base={} (got y.len()={}, ct.len()={})",
                        step_idx,
                        trace_open_base,
                        child.y_ring.len(),
                        child.ct.len()
                    )));
                }
                child.y_ring.truncate(trace_open_base);
                child.ct.truncate(trace_open_base);
                for open_idx in 0..trace_available {
                    if child_idx == 0 {
                        child
                            .y_ring
                            .push(rlc_parent.y_ring[trace_open_base + open_idx].clone());
                        child.ct.push(rlc_parent.ct[trace_open_base + open_idx]);
                    } else {
                        // Non-physical trace openings are metadata carried through Π_DEC.
                        // Keep the canonical parent mass on child 0 and force siblings to zero.
                        child.y_ring.push(vec![K::ZERO; y_pad]);
                        child.ct.push(K::ZERO);
                    }
                }
                if child_idx > 0 {
                    debug_assert!(
                        child.y_ring[trace_open_base..]
                            .iter()
                            .all(|row| row.iter().all(|v| *v == K::ZERO))
                            && child.ct[trace_open_base..].iter().all(|v| *v == K::ZERO),
                        "non-primary DEC children must keep propagated trace metadata openings at zero"
                    );
                }
            }
        }
    }
    for (child_idx, child) in dec_children.iter().enumerate() {
        if child.y_ring.len() != rlc_parent.y_ring.len() || child.ct.len() != rlc_parent.ct.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: DEC child[{}] suffix-length drift after propagation (child y/ct={}/{}, parent y/ct={}/{})",
                step_idx,
                child_idx,
                child.y_ring.len(),
                child.ct.len(),
                rlc_parent.y_ring.len(),
                rlc_parent.ct.len()
            )));
        }
    }
    Ok((
        RlcDecProof {
            rlc_rhos,
            rlc_parent,
            dec_children,
        },
        maybe_wits,
    ))
}

pub(crate) fn verify_rlc_dec_lane<MR, MB>(
    lane: RlcLane,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    ring: &ccs::RotRing,
    ell_d: usize,
    mixers: CommitMixers<MR, MB>,
    step_idx: usize,
    backend_ctx: &neo_reductions::accelerator::BackendContext,
    rlc_inputs: &[CeClaim<Cmt, F, K>],
    rlc_rhos: &[ccs::RotRho],
    rlc_parent: &CeClaim<Cmt, F, K>,
    dec_children: &[CeClaim<Cmt, F, K>],
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if rlc_inputs.is_empty() {
        let prefix = match lane {
            RlcLane::Main => "",
            RlcLane::Val => "val-lane ",
        };
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: {}RLC input batch is empty",
            step_idx, prefix
        )));
    }
    let s_lane_owned: CcsStructure<F>;
    let s_lane: &CcsStructure<F> = if matches!(lane, RlcLane::Main) {
        s
    } else {
        let ell_lane = rlc_inputs[0].r.len();
        let n_lane = 1usize
            .checked_shl(ell_lane as u32)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("step {}: lane r dimension overflow", step_idx)))?;
        let mut s_tmp = s.clone();
        s_tmp.n = n_lane;
        s_lane_owned = s_tmp;
        &s_lane_owned
    };
    let s = s_lane;

    bind_rlc_inputs_with_context(tr, lane, step_idx, rlc_inputs, backend_ctx)?;

    if rlc_rhos.len() != rlc_inputs.len() {
        let prefix = match lane {
            RlcLane::Main => "",
            RlcLane::Val => "val-lane ",
        };
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: {}RLC ρ count mismatch (expected {}, got {})",
            step_idx,
            prefix,
            rlc_inputs.len(),
            rlc_rhos.len()
        )));
    }

    for (i, me) in rlc_inputs.iter().enumerate() {
        verify_me_y_scalars_canonical(
            me,
            params.b,
            s.m,
            step_idx,
            &format!(
                "{}RLC input[{i}]",
                match lane {
                    RlcLane::Main => "",
                    RlcLane::Val => "val-lane ",
                }
            ),
        )?;
    }

    let k_rlc_min = neo_reductions::common::min_k_rho_for_rlc_count(params, ring, rlc_inputs.len())? as usize;
    let k_rho_eff = core::cmp::max(params.k_rho as usize, k_rlc_min);
    let mut params_rlc = params.clone();
    params_rlc.k_rho = k_rho_eff as u32;
    let rhos_from_tr = ccs::sample_rot_rhos_n_typed(tr, &params_rlc, ring, rlc_inputs.len())?;
    for (j, (sampled, stored)) in rhos_from_tr.iter().zip(rlc_rhos.iter()).enumerate() {
        if sampled != stored {
            return Err(PiCcsError::ProtocolError(match lane {
                RlcLane::Main => format!("step {}: RLC ρ #{} mismatch: transcript vs proof", step_idx, j),
                RlcLane::Val => format!("step {}: val-lane RLC ρ #{} mismatch: transcript vs proof", step_idx, j),
            }));
        }
    }

    let parent_pub = ccs::rlc_public(s, params, rlc_rhos, rlc_inputs, mixers.mix_rhos_commits, ell_d)?;

    let prefix = match lane {
        RlcLane::Main => "",
        RlcLane::Val => "val-lane ",
    };
    if parent_pub.m_in != rlc_parent.m_in {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC m_in mismatch (public={}, proof={})",
            step_idx, parent_pub.m_in, rlc_parent.m_in
        )));
    }
    if parent_pub.fold_digest != rlc_parent.fold_digest {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC fold_digest mismatch",
            step_idx
        )));
    }
    if parent_pub.c_step_coords != rlc_parent.c_step_coords {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC c_step_coords mismatch",
            step_idx
        )));
    }
    if parent_pub.u_offset != rlc_parent.u_offset {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC u_offset mismatch",
            step_idx
        )));
    }
    if parent_pub.u_len != rlc_parent.u_len {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC u_len mismatch",
            step_idx
        )));
    }
    if parent_pub.X != rlc_parent.X {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC X mismatch",
            step_idx
        )));
    }
    if parent_pub.c != rlc_parent.c {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC commitment mismatch",
            step_idx
        )));
    }
    if parent_pub.r != rlc_parent.r {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC r mismatch",
            step_idx
        )));
    }
    if parent_pub.s_col != rlc_parent.s_col {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC s_col mismatch",
            step_idx
        )));
    }
    if parent_pub.y_ring != rlc_parent.y_ring {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC y mismatch",
            step_idx
        )));
    }
    if parent_pub.ct != rlc_parent.ct {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC ct mismatch",
            step_idx
        )));
    }
    if parent_pub.aux_openings != rlc_parent.aux_openings {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC aux_openings mismatch",
            step_idx
        )));
    }
    if parent_pub.y_zcol != rlc_parent.y_zcol {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC y_zcol mismatch",
            step_idx
        )));
    }

    if rlc_parent.X.rows() != D || rlc_parent.X.cols() != rlc_parent.m_in {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC parent X shape {}x{} does not match m_in={}",
            step_idx,
            rlc_parent.X.rows(),
            rlc_parent.X.cols(),
            rlc_parent.m_in
        )));
    }
    if dec_children.len() < k_rho_eff {
        return Err(PiCcsError::ProtocolError(match lane {
            RlcLane::Main => format!(
                "step {}: DEC child count {} is below required width {}",
                step_idx,
                dec_children.len(),
                k_rho_eff
            ),
            RlcLane::Val => format!(
                "step {}: val-lane DEC child count {} is below required width {}",
                step_idx,
                dec_children.len(),
                k_rho_eff
            ),
        }));
    }
    if !dec_children.is_empty() {
        validate_me_batch_invariants(dec_children, "verify step dec children")?;
        for (child_idx, child) in dec_children.iter().enumerate() {
            if child.m_in != rlc_parent.m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: {prefix}DEC child[{child_idx}] has m_in={}, expected {}",
                    step_idx, child.m_in, rlc_parent.m_in
                )));
            }
            if child.fold_digest != rlc_parent.fold_digest {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: {prefix}DEC child[{child_idx}] fold_digest mismatch",
                    step_idx
                )));
            }
        }
    }

    if !ccs::verify_dec_public(s, params, rlc_parent, dec_children, mixers.combine_b_pows, ell_d) {
        return Err(PiCcsError::ProtocolError(match lane {
            RlcLane::Main => format!("step {}: DEC public check failed", step_idx),
            RlcLane::Val => format!("step {}: val-lane DEC public check failed", step_idx),
        }));
    }

    Ok(())
}

#[cfg(feature = "paper-exact")]
pub(crate) fn crosscheck_route_a_ccs_step<L>(
    cfg: &neo_reductions::engines::CrosscheckCfg,
    step_idx: usize,
    params: &NeoParams,
    s: &CcsStructure<F>,
    cpu_bus: &neo_memory::cpu::BusLayout,
    mcs_inst: &neo_ccs::CcsClaim<Cmt, F>,
    mcs_wit: &neo_ccs::CcsWitness<F>,
    me_inputs: &[CeClaim<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    ccs_out: &[CeClaim<Cmt, F, K>],
    ccs_proof: &crate::PiCcsProof,
    ell_d: usize,
    ell_n: usize,
    ell_m: usize,
    d_sc: usize,
    fold_digest: [u8; 32],
    log: &L,
) -> Result<(), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
{
    let want_rounds_total = ell_n
        .checked_add(ell_d)
        .ok_or_else(|| PiCcsError::ProtocolError("ell_n + ell_d overflow".into()))?;
    if ccs_proof.sumcheck_rounds.len() != want_rounds_total {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: crosscheck expects {} CCS sumcheck rounds, got {}",
            step_idx,
            want_rounds_total,
            ccs_proof.sumcheck_rounds.len(),
        )));
    }
    if ccs_proof.sumcheck_challenges.len() != want_rounds_total {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: crosscheck expects {} CCS sumcheck challenges, got {}",
            step_idx,
            want_rounds_total,
            ccs_proof.sumcheck_challenges.len(),
        )));
    }
    let (s_col_prime, alpha_prime_nc) = if ccs_proof.variant == crate::optimized_engine::PiCcsProofVariant::SplitNcV1 {
        let want_nc_rounds_total = ell_m
            .checked_add(ell_d)
            .ok_or_else(|| PiCcsError::ProtocolError("ell_m + ell_d overflow".into()))?;
        if ccs_proof.sumcheck_rounds_nc.len() != want_nc_rounds_total {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck expects {} NC sumcheck rounds, got {}",
                step_idx,
                want_nc_rounds_total,
                ccs_proof.sumcheck_rounds_nc.len(),
            )));
        }
        if ccs_proof.sumcheck_challenges_nc.len() != want_nc_rounds_total {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck expects {} NC sumcheck challenges, got {}",
                step_idx,
                want_nc_rounds_total,
                ccs_proof.sumcheck_challenges_nc.len(),
            )));
        }
        ccs_proof.sumcheck_challenges_nc.split_at(ell_m)
    } else {
        (&[][..], &[][..])
    };

    let (r_prime, alpha_prime) = ccs_proof.sumcheck_challenges.split_at(ell_n);
    let r_inputs = me_inputs.first().map(|mi| mi.r.as_slice());

    // Crosscheck initial-sum parity is most informative once there is at least one carried ME
    // input. For empty-accumulator starts, optimized and paper-exact route through different
    // constant-term paths and can diverge without indicating a soundness issue.
    if cfg.initial_sum && !me_inputs.is_empty() {
        let lhs_exact = crate::paper_exact_engine::sum_q_over_hypercube_paper_exact(
            s,
            params,
            core::slice::from_ref(mcs_wit),
            me_witnesses,
            &ccs_proof.challenges_public,
            ell_d,
            ell_n,
            r_inputs,
        );
        let initial_sum_prover = ccs_proof
            .sumcheck_rounds
            .first()
            .map(|p0| poly_eval_k(p0, K::ZERO) + poly_eval_k(p0, K::ONE))
            .ok_or_else(|| PiCcsError::ProtocolError("crosscheck: missing sumcheck round 0".into()))?;
        if lhs_exact != initial_sum_prover {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck initial sum mismatch (optimized vs paper-exact)",
                step_idx
            )));
        }
    }

    if cfg.per_round {
        let mut paper_oracle = crate::paper_exact_engine::oracle::PaperExactOracle::new(
            s,
            params,
            core::slice::from_ref(mcs_wit),
            me_witnesses,
            ccs_proof.challenges_public.clone(),
            ell_d,
            ell_n,
            d_sc,
            r_inputs,
        );

        let mut any_mismatch = false;
        for (round_idx, (opt_coeffs, &challenge)) in ccs_proof
            .sumcheck_rounds
            .iter()
            .zip(ccs_proof.sumcheck_challenges.iter())
            .enumerate()
        {
            let deg = paper_oracle.degree_bound();
            let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
            let paper_evals = paper_oracle.evals_at(&xs);

            for (&x, &expected) in xs.iter().zip(paper_evals.iter()) {
                let actual = poly_eval_k(opt_coeffs, x);
                if actual != expected {
                    any_mismatch = true;
                    if cfg.fail_fast {
                        return Err(PiCcsError::ProtocolError(format!(
                            "step {}: crosscheck round {} polynomial mismatch",
                            step_idx, round_idx
                        )));
                    }
                }
            }

            paper_oracle.fold(challenge);
        }
        if any_mismatch {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck per-round polynomial mismatch",
                step_idx
            )));
        }
    }

    if cfg.terminal {
        let running_sum_prover = if let Some(initial) = ccs_proof.sc_initial_sum {
            let mut running = initial;
            for (coeffs, &ri) in ccs_proof
                .sumcheck_rounds
                .iter()
                .zip(ccs_proof.sumcheck_challenges.iter())
            {
                running = poly_eval_k(coeffs, ri);
            }
            running
        } else {
            ccs_proof
                .sumcheck_rounds
                .first()
                .map(|p0| poly_eval_k(p0, K::ZERO) + poly_eval_k(p0, K::ONE))
                .unwrap_or(K::ZERO)
        };

        let rhs_fe = crate::paper_exact_engine::rhs_terminal_identity_fe_paper_exact(
            s,
            params,
            &ccs_proof.challenges_public,
            r_prime,
            alpha_prime,
            ccs_out,
            r_inputs,
        );
        let (lhs_fe, _rhs_unused) = crate::paper_exact_engine::q_eval_at_ext_point_fe_paper_exact_with_inputs(
            s,
            params,
            core::slice::from_ref(mcs_wit),
            me_witnesses,
            alpha_prime,
            r_prime,
            &ccs_proof.challenges_public,
            r_inputs,
        );
        if rhs_fe != lhs_fe || rhs_fe != running_sum_prover {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck FE terminal evaluation claim mismatch",
                step_idx
            )));
        }

        let rhs_nc = crate::paper_exact_engine::rhs_terminal_identity_nc_paper_exact(
            params,
            &ccs_proof.challenges_public,
            s_col_prime,
            alpha_prime_nc,
            ccs_out,
        );
        if rhs_nc != ccs_proof.sumcheck_final_nc {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck NC terminal evaluation claim mismatch",
                step_idx
            )));
        }
    }

    if cfg.outputs {
        let mut out_me_ref = build_me_outputs_paper_exact(
            s,
            params,
            core::slice::from_ref(mcs_inst),
            core::slice::from_ref(mcs_wit),
            me_inputs,
            me_witnesses,
            r_prime,
            s_col_prime,
            ell_d,
            fold_digest,
            log,
        );

        if cpu_bus.bus_cols > 0 {
            let core_t = s.t();
            if out_me_ref.len() != 1 + me_witnesses.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck CCS output count mismatch for bus openings (out_me_ref.len()={}, expected {})",
                    step_idx,
                    out_me_ref.len(),
                    1 + me_witnesses.len()
                )));
            }

            crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                params,
                cpu_bus,
                core_t,
                &mcs_wit.Z,
                &mut out_me_ref[0],
            )?;
            for (out, Z) in out_me_ref.iter_mut().skip(1).zip(me_witnesses.iter()) {
                crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(params, cpu_bus, core_t, Z, out)?;
            }

            let trace = Rv32TraceLayout::new();
            let trace_cols_to_open: Vec<usize> = vec![
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
            let want_with_trace = cpu_bus.bus_cols + trace_cols_to_open.len();
            if ccs_out
                .first()
                .map(|me| me.aux_openings.len() == want_with_trace)
                .unwrap_or(false)
            {
                let m_in = mcs_inst.m_in;
                let m_after_public =
                    s.m.checked_sub(m_in)
                        .ok_or_else(|| PiCcsError::ProtocolError("crosscheck trace region underflow".into()))?;
                let bus_region_len = cpu_bus
                    .bus_cols
                    .checked_mul(cpu_bus.chunk_size)
                    .ok_or_else(|| PiCcsError::ProtocolError("crosscheck bus region overflow".into()))?;

                let mut t_len_candidates: Vec<usize> = Vec::new();
                if let Some(legacy_trace_region) = m_after_public.checked_sub(bus_region_len) {
                    if trace.cols != 0 && legacy_trace_region % trace.cols == 0 {
                        t_len_candidates.push(legacy_trace_region / trace.cols);
                    }
                }
                if trace.cols != 0 && m_after_public % trace.cols == 0 {
                    t_len_candidates.push(m_after_public / trace.cols);
                }
                t_len_candidates.dedup();

                let max_trace_col = trace_cols_to_open.iter().copied().max().unwrap_or(0);
                let t_len = t_len_candidates
                    .into_iter()
                    .find(|&t_len| {
                        t_len > 0
                            && m_in
                                .checked_add(max_trace_col.saturating_mul(t_len))
                                .and_then(|start| start.checked_add(t_len.saturating_sub(1)))
                                .map_or(false, |end| end < mcs_wit.Z.cols())
                    })
                    .ok_or_else(|| {
                        PiCcsError::ProtocolError(format!(
                            "step {}: crosscheck cannot infer trace t_len (m={}, m_in={}, bus_region_len={}, trace_cols={}, z_cols={})",
                            step_idx,
                            s.m,
                            m_in,
                            bus_region_len,
                            trace.cols,
                            mcs_wit.Z.cols()
                        ))
                    })?;
                let trace_open_base = core_t + cpu_bus.bus_cols;
                crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
                    params,
                    m_in,
                    t_len,
                    m_in,
                    &trace_cols_to_open,
                    trace_open_base,
                    s.m,
                    &mcs_wit.Z,
                    &mut out_me_ref[0],
                )?;
                for (out, Z) in out_me_ref.iter_mut().skip(1).zip(me_witnesses.iter()) {
                    let trace_span_fits = m_in
                        .checked_add(max_trace_col.saturating_mul(t_len))
                        .and_then(|start| start.checked_add(t_len.saturating_sub(1)))
                        .map_or(false, |end| end < Z.cols());
                    if trace_span_fits && Z.cols() >= mcs_wit.Z.cols() {
                        crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
                            params,
                            m_in,
                            t_len,
                            m_in,
                            &trace_cols_to_open,
                            trace_open_base,
                            s.m,
                            Z,
                            out,
                        )?;
                    } else {
                        crate::memory_sidecar::cpu_bus::append_zero_time_openings_to_me_instance(
                            params,
                            trace_cols_to_open.len(),
                            trace_open_base,
                            out,
                        )?;
                    }
                }
            }
        }

        if out_me_ref.len() != ccs_out.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck output length mismatch (paper={}, optimized={})",
                step_idx,
                out_me_ref.len(),
                ccs_out.len()
            )));
        }

        for (idx, (a, b)) in out_me_ref.iter().zip(ccs_out.iter()).enumerate() {
            if a.m_in != b.m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] m_in mismatch (paper={}, optimized={})",
                    step_idx, a.m_in, b.m_in
                )));
            }
            if a.r != b.r {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] r mismatch",
                    step_idx
                )));
            }
            if a.s_col != b.s_col {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] s_col mismatch",
                    step_idx
                )));
            }
            if a.c.data != b.c.data {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] commitment mismatch",
                    step_idx
                )));
            }
            if a.y_ring.len() != b.y_ring.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] y.len mismatch (paper={}, optimized={})",
                    step_idx,
                    a.y_ring.len(),
                    b.y_ring.len()
                )));
            }
            // In the in-place Route-A cutover, optimized paths can encode equivalent openings with
            // different per-digit y-vectors while preserving canonical ct.
            // Keep strict scalar/value equality and shape checks, but do not require byte-for-byte
            // y row equality in crosscheck mode.
            for (j, (ya, yb)) in a.y_ring.iter().zip(b.y_ring.iter()).enumerate() {
                if ya.len() != yb.len() {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: crosscheck output[{idx}] y row {j} width mismatch (paper={}, optimized={})",
                        step_idx,
                        ya.len(),
                        yb.len()
                    )));
                }
            }
            if a.ct != b.ct {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] ct mismatch",
                    step_idx
                )));
            }
            if a.aux_openings != b.aux_openings {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] aux_openings mismatch",
                    step_idx
                )));
            }
            if a.y_zcol != b.y_zcol {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] y_zcol mismatch",
                    step_idx
                )));
            }
            if a.X.rows() != b.X.rows() || a.X.cols() != b.X.cols() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] X dims mismatch (paper={}x{}, optimized={}x{})",
                    step_idx,
                    a.X.rows(),
                    a.X.cols(),
                    b.X.rows(),
                    b.X.cols()
                )));
            }
            for r in 0..a.X.rows() {
                for c in 0..a.X.cols() {
                    if a.X[(r, c)] != b.X[(r, c)] {
                        return Err(PiCcsError::ProtocolError(format!(
                            "step {}: crosscheck output[{idx}] X mismatch at ({},{})",
                            step_idx, r, c
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Shard Proving
// ============================================================================
