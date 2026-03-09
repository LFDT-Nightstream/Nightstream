use super::*;

pub(crate) struct PoseidonProverSetup {
    pub cycle_enabled: bool,
    pub sidecar: Option<neo_memory::riscv::exec_table::RiscvPoseidonSidecarTable>,
    pub cycle_wit: Option<Mat<F>>,
    pub cycle_open_spec: Option<(usize, usize, Vec<usize>)>,
    pub local_wit_full: Option<Mat<F>>,
    pub local_wits: Option<Vec<Mat<F>>>,
    pub local_open_specs: Option<Vec<(usize, usize, Vec<usize>)>>,
    pub local_t_len: Option<usize>,
    pub local_layout: Option<crate::memory_sidecar::memory::PoseidonLocalTraceLayout>,
    pub local_ell: Option<usize>,
}

#[inline]
fn poseidon_lane_pp_seed(m: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&(m as u64).to_le_bytes());
    seed[8..16].copy_from_slice(&(D as u64).to_le_bytes());
    seed[16..24].copy_from_slice(&0x504f_5345_4944_4f4eu64.to_le_bytes()); // "POSEIDON"
    seed[24..32].copy_from_slice(&0x4c41_4e45_5f50_505fu64.to_le_bytes()); // "LANE_PP_"
    seed
}

pub(crate) fn poseidon_lane_committer(
    params: &NeoParams,
    m: usize,
    label: &str,
) -> Result<neo_ajtai::AjtaiSModule, PiCcsError> {
    if m == 0 {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: poseidon lane m must be > 0"
        )));
    }
    let want_kappa = params.kappa as usize;
    if !neo_ajtai::has_global_pp_for_dims(D, m) {
        let seed = poseidon_lane_pp_seed(m);
        match neo_ajtai::set_global_pp_seeded(D, want_kappa, m, seed) {
            Ok(()) => {}
            Err(e) if neo_ajtai::has_global_pp_for_dims(D, m) => {
                let _ = e;
            }
            Err(e) => {
                return Err(PiCcsError::ProtocolError(format!(
                    "{label}: failed to register seeded Ajtai PP for (D,m)=({D},{m}): {e}"
                )));
            }
        }
    }
    neo_ajtai::AjtaiSModule::from_global_for_dims(D, m)
        .map_err(|e| PiCcsError::ProtocolError(format!("{label}: committer unavailable for (D,m)=({D},{m}): {e}")))
}

#[inline]
fn local_ell_from_t_len(t_len: usize) -> Result<usize, PiCcsError> {
    const POSEIDON_LOCAL_MIN_ELL: usize = 5; // 32 in-slot rows => 5 selector bits.
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("poseidon local: t_len must be > 0".into()));
    }
    if !t_len.is_power_of_two() {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon local: t_len must be a power of two, got {t_len}"
        )));
    }
    Ok(core::cmp::max(t_len.trailing_zeros() as usize, POSEIDON_LOCAL_MIN_ELL))
}

pub(crate) fn build_poseidon_prover_setup(
    _tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    step_idx: usize,
    _ell_n: usize,
    poseidon_carry: &mut crate::memory_sidecar::memory::PoseidonSidecarCarryState,
) -> Result<PoseidonProverSetup, PiCcsError> {
    let cycle_enabled =
        crate::memory_sidecar::claim_plan::RouteATimeClaimPlan::poseidon_stage_required_for_step_witness(step)?;
    if cycle_enabled && !cfg!(feature = "poseidon-precompile") {
        return Err(PiCcsError::InvalidInput(format!(
            "step {} uses Poseidon2 precompile instructions, but feature `poseidon-precompile` is disabled",
            step_idx
        )));
    }
    if !cycle_enabled {
        return Ok(PoseidonProverSetup {
            cycle_enabled,
            sidecar: None,
            cycle_wit: None,
            cycle_open_spec: None,
            local_wit_full: None,
            local_wits: None,
            local_open_specs: None,
            local_t_len: None,
            local_layout: None,
            local_ell: None,
        });
    }

    let sidecar =
        crate::memory_sidecar::memory::build_poseidon_sidecar_table_from_step_witness(params, step, poseidon_carry)?;
    let (cycle_z_raw, cycle_m_in, cycle_t_len, cycle_open_cols) =
        crate::memory_sidecar::memory::build_poseidon_cycle_trace_matrix(step, &sidecar)?;
    let (local_z_raw, _local_m_in, local_t_len, local_layout) =
        crate::memory_sidecar::memory::build_poseidon_local_trace_matrix(&sidecar)?;
    let local_ell = local_ell_from_t_len(local_t_len)?;
    Ok(PoseidonProverSetup {
        cycle_enabled,
        sidecar: Some(sidecar),
        cycle_wit: Some(cycle_z_raw),
        cycle_open_spec: Some((cycle_m_in, cycle_t_len, cycle_open_cols)),
        local_wit_full: Some(local_z_raw),
        local_wits: None,
        local_open_specs: None,
        local_t_len: Some(local_t_len),
        local_layout: Some(local_layout),
        local_ell: Some(local_ell),
    })
}

pub(crate) fn absorb_poseidon_lane_commitments_prover(
    tr: &mut Poseidon2Transcript,
    cycle_commits: &[Cmt],
    local_commits: &[Cmt],
) {
    let mut comms = Vec::with_capacity(cycle_commits.len() + local_commits.len());
    comms.extend(cycle_commits.iter().cloned());
    comms.extend(local_commits.iter().cloned());
    ts::absorb_ajtai_commitments(tr, b"poseidon/commit/count", b"poseidon/commit/idx", &comms);
}

pub(crate) struct PoseidonVerifySetup {
    pub cycle_enabled: bool,
    pub local_ell: Option<usize>,
}

pub(crate) fn build_poseidon_verify_setup(
    _tr: &mut Poseidon2Transcript,
    step: &StepInstanceBundle<Cmt, F, K>,
    step_proof: &StepProof,
    step_idx: usize,
) -> Result<PoseidonVerifySetup, PiCcsError> {
    let cycle_enabled =
        crate::memory_sidecar::claim_plan::RouteATimeClaimPlan::poseidon_stage_required_for_step_instance(step)?;
    if cycle_enabled && !cfg!(feature = "poseidon-precompile") {
        return Err(PiCcsError::InvalidInput(format!(
            "step {} uses Poseidon2 precompile instructions, but feature `poseidon-precompile` is disabled",
            step_idx
        )));
    }
    if !cycle_enabled {
        if step_proof.poseidon_local_time.is_some()
            || !step_proof.mem.poseidon_cycle_me_claims.is_empty()
            || !step_proof.mem.poseidon_local_me_claims.is_empty()
            || !step_proof.poseidon_cycle_fold.is_empty()
            || !step_proof.poseidon_local_fold.is_empty()
        {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: unexpected poseidon proof artifact(s) when stage is not required",
                step_idx
            )));
        }
        return Ok(PoseidonVerifySetup {
            cycle_enabled,
            local_ell: None,
        });
    }

    let local_ell = step_proof
        .mem
        .poseidon_local_me_claims
        .first()
        .map(|me| me.r.len())
        .ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "step {}: missing poseidon_local_me_claim(s) when stage is required",
                step_idx
            ))
        })?;
    if local_ell == 0 {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: invalid poseidon local challenge length 0",
            step_idx
        )));
    }
    if step_proof
        .mem
        .poseidon_local_me_claims
        .iter()
        .any(|me| me.r.len() != local_ell)
    {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: inconsistent poseidon_local_me_claim challenge dimensions",
            step_idx
        )));
    }
    if step_proof.poseidon_local_time.is_none() {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: missing poseidon_local_time proof when stage is required",
            step_idx
        )));
    }
    Ok(PoseidonVerifySetup {
        cycle_enabled,
        local_ell: Some(local_ell),
    })
}

pub(crate) fn absorb_poseidon_lane_commitments_verifier(
    tr: &mut Poseidon2Transcript,
    step_proof: &StepProof,
    step_idx: usize,
    cycle_enabled: bool,
) -> Result<(), PiCcsError> {
    if !cycle_enabled {
        return Ok(());
    }
    if step_proof.mem.poseidon_cycle_me_claims.is_empty() {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: missing poseidon_cycle_me_claim for commitment binding",
            step_idx
        )));
    }
    if step_proof.mem.poseidon_local_me_claims.is_empty() {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: missing poseidon_local_me_claim(s) for commitment binding",
            step_idx
        )));
    }
    let mut comms = Vec::with_capacity(
        step_proof.mem.poseidon_cycle_me_claims.len() + step_proof.mem.poseidon_local_me_claims.len(),
    );
    comms.extend(
        step_proof
            .mem
            .poseidon_cycle_me_claims
            .iter()
            .map(|me| me.c.clone()),
    );
    comms.extend(
        step_proof
            .mem
            .poseidon_local_me_claims
            .iter()
            .map(|me| me.c.clone()),
    );
    ts::absorb_ajtai_commitments(tr, b"poseidon/commit/count", b"poseidon/commit/idx", &comms);
    Ok(())
}

pub(crate) fn verify_poseidon_local_time_from_setup(
    tr: &mut Poseidon2Transcript,
    step_proof: &StepProof,
    step_idx: usize,
    setup: &PoseidonVerifySetup,
) -> Result<Option<(Vec<K>, Vec<K>, Vec<K>)>, PiCcsError> {
    if !setup.cycle_enabled {
        return Ok(None);
    }
    let local_proof = step_proof.poseidon_local_time.as_ref().ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "step {}: missing poseidon_local_time proof when stage is required",
            step_idx
        ))
    })?;
    let local_ell = setup
        .local_ell
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local ell".into()))?;
    let local_anchor = ts::sample_ext_point(
        tr,
        b"route_a/r_local_anchor",
        b"route_a/local_anchor/0",
        b"route_a/local_anchor/1",
        local_ell,
    );
    let crate::memory_sidecar::route_a_time::PoseidonLocalTimeVerifyOutput {
        r_local,
        final_values: local_final_values,
    } = crate::memory_sidecar::route_a_time::verify_poseidon_local_time(tr, step_idx, local_ell, local_proof)?;
    Ok(Some((r_local, local_final_values, local_anchor)))
}

pub(crate) struct PoseidonVerifierRuntime {
    pub cycle_enabled: bool,
    pub verify_setup: PoseidonVerifySetup,
    pub link_chals: Option<crate::memory_sidecar::memory::PoseidonLinkChallenges>,
    pub cont_chals: Option<crate::memory_sidecar::memory::PoseidonContinuityChallenges>,
}

pub(crate) fn prepare_poseidon_verifier_runtime(
    tr: &mut Poseidon2Transcript,
    step: &StepInstanceBundle<Cmt, F, K>,
    step_proof: &StepProof,
    step_idx: usize,
) -> Result<PoseidonVerifierRuntime, PiCcsError> {
    let poseidon_verify_setup = build_poseidon_verify_setup(tr, step, step_proof, step_idx)?;
    let cycle_enabled = poseidon_verify_setup.cycle_enabled;
    let (link_chals, cont_chals) = if cycle_enabled {
        let chals = crate::memory_sidecar::memory::sample_poseidon_link_challenges(tr);
        let cont = crate::memory_sidecar::memory::sample_poseidon_continuity_challenges(tr);
        (Some(chals), Some(cont))
    } else {
        (None, None)
    };
    absorb_poseidon_lane_commitments_verifier(tr, step_proof, step_idx, cycle_enabled)?;
    Ok(PoseidonVerifierRuntime {
        cycle_enabled,
        verify_setup: poseidon_verify_setup,
        link_chals,
        cont_chals,
    })
}

pub(crate) fn ensure_poseidon_link_sums_match_verify(
    poseidon_cycle_enabled: bool,
    batched_time: &BatchedTimeProof,
    poseidon_local_time: Option<&BatchedTimeProof>,
) -> Result<(), PiCcsError> {
    if !poseidon_cycle_enabled {
        return Ok(());
    }
    let cycle_sum_idx = batched_time
        .labels
        .iter()
        .position(|l| l.as_slice() == b"poseidon/link_cycle_sum")
        .ok_or_else(|| {
            PiCcsError::ProtocolError("missing poseidon/link_cycle_sum claim in batched_time proof".into())
        })?;
    let local_proof = poseidon_local_time
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon_local_time proof for link-sum check".into()))?;
    let local_sum_idx = local_proof
        .labels
        .iter()
        .position(|l| l.as_slice() == b"poseidon/link_local_sum")
        .ok_or_else(|| {
            PiCcsError::ProtocolError("missing poseidon/link_local_sum claim in poseidon_local_time proof".into())
        })?;
    if batched_time.claimed_sums[cycle_sum_idx] != local_proof.claimed_sums[local_sum_idx] {
        return Err(PiCcsError::ProtocolError(
            "poseidon compressed-link sum mismatch (cycle != local)".into(),
        ));
    }
    Ok(())
}

pub(crate) struct PoseidonCycleTimeClaims {
    pub io_link: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim>,
    pub bitness: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim>,
    pub canonical_u64: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim>,
    pub sidecar_link: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim>,
    pub mode: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim>,
    pub link_cycle_inv: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim>,
    pub link_cycle_sum: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim>,
    pub cont_inv: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim>,
    pub cont_sum: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim>,
}

pub(crate) fn build_poseidon_cycle_time_claims(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
    ell_n: usize,
    poseidon_cycle_enabled: bool,
    poseidon_sidecar: Option<&neo_memory::riscv::exec_table::RiscvPoseidonSidecarTable>,
    poseidon_cycle_wit: Option<&Mat<F>>,
    poseidon_cycle_open_spec: Option<&(usize, usize, Vec<usize>)>,
    poseidon_link_chals: Option<&crate::memory_sidecar::memory::PoseidonLinkChallenges>,
    poseidon_cont_chals: Option<&crate::memory_sidecar::memory::PoseidonContinuityChallenges>,
) -> Result<PoseidonCycleTimeClaims, PiCcsError> {
    let mut out = PoseidonCycleTimeClaims {
        io_link: None,
        bitness: None,
        canonical_u64: None,
        sidecar_link: None,
        mode: None,
        link_cycle_inv: None,
        link_cycle_sum: None,
        cont_inv: None,
        cont_sum: None,
    };
    if !poseidon_cycle_enabled {
        return Ok(out);
    }

    let sidecar_ref =
        poseidon_sidecar.ok_or_else(|| PiCcsError::ProtocolError("missing poseidon sidecar table".into()))?;
    let (
        poseidon_io_link_built,
        poseidon_bitness_built,
        poseidon_canonical_u64_built,
        poseidon_sidecar_link_built,
        poseidon_mode_built,
    ) = crate::memory_sidecar::memory::build_route_a_poseidon_cycle_claims(
        params,
        step,
        r_cycle,
        true,
        Some(sidecar_ref),
    )?;
    out.io_link =
        poseidon_io_link_built.map(
            |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"poseidon/io_link",
            },
        );
    out.bitness =
        poseidon_bitness_built.map(
            |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"poseidon/bitness",
            },
        );
    out.canonical_u64 =
        poseidon_canonical_u64_built.map(
            |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"poseidon/canonical_u64",
            },
        );
    out.sidecar_link =
        poseidon_sidecar_link_built.map(
            |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"poseidon/sidecar_link",
            },
        );
    out.mode = poseidon_mode_built.map(
        |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
            oracle,
            claimed_sum: K::ZERO,
            label: b"poseidon/mode",
        },
    );

    let cycle_z =
        poseidon_cycle_wit.ok_or_else(|| PiCcsError::ProtocolError("missing poseidon cycle witness".into()))?;
    let cycle_layout = crate::memory_sidecar::memory::PoseidonCycleTraceLayout::new();
    let cycle_open_spec = poseidon_cycle_open_spec
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon cycle opening spec".into()))?;
    let link_chals =
        poseidon_link_chals.ok_or_else(|| PiCcsError::ProtocolError("missing poseidon link challenges".into()))?;
    let cont_chals = poseidon_cont_chals
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon continuity challenges".into()))?;
    let (cycle_link_inv_built, cycle_link_sum_built) =
        crate::memory_sidecar::memory::build_route_a_poseidon_cycle_link_claims(
            cycle_z,
            cycle_open_spec.1,
            cycle_open_spec.0,
            ell_n,
            &cycle_layout,
            r_cycle,
            link_chals,
        )?;
    let (cycle_cont_inv_built, cycle_cont_sum_built) =
        crate::memory_sidecar::memory::build_route_a_poseidon_cycle_continuity_claims(
            cycle_z,
            cycle_open_spec.1,
            cycle_open_spec.0,
            ell_n,
            &cycle_layout,
            r_cycle,
            cont_chals,
        )?;
    out.link_cycle_inv =
        cycle_link_inv_built.map(
            |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"poseidon/link_cycle_inv",
            },
        );
    out.link_cycle_sum =
        cycle_link_sum_built.map(
            |(oracle, claimed_sum)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum,
                label: b"poseidon/link_cycle_sum",
            },
        );
    out.cont_inv = cycle_cont_inv_built.map(
        |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
            oracle,
            claimed_sum: K::ZERO,
            label: b"poseidon/cont_inv",
        },
    );
    out.cont_sum = cycle_cont_sum_built.map(
        |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
            oracle,
            claimed_sum: K::ZERO,
            label: b"poseidon/cont_sum",
        },
    );
    Ok(out)
}

pub(crate) struct PoseidonLocalTimeArtifacts {
    pub local_time: Option<BatchedTimeProof>,
    pub r_local: Option<Vec<K>>,
}

pub(crate) fn prove_poseidon_local_time_artifacts(
    tr: &mut Poseidon2Transcript,
    step_idx: usize,
    poseidon_cycle_enabled: bool,
    poseidon_local_ell: Option<usize>,
    poseidon_local_open_specs: Option<&Vec<(usize, usize, Vec<usize>)>>,
    poseidon_local_t_len: Option<usize>,
    poseidon_local_layout: Option<crate::memory_sidecar::memory::PoseidonLocalTraceLayout>,
    poseidon_local_wit_full: Option<&Mat<F>>,
    poseidon_link_chals: Option<&crate::memory_sidecar::memory::PoseidonLinkChallenges>,
) -> Result<PoseidonLocalTimeArtifacts, PiCcsError> {
    if !poseidon_cycle_enabled {
        return Ok(PoseidonLocalTimeArtifacts {
            local_time: None,
            r_local: None,
        });
    }

    let base_ell_local =
        poseidon_local_ell.ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local ell".into()))?;
    let local_t_len =
        poseidon_local_t_len.ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local t_len".into()))?;
    let local_layout =
        poseidon_local_layout.ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local layout".into()))?;
    let local_open_specs = poseidon_local_open_specs
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local opening specs".into()))?;
    let local_wit_ref =
        poseidon_local_wit_full.ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local witness".into()))?;
    let mut ell_local = base_ell_local;
    for (local_m_in, local_t, _) in local_open_specs.iter() {
        ell_local = core::cmp::max(
            ell_local,
            required_ell_for_time_rows(*local_m_in, *local_t, "poseidon local proving ell")?,
        );
    }
    let r_local_anchor = ts::sample_ext_point(
        tr,
        b"route_a/r_local_anchor",
        b"route_a/local_anchor/0",
        b"route_a/local_anchor/1",
        ell_local,
    );
    let (round_built, transition_built, link_built) =
        crate::memory_sidecar::memory::build_route_a_poseidon_local_claims(
            local_wit_ref,
            local_t_len,
            ell_local,
            &local_layout,
            &r_local_anchor,
        )?;
    let link_chals =
        poseidon_link_chals.ok_or_else(|| PiCcsError::ProtocolError("missing poseidon link challenges".into()))?;
    let (local_link_inv_built, local_link_sum_built) =
        crate::memory_sidecar::memory::build_route_a_poseidon_local_link_claims(
            local_wit_ref,
            local_t_len,
            /*local_m_in=*/ 0,
            ell_local,
            &local_layout,
            &r_local_anchor,
            link_chals,
        )?;
    let mut poseidon_local_round_claim =
        round_built.map(
            |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"poseidon/round",
            },
        );
    let mut poseidon_local_transition_claim =
        transition_built.map(
            |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"poseidon/transition",
            },
        );
    let mut poseidon_local_link_claim =
        link_built.map(
            |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"poseidon/cycle_local_link",
            },
        );
    let mut poseidon_local_link_inv_claim =
        local_link_inv_built.map(
            |(oracle, _)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"poseidon/link_local_inv",
            },
        );
    let mut poseidon_local_link_sum_claim =
        local_link_sum_built.map(
            |(oracle, claimed_sum)| crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum,
                label: b"poseidon/link_local_sum",
            },
        );
    let crate::memory_sidecar::route_a_time::PoseidonLocalTimeProverOutput { r_local, proof } =
        crate::memory_sidecar::route_a_time::prove_poseidon_local_time(
            tr,
            step_idx,
            ell_local,
            poseidon_local_round_claim.take(),
            poseidon_local_transition_claim.take(),
            poseidon_local_link_claim.take(),
            poseidon_local_link_inv_claim.take(),
            poseidon_local_link_sum_claim.take(),
        )?;
    Ok(PoseidonLocalTimeArtifacts {
        local_time: Some(proof),
        r_local: Some(r_local),
    })
}

pub(crate) fn ensure_poseidon_link_sums_match(
    poseidon_cycle_enabled: bool,
    batched_time: &BatchedTimeProof,
    poseidon_local_time: Option<&BatchedTimeProof>,
) -> Result<(), PiCcsError> {
    if !poseidon_cycle_enabled {
        return Ok(());
    }
    let cycle_sum_idx = batched_time
        .labels
        .iter()
        .position(|l| l.as_slice() == b"poseidon/link_cycle_sum")
        .ok_or_else(|| {
            PiCcsError::ProtocolError("missing poseidon/link_cycle_sum claim in batched_time proof".into())
        })?;
    let local_proof =
        poseidon_local_time.ok_or_else(|| PiCcsError::ProtocolError("missing poseidon_local_time proof".into()))?;
    let local_sum_idx = local_proof
        .labels
        .iter()
        .position(|l| l.as_slice() == b"poseidon/link_local_sum")
        .ok_or_else(|| {
            PiCcsError::ProtocolError("missing poseidon/link_local_sum claim in poseidon_local_time proof".into())
        })?;
    if batched_time.claimed_sums[cycle_sum_idx] != local_proof.claimed_sums[local_sum_idx] {
        return Err(PiCcsError::ProtocolError(
            "poseidon compressed-link sum mismatch during proving (cycle != local)".into(),
        ));
    }
    Ok(())
}

pub(crate) struct PoseidonMeClaimsInputs<'a> {
    pub poseidon_cycle_enabled: bool,
    pub poseidon_cycle_wits: Option<&'a Vec<Mat<F>>>,
    pub poseidon_cycle_commits: Option<&'a Vec<Cmt>>,
    pub poseidon_cycle_open_specs: Option<&'a Vec<(usize, usize, Vec<usize>)>>,
    pub poseidon_local_wits: Option<&'a Vec<Mat<F>>>,
    pub poseidon_local_commits: Option<&'a Vec<Cmt>>,
    pub poseidon_local_open_specs: Option<&'a Vec<(usize, usize, Vec<usize>)>>,
    pub poseidon_local_t_len: Option<usize>,
    pub poseidon_local_layout: Option<crate::memory_sidecar::memory::PoseidonLocalTraceLayout>,
    pub poseidon_r_local: Option<&'a Vec<K>>,
}

fn poseidon_open_cols_to_time_vectors(
    params: &NeoParams,
    expected_m: usize,
    z: &Mat<F>,
    t_len: usize,
    open_cols: &[usize],
    col_base: usize,
    label: &str,
) -> Result<(Vec<Vec<F>>, Vec<usize>), PiCcsError> {
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput(format!("{label}: t_len must be > 0")));
    }
    let logical = neo_memory::ajtai::decode_vector_for_ccs_m(params, expected_m, z).map_err(|e| {
        PiCcsError::ProtocolError(format!(
            "{label}: failed to decode packed witness coefficients for m={expected_m}: {e}"
        ))
    })?;
    let mut cols_vals: Vec<Vec<F>> = Vec::with_capacity(open_cols.len());
    for &col_id in open_cols.iter() {
        let start = col_base
            .checked_add(
                col_id
                    .checked_mul(t_len)
                    .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: col_id * t_len overflow")))?,
            )
            .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: col_base + col span overflow")))?;
        let end = start
            .checked_add(t_len)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: col span overflow")))?;
        if end > logical.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "{label}: opening col out of range (col_id={col_id}, start={start}, end={end}, z.cols()={})",
                logical.len()
            )));
        }
        cols_vals.push(logical[start..end].to_vec());
    }
    let cols_idx: Vec<usize> = (0..cols_vals.len()).collect();
    Ok((cols_vals, cols_idx))
}

fn required_ell_for_time_rows(m_in: usize, t_len: usize, label: &str) -> Result<usize, PiCcsError> {
    let end = m_in
        .checked_add(t_len)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: m_in + t_len overflow")))?;
    let mut n = 1usize;
    let mut ell = 0usize;
    while n < end {
        n = n
            .checked_mul(2)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: 2^ell overflow")))?;
        ell += 1;
    }
    Ok(ell)
}

fn validate_poseidon_tile_metadata(
    claims: &[CeClaim<Cmt, F, K>],
    lane_base_m_in: usize,
    lane_label: &str,
) -> Result<(), PiCcsError> {
    let mut prev_offset: Option<usize> = None;
    for me in claims.iter() {
        if me.m_in != lane_base_m_in {
            return Err(PiCcsError::ProtocolError(format!(
                "{lane_label} ME claim m_in mismatch (got {}, expected lane base {})",
                me.m_in, lane_base_m_in
            )));
        }
        if me.u_len == 0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{lane_label} ME claim must have non-zero u_len"
            )));
        }
        if let Some(prev) = prev_offset {
            if me.u_offset < prev {
                return Err(PiCcsError::ProtocolError(format!(
                    "{lane_label} ME claim u_offset order mismatch (prev {}, next {})",
                    prev, me.u_offset
                )));
            }
        }
        prev_offset = Some(me.u_offset);
    }
    Ok(())
}

pub(crate) fn emit_poseidon_me_claims(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    r_time: &[K],
    ell_t: usize,
    ell_d: usize,
    mem_proof: &mut MemSidecarProof<Cmt, F, K>,
    input: PoseidonMeClaimsInputs<'_>,
) -> Result<(), PiCcsError> {
    if !input.poseidon_cycle_enabled {
        return Ok(());
    }

    let cycle_z_chunks = input
        .poseidon_cycle_wits
        .ok_or_else(|| PiCcsError::ProtocolError("missing prebuilt poseidon cycle witness chunks".into()))?;
    let cycle_cs = input
        .poseidon_cycle_commits
        .ok_or_else(|| PiCcsError::ProtocolError("missing prebound poseidon cycle commitment(s)".into()))?;
    let cycle_open_specs = input
        .poseidon_cycle_open_specs
        .ok_or_else(|| PiCcsError::ProtocolError("missing prebuilt poseidon cycle opening specs".into()))?;
    if cycle_z_chunks.len() != cycle_open_specs.len() || cycle_z_chunks.len() != cycle_cs.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon cycle chunk mismatch (wits={}, open_specs={}, commits={})",
            cycle_z_chunks.len(),
            cycle_open_specs.len(),
            cycle_cs.len()
        )));
    }
    let cycle_col_base = cycle_open_specs
        .iter()
        .map(|(chunk_m_in, _, _)| *chunk_m_in)
        .min()
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon cycle opening specs".into()))?;
    let cycle_claim_m_in = cycle_col_base;
    let mut cycle_ell = ell_t.max(r_time.len());
    for (cycle_m_in, cycle_t_len, _) in cycle_open_specs.iter() {
        cycle_ell = core::cmp::max(
            cycle_ell,
            required_ell_for_time_rows(*cycle_m_in, *cycle_t_len, "poseidon cycle ell")?,
        );
    }
    let mut r_cycle_for_me = r_time.to_vec();
    r_cycle_for_me.resize(cycle_ell, K::ZERO);
    let mut s_cycle = s.clone();
    s_cycle.n = 1usize
        .checked_shl(cycle_ell as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("poseidon cycle n overflow".into()))?;
    let cycle_me_ctx = ts::precompute_me_claims_context(tr, b"poseidon/me_cycle_time", &s_cycle, &r_cycle_for_me);
    let mut cycle_claims: Vec<CeClaim<Cmt, F, K>> = Vec::new();
    let mut cycle_idx = 0usize;
    while cycle_idx < cycle_z_chunks.len() {
        let (cycle_m_in, cycle_t_len, _) = &cycle_open_specs[cycle_idx];
        let mut cycle_group_end = cycle_idx + 1;
        while cycle_group_end < cycle_open_specs.len() {
            let (next_m_in, next_t_len, _) = &cycle_open_specs[cycle_group_end];
            if next_m_in != cycle_m_in || next_t_len != cycle_t_len {
                break;
            }
            cycle_group_end += 1;
        }

        let group_claims = ts::emit_me_claims_for_mats_with_context(
            params,
            &s_cycle,
            &cycle_cs[cycle_idx..cycle_group_end],
            &cycle_z_chunks[cycle_idx..cycle_group_end],
            &r_cycle_for_me,
            cycle_claim_m_in,
            &cycle_me_ctx,
        )?;
        if group_claims.len() != (cycle_group_end - cycle_idx) {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon cycle grouped ME claim count mismatch (expected {}, got {})",
                cycle_group_end - cycle_idx,
                group_claims.len()
            )));
        }

        for (group_off, mut me) in group_claims.into_iter().enumerate() {
            let spec_idx = cycle_idx + group_off;
            let cycle_z = &cycle_z_chunks[spec_idx];
            let (_, _, cycle_open_cols) = &cycle_open_specs[spec_idx];
            let (cycle_cols_vals, cycle_cols_idx) = poseidon_open_cols_to_time_vectors(
                params,
                s.m,
                cycle_z,
                *cycle_t_len,
                cycle_open_cols,
                cycle_col_base,
                "poseidon cycle ME openings",
            )?;
            crate::memory_sidecar::cpu_bus::append_time_columns_openings_to_me_instance_with_row_base(
                params,
                cycle_claim_m_in,
                *cycle_m_in,
                *cycle_t_len,
                &cycle_cols_vals,
                &cycle_cols_idx,
                s.t(),
                &mut me,
            )?;
            if me.m_in != cycle_claim_m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "poseidon cycle ME claim m_in mismatch (got {}, expected lane base {})",
                    me.m_in, cycle_claim_m_in
                )));
            }
            me.u_offset = *cycle_m_in;
            me.u_len = *cycle_t_len;
            if me.u_len == 0 {
                return Err(PiCcsError::ProtocolError(
                    "poseidon cycle ME claim must have non-zero u_len".into(),
                ));
            }
            let t = me.y_ring.len();
            normalize_me_claims(core::slice::from_mut(&mut me), cycle_ell, ell_d, t)?;
            cycle_claims.push(me);
        }

        cycle_idx = cycle_group_end;
    }
    validate_poseidon_tile_metadata(cycle_claims.as_slice(), cycle_claim_m_in, "poseidon cycle")?;
    mem_proof.poseidon_cycle_me_claims = cycle_claims;

    let local_z_chunks = input
        .poseidon_local_wits
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local witness chunks".into()))?;
    let local_t_len = input
        .poseidon_local_t_len
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local t_len".into()))?;
    let _local_layout = input
        .poseidon_local_layout
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local layout".into()))?;
    let local_open_specs = input
        .poseidon_local_open_specs
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local opening specs".into()))?;
    let local_cs = input
        .poseidon_local_commits
        .ok_or_else(|| PiCcsError::ProtocolError("missing prebound poseidon local commitment(s)".into()))?;
    if local_z_chunks.len() != local_open_specs.len() || local_z_chunks.len() != local_cs.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon local chunk mismatch (wits={}, open_specs={}, commits={})",
            local_z_chunks.len(),
            local_open_specs.len(),
            local_cs.len()
        )));
    }
    let local_col_base = local_open_specs
        .iter()
        .map(|(chunk_m_in, _, _)| *chunk_m_in)
        .min()
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local opening specs".into()))?;
    let local_claim_m_in = local_col_base;
    let r_local = input
        .poseidon_r_local
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local r_local".into()))?;
    let local_ell_from_t = input
        .poseidon_local_t_len
        .map(|t| t.trailing_zeros() as usize)
        .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local ell".into()))?;
    if r_local.len() < local_ell_from_t {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon local r_local length {} is smaller than local_ell {}",
            r_local.len(),
            local_ell_from_t
        )));
    }
    let mut local_ell = r_local.len();
    for (local_m_in, local_t, _) in local_open_specs.iter() {
        local_ell = core::cmp::max(
            local_ell,
            required_ell_for_time_rows(*local_m_in, *local_t, "poseidon local ell")?,
        );
    }
    let mut r_local_for_me = r_local.clone();
    r_local_for_me.resize(local_ell, K::ZERO);
    let mut s_local = s.clone();
    s_local.n = 1usize
        .checked_shl(local_ell as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("poseidon local n overflow".into()))?;
    let local_me_ctx = ts::precompute_me_claims_context(tr, b"poseidon/me_local_time", &s_local, &r_local_for_me);
    let mut local_claims: Vec<CeClaim<Cmt, F, K>> = Vec::new();
    let mut local_idx = 0usize;
    while local_idx < local_z_chunks.len() {
        let (local_m_in, local_t, _) = &local_open_specs[local_idx];
        let mut local_group_end = local_idx + 1;
        while local_group_end < local_open_specs.len() {
            let (next_m_in, next_t, _) = &local_open_specs[local_group_end];
            if next_m_in != local_m_in || next_t != local_t {
                break;
            }
            local_group_end += 1;
        }

        let local_end = local_m_in
            .checked_add(*local_t)
            .ok_or_else(|| PiCcsError::InvalidInput("poseidon local chunk time span overflow".into()))?;
        if local_end > local_t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon local chunk time span out of range (m_in={}, t_len={}, local_t_len={})",
                local_m_in, local_t, local_t_len
            )));
        }

        let group_claims = ts::emit_me_claims_for_mats_with_context(
            params,
            &s_local,
            &local_cs[local_idx..local_group_end],
            &local_z_chunks[local_idx..local_group_end],
            &r_local_for_me,
            local_claim_m_in,
            &local_me_ctx,
        )?;
        if group_claims.len() != (local_group_end - local_idx) {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon local grouped ME claim count mismatch (expected {}, got {})",
                local_group_end - local_idx,
                group_claims.len()
            )));
        }

        for (group_off, mut me) in group_claims.into_iter().enumerate() {
            let spec_idx = local_idx + group_off;
            let local_z = &local_z_chunks[spec_idx];
            let (_, _, local_cols) = &local_open_specs[spec_idx];
            let (local_cols_vals, local_cols_idx) = poseidon_open_cols_to_time_vectors(
                params,
                s.m,
                local_z,
                *local_t,
                local_cols,
                local_col_base,
                "poseidon local ME openings",
            )?;
            crate::memory_sidecar::cpu_bus::append_time_columns_openings_to_me_instance_with_row_base(
                params,
                local_claim_m_in,
                *local_m_in,
                *local_t,
                &local_cols_vals,
                &local_cols_idx,
                s.t(),
                &mut me,
            )?;
            if me.m_in != local_claim_m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "poseidon local ME claim m_in mismatch (got {}, expected lane base {})",
                    me.m_in, local_claim_m_in
                )));
            }
            me.u_offset = *local_m_in;
            me.u_len = *local_t;
            if me.u_len == 0 {
                return Err(PiCcsError::ProtocolError(
                    "poseidon local ME claim must have non-zero u_len".into(),
                ));
            }
            let t = me.y_ring.len();
            normalize_me_claims(core::slice::from_mut(&mut me), local_ell, ell_d, t)?;
            local_claims.push(me);
        }

        local_idx = local_group_end;
    }
    validate_poseidon_tile_metadata(local_claims.as_slice(), local_claim_m_in, "poseidon local")?;
    mem_proof.poseidon_local_me_claims = local_claims;

    Ok(())
}

pub(crate) struct PoseidonFoldLanes {
    pub cycle_fold: Vec<RlcDecProof>,
    pub local_fold: Vec<RlcDecProof>,
}

fn poseidon_strip_me_for_fold(
    me: &CeClaim<Cmt, F, K>,
    core_t: usize,
    fold_m_in: usize,
) -> Result<CeClaim<Cmt, F, K>, PiCcsError> {
    if me.y_ring.len() < core_t || me.ct.len() < core_t {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon fold expects ME core rows at least core_t={} (y.len()={}, ct.len()={})",
            core_t,
            me.y_ring.len(),
            me.ct.len()
        )));
    }
    let mut out = me.clone();
    out.m_in = fold_m_in;
    if out.X.cols() < fold_m_in {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon fold strip cannot normalize X width: X.cols()={} < fold_m_in={}",
            out.X.cols(),
            fold_m_in
        )));
    }
    if out.X.cols() != fold_m_in {
        let mut x_norm = Mat::zero(out.X.rows(), fold_m_in, F::ZERO);
        for r in 0..out.X.rows() {
            for c in 0..fold_m_in {
                x_norm[(r, c)] = out.X[(r, c)];
            }
        }
        out.X = x_norm;
    }
    out.y_ring.truncate(core_t);
    out.ct.truncate(core_t);
    out.aux_openings.clear();
    out.u_offset = 0;
    out.u_len = 0;
    Ok(out)
}

pub(crate) fn prove_poseidon_fold_lanes<L, MR, MB>(
    mode: &FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    _ccs_sparse_cache: Option<&SparseCache<F>>,
    ring: &ccs::RotRing,
    ell_d: usize,
    step_idx: usize,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    poseidon_cycle_wits: Option<&Vec<Mat<F>>>,
    poseidon_cycle_open_specs: Option<&Vec<(usize, usize, Vec<usize>)>>,
    poseidon_local_wits: Option<&Vec<Mat<F>>>,
    poseidon_local_open_specs: Option<&Vec<(usize, usize, Vec<usize>)>>,
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<PoseidonFoldLanes, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let mut cycle_fold: Vec<RlcDecProof> = Vec::new();
    if !mem_proof.poseidon_cycle_me_claims.is_empty() {
        tr.append_message(b"fold/poseidon_cycle_lane_start", &(step_idx as u64).to_le_bytes());
        let cycle_wits = poseidon_cycle_wits
            .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon cycle witness chunks for fold lane".into()))?;
        let cycle_open_specs = poseidon_cycle_open_specs
            .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon cycle opening specs for fold lane".into()))?;
        if mem_proof.poseidon_cycle_me_claims.len() != cycle_wits.len()
            || mem_proof.poseidon_cycle_me_claims.len() != cycle_open_specs.len()
        {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon cycle fold shape mismatch (claims={}, wits={}, specs={})",
                mem_proof.poseidon_cycle_me_claims.len(),
                cycle_wits.len(),
                cycle_open_specs.len()
            )));
        }
        let k_dec_poseidon = 64usize;
        let cycle_fold_m_in = mem_proof
            .poseidon_cycle_me_claims
            .iter()
            .map(|me| me.m_in)
            .min()
            .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon cycle ME claim(s)".into()))?;
        let mut cycle_claims_for_fold: Vec<CeClaim<Cmt, F, K>> =
            Vec::with_capacity(mem_proof.poseidon_cycle_me_claims.len());
        for me in mem_proof.poseidon_cycle_me_claims.iter() {
            cycle_claims_for_fold.push(poseidon_strip_me_for_fold(me, s.t(), cycle_fold_m_in)?);
        }
        let cycle_open_specs_slim: Vec<(usize, usize, &[usize])> = cycle_open_specs
            .iter()
            .map(|(m_in, t_len, open_cols)| (*m_in, *t_len, open_cols.as_slice()))
            .collect();
        cycle_fold = prove_poseidon_lane_fold(
            mode,
            tr,
            params,
            s,
            None,
            ring,
            ell_d,
            k_dec_poseidon,
            step_idx,
            cycle_claims_for_fold.as_slice(),
            cycle_wits.as_slice(),
            cycle_open_specs_slim.as_slice(),
            s.t(),
            l,
            mixers,
        )
        .map_err(|e| PiCcsError::ProtocolError(format!("poseidon cycle fold lane failed: {e}")))?;
    }

    let mut local_fold: Vec<RlcDecProof> = Vec::new();
    if !mem_proof.poseidon_local_me_claims.is_empty() {
        tr.append_message(b"fold/poseidon_local_lane_start", &(step_idx as u64).to_le_bytes());
        let local_wits = poseidon_local_wits
            .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local witness chunks for fold lane".into()))?;
        let local_open_specs = poseidon_local_open_specs
            .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local opening specs for fold lane".into()))?;
        if mem_proof.poseidon_local_me_claims.len() != local_wits.len()
            || mem_proof.poseidon_local_me_claims.len() != local_open_specs.len()
        {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon local fold shape mismatch (claims={}, wits={}, specs={})",
                mem_proof.poseidon_local_me_claims.len(),
                local_wits.len(),
                local_open_specs.len()
            )));
        }
        let k_dec_poseidon = 64usize;
        let local_fold_m_in = mem_proof
            .poseidon_local_me_claims
            .iter()
            .map(|me| me.m_in)
            .min()
            .ok_or_else(|| PiCcsError::ProtocolError("missing poseidon local ME claim(s)".into()))?;
        let mut local_claims_for_fold: Vec<CeClaim<Cmt, F, K>> =
            Vec::with_capacity(mem_proof.poseidon_local_me_claims.len());
        for me in mem_proof.poseidon_local_me_claims.iter() {
            local_claims_for_fold.push(poseidon_strip_me_for_fold(me, s.t(), local_fold_m_in)?);
        }
        let local_open_specs_slim: Vec<(usize, usize, &[usize])> = local_open_specs
            .iter()
            .map(|(m_in, t_len, open_cols)| (*m_in, *t_len, open_cols.as_slice()))
            .collect();
        local_fold = prove_poseidon_lane_fold(
            mode,
            tr,
            params,
            s,
            None,
            ring,
            ell_d,
            k_dec_poseidon,
            step_idx,
            local_claims_for_fold.as_slice(),
            local_wits.as_slice(),
            local_open_specs_slim.as_slice(),
            s.t(),
            l,
            mixers,
        )
        .map_err(|e| PiCcsError::ProtocolError(format!("poseidon local fold lane failed: {e}")))?;
    }

    Ok(PoseidonFoldLanes { cycle_fold, local_fold })
}

pub(crate) fn verify_poseidon_fold_lanes<MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    ring: &ccs::RotRing,
    ell_d: usize,
    mixers: CommitMixers<MR, MB>,
    step_idx: usize,
    idx: usize,
    step_proof: &StepProof,
    val_lane_obligations: &mut Vec<CeClaim<Cmt, F, K>>,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let lanes = [
        (
            b"fold/poseidon_cycle_lane_start".as_slice(),
            "poseidon_cycle",
            &step_proof.mem.poseidon_cycle_me_claims,
            &step_proof.poseidon_cycle_fold,
        ),
        (
            b"fold/poseidon_local_lane_start".as_slice(),
            "poseidon_local",
            &step_proof.mem.poseidon_local_me_claims,
            &step_proof.poseidon_local_fold,
        ),
    ];
    for (start_label, lane_label, me_claims, folds) in lanes {
        if me_claims.is_empty() {
            if !folds.is_empty() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: unexpected {lane_label}_fold proof(s) (no {lane_label} ME claims)",
                    idx
                )));
            }
            continue;
        }
        tr.append_message(start_label, &(step_idx as u64).to_le_bytes());
        let fold_m_in = me_claims
            .iter()
            .map(|me| me.m_in)
            .min()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("step {}: missing {lane_label} ME claim(s)", idx)))?;
        let mut stripped_claims = Vec::with_capacity(me_claims.len());
        for me in me_claims.iter() {
            stripped_claims.push(poseidon_strip_me_for_fold(me, s.t(), fold_m_in)?);
        }
        verify_poseidon_lane_fold(
            tr,
            params,
            s,
            ring,
            ell_d,
            mixers,
            step_idx,
            idx,
            lane_label,
            &stripped_claims,
            folds,
            val_lane_obligations,
        )?;
    }
    Ok(())
}

pub(crate) fn split_poseidon_lane_wit_by_time_cols(
    _params: &NeoParams,
    z: &Mat<F>,
    selected_open_cols: &[usize],
    t_len: usize,
    m_in: usize,
    public_prefix_vals: Option<&[F]>,
    ccs_m: usize,
) -> Result<(Vec<Mat<F>>, Vec<(usize, usize, Vec<usize>)>), PiCcsError> {
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("poseidon split: t_len must be > 0".into()));
    }
    if z.rows() != neo_math::D {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon split: raw witness row count mismatch (got {}, expected {})",
            z.rows(),
            neo_math::D
        )));
    }
    if z.cols() % t_len != 0 {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon split: matrix cols must be divisible by t_len (z.cols()={}, t_len={})",
            z.cols(),
            t_len
        )));
    }
    let source_cols = z.cols() / t_len;
    if selected_open_cols.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "poseidon split: selected_open_cols must not be empty".into(),
        ));
    }
    if selected_open_cols.iter().any(|&col| col >= source_cols) {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon split: selected open column out of range (source_cols={}, max_selected={})",
            source_cols,
            selected_open_cols.iter().copied().max().unwrap_or(0)
        )));
    }
    if selected_open_cols.windows(2).any(|w| w[0] >= w[1]) {
        return Err(PiCcsError::ProtocolError(
            "poseidon split: selected_open_cols must be strictly increasing".into(),
        ));
    }
    let logical_cols = selected_open_cols.len();
    let selected_is_identity = logical_cols == source_cols
        && selected_open_cols
            .iter()
            .enumerate()
            .all(|(idx, &col_id)| idx == col_id);
    if m_in > ccs_m {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon split: m_in exceeds ccs_m (m_in={}, ccs_m={})",
            m_in, ccs_m
        )));
    }
    if let Some(prefix_vals) = public_prefix_vals {
        if prefix_vals.len() < m_in {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon split: public prefix values too short (got {}, need >= {})",
                prefix_vals.len(),
                m_in
            )));
        }
    }
    let available_raw = ccs_m - m_in;
    if available_raw == 0 {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon split: ccs_m too small after m_in offset (ccs_m={}, m_in={}, t_len={})",
            ccs_m, m_in, t_len
        )));
    }

    let packed_cols = ccs_m.div_ceil(neo_math::D);
    let mut chunk_wits: Vec<Mat<F>> = Vec::new();
    let mut chunk_specs: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    let row0 = z.row(0);

    // 2D tiling over time and columns. Each tile satisfies:
    //   chunk_cols * chunk_t_len <= ccs_m - m_in
    // which removes the old hard dependency `ccs_m - m_in >= t_len`.
    let mut time_start = 0usize;
    while time_start < t_len {
        let time_remaining = t_len - time_start;
        let chunk_t_len = core::cmp::min(time_remaining, available_raw);
        if chunk_t_len == 0 {
            return Err(PiCcsError::ProtocolError(
                "poseidon split: computed zero chunk_t_len".into(),
            ));
        }
        let max_cols_for_time = available_raw / chunk_t_len;
        if max_cols_for_time == 0 {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon split: no capacity for any column at chunk_t_len={} (ccs_m={}, m_in={})",
                chunk_t_len, ccs_m, m_in
            )));
        }

        let mut col_start = 0usize;
        while col_start < logical_cols {
            let chunk_cols = core::cmp::min(max_cols_for_time, logical_cols - col_start);
            let mut logical = vec![F::ZERO; ccs_m];
            if let Some(prefix_vals) = public_prefix_vals {
                logical[..m_in].copy_from_slice(&prefix_vals[..m_in]);
            }

            for local_col in 0..chunk_cols {
                let global_col = col_start
                    .checked_add(local_col)
                    .ok_or_else(|| PiCcsError::InvalidInput("poseidon split: global col overflow".into()))?;
                let source_col = if selected_is_identity {
                    global_col
                } else {
                    selected_open_cols[global_col]
                };
                let src_start = source_col
                    .checked_mul(t_len)
                    .and_then(|v| v.checked_add(time_start))
                    .ok_or_else(|| PiCcsError::InvalidInput("poseidon split: src start overflow".into()))?;
                let src_end = src_start
                    .checked_add(chunk_t_len)
                    .ok_or_else(|| PiCcsError::InvalidInput("poseidon split: src end overflow".into()))?;
                if src_end > row0.len() {
                    return Err(PiCcsError::ProtocolError(format!(
                        "poseidon split: src span out of range (src_start={}, src_end={}, row_len={})",
                        src_start,
                        src_end,
                        row0.len()
                    )));
                }

                let dst_start = m_in
                    .checked_add(
                        local_col
                            .checked_mul(chunk_t_len)
                            .ok_or_else(|| PiCcsError::InvalidInput("poseidon split: dst mul overflow".into()))?,
                    )
                    .ok_or_else(|| PiCcsError::InvalidInput("poseidon split: dst start overflow".into()))?;
                let dst_end = dst_start
                    .checked_add(chunk_t_len)
                    .ok_or_else(|| PiCcsError::InvalidInput("poseidon split: dst end overflow".into()))?;
                if dst_end > logical.len() {
                    return Err(PiCcsError::ProtocolError(format!(
                        "poseidon split: dst span out of range (dst_start={}, dst_end={}, ccs_m={})",
                        dst_start,
                        dst_end,
                        logical.len()
                    )));
                }
                logical[dst_start..dst_end].copy_from_slice(&row0[src_start..src_end]);
            }

            let mut chunk = Mat::zero(neo_math::D, packed_cols, F::ZERO);
            for (c, &v) in logical.iter().enumerate() {
                let blk = c / neo_math::D;
                let off = c % neo_math::D;
                chunk[(off, blk)] = v;
            }
            chunk_wits.push(chunk);
            chunk_specs.push((
                m_in.checked_add(time_start)
                    .ok_or_else(|| PiCcsError::InvalidInput("poseidon split: chunk m_in overflow".into()))?,
                chunk_t_len,
                (0..chunk_cols).collect(),
            ));
            col_start += chunk_cols;
        }
        time_start += chunk_t_len;
    }

    if chunk_wits.is_empty() {
        return Err(PiCcsError::ProtocolError("poseidon split: produced zero chunks".into()));
    }
    Ok((chunk_wits, chunk_specs))
}

pub(crate) fn prove_poseidon_lane_fold<L, MR, MB>(
    mode: &FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    ccs_sparse_cache: Option<&SparseCache<F>>,
    ring: &ccs::RotRing,
    ell_d: usize,
    k_dec: usize,
    step_idx: usize,
    me_claims: &[CeClaim<Cmt, F, K>],
    lane_wits: &[Mat<F>],
    open_specs: &[(usize, usize, &[usize])],
    _core_t: usize,
    _l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<Vec<RlcDecProof>, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if me_claims.is_empty() {
        return Ok(Vec::new());
    }
    if me_claims.len() != lane_wits.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon fold lane shape mismatch (claims={}, wits={})",
            me_claims.len(),
            lane_wits.len()
        )));
    }
    if me_claims.len() != open_specs.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon fold lane shape mismatch (claims={}, open_specs={})",
            me_claims.len(),
            open_specs.len()
        )));
    }
    let committer_cols = lane_wits[0].cols();
    if lane_wits.iter().any(|z| z.cols() != committer_cols) {
        return Err(PiCcsError::ProtocolError(
            "poseidon fold lane expects equal packed witness width across chunks".into(),
        ));
    }
    let lane_committer = poseidon_lane_committer(params, committer_cols, "poseidon lane fold")?;
    let wit_refs: Vec<&Mat<F>> = lane_wits.iter().collect();
    let (proof, _dec_wits) = prove_rlc_dec_lane(
        mode,
        RlcLane::Val,
        tr,
        params,
        s,
        ccs_sparse_cache,
        None,
        ring,
        ell_d,
        k_dec,
        step_idx,
        None,
        me_claims,
        wit_refs.as_slice(),
        false,
        &lane_committer,
        mixers,
    )?;
    Ok(vec![proof])
}

pub(crate) fn verify_poseidon_lane_fold<MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    ring: &ccs::RotRing,
    ell_d: usize,
    mixers: CommitMixers<MR, MB>,
    step_idx: usize,
    idx: usize,
    label: &str,
    me_claims: &[CeClaim<Cmt, F, K>],
    folds: &[RlcDecProof],
    val_lane_obligations: &mut Vec<CeClaim<Cmt, F, K>>,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if me_claims.is_empty() {
        if !folds.is_empty() {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: unexpected {} fold proof(s) (no ME claims)",
                idx, label
            )));
        }
        return Ok(());
    }
    if folds.len() != 1 {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {} fold count mismatch (have {}, expected 1 batched proof)",
            idx,
            label,
            folds.len(),
        )));
    }
    let proof = folds
        .first()
        .ok_or_else(|| PiCcsError::ProtocolError(format!("step {}: missing {} fold proof", idx, label)))?;
    verify_rlc_dec_lane(
        RlcLane::Val,
        tr,
        params,
        s,
        ring,
        ell_d,
        mixers,
        step_idx,
        me_claims,
        &proof.rlc_rhos,
        &proof.rlc_parent,
        &proof.dec_children,
    )
    .map_err(|e| PiCcsError::ProtocolError(format!("step {} {} batched fold verify failed: {e:?}", idx, label)))?;
    val_lane_obligations.extend_from_slice(&proof.dec_children);
    Ok(())
}
