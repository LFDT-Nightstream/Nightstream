use super::*;
use p3_field::PrimeField64;

pub(crate) enum CcsOracleDispatch<'a> {
    Optimized(neo_reductions::accelerator::SplitNcOptimizedOracle<'a, F>),
    #[cfg(feature = "paper-exact")]
    PaperExact(neo_reductions::engines::paper_exact_engine::oracle::PaperExactOracle<'a, F>),
}

impl<'a> RoundOracle for CcsOracleDispatch<'a> {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        match self {
            Self::Optimized(oracle) => oracle.evals_at(points),
            #[cfg(feature = "paper-exact")]
            Self::PaperExact(oracle) => oracle.evals_at(points),
        }
    }

    fn num_rounds(&self) -> usize {
        match self {
            Self::Optimized(oracle) => oracle.num_rounds(),
            #[cfg(feature = "paper-exact")]
            Self::PaperExact(oracle) => oracle.num_rounds(),
        }
    }

    fn degree_bound(&self) -> usize {
        match self {
            Self::Optimized(oracle) => oracle.degree_bound(),
            #[cfg(feature = "paper-exact")]
            Self::PaperExact(oracle) => oracle.degree_bound(),
        }
    }

    fn fold(&mut self, r: K) {
        match self {
            Self::Optimized(oracle) => oracle.fold(r),
            #[cfg(feature = "paper-exact")]
            Self::PaperExact(oracle) => oracle.fold(r),
        }
    }
}

// ============================================================================
// Utilities
// ============================================================================

pub use crate::memory_sidecar::memory::absorb_step_memory;

#[inline]
fn time_column_commit_seed(t: usize) -> [u8; 32] {
    #[inline]
    fn mix64(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    let d = neo_math::D as u64;
    let tt = t as u64;
    let words = [
        mix64(0x6e65_6f2d_666f_6c64 ^ d ^ (tt << 1)),
        mix64(0x7469_6d65_2d63_6f6c ^ (tt << 7)),
        mix64(0x636f_6d6d_6974_2d76 ^ (d << 13) ^ (tt << 5)),
        mix64(0x312f_6465_7465_726d ^ (d << 17) ^ (tt << 19)),
    ];
    let mut seed = [0u8; 32];
    for (i, w) in words.iter().enumerate() {
        seed[i * 8..(i + 1) * 8].copy_from_slice(&w.to_le_bytes());
    }
    seed
}

#[inline]
pub(crate) fn stage8_time_decomp_params(params: &NeoParams) -> Result<NeoParams, PiCcsError> {
    let mut out = *params;
    out.b = crate::time_opening::STAGE8_TIME_DECOMP_BASE;
    out.B = (out.b as u64).checked_pow(out.k_rho).ok_or_else(|| {
        PiCcsError::InvalidInput(format!(
            "stage8/time params: b^k_rho overflow (b={}, k_rho={})",
            out.b, out.k_rho
        ))
    })?;
    let lhs = (out.k_rho as u128 + 1) * (out.T as u128) * ((out.b as u128).saturating_sub(1));
    if lhs >= out.B as u128 {
        return Err(PiCcsError::InvalidInput(format!(
            "stage8/time params: guard inequality fails ((k_rho+1)·T·(b-1)={} >= B={})",
            lhs, out.B
        )));
    }
    Ok(out)
}

pub(crate) fn commit_time_column_sets(
    params: &NeoParams,
    t: usize,
    cpu_cols: &[Vec<F>],
    mem_cols: &[Vec<F>],
    label: &str,
) -> Result<(Vec<Cmt>, Vec<Cmt>), PiCcsError> {
    if t == 0 {
        if cpu_cols.is_empty() && mem_cols.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: t must be > 0 when time columns are present"
        )));
    }

    for (idx, col) in cpu_cols.iter().enumerate() {
        if col.len() != t {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: cpu_cols[{idx}].len()={} != t={}",
                col.len(),
                t
            )));
        }
    }
    for (idx, col) in mem_cols.iter().enumerate() {
        if col.len() != t {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: mem_cols[{idx}].len()={} != t={}",
                col.len(),
                t
            )));
        }
    }

    let want_kappa = params.kappa as usize;
    let expected_seed = time_column_commit_seed(t);
    if has_global_pp_for_dims(D, t) {
        if let Ok((kappa, seed)) = get_global_pp_seeded_params_for_dims(D, t) {
            if kappa != want_kappa {
                return Err(PiCcsError::InvalidInput(format!(
                    "{label}: time-column PP kappa mismatch for (D,t)=({D},{t}) (have {kappa}, want {want_kappa})"
                )));
            }
            if seed != expected_seed {
                return Err(PiCcsError::InvalidInput(format!(
                    "{label}: time-column PP seed mismatch for (D,t)=({D},{t})"
                )));
            }
        } else {
            let pp = get_global_pp_for_dims(D, t).map_err(|e| {
                PiCcsError::InvalidInput(format!(
                    "{label}: failed to load existing time-column PP for (D,t)=({D},{t}): {e}"
                ))
            })?;
            if pp.kappa != want_kappa {
                return Err(PiCcsError::InvalidInput(format!(
                    "{label}: time-column PP kappa mismatch for (D,t)=({D},{t}) (have {}, want {want_kappa})",
                    pp.kappa
                )));
            }
        }
    } else {
        match neo_ajtai::set_global_pp_seeded(D, want_kappa, t, expected_seed) {
            Ok(()) => {}
            Err(e) if has_global_pp_for_dims(D, t) => {
                // Benign race: another thread may have registered it concurrently.
                if let Ok((kappa, seed)) = get_global_pp_seeded_params_for_dims(D, t) {
                    if kappa != want_kappa {
                        return Err(PiCcsError::InvalidInput(format!(
                            "{label}: time-column PP race produced kappa mismatch for (D,t)=({D},{t}) (have {kappa}, want {want_kappa})"
                        )));
                    }
                    if seed != expected_seed {
                        return Err(PiCcsError::InvalidInput(format!(
                            "{label}: time-column PP race produced seed mismatch for (D,t)=({D},{t})"
                        )));
                    }
                } else {
                    return Err(PiCcsError::InvalidInput(format!(
                        "{label}: failed to register/load seeded time-column PP for (D,t)=({D},{t}): {e}"
                    )));
                }
            }
            Err(e) => {
                return Err(PiCcsError::InvalidInput(format!(
                    "{label}: failed to register seeded time-column PP for (D,t)=({D},{t}): {e}"
                )));
            }
        }
    }

    let committer = neo_ajtai::AjtaiSModule::from_global_for_dims(D, t).map_err(|e| {
        PiCcsError::InvalidInput(format!(
            "{label}: time-column committer unavailable for (D,t)=({D},{t}): {e}"
        ))
    })?;

    // Reuse one seeded PP stream across small batches of columns to avoid regenerating
    // the same Ajtai row stream independently for every single column commitment.
    const TIME_COMMIT_BATCH: usize = 512;
    let mut all_cols: Vec<&Vec<F>> = Vec::with_capacity(cpu_cols.len() + mem_cols.len());
    all_cols.extend(cpu_cols.iter());
    all_cols.extend(mem_cols.iter());
    let mut all_comms = Vec::with_capacity(all_cols.len());
    for chunk in all_cols.chunks(TIME_COMMIT_BATCH) {
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let mats: Vec<Mat<F>> = if chunk.len() >= 32 {
            chunk
                .par_iter()
                .map(|col| {
                    neo_memory::ajtai::encode_vector_balanced_to_mat_with_base(
                        params,
                        *col,
                        crate::time_opening::STAGE8_TIME_DECOMP_BASE,
                    )
                })
                .collect()
        } else {
            chunk
                .iter()
                .map(|col| {
                    neo_memory::ajtai::encode_vector_balanced_to_mat_with_base(
                        params,
                        *col,
                        crate::time_opening::STAGE8_TIME_DECOMP_BASE,
                    )
                })
                .collect()
        };
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let mats: Vec<Mat<F>> = chunk
            .iter()
            .map(|col| {
                neo_memory::ajtai::encode_vector_balanced_to_mat_with_base(
                    params,
                    *col,
                    crate::time_opening::STAGE8_TIME_DECOMP_BASE,
                )
            })
            .collect();
        let refs: Vec<&Mat<F>> = mats.iter().collect();
        all_comms.extend(committer.commit_many(&refs));
    }

    let mem_comms = all_comms.split_off(cpu_cols.len());
    let cpu_comms = all_comms;

    Ok((cpu_comms, mem_comms))
}

pub(crate) fn bind_time_column_commitments(
    tr: &mut Poseidon2Transcript,
    step_idx: usize,
    time_t: usize,
    time_declared_len: usize,
    time_col_ids: &[usize],
    cpu_commitments: &[Cmt],
    mem_commitments: &[Cmt],
) {
    tr.append_message(b"time_columns/commit_bind/v2", &[]);
    let header = [
        step_idx as u64,
        time_t as u64,
        time_declared_len as u64,
        time_col_ids.len() as u64,
        cpu_commitments.len() as u64,
        mem_commitments.len() as u64,
        crate::time_opening::STAGE8_TIME_DECOMP_BASE as u64,
    ];
    tr.append_u64s(b"time_columns/commit_bind/header", &header);
    let time_col_ids_u64: Vec<u64> = time_col_ids.iter().map(|&id| id as u64).collect();
    tr.append_u64s(b"time_columns/commit_bind/time_col_ids", &time_col_ids_u64);
    let cpu_lens: Vec<u64> = cpu_commitments
        .iter()
        .map(|c| c.data.len() as u64)
        .collect();
    tr.append_u64s(b"time_columns/commit_bind/cpu_c_data_lens", &cpu_lens);
    let cpu_data_len: usize = cpu_commitments.iter().map(|c| c.data.len()).sum();
    tr.append_fields_iter(
        b"time_columns/commit_bind/cpu_c_data_flat",
        cpu_data_len,
        cpu_commitments.iter().flat_map(|c| c.data.iter().copied()),
    );

    let mem_lens: Vec<u64> = mem_commitments
        .iter()
        .map(|c| c.data.len() as u64)
        .collect();
    tr.append_u64s(b"time_columns/commit_bind/mem_c_data_lens", &mem_lens);
    let mem_data_len: usize = mem_commitments.iter().map(|c| c.data.len()).sum();
    tr.append_fields_iter(
        b"time_columns/commit_bind/mem_c_data_flat",
        mem_data_len,
        mem_commitments.iter().flat_map(|c| c.data.iter().copied()),
    );
}

pub(crate) fn validate_time_active_mask_and_count(
    active_col: &[F],
    time_t: usize,
    label: &str,
) -> Result<usize, PiCcsError> {
    if time_t == 0 {
        if active_col.is_empty() {
            return Ok(0);
        }
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: time_t=0 requires empty active_col (got len={})",
            active_col.len()
        )));
    }
    if active_col.len() != time_t {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: active_col.len()={} != time_t={time_t}",
            active_col.len()
        )));
    }
    let mut seen_zero = false;
    let mut active_count = 0usize;
    for (j, &a) in active_col.iter().enumerate() {
        if a != F::ZERO && a != F::ONE {
            return Err(PiCcsError::ProtocolError(format!(
                "{label}: active_col[{j}] must be boolean (got {a:?})"
            )));
        }
        if a == F::ONE {
            if seen_zero {
                return Err(PiCcsError::ProtocolError(format!(
                    "{label}: active_col is not monotone tail at row {j}"
                )));
            }
            active_count = active_count
                .checked_add(1)
                .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: active_count overflow")))?;
        } else {
            seen_zero = true;
        }
    }
    if active_count > time_t {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: active_count {} exceeds time_t {}",
            active_count, time_t
        )));
    }
    Ok(active_count)
}

pub(crate) fn bind_time_opening_batches_and_sample_coeffs(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step_idx: usize,
    opening_proofs: &[crate::shard_proof_types::TimeOpeningProof],
) -> Result<Vec<Vec<Mat<F>>>, PiCcsError> {
    #[inline]
    fn f_from_i64(x: i64) -> F {
        if x >= 0 {
            F::from_u64(x as u64)
        } else {
            F::ZERO - F::from_u64((-x) as u64)
        }
    }

    #[inline]
    fn rot_matrix_to_rq(mat: &Mat<F>) -> Result<neo_math::ring::Rq, PiCcsError> {
        if mat.rows() != D || mat.cols() != D {
            return Err(PiCcsError::InvalidInput(format!(
                "time_openings/batch_bind: sampled rho must be {D}x{D} (got {}x{})",
                mat.rows(),
                mat.cols()
            )));
        }
        let mut coeffs = [F::ZERO; D];
        for i in 0..D {
            coeffs[i] = mat[(i, 0)];
        }
        Ok(neo_math::ring::Rq(coeffs))
    }

    #[inline]
    fn rot_from_coeffs(coeffs: &[F; D], neg_phi_coeffs: &[F; D]) -> Mat<F> {
        let mut out = Mat::zero(D, D, F::ZERO);
        let mut col = *coeffs;
        for j in 0..D {
            for r in 0..D {
                out[(r, j)] = col[r];
            }
            let last = col[D - 1];
            let mut next = [F::ZERO; D];
            next[0] = last * neg_phi_coeffs[0];
            for r in 1..D {
                next[r] = col[r - 1] + last * neg_phi_coeffs[r];
            }
            col = next;
        }
        out
    }

    tr.append_message(b"time_openings/batch_bind/v2", &[]);
    tr.append_u64s(b"time_openings/batch_bind/step_idx", &[step_idx as u64]);
    tr.append_u64s(b"time_openings/batch_bind/proof_count", &[opening_proofs.len() as u64]);

    let mut all_coeffs = Vec::with_capacity(opening_proofs.len());
    let ring = ccs::RotRing::goldilocks();
    let mut neg_phi_coeffs = [F::ZERO; D];
    for (i, &c) in ring.phi_coeffs.iter().enumerate() {
        neg_phi_coeffs[i] = f_from_i64(-(c as i64));
    }

    let mut point_field_lens = Vec::with_capacity(opening_proofs.len());
    let mut col_id_lens = Vec::with_capacity(opening_proofs.len());
    let mut eval_field_lens = Vec::with_capacity(opening_proofs.len());
    let mut digit_row_lens = Vec::with_capacity(opening_proofs.len());
    let mut digit_field_lens = Vec::with_capacity(opening_proofs.len());
    let mut total_point_fields = 0usize;
    let mut total_eval_fields = 0usize;
    let mut total_digit_fields = 0usize;
    let mut total_col_ids = 0usize;

    for (proof_idx, pf) in opening_proofs.iter().enumerate() {
        if pf.col_ids.len() != pf.evals.len() || pf.col_ids.len() != pf.digit_evals.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "time_openings/batch_bind: proof[{proof_idx}] malformed (col_ids={}, evals={}, digit_evals={})",
                pf.col_ids.len(),
                pf.evals.len(),
                pf.digit_evals.len()
            )));
        }
        for (digit_idx, row) in pf.digit_evals.iter().enumerate() {
            if row.len() != D {
                return Err(PiCcsError::ProtocolError(format!(
                    "time_openings/batch_bind: proof[{proof_idx}] digit_evals[{digit_idx}] len {} != D={D}",
                    row.len()
                )));
            }
        }

        let point_coeffs_per_elem = pf.point.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
        let point_len = pf
            .point
            .len()
            .checked_mul(point_coeffs_per_elem)
            .ok_or_else(|| PiCcsError::ProtocolError("time_openings/batch_bind point length overflow".into()))?;

        let eval_coeffs_per_elem = pf.evals.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
        let eval_len = pf
            .evals
            .len()
            .checked_mul(eval_coeffs_per_elem)
            .ok_or_else(|| PiCcsError::ProtocolError("time_openings/batch_bind eval length overflow".into()))?;
        let digit_coeffs_per_elem = pf
            .digit_evals
            .first()
            .and_then(|row| row.first())
            .map(|v| v.as_coeffs().len())
            .unwrap_or(0);
        let digit_len = pf
            .digit_evals
            .len()
            .checked_mul(D)
            .and_then(|n| n.checked_mul(digit_coeffs_per_elem))
            .ok_or_else(|| PiCcsError::ProtocolError("time_openings/batch_bind digit eval length overflow".into()))?;

        point_field_lens.push(point_len as u64);
        eval_field_lens.push(eval_len as u64);
        digit_row_lens.push(pf.digit_evals.len() as u64);
        digit_field_lens.push(digit_len as u64);
        col_id_lens.push(pf.col_ids.len() as u64);
        total_point_fields = total_point_fields
            .checked_add(point_len)
            .ok_or_else(|| PiCcsError::ProtocolError("time_openings/batch_bind point total overflow".into()))?;
        total_eval_fields = total_eval_fields
            .checked_add(eval_len)
            .ok_or_else(|| PiCcsError::ProtocolError("time_openings/batch_bind eval total overflow".into()))?;
        total_digit_fields = total_digit_fields
            .checked_add(digit_len)
            .ok_or_else(|| PiCcsError::ProtocolError("time_openings/batch_bind digit total overflow".into()))?;
        total_col_ids = total_col_ids
            .checked_add(pf.col_ids.len())
            .ok_or_else(|| PiCcsError::ProtocolError("time_openings/batch_bind col_id total overflow".into()))?;
    }

    tr.append_u64s(b"time_openings/batch_bind/point_field_lens", &point_field_lens);
    tr.append_fields_iter(
        b"time_openings/batch_bind/points_flat",
        total_point_fields,
        opening_proofs
            .iter()
            .flat_map(|pf| pf.point.iter())
            .flat_map(|v| v.as_coeffs()),
    );

    tr.append_u64s(b"time_openings/batch_bind/col_id_lens", &col_id_lens);
    let col_ids_u64: Vec<u64> = opening_proofs
        .iter()
        .flat_map(|pf| pf.col_ids.iter())
        .map(|&id| id as u64)
        .collect();
    debug_assert_eq!(col_ids_u64.len(), total_col_ids);
    tr.append_u64s(b"time_openings/batch_bind/col_ids_flat", &col_ids_u64);

    tr.append_u64s(b"time_openings/batch_bind/eval_field_lens", &eval_field_lens);
    tr.append_fields_iter(
        b"time_openings/batch_bind/evals_flat",
        total_eval_fields,
        opening_proofs
            .iter()
            .flat_map(|pf| pf.evals.iter())
            .flat_map(|v| v.as_coeffs()),
    );

    tr.append_u64s(b"time_openings/batch_bind/digit_row_lens", &digit_row_lens);
    tr.append_u64s(b"time_openings/batch_bind/digit_field_lens", &digit_field_lens);
    tr.append_fields_iter(
        b"time_openings/batch_bind/eval_digits_flat",
        total_digit_fields,
        opening_proofs
            .iter()
            .flat_map(|pf| pf.digit_evals.iter())
            .flat_map(|row| row.iter())
            .flat_map(|v| v.as_coeffs()),
    );

    tr.append_message(b"time_openings/batch_bind/rho_bases/v2", &[]);
    let bases = ccs::sample_rot_rhos_n(tr, params, &ring, opening_proofs.len())?;
    if bases.len() != opening_proofs.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "time_openings/batch_bind: sampled {} rho bases for {} proofs",
            bases.len(),
            opening_proofs.len()
        )));
    }

    for (pf, base) in opening_proofs.iter().zip(bases.into_iter()) {
        let base_rq = rot_matrix_to_rq(&base)?;
        let mut coeffs = Vec::with_capacity(pf.col_ids.len());
        let mut cur_rq = neo_math::ring::Rq::one();
        for _ in 0..pf.col_ids.len() {
            coeffs.push(rot_from_coeffs(&cur_rq.0, &neg_phi_coeffs));
            cur_rq = cur_rq.mul(&base_rq);
        }
        all_coeffs.push(coeffs);
    }

    Ok(all_coeffs)
}

// ============================================================================
// Optional step-to-step (cross-chunk) linking
// ============================================================================

/// Optional verifier-side linking constraints across adjacent shard steps.
///
/// This is intended for chunked CPU circuits that expose boundary state as part of the public
/// input vector `x` per step, and need the verifier to enforce that the state chains across steps.
#[derive(Clone, Debug)]
pub struct StepLinkingConfig {
    /// Equalities on adjacent steps: require `steps[i].x[prev_idx] == steps[i+1].x[next_idx]`.
    pub prev_next_equalities: Vec<(usize, usize)>,
}

impl StepLinkingConfig {
    pub fn new(prev_next_equalities: Vec<(usize, usize)>) -> Self {
        Self { prev_next_equalities }
    }
}

pub fn check_step_linking(steps: &[StepInstanceBundle<Cmt, F, K>], cfg: &StepLinkingConfig) -> Result<(), PiCcsError> {
    if steps.len() <= 1 || cfg.prev_next_equalities.is_empty() {
        return Ok(());
    }
    for (i, (prev, next)) in steps.iter().zip(steps.iter().skip(1)).enumerate() {
        let prev_x = &prev.mcs_inst.x;
        let next_x = &next.mcs_inst.x;
        for &(prev_idx, next_idx) in &cfg.prev_next_equalities {
            if prev_idx >= prev_x.len() || next_idx >= next_x.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "step linking index out of range at boundary {i}: prev_x.len()={}, next_x.len()={}, pair=({prev_idx},{next_idx})",
                    prev_x.len(),
                    next_x.len(),
                )));
            }
            if prev_x[prev_idx] != next_x[next_idx] {
                return Err(PiCcsError::ProtocolError(format!(
                    "step linking failed at boundary {i}: prev_x[{prev_idx}] != next_x[{next_idx}]",
                )));
            }
        }
    }
    Ok(())
}

/// Commitment mixers so the coordinator stays scheme-agnostic.
/// - `mix_rhos_commits(ρ, cs)` returns Σ ρ_i · c_i  (S-action).
/// - `combine_b_pows(cs, b)` returns Σ \bar b^{i-1} c_i  (DEC check).
#[derive(Clone, Copy)]
pub struct CommitMixers<MR, MB>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    MB: Fn(&[Cmt], u32) -> Cmt,
{
    pub mix_rhos_commits: MR,
    pub combine_b_pows: MB,
}

pub fn normalize_me_claims(
    me_claims: &mut [CeClaim<Cmt, F, K>],
    ell_n: usize,
    ell_d: usize,
    t: usize,
) -> Result<(), PiCcsError> {
    let y_pad = 1usize << ell_d;
    for (i, me) in me_claims.iter_mut().enumerate() {
        if me.r.len() != ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] r.len()={}, expected ell_n={}",
                i,
                me.r.len(),
                ell_n
            )));
        }
        if me.y_ring.len() > t {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] y.len()={}, expected <= t={}",
                i,
                me.y_ring.len(),
                t
            )));
        }
        for (j, row) in me.y_ring.iter_mut().enumerate() {
            if row.len() > y_pad {
                return Err(PiCcsError::InvalidInput(format!(
                    "ME[{}] y[{}].len()={}, expected <= {}",
                    i,
                    j,
                    row.len(),
                    y_pad
                )));
            }
            row.resize(y_pad, K::ZERO);
        }
        me.y_ring.resize_with(t, || vec![K::ZERO; y_pad]);
        if me.ct.len() > t {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] y_scalars.len()={}, expected <= t={}",
                i,
                me.ct.len(),
                t
            )));
        }
        me.ct.resize(t, K::ZERO);
    }
    Ok(())
}

pub(crate) fn validate_me_batch_invariants(batch: &[CeClaim<Cmt, F, K>], context: &str) -> Result<(), PiCcsError> {
    if batch.is_empty() {
        return Ok(());
    }
    let me0 = &batch[0];
    let r0 = &me0.r;
    let m_in0 = me0.m_in;
    let y_len0 = me0.y_ring.len();
    let y_row_len0 = me0.y_ring.first().map(|r| r.len()).unwrap_or(0);
    let y_scalars_len0 = me0.ct.len();

    if me0.X.rows() != D {
        return Err(PiCcsError::ProtocolError(format!(
            "{}: ME claim 0 has X.rows()={}, expected D={}",
            context,
            me0.X.rows(),
            D
        )));
    }
    if me0.X.cols() != m_in0 {
        return Err(PiCcsError::ProtocolError(format!(
            "{}: ME claim 0 has X.cols()={}, expected m_in={}",
            context,
            me0.X.cols(),
            m_in0
        )));
    }

    for (i, me) in batch.iter().enumerate().skip(1) {
        if me.r != *r0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has different r than claim 0 (r-alignment required for RLC)",
                context, i
            )));
        }
        if me.m_in != m_in0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has m_in={}, expected {}",
                context, i, me.m_in, m_in0
            )));
        }
        if me.X.rows() != D || me.X.cols() != m_in0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has X shape {}x{}, expected {}x{}",
                context,
                i,
                me.X.rows(),
                me.X.cols(),
                D,
                m_in0
            )));
        }
        if me.y_ring.len() != y_len0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has y.len()={}, expected {}",
                context,
                i,
                me.y_ring.len(),
                y_len0
            )));
        }
        for (j, row) in me.y_ring.iter().enumerate() {
            if row.len() != y_row_len0 {
                return Err(PiCcsError::ProtocolError(format!(
                    "{}: ME claim {} has y[{}].len()={}, expected {}",
                    context,
                    i,
                    j,
                    row.len(),
                    y_row_len0
                )));
            }
        }
        if me.ct.len() != y_scalars_len0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has y_scalars.len()={}, expected {}",
                context,
                i,
                me.ct.len(),
                y_scalars_len0
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RlcLane {
    Main,
    Val,
}

#[inline]
pub(crate) fn balanced_divrem_i64(v: i64, b: i64) -> (i64, i64) {
    debug_assert!(b >= 2);
    let mut r = v % b;
    let mut q = (v - r) / b;
    let half = b / 2;
    if r > half {
        r -= b;
        q += 1;
    } else if r < -half {
        r += b;
        q -= 1;
    }
    (r, q)
}

#[inline]
pub(crate) fn balanced_divrem_i128(v: i128, b: i128) -> (i128, i128) {
    debug_assert!(b >= 2);
    let mut r = v % b;
    let mut q = (v - r) / b;
    let half = b / 2;
    if r > half {
        r -= b;
        q += 1;
    } else if r < -half {
        r += b;
        q -= 1;
    }
    (r, q)
}

#[inline]
pub(crate) fn f_from_i64(x: i64) -> F {
    if x >= 0 {
        F::from_u64(x as u64)
    } else {
        F::ZERO - F::from_u64((-x) as u64)
    }
}

#[inline]
fn balanced_abs_u128(v: F) -> u128 {
    let p = F::ORDER_U64 as u128;
    let u = v.as_canonical_u64() as u128;
    core::cmp::min(u, p.saturating_sub(u))
}

#[inline]
fn min_balanced_digits_for_abs(abs: u128, b: u32) -> Result<usize, PiCcsError> {
    if b < 2 {
        return Err(PiCcsError::InvalidInput(format!("invalid base b={b}")));
    }
    if abs == 0 {
        return Ok(1);
    }
    let base = b as u128;
    let half = (b / 2) as u128;
    if half == 0 {
        return Err(PiCcsError::InvalidInput(format!(
            "invalid balanced digit range for b={b}"
        )));
    }
    let mut k = 0usize;
    let mut place = 1u128;
    let mut geom_sum = 0u128; // 1 + b + ... + b^{k-1}
    loop {
        k = k
            .checked_add(1)
            .ok_or_else(|| PiCcsError::InvalidInput("k_dec overflow".into()))?;
        geom_sum = geom_sum
            .checked_add(place)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("balanced range overflow for b={b}, k={k}")))?;
        let max_abs = half
            .checked_mul(geom_sum)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("balanced range overflow for b={b}, k={k}")))?;
        if abs <= max_abs {
            return Ok(k);
        }
        place = place
            .checked_mul(base)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("b^k overflow for b={b}, k={k}")))?;
    }
}

/// Lower bound on DEC digit count needed so every entry of `Z` fits in balanced base-`b`.
pub(crate) fn required_dec_digits_for_matrix(params: &NeoParams, z: &Mat<F>) -> Result<usize, PiCcsError> {
    let mut need = 1usize;
    for &v in z.as_slice() {
        let k = min_balanced_digits_for_abs(balanced_abs_u128(v), params.b)?;
        need = core::cmp::max(need, k);
    }
    Ok(need)
}

#[inline]
pub(crate) fn verify_me_y_scalars_canonical(
    me: &CeClaim<Cmt, F, K>,
    _b: u32,
    ccs_m: usize,
    step_idx: usize,
    context: &str,
) -> Result<(), PiCcsError> {
    if me.ct.len() != me.y_ring.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: {}: y_scalars.len()={} must equal y.len()={}",
            step_idx,
            context,
            me.ct.len(),
            me.y_ring.len()
        )));
    }
    for (j, row) in me.y_ring.iter().enumerate() {
        if row.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: {}: y[{}].len()={} must be >= 1",
                step_idx,
                context,
                j,
                row.len(),
            )));
        }
        if ccs_m == 0 {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: {}: invalid ccs_m=0",
                step_idx, context
            )));
        }
        let expect = neo_reductions::common::ct_from_y_digits(row);
        if me.ct[j] != expect {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: {}: ct[{}] does not match layout-aware CE scalar semantics",
                step_idx, context, j
            )));
        }
    }
    Ok(())
}

pub(crate) fn dec_stream_no_witness<MB>(
    params: &NeoParams,
    s: &CcsStructure<F>,
    parent: &CeClaim<Cmt, F, K>,
    Z_mix: &Mat<F>,
    ell_d: usize,
    k_dec: usize,
    combine_b_pows: MB,
    sparse: Option<&SparseCache<F>>,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, Vec<Cmt>, bool, bool, bool), PiCcsError>
where
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if k_dec == 0 {
        return Err(PiCcsError::InvalidInput("DEC: k_dec must be > 0".into()));
    }
    if Z_mix.rows() != D {
        return Err(PiCcsError::InvalidInput(format!(
            "DEC: Z_mix must have {} rows (got {})",
            D,
            Z_mix.rows()
        )));
    }
    let z_layout = neo_reductions::common::witness_mat_layout(Z_mix, s.m).map_err(|e| {
        PiCcsError::InvalidInput(format!(
            "DEC: Z_mix shape is incompatible with logical CCS width m={} ({})",
            s.m, e
        ))
    })?;
    let m_commit = Z_mix.cols();

    let d_pad = 1usize << ell_d;
    let want_nc_channel = !(parent.s_col.is_empty() && parent.y_zcol.is_empty());
    if want_nc_channel && (parent.s_col.is_empty() || parent.y_zcol.is_empty()) {
        return Err(PiCcsError::InvalidInput(
            "DEC: incomplete NC channel on parent (expected both s_col and y_zcol)".into(),
        ));
    }
    if want_nc_channel && parent.y_zcol.len() != d_pad {
        return Err(PiCcsError::InvalidInput(format!(
            "DEC: parent y_zcol length mismatch (expected {}, got {})",
            d_pad,
            parent.y_zcol.len()
        )));
    }

    enum PpAccess {
        Seeded {
            kappa: usize,
            chunk_size: usize,
            chunk_seeds_by_row: Vec<Vec<[u8; 32]>>,
        },
        Loaded {
            pp: Arc<neo_ajtai::PP<neo_math::ring::Rq>>,
        },
    }

    let pp_access = if let Some(pp) = try_get_loaded_global_pp_for_dims(D, m_commit) {
        if pp.kappa == 0 {
            return Err(PiCcsError::InvalidInput("DEC: PP.kappa must be > 0".into()));
        }
        PpAccess::Loaded { pp }
    } else if let Ok((kappa, seed)) = get_global_pp_seeded_params_for_dims(D, m_commit) {
        if kappa == 0 {
            return Err(PiCcsError::InvalidInput("DEC: PP.kappa must be > 0".into()));
        }
        let (chunk_size, chunk_seeds_by_row) = seeded_pp_chunk_seeds(seed, kappa, m_commit);
        PpAccess::Seeded {
            kappa,
            chunk_size,
            chunk_seeds_by_row,
        }
    } else {
        // Fallback: non-seeded entry. This will materialize PP if needed.
        let pp = get_global_pp_for_dims(D, m_commit).map_err(|e| {
            PiCcsError::InvalidInput(format!(
                "DEC: Ajtai PP unavailable for (d,m_commit)=({},{}) ({})",
                D, m_commit, e
            ))
        })?;
        if pp.kappa == 0 {
            return Err(PiCcsError::InvalidInput("DEC: PP.kappa must be > 0".into()));
        }
        PpAccess::Loaded { pp }
    };

    // Build χ_r and v_j = M_j^T · χ_r (same as the reference DEC).
    let ell_n = parent.r.len();
    let n_sz = 1usize
        .checked_shl(ell_n as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("DEC: 2^ell_n overflow".into()))?;
    let n_eff = core::cmp::min(s.n, n_sz);

    // χ_r table over the row/time hypercube.
    //
    // IMPORTANT: Use the same bit order as `eq_points_bool_mask` / `chi_tail_weights`
    // (bit 0 = LSB) so CSC column traversals match the reference DEC.
    #[inline]
    fn chi_tail_weights(bits: &[K]) -> Vec<K> {
        let t = bits.len();
        let len = 1usize << t;
        let mut w = vec![K::ZERO; len];
        w[0] = K::ONE;
        for (i, &b) in bits.iter().enumerate() {
            let step = 1usize << i;
            let one_minus = K::ONE - b;
            for mask in 0..step {
                let v = w[mask];
                w[mask] = v * one_minus;
                w[mask + step] = v * b;
            }
        }
        w
    }

    let chi_r = chi_tail_weights(&parent.r);
    debug_assert_eq!(chi_r.len(), n_sz);

    let chi_s = if want_nc_channel {
        let chi = chi_tail_weights(&parent.s_col);
        if chi.len() < s.m {
            return Err(PiCcsError::InvalidInput(format!(
                "DEC: chi(s_col) too short for CCS width (need >= {}, got {})",
                s.m,
                chi.len()
            )));
        }
        chi
    } else {
        Vec::new()
    };

    let t_mats = s.t();

    enum VjsAccess<'a> {
        Dense(Vec<Vec<K>>),
        Sparse {
            cap: usize,
            cache: &'a SparseCache<F>,
        },
    }

    let vjs_access = if let Some(cache) = sparse {
        if cache.len() != t_mats {
            return Err(PiCcsError::InvalidInput(format!(
                "DEC: sparse cache matrix count mismatch: got {}, expected {}",
                cache.len(),
                t_mats
            )));
        }
        let cap = core::cmp::min(s.m, n_eff);
        VjsAccess::Sparse { cap, cache }
    } else {
        let mut vjs: Vec<Vec<K>> = vec![vec![K::ZERO; s.m]; t_mats];
        for j in 0..t_mats {
            s.matrices[j].add_mul_transpose_into(&chi_r, &mut vjs[j], n_eff);
        }
        VjsAccess::Dense(vjs)
    };

    // Base in K for decomposition checks.
    let bF = F::from_u64(params.b as u64);
    let bK = K::from(bF);

    // Precompute parameters for bounded signed decoding of Z_mix entries.
    let b_u = params.b as u128;
    let mut B_u: u128 = 1;
    for _ in 0..k_dec {
        B_u = B_u.saturating_mul(b_u);
    }
    let p: u128 = F::ORDER_U64 as u128;

    // Fast row-major access.
    let z_rows: Vec<&[F]> = (0..D).map(|r| Z_mix.row(r)).collect();

    struct Acc {
        commit: Vec<[F; D]>, // [digit][kappa] -> [D]
        y_ring: Vec<[K; D]>, // [digit][t] -> [D]
        y_zcol: Vec<[K; D]>, // [digit] -> [D]
        any_nonzero: Vec<bool>,
        vj: Vec<K>,          // scratch: t
        digits: Vec<i32>,    // scratch: k*D (balanced digits)
        rot_next: [F; D],    // scratch: rotation step output (written fully each time)
        err: Option<String>, // first error wins
    }

    impl Acc {
        fn new(k_dec: usize, kappa: usize, t: usize) -> Self {
            Self {
                commit: vec![[F::ZERO; D]; k_dec * kappa],
                y_ring: vec![[K::ZERO; D]; k_dec * t],
                y_zcol: vec![[K::ZERO; D]; k_dec],
                any_nonzero: vec![false; k_dec],
                vj: vec![K::ZERO; t],
                digits: vec![0i32; k_dec * D],
                rot_next: [F::ZERO; D],
                err: None,
            }
        }

        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        fn add_inplace(&mut self, rhs: &Acc, k_dec: usize, kappa: usize, t: usize) {
            for (dst, src) in self.commit.iter_mut().zip(rhs.commit.iter()) {
                for r in 0..D {
                    dst[r] += src[r];
                }
            }
            for (dst, src) in self.y_ring.iter_mut().zip(rhs.y_ring.iter()) {
                for r in 0..D {
                    dst[r] += src[r];
                }
            }
            for (dst, src) in self.y_zcol.iter_mut().zip(rhs.y_zcol.iter()) {
                for r in 0..D {
                    dst[r] += src[r];
                }
            }
            for i in 0..k_dec {
                self.any_nonzero[i] |= rhs.any_nonzero[i];
            }
            if self.err.is_none() {
                self.err = rhs.err.clone();
            }
            // silence unused warnings when parameters are const-propagated
            let _ = (k_dec, kappa, t);
        }
    }

    let m_phys = m_commit;
    let b_i64 = params.b as i64;
    let b_i128 = params.b as i128;

    #[inline]
    fn logical_col_for_entry(
        layout: neo_reductions::common::WitnessMatLayout,
        col_phys: usize,
        rho: usize,
        logical_m: usize,
    ) -> Option<usize> {
        match layout {
            neo_reductions::common::WitnessMatLayout::SuperneoPacked => {
                let col = col_phys
                    .checked_mul(D)
                    .and_then(|v| v.checked_add(rho))
                    .unwrap_or(usize::MAX);
                if col < logical_m {
                    Some(col)
                } else {
                    None
                }
            }
            neo_reductions::common::WitnessMatLayout::DenseUnpacked => {
                if col_phys < logical_m {
                    Some(col_phys)
                } else {
                    None
                }
            }
        }
    }

    // Specialized rot_step for Φ₈₁(X) = X^54 + X^27 + 1 (η=81, D=54).
    // Mirrors `neo_ajtai::commit::rot_step_phi_81` but kept local to avoid pulling a large
    // D×D scratch table (`precompute_rot_columns`) into the hot DEC streaming loop.
    #[inline]
    fn rot_step_phi_81(cur: &[F; D], next: &mut [F; D]) {
        let last = cur[D - 1];
        next[0] = F::ZERO;
        next[1..D].copy_from_slice(&cur[..(D - 1)]);
        next[0] -= last;
        next[27] -= last;
    }

    #[inline]
    fn acc_add_assign(acc: &mut [F; D], col: &[F; D]) {
        type P = <F as Field>::Packing;
        let prefix_len = D - (D % P::WIDTH);
        let (acc_prefix, acc_suffix) = acc.split_at_mut(prefix_len);
        let (col_prefix, col_suffix) = col.split_at(prefix_len);

        for (a, b) in P::pack_slice_mut(acc_prefix)
            .iter_mut()
            .zip(P::pack_slice(col_prefix).iter())
        {
            *a += *b;
        }
        for (a, &b) in acc_suffix.iter_mut().zip(col_suffix.iter()) {
            *a += b;
        }
    }

    #[inline]
    fn acc_sub_assign(acc: &mut [F; D], col: &[F; D]) {
        type P = <F as Field>::Packing;
        let prefix_len = D - (D % P::WIDTH);
        let (acc_prefix, acc_suffix) = acc.split_at_mut(prefix_len);
        let (col_prefix, col_suffix) = col.split_at(prefix_len);

        for (a, b) in P::pack_slice_mut(acc_prefix)
            .iter_mut()
            .zip(P::pack_slice(col_prefix).iter())
        {
            *a -= *b;
        }
        for (a, &b) in acc_suffix.iter_mut().zip(col_suffix.iter()) {
            *a -= b;
        }
    }

    #[inline]
    fn acc_mul_add_assign(acc: &mut [F; D], col: &[F; D], scalar: F) {
        type P = <F as Field>::Packing;
        let prefix_len = D - (D % P::WIDTH);
        let (acc_prefix, acc_suffix) = acc.split_at_mut(prefix_len);
        let (col_prefix, col_suffix) = col.split_at(prefix_len);
        let scalar_p: P = scalar.into();

        for (a, b) in P::pack_slice_mut(acc_prefix)
            .iter_mut()
            .zip(P::pack_slice(col_prefix).iter())
        {
            *a += *b * scalar_p;
        }
        for (a, &b) in acc_suffix.iter_mut().zip(col_suffix.iter()) {
            *a += b * scalar;
        }
    }

    let (kappa, acc) = match &pp_access {
        PpAccess::Loaded { pp } => {
            let kappa = pp.kappa;
            let process_col = |mut st: Acc, col: usize| -> Acc {
                if st.err.is_some() {
                    return st;
                }

                // Decompose the column's D entries into balanced base-b digits for each DEC child.
                for rho in 0..D {
                    let u = z_rows[rho][col].as_canonical_u64() as u128;
                    if B_u <= i64::MAX as u128 {
                        let val_opt: Option<i64> = if u < B_u {
                            Some(u as i64)
                        } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                            Some(-((p - u) as i64))
                        } else {
                            None
                        };
                        let mut v = match val_opt {
                            Some(v) => v,
                            None => {
                                st.err = Some(format!(
                                    "DEC split: Z_mix[{},{}] is out of range for k_rho={}, b={}",
                                    rho, col, k_dec, params.b
                                ));
                                return st;
                            }
                        };
                        for i in 0..k_dec {
                            if v == 0 {
                                st.digits[i * D + rho] = 0;
                                continue;
                            }
                            let (r_i, q) = balanced_divrem_i64(v, b_i64);
                            if r_i != 0 {
                                st.any_nonzero[i] = true;
                            }
                            st.digits[i * D + rho] = r_i as i32;
                            v = q;
                        }
                        if v != 0 {
                            st.err = Some(format!(
                                "DEC split: Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                                rho, col, k_dec, params.b
                            ));
                            return st;
                        }
                    } else {
                        let val_opt: Option<i128> = if u < B_u {
                            Some(u as i128)
                        } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                            Some(-((p - u) as i128))
                        } else {
                            None
                        };
                        let mut v = match val_opt {
                            Some(v) => v,
                            None => {
                                st.err = Some(format!(
                                    "DEC split: Z_mix[{},{}] is out of range for k_rho={}, b={}",
                                    rho, col, k_dec, params.b
                                ));
                                return st;
                            }
                        };
                        for i in 0..k_dec {
                            if v == 0 {
                                st.digits[i * D + rho] = 0;
                                continue;
                            }
                            let (r_i, q) = balanced_divrem_i128(v, b_i128);
                            if r_i != 0 {
                                st.any_nonzero[i] = true;
                            }
                            st.digits[i * D + rho] = r_i as i32;
                            v = q;
                        }
                        if v != 0 {
                            st.err = Some(format!(
                                "DEC split: Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                                rho, col, k_dec, params.b
                            ));
                            return st;
                        }
                    }
                }

                // y_(i,j)[rho] += Z_i[rho,col_phys] * (M_j^T · χ_r)[col_logical]
                // and optional y_zcol_i[rho] += Z_i[rho,col_phys] * χ_{s_col}[col_logical].
                for rho in 0..D {
                    let logical_col = logical_col_for_entry(z_layout, col, rho, s.m);
                    if let Some(logical_col) = logical_col {
                        match &vjs_access {
                            VjsAccess::Dense(vjs) => {
                                for j in 0..t_mats {
                                    st.vj[j] = vjs[j][logical_col];
                                }
                            }
                            VjsAccess::Sparse { cap, cache } => {
                                for j in 0..t_mats {
                                    st.vj[j] = if let Some(csc) = cache.csc(j) {
                                        let mut sum = K::ZERO;
                                        let s_ptr = csc.col_ptr[logical_col];
                                        let e_ptr = csc.col_ptr[logical_col + 1];
                                        for k in s_ptr..e_ptr {
                                            let r = csc.row_idx[k];
                                            if r < n_eff {
                                                sum += K::from(csc.vals[k]) * chi_r[r];
                                            }
                                        }
                                        sum
                                    } else if logical_col < *cap {
                                        chi_r[logical_col]
                                    } else {
                                        K::ZERO
                                    };
                                }
                            }
                        }
                    } else {
                        for j in 0..t_mats {
                            st.vj[j] = K::ZERO;
                        }
                    }

                    for i in 0..k_dec {
                        let digit = st.digits[i * D + rho];
                        if digit == 0 {
                            continue;
                        }
                        let y_base = i * t_mats;
                        for j in 0..t_mats {
                            let vj = st.vj[j];
                            if vj != K::ZERO {
                                match digit {
                                    1 => st.y_ring[y_base + j][rho] += vj,
                                    -1 => st.y_ring[y_base + j][rho] -= vj,
                                    _ => st.y_ring[y_base + j][rho] += vj.scale_base(f_from_i64(digit as i64)),
                                }
                            }
                        }
                    }

                    if let Some(logical_col) = logical_col {
                        if !chi_s.is_empty() {
                            let w_col = chi_s[logical_col];
                            if w_col != K::ZERO {
                                for i in 0..k_dec {
                                    let digit = st.digits[i * D + rho];
                                    if digit == 0 {
                                        continue;
                                    }
                                    match digit {
                                        1 => st.y_zcol[i][rho] += w_col,
                                        -1 => st.y_zcol[i][rho] -= w_col,
                                        _ => st.y_zcol[i][rho] += w_col.scale_base(f_from_i64(digit as i64)),
                                    }
                                }
                            }
                        }
                    }
                }

                // Commitment accumulators per digit.
                for kr in 0..kappa {
                    let mut rot_col = neo_math::ring::cf(pp.m_rows[kr][col]);
                    for rho in 0..D {
                        for i in 0..k_dec {
                            let digit = st.digits[i * D + rho];
                            if digit == 0 {
                                continue;
                            }
                            let acc = &mut st.commit[i * kappa + kr];
                            match digit {
                                1 => acc_add_assign(acc, &rot_col),
                                -1 => acc_sub_assign(acc, &rot_col),
                                _ => acc_mul_add_assign(acc, &rot_col, f_from_i64(digit as i64)),
                            }
                        }
                        rot_step_phi_81(&rot_col, &mut st.rot_next);
                        core::mem::swap(&mut rot_col, &mut st.rot_next);
                    }
                }

                st
            };

            let acc = {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    (0..m_phys)
                        .into_par_iter()
                        .fold(|| Acc::new(k_dec, kappa, t_mats), |st, col| process_col(st, col))
                        .reduce(
                            || Acc::new(k_dec, kappa, t_mats),
                            |mut a, b| {
                                if a.err.is_none() {
                                    a.add_inplace(&b, k_dec, kappa, t_mats);
                                }
                                a
                            },
                        )
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    let mut st = Acc::new(k_dec, kappa, t_mats);
                    for col in 0..m_phys {
                        st = process_col(st, col);
                    }
                    st
                }
            };
            (kappa, acc)
        }
        PpAccess::Seeded {
            kappa,
            chunk_size,
            chunk_seeds_by_row,
        } => {
            let kappa = *kappa;
            let chunk_size = *chunk_size;
            let num_chunks = (m_phys + chunk_size - 1) / chunk_size;

            let process_chunk = |mut st: Acc, chunk_idx: usize| -> Acc {
                if st.err.is_some() {
                    return st;
                }

                let start = chunk_idx * chunk_size;
                let end = core::cmp::min(m_phys, start + chunk_size);

                let mut rngs: Vec<ChaCha8Rng> = (0..kappa)
                    .map(|kr| ChaCha8Rng::from_seed(chunk_seeds_by_row[kr][chunk_idx]))
                    .collect();

                for col in start..end {
                    // Decompose the column's D entries into balanced base-b digits for each DEC child.
                    for rho in 0..D {
                        let u = z_rows[rho][col].as_canonical_u64() as u128;
                        if B_u <= i64::MAX as u128 {
                            let val_opt: Option<i64> = if u < B_u {
                                Some(u as i64)
                            } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                                Some(-((p - u) as i64))
                            } else {
                                None
                            };
                            let mut v = match val_opt {
                                Some(v) => v,
                                None => {
                                    st.err = Some(format!(
                                        "DEC split: Z_mix[{},{}] is out of range for k_rho={}, b={}",
                                        rho, col, k_dec, params.b
                                    ));
                                    return st;
                                }
                            };
                            for i in 0..k_dec {
                                if v == 0 {
                                    st.digits[i * D + rho] = 0;
                                    continue;
                                }
                                let (r_i, q) = balanced_divrem_i64(v, b_i64);
                                if r_i != 0 {
                                    st.any_nonzero[i] = true;
                                }
                                st.digits[i * D + rho] = r_i as i32;
                                v = q;
                            }
                            if v != 0 {
                                st.err = Some(format!(
                                    "DEC split: Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                                    rho, col, k_dec, params.b
                                ));
                                return st;
                            }
                        } else {
                            let val_opt: Option<i128> = if u < B_u {
                                Some(u as i128)
                            } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                                Some(-((p - u) as i128))
                            } else {
                                None
                            };
                            let mut v = match val_opt {
                                Some(v) => v,
                                None => {
                                    st.err = Some(format!(
                                        "DEC split: Z_mix[{},{}] is out of range for k_rho={}, b={}",
                                        rho, col, k_dec, params.b
                                    ));
                                    return st;
                                }
                            };
                            for i in 0..k_dec {
                                if v == 0 {
                                    st.digits[i * D + rho] = 0;
                                    continue;
                                }
                                let (r_i, q) = balanced_divrem_i128(v, b_i128);
                                if r_i != 0 {
                                    st.any_nonzero[i] = true;
                                }
                                st.digits[i * D + rho] = r_i as i32;
                                v = q;
                            }
                            if v != 0 {
                                st.err = Some(format!(
                                    "DEC split: Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                                    rho, col, k_dec, params.b
                                ));
                                return st;
                            }
                        }
                    }

                    // y_(i,j)[rho] += Z_i[rho,col_phys] * (M_j^T · χ_r)[col_logical]
                    // and optional y_zcol_i[rho] += Z_i[rho,col_phys] * χ_{s_col}[col_logical].
                    for rho in 0..D {
                        let logical_col = logical_col_for_entry(z_layout, col, rho, s.m);
                        if let Some(logical_col) = logical_col {
                            match &vjs_access {
                                VjsAccess::Dense(vjs) => {
                                    for j in 0..t_mats {
                                        st.vj[j] = vjs[j][logical_col];
                                    }
                                }
                                VjsAccess::Sparse { cap, cache } => {
                                    for j in 0..t_mats {
                                        st.vj[j] = if let Some(csc) = cache.csc(j) {
                                            let mut sum = K::ZERO;
                                            let s_ptr = csc.col_ptr[logical_col];
                                            let e_ptr = csc.col_ptr[logical_col + 1];
                                            for k in s_ptr..e_ptr {
                                                let r = csc.row_idx[k];
                                                if r < n_eff {
                                                    sum += K::from(csc.vals[k]) * chi_r[r];
                                                }
                                            }
                                            sum
                                        } else if logical_col < *cap {
                                            chi_r[logical_col]
                                        } else {
                                            K::ZERO
                                        };
                                    }
                                }
                            }
                        } else {
                            for j in 0..t_mats {
                                st.vj[j] = K::ZERO;
                            }
                        }

                        for i in 0..k_dec {
                            let digit = st.digits[i * D + rho];
                            if digit == 0 {
                                continue;
                            }
                            let y_base = i * t_mats;
                            for j in 0..t_mats {
                                let vj = st.vj[j];
                                if vj != K::ZERO {
                                    match digit {
                                        1 => st.y_ring[y_base + j][rho] += vj,
                                        -1 => st.y_ring[y_base + j][rho] -= vj,
                                        _ => st.y_ring[y_base + j][rho] += vj.scale_base(f_from_i64(digit as i64)),
                                    }
                                }
                            }
                        }

                        if let Some(logical_col) = logical_col {
                            if !chi_s.is_empty() {
                                let w_col = chi_s[logical_col];
                                if w_col != K::ZERO {
                                    for i in 0..k_dec {
                                        let digit = st.digits[i * D + rho];
                                        if digit == 0 {
                                            continue;
                                        }
                                        match digit {
                                            1 => st.y_zcol[i][rho] += w_col,
                                            -1 => st.y_zcol[i][rho] -= w_col,
                                            _ => st.y_zcol[i][rho] += w_col.scale_base(f_from_i64(digit as i64)),
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Commitment accumulators per digit.
                    for kr in 0..kappa {
                        let a_kr_col = sample_uniform_rq(&mut rngs[kr]);
                        let mut rot_col = neo_math::ring::cf(a_kr_col);
                        for rho in 0..D {
                            for i in 0..k_dec {
                                let digit = st.digits[i * D + rho];
                                if digit == 0 {
                                    continue;
                                }
                                let acc = &mut st.commit[i * kappa + kr];
                                match digit {
                                    1 => acc_add_assign(acc, &rot_col),
                                    -1 => acc_sub_assign(acc, &rot_col),
                                    _ => acc_mul_add_assign(acc, &rot_col, f_from_i64(digit as i64)),
                                }
                            }
                            rot_step_phi_81(&rot_col, &mut st.rot_next);
                            core::mem::swap(&mut rot_col, &mut st.rot_next);
                        }
                    }
                }

                st
            };

            let acc = {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    (0..num_chunks)
                        .into_par_iter()
                        .fold(
                            || Acc::new(k_dec, kappa, t_mats),
                            |st, chunk_idx| process_chunk(st, chunk_idx),
                        )
                        .reduce(
                            || Acc::new(k_dec, kappa, t_mats),
                            |mut a, b| {
                                if a.err.is_none() {
                                    a.add_inplace(&b, k_dec, kappa, t_mats);
                                }
                                a
                            },
                        )
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    let mut st = Acc::new(k_dec, kappa, t_mats);
                    for chunk_idx in 0..num_chunks {
                        st = process_chunk(st, chunk_idx);
                    }
                    st
                }
            };
            (kappa, acc)
        }
    };

    if let Some(err) = acc.err {
        return Err(PiCcsError::ProtocolError(err));
    }

    // Commitments c_i from accumulated columns.
    let mut child_cs: Vec<Cmt> = Vec::with_capacity(k_dec);
    for i in 0..k_dec {
        if !acc.any_nonzero[i] {
            child_cs.push(Cmt::zeros(D, kappa));
            continue;
        }
        let mut c = Cmt::zeros(D, kappa);
        for kr in 0..kappa {
            c.col_mut(kr).copy_from_slice(&acc.commit[i * kappa + kr]);
        }
        child_cs.push(c);
    }

    // X_i: project first m_in columns from Z_i (small; compute sequentially).
    let m_in = parent.m_in;
    let mut xs_row_major: Vec<Vec<F>> = vec![vec![F::ZERO; D * m_in]; k_dec];
    for col in 0..m_in {
        for rho in 0..D {
            let z_rc = neo_reductions::common::witness_mat_get_f(Z_mix, z_layout, s.m, rho, col);
            let u = z_rc.as_canonical_u64() as u128;
            if B_u <= i64::MAX as u128 {
                let val_opt: Option<i64> = if u < B_u {
                    Some(u as i64)
                } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                    Some(-((p - u) as i64))
                } else {
                    None
                };
                let mut v = val_opt.ok_or_else(|| {
                    PiCcsError::ProtocolError(format!(
                        "DEC split(X): Z_mix[{},{}] out of range for k_rho={}, b={}",
                        rho, col, k_dec, params.b
                    ))
                })?;
                for i in 0..k_dec {
                    if v == 0 {
                        break;
                    }
                    let (r_i, q) = balanced_divrem_i64(v, b_i64);
                    xs_row_major[i][rho * m_in + col] = f_from_i64(r_i);
                    v = q;
                }
                if v != 0 {
                    return Err(PiCcsError::ProtocolError(format!(
                        "DEC split(X): Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                        rho, col, k_dec, params.b
                    )));
                }
            } else {
                let val_opt: Option<i128> = if u < B_u {
                    Some(u as i128)
                } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                    Some(-((p - u) as i128))
                } else {
                    None
                };
                let mut v = val_opt.ok_or_else(|| {
                    PiCcsError::ProtocolError(format!(
                        "DEC split(X): Z_mix[{},{}] out of range for k_rho={}, b={}",
                        rho, col, k_dec, params.b
                    ))
                })?;
                for i in 0..k_dec {
                    if v == 0 {
                        break;
                    }
                    let (r_i, q) = balanced_divrem_i128(v, b_i128);
                    xs_row_major[i][rho * m_in + col] = f_from_i64(r_i as i64);
                    v = q;
                }
                if v != 0 {
                    return Err(PiCcsError::ProtocolError(format!(
                        "DEC split(X): Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                        rho, col, k_dec, params.b
                    )));
                }
            }
        }
    }

    let parent_r = parent.r.clone();
    let fold_digest = parent.fold_digest;

    let mut children: Vec<CeClaim<Cmt, F, K>> = Vec::with_capacity(k_dec);
    for i in 0..k_dec {
        let Xi = Mat::from_row_major(D, m_in, xs_row_major[i].clone());
        let mut y_i: Vec<Vec<K>> = Vec::with_capacity(t_mats);
        for j in 0..t_mats {
            let mut yj = vec![K::ZERO; d_pad];
            let row = &acc.y_ring[i * t_mats + j];
            for rho in 0..D {
                yj[rho] = row[rho];
            }
            y_i.push(yj);
        }
        let y_scalars_i = neo_reductions::common::ct_from_y_ring_for_ccs_m(&y_i, params, s.m);

        let y_zcol = if chi_s.is_empty() {
            Vec::new()
        } else {
            let mut yz = vec![K::ZERO; d_pad];
            let row = &acc.y_zcol[i];
            for rho in 0..D {
                yz[rho] = row[rho];
            }
            yz
        };

        children.push(CeClaim::<Cmt, F, K> {
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c: child_cs[i].clone(),
            X: Xi,
            r: parent_r.clone(),
            s_col: parent.s_col.clone(),
            y_ring: y_i,
            ct: y_scalars_i,
            aux_openings: Vec::new(),
            y_zcol,
            m_in,
            fold_digest,
        });
    }

    // Public checks (mirror paper-exact DEC).
    let mut ok_y = true;
    for j in 0..t_mats {
        let mut lhs = vec![K::ZERO; d_pad];
        let mut pow = K::ONE;
        for i in 0..k_dec {
            for t in 0..d_pad {
                lhs[t] += pow * children[i].y_ring[j][t];
            }
            pow *= bK;
        }
        if lhs != parent.y_ring[j] {
            ok_y = false;
            break;
        }
    }

    // y_zcol: column-domain opening must also decompose (when present).
    if ok_y && !chi_s.is_empty() {
        let mut lhs = vec![K::ZERO; d_pad];
        let mut pow = K::ONE;
        for i in 0..k_dec {
            for t in 0..d_pad {
                lhs[t] += pow * children[i].y_zcol[t];
            }
            pow *= bK;
        }
        if lhs != parent.y_zcol {
            ok_y = false;
        }
    }

    let mut lhs_X = Mat::zero(D, m_in, F::ZERO);
    let mut pow = F::ONE;
    for i in 0..k_dec {
        for r in 0..D {
            for c in 0..m_in {
                lhs_X[(r, c)] += pow * children[i].X[(r, c)];
            }
        }
        pow *= bF;
    }
    let ok_X = lhs_X.as_slice() == parent.X.as_slice();

    let ok_c = combine_b_pows(&child_cs, params.b) == parent.c;
    Ok((children, child_cs, ok_y, ok_X, ok_c))
}

pub(crate) fn bind_rlc_inputs_with_context(
    tr: &mut Poseidon2Transcript,
    lane: RlcLane,
    step_idx: usize,
    me_inputs: &[CeClaim<Cmt, F, K>],
    backend_ctx: &neo_reductions::accelerator::BackendContext,
) -> Result<(), PiCcsError> {
    let lane_scope: &'static [u8] = match lane {
        RlcLane::Main => b"main",
        RlcLane::Val => b"val",
    };

    // v6: compact transcript binding via per-input Poseidon2 digest.
    // Encoding update: K-coefficient width is encoded once per K-slice (not per K element).
    tr.append_message(b"fold/rlc_inputs/v6", lane_scope);
    tr.append_u64s(b"step_idx", &[step_idx as u64]);
    tr.append_u64s(b"me_count", &[me_inputs.len() as u64]);

    #[inline]
    fn extend_packed_bytes_as_fields(dst: &mut Vec<F>, bytes: &[u8]) {
        const BYTES_PER_LIMB: usize = 7;
        dst.push(F::from_u64(bytes.len() as u64));
        for chunk in bytes.chunks(BYTES_PER_LIMB) {
            let mut limb = [0u8; 8];
            limb[..chunk.len()].copy_from_slice(chunk);
            dst.push(F::from_u64(u64::from_le_bytes(limb)));
        }
    }

    #[inline]
    fn extend_f_slice(dst: &mut Vec<F>, vals: &[F]) {
        dst.push(F::from_u64(vals.len() as u64));
        dst.extend_from_slice(vals);
    }

    #[inline]
    fn extend_k_slice(dst: &mut Vec<F>, vals: &[K]) {
        dst.push(F::from_u64(vals.len() as u64));
        let coeffs_len = vals.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
        dst.push(F::from_u64(coeffs_len as u64));
        for v in vals {
            let coeffs = v.as_coeffs();
            debug_assert_eq!(
                coeffs.len(),
                coeffs_len,
                "non-uniform K coeff length in RLC ME digest encoding"
            );
            dst.extend(coeffs.iter().copied());
        }
    }

    #[inline]
    fn poseidon_digest32_fields(input: &[F]) -> [u8; 32] {
        let digest = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(input);
        let mut out = [0u8; 32];
        for (i, limb) in digest.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&limb.as_canonical_u64().to_le_bytes());
        }
        out
    }

    #[inline]
    fn packed_bytes_as_fields_len(bytes_len: usize) -> usize {
        1 + bytes_len.div_ceil(7)
    }

    #[inline]
    fn f_slice_as_fields_len(vals_len: usize) -> usize {
        1 + vals_len
    }

    #[inline]
    fn k_slice_as_fields_len(vals: &[K]) -> usize {
        let coeffs_len = vals.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
        2 + vals.len() * coeffs_len
    }

    #[inline]
    fn me_digest_poseidon_input_capacity(lane_scope: &'static [u8], me: &CeClaim<Cmt, F, K>) -> usize {
        let mut cap = 0usize;
        cap += packed_bytes_as_fields_len(b"neo/fold/rlc_me_input_digest_poseidon/v2".len());
        cap += packed_bytes_as_fields_len(lane_scope.len());
        cap += f_slice_as_fields_len(me.c.data.len());
        cap += f_slice_as_fields_len(me.X.as_slice().len());
        cap += k_slice_as_fields_len(&me.r);
        cap += k_slice_as_fields_len(&me.s_col);
        cap += k_slice_as_fields_len(&me.y_zcol);
        cap += 1; // y_ring.len marker
        for row in &me.y_ring {
            cap += k_slice_as_fields_len(row);
        }
        cap += k_slice_as_fields_len(&me.ct);
        cap += k_slice_as_fields_len(&me.aux_openings);
        cap += f_slice_as_fields_len(me.c_step_coords.len());
        cap += 3; // m_in, u_offset, u_len
        cap += packed_bytes_as_fields_len(me.fold_digest.len());
        cap
    }

    #[inline]
    fn me_digest_poseidon_input(lane_scope: &'static [u8], me: &CeClaim<Cmt, F, K>, capacity_hint: usize) -> Vec<F> {
        let mut digest_input = Vec::<F>::with_capacity(capacity_hint.max(256));
        extend_packed_bytes_as_fields(&mut digest_input, b"neo/fold/rlc_me_input_digest_poseidon/v2");
        extend_packed_bytes_as_fields(&mut digest_input, lane_scope);

        extend_f_slice(&mut digest_input, &me.c.data);
        extend_f_slice(&mut digest_input, me.X.as_slice());
        extend_k_slice(&mut digest_input, &me.r);
        extend_k_slice(&mut digest_input, &me.s_col);
        extend_k_slice(&mut digest_input, &me.y_zcol);
        digest_input.push(F::from_u64(me.y_ring.len() as u64));
        for row in &me.y_ring {
            extend_k_slice(&mut digest_input, row);
        }
        extend_k_slice(&mut digest_input, &me.ct);
        extend_k_slice(&mut digest_input, &me.aux_openings);
        extend_f_slice(&mut digest_input, &me.c_step_coords);
        digest_input.push(F::from_u64(me.m_in as u64));
        digest_input.push(F::from_u64(me.u_offset as u64));
        digest_input.push(F::from_u64(me.u_len as u64));
        extend_packed_bytes_as_fields(&mut digest_input, &me.fold_digest);
        digest_input
    }

    let digest_capacity_hint = me_inputs
        .first()
        .map(|me| me_digest_poseidon_input_capacity(lane_scope, me))
        .unwrap_or(256);

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let allow_parallel = rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none();
    #[cfg(not(any(not(target_arch = "wasm32"), feature = "wasm-threads")))]
    let _allow_parallel = false;

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let digest_inputs: Vec<Vec<F>> = if allow_parallel && me_inputs.len() >= 8 {
        use rayon::prelude::*;
        me_inputs
            .par_iter()
            .map(|me| me_digest_poseidon_input(lane_scope, me, digest_capacity_hint))
            .collect()
    } else {
        me_inputs
            .iter()
            .map(|me| me_digest_poseidon_input(lane_scope, me, digest_capacity_hint))
            .collect()
    };
    #[cfg(not(any(not(target_arch = "wasm32"), feature = "wasm-threads")))]
    let digest_inputs: Vec<Vec<F>> = me_inputs
        .iter()
        .map(|me| me_digest_poseidon_input(lane_scope, me, digest_capacity_hint))
        .collect();

    if let Some(digests) =
        neo_reductions::accelerator::poseidon2_digest32_many_with_context(backend_ctx, &digest_inputs)?
    {
        for digest in digests {
            tr.append_bytes_packed(b"me/rlc_digest", &digest);
        }
        return Ok(());
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let digests: Vec<[u8; 32]> = if allow_parallel && digest_inputs.len() >= 8 {
        use rayon::prelude::*;
        digest_inputs
            .par_iter()
            .map(|input| poseidon_digest32_fields(input))
            .collect()
    } else {
        digest_inputs
            .iter()
            .map(|input| poseidon_digest32_fields(input))
            .collect()
    };
    #[cfg(not(any(not(target_arch = "wasm32"), feature = "wasm-threads")))]
    let digests: Vec<[u8; 32]> = digest_inputs
        .iter()
        .map(|input| poseidon_digest32_fields(input))
        .collect();

    for digest in digests {
        tr.append_bytes_packed(b"me/rlc_digest", &digest);
    }

    Ok(())
}
