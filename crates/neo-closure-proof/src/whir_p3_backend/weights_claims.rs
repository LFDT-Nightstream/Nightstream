//! Obligations-public weights/claims computation for the WHIR full-closure backend.
//!
//! This module centralizes the deterministic randomness + weight-table construction used by both
//! the prover and verifier paths. It makes the obligations dependency explicit so Phase 2 can
//! later swap from "recompute from obligations" to "verify from commitments/proofs".

#![forbid(unsafe_code)]

use super::{
    derive_seed_v1, neo_f_to_whir, u64_to_whir_f, Buffer, ChaCha8Rng, ClosureProofError, ClosureStatementV1,
    EvaluationsList, MmapBuffer, NeoCmt, NeoD, NeoF, DEFAULT_MMAP_THRESHOLD_BYTES, F,
};
use neo_math::KExtensions;
use p3_field::PrimeCharacteristicRing as _;
use rand::{RngCore, SeedableRng};

pub(crate) struct FullClosurePublicWeightsAndClaims {
    pub(crate) claimed_sum: F,
    pub(crate) delta_range: F,
    pub(crate) r0: Vec<F>,
    pub(crate) w_evals: EvaluationsList<F>,
}

pub(crate) struct FullClosurePublicClaims {
    pub(crate) claimed_sum: F,
    pub(crate) delta_range: F,
    pub(crate) r0: Vec<F>,
}

pub(crate) fn compute_full_closure_public_w_r_expected_at_point(
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    d: usize,
    m: usize,
    kappa: usize,
    pp_seed: [u8; 32],
    commitment_root_u64: &[u64],
    z_len_padded: usize,
    num_vars: usize,
    point_msb: &[F],
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<F, ClosureProofError> {
    let _ = params; // b is only used by the range-check term, not for W weights.
    if d != NeoD {
        return Err(ClosureProofError::WhirP3(format!(
            "unexpected d (must match neo_math::D): got {d}, expected {NeoD}",
        )));
    }
    if point_msb.len() != num_vars {
        return Err(ClosureProofError::WhirP3("W(r) point length mismatch".into()));
    }

    let obligation_count = obligations.main.len() + obligations.val.len();
    if obligation_count == 0 {
        return Ok(F::ZERO);
    }

    let z_len = obligation_count
        .checked_mul(d)
        .and_then(|x| x.checked_mul(m))
        .ok_or_else(|| ClosureProofError::WhirP3("z_len overflow".into()))?;
    if z_len > z_len_padded {
        return Err(ClosureProofError::WhirP3("z_len exceeds z_len_padded".into()));
    }
    let expected_padded = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| ClosureProofError::WhirP3("z_len_padded overflow".into()))?;
    if expected_padded != z_len_padded {
        return Err(ClosureProofError::WhirP3("z_len_padded mismatch vs num_vars".into()));
    }
    if z_len > expected_padded {
        return Err(ClosureProofError::WhirP3("z_len exceeds 2^num_vars".into()));
    }

    let (u_vecs, lambdas) = derive_u_and_lambdas(stmt, commitment_root_u64, kappa, obligation_count);

    let w_u_len = d
        .checked_mul(m)
        .ok_or_else(|| ClosureProofError::WhirP3("w_u_len overflow".into()))?;
    let w_u_bytes = w_u_len.saturating_mul(core::mem::size_of::<F>());
    let mut w_u = if w_u_bytes >= DEFAULT_MMAP_THRESHOLD_BYTES {
        Buffer::Mmap(
            MmapBuffer::new_zeroed(w_u_len)
                .map_err(|e| ClosureProofError::WhirP3(format!("mmap alloc w_u failed: {e}")))?,
        )
    } else {
        Buffer::Vec(F::zero_vec(w_u_len))
    };
    neo_ajtai::compute_opening_weights_for_u_seeded_into(pp_seed, m, &u_vecs, w_u.as_mut_slice());

    // X-projection RNG + mixer scalar.
    let seed_x = derive_seed_v1(b"ajtai_opening_plus_x/rng", stmt, Some(commitment_root_u64));
    let mut rng_x = ChaCha8Rng::from_seed(seed_x);
    let mut gamma_x = NeoF::from_u64(rng_x.next_u64());
    if gamma_x == NeoF::ZERO {
        gamma_x = NeoF::ONE;
    }
    let gamma_x = neo_f_to_whir(gamma_x);

    // ME-consistency RNG + mixer scalars/weights.
    let seed_me = derive_seed_v1(b"full_closure/rng", stmt, Some(commitment_root_u64));
    let mut rng_me = ChaCha8Rng::from_seed(seed_me);

    let mut gamma_me = neo_math::F::from_u64(rng_me.next_u64());
    if gamma_me == neo_math::F::ZERO {
        gamma_me = neo_math::F::ONE;
    }
    let gamma_me = neo_f_to_whir(gamma_me);

    let mut delta_k = neo_math::F::from_u64(rng_me.next_u64());
    if delta_k == neo_math::F::ZERO {
        delta_k = neo_math::F::ONE;
    }

    let mut nu = vec![neo_math::F::ZERO; d];
    for rho in 0..d {
        nu[rho] = neo_math::F::from_u64(rng_me.next_u64());
    }

    let core_t = ccs.t();
    let mut bus_cols_expected: Option<usize> = None;
    for me in obligations.main.iter().chain(obligations.val.iter()) {
        if me.y.len() != me.y_scalars.len() {
            return Err(ClosureProofError::WhirP3("ME y/y_scalars length mismatch".into()));
        }
        if me.y.len() < core_t {
            return Err(ClosureProofError::WhirP3("ME y.len() < core_t".into()));
        }
        let bus_cols = me.y.len() - core_t;
        match bus_cols_expected {
            None => bus_cols_expected = Some(bus_cols),
            Some(prev) if prev != bus_cols => {
                return Err(ClosureProofError::WhirP3("ME bus_cols mismatch across obligations".into()));
            }
            _ => {}
        }
        if bus_cols > 0 {
            let bus = bus
                .ok_or_else(|| ClosureProofError::WhirP3("ME has bus openings but no BusLayout provided".into()))?;
            if bus.bus_cols != bus_cols || bus.m != m {
                return Err(ClosureProofError::WhirP3("BusLayout mismatch".into()));
            }
            if me.m_in != bus.m_in {
                return Err(ClosureProofError::WhirP3("ME m_in != bus.m_in".into()));
            }
        }
    }
    let bus_cols = bus_cols_expected.unwrap_or(0);

    let mut mu_core = vec![neo_math::F::ZERO; core_t];
    for j in 0..core_t {
        mu_core[j] = neo_math::F::from_u64(rng_me.next_u64());
    }
    let mut mu_bus = vec![neo_math::F::ZERO; bus_cols];
    for col_id in 0..bus_cols {
        mu_bus[col_id] = neo_math::F::from_u64(rng_me.next_u64());
    }

    fn compute_col_weights_for_me(
        ccs: &neo_ccs::CcsStructure<neo_math::F>,
        me: &neo_ccs::MeInstance<NeoCmt, neo_math::F, neo_math::K>,
        d: usize,
        m: usize,
        delta_k: neo_math::F,
        mu_core: &[neo_math::F],
        mu_bus: &[neo_math::F],
        bus_cols: usize,
        bus: Option<&neo_memory::cpu::BusLayout>,
    ) -> Result<Vec<neo_math::F>, ClosureProofError> {
        let rb_mix = compute_rb_mix(&me.r, delta_k);
        let n_eff = core::cmp::min(ccs.n, rb_mix.len());
        let mut col_weights = vec![neo_math::F::ZERO; m];

        for (j, mat) in ccs.matrices.iter().enumerate() {
            let mu_j = mu_core
                .get(j)
                .copied()
                .ok_or_else(|| ClosureProofError::WhirP3("mu_core count mismatch".into()))?;
            if mu_j == neo_math::F::ZERO {
                continue;
            }
            match mat {
                neo_ccs::CcsMatrix::Identity { n } => {
                    let cap = core::cmp::min(n_eff, *n);
                    if cap > m {
                        return Err(ClosureProofError::WhirP3("identity matrix n exceeds m".into()));
                    }
                    for idx in 0..cap {
                        col_weights[idx] += mu_j * rb_mix[idx];
                    }
                }
                neo_ccs::CcsMatrix::Csc(csc) => {
                    if csc.ncols > m {
                        return Err(ClosureProofError::WhirP3("CSC matrix ncols exceeds m".into()));
                    }
                    for c in 0..csc.ncols {
                        let s0 = csc.col_ptr[c];
                        let e0 = csc.col_ptr[c + 1];
                        for k in s0..e0 {
                            let row = csc.row_idx[k];
                            if row >= n_eff {
                                continue;
                            }
                            let wr = rb_mix[row];
                            if wr == neo_math::F::ZERO {
                                continue;
                            }
                            col_weights[c] += mu_j * wr * csc.vals[k];
                        }
                    }
                }
            }
        }

        if bus_cols > 0 {
            let bus = bus
                .ok_or_else(|| ClosureProofError::WhirP3("ME has bus openings but no BusLayout provided".into()))?;
            for col_id in 0..bus_cols {
                let mu = mu_bus[col_id];
                if mu == neo_math::F::ZERO {
                    continue;
                }
                for j in 0..bus.chunk_size {
                    let row = bus.time_index(j);
                    let w_time = chi_for_row_index(&me.r, row);
                    let w_time_mix = mix_k_to_f(w_time, delta_k);
                    let z_idx = bus.bus_cell(col_id, j);
                    if z_idx >= m {
                        return Err(ClosureProofError::WhirP3("bus_cell out of range".into()));
                    }
                    col_weights[z_idx] += mu * w_time_mix;
                }
            }
        }

        let _ = d; // used implicitly via `chi_for_row_index` and ME constraints.
        Ok(col_weights)
    }

    // Enumerate eq(x, point) for x in [0, z_len) and accumulate W(x) * eq(x, point).
    let mut out = F::ZERO;
    let mut produced = 0usize;

    let mes: Vec<_> = obligations.main.iter().chain(obligations.val.iter()).collect();
    if mes.len() != obligation_count {
        return Err(ClosureProofError::WhirP3("obligation count mismatch".into()));
    }

    struct WCursor {
        obligation_idx: usize,
        row: usize,
        col: usize,
    }
    let mut cursor = WCursor {
        obligation_idx: 0,
        row: 0,
        col: 0,
    };

    let mut col_weights = compute_col_weights_for_me(
        ccs,
        mes[0],
        d,
        m,
        delta_k,
        &mu_core,
        &mu_bus,
        bus_cols,
        bus,
    )?;

    fn rec(
        point_msb: &[F],
        bit: usize,
        acc: F,
        limit: usize,
        produced: &mut usize,
        out: &mut F,
        cursor: &mut WCursor,
        obligations: &[&neo_ccs::MeInstance<NeoCmt, neo_math::F, neo_math::K>],
        lambdas: &[neo_math::F],
        d: usize,
        m: usize,
        w_u: &[F],
        gamma_x: F,
        rng_x: &mut ChaCha8Rng,
        gamma_me: F,
        nu: &[neo_math::F],
        col_weights: &mut Vec<neo_math::F>,
        ccs: &neo_ccs::CcsStructure<neo_math::F>,
        delta_k: neo_math::F,
        mu_core: &[neo_math::F],
        mu_bus: &[neo_math::F],
        bus_cols: usize,
        bus: Option<&neo_memory::cpu::BusLayout>,
    ) -> Result<(), ClosureProofError> {
        if *produced >= limit {
            return Ok(());
        }
        if bit == point_msb.len() {
            let me = *obligations
                .get(cursor.obligation_idx)
                .ok_or_else(|| ClosureProofError::WhirP3("obligation index overflow".into()))?;
            let lambda_i = lambdas
                .get(cursor.obligation_idx)
                .copied()
                .ok_or_else(|| ClosureProofError::WhirP3("lambda count mismatch".into()))?;

            // Base Ajtai opening weights.
            let w_u_idx = cursor
                .row
                .checked_mul(m)
                .and_then(|x| x.checked_add(cursor.col))
                .ok_or_else(|| ClosureProofError::WhirP3("w_u index overflow".into()))?;
            let w_u_entry = *w_u
                .get(w_u_idx)
                .ok_or_else(|| ClosureProofError::WhirP3("w_u index out of range".into()))?;
            let mut w_entry = neo_f_to_whir(lambda_i) * w_u_entry;

            // X projection weights: sample betas only for col < m_in.
            if cursor.col < me.m_in {
                let beta = neo_f_to_whir(NeoF::from_u64(rng_x.next_u64()));
                w_entry += gamma_x * beta;
            }

            // ME consistency weights.
            let nu_rho = nu
                .get(cursor.row)
                .copied()
                .ok_or_else(|| ClosureProofError::WhirP3("nu index out of range".into()))?;
            let cw = col_weights
                .get(cursor.col)
                .copied()
                .ok_or_else(|| ClosureProofError::WhirP3("col_weights index out of range".into()))?;
            w_entry += gamma_me * neo_f_to_whir(lambda_i * nu_rho * cw);

            *out += acc * w_entry;
            *produced += 1;

            // Advance cursor.
            cursor.col += 1;
            if cursor.col == m {
                cursor.col = 0;
                cursor.row += 1;
                if cursor.row == d {
                    cursor.row = 0;
                    cursor.obligation_idx += 1;
                    if cursor.obligation_idx < obligations.len() {
                        *col_weights = compute_col_weights_for_me(
                            ccs,
                            obligations[cursor.obligation_idx],
                            d,
                            m,
                            delta_k,
                            mu_core,
                            mu_bus,
                            bus_cols,
                            bus,
                        )?;
                    }
                }
            }

            return Ok(());
        }

        let r = point_msb[bit];
        rec(
            point_msb,
            bit + 1,
            acc * (F::ONE - r),
            limit,
            produced,
            out,
            cursor,
            obligations,
            lambdas,
            d,
            m,
            w_u,
            gamma_x,
            rng_x,
            gamma_me,
            nu,
            col_weights,
            ccs,
            delta_k,
            mu_core,
            mu_bus,
            bus_cols,
            bus,
        )?;
        rec(
            point_msb,
            bit + 1,
            acc * r,
            limit,
            produced,
            out,
            cursor,
            obligations,
            lambdas,
            d,
            m,
            w_u,
            gamma_x,
            rng_x,
            gamma_me,
            nu,
            col_weights,
            ccs,
            delta_k,
            mu_core,
            mu_bus,
            bus_cols,
            bus,
        )
    }

    // Validate X shapes once (keeps the cursor logic lean).
    for me in mes.iter() {
        if me.m_in > m {
            return Err(ClosureProofError::WhirP3("m_in exceeds commitment width".into()));
        }
        if me.X.rows() != d || me.X.cols() != me.m_in {
            return Err(ClosureProofError::WhirP3("X shape mismatch".into()));
        }
    }

    rec(
        point_msb,
        0,
        F::ONE,
        z_len,
        &mut produced,
        &mut out,
        &mut cursor,
        mes.as_slice(),
        &lambdas,
        d,
        m,
        w_u.as_slice(),
        gamma_x,
        &mut rng_x,
        gamma_me,
        &nu,
        &mut col_weights,
        ccs,
        delta_k,
        &mu_core,
        &mu_bus,
        bus_cols,
        bus,
    )?;

    Ok(out)
}

pub(crate) fn compute_full_closure_public_weights_and_claims(
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    d: usize,
    m: usize,
    kappa: usize,
    pp_seed: [u8; 32],
    commitment_root_u64: &[u64],
    z_len_padded: usize,
    num_vars: usize,
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<FullClosurePublicWeightsAndClaims, ClosureProofError> {
    if d != NeoD {
        return Err(ClosureProofError::WhirP3(format!(
            "unexpected d (must match neo_math::D): got {d}, expected {NeoD}",
        )));
    }

    let obligation_count = obligations.main.len() + obligations.val.len();
    let z_len = obligation_count
        .checked_mul(d)
        .and_then(|x| x.checked_mul(m))
        .ok_or_else(|| ClosureProofError::WhirP3("z_len overflow".into()))?;
    if z_len > z_len_padded {
        return Err(ClosureProofError::WhirP3("z_len exceeds z_len_padded".into()));
    }

    // Derive batching randomness (u vectors and per-obligation weights).
    let (u_vecs, lambdas) = derive_u_and_lambdas(stmt, commitment_root_u64, kappa, obligation_count);

    // Build the Ajtai opening weight vector for the chosen u (one per Z entry).
    let w_u_len = d
        .checked_mul(m)
        .ok_or_else(|| ClosureProofError::WhirP3("w_u_len overflow".into()))?;
    let w_u_bytes = w_u_len.saturating_mul(core::mem::size_of::<F>());
    let mut w_u = if w_u_bytes >= DEFAULT_MMAP_THRESHOLD_BYTES {
        Buffer::Mmap(
            MmapBuffer::new_zeroed(w_u_len)
                .map_err(|e| ClosureProofError::WhirP3(format!("mmap alloc w_u failed: {e}")))?,
        )
    } else {
        Buffer::Vec(F::zero_vec(w_u_len))
    };
    neo_ajtai::compute_opening_weights_for_u_seeded_into(pp_seed, m, &u_vecs, w_u.as_mut_slice());

    // Compute the claimed sum t = Σ_i λ_i · <u, c_i>.
    let mut claimed_sum_neo = NeoF::ZERO;
    for (idx, me) in obligations
        .main
        .iter()
        .chain(obligations.val.iter())
        .enumerate()
    {
        let t_i = dot_u_commitment(&u_vecs, &me.c)?;
        claimed_sum_neo += lambdas[idx] * t_i;
    }

    // Build w_total = concat_i (λ_i * w_u), padded.
    let mut w_evals = EvaluationsList::<F>::new_zeroed(z_len_padded);
    {
        let w_out = w_evals.as_mut_slice();
        let mut w_idx = 0usize;
        for lambda in lambdas.iter() {
            let lambda_whir = neo_f_to_whir(*lambda);
            for &w in w_u.as_slice() {
                w_out[w_idx] = lambda_whir * neo_f_to_whir(w);
                w_idx += 1;
            }
        }
        debug_assert_eq!(w_idx, z_len);
    }

    // X-projection: fold a random linear check into the same weight vector.
    let (gamma_x, claimed_x) =
        mix_in_x_projection(stmt, commitment_root_u64, obligations, d, m, w_evals.as_mut_slice())?;
    claimed_sum_neo += gamma_x * claimed_x;

    // ME consistency: fold a random linear check into the same weight vector.
    let (gamma_me, claimed_me) = mix_in_me_consistency_core_and_bus(
        stmt,
        commitment_root_u64,
        params,
        ccs,
        obligations,
        d,
        m,
        w_evals.as_mut_slice(),
        &lambdas,
        bus,
    )?;
    claimed_sum_neo += gamma_me * claimed_me;

    let claimed_sum = neo_f_to_whir(claimed_sum_neo);

    // Range check: derive a random "Eq" point r0 and a mixing scalar δ_range.
    let mut rng = ChaCha8Rng::from_seed(derive_seed_v1(
        b"full_closure/range_rng",
        stmt,
        Some(commitment_root_u64),
    ));
    let mut delta_range_neo = NeoF::from_u64(rng.next_u64());
    if delta_range_neo == NeoF::ZERO {
        delta_range_neo = NeoF::ONE;
    }
    let delta_range = neo_f_to_whir(delta_range_neo);

    let mut r0 = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        r0.push(u64_to_whir_f(rng.next_u64()));
    }

    Ok(FullClosurePublicWeightsAndClaims {
        claimed_sum,
        delta_range,
        r0,
        w_evals,
    })
}

pub(crate) fn compute_full_closure_public_claims_and_rng(
    stmt: &ClosureStatementV1,
    _params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    d: usize,
    m: usize,
    kappa: usize,
    commitment_root_u64: &[u64],
    num_vars: usize,
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<FullClosurePublicClaims, ClosureProofError> {
    if d != NeoD {
        return Err(ClosureProofError::WhirP3(format!(
            "unexpected d (must match neo_math::D): got {d}, expected {NeoD}",
        )));
    }

    let obligation_count = obligations.main.len() + obligations.val.len();

    // Derive batching randomness (u vectors and per-obligation weights).
    let (u_vecs, lambdas) = derive_u_and_lambdas(stmt, commitment_root_u64, kappa, obligation_count);

    // Claimed opening sum from commitments.
    let mut claimed_sum_neo = NeoF::ZERO;
    for (idx, me) in obligations
        .main
        .iter()
        .chain(obligations.val.iter())
        .enumerate()
    {
        let t_i = dot_u_commitment(&u_vecs, &me.c)?;
        claimed_sum_neo += lambdas[idx] * t_i;
    }

    // X-projection claimed sum (no Z weights needed for verification).
    let (gamma_x, claimed_x) = compute_x_projection_claim(stmt, commitment_root_u64, obligations, d, m)?;
    claimed_sum_neo += gamma_x * claimed_x;

    // ME consistency claimed sum (no weights table needed for verification).
    let (gamma_me, claimed_me) =
        compute_me_consistency_claim(stmt, commitment_root_u64, ccs, obligations, d, m, &lambdas, bus)?;
    claimed_sum_neo += gamma_me * claimed_me;

    let claimed_sum = neo_f_to_whir(claimed_sum_neo);

    // Range check randomness: derive a random "Eq" point r0 and a mixing scalar δ_range.
    let mut rng = ChaCha8Rng::from_seed(derive_seed_v1(
        b"full_closure/range_rng",
        stmt,
        Some(commitment_root_u64),
    ));
    let mut delta_range_neo = NeoF::from_u64(rng.next_u64());
    if delta_range_neo == NeoF::ZERO {
        delta_range_neo = NeoF::ONE;
    }
    let delta_range = neo_f_to_whir(delta_range_neo);

    let mut r0 = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        r0.push(u64_to_whir_f(rng.next_u64()));
    }

    Ok(FullClosurePublicClaims {
        claimed_sum,
        delta_range,
        r0,
    })
}

fn derive_u_and_lambdas(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    kappa: usize,
    obligation_count: usize,
) -> (Vec<[NeoF; NeoD]>, Vec<NeoF>) {
    // u: κ vectors in F^d
    // λ: per-obligation weights in F
    let seed = derive_seed_v1(b"ajtai_opening_only/u_and_lambdas", stmt, Some(commitment_root_u64));
    let mut rng = ChaCha8Rng::from_seed(seed);

    let mut u_vecs = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        let mut v = [NeoF::ZERO; NeoD];
        for i in 0..NeoD {
            v[i] = NeoF::from_u64(rng.next_u64());
        }
        u_vecs.push(v);
    }

    let mut lambdas = Vec::with_capacity(obligation_count);
    for _ in 0..obligation_count {
        lambdas.push(NeoF::from_u64(rng.next_u64()));
    }

    (u_vecs, lambdas)
}

fn dot_u_commitment(u_vecs: &[[NeoF; NeoD]], c: &NeoCmt) -> Result<NeoF, ClosureProofError> {
    let d = c.d;
    let kappa = c.kappa;
    if d != NeoD {
        return Err(ClosureProofError::WhirP3(format!(
            "commitment d mismatch: got {d}, expected {NeoD}",
        )));
    }
    if kappa != u_vecs.len() {
        return Err(ClosureProofError::WhirP3(format!(
            "commitment κ mismatch: got {kappa}, expected {}",
            u_vecs.len(),
        )));
    }
    if c.data.len() != d * kappa {
        return Err(ClosureProofError::WhirP3("commitment data length mismatch".into()));
    }

    let mut acc = NeoF::ZERO;
    for i in 0..kappa {
        for r in 0..d {
            acc += u_vecs[i][r] * c.data[i * d + r];
        }
    }
    Ok(acc)
}

fn mix_k_to_f(k: neo_math::K, delta: neo_math::F) -> neo_math::F {
    let coeffs = k.as_coeffs();
    coeffs[0] + delta * coeffs[1]
}

fn compute_rb_mix(r: &[neo_math::K], delta: neo_math::F) -> Vec<neo_math::F> {
    let rb = neo_ccs::utils::tensor_point::<neo_math::K>(r);
    rb.into_iter().map(|x| mix_k_to_f(x, delta)).collect()
}

fn chi_for_row_index(r: &[neo_math::K], idx: usize) -> neo_math::K {
    let mut acc = neo_math::K::ONE;
    for (bit, &ri) in r.iter().enumerate() {
        let is_one = ((idx >> bit) & 1) == 1;
        acc *= if is_one { ri } else { neo_math::K::ONE - ri };
    }
    acc
}

fn mix_in_me_consistency_core_and_bus(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    _params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    d: usize,
    m: usize,
    w_evals_whir: &mut [F],
    lambdas: &[neo_math::F],
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<(neo_math::F, neo_math::F), ClosureProofError> {
    // Randomize a single scalar ME-consistency check:
    //   Σ_i λ_i * Σ_j μ_j * Σ_ρ ν_ρ * y_i,j[ρ]  ==  Σ_i λ_i * Σ_ρ ν_ρ * Σ_c Z_i[ρ,c] * (Σ_j μ_j v_i,j[c])
    //
    // We work over the base field by mixing K components with a random δ.
    let seed = derive_seed_v1(b"full_closure/rng", stmt, Some(commitment_root_u64));
    let mut rng = ChaCha8Rng::from_seed(seed);

    // Mixer scalar γ_me; ensure nonzero to avoid accidentally disabling the check.
    let mut gamma_me = neo_math::F::from_u64(rng.next_u64());
    if gamma_me == neo_math::F::ZERO {
        gamma_me = neo_math::F::ONE;
    }
    let gamma_me_whir = neo_f_to_whir(gamma_me);

    // Mix K → F as c0 + δ*c1; δ must be nonzero with overwhelming probability.
    let mut delta_k = neo_math::F::from_u64(rng.next_u64());
    if delta_k == neo_math::F::ZERO {
        delta_k = neo_math::F::ONE;
    }

    // Row weights ν_ρ for ρ in 0..d.
    let mut nu = vec![neo_math::F::ZERO; d];
    for rho in 0..d {
        nu[rho] = neo_math::F::from_u64(rng.next_u64());
    }

    let core_t = ccs.t();
    let mut bus_cols_expected: Option<usize> = None;
    for me in obligations.main.iter().chain(obligations.val.iter()) {
        if me.y.len() != me.y_scalars.len() {
            return Err(ClosureProofError::WhirP3("ME y/y_scalars length mismatch".into()));
        }
        if me.y.len() < core_t {
            return Err(ClosureProofError::WhirP3("ME y.len() < core_t".into()));
        }
        let bus_cols = me.y.len() - core_t;
        match bus_cols_expected {
            None => bus_cols_expected = Some(bus_cols),
            Some(prev) if prev != bus_cols => {
                return Err(ClosureProofError::WhirP3(
                    "ME bus_cols mismatch across obligations".into(),
                ));
            }
            _ => {}
        }
        if bus_cols > 0 {
            let bus =
                bus.ok_or_else(|| ClosureProofError::WhirP3("ME has bus openings but no BusLayout provided".into()))?;
            if bus.bus_cols != bus_cols || bus.m != m {
                return Err(ClosureProofError::WhirP3("BusLayout mismatch".into()));
            }
            if me.m_in != bus.m_in {
                return Err(ClosureProofError::WhirP3("ME m_in != bus.m_in".into()));
            }
        }
    }
    let bus_cols = bus_cols_expected.unwrap_or(0);

    // Matrix weights μ_j (core) and μ_bus[col_id] (bus openings).
    let mut mu_core = vec![neo_math::F::ZERO; core_t];
    for j in 0..core_t {
        mu_core[j] = neo_math::F::from_u64(rng.next_u64());
    }
    let mut mu_bus = vec![neo_math::F::ZERO; bus_cols];
    for col_id in 0..bus_cols {
        mu_bus[col_id] = neo_math::F::from_u64(rng.next_u64());
    }

    let mut claimed_me = neo_math::F::ZERO;
    let mut base_idx = 0usize;

    for (i, me) in obligations
        .main
        .iter()
        .chain(obligations.val.iter())
        .enumerate()
    {
        let lambda_i = lambdas
            .get(i)
            .copied()
            .ok_or_else(|| ClosureProofError::WhirP3("lambda count mismatch".into()))?;

        // Precompute r^b mix for this obligation.
        let rb_mix = compute_rb_mix(&me.r, delta_k);
        let n_eff = core::cmp::min(ccs.n, rb_mix.len());

        // Column weights s[c] = Σ_j μ_j * v_j[c] (mixed into base field).
        let mut col_weights = vec![neo_math::F::ZERO; m];

        for (j, mat) in ccs.matrices.iter().enumerate() {
            let mu_j = mu_core[j];
            if mu_j == neo_math::F::ZERO {
                continue;
            }
            match mat {
                neo_ccs::CcsMatrix::Identity { n } => {
                    let cap = core::cmp::min(n_eff, *n);
                    for idx in 0..cap {
                        col_weights[idx] += mu_j * rb_mix[idx];
                    }
                }
                neo_ccs::CcsMatrix::Csc(csc) => {
                    for c in 0..csc.ncols {
                        let s0 = csc.col_ptr[c];
                        let e0 = csc.col_ptr[c + 1];
                        for k in s0..e0 {
                            let row = csc.row_idx[k];
                            if row >= n_eff {
                                continue;
                            }
                            let wr = rb_mix[row];
                            if wr == neo_math::F::ZERO {
                                continue;
                            }
                            col_weights[c] += mu_j * wr * csc.vals[k];
                        }
                    }
                }
            }
        }

        if bus_cols > 0 {
            let bus =
                bus.ok_or_else(|| ClosureProofError::WhirP3("ME has bus openings but no BusLayout provided".into()))?;
            for col_id in 0..bus_cols {
                let mu = mu_bus[col_id];
                if mu == neo_math::F::ZERO {
                    continue;
                }
                for j in 0..bus.chunk_size {
                    let row = bus.time_index(j);
                    let w_time = chi_for_row_index(&me.r, row);
                    let w_time_mix = mix_k_to_f(w_time, delta_k);
                    let z_idx = bus.bus_cell(col_id, j);
                    if z_idx >= m {
                        return Err(ClosureProofError::WhirP3("bus_cell out of range".into()));
                    }
                    col_weights[z_idx] += mu * w_time_mix;
                }
            }
        }

        // Claimed sum from public y values.
        for j in 0..core_t {
            let mu_j = mu_core[j];
            if mu_j == neo_math::F::ZERO {
                continue;
            }
            let yj =
                me.y.get(j)
                    .ok_or_else(|| ClosureProofError::WhirP3("ME y missing core entry".into()))?;
            if yj.len() < d {
                return Err(ClosureProofError::WhirP3("ME y row too short".into()));
            }
            let mut dot = neo_math::F::ZERO;
            for rho in 0..d {
                dot += nu[rho] * mix_k_to_f(yj[rho], delta_k);
            }
            claimed_me += lambda_i * mu_j * dot;
        }

        for col_id in 0..bus_cols {
            let mu = mu_bus[col_id];
            if mu == neo_math::F::ZERO {
                continue;
            }
            let yj =
                me.y.get(core_t + col_id)
                    .ok_or_else(|| ClosureProofError::WhirP3("ME y missing bus entry".into()))?;
            if yj.len() < d {
                return Err(ClosureProofError::WhirP3("ME y bus row too short".into()));
            }
            let mut dot = neo_math::F::ZERO;
            for rho in 0..d {
                dot += nu[rho] * mix_k_to_f(yj[rho], delta_k);
            }
            claimed_me += lambda_i * mu * dot;
        }

        // Add Z weights for this obligation.
        for rho in 0..d {
            let row_scale = lambda_i * nu[rho];
            if row_scale == neo_math::F::ZERO {
                continue;
            }
            let row_base = base_idx
                .checked_add(rho * m)
                .ok_or_else(|| ClosureProofError::WhirP3("weight index overflow".into()))?;
            for c in 0..m {
                let idx = row_base + c;
                let w = row_scale * col_weights[c];
                if w != neo_math::F::ZERO {
                    w_evals_whir[idx] += gamma_me_whir * neo_f_to_whir(w);
                }
            }
        }

        base_idx = base_idx
            .checked_add(d * m)
            .ok_or_else(|| ClosureProofError::WhirP3("base_idx overflow".into()))?;
    }

    Ok((gamma_me, claimed_me))
}

fn mix_in_x_projection(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    d: usize,
    m: usize,
    w_evals_whir: &mut [F],
) -> Result<(NeoF, NeoF), ClosureProofError> {
    let seed = derive_seed_v1(b"ajtai_opening_plus_x/rng", stmt, Some(commitment_root_u64));
    let mut rng = ChaCha8Rng::from_seed(seed);

    // Mixer scalar γ; if γ=0 (prob ~2^-64), bump to 1 so the check isn't accidentally disabled.
    let mut gamma = NeoF::from_u64(rng.next_u64());
    if gamma == NeoF::ZERO {
        gamma = NeoF::ONE;
    }
    let gamma_whir = neo_f_to_whir(gamma);

    let mut claimed_x = NeoF::ZERO;
    let mut base_idx = 0usize;

    for me in obligations.main.iter().chain(obligations.val.iter()) {
        let m_in = me.m_in;
        if m_in > m {
            return Err(ClosureProofError::WhirP3("m_in exceeds commitment width".into()));
        }
        if me.X.rows() != d || me.X.cols() != m_in {
            return Err(ClosureProofError::WhirP3("X shape mismatch".into()));
        }
        for row in 0..d {
            for col in 0..m_in {
                let beta = NeoF::from_u64(rng.next_u64());
                claimed_x += beta * me.X[(row, col)];

                let idx = base_idx + row * m + col;
                w_evals_whir[idx] += gamma_whir * neo_f_to_whir(beta);
            }
        }
        base_idx += d * m;
    }

    Ok((gamma, claimed_x))
}

fn compute_x_projection_claim(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    d: usize,
    m: usize,
) -> Result<(NeoF, NeoF), ClosureProofError> {
    let seed = derive_seed_v1(b"ajtai_opening_plus_x/rng", stmt, Some(commitment_root_u64));
    let mut rng = ChaCha8Rng::from_seed(seed);

    // Mixer scalar γ; if γ=0 (prob ~2^-64), bump to 1 so the check isn't accidentally disabled.
    let mut gamma = NeoF::from_u64(rng.next_u64());
    if gamma == NeoF::ZERO {
        gamma = NeoF::ONE;
    }

    let mut claimed_x = NeoF::ZERO;

    for me in obligations.main.iter().chain(obligations.val.iter()) {
        let m_in = me.m_in;
        if m_in > m {
            return Err(ClosureProofError::WhirP3("m_in exceeds commitment width".into()));
        }
        if me.X.rows() != d || me.X.cols() != m_in {
            return Err(ClosureProofError::WhirP3("X shape mismatch".into()));
        }
        for row in 0..d {
            for col in 0..m_in {
                let beta = NeoF::from_u64(rng.next_u64());
                claimed_x += beta * me.X[(row, col)];
            }
        }
    }

    Ok((gamma, claimed_x))
}

fn compute_me_consistency_claim(
    stmt: &ClosureStatementV1,
    commitment_root_u64: &[u64],
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<NeoCmt, neo_math::F, neo_math::K>,
    d: usize,
    m: usize,
    lambdas: &[neo_math::F],
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<(neo_math::F, neo_math::F), ClosureProofError> {
    // Claimed ME-consistency value (the weights table itself is proven via the sumcheck).
    //
    // We work over the base field by mixing K components with a random δ.
    let seed = derive_seed_v1(b"full_closure/rng", stmt, Some(commitment_root_u64));
    let mut rng = ChaCha8Rng::from_seed(seed);

    // Mixer scalar γ_me; ensure nonzero to avoid accidentally disabling the check.
    let mut gamma_me = neo_math::F::from_u64(rng.next_u64());
    if gamma_me == neo_math::F::ZERO {
        gamma_me = neo_math::F::ONE;
    }

    // Mix K → F as c0 + δ*c1; δ must be nonzero with overwhelming probability.
    let mut delta_k = neo_math::F::from_u64(rng.next_u64());
    if delta_k == neo_math::F::ZERO {
        delta_k = neo_math::F::ONE;
    }

    // Row weights ν_ρ for ρ in 0..d.
    let mut nu = vec![neo_math::F::ZERO; d];
    for rho in 0..d {
        nu[rho] = neo_math::F::from_u64(rng.next_u64());
    }

    let core_t = ccs.t();
    let mut bus_cols_expected: Option<usize> = None;
    for me in obligations.main.iter().chain(obligations.val.iter()) {
        if me.y.len() != me.y_scalars.len() {
            return Err(ClosureProofError::WhirP3("ME y/y_scalars length mismatch".into()));
        }
        if me.y.len() < core_t {
            return Err(ClosureProofError::WhirP3("ME y.len() < core_t".into()));
        }
        let bus_cols = me.y.len() - core_t;
        match bus_cols_expected {
            None => bus_cols_expected = Some(bus_cols),
            Some(prev) if prev != bus_cols => {
                return Err(ClosureProofError::WhirP3(
                    "ME bus_cols mismatch across obligations".into(),
                ));
            }
            _ => {}
        }
        if bus_cols > 0 {
            let bus =
                bus.ok_or_else(|| ClosureProofError::WhirP3("ME has bus openings but no BusLayout provided".into()))?;
            if bus.bus_cols != bus_cols || bus.m != m {
                return Err(ClosureProofError::WhirP3("BusLayout mismatch".into()));
            }
            if me.m_in != bus.m_in {
                return Err(ClosureProofError::WhirP3("ME m_in != bus.m_in".into()));
            }
        }
    }
    let bus_cols = bus_cols_expected.unwrap_or(0);

    // Matrix weights μ_j (core) and μ_bus[col_id] (bus openings).
    let mut mu_core = vec![neo_math::F::ZERO; core_t];
    for j in 0..core_t {
        mu_core[j] = neo_math::F::from_u64(rng.next_u64());
    }
    let mut mu_bus = vec![neo_math::F::ZERO; bus_cols];
    for col_id in 0..bus_cols {
        mu_bus[col_id] = neo_math::F::from_u64(rng.next_u64());
    }

    let mut claimed_me = neo_math::F::ZERO;

    for (i, me) in obligations
        .main
        .iter()
        .chain(obligations.val.iter())
        .enumerate()
    {
        let lambda_i = lambdas
            .get(i)
            .copied()
            .ok_or_else(|| ClosureProofError::WhirP3("lambda count mismatch".into()))?;

        // Claimed sum from public y values.
        for j in 0..core_t {
            let mu_j = mu_core[j];
            if mu_j == neo_math::F::ZERO {
                continue;
            }
            let yj =
                me.y.get(j)
                    .ok_or_else(|| ClosureProofError::WhirP3("ME y missing core entry".into()))?;
            if yj.len() < d {
                return Err(ClosureProofError::WhirP3("ME y row too short".into()));
            }
            let mut dot = neo_math::F::ZERO;
            for rho in 0..d {
                dot += nu[rho] * mix_k_to_f(yj[rho], delta_k);
            }
            claimed_me += lambda_i * mu_j * dot;
        }

        for col_id in 0..bus_cols {
            let mu = mu_bus[col_id];
            if mu == neo_math::F::ZERO {
                continue;
            }
            let yj =
                me.y.get(core_t + col_id)
                    .ok_or_else(|| ClosureProofError::WhirP3("ME y missing bus entry".into()))?;
            if yj.len() < d {
                return Err(ClosureProofError::WhirP3("ME y bus row too short".into()));
            }
            let mut dot = neo_math::F::ZERO;
            for rho in 0..d {
                dot += nu[rho] * mix_k_to_f(yj[rho], delta_k);
            }
            claimed_me += lambda_i * mu * dot;
        }
    }

    Ok((gamma_me, claimed_me))
}
