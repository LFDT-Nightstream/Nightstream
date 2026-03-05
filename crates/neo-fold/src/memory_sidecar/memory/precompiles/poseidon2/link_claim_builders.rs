use super::*;
use neo_ccs::Mat;
use neo_memory::riscv::exec_table::Rv32PoseidonSidecarTable;
use p3_field::{Field, PrimeCharacteristicRing};

const POSEIDON_LINK_ETA_LEN: usize = 19;
const POSEIDON_CONT_ETA_LEN: usize = 13;
const POSEIDON_WIDTH: usize = 8;
const POSEIDON_LOCAL_ROWS_PER_SLOT: usize = 32;

#[derive(Clone, Debug)]
pub struct PoseidonLinkChallenges {
    pub eta: [K; POSEIDON_LINK_ETA_LEN],
    pub beta: K,
}

#[derive(Clone, Debug)]
pub struct PoseidonContinuityChallenges {
    pub eta: [K; POSEIDON_CONT_ETA_LEN],
    pub beta: K,
}

pub(crate) type PoseidonCycleLinkClaims = (
    Option<(Box<dyn RoundOracle + Send>, K)>,
    Option<(Box<dyn RoundOracle + Send>, K)>,
);

pub(crate) type PoseidonLocalLinkClaims = (
    Option<(Box<dyn RoundOracle + Send>, K)>,
    Option<(Box<dyn RoundOracle + Send>, K)>,
);

pub(crate) const POSEIDON_LINK_CYCLE_INV_RESIDUAL_COUNT: usize = 6;
pub(crate) const POSEIDON_LINK_LOCAL_INV_RESIDUAL_COUNT: usize = 3;
pub(crate) const POSEIDON_CONT_CYCLE_INV_RESIDUAL_COUNT: usize = 6;

#[inline]
pub(crate) fn poseidon_link_cycle_inv_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5032_4C43_5949_4E56u64)
}

#[inline]
pub(crate) fn poseidon_link_local_inv_weight_vector(r_local_anchor: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_local_anchor, len, 0x5032_4C4C_494E_5655u64)
}

#[inline]
pub(crate) fn poseidon_cont_cycle_inv_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5032_434F_4E54_494Eu64)
}

#[inline]
pub(crate) fn poseidon_cycle_link_inv_residuals(a0: K, a1: K, z0: K, z1: K, u0: K, u1: K, beta: K) -> [K; 6] {
    [
        a0 * ((beta - z0) * u0 - K::ONE),
        (K::ONE - a0) * u0,
        a0 * (a0 - K::ONE),
        a1 * ((beta - z1) * u1 - K::ONE),
        (K::ONE - a1) * u1,
        a1 * (a1 - K::ONE),
    ]
}

#[inline]
pub(crate) fn poseidon_local_link_inv_residuals(a: K, z: K, u: K, beta: K) -> [K; 3] {
    [a * ((beta - z) * u - K::ONE), (K::ONE - a) * u, a * (a - K::ONE)]
}

#[inline]
pub(crate) fn poseidon_cycle_cont_inv_residuals(
    a_pre: K,
    a_post: K,
    z_pre: K,
    z_post: K,
    u_pre: K,
    u_post: K,
    beta: K,
) -> [K; 6] {
    [
        a_pre * ((beta - z_pre) * u_pre - K::ONE),
        (K::ONE - a_pre) * u_pre,
        a_pre * (a_pre - K::ONE),
        a_post * ((beta - z_post) * u_post - K::ONE),
        (K::ONE - a_post) * u_post,
        a_post * (a_post - K::ONE),
    ]
}

#[inline]
fn k_to_base_field(x: K, ctx: &str) -> Result<F, PiCcsError> {
    let coeffs = x.as_coeffs();
    if coeffs.iter().skip(1).any(|&c| c != F::ZERO) {
        return Err(PiCcsError::ProtocolError(format!(
            "{ctx}: expected base-field value (non-zero extension coeff)"
        )));
    }
    coeffs
        .first()
        .copied()
        .ok_or_else(|| PiCcsError::ProtocolError(format!("{ctx}: missing base coefficient")))
}

#[inline]
pub(crate) fn poseidon_link_compress_tuple(
    slot: K,
    call_ctr: K,
    state_in: [K; POSEIDON_WIDTH],
    state_out: [K; POSEIDON_WIDTH],
    eta: &[K; POSEIDON_LINK_ETA_LEN],
) -> K {
    let mut z = eta[0] + eta[1] * slot + eta[2] * call_ctr;
    for i in 0..POSEIDON_WIDTH {
        z += eta[3 + i] * state_in[i];
    }
    for i in 0..POSEIDON_WIDTH {
        z += eta[11 + i] * state_out[i];
    }
    z
}

#[inline]
pub(crate) fn poseidon_cont_compress_tuple(
    cycle: K,
    call_ctr: K,
    mode_finalized: K,
    cursor: K,
    state: [K; POSEIDON_WIDTH],
    eta: &[K; POSEIDON_CONT_ETA_LEN],
) -> K {
    let mut z = eta[0] + eta[1] * cycle + eta[2] * call_ctr + eta[3] * mode_finalized + eta[4] * cursor;
    for i in 0..POSEIDON_WIDTH {
        z += eta[5 + i] * state[i];
    }
    z
}

#[inline]
fn poseidon_reciprocal(beta: K, z: K, ctx: &str) -> Result<K, PiCcsError> {
    let denom = beta - z;
    if denom == K::ZERO {
        return Err(PiCcsError::ProtocolError(format!(
            "{ctx}: beta equals compressed tuple (pole)"
        )));
    }
    Ok(denom.inverse())
}

pub(crate) fn sample_poseidon_link_challenges(tr: &mut Poseidon2Transcript) -> PoseidonLinkChallenges {
    let eta_vec = ts::sample_base_addr_point(
        tr,
        b"poseidon/link/eta",
        b"poseidon/link/eta_base",
        POSEIDON_LINK_ETA_LEN,
    );
    let mut eta = [K::ZERO; POSEIDON_LINK_ETA_LEN];
    eta.copy_from_slice(&eta_vec);
    let beta = ts::sample_base_addr_point(tr, b"poseidon/link/beta", b"poseidon/link/beta_base", 1)[0];
    PoseidonLinkChallenges { eta, beta }
}

pub(crate) fn sample_poseidon_continuity_challenges(tr: &mut Poseidon2Transcript) -> PoseidonContinuityChallenges {
    let eta_vec = ts::sample_base_addr_point(
        tr,
        b"poseidon/cont/eta",
        b"poseidon/cont/eta_base",
        POSEIDON_CONT_ETA_LEN,
    );
    let mut eta = [K::ZERO; POSEIDON_CONT_ETA_LEN];
    eta.copy_from_slice(&eta_vec);
    let beta = ts::sample_base_addr_point(tr, b"poseidon/cont/beta", b"poseidon/cont/beta_base", 1)[0];
    PoseidonContinuityChallenges { eta, beta }
}

pub(crate) fn populate_poseidon_cycle_link_aux_columns(
    t_len: usize,
    cycle_z: &mut Mat<F>,
    cycle_layout: &PoseidonCycleTraceLayout,
    sidecar: &Rv32PoseidonSidecarTable,
    link_chals: &PoseidonLinkChallenges,
    cont_chals: &PoseidonContinuityChallenges,
) -> Result<(), PiCcsError> {
    let zrow = cycle_z.row_mut(0);
    let mut perm_by_cycle_slot =
        BTreeMap::<(u64, u8), &neo_memory::riscv::exec_table::Rv32PoseidonPermSlotMetaRow>::new();
    for perm in sidecar.perm_rows.iter() {
        let key = (perm.cycle, perm.slot);
        if perm_by_cycle_slot.insert(key, perm).is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon cycle-link aux: duplicate permutation row at cycle={} slot={}",
                perm.cycle, perm.slot
            )));
        }
    }

    for (idx, cycle_row) in sidecar.cycle_rows.iter().enumerate() {
        let j = cycle_row.cycle as usize;
        if j >= t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon cycle-link aux: cycle out of range (cycle={}, t_len={})",
                cycle_row.cycle, t_len
            )));
        }
        if cycle_row.do_perm_slot0 {
            let perm = perm_by_cycle_slot
                .get(&(cycle_row.cycle, 0))
                .copied()
                .ok_or_else(|| {
                    PiCcsError::ProtocolError(format!(
                        "poseidon cycle-link aux: missing slot0 permutation row at cycle={}",
                        cycle_row.cycle
                    ))
                })?;
            let in_state: [K; POSEIDON_WIDTH] = core::array::from_fn(|i| K::from(F::from_u64(perm.state_in[i])));
            let out_state: [K; POSEIDON_WIDTH] = core::array::from_fn(|i| K::from(F::from_u64(perm.state_out[i])));
            let z = poseidon_link_compress_tuple(
                K::ZERO,
                K::from(F::from_u64(cycle_row.call_ctr)),
                in_state,
                out_state,
                &link_chals.eta,
            );
            let u = poseidon_reciprocal(link_chals.beta, z, "poseidon cycle-link aux slot0")?;
            zrow[cycle_layout.link_u_slot0 * t_len + j] = k_to_base_field(u, "poseidon cycle-link aux slot0")?;
        }
        if cycle_row.do_perm_slot1 {
            let perm = perm_by_cycle_slot
                .get(&(cycle_row.cycle, 1))
                .copied()
                .ok_or_else(|| {
                    PiCcsError::ProtocolError(format!(
                        "poseidon cycle-link aux: missing slot1 permutation row at cycle={}",
                        cycle_row.cycle
                    ))
                })?;
            let in_state: [K; POSEIDON_WIDTH] = core::array::from_fn(|i| K::from(F::from_u64(perm.state_in[i])));
            let out_state: [K; POSEIDON_WIDTH] = core::array::from_fn(|i| K::from(F::from_u64(perm.state_out[i])));
            let z = poseidon_link_compress_tuple(
                K::ONE,
                K::from(F::from_u64(cycle_row.call_ctr)),
                in_state,
                out_state,
                &link_chals.eta,
            );
            let u = poseidon_reciprocal(link_chals.beta, z, "poseidon cycle-link aux slot1")?;
            zrow[cycle_layout.link_u_slot1 * t_len + j] = k_to_base_field(u, "poseidon cycle-link aux slot1")?;
        }

        let is_first = idx == 0 || poseidon_cycle_continuity_break_before(cycle_row);
        let is_last =
            idx + 1 == sidecar.cycle_rows.len()
                || poseidon_cycle_continuity_break_before(sidecar.cycle_rows.get(idx + 1).ok_or_else(|| {
                    PiCcsError::ProtocolError("poseidon cycle-link aux next-row lookup failed".into())
                })?);
        let a_pre = if is_first { K::ZERO } else { K::ONE };
        let a_post = if is_last { K::ZERO } else { K::ONE };

        let cycle_pre = K::from(F::from_u64(cycle_row.cycle));
        let cycle_post = cycle_pre + K::ONE;
        let call_pre = K::from(F::from_u64(cycle_row.call_ctr));
        let mode_pre = if cycle_row.mode_finalized { K::ONE } else { K::ZERO };
        let cursor_pre = K::from(F::from_u64(cycle_row.cursor_before as u64));
        let cursor_post = K::from(F::from_u64(cycle_row.cursor_after as u64));
        let op_absorb = if cycle_row.op_absorb { K::ONE } else { K::ZERO };
        let op_finalize = if cycle_row.op_finalize { K::ONE } else { K::ZERO };
        let call_post = call_pre;
        let mode_post = op_finalize + (K::ONE - op_finalize - op_absorb) * mode_pre;
        let state_pre: [K; POSEIDON_WIDTH] = core::array::from_fn(|i| K::from(F::from_u64(cycle_row.state_pre[i])));
        let state_post: [K; POSEIDON_WIDTH] = core::array::from_fn(|i| K::from(F::from_u64(cycle_row.state_post[i])));
        let z_pre = poseidon_cont_compress_tuple(cycle_pre, call_pre, mode_pre, cursor_pre, state_pre, &cont_chals.eta);
        let z_post = poseidon_cont_compress_tuple(
            cycle_post,
            call_post,
            mode_post,
            cursor_post,
            state_post,
            &cont_chals.eta,
        );

        if a_pre != K::ZERO {
            let u_pre = poseidon_reciprocal(cont_chals.beta, z_pre, "poseidon continuity aux pre")?;
            zrow[cycle_layout.cont_u_pre * t_len + j] = k_to_base_field(u_pre, "poseidon continuity aux pre")?;
        }
        if a_post != K::ZERO {
            let u_post = poseidon_reciprocal(cont_chals.beta, z_post, "poseidon continuity aux post")?;
            zrow[cycle_layout.cont_u_post * t_len + j] = k_to_base_field(u_post, "poseidon continuity aux post")?;
        }
    }

    Ok(())
}

pub(crate) fn populate_poseidon_local_link_aux_column(
    t_local: usize,
    local_z: &mut Mat<F>,
    local_layout: &PoseidonLocalTraceLayout,
    sidecar: &Rv32PoseidonSidecarTable,
    link_chals: &PoseidonLinkChallenges,
) -> Result<(), PiCcsError> {
    let zrow = local_z.row_mut(0);
    for (perm_idx, perm) in sidecar.perm_rows.iter().enumerate() {
        let base = perm_idx
            .checked_mul(POSEIDON_LOCAL_ROWS_PER_SLOT)
            .ok_or_else(|| PiCcsError::InvalidInput("poseidon local-link aux: row base overflow".into()))?;
        if base >= t_local {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon local-link aux: row base out of range (perm_idx={}, base={}, t_local={})",
                perm_idx, base, t_local
            )));
        }
        let in_state: [K; POSEIDON_WIDTH] = core::array::from_fn(|i| K::from(F::from_u64(perm.state_in[i])));
        let out_state: [K; POSEIDON_WIDTH] = core::array::from_fn(|i| K::from(F::from_u64(perm.state_out[i])));
        let slot_k = K::from(F::from_u64(perm.slot as u64));
        let call_k = K::from(F::from_u64(perm.call_ctr));
        let z = poseidon_link_compress_tuple(slot_k, call_k, in_state, out_state, &link_chals.eta);
        let u = poseidon_reciprocal(link_chals.beta, z, "poseidon local-link aux")?;
        zrow[local_layout.link_u_local * t_local + base] = k_to_base_field(u, "poseidon local-link aux")?;
    }

    Ok(())
}

fn sparse_col_from_mat(
    z: &Mat<F>,
    t_len: usize,
    ell: usize,
    col: usize,
    m_in: usize,
) -> Result<SparseIdxVec<K>, PiCcsError> {
    let row0 = z.row(0);
    let start = col
        .checked_mul(t_len)
        .ok_or_else(|| PiCcsError::InvalidInput("poseidon link sparse col start overflow".into()))?;
    let end = start
        .checked_add(t_len)
        .ok_or_else(|| PiCcsError::InvalidInput("poseidon link sparse col end overflow".into()))?;
    if end > row0.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon link sparse col out of range (col={}, t_len={}, row_len={})",
            col,
            t_len,
            row0.len()
        )));
    }
    let vals: Vec<K> = row0[start..end].iter().copied().map(K::from).collect();
    sparse_trace_col_from_values(m_in, ell, &vals)
}

pub(crate) fn build_route_a_poseidon_cycle_link_claims(
    cycle_z: &Mat<F>,
    cycle_t_len: usize,
    cycle_m_in: usize,
    ell_n: usize,
    cycle_layout: &PoseidonCycleTraceLayout,
    r_cycle: &[K],
    link_chals: &PoseidonLinkChallenges,
) -> Result<PoseidonCycleLinkClaims, PiCcsError> {
    let a0 = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.do_perm_slot0, cycle_m_in)?;
    let a1 = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.do_perm_slot1, cycle_m_in)?;
    let call_ctr = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.call_ctr, cycle_m_in)?;
    let mut slot0_in_v = Vec::with_capacity(POSEIDON_WIDTH);
    let mut slot0_out_v = Vec::with_capacity(POSEIDON_WIDTH);
    let mut slot1_in_v = Vec::with_capacity(POSEIDON_WIDTH);
    let mut slot1_out_v = Vec::with_capacity(POSEIDON_WIDTH);
    for i in 0..POSEIDON_WIDTH {
        slot0_in_v.push(sparse_col_from_mat(
            cycle_z,
            cycle_t_len,
            ell_n,
            cycle_layout.slot0_in(i),
            cycle_m_in,
        )?);
        slot0_out_v.push(sparse_col_from_mat(
            cycle_z,
            cycle_t_len,
            ell_n,
            cycle_layout.slot0_out(i),
            cycle_m_in,
        )?);
        slot1_in_v.push(sparse_col_from_mat(
            cycle_z,
            cycle_t_len,
            ell_n,
            cycle_layout.slot1_in(i),
            cycle_m_in,
        )?);
        slot1_out_v.push(sparse_col_from_mat(
            cycle_z,
            cycle_t_len,
            ell_n,
            cycle_layout.slot1_out(i),
            cycle_m_in,
        )?);
    }
    let slot0_in: [SparseIdxVec<K>; POSEIDON_WIDTH] = slot0_in_v
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("poseidon cycle-link sparse slot0_in conversion failed".into()))?;
    let slot0_out: [SparseIdxVec<K>; POSEIDON_WIDTH] = slot0_out_v
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("poseidon cycle-link sparse slot0_out conversion failed".into()))?;
    let slot1_in: [SparseIdxVec<K>; POSEIDON_WIDTH] = slot1_in_v
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("poseidon cycle-link sparse slot1_in conversion failed".into()))?;
    let slot1_out: [SparseIdxVec<K>; POSEIDON_WIDTH] = slot1_out_v
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("poseidon cycle-link sparse slot1_out conversion failed".into()))?;
    let u0 = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.link_u_slot0, cycle_m_in)?;
    let u1 = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.link_u_slot1, cycle_m_in)?;

    let inv_weights = poseidon_link_cycle_inv_weight_vector(r_cycle, POSEIDON_LINK_CYCLE_INV_RESIDUAL_COUNT);
    let eta = link_chals.eta;
    let beta = link_chals.beta;
    let inv_oracle = FormulaOracleSparseTime::new(
        {
            let mut cols = Vec::with_capacity(2 + 1 + 4 * POSEIDON_WIDTH + 2);
            cols.push(a0.clone());
            cols.push(a1.clone());
            cols.push(call_ctr.clone());
            for c in slot0_in.iter() {
                cols.push(c.clone());
            }
            for c in slot0_out.iter() {
                cols.push(c.clone());
            }
            for c in slot1_in.iter() {
                cols.push(c.clone());
            }
            for c in slot1_out.iter() {
                cols.push(c.clone());
            }
            cols.push(u0.clone());
            cols.push(u1.clone());
            cols
        },
        4,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let mut slot0_in_vals = [K::ZERO; POSEIDON_WIDTH];
            let mut slot0_out_vals = [K::ZERO; POSEIDON_WIDTH];
            let mut slot1_in_vals = [K::ZERO; POSEIDON_WIDTH];
            let mut slot1_out_vals = [K::ZERO; POSEIDON_WIDTH];
            slot0_in_vals.copy_from_slice(&vals[3..3 + POSEIDON_WIDTH]);
            slot0_out_vals.copy_from_slice(&vals[3 + POSEIDON_WIDTH..3 + 2 * POSEIDON_WIDTH]);
            slot1_in_vals.copy_from_slice(&vals[3 + 2 * POSEIDON_WIDTH..3 + 3 * POSEIDON_WIDTH]);
            slot1_out_vals.copy_from_slice(&vals[3 + 3 * POSEIDON_WIDTH..3 + 4 * POSEIDON_WIDTH]);
            let z0 = poseidon_link_compress_tuple(K::ZERO, vals[2], slot0_in_vals, slot0_out_vals, &eta);
            let z1 = poseidon_link_compress_tuple(K::ONE, vals[2], slot1_in_vals, slot1_out_vals, &eta);
            let residuals = poseidon_cycle_link_inv_residuals(vals[0], vals[1], z0, z1, vals[35], vals[36], beta);
            residuals
                .iter()
                .zip(inv_weights.iter())
                .fold(K::ZERO, |acc, (r, w)| acc + (*w * *r))
        }),
    );

    let sum_oracle = FormulaOracleSparseSum::new(
        vec![a0.clone(), a1.clone(), u0.clone(), u1.clone()],
        3,
        Box::new(move |vals: &[K]| vals[0] * vals[2] + vals[1] * vals[3]),
    );

    let sum_claimed = {
        let row0 = cycle_z.row(0);
        let mut out = K::ZERO;
        for j in 0..cycle_t_len {
            let a0_v = K::from(row0[cycle_layout.do_perm_slot0 * cycle_t_len + j]);
            let a1_v = K::from(row0[cycle_layout.do_perm_slot1 * cycle_t_len + j]);
            let u0_v = K::from(row0[cycle_layout.link_u_slot0 * cycle_t_len + j]);
            let u1_v = K::from(row0[cycle_layout.link_u_slot1 * cycle_t_len + j]);
            out += a0_v * u0_v + a1_v * u1_v;
        }
        out
    };

    Ok((
        Some((Box::new(inv_oracle), K::ZERO)),
        Some((Box::new(sum_oracle), sum_claimed)),
    ))
}

pub(crate) fn build_route_a_poseidon_cycle_continuity_claims(
    cycle_z: &Mat<F>,
    cycle_t_len: usize,
    cycle_m_in: usize,
    ell_n: usize,
    cycle_layout: &PoseidonCycleTraceLayout,
    r_cycle: &[K],
    cont_chals: &PoseidonContinuityChallenges,
) -> Result<PoseidonCycleLinkClaims, PiCcsError> {
    let row_active = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.row_active, cycle_m_in)?;
    let is_first = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.is_first, cycle_m_in)?;
    let is_last = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.is_last, cycle_m_in)?;
    let cycle = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.cycle, cycle_m_in)?;
    let call_pre = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.call_ctr, cycle_m_in)?;
    let mode_pre = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.mode_finalized, cycle_m_in)?;
    let cursor_pre = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.cursor_before, cycle_m_in)?;
    let cursor_post = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.cursor_after, cycle_m_in)?;
    let op_absorb = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.op_absorb, cycle_m_in)?;
    let op_finalize = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.op_finalize, cycle_m_in)?;
    let mut state_pre_v = Vec::with_capacity(POSEIDON_WIDTH);
    let mut state_post_v = Vec::with_capacity(POSEIDON_WIDTH);
    for i in 0..POSEIDON_WIDTH {
        state_pre_v.push(sparse_col_from_mat(
            cycle_z,
            cycle_t_len,
            ell_n,
            cycle_layout.state_pre(i),
            cycle_m_in,
        )?);
        state_post_v.push(sparse_col_from_mat(
            cycle_z,
            cycle_t_len,
            ell_n,
            cycle_layout.state_post(i),
            cycle_m_in,
        )?);
    }
    let state_pre: [SparseIdxVec<K>; POSEIDON_WIDTH] = state_pre_v
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("poseidon cont sparse state_pre conversion failed".into()))?;
    let state_post: [SparseIdxVec<K>; POSEIDON_WIDTH] = state_post_v
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("poseidon cont sparse state_post conversion failed".into()))?;
    let u_pre = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.cont_u_pre, cycle_m_in)?;
    let u_post = sparse_col_from_mat(cycle_z, cycle_t_len, ell_n, cycle_layout.cont_u_post, cycle_m_in)?;

    let inv_weights = poseidon_cont_cycle_inv_weight_vector(r_cycle, POSEIDON_CONT_CYCLE_INV_RESIDUAL_COUNT);
    let eta = cont_chals.eta;
    let beta = cont_chals.beta;
    let inv_oracle = FormulaOracleSparseTime::new(
        {
            let mut cols = Vec::with_capacity(10 + 2 * POSEIDON_WIDTH + 2);
            cols.push(row_active.clone());
            cols.push(is_first.clone());
            cols.push(is_last.clone());
            cols.push(cycle.clone());
            cols.push(call_pre.clone());
            cols.push(mode_pre.clone());
            cols.push(cursor_pre.clone());
            cols.push(cursor_post.clone());
            cols.push(op_absorb.clone());
            cols.push(op_finalize.clone());
            for c in state_pre.iter() {
                cols.push(c.clone());
            }
            for c in state_post.iter() {
                cols.push(c.clone());
            }
            cols.push(u_pre.clone());
            cols.push(u_post.clone());
            cols
        },
        6,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let mut pre_vals = [K::ZERO; POSEIDON_WIDTH];
            let mut post_vals = [K::ZERO; POSEIDON_WIDTH];
            pre_vals.copy_from_slice(&vals[10..10 + POSEIDON_WIDTH]);
            post_vals.copy_from_slice(&vals[10 + POSEIDON_WIDTH..10 + 2 * POSEIDON_WIDTH]);
            let row_active = vals[0];
            let is_first = vals[1];
            let is_last = vals[2];
            let cycle_pre = vals[3];
            let call_pre = vals[4];
            let mode_pre = vals[5];
            let cursor_pre = vals[6];
            let cursor_post = vals[7];
            let op_absorb = vals[8];
            let op_finalize = vals[9];
            let a_pre = row_active * (K::ONE - is_first);
            let a_post = row_active * (K::ONE - is_last);
            let call_post = call_pre;
            let mode_post = op_finalize + (K::ONE - op_finalize - op_absorb) * mode_pre;
            let z_pre = poseidon_cont_compress_tuple(cycle_pre, call_pre, mode_pre, cursor_pre, pre_vals, &eta);
            let z_post =
                poseidon_cont_compress_tuple(cycle_pre + K::ONE, call_post, mode_post, cursor_post, post_vals, &eta);
            let residuals = poseidon_cycle_cont_inv_residuals(
                a_pre,
                a_post,
                z_pre,
                z_post,
                vals[10 + 2 * POSEIDON_WIDTH],
                vals[11 + 2 * POSEIDON_WIDTH],
                beta,
            );
            residuals
                .iter()
                .zip(inv_weights.iter())
                .fold(K::ZERO, |acc, (r, w)| acc + (*w * *r))
        }),
    );

    let sum_oracle = FormulaOracleSparseSum::new(
        vec![
            row_active.clone(),
            is_first.clone(),
            is_last.clone(),
            u_pre.clone(),
            u_post.clone(),
        ],
        3,
        Box::new(move |vals: &[K]| {
            let a_pre = vals[0] * (K::ONE - vals[1]);
            let a_post = vals[0] * (K::ONE - vals[2]);
            a_pre * vals[3] - a_post * vals[4]
        }),
    );

    let sum_from_rows = {
        let row0 = cycle_z.row(0);
        let mut out = K::ZERO;
        for j in 0..cycle_t_len {
            let row_active_v = K::from(row0[cycle_layout.row_active * cycle_t_len + j]);
            let is_first_v = K::from(row0[cycle_layout.is_first * cycle_t_len + j]);
            let is_last_v = K::from(row0[cycle_layout.is_last * cycle_t_len + j]);
            let a_pre = row_active_v * (K::ONE - is_first_v);
            let a_post = row_active_v * (K::ONE - is_last_v);
            let u_pre_v = K::from(row0[cycle_layout.cont_u_pre * cycle_t_len + j]);
            let u_post_v = K::from(row0[cycle_layout.cont_u_post * cycle_t_len + j]);
            out += a_pre * u_pre_v - a_post * u_post_v;
        }
        out
    };
    if sum_from_rows != K::ZERO {
        return Err(PiCcsError::ProtocolError(
            "poseidon continuity sum over cycle rows is non-zero".into(),
        ));
    }

    Ok((
        Some((Box::new(inv_oracle), K::ZERO)),
        Some((Box::new(sum_oracle), K::ZERO)),
    ))
}

pub(crate) fn build_route_a_poseidon_local_link_claims(
    local_z: &Mat<F>,
    local_t_len: usize,
    local_m_in: usize,
    ell_local: usize,
    local_layout: &PoseidonLocalTraceLayout,
    r_local_anchor: &[K],
    link_chals: &PoseidonLinkChallenges,
) -> Result<PoseidonLocalLinkClaims, PiCcsError> {
    let row_active = sparse_col_from_mat(local_z, local_t_len, ell_local, local_layout.row_active, local_m_in)?;
    let is_row_start = sparse_col_from_mat(local_z, local_t_len, ell_local, local_layout.is_row_start, local_m_in)?;
    let slot = sparse_col_from_mat(local_z, local_t_len, ell_local, local_layout.slot, local_m_in)?;
    let call_ctr = sparse_col_from_mat(local_z, local_t_len, ell_local, local_layout.call_ctr, local_m_in)?;
    let mut selected_in_v = Vec::with_capacity(POSEIDON_WIDTH);
    let mut selected_out_v = Vec::with_capacity(POSEIDON_WIDTH);
    for i in 0..POSEIDON_WIDTH {
        selected_in_v.push(sparse_col_from_mat(
            local_z,
            local_t_len,
            ell_local,
            local_layout.cycle_selected_in(i),
            local_m_in,
        )?);
        selected_out_v.push(sparse_col_from_mat(
            local_z,
            local_t_len,
            ell_local,
            local_layout.cycle_selected_out(i),
            local_m_in,
        )?);
    }
    let selected_in: [SparseIdxVec<K>; POSEIDON_WIDTH] = selected_in_v
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("poseidon local-link sparse selected_in conversion failed".into()))?;
    let selected_out: [SparseIdxVec<K>; POSEIDON_WIDTH] = selected_out_v
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("poseidon local-link sparse selected_out conversion failed".into()))?;
    let u_local = sparse_col_from_mat(local_z, local_t_len, ell_local, local_layout.link_u_local, local_m_in)?;

    let inv_weights = poseidon_link_local_inv_weight_vector(r_local_anchor, POSEIDON_LINK_LOCAL_INV_RESIDUAL_COUNT);
    let eta = link_chals.eta;
    let beta = link_chals.beta;
    let inv_oracle = FormulaOracleSparseTime::new(
        {
            let mut cols = Vec::with_capacity(4 + 2 * POSEIDON_WIDTH + 1);
            cols.push(row_active.clone());
            cols.push(is_row_start.clone());
            cols.push(slot.clone());
            cols.push(call_ctr.clone());
            for c in selected_in.iter() {
                cols.push(c.clone());
            }
            for c in selected_out.iter() {
                cols.push(c.clone());
            }
            cols.push(u_local.clone());
            cols
        },
        5,
        r_local_anchor,
        Box::new(move |vals: &[K]| {
            let mut in_vals = [K::ZERO; POSEIDON_WIDTH];
            let mut out_vals = [K::ZERO; POSEIDON_WIDTH];
            in_vals.copy_from_slice(&vals[4..4 + POSEIDON_WIDTH]);
            out_vals.copy_from_slice(&vals[4 + POSEIDON_WIDTH..4 + 2 * POSEIDON_WIDTH]);
            let a = vals[0] * vals[1];
            let z = poseidon_link_compress_tuple(vals[2], vals[3], in_vals, out_vals, &eta);
            let residuals = poseidon_local_link_inv_residuals(a, z, vals[20], beta);
            residuals
                .iter()
                .zip(inv_weights.iter())
                .fold(K::ZERO, |acc, (r, w)| acc + (*w * *r))
        }),
    );

    let sum_oracle = FormulaOracleSparseSum::new(
        vec![row_active.clone(), is_row_start.clone(), u_local.clone()],
        3,
        Box::new(move |vals: &[K]| vals[0] * vals[1] * vals[2]),
    );

    let sum_claimed = {
        let row0 = local_z.row(0);
        let mut out = K::ZERO;
        for j in 0..local_t_len {
            let a = K::from(row0[local_layout.row_active * local_t_len + j])
                * K::from(row0[local_layout.is_row_start * local_t_len + j]);
            let u = K::from(row0[local_layout.link_u_local * local_t_len + j]);
            out += a * u;
        }
        out
    };

    Ok((
        Some((Box::new(inv_oracle), K::ZERO)),
        Some((Box::new(sum_oracle), sum_claimed)),
    ))
}
