use super::*;
use neo_ccs::Mat;
use neo_memory::riscv::exec_table::RiscvPoseidonSidecarTable;
use p3_field::PrimeField64;
use p3_goldilocks::{Goldilocks, MATRIX_DIAG_8_GOLDILOCKS};
use p3_poseidon2::{
    matmul_internal, mds_light_permutation, poseidon2_round_numbers_128, ExternalLayerConstants, MDSMat4,
};
use rand::distr::StandardUniform;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::{Mutex, OnceLock};

const POSEIDON_WIDTH: usize = neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
const POSEIDON_LOCAL_ROW_BITS: usize = 5;
const POSEIDON_LOCAL_ROWS_PER_SLOT: usize = 1usize << POSEIDON_LOCAL_ROW_BITS; // 32

pub(crate) const POSEIDON_LOCAL_ROUND_RESIDUAL_COUNT: usize = 11;
pub(crate) const POSEIDON_LOCAL_TRANSITION_RESIDUAL_COUNT: usize = POSEIDON_WIDTH;
pub(crate) const POSEIDON_LOCAL_LINK_RESIDUAL_COUNT: usize =
    12 + (2 * POSEIDON_WIDTH) + (POSEIDON_LOCAL_ROWS_PER_SLOT - 1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PoseidonLocalTraceLayout {
    pub row_active: usize,
    pub has_round: usize,
    pub is_row_start: usize,
    pub slot: usize,
    pub call_ctr: usize,
    pub cycle_call_ctr: usize,
    pub cycle_selected_perm: usize,
    pub cycle_selected_in_start: usize,
    pub cycle_selected_out_start: usize,
    pub state_in_start: usize,
    pub state_out_start: usize,
    pub link_u_local: usize,
}

impl PoseidonLocalTraceLayout {
    pub fn new() -> Self {
        let mut col = 0usize;
        let row_active = col;
        col += 1;
        let has_round = col;
        col += 1;
        let is_row_start = col;
        col += 1;
        let slot = col;
        col += 1;
        let call_ctr = col;
        col += 1;
        let cycle_call_ctr = col;
        col += 1;
        let cycle_selected_perm = col;
        col += 1;
        let cycle_selected_in_start = col;
        col += POSEIDON_WIDTH;
        let cycle_selected_out_start = col;
        col += POSEIDON_WIDTH;
        let state_in_start = col;
        col += POSEIDON_WIDTH;
        let state_out_start = col;
        col += POSEIDON_WIDTH;
        let link_u_local = col;
        Self {
            row_active,
            has_round,
            is_row_start,
            slot,
            call_ctr,
            cycle_call_ctr,
            cycle_selected_perm,
            cycle_selected_in_start,
            cycle_selected_out_start,
            state_in_start,
            state_out_start,
            link_u_local,
        }
    }

    #[inline]
    pub fn cycle_selected_in(&self, i: usize) -> usize {
        debug_assert!(i < POSEIDON_WIDTH);
        self.cycle_selected_in_start + i
    }

    #[inline]
    pub fn cycle_selected_out(&self, i: usize) -> usize {
        debug_assert!(i < POSEIDON_WIDTH);
        self.cycle_selected_out_start + i
    }

    #[inline]
    pub fn state_in(&self, i: usize) -> usize {
        debug_assert!(i < POSEIDON_WIDTH);
        self.state_in_start + i
    }

    #[inline]
    pub fn state_out(&self, i: usize) -> usize {
        debug_assert!(i < POSEIDON_WIDTH);
        self.state_out_start + i
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.link_u_local + 1
    }
}

#[inline]
pub(crate) fn poseidon_local_open_col_ids(layout: &PoseidonLocalTraceLayout) -> Vec<usize> {
    // Explicit helper so prover and verifier share one deterministic column set.
    (0..layout.cols()).collect()
}

#[inline]
pub(crate) fn poseidon_local_round_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5032_4C52_4F55_4E44u64)
}

#[inline]
pub(crate) fn poseidon_local_transition_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5032_4C54_5241_4E53u64)
}

#[inline]
pub(crate) fn poseidon_local_link_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5032_4C4C_494E_4B21u64)
}

#[inline]
pub(crate) fn poseidon_local_round_residuals(
    row_active: K,
    has_round: K,
    is_step_mds: K,
    is_step_external: K,
    is_step_internal: K,
    is_step_no_round: K,
    state_in: [K; POSEIDON_WIDTH],
    state_out: [K; POSEIDON_WIDTH],
    external_rc: [K; POSEIDON_WIDTH],
    internal_rc: K,
) -> [K; POSEIDON_LOCAL_ROUND_RESIDUAL_COUNT] {
    let mut mds_out = state_in;
    mds_light_permutation_8_k(&mut mds_out);

    let mut external_out = state_in;
    for i in 0..POSEIDON_WIDTH {
        external_out[i] = sbox7_k(external_out[i] + external_rc[i]);
    }
    mds_light_permutation_8_k(&mut external_out);

    let mut internal_out = state_in;
    internal_out[0] = sbox7_k(internal_out[0] + internal_rc);
    matmul_internal_8_k(&mut internal_out);

    let mut residuals = [K::ZERO; POSEIDON_LOCAL_ROUND_RESIDUAL_COUNT];
    residuals[0] = row_active * w2_bool01(row_active);
    residuals[1] = row_active * has_round * (has_round - K::ONE);
    residuals[2] = row_active * (has_round - (is_step_mds + is_step_external + is_step_internal));
    for i in 0..POSEIDON_WIDTH {
        let expected = is_step_mds * mds_out[i]
            + is_step_external * external_out[i]
            + is_step_internal * internal_out[i]
            + is_step_no_round * state_in[i];
        residuals[3 + i] = row_active * (state_out[i] - expected);
    }
    residuals
}

#[inline]
fn sbox7_k(x: K) -> K {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x2 * x
}

#[inline]
fn apply_mat4_k(x: &mut [K; 4]) {
    let t01 = x[0] + x[1];
    let t23 = x[2] + x[3];
    let t0123 = t01 + t23;
    let t01123 = t0123 + x[1];
    let t01233 = t0123 + x[3];
    x[3] = t01233 + (x[0] + x[0]); // 2*x0 + ...
    x[1] = t01123 + (x[2] + x[2]); // 2*x2 + ...
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

#[inline]
fn mds_light_permutation_8_k(state: &mut [K; POSEIDON_WIDTH]) {
    debug_assert_eq!(POSEIDON_WIDTH, 8);
    let mut c0 = [state[0], state[1], state[2], state[3]];
    let mut c1 = [state[4], state[5], state[6], state[7]];
    apply_mat4_k(&mut c0);
    apply_mat4_k(&mut c1);
    state[0] = c0[0];
    state[1] = c0[1];
    state[2] = c0[2];
    state[3] = c0[3];
    state[4] = c1[0];
    state[5] = c1[1];
    state[6] = c1[2];
    state[7] = c1[3];

    let sums = [
        state[0] + state[4],
        state[1] + state[5],
        state[2] + state[6],
        state[3] + state[7],
    ];
    for i in 0..POSEIDON_WIDTH {
        state[i] += sums[i % 4];
    }
}

#[inline]
fn matmul_internal_8_k(state: &mut [K; POSEIDON_WIDTH]) {
    debug_assert_eq!(POSEIDON_WIDTH, 8);
    let mut sum = K::ZERO;
    for i in 0..POSEIDON_WIDTH {
        sum += state[i];
    }
    for i in 0..POSEIDON_WIDTH {
        let diag = K::from(F::from_u64(MATRIX_DIAG_8_GOLDILOCKS[i].as_canonical_u64()));
        state[i] = state[i] * diag + sum;
    }
}

#[inline]
pub(crate) fn poseidon_local_transition_residuals(
    row_active: K,
    has_round: K,
    state_in: [K; POSEIDON_WIDTH],
    state_out: [K; POSEIDON_WIDTH],
) -> [K; POSEIDON_LOCAL_TRANSITION_RESIDUAL_COUNT] {
    core::array::from_fn(|i| row_active * (K::ONE - has_round) * (state_out[i] - state_in[i]))
}

#[inline]
pub(crate) fn poseidon_local_link_residuals(
    row_active: K,
    has_round: K,
    is_row_start: K,
    slot: K,
    call_ctr: K,
    cycle_call_ctr: K,
    cycle_selected_perm: K,
    cycle_selected_in: [K; POSEIDON_WIDTH],
    cycle_selected_out: [K; POSEIDON_WIDTH],
    state_in: [K; POSEIDON_WIDTH],
    state_out: [K; POSEIDON_WIDTH],
    step_sel: [K; POSEIDON_LOCAL_ROWS_PER_SLOT],
    step_chi_inv: [K; POSEIDON_LOCAL_ROWS_PER_SLOT],
    limb_mix: [K; POSEIDON_WIDTH],
) -> [K; POSEIDON_LOCAL_LINK_RESIDUAL_COUNT] {
    let is_row_end = step_sel[POSEIDON_LOCAL_ROWS_PER_SLOT - 1];
    let slot_bool = slot * (slot - K::ONE);
    let cycle_selected_perm_bool = cycle_selected_perm * (cycle_selected_perm - K::ONE);
    let mut inactive_sum = has_round + is_row_start + slot + call_ctr + cycle_call_ctr + cycle_selected_perm;
    for i in 0..POSEIDON_WIDTH {
        inactive_sum += state_in[i] + state_out[i] + cycle_selected_in[i] + cycle_selected_out[i];
    }
    let mut residuals = [K::ZERO; POSEIDON_LOCAL_LINK_RESIDUAL_COUNT];
    residuals[0] = row_active * (row_active - K::ONE);
    residuals[1] = has_round * (has_round - K::ONE);
    residuals[2] = is_row_start * (is_row_start - K::ONE);
    residuals[3] = slot_bool;
    residuals[4] = cycle_selected_perm_bool;
    residuals[5] = row_active - cycle_selected_perm;
    residuals[6] = row_active * (call_ctr - cycle_call_ctr);
    residuals[7] = row_active * (is_row_start - step_sel[0]);
    residuals[8] = row_active * ((K::ONE - has_round) - is_row_end);
    residuals[9] = row_active * is_row_start * is_row_end;
    residuals[10] = row_active * is_row_end * has_round;
    residuals[11] = (K::ONE - row_active) * inactive_sum;

    // Boundary linkage: selected permutation endpoints must match local row0 / row31.
    for i in 0..POSEIDON_WIDTH {
        residuals[12 + i] = row_active * step_sel[0] * (state_in[i] - cycle_selected_in[i]);
        residuals[12 + POSEIDON_WIDTH + i] = row_active * is_row_end * (state_out[i] - cycle_selected_out[i]);
    }

    // In-block chain linkage (compressed):
    //   for each r in [0, 30], enforce a random-linear combination over limbs
    //   of state_out(r) - state_in(r+1) at the anchored row selectors.
    let mut idx = 12 + (2 * POSEIDON_WIDTH);
    for r in 0..(POSEIDON_LOCAL_ROWS_PER_SLOT - 1) {
        let sel_r = step_sel[r];
        let sel_next = step_sel[r + 1];
        let scale_r = step_chi_inv[r];
        let scale_next = step_chi_inv[r + 1];
        let mut combined = K::ZERO;
        for i in 0..POSEIDON_WIDTH {
            combined += limb_mix[i] * (sel_r * scale_r * state_out[i] - sel_next * scale_next * state_in[i]);
        }
        residuals[idx] = row_active * combined;
        idx += 1;
    }
    residuals
}

fn build_poseidon_local_sparse_cols(
    local_z: &Mat<F>,
    t_local: usize,
    ell_local: usize,
    layout: &PoseidonLocalTraceLayout,
) -> Result<BTreeMap<usize, SparseIdxVec<K>>, PiCcsError> {
    let m = local_z.cols();
    let cols = layout.cols();
    let expected_m = cols
        .checked_mul(t_local)
        .ok_or_else(|| PiCcsError::InvalidInput("poseidon local cols * t_local overflow".into()))?;
    if m < expected_m {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon local matrix shape mismatch (m={}, expected at least {})",
            m, expected_m
        )));
    }
    if local_z.rows() != neo_math::D {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon local matrix row mismatch (rows={}, expected D={})",
            local_z.rows(),
            neo_math::D
        )));
    }
    let domain_len = 1usize
        .checked_shl(ell_local as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("poseidon local: 2^ell_local overflow".into()))?;
    if t_local > domain_len {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon local matrix shape mismatch (t_local={}, domain_len=2^ell_local={})",
            t_local, domain_len
        )));
    }

    let row0 = local_z.row(0);
    let mut by_col = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for col in 0..cols {
        let base = col * t_local;
        let end = base + t_local;
        let mut entries: Vec<(usize, K)> = Vec::new();
        for (j, &v_f) in row0[base..end].iter().enumerate() {
            let v = K::from(v_f);
            if v != K::ZERO {
                entries.push((j, v));
            }
        }
        by_col.insert(col, SparseIdxVec::from_entries(domain_len, entries));
    }
    Ok(by_col)
}

pub(crate) type PoseidonLocalClaims = (
    Option<(Box<dyn RoundOracle + Send>, K)>,
    Option<(Box<dyn RoundOracle + Send>, K)>,
    Option<(Box<dyn RoundOracle + Send>, K)>,
);

#[derive(Clone)]
struct PoseidonRoundPublicSparseCols {
    is_step_mds: SparseIdxVec<K>,
    is_step_external: SparseIdxVec<K>,
    is_step_internal: SparseIdxVec<K>,
    is_step_no_round: SparseIdxVec<K>,
    step_sel: [SparseIdxVec<K>; POSEIDON_LOCAL_ROWS_PER_SLOT],
    external_rc: [SparseIdxVec<K>; POSEIDON_WIDTH],
    internal_rc: SparseIdxVec<K>,
}

#[derive(Clone, Copy)]
pub(crate) struct PoseidonRoundPublicEval {
    pub is_step_mds: K,
    pub is_step_external: K,
    pub is_step_internal: K,
    pub is_step_no_round: K,
    pub external_rc: [K; POSEIDON_WIDTH],
    pub internal_rc: K,
}

#[inline]
fn poseidon_step_selector(bits: &[K; POSEIDON_LOCAL_ROW_BITS], step: usize) -> K {
    let mut acc = K::ONE;
    for (bit_idx, bit) in bits.iter().enumerate() {
        if ((step >> bit_idx) & 1) == 1 {
            acc *= *bit;
        } else {
            acc *= K::ONE - *bit;
        }
    }
    acc
}

pub(crate) fn poseidon_step_selectors_from_point(r_point: &[K]) -> [K; POSEIDON_LOCAL_ROWS_PER_SLOT] {
    let mut bits = [K::ZERO; POSEIDON_LOCAL_ROW_BITS];
    for i in 0..POSEIDON_LOCAL_ROW_BITS {
        if i < r_point.len() {
            bits[i] = r_point[i];
        }
    }
    core::array::from_fn(|step| poseidon_step_selector(&bits, step))
}

pub(crate) fn poseidon_step_selector_inv_weights_from_anchor(
    r_local_anchor: &[K],
) -> Result<[K; POSEIDON_LOCAL_ROWS_PER_SLOT], PiCcsError> {
    let chi = poseidon_step_selectors_from_point(r_local_anchor);
    let mut out = [K::ZERO; POSEIDON_LOCAL_ROWS_PER_SLOT];
    for (idx, v) in chi.iter().enumerate() {
        if *v == K::ZERO {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon local: anchor row selector is zero at step {}",
                idx
            )));
        }
        out[idx] = v.inverse();
    }
    Ok(out)
}

#[inline]
fn goldilocks_to_k(x: Goldilocks) -> K {
    K::from(F::from_u64(x.as_canonical_u64()))
}

fn poseidon_round_public_eval_from_bits(
    bits: &[K; POSEIDON_LOCAL_ROW_BITS],
    params: &PoseidonRoundParams,
) -> PoseidonRoundPublicEval {
    let mut selectors = [K::ZERO; POSEIDON_LOCAL_ROWS_PER_SLOT];
    for step in 0..POSEIDON_LOCAL_ROWS_PER_SLOT {
        selectors[step] = poseidon_step_selector(bits, step);
    }

    let half_f = params.external_initial.len();
    let internal_len = params.internal.len();
    let terminal_start = 1 + half_f + internal_len;
    let mut external_rc = [K::ZERO; POSEIDON_WIDTH];
    let mut internal_rc = K::ZERO;
    let mut is_step_external = K::ZERO;
    let mut is_step_internal = K::ZERO;
    for step in 1..(1 + half_f) {
        let sel = selectors[step];
        is_step_external += sel;
        let rc = &params.external_initial[step - 1];
        for i in 0..POSEIDON_WIDTH {
            external_rc[i] += sel * goldilocks_to_k(rc[i]);
        }
    }
    for step in (1 + half_f)..(1 + half_f + internal_len) {
        let sel = selectors[step];
        is_step_internal += sel;
        internal_rc += sel * goldilocks_to_k(params.internal[step - (1 + half_f)]);
    }
    for step in terminal_start..(terminal_start + half_f) {
        let sel = selectors[step];
        is_step_external += sel;
        let rc = &params.external_terminal[step - terminal_start];
        for i in 0..POSEIDON_WIDTH {
            external_rc[i] += sel * goldilocks_to_k(rc[i]);
        }
    }

    PoseidonRoundPublicEval {
        is_step_mds: selectors[0],
        is_step_external,
        is_step_internal,
        is_step_no_round: selectors[POSEIDON_LOCAL_ROWS_PER_SLOT - 1],
        external_rc,
        internal_rc,
    }
}

fn build_poseidon_round_public_sparse_cols_uncached(
    ell_local: usize,
    params: &PoseidonRoundParams,
) -> Result<PoseidonRoundPublicSparseCols, PiCcsError> {
    let domain_len = 1usize
        .checked_shl(ell_local as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("poseidon local: 2^ell_local overflow".into()))?;
    let mut is_step_mds_vals = vec![K::ZERO; domain_len];
    let mut is_step_external_vals = vec![K::ZERO; domain_len];
    let mut is_step_internal_vals = vec![K::ZERO; domain_len];
    let mut is_step_no_round_vals = vec![K::ZERO; domain_len];
    let mut step_sel_vals: [Vec<K>; POSEIDON_LOCAL_ROWS_PER_SLOT] = core::array::from_fn(|_| vec![K::ZERO; domain_len]);
    let mut external_rc_vals: [Vec<K>; POSEIDON_WIDTH] = core::array::from_fn(|_| vec![K::ZERO; domain_len]);
    let mut internal_rc_vals = vec![K::ZERO; domain_len];

    let half_f = params.external_initial.len();
    let internal_len = params.internal.len();
    let terminal_start = 1 + half_f + internal_len;
    for j in 0..domain_len {
        let step = j & (POSEIDON_LOCAL_ROWS_PER_SLOT - 1);
        step_sel_vals[step][j] = K::ONE;
        if step == 0 {
            is_step_mds_vals[j] = K::ONE;
            continue;
        }
        if step < 1 + half_f {
            is_step_external_vals[j] = K::ONE;
            let rc = &params.external_initial[step - 1];
            for i in 0..POSEIDON_WIDTH {
                external_rc_vals[i][j] = goldilocks_to_k(rc[i]);
            }
            continue;
        }
        if step < 1 + half_f + internal_len {
            is_step_internal_vals[j] = K::ONE;
            internal_rc_vals[j] = goldilocks_to_k(params.internal[step - (1 + half_f)]);
            continue;
        }
        if step < terminal_start + half_f {
            is_step_external_vals[j] = K::ONE;
            let rc = &params.external_terminal[step - terminal_start];
            for i in 0..POSEIDON_WIDTH {
                external_rc_vals[i][j] = goldilocks_to_k(rc[i]);
            }
            continue;
        }
        is_step_no_round_vals[j] = K::ONE;
    }

    let mut external_rc_sparse: Vec<SparseIdxVec<K>> = Vec::with_capacity(POSEIDON_WIDTH);
    for vals in external_rc_vals.iter() {
        external_rc_sparse.push(sparse_trace_col_from_values(/*m_in=*/ 0, ell_local, vals)?);
    }
    let external_rc: [SparseIdxVec<K>; POSEIDON_WIDTH] = external_rc_sparse
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("poseidon local: external_rc sparse conversion failed".into()))?;
    let mut step_sel_sparse: Vec<SparseIdxVec<K>> = Vec::with_capacity(POSEIDON_LOCAL_ROWS_PER_SLOT);
    for vals in step_sel_vals.iter() {
        step_sel_sparse.push(sparse_trace_col_from_values(/*m_in=*/ 0, ell_local, vals)?);
    }
    let step_sel: [SparseIdxVec<K>; POSEIDON_LOCAL_ROWS_PER_SLOT] = step_sel_sparse
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("poseidon local: step_sel sparse conversion failed".into()))?;

    Ok(PoseidonRoundPublicSparseCols {
        is_step_mds: sparse_trace_col_from_values(/*m_in=*/ 0, ell_local, &is_step_mds_vals)?,
        is_step_external: sparse_trace_col_from_values(/*m_in=*/ 0, ell_local, &is_step_external_vals)?,
        is_step_internal: sparse_trace_col_from_values(/*m_in=*/ 0, ell_local, &is_step_internal_vals)?,
        is_step_no_round: sparse_trace_col_from_values(/*m_in=*/ 0, ell_local, &is_step_no_round_vals)?,
        step_sel,
        external_rc,
        internal_rc: sparse_trace_col_from_values(/*m_in=*/ 0, ell_local, &internal_rc_vals)?,
    })
}

static POSEIDON_ROUND_PUBLIC_SPARSE_CACHE: OnceLock<Mutex<BTreeMap<usize, PoseidonRoundPublicSparseCols>>> =
    OnceLock::new();

fn build_poseidon_round_public_sparse_cols(
    ell_local: usize,
    params: &PoseidonRoundParams,
) -> Result<PoseidonRoundPublicSparseCols, PiCcsError> {
    let cache = POSEIDON_ROUND_PUBLIC_SPARSE_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(found) = cache
        .lock()
        .map_err(|_| PiCcsError::ProtocolError("poseidon local: sparse public cache lock poisoned".into()))?
        .get(&ell_local)
        .cloned()
    {
        return Ok(found);
    }
    let built = build_poseidon_round_public_sparse_cols_uncached(ell_local, params)?;
    cache
        .lock()
        .map_err(|_| PiCcsError::ProtocolError("poseidon local: sparse public cache lock poisoned".into()))?
        .insert(ell_local, built.clone());
    Ok(built)
}

pub(crate) fn poseidon_round_public_eval_at_point(r_point: &[K]) -> Result<PoseidonRoundPublicEval, PiCcsError> {
    let params = poseidon_round_params()?;
    let mut bits = [K::ZERO; POSEIDON_LOCAL_ROW_BITS];
    for i in 0..POSEIDON_LOCAL_ROW_BITS {
        if i < r_point.len() {
            bits[i] = r_point[i];
        }
    }
    Ok(poseidon_round_public_eval_from_bits(&bits, params))
}

pub(crate) fn build_route_a_poseidon_local_claims(
    local_z: &Mat<F>,
    t_local: usize,
    ell_local: usize,
    layout: &PoseidonLocalTraceLayout,
    r_local_anchor: &[K],
) -> Result<PoseidonLocalClaims, PiCcsError> {
    let col_sparse_map = build_poseidon_local_sparse_cols(local_z, t_local, ell_local, layout)?;
    let col_sparse = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        col_sparse_map
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("poseidon local missing col={col_id}")))
    };
    let round_params = poseidon_round_params()?;
    let round_public = build_poseidon_round_public_sparse_cols(ell_local, round_params)?;

    let round_weights = poseidon_local_round_weight_vector(r_local_anchor, POSEIDON_LOCAL_ROUND_RESIDUAL_COUNT);
    let mut round_sparse: Vec<SparseIdxVec<K>> = Vec::with_capacity(2 + 2 * POSEIDON_WIDTH + 4 + POSEIDON_WIDTH + 1);
    round_sparse.push(col_sparse(layout.row_active)?);
    round_sparse.push(col_sparse(layout.has_round)?);
    for i in 0..POSEIDON_WIDTH {
        round_sparse.push(col_sparse(layout.state_in(i))?);
    }
    for i in 0..POSEIDON_WIDTH {
        round_sparse.push(col_sparse(layout.state_out(i))?);
    }
    round_sparse.push(round_public.is_step_mds.clone());
    round_sparse.push(round_public.is_step_external.clone());
    round_sparse.push(round_public.is_step_internal.clone());
    round_sparse.push(round_public.is_step_no_round.clone());
    for i in 0..POSEIDON_WIDTH {
        round_sparse.push(round_public.external_rc[i].clone());
    }
    round_sparse.push(round_public.internal_rc.clone());
    let round_oracle = FormulaOracleSparseTime::new(
        round_sparse,
        10,
        r_local_anchor,
        Box::new(move |vals: &[K]| {
            let mut state_in = [K::ZERO; POSEIDON_WIDTH];
            let mut state_out = [K::ZERO; POSEIDON_WIDTH];
            let mut external_rc = [K::ZERO; POSEIDON_WIDTH];
            state_in.copy_from_slice(&vals[2..2 + POSEIDON_WIDTH]);
            state_out.copy_from_slice(&vals[2 + POSEIDON_WIDTH..2 + 2 * POSEIDON_WIDTH]);
            external_rc.copy_from_slice(&vals[22..22 + POSEIDON_WIDTH]);
            let residuals = poseidon_local_round_residuals(
                vals[0],
                vals[1],
                vals[18],
                vals[19],
                vals[20],
                vals[21],
                state_in,
                state_out,
                external_rc,
                vals[30],
            );
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(round_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    let mut transition_sparse: Vec<SparseIdxVec<K>> = Vec::with_capacity(2 + 2 * POSEIDON_WIDTH);
    transition_sparse.push(col_sparse(layout.row_active)?);
    transition_sparse.push(col_sparse(layout.has_round)?);
    for i in 0..POSEIDON_WIDTH {
        transition_sparse.push(col_sparse(layout.state_in(i))?);
    }
    for i in 0..POSEIDON_WIDTH {
        transition_sparse.push(col_sparse(layout.state_out(i))?);
    }
    let transition_weights =
        poseidon_local_transition_weight_vector(r_local_anchor, POSEIDON_LOCAL_TRANSITION_RESIDUAL_COUNT);
    let transition_oracle = FormulaOracleSparseTime::new(
        transition_sparse,
        4,
        r_local_anchor,
        Box::new(move |vals: &[K]| {
            let mut state_in = [K::ZERO; POSEIDON_WIDTH];
            let mut state_out = [K::ZERO; POSEIDON_WIDTH];
            state_in.copy_from_slice(&vals[2..2 + POSEIDON_WIDTH]);
            state_out.copy_from_slice(&vals[2 + POSEIDON_WIDTH..2 + 2 * POSEIDON_WIDTH]);
            let residuals = poseidon_local_transition_residuals(vals[0], vals[1], state_in, state_out);
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(transition_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    let mut link_sparse: Vec<SparseIdxVec<K>> = Vec::with_capacity(9 + 4 * POSEIDON_WIDTH);
    link_sparse.push(col_sparse(layout.row_active)?);
    link_sparse.push(col_sparse(layout.has_round)?);
    link_sparse.push(col_sparse(layout.is_row_start)?);
    link_sparse.push(col_sparse(layout.slot)?);
    link_sparse.push(col_sparse(layout.call_ctr)?);
    link_sparse.push(col_sparse(layout.cycle_call_ctr)?);
    link_sparse.push(col_sparse(layout.cycle_selected_perm)?);
    for i in 0..POSEIDON_WIDTH {
        link_sparse.push(col_sparse(layout.cycle_selected_in(i))?);
    }
    for i in 0..POSEIDON_WIDTH {
        link_sparse.push(col_sparse(layout.cycle_selected_out(i))?);
    }
    for i in 0..POSEIDON_WIDTH {
        link_sparse.push(col_sparse(layout.state_in(i))?);
    }
    for i in 0..POSEIDON_WIDTH {
        link_sparse.push(col_sparse(layout.state_out(i))?);
    }
    for step in 0..POSEIDON_LOCAL_ROWS_PER_SLOT {
        link_sparse.push(round_public.step_sel[step].clone());
    }
    let step_chi_inv = poseidon_step_selector_inv_weights_from_anchor(r_local_anchor)?;
    let limb_mix_vec = bitness_weights(r_local_anchor, POSEIDON_WIDTH, 0x5032_4C4C_494D_4258u64);
    let limb_mix: [K; POSEIDON_WIDTH] = limb_mix_vec
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("poseidon local: limb mix conversion failed".into()))?;
    let link_weights = poseidon_local_link_weight_vector(r_local_anchor, POSEIDON_LOCAL_LINK_RESIDUAL_COUNT);
    let link_oracle = FormulaOracleSparseTime::new(
        link_sparse,
        8,
        r_local_anchor,
        Box::new(move |vals: &[K]| {
            let mut cycle_selected_in = [K::ZERO; POSEIDON_WIDTH];
            let mut cycle_selected_out = [K::ZERO; POSEIDON_WIDTH];
            let mut state_in = [K::ZERO; POSEIDON_WIDTH];
            let mut state_out = [K::ZERO; POSEIDON_WIDTH];
            let mut step_sel = [K::ZERO; POSEIDON_LOCAL_ROWS_PER_SLOT];
            cycle_selected_in.copy_from_slice(&vals[7..7 + POSEIDON_WIDTH]);
            cycle_selected_out.copy_from_slice(&vals[7 + POSEIDON_WIDTH..7 + 2 * POSEIDON_WIDTH]);
            state_in.copy_from_slice(&vals[7 + 2 * POSEIDON_WIDTH..7 + 3 * POSEIDON_WIDTH]);
            state_out.copy_from_slice(&vals[7 + 3 * POSEIDON_WIDTH..7 + 4 * POSEIDON_WIDTH]);
            step_sel
                .copy_from_slice(&vals[7 + 4 * POSEIDON_WIDTH..7 + 4 * POSEIDON_WIDTH + POSEIDON_LOCAL_ROWS_PER_SLOT]);
            let residuals = poseidon_local_link_residuals(
                vals[0],
                vals[1],
                vals[2],
                vals[3],
                vals[4],
                vals[5],
                vals[6],
                cycle_selected_in,
                cycle_selected_out,
                state_in,
                state_out,
                step_sel,
                step_chi_inv,
                limb_mix,
            );
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(link_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    Ok((
        Some((Box::new(round_oracle), K::ZERO)),
        Some((Box::new(transition_oracle), K::ZERO)),
        Some((Box::new(link_oracle), K::ZERO)),
    ))
}

#[derive(Clone, Debug)]
struct PoseidonRoundParams {
    external_initial: Vec<[Goldilocks; POSEIDON_WIDTH]>,
    internal: Vec<Goldilocks>,
    external_terminal: Vec<[Goldilocks; POSEIDON_WIDTH]>,
}

static POSEIDON_ROUND_PARAMS: OnceLock<PoseidonRoundParams> = OnceLock::new();

fn poseidon_round_params() -> Result<&'static PoseidonRoundParams, PiCcsError> {
    if let Some(params) = POSEIDON_ROUND_PARAMS.get() {
        return Ok(params);
    }
    let mut rng = ChaCha8Rng::from_seed(neo_ccs::crypto::poseidon2_goldilocks::SEED);
    let (rounds_f, rounds_p) =
        poseidon2_round_numbers_128::<Goldilocks>(POSEIDON_WIDTH, /*sbox_degree=*/ 7).map_err(|e| {
            PiCcsError::ProtocolError(format!("poseidon local trace: failed to derive rounds_f/rounds_p: {e}"))
        })?;
    let external = ExternalLayerConstants::<Goldilocks, POSEIDON_WIDTH>::new_from_rng(rounds_f, &mut rng);
    let internal = rng
        .sample_iter(StandardUniform)
        .take(rounds_p)
        .collect::<Vec<Goldilocks>>();
    let params = PoseidonRoundParams {
        external_initial: external.get_initial_constants().clone(),
        internal,
        external_terminal: external.get_terminal_constants().clone(),
    };
    let _ = POSEIDON_ROUND_PARAMS.set(params);
    POSEIDON_ROUND_PARAMS
        .get()
        .ok_or_else(|| PiCcsError::ProtocolError("poseidon local trace: round param init failed".into()))
}

#[inline]
fn sbox7(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x2 * x
}

#[inline]
fn apply_external_round(state: &mut [Goldilocks; POSEIDON_WIDTH], rc: &[Goldilocks; POSEIDON_WIDTH]) {
    for i in 0..POSEIDON_WIDTH {
        state[i] = sbox7(state[i] + rc[i]);
    }
    mds_light_permutation(state, &MDSMat4);
}

#[inline]
fn apply_internal_round(state: &mut [Goldilocks; POSEIDON_WIDTH], rc: Goldilocks) {
    state[0] = sbox7(state[0] + rc);
    matmul_internal(state, MATRIX_DIAG_8_GOLDILOCKS);
}

fn apply_round_step(
    state: &mut [Goldilocks; POSEIDON_WIDTH],
    step_idx: usize,
    params: &PoseidonRoundParams,
) -> Result<(), PiCcsError> {
    let half_f = params.external_initial.len();
    if step_idx == 0 {
        mds_light_permutation(state, &MDSMat4);
        return Ok(());
    }

    let mut idx = step_idx - 1;
    if idx < half_f {
        apply_external_round(state, &params.external_initial[idx]);
        return Ok(());
    }
    idx -= half_f;
    if idx < params.internal.len() {
        apply_internal_round(state, params.internal[idx]);
        return Ok(());
    }
    idx -= params.internal.len();
    if idx < params.external_terminal.len() {
        apply_external_round(state, &params.external_terminal[idx]);
        return Ok(());
    }

    Err(PiCcsError::ProtocolError(format!(
        "poseidon local trace: invalid round step index {} (half_f={}, rounds_p={})",
        step_idx,
        half_f,
        params.internal.len()
    )))
}

#[inline]
fn state_u64_to_goldilocks(words: &[u64; POSEIDON_WIDTH]) -> [Goldilocks; POSEIDON_WIDTH] {
    let mut out = [Goldilocks::ZERO; POSEIDON_WIDTH];
    for (i, &w) in words.iter().enumerate() {
        out[i] = Goldilocks::from_u64(w);
    }
    out
}

pub(crate) fn build_poseidon_local_trace_matrix(
    sidecar: &RiscvPoseidonSidecarTable,
) -> Result<(Mat<F>, usize, usize, PoseidonLocalTraceLayout), PiCcsError> {
    let active_rows = sidecar
        .perm_rows
        .len()
        .checked_mul(POSEIDON_LOCAL_ROWS_PER_SLOT)
        .ok_or_else(|| PiCcsError::InvalidInput("poseidon local trace: active row count overflow".into()))?;
    let t_local = active_rows.max(1).next_power_of_two();
    let layout = PoseidonLocalTraceLayout::new();
    let cols = layout.cols();
    let m = cols
        .checked_mul(t_local)
        .ok_or_else(|| PiCcsError::InvalidInput("poseidon local trace: cols * t_local overflow".into()))?;
    let mut data = vec![F::ZERO; neo_math::D * m];
    let set = |data: &mut [F], col: usize, row: usize, v: F| {
        data[col * t_local + row] = v;
    };

    let round_params = poseidon_round_params()?;
    let transition_steps = 1usize
        .checked_add(round_params.external_initial.len())
        .and_then(|v| v.checked_add(round_params.internal.len()))
        .and_then(|v| v.checked_add(round_params.external_terminal.len()))
        .ok_or_else(|| PiCcsError::InvalidInput("poseidon local trace: transition step count overflow".into()))?;
    if transition_steps + 1 != POSEIDON_LOCAL_ROWS_PER_SLOT {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon local trace: unexpected transition_steps={} (expected {} for 32-row slot)",
            transition_steps,
            POSEIDON_LOCAL_ROWS_PER_SLOT - 1
        )));
    }

    let mut cycle_row_by_cycle = BTreeMap::<u64, &neo_memory::riscv::exec_table::RiscvPoseidonCycleEventRow>::new();
    for row in sidecar.cycle_rows.iter() {
        if cycle_row_by_cycle.insert(row.cycle, row).is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon local trace: duplicate cycle row at cycle={}",
                row.cycle
            )));
        }
    }
    let mut perm_by_cycle_slot =
        BTreeMap::<(u64, u8), &neo_memory::riscv::exec_table::RiscvPoseidonPermSlotMetaRow>::new();
    for perm in sidecar.perm_rows.iter() {
        let key = (perm.cycle, perm.slot);
        if perm_by_cycle_slot.insert(key, perm).is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon local trace: duplicate perm row at cycle={} slot={}",
                perm.cycle, perm.slot
            )));
        }
    }

    for (perm_idx, perm) in sidecar.perm_rows.iter().enumerate() {
        if perm.slot > 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon local trace: invalid slot={} at cycle={}",
                perm.slot, perm.cycle
            )));
        }
        let base = perm_idx
            .checked_mul(POSEIDON_LOCAL_ROWS_PER_SLOT)
            .ok_or_else(|| PiCcsError::InvalidInput("poseidon local trace: local row base overflow".into()))?;
        if base + (POSEIDON_LOCAL_ROWS_PER_SLOT - 1) >= t_local {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon local trace: permutation row base out of range (perm_idx={}, cycle={}, slot={}, base={}, t_local={})",
                perm_idx, perm.cycle, perm.slot, base, t_local
            )));
        }

        let cycle_row = cycle_row_by_cycle.get(&perm.cycle).ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "poseidon local trace: missing cycle row for perm cycle={}",
                perm.cycle
            ))
        })?;
        let (cycle_selected_perm, cycle_selected_in, cycle_selected_out) = if perm.slot == 0 {
            let s0 = perm_by_cycle_slot
                .get(&(perm.cycle, 0))
                .copied()
                .ok_or_else(|| {
                    PiCcsError::ProtocolError(format!(
                        "poseidon local trace: missing slot0 perm row for cycle={}",
                        perm.cycle
                    ))
                })?;
            (
                if cycle_row.do_perm_slot0 { 1u64 } else { 0u64 },
                s0.state_in,
                s0.state_out,
            )
        } else {
            let s1 = perm_by_cycle_slot
                .get(&(perm.cycle, 1))
                .copied()
                .ok_or_else(|| {
                    PiCcsError::ProtocolError(format!(
                        "poseidon local trace: missing slot1 perm row for cycle={}",
                        perm.cycle
                    ))
                })?;
            (
                if cycle_row.do_perm_slot1 { 1u64 } else { 0u64 },
                s1.state_in,
                s1.state_out,
            )
        };

        let mut states = [[Goldilocks::ZERO; POSEIDON_WIDTH]; POSEIDON_LOCAL_ROWS_PER_SLOT];
        states[0] = state_u64_to_goldilocks(&perm.state_in);
        for r in 0..(POSEIDON_LOCAL_ROWS_PER_SLOT - 1) {
            let mut next = states[r];
            apply_round_step(&mut next, r, round_params)?;
            states[r + 1] = next;
        }
        for i in 0..POSEIDON_WIDTH {
            let expected = states[POSEIDON_LOCAL_ROWS_PER_SLOT - 1][i].as_canonical_u64();
            if expected != perm.state_out[i] {
                return Err(PiCcsError::ProtocolError(format!(
                    "poseidon local trace: permutation output mismatch at cycle={}, slot={}, limb={} (got {}, expected {})",
                    perm.cycle, perm.slot, i, perm.state_out[i], expected
                )));
            }
        }

        for r in 0..POSEIDON_LOCAL_ROWS_PER_SLOT {
            let j = base + r;
            if data[layout.row_active * t_local + j] != F::ZERO {
                return Err(PiCcsError::ProtocolError(format!(
                    "poseidon local trace: duplicate active local row at idx={} (cycle={}, slot={})",
                    j, perm.cycle, perm.slot
                )));
            }
            set(&mut data, layout.row_active, j, F::ONE);
            set(
                &mut data,
                layout.has_round,
                j,
                if r + 1 < POSEIDON_LOCAL_ROWS_PER_SLOT {
                    F::ONE
                } else {
                    F::ZERO
                },
            );
            set(&mut data, layout.is_row_start, j, if r == 0 { F::ONE } else { F::ZERO });
            set(&mut data, layout.slot, j, F::from_u64(perm.slot as u64));
            set(&mut data, layout.call_ctr, j, F::from_u64(perm.call_ctr));
            set(&mut data, layout.cycle_call_ctr, j, F::from_u64(cycle_row.call_ctr));
            set(
                &mut data,
                layout.cycle_selected_perm,
                j,
                F::from_u64(cycle_selected_perm),
            );
            for i in 0..POSEIDON_WIDTH {
                set(
                    &mut data,
                    layout.cycle_selected_in(i),
                    j,
                    F::from_u64(cycle_selected_in[i]),
                );
                set(
                    &mut data,
                    layout.cycle_selected_out(i),
                    j,
                    F::from_u64(cycle_selected_out[i]),
                );
            }
            for i in 0..POSEIDON_WIDTH {
                set(
                    &mut data,
                    layout.state_in(i),
                    j,
                    F::from_u64(states[r][i].as_canonical_u64()),
                );
                let out_state = if r + 1 < POSEIDON_LOCAL_ROWS_PER_SLOT {
                    states[r + 1][i]
                } else {
                    states[r][i]
                };
                set(
                    &mut data,
                    layout.state_out(i),
                    j,
                    F::from_u64(out_state.as_canonical_u64()),
                );
            }
        }
    }

    let z = Mat::from_row_major(neo_math::D, m, data);
    Ok((z, /*m_in=*/ 0, t_local, layout))
}
