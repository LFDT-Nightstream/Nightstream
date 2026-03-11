use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use neo_math::{from_complex, Fq, KExtensions, D, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;

#[repr(C)]
pub struct DeviceRequest {
    pub api: u32,
    pub device_id: u32,
}

#[repr(C)]
pub struct DeviceResponse {
    pub status: i32,
    pub available: i32,
}

#[repr(C)]
pub struct SessionRequest {
    pub api: u32,
    pub device_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlatK {
    pub re: u64,
    pub im: u64,
}

#[repr(C)]
pub struct FoldRequest {
    pub session: usize,
    pub evaluator: usize,
    pub snapshot_ptr: *mut u8,
    pub snapshot_len: usize,
    pub challenge: FlatK,
}

#[repr(C)]
pub struct FlatFq {
    pub limb: u64,
}

const SPLIT_NC_SNAPSHOT_MAGIC: u64 = 0x4E53_504C_4954_4E43;
const SPLIT_NC_SNAPSHOT_VERSION: u64 = 1;
const SPLIT_NC_FE_ROW_V1: u64 = 1;
const SPLIT_NC_NC_COL_V1: u64 = 2;

static NEXT_EVALUATOR_HANDLE: AtomicUsize = AtomicUsize::new(2);
static FE_EVALS_AT_CALLS: AtomicUsize = AtomicUsize::new(0);
static NC_EVALS_AT_CALLS: AtomicUsize = AtomicUsize::new(0);
static POSEIDON2_BATCH_CALLS: AtomicUsize = AtomicUsize::new(0);
static SESSION_OPEN_CALLS: AtomicUsize = AtomicUsize::new(0);
static EVALUATORS: OnceLock<Mutex<HashMap<usize, EvaluatorState>>> = OnceLock::new();

#[derive(Clone)]
struct FeSnapshotTerm {
    coeff: K,
    vars: Vec<(usize, u32)>,
}

#[derive(Clone)]
struct FeSnapshotState {
    cur_len: usize,
    eq_beta_r_tbl: Vec<K>,
    eq_r_inputs_tbl: Option<Vec<K>>,
    gamma_pow_mcs: Vec<K>,
    f_terms: Vec<FeSnapshotTerm>,
    f_var_tables_by_mcs: Vec<Vec<Vec<K>>>,
    eval_tbl: Option<Vec<K>>,
    gamma_to_k: K,
}

#[derive(Clone)]
struct NcSnapshotState {
    cur_len: usize,
    eq_beta_m_tbl: Vec<K>,
    digits_tables: Vec<Vec<[K; D]>>,
    weights: Vec<[K; D]>,
    range_t_sq: Vec<K>,
}

#[derive(Clone)]
enum EvaluatorState {
    Fe(FeSnapshotState),
    Nc(NcSnapshotState),
    Passthrough,
}

struct WordReader {
    words: Vec<u64>,
    idx: usize,
}

impl WordReader {
    fn new(bytes: &[u8]) -> Result<Self, i32> {
        if bytes.is_empty() {
            return Ok(Self {
                words: Vec::new(),
                idx: 0,
            });
        }
        if !bytes.len().is_multiple_of(8) {
            return Err(-11);
        }
        let mut words = Vec::with_capacity(bytes.len() / 8);
        for chunk in bytes.chunks_exact(8) {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            words.push(u64::from_le_bytes(word));
        }
        Ok(Self { words, idx: 0 })
    }

    fn remaining(&self) -> usize {
        self.words.len().saturating_sub(self.idx)
    }

    fn read_u64(&mut self) -> Result<u64, i32> {
        let word = *self.words.get(self.idx).ok_or(-12)?;
        self.idx += 1;
        Ok(word)
    }

    fn read_usize(&mut self) -> Result<usize, i32> {
        usize::try_from(self.read_u64()?).map_err(|_| -13)
    }

    fn read_flat_k(&mut self) -> Result<K, i32> {
        Ok(from_complex(
            Fq::from_u64(self.read_u64()?),
            Fq::from_u64(self.read_u64()?),
        ))
    }

    fn read_k_vec(&mut self, len: usize) -> Result<Vec<K>, i32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_flat_k()?);
        }
        Ok(out)
    }
}

fn evaluators() -> &'static Mutex<HashMap<usize, EvaluatorState>> {
    EVALUATORS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn flat_to_k(x: FlatK) -> K {
    from_complex(Fq::from_u64(x.re), Fq::from_u64(x.im))
}

fn k_to_flat(x: K) -> FlatK {
    let (re, im) = x.to_limbs_u64();
    FlatK { re, im }
}

fn fold_table_inplace(table: &mut Vec<K>, r: K) {
    let half = table.len() / 2;
    for i in 0..half {
        let lo = table[2 * i];
        let hi = table[2 * i + 1];
        table[i] = lo + (hi - lo) * r;
    }
    table.truncate(half);
}

fn fold_digits_table_inplace(table: &mut Vec<[K; D]>, r: K) {
    let half = table.len() / 2;
    for i in 0..half {
        let base = 2 * i;
        for rho in 0..D {
            let lo = table[base][rho];
            let hi = table[base + 1][rho];
            table[i][rho] = lo + (hi - lo) * r;
        }
    }
    table.truncate(half);
}

fn range_product_cached(y: K, range_t_sq: &[K]) -> K {
    if range_t_sq.is_empty() {
        return y;
    }
    let y2 = y * y;
    let mut prod = y;
    for &tt2 in range_t_sq {
        prod *= y2 - tt2;
    }
    prod
}

fn parse_fe_snapshot(bytes: &[u8]) -> Result<Option<FeSnapshotState>, i32> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(8) || bytes.len() < 24 {
        return Ok(None);
    }

    let mut reader = WordReader::new(bytes)?;
    let magic = reader.read_u64()?;
    if magic != SPLIT_NC_SNAPSHOT_MAGIC {
        return Ok(None);
    }
    let version = reader.read_u64()?;
    let kind = reader.read_u64()?;
    if version != SPLIT_NC_SNAPSHOT_VERSION || kind != SPLIT_NC_FE_ROW_V1 {
        return Err(-14);
    }

    let _b = reader.read_u64()?;
    let _d_sc = reader.read_u64()?;
    let cur_len = reader.read_usize()?;
    let eq_beta_len = reader.read_usize()?;
    let eq_r_inputs_len = reader.read_usize()?;
    let gamma_pow_len = reader.read_usize()?;
    let term_len = reader.read_usize()?;
    let num_mcs = reader.read_usize()?;
    let num_vars = reader.read_usize()?;
    let table_len = reader.read_usize()?;
    let eval_tbl_len = reader.read_usize()?;
    let gamma_to_k = reader.read_flat_k()?;

    let eq_beta_r_tbl = reader.read_k_vec(eq_beta_len)?;
    let eq_r_inputs_tbl = if eq_r_inputs_len == 0 {
        None
    } else {
        Some(reader.read_k_vec(eq_r_inputs_len)?)
    };
    let gamma_pow_mcs = reader.read_k_vec(gamma_pow_len)?;

    let mut f_terms = Vec::with_capacity(term_len);
    for _ in 0..term_len {
        let coeff = reader.read_flat_k()?;
        let vars_len = reader.read_usize()?;
        let mut vars = Vec::with_capacity(vars_len);
        for _ in 0..vars_len {
            vars.push((reader.read_usize()?, u32::try_from(reader.read_u64()?).map_err(|_| -15)?));
        }
        f_terms.push(FeSnapshotTerm { coeff, vars });
    }

    let mut f_var_tables_by_mcs = Vec::with_capacity(num_mcs);
    for _ in 0..num_mcs {
        let mut per_mcs = Vec::with_capacity(num_vars);
        for _ in 0..num_vars {
            per_mcs.push(reader.read_k_vec(table_len)?);
        }
        f_var_tables_by_mcs.push(per_mcs);
    }

    let eval_tbl = if eval_tbl_len == 0 {
        None
    } else {
        Some(reader.read_k_vec(eval_tbl_len)?)
    };

    if reader.remaining() != 0 {
        return Err(-16);
    }

    Ok(Some(FeSnapshotState {
        cur_len,
        eq_beta_r_tbl,
        eq_r_inputs_tbl,
        gamma_pow_mcs,
        f_terms,
        f_var_tables_by_mcs,
        eval_tbl,
        gamma_to_k,
    }))
}

fn parse_nc_snapshot(bytes: &[u8]) -> Result<Option<NcSnapshotState>, i32> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(8) || bytes.len() < 24 {
        return Ok(None);
    }

    let mut reader = WordReader::new(bytes)?;
    let magic = reader.read_u64()?;
    if magic != SPLIT_NC_SNAPSHOT_MAGIC {
        return Ok(None);
    }
    let version = reader.read_u64()?;
    let kind = reader.read_u64()?;
    if version != SPLIT_NC_SNAPSHOT_VERSION || kind != SPLIT_NC_NC_COL_V1 {
        return Err(-21);
    }

    let _b = reader.read_u64()?;
    let _d_sc = reader.read_u64()?;
    let cur_len = reader.read_usize()?;
    let eq_beta_len = reader.read_usize()?;
    let num_tables = reader.read_usize()?;
    let table_len = reader.read_usize()?;
    let d_width = reader.read_usize()?;
    let weights_tables = reader.read_usize()?;
    let weights_width = reader.read_usize()?;
    let range_len = reader.read_usize()?;

    if (num_tables > 0 && d_width != D) || (weights_tables > 0 && weights_width != D) {
        return Err(-22);
    }

    let eq_beta_m_tbl = reader.read_k_vec(eq_beta_len)?;

    let mut digits_tables = Vec::with_capacity(num_tables);
    for _ in 0..num_tables {
        let mut table = Vec::with_capacity(table_len);
        for _ in 0..table_len {
            let mut entry = [K::ZERO; D];
            for slot in &mut entry {
                *slot = reader.read_flat_k()?;
            }
            table.push(entry);
        }
        digits_tables.push(table);
    }

    let mut weights = Vec::with_capacity(weights_tables);
    for _ in 0..weights_tables {
        let mut entry = [K::ZERO; D];
        for slot in &mut entry {
            *slot = reader.read_flat_k()?;
        }
        weights.push(entry);
    }

    let range_t_sq = reader.read_k_vec(range_len)?;
    if reader.remaining() != 0 {
        return Err(-23);
    }

    Ok(Some(NcSnapshotState {
        cur_len,
        eq_beta_m_tbl,
        digits_tables,
        weights,
        range_t_sq,
    }))
}

fn fe_evals_generic(state: &FeSnapshotState, points: &[FlatK], out: &mut [FlatK]) {
    let tail_len = state.cur_len / 2;
    let f_arity = state
        .f_var_tables_by_mcs
        .first()
        .map(Vec::len)
        .unwrap_or(0);

    for (idx, point) in points.iter().copied().enumerate() {
        let x = flat_to_k(point);
        let one_minus = K::ONE - x;
        let mut var_vals = vec![K::ZERO; f_arity];
        let mut sum_x = K::ZERO;

        for t in 0..tail_len {
            let table_idx = 2 * t;
            let eq_beta_r = one_minus * state.eq_beta_r_tbl[table_idx] + x * state.eq_beta_r_tbl[table_idx + 1];
            let mut f_prime = K::ZERO;

            for (mcs_idx, per_mcs_tables) in state.f_var_tables_by_mcs.iter().enumerate() {
                for (pos, tbl) in per_mcs_tables.iter().enumerate() {
                    var_vals[pos] = one_minus * tbl[table_idx] + x * tbl[table_idx + 1];
                }

                let mut f_i = K::ZERO;
                for term in &state.f_terms {
                    let mut acc = term.coeff;
                    for &(var_pos, exp) in &term.vars {
                        let xi = var_vals[var_pos];
                        let mut power = xi;
                        for _ in 1..exp {
                            power *= xi;
                        }
                        acc *= power;
                    }
                    f_i += acc;
                }

                let gamma = state.gamma_pow_mcs.get(mcs_idx).copied().unwrap_or(K::ONE);
                f_prime += gamma * f_i;
            }

            let mut acc = eq_beta_r * f_prime;
            if let (Some(eq_tbl), Some(eval_tbl)) = (&state.eq_r_inputs_tbl, &state.eval_tbl) {
                let eq_r_inputs = one_minus * eq_tbl[table_idx] + x * eq_tbl[table_idx + 1];
                if eq_r_inputs != K::ZERO {
                    let e = one_minus * eval_tbl[table_idx] + x * eval_tbl[table_idx + 1];
                    acc += eq_r_inputs * (state.gamma_to_k * e);
                }
            }
            sum_x += acc;
        }

        out[idx] = k_to_flat(sum_x);
    }
}

fn nc_evals_generic(state: &NcSnapshotState, points: &[FlatK], out: &mut [FlatK]) {
    let tail_len = state.cur_len / 2;
    for (idx, point) in points.iter().copied().enumerate() {
        let x = flat_to_k(point);
        let mut acc = K::ZERO;

        for t in 0..tail_len {
            let table_idx = 2 * t;
            let e0 = state.eq_beta_m_tbl[table_idx];
            let e1 = state.eq_beta_m_tbl[table_idx + 1] - e0;
            let eq_beta = e0 + e1 * x;
            let mut nc_sum = K::ZERO;

            for (table, weights) in state.digits_tables.iter().zip(state.weights.iter()) {
                let lo = &table[table_idx];
                let hi = &table[table_idx + 1];
                for rho in 0..D {
                    let y0 = lo[rho];
                    let dy = hi[rho] - y0;
                    let y = y0 + dy * x;
                    nc_sum += weights[rho] * range_product_cached(y, &state.range_t_sq);
                }
            }

            acc += eq_beta * nc_sum;
        }

        out[idx] = k_to_flat(acc);
    }
}

fn create_handle(state: EvaluatorState) -> usize {
    let handle = NEXT_EVALUATOR_HANDLE.fetch_add(1, Ordering::Relaxed);
    evaluators().lock().expect("mock evaluator registry").insert(handle, state);
    handle
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_abi_version() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_device_probe(_req: *const DeviceRequest, out: *mut DeviceResponse) -> i32 {
    unsafe {
        if let Some(out) = out.as_mut() {
            out.status = 0;
            out.available = 1;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_session_open(req: *const SessionRequest, out_handle: *mut usize) -> i32 {
    SESSION_OPEN_CALLS.fetch_add(1, Ordering::Relaxed);
    unsafe {
        let Some(req) = req.as_ref() else {
            return -1;
        };
        let Some(out_handle) = out_handle.as_mut() else {
            return -2;
        };
        *out_handle = ((req.api as usize) << 32) | (req.device_id as usize + 1);
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_session_close(_session: usize) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_fe_create(
    _session: u64,
    snapshot_words: *mut u64,
    snapshot_len: u64,
    out_handle: *mut usize,
) -> i32 {
    unsafe {
        let Some(out_handle) = out_handle.as_mut() else {
            return -2;
        };
        let snapshot = std::slice::from_raw_parts(snapshot_words.cast::<u8>(), snapshot_len as usize);
        let state = match parse_fe_snapshot(snapshot) {
            Ok(Some(state)) => EvaluatorState::Fe(state),
            Ok(None) => EvaluatorState::Passthrough,
            Err(status) => return status,
        };
        *out_handle = create_handle(state);
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_fe_destroy(_session: usize, evaluator: usize) -> i32 {
    evaluators().lock().expect("mock evaluator registry").remove(&evaluator);
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_fe_evals_at(
    _session: u64,
    evaluator: u64,
    _snapshot_words: *mut u64,
    _snapshot_len: u64,
    points_words: *mut u64,
    points_len: u64,
    out_ptr: *mut u64,
    out_len: usize,
) -> i32 {
    FE_EVALS_AT_CALLS.fetch_add(1, Ordering::Relaxed);
    unsafe {
        let n = (points_len as usize).min(out_len);
        if n == 0 {
            return 0;
        }
        let points = std::slice::from_raw_parts(points_words.cast::<FlatK>(), n);
        let out = std::slice::from_raw_parts_mut(out_ptr.cast::<FlatK>(), n);
        let evaluators = evaluators().lock().expect("mock evaluator registry");
        let Some(state) = evaluators.get(&(evaluator as usize)) else {
            return -3;
        };
        match state {
            EvaluatorState::Fe(state) => fe_evals_generic(state, points, out),
            EvaluatorState::Passthrough => out.copy_from_slice(points),
            EvaluatorState::Nc(_) => return -4,
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_fe_fold(
    _session: usize,
    evaluator: usize,
    challenge_re: u64,
    challenge_im: u64,
) -> i32 {
    unsafe {
        let mut evaluators = evaluators().lock().expect("mock evaluator registry");
        let Some(state) = evaluators.get_mut(&evaluator) else {
            return -3;
        };
        match state {
            EvaluatorState::Fe(state) => {
                let challenge = flat_to_k(FlatK {
                    re: challenge_re,
                    im: challenge_im,
                });
                fold_table_inplace(&mut state.eq_beta_r_tbl, challenge);
                if let Some(eq_tbl) = state.eq_r_inputs_tbl.as_mut() {
                    fold_table_inplace(eq_tbl, challenge);
                }
                for per_mcs in &mut state.f_var_tables_by_mcs {
                    for table in per_mcs {
                        fold_table_inplace(table, challenge);
                    }
                }
                if let Some(eval_tbl) = state.eval_tbl.as_mut() {
                    fold_table_inplace(eval_tbl, challenge);
                }
                state.cur_len /= 2;
            }
            EvaluatorState::Passthrough => {}
            EvaluatorState::Nc(_) => return -4,
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_nc_create(
    _session: u64,
    snapshot_words: *mut u64,
    snapshot_len: u64,
    out_handle: *mut usize,
) -> i32 {
    unsafe {
        let Some(out_handle) = out_handle.as_mut() else {
            return -2;
        };
        let snapshot = std::slice::from_raw_parts(snapshot_words.cast::<u8>(), snapshot_len as usize);
        let state = match parse_nc_snapshot(snapshot) {
            Ok(Some(state)) => EvaluatorState::Nc(state),
            Ok(None) => EvaluatorState::Passthrough,
            Err(status) => return status,
        };
        *out_handle = create_handle(state);
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_nc_destroy(_session: usize, evaluator: usize) -> i32 {
    evaluators().lock().expect("mock evaluator registry").remove(&evaluator);
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_nc_evals_at(
    _session: u64,
    evaluator: u64,
    _snapshot_words: *mut u64,
    _snapshot_len: u64,
    points_words: *mut u64,
    points_len: u64,
    out_ptr: *mut u64,
    out_len: usize,
) -> i32 {
    NC_EVALS_AT_CALLS.fetch_add(1, Ordering::Relaxed);
    unsafe {
        let n = (points_len as usize).min(out_len);
        if n == 0 {
            return 0;
        }
        let points = std::slice::from_raw_parts(points_words.cast::<FlatK>(), n);
        let out = std::slice::from_raw_parts_mut(out_ptr.cast::<FlatK>(), n);
        let evaluators = evaluators().lock().expect("mock evaluator registry");
        let Some(state) = evaluators.get(&(evaluator as usize)) else {
            return -3;
        };
        match state {
            EvaluatorState::Nc(state) => nc_evals_generic(state, points, out),
            EvaluatorState::Passthrough => out.copy_from_slice(points),
            EvaluatorState::Fe(_) => return -4,
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_nc_fold(
    _session: usize,
    evaluator: usize,
    challenge_re: u64,
    challenge_im: u64,
) -> i32 {
    unsafe {
        let mut evaluators = evaluators().lock().expect("mock evaluator registry");
        let Some(state) = evaluators.get_mut(&evaluator) else {
            return -3;
        };
        match state {
            EvaluatorState::Nc(state) => {
                let challenge = flat_to_k(FlatK {
                    re: challenge_re,
                    im: challenge_im,
                });
                fold_table_inplace(&mut state.eq_beta_m_tbl, challenge);
                for table in &mut state.digits_tables {
                    fold_digits_table_inplace(table, challenge);
                }
                state.cur_len /= 2;
            }
            EvaluatorState::Passthrough => {}
            EvaluatorState::Fe(_) => return -4,
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_debug_snapshot_head(
    session: u64,
    snapshot_words: *mut u64,
    snapshot_len: u64,
    out_words: *mut u64,
    out_len: u32,
) -> i32 {
    unsafe {
        if out_words.is_null() {
            return -2;
        }
        *out_words.add(0) = session;
        *out_words.add(1) = snapshot_words as usize as u64;
        *out_words.add(2) = snapshot_len;
        let snapshot_head = std::slice::from_raw_parts(snapshot_words, snapshot_len as usize / 8);
        for (idx, word) in snapshot_head
            .iter()
            .copied()
            .take(out_len as usize - 3)
            .enumerate()
        {
            *out_words.add(idx + 3) = word;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_poseidon2_permute_u64x8(
    _session: usize,
    state_ptr: *mut FlatFq,
    width: u32,
) -> i32 {
    unsafe {
        if width != 8 {
            return -2;
        }
        let Some(_state_ptr) = state_ptr.as_ref() else {
            return -3;
        };
        let state = std::slice::from_raw_parts_mut(state_ptr, width as usize);
        let perm = neo_ccs::crypto::poseidon2_goldilocks::permutation();

        let mut input = [Goldilocks::ZERO; 8];
        for (dst, src) in input.iter_mut().zip(state.iter()) {
            *dst = Goldilocks::from_u64(src.limb);
        }
        let result = perm.permute(input);
        for (dst, src) in state.iter_mut().zip(result.iter()) {
            dst.limb = src.as_canonical_u64();
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_poseidon2_permute_batch_u64x8(
    _session: usize,
    state_ptr: *mut FlatFq,
    num_states: u32,
    width: u32,
) -> i32 {
    POSEIDON2_BATCH_CALLS.fetch_add(1, Ordering::Relaxed);
    unsafe {
        if width != 8 {
            return -2;
        }
        let Some(_state_ptr) = state_ptr.as_ref() else {
            return -3;
        };
        let total_words = width as usize * num_states as usize;
        let state_words = std::slice::from_raw_parts_mut(state_ptr, total_words);
        let perm = neo_ccs::crypto::poseidon2_goldilocks::permutation();

        for state in state_words.chunks_exact_mut(width as usize) {
            let mut input = [Goldilocks::ZERO; 8];
            for (dst, src) in input.iter_mut().zip(state.iter()) {
                *dst = Goldilocks::from_u64(src.limb);
            }
            let result = perm.permute(input);
            for (dst, src) in state.iter_mut().zip(result.iter()) {
                dst.limb = src.as_canonical_u64();
            }
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_test_reset_counters() {
    FE_EVALS_AT_CALLS.store(0, Ordering::Relaxed);
    NC_EVALS_AT_CALLS.store(0, Ordering::Relaxed);
    POSEIDON2_BATCH_CALLS.store(0, Ordering::Relaxed);
    SESSION_OPEN_CALLS.store(0, Ordering::Relaxed);
    evaluators().lock().expect("mock evaluator registry").clear();
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_test_poseidon2_batch_calls() -> usize {
    POSEIDON2_BATCH_CALLS.load(Ordering::Relaxed)
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_test_fe_evals_at_calls() -> usize {
    FE_EVALS_AT_CALLS.load(Ordering::Relaxed)
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_test_nc_evals_at_calls() -> usize {
    NC_EVALS_AT_CALLS.load(Ordering::Relaxed)
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_test_session_open_calls() -> usize {
    SESSION_OPEN_CALLS.load(Ordering::Relaxed)
}
