//! Π_CCS FE-channel (row-phase) round-evaluation kernel.
//!
//! Contract: bit-identical to `RowStreamState::evals_row_phase`'s b=2
//! coefficient accumulation in neo-reductions. Field sums are
//! order-independent, so the parallel reduction is exact by construction.
//!
//! Thread shape: one thread per (row-pair chunk, output coefficient) with a
//! scalar register accumulator — indexed per-thread arrays land in PTX
//! local memory and throttle the kernel. Every term shape the host accepts
//! (`CompiledPolyTermKind`'s fast kinds) has a closed form for a single
//! coefficient of `f` over affine inputs; other shapes stay on the CPU.
//!
//! Layouts:
//! - `tables`: all row tables in one buffer; table `slot` occupies
//!   K-elements `[slot * stride, slot * stride + cur_len)` as (c0, c1)
//!   words. Slot order and meta live host-side (`src/pi_ccs.rs`).
//! - `header`: [deg_max, num_mcs, num_terms, r_inputs_slot, eval_slot,
//!   gamma_to_k.c0, gamma_to_k.c1, f_at_zero.c0, f_at_zero.c1].
//! - `mcs_meta`: per MCS [gamma.c0, gamma.c1, zero_flag, var_slot_base].
//! - `term_meta`: per term [coeff.c0, coeff.c1, var_off, var_count].
//! - `term_vars`: flattened [var_pos, exponent] pairs.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cooperative_launch, cuda_module, grid, kernel, thread, DisjointSlice};
use cuda_host::EmbeddedModuleError;

use crate::kernels::goldilocks::{Gl, Kx};
use crate::kernels::poseidon2::{permute, P2State, RATE as P2_RATE, ST_WORDS as P2_ST_WORDS, WIDTH as P2_WIDTH};
use crate::kernels::sumcheck_common::SUM_BLOCKS;

/// Slot value meaning "this optional table is absent".
pub const NO_TABLE: u64 = u64::MAX;
/// Row pairs each eval thread walks.
pub const EVAL_CHUNK_PAIRS: usize = 4;
/// Maximum univariate width (degree + 1). The CPU asserts exponents ≤ 8.
pub const MAX_WIDTH: usize = 9;
const FE_COOP_THREADS: u32 = 256;
const FE_COOP_BLOCK_CAP: u32 = 64;

pub use pi_ccs_fe_kernels::LoadedModule as FeKernelModule;

pub fn load_fe_kernels(ctx: &Arc<CudaContext>) -> Result<FeKernelModule, EmbeddedModuleError> {
    pi_ccs_fe_kernels::load(ctx)
}

#[allow(clippy::too_many_arguments)]
pub fn launch_fe_round_partials(
    module: &FeKernelModule,
    stream: &Arc<CudaStream>,
    tables: &DeviceBuffer<u64>,
    header: &DeviceBuffer<u64>,
    mcs_meta: &DeviceBuffer<u64>,
    term_meta: &DeviceBuffer<u64>,
    term_vars: &DeviceBuffer<u64>,
    stride: usize,
    tail_len: usize,
    groups: usize,
    width: usize,
    partials: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let threads = (groups * width) as u32;
    module.fe_round_partials(
        stream,
        fe_round_launch_config(threads),
        tables,
        header,
        mcs_meta,
        term_meta,
        term_vars,
        stride as u32,
        tail_len as u32,
        width as u32,
        partials,
    )
}

fn fe_round_launch_config(threads: u32) -> LaunchConfig {
    const DEFAULT_BLOCK: u32 = 256;
    const UNDERFILLED_BLOCK: u32 = 64;
    const TARGET_SMS: u32 = 128;

    let default_blocks = threads.div_ceil(DEFAULT_BLOCK);
    if default_blocks < TARGET_SMS && threads.div_ceil(UNDERFILLED_BLOCK) > default_blocks {
        return LaunchConfig {
            grid_dim: (threads.div_ceil(UNDERFILLED_BLOCK), 1, 1),
            block_dim: (UNDERFILLED_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
    }
    LaunchConfig::for_num_elems(threads)
}

/// One cooperative kernel owns a full FE row round:
/// coefficient partials, reduction, transcript challenge, and table fold.
#[allow(clippy::too_many_arguments)]
pub fn launch_fe_cooperative_row_round(
    module: &FeKernelModule,
    stream: &Arc<CudaStream>,
    tables: &DeviceBuffer<u64>,
    header: &DeviceBuffer<u64>,
    mcs_meta: &DeviceBuffer<u64>,
    term_meta: &DeviceBuffer<u64>,
    term_vars: &DeviceBuffer<u64>,
    stride: usize,
    tail_len: usize,
    groups: usize,
    width: usize,
    num_tables: usize,
    cur_len: usize,
    partials: &mut DeviceBuffer<u64>,
    sum_scratch: &mut DeviceBuffer<u64>,
    coeffs_out: &mut DeviceBuffer<u64>,
    transcript_state: &mut DeviceBuffer<u64>,
    coeff_log: &mut DeviceBuffer<u64>,
    coeff_log_offset: usize,
    challenges: &mut DeviceBuffer<u64>,
    challenge_offset: usize,
    rc: &DeviceBuffer<u64>,
    folded_tables: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    if width == 0 || cur_len < 2 {
        return Ok(());
    }
    let width_words = width * 2;
    let logical_threads = (groups * width)
        .max(SUM_BLOCKS * width_words)
        .max(num_tables * (cur_len / 2))
        .max(1);
    let blocks = (logical_threads.div_ceil(FE_COOP_THREADS as usize) as u32).clamp(1, FE_COOP_BLOCK_CAP);
    module.fe_cooperative_row_round(
        stream,
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (FE_COOP_THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        tables,
        header,
        mcs_meta,
        term_meta,
        term_vars,
        stride as u32,
        tail_len as u32,
        groups as u32,
        width as u32,
        num_tables as u32,
        cur_len as u32,
        partials,
        sum_scratch,
        coeffs_out,
        transcript_state,
        coeff_log,
        coeff_log_offset as u32,
        challenges,
        challenge_offset as u32,
        rc,
        folded_tables,
    )
}

/// One cooperative kernel owns all FE row rounds in a single launch.
#[allow(clippy::too_many_arguments)]
pub fn launch_fe_cooperative_row_rounds(
    module: &FeKernelModule,
    stream: &Arc<CudaStream>,
    tables_a: &mut DeviceBuffer<u64>,
    tables_b: &mut DeviceBuffer<u64>,
    header: &DeviceBuffer<u64>,
    mcs_meta: &DeviceBuffer<u64>,
    term_meta: &DeviceBuffer<u64>,
    term_vars: &DeviceBuffer<u64>,
    stride: usize,
    active_len: usize,
    cur_len: usize,
    front_is_a: bool,
    width: usize,
    num_tables: usize,
    rounds: usize,
    partials: &mut DeviceBuffer<u64>,
    sum_scratch: &mut DeviceBuffer<u64>,
    coeffs_out: &mut DeviceBuffer<u64>,
    transcript_state: &mut DeviceBuffer<u64>,
    coeff_log: &mut DeviceBuffer<u64>,
    challenges: &mut DeviceBuffer<u64>,
    rc: &DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    if width == 0 || rounds == 0 || cur_len < 2 {
        return Ok(());
    }
    let width_words = width * 2;
    let tail_len = active_len.div_ceil(2);
    let groups = tail_len.div_ceil(EVAL_CHUNK_PAIRS).max(1);
    let logical_threads = (groups * width)
        .max(SUM_BLOCKS * width_words)
        .max(num_tables * (cur_len / 2))
        .max(1);
    let blocks = (logical_threads.div_ceil(FE_COOP_THREADS as usize) as u32).clamp(1, FE_COOP_BLOCK_CAP);
    module.fe_cooperative_row_rounds(
        stream,
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (FE_COOP_THREADS, 1, 1),
            shared_mem_bytes: 0,
        },
        tables_a,
        tables_b,
        header,
        mcs_meta,
        term_meta,
        term_vars,
        stride as u32,
        active_len as u32,
        cur_len as u32,
        u32::from(front_is_a),
        width as u32,
        num_tables as u32,
        rounds as u32,
        partials,
        sum_scratch,
        coeffs_out,
        transcript_state,
        coeff_log,
        challenges,
        rc,
    )
}

#[cuda_module]
pub mod pi_ccs_fe_kernels {
    use super::*;

    fn read_k(words: &[u64], k_index: usize) -> Kx {
        Kx::from_words(words[2 * k_index], words[2 * k_index + 1])
    }

    /// One thread per (row-pair chunk, output coefficient `d`): accumulate
    /// `coeffs[d]` over the chunk's pairs in a scalar register — indexed
    /// per-thread arrays land in PTX local memory and throttle the kernel.
    /// Every term shape the host accepts (constant / linear / power ≤ 8 /
    /// two-linear product — `CompiledPolyTermKind`'s fast kinds) has a
    /// closed form for a single coefficient of `f` over affine inputs.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn fe_round_partials(
        tables: &[u64],
        header: &[u64],
        mcs_meta: &[u64],
        term_meta: &[u64],
        term_vars: &[u64],
        stride: u32,
        tail_len: u32,
        width: u32,
        mut partials: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let stride = stride as usize;
        let tail_len = tail_len as usize;
        let width = width as usize;
        if width == 0 {
            return;
        }
        let d = idx % width;
        let group = idx / width;
        let (ok, coeff_d) = fe_round_coeff_for_group(
            tables, header, mcs_meta, term_meta, term_vars, stride, tail_len, group, d,
        );
        if !ok {
            return;
        }

        let words = coeff_d.as_words();
        let out_at = (group * width + d) * 2;
        if out_at + 2 > partials.len() {
            return;
        }
        unsafe {
            *partials.get_unchecked_mut(out_at) = words[0];
            *partials.get_unchecked_mut(out_at + 1) = words[1];
        }
    }

    fn fe_round_coeff_for_group(
        tables: &[u64],
        header: &[u64],
        mcs_meta: &[u64],
        term_meta: &[u64],
        term_vars: &[u64],
        stride: usize,
        tail_len: usize,
        group: usize,
        d: usize,
    ) -> (bool, Kx) {
        let pair_start = group * EVAL_CHUNK_PAIRS;
        if pair_start >= tail_len || header.len() < 9 {
            return (false, Kx::ZERO);
        }
        let pair_end = if pair_start + EVAL_CHUNK_PAIRS < tail_len {
            pair_start + EVAL_CHUNK_PAIRS
        } else {
            tail_len
        };

        let deg_max = header[0] as usize;
        let num_mcs = header[1] as usize;
        let num_terms = header[2] as usize;
        let r_inputs_slot = header[3];
        let eval_slot = header[4];
        let gamma_to_k = Kx::from_words(header[5], header[6]);
        let f_at_zero = Kx::from_words(header[7], header[8]);

        // Coefficient `deg` of `coeff · ∏ (a_v + b_v·X)^{e_v}` for the fast
        // term shapes; the host rejects snapshots with any other shape.
        let inner_coeff_at = |t: usize, var_slot_base: usize, tm_base: usize, deg: usize| -> Kx {
            let coeff = Kx::from_words(term_meta[tm_base], term_meta[tm_base + 1]);
            let var_off = term_meta[tm_base + 2] as usize;
            let var_count = term_meta[tm_base + 3] as usize;
            if var_count == 0 {
                return if deg == 0 { coeff } else { Kx::ZERO };
            }
            let read_var = |v: usize| -> (Kx, Kx, usize) {
                let pair_base = (var_off + v) * 2;
                let var_pos = term_vars[pair_base] as usize;
                let exp = term_vars[pair_base + 1] as usize;
                let tbl_base = (var_slot_base + var_pos) * stride * 2;
                let a = read_k(&tables[tbl_base..], 2 * t);
                let b = read_k(&tables[tbl_base..], 2 * t + 1) - a;
                (a, b, exp)
            };
            if var_count == 1 {
                let (a, b, exp) = read_var(0);
                if deg > exp {
                    return Kx::ZERO;
                }
                // binom(exp, deg) · a^{exp-deg} · b^{deg}
                let mut binom = 1u64;
                for i in 0..deg {
                    binom = binom * ((exp - i) as u64) / ((i + 1) as u64);
                }
                let mut acc = coeff * Kx::from_words(binom, 0);
                for _ in 0..(exp - deg) {
                    acc = acc * a;
                }
                for _ in 0..deg {
                    acc = acc * b;
                }
                return acc;
            }
            // Two linear factors (the host guarantees this shape).
            let (a0, b0, _) = read_var(0);
            let (a1, b1, _) = read_var(1);
            if deg == 0 {
                coeff * (a0 * a1)
            } else if deg == 1 {
                coeff * (a0 * b1 + b0 * a1)
            } else if deg == 2 {
                coeff * (b0 * b1)
            } else {
                Kx::ZERO
            }
        };

        // inner(X)'s coefficient `deg`, summed over MCS slots.
        let inner_at = |t: usize, deg: usize| -> Kx {
            let mut acc = Kx::ZERO;
            for mcs in 0..num_mcs {
                let meta_base = mcs * 4;
                if meta_base + 4 > mcs_meta.len() {
                    return acc;
                }
                let g = Kx::from_words(mcs_meta[meta_base], mcs_meta[meta_base + 1]);
                if g == Kx::ZERO {
                    continue;
                }
                if mcs_meta[meta_base + 2] != 0 {
                    if deg == 0 {
                        acc = acc + f_at_zero * g;
                    }
                    continue;
                }
                let var_slot_base = mcs_meta[meta_base + 3] as usize;
                for term in 0..num_terms {
                    let tm_base = term * 4;
                    if tm_base + 4 > term_meta.len() {
                        return acc;
                    }
                    acc = acc + g * inner_coeff_at(t, var_slot_base, tm_base, deg);
                }
            }
            acc
        };

        let mut coeff_d = Kx::ZERO;
        for t in pair_start..pair_end {
            if 4 * t + 4 > tables.len() {
                return (false, Kx::ZERO);
            }
            let e0 = read_k(tables, 2 * t);
            let e1 = read_k(tables, 2 * t + 1) - e0;

            // coeffs[d] of eq(X) · inner(X).
            coeff_d = coeff_d + e0 * inner_at(t, d);
            if d >= 1 {
                coeff_d = coeff_d + e1 * inner_at(t, d - 1);
            }

            // Eval channel: gamma_to_k · eq_r_inputs(X) · eval_tbl(X).
            if r_inputs_slot != NO_TABLE && eval_slot != NO_TABLE && d <= 2 && deg_max >= d {
                let eq_base = (r_inputs_slot as usize) * stride * 2;
                let ev_base = (eval_slot as usize) * stride * 2;
                if eq_base + 4 * t + 4 > tables.len() || ev_base + 4 * t + 4 > tables.len() {
                    return (false, Kx::ZERO);
                }
                let r0 = read_k(&tables[eq_base..], 2 * t);
                let r1 = read_k(&tables[eq_base..], 2 * t + 1) - r0;
                let v0 = read_k(&tables[ev_base..], 2 * t);
                let v1 = read_k(&tables[ev_base..], 2 * t + 1) - v0;
                let contribution = if d == 0 {
                    r0 * v0
                } else if d == 1 {
                    r0 * v1 + r1 * v0
                } else {
                    r1 * v1
                };
                coeff_d = coeff_d + gamma_to_k * contribution;
            }
        }
        (true, coeff_d)
    }

    #[kernel]
    #[cooperative_launch]
    #[allow(clippy::too_many_arguments)]
    pub fn fe_cooperative_row_round(
        tables: &[u64],
        header: &[u64],
        mcs_meta: &[u64],
        term_meta: &[u64],
        term_vars: &[u64],
        stride: u32,
        tail_len: u32,
        groups: u32,
        width: u32,
        num_tables: u32,
        cur_len: u32,
        mut partials: DisjointSlice<u64>,
        mut sum_scratch: DisjointSlice<u64>,
        mut coeffs_out: DisjointSlice<u64>,
        mut transcript_state: DisjointSlice<u64>,
        mut coeff_log: DisjointSlice<u64>,
        coeff_log_offset: u32,
        mut challenges: DisjointSlice<u64>,
        challenge_offset: u32,
        rc: &[u64],
        mut folded_tables: DisjointSlice<u64>,
    ) {
        let global = thread::index_1d().get();
        let step = (thread::gridDim_x() as usize) * (thread::blockDim_x() as usize);
        let stride = stride as usize;
        let tail_len = tail_len as usize;
        let groups = groups as usize;
        let width = width as usize;
        let width_words = width * 2;
        let num_tables = num_tables as usize;
        let cur_len = cur_len as usize;
        let coeff_log_offset = coeff_log_offset as usize;
        let challenge_offset = challenge_offset as usize;
        if step == 0 || width == 0 || cur_len < 2 {
            return;
        }

        let mut idx = global;
        let partial_total = groups * width;
        while idx < partial_total {
            let d = idx % width;
            let group = idx / width;
            let (ok, coeff_d) = fe_round_coeff_for_group(
                tables, header, mcs_meta, term_meta, term_vars, stride, tail_len, group, d,
            );
            if ok {
                let out_at = idx * 2;
                if out_at + 1 < partials.len() {
                    let words = coeff_d.as_words();
                    unsafe {
                        *partials.get_unchecked_mut(out_at) = words[0];
                        *partials.get_unchecked_mut(out_at + 1) = words[1];
                    }
                }
            }
            idx += step;
        }

        grid::sync();

        idx = global;
        let reduce_total = SUM_BLOCKS * width_words;
        while idx < reduce_total {
            let block = idx / width_words;
            let word = idx % width_words;
            let mut acc = Gl::ZERO;
            let mut group = block;
            while group < groups {
                let slot = group * width_words + word;
                if slot >= partials.len() {
                    break;
                }
                unsafe {
                    acc = acc + Gl::from_u64(*partials.get_unchecked_mut(slot));
                }
                group += SUM_BLOCKS;
            }
            if idx < sum_scratch.len() {
                unsafe {
                    *sum_scratch.get_unchecked_mut(idx) = acc.as_canonical_u64();
                }
            }
            idx += step;
        }

        grid::sync();

        if global < width_words {
            let mut acc = Gl::ZERO;
            for block in 0..SUM_BLOCKS {
                let slot = block * width_words + global;
                if slot >= sum_scratch.len() {
                    return;
                }
                unsafe {
                    acc = acc + Gl::from_u64(*sum_scratch.get_unchecked_mut(slot));
                }
            }
            let word = acc.as_canonical_u64();
            if global < coeffs_out.len() {
                unsafe {
                    *coeffs_out.get_unchecked_mut(global) = word;
                }
            }
            let log_slot = coeff_log_offset + global;
            if log_slot < coeff_log.len() {
                unsafe {
                    *coeff_log.get_unchecked_mut(log_slot) = word;
                }
            }
        }

        grid::sync();

        if global == 0 {
            absorb_coeffs_and_squeeze_two(
                &mut transcript_state,
                width_words as u64,
                &mut coeffs_out,
                width_words,
                &mut challenges,
                challenge_offset,
                rc,
            );
        }

        grid::sync();

        if challenge_offset + 1 >= challenges.len() {
            return;
        }
        let (r_c0, r_c1);
        unsafe {
            r_c0 = *challenges.get_unchecked_mut(challenge_offset);
            r_c1 = *challenges.get_unchecked_mut(challenge_offset + 1);
        }
        idx = global;
        let half = cur_len / 2;
        let fold_total = num_tables * half;
        while idx < fold_total {
            table_fold_at(idx, tables, num_tables, stride, cur_len, r_c0, r_c1, &mut folded_tables);
            idx += step;
        }
    }

    #[kernel]
    #[cooperative_launch]
    #[allow(clippy::too_many_arguments)]
    pub fn fe_cooperative_row_rounds(
        mut tables_a: DisjointSlice<u64>,
        mut tables_b: DisjointSlice<u64>,
        header: &[u64],
        mcs_meta: &[u64],
        term_meta: &[u64],
        term_vars: &[u64],
        stride: u32,
        active_len: u32,
        cur_len: u32,
        front_is_a: u32,
        width: u32,
        num_tables: u32,
        rounds: u32,
        mut partials: DisjointSlice<u64>,
        mut sum_scratch: DisjointSlice<u64>,
        mut coeffs_out: DisjointSlice<u64>,
        mut transcript_state: DisjointSlice<u64>,
        mut coeff_log: DisjointSlice<u64>,
        mut challenges: DisjointSlice<u64>,
        rc: &[u64],
    ) {
        let global = thread::index_1d().get();
        let step = (thread::gridDim_x() as usize) * (thread::blockDim_x() as usize);
        let stride = stride as usize;
        let mut active_len = active_len as usize;
        let mut cur_len = cur_len as usize;
        let mut front_is_a = front_is_a != 0;
        let width = width as usize;
        let width_words = width * 2;
        let num_tables = num_tables as usize;
        let rounds = rounds as usize;
        if step == 0 || width == 0 || cur_len < 2 {
            return;
        }

        let mut round = 0usize;
        while round < rounds {
            if cur_len < 2 {
                return;
            }
            let tail_len = active_len.div_ceil(2);
            let groups = tail_len.div_ceil(EVAL_CHUNK_PAIRS).max(1);

            let mut idx = global;
            let partial_total = groups * width;
            while idx < partial_total {
                let d = idx % width;
                let group = idx / width;
                let (ok, coeff_d) = fe_round_coeff_for_group_disjoint(
                    &mut tables_a,
                    &mut tables_b,
                    front_is_a,
                    header,
                    mcs_meta,
                    term_meta,
                    term_vars,
                    stride,
                    tail_len,
                    group,
                    d,
                );
                if ok {
                    let out_at = idx * 2;
                    if out_at + 1 < partials.len() {
                        let words = coeff_d.as_words();
                        unsafe {
                            *partials.get_unchecked_mut(out_at) = words[0];
                            *partials.get_unchecked_mut(out_at + 1) = words[1];
                        }
                    }
                }
                idx += step;
            }

            grid::sync();

            idx = global;
            let reduce_total = SUM_BLOCKS * width_words;
            while idx < reduce_total {
                let block = idx / width_words;
                let word = idx % width_words;
                let mut acc = Gl::ZERO;
                let mut group = block;
                while group < groups {
                    let slot = group * width_words + word;
                    if slot >= partials.len() {
                        break;
                    }
                    unsafe {
                        acc = acc + Gl::from_u64(*partials.get_unchecked_mut(slot));
                    }
                    group += SUM_BLOCKS;
                }
                if idx < sum_scratch.len() {
                    unsafe {
                        *sum_scratch.get_unchecked_mut(idx) = acc.as_canonical_u64();
                    }
                }
                idx += step;
            }

            grid::sync();

            if global < width_words {
                let mut acc = Gl::ZERO;
                for block in 0..SUM_BLOCKS {
                    let slot = block * width_words + global;
                    if slot >= sum_scratch.len() {
                        return;
                    }
                    unsafe {
                        acc = acc + Gl::from_u64(*sum_scratch.get_unchecked_mut(slot));
                    }
                }
                let word = acc.as_canonical_u64();
                if global < coeffs_out.len() {
                    unsafe {
                        *coeffs_out.get_unchecked_mut(global) = word;
                    }
                }
                let log_slot = round * width_words + global;
                if log_slot < coeff_log.len() {
                    unsafe {
                        *coeff_log.get_unchecked_mut(log_slot) = word;
                    }
                }
            }

            grid::sync();

            if global == 0 {
                absorb_coeffs_and_squeeze_two(
                    &mut transcript_state,
                    width_words as u64,
                    &mut coeffs_out,
                    width_words,
                    &mut challenges,
                    2 * round,
                    rc,
                );
            }

            grid::sync();

            if 2 * round + 1 >= challenges.len() {
                return;
            }
            let (r_c0, r_c1);
            unsafe {
                r_c0 = *challenges.get_unchecked_mut(2 * round);
                r_c1 = *challenges.get_unchecked_mut(2 * round + 1);
            }
            idx = global;
            let half = cur_len / 2;
            let fold_total = num_tables * half;
            while idx < fold_total {
                table_fold_at_disjoint(
                    idx,
                    &mut tables_a,
                    &mut tables_b,
                    front_is_a,
                    num_tables,
                    stride,
                    cur_len,
                    r_c0,
                    r_c1,
                );
                idx += step;
            }

            grid::sync();

            front_is_a = !front_is_a;
            cur_len /= 2;
            active_len = active_len.div_ceil(2).max(1);
            round += 1;
        }
    }

    fn table_fold_at_disjoint(
        idx: usize,
        tables_a: &mut DisjointSlice<u64>,
        tables_b: &mut DisjointSlice<u64>,
        front_is_a: bool,
        num_tables: usize,
        stride: usize,
        cur_len: usize,
        r_c0: u64,
        r_c1: u64,
    ) {
        let half = cur_len / 2;
        if idx >= num_tables * half {
            return;
        }
        let table = idx / half;
        let i = idx % half;
        let base = table * stride * 2;
        let lo_at = base + 4 * i;
        let (ok_lo, lo) = read_k_disjoint(tables_a, tables_b, front_is_a, lo_at);
        let (ok_hi, hi) = read_k_disjoint(tables_a, tables_b, front_is_a, lo_at + 2);
        if !ok_lo || !ok_hi {
            return;
        }
        let r = Kx::from_words(r_c0, r_c1);
        let folded = (lo + (hi - lo) * r).as_words();
        let out_at = base + 2 * i;
        write_pair_disjoint(tables_a, tables_b, !front_is_a, out_at, folded);
    }

    fn fe_round_coeff_for_group_disjoint(
        tables_a: &mut DisjointSlice<u64>,
        tables_b: &mut DisjointSlice<u64>,
        front_is_a: bool,
        header: &[u64],
        mcs_meta: &[u64],
        term_meta: &[u64],
        term_vars: &[u64],
        stride: usize,
        tail_len: usize,
        group: usize,
        d: usize,
    ) -> (bool, Kx) {
        let pair_start = group * EVAL_CHUNK_PAIRS;
        if pair_start >= tail_len || header.len() < 9 {
            return (false, Kx::ZERO);
        }
        let pair_end = if pair_start + EVAL_CHUNK_PAIRS < tail_len {
            pair_start + EVAL_CHUNK_PAIRS
        } else {
            tail_len
        };

        let deg_max = header[0] as usize;
        let num_mcs = header[1] as usize;
        let num_terms = header[2] as usize;
        let r_inputs_slot = header[3];
        let eval_slot = header[4];
        let gamma_to_k = Kx::from_words(header[5], header[6]);
        let f_at_zero = Kx::from_words(header[7], header[8]);

        let mut coeff_d = Kx::ZERO;
        for t in pair_start..pair_end {
            let (ok_e0, e0) = read_k_disjoint(tables_a, tables_b, front_is_a, 4 * t);
            let (ok_e1, e1_hi) = read_k_disjoint(tables_a, tables_b, front_is_a, 4 * t + 2);
            if !ok_e0 || !ok_e1 {
                return (false, Kx::ZERO);
            }
            let e1 = e1_hi - e0;
            let (ok_inner_d, inner_d) = inner_at_disjoint(
                tables_a, tables_b, front_is_a, mcs_meta, term_meta, term_vars, stride, num_mcs, num_terms, f_at_zero,
                t, d,
            );
            if !ok_inner_d {
                return (false, Kx::ZERO);
            }
            coeff_d = coeff_d + e0 * inner_d;
            if d >= 1 {
                let (ok_inner_prev, inner_prev) = inner_at_disjoint(
                    tables_a,
                    tables_b,
                    front_is_a,
                    mcs_meta,
                    term_meta,
                    term_vars,
                    stride,
                    num_mcs,
                    num_terms,
                    f_at_zero,
                    t,
                    d - 1,
                );
                if !ok_inner_prev {
                    return (false, Kx::ZERO);
                }
                coeff_d = coeff_d + e1 * inner_prev;
            }

            if r_inputs_slot != NO_TABLE && eval_slot != NO_TABLE && d <= 2 && deg_max >= d {
                let eq_base = (r_inputs_slot as usize) * stride * 2;
                let ev_base = (eval_slot as usize) * stride * 2;
                let (ok_r0, r0) = read_k_disjoint(tables_a, tables_b, front_is_a, eq_base + 4 * t);
                let (ok_r1, r1_hi) = read_k_disjoint(tables_a, tables_b, front_is_a, eq_base + 4 * t + 2);
                let (ok_v0, v0) = read_k_disjoint(tables_a, tables_b, front_is_a, ev_base + 4 * t);
                let (ok_v1, v1_hi) = read_k_disjoint(tables_a, tables_b, front_is_a, ev_base + 4 * t + 2);
                if !ok_r0 || !ok_r1 || !ok_v0 || !ok_v1 {
                    return (false, Kx::ZERO);
                }
                let r1 = r1_hi - r0;
                let v1 = v1_hi - v0;
                let contribution = if d == 0 {
                    r0 * v0
                } else if d == 1 {
                    r0 * v1 + r1 * v0
                } else {
                    r1 * v1
                };
                coeff_d = coeff_d + gamma_to_k * contribution;
            }
        }
        (true, coeff_d)
    }

    fn inner_at_disjoint(
        tables_a: &mut DisjointSlice<u64>,
        tables_b: &mut DisjointSlice<u64>,
        front_is_a: bool,
        mcs_meta: &[u64],
        term_meta: &[u64],
        term_vars: &[u64],
        stride: usize,
        num_mcs: usize,
        num_terms: usize,
        f_at_zero: Kx,
        t: usize,
        deg: usize,
    ) -> (bool, Kx) {
        let mut acc = Kx::ZERO;
        for mcs in 0..num_mcs {
            let meta_base = mcs * 4;
            if meta_base + 4 > mcs_meta.len() {
                return (true, acc);
            }
            let g = Kx::from_words(mcs_meta[meta_base], mcs_meta[meta_base + 1]);
            if g == Kx::ZERO {
                continue;
            }
            if mcs_meta[meta_base + 2] != 0 {
                if deg == 0 {
                    acc = acc + f_at_zero * g;
                }
                continue;
            }
            let var_slot_base = mcs_meta[meta_base + 3] as usize;
            for term in 0..num_terms {
                let tm_base = term * 4;
                if tm_base + 4 > term_meta.len() {
                    return (true, acc);
                }
                let (ok, inner) = inner_coeff_at_disjoint(
                    tables_a,
                    tables_b,
                    front_is_a,
                    term_meta,
                    term_vars,
                    stride,
                    t,
                    var_slot_base,
                    tm_base,
                    deg,
                );
                if !ok {
                    return (false, Kx::ZERO);
                }
                acc = acc + g * inner;
            }
        }
        (true, acc)
    }

    fn inner_coeff_at_disjoint(
        tables_a: &mut DisjointSlice<u64>,
        tables_b: &mut DisjointSlice<u64>,
        front_is_a: bool,
        term_meta: &[u64],
        term_vars: &[u64],
        stride: usize,
        t: usize,
        var_slot_base: usize,
        tm_base: usize,
        deg: usize,
    ) -> (bool, Kx) {
        let coeff = Kx::from_words(term_meta[tm_base], term_meta[tm_base + 1]);
        let var_off = term_meta[tm_base + 2] as usize;
        let var_count = term_meta[tm_base + 3] as usize;
        if var_count == 0 {
            return (true, if deg == 0 { coeff } else { Kx::ZERO });
        }
        if var_count == 1 {
            let (ok, a, b, exp) = read_var_disjoint(
                tables_a,
                tables_b,
                front_is_a,
                term_vars,
                stride,
                t,
                var_slot_base,
                var_off,
            );
            if !ok {
                return (false, Kx::ZERO);
            }
            if deg > exp {
                return (true, Kx::ZERO);
            }
            let mut binom = 1u64;
            for i in 0..deg {
                binom = binom * ((exp - i) as u64) / ((i + 1) as u64);
            }
            let mut acc = coeff * Kx::from_words(binom, 0);
            for _ in 0..(exp - deg) {
                acc = acc * a;
            }
            for _ in 0..deg {
                acc = acc * b;
            }
            return (true, acc);
        }
        let (ok0, a0, b0, _) = read_var_disjoint(
            tables_a,
            tables_b,
            front_is_a,
            term_vars,
            stride,
            t,
            var_slot_base,
            var_off,
        );
        let (ok1, a1, b1, _) = read_var_disjoint(
            tables_a,
            tables_b,
            front_is_a,
            term_vars,
            stride,
            t,
            var_slot_base,
            var_off + 1,
        );
        if !ok0 || !ok1 {
            return (false, Kx::ZERO);
        }
        let out = if deg == 0 {
            coeff * (a0 * a1)
        } else if deg == 1 {
            coeff * (a0 * b1 + b0 * a1)
        } else if deg == 2 {
            coeff * (b0 * b1)
        } else {
            Kx::ZERO
        };
        (true, out)
    }

    fn read_var_disjoint(
        tables_a: &mut DisjointSlice<u64>,
        tables_b: &mut DisjointSlice<u64>,
        front_is_a: bool,
        term_vars: &[u64],
        stride: usize,
        t: usize,
        var_slot_base: usize,
        var_idx: usize,
    ) -> (bool, Kx, Kx, usize) {
        let pair_base = var_idx * 2;
        if pair_base + 1 >= term_vars.len() {
            return (false, Kx::ZERO, Kx::ZERO, 0);
        }
        let var_pos = term_vars[pair_base] as usize;
        let exp = term_vars[pair_base + 1] as usize;
        let tbl_base = (var_slot_base + var_pos) * stride * 2;
        let (ok_a, a) = read_k_disjoint(tables_a, tables_b, front_is_a, tbl_base + 4 * t);
        let (ok_b, b_hi) = read_k_disjoint(tables_a, tables_b, front_is_a, tbl_base + 4 * t + 2);
        if !ok_a || !ok_b {
            return (false, Kx::ZERO, Kx::ZERO, 0);
        }
        (true, a, b_hi - a, exp)
    }

    fn read_k_disjoint(
        tables_a: &mut DisjointSlice<u64>,
        tables_b: &mut DisjointSlice<u64>,
        front_is_a: bool,
        word_base: usize,
    ) -> (bool, Kx) {
        if front_is_a {
            if word_base + 1 >= tables_a.len() {
                return (false, Kx::ZERO);
            }
            unsafe {
                (
                    true,
                    Kx::from_words(
                        *tables_a.get_unchecked_mut(word_base),
                        *tables_a.get_unchecked_mut(word_base + 1),
                    ),
                )
            }
        } else {
            if word_base + 1 >= tables_b.len() {
                return (false, Kx::ZERO);
            }
            unsafe {
                (
                    true,
                    Kx::from_words(
                        *tables_b.get_unchecked_mut(word_base),
                        *tables_b.get_unchecked_mut(word_base + 1),
                    ),
                )
            }
        }
    }

    fn write_pair_disjoint(
        tables_a: &mut DisjointSlice<u64>,
        tables_b: &mut DisjointSlice<u64>,
        write_a: bool,
        word_base: usize,
        words: [u64; 2],
    ) {
        if write_a {
            if word_base + 1 >= tables_a.len() {
                return;
            }
            unsafe {
                *tables_a.get_unchecked_mut(word_base) = words[0];
                *tables_a.get_unchecked_mut(word_base + 1) = words[1];
            }
        } else {
            if word_base + 1 >= tables_b.len() {
                return;
            }
            unsafe {
                *tables_b.get_unchecked_mut(word_base) = words[0];
                *tables_b.get_unchecked_mut(word_base + 1) = words[1];
            }
        }
    }

    fn table_fold_at(
        idx: usize,
        src: &[u64],
        num_tables: usize,
        stride: usize,
        cur_len: usize,
        r_c0: u64,
        r_c1: u64,
        dst: &mut DisjointSlice<u64>,
    ) {
        let half = cur_len / 2;
        if idx >= num_tables * half {
            return;
        }
        let table = idx / half;
        let i = idx % half;
        let base = table * stride * 2;
        let lo_at = base + 4 * i;
        if lo_at + 4 > src.len() {
            return;
        }
        let lo = Kx::from_words(src[lo_at], src[lo_at + 1]);
        let hi = Kx::from_words(src[lo_at + 2], src[lo_at + 3]);
        let r = Kx::from_words(r_c0, r_c1);
        let folded = (lo + (hi - lo) * r).as_words();
        let out_at = base + 2 * i;
        if out_at + 2 > dst.len() {
            return;
        }
        unsafe {
            *dst.get_unchecked_mut(out_at) = folded[0];
            *dst.get_unchecked_mut(out_at + 1) = folded[1];
        }
    }

    fn absorb_coeffs_and_squeeze_two(
        st: &mut DisjointSlice<u64>,
        len_prefix: u64,
        coeffs: &mut DisjointSlice<u64>,
        coeff_words: usize,
        out: &mut DisjointSlice<u64>,
        out_offset: usize,
        rc: &[u64],
    ) {
        if st.len() < P2_ST_WORDS || coeff_words > coeffs.len() || out_offset + 1 >= out.len() || rc.len() == 0 {
            return;
        }
        let mut state = load_p2_state(st);
        let mut cursor = unsafe { *st.get_unchecked_mut(P2_WIDTH) } as usize;
        absorb_word(&mut state, &mut cursor, len_prefix, rc);
        for i in 0..coeff_words {
            unsafe {
                absorb_word(&mut state, &mut cursor, *coeffs.get_unchecked_mut(i), rc);
            }
        }
        squeeze_two(&mut state, &mut cursor, out, out_offset, rc);
        store_p2_state(st, state, cursor);
    }

    fn load_p2_state(st: &mut DisjointSlice<u64>) -> P2State {
        unsafe {
            P2State {
                s0: Gl::from_u64(*st.get_unchecked_mut(0)),
                s1: Gl::from_u64(*st.get_unchecked_mut(1)),
                s2: Gl::from_u64(*st.get_unchecked_mut(2)),
                s3: Gl::from_u64(*st.get_unchecked_mut(3)),
                s4: Gl::from_u64(*st.get_unchecked_mut(4)),
                s5: Gl::from_u64(*st.get_unchecked_mut(5)),
                s6: Gl::from_u64(*st.get_unchecked_mut(6)),
                s7: Gl::from_u64(*st.get_unchecked_mut(7)),
            }
        }
    }

    fn store_p2_state(st: &mut DisjointSlice<u64>, state: P2State, cursor: usize) {
        unsafe {
            *st.get_unchecked_mut(0) = state.s0.as_canonical_u64();
            *st.get_unchecked_mut(1) = state.s1.as_canonical_u64();
            *st.get_unchecked_mut(2) = state.s2.as_canonical_u64();
            *st.get_unchecked_mut(3) = state.s3.as_canonical_u64();
            *st.get_unchecked_mut(4) = state.s4.as_canonical_u64();
            *st.get_unchecked_mut(5) = state.s5.as_canonical_u64();
            *st.get_unchecked_mut(6) = state.s6.as_canonical_u64();
            *st.get_unchecked_mut(7) = state.s7.as_canonical_u64();
            *st.get_unchecked_mut(P2_WIDTH) = cursor as u64;
        }
    }

    fn absorb_word(state: &mut P2State, cursor: &mut usize, word: u64, rc: &[u64]) {
        let v = Gl::from_u64(word);
        if *cursor >= P2_RATE {
            *state = permute(*state, rc);
            *cursor = 0;
        }
        match *cursor {
            0 => state.s0 = v,
            1 => state.s1 = v,
            2 => state.s2 = v,
            _ => state.s3 = v,
        }
        *cursor += 1;
    }

    fn squeeze_two(
        state: &mut P2State,
        cursor: &mut usize,
        out: &mut DisjointSlice<u64>,
        out_offset: usize,
        rc: &[u64],
    ) {
        if *cursor >= P2_RATE {
            *state = permute(*state, rc);
            *cursor = 0;
        }
        match *cursor {
            0 => state.s0 = Gl::ONE,
            1 => state.s1 = Gl::ONE,
            2 => state.s2 = Gl::ONE,
            _ => state.s3 = Gl::ONE,
        }
        *state = permute(*state, rc);
        *cursor = 0;
        unsafe {
            *out.get_unchecked_mut(out_offset) = state.s0.as_canonical_u64();
            *out.get_unchecked_mut(out_offset + 1) = state.s1.as_canonical_u64();
        }
    }
}
