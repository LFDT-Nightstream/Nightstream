//! Kernels shared by both Π_CCS sumcheck channels.
//!
//! Owns the generic K-table fold (`dst[i] = lo + (hi − lo)·r` over every
//! table in a strided buffer) and the two-stage per-group partials
//! reduction. Owns no channel semantics — round evaluation lives in
//! `pi_ccs_fe` / `pi_ccs_nc`.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::EmbeddedModuleError;

use crate::kernels::goldilocks::{Gl, Kx};

/// Blocks in the first reduction stage: enough threads to spread tens of
/// thousands of groups, small enough that stage B is a trivial second pass.
pub const SUM_BLOCKS: usize = 256;

pub use sumcheck_common_kernels::LoadedModule as SumcheckCommonModule;

pub fn load_sumcheck_common(ctx: &Arc<CudaContext>) -> Result<SumcheckCommonModule, EmbeddedModuleError> {
    sumcheck_common_kernels::load(ctx)
}

/// Fold every K-table in a strided buffer at challenge `r`; ping-pong
/// buffers (an in-place parallel fold races on its own reads).
#[allow(clippy::too_many_arguments)]
pub fn launch_table_fold(
    module: &SumcheckCommonModule,
    stream: &Arc<CudaStream>,
    src: &DeviceBuffer<u64>,
    num_tables: usize,
    stride: usize,
    cur_len: usize,
    r_c0: u64,
    r_c1: u64,
    dst: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let half = cur_len / 2;
    module.table_fold(
        stream,
        LaunchConfig::for_num_elems((num_tables * half) as u32),
        src,
        num_tables as u32,
        stride as u32,
        cur_len as u32,
        r_c0,
        r_c1,
        dst,
    )
}

/// Fold every K-table at `challenge[offset..offset+2]`.
#[allow(clippy::too_many_arguments)]
pub fn launch_table_fold_from_challenge(
    module: &SumcheckCommonModule,
    stream: &Arc<CudaStream>,
    src: &DeviceBuffer<u64>,
    num_tables: usize,
    stride: usize,
    cur_len: usize,
    challenge: &DeviceBuffer<u64>,
    challenge_offset: usize,
    dst: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let half = cur_len / 2;
    module.table_fold_from_challenge(
        stream,
        LaunchConfig::for_num_elems((num_tables * half) as u32),
        src,
        num_tables as u32,
        stride as u32,
        cur_len as u32,
        challenge,
        challenge_offset as u32,
        dst,
    )
}

/// Reduce `[groups][width_words]` partials to `[width_words]` in two
/// stages — a single-stage sum serializes at large group counts.
/// `scratch` must hold `SUM_BLOCKS * width_words` words.
pub fn launch_sum_partials(
    module: &SumcheckCommonModule,
    stream: &Arc<CudaStream>,
    partials: &DeviceBuffer<u64>,
    groups: usize,
    width_words: usize,
    scratch: &mut DeviceBuffer<u64>,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.sum_partials_blocks(
        stream,
        LaunchConfig::for_num_elems((SUM_BLOCKS * width_words) as u32),
        partials,
        groups as u32,
        width_words as u32,
        scratch,
    )?;
    module.sum_partials(
        stream,
        LaunchConfig::for_num_elems(width_words as u32),
        scratch,
        SUM_BLOCKS as u32,
        width_words as u32,
        out,
    )
}

#[cuda_module]
pub mod sumcheck_common_kernels {
    use super::*;

    /// One thread per (table, output index): `dst[i] = lo + (hi - lo) · r`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn table_fold(
        src: &[u64],
        num_tables: u32,
        stride: u32,
        cur_len: u32,
        r_c0: u64,
        r_c1: u64,
        mut dst: DisjointSlice<u64>,
    ) {
        table_fold_at(
            thread::index_1d().get(),
            src,
            num_tables as usize,
            stride as usize,
            cur_len as usize,
            r_c0,
            r_c1,
            &mut dst,
        );
    }

    /// Same fold, but the challenge is read from a device buffer.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn table_fold_from_challenge(
        src: &[u64],
        num_tables: u32,
        stride: u32,
        cur_len: u32,
        challenge: &[u64],
        challenge_offset: u32,
        mut dst: DisjointSlice<u64>,
    ) {
        let offset = challenge_offset as usize;
        if offset + 1 >= challenge.len() {
            return;
        }
        table_fold_at(
            thread::index_1d().get(),
            src,
            num_tables as usize,
            stride as usize,
            cur_len as usize,
            challenge[offset],
            challenge[offset + 1],
            &mut dst,
        );
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

    /// Stage A: thread (block, word) sums its block-strided slice of the
    /// groups. Word-level summation is exact — the c0/c1 Goldilocks lanes
    /// add independently.
    #[kernel]
    pub fn sum_partials_blocks(partials: &[u64], groups: u32, width_words: u32, mut scratch: DisjointSlice<u64>) {
        let idx = thread::index_1d().get();
        let groups = groups as usize;
        let width_words = width_words as usize;
        if idx >= SUM_BLOCKS * width_words {
            return;
        }
        let block = idx / width_words;
        let word = idx % width_words;
        let mut acc = Gl::ZERO;
        let mut group = block;
        while group < groups {
            let slot = group * width_words + word;
            if slot >= partials.len() {
                break;
            }
            acc = acc + Gl::from_u64(partials[slot]);
            group += SUM_BLOCKS;
        }
        if idx >= scratch.len() {
            return;
        }
        unsafe {
            *scratch.get_unchecked_mut(idx) = acc.as_canonical_u64();
        }
    }

    /// Stage B: one thread per output word folds the block sums.
    #[kernel]
    pub fn sum_partials(partials: &[u64], groups: u32, width_words: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d().get();
        let groups = groups as usize;
        let width_words = width_words as usize;
        if idx >= width_words {
            return;
        }
        let mut acc = Gl::ZERO;
        for group in 0..groups {
            let slot = group * width_words + idx;
            if slot >= partials.len() {
                return;
            }
            acc = acc + Gl::from_u64(partials[slot]);
        }
        if idx >= out.len() {
            return;
        }
        unsafe {
            *out.get_unchecked_mut(idx) = acc.as_canonical_u64();
        }
    }
}
