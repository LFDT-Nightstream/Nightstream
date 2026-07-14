//! Pi_CCS output-surface packing.
//!
//! Owns the device layout that hands Pi_CCS K-valued output surfaces to
//! Pi_RLC: `[claim][surface][d_pad][c0,c1]`, where surfaces are every
//! `y_ring[j]` followed by optional `y_zcol`.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::EmbeddedModuleError;

use crate::kernels::ajtai::RING_D;
use crate::kernels::goldilocks::Kx;

pub use pi_ccs_output_kernels::LoadedModule as CcsOutputKernelModule;

pub fn load_ccs_output_kernels(ctx: &Arc<CudaContext>) -> Result<CcsOutputKernelModule, EmbeddedModuleError> {
    pi_ccs_output_kernels::load(ctx)
}

#[allow(clippy::too_many_arguments)]
pub fn launch_ccs_pack_k_surfaces(
    module: &CcsOutputKernelModule,
    stream: &Arc<CudaStream>,
    y_eval_words: &DeviceBuffer<u64>,
    nc_final_words: &DeviceBuffer<u64>,
    claims: usize,
    t_core: usize,
    include_y_zcol: bool,
    d_pad: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let surface_count = t_core + usize::from(include_y_zcol);
    if claims == 0 || surface_count == 0 {
        return Ok(());
    }
    module.ccs_pack_k_surfaces(
        stream,
        LaunchConfig::for_num_elems((claims * surface_count * d_pad) as u32),
        y_eval_words,
        nc_final_words,
        claims as u32,
        t_core as u32,
        include_y_zcol as u32,
        d_pad as u32,
        out,
    )
}

/// Compute the public FE claimed sum from resident running-child y surfaces.
///
/// One thread computes each `(matrix, child)` contribution; a final canonical
/// reduction preserves the protocol's matrix-major, child-minor order.
#[allow(clippy::too_many_arguments)]
pub fn launch_ccs_running_initial_sum(
    module: &CcsOutputKernelModule,
    stream: &Arc<CudaStream>,
    surfaces: &DeviceBuffer<u64>,
    claims: usize,
    surface_count: usize,
    t_core: usize,
    d_pad: usize,
    chi_words: &DeviceBuffer<u64>,
    eval_lanes: usize,
    weights: &DeviceBuffer<u64>,
    partials: &mut DeviceBuffer<u64>,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let count = claims * t_core;
    module.ccs_running_initial_sum_partials(
        stream,
        LaunchConfig::for_num_elems(count as u32),
        surfaces,
        claims as u32,
        surface_count as u32,
        t_core as u32,
        d_pad as u32,
        chi_words,
        eval_lanes as u32,
        partials,
    )?;
    module.ccs_running_initial_sum_reduce(
        stream,
        LaunchConfig::for_num_elems(1),
        partials,
        weights,
        count as u32,
        out,
    )
}

#[cuda_module]
mod pi_ccs_output_kernels {
    use super::*;

    #[kernel]
    pub fn ccs_pack_k_surfaces(
        y_eval_words: &[u64],
        nc_final_words: &[u64],
        claims: u32,
        t_core: u32,
        include_y_zcol: u32,
        d_pad: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let claims = claims as usize;
        let t_core = t_core as usize;
        let include_y_zcol = include_y_zcol != 0;
        let d_pad = d_pad as usize;
        let surface_count = t_core + if include_y_zcol { 1 } else { 0 };
        if claims == 0 || surface_count == 0 || d_pad < RING_D || idx >= claims * surface_count * d_pad {
            return;
        }

        let lane = idx % d_pad;
        let surface = (idx / d_pad) % surface_count;
        let claim = idx / (surface_count * d_pad);
        let out_at = idx * 2;
        if out_at + 1 >= out.len() {
            return;
        }
        if lane >= RING_D {
            unsafe {
                *out.get_unchecked_mut(out_at) = 0;
                *out.get_unchecked_mut(out_at + 1) = 0;
            }
            return;
        }

        let (c0, c1) = if surface < t_core {
            let per_claim = 2 * t_core * RING_D;
            let re = claim * per_claim + (2 * surface) * RING_D + lane;
            let im = claim * per_claim + (2 * surface + 1) * RING_D + lane;
            if im >= y_eval_words.len() {
                return;
            }
            (y_eval_words[re], y_eval_words[im])
        } else {
            let base = (claim * RING_D + lane) * 2;
            if base + 1 >= nc_final_words.len() {
                return;
            }
            (nc_final_words[base], nc_final_words[base + 1])
        };

        unsafe {
            *out.get_unchecked_mut(out_at) = c0;
            *out.get_unchecked_mut(out_at + 1) = c1;
        }
    }

    #[kernel]
    pub fn ccs_running_initial_sum_partials(
        surfaces: &[u64],
        claims: u32,
        surface_count: u32,
        t_core: u32,
        d_pad: u32,
        chi_words: &[u64],
        eval_lanes: u32,
        mut partials: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let claims = claims as usize;
        let surface_count = surface_count as usize;
        let t_core = t_core as usize;
        let d_pad = d_pad as usize;
        let eval_lanes = eval_lanes as usize;
        if claims == 0
            || t_core == 0
            || surface_count < t_core
            || idx >= claims * t_core
            || chi_words.len() < 2 * eval_lanes
            || 2 * idx + 1 >= partials.len()
        {
            return;
        }
        if eval_lanes > d_pad {
            return;
        }

        let claim = idx % claims;
        let surface = idx / claims;
        let mut y_eval = Kx::ZERO;
        for lane in 0..eval_lanes {
            let at = ((claim * surface_count + surface) * d_pad + lane) * 2;
            if at + 1 >= surfaces.len() {
                return;
            }
            let chi = Kx::from_words(chi_words[2 * lane], chi_words[2 * lane + 1]);
            y_eval = y_eval + Kx::from_words(surfaces[at], surfaces[at + 1]) * chi;
        }
        let words = y_eval.as_words();
        unsafe {
            *partials.get_unchecked_mut(2 * idx) = words[0];
            *partials.get_unchecked_mut(2 * idx + 1) = words[1];
        }
    }

    #[kernel]
    pub fn ccs_running_initial_sum_reduce(partials: &[u64], weights: &[u64], count: u32, mut out: DisjointSlice<u64>) {
        if thread::index_1d().get() != 0 || out.len() < 2 {
            return;
        }
        let mut acc = Kx::ZERO;
        for idx in 0..count as usize {
            let at = 2 * idx;
            if at + 1 >= partials.len() || at + 1 >= weights.len() {
                return;
            }
            acc = acc + Kx::from_words(partials[at], partials[at + 1]) * Kx::from_words(weights[at], weights[at + 1]);
        }
        let words = acc.as_words();
        unsafe {
            *out.get_unchecked_mut(0) = words[0];
            *out.get_unchecked_mut(1) = words[1];
        }
    }
}
