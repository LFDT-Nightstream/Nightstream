//! Π_RLC output-surface kernels.
//!
//! Owns small CE-claim surfaces derived from the device-resident mixed witness.
//! The bulk ring product `Z_mix = Σρ_i Z_i` lives in `kernels::ajtai`; this
//! module only packs public claim data from that resident witness.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::EmbeddedModuleError;

use crate::kernels::ajtai::RING_D;
use crate::kernels::goldilocks::{Gl, Kx};

pub use rlc_kernels::LoadedModule as RlcKernelModule;

pub fn load_rlc_kernels(ctx: &Arc<CudaContext>) -> Result<RlcKernelModule, EmbeddedModuleError> {
    rlc_kernels::load(ctx)
}

/// Pack `X ∈ F^{D×m_in}` from the resident mixed witness.
///
/// Output is row-major `Mat<F>` data. Columns outside
/// `ceil(m_in / D)` are structural zeros, matching
/// `project_x_from_witness_mat`.
pub fn launch_rlc_pack_public_x(
    module: &RlcKernelModule,
    stream: &Arc<CudaStream>,
    z_mix: &DeviceBuffer<u64>,
    m_in: usize,
    z_cols: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.rlc_pack_public_x(
        stream,
        LaunchConfig::for_num_elems((RING_D * m_in) as u32),
        z_mix,
        m_in as u32,
        z_cols as u32,
        out,
    )
}

/// Pack only the authority-bearing public columns from resident witness
/// planes. Output layout is `[claim][row][active_col]`.
#[allow(clippy::too_many_arguments)]
pub fn launch_rlc_pack_active_public_x(
    module: &RlcKernelModule,
    stream: &Arc<CudaStream>,
    planes: &DeviceBuffer<u64>,
    claims: usize,
    plane_stride: usize,
    m_in: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let active_cols = m_in.div_ceil(RING_D);
    module.rlc_pack_active_public_x(
        stream,
        LaunchConfig::for_num_elems((claims * RING_D * active_cols) as u32),
        planes,
        claims as u32,
        plane_stride as u32,
        active_cols as u32,
        out,
    )
}

/// Combine K-valued CE surfaces under the device-resident rho coefficients.
///
/// `inputs` layout is `[input][surface][d_pad][c0,c1]`; `out` layout is
/// `[surface][d_pad][c0,c1]`. Tail lanes `D..d_pad` are written as zero.
#[allow(clippy::too_many_arguments)]
pub fn launch_rlc_combine_k_surfaces(
    module: &RlcKernelModule,
    stream: &Arc<CudaStream>,
    rhos: &DeviceBuffer<u64>,
    inputs: &DeviceBuffer<u64>,
    input_count: usize,
    surface_count: usize,
    d_pad: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.rlc_combine_k_surfaces(
        stream,
        LaunchConfig::for_num_elems((surface_count * d_pad) as u32),
        rhos,
        inputs,
        input_count as u32,
        surface_count as u32,
        d_pad as u32,
        out,
    )
}

#[cuda_module]
mod rlc_kernels {
    use super::*;

    #[kernel]
    pub fn rlc_pack_public_x(z_mix: &[u64], m_in: u32, z_cols: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d().get();
        let m_in = m_in as usize;
        let z_cols = z_cols as usize;
        if m_in == 0 || idx >= RING_D * m_in || idx >= out.len() {
            return;
        }

        let x_col = idx % m_in;
        let row = idx / m_in;
        let required_cols = m_in.div_ceil(RING_D);
        if x_col >= required_cols || x_col >= z_cols {
            return;
        }

        let src = x_col * RING_D + row;
        if src >= z_mix.len() {
            return;
        }
        unsafe {
            *out.get_unchecked_mut(idx) = z_mix[src];
        }
    }

    #[kernel]
    pub fn rlc_pack_active_public_x(
        planes: &[u64],
        claims: u32,
        plane_stride: u32,
        active_cols: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let claims = claims as usize;
        let plane_stride = plane_stride as usize;
        let active_cols = active_cols as usize;
        let words_per_claim = RING_D * active_cols;
        if active_cols == 0 || idx >= claims * words_per_claim || idx >= out.len() {
            return;
        }

        let claim = idx / words_per_claim;
        let within = idx % words_per_claim;
        let row = within / active_cols;
        let col = within % active_cols;
        let src = claim * plane_stride + col * RING_D + row;
        if src >= planes.len() {
            return;
        }
        unsafe {
            *out.get_unchecked_mut(idx) = planes[src];
        }
    }

    #[kernel]
    pub fn rlc_combine_k_surfaces(
        rhos: &[u64],
        inputs: &[u64],
        input_count: u32,
        surface_count: u32,
        d_pad: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let input_count = input_count as usize;
        let surface_count = surface_count as usize;
        let d_pad = d_pad as usize;
        if input_count == 0 || surface_count == 0 || d_pad < RING_D || idx >= surface_count * d_pad {
            return;
        }

        let lane = idx % d_pad;
        let surface = idx / d_pad;
        let out_base = idx * 2;
        if out_base + 1 >= out.len() {
            return;
        }
        if lane >= RING_D {
            unsafe {
                *out.get_unchecked_mut(out_base) = 0;
                *out.get_unchecked_mut(out_base + 1) = 0;
            }
            return;
        }

        let mut acc = Kx::ZERO;
        let input_stride = surface_count * d_pad * 2;
        for input in 0..input_count {
            let input_base = input * input_stride + surface * d_pad * 2;
            for source_lane in 0..RING_D {
                let value_base = input_base + source_lane * 2;
                if value_base + 1 >= inputs.len() {
                    return;
                }
                let value = Kx::from_words(inputs[value_base], inputs[value_base + 1]);
                if value == Kx::ZERO {
                    continue;
                }
                let rho = rho_entry_phi81(rhos, input, lane, source_lane);
                if rho != Gl::ZERO {
                    acc = acc + value.scale_base(rho);
                }
            }
        }

        let words = acc.as_words();
        unsafe {
            *out.get_unchecked_mut(out_base) = words[0];
            *out.get_unchecked_mut(out_base + 1) = words[1];
        }
    }

    fn rho_entry_phi81(rhos: &[u64], input: usize, row: usize, col: usize) -> Gl {
        let base = input * RING_D;
        let mut acc = Gl::ZERO;
        for coeff_idx in 0..RING_D {
            let coeff_at = base + coeff_idx;
            if coeff_at >= rhos.len() {
                return Gl::ZERO;
            }
            let coeff = Gl::from_u64(rhos[coeff_at]);
            if coeff == Gl::ZERO {
                continue;
            }
            let shifted = coeff_idx + col;
            if shifted < RING_D {
                if shifted == row {
                    acc = acc + coeff;
                }
            } else if shifted < RING_D + RING_D / 2 {
                let reduced = shifted - RING_D;
                if reduced == row {
                    acc = acc - coeff;
                }
                if reduced + RING_D / 2 == row {
                    acc = acc - coeff;
                }
            } else if shifted - RING_D - RING_D / 2 == row {
                acc = acc + coeff;
            }
        }
        acc
    }
}
