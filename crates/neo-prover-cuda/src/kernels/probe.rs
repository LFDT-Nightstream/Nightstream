//! Toolchain and arithmetic smoke kernels.
//!
//! Owns the `parity smoke` gate surface: proves the cargo-oxide pipeline,
//! device buffers, and `kernels::goldilocks` agree with the CPU field
//! implementations. Not part of the prover path.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cooperative_launch, cuda_module, grid, kernel, thread, DisjointSlice};
use cuda_host::EmbeddedModuleError;

use crate::kernels::goldilocks::Kx;

pub use probe_kernels::LoadedModule as ProbeKernelModule;

pub fn load_probe_kernels(ctx: &Arc<CudaContext>) -> Result<ProbeKernelModule, EmbeddedModuleError> {
    probe_kernels::load(ctx)
}

/// `out[i] = a[i] * b[i] + a[i]` over K. All buffers hold 2 words per element.
pub fn launch_k_mul_add(
    module: &ProbeKernelModule,
    stream: &Arc<CudaStream>,
    elems: usize,
    a: &DeviceBuffer<u64>,
    b: &DeviceBuffer<u64>,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    if elems == 0 {
        return Ok(());
    }
    module.k_mul_add(stream, LaunchConfig::for_num_elems(elems as u32), a, b, out)
}

/// Launch a tiny cooperative-grid barrier probe.
///
/// `out` must have at least `blocks + 2` slots. The kernel writes one marker
/// per block, performs `grid::sync()`, then block 0 sums the markers into
/// `out[blocks]` and records `blocks` in `out[blocks + 1]`.
pub fn launch_cooperative_grid_sync_probe(
    module: &ProbeKernelModule,
    stream: &Arc<CudaStream>,
    blocks: u32,
    out: &mut DeviceBuffer<u32>,
) -> Result<(), DriverError> {
    if blocks == 0 {
        return Ok(());
    }
    module.cooperative_grid_sync_probe(
        stream,
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        },
        out,
    )
}

#[cuda_module]
pub mod probe_kernels {
    use super::*;

    #[kernel]
    pub fn k_mul_add(a: &[u64], b: &[u64], mut out: DisjointSlice<u64>) {
        let i = thread::index_1d().get();
        let base = 2 * i;
        if base + 1 >= a.len() || base + 1 >= b.len() || base + 1 >= out.len() {
            return;
        }
        let av = Kx::from_words(a[base], a[base + 1]);
        let bv = Kx::from_words(b[base], b[base + 1]);
        let words = (av * bv + av).as_words();
        unsafe {
            *out.get_unchecked_mut(base) = words[0];
            *out.get_unchecked_mut(base + 1) = words[1];
        }
    }

    #[kernel]
    #[cooperative_launch]
    pub fn cooperative_grid_sync_probe(mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x() as usize;
        let blocks = thread::gridDim_x() as usize;
        if out.len() < blocks + 2 {
            return;
        }

        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(bid) = (bid as u32) + 1;
            }
        }

        grid::sync();

        if bid == 0 && tid == 0 {
            let mut sum = 0u32;
            for block in 0..blocks {
                unsafe {
                    sum = sum.wrapping_add(*out.get_unchecked_mut(block));
                }
            }
            unsafe {
                *out.get_unchecked_mut(blocks) = sum;
                *out.get_unchecked_mut(blocks + 1) = blocks as u32;
            }
        }
    }
}
