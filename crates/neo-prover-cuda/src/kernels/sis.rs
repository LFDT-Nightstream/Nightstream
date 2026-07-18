//! Fixed-seed SIS message preparation.
//!
//! The canonical seeded Ajtai map is still supplied by `neo-ajtai`; this
//! kernel only converts canonical Goldilocks fields into its 41-digit
//! balanced-ternary message layout without a host round trip.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::EmbeddedModuleError;

pub const BALANCED_TERNARY_DIGITS: usize = 41;
const SHIFT: u64 = 18_236_498_188_585_393_201;
const MODULUS_MINUS_SHIFT: u64 = 210_245_880_829_191_120;
const GOLDILOCKS_NEG_ONE: u64 = 18_446_744_069_414_584_320;

pub use sis_kernels::LoadedModule as SisKernelModule;

pub fn load_sis_kernels(ctx: &Arc<CudaContext>) -> Result<SisKernelModule, EmbeddedModuleError> {
    sis_kernels::load(ctx)
}

pub fn launch_balanced_ternary_message(
    module: &SisKernelModule,
    stream: &Arc<CudaStream>,
    fields: &DeviceBuffer<u64>,
    field_count: usize,
    message_cols: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    if field_count == 0 {
        return Ok(());
    }
    assert_eq!(fields.len(), field_count, "SIS field count");
    assert_eq!(out.len(), message_cols * neo_math::D, "SIS message shape");
    module.sis_balanced_ternary_message(
        stream,
        LaunchConfig::for_num_elems(field_count as u32),
        fields,
        field_count as u32,
        message_cols as u32,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_pi_ccs_outputs_preimage(
    module: &SisKernelModule,
    stream: &Arc<CudaStream>,
    surfaces: &DeviceBuffer<u64>,
    claims: usize,
    t_core: usize,
    d_pad: usize,
    active_lanes: usize,
    include_y_zcol: bool,
    domains: &DeviceBuffer<u64>,
    output_domain_len: usize,
    claim_domain_len: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    assert!(active_lanes <= d_pad, "Pi_CCS active lanes exceed resident stride");
    let surface_count = t_core + usize::from(include_y_zcol);
    assert_eq!(
        surfaces.len(),
        claims * surface_count * d_pad * 2,
        "Pi_CCS SIS surfaces"
    );
    assert_eq!(
        domains.len(),
        output_domain_len + claim_domain_len,
        "Pi_CCS SIS domains"
    );
    module.sis_pi_ccs_outputs_preimage(
        stream,
        LaunchConfig::for_num_elems(out.len() as u32),
        surfaces,
        claims as u32,
        t_core as u32,
        d_pad as u32,
        active_lanes as u32,
        u32::from(include_y_zcol),
        domains,
        output_domain_len as u32,
        claim_domain_len as u32,
        out,
    )
}

#[cuda_module]
mod sis_kernels {
    use super::*;

    #[kernel]
    pub fn sis_balanced_ternary_message(
        fields: &[u64],
        field_count: u32,
        message_cols: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let field = thread::index_1d().get();
        if field >= field_count as usize || field >= fields.len() || message_cols == 0 {
            return;
        }
        let value = fields[field];
        let mut remaining = if value >= MODULUS_MINUS_SHIFT {
            value - MODULUS_MINUS_SHIFT
        } else {
            value + SHIFT
        };
        for digit in 0..BALANCED_TERNARY_DIGITS {
            let trit = remaining % 3;
            remaining /= 3;
            let index = field * BALANCED_TERNARY_DIGITS + digit;
            let row = index / message_cols as usize;
            let col = index % message_cols as usize;
            let slot = col * neo_math::D + row;
            if slot >= out.len() {
                return;
            }
            let centered = match trit {
                0 => GOLDILOCKS_NEG_ONE,
                1 => 0,
                _ => 1,
            };
            unsafe {
                *out.get_unchecked_mut(slot) = centered;
            }
        }
    }

    #[kernel]
    pub fn sis_pi_ccs_outputs_preimage(
        surfaces: &[u64],
        claims: u32,
        t_core: u32,
        d_pad: u32,
        active_lanes: u32,
        include_y_zcol: u32,
        domains: &[u64],
        output_domain_len: u32,
        claim_domain_len: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let index = thread::index_1d().get();
        if index >= out.len() {
            return;
        }
        let output_domain_len = output_domain_len as usize;
        let claim_domain_len = claim_domain_len as usize;
        let claims = claims as usize;
        let t_core = t_core as usize;
        let d_pad = d_pad as usize;
        let active_lanes = active_lanes as usize;
        let include_y_zcol = include_y_zcol != 0;
        let outer_len = output_domain_len + 1;
        let surface_span = 1 + 2 * active_lanes;
        let surface_count = t_core + usize::from(include_y_zcol);
        let claim_len = claim_domain_len + 1 + t_core * surface_span + if include_y_zcol { surface_span } else { 1 };

        let value = if index < output_domain_len {
            if index >= domains.len() {
                return;
            }
            domains[index]
        } else if index == output_domain_len {
            claims as u64
        } else {
            let relative = index - outer_len;
            let claim = relative / claim_len;
            if claim >= claims {
                return;
            }
            let mut local = relative % claim_len;
            if local < claim_domain_len {
                let domain_index = output_domain_len + local;
                if domain_index >= domains.len() {
                    return;
                }
                domains[domain_index]
            } else {
                local -= claim_domain_len;
                if local == 0 {
                    t_core as u64
                } else {
                    local -= 1;
                    let surface_area = surface_count * surface_span;
                    if local >= surface_area {
                        0
                    } else {
                        let surface = local / surface_span;
                        let slot = local % surface_span;
                        if slot == 0 {
                            active_lanes as u64
                        } else {
                            let surface_word = ((claim * surface_count + surface) * d_pad * 2) + slot - 1;
                            if surface_word >= surfaces.len() {
                                return;
                            }
                            surfaces[surface_word]
                        }
                    }
                }
            }
        };
        unsafe {
            *out.get_unchecked_mut(index) = value;
        }
    }
}
