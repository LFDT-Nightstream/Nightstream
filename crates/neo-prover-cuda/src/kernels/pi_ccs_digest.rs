//! Pi_CCS output-digest preimage assembly.
//!
//! Owns only the device copy schedule for the canonical
//! `pi_ccs_outputs_digest` preimages. Poseidon2 hashing stays in the shared
//! `poseidon2` kernel module.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::EmbeddedModuleError;

pub use pi_ccs_digest_kernels::LoadedModule as CcsDigestKernelModule;

pub fn load_ccs_digest_kernels(ctx: &Arc<CudaContext>) -> Result<CcsDigestKernelModule, EmbeddedModuleError> {
    pi_ccs_digest_kernels::load(ctx)
}

#[allow(clippy::too_many_arguments)]
pub fn launch_ccs_build_output_claim_digest_preimages(
    module: &CcsDigestKernelModule,
    stream: &Arc<CudaStream>,
    surfaces: &DeviceBuffer<u64>,
    commitment_words: &DeviceBuffer<u64>,
    use_device_commitments: bool,
    commitment_stride: usize,
    public_x_words: &DeviceBuffer<u64>,
    use_device_x: bool,
    public_x_stride: usize,
    claims: usize,
    surface_count: usize,
    t_core: usize,
    d_pad: usize,
    surface_lanes: usize,
    include_y_zcol: bool,
    write_ct_field: bool,
    write_y_zcol_field: bool,
    plan: &DeviceBuffer<u64>,
    prefix_fields_start: usize,
    prefix_fields_len: usize,
    prefix_offsets_start: usize,
    prefix_lengths_start: usize,
    commitment_offsets_start: usize,
    commitment_lengths_start: usize,
    x_offsets_start: usize,
    x_lengths_start: usize,
    suffix_fields_start: usize,
    suffix_fields_len: usize,
    suffix_offsets_start: usize,
    suffix_lengths_start: usize,
    preimage_offsets_start: usize,
    preimage_lengths_start: usize,
    preimages_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    if claims == 0 {
        return Ok(());
    }
    module.ccs_build_output_claim_digest_preimages(
        stream,
        LaunchConfig::for_num_elems(claims as u32),
        surfaces,
        surfaces,
        commitment_words,
        commitment_words,
        use_device_commitments,
        commitment_stride as u32,
        public_x_words,
        public_x_words,
        use_device_x,
        public_x_stride as u32,
        claims as u32,
        claims as u32,
        u32::MAX,
        surface_count as u32,
        t_core as u32,
        d_pad as u32,
        surface_lanes as u32,
        include_y_zcol,
        write_ct_field,
        write_y_zcol_field,
        plan,
        prefix_fields_start as u32,
        prefix_fields_len as u32,
        prefix_offsets_start as u32,
        prefix_lengths_start as u32,
        commitment_offsets_start as u32,
        commitment_lengths_start as u32,
        x_offsets_start as u32,
        x_lengths_start as u32,
        suffix_fields_start as u32,
        suffix_fields_len as u32,
        suffix_offsets_start as u32,
        suffix_lengths_start as u32,
        preimage_offsets_start as u32,
        preimage_lengths_start as u32,
        preimages_out,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_ccs_build_accumulator_claim_digest_preimages(
    module: &CcsDigestKernelModule,
    stream: &Arc<CudaStream>,
    child_surfaces: &DeviceBuffer<u64>,
    parent_surfaces: &DeviceBuffer<u64>,
    child_commitments: &DeviceBuffer<u64>,
    parent_commitment: &DeviceBuffer<u64>,
    commitment_stride: usize,
    child_public_x: &DeviceBuffer<u64>,
    parent_public_x: &DeviceBuffer<u64>,
    public_x_stride: usize,
    child_claims: usize,
    surface_count: usize,
    t_core: usize,
    d_pad: usize,
    plan: &DeviceBuffer<u64>,
    prefix_fields_start: usize,
    prefix_fields_len: usize,
    prefix_offsets_start: usize,
    prefix_lengths_start: usize,
    commitment_offsets_start: usize,
    commitment_lengths_start: usize,
    x_offsets_start: usize,
    x_lengths_start: usize,
    suffix_fields_start: usize,
    suffix_fields_len: usize,
    suffix_offsets_start: usize,
    suffix_lengths_start: usize,
    preimage_offsets_start: usize,
    preimage_lengths_start: usize,
    preimages_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let claims = child_claims + 2;
    module.ccs_build_output_claim_digest_preimages(
        stream,
        LaunchConfig::for_num_elems(claims as u32),
        child_surfaces,
        parent_surfaces,
        child_commitments,
        parent_commitment,
        true,
        commitment_stride as u32,
        child_public_x,
        parent_public_x,
        public_x_stride != 0,
        public_x_stride as u32,
        claims as u32,
        child_claims as u32,
        (child_claims + 1) as u32,
        surface_count as u32,
        t_core as u32,
        d_pad as u32,
        d_pad as u32,
        false,
        true,
        false,
        plan,
        prefix_fields_start as u32,
        prefix_fields_len as u32,
        prefix_offsets_start as u32,
        prefix_lengths_start as u32,
        commitment_offsets_start as u32,
        commitment_lengths_start as u32,
        x_offsets_start as u32,
        x_lengths_start as u32,
        suffix_fields_start as u32,
        suffix_fields_len as u32,
        suffix_offsets_start as u32,
        suffix_lengths_start as u32,
        preimage_offsets_start as u32,
        preimage_lengths_start as u32,
        preimages_out,
    )
}

pub fn launch_ccs_build_digest_preimage_with_trailer(
    module: &CcsDigestKernelModule,
    stream: &Arc<CudaStream>,
    layout_fields: &DeviceBuffer<u64>,
    header_len: usize,
    claim_digests: &DeviceBuffer<u64>,
    claims: usize,
    preimage_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.ccs_build_digest_preimage_with_trailer(
        stream,
        LaunchConfig::for_num_elems(1),
        layout_fields,
        header_len as u32,
        claim_digests,
        claims as u32,
        preimage_out,
    )
}

pub fn launch_ccs_build_accumulator_digest_preimage(
    module: &CcsDigestKernelModule,
    stream: &Arc<CudaStream>,
    header_fields: &DeviceBuffer<u64>,
    child_digests: &DeviceBuffer<u64>,
    children: usize,
    parent_digest: &DeviceBuffer<u64>,
    preimage_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.ccs_build_accumulator_digest_preimage(
        stream,
        LaunchConfig::for_num_elems(1),
        header_fields,
        child_digests,
        children as u32,
        parent_digest,
        preimage_out,
    )
}

#[cuda_module]
mod pi_ccs_digest_kernels {
    use super::*;

    const DIGEST_LEN: usize = crate::kernels::poseidon2::DIGEST_LEN;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn ccs_build_output_claim_digest_preimages(
        surfaces: &[u64],
        parent_surfaces: &[u64],
        commitment_words: &[u64],
        parent_commitment_words: &[u64],
        use_device_commitments: bool,
        commitment_stride: u32,
        public_x_words: &[u64],
        parent_public_x_words: &[u64],
        use_device_x: bool,
        public_x_stride: u32,
        claims: u32,
        parent_start: u32,
        ce_claim_index: u32,
        surface_count: u32,
        t_core: u32,
        d_pad: u32,
        surface_lanes: u32,
        include_y_zcol: bool,
        write_ct_field: bool,
        write_y_zcol_field: bool,
        plan: &[u64],
        prefix_fields_start: u32,
        prefix_fields_len: u32,
        prefix_offsets_start: u32,
        prefix_lengths_start: u32,
        commitment_offsets_start: u32,
        commitment_lengths_start: u32,
        x_offsets_start: u32,
        x_lengths_start: u32,
        suffix_fields_start: u32,
        suffix_fields_len: u32,
        suffix_offsets_start: u32,
        suffix_lengths_start: u32,
        preimage_offsets_start: u32,
        preimage_lengths_start: u32,
        mut preimages_out: DisjointSlice<u64>,
    ) {
        let claim = thread::index_1d().get();
        let claims = claims as usize;
        let parent_start = parent_start as usize;
        let ce_claim_index = ce_claim_index as usize;
        let commitment_stride = commitment_stride as usize;
        let public_x_stride = public_x_stride as usize;
        let surface_count = surface_count as usize;
        let t_core = t_core as usize;
        let d_pad = d_pad as usize;
        let surface_lanes = surface_lanes as usize;
        let prefix_fields_start = prefix_fields_start as usize;
        let prefix_fields_len = prefix_fields_len as usize;
        let prefix_offsets_start = prefix_offsets_start as usize;
        let prefix_lengths_start = prefix_lengths_start as usize;
        let commitment_offsets_start = commitment_offsets_start as usize;
        let commitment_lengths_start = commitment_lengths_start as usize;
        let x_offsets_start = x_offsets_start as usize;
        let x_lengths_start = x_lengths_start as usize;
        let suffix_fields_start = suffix_fields_start as usize;
        let suffix_fields_len = suffix_fields_len as usize;
        let suffix_offsets_start = suffix_offsets_start as usize;
        let suffix_lengths_start = suffix_lengths_start as usize;
        let preimage_offsets_start = preimage_offsets_start as usize;
        let preimage_lengths_start = preimage_lengths_start as usize;
        if claim >= claims
            || prefix_offsets_start + claim >= plan.len()
            || prefix_lengths_start + claim >= plan.len()
            || commitment_offsets_start + claim >= plan.len()
            || commitment_lengths_start + claim >= plan.len()
            || x_offsets_start + claim >= plan.len()
            || x_lengths_start + claim >= plan.len()
            || suffix_offsets_start + claim >= plan.len()
            || suffix_lengths_start + claim >= plan.len()
            || preimage_offsets_start + claim >= plan.len()
            || preimage_lengths_start + claim >= plan.len()
        {
            return;
        }

        let prefix_start = plan[prefix_offsets_start + claim] as usize;
        let prefix_len = plan[prefix_lengths_start + claim] as usize;
        let suffix_start = plan[suffix_offsets_start + claim] as usize;
        let suffix_len = plan[suffix_lengths_start + claim] as usize;
        let out_start = plan[preimage_offsets_start + claim] as usize;
        let out_len = plan[preimage_lengths_start + claim] as usize;
        let is_parent = claim >= parent_start;
        let source_claim = if is_parent { 0 } else { claim };
        let claim_writes_ct = write_ct_field && claim != ce_claim_index;
        let y_zcol_words = if include_y_zcol { surface_lanes * 2 } else { 0 };
        let ct_field_words = if claim_writes_ct { 1 + t_core * 2 } else { 0 };
        let y_zcol_field_words = if write_y_zcol_field { 1 + y_zcol_words } else { 0 };
        let required =
            prefix_len + 1 + t_core * (1 + surface_lanes * 2) + ct_field_words + y_zcol_field_words + suffix_len;
        if required != out_len
            || prefix_start + prefix_len > prefix_fields_len
            || suffix_start + suffix_len > suffix_fields_len
            || prefix_fields_start + prefix_fields_len > plan.len()
            || suffix_fields_start + suffix_fields_len > plan.len()
            || out_start + out_len > preimages_out.len()
            || surface_count < t_core + usize::from(include_y_zcol)
            || surface_lanes > d_pad
        {
            return;
        }

        let commitment_offset = plan[commitment_offsets_start + claim] as usize;
        let commitment_len = plan[commitment_lengths_start + claim] as usize;
        let x_offset = plan[x_offsets_start + claim] as usize;
        let x_len = plan[x_lengths_start + claim] as usize;
        let commitment_start = source_claim * commitment_stride;
        let x_start = source_claim * public_x_stride;
        let commitment_source_len = if is_parent {
            parent_commitment_words.len()
        } else {
            commitment_words.len()
        };
        let x_source_len = if is_parent {
            parent_public_x_words.len()
        } else {
            public_x_words.len()
        };
        if (use_device_commitments
            && (commitment_stride == 0
                || commitment_len != commitment_stride
                || commitment_offset + commitment_len > prefix_len
                || commitment_start + commitment_len > commitment_source_len))
            || (use_device_x
                && (x_len != public_x_stride || x_offset + x_len > prefix_len || x_start + x_len > x_source_len))
        {
            return;
        }
        let mut dst = out_start;
        for at in 0..prefix_len {
            let value = if use_device_commitments && at >= commitment_offset && at < commitment_offset + commitment_len
            {
                if is_parent {
                    parent_commitment_words[commitment_start + at - commitment_offset]
                } else {
                    commitment_words[commitment_start + at - commitment_offset]
                }
            } else if use_device_x && at >= x_offset && at < x_offset + x_len {
                if is_parent {
                    parent_public_x_words[x_start + at - x_offset]
                } else {
                    public_x_words[x_start + at - x_offset]
                }
            } else {
                plan[prefix_fields_start + prefix_start + at]
            };
            write_word(value, &mut preimages_out, &mut dst);
        }

        write_word(t_core as u64, &mut preimages_out, &mut dst);
        for surface in 0..t_core {
            write_word(surface_lanes as u64, &mut preimages_out, &mut dst);
            for lane in 0..surface_lanes {
                if is_parent {
                    copy_surface_k(
                        parent_surfaces,
                        0,
                        surface_count,
                        surface,
                        d_pad,
                        lane,
                        &mut preimages_out,
                        &mut dst,
                    );
                } else {
                    copy_surface_k(
                        surfaces,
                        source_claim,
                        surface_count,
                        surface,
                        d_pad,
                        lane,
                        &mut preimages_out,
                        &mut dst,
                    );
                }
            }
        }

        if claim_writes_ct {
            write_word(t_core as u64, &mut preimages_out, &mut dst);
            for surface in 0..t_core {
                if is_parent {
                    copy_surface_k(
                        parent_surfaces,
                        0,
                        surface_count,
                        surface,
                        d_pad,
                        0,
                        &mut preimages_out,
                        &mut dst,
                    );
                } else {
                    copy_surface_k(
                        surfaces,
                        source_claim,
                        surface_count,
                        surface,
                        d_pad,
                        0,
                        &mut preimages_out,
                        &mut dst,
                    );
                }
            }
        }

        if write_y_zcol_field {
            if include_y_zcol {
                write_word(surface_lanes as u64, &mut preimages_out, &mut dst);
                for lane in 0..surface_lanes {
                    copy_surface_k(
                        surfaces,
                        claim,
                        surface_count,
                        t_core,
                        d_pad,
                        lane,
                        &mut preimages_out,
                        &mut dst,
                    );
                }
            } else {
                write_word(0, &mut preimages_out, &mut dst);
            }
        }

        copy_slice(
            plan,
            suffix_fields_start + suffix_start,
            suffix_len,
            &mut preimages_out,
            &mut dst,
        );
    }

    #[kernel]
    pub fn ccs_build_digest_preimage_with_trailer(
        layout_fields: &[u64],
        header_len: u32,
        claim_digests: &[u64],
        claims: u32,
        mut preimage_out: DisjointSlice<u64>,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let claims = claims as usize;
        let header_len = header_len as usize;
        let required = layout_fields.len() + claims * DIGEST_LEN;
        if header_len > layout_fields.len()
            || required > preimage_out.len()
            || claims * DIGEST_LEN > claim_digests.len()
        {
            return;
        }
        let mut dst = 0usize;
        copy_slice(layout_fields, 0, header_len, &mut preimage_out, &mut dst);
        copy_slice(claim_digests, 0, claims * DIGEST_LEN, &mut preimage_out, &mut dst);
        copy_slice(
            layout_fields,
            header_len,
            layout_fields.len() - header_len,
            &mut preimage_out,
            &mut dst,
        );
    }

    #[kernel]
    pub fn ccs_build_accumulator_digest_preimage(
        header_fields: &[u64],
        child_digests: &[u64],
        children: u32,
        parent_digest: &[u64],
        mut preimage_out: DisjointSlice<u64>,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let child_words = children as usize * DIGEST_LEN;
        let required = header_fields.len() + child_words + 1 + DIGEST_LEN;
        if required > preimage_out.len() || child_words > child_digests.len() || DIGEST_LEN > parent_digest.len() {
            return;
        }
        let mut dst = 0usize;
        copy_slice(header_fields, 0, header_fields.len(), &mut preimage_out, &mut dst);
        copy_slice(child_digests, 0, child_words, &mut preimage_out, &mut dst);
        write_word(1, &mut preimage_out, &mut dst);
        copy_slice(parent_digest, 0, DIGEST_LEN, &mut preimage_out, &mut dst);
    }

    fn copy_slice(src: &[u64], start: usize, len: usize, dst: &mut DisjointSlice<u64>, out: &mut usize) {
        for i in 0..len {
            unsafe {
                *dst.get_unchecked_mut(*out) = src[start + i];
            }
            *out += 1;
        }
    }

    fn copy_surface_k(
        surfaces: &[u64],
        claim: usize,
        surface_count: usize,
        surface: usize,
        d_pad: usize,
        lane: usize,
        dst: &mut DisjointSlice<u64>,
        out: &mut usize,
    ) {
        let src = ((claim * surface_count + surface) * d_pad + lane) * 2;
        if src + 1 >= surfaces.len() {
            write_word(0, dst, out);
            write_word(0, dst, out);
            return;
        }
        unsafe {
            *dst.get_unchecked_mut(*out) = surfaces[src];
            *dst.get_unchecked_mut(*out + 1) = surfaces[src + 1];
        }
        *out += 2;
    }

    fn write_word(value: u64, dst: &mut DisjointSlice<u64>, out: &mut usize) {
        unsafe {
            *dst.get_unchecked_mut(*out) = value;
        }
        *out += 1;
    }
}
