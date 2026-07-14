//! Plane staging: how witness data arrives on device each fold.
//!
//! Owns the fold-planes buffer layout (`[count][cols * D]`, fresh then
//! running), the host flattening, the fresh H2D upload, the resident
//! device-to-device composition, and the host download of a single
//! witness. Owns no reduction semantics.

use cuda_core::DeviceBuffer;
use neo_ccs::Mat;
use neo_math::{D, F};

use crate::device::{copy_host_to_device, upload_u64_device_buffer, zeroed_u64_device_buffer, Device};
use crate::kernels::ajtai::{launch_plane_copy, AjtaiKernelModule};
use crate::reduce::ccs::CcsDeviceError;
use crate::ring_layout;

/// One fold's witness planes, flattened and uploaded as a single
/// `[count][cols * D]` device buffer in caller order. Sessions build this
/// once per fold and hand it to every consumer (Ajtai `Y_eval`, Π_RLC mix).
pub fn upload_witness_planes(device: &Device, witnesses: &[&Mat<F>]) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
    let z_words = flatten_planes(witnesses)?;
    Ok(upload_u64_device_buffer(device.stream(), &z_words)?)
}

pub(crate) fn upload_witness_planes_into(
    device: &Device,
    witnesses: &[&Mat<F>],
    out: &mut Option<DeviceBuffer<u64>>,
) -> Result<(), CcsDeviceError> {
    let z_words = flatten_planes(witnesses)?;
    let buffer = ensure_plane_buffer(device, out, z_words.len())?;
    copy_host_to_device(device.stream(), buffer, &z_words)?;
    Ok(())
}

/// [`upload_witness_planes`] where the trailing planes are already on
/// device: uploads only the `fresh` witnesses and appends `resident`
/// (the retained Π_DEC split planes — byte-equal to `mat_to_words` of the
/// running child witnesses) with a device-to-device copy. The result is
/// bit-identical to uploading `fresh ++ running` from host.
pub fn compose_fold_planes(
    device: &Device,
    ring: &AjtaiKernelModule,
    fresh: &[&Mat<F>],
    resident: &DeviceBuffer<u64>,
) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
    let stream = device.stream();
    let fresh_words;
    crate::perf_timed!("fold.ingest.layout", {
        fresh_words = flatten_planes(fresh)?;
    });
    let fresh_len = fresh_words.len();
    let fresh_dev;
    crate::perf_timed!("fold.ingest.fresh", {
        fresh_dev = upload_u64_device_buffer(stream, &fresh_words)?;
    });
    let mut out;
    crate::perf_timed!("fold.ingest.running", {
        out = zeroed_u64_device_buffer(stream, fresh_len + resident.len())?;
        launch_plane_copy(ring, stream, &fresh_dev, 0, &mut out)?;
        launch_plane_copy(ring, stream, resident, fresh_len, &mut out)?;
    });
    Ok(out)
}

pub(crate) fn compose_fold_planes_into(
    device: &Device,
    ring: &AjtaiKernelModule,
    fresh: &[&Mat<F>],
    resident: &DeviceBuffer<u64>,
    out: &mut Option<DeviceBuffer<u64>>,
) -> Result<(), CcsDeviceError> {
    let stream = device.stream();
    let fresh_words;
    crate::perf_timed!("fold.ingest.layout", {
        fresh_words = flatten_planes(fresh)?;
    });
    let fresh_len = fresh_words.len();
    let fresh_dev;
    crate::perf_timed!("fold.ingest.fresh", {
        fresh_dev = upload_u64_device_buffer(stream, &fresh_words)?;
    });
    crate::perf_timed!("fold.ingest.running", {
        let out = ensure_plane_buffer(device, out, fresh_len + resident.len())?;
        launch_plane_copy(ring, stream, &fresh_dev, 0, out)?;
        launch_plane_copy(ring, stream, resident, fresh_len, out)?;
    });
    Ok(())
}

/// Compose fresh and running planes that are both already resident.
///
/// This is the cross-fold version of [`compose_fold_planes_into`]: the fresh
/// planes were uploaded once while building their Ajtai commitments, and the
/// running planes are the previous Π_DEC children retained for the next fold.
pub(crate) fn compose_resident_fold_planes_into(
    device: &Device,
    ring: &AjtaiKernelModule,
    fresh: &DeviceBuffer<u64>,
    resident: &DeviceBuffer<u64>,
    out: &mut Option<DeviceBuffer<u64>>,
) -> Result<(), CcsDeviceError> {
    if fresh.len() == 0 || resident.len() == 0 {
        return Err(CcsDeviceError::Shape("resident fold-plane pieces must be nonempty"));
    }
    let stream = device.stream();
    crate::perf_timed!("fold.ingest.running", {
        let out = ensure_plane_buffer(device, out, fresh.len() + resident.len())?;
        launch_plane_copy(ring, stream, fresh, 0, out)?;
        launch_plane_copy(ring, stream, resident, fresh.len(), out)?;
    });
    Ok(())
}

/// The shared `[count][cols * D]` host flattening behind the plane uploads.
fn flatten_planes(witnesses: &[&Mat<F>]) -> Result<Vec<u64>, CcsDeviceError> {
    if witnesses.is_empty() {
        return Err(CcsDeviceError::Shape("witness planes need at least one witness"));
    }
    let cols = witnesses[0].cols();
    if witnesses.iter().any(|w| w.rows() != D || w.cols() != cols) {
        return Err(CcsDeviceError::Shape("witnesses must be D × shared width"));
    }
    let wit_stride = cols * D;
    let mut z_words = vec![0u64; witnesses.len() * wit_stride];
    for (i, witness) in witnesses.iter().enumerate() {
        let words = ring_layout::mat_to_words(witness);
        z_words[i * wit_stride..(i + 1) * wit_stride].copy_from_slice(&words);
    }
    Ok(z_words)
}

fn ensure_plane_buffer<'a>(
    device: &Device,
    slot: &'a mut Option<DeviceBuffer<u64>>,
    len: usize,
) -> Result<&'a mut DeviceBuffer<u64>, CcsDeviceError> {
    if slot.as_ref().is_none_or(|buffer| buffer.len() < len) {
        crate::perf_timed!("session.buffers", {
            *slot = Some(zeroed_u64_device_buffer(device.stream(), len)?);
        });
    }
    Ok(slot.as_mut().expect("fold-plane buffer prepared"))
}
