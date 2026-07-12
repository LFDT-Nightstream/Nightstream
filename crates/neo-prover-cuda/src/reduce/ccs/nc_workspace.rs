//! Reusable device buffers for the NC column oracle.
//!
//! Owns only allocation lifetime. `DeviceNcOracle` owns the column-round
//! state while a prove is active, then returns these buffers to the workspace.

use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer, PinnedHostBuffer};

use crate::device::{uninit_u64_device_buffer, Device};

use super::CcsDeviceError;

#[derive(Default)]
pub(crate) struct NcOracleWorkspace {
    pub(super) digits_a: Option<DeviceBuffer<u64>>,
    pub(super) digits_b: Option<DeviceBuffer<u64>>,
    pub(super) eq_a: Option<DeviceBuffer<u64>>,
    pub(super) eq_b: Option<DeviceBuffer<u64>>,
    pub(super) weights: Option<DeviceBuffer<u64>>,
    pub(super) inner_partials: Option<DeviceBuffer<u64>>,
    pub(super) partials: Option<DeviceBuffer<u64>>,
    pub(super) sum_scratch: Option<DeviceBuffer<u64>>,
    pub(super) coeffs_out: Option<DeviceBuffer<u64>>,
    pub(super) coeffs_host: Option<PinnedHostBuffer<u64>>,
}

impl NcOracleWorkspace {
    pub(super) fn new() -> Self {
        Self::default()
    }
}

pub(super) fn take_buffer(
    slot: &mut Option<DeviceBuffer<u64>>,
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
    let len = len.max(1);
    if slot.as_ref().is_none_or(|buffer| buffer.len() < len) {
        // Every NC oracle buffer taken through this workspace is either
        // fully overwritten before read, or read through an explicit live
        // length that treats stale headroom as zero.
        return Ok(uninit_u64_device_buffer(stream, len)?);
    }
    Ok(slot.take().expect("checked above"))
}

pub(super) fn store_buffer(slot: &mut Option<DeviceBuffer<u64>>, buffer: DeviceBuffer<u64>) {
    *slot = Some(buffer);
}

pub(super) fn take_pinned(
    slot: &mut Option<PinnedHostBuffer<u64>>,
    device: &Device,
    len: usize,
) -> Result<PinnedHostBuffer<u64>, CcsDeviceError> {
    let len = len.max(1);
    if slot.as_ref().is_none_or(|buffer| buffer.len() < len) {
        return Ok(PinnedHostBuffer::zeroed(device.ctx(), len)?);
    }
    Ok(slot.take().expect("checked above"))
}

pub(super) fn store_pinned(slot: &mut Option<PinnedHostBuffer<u64>>, buffer: PinnedHostBuffer<u64>) {
    *slot = Some(buffer);
}
