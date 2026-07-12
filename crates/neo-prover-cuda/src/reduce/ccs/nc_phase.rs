//! Reusable device buffers for the NC phase trace.
//!
//! Owns only phase-log and transcript-I/O buffer lifetime. The NC oracle owns
//! folded digit state; the reductions engine owns the protocol schedule.

use cuda_core::{CudaStream, DeviceBuffer, PinnedHostBuffer};

use crate::device::copy_host_to_device;
use crate::graph::GraphAllocations;

use super::CcsDeviceError;

#[derive(Default)]
pub(crate) struct NcPhaseWorkspace {
    col_coeff_log: Option<DeviceBuffer<u64>>,
    tail_coeff_log: Option<DeviceBuffer<u64>>,
    tail_coeffs_out: Option<DeviceBuffer<u64>>,
    challenges: Option<DeviceBuffer<u64>>,
    prolog_ops: Option<DeviceBuffer<u64>>,
    prolog_payload: Option<DeviceBuffer<u64>>,
    beta: Option<DeviceBuffer<u64>>,
    gamma: Option<DeviceBuffer<u64>>,
    device_payload_dummy: Option<DeviceBuffer<u64>>,
    host_out_dummy: Option<DeviceBuffer<u64>>,
    device_out_dummy: Option<DeviceBuffer<u64>>,
    col_coeff_host: Option<PinnedHostBuffer<u64>>,
    tail_coeff_host: Option<PinnedHostBuffer<u64>>,
}

impl NcPhaseWorkspace {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn prepare_logs(
        &mut self,
        stream: &CudaStream,
        col_coeff_words: usize,
        tail_coeff_words: usize,
        tail_coeffs_out_words: usize,
        challenge_words: usize,
    ) -> Result<(), CcsDeviceError> {
        ensure_len(stream, &mut self.col_coeff_log, col_coeff_words.max(1))?;
        ensure_len(stream, &mut self.tail_coeff_log, tail_coeff_words.max(1))?;
        ensure_len(stream, &mut self.tail_coeffs_out, tail_coeffs_out_words.max(1))?;
        ensure_len(stream, &mut self.challenges, challenge_words.max(1))?;
        Ok(())
    }

    pub(super) fn upload_prolog(
        &mut self,
        stream: &CudaStream,
        op_words: &[u64],
        host_payload: &[u64],
    ) -> Result<(), CcsDeviceError> {
        upload_into(stream, &mut self.prolog_ops, op_words)?;
        upload_into(stream, &mut self.prolog_payload, host_payload)?;
        ensure_len(stream, &mut self.device_payload_dummy, 1)?;
        ensure_len(stream, &mut self.host_out_dummy, 1)?;
        ensure_len(stream, &mut self.device_out_dummy, 1)?;
        Ok(())
    }

    pub(super) fn upload_beta(&mut self, stream: &CudaStream, words: &[u64]) -> Result<(), CcsDeviceError> {
        upload_into(stream, &mut self.beta, words)
    }

    pub(super) fn upload_gamma(&mut self, stream: &CudaStream, words: &[u64; 2]) -> Result<(), CcsDeviceError> {
        upload_into(stream, &mut self.gamma, words)
    }

    pub(super) fn prepare_beta(&mut self, stream: &CudaStream, len: usize) -> Result<(), CcsDeviceError> {
        ensure_len(stream, &mut self.beta, len.max(1))
    }

    pub(super) fn prepare_gamma(&mut self, stream: &CudaStream) -> Result<(), CcsDeviceError> {
        ensure_len(stream, &mut self.gamma, 2)
    }

    pub(super) fn beta_mut(&mut self) -> &mut DeviceBuffer<u64> {
        self.beta.as_mut().expect("NC phase beta buffer prepared")
    }

    pub(super) fn gamma_mut(&mut self) -> &mut DeviceBuffer<u64> {
        self.gamma.as_mut().expect("NC phase gamma buffer prepared")
    }

    pub(super) fn prolog_io_buffers(
        &mut self,
    ) -> (
        &DeviceBuffer<u64>,
        &DeviceBuffer<u64>,
        &DeviceBuffer<u64>,
        &mut DeviceBuffer<u64>,
        &mut DeviceBuffer<u64>,
    ) {
        (
            self.prolog_ops
                .as_ref()
                .expect("NC phase prolog op buffer prepared"),
            self.prolog_payload
                .as_ref()
                .expect("NC phase prolog payload prepared"),
            self.device_payload_dummy
                .as_ref()
                .expect("NC phase device payload dummy prepared"),
            self.host_out_dummy
                .as_mut()
                .expect("NC phase host output dummy prepared"),
            self.device_out_dummy
                .as_mut()
                .expect("NC phase device output dummy prepared"),
        )
    }

    pub(super) fn col_coeff_log_mut(&mut self) -> &mut DeviceBuffer<u64> {
        self.col_coeff_log
            .as_mut()
            .expect("NC phase column coeff log prepared")
    }

    pub(super) fn tail_coeffs_out_mut(&mut self) -> &mut DeviceBuffer<u64> {
        self.tail_coeffs_out
            .as_mut()
            .expect("NC phase tail coeffs-out buffer prepared")
    }

    pub(super) fn tail_coeffs_and_log_mut(&mut self) -> (&DeviceBuffer<u64>, &mut DeviceBuffer<u64>) {
        (
            self.tail_coeffs_out
                .as_ref()
                .expect("NC phase tail coeffs-out buffer prepared"),
            self.tail_coeff_log
                .as_mut()
                .expect("NC phase tail coeff log prepared"),
        )
    }

    pub(super) fn tail_coeffs_and_challenges_mut(&mut self) -> (&DeviceBuffer<u64>, &mut DeviceBuffer<u64>) {
        (
            self.tail_coeffs_out
                .as_ref()
                .expect("NC phase tail coeffs-out buffer prepared"),
            self.challenges
                .as_mut()
                .expect("NC phase challenge log prepared"),
        )
    }

    pub(super) fn challenges_mut(&mut self) -> &mut DeviceBuffer<u64> {
        self.challenges
            .as_mut()
            .expect("NC phase challenge log prepared")
    }

    pub(super) fn beta(&self) -> &DeviceBuffer<u64> {
        self.beta.as_ref().expect("NC phase beta buffer prepared")
    }

    pub(super) fn gamma(&self) -> &DeviceBuffer<u64> {
        self.gamma.as_ref().expect("NC phase gamma buffer prepared")
    }

    pub(super) fn col_coeff_log(&self) -> &DeviceBuffer<u64> {
        self.col_coeff_log
            .as_ref()
            .expect("NC phase column coeff log prepared")
    }

    pub(super) fn tail_coeff_log(&self) -> &DeviceBuffer<u64> {
        self.tail_coeff_log
            .as_ref()
            .expect("NC phase tail coeff log prepared")
    }

    pub(super) fn challenges(&self) -> &DeviceBuffer<u64> {
        self.challenges
            .as_ref()
            .expect("NC phase challenge log prepared")
    }

    pub(super) fn take_col_coeff_host(
        &mut self,
        stream: &CudaStream,
        len: usize,
    ) -> Result<PinnedHostBuffer<u64>, CcsDeviceError> {
        take_pinned(&mut self.col_coeff_host, stream, len)
    }

    pub(super) fn store_col_coeff_host(&mut self, buffer: PinnedHostBuffer<u64>) {
        self.col_coeff_host = Some(buffer);
    }

    pub(super) fn take_tail_coeff_host(
        &mut self,
        stream: &CudaStream,
        len: usize,
    ) -> Result<PinnedHostBuffer<u64>, CcsDeviceError> {
        take_pinned(&mut self.tail_coeff_host, stream, len)
    }

    pub(super) fn store_tail_coeff_host(&mut self, buffer: PinnedHostBuffer<u64>) {
        self.tail_coeff_host = Some(buffer);
    }

    pub(super) fn record_graph_allocations(&self, allocations: &mut GraphAllocations) {
        record_buffer(&self.col_coeff_log, allocations);
        record_buffer(&self.tail_coeff_log, allocations);
        record_buffer(&self.tail_coeffs_out, allocations);
        record_buffer(&self.challenges, allocations);
        record_buffer(&self.prolog_ops, allocations);
        record_buffer(&self.prolog_payload, allocations);
        record_buffer(&self.beta, allocations);
        record_buffer(&self.gamma, allocations);
        record_buffer(&self.device_payload_dummy, allocations);
        record_buffer(&self.host_out_dummy, allocations);
        record_buffer(&self.device_out_dummy, allocations);
    }
}

fn record_buffer(slot: &Option<DeviceBuffer<u64>>, allocations: &mut GraphAllocations) {
    if let Some(buffer) = slot {
        allocations.push(buffer);
    }
}

fn ensure_len(stream: &CudaStream, slot: &mut Option<DeviceBuffer<u64>>, len: usize) -> Result<(), CcsDeviceError> {
    if slot.as_ref().is_some_and(|buffer| buffer.len() >= len) {
        return Ok(());
    }
    *slot = Some(DeviceBuffer::zeroed(stream, len)?);
    Ok(())
}

fn take_pinned(
    slot: &mut Option<PinnedHostBuffer<u64>>,
    stream: &CudaStream,
    len: usize,
) -> Result<PinnedHostBuffer<u64>, CcsDeviceError> {
    let len = len.max(1);
    if slot.as_ref().is_none_or(|buffer| buffer.len() < len) {
        return Ok(PinnedHostBuffer::zeroed(stream.context(), len)?);
    }
    Ok(slot.take().expect("checked above"))
}

fn upload_into(stream: &CudaStream, slot: &mut Option<DeviceBuffer<u64>>, words: &[u64]) -> Result<(), CcsDeviceError> {
    let len = words.len().max(1);
    ensure_len(stream, slot, len)?;
    let buffer = slot.as_ref().expect("buffer prepared above");
    if words.is_empty() {
        copy_host_to_device(stream, buffer, &[0])?;
    } else {
        copy_host_to_device(stream, buffer, words)?;
    }
    Ok(())
}
