//! Reusable device buffers for the whole-FE Π_CCS path.
//!
//! Owns only buffer lifetime. The FE backend decides what kernels to enqueue
//! and the optimized engine remains the protocol owner.

use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer, PinnedHostBuffer};
use neo_math::F;

use crate::device::{copy_host_to_device, uninit_u64_device_buffer, Device};
use crate::graph::{CapturedGraph, GraphAllocations};
use crate::kernels::poseidon2::{Poseidon2KernelModule, WIDTH};
use crate::transcript::{DeviceIoSlots, DeviceTranscript};

use super::{CcsDeviceError, NcPhaseGraphKey};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FePhaseGraphKey {
    pub(super) width: usize,
    pub(super) row_rounds: usize,
    pub(super) tail_rounds: usize,
    pub(super) table_count: usize,
    pub(super) table_stride: usize,
    pub(super) active_len: usize,
    pub(super) cur_len: usize,
    pub(super) y_eval_witnesses: usize,
    pub(super) y_eval_matrices: usize,
    pub(super) tail_partial_count: usize,
    pub(super) has_inputs: bool,
    pub(super) allocations: GraphAllocations,
}

struct FePhaseGraph {
    key: FePhaseGraphKey,
    graph: CapturedGraph,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PiCcsPhaseGraphKey {
    pub(super) fe: FePhaseGraphKey,
    pub(super) nc: NcPhaseGraphKey,
}

struct PiCcsPhaseGraph {
    key: PiCcsPhaseGraphKey,
    graph: CapturedGraph,
}

pub(crate) struct FePhaseWorkspace {
    coeff_log: Option<DeviceBuffer<u64>>,
    challenges: Option<DeviceBuffer<u64>>,
    points: Option<DeviceBuffer<u64>>,
    tail_headers: Option<DeviceBuffer<u64>>,
    tail_partials: Option<DeviceBuffer<u64>>,
    tail_partial_scratch: Option<DeviceBuffer<u64>>,
    tail_inner_sums: Option<DeviceBuffer<u64>>,
    transcript: Option<DeviceTranscript>,
    forms_chi: Option<DeviceBuffer<u64>>,
    forms: Option<DeviceBuffer<u64>>,
    y_eval_words: Option<DeviceBuffer<u64>>,
    coeff_log_host: Option<PinnedHostBuffer<u64>>,
    challenges_host: Option<PinnedHostBuffer<u64>>,
    y_eval_host: Option<PinnedHostBuffer<u64>>,
    graph: Option<FePhaseGraph>,
    pi_ccs_graph: Option<PiCcsPhaseGraph>,
}

impl FePhaseWorkspace {
    pub(super) fn new() -> Self {
        Self {
            coeff_log: None,
            challenges: None,
            points: None,
            tail_headers: None,
            tail_partials: None,
            tail_partial_scratch: None,
            tail_inner_sums: None,
            transcript: None,
            forms_chi: None,
            forms: None,
            y_eval_words: None,
            coeff_log_host: None,
            challenges_host: None,
            y_eval_host: None,
            graph: None,
            pi_ccs_graph: None,
        }
    }

    pub(super) fn prepare_logs(
        &mut self,
        stream: &Arc<CudaStream>,
        coeff_words: usize,
        challenge_words: usize,
    ) -> Result<(), CcsDeviceError> {
        ensure_uninit_len(stream, &mut self.coeff_log, coeff_words.max(1))?;
        ensure_uninit_len(stream, &mut self.challenges, challenge_words.max(1))?;
        Ok(())
    }

    pub(super) fn prepare_tail_scratch(
        &mut self,
        stream: &Arc<CudaStream>,
        partial_words: usize,
        scratch_words: usize,
        inner_sum_words: usize,
    ) -> Result<(), CcsDeviceError> {
        ensure_uninit_len(stream, &mut self.tail_partials, partial_words.max(1))?;
        ensure_uninit_len(stream, &mut self.tail_partial_scratch, scratch_words.max(1))?;
        ensure_uninit_len(stream, &mut self.tail_inner_sums, inner_sum_words.max(1))?;
        Ok(())
    }

    pub(super) fn prepare_y_eval_surface(
        &mut self,
        stream: &Arc<CudaStream>,
        chi_words: usize,
        form_words: usize,
        y_eval_words: usize,
    ) -> Result<(), CcsDeviceError> {
        ensure_uninit_len(stream, &mut self.forms_chi, chi_words.max(1))?;
        ensure_uninit_len(stream, &mut self.forms, form_words.max(1))?;
        ensure_uninit_len(stream, &mut self.y_eval_words, y_eval_words.max(1))?;
        Ok(())
    }

    pub(super) fn upload_points(&mut self, stream: &Arc<CudaStream>, words: &[u64]) -> Result<(), CcsDeviceError> {
        upload_into(stream, &mut self.points, words)
    }

    pub(super) fn prepare_points(&mut self, stream: &Arc<CudaStream>, len: usize) -> Result<(), CcsDeviceError> {
        ensure_uninit_len(stream, &mut self.points, len.max(1))
    }

    pub(super) fn points_mut(&mut self) -> &mut DeviceBuffer<u64> {
        self.points.as_mut().expect("FE phase points prepared")
    }

    pub(super) fn upload_tail_headers(
        &mut self,
        stream: &Arc<CudaStream>,
        words: &[u64],
    ) -> Result<(), CcsDeviceError> {
        upload_into(stream, &mut self.tail_headers, words)
    }

    pub(super) fn reset_transcript(
        &mut self,
        device: &Device,
        state: [F; WIDTH],
        absorbed: usize,
    ) -> Result<(), CcsDeviceError> {
        match self.transcript.as_mut() {
            Some(transcript) => transcript.reset_state_and_absorbed(device, state, absorbed)?,
            None => {
                self.transcript = Some(DeviceTranscript::from_state_and_absorbed(device, state, absorbed)?);
            }
        }
        Ok(())
    }

    pub(super) fn enqueue_coeff_challenge(
        &mut self,
        device: &Device,
        module: &Poseidon2KernelModule,
        rc: &DeviceBuffer<u64>,
        coeffs: &DeviceBuffer<u64>,
        coeff_words: usize,
        challenge_offset: usize,
    ) -> Result<(), CcsDeviceError> {
        let transcript = self
            .transcript
            .as_mut()
            .expect("FE phase transcript prepared");
        let challenges = self
            .challenges
            .as_mut()
            .expect("FE phase challenge log prepared");
        transcript.enqueue_absorb_device_challenge(
            device,
            module,
            rc,
            coeff_words as u64,
            DeviceIoSlots {
                payload: coeffs,
                payload_offset: 0,
                payload_len: coeff_words,
                out: challenges,
                out_offset: challenge_offset,
            },
        )?;
        Ok(())
    }

    pub(super) fn transcript_state_words_to_host(&self, device: &Device) -> Result<Vec<u64>, CcsDeviceError> {
        Ok(self
            .transcript
            .as_ref()
            .expect("FE phase transcript prepared")
            .state_words_to_host(device)?)
    }

    pub(super) fn transcript_mut(&mut self) -> &mut DeviceTranscript {
        self.transcript
            .as_mut()
            .expect("FE phase transcript prepared")
    }

    pub(super) fn coeff_log_mut(&mut self) -> &mut DeviceBuffer<u64> {
        self.coeff_log
            .as_mut()
            .expect("FE phase coeff log prepared")
    }

    pub(super) fn coeff_log(&self) -> &DeviceBuffer<u64> {
        self.coeff_log
            .as_ref()
            .expect("FE phase coeff log prepared")
    }

    pub(super) fn challenges(&self) -> &DeviceBuffer<u64> {
        self.challenges
            .as_ref()
            .expect("FE phase challenge log prepared")
    }

    pub(super) fn cooperative_row_round_buffers(
        &mut self,
    ) -> (&mut DeviceBuffer<u64>, &mut DeviceBuffer<u64>, &mut DeviceBuffer<u64>) {
        (
            self.transcript
                .as_mut()
                .expect("FE phase transcript prepared")
                .state_words_mut(),
            self.coeff_log
                .as_mut()
                .expect("FE phase coeff log prepared"),
            self.challenges
                .as_mut()
                .expect("FE phase challenge log prepared"),
        )
    }

    pub(super) fn take_coeff_log_host_for_stream(
        &mut self,
        stream: &CudaStream,
        len: usize,
    ) -> Result<PinnedHostBuffer<u64>, CcsDeviceError> {
        take_pinned_for_stream(&mut self.coeff_log_host, stream, len)
    }

    pub(super) fn store_coeff_log_host(&mut self, buffer: PinnedHostBuffer<u64>) {
        self.coeff_log_host = Some(buffer);
    }

    pub(super) fn take_challenges_host(
        &mut self,
        device: &Device,
        len: usize,
    ) -> Result<PinnedHostBuffer<u64>, CcsDeviceError> {
        take_pinned(&mut self.challenges_host, device, len)
    }

    pub(super) fn store_challenges_host(&mut self, buffer: PinnedHostBuffer<u64>) {
        self.challenges_host = Some(buffer);
    }

    pub(super) fn take_y_eval_host(
        &mut self,
        device: &Device,
        len: usize,
    ) -> Result<PinnedHostBuffer<u64>, CcsDeviceError> {
        take_pinned(&mut self.y_eval_host, device, len)
    }

    pub(super) fn store_y_eval_host(&mut self, buffer: PinnedHostBuffer<u64>) {
        self.y_eval_host = Some(buffer);
    }

    pub(super) fn tail_round_buffers(
        &mut self,
    ) -> (
        &DeviceBuffer<u64>,
        &DeviceBuffer<u64>,
        &DeviceBuffer<u64>,
        &mut DeviceBuffer<u64>,
        &mut DeviceBuffer<u64>,
        &mut DeviceBuffer<u64>,
    ) {
        (
            self.points.as_ref().expect("FE phase points prepared"),
            self.tail_headers
                .as_ref()
                .expect("FE phase tail headers prepared"),
            self.challenges
                .as_ref()
                .expect("FE phase challenge log prepared"),
            self.tail_partials
                .as_mut()
                .expect("FE phase tail partials prepared"),
            self.tail_partial_scratch
                .as_mut()
                .expect("FE phase tail scratch prepared"),
            self.tail_inner_sums
                .as_mut()
                .expect("FE phase tail inner sums prepared"),
        )
    }

    pub(super) fn forms_buffers(&mut self) -> (&mut DeviceBuffer<u64>, &mut DeviceBuffer<u64>) {
        (
            self.forms_chi
                .as_mut()
                .expect("FE phase forms chi buffer prepared"),
            self.forms.as_mut().expect("FE phase forms buffer prepared"),
        )
    }

    pub(super) fn take_forms(&mut self) -> DeviceBuffer<u64> {
        self.forms.take().expect("FE phase forms buffer prepared")
    }

    pub(crate) fn store_forms(&mut self, forms: DeviceBuffer<u64>) {
        self.forms = Some(forms);
    }

    pub(super) fn challenge_and_forms_buffers(
        &mut self,
    ) -> (&DeviceBuffer<u64>, &mut DeviceBuffer<u64>, &mut DeviceBuffer<u64>) {
        (
            self.challenges
                .as_ref()
                .expect("FE phase challenge log prepared"),
            self.forms_chi
                .as_mut()
                .expect("FE phase forms chi buffer prepared"),
            self.forms.as_mut().expect("FE phase forms buffer prepared"),
        )
    }

    pub(super) fn take_y_eval_words(&mut self) -> DeviceBuffer<u64> {
        self.y_eval_words
            .take()
            .expect("FE phase Y_eval buffer prepared")
    }

    pub(crate) fn store_y_eval_words(&mut self, words: DeviceBuffer<u64>) {
        self.y_eval_words = Some(words);
    }

    pub(super) fn launch_graph_if_matching(
        &self,
        stream: &CudaStream,
        key: &FePhaseGraphKey,
    ) -> Result<bool, CcsDeviceError> {
        let Some(graph) = self.graph.as_ref().filter(|graph| &graph.key == key) else {
            return Ok(false);
        };
        graph.graph.launch(stream)?;
        Ok(true)
    }

    pub(super) fn store_graph(&mut self, key: FePhaseGraphKey, graph: CapturedGraph) {
        self.graph = Some(FePhaseGraph { key, graph });
    }

    pub(super) fn launch_pi_ccs_graph_if_matching(
        &self,
        stream: &CudaStream,
        key: &PiCcsPhaseGraphKey,
    ) -> Result<bool, CcsDeviceError> {
        let Some(graph) = self.pi_ccs_graph.as_ref().filter(|graph| &graph.key == key) else {
            return Ok(false);
        };
        graph.graph.launch(stream)?;
        Ok(true)
    }

    pub(super) fn store_pi_ccs_graph(&mut self, key: PiCcsPhaseGraphKey, graph: CapturedGraph) {
        self.pi_ccs_graph = Some(PiCcsPhaseGraph { key, graph });
    }

    pub(super) fn record_graph_allocations(&self, allocations: &mut GraphAllocations) {
        record_buffer(&self.coeff_log, allocations);
        record_buffer(&self.challenges, allocations);
        record_buffer(&self.points, allocations);
        record_buffer(&self.tail_headers, allocations);
        record_buffer(&self.tail_partials, allocations);
        record_buffer(&self.tail_partial_scratch, allocations);
        record_buffer(&self.tail_inner_sums, allocations);
        record_buffer(&self.forms_chi, allocations);
        record_buffer(&self.forms, allocations);
        record_buffer(&self.y_eval_words, allocations);
        if let Some(transcript) = &self.transcript {
            transcript.record_graph_allocations(allocations);
        }
    }
}

fn record_buffer(slot: &Option<DeviceBuffer<u64>>, allocations: &mut GraphAllocations) {
    if let Some(buffer) = slot {
        allocations.push(buffer);
    }
}

fn ensure_uninit_len(
    stream: &Arc<CudaStream>,
    slot: &mut Option<DeviceBuffer<u64>>,
    len: usize,
) -> Result<(), CcsDeviceError> {
    if slot.as_ref().is_some_and(|buffer| buffer.len() >= len) {
        return Ok(());
    }
    *slot = Some(uninit_u64_device_buffer(stream, len)?);
    Ok(())
}

fn upload_into(
    stream: &Arc<CudaStream>,
    slot: &mut Option<DeviceBuffer<u64>>,
    words: &[u64],
) -> Result<(), CcsDeviceError> {
    let len = words.len().max(1);
    ensure_uninit_len(stream, slot, len)?;
    let buffer = slot.as_ref().expect("buffer prepared above");
    if words.is_empty() {
        copy_host_to_device(stream, buffer, &[0])?;
    } else {
        copy_host_to_device(stream, buffer, words)?;
    }
    Ok(())
}

fn take_pinned(
    slot: &mut Option<PinnedHostBuffer<u64>>,
    device: &Device,
    len: usize,
) -> Result<PinnedHostBuffer<u64>, CcsDeviceError> {
    take_pinned_for_context(slot, device.ctx(), len)
}

fn take_pinned_for_stream(
    slot: &mut Option<PinnedHostBuffer<u64>>,
    stream: &CudaStream,
    len: usize,
) -> Result<PinnedHostBuffer<u64>, CcsDeviceError> {
    take_pinned_for_context(slot, stream.context(), len)
}

fn take_pinned_for_context(
    slot: &mut Option<PinnedHostBuffer<u64>>,
    ctx: &std::sync::Arc<cuda_core::CudaContext>,
    len: usize,
) -> Result<PinnedHostBuffer<u64>, CcsDeviceError> {
    let len = len.max(1);
    if slot.as_ref().is_none_or(|buffer| buffer.len() < len) {
        return Ok(PinnedHostBuffer::zeroed(ctx, len)?);
    }
    Ok(slot.take().expect("checked above"))
}
