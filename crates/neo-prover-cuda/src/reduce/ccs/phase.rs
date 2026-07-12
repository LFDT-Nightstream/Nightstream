//! Whole-channel Π_CCS device scheduling.
//!
//! Owns the coarse FE/NC device transcript flow. The optimized reductions
//! engine remains the protocol owner and validates every returned coefficient
//! and challenge.

use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer, PinnedHostBuffer};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::oracle::{NcColSnapshot, RowPhaseSnapshot};
use neo_reductions::optimized_engine::{
    Challenges, FePhaseTraceRequest, FeRowRoundTrace, FeSumcheckBackend, NcFinalizedColState, NcSumcheckBackend,
    PiCcsPhaseBackend, PiCcsPhaseProofLog, PiCcsPhaseSummary, PiCcsPhaseTrace, PiCcsPhaseTraceRequest,
    PiCcsTerminalOutputSurfaces, TranscriptSnapshot,
};
use p3_field::PrimeCharacteristicRing;

use crate::device::{uninit_u64_device_buffer, upload_u64_device_buffer};
use crate::field::k_from_device_words;
use crate::graph::{CaptureError, CapturedGraph, GraphAllocations};
use crate::kernels::ajtai::{launch_plane_copy, launch_plane_copy_slice, RingMatVecScratch};
use crate::kernels::pi_ccs_nc::NC_COEFFS;
use crate::kernels::sumcheck_common::SUM_BLOCKS;
use crate::reduce::ccs::{
    CcsDeviceError, DeviceAjtaiYEval, DeviceFeBackend, DeviceNcBackend, DevicePublicChallenges, FeOracleWorkspace,
    FePhaseGraphKey, FePhaseWorkspace, NcOracleWorkspace, NcPhaseWorkspace, PendingNcPhase, PiCcsPhaseGraphKey,
    SumcheckKernels,
};
use crate::ring_forms::{DeviceBarMatrices, DeviceRowMatrices};
use crate::transcript::{encode_transcript_io_ops, DeviceTranscript, TranscriptIoOp};

use super::fe::TAIL_CHALLENGE_HEADER_WORDS;

pub struct DevicePiCcsPhaseBackend<'a> {
    fe: DeviceFeBackend<'a>,
    nc: DeviceNcBackend<'a>,
    public_challenges: Option<DevicePublicChallenges>,
}

#[allow(dead_code)]
impl<'a> DevicePiCcsPhaseBackend<'a> {
    pub fn new(device: &'a crate::device::Device, kernels: &'a SumcheckKernels) -> Self {
        Self {
            fe: DeviceFeBackend::new(device, kernels),
            nc: DeviceNcBackend::new(device, kernels),
            public_challenges: None,
        }
    }

    pub fn set_statics(&mut self, bar: Option<DeviceBarMatrices>, rows: Option<DeviceRowMatrices>) {
        self.fe.set_statics(bar, rows);
    }

    pub fn take_statics(&mut self) -> (Option<DeviceBarMatrices>, Option<DeviceRowMatrices>) {
        self.fe.take_statics()
    }

    pub(crate) fn set_phase_workspace(&mut self, workspace: Option<FePhaseWorkspace>) {
        self.fe.set_phase_workspace(workspace);
    }

    pub(crate) fn take_phase_workspace(&mut self) -> Option<FePhaseWorkspace> {
        self.fe.take_phase_workspace()
    }

    pub(crate) fn set_oracle_workspace(&mut self, workspace: Option<FeOracleWorkspace>) {
        self.fe.set_oracle_workspace(workspace);
    }

    pub(crate) fn take_oracle_workspace(&mut self) -> Option<FeOracleWorkspace> {
        self.fe.take_oracle_workspace()
    }

    pub(crate) fn set_nc_oracle_workspace(&mut self, workspace: Option<NcOracleWorkspace>) {
        self.nc.set_oracle_workspace(workspace);
    }

    pub(crate) fn take_nc_oracle_workspace(&mut self) -> Option<NcOracleWorkspace> {
        self.nc.take_oracle_workspace()
    }

    pub(crate) fn set_nc_phase_workspace(&mut self, workspace: Option<NcPhaseWorkspace>) {
        self.nc.set_phase_workspace(workspace);
    }

    pub(crate) fn take_nc_phase_workspace(&mut self) -> Option<NcPhaseWorkspace> {
        self.nc.take_phase_workspace()
    }

    pub(crate) fn take_last_y_eval_surface(&mut self) -> Option<DeviceAjtaiYEval> {
        self.fe.take_last_y_eval_surface()
    }

    pub(crate) fn take_last_nc_final_state(&mut self) -> Option<crate::reduce::ccs::DeviceNcFinalState> {
        self.nc.take_last_final_state()
    }

    pub(crate) fn set_ring_scratch(&mut self, scratch: Option<RingMatVecScratch>) {
        self.fe.set_ring_scratch(scratch);
    }

    pub(crate) fn take_ring_scratch(&mut self) -> Option<RingMatVecScratch> {
        self.fe.take_ring_scratch()
    }

    pub(crate) fn take_proof_log_exporter(&mut self) -> Option<DevicePiCcsProofLogExporter> {
        if self.fe.last_phase_log_shape.is_none()
            || self.fe.phase_workspace.is_none()
            || self.nc.last_phase_log_shape.is_none()
            || self.nc.phase_workspace.is_none()
        {
            return None;
        }

        Some(DevicePiCcsProofLogExporter {
            stream: Arc::clone(self.fe.device.stream()),
            fe: Some(FeProofLogExport {
                shape: self
                    .fe
                    .last_phase_log_shape
                    .take()
                    .expect("FE proof-log shape checked"),
                workspace: self
                    .fe
                    .phase_workspace
                    .take()
                    .expect("FE proof-log workspace checked"),
            }),
            nc: Some(NcProofLogExport {
                shape: self
                    .nc
                    .last_phase_log_shape
                    .take()
                    .expect("NC proof-log shape checked"),
                workspace: self
                    .nc
                    .phase_workspace
                    .take()
                    .expect("NC proof-log workspace checked"),
            }),
        })
    }

    pub fn set_witness_planes(&mut self, planes: &'a DeviceBuffer<u64>, count: usize) {
        self.fe.set_witness_planes(planes, count);
        self.nc.set_witness_planes(planes, count);
    }

    pub(crate) fn set_running_surfaces(&mut self, surfaces: Option<&'a crate::reduce::ccs::DevicePiCcsKSurfaces>) {
        self.fe.set_running_surfaces(surfaces);
    }

    pub fn enable_whole_fe_trace_for_parity(&mut self) {
        self.fe.enable_whole_fe_trace_for_parity();
    }

    pub fn enable_whole_fe_graph_for_parity(&mut self) {
        self.fe.enable_whole_fe_graph_for_parity();
    }

    pub fn enable_whole_fe_trace_recapture_for_parity(&mut self) {
        self.fe.enable_whole_fe_trace_recapture_for_parity();
    }
}

pub(crate) struct DevicePiCcsProofLogExporter {
    stream: Arc<CudaStream>,
    fe: Option<FeProofLogExport>,
    nc: Option<NcProofLogExport>,
}

struct FeProofLogExport {
    shape: super::fe::FePhaseLogShape,
    workspace: FePhaseWorkspace,
}

struct NcProofLogExport {
    shape: PendingNcPhase,
    workspace: NcPhaseWorkspace,
}

impl DevicePiCcsProofLogExporter {
    pub(crate) fn into_workspaces(mut self) -> (Option<FePhaseWorkspace>, Option<NcPhaseWorkspace>) {
        (
            self.fe.take().map(|export| export.workspace),
            self.nc.take().map(|export| export.workspace),
        )
    }

    fn export_logs(&mut self) -> Result<PiCcsPhaseProofLog, CcsDeviceError> {
        let fe = self
            .fe
            .as_mut()
            .ok_or(CcsDeviceError::Shape("FE proof-log exporter already consumed"))?;
        let nc = self
            .nc
            .as_mut()
            .ok_or(CcsDeviceError::Shape("NC proof-log exporter already consumed"))?;

        let mut fe_coeff_words = fe
            .workspace
            .take_coeff_log_host_for_stream(&self.stream, fe.workspace.coeff_log().len())?;
        let mut nc_col_coeff_words = nc
            .workspace
            .take_col_coeff_host(&self.stream, nc.workspace.col_coeff_log().len())?;
        let mut nc_tail_coeff_words = nc
            .workspace
            .take_tail_coeff_host(&self.stream, nc.workspace.tail_coeff_log().len())?;
        unsafe {
            fe.workspace
                .coeff_log()
                .copy_to_pinned_host_async(&self.stream, &mut fe_coeff_words)?;
            nc.workspace
                .col_coeff_log()
                .copy_to_pinned_host_async(&self.stream, &mut nc_col_coeff_words)?;
            nc.workspace
                .tail_coeff_log()
                .copy_to_pinned_host_async(&self.stream, &mut nc_tail_coeff_words)?;
        }
        self.stream.synchronize()?;

        let fe_coeffs = decode_round_log(
            fe_coeff_words.as_slice(),
            fe.shape.total_rounds,
            fe.shape.coeff_words_per_round,
            fe.shape.width,
        );
        let nc_coeffs = decode_nc_round_log(nc_col_coeff_words.as_slice(), nc_tail_coeff_words.as_slice(), nc.shape);
        fe.workspace.store_coeff_log_host(fe_coeff_words);
        nc.workspace.store_col_coeff_host(nc_col_coeff_words);
        nc.workspace.store_tail_coeff_host(nc_tail_coeff_words);
        Ok(PiCcsPhaseProofLog { fe_coeffs, nc_coeffs })
    }
}

impl PiCcsPhaseBackend for DevicePiCcsProofLogExporter {
    fn export_pi_ccs_phase_rounds(&mut self) -> Option<PiCcsPhaseProofLog> {
        Some(
            self.export_logs()
                .expect("device Pi_CCS deferred proof-log export failed mid-prove"),
        )
    }
}

impl PiCcsPhaseBackend for DevicePiCcsPhaseBackend<'_> {
    fn claimed_initial_sum(
        &mut self,
        challenges: &Challenges,
        k_mcs: usize,
        me_input_count: usize,
        matrix_count: usize,
    ) -> Option<K> {
        self.fe
            .claimed_initial_sum(challenges, k_mcs, me_input_count, matrix_count)
    }

    fn sample_public_challenges(
        &mut self,
        snapshot: TranscriptSnapshot,
        ell_d: usize,
        ell: usize,
        ell_m: usize,
    ) -> Option<(Challenges, TranscriptSnapshot)> {
        let (challenges, snapshot, public_challenges) =
            sample_public_challenges_on_device(self.fe.device, self.fe.kernels, snapshot, ell_d, ell, ell_m)
                .expect("device Pi_CCS public challenge sampling failed mid-prove");
        self.public_challenges = Some(public_challenges);
        Some((challenges, snapshot))
    }

    fn fe_backend_for_oracle(&mut self) -> Option<&mut dyn FeSumcheckBackend> {
        Some(&mut self.fe)
    }

    fn defers_nc_digit_tables(&self) -> bool {
        true
    }

    fn start(&mut self, fe_snapshot: &RowPhaseSnapshot<'_>, nc_snapshot: &NcColSnapshot<'_>) -> bool {
        let fe_ready = self.fe.start(fe_snapshot);
        let nc_ready = self.nc.start(nc_snapshot);
        fe_ready && nc_ready
    }

    fn prove_pi_ccs_phase(&mut self, request: PiCcsPhaseTraceRequest<'_>) -> Option<PiCcsPhaseTrace> {
        Some(
            self.fe
                .pi_ccs_phase_trace_from_transcript(&mut self.nc, request, self.public_challenges.as_ref())
                .expect("device Pi_CCS phase trace failed mid-prove"),
        )
    }

    fn summarize_pi_ccs_phase(&mut self, request: PiCcsPhaseTraceRequest<'_>) -> Option<PiCcsPhaseSummary> {
        Some(
            self.fe
                .pi_ccs_phase_summary_from_transcript(&mut self.nc, request, self.public_challenges.as_ref())
                .expect("device Pi_CCS phase summary failed mid-prove"),
        )
    }

    fn export_pi_ccs_phase_rounds(&mut self) -> Option<PiCcsPhaseProofLog> {
        Some(PiCcsPhaseProofLog {
            fe_coeffs: self
                .fe
                .export_last_phase_coeffs()
                .expect("device FE proof-log export failed mid-prove"),
            nc_coeffs: self
                .nc
                .export_last_phase_coeffs()
                .expect("device NC proof-log export failed mid-prove"),
        })
    }
}

fn sample_public_challenges_on_device(
    device: &crate::device::Device,
    kernels: &SumcheckKernels,
    snapshot: TranscriptSnapshot,
    ell_d: usize,
    ell: usize,
    ell_m: usize,
) -> Result<(Challenges, TranscriptSnapshot, DevicePublicChallenges), CcsDeviceError> {
    let alpha_beta_gamma_count = ell_d
        .checked_add(ell)
        .and_then(|value| value.checked_add(1))
        .ok_or(CcsDeviceError::Shape("Pi_CCS public challenge count overflow"))?;
    let beta_m_offset = alpha_beta_gamma_count
        .checked_mul(2)
        .ok_or(CcsDeviceError::Shape("Pi_CCS public challenge word count overflow"))?;
    let out_words = beta_m_offset
        .checked_add(
            ell_m
                .checked_mul(2)
                .ok_or(CcsDeviceError::Shape("Pi_CCS beta_m challenge word count overflow"))?,
        )
        .ok_or(CcsDeviceError::Shape("Pi_CCS public challenge output overflow"))?;

    let stream = device.stream();
    let mut transcript = DeviceTranscript::from_state_and_absorbed(device, snapshot.0, snapshot.1)?;
    let ops = [
        TranscriptIoOp::AbsorbHost(raw_append_words(&[F::from_u64(2)])),
        TranscriptIoOp::ChallengeDevice {
            offset: 0,
            len: beta_m_offset,
        },
        TranscriptIoOp::AbsorbHost(raw_append_words(&[F::from_u64(3)])),
        TranscriptIoOp::ChallengeDevice {
            offset: beta_m_offset,
            len: ell_m * 2,
        },
    ];
    let encoded = encode_transcript_io_ops(&ops);
    let ops_dev = upload_u64_device_buffer(stream, &encoded.op_words)?;
    let host_payload = upload_u64_device_buffer(stream, &encoded.host_payload)?;
    let device_payload = uninit_u64_device_buffer(stream, 1)?;
    let mut host_out = uninit_u64_device_buffer(stream, 1)?;
    let mut device_out = uninit_u64_device_buffer(stream, out_words.max(1))?;
    transcript.enqueue_io(
        device,
        &kernels.poseidon,
        &kernels.poseidon_rc,
        &ops_dev,
        &host_payload,
        &device_payload,
        &mut host_out,
        &mut device_out,
    )?;

    let words = device_out.to_host_vec(stream)?;
    let transcript_words = transcript.state_words_to_host(device)?;
    device.sync()?;
    let alpha_beta_gamma: Vec<K> = (0..alpha_beta_gamma_count)
        .map(|idx| k_from_device_words(words[2 * idx], words[2 * idx + 1]))
        .collect();
    let alpha = alpha_beta_gamma[..ell_d].to_vec();
    let beta = &alpha_beta_gamma[ell_d..ell_d + ell];
    let (beta_a, beta_r) = beta.split_at(ell_d);
    let gamma = alpha_beta_gamma[ell_d + ell];
    let beta_m = (0..ell_m)
        .map(|idx| {
            let word = beta_m_offset + 2 * idx;
            k_from_device_words(words[word], words[word + 1])
        })
        .collect();

    Ok((
        Challenges {
            alpha,
            beta_a: beta_a.to_vec(),
            beta_r: beta_r.to_vec(),
            beta_m,
            gamma,
        },
        DeviceTranscript::decode_state_words(&transcript_words),
        DevicePublicChallenges::new(device_out, ell_d, ell, ell_m),
    ))
}

fn raw_append_words(fields: &[F]) -> Vec<F> {
    let mut words = Vec::with_capacity(fields.len() + 1);
    words.push(F::from_u64(fields.len() as u64));
    words.extend(fields.iter().copied());
    words
}

impl DeviceFeBackend<'_> {
    pub(crate) fn pi_ccs_phase_summary_from_transcript(
        &mut self,
        nc_backend: &mut DeviceNcBackend<'_>,
        request: PiCcsPhaseTraceRequest<'_>,
        public_challenges: Option<&DevicePublicChallenges>,
    ) -> Result<PiCcsPhaseSummary, CcsDeviceError> {
        if self.whole_fe_graph_enabled {
            return self.pi_ccs_phase_summary_from_graph(nc_backend, request, public_challenges);
        }

        let fe_initial_sum = request.fe_initial_sum;
        let nc_initial_sum = request.nc_initial_sum;
        let include_y_zcol = request.nc_col_rounds > 0;
        let fe_request = request.fe;
        let beta_a = fe_request.beta_a;
        let gamma = fe_request.gamma;
        let mut fe_phase = self.begin_full_fe_phase(fe_request, public_challenges)?;
        let fe_download = fe_phase.start_summary_download(self)?;
        let nc_phase = nc_backend.begin_phase_with_prolog_and_tail_from_device_transcript(
            fe_phase.transcript_mut(),
            request.nc_col_rounds,
            request.nc_tail_rounds,
            request.nc_tail_coeff_count,
            nc_initial_sum,
            beta_a,
            gamma,
            public_challenges,
        )?;
        let nc_summary = nc_backend.finish_phase_summary(fe_phase.transcript_mut(), nc_phase, nc_initial_sum)?;
        let fe_summary = fe_phase.finish_summary_download(self, fe_initial_sum, fe_download)?;
        let terminal_surfaces = Some(PiCcsTerminalOutputSurfaces {
            y_ring: fe_summary.terminal_y_ring,
            y_zcol: include_y_zcol.then(|| terminal_y_zcol_from_finalized(&nc_summary.finalized)),
        });
        Ok(PiCcsPhaseSummary {
            fe_challenges: fe_summary.challenges,
            ajtai_y_eval: None,
            terminal_surfaces,
            nc_challenges: nc_summary.challenges,
            nc_finalized: nc_summary.finalized,
            sumcheck_final: fe_summary.sumcheck_final,
            sumcheck_final_nc: nc_summary.sumcheck_final,
            transcript_after: nc_summary.transcript_after,
        })
    }

    pub(crate) fn pi_ccs_phase_trace_from_transcript(
        &mut self,
        nc_backend: &mut DeviceNcBackend<'_>,
        request: PiCcsPhaseTraceRequest<'_>,
        public_challenges: Option<&DevicePublicChallenges>,
    ) -> Result<PiCcsPhaseTrace, CcsDeviceError> {
        if self.whole_fe_graph_enabled {
            return self.pi_ccs_phase_trace_from_graph(nc_backend, request, public_challenges);
        }

        let fe_request = request.fe;
        let beta_a = fe_request.beta_a;
        let gamma = fe_request.gamma;
        let mut fe_phase = self.begin_full_fe_phase(fe_request, public_challenges)?;
        let fe_download = fe_phase.start_trace_download(self)?;
        let nc_phase = nc_backend.begin_phase_with_prolog_and_tail_from_device_transcript(
            fe_phase.transcript_mut(),
            request.nc_col_rounds,
            request.nc_tail_rounds,
            request.nc_tail_coeff_count,
            request.nc_initial_sum,
            beta_a,
            gamma,
            public_challenges,
        )?;
        let nc_trace = nc_backend.finish_phase_trace(fe_phase.transcript_mut(), nc_phase)?;
        let fe_trace = fe_phase.finish_trace_download(self, fe_download, None)?;
        Ok(PiCcsPhaseTrace {
            fe_coeffs: fe_trace.coeffs,
            fe_challenges: fe_trace.challenges,
            ajtai_y_eval: fe_trace.ajtai_y_eval,
            nc_coeffs: nc_trace.coeffs,
            nc_challenges: nc_trace.challenges,
            nc_finalized: nc_trace.finalized,
            transcript_after: nc_trace.transcript_after,
        })
    }

    fn pi_ccs_phase_summary_from_graph(
        &mut self,
        nc_backend: &mut DeviceNcBackend<'_>,
        request: PiCcsPhaseTraceRequest<'_>,
        public_challenges: Option<&DevicePublicChallenges>,
    ) -> Result<PiCcsPhaseSummary, CcsDeviceError> {
        let fe_initial_sum = request.fe_initial_sum;
        let nc_initial_sum = request.nc_initial_sum;
        let fe_request = request.fe;
        let beta_a = fe_request.beta_a;
        let gamma = fe_request.gamma;
        let mut fe_phase = self.prepare_full_fe_phase(fe_request, public_challenges)?;
        let nc_prepared = nc_backend.prepare_phase_with_prolog_and_tail(
            request.nc_col_rounds,
            request.nc_tail_rounds,
            request.nc_tail_coeff_count,
            nc_initial_sum,
            beta_a,
            gamma,
            public_challenges,
        )?;

        let stream = self.device.stream();
        let graph_key = PiCcsPhaseGraphKey {
            fe: fe_phase.graph_key.clone(),
            nc: nc_backend.graph_key(&nc_prepared)?,
        };
        if !self.whole_fe_graph_recapture
            && fe_phase
                .workspace
                .launch_pi_ccs_graph_if_matching(stream, &graph_key)?
        {
            self.oracle
                .as_mut()
                .ok_or(CcsDeviceError::Shape("FE backend used before start"))?
                .mark_row_rounds_replayed(fe_phase.request.row_rounds)?;
            let y_eval = DeviceAjtaiYEval {
                words: fe_phase.workspace.take_y_eval_words(),
                forms: Some(fe_phase.workspace.take_forms()),
                witnesses: fe_phase.request.witnesses.len(),
                matrices: fe_phase.y_eval_matrices,
            };
            let nc_phase = nc_backend.mark_prepared_phase_replayed(&nc_prepared)?;
            return self.finish_pi_ccs_graph_summary(
                nc_backend,
                fe_phase,
                nc_phase,
                y_eval,
                request.nc_col_rounds > 0,
                fe_initial_sum,
                nc_initial_sum,
            );
        }

        let mut captured_y_eval = None;
        let mut captured_nc_phase = None;
        let graph = CapturedGraph::capture_checked(stream, || -> Result<(), CcsDeviceError> {
            captured_y_eval = Some(self.enqueue_prepared_full_fe_phase_body(&mut fe_phase)?);
            captured_nc_phase = Some(
                nc_backend.enqueue_prepared_phase_with_prolog_and_tail_from_device_transcript(
                    fe_phase.transcript_mut(),
                    &nc_prepared,
                )?,
            );
            Ok(())
        })
        .map_err(|error| match error {
            CaptureError::Body(error) => error,
            CaptureError::Driver(error) => CcsDeviceError::from(error),
        })?;
        graph.launch(stream)?;
        fe_phase.workspace.store_pi_ccs_graph(graph_key, graph);

        self.finish_pi_ccs_graph_summary(
            nc_backend,
            fe_phase,
            captured_nc_phase.expect("capture body produced NC phase"),
            captured_y_eval.expect("capture body produced Y_eval"),
            request.nc_col_rounds > 0,
            fe_initial_sum,
            nc_initial_sum,
        )
    }

    fn pi_ccs_phase_trace_from_graph(
        &mut self,
        nc_backend: &mut DeviceNcBackend<'_>,
        request: PiCcsPhaseTraceRequest<'_>,
        public_challenges: Option<&DevicePublicChallenges>,
    ) -> Result<PiCcsPhaseTrace, CcsDeviceError> {
        let fe_request = request.fe;
        let beta_a = fe_request.beta_a;
        let gamma = fe_request.gamma;
        let mut fe_phase = self.prepare_full_fe_phase(fe_request, public_challenges)?;
        let nc_prepared = nc_backend.prepare_phase_with_prolog_and_tail(
            request.nc_col_rounds,
            request.nc_tail_rounds,
            request.nc_tail_coeff_count,
            request.nc_initial_sum,
            beta_a,
            gamma,
            public_challenges,
        )?;

        let stream = self.device.stream();
        let graph_key = PiCcsPhaseGraphKey {
            fe: fe_phase.graph_key.clone(),
            nc: nc_backend.graph_key(&nc_prepared)?,
        };
        if !self.whole_fe_graph_recapture
            && fe_phase
                .workspace
                .launch_pi_ccs_graph_if_matching(stream, &graph_key)?
        {
            self.oracle
                .as_mut()
                .ok_or(CcsDeviceError::Shape("FE backend used before start"))?
                .mark_row_rounds_replayed(fe_phase.request.row_rounds)?;
            let y_eval = DeviceAjtaiYEval {
                words: fe_phase.workspace.take_y_eval_words(),
                forms: Some(fe_phase.workspace.take_forms()),
                witnesses: fe_phase.request.witnesses.len(),
                matrices: fe_phase.y_eval_matrices,
            };
            let nc_phase = nc_backend.mark_prepared_phase_replayed(&nc_prepared)?;
            return self.finish_pi_ccs_graph_trace(nc_backend, fe_phase, nc_phase, y_eval);
        }

        let mut captured_y_eval = None;
        let mut captured_nc_phase = None;
        let graph = CapturedGraph::capture_checked(stream, || -> Result<(), CcsDeviceError> {
            captured_y_eval = Some(self.enqueue_prepared_full_fe_phase_body(&mut fe_phase)?);
            captured_nc_phase = Some(
                nc_backend.enqueue_prepared_phase_with_prolog_and_tail_from_device_transcript(
                    fe_phase.transcript_mut(),
                    &nc_prepared,
                )?,
            );
            Ok(())
        })
        .map_err(|error| match error {
            CaptureError::Body(error) => error,
            CaptureError::Driver(error) => CcsDeviceError::from(error),
        })?;
        graph.launch(stream)?;
        fe_phase.workspace.store_pi_ccs_graph(graph_key, graph);

        self.finish_pi_ccs_graph_trace(
            nc_backend,
            fe_phase,
            captured_nc_phase.expect("capture body produced NC phase"),
            captured_y_eval.expect("capture body produced Y_eval"),
        )
    }

    fn finish_pi_ccs_graph_trace(
        &mut self,
        nc_backend: &mut DeviceNcBackend<'_>,
        mut fe_phase: PreparedFePhase<'_>,
        nc_phase: PendingNcPhase,
        y_eval: DeviceAjtaiYEval,
    ) -> Result<PiCcsPhaseTrace, CcsDeviceError> {
        let nc_trace = nc_backend.finish_phase_trace(fe_phase.transcript_mut(), nc_phase)?;
        let fe_trace = fe_phase.into_pending(y_eval).download_trace(self, None)?;
        Ok(PiCcsPhaseTrace {
            fe_coeffs: fe_trace.coeffs,
            fe_challenges: fe_trace.challenges,
            ajtai_y_eval: fe_trace.ajtai_y_eval,
            nc_coeffs: nc_trace.coeffs,
            nc_challenges: nc_trace.challenges,
            nc_finalized: nc_trace.finalized,
            transcript_after: nc_trace.transcript_after,
        })
    }

    fn finish_pi_ccs_graph_summary(
        &mut self,
        nc_backend: &mut DeviceNcBackend<'_>,
        mut fe_phase: PreparedFePhase<'_>,
        nc_phase: PendingNcPhase,
        y_eval: DeviceAjtaiYEval,
        include_y_zcol: bool,
        fe_initial_sum: K,
        nc_initial_sum: K,
    ) -> Result<PiCcsPhaseSummary, CcsDeviceError> {
        let nc_summary = nc_backend.finish_phase_summary(fe_phase.transcript_mut(), nc_phase, nc_initial_sum)?;
        let fe_summary = fe_phase
            .into_pending(y_eval)
            .download_summary(self, fe_initial_sum)?;
        let terminal_surfaces = Some(PiCcsTerminalOutputSurfaces {
            y_ring: fe_summary.terminal_y_ring,
            y_zcol: include_y_zcol.then(|| terminal_y_zcol_from_finalized(&nc_summary.finalized)),
        });
        Ok(PiCcsPhaseSummary {
            fe_challenges: fe_summary.challenges,
            ajtai_y_eval: None,
            terminal_surfaces,
            nc_challenges: nc_summary.challenges,
            nc_finalized: nc_summary.finalized,
            sumcheck_final: fe_summary.sumcheck_final,
            sumcheck_final_nc: nc_summary.sumcheck_final,
            transcript_after: nc_summary.transcript_after,
        })
    }

    pub fn full_fe_trace_from_transcript(
        &mut self,
        request: FePhaseTraceRequest<'_>,
    ) -> Result<FeRowRoundTrace, CcsDeviceError> {
        let phase = self.begin_full_fe_phase(request, None)?;
        let transcript_words = phase
            .workspace
            .transcript_state_words_to_host(self.device)?;
        let trace = phase.download_trace(self, Some(transcript_words))?;
        Ok(trace)
    }

    fn begin_full_fe_phase(
        &mut self,
        request: FePhaseTraceRequest<'_>,
        public_challenges: Option<&DevicePublicChallenges>,
    ) -> Result<PendingFePhase, CcsDeviceError> {
        let mut phase = self.prepare_full_fe_phase(request, public_challenges)?;
        let y_eval = if self.whole_fe_graph_enabled {
            self.run_or_replay_full_fe_graph(&mut phase)?
        } else {
            self.enqueue_prepared_full_fe_phase_body(&mut phase)?
        };
        Ok(phase.into_pending(y_eval))
    }

    fn prepare_full_fe_phase<'r>(
        &mut self,
        request: FePhaseTraceRequest<'r>,
        public_challenges: Option<&DevicePublicChallenges>,
    ) -> Result<PreparedFePhase<'r>, CcsDeviceError> {
        validate_fe_phase_request(&request)?;

        let width = self
            .oracle
            .as_ref()
            .ok_or(CcsDeviceError::Shape("FE backend used before start"))?
            .coeff_width;
        let total_rounds = request.row_rounds + request.tail_rounds;
        let stream = self.device.stream();
        let coeff_words_per_round = width * 2;
        let mut phase_workspace = self
            .phase_workspace
            .take()
            .unwrap_or_else(FePhaseWorkspace::new);
        phase_workspace.prepare_logs(stream, total_rounds * coeff_words_per_round, total_rounds * 2)?;
        phase_workspace.reset_transcript(self.device, request.transcript_state, request.transcript_absorbed)?;

        let (y_eval_matrices, _) = self
            .prepare_ajtai_y_eval_workspace(
                request.cache,
                request.row_rounds,
                &request.witnesses,
                &mut phase_workspace,
            )?
            .ok_or(CcsDeviceError::Shape("device Ajtai Y_eval not applicable"))?;

        prepare_fe_phase_points(
            self.device,
            self.kernels,
            &mut phase_workspace,
            &request,
            public_challenges,
        )?;
        let tail_partial_count =
            self.prepare_tail_headers_and_scratch(&request, y_eval_matrices, &mut phase_workspace)?;
        let graph_key = self.graph_key(&request, width, y_eval_matrices, tail_partial_count, &phase_workspace)?;
        Ok(PreparedFePhase {
            request,
            workspace: phase_workspace,
            total_rounds,
            coeff_words_per_round,
            width,
            y_eval_matrices,
            tail_partial_count,
            graph_key,
        })
    }

    fn prepare_tail_headers_and_scratch(
        &self,
        request: &FePhaseTraceRequest<'_>,
        y_eval_matrices: usize,
        phase_workspace: &mut FePhaseWorkspace,
    ) -> Result<usize, CcsDeviceError> {
        let mut gamma_to_k = K::ONE;
        for _ in 0..request.witnesses.len() {
            gamma_to_k *= request.gamma;
        }
        let (gamma_re, gamma_im) = request.gamma.to_limbs_u64();
        let (gamma_k_re, gamma_k_im) = gamma_to_k.to_limbs_u64();
        let mut tail_header_words = Vec::with_capacity(request.tail_rounds * TAIL_CHALLENGE_HEADER_WORDS);
        for tail_round in 0..request.tail_rounds {
            tail_header_words.extend([
                request.k_mcs as u64,
                request.witnesses.len() as u64,
                y_eval_matrices as u64,
                request.tail_rounds as u64,
                request.row_rounds as u64,
                tail_round as u64,
                self.oracle
                    .as_ref()
                    .ok_or(CcsDeviceError::Shape("FE backend used before start"))?
                    .coeff_width as u64,
                request.r_inputs.is_some() as u64,
                gamma_re,
                gamma_im,
                gamma_k_re,
                gamma_k_im,
            ]);
        }
        if tail_header_words.is_empty() {
            tail_header_words.push(0);
        }
        phase_workspace.upload_tail_headers(self.device.stream(), &tail_header_words)?;
        let tail_partial_count = if request.r_inputs.is_some()
            && request.witnesses.len() > request.k_mcs
            && y_eval_matrices > 0
            && request.tail_rounds > 0
        {
            (request.witnesses.len() - request.k_mcs)
                .checked_mul(y_eval_matrices)
                .and_then(|value| value.checked_mul(1usize << (request.tail_rounds - 1)))
                .ok_or(CcsDeviceError::Shape("Ajtai-tail partial count overflow"))?
        } else {
            0
        };
        phase_workspace.prepare_tail_scratch(self.device.stream(), tail_partial_count.max(1) * 4, SUM_BLOCKS * 4, 4)?;
        Ok(tail_partial_count)
    }

    fn graph_key(
        &self,
        request: &FePhaseTraceRequest<'_>,
        width: usize,
        y_eval_matrices: usize,
        tail_partial_count: usize,
        phase_workspace: &FePhaseWorkspace,
    ) -> Result<FePhaseGraphKey, CcsDeviceError> {
        let oracle = self
            .oracle
            .as_ref()
            .ok_or(CcsDeviceError::Shape("FE backend used before start"))?;
        let mut allocations = GraphAllocations::new();
        phase_workspace.record_graph_allocations(&mut allocations);
        oracle.record_graph_allocations(&mut allocations);
        self.ring_scratch.record_graph_allocations(&mut allocations);
        self.oracle_plan.record_graph_allocations(&mut allocations);
        Ok(FePhaseGraphKey {
            width,
            row_rounds: request.row_rounds,
            tail_rounds: request.tail_rounds,
            table_count: oracle.num_tables,
            table_stride: oracle.stride,
            active_len: oracle.active_len,
            cur_len: oracle.cur_len,
            y_eval_witnesses: request.witnesses.len(),
            y_eval_matrices,
            tail_partial_count,
            has_inputs: request.r_inputs.is_some(),
            allocations,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_or_replay_full_fe_graph(
        &mut self,
        phase: &mut PreparedFePhase<'_>,
    ) -> Result<DeviceAjtaiYEval, CcsDeviceError> {
        let stream = self.device.stream();
        if !self.whole_fe_graph_recapture
            && phase
                .workspace
                .launch_graph_if_matching(stream, &phase.graph_key)?
        {
            self.oracle
                .as_mut()
                .ok_or(CcsDeviceError::Shape("FE backend used before start"))?
                .mark_row_rounds_replayed(phase.request.row_rounds)?;
            return Ok(DeviceAjtaiYEval {
                words: phase.workspace.take_y_eval_words(),
                forms: Some(phase.workspace.take_forms()),
                witnesses: phase.request.witnesses.len(),
                matrices: phase.y_eval_matrices,
            });
        }
        let mut captured_y_eval = None;
        let graph = CapturedGraph::capture_checked(stream, || -> Result<(), CcsDeviceError> {
            captured_y_eval = Some(self.enqueue_prepared_full_fe_phase_body(phase)?);
            Ok(())
        })
        .map_err(|error| match error {
            CaptureError::Body(error) => error,
            CaptureError::Driver(error) => CcsDeviceError::from(error),
        })?;
        graph.launch(stream)?;
        phase.workspace.store_graph(phase.graph_key.clone(), graph);
        Ok(captured_y_eval.expect("capture body produced Y_eval"))
    }

    fn enqueue_prepared_full_fe_phase_body(
        &mut self,
        phase: &mut PreparedFePhase<'_>,
    ) -> Result<DeviceAjtaiYEval, CcsDeviceError> {
        self.enqueue_full_fe_phase_body(
            &phase.request,
            phase.width,
            phase.coeff_words_per_round,
            phase.tail_partial_count,
            &mut phase.workspace,
        )
    }
}

struct PreparedFePhase<'a> {
    request: FePhaseTraceRequest<'a>,
    workspace: FePhaseWorkspace,
    total_rounds: usize,
    coeff_words_per_round: usize,
    width: usize,
    y_eval_matrices: usize,
    tail_partial_count: usize,
    graph_key: FePhaseGraphKey,
}

impl PreparedFePhase<'_> {
    fn transcript_mut(&mut self) -> &mut DeviceTranscript {
        self.workspace.transcript_mut()
    }

    fn into_pending(self, y_eval: DeviceAjtaiYEval) -> PendingFePhase {
        PendingFePhase {
            workspace: self.workspace,
            y_eval,
            total_rounds: self.total_rounds,
            coeff_words_per_round: self.coeff_words_per_round,
            width: self.width,
        }
    }
}

struct PendingFePhase {
    workspace: FePhaseWorkspace,
    y_eval: DeviceAjtaiYEval,
    total_rounds: usize,
    coeff_words_per_round: usize,
    width: usize,
}

struct FePhaseSummary {
    challenges: Vec<K>,
    terminal_y_ring: Vec<Vec<Vec<K>>>,
    sumcheck_final: K,
}

struct PendingFeSummaryDownload {
    stream: Arc<CudaStream>,
    challenge_words: Option<PinnedHostBuffer<u64>>,
    y_eval_words: Option<PinnedHostBuffer<u64>>,
    last_coeff_words: Option<PinnedHostBuffer<u64>>,
    _last_coeff_scratch: Option<DeviceBuffer<u64>>,
}

impl PendingFeSummaryDownload {
    fn into_parts(
        mut self,
    ) -> Result<(PinnedHostBuffer<u64>, PinnedHostBuffer<u64>, PinnedHostBuffer<u64>), CcsDeviceError> {
        self.stream.synchronize()?;
        Ok((
            self.challenge_words
                .take()
                .expect("FE challenge transfer already consumed"),
            self.y_eval_words
                .take()
                .expect("FE Y_eval transfer already consumed"),
            self.last_coeff_words
                .take()
                .expect("FE last coeff transfer already consumed"),
        ))
    }
}

impl Drop for PendingFeSummaryDownload {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

struct PendingFeTraceDownload {
    stream: Arc<CudaStream>,
    coeff_words: Option<PinnedHostBuffer<u64>>,
    challenge_words: Option<PinnedHostBuffer<u64>>,
    y_eval_words: Option<PinnedHostBuffer<u64>>,
}

impl PendingFeTraceDownload {
    fn into_parts(
        mut self,
    ) -> Result<(PinnedHostBuffer<u64>, PinnedHostBuffer<u64>, PinnedHostBuffer<u64>), CcsDeviceError> {
        self.stream.synchronize()?;
        Ok((
            self.coeff_words
                .take()
                .expect("FE coeff transfer already consumed"),
            self.challenge_words
                .take()
                .expect("FE challenge transfer already consumed"),
            self.y_eval_words
                .take()
                .expect("FE Y_eval transfer already consumed"),
        ))
    }
}

impl Drop for PendingFeTraceDownload {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

impl PendingFePhase {
    fn transcript_mut(&mut self) -> &mut DeviceTranscript {
        self.workspace.transcript_mut()
    }

    fn start_summary_download(
        &mut self,
        backend: &mut DeviceFeBackend<'_>,
    ) -> Result<PendingFeSummaryDownload, CcsDeviceError> {
        let transfer_stream = backend.device.stream().fork()?;
        let mut challenge_words = self
            .workspace
            .take_challenges_host(backend.device, self.workspace.challenges().len())?;
        let mut y_eval_words = self
            .workspace
            .take_y_eval_host(backend.device, self.y_eval.words.len())?;
        let last_coeff_words = self.width * 2;
        let mut last_coeff_words_host = PinnedHostBuffer::zeroed(backend.device.ctx(), last_coeff_words.max(1))?;
        let mut last_coeff_scratch = if self.total_rounds == 0 {
            None
        } else {
            let mut scratch = uninit_u64_device_buffer(&transfer_stream, last_coeff_words)?;
            launch_plane_copy_slice(
                &backend.kernels.ring,
                &transfer_stream,
                self.workspace.coeff_log(),
                (self.total_rounds - 1) * self.coeff_words_per_round,
                0,
                last_coeff_words,
                &mut scratch,
            )?;
            unsafe {
                scratch.copy_to_pinned_host_async(&transfer_stream, &mut last_coeff_words_host)?;
            }
            Some(scratch)
        };
        unsafe {
            self.workspace
                .challenges()
                .copy_to_pinned_host_async(&transfer_stream, &mut challenge_words)?;
            self.y_eval
                .words
                .copy_to_pinned_host_async(&transfer_stream, &mut y_eval_words)?;
        }
        Ok(PendingFeSummaryDownload {
            stream: transfer_stream,
            challenge_words: Some(challenge_words),
            y_eval_words: Some(y_eval_words),
            last_coeff_words: Some(last_coeff_words_host),
            _last_coeff_scratch: last_coeff_scratch.take(),
        })
    }

    fn start_trace_download(
        &mut self,
        backend: &mut DeviceFeBackend<'_>,
    ) -> Result<PendingFeTraceDownload, CcsDeviceError> {
        let transfer_stream = backend.device.stream().fork()?;
        let mut coeff_words = self
            .workspace
            .take_coeff_log_host_for_stream(&transfer_stream, self.workspace.coeff_log().len())?;
        let mut challenge_words = self
            .workspace
            .take_challenges_host(backend.device, self.workspace.challenges().len())?;
        let mut y_eval_words = self
            .workspace
            .take_y_eval_host(backend.device, self.y_eval.words.len())?;
        unsafe {
            self.workspace
                .coeff_log()
                .copy_to_pinned_host_async(&transfer_stream, &mut coeff_words)?;
            self.workspace
                .challenges()
                .copy_to_pinned_host_async(&transfer_stream, &mut challenge_words)?;
            self.y_eval
                .words
                .copy_to_pinned_host_async(&transfer_stream, &mut y_eval_words)?;
        }
        Ok(PendingFeTraceDownload {
            stream: transfer_stream,
            coeff_words: Some(coeff_words),
            challenge_words: Some(challenge_words),
            y_eval_words: Some(y_eval_words),
        })
    }

    fn download_trace(
        mut self,
        backend: &mut DeviceFeBackend<'_>,
        transcript_words: Option<Vec<u64>>,
    ) -> Result<FeRowRoundTrace, CcsDeviceError> {
        let download = self.start_trace_download(backend)?;
        self.finish_trace_download(backend, download, transcript_words)
    }

    fn finish_trace_download(
        mut self,
        backend: &mut DeviceFeBackend<'_>,
        download: PendingFeTraceDownload,
        transcript_words: Option<Vec<u64>>,
    ) -> Result<FeRowRoundTrace, CcsDeviceError> {
        let transcript_words = match transcript_words {
            Some(words) => words,
            None => self
                .workspace
                .transcript_state_words_to_host(backend.device)?,
        };
        let (coeff_words, challenge_words, y_eval_words) = download.into_parts()?;
        let ajtai_y_eval = decode_ajtai_y_eval_words(y_eval_words.as_slice(), &self.y_eval);
        backend.last_y_eval = Some(self.y_eval);
        backend.last_phase_log_shape = None;
        let coeffs = (0..self.total_rounds)
            .map(|round| {
                let base = round * self.coeff_words_per_round;
                (0..self.width)
                    .map(|d| k_from_device_words(coeff_words[base + 2 * d], coeff_words[base + 2 * d + 1]))
                    .collect()
            })
            .collect();
        let challenges = (0..self.total_rounds)
            .map(|round| k_from_device_words(challenge_words[2 * round], challenge_words[2 * round + 1]))
            .collect();
        self.workspace.store_coeff_log_host(coeff_words);
        self.workspace.store_challenges_host(challenge_words);
        self.workspace.store_y_eval_host(y_eval_words);
        backend.phase_workspace = Some(self.workspace);
        Ok(FeRowRoundTrace {
            coeffs,
            challenges,
            transcript_after: Some(DeviceTranscript::decode_state_words(&transcript_words)),
            ajtai_y_eval: Some(ajtai_y_eval),
        })
    }

    fn download_summary(
        mut self,
        backend: &mut DeviceFeBackend<'_>,
        initial_sum: K,
    ) -> Result<FePhaseSummary, CcsDeviceError> {
        let download = self.start_summary_download(backend)?;
        self.finish_summary_download(backend, initial_sum, download)
    }

    fn finish_summary_download(
        mut self,
        backend: &mut DeviceFeBackend<'_>,
        initial_sum: K,
        download: PendingFeSummaryDownload,
    ) -> Result<FePhaseSummary, CcsDeviceError> {
        let (challenge_words, y_eval_words, last_coeff_words_host) = download.into_parts()?;
        let challenges = (0..self.total_rounds)
            .map(|round| k_from_device_words(challenge_words[2 * round], challenge_words[2 * round + 1]))
            .collect::<Vec<_>>();
        let sumcheck_final = match challenges.last().copied() {
            Some(challenge) => {
                let last_coeffs = (0..self.width)
                    .map(|idx| k_from_device_words(last_coeff_words_host[2 * idx], last_coeff_words_host[2 * idx + 1]))
                    .collect::<Vec<_>>();
                neo_reductions::sumcheck::poly_eval_k(&last_coeffs, challenge)
            }
            None => initial_sum,
        };
        let terminal_y_ring = decode_terminal_y_ring_words(y_eval_words.as_slice(), &self.y_eval);
        backend.last_y_eval = Some(self.y_eval);
        backend.last_phase_log_shape = Some(super::fe::FePhaseLogShape {
            total_rounds: self.total_rounds,
            coeff_words_per_round: self.coeff_words_per_round,
            width: self.width,
        });
        self.workspace.store_challenges_host(challenge_words);
        self.workspace.store_y_eval_host(y_eval_words);
        backend.phase_workspace = Some(self.workspace);
        Ok(FePhaseSummary {
            challenges,
            terminal_y_ring,
            sumcheck_final,
        })
    }
}

impl DeviceFeBackend<'_> {
    fn export_last_phase_coeffs(&mut self) -> Result<Vec<Vec<K>>, CcsDeviceError> {
        let shape = self
            .last_phase_log_shape
            .take()
            .ok_or(CcsDeviceError::Shape("FE compact phase summary log shape missing"))?;
        let workspace = self
            .phase_workspace
            .as_ref()
            .ok_or(CcsDeviceError::Shape("FE phase workspace missing for proof-log export"))?;
        let coeff_words = workspace.coeff_log().to_host_vec(self.device.stream())?;
        self.device.sync()?;
        Ok(decode_round_log(
            &coeff_words,
            shape.total_rounds,
            shape.coeff_words_per_round,
            shape.width,
        ))
    }
}

fn decode_round_log(words: &[u64], rounds: usize, coeff_words_per_round: usize, width: usize) -> Vec<Vec<K>> {
    (0..rounds)
        .map(|round| {
            let base = round * coeff_words_per_round;
            (0..width)
                .map(|d| k_from_device_words(words[base + 2 * d], words[base + 2 * d + 1]))
                .collect()
        })
        .collect()
}

fn decode_nc_round_log(col_words: &[u64], tail_words: &[u64], shape: PendingNcPhase) -> Vec<Vec<K>> {
    let mut coeffs: Vec<Vec<K>> = (0..shape.col_rounds)
        .map(|round| {
            let base = round * shape.col_coeff_words_per_round;
            (0..NC_COEFFS)
                .map(|d| k_from_device_words(col_words[base + 2 * d], col_words[base + 2 * d + 1]))
                .collect()
        })
        .collect();
    coeffs.extend((0..shape.tail_rounds).map(|round| {
        let base = round * shape.tail_coeff_words_per_round;
        (0..shape.tail_coeff_count)
            .map(|d| k_from_device_words(tail_words[base + 2 * d], tail_words[base + 2 * d + 1]))
            .collect()
    }));
    coeffs
}

fn decode_ajtai_y_eval_words(words: &[u64], y_eval: &DeviceAjtaiYEval) -> Vec<Vec<[K; D]>> {
    let per_wit = 2 * y_eval.matrices * D;
    (0..y_eval.witnesses)
        .map(|wit| {
            (0..y_eval.matrices)
                .map(|j| {
                    let re = &words[wit * per_wit + (2 * j) * D..];
                    let im = &words[wit * per_wit + (2 * j + 1) * D..];
                    let mut row = [K::ZERO; D];
                    for (rho, slot) in row.iter_mut().enumerate() {
                        *slot = k_from_device_words(re[rho], im[rho]);
                    }
                    row
                })
                .collect()
        })
        .collect()
}

fn decode_terminal_y_ring_words(words: &[u64], y_eval: &DeviceAjtaiYEval) -> Vec<Vec<Vec<K>>> {
    let d_pad = D.next_power_of_two();
    decode_ajtai_y_eval_words(words, y_eval)
        .into_iter()
        .map(|claim| {
            claim
                .into_iter()
                .map(|digits| {
                    let mut row = vec![K::ZERO; d_pad];
                    row[..D].copy_from_slice(&digits);
                    row
                })
                .collect()
        })
        .collect()
}

fn terminal_y_zcol_from_finalized(finalized: &NcFinalizedColState) -> Vec<Vec<K>> {
    let d_pad = D.next_power_of_two();
    finalized
        .digit_rows
        .iter()
        .map(|digits| {
            let mut row = vec![K::ZERO; d_pad];
            row[..D].copy_from_slice(digits);
            row
        })
        .collect()
}

fn validate_fe_phase_request(request: &FePhaseTraceRequest<'_>) -> Result<(), CcsDeviceError> {
    if request.alpha.len() != request.tail_rounds
        || request.beta_a.len() != request.tail_rounds
        || request.beta_r.len() != request.row_rounds
        || request
            .r_inputs
            .is_some_and(|r_inputs| r_inputs.len() != request.row_rounds)
    {
        return Err(CcsDeviceError::Shape("FE phase trace request length mismatch"));
    }
    Ok(())
}

fn prepare_fe_phase_points(
    device: &crate::device::Device,
    kernels: &SumcheckKernels,
    phase_workspace: &mut FePhaseWorkspace,
    request: &FePhaseTraceRequest<'_>,
    public_challenges: Option<&DevicePublicChallenges>,
) -> Result<(), CcsDeviceError> {
    let stream = device.stream();
    let r_input_words = request.r_inputs.map(|values| 2 * values.len()).unwrap_or(0);
    if let Some(public) = public_challenges.filter(|public| {
        public.matches_shape(
            request.tail_rounds,
            request.tail_rounds + request.row_rounds,
            request.row_rounds,
        )
    }) {
        let public_words = public.fe_point_words();
        phase_workspace.prepare_points(stream, public_words + r_input_words)?;
        launch_plane_copy_slice(
            &kernels.ring,
            stream,
            public.words(),
            0,
            0,
            public_words,
            phase_workspace.points_mut(),
        )?;
        if let Some(r_inputs) = request.r_inputs {
            let mut r_words = Vec::with_capacity(r_input_words);
            for value in r_inputs {
                let (re, im) = value.to_limbs_u64();
                r_words.extend([re, im]);
            }
            let r_dev = upload_u64_device_buffer(stream, &r_words)?;
            launch_plane_copy(
                &kernels.ring,
                stream,
                &r_dev,
                public_words,
                phase_workspace.points_mut(),
            )?;
        }
        return Ok(());
    }

    let public_words = 2 * (request.tail_rounds * 2 + request.row_rounds);
    let mut points_words = Vec::with_capacity(public_words + r_input_words);
    for value in request.alpha {
        let (re, im) = value.to_limbs_u64();
        points_words.extend([re, im]);
    }
    for value in request.beta_a {
        let (re, im) = value.to_limbs_u64();
        points_words.extend([re, im]);
    }
    for value in request.beta_r {
        let (re, im) = value.to_limbs_u64();
        points_words.extend([re, im]);
    }
    if let Some(r_inputs) = request.r_inputs {
        for value in r_inputs {
            let (re, im) = value.to_limbs_u64();
            points_words.extend([re, im]);
        }
    }
    phase_workspace.upload_points(stream, &points_words)
}
