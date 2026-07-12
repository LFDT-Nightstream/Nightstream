//! Host materialization of completed NC phase device state.
//!
//! The NC backend schedules and folds device tables in `nc.rs`; this file owns
//! the narrow D2H surfaces needed by parity/proof-log and terminal-summary
//! consumers.

use cuda_core::DeviceBuffer;
use neo_math::K;
use neo_reductions::optimized_engine::{NcFinalizedColState, NcPhaseRoundTrace, TranscriptSnapshot};

use crate::device::Device;
use crate::field::k_from_device_words;
use crate::kernels::ajtai::launch_plane_copy_slice;
use crate::kernels::pi_ccs_nc::NC_COEFFS;
use crate::reduce::ccs::{
    nc::{download_finalized_col_state, DeviceNcBackend, DeviceNcFinalState, DeviceNcOracle, PendingNcPhase},
    CcsDeviceError, NcPhaseWorkspace, SumcheckKernels,
};
use crate::transcript::DeviceTranscript;

pub(crate) struct NcPhaseSummary {
    pub(crate) challenges: Vec<K>,
    pub(crate) finalized: NcFinalizedColState,
    pub(crate) sumcheck_final: K,
    pub(crate) transcript_after: Option<TranscriptSnapshot>,
}

impl PendingNcPhase {
    pub(crate) fn download_trace(
        self,
        device: &Device,
        kernels: &SumcheckKernels,
        transcript: &DeviceTranscript,
        oracle: &DeviceNcOracle,
        workspace: &NcPhaseWorkspace,
    ) -> Result<(NcPhaseRoundTrace, DeviceNcFinalState), CcsDeviceError> {
        let stream = device.stream();
        let total_rounds = self.col_rounds + self.tail_rounds;
        let transcript_words = transcript.state_words_to_host(device)?;
        let col_coeff_words = workspace.col_coeff_log().to_host_vec(stream)?;
        let tail_coeff_words = workspace.tail_coeff_log().to_host_vec(stream)?;
        let challenge_words = workspace.challenges().to_host_vec(stream)?;
        let packed_finalized = oracle.finalized_col_state_device(device, kernels)?;
        let finalized = download_finalized_col_state(device, &packed_finalized)?;
        device.sync()?;

        let mut coeffs: Vec<Vec<K>> = (0..self.col_rounds)
            .map(|round| {
                let base = round * self.col_coeff_words_per_round;
                (0..NC_COEFFS)
                    .map(|d| k_from_device_words(col_coeff_words[base + 2 * d], col_coeff_words[base + 2 * d + 1]))
                    .collect()
            })
            .collect();
        coeffs.extend((0..self.tail_rounds).map(|round| {
            let base = round * self.tail_coeff_words_per_round;
            (0..self.tail_coeff_count)
                .map(|d| k_from_device_words(tail_coeff_words[base + 2 * d], tail_coeff_words[base + 2 * d + 1]))
                .collect()
        }));
        let challenges = (0..total_rounds)
            .map(|round| k_from_device_words(challenge_words[2 * round], challenge_words[2 * round + 1]))
            .collect();
        Ok((
            NcPhaseRoundTrace {
                coeffs,
                challenges,
                transcript_after: Some(DeviceTranscript::decode_state_words(&transcript_words)),
                finalized,
            },
            packed_finalized,
        ))
    }

    pub(crate) fn download_summary(
        self,
        device: &Device,
        kernels: &SumcheckKernels,
        transcript: &DeviceTranscript,
        oracle: &DeviceNcOracle,
        workspace: &NcPhaseWorkspace,
        initial_sum: K,
    ) -> Result<(NcPhaseSummary, DeviceNcFinalState), CcsDeviceError> {
        let stream = device.stream();
        let total_rounds = self.col_rounds + self.tail_rounds;
        let transcript_words = transcript.state_words_to_host(device)?;
        let challenge_words = workspace.challenges().to_host_vec(stream)?;
        let last_coeffs = if total_rounds == 0 {
            Vec::new()
        } else if self.tail_rounds > 0 {
            let round = self.tail_rounds - 1;
            download_k_slice(
                device,
                kernels,
                workspace.tail_coeff_log(),
                round * self.tail_coeff_words_per_round,
                self.tail_coeff_count,
            )?
        } else {
            let round = self.col_rounds - 1;
            download_k_slice(
                device,
                kernels,
                workspace.col_coeff_log(),
                round * self.col_coeff_words_per_round,
                NC_COEFFS,
            )?
        };
        let packed_finalized = oracle.finalized_col_state_device(device, kernels)?;
        let finalized = download_finalized_col_state(device, &packed_finalized)?;
        device.sync()?;

        let challenges: Vec<K> = (0..total_rounds)
            .map(|round| k_from_device_words(challenge_words[2 * round], challenge_words[2 * round + 1]))
            .collect();
        let sumcheck_final = match challenges.last().copied() {
            Some(challenge) => neo_reductions::sumcheck::poly_eval_k(&last_coeffs, challenge),
            None => initial_sum,
        };
        Ok((
            NcPhaseSummary {
                challenges,
                finalized,
                sumcheck_final,
                transcript_after: Some(DeviceTranscript::decode_state_words(&transcript_words)),
            },
            packed_finalized,
        ))
    }
}

impl DeviceNcBackend<'_> {
    pub(crate) fn export_last_phase_coeffs(&mut self) -> Result<Vec<Vec<K>>, CcsDeviceError> {
        let shape = self
            .last_phase_log_shape
            .take()
            .ok_or(CcsDeviceError::Shape("NC compact phase summary log shape missing"))?;
        let workspace = self
            .phase_workspace
            .as_ref()
            .ok_or(CcsDeviceError::Shape("NC phase workspace missing for proof-log export"))?;
        let stream = self.device.stream();
        let col_coeff_words = workspace.col_coeff_log().to_host_vec(stream)?;
        let tail_coeff_words = workspace.tail_coeff_log().to_host_vec(stream)?;
        self.device.sync()?;

        let mut coeffs: Vec<Vec<K>> = (0..shape.col_rounds)
            .map(|round| {
                let base = round * shape.col_coeff_words_per_round;
                (0..NC_COEFFS)
                    .map(|d| k_from_device_words(col_coeff_words[base + 2 * d], col_coeff_words[base + 2 * d + 1]))
                    .collect()
            })
            .collect();
        coeffs.extend((0..shape.tail_rounds).map(|round| {
            let base = round * shape.tail_coeff_words_per_round;
            (0..shape.tail_coeff_count)
                .map(|d| k_from_device_words(tail_coeff_words[base + 2 * d], tail_coeff_words[base + 2 * d + 1]))
                .collect()
        }));
        Ok(coeffs)
    }
}

fn download_k_slice(
    device: &Device,
    kernels: &SumcheckKernels,
    src: &DeviceBuffer<u64>,
    word_offset: usize,
    count: usize,
) -> Result<Vec<K>, CcsDeviceError> {
    let stream = device.stream();
    let words = count * 2;
    let mut scratch = crate::device::uninit_u64_device_buffer(stream, words.max(1))?;
    launch_plane_copy_slice(&kernels.ring, stream, src, word_offset, 0, words, &mut scratch)?;
    let host = scratch.to_host_vec(stream)?;
    device.sync()?;
    Ok((0..count)
        .map(|idx| k_from_device_words(host[2 * idx], host[2 * idx + 1]))
        .collect())
}
