//! Device FE row-summary helpers.
//!
//! Owns the row-only device transcript summary used by the default Pi_CCS
//! path. Whole-phase proof-log ownership belongs to the phase backend.

use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer};
use neo_math::{F, K};
use neo_reductions::optimized_engine::FeRowRoundSummary;

use crate::device::Device;
use crate::field::k_from_device_words;
use crate::kernels::ajtai::{launch_plane_copy, launch_plane_copy_slice};

use crate::reduce::ccs::{CcsDeviceError, DeviceFeBackend, DeviceFeOracle, FePhaseWorkspace, SumcheckKernels};
use crate::transcript::DeviceTranscript;

/// Compact device-owned FE row proof log detached from the reusable phase
/// workspace.
///
/// Archiving is D2D and stream-ordered, so the next fold can immediately
/// reuse the workspace without forcing the current proof log through a host
/// join. Host decoding happens only when the proof carrier is materialized.
pub(crate) struct DeviceFeRowProofLogArchive {
    stream: Arc<CudaStream>,
    coeff_log: DeviceBuffer<u64>,
    shape: super::fe::FePhaseLogShape,
}

impl DeviceFeRowProofLogArchive {
    pub(crate) fn export_rounds(self) -> Result<Vec<Vec<K>>, CcsDeviceError> {
        let words = self.coeff_log.to_host_vec(&self.stream)?;
        Ok(decode_round_log(
            &words,
            self.shape.total_rounds,
            self.shape.coeff_words_per_round,
            self.shape.width,
        ))
    }
}

impl DeviceFeOracle {
    pub(super) fn row_round_summary_from_transcript_with_workspace(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        phase_workspace: &mut FePhaseWorkspace,
        transcript_state: [F; crate::kernels::poseidon2::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
        initial_sum: K,
    ) -> Result<FeRowRoundSummary, CcsDeviceError> {
        let width = self.coeff_width;
        let stream = device.stream();
        let coeff_words_per_round = width * 2;
        phase_workspace.prepare_logs(stream, rounds * coeff_words_per_round, rounds * 2)?;
        phase_workspace.reset_transcript(device, transcript_state, transcript_absorbed)?;

        for round in 0..rounds {
            let coeff_offset = round * coeff_words_per_round;
            let emitted_width = self.write_round_coeffs(device, kernels)?;
            if emitted_width != width {
                return Err(CcsDeviceError::Shape("FE round width changed"));
            }
            launch_plane_copy(
                &kernels.ring,
                stream,
                &self.coeffs_out,
                coeff_offset,
                phase_workspace.coeff_log_mut(),
            )?;
            phase_workspace.enqueue_coeff_challenge(
                device,
                &kernels.poseidon,
                &kernels.poseidon_rc,
                &self.coeffs_out,
                coeff_words_per_round,
                2 * round,
            )?;
            self.fold_from_challenge(device, kernels, phase_workspace.challenges(), 2 * round)?;
        }

        let transcript_state_words = phase_workspace.transcript_state_words_to_host(device)?;
        let challenge_words = phase_workspace.challenges().to_host_vec(stream)?;
        let last_coeffs = if rounds == 0 {
            Vec::new()
        } else {
            download_k_slice(
                device,
                kernels,
                phase_workspace.coeff_log(),
                (rounds - 1) * coeff_words_per_round,
                width,
            )?
        };
        device.sync()?;
        let (challenges, sumcheck_final, transcript_after) = {
            let challenges = (0..rounds)
                .map(|round| k_from_device_words(challenge_words[2 * round], challenge_words[2 * round + 1]))
                .collect::<Vec<_>>();
            let sumcheck_final = match challenges.last().copied() {
                Some(challenge) => neo_reductions::sumcheck::poly_eval_k(&last_coeffs, challenge),
                None => initial_sum,
            };
            let transcript_after = Some(DeviceTranscript::decode_state_words(&transcript_state_words));
            (challenges, sumcheck_final, transcript_after)
        };
        Ok(FeRowRoundSummary {
            challenges,
            sumcheck_final,
            transcript_after,
        })
    }
}

impl DeviceFeBackend<'_> {
    pub(super) fn row_round_summary_from_transcript_retained(
        &mut self,
        transcript_state: [F; crate::kernels::poseidon2::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
        initial_sum: K,
    ) -> Result<FeRowRoundSummary, CcsDeviceError> {
        let mut phase_workspace = self
            .phase_workspace
            .take()
            .unwrap_or_else(FePhaseWorkspace::new);
        let oracle = self
            .oracle
            .as_mut()
            .ok_or(CcsDeviceError::Shape("FE backend used before start"))?;
        let width = oracle.coeff_width;
        let summary = oracle.row_round_summary_from_transcript_with_workspace(
            self.device,
            self.kernels,
            &mut phase_workspace,
            transcript_state,
            transcript_absorbed,
            rounds,
            initial_sum,
        )?;
        self.last_phase_log_shape = Some(super::fe::FePhaseLogShape {
            total_rounds: rounds,
            coeff_words_per_round: width * 2,
            width,
        });
        self.phase_workspace = Some(phase_workspace);
        Ok(summary)
    }

    pub(super) fn export_retained_row_rounds(&mut self) -> Result<Vec<Vec<K>>, CcsDeviceError> {
        let shape = self
            .last_phase_log_shape
            .take()
            .ok_or(CcsDeviceError::Shape("FE row proof-log shape missing"))?;
        let workspace = self.phase_workspace.as_mut().ok_or(CcsDeviceError::Shape(
            "FE phase workspace missing for row proof-log export",
        ))?;
        let coeff_words = workspace.coeff_log().to_host_vec(self.device.stream())?;
        self.device.sync()?;
        let rounds = decode_round_log(
            &coeff_words,
            shape.total_rounds,
            shape.coeff_words_per_round,
            shape.width,
        );
        Ok(rounds)
    }

    pub(crate) fn archive_retained_row_rounds(&mut self) -> Result<Option<DeviceFeRowProofLogArchive>, CcsDeviceError> {
        let Some(shape) = self.last_phase_log_shape.take() else {
            return Ok(None);
        };
        let workspace = self.phase_workspace.as_ref().ok_or(CcsDeviceError::Shape(
            "FE phase workspace missing for row proof-log archive",
        ))?;
        let stream = Arc::clone(self.device.stream());
        let mut coeff_log = crate::device::uninit_u64_device_buffer(&stream, workspace.coeff_log().len())?;
        coeff_log.copy_from_device_async(workspace.coeff_log(), &stream)?;
        Ok(Some(DeviceFeRowProofLogArchive {
            stream,
            coeff_log,
            shape,
        }))
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
