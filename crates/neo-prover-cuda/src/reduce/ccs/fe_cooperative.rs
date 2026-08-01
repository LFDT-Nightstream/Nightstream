//! Cooperative FE row-round trace variants.
//!
//! Owns experimental command grains for the FE row ladder. These paths are
//! parity gates until a measured mode proves they beat the default row trace.

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_math::{F, K};
use neo_reductions::optimized_engine::legacy_split_nc::FeRowRoundTrace;

use crate::field::k_from_device_words;
use crate::kernels::pi_ccs_fe::launch_fe_cooperative_row_rounds;
use crate::transcript::DeviceTranscript;

use super::{CcsDeviceError, DeviceFeBackend, DeviceFeOracle, FePhaseWorkspace};

impl DeviceFeOracle {
    pub(super) fn row_round_trace_from_transcript_cooperative_all_with_workspace(
        &mut self,
        device: &crate::device::Device,
        kernels: &super::SumcheckKernels,
        phase_workspace: &mut FePhaseWorkspace,
        transcript_state: [F; WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Result<FeRowRoundTrace, CcsDeviceError> {
        let width = self.coeff_width;
        let stream = device.stream();
        let coeff_words_per_round = width * 2;
        phase_workspace.prepare_logs(stream, rounds * coeff_words_per_round, rounds * 2)?;
        phase_workspace.reset_transcript(device, transcript_state, transcript_absorbed)?;

        {
            let (transcript_state, coeff_log, challenges) = phase_workspace.cooperative_row_round_buffers();
            launch_fe_cooperative_row_rounds(
                &kernels.fe,
                stream,
                &mut self.tables_a,
                &mut self.tables_b,
                &self.header,
                &self.mcs_meta,
                &self.term_meta,
                &self.term_vars,
                self.stride,
                self.active_len,
                self.cur_len,
                self.front_is_a,
                width,
                self.num_tables,
                rounds,
                &mut self.partials,
                &mut self.sum_scratch,
                &mut self.coeffs_out,
                transcript_state,
                coeff_log,
                challenges,
                &kernels.poseidon_rc,
            )?;
        }
        self.mark_row_rounds_replayed(rounds)?;

        let transcript_words = phase_workspace.transcript_state_words_to_host(device)?;
        let coeff_words = phase_workspace.coeff_log().to_host_vec(stream)?;
        let challenge_words = phase_workspace.challenges().to_host_vec(stream)?;
        device.sync()?;

        Ok(FeRowRoundTrace {
            coeffs: decode_round_coeffs(&coeff_words, rounds, coeff_words_per_round, width),
            challenges: decode_round_challenges(&challenge_words, rounds),
            transcript_after: Some(DeviceTranscript::decode_state_words(&transcript_words)),
            ajtai_y_eval: None,
        })
    }
}

impl DeviceFeBackend<'_> {
    pub fn row_round_trace_from_transcript_cooperative_all(
        &mut self,
        transcript_state: [F; WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> FeRowRoundTrace {
        let mut phase_workspace = self
            .phase_workspace
            .take()
            .unwrap_or_else(FePhaseWorkspace::new);
        let trace = self
            .oracle
            .as_mut()
            .expect("FE backend used before start")
            .row_round_trace_from_transcript_cooperative_all_with_workspace(
                self.device,
                self.kernels,
                &mut phase_workspace,
                transcript_state,
                transcript_absorbed,
                rounds,
            )
            .expect("cooperative all-round device FE row trace failed mid-prove");
        self.phase_workspace = Some(phase_workspace);
        trace
    }
}

fn decode_round_coeffs(words: &[u64], rounds: usize, coeff_words_per_round: usize, width: usize) -> Vec<Vec<K>> {
    (0..rounds)
        .map(|round| {
            let base = round * coeff_words_per_round;
            (0..width)
                .map(|d| k_from_device_words(words[base + 2 * d], words[base + 2 * d + 1]))
                .collect()
        })
        .collect()
}

fn decode_round_challenges(words: &[u64], rounds: usize) -> Vec<K> {
    (0..rounds)
        .map(|round| k_from_device_words(words[2 * round], words[2 * round + 1]))
        .collect()
}
