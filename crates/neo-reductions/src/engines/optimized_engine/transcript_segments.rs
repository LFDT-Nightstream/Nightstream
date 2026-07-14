//! Host transcript handoff helpers for device-backed Π_CCS segments.
//!
//! Owns only the small transcript transitions between backend-owned device
//! segments and the canonical host transcript.

use neo_math::{KExtensions, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::engines::utils;
use crate::error::PiCcsError;
use crate::optimized_engine::backend::{BackendTranscriptMode, PiCcsPhaseBackend, TranscriptSnapshot};
use crate::optimized_engine::Challenges;

pub(super) fn finish_backend_transcript(
    tr: &mut Poseidon2Transcript,
    snapshot: Option<TranscriptSnapshot>,
    mode: BackendTranscriptMode,
    label: &'static str,
) -> Result<(), PiCcsError> {
    let Some((state, absorbed)) = snapshot else {
        if mode.replays() {
            return Ok(());
        }
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: device transcript snapshot missing"
        )));
    };
    if mode.replays() {
        if tr.state() != state || tr.absorbed() != absorbed {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: device transcript snapshot mismatch after replay"
            )));
        }
    } else {
        *tr = Poseidon2Transcript::from_state_and_absorbed(state, absorbed);
    }
    Ok(())
}

pub(super) fn append_nc_sumcheck_prolog(tr: &mut Poseidon2Transcript, initial_sum: K) {
    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)]);
    tr.append_fields_raw(&[F::from_u64(crate::engines::utils::PI_CCS_SUMCHECK_INITIAL_RAW_TAG)]);
    tr.append_fields_raw(&initial_sum.as_coeffs());
    tr.append_fields_raw(&[F::from_u64(crate::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)]);
}

pub(super) fn sample_public_challenges_with_backend(
    tr: &mut Poseidon2Transcript,
    phase_backend: &mut Option<&mut dyn PiCcsPhaseBackend>,
    mode: BackendTranscriptMode,
    ell_d: usize,
    ell: usize,
    ell_m: usize,
) -> Result<Challenges, PiCcsError> {
    let snapshot = (tr.state(), tr.absorbed());
    let Some((device_challenges, device_snapshot)) = phase_backend
        .as_mut()
        .and_then(|backend| backend.sample_public_challenges(snapshot, ell_d, ell, ell_m))
    else {
        let mut ch = utils::sample_challenges(tr, ell_d, ell)?;
        ch.beta_m = utils::sample_beta_m(tr, ell_m)?;
        return Ok(ch);
    };

    if mode.replays() {
        let mut host_challenges = utils::sample_challenges(tr, ell_d, ell)?;
        host_challenges.beta_m = utils::sample_beta_m(tr, ell_m)?;
        if host_challenges != device_challenges {
            return Err(PiCcsError::InvalidInput(
                "Pi_CCS public challenges mismatch during device replay".into(),
            ));
        }
        if tr.state() != device_snapshot.0 || tr.absorbed() != device_snapshot.1 {
            return Err(PiCcsError::InvalidInput(
                "Pi_CCS public challenge transcript snapshot mismatch during device replay".into(),
            ));
        }
    } else {
        *tr = Poseidon2Transcript::from_state_and_absorbed(device_snapshot.0, device_snapshot.1);
    }

    Ok(device_challenges)
}
