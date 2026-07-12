//! Applying whole-phase Π_CCS backend traces.
//!
//! Owns only host-side validation and bookkeeping for a backend-returned
//! FE→NC trace. The backend schedules device work; the optimized engine
//! still owns transcript adoption, sumcheck invariants, oracle state, and
//! proof-round assembly.

use neo_math::{F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::error::PiCcsError;
use crate::optimized_engine::backend::{
    BackendTranscriptMode, FeRowRoundSummary, FeRowRoundTrace, NcColRoundTrace, PiCcsPhaseSummary, PiCcsPhaseTrace,
    PiCcsTerminalOutputSurfaces,
};
use crate::sumcheck::RoundOracle;

use super::oracle::{NcOracle, OptimizedOracle};
use super::transcript_segments::{append_nc_sumcheck_prolog, finish_backend_transcript};

pub(super) struct PiCcsPhaseApply<'a, 'b> {
    pub(super) tr: &'a mut Poseidon2Transcript,
    pub(super) oracle: &'a mut OptimizedOracle<'b, F>,
    pub(super) oracle_nc: &'a mut NcOracle<'b, F>,
    pub(super) trace: PiCcsPhaseTrace,
    pub(super) fe_row_rounds: usize,
    pub(super) nc_col_rounds: usize,
    pub(super) nc_tail_rounds: usize,
    pub(super) transcript_mode: BackendTranscriptMode,
    pub(super) sumcheck_rounds: &'a mut Option<Vec<Vec<K>>>,
    pub(super) sumcheck_chals: &'a mut Vec<K>,
    pub(super) running_sum: &'a mut K,
    pub(super) sumcheck_rounds_nc: &'a mut Option<Vec<Vec<K>>>,
    pub(super) sumcheck_chals_nc: &'a mut Vec<K>,
    pub(super) running_sum_nc: &'a mut K,
}

pub(super) struct PiCcsPhaseSummaryApply<'a, 'b> {
    pub(super) tr: &'a mut Poseidon2Transcript,
    pub(super) oracle: &'a mut OptimizedOracle<'b, F>,
    pub(super) oracle_nc: &'a mut NcOracle<'b, F>,
    pub(super) summary: PiCcsPhaseSummary,
    pub(super) fe_row_rounds: usize,
    pub(super) nc_col_rounds: usize,
    pub(super) nc_tail_rounds: usize,
    pub(super) transcript_mode: BackendTranscriptMode,
    pub(super) sumcheck_chals: &'a mut Vec<K>,
    pub(super) running_sum: &'a mut K,
    pub(super) sumcheck_chals_nc: &'a mut Vec<K>,
    pub(super) running_sum_nc: &'a mut K,
}

pub(super) struct AppliedPiCcsPhase {
    pub(super) first_host_round: usize,
    pub(super) first_nc_host_round: usize,
    pub(super) terminal_surfaces: Option<PiCcsTerminalOutputSurfaces>,
}

pub(super) struct FeBackendTraceApply<'a, 'b> {
    pub(super) tr: &'a mut Poseidon2Transcript,
    pub(super) oracle: &'a mut OptimizedOracle<'b, F>,
    pub(super) trace: FeRowRoundTrace,
    pub(super) expected_rounds: usize,
    pub(super) row_rounds: usize,
    pub(super) transcript_mode: BackendTranscriptMode,
    pub(super) sumcheck_rounds: &'a mut Option<Vec<Vec<K>>>,
    pub(super) sumcheck_chals: &'a mut Vec<K>,
    pub(super) running_sum: &'a mut K,
    pub(super) trace_name: &'static str,
}

pub(super) struct FeBackendSummaryApply<'a, 'b> {
    pub(super) tr: &'a mut Poseidon2Transcript,
    pub(super) oracle: &'a mut OptimizedOracle<'b, F>,
    pub(super) summary: FeRowRoundSummary,
    pub(super) expected_rounds: usize,
    pub(super) transcript_mode: BackendTranscriptMode,
    pub(super) sumcheck_chals: &'a mut Vec<K>,
    pub(super) running_sum: &'a mut K,
    pub(super) trace_name: &'static str,
}

pub(super) struct NcBackendTraceApply<'a, 'b> {
    pub(super) tr: &'a mut Poseidon2Transcript,
    pub(super) oracle_nc: &'a mut NcOracle<'b, F>,
    pub(super) trace: NcColRoundTrace,
    pub(super) expected_rounds: usize,
    pub(super) transcript_mode: BackendTranscriptMode,
    pub(super) append_prolog: bool,
    pub(super) initial_sum: K,
    pub(super) sumcheck_rounds_nc: &'a mut Option<Vec<Vec<K>>>,
    pub(super) sumcheck_chals_nc: &'a mut Vec<K>,
    pub(super) running_sum_nc: &'a mut K,
}

pub(super) fn apply_pi_ccs_phase_summary(ctx: PiCcsPhaseSummaryApply<'_, '_>) -> Result<AppliedPiCcsPhase, PiCcsError> {
    if ctx.transcript_mode.replays() {
        return Err(PiCcsError::InvalidInput(
            "Pi_CCS phase summary cannot replace replay proof logs".into(),
        ));
    }

    let expected_fe_rounds = ctx.oracle.num_rounds();
    let expected_nc_rounds = ctx.nc_col_rounds + ctx.nc_tail_rounds;
    let mut summary = ctx.summary;
    if summary.fe_challenges.len() != expected_fe_rounds || summary.nc_challenges.len() != expected_nc_rounds {
        return Err(PiCcsError::InvalidInput(
            "Pi_CCS phase summary challenge length mismatch".into(),
        ));
    }

    let mut ajtai_y_eval = summary.ajtai_y_eval.take();
    let terminal_surfaces = summary.terminal_surfaces.take();
    let terminal_surfaces_supplied = terminal_surfaces.is_some();
    for (round_idx, &r_i) in summary.fe_challenges.iter().enumerate() {
        ctx.sumcheck_chals.push(r_i);
        if round_idx < ctx.fe_row_rounds {
            ctx.oracle.advance_row_round_without_fold(r_i);
        } else if terminal_surfaces_supplied {
            // The backend supplied the terminal output surfaces directly, so
            // the CPU oracle does not need the Ajtai-tail state for output
            // materialization. The engine still owns challenge bookkeeping
            // and terminal sums.
        } else {
            if round_idx == ctx.fe_row_rounds {
                let y_eval = ajtai_y_eval
                    .take()
                    .ok_or_else(|| PiCcsError::InvalidInput("Pi_CCS phase summary omitted Ajtai Y_eval".into()))?;
                ctx.oracle.inject_ajtai_y_eval(y_eval);
            }
            ctx.oracle.fold(r_i);
        }
    }
    *ctx.running_sum = summary.sumcheck_final;

    let mut nc_finalized = Some(summary.nc_finalized);
    for (round_idx, &r_i) in summary.nc_challenges.iter().enumerate() {
        ctx.sumcheck_chals_nc.push(r_i);
        if round_idx < ctx.nc_col_rounds {
            ctx.oracle_nc.advance_col_round_without_fold(r_i);
        } else if terminal_surfaces_supplied {
            // Terminal y_zcol rows came from the backend-owned output
            // surface; no CPU NC tail state is needed for output assembly.
        } else {
            if round_idx == ctx.nc_col_rounds {
                let finalized = nc_finalized
                    .take()
                    .ok_or_else(|| PiCcsError::InvalidInput("Pi_CCS phase summary omitted NC final state".into()))?;
                ctx.oracle_nc
                    .inject_finalized_col_state(finalized.digit_rows, finalized.eq_beta_m0);
            }
            ctx.oracle_nc.fold(r_i);
        }
    }
    if !terminal_surfaces_supplied && expected_nc_rounds == ctx.nc_col_rounds {
        let finalized = nc_finalized
            .take()
            .ok_or_else(|| PiCcsError::InvalidInput("Pi_CCS phase summary omitted NC final state".into()))?;
        ctx.oracle_nc
            .inject_finalized_col_state(finalized.digit_rows, finalized.eq_beta_m0);
    }
    *ctx.running_sum_nc = summary.sumcheck_final_nc;

    finish_backend_transcript(
        ctx.tr,
        summary.transcript_after,
        ctx.transcript_mode,
        "Pi_CCS phase backend summary",
    )?;
    Ok(AppliedPiCcsPhase {
        first_host_round: expected_fe_rounds,
        first_nc_host_round: expected_nc_rounds,
        terminal_surfaces,
    })
}

pub(super) fn apply_fe_backend_trace(ctx: FeBackendTraceApply<'_, '_>) -> Result<usize, PiCcsError> {
    let FeRowRoundTrace {
        coeffs,
        challenges,
        transcript_after,
        mut ajtai_y_eval,
    } = ctx.trace;
    if coeffs.len() != ctx.expected_rounds || challenges.len() != ctx.expected_rounds {
        return Err(PiCcsError::InvalidInput(format!("{} length mismatch", ctx.trace_name)));
    }

    let mut ajtai_injected = false;
    for (round_idx, (coeffs, &r_i)) in coeffs.iter().zip(challenges.iter()).enumerate() {
        if ctx.transcript_mode.replays() {
            validate_fe_round(ctx.tr, coeffs, r_i, *ctx.running_sum, round_idx)?;
        }
        ctx.sumcheck_chals.push(r_i);
        *ctx.running_sum = crate::sumcheck::poly_eval_k(coeffs, r_i);
        if round_idx < ctx.row_rounds {
            ctx.oracle.advance_row_round_without_fold(r_i);
        } else {
            if !ajtai_injected {
                if let Some(y_eval) = ajtai_y_eval.take() {
                    ctx.oracle.inject_ajtai_y_eval(y_eval);
                }
                ajtai_injected = true;
            }
            ctx.oracle.fold(r_i);
        }
        if let Some(rounds) = ctx.sumcheck_rounds.as_mut() {
            rounds.push(coeffs.clone());
        }
    }
    finish_backend_transcript(ctx.tr, transcript_after, ctx.transcript_mode, ctx.trace_name)?;
    Ok(ctx.expected_rounds)
}

pub(super) fn apply_fe_backend_summary(ctx: FeBackendSummaryApply<'_, '_>) -> Result<usize, PiCcsError> {
    if ctx.transcript_mode.replays() {
        return Err(PiCcsError::InvalidInput(format!(
            "{} cannot replace replay proof logs",
            ctx.trace_name
        )));
    }
    let FeRowRoundSummary {
        challenges,
        sumcheck_final,
        transcript_after,
    } = ctx.summary;
    if challenges.len() != ctx.expected_rounds {
        return Err(PiCcsError::InvalidInput(format!("{} length mismatch", ctx.trace_name)));
    }
    for &r_i in &challenges {
        ctx.sumcheck_chals.push(r_i);
        ctx.oracle.advance_row_round_without_fold(r_i);
    }
    *ctx.running_sum = sumcheck_final;
    finish_backend_transcript(ctx.tr, transcript_after, ctx.transcript_mode, ctx.trace_name)?;
    Ok(ctx.expected_rounds)
}

pub(super) fn apply_nc_backend_trace(ctx: NcBackendTraceApply<'_, '_>) -> Result<usize, PiCcsError> {
    let NcColRoundTrace {
        coeffs,
        challenges,
        transcript_after,
        finalized,
    } = ctx.trace;
    if coeffs.len() != ctx.expected_rounds || challenges.len() != ctx.expected_rounds {
        return Err(PiCcsError::InvalidInput(
            "NC backend column-round trace length mismatch".into(),
        ));
    }
    if ctx.append_prolog && ctx.transcript_mode.replays() {
        append_nc_sumcheck_prolog(ctx.tr, ctx.initial_sum);
    }
    for (round_idx, (coeffs, &r_i)) in coeffs.iter().zip(challenges.iter()).enumerate() {
        if ctx.transcript_mode.replays() {
            validate_nc_round(ctx.tr, coeffs, r_i, *ctx.running_sum_nc, round_idx)?;
        }
        ctx.sumcheck_chals_nc.push(r_i);
        *ctx.running_sum_nc = crate::sumcheck::poly_eval_k(coeffs, r_i);
        ctx.oracle_nc.advance_col_round_without_fold(r_i);
        if let Some(rounds) = ctx.sumcheck_rounds_nc.as_mut() {
            rounds.push(coeffs.clone());
        }
    }
    finish_backend_transcript(ctx.tr, transcript_after, ctx.transcript_mode, "NC backend column trace")?;
    ctx.oracle_nc
        .inject_finalized_col_state(finalized.digit_rows, finalized.eq_beta_m0);
    Ok(ctx.expected_rounds)
}

pub(super) fn apply_pi_ccs_phase_trace(ctx: PiCcsPhaseApply<'_, '_>) -> Result<AppliedPiCcsPhase, PiCcsError> {
    let expected_fe_rounds = ctx.oracle.num_rounds();
    let expected_nc_rounds = ctx.nc_col_rounds + ctx.nc_tail_rounds;
    let mut trace = ctx.trace;

    if trace.fe_coeffs.len() != expected_fe_rounds || trace.fe_challenges.len() != expected_fe_rounds {
        return Err(PiCcsError::InvalidInput(
            "Pi_CCS phase backend FE trace length mismatch".into(),
        ));
    }
    if trace.nc_coeffs.len() != expected_nc_rounds || trace.nc_challenges.len() != expected_nc_rounds {
        return Err(PiCcsError::InvalidInput(
            "Pi_CCS phase backend NC trace length mismatch".into(),
        ));
    }

    let mut ajtai_y_eval = trace.ajtai_y_eval.take();
    for (round_idx, (coeffs, &r_i)) in trace
        .fe_coeffs
        .iter()
        .zip(trace.fe_challenges.iter())
        .enumerate()
    {
        if ctx.transcript_mode.replays() {
            validate_fe_round(ctx.tr, coeffs, r_i, *ctx.running_sum, round_idx)?;
        }
        ctx.sumcheck_chals.push(r_i);
        *ctx.running_sum = crate::sumcheck::poly_eval_k(coeffs, r_i);
        if round_idx < ctx.fe_row_rounds {
            ctx.oracle.advance_row_round_without_fold(r_i);
        } else {
            if round_idx == ctx.fe_row_rounds {
                let y_eval = ajtai_y_eval
                    .take()
                    .ok_or_else(|| PiCcsError::InvalidInput("Pi_CCS phase backend omitted Ajtai Y_eval".into()))?;
                ctx.oracle.inject_ajtai_y_eval(y_eval);
            }
            ctx.oracle.fold(r_i);
        }
        if let Some(rounds) = ctx.sumcheck_rounds.as_mut() {
            rounds.push(coeffs.clone());
        }
    }

    if ctx.transcript_mode.replays() {
        append_nc_sumcheck_prolog(ctx.tr, *ctx.running_sum_nc);
    }
    let mut nc_finalized = Some(trace.nc_finalized);
    for (round_idx, (coeffs, &r_i)) in trace
        .nc_coeffs
        .iter()
        .zip(trace.nc_challenges.iter())
        .enumerate()
    {
        if ctx.transcript_mode.replays() {
            validate_nc_round(ctx.tr, coeffs, r_i, *ctx.running_sum_nc, round_idx)?;
        }
        ctx.sumcheck_chals_nc.push(r_i);
        *ctx.running_sum_nc = crate::sumcheck::poly_eval_k(coeffs, r_i);
        if round_idx < ctx.nc_col_rounds {
            ctx.oracle_nc.advance_col_round_without_fold(r_i);
        } else {
            if round_idx == ctx.nc_col_rounds {
                let finalized = nc_finalized
                    .take()
                    .ok_or_else(|| PiCcsError::InvalidInput("Pi_CCS phase backend omitted NC final state".into()))?;
                ctx.oracle_nc
                    .inject_finalized_col_state(finalized.digit_rows, finalized.eq_beta_m0);
            }
            ctx.oracle_nc.fold(r_i);
        }
        if let Some(rounds) = ctx.sumcheck_rounds_nc.as_mut() {
            rounds.push(coeffs.clone());
        }
    }
    finish_backend_transcript(
        ctx.tr,
        trace.transcript_after,
        ctx.transcript_mode,
        "Pi_CCS phase backend trace",
    )?;
    if expected_nc_rounds == ctx.nc_col_rounds {
        let finalized = nc_finalized
            .take()
            .ok_or_else(|| PiCcsError::InvalidInput("Pi_CCS phase backend omitted NC final state".into()))?;
        ctx.oracle_nc
            .inject_finalized_col_state(finalized.digit_rows, finalized.eq_beta_m0);
    }

    Ok(AppliedPiCcsPhase {
        first_host_round: expected_fe_rounds,
        first_nc_host_round: expected_nc_rounds,
        terminal_surfaces: None,
    })
}

fn validate_fe_round(
    tr: &mut Poseidon2Transcript,
    coeffs: &[K],
    challenge: K,
    running_sum: K,
    round_idx: usize,
) -> Result<(), PiCcsError> {
    let p0 = crate::sumcheck::poly_eval_k(coeffs, K::ZERO);
    let p1 = crate::sumcheck::poly_eval_k(coeffs, K::ONE);
    if p0 + p1 != running_sum {
        return Err(PiCcsError::SumcheckError(format!(
            "round {round_idx} invariant failed: p(0)+p(1) != running_sum (Pi_CCS phase backend)"
        )));
    }
    replay_round_challenge(tr, coeffs, challenge, "Pi_CCS phase backend FE challenge")
}

fn validate_nc_round(
    tr: &mut Poseidon2Transcript,
    coeffs: &[K],
    challenge: K,
    running_sum: K,
    round_idx: usize,
) -> Result<(), PiCcsError> {
    let p0 = coeffs[0];
    let p1 = crate::sumcheck::poly_eval_k_base(coeffs, F::ONE);
    if p0 + p1 != running_sum {
        return Err(PiCcsError::SumcheckError(format!(
            "NC round {round_idx} invariant failed: p(0)+p(1) != running_sum (Pi_CCS phase backend)"
        )));
    }
    replay_round_challenge(tr, coeffs, challenge, "Pi_CCS phase backend NC challenge")
}

fn replay_round_challenge(
    tr: &mut Poseidon2Transcript,
    coeffs: &[K],
    expected: K,
    label: &'static str,
) -> Result<(), PiCcsError> {
    let coeff_fields = crate::sumcheck::round_coeff_fields(coeffs);
    tr.append_fields_raw(&coeff_fields);
    let c = tr.challenge_fields_raw(2);
    let replayed = neo_math::from_complex(c[0], c[1]);
    if replayed != expected {
        return Err(PiCcsError::InvalidInput(format!("{label} mismatch during host replay")));
    }
    Ok(())
}
