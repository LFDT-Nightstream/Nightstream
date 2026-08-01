//! Proof assembly for optimized Π_CCS replay output.
//!
//! Owns only conversion from replay terminal state plus captured round logs
//! into the public `PiCcsProof` wire object.

use neo_math::K;

use crate::error::PiCcsError;
use crate::optimized_engine::{PiCcsProof, PiCcsProvePerf};

use super::backend::{FeSumcheckBackend, PiCcsPhaseBackend, PiCcsPhaseProofLog};
use super::legacy_types::{PiCcsReplayTerminalState, PiCcsTerminalOutputShell};

pub(super) struct OptimizedProofRounds {
    pub(super) sumcheck_rounds: Vec<Vec<K>>,
    pub(super) initial_sum: K,
    pub(super) sumcheck_rounds_nc: Vec<Vec<K>>,
    pub(super) initial_sum_nc: K,
}

pub(super) struct DeferredFeRowRounds {
    pub(super) row_rounds: usize,
    pub(super) sumcheck_tail_rounds: Vec<Vec<K>>,
    pub(super) initial_sum: K,
    pub(super) sumcheck_rounds_nc: Vec<Vec<K>>,
    pub(super) initial_sum_nc: K,
}

pub(super) enum DeferredProofRounds {
    Owned(OptimizedProofRounds),
    PhaseBackend,
    FeRows(DeferredFeRowRounds),
}

impl DeferredProofRounds {
    pub(super) fn into_owned(self, context: &'static str) -> Result<OptimizedProofRounds, PiCcsError> {
        match self {
            Self::Owned(rounds) => Ok(rounds),
            Self::PhaseBackend | Self::FeRows(_) => Err(PiCcsError::InvalidInput(context.into())),
        }
    }
}

pub(super) fn proof_from_terminal_state(
    terminal_state: &PiCcsReplayTerminalState,
    rounds: OptimizedProofRounds,
) -> PiCcsProof {
    let mut proof = PiCcsProof::new(rounds.sumcheck_rounds, Some(rounds.initial_sum));
    proof.variant = terminal_state.variant;
    proof.sumcheck_challenges = [terminal_state.row_chals.clone(), terminal_state.alpha_prime.clone()].concat();
    proof.sumcheck_rounds_nc = rounds.sumcheck_rounds_nc;
    proof.sc_initial_sum_nc = Some(rounds.initial_sum_nc);
    proof.sumcheck_challenges_nc = [terminal_state.s_col.clone(), terminal_state.alpha_prime_nc.clone()].concat();
    proof.challenges_public = terminal_state.challenges_public.clone();
    proof.sumcheck_final = terminal_state.sumcheck_final;
    proof.sumcheck_final_nc = terminal_state.sumcheck_final_nc;
    proof.header_digest = terminal_state.fold_digest.to_vec();
    proof.canonicalize();
    proof
}

/// Terminal Pi_CCS state whose proof round logs are still backend-owned.
///
/// This is the protocol-owned handoff for GPU scheduling: downstream prover
/// work can consume the terminal CE claims immediately, while proof bytes are
/// assembled later by exporting the resident FE/NC coefficient logs from the
/// same phase backend.
pub struct PiCcsDeferredProof {
    terminal_state: PiCcsReplayTerminalState,
    rounds: DeferredProofRounds,
}

impl PiCcsDeferredProof {
    pub(super) fn new(terminal_state: PiCcsReplayTerminalState, rounds: DeferredProofRounds) -> Self {
        Self { terminal_state, rounds }
    }

    pub fn outputs(&self) -> &[neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>] {
        &self.terminal_state.me_outputs
    }

    pub fn outputs_mut(&mut self) -> &mut [neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>] {
        &mut self.terminal_state.me_outputs
    }

    pub fn output_shell(&self) -> &PiCcsTerminalOutputShell {
        &self.terminal_state.output_shell
    }

    pub fn output_count(&self) -> usize {
        self.terminal_state.output_shell.count
    }

    pub fn row_challenges(&self) -> &[K] {
        &self.terminal_state.output_shell.row_chals
    }

    pub fn column_challenges(&self) -> &[K] {
        &self.terminal_state.output_shell.s_col
    }

    pub fn fold_digest(&self) -> [u8; 32] {
        self.terminal_state.output_shell.fold_digest
    }

    pub fn perf(&self) -> PiCcsProvePerf {
        self.terminal_state.perf
    }

    pub fn finish_with_phase_backend(
        self,
        phase_backend: &mut dyn PiCcsPhaseBackend,
    ) -> Result<
        (
            Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>>,
            PiCcsProof,
            PiCcsProvePerf,
        ),
        PiCcsError,
    > {
        let PiCcsDeferredProof { terminal_state, rounds } = self;
        let rounds = match rounds {
            DeferredProofRounds::Owned(rounds) => rounds,
            DeferredProofRounds::PhaseBackend => {
                let Some(PiCcsPhaseProofLog { fe_coeffs, nc_coeffs }) = phase_backend.export_pi_ccs_phase_rounds()
                else {
                    return Err(PiCcsError::InvalidInput(
                        "Pi_CCS deferred proof requires backend-owned proof rounds".into(),
                    ));
                };
                OptimizedProofRounds {
                    sumcheck_rounds: fe_coeffs,
                    initial_sum: terminal_state.sc_initial_sum,
                    sumcheck_rounds_nc: nc_coeffs,
                    initial_sum_nc: terminal_state.sc_initial_sum_nc,
                }
            }
            DeferredProofRounds::FeRows(_) => {
                return Err(PiCcsError::InvalidInput(
                    "Pi_CCS deferred proof requires FE row backend, not phase backend".into(),
                ));
            }
        };
        Self::finish_with_rounds(terminal_state, rounds)
    }

    pub fn finish_with_fe_backend(
        self,
        fe_backend: &mut dyn FeSumcheckBackend,
    ) -> Result<
        (
            Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>>,
            PiCcsProof,
            PiCcsProvePerf,
        ),
        PiCcsError,
    > {
        let PiCcsDeferredProof { terminal_state, rounds } = self;
        let rounds = match rounds {
            DeferredProofRounds::Owned(rounds) => rounds,
            DeferredProofRounds::FeRows(deferred) => {
                let Some(mut row_rounds) = fe_backend.export_row_rounds() else {
                    return Err(PiCcsError::InvalidInput(
                        "Pi_CCS deferred FE row proof requires backend-owned row rounds".into(),
                    ));
                };
                if row_rounds.len() != deferred.row_rounds {
                    return Err(PiCcsError::InvalidInput(
                        "Pi_CCS deferred FE row proof round count mismatch".into(),
                    ));
                }
                row_rounds.extend(deferred.sumcheck_tail_rounds);
                OptimizedProofRounds {
                    sumcheck_rounds: row_rounds,
                    initial_sum: deferred.initial_sum,
                    sumcheck_rounds_nc: deferred.sumcheck_rounds_nc,
                    initial_sum_nc: deferred.initial_sum_nc,
                }
            }
            DeferredProofRounds::PhaseBackend => {
                return Err(PiCcsError::InvalidInput(
                    "Pi_CCS deferred proof requires phase backend, not FE row backend".into(),
                ));
            }
        };
        Self::finish_with_rounds(terminal_state, rounds)
    }

    /// Finish a row-trace proof from coefficient rounds archived by the
    /// execution backend.
    ///
    /// This is the owned-log counterpart to `finish_with_fe_backend`: the
    /// protocol layer still validates the expected row count and assembles
    /// the canonical proof, while an accelerator may return its reusable
    /// workspace immediately after copying the tiny proof log device-to-device.
    pub fn finish_with_fe_rounds(
        self,
        mut row_rounds: Vec<Vec<K>>,
    ) -> Result<
        (
            Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>>,
            PiCcsProof,
            PiCcsProvePerf,
        ),
        PiCcsError,
    > {
        let PiCcsDeferredProof { terminal_state, rounds } = self;
        let rounds = match rounds {
            DeferredProofRounds::Owned(rounds) => rounds,
            DeferredProofRounds::FeRows(deferred) => {
                if row_rounds.len() != deferred.row_rounds {
                    return Err(PiCcsError::InvalidInput(
                        "Pi_CCS archived FE row proof round count mismatch".into(),
                    ));
                }
                row_rounds.extend(deferred.sumcheck_tail_rounds);
                OptimizedProofRounds {
                    sumcheck_rounds: row_rounds,
                    initial_sum: deferred.initial_sum,
                    sumcheck_rounds_nc: deferred.sumcheck_rounds_nc,
                    initial_sum_nc: deferred.initial_sum_nc,
                }
            }
            DeferredProofRounds::PhaseBackend => {
                return Err(PiCcsError::InvalidInput(
                    "Pi_CCS deferred proof requires phase-backend logs, not FE row rounds".into(),
                ));
            }
        };
        Self::finish_with_rounds(terminal_state, rounds)
    }

    fn finish_with_rounds(
        terminal_state: PiCcsReplayTerminalState,
        rounds: OptimizedProofRounds,
    ) -> Result<
        (
            Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>>,
            PiCcsProof,
            PiCcsProvePerf,
        ),
        PiCcsError,
    > {
        let proof = proof_from_terminal_state(&terminal_state, rounds);
        Ok((terminal_state.me_outputs, proof, terminal_state.perf))
    }
}
