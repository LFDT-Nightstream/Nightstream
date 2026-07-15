//! Accelerator backend contracts for the optimized Π_CCS prover.
//!
//! These traits describe optional execution backends only. The optimized
//! engine remains the protocol owner: it builds the transcript, checks
//! returned rounds, and decides when backend state may be injected.

use neo_ccs::Mat;
use neo_math::{F, K};

use super::common::Challenges;

pub type TranscriptSnapshot = ([F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH], usize);

/// How a device-backed prove advances the host transcript after a device
/// transcript segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendTranscriptMode {
    /// Replay every downloaded absorb/challenge into the host transcript and
    /// fail immediately on mismatch. This is the parity/debug path.
    Replay,
    /// Trust the device transcript snapshot for the prover fast path. The
    /// verifier still recomputes every challenge from the proof.
    DeviceSnapshot,
}

impl BackendTranscriptMode {
    pub fn replays(self) -> bool {
        matches!(self, Self::Replay)
    }
}

/// Bulk row-round output from a backend that derived FE challenges without
/// per-round host readback.
pub struct FeRowRoundTrace {
    pub coeffs: Vec<Vec<K>>,
    pub challenges: Vec<K>,
    /// Device transcript position after the returned rounds. Fast prover
    /// mode restores this directly; parity mode replays and checks it.
    pub transcript_after: Option<TranscriptSnapshot>,
    /// Optional Ajtai-phase linear-form surface at the finalized row point.
    /// Whole-phase backends may return this so the host oracle can replay
    /// and emit outputs without recomputing the same device work.
    pub ajtai_y_eval: Option<Vec<Vec<[K; neo_math::D]>>>,
}

/// Compact row-round result from a backend that keeps proof coefficients
/// resident until proof assembly.
pub struct FeRowRoundSummary {
    pub challenges: Vec<K>,
    pub sumcheck_final: K,
    pub transcript_after: Option<TranscriptSnapshot>,
}

/// Inputs for a backend that can drive the complete FE sumcheck channel:
/// row-domain rounds followed by the Ajtai tail, all from one transcript
/// snapshot. The backend owns execution scheduling only; the caller still
/// replays the returned trace into the canonical transcript.
pub struct FePhaseTraceRequest<'a> {
    pub transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
    pub transcript_absorbed: usize,
    pub row_rounds: usize,
    pub tail_rounds: usize,
    pub cache: &'a crate::superneo_eval::SuperneoEvalCache,
    pub n_eff: usize,
    pub witnesses: Vec<&'a Mat<F>>,
    pub alpha: &'a [K],
    pub beta_a: &'a [K],
    pub beta_r: &'a [K],
    pub r_inputs: Option<&'a [K]>,
    pub gamma: K,
    pub k_mcs: usize,
}

pub struct NcColRoundTrace {
    pub coeffs: Vec<Vec<K>>,
    pub challenges: Vec<K>,
    pub transcript_after: Option<TranscriptSnapshot>,
    pub finalized: NcFinalizedColState,
}

pub struct NcPhaseRoundTrace {
    pub coeffs: Vec<Vec<K>>,
    pub challenges: Vec<K>,
    pub transcript_after: Option<TranscriptSnapshot>,
    pub finalized: NcFinalizedColState,
}

pub struct NcColTraceRequest {
    pub transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
    pub transcript_absorbed: usize,
    pub rounds: usize,
    pub initial_sum: K,
}

/// Inputs for a backend that owns the complete Π_CCS Fiat-Shamir chain:
/// FE row rounds, FE Ajtai-tail rounds, NC prolog absorbs, and NC column
/// rounds. The optimized engine still owns protocol validation and proof
/// assembly; the backend owns only device scheduling and resident state.
pub struct PiCcsPhaseTraceRequest<'a> {
    pub fe: FePhaseTraceRequest<'a>,
    pub fe_initial_sum: K,
    pub nc_col_rounds: usize,
    pub nc_tail_rounds: usize,
    pub nc_tail_coeff_count: usize,
    pub nc_initial_sum: K,
}

/// Bulk output from a backend that ran the complete Π_CCS FE+NC challenge
/// chain without handing transcript control back to the host between
/// segments.
pub struct PiCcsPhaseTrace {
    pub fe_coeffs: Vec<Vec<K>>,
    pub fe_challenges: Vec<K>,
    pub ajtai_y_eval: Option<Vec<Vec<[K; neo_math::D]>>>,
    pub nc_coeffs: Vec<Vec<K>>,
    pub nc_challenges: Vec<K>,
    pub nc_finalized: NcFinalizedColState,
    /// Device transcript position after FE rows + tail + NC prolog + NC
    /// columns + NC Ajtai tail. Fast prover mode restores this directly;
    /// parity mode replays the returned logs and checks this snapshot.
    pub transcript_after: Option<TranscriptSnapshot>,
}

/// Terminal K-valued output surfaces measured by a backend-owned Π_CCS phase.
///
/// This is not a proof object and not verifier authority. It is the narrow
/// host-visible surface the protocol engine needs to assemble canonical
/// `CeClaim`s without forcing the CPU oracle to reconstruct the same values
/// from backend-owned Ajtai state.
pub struct PiCcsTerminalOutputSurfaces {
    /// `[claim][matrix][lane]` y-ring rows, padded to the same width the CPU
    /// output builder would put in `CeClaim::y_ring`.
    pub y_ring: Vec<Vec<Vec<K>>>,
    /// `[claim][lane]` NC column opening rows when the NC surface is active.
    pub y_zcol: Option<Vec<Vec<K>>>,
}

/// Proof-round coefficients exported after a summary-driven phase.
///
/// This is intentionally narrower than [`PiCcsPhaseTrace`]: the engine has
/// already advanced transcript/oracle state from [`PiCcsPhaseSummary`], so
/// proof assembly only needs the round polynomials.
pub struct PiCcsPhaseProofLog {
    pub fe_coeffs: Vec<Vec<K>>,
    pub nc_coeffs: Vec<Vec<K>>,
}

/// Host-visible terminal summary from a backend-owned Π_CCS phase.
///
/// This is the proof-log deferral seam: the engine can advance canonical
/// oracle/transcript bookkeeping from challenges, terminal sums, and final
/// surfaces without requiring the backend to materialize every round
/// polynomial before the rest of the fold can proceed. Parity/proof-export
/// paths still need the full [`PiCcsPhaseTrace`].
pub struct PiCcsPhaseSummary {
    pub fe_challenges: Vec<K>,
    pub ajtai_y_eval: Option<Vec<Vec<[K; neo_math::D]>>>,
    pub terminal_surfaces: Option<PiCcsTerminalOutputSurfaces>,
    pub nc_challenges: Vec<K>,
    pub nc_finalized: NcFinalizedColState,
    pub sumcheck_final: K,
    pub sumcheck_final_nc: K,
    pub transcript_after: Option<TranscriptSnapshot>,
}

impl PiCcsPhaseSummary {
    pub fn from_trace(trace: PiCcsPhaseTrace, fe_initial_sum: K, nc_initial_sum: K) -> Self {
        let sumcheck_final = trace
            .fe_coeffs
            .iter()
            .zip(trace.fe_challenges.iter())
            .fold(fe_initial_sum, |_, (coeffs, &challenge)| {
                crate::sumcheck::poly_eval_k(coeffs, challenge)
            });
        let sumcheck_final_nc = trace
            .nc_coeffs
            .iter()
            .zip(trace.nc_challenges.iter())
            .fold(nc_initial_sum, |_, (coeffs, &challenge)| {
                crate::sumcheck::poly_eval_k(coeffs, challenge)
            });
        Self {
            fe_challenges: trace.fe_challenges,
            ajtai_y_eval: trace.ajtai_y_eval,
            terminal_surfaces: None,
            nc_challenges: trace.nc_challenges,
            nc_finalized: trace.nc_finalized,
            sumcheck_final,
            sumcheck_final_nc,
            transcript_after: trace.transcript_after,
        }
    }
}

/// Optional whole-phase Π_CCS backend hook.
///
/// This is the structural CUDA seam: one call is allowed to keep the
/// transcript and proof logs device-resident across FE and NC. Implementations
/// must return byte-identical round polynomials/challenges so the engine can
/// assemble the same proof and the verifier can recompute every challenge.
pub trait PiCcsPhaseBackend {
    /// Compute the public FE claimed sum from backend-owned running-claim
    /// surfaces. Returning `None` keeps the canonical host calculation.
    fn claimed_initial_sum(
        &mut self,
        challenges: &Challenges,
        k_mcs: usize,
        me_input_count: usize,
        matrix_count: usize,
    ) -> Option<K> {
        let _ = (challenges, k_mcs, me_input_count, matrix_count);
        None
    }

    /// Optionally sample the public Π_CCS challenges on the backend transcript.
    ///
    /// The optimized engine has already bound the header, public instances,
    /// and ME inputs into the canonical transcript. Backends that are moving
    /// the whole Π_CCS Fiat-Shamir chain to the device can mirror the next
    /// `sample_challenges` / `sample_beta_m` transition from this snapshot.
    ///
    /// Returning `None` keeps the host transcript path. Returning `Some`
    /// gives the engine the sampled challenge values plus the backend's
    /// transcript position after the same transition; replay mode checks both
    /// against the host transcript, while fast mode adopts the snapshot.
    fn sample_public_challenges(
        &mut self,
        snapshot: TranscriptSnapshot,
        ell_d: usize,
        ell: usize,
        ell_m: usize,
    ) -> Option<(Challenges, TranscriptSnapshot)> {
        let _ = (snapshot, ell_d, ell, ell_m);
        None
    }

    /// Optional construction-time FE backend.
    ///
    /// Whole-phase device backends need this hook while the CPU oracle builds
    /// its row-phase snapshot: it lets them keep selected row tables
    /// device-resident without making the optimized engine give up protocol
    /// ownership.
    fn fe_backend_for_oracle(&mut self) -> Option<&mut dyn FeSumcheckBackend> {
        None
    }

    /// Whether the phase backend will source NC digit tables from device
    /// state. If it later declines the phase snapshot, the CPU oracle
    /// materializes the skipped tables before fallback.
    fn defers_nc_digit_tables(&self) -> bool {
        false
    }

    fn start(
        &mut self,
        fe_snapshot: &super::oracle::RowPhaseSnapshot<'_>,
        nc_snapshot: &super::oracle::NcColSnapshot<'_>,
    ) -> bool {
        let _ = (fe_snapshot, nc_snapshot);
        false
    }

    fn prove_pi_ccs_phase(&mut self, request: PiCcsPhaseTraceRequest<'_>) -> Option<PiCcsPhaseTrace> {
        let _ = request;
        None
    }

    fn summarize_pi_ccs_phase(&mut self, request: PiCcsPhaseTraceRequest<'_>) -> Option<PiCcsPhaseSummary> {
        let fe_initial_sum = request.fe_initial_sum;
        let nc_initial_sum = request.nc_initial_sum;
        self.prove_pi_ccs_phase(request)
            .map(|trace| PiCcsPhaseSummary::from_trace(trace, fe_initial_sum, nc_initial_sum))
    }

    /// Export full proof-round coefficients after a successful
    /// `summarize_pi_ccs_phase` call in fast prove mode.
    ///
    /// Backends may keep these logs device-resident while the engine advances
    /// the rest of Pi_CCS terminal bookkeeping. Returning `None` means the
    /// backend cannot support deferred proof assembly for this phase.
    fn export_pi_ccs_phase_rounds(&mut self) -> Option<PiCcsPhaseProofLog> {
        None
    }
}

/// Device-backend hook for the FE (row-phase) sumcheck rounds.
///
/// `start` receives the built oracle's row tables once per prove; returning
/// `false` keeps the canonical CPU path. While active, `round_coeffs` must
/// return exactly the univariate coefficients the CPU b=2 path accumulates
/// and `fold` must mirror `RowStreamState::fold_inplace`. The CPU oracle
/// still owns challenge bookkeeping and every later phase.
pub trait FeSumcheckBackend {
    /// Whether the backend builds row-domain equality tables from their small
    /// challenge points. When true the host oracle leaves those tables
    /// unmaterialized unless the backend declines the row phase.
    fn defers_row_equality_tables(&self) -> bool {
        false
    }

    /// Compute the public FE claimed sum from backend-owned running-claim
    /// surfaces. Returning `None` keeps the canonical host calculation.
    fn claimed_initial_sum(
        &mut self,
        challenges: &Challenges,
        k_mcs: usize,
        me_input_count: usize,
        matrix_count: usize,
    ) -> Option<K> {
        let _ = (challenges, k_mcs, me_input_count, matrix_count);
        None
    }

    fn start(&mut self, snapshot: &super::oracle::RowPhaseSnapshot<'_>) -> bool;
    fn round_coeffs(&mut self) -> Vec<K>;
    fn fold(&mut self, r: K);

    /// Optionally run the row-domain FE rounds from a transcript snapshot.
    /// Implementations must leave their row tables folded through `rounds`
    /// challenges and return every round polynomial/challenge for host
    /// replay. `None` keeps the per-round host-driven path.
    fn row_round_trace_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Option<FeRowRoundTrace> {
        let _ = (transcript_state, transcript_absorbed, rounds);
        None
    }

    /// Like [`FeSumcheckBackend::row_round_trace_from_transcript`], but returns
    /// only the data needed to advance canonical bookkeeping. Proof coefficients
    /// remain backend-owned and must later be returned by
    /// [`FeSumcheckBackend::export_row_rounds`].
    fn row_round_summary_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
        initial_sum: K,
    ) -> Option<FeRowRoundSummary> {
        let _ = (transcript_state, transcript_absorbed, rounds, initial_sum);
        None
    }

    /// Export row-round proof coefficients retained by
    /// [`FeSumcheckBackend::row_round_summary_from_transcript`].
    fn export_row_rounds(&mut self) -> Option<Vec<Vec<K>>> {
        None
    }

    /// Optionally run the full FE channel on the backend: row rounds,
    /// Ajtai-tail coefficient rounds, and challenge sampling. `None` keeps
    /// the narrower row-only hook plus canonical host tail path.
    fn fe_phase_trace_from_transcript(&mut self, request: FePhaseTraceRequest<'_>) -> Option<FeRowRoundTrace> {
        let _ = request;
        None
    }

    /// Optionally compute the Ajtai-phase `Y_eval[witness][matrix]` (the
    /// ring linear forms at chi_r', evaluated against every witness) on the
    /// device. `None` keeps the CPU `precompute_for_r`.
    fn ajtai_y_eval(
        &mut self,
        cache: &crate::superneo_eval::SuperneoEvalCache,
        chi_r: &[K],
        n_eff: usize,
        witnesses: &[&Mat<F>],
    ) -> Option<Vec<Vec<[K; neo_math::D]>>> {
        let _ = (cache, chi_r, n_eff, witnesses);
        None
    }

    /// Like [`Self::ajtai_y_eval`], but lets the backend build the row-point
    /// tensor from its small challenge vector instead of receiving a host
    /// materialization. `None` falls back to the canonical host table path.
    fn ajtai_y_eval_from_row_challenges(
        &mut self,
        cache: &crate::superneo_eval::SuperneoEvalCache,
        row_challenges: &[K],
        n_eff: usize,
        witnesses: &[&Mat<F>],
    ) -> Option<Vec<Vec<[K; neo_math::D]>>> {
        let _ = (cache, row_challenges, n_eff, witnesses);
        None
    }

    /// Optionally build one non-zero MCS witness's f-var row tables
    /// (`m_j(row) = (M_j z)[row]` for each `f_var_indices` entry, each table
    /// `n_pad` long). `None` keeps the host build.
    ///
    /// `Host` returns canonical host tables. `Deferred` means the backend has
    /// already retained the corresponding tables in its own device-resident
    /// state and will serve every row-phase round from there; the CPU oracle
    /// must not attempt to evaluate or fold those placeholder tables.
    fn mcs_row_tables(
        &mut self,
        cache: &crate::superneo_eval::SuperneoEvalCache,
        mcs_idx: usize,
        f_var_indices: &[usize],
        z_blocks: &crate::superneo_eval::SuperneoZBlocks,
        n_eff: usize,
        n_pad: usize,
    ) -> Option<FeMcsRowTables> {
        let _ = (cache, mcs_idx, f_var_indices, z_blocks, n_eff, n_pad);
        None
    }

    /// Whether this backend will serve [`Self::carried_eval_table`] from its
    /// own resident witness planes. When true the oracle skips building the
    /// running witnesses' host `SuperneoZBlocks` entirely: the backend owns
    /// the carried combination.
    fn serves_carried_eval_table(&self) -> bool {
        false
    }

    /// Build the carried-ME eval table from the backend's resident witness
    /// planes: the carried combination `sum_i carried_coeffs[i] * plane[k_mcs+i]`
    /// followed by the weighted row table over it. Values must be
    /// field-identical to the host path.
    ///
    /// `Deferred` means the backend retained the table out of band and must
    /// later accept the row-phase snapshot; the CPU oracle must not evaluate
    /// row rounds from a missing carried eval table.
    #[allow(clippy::too_many_arguments)]
    fn carried_eval_table(
        &mut self,
        cache: &crate::superneo_eval::SuperneoEvalCache,
        carried_coeffs: &[K],
        k_mcs: usize,
        weights: &[K; neo_math::D],
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Option<FeEvalTable> {
        let _ = (cache, carried_coeffs, k_mcs, weights, mat_coeffs, n_eff, n_pad);
        None
    }

    /// Optionally build the carried-ME eval table on the device:
    /// `eval_weighted_row_table` over the carried witness combination.
    /// `None` keeps the host build; results must be bit-identical.
    fn eval_weighted_row_table(
        &mut self,
        cache: &crate::superneo_eval::SuperneoEvalCache,
        z_blocks: &crate::superneo_eval::SuperneoZBlocks,
        weights: &[K; neo_math::D],
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Option<Vec<K>> {
        let _ = (cache, z_blocks, weights, mat_coeffs, n_eff, n_pad);
        None
    }
}

/// Result of [`FeSumcheckBackend::mcs_row_tables`].
pub enum FeMcsRowTables {
    /// Host-owned row-domain tables, exactly as the CPU oracle would build.
    Host(Vec<Vec<K>>),
    /// Backend-owned row-domain tables retained out of band, typically on a
    /// device. Valid only when the backend later accepts the row-phase
    /// snapshot and drives every row round.
    Deferred,
}

/// Result of [`FeSumcheckBackend::carried_eval_table`].
pub enum FeEvalTable {
    /// Host-owned carried eval table, exactly as the CPU oracle would build.
    Host(Vec<K>),
    /// Backend-owned carried eval table retained out of band, typically on a
    /// device. Valid only when the backend later accepts the row-phase
    /// snapshot and drives every row round.
    Deferred,
}

/// The device-measured column state handed back after the last NC column
/// round: one fully folded digit row per witness plus `eq_beta_m` folded to
/// its single entry. Injected into the CPU `NcOracle` for the Ajtai tail.
pub struct NcFinalizedColState {
    pub digit_rows: Vec<[K; neo_math::D]>,
    pub eq_beta_m0: K,
}

/// Device-backend hook for the NC column-phase sumcheck rounds, mirroring
/// [`FeSumcheckBackend`]. The NC Ajtai tail reads the folded digit rows, so
/// an active backend must return them via `finalized_col_state`.
pub trait NcSumcheckBackend {
    fn start(&mut self, snapshot: &super::oracle::NcColSnapshot<'_>) -> bool;
    fn round_coeffs(&mut self) -> Vec<K>;
    fn fold(&mut self, r: K);
    fn finalized_col_state(&mut self) -> NcFinalizedColState;

    fn col_round_trace_from_transcript(
        &mut self,
        transcript_state: [F; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
        transcript_absorbed: usize,
        rounds: usize,
    ) -> Option<NcColRoundTrace> {
        let _ = (transcript_state, transcript_absorbed, rounds);
        None
    }

    /// Optionally own the NC sumcheck prolog absorbs as well as the column
    /// rounds. `None` keeps the older host-prolog plus device-rounds path.
    fn col_round_trace_with_prolog(&mut self, request: NcColTraceRequest) -> Option<NcColRoundTrace> {
        let _ = request;
        None
    }
}
