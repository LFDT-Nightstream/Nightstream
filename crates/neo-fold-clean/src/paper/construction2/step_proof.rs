//! `StepProof` — one IVC step's wire-format output.
//!
//! `FoldProof` distinguishes the base case (no NIFS.P ran) from the
//! recursive case (NIFS.P folded the previous latest into running). The
//! verifier matches on this variant in `paper::f_prime::verify`.

use crate::paper::construction2::enc_inst::EncInst;
use crate::paper::construction2::{LatestInstance, RunningInstance};
use crate::paper::nifs::{NifsProof, NifsProofCarrier};

/// What kind of fold this step produced.
///
/// `NoFold` (not "Base") because the variant describes what happened —
/// no NIFS.P ran — rather than naming a state. `Recursive` carries the
/// NIFS replay material when NIFS.P did run.
#[derive(Clone, Debug)]
pub enum FoldProof {
    /// i = 0 initialization step: no NIFS.P ran, so there's nothing to replay.
    NoFold,
    /// i ≥ 1: NIFS.P folded the previous latest into running.
    Recursive(NifsProofCarrier),
}

impl FoldProof {
    pub fn recursive_materialized(proof: NifsProof) -> Self {
        Self::Recursive(NifsProofCarrier::materialized(proof))
    }

    pub fn recursive_carrier(proof: NifsProofCarrier) -> Self {
        Self::Recursive(proof)
    }

    pub fn materialized_recursive(&self) -> Result<Option<NifsProof>, crate::paper::nifs::Error> {
        match self {
            Self::NoFold => Ok(None),
            Self::Recursive(proof) => Ok(Some(proof.materialize()?)),
        }
    }
}

/// One IVC step's output: the fold proof + the F' hash-chain output.
///
/// `fold` carries everything NIFS.V needs to replay (the three sub-proofs
/// plus the K+k Π_CCS output claims via `pi_ccs::Proof::outputs`). The
/// claims that were folded come from the verifier's own walking state
/// (`state.proof.latest.claims()` at the time of verify_step).
#[derive(Clone, Debug)]
pub struct StepProof {
    pub fold: FoldProof,
    /// Nebula segment-open payload (spec §6.2, L0b): the prover-claimed
    /// per-lane `D_pre` chain digests, present exactly on the step that
    /// opens a segment. The verifier replays `open_segment` from it; its
    /// authority is retroactive via the close equality `D_seen == D_pre`.
    pub nebula_open: Option<[[neo_math::F; 4]; 3]>,
    /// Outgoing semantic state digest for this F' step. Stateless
    /// frontends set this equal to the outgoing accumulator digest,
    /// preserving the legacy `semantic_acc == construction2_acc` path.
    pub semantic_state_digest: [u8; 32],
    pub x_out: EncInst,
}

/// Inputs to the terminal NIFS fold, recorded by the prover so the
/// non-replay IVC verifier can re-run NIFS.V on the final step alone.
///
/// **Why this exists.** A non-replay IVC verifier must still authenticate
/// the chain through the **terminal** NIFS.V (HyperNova §6.3 Construction 2;
/// SuperNeo §7). Without storing these inputs explicitly the verifier
/// cannot rederive the verifier-driven `r` for the post-fold running, so
/// the running's CE relation check would be sound only at a prover-chosen
/// point — the soundness gap the council flagged on Phase 1.7's first
/// attempt.
///
/// **Witnesses are stripped.** Both `pre_final_running` and `latest` carry
/// claims only; the prover-private witness matrices never cross the
/// proof boundary. `latest.instances[i].witness` is a placeholder
/// zero-shape matrix when read from a proof.
#[derive(Clone, Debug)]
pub struct TerminalFoldInputs {
    /// `U_{N-1}` — the running accumulator's CE claims (and Π_RLC parent
    /// authority) **before** the terminal fold. Witness matrices are
    /// empty when this is read from a proof.
    pub pre_final_running: RunningInstance,
    /// `u_N` — the trailing latest's CCS claims (the batch that was about
    /// to be folded into the running). Witnesses inside are empty when
    /// read from a proof.
    pub latest: LatestInstance,
    /// Nebula lane before terminal finalization consumes the trailing
    /// delayed claim. `None` for plain chains. The verifier advances this
    /// constant-size public state and binds the result to the terminal lane.
    pub pre_nebula: Option<crate::paper::construction2::NebulaLane>,
}

/// Terminal fold proof emitted when finalization folds the last trailing
/// `latest` into the running accumulator without advancing the chunk/state
/// counters.
///
/// `terminal_inputs` lets a non-replay IVC verifier re-run NIFS.V on the
/// terminal fold without having to walk all prior steps. See
/// [`TerminalFoldInputs`] and [`crate::lifecycle::verify_uncompressed`].
#[derive(Clone, Debug)]
pub struct FinalFoldProof {
    pub nifs: NifsProof,
    pub x_out: EncInst,
    pub terminal_inputs: TerminalFoldInputs,
}
