//! `StepProof` — one IVC step's wire-format output.
//!
//! `FoldProof` distinguishes the base case (no NIFS.P ran) from the
//! recursive case (NIFS.P folded the previous latest into running). The
//! verifier matches on this variant in `paper::f_prime::verify`.

use crate::paper::construction2::enc_inst::EncInst;
use crate::paper::nifs::NifsProof;

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
    Recursive(NifsProof),
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
    pub x_out: EncInst,
}

/// Terminal fold proof emitted when finalization folds the last trailing
/// `latest` into the running accumulator without advancing the chunk/state
/// counters.
#[derive(Clone, Debug)]
pub struct FinalFoldProof {
    pub nifs: NifsProof,
    pub x_out: EncInst,
}
