//! `State` — IVC carrier for Construction 2 §6.3.
//!
//! Holds the chain-binding fields the hash chain absorbs into `x_out`
//! plus a `proof: ProofState` slot for the soundness-relevant fold pair.
//! The auditor's eye reads State's fields top-down as paper symbols.

use crate::paper::construction2::proof_state::ProofState;
use crate::paper::construction2::TRIVIAL_PC;

/// IVC carrier — Construction-2 §6.3.
///
/// `proof` holds the soundness-relevant fold pair (running, latest); the
/// rest are chain-binding fields the hash chain absorbs into `x_out`.
///
/// `z_i` and `public_trace` are **chained internally** from each chunk's
/// public-instance digest — the caller does not pick them.
#[derive(Clone, Debug)]
pub struct State {
    /// i — chunk counter (number of `extend` calls).
    pub chunk_count: u64,
    /// total number of fresh CCS instances passed via `latest` so far.
    pub step_count: u64,
    /// z_0 — initial boundary digest, fixed by `Preprocessing`.
    pub z_0: [u8; 32],
    /// z_i — current boundary digest, chained via `boundary_update_digest`.
    pub z_i: [u8; 32],
    /// pc_i — program counter (always `TRIVIAL_PC` in this build).
    pub pc: u64,
    /// Derived accumulator handle used inside `x_out`/public-image hashing.
    /// Not verifier authority: verification replays reductions and compares
    /// the actual final accumulator claims.
    pub acc_digest: [u8; 32],
    /// Running chain of every chunk's public-instance digest so far.
    pub public_trace: [u8; 32],
    /// The fold pair `(U_i, u_i)`, tagged Initial/Active.
    pub proof: ProofState,
}

impl State {
    /// Initial case — `chunk_count = 0`, `proof = ProofState::Initial`, `z_i = z_0`.
    /// Constructed via [`crate::lifecycle::preprocess`] then `prove(&prep, [])`.
    pub fn base(z_0: [u8; 32], public_trace: [u8; 32], acc_digest: [u8; 32]) -> Self {
        Self {
            chunk_count: 0,
            step_count: 0,
            z_0,
            z_i: z_0,
            pc: TRIVIAL_PC,
            acc_digest,
            public_trace,
            proof: ProofState::Initial,
        }
    }
}
