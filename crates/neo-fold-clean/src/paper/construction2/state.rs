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
/// ## What binds what
///
/// - `z_i` and `public_trace` chain `f_prime_chunk_public_digest`, which
///   under the direct-CCS interim is a **step/shape digest only** — it
///   absorbs `(commitment.d, commitment.kappa, m_in, start_index,
///   fresh.len())` but **not** `claim.x` or `claim.c.data` (both depend
///   on the recursive-link `x` in direct-CCS and would otherwise create
///   a hash fixed point; see `digest::f_prime_chunk_claim_digest`). So
///   for same-shape chunks across different proofs `z_i` and
///   `public_trace` can be identical even though the underlying CCS
///   instances differ.
/// - The content-binding public coordinate is `acc_digest` after
///   finalization: it equals `digest(final running CE claims)`. The
///   **non-replay IVC verifier** (`lifecycle::verify_uncompressed`)
///   re-runs the terminal NIFS fold against the prover-snapshotted
///   pre-fold inputs and recomputes `acc_digest` from the derived
///   final running; the **chain-replay verifier**
///   (`lifecycle::verify_uncompressed_audit`) additionally walks the
///   per-step NIFS chain, so a tamper to a public batch's `x` or
///   `c.data` breaks an algebraic check before `acc_digest` is even
///   compared.
/// - `z_i` / `public_trace` are still useful for **domain separation
///   per step** inside the F' transcript prefix; they are not the
///   authority for "this chunk had these claims."
#[derive(Clone, Debug)]
pub struct State {
    /// i — chunk counter (number of `extend` calls).
    pub chunk_count: u64,
    /// total number of fresh CCS instances passed via `latest` so far.
    pub step_count: u64,
    /// z_0 — initial boundary digest, fixed by `Preprocessing`.
    pub z_0: [u8; 32],
    /// z_i — chained F' step/shape digest. Domain-separates F' transcript
    /// prefixes per step; not the authority for chunk *content*. Content
    /// binding lives on the accumulator path (see `acc_digest` and the
    /// terminal-fold re-run that `verify_uncompressed` performs via
    /// `final_fold.terminal_inputs`).
    pub z_i: [u8; 32],
    /// Initial app / VM semantic state digest, fixed at `State::base` time.
    /// Exposed on the terminal public image so external verifiers can pin
    /// the start state. Never mutated by `advance_state`.
    pub initial_semantic_state_digest: [u8; 32],
    /// Current app / VM semantic state digest. Fed to the `semantic_acc`
    /// lane of `state_x_out_digest`; `acc_digest` remains the
    /// Construction-2 / SuperNeo accumulator handle in the
    /// `construction2_acc` lane.
    pub semantic_state_digest: [u8; 32],
    /// pc_i — program counter (always `TRIVIAL_PC` in this build).
    pub pc: u64,
    /// Derived accumulator handle. After finalization, equals the digest
    /// of the final running CE claims and **is** the content-binding
    /// public coordinate. The non-replay IVC verifier
    /// (`lifecycle::verify_uncompressed`) re-derives this value from the
    /// terminal NIFS.V re-run's output running and rejects on mismatch;
    /// the chain-replay verifier
    /// (`lifecycle::verify_uncompressed_audit`) additionally walks the
    /// per-step NIFS.V chain, so any tamper to a public batch's `(c, x)`
    /// breaks an algebraic check before `acc_digest` is even compared.
    /// Pre-finalization (trailing `latest` still un-folded), this is a
    /// derived handle, not the final authority.
    pub acc_digest: [u8; 32],
    /// Chained F' step/shape digest, same role as `z_i`: useful for
    /// per-step domain separation, not the chunk-content authority.
    pub public_trace: [u8; 32],
    /// The fold pair `(U_i, u_i)`, tagged Initial/Active.
    pub proof: ProofState,
}

impl State {
    /// Initial case — `chunk_count = 0`, `proof = ProofState::Initial`, `z_i = z_0`.
    /// Constructed via [`crate::lifecycle::preprocess`] then `prove(&prep, [])`.
    pub fn base(z_0: [u8; 32], public_trace: [u8; 32], acc_digest: [u8; 32], semantic_state_digest: [u8; 32]) -> Self {
        Self {
            chunk_count: 0,
            step_count: 0,
            z_0,
            z_i: z_0,
            pc: TRIVIAL_PC,
            initial_semantic_state_digest: semantic_state_digest,
            semantic_state_digest,
            acc_digest,
            public_trace,
            proof: ProofState::Initial,
        }
    }
}
