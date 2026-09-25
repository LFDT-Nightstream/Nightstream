//! `State` — IVC carrier for Construction 2 §6.3.
//!
//! Holds the chain-binding fields the hash chain absorbs into `x_out`
//! plus a `proof: ProofState` slot for the soundness-relevant fold pair.
//! The auditor's eye reads State's fields top-down as paper symbols.

use crate::paper::construction2::proof_state::ProofState;
use crate::paper::construction2::TRIVIAL_PC;

/// Public state coordinates that do not depend on the pending `latest`
/// instance. The recursive frontend uses this form while it synthesizes the
/// real next instance from the fold result.
#[derive(Clone, Debug)]
pub(crate) struct StateCoordinates {
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_0: [u8; 32],
    pub z_i: [u8; 32],
    pub pc: u64,
    pub initial_semantic_state_digest: [u8; 32],
    pub semantic_state_digest: [u8; 32],
    pub acc_digest: [u8; 32],
    pub public_trace: [u8; 32],
    pub nebula: Option<crate::paper::construction2::nebula_lane::NebulaLane>,
}

impl StateCoordinates {
    pub(crate) fn with_proof(self, proof: ProofState) -> State {
        State {
            chunk_count: self.chunk_count,
            step_count: self.step_count,
            z_0: self.z_0,
            z_i: self.z_i,
            pc: self.pc,
            initial_semantic_state_digest: self.initial_semantic_state_digest,
            semantic_state_digest: self.semantic_state_digest,
            acc_digest: self.acc_digest,
            public_trace: self.public_trace,
            proof,
            nebula: self.nebula,
        }
    }
}

impl From<&State> for StateCoordinates {
    fn from(state: &State) -> Self {
        Self {
            chunk_count: state.chunk_count,
            step_count: state.step_count,
            z_0: state.z_0,
            z_i: state.z_i,
            pc: state.pc,
            initial_semantic_state_digest: state.initial_semantic_state_digest,
            semantic_state_digest: state.semantic_state_digest,
            acc_digest: state.acc_digest,
            public_trace: state.public_trace,
            nebula: state.nebula.clone(),
        }
    }
}

/// IVC carrier — Construction-2 §6.3.
///
/// `proof` holds the soundness-relevant fold pair (running, latest); the
/// rest are chain-binding fields the hash chain absorbs into `x_out`.
///
/// ## What binds what
///
/// - `z_i` and `public_trace` chain `f_prime_chunk_public_digest`, which
///   is a **step/shape digest only** — it
///   absorbs `(commitment.d, commitment.kappa, m_in, start_index,
///   fresh.len())` but **not** `claim.x` or `claim.c.data` (both depend
///   on the recursive-link `x` and would otherwise create
///   a hash fixed point; see `digest::f_prime_chunk_claim_digest`). So
///   for same-shape chunks across different proofs `z_i` and
///   `public_trace` can be identical even though the underlying CCS
///   instances differ.
/// - The content-binding public coordinate is `acc_digest` after
///   finalization: it equals `digest(final running CE claims)`. The
///   **non-replay IVC verifier** (`lifecycle::verify_uncompressed`)
///   checks the running accumulator and latest F' relation (or, for
///   Nebula, re-runs the terminal NIFS fold) and recomputes
///   `acc_digest`; the **chain-replay verifier**
///   (`lifecycle::verify_uncompressed_audit`) additionally walks the
///   per-step NIFS chain, so a tamper to a public batch's `x` or
///   `c.data` breaks an algebraic check before `acc_digest` is even
///   compared.
/// - `z_i` is still useful for **domain separation per step** inside
///   the F' transcript prefix; it is not the authority for "this chunk
///   had these claims." `public_trace` is retained in the public image
///   shape but mirrors `z_i` after the first step so the hot F' image
///   does not spend a second bit-backed hash trace over the same
///   shape-only chunk digest.
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
    /// running/latest checks in `verify_uncompressed`).
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
    /// Derived accumulator handle. For plain HyperNova proofs it identifies
    /// the running CE claims while `latest` remains separately checked; after
    /// Nebula finalization it identifies the terminal-fold output. The compact
    /// verifier re-derives this value from the authenticated running claims;
    /// the chain-replay verifier
    /// (`lifecycle::verify_uncompressed_audit`) additionally walks the
    /// per-step NIFS.V chain, so any tamper to a public batch's `(c, x)`
    /// breaks an algebraic check before `acc_digest` is even compared.
    /// The digest is compression, not standalone authority: the verifier also
    /// checks the corresponding opened relation witnesses.
    pub acc_digest: [u8; 32],
    /// Chained F' step/shape digest, same role as `z_i`. In the current
    /// compact F' shape this mirrors `z_i` after a step advances; the
    /// field remains present for public-image compatibility while we
    /// avoid a duplicate `public_trace_update` hash trace.
    pub public_trace: [u8; 32],
    /// The fold pair `(U_i, u_i)`, tagged Initial/Active.
    pub proof: ProofState,
    /// Nebula commitment-carrying memory lane (the carried-lane rules); `None` for
    /// plain chains. Its digest is absorbed into `state_x_out` and the
    /// F′ step transcript (present-only, so plain chains keep the
    /// pre-Nebula preimages and the F′ R1CS mirrors stay in parity).
    pub nebula: Option<crate::paper::construction2::nebula_lane::NebulaLane>,
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
            nebula: None,
        }
    }
}
