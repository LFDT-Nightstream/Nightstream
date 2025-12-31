use crate::sumcheck_proof::BatchedAddrProof;
use serde::{Deserialize, Serialize};

// ============================================================================
// Proof metadata (Route A)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwistProof<F> {
    /// Address-domain sum-check metadata for Route A (two-claim batch: read/write).
    pub addr_pre: BatchedAddrProof<F>,
    pub val_eval: Option<TwistValEvalProof<F>>,
}

impl<F: Default> Default for TwistProof<F> {
    fn default() -> Self {
        Self {
            addr_pre: BatchedAddrProof::default(),
            val_eval: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwistValEvalProof<F> {
    /// Σ_t Inc(r_addr, t) · LT(t, r_time) (init term excluded).
    pub claimed_inc_sum_lt: F,
    /// Sum-check rounds for the LT-weighted claim (ell_n rounds, cycle/time variables).
    pub rounds_lt: Vec<Vec<F>>,

    /// Σ_t Inc(r_addr, t) (total increment over the whole chunk).
    pub claimed_inc_sum_total: F,
    /// Sum-check rounds for the total-increment claim (ell_n rounds, cycle/time variables).
    pub rounds_total: Vec<Vec<F>>,

    /// Optional rollover claim for linking consecutive chunks (Route A):
    /// Σ_t Inc_prev(r_addr_current, t) (total increment over the *previous* chunk, evaluated at the *current* r_addr).
    ///
    /// Present iff the prover links this step to a previous step.
    pub claimed_prev_inc_sum_total: Option<F>,
    /// Sum-check rounds for the rollover total-increment claim (ell_n rounds).
    pub rounds_prev_total: Option<Vec<Vec<F>>>,
}
