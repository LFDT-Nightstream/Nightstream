use serde::{Deserialize, Serialize};

/// Generic batched sumcheck proof metadata for an address-domain subprotocol.
///
/// `claimed_sums.len()` must equal `round_polys.len()` (one entry per claim).
/// `r_addr` is the shared terminal point derived from the transcript.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchedAddrProof<F> {
    pub claimed_sums: Vec<F>,
    /// `round_polys[claim][round] = coeffs`.
    pub round_polys: Vec<Vec<Vec<F>>>,
    pub r_addr: Vec<F>,
}

impl<F: Default> Default for BatchedAddrProof<F> {
    fn default() -> Self {
        Self {
            claimed_sums: Vec::new(),
            round_polys: Vec::new(),
            r_addr: Vec::new(),
        }
    }
}
