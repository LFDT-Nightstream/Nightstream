use crate::sumcheck_proof::BatchedAddrProof;
use serde::{Deserialize, Serialize};

// ============================================================================
// Proof metadata (Route A)
// ============================================================================

/// Route A Shout proof metadata.
///
/// In Route A, the time-domain rounds are carried by the shard’s `BatchedTimeProof`.
/// Shout contributes only the address-domain sum-check metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShoutProof<F> {
    /// Address-domain sum-check metadata for Route A (single-claim batch).
    pub addr_pre: BatchedAddrProof<F>,
}

impl<F: Default> Default for ShoutProof<F> {
    fn default() -> Self {
        Self {
            addr_pre: BatchedAddrProof::default(),
        }
    }
}
