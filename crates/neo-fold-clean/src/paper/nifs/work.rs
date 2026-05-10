//! Witness-threading helpers used by NIFS.P.
//!
//! NIFS.P needs to assemble a parallel `(claims, witnesses)` array of
//! length K+k for Π_RLC after Π_CCS hands it back the K+k output claims.
//! These helpers are pure data-movement — no math.

use crate::paper::relations::{CcsInstance, WitnessMat};

/// Pull the witness `Z` matrix out of every fresh CCS instance, in input
/// order. These are the prover-side data Π_RLC and Π_DEC consume.
pub(super) fn collect_fresh_witness_mats(fresh: &[CcsInstance]) -> Vec<WitnessMat> {
    fresh.iter().map(|i| i.witness.Z.clone()).collect()
}

/// Concatenate the K fresh witness Z's and the k running witness Z's into
/// the K+k array Π_RLC expects, parallel to the Π_CCS output claims.
pub(super) fn chain_witnesses(fresh: Vec<WitnessMat>, running: Vec<WitnessMat>) -> Vec<WitnessMat> {
    let mut out = Vec::with_capacity(fresh.len() + running.len());
    out.extend(fresh);
    out.extend(running);
    out
}
