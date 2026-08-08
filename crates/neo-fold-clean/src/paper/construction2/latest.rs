//! `LatestInstance` — the K fresh CCS instances u_i the next NIFS.P will
//! fold into the running accumulator.
//!
//! These instances encode F'_{i-1}. Authoritative frontends synthesize them
//! from the verifier-derived post-fold coordinates.

use crate::paper::relations::{CcsClaim, CcsInstance};

/// The K fresh CCS instances the next NIFS.P will fold into the running
/// accumulator.
#[derive(Clone, Debug)]
pub struct LatestInstance {
    pub instances: Vec<CcsInstance>,
}

impl LatestInstance {
    pub fn from_instances(instances: Vec<CcsInstance>) -> Self {
        Self { instances }
    }

    /// Verifier-side view: just the K public claims, no witnesses.
    pub fn claims(&self) -> Vec<CcsClaim> {
        self.instances.iter().map(|i| i.claim.clone()).collect()
    }
}
