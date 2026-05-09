//! Owns the shared opening-claim and time-opening summary boundary.
//!
//! It owns:
//! - canonical opening claim/source/domain records
//! - reduced opening group summaries
//! - the final time-opening proof summary surface
//! - `time`, which builds and verifies the shared time-opening reduction
//!
//! It does not own:
//! - VM-specific opening manifests

pub mod time;

use neo_math::K;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum OpeningSource {
    MainLane,
    ExtensionKernel,
    ExtensionRoot,
    Rv32imKernel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum OpeningDomain {
    Cpu,
    Mem,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OpeningClaim {
    pub source: OpeningSource,
    pub domain: OpeningDomain,
    pub point: Vec<K>,
    pub ordinal: u64,
    pub column_ids: Vec<u32>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TimeOpeningGroupSummary {
    pub sources: Vec<OpeningSource>,
    pub domain: OpeningDomain,
    pub point: Vec<K>,
    pub claim_indices: Vec<usize>,
    pub coefficients: Vec<K>,
    pub group_digest: [u8; 32],
    pub reduced_digest: [u8; 32],
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TimeOpeningUnificationProof {
    pub claimed_sum: K,
    pub round_polys: Vec<Vec<K>>,
    pub r_unify: Vec<K>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TimeOpeningCompactProof {
    pub unification: TimeOpeningUnificationProof,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeOpeningProofSummary {
    pub manifest_digest: [u8; 32],
    pub proof_digest: [u8; 32],
    pub groups: Vec<TimeOpeningGroupSummary>,
    pub unification: TimeOpeningUnificationProof,
    pub can_unify: bool,
    pub unified_domain: OpeningDomain,
    pub unified_point: Vec<K>,
    pub unified_digest: [u8; 32],
}
