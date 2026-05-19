//! Internal data shapes for the time-opening reduction.

use neo_math::K;

use crate::opening::{OpeningClaim, OpeningDomain, OpeningSource};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OpeningManifest {
    pub(super) claims: Vec<OpeningClaim>,
    pub(super) digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OpeningReduction {
    pub(super) groups: Vec<OpeningReductionGroup>,
    pub(super) can_unify: bool,
    pub(super) unified_domain: OpeningDomain,
    pub(super) unified_point: Vec<K>,
    pub(super) unified_digest: [u8; 32],
    pub(super) digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OpeningReductionGroup {
    pub(super) sources: Vec<OpeningSource>,
    pub(super) domain: OpeningDomain,
    pub(super) point: Vec<K>,
    pub(super) claim_indices: Vec<usize>,
    pub(super) coefficients: Vec<K>,
    pub(super) group_digest: [u8; 32],
    pub(super) reduced_digest: [u8; 32],
}
