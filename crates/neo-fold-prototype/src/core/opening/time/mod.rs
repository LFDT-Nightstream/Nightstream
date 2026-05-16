//! Owns the final opening manifest, reduction, and unification proof for the active pipeline.
//!
//! The public flow is intentionally small: collect claims, build a canonical
//! manifest, reduce compatible claims into groups, and prove or verify the
//! unification sumcheck. The submodules own those steps separately.

mod digest;
mod main_lane;
mod manifest;
mod reduction;
mod types;
mod unification;

pub use digest::time_opening_compact_proof_digest;
pub use main_lane::main_lane_opening_claims;

use neo_reductions::error::PiCcsError;

use crate::opening::{OpeningClaim, TimeOpeningCompactProof, TimeOpeningGroupSummary, TimeOpeningProofSummary};

use digest::digest_opening_proof;
use manifest::build_manifest;
use reduction::build_reduction;
use types::OpeningReductionGroup;
use unification::{prove_opening_unification, verify_opening_unification};

pub fn prove_time_opening(
    main_lane_claims: &[OpeningClaim],
    extension_claims: &[OpeningClaim],
) -> Result<TimeOpeningProofSummary, PiCcsError> {
    let manifest = build_manifest(main_lane_claims, extension_claims)?;
    let reduction = build_reduction(&manifest);
    let unification = prove_opening_unification(&reduction)?;
    Ok(TimeOpeningProofSummary {
        manifest_digest: manifest.digest,
        proof_digest: digest_opening_proof(&reduction, &unification),
        groups: reduction.groups.iter().map(summarize_group).collect(),
        unification,
        can_unify: reduction.can_unify,
        unified_domain: reduction.unified_domain,
        unified_point: reduction.unified_point.clone(),
        unified_digest: reduction.unified_digest,
    })
}

pub fn prove_time_opening_compact(
    main_lane_claims: &[OpeningClaim],
    extension_claims: &[OpeningClaim],
) -> Result<TimeOpeningCompactProof, PiCcsError> {
    let manifest = build_manifest(main_lane_claims, extension_claims)?;
    let reduction = build_reduction(&manifest);
    Ok(TimeOpeningCompactProof {
        unification: prove_opening_unification(&reduction)?,
    })
}

pub fn verify_time_opening(
    main_lane_claims: &[OpeningClaim],
    extension_claims: &[OpeningClaim],
    summary: &Option<TimeOpeningProofSummary>,
) -> Result<(), PiCcsError> {
    let summary = summary
        .as_ref()
        .ok_or_else(|| PiCcsError::ProtocolError("missing time-opening summary".into()))?;
    let manifest = build_manifest(main_lane_claims, extension_claims)?;
    if summary.manifest_digest != manifest.digest {
        return Err(PiCcsError::ProtocolError(
            "time-opening manifest digest mismatch".into(),
        ));
    }
    let reduction = build_reduction(&manifest);
    let expected_groups: Vec<TimeOpeningGroupSummary> = reduction.groups.iter().map(summarize_group).collect();
    if summary.groups != expected_groups {
        return Err(PiCcsError::ProtocolError("time-opening group summary mismatch".into()));
    }
    if summary.can_unify != reduction.can_unify {
        return Err(PiCcsError::ProtocolError(
            "time-opening unification flag mismatch".into(),
        ));
    }
    if summary.unified_domain != reduction.unified_domain {
        return Err(PiCcsError::ProtocolError("time-opening unified domain mismatch".into()));
    }
    if summary.unified_point != reduction.unified_point {
        return Err(PiCcsError::ProtocolError("time-opening unified point mismatch".into()));
    }
    if summary.unified_digest != reduction.unified_digest {
        return Err(PiCcsError::ProtocolError("time-opening unified digest mismatch".into()));
    }
    verify_opening_unification(&reduction, &summary.unification)?;
    if summary.proof_digest != digest_opening_proof(&reduction, &summary.unification) {
        return Err(PiCcsError::ProtocolError("time-opening proof digest mismatch".into()));
    }
    Ok(())
}

pub fn verify_time_opening_compact(
    main_lane_claims: &[OpeningClaim],
    extension_claims: &[OpeningClaim],
    proof: &TimeOpeningCompactProof,
) -> Result<(), PiCcsError> {
    let manifest = build_manifest(main_lane_claims, extension_claims)?;
    let reduction = build_reduction(&manifest);
    verify_opening_unification(&reduction, &proof.unification)
}

fn summarize_group(group: &OpeningReductionGroup) -> TimeOpeningGroupSummary {
    TimeOpeningGroupSummary {
        sources: group.sources.clone(),
        domain: group.domain,
        point: group.point.clone(),
        claim_indices: group.claim_indices.clone(),
        coefficients: group.coefficients.clone(),
        group_digest: group.group_digest,
        reduced_digest: group.reduced_digest,
    }
}
