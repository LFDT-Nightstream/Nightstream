//! Grouping and reduction construction for time-opening claims.

use neo_math::{from_complex, K};
use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::opening::{OpeningClaim, OpeningDomain, OpeningSource};

use super::digest::{
    append_k_vec, append_point, digest_reduced_group, digest_reduction_group, digest_unified_reduction,
    opening_domain_tag, opening_source_tag, sample_group_coeffs,
};
use super::types::{OpeningManifest, OpeningReduction, OpeningReductionGroup};

pub(super) fn build_reduction(manifest: &OpeningManifest) -> OpeningReduction {
    let groups = build_reduction_groups(manifest);
    let (can_unify, unified_domain, unified_point) = compute_unified_anchor(&groups);
    let unified_digest = digest_unified_reduction(&groups, can_unify, unified_domain, &unified_point);

    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction");
    tr.append_message(b"neo.fold.next/time_opening/reduction_manifest", &manifest.digest);
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_group_count",
        &[groups.len() as u64],
    );
    for group in &groups {
        tr.append_u64s(
            b"neo.fold.next/time_opening/reduction_group_meta",
            &[
                opening_domain_tag(group.domain),
                group.sources.len() as u64,
                group.claim_indices.len() as u64,
                group.coefficients.len() as u64,
            ],
        );
        let source_tags: Vec<u64> = group
            .sources
            .iter()
            .map(|&source| opening_source_tag(source))
            .collect();
        tr.append_u64s(b"neo.fold.next/time_opening/reduction_group_sources", &source_tags);
        append_point(
            &mut tr,
            b"neo.fold.next/time_opening/reduction_group_point",
            &group.point,
        );
        append_k_vec(
            &mut tr,
            b"neo.fold.next/time_opening/reduction_group_coefficients",
            &group.coefficients,
        );
        let claim_indices_u64: Vec<u64> = group.claim_indices.iter().map(|&idx| idx as u64).collect();
        tr.append_u64s(
            b"neo.fold.next/time_opening/reduction_group_indices",
            &claim_indices_u64,
        );
        tr.append_message(
            b"neo.fold.next/time_opening/reduction_group_digest",
            &group.group_digest,
        );
        tr.append_message(
            b"neo.fold.next/time_opening/reduction_group_reduced_digest",
            &group.reduced_digest,
        );
    }
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_unify_meta",
        &[
            can_unify as u64,
            opening_domain_tag(unified_domain),
            unified_point.len() as u64,
        ],
    );
    append_point(
        &mut tr,
        b"neo.fold.next/time_opening/reduction_unify_point",
        &unified_point,
    );
    tr.append_message(b"neo.fold.next/time_opening/reduction_unified_digest", &unified_digest);

    OpeningReduction {
        groups,
        can_unify,
        unified_domain,
        unified_point,
        unified_digest,
        digest: tr.digest32(),
    }
}

pub(super) fn build_reduction_groups(manifest: &OpeningManifest) -> Vec<OpeningReductionGroup> {
    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < manifest.claims.len() {
        let first = &manifest.claims[start];
        let first_family_tag = reduction_family_tag(first);
        let mut end = start + 1;
        while end < manifest.claims.len() {
            let next = &manifest.claims[end];
            if next.source != first.source || next.domain != first.domain || next.point != first.point {
                break;
            }
            if reduction_family_tag(next) != first_family_tag {
                break;
            }
            end += 1;
        }
        let claim_indices: Vec<usize> = (start..end).collect();
        let sources = group_sources(manifest, &claim_indices);
        let group_digest = digest_reduction_group(manifest, &sources, first.domain, &first.point, &claim_indices);
        let coefficients = sample_group_coeffs(manifest, &group_digest, claim_indices.len());
        let reduced_digest = digest_reduced_group(manifest, &group_digest, &claim_indices, &coefficients);
        groups.push(OpeningReductionGroup {
            sources,
            domain: first.domain,
            point: first.point.clone(),
            claim_indices,
            coefficients,
            group_digest,
            reduced_digest,
        });
        start = end;
    }
    groups
}

fn reduction_family_tag(claim: &OpeningClaim) -> Option<u64> {
    match claim.source {
        OpeningSource::ExtensionKernel | OpeningSource::ExtensionRoot => Some(claim.ordinal),
        OpeningSource::MainLane | OpeningSource::Rv32imKernel => None,
    }
}

fn compute_unified_anchor(groups: &[OpeningReductionGroup]) -> (bool, OpeningDomain, Vec<K>) {
    let Some(first) = groups.first() else {
        return (true, OpeningDomain::Cpu, Vec::new());
    };
    let can_unify = groups
        .iter()
        .all(|group| group.domain == first.domain && group.point == first.point);
    if can_unify {
        return (true, first.domain, first.point.clone());
    }

    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction_unify_anchor");
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_unify_anchor_group_count",
        &[groups.len() as u64],
    );
    for group in groups {
        tr.append_u64s(
            b"neo.fold.next/time_opening/reduction_unify_anchor_group_meta",
            &[
                opening_domain_tag(group.domain),
                group.sources.len() as u64,
                group.point.len() as u64,
                group.claim_indices.len() as u64,
                group.coefficients.len() as u64,
            ],
        );
        let source_tags: Vec<u64> = group
            .sources
            .iter()
            .map(|&source| opening_source_tag(source))
            .collect();
        tr.append_u64s(
            b"neo.fold.next/time_opening/reduction_unify_anchor_group_sources",
            &source_tags,
        );
        append_point(
            &mut tr,
            b"neo.fold.next/time_opening/reduction_unify_anchor_group_point",
            &group.point,
        );
        append_k_vec(
            &mut tr,
            b"neo.fold.next/time_opening/reduction_unify_anchor_group_coefficients",
            &group.coefficients,
        );
        tr.append_message(
            b"neo.fold.next/time_opening/reduction_unify_anchor_group_digest",
            &group.group_digest,
        );
        tr.append_message(
            b"neo.fold.next/time_opening/reduction_unify_anchor_group_reduced_digest",
            &group.reduced_digest,
        );
    }
    let point_len = first.point.len();
    let unified_point = (0..point_len)
        .map(|_| {
            let re = tr.challenge_field(b"neo.fold.next/time_opening/reduction_unify_anchor/re");
            let im = tr.challenge_field(b"neo.fold.next/time_opening/reduction_unify_anchor/im");
            from_complex(re, im)
        })
        .collect();
    (false, OpeningDomain::Cpu, unified_point)
}

fn group_sources(manifest: &OpeningManifest, claim_indices: &[usize]) -> Vec<OpeningSource> {
    let mut sources = Vec::new();
    for &claim_idx in claim_indices {
        let source = manifest.claims[claim_idx].source;
        if !sources.contains(&source) {
            sources.push(source);
        }
    }
    sources.sort_by_key(|&source| opening_source_tag(source));
    sources
}
