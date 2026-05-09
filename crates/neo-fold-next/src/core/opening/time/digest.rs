//! Transcript and digest helpers for the time-opening boundary.

use neo_math::{from_complex, KExtensions, K};
use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::opening::{
    OpeningClaim, OpeningDomain, OpeningSource, TimeOpeningCompactProof, TimeOpeningUnificationProof,
};

use super::types::{OpeningManifest, OpeningReduction, OpeningReductionGroup};

pub fn time_opening_compact_proof_digest(proof: &TimeOpeningCompactProof) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/compact_proof");
    tr.append_fields(
        b"neo.fold.next/time_opening/compact_proof/claimed_sum",
        &proof.unification.claimed_sum.as_coeffs(),
    );
    tr.append_u64s(
        b"neo.fold.next/time_opening/compact_proof/meta",
        &[
            proof.unification.round_polys.len() as u64,
            proof.unification.r_unify.len() as u64,
        ],
    );
    for round in &proof.unification.round_polys {
        append_k_vec(&mut tr, b"neo.fold.next/time_opening/compact_proof/round", round);
    }
    append_k_vec(
        &mut tr,
        b"neo.fold.next/time_opening/compact_proof/selector_point",
        &proof.unification.r_unify,
    );
    tr.digest32()
}

pub(super) fn digest_opening_proof(
    reduction: &OpeningReduction,
    unification: &TimeOpeningUnificationProof,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/proof");
    tr.append_message(b"neo.fold.next/time_opening/proof_reduction_digest", &reduction.digest);
    tr.append_fields(
        b"neo.fold.next/time_opening/proof_unify_claimed_sum",
        &unification.claimed_sum.as_coeffs(),
    );
    tr.append_u64s(
        b"neo.fold.next/time_opening/proof_unify_meta",
        &[unification.round_polys.len() as u64, unification.r_unify.len() as u64],
    );
    for round in &unification.round_polys {
        append_k_vec(&mut tr, b"neo.fold.next/time_opening/proof_unify_round", round);
    }
    append_k_vec(
        &mut tr,
        b"neo.fold.next/time_opening/proof_unify_point",
        &unification.r_unify,
    );
    tr.digest32()
}

pub(super) fn digest_reduction_group(
    manifest: &OpeningManifest,
    sources: &[OpeningSource],
    domain: OpeningDomain,
    point: &[K],
    claim_indices: &[usize],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction_group");
    tr.append_message(b"neo.fold.next/time_opening/reduction_group_manifest", &manifest.digest);
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_group_meta",
        &[
            opening_domain_tag(domain),
            sources.len() as u64,
            point.len() as u64,
            claim_indices.len() as u64,
        ],
    );
    let source_tags: Vec<u64> = sources
        .iter()
        .map(|&source| opening_source_tag(source))
        .collect();
    tr.append_u64s(b"neo.fold.next/time_opening/reduction_group_sources", &source_tags);
    append_point(&mut tr, b"neo.fold.next/time_opening/reduction_group_point", point);
    let claim_indices_u64: Vec<u64> = claim_indices.iter().map(|&idx| idx as u64).collect();
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_group_indices",
        &claim_indices_u64,
    );
    tr.digest32()
}

pub(super) fn digest_reduced_group(
    manifest: &OpeningManifest,
    group_digest: &[u8; 32],
    claim_indices: &[usize],
    coefficients: &[K],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction_group_value");
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_group_value_manifest",
        &manifest.digest,
    );
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_group_value_group_digest",
        group_digest,
    );
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_group_value_len",
        &[claim_indices.len() as u64],
    );
    append_k_vec(
        &mut tr,
        b"neo.fold.next/time_opening/reduction_group_value_coefficients",
        coefficients,
    );
    for (position, &claim_idx) in claim_indices.iter().enumerate() {
        let claim = &manifest.claims[claim_idx];
        tr.append_u64s(
            b"neo.fold.next/time_opening/reduction_group_value_claim_idx",
            &[claim_idx as u64, position as u64],
        );
        append_k_vec(
            &mut tr,
            b"neo.fold.next/time_opening/reduction_group_value_coeff",
            core::slice::from_ref(&coefficients[position]),
        );
        tr.append_message(
            b"neo.fold.next/time_opening/reduction_group_value_claim_digest",
            &claim.digest,
        );
    }
    tr.digest32()
}

pub(super) fn sample_group_coeffs(manifest: &OpeningManifest, group_digest: &[u8; 32], count: usize) -> Vec<K> {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction_group_coeff");
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_group_coeff_manifest",
        &manifest.digest,
    );
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_group_coeff_group_digest",
        group_digest,
    );
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_group_coeff_count",
        &[count as u64],
    );
    (0..count)
        .map(|position| {
            tr.append_u64s(
                b"neo.fold.next/time_opening/reduction_group_coeff_position",
                &[position as u64],
            );
            let real = tr.challenge_field(b"neo.fold.next/time_opening/reduction_group_coeff/re");
            let imag = tr.challenge_field(b"neo.fold.next/time_opening/reduction_group_coeff/im");
            from_complex(real, imag)
        })
        .collect()
}

pub(super) fn digest_unified_reduction(
    groups: &[OpeningReductionGroup],
    can_unify: bool,
    unified_domain: OpeningDomain,
    unified_point: &[K],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction_unified");
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_unified_len",
        &[groups.len() as u64],
    );
    for group in groups {
        tr.append_u64s(
            b"neo.fold.next/time_opening/reduction_unified_meta",
            &[
                opening_domain_tag(group.domain),
                group.sources.len() as u64,
                group.point.len() as u64,
                group.coefficients.len() as u64,
            ],
        );
        let source_tags: Vec<u64> = group
            .sources
            .iter()
            .map(|&source| opening_source_tag(source))
            .collect();
        tr.append_u64s(b"neo.fold.next/time_opening/reduction_unified_sources", &source_tags);
        append_point(
            &mut tr,
            b"neo.fold.next/time_opening/reduction_unified_point",
            &group.point,
        );
        append_k_vec(
            &mut tr,
            b"neo.fold.next/time_opening/reduction_unified_coefficients",
            &group.coefficients,
        );
        tr.append_message(
            b"neo.fold.next/time_opening/reduction_unified_group_digest",
            &group.group_digest,
        );
        tr.append_message(
            b"neo.fold.next/time_opening/reduction_unified_reduced_digest",
            &group.reduced_digest,
        );
    }
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_unified_anchor_meta",
        &[
            can_unify as u64,
            opening_domain_tag(unified_domain),
            unified_point.len() as u64,
        ],
    );
    append_point(
        &mut tr,
        b"neo.fold.next/time_opening/reduction_unified_anchor_point",
        unified_point,
    );
    tr.digest32()
}

pub(super) fn opening_source_tag(source: OpeningSource) -> u64 {
    match source {
        OpeningSource::MainLane => 1,
        OpeningSource::ExtensionKernel => 2,
        OpeningSource::ExtensionRoot => 3,
        OpeningSource::Rv32imKernel => 4,
    }
}

pub(super) fn opening_domain_tag(domain: OpeningDomain) -> u64 {
    match domain {
        OpeningDomain::Cpu => 1,
        OpeningDomain::Mem => 2,
    }
}

pub(super) fn append_point(tr: &mut Poseidon2Transcript, label: &'static [u8], point: &[K]) {
    tr.append_u64s(b"neo.fold.next/time_opening/point_len", &[point.len() as u64]);
    let coeffs_per_elem = point.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
    tr.append_fields_iter(
        label,
        point.len().saturating_mul(coeffs_per_elem),
        point.iter().flat_map(|v| v.as_coeffs()),
    );
}

pub(super) fn append_k_vec(tr: &mut Poseidon2Transcript, label: &'static [u8], values: &[K]) {
    tr.append_u64s(b"neo.fold.next/time_opening/k_vec_len", &[values.len() as u64]);
    let coeffs_per_elem = values.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
    tr.append_fields_iter(
        label,
        values.len().saturating_mul(coeffs_per_elem),
        values.iter().flat_map(|v| v.as_coeffs()),
    );
}

fn point_digest(point: &[K]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/point_digest");
    append_point(&mut tr, b"neo.fold.next/time_opening/point_digest_point", point);
    tr.digest32()
}

pub(super) fn canonical_claim_cmp(left: &OpeningClaim, right: &OpeningClaim) -> core::cmp::Ordering {
    (
        opening_domain_tag(left.domain),
        point_digest(&left.point),
        opening_source_tag(left.source),
        left.ordinal,
        &left.column_ids,
        &left.digest,
    )
        .cmp(&(
            opening_domain_tag(right.domain),
            point_digest(&right.point),
            opening_source_tag(right.source),
            right.ordinal,
            &right.column_ids,
            &right.digest,
        ))
}
