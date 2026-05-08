//! Owns the final opening manifest, reduction, and unification proof for the active pipeline.

use neo_ajtai::Commitment;
use neo_math::{from_complex, KExtensions, F, K};
use neo_reductions::error::PiCcsError;
use neo_reductions::sumcheck::{run_sumcheck_prover, verify_sumcheck_rounds, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::opening::{
    OpeningClaim, OpeningDomain, OpeningSource, TimeOpeningCompactProof, TimeOpeningGroupSummary,
    TimeOpeningProofSummary, TimeOpeningUnificationProof,
};
use crate::proof::RunProof;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpeningManifest {
    pub claims: Vec<OpeningClaim>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpeningReduction {
    pub groups: Vec<OpeningReductionGroup>,
    pub can_unify: bool,
    pub unified_domain: OpeningDomain,
    pub unified_point: Vec<K>,
    pub unified_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpeningReductionGroup {
    pub sources: Vec<OpeningSource>,
    pub domain: OpeningDomain,
    pub point: Vec<K>,
    pub claim_indices: Vec<usize>,
    pub coefficients: Vec<K>,
    pub group_digest: [u8; 32],
    pub reduced_digest: [u8; 32],
}

pub fn main_lane_opening_claims(session: &RunProof) -> Result<Vec<OpeningClaim>, PiCcsError> {
    let mut claims = Vec::with_capacity(session.chunks.len() + 1);
    for (chunk_idx, chunk) in session.chunks.iter().enumerate() {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/main_lane_chunk");
        tr.append_u64s(
            b"neo.fold.next/time_opening/chunk_meta",
            &[
                chunk_idx as u64,
                chunk.chunk.start_index as u64,
                chunk.chunk.steps.len() as u64,
            ],
        );
        for step in &chunk.chunk.steps {
            tr.append_message(b"neo.fold.next/time_opening/chunk_step_label", step.label.as_bytes());
        }
        tr.append_u64s(
            b"neo.fold.next/time_opening/chunk_fold_meta",
            &[chunk.ccs_outputs.len() as u64, chunk.dec.children.len() as u64],
        );
        let point = chunk
            .ccs_outputs
            .first()
            .map(|claim| claim.r.clone())
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing main-lane CE output for chunk {chunk_idx}")))?;
        claims.push(OpeningClaim {
            source: OpeningSource::MainLane,
            domain: OpeningDomain::Cpu,
            point,
            ordinal: chunk_idx as u64,
            column_ids: vec![0],
            digest: tr.digest32(),
        });
    }

    let mut footer = Poseidon2Transcript::new(b"neo.fold.next/time_opening/main_lane_footer");
    footer.append_u64s(
        b"neo.fold.next/time_opening/footer_meta",
        &[session.final_main_claims.len() as u64],
    );
    for claim in &session.final_main_claims {
        absorb_ce_footer(&mut footer, claim);
    }
    let footer_point = session
        .final_main_claims
        .first()
        .map(|claim| claim.r.clone())
        .ok_or_else(|| PiCcsError::ProtocolError("missing final main claims for time opening".into()))?;
    claims.push(OpeningClaim {
        source: OpeningSource::MainLane,
        domain: OpeningDomain::Cpu,
        point: footer_point,
        ordinal: 0,
        column_ids: vec![1],
        digest: footer.digest32(),
    });
    Ok(claims)
}

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

fn digest_opening_proof(reduction: &OpeningReduction, unification: &TimeOpeningUnificationProof) -> [u8; 32] {
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

fn build_manifest(
    main_lane_claims: &[OpeningClaim],
    extension_claims: &[OpeningClaim],
) -> Result<OpeningManifest, PiCcsError> {
    let mut claims = Vec::with_capacity(main_lane_claims.len() + extension_claims.len());
    claims.extend_from_slice(main_lane_claims);
    claims.extend_from_slice(extension_claims);
    claims.sort_by(canonical_claim_cmp);

    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/manifest");
    tr.append_u64s(b"neo.fold.next/time_opening/manifest_len", &[claims.len() as u64]);
    for claim in &claims {
        if claim.column_ids.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "time-opening manifest contains claim with empty column_ids".into(),
            ));
        }
        if !claim.column_ids.windows(2).all(|w| w[0] < w[1]) {
            return Err(PiCcsError::ProtocolError(
                "time-opening manifest contains claim with unsorted or duplicate column_ids".into(),
            ));
        }
        tr.append_u64s(
            b"neo.fold.next/time_opening/manifest_meta",
            &[
                opening_source_tag(claim.source),
                opening_domain_tag(claim.domain),
                claim.ordinal,
                claim.point.len() as u64,
                claim.column_ids.len() as u64,
            ],
        );
        append_point(&mut tr, b"neo.fold.next/time_opening/manifest_point", &claim.point);
        let column_ids_u64: Vec<u64> = claim.column_ids.iter().map(|&id| id as u64).collect();
        tr.append_u64s(b"neo.fold.next/time_opening/manifest_column_ids", &column_ids_u64);
        tr.append_message(b"neo.fold.next/time_opening/manifest_digest", &claim.digest);
    }

    for pair in claims.windows(2) {
        if pair[0] == pair[1] {
            return Err(PiCcsError::ProtocolError(
                "time-opening manifest contains duplicate claims".into(),
            ));
        }
    }

    Ok(OpeningManifest {
        claims,
        digest: tr.digest32(),
    })
}

fn build_reduction(manifest: &OpeningManifest) -> OpeningReduction {
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

fn build_reduction_groups(manifest: &OpeningManifest) -> Vec<OpeningReductionGroup> {
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

fn digest_reduction_group(
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

fn digest_reduced_group(
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

fn sample_group_coeffs(manifest: &OpeningManifest, group_digest: &[u8; 32], count: usize) -> Vec<K> {
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

fn digest_unified_reduction(
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

#[inline]
fn ceil_log2_at_least_1(n: usize) -> usize {
    let need = n.max(1).next_power_of_two();
    (need.trailing_zeros() as usize).max(1)
}

fn group_value(group: &OpeningReductionGroup) -> K {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction_unify_group_value");
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_unify_group_value_meta",
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
        b"neo.fold.next/time_opening/reduction_unify_group_value_sources",
        &source_tags,
    );
    append_point(
        &mut tr,
        b"neo.fold.next/time_opening/reduction_unify_group_value_point",
        &group.point,
    );
    let claim_indices_u64: Vec<u64> = group.claim_indices.iter().map(|&idx| idx as u64).collect();
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_unify_group_value_indices",
        &claim_indices_u64,
    );
    append_k_vec(
        &mut tr,
        b"neo.fold.next/time_opening/reduction_unify_group_value_coefficients",
        &group.coefficients,
    );
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_unify_group_value_group_digest",
        &group.group_digest,
    );
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_unify_group_value_reduced_digest",
        &group.reduced_digest,
    );
    let re = tr.challenge_field(b"neo.fold.next/time_opening/reduction_unify_group_value/re");
    let im = tr.challenge_field(b"neo.fold.next/time_opening/reduction_unify_group_value/im");
    from_complex(re, im)
}

fn bind_opening_unification_statement(
    tr: &mut Poseidon2Transcript,
    reduction: &OpeningReduction,
    ell_sel: usize,
    values: &[K],
) {
    tr.append_message(b"neo.fold.next/time_opening/reduction_unify_bind", &[]);
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_unify_bind_reduction_digest",
        &reduction.digest,
    );
    tr.append_u64s(
        b"neo.fold.next/time_opening/reduction_unify_bind_meta",
        &[
            reduction.groups.len() as u64,
            ell_sel as u64,
            reduction.can_unify as u64,
            opening_domain_tag(reduction.unified_domain),
            reduction.unified_point.len() as u64,
        ],
    );
    append_point(
        tr,
        b"neo.fold.next/time_opening/reduction_unify_bind_unified_point",
        &reduction.unified_point,
    );
    tr.append_message(
        b"neo.fold.next/time_opening/reduction_unify_bind_unified_digest",
        &reduction.unified_digest,
    );
    for (group_idx, (group, value)) in reduction.groups.iter().zip(values.iter()).enumerate() {
        tr.append_u64s(
            b"neo.fold.next/time_opening/reduction_unify_bind_group_meta",
            &[
                group_idx as u64,
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
            b"neo.fold.next/time_opening/reduction_unify_bind_group_sources",
            &source_tags,
        );
        append_point(
            tr,
            b"neo.fold.next/time_opening/reduction_unify_bind_group_point",
            &group.point,
        );
        let claim_indices_u64: Vec<u64> = group.claim_indices.iter().map(|&idx| idx as u64).collect();
        tr.append_u64s(
            b"neo.fold.next/time_opening/reduction_unify_bind_group_indices",
            &claim_indices_u64,
        );
        append_k_vec(
            tr,
            b"neo.fold.next/time_opening/reduction_unify_bind_group_coefficients",
            &group.coefficients,
        );
        tr.append_message(
            b"neo.fold.next/time_opening/reduction_unify_bind_group_digest",
            &group.group_digest,
        );
        tr.append_message(
            b"neo.fold.next/time_opening/reduction_unify_bind_reduced_digest",
            &group.reduced_digest,
        );
        tr.append_fields(
            b"neo.fold.next/time_opening/reduction_unify_bind_group_value",
            &value.as_coeffs(),
        );
    }
}

struct GroupSelectorOracle {
    values: Vec<K>,
    ell_sel: usize,
    prefix: Vec<K>,
}

impl GroupSelectorOracle {
    fn new(values: Vec<K>, ell_sel: usize) -> Self {
        Self {
            values,
            ell_sel,
            prefix: Vec::with_capacity(ell_sel),
        }
    }

    #[inline]
    fn bit_at(index: usize, bit: usize) -> bool {
        ((index >> bit) & 1usize) == 1
    }

    #[inline]
    fn bit_weight(bit: bool, x: K) -> K {
        if bit {
            x
        } else {
            K::ONE - x
        }
    }

    fn eval_at_point(values: &[K], r: &[K]) -> K {
        let mut acc = K::ZERO;
        for (group_idx, value) in values.iter().enumerate() {
            let mut weight = K::ONE;
            for (bit_idx, &rv) in r.iter().enumerate() {
                weight *= Self::bit_weight(Self::bit_at(group_idx, bit_idx), rv);
            }
            acc += weight * *value;
        }
        acc
    }
}

impl RoundOracle for GroupSelectorOracle {
    fn num_rounds(&self) -> usize {
        self.ell_sel.saturating_sub(self.prefix.len())
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.prefix.len() >= self.ell_sel {
            return vec![K::ZERO; points.len()];
        }
        let round_idx = self.prefix.len();
        let mut out = vec![K::ZERO; points.len()];
        for (group_idx, value) in self.values.iter().enumerate() {
            let mut prefix_weight = K::ONE;
            for (bit_idx, &bound) in self.prefix.iter().enumerate() {
                prefix_weight *= Self::bit_weight(Self::bit_at(group_idx, bit_idx), bound);
            }
            for (slot, &x) in out.iter_mut().zip(points.iter()) {
                *slot += prefix_weight * Self::bit_weight(Self::bit_at(group_idx, round_idx), x) * *value;
            }
        }
        out
    }

    fn fold(&mut self, r: K) {
        self.prefix.push(r);
    }
}

fn prove_opening_unification(reduction: &OpeningReduction) -> Result<TimeOpeningUnificationProof, PiCcsError> {
    if reduction.groups.is_empty() {
        return Ok(TimeOpeningUnificationProof::default());
    }
    let values: Vec<K> = reduction.groups.iter().map(group_value).collect();
    let ell_sel = ceil_log2_at_least_1(reduction.groups.len());
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction_unify");
    bind_opening_unification_statement(&mut tr, reduction, ell_sel, &values);
    let claimed_sum = values
        .iter()
        .copied()
        .fold(K::ZERO, |acc, value| acc + value);
    let mut oracle = GroupSelectorOracle::new(values, ell_sel);
    let (round_polys, r_unify) = run_sumcheck_prover(&mut tr, &mut oracle, claimed_sum)
        .map_err(|err| PiCcsError::ProtocolError(format!("time-opening unification prove failed: {err}")))?;
    Ok(TimeOpeningUnificationProof {
        claimed_sum,
        round_polys,
        r_unify,
    })
}

fn verify_opening_unification(
    reduction: &OpeningReduction,
    proof: &TimeOpeningUnificationProof,
) -> Result<(), PiCcsError> {
    if reduction.groups.is_empty() {
        if proof.claimed_sum == K::ZERO && proof.round_polys.is_empty() && proof.r_unify.is_empty() {
            return Ok(());
        }
        return Err(PiCcsError::ProtocolError(
            "time-opening unification proof must be empty when there are no groups".into(),
        ));
    }
    let values: Vec<K> = reduction.groups.iter().map(group_value).collect();
    let ell_sel = ceil_log2_at_least_1(reduction.groups.len());
    if proof.round_polys.len() != ell_sel {
        return Err(PiCcsError::ProtocolError(format!(
            "time-opening unification round count {} != expected {ell_sel}",
            proof.round_polys.len()
        )));
    }
    let expected_sum = values
        .iter()
        .copied()
        .fold(K::ZERO, |acc, value| acc + value);
    if proof.claimed_sum != expected_sum {
        return Err(PiCcsError::ProtocolError(
            "time-opening unification claimed_sum mismatch".into(),
        ));
    }
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/reduction_unify");
    bind_opening_unification_statement(&mut tr, reduction, ell_sel, &values);
    let (r_unify, final_value, ok) = verify_sumcheck_rounds(&mut tr, 1, proof.claimed_sum, &proof.round_polys);
    if !ok {
        return Err(PiCcsError::ProtocolError(
            "time-opening unification sumcheck verification failed".into(),
        ));
    }
    if proof.r_unify != r_unify {
        return Err(PiCcsError::ProtocolError(
            "time-opening unification selector point mismatch".into(),
        ));
    }
    let expected_final = GroupSelectorOracle::eval_at_point(&values, &proof.r_unify);
    if final_value != expected_final {
        return Err(PiCcsError::ProtocolError(
            "time-opening unification final value mismatch".into(),
        ));
    }
    Ok(())
}

fn opening_source_tag(source: OpeningSource) -> u64 {
    match source {
        OpeningSource::MainLane => 1,
        OpeningSource::ExtensionKernel => 2,
        OpeningSource::ExtensionRoot => 3,
        OpeningSource::Rv32imKernel => 4,
    }
}

fn opening_domain_tag(domain: OpeningDomain) -> u64 {
    match domain {
        OpeningDomain::Cpu => 1,
        OpeningDomain::Mem => 2,
    }
}

fn append_point(tr: &mut Poseidon2Transcript, label: &'static [u8], point: &[K]) {
    tr.append_u64s(b"neo.fold.next/time_opening/point_len", &[point.len() as u64]);
    let coeffs_per_elem = point.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
    tr.append_fields_iter(
        label,
        point.len().saturating_mul(coeffs_per_elem),
        point.iter().flat_map(|v| v.as_coeffs()),
    );
}

fn append_k_vec(tr: &mut Poseidon2Transcript, label: &'static [u8], values: &[K]) {
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

pub(crate) fn canonical_claim_cmp(left: &OpeningClaim, right: &OpeningClaim) -> core::cmp::Ordering {
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

fn absorb_ce_footer(tr: &mut Poseidon2Transcript, claim: &neo_ccs::CeClaim<Commitment, F, K>) {
    tr.append_u64s(
        b"neo.fold.next/time_opening/footer_claim_meta",
        &[claim.m_in as u64, claim.u_offset as u64, claim.u_len as u64],
    );
    tr.append_message(b"neo.fold.next/time_opening/footer_fold_digest", &claim.fold_digest);
}
