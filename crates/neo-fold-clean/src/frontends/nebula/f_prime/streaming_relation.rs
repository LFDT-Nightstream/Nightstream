//! Exact production schedule adapter for the shared-row phase composer.
//!
//! Owns only the transfer of the verifier-owned work-item maps into the generic
//! scheduled composers. It does not own component circuits or public fields.

mod lifecycle_kernel;

use std::ops::Range;

use neo_math::D;
use thiserror::Error;

use crate::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment,
    build_scheduled_grouped_phase_low_norm_r1cs_with_field_links,
    build_scheduled_linked_overlay_low_norm_r1cs_with_phase_field_links,
    prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix, LinkedOverlayError,
    LowNormR1csError, MultiBranchLowNormR1cs, OverlayKindLinks,
    ScheduledCommonPhaseFieldLink, ScheduledCursorBits, ScheduledGroupedPhaseError,
    ScheduledGroupedPhaseLowNormR1cs, ScheduledLinkedOverlayLowNormR1cs,
    ScheduledPhaseKindLinks, SparseR1cs,
};

use super::streaming_claim_replay::{
    production_claim_coordinate_overlay_kind_count,
    production_claim_coordinate_overlay_kind_map, production_claim_coordinate_overlay_links,
    production_claim_coordinate_overlay_sparse_arms, NebulaFPrimeClaimReplayError,
};
use super::streaming_lifecycle_relation::{
    NebulaFPrimeStreamingLifecycleArm, NebulaFPrimeStreamingLifecycleSourceArms,
};
use super::streaming_phase_envelope::{
    STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY,
    STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY,
};
use super::streaming_pi_rlc_family_relation::{
    production_pi_rlc_family_overlay_kind_map, production_pi_rlc_family_overlay_links,
    production_pi_rlc_family_overlay_sparse_arms, NebulaFPrimePiRlcFamilyRelationError,
    PI_RLC_FAMILY_COUNT,
};
use super::streaming_prior_state_replay_relation::STREAMING_PRIOR_STATE_REPLAY_FINAL_TARGET_FAMILY;
use super::streaming_program::{
    NebulaFPrimeStreamingCircuitKind, NebulaFPrimeStreamingProgramAudit,
};
use super::streaming_public::NebulaFPrimeStreamingPublicLayout;

const STREAMING_JOINT_ROW_BOUND: usize = 1 << 24;
const STREAMING_MAX_ASSIGNMENT_FIELDS: usize = 16_777_206;

#[derive(Debug, Error)]
pub enum NebulaFPrimeStreamingRelationError {
    #[error(transparent)]
    ClaimReplay(#[from] NebulaFPrimeClaimReplayError),
    #[error(transparent)]
    PiRlcFamily(#[from] NebulaFPrimePiRlcFamilyRelationError),
    #[error(transparent)]
    LowNorm(#[from] LowNormR1csError),
    #[error(transparent)]
    Scheduled(#[from] ScheduledGroupedPhaseError),
    #[error(transparent)]
    LinkedOverlay(#[from] LinkedOverlayError),
    #[error("streaming F-prime phase-envelope link profile: {0}")]
    PhaseEnvelope(String),
    #[error(
        "streaming F-prime {relation} relation exceeds the production joint domain: rows {rows} (bound {row_bound}), assignment fields {assignment_fields} (bound {assignment_bound})"
    )]
    JointDomain {
        relation: &'static str,
        rows: usize,
        assignment_fields: usize,
        row_bound: usize,
        assignment_bound: usize,
    },
}

/// Checked source-field links for all production phase kinds.
///
/// Construction scans exact field-R1CS source metadata. Callers cannot create
/// a partial profile and pass it to a production schedule constructor.
#[derive(Clone, Debug)]
pub struct NebulaFPrimeStreamingPhaseEnvelopeLinkProfile {
    links: Vec<ScheduledPhaseKindLinks>,
    fields_per_kind: usize,
}

impl NebulaFPrimeStreamingPhaseEnvelopeLinkProfile {
    pub fn phase_kind_count(&self) -> usize {
        self.links.len()
    }

    pub const fn fields_per_kind(&self) -> usize {
        self.fields_per_kind
    }

    pub fn total_links(&self) -> usize {
        self.links
            .iter()
            .map(|contract| contract.fields.len())
            .sum()
    }

    /// Compile the exact two-arm compact lifecycle common relation and derive
    /// all common-to-phase source-field links from the same Rust sources.
    ///
    /// The returned common relation owns XOut recomputation, cursor succession,
    /// and the semantic envelope. It does not replace the phase circuits or the
    /// monolithic lifecycle correspondence audit.
    pub fn compile_compact_lifecycle(
        phase_sources: &[SparseR1cs],
    ) -> Result<(MultiBranchLowNormR1cs, Self), NebulaFPrimeStreamingRelationError> {
        let lifecycle = lifecycle_kernel::production_streaming_lifecycle_kernel_source_arms()
            .map_err(|error| NebulaFPrimeStreamingRelationError::PhaseEnvelope(error.to_string()))?;
        let common = build_multi_branch_selective_low_norm_r1cs_with_alignment(lifecycle.arms(), 0, D, 0)?;
        validate_production_joint_domain(
            "compact lifecycle common",
            common.structure().n,
            common.structure().m,
        )?;

        let common_sources = [
            lifecycle.arm(NebulaFPrimeStreamingLifecycleArm::Base),
            lifecycle.arm(NebulaFPrimeStreamingLifecycleArm::Recursive),
        ];
        let common_ranges = [
            kernel_envelope_ranges(lifecycle.phase_envelope_fields(NebulaFPrimeStreamingLifecycleArm::Base)),
            kernel_envelope_ranges(lifecycle.phase_envelope_fields(NebulaFPrimeStreamingLifecycleArm::Recursive)),
        ];
        let profile = production_phase_envelope_link_profile_from_sources(
            common_sources,
            common_ranges,
            lifecycle.recursive_prior_state_digest_columns(),
            phase_sources,
        )?;
        Ok((common, profile))
    }

    fn into_links(self) -> Vec<ScheduledPhaseKindLinks> {
        self.links
    }
}

#[derive(Clone, Debug)]
struct PhaseEnvelopeRanges {
    before_local_state_digest: Range<usize>,
    before_delayed_payload: Range<usize>,
    after_local_state_digest: Range<usize>,
    after_delayed_payload: Range<usize>,
}

/// Derive the complete private source-field link profile from the exact Rust
/// lifecycle and phase source artifacts. Phase arms use circuit-kind order.
pub fn production_phase_envelope_link_profile(
    lifecycle: &NebulaFPrimeStreamingLifecycleSourceArms,
    phase_sources: &[SparseR1cs],
) -> Result<NebulaFPrimeStreamingPhaseEnvelopeLinkProfile, NebulaFPrimeStreamingRelationError> {
    let common_sources = [
        lifecycle.arm(NebulaFPrimeStreamingLifecycleArm::Base),
        lifecycle.arm(NebulaFPrimeStreamingLifecycleArm::Recursive),
    ];
    let common_ranges = [
        lifecycle_envelope_ranges(
            lifecycle.phase_envelope_fields(NebulaFPrimeStreamingLifecycleArm::Base),
        ),
        lifecycle_envelope_ranges(
            lifecycle.phase_envelope_fields(NebulaFPrimeStreamingLifecycleArm::Recursive),
        ),
    ];
    production_phase_envelope_link_profile_from_sources(
        common_sources,
        common_ranges,
        lifecycle.recursive_prior_state_digest_columns(),
        phase_sources,
    )
}

fn lifecycle_envelope_ranges(
    fields: &super::streaming_lifecycle_relation::NebulaFPrimeStreamingPhaseEnvelopeFields,
) -> PhaseEnvelopeRanges {
    PhaseEnvelopeRanges {
        before_local_state_digest: fields.before_local_state_digest(),
        before_delayed_payload: fields.before_delayed_payload(),
        after_local_state_digest: fields.after_local_state_digest(),
        after_delayed_payload: fields.after_delayed_payload(),
    }
}

fn kernel_envelope_ranges(
    fields: &lifecycle_kernel::NebulaFPrimeStreamingLifecycleKernelEnvelopeFields,
) -> PhaseEnvelopeRanges {
    PhaseEnvelopeRanges {
        before_local_state_digest: fields.before_local_state_digest(),
        before_delayed_payload: fields.before_delayed_payload(),
        after_local_state_digest: fields.after_local_state_digest(),
        after_delayed_payload: fields.after_delayed_payload(),
    }
}

fn production_phase_envelope_link_profile_from_sources(
    common_sources: [&SparseR1cs; 2],
    common_ranges: [PhaseEnvelopeRanges; 2],
    recursive_prior_state_digest_columns: [usize; 4],
    phase_sources: &[SparseR1cs],
) -> Result<NebulaFPrimeStreamingPhaseEnvelopeLinkProfile, NebulaFPrimeStreamingRelationError> {
    let program = NebulaFPrimeStreamingProgramAudit::production();
    if phase_sources.len() != program.circuit_kind_count() {
        return Err(NebulaFPrimeStreamingRelationError::PhaseEnvelope(format!(
            "phase source count {} != {}",
            phase_sources.len(),
            program.circuit_kind_count()
        )));
    }

    let delayed_payload_fields = common_ranges[NebulaFPrimeStreamingLifecycleArm::Recursive.index()]
        .before_delayed_payload
        .len();
    for (source, ranges) in common_sources.iter().zip(&common_ranges) {
        validate_envelope_ranges("lifecycle", source, ranges, delayed_payload_fields)?;
    }

    let mut lifecycle_group_by_kind = vec![None; program.circuit_kind_count()];
    for (&group, &kind) in program
        .lifecycle_group_map()
        .iter()
        .zip(program.circuit_kind_map().iter())
    {
        match lifecycle_group_by_kind[kind] {
            None => lifecycle_group_by_kind[kind] = Some(group),
            Some(expected) if expected == group => {}
            Some(expected) => {
                return Err(NebulaFPrimeStreamingRelationError::PhaseEnvelope(format!(
                    "phase kind {kind} occurs in lifecycle groups {expected} and {group}"
                )));
            }
        }
    }

    let mut links = Vec::with_capacity(phase_sources.len());
    let fields_per_kind = 2 * 4 + 2 * delayed_payload_fields;
    for (kind, phase_source) in phase_sources.iter().enumerate() {
        let lifecycle_group = lifecycle_group_by_kind[kind].ok_or_else(|| {
            NebulaFPrimeStreamingRelationError::PhaseEnvelope(format!(
                "phase kind {kind} is absent from the production schedule"
            ))
        })?;
        let phase_ranges = exact_phase_envelope_ranges(phase_source, delayed_payload_fields)?;
        let common = &common_ranges[lifecycle_group];
        let is_final_prior_state_replay =
            kind == NebulaFPrimeStreamingCircuitKind::PriorStateReplayFinal.code() as usize;
        let mut fields = Vec::with_capacity(fields_per_kind + usize::from(is_final_prior_state_replay) * 4);
        append_range_links(
            &mut fields,
            &common.before_local_state_digest,
            &phase_ranges.before_local_state_digest,
        );
        append_range_links(
            &mut fields,
            &common.before_delayed_payload,
            &phase_ranges.before_delayed_payload,
        );
        append_range_links(
            &mut fields,
            &common.after_local_state_digest,
            &phase_ranges.after_local_state_digest,
        );
        append_range_links(
            &mut fields,
            &common.after_delayed_payload,
            &phase_ranges.after_delayed_payload,
        );
        if is_final_prior_state_replay {
            if lifecycle_group != NebulaFPrimeStreamingLifecycleArm::Recursive.index() {
                return Err(NebulaFPrimeStreamingRelationError::PhaseEnvelope(
                    "final prior-state replay is not in the recursive lifecycle group".into(),
                ));
            }
            let target = exact_private_family(phase_source, STREAMING_PRIOR_STATE_REPLAY_FINAL_TARGET_FAMILY, 4)?;
            fields.extend(
                recursive_prior_state_digest_columns
                    .into_iter()
                    .zip(target)
                    .map(|(common_field, phase_field)| ScheduledCommonPhaseFieldLink {
                        common_field,
                        phase_field,
                    }),
            );
        }
        debug_assert_eq!(
            fields.len(),
            fields_per_kind + usize::from(is_final_prior_state_replay) * 4
        );
        links.push(ScheduledPhaseKindLinks {
            lifecycle_group,
            phase_kind: kind,
            fields,
        });
    }

    Ok(NebulaFPrimeStreamingPhaseEnvelopeLinkProfile {
        links,
        fields_per_kind,
    })
}

fn exact_phase_envelope_ranges(
    source: &SparseR1cs,
    delayed_payload_fields: usize,
) -> Result<PhaseEnvelopeRanges, NebulaFPrimeStreamingRelationError> {
    let ranges = PhaseEnvelopeRanges {
        before_local_state_digest: exact_private_family(source, STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY, 4)?,
        before_delayed_payload: exact_private_family(
            source,
            STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY,
            delayed_payload_fields,
        )?,
        after_local_state_digest: exact_private_family(source, STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY, 4)?,
        after_delayed_payload: exact_private_family(
            source,
            STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY,
            delayed_payload_fields,
        )?,
    };
    validate_envelope_ranges("phase", source, &ranges, delayed_payload_fields)?;
    Ok(ranges)
}

fn exact_private_family(
    source: &SparseR1cs,
    family_name: &'static str,
    expected_len: usize,
) -> Result<Range<usize>, NebulaFPrimeStreamingRelationError> {
    let mut matches = source
        .column_family_ranges()
        .iter()
        .filter(|family| family.name == family_name);
    let family = matches.next().ok_or_else(|| {
        NebulaFPrimeStreamingRelationError::PhaseEnvelope(format!("source is missing {family_name}"))
    })?;
    if matches.next().is_some() {
        return Err(NebulaFPrimeStreamingRelationError::PhaseEnvelope(format!(
            "source contains duplicate {family_name} ranges"
        )));
    }
    let range = family.column_start..family.column_end;
    validate_private_range("phase", source, family_name, &range, expected_len)?;
    Ok(range)
}

fn validate_envelope_ranges(
    owner: &'static str,
    source: &SparseR1cs,
    ranges: &PhaseEnvelopeRanges,
    delayed_payload_fields: usize,
) -> Result<(), NebulaFPrimeStreamingRelationError> {
    for (name, range, expected_len) in [
        ("before local-state digest", &ranges.before_local_state_digest, 4),
        (
            "before delayed payload",
            &ranges.before_delayed_payload,
            delayed_payload_fields,
        ),
        ("after local-state digest", &ranges.after_local_state_digest, 4),
        (
            "after delayed payload",
            &ranges.after_delayed_payload,
            delayed_payload_fields,
        ),
    ] {
        validate_private_range(owner, source, name, range, expected_len)?;
    }
    Ok(())
}

fn validate_private_range(
    owner: &'static str,
    source: &SparseR1cs,
    name: &str,
    range: &Range<usize>,
    expected_len: usize,
) -> Result<(), NebulaFPrimeStreamingRelationError> {
    if range.len() != expected_len {
        return Err(NebulaFPrimeStreamingRelationError::PhaseEnvelope(format!(
            "{owner} {name} width {} != {expected_len}",
            range.len()
        )));
    }
    if range.start < source.m_in || range.end > source.m {
        return Err(NebulaFPrimeStreamingRelationError::PhaseEnvelope(format!(
            "{owner} {name} range [{}, {}) is not private inside {}..{}",
            range.start, range.end, source.m_in, source.m
        )));
    }
    Ok(())
}

fn append_range_links(
    links: &mut Vec<ScheduledCommonPhaseFieldLink>,
    common: &Range<usize>,
    phase: &Range<usize>,
) {
    debug_assert_eq!(common.len(), phase.len());
    links.extend(
        common
            .clone()
            .zip(phase.clone())
            .map(|(common_field, phase_field)| ScheduledCommonPhaseFieldLink {
                common_field,
                phase_field,
            }),
    );
}

fn validate_production_joint_domain(
    relation: &'static str,
    rows: usize,
    assignment_fields: usize,
) -> Result<(), NebulaFPrimeStreamingRelationError> {
    if rows > STREAMING_JOINT_ROW_BOUND || assignment_fields > STREAMING_MAX_ASSIGNMENT_FIELDS {
        return Err(NebulaFPrimeStreamingRelationError::JointDomain {
            relation,
            rows,
            assignment_fields,
            row_bound: STREAMING_JOINT_ROW_BOUND,
            assignment_bound: STREAMING_MAX_ASSIGNMENT_FIELDS,
        });
    }
    Ok(())
}

pub fn build_production_streaming_schedule_low_norm_r1cs(
    common: MultiBranchLowNormR1cs,
    phase_kinds: MultiBranchLowNormR1cs,
    phase_envelope_links: NebulaFPrimeStreamingPhaseEnvelopeLinkProfile,
) -> Result<ScheduledGroupedPhaseLowNormR1cs, NebulaFPrimeStreamingRelationError> {
    let program = NebulaFPrimeStreamingProgramAudit::production();
    let public = NebulaFPrimeStreamingPublicLayout::production();
    let relation = build_scheduled_grouped_phase_low_norm_r1cs_with_field_links(
        common,
        phase_kinds,
        program.lifecycle_group_map(),
        program.circuit_kind_map(),
        ScheduledCursorBits::new(public.before_cursor_bits(), public.after_cursor_bits()),
        phase_envelope_links.into_links(),
    )?;
    validate_production_joint_domain(
        "recursive scheduled",
        relation.structure().n,
        relation.structure().m,
    )?;
    Ok(relation)
}

pub const fn production_combined_overlay_kind_count() -> usize {
    production_claim_coordinate_overlay_kind_count() + PI_RLC_FAMILY_COUNT
}

/// Exact combined overlay kind selected by each verifier-owned work item.
/// Claim and PiRLC family work items are disjoint by construction.
pub fn production_combined_overlay_kind_map() -> Vec<usize> {
    let claim = production_claim_coordinate_overlay_kind_map();
    let pi_rlc = production_pi_rlc_family_overlay_kind_map(0, production_claim_coordinate_overlay_kind_count());
    assert_eq!(claim.len(), pi_rlc.len());
    claim
        .into_iter()
        .zip(pi_rlc)
        .map(|(claim_kind, pi_rlc_kind)| match (claim_kind, pi_rlc_kind) {
            (claim_kind, 0) => claim_kind,
            (0, pi_rlc_kind) => pi_rlc_kind,
            _ => unreachable!("claim and PiRLC family overlays cannot select the same work item"),
        })
        .collect()
}

pub fn production_combined_overlay_links() -> Vec<OverlayKindLinks> {
    let mut links = production_claim_coordinate_overlay_links();
    links.extend(production_pi_rlc_family_overlay_links(
        production_claim_coordinate_overlay_kind_count(),
    ));
    links
}

pub fn build_production_combined_overlay_low_norm_r1cs(
) -> Result<MultiBranchLowNormR1cs, NebulaFPrimeStreamingRelationError> {
    let mut arms = production_claim_coordinate_overlay_sparse_arms()?;
    arms.extend(production_pi_rlc_family_overlay_sparse_arms()?);
    debug_assert_eq!(arms.len(), production_combined_overlay_kind_count());
    let overlay = prepare_owned_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        arms,
        0,
        0,
        1,
        0,
        crate::config::B_BASE,
    )?
    .finish()?;
    validate_production_joint_domain(
        "combined overlay",
        overlay.structure().n,
        overlay.structure().m,
    )?;
    Ok(overlay)
}

/// Add the exact claim-coordinate and PiRLC family overlays to the production
/// schedule. The caller owns the two lifecycle circuits and all 23 phase-kind
/// circuits. Phase kinds 3, 4, 10, and 11 must be the matching production
/// bodies used by the generated link contracts.
pub fn build_production_streaming_schedule_with_overlays_low_norm_r1cs(
    common: MultiBranchLowNormR1cs,
    phase_kinds: MultiBranchLowNormR1cs,
    phase_envelope_links: NebulaFPrimeStreamingPhaseEnvelopeLinkProfile,
) -> Result<ScheduledLinkedOverlayLowNormR1cs, NebulaFPrimeStreamingRelationError> {
    let program = NebulaFPrimeStreamingProgramAudit::production();
    let public = NebulaFPrimeStreamingPublicLayout::production();
    let overlay = build_production_combined_overlay_low_norm_r1cs()?;
    let relation = build_scheduled_linked_overlay_low_norm_r1cs_with_phase_field_links(
        common,
        phase_kinds,
        overlay,
        program.lifecycle_group_map(),
        program.circuit_kind_map(),
        production_combined_overlay_kind_map(),
        ScheduledCursorBits::new(public.before_cursor_bits(), public.after_cursor_bits()),
        phase_envelope_links.into_links(),
        production_combined_overlay_links(),
    )?;
    validate_production_joint_domain(
        "recursive scheduled plus overlays",
        relation.structure().n,
        relation.structure().m,
    )?;
    Ok(relation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_joint_domain_is_fail_closed() {
        validate_production_joint_domain(
            "boundary",
            STREAMING_JOINT_ROW_BOUND,
            STREAMING_MAX_ASSIGNMENT_FIELDS,
        )
        .unwrap();
        assert!(validate_production_joint_domain(
            "row overflow",
            STREAMING_JOINT_ROW_BOUND + 1,
            STREAMING_MAX_ASSIGNMENT_FIELDS,
        )
        .is_err());
        assert!(validate_production_joint_domain(
            "assignment overflow",
            STREAMING_JOINT_ROW_BOUND,
            STREAMING_MAX_ASSIGNMENT_FIELDS + 1,
        )
        .is_err());
    }
}
