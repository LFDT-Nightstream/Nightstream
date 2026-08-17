//! Exact terminal source-to-final placement for the streaming F-prime relation.
//!
//! Owns the checked production terminal arm, its recursive lifecycle and
//! `SemanticLinks` scope, and the final low-norm columns and link rows for the
//! trailing fresh witness's private after-state fields.
//!
//! Does not emit constraints. It does not make a digest authoritative and it
//! does not replace the terminal opening, delayed-Nebula finalizer, or final
//! closed-lane predicate.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use neo_math::F;
use thiserror::Error;

use super::streaming_lifecycle_relation::{
    NebulaFPrimeStreamingLifecycleArm, NebulaFPrimeStreamingLifecycleSourceArms,
};
use super::streaming_program::{
    NebulaFPrimeStreamingCircuitKind, NebulaFPrimeStreamingPhase, NebulaFPrimeStreamingProgramAudit,
};
use crate::frontends::r1cs_f_prime::{ScheduledLinkedOverlayLowNormR1cs, SelectiveEmittedRowFamily};
use crate::paper::f_prime::stage as fprime_stage;

/// Exact number of work items accepted by the frozen streaming terminal.
pub const STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS: usize = 436;

/// Stable identity of the exact terminal-slice profile schema.
pub const STREAMING_TERMINAL_PROFILE_ID: &str = "nightstream/goldilocks/streaming-terminal-slice/v1";

/// Diagnostic source-artifact format. Exact Rust rows remain authoritative.
pub const STREAMING_TERMINAL_SOURCE_ARTIFACT_ID: &str = "rust:nightstream/streaming-lifecycle-recursive/source-rows/v1";

/// Diagnostic final-artifact format. Exact selective CCS rows remain authoritative.
pub const STREAMING_TERMINAL_FINAL_ARTIFACT_ID: &str = "rust:nightstream/streaming-selective-ccs/final-rows/v1";

/// Source-field domain checked by the Rust source relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimeStreamingTerminalFieldDomain {
    Goldilocks,
    Boolean,
}

/// One final-column term in an exact affine source-field decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingTerminalDecoderTerm {
    final_column: usize,
    coefficient: F,
}

impl NebulaFPrimeStreamingTerminalDecoderTerm {
    pub const fn final_column(self) -> usize {
        self.final_column
    }

    pub const fn coefficient(self) -> F {
        self.coefficient
    }
}

/// One source field and its exact affine decoder in the final relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingTerminalFieldBinding {
    source_field: usize,
    decoder_terms: Vec<NebulaFPrimeStreamingTerminalDecoderTerm>,
}

impl NebulaFPrimeStreamingTerminalFieldBinding {
    pub const fn source_field(&self) -> usize {
        self.source_field
    }

    pub fn decoder_terms(&self) -> &[NebulaFPrimeStreamingTerminalDecoderTerm] {
        &self.decoder_terms
    }
}

/// One authoritative trailing-fresh source slice and its final link rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingTerminalSliceBinding {
    source_fields: Range<usize>,
    source_domain: NebulaFPrimeStreamingTerminalFieldDomain,
    fields: Vec<NebulaFPrimeStreamingTerminalFieldBinding>,
    final_common_phase_link_rows: Range<usize>,
}

/// Canonical 32-field post-step `x_out` preimage from the trailing fresh
/// witness. Slice accessors follow the exact Poseidon2 preimage order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingTerminalXOutBinding {
    source_fields: [usize; 32],
    fields: Vec<NebulaFPrimeStreamingTerminalFieldBinding>,
}

impl NebulaFPrimeStreamingTerminalXOutBinding {
    pub fn source_fields(&self) -> &[usize; 32] {
        &self.source_fields
    }

    pub fn fields(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields
    }

    pub fn domain_tag(&self) -> &NebulaFPrimeStreamingTerminalFieldBinding {
        &self.fields[0]
    }

    pub fn verifier_key_digest(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[1..5]
    }

    pub fn pi_ccs_header(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[5..9]
    }

    pub fn chunk_count_halves(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[9..11]
    }

    pub fn step_count_halves(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[11..13]
    }

    pub fn program_counter_halves(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[13..15]
    }

    pub fn boundary(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[15..19]
    }

    pub fn semantic_state_digest(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[19..23]
    }

    pub fn construction2_accumulator_digest(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[23..27]
    }

    pub fn nebula_presence_marker(&self) -> &NebulaFPrimeStreamingTerminalFieldBinding {
        &self.fields[27]
    }

    pub fn nebula_state_digest(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[28..32]
    }
}

/// Canonical 50-field post-step Nebula lane from the same fresh witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingTerminalLaneBinding {
    source_fields: Vec<usize>,
    fields: Vec<NebulaFPrimeStreamingTerminalFieldBinding>,
}

impl NebulaFPrimeStreamingTerminalLaneBinding {
    pub fn source_fields(&self) -> &[usize] {
        &self.source_fields
    }

    pub fn fields(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields
    }

    pub fn program_binding_digest(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[0..4]
    }

    pub fn open(&self) -> &NebulaFPrimeStreamingTerminalFieldBinding {
        &self.fields[4]
    }

    pub fn segment_index(&self) -> &NebulaFPrimeStreamingTerminalFieldBinding {
        &self.fields[5]
    }

    pub fn step_index(&self) -> &NebulaFPrimeStreamingTerminalFieldBinding {
        &self.fields[6]
    }

    pub fn timestamp(&self) -> &NebulaFPrimeStreamingTerminalFieldBinding {
        &self.fields[7]
    }

    pub fn gamma(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[8..12]
    }

    pub fn running_products(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[12..20]
    }

    pub fn stack_pointers(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[20..22]
    }

    pub fn pre_chains(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[22..34]
    }

    pub fn seen_chains(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[34..46]
    }

    pub fn memory_digest(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields[46..50]
    }
}

impl NebulaFPrimeStreamingTerminalSliceBinding {
    pub fn source_fields(&self) -> Range<usize> {
        self.source_fields.clone()
    }

    pub const fn source_domain(&self) -> NebulaFPrimeStreamingTerminalFieldDomain {
        self.source_domain
    }

    pub fn fields(&self) -> &[NebulaFPrimeStreamingTerminalFieldBinding] {
        &self.fields
    }

    pub fn final_common_phase_link_rows(&self) -> Range<usize> {
        self.final_common_phase_link_rows.clone()
    }
}

/// One emitted selective-CCS row run owned by the recursive semantic stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingTerminalFinalRowRun {
    family: SelectiveEmittedRowFamily,
    rows: Range<usize>,
}

/// Canonical final-column partition for the linked streaming relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingTerminalColumnLayout {
    public: Range<usize>,
    lifecycle_private: Range<usize>,
    phase_private: Range<usize>,
    schedule_selectors: Vec<usize>,
    scheduled_ring_padding: Range<usize>,
    overlay_private: Range<usize>,
    overlay_selectors: Vec<usize>,
    final_ring_padding: Range<usize>,
}

impl NebulaFPrimeStreamingTerminalColumnLayout {
    pub fn public(&self) -> Range<usize> {
        self.public.clone()
    }

    pub fn lifecycle_private(&self) -> Range<usize> {
        self.lifecycle_private.clone()
    }

    pub fn phase_private(&self) -> Range<usize> {
        self.phase_private.clone()
    }

    pub fn schedule_selectors(&self) -> &[usize] {
        &self.schedule_selectors
    }

    pub fn scheduled_ring_padding(&self) -> Range<usize> {
        self.scheduled_ring_padding.clone()
    }

    pub fn overlay_private(&self) -> Range<usize> {
        self.overlay_private.clone()
    }

    pub fn overlay_selectors(&self) -> &[usize] {
        &self.overlay_selectors
    }

    pub fn final_ring_padding(&self) -> Range<usize> {
        self.final_ring_padding.clone()
    }
}

/// Exact source-stage owner and final emitted rows for a set of terminal
/// source fields. Stage paths are diagnostic labels; the source and final rows
/// are the authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingTerminalSourceStageBinding {
    source_stage_occurrence: usize,
    source_stage_path: &'static str,
    source_rows: Range<usize>,
    source_fields: Vec<usize>,
    final_row_runs: Vec<NebulaFPrimeStreamingTerminalFinalRowRun>,
}

impl NebulaFPrimeStreamingTerminalSourceStageBinding {
    pub const fn source_stage_occurrence(&self) -> usize {
        self.source_stage_occurrence
    }

    pub const fn source_stage_path(&self) -> &'static str {
        self.source_stage_path
    }

    pub fn source_rows(&self) -> Range<usize> {
        self.source_rows.clone()
    }

    pub fn source_fields(&self) -> &[usize] {
        &self.source_fields
    }

    pub fn final_row_runs(&self) -> &[NebulaFPrimeStreamingTerminalFinalRowRun] {
        &self.final_row_runs
    }
}

impl NebulaFPrimeStreamingTerminalFinalRowRun {
    pub const fn family(&self) -> SelectiveEmittedRowFamily {
        self.family
    }

    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }
}

/// Checked placement facts needed by the exact terminal compiler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingTerminalProfile {
    accepted_work_items: usize,
    terminal_arm: usize,
    lifecycle_group: usize,
    phase_kind: usize,
    schedule_selector_column: usize,
    lifecycle_selector_column: usize,
    phase_selector_column: usize,
    source_rows: usize,
    source_columns: usize,
    source_public_columns: usize,
    final_rows: usize,
    final_columns: usize,
    final_public_columns: usize,
    column_layout: NebulaFPrimeStreamingTerminalColumnLayout,
    source_stage_occurrence: usize,
    source_stage_rows: Range<usize>,
    final_stage_row_runs: Vec<NebulaFPrimeStreamingTerminalFinalRowRun>,
    after_x_out: NebulaFPrimeStreamingTerminalXOutBinding,
    after_nebula_lane: NebulaFPrimeStreamingTerminalLaneBinding,
    after_local_state_digest: NebulaFPrimeStreamingTerminalSliceBinding,
    after_delayed_payload: NebulaFPrimeStreamingTerminalSliceBinding,
    source_stage_bindings: Vec<NebulaFPrimeStreamingTerminalSourceStageBinding>,
}

impl NebulaFPrimeStreamingTerminalProfile {
    pub const fn profile_id(&self) -> &'static str {
        STREAMING_TERMINAL_PROFILE_ID
    }

    pub const fn lifecycle_scope(&self) -> &'static str {
        "recursive-terminal-arm-435"
    }

    pub const fn source_artifact_identity(&self) -> &'static str {
        STREAMING_TERMINAL_SOURCE_ARTIFACT_ID
    }

    pub const fn final_artifact_identity(&self) -> &'static str {
        STREAMING_TERMINAL_FINAL_ARTIFACT_ID
    }

    pub const fn accepted_work_items(&self) -> usize {
        self.accepted_work_items
    }

    pub const fn terminal_arm(&self) -> usize {
        self.terminal_arm
    }

    pub const fn lifecycle_group(&self) -> usize {
        self.lifecycle_group
    }

    pub const fn phase_kind(&self) -> usize {
        self.phase_kind
    }

    pub const fn schedule_selector_column(&self) -> usize {
        self.schedule_selector_column
    }

    pub const fn lifecycle_selector_column(&self) -> usize {
        self.lifecycle_selector_column
    }

    pub const fn phase_selector_column(&self) -> usize {
        self.phase_selector_column
    }

    pub const fn source_rows(&self) -> usize {
        self.source_rows
    }

    pub const fn source_columns(&self) -> usize {
        self.source_columns
    }

    pub const fn source_public_columns(&self) -> usize {
        self.source_public_columns
    }

    pub const fn final_rows(&self) -> usize {
        self.final_rows
    }

    pub const fn final_columns(&self) -> usize {
        self.final_columns
    }

    pub const fn final_public_columns(&self) -> usize {
        self.final_public_columns
    }

    pub fn column_layout(&self) -> &NebulaFPrimeStreamingTerminalColumnLayout {
        &self.column_layout
    }

    pub const fn source_stage_occurrence(&self) -> usize {
        self.source_stage_occurrence
    }

    pub const fn source_stage_path(&self) -> &'static str {
        fprime_stage::RECURSIVE_SEMANTIC_LINKS
    }

    pub fn source_stage_rows(&self) -> Range<usize> {
        self.source_stage_rows.clone()
    }

    pub fn final_stage_row_runs(&self) -> &[NebulaFPrimeStreamingTerminalFinalRowRun] {
        &self.final_stage_row_runs
    }

    pub fn after_x_out(&self) -> &NebulaFPrimeStreamingTerminalXOutBinding {
        &self.after_x_out
    }

    pub fn after_nebula_lane(&self) -> &NebulaFPrimeStreamingTerminalLaneBinding {
        &self.after_nebula_lane
    }

    pub fn after_local_state_digest(&self) -> &NebulaFPrimeStreamingTerminalSliceBinding {
        &self.after_local_state_digest
    }

    pub fn after_delayed_payload(&self) -> &NebulaFPrimeStreamingTerminalSliceBinding {
        &self.after_delayed_payload
    }

    pub fn source_stage_bindings(&self) -> &[NebulaFPrimeStreamingTerminalSourceStageBinding] {
        &self.source_stage_bindings
    }
}

#[derive(Debug, Error)]
pub enum NebulaFPrimeStreamingTerminalProfileError {
    #[error("streaming F-prime terminal profile: {0}")]
    Invalid(String),
}

fn invalid(message: impl Into<String>) -> NebulaFPrimeStreamingTerminalProfileError {
    NebulaFPrimeStreamingTerminalProfileError::Invalid(message.into())
}

/// Derive the terminal profile from exact Rust source rows and their final
/// linked selective-CCS relation.
pub fn production_streaming_terminal_profile(
    lifecycle: &NebulaFPrimeStreamingLifecycleSourceArms,
    relation: &ScheduledLinkedOverlayLowNormR1cs,
) -> Result<NebulaFPrimeStreamingTerminalProfile, NebulaFPrimeStreamingTerminalProfileError> {
    let program = NebulaFPrimeStreamingProgramAudit::production();
    if program.work_items().len() != STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS {
        return Err(invalid(format!(
            "production program has {} work items, expected {STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS}",
            program.work_items().len()
        )));
    }
    let terminal_arm = STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS - 1;
    let terminal_item = program.work_items()[terminal_arm];
    if terminal_item.phase() != NebulaFPrimeStreamingPhase::SemanticLinks || terminal_item.index() != 0 {
        return Err(invalid("the final work item is not the sole SemanticLinks item"));
    }

    let expected_lifecycle_groups = program.lifecycle_group_map();
    let expected_phase_kinds = program.circuit_kind_map();
    let scheduled = relation.scheduled_relation();
    if scheduled.layout().lifecycle_groups() != expected_lifecycle_groups {
        return Err(invalid(
            "final relation lifecycle-group map differs from the production program",
        ));
    }
    if scheduled.layout().phase_kinds() != expected_phase_kinds {
        return Err(invalid(
            "final relation phase-kind map differs from the production program",
        ));
    }
    if relation.layout().overlay_kinds().len() != STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS {
        return Err(invalid(
            "final relation overlay map does not match the production program",
        ));
    }
    if relation.layout().scheduled_rows().start != 0
        || scheduled.layout().common_rows().start != 0
        || scheduled.layout().common_rows().len() != scheduled.common_relation().structure().n
    {
        return Err(invalid(
            "the final relation does not keep lifecycle rows at the row-zero prefix",
        ));
    }

    let lifecycle_group = expected_lifecycle_groups[terminal_arm];
    let phase_kind = expected_phase_kinds[terminal_arm];
    let recursive_group = 1;
    let semantic_links_kind = NebulaFPrimeStreamingCircuitKind::SemanticLinks.code() as usize;
    if lifecycle_group != recursive_group || phase_kind != semantic_links_kind {
        return Err(invalid("the terminal arm is not recursive SemanticLinks"));
    }
    let semantic_occurrences = expected_phase_kinds
        .iter()
        .enumerate()
        .filter(|(_, kind)| **kind == semantic_links_kind)
        .map(|(arm, _)| arm)
        .collect::<Vec<_>>();
    if semantic_occurrences != [terminal_arm] {
        return Err(invalid("SemanticLinks is not selected by exactly the terminal arm"));
    }

    let recursive = lifecycle.arm(NebulaFPrimeStreamingLifecycleArm::Recursive);
    let source_fields = lifecycle.phase_envelope_fields(NebulaFPrimeStreamingLifecycleArm::Recursive);
    let after_local = source_fields.after_local_state_digest();
    let after_payload = source_fields.after_delayed_payload();
    validate_private_source_range(recursive.m_in, recursive.m, "after local-state digest", &after_local)?;
    validate_private_source_range(recursive.m_in, recursive.m, "after delayed payload", &after_payload)?;
    if after_local
        .clone()
        .any(|field| recursive.boolean_columns().contains(&field))
    {
        return Err(invalid(
            "an after local-state digest field is Boolean in the source relation",
        ));
    }
    if after_payload
        .clone()
        .any(|field| !recursive.boolean_columns().contains(&field))
    {
        return Err(invalid("an after delayed-payload field lacks source Boolean authority"));
    }

    let compiler = scheduled
        .common_relation()
        .selective_compiler_audit()
        .ok_or_else(|| invalid("lifecycle selective compiler audit is absent"))?;
    let compiler_stages = compiler
        .source_arm_physical_stages()
        .get(recursive_group)
        .ok_or_else(|| invalid("recursive compiler stage ledger is absent"))?;
    if compiler_stages != recursive.physical_stage_ranges() {
        return Err(invalid(
            "recursive compiler stage ledger differs from the lifecycle source rows",
        ));
    }
    let local_stage = exact_stage_occurrence(compiler_stages, &after_local)?;
    let payload_stage = exact_stage_occurrence(compiler_stages, &after_payload)?;
    if local_stage != payload_stage {
        return Err(invalid(
            "terminal after-state slices have different physical stage owners",
        ));
    }
    let source_stage = compiler_stages[local_stage];
    if source_stage.path() != fprime_stage::RECURSIVE_SEMANTIC_LINKS {
        return Err(invalid(format!(
            "terminal after-state stage is {}, expected {}",
            source_stage.path(),
            fprime_stage::RECURSIVE_SEMANTIC_LINKS
        )));
    }

    let final_stage_row_runs = compiler
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| {
            run.arm() == Some(recursive_group)
                && run.source_stage_occurrence() == Some(local_stage)
                && !run.emitted_rows().is_empty()
        })
        .map(|run| NebulaFPrimeStreamingTerminalFinalRowRun {
            family: run.family(),
            rows: run.emitted_rows(),
        })
        .collect::<Vec<_>>();
    if final_stage_row_runs.is_empty() {
        return Err(invalid("recursive semantic stage owns no final selective rows"));
    }
    let common_rows = scheduled.layout().common_rows();
    if final_stage_row_runs
        .iter()
        .any(|run| run.rows.start < common_rows.start || run.rows.end > common_rows.end)
    {
        return Err(invalid("recursive semantic stage row run escapes final lifecycle rows"));
    }

    let link_rows = scheduled
        .layout()
        .common_phase_link_rows_for_kind(semantic_links_kind)
        .ok_or_else(|| invalid("SemanticLinks has no common-to-phase link-row entry"))?;
    let payload_fields = after_payload.len();
    let expected_link_rows = 2 * 4 + 2 * payload_fields;
    if link_rows.len() != expected_link_rows {
        return Err(invalid(format!(
            "SemanticLinks has {} private link rows, expected {expected_link_rows}",
            link_rows.len()
        )));
    }
    let link_contract = scheduled
        .layout()
        .common_phase_links_for_kind(semantic_links_kind)
        .ok_or_else(|| invalid("SemanticLinks has no retained source-field link contract"))?;
    if link_contract.lifecycle_group != recursive_group || link_contract.phase_kind != semantic_links_kind {
        return Err(invalid("SemanticLinks retained link contract has the wrong scope"));
    }
    let expected_common_fields = source_fields
        .before_local_state_digest()
        .chain(source_fields.before_delayed_payload())
        .chain(after_local.clone())
        .chain(after_payload.clone())
        .collect::<Vec<_>>();
    if link_contract
        .fields
        .iter()
        .map(|link| link.common_field)
        .ne(expected_common_fields)
    {
        return Err(invalid(
            "SemanticLinks retained link contract does not use the complete lifecycle envelope in canonical order",
        ));
    }
    let final_scheduled_row_start = relation.layout().scheduled_rows().start;
    let after_local_link_start = link_rows.start + 4 + payload_fields;
    let after_local_link_rows =
        final_scheduled_row_start + after_local_link_start..final_scheduled_row_start + after_local_link_start + 4;
    let after_payload_link_rows = after_local_link_rows.end..after_local_link_rows.end + payload_fields;

    let after_x_out_source_fields = *lifecycle
        .x_out_preimage_columns(NebulaFPrimeStreamingLifecycleArm::Recursive)
        .after();
    let after_x_out_fields = after_x_out_source_fields
        .into_iter()
        .map(|source_field| map_field_binding(relation, recursive_group, source_field, "after x_out preimage"))
        .collect::<Result<Vec<_>, _>>()?;
    let after_x_out = NebulaFPrimeStreamingTerminalXOutBinding {
        source_fields: after_x_out_source_fields,
        fields: after_x_out_fields,
    };

    let after_nebula_lane_source_fields = lifecycle
        .after_nebula_lane_columns(NebulaFPrimeStreamingLifecycleArm::Recursive)
        .all();
    let after_nebula_lane_fields = after_nebula_lane_source_fields
        .iter()
        .copied()
        .map(|source_field| map_field_binding(relation, recursive_group, source_field, "after Nebula lane"))
        .collect::<Result<Vec<_>, _>>()?;
    let after_nebula_lane = NebulaFPrimeStreamingTerminalLaneBinding {
        source_fields: after_nebula_lane_source_fields,
        fields: after_nebula_lane_fields,
    };

    let after_local_state_digest = map_slice(
        relation,
        recursive_group,
        after_local,
        NebulaFPrimeStreamingTerminalFieldDomain::Goldilocks,
        after_local_link_rows,
        "after local-state digest",
    )?;
    let after_delayed_payload = map_slice(
        relation,
        recursive_group,
        after_payload,
        NebulaFPrimeStreamingTerminalFieldDomain::Boolean,
        after_payload_link_rows,
        "after delayed payload",
    )?;

    let terminal_source_fields = after_x_out
        .source_fields()
        .iter()
        .copied()
        .chain(after_nebula_lane.source_fields().iter().copied())
        .chain(after_local_state_digest.source_fields())
        .chain(after_delayed_payload.source_fields())
        .collect::<BTreeSet<_>>();
    let source_stage_bindings = map_source_stage_bindings(
        compiler_stages,
        compiler.rows().emitted_runs(),
        recursive_group,
        &terminal_source_fields,
    )?;

    let outer_layout = relation.layout();
    let scheduled_layout = scheduled.layout();
    if outer_layout.public_columns() != scheduled_layout.public_columns()
        || outer_layout.scheduled_private_columns()
            != (scheduled_layout.public_columns().end..scheduled_layout.ring_padding_columns().end)
        || outer_layout.overlay_private_columns().end != outer_layout.ring_padding_columns().start
        || outer_layout.ring_padding_columns().end != relation.structure().m
    {
        return Err(invalid("canonical final-column partitions are not contiguous"));
    }
    let column_layout = NebulaFPrimeStreamingTerminalColumnLayout {
        public: outer_layout.public_columns(),
        lifecycle_private: scheduled_layout.common_private_columns(),
        phase_private: scheduled_layout.phase_private_columns(),
        schedule_selectors: scheduled_layout.schedule_selector_columns().to_vec(),
        scheduled_ring_padding: scheduled_layout.ring_padding_columns(),
        overlay_private: outer_layout.overlay_private_columns(),
        overlay_selectors: outer_layout.overlay_selector_columns().to_vec(),
        final_ring_padding: outer_layout.ring_padding_columns(),
    };

    let schedule_selector_column = scheduled.layout().schedule_selector_columns()[terminal_arm];
    let lifecycle_selector_column = scheduled.layout().common_selector_columns()[recursive_group];
    let phase_selector_column = scheduled.layout().phase_kind_selector_columns()[semantic_links_kind];
    for (name, column) in [
        ("terminal schedule selector", schedule_selector_column),
        ("recursive lifecycle selector", lifecycle_selector_column),
        ("SemanticLinks selector", phase_selector_column),
    ] {
        if column >= relation.structure().m {
            return Err(invalid(format!("{name} column {column} is outside the final relation")));
        }
    }

    Ok(NebulaFPrimeStreamingTerminalProfile {
        accepted_work_items: STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS,
        terminal_arm,
        lifecycle_group,
        phase_kind,
        schedule_selector_column,
        lifecycle_selector_column,
        phase_selector_column,
        source_rows: recursive.n,
        source_columns: recursive.m,
        source_public_columns: recursive.m_in,
        final_rows: relation.structure().n,
        final_columns: relation.structure().m,
        final_public_columns: relation.public_input_len(),
        column_layout,
        source_stage_occurrence: local_stage,
        source_stage_rows: source_stage.rows(),
        final_stage_row_runs,
        after_x_out,
        after_nebula_lane,
        after_local_state_digest,
        after_delayed_payload,
        source_stage_bindings,
    })
}

fn map_source_stage_bindings(
    stages: &[crate::engine::r1cs_circuit::PhysicalStageRange],
    emitted_runs: &[crate::frontends::r1cs_f_prime::SelectiveEmittedRowRunAudit],
    lifecycle_group: usize,
    source_fields: &BTreeSet<usize>,
) -> Result<Vec<NebulaFPrimeStreamingTerminalSourceStageBinding>, NebulaFPrimeStreamingTerminalProfileError> {
    if source_fields.is_empty() {
        return Err(invalid("terminal source-field ownership set is empty"));
    }
    let mut fields_by_stage = BTreeMap::<usize, Vec<usize>>::new();
    for &field in source_fields {
        let matches = stages
            .iter()
            .enumerate()
            .filter(|(_, stage)| stage.column_start() <= field && field < stage.column_end())
            .map(|(occurrence, _)| occurrence)
            .collect::<Vec<_>>();
        let occurrence = match matches.as_slice() {
            [occurrence] => *occurrence,
            [] => {
                return Err(invalid(format!(
                    "terminal source field {field} has no physical-stage owner"
                )))
            }
            _ => {
                return Err(invalid(format!(
                    "terminal source field {field} has more than one physical-stage owner"
                )));
            }
        };
        fields_by_stage.entry(occurrence).or_default().push(field);
    }

    fields_by_stage
        .into_iter()
        .map(|(occurrence, source_fields)| {
            let stage = stages[occurrence];
            let final_row_runs = emitted_runs
                .iter()
                .filter(|run| {
                    run.arm() == Some(lifecycle_group)
                        && run.source_stage_occurrence() == Some(occurrence)
                        && !run.emitted_rows().is_empty()
                })
                .map(|run| NebulaFPrimeStreamingTerminalFinalRowRun {
                    family: run.family(),
                    rows: run.emitted_rows(),
                })
                .collect::<Vec<_>>();
            if final_row_runs.is_empty() {
                return Err(invalid(format!(
                    "terminal source stage {} owns no final selective rows",
                    stage.path()
                )));
            }
            Ok(NebulaFPrimeStreamingTerminalSourceStageBinding {
                source_stage_occurrence: occurrence,
                source_stage_path: stage.path(),
                source_rows: stage.rows(),
                source_fields,
                final_row_runs,
            })
        })
        .collect()
}

fn validate_private_source_range(
    public: usize,
    columns: usize,
    name: &'static str,
    range: &Range<usize>,
) -> Result<(), NebulaFPrimeStreamingTerminalProfileError> {
    if range.is_empty() || range.start < public || range.end > columns {
        return Err(invalid(format!(
            "{name} source range [{}, {}) is not a nonempty private range inside {public}..{columns}",
            range.start, range.end
        )));
    }
    Ok(())
}

fn exact_stage_occurrence(
    stages: &[crate::engine::r1cs_circuit::PhysicalStageRange],
    fields: &Range<usize>,
) -> Result<usize, NebulaFPrimeStreamingTerminalProfileError> {
    let matches = stages
        .iter()
        .enumerate()
        .filter(|(_, stage)| stage.column_start() <= fields.start && fields.end <= stage.column_end())
        .map(|(occurrence, _)| occurrence)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [occurrence] => Ok(*occurrence),
        [] => Err(invalid(format!(
            "source range [{}, {}) has no physical stage owner",
            fields.start, fields.end
        ))),
        _ => Err(invalid(format!(
            "source range [{}, {}) has more than one physical stage owner",
            fields.start, fields.end
        ))),
    }
}

fn map_slice(
    relation: &ScheduledLinkedOverlayLowNormR1cs,
    lifecycle_group: usize,
    source_fields: Range<usize>,
    source_domain: NebulaFPrimeStreamingTerminalFieldDomain,
    final_common_phase_link_rows: Range<usize>,
    name: &'static str,
) -> Result<NebulaFPrimeStreamingTerminalSliceBinding, NebulaFPrimeStreamingTerminalProfileError> {
    if final_common_phase_link_rows.len() != source_fields.len() {
        return Err(invalid(format!(
            "{name} has {} source fields but {} final link rows",
            source_fields.len(),
            final_common_phase_link_rows.len()
        )));
    }
    let fields = source_fields
        .clone()
        .map(|source_field| map_field_binding(relation, lifecycle_group, source_field, name))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NebulaFPrimeStreamingTerminalSliceBinding {
        source_fields,
        source_domain,
        fields,
        final_common_phase_link_rows,
    })
}

fn map_field_binding(
    relation: &ScheduledLinkedOverlayLowNormR1cs,
    lifecycle_group: usize,
    source_field: usize,
    name: &'static str,
) -> Result<NebulaFPrimeStreamingTerminalFieldBinding, NebulaFPrimeStreamingTerminalProfileError> {
    let decoder_terms = relation
        .common_field_decoding_terms(lifecycle_group, source_field)
        .map_err(|error| invalid(format!("{name} source field {source_field}: {error}")))?
        .into_iter()
        .map(|(final_column, coefficient)| NebulaFPrimeStreamingTerminalDecoderTerm {
            final_column,
            coefficient,
        })
        .collect::<Vec<_>>();
    if decoder_terms.is_empty() {
        return Err(invalid(format!(
            "{name} source field {source_field} has an empty final decoder"
        )));
    }
    Ok(NebulaFPrimeStreamingTerminalFieldBinding {
        source_field,
        decoder_terms,
    })
}
