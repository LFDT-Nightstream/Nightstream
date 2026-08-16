//! Exact source-to-selective placement for the streaming lifecycle relation.
//!
//! Owns the base and recursive source geometry, the canonical two-arm
//! selective layout, the exclusive physical-stage schedule, and exact
//! source-to-final row and XOut-field bindings.
//!
//! It does not own phase circuits, the 400-arm scheduled relation, protocol
//! authority, terminal acceptance, or permission to remove a constraint.

use std::ops::Range;

use neo_math::F;
use thiserror::Error;

use super::streaming_lifecycle_relation::{
    NebulaFPrimeStreamingLifecycleArm, NebulaFPrimeStreamingLifecycleSourceArms,
};
use crate::frontends::r1cs_f_prime::{
    MultiBranchLowNormR1cs, SelectiveEmittedRowFamily, SelectiveRewriteKind, SelectiveSourceRowDisposition,
};

pub const STREAMING_LIFECYCLE_PROFILE_ID: &str = "nightstream/goldilocks/streaming-lifecycle-selective/v1";
pub const STREAMING_LIFECYCLE_BASE_SOURCE_ARTIFACT_ID: &str =
    "rust:nightstream/streaming-lifecycle-base/source-rows/v1";
pub const STREAMING_LIFECYCLE_RECURSIVE_SOURCE_ARTIFACT_ID: &str =
    "rust:nightstream/streaming-lifecycle-recursive/source-rows/v1";
pub const STREAMING_LIFECYCLE_FINAL_ARTIFACT_ID: &str = "rust:nightstream/streaming-lifecycle-selective/final-rows/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleDecoderTerm {
    final_column: usize,
    coefficient: F,
}

impl NebulaFPrimeStreamingLifecycleDecoderTerm {
    pub const fn final_column(self) -> usize {
        self.final_column
    }

    pub const fn coefficient(self) -> F {
        self.coefficient
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleFieldBinding {
    source_column: usize,
    decoder_terms: Vec<NebulaFPrimeStreamingLifecycleDecoderTerm>,
}

impl NebulaFPrimeStreamingLifecycleFieldBinding {
    pub const fn source_column(&self) -> usize {
        self.source_column
    }

    pub fn decoder_terms(&self) -> &[NebulaFPrimeStreamingLifecycleDecoderTerm] {
        &self.decoder_terms
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleXOutBinding {
    source_columns: [usize; 32],
    fields: Vec<NebulaFPrimeStreamingLifecycleFieldBinding>,
}

impl NebulaFPrimeStreamingLifecycleXOutBinding {
    pub fn source_columns(&self) -> &[usize; 32] {
        &self.source_columns
    }

    pub fn fields(&self) -> &[NebulaFPrimeStreamingLifecycleFieldBinding] {
        &self.fields
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleSourceRunBinding {
    source_rows: Range<usize>,
    disposition: SelectiveSourceRowDisposition,
    emitted_start: Option<usize>,
}

impl NebulaFPrimeStreamingLifecycleSourceRunBinding {
    pub fn source_rows(&self) -> Range<usize> {
        self.source_rows.clone()
    }

    pub const fn disposition(&self) -> SelectiveSourceRowDisposition {
        self.disposition
    }

    pub const fn emitted_start(&self) -> Option<usize> {
        self.emitted_start
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleFinalRowRun {
    family: SelectiveEmittedRowFamily,
    rows: Range<usize>,
    rewrite_id: Option<usize>,
}

impl NebulaFPrimeStreamingLifecycleFinalRowRun {
    pub const fn family(&self) -> SelectiveEmittedRowFamily {
        self.family
    }

    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }

    pub const fn rewrite_id(&self) -> Option<usize> {
        self.rewrite_id
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleRewriteBinding {
    rewrite_id: usize,
    kind: SelectiveRewriteKind,
    source_rows: Vec<Range<usize>>,
    final_rows: Range<usize>,
}

impl NebulaFPrimeStreamingLifecycleRewriteBinding {
    pub const fn rewrite_id(&self) -> usize {
        self.rewrite_id
    }

    pub const fn kind(&self) -> SelectiveRewriteKind {
        self.kind
    }

    pub fn source_rows(&self) -> &[Range<usize>] {
        &self.source_rows
    }

    pub fn final_rows(&self) -> Range<usize> {
        self.final_rows.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleStageBinding {
    occurrence: usize,
    path: &'static str,
    source_rows: Range<usize>,
    source_columns: Range<usize>,
    source_runs: Vec<NebulaFPrimeStreamingLifecycleSourceRunBinding>,
    final_row_runs: Vec<NebulaFPrimeStreamingLifecycleFinalRowRun>,
    rewrites: Vec<NebulaFPrimeStreamingLifecycleRewriteBinding>,
}

impl NebulaFPrimeStreamingLifecycleStageBinding {
    pub const fn occurrence(&self) -> usize {
        self.occurrence
    }

    pub const fn path(&self) -> &'static str {
        self.path
    }

    pub fn source_rows(&self) -> Range<usize> {
        self.source_rows.clone()
    }

    pub fn source_columns(&self) -> Range<usize> {
        self.source_columns.clone()
    }

    pub fn source_runs(&self) -> &[NebulaFPrimeStreamingLifecycleSourceRunBinding] {
        &self.source_runs
    }

    pub fn final_row_runs(&self) -> &[NebulaFPrimeStreamingLifecycleFinalRowRun] {
        &self.final_row_runs
    }

    pub fn rewrites(&self) -> &[NebulaFPrimeStreamingLifecycleRewriteBinding] {
        &self.rewrites
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleArmProfile {
    arm: NebulaFPrimeStreamingLifecycleArm,
    source_rows: usize,
    source_columns: usize,
    source_public_columns: usize,
    stages: Vec<NebulaFPrimeStreamingLifecycleStageBinding>,
    compiler_row_runs: Vec<NebulaFPrimeStreamingLifecycleFinalRowRun>,
    before_x_out: NebulaFPrimeStreamingLifecycleXOutBinding,
    after_x_out: NebulaFPrimeStreamingLifecycleXOutBinding,
}

impl NebulaFPrimeStreamingLifecycleArmProfile {
    pub const fn arm(&self) -> NebulaFPrimeStreamingLifecycleArm {
        self.arm
    }

    pub const fn lifecycle_scope(&self) -> &'static str {
        match self.arm {
            NebulaFPrimeStreamingLifecycleArm::Base => "base",
            NebulaFPrimeStreamingLifecycleArm::Recursive => "recursive",
        }
    }

    pub const fn source_artifact_identity(&self) -> &'static str {
        match self.arm {
            NebulaFPrimeStreamingLifecycleArm::Base => STREAMING_LIFECYCLE_BASE_SOURCE_ARTIFACT_ID,
            NebulaFPrimeStreamingLifecycleArm::Recursive => STREAMING_LIFECYCLE_RECURSIVE_SOURCE_ARTIFACT_ID,
        }
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

    pub fn stages(&self) -> &[NebulaFPrimeStreamingLifecycleStageBinding] {
        &self.stages
    }

    /// Arm-local rows introduced by selective lowering rather than by one
    /// physical source stage. These are the exact arm-domain rows.
    pub fn compiler_row_runs(&self) -> &[NebulaFPrimeStreamingLifecycleFinalRowRun] {
        &self.compiler_row_runs
    }

    pub fn before_x_out(&self) -> &NebulaFPrimeStreamingLifecycleXOutBinding {
        &self.before_x_out
    }

    pub fn after_x_out(&self) -> &NebulaFPrimeStreamingLifecycleXOutBinding {
        &self.after_x_out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleColumnLayout {
    logical_public_columns: Range<usize>,
    public_columns: Range<usize>,
    public_padding_columns: Vec<usize>,
    selector_columns: Vec<usize>,
    private_alignment_padding_columns: Vec<usize>,
    shared_private_columns: Range<usize>,
    branch_columns: Range<usize>,
    ring_alignment_padding_columns: Range<usize>,
}

impl NebulaFPrimeStreamingLifecycleColumnLayout {
    pub fn logical_public_columns(&self) -> Range<usize> {
        self.logical_public_columns.clone()
    }

    pub fn public_columns(&self) -> Range<usize> {
        self.public_columns.clone()
    }

    pub fn public_padding_columns(&self) -> &[usize] {
        &self.public_padding_columns
    }

    pub fn selector_columns(&self) -> &[usize] {
        &self.selector_columns
    }

    pub fn private_alignment_padding_columns(&self) -> &[usize] {
        &self.private_alignment_padding_columns
    }

    pub fn shared_private_columns(&self) -> Range<usize> {
        self.shared_private_columns.clone()
    }

    pub fn branch_columns(&self) -> Range<usize> {
        self.branch_columns.clone()
    }

    pub fn ring_alignment_padding_columns(&self) -> Range<usize> {
        self.ring_alignment_padding_columns.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingLifecycleProfile {
    final_rows: usize,
    final_columns: usize,
    final_public_columns: usize,
    column_layout: NebulaFPrimeStreamingLifecycleColumnLayout,
    global_row_runs: Vec<NebulaFPrimeStreamingLifecycleFinalRowRun>,
    arms: [NebulaFPrimeStreamingLifecycleArmProfile; 2],
}

impl NebulaFPrimeStreamingLifecycleProfile {
    pub const fn profile_id(&self) -> &'static str {
        STREAMING_LIFECYCLE_PROFILE_ID
    }

    pub const fn final_artifact_identity(&self) -> &'static str {
        STREAMING_LIFECYCLE_FINAL_ARTIFACT_ID
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

    pub fn column_layout(&self) -> &NebulaFPrimeStreamingLifecycleColumnLayout {
        &self.column_layout
    }

    /// Final rows owned by the shared selective compiler, not by one source
    /// arm. This includes the selector domains, the common public domain, and
    /// the exactly-one selector row.
    pub fn global_row_runs(&self) -> &[NebulaFPrimeStreamingLifecycleFinalRowRun] {
        &self.global_row_runs
    }

    pub fn arm(&self, arm: NebulaFPrimeStreamingLifecycleArm) -> &NebulaFPrimeStreamingLifecycleArmProfile {
        &self.arms[arm_index(arm)]
    }
}

#[derive(Debug, Error)]
pub enum NebulaFPrimeStreamingLifecycleProfileError {
    #[error("streaming F-prime lifecycle profile: {0}")]
    Invalid(String),
}

fn invalid(message: impl Into<String>) -> NebulaFPrimeStreamingLifecycleProfileError {
    NebulaFPrimeStreamingLifecycleProfileError::Invalid(message.into())
}

pub fn production_streaming_lifecycle_profile(
    lifecycle: &NebulaFPrimeStreamingLifecycleSourceArms,
    relation: &MultiBranchLowNormR1cs,
) -> Result<NebulaFPrimeStreamingLifecycleProfile, NebulaFPrimeStreamingLifecycleProfileError> {
    let compiler = relation
        .selective_compiler_audit()
        .ok_or_else(|| invalid("selective compiler audit is absent"))?;
    let layout = compiler.layout();
    if compiler.rows().arms().len() != 2
        || compiler.source_arm_physical_stages().len() != 2
        || layout.selector_columns().len() != 2
        || compiler.rows().total_rows() != relation.structure().n
        || layout.total_columns() != relation.structure().m
        || layout.public_input_len() != relation.public_input_len()
    {
        return Err(invalid(
            "selective relation geometry differs from its two-arm compiler audit",
        ));
    }
    validate_final_row_partition(compiler.rows().emitted_runs(), relation.structure().n)?;
    let global_row_runs = compiler
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.arm().is_none())
        .map(|run| NebulaFPrimeStreamingLifecycleFinalRowRun {
            family: run.family(),
            rows: run.emitted_rows(),
            rewrite_id: run.rewrite_id().map(|id| id.index()),
        })
        .collect::<Vec<_>>();
    validate_global_emitted_ownership(&global_row_runs)?;

    let arms = [
        build_arm_profile(lifecycle, relation, NebulaFPrimeStreamingLifecycleArm::Base)?,
        build_arm_profile(lifecycle, relation, NebulaFPrimeStreamingLifecycleArm::Recursive)?,
    ];
    let column_layout = NebulaFPrimeStreamingLifecycleColumnLayout {
        logical_public_columns: 0..layout.logical_public_input_len(),
        public_columns: 0..layout.public_input_len(),
        public_padding_columns: layout.public_padding_columns().to_vec(),
        selector_columns: layout.selector_columns().to_vec(),
        private_alignment_padding_columns: layout.private_alignment_padding_columns().to_vec(),
        shared_private_columns: layout.shared_private_columns(),
        branch_columns: layout.branch_columns(),
        ring_alignment_padding_columns: layout.ring_alignment_padding_columns(),
    };
    validate_column_layout(&column_layout, relation.structure().m)?;

    Ok(NebulaFPrimeStreamingLifecycleProfile {
        final_rows: relation.structure().n,
        final_columns: relation.structure().m,
        final_public_columns: relation.public_input_len(),
        column_layout,
        global_row_runs,
        arms,
    })
}

fn validate_global_emitted_ownership(
    runs: &[NebulaFPrimeStreamingLifecycleFinalRowRun],
) -> Result<(), NebulaFPrimeStreamingLifecycleProfileError> {
    for required in [
        SelectiveEmittedRowFamily::SelectorDomain,
        SelectiveEmittedRowFamily::SharedDomain,
        SelectiveEmittedRowFamily::OneHot,
    ] {
        let matches = runs.iter().filter(|run| run.family() == required).count();
        if matches != 1 {
            return Err(invalid(format!(
                "global selective owner {required:?} has {matches} runs, expected one"
            )));
        }
    }
    for required in [
        SelectiveEmittedRowFamily::SelectorDomain,
        SelectiveEmittedRowFamily::OneHot,
    ] {
        if runs
            .iter()
            .find(|run| run.family() == required)
            .is_none_or(|run| run.rows().is_empty())
        {
            return Err(invalid(format!(
                "global selective owner {required:?} has an empty row run"
            )));
        }
    }
    if runs.iter().any(|run| run.rewrite_id().is_some()) {
        return Err(invalid(
            "global selective row ownership unexpectedly uses a source rewrite",
        ));
    }
    Ok(())
}

fn build_arm_profile(
    lifecycle: &NebulaFPrimeStreamingLifecycleSourceArms,
    relation: &MultiBranchLowNormR1cs,
    arm: NebulaFPrimeStreamingLifecycleArm,
) -> Result<NebulaFPrimeStreamingLifecycleArmProfile, NebulaFPrimeStreamingLifecycleProfileError> {
    let arm_index = arm_index(arm);
    let source = lifecycle.arm(arm);
    let compiler = relation
        .selective_compiler_audit()
        .ok_or_else(|| invalid("selective compiler audit is absent"))?;
    let stages = compiler
        .source_arm_physical_stages()
        .get(arm_index)
        .ok_or_else(|| invalid(format!("{} compiler stage schedule is absent", scope(arm))))?;
    if stages != source.physical_stage_ranges() {
        return Err(invalid(format!(
            "{} compiler stage schedule differs from exact source rows",
            scope(arm)
        )));
    }
    validate_source_stages(source.m_in, source.n, source.m, stages)?;
    let mapping = compiler
        .rows()
        .arms()
        .get(arm_index)
        .ok_or_else(|| invalid(format!("{} source-row mapping is absent", scope(arm))))?;
    validate_source_runs(source.n, stages, mapping.source_runs())?;

    let stage_bindings = stages
        .iter()
        .enumerate()
        .map(|(occurrence, stage)| {
            let source_runs = mapping
                .source_runs()
                .iter()
                .filter(|run| run.stage_occurrence() == Some(occurrence))
                .map(|run| NebulaFPrimeStreamingLifecycleSourceRunBinding {
                    source_rows: run.source_rows(),
                    disposition: run.disposition(),
                    emitted_start: run.emitted_start(),
                })
                .collect::<Vec<_>>();
            let final_row_runs = compiler
                .rows()
                .emitted_runs()
                .iter()
                .filter(|run| run.arm() == Some(arm_index) && run.source_stage_occurrence() == Some(occurrence))
                .map(|run| NebulaFPrimeStreamingLifecycleFinalRowRun {
                    family: run.family(),
                    rows: run.emitted_rows(),
                    rewrite_id: run.rewrite_id().map(|id| id.index()),
                })
                .collect::<Vec<_>>();
            let rewrites = compiler
                .rows()
                .rewrites()
                .iter()
                .filter(|rewrite| rewrite.arm() == arm_index && rewrite.source_stage_occurrence() == Some(occurrence))
                .map(|rewrite| NebulaFPrimeStreamingLifecycleRewriteBinding {
                    rewrite_id: rewrite.id().index(),
                    kind: rewrite.kind(),
                    source_rows: rewrite.source_rows().to_vec(),
                    final_rows: rewrite.emitted_rows(),
                })
                .collect::<Vec<_>>();
            Ok(NebulaFPrimeStreamingLifecycleStageBinding {
                occurrence,
                path: stage.path(),
                source_rows: stage.rows(),
                source_columns: stage.columns(),
                source_runs,
                final_row_runs,
                rewrites,
            })
        })
        .collect::<Result<Vec<_>, NebulaFPrimeStreamingLifecycleProfileError>>()?;
    validate_arm_emitted_ownership(compiler, arm_index, stage_bindings.len())?;
    let compiler_row_runs = compiler
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.arm() == Some(arm_index) && run.source_stage_occurrence().is_none())
        .map(|run| NebulaFPrimeStreamingLifecycleFinalRowRun {
            family: run.family(),
            rows: run.emitted_rows(),
            rewrite_id: run.rewrite_id().map(|id| id.index()),
        })
        .collect();

    let x_out = lifecycle.x_out_preimage_columns(arm);
    Ok(NebulaFPrimeStreamingLifecycleArmProfile {
        arm,
        source_rows: source.n,
        source_columns: source.m,
        source_public_columns: source.m_in,
        stages: stage_bindings,
        compiler_row_runs,
        before_x_out: map_x_out(relation, arm_index, *x_out.before(), "before XOut")?,
        after_x_out: map_x_out(relation, arm_index, *x_out.after(), "after XOut")?,
    })
}

fn validate_source_stages(
    public_columns: usize,
    source_rows: usize,
    source_columns: usize,
    stages: &[crate::engine::r1cs_circuit::PhysicalStageRange],
) -> Result<(), NebulaFPrimeStreamingLifecycleProfileError> {
    if stages.is_empty() {
        return Err(invalid("source physical-stage schedule is empty"));
    }
    let mut row_cursor = 0usize;
    let mut column_cursor = public_columns;
    for (occurrence, stage) in stages.iter().enumerate() {
        if stage.path().trim().is_empty()
            || stage.row_start() != row_cursor
            || stage.column_start() != column_cursor
            || stage.row_end() < stage.row_start()
            || stage.column_end() < stage.column_start()
        {
            return Err(invalid(format!(
                "physical stage {occurrence} does not continue the exact row and private-column partitions"
            )));
        }
        row_cursor = stage.row_end();
        column_cursor = stage.column_end();
    }
    if row_cursor != source_rows || column_cursor != source_columns {
        return Err(invalid(format!(
            "physical stages end at rows {row_cursor} and columns {column_cursor}; expected {source_rows} and {source_columns}"
        )));
    }
    Ok(())
}

fn validate_source_runs(
    source_rows: usize,
    stages: &[crate::engine::r1cs_circuit::PhysicalStageRange],
    runs: &[crate::frontends::r1cs_f_prime::SelectiveSourceRowRunAudit],
) -> Result<(), NebulaFPrimeStreamingLifecycleProfileError> {
    let mut cursor = 0usize;
    for (index, run) in runs.iter().enumerate() {
        let rows = run.source_rows();
        let occurrence = run
            .stage_occurrence()
            .ok_or_else(|| invalid(format!("source run {index} has no physical-stage owner")))?;
        let stage = stages
            .get(occurrence)
            .ok_or_else(|| invalid(format!("source run {index} has invalid stage {occurrence}")))?;
        if rows.start != cursor || rows.is_empty() || rows.start < stage.row_start() || rows.end > stage.row_end() {
            return Err(invalid(format!(
                "source run {index} does not continue its exact stage-owned row partition"
            )));
        }
        cursor = rows.end;
    }
    if cursor != source_rows {
        return Err(invalid(format!(
            "source-row runs end at {cursor}, expected {source_rows}"
        )));
    }
    Ok(())
}

fn validate_arm_emitted_ownership(
    compiler: &crate::frontends::r1cs_f_prime::SelectiveCompilerAudit,
    arm: usize,
    stage_count: usize,
) -> Result<(), NebulaFPrimeStreamingLifecycleProfileError> {
    for run in compiler
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.arm() == Some(arm))
    {
        match run.source_stage_occurrence() {
            Some(occurrence) if occurrence < stage_count => {}
            None if run.family() == SelectiveEmittedRowFamily::ArmDomain && run.rewrite_id().is_none() => {}
            _ => {
                return Err(invalid(format!(
                    "arm {arm} {:?} row run has no valid compiler or physical-stage owner",
                    run.family()
                )));
            }
        }
    }
    for rewrite in compiler
        .rows()
        .rewrites()
        .iter()
        .filter(|rewrite| rewrite.arm() == arm)
    {
        if rewrite
            .source_stage_occurrence()
            .is_none_or(|occurrence| occurrence >= stage_count)
        {
            return Err(invalid(format!(
                "arm {arm} rewrite {} has no valid physical-stage owner",
                rewrite.id().index()
            )));
        }
    }
    Ok(())
}

fn validate_final_row_partition(
    runs: &[crate::frontends::r1cs_f_prime::SelectiveEmittedRowRunAudit],
    final_rows: usize,
) -> Result<(), NebulaFPrimeStreamingLifecycleProfileError> {
    let mut cursor = 0usize;
    for (index, run) in runs.iter().enumerate() {
        let rows = run.emitted_rows();
        if rows.is_empty() {
            if rows.start != rows.end || rows.start > final_rows {
                return Err(invalid(format!("empty final row run {index} is out of range")));
            }
            continue;
        }
        if rows.start != cursor || rows.end > final_rows {
            return Err(invalid(format!(
                "final row run {index} does not continue the exact emitted partition"
            )));
        }
        cursor = rows.end;
    }
    if cursor != final_rows {
        return Err(invalid(format!(
            "final emitted row runs end at {cursor}, expected {final_rows}"
        )));
    }
    Ok(())
}

fn validate_column_layout(
    layout: &NebulaFPrimeStreamingLifecycleColumnLayout,
    final_columns: usize,
) -> Result<(), NebulaFPrimeStreamingLifecycleProfileError> {
    if layout.logical_public_columns.start != 0
        || layout.public_columns.start != 0
        || layout.logical_public_columns.end > layout.public_columns.end
        || layout.ring_alignment_padding_columns.end != final_columns
    {
        return Err(invalid("canonical selective column boundaries are invalid"));
    }
    let mut columns = (0..layout.logical_public_columns.end).collect::<Vec<_>>();
    columns.extend(layout.public_padding_columns.iter().copied());
    columns.extend(layout.selector_columns.iter().copied());
    columns.extend(layout.private_alignment_padding_columns.iter().copied());
    columns.extend(layout.shared_private_columns.clone());
    columns.extend(layout.branch_columns.clone());
    columns.extend(layout.ring_alignment_padding_columns.clone());
    if columns.iter().copied().ne(0..final_columns) {
        return Err(invalid(
            "canonical selective column regions do not partition every final column",
        ));
    }
    Ok(())
}

fn map_x_out(
    relation: &MultiBranchLowNormR1cs,
    arm: usize,
    source_columns: [usize; 32],
    name: &'static str,
) -> Result<NebulaFPrimeStreamingLifecycleXOutBinding, NebulaFPrimeStreamingLifecycleProfileError> {
    let fields = source_columns
        .iter()
        .copied()
        .map(|source_column| {
            let decoder_terms = relation
                .source_field_decoding_terms(arm, source_column)
                .map_err(|error| invalid(format!("{name} source column {source_column}: {error}")))?
                .into_iter()
                .map(|(final_column, coefficient)| {
                    if final_column >= relation.structure().m {
                        return Err(invalid(format!(
                            "{name} source column {source_column} decodes outside the final relation"
                        )));
                    }
                    Ok(NebulaFPrimeStreamingLifecycleDecoderTerm {
                        final_column,
                        coefficient,
                    })
                })
                .collect::<Result<Vec<_>, NebulaFPrimeStreamingLifecycleProfileError>>()?;
            if decoder_terms.is_empty() {
                return Err(invalid(format!(
                    "{name} source column {source_column} has an empty final decoder"
                )));
            }
            Ok(NebulaFPrimeStreamingLifecycleFieldBinding {
                source_column,
                decoder_terms,
            })
        })
        .collect::<Result<Vec<_>, NebulaFPrimeStreamingLifecycleProfileError>>()?;
    Ok(NebulaFPrimeStreamingLifecycleXOutBinding { source_columns, fields })
}

const fn arm_index(arm: NebulaFPrimeStreamingLifecycleArm) -> usize {
    match arm {
        NebulaFPrimeStreamingLifecycleArm::Base => 0,
        NebulaFPrimeStreamingLifecycleArm::Recursive => 1,
    }
}

const fn scope(arm: NebulaFPrimeStreamingLifecycleArm) -> &'static str {
    match arm {
        NebulaFPrimeStreamingLifecycleArm::Base => "base",
        NebulaFPrimeStreamingLifecycleArm::Recursive => "recursive",
    }
}
