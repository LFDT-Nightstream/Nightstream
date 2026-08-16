//! Shared-public composition of lifecycle-common and phase-local CCS relations.
//!
//! Owns: block composition of two selective low-norm relations, the fixed
//! phase-to-lifecycle-group map, exact group-sum rows, phase-activation rows,
//! optional private field links, shared-public assignment checks, and
//! zero-constrained final padding.
//!
//! Does not own: either source relation, the meaning of the shared public
//! fields, phase scheduling, or semantic refinement to the Nebula F' relation.
//!
//! Emits constraints: yes. Common rows and phase rows are each embedded once.
//! Group equality rows enforce `g_j = sum(i in j, s_i)`. One activation row per
//! phase enforces `s_i * g_group(i) = s_i`.
//!
//! Authority boundary: private assignments remain separate unless an explicit
//! verifier-owned field-link contract joins them. A field link reconstructs
//! both source fields from their exact low-norm slots and gates equality by the
//! selected phase-kind selector.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, GeometricRowRun, SeededPhi81LinearBlock};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use super::selective::{A, B, C, GENERAL_SELECTOR, SELECTIVE_ARITY};
use super::{is_canonical_selective_low_norm_polynomial, LowNormR1csError, MultiBranchLowNormR1cs};
use crate::paper::relations::Structure;

/// Exact row and column placement of one grouped phase composition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupedPhaseLayout {
    public_columns: Range<usize>,
    common_private_columns: Range<usize>,
    phase_private_columns: Range<usize>,
    ring_padding_columns: Range<usize>,
    common_rows: Range<usize>,
    phase_rows: Range<usize>,
    group_equality_rows: Range<usize>,
    phase_activation_rows: Range<usize>,
    ring_padding_rows: Range<usize>,
    common_selector_columns: Vec<usize>,
    phase_selector_columns: Vec<usize>,
    phase_groups: Vec<usize>,
}

impl GroupedPhaseLayout {
    /// Shared encoded public prefix used by both component relations.
    pub fn public_columns(&self) -> Range<usize> {
        self.public_columns.clone()
    }

    /// Private columns owned by the common lifecycle relation.
    pub fn common_private_columns(&self) -> Range<usize> {
        self.common_private_columns.clone()
    }

    /// Private columns owned by the phase-local relation.
    pub fn phase_private_columns(&self) -> Range<usize> {
        self.phase_private_columns.clone()
    }

    /// Final zero padding that aligns the joint assignment to the ring width.
    pub fn ring_padding_columns(&self) -> Range<usize> {
        self.ring_padding_columns.clone()
    }

    /// Rows copied once from the lifecycle-common relation.
    pub fn common_rows(&self) -> Range<usize> {
        self.common_rows.clone()
    }

    /// Rows copied once from the phase-local relation.
    pub fn phase_rows(&self) -> Range<usize> {
        self.phase_rows.clone()
    }

    /// Linear rows `g_j = sum(i in j, s_i)` in lifecycle-group order.
    pub fn group_equality_rows(&self) -> Range<usize> {
        self.group_equality_rows.clone()
    }

    /// Product rows `s_i * g_group(i) = s_i` in phase order.
    pub fn phase_activation_rows(&self) -> Range<usize> {
        self.phase_activation_rows.clone()
    }

    /// Rows that force final ring-alignment columns to zero.
    pub fn ring_padding_rows(&self) -> Range<usize> {
        self.ring_padding_rows.clone()
    }

    /// Embedded selector column for each lifecycle group.
    pub fn common_selector_columns(&self) -> &[usize] {
        &self.common_selector_columns
    }

    /// Embedded selector column for each phase.
    pub fn phase_selector_columns(&self) -> &[usize] {
        &self.phase_selector_columns
    }

    /// Lifecycle group selected by each phase.
    pub fn phase_groups(&self) -> &[usize] {
        &self.phase_groups
    }

    /// Total joint assignment width, including ring padding.
    pub fn columns(&self) -> usize {
        self.ring_padding_columns.end
    }

    /// Total joint row count.
    pub fn rows(&self) -> usize {
        self.ring_padding_rows.end
    }
}

/// Failure to construct or encode one grouped phase relation.
#[derive(Debug, Error)]
pub enum GroupedPhaseError {
    /// One component is not the exact selective low-norm relation.
    #[error("grouped phase composition: {owner} relation does not use the canonical selective polynomial")]
    NonCanonicalPolynomial {
        /// Component name.
        owner: &'static str,
    },
    /// The component matrix count differs from the selective arity.
    #[error("grouped phase composition: {owner} relation has {actual} matrices, expected {SELECTIVE_ARITY}")]
    MatrixCount {
        /// Component name.
        owner: &'static str,
        /// Actual matrix count.
        actual: usize,
    },
    /// The two component relations do not expose the same encoded public prefix.
    #[error("grouped phase composition: common public width {common} differs from phase public width {phase}")]
    PublicWidthMismatch {
        /// Common relation public width.
        common: usize,
        /// Phase relation public width.
        phase: usize,
    },
    /// A component public prefix exceeds its assignment width.
    #[error("grouped phase composition: {owner} public width {public} exceeds relation width {columns}")]
    PublicWidthOutOfBounds {
        /// Component name.
        owner: &'static str,
        /// Public width.
        public: usize,
        /// Relation width.
        columns: usize,
    },
    /// The fixed phase-group map has the wrong length.
    #[error("grouped phase composition: phase-group map has {actual} entries, expected {expected}")]
    PhaseGroupCount { actual: usize, expected: usize },
    /// One phase names a lifecycle group that does not exist.
    #[error("grouped phase composition: phase {phase} names group {group}, but only {groups} groups exist")]
    PhaseGroupOutOfRange {
        /// Phase index.
        phase: usize,
        /// Invalid group index.
        group: usize,
        /// Lifecycle group count.
        groups: usize,
    },
    /// One selector column lies outside its component relation.
    #[error("grouped phase composition: {owner} selector {selector} is outside relation width {columns}")]
    SelectorOutOfBounds {
        /// Component name.
        owner: &'static str,
        /// Selector column.
        selector: usize,
        /// Relation width.
        columns: usize,
    },
    /// Artifact-backed matrices cannot be embedded without their evaluator.
    #[error("grouped phase composition: {owner} matrix {matrix} is verifier-artifact backed")]
    VerifierArtifactMatrix {
        /// Component name.
        owner: &'static str,
        /// Matrix index.
        matrix: usize,
    },
    /// A compact source range does not stay contiguous under the embedding.
    #[error("grouped phase composition: {owner} matrix {matrix} compact range [{start}, {end}) is not contiguous")]
    NonContiguousCompactRange {
        /// Component name.
        owner: &'static str,
        /// Matrix index.
        matrix: usize,
        /// Source range start.
        start: usize,
        /// Source range end.
        end: usize,
    },
    /// Compact matrix construction failed.
    #[error("grouped phase composition: compact matrix construction failed: {0}")]
    CompactMatrix(String),
    /// Joint CCS construction failed.
    #[error("grouped phase composition: joint CCS construction failed: {0}")]
    Structure(String),
    /// A requested phase does not exist.
    #[error("grouped phase encoding: phase {phase} is outside 0..{phases}")]
    PhaseOutOfRange {
        /// Requested phase.
        phase: usize,
        /// Phase count.
        phases: usize,
    },
    /// The two encoded public assignments differ.
    #[error("grouped phase encoding: shared public coordinate {coordinate} differs")]
    PublicAssignmentMismatch {
        /// First differing public coordinate.
        coordinate: usize,
    },
    /// One component low-norm encoder rejected its source assignment.
    #[error(transparent)]
    LowNorm(#[from] LowNormR1csError),
    /// A compact seeded block could not be moved to the joint matrix.
    #[error(transparent)]
    SeededPhi81(#[from] neo_ccs::SeededPhi81Error),
}

/// Public bit ranges used to bind one schedule selector to its exact cursor
/// transition. Both ranges use little-endian order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledCursorBits {
    before: Range<usize>,
    after: Range<usize>,
}

/// Equality between one lifecycle source field and one phase source field.
/// Field numbers use the normalized field-R1CS numbering before low-norm
/// lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduledCommonPhaseFieldLink {
    pub common_field: usize,
    pub phase_field: usize,
}

/// Private field links for one phase kind and its only valid lifecycle group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledPhaseKindLinks {
    pub lifecycle_group: usize,
    pub phase_kind: usize,
    pub fields: Vec<ScheduledCommonPhaseFieldLink>,
}

impl ScheduledCursorBits {
    pub fn new(before: Range<usize>, after: Range<usize>) -> Self {
        Self { before, after }
    }

    pub fn before(&self) -> Range<usize> {
        self.before.clone()
    }

    pub fn after(&self) -> Range<usize> {
        self.after.clone()
    }
}

/// Failure to compose one exact work schedule from shared lifecycle and
/// phase-kind relations.
#[derive(Debug, Error)]
pub enum ScheduledGroupedPhaseError {
    #[error(transparent)]
    Grouped(#[from] GroupedPhaseError),
    #[error("scheduled grouped phase composition: schedule is empty")]
    EmptySchedule,
    #[error(
        "scheduled grouped phase composition: lifecycle map has {lifecycle} entries but phase-kind map has {phase_kinds}"
    )]
    ScheduleLengthMismatch {
        lifecycle: usize,
        phase_kinds: usize,
    },
    #[error(
        "scheduled grouped phase composition: schedule arm {arm} names lifecycle group {group}, but only {groups} groups exist"
    )]
    LifecycleGroupOutOfRange {
        arm: usize,
        group: usize,
        groups: usize,
    },
    #[error(
        "scheduled grouped phase composition: schedule arm {arm} names phase kind {kind}, but only {phase_kinds} kinds exist"
    )]
    PhaseKindOutOfRange {
        arm: usize,
        kind: usize,
        phase_kinds: usize,
    },
    #[error("scheduled grouped phase composition: lifecycle group {group} is not used by any schedule arm")]
    UnusedLifecycleGroup { group: usize },
    #[error("scheduled grouped phase composition: phase kind {kind} is not used by any schedule arm")]
    UnusedPhaseKind { kind: usize },
    #[error("scheduled grouped phase composition: private link contract for phase kind {kind} occurs more than once")]
    DuplicateLinkKind { kind: usize },
    #[error(
        "scheduled grouped phase composition: private link contract names lifecycle group {group}, but only {groups} groups exist"
    )]
    LinkLifecycleGroupOutOfRange { group: usize, groups: usize },
    #[error(
        "scheduled grouped phase composition: private link contract names phase kind {kind}, but only {phase_kinds} kinds exist"
    )]
    LinkPhaseKindOutOfRange { kind: usize, phase_kinds: usize },
    #[error(
        "scheduled grouped phase composition: schedule arm {arm} pairs linked phase kind {phase_kind} with lifecycle group {actual_group}, expected {expected_group}"
    )]
    LinkLifecycleGroupMismatch {
        arm: usize,
        phase_kind: usize,
        actual_group: usize,
        expected_group: usize,
    },
    #[error(
        "scheduled grouped phase composition: {owner} branch {branch} source field {field} has no retained low-norm slot"
    )]
    MissingFieldSlot {
        owner: &'static str,
        branch: usize,
        field: usize,
    },
    #[error(
        "scheduled grouped phase composition: {owner} branch {branch} source field {field} has unsupported encoded width {width}"
    )]
    UnsupportedFieldWidth {
        owner: &'static str,
        branch: usize,
        field: usize,
        width: usize,
    },
    #[error(
        "scheduled grouped phase composition: {owner} branch {branch} source field {field} cannot be reconstructed: {reason}"
    )]
    SourceFieldDecoder {
        owner: &'static str,
        branch: usize,
        field: usize,
        reason: String,
    },
    #[error(
        "scheduled grouped phase composition: {owner} cursor bit range [{start}, {end}) must have width 1..=64 inside public width {public}"
    )]
    CursorRange {
        owner: &'static str,
        start: usize,
        end: usize,
        public: usize,
    },
    #[error("scheduled grouped phase composition: before and after cursor bit ranges overlap")]
    CursorRangesOverlap,
    #[error("scheduled grouped phase composition: {owner} cursor width {width} cannot encode required value {value}")]
    CursorValueOutOfRange {
        owner: &'static str,
        width: usize,
        value: usize,
    },
    #[error("scheduled grouped phase encoding: schedule arm {arm} is outside 0..{arms}")]
    ScheduleArmOutOfRange { arm: usize, arms: usize },
}

/// Exact placement of one schedule that shares lifecycle and phase-kind rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledGroupedPhaseLayout {
    public_columns: Range<usize>,
    common_private_columns: Range<usize>,
    phase_private_columns: Range<usize>,
    schedule_selector_columns: Vec<usize>,
    ring_padding_columns: Range<usize>,
    common_rows: Range<usize>,
    phase_rows: Range<usize>,
    schedule_total_rows: Range<usize>,
    lifecycle_equality_rows: Range<usize>,
    phase_kind_equality_rows: Range<usize>,
    lifecycle_activation_rows: Range<usize>,
    phase_kind_activation_rows: Range<usize>,
    cursor_binding_rows: Range<usize>,
    common_phase_link_rows: Range<usize>,
    ring_padding_rows: Range<usize>,
    common_selector_columns: Vec<usize>,
    phase_kind_selector_columns: Vec<usize>,
    lifecycle_groups: Vec<usize>,
    phase_kinds: Vec<usize>,
    cursor_bits: ScheduledCursorBits,
    link_row_offsets: Vec<Range<usize>>,
    phase_kind_links: Vec<Option<ScheduledPhaseKindLinks>>,
}

impl ScheduledGroupedPhaseLayout {
    pub fn public_columns(&self) -> Range<usize> {
        self.public_columns.clone()
    }

    pub fn common_private_columns(&self) -> Range<usize> {
        self.common_private_columns.clone()
    }

    pub fn phase_private_columns(&self) -> Range<usize> {
        self.phase_private_columns.clone()
    }

    pub fn schedule_selector_columns(&self) -> &[usize] {
        &self.schedule_selector_columns
    }

    pub fn ring_padding_columns(&self) -> Range<usize> {
        self.ring_padding_columns.clone()
    }

    pub fn common_rows(&self) -> Range<usize> {
        self.common_rows.clone()
    }

    pub fn phase_rows(&self) -> Range<usize> {
        self.phase_rows.clone()
    }

    pub fn schedule_total_rows(&self) -> Range<usize> {
        self.schedule_total_rows.clone()
    }

    pub fn lifecycle_equality_rows(&self) -> Range<usize> {
        self.lifecycle_equality_rows.clone()
    }

    pub fn phase_kind_equality_rows(&self) -> Range<usize> {
        self.phase_kind_equality_rows.clone()
    }

    pub fn lifecycle_activation_rows(&self) -> Range<usize> {
        self.lifecycle_activation_rows.clone()
    }

    pub fn phase_kind_activation_rows(&self) -> Range<usize> {
        self.phase_kind_activation_rows.clone()
    }

    /// Two rows per schedule arm: exact before cursor, then exact after cursor.
    pub fn cursor_binding_rows(&self) -> Range<usize> {
        self.cursor_binding_rows.clone()
    }

    /// Selector-gated equalities between lifecycle and phase source fields.
    pub fn common_phase_link_rows(&self) -> Range<usize> {
        self.common_phase_link_rows.clone()
    }

    /// Exact private-link rows for one phase kind.
    pub fn common_phase_link_rows_for_kind(&self, kind: usize) -> Option<Range<usize>> {
        self.link_row_offsets.get(kind).cloned()
    }

    /// Exact source-field contract used to emit one phase kind's private
    /// lifecycle links. `None` means that the kind owns no such links.
    pub fn common_phase_links_for_kind(&self, kind: usize) -> Option<&ScheduledPhaseKindLinks> {
        self.phase_kind_links.get(kind)?.as_ref()
    }

    pub fn ring_padding_rows(&self) -> Range<usize> {
        self.ring_padding_rows.clone()
    }

    pub fn common_selector_columns(&self) -> &[usize] {
        &self.common_selector_columns
    }

    pub fn phase_kind_selector_columns(&self) -> &[usize] {
        &self.phase_kind_selector_columns
    }

    pub fn lifecycle_groups(&self) -> &[usize] {
        &self.lifecycle_groups
    }

    pub fn phase_kinds(&self) -> &[usize] {
        &self.phase_kinds
    }

    pub fn cursor_bits(&self) -> &ScheduledCursorBits {
        &self.cursor_bits
    }

    pub fn columns(&self) -> usize {
        self.ring_padding_columns.end
    }

    pub fn rows(&self) -> usize {
        self.ring_padding_rows.end
    }
}

/// Joint relation with one copy of each lifecycle group, one copy of each
/// phase kind, and one small authority binding per exact schedule arm.
#[derive(Debug)]
pub struct ScheduledGroupedPhaseLowNormR1cs {
    structure: Structure,
    common: MultiBranchLowNormR1cs,
    phase_kinds: MultiBranchLowNormR1cs,
    layout: ScheduledGroupedPhaseLayout,
}

impl ScheduledGroupedPhaseLowNormR1cs {
    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    pub fn layout(&self) -> &ScheduledGroupedPhaseLayout {
        &self.layout
    }

    pub fn public_input_len(&self) -> usize {
        self.layout.public_columns.end
    }

    pub fn common_relation(&self) -> &MultiBranchLowNormR1cs {
        &self.common
    }

    pub fn phase_kind_relation(&self) -> &MultiBranchLowNormR1cs {
        &self.phase_kinds
    }

    /// Final low-norm slot for one lifecycle source field.
    ///
    /// The scheduled composer embeds the complete common relation at column
    /// zero. This method keeps that placement rule inside the composer instead
    /// of making callers repeat its offset arithmetic.
    pub fn common_field_slot(&self, lifecycle_group: usize, source_field: usize) -> Option<(usize, usize)> {
        let (start, width) = self.common.field_slot(lifecycle_group, source_field)?;
        let end = start.checked_add(width)?;
        (end <= self.layout.phase_private_columns.start).then_some((start, width))
    }

    pub fn encode(
        &self,
        arm: usize,
        common_field_assignment: &[F],
        phase_field_assignment: &[F],
    ) -> Result<Vec<F>, ScheduledGroupedPhaseError> {
        let lifecycle =
            *self
                .layout
                .lifecycle_groups
                .get(arm)
                .ok_or(ScheduledGroupedPhaseError::ScheduleArmOutOfRange {
                    arm,
                    arms: self.layout.lifecycle_groups.len(),
                })?;
        let phase_kind = self.layout.phase_kinds[arm];
        let common = self
            .common
            .encode(lifecycle, common_field_assignment)
            .map_err(GroupedPhaseError::LowNorm)?;
        let phase = self
            .phase_kinds
            .encode(phase_kind, phase_field_assignment)
            .map_err(GroupedPhaseError::LowNorm)?;
        let public = self.public_input_len();
        for coordinate in 0..public {
            if common[coordinate] != phase[coordinate] {
                return Err(GroupedPhaseError::PublicAssignmentMismatch { coordinate }.into());
            }
        }

        let mut assignment = vec![F::ZERO; self.structure.m];
        assignment[..common.len()].copy_from_slice(&common);
        let phase_private = self.layout.phase_private_columns.start;
        assignment[phase_private..phase_private + phase.len() - public].copy_from_slice(&phase[public..]);
        assignment[self.layout.schedule_selector_columns[arm]] = F::ONE;
        Ok(assignment)
    }

    pub fn is_satisfied(&self, assignment: &[F]) -> bool {
        self.first_unsatisfied_row(assignment).is_none()
    }

    pub fn first_unsatisfied_row(&self, assignment: &[F]) -> Option<usize> {
        if assignment.len() != self.structure.m {
            return Some(0);
        }
        let mut images = vec![vec![F::ZERO; self.structure.n]; self.structure.matrices.len()];
        for (matrix, image) in self.structure.matrices.iter().zip(&mut images) {
            matrix.add_mul_into(assignment, image, self.structure.n);
        }
        (0..self.structure.n).find(|&row| {
            let point = images.iter().map(|image| image[row]).collect::<Vec<_>>();
            self.structure.f.eval(&point) != F::ZERO
        })
    }
}

/// One joint CCS relation with common verifier rows and phase rows each stored
/// once. The source encoders are retained so one honest joint assignment can
/// be produced without a second layout implementation.
#[derive(Debug)]
pub struct GroupedPhaseLowNormR1cs {
    structure: Structure,
    common: MultiBranchLowNormR1cs,
    phases: MultiBranchLowNormR1cs,
    layout: GroupedPhaseLayout,
}

impl GroupedPhaseLowNormR1cs {
    /// Joint CCS structure.
    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    /// Exact joint row and column placement.
    pub fn layout(&self) -> &GroupedPhaseLayout {
        &self.layout
    }

    /// Shared encoded public-input length.
    pub fn public_input_len(&self) -> usize {
        self.layout.public_columns.end
    }

    /// Lifecycle-common source relation.
    pub fn common_relation(&self) -> &MultiBranchLowNormR1cs {
        &self.common
    }

    /// Phase-local source relation.
    pub fn phase_relation(&self) -> &MultiBranchLowNormR1cs {
        &self.phases
    }

    /// Encode one phase and its fixed lifecycle group into the joint arena.
    pub fn encode(
        &self,
        phase: usize,
        common_field_assignment: &[F],
        phase_field_assignment: &[F],
    ) -> Result<Vec<F>, GroupedPhaseError> {
        let group = *self
            .layout
            .phase_groups
            .get(phase)
            .ok_or(GroupedPhaseError::PhaseOutOfRange {
                phase,
                phases: self.layout.phase_groups.len(),
            })?;
        let common = self.common.encode(group, common_field_assignment)?;
        let local = self.phases.encode(phase, phase_field_assignment)?;
        let public = self.public_input_len();
        for coordinate in 0..public {
            if common[coordinate] != local[coordinate] {
                return Err(GroupedPhaseError::PublicAssignmentMismatch { coordinate });
            }
        }

        let mut assignment = vec![F::ZERO; self.structure.m];
        assignment[..common.len()].copy_from_slice(&common);
        let phase_private = self.layout.phase_private_columns.start;
        assignment[phase_private..phase_private + local.len() - public].copy_from_slice(&local[public..]);
        Ok(assignment)
    }

    /// Return whether the complete joint relation accepts an assignment.
    pub fn is_satisfied(&self, assignment: &[F]) -> bool {
        self.first_unsatisfied_row(assignment).is_none()
    }

    /// First rejected joint row, or `None` when every row accepts.
    pub fn first_unsatisfied_row(&self, assignment: &[F]) -> Option<usize> {
        if assignment.len() != self.structure.m {
            return Some(0);
        }
        let mut images = vec![vec![F::ZERO; self.structure.n]; self.structure.matrices.len()];
        for (matrix, image) in self.structure.matrices.iter().zip(&mut images) {
            matrix.add_mul_into(assignment, image, self.structure.n);
        }
        (0..self.structure.n).find(|&row| {
            let point = images.iter().map(|image| image[row]).collect::<Vec<_>>();
            self.structure.f.eval(&point) != F::ZERO
        })
    }
}

/// Compose a lifecycle-common selective relation and a phase-local selective
/// relation over one exact shared public prefix.
pub fn build_grouped_phase_low_norm_r1cs(
    common: MultiBranchLowNormR1cs,
    phases: MultiBranchLowNormR1cs,
    phase_groups: Vec<usize>,
) -> Result<GroupedPhaseLowNormR1cs, GroupedPhaseError> {
    validate_component("common", &common)?;
    validate_component("phase", &phases)?;

    let public = common.public_input_len();
    if public != phases.public_input_len() {
        return Err(GroupedPhaseError::PublicWidthMismatch {
            common: public,
            phase: phases.public_input_len(),
        });
    }
    for (owner, relation) in [("common", &common), ("phase", &phases)] {
        if public > relation.structure().m {
            return Err(GroupedPhaseError::PublicWidthOutOfBounds {
                owner,
                public,
                columns: relation.structure().m,
            });
        }
    }

    let phase_count = phases.selector_cols().len();
    let group_count = common.selector_cols().len();
    if phase_groups.len() != phase_count {
        return Err(GroupedPhaseError::PhaseGroupCount {
            actual: phase_groups.len(),
            expected: phase_count,
        });
    }
    if let Some((phase, &group)) = phase_groups
        .iter()
        .enumerate()
        .find(|(_, group)| **group >= group_count)
    {
        return Err(GroupedPhaseError::PhaseGroupOutOfRange {
            phase,
            group,
            groups: group_count,
        });
    }

    let common_columns = common.structure().m;
    let phase_columns = phases.structure().m;
    let unpadded_columns = common_columns + phase_columns - public;
    let columns = unpadded_columns.next_multiple_of(D);
    let common_rows = 0..common.structure().n;
    let phase_rows = common_rows.end..common_rows.end + phases.structure().n;
    let group_equality_rows = phase_rows.end..phase_rows.end + group_count;
    let phase_activation_rows = group_equality_rows.end..group_equality_rows.end + phase_count;
    let ring_padding_rows = phase_activation_rows.end..phase_activation_rows.end + columns - unpadded_columns;

    let common_embedding = ColumnEmbedding {
        public,
        private_start: public,
    };
    let phase_embedding = ColumnEmbedding {
        public,
        private_start: common_columns,
    };
    let common_selector_columns = common
        .selector_cols()
        .iter()
        .map(|&column| common_embedding.map(column))
        .collect::<Vec<_>>();
    let phase_selector_columns = phases
        .selector_cols()
        .iter()
        .map(|&column| phase_embedding.map(column))
        .collect::<Vec<_>>();

    let rows = ring_padding_rows.end;
    let mut explicit = (0..SELECTIVE_ARITY).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut blocks = (0..SELECTIVE_ARITY).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut geometric = (0..SELECTIVE_ARITY).map(|_| Vec::new()).collect::<Vec<_>>();
    for matrix in 0..SELECTIVE_ARITY {
        append_embedded_matrix(
            "common",
            matrix,
            &common.structure().matrices[matrix],
            common_rows.start,
            common_embedding,
            &mut explicit[matrix],
            &mut blocks[matrix],
            &mut geometric[matrix],
        )?;
        append_embedded_matrix(
            "phase",
            matrix,
            &phases.structure().matrices[matrix],
            phase_rows.start,
            phase_embedding,
            &mut explicit[matrix],
            &mut blocks[matrix],
            &mut geometric[matrix],
        )?;
    }

    for group in 0..group_count {
        let row = group_equality_rows.start + group;
        explicit[GENERAL_SELECTOR].push((row, 0, F::ONE));
        explicit[C].push((row, common_selector_columns[group], F::ONE));
        for (phase, &phase_group) in phase_groups.iter().enumerate() {
            if phase_group == group {
                explicit[C].push((row, phase_selector_columns[phase], -F::ONE));
            }
        }
    }
    for phase in 0..phase_count {
        let row = phase_activation_rows.start + phase;
        let phase_selector = phase_selector_columns[phase];
        let group_selector = common_selector_columns[phase_groups[phase]];
        explicit[GENERAL_SELECTOR].push((row, 0, F::ONE));
        explicit[A].push((row, phase_selector, F::ONE));
        explicit[B].push((row, group_selector, F::ONE));
        explicit[C].push((row, phase_selector, F::ONE));
    }
    for (row, column) in ring_padding_rows.clone().zip(unpadded_columns..columns) {
        explicit[GENERAL_SELECTOR].push((row, 0, F::ONE));
        explicit[C].push((row, column, F::ONE));
    }

    let mut matrices = Vec::with_capacity(SELECTIVE_ARITY);
    for matrix in 0..SELECTIVE_ARITY {
        let csc = CscMat::from_counted_triplets(core::mem::take(&mut explicit[matrix]), rows, columns);
        matrices.push(
            CcsMatrix::csc_with_compact_rows(
                csc,
                core::mem::take(&mut blocks[matrix]),
                core::mem::take(&mut geometric[matrix]),
            )
            .map_err(GroupedPhaseError::CompactMatrix)?,
        );
    }
    let structure = CcsStructure::new_sparse(matrices, common.structure().f.clone())
        .map_err(|error| GroupedPhaseError::Structure(error.to_string()))?;
    let layout = GroupedPhaseLayout {
        public_columns: 0..public,
        common_private_columns: public..common_columns,
        phase_private_columns: common_columns..unpadded_columns,
        ring_padding_columns: unpadded_columns..columns,
        common_rows,
        phase_rows,
        group_equality_rows,
        phase_activation_rows,
        ring_padding_rows,
        common_selector_columns,
        phase_selector_columns,
        phase_groups,
    };
    Ok(GroupedPhaseLowNormR1cs {
        structure,
        common,
        phases,
        layout,
    })
}

/// Compose an exact work schedule while storing each lifecycle group and each
/// phase kind only once.
///
/// `lifecycle_groups[arm]` and `phase_kinds[arm]` are verifier constants. The
/// two cursor rows for `arm` enforce `before = arm` and `after = arm + 1`
/// whenever that schedule selector is nonzero.
pub fn build_scheduled_grouped_phase_low_norm_r1cs(
    common: MultiBranchLowNormR1cs,
    phase_kinds: MultiBranchLowNormR1cs,
    lifecycle_groups: Vec<usize>,
    phase_kind_map: Vec<usize>,
    cursor_bits: ScheduledCursorBits,
) -> Result<ScheduledGroupedPhaseLowNormR1cs, ScheduledGroupedPhaseError> {
    build_scheduled_grouped_phase_low_norm_r1cs_with_field_links(
        common,
        phase_kinds,
        lifecycle_groups,
        phase_kind_map,
        cursor_bits,
        Vec::new(),
    )
}

/// Compose an exact work schedule and add verifier-owned private source-field
/// equalities. One contract is stored per linked phase kind. The constructor
/// rejects a schedule that pairs that kind with a different lifecycle group.
#[allow(clippy::too_many_arguments)]
pub fn build_scheduled_grouped_phase_low_norm_r1cs_with_field_links(
    common: MultiBranchLowNormR1cs,
    phase_kinds: MultiBranchLowNormR1cs,
    lifecycle_groups: Vec<usize>,
    phase_kind_map: Vec<usize>,
    cursor_bits: ScheduledCursorBits,
    field_links: Vec<ScheduledPhaseKindLinks>,
) -> Result<ScheduledGroupedPhaseLowNormR1cs, ScheduledGroupedPhaseError> {
    validate_component("common", &common)?;
    validate_component("phase-kind", &phase_kinds)?;
    if lifecycle_groups.is_empty() {
        return Err(ScheduledGroupedPhaseError::EmptySchedule);
    }
    if lifecycle_groups.len() != phase_kind_map.len() {
        return Err(ScheduledGroupedPhaseError::ScheduleLengthMismatch {
            lifecycle: lifecycle_groups.len(),
            phase_kinds: phase_kind_map.len(),
        });
    }

    let public = common.public_input_len();
    if public != phase_kinds.public_input_len() {
        return Err(GroupedPhaseError::PublicWidthMismatch {
            common: public,
            phase: phase_kinds.public_input_len(),
        }
        .into());
    }
    for (owner, relation) in [("common", &common), ("phase-kind", &phase_kinds)] {
        if public > relation.structure().m {
            return Err(GroupedPhaseError::PublicWidthOutOfBounds {
                owner,
                public,
                columns: relation.structure().m,
            }
            .into());
        }
    }

    let lifecycle_count = common.selector_cols().len();
    let phase_kind_count = phase_kinds.selector_cols().len();
    if let Some((arm, &group)) = lifecycle_groups
        .iter()
        .enumerate()
        .find(|(_, group)| **group >= lifecycle_count)
    {
        return Err(ScheduledGroupedPhaseError::LifecycleGroupOutOfRange {
            arm,
            group,
            groups: lifecycle_count,
        });
    }
    if let Some((arm, &kind)) = phase_kind_map
        .iter()
        .enumerate()
        .find(|(_, kind)| **kind >= phase_kind_count)
    {
        return Err(ScheduledGroupedPhaseError::PhaseKindOutOfRange {
            arm,
            kind,
            phase_kinds: phase_kind_count,
        });
    }
    if let Some(group) = (0..lifecycle_count).find(|group| !lifecycle_groups.contains(group)) {
        return Err(ScheduledGroupedPhaseError::UnusedLifecycleGroup { group });
    }
    if let Some(kind) = (0..phase_kind_count).find(|kind| !phase_kind_map.contains(kind)) {
        return Err(ScheduledGroupedPhaseError::UnusedPhaseKind { kind });
    }
    let mut links_by_kind = vec![None; phase_kind_count];
    for contract in field_links {
        if contract.lifecycle_group >= lifecycle_count {
            return Err(ScheduledGroupedPhaseError::LinkLifecycleGroupOutOfRange {
                group: contract.lifecycle_group,
                groups: lifecycle_count,
            });
        }
        if contract.phase_kind >= phase_kind_count {
            return Err(ScheduledGroupedPhaseError::LinkPhaseKindOutOfRange {
                kind: contract.phase_kind,
                phase_kinds: phase_kind_count,
            });
        }
        let kind = contract.phase_kind;
        if links_by_kind[kind].replace(contract).is_some() {
            return Err(ScheduledGroupedPhaseError::DuplicateLinkKind { kind });
        }
    }
    for (arm, (&actual_group, &phase_kind)) in lifecycle_groups.iter().zip(&phase_kind_map).enumerate() {
        if let Some(contract) = &links_by_kind[phase_kind] {
            if actual_group != contract.lifecycle_group {
                return Err(ScheduledGroupedPhaseError::LinkLifecycleGroupMismatch {
                    arm,
                    phase_kind,
                    actual_group,
                    expected_group: contract.lifecycle_group,
                });
            }
        }
    }
    validate_cursor_range("before", &cursor_bits.before, public)?;
    validate_cursor_range("after", &cursor_bits.after, public)?;
    if cursor_bits.before.start < cursor_bits.after.end && cursor_bits.after.start < cursor_bits.before.end {
        return Err(ScheduledGroupedPhaseError::CursorRangesOverlap);
    }
    let final_arm = lifecycle_groups.len() - 1;
    validate_cursor_value("before", cursor_bits.before.len(), final_arm)?;
    validate_cursor_value("after", cursor_bits.after.len(), lifecycle_groups.len())?;

    let common_columns = common.structure().m;
    let phase_columns = phase_kinds.structure().m;
    let component_columns = common_columns + phase_columns - public;
    let schedule_selector_columns = (component_columns..component_columns + lifecycle_groups.len()).collect::<Vec<_>>();
    let unpadded_columns = component_columns + schedule_selector_columns.len();
    let columns = unpadded_columns.next_multiple_of(D);

    let common_rows = 0..common.structure().n;
    let phase_rows = common_rows.end..common_rows.end + phase_kinds.structure().n;
    let schedule_total_rows = phase_rows.end..phase_rows.end + 1;
    let lifecycle_equality_rows = schedule_total_rows.end..schedule_total_rows.end + lifecycle_count;
    let phase_kind_equality_rows = lifecycle_equality_rows.end..lifecycle_equality_rows.end + phase_kind_count;
    let lifecycle_activation_rows = phase_kind_equality_rows.end..phase_kind_equality_rows.end + lifecycle_groups.len();
    let phase_kind_activation_rows =
        lifecycle_activation_rows.end..lifecycle_activation_rows.end + lifecycle_groups.len();
    let cursor_binding_rows =
        phase_kind_activation_rows.end..phase_kind_activation_rows.end + 2 * lifecycle_groups.len();
    let common_phase_link_count = links_by_kind
        .iter()
        .flatten()
        .map(|contract| contract.fields.len())
        .sum::<usize>();
    let common_phase_link_rows = cursor_binding_rows.end..cursor_binding_rows.end + common_phase_link_count;
    let ring_padding_rows = common_phase_link_rows.end..common_phase_link_rows.end + columns - unpadded_columns;

    let common_embedding = ColumnEmbedding {
        public,
        private_start: public,
    };
    let phase_embedding = ColumnEmbedding {
        public,
        private_start: common_columns,
    };
    let common_selector_columns = common
        .selector_cols()
        .iter()
        .map(|&column| common_embedding.map(column))
        .collect::<Vec<_>>();
    let phase_kind_selector_columns = phase_kinds
        .selector_cols()
        .iter()
        .map(|&column| phase_embedding.map(column))
        .collect::<Vec<_>>();

    let rows = ring_padding_rows.end;
    let mut explicit = (0..SELECTIVE_ARITY).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut blocks = (0..SELECTIVE_ARITY).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut geometric = (0..SELECTIVE_ARITY).map(|_| Vec::new()).collect::<Vec<_>>();
    for matrix in 0..SELECTIVE_ARITY {
        append_embedded_matrix(
            "common",
            matrix,
            &common.structure().matrices[matrix],
            common_rows.start,
            common_embedding,
            &mut explicit[matrix],
            &mut blocks[matrix],
            &mut geometric[matrix],
        )?;
        append_embedded_matrix(
            "phase-kind",
            matrix,
            &phase_kinds.structure().matrices[matrix],
            phase_rows.start,
            phase_embedding,
            &mut explicit[matrix],
            &mut blocks[matrix],
            &mut geometric[matrix],
        )?;
    }

    let total_row = schedule_total_rows.start;
    explicit[GENERAL_SELECTOR].push((total_row, 0, F::ONE));
    explicit[C].push((total_row, 0, F::ONE));
    for &selector in &schedule_selector_columns {
        explicit[C].push((total_row, selector, -F::ONE));
    }
    for group in 0..lifecycle_count {
        let row = lifecycle_equality_rows.start + group;
        explicit[GENERAL_SELECTOR].push((row, 0, F::ONE));
        explicit[C].push((row, common_selector_columns[group], F::ONE));
        for (arm, &arm_group) in lifecycle_groups.iter().enumerate() {
            if arm_group == group {
                explicit[C].push((row, schedule_selector_columns[arm], -F::ONE));
            }
        }
    }
    for kind in 0..phase_kind_count {
        let row = phase_kind_equality_rows.start + kind;
        explicit[GENERAL_SELECTOR].push((row, 0, F::ONE));
        explicit[C].push((row, phase_kind_selector_columns[kind], F::ONE));
        for (arm, &arm_kind) in phase_kind_map.iter().enumerate() {
            if arm_kind == kind {
                explicit[C].push((row, schedule_selector_columns[arm], -F::ONE));
            }
        }
    }
    for arm in 0..lifecycle_groups.len() {
        let schedule_selector = schedule_selector_columns[arm];
        let lifecycle_selector = common_selector_columns[lifecycle_groups[arm]];
        let phase_kind_selector = phase_kind_selector_columns[phase_kind_map[arm]];

        let lifecycle_row = lifecycle_activation_rows.start + arm;
        explicit[GENERAL_SELECTOR].push((lifecycle_row, 0, F::ONE));
        explicit[A].push((lifecycle_row, schedule_selector, F::ONE));
        explicit[B].push((lifecycle_row, lifecycle_selector, F::ONE));
        explicit[C].push((lifecycle_row, schedule_selector, F::ONE));

        let phase_kind_row = phase_kind_activation_rows.start + arm;
        explicit[GENERAL_SELECTOR].push((phase_kind_row, 0, F::ONE));
        explicit[A].push((phase_kind_row, schedule_selector, F::ONE));
        explicit[B].push((phase_kind_row, phase_kind_selector, F::ONE));
        explicit[C].push((phase_kind_row, schedule_selector, F::ONE));

        append_cursor_binding_row(
            &mut explicit,
            cursor_binding_rows.start + 2 * arm,
            schedule_selector,
            cursor_bits.before.clone(),
            arm,
        );
        append_cursor_binding_row(
            &mut explicit,
            cursor_binding_rows.start + 2 * arm + 1,
            schedule_selector,
            cursor_bits.after.clone(),
            arm + 1,
        );
    }
    let mut link_cursor = common_phase_link_rows.start;
    let mut link_row_offsets = vec![link_cursor..link_cursor; phase_kind_count];
    for phase_kind in 0..phase_kind_count {
        let start = link_cursor;
        if let Some(contract) = &links_by_kind[phase_kind] {
            for link in &contract.fields {
                let row = link_cursor;
                link_cursor += 1;
                explicit[GENERAL_SELECTOR].push((row, 0, F::ONE));
                explicit[A].push((row, phase_kind_selector_columns[phase_kind], F::ONE));
                append_scheduled_source_field_terms(
                    &mut explicit[B],
                    row,
                    "common",
                    contract.lifecycle_group,
                    link.common_field,
                    &common,
                    common_embedding,
                    F::ONE,
                )?;
                append_scheduled_source_field_terms(
                    &mut explicit[B],
                    row,
                    "phase-kind",
                    phase_kind,
                    link.phase_field,
                    &phase_kinds,
                    phase_embedding,
                    -F::ONE,
                )?;
            }
        }
        link_row_offsets[phase_kind] = start..link_cursor;
    }
    debug_assert_eq!(link_cursor, common_phase_link_rows.end);
    for (row, column) in ring_padding_rows.clone().zip(unpadded_columns..columns) {
        explicit[GENERAL_SELECTOR].push((row, 0, F::ONE));
        explicit[C].push((row, column, F::ONE));
    }

    let mut matrices = Vec::with_capacity(SELECTIVE_ARITY);
    for matrix in 0..SELECTIVE_ARITY {
        let csc = CscMat::from_counted_triplets(core::mem::take(&mut explicit[matrix]), rows, columns);
        matrices.push(
            CcsMatrix::csc_with_compact_rows(
                csc,
                core::mem::take(&mut blocks[matrix]),
                core::mem::take(&mut geometric[matrix]),
            )
            .map_err(GroupedPhaseError::CompactMatrix)?,
        );
    }
    let structure = CcsStructure::new_sparse(matrices, common.structure().f.clone())
        .map_err(|error| GroupedPhaseError::Structure(error.to_string()))?;
    let layout = ScheduledGroupedPhaseLayout {
        public_columns: 0..public,
        common_private_columns: public..common_columns,
        phase_private_columns: common_columns..component_columns,
        schedule_selector_columns,
        ring_padding_columns: unpadded_columns..columns,
        common_rows,
        phase_rows,
        schedule_total_rows,
        lifecycle_equality_rows,
        phase_kind_equality_rows,
        lifecycle_activation_rows,
        phase_kind_activation_rows,
        cursor_binding_rows,
        common_phase_link_rows,
        ring_padding_rows,
        common_selector_columns,
        phase_kind_selector_columns,
        lifecycle_groups,
        phase_kinds: phase_kind_map,
        cursor_bits,
        link_row_offsets,
        phase_kind_links: links_by_kind,
    };
    Ok(ScheduledGroupedPhaseLowNormR1cs {
        structure,
        common,
        phase_kinds,
        layout,
    })
}

fn validate_cursor_range(
    owner: &'static str,
    range: &Range<usize>,
    public: usize,
) -> Result<(), ScheduledGroupedPhaseError> {
    if range.is_empty() || range.len() > 64 || range.end > public {
        return Err(ScheduledGroupedPhaseError::CursorRange {
            owner,
            start: range.start,
            end: range.end,
            public,
        });
    }
    Ok(())
}

fn validate_cursor_value(owner: &'static str, width: usize, value: usize) -> Result<(), ScheduledGroupedPhaseError> {
    if (value as u128) >= (1u128 << width) {
        return Err(ScheduledGroupedPhaseError::CursorValueOutOfRange { owner, width, value });
    }
    Ok(())
}

fn append_cursor_binding_row(
    matrices: &mut [Vec<(usize, usize, F)>],
    row: usize,
    selector: usize,
    bits: Range<usize>,
    expected: usize,
) {
    matrices[GENERAL_SELECTOR].push((row, 0, F::ONE));
    matrices[A].push((row, selector, F::ONE));
    if expected != 0 {
        matrices[B].push((row, 0, -F::from_usize(expected)));
    }
    let mut coefficient = F::ONE;
    for column in bits {
        matrices[B].push((row, column, coefficient));
        coefficient += coefficient;
    }
}

#[allow(clippy::too_many_arguments)]
fn append_scheduled_source_field_terms(
    terms: &mut Vec<(usize, usize, F)>,
    row: usize,
    owner: &'static str,
    branch: usize,
    field: usize,
    relation: &MultiBranchLowNormR1cs,
    embedding: ColumnEmbedding,
    scale: F,
) -> Result<(), ScheduledGroupedPhaseError> {
    let decoded = decoded_source_field_terms(relation, branch, field).map_err(|reason| {
        ScheduledGroupedPhaseError::SourceFieldDecoder {
            owner,
            branch,
            field,
            reason,
        }
    })?;
    for (coordinate, coefficient) in decoded {
        terms.push((row, embedding.map(coordinate), scale * coefficient));
    }
    Ok(())
}

pub(super) fn decoded_source_field_terms(
    relation: &MultiBranchLowNormR1cs,
    branch: usize,
    field: usize,
) -> Result<Vec<(usize, F)>, String> {
    let mut terms = BTreeMap::<usize, F>::new();
    let mut visiting = BTreeSet::<usize>::new();
    expand_source_field(relation, branch, field, F::ONE, &mut visiting, &mut terms)?;
    terms.retain(|_, coefficient| *coefficient != F::ZERO);
    Ok(terms.into_iter().collect())
}

fn expand_source_field(
    relation: &MultiBranchLowNormR1cs,
    branch: usize,
    field: usize,
    scale: F,
    visiting: &mut BTreeSet<usize>,
    terms: &mut BTreeMap<usize, F>,
) -> Result<(), String> {
    if scale == F::ZERO {
        return Ok(());
    }
    if field == 0 {
        *terms.entry(0).or_insert(F::ZERO) += scale;
        return Ok(());
    }
    if let Some((start, width)) = relation.field_slot(branch, field) {
        let radix = match width {
            41 => F::from_u64(3),
            23 => F::from_u64(7),
            1..=64 => F::from_u64(2),
            _ => return Err(format!("retained slot has unsupported encoded width {width}")),
        };
        let mut coefficient = scale;
        for coordinate in start..start + width {
            *terms.entry(coordinate).or_insert(F::ZERO) += coefficient;
            coefficient *= radix;
        }
        return Ok(());
    }

    if !visiting.insert(field) {
        return Err(format!("affine definition cycle reaches source field {field}"));
    }
    let compiler = relation
        .selective_compiler_audit()
        .ok_or_else(|| "selective compiler audit is absent".to_owned())?;
    let definition = compiler
        .source_arm_linear_definition(branch, field)
        .ok_or_else(|| "field has neither a retained slot nor an affine definition".to_owned())?;
    *terms.entry(0).or_insert(F::ZERO) += scale * definition.constant();
    for term in definition.terms() {
        expand_source_field(
            relation,
            branch,
            term.column(),
            scale * term.coefficient(),
            visiting,
            terms,
        )?;
    }
    visiting.remove(&field);
    Ok(())
}

pub(super) fn validate_component(
    owner: &'static str,
    relation: &MultiBranchLowNormR1cs,
) -> Result<(), GroupedPhaseError> {
    let structure = relation.structure();
    if structure.matrices.len() != SELECTIVE_ARITY {
        return Err(GroupedPhaseError::MatrixCount {
            owner,
            actual: structure.matrices.len(),
        });
    }
    if !is_canonical_selective_low_norm_polynomial(&structure.f) {
        return Err(GroupedPhaseError::NonCanonicalPolynomial { owner });
    }
    for &selector in relation.selector_cols() {
        if selector >= structure.m {
            return Err(GroupedPhaseError::SelectorOutOfBounds {
                owner,
                selector,
                columns: structure.m,
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct ColumnEmbedding {
    pub(super) public: usize,
    pub(super) private_start: usize,
}

impl ColumnEmbedding {
    pub(super) fn map(self, column: usize) -> usize {
        if column < self.public {
            column
        } else {
            self.private_start + column - self.public
        }
    }

    fn map_contiguous(
        self,
        owner: &'static str,
        matrix: usize,
        start: usize,
        len: usize,
    ) -> Result<usize, GroupedPhaseError> {
        let mapped = self.map(start);
        if (0..len).any(|offset| self.map(start + offset) != mapped + offset) {
            return Err(GroupedPhaseError::NonContiguousCompactRange {
                owner,
                matrix,
                start,
                end: start + len,
            });
        }
        Ok(mapped)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append_embedded_matrix(
    owner: &'static str,
    matrix_index: usize,
    matrix: &CcsMatrix<F>,
    row_offset: usize,
    columns: ColumnEmbedding,
    explicit: &mut Vec<(usize, usize, F)>,
    blocks: &mut Vec<SeededPhi81LinearBlock>,
    geometric: &mut Vec<GeometricRowRun<F>>,
) -> Result<(), GroupedPhaseError> {
    let mut append_csc = |csc: &CscMat<F>| {
        for column in 0..csc.ncols {
            let target_column = columns.map(column);
            for entry in csc.column_range(column) {
                explicit.push((row_offset + csc.row_index(entry), target_column, csc.vals[entry]));
            }
        }
    };
    match matrix {
        CcsMatrix::Identity { n } => {
            for index in 0..*n {
                explicit.push((row_offset + index, columns.map(index), F::ONE));
            }
        }
        CcsMatrix::Csc(csc) => append_csc(csc),
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks: source_blocks,
            geometric_runs,
        } => {
            append_csc(csc);
            for block in source_blocks {
                let word_starts = block
                    .word_starts()
                    .iter()
                    .map(|&start| columns.map_contiguous(owner, matrix_index, start, block.word_width()))
                    .collect::<Result<Vec<_>, _>>()?;
                blocks.push(block.with_geometry(row_offset + block.row_start(), word_starts)?);
            }
            for run in geometric_runs {
                let column_start = columns.map_contiguous(owner, matrix_index, run.column_start(), run.len())?;
                geometric.push(GeometricRowRun::new(
                    row_offset + run.row(),
                    column_start,
                    run.len(),
                    *run.initial(),
                    *run.ratio(),
                ));
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            return Err(GroupedPhaseError::VerifierArtifactMatrix {
                owner,
                matrix: matrix_index,
            });
        }
    }
    Ok(())
}
