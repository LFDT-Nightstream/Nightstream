//! Schedule-bound composition of a phased relation and a private overlay.
//!
//! Owns the third selective component used for small schedule-specific work.
//! The existing scheduled relation keeps the lifecycle and phase bodies. This
//! module stores each overlay kind once, maps every schedule arm to one kind,
//! and adds exact private-field equality rows between the selected phase and
//! overlay assignments.
//!
//! Does not own component semantics, schedule meaning, or the meaning of a
//! linked source field.

use std::ops::Range;

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, GeometricRowRun, SeededPhi81LinearBlock};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use super::grouped_phase::{append_embedded_matrix, decoded_source_field_terms, validate_component, ColumnEmbedding};
use super::selective::{A, B, C, GENERAL_SELECTOR, SELECTIVE_ARITY};
use super::{
    build_scheduled_grouped_phase_low_norm_r1cs_with_field_links, GroupedPhaseError, LowNormR1csError,
    MultiBranchLowNormR1cs, ScheduledCursorBits, ScheduledGroupedPhaseError, ScheduledGroupedPhaseLowNormR1cs,
    ScheduledPhaseKindLinks,
};
use crate::paper::relations::Structure;

/// One equality between a source field in a phase kind and a source field in
/// an overlay kind. Columns use each field-R1CS arm's normalized numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlayFieldLink {
    pub phase_field: usize,
    pub overlay_field: usize,
}

/// Private links owned by one overlay kind. A linked kind can be paired with
/// only one phase kind in the verifier-owned schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlayKindLinks {
    pub overlay_kind: usize,
    pub phase_kind: usize,
    pub fields: Vec<OverlayFieldLink>,
}

/// Exact placement of one schedule-bound overlay composition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledLinkedOverlayLayout {
    public_columns: Range<usize>,
    scheduled_private_columns: Range<usize>,
    overlay_private_columns: Range<usize>,
    ring_padding_columns: Range<usize>,
    scheduled_rows: Range<usize>,
    overlay_rows: Range<usize>,
    overlay_kind_equality_rows: Range<usize>,
    overlay_activation_rows: Range<usize>,
    field_link_rows: Range<usize>,
    ring_padding_rows: Range<usize>,
    overlay_selector_columns: Vec<usize>,
    overlay_kinds: Vec<usize>,
    link_row_offsets: Vec<Range<usize>>,
}

impl ScheduledLinkedOverlayLayout {
    pub fn public_columns(&self) -> Range<usize> {
        self.public_columns.clone()
    }

    pub fn scheduled_private_columns(&self) -> Range<usize> {
        self.scheduled_private_columns.clone()
    }

    pub fn overlay_private_columns(&self) -> Range<usize> {
        self.overlay_private_columns.clone()
    }

    pub fn ring_padding_columns(&self) -> Range<usize> {
        self.ring_padding_columns.clone()
    }

    pub fn scheduled_rows(&self) -> Range<usize> {
        self.scheduled_rows.clone()
    }

    pub fn overlay_rows(&self) -> Range<usize> {
        self.overlay_rows.clone()
    }

    pub fn overlay_kind_equality_rows(&self) -> Range<usize> {
        self.overlay_kind_equality_rows.clone()
    }

    pub fn overlay_activation_rows(&self) -> Range<usize> {
        self.overlay_activation_rows.clone()
    }

    pub fn field_link_rows(&self) -> Range<usize> {
        self.field_link_rows.clone()
    }

    pub fn field_link_rows_for_kind(&self, kind: usize) -> Option<Range<usize>> {
        self.link_row_offsets.get(kind).cloned()
    }

    pub fn ring_padding_rows(&self) -> Range<usize> {
        self.ring_padding_rows.clone()
    }

    pub fn overlay_selector_columns(&self) -> &[usize] {
        &self.overlay_selector_columns
    }

    pub fn overlay_kinds(&self) -> &[usize] {
        &self.overlay_kinds
    }

    pub fn columns(&self) -> usize {
        self.ring_padding_columns.end
    }

    pub fn rows(&self) -> usize {
        self.ring_padding_rows.end
    }
}

#[derive(Debug, Error)]
pub enum LinkedOverlayError {
    #[error(transparent)]
    Scheduled(#[from] ScheduledGroupedPhaseError),
    #[error(transparent)]
    Grouped(#[from] GroupedPhaseError),
    #[error(transparent)]
    LowNorm(#[from] LowNormR1csError),
    #[error("linked overlay: overlay public width is {actual}, expected one constant column")]
    OverlayPublicWidth { actual: usize },
    #[error("linked overlay: overlay map has {actual} entries, expected {expected}")]
    OverlayMapCount { actual: usize, expected: usize },
    #[error("linked overlay: schedule arm {arm} names overlay kind {kind}, but only {kinds} kinds exist")]
    OverlayKindOutOfRange {
        arm: usize,
        kind: usize,
        kinds: usize,
    },
    #[error("linked overlay: overlay kind {kind} is not used by the schedule")]
    UnusedOverlayKind { kind: usize },
    #[error("linked overlay: link contract for overlay kind {kind} occurs more than once")]
    DuplicateLinkKind { kind: usize },
    #[error("linked overlay: link contract names overlay kind {kind}, but only {kinds} kinds exist")]
    LinkKindOutOfRange { kind: usize, kinds: usize },
    #[error("linked overlay: link contract for overlay kind {overlay_kind} names phase kind {phase_kind}, but only {phase_kinds} kinds exist")]
    LinkPhaseKindOutOfRange {
        overlay_kind: usize,
        phase_kind: usize,
        phase_kinds: usize,
    },
    #[error("linked overlay: schedule arm {arm} pairs overlay kind {overlay_kind} with phase kind {actual_phase_kind}, expected {expected_phase_kind}")]
    LinkPhaseKindMismatch {
        arm: usize,
        overlay_kind: usize,
        actual_phase_kind: usize,
        expected_phase_kind: usize,
    },
    #[error("linked overlay: {owner} kind {kind} source field {field} has no retained low-norm slot")]
    MissingFieldSlot {
        owner: &'static str,
        kind: usize,
        field: usize,
    },
    #[error("linked overlay: {owner} kind {kind} source field {field} has unsupported encoded width {width}")]
    UnsupportedFieldWidth {
        owner: &'static str,
        kind: usize,
        field: usize,
        width: usize,
    },
    #[error("linked overlay: {owner} kind {kind} source field {field} cannot be reconstructed: {reason}")]
    SourceFieldDecoder {
        owner: &'static str,
        kind: usize,
        field: usize,
        reason: String,
    },
    #[error("linked overlay: compact matrix construction failed: {0}")]
    CompactMatrix(String),
    #[error("linked overlay: joint CCS construction failed: {0}")]
    Structure(String),
    #[error("linked overlay encoding: schedule arm {arm} is outside 0..{arms}")]
    ScheduleArmOutOfRange { arm: usize, arms: usize },
    #[error("linked overlay encoding: overlay constant column is not one")]
    OverlayConstant,
}

/// One joint relation that adds a small schedule-specific private overlay to
/// the existing lifecycle-plus-phase relation.
#[derive(Debug)]
pub struct ScheduledLinkedOverlayLowNormR1cs {
    structure: Structure,
    scheduled: ScheduledGroupedPhaseLowNormR1cs,
    overlays: MultiBranchLowNormR1cs,
    layout: ScheduledLinkedOverlayLayout,
}

impl ScheduledLinkedOverlayLowNormR1cs {
    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    pub fn layout(&self) -> &ScheduledLinkedOverlayLayout {
        &self.layout
    }

    pub fn scheduled_relation(&self) -> &ScheduledGroupedPhaseLowNormR1cs {
        &self.scheduled
    }

    pub fn overlay_relation(&self) -> &MultiBranchLowNormR1cs {
        &self.overlays
    }

    /// Final low-norm slot for one lifecycle source field.
    ///
    /// The linked overlay embeds the complete scheduled relation at column
    /// zero, so lifecycle slots remain unchanged in the final relation.
    pub fn common_field_slot(&self, lifecycle_group: usize, source_field: usize) -> Option<(usize, usize)> {
        let (start, width) = self
            .scheduled
            .common_field_slot(lifecycle_group, source_field)?;
        let end = start.checked_add(width)?;
        (end <= self.layout.overlay_private_columns.start).then_some((start, width))
    }

    /// Exact affine decoder for one lifecycle source field in final columns.
    ///
    /// A field can have a direct low-norm slot or can be removed by the
    /// selective compiler as a proved affine definition. The returned terms
    /// cover both cases and include column zero when the decoder has a
    /// constant term.
    pub fn common_field_decoding_terms(
        &self,
        lifecycle_group: usize,
        source_field: usize,
    ) -> Result<Vec<(usize, F)>, LinkedOverlayError> {
        let terms = decoded_source_field_terms(self.scheduled.common_relation(), lifecycle_group, source_field)
            .map_err(|reason| LinkedOverlayError::SourceFieldDecoder {
                owner: "common",
                kind: lifecycle_group,
                field: source_field,
                reason,
            })?;
        if terms
            .iter()
            .any(|&(column, _)| column >= self.layout.overlay_private_columns.start)
        {
            return Err(LinkedOverlayError::SourceFieldDecoder {
                owner: "common",
                kind: lifecycle_group,
                field: source_field,
                reason: "decoded coordinate escapes the scheduled-relation prefix".into(),
            });
        }
        Ok(terms)
    }

    pub fn public_input_len(&self) -> usize {
        self.layout.public_columns.end
    }

    pub fn encode(
        &self,
        arm: usize,
        common_field_assignment: &[F],
        phase_field_assignment: &[F],
        overlay_field_assignment: &[F],
    ) -> Result<Vec<F>, LinkedOverlayError> {
        let overlay_kind = *self
            .layout
            .overlay_kinds
            .get(arm)
            .ok_or(LinkedOverlayError::ScheduleArmOutOfRange {
                arm,
                arms: self.layout.overlay_kinds.len(),
            })?;
        let scheduled = self
            .scheduled
            .encode(arm, common_field_assignment, phase_field_assignment)?;
        let overlay = self
            .overlays
            .encode(overlay_kind, overlay_field_assignment)?;
        if overlay.first().copied() != Some(F::ONE) {
            return Err(LinkedOverlayError::OverlayConstant);
        }

        let mut assignment = vec![F::ZERO; self.structure.m];
        assignment[..scheduled.len()].copy_from_slice(&scheduled);
        let overlay_start = self.layout.overlay_private_columns.start;
        assignment[overlay_start..overlay_start + overlay.len() - 1].copy_from_slice(&overlay[1..]);
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

#[allow(clippy::too_many_arguments)]
pub fn build_scheduled_linked_overlay_low_norm_r1cs(
    common: MultiBranchLowNormR1cs,
    phase_kinds: MultiBranchLowNormR1cs,
    overlays: MultiBranchLowNormR1cs,
    lifecycle_groups: Vec<usize>,
    phase_kind_map: Vec<usize>,
    overlay_kind_map: Vec<usize>,
    cursor_bits: ScheduledCursorBits,
    links: Vec<OverlayKindLinks>,
) -> Result<ScheduledLinkedOverlayLowNormR1cs, LinkedOverlayError> {
    build_scheduled_linked_overlay_low_norm_r1cs_with_phase_field_links(
        common,
        phase_kinds,
        overlays,
        lifecycle_groups,
        phase_kind_map,
        overlay_kind_map,
        cursor_bits,
        Vec::new(),
        links,
    )
}

/// Compose a schedule-bound overlay and retain the exact private links between
/// the lifecycle-common and selected phase-kind source fields.
#[allow(clippy::too_many_arguments)]
pub fn build_scheduled_linked_overlay_low_norm_r1cs_with_phase_field_links(
    common: MultiBranchLowNormR1cs,
    phase_kinds: MultiBranchLowNormR1cs,
    overlays: MultiBranchLowNormR1cs,
    lifecycle_groups: Vec<usize>,
    phase_kind_map: Vec<usize>,
    overlay_kind_map: Vec<usize>,
    cursor_bits: ScheduledCursorBits,
    phase_field_links: Vec<ScheduledPhaseKindLinks>,
    links: Vec<OverlayKindLinks>,
) -> Result<ScheduledLinkedOverlayLowNormR1cs, LinkedOverlayError> {
    validate_component("overlay", &overlays)?;
    if overlays.public_input_len() != 1 {
        return Err(LinkedOverlayError::OverlayPublicWidth {
            actual: overlays.public_input_len(),
        });
    }
    let scheduled = build_scheduled_grouped_phase_low_norm_r1cs_with_field_links(
        common,
        phase_kinds,
        lifecycle_groups,
        phase_kind_map,
        cursor_bits,
        phase_field_links,
    )?;
    let arm_count = scheduled.layout().phase_kinds().len();
    if overlay_kind_map.len() != arm_count {
        return Err(LinkedOverlayError::OverlayMapCount {
            actual: overlay_kind_map.len(),
            expected: arm_count,
        });
    }
    let overlay_kind_count = overlays.selector_cols().len();
    if let Some((arm, &kind)) = overlay_kind_map
        .iter()
        .enumerate()
        .find(|(_, kind)| **kind >= overlay_kind_count)
    {
        return Err(LinkedOverlayError::OverlayKindOutOfRange {
            arm,
            kind,
            kinds: overlay_kind_count,
        });
    }
    if let Some(kind) = (0..overlay_kind_count).find(|kind| !overlay_kind_map.contains(kind)) {
        return Err(LinkedOverlayError::UnusedOverlayKind { kind });
    }

    let phase_kind_count = scheduled.phase_kind_relation().selector_cols().len();
    let mut links_by_kind = vec![None; overlay_kind_count];
    for contract in links {
        if contract.overlay_kind >= overlay_kind_count {
            return Err(LinkedOverlayError::LinkKindOutOfRange {
                kind: contract.overlay_kind,
                kinds: overlay_kind_count,
            });
        }
        if contract.phase_kind >= phase_kind_count {
            return Err(LinkedOverlayError::LinkPhaseKindOutOfRange {
                overlay_kind: contract.overlay_kind,
                phase_kind: contract.phase_kind,
                phase_kinds: phase_kind_count,
            });
        }
        let kind = contract.overlay_kind;
        if links_by_kind[kind].replace(contract).is_some() {
            return Err(LinkedOverlayError::DuplicateLinkKind { kind });
        }
    }
    for (arm, (&overlay_kind, &phase_kind)) in overlay_kind_map
        .iter()
        .zip(scheduled.layout().phase_kinds())
        .enumerate()
    {
        if let Some(contract) = &links_by_kind[overlay_kind] {
            if contract.phase_kind != phase_kind {
                return Err(LinkedOverlayError::LinkPhaseKindMismatch {
                    arm,
                    overlay_kind,
                    actual_phase_kind: phase_kind,
                    expected_phase_kind: contract.phase_kind,
                });
            }
        }
    }

    let scheduled_columns = scheduled.structure().m;
    let overlay_columns = overlays.structure().m;
    let unpadded_columns = scheduled_columns + overlay_columns - 1;
    let columns = unpadded_columns.next_multiple_of(D);
    let scheduled_rows = 0..scheduled.structure().n;
    let overlay_rows = scheduled_rows.end..scheduled_rows.end + overlays.structure().n;
    let overlay_kind_equality_rows = overlay_rows.end..overlay_rows.end + overlay_kind_count;
    let overlay_activation_rows = overlay_kind_equality_rows.end..overlay_kind_equality_rows.end + arm_count;
    let field_link_count = links_by_kind
        .iter()
        .flatten()
        .map(|contract| contract.fields.len())
        .sum::<usize>();
    let field_link_rows = overlay_activation_rows.end..overlay_activation_rows.end + field_link_count;
    let ring_padding_rows = field_link_rows.end..field_link_rows.end + columns - unpadded_columns;

    let scheduled_embedding = ColumnEmbedding {
        public: scheduled_columns,
        private_start: scheduled_columns,
    };
    let overlay_embedding = ColumnEmbedding {
        public: 1,
        private_start: scheduled_columns,
    };
    let overlay_selector_columns = overlays
        .selector_cols()
        .iter()
        .map(|&column| overlay_embedding.map(column))
        .collect::<Vec<_>>();

    let rows = ring_padding_rows.end;
    let mut explicit = (0..SELECTIVE_ARITY).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut blocks = (0..SELECTIVE_ARITY)
        .map(|_| Vec::<SeededPhi81LinearBlock>::new())
        .collect::<Vec<_>>();
    let mut geometric = (0..SELECTIVE_ARITY)
        .map(|_| Vec::<GeometricRowRun<F>>::new())
        .collect::<Vec<_>>();
    for matrix in 0..SELECTIVE_ARITY {
        append_embedded_matrix(
            "scheduled",
            matrix,
            &scheduled.structure().matrices[matrix],
            scheduled_rows.start,
            scheduled_embedding,
            &mut explicit[matrix],
            &mut blocks[matrix],
            &mut geometric[matrix],
        )?;
        append_embedded_matrix(
            "overlay",
            matrix,
            &overlays.structure().matrices[matrix],
            overlay_rows.start,
            overlay_embedding,
            &mut explicit[matrix],
            &mut blocks[matrix],
            &mut geometric[matrix],
        )?;
    }

    let schedule_selectors = scheduled.layout().schedule_selector_columns();
    for kind in 0..overlay_kind_count {
        let row = overlay_kind_equality_rows.start + kind;
        explicit[GENERAL_SELECTOR].push((row, 0, F::ONE));
        explicit[C].push((row, overlay_selector_columns[kind], F::ONE));
        for (arm, &arm_kind) in overlay_kind_map.iter().enumerate() {
            if arm_kind == kind {
                explicit[C].push((row, schedule_selectors[arm], -F::ONE));
            }
        }
    }
    for arm in 0..arm_count {
        let row = overlay_activation_rows.start + arm;
        explicit[GENERAL_SELECTOR].push((row, 0, F::ONE));
        explicit[A].push((row, schedule_selectors[arm], F::ONE));
        explicit[B].push((row, overlay_selector_columns[overlay_kind_map[arm]], F::ONE));
        explicit[C].push((row, schedule_selectors[arm], F::ONE));
    }

    let phase_public = scheduled.phase_kind_relation().public_input_len();
    let phase_embedding = ColumnEmbedding {
        public: phase_public,
        private_start: scheduled.layout().phase_private_columns().start,
    };
    let mut link_cursor = field_link_rows.start;
    let mut link_row_offsets = vec![link_cursor..link_cursor; overlay_kind_count];
    for kind in 0..overlay_kind_count {
        let start = link_cursor;
        if let Some(contract) = &links_by_kind[kind] {
            for link in &contract.fields {
                let row = link_cursor;
                link_cursor += 1;
                explicit[GENERAL_SELECTOR].push((row, 0, F::ONE));
                explicit[A].push((row, overlay_selector_columns[kind], F::ONE));
                append_source_field_terms(
                    &mut explicit[B],
                    row,
                    "phase",
                    contract.phase_kind,
                    link.phase_field,
                    scheduled.phase_kind_relation(),
                    phase_embedding,
                    F::ONE,
                )?;
                append_source_field_terms(
                    &mut explicit[B],
                    row,
                    "overlay",
                    kind,
                    link.overlay_field,
                    &overlays,
                    overlay_embedding,
                    -F::ONE,
                )?;
            }
        }
        link_row_offsets[kind] = start..link_cursor;
    }
    debug_assert_eq!(link_cursor, field_link_rows.end);
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
            .map_err(LinkedOverlayError::CompactMatrix)?,
        );
    }
    let structure = CcsStructure::new_sparse(matrices, scheduled.structure().f.clone())
        .map_err(|error| LinkedOverlayError::Structure(error.to_string()))?;
    let layout = ScheduledLinkedOverlayLayout {
        public_columns: 0..scheduled.public_input_len(),
        scheduled_private_columns: scheduled.public_input_len()..scheduled_columns,
        overlay_private_columns: scheduled_columns..unpadded_columns,
        ring_padding_columns: unpadded_columns..columns,
        scheduled_rows,
        overlay_rows,
        overlay_kind_equality_rows,
        overlay_activation_rows,
        field_link_rows,
        ring_padding_rows,
        overlay_selector_columns,
        overlay_kinds: overlay_kind_map,
        link_row_offsets,
    };
    Ok(ScheduledLinkedOverlayLowNormR1cs {
        structure,
        scheduled,
        overlays,
        layout,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_source_field_terms(
    terms: &mut Vec<(usize, usize, F)>,
    row: usize,
    owner: &'static str,
    kind: usize,
    field: usize,
    relation: &MultiBranchLowNormR1cs,
    embedding: ColumnEmbedding,
    scale: F,
) -> Result<(), LinkedOverlayError> {
    let decoded =
        decoded_source_field_terms(relation, kind, field).map_err(|reason| LinkedOverlayError::SourceFieldDecoder {
            owner,
            kind,
            field,
            reason,
        })?;
    for (coordinate, coefficient) in decoded {
        terms.push((row, embedding.map(coordinate), scale * coefficient));
    }
    Ok(())
}
