//! Stage-local common coordinate gates for gadget-native lowering.
//!
//! Owns: physical-stage attribution, the seven disjoint Boolean coordinate
//! families, deterministic adjacent pairing, odd tails, two disjoint
//! centered-unit families, and the exact emitted row identity of every
//! retained coordinate.
//!
//! Does not own: source-column classification, specialized acceptance/Mod-5
//! rows, balanced-coordinate omission, or Goldilocks canonicality relations.
//!
//! Emits constraints: yes. Boolean pairs use the fixed nonresidue-seven
//! quadratic pair gate; odd tails use ordinary bitness; centered coordinates
//! use the ordinary centered-unit tail gate.
//!
//! Authority boundary: callers supply only already-validated coordinate
//! ownership. The builder rejects duplicate or missing retained coordinates.
//! The generated Rust-row-to-Lean instantiation bridge remains open; this
//! module does not claim Lean-kernel closure for the complete lowering.
//!
//! | Stage family | Mathematical obligation | Row formula | Lean owner |
//! |---|---|---|---|
//! | `common` | retained ordinary coordinates are Boolean | `floor(n/2)` pair + `n mod 2` tail | `ConstraintEncoding.BooleanPairRows` |
//! | `source.raw64` | canonical source raw bits are Boolean | same, stage-local | `ConstraintEncoding.BooleanPairRows` |
//! | `source.prefix31` | canonical source prefix auxiliaries are Boolean | same, stage-local | `ConstraintEncoding.BooleanPairRows` |
//! | `ring.raw64` / `ring.prefix31` | synthetic ring coordinates are Boolean | same, stage-local | `ConstraintEncoding.BooleanPairRows` |
//! | `product_sum.raw64` / `product_sum.prefix31` | synthetic carry coordinates are Boolean | same, stage-local | `ConstraintEncoding.BooleanPairRows` |
//! | `ordinary_private.centered` | ordinary-private 41-coordinate words lie in `{-1,0,1}` | `floor(n/2)` residual pairs + `n mod 2` tail, stage-local | `CenteredTernaryField.gateWord_iff_alphabetWord` |
//! | `sis_opening.centered` | retained SIS shifted-ternary coordinates lie in `{-1,0,1}` | same, separately paired | `ConstraintEncoding.ResidualPairFamilies` |

use std::ops::Range;

use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::{R1csEncodingTrace, R1csSnapshot};

use super::acceptance::AcceptanceSlots;
use super::balanced_ternary::ValidatedBalancedTernary;
use super::gates::TraceGateBuilder;
use super::mod5::PackedMod5Slots;
use super::product_sum::{ProductSumSlots, ValidatedProductSums};
use super::shared_slots::ReductionPlan;
use super::slots::{ValueEncoding, ValueSlot};
use super::source_schedule::{CanonicalFieldKind, SourceColumnDecision, ValidatedSourceSchedule};
use super::{
    mod5, GadgetNativeError, RingSyntheticSlots, SourceColumn, CANONICAL_PREFIX_AUX, FIELD_BITS, TOOM_COEFFICIENTS,
    TOOM_EVALUATIONS,
};

const UNSTAGED: &str = "gadget_native.unstaged";

/// One independently paired Boolean coordinate family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GadgetNativeBooleanFamily {
    Common,
    SourceRaw64,
    SourcePrefix31,
    SyntheticRingRaw64,
    SyntheticRingPrefix31,
    SyntheticProductSumRaw64,
    SyntheticProductSumPrefix31,
}

impl GadgetNativeBooleanFamily {
    pub const ALL: [Self; 7] = [
        Self::Common,
        Self::SourceRaw64,
        Self::SourcePrefix31,
        Self::SyntheticRingRaw64,
        Self::SyntheticRingPrefix31,
        Self::SyntheticProductSumRaw64,
        Self::SyntheticProductSumPrefix31,
    ];

    const fn index(self) -> usize {
        match self {
            Self::Common => 0,
            Self::SourceRaw64 => 1,
            Self::SourcePrefix31 => 2,
            Self::SyntheticRingRaw64 => 3,
            Self::SyntheticRingPrefix31 => 4,
            Self::SyntheticProductSumRaw64 => 5,
            Self::SyntheticProductSumPrefix31 => 6,
        }
    }
}

/// Exact pair/tail census for one ordered coordinate family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GadgetNativePairTailCount {
    pub coordinates: usize,
    pub pair_rows: usize,
    pub tail_rows: usize,
}

impl GadgetNativePairTailCount {
    pub fn total_rows(self) -> usize {
        self.pair_rows + self.tail_rows
    }

    pub fn from_coordinates(coordinates: usize) -> Self {
        Self {
            coordinates,
            pair_rows: coordinates / 2,
            tail_rows: coordinates % 2,
        }
    }

    pub(super) fn add(&mut self, other: Self) {
        self.coordinates += other.coordinates;
        self.pair_rows += other.pair_rows;
        self.tail_rows += other.tail_rows;
    }
}

/// Exact stage-reset pairing census, split by mathematical owner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GadgetNativeBooleanPairingBreakdown {
    pub common: GadgetNativePairTailCount,
    pub source_raw64: GadgetNativePairTailCount,
    pub source_prefix31: GadgetNativePairTailCount,
    pub synthetic_ring_raw64: GadgetNativePairTailCount,
    pub synthetic_ring_prefix31: GadgetNativePairTailCount,
    pub synthetic_product_sum_raw64: GadgetNativePairTailCount,
    pub synthetic_product_sum_prefix31: GadgetNativePairTailCount,
}

/// Independently paired centered-coordinate owner. Keeping the two owners
/// disjoint prevents a residual-pair row from hiding half of each protocol
/// obligation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GadgetNativeCenteredFamily {
    OrdinaryPrivateField,
    SisOpening,
}

impl GadgetNativeCenteredFamily {
    pub const ALL: [Self; 2] = [Self::OrdinaryPrivateField, Self::SisOpening];

    const fn index(self) -> usize {
        match self {
            Self::OrdinaryPrivateField => 0,
            Self::SisOpening => 1,
        }
    }
}

impl GadgetNativeBooleanPairingBreakdown {
    pub fn family(self, family: GadgetNativeBooleanFamily) -> GadgetNativePairTailCount {
        match family {
            GadgetNativeBooleanFamily::Common => self.common,
            GadgetNativeBooleanFamily::SourceRaw64 => self.source_raw64,
            GadgetNativeBooleanFamily::SourcePrefix31 => self.source_prefix31,
            GadgetNativeBooleanFamily::SyntheticRingRaw64 => self.synthetic_ring_raw64,
            GadgetNativeBooleanFamily::SyntheticRingPrefix31 => self.synthetic_ring_prefix31,
            GadgetNativeBooleanFamily::SyntheticProductSumRaw64 => self.synthetic_product_sum_raw64,
            GadgetNativeBooleanFamily::SyntheticProductSumPrefix31 => self.synthetic_product_sum_prefix31,
        }
    }

    pub fn total_rows(self) -> usize {
        GadgetNativeBooleanFamily::ALL
            .into_iter()
            .map(|family| self.family(family).total_rows())
            .sum()
    }

    pub(super) fn add(&mut self, other: Self) {
        self.common.add(other.common);
        self.source_raw64.add(other.source_raw64);
        self.source_prefix31.add(other.source_prefix31);
        self.synthetic_ring_raw64.add(other.synthetic_ring_raw64);
        self.synthetic_ring_prefix31
            .add(other.synthetic_ring_prefix31);
        self.synthetic_product_sum_raw64
            .add(other.synthetic_product_sum_raw64);
        self.synthetic_product_sum_prefix31
            .add(other.synthetic_product_sum_prefix31);
    }

    pub(super) fn one_stage(
        common: usize,
        source_fields: usize,
        synthetic_ring_fields: usize,
        synthetic_product_sum_fields: usize,
    ) -> Self {
        Self {
            common: GadgetNativePairTailCount::from_coordinates(common),
            source_raw64: GadgetNativePairTailCount::from_coordinates(source_fields * FIELD_BITS),
            source_prefix31: GadgetNativePairTailCount::from_coordinates(source_fields * CANONICAL_PREFIX_AUX),
            synthetic_ring_raw64: GadgetNativePairTailCount::from_coordinates(synthetic_ring_fields * FIELD_BITS),
            synthetic_ring_prefix31: GadgetNativePairTailCount::from_coordinates(
                synthetic_ring_fields * CANONICAL_PREFIX_AUX,
            ),
            synthetic_product_sum_raw64: GadgetNativePairTailCount::from_coordinates(
                synthetic_product_sum_fields * FIELD_BITS,
            ),
            synthetic_product_sum_prefix31: GadgetNativePairTailCount::from_coordinates(
                synthetic_product_sum_fields * CANONICAL_PREFIX_AUX,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PhysicalStageRange {
    pub(super) label: &'static str,
    pub(super) row_start: usize,
    pub(super) row_end: usize,
    pub(super) col_start: usize,
    pub(super) col_end: usize,
}

#[derive(Clone, Debug)]
pub(super) struct PhysicalStageLayout {
    ranges: Vec<PhysicalStageRange>,
}

impl PhysicalStageLayout {
    pub(super) fn checked(source: &R1csSnapshot, trace: &R1csEncodingTrace) -> Result<Self, GadgetNativeError> {
        let checkpoints = trace.stages();
        if checkpoints.is_empty() {
            return Ok(Self {
                ranges: vec![PhysicalStageRange {
                    label: UNSTAGED,
                    row_start: 0,
                    row_end: source.rows(),
                    col_start: 1,
                    col_end: source.cols(),
                }],
            });
        }
        if checkpoints.len() < 2
            || checkpoints[0].row != 0
            || checkpoints[0].col != 1
            || checkpoints
                .last()
                .is_none_or(|last| last.row != source.rows() || last.col != source.cols())
        {
            return Err(schedule_error("physical stage boundaries"));
        }
        let mut ranges = Vec::with_capacity(checkpoints.len() - 1);
        for pair in checkpoints.windows(2) {
            let (start, end) = (&pair[0], &pair[1]);
            if start.row > end.row || start.col > end.col {
                return Err(schedule_error("physical stage order"));
            }
            ranges.push(PhysicalStageRange {
                label: start.label,
                row_start: start.row,
                row_end: end.row,
                col_start: start.col,
                col_end: end.col,
            });
        }
        Ok(Self { ranges })
    }

    pub(super) fn source_column(&self, column: usize) -> Result<usize, GadgetNativeError> {
        self.ranges
            .iter()
            .position(|range| range.col_start <= column && column < range.col_end)
            .ok_or_else(|| schedule_error("source-column stage ownership"))
    }

    pub(super) fn source_column_label(&self, column: usize) -> Result<&'static str, GadgetNativeError> {
        let stage = self.source_column(column)?;
        Ok(self.ranges[stage].label)
    }

    pub(super) fn source_row(&self, row: usize) -> Result<usize, GadgetNativeError> {
        self.ranges
            .iter()
            .position(|range| range.row_start <= row && row < range.row_end)
            .ok_or_else(|| schedule_error("source-row stage ownership"))
    }
}

#[derive(Clone, Debug)]
pub(super) struct PairingCensusAccumulator {
    counts: Vec<[usize; 7]>,
    centered: Vec<[usize; 2]>,
}

impl PairingCensusAccumulator {
    pub(super) fn new(layout: &PhysicalStageLayout) -> Self {
        Self {
            counts: vec![[0; 7]; layout.ranges.len()],
            centered: vec![[0; 2]; layout.ranges.len()],
        }
    }

    pub(super) fn add(&mut self, stage: usize, family: GadgetNativeBooleanFamily, coordinates: usize) {
        self.counts[stage][family.index()] += coordinates;
    }

    pub(super) fn add_centered(&mut self, stage: usize, family: GadgetNativeCenteredFamily, coordinates: usize) {
        self.centered[stage][family.index()] += coordinates;
    }

    pub(super) fn stage(&self, stage: usize) -> GadgetNativeBooleanPairingBreakdown {
        let counts = self.counts[stage];
        GadgetNativeBooleanPairingBreakdown {
            common: GadgetNativePairTailCount::from_coordinates(counts[0]),
            source_raw64: GadgetNativePairTailCount::from_coordinates(counts[1]),
            source_prefix31: GadgetNativePairTailCount::from_coordinates(counts[2]),
            synthetic_ring_raw64: GadgetNativePairTailCount::from_coordinates(counts[3]),
            synthetic_ring_prefix31: GadgetNativePairTailCount::from_coordinates(counts[4]),
            synthetic_product_sum_raw64: GadgetNativePairTailCount::from_coordinates(counts[5]),
            synthetic_product_sum_prefix31: GadgetNativePairTailCount::from_coordinates(counts[6]),
        }
    }

    pub(super) fn total(&self) -> GadgetNativeBooleanPairingBreakdown {
        let mut total = GadgetNativeBooleanPairingBreakdown::default();
        for stage in 0..self.counts.len() {
            total.add(self.stage(stage));
        }
        total
    }

    pub(super) fn centered_total(&self) -> GadgetNativePairTailCount {
        GadgetNativeCenteredFamily::ALL
            .into_iter()
            .fold(GadgetNativePairTailCount::default(), |mut total, family| {
                total.add(self.centered_family_total(family));
                total
            })
    }

    fn centered_stage(&self, stage: usize) -> GadgetNativePairTailCount {
        GadgetNativeCenteredFamily::ALL
            .into_iter()
            .fold(GadgetNativePairTailCount::default(), |mut total, family| {
                total.add(GadgetNativePairTailCount::from_coordinates(
                    self.centered[stage][family.index()],
                ));
                total
            })
    }

    pub(super) fn centered_family_total(&self, family: GadgetNativeCenteredFamily) -> GadgetNativePairTailCount {
        self.centered
            .iter()
            .fold(GadgetNativePairTailCount::default(), |mut total, coordinates| {
                total.add(GadgetNativePairTailCount::from_coordinates(coordinates[family.index()]));
                total
            })
    }
}

/// Allocation-free coordinate census shared by the production estimator and
/// selective physical-row audits. Pairing resets at every physical source
/// stage and mathematical coordinate family, exactly as in `build_schedule`.
pub(super) struct PlannedCoordinatePairing {
    layout: PhysicalStageLayout,
    census: PairingCensusAccumulator,
}

impl PlannedCoordinatePairing {
    pub(super) fn checked(
        source: &R1csSnapshot,
        trace: &R1csEncodingTrace,
        schedule: &ValidatedSourceSchedule,
        linearly_derived: &[bool],
    ) -> Result<Self, GadgetNativeError> {
        if linearly_derived.len() != source.cols() {
            return Err(schedule_error("linear source mask width"));
        }
        let layout = PhysicalStageLayout::checked(source, trace)?;
        let specialized_boolean = specialized_boolean_columns(trace, source.cols());
        let mut census = PairingCensusAccumulator::new(&layout);
        for (column, decision) in schedule.decisions().iter().enumerate().skip(1) {
            if schedule.marks.gadget_columns[column]
                || linearly_derived[column]
                || schedule
                    .marks
                    .balanced_ternary
                    .digit_alias(column)
                    .is_some()
                || schedule
                    .marks
                    .balanced_ternary
                    .opening_for_field(column)
                    .is_some()
                || schedule.marks.balanced_ternary.is_binary(column)
                || specialized_boolean[column]
            {
                continue;
            }
            if schedule.is_public[column] || schedule.explicit_bits[column] {
                add_source_count(&layout, &mut census, column, GadgetNativeBooleanFamily::Common, 1)?;
            } else if matches!(
                decision,
                SourceColumnDecision::CanonicalField(CanonicalFieldKind::DirectCanonicalU64)
            ) {
                add_source_count(
                    &layout,
                    &mut census,
                    column,
                    GadgetNativeBooleanFamily::SourceRaw64,
                    FIELD_BITS,
                )?;
                add_source_count(
                    &layout,
                    &mut census,
                    column,
                    GadgetNativeBooleanFamily::SourcePrefix31,
                    CANONICAL_PREFIX_AUX,
                )?;
            } else if matches!(
                decision,
                SourceColumnDecision::CanonicalField(CanonicalFieldKind::OrdinaryPrivate)
            ) {
                add_centered_source_count(
                    &layout,
                    &mut census,
                    column,
                    GadgetNativeCenteredFamily::OrdinaryPrivateField,
                    super::ordinary_private_field::ORDINARY_PRIVATE_DIGITS,
                )?;
            }
        }
        add_synthetic_counts(&layout, trace, &schedule.marks.product_sums, &mut census)?;
        for opening in trace.balanced_ternary_openings() {
            add_centered_count(
                &layout,
                &mut census,
                opening.digit_rows.start,
                GadgetNativeCenteredFamily::SisOpening,
                super::BALANCED_TERNARY_DIGITS,
            )?;
        }
        Ok(Self { layout, census })
    }

    pub(super) fn source_stage(&self, column: usize) -> Result<usize, GadgetNativeError> {
        self.layout.source_column(column)
    }

    pub(super) fn stage_count(&self) -> usize {
        self.layout.ranges.len()
    }

    pub(super) fn stage_boolean(&self, stage: usize) -> GadgetNativeBooleanPairingBreakdown {
        self.census.stage(stage)
    }

    pub(super) fn stage_centered(&self, stage: usize) -> GadgetNativePairTailCount {
        self.census.centered_stage(stage)
    }

    pub(super) fn stage_row_start(&self, stage: usize) -> usize {
        (0..stage)
            .map(|prior| self.stage_boolean(prior).total_rows() + self.stage_centered(prior).total_rows())
            .sum()
    }

    pub(super) fn total_rows(&self) -> usize {
        self.stage_row_start(self.stage_count())
    }

    pub(super) fn boolean_total(&self) -> GadgetNativeBooleanPairingBreakdown {
        self.census.total()
    }

    pub(super) fn centered_total(&self) -> GadgetNativePairTailCount {
        self.census.centered_total()
    }

    pub(super) fn centered_family_total(&self, family: GadgetNativeCenteredFamily) -> GadgetNativePairTailCount {
        self.census.centered_family_total(family)
    }
}

/// Audit classification for one emitted coordinate group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GadgetNativeCoordinateGroupFamily {
    Boolean(GadgetNativeBooleanFamily),
    CenteredUnit(GadgetNativeCenteredFamily),
}

/// One physical-stage/family group in exact emission order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GadgetNativeCoordinateGroupAudit {
    pub stage: &'static str,
    pub family: GadgetNativeCoordinateGroupFamily,
    pub coordinates: Vec<usize>,
    pub encoded_rows: Range<usize>,
}

/// One exact common coordinate row in the materialized relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GadgetNativeCoordinateRowAudit {
    BooleanPair {
        row: usize,
        left: usize,
        right: usize,
    },
    BooleanTail {
        row: usize,
        coordinate: usize,
    },
    CenteredUnitPair {
        row: usize,
        family: GadgetNativeCenteredFamily,
        left: usize,
        right: usize,
    },
    CenteredUnitTail {
        row: usize,
        family: GadgetNativeCenteredFamily,
        coordinate: usize,
    },
}

impl GadgetNativeCoordinateRowAudit {
    pub fn row(self) -> usize {
        match self {
            Self::BooleanPair { row, .. }
            | Self::BooleanTail { row, .. }
            | Self::CenteredUnitPair { row, .. }
            | Self::CenteredUnitTail { row, .. } => row,
        }
    }
}

/// Exact stage-local schedule stored in the executable plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GadgetNativeCoordinateGateSchedule {
    groups: Vec<GadgetNativeCoordinateGroupAudit>,
    rows: Vec<GadgetNativeCoordinateRowAudit>,
    row_by_column: Vec<Option<usize>>,
    pairing: GadgetNativeBooleanPairingBreakdown,
    centered_pairing: GadgetNativePairTailCount,
    ordinary_private_centered_pairing: GadgetNativePairTailCount,
    sis_centered_pairing: GadgetNativePairTailCount,
}

impl GadgetNativeCoordinateGateSchedule {
    pub fn groups(&self) -> &[GadgetNativeCoordinateGroupAudit] {
        &self.groups
    }

    pub fn rows(&self) -> &[GadgetNativeCoordinateRowAudit] {
        &self.rows
    }

    pub fn row_for_column(&self, column: usize) -> Option<usize> {
        self.row_by_column.get(column).copied().flatten()
    }

    pub fn centered_family_for_column(&self, column: usize) -> Option<GadgetNativeCenteredFamily> {
        let row = self.row_for_column(column)?;
        match self.rows.get(row)? {
            GadgetNativeCoordinateRowAudit::CenteredUnitPair {
                family, left, right, ..
            } if *left == column || *right == column => Some(*family),
            GadgetNativeCoordinateRowAudit::CenteredUnitTail { family, coordinate, .. } if *coordinate == column => {
                Some(*family)
            }
            _ => None,
        }
    }

    pub fn pairing(&self) -> GadgetNativeBooleanPairingBreakdown {
        self.pairing
    }

    pub fn centered_pairing(&self) -> GadgetNativePairTailCount {
        self.centered_pairing
    }

    pub fn centered_pairing_for(&self, family: GadgetNativeCenteredFamily) -> GadgetNativePairTailCount {
        match family {
            GadgetNativeCenteredFamily::OrdinaryPrivateField => self.ordinary_private_centered_pairing,
            GadgetNativeCenteredFamily::SisOpening => self.sis_centered_pairing,
        }
    }

    pub(super) fn emit(&self, gates: &mut TraceGateBuilder) -> Result<(), GadgetNativeError> {
        if gates.rows != 0 {
            return Err(schedule_error("coordinate gates must be emitted first"));
        }
        for row in &self.rows {
            if gates.rows != row.row() {
                return Err(schedule_error("recorded/emitted coordinate row identity"));
            }
            match *row {
                GadgetNativeCoordinateRowAudit::BooleanPair { left, right, .. } => {
                    gates.quadratic_bit_pair(vec![(left, neo_math::F::ONE)], vec![(right, neo_math::F::ONE)]);
                }
                GadgetNativeCoordinateRowAudit::BooleanTail { coordinate, .. } => gates.bitness(coordinate),
                GadgetNativeCoordinateRowAudit::CenteredUnitPair { left, right, .. } => {
                    gates.centered_unit_pair(left, right);
                }
                GadgetNativeCoordinateRowAudit::CenteredUnitTail { coordinate, .. } => {
                    gates.centered_unit_tail(coordinate);
                }
            }
        }
        Ok(())
    }
}

pub(super) struct CoordinateGateInputs<'a> {
    pub(super) source: &'a R1csSnapshot,
    pub(super) trace: &'a R1csEncodingTrace,
    pub(super) source_columns: &'a [SourceColumn],
    pub(super) ring_slots: &'a [RingSyntheticSlots],
    pub(super) product_sum_slots: &'a ProductSumSlots,
    pub(super) product_sums: &'a ValidatedProductSums,
    pub(super) balanced: &'a ValidatedBalancedTernary,
    pub(super) reduction: &'a ReductionPlan,
    pub(super) acceptance: &'a AcceptanceSlots,
    pub(super) mod5: &'a PackedMod5Slots,
    pub(super) encoded_columns: usize,
}

pub(super) fn build_schedule(
    inputs: CoordinateGateInputs<'_>,
) -> Result<GadgetNativeCoordinateGateSchedule, GadgetNativeError> {
    let layout = PhysicalStageLayout::checked(inputs.source, inputs.trace)?;
    let mut builder = ScheduleBuilder::new(&layout, inputs.encoded_columns);

    for (source_column, definition) in inputs.source_columns.iter().enumerate().skip(1) {
        let SourceColumn::Encoded(slot) = definition else {
            continue;
        };
        match slot.encoding {
            ValueEncoding::Boolean => {
                if !omitted(&inputs, slot.start) {
                    let stage = layout.source_column(source_column)?;
                    builder.push(stage, GadgetNativeBooleanFamily::Common, slot.start)?;
                }
            }
            ValueEncoding::CanonicalBinary { auxiliary_start } => {
                let stage = layout.source_column(source_column)?;
                builder.extend(
                    stage,
                    GadgetNativeBooleanFamily::SourceRaw64,
                    slot.start..slot.start + FIELD_BITS,
                )?;
                builder.extend(
                    stage,
                    GadgetNativeBooleanFamily::SourcePrefix31,
                    auxiliary_start..auxiliary_start + CANONICAL_PREFIX_AUX,
                )?;
            }
            ValueEncoding::OrdinaryCenteredTernary => {
                let stage = layout.source_column(source_column)?;
                builder.extend_centered(
                    stage,
                    GadgetNativeCenteredFamily::OrdinaryPrivateField,
                    slot.start..slot.start + slot.width,
                )?;
            }
            ValueEncoding::BalancedTernary => {
                let opening = inputs
                    .balanced
                    .opening_for_field(source_column)
                    .ok_or_else(|| schedule_error("balanced field stage ownership"))?;
                let stage = layout.source_row(
                    inputs.trace.balanced_ternary_openings()[opening]
                        .digit_rows
                        .start,
                )?;
                builder.extend_centered(
                    stage,
                    GadgetNativeCenteredFamily::SisOpening,
                    slot.start..slot.start + slot.width,
                )?;
            }
            ValueEncoding::CenteredUnit => {
                // Balanced digit aliases point into the parent field slot and
                // therefore do not own another coordinate occurrence.
            }
        }
    }

    for (event, slots) in inputs.trace.ring_muls_toom3().iter().zip(inputs.ring_slots) {
        let stage = layout.source_row(event.source_rows.start)?;
        if slots.coefficients.len() != TOOM_EVALUATIONS * TOOM_COEFFICIENTS {
            return Err(schedule_error("synthetic ring slot census"));
        }
        for slot in &slots.coefficients {
            push_canonical(
                &mut builder,
                stage,
                GadgetNativeBooleanFamily::SyntheticRingRaw64,
                GadgetNativeBooleanFamily::SyntheticRingPrefix31,
                *slot,
            )?;
        }
    }
    for (stage_row, slot) in inputs.product_sum_slots.staged_fields(inputs.product_sums) {
        let stage = layout.source_row(stage_row)?;
        push_canonical(
            &mut builder,
            stage,
            GadgetNativeBooleanFamily::SyntheticProductSumRaw64,
            GadgetNativeBooleanFamily::SyntheticProductSumPrefix31,
            slot,
        )?;
    }

    builder.finish(|column| omitted(&inputs, column))
}

fn omitted(inputs: &CoordinateGateInputs<'_>, column: usize) -> bool {
    inputs.reduction.omits_coordinate_column(column)
        || inputs.acceptance.omits_coordinate(column)
        || inputs.mod5.omits_coordinate(column)
}

fn push_canonical(
    builder: &mut ScheduleBuilder,
    stage: usize,
    raw: GadgetNativeBooleanFamily,
    prefix: GadgetNativeBooleanFamily,
    slot: ValueSlot,
) -> Result<(), GadgetNativeError> {
    let ValueEncoding::CanonicalBinary { auxiliary_start } = slot.encoding else {
        return Err(schedule_error("synthetic canonical slot encoding"));
    };
    builder.extend(stage, raw, slot.start..slot.start + FIELD_BITS)?;
    builder.extend(stage, prefix, auxiliary_start..auxiliary_start + CANONICAL_PREFIX_AUX)
}

struct ScheduleBuilder {
    labels: Vec<&'static str>,
    boolean: Vec<[Vec<usize>; 7]>,
    centered: Vec<[Vec<usize>; 2]>,
    claimed: Vec<bool>,
}

impl ScheduleBuilder {
    fn new(layout: &PhysicalStageLayout, encoded_columns: usize) -> Self {
        Self {
            labels: layout.ranges.iter().map(|range| range.label).collect(),
            boolean: (0..layout.ranges.len())
                .map(|_| std::array::from_fn(|_| Vec::new()))
                .collect(),
            centered: (0..layout.ranges.len())
                .map(|_| std::array::from_fn(|_| Vec::new()))
                .collect(),
            claimed: vec![false; encoded_columns],
        }
    }

    fn push(
        &mut self,
        stage: usize,
        family: GadgetNativeBooleanFamily,
        column: usize,
    ) -> Result<(), GadgetNativeError> {
        self.claim(stage, column)?;
        self.boolean[stage][family.index()].push(column);
        Ok(())
    }

    fn extend(
        &mut self,
        stage: usize,
        family: GadgetNativeBooleanFamily,
        columns: Range<usize>,
    ) -> Result<(), GadgetNativeError> {
        for column in columns {
            self.push(stage, family, column)?;
        }
        Ok(())
    }

    fn extend_centered(
        &mut self,
        stage: usize,
        family: GadgetNativeCenteredFamily,
        columns: Range<usize>,
    ) -> Result<(), GadgetNativeError> {
        for column in columns {
            self.claim(stage, column)?;
            self.centered[stage][family.index()].push(column);
        }
        Ok(())
    }

    fn claim(&mut self, stage: usize, column: usize) -> Result<(), GadgetNativeError> {
        if stage >= self.labels.len()
            || column == 0
            || column >= self.claimed.len()
            || std::mem::replace(&mut self.claimed[column], true)
        {
            return Err(schedule_error("duplicate or out-of-range coordinate ownership"));
        }
        Ok(())
    }

    fn finish(
        self,
        is_omitted: impl Fn(usize) -> bool,
    ) -> Result<GadgetNativeCoordinateGateSchedule, GadgetNativeError> {
        for column in 1..self.claimed.len() {
            if self.claimed[column] == is_omitted(column) {
                return Err(schedule_error("retained coordinate coverage"));
            }
        }

        let groups_per_stage = GadgetNativeBooleanFamily::ALL.len() + GadgetNativeCenteredFamily::ALL.len();
        let mut groups = Vec::with_capacity(self.labels.len() * groups_per_stage);
        let mut rows = Vec::new();
        let mut row_by_column = vec![None; self.claimed.len()];
        let mut pairing = GadgetNativeBooleanPairingBreakdown::default();
        for stage in 0..self.labels.len() {
            for family in GadgetNativeBooleanFamily::ALL {
                let coordinates = self.boolean[stage][family.index()].clone();
                let start = rows.len();
                let mut chunks = coordinates.chunks_exact(2);
                for pair in &mut chunks {
                    let row = rows.len();
                    row_by_column[pair[0]] = Some(row);
                    row_by_column[pair[1]] = Some(row);
                    rows.push(GadgetNativeCoordinateRowAudit::BooleanPair {
                        row,
                        left: pair[0],
                        right: pair[1],
                    });
                }
                if let [coordinate] = chunks.remainder() {
                    let row = rows.len();
                    row_by_column[*coordinate] = Some(row);
                    rows.push(GadgetNativeCoordinateRowAudit::BooleanTail {
                        row,
                        coordinate: *coordinate,
                    });
                }
                let count = GadgetNativePairTailCount::from_coordinates(coordinates.len());
                let mut one = GadgetNativeBooleanPairingBreakdown::default();
                match family {
                    GadgetNativeBooleanFamily::Common => one.common = count,
                    GadgetNativeBooleanFamily::SourceRaw64 => one.source_raw64 = count,
                    GadgetNativeBooleanFamily::SourcePrefix31 => one.source_prefix31 = count,
                    GadgetNativeBooleanFamily::SyntheticRingRaw64 => one.synthetic_ring_raw64 = count,
                    GadgetNativeBooleanFamily::SyntheticRingPrefix31 => one.synthetic_ring_prefix31 = count,
                    GadgetNativeBooleanFamily::SyntheticProductSumRaw64 => {
                        one.synthetic_product_sum_raw64 = count;
                    }
                    GadgetNativeBooleanFamily::SyntheticProductSumPrefix31 => {
                        one.synthetic_product_sum_prefix31 = count;
                    }
                }
                pairing.add(one);
                groups.push(GadgetNativeCoordinateGroupAudit {
                    stage: self.labels[stage],
                    family: GadgetNativeCoordinateGroupFamily::Boolean(family),
                    coordinates,
                    encoded_rows: start..rows.len(),
                });
            }
            for family in GadgetNativeCenteredFamily::ALL {
                let coordinates = self.centered[stage][family.index()].clone();
                let start = rows.len();
                let mut chunks = coordinates.chunks_exact(2);
                for pair in &mut chunks {
                    let row = rows.len();
                    row_by_column[pair[0]] = Some(row);
                    row_by_column[pair[1]] = Some(row);
                    rows.push(GadgetNativeCoordinateRowAudit::CenteredUnitPair {
                        row,
                        family,
                        left: pair[0],
                        right: pair[1],
                    });
                }
                if let [coordinate] = chunks.remainder() {
                    let row = rows.len();
                    row_by_column[*coordinate] = Some(row);
                    rows.push(GadgetNativeCoordinateRowAudit::CenteredUnitTail {
                        row,
                        family,
                        coordinate: *coordinate,
                    });
                }
                groups.push(GadgetNativeCoordinateGroupAudit {
                    stage: self.labels[stage],
                    family: GadgetNativeCoordinateGroupFamily::CenteredUnit(family),
                    coordinates,
                    encoded_rows: start..rows.len(),
                });
            }
        }
        let family_pairing = |family: GadgetNativeCenteredFamily| {
            self.centered
                .iter()
                .fold(GadgetNativePairTailCount::default(), |mut total, coordinates| {
                    total.add(GadgetNativePairTailCount::from_coordinates(
                        coordinates[family.index()].len(),
                    ));
                    total
                })
        };
        let ordinary_private_centered_pairing = family_pairing(GadgetNativeCenteredFamily::OrdinaryPrivateField);
        let sis_centered_pairing = family_pairing(GadgetNativeCenteredFamily::SisOpening);
        let mut centered_pairing = ordinary_private_centered_pairing;
        centered_pairing.add(sis_centered_pairing);
        Ok(GadgetNativeCoordinateGateSchedule {
            groups,
            rows,
            row_by_column,
            pairing,
            centered_pairing,
            ordinary_private_centered_pairing,
            sis_centered_pairing,
        })
    }
}

pub(super) fn specialized_boolean_columns(trace: &R1csEncodingTrace, source_columns: usize) -> Vec<bool> {
    let mut specialized = vec![false; source_columns];
    for event in trace.acceptance_chunks() {
        specialized[event.accept.col()] = true;
    }
    for event in trace.mod5_chunks() {
        for bit in &event.quotient_bits[..mod5::LOW_QUOTIENT_BITS] {
            specialized[bit.col()] = true;
        }
    }
    specialized
}

pub(super) fn add_synthetic_counts(
    layout: &PhysicalStageLayout,
    trace: &R1csEncodingTrace,
    product_sums: &ValidatedProductSums,
    census: &mut PairingCensusAccumulator,
) -> Result<(), GadgetNativeError> {
    for event in trace.ring_muls_toom3() {
        let stage = layout.source_row(event.source_rows.start)?;
        census.add(
            stage,
            GadgetNativeBooleanFamily::SyntheticRingRaw64,
            TOOM_EVALUATIONS * TOOM_COEFFICIENTS * FIELD_BITS,
        );
        census.add(
            stage,
            GadgetNativeBooleanFamily::SyntheticRingPrefix31,
            TOOM_EVALUATIONS * TOOM_COEFFICIENTS * CANONICAL_PREFIX_AUX,
        );
    }
    for cost in product_sums.costs() {
        let stage = layout.source_row(cost.stage_row)?;
        census.add(
            stage,
            GadgetNativeBooleanFamily::SyntheticProductSumRaw64,
            cost.synthetic_fields * FIELD_BITS,
        );
        census.add(
            stage,
            GadgetNativeBooleanFamily::SyntheticProductSumPrefix31,
            cost.synthetic_fields * CANONICAL_PREFIX_AUX,
        );
    }
    Ok(())
}

pub(super) fn add_source_count(
    layout: &PhysicalStageLayout,
    census: &mut PairingCensusAccumulator,
    source_column: usize,
    family: GadgetNativeBooleanFamily,
    coordinates: usize,
) -> Result<(), GadgetNativeError> {
    let stage = layout.source_column(source_column)?;
    census.add(stage, family, coordinates);
    Ok(())
}

pub(super) fn add_centered_count(
    layout: &PhysicalStageLayout,
    census: &mut PairingCensusAccumulator,
    source_row: usize,
    family: GadgetNativeCenteredFamily,
    coordinates: usize,
) -> Result<(), GadgetNativeError> {
    let stage = layout.source_row(source_row)?;
    census.add_centered(stage, family, coordinates);
    Ok(())
}

pub(super) fn add_centered_source_count(
    layout: &PhysicalStageLayout,
    census: &mut PairingCensusAccumulator,
    source_column: usize,
    family: GadgetNativeCenteredFamily,
    coordinates: usize,
) -> Result<(), GadgetNativeError> {
    let stage = layout.source_column(source_column)?;
    census.add_centered(stage, family, coordinates);
    Ok(())
}

fn schedule_error(detail: &'static str) -> GadgetNativeError {
    GadgetNativeError::CoordinateGateSchedule { detail }
}
