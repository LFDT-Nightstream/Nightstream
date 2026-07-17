//! Read-only snapshot of the selective low-norm encoder plan.
//!
//! Owns: checked projection of private compiler state into immutable audit
//! views and an interpreter for that projected plan.
//!
//! Does not own: CCS semantics, source-R1CS semantics, serialization, generated
//! artifacts, cost authority, or permission to remove constraints.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the snapshot borrows the compiled structure and encoder
//! plan without copying either. It is plan-replay evidence, never proof that the
//! emitted CCS relation is a sound or minimal implementation of the protocol.
//!
//! | Snapshot branch | Mathematical object | Validation performed |
//! |---|---|---|
//! | Relation | selected CCS structure and public/selector boundary | arity and contiguous layout partition |
//! | Source fields | encoded slots, coordinate aliases, equality sources, centered flags | shape, domain, alias, and owned-slot placement |
//! | Derived values | ordered product sums and optional predecessor | factor arity, source-LC bounds, ordered destinations, and acyclic predecessors |

use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::paper::relations::Structure;

use super::{CompactIndex, CompactSlot, DerivedProductSumEncoding, LowNormR1csError, MultiBranchLowNormR1cs};
use crate::frontends::r1cs_f_prime::{selective::EVAL_GROUP_SIZE, SelectiveCompilerAudit, SelectiveLayoutAudit};

const SELECTIVE_ARITY: usize = 13;
const BINARY_FIELD_WIDTH: usize = 64;

/// One contiguous destination in the low-norm assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodedSlotSnapshot {
    start: usize,
    len: usize,
}

impl EncodedSlotSnapshot {
    pub fn start(&self) -> usize {
        self.start
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn range(&self) -> core::ops::Range<usize> {
        self.start..self.start + self.len
    }

    fn pair(self) -> (usize, usize) {
        (self.start, self.len)
    }
}

/// Reuse one binary-bit or balanced-digit coordinate of an earlier field slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceCoordinateAliasSnapshot {
    source_field: usize,
    coordinate: usize,
}

impl SourceCoordinateAliasSnapshot {
    pub fn source_field(&self) -> usize {
        self.source_field
    }

    pub fn coordinate(&self) -> usize {
        self.coordinate
    }

    fn pair(self) -> (usize, usize) {
        (self.source_field, self.coordinate)
    }
}

/// Borrowed source-field linear combination, preserving term order and repeats.
#[derive(Clone, Copy, Debug)]
pub struct LinearCombinationSnapshot<'a> {
    inner: &'a crate::engine::r1cs_circuit::Lc,
}

impl<'a> LinearCombinationSnapshot<'a> {
    pub fn terms(&self) -> &[(usize, F)] {
        &self.inner.terms
    }

    pub fn constant(&self) -> F {
        self.inner.constant
    }

    fn evaluate(&self, source: &[F]) -> F {
        self.inner
            .terms
            .iter()
            .fold(self.inner.constant, |sum, &(column, coefficient)| {
                sum + coefficient * source[column]
            })
    }
}

/// Borrowed ordered factor in a derived product sum.
#[derive(Clone, Copy, Debug)]
pub struct ProductFactorSnapshot<'a> {
    inner: &'a crate::engine::r1cs_circuit::builder::ProductFactorTrace,
}

impl<'a> ProductFactorSnapshot<'a> {
    pub fn left(&self) -> LinearCombinationSnapshot<'a> {
        LinearCombinationSnapshot {
            inner: &self.inner.left,
        }
    }

    pub fn right(&self) -> LinearCombinationSnapshot<'a> {
        LinearCombinationSnapshot {
            inner: &self.inner.right,
        }
    }

    pub fn coefficient(&self) -> F {
        self.inner.coefficient
    }
}

/// Borrowed materialized partial product sum in compiler order.
#[derive(Clone, Copy, Debug)]
pub struct DerivedProductSumSnapshot<'a> {
    inner: &'a DerivedProductSumEncoding,
}

impl<'a> DerivedProductSumSnapshot<'a> {
    pub fn slot(&self) -> EncodedSlotSnapshot {
        EncodedSlotSnapshot {
            start: self.inner.slot.0,
            len: self.inner.slot.1,
        }
    }

    pub fn factors(&self) -> impl ExactSizeIterator<Item = ProductFactorSnapshot<'a>> + '_ {
        self.inner
            .factors
            .iter()
            .map(|inner| ProductFactorSnapshot { inner })
    }

    pub fn previous(&self) -> Option<usize> {
        self.inner.previous
    }
}

/// Borrowed encoding plan for one selectable source arm.
#[derive(Clone, Copy, Debug)]
pub struct SelectiveArmEncodingSnapshot<'a> {
    slots: &'a [CompactSlot],
    bit_aliases: &'a [CompactSlot],
    equality_sources: &'a [CompactIndex],
    centered_columns: &'a [bool],
    derived_product_sums: &'a [DerivedProductSumEncoding],
}

impl<'a> SelectiveArmEncodingSnapshot<'a> {
    pub fn field_count(&self) -> usize {
        self.slots.len()
    }

    pub fn slot(&self, field: usize) -> Option<EncodedSlotSnapshot> {
        self.slots
            .get(field)
            .copied()
            .and_then(CompactSlot::get)
            .map(|(start, len)| EncodedSlotSnapshot { start, len })
    }

    pub fn slots(&self) -> impl ExactSizeIterator<Item = Option<EncodedSlotSnapshot>> + '_ {
        self.slots.iter().copied().map(|slot| {
            slot.get()
                .map(|(start, len)| EncodedSlotSnapshot { start, len })
        })
    }

    pub fn coordinate_alias(&self, field: usize) -> Option<SourceCoordinateAliasSnapshot> {
        self.bit_aliases
            .get(field)
            .copied()
            .and_then(CompactSlot::get)
            .map(|(source_field, coordinate)| SourceCoordinateAliasSnapshot {
                source_field,
                coordinate,
            })
    }

    pub fn coordinate_aliases(&self) -> impl ExactSizeIterator<Item = Option<SourceCoordinateAliasSnapshot>> + '_ {
        self.bit_aliases.iter().copied().map(|alias| {
            alias
                .get()
                .map(|(source_field, coordinate)| SourceCoordinateAliasSnapshot {
                    source_field,
                    coordinate,
                })
        })
    }

    pub fn equality_source(&self, field: usize) -> Option<usize> {
        self.equality_sources
            .get(field)
            .copied()
            .and_then(CompactIndex::get)
    }

    pub fn equality_sources(&self) -> impl ExactSizeIterator<Item = Option<usize>> + '_ {
        self.equality_sources.iter().copied().map(CompactIndex::get)
    }

    pub fn centered_columns(&self) -> &[bool] {
        self.centered_columns
    }

    pub fn derived_product_sums(&self) -> impl ExactSizeIterator<Item = DerivedProductSumSnapshot<'a>> + '_ {
        self.derived_product_sums
            .iter()
            .map(|inner| DerivedProductSumSnapshot { inner })
    }
}

/// Checked, immutable view of one compiled selective relation.
#[derive(Clone, Copy, Debug)]
pub struct SelectiveLowNormSnapshot<'a> {
    relation: &'a MultiBranchLowNormR1cs,
    compiler_audit: &'a SelectiveCompilerAudit,
}

impl<'a> SelectiveLowNormSnapshot<'a> {
    pub fn structure(&self) -> &'a Structure {
        self.relation.structure.as_ref()
    }

    pub fn layout(&self) -> &'a SelectiveLayoutAudit {
        self.compiler_audit.layout()
    }

    /// Exact row/width/layout ledger produced by the same selective compiler
    /// run as the borrowed final structure.
    pub fn compiler_audit(&self) -> &'a SelectiveCompilerAudit {
        self.compiler_audit
    }

    pub fn public_input_len(&self) -> usize {
        self.relation.public_input_len
    }

    pub fn public_field_count(&self) -> usize {
        self.relation.public_field_count
    }

    pub fn selector_cols(&self) -> &[usize] {
        &self.relation.selector_cols
    }

    pub fn arm_count(&self) -> usize {
        self.relation.arm_slots.len()
    }

    pub fn arm(&self, arm: usize) -> Option<SelectiveArmEncodingSnapshot<'a>> {
        Some(SelectiveArmEncodingSnapshot {
            slots: self.relation.arm_slots.get(arm)?,
            bit_aliases: &self.relation.arm_aliases[arm],
            equality_sources: &self.relation.arm_equal_aliases[arm],
            centered_columns: &self.relation.arm_centered_columns[arm],
            derived_product_sums: &self.relation.arm_derived_product_sums[arm],
        })
    }

    pub fn arms(&self) -> impl ExactSizeIterator<Item = SelectiveArmEncodingSnapshot<'a>> + '_ {
        (0..self.arm_count()).map(|arm| self.arm(arm).expect("validated arm census"))
    }

    /// Interpret the borrowed plan without delegating to the live
    /// relation encoder and therefore supports differential conformance tests.
    pub fn encode(&self, arm: usize, field_assignment: &[F]) -> Result<Vec<F>, LowNormR1csError> {
        let plan = self.arm(arm).ok_or(LowNormR1csError::ArmIndexOutOfRange {
            arm,
            arms: self.arm_count(),
        })?;
        if field_assignment.len() != plan.field_count() {
            return Err(LowNormR1csError::AssignmentLength {
                got: field_assignment.len(),
                expected: plan.field_count(),
            });
        }
        if field_assignment.first().copied() != Some(F::ONE) {
            return Err(LowNormR1csError::ConstantOne);
        }

        let mut assignment = vec![F::ZERO; self.structure().m];
        assignment[0] = F::ONE;
        assignment[self.selector_cols()[arm]] = F::ONE;
        for field_col in 1..plan.field_count() {
            let Some(slot) = plan.slot(field_col) else {
                continue;
            };
            if let Some(source_col) = plan.equality_source(field_col) {
                if field_assignment[field_col] != field_assignment[source_col] {
                    return Err(LowNormR1csError::AliasedFieldMismatch { field_col, source_col });
                }
                continue;
            }
            replay_write_encoded_value(
                &mut assignment,
                Some(slot.pair()),
                plan.coordinate_alias(field_col)
                    .map(SourceCoordinateAliasSnapshot::pair),
                plan.centered_columns[field_col],
                field_assignment[field_col],
                field_col,
            )?;
        }

        let mut derived_values = Vec::with_capacity(plan.derived_product_sums.len());
        for derived in plan.derived_product_sums() {
            let mut value = derived.factors().fold(F::ZERO, |sum, factor| {
                sum + factor.coefficient()
                    * factor.left().evaluate(field_assignment)
                    * factor.right().evaluate(field_assignment)
            });
            if let Some(previous) = derived.previous() {
                value += derived_values[previous];
            }
            replay_write_encoded_value(
                &mut assignment,
                Some(derived.slot().pair()),
                None,
                false,
                value,
                usize::MAX,
            )?;
            derived_values.push(value);
        }
        Ok(assignment)
    }
}

/// Internal-plan corruption found while taking a selective snapshot.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SelectiveSnapshotError {
    #[error("selective snapshot requires a selectively compiled relation")]
    NotSelective,
    #[error("selective snapshot structure has no constant-one column")]
    EmptyStructure,
    #[error("selective snapshot requires 13 matrix/polynomial ports, got {matrices}/{polynomial_arity}")]
    SelectiveArityMismatch {
        matrices: usize,
        polynomial_arity: usize,
    },
    #[error("selective snapshot layout width {layout_columns} != structure width {structure_columns}")]
    LayoutColumnMismatch {
        layout_columns: usize,
        structure_columns: usize,
    },
    #[error("selective snapshot layout public input length {layout_len} != encoder length {encoder_len}")]
    LayoutPublicInputMismatch {
        layout_len: usize,
        encoder_len: usize,
    },
    #[error("selective snapshot layout selectors {layout:?} != encoder selectors {encoder:?}")]
    LayoutSelectorMismatch {
        layout: Vec<usize>,
        encoder: Vec<usize>,
    },
    #[error("selective snapshot layout component {component} is not the compiler's contiguous partition")]
    LayoutPartitionMismatch { component: &'static str },
    #[error(
        "selective snapshot public field count {public_field_count} != logical public width {logical_public_input_len}"
    )]
    PublicFieldLayoutMismatch {
        public_field_count: usize,
        logical_public_input_len: usize,
    },
    #[error("selective snapshot public input length {public_input_len} exceeds structure width {columns}")]
    PublicInputOutOfBounds {
        public_input_len: usize,
        columns: usize,
    },
    #[error("selective snapshot {component} arm count {got} != selector count {expected}")]
    ArmCountMismatch {
        component: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("selective snapshot arm {arm} {component} length {got} != field count {expected}")]
    ArmFieldCountMismatch {
        arm: usize,
        component: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("selective snapshot public field count {public_field_count} is outside 1..={fields} for arm {arm}")]
    PublicFieldCountOutOfBounds {
        arm: usize,
        public_field_count: usize,
        fields: usize,
    },
    #[error("selective snapshot arm {arm} field {field} {kind} slot [{start}, {end}) is outside 1..{columns}")]
    SlotOutOfBounds {
        arm: usize,
        field: usize,
        kind: &'static str,
        start: usize,
        end: usize,
        columns: usize,
    },
    #[error("selective snapshot arm {arm} field {field} aliases missing source field {source_field}")]
    AliasSourceOutOfBounds {
        arm: usize,
        field: usize,
        source_field: usize,
    },
    #[error("selective snapshot arm {arm} field {field} alias source {source_field} is not earlier")]
    AliasSourceNotEarlier {
        arm: usize,
        field: usize,
        source_field: usize,
    },
    #[error("selective snapshot arm {arm} field {field} aliases bit {bit} outside source field {source_field} width {width}")]
    AliasBitOutOfBounds {
        arm: usize,
        field: usize,
        source_field: usize,
        bit: usize,
        width: usize,
    },
    #[error(
        "selective snapshot arm {arm} field {field} alias slot does not equal source field {source_field} bit {bit}"
    )]
    AliasSlotMismatch {
        arm: usize,
        field: usize,
        source_field: usize,
        bit: usize,
    },
    #[error("selective snapshot arm {arm} field {field} equality source {source_field} is outside 0..{fields}")]
    EqualitySourceOutOfBounds {
        arm: usize,
        field: usize,
        source_field: usize,
        fields: usize,
    },
    #[error("selective snapshot arm {arm} field {field} equality source {source_field} is not earlier")]
    EqualitySourceNotEarlier {
        arm: usize,
        field: usize,
        source_field: usize,
    },
    #[error("selective snapshot arm {arm} field {field} does not reuse equality source {source_field}'s exact slot")]
    EqualitySlotMismatch {
        arm: usize,
        field: usize,
        source_field: usize,
    },
    #[error("selective snapshot arm {arm} centered field {field} does not have a one-coordinate slot")]
    CenteredSlotWidth { arm: usize, field: usize },
    #[error("selective snapshot arm {arm} field {field} violates compiler invariant: {reason}")]
    FieldInvariant {
        arm: usize,
        field: usize,
        reason: &'static str,
    },
    #[error(
        "selective snapshot arm {arm} {kind} {field} starts at {actual}, expected the next owned coordinate {expected}"
    )]
    SlotPlacementMismatch {
        arm: usize,
        field: usize,
        kind: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("selective snapshot arm {arm} derived value {derived} has {got} factors, expected {expected}")]
    DerivedFactorCount {
        arm: usize,
        derived: usize,
        got: usize,
        expected: usize,
    },
    #[error("selective snapshot arm {arm} derived value {derived} predecessor {previous} is not earlier")]
    DerivedPreviousOutOfBounds {
        arm: usize,
        derived: usize,
        previous: usize,
    },
    #[error("selective snapshot arm {arm} derived value {derived} {side} factor {factor} references source column {column} outside 0..{fields}")]
    LinearCombinationSourceOutOfBounds {
        arm: usize,
        derived: usize,
        factor: usize,
        side: &'static str,
        column: usize,
        fields: usize,
    },
    #[error("selective snapshot shared source prefix ends at field {got} in arm {arm}, expected field {expected}")]
    SharedFieldCountMismatch {
        arm: usize,
        got: usize,
        expected: usize,
    },
    #[error("selective snapshot branch layout ends at {layout_end}, but the largest arm ends at {arm_end}")]
    BranchExtentMismatch { layout_end: usize, arm_end: usize },
}

impl MultiBranchLowNormR1cs {
    /// Take a checked, read-only snapshot of a selectively compiled encoder.
    pub fn selective_snapshot(&self) -> Result<SelectiveLowNormSnapshot<'_>, SelectiveSnapshotError> {
        let compiler_audit = self
            .selective_compiler_audit
            .as_ref()
            .ok_or(SelectiveSnapshotError::NotSelective)?;
        let layout = compiler_audit.layout();
        let columns = self.structure.m;
        if columns == 0 {
            return Err(SelectiveSnapshotError::EmptyStructure);
        }
        if self.structure.matrices.len() != SELECTIVE_ARITY || self.structure.f.arity() != SELECTIVE_ARITY {
            return Err(SelectiveSnapshotError::SelectiveArityMismatch {
                matrices: self.structure.matrices.len(),
                polynomial_arity: self.structure.f.arity(),
            });
        }
        if layout.total_columns() != columns {
            return Err(SelectiveSnapshotError::LayoutColumnMismatch {
                layout_columns: layout.total_columns(),
                structure_columns: columns,
            });
        }
        if layout.public_input_len() != self.public_input_len {
            return Err(SelectiveSnapshotError::LayoutPublicInputMismatch {
                layout_len: layout.public_input_len(),
                encoder_len: self.public_input_len,
            });
        }
        if layout.selector_columns() != self.selector_cols.as_slice() {
            return Err(SelectiveSnapshotError::LayoutSelectorMismatch {
                layout: layout.selector_columns().to_vec(),
                encoder: self.selector_cols.clone(),
            });
        }
        if self.public_input_len > columns {
            return Err(SelectiveSnapshotError::PublicInputOutOfBounds {
                public_input_len: self.public_input_len,
                columns,
            });
        }

        let arms = self.selector_cols.len();
        for (component, got) in [
            ("slot", self.arm_slots.len()),
            ("bit-alias", self.arm_aliases.len()),
            ("equality-source", self.arm_equal_aliases.len()),
            ("centered-flag", self.arm_centered_columns.len()),
            ("derived-product-sum", self.arm_derived_product_sums.len()),
        ] {
            if got != arms {
                return Err(SelectiveSnapshotError::ArmCountMismatch {
                    component,
                    got,
                    expected: arms,
                });
            }
        }
        validate_layout(self, layout, columns)?;

        let mut expected_shared_field_end = None;
        let mut largest_arm_end = layout.branch_columns().start;
        for arm in 0..arms {
            let fields = self.arm_slots[arm].len();
            for (component, got) in [
                ("bit-alias", self.arm_aliases[arm].len()),
                ("equality-source", self.arm_equal_aliases[arm].len()),
                ("centered-flag", self.arm_centered_columns[arm].len()),
            ] {
                if got != fields {
                    return Err(SelectiveSnapshotError::ArmFieldCountMismatch {
                        arm,
                        component,
                        got,
                        expected: fields,
                    });
                }
            }
            if self.public_field_count == 0 || self.public_field_count > fields {
                return Err(SelectiveSnapshotError::PublicFieldCountOutOfBounds {
                    arm,
                    public_field_count: self.public_field_count,
                    fields,
                });
            }
            let (arm_end, shared_field_end) = validate_arm(self, layout, arm, columns)?;
            if let Some(expected) = expected_shared_field_end {
                if shared_field_end != expected {
                    return Err(SelectiveSnapshotError::SharedFieldCountMismatch {
                        arm,
                        got: shared_field_end,
                        expected,
                    });
                }
                validate_shared_source_plan(self, arm, shared_field_end)?;
            } else {
                expected_shared_field_end = Some(shared_field_end);
            }
            largest_arm_end = largest_arm_end.max(arm_end);
        }
        if largest_arm_end != layout.branch_columns().end {
            return Err(SelectiveSnapshotError::BranchExtentMismatch {
                layout_end: layout.branch_columns().end,
                arm_end: largest_arm_end,
            });
        }

        Ok(SelectiveLowNormSnapshot {
            relation: self,
            compiler_audit,
        })
    }
}

fn replay_write_encoded_value(
    assignment: &mut [F],
    slot: Option<(usize, usize)>,
    alias: Option<(usize, usize)>,
    centered: bool,
    value: F,
    field_col: usize,
) -> Result<(), LowNormR1csError> {
    let (start, width) = slot.expect("validated snapshot field has an encoded slot");
    if centered {
        assignment[start] = value;
        return Ok(());
    }
    if width == super::BALANCED_TERNARY_FIELD_WIDTH {
        return replay_write_balanced_ternary(assignment, start, value, field_col);
    }
    let canonical = value.as_canonical_u64();
    if width < BINARY_FIELD_WIDTH && canonical >= (1u64 << width) {
        return Err(LowNormR1csError::InferredWidthViolation {
            col: field_col,
            width,
            value: canonical,
        });
    }
    if let Some((source_col, bit)) = alias {
        if assignment[start] != F::from_u64(canonical) {
            return Err(LowNormR1csError::AliasedBitMismatch {
                field_col: source_col,
                bit_col: field_col,
                bit,
            });
        }
        return Ok(());
    }
    for bit in 0..width {
        assignment[start + bit] = F::from_u64((canonical >> bit) & 1);
    }
    Ok(())
}

fn replay_write_balanced_ternary(
    assignment: &mut [F],
    start: usize,
    value: F,
    field_col: usize,
) -> Result<(), LowNormR1csError> {
    let modulus = F::ORDER_U64;
    let canonical = value.as_canonical_u64();
    let negative = canonical > modulus / 2;
    let mut remaining = if negative { modulus - canonical } else { canonical };
    for digit_index in 0..super::BALANCED_TERNARY_FIELD_WIDTH {
        let residue = remaining % 3;
        remaining /= 3;
        let digit = match residue {
            0 => F::ZERO,
            1 => F::ONE,
            2 => {
                remaining += 1;
                -F::ONE
            }
            _ => unreachable!("residue modulo three"),
        };
        assignment[start + digit_index] = if negative { -digit } else { digit };
    }
    if remaining != 0 {
        return Err(LowNormR1csError::BalancedTernaryOverflow { col: field_col });
    }
    Ok(())
}

fn validate_layout(
    relation: &MultiBranchLowNormR1cs,
    layout: &SelectiveLayoutAudit,
    columns: usize,
) -> Result<(), SelectiveSnapshotError> {
    let logical_public = layout.logical_public_input_len();
    if logical_public == 0 || logical_public > relation.public_input_len {
        return Err(SelectiveSnapshotError::LayoutPartitionMismatch {
            component: "logical public prefix",
        });
    }
    if relation.public_field_count != logical_public {
        return Err(SelectiveSnapshotError::PublicFieldLayoutMismatch {
            public_field_count: relation.public_field_count,
            logical_public_input_len: logical_public,
        });
    }
    if !is_exact_range(
        layout.public_padding_columns(),
        logical_public,
        relation.public_input_len,
    ) {
        return Err(SelectiveSnapshotError::LayoutPartitionMismatch {
            component: "public padding",
        });
    }
    let selector_end = relation
        .public_input_len
        .checked_add(relation.selector_cols.len())
        .filter(|&end| end <= columns)
        .ok_or(SelectiveSnapshotError::LayoutPartitionMismatch {
            component: "selector range",
        })?;
    if !is_exact_range(layout.selector_columns(), relation.public_input_len, selector_end) {
        return Err(SelectiveSnapshotError::LayoutPartitionMismatch { component: "selectors" });
    }

    let shared = layout.shared_private_columns();
    let branch = layout.branch_columns();
    let ring = layout.ring_alignment_padding_columns();
    if !is_exact_range(layout.private_alignment_padding_columns(), selector_end, shared.start) {
        return Err(SelectiveSnapshotError::LayoutPartitionMismatch {
            component: "private alignment padding",
        });
    }
    if shared.start > shared.end || shared.end != branch.start {
        return Err(SelectiveSnapshotError::LayoutPartitionMismatch {
            component: "shared/branch boundary",
        });
    }
    if branch.start > branch.end || branch.end != ring.start || ring.start > ring.end || ring.end != columns {
        return Err(SelectiveSnapshotError::LayoutPartitionMismatch {
            component: "branch/ring boundary",
        });
    }
    let expected_ring_end = if ring.start % D == 0 {
        ring.start
    } else {
        ring.start
            .checked_add(D - ring.start % D)
            .ok_or(SelectiveSnapshotError::LayoutPartitionMismatch {
                component: "ring alignment padding",
            })?
    };
    if ring.end != expected_ring_end {
        return Err(SelectiveSnapshotError::LayoutPartitionMismatch {
            component: "ring alignment padding",
        });
    }
    Ok(())
}

fn is_exact_range(columns: &[usize], start: usize, end: usize) -> bool {
    end >= start
        && columns.len() == end - start
        && columns
            .iter()
            .enumerate()
            .all(|(offset, &column)| column == start + offset)
}

fn validate_arm(
    relation: &MultiBranchLowNormR1cs,
    layout: &SelectiveLayoutAudit,
    arm: usize,
    columns: usize,
) -> Result<(usize, usize), SelectiveSnapshotError> {
    let slots = &relation.arm_slots[arm];
    let aliases = &relation.arm_aliases[arm];
    let equalities = &relation.arm_equal_aliases[arm];
    let centered = &relation.arm_centered_columns[arm];
    let fields = slots.len();
    if slots[0].get().is_some() || aliases[0].get().is_some() || equalities[0].get().is_some() || centered[0] {
        return Err(SelectiveSnapshotError::FieldInvariant {
            arm,
            field: 0,
            reason: "constant-one field owns encoding metadata",
        });
    }
    for field in 1..relation.public_field_count {
        if slots[field].get() != Some((field, 1))
            || aliases[field].get().is_some()
            || equalities[field].get().is_some()
            || centered[field]
        {
            return Err(SelectiveSnapshotError::FieldInvariant {
                arm,
                field,
                reason: "public bit field does not own its canonical prefix coordinate",
            });
        }
    }

    let shared = layout.shared_private_columns();
    let branch = layout.branch_columns();
    let mut shared_cursor = shared.start;
    let mut branch_cursor = branch.start;
    let mut shared_field_end = (shared.start == shared.end).then_some(relation.public_field_count);
    for field in relation.public_field_count..fields {
        let slot = slots[field]
            .get()
            .map(|(start, len)| EncodedSlotSnapshot { start, len });
        if let Some(slot) = slot {
            validate_slot(arm, field, "field", slot.start, slot.len, columns)?;
            if !is_supported_source_width(slot.len) {
                return Err(SelectiveSnapshotError::FieldInvariant {
                    arm,
                    field,
                    reason: "source slot width is not 1, balanced-ternary, or binary",
                });
            }
        }
        let alias = aliases[field].get();
        let equality = equalities[field].get();
        if alias.is_some() && equality.is_some() {
            return Err(SelectiveSnapshotError::FieldInvariant {
                arm,
                field,
                reason: "field has both decomposition and equality aliases",
            });
        }
        if let Some((source_field, bit)) = alias {
            if shared_cursor != shared.end {
                return Err(SelectiveSnapshotError::FieldInvariant {
                    arm,
                    field,
                    reason: "alias appears before the shared source prefix is complete",
                });
            }
            if source_field >= fields {
                return Err(SelectiveSnapshotError::AliasSourceOutOfBounds {
                    arm,
                    field,
                    source_field,
                });
            }
            if source_field >= field {
                return Err(SelectiveSnapshotError::AliasSourceNotEarlier {
                    arm,
                    field,
                    source_field,
                });
            }
            let Some((source_start, source_width)) = slots[source_field].get() else {
                return Err(SelectiveSnapshotError::AliasSourceOutOfBounds {
                    arm,
                    field,
                    source_field,
                });
            };
            if !matches!(source_width, super::BALANCED_TERNARY_FIELD_WIDTH | BINARY_FIELD_WIDTH) {
                return Err(SelectiveSnapshotError::FieldInvariant {
                    arm,
                    field,
                    reason: "decomposition source is not a full field encoding",
                });
            }
            if bit >= source_width {
                return Err(SelectiveSnapshotError::AliasBitOutOfBounds {
                    arm,
                    field,
                    source_field,
                    bit,
                    width: source_width,
                });
            }
            if slot
                != Some(EncodedSlotSnapshot {
                    start: source_start + bit,
                    len: 1,
                })
            {
                return Err(SelectiveSnapshotError::AliasSlotMismatch {
                    arm,
                    field,
                    source_field,
                    bit,
                });
            }
            if centered[field] != (source_width == super::BALANCED_TERNARY_FIELD_WIDTH) {
                return Err(SelectiveSnapshotError::FieldInvariant {
                    arm,
                    field,
                    reason: "decomposition child domain does not match its source encoding",
                });
            }
            continue;
        }
        if let Some(source) = equality {
            if shared_cursor != shared.end {
                return Err(SelectiveSnapshotError::FieldInvariant {
                    arm,
                    field,
                    reason: "equality alias appears before the shared source prefix is complete",
                });
            }
            if source >= fields {
                return Err(SelectiveSnapshotError::EqualitySourceOutOfBounds {
                    arm,
                    field,
                    source_field: source,
                    fields,
                });
            }
            if source >= field {
                return Err(SelectiveSnapshotError::EqualitySourceNotEarlier {
                    arm,
                    field,
                    source_field: source,
                });
            }
            if slot.is_none()
                || slot
                    != slots[source]
                        .get()
                        .map(|(start, len)| EncodedSlotSnapshot { start, len })
            {
                return Err(SelectiveSnapshotError::EqualitySlotMismatch {
                    arm,
                    field,
                    source_field: source,
                });
            }
            if centered[field] != centered[source] {
                return Err(SelectiveSnapshotError::FieldInvariant {
                    arm,
                    field,
                    reason: "equality alias changes the source field domain",
                });
            }
            continue;
        }
        let Some(slot) = slot else {
            if shared_cursor != shared.end {
                return Err(SelectiveSnapshotError::FieldInvariant {
                    arm,
                    field,
                    reason: "eliminated field appears before the shared source prefix is complete",
                });
            }
            continue;
        };
        if centered[field] && slot.len != 1 {
            return Err(SelectiveSnapshotError::CenteredSlotWidth { arm, field });
        }
        if shared_cursor != shared.end {
            if slot.start != shared_cursor {
                return Err(SelectiveSnapshotError::SlotPlacementMismatch {
                    arm,
                    field,
                    kind: "shared field",
                    actual: slot.start,
                    expected: shared_cursor,
                });
            }
            let end = slot.start + slot.len;
            if end > shared.end {
                return Err(SelectiveSnapshotError::FieldInvariant {
                    arm,
                    field,
                    reason: "shared field crosses into the branch arena",
                });
            }
            shared_cursor = end;
            if shared_cursor == shared.end {
                shared_field_end = Some(field + 1);
            }
        } else {
            if slot.start != branch_cursor {
                return Err(SelectiveSnapshotError::SlotPlacementMismatch {
                    arm,
                    field,
                    kind: "branch field",
                    actual: slot.start,
                    expected: branch_cursor,
                });
            }
            let end = slot.start + slot.len;
            if end > branch.end {
                return Err(SelectiveSnapshotError::FieldInvariant {
                    arm,
                    field,
                    reason: "branch field crosses the branch arena",
                });
            }
            branch_cursor = end;
        }
    }
    let shared_field_end = shared_field_end.ok_or(SelectiveSnapshotError::FieldInvariant {
        arm,
        field: fields,
        reason: "shared source fields do not cover the shared coordinate range",
    })?;

    for (derived_index, derived) in relation.arm_derived_product_sums[arm].iter().enumerate() {
        validate_slot(arm, derived_index, "derived", derived.slot.0, derived.slot.1, columns)?;
        if derived.slot.1 != super::BALANCED_TERNARY_FIELD_WIDTH {
            return Err(SelectiveSnapshotError::FieldInvariant {
                arm,
                field: derived_index,
                reason: "derived product sum does not use one balanced field slot",
            });
        }
        if derived.slot.0 != branch_cursor {
            return Err(SelectiveSnapshotError::SlotPlacementMismatch {
                arm,
                field: derived_index,
                kind: "derived value",
                actual: derived.slot.0,
                expected: branch_cursor,
            });
        }
        branch_cursor += derived.slot.1;
        if branch_cursor > branch.end {
            return Err(SelectiveSnapshotError::FieldInvariant {
                arm,
                field: derived_index,
                reason: "derived product sum crosses the branch arena",
            });
        }
        if derived.factors.len() != EVAL_GROUP_SIZE {
            return Err(SelectiveSnapshotError::DerivedFactorCount {
                arm,
                derived: derived_index,
                got: derived.factors.len(),
                expected: EVAL_GROUP_SIZE,
            });
        }
        if let Some(previous) = derived.previous {
            if previous >= derived_index {
                return Err(SelectiveSnapshotError::DerivedPreviousOutOfBounds {
                    arm,
                    derived: derived_index,
                    previous,
                });
            }
        }
        for (factor_index, factor) in derived.factors.iter().enumerate() {
            validate_lc(arm, derived_index, factor_index, "left", &factor.left, fields)?;
            validate_lc(arm, derived_index, factor_index, "right", &factor.right, fields)?;
        }
    }
    Ok((branch_cursor, shared_field_end))
}

fn validate_shared_source_plan(
    relation: &MultiBranchLowNormR1cs,
    arm: usize,
    shared_field_end: usize,
) -> Result<(), SelectiveSnapshotError> {
    for field in 0..shared_field_end {
        if relation.arm_slots[arm][field].get() != relation.arm_slots[0][field].get()
            || relation.arm_aliases[arm][field].get() != relation.arm_aliases[0][field].get()
            || relation.arm_equal_aliases[arm][field].get() != relation.arm_equal_aliases[0][field].get()
            || relation.arm_centered_columns[arm][field] != relation.arm_centered_columns[0][field]
        {
            return Err(SelectiveSnapshotError::FieldInvariant {
                arm,
                field,
                reason: "shared source encoding differs from arm zero",
            });
        }
    }
    Ok(())
}

fn is_supported_source_width(width: usize) -> bool {
    matches!(width, 1 | super::BALANCED_TERNARY_FIELD_WIDTH | BINARY_FIELD_WIDTH)
}

fn validate_slot(
    arm: usize,
    field: usize,
    kind: &'static str,
    start: usize,
    len: usize,
    columns: usize,
) -> Result<(), SelectiveSnapshotError> {
    let end = start.checked_add(len).unwrap_or(usize::MAX);
    if start == 0 || len == 0 || end > columns {
        return Err(SelectiveSnapshotError::SlotOutOfBounds {
            arm,
            field,
            kind,
            start,
            end,
            columns,
        });
    }
    Ok(())
}

fn validate_lc(
    arm: usize,
    derived: usize,
    factor: usize,
    side: &'static str,
    lc: &crate::engine::r1cs_circuit::Lc,
    fields: usize,
) -> Result<(), SelectiveSnapshotError> {
    for &(column, _) in &lc.terms {
        if column >= fields {
            return Err(SelectiveSnapshotError::LinearCombinationSourceOutOfBounds {
                arm,
                derived,
                factor,
                side,
                column,
                fields,
            });
        }
    }
    Ok(())
}
