//! Exact source-loop coordinate allocation for gadget-native lowering.
//!
//! Owns: the public-bit prefix boundary, fixed allocation width of every
//! validated source role, the observing production cursor, and the compact
//! ordinary-private placement audit.
//!
//! Does not own: source-role classification, slot construction, deferred
//! gadget allocation, coordinate gates, constraint emission, CE ownership,
//! or permission to remove rows.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the validated source schedule chooses each role. The
//! cursor only checks where the existing materializer places source-loop
//! coordinates; generated metadata remains non-authoritative drift evidence.
//!
//! | Source role family | Allocation equation | Returned value width |
//! |---|---:|---:|
//! | private Boolean | `next = cursor + 1` | 1 |
//! | ordinary private field | `next = cursor + 41` | 41 |
//! | direct canonical-u64 | `next = cursor + 64 + 31` | 64 |
//! | SIS balanced opening | `next = cursor + 41` | 41 |
//! | constant/public/alias/derived | `next = cursor` | none in source loop |

use std::ops::Range;

use crate::engine::r1cs_circuit::{R1csEncodingTrace, R1csSnapshot};

use super::slots::{ValueEncoding, ValueSlot};
use super::source_schedule::{
    CanonicalFieldKind, GadgetNativeSourceRole, SourceColumnDecision, ValidatedSourceSchedule,
};
use super::{
    estimate_r1cs_gadget_native, GadgetNativeError, BALANCED_TERNARY_DIGITS, CANONICAL_SLOT_WIDTH,
    ORDINARY_PRIVATE_DIGITS,
};

/// Exact number of fresh assignment coordinates allocated while the source
/// loop visits one column of `role`. Public bits are already in the prefix.
#[doc(hidden)]
pub const fn gadget_native_source_loop_width(role: GadgetNativeSourceRole) -> usize {
    match role {
        GadgetNativeSourceRole::PrivateBoolean => 1,
        GadgetNativeSourceRole::OrdinaryPrivateField => ORDINARY_PRIVATE_DIGITS,
        GadgetNativeSourceRole::CanonicalU64 => CANONICAL_SLOT_WIDTH,
        GadgetNativeSourceRole::SisOpening => BALANCED_TERNARY_DIGITS,
        GadgetNativeSourceRole::ConstantOne
        | GadgetNativeSourceRole::PublicBit
        | GadgetNativeSourceRole::LinearlyDerived
        | GadgetNativeSourceRole::StructuralBalancedAlias
        | GadgetNativeSourceRole::GadgetDerived
        | GadgetNativeSourceRole::ProductDerived
        | GadgetNativeSourceRole::GadgetTemporary => 0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SourceAllocationStep {
    source_column: usize,
    encoded_start: usize,
    allocated_width: usize,
    role: GadgetNativeSourceRole,
}

impl SourceAllocationStep {
    pub(super) fn encoded_range(self) -> Range<usize> {
        self.encoded_start..self.encoded_start + self.allocated_width
    }

    fn ordinary_placement(self) -> Option<GadgetNativeOrdinaryPlacement> {
        (self.role == GadgetNativeSourceRole::OrdinaryPrivateField).then_some(GadgetNativeOrdinaryPlacement {
            source_column: self.source_column,
            encoded_start: self.encoded_start,
        })
    }

    pub(super) fn check_observed(self, actual_start: usize, actual_end: usize) -> Result<(), GadgetNativeError> {
        let expected = self.encoded_range();
        if actual_start != expected.start || actual_end != expected.end {
            return Err(GadgetNativeError::SourceAllocationMismatch {
                column: self.source_column,
                expected_start: expected.start,
                expected_end: expected.end,
                actual_start,
                actual_end,
            });
        }
        Ok(())
    }
}

/// Visit every source-loop-owned slot using only the validated decision
/// schedule and the production allocation cursor. No witness coordinate or
/// constraint row is materialized. Structural aliases revisit their parent's
/// exact coordinate rather than allocating another one.
pub(super) fn visit_planned_source_slots(
    schedule: &ValidatedSourceSchedule,
    public_bit_columns: &[usize],
    mut visit: impl FnMut(usize, ValueSlot) -> Result<(), GadgetNativeError>,
) -> Result<usize, GadgetNativeError> {
    let public_input_len = super::canonical_superneo_public_input_len(public_bit_columns.len())?;
    let public_starts = public_bit_columns
        .iter()
        .copied()
        .enumerate()
        .map(|(offset, column)| (column, 1 + offset))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut balanced_slots = std::collections::BTreeMap::<usize, ValueSlot>::new();
    let mut cursor = SourceAllocationCursor::new(public_input_len);

    for (column, decision) in schedule.decisions().iter().enumerate() {
        let step = cursor.step(column, decision.role())?;
        let range = step.encoded_range();
        let slot = match decision {
            SourceColumnDecision::ConstantOne => None,
            SourceColumnDecision::PublicBit => {
                let start = public_starts
                    .get(&column)
                    .copied()
                    .ok_or(GadgetNativeError::SourceMaterializationMismatch { column })?;
                Some(ValueSlot {
                    start,
                    width: 1,
                    encoding: ValueEncoding::Boolean,
                })
            }
            SourceColumnDecision::PrivateBoolean(_) => Some(ValueSlot {
                start: range.start,
                width: 1,
                encoding: ValueEncoding::Boolean,
            }),
            SourceColumnDecision::BalancedOpening { .. } => {
                let slot = ValueSlot {
                    start: range.start,
                    width: super::BALANCED_TERNARY_DIGITS,
                    encoding: ValueEncoding::BalancedTernary,
                };
                if balanced_slots.insert(column, slot).is_some() {
                    return Err(GadgetNativeError::SourceMaterializationMismatch { column });
                }
                Some(slot)
            }
            SourceColumnDecision::BalancedDigitAlias { field, digit } => {
                let parent = balanced_slots
                    .get(field)
                    .copied()
                    .ok_or(GadgetNativeError::SourceMaterializationMismatch { column })?;
                Some(ValueSlot::centered_alias(parent, *digit))
            }
            SourceColumnDecision::CanonicalField(CanonicalFieldKind::OrdinaryPrivate) => Some(ValueSlot {
                start: range.start,
                width: super::ordinary_private_field::ORDINARY_PRIVATE_DIGITS,
                encoding: ValueEncoding::OrdinaryCenteredTernary,
            }),
            SourceColumnDecision::CanonicalField(CanonicalFieldKind::DirectCanonicalU64) => Some(ValueSlot {
                start: range.start,
                width: super::FIELD_BITS,
                encoding: ValueEncoding::CanonicalBinary {
                    auxiliary_start: range.start + super::FIELD_BITS,
                },
            }),
            SourceColumnDecision::GenericLinear(_) | SourceColumnDecision::Projected(_) => None,
        };
        if let Some(slot) = slot {
            visit(column, slot)?;
        }
    }
    Ok(cursor.source_phase_end())
}

/// Allocation-only cursor shared by production materialization and the audit
/// generator. It deliberately has no access to witness values or row emitters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SourceAllocationCursor {
    next_encoded_column: usize,
}

impl SourceAllocationCursor {
    pub(super) const fn new(public_input_len: usize) -> Self {
        Self {
            next_encoded_column: public_input_len,
        }
    }

    pub(super) fn step(
        &mut self,
        source_column: usize,
        role: GadgetNativeSourceRole,
    ) -> Result<SourceAllocationStep, GadgetNativeError> {
        let encoded_start = self.next_encoded_column;
        let allocated_width = gadget_native_source_loop_width(role);
        self.next_encoded_column = encoded_start
            .checked_add(allocated_width)
            .ok_or(GadgetNativeError::SourceAllocationOverflow { column: source_column })?;
        Ok(SourceAllocationStep {
            source_column,
            encoded_start,
            allocated_width,
            role,
        })
    }

    pub(super) const fn source_phase_end(self) -> usize {
        self.next_encoded_column
    }

    pub(super) fn check_phase_end(self, actual: usize) -> Result<(), GadgetNativeError> {
        if actual != self.next_encoded_column {
            return Err(GadgetNativeError::SourceAllocationPhaseEnd {
                expected: self.next_encoded_column,
                actual,
            });
        }
        Ok(())
    }
}

/// One production-derived ordinary-private 41-coordinate word start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GadgetNativeOrdinaryPlacement {
    source_column: usize,
    encoded_start: usize,
}

impl GadgetNativeOrdinaryPlacement {
    pub fn source_column(self) -> usize {
        self.source_column
    }

    pub fn encoded_range(self) -> Range<usize> {
        self.encoded_start..self.encoded_start + ORDINARY_PRIVATE_DIGITS
    }
}

/// Allocation-only audit artifact. `source_roles` is retained privately so
/// mutations can be rechecked without trusting aggregate counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GadgetNativeOrdinaryPlacementManifest {
    source_columns: usize,
    public_input_len: usize,
    source_phase_end: usize,
    encoded_columns: usize,
    source_roles: Vec<GadgetNativeSourceRole>,
    placements: Vec<GadgetNativeOrdinaryPlacement>,
}

impl GadgetNativeOrdinaryPlacementManifest {
    pub fn source_columns(&self) -> usize {
        self.source_columns
    }

    pub fn public_input_len(&self) -> usize {
        self.public_input_len
    }

    pub fn source_phase_end(&self) -> usize {
        self.source_phase_end
    }

    pub fn encoded_columns(&self) -> usize {
        self.encoded_columns
    }

    pub fn placement_count(&self) -> usize {
        self.placements.len()
    }

    pub fn placement(&self, index: usize) -> Option<GadgetNativeOrdinaryPlacement> {
        self.placements.get(index).copied()
    }

    /// Fresh range consumed by a nonzero-width source-loop allocation. Public
    /// bits are absent because they were allocated in the public prefix.
    #[doc(hidden)]
    pub fn source_loop_allocation_range_for_column(&self, column: usize) -> Option<Range<usize>> {
        if column >= self.source_roles.len() {
            return None;
        }
        let mut cursor = SourceAllocationCursor::new(self.public_input_len);
        for (source_column, &role) in self.source_roles.iter().enumerate() {
            let step = cursor.step(source_column, role).ok()?;
            if source_column == column {
                return (step.allocated_width != 0).then(|| step.encoded_range());
            }
        }
        None
    }

    /// Recompute every placement and branch boundary from the private role
    /// sequence. This is the mutation-test and generator fail-closed gate.
    #[doc(hidden)]
    pub fn validate(&self) -> Result<(), GadgetNativeError> {
        if self.source_columns != self.source_roles.len()
            || self.source_roles.first() != Some(&GadgetNativeSourceRole::ConstantOne)
            || self
                .source_roles
                .iter()
                .filter(|&&role| role == GadgetNativeSourceRole::ConstantOne)
                .count()
                != 1
        {
            return Err(GadgetNativeError::SourceAllocationManifest {
                detail: "source role universe",
            });
        }
        let public_bits = self
            .source_roles
            .iter()
            .filter(|&&role| role == GadgetNativeSourceRole::PublicBit)
            .count();
        if self.public_input_len != super::canonical_superneo_public_input_len(public_bits)? {
            return Err(GadgetNativeError::SourceAllocationManifest {
                detail: "public prefix length",
            });
        }
        let mut cursor = SourceAllocationCursor::new(self.public_input_len);
        let mut expected = Vec::with_capacity(self.placements.len());
        for (column, &role) in self.source_roles.iter().enumerate() {
            if let Some(placement) = cursor.step(column, role)?.ordinary_placement() {
                expected.push(placement);
            }
        }
        if expected != self.placements {
            return Err(GadgetNativeError::SourceAllocationManifest {
                detail: "pointwise ordinary placements",
            });
        }
        if cursor.source_phase_end() != self.source_phase_end {
            return Err(GadgetNativeError::SourceAllocationManifest {
                detail: "source phase end",
            });
        }
        if self.source_phase_end > self.encoded_columns {
            return Err(GadgetNativeError::SourceAllocationManifest {
                detail: "encoded column bound",
            });
        }
        Ok(())
    }

    #[doc(hidden)]
    pub fn apply_test_mutation(&mut self, mutation: GadgetNativeOrdinaryPlacementManifestTestMutation) {
        match mutation {
            GadgetNativeOrdinaryPlacementManifestTestMutation::PlacementSource {
                placement,
                source_column,
            } => {
                self.placements[placement].source_column = source_column;
            }
            GadgetNativeOrdinaryPlacementManifestTestMutation::PlacementStart {
                placement,
                encoded_start,
            } => {
                self.placements[placement].encoded_start = encoded_start;
            }
            GadgetNativeOrdinaryPlacementManifestTestMutation::DropPlacement { placement } => {
                self.placements.remove(placement);
            }
            GadgetNativeOrdinaryPlacementManifestTestMutation::DuplicatePlacement { placement } => {
                let duplicate = self.placements[placement];
                self.placements.insert(placement, duplicate);
            }
            GadgetNativeOrdinaryPlacementManifestTestMutation::PublicInputLen { value } => {
                self.public_input_len = value;
            }
            GadgetNativeOrdinaryPlacementManifestTestMutation::SourcePhaseEnd { value } => {
                self.source_phase_end = value;
            }
            GadgetNativeOrdinaryPlacementManifestTestMutation::EncodedColumns { value } => {
                self.encoded_columns = value;
            }
            GadgetNativeOrdinaryPlacementManifestTestMutation::SourceRole { column, role } => {
                self.source_roles[column] = role;
            }
        }
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub enum GadgetNativeOrdinaryPlacementManifestTestMutation {
    PlacementSource {
        placement: usize,
        source_column: usize,
    },
    PlacementStart {
        placement: usize,
        encoded_start: usize,
    },
    DropPlacement {
        placement: usize,
    },
    DuplicatePlacement {
        placement: usize,
    },
    PublicInputLen {
        value: usize,
    },
    SourcePhaseEnd {
        value: usize,
    },
    EncodedColumns {
        value: usize,
    },
    SourceRole {
        column: usize,
        role: GadgetNativeSourceRole,
    },
}

/// Derive every ordinary-private word start without allocating the encoded
/// witness or any constraint row.
pub fn audit_r1cs_gadget_native_ordinary_placement(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    public_bit_columns: &[usize],
) -> Result<GadgetNativeOrdinaryPlacementManifest, GadgetNativeError> {
    let schedule = ValidatedSourceSchedule::checked(source, trace, public_bit_columns)?;
    let estimate = estimate_r1cs_gadget_native(source, trace, public_bit_columns)?;
    let source_roles = schedule
        .decisions()
        .iter()
        .map(|decision| decision.role())
        .collect::<Vec<_>>();
    let mut cursor = SourceAllocationCursor::new(estimate.public_input_len);
    let mut placements = Vec::with_capacity(estimate.ordinary_private_field_source_cols);
    for (column, &role) in source_roles.iter().enumerate() {
        if let Some(placement) = cursor.step(column, role)?.ordinary_placement() {
            placements.push(placement);
        }
    }
    let manifest = GadgetNativeOrdinaryPlacementManifest {
        source_columns: source.cols(),
        public_input_len: estimate.public_input_len,
        source_phase_end: cursor.source_phase_end(),
        encoded_columns: estimate.encoded_cols,
        source_roles,
        placements,
    };
    manifest.validate()?;
    Ok(manifest)
}
