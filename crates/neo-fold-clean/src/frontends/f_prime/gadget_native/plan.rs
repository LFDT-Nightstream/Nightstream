//! Executable inverse map from the low-norm assignment to source R1CS.
//!
//! Owns: the public read-only plan API and exact source-witness decoding.
//!
//! Does not own: source-role classification, assignment allocation, constraint
//! emission, or CCS satisfaction checks.
//!
//! Emits constraints: no.
//!
//! Authority boundary: decoding is available only after the parent relation
//! checks satisfaction; projected temporaries are recomputed in source order.
//!
//! | Decoder node | Reconstruction | Authority |
//! |---|---|---|
//! | encoded slot | slot-specific linear decode with alphabet validation | committed assignment |
//! | linear node | exact prior-source linear combination | derived |
//! | product node | exact prior-source product | derived |
//! | inverse node | canonical inverse of validated difference | derived |

use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

use super::model::{RingSyntheticSlots, SourceColumn};
use super::slots::ValueEncoding;
use super::{
    acceptance, decode_slot, mod5, product_sum, BalancedTernarySharedSlotPlan, GadgetNativeCoordinateGateSchedule,
    GadgetNativeError, GadgetNativeSourceRole, TOOM_COEFFICIENTS,
};

/// Inverse map from the committed low-norm assignment to the complete source
/// R1CS witness, including every projected product wire.
#[derive(Clone, Debug)]
pub struct GadgetNativePlan {
    pub(super) source_columns: Vec<SourceColumn>,
    pub(super) source_roles: Vec<GadgetNativeSourceRole>,
    pub(super) ring_slots: Vec<RingSyntheticSlots>,
    pub(super) product_sum_slots: product_sum::ProductSumSlots,
    pub(super) acceptance_slots: acceptance::AcceptanceSlots,
    pub(super) mod5_slots: mod5::PackedMod5Slots,
    pub(super) balanced_ternary_openings: Vec<BalancedTernarySharedSlotPlan>,
    pub(super) public_columns: Vec<usize>,
    pub(super) public_input_len: usize,
    pub(super) encoded_cols: usize,
    pub(super) coordinate_gates: GadgetNativeCoordinateGateSchedule,
    pub(super) acceptance_translated_boolean_rows: Vec<(usize, usize)>,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub enum GadgetNativePlanTestMutation {
    OrdinaryAsSis { source_column: usize },
}

impl GadgetNativePlan {
    pub(super) fn validate_materialization_contract(&self) -> Result<(), GadgetNativeError> {
        for (column, (&role, definition)) in self
            .source_roles
            .iter()
            .zip(&self.source_columns)
            .enumerate()
        {
            let exact = match (role, definition) {
                (GadgetNativeSourceRole::ConstantOne, SourceColumn::One) => true,
                (
                    GadgetNativeSourceRole::PublicBit | GadgetNativeSourceRole::PrivateBoolean,
                    SourceColumn::Encoded(slot),
                ) => slot.width == 1 && slot.encoding == ValueEncoding::Boolean,
                (GadgetNativeSourceRole::OrdinaryPrivateField, SourceColumn::Encoded(slot)) => {
                    slot.width == super::ORDINARY_PRIVATE_DIGITS
                        && slot.encoding == ValueEncoding::OrdinaryCenteredTernary
                }
                (GadgetNativeSourceRole::CanonicalU64, SourceColumn::Encoded(slot)) => {
                    matches!(slot.encoding, ValueEncoding::CanonicalBinary { .. })
                }
                (GadgetNativeSourceRole::SisOpening, SourceColumn::Encoded(slot)) => {
                    slot.encoding == ValueEncoding::BalancedTernary
                }
                (GadgetNativeSourceRole::StructuralBalancedAlias, SourceColumn::Encoded(slot)) => {
                    slot.width == 1 && slot.encoding == ValueEncoding::CenteredUnit
                }
                (GadgetNativeSourceRole::LinearlyDerived, SourceColumn::Linear(_) | SourceColumn::EncodedLinear(_))
                | (GadgetNativeSourceRole::GadgetDerived, SourceColumn::GadgetLinear(_))
                | (GadgetNativeSourceRole::ProductDerived, SourceColumn::Product(_))
                | (GadgetNativeSourceRole::GadgetTemporary, SourceColumn::CanonicalNonzeroInverse(_)) => true,
                _ => false,
            };
            if !exact {
                return Err(GadgetNativeError::SourceMaterializationMismatch { column });
            }
            match definition {
                SourceColumn::Linear(definition) if definition.source_row.is_none() => {
                    return Err(GadgetNativeError::SourceMaterializationMismatch { column });
                }
                SourceColumn::GadgetLinear(definition) if definition.source_row.is_some() => {
                    return Err(GadgetNativeError::SourceMaterializationMismatch { column });
                }
                _ => {}
            }
        }
        Ok(())
    }

    #[doc(hidden)]
    pub fn apply_test_mutation(&mut self, mutation: GadgetNativePlanTestMutation) {
        match mutation {
            GadgetNativePlanTestMutation::OrdinaryAsSis { source_column } => {
                let SourceColumn::Encoded(slot) = &mut self.source_columns[source_column] else {
                    panic!("test mutation requires an encoded source column")
                };
                slot.encoding = ValueEncoding::BalancedTernary;
            }
        }
    }

    #[doc(hidden)]
    pub fn validate_materialization_for_test(&self) -> Result<(), GadgetNativeError> {
        self.validate_materialization_contract()
    }

    pub fn public_columns(&self) -> &[usize] {
        &self.public_columns
    }

    pub fn public_input_len(&self) -> usize {
        self.public_input_len
    }

    pub fn encoded_cols(&self) -> usize {
        self.encoded_cols
    }

    pub fn coordinate_gate_schedule(&self) -> &GadgetNativeCoordinateGateSchedule {
        &self.coordinate_gates
    }

    pub fn encoded_range_for_source_column(&self, column: usize) -> Option<std::ops::Range<usize>> {
        match self.source_columns.get(column) {
            Some(SourceColumn::Encoded(slot)) => Some(slot.start..slot.start + slot.width),
            _ => None,
        }
    }

    pub fn source_role_for_column(&self, column: usize) -> Option<GadgetNativeSourceRole> {
        self.source_roles.get(column).copied()
    }

    pub fn is_gadget_derived(&self, column: usize) -> bool {
        matches!(
            self.source_columns.get(column),
            Some(SourceColumn::GadgetLinear(_) | SourceColumn::Product(_) | SourceColumn::CanonicalNonzeroInverse(_))
        )
    }

    pub fn synthetic_ring_coefficient_range(
        &self,
        ring: usize,
        evaluation: usize,
        coefficient: usize,
    ) -> Option<std::ops::Range<usize>> {
        let slot = self.ring_slots.get(ring)?.coefficients.get(
            evaluation
                .checked_mul(TOOM_COEFFICIENTS)?
                .checked_add(coefficient)?,
        )?;
        Some(slot.start..slot.start + slot.width)
    }

    pub fn synthetic_product_sum_field_range(
        &self,
        batch: usize,
        identity: usize,
        carry: usize,
    ) -> Option<std::ops::Range<usize>> {
        self.product_sum_slots.field_range(batch, identity, carry)
    }

    pub fn first_synthetic_product_sum_field_range(&self) -> Option<std::ops::Range<usize>> {
        self.product_sum_slots.first_field_range()
    }

    /// Reconstruct every source column in exact allocation order.
    pub fn decode_source(&self, encoded: &[F]) -> Result<Vec<F>, GadgetNativeError> {
        if encoded.len() != self.encoded_cols {
            return Err(GadgetNativeError::EncodedLength {
                expected: self.encoded_cols,
                got: encoded.len(),
            });
        }
        if encoded.first().copied() != Some(F::ONE) {
            return Err(GadgetNativeError::EncodedConstantOne);
        }
        let mut source = vec![F::ZERO; self.source_columns.len()];
        for (column, definition) in self.source_columns.iter().enumerate() {
            source[column] = match definition {
                SourceColumn::One => F::ONE,
                SourceColumn::Encoded(slot) => decode_slot(*slot, column, encoded)?,
                SourceColumn::EncodedLinear(terms) => terms.iter().fold(F::ZERO, |value, &(input, coefficient)| {
                    value + coefficient * encoded[input]
                }),
                SourceColumn::Linear(definition) | SourceColumn::GadgetLinear(definition) => definition
                    .terms
                    .iter()
                    .fold(F::ZERO, |value, &(input, coefficient)| {
                        value + coefficient * source[input]
                    }),
                SourceColumn::Product(definition) => {
                    super::eval_lc_from_source(&definition.left, &source)
                        * super::eval_lc_from_source(&definition.right, &source)
                }
                SourceColumn::CanonicalNonzeroInverse(difference) => {
                    let difference = super::eval_lc_from_source(difference, &source);
                    if difference == F::ZERO {
                        F::ZERO
                    } else {
                        difference.inverse()
                    }
                }
            };
        }
        Ok(source)
    }
}
