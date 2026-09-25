//! Shared dense and signed-unit materialization for multi-branch assignments.

use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::paper::nifs::NifsFreshSignedUnitAssignment;

use super::support::eval_source_lc;
use super::{write_encoded_value, LowNormR1csError, MultiBranchLowNormR1cs};

pub(super) trait LowNormAssignmentWriter {
    fn get(&self, index: usize) -> F;
    fn set(&mut self, index: usize, value: F) -> Result<(), LowNormR1csError>;
}

impl LowNormAssignmentWriter for Vec<F> {
    #[inline]
    fn get(&self, index: usize) -> F {
        self[index]
    }

    #[inline]
    fn set(&mut self, index: usize, value: F) -> Result<(), LowNormR1csError> {
        self[index] = value;
        Ok(())
    }
}

struct SignedUnitMaskWriter {
    len: usize,
    positive: Vec<u64>,
    negative: Vec<u64>,
}

impl SignedUnitMaskWriter {
    fn new(len: usize) -> Self {
        let columns = len.div_ceil(D);
        Self {
            len,
            positive: vec![0u64; columns],
            negative: vec![0u64; columns],
        }
    }

    fn finish(self) -> NifsFreshSignedUnitAssignment {
        NifsFreshSignedUnitAssignment::from_masks(self.len, self.positive, self.negative)
    }
}

impl LowNormAssignmentWriter for SignedUnitMaskWriter {
    #[inline]
    fn get(&self, index: usize) -> F {
        let mask = 1u64 << (index % D);
        if self.positive[index / D] & mask != 0 {
            F::ONE
        } else if self.negative[index / D] & mask != 0 {
            -F::ONE
        } else {
            F::ZERO
        }
    }

    #[inline]
    fn set(&mut self, index: usize, value: F) -> Result<(), LowNormR1csError> {
        let word = index / D;
        let mask = 1u64 << (index % D);
        self.positive[word] &= !mask;
        self.negative[word] &= !mask;
        if value == F::ONE {
            self.positive[word] |= mask;
        } else if value == -F::ONE {
            self.negative[word] |= mask;
        } else if value != F::ZERO {
            return Err(LowNormR1csError::PackedNonSignedUnit {
                index,
                value: value.as_canonical_u64(),
            });
        }
        Ok(())
    }
}

impl MultiBranchLowNormR1cs {
    pub fn encode(&self, arm: usize, field_assignment: &[F]) -> Result<Vec<F>, LowNormR1csError> {
        let mut assignment = vec![F::ZERO; self.structure.m];
        self.encode_into(arm, field_assignment, &mut assignment)?;
        Ok(assignment)
    }

    /// Encode directly into the signed-unit masks consumed by accelerator
    /// fresh-instance builders, sharing policy with the dense encoder.
    #[doc(hidden)]
    pub fn encode_signed_unit(
        &self,
        arm: usize,
        field_assignment: &[F],
    ) -> Result<NifsFreshSignedUnitAssignment, LowNormR1csError> {
        let mut assignment = SignedUnitMaskWriter::new(self.structure.m);
        self.encode_into(arm, field_assignment, &mut assignment)?;
        Ok(assignment.finish())
    }

    fn encode_into(
        &self,
        arm: usize,
        field_assignment: &[F],
        assignment: &mut impl LowNormAssignmentWriter,
    ) -> Result<(), LowNormR1csError> {
        let slots = self
            .arm_slots
            .get(arm)
            .ok_or(LowNormR1csError::ArmIndexOutOfRange {
                arm,
                arms: self.arm_slots.len(),
            })?;
        if field_assignment.len() != slots.len() {
            return Err(LowNormR1csError::AssignmentLength {
                got: field_assignment.len(),
                expected: slots.len(),
            });
        }
        if field_assignment.first().copied() != Some(F::ONE) {
            return Err(LowNormR1csError::ConstantOne);
        }

        assignment.set(0, F::ONE)?;
        assignment.set(self.selector_cols[arm], F::ONE)?;
        for col in 1..self.public_field_count {
            self.write_field(arm, col, slots, field_assignment, assignment)?;
        }
        for col in self.public_field_count..slots.len() {
            self.write_field(arm, col, slots, field_assignment, assignment)?;
        }
        let mut derived_values = Vec::with_capacity(self.arm_derived_product_sums[arm].len());
        for derived in &self.arm_derived_product_sums[arm] {
            let mut value = derived.factors.iter().fold(F::ZERO, |sum, factor| {
                sum + factor.coefficient
                    * eval_source_lc(&factor.left, field_assignment)
                    * eval_source_lc(&factor.right, field_assignment)
            });
            if let Some(previous) = derived.previous {
                value += derived_values[previous];
            }
            write_encoded_value(assignment, Some(derived.slot), None, false, value, usize::MAX)?;
            derived_values.push(value);
        }
        Ok(())
    }

    fn write_field(
        &self,
        arm: usize,
        col: usize,
        slots: &[super::CompactSlot],
        field_assignment: &[F],
        assignment: &mut impl LowNormAssignmentWriter,
    ) -> Result<(), LowNormR1csError> {
        if slots[col].is_none() {
            return Ok(());
        }
        if let Some(source) = self.arm_equal_aliases[arm][col].get() {
            if field_assignment[col] != field_assignment[source] {
                return Err(LowNormR1csError::AliasedFieldMismatch {
                    field_col: col,
                    source_col: source,
                });
            }
            return Ok(());
        }
        write_encoded_value(
            assignment,
            slots[col].get(),
            self.arm_aliases[arm][col].get(),
            self.arm_centered_columns[arm][col],
            field_assignment[col],
            col,
        )
    }
}
