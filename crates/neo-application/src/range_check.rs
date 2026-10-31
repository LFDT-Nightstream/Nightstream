//! Explicit bit-decomposition layout, constraints, and witness assignment for
//! declared application column widths.

use std::ops::Range;

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{ColumnFamilySpec, ColumnRegistry, ColumnRegistryError, ColumnWidth, ConstraintTag, TaggedR1csBuilder};

/// Metadata for the Boolean column family appended by a range-check layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeCheckBitFamily {
    pub region: &'static str,
    pub name: &'static str,
    pub role: &'static str,
}

/// Number of explicit decomposition bits required by one declared column.
pub const fn decomposition_bit_count(width: ColumnWidth) -> usize {
    match width {
        ColumnWidth::Boolean | ColumnWidth::Field => 0,
        ColumnWidth::Byte => 8,
        ColumnWidth::U32 => 32,
        ColumnWidth::Bits(bits) => {
            assert!(bits > 0 && bits <= 63, "custom bit width must be in 1..=63");
            bits as usize
        }
    }
}

/// Validated base columns together with their appended decomposition-bit
/// layout. Generated bits are not treated as new base declarations: their
/// Booleanity is enforced by the decompositions that allocated them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeCheckLayout {
    columns: ColumnRegistry,
    base_column_count: usize,
    bit_columns_by_base_column: Vec<Option<Range<usize>>>,
}

impl RangeCheckLayout {
    pub fn new(
        base_families: impl IntoIterator<Item = ColumnFamilySpec>,
        bit_family: RangeCheckBitFamily,
    ) -> Result<Self, RangeCheckLayoutError> {
        let base_columns = ColumnRegistry::new(base_families)?;
        let base_column_count = base_columns.column_count();
        let mut next_bit_column = base_column_count;
        let mut bit_columns_by_base_column = Vec::with_capacity(base_column_count);

        for family in base_columns.families() {
            let bit_count = decomposition_bit_count(family.width);
            for column in family.start..family.end() {
                if bit_count == 0 {
                    bit_columns_by_base_column.push(None);
                    continue;
                }
                let end = next_bit_column
                    .checked_add(bit_count)
                    .ok_or(RangeCheckLayoutError::IndexOverflow { column })?;
                bit_columns_by_base_column.push(Some(next_bit_column..end));
                next_bit_column = end;
            }
        }

        let bit_column_count = next_bit_column - base_column_count;
        let columns = if bit_column_count == 0 {
            base_columns
        } else {
            ColumnRegistry::new(
                base_columns
                    .families()
                    .iter()
                    .copied()
                    .chain([ColumnFamilySpec {
                        region: bit_family.region,
                        start: base_column_count,
                        len: bit_column_count,
                        name: bit_family.name,
                        role: bit_family.role,
                        width: ColumnWidth::Boolean,
                    }]),
            )?
        };

        Ok(Self {
            columns,
            base_column_count,
            bit_columns_by_base_column,
        })
    }

    pub const fn base_column_count(&self) -> usize {
        self.base_column_count
    }

    pub fn column_count(&self) -> usize {
        self.columns.column_count()
    }

    pub fn columns(&self) -> &ColumnRegistry {
        &self.columns
    }

    pub fn bit_columns(&self) -> Range<usize> {
        self.base_column_count..self.column_count()
    }

    /// Return this base column's decomposition columns least-significant bit
    /// first: `range.start + i` represents the bit with weight `2^i`.
    pub fn bit_columns_for(&self, column: usize) -> Option<Range<usize>> {
        self.bit_columns_by_base_column
            .get(column)
            .cloned()
            .flatten()
    }

    /// Emit the standard Booleanity and recomposition rows. Each base family
    /// supplies the row label, while `owner` supplies the application domain.
    pub fn push_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>, owner: Owner) {
        for family in self.columns.families() {
            if family.start >= self.base_column_count {
                break;
            }
            match family.width {
                ColumnWidth::Field => {}
                ColumnWidth::Boolean => {
                    builder.with_tag(ConstraintTag::new(family.name, owner.clone()), |builder| {
                        for column in family.start..family.end() {
                            builder.push_boolean(column);
                        }
                    });
                }
                ColumnWidth::Byte | ColumnWidth::U32 | ColumnWidth::Bits(_) => {
                    builder.with_tag(ConstraintTag::new(family.name, owner.clone()), |builder| {
                        for column in family.start..family.end() {
                            let bits = self
                                .bit_columns_for(column)
                                .expect("decomposed base column has an allocated bit range");
                            for bit in bits.clone() {
                                builder.push_boolean(bit);
                            }
                            builder.push_row(
                                bits.enumerate()
                                    .map(|(index, bit)| (bit, F::from_u64(1u64 << index))),
                                [(builder.const_one_column(), F::ONE)],
                                [(column, F::ONE)],
                            );
                        }
                    });
                }
            }
        }
    }

    /// Compute or refresh every decomposition bit from its base column.
    ///
    /// Out-of-range values have their low bits assigned normally, leaving the
    /// recomposition row unsatisfied.
    pub fn assign_bits(&self, witness: &mut Vec<F>) -> Result<(), RangeCheckAssignmentError> {
        if witness.len() != self.base_column_count && witness.len() != self.column_count() {
            return Err(RangeCheckAssignmentError::WitnessWidth {
                base: self.base_column_count,
                range_checked: self.column_count(),
                actual: witness.len(),
            });
        }
        witness.resize(self.column_count(), F::ZERO);
        for column in 0..self.base_column_count {
            let Some(bits) = self.bit_columns_for(column) else {
                continue;
            };
            let value = witness[column].as_canonical_u64();
            for (index, bit) in bits.enumerate() {
                witness[bit] = F::from_u64((value >> index) & 1);
            }
        }
        Ok(())
    }
}

/// F′ representation width for every column in a completed range-check
/// registry, including one bit per generated Boolean column.
pub fn range_checked_variable_widths(columns: &ColumnRegistry) -> Vec<usize> {
    columns
        .families()
        .iter()
        .flat_map(|family| core::iter::repeat_n(variable_width(family.width), family.len))
        .collect()
}

const fn variable_width(width: ColumnWidth) -> usize {
    match width {
        ColumnWidth::Boolean => 1,
        ColumnWidth::Byte => 8,
        ColumnWidth::U32 => 32,
        ColumnWidth::Bits(bits) => bits as usize,
        // Goldilocks is not exactly 64 bits wide, but its canonical field
        // elements require a 64-bit representation.
        ColumnWidth::Field => 64,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RangeCheckLayoutError {
    #[error(transparent)]
    Columns(#[from] ColumnRegistryError),
    #[error("range-check bit allocation overflowed at base column {column}")]
    IndexOverflow { column: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RangeCheckAssignmentError {
    #[error(
        "range-check assignment has width {actual}; expected either base width {base} or range-checked width {range_checked}"
    )]
    WitnessWidth {
        base: usize,
        range_checked: usize,
        actual: usize,
    },
}
