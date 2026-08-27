//! Domain-neutral algebraic gadgets and their retained semantic descriptions.

use std::ops::Range;

use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::{ConstraintTag, TaggedR1csBuilder};

/// Columns participating in the relation `is_zero = (expression == 0)`.
///
/// The constraints uniquely determine `is_zero` as either zero or one; no
/// separate boolean constraint is needed for soundness. When the expression is
/// zero the inverse column is unconstrained; [`Self::assign`] writes zero there
/// as the canonical assignment.
///
/// The expression is evaluated in the field. When it represents an integer
/// predicate, the caller must establish that no nonzero integer multiple of
/// the field modulus lies in the expression's possible range; otherwise
/// modular aliasing can make a nonzero intended value test as zero.
///
/// The fixed-size expression keeps witness assignment allocation-free. Its
/// retained [`GadgetDescriptor`] erases `N` so one catalog can hold expressions
/// of different sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroTest<const N: usize = 1> {
    pub expression: [(usize, F); N],
    pub inverse: usize,
    pub is_zero: usize,
}

impl ZeroTest<1> {
    pub const fn column(value: usize, inverse: usize, is_zero: usize) -> Self {
        Self {
            expression: [(value, F::ONE)],
            inverse,
            is_zero,
        }
    }
}

impl<const N: usize> ZeroTest<N> {
    /// Emit the zero-test rows and retain this descriptor in the relation's
    /// gadget-occurrence catalog.
    pub fn push_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let first_row = builder.next_row_index();
        let const_one = builder.const_one_column();
        builder.push_row(
            self.expression,
            [(self.inverse, F::ONE)],
            [(const_one, F::ONE), (self.is_zero, -F::ONE)],
        );
        builder.push_row(self.expression, [(self.is_zero, F::ONE)], []);
        builder.record_gadget(
            GadgetDescriptor::ZeroTest {
                expression: self.expression.to_vec(),
                inverse: self.inverse,
                is_zero: self.is_zero,
            },
            first_row,
        );
    }

    /// Derive the canonical auxiliary assignment from the current expression.
    /// Callers remain responsible for ordering dependent gadgets.
    pub fn assign(&self, assignment: &mut [F]) {
        let value = self
            .expression
            .iter()
            .fold(F::ZERO, |value, &(column, coefficient)| {
                value + assignment[column] * coefficient
            });
        if value == F::ZERO {
            assignment[self.inverse] = F::ZERO;
            assignment[self.is_zero] = F::ONE;
        } else {
            assignment[self.inverse] = value.try_inverse().expect("nonzero field inverse");
            assignment[self.is_zero] = F::ZERO;
        }
    }
}

/// Machine-readable semantics retained for one shared gadget invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GadgetDescriptor {
    ZeroTest {
        expression: Vec<(usize, F)>,
        inverse: usize,
        is_zero: usize,
    },
}

/// One tagged gadget invocation and the exact relation rows it emitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GadgetOccurrence<Owner> {
    tag: ConstraintTag<Owner>,
    descriptor: GadgetDescriptor,
    row_range: Range<usize>,
}

impl<Owner> GadgetOccurrence<Owner> {
    pub(crate) const fn new(tag: ConstraintTag<Owner>, descriptor: GadgetDescriptor, row_range: Range<usize>) -> Self {
        Self {
            tag,
            descriptor,
            row_range,
        }
    }

    pub const fn tag(&self) -> &ConstraintTag<Owner> {
        &self.tag
    }

    pub const fn descriptor(&self) -> &GadgetDescriptor {
        &self.descriptor
    }

    pub const fn row_range(&self) -> &Range<usize> {
        &self.row_range
    }
}
