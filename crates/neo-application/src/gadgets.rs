//! Domain-neutral algebraic gadgets and their retained semantic descriptions.

use std::ops::Range;

use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::{ConstraintTag, TaggedR1csBuilder};

/// Columns participating in the relation `is_zero = (value == 0)`.
///
/// The constraints uniquely determine `is_zero`. When `value` is zero the
/// inverse column is unconstrained; [`Self::assign`] writes zero there as the
/// canonical assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroTest {
    pub value: usize,
    pub inverse: usize,
    pub is_zero: usize,
}

impl ZeroTest {
    /// Emit the zero-test rows and retain this descriptor in the relation's
    /// gadget-occurrence catalog.
    pub fn push_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let first_row = builder.next_row_index();
        let const_one = builder.const_one_column();
        builder.push_row(
            [(self.value, F::ONE)],
            [(self.inverse, F::ONE)],
            [(const_one, F::ONE), (self.is_zero, -F::ONE)],
        );
        builder.push_row([(self.value, F::ONE)], [(self.is_zero, F::ONE)], []);
        builder.record_gadget(GadgetDescriptor::ZeroTest(*self), first_row);
    }

    /// Derive the canonical auxiliary assignment from the current value
    /// column. Callers remain responsible for ordering dependent gadgets.
    pub fn assign(&self, assignment: &mut [F]) {
        let value = assignment[self.value];
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GadgetDescriptor {
    ZeroTest(ZeroTest),
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
