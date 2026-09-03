//! Domain-neutral algebraic gadgets and their retained semantic descriptions.

use std::ops::Range;

use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::{ConstraintTag, TaggedR1csBuilder};

fn evaluate_linear_expression(expression: &[(usize, F)], assignment: &[F]) -> F {
    expression
        .iter()
        .fold(F::ZERO, |value, &(column, coefficient)| {
            value + assignment[column] * coefficient
        })
}

pub(crate) fn push_pow7_expression<Owner: Clone>(
    builder: &mut TaggedR1csBuilder<'_, Owner>,
    expression: &[(usize, F)],
    powers: [usize; 4],
) {
    let [x2, x4, x6, x7] = powers;
    builder.push_row(expression.iter().copied(), expression.iter().copied(), [(x2, F::ONE)]);
    builder.push_row([(x2, F::ONE)], [(x2, F::ONE)], [(x4, F::ONE)]);
    builder.push_row([(x4, F::ONE)], [(x2, F::ONE)], [(x6, F::ONE)]);
    builder.push_row([(x6, F::ONE)], expression.iter().copied(), [(x7, F::ONE)]);
}

pub(crate) fn assign_pow7_expression(expression: &[(usize, F)], powers: [usize; 4], assignment: &mut [F]) {
    let x = evaluate_linear_expression(expression, assignment);
    let [x2, x4, x6, x7] = powers;
    assignment[x2] = x * x;
    assignment[x4] = assignment[x2] * assignment[x2];
    assignment[x6] = assignment[x2] * assignment[x4];
    assignment[x7] = assignment[x6] * x;
}

/// Constrains four auxiliary columns to `x²`, `x⁴`, `x⁶`, and `x⁷` for a
/// linear field expression `x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pow7<const N: usize = 1> {
    pub expression: [(usize, F); N],
    pub powers: [usize; 4],
}

impl<const N: usize> Pow7<N> {
    pub fn push_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let first_row = builder.next_row_index();
        push_pow7_expression(builder, &self.expression, self.powers);
        builder.record_gadget(
            GadgetDescriptor::Pow7 {
                expression: self.expression.to_vec(),
                powers: self.powers,
            },
            first_row,
        );
    }

    pub fn assign(&self, assignment: &mut [F]) {
        assign_pow7_expression(&self.expression, self.powers, assignment);
    }
}

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
        let value = evaluate_linear_expression(&self.expression, assignment);
        if value == F::ZERO {
            assignment[self.inverse] = F::ZERO;
            assignment[self.is_zero] = F::ONE;
        } else {
            assignment[self.inverse] = value.try_inverse().expect("nonzero field inverse");
            assignment[self.is_zero] = F::ZERO;
        }
    }
}

/// Conditionally selects `lhs` or `rhs` into `output` on active rows.
///
/// `condition` is a linear expression that must evaluate to zero or one:
/// one selects `lhs`, and zero selects `rhs`. If it is not Boolean, the rows
/// remain satisfiable but enforce affine interpolation in the field rather
/// than selection.
///
/// `activation` is the application gate. Zero leaves `output` unconstrained,
/// while any nonzero field value enforces the output relation. Callers must
/// prove that it has the intended gate semantics, normally by constraining it
/// to be Boolean. This gadget emits neither Boolean constraint.
///
/// Without `delta`, directly gating the select relation would multiply
/// `activation`, `condition`, and `lhs - rhs`, exceeding R1CS degree. The
/// global delta row computes that intermediate product so the output relation
/// can be gated in a second R1CS row.
///
/// The delta relation is global, including on inactive rows. This gives the
/// auxiliary column a canonical value while leaving `output` unconstrained
/// when `activation` is zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConditionalSelect<const N: usize = 1> {
    pub activation: usize,
    pub condition: [(usize, F); N],
    pub lhs: usize,
    pub rhs: usize,
    pub output: usize,
    pub delta: usize,
}

impl<const N: usize> ConditionalSelect<N> {
    /// Emit the select rows and retain this descriptor in the relation's
    /// gadget-occurrence catalog.
    pub fn push_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let first_row = builder.next_row_index();
        builder.push_row(
            self.condition,
            [(self.lhs, F::ONE), (self.rhs, -F::ONE)],
            [(self.delta, F::ONE)],
        );
        builder.push_row(
            [(self.activation, F::ONE)],
            [(self.output, F::ONE), (self.rhs, -F::ONE), (self.delta, -F::ONE)],
            [],
        );
        builder.record_gadget(
            GadgetDescriptor::ConditionalSelect {
                activation: self.activation,
                condition: self.condition.to_vec(),
                lhs: self.lhs,
                rhs: self.rhs,
                output: self.output,
                delta: self.delta,
            },
            first_row,
        );
    }

    /// Assign the gadget-owned delta without overwriting the semantic output.
    pub fn assign_delta(&self, assignment: &mut [F]) {
        let condition = evaluate_linear_expression(&self.condition, assignment);
        assignment[self.delta] = condition * (assignment[self.lhs] - assignment[self.rhs]);
    }
}

/// Machine-readable semantics retained for one shared gadget invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GadgetDescriptor {
    Pow7 {
        expression: Vec<(usize, F)>,
        powers: [usize; 4],
    },
    ZeroTest {
        expression: Vec<(usize, F)>,
        inverse: usize,
        is_zero: usize,
    },
    ConditionalSelect {
        activation: usize,
        condition: Vec<(usize, F)>,
        lhs: usize,
        rhs: usize,
        output: usize,
        delta: usize,
    },
    Poseidon2FullRound12 {
        choices: Vec<(usize, usize)>,
        state_before: [usize; 12],
        state_after: [usize; 12],
        powers: Vec<[usize; 4]>,
    },
    Poseidon2PartialPair12 {
        choices: Vec<(usize, usize)>,
        state_before: [usize; 12],
        state_after: [usize; 12],
        powers: [usize; 8],
    },
    Poseidon2Permutation12 {
        input: [usize; 12],
        output: [usize; 12],
        auxiliary_start: usize,
        auxiliary_len: usize,
    },
    EventCommitment {
        previous: [usize; 4],
        block: [usize; 8],
        output: [usize; 4],
        auxiliary_start: usize,
        auxiliary_len: usize,
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
