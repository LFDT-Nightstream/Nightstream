//! Column-oriented views over compiled relation metadata.
//!
//! This module reports where columns occur. It does not infer whether those
//! occurrences are sufficient to enforce an application's intended semantics.
//! R1CS occurrences retain the builder's sparse terms before matrix
//! coalescing, so repeated same-side terms may combine or cancel in the
//! compiled matrix.

use neo_math::F;

use crate::{
    ApplicationRelation, ContinuityCatalog, ContinuityGroup, ContinuityLink, GadgetDescriptor, GadgetOccurrence,
    MemoryCatalog, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec, R1csSide, TaggedR1csRow,
};

/// One occurrence of a column in a sparse R1CS term list.
#[derive(Clone, Copy, Debug)]
pub struct R1csColumnOccurrence<'a, Owner> {
    row_index: usize,
    side: R1csSide,
    coefficient: F,
    tagged_row: &'a TaggedR1csRow<Owner>,
}

impl<'a, Owner> R1csColumnOccurrence<'a, Owner> {
    pub const fn row_index(&self) -> usize {
        self.row_index
    }

    pub const fn side(&self) -> R1csSide {
        self.side
    }

    pub const fn coefficient(&self) -> F {
        self.coefficient
    }

    /// Complete tagged row containing this term.
    pub const fn tagged_row(&self) -> &'a TaggedR1csRow<Owner> {
        self.tagged_row
    }
}

/// Semantic role played by a column in one retained shared gadget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GadgetColumnRole {
    ZeroTestExpression { term_index: usize, coefficient: F },
    ZeroTestInverse,
    ZeroTestIsZero,
    ConditionalSelectActivation,
    ConditionalSelectCondition { term_index: usize, coefficient: F },
    ConditionalSelectLhs,
    ConditionalSelectRhs,
    ConditionalSelectOutput,
    ConditionalSelectDelta,
}

/// One semantic gadget role involving the queried column.
#[derive(Clone, Copy, Debug)]
pub struct GadgetColumnOccurrence<'a, Owner> {
    occurrence: &'a GadgetOccurrence<Owner>,
    role: GadgetColumnRole,
}

impl<'a, Owner> GadgetColumnOccurrence<'a, Owner> {
    pub const fn occurrence(&self) -> &'a GadgetOccurrence<Owner> {
        self.occurrence
    }

    pub const fn role(&self) -> GadgetColumnRole {
        self.role
    }
}

/// Reverse index over the exact R1CS rows and retained gadgets in a relation.
pub struct ColumnConstraintIndex<'a, Owner> {
    r1cs_by_column: Vec<Vec<R1csColumnOccurrence<'a, Owner>>>,
    gadgets_by_column: Vec<Vec<GadgetColumnOccurrence<'a, Owner>>>,
}

impl<'a, Owner> ColumnConstraintIndex<'a, Owner> {
    pub fn new(relation: &'a ApplicationRelation<Owner>) -> Self {
        let column_count = relation.columns().column_count();
        let catalog = relation.r1cs().catalog();
        let mut r1cs_by_column: Vec<Vec<_>> = (0..column_count).map(|_| Vec::new()).collect();

        for (row_index, tagged_row) in catalog.rows().iter().enumerate() {
            for (side, terms) in [
                (R1csSide::A, tagged_row.row().a_terms()),
                (R1csSide::B, tagged_row.row().b_terms()),
                (R1csSide::C, tagged_row.row().c_terms()),
            ] {
                for &(column, coefficient) in terms {
                    // The relation builder rejects out-of-range terms before
                    // an ApplicationRelation can be constructed.
                    r1cs_by_column[column].push(R1csColumnOccurrence {
                        row_index,
                        side,
                        coefficient,
                        tagged_row,
                    });
                }
            }
        }

        let mut gadgets_by_column: Vec<Vec<_>> = (0..column_count).map(|_| Vec::new()).collect();
        for occurrence in catalog.gadget_occurrences() {
            index_gadget_occurrence(&mut gadgets_by_column, occurrence);
        }

        Self {
            r1cs_by_column,
            gadgets_by_column,
        }
    }

    /// Occurrences in ascending row order, then A, B, C order within each row,
    /// preserving the builder's term order within each side.
    pub fn r1cs_occurrences(&self, column: usize) -> Option<&[R1csColumnOccurrence<'a, Owner>]> {
        self.r1cs_by_column.get(column).map(Vec::as_slice)
    }

    pub fn gadget_occurrences(&self, column: usize) -> Option<&[GadgetColumnOccurrence<'a, Owner>]> {
        self.gadgets_by_column.get(column).map(Vec::as_slice)
    }
}

fn index_gadget_occurrence<'a, Owner>(
    by_column: &mut [Vec<GadgetColumnOccurrence<'a, Owner>>],
    occurrence: &'a GadgetOccurrence<Owner>,
) {
    let mut push = |column: usize, role: GadgetColumnRole| {
        by_column[column].push(GadgetColumnOccurrence { occurrence, role });
    };

    match occurrence.descriptor() {
        GadgetDescriptor::ZeroTest {
            expression,
            inverse,
            is_zero,
        } => {
            for (term_index, &(column, coefficient)) in expression.iter().enumerate() {
                push(
                    column,
                    GadgetColumnRole::ZeroTestExpression {
                        term_index,
                        coefficient,
                    },
                );
            }
            push(*inverse, GadgetColumnRole::ZeroTestInverse);
            push(*is_zero, GadgetColumnRole::ZeroTestIsZero);
        }
        GadgetDescriptor::ConditionalSelect {
            activation,
            condition,
            lhs,
            rhs,
            output,
            delta,
        } => {
            push(*activation, GadgetColumnRole::ConditionalSelectActivation);
            for (term_index, &(column, coefficient)) in condition.iter().enumerate() {
                push(
                    column,
                    GadgetColumnRole::ConditionalSelectCondition {
                        term_index,
                        coefficient,
                    },
                );
            }
            push(*lhs, GadgetColumnRole::ConditionalSelectLhs);
            push(*rhs, GadgetColumnRole::ConditionalSelectRhs);
            push(*output, GadgetColumnRole::ConditionalSelectOutput);
            push(*delta, GadgetColumnRole::ConditionalSelectDelta);
        }
    }
}

/// Role played by a column in one logical memory port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryColumnRole {
    Address { position: usize },
    Value,
    ValueBefore,
    Activation,
}

/// One logical-memory declaration involving the queried column.
#[derive(Clone, Copy, Debug)]
pub struct MemoryColumnOccurrence<'a, Id> {
    memory_index: usize,
    port_index: usize,
    memory: &'a MemorySpec<Id>,
    port: &'a MemoryPortSpec,
    role: MemoryColumnRole,
}

impl<'a, Id> MemoryColumnOccurrence<'a, Id> {
    pub const fn memory_index(&self) -> usize {
        self.memory_index
    }

    pub const fn port_index(&self) -> usize {
        self.port_index
    }

    pub const fn memory(&self) -> &'a MemorySpec<Id> {
        self.memory
    }

    pub const fn port(&self) -> &'a MemoryPortSpec {
        self.port
    }

    pub const fn role(&self) -> MemoryColumnRole {
        self.role
    }
}

/// Find every declared logical-memory role involving `column`.
pub fn memory_column_occurrences<Id>(
    catalog: &MemoryCatalog<Id>,
    column: usize,
) -> Vec<MemoryColumnOccurrence<'_, Id>> {
    let mut occurrences = Vec::new();
    for (memory_index, memory) in catalog.entries().iter().enumerate() {
        for (port_index, port) in memory.ports.iter().enumerate() {
            for (position, &address_column) in port.address_columns.iter().enumerate() {
                if address_column == column {
                    occurrences.push(MemoryColumnOccurrence {
                        memory_index,
                        port_index,
                        memory,
                        port,
                        role: MemoryColumnRole::Address { position },
                    });
                }
            }
            if port.value_column == column {
                occurrences.push(MemoryColumnOccurrence {
                    memory_index,
                    port_index,
                    memory,
                    port,
                    role: MemoryColumnRole::Value,
                });
            }
            if matches!(
                port.kind,
                MemoryPortKind::Write {
                    value_before_column: Some(value_before)
                } if value_before == column
            ) {
                occurrences.push(MemoryColumnOccurrence {
                    memory_index,
                    port_index,
                    memory,
                    port,
                    role: MemoryColumnRole::ValueBefore,
                });
            }
            if matches!(
                port.activation,
                MemoryPortActivation::When(activation) | MemoryPortActivation::Unless(activation)
                    if activation == column
            ) {
                occurrences.push(MemoryColumnOccurrence {
                    memory_index,
                    port_index,
                    memory,
                    port,
                    role: MemoryColumnRole::Activation,
                });
            }
        }
    }
    occurrences
}

/// Endpoint role played by a column in one cross-step equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinuityColumnRole {
    PreviousStep,
    NextStep,
}

/// One cross-step continuity declaration involving the queried column.
#[derive(Clone, Copy, Debug)]
pub struct ContinuityColumnOccurrence<'a> {
    group_index: usize,
    link_index: usize,
    group: &'a ContinuityGroup,
    link: &'a ContinuityLink,
    role: ContinuityColumnRole,
}

impl<'a> ContinuityColumnOccurrence<'a> {
    pub const fn group_index(&self) -> usize {
        self.group_index
    }

    pub const fn link_index(&self) -> usize {
        self.link_index
    }

    pub const fn group(&self) -> &'a ContinuityGroup {
        self.group
    }

    pub const fn link(&self) -> &'a ContinuityLink {
        self.link
    }

    pub const fn role(&self) -> ContinuityColumnRole {
        self.role
    }
}

/// Find every declared cross-step endpoint involving `column`.
pub fn continuity_column_occurrences(
    catalog: &ContinuityCatalog,
    column: usize,
) -> Vec<ContinuityColumnOccurrence<'_>> {
    let mut occurrences = Vec::new();
    for (group_index, group) in catalog.groups().iter().enumerate() {
        for (link_index, link) in group.links.iter().enumerate() {
            if link.previous_step_column == column {
                occurrences.push(ContinuityColumnOccurrence {
                    group_index,
                    link_index,
                    group,
                    link,
                    role: ContinuityColumnRole::PreviousStep,
                });
            }
            if link.next_step_column == column {
                occurrences.push(ContinuityColumnOccurrence {
                    group_index,
                    link_index,
                    group,
                    link,
                    role: ContinuityColumnRole::NextStep,
                });
            }
        }
    }
    occurrences
}
