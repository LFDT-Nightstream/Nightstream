//! Raw-row evaluation helpers for named owner mutations.
//!
//! The full independent assignment evaluator lives in `canonical_assignment`.

use super::{
    add_mod, changed_word, event_row_count, mul_mod, word, Event, Invocation, RawCombination, RawCompactInvocation,
    RawCompactTemplate, RawPackage, RawTemplateCombination, RawTemplateRow, GOLDILOCKS_MODULUS,
};

#[derive(Clone, Copy)]
enum Side {
    A,
    B,
    C,
}

#[derive(Clone, Copy)]
pub(super) enum RawRowMutation {
    None,
    AssignmentColumn(usize),
    CConstant(u64),
}

struct RawAssignment<'a> {
    private_values: &'a [u64],
    public_values: &'a [u64],
    constant_column: usize,
    total_columns: usize,
    changed_column: Option<usize>,
}

impl RawAssignment<'_> {
    fn value(&self, column: usize) -> u64 {
        assert!(column < self.total_columns, "raw assignment column");
        let value = if column < self.constant_column {
            self.private_values[column]
        } else if column == self.constant_column {
            1
        } else {
            self.public_values[column - self.constant_column - 1]
        };
        if self.changed_column == Some(column) {
            assert_ne!(column, self.constant_column, "raw constant column mutation");
            changed_word(value)
        } else {
            value
        }
    }
}

fn canonical(value: u64) -> u64 {
    assert!(value < GOLDILOCKS_MODULUS, "canonical raw package word");
    value
}

fn sparse_value_with_constant_delta(
    combination: &RawCombination,
    constant_delta: u64,
    assignment: &RawAssignment<'_>,
) -> u64 {
    combination
        .1
        .iter()
        .fold(add_mod(canonical(combination.0), constant_delta), |sum, term| {
            add_mod(sum, mul_mod(canonical(term.1), assignment.value(word(term.0))))
        })
}

fn sparse_value(combination: &RawCombination, assignment: &RawAssignment<'_>) -> u64 {
    sparse_value_with_constant_delta(combination, 0, assignment)
}

fn hash_input_value(
    chain: &super::RawChain,
    ordinal: usize,
    lane: usize,
    rate: usize,
    local_count: usize,
    output_local_start: usize,
    assignment: &RawAssignment<'_>,
) -> u64 {
    let mut value = 0;
    if ordinal > 0 {
        let previous_output = word(chain.5) + (ordinal - 1) * local_count + output_local_start + lane;
        value = add_mod(value, assignment.value(previous_output));
    }
    if ordinal < word(chain.7) {
        let input_offset = ordinal * rate + lane;
        if lane < rate && input_offset < word(chain.4) {
            value = add_mod(value, assignment.value(word(chain.3) + input_offset));
        }
    } else if lane == 0 {
        value = add_mod(value, 1);
    }
    value
}

fn template_input_value(
    invocation: Invocation<'_>,
    lane: usize,
    rate: usize,
    local_count: usize,
    output_local_start: usize,
    assignment: &RawAssignment<'_>,
) -> u64 {
    match invocation {
        Invocation::Hash { chain, ordinal } => {
            hash_input_value(chain, ordinal, lane, rate, local_count, output_local_start, assignment)
        }
        Invocation::Explicit(invocation) => sparse_value(&invocation.3[lane], assignment),
    }
}

fn template_local_value(
    invocation: Invocation<'_>,
    lane: usize,
    local_count: usize,
    assignment: &RawAssignment<'_>,
) -> u64 {
    let witness_start = match invocation {
        Invocation::Hash { chain, ordinal } => word(chain.5) + ordinal * local_count,
        Invocation::Explicit(invocation) => word(invocation.2),
    };
    assignment.value(witness_start + lane)
}

fn template_value(
    combination: &RawTemplateCombination,
    constant_delta: u64,
    invocation: Invocation<'_>,
    rate: usize,
    local_count: usize,
    output_local_start: usize,
    assignment: &RawAssignment<'_>,
) -> u64 {
    combination
        .1
        .iter()
        .fold(add_mod(canonical(combination.0), constant_delta), |sum, term| {
            let lane = word(term.0 .1);
            let value = match term.0 .0 {
                0 => template_input_value(invocation, lane, rate, local_count, output_local_start, assignment),
                1 => template_local_value(invocation, lane, local_count, assignment),
                _ => panic!("raw template column tag"),
            };
            add_mod(sum, mul_mod(canonical(term.1), value))
        })
}

fn compact_input_column(invocation: &RawCompactInvocation, input: usize) -> usize {
    invocation
        .4
        .iter()
        .find_map(|range| {
            let start = word(range.0);
            let count = word(range.1);
            (start <= input && input < start + count).then(|| word(range.2) + (input - start) * word(range.3))
        })
        .expect("raw compact input coverage")
}

fn compact_value(
    combination: &RawTemplateCombination,
    constant_delta: u64,
    invocation: &RawCompactInvocation,
    assignment: &RawAssignment<'_>,
) -> u64 {
    combination
        .1
        .iter()
        .fold(add_mod(canonical(combination.0), constant_delta), |sum, term| {
            let index = word(term.0 .1);
            let column = match term.0 .0 {
                0 => compact_input_column(invocation, index),
                1 => word(invocation.3) + index,
                _ => panic!("raw compact column tag"),
            };
            add_mod(sum, mul_mod(canonical(term.1), assignment.value(column)))
        })
}

fn template_side(row: &RawTemplateRow, side: Side) -> &RawTemplateCombination {
    match side {
        Side::A => &row.1,
        Side::B => &row.2,
        Side::C => &row.3,
    }
}

fn event_value(
    event: Event<'_>,
    template_ordinal: usize,
    template: &super::RawTemplate,
    side: Side,
    rate: usize,
    local_count: usize,
    output_local_start: usize,
    assignment: &RawAssignment<'_>,
    constant_delta: u64,
) -> u64 {
    match event {
        Event::Permutation { invocation, .. } => {
            let row = &template.3[template_ordinal];
            assert_eq!(word(row.0), template_ordinal, "raw template row order");
            template_value(
                template_side(row, side),
                constant_delta,
                invocation,
                rate,
                local_count,
                output_local_start,
                assignment,
            )
        }
        Event::Compact { invocation, template } => {
            let row: &RawCompactTemplate = template;
            let row = &row.4[template_ordinal];
            let combination = match side {
                Side::A => &row.1,
                Side::B => &row.2,
                Side::C => &row.3,
            };
            compact_value(combination, constant_delta, invocation, assignment)
        }
        Event::Witness(instruction) => match side {
            Side::A => sparse_value_with_constant_delta(&instruction.2, constant_delta, assignment),
            Side::B => sparse_value_with_constant_delta(&instruction.3, constant_delta, assignment),
            Side::C => add_mod(constant_delta, assignment.value(word(instruction.1))),
        },
        Event::Assertion(row) => {
            let combination = match side {
                Side::A => &row.1,
                Side::B => &row.2,
                Side::C => &row.3,
            };
            sparse_value_with_constant_delta(combination, constant_delta, assignment)
        }
    }
}

pub(super) fn event_row_values(
    raw: &RawPackage,
    event: Event<'_>,
    ordinal: usize,
    private_values: &[u64],
    public_values: &[u64],
    mutation: RawRowMutation,
) -> [u64; 3] {
    assert!(ordinal < event_row_count(event, raw), "direct raw event row");
    let rate = word(raw.2 .1);
    let local_count = word(raw.5 .1);
    let output_local_start = word(raw.5 .2);
    let assignment = RawAssignment {
        private_values,
        public_values,
        constant_column: word(raw.3 .2),
        total_columns: word(raw.3 .4),
        changed_column: match mutation {
            RawRowMutation::AssignmentColumn(column) => Some(column),
            RawRowMutation::None | RawRowMutation::CConstant(_) => None,
        },
    };
    assert!(
        private_values.len() <= assignment.constant_column,
        "raw private assignment exceeds the sealed private-column bound"
    );
    assert_eq!(public_values.len(), word(raw.3 .3));
    assert_eq!(
        assignment.total_columns,
        assignment.constant_column + 1 + public_values.len()
    );
    [Side::A, Side::B, Side::C].map(|side| {
        let constant_delta = match (side, mutation) {
            (Side::C, RawRowMutation::CConstant(delta)) => delta,
            _ => 0,
        };
        event_value(
            event,
            ordinal,
            &raw.5,
            side,
            rate,
            local_count,
            output_local_start,
            &assignment,
            constant_delta,
        )
    })
}
