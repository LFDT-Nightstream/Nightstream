//! Independent evaluation of a canonical schema-8 Lean package assignment.
//!
//! This module owns its raw decoder, row schedule, column mapping, and field
//! arithmetic. It does not use package witness, row, matrix, or constraint
//! evaluation code.

use rayon::prelude::*;
use serde::{de::IgnoredAny, Deserialize};
use std::ops::Range;

const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;

#[derive(Deserialize)]
struct RawPackage(
    u64,
    IgnoredAny,
    RawPoseidonSchedule,
    RawLayout,
    IgnoredAny,
    RawTemplate,
    Vec<RawChain>,
    Vec<RawInvocation>,
    Vec<RawCompactTemplate>,
    Vec<RawCompactInvocation>,
    IgnoredAny,
    Vec<RawInstruction>,
    Vec<RawRow>,
    IgnoredAny,
);

#[derive(Deserialize)]
struct RawPoseidonSchedule(u64, u64, IgnoredAny, IgnoredAny, IgnoredAny, IgnoredAny, u64, u64);

#[derive(Deserialize)]
struct RawLayout(u64, u64, u64, u64, u64, IgnoredAny, IgnoredAny);

#[derive(Clone, Deserialize)]
struct RawTemplate(u64, u64, u64, Vec<RawTemplateRow>);

#[derive(Clone, Deserialize)]
struct RawTemplateRow(
    u64,
    RawTemplateCombination,
    RawTemplateCombination,
    RawTemplateCombination,
);

#[derive(Clone, Deserialize)]
struct RawTemplateCombination(u64, Vec<RawTemplateTerm>);

#[derive(Clone, Deserialize)]
struct RawTemplateTerm(RawColumnRef, u64);

#[derive(Clone, Deserialize)]
struct RawColumnRef(u64, u64);

#[derive(Deserialize)]
struct RawChain(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

#[derive(Deserialize)]
struct RawInvocation(u64, u64, u64, Vec<RawCombination>);

#[derive(Deserialize)]
struct RawCompactTemplate(u64, u64, u64, IgnoredAny, Vec<RawCompactRow>);

#[derive(Deserialize)]
struct RawCompactRow(
    IgnoredAny,
    RawTemplateCombination,
    RawTemplateCombination,
    RawTemplateCombination,
);

#[derive(Deserialize)]
struct RawCompactRange(u64, u64, u64, u64);

#[derive(Deserialize)]
struct RawCompactInvocation(u64, u64, u64, u64, Vec<RawCompactRange>);

#[derive(Deserialize)]
struct RawInstruction(u64, u64, RawCombination, RawCombination);

#[derive(Deserialize)]
struct RawRow(u64, RawCombination, RawCombination, RawCombination);

#[derive(Deserialize)]
struct RawCombination(u64, Vec<RawTerm>);

#[derive(Deserialize)]
struct RawTerm(u64, u64);

#[derive(Clone, Copy)]
enum Side {
    A,
    B,
    C,
}

#[derive(Clone, Copy)]
enum Invocation<'a> {
    Hash { chain: &'a RawChain, ordinal: usize },
    Explicit(&'a RawInvocation),
}

#[derive(Clone, Copy)]
enum Event<'a> {
    Permutation {
        row_start: usize,
        invocation: Invocation<'a>,
    },
    Compact {
        invocation: &'a RawCompactInvocation,
        template: &'a RawCompactTemplate,
    },
    Witness(&'a RawInstruction),
    Assertion(&'a RawRow),
}

impl Event<'_> {
    fn row_start(self) -> usize {
        match self {
            Self::Permutation { row_start, .. } => row_start,
            Self::Compact { invocation, .. } => word(invocation.2),
            Self::Witness(instruction) => word(instruction.0),
            Self::Assertion(row) => word(row.0),
        }
    }
}

struct Assignment<'a> {
    private_values: &'a [u64],
    public_values: &'a [u64],
    constant_column: usize,
    total_columns: usize,
    unavailable_private: Option<Range<usize>>,
    changed_column: Option<usize>,
}

impl Assignment<'_> {
    fn value(&self, column: usize) -> u64 {
        assert!(column < self.total_columns, "canonical assignment column");
        assert!(
            !self
                .unavailable_private
                .as_ref()
                .is_some_and(|range| range.contains(&column)),
            "canonical assignment reads an unavailable private column",
        );
        let value = if column < self.constant_column {
            self.private_values[column]
        } else if column == self.constant_column {
            1
        } else {
            self.public_values[column - self.constant_column - 1]
        };
        if self.changed_column == Some(column) {
            (value + 1) % GOLDILOCKS_MODULUS
        } else {
            value
        }
    }
}

fn word(value: u64) -> usize {
    usize::try_from(value).expect("canonical package word fits usize")
}

fn canonical(value: u64) -> u64 {
    assert!(value < GOLDILOCKS_MODULUS, "canonical package field word");
    value
}

fn add_mod(left: u64, right: u64) -> u64 {
    ((u128::from(left) + u128::from(right)) % u128::from(GOLDILOCKS_MODULUS)) as u64
}

fn mul_mod(left: u64, right: u64) -> u64 {
    ((u128::from(left) * u128::from(right)) % u128::from(GOLDILOCKS_MODULUS)) as u64
}

fn sparse_value(combination: &RawCombination, assignment: &Assignment<'_>) -> u64 {
    combination
        .1
        .iter()
        .fold(canonical(combination.0), |sum, term| {
            add_mod(sum, mul_mod(canonical(term.1), assignment.value(word(term.0))))
        })
}

fn hash_input_value(
    chain: &RawChain,
    ordinal: usize,
    lane: usize,
    rate: usize,
    local_count: usize,
    output_local_start: usize,
    assignment: &Assignment<'_>,
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
    assignment: &Assignment<'_>,
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
    assignment: &Assignment<'_>,
) -> u64 {
    let witness_start = match invocation {
        Invocation::Hash { chain, ordinal } => word(chain.5) + ordinal * local_count,
        Invocation::Explicit(invocation) => word(invocation.2),
    };
    assignment.value(witness_start + lane)
}

fn template_value(
    combination: &RawTemplateCombination,
    invocation: Invocation<'_>,
    rate: usize,
    local_count: usize,
    output_local_start: usize,
    assignment: &Assignment<'_>,
) -> u64 {
    combination
        .1
        .iter()
        .fold(canonical(combination.0), |sum, term| {
            let lane = word(term.0 .1);
            let value = match term.0 .0 {
                0 => template_input_value(invocation, lane, rate, local_count, output_local_start, assignment),
                1 => template_local_value(invocation, lane, local_count, assignment),
                _ => panic!("canonical template column tag"),
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
        .expect("canonical compact input coverage")
}

fn compact_value(
    combination: &RawTemplateCombination,
    invocation: &RawCompactInvocation,
    assignment: &Assignment<'_>,
) -> u64 {
    combination
        .1
        .iter()
        .fold(canonical(combination.0), |sum, term| {
            let index = word(term.0 .1);
            let column = match term.0 .0 {
                0 => compact_input_column(invocation, index),
                1 => word(invocation.3) + index,
                _ => panic!("canonical compact column tag"),
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

fn template_side_mut(row: &mut RawTemplateRow, side: Side) -> &mut RawTemplateCombination {
    match side {
        Side::A => &mut row.1,
        Side::B => &mut row.2,
        Side::C => &mut row.3,
    }
}

fn event_value(
    event: Event<'_>,
    template_ordinal: usize,
    template: &RawTemplate,
    side: Side,
    rate: usize,
    local_count: usize,
    output_local_start: usize,
    assignment: &Assignment<'_>,
) -> u64 {
    match event {
        Event::Permutation { invocation, .. } => {
            let row = &template.3[template_ordinal];
            assert_eq!(word(row.0), template_ordinal, "canonical template row order");
            template_value(
                template_side(row, side),
                invocation,
                rate,
                local_count,
                output_local_start,
                assignment,
            )
        }
        Event::Compact { invocation, template } => {
            let row = &template.4[template_ordinal];
            let combination = match side {
                Side::A => &row.1,
                Side::B => &row.2,
                Side::C => &row.3,
            };
            compact_value(combination, invocation, assignment)
        }
        Event::Witness(instruction) => match side {
            Side::A => sparse_value(&instruction.2, assignment),
            Side::B => sparse_value(&instruction.3, assignment),
            Side::C => assignment.value(word(instruction.1)),
        },
        Event::Assertion(row) => {
            let combination = match side {
                Side::A => &row.1,
                Side::B => &row.2,
                Side::C => &row.3,
            };
            sparse_value(combination, assignment)
        }
    }
}

fn events(raw: &RawPackage) -> Vec<Event<'_>> {
    let template_rows = raw.5 .3.len();
    let mut events = Vec::new();
    for chain in &raw.6 {
        assert_ne!(chain.0, 0, "canonical hash-chain phase");
        assert_eq!(
            word(chain.2),
            word(chain.6) + word(chain.8),
            "canonical hash-chain rows"
        );
        assert_eq!(
            word(chain.6),
            (word(chain.7) + 1) * template_rows,
            "canonical hash-chain witness rows",
        );
        assert!(
            word(chain.8) == 0
                || (word(chain.9) >= word(raw.3 .2) + 1 && word(chain.9) + word(chain.8) <= word(raw.3 .4)),
            "canonical hash-chain digest range",
        );
        for ordinal in 0..=word(chain.7) {
            events.push(Event::Permutation {
                row_start: word(chain.1) + ordinal * template_rows,
                invocation: Invocation::Hash { chain, ordinal },
            });
        }
    }
    events.extend(raw.7.iter().map(|invocation| {
        assert_ne!(invocation.0, 0, "canonical invocation phase");
        Event::Permutation {
            row_start: word(invocation.1),
            invocation: Invocation::Explicit(invocation),
        }
    }));
    events.extend(raw.9.iter().map(|invocation| {
        assert_ne!(invocation.0, 0, "canonical compact phase");
        let template = &raw.8[word(invocation.1)];
        assert!(word(template.2) < word(template.0), "canonical compact output input");
        assert_eq!(template.4.len(), word(template.1) + 1, "canonical compact rows");
        let mut input_cursor = 0;
        for range in &invocation.4 {
            assert_eq!(word(range.0), input_cursor, "canonical compact input order");
            assert_ne!(word(range.1), 0, "canonical compact input count");
            assert_ne!(word(range.3), 0, "canonical compact column stride");
            input_cursor += word(range.1);
        }
        assert_eq!(input_cursor, word(template.0), "canonical compact input coverage");
        Event::Compact { invocation, template }
    }));
    events.extend(raw.11.iter().map(Event::Witness));
    events.extend(raw.12.iter().map(Event::Assertion));
    events.sort_unstable_by_key(|event| event.row_start());
    events
}

fn event_row_count(event: Event<'_>, raw: &RawPackage) -> usize {
    match event {
        Event::Permutation { .. } => raw.5 .3.len(),
        Event::Compact { template, .. } => template.4.len(),
        Event::Witness(_) | Event::Assertion(_) => 1,
    }
}

/// Check every physical row in one canonical schema-8 Lean raw package.
/// `Err` contains one unsatisfied physical row.
pub fn evaluate_canonical_assignment(
    bytes: &[u8],
    private_values: &[u64],
    public_values: &[u64],
) -> Result<usize, usize> {
    let raw: RawPackage = serde_json::from_slice(bytes).expect("canonical raw-package decode");
    assert_eq!(raw.0, 8, "canonical raw-package schema");
    assert_eq!(raw.3 .1, raw.3 .2, "canonical private/constant boundary");
    assert_eq!(raw.2 .0, raw.5 .0, "canonical template width");
    assert_eq!(raw.2 .6, raw.5 .1, "canonical template local count");
    assert_eq!(raw.2 .7, raw.5 .2, "canonical template output start");

    let rate = word(raw.2 .1);
    assert_ne!(rate, 0, "canonical Poseidon rate");
    let local_count = word(raw.5 .1);
    let output_local_start = word(raw.5 .2);
    let assignment = Assignment {
        private_values,
        public_values,
        constant_column: word(raw.3 .2),
        total_columns: word(raw.3 .4),
        unavailable_private: None,
        changed_column: None,
    };
    assert_eq!(private_values.len(), assignment.constant_column);
    assert_eq!(public_values.len(), word(raw.3 .3));
    assert_eq!(
        assignment.total_columns,
        assignment.constant_column + 1 + public_values.len(),
    );
    assert!(private_values
        .iter()
        .chain(public_values)
        .all(|value| *value < GOLDILOCKS_MODULUS));

    let schedule = events(&raw);
    let mut row_cursor = 0;
    for &event in &schedule {
        assert_eq!(event.row_start(), row_cursor, "canonical raw row schedule");
        row_cursor += event_row_count(event, &raw);
    }
    assert_eq!(row_cursor, word(raw.3 .0), "canonical raw row coverage");

    schedule.par_iter().try_for_each(|&event| {
        for ordinal in 0..event_row_count(event, &raw) {
            let row = event.row_start() + ordinal;
            let [a, b, c] = [Side::A, Side::B, Side::C].map(|side| {
                event_value(
                    event,
                    ordinal,
                    &raw.5,
                    side,
                    rate,
                    local_count,
                    output_local_start,
                    &assignment,
                )
            });
            if mul_mod(a, b) != c {
                return Err(row);
            }
        }
        Ok(())
    })?;

    Ok(row_cursor)
}

/// Check exactly the Pilot and PiCCS row prefix in one canonical schema-8
/// Lean raw package. The function still checks the complete physical row
/// schedule before it selects the prefix. `Err` contains one unsatisfied
/// prefix row.
pub fn evaluate_pi_ccs_prefix_assignment(
    bytes: &[u8],
    private_values: &[u64],
    public_values: &[u64],
) -> Result<usize, usize> {
    const PI_CCS_ROW_END: usize = 19_936_967;
    const PI_CCS_PRIVATE_END: usize = 20_064_545;

    let raw: RawPackage = serde_json::from_slice(bytes).expect("canonical raw-package decode");
    assert_eq!(raw.0, 8, "canonical raw-package schema");
    assert_eq!(raw.3 .1, raw.3 .2, "canonical private/constant boundary");
    assert_eq!(raw.2 .0, raw.5 .0, "canonical template width");
    assert_eq!(raw.2 .6, raw.5 .1, "canonical template local count");
    assert_eq!(raw.2 .7, raw.5 .2, "canonical template output start");

    let rate = word(raw.2 .1);
    assert_ne!(rate, 0, "canonical Poseidon rate");
    let local_count = word(raw.5 .1);
    let output_local_start = word(raw.5 .2);
    // This assignment contains no private suffix. The existing indexed read
    // therefore fails if any selected row reads a private column at or after
    // the PiCCS boundary.
    let assignment = Assignment {
        private_values,
        public_values,
        constant_column: word(raw.3 .2),
        total_columns: word(raw.3 .4),
        unavailable_private: None,
        changed_column: None,
    };
    assert_eq!(private_values.len(), PI_CCS_PRIVATE_END);
    assert!(PI_CCS_PRIVATE_END < assignment.constant_column);
    assert_eq!(public_values.len(), word(raw.3 .3));
    assert_eq!(
        assignment.total_columns,
        assignment.constant_column + 1 + public_values.len(),
    );
    assert!(private_values
        .iter()
        .chain(public_values)
        .all(|value| *value < GOLDILOCKS_MODULUS));

    let schedule = events(&raw);
    let mut row_cursor = 0;
    for &event in &schedule {
        assert_eq!(event.row_start(), row_cursor, "canonical raw row schedule");
        row_cursor += event_row_count(event, &raw);
    }
    assert_eq!(row_cursor, word(raw.3 .0), "canonical raw row coverage");

    let prefix_event_count = schedule.partition_point(|event| event.row_start() < PI_CCS_ROW_END);
    let prefix = &schedule[..prefix_event_count];
    let suffix = &schedule[prefix_event_count..];
    assert!(!prefix.is_empty(), "canonical PiCCS prefix rows");
    assert_eq!(
        prefix
            .last()
            .map(|event| event.row_start() + event_row_count(*event, &raw)),
        Some(PI_CCS_ROW_END),
        "canonical PiCCS row end",
    );
    assert_eq!(
        suffix.first().map(|event| event.row_start()),
        Some(PI_CCS_ROW_END),
        "canonical first PiRLC row",
    );
    match suffix.first().copied() {
        Some(Event::Permutation {
            row_start,
            invocation: Invocation::Explicit(invocation),
        }) => {
            assert_eq!(row_start, PI_CCS_ROW_END, "canonical first PiRLC row");
            assert_eq!(invocation.0, 7, "canonical first PiRLC phase");
            assert_eq!(word(invocation.2), PI_CCS_PRIVATE_END, "canonical first PiRLC column");
        }
        _ => panic!("canonical first PiRLC event"),
    }
    assert!(prefix
        .iter()
        .all(|event| event.row_start() + event_row_count(*event, &raw) <= PI_CCS_ROW_END));
    assert!(suffix
        .iter()
        .all(|event| event.row_start() >= PI_CCS_ROW_END));

    prefix.par_iter().try_for_each(|&event| {
        for ordinal in 0..event_row_count(event, &raw) {
            let row = event.row_start() + ordinal;
            let [a, b, c] = [Side::A, Side::B, Side::C].map(|side| {
                event_value(
                    event,
                    ordinal,
                    &raw.5,
                    side,
                    rate,
                    local_count,
                    output_local_start,
                    &assignment,
                )
            });
            if mul_mod(a, b) != c {
                return Err(row);
            }
        }
        Ok(())
    })?;

    Ok(PI_CCS_ROW_END)
}

#[derive(Debug)]
pub struct PilotAssignmentReport {
    pub rows: usize,
    pub public_mutations: usize,
    pub generated_mutations: usize,
    pub row_mutations: usize,
    pub column_mutations: usize,
}

fn pilot_hash_row_mutations(raw: &RawPackage, assignment: &Assignment<'_>) -> (usize, usize) {
    let mut row_mutations = 0;
    let mut column_mutations = 0;
    for chain in &raw.6 {
        // Each owner starts with the first row of its first hash invocation.
        let event = Event::Permutation {
            row_start: word(chain.1),
            invocation: Invocation::Hash { chain, ordinal: 0 },
        };
        let holds = |template: &RawTemplate| {
            let [a, b, c] = [Side::A, Side::B, Side::C].map(|side| {
                event_value(
                    event,
                    0,
                    template,
                    side,
                    word(raw.2 .1),
                    word(raw.5 .1),
                    word(raw.5 .2),
                    assignment,
                )
            });
            mul_mod(a, b) == c
        };
        assert!(holds(&raw.5), "pilot hash owner {} original row", chain.0);

        let mut changed = raw.5.clone();
        changed.3[0].3 .0 = add_mod(changed.3[0].3 .0, 1);
        assert!(!holds(&changed), "pilot hash owner {} canonical row mutation", chain.0);
        row_mutations += 1;
        println!(
            "pilot physical owner {}: row {} C constant mutation rejected",
            chain.0, chain.1
        );

        let mut rejected = false;
        'sides: for side in [Side::A, Side::B, Side::C] {
            for (term_index, term) in template_side(&raw.5 .3[0], side).1.iter().enumerate() {
                // These are the exact input and local column domains in the
                // decoded permutation schema, not a sample of assignment data.
                for replacement in (0..word(raw.5 .0))
                    .map(|lane| RawColumnRef(0, lane as u64))
                    .chain((0..word(raw.5 .1)).map(|lane| RawColumnRef(1, lane as u64)))
                {
                    if (replacement.0, replacement.1) == (term.0 .0, term.0 .1) {
                        continue;
                    }
                    let mut changed = raw.5.clone();
                    template_side_mut(&mut changed.3[0], side).1[term_index].0 = replacement.clone();
                    if !holds(&changed) {
                        println!(
                            "pilot physical owner {}: row {} column ({}, {}) -> ({}, {}) rejected",
                            chain.0, chain.1, term.0 .0, term.0 .1, replacement.0, replacement.1
                        );
                        rejected = true;
                        column_mutations += 1;
                        break 'sides;
                    }
                }
            }
        }
        assert!(
            rejected,
            "pilot hash owner {} has no effective canonical column mutation",
            chain.0
        );
    }
    (row_mutations, column_mutations)
}

fn evaluate_event_range(
    raw: &RawPackage,
    schedule: &[Event<'_>],
    assignment: &Assignment<'_>,
    range: Range<usize>,
) -> Result<usize, usize> {
    let first = schedule.partition_point(|event| event.row_start() + event_row_count(*event, raw) <= range.start);
    let end = schedule.partition_point(|event| event.row_start() < range.end);
    schedule[first..end].par_iter().try_for_each(|&event| {
        let start = range.start.saturating_sub(event.row_start());
        let end = (range.end - event.row_start()).min(event_row_count(event, raw));
        for ordinal in start..end {
            let [a, b, c] = [Side::A, Side::B, Side::C].map(|side| {
                event_value(
                    event,
                    ordinal,
                    &raw.5,
                    side,
                    word(raw.2 .1),
                    word(raw.5 .1),
                    word(raw.5 .2),
                    assignment,
                )
            });
            if mul_mod(a, b) != c {
                return Err(event.row_start() + ordinal);
            }
        }
        Ok(())
    })?;
    Ok(range.len())
}

/// Check every canonical pilot row on the standalone pilot assignment.
/// The proof-input gap, private suffix, and non-pilot public context are
/// unavailable. Mutations use the same raw evaluator and exact binding rows.
pub fn evaluate_pilot_assignment(
    bytes: &[u8],
    private_values: &[u64],
    public_values: &[u64],
) -> Result<PilotAssignmentReport, usize> {
    // PilotProduction.physicalRowCountValue_eq and PilotValues fix the
    // rows. Stage1.sourceToSpartan relocates the pilot private boundary.
    const PILOT_ROW_END: usize = 14_623_730;
    const PILOT_PRIVATE_END: usize = 14_751_526;
    const PILOT_PUBLIC_COUNT: usize = 274;
    let raw: RawPackage = serde_json::from_slice(bytes).expect("canonical pilot raw-package decode");
    assert_eq!(raw.0, 8, "canonical pilot raw-package schema");
    assert_eq!(raw.3 .1, raw.3 .2, "canonical private/constant boundary");
    assert_eq!((raw.2 .0, raw.2 .1, raw.2 .6, raw.2 .7), (8, 4, 592, 584));
    assert_eq!((raw.5 .0, raw.5 .1, raw.5 .2), (8, 592, 584));
    assert_eq!(private_values.len(), PILOT_PRIVATE_END);
    assert_eq!(public_values.len(), word(raw.3 .3));
    assert_eq!(public_values.len(), PILOT_PUBLIC_COUNT + 4);
    assert_eq!(word(raw.3 .4), word(raw.3 .2) + 1 + public_values.len());
    assert!(private_values
        .iter()
        .chain(public_values)
        .all(|value| *value < GOLDILOCKS_MODULUS));
    assert_eq!(raw.6.len(), 2, "pilot hash-owner count");
    let prior = &raw.6[0];
    let output = &raw.6[1];
    assert_eq!(
        (prior.0, prior.1, prior.3, prior.4, prior.5),
        (1, 0, 0, 49_393, 128_074)
    );
    assert_eq!((output.0, output.1, output.3, output.4), (2, 7_312_526, 49_393, 49_393));
    assert_eq!(word(output.1 + output.2), PILOT_ROW_END);
    let binding_rows = [
        word(prior.1 + prior.6)..word(output.1),
        word(output.1 + output.6)..PILOT_ROW_END,
    ];

    let schedule = events(&raw);
    let mut row_cursor = 0;
    for &event in &schedule {
        assert_eq!(event.row_start(), row_cursor, "canonical raw row schedule");
        row_cursor += event_row_count(event, &raw);
    }
    assert_eq!(row_cursor, word(raw.3 .0), "canonical raw row coverage");
    let suffix = schedule.partition_point(|event| event.row_start() < PILOT_ROW_END);
    assert_eq!(schedule[suffix].row_start(), PILOT_ROW_END, "first PiCCS assertion row");
    assert_eq!(
        schedule[suffix - 1].row_start() + event_row_count(schedule[suffix - 1], &raw),
        PILOT_ROW_END
    );

    let mut assignment = Assignment {
        private_values,
        public_values: &public_values[..PILOT_PUBLIC_COUNT],
        constant_column: word(raw.3 .2),
        total_columns: word(raw.3 .4),
        unavailable_private: Some(98_786..128_074),
        changed_column: None,
    };
    let rows = evaluate_event_range(&raw, &schedule, &assignment, 0..PILOT_ROW_END)?;
    println!("independent pilot physical rows: {rows}");
    let (row_mutations, column_mutations) = pilot_hash_row_mutations(&raw, &assignment);

    let mut public_mutations = 0;
    for public in 0..PILOT_PUBLIC_COUNT {
        assignment.changed_column = Some(assignment.constant_column + 1 + public);
        let owner = usize::from(public >= PILOT_PUBLIC_COUNT - word(raw.2 .1));
        assert!(
            evaluate_event_range(&raw, &schedule, &assignment, binding_rows[owner].clone()).is_err(),
            "pilot public word {public} mutation must reject",
        );
        public_mutations += 1;
    }

    let mut generated_mutations = 0;
    for (owner, chain) in raw.6.iter().enumerate() {
        assignment.changed_column = Some(word(chain.5));
        assert!(
            evaluate_event_range(&raw, &schedule, &assignment, word(chain.1)..word(chain.1) + 1).is_err(),
            "pilot hash owner {owner} first generated value mutation must reject",
        );
        generated_mutations += 1;
        for lane in 0..word(raw.2 .1) {
            assignment.changed_column = Some(word(chain.5) + word(chain.7) * word(raw.5 .1) + word(raw.5 .2) + lane);
            assert!(
                evaluate_event_range(&raw, &schedule, &assignment, binding_rows[owner].clone()).is_err(),
                "pilot hash owner {owner} generated digest lane {lane} mutation must reject",
            );
            generated_mutations += 1;
        }
    }
    Ok(PilotAssignmentReport {
        rows,
        public_mutations,
        generated_mutations,
        row_mutations,
        column_mutations,
    })
}
