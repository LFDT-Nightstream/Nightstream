use nightstream_fprime::PackageSparseMatrix;

use serde::de::IgnoredAny;
use serde::Deserialize;
use serde_json::Value;

#[allow(dead_code)]
mod canonical_assignment;
#[allow(dead_code)]
mod independent_assignment;
mod owner_mutations;
#[allow(dead_code)]
mod raw_assignment;
mod reference;
#[allow(unused_imports)]
pub use canonical_assignment::{
    evaluate_canonical_assignment, evaluate_pi_ccs_prefix_assignment, evaluate_pilot_assignment,
};
#[allow(unused_imports)]
pub use independent_assignment::{
    check_piccs_owner_mutations, compare_lean_expanded_matrices, compare_sealed_matrices,
};
use reference::*;

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
struct RawPoseidonSchedule(
    IgnoredAny,
    u64,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
);

#[derive(Deserialize)]
struct RawLayout(u64, u64, u64, u64, u64, IgnoredAny, Vec<RawSegment>);

#[derive(Deserialize)]
struct RawSegment(u64, u64, u64);

#[derive(Deserialize)]
struct RawTemplate(u64, u64, u64, Vec<RawTemplateRow>);

#[derive(Deserialize)]
struct RawTemplateRow(
    u64,
    RawTemplateCombination,
    RawTemplateCombination,
    RawTemplateCombination,
);

#[derive(Deserialize)]
struct RawTemplateCombination(u64, Vec<RawTemplateTerm>);

#[derive(Deserialize)]
struct RawTemplateTerm(RawColumnRef, u64);

#[derive(Deserialize)]
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
enum MatrixSide {
    A,
    B,
    C,
}

#[derive(Clone, Copy, Debug)]
struct OwnerSpan {
    name: &'static str,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug)]
struct ColumnOwnerSpan {
    name: &'static str,
    rows: OwnerSpan,
    columns: OwnerSpan,
}

#[derive(Debug)]
struct OwnershipInventory {
    row_spans: Vec<OwnerSpan>,
    column_spans: Vec<OwnerSpan>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiCcsOwnerMutationReport {
    pub row_families: usize,
    pub row_mutations: usize,
    pub column_families: usize,
    pub zero_column_families: usize,
    pub column_mutations: usize,
    pub public_segments: usize,
    pub public_mutations: usize,
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
            Self::Witness(row) => word(row.0),
            Self::Assertion(row) => word(row.0),
        }
    }
}

fn add_term(terms: &mut Vec<(usize, u64)>, column: usize, coefficient: u64) {
    if coefficient != 0 {
        terms.push((column, coefficient));
    }
}

fn canonicalize(mut terms: Vec<(usize, u64)>) -> Vec<(usize, u64)> {
    terms.sort_unstable_by_key(|term| term.0);
    let mut canonical: Vec<(usize, u64)> = Vec::with_capacity(terms.len());
    for (column, coefficient) in terms {
        match canonical.last_mut() {
            Some((last_column, last_coefficient)) if *last_column == column => {
                *last_coefficient = add_mod(*last_coefficient, coefficient);
                if *last_coefficient == 0 {
                    canonical.pop();
                }
            }
            _ => {
                canonical.push((column, coefficient));
            }
        }
    }
    canonical
}

fn sparse_terms(combination: &RawCombination, layout: &ReferenceLayout) -> Vec<(usize, u64)> {
    let mut terms = Vec::with_capacity(combination.1.len() + 1);
    add_term(&mut terms, layout.constant_column(), combination.0);
    for term in &combination.1 {
        add_term(&mut terms, layout.map_column(word(term.0)), term.1);
    }
    canonicalize(terms)
}

fn explicit_input_terms(
    coefficient: u64,
    input: &RawCombination,
    layout: &ReferenceLayout,
    terms: &mut Vec<(usize, u64)>,
) {
    add_term(terms, layout.constant_column(), mul_mod(coefficient, input.0));
    for term in &input.1 {
        add_term(terms, layout.map_column(word(term.0)), mul_mod(coefficient, term.1));
    }
}

fn template_terms(
    combination: &RawTemplateCombination,
    invocation: Invocation<'_>,
    layout: &ReferenceLayout,
) -> Vec<(usize, u64)> {
    let mut terms = Vec::with_capacity(combination.1.len() * 3 + 1);
    add_term(&mut terms, layout.constant_column(), combination.0);
    for term in &combination.1 {
        let lane = word(term.0 .1);
        match term.0 .0 {
            1 => {
                let witness_start = match invocation {
                    Invocation::Hash { chain, ordinal } => word(chain.5) + ordinal * 592,
                    Invocation::Explicit(invocation) => word(invocation.2),
                };
                add_term(&mut terms, witness_start + lane, term.1);
            }
            0 => match invocation {
                Invocation::Hash { chain, ordinal } => {
                    if ordinal > 0 {
                        add_term(&mut terms, word(chain.5) + (ordinal - 1) * 592 + 584 + lane, term.1);
                    }
                    let absorb_count = word(chain.7);
                    if ordinal < absorb_count {
                        let input_offset = ordinal * 4 + lane;
                        if lane < 4 && input_offset < word(chain.4) {
                            add_term(&mut terms, word(chain.3) + input_offset, term.1);
                        }
                    } else if lane == 0 {
                        add_term(&mut terms, layout.constant_column(), term.1);
                    }
                }
                Invocation::Explicit(invocation) => {
                    explicit_input_terms(term.1, &invocation.3[lane], layout, &mut terms);
                }
            },
            _ => panic!("reference template column tag"),
        }
    }
    canonicalize(terms)
}

fn compact_input_column(invocation: &RawCompactInvocation, input: usize) -> usize {
    for range in &invocation.4 {
        let input_start = word(range.0);
        let input_count = word(range.1);
        if input_start <= input && input < input_start + input_count {
            return word(range.2) + (input - input_start) * word(range.3);
        }
    }
    panic!("reference compact input coverage")
}

fn compact_terms(
    combination: &RawTemplateCombination,
    invocation: &RawCompactInvocation,
    layout: &ReferenceLayout,
) -> Vec<(usize, u64)> {
    let mut terms = Vec::with_capacity(combination.1.len() + 1);
    add_term(&mut terms, layout.constant_column(), combination.0);
    for term in &combination.1 {
        let index = word(term.0 .1);
        let column = match term.0 .0 {
            0 => compact_input_column(invocation, index),
            1 => word(invocation.3) + index,
            _ => panic!("reference compact column tag"),
        };
        add_term(&mut terms, layout.map_column(column), term.1);
    }
    canonicalize(terms)
}

fn template_side(row: &RawTemplateRow, side: MatrixSide) -> &RawTemplateCombination {
    match side {
        MatrixSide::A => &row.1,
        MatrixSide::B => &row.2,
        MatrixSide::C => &row.3,
    }
}

fn compact_side(row: &RawCompactRow, side: MatrixSide) -> &RawTemplateCombination {
    match side {
        MatrixSide::A => &row.1,
        MatrixSide::B => &row.2,
        MatrixSide::C => &row.3,
    }
}

fn row_side(row: &RawRow, side: MatrixSide) -> &RawCombination {
    match side {
        MatrixSide::A => &row.1,
        MatrixSide::B => &row.2,
        MatrixSide::C => &row.3,
    }
}

fn expected_row(
    event: Event<'_>,
    template: &RawTemplate,
    template_ordinal: usize,
    side: MatrixSide,
    layout: &ReferenceLayout,
) -> Vec<(usize, u64)> {
    match event {
        Event::Permutation { invocation, .. } => {
            let row = &template.3[template_ordinal];
            assert_eq!(word(row.0), template_ordinal);
            template_terms(template_side(row, side), invocation, layout)
        }
        Event::Compact { invocation, template } => {
            compact_terms(compact_side(&template.4[template_ordinal], side), invocation, layout)
        }
        Event::Witness(instruction) => match side {
            MatrixSide::A => sparse_terms(&instruction.2, layout),
            MatrixSide::B => sparse_terms(&instruction.3, layout),
            MatrixSide::C => vec![(layout.map_column(word(instruction.1)), 1)],
        },
        Event::Assertion(row) => sparse_terms(row_side(row, side), layout),
    }
}

fn actual_row(matrix: &PackageSparseMatrix, row: usize) -> Vec<(usize, u64)> {
    let start = matrix.row_offsets()[row];
    let end = matrix.row_offsets()[row + 1];
    matrix.column_indices()[start..end]
        .iter()
        .copied()
        .zip(matrix.values()[start..end].iter().copied())
        .collect()
}

fn exact_row_accepts(matrix: &PackageSparseMatrix, row: usize, candidate: &[(usize, u64)]) -> bool {
    actual_row(matrix, row) == candidate
}

fn compare_row(matrix: &PackageSparseMatrix, row: usize, expected: &[(usize, u64)], side: &str) {
    if !exact_row_accepts(matrix, row, expected) {
        assert_eq!(actual_row(matrix, row), expected, "{side} row {row}");
    }
}

fn events(raw: &RawPackage) -> Vec<Event<'_>> {
    let template_rows = raw.5 .3.len();
    let mut events = Vec::new();
    for chain in &raw.6 {
        assert_ne!(chain.0, 0, "reference hash-chain phase");
        assert_eq!(
            word(chain.2),
            word(chain.6) + word(chain.8),
            "reference hash-chain rows",
        );
        assert_eq!(
            word(chain.6),
            (word(chain.7) + 1) * template_rows,
            "reference hash-chain witness rows",
        );
        assert!(
            word(chain.8) == 0
                || (word(chain.9) >= word(raw.3 .2) + 1 && word(chain.9) + word(chain.8) <= word(raw.3 .4)),
            "reference hash-chain digest range",
        );
        for ordinal in 0..=word(chain.7) {
            events.push(Event::Permutation {
                row_start: word(chain.1) + ordinal * template_rows,
                invocation: Invocation::Hash { chain, ordinal },
            });
        }
    }
    events.extend(raw.7.iter().map(|invocation| {
        assert_ne!(invocation.0, 0, "reference invocation phase");
        Event::Permutation {
            row_start: word(invocation.1),
            invocation: Invocation::Explicit(invocation),
        }
    }));
    events.extend(raw.9.iter().map(|invocation| {
        assert_ne!(invocation.0, 0, "reference compact phase");
        let template = &raw.8[word(invocation.1)];
        assert!(word(template.2) < word(template.0), "reference compact output input");
        assert_eq!(template.4.len(), word(template.1) + 1, "reference compact rows");
        let mut input_cursor = 0usize;
        for range in &invocation.4 {
            assert_eq!(word(range.0), input_cursor, "reference compact input order");
            assert_ne!(word(range.1), 0, "reference compact input count");
            assert_ne!(word(range.3), 0, "reference compact column stride");
            input_cursor += word(range.1);
        }
        assert_eq!(input_cursor, word(template.0), "reference compact input coverage");
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

/// The separately emitted physical reference must be the exact inner
/// package of the sealed candidate used for every later check.
pub fn require_sealed_expansion(sealed_bytes: &[u8], expanded_bytes: &[u8]) {
    let sealed: Value = serde_json::from_slice(sealed_bytes).expect("sealed candidate JSON");
    let inner = sealed
        .as_array()
        .and_then(|fields| fields.get(1))
        .expect("sealed candidate inner package");
    let mut canonical = serde_json::to_vec(inner).expect("canonical inner package");
    canonical.push(b'\n');
    assert_eq!(canonical.len(), expanded_bytes.len(), "Lean physical expansion length");
    if let Some(index) = canonical
        .iter()
        .zip(expanded_bytes)
        .position(|(actual, expected)| actual != expected)
    {
        panic!("sealed candidate and Lean physical expansion differ at byte {index}");
    }
}
