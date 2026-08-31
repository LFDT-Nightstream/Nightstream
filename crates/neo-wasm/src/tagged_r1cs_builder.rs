//! WASM constraint ownership and catalog diagnostics.
//!
//! Generic row construction, tagging, and storage live in `neo-application`.

use std::collections::{BTreeMap, BTreeSet};

use neo_application::{ConstraintCatalog, ConstraintTag, R1csBuilder, R1csRow, TaggedR1csBuilder};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::isa::WasmOpcode;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WasmConstraintScope {
    /// Relation-wide VM invariant without a narrower semantic owner.
    Always,
    /// Constraint owned by the host-event state machine. The label identifies
    /// its event-binding, buffering, or permutation phase.
    HostEvent,
    Opcode(WasmOpcode),
    Opcodes(Box<[WasmOpcode]>),
}

impl WasmConstraintScope {
    pub fn applies_to(&self, opcode: WasmOpcode) -> bool {
        match self {
            Self::Always | Self::HostEvent => true,
            Self::Opcode(single) => *single == opcode,
            Self::Opcodes(opcodes) => opcodes.contains(&opcode),
        }
    }
}

impl std::fmt::Display for WasmConstraintScope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Always => formatter.write_str("Always"),
            Self::HostEvent => formatter.write_str("HostEvent"),
            Self::Opcode(opcode) => write!(formatter, "Opcode({})", opcode.name()),
            Self::Opcodes(opcodes) => {
                formatter.write_str("Opcodes(")?;
                for (index, opcode) in opcodes.iter().enumerate() {
                    if index > 0 {
                        formatter.write_str(", ")?;
                    }
                    formatter.write_str(opcode.name())?;
                }
                formatter.write_str(")")
            }
        }
    }
}

pub type WasmConstraintTag = ConstraintTag<WasmConstraintScope>;
pub type WasmConstraintCatalog = ConstraintCatalog<WasmConstraintScope>;
pub type WasmR1csBuilder = R1csBuilder<WasmConstraintScope>;
pub type WasmTaggedR1csBuilder<'a> = TaggedR1csBuilder<'a, WasmConstraintScope>;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct WasmNormalizedTerms(pub Vec<(usize, u64)>);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct WasmConstraintBodyFingerprint {
    pub a_terms: WasmNormalizedTerms,
    pub b_terms: WasmNormalizedTerms,
    pub c_terms: WasmNormalizedTerms,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmDuplicateConstraintGroup {
    pub fingerprint: WasmConstraintBodyFingerprint,
    pub rows: Vec<usize>,
    pub selector_a_terms_by_row: Vec<Vec<(usize, u64)>>,
    pub selector_b_terms_by_row: Vec<Vec<(usize, u64)>>,
}

pub fn count_always_rows(catalog: &WasmConstraintCatalog) -> usize {
    catalog
        .rows()
        .iter()
        .filter(|row| matches!(row.tag().owner(), WasmConstraintScope::Always))
        .count()
}

pub fn always_rows(catalog: &WasmConstraintCatalog) -> Vec<usize> {
    catalog
        .rows()
        .iter()
        .enumerate()
        .filter_map(|(index, row)| matches!(row.tag().owner(), WasmConstraintScope::Always).then_some(index))
        .collect()
}

pub fn host_event_rows(catalog: &WasmConstraintCatalog) -> Vec<usize> {
    catalog
        .rows()
        .iter()
        .enumerate()
        .filter_map(|(index, row)| matches!(row.tag().owner(), WasmConstraintScope::HostEvent).then_some(index))
        .collect()
}

pub fn count_host_event_rows(catalog: &WasmConstraintCatalog) -> usize {
    catalog
        .rows()
        .iter()
        .filter(|row| matches!(row.tag().owner(), WasmConstraintScope::HostEvent))
        .count()
}

pub fn rows_owned_by_opcode(catalog: &WasmConstraintCatalog, opcode: WasmOpcode) -> Vec<usize> {
    catalog
        .rows()
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            matches!(row.tag().owner(), WasmConstraintScope::Opcode(single) if *single == opcode).then_some(index)
        })
        .collect()
}

pub fn shared_rows_for_opcode(catalog: &WasmConstraintCatalog, opcode: WasmOpcode) -> Vec<usize> {
    catalog
        .rows()
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            matches!(row.tag().owner(), WasmConstraintScope::Opcodes(opcodes) if opcodes.contains(&opcode))
                .then_some(index)
        })
        .collect()
}

pub fn count_shared_rows_for_opcode(catalog: &WasmConstraintCatalog, opcode: WasmOpcode) -> usize {
    catalog
        .rows()
        .iter()
        .filter(|row| matches!(row.tag().owner(), WasmConstraintScope::Opcodes(opcodes) if opcodes.contains(&opcode)))
        .count()
}

pub fn rows_for_opcode(catalog: &WasmConstraintCatalog, opcode: WasmOpcode) -> Vec<usize> {
    catalog
        .rows()
        .iter()
        .enumerate()
        .filter_map(|(index, row)| row.tag().owner().applies_to(opcode).then_some(index))
        .collect()
}

pub fn count_owned_by_opcode(catalog: &WasmConstraintCatalog, opcode: WasmOpcode) -> usize {
    catalog
        .rows()
        .iter()
        .filter(|row| matches!(row.tag().owner(), WasmConstraintScope::Opcode(single) if *single == opcode))
        .count()
}

pub fn count_for_opcode(catalog: &WasmConstraintCatalog, opcode: WasmOpcode) -> usize {
    catalog
        .rows()
        .iter()
        .filter(|row| row.tag().owner().applies_to(opcode))
        .count()
}

pub fn duplicate_bodies_ignoring_selectors(
    catalog: &WasmConstraintCatalog,
    selector_cols: &[usize],
) -> Vec<WasmDuplicateConstraintGroup> {
    let selector_cols: BTreeSet<_> = selector_cols.iter().copied().collect();
    let mut groups: BTreeMap<WasmConstraintBodyFingerprint, Vec<(usize, Vec<(usize, u64)>, Vec<(usize, u64)>)>> =
        BTreeMap::new();

    for (row_index, tagged) in catalog.rows().iter().enumerate() {
        let row = tagged.row();
        let gate_shape = is_selector_gate_row(row, &selector_cols);
        let (selector_a_terms, non_selector_a_terms): (Vec<_>, Vec<_>) = if gate_shape {
            row.a_terms()
                .iter()
                .map(|&(column, coefficient)| (column, coefficient.as_canonical_u64()))
                .partition(|(column, _)| selector_cols.contains(column))
        } else {
            (Vec::new(), normalize_terms(row.a_terms()).0)
        };
        let (selector_b_terms, non_selector_b_terms): (Vec<_>, Vec<_>) = if gate_shape {
            row.b_terms()
                .iter()
                .map(|&(column, coefficient)| (column, coefficient.as_canonical_u64()))
                .partition(|(column, _)| selector_cols.contains(column))
        } else {
            (Vec::new(), normalize_terms(row.b_terms()).0)
        };
        let fingerprint = WasmConstraintBodyFingerprint {
            a_terms: WasmNormalizedTerms(non_selector_a_terms),
            b_terms: WasmNormalizedTerms(non_selector_b_terms),
            c_terms: normalize_terms(row.c_terms()),
        };
        groups
            .entry(fingerprint)
            .or_default()
            .push((row_index, selector_a_terms, selector_b_terms));
    }

    groups
        .into_iter()
        .filter_map(|(fingerprint, rows)| {
            if rows.len() < 2 {
                return None;
            }
            let mut distinct_selector_sets = rows
                .iter()
                .map(|(_, selector_a, selector_b)| (selector_a, selector_b))
                .collect::<Vec<_>>();
            distinct_selector_sets.sort();
            distinct_selector_sets.dedup();
            (distinct_selector_sets.len() >= 2).then(|| WasmDuplicateConstraintGroup {
                fingerprint,
                rows: rows.iter().map(|(row_index, _, _)| *row_index).collect(),
                selector_a_terms_by_row: rows
                    .iter()
                    .map(|(_, selector_a, _)| selector_a.clone())
                    .collect(),
                selector_b_terms_by_row: rows
                    .into_iter()
                    .map(|(_, _, selector_b)| selector_b)
                    .collect(),
            })
        })
        .collect()
}

fn is_selector_gate_row(row: &R1csRow, selector_cols: &BTreeSet<usize>) -> bool {
    row.c_terms().is_empty()
        && row.a_terms().len() == 1
        && row.a_terms()[0].1 == F::ONE
        && selector_cols.contains(&row.a_terms()[0].0)
        && !row
            .b_terms()
            .iter()
            .any(|(column, _)| selector_cols.contains(column))
}

fn normalize_terms(terms: &[(usize, F)]) -> WasmNormalizedTerms {
    WasmNormalizedTerms(
        terms
            .iter()
            .map(|&(column, coefficient)| (column, coefficient.as_canonical_u64()))
            .collect(),
    )
}
