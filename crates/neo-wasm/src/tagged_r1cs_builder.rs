//! Wraps the simple R1csBuilder with tagging capabilities, so that it's
//! possible to dump labeled constraints, for debugging and optimizing
//! plus maybe exporting for formal verification

use std::collections::BTreeMap;

use super::isa::WasmOpcode;
use crate::r1cs_builder::R1csBuilder;
use neo_ccs::CcsStructure;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WasmConstraintScope {
    /// Relation-wide VM invariant without a narrower semantic owner.
    Always,
    /// Constraint owned by the host-event state machine. The label identifies
    /// its grammar, buffering, or permutation phase.
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmConstraintTag {
    pub label: &'static str,
    pub scope: WasmConstraintScope,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WasmConstraintCatalog {
    pub row_tags: Vec<WasmConstraintTag>,
    pub rows: Vec<WasmR1csRow>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmR1csRow {
    pub a_terms: Vec<(usize, F)>,
    pub b_terms: Vec<(usize, F)>,
    pub c_terms: Vec<(usize, F)>,
}

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

impl WasmConstraintCatalog {
    pub fn count_always_rows(&self) -> usize {
        self.row_tags
            .iter()
            .filter(|tag| matches!(&tag.scope, WasmConstraintScope::Always))
            .count()
    }

    pub fn always_rows(&self) -> Vec<usize> {
        self.row_tags
            .iter()
            .enumerate()
            .filter_map(|(row, tag)| matches!(&tag.scope, WasmConstraintScope::Always).then_some(row))
            .collect()
    }

    pub fn host_event_rows(&self) -> Vec<usize> {
        self.row_tags
            .iter()
            .enumerate()
            .filter_map(|(row, tag)| matches!(&tag.scope, WasmConstraintScope::HostEvent).then_some(row))
            .collect()
    }

    pub fn count_host_event_rows(&self) -> usize {
        self.row_tags
            .iter()
            .filter(|tag| matches!(&tag.scope, WasmConstraintScope::HostEvent))
            .count()
    }

    pub fn rows_owned_by_opcode(&self, opcode: WasmOpcode) -> Vec<usize> {
        self.row_tags
            .iter()
            .enumerate()
            .filter_map(|(row, tag)| {
                matches!(&tag.scope, WasmConstraintScope::Opcode(single) if *single == opcode).then_some(row)
            })
            .collect()
    }

    pub fn shared_rows_for_opcode(&self, opcode: WasmOpcode) -> Vec<usize> {
        self.row_tags
            .iter()
            .enumerate()
            .filter_map(|(row, tag)| {
                matches!(&tag.scope, WasmConstraintScope::Opcodes(opcodes) if opcodes.contains(&opcode)).then_some(row)
            })
            .collect()
    }

    pub fn count_shared_rows_for_opcode(&self, opcode: WasmOpcode) -> usize {
        self.row_tags
            .iter()
            .filter(|tag| matches!(&tag.scope, WasmConstraintScope::Opcodes(opcodes) if opcodes.contains(&opcode)))
            .count()
    }

    pub fn rows_for_opcode(&self, opcode: WasmOpcode) -> Vec<usize> {
        self.row_tags
            .iter()
            .enumerate()
            .filter_map(|(row, tag)| tag.scope.applies_to(opcode).then_some(row))
            .collect()
    }

    pub fn count_owned_by_opcode(&self, opcode: WasmOpcode) -> usize {
        self.row_tags
            .iter()
            .filter(|tag| matches!(&tag.scope, WasmConstraintScope::Opcode(single) if *single == opcode))
            .count()
    }

    pub fn count_for_opcode(&self, opcode: WasmOpcode) -> usize {
        self.row_tags
            .iter()
            .filter(|tag| tag.scope.applies_to(opcode))
            .count()
    }

    pub fn duplicate_bodies_ignoring_selectors(&self, selector_cols: &[usize]) -> Vec<WasmDuplicateConstraintGroup> {
        let selector_cols: std::collections::BTreeSet<_> = selector_cols.iter().copied().collect();
        let mut groups: BTreeMap<WasmConstraintBodyFingerprint, Vec<(usize, Vec<(usize, u64)>, Vec<(usize, u64)>)>> =
            BTreeMap::new();

        for (row_idx, row) in self.rows.iter().enumerate() {
            let gate_shape = is_selector_gate_row(row, &selector_cols);
            let (selector_a_terms, non_selector_a_terms): (Vec<_>, Vec<_>) = if gate_shape {
                row.a_terms
                    .iter()
                    .map(|&(col, coeff)| (col, coeff.as_canonical_u64()))
                    .partition(|(col, _)| selector_cols.contains(col))
            } else {
                (Vec::new(), normalize_terms(&row.a_terms).0)
            };
            let (selector_b_terms, non_selector_b_terms): (Vec<_>, Vec<_>) = if gate_shape {
                row.b_terms
                    .iter()
                    .map(|&(col, coeff)| (col, coeff.as_canonical_u64()))
                    .partition(|(col, _)| selector_cols.contains(col))
            } else {
                (Vec::new(), normalize_terms(&row.b_terms).0)
            };
            let fingerprint = WasmConstraintBodyFingerprint {
                a_terms: WasmNormalizedTerms(non_selector_a_terms),
                b_terms: WasmNormalizedTerms(non_selector_b_terms),
                c_terms: normalize_terms(&row.c_terms),
            };
            groups
                .entry(fingerprint)
                .or_default()
                .push((row_idx, selector_a_terms, selector_b_terms));
        }

        let mut out = Vec::new();
        for (fingerprint, rows) in groups {
            if rows.len() < 2 {
                continue;
            }
            let mut distinct_selector_sets = rows
                .iter()
                .map(|(_, selector_a, selector_b)| (selector_a, selector_b))
                .collect::<Vec<_>>();
            distinct_selector_sets.sort();
            distinct_selector_sets.dedup();
            if distinct_selector_sets.len() < 2 {
                continue;
            }
            out.push(WasmDuplicateConstraintGroup {
                fingerprint,
                rows: rows.iter().map(|(row_idx, _, _)| *row_idx).collect(),
                selector_a_terms_by_row: rows
                    .iter()
                    .map(|(_, selector_a, _)| selector_a.clone())
                    .collect(),
                selector_b_terms_by_row: rows
                    .into_iter()
                    .map(|(_, _, selector_b)| selector_b)
                    .collect(),
            });
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct WasmTaggedR1csBuilder {
    inner: R1csBuilder,
    current_tag: WasmConstraintTag,
    row_tags: Vec<WasmConstraintTag>,
    rows: Vec<WasmR1csRow>,
}

impl WasmTaggedR1csBuilder {
    pub fn new(width: usize, const_one_col: usize) -> Result<Self, String> {
        Ok(Self {
            inner: R1csBuilder::new(width, const_one_col)?,
            current_tag: WasmConstraintTag {
                label: "unlabeled",
                scope: WasmConstraintScope::Always,
            },
            row_tags: Vec::new(),
            rows: Vec::new(),
        })
    }

    pub fn with_tag<R>(&mut self, tag: WasmConstraintTag, f: impl FnOnce(&mut Self) -> R) -> R {
        let prev = self.current_tag.clone();
        self.current_tag = tag;
        let out = f(self);
        self.current_tag = prev;
        out
    }

    pub fn push_row(
        &mut self,
        a_terms: impl IntoIterator<Item = (usize, F)>,
        b_terms: impl IntoIterator<Item = (usize, F)>,
        c_terms: impl IntoIterator<Item = (usize, F)>,
    ) -> &mut Self {
        let a_terms: Vec<_> = a_terms.into_iter().collect();
        let b_terms: Vec<_> = b_terms.into_iter().collect();
        let c_terms: Vec<_> = c_terms.into_iter().collect();
        self.row_tags.push(self.current_tag.clone());
        self.rows.push(WasmR1csRow {
            a_terms: a_terms.clone(),
            b_terms: b_terms.clone(),
            c_terms: c_terms.clone(),
        });
        self.inner.push_row(a_terms, b_terms, c_terms);
        self
    }

    pub fn push_linear_zero(&mut self, terms: impl IntoIterator<Item = (usize, F)>) -> &mut Self {
        let terms: Vec<_> = terms.into_iter().collect();
        self.row_tags.push(self.current_tag.clone());
        self.rows.push(WasmR1csRow {
            a_terms: terms.clone(),
            b_terms: vec![(self.inner.const_one_col(), F::ONE)],
            c_terms: Vec::new(),
        });
        self.inner.push_linear_zero(terms);
        self
    }

    pub fn push_boolean(&mut self, col: usize) -> &mut Self {
        self.row_tags.push(self.current_tag.clone());
        self.rows.push(WasmR1csRow {
            a_terms: vec![(col, F::ONE)],
            b_terms: vec![(col, F::ONE), (self.inner.const_one_col(), -F::ONE)],
            c_terms: Vec::new(),
        });
        self.inner.push_boolean(col);
        self
    }

    pub fn build(self) -> Result<(CcsStructure<F>, WasmConstraintCatalog), String> {
        let structure = self.inner.build()?;
        Ok((
            structure,
            WasmConstraintCatalog {
                row_tags: self.row_tags,
                rows: self.rows,
            },
        ))
    }
}

fn is_selector_gate_row(row: &WasmR1csRow, selector_cols: &std::collections::BTreeSet<usize>) -> bool {
    row.c_terms.is_empty()
        && row.a_terms.len() == 1
        && row.a_terms[0].1 == F::ONE
        && selector_cols.contains(&row.a_terms[0].0)
        && !row
            .b_terms
            .iter()
            .any(|(col, _)| selector_cols.contains(col))
}

fn normalize_terms(terms: &[(usize, F)]) -> WasmNormalizedTerms {
    WasmNormalizedTerms(
        terms
            .iter()
            .map(|&(col, coeff)| (col, coeff.as_canonical_u64()))
            .collect(),
    )
}
