//! Exact lowering of traced product-sum batches.
//!
//! Owns: exact trace validation, removed-column reconstruction, retained
//! boundary rank checking, and bounded-arity product-sum emission.
//!
//! Does not own: creation of source traces or the semantic truth of Π_RLC.
//!
//! Emits constraints: yes, one CCS product-sum row per at-most-18-term group.
//!
//! Authority boundary: the source R1CS remains the local implementation
//! arithmetic reference. A batch is accepted only after its complete
//! row/column interval and identities match. Matching alone does not prove the
//! source batch is a sufficient paper-level verifier obligation.
//!
//! | Stage path | Function | Equation | Multiplicity | Emitted rows/formula | Lowered gate | Lean theorem |
//! |---|---|---|---:|---|---|---|
//! | caller-owned | `validate_batch` | exact mixed topological SSA program | one per trace batch | none | none | mixed SSA refinement |
//! | caller-owned | `validate_identities` | retained results equal substituted products | one per retained result | none | none | retained-rank bridge open |
//! | caller-owned | `validate_emitted_dependencies` | emitted LCs use available columns | one global pass | none | none | concrete trace bridge open |
//! | caller-owned | `emit` | `result = sum_i a_i * left_i * right_i` | `ceil(terms/18)` | one row per group | product-sum | `carryChain_zero_iff_direct` |
//! | nested K-mul | `emit_k_mul` | two exact extension-field limbs | two per K product | two rows | product-sum | exact scalar substitution |
//!
//! Digests or witness-only equality never authorize replacement.

use std::collections::{BTreeMap, BTreeSet};

use neo_math::{Fq, F};
use p3_field::extension::BinomiallyExtendable;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::engine::r1cs_circuit::builder::{ProductSumBatchTrace, ProductSumIdentityTrace};
use crate::engine::r1cs_circuit::{KMulTraceEntry, Lc, R1csEncodingTrace, R1csSnapshot, Var};

use super::slots::{push_field_slot, slot_terms, ValueSlot};
use super::{
    claim_gadget_column, claim_rows, one_selector, scaled_terms, set_product_definition, source_terms,
    translate_event_lc, validate_expected_rows, GadgetNativeError, LinearDefinition, ProductDefinition,
    TraceGateBuilder, MAX_PRODUCT_TERMS,
};

const GADGET: &str = "product-sum batch";
const K_MUL_GADGET: &str = "K multiplication";

pub(super) fn validate_k_mul(source: &R1csSnapshot, event: &KMulTraceEntry) -> Result<(), GadgetNativeError> {
    if event.source_rows.len() != 5 {
        return Err(GadgetNativeError::TraceArity { gadget: K_MUL_GADGET });
    }
    let [p, q, r] = event.intermediates;
    let sum_a = event.a[0].clone().add_scaled(&event.a[1], F::ONE);
    let sum_b = event.b[0].clone().add_scaled(&event.b[1], F::ONE);
    let w = <Fq as BinomiallyExtendable<2>>::W;
    let out0_diff = Lc::from_var(event.output[0])
        .add_scaled(&Lc::from_var(p), -F::ONE)
        .add_scaled(&Lc::from_var(q), -w);
    let out1_diff = Lc::from_var(event.output[1])
        .add_scaled(&Lc::from_var(r), -F::ONE)
        .add_scaled(&Lc::from_var(p), F::ONE)
        .add_scaled(&Lc::from_var(q), F::ONE);
    let rows = [
        (event.a[0].clone(), event.b[0].clone(), Lc::from_var(p)),
        (event.a[1].clone(), event.b[1].clone(), Lc::from_var(q)),
        (sum_a, sum_b, Lc::from_var(r)),
        (out0_diff, Lc::from_var(Var::ONE), Lc::zero()),
        (out1_diff, Lc::from_var(Var::ONE), Lc::zero()),
    ];
    validate_expected_rows(source, K_MUL_GADGET, event.source_rows.start, &rows)
}

pub(super) fn define_unbatched_k_muls(
    definitions: &mut [Option<ProductDefinition>],
    trace: &R1csEncodingTrace,
    product_sums: &ValidatedProductSums,
) -> Result<(), GadgetNativeError> {
    for (index, event) in trace.k_muls().iter().enumerate() {
        if product_sums.is_nested_k_mul(index) {
            continue;
        }
        let [p, q, r] = event.intermediates;
        set_product_definition(definitions, p, event.a[0].clone(), event.b[0].clone())?;
        set_product_definition(definitions, q, event.a[1].clone(), event.b[1].clone())?;
        set_product_definition(
            definitions,
            r,
            event.a[0].clone().add_scaled(&event.a[1], F::ONE),
            event.b[0].clone().add_scaled(&event.b[1], F::ONE),
        )?;
    }
    Ok(())
}

pub(super) fn emit_k_mul(
    event: &KMulTraceEntry,
    decoded: &[Option<Vec<(usize, F)>>],
    gates: &mut TraceGateBuilder,
) -> Result<(), GadgetNativeError> {
    let row = event.source_rows.start;
    let a0 = translate_event_lc(&event.a[0], decoded, row)?;
    let a1 = translate_event_lc(&event.a[1], decoded, row)?;
    let b0 = translate_event_lc(&event.b[0], decoded, row)?;
    let b1 = translate_event_lc(&event.b[1], decoded, row)?;
    let out0 = source_terms(event.output[0].col(), decoded, row)?;
    let out1 = source_terms(event.output[1].col(), decoded, row)?;
    let w = <Fq as BinomiallyExtendable<2>>::W;
    gates.product_sum(
        one_selector(),
        vec![(a0.clone(), b0.clone()), (scaled_terms(a1.clone(), w), b1.clone())],
        out0,
    );
    gates.product_sum(one_selector(), vec![(a0, b1), (a1, b0)], out1);
    Ok(())
}

#[derive(Clone, Debug)]
enum ExactDefinition {
    Product {
        column: usize,
        left: Lc,
        right: Lc,
    },
    Linear {
        column: usize,
        terms: Vec<(usize, F)>,
    },
}

impl ExactDefinition {
    fn column(&self) -> usize {
        match self {
            Self::Product { column, .. } | Self::Linear { column, .. } => *column,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ProductSumIdentityCost {
    pub(super) stage_row: usize,
    pub(super) starts_batch: bool,
    pub(super) encoded_rows: usize,
    pub(super) synthetic_fields: usize,
}

#[derive(Clone, Debug)]
pub(super) struct ProductSumBatchPlan {
    trace: ProductSumBatchTrace,
    identity_stage_rows: Vec<usize>,
    terminal_rows: Vec<usize>,
}

impl ProductSumBatchPlan {
    pub(super) fn recorded(trace: &ProductSumBatchTrace) -> Self {
        Self {
            trace: trace.clone(),
            identity_stage_rows: vec![trace.row_start; trace.identities.len()],
            terminal_rows: Vec::new(),
        }
    }

    pub(super) fn traced_terminal(
        trace: ProductSumBatchTrace,
        identity_stage_rows: Vec<usize>,
        terminal_rows: Vec<usize>,
    ) -> Self {
        Self {
            trace,
            identity_stage_rows,
            terminal_rows,
        }
    }

    fn row_range(&self) -> std::ops::Range<usize> {
        self.trace.row_start..self.trace.row_end
    }

    pub(super) fn trace(&self) -> &ProductSumBatchTrace {
        &self.trace
    }
}

#[derive(Clone, Debug)]
struct ValidatedBatch {
    index: usize,
    trace: ProductSumBatchTrace,
    definitions: Vec<ExactDefinition>,
    removed: Vec<bool>,
    identity_costs: Vec<ProductSumIdentityCost>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ValidatedProductSums {
    batches: Vec<ValidatedBatch>,
    nested_k_muls: Vec<bool>,
}

impl ValidatedProductSums {
    pub(super) fn validate_and_claim(
        source: &R1csSnapshot,
        trace: &R1csEncodingTrace,
        replacement_batches: Vec<ProductSumBatchPlan>,
        covered_rows: &mut [bool],
        gadget_columns: &mut [bool],
    ) -> Result<Self, GadgetNativeError> {
        let mut plans = Vec::with_capacity(trace.product_sum_batches().len() + replacement_batches.len());
        for (index, batch) in trace.product_sum_batches().iter().enumerate() {
            let overlapping = replacement_batches
                .iter()
                .filter(|replacement| ranges_overlap(batch.row_start..batch.row_end, replacement.row_range()))
                .collect::<Vec<_>>();
            if !overlapping.is_empty() {
                if overlapping.len() == 1
                    && overlapping[0].trace.row_start <= batch.row_start
                    && batch.row_end <= overlapping[0].trace.row_end
                {
                    continue;
                }
                return Err(GadgetNativeError::ProductSumGeometry {
                    batch: index,
                    detail: "partial replacement overlap",
                });
            }
            plans.push(ProductSumBatchPlan::recorded(batch));
        }
        plans.extend(replacement_batches);

        let mut batches = Vec::with_capacity(plans.len());
        for (index, plan) in plans.iter().enumerate() {
            let validated = validate_batch(source, plan, index)?;
            let batch = &plan.trace;
            claim_rows(source, GADGET, &(batch.row_start..batch.row_end), covered_rows)?;
            for (&column, &removed) in batch.allocated_columns.iter().zip(&validated.removed) {
                if removed {
                    claim_gadget_column(column, gadget_columns)?;
                }
            }
            batches.push(validated);
        }

        let mut nested_k_muls = vec![false; trace.k_muls().len()];
        for (index, event) in trace.k_muls().iter().enumerate() {
            for batch in &batches {
                let start = event.source_rows.start.max(batch.trace.row_start);
                let end = event.source_rows.end.min(batch.trace.row_end);
                if start >= end {
                    continue;
                }
                if event.source_rows.start >= batch.trace.row_start && event.source_rows.end <= batch.trace.row_end {
                    nested_k_muls[index] = true;
                } else {
                    return Err(GadgetNativeError::OverlappingTraceRow { row: start });
                }
            }
        }
        Ok(Self { batches, nested_k_muls })
    }

    pub(super) fn is_nested_k_mul(&self, index: usize) -> bool {
        self.nested_k_muls.get(index).copied().unwrap_or(false)
    }

    pub(super) fn encoded_rows(&self) -> usize {
        self.batches
            .iter()
            .flat_map(|batch| &batch.identity_costs)
            .map(|cost| cost.encoded_rows)
            .sum()
    }

    pub(super) fn synthetic_fields(&self) -> usize {
        self.batches
            .iter()
            .flat_map(|batch| &batch.identity_costs)
            .map(|cost| cost.synthetic_fields)
            .sum()
    }

    pub(super) fn costs(&self) -> impl Iterator<Item = ProductSumIdentityCost> + '_ {
        self.batches
            .iter()
            .flat_map(|batch| batch.identity_costs.iter().copied())
    }

    pub(super) fn validate_emitted_dependencies(&self, projected: &[bool]) -> Result<(), GadgetNativeError> {
        for batch in &self.batches {
            for identity in &batch.trace.identities {
                for &(column, _) in identity
                    .factors
                    .iter()
                    .flat_map(|factor| factor.left.terms.iter().chain(&factor.right.terms))
                    .chain(identity.result.terms.iter())
                {
                    if projected[column] {
                        return Err(GadgetNativeError::ProductSumUnavailableDependency {
                            batch: batch.index,
                            column,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn install_definitions(
        &self,
        products: &mut [Option<ProductDefinition>],
        linear: &mut [Option<LinearDefinition>],
    ) -> Result<(), GadgetNativeError> {
        for batch in &self.batches {
            for definition in &batch.definitions {
                let column = definition.column();
                if !batch.removed[column - batch.column_start()] {
                    continue;
                }
                if products[column].is_some() || linear[column].is_some() {
                    return Err(GadgetNativeError::DuplicateGadgetDefinition { column });
                }
                match definition {
                    ExactDefinition::Product { left, right, .. } => {
                        products[column] = Some(ProductDefinition {
                            left: left.clone(),
                            right: right.clone(),
                        });
                    }
                    ExactDefinition::Linear { terms, .. } => {
                        linear[column] = Some(LinearDefinition {
                            terms: terms.clone(),
                            source_row: None,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

impl ValidatedBatch {
    fn column_start(&self) -> usize {
        self.definitions
            .first()
            .expect("validated batch has definitions")
            .column()
    }
}

fn ranges_overlap(left: std::ops::Range<usize>, right: std::ops::Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

#[derive(Clone, Debug, Default)]
pub(super) struct ProductSumSlots {
    carries: Vec<Vec<Vec<ValueSlot>>>,
}

impl ProductSumSlots {
    pub(super) fn first_field_range(&self) -> Option<std::ops::Range<usize>> {
        let slot = self
            .carries
            .iter()
            .flat_map(|batch| batch.iter())
            .flat_map(|identity| identity.iter())
            .next()?;
        Some(slot.start..slot.start + slot.width)
    }

    pub(super) fn field_range(&self, batch: usize, identity: usize, carry: usize) -> Option<std::ops::Range<usize>> {
        let slot = *self.carries.get(batch)?.get(identity)?.get(carry)?;
        Some(slot.start..slot.start + slot.width)
    }

    pub(super) fn staged_fields<'a>(
        &'a self,
        product_sums: &'a ValidatedProductSums,
    ) -> impl Iterator<Item = (usize, ValueSlot)> + 'a {
        product_sums
            .batches
            .iter()
            .zip(&self.carries)
            .flat_map(|(batch, slots)| {
                batch
                    .identity_costs
                    .iter()
                    .zip(slots)
                    .flat_map(|(cost, carries)| {
                        carries
                            .iter()
                            .copied()
                            .map(move |slot| (cost.stage_row, slot))
                    })
            })
    }
}

pub(super) fn allocate_carries(
    source: &R1csSnapshot,
    product_sums: &ValidatedProductSums,
    assignment: &mut Vec<F>,
    canonical_slots: &mut Vec<ValueSlot>,
) -> ProductSumSlots {
    let mut batches = Vec::with_capacity(product_sums.batches.len());
    for batch in &product_sums.batches {
        let mut identities = Vec::with_capacity(batch.trace.identities.len());
        for identity in &batch.trace.identities {
            let mut running = F::ZERO;
            let groups = identity
                .factors
                .chunks(MAX_PRODUCT_TERMS)
                .collect::<Vec<_>>();
            let mut carries = Vec::with_capacity(groups.len().saturating_sub(1));
            for group in groups.iter().take(groups.len().saturating_sub(1)) {
                for factor in *group {
                    running += factor.coefficient
                        * eval_lc(&factor.left, source.witness())
                        * eval_lc(&factor.right, source.witness());
                }
                let slot = push_field_slot(assignment, running);
                canonical_slots.push(slot);
                carries.push(slot);
            }
            identities.push(carries);
        }
        batches.push(identities);
    }
    ProductSumSlots { carries: batches }
}

pub(super) fn emit(
    product_sums: &ValidatedProductSums,
    slots: &ProductSumSlots,
    decoded: &[Option<Vec<(usize, F)>>],
    gates: &mut TraceGateBuilder,
) -> Result<(), GadgetNativeError> {
    for (batch_index, batch) in product_sums.batches.iter().enumerate() {
        for (identity_index, identity) in batch.trace.identities.iter().enumerate() {
            let groups = identity
                .factors
                .chunks(MAX_PRODUCT_TERMS)
                .collect::<Vec<_>>();
            let carries = &slots.carries[batch_index][identity_index];
            let mut previous = None;
            for (group_index, group) in groups.iter().enumerate() {
                let products = group
                    .iter()
                    .map(|factor| {
                        let mut left = translate_event_lc(&factor.left, decoded, batch.trace.row_start)?;
                        for (_, coefficient) in &mut left {
                            *coefficient *= factor.coefficient;
                        }
                        Ok((left, translate_event_lc(&factor.right, decoded, batch.trace.row_start)?))
                    })
                    .collect::<Result<Vec<_>, GadgetNativeError>>()?;
                let mut out = if group_index + 1 == groups.len() {
                    translate_event_lc(&identity.result, decoded, batch.trace.row_start)?
                } else {
                    slot_terms(carries[group_index])
                };
                if let Some(previous) = previous {
                    out.extend(
                        slot_terms(previous)
                            .into_iter()
                            .map(|(column, coefficient)| (column, -coefficient)),
                    );
                }
                gates.product_sum(one_selector(), products, out);
                if group_index + 1 != groups.len() {
                    previous = Some(carries[group_index]);
                }
            }
        }
    }
    Ok(())
}

fn validate_batch(
    source: &R1csSnapshot,
    plan: &ProductSumBatchPlan,
    batch_index: usize,
) -> Result<ValidatedBatch, GadgetNativeError> {
    let batch = &plan.trace;
    let definition_row_end = batch
        .row_start
        .saturating_add(batch.allocated_columns.len());
    if batch.row_start >= batch.row_end
        || batch.row_end > source.rows()
        || batch.allocated_columns.is_empty()
        || batch.identities.is_empty()
        || definition_row_end > batch.row_end
        || batch.row_end - definition_row_end != plan.terminal_rows.len()
        || plan
            .terminal_rows
            .iter()
            .copied()
            .ne(definition_row_end..batch.row_end)
        || plan.identity_stage_rows.len() != batch.identities.len()
        || plan
            .identity_stage_rows
            .iter()
            .any(|&row| row < batch.row_start || row >= batch.row_end)
    {
        return geometry(batch_index, "row/column geometry");
    }
    let first_column = batch.allocated_columns[0];
    if first_column == 0
        || batch
            .allocated_columns
            .iter()
            .copied()
            .ne(first_column..first_column + batch.allocated_columns.len())
        || batch
            .allocated_columns
            .last()
            .is_some_and(|&column| column >= source.cols())
    {
        return geometry(batch_index, "fresh-column interval");
    }

    let allocated = batch
        .allocated_columns
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let retained = batch
        .retained_columns
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if retained.is_empty()
        || retained.len() != batch.retained_columns.len()
        || !retained.is_subset(&allocated)
        || batch.identities.len() != retained.len() + plan.terminal_rows.len()
    {
        return geometry(batch_index, "retained boundary");
    }

    let mut definitions = Vec::with_capacity(batch.allocated_columns.len());
    let mut expressions = BTreeMap::<usize, Polynomial>::new();
    for (offset, &target) in batch.allocated_columns.iter().enumerate() {
        let row = batch.row_start + offset;
        let definition = exact_definition(source, row, target, batch_index)?;
        let expression = expression_for_definition(&definition, &expressions, source.cols(), batch_index)?;
        expressions.insert(target, expression);
        definitions.push(definition);
    }

    validate_identities(
        source,
        batch,
        batch_index,
        &allocated,
        &retained,
        &expressions,
        &plan.terminal_rows,
    )?;
    let removed = batch
        .allocated_columns
        .iter()
        .map(|column| !retained.contains(column))
        .collect::<Vec<_>>();
    let identity_costs = batch
        .identities
        .iter()
        .zip(&plan.identity_stage_rows)
        .enumerate()
        .map(|(identity, (trace, &stage_row))| {
            let encoded_rows = identity_rows(trace);
            ProductSumIdentityCost {
                stage_row,
                starts_batch: identity == 0,
                encoded_rows,
                synthetic_fields: encoded_rows - 1,
            }
        })
        .collect();
    Ok(ValidatedBatch {
        index: batch_index,
        trace: batch.clone(),
        definitions,
        removed,
        identity_costs,
    })
}

fn exact_definition(
    source: &R1csSnapshot,
    row: usize,
    target: usize,
    batch: usize,
) -> Result<ExactDefinition, GadgetNativeError> {
    if source.c_row(row) == [(target, F::ONE)] {
        let left = lc_from_terms(source.a_row(row));
        let right = lc_from_terms(source.b_row(row));
        if references_at_or_after(&left, target) || references_at_or_after(&right, target) {
            return geometry(batch, "non-topological product row");
        }
        return Ok(ExactDefinition::Product {
            column: target,
            left,
            right,
        });
    }

    let (positive, negative) = if source.b_row(row) == [(0, F::ONE)] {
        (source.a_row(row), source.c_row(row))
    } else if source.a_row(row) == [(0, F::ONE)] {
        (source.b_row(row), source.c_row(row))
    } else {
        return Err(GadgetNativeError::TraceRowMismatch { gadget: GADGET, row });
    };
    let mut difference = BTreeMap::<usize, F>::new();
    for &(column, coefficient) in positive {
        *difference.entry(column).or_insert(F::ZERO) += coefficient;
    }
    for &(column, coefficient) in negative {
        *difference.entry(column).or_insert(F::ZERO) -= coefficient;
    }
    difference.retain(|_, coefficient| *coefficient != F::ZERO);
    let Some(target_coefficient) = difference.remove(&target) else {
        return Err(GadgetNativeError::TraceRowMismatch { gadget: GADGET, row });
    };
    if difference.keys().any(|&column| column >= target) {
        return geometry(batch, "non-topological linear row");
    }
    let inverse = target_coefficient.inverse();
    let terms = difference
        .into_iter()
        .map(|(column, coefficient)| (column, -coefficient * inverse))
        .collect();
    Ok(ExactDefinition::Linear { column: target, terms })
}

fn validate_identities(
    source: &R1csSnapshot,
    batch: &ProductSumBatchTrace,
    batch_index: usize,
    allocated: &BTreeSet<usize>,
    retained: &BTreeSet<usize>,
    expressions: &BTreeMap<usize, Polynomial>,
    terminal_rows: &[usize],
) -> Result<(), GadgetNativeError> {
    let retained_columns = batch.retained_columns.as_slice();
    let mut rank_matrix = vec![vec![F::ZERO; retained_columns.len()]; retained_columns.len()];
    for (identity_index, identity) in batch.identities.iter().enumerate() {
        if identity.factors.is_empty()
            || identity
                .factors
                .iter()
                .any(|factor| factor.coefficient == F::ZERO)
        {
            return geometry(batch_index, "empty or zero product identity");
        }
        let is_terminal = identity_index >= retained_columns.len();
        validate_identity_columns(source, identity, allocated, retained, batch_index, is_terminal)?;
        if !is_terminal {
            for (column_index, &column) in retained_columns.iter().enumerate() {
                rank_matrix[identity_index][column_index] = lc_coefficient(&identity.result, column);
            }
        }

        let mut claimed = Polynomial::default();
        for factor in &identity.factors {
            let left = polynomial_from_lc(&factor.left, expressions, source.cols(), batch_index)?;
            let right = polynomial_from_lc(&factor.right, expressions, source.cols(), batch_index)?;
            claimed.add_scaled(&left.multiply(&right, batch_index)?, factor.coefficient);
        }
        let result = polynomial_from_lc(&identity.result, expressions, source.cols(), batch_index)?;
        claimed.add_scaled(&result, -F::ONE);
        let expected = if is_terminal {
            polynomial_from_source_row(
                source,
                terminal_rows[identity_index - retained_columns.len()],
                expressions,
                batch_index,
            )?
        } else {
            Polynomial::default()
        };
        if claimed != expected {
            return Err(GadgetNativeError::ProductSumIdentityMismatch {
                batch: batch_index,
                identity: identity_index,
            });
        }
    }
    if matrix_rank(rank_matrix) != retained_columns.len() {
        return Err(GadgetNativeError::ProductSumRetainedRank { batch: batch_index });
    }
    Ok(())
}

fn validate_identity_columns(
    source: &R1csSnapshot,
    identity: &ProductSumIdentityTrace,
    allocated: &BTreeSet<usize>,
    retained: &BTreeSet<usize>,
    batch: usize,
    is_terminal: bool,
) -> Result<(), GadgetNativeError> {
    for factor in &identity.factors {
        for &(column, _) in factor.left.terms.iter().chain(&factor.right.terms) {
            let invalid_allocated = allocated.contains(&column) && (!is_terminal || !retained.contains(&column));
            if column >= source.cols() || invalid_allocated {
                return geometry(batch, "product factor authority");
            }
        }
    }
    for &(column, _) in &identity.result.terms {
        if column >= source.cols() || (allocated.contains(&column) && !retained.contains(&column)) {
            return geometry(batch, "identity result boundary");
        }
    }
    Ok(())
}

fn polynomial_from_source_row(
    source: &R1csSnapshot,
    row: usize,
    expressions: &BTreeMap<usize, Polynomial>,
    batch: usize,
) -> Result<Polynomial, GadgetNativeError> {
    if row >= source.rows() {
        return geometry(batch, "terminal row range");
    }
    let left = polynomial_from_terms(source.a_row(row), expressions, source.cols(), batch)?;
    let right = polynomial_from_terms(source.b_row(row), expressions, source.cols(), batch)?;
    let mut constraint = left.multiply(&right, batch)?;
    let output = polynomial_from_terms(source.c_row(row), expressions, source.cols(), batch)?;
    constraint.add_scaled(&output, -F::ONE);
    Ok(constraint)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Polynomial(BTreeMap<Vec<usize>, F>);

impl Polynomial {
    fn constant(value: F) -> Self {
        let mut out = Self::default();
        if value != F::ZERO {
            out.0.insert(Vec::new(), value);
        }
        out
    }

    fn variable(column: usize) -> Self {
        if column == 0 {
            return Self::constant(F::ONE);
        }
        let mut out = Self::default();
        out.0.insert(vec![column], F::ONE);
        out
    }

    fn add_scaled(&mut self, other: &Self, scale: F) {
        for (monomial, coefficient) in &other.0 {
            let entry = self.0.entry(monomial.clone()).or_insert(F::ZERO);
            *entry += *coefficient * scale;
            if *entry == F::ZERO {
                self.0.remove(monomial);
            }
        }
    }

    fn multiply(&self, other: &Self, batch: usize) -> Result<Self, GadgetNativeError> {
        let mut out = Self::default();
        for (left, left_coefficient) in &self.0 {
            for (right, right_coefficient) in &other.0 {
                let mut monomial = left.clone();
                monomial.extend(right);
                monomial.sort_unstable();
                if monomial.len() > 3 {
                    return geometry(batch, "symbolic degree above three");
                }
                *out.0.entry(monomial).or_insert(F::ZERO) += *left_coefficient * *right_coefficient;
            }
        }
        out.0.retain(|_, coefficient| *coefficient != F::ZERO);
        Ok(out)
    }
}

fn expression_for_definition(
    definition: &ExactDefinition,
    expressions: &BTreeMap<usize, Polynomial>,
    cols: usize,
    batch: usize,
) -> Result<Polynomial, GadgetNativeError> {
    match definition {
        ExactDefinition::Product { left, right, .. } => Ok(polynomial_from_lc(left, expressions, cols, batch)?
            .multiply(&polynomial_from_lc(right, expressions, cols, batch)?, batch)?),
        ExactDefinition::Linear { terms, .. } => {
            let mut out = Polynomial::default();
            for &(column, coefficient) in terms {
                let term = expressions
                    .get(&column)
                    .cloned()
                    .unwrap_or_else(|| Polynomial::variable(column));
                out.add_scaled(&term, coefficient);
            }
            Ok(out)
        }
    }
}

fn polynomial_from_lc(
    lc: &Lc,
    expressions: &BTreeMap<usize, Polynomial>,
    cols: usize,
    batch: usize,
) -> Result<Polynomial, GadgetNativeError> {
    let mut out = Polynomial::constant(lc.constant);
    for &(column, coefficient) in &lc.terms {
        if column >= cols {
            return geometry(batch, "identity column range");
        }
        let term = expressions
            .get(&column)
            .cloned()
            .unwrap_or_else(|| Polynomial::variable(column));
        out.add_scaled(&term, coefficient);
    }
    Ok(out)
}

fn polynomial_from_terms(
    terms: &[(usize, F)],
    expressions: &BTreeMap<usize, Polynomial>,
    cols: usize,
    batch: usize,
) -> Result<Polynomial, GadgetNativeError> {
    let mut lc = Lc::zero();
    for &(column, coefficient) in terms {
        if column == 0 {
            lc.constant += coefficient;
        } else {
            lc.terms.push((column, coefficient));
        }
    }
    polynomial_from_lc(&lc, expressions, cols, batch)
}

fn matrix_rank(mut matrix: Vec<Vec<F>>) -> usize {
    let rows = matrix.len();
    let columns = matrix.first().map_or(0, Vec::len);
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..rows).find(|&row| matrix[row][column] != F::ZERO) else {
            continue;
        };
        matrix.swap(rank, pivot);
        let inverse = matrix[rank][column].inverse();
        for entry in &mut matrix[rank][column..] {
            *entry *= inverse;
        }
        let pivot_row = matrix[rank].clone();
        for (row, values) in matrix.iter_mut().enumerate() {
            if row == rank || values[column] == F::ZERO {
                continue;
            }
            let scale = values[column];
            for index in column..columns {
                values[index] -= scale * pivot_row[index];
            }
        }
        rank += 1;
    }
    rank
}

fn identity_rows(identity: &ProductSumIdentityTrace) -> usize {
    identity.factors.len().div_ceil(MAX_PRODUCT_TERMS)
}

fn lc_from_terms(terms: &[(usize, F)]) -> Lc {
    let mut lc = Lc::zero();
    for &(column, coefficient) in terms {
        if column == 0 {
            lc.constant += coefficient;
        } else {
            lc.terms.push((column, coefficient));
        }
    }
    lc
}

fn references_at_or_after(lc: &Lc, column: usize) -> bool {
    lc.terms.iter().any(|&(input, _)| input >= column)
}

fn lc_coefficient(lc: &Lc, column: usize) -> F {
    lc.terms
        .iter()
        .filter(|&&(candidate, _)| candidate == column)
        .fold(F::ZERO, |sum, &(_, coefficient)| sum + coefficient)
}

fn eval_lc(lc: &Lc, witness: &[F]) -> F {
    lc.terms
        .iter()
        .fold(lc.constant, |value, &(column, coefficient)| {
            value + coefficient * witness[column]
        })
}

fn geometry<T>(batch: usize, detail: &'static str) -> Result<T, GadgetNativeError> {
    Err(GadgetNativeError::ProductSumGeometry { batch, detail })
}
