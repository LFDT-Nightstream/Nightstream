//! Indexed compiler for every terminal raw-old-block projection row.
//!
//! The production emitter consumes this program directly.  It defines one
//! canonical terminal-local column space, a bijective physical row owner, and
//! exact normalized sparse A/B/C rows without constructing a production-sized
//! row list.  `RawOldBlockProjectionColumnMap` then maps those canonical input
//! and internal columns to the actual builder wires.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::field_ext::{k_mul_constraint_rows, klc_add_scaled, KLc, KVar};
use super::row_formula::{canonical_sparse_row, equality_constraint_row, multiplication_constraint_row};
use super::{CanonicalSparseRow, Lc, RawOldBlockProjectionPlan, Var, RAW_OLD_BLOCK_K_LIMBS, RAW_OLD_BLOCK_K_MUL_ROWS};

const CONSTANT_COLUMN: usize = 0;
const FIRST_INPUT_COLUMN: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawOldBlockProjectionCanonicalLayout {
    plan: RawOldBlockProjectionPlan,
}

impl RawOldBlockProjectionCanonicalLayout {
    pub fn new(plan: RawOldBlockProjectionPlan) -> Result<Self, &'static str> {
        if plan.logical_columns() != plan.active_lanes() * plan.packed_columns() {
            return Err("raw old-block row-at requires exact active-lane packing");
        }
        Ok(Self { plan })
    }

    pub fn plan(self) -> RawOldBlockProjectionPlan {
        self.plan
    }

    pub fn old_block_first(self) -> usize {
        FIRST_INPUT_COLUMN
    }

    pub fn old_block(self, round: usize) -> Option<[usize; RAW_OLD_BLOCK_K_LIMBS]> {
        (round < self.plan.block_variables()).then(|| {
            let first = self.old_block_first() + RAW_OLD_BLOCK_K_LIMBS * round;
            [first, first + 1]
        })
    }

    pub fn parent_first(self) -> usize {
        self.old_block_first() + RAW_OLD_BLOCK_K_LIMBS * self.plan.block_variables()
    }

    pub fn parent(self, lane: usize) -> Option<[usize; RAW_OLD_BLOCK_K_LIMBS]> {
        (lane < self.plan.active_lanes()).then(|| {
            let first = self.parent_first() + RAW_OLD_BLOCK_K_LIMBS * lane;
            [first, first + 1]
        })
    }

    pub fn child_witness_first(self, child: usize) -> Option<usize> {
        (child < self.plan.child_count())
            .then_some(self.witness_family_first() + child * self.plan.active_lanes() * self.plan.packed_columns())
    }

    pub fn witness_family_first(self) -> usize {
        self.parent_first() + RAW_OLD_BLOCK_K_LIMBS * self.plan.active_lanes()
    }

    pub fn witness_column(self, child: usize, lane: usize, block: usize) -> Option<usize> {
        Some(self.child_witness_first(child)? + self.plan.witness_flat_index(lane, block)?)
    }

    pub fn tensor_first(self) -> usize {
        self.witness_family_first() + self.plan.child_count() * self.plan.active_lanes() * self.plan.packed_columns()
    }

    pub fn product_first(self) -> usize {
        self.tensor_first() + self.plan.tensor_rows()
    }

    pub fn product_column(self, lane: usize, block: usize, limb: usize) -> Option<usize> {
        self.plan
            .projection_product_column(self.product_first(), lane, block, limb)
    }

    pub fn final_scale_first(self) -> usize {
        self.product_first() + self.plan.projection_product_rows()
    }

    pub fn final_scale_output(self, lane: usize) -> Option<[usize; RAW_OLD_BLOCK_K_LIMBS]> {
        self.plan
            .final_scale_output_columns(self.final_scale_first(), lane)
    }

    pub fn column_count(self) -> usize {
        self.final_scale_first() + self.plan.final_scale_rows()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawOldBlockProjectionRowOwner {
    Tensor {
        round: usize,
        parent: usize,
        k_row: usize,
    },
    Product {
        lane: usize,
        block: usize,
        limb: usize,
    },
    FinalScale {
        lane: usize,
        k_row: usize,
    },
    Terminal {
        lane: usize,
        limb: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawOldBlockProjectionProgram {
    layout: RawOldBlockProjectionCanonicalLayout,
    radix: u32,
}

impl RawOldBlockProjectionProgram {
    pub fn new(plan: RawOldBlockProjectionPlan, radix: u32) -> Result<Self, &'static str> {
        if radix == 0 {
            return Err("raw old-block row-at radix must be nonzero");
        }
        Ok(Self {
            layout: RawOldBlockProjectionCanonicalLayout::new(plan)?,
            radix,
        })
    }

    pub fn layout(self) -> RawOldBlockProjectionCanonicalLayout {
        self.layout
    }

    pub fn plan(self) -> RawOldBlockProjectionPlan {
        self.layout.plan()
    }

    pub fn radix(self) -> u32 {
        self.radix
    }

    pub fn row_count(self) -> usize {
        self.plan().total_rows()
    }

    pub fn owner(self, relative_row: usize) -> Option<RawOldBlockProjectionRowOwner> {
        let plan = self.plan();
        if relative_row < plan.tensor_rows() {
            let ordinal = relative_row / RAW_OLD_BLOCK_K_MUL_ROWS;
            let k_row = relative_row % RAW_OLD_BLOCK_K_MUL_ROWS;
            let mut first = 0;
            for round in 0..plan.tensor_variables() {
                let count = plan.tensor_round_mul_count(round).expect("round in range");
                if ordinal < first + count {
                    return Some(RawOldBlockProjectionRowOwner::Tensor {
                        round,
                        parent: ordinal - first,
                        k_row,
                    });
                }
                first += count;
            }
            return None;
        }
        let after_tensor = relative_row - plan.tensor_rows();
        if after_tensor < plan.projection_product_rows() {
            let ordinal = after_tensor / RAW_OLD_BLOCK_K_LIMBS;
            return Some(RawOldBlockProjectionRowOwner::Product {
                lane: ordinal / plan.packed_columns(),
                block: ordinal % plan.packed_columns(),
                limb: after_tensor % RAW_OLD_BLOCK_K_LIMBS,
            });
        }
        let after_products = after_tensor - plan.projection_product_rows();
        if after_products < plan.final_scale_rows() {
            return Some(RawOldBlockProjectionRowOwner::FinalScale {
                lane: after_products / RAW_OLD_BLOCK_K_MUL_ROWS,
                k_row: after_products % RAW_OLD_BLOCK_K_MUL_ROWS,
            });
        }
        let after_final_scale = after_products - plan.final_scale_rows();
        (after_final_scale < plan.terminal_rows()).then_some(RawOldBlockProjectionRowOwner::Terminal {
            lane: after_final_scale / RAW_OLD_BLOCK_K_LIMBS,
            limb: after_final_scale % RAW_OLD_BLOCK_K_LIMBS,
        })
    }

    pub fn row_at(self, relative_row: usize) -> Option<CanonicalSparseRow> {
        match self.owner(relative_row)? {
            RawOldBlockProjectionRowOwner::Tensor { round, parent, k_row } => {
                let (left, right, p, q, r, output) = self.tensor_operation(round, parent)?;
                let rows = k_mul_constraint_rows(&left, &right, p, q, r, output);
                Some(canonical_sparse_row(&rows[k_row]))
            }
            RawOldBlockProjectionRowOwner::Product { lane, block, limb } => {
                let weight = self.chi_terms(block)?;
                let raw = self.raw_terms(lane, block)?;
                let output = Var::from_column_for_trace(self.layout.product_column(lane, block, limb)?);
                let weight_limb = if limb == 0 { &weight.c0 } else { &weight.c1 };
                Some(canonical_sparse_row(&multiplication_constraint_row(
                    &raw,
                    weight_limb,
                    output,
                )))
            }
            RawOldBlockProjectionRowOwner::FinalScale { lane, k_row } => {
                let (left, right, p, q, r, output) = self.final_scale_operation(lane)?;
                let rows = k_mul_constraint_rows(&left, &right, p, q, r, output);
                Some(canonical_sparse_row(&rows[k_row]))
            }
            RawOldBlockProjectionRowOwner::Terminal { lane, limb } => {
                let (parent, sum) = self.terminal_operands(lane, limb)?;
                Some(canonical_sparse_row(&equality_constraint_row(&parent, &sum)))
            }
        }
    }

    pub(crate) fn tensor_operation(self, round: usize, parent: usize) -> Option<(KLc, KLc, Var, Var, Var, KVar)> {
        let plan = self.plan();
        if parent >= plan.tensor_round_mul_count(round)? {
            return None;
        }
        let left = self.tensor_terms_at(round, parent)?;
        let high_count = plan.tensor_round_high_count(round)?;
        let point = self.point_terms(round)?;
        let right = if parent < high_count {
            point
        } else {
            KLc {
                c0: Lc::from_const(F::ONE).add_scaled(&point.c0, -F::ONE),
                c1: Lc::zero().add_scaled(&point.c1, -F::ONE),
            }
        };
        let first = plan.tensor_mul_first_column(self.layout.tensor_first(), round, parent)?;
        Some((
            left,
            right,
            Var::from_column_for_trace(first),
            Var::from_column_for_trace(first + 1),
            Var::from_column_for_trace(first + 2),
            KVar::new(
                Var::from_column_for_trace(first + 3),
                Var::from_column_for_trace(first + 4),
            ),
        ))
    }

    pub(crate) fn chi_terms(self, block: usize) -> Option<KLc> {
        self.tensor_terms_at(self.plan().tensor_variables(), block)
    }

    pub(crate) fn raw_terms(self, lane: usize, block: usize) -> Option<Lc> {
        let plan = self.plan();
        plan.witness_flat_index(lane, block)?;
        let mut raw = Lc::zero();
        let mut coefficient = F::ONE;
        let radix = F::from_u64(self.radix as u64);
        for child in 0..plan.child_count() {
            raw.terms
                .push((self.layout.witness_column(child, lane, block)?, coefficient));
            coefficient *= radix;
        }
        Some(raw)
    }

    pub(crate) fn final_scale_operation(self, lane: usize) -> Option<(KLc, KLc, Var, Var, Var, KVar)> {
        let factored_variable = self.plan().factored_variable()?;
        let left = self.projection_sum_terms(lane)?;
        let point = self.point_terms(factored_variable)?;
        let right = KLc {
            c0: Lc::from_const(F::ONE).add_scaled(&point.c0, -F::ONE),
            c1: Lc::zero().add_scaled(&point.c1, -F::ONE),
        };
        let first = self
            .plan()
            .final_scale_mul_first_column(self.layout.final_scale_first(), lane)?;
        Some((
            left,
            right,
            Var::from_column_for_trace(first),
            Var::from_column_for_trace(first + 1),
            Var::from_column_for_trace(first + 2),
            KVar::new(
                Var::from_column_for_trace(first + 3),
                Var::from_column_for_trace(first + 4),
            ),
        ))
    }

    pub(crate) fn terminal_operands(self, lane: usize, limb: usize) -> Option<(Lc, Lc)> {
        let parent = self.layout.parent(lane)?;
        if limb >= RAW_OLD_BLOCK_K_LIMBS {
            return None;
        }
        let projected = if let Some(output) = self.layout.final_scale_output(lane) {
            Lc::from_var(Var::from_column_for_trace(output[limb]))
        } else {
            let sum = self.projection_sum_terms(lane)?;
            if limb == 0 {
                sum.c0
            } else {
                sum.c1
            }
        };
        Some((Lc::from_var(Var::from_column_for_trace(parent[limb])), projected))
    }

    fn projection_sum_terms(self, lane: usize) -> Option<KLc> {
        if lane >= self.plan().active_lanes() {
            return None;
        }
        let mut sum = Lc::zero();
        let mut sum_c1 = Lc::zero();
        sum.terms.reserve(self.plan().packed_columns());
        sum_c1.terms.reserve(self.plan().packed_columns());
        for block in 0..self.plan().packed_columns() {
            sum.terms
                .push((self.layout.product_column(lane, block, 0)?, F::ONE));
            sum_c1
                .terms
                .push((self.layout.product_column(lane, block, 1)?, F::ONE));
        }
        Some(KLc { c0: sum, c1: sum_c1 })
    }

    fn point_terms(self, round: usize) -> Option<KLc> {
        let [c0, c1] = self.layout.old_block(round)?;
        Some(KLc::from_var(KVar::new(
            Var::from_column_for_trace(c0),
            Var::from_column_for_trace(c1),
        )))
    }

    fn tensor_output(self, round: usize, parent: usize) -> Option<KLc> {
        let [c0, c1] = self
            .plan()
            .tensor_mul_output_columns(self.layout.tensor_first(), round, parent)?;
        Some(KLc::from_var(KVar::new(
            Var::from_column_for_trace(c0),
            Var::from_column_for_trace(c1),
        )))
    }

    fn tensor_terms_at(self, round: usize, index: usize) -> Option<KLc> {
        if round == 0 {
            return (index == 0).then(|| KLc::from_base_const(F::ONE));
        }
        let prior_round = round - 1;
        let parent_count = self.plan().tensor_round_mul_count(prior_round)?;
        let high_count = self.plan().tensor_round_high_count(prior_round)?;
        if index < parent_count {
            if index < high_count {
                Some(klc_add_scaled(
                    &self.tensor_terms_at(prior_round, index)?,
                    &self.tensor_output(prior_round, index)?,
                    -F::ONE,
                ))
            } else {
                self.tensor_output(prior_round, index)
            }
        } else {
            let high_parent = index - parent_count;
            (high_parent < high_count)
                .then(|| self.tensor_output(prior_round, high_parent))
                .flatten()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawOldBlockProjectionColumnMap {
    layout: RawOldBlockProjectionCanonicalLayout,
    actual_old_block: Vec<[usize; RAW_OLD_BLOCK_K_LIMBS]>,
    actual_parent: Vec<[usize; RAW_OLD_BLOCK_K_LIMBS]>,
    actual_child_witness_first: Vec<usize>,
    actual_tensor_first: usize,
    actual_product_first: usize,
    actual_final_scale_first: usize,
}

impl RawOldBlockProjectionColumnMap {
    pub(crate) fn new(
        layout: RawOldBlockProjectionCanonicalLayout,
        actual_old_block: Vec<[usize; RAW_OLD_BLOCK_K_LIMBS]>,
        actual_parent: Vec<[usize; RAW_OLD_BLOCK_K_LIMBS]>,
        actual_child_witness_first: Vec<usize>,
        actual_tensor_first: usize,
        actual_product_first: usize,
        actual_final_scale_first: usize,
    ) -> Result<Self, &'static str> {
        if actual_old_block.len() != layout.plan().block_variables()
            || actual_parent.len() != layout.plan().active_lanes()
            || actual_child_witness_first.len() != layout.plan().child_count()
            || actual_product_first != actual_tensor_first + layout.plan().tensor_rows()
            || actual_final_scale_first != actual_product_first + layout.plan().projection_product_rows()
        {
            return Err("raw old-block canonical/actual column-map shape mismatch");
        }
        let mut scalar_columns = vec![CONSTANT_COLUMN];
        scalar_columns.extend(actual_old_block.iter().flatten().copied());
        scalar_columns.extend(actual_parent.iter().flatten().copied());
        scalar_columns.sort_unstable();
        if scalar_columns.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("raw old-block canonical/actual scalar columns alias");
        }

        let witness_entries = layout.plan().active_lanes() * layout.plan().packed_columns();
        let mut intervals = actual_child_witness_first
            .iter()
            .copied()
            .map(|first| {
                first
                    .checked_add(witness_entries)
                    .map(|stop| (first, stop))
                    .ok_or("raw old-block actual witness interval overflows")
            })
            .collect::<Result<Vec<_>, _>>()?;
        intervals.push((actual_tensor_first, actual_product_first));
        intervals.push((actual_product_first, actual_final_scale_first));
        intervals.push((
            actual_final_scale_first,
            actual_final_scale_first
                .checked_add(layout.plan().final_scale_rows())
                .ok_or("raw old-block actual final-scale interval overflows")?,
        ));
        intervals.sort_unstable_by_key(|interval| interval.0);
        if intervals.windows(2).any(|pair| pair[0].1 > pair[1].0)
            || scalar_columns.iter().any(|column| {
                intervals
                    .iter()
                    .any(|(first, stop)| (*first..*stop).contains(column))
            })
        {
            return Err("raw old-block canonical/actual column ranges alias");
        }
        Ok(Self {
            layout,
            actual_old_block,
            actual_parent,
            actual_child_witness_first,
            actual_tensor_first,
            actual_product_first,
            actual_final_scale_first,
        })
    }

    pub fn layout(&self) -> RawOldBlockProjectionCanonicalLayout {
        self.layout
    }

    pub fn actual_old_block(&self) -> &[[usize; RAW_OLD_BLOCK_K_LIMBS]] {
        &self.actual_old_block
    }

    pub fn actual_parent(&self) -> &[[usize; RAW_OLD_BLOCK_K_LIMBS]] {
        &self.actual_parent
    }

    pub fn actual_child_witness_first(&self) -> &[usize] {
        &self.actual_child_witness_first
    }

    pub fn actual_tensor_first(&self) -> usize {
        self.actual_tensor_first
    }

    pub fn actual_product_first(&self) -> usize {
        self.actual_product_first
    }

    pub fn actual_final_scale_first(&self) -> usize {
        self.actual_final_scale_first
    }

    pub(crate) fn map_lc(&self, canonical: &Lc) -> Option<Lc> {
        let mut actual = Lc::from_const(canonical.constant);
        actual.terms.reserve(canonical.terms.len());
        for &(column, coefficient) in &canonical.terms {
            actual
                .terms
                .push((self.canonical_to_actual(column)?, coefficient));
        }
        Some(actual)
    }

    pub(crate) fn map_klc(&self, canonical: &KLc) -> Option<KLc> {
        Some(KLc {
            c0: self.map_lc(&canonical.c0)?,
            c1: self.map_lc(&canonical.c1)?,
        })
    }

    pub(crate) fn map_lc_owned(&self, mut canonical: Lc) -> Option<Lc> {
        for (column, _) in &mut canonical.terms {
            *column = self.canonical_to_actual(*column)?;
        }
        Some(canonical)
    }

    pub(crate) fn map_klc_owned(&self, canonical: KLc) -> Option<KLc> {
        Some(KLc {
            c0: self.map_lc_owned(canonical.c0)?,
            c1: self.map_lc_owned(canonical.c1)?,
        })
    }

    pub fn canonical_to_actual(&self, column: usize) -> Option<usize> {
        if column == CONSTANT_COLUMN {
            return Some(CONSTANT_COLUMN);
        }
        if (self.layout.old_block_first()..self.layout.parent_first()).contains(&column) {
            let offset = column - self.layout.old_block_first();
            return Some(self.actual_old_block[offset / RAW_OLD_BLOCK_K_LIMBS][offset % RAW_OLD_BLOCK_K_LIMBS]);
        }
        if (self.layout.parent_first()..self.layout.witness_family_first()).contains(&column) {
            let offset = column - self.layout.parent_first();
            return Some(self.actual_parent[offset / RAW_OLD_BLOCK_K_LIMBS][offset % RAW_OLD_BLOCK_K_LIMBS]);
        }
        let entries = self.layout.plan().active_lanes() * self.layout.plan().packed_columns();
        if (self.layout.witness_family_first()..self.layout.tensor_first()).contains(&column) {
            let offset = column - self.layout.witness_family_first();
            return Some(self.actual_child_witness_first[offset / entries] + offset % entries);
        }
        if (self.layout.tensor_first()..self.layout.product_first()).contains(&column) {
            return Some(self.actual_tensor_first + column - self.layout.tensor_first());
        }
        if (self.layout.product_first()..self.layout.final_scale_first()).contains(&column) {
            return Some(self.actual_product_first + column - self.layout.product_first());
        }
        if (self.layout.final_scale_first()..self.layout.column_count()).contains(&column) {
            return Some(self.actual_final_scale_first + column - self.layout.final_scale_first());
        }
        None
    }

    pub fn actual_to_canonical(&self, column: usize) -> Option<usize> {
        if column == CONSTANT_COLUMN {
            return Some(CONSTANT_COLUMN);
        }
        if (self.actual_tensor_first..self.actual_product_first).contains(&column) {
            return Some(self.layout.tensor_first() + column - self.actual_tensor_first);
        }
        if (self.actual_product_first..self.actual_final_scale_first).contains(&column) {
            return Some(self.layout.product_first() + column - self.actual_product_first);
        }
        let actual_column_stop = self.actual_final_scale_first + self.layout.plan().final_scale_rows();
        if (self.actual_final_scale_first..actual_column_stop).contains(&column) {
            return Some(self.layout.final_scale_first() + column - self.actual_final_scale_first);
        }
        for (round, actual) in self.actual_old_block.iter().enumerate() {
            for limb in 0..RAW_OLD_BLOCK_K_LIMBS {
                if column == actual[limb] {
                    return Some(self.layout.old_block(round)?[limb]);
                }
            }
        }
        for (lane, actual) in self.actual_parent.iter().enumerate() {
            for limb in 0..RAW_OLD_BLOCK_K_LIMBS {
                if column == actual[limb] {
                    return Some(self.layout.parent(lane)?[limb]);
                }
            }
        }
        let entries = self.layout.plan().active_lanes() * self.layout.plan().packed_columns();
        for (child, &first) in self.actual_child_witness_first.iter().enumerate() {
            if (first..first + entries).contains(&column) {
                return Some(self.layout.child_witness_first(child)? + column - first);
            }
        }
        None
    }

    pub fn normalize_actual_row(&self, row: &CanonicalSparseRow) -> Option<CanonicalSparseRow> {
        let normalize = |terms: &[(usize, F)]| {
            let mut normalized = terms
                .iter()
                .map(|&(column, coefficient)| Some((self.actual_to_canonical(column)?, coefficient)))
                .collect::<Option<Vec<_>>>()?;
            normalized.sort_unstable_by_key(|(column, _)| *column);
            Some(normalized)
        };
        Some(CanonicalSparseRow {
            a: normalize(&row.a)?,
            b: normalize(&row.b)?,
            c: normalize(&row.c)?,
        })
    }
}
