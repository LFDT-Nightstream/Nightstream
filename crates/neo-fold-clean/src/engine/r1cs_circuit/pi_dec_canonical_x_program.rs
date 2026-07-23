//! Indexed compiler for the strict binary PiDEC public-X rows.
//!
//! The program owns only two equation families: radix recomposition and the
//! uniform-sign canonical child split.  Its row and column costs are derived
//! from the typed plan; no production-sized assignment or row census is
//! materialized.

use std::collections::BTreeSet;
use std::ops::Range;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::row_formula::{canonical_sparse_row, equality_constraint_row, multiplication_constraint_row};
use super::{CanonicalSparseRow, Lc, Var};

const CONSTANT_COLUMN: usize = 0;
const CENTERED_SIGN_ROWS: usize = 2;

/// Shape of the binary public-X lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiDecCanonicalXPlan {
    x_rows: usize,
    active_columns: usize,
    child_count: usize,
}

impl PiDecCanonicalXPlan {
    pub fn new(x_rows: usize, active_columns: usize, child_count: usize) -> Result<Self, &'static str> {
        if x_rows == 0 || active_columns == 0 || child_count == 0 {
            return Err("PiDEC canonical-X plan dimensions must be nonzero");
        }
        let logical_coordinates = x_rows
            .checked_mul(active_columns)
            .ok_or("PiDEC canonical-X plan overflows usize")?;
        let rows_per_coordinate = child_count
            .checked_add(CENTERED_SIGN_ROWS)
            .ok_or("PiDEC canonical-X plan overflows usize")?;
        logical_coordinates
            .checked_mul(rows_per_coordinate)
            .and_then(|_| child_count.checked_add(1 + CENTERED_SIGN_ROWS))
            .and_then(|columns_per_coordinate| logical_coordinates.checked_mul(columns_per_coordinate))
            .and_then(|columns| columns.checked_add(1))
            .ok_or("PiDEC canonical-X plan overflows usize")?;
        Ok(Self {
            x_rows,
            active_columns,
            child_count,
        })
    }

    pub fn x_rows(self) -> usize {
        self.x_rows
    }

    pub fn active_columns(self) -> usize {
        self.active_columns
    }

    pub fn child_count(self) -> usize {
        self.child_count
    }

    pub fn logical_coordinates(self) -> usize {
        self.x_rows * self.active_columns
    }

    pub fn recomposition_rows(self) -> usize {
        self.logical_coordinates()
    }

    pub fn canonicality_rows(self) -> usize {
        self.logical_coordinates() * (self.child_count + CENTERED_SIGN_ROWS)
    }

    pub fn total_rows(self) -> usize {
        self.recomposition_rows() + self.canonicality_rows()
    }

    pub fn canonical_column_count(self) -> usize {
        1 + self.logical_coordinates() * (1 + self.child_count + CENTERED_SIGN_ROWS)
    }

    /// Row-major coordinate consumed by the emitter.
    pub fn active_index(self, x_row: usize, active_column: usize) -> Option<usize> {
        (x_row < self.x_rows && active_column < self.active_columns)
            .then_some(x_row * self.active_columns + active_column)
    }

    /// Column-major public coordinate represented by `(x_row, active_column)`.
    pub fn public_column(self, x_row: usize, active_column: usize) -> Option<usize> {
        (x_row < self.x_rows && active_column < self.active_columns).then_some(active_column * self.x_rows + x_row)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CanonicalLayout {
    plan: PiDecCanonicalXPlan,
}

impl CanonicalLayout {
    fn parent(self, active_index: usize) -> Option<usize> {
        (active_index < self.plan.logical_coordinates()).then_some(1 + active_index)
    }

    fn child(self, child: usize, active_index: usize) -> Option<usize> {
        (child < self.plan.child_count() && active_index < self.plan.logical_coordinates())
            .then_some(1 + self.plan.logical_coordinates() + child * self.plan.logical_coordinates() + active_index)
    }

    fn trace_first(self) -> usize {
        1 + self.plan.logical_coordinates() * (1 + self.plan.child_count())
    }

    fn sign(self, active_index: usize) -> Option<usize> {
        (active_index < self.plan.logical_coordinates()).then_some(self.trace_first() + 2 * active_index)
    }

    fn product(self, active_index: usize) -> Option<usize> {
        self.sign(active_index).map(|column| column + 1)
    }
}

/// Unique semantic owner of one row in the indexed program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PiDecCanonicalXRowOwner {
    Recomposition { active_index: usize },
    SignProduct { active_index: usize },
    SignZero { active_index: usize },
    ChildDigit { active_index: usize, child: usize },
}

/// Pure indexed binary public-X compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiDecCanonicalXProgram {
    plan: PiDecCanonicalXPlan,
}

impl PiDecCanonicalXProgram {
    pub fn new(plan: PiDecCanonicalXPlan) -> Self {
        Self { plan }
    }

    pub fn plan(self) -> PiDecCanonicalXPlan {
        self.plan
    }

    pub fn row_count(self) -> usize {
        self.plan.total_rows()
    }

    pub fn parent_canonical_column(self, active_index: usize) -> Option<usize> {
        CanonicalLayout { plan: self.plan }.parent(active_index)
    }

    pub fn child_canonical_column(self, child: usize, active_index: usize) -> Option<usize> {
        CanonicalLayout { plan: self.plan }.child(child, active_index)
    }

    pub fn sign_canonical_column(self, active_index: usize) -> Option<usize> {
        CanonicalLayout { plan: self.plan }.sign(active_index)
    }

    pub fn product_canonical_column(self, active_index: usize) -> Option<usize> {
        CanonicalLayout { plan: self.plan }.product(active_index)
    }

    pub fn owner(self, relative_row: usize) -> Option<PiDecCanonicalXRowOwner> {
        let logical = self.plan.logical_coordinates();
        if relative_row < logical {
            return Some(PiDecCanonicalXRowOwner::Recomposition {
                active_index: relative_row,
            });
        }
        let canonical_row = relative_row - logical;
        if canonical_row >= self.plan.canonicality_rows() {
            return None;
        }
        let rows_per_coordinate = self.plan.child_count() + CENTERED_SIGN_ROWS;
        let active_index = canonical_row / rows_per_coordinate;
        match canonical_row % rows_per_coordinate {
            0 => Some(PiDecCanonicalXRowOwner::SignProduct { active_index }),
            1 => Some(PiDecCanonicalXRowOwner::SignZero { active_index }),
            child_row => Some(PiDecCanonicalXRowOwner::ChildDigit {
                active_index,
                child: child_row - CENTERED_SIGN_ROWS,
            }),
        }
    }

    pub fn row_at(self, relative_row: usize) -> Option<CanonicalSparseRow> {
        let layout = CanonicalLayout { plan: self.plan };
        match self.owner(relative_row)? {
            PiDecCanonicalXRowOwner::Recomposition { active_index } => {
                let children = (0..self.plan.child_count())
                    .map(|child| layout.child(child, active_index))
                    .collect::<Option<Vec<_>>>()?;
                self.recomposition_row(layout.parent(active_index)?, &children)
            }
            PiDecCanonicalXRowOwner::SignProduct { active_index } => Some(sign_product_row(
                layout.sign(active_index)?,
                layout.product(active_index)?,
            )),
            PiDecCanonicalXRowOwner::SignZero { active_index } => {
                Some(sign_zero_row(layout.sign(active_index)?, layout.product(active_index)?))
            }
            PiDecCanonicalXRowOwner::ChildDigit { active_index, child } => Some(child_digit_row(
                layout.child(child, active_index)?,
                layout.sign(active_index)?,
            )),
        }
    }

    /// Exact recomposition row over caller-owned physical columns.
    pub fn recomposition_row(self, parent: usize, children: &[usize]) -> Option<CanonicalSparseRow> {
        if children.len() != self.plan.child_count() {
            return None;
        }
        let mut combination = Lc::zero();
        let mut weight = F::ONE;
        let radix = F::from_u64(2);
        for &child in children {
            combination.add_term(Var::from_column_for_trace(child), weight);
            weight *= radix;
        }
        Some(canonical_sparse_row(&equality_constraint_row(
            &Lc::from_var(Var::from_column_for_trace(parent)),
            &combination,
        )))
    }

    /// Exact per-coordinate canonicality row over physical columns.
    pub fn canonicality_row(
        self,
        relative_row: usize,
        sign: usize,
        product: usize,
        children: &[usize],
    ) -> Option<CanonicalSparseRow> {
        if children.len() != self.plan.child_count() || relative_row >= self.plan.child_count() + CENTERED_SIGN_ROWS {
            return None;
        }
        match relative_row {
            0 => Some(sign_product_row(sign, product)),
            1 => Some(sign_zero_row(sign, product)),
            child_row => Some(child_digit_row(children[child_row - CENTERED_SIGN_ROWS], sign)),
        }
    }
}

fn sign_product_row(sign: usize, product: usize) -> CanonicalSparseRow {
    let sign = Var::from_column_for_trace(sign);
    let left = Lc::from_var(sign).add_scaled(&Lc::from_const(F::ONE), F::ONE);
    canonical_sparse_row(&multiplication_constraint_row(
        &left,
        &Lc::from_var(sign),
        Var::from_column_for_trace(product),
    ))
}

fn sign_zero_row(sign: usize, product: usize) -> CanonicalSparseRow {
    let right = Lc::from_var(Var::from_column_for_trace(sign)).add_scaled(&Lc::from_const(F::ONE), -F::ONE);
    canonical_sparse_row(&(Lc::from_var(Var::from_column_for_trace(product)), right, Lc::zero()))
}

fn child_digit_row(digit: usize, sign: usize) -> CanonicalSparseRow {
    let digit = Var::from_column_for_trace(digit);
    let right = Lc::from_var(digit).add_scaled(&Lc::from_var(Var::from_column_for_trace(sign)), -F::ONE);
    canonical_sparse_row(&(Lc::from_var(digit), right, Lc::zero()))
}

/// Bijection from the program's compact canonical columns to live builder
/// columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiDecCanonicalXColumnMap {
    canonical_to_actual: Vec<usize>,
}

impl PiDecCanonicalXColumnMap {
    pub fn new(program: PiDecCanonicalXProgram, canonical_to_actual: Vec<usize>) -> Result<Self, &'static str> {
        if canonical_to_actual.len() != program.plan().canonical_column_count() {
            return Err("PiDEC canonical-X column-map length drift");
        }
        if canonical_to_actual.first().copied() != Some(CONSTANT_COLUMN) {
            return Err("PiDEC canonical-X constant column drift");
        }
        if canonical_to_actual
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != canonical_to_actual.len()
        {
            return Err("PiDEC canonical-X column map is not injective");
        }
        Ok(Self { canonical_to_actual })
    }

    pub fn canonical_to_actual(&self) -> &[usize] {
        &self.canonical_to_actual
    }

    pub fn actual_column(&self, canonical_column: usize) -> Option<usize> {
        self.canonical_to_actual.get(canonical_column).copied()
    }

    pub fn relabel(&self, row: &CanonicalSparseRow) -> Option<CanonicalSparseRow> {
        let relabel_terms = |terms: &[(usize, F)]| {
            let mut out = terms
                .iter()
                .map(|&(column, coefficient)| Some((self.actual_column(column)?, coefficient)))
                .collect::<Option<Vec<_>>>()?;
            out.sort_unstable_by_key(|(column, _)| *column);
            Some(out)
        };
        Some(CanonicalSparseRow {
            a: relabel_terms(&row.a)?,
            b: relabel_terms(&row.b)?,
            c: relabel_terms(&row.c)?,
        })
    }
}

/// Non-optional receipt returned by the production strict PiDEC emitter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiDecCanonicalXReceipt {
    program: PiDecCanonicalXProgram,
    strict_rows: Range<usize>,
    recomposition_rows: Range<usize>,
    canonicality_rows: Range<usize>,
    columns: PiDecCanonicalXColumnMap,
}

impl PiDecCanonicalXReceipt {
    pub fn new(
        program: PiDecCanonicalXProgram,
        strict_rows: Range<usize>,
        recomposition_rows: Range<usize>,
        canonicality_rows: Range<usize>,
        columns: PiDecCanonicalXColumnMap,
    ) -> Result<Self, &'static str> {
        if strict_rows.start > recomposition_rows.start
            || recomposition_rows.end > canonicality_rows.start
            || canonicality_rows.end > strict_rows.end
            || recomposition_rows.len() != program.plan().recomposition_rows()
            || canonicality_rows.len() != program.plan().canonicality_rows()
        {
            return Err("PiDEC canonical-X receipt row schedule drift");
        }
        if columns.canonical_to_actual().len() != program.plan().canonical_column_count() {
            return Err("PiDEC canonical-X receipt column schedule drift");
        }
        Ok(Self {
            program,
            strict_rows,
            recomposition_rows,
            canonicality_rows,
            columns,
        })
    }

    pub fn program(&self) -> PiDecCanonicalXProgram {
        self.program
    }

    pub fn strict_rows(&self) -> Range<usize> {
        self.strict_rows.clone()
    }

    pub fn recomposition_rows(&self) -> Range<usize> {
        self.recomposition_rows.clone()
    }

    pub fn canonicality_rows(&self) -> Range<usize> {
        self.canonicality_rows.clone()
    }

    pub fn columns(&self) -> &PiDecCanonicalXColumnMap {
        &self.columns
    }

    pub fn physical_row(&self, relative_row: usize) -> Option<usize> {
        if relative_row < self.program.plan().recomposition_rows() {
            return Some(self.recomposition_rows.start + relative_row);
        }
        let canonicality_row = relative_row - self.program.plan().recomposition_rows();
        (canonicality_row < self.program.plan().canonicality_rows())
            .then_some(self.canonicality_rows.start + canonicality_row)
    }

    pub fn actual_row_at(&self, relative_row: usize) -> Option<CanonicalSparseRow> {
        self.columns.relabel(&self.program.row_at(relative_row)?)
    }
}
