//! A fixed R1CS system over caller columns, for constraints built outside this crate (the
//! Lean terminal package).
//!
//! Owns the compact row storage. Row `i` is `(Σ a)·(Σ b) = Σ c` with terms
//! `(column, coefficient)`; the caller binds column `k` to its `k`-th wire.

use crate::field::{gl_add, gl_mul, gl_word, reduce, Gl, GL_ZERO};

/// The largest column count and term count: a column and a term end are `u32`s.
const LIMIT: usize = 0xffff_ffff;

pub struct R1cs {
    columns: usize,
    /// Per row, the ends of its `a`, `b` and `c` terms.
    ends: Vec<[u32; 3]>,
    column: Vec<u32>,
    coefficient: Vec<Gl>,
}

impl R1cs {
    /// Panics unless `columns` fits a `u32`.
    pub fn new(columns: usize) -> Self {
        assert!(columns <= LIMIT);
        Self {
            columns,
            ends: Vec::new(),
            column: Vec::new(),
            coefficient: Vec::new(),
        }
    }

    /// Add `(Σ a)·(Σ b) = Σ c`; each coefficient is taken modulo p. Panics on a column
    /// outside the system or past `2^32 - 1` terms.
    pub fn push(&mut self, a: &[(usize, u64)], b: &[(usize, u64)], c: &[(usize, u64)]) {
        let end_a = self.push_side(a);
        let end_b = self.push_side(b);
        let end_c = self.push_side(c);
        self.ends.push([end_a, end_b, end_c]);
    }

    /// The terms of one side; returns the end of the system's terms.
    fn push_side(&mut self, side: &[(usize, u64)]) -> u32 {
        let mut k = 0;
        while k < side.len() {
            let (column, coefficient) = side[k];
            assert!(column < self.columns);
            self.column.push(column as u32);
            self.coefficient.push(reduce(coefficient));
            k = k.wrapping_add(1);
        }
        assert!(self.column.len() <= LIMIT);
        self.column.len() as u32
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn rows(&self) -> usize {
        self.ends.len()
    }

    /// The first term of row `row` and the ends of its `a`, `b` and `c` terms.
    pub fn bounds(&self, row: usize) -> (usize, [usize; 3]) {
        let start = if row == 0 {
            0
        } else {
            self.ends[row.wrapping_sub(1)][2] as usize
        };
        let [a, b, c] = self.ends[row];
        (start, [a as usize, b as usize, c as usize])
    }

    /// The column of every term, in order.
    pub fn term_columns(&self) -> &[u32] {
        &self.column
    }

    /// The coefficient of every term, in order.
    pub fn term_coefficients(&self) -> &[Gl] {
        &self.coefficient
    }

    /// Row `row` as `push` takes it, each coefficient a canonical word.
    pub fn row(&self, row: usize) -> [Vec<(usize, u64)>; 3] {
        let (start, ends) = self.bounds(row);
        let a = self.side_terms(start, ends[0]);
        let b = self.side_terms(ends[0], ends[1]);
        let c = self.side_terms(ends[1], ends[2]);
        [a, b, c]
    }

    fn side_terms(&self, from: usize, to: usize) -> Vec<(usize, u64)> {
        let mut terms = Vec::new();
        let mut k = from;
        while k < to {
            terms.push((self.column[k] as usize, gl_word(self.coefficient[k])));
            k = k.wrapping_add(1);
        }
        terms
    }

    /// `Σ coefficient·values[column]` over the terms `from..to`.
    fn side_value(&self, values: &[Gl], from: usize, to: usize) -> Gl {
        let mut sum = GL_ZERO;
        let mut k = from;
        while k < to {
            sum = gl_add(sum, gl_mul(self.coefficient[k], values[self.column[k] as usize]));
            k = k.wrapping_add(1);
        }
        sum
    }
}

/// Whether some row of `system` fails on the column values `values`.
pub fn fails(system: &R1cs, values: &[Gl]) -> bool {
    let mut failed = false;
    let mut row = 0;
    while row < system.rows() && !failed {
        let (start, [a, b, c]) = system.bounds(row);
        let left = gl_mul(system.side_value(values, start, a), system.side_value(values, a, b));
        failed = gl_word(left) != gl_word(system.side_value(values, b, c));
        row = row.wrapping_add(1);
    }
    failed
}
