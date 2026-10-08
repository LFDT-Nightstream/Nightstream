//! A fixed R1CS system over caller columns, for constraints built outside this
//! crate (the Lean terminal package).
//!
//! Owns the compact row storage. Row `i` is `(Σ a)·(Σ b) = Σ c` with terms
//! `(column, coefficient)`; the caller binds column `k` to its `k`-th wire.

use p3_field_v08::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::field::Gl;

pub struct R1cs {
    columns: usize,
    /// Per row, the ends of its `a`, `b` and `c` terms.
    ends: Vec<[u32; 3]>,
    column: Vec<u32>,
    coefficient: Vec<Gl>,
}

/// The columns and coefficients of one side of a row.
pub(crate) type Side<'a> = (&'a [u32], &'a [Gl]);

impl R1cs {
    pub fn new(columns: usize) -> Self {
        assert!(
            u32::try_from(columns).is_ok(),
            "an R1CS system has fewer than 2^32 columns"
        );
        Self {
            columns,
            ends: Vec::new(),
            column: Vec::new(),
            coefficient: Vec::new(),
        }
    }

    /// Add `(Σ a)·(Σ b) = Σ c`; each coefficient is taken modulo p.
    pub fn push(&mut self, a: &[(usize, u64)], b: &[(usize, u64)], c: &[(usize, u64)]) {
        let mut ends = [0u32; 3];
        for (end, side) in ends.iter_mut().zip([a, b, c]) {
            for &(column, coefficient) in side {
                assert!(column < self.columns, "an R1CS term names a column of its system");
                self.column.push(column as u32);
                self.coefficient.push(Gl::from_u64(coefficient));
            }
            *end = u32::try_from(self.column.len()).expect("an R1CS system has fewer than 2^32 terms");
        }
        self.ends.push(ends);
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn rows(&self) -> usize {
        self.ends.len()
    }

    /// The `a`, `b` and `c` sides of row `row`.
    pub(crate) fn sides(&self, row: usize) -> [Side<'_>; 3] {
        let start = if row == 0 { 0 } else { self.ends[row - 1][2] as usize };
        let [a, b, c] = self.ends[row].map(|end| end as usize);
        let side = |from: usize, to: usize| (&self.column[from..to], &self.coefficient[from..to]);
        [side(start, a), side(a, b), side(b, c)]
    }
}

/// Whether some row fails on the column values `values`.
pub(crate) fn fails(system: &R1cs, values: &[Gl]) -> bool {
    let sum = |(columns, coefficients): Side<'_>| -> Gl {
        columns
            .iter()
            .zip(coefficients)
            .map(|(&column, &coefficient)| coefficient * values[column as usize])
            .sum()
    };
    (0..system.rows()).into_par_iter().any(|row| {
        let [a, b, c] = system.sides(row);
        sum(a) * sum(b) != sum(c)
    })
}
