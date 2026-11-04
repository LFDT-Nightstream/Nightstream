//! The glue part of `Σ_j ρ^j·M̃_j(rx, ry)`, accumulated while a shape run of
//! the program emits its rows. No row is stored.

use neo_verified::circuit::{Kind, KINDS, MATRICES};
use neo_verified::layout::{column, glue_row, Shape};
use p3_field_v08::PrimeCharacteristicRing;
use rayon::prelude::*;

use neo_verified::field as verified;
use neo_verified::r1cs::R1cs;

use crate::circuit::block::C;
use crate::circuit::r1cs::{sides, Side};
use crate::circuit::record::{evaluate, Form, Sink, Term};
use crate::field::{from_verified, verified_eq_table, Ext, Gl};

/// `eq(point, i)` from a low and a high table.
pub(crate) struct SplitEq {
    low: Vec<Ext>,
    high: Vec<Ext>,
    bits: usize,
}

impl SplitEq {
    pub(crate) fn new(point: &[Ext]) -> Self {
        let bits = point.len() / 2;
        Self {
            low: verified_eq_table(&point[..bits]),
            high: verified_eq_table(&point[bits..]),
            bits,
        }
    }

    pub(crate) fn at(&self, index: usize) -> Ext {
        self.low[index & ((1 << self.bits) - 1)] * self.high[index >> self.bits]
    }
}

/// A sink that sums the glue rows' matrix entries at `(rx, ry)` and keeps the
/// counts and the statement words.
pub(crate) struct Evaluate<'l> {
    shape: &'l Shape,
    rows: SplitEq,
    columns: SplitEq,
    rho: [Ext; MATRICES],
    pub(crate) total: Ext,
    pub(crate) glue_rows: usize,
    pub(crate) glue_cells: usize,
    pub(crate) blocks: [usize; KINDS.len()],
    pub(crate) publics: Vec<Gl>,
}

impl<'l> Evaluate<'l> {
    pub(crate) fn new(shape: &'l Shape, rx: &[Ext], ry: &[Ext], rho: [Ext; MATRICES]) -> Self {
        Self {
            shape,
            rows: SplitEq::new(rx),
            columns: SplitEq::new(ry),
            rho,
            total: Ext::ZERO,
            glue_rows: 0,
            glue_cells: 0,
            blocks: [0; KINDS.len()],
            publics: Vec::new(),
        }
    }

    /// Whether every count so far, plus `rows` more glue rows, lies inside
    /// the shape. Past it an index would leave the eq tables; such a run's
    /// counts differ from the shape, so `verify` rejects it.
    fn fits(&self, rows: usize) -> bool {
        let shape = self.shape;
        self.glue_rows + rows <= shape.glue_rows
            && self.glue_cells <= shape.glue_cells
            && self.publics.len() <= shape.publics
            && self
                .blocks
                .iter()
                .zip(&shape.blocks)
                .all(|(count, limit)| count <= limit)
    }

    fn side(&self, terms: &[Term]) -> Ext {
        terms
            .iter()
            .map(|&(wire, coefficient)| self.columns.at(column(self.shape, wire)) * from_verified(coefficient))
            .sum()
    }
}

impl Sink for Evaluate<'_> {
    fn row(&mut self, products: &[(Vec<Term>, Vec<Term>)], linear: &[Term]) {
        if self.fits(1) {
            let mut sum = self.rho[C] * self.side(linear);
            for (i, (a, b)) in products.iter().enumerate() {
                sum += self.rho[1 + 2 * i] * self.side(a) + self.rho[2 + 2 * i] * self.side(b);
            }
            self.total += self.rows.at(glue_row(self.shape, self.glue_rows)) * sum;
        }
        self.glue_rows += 1;
    }

    fn cell(&mut self, _: verified::Gl) {
        self.glue_cells += 1;
    }

    fn block(&mut self, kind: Kind, _: &[verified::Gl]) -> Vec<verified::Gl> {
        self.blocks[kind.index()] += 1;
        Vec::new()
    }

    fn public(&mut self, value: verified::Gl) {
        self.publics.push(from_verified(value));
    }

    /// One `eq` per column, then the rows in parallel.
    fn r1cs(&mut self, system: &R1cs, columns: &[Form]) {
        if !self.fits(system.rows()) {
            self.glue_rows += system.rows();
            return;
        }
        let at: Vec<Ext> = columns
            .par_iter()
            .map(|form| evaluate(form, |wire| self.columns.at(column(self.shape, wire))))
            .collect();
        let side = |(columns, coefficients): Side<'_>| -> Ext {
            columns
                .iter()
                .zip(coefficients)
                .map(|(&column, &coefficient)| at[column as usize] * from_verified(coefficient))
                .sum()
        };
        let start = self.glue_rows;
        let total: Ext = (0..system.rows())
            .into_par_iter()
            .map(|row| {
                let [a, b, c] = sides(system, row);
                let sum = self.rho[1] * side(a) + self.rho[2] * side(b) + self.rho[C] * side(c);
                self.rows.at(glue_row(self.shape, start + row)) * sum
            })
            .sum();
        self.total += total;
        self.glue_rows += system.rows();
    }

    fn values(&self) -> bool {
        false
    }
}
