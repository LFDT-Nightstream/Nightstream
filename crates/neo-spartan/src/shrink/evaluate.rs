//! The glue part of `Σ_j ρ^j·M̃_j(rx, ry)`, accumulated while a shape run of
//! the program emits its rows. No row is stored.

use p3_field_v08::PrimeCharacteristicRing;

use super::layout::{Layout, C, MATRICES};
use crate::circuit::record::{Sink, Term};
use crate::field::{eq_table, Ext, Gl};

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
            low: eq_table(&point[..bits]),
            high: eq_table(&point[bits..]),
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
    layout: &'l Layout,
    rows: SplitEq,
    columns: SplitEq,
    rho: [Ext; MATRICES],
    pub(crate) total: Ext,
    pub(crate) glue_rows: usize,
    pub(crate) glue_cells: usize,
    pub(crate) blocks: usize,
    pub(crate) publics: Vec<Gl>,
}

impl<'l> Evaluate<'l> {
    pub(crate) fn new(layout: &'l Layout, rx: &[Ext], ry: &[Ext], rho: [Ext; MATRICES]) -> Self {
        Self {
            layout,
            rows: SplitEq::new(rx),
            columns: SplitEq::new(ry),
            rho,
            total: Ext::ZERO,
            glue_rows: 0,
            glue_cells: 0,
            blocks: 0,
            publics: Vec::new(),
        }
    }

    fn side(&self, terms: &[Term]) -> Ext {
        terms
            .iter()
            .map(|&(wire, coefficient)| self.columns.at(self.layout.column(wire)) * coefficient)
            .sum()
    }
}

impl Sink for Evaluate<'_> {
    fn row(&mut self, products: &[(Vec<Term>, Vec<Term>)], linear: &[Term]) {
        let mut sum = self.rho[C] * self.side(linear);
        for (i, (a, b)) in products.iter().enumerate() {
            sum += self.rho[1 + 2 * i] * self.side(a) + self.rho[2 + 2 * i] * self.side(b);
        }
        self.total += self.rows.at(self.layout.glue_row(self.glue_rows)) * sum;
        self.glue_rows += 1;
    }

    fn cell(&mut self, _: Gl) {
        self.glue_cells += 1;
    }

    fn block(&mut self, _: &[Gl]) {
        self.blocks += 1;
    }

    fn public(&mut self, value: Gl) {
        self.publics.push(value);
    }
}
