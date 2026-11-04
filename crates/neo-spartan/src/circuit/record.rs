//! The prover's sink: `Trace` keeps every row, cell and block of a run in Plonky3 0.8 values.
//!
//! `Recorder`, `Form` and `Sink` are in `neo_verified::record`. A trace computes each block's
//! cells here (`block::trace`) and fills R1CS rows in parallel.

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_verified::circuit::{Kind, Wire, CELLS, KINDS};
use neo_verified::field as verified;
use neo_verified::r1cs::R1cs;
use neo_verified::record::expand;
pub(crate) use neo_verified::record::{Form, Recorder, Sink, Term, ONE};
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;

use super::block::trace;
use super::poseidon2::OUTPUT;
use super::r1cs::{sides, Side};
use crate::field::{from_verified, to_verified, Ext, Gl};
use crate::hash::{absorb, squeeze};

/// A row entry of a trace: a wire and its coefficient.
pub(crate) type Entry = (Wire, Gl);

/// The terms of `form`.
pub(crate) fn terms(form: &Form) -> impl Iterator<Item = Term> + '_ {
    (0..form.term_count()).map(|k| form.term(k))
}

/// `Σ coefficient·at(wire)` of `form`, the constant on the one column.
pub(crate) fn evaluate(form: &Form, at: impl Fn(Wire) -> Ext) -> Ext {
    let offset = from_verified(form.offset());
    let constant = if offset == Gl::ZERO {
        Ext::ZERO
    } else {
        at(ONE) * offset
    };
    terms(form).fold(constant, |sum, (wire, coefficient)| {
        sum + at(wire) * from_verified(coefficient)
    })
}

fn entries(terms: &[Term]) -> impl Iterator<Item = Entry> + '_ {
    terms
        .iter()
        .map(|&(wire, coefficient)| (wire, from_verified(coefficient)))
}

/// Every row, cell and block of a run, for the shrink prover and for tests.
#[derive(Default, PartialEq)]
pub(crate) struct Trace {
    /// Glue cell values.
    pub(crate) glue: Vec<Gl>,
    /// Block cell values per kind, `kind.cells()` per block.
    pub(crate) blocks: [Vec<Gl>; KINDS.len()],
    /// ClaimWords words (index 1 and up; index 0 is the constant one).
    pub(crate) public: Vec<Gl>,
    /// Row entries; `rows[r]` holds the start of each of the seven sides
    /// `A0, B0, A1, B1, A2, B2, C` and the end.
    pub(crate) entries: Vec<Entry>,
    pub(crate) rows: Vec<[u32; 8]>,
}

impl Sink for Trace {
    fn row(&mut self, products: &[(Vec<Term>, Vec<Term>)], linear: &[Term]) {
        assert!(products.len() <= 3);
        let mut offsets = [0u32; 8];
        for side in 0..6 {
            offsets[side] = self.entries.len() as u32;
            if let Some(pair) = products.get(side / 2) {
                let terms = if side % 2 == 0 { &pair.0 } else { &pair.1 };
                self.entries.extend(entries(terms));
            }
        }
        offsets[6] = self.entries.len() as u32;
        self.entries.extend(entries(linear));
        offsets[7] = self.entries.len() as u32;
        self.rows.push(offsets);
    }

    fn cell(&mut self, value: verified::Gl) {
        self.glue.push(from_verified(value));
    }

    fn block(&mut self, kind: Kind, inputs: &[verified::Gl]) -> Vec<verified::Gl> {
        let inputs: Vec<Gl> = inputs.iter().map(|&value| from_verified(value)).collect();
        let cells = trace(kind, &inputs);
        self.blocks[kind.index()].extend_from_slice(&cells);
        cells[OUTPUT..CELLS]
            .iter()
            .map(|&value| to_verified(value))
            .collect()
    }

    fn public(&mut self, value: verified::Gl) {
        self.public.push(from_verified(value));
    }

    /// The default's rows and entries, built in parallel: every row's entries
    /// depend only on `system` and `columns`, so the rows are sized, placed
    /// in order, then filled.
    fn r1cs(&mut self, system: &R1cs, columns: &[Form]) {
        let size = |(side, _): Side<'_>| -> usize {
            side.iter()
                .map(|&column| {
                    let form = &columns[column as usize];
                    form.term_count() + usize::from(form.offset() != verified::GL_ZERO)
                })
                .sum()
        };
        let sizes: Vec<[usize; 3]> = (0..system.rows())
            .into_par_iter()
            .map(|row| sides(system, row).map(size))
            .collect();
        let start = self.entries.len();
        let mut at = start;
        self.rows.reserve(sizes.len());
        for [a, b, c] in &sizes {
            // A0 = a, B0 = b, the other products empty, then C = c.
            let ends = [
                at,
                at + a,
                at + a + b,
                at + a + b,
                at + a + b,
                at + a + b,
                at + a + b,
                at + a + b + c,
            ];
            self.rows.push(ends.map(|end| end as u32));
            at += a + b + c;
        }
        self.entries.resize(at, (ONE, Gl::ZERO));
        let mut rest = &mut self.entries[start..];
        let mut slices = Vec::with_capacity(sizes.len());
        for [a, b, c] in &sizes {
            let (row, tail) = rest.split_at_mut(a + b + c);
            slices.push(row);
            rest = tail;
        }
        slices.into_par_iter().enumerate().for_each(|(row, out)| {
            let (first, [a, b, c]) = system.bounds(row);
            let terms = [(first, a), (a, b), (b, c)]
                .into_iter()
                .flat_map(|(from, to)| expand(system, from, to, columns));
            for (slot, (wire, coefficient)) in out.iter_mut().zip(terms) {
                *slot = (wire, from_verified(coefficient));
            }
        });
    }
}

impl Trace {
    #[cfg(test)]
    pub(crate) fn value(&self, wire: Wire) -> Gl {
        match wire {
            Wire::Glue(cell) => self.glue[cell as usize],
            Wire::Block(kind, block, cell) => self.blocks[kind.index()][block as usize * kind.cells() + cell as usize],
            Wire::Public(0) => Gl::ONE,
            Wire::Public(index) => self.public[index as usize - 1],
        }
    }

    /// The value of glue row `r` (zero when it holds).
    #[cfg(test)]
    pub(crate) fn row_value(&self, r: usize) -> Gl {
        let offsets = self.rows[r];
        let side = |s: usize| {
            self.entries[offsets[s] as usize..offsets[s + 1] as usize]
                .iter()
                .fold(Gl::ZERO, |acc, &(wire, c)| acc + c * self.value(wire))
        };
        (0..3).fold(-side(6), |acc, i| acc + side(2 * i) * side(2 * i + 1))
    }

    /// The number of blocks of `kind`.
    pub(crate) fn count(&self, kind: Kind) -> usize {
        self.blocks[kind.index()].len() / kind.cells()
    }

    /// Row `r`'s seven sides, each as its term count and, per term, the wire
    /// (`0 cell`, `1 block cell` or `2 index`) and the coefficient.
    pub(crate) fn row_words(&self, r: usize) -> Vec<u64> {
        let offsets = self.rows[r];
        let mut words = Vec::new();
        for side in 0..7 {
            let terms = &self.entries[offsets[side] as usize..offsets[side + 1] as usize];
            words.push(terms.len() as u64);
            for &(name, coefficient) in terms {
                match name {
                    Wire::Glue(cell) => words.extend([0, u64::from(cell)]),
                    Wire::Block(_, block, cell) => words.extend([1, u64::from(block), u64::from(cell)]),
                    Wire::Public(index) => words.extend([2, u64::from(index)]),
                }
                words.push(coefficient.as_canonical_u64());
            }
        }
        words
    }

    /// The run digest (`runDigest` in `formal/compression-lean`,
    /// `CompressionLeanExe/Digest.lean`): the Poseidon2 leaf sponge (`hash_leaf`) of the
    /// glue cell, block, statement word and row counts, then every row's
    /// words. Statement and cell values are not part of it.
    pub(crate) fn digest(&self) -> [u64; 4] {
        let counts = [
            self.glue.len(),
            self.count(Kind::Permutation),
            self.public.len(),
            self.rows.len(),
        ];
        let mut state = [Gl::ZERO; WIDTH];
        let mut at = absorb(&mut state, 0, &counts.map(Gl::from_usize));
        for r in 0..self.rows.len() {
            let words: Vec<Gl> = self.row_words(r).into_iter().map(Gl::from_u64).collect();
            at = absorb(&mut state, at, &words);
        }
        squeeze(&mut state, at).map(|word| word.as_canonical_u64())
    }

    /// The first block, of any kind, with a failing row.
    #[cfg(test)]
    pub(crate) fn failing_block(&self) -> Option<(Kind, usize)> {
        KINDS.into_iter().find_map(|kind| {
            (0..self.count(kind))
                .find(|&block| self.block_failure(kind, block).is_some())
                .map(|block| (kind, block))
        })
    }

    /// The first row of block `block` of `kind` that fails, if any.
    #[cfg(test)]
    pub(crate) fn block_failure(&self, kind: Kind, block: usize) -> Option<usize> {
        let cells = kind.cells();
        super::block::failure(kind, &self.blocks[kind.index()][block * cells..(block + 1) * cells])
    }
}
