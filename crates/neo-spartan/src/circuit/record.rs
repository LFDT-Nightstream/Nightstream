//! The constraint reading of `Backend`.
//!
//! Owns: forms (short linear combinations with their value), wires, the glue
//! rows, the permutation blocks, and the first failed check. Does not own
//! where rows and cells sit in the shrink cube.
//!
//! A glue row is `Σ_{i<3} (A_i·z)·(B_i·z) − C·z = 0`. A permutation is one
//! block (`block.rs`) whose rows its template fixes; its inputs are tied to
//! the caller's forms by one linear glue row each. A fixed R1CS system
//! (`r1cs.rs`) is one glue row per row. The value of every row is computed
//! when it is emitted, so the first failed check is known without stopping:
//! the rows never depend on the values.

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use p3_field_v08::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;

use super::block::{Kind, KINDS};
use super::r1cs::{self, R1cs, Side};
use super::Backend;
use crate::field::{Ext, Gl};
use crate::hash::{absorb, squeeze};
use crate::Error;

/// A witness cell or a public word. `Public(0)` is the constant one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Wire {
    Glue(u32),
    /// Kind, block, cell.
    Block(Kind, u32, u16),
    Public(u32),
}

pub(crate) const ONE: Wire = Wire::Public(0);
pub(crate) type Term = (Wire, Gl);

/// Terms a form carries before it becomes a cell.
const FORM_TERMS: usize = 4;

/// `Σ coefficient·wire + constant`, and its value.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Form {
    terms: [Term; FORM_TERMS],
    len: u8,
    constant: Gl,
    value: Gl,
}

impl Form {
    fn constant(value: Gl) -> Self {
        Self {
            terms: [(ONE, Gl::ZERO); FORM_TERMS],
            len: 0,
            constant: value,
            value,
        }
    }

    fn wire(wire: Wire, value: Gl) -> Self {
        let mut form = Self::constant(Gl::ZERO);
        form.terms[0] = (wire, Gl::ONE);
        form.len = 1;
        form.value = value;
        form
    }

    pub(crate) fn value(&self) -> Gl {
        self.value
    }

    fn is_constant(&self) -> bool {
        self.len == 0
    }

    fn terms(&self) -> &[Term] {
        &self.terms[..self.len as usize]
    }

    /// `Σ coefficient·at(wire)`, the constant on the one column.
    pub(crate) fn evaluate(&self, at: impl Fn(Wire) -> Ext) -> Ext {
        let constant = if self.constant == Gl::ZERO {
            Ext::ZERO
        } else {
            at(ONE) * self.constant
        };
        self.terms()
            .iter()
            .fold(constant, |sum, &(wire, coefficient)| sum + at(wire) * coefficient)
    }

    /// The terms with the constant as a term on the one column.
    fn row_terms(&self) -> Vec<Term> {
        let mut terms = self.terms().to_vec();
        if self.constant != Gl::ZERO {
            terms.push((ONE, self.constant));
        }
        terms
    }
}

/// `Σ weight·form` as a merged term list, a constant and a value.
fn combine(parts: &[(Form, Gl)]) -> (Vec<Term>, Gl, Gl) {
    let mut terms: Vec<Term> = Vec::new();
    let (mut constant, mut value) = (Gl::ZERO, Gl::ZERO);
    for &(form, weight) in parts {
        constant += form.constant * weight;
        value += form.value * weight;
        for &(wire, coefficient) in form.terms() {
            match terms.iter_mut().find(|(w, _)| *w == wire) {
                Some((_, existing)) => *existing += coefficient * weight,
                None => terms.push((wire, coefficient * weight)),
            }
        }
    }
    terms.retain(|&(_, coefficient)| coefficient != Gl::ZERO);
    (terms, constant, value)
}

/// `Σ coefficient·columns[column]` as row terms, the constants on the one
/// column. Repeated wires stay separate terms; a row sums them.
pub(crate) fn expand(side: Side<'_>, columns: &[Form]) -> Vec<Term> {
    let mut terms = Vec::new();
    for (&column, &coefficient) in side.0.iter().zip(side.1) {
        let form = &columns[column as usize];
        terms.extend(
            form.terms()
                .iter()
                .map(|&(wire, c)| (wire, c * coefficient)),
        );
        if form.constant != Gl::ZERO {
            terms.push((ONE, form.constant * coefficient));
        }
    }
    terms
}

/// Where recorded rows, cells and blocks go.
pub(crate) trait Sink {
    /// One glue row: `Σ (Σ a)·(Σ b) − Σ c`, with each side's terms.
    fn row(&mut self, products: &[(Vec<Term>, Vec<Term>)], linear: &[Term]);
    fn cell(&mut self, value: Gl);
    /// The cells of one block, in template order (empty when the sink
    /// takes no values).
    fn block(&mut self, kind: Kind, cells: &[Gl]);
    fn public(&mut self, value: Gl);
    /// The rows of `system` as glue rows, in order, with `columns[k]` as its
    /// column `k`.
    fn r1cs(&mut self, system: &R1cs, columns: &[Form]) {
        for row in 0..system.rows() {
            let [a, b, c] = system.sides(row).map(|side| expand(side, columns));
            self.row(&[(a, b)], &c);
        }
    }
    /// Whether the sink reads block cell values. A shape run that only
    /// counts or evaluates rows does not, so blocks are not computed.
    fn values(&self) -> bool {
        true
    }
}

/// The constraint reading of a verifier run.
pub(crate) struct Recorder<S: Sink> {
    sink: S,
    glue: u32,
    blocks: [u32; KINDS.len()],
    publics: u32,
    failure: Option<&'static str>,
}

impl<S: Sink> Recorder<S> {
    pub(crate) fn new(sink: S) -> Self {
        Self {
            sink,
            glue: 0,
            blocks: [0; KINDS.len()],
            publics: 1,
            failure: None,
        }
    }

    /// The sink and the first failed check, if any.
    pub(crate) fn finish(self) -> (S, Option<&'static str>) {
        (self.sink, self.failure)
    }

    fn fail(&mut self, what: &'static str) {
        self.failure.get_or_insert(what);
    }

    fn cell(&mut self, value: Gl) -> Form {
        let wire = Wire::Glue(self.glue);
        self.glue += 1;
        self.sink.cell(value);
        Form::wire(wire, value)
    }

    /// Emit a row; `value` is its value under the witness.
    fn row(&mut self, products: &[(Vec<Term>, Vec<Term>)], linear: &[Term], value: Gl, what: &'static str) {
        if value != Gl::ZERO {
            self.fail(what);
        }
        self.sink.row(products, linear);
    }

    /// A form from merged terms; a new cell and one linear row if too long.
    fn form(&mut self, terms: Vec<Term>, constant: Gl, value: Gl) -> Form {
        if terms.len() <= FORM_TERMS {
            let mut form = Form::constant(constant);
            form.terms[..terms.len()].copy_from_slice(&terms);
            form.len = terms.len() as u8;
            form.value = value;
            return form;
        }
        let cell = self.cell(value);
        let mut linear = vec![(cell.terms[0].0, Gl::ONE)];
        linear.extend(
            terms
                .iter()
                .map(|&(wire, coefficient)| (wire, -coefficient)),
        );
        if constant != Gl::ZERO {
            linear.push((ONE, -constant));
        }
        self.row(&[], &linear, Gl::ZERO, "linear cell");
        cell
    }

    /// One block of `kind` on `inputs`; returns its output cells.
    fn block(&mut self, kind: Kind, inputs: &[Form]) -> Vec<Form> {
        let block = self.blocks[kind.index()];
        self.blocks[kind.index()] += 1;
        let cells = if self.sink.values() {
            let values: Vec<Gl> = inputs.iter().map(Form::value).collect();
            kind.trace(&values)
        } else {
            Vec::new()
        };
        self.sink.block(kind, &cells);
        for (cell, form) in inputs.iter().enumerate() {
            let mut linear = vec![(Wire::Block(kind, block, cell as u16), Gl::ONE)];
            linear.extend(form.row_terms().into_iter().map(|(wire, c)| (wire, -c)));
            self.row(&[], &linear, Gl::ZERO, "block input");
        }
        kind.outputs()
            .map(|cell| {
                let value = cells.get(cell).copied().unwrap_or(Gl::ZERO);
                Form::wire(Wire::Block(kind, block, cell as u16), value)
            })
            .collect()
    }

    fn linear(&mut self, parts: &[(Form, Gl)]) -> Form {
        let (terms, constant, value) = combine(parts);
        self.form(terms, constant, value)
    }

    /// Row `k` of a cubic-extension product `a·b`: `Σ_i a_i·L_ik(b)` with the
    /// structure constants of `x^3 = x + 1`.
    fn ext_rows(a: &[Form; 3], b: &[Form; 3]) -> [Vec<(Vec<Term>, Vec<Term>)>; 3] {
        let sum = |forms: &[Form]| -> Vec<Term> {
            let parts: Vec<(Form, Gl)> = forms.iter().map(|&f| (f, Gl::ONE)).collect();
            let (mut terms, constant, _) = combine(&parts);
            if constant != Gl::ZERO {
                terms.push((ONE, constant));
            }
            terms
        };
        let pair = |i: usize, forms: &[Form]| (a[i].row_terms(), sum(forms));
        [
            vec![pair(0, &[b[0]]), pair(1, &[b[2]]), pair(2, &[b[1]])],
            vec![pair(0, &[b[1]]), pair(1, &[b[0], b[2]]), pair(2, &[b[1], b[2]])],
            vec![pair(0, &[b[2]]), pair(1, &[b[1]]), pair(2, &[b[0], b[2]])],
        ]
    }
}

fn ext_value(a: &[Form; 3]) -> Ext {
    Ext::from_basis_coefficients_fn(|i| a[i].value)
}

fn ext_coordinates(value: Ext) -> [Gl; 3] {
    let slice = <Ext as BasedVectorSpace<Gl>>::as_basis_coefficients_slice(&value);
    [slice[0], slice[1], slice[2]]
}

impl<S: Sink> Backend for Recorder<S> {
    type F = Form;
    type E = [Form; 3];

    fn constant(&mut self, value: u64) -> Form {
        Form::constant(Gl::from_u64(value))
    }

    fn private(&mut self, value: u64) -> Form {
        self.cell(Gl::from_u64(value))
    }

    fn public(&mut self, value: u64) -> Form {
        let value = Gl::from_u64(value);
        let wire = Wire::Public(self.publics);
        self.publics += 1;
        self.sink.public(value);
        Form::wire(wire, value)
    }

    fn add(&mut self, a: Form, b: Form) -> Form {
        self.linear(&[(a, Gl::ONE), (b, Gl::ONE)])
    }

    fn sub(&mut self, a: Form, b: Form) -> Form {
        self.linear(&[(a, Gl::ONE), (b, -Gl::ONE)])
    }

    fn scale(&mut self, a: Form, by: u64) -> Form {
        self.linear(&[(a, Gl::from_u64(by))])
    }

    fn mul(&mut self, a: Form, b: Form) -> Form {
        if a.is_constant() {
            return self.linear(&[(b, a.constant)]);
        }
        if b.is_constant() {
            return self.linear(&[(a, b.constant)]);
        }
        let product = self.cell(a.value * b.value);
        self.row(&[(a.row_terms(), b.row_terms())], product.terms(), Gl::ZERO, "product");
        product
    }

    fn ext(&mut self, coordinates: [Form; 3]) -> [Form; 3] {
        coordinates
    }

    fn coordinates(&mut self, a: [Form; 3]) -> [Form; 3] {
        a
    }

    fn ext_constant(&mut self, value: [u64; 3]) -> [Form; 3] {
        value.map(|word| Form::constant(Gl::from_u64(word)))
    }

    fn ext_add(&mut self, a: [Form; 3], b: [Form; 3]) -> [Form; 3] {
        std::array::from_fn(|i| self.add(a[i], b[i]))
    }

    fn ext_sub(&mut self, a: [Form; 3], b: [Form; 3]) -> [Form; 3] {
        std::array::from_fn(|i| self.sub(a[i], b[i]))
    }

    fn ext_mul(&mut self, a: [Form; 3], b: [Form; 3]) -> [Form; 3] {
        let product = ext_coordinates(ext_value(&a) * ext_value(&b));
        if a.iter().all(Form::is_constant) || b.iter().all(Form::is_constant) {
            let (constant, other) = if a.iter().all(Form::is_constant) {
                (a, b)
            } else {
                (b, a)
            };
            let c = Ext::from_basis_coefficients_fn(|i| constant[i].constant);
            // Column j of multiplication by c: the coordinates of c·w^j.
            let columns: [[Gl; 3]; 3] = std::array::from_fn(|j| {
                ext_coordinates(c * Ext::from_basis_coefficients_fn(|i| Gl::from_bool(i == j)))
            });
            return std::array::from_fn(|k| {
                let parts: Vec<(Form, Gl)> = (0..3).map(|j| (other[j], columns[j][k])).collect();
                self.linear(&parts)
            });
        }
        let rows = Self::ext_rows(&a, &b);
        std::array::from_fn(|k| {
            let cell = self.cell(product[k]);
            self.row(&rows[k], cell.terms(), Gl::ZERO, "extension product");
            cell
        })
    }

    fn ext_scale(&mut self, a: [Form; 3], by: Form) -> [Form; 3] {
        std::array::from_fn(|i| self.mul(a[i], by))
    }

    fn ext_inverse(&mut self, a: [Form; 3], what: &'static str) -> Result<[Form; 3], Error> {
        let value = ext_value(&a);
        let inverse = value.try_inverse();
        if let (true, Some(inverse)) = (a.iter().all(Form::is_constant), inverse) {
            return Ok(ext_coordinates(inverse).map(Form::constant));
        }
        // A constant zero falls through: its rows cannot hold, so the shrink
        // proves the failure too.
        let coordinates = ext_coordinates(inverse.unwrap_or(Ext::ZERO));
        let inverse: [Form; 3] = std::array::from_fn(|i| self.cell(coordinates[i]));
        let rows = Self::ext_rows(&a, &inverse);
        let product = ext_coordinates(value * ext_value(&inverse));
        for (k, row) in rows.iter().enumerate() {
            let one = Gl::from_bool(k == 0);
            let linear = if k == 0 { vec![(ONE, Gl::ONE)] } else { Vec::new() };
            self.row(row, &linear, product[k] - one, what);
        }
        Ok(inverse)
    }

    fn assert_zero(&mut self, a: Form, what: &'static str) -> Result<(), Error> {
        // A nonzero constant still gets its row, `c·1 = 0`, which cannot
        // hold, so the shrink proves the failure too.
        if a.is_constant() && a.constant == Gl::ZERO {
            return Ok(());
        }
        self.row(&[], &a.row_terms(), a.value, what);
        Ok(())
    }

    fn bits(&mut self, a: Form, count: usize, what: &'static str) -> Result<Vec<Form>, Error> {
        let value = a.value.as_canonical_u64();
        let bits: Vec<Form> = (0..count)
            .map(|i| self.cell(Gl::from_u64((value >> i) & 1)))
            .collect();
        for bit in &bits {
            let wire = bit.terms();
            self.row(&[(wire.to_vec(), wire.to_vec())], wire, Gl::ZERO, what);
        }
        let mut linear = a.row_terms();
        let mut sum = Gl::ZERO;
        for (i, bit) in bits.iter().enumerate() {
            let weight = Gl::from_u64(1u64 << i);
            linear.push((bit.terms[0].0, -weight));
            sum += weight * bit.value;
        }
        self.row(&[], &linear, a.value - sum, what);
        if count == 64 {
            // Canonical: the high 32 bits all one forces the low 32 bits to zero.
            let mut high = bits[32];
            for bit in &bits[33..] {
                high = self.mul(high, *bit);
            }
            let low: Vec<Term> = bits[..32].iter().map(|bit| bit.terms[0]).collect();
            let low_value = bits[..32].iter().fold(Gl::ZERO, |acc, bit| acc + bit.value);
            self.row(&[(high.row_terms(), low)], &[], high.value * low_value, what);
        }
        Ok(bits)
    }

    fn select(&mut self, bit: Form, zero: Form, one: Form) -> Form {
        let difference = self.sub(one, zero);
        let chosen = self.mul(bit, difference);
        self.add(zero, chosen)
    }

    fn permute(&mut self, state: [Form; WIDTH]) -> [Form; WIDTH] {
        let outputs = self.block(Kind::Permutation, &state);
        std::array::from_fn(|lane| outputs[lane])
    }

    fn r1cs(&mut self, system: &R1cs, columns: &[Form], what: &'static str) -> Result<(), Error> {
        assert_eq!(columns.len(), system.columns(), "one wire per R1CS column");
        if self.sink.values() {
            let values: Vec<Gl> = columns.iter().map(Form::value).collect();
            if r1cs::fails(system, &values) {
                self.fail(what);
            }
        }
        self.sink.r1cs(system, columns);
        Ok(())
    }

    fn hint(&mut self, inputs: &[Form], len: usize, compute: &dyn Fn(&[u64]) -> Vec<u64>) -> Vec<Form> {
        let values: Vec<u64> = inputs
            .iter()
            .map(|form| form.value.as_canonical_u64())
            .collect();
        let outputs = compute(&values);
        assert_eq!(outputs.len(), len, "a hint must return its declared length");
        outputs
            .into_iter()
            .map(|value| self.cell(Gl::from_u64(value)))
            .collect()
    }
}

/// Every row, cell and block of a run, for the shrink prover and for tests.
#[derive(Default, Debug, PartialEq)]
pub(crate) struct Trace {
    /// Glue cell values.
    pub(crate) glue: Vec<Gl>,
    /// Block cell values per kind, `kind.cells()` per block.
    pub(crate) blocks: [Vec<Gl>; KINDS.len()],
    /// ClaimWords words (index 1 and up; index 0 is the constant one).
    pub(crate) public: Vec<Gl>,
    /// Row entries; `rows[r]` holds the start of each of the seven sides
    /// `A0, B0, A1, B1, A2, B2, C` and the end.
    pub(crate) entries: Vec<Term>,
    pub(crate) rows: Vec<[u32; 8]>,
}

impl Sink for Trace {
    fn row(&mut self, products: &[(Vec<Term>, Vec<Term>)], linear: &[Term]) {
        assert!(products.len() <= 3);
        let mut offsets = [0u32; 8];
        let empty: Vec<Term> = Vec::new();
        for side in 0..6 {
            offsets[side] = self.entries.len() as u32;
            let terms = products
                .get(side / 2)
                .map_or(&empty, |pair| if side % 2 == 0 { &pair.0 } else { &pair.1 });
            self.entries.extend_from_slice(terms);
        }
        offsets[6] = self.entries.len() as u32;
        self.entries.extend_from_slice(linear);
        offsets[7] = self.entries.len() as u32;
        self.rows.push(offsets);
    }

    fn cell(&mut self, value: Gl) {
        self.glue.push(value);
    }

    fn block(&mut self, kind: Kind, cells: &[Gl]) {
        self.blocks[kind.index()].extend_from_slice(cells);
    }

    fn public(&mut self, value: Gl) {
        self.public.push(value);
    }

    /// The default's rows and entries, built in parallel: every row's entries
    /// depend only on `system` and `columns`, so the rows are sized, placed
    /// in order, then filled.
    fn r1cs(&mut self, system: &R1cs, columns: &[Form]) {
        let size = |(side, _): Side<'_>| -> usize {
            side.iter()
                .map(|&column| {
                    let form = &columns[column as usize];
                    form.terms().len() + usize::from(form.constant != Gl::ZERO)
                })
                .sum()
        };
        let sizes: Vec<[usize; 3]> = (0..system.rows())
            .into_par_iter()
            .map(|row| system.sides(row).map(size))
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
            let terms = system
                .sides(row)
                .into_iter()
                .flat_map(|side| expand(side, columns));
            for (slot, term) in out.iter_mut().zip(terms) {
                *slot = term;
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
    /// `Circuit/Digest.lean`): the Poseidon2 leaf sponge (`hash_leaf`) of the
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
        kind.failure(&self.blocks[kind.index()][block * cells..(block + 1) * cells])
    }
}
