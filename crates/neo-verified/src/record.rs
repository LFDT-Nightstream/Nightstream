//! The constraint reading of `Backend`.
//!
//! Owns: forms (short linear combinations with their value), wires, the glue rows, the
//! permutation blocks, and the first failed check. Does not own where rows and cells sit in
//! the shrink cube, or what a sink does with them.
//!
//! A glue row is `Σ_{i<3} (A_i·z)·(B_i·z) − C·z = 0`. A permutation is one block (block.rs)
//! whose rows its template fixes; its inputs are tied to the caller's forms by one linear
//! glue row each. A fixed R1CS system (r1cs.rs) is one glue row per row. The value of every
//! row is computed when it is emitted, so the first failed check is known without stopping:
//! the rows never depend on the values.

use crate::backend::{Backend, Hint, Rejected};
use crate::block::OUTPUT;
use crate::circuit::{Kind, Wire, CELLS, KINDS, WIDTH};
use crate::field::{
    ext_inv, ext_mul, gl_add, gl_mul, gl_neg, gl_sub, gl_word, reduce, Ext, Gl, EXT_ZERO, GL_ONE, GL_ZERO,
};
use crate::native::{ext_is_zero, hint_words};
use crate::r1cs::{fails, R1cs};

pub const ONE: Wire = Wire::Public(0);
pub type Term = (Wire, Gl);

/// Terms a form carries before it becomes a cell.
const FORM_TERMS: usize = 4;

/// `Σ coefficient·wire + constant`, and its value.
#[derive(Clone, Copy)]
pub struct Form {
    terms: [Term; FORM_TERMS],
    len: u8,
    constant: Gl,
    value: Gl,
}

impl Form {
    fn of_constant(value: Gl) -> Self {
        Self {
            terms: [(ONE, GL_ZERO); FORM_TERMS],
            len: 0,
            constant: value,
            value,
        }
    }

    fn of_wire(wire: Wire, value: Gl) -> Self {
        let mut form = Self::of_constant(GL_ZERO);
        form.terms[0] = (wire, GL_ONE);
        form.len = 1;
        form.value = value;
        form
    }

    pub fn value(&self) -> Gl {
        self.value
    }

    /// The constant part.
    pub fn offset(&self) -> Gl {
        self.constant
    }

    pub fn term_count(&self) -> usize {
        self.len as usize
    }

    /// Term `k`, for `k` below `term_count`.
    pub fn term(&self, k: usize) -> Term {
        self.terms[k]
    }

    pub fn is_constant(&self) -> bool {
        self.len == 0
    }

    /// The terms with the constant as a term on the one column.
    fn row_terms(&self) -> Vec<Term> {
        let mut terms = self.wire_terms();
        if self.constant != GL_ZERO {
            terms.push((ONE, self.constant));
        }
        terms
    }

    fn wire_terms(&self) -> Vec<Term> {
        let mut terms = Vec::new();
        let mut k = 0;
        while k < self.term_count() {
            terms.push(self.terms[k]);
            k = k.wrapping_add(1);
        }
        terms
    }
}

/// `terms` with `coefficient·wire` added to the term of `wire`, or appended.
fn add_term(terms: &mut Vec<Term>, wire: Wire, coefficient: Gl) {
    let mut found = terms.len();
    let mut k = 0;
    while k < terms.len() {
        if found == terms.len() && terms[k].0 == wire {
            found = k;
        }
        k = k.wrapping_add(1);
    }
    if found < terms.len() {
        terms[found].1 = gl_add(terms[found].1, coefficient);
    } else {
        terms.push((wire, coefficient));
    }
}

/// `Σ weight·form` as a merged term list without zero coefficients, a constant and a value.
fn combine(parts: &[(Form, Gl)]) -> (Vec<Term>, Gl, Gl) {
    let mut merged = Vec::new();
    let mut constant = GL_ZERO;
    let mut value = GL_ZERO;
    let mut p = 0;
    while p < parts.len() {
        let (form, weight) = parts[p];
        constant = gl_add(constant, gl_mul(form.constant, weight));
        value = gl_add(value, gl_mul(form.value, weight));
        let mut k = 0;
        while k < form.term_count() {
            let (wire, coefficient) = form.terms[k];
            add_term(&mut merged, wire, gl_mul(coefficient, weight));
            k = k.wrapping_add(1);
        }
        p = p.wrapping_add(1);
    }
    let mut terms = Vec::new();
    let mut k = 0;
    while k < merged.len() {
        if merged[k].1 != GL_ZERO {
            terms.push(merged[k]);
        }
        k = k.wrapping_add(1);
    }
    (terms, constant, value)
}

/// `Σ coefficient·columns[column]` over the terms `from..to` of `system`, as row terms, the
/// constants on the one column. Repeated wires stay separate terms; a row sums them.
pub fn expand(system: &R1cs, from: usize, to: usize, columns: &[Form]) -> Vec<Term> {
    let column_of = system.term_columns();
    let coefficient_of = system.term_coefficients();
    let mut terms = Vec::new();
    let mut k = from;
    while k < to {
        let form = columns[column_of[k] as usize];
        let coefficient = coefficient_of[k];
        let mut t = 0;
        while t < form.term_count() {
            let (wire, c) = form.terms[t];
            terms.push((wire, gl_mul(c, coefficient)));
            t = t.wrapping_add(1);
        }
        if form.constant != GL_ZERO {
            terms.push((ONE, gl_mul(form.constant, coefficient)));
        }
        k = k.wrapping_add(1);
    }
    terms
}

/// Where recorded rows, cells and blocks go.
pub trait Sink {
    /// One glue row: `Σ (Σ a)·(Σ b) − Σ c`, with each side's terms.
    fn row(&mut self, products: &[(Vec<Term>, Vec<Term>)], linear: &[Term]);
    fn cell(&mut self, value: Gl);
    /// One block of `kind` on the input values `inputs` (empty when the sink takes no
    /// values). Returns the values of its outputs, in order, or none.
    fn block(&mut self, kind: Kind, inputs: &[Gl]) -> Vec<Gl>;
    fn public(&mut self, value: Gl);
    /// The rows of `system` as glue rows, in order, with `columns[k]` as its column `k`.
    fn r1cs(&mut self, system: &R1cs, columns: &[Form]) {
        let mut row = 0;
        while row < system.rows() {
            let (start, ends) = system.bounds(row);
            let a = expand(system, start, ends[0], columns);
            let b = expand(system, ends[0], ends[1], columns);
            let c = expand(system, ends[1], ends[2], columns);
            self.row(&[(a, b)], &c);
            row = row.wrapping_add(1);
        }
    }
    /// Whether the sink reads values. A shape run that only counts or evaluates rows does
    /// not, so blocks and R1CS rows are not computed.
    fn values(&self) -> bool {
        true
    }
}

/// The constraint reading of a verifier run.
pub struct Recorder<S: Sink> {
    sink: S,
    glue: u32,
    blocks: [u32; KINDS.len()],
    publics: u32,
    failure: Option<&'static str>,
}

/// The value of a cubic-extension form.
fn ext_value(a: &[Form; 3]) -> Ext {
    Ext {
        c0: a[0].value,
        c1: a[1].value,
        c2: a[2].value,
    }
}

fn ext_coordinates(value: Ext) -> [Gl; 3] {
    [value.c0, value.c1, value.c2]
}

fn all_constant(a: &[Form; 3]) -> bool {
    a[0].is_constant() && a[1].is_constant() && a[2].is_constant()
}

/// `Σ forms` as row terms, the constant on the one column.
fn sum_terms(forms: &[Form]) -> Vec<Term> {
    let mut parts = Vec::new();
    let mut k = 0;
    while k < forms.len() {
        parts.push((forms[k], GL_ONE));
        k = k.wrapping_add(1);
    }
    let (mut terms, constant, _) = combine(&parts);
    if constant != GL_ZERO {
        terms.push((ONE, constant));
    }
    terms
}

/// Row `k` of a cubic-extension product `a·b`: `Σ_i a_i·L_ik(b)` with the structure constants
/// of `x^3 = x + 1`.
fn ext_rows(a: &[Form; 3], b: &[Form; 3]) -> [Vec<(Vec<Term>, Vec<Term>)>; 3] {
    let row0 = Vec::from([
        (a[0].row_terms(), sum_terms(&[b[0]])),
        (a[1].row_terms(), sum_terms(&[b[2]])),
        (a[2].row_terms(), sum_terms(&[b[1]])),
    ]);
    let row1 = Vec::from([
        (a[0].row_terms(), sum_terms(&[b[1]])),
        (a[1].row_terms(), sum_terms(&[b[0], b[2]])),
        (a[2].row_terms(), sum_terms(&[b[1], b[2]])),
    ]);
    let row2 = Vec::from([
        (a[0].row_terms(), sum_terms(&[b[2]])),
        (a[1].row_terms(), sum_terms(&[b[1]])),
        (a[2].row_terms(), sum_terms(&[b[0], b[2]])),
    ]);
    [row0, row1, row2]
}

impl<S: Sink> Recorder<S> {
    pub fn new(sink: S) -> Self {
        Self {
            sink,
            glue: 0,
            blocks: [0; KINDS.len()],
            publics: 1,
            failure: None,
        }
    }

    /// The sink and the first failed check, if any.
    pub fn finish(self) -> (S, Option<&'static str>) {
        (self.sink, self.failure)
    }

    fn fail(&mut self, what: &'static str) {
        if self.failure.is_none() {
            self.failure = Some(what);
        }
    }

    fn cell(&mut self, value: Gl) -> Form {
        let wire = Wire::Glue(self.glue);
        self.glue = self.glue.wrapping_add(1);
        self.sink.cell(value);
        Form::of_wire(wire, value)
    }

    /// Emit a row whose value under the witness is `value`.
    fn row(&mut self, products: &[(Vec<Term>, Vec<Term>)], linear: &[Term], value: Gl, what: &'static str) {
        if value != GL_ZERO {
            self.fail(what);
        }
        self.sink.row(products, linear);
    }

    /// A form from merged terms; a new cell and one linear row if too long.
    fn form(&mut self, terms: Vec<Term>, constant: Gl, value: Gl) -> Form {
        if terms.len() <= FORM_TERMS {
            let mut form = Form::of_constant(constant);
            let mut k = 0;
            while k < terms.len() {
                form.terms[k] = terms[k];
                k = k.wrapping_add(1);
            }
            form.len = terms.len() as u8;
            form.value = value;
            form
        } else {
            let cell = self.cell(value);
            let mut linear = Vec::new();
            linear.push((cell.terms[0].0, GL_ONE));
            let mut k = 0;
            while k < terms.len() {
                let (wire, coefficient) = terms[k];
                linear.push((wire, gl_neg(coefficient)));
                k = k.wrapping_add(1);
            }
            if constant != GL_ZERO {
                linear.push((ONE, gl_neg(constant)));
            }
            self.sink.row(&[], &linear);
            cell
        }
    }

    /// One block of `kind` on `inputs`; returns its output cells.
    fn block(&mut self, kind: Kind, inputs: &[Form]) -> Vec<Form> {
        let index = self.blocks[kind.index()];
        self.blocks[kind.index()] = index.wrapping_add(1);
        let mut values = Vec::new();
        if self.sink.values() {
            let mut k = 0;
            while k < inputs.len() {
                values.push(inputs[k].value);
                k = k.wrapping_add(1);
            }
        }
        let outputs = self.sink.block(kind, &values);
        let mut cell = 0;
        while cell < inputs.len() {
            let mut linear = Vec::new();
            linear.push((Wire::Block(kind, index, cell as u16), GL_ONE));
            let terms = inputs[cell].row_terms();
            let mut k = 0;
            while k < terms.len() {
                let (wire, c) = terms[k];
                linear.push((wire, gl_neg(c)));
                k = k.wrapping_add(1);
            }
            self.sink.row(&[], &linear);
            cell = cell.wrapping_add(1);
        }
        let mut forms = Vec::new();
        let mut cell = OUTPUT;
        while cell < CELLS {
            let at = cell.wrapping_sub(OUTPUT);
            let value = if at < outputs.len() { outputs[at] } else { GL_ZERO };
            forms.push(Form::of_wire(Wire::Block(kind, index, cell as u16), value));
            cell = cell.wrapping_add(1);
        }
        forms
    }

    fn linear(&mut self, parts: &[(Form, Gl)]) -> Form {
        let (terms, constant, value) = combine(parts);
        self.form(terms, constant, value)
    }

    /// Multiplication by the constant `c` as three linear forms of `other`.
    fn ext_scale_by(&mut self, c: Ext, other: [Form; 3]) -> [Form; 3] {
        let columns = [
            ext_coordinates(ext_mul(
                c,
                Ext {
                    c0: GL_ONE,
                    c1: GL_ZERO,
                    c2: GL_ZERO,
                },
            )),
            ext_coordinates(ext_mul(
                c,
                Ext {
                    c0: GL_ZERO,
                    c1: GL_ONE,
                    c2: GL_ZERO,
                },
            )),
            ext_coordinates(ext_mul(
                c,
                Ext {
                    c0: GL_ZERO,
                    c1: GL_ZERO,
                    c2: GL_ONE,
                },
            )),
        ];
        let x = self.linear(&[
            (other[0], columns[0][0]),
            (other[1], columns[1][0]),
            (other[2], columns[2][0]),
        ]);
        let y = self.linear(&[
            (other[0], columns[0][1]),
            (other[1], columns[1][1]),
            (other[2], columns[2][1]),
        ]);
        let z = self.linear(&[
            (other[0], columns[0][2]),
            (other[1], columns[1][2]),
            (other[2], columns[2][2]),
        ]);
        [x, y, z]
    }

    fn ext_cell(&mut self, row: &[(Vec<Term>, Vec<Term>)], value: Gl) -> Form {
        let cell = self.cell(value);
        self.sink.row(row, &cell.wire_terms());
        cell
    }
}

impl<S: Sink> Backend for Recorder<S> {
    type F = Form;
    type E = [Form; 3];

    fn constant(&mut self, value: u64) -> Form {
        Form::of_constant(reduce(value))
    }

    fn private(&mut self, value: u64) -> Form {
        self.cell(reduce(value))
    }

    fn public(&mut self, value: u64) -> Form {
        let value = reduce(value);
        let wire = Wire::Public(self.publics);
        self.publics = self.publics.wrapping_add(1);
        self.sink.public(value);
        Form::of_wire(wire, value)
    }

    fn add(&mut self, a: Form, b: Form) -> Form {
        self.linear(&[(a, GL_ONE), (b, GL_ONE)])
    }

    fn sub(&mut self, a: Form, b: Form) -> Form {
        self.linear(&[(a, GL_ONE), (b, gl_neg(GL_ONE))])
    }

    fn scale(&mut self, a: Form, by: u64) -> Form {
        self.linear(&[(a, reduce(by))])
    }

    fn mul(&mut self, a: Form, b: Form) -> Form {
        if a.is_constant() {
            self.linear(&[(b, a.constant)])
        } else if b.is_constant() {
            self.linear(&[(a, b.constant)])
        } else {
            let product = self.cell(gl_mul(a.value, b.value));
            self.sink
                .row(&[(a.row_terms(), b.row_terms())], &product.wire_terms());
            product
        }
    }

    fn ext(&mut self, coordinates: [Form; 3]) -> [Form; 3] {
        coordinates
    }

    fn coordinates(&mut self, a: [Form; 3]) -> [Form; 3] {
        a
    }

    fn ext_constant(&mut self, value: [u64; 3]) -> [Form; 3] {
        [
            Form::of_constant(reduce(value[0])),
            Form::of_constant(reduce(value[1])),
            Form::of_constant(reduce(value[2])),
        ]
    }

    fn ext_add(&mut self, a: [Form; 3], b: [Form; 3]) -> [Form; 3] {
        let x = self.add(a[0], b[0]);
        let y = self.add(a[1], b[1]);
        let z = self.add(a[2], b[2]);
        [x, y, z]
    }

    fn ext_sub(&mut self, a: [Form; 3], b: [Form; 3]) -> [Form; 3] {
        let x = self.sub(a[0], b[0]);
        let y = self.sub(a[1], b[1]);
        let z = self.sub(a[2], b[2]);
        [x, y, z]
    }

    fn ext_mul(&mut self, a: [Form; 3], b: [Form; 3]) -> [Form; 3] {
        let product = ext_coordinates(ext_mul(ext_value(&a), ext_value(&b)));
        if all_constant(&a) {
            self.ext_scale_by(
                Ext {
                    c0: a[0].constant,
                    c1: a[1].constant,
                    c2: a[2].constant,
                },
                b,
            )
        } else if all_constant(&b) {
            self.ext_scale_by(
                Ext {
                    c0: b[0].constant,
                    c1: b[1].constant,
                    c2: b[2].constant,
                },
                a,
            )
        } else {
            let rows = ext_rows(&a, &b);
            let x = self.ext_cell(&rows[0], product[0]);
            let y = self.ext_cell(&rows[1], product[1]);
            let z = self.ext_cell(&rows[2], product[2]);
            [x, y, z]
        }
    }

    fn ext_scale(&mut self, a: [Form; 3], by: Form) -> [Form; 3] {
        let x = self.mul(a[0], by);
        let y = self.mul(a[1], by);
        let z = self.mul(a[2], by);
        [x, y, z]
    }

    fn ext_inverse(&mut self, a: [Form; 3], what: &'static str) -> Result<[Form; 3], Rejected> {
        let value = ext_value(&a);
        let zero = ext_is_zero(value);
        let inverse_value = if zero { EXT_ZERO } else { ext_inv(value) };
        if all_constant(&a) && !zero {
            let c = ext_coordinates(inverse_value);
            Ok([
                Form::of_constant(c[0]),
                Form::of_constant(c[1]),
                Form::of_constant(c[2]),
            ])
        } else {
            // A constant zero falls through: its rows cannot hold, so the shrink proves the
            // failure too.
            let c = ext_coordinates(inverse_value);
            let x = self.cell(c[0]);
            let y = self.cell(c[1]);
            let z = self.cell(c[2]);
            let inverse = [x, y, z];
            let rows = ext_rows(&a, &inverse);
            let product = ext_coordinates(ext_mul(value, ext_value(&inverse)));
            self.row(&rows[0], &[(ONE, GL_ONE)], gl_sub(product[0], GL_ONE), what);
            self.row(&rows[1], &[], product[1], what);
            self.row(&rows[2], &[], product[2], what);
            Ok(inverse)
        }
    }

    fn assert_zero(&mut self, a: Form, what: &'static str) -> Result<(), Rejected> {
        // A nonzero constant still gets its row, `c·1 = 0`, which cannot hold, so the shrink
        // proves the failure too.
        if !(a.is_constant() && a.constant == GL_ZERO) {
            self.row(&[], &a.row_terms(), a.value, what);
        }
        Ok(())
    }

    fn bits(&mut self, a: Form, count: usize, what: &'static str) -> Result<Vec<Form>, Rejected> {
        let value = gl_word(a.value);
        let mut bits = Vec::new();
        let mut i = 0;
        while i < count {
            let bit = if i < 64 { value.wrapping_shr(i as u32) & 1 } else { 0 };
            bits.push(self.cell(reduce(bit)));
            i = i.wrapping_add(1);
        }
        let mut i = 0;
        while i < bits.len() {
            let wire = bits[i].wire_terms();
            self.sink
                .row(&[(bits[i].wire_terms(), bits[i].wire_terms())], &wire);
            i = i.wrapping_add(1);
        }
        let mut linear = a.row_terms();
        let mut sum = GL_ZERO;
        let mut i = 0;
        while i < bits.len() {
            let weight = reduce(1u64.wrapping_shl(i as u32));
            linear.push((bits[i].terms[0].0, gl_neg(weight)));
            sum = gl_add(sum, gl_mul(weight, bits[i].value));
            i = i.wrapping_add(1);
        }
        self.row(&[], &linear, gl_sub(a.value, sum), what);
        if count == 64 {
            // Canonical: the high 32 bits all one forces the low 32 bits to zero.
            let mut high = bits[32];
            let mut i = 33;
            while i < 64 {
                high = self.mul(high, bits[i]);
                i = i.wrapping_add(1);
            }
            let mut low = Vec::new();
            let mut low_value = GL_ZERO;
            let mut i = 0;
            while i < 32 {
                low.push(bits[i].terms[0]);
                low_value = gl_add(low_value, bits[i].value);
                i = i.wrapping_add(1);
            }
            self.row(&[(high.row_terms(), low)], &[], gl_mul(high.value, low_value), what);
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
        let mut lanes = state;
        let mut lane = 0;
        while lane < WIDTH {
            lanes[lane] = outputs[lane];
            lane = lane.wrapping_add(1);
        }
        lanes
    }

    fn hint(&mut self, inputs: &[Form], len: usize, hint: &Hint) -> Vec<Form> {
        let mut values = Vec::new();
        let mut k = 0;
        while k < inputs.len() {
            values.push(gl_word(inputs[k].value));
            k = k.wrapping_add(1);
        }
        let words = hint_words(hint, &values);
        let mut outputs = Vec::new();
        let mut k = 0;
        while k < len {
            let word = if k < words.len() { words[k] } else { 0 };
            outputs.push(self.cell(reduce(word)));
            k = k.wrapping_add(1);
        }
        outputs
    }

    fn r1cs(&mut self, system: &R1cs, columns: &[Form], what: &'static str) -> Result<(), Rejected> {
        assert!(columns.len() == system.columns());
        if self.sink.values() {
            let mut values = Vec::new();
            let mut k = 0;
            while k < columns.len() {
                values.push(columns[k].value);
                k = k.wrapping_add(1);
            }
            if fails(system, &values) {
                self.fail(what);
            }
        }
        self.sink.r1cs(system, columns);
        Ok(())
    }
}
