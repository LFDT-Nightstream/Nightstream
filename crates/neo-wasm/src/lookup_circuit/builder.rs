//! Small R1CS emitter for the compact operation-table relation.

use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

pub(super) type Bit = usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct LookupR1csRow {
    pub(super) a_terms: Vec<(usize, F)>,
    pub(super) b_terms: Vec<(usize, F)>,
    pub(super) c_terms: Vec<(usize, F)>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct Lc {
    pub(super) terms: Vec<(usize, F)>,
}

impl Lc {
    pub(super) fn zero() -> Self {
        Self::default()
    }

    pub(super) fn one() -> Self {
        Self::var(crate::layout::COL_ONE)
    }

    pub(super) fn var(column: usize) -> Self {
        Self {
            terms: vec![(column, F::ONE)],
        }
    }

    pub(super) fn from_bits(bits: &[Bit]) -> Self {
        let mut coefficient = F::ONE;
        let mut out = Self::zero();
        for &bit in bits {
            out.terms.push((bit, coefficient));
            coefficient += coefficient;
        }
        out
    }

    pub(super) fn add_scaled(mut self, other: &Self, scale: F) -> Self {
        self.terms.extend(
            other
                .terms
                .iter()
                .map(|&(column, coefficient)| (column, coefficient * scale)),
        );
        self
    }

    pub(super) fn scaled(mut self, scale: F) -> Self {
        for (_, coefficient) in &mut self.terms {
            *coefficient *= scale;
        }
        self
    }

    pub(super) fn plus(self, other: &Self) -> Self {
        self.add_scaled(other, F::ONE)
    }

    pub(super) fn minus(self, other: &Self) -> Self {
        self.add_scaled(other, -F::ONE)
    }
}

pub(super) struct LookupR1csBuilder {
    values: Vec<F>,
    aux_start: usize,
    widths: Vec<usize>,
    rows: Vec<LookupR1csRow>,
}

impl LookupR1csBuilder {
    pub(super) fn new(base_assignment: &[F]) -> Self {
        Self {
            values: base_assignment.to_vec(),
            widths: Vec::new(),
            aux_start: base_assignment.len(),
            rows: Vec::new(),
        }
    }

    pub(super) fn value_bit(&self, column: usize) -> Result<bool, String> {
        let value = self
            .values
            .get(column)
            .ok_or_else(|| format!("lookup R1CS references missing column {column}"))?
            .as_canonical_u64();
        match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(format!("lookup R1CS expected Boolean column {column}, got {value}")),
        }
    }

    pub(super) fn eval_bit(&self, lc: &Lc) -> Result<bool, String> {
        let value = lc
            .terms
            .iter()
            .fold(F::ZERO, |sum, &(column, coefficient)| {
                sum + self.values[column] * coefficient
            });
        match value.as_canonical_u64() {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(format!("lookup R1CS expected Boolean linear form, got {value}")),
        }
    }

    pub(super) fn eval(&self, lc: &Lc) -> F {
        lc.terms
            .iter()
            .fold(F::ZERO, |sum, &(column, coefficient)| {
                sum + self.values[column] * coefficient
            })
    }

    /// Allocate `left * right` of two Boolean forms with one row.
    pub(super) fn alloc_product(&mut self, left: &Lc, right: &Lc) -> usize {
        self.alloc_product_with_width(left, right, 1)
    }

    /// Allocate the product of two 16-bit limbs, a 32-bit value.
    pub(super) fn alloc_limb_product(&mut self, left: &Lc, right: &Lc) -> usize {
        self.alloc_product_with_width(left, right, 32)
    }

    fn alloc_product_with_width(&mut self, left: &Lc, right: &Lc, width: usize) -> usize {
        let column = self.values.len();
        self.values.push(self.eval(left) * self.eval(right));
        self.widths.push(width);
        self.push_row(left.clone(), right.clone(), Lc::var(column));
        column
    }

    /// Allocate `select ? when_true : when_false` with one row. Boolean inputs
    /// give a Boolean output.
    pub(super) fn alloc_select(&mut self, select: &Lc, when_true: &Lc, when_false: &Lc) -> usize {
        let column = self.values.len();
        let value = self.eval(when_false) + self.eval(select) * (self.eval(when_true) - self.eval(when_false));
        self.values.push(value);
        self.widths.push(1);
        self.push_row(
            select.clone(),
            when_true.clone().minus(when_false),
            Lc::var(column).minus(when_false),
        );
        column
    }

    /// Packed equality of two Boolean vectors: one row per 32 bits.
    pub(super) fn enforce_bits_equal_when(&mut self, gate: &Lc, left: &[Bit], right: &[Bit]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.chunks(32).zip(right.chunks(32)) {
            self.enforce_equal_when(gate, &Lc::from_bits(left), &Lc::from_bits(right));
        }
    }

    /// Packed zero test of a Boolean vector: one row per 32 bits.
    pub(super) fn enforce_bits_zero_when(&mut self, gate: &Lc, bits: &[Bit]) {
        for chunk in bits.chunks(32) {
            self.enforce_zero_when(gate, &Lc::from_bits(chunk));
        }
    }

    pub(super) fn alloc_bit(&mut self, value: bool) -> Bit {
        let column = self.values.len();
        self.values.push(if value { F::ONE } else { F::ZERO });
        self.widths.push(1);
        self.push_row(Lc::var(column), Lc::var(column).minus(&Lc::one()), Lc::zero());
        column
    }

    pub(super) fn alloc_and(&mut self, left: &Lc, right: &Lc) -> Result<Bit, String> {
        let value = self.eval_bit(left)? & self.eval_bit(right)?;
        let output = self.alloc_bit(value);
        self.push_row(left.clone(), right.clone(), Lc::var(output));
        Ok(output)
    }

    pub(super) fn alloc_xor(&mut self, left: &Lc, right: &Lc) -> Result<Bit, String> {
        let value = self.eval_bit(left)? ^ self.eval_bit(right)?;
        let output = self.alloc_bit(value);
        let numerator = left.clone().plus(right).minus(&Lc::var(output));
        self.push_row(left.clone(), right.clone(), numerator.scaled(F::from_u64(2).inverse()));
        Ok(output)
    }

    pub(super) fn enforce_equal_when(&mut self, gate: &Lc, left: &Lc, right: &Lc) {
        self.push_row(gate.clone(), left.clone().minus(right), Lc::zero());
    }

    pub(super) fn enforce_zero_when(&mut self, gate: &Lc, value: &Lc) {
        self.push_row(gate.clone(), value.clone(), Lc::zero());
    }

    pub(super) fn enforce_linear_zero(&mut self, value: Lc) {
        self.push_row(value, Lc::one(), Lc::zero());
    }

    /// Rows, auxiliary values, and the declared bit width of each auxiliary.
    pub(super) fn finish(self) -> (Vec<LookupR1csRow>, Vec<F>, Vec<usize>) {
        (self.rows, self.values[self.aux_start..].to_vec(), self.widths)
    }

    fn push_row(&mut self, left: Lc, right: Lc, output: Lc) {
        self.rows.push(LookupR1csRow {
            a_terms: left.terms,
            b_terms: right.terms,
            c_terms: output.terms,
        });
    }
}

pub(super) fn selector_lc(selectors: &[usize]) -> Lc {
    Lc {
        terms: selectors.iter().map(|&column| (column, F::ONE)).collect(),
    }
}
