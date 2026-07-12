//! Small R1CS emitter for the compact operation-table relation.

use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use crate::tagged_r1cs_builder::WasmR1csRow;

pub(super) type Bit = usize;

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
    rows: Vec<WasmR1csRow>,
}

impl LookupR1csBuilder {
    pub(super) fn new(base_assignment: &[F]) -> Self {
        Self {
            values: base_assignment.to_vec(),
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

    pub(super) fn alloc_bit(&mut self, value: bool) -> Bit {
        let column = self.values.len();
        self.values.push(if value { F::ONE } else { F::ZERO });
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

    pub(super) fn alloc_mux(&mut self, select: &Lc, when_true: &Lc, when_false: &Lc) -> Result<Bit, String> {
        let value = if self.eval_bit(select)? {
            self.eval_bit(when_true)?
        } else {
            self.eval_bit(when_false)?
        };
        let output = self.alloc_bit(value);
        self.push_row(
            select.clone(),
            when_true.clone().minus(when_false),
            Lc::var(output).minus(when_false),
        );
        Ok(output)
    }

    pub(super) fn mask_bit(&mut self, gate: &Lc, source: Bit) -> Result<Bit, String> {
        self.alloc_and(gate, &Lc::var(source))
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

    pub(super) fn enforce_product(&mut self, left: Lc, right: Lc, output: Lc) {
        self.push_row(left, right, output);
    }

    pub(super) fn finish(self) -> (Vec<WasmR1csRow>, Vec<F>) {
        (self.rows, self.values[self.aux_start..].to_vec())
    }

    fn push_row(&mut self, left: Lc, right: Lc, output: Lc) {
        self.rows.push(WasmR1csRow {
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
