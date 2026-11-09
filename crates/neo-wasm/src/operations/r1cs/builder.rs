//! Fixed operation-circuit construction; assignment follows the allocated schedule.

use crate::operations::r1cs::OperationRelation;
use crate::operations::OPERATIONS_REGION;
use crate::tagged_r1cs_builder::{WasmConstraintScope, WasmConstraintTag};
use neo_application::{AuxiliaryBuilder, AuxiliaryLayout, ColumnWidth, ConstraintTag, R1csBuilder};
pub(super) use neo_application::{ColumnId, LinearExpression as Lc};
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

pub(super) struct CircuitBuilder {
    allocation: AuxiliaryBuilder,
    rows: R1csBuilder<WasmConstraintScope>,
    tag: WasmConstraintTag,
    scope: &'static str,
    next_family: usize,
}

impl CircuitBuilder {
    pub(super) fn new(base: &OperationRelation) -> Self {
        Self {
            allocation: AuxiliaryBuilder::new(base.columns()),
            rows: R1csBuilder::from_relation(base.r1cs()),
            tag: ConstraintTag::new(OPERATIONS_REGION, WasmConstraintScope::Always),
            scope: OPERATIONS_REGION,
            next_family: 0,
        }
    }

    pub(super) fn section(&mut self, name: &'static str, operations: Vec<crate::WasmOpcode>) {
        self.scope = name;
        self.next_family = 0;
        self.tag = ConstraintTag::new(name, WasmConstraintScope::Opcodes(operations.into_boxed_slice()));
    }

    pub(super) fn position(&mut self) -> (usize, usize) {
        (
            self.allocation.column_count(),
            self.rows.tagged(self.tag.clone()).next_row_index(),
        )
    }

    pub(super) fn one(&self) -> Lc {
        Lc::var(self.rows.const_one_column())
    }

    fn allocate(
        &mut self,
        role: &'static str,
        width: ColumnWidth,
        len: usize,
        assign: impl Fn(&[F], &mut [F]) -> Result<(), String> + Send + Sync + 'static,
    ) -> Vec<ColumnId> {
        let name = format!("{OPERATIONS_REGION}.{}.{}[{}]", self.scope, role, self.next_family);
        self.next_family += 1;
        self.allocation
            .allocate(OPERATIONS_REGION, name, role, width, len, assign)
            .expect("operation family has a bounded, nonzero size")
            .map(ColumnId)
            .collect()
    }

    pub(super) fn alloc_bits(
        &mut self,
        role: &'static str,
        width: usize,
        value: impl Fn(&[F]) -> Result<u64, String> + Send + Sync + 'static,
    ) -> Vec<ColumnId> {
        assert!((1..=64).contains(&width));
        let columns = self.allocate(role, ColumnWidth::Boolean, width, move |input, output| {
            let value = value(input)?;
            for (bit, output) in output.iter_mut().enumerate() {
                *output = F::from_u64((value >> bit) & 1);
            }
            Ok(())
        });
        for &column in &columns {
            self.push_row(Lc::var(column), Lc::var(column).minus(&self.one()), Lc::zero());
        }
        columns
    }

    pub(super) fn alloc_bit(
        &mut self,
        role: &'static str,
        value: impl Fn(&[F]) -> Result<bool, String> + Send + Sync + 'static,
    ) -> ColumnId {
        self.alloc_bits(role, 1, move |z| value(z).map(u64::from))[0]
    }

    pub(super) fn alloc_product(&mut self, left: &Lc, right: &Lc) -> ColumnId {
        self.product(left, right, ColumnWidth::Boolean)
    }

    pub(super) fn alloc_limb_product(&mut self, left: &Lc, right: &Lc) -> ColumnId {
        self.product(left, right, ColumnWidth::U32)
    }

    fn product(&mut self, left: &Lc, right: &Lc, width: ColumnWidth) -> ColumnId {
        let (l, r) = (left.clone(), right.clone());
        let column = self.allocate("product", width, 1, move |z, output| {
            output[0] = l.evaluate(z) * r.evaluate(z);
            Ok(())
        })[0];
        self.push_row(left.clone(), right.clone(), Lc::var(column));
        column
    }

    pub(super) fn alloc_select(&mut self, select: &Lc, when_true: &Lc, when_false: &Lc) -> ColumnId {
        let (s, t, f) = (select.clone(), when_true.clone(), when_false.clone());
        let column = self.allocate("select", ColumnWidth::Boolean, 1, move |z, output| {
            output[0] = f.evaluate(z) + s.evaluate(z) * (t.evaluate(z) - f.evaluate(z));
            Ok(())
        })[0];
        self.push_row(
            select.clone(),
            when_true.clone().minus(when_false),
            Lc::var(column).minus(when_false),
        );
        column
    }

    pub(super) fn alloc_and(&mut self, left: &Lc, right: &Lc) -> Result<ColumnId, String> {
        let (l, r) = (left.clone(), right.clone());
        let column = self.alloc_bit("and", move |z| Ok(eval_bit(z, &l)? & eval_bit(z, &r)?));
        self.push_row(left.clone(), right.clone(), Lc::var(column));
        Ok(column)
    }

    pub(super) fn alloc_xor(&mut self, left: &Lc, right: &Lc) -> Result<ColumnId, String> {
        let (l, r) = (left.clone(), right.clone());
        let column = self.alloc_bit("xor", move |z| Ok(eval_bit(z, &l)? ^ eval_bit(z, &r)?));
        self.push_row(
            left.clone(),
            right.clone(),
            left.clone()
                .plus(right)
                .minus(&Lc::var(column))
                .scaled(F::from_u64(2).inverse()),
        );
        Ok(column)
    }

    pub(super) fn enforce_bits_equal_when(&mut self, gate: &Lc, left: &[ColumnId], right: &[ColumnId]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.chunks(32).zip(right.chunks(32)) {
            self.enforce_equal_when(gate, &Lc::from_bits(left), &Lc::from_bits(right));
        }
    }

    pub(super) fn enforce_bits_zero_when(&mut self, gate: &Lc, bits: &[ColumnId]) {
        for chunk in bits.chunks(32) {
            self.enforce_zero_when(gate, &Lc::from_bits(chunk));
        }
    }

    pub(super) fn enforce_equal_when(&mut self, gate: &Lc, left: &Lc, right: &Lc) {
        self.push_row(gate.clone(), left.clone().minus(right), Lc::zero());
    }
    pub(super) fn enforce_zero_when(&mut self, gate: &Lc, value: &Lc) {
        self.push_row(gate.clone(), value.clone(), Lc::zero());
    }
    pub(super) fn enforce_linear_zero(&mut self, value: Lc) {
        self.push_row(value, self.one(), Lc::zero());
    }
    fn push_row(&mut self, left: Lc, right: Lc, output: Lc) {
        self.rows
            .tagged(self.tag.clone())
            .push_row(left.terms, right.terms, output.terms);
    }
    pub(super) fn finish(self) -> Result<(OperationRelation, AuxiliaryLayout), String> {
        let layout = self.allocation.finish().map_err(|e| e.to_string())?;
        let relation = self
            .rows
            .build(layout.columns())
            .map_err(|e| e.to_string())?;
        let relation = OperationRelation::new(relation, layout.columns().clone()).map_err(|e| e.to_string())?;
        Ok((relation, layout))
    }
}

pub(super) fn selector_lc(selectors: &[ColumnId]) -> Lc {
    Lc {
        terms: selectors.iter().map(|&column| (column.0, F::ONE)).collect(),
    }
}

pub(super) fn eval_bit(z: &[F], lc: &Lc) -> Result<bool, String> {
    match lc.evaluate(z).as_canonical_u64() {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(format!("operation expected Boolean expression, got {value}")),
    }
}

pub(super) fn bits_u64(z: &[F], bits: &[ColumnId]) -> Result<u64, String> {
    bits.iter()
        .enumerate()
        .try_fold(0, |value, (bit, &column)| {
            Ok(value | (u64::from(eval_bit(z, &Lc::var(column))?) << bit))
        })
}
