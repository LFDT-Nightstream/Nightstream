//! `Export.Generic.ReadOnlyValues`.

use std::collections::{BTreeMap, BTreeSet};

use neo_math::Fq as F;
use p3_field::PrimeCharacteristicRing;

use super::{
    expr::{lower_affine, rank_one_row, Row},
    words::Layout,
    Expr, Geometry,
};
use crate::application::{Affine, Variable};

#[derive(Default)]
pub(super) struct ReadOnlyValues {
    aliases: BTreeMap<Variable, Variable>,
}

impl ReadOnlyValues {
    pub(super) fn checked(rows: &[Row], geometry: &Geometry, layout: &Layout, wires: &[Variable]) -> Self {
        let zero_columns: BTreeSet<_> = rows
            .iter()
            .filter_map(Row::lower_affine)
            .filter_map(|row| match row.terms() {
                [(variable, coefficient)] if row.constant_term() == F::ZERO && *coefficient != F::ZERO => {
                    Some(*variable)
                }
                _ => None,
            })
            .collect();
        let mut aliases = BTreeMap::new();
        for slot in 0..geometry.b_ops {
            if zero_columns.contains(&wires[layout.op(slot, 1)]) {
                for bit in 0..32 {
                    aliases.insert(
                        wires[layout.op(slot, layout.op_vw(bit))],
                        wires[layout.op(slot, layout.op_vr(bit))],
                    );
                }
            }
        }
        Self { aliases }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.aliases.is_empty()
    }

    pub(super) fn wires(&self, wires: &[Variable]) -> Vec<Variable> {
        wires
            .iter()
            .map(|wire| *self.aliases.get(wire).unwrap_or(wire))
            .collect()
    }

    pub(super) fn rename(&self, row: &mut Row) {
        if !self.is_empty() {
            row.rename(&self.aliases);
        }
    }

    pub(super) fn rename_expr(&self, expression: &mut Expr) {
        if !self.is_empty() {
            expression.rename(&self.aliases);
        }
    }
}

fn zero(form: &Affine) -> bool {
    form.constant_term() == F::ZERO && form.terms().is_empty()
}

pub(super) fn repeated_bit(row: &Expr, seen: &mut BTreeSet<Variable>) -> bool {
    let Some((a, b, c)) = rank_one_row(row) else {
        return false;
    };
    let Some(variable) = a.as_variable() else {
        return false;
    };
    zero(&c) && b.constant_term() == F::NEG_ONE && b.terms() == [(variable, F::ONE)] && !seen.insert(variable)
}

pub(super) fn zero_row(row: &Expr) -> bool {
    lower_affine(row).is_some_and(|form| zero(&form))
        || rank_one_row(row).is_some_and(|(a, b, c)| zero(&c) && (zero(&a) || zero(&b)))
}

#[cfg(test)]
#[path = "../../tests/nebula/readonly.rs"]
mod tests;
