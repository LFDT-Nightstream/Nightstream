//! Checked relocation of application-private Poseidon2 traces into native blocks.
//! Source rows remain authority. The map changes only the described aux words.

use std::{collections::BTreeMap, ops::ControlFlow, rc::Rc};

use neo_ccs::crypto::poseidon2_goldilocks::{round_constants, WIDTH};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::{
    expr::{lower_affine, lower_constraint, rank_one_row},
    Expr, View,
};
use crate::application::{permutation, Affine, ApplicationBuilder, ApplicationCircuit, ApplicationError, Variable};

const WORDS: usize = 1096;

/// An untrusted hint for 1,096 rows of the form `c - a*b`, and 1,096
/// contiguous private words in the canonical Poseidon2 trace order.
#[derive(Clone, Debug)]
pub struct Poseidon2Hint {
    pub row_start: usize,
    pub aux_start: usize,
    pub lanes: [Expr; WIDTH],
}

type Aliases = BTreeMap<Variable, Variable>;

fn invalid() -> ApplicationError {
    ApplicationError::Prepared("Nebula Poseidon2 hint")
}

fn variable(expression: Expr) -> Result<Variable, ApplicationError> {
    match expression {
        Expr::Var(variable) => Ok(variable),
        _ => Err(invalid()),
    }
}

fn substitute(expression: &Expr, aliases: &Aliases) -> Expr {
    match expression {
        Expr::Var(variable) => Expr::Var(*aliases.get(variable).unwrap_or(variable)),
        Expr::Const(value) => Expr::Const(*value),
        Expr::Add(left, right) => Expr::Add(Rc::new(substitute(left, aliases)), Rc::new(substitute(right, aliases))),
        Expr::Mul(left, right) => Expr::Mul(Rc::new(substitute(left, aliases)), Rc::new(substitute(right, aliases))),
    }
}

fn reference() -> Result<ApplicationCircuit, ApplicationError> {
    let mut builder = ApplicationBuilder::new(WIDTH)?;
    let input = std::array::from_fn(|lane| builder.private_inputs()[lane].into());
    let output = permutation(&mut builder, input, &round_constants())?;
    builder.finish(std::array::from_fn(|lane| output[lane].clone()))
}

fn expected_row(
    reference: &ApplicationCircuit,
    row: usize,
    lanes: &[Affine; WIDTH],
    aux: &[Variable],
) -> Result<[Affine; 3], ApplicationError> {
    let records = reference.records();
    let mut sides = records
        .row_header(row)?
        .constants
        .map(|value| Affine::constant(F::from_u64(value)));
    let first = reference.generated_range().start;
    let _: ControlFlow<()> = records.visit_terms(row, |term| {
        let value = if (4..4 + WIDTH).contains(&term.variable) {
            lanes[term.variable - 4].clone()
        } else {
            Affine::from(aux[term.variable - first])
        };
        sides[term.form.index()] = sides[term.form.index()].clone() + value * F::from_u64(term.coefficient);
        Ok(ControlFlow::Continue(()))
    })?;
    Ok(sides)
}

fn same(left: &Affine, right: &Affine) -> bool {
    left.constant_term() == right.constant_term() && left.terms() == right.terms()
}

pub(super) fn lower(
    builder: &mut ApplicationBuilder,
    rows: &[Expr],
    mut hints: Vec<Poseidon2Hint>,
    view: &View,
    aux_count: usize,
) -> Result<Aliases, ApplicationError> {
    let mut aliases = Aliases::new();
    if hints.is_empty() {
        for row in rows {
            lower_constraint(builder, row)?;
        }
        return Ok(aliases);
    }
    hints.sort_by_key(|hint| hint.row_start);
    let mut covered = vec![false; rows.len()];
    let mut owners = BTreeMap::new();
    let readable = view.variables();
    let mut traces = Vec::with_capacity(hints.len());
    for (index, hint) in hints.iter().enumerate() {
        let row_end = hint.row_start.checked_add(WORDS).ok_or_else(invalid)?;
        let aux_end = hint.aux_start.checked_add(WORDS).ok_or_else(invalid)?;
        if row_end > rows.len() || aux_end > aux_count {
            return Err(invalid());
        }
        for row in &mut covered[hint.row_start..row_end] {
            if std::mem::replace(row, true) {
                return Err(invalid());
            }
        }
        let mut trace = Vec::with_capacity(WORDS);
        for offset in hint.aux_start..aux_end {
            let column = variable(view.aux(offset))?;
            if owners.insert(column, index).is_some() {
                return Err(invalid());
            }
            trace.push(column);
        }
        for lane in &hint.lanes {
            if let Some(variable) = lane.variable_outside(&readable) {
                return Err(ApplicationError::VariableOutOfScope(variable.index()));
            }
        }
        traces.push(trace);
    }
    let reference = reference()?;
    let mut inputs = Vec::with_capacity(hints.len());
    for (index, (hint, trace)) in hints.iter().zip(&traces).enumerate() {
        let lanes: [Affine; WIDTH] = hint
            .lanes
            .iter()
            .map(|lane| lower_affine(lane).ok_or_else(invalid))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| invalid())?;
        for lane in &lanes {
            if lane
                .terms()
                .iter()
                .any(|(variable, _)| owners.get(variable).is_some_and(|owner| *owner >= index))
            {
                return Err(invalid());
            }
        }
        for offset in 0..WORDS {
            let (a, b, c) = rank_one_row(&rows[hint.row_start + offset]).ok_or_else(invalid)?;
            let actual = [a, b, c];
            let expected = expected_row(&reference, offset, &lanes, trace)?;
            if !actual.iter().zip(&expected).all(|(a, b)| same(a, b)) {
                return Err(invalid());
            }
        }
        inputs.push(lanes);
    }
    for (lanes, trace) in inputs.iter().zip(&traces) {
        let mapped = std::array::from_fn(|lane| {
            lanes[lane].terms().iter().fold(
                Affine::constant(lanes[lane].constant_term()),
                |value, (var, coefficient)| value + Affine::from(*aliases.get(var).unwrap_or(var)) * *coefficient,
            )
        });
        let first = builder.private_inputs().len() + 8 + builder.local_count();
        permutation(builder, mapped, &round_constants())?;
        for (offset, source) in trace.iter().enumerate() {
            aliases.insert(*source, Variable(first + offset));
        }
    }
    for (row, covered) in rows.iter().zip(covered) {
        if !covered {
            lower_constraint(builder, &substitute(row, &aliases))?;
        }
    }
    Ok(aliases)
}

pub(super) fn check_prefix(
    circuit: &ApplicationCircuit,
    end: usize,
    aliases: &Aliases,
) -> Result<(), ApplicationError> {
    if aliases.is_empty() {
        return Ok(());
    }
    for row in 0..end {
        let found: ControlFlow<()> = circuit.records().visit_terms(row, |term| {
            Ok(if aliases.contains_key(&Variable(term.variable)) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            })
        })?;
        if found.is_break() {
            return Err(ApplicationError::Prepared("memory row reads relocated application aux"));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/nebula/poseidon.rs"]
mod tests;
