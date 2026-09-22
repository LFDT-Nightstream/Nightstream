//! One sealed application snapshot with its final physical-column mapping.
//! Raw records bind identity; row lookup aggregates only distinct local variables.

use std::{collections::BTreeMap, ops::ControlFlow, sync::Arc};

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use crate::application_records::{ApplicationRecipeNode, ApplicationRecords, RecipeEvaluator};
use crate::sparse::{eval_sparse_combination, SparseCombination, SparseRow, SparseTerm};

use super::{canonical_field, sealed::LoadedApplicationPlan, Layout, LoadedPackage, PackageError};

#[derive(Clone, Debug)]
pub(crate) struct PreparedApplication {
    plan: LoadedApplicationPlan,
    records: Arc<ApplicationRecords>,
    variable_count: usize,
}

impl PreparedApplication {
    pub(crate) fn from_metadata(
        value: &serde_json::Value,
        records: Arc<ApplicationRecords>,
    ) -> Result<Self, PackageError> {
        Self::new(super::sealed::decode_native_application_plan(value)?, records)
    }

    pub(crate) fn new(plan: LoadedApplicationPlan, records: Arc<ApplicationRecords>) -> Result<Self, PackageError> {
        if plan.row_range().len() != records.row_count() || plan.private_range().len() != records.recipe_count() {
            return Err(PackageError::Invalid("native application record counts"));
        }
        let variable_count = plan
            .witness_columns()
            .len()
            .checked_add(plan.private_range().len())
            .and_then(|count| count.checked_add(plan.input_columns().len() + plan.output_columns().len()))
            .ok_or(PackageError::Invalid("native application variable count"))?;
        Ok(Self {
            plan,
            records,
            variable_count,
        })
    }

    pub(crate) fn plan(&self) -> &LoadedApplicationPlan {
        &self.plan
    }
    pub(crate) fn records(&self) -> &ApplicationRecords {
        &self.records
    }

    pub(crate) fn records_arc(&self) -> &Arc<ApplicationRecords> {
        &self.records
    }

    pub(crate) fn column(&self, local: usize) -> Result<usize, PackageError> {
        if local >= self.variable_count {
            return Err(PackageError::Invalid("native application variable"));
        }
        let inputs = self.plan.input_columns();
        if local < inputs.len() {
            return Ok(inputs[local]);
        }
        let local = local - inputs.len();
        let witness = self.plan.witness_columns();
        if local < witness.len() {
            return Ok(witness[local]);
        }
        let local = local - witness.len();
        let outputs = self.plan.output_columns();
        if local < outputs.len() {
            return Ok(outputs[local]);
        }
        self.plan
            .private_range()
            .start
            .checked_add(local - outputs.len())
            .ok_or(PackageError::Invalid("native application variable"))
    }

    pub(super) fn validate(&self, layout: &Layout) -> Result<(), PackageError> {
        let witness = layout
            .private_segments
            .iter()
            .find(|s| s.role == super::sealed::APPLICATION_WITNESS_ROLE)
            .ok_or(PackageError::Invalid("application witness segment"))?;
        let local = layout
            .private_segments
            .iter()
            .find(|s| s.role == super::sealed::APPLICATION_LOCAL_ROLE)
            .ok_or(PackageError::Invalid("application local segment"))?;
        let private = self.plan.private_range();
        let rows = self.plan.row_range();
        let witness_end = super::checked_end(witness.start, witness.length)?;
        if self.plan.witness_word_count() != witness.length
            || self
                .plan
                .witness_columns()
                .iter()
                .copied()
                .ne(witness.start..witness_end)
            || private.start != local.start
            || private.len() != local.length
            || private.end > layout.constant_column
            || rows.end > layout.row_count
            || self
                .plan
                .input_columns()
                .iter()
                .chain(self.plan.output_columns().iter())
                .any(|&c| c >= witness.start)
            || self
                .plan
                .input_columns()
                .iter()
                .chain(self.plan.witness_columns())
                .chain(self.plan.output_columns().iter())
                .any(|&c| c >= layout.total_column_count || c == layout.constant_column)
        {
            return Err(PackageError::Invalid("native application column or row ownership"));
        }
        for row in 0..self.records.row_count() {
            let header = self.records.row_header(row)?;
            for constant in header.constants {
                canonical_field(constant, "native application constant")?;
            }
            let mut counts = [0usize; 3];
            let mut previous = 0;
            let flow = self.records.visit_terms(row, |term| {
                let form = term.form.index();
                if form < previous {
                    return Err(PackageError::Invalid("native application term order"));
                }
                previous = form;
                self.column(term.variable)?;
                canonical_field(term.coefficient, "native application coefficient")?;
                counts[form] = counts[form]
                    .checked_add(1)
                    .ok_or(PackageError::Invalid("native application term count"))?;
                Ok(ControlFlow::Continue(()))
            })?;
            if flow.is_break() || counts != header.term_counts {
                return Err(PackageError::Invalid("native application row coverage"));
            }
        }
        let mut previous_row = None;
        let generated_start = self.variable_count - private.len();
        let output_start = generated_start - self.plan.output_columns().len();
        for recipe in 0..self.records.recipe_count() {
            let row = self.records.recipe_row(recipe)?;
            if row >= self.records.row_count() || previous_row.is_some_and(|previous| previous >= row) {
                return Err(PackageError::Invalid("native application recipe row order"));
            }
            previous_row = Some(row);
            let flow = self.records.visit_recipe_nodes(recipe, |node| {
                match node {
                    ApplicationRecipeNode::Variable(variable) => {
                        self.column(variable)?;
                        if variable >= generated_start + recipe || (output_start..generated_start).contains(&variable) {
                            return Err(PackageError::Invalid("noncausal witness expression"));
                        }
                    }
                    ApplicationRecipeNode::Constant(value) => {
                        canonical_field(value, "native application recipe")?;
                    }
                    ApplicationRecipeNode::Add | ApplicationRecipeNode::Multiply => {}
                }
                Ok(ControlFlow::Continue(()))
            })?;
            if flow.is_break() {
                return Err(PackageError::Invalid("native application recipe coverage"));
            }
        }
        Ok(())
    }

    pub(super) fn assertion(&self, row: usize) -> Result<SparseRow, PackageError> {
        let header = self.records.row_header(row)?;
        let mut coefficients: [BTreeMap<usize, Goldilocks>; 3] = std::array::from_fn(|_| BTreeMap::new());
        let _ = self.records.visit_terms(row, |term| {
            let column = self.column(term.variable)?;
            let terms = &mut coefficients[term.form.index()];
            let coefficient =
                terms.get(&column).copied().unwrap_or(Goldilocks::ZERO) + Goldilocks::from_u64(term.coefficient);
            if coefficient == Goldilocks::ZERO {
                terms.remove(&column);
            } else {
                terms.insert(column, coefficient);
            }
            Ok(ControlFlow::Continue(()))
        })?;
        let mut constants = header.constants.into_iter();
        let [a, b, c] = coefficients.map(|terms| SparseCombination {
            constant: Goldilocks::from_u64(constants.next().expect("three affine constants")),
            terms: terms
                .into_iter()
                .map(|(column, coefficient)| SparseTerm { column, coefficient })
                .collect(),
        });
        Ok(SparseRow {
            row_index: self.plan.row_range().start + row,
            a,
            b,
            c,
        })
    }

    pub(super) fn execute_recipes(&self, assignment: &mut [Goldilocks]) -> Result<(), PackageError> {
        let count = self.records.recipe_count();
        if count == 0 {
            return Ok(());
        }
        let mut evaluator = RecipeEvaluator::new(&self.records)?;
        for recipe in 0..count {
            let value = evaluator.evaluate(recipe, |variable| {
                Ok(assignment[self.column(variable)?].as_canonical_u64())
            })?;
            assignment[self.plan.private_range().start + recipe] = Goldilocks::from_u64(value);
        }
        Ok(())
    }

    pub(super) fn apply_values(
        &self,
        values: &[Goldilocks],
        assignment: &mut [Goldilocks],
    ) -> Result<(), PackageError> {
        if values.len() != self.variable_count {
            return Err(PackageError::Invalid("precomputed application value count"));
        }
        let generated = self.variable_count - self.plan.private_range().len();
        for (local, &value) in values.iter().enumerate() {
            let column = self.column(local)?;
            if local < generated {
                if value != assignment[column] {
                    return Err(PackageError::Invalid("precomputed application input or output differs"));
                }
            } else {
                assignment[column] = value;
            }
        }
        // The enclosing witness executor still checks every assertion row.
        Ok(())
    }

    fn check_assertions(&self, assignment: &[Goldilocks]) -> Result<(), PackageError> {
        for row in 0..self.records.row_count() {
            let values = self
                .records
                .evaluate_row(
                    row,
                    |variable| Ok(assignment[self.column(variable)?].as_canonical_u64()),
                )?
                .map(Goldilocks::from_u64);
            if values[0] * values[1] != values[2] {
                return Err(PackageError::UnsatisfiedAssertionRow {
                    row: self.plan.row_range().start + row,
                });
            }
        }
        Ok(())
    }
}

impl LoadedPackage {
    pub(super) fn check_assertions(&self, assignment: &[Goldilocks]) -> Result<(), PackageError> {
        let split = self
            .native_application
            .as_ref()
            .map_or(self.assertion_rows.len(), |application| {
                self.assertion_rows
                    .partition_point(|row| row.row_index < application.plan().row_range().start)
            });
        let check = |rows: &[SparseRow]| -> Result<(), PackageError> {
            for row in rows {
                if eval_sparse_combination(&row.a, assignment) * eval_sparse_combination(&row.b, assignment)
                    != eval_sparse_combination(&row.c, assignment)
                {
                    return Err(PackageError::UnsatisfiedAssertionRow { row: row.row_index });
                }
            }
            Ok(())
        };
        check(&self.assertion_rows[..split])?;
        if let Some(application) = &self.native_application {
            application.check_assertions(assignment)?;
        }
        check(&self.assertion_rows[split..])
    }
}

#[cfg(test)]
#[path = "../../tests/unit/native_application.rs"]
mod tests;
