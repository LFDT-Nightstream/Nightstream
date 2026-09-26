use super::wire::{ApplicationPlan, Batch, Combination, Row};
use super::*;
use nightstream_fprime::{ApplicationForm, ApplicationRecipeNode};
use std::ops::ControlFlow;

/// Reconstruct every stored row and recipe for exact test comparison.
pub(super) fn materialized_plan(
    application: &ApplicationCircuit,
    manifest: &Manifest,
) -> Result<ApplicationPlan, AssemblyError> {
    let mut plan = application::plan(application, manifest)?;
    let mut columns = plan.input_columns.clone();
    columns.extend_from_slice(&plan.witness_columns);
    columns.extend_from_slice(&plan.output_columns);
    columns.extend(plan.private_start..plan.private_start + plan.private_count);
    let records = application.records();
    for row in 0..records.row_count() {
        let header = records.row_header(row)?;
        let mut forms = header.constants.map(|constant| Combination {
            constant,
            terms: Vec::new(),
        });
        let mut missing = false;
        let _ = records.visit_terms(row, |term| {
            let form = match term.form {
                ApplicationForm::A => 0,
                ApplicationForm::B => 1,
                ApplicationForm::C => 2,
            };
            match columns.get(term.variable) {
                Some(&column) => forms[form].terms.push((column, term.coefficient)),
                None => missing = true,
            }
            Ok(ControlFlow::Continue(()))
        })?;
        if missing {
            return Err(AssemblyError::Invalid("application row variable"));
        }
        let [a, b, c] = forms;
        plan.rows.push(Row {
            index: plan
                .row_start
                .checked_add(row)
                .ok_or(AssemblyError::Overflow)?,
            a,
            b,
            c,
        });
    }
    if records.recipe_count() != 0 {
        let mut recipes = Vec::with_capacity(records.recipe_count());
        for recipe in 0..records.recipe_count() {
            let mut nodes = Vec::new();
            let _ = records.visit_recipe_nodes(recipe, |node| {
                nodes.push(node);
                Ok(ControlFlow::Continue(()))
            })?;
            let mut nodes = nodes.into_iter();
            recipes.push(recipe_value(&mut nodes, &columns)?);
            if nodes.next().is_some() {
                return Err(AssemblyError::Invalid("application recipe length"));
            }
        }
        plan.batches.push(Batch {
            start: plan.private_start,
            recipes,
            hints: Vec::new(),
        });
    }
    Ok(plan)
}

fn recipe_value(
    nodes: &mut impl Iterator<Item = ApplicationRecipeNode>,
    columns: &[usize],
) -> Result<Value, AssemblyError> {
    Ok(
        match nodes
            .next()
            .ok_or(AssemblyError::Invalid("application recipe length"))?
        {
            ApplicationRecipeNode::Variable(variable) => json!([
                0,
                columns
                    .get(variable)
                    .ok_or(AssemblyError::Invalid("application recipe variable"))?
            ]),
            ApplicationRecipeNode::Constant(value) => json!([1, value]),
            ApplicationRecipeNode::Add => json!([2, recipe_value(nodes, columns)?, recipe_value(nodes, columns)?]),
            ApplicationRecipeNode::Multiply => json!([3, recipe_value(nodes, columns)?, recipe_value(nodes, columns)?]),
        },
    )
}
