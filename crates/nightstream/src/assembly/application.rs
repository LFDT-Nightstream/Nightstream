use std::ops::ControlFlow;

use nightstream_fprime::{ApplicationForm, ApplicationRecipeNode};
use serde_json::{json, Value};

use crate::application::ApplicationCircuit;

use super::{
    manifest::{Counts, Manifest},
    wire::{ApplicationPlan, Batch, Combination, Row},
    AssemblyError,
};

pub(super) fn plan(application: &ApplicationCircuit, manifest: &Manifest) -> Result<ApplicationPlan, AssemblyError> {
    let counts = Counts::of(application);
    let input_columns = manifest.port("state_input")?.source_columns(counts)?;
    let witness_columns = manifest
        .port("application_witness")?
        .source_columns(counts)?;
    let output_columns = manifest.port("state_output")?.source_columns(counts)?;
    let private_start = manifest
        .port("application_local")?
        .source_start
        .eval(counts)?;
    let row_start = manifest.source_relocation.application_row_start;
    private_start
        .checked_add(counts.local)
        .ok_or(AssemblyError::Overflow)?;
    let variables = input_columns
        .len()
        .checked_add(witness_columns.len())
        .and_then(|count| count.checked_add(output_columns.len()))
        .and_then(|count| count.checked_add(counts.local))
        .ok_or(AssemblyError::Overflow)?;
    if variables != application.variable_count() {
        return Err(AssemblyError::Invalid("application port width"));
    }
    Ok(ApplicationPlan {
        schema: 1,
        witness_count: counts.witness,
        input_columns,
        witness_columns,
        output_columns,
        private_start,
        private_count: counts.local,
        row_start,
        row_count: counts.rows,
        hash_chains: Vec::new(),
        permutations: Vec::new(),
        compact_templates: Vec::new(),
        compact_invocations: Vec::new(),
        batches: Vec::new(),
        instructions: Vec::new(),
        rows: Vec::new(),
    })
}

/// The stored plan with every application row and recipe. Assembly uses it
/// only to recognize the proved selected specialization, whose reference
/// already stores a plan of this size.
pub(super) fn materialized_plan(
    application: &ApplicationCircuit,
    manifest: &Manifest,
) -> Result<ApplicationPlan, AssemblyError> {
    let mut plan = plan(application, manifest)?;
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
