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
    let mut columns = input_columns.clone();
    columns.extend_from_slice(&witness_columns);
    columns.extend_from_slice(&output_columns);
    columns.extend(
        private_start
            ..private_start
                .checked_add(counts.local)
                .ok_or(AssemblyError::Overflow)?,
    );
    if columns.len() != application.variable_count() {
        return Err(AssemblyError::Invalid("application port width"));
    }
    let combination = |value: &crate::application::Affine| -> Result<Combination, AssemblyError> {
        Ok(serde_json::from_value(value.expression.affine(&columns))?)
    };
    let rows = application
        .rows()
        .iter()
        .enumerate()
        .map(|(offset, row)| {
            Ok(Row {
                index: row_start
                    .checked_add(offset)
                    .ok_or(AssemblyError::Overflow)?,
                a: combination(row.a())?,
                b: combination(row.b())?,
                c: combination(row.c())?,
            })
        })
        .collect::<Result<Vec<_>, AssemblyError>>()?;
    let batches = if application.recipes().is_empty() {
        Vec::new()
    } else {
        vec![Batch {
            start: private_start,
            recipes: application
                .recipes()
                .iter()
                .map(|recipe| recipe.encode(&columns))
                .collect(),
            hints: Vec::new(),
        }]
    };
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
        batches,
        instructions: Vec::new(),
        rows,
    })
}
