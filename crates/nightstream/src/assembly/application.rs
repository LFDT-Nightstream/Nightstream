use crate::application::ApplicationCircuit;

use super::{
    manifest::{Counts, Manifest},
    wire::ApplicationPlan,
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
