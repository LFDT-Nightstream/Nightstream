//! Instantiates the exported connector paths and canonical assignment runs.

use std::collections::BTreeSet;

use serde_json::{json, Value};

use super::{
    manifest::{Counts, Manifest, Relocation, Run},
    wire::{AssignmentBlock, Envelope},
    AssemblyError,
};

fn path_mut<'a>(mut value: &'a mut Value, path: &[usize]) -> Result<&'a mut Value, AssemblyError> {
    for index in path {
        value = value
            .as_array_mut()
            .and_then(|items| items.get_mut(*index))
            .ok_or(AssemblyError::Invalid("exported relocation path"))?;
    }
    Ok(value)
}

fn relocate(value: &mut Value, relocations: &[Relocation], from: Counts, to: Counts) -> Result<(), AssemblyError> {
    let mut used = BTreeSet::new();
    for relocation in relocations {
        if relocation.path.is_empty() || !used.insert(&relocation.path) {
            return Err(AssemblyError::Invalid("duplicate or empty relocation path"));
        }
        let target = path_mut(value, &relocation.path)?;
        if super::word(target)? != relocation.value.eval(from)? {
            return Err(AssemblyError::Invalid("relocation does not match reference field"));
        }
        *target = json!(relocation.value.eval(to)?);
    }
    Ok(())
}

pub(super) fn matrix(reference: &mut Envelope, manifest: &Manifest, counts: Counts) -> Result<(), AssemblyError> {
    let child = manifest.application_child();
    let end = child
        .block_start
        .checked_add(child.block_count)
        .ok_or(AssemblyError::Overflow)?;
    let old = manifest.reference();
    let template = &manifest.application_matrix_template;
    let connector = template
        .as_array()
        .ok_or(AssemblyError::Invalid("application connector program"))?;
    if reference.matrix.get(child.block_start..end) != Some(connector.as_slice()) {
        return Err(AssemblyError::Invalid("application connector differs from reference"));
    }
    for relocation in &manifest.matrix_relocations {
        if relocation
            .path
            .first()
            .is_none_or(|block| *block >= reference.matrix.len() || (child.block_start..end).contains(block))
        {
            return Err(AssemblyError::Invalid(
                "shared relocation targets application or missing child",
            ));
        }
    }
    let original = serde_json::to_value(&reference.matrix)?;
    let mut relocated = original.clone();
    relocate(&mut relocated, &manifest.matrix_relocations, old, counts)?;
    let mut restored = relocated.clone();
    relocate(&mut restored, &manifest.matrix_relocations, counts, old)?;
    if restored != original {
        return Err(AssemblyError::Invalid(
            "shared verifier child changed outside declared relocation",
        ));
    }
    let mut actual_connector = template.clone();
    relocate(
        &mut actual_connector,
        &manifest.application_matrix_relocations,
        old,
        counts,
    )?;
    let mut restored_connector = actual_connector.clone();
    relocate(
        &mut restored_connector,
        &manifest.application_matrix_relocations,
        counts,
        old,
    )?;
    if restored_connector != *template {
        return Err(AssemblyError::Invalid("application connector relocation"));
    }
    reference.matrix = serde_json::from_value(relocated)?;
    let replacement: Vec<Value> = serde_json::from_value(actual_connector)?;
    if replacement.len() != child.block_count {
        return Err(AssemblyError::Invalid("application connector block count"));
    }
    reference.matrix.splice(child.block_start..end, replacement);
    Ok(())
}

/// Same right-to-left greedy compression as Export/AffineRuns.compressIndexedTR.
/// Source values are streamed; no expanded source-index vector is allocated.
fn runs(templates: &[Run], counts: Counts) -> Result<Vec<[usize; 3]>, AssemblyError> {
    let mut reversed: Vec<[usize; 3]> = Vec::new();
    for run in templates.iter().rev() {
        let first = run.first.eval(counts)?;
        for offset in (0..run.count.eval(counts)?).rev() {
            let value = first
                .checked_add(
                    run.step
                        .checked_mul(offset)
                        .ok_or(AssemblyError::Overflow)?,
                )
                .ok_or(AssemblyError::Overflow)?;
            if let Some(last) = reversed.last_mut() {
                if last[2] == 1 && value <= last[0] {
                    *last = [value, last[0] - value, 2];
                    continue;
                }
                if last[2] > 1 && value.checked_add(last[1]) == Some(last[0]) {
                    last[0] = value;
                    last[2] = last[2].checked_add(1).ok_or(AssemblyError::Overflow)?;
                    continue;
                }
            }
            reversed.push([value, 0, 1]);
        }
    }
    reversed.reverse();
    Ok(reversed)
}

fn assignment_blocks(manifest: &Manifest, counts: Counts) -> Result<Vec<AssignmentBlock>, AssemblyError> {
    manifest
        .assignment_blocks
        .iter()
        .enumerate()
        .map(|(opcode, block)| {
            if block.opcode != opcode {
                return Err(AssemblyError::Invalid("assignment block order"));
            }
            let sources = runs(&block.source_runs, counts)?;
            let total = sources.iter().try_fold(0usize, |sum, run| {
                sum.checked_add(run[2]).ok_or(AssemblyError::Overflow)
            })?;
            let slot_count = block.slot_count.eval(counts)?;
            if total != slot_count {
                return Err(AssemblyError::Invalid("assignment run coverage"));
            }
            Ok(AssignmentBlock {
                opcode,
                slot_kind: block.slot_kind,
                slot_count,
                source_domain: block.source_domain,
                runs: sources,
            })
        })
        .collect()
}

pub(super) fn assignment(reference: &mut Envelope, manifest: &Manifest, counts: Counts) -> Result<(), AssemblyError> {
    if assignment_blocks(manifest, manifest.reference())? != reference.assignment.blocks {
        return Err(AssemblyError::Invalid("assignment templates differ from reference"));
    }
    reference.assignment.blocks = assignment_blocks(manifest, counts)?;
    // The schema-3 quotient recipe stores valueSources at field 9.
    let phi81 = reference
        .assignment
        .phi81
        .as_array_mut()
        .filter(|fields| fields.len() == 11)
        .ok_or(AssemblyError::Invalid("Phi81 assignment recipe"))?;
    if phi81[9] != serde_json::to_value(runs(&manifest.phi81_value_sources, manifest.reference())?)? {
        return Err(AssemblyError::Invalid(
            "Phi81 value source template differs from reference",
        ));
    }
    phi81[9] = serde_json::to_value(runs(&manifest.phi81_value_sources, counts)?)?;
    Ok(())
}
