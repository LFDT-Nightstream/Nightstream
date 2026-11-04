//! Canonical identity replay from fixed package metadata and sealed application records.

use std::ops::ControlFlow;

use crate::application_records::{ApplicationForm, ApplicationRecipeNode, ApplicationRowHeader};
use crate::package::native_application::PreparedApplication;

use super::*;

const FIXED_ABI_WORDS: usize = 100;
const RECIPE_BATCH_HEADER_WORDS: usize = 16;
const ENVELOPE_FIELDS: usize = 7;
const PACKAGE_FIELDS: usize = 14;
const PACKAGE_WITNESS_BATCHES: usize = 10;
const PACKAGE_WITNESS_INSTRUCTIONS: usize = 11;
const PACKAGE_ASSERTION_ROWS: usize = 12;
/// Lean `PerApplicationCanonicalPackage.nextPreimageRowCount`.
pub(crate) const NEXT_PREIMAGE_ROWS: usize = 5;

/// The application entries that the package of an envelope still holds: the
/// last `batches` witness batches, the last `instructions` witness
/// instructions, and the `rows` assertion rows before the last
/// `NEXT_PREIMAGE_ROWS` rows (Lean `PerApplicationCanonicalPackage.reassemble`).
/// A native fixed envelope holds none.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ApplicationEntries {
    pub(crate) batches: usize,
    pub(crate) instructions: usize,
    pub(crate) rows: usize,
}

/// Lean `PerApplicationCanonicalPackage.structuralPackageIdentity`: the hash of
/// `structuralValue`, the envelope with the application digest at index 3 and
/// the package without the application entries. The application rows enter
/// only through `application`, so they are hashed once.
pub(crate) fn structural_identifier(
    envelope: &Value,
    application: &ApplicationIdentity,
    entries: ApplicationEntries,
) -> Result<StructuralIdentifier, PackageError> {
    let envelope = tuple(envelope, ENVELOPE_FIELDS)?;
    let mut input = poseidon2::Poseidon2Hasher::default();
    update_words(&mut input, &IDENTITY_DOMAIN);
    let mut emit = |words: &[u64]| {
        update_words(&mut input, words);
        Ok(())
    };
    let mut sink = CanonicalSink { emit: &mut emit };
    sink.array(envelope.len())?;
    for (index, value) in envelope.iter().enumerate() {
        match index {
            1 => fixed_package(value, entries, &mut sink)?,
            3 => {
                sink.array(application.digest.len())?;
                for word in application.digest {
                    sink.number(word)?;
                }
            }
            _ => sink.value(value)?,
        }
    }
    Ok(StructuralIdentifier(
        input.finalize().map(|value| value.as_canonical_u64()),
    ))
}

pub(crate) fn native_application_identity(
    application: &PreparedApplication,
) -> Result<ApplicationIdentity, PackageError> {
    let word_count = native_application_word_count(application)?;
    let mut input = component_hasher(2, word_count)?;
    let mut emitted = 0usize;
    visit_native_application_words(application, &mut |words| {
        emitted = add(emitted, words.len())?;
        update_words(&mut input, words);
        Ok(())
    })?;
    if emitted != word_count {
        return Err(PackageError::Malformed("application identity word coverage"));
    }
    Ok(ApplicationIdentity {
        digest: input.finalize().map(|value| value.as_canonical_u64()),
        word_count,
    })
}

pub(crate) fn native_application_word_count(application: &PreparedApplication) -> Result<usize, PackageError> {
    let records = application.records();
    let mut word_count = FIXED_ABI_WORDS
        .checked_add(
            application
                .plan()
                .witness_columns()
                .len()
                .checked_mul(4)
                .ok_or(PackageError::Malformed("application identity length"))?,
        )
        .and_then(|count| count.checked_add(records.row_preimage_word_count()))
        .ok_or(PackageError::Malformed("application identity length"))?;
    if records.recipe_count() != 0 {
        word_count = word_count
            .checked_add(RECIPE_BATCH_HEADER_WORDS)
            .and_then(|count| count.checked_add(records.recipe_preimage_word_count()))
            .ok_or(PackageError::Malformed("application identity length"))?;
    }
    Ok(word_count)
}

pub(crate) fn visit_native_application_words(
    application: &PreparedApplication,
    emit: &mut dyn FnMut(&[u64]) -> Result<(), PackageError>,
) -> Result<(), PackageError> {
    application_value(application, &mut CanonicalSink { emit })
}

fn fixed_package(value: &Value, entries: ApplicationEntries, sink: &mut CanonicalSink<'_>) -> Result<(), PackageError> {
    let package = tuple(value, PACKAGE_FIELDS)?;
    sink.array(package.len())?;
    for (index, value) in package.iter().enumerate() {
        match index {
            PACKAGE_WITNESS_BATCHES => fixed_list(value, entries.batches, 0, sink)?,
            PACKAGE_WITNESS_INSTRUCTIONS => fixed_list(value, entries.instructions, 0, sink)?,
            PACKAGE_ASSERTION_ROWS => fixed_list(value, entries.rows, NEXT_PREIMAGE_ROWS, sink)?,
            _ => sink.value(value)?,
        }
    }
    Ok(())
}

/// Emit `value` without the `removed` entries before its last `kept` entries.
fn fixed_list(value: &Value, removed: usize, kept: usize, sink: &mut CanonicalSink<'_>) -> Result<(), PackageError> {
    if removed == 0 {
        return sink.value(value);
    }
    let values = array(value)?;
    let end = values
        .len()
        .checked_sub(kept)
        .ok_or(PackageError::Malformed("native identity application entries"))?;
    let start = end
        .checked_sub(removed)
        .ok_or(PackageError::Malformed("native identity application entries"))?;
    sink.array(values.len() - removed)?;
    for value in values[..start].iter().chain(&values[end..]) {
        sink.value(value)?;
    }
    Ok(())
}

fn application_value(application: &PreparedApplication, sink: &mut CanonicalSink<'_>) -> Result<(), PackageError> {
    let plan = application.plan();
    sink.array(16)?;
    sink.number(1)?;
    sink.index(plan.witness_word_count())?;
    let input_columns = plan.input_columns();
    let output_columns = plan.output_columns();
    for columns in [
        input_columns.as_slice(),
        plan.witness_columns(),
        output_columns.as_slice(),
    ] {
        sink.array(columns.len())?;
        for &column in columns {
            sink.index(column)?;
        }
    }
    let private = plan.private_range();
    let rows = plan.row_range();
    for value in [private.start, private.len(), rows.start, rows.len()] {
        sink.index(value)?;
    }
    for _ in 0..4 {
        sink.array(0)?;
    }
    let has_recipes = application.records().recipe_count() != 0;
    sink.array(usize::from(has_recipes))?;
    if has_recipes {
        application_batch(application, sink)?;
    }
    sink.array(0)?;
    sink.array(application.records().row_count())?;
    application_rows(application, sink)
}

fn application_rows(application: &PreparedApplication, sink: &mut CanonicalSink<'_>) -> Result<(), PackageError> {
    let records = application.records();
    for row in 0..records.row_count() {
        let header = records.row_header(row)?;
        sink.array(4)?;
        sink.index(add(application.plan().row_range().start, row)?)?;
        let mut form = 0;
        let mut counts = [0usize; 3];
        combination_header(&header, form, sink)?;
        let flow = records.visit_terms(row, |term| {
            let next = match term.form {
                ApplicationForm::A => 0,
                ApplicationForm::B => 1,
                ApplicationForm::C => 2,
            };
            if next < form {
                return Err(PackageError::Malformed("native application term order"));
            }
            while form < next {
                form += 1;
                combination_header(&header, form, sink)?;
            }
            counts[form] = add(counts[form], 1)?;
            sink.array(2)?;
            sink.index(application.column(term.variable)?)?;
            sink.number(term.coefficient)?;
            Ok(ControlFlow::Continue(()))
        })?;
        while form < 2 {
            form += 1;
            combination_header(&header, form, sink)?;
        }
        if flow.is_break() || counts != header.term_counts {
            return Err(PackageError::Malformed("native application row coverage"));
        }
    }
    Ok(())
}

fn combination_header(
    header: &ApplicationRowHeader,
    form: usize,
    sink: &mut CanonicalSink<'_>,
) -> Result<(), PackageError> {
    sink.array(2)?;
    sink.number(header.constants[form])?;
    sink.array(header.term_counts[form])
}

fn application_batch(application: &PreparedApplication, sink: &mut CanonicalSink<'_>) -> Result<(), PackageError> {
    let records = application.records();
    sink.array(3)?;
    sink.index(application.plan().private_range().start)?;
    sink.array(records.recipe_count())?;
    for recipe in 0..records.recipe_count() {
        let flow = records.visit_recipe_nodes(recipe, |node| {
            match node {
                ApplicationRecipeNode::Variable(variable) => {
                    sink.array(2)?;
                    sink.number(0)?;
                    sink.index(application.column(variable)?)?;
                }
                ApplicationRecipeNode::Constant(value) => {
                    sink.array(2)?;
                    sink.number(1)?;
                    sink.number(value)?;
                }
                ApplicationRecipeNode::Add | ApplicationRecipeNode::Multiply => {
                    sink.array(3)?;
                    sink.number(if node == ApplicationRecipeNode::Add { 2 } else { 3 })?;
                }
            }
            Ok(ControlFlow::Continue(()))
        })?;
        if flow.is_break() {
            return Err(PackageError::Malformed("native application recipe coverage"));
        }
    }
    sink.array(0)
}

fn add(left: usize, right: usize) -> Result<usize, PackageError> {
    left.checked_add(right)
        .ok_or(PackageError::Malformed("native identity length"))
}

fn array(value: &Value) -> Result<&[Value], PackageError> {
    crate::package::json::array(value, "native identity array")
}

fn tuple(value: &Value, length: usize) -> Result<&[Value], PackageError> {
    let values = array(value)?;
    if values.len() != length {
        return Err(PackageError::Malformed("native identity tuple"));
    }
    Ok(values)
}
