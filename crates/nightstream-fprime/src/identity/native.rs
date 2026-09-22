//! Canonical identity replay from fixed package metadata and sealed application records.
//! Local record columns are mapped at read time; both envelope occurrences use the same owner.

use std::ops::ControlFlow;

use crate::application_records::{ApplicationForm, ApplicationRecipeNode, ApplicationRowHeader};
use crate::package::native_application::PreparedApplication;

use super::*;

pub(crate) fn native_relation_identifier(
    fixed: &Value,
    application: &PreparedApplication,
) -> Result<[u64; 4], PackageError> {
    let mut input = poseidon2::Poseidon2Hasher::default();
    update_words(&mut input, &IDENTITY_DOMAIN);
    visit_native_preimage_words(fixed, application, &mut |words| {
        update_words(&mut input, words);
        Ok(())
    })?;
    Ok(input.finalize().map(|value| value.as_canonical_u64()))
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
        return Err(PackageError::Invalid("application identity word coverage"));
    }
    Ok(ApplicationIdentity {
        digest: input.finalize().map(|value| value.as_canonical_u64()),
        word_count,
    })
}

pub(crate) fn native_application_word_count(application: &PreparedApplication) -> Result<usize, PackageError> {
    let records = application.records();
    // The 16-field ABI has two four-column lists, four empty row families,
    // empty instructions, and the row/batch array headers.
    let mut word_count = 100usize
        .checked_add(
            application
                .plan()
                .witness_columns()
                .len()
                .checked_mul(4)
                .ok_or(PackageError::Invalid("application identity length"))?,
        )
        .and_then(|count| count.checked_add(records.row_preimage_word_count()))
        .ok_or(PackageError::Invalid("application identity length"))?;
    if records.recipe_count() != 0 {
        word_count = word_count
            .checked_add(16)
            .and_then(|count| count.checked_add(records.recipe_preimage_word_count()))
            .ok_or(PackageError::Invalid("application identity length"))?;
    }
    Ok(word_count)
}

pub(crate) fn visit_native_application_words(
    application: &PreparedApplication,
    emit: &mut dyn FnMut(&[u64]) -> Result<(), PackageError>,
) -> Result<(), PackageError> {
    application_value(application, &mut CanonicalSink { emit })
}

pub(crate) fn visit_native_preimage_words(
    fixed: &Value,
    application: &PreparedApplication,
    emit: &mut dyn FnMut(&[u64]) -> Result<(), PackageError>,
) -> Result<(), PackageError> {
    let envelope = tuple(fixed, 7)?;
    let mut sink = CanonicalSink { emit };
    sink.array(envelope.len())?;
    for (index, value) in envelope.iter().enumerate() {
        match index {
            1 => source_value(value, application, &mut sink)?,
            3 => application_value(application, &mut sink)?,
            _ => sink.value(value)?,
        }
    }
    Ok(())
}

fn source_value(
    value: &Value,
    application: &PreparedApplication,
    sink: &mut CanonicalSink<'_>,
) -> Result<(), PackageError> {
    let source = tuple(value, 14)?;
    sink.array(source.len())?;
    for (index, value) in source.iter().enumerate() {
        match index {
            10 => {
                let batches = array(value)?;
                let added = usize::from(application.records().recipe_count() != 0);
                sink.array(add(batches.len(), added)?)?;
                for batch in batches {
                    sink.value(batch)?;
                }
                if added != 0 {
                    application_batch(application, sink)?;
                }
            }
            12 => source_rows(value, application, sink)?,
            _ => sink.value(value)?,
        }
    }
    Ok(())
}

fn source_rows(
    value: &Value,
    application: &PreparedApplication,
    sink: &mut CanonicalSink<'_>,
) -> Result<(), PackageError> {
    let fixed = array(value)?;
    let range = application.plan().row_range();
    sink.array(add(fixed.len(), application.records().row_count())?)?;
    let mut inserted = false;
    for row in fixed {
        let index = tuple(row, 4)?[0]
            .as_u64()
            .and_then(|index| usize::try_from(index).ok())
            .ok_or(PackageError::Invalid("native fixed row index"))?;
        if range.contains(&index) || (inserted && index < range.start) {
            return Err(PackageError::Invalid("native application row insertion"));
        }
        if !inserted && index >= range.start {
            application_rows(application, sink)?;
            inserted = true;
        }
        sink.value(row)?;
    }
    if !inserted {
        application_rows(application, sink)?;
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
                return Err(PackageError::Invalid("native application term order"));
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
            return Err(PackageError::Invalid("native application row coverage"));
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
            return Err(PackageError::Invalid("native application recipe coverage"));
        }
    }
    sink.array(0)
}

fn add(left: usize, right: usize) -> Result<usize, PackageError> {
    left.checked_add(right)
        .ok_or(PackageError::Invalid("native identity length"))
}

fn array(value: &Value) -> Result<&[Value], PackageError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or(PackageError::Invalid("native identity array"))
}

fn tuple(value: &Value, length: usize) -> Result<&[Value], PackageError> {
    let values = array(value)?;
    if values.len() != length {
        return Err(PackageError::Invalid("native identity tuple"));
    }
    Ok(values)
}
