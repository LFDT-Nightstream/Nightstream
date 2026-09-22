//! Substitutes matrix-program ports into the saved Lean formula library.

use std::{
    ops::{ControlFlow, Range},
    sync::OnceLock,
};

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

use crate::components::{ComponentError, FormulaLibrary, FormulaVariant, SparseForm};

use super::{form::normalize_terms, validate_form, Form, MatrixRun, PackageError, RowView, MEANINGFUL_PORTS};

fn variant(id: &str, index: usize) -> Result<&'static FormulaVariant, PackageError> {
    static LIBRARY: OnceLock<Result<FormulaLibrary, ComponentError>> = OnceLock::new();
    let library = LIBRARY
        .get_or_init(|| FormulaLibrary::from_json(include_bytes!("../../../artifacts/shared-formulas-v1.json")))
        .as_ref()
        .map_err(|_| PackageError::Invalid("compiled Lean formula library"))?;
    library
        .component(id)
        .and_then(|component| component.variant(index))
        .ok_or(PackageError::Invalid("compiled Lean formula component"))
}

fn check_inputs(template: &FormulaVariant, forms: &[Form], logical_width: usize) -> Result<(), PackageError> {
    if forms.len() != template.input_count() {
        return Err(PackageError::Invalid("matrix template input count"));
    }
    for form in forms {
        validate_form(form, logical_width)?;
    }
    Ok(())
}

pub(super) fn substitute_into(
    form: &SparseForm,
    inputs: &[Form],
    terms: &mut Vec<MatrixRun>,
) -> Result<(), PackageError> {
    terms.clear();
    let required = form
        .entries()
        .try_fold(0usize, |count, (input, coefficient)| {
            if coefficient == 0 {
                return Ok(count);
            }
            count
                .checked_add(inputs[input].terms().len())
                .ok_or(PackageError::Invalid("matrix template run count"))
        })?;
    if required > terms.capacity() {
        terms
            .try_reserve_exact(required)
            .map_err(|_| PackageError::Invalid("matrix template run allocation"))?;
    }
    for (input, coefficient) in form.entries() {
        if coefficient != 0 {
            let scalar = Goldilocks::from_u64(coefficient);
            terms.extend(inputs[input].terms().iter().map(|term| term.scaled(scalar)));
        }
    }
    // Input Forms have sorted unique keys and nonzero coefficients.
    // Nonzero SparseForm coefficients preserve this under single-input scaling.
    if form.entries().len() > 1 {
        normalize_terms(terms);
    }
    Ok(())
}

#[derive(Default)]
pub(super) struct RowScratch {
    ports: [Vec<MatrixRun>; MEANINGFUL_PORTS],
}

impl RowScratch {
    pub(super) fn visit_rows_until(
        &mut self,
        id: &str,
        variant_index: usize,
        forms: &[Form],
        logical_width: usize,
        range: Range<usize>,
        mut visit: impl FnMut(RowView<'_>) -> Result<ControlFlow<()>, PackageError>,
    ) -> Result<ControlFlow<()>, PackageError> {
        let template = variant(id, variant_index)?;
        check_inputs(template, forms, logical_width)?;
        for row in template
            .row_templates(range)
            .map_err(|_| PackageError::Invalid("matrix template row substitution"))?
        {
            for (port, terms) in self.ports.iter_mut().enumerate() {
                substitute_into(&row.ports()[port], forms, terms)?;
            }
            if visit(std::array::from_fn(|port| self.ports[port].as_slice()))?.is_break() {
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    }
}

pub(super) fn outputs(id: &str, forms: &[Form], logical_width: usize) -> Result<Vec<Form>, PackageError> {
    let template = variant(id, 0)?;
    check_inputs(template, forms, logical_width)?;
    template
        .output_templates()
        .iter()
        .map(|output| {
            let mut terms = Vec::new();
            substitute_into(output, forms, &mut terms)?;
            Ok(Form::from_terms(terms))
        })
        .collect()
}
