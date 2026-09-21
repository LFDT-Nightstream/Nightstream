//! Substitutes matrix-program ports into the saved Lean formula library.

use std::{ops::Range, sync::OnceLock};

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

use crate::components::{ComponentError, FormulaLibrary, FormulaVariant, SparseForm};

use super::{validate_form, Form, PackageError, RowForms};

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

fn substitute(form: &SparseForm, inputs: &[Form]) -> Form {
    form.entries()
        .fold(Form::default(), |sum, (input, coefficient)| {
            sum.append(
                inputs[input]
                    .clone()
                    .scaled(Goldilocks::from_u64(coefficient)),
            )
        })
}

pub(super) fn rows(
    id: &str,
    variant_index: usize,
    forms: &[Form],
    logical_width: usize,
    range: Range<usize>,
) -> Result<Vec<RowForms>, PackageError> {
    let template = variant(id, variant_index)?;
    check_inputs(template, forms, logical_width)?;
    Ok(template
        .row_templates(range)
        .map_err(|_| PackageError::Invalid("matrix template row substitution"))?
        .iter()
        .map(|row| std::array::from_fn(|port| substitute(&row.ports()[port], forms)))
        .collect())
}

pub(super) fn outputs(id: &str, forms: &[Form], logical_width: usize) -> Result<Vec<Form>, PackageError> {
    let template = variant(id, 0)?;
    check_inputs(template, forms, logical_width)?;
    Ok(template
        .output_templates()
        .iter()
        .map(|output| substitute(output, forms))
        .collect())
}
