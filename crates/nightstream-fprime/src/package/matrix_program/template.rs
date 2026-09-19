//! Substitutes matrix-program ports into the saved Lean formula library.

use std::{ops::Range, sync::OnceLock};

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use crate::components::{ComponentError, FormulaLibrary, FormulaVariant, SparseForm};

use super::{Entry, Form, PackageError, RowForms};

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

fn inputs(forms: &[Form]) -> Result<Vec<SparseForm>, PackageError> {
    forms
        .iter()
        .map(|form| {
            SparseForm::new(
                form.entries()
                    .iter()
                    .map(|entry| (entry.column, entry.coefficient.as_canonical_u64())),
            )
            .map_err(|_| PackageError::Invalid("matrix template input form"))
        })
        .collect()
}

fn form(value: &SparseForm) -> Form {
    Form::from_canonical_entries(
        value
            .entries()
            .map(|(column, coefficient)| Entry {
                column,
                coefficient: Goldilocks::from_u64(coefficient),
            })
            .collect(),
    )
}

pub(super) fn rows(
    id: &str,
    variant_index: usize,
    forms: &[Form],
    logical_width: usize,
    range: Range<usize>,
) -> Result<Vec<RowForms>, PackageError> {
    Ok(variant(id, variant_index)?
        .rows_range(&inputs(forms)?, logical_width, range)
        .map_err(|_| PackageError::Invalid("matrix template row substitution"))?
        .into_iter()
        .map(|row| std::array::from_fn(|port| form(&row.ports()[port])))
        .collect())
}

pub(super) fn outputs(id: &str, forms: &[Form], logical_width: usize) -> Result<Vec<Form>, PackageError> {
    Ok(variant(id, 0)?
        .outputs(&inputs(forms)?, logical_width)
        .map_err(|_| PackageError::Invalid("matrix template output substitution"))?
        .iter()
        .map(form)
        .collect())
}
