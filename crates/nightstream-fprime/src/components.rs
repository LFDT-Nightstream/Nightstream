//! Decodes exported matrix formulas and substitutes their named input ports.
//! Formula data comes from Lean. This module supplies only sparse linear algebra;
//! loading a library does not establish its authority or a complete verifier.

use std::collections::BTreeSet;

use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::Deserialize;
use thiserror::Error;

// The selected Nightstream profile, in the exported profile field order.
const PROFILE: [u64; 8] = [
    Fq::ORDER_U64,
    crate::DECOMPOSITION_BASE,
    crate::DECOMPOSITION_K_RHO as u64,
    crate::DECOMPOSITION_NORM_BOUND,
    crate::RING_DEGREE as u64,
    crate::PI_CCS_V1_2_ROUND_COUNT as u64,
    crate::PI_CCS_V1_2_MATRIX_COUNT as u64,
    MATRIX_PORTS as u64,
];
const MATRIX_PORTS: usize = 4;

#[derive(Debug, Error)]
pub(crate) enum ComponentError {
    #[error("invalid component JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid component: {0}")]
    Invalid(&'static str),
}

/// Canonical sparse linear form over field-valued columns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SparseForm {
    entries: Vec<(usize, Fq)>,
}

impl SparseForm {
    pub(crate) fn new(entries: impl IntoIterator<Item = (usize, u64)>) -> Result<Self, ComponentError> {
        let mut terms = Vec::new();
        for (column, coefficient) in entries {
            if coefficient >= Fq::ORDER_U64 {
                return Err(ComponentError::Invalid("noncanonical field coefficient"));
            }
            terms.push((column, Fq::from_u64(coefficient)));
        }
        Ok(Self::canonical(terms))
    }

    pub(crate) fn variable(column: usize) -> Self {
        Self {
            entries: vec![(column, Fq::ONE)],
        }
    }

    pub(crate) fn entries(&self) -> impl ExactSizeIterator<Item = (usize, u64)> + '_ {
        self.entries
            .iter()
            .map(|(column, coefficient)| (*column, coefficient.as_canonical_u64()))
    }

    fn canonical(mut entries: Vec<(usize, Fq)>) -> Self {
        entries.sort_unstable_by_key(|entry| entry.0);
        let mut result: Vec<(usize, Fq)> = Vec::with_capacity(entries.len());
        for (column, coefficient) in entries {
            if coefficient == Fq::ZERO {
                continue;
            }
            if let Some(last) = result.last_mut() {
                if last.0 == column {
                    last.1 += coefficient;
                    if last.1 == Fq::ZERO {
                        result.pop();
                    }
                    continue;
                }
            }
            result.push((column, coefficient));
        }
        Self { entries: result }
    }

    fn substitute(&self, inputs: &[Self]) -> Result<Self, ComponentError> {
        let mut entries = Vec::new();
        for (index, coefficient) in &self.entries {
            let input = inputs
                .get(*index)
                .ok_or(ComponentError::Invalid("forward or invalid register reference"))?;
            entries.extend(
                input
                    .entries
                    .iter()
                    .map(|(column, value)| (*column, *coefficient * *value)),
            );
        }
        Ok(Self::canonical(entries))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum PortRole {
    Constant,
    Input,
    Witness,
    Output,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Port {
    name: String,
    #[serde(rename = "role")]
    _role: PortRole,
    start: usize,
    count: usize,
}

type RawForm = Vec<(usize, u64)>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawVariant {
    linear_forms: Vec<RawForm>,
    rows: Vec<[RawForm; MATRIX_PORTS]>,
    output_registers: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawComponent {
    id: String,
    input_count: usize,
    ports: Vec<Port>,
    variants: Vec<RawVariant>,
    definitions: Vec<String>,
    #[serde(rename = "contracts")]
    _contracts: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLibrary {
    format: String,
    version: u64,
    profile: [u64; 8],
    components: Vec<RawComponent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MatrixRow {
    ports: [SparseForm; MATRIX_PORTS],
}

impl MatrixRow {
    pub(crate) fn ports(&self) -> &[SparseForm; MATRIX_PORTS] {
        &self.ports
    }
}

#[derive(Debug)]
pub(crate) struct FormulaVariant {
    input_count: usize,
    rows: Vec<MatrixRow>,
    outputs: Vec<SparseForm>,
}

impl FormulaVariant {
    fn compile(raw: RawVariant, input_count: usize) -> Result<Self, ComponentError> {
        let register_count = input_count
            .checked_add(raw.linear_forms.len())
            .ok_or(ComponentError::Invalid("register count overflow"))?;
        let mut registers: Vec<_> = (0..input_count).map(SparseForm::variable).collect();
        for form in raw.linear_forms {
            if form.iter().any(|(index, _)| *index >= registers.len()) {
                return Err(ComponentError::Invalid("forward or invalid register reference"));
            }
            registers.push(SparseForm::new(form)?.substitute(&registers)?);
        }
        let mut rows = Vec::with_capacity(raw.rows.len());
        for row in raw.rows {
            let mut ports = std::array::from_fn(|_| SparseForm::default());
            for (port, form) in ports.iter_mut().zip(row) {
                if form.iter().any(|(index, _)| *index >= register_count) {
                    return Err(ComponentError::Invalid("row register out of range"));
                }
                *port = SparseForm::new(form)?.substitute(&registers)?;
            }
            rows.push(MatrixRow { ports });
        }
        let outputs = raw
            .output_registers
            .into_iter()
            .map(|index| {
                registers
                    .get(index)
                    .cloned()
                    .ok_or(ComponentError::Invalid("output register out of range"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            input_count,
            rows,
            outputs,
        })
    }

    pub(crate) fn input_count(&self) -> usize {
        self.input_count
    }

    pub(crate) fn row_templates(&self, range: std::ops::Range<usize>) -> Result<&[MatrixRow], ComponentError> {
        self.rows
            .get(range)
            .ok_or(ComponentError::Invalid("row range out of bounds"))
    }

    pub(crate) fn output_templates(&self) -> &[SparseForm] {
        &self.outputs
    }
}

#[derive(Debug)]
pub(crate) struct FormulaComponent {
    id: String,
    variants: Vec<FormulaVariant>,
}

impl FormulaComponent {
    fn compile(raw: RawComponent) -> Result<Self, ComponentError> {
        if raw.id.is_empty() || raw.variants.is_empty() || raw.definitions.is_empty() {
            return Err(ComponentError::Invalid("missing component definition"));
        }
        let mut end = 0usize;
        let mut names = BTreeSet::new();
        for port in &raw.ports {
            if port.name.is_empty() || !names.insert(&port.name) || port.start != end || port.count == 0 {
                return Err(ComponentError::Invalid("port names, order or coverage"));
            }
            end = end
                .checked_add(port.count)
                .ok_or(ComponentError::Invalid("port count overflow"))?;
        }
        if end != raw.input_count {
            return Err(ComponentError::Invalid("incomplete port coverage"));
        }
        let variants = raw
            .variants
            .into_iter()
            .map(|variant| FormulaVariant::compile(variant, raw.input_count))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { id: raw.id, variants })
    }

    pub(crate) fn variant(&self, index: usize) -> Option<&FormulaVariant> {
        self.variants.get(index)
    }
}

/// Parsed formula data. The circuit owner must supply or check the expected
/// library; these structural checks are not a proof of provenance.
#[derive(Debug)]
pub(crate) struct FormulaLibrary {
    components: Vec<FormulaComponent>,
}

impl FormulaLibrary {
    pub(crate) fn from_json(bytes: &[u8]) -> Result<Self, ComponentError> {
        let raw: RawLibrary = serde_json::from_slice(bytes)?;
        if raw.format != "nightstream.matrix-templates" || raw.version != 1 || raw.profile != PROFILE {
            return Err(ComponentError::Invalid("unsupported format, version or profile"));
        }
        if raw.components.is_empty() {
            return Err(ComponentError::Invalid("empty formula library"));
        }
        let mut ids = BTreeSet::new();
        let mut components = Vec::with_capacity(raw.components.len());
        for component in raw.components {
            if !ids.insert(component.id.clone()) {
                return Err(ComponentError::Invalid("duplicate component identifier"));
            }
            components.push(FormulaComponent::compile(component)?);
        }
        Ok(Self { components })
    }

    pub(crate) fn component(&self, id: &str) -> Option<&FormulaComponent> {
        self.components.iter().find(|component| component.id == id)
    }
}

#[cfg(test)]
#[path = "../tests/unit/component_formula_library.rs"]
mod tests;
