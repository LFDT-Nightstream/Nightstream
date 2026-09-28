//! Decodes exported matrix formulas and substitutes their named input ports.
//! Formula data comes from Lean. This module supplies only sparse linear algebra;
//! loading a library does not establish its authority or a complete verifier.

use std::collections::BTreeSet;

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::Deserialize;
use thiserror::Error;

const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;
// The selected Nightstream profile, in the exported profile field order.
const PROFILE: [u64; 8] = [GOLDILOCKS_MODULUS, 2, 16, 65536, 54, 28, 14, 13];
const MATRIX_PORTS: usize = 13;

#[derive(Debug, Error)]
pub enum ComponentError {
    #[error("invalid component JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid component: {0}")]
    Invalid(&'static str),
}

/// Canonical sparse linear form over field-valued columns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SparseForm {
    entries: Vec<(usize, Goldilocks)>,
}

impl SparseForm {
    pub fn new(entries: impl IntoIterator<Item = (usize, u64)>) -> Result<Self, ComponentError> {
        let mut terms = Vec::new();
        for (column, coefficient) in entries {
            if coefficient >= GOLDILOCKS_MODULUS {
                return Err(ComponentError::Invalid("noncanonical field coefficient"));
            }
            terms.push((column, Goldilocks::from_u64(coefficient)));
        }
        Ok(Self::canonical(terms))
    }

    pub fn variable(column: usize) -> Self {
        Self {
            entries: vec![(column, Goldilocks::ONE)],
        }
    }

    pub fn entries(&self) -> impl ExactSizeIterator<Item = (usize, u64)> + '_ {
        self.entries
            .iter()
            .map(|(column, coefficient)| (*column, coefficient.as_canonical_u64()))
    }

    pub fn evaluate(&self, values: &[Goldilocks]) -> Result<Goldilocks, ComponentError> {
        self.entries
            .iter()
            .try_fold(Goldilocks::ZERO, |sum, (column, coefficient)| {
                values
                    .get(*column)
                    .map(|value| sum + *coefficient * *value)
                    .ok_or(ComponentError::Invalid("assignment column out of range"))
            })
    }

    fn canonical(mut entries: Vec<(usize, Goldilocks)>) -> Self {
        entries.sort_unstable_by_key(|entry| entry.0);
        let mut result: Vec<(usize, Goldilocks)> = Vec::with_capacity(entries.len());
        for (column, coefficient) in entries {
            if coefficient == Goldilocks::ZERO {
                continue;
            }
            if let Some(last) = result.last_mut() {
                if last.0 == column {
                    last.1 += coefficient;
                    if last.1 == Goldilocks::ZERO {
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
pub enum PortRole {
    Constant,
    Input,
    Witness,
    Output,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Port {
    name: String,
    role: PortRole,
    start: usize,
    count: usize,
}

impl Port {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn role(&self) -> PortRole {
        self.role
    }
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start..self.start + self.count
    }
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
    contracts: Vec<String>,
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
pub struct MatrixRow {
    ports: [SparseForm; MATRIX_PORTS],
}

impl MatrixRow {
    /// The fourteenth matrix is fixed to zero by the exported production relation.
    pub fn ports(&self) -> &[SparseForm; MATRIX_PORTS] {
        &self.ports
    }
}

#[derive(Debug)]
pub struct FormulaVariant {
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
            // Validate before normalization: even a zero coefficient cannot hide
            // an out-of-range or forward reference in the input format.
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

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    pub(crate) fn input_count(&self) -> usize {
        self.input_count
    }

    /// Linear maps over the original input-port ordinals, before substitution.
    pub(crate) fn row_templates(&self, range: std::ops::Range<usize>) -> Result<&[MatrixRow], ComponentError> {
        self.rows
            .get(range)
            .ok_or(ComponentError::Invalid("row range out of bounds"))
    }

    pub(crate) fn output_templates(&self) -> &[SparseForm] {
        &self.outputs
    }

    fn check_inputs(&self, inputs: &[SparseForm], column_count: usize) -> Result<(), ComponentError> {
        if inputs.len() != self.input_count {
            return Err(ComponentError::Invalid("input port width mismatch"));
        }
        if inputs.iter().any(|input| {
            input
                .entries
                .iter()
                .any(|(column, _)| *column >= column_count)
        }) {
            return Err(ComponentError::Invalid("input column out of range"));
        }
        Ok(())
    }

    pub fn rows(&self, inputs: &[SparseForm], column_count: usize) -> Result<Vec<MatrixRow>, ComponentError> {
        self.rows_range(inputs, column_count, 0..self.rows.len())
    }

    pub fn rows_range(
        &self,
        inputs: &[SparseForm],
        column_count: usize,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<MatrixRow>, ComponentError> {
        let rows = self
            .rows
            .get(range)
            .ok_or(ComponentError::Invalid("row range out of bounds"))?;
        self.check_inputs(inputs, column_count)?;
        rows.iter()
            .map(|row| {
                let mut ports = std::array::from_fn(|_| SparseForm::default());
                for (port, form) in ports.iter_mut().zip(&row.ports) {
                    *port = form.substitute(inputs)?;
                }
                Ok(MatrixRow { ports })
            })
            .collect()
    }

    pub fn outputs(&self, inputs: &[SparseForm], column_count: usize) -> Result<Vec<SparseForm>, ComponentError> {
        self.check_inputs(inputs, column_count)?;
        self.outputs
            .iter()
            .map(|form| form.substitute(inputs))
            .collect()
    }
}

#[derive(Debug)]
pub struct FormulaComponent {
    id: String,
    input_count: usize,
    ports: Vec<Port>,
    variants: Vec<FormulaVariant>,
    definitions: Vec<String>,
    contracts: Vec<String>,
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
        Ok(Self {
            id: raw.id,
            input_count: raw.input_count,
            ports: raw.ports,
            variants,
            definitions: raw.definitions,
            contracts: raw.contracts,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn input_count(&self) -> usize {
        self.input_count
    }
    pub fn ports(&self) -> &[Port] {
        &self.ports
    }
    pub fn variant(&self, index: usize) -> Option<&FormulaVariant> {
        self.variants.get(index)
    }
    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }
    pub fn definitions(&self) -> &[String] {
        &self.definitions
    }
    pub fn contracts(&self) -> &[String] {
        &self.contracts
    }
}

/// Parsed formula data. The circuit owner must supply or check the expected
/// library; these structural checks are not a proof of provenance.
#[derive(Debug)]
pub struct FormulaLibrary {
    components: Vec<FormulaComponent>,
}

impl FormulaLibrary {
    pub fn from_json(bytes: &[u8]) -> Result<Self, ComponentError> {
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

    pub fn component(&self, id: &str) -> Option<&FormulaComponent> {
        self.components.iter().find(|component| component.id == id)
    }

    pub fn components(&self) -> &[FormulaComponent] {
        &self.components
    }
}
