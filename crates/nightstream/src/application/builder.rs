use std::{fmt, ops::Range, sync::Arc};

use nightstream_fprime::{
    ApplicationForm, ApplicationRecords, ApplicationRecordsWriter, ApplicationRowHeader, ApplicationTerm, PackageError,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::{Affine, Expression, R1csRow, Variable};

#[derive(Debug)]
pub enum ApplicationError {
    DimensionOverflow,
    Allocation,
    VariableOutOfScope(usize),
    OutputDependency(usize),
    PrivateInputCount { expected: usize, actual: usize },
    WitnessLength { expected: usize, actual: usize },
    UnsatisfiedRow(usize),
    Records(PackageError),
}

impl fmt::Display for ApplicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOverflow => write!(f, "application dimension overflow"),
            Self::Allocation => write!(f, "application allocation failed"),
            Self::VariableOutOfScope(index) => write!(f, "application variable {index} is out of scope"),
            Self::OutputDependency(index) => write!(f, "application witness reads output variable {index}"),
            Self::PrivateInputCount { expected, actual } => {
                write!(f, "application needs {expected} private inputs; got {actual}")
            }
            Self::WitnessLength { expected, actual } => {
                write!(f, "application needs {expected} witness values; got {actual}")
            }
            Self::UnsatisfiedRow(index) => write!(f, "application row {index} is not satisfied"),
            Self::Records(error) => write!(f, "application records: {error}"),
        }
    }
}

impl std::error::Error for ApplicationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Records(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PackageError> for ApplicationError {
    fn from(error: PackageError) -> Self {
        Self::Records(error)
    }
}

/// Builds a four-word state transition from affine and multiplication rows.
/// Local variables are input state, private inputs, output state, then locals.
pub struct ApplicationBuilder {
    private_inputs: Vec<Variable>,
    generated_start: usize,
    variable_count: usize,
    records: ApplicationRecordsWriter,
}

impl ApplicationBuilder {
    pub fn new(private_input_count: usize) -> Result<Self, ApplicationError> {
        let generated_start = private_input_count
            .checked_add(8)
            .ok_or(ApplicationError::DimensionOverflow)?;
        let mut private_inputs = Vec::new();
        private_inputs
            .try_reserve_exact(private_input_count)
            .map_err(|_| ApplicationError::Allocation)?;
        private_inputs.extend((4..generated_start - 4).map(Variable));
        Ok(Self {
            private_inputs,
            generated_start,
            variable_count: generated_start,
            records: ApplicationRecordsWriter::new()?,
        })
    }

    pub fn input_state(&self) -> [Variable; 4] {
        std::array::from_fn(Variable)
    }

    pub fn private_inputs(&self) -> &[Variable] {
        &self.private_inputs
    }

    pub fn output_state(&self) -> [Variable; 4] {
        std::array::from_fn(|lane| Variable(self.generated_start - 4 + lane))
    }

    fn validate(&self, expression: &Affine, causal: bool) -> Result<(), ApplicationError> {
        expression.expression.validate(
            self.variable_count,
            self.generated_start - 4..self.generated_start,
            causal,
        )
    }

    pub fn affine(&mut self, value: Affine) -> Result<Variable, ApplicationError> {
        let recipe = value.expression.clone();
        self.generate(value, Affine::constant(Goldilocks::ONE), recipe)
    }

    pub fn multiply(&mut self, a: Affine, b: Affine) -> Result<Variable, ApplicationError> {
        let recipe = Arc::new(Expression::Multiply(a.expression.clone(), b.expression.clone()));
        self.generate(a, b, recipe)
    }

    fn generate(&mut self, a: Affine, b: Affine, recipe: Arc<Expression>) -> Result<Variable, ApplicationError> {
        self.validate(&a, true)?;
        self.validate(&b, true)?;
        let next = self
            .variable_count
            .checked_add(1)
            .ok_or(ApplicationError::DimensionOverflow)?;
        let variable = Variable(self.variable_count);
        let row = self.append_row(&R1csRow {
            a,
            b,
            c: variable.into(),
        })?;
        self.records.append_recipe(row, recipe.nodes().map(Ok))?;
        self.variable_count = next;
        Ok(variable)
    }

    pub fn assert_equal(&mut self, left: Affine, right: Affine) -> Result<(), ApplicationError> {
        self.equality_row(left, right).map(|_| ())
    }

    fn equality_row(&mut self, left: Affine, right: Affine) -> Result<OutputRow, ApplicationError> {
        self.validate(&left, false)?;
        self.validate(&right, false)?;
        // This direction preserves the existing Lean direct-recipe lowering.
        let (row, form) = if left.as_variable().is_some() {
            (
                R1csRow {
                    a: right,
                    b: Affine::constant(Goldilocks::ONE),
                    c: left,
                },
                ApplicationForm::C,
            )
        } else {
            (
                R1csRow {
                    a: left - right,
                    b: Affine::constant(Goldilocks::ONE),
                    c: Affine::constant(Goldilocks::ZERO),
                },
                ApplicationForm::A,
            )
        };
        Ok(OutputRow {
            row: self.append_row(&row)?,
            form,
        })
    }

    fn append_row(&mut self, row: &R1csRow) -> Result<usize, ApplicationError> {
        let forms = [&row.a, &row.b, &row.c];
        let header = ApplicationRowHeader {
            constants: forms.map(|form| form.constant_term().as_canonical_u64()),
            term_counts: [
                row.a.expression.term_count()?,
                row.b.expression.term_count()?,
                row.c.expression.term_count()?,
            ],
        };
        let terms = [ApplicationForm::A, ApplicationForm::B, ApplicationForm::C]
            .into_iter()
            .zip(forms)
            .flat_map(|(form, value)| {
                value
                    .expression
                    .ordered_terms()
                    .map(move |(variable, coefficient)| {
                        Ok(ApplicationTerm {
                            form,
                            variable: variable.index(),
                            coefficient: coefficient.as_canonical_u64(),
                        })
                    })
            });
        Ok(self.records.append_row(header, terms)?)
    }

    pub fn finish(mut self, outputs: [Affine; 4]) -> Result<ApplicationCircuit, ApplicationError> {
        for expression in &outputs {
            self.validate(expression, true)?;
        }
        let mut output_rows = [OutputRow {
            row: 0,
            form: ApplicationForm::A,
        }; 4];
        for ((slot, expression), output) in output_rows.iter_mut().zip(outputs).zip(self.output_state()) {
            *slot = self.equality_row(expression, output.into())?;
        }
        Ok(ApplicationCircuit {
            private_inputs: self.private_inputs.into(),
            generated_start: self.generated_start,
            variable_count: self.variable_count,
            records: Arc::new(self.records.finish()?),
            outputs: output_rows,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct OutputRow {
    row: usize,
    form: ApplicationForm,
}

#[derive(Clone, Debug)]
pub struct ApplicationCircuit {
    private_inputs: Arc<[Variable]>,
    generated_start: usize,
    variable_count: usize,
    records: Arc<ApplicationRecords>,
    outputs: [OutputRow; 4],
}

impl ApplicationCircuit {
    pub(crate) fn prepared_output_forms(&self) -> Result<[u8; 4], ApplicationError> {
        let first = self
            .row_count()
            .checked_sub(4)
            .ok_or(PackageError::Invalid("prepared output rows"))?;
        let mut forms = [0; 4];
        for (lane, output) in self.outputs.iter().enumerate() {
            if output.row != first + lane {
                return Err(PackageError::Invalid("prepared output row order").into());
            }
            forms[lane] = match output.form {
                ApplicationForm::A => 0,
                ApplicationForm::C => 2,
                ApplicationForm::B => return Err(PackageError::Invalid("prepared output form").into()),
            };
        }
        Ok(forms)
    }

    pub(crate) fn from_prepared(
        package: &nightstream_fprime::LoadedPerApplicationPackage,
        forms: [u8; 4],
    ) -> Result<Self, ApplicationError> {
        let records = Arc::clone(
            package
                .application_records()
                .ok_or(PackageError::Invalid("prepared native records"))?,
        );
        let first = records
            .row_count()
            .checked_sub(4)
            .ok_or(PackageError::Invalid("prepared output rows"))?;
        let private_count = package.application().witness_word_count();
        let generated_start = private_count
            .checked_add(8)
            .ok_or(ApplicationError::DimensionOverflow)?;
        let variable_count = generated_start
            .checked_add(records.recipe_count())
            .ok_or(ApplicationError::DimensionOverflow)?;
        let mut outputs = [OutputRow {
            row: 0,
            form: ApplicationForm::A,
        }; 4];
        for (lane, tag) in forms.into_iter().enumerate() {
            outputs[lane] = OutputRow {
                row: first + lane,
                form: match tag {
                    0 => ApplicationForm::A,
                    2 => ApplicationForm::C,
                    _ => return Err(PackageError::Invalid("prepared output form").into()),
                },
            };
        }
        let mut private_inputs = Vec::new();
        private_inputs
            .try_reserve_exact(private_count)
            .map_err(|_| ApplicationError::Allocation)?;
        private_inputs.extend((4..generated_start - 4).map(Variable));
        Ok(Self {
            private_inputs: private_inputs.into(),
            generated_start,
            variable_count,
            records,
            outputs,
        })
    }

    /// Immutable original terms and recipe syntax, read with fallible visitors.
    pub fn records(&self) -> &Arc<ApplicationRecords> {
        &self.records
    }
    pub fn row_count(&self) -> usize {
        self.records.row_count()
    }
    pub fn private_input_count(&self) -> usize {
        self.private_inputs.len()
    }
    pub fn variable_count(&self) -> usize {
        self.variable_count
    }
    pub fn generated_range(&self) -> Range<usize> {
        self.generated_start..self.variable_count
    }
    pub fn input_state(&self) -> [Variable; 4] {
        std::array::from_fn(Variable)
    }
    pub fn private_inputs(&self) -> &[Variable] {
        &self.private_inputs
    }
    pub fn output_state(&self) -> [Variable; 4] {
        std::array::from_fn(|lane| Variable(self.generated_start - 4 + lane))
    }

    pub fn execute(
        &self,
        input: [Goldilocks; 4],
        private_inputs: &[Goldilocks],
    ) -> Result<ApplicationWitness, ApplicationError> {
        if private_inputs.len() != self.private_inputs.len() {
            return Err(ApplicationError::PrivateInputCount {
                expected: self.private_inputs.len(),
                actual: private_inputs.len(),
            });
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.variable_count)
            .map_err(|_| ApplicationError::Allocation)?;
        values.resize(self.variable_count, Goldilocks::ZERO);
        values[..4].copy_from_slice(&input);
        values[4..self.generated_start - 4].copy_from_slice(private_inputs);
        for offset in 0..self.records.recipe_count() {
            let row = self.records.recipe_row(offset)?;
            values[self.generated_start + offset] =
                self.evaluate(row, ApplicationForm::A, &values)? * self.evaluate(row, ApplicationForm::B, &values)?;
        }
        // All output variables remain zero here. The non-variable lowering's
        // A form is original_expression - output_variable, so it has the same value.
        let mut output_state = [Goldilocks::ZERO; 4];
        for (value, output) in output_state.iter_mut().zip(self.outputs) {
            *value = self.evaluate(output.row, output.form, &values)?;
        }
        values[self.generated_start - 4..self.generated_start].copy_from_slice(&output_state);
        self.check(&values)?;
        Ok(ApplicationWitness { values, output_state })
    }

    pub fn check(&self, values: &[Goldilocks]) -> Result<(), ApplicationError> {
        if values.len() != self.variable_count {
            return Err(ApplicationError::WitnessLength {
                expected: self.variable_count,
                actual: values.len(),
            });
        }
        for index in 0..self.row_count() {
            if self.evaluate(index, ApplicationForm::A, values)? * self.evaluate(index, ApplicationForm::B, values)?
                != self.evaluate(index, ApplicationForm::C, values)?
            {
                return Err(ApplicationError::UnsatisfiedRow(index));
            }
        }
        Ok(())
    }

    fn evaluate(
        &self,
        row: usize,
        form: ApplicationForm,
        values: &[Goldilocks],
    ) -> Result<Goldilocks, ApplicationError> {
        let value = self.records.evaluate_form(row, form, |index| {
            values
                .get(index)
                .map(PrimeField64::as_canonical_u64)
                .ok_or(PackageError::Invalid("application record variable out of scope"))
        })?;
        Ok(Goldilocks::from_u64(value))
    }
}

#[derive(Clone, Debug)]
pub struct ApplicationWitness {
    values: Vec<Goldilocks>,
    output_state: [Goldilocks; 4],
}

impl ApplicationWitness {
    pub fn values(&self) -> &[Goldilocks] {
        &self.values
    }
    pub fn output_state(&self) -> [Goldilocks; 4] {
        self.output_state
    }
}
