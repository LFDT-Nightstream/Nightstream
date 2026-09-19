use std::{fmt, ops::Range, sync::Arc};

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

use super::{Affine, Expression, R1csRow, Variable};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplicationError {
    DimensionOverflow,
    Allocation,
    VariableOutOfScope(usize),
    OutputDependency(usize),
    PrivateInputCount { expected: usize, actual: usize },
    WitnessLength { expected: usize, actual: usize },
    UnsatisfiedRow(usize),
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
        }
    }
}

impl std::error::Error for ApplicationError {}

/// Builds a four-word state transition from affine and multiplication rows.
/// Local variables are input state, private inputs, output state, then locals.
pub struct ApplicationBuilder {
    private_inputs: Vec<Variable>,
    generated_start: usize,
    variable_count: usize,
    rows: Vec<R1csRow>,
    generated_rows: Vec<usize>,
    recipes: Vec<Arc<Expression>>,
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
            rows: Vec::new(),
            generated_rows: Vec::new(),
            recipes: Vec::new(),
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
        self.rows
            .try_reserve(1)
            .map_err(|_| ApplicationError::Allocation)?;
        self.generated_rows
            .try_reserve(1)
            .map_err(|_| ApplicationError::Allocation)?;
        self.recipes
            .try_reserve(1)
            .map_err(|_| ApplicationError::Allocation)?;
        self.generated_rows.push(self.rows.len());
        self.rows.push(R1csRow {
            a,
            b,
            c: variable.into(),
        });
        self.recipes.push(recipe);
        self.variable_count = next;
        Ok(variable)
    }

    pub fn assert_equal(&mut self, left: Affine, right: Affine) -> Result<(), ApplicationError> {
        self.validate(&left, false)?;
        self.validate(&right, false)?;
        // This direction preserves the existing Lean direct-recipe lowering.
        let row = if left.as_variable().is_some() {
            R1csRow {
                a: right,
                b: Affine::constant(Goldilocks::ONE),
                c: left,
            }
        } else {
            R1csRow {
                a: left - right,
                b: Affine::constant(Goldilocks::ONE),
                c: Affine::constant(Goldilocks::ZERO),
            }
        };
        self.rows
            .try_reserve(1)
            .map_err(|_| ApplicationError::Allocation)?;
        self.rows.push(row);
        Ok(())
    }

    pub fn finish(mut self, outputs: [Affine; 4]) -> Result<ApplicationCircuit, ApplicationError> {
        for expression in &outputs {
            self.validate(expression, true)?;
        }
        for (expression, output) in outputs.iter().zip(self.output_state()) {
            self.assert_equal(expression.clone(), output.into())?;
        }
        Ok(ApplicationCircuit {
            private_inputs: self.private_inputs,
            generated_start: self.generated_start,
            variable_count: self.variable_count,
            rows: self.rows,
            generated_rows: self.generated_rows,
            recipes: self.recipes,
            outputs,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplicationCircuit {
    private_inputs: Vec<Variable>,
    generated_start: usize,
    variable_count: usize,
    rows: Vec<R1csRow>,
    generated_rows: Vec<usize>,
    recipes: Vec<Arc<Expression>>,
    outputs: [Affine; 4],
}

impl ApplicationCircuit {
    pub(crate) fn recipes(&self) -> &[Arc<Expression>] {
        &self.recipes
    }
    pub fn rows(&self) -> &[R1csRow] {
        &self.rows
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
        for (offset, &row_index) in self.generated_rows.iter().enumerate() {
            let row = &self.rows[row_index];
            values[self.generated_start + offset] = row.a.evaluate(&values) * row.b.evaluate(&values);
        }
        let output_state = std::array::from_fn(|lane| self.outputs[lane].evaluate(&values));
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
        for (index, row) in self.rows.iter().enumerate() {
            if !row.holds(values) {
                return Err(ApplicationError::UnsatisfiedRow(index));
            }
        }
        Ok(())
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
