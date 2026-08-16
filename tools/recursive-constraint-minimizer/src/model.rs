//! Strict parsing and local replay of cvc5 Goldilocks assignments.

use std::error::Error;
use std::fmt;

use crate::{Row, TypedTargetRow, GOLDILOCKS_MODULUS};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldModel {
    values: Vec<u64>,
}

impl FieldModel {
    pub fn values(&self) -> &[u64] {
        &self.values
    }

    /// Build a model from canonical residues produced by a Rust witness
    /// constructor. The same replay and Lean checks apply to every model,
    /// independent of how the model was found.
    pub fn from_canonical_values(values: Vec<u64>) -> Result<Self, ModelError> {
        if values.is_empty() {
            return Err(ModelError::new("model must contain at least one column"));
        }
        let modulus = GOLDILOCKS_MODULUS
            .parse::<u64>()
            .expect("fixed Goldilocks modulus fits in u64");
        if values.iter().any(|&value| value >= modulus) {
            return Err(ModelError::new("model contains a noncanonical residue"));
        }
        Ok(Self { values })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelError(String);

impl ModelError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ModelError {}

/// Parse all `x_N` definitions from a cvc5 finite-field model.
pub fn parse_model(stdout: &str, column_count: usize) -> Result<FieldModel, ModelError> {
    let expected_columns = (0..column_count).collect::<Vec<_>>();
    parse_model_with_defaults(stdout, &vec![0; column_count], &expected_columns)
}

/// Parse exactly the columns declared by a bounded query and fill every other
/// source column from a complete canonical background assignment.
pub fn parse_model_with_defaults(
    stdout: &str,
    defaults: &[u64],
    expected_columns: &[usize],
) -> Result<FieldModel, ModelError> {
    let column_count = defaults.len();
    if column_count == 0 {
        return Err(ModelError::new("column_count must be positive"));
    }
    let modulus = GOLDILOCKS_MODULUS
        .parse::<u64>()
        .expect("fixed Goldilocks modulus fits in u64");
    if defaults.iter().any(|&value| value >= modulus) {
        return Err(ModelError::new("default assignment contains a noncanonical residue"));
    }
    let mut prior = None;
    for &column in expected_columns {
        if column >= column_count {
            return Err(ModelError::new(format!(
                "expected model column x_{column} is out of range"
            )));
        }
        if prior.is_some_and(|prior| column <= prior) {
            return Err(ModelError::new(
                "expected model columns must be strictly ordered and unique",
            ));
        }
        prior = Some(column);
    }
    let mut assignments = vec![None; column_count];
    let mut tokens = Tokens::new(stdout);
    while let Some(token) = tokens.next() {
        if token == Token::Atom("define-fun") {
            parse_definition(&mut tokens, &mut assignments)?;
        }
    }
    let mut values = defaults.to_vec();
    let mut expected_cursor = 0usize;
    for (column, value) in assignments.into_iter().enumerate() {
        if expected_columns.get(expected_cursor) == Some(&column) {
            values[column] = value.ok_or_else(|| ModelError::new(format!("model does not define x_{column}")))?;
            expected_cursor += 1;
        } else if value.is_some() {
            return Err(ModelError::new(format!(
                "model defines undeclared source column x_{column}"
            )));
        }
    }
    debug_assert_eq!(expected_cursor, expected_columns.len());
    Ok(FieldModel { values })
}

/// Evaluate one exported R1CS equation with a parsed assignment.
pub fn row_is_satisfied(row: &Row, model: &FieldModel) -> Result<bool, ModelError> {
    let a = evaluate(&row.a, model)?;
    let b = evaluate(&row.b, model)?;
    let c = evaluate(&row.c, model)?;
    Ok(multiply(a, b) == c)
}

/// Evaluate one independently defined typed-target equation.
pub fn typed_target_row_is_satisfied(row: &TypedTargetRow, model: &FieldModel) -> Result<bool, ModelError> {
    let a = evaluate(&row.a, model)?;
    let b = evaluate(&row.b, model)?;
    let c = evaluate(&row.c, model)?;
    Ok(multiply(a, b) == c)
}

fn parse_definition(tokens: &mut Tokens<'_>, assignments: &mut [Option<u64>]) -> Result<(), ModelError> {
    let name = match tokens.next() {
        Some(Token::Atom(name)) => name,
        _ => return Err(ModelError::new("malformed define-fun name")),
    };
    let column = match name.strip_prefix("x_") {
        Some(suffix) => {
            let column = suffix
                .parse::<usize>()
                .map_err(|_| ModelError::new(format!("invalid model symbol {name:?}")))?;
            if suffix != column.to_string() {
                return Err(ModelError::new(format!("noncanonical model symbol {name:?}")));
            }
            Some(column)
        }
        None => None,
    };
    let mut depth = 1usize;
    let mut literal = None;
    while depth > 0 {
        match tokens.next() {
            Some(Token::Open) => depth += 1,
            Some(Token::Close) => depth -= 1,
            Some(Token::Atom(atom)) => {
                if let Some(value) = parse_literal(atom)? {
                    if literal.replace(value).is_some() {
                        return Err(ModelError::new(format!(
                            "model definition {name:?} has more than one field literal"
                        )));
                    }
                }
            }
            None => return Err(ModelError::new(format!("unterminated model definition {name:?}"))),
        }
    }
    let Some(column) = column else {
        return Ok(());
    };
    if column >= assignments.len() {
        return Err(ModelError::new(format!("model defines out-of-range column x_{column}")));
    }
    let value =
        literal.ok_or_else(|| ModelError::new(format!("model definition x_{column} has no finite-field literal")))?;
    if assignments[column].replace(value).is_some() {
        return Err(ModelError::new(format!("model defines x_{column} more than once")));
    }
    Ok(())
}

fn parse_literal(atom: &str) -> Result<Option<u64>, ModelError> {
    if let Some(encoded) = atom.strip_prefix("#f") {
        let (value, modulus) = encoded
            .rsplit_once('m')
            .ok_or_else(|| ModelError::new(format!("invalid finite-field literal {atom:?}")))?;
        if modulus != GOLDILOCKS_MODULUS {
            return Err(ModelError::new(format!(
                "finite-field literal {atom:?} has the wrong modulus"
            )));
        }
        return parse_residue(value).map(Some);
    }
    atom.strip_prefix("ff").map(parse_residue).transpose()
}

fn parse_residue(value: &str) -> Result<u64, ModelError> {
    let modulus = GOLDILOCKS_MODULUS
        .parse::<i128>()
        .expect("fixed Goldilocks modulus fits in i128");
    let value = value
        .parse::<i128>()
        .map_err(|_| ModelError::new(format!("invalid finite-field residue {value:?}")))?;
    Ok(value.rem_euclid(modulus) as u64)
}

fn evaluate(terms: &[crate::Term], model: &FieldModel) -> Result<u64, ModelError> {
    let modulus = GOLDILOCKS_MODULUS
        .parse::<u64>()
        .expect("fixed Goldilocks modulus fits in u64");
    let mut sum = 0u64;
    for term in terms {
        let value = model
            .values
            .get(term.column)
            .ok_or_else(|| ModelError::new(format!("row uses missing model column {}", term.column)))?;
        let coefficient = term
            .coefficient
            .parse::<u64>()
            .map_err(|_| ModelError::new(format!("invalid row coefficient {:?}", term.coefficient)))?;
        let product = multiply(coefficient, *value);
        sum = ((u128::from(sum) + u128::from(product)) % u128::from(modulus)) as u64;
    }
    Ok(sum)
}

fn multiply(left: u64, right: u64) -> u64 {
    let modulus = GOLDILOCKS_MODULUS
        .parse::<u128>()
        .expect("fixed Goldilocks modulus fits in u128");
    (u128::from(left) * u128::from(right) % modulus) as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Token<'a> {
    Open,
    Close,
    Atom(&'a str),
}

struct Tokens<'a> {
    input: &'a str,
    cursor: usize,
}

impl<'a> Tokens<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, cursor: 0 }
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.input.as_bytes();
        while self.cursor < bytes.len() && bytes[self.cursor].is_ascii_whitespace() {
            self.cursor += 1;
        }
        if self.cursor == bytes.len() {
            return None;
        }
        match bytes[self.cursor] {
            b'(' => {
                self.cursor += 1;
                Some(Token::Open)
            }
            b')' => {
                self.cursor += 1;
                Some(Token::Close)
            }
            _ => {
                let start = self.cursor;
                while self.cursor < bytes.len()
                    && !bytes[self.cursor].is_ascii_whitespace()
                    && bytes[self.cursor] != b'('
                    && bytes[self.cursor] != b')'
                {
                    self.cursor += 1;
                }
                Some(Token::Atom(&self.input[start..self.cursor]))
            }
        }
    }
}
