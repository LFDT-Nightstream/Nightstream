//! Typed input boundary for one normalized Goldilocks R1CS slice.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const PROBLEM_SCHEMA: &str = "nightstream/r1cs-redundancy-problem/v3";
pub const GOLDILOCKS_MODULUS: &str = "18446744069414584321";

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Local,
    Branch,
    Lifecycle,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Source {
    pub profile: String,
    pub artifact_digest: String,
    pub scope: Scope,
    pub total_rows: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Term {
    pub column: usize,
    /// Canonical decimal residue in `[1, GOLDILOCKS_MODULUS)`.
    pub coefficient: String,
}

pub type LinearCombination = Vec<Term>;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Row {
    pub id: String,
    pub source_index: usize,
    pub family: String,
    pub a: LinearCombination,
    pub b: LinearCombination,
    pub c: LinearCombination,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Problem {
    pub schema: String,
    pub source: Source,
    pub field_modulus: String,
    pub column_count: usize,
    pub constant_one_column: usize,
    /// Exclusive end of the normalized public-column prefix.
    pub public_input_count: usize,
    /// Families for which this input contains every source row owned by the family.
    pub complete_families: Vec<String>,
    pub rows: Vec<Row>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Selection {
    Row(String),
    Family(String),
}

pub(crate) struct Partition<'a> {
    pub retained: Vec<(usize, &'a Row)>,
    pub removed: Vec<(usize, &'a Row)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProblemError(String);

impl ProblemError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ProblemError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ProblemError {}

impl Problem {
    pub fn validate(&self) -> Result<(), ProblemError> {
        if self.schema != PROBLEM_SCHEMA {
            return Err(ProblemError::new(format!(
                "unsupported problem schema {:?}; expected {PROBLEM_SCHEMA:?}",
                self.schema
            )));
        }
        if self.field_modulus != GOLDILOCKS_MODULUS {
            return Err(ProblemError::new(format!(
                "field modulus must be the production Goldilocks modulus {GOLDILOCKS_MODULUS}"
            )));
        }
        if self.column_count == 0 {
            return Err(ProblemError::new("column_count must be positive"));
        }
        if self.constant_one_column >= self.column_count {
            return Err(ProblemError::new("constant_one_column is out of range"));
        }
        if self.public_input_count == 0 || self.public_input_count > self.column_count {
            return Err(ProblemError::new("public_input_count is out of range"));
        }
        if self.constant_one_column >= self.public_input_count {
            return Err(ProblemError::new(
                "constant_one_column must be in the public-column prefix",
            ));
        }
        if self.source.profile.trim().is_empty() {
            return Err(ProblemError::new("source.profile must not be empty"));
        }
        if self.source.artifact_digest.trim().is_empty() {
            return Err(ProblemError::new("source.artifact_digest must not be empty"));
        }
        if self.source.total_rows == 0 {
            return Err(ProblemError::new("source.total_rows must be positive"));
        }
        if self.rows.is_empty() {
            return Err(ProblemError::new("the problem must contain at least one row"));
        }

        let mut prior_family: Option<&str> = None;
        for family in &self.complete_families {
            if family.trim().is_empty() {
                return Err(ProblemError::new("complete_families contains an empty name"));
            }
            if prior_family.is_some_and(|prior| family.as_str() <= prior) {
                return Err(ProblemError::new(
                    "complete_families must be strictly ordered and unique",
                ));
            }
            prior_family = Some(family);
        }

        let mut ids = HashSet::with_capacity(self.rows.len());
        let mut prior_source_index = None;
        let mut seen_families = HashSet::new();
        for (row_index, row) in self.rows.iter().enumerate() {
            if row.id.trim().is_empty() {
                return Err(ProblemError::new(format!("row {row_index} has an empty id")));
            }
            if !ids.insert(row.id.as_str()) {
                return Err(ProblemError::new(format!("duplicate row id {:?}", row.id)));
            }
            if row.family.trim().is_empty() {
                return Err(ProblemError::new(format!("row {:?} has an empty family", row.id)));
            }
            if row.source_index >= self.source.total_rows {
                return Err(ProblemError::new(format!(
                    "row {:?} has out-of-range source_index {}",
                    row.id, row.source_index
                )));
            }
            if prior_source_index.is_some_and(|prior| row.source_index <= prior) {
                return Err(ProblemError::new(
                    "rows must be strictly ordered and unique by source_index",
                ));
            }
            prior_source_index = Some(row.source_index);
            seen_families.insert(row.family.as_str());
            self.validate_linear_combination(row, "a", &row.a)?;
            self.validate_linear_combination(row, "b", &row.b)?;
            self.validate_linear_combination(row, "c", &row.c)?;
        }
        for family in &self.complete_families {
            if !seen_families.contains(family.as_str()) {
                return Err(ProblemError::new(format!(
                    "complete family {family:?} has no exported rows"
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn partition(&self, selection: &Selection) -> Result<Partition<'_>, ProblemError> {
        self.validate()?;
        if let Selection::Family(family) = selection {
            if self.complete_families.binary_search(family).is_err() {
                return Err(ProblemError::new(format!(
                    "family {family:?} is not complete in this input"
                )));
            }
        }
        let mut retained = Vec::with_capacity(self.rows.len());
        let mut removed = Vec::new();
        for (index, row) in self.rows.iter().enumerate() {
            if selection.matches(row) {
                removed.push((index, row));
            } else {
                retained.push((index, row));
            }
        }
        if removed.is_empty() {
            return Err(ProblemError::new(format!(
                "selection {} does not match a row",
                selection.description()
            )));
        }
        Ok(Partition { retained, removed })
    }

    fn validate_linear_combination(
        &self,
        row: &Row,
        side: &str,
        terms: &LinearCombination,
    ) -> Result<(), ProblemError> {
        let modulus = GOLDILOCKS_MODULUS
            .parse::<u64>()
            .expect("the fixed Goldilocks modulus is a u64");
        let mut prior_column = None;
        for term in terms {
            if term.column >= self.column_count {
                return Err(ProblemError::new(format!(
                    "row {:?} side {side} uses out-of-range column {}",
                    row.id, term.column
                )));
            }
            if prior_column.is_some_and(|prior| term.column <= prior) {
                return Err(ProblemError::new(format!(
                    "row {:?} side {side} is not strictly ordered by column",
                    row.id
                )));
            }
            prior_column = Some(term.column);
            let coefficient = term.coefficient.parse::<u64>().map_err(|_| {
                ProblemError::new(format!(
                    "row {:?} side {side} has a non-decimal coefficient {:?}",
                    row.id, term.coefficient
                ))
            })?;
            if coefficient == 0 || coefficient >= modulus || term.coefficient != coefficient.to_string() {
                return Err(ProblemError::new(format!(
                    "row {:?} side {side} has a noncanonical coefficient {:?}",
                    row.id, term.coefficient
                )));
            }
        }
        Ok(())
    }
}

impl Selection {
    pub fn description(&self) -> String {
        match self {
            Self::Row(id) => format!("row {id:?}"),
            Self::Family(family) => format!("family {family:?}"),
        }
    }

    fn matches(&self, row: &Row) -> bool {
        match self {
            Self::Row(id) => row.id == *id,
            Self::Family(family) => row.family == *family,
        }
    }
}
