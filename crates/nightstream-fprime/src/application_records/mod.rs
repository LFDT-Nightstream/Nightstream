//! Immutable application rows and exact recipe syntax. Append ingestion owns the
//! snapshot; file paths and mutable handles never cross the package boundary.

use std::{fmt, ops::ControlFlow};

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use crate::PackageError;

mod codec;
mod storage;
pub(crate) use storage::PrivateSnapshot;
use storage::{read_word, Reader, RecipeStack, Writer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplicationForm {
    A,
    B,
    C,
}

impl ApplicationForm {
    pub fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::C => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApplicationRowHeader {
    pub constants: [u64; 3],
    pub term_counts: [usize; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApplicationTerm {
    pub form: ApplicationForm,
    pub variable: usize,
    pub coefficient: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplicationRecipeNode {
    Variable(usize),
    Constant(u64),
    Add,
    Multiply,
}

fn invalid(message: &'static str) -> PackageError {
    PackageError::Invalid(message)
}
fn word(value: usize) -> Result<u64, PackageError> {
    u64::try_from(value).map_err(|_| invalid("application index exceeds u64"))
}
fn index(value: u64) -> Result<usize, PackageError> {
    usize::try_from(value).map_err(|_| invalid("application index exceeds address space"))
}
fn add(a: usize, b: usize) -> Result<usize, PackageError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("application record count overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize, PackageError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("application record count overflow"))
}
fn canonical(value: u64) -> Result<Goldilocks, PackageError> {
    if value >= crate::package::GOLDILOCKS_MODULUS {
        return Err(PackageError::NonCanonicalField {
            location: "application record",
            value,
        });
    }
    Ok(Goldilocks::from_u64(value))
}

/// Always writes private records; no size-dependent storage mode is selected.
pub struct ApplicationRecordsWriter {
    data: Writer,
    rows: Writer,
    recipes: Writer,
    row_count: usize,
    recipe_count: usize,
    row_words: usize,
    recipe_words: usize,
    poisoned: bool,
}

impl ApplicationRecordsWriter {
    pub fn new() -> Result<Self, PackageError> {
        Ok(Self {
            data: Writer::new()?,
            rows: Writer::new()?,
            recipes: Writer::new()?,
            row_count: 0,
            recipe_count: 0,
            row_words: 0,
            recipe_words: 0,
            poisoned: false,
        })
    }

    pub fn append_row(
        &mut self,
        header: ApplicationRowHeader,
        terms: impl IntoIterator<Item = Result<ApplicationTerm, PackageError>>,
    ) -> Result<usize, PackageError> {
        if self.poisoned {
            return Err(invalid("application record writer is poisoned"));
        }
        self.poisoned = true;
        let row = self.row_count;
        let next_count = add(row, 1)?;
        let count = header.term_counts.into_iter().try_fold(0, add)?;
        let next_words = add(self.row_words, add(44, mul(12, count)?)?)?;
        word(next_words)?;
        let offset = self.data.position();
        for form in 0..3 {
            let _ = canonical(header.constants[form])?;
            self.data.word(header.constants[form])?;
            self.data.word(word(header.term_counts[form])?)?;
        }
        let mut terms = terms.into_iter();
        for form in [ApplicationForm::A, ApplicationForm::B, ApplicationForm::C] {
            for _ in 0..header.term_counts[form.index()] {
                let term = terms
                    .next()
                    .ok_or_else(|| invalid("application row has missing terms"))??;
                if term.form != form {
                    return Err(invalid("application terms are not in form order"));
                }
                let _ = canonical(term.coefficient)?;
                self.data.word(word(term.variable)?)?;
                self.data.word(term.coefficient)?;
            }
        }
        if terms.next().transpose()?.is_some() {
            return Err(invalid("application row has extra terms"));
        }
        self.rows.word(offset)?;
        self.row_count = next_count;
        self.row_words = next_words;
        self.poisoned = false;
        Ok(row)
    }

    pub fn append_recipe(
        &mut self,
        generated_row: usize,
        nodes: impl IntoIterator<Item = Result<ApplicationRecipeNode, PackageError>>,
    ) -> Result<usize, PackageError> {
        if self.poisoned {
            return Err(invalid("application record writer is poisoned"));
        }
        self.poisoned = true;
        if generated_row >= self.row_count {
            return Err(invalid("application recipe row is out of range"));
        }
        let recipe = self.recipe_count;
        let next_count = add(recipe, 1)?;
        let offset = self.data.position();
        let mut count = 0;
        let mut pending = 1;
        let mut words = self.recipe_words;
        for node in nodes {
            let node = node?;
            if pending == 0 {
                return Err(invalid("application recipe has extra nodes"));
            }
            pending -= 1;
            let (tag, payload) = match node {
                ApplicationRecipeNode::Variable(variable) => (0, word(variable)?),
                ApplicationRecipeNode::Constant(value) => {
                    let _ = canonical(value)?;
                    (1, value)
                }
                ApplicationRecipeNode::Add => {
                    pending = add(pending, 2)?;
                    (2, 0)
                }
                ApplicationRecipeNode::Multiply => {
                    pending = add(pending, 2)?;
                    (3, 0)
                }
            };
            self.data.word(tag)?;
            self.data.word(payload)?;
            words = add(words, if tag < 2 { 12 } else { 8 })?;
            count = add(count, 1)?;
        }
        if pending != 0 {
            return Err(invalid("application recipe has missing nodes"));
        }
        word(words)?;
        for value in [offset, word(count)?, word(generated_row)?] {
            self.recipes.word(value)?;
        }
        self.recipe_count = next_count;
        self.recipe_words = words;
        self.poisoned = false;
        Ok(recipe)
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }
    pub fn recipe_count(&self) -> usize {
        self.recipe_count
    }

    pub fn finish(self) -> Result<ApplicationRecords, PackageError> {
        if self.poisoned {
            return Err(invalid("application record writer is poisoned"));
        }
        Ok(ApplicationRecords {
            data: self.data.finish()?,
            rows: self.rows.finish()?,
            recipes: self.recipes.finish()?,
            row_count: self.row_count,
            recipe_count: self.recipe_count,
            row_words: self.row_words,
            recipe_words: self.recipe_words,
        })
    }

    #[cfg(test)]
    fn buffer_bytes(&self) -> usize {
        self.data.buffer_bytes() + self.rows.buffer_bytes() + self.recipes.buffer_bytes()
    }
}

/// Sealed records are shared through `Arc`; readers own independent cursors.
pub struct ApplicationRecords {
    data: Reader,
    rows: Reader,
    recipes: Reader,
    row_count: usize,
    recipe_count: usize,
    row_words: usize,
    recipe_words: usize,
}

impl fmt::Debug for ApplicationRecords {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApplicationRecords")
            .field("rows", &self.row_count)
            .field("recipes", &self.recipe_count)
            .finish()
    }
}

impl ApplicationRecords {
    pub fn row_count(&self) -> usize {
        self.row_count
    }
    pub fn recipe_count(&self) -> usize {
        self.recipe_count
    }
    pub fn row_preimage_word_count(&self) -> usize {
        self.row_words
    }
    pub fn recipe_preimage_word_count(&self) -> usize {
        self.recipe_words
    }

    fn row_offset(&self, row: usize) -> Result<u64, PackageError> {
        if row >= self.row_count {
            return Err(invalid("application row is out of range"));
        }
        self.rows.word(word(mul(row, size_of::<u64>())?)?)
    }

    pub fn row_header(&self, row: usize) -> Result<ApplicationRowHeader, PackageError> {
        let mut input = self
            .data
            .cursor(self.row_offset(row)?, 6 * size_of::<u64>() as u64)?;
        let mut header = ApplicationRowHeader {
            constants: [0; 3],
            term_counts: [0; 3],
        };
        for form in 0..3 {
            header.constants[form] = read_word(&mut input)?;
            let _ = canonical(header.constants[form])?;
            header.term_counts[form] = index(read_word(&mut input)?)?;
        }
        Ok(header)
    }

    pub fn visit_terms(
        &self,
        row: usize,
        mut visit: impl FnMut(ApplicationTerm) -> Result<ControlFlow<()>, PackageError>,
    ) -> Result<ControlFlow<()>, PackageError> {
        let header = self.row_header(row)?;
        let count = header.term_counts.into_iter().try_fold(0, add)?;
        let start = self
            .row_offset(row)?
            .checked_add(6 * size_of::<u64>() as u64)
            .ok_or_else(|| invalid("application term offset overflow"))?;
        let mut input = self
            .data
            .cursor(start, word(mul(count, 2 * size_of::<u64>())?)?)?;
        for form in [ApplicationForm::A, ApplicationForm::B, ApplicationForm::C] {
            for _ in 0..header.term_counts[form.index()] {
                let variable = index(read_word(&mut input)?)?;
                let coefficient = read_word(&mut input)?;
                let _ = canonical(coefficient)?;
                if visit(ApplicationTerm {
                    form,
                    variable,
                    coefficient,
                })?
                .is_break()
                {
                    return Ok(ControlFlow::Break(()));
                }
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    fn recipe_location(&self, recipe: usize) -> Result<(u64, usize, usize), PackageError> {
        if recipe >= self.recipe_count {
            return Err(invalid("application recipe is out of range"));
        }
        let mut input = self
            .recipes
            .cursor(word(mul(recipe, 3 * size_of::<u64>())?)?, 3 * size_of::<u64>() as u64)?;
        Ok((
            read_word(&mut input)?,
            index(read_word(&mut input)?)?,
            index(read_word(&mut input)?)?,
        ))
    }

    pub fn recipe_row(&self, recipe: usize) -> Result<usize, PackageError> {
        Ok(self.recipe_location(recipe)?.2)
    }

    pub fn visit_recipe_nodes(
        &self,
        recipe: usize,
        mut visit: impl FnMut(ApplicationRecipeNode) -> Result<ControlFlow<()>, PackageError>,
    ) -> Result<ControlFlow<()>, PackageError> {
        let (offset, count, _) = self.recipe_location(recipe)?;
        let mut input = self
            .data
            .cursor(offset, word(mul(count, 2 * size_of::<u64>())?)?)?;
        let mut pending = 1;
        for _ in 0..count {
            if pending == 0 {
                return Err(invalid("application recipe has extra stored nodes"));
            }
            pending -= 1;
            let tag = read_word(&mut input)?;
            let payload = read_word(&mut input)?;
            let node = match (tag, payload) {
                (0, variable) => ApplicationRecipeNode::Variable(index(variable)?),
                (1, value) => {
                    let _ = canonical(value)?;
                    ApplicationRecipeNode::Constant(value)
                }
                (2, 0) => {
                    pending = add(pending, 2)?;
                    ApplicationRecipeNode::Add
                }
                (3, 0) => {
                    pending = add(pending, 2)?;
                    ApplicationRecipeNode::Multiply
                }
                _ => return Err(invalid("invalid stored application recipe node")),
            };
            if visit(node)?.is_break() {
                return Ok(ControlFlow::Break(()));
            }
        }
        if pending != 0 {
            return Err(invalid("application recipe has missing stored nodes"));
        }
        Ok(ControlFlow::Continue(()))
    }

    pub fn evaluate_form(
        &self,
        row: usize,
        form: ApplicationForm,
        mut value: impl FnMut(usize) -> Result<u64, PackageError>,
    ) -> Result<u64, PackageError> {
        let header = self.row_header(row)?;
        let mut result = canonical(header.constants[form.index()])?;
        let preceding = header.term_counts[..form.index()]
            .iter()
            .copied()
            .try_fold(0, add)?;
        let start = self
            .row_offset(row)?
            .checked_add(word(add(6 * size_of::<u64>(), mul(preceding, 2 * size_of::<u64>())?)?)?)
            .ok_or_else(|| invalid("application form offset overflow"))?;
        let mut input = self.data.cursor(
            start,
            word(mul(header.term_counts[form.index()], 2 * size_of::<u64>())?)?,
        )?;
        for _ in 0..header.term_counts[form.index()] {
            let variable = index(read_word(&mut input)?)?;
            let coefficient = canonical(read_word(&mut input)?)?;
            result += coefficient * canonical(value(variable)?)?;
        }
        Ok(result.as_canonical_u64())
    }

    pub fn evaluate_recipe(
        &self,
        recipe: usize,
        value: impl FnMut(usize) -> Result<u64, PackageError>,
    ) -> Result<u64, PackageError> {
        RecipeEvaluator::new(self)?.evaluate(recipe, value)
    }
}

/// One batch's disk scratch; the authoritative records stay immutable.
pub(crate) struct RecipeEvaluator<'a> {
    records: &'a ApplicationRecords,
    stack: RecipeStack,
}

impl<'a> RecipeEvaluator<'a> {
    pub(crate) fn new(records: &'a ApplicationRecords) -> Result<Self, PackageError> {
        Ok(Self {
            records,
            stack: RecipeStack::new()?,
        })
    }

    pub(crate) fn evaluate(
        &mut self,
        recipe: usize,
        mut value: impl FnMut(usize) -> Result<u64, PackageError>,
    ) -> Result<u64, PackageError> {
        self.stack.reset();
        let stack = &mut self.stack;
        let mut result = None;
        let flow = self.records.visit_recipe_nodes(recipe, |node| {
            let mut current = match node {
                ApplicationRecipeNode::Add | ApplicationRecipeNode::Multiply => {
                    stack.push([u64::from(node == ApplicationRecipeNode::Multiply), 0, 0])?;
                    return Ok(ControlFlow::Continue(()));
                }
                ApplicationRecipeNode::Variable(variable) => canonical(value(variable)?)?,
                ApplicationRecipeNode::Constant(constant) => canonical(constant)?,
            };
            while let Some([operation, present, left]) = stack.pop()? {
                if present == 0 {
                    stack.push([operation, 1, current.as_canonical_u64()])?;
                    return Ok(ControlFlow::Continue(()));
                }
                let left = canonical(left)?;
                current = if operation == 0 { left + current } else { left * current };
            }
            if result.replace(current.as_canonical_u64()).is_some() {
                return Err(invalid("application recipe produced extra values"));
            }
            Ok(ControlFlow::Continue(()))
        })?;
        if flow.is_break() {
            return Err(invalid("application recipe evaluation stopped early"));
        }
        result.ok_or_else(|| invalid("application recipe produced no value"))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/application_records.rs"]
mod tests;
