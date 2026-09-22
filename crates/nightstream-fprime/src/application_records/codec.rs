//! Streams exact records and rebuilds private indices on import. Artifact
//! framing and trust in saved bindings belong to the enclosing package.

use std::io::{Read, Write};

use super::{
    index, invalid, read_word, word, ApplicationForm, ApplicationRecipeNode, ApplicationRecords,
    ApplicationRecordsWriter, ApplicationRowHeader, ApplicationTerm, PackageError,
};
use std::ops::ControlFlow;

fn write_word(output: &mut (impl Write + ?Sized), value: u64) -> Result<(), PackageError> {
    output.write_all(&value.to_le_bytes())?;
    Ok(())
}

impl ApplicationRecords {
    pub(crate) fn write_to(&self, output: &mut (impl Write + ?Sized)) -> Result<(), PackageError> {
        write_word(output, word(self.row_count())?)?;
        write_word(output, word(self.recipe_count())?)?;
        for row in 0..self.row_count() {
            let header = self.row_header(row)?;
            for form in 0..3 {
                write_word(output, header.constants[form])?;
                write_word(output, word(header.term_counts[form])?)?;
            }
            let flow = self.visit_terms(row, |term| {
                write_word(output, word(term.variable)?)?;
                write_word(output, term.coefficient)?;
                Ok(ControlFlow::Continue(()))
            })?;
            if flow.is_break() {
                return Err(invalid("application row serialization stopped early"));
            }
        }
        for recipe in 0..self.recipe_count() {
            let (_, count, row) = self.recipe_location(recipe)?;
            write_word(output, word(row)?)?;
            write_word(output, word(count)?)?;
            let flow = self.visit_recipe_nodes(recipe, |node| {
                let (tag, payload) = match node {
                    ApplicationRecipeNode::Variable(variable) => (0, word(variable)?),
                    ApplicationRecipeNode::Constant(value) => (1, value),
                    ApplicationRecipeNode::Add => (2, 0),
                    ApplicationRecipeNode::Multiply => (3, 0),
                };
                write_word(output, tag)?;
                write_word(output, payload)?;
                Ok(ControlFlow::Continue(()))
            })?;
            if flow.is_break() {
                return Err(invalid("application recipe serialization stopped early"));
            }
        }
        Ok(())
    }

    pub(crate) fn read_from(
        input: &mut (impl Read + ?Sized),
        expected_rows: usize,
        expected_recipes: usize,
    ) -> Result<Self, PackageError> {
        let rows = index(read_word(input)?)?;
        let recipes = index(read_word(input)?)?;
        if rows != expected_rows || recipes != expected_recipes {
            return Err(invalid("application stream record counts"));
        }
        let mut writer = ApplicationRecordsWriter::new()?;
        for _ in 0..rows {
            let mut header = ApplicationRowHeader {
                constants: [0; 3],
                term_counts: [0; 3],
            };
            for form in 0..3 {
                header.constants[form] = read_word(input)?;
                header.term_counts[form] = index(read_word(input)?)?;
            }
            let forms = [ApplicationForm::A, ApplicationForm::B, ApplicationForm::C];
            let mut form = 0;
            let mut remaining = header.term_counts;
            let terms = std::iter::from_fn(|| {
                while form < forms.len() && remaining[form] == 0 {
                    form += 1;
                }
                if form == forms.len() {
                    return None;
                }
                remaining[form] -= 1;
                Some((|| {
                    Ok(ApplicationTerm {
                        form: forms[form],
                        variable: index(read_word(input)?)?,
                        coefficient: read_word(input)?,
                    })
                })())
            });
            writer.append_row(header, terms)?;
        }
        for _ in 0..recipes {
            let row = index(read_word(input)?)?;
            let count = index(read_word(input)?)?;
            let nodes = (0..count).map(|_| {
                let tag = read_word(input)?;
                let payload = read_word(input)?;
                match (tag, payload) {
                    (0, variable) => Ok(ApplicationRecipeNode::Variable(index(variable)?)),
                    (1, value) => Ok(ApplicationRecipeNode::Constant(value)),
                    (2, 0) => Ok(ApplicationRecipeNode::Add),
                    (3, 0) => Ok(ApplicationRecipeNode::Multiply),
                    _ => Err(invalid("invalid application stream recipe node")),
                }
            });
            writer.append_recipe(row, nodes)?;
        }
        writer.finish()
    }
}
