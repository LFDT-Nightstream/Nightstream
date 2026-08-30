//! Random-access reconstruction of identity-checked package R1CS rows.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

use super::matrix_program::{Entry, SourceCombination, SourceRow};
use super::{
    ColumnRef, CompactRowInvocation, HashChain, LoadedPackage, PackageError, PermutationInvocation,
    PermutationTemplate, ScheduledInvocation, SparseCombination, SparseRow, TemplateCombination, TemplateRow,
    WitnessInstruction,
};

#[cfg(test)]
#[path = "../../tests/unit/source_row.rs"]
mod source_row_tests;

impl LoadedPackage {
    pub(super) fn source_row(&self, row_index: usize) -> Result<SourceRow, PackageError> {
        if row_index >= self.layout.row_count {
            return Err(PackageError::Invalid("matrix source row index"));
        }

        let mut found = None;
        if let Ok(index) = self
            .assertion_rows
            .binary_search_by_key(&row_index, |row| row.row_index)
        {
            merge_row(&mut found, source_assertion(&self.assertion_rows[index]))?;
        }
        if let Ok(index) = self
            .witness_instructions
            .binary_search_by_key(&row_index, |instruction| instruction.row_index)
        {
            merge_row(&mut found, source_witness(&self.witness_instructions[index]))?;
        }
        if let Some((chain, ordinal, local_row)) = locate_hash_row(self, row_index)? {
            merge_row(
                &mut found,
                source_template_row(
                    &self.permutation,
                    &self.permutation.rows[local_row],
                    ScheduledInvocation::Hash {
                        chain,
                        ordinal,
                        row_start: chain.row_start + ordinal * self.permutation.rows.len(),
                        witness_start: chain.witness_start + ordinal * self.permutation.local_column_count,
                    },
                )?,
            )?;
        }
        if let Some((invocation, local_row)) = locate_permutation_row(self, row_index)? {
            merge_row(
                &mut found,
                source_template_row(
                    &self.permutation,
                    &self.permutation.rows[local_row],
                    ScheduledInvocation::Explicit(invocation),
                )?,
            )?;
        }
        if let Some((invocation, local_row)) = locate_compact_row(self, row_index)? {
            merge_row(&mut found, source_compact_row(self, invocation, local_row))?;
        }

        found.ok_or(PackageError::Invalid("missing matrix source row"))
    }
}

fn merge_row(found: &mut Option<SourceRow>, candidate: SourceRow) -> Result<(), PackageError> {
    if found.replace(candidate).is_some() {
        return Err(PackageError::Invalid("duplicate matrix source row"));
    }
    Ok(())
}

fn locate_hash_row(
    package: &LoadedPackage,
    row_index: usize,
) -> Result<Option<(HashChain, usize, usize)>, PackageError> {
    let template_rows = package.permutation.rows.len();
    if template_rows == 0 {
        return Err(PackageError::Invalid("matrix source permutation template"));
    }
    let Some(position) = package
        .hash_chains
        .partition_point(|chain| chain.row_start <= row_index)
        .checked_sub(1)
    else {
        return Ok(None);
    };
    let chain = package.hash_chains[position];
    let invocation_count = chain
        .absorb_count
        .checked_add(1)
        .ok_or(PackageError::Invalid("matrix source hash count"))?;
    let row_count = invocation_count
        .checked_mul(template_rows)
        .ok_or(PackageError::Invalid("matrix source hash rows"))?;
    let end = chain
        .row_start
        .checked_add(row_count)
        .ok_or(PackageError::Invalid("matrix source hash rows"))?;
    if row_index >= end {
        return Ok(None);
    }
    let delta = row_index - chain.row_start;
    Ok(Some((chain, delta / template_rows, delta % template_rows)))
}

fn locate_permutation_row<'a>(
    package: &'a LoadedPackage,
    row_index: usize,
) -> Result<Option<(&'a PermutationInvocation, usize)>, PackageError> {
    let template_rows = package.permutation.rows.len();
    let Some(position) = package
        .permutation_invocations
        .partition_point(|invocation| invocation.row_start <= row_index)
        .checked_sub(1)
    else {
        return Ok(None);
    };
    let invocation = &package.permutation_invocations[position];
    let end = invocation
        .row_start
        .checked_add(template_rows)
        .ok_or(PackageError::Invalid("matrix source permutation rows"))?;
    Ok((row_index < end).then_some((invocation, row_index - invocation.row_start)))
}

fn locate_compact_row<'a>(
    package: &'a LoadedPackage,
    row_index: usize,
) -> Result<Option<(&'a CompactRowInvocation, usize)>, PackageError> {
    let Some(position) = package
        .compact_invocations
        .partition_point(|invocation| invocation.row_start <= row_index)
        .checked_sub(1)
    else {
        return Ok(None);
    };
    let invocation = &package.compact_invocations[position];
    let row_count = invocation.row_count(&package.compact_templates);
    let end = invocation
        .row_start
        .checked_add(row_count)
        .ok_or(PackageError::Invalid("matrix source compact rows"))?;
    Ok((row_index < end).then_some((invocation, row_index - invocation.row_start)))
}

fn source_assertion(row: &SparseRow) -> SourceRow {
    SourceRow {
        a: source_sparse(&row.a),
        b: source_sparse(&row.b),
        c: source_sparse(&row.c),
    }
}

fn source_witness(instruction: &WitnessInstruction) -> SourceRow {
    SourceRow {
        a: source_sparse(&instruction.a),
        b: source_sparse(&instruction.b),
        c: SourceCombination {
            constant: Goldilocks::ZERO,
            terms: vec![Entry {
                column: instruction.target,
                coefficient: Goldilocks::ONE,
            }],
        },
    }
}

fn source_sparse(combination: &SparseCombination) -> SourceCombination {
    SourceCombination {
        constant: combination.constant,
        terms: combination
            .terms
            .iter()
            .map(|term| Entry {
                column: term.column,
                coefficient: term.coefficient,
            })
            .collect(),
    }
}

fn source_template_row(
    permutation: &PermutationTemplate,
    row: &TemplateRow,
    invocation: ScheduledInvocation<'_>,
) -> Result<SourceRow, PackageError> {
    Ok(SourceRow {
        a: instantiate_template(permutation, &row.a, invocation)?,
        b: instantiate_template(permutation, &row.b, invocation)?,
        c: instantiate_template(permutation, &row.c, invocation)?,
    })
}

fn instantiate_template(
    permutation: &PermutationTemplate,
    combination: &TemplateCombination,
    invocation: ScheduledInvocation<'_>,
) -> Result<SourceCombination, PackageError> {
    let mut result = SourceCombination {
        constant: combination.constant,
        terms: Vec::new(),
    };
    for term in &combination.terms {
        match term.column {
            ColumnRef::Local(index) => result.terms.push(Entry {
                column: invocation
                    .witness_start()
                    .checked_add(index)
                    .ok_or(PackageError::Invalid("matrix source local column"))?,
                coefficient: term.coefficient,
            }),
            ColumnRef::Input(lane) => {
                let input = invocation_input(permutation, invocation, lane)?;
                result.constant += term.coefficient * input.constant;
                result
                    .terms
                    .extend(input.terms.into_iter().map(|entry| Entry {
                        column: entry.column,
                        coefficient: term.coefficient * entry.coefficient,
                    }));
            }
        }
    }
    Ok(result)
}

fn invocation_input(
    permutation: &PermutationTemplate,
    invocation: ScheduledInvocation<'_>,
    lane: usize,
) -> Result<SourceCombination, PackageError> {
    match invocation {
        ScheduledInvocation::Explicit(explicit) => explicit
            .inputs
            .get(lane)
            .map(source_sparse)
            .ok_or(PackageError::Invalid("matrix source permutation input")),
        ScheduledInvocation::Hash { chain, ordinal, .. } => {
            if lane >= permutation.input_count {
                return Err(PackageError::Invalid("matrix source hash input"));
            }
            let mut combination = SourceCombination {
                constant: Goldilocks::ZERO,
                terms: Vec::with_capacity(2),
            };
            if ordinal > 0 {
                combination.terms.push(Entry {
                    column: chain.witness_start
                        + (ordinal - 1) * permutation.local_column_count
                        + permutation.output_local_start
                        + lane,
                    coefficient: Goldilocks::ONE,
                });
            }
            if ordinal < chain.absorb_count {
                let input_offset = ordinal
                    .checked_mul(4)
                    .and_then(|offset| offset.checked_add(lane))
                    .ok_or(PackageError::Invalid("matrix source hash input"))?;
                if lane < 4 && input_offset < chain.input_length {
                    combination.terms.push(Entry {
                        column: chain.input_start + input_offset,
                        coefficient: Goldilocks::ONE,
                    });
                }
            } else if lane == 0 {
                combination.constant = Goldilocks::ONE;
            }
            Ok(combination)
        }
    }
}

fn source_compact_row(package: &LoadedPackage, invocation: &CompactRowInvocation, local_row: usize) -> SourceRow {
    let row = &package.compact_templates[invocation.template_index].rows[local_row];
    SourceRow {
        a: instantiate_compact(&row.a, invocation),
        b: instantiate_compact(&row.b, invocation),
        c: instantiate_compact(&row.c, invocation),
    }
}

fn instantiate_compact(combination: &TemplateCombination, invocation: &CompactRowInvocation) -> SourceCombination {
    SourceCombination {
        constant: combination.constant,
        terms: combination
            .terms
            .iter()
            .map(|term| Entry {
                column: match term.column {
                    ColumnRef::Input(index) => invocation.input_column(index),
                    ColumnRef::Local(index) => invocation.local_start + index,
                },
                coefficient: term.coefficient,
            })
            .collect(),
    }
}
