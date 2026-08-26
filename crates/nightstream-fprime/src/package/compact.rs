//! Strict decoding and execution of Lean-owned compact row templates.
//!
//! The module maps template inputs and locals to final package columns. It
//! requires total input bindings, causal witness targets, exact row/column
//! ownership, and direct checks of every instantiated `A * B = C` row.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use serde::Deserialize;
use serde_json::Value;

use super::{
    checked_end, validate_template_combination, word_to_usize, ColumnRef, Layout, PackageError, RawTemplateCombination,
    TemplateCombination,
};
use crate::witness::{decode_template_expr, WitnessExpr};

#[derive(Debug, Deserialize)]
pub(super) struct RawCompactTemplateRow(
    Value,
    RawTemplateCombination,
    RawTemplateCombination,
    RawTemplateCombination,
);

#[derive(Debug, Deserialize)]
pub(super) struct RawCompactRowTemplate(u64, u64, u64, Value, Vec<RawCompactTemplateRow>);

#[derive(Debug, Deserialize)]
pub(super) struct RawCompactInputRange(u64, u64, u64, u64);

#[derive(Debug, Deserialize)]
pub(super) struct RawCompactRowInvocation(u64, u64, u64, u64, Vec<RawCompactInputRange>);

pub(super) fn raw_invocation(
    phase: u64,
    template_index: u64,
    row_start: u64,
    local_start: u64,
    input_ranges: Vec<[u64; 4]>,
) -> RawCompactRowInvocation {
    RawCompactRowInvocation(
        phase,
        template_index,
        row_start,
        local_start,
        input_ranges
            .into_iter()
            .map(|[input_start, input_count, column_start, column_stride]| {
                RawCompactInputRange(input_start, input_count, column_start, column_stride)
            })
            .collect(),
    )
}

#[derive(Clone, Debug)]
pub(super) struct CompactTemplateRow {
    pub(super) output_local: Option<usize>,
    pub(super) a: TemplateCombination,
    pub(super) b: TemplateCombination,
    pub(super) c: TemplateCombination,
}

#[derive(Clone, Debug)]
pub(super) struct CompactRowTemplate {
    pub(super) input_count: usize,
    pub(super) local_column_count: usize,
    pub(super) output_input: usize,
    output_recipe: WitnessExpr,
    pub(super) rows: Vec<CompactTemplateRow>,
}

#[derive(Clone, Copy, Debug)]
struct CompactInputRange {
    input_start: usize,
    input_count: usize,
    column_start: usize,
    column_stride: usize,
}

#[derive(Clone, Debug)]
pub(super) struct CompactRowInvocation {
    pub(super) phase: u64,
    pub(super) template_index: usize,
    pub(super) row_start: usize,
    pub(super) local_start: usize,
    input_ranges: Vec<CompactInputRange>,
    pub(super) output_column: usize,
}

impl CompactRowInvocation {
    pub(super) fn input_column(&self, input: usize) -> usize {
        for range in &self.input_ranges {
            if range.input_start <= input && input < range.input_start + range.input_count {
                return range.column_start + (input - range.input_start) * range.column_stride;
            }
        }
        unreachable!("validated compact input coverage")
    }

    pub(super) fn row_count(&self, templates: &[CompactRowTemplate]) -> usize {
        templates[self.template_index].rows.len()
    }

    pub(super) fn local_column_count(&self, templates: &[CompactRowTemplate]) -> usize {
        templates[self.template_index].local_column_count
    }
}

pub(super) fn validate_templates(
    raw_templates: Vec<RawCompactRowTemplate>,
) -> Result<Vec<CompactRowTemplate>, PackageError> {
    raw_templates
        .into_iter()
        .map(validate_template)
        .collect::<Result<Vec<_>, _>>()
}

fn validate_template(raw: RawCompactRowTemplate) -> Result<CompactRowTemplate, PackageError> {
    let RawCompactRowTemplate(input_count, local_column_count, output_input, output_recipe, rows) = raw;
    let input_count = word_to_usize(input_count, "compact template input count")?;
    let local_column_count = word_to_usize(local_column_count, "compact template local count")?;
    let output_input = word_to_usize(output_input, "compact template output input")?;
    let expected_rows = local_column_count
        .checked_add(1)
        .ok_or(PackageError::Invalid("compact template row count overflow"))?;
    if input_count == 0 || output_input >= input_count || rows.len() != expected_rows {
        return Err(PackageError::Invalid("compact template shape"));
    }
    let output_recipe = decode_template_expr(&output_recipe, input_count, output_input)?;
    let rows = rows
        .into_iter()
        .enumerate()
        .map(|(ordinal, row)| validate_template_row(row, ordinal, input_count, local_column_count))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CompactRowTemplate {
        input_count,
        local_column_count,
        output_input,
        output_recipe,
        rows,
    })
}

fn validate_template_row(
    raw: RawCompactTemplateRow,
    ordinal: usize,
    input_count: usize,
    local_count: usize,
) -> Result<CompactTemplateRow, PackageError> {
    let RawCompactTemplateRow(output, a, b, c) = raw;
    let output_local = decode_optional_nat(output)?;
    let causal_before = ordinal.min(local_count);
    let a = validate_template_combination(a, input_count, local_count, Some(causal_before))?;
    let b = validate_template_combination(b, input_count, local_count, Some(causal_before))?;
    let c = validate_template_combination(c, input_count, local_count, None)?;

    if ordinal < local_count {
        if output_local != Some(ordinal)
            || c.constant != Goldilocks::ZERO
            || c.terms.len() != 1
            || c.terms[0].column != ColumnRef::Local(ordinal)
            || c.terms[0].coefficient != Goldilocks::ONE
        {
            return Err(PackageError::Invalid("compact witness row shape"));
        }
    } else if output_local.is_some()
        || b.constant != Goldilocks::ONE
        || !b.terms.is_empty()
        || c.constant != Goldilocks::ZERO
        || !c.terms.is_empty()
    {
        return Err(PackageError::Invalid("compact assertion row shape"));
    }

    Ok(CompactTemplateRow { output_local, a, b, c })
}

fn decode_optional_nat(value: Value) -> Result<Option<usize>, PackageError> {
    let fields = value
        .as_array()
        .ok_or(PackageError::Invalid("compact optional output"))?;
    match fields.as_slice() {
        [tag] if tag.as_u64() == Some(0) => Ok(None),
        [tag, output] if tag.as_u64() == Some(1) => output
            .as_u64()
            .ok_or(PackageError::Invalid("compact optional output"))
            .and_then(|output| word_to_usize(output, "compact optional output"))
            .map(Some),
        _ => Err(PackageError::Invalid("compact optional output")),
    }
}

pub(super) fn validate_invocations(
    raw_invocations: Vec<RawCompactRowInvocation>,
    templates: &[CompactRowTemplate],
    layout: &Layout,
    witness_start: usize,
) -> Result<Vec<CompactRowInvocation>, PackageError> {
    let invocations = raw_invocations
        .into_iter()
        .map(|invocation| validate_invocation(invocation, templates, layout, witness_start))
        .collect::<Result<Vec<_>, _>>()?;
    if invocations.windows(2).any(|pair| {
        pair[0].row_start >= pair[1].row_start
            || pair[0].phase > pair[1].phase
            || pair[0].row_start + pair[0].row_count(templates) > pair[1].row_start
    }) {
        return Err(PackageError::Invalid("compact invocation order"));
    }
    Ok(invocations)
}

fn validate_invocation(
    raw: RawCompactRowInvocation,
    templates: &[CompactRowTemplate],
    layout: &Layout,
    witness_start: usize,
) -> Result<CompactRowInvocation, PackageError> {
    let RawCompactRowInvocation(phase, template_index, row_start, local_start, input_ranges) = raw;
    let template_index = word_to_usize(template_index, "compact invocation template")?;
    let template = templates
        .get(template_index)
        .ok_or(PackageError::Invalid("compact invocation template"))?;
    let row_start = word_to_usize(row_start, "compact invocation row start")?;
    let local_start = word_to_usize(local_start, "compact invocation local start")?;
    if phase == 0
        || checked_end(row_start, template.rows.len())? > layout.row_count
        || local_start < witness_start
        || checked_end(local_start, template.local_column_count)? > layout.private_column_count
    {
        return Err(PackageError::Invalid("compact invocation shape"));
    }

    let input_ranges = validate_input_ranges(input_ranges, template.input_count, layout)?;
    let mut invocation = CompactRowInvocation {
        phase,
        template_index,
        row_start,
        local_start,
        input_ranges,
        output_column: 0,
    };
    let output_column = invocation.input_column(template.output_input);
    invocation.output_column = output_column;
    if output_column < witness_start || output_column >= layout.private_column_count || output_column >= local_start {
        return Err(PackageError::Invalid("compact invocation output"));
    }
    for input in 0..template.input_count {
        if input == template.output_input {
            continue;
        }
        let column = invocation.input_column(input);
        if column < layout.constant_column && column >= output_column {
            return Err(PackageError::Invalid("noncausal compact input"));
        }
    }
    Ok(invocation)
}

fn validate_input_ranges(
    raw_ranges: Vec<RawCompactInputRange>,
    input_count: usize,
    layout: &Layout,
) -> Result<Vec<CompactInputRange>, PackageError> {
    let mut cursor = 0usize;
    let mut ranges = Vec::with_capacity(raw_ranges.len());
    for raw in raw_ranges {
        let RawCompactInputRange(input_start, range_count, column_start, column_stride) = raw;
        let input_start = word_to_usize(input_start, "compact input start")?;
        let range_count = word_to_usize(range_count, "compact input count")?;
        let column_start = word_to_usize(column_start, "compact column start")?;
        let column_stride = word_to_usize(column_stride, "compact column stride")?;
        if input_start != cursor || range_count == 0 || column_stride == 0 {
            return Err(PackageError::Invalid("compact input partition"));
        }
        cursor = checked_end(input_start, range_count)?;
        if cursor > input_count {
            return Err(PackageError::Invalid("compact input partition"));
        }
        let last_column = column_start
            .checked_add(
                (range_count - 1)
                    .checked_mul(column_stride)
                    .ok_or(PackageError::Invalid("compact input range overflow"))?,
            )
            .ok_or(PackageError::Invalid("compact input range overflow"))?;
        if last_column >= layout.total_column_count
            || column_start == layout.constant_column
            || (column_start < layout.constant_column && last_column >= layout.constant_column)
        {
            return Err(PackageError::Invalid("compact input column range"));
        }
        ranges.push(CompactInputRange {
            input_start,
            input_count: range_count,
            column_start,
            column_stride,
        });
    }
    if cursor != input_count {
        return Err(PackageError::Invalid("compact input coverage"));
    }
    Ok(ranges)
}

pub(super) fn execute_invocation(
    invocation: &CompactRowInvocation,
    templates: &[CompactRowTemplate],
    assignment: &mut [Goldilocks],
) -> Result<(), PackageError> {
    let template = &templates[invocation.template_index];
    assignment[invocation.output_column] = template
        .output_recipe
        .eval_with(&|input| assignment[invocation.input_column(input)]);
    for row in &template.rows {
        let left = eval_combination(&row.a, invocation, assignment);
        let right = eval_combination(&row.b, invocation, assignment);
        if let Some(output_local) = row.output_local {
            assignment[invocation.local_start + output_local] = left * right;
        }
        let output = eval_combination(&row.c, invocation, assignment);
        if left * right != output {
            return Err(PackageError::Invalid("unsatisfied compact row"));
        }
    }
    Ok(())
}

fn eval_combination(
    combination: &TemplateCombination,
    invocation: &CompactRowInvocation,
    assignment: &[Goldilocks],
) -> Goldilocks {
    combination
        .terms
        .iter()
        .fold(combination.constant, |sum, term| {
            let column = match term.column {
                ColumnRef::Input(input) => invocation.input_column(input),
                ColumnRef::Local(local) => invocation.local_start + local,
            };
            sum + term.coefficient * assignment[column]
        })
}
