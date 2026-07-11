//! Width attribution for the selective low-norm compiler.

use neo_ccs::{CcsMatrix, CscMat};

/// Retained low-norm coordinates touched by one non-authoritative row-family
/// marker. Nested families overlap by design.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveFamilyWidthAudit {
    pub name: &'static str,
    pub unit_columns: usize,
    pub balanced_columns: usize,
    pub binary_columns: usize,
    pub coordinates_before_aliases: usize,
    pub poseidon2_permutations: usize,
    pub poseidon2_coordinates: usize,
}

/// Retained source values owned by direct selective trace classes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveTraceWidthAudit {
    pub poseidon2_permutations: usize,
    pub poseidon2_columns: usize,
    pub poseidon2_coordinates: usize,
    pub polynomial_evaluation_columns: usize,
    pub polynomial_evaluation_coordinates: usize,
    pub product_sum_columns: usize,
    pub product_sum_coordinates: usize,
    pub product_sum_internal_columns: usize,
    pub product_sum_internal_coordinates: usize,
}

/// Exact committed-width census for one branch-private suffix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveArmWidthAudit {
    pub branch_source_columns: usize,
    pub eliminated_columns: usize,
    pub unit_columns: usize,
    pub balanced_columns: usize,
    pub binary_columns: usize,
    pub retained_coordinates_before_aliases: usize,
    pub decomposition_aliases: usize,
    pub equality_aliases: usize,
    pub branch_coordinates: usize,
    pub derived_product_sums: usize,
    pub derived_coordinates: usize,
    pub total_branch_coordinates: usize,
    pub traces: SelectiveTraceWidthAudit,
    pub row_families: Vec<SelectiveFamilyWidthAudit>,
}

/// Exact width contract produced before any CCS matrices are allocated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveLowNormWidthAudit {
    pub constant_coordinate: usize,
    pub public_coordinates: usize,
    pub selector_coordinates: usize,
    pub alignment_padding: usize,
    pub shared_private_coordinates: usize,
    pub branch_start: usize,
    pub arms: Vec<SelectiveArmWidthAudit>,
    pub total_coordinates: usize,
}

pub(super) fn row_family_width_audits(
    arm: &super::SparseR1cs,
    widths: &[usize],
    branch_start: usize,
    balanced_width: usize,
    binary_width: usize,
) -> Vec<SelectiveFamilyWidthAudit> {
    let mut families = Vec::<(&'static str, Vec<(usize, usize)>)>::new();
    for family in arm.row_family_ranges() {
        if let Some((_, ranges)) = families.iter_mut().find(|(name, _)| *name == family.name) {
            ranges.push((family.row_start, family.row_end));
        } else {
            families.push((family.name, vec![(family.row_start, family.row_end)]));
        }
    }
    assert!(
        families.len() <= u64::BITS as usize,
        "too many row families for width audit"
    );
    let mut family_masks = vec![0u64; arm.m - branch_start];
    for matrix in [&arm.a, &arm.b, &arm.c] {
        for_each_explicit_term(matrix, |row, column| {
            if column < branch_start || widths[column] == 0 {
                return;
            }
            for (family_index, (_, ranges)) in families.iter().enumerate() {
                if ranges
                    .iter()
                    .any(|&(start, end)| (start..end).contains(&row))
                {
                    family_masks[column - branch_start] |= 1 << family_index;
                }
            }
        });
    }
    families
        .iter()
        .enumerate()
        .map(|(family_index, (name, ranges))| {
            let mut unit_columns = 0;
            let mut balanced_columns = 0;
            let mut binary_columns = 0;
            let mut coordinates_before_aliases = 0;
            for (offset, mask) in family_masks.iter().enumerate() {
                if mask & (1 << family_index) == 0 {
                    continue;
                }
                let width = widths[branch_start + offset];
                coordinates_before_aliases += width;
                unit_columns += usize::from(width == 1);
                balanced_columns += usize::from(width == balanced_width);
                binary_columns += usize::from(width == binary_width);
            }
            let mut poseidon2_permutations = 0;
            let mut poseidon2_coordinates = 0;
            for trace in arm.poseidon2_traces() {
                if !ranges
                    .iter()
                    .any(|&(start, end)| trace.row_start >= start && trace.row_end <= end)
                {
                    continue;
                }
                poseidon2_permutations += 1;
                let mut columns = trace
                    .sboxes
                    .iter()
                    .map(|sbox| sbox.output_col)
                    .collect::<Vec<_>>();
                columns.extend(trace.output_cols);
                columns.sort_unstable();
                columns.dedup();
                poseidon2_coordinates += columns
                    .into_iter()
                    .map(|column| widths[column])
                    .sum::<usize>();
            }
            SelectiveFamilyWidthAudit {
                name: *name,
                unit_columns,
                balanced_columns,
                binary_columns,
                coordinates_before_aliases,
                poseidon2_permutations,
                poseidon2_coordinates,
            }
        })
        .collect()
}

pub(super) fn retained_trace_widths(arm: &super::SparseR1cs, widths: &[usize]) -> SelectiveTraceWidthAudit {
    let mut poseidon2 = vec![false; arm.m];
    for trace in arm.poseidon2_traces() {
        for sbox in &trace.sboxes {
            poseidon2[sbox.output_col] = true;
        }
        for &column in &trace.output_cols {
            poseidon2[column] = true;
        }
    }
    let mut polynomial_evaluation = vec![false; arm.m];
    for trace in arm.polynomial_evaluation_traces() {
        for &column in &trace.output_cols {
            polynomial_evaluation[column] = true;
        }
    }
    let mut product_sum = vec![false; arm.m];
    for trace in arm.product_sum_batch_traces() {
        for &column in &trace.retained_columns {
            product_sum[column] = true;
        }
    }
    let census = |present: Vec<bool>| {
        present
            .into_iter()
            .enumerate()
            .filter(|(column, present)| *present && widths[*column] != 0)
            .fold((0usize, 0usize), |(columns, coordinates), (column, _)| {
                (columns + 1, coordinates + widths[column])
            })
    };
    let (poseidon2_columns, poseidon2_coordinates) = census(poseidon2);
    let (polynomial_evaluation_columns, polynomial_evaluation_coordinates) = census(polynomial_evaluation);
    let (product_sum_columns, product_sum_coordinates) = census(product_sum);
    let (product_sum_internal_columns, product_sum_internal_coordinates) = internal_product_sum_widths(arm, widths);
    SelectiveTraceWidthAudit {
        poseidon2_permutations: arm.poseidon2_traces().len(),
        poseidon2_columns,
        poseidon2_coordinates,
        polynomial_evaluation_columns,
        polynomial_evaluation_coordinates,
        product_sum_columns,
        product_sum_coordinates,
        product_sum_internal_columns,
        product_sum_internal_coordinates,
    }
}

fn internal_product_sum_widths(arm: &super::SparseR1cs, widths: &[usize]) -> (usize, usize) {
    let mut product_rows = vec![false; arm.n];
    let mut outputs = vec![false; arm.m];
    for trace in arm.product_sum_batch_traces() {
        product_rows[trace.row_start..trace.row_end].fill(true);
        for &column in &trace.retained_columns {
            outputs[column] = true;
        }
    }

    let mut external = vec![false; arm.m];
    for matrix in [&arm.a, &arm.b, &arm.c] {
        for_each_explicit_term(matrix, |row, column| {
            if !product_rows[row] {
                external[column] = true;
            }
        });
        if let CcsMatrix::CscWithSeededPhi81 { blocks, .. } = matrix {
            for block in blocks {
                for &start in block.word_starts() {
                    external[start..start + block.word_width()].fill(true);
                }
            }
        }
    }

    outputs
        .into_iter()
        .enumerate()
        .filter(|(column, output)| *output && !external[*column] && widths[*column] != 0)
        .fold((0, 0), |(columns, coordinates), (column, _)| {
            (columns + 1, coordinates + widths[column])
        })
}

fn for_each_explicit_term(matrix: &CcsMatrix<neo_math::F>, mut visit: impl FnMut(usize, usize)) {
    let mut visit_csc = |csc: &CscMat<neo_math::F>| {
        for column in 0..csc.ncols {
            for index in csc.col_ptr[column]..csc.col_ptr[column + 1] {
                visit(csc.row_idx[index], column);
            }
        }
    };
    match matrix {
        CcsMatrix::Identity { n } => {
            for row in 0..*n {
                visit(row, row);
            }
        }
        CcsMatrix::Csc(csc) => visit_csc(csc),
        CcsMatrix::CscWithSeededPhi81 { csc, .. } => visit_csc(csc),
    }
}
