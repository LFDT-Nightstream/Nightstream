//! Selective low-norm compiler for the authoritative Road A relation.
//!
//! Ordinary verifier R1CS rows remain authoritative. Recorded Poseidon2,
//! projection, K-arithmetic, and centered-range traces remove only their
//! materialized temporaries. Canonical-u64 fields reuse their 64 bit slots;
//! other full fields use 41 balanced-ternary unit digits.

use std::collections::HashMap;

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SeededPhi81LinearBlock, SparsePoly, Term};
use neo_math::{D, F};
use p3_field::{Field, PrimeCharacteristicRing};

use super::lowering::{DerivedProductSumEncoding, LowNormR1csError, MultiBranchLowNormR1cs};
use super::selective_audit::{
    retained_trace_widths, row_family_width_audits, SelectiveArmWidthAudit, SelectiveLowNormWidthAudit,
};
use super::SparseR1cs;
use crate::engine::r1cs_circuit::builder::{
    BalancedTernaryDecomposition, CanonicalU64Decomposition, ProductFactorTrace,
};
use crate::engine::r1cs_circuit::Lc;
use crate::paper::relations::Structure;

const EVAL_GROUP_SIZE: usize = 5;
const BALANCED_FIELD_WIDTH: usize = 41;
const BINARY_FIELD_WIDTH: usize = 64;

struct SelectiveLayout {
    plans: Vec<SelectiveArmPlan>,
    slots: Vec<Vec<Option<(usize, usize)>>>,
    aliases: Vec<Vec<Option<(usize, usize)>>>,
    equal_aliases: Vec<Vec<Option<usize>>>,
    derived_product_sums: Vec<Vec<DerivedProductSumEncoding>>,
    selector_cols: Vec<usize>,
    zero_padding_cols: Vec<usize>,
    public_input_len: usize,
    columns: usize,
    audit: SelectiveLowNormWidthAudit,
}

struct SelectiveArmPlan {
    widths: Vec<usize>,
    centered: Vec<bool>,
    equality_roots: Vec<usize>,
    definitions: LinearDefinitions,
}

struct LinearDefinition {
    row: Option<usize>,
    target: usize,
    rhs: Lc,
}

struct LinearDefinitions {
    by_column: Vec<Option<usize>>,
    entries: Vec<LinearDefinition>,
}

impl LinearDefinitions {
    fn get(&self, column: usize) -> Option<&Lc> {
        self.by_column[column].map(|index| &self.entries[index].rhs)
    }
}

/// Compile one-hot field-R1CS arms while lowering Poseidon2 directly to
/// degree-seven CCS rows.
pub fn build_multi_branch_selective_low_norm_r1cs_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<MultiBranchLowNormR1cs, LowNormR1csError> {
    let layout = prepare_selective_layout(arms, shared_private_fields, modulus, residue)?;
    let structure = build_structure(
        arms,
        &layout.plans,
        &layout.slots,
        &layout.aliases,
        &layout.equal_aliases,
        shared_private_fields,
        &layout.derived_product_sums,
        &layout.selector_cols,
        &layout.zero_padding_cols,
        layout.columns,
    )?;
    Ok(MultiBranchLowNormR1cs::from_compiler_parts(
        structure,
        layout.public_input_len,
        layout.selector_cols,
        arms[0].m_in,
        layout.slots,
        layout.aliases,
        layout.equal_aliases,
        layout
            .plans
            .iter()
            .map(|plan| plan.centered.clone())
            .collect(),
        layout.derived_product_sums,
    ))
}

/// Compute the exact selective-lowering width without constructing the output
/// matrices. Production budget tests use this to attribute width regressions.
pub fn audit_multi_branch_selective_low_norm_width_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLowNormWidthAudit, LowNormR1csError> {
    Ok(prepare_selective_layout(arms, shared_private_fields, modulus, residue)?.audit)
}

fn prepare_selective_layout(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLayout, LowNormR1csError> {
    if arms.len() < 2 {
        return Err(LowNormR1csError::TooFewArms(arms.len()));
    }
    if modulus == 0 {
        return Err(LowNormR1csError::ZeroAlignmentModulus);
    }
    for arm in arms {
        arm.validate_shape()?;
        if arm.m_in == 0 {
            return Err(LowNormR1csError::MissingPublicConstant);
        }
    }

    let public_field_count = arms[0].m_in;
    let plans = arms
        .iter()
        .map(|arm| selective_arm_plan(arm, shared_private_fields))
        .collect::<Result<Vec<_>, _>>()?;
    let widths = plans
        .iter()
        .map(|plan| plan.widths.as_slice())
        .collect::<Vec<_>>();
    validate_shared_shapes(arms, &widths, public_field_count, shared_private_fields)?;
    let aliases = arms
        .iter()
        .zip(&plans)
        .map(|(arm, plan)| decomposition_aliases(arm, &plan.widths, shared_private_fields))
        .collect::<Vec<_>>();
    let equal_aliases = arms
        .iter()
        .zip(&plans)
        .zip(&aliases)
        .map(|((arm, plan), aliases)| equality_aliases(arm, plan, aliases, shared_private_fields))
        .collect::<Vec<_>>();
    let mut cursor = 1usize;
    let mut slots = arms.iter().map(|arm| vec![None; arm.m]).collect::<Vec<_>>();
    for col in 1..public_field_count {
        let width = widths[0][col];
        let slot = Some((cursor, width));
        for arm_slots in &mut slots {
            arm_slots[col] = slot;
        }
        cursor += width;
    }
    let public_input_len = cursor;
    let selector_cols = (0..arms.len())
        .map(|_| {
            let selector = cursor;
            cursor += 1;
            selector
        })
        .collect::<Vec<_>>();
    let residue = residue % modulus;
    let padding_len = (residue + modulus - cursor % modulus) % modulus;
    let zero_padding_cols = (cursor..cursor + padding_len).collect::<Vec<_>>();
    cursor += padding_len;

    for offset in 0..shared_private_fields {
        let source = arms[0].m_in + offset;
        let width = widths[0][source];
        let slot = Some((cursor, width));
        for (arm_index, arm) in arms.iter().enumerate() {
            slots[arm_index][arm.m_in + offset] = slot;
        }
        cursor += width;
    }
    let branch_start = cursor;
    let mut arm_cursors = vec![branch_start; arms.len()];
    for (arm_index, arm) in arms.iter().enumerate() {
        let mut arm_cursor = branch_start;
        for col in arm.m_in + shared_private_fields..arm.m {
            assign_slot(
                &mut slots[arm_index],
                &widths[arm_index],
                &aliases[arm_index],
                &equal_aliases[arm_index],
                col,
                &mut arm_cursor,
            )?;
        }
        arm_cursors[arm_index] = arm_cursor;
    }
    let branch_coordinates = arm_cursors
        .iter()
        .map(|&arm_cursor| arm_cursor - branch_start)
        .collect::<Vec<_>>();
    let mut derived_product_sums = (0..arms.len()).map(|_| Vec::new()).collect::<Vec<Vec<_>>>();
    for (arm_index, arm) in arms.iter().enumerate() {
        for trace in arm.polynomial_evaluation_traces() {
            for limb in 0..2 {
                let product_indices = (1..trace.coefficient_cols.len()).collect::<Vec<_>>();
                let groups = product_indices.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
                let mut previous = None;
                for group in groups.iter().take(groups.len().saturating_sub(1)) {
                    let slot = (arm_cursors[arm_index], BALANCED_FIELD_WIDTH);
                    arm_cursors[arm_index] += BALANCED_FIELD_WIDTH;
                    let index = derived_product_sums[arm_index].len();
                    derived_product_sums[arm_index].push(DerivedProductSumEncoding {
                        slot,
                        factors: group
                            .iter()
                            .map(|&index| ProductFactorTrace {
                                left: lc_from_column(trace.coefficient_cols[index]),
                                right: lc_from_column(trace.power_cols[index][limb]),
                                coefficient: F::ONE,
                            })
                            .collect(),
                        previous,
                    });
                    previous = Some(index);
                }
            }
        }
        for batch in arm.product_sum_batch_traces() {
            for identity in &batch.identities {
                if identity.factors.len() <= EVAL_GROUP_SIZE {
                    continue;
                }
                let groups = identity.factors.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
                let mut previous = None;
                for group in groups.iter().take(groups.len() - 1) {
                    let slot = (arm_cursors[arm_index], BALANCED_FIELD_WIDTH);
                    arm_cursors[arm_index] += BALANCED_FIELD_WIDTH;
                    let index = derived_product_sums[arm_index].len();
                    derived_product_sums[arm_index].push(DerivedProductSumEncoding {
                        slot,
                        factors: group.to_vec(),
                        previous,
                    });
                    previous = Some(index);
                }
            }
        }
        cursor = cursor.max(arm_cursors[arm_index]);
    }

    let arms_audit = arms
        .iter()
        .enumerate()
        .map(|(arm_index, arm)| {
            let branch_range = arm.m_in + shared_private_fields..arm.m;
            let branch_widths = &plans[arm_index].widths[branch_range.clone()];
            let eliminated_columns = branch_widths.iter().filter(|&&width| width == 0).count();
            let unit_columns = branch_widths.iter().filter(|&&width| width == 1).count();
            let balanced_columns = branch_widths
                .iter()
                .filter(|&&width| width == BALANCED_FIELD_WIDTH)
                .count();
            let binary_columns = branch_widths
                .iter()
                .filter(|&&width| width == BINARY_FIELD_WIDTH)
                .count();
            let retained_coordinates_before_aliases = branch_widths.iter().sum();
            let decomposition_aliases = aliases[arm_index][branch_range.clone()]
                .iter()
                .filter(|alias| alias.is_some())
                .count();
            let equality_aliases = equal_aliases[arm_index][branch_range]
                .iter()
                .filter(|alias| alias.is_some())
                .count();
            let derived_product_sums = derived_product_sums[arm_index].len();
            let derived_coordinates = derived_product_sums * BALANCED_FIELD_WIDTH;
            SelectiveArmWidthAudit {
                branch_source_columns: branch_widths.len(),
                eliminated_columns,
                unit_columns,
                balanced_columns,
                binary_columns,
                retained_coordinates_before_aliases,
                decomposition_aliases,
                equality_aliases,
                branch_coordinates: branch_coordinates[arm_index],
                derived_product_sums,
                derived_coordinates,
                total_branch_coordinates: branch_coordinates[arm_index] + derived_coordinates,
                traces: retained_trace_widths(arm, &plans[arm_index].widths),
                row_families: row_family_width_audits(
                    arm,
                    &plans[arm_index].widths,
                    arm.m_in + shared_private_fields,
                    BALANCED_FIELD_WIDTH,
                    BINARY_FIELD_WIDTH,
                ),
            }
        })
        .collect();
    let audit = SelectiveLowNormWidthAudit {
        constant_coordinate: 1,
        public_coordinates: public_input_len - 1,
        selector_coordinates: selector_cols.len(),
        alignment_padding: zero_padding_cols.len(),
        shared_private_coordinates: branch_start - public_input_len - selector_cols.len() - zero_padding_cols.len(),
        branch_start,
        arms: arms_audit,
        total_coordinates: cursor,
    };
    Ok(SelectiveLayout {
        plans,
        slots,
        aliases,
        equal_aliases,
        derived_product_sums,
        selector_cols,
        zero_padding_cols,
        public_input_len,
        columns: cursor,
        audit,
    })
}

fn selective_arm_plan(arm: &SparseR1cs, shared_private_fields: usize) -> Result<SelectiveArmPlan, LowNormR1csError> {
    let shared_end = arm
        .m_in
        .checked_add(shared_private_fields)
        .filter(|&end| end <= arm.m)
        .ok_or_else(|| trace_error("shared private prefix exceeds the source arm"))?;
    let mut widths = vec![BALANCED_FIELD_WIDTH; arm.m];
    let mut centered = vec![false; arm.m];
    widths[0] = 0;
    let mut eliminated = vec![false; arm.m];
    for trace in arm.poseidon2_traces() {
        if trace.row_start >= trace.row_end || trace.row_end > arm.n {
            return Err(trace_error("Poseidon2 row range is outside the source arm"));
        }
        for &col in &trace.allocated_columns {
            if col == 0 || col >= arm.m {
                return Err(trace_error("Poseidon2 temporary column is outside the source arm"));
            }
            eliminated[col] = true;
        }
        for sbox in &trace.sboxes {
            eliminated[sbox.output_col] = false;
        }
        for &col in &trace.output_cols {
            eliminated[col] = false;
        }
    }
    for trace in arm.polynomial_evaluation_traces() {
        if trace.row_start >= trace.row_end || trace.row_end > arm.n {
            return Err(trace_error("polynomial-evaluation row range is outside the source arm"));
        }
        if trace.coefficient_cols.is_empty()
            || trace.coefficient_cols.len() != trace.power_cols.len()
            || trace.coefficient_cols.len() > D
        {
            return Err(trace_error(
                "polynomial-evaluation trace has invalid coefficient geometry",
            ));
        }
        for &col in &trace.allocated_columns {
            if col == 0 || col >= arm.m {
                return Err(trace_error("polynomial-evaluation temporary is outside the source arm"));
            }
            eliminated[col] = true;
        }
        for &col in &trace.output_cols {
            eliminated[col] = false;
        }
    }
    for trace in arm.product_sum_batch_traces() {
        if trace.row_start >= trace.row_end || trace.row_end > arm.n || trace.identities.is_empty() {
            return Err(trace_error("product-sum batch trace has invalid geometry"));
        }
        for &column in &trace.allocated_columns {
            if column == 0 || column >= arm.m {
                return Err(trace_error("product-sum temporary is outside the source arm"));
            }
            eliminated[column] = true;
        }
        for &column in &trace.retained_columns {
            eliminated[column] = false;
        }
    }
    for trace in arm.centered_unit_traces() {
        if trace.row_start >= trace.row_end || trace.row_end > arm.n {
            return Err(trace_error("centered-unit trace has invalid row geometry"));
        }
        for &column in &trace.allocated_columns {
            if column == 0 || column >= arm.m {
                return Err(trace_error("centered-unit temporary is outside the source arm"));
            }
            eliminated[column] = true;
        }
        eliminated[trace.value_col] = false;
    }
    for col in 1..arm.m {
        if eliminated[col] {
            widths[col] = 0;
        }
    }
    if eliminated[1..arm.m_in].iter().any(|&value| value) {
        return Err(trace_error("selective trace attempted to eliminate a public output"));
    }
    let definitions = find_linear_definitions(arm, shared_end, &eliminated)?;
    for (column, definition) in definitions.by_column.iter().enumerate() {
        if definition.is_some() {
            widths[column] = 0;
        }
    }
    for decomposition in arm.canonical_u64_decompositions() {
        if widths[decomposition.field_col] != 0 {
            widths[decomposition.field_col] = BINARY_FIELD_WIDTH;
        }
        for &bit_col in &decomposition.bit_cols {
            if widths[bit_col] != 0 {
                widths[bit_col] = 1;
            }
        }
    }
    let canonical_sources = arm
        .canonical_u64_decompositions()
        .iter()
        .map(|decomposition| decomposition.field_col)
        .collect::<std::collections::HashSet<_>>();
    for decomposition in arm.balanced_ternary_decompositions() {
        if widths[decomposition.field_col] != 0 && !canonical_sources.contains(&decomposition.field_col) {
            widths[decomposition.field_col] = BALANCED_FIELD_WIDTH;
        }
        for &digit_col in &decomposition.digit_cols {
            if widths[digit_col] != 0 {
                widths[digit_col] = 1;
                centered[digit_col] = true;
            }
        }
    }
    for &column in arm.centered_unit_columns() {
        if widths[column] != 0 {
            widths[column] = 1;
            centered[column] = true;
        }
    }
    for &column in arm.boolean_columns() {
        if widths[column] != 0 {
            widths[column] = 1;
            centered[column] = false;
        }
    }
    // This compiler's public prefix is `FPrimeStepOutput::public_outputs`,
    // and its shared private prefix is the current S_mem bit assignment.
    // Both are verifier-owned bit surfaces, not arbitrary field advice.
    widths[1..arm.m_in].fill(1);
    widths[arm.m_in..shared_end].fill(1);
    centered[..shared_end].fill(false);
    let equality_roots = propagate_low_norm_equalities(arm, &mut widths, &mut centered, shared_end)?;
    Ok(SelectiveArmPlan {
        widths,
        centered,
        equality_roots,
        definitions,
    })
}

fn find_linear_definitions(
    arm: &SparseR1cs,
    shared_end: usize,
    directly_eliminated: &[bool],
) -> Result<LinearDefinitions, LowNormR1csError> {
    let skipped = skipped_selective_rows(arm)?;
    let mut protected = directly_eliminated.to_vec();
    protected[..shared_end].fill(true);
    for decomposition in arm.canonical_u64_decompositions() {
        protected[decomposition.field_col] = true;
        for &column in &decomposition.bit_cols {
            protected[column] = true;
        }
    }
    for decomposition in arm.balanced_ternary_decompositions() {
        protected[decomposition.field_col] = true;
        for &column in &decomposition.digit_cols {
            protected[column] = true;
        }
    }
    for trace in arm.poseidon2_traces() {
        for sbox in &trace.sboxes {
            protected[sbox.output_col] = true;
        }
    }
    for trace in arm.polynomial_evaluation_traces() {
        for &column in &trace.output_cols {
            protected[column] = true;
        }
    }
    for trace in arm.product_sum_batch_traces() {
        for &column in &trace.retained_columns {
            protected[column] = true;
        }
    }
    if let CcsMatrix::CscWithSeededPhi81 { blocks, .. } = &arm.a {
        for block in blocks {
            for &start in block.word_starts() {
                protected[start..start + block.word_width()].fill(true);
            }
        }
    }

    let mut by_column = vec![None; arm.m];
    let mut entries = Vec::<LinearDefinition>::new();
    for trace in arm.poseidon2_traces() {
        for (&target, rhs) in trace.output_cols.iter().zip(&trace.output_linear_forms) {
            if target == 0 || target >= arm.m {
                return Err(trace_error("Poseidon2 output column is outside the source arm"));
            }
            if target < shared_end || protected[target] {
                protected[target] = true;
                continue;
            }
            if by_column[target].is_some() {
                return Err(trace_error("Poseidon2 output column has multiple linear definitions"));
            }
            let index = entries.len();
            by_column[target] = Some(index);
            entries.push(LinearDefinition {
                row: None,
                target,
                rhs: rhs.clone(),
            });
        }
    }

    let mut b_state = vec![0u8; arm.n];
    for_each_explicit_term(&arm.b, |row, column, coefficient| {
        b_state[row] = if b_state[row] == 0 && column == 0 && coefficient == F::ONE {
            1
        } else {
            2
        };
    });
    let mut c_nonzero = vec![false; arm.n];
    for_each_explicit_term(&arm.c, |row, _, _| c_nonzero[row] = true);

    let mut candidates = HashMap::<usize, (usize, F)>::new();
    for_each_explicit_term(&arm.a, |row, column, coefficient| {
        if skipped[row] || b_state[row] != 1 || c_nonzero[row] || column == 0 {
            return;
        }
        let candidate = candidates.entry(row).or_insert((column, coefficient));
        if column > candidate.0 {
            *candidate = (column, coefficient);
        }
    });
    let mut candidates = candidates
        .into_iter()
        .filter(|(_, (target, _))| !protected[*target])
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|(row, _)| *row);

    let mut row_to_definition = HashMap::<usize, usize>::new();
    for (row, (target, _)) in &candidates {
        if by_column[*target].is_some() {
            continue;
        }
        let index = entries.len();
        by_column[*target] = Some(index);
        row_to_definition.insert(*row, index);
        entries.push(LinearDefinition {
            row: Some(*row),
            target: *target,
            rhs: Lc::zero(),
        });
    }
    let target_coefficients = candidates.into_iter().collect::<HashMap<_, _>>();
    for_each_explicit_term(&arm.a, |row, column, coefficient| {
        let Some(&definition_index) = row_to_definition.get(&row) else {
            return;
        };
        let definition = &mut entries[definition_index];
        if column == definition.target {
            return;
        }
        let target_coefficient = target_coefficients[&row].1;
        let scale = -target_coefficient.inverse();
        if column == 0 {
            definition.rhs.constant += coefficient * scale;
        } else {
            definition.rhs.terms.push((column, coefficient * scale));
        }
    });
    if entries.iter().any(|definition| {
        definition.rhs.terms.iter().any(|&(column, _)| {
            column >= definition.target || column >= directly_eliminated.len() || directly_eliminated[column]
        })
    }) {
        return Err(trace_error("linear definition is not acyclic over retained columns"));
    }
    Ok(LinearDefinitions { by_column, entries })
}

fn for_each_explicit_term(matrix: &CcsMatrix<F>, mut visit: impl FnMut(usize, usize, F)) {
    let mut visit_csc = |csc: &CscMat<F>| {
        for column in 0..csc.ncols {
            for index in csc.col_ptr[column]..csc.col_ptr[column + 1] {
                visit(csc.row_idx[index], column, csc.vals[index]);
            }
        }
    };
    match matrix {
        CcsMatrix::Identity { n } => {
            for row in 0..*n {
                visit(row, row, F::ONE);
            }
        }
        CcsMatrix::Csc(csc) => visit_csc(csc),
        CcsMatrix::CscWithSeededPhi81 { csc, .. } => visit_csc(csc),
    }
}

fn propagate_low_norm_equalities(
    arm: &SparseR1cs,
    widths: &mut [usize],
    centered: &mut [bool],
    shared_end: usize,
) -> Result<Vec<usize>, LowNormR1csError> {
    let skipped = skipped_selective_rows(arm)?;
    let mut parents = (0..arm.m).collect::<Vec<_>>();
    for &(row, lhs, rhs) in arm.equality_pairs() {
        if !skipped[row] {
            union(&mut parents, lhs, rhs);
        }
    }
    for column in 0..arm.m {
        let root = find(&mut parents, column);
        parents[column] = root;
    }

    let mut boolean = vec![false; arm.m];
    boolean[..shared_end].fill(true);
    for &column in arm.boolean_columns() {
        boolean[column] = true;
    }
    for decomposition in arm.canonical_u64_decompositions() {
        for &column in &decomposition.bit_cols {
            boolean[column] = true;
        }
    }
    let mut class_boolean = vec![false; arm.m];
    let mut class_centered = vec![false; arm.m];
    for column in 1..arm.m {
        if widths[column] == 0 {
            continue;
        }
        let root = parents[column];
        class_boolean[root] |= boolean[column];
        class_centered[root] |= centered[column];
    }
    for column in 1..arm.m {
        if widths[column] == 0 {
            continue;
        }
        let root = parents[column];
        if class_boolean[root] {
            widths[column] = 1;
            centered[column] = false;
        } else if class_centered[root] {
            widths[column] = 1;
            centered[column] = true;
        }
    }
    Ok(parents)
}

fn equality_aliases(
    arm: &SparseR1cs,
    plan: &SelectiveArmPlan,
    canonical_aliases: &[Option<(usize, usize)>],
    shared_private_fields: usize,
) -> Vec<Option<usize>> {
    let branch_start = arm.m_in + shared_private_fields;
    let mut representative = vec![None; arm.m];
    for column in 1..arm.m {
        if plan.widths[column] == 0 || canonical_aliases[column].is_some() {
            continue;
        }
        let root = plan.equality_roots[column];
        representative[root].get_or_insert(column);
    }

    let mut aliases = vec![None; arm.m];
    for column in branch_start..arm.m {
        if plan.widths[column] == 0 || canonical_aliases[column].is_some() {
            continue;
        }
        let Some(source) = representative[plan.equality_roots[column]] else {
            continue;
        };
        if source < column
            && plan.widths[source] == plan.widths[column]
            && plan.centered[source] == plan.centered[column]
        {
            aliases[column] = Some(source);
        }
    }
    aliases
}

fn find(parents: &mut [usize], mut value: usize) -> usize {
    while parents[value] != value {
        parents[value] = parents[parents[value]];
        value = parents[value];
    }
    value
}

fn union(parents: &mut [usize], lhs: usize, rhs: usize) {
    let lhs = find(parents, lhs);
    let rhs = find(parents, rhs);
    if lhs != rhs {
        let (root, child) = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
        parents[child] = root;
    }
}

fn validate_shared_shapes(
    arms: &[SparseR1cs],
    widths: &[&[usize]],
    public_fields: usize,
    shared_private_fields: usize,
) -> Result<(), LowNormR1csError> {
    for (arm_index, arm) in arms.iter().enumerate().skip(1) {
        if arm.m_in != public_fields {
            return Err(LowNormR1csError::ArmPublicInputArity {
                arm: arm_index,
                actual: arm.m_in,
                expected: public_fields,
            });
        }
    }
    for col in 1..public_fields {
        for arm_index in 1..arms.len() {
            if widths[arm_index][col] != widths[0][col] {
                return Err(LowNormR1csError::ArmFieldWidth {
                    arm: arm_index,
                    col,
                    actual: widths[arm_index][col],
                    expected: widths[0][col],
                });
            }
        }
    }
    for (arm_index, arm) in arms.iter().enumerate() {
        let private_fields = arm.m - arm.m_in;
        if private_fields < shared_private_fields {
            return Err(LowNormR1csError::ArmSharedPrefixTooLong {
                arm: arm_index,
                actual: private_fields,
                required: shared_private_fields,
            });
        }
    }
    for offset in 0..shared_private_fields {
        let expected = widths[0][arms[0].m_in + offset];
        for arm_index in 1..arms.len() {
            let col = arms[arm_index].m_in + offset;
            if widths[arm_index][col] != expected {
                return Err(LowNormR1csError::ArmFieldWidth {
                    arm: arm_index,
                    col,
                    actual: widths[arm_index][col],
                    expected,
                });
            }
        }
    }
    Ok(())
}

fn decomposition_aliases(
    arm: &SparseR1cs,
    widths: &[usize],
    shared_private_fields: usize,
) -> Vec<Option<(usize, usize)>> {
    let mut aliases = vec![None; arm.m];
    let shared_end = arm.m_in + shared_private_fields;
    for CanonicalU64Decomposition { field_col, bit_cols } in arm.canonical_u64_decompositions() {
        if *field_col == 0 || widths[*field_col] != BINARY_FIELD_WIDTH {
            continue;
        }
        if (arm.m_in..shared_end).contains(field_col) {
            continue;
        }
        let usable = bit_cols.iter().all(|&bit_col| {
            bit_col > *field_col
                && bit_col < arm.m
                && bit_col >= arm.m_in
                && !(arm.m_in..shared_end).contains(&bit_col)
                && widths[bit_col] == 1
                && aliases[bit_col].is_none()
        });
        if usable {
            for (bit, &bit_col) in bit_cols.iter().enumerate() {
                aliases[bit_col] = Some((*field_col, bit));
            }
        }
    }
    for BalancedTernaryDecomposition { field_col, digit_cols } in arm.balanced_ternary_decompositions() {
        if *field_col == 0 || widths[*field_col] != BALANCED_FIELD_WIDTH {
            continue;
        }
        if (arm.m_in..shared_end).contains(field_col) {
            continue;
        }
        let usable = digit_cols.iter().all(|&digit_col| {
            digit_col > *field_col
                && digit_col < arm.m
                && digit_col >= arm.m_in
                && !(arm.m_in..shared_end).contains(&digit_col)
                && widths[digit_col] == 1
                && aliases[digit_col].is_none()
        });
        if usable {
            for (digit, &digit_col) in digit_cols.iter().enumerate() {
                aliases[digit_col] = Some((*field_col, digit));
            }
        }
    }
    aliases
}

fn assign_slot(
    slots: &mut [Option<(usize, usize)>],
    widths: &[usize],
    aliases: &[Option<(usize, usize)>],
    equal_aliases: &[Option<usize>],
    field_col: usize,
    cursor: &mut usize,
) -> Result<(), LowNormR1csError> {
    if widths[field_col] == 0 {
        return Ok(());
    }
    if let Some(source) = equal_aliases[field_col] {
        slots[field_col] = slots[source];
        return Ok(());
    }
    if let Some((source, bit)) = aliases[field_col] {
        let (start, width) =
            slots[source].ok_or_else(|| trace_error("decomposition source slot does not precede its child"))?;
        if bit >= width {
            return Err(trace_error("decomposition child exceeds its source slot"));
        }
        slots[field_col] = Some((start + bit, 1));
    } else {
        slots[field_col] = Some((*cursor, widths[field_col]));
        *cursor += widths[field_col];
    }
    Ok(())
}

fn build_structure(
    arms: &[SparseR1cs],
    plans: &[SelectiveArmPlan],
    slots: &[Vec<Option<(usize, usize)>>],
    aliases: &[Vec<Option<(usize, usize)>>],
    equal_aliases: &[Vec<Option<usize>>],
    shared_private_fields: usize,
    derived_product_sums: &[Vec<DerivedProductSumEncoding>],
    selectors: &[usize],
    zero_padding_cols: &[usize],
    cols: usize,
) -> Result<Structure, LowNormR1csError> {
    const BIT: usize = 0;
    const GENERAL_SELECTOR: usize = 1;
    const A: usize = 2;
    const B: usize = 3;
    const C: usize = 4;
    const SBOX_INPUT: usize = 5;
    const CENTERED_UNIT: usize = 6;
    const EVAL_SELECTOR: usize = 7;
    const EVAL_PAIRS: [(usize, usize); EVAL_GROUP_SIZE] =
        [(BIT, A), (B, SBOX_INPUT), (CENTERED_UNIT, 8), (9, 10), (11, 12)];
    const ARITY: usize = 13;

    let eval_pair = |pair_index: usize| EVAL_PAIRS[pair_index];

    let mut trips = (0..ARITY)
        .map(|_| Vec::new())
        .collect::<Vec<Vec<(usize, usize, F)>>>();
    let mut seeded = (0..ARITY)
        .map(|_| Vec::new())
        .collect::<Vec<Vec<SeededPhi81LinearBlock>>>();
    let mut row_cursor = 0usize;
    {
        let mut emit_digit = |selector: Option<usize>, column: usize, centered: bool| {
            // SuperNeo's NC channel proves every committed coordinate lies in
            // {-1, 0, 1}. Only binary coordinates need an additional CCS row
            // to exclude -1; duplicating centered-unit checks here makes the
            // relation wider than its assignment and prevents M0 = I.
            if centered {
                return;
            }
            trips[GENERAL_SELECTOR].push((row_cursor, selector.unwrap_or(0), F::ONE));
            trips[BIT].push((row_cursor, column, F::ONE));
            row_cursor += 1;
        };

        for &selector in selectors {
            emit_digit(None, selector, false);
        }
        for source in 1..arms[0].m_in + shared_private_fields {
            if aliases[0][source].is_some() {
                continue;
            }
            if let Some((start, width)) = slots[0][source] {
                for column in start..start + width {
                    emit_digit(None, column, plans[0].centered[source] || width == BALANCED_FIELD_WIDTH);
                }
            }
        }
        for (arm_index, arm) in arms.iter().enumerate() {
            for source in arm.m_in + shared_private_fields..arm.m {
                if aliases[arm_index][source].is_some() || equal_aliases[arm_index][source].is_some() {
                    continue;
                }
                if let Some((start, width)) = slots[arm_index][source] {
                    for column in start..start + width {
                        emit_digit(
                            Some(selectors[arm_index]),
                            column,
                            plans[arm_index].centered[source] || width == BALANCED_FIELD_WIDTH,
                        );
                    }
                }
            }
            for derived in &derived_product_sums[arm_index] {
                for column in derived.slot.0..derived.slot.0 + derived.slot.1 {
                    emit_digit(Some(selectors[arm_index]), column, true);
                }
            }
        }
    }
    let selector_row = row_cursor;
    trips[GENERAL_SELECTOR].push((selector_row, 0, F::ONE));
    trips[C].push((selector_row, 0, -F::ONE));
    for &selector in selectors {
        trips[C].push((selector_row, selector, F::ONE));
    }
    row_cursor += 1;
    for &col in zero_padding_cols {
        trips[GENERAL_SELECTOR].push((row_cursor, 0, F::ONE));
        trips[C].push((row_cursor, col, F::ONE));
        row_cursor += 1;
    }

    for (arm_index, arm) in arms.iter().enumerate() {
        let definitions = &plans[arm_index].definitions;
        let mut skipped = skipped_selective_rows(arm)?;
        for definition in &definitions.entries {
            if let Some(row) = definition.row {
                if core::mem::replace(&mut skipped[row], true) {
                    return Err(trace_error("linear definition overlaps a direct selective trace"));
                }
            }
        }
        let mut row_map = vec![None; arm.n];
        for (source_row, skip) in skipped.iter().copied().enumerate() {
            if !skip {
                row_map[source_row] = Some(row_cursor);
                trips[GENERAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                row_cursor += 1;
            }
        }
        append_source_matrix(
            &mut trips[A],
            &mut seeded[A],
            &arm.a,
            &slots[arm_index],
            definitions,
            &row_map,
        )?;
        append_source_matrix(
            &mut trips[B],
            &mut seeded[B],
            &arm.b,
            &slots[arm_index],
            definitions,
            &row_map,
        )?;
        append_source_matrix(
            &mut trips[C],
            &mut seeded[C],
            &arm.c,
            &slots[arm_index],
            definitions,
            &row_map,
        )?;
        for trace in arm.poseidon2_traces() {
            for sbox in &trace.sboxes {
                trips[GENERAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                append_lc(
                    &mut trips[SBOX_INPUT],
                    row_cursor,
                    &sbox.input,
                    &slots[arm_index],
                    definitions,
                )?;
                append_field(
                    &mut trips[C],
                    row_cursor,
                    sbox.output_col,
                    F::ONE,
                    &slots[arm_index],
                    definitions,
                )?;
                row_cursor += 1;
            }
            for lane in 0..trace.output_cols.len() {
                if definitions.get(trace.output_cols[lane]).is_some() {
                    continue;
                }
                trips[GENERAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                append_field(
                    &mut trips[C],
                    row_cursor,
                    trace.output_cols[lane],
                    F::ONE,
                    &slots[arm_index],
                    definitions,
                )?;
                append_lc_scaled(
                    &mut trips[C],
                    row_cursor,
                    &trace.output_linear_forms[lane],
                    -F::ONE,
                    &slots[arm_index],
                    definitions,
                )?;
                row_cursor += 1;
            }
        }
        for trace in arm.centered_unit_traces() {
            if plans[arm_index].widths[trace.value_col] != 0 {
                continue;
            }
            trips[GENERAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
            append_field(
                &mut trips[CENTERED_UNIT],
                row_cursor,
                trace.value_col,
                F::ONE,
                &slots[arm_index],
                definitions,
            )?;
            row_cursor += 1;
        }
        let mut derived_cursor = 0usize;
        for trace in arm.polynomial_evaluation_traces() {
            for limb in 0..2 {
                let product_indices = (1..trace.coefficient_cols.len()).collect::<Vec<_>>();
                let groups = product_indices.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
                if groups.is_empty() {
                    trips[EVAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                    append_field(
                        &mut trips[C],
                        row_cursor,
                        trace.output_cols[limb],
                        F::ONE,
                        &slots[arm_index],
                        definitions,
                    )?;
                    if limb == 0 {
                        append_field(
                            &mut trips[C],
                            row_cursor,
                            trace.coefficient_cols[0],
                            -F::ONE,
                            &slots[arm_index],
                            definitions,
                        )?;
                    }
                    row_cursor += 1;
                    continue;
                }
                let mut previous = None;
                for (group_index, group) in groups.iter().enumerate() {
                    trips[EVAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                    if group_index + 1 == groups.len() {
                        append_field(
                            &mut trips[C],
                            row_cursor,
                            trace.output_cols[limb],
                            F::ONE,
                            &slots[arm_index],
                            definitions,
                        )?;
                        if limb == 0 {
                            append_field(
                                &mut trips[C],
                                row_cursor,
                                trace.coefficient_cols[0],
                                -F::ONE,
                                &slots[arm_index],
                                definitions,
                            )?;
                        }
                    } else {
                        let derived = &derived_product_sums[arm_index][derived_cursor];
                        derived_cursor += 1;
                        append_slot(&mut trips[C], row_cursor, derived.slot, F::ONE);
                    }
                    if let Some(previous) = previous {
                        append_slot(&mut trips[C], row_cursor, previous, -F::ONE);
                    }
                    for (pair_index, &term_index) in group.iter().enumerate() {
                        let (left, right) = eval_pair(pair_index);
                        append_field(
                            &mut trips[left],
                            row_cursor,
                            trace.coefficient_cols[term_index],
                            F::ONE,
                            &slots[arm_index],
                            definitions,
                        )?;
                        append_field(
                            &mut trips[right],
                            row_cursor,
                            trace.power_cols[term_index][limb],
                            F::ONE,
                            &slots[arm_index],
                            definitions,
                        )?;
                    }
                    if group_index + 1 != groups.len() {
                        previous = Some(derived_product_sums[arm_index][derived_cursor - 1].slot);
                    }
                    row_cursor += 1;
                }
            }
        }
        for batch in arm.product_sum_batch_traces() {
            for identity in &batch.identities {
                if identity.factors.len() <= EVAL_GROUP_SIZE {
                    trips[EVAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                    append_lc(
                        &mut trips[C],
                        row_cursor,
                        &identity.result,
                        &slots[arm_index],
                        definitions,
                    )?;
                    for (pair_index, factor) in identity.factors.iter().enumerate() {
                        let (left, right) = eval_pair(pair_index);
                        append_lc_scaled(
                            &mut trips[left],
                            row_cursor,
                            &factor.left,
                            factor.coefficient,
                            &slots[arm_index],
                            definitions,
                        )?;
                        append_lc(
                            &mut trips[right],
                            row_cursor,
                            &factor.right,
                            &slots[arm_index],
                            definitions,
                        )?;
                    }
                    row_cursor += 1;
                    continue;
                }
                let groups = identity.factors.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
                let mut previous = None;
                for (group_index, group) in groups.iter().enumerate() {
                    trips[EVAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                    if group_index + 1 == groups.len() {
                        append_lc(
                            &mut trips[C],
                            row_cursor,
                            &identity.result,
                            &slots[arm_index],
                            definitions,
                        )?;
                    } else {
                        let derived = &derived_product_sums[arm_index][derived_cursor];
                        derived_cursor += 1;
                        append_slot(&mut trips[C], row_cursor, derived.slot, F::ONE);
                    }
                    if let Some(previous) = previous {
                        append_slot(&mut trips[C], row_cursor, previous, -F::ONE);
                    }
                    for (pair_index, factor) in group.iter().enumerate() {
                        let (left, right) = eval_pair(pair_index);
                        append_lc_scaled(
                            &mut trips[left],
                            row_cursor,
                            &factor.left,
                            factor.coefficient,
                            &slots[arm_index],
                            definitions,
                        )?;
                        append_lc(
                            &mut trips[right],
                            row_cursor,
                            &factor.right,
                            &slots[arm_index],
                            definitions,
                        )?;
                    }
                    if group_index + 1 != groups.len() {
                        previous = Some(derived_product_sums[arm_index][derived_cursor - 1].slot);
                    }
                    row_cursor += 1;
                }
            }
        }
        if derived_cursor != derived_product_sums[arm_index].len() {
            return Err(trace_error(
                "derived evaluation-product census drifted during row emission",
            ));
        }
    }

    // SuperNeo's NC relation is defined over M0 = I. Pad the semantic rows
    // and private assignment with ignored zero coordinates to one square,
    // D-aligned domain, then prepend the identity without changing f.
    let rows = row_cursor.max(cols).next_multiple_of(D);
    let mut matrices = Vec::with_capacity(ARITY + 1);
    matrices.push(CcsMatrix::Identity { n: rows });
    for index in 0..ARITY {
        let csc = CscMat::from_triplets(core::mem::take(&mut trips[index]), rows, rows);
        matrices.push(CcsMatrix::csc_with_seeded_phi81(
            csc,
            core::mem::take(&mut seeded[index]),
        )?);
    }
    let term = |coefficient: F, powers: &[(usize, u32)]| {
        let mut exps = vec![0u32; ARITY];
        for &(index, power) in powers {
            exps[index] = power;
        }
        Term {
            coeff: coefficient,
            exps,
        }
    };
    let mut terms = vec![
        term(F::ONE, &[(GENERAL_SELECTOR, 1), (BIT, 2)]),
        term(-F::ONE, &[(GENERAL_SELECTOR, 1), (BIT, 1)]),
        term(F::ONE, &[(GENERAL_SELECTOR, 1), (A, 1), (B, 1)]),
        term(-F::ONE, &[(GENERAL_SELECTOR, 1), (C, 1)]),
        term(F::ONE, &[(GENERAL_SELECTOR, 1), (SBOX_INPUT, 7)]),
        term(F::ONE, &[(GENERAL_SELECTOR, 1), (CENTERED_UNIT, 3)]),
        term(-F::ONE, &[(GENERAL_SELECTOR, 1), (CENTERED_UNIT, 1)]),
        term(-F::ONE, &[(EVAL_SELECTOR, 1), (C, 1)]),
    ];
    for &(left, right) in &EVAL_PAIRS {
        terms.push(term(F::ONE, &[(EVAL_SELECTOR, 1), (left, 1), (right, 1)]));
    }
    let polynomial = SparsePoly::new(ARITY, terms).insert_var_at_front();
    CcsStructure::new_sparse(matrices, polynomial).map_err(|error| trace_error(&error.to_string()))
}

fn skipped_selective_rows(arm: &SparseR1cs) -> Result<Vec<bool>, LowNormR1csError> {
    let mut skipped = vec![false; arm.n];
    for trace in arm.poseidon2_traces() {
        for row in trace.row_start..trace.row_end {
            if core::mem::replace(&mut skipped[row], true) {
                return Err(trace_error("Poseidon2 traces overlap"));
            }
        }
    }
    for trace in arm.polynomial_evaluation_traces() {
        for row in trace.row_start..trace.row_end {
            if core::mem::replace(&mut skipped[row], true) {
                return Err(trace_error("selective trace row ranges overlap"));
            }
        }
    }
    for trace in arm.product_sum_batch_traces() {
        for row in trace.row_start..trace.row_end {
            if core::mem::replace(&mut skipped[row], true) {
                return Err(trace_error("product-sum trace overlaps another selective trace"));
            }
        }
    }
    for trace in arm.centered_unit_traces() {
        for row in trace.row_start..trace.row_end {
            if core::mem::replace(&mut skipped[row], true) {
                return Err(trace_error("centered-unit trace overlaps another selective trace"));
            }
        }
    }
    Ok(skipped)
}

fn append_source_matrix(
    trips: &mut Vec<(usize, usize, F)>,
    seeded: &mut Vec<SeededPhi81LinearBlock>,
    matrix: &CcsMatrix<F>,
    slots: &[Option<(usize, usize)>],
    definitions: &LinearDefinitions,
    row_map: &[Option<usize>],
) -> Result<(), LowNormR1csError> {
    let mut append_csc = |csc: &CscMat<F>| -> Result<(), LowNormR1csError> {
        for field_col in 0..csc.ncols.min(slots.len()) {
            for index in csc.col_ptr[field_col]..csc.col_ptr[field_col + 1] {
                if let Some(target_row) = row_map[csc.row_idx[index]] {
                    append_field(trips, target_row, field_col, csc.vals[index], slots, definitions)?;
                }
            }
        }
        Ok(())
    };
    match matrix {
        CcsMatrix::Identity { n } => {
            for source_row in 0..(*n).min(row_map.len()).min(slots.len()) {
                if let Some(target_row) = row_map[source_row] {
                    append_field(trips, target_row, source_row, F::ONE, slots, definitions)?;
                }
            }
        }
        CcsMatrix::Csc(csc) => append_csc(csc)?,
        CcsMatrix::CscWithSeededPhi81 { csc, blocks } => {
            append_csc(csc)?;
            for block in blocks {
                let target_start = row_map[block.row_start()]
                    .ok_or_else(|| trace_error("seeded Phi81 block overlaps a removed Poseidon2 row"))?;
                for offset in 0..neo_math::D * block.kappa() {
                    if row_map[block.row_start() + offset] != Some(target_start + offset) {
                        return Err(trace_error(
                            "seeded Phi81 rows are not contiguous after selective lowering",
                        ));
                    }
                }
                let mut starts = Vec::with_capacity(block.word_starts().len());
                for &source_start in block.word_starts() {
                    let (target, width) =
                        slots[source_start].ok_or_else(|| trace_error("seeded Phi81 input bit was eliminated"))?;
                    if width != 1 {
                        return Err(trace_error("seeded Phi81 input is not a one-bit slot"));
                    }
                    for offset in 0..block.word_width() {
                        if slots[source_start + offset] != Some((target + offset, 1)) {
                            return Err(trace_error("seeded Phi81 input word is not contiguous"));
                        }
                    }
                    starts.push(target);
                }
                seeded.push(block.with_geometry(target_start, starts)?);
            }
        }
    }
    Ok(())
}

fn append_lc(
    trips: &mut Vec<(usize, usize, F)>,
    row: usize,
    lc: &Lc,
    slots: &[Option<(usize, usize)>],
    definitions: &LinearDefinitions,
) -> Result<(), LowNormR1csError> {
    append_lc_scaled(trips, row, lc, F::ONE, slots, definitions)
}

fn append_lc_scaled(
    trips: &mut Vec<(usize, usize, F)>,
    row: usize,
    lc: &Lc,
    scale: F,
    slots: &[Option<(usize, usize)>],
    definitions: &LinearDefinitions,
) -> Result<(), LowNormR1csError> {
    if lc.constant != F::ZERO {
        trips.push((row, 0, lc.constant * scale));
    }
    for &(field_col, coefficient) in &lc.terms {
        append_field(trips, row, field_col, coefficient * scale, slots, definitions)?;
    }
    Ok(())
}

fn append_field(
    trips: &mut Vec<(usize, usize, F)>,
    row: usize,
    field_col: usize,
    coefficient: F,
    slots: &[Option<(usize, usize)>],
    definitions: &LinearDefinitions,
) -> Result<(), LowNormR1csError> {
    if coefficient == F::ZERO {
        return Ok(());
    }
    let mut stack = vec![(field_col, coefficient)];
    while let Some((column, scale)) = stack.pop() {
        if column == 0 {
            trips.push((row, 0, scale));
            continue;
        }
        if let Some(rhs) = definitions.get(column) {
            if rhs.constant != F::ZERO {
                trips.push((row, 0, rhs.constant * scale));
            }
            stack.extend(
                rhs.terms
                    .iter()
                    .map(|&(rhs_column, rhs_coefficient)| (rhs_column, rhs_coefficient * scale)),
            );
            continue;
        }
        let (start, width) =
            slots[column].ok_or_else(|| trace_error("retained row references an unencoded selective temporary"))?;
        let radix = if width == BALANCED_FIELD_WIDTH {
            F::from_u64(3)
        } else {
            F::from_u64(2)
        };
        let mut power = scale;
        for bit in 0..width {
            trips.push((row, start + bit, power));
            power *= radix;
        }
    }
    Ok(())
}

fn append_slot(trips: &mut Vec<(usize, usize, F)>, row: usize, slot: (usize, usize), coefficient: F) {
    let (start, width) = slot;
    let radix = if width == BALANCED_FIELD_WIDTH {
        F::from_u64(3)
    } else {
        F::from_u64(2)
    };
    let mut power = coefficient;
    for bit in 0..width {
        trips.push((row, start + bit, power));
        power *= radix;
    }
}

fn lc_from_column(column: usize) -> Lc {
    Lc {
        terms: vec![(column, F::ONE)],
        constant: F::ZERO,
    }
}

fn trace_error(message: &str) -> LowNormR1csError {
    LowNormR1csError::SelectiveTrace(message.to_owned())
}
