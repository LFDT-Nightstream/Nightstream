//! Exact shape discovery for selective low-norm lowering.

use neo_ccs::{SparsePoly, Term};
use neo_math::{D, F};
use p3_field::{Field, PrimeCharacteristicRing};

use super::super::lowering::{DerivedProductSumEncoding, LowNormR1csError};
use super::{
    prepare_selective_layout, skipped_selective_rows, trace_error, SelectiveArmPlan, SelectiveLowNormWidthAudit,
    SparseR1cs, A, B, BALANCED_FIELD_WIDTH, BIT, C, CANON_BORROW, CANON_BOUND_DIGIT, CANON_DIGIT, CANON_NEXT_BORROW,
    CENTERED_UNIT, EVAL_GROUP_SIZE, EVAL_PAIRS, EVAL_SELECTOR, GENERAL_SELECTOR, SBOX_INPUT, SELECTIVE_ARITY,
};

pub(crate) struct SelectiveLowNormShape {
    pub rows: usize,
    pub columns: usize,
    pub polynomial: SparsePoly<F>,
    pub audit: SelectiveLowNormWidthAudit,
}

pub(crate) fn audit_multi_branch_selective_low_norm_shape_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLowNormShape, LowNormR1csError> {
    audit_multi_branch_selective_low_norm_shape_with_shared_bit_prefix(
        arms,
        shared_private_fields,
        shared_private_fields,
        modulus,
        residue,
    )
}

pub(crate) fn audit_multi_branch_selective_low_norm_shape_with_shared_bit_prefix(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLowNormShape, LowNormR1csError> {
    let layout = prepare_selective_layout(arms, shared_private_fields, shared_private_bit_fields, modulus, residue)?;
    let columns = layout.columns.next_multiple_of(D);
    let rows = count_structure_rows(
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
    Ok(SelectiveLowNormShape {
        rows,
        columns,
        polynomial: selective_polynomial(),
        audit: layout.audit,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn count_structure_rows(
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
) -> Result<usize, LowNormR1csError> {
    let mut rows = selectors.len();
    for source in 1..arms[0].m_in + shared_private_fields {
        if aliases[0][source].is_some() {
            continue;
        }
        if let Some((_, width)) = slots[0][source] {
            let source_proves_boolean = plans.iter().all(|plan| plan.source_boolean_rows[source]);
            if !source_proves_boolean && !plans[0].centered[source] && width != BALANCED_FIELD_WIDTH {
                rows += width;
            }
        }
    }
    for (arm_index, arm) in arms.iter().enumerate() {
        for source in arm.m_in + shared_private_fields..arm.m {
            if aliases[arm_index][source].is_some() || equal_aliases[arm_index][source].is_some() {
                continue;
            }
            if let Some((_, width)) = slots[arm_index][source] {
                if !plans[arm_index].source_boolean_rows[source]
                    && !plans[arm_index].centered[source]
                    && width != BALANCED_FIELD_WIDTH
                {
                    rows += width;
                }
            }
        }
    }

    rows += 1 + zero_padding_cols.len();
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
        rows += skipped.into_iter().filter(|skip| !skip).count();
        rows += arm
            .poseidon2_traces()
            .iter()
            .map(|trace| {
                trace.sboxes.len()
                    + trace
                        .output_cols
                        .iter()
                        .filter(|&&column| definitions.get(column).is_none())
                        .count()
            })
            .sum::<usize>();
        rows += arm
            .centered_unit_traces()
            .iter()
            .filter(|trace| plans[arm_index].widths[trace.value_col] == 0)
            .count();
        rows += 2 * BALANCED_FIELD_WIDTH * arm.shifted_ternary_canonical_traces().len();

        let mut derived_count = 0usize;
        for trace in arm.polynomial_evaluation_traces() {
            let groups = trace
                .coefficient_cols
                .len()
                .saturating_sub(1)
                .div_ceil(EVAL_GROUP_SIZE)
                .max(1);
            rows += 2 * groups;
            derived_count += 2 * groups.saturating_sub(1);
        }
        for batch in arm.product_sum_batch_traces() {
            for identity in &batch.identities {
                let groups = identity.factors.len().div_ceil(EVAL_GROUP_SIZE).max(1);
                rows += groups;
                derived_count += groups.saturating_sub(1);
            }
        }
        if derived_count != derived_product_sums[arm_index].len() {
            return Err(trace_error(
                "derived evaluation-product census drifted during row counting",
            ));
        }
    }
    Ok(rows + cols.next_multiple_of(D) - cols)
}

pub(super) fn selective_polynomial() -> SparsePoly<F> {
    let term = |coefficient: F, powers: &[(usize, u32)]| {
        let mut exps = vec![0u32; SELECTIVE_ARITY];
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

    // Exact shifted-base-3 transition over
    // d,h in {-1,0,1}, b in {0,1}:
    //
    //   b' = [d + 1 + b > h + 1].
    //
    // This is the Lagrange interpolation of the 18-point transition table.
    // GENERAL_SELECTOR isolates these rows from the evaluation ports reused
    // below. Its degree is six, within the existing degree-eight relation.
    let half = F::from_u64(2).inverse();
    let quarter = half * half;
    let transition = [
        (half, vec![(CANON_BOUND_DIGIT, 1)]),
        (F::ONE, vec![(CANON_NEXT_BORROW, 1)]),
        (-F::ONE, vec![(CANON_BORROW, 1)]),
        (-half, vec![(CANON_DIGIT, 1)]),
        (-half, vec![(CANON_BOUND_DIGIT, 2)]),
        (quarter, vec![(CANON_DIGIT, 1), (CANON_BOUND_DIGIT, 1)]),
        (-half, vec![(CANON_DIGIT, 2)]),
        (F::ONE, vec![(CANON_BORROW, 1), (CANON_BOUND_DIGIT, 2)]),
        (quarter, vec![(CANON_DIGIT, 1), (CANON_BOUND_DIGIT, 2)]),
        (-half, vec![(CANON_DIGIT, 1), (CANON_BORROW, 1), (CANON_BOUND_DIGIT, 1)]),
        (-quarter, vec![(CANON_DIGIT, 2), (CANON_BOUND_DIGIT, 1)]),
        (F::ONE, vec![(CANON_DIGIT, 2), (CANON_BORROW, 1)]),
        (F::from_u64(3) * quarter, vec![(CANON_DIGIT, 2), (CANON_BOUND_DIGIT, 2)]),
        (
            -F::from_u64(3) * half,
            vec![(CANON_DIGIT, 2), (CANON_BORROW, 1), (CANON_BOUND_DIGIT, 2)],
        ),
    ];
    for (coefficient, mut powers) in transition {
        powers.push((GENERAL_SELECTOR, 1));
        terms.push(term(coefficient, &powers));
    }
    SparsePoly::new(SELECTIVE_ARITY, terms)
}
