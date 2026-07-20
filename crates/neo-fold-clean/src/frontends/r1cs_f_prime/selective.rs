//! Selective low-norm CCS compiler for recorded verifier traces.
//!
//! Owns: selective slot planning, exact temporary substitution, retained-value
//! encodings, direct trace rows, and width-audit composition.
//!
//! Does not own: source trace recording, semantic proof of each trace family,
//! outer `F'` orchestration, or folding verification.
//!
//! Emits constraints: yes. It builds the selective CCS matrices and polynomial.
//!
//! Authority boundary: ordinary source rows remain the local implementation
//! arithmetic reference; a temporary is removed only when reconstructed from
//! retained constrained operands under the recorded trace contract. Protocol
//! sufficiency and necessity remain separate Lean obligations.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Selective layout | `prepare_selective_layout` | no | Recorded source traces |
//! | Compiler composition | [`build_multi_branch_selective_low_norm_r1cs_with_alignment`] | no | Prepared layout and exact emitter result |
//! | CCS matrix emission | `structure::build_structure` | yes | Retained source rows and selectors |
//! | Width audit | selective audit entrypoints | no | Exact prepared layout |

use std::collections::HashMap;

use neo_ccs::{CcsMatrix, CscMat};
use neo_math::{D, F};
use p3_field::{Field, PrimeCharacteristicRing};

use super::lowering::{DerivedProductSumEncoding, LowNormR1csError, MultiBranchLowNormR1cs};
use super::selective_audit::{
    retained_trace_widths, row_family_width_audits, SelectiveArmWidthAudit, SelectiveCompilerAudit,
    SelectiveLayoutAudit, SelectiveLowNormWidthAudit,
};
use super::SparseR1cs;
use crate::engine::r1cs_circuit::builder::{
    BalancedTernaryDecomposition, CanonicalU64Decomposition, ProductFactorTrace,
};
use crate::engine::r1cs_circuit::Lc;

#[path = "selective_canonical.rs"]
mod canonical;
#[path = "selective_emit.rs"]
mod emit;
#[path = "selective_projected_decoder.rs"]
mod projected_decoder;
#[path = "selective_projected_rows.rs"]
mod projected_rows;
#[path = "selective_rows.rs"]
mod rows;
#[path = "selective_shape.rs"]
mod shape;
#[path = "selective_structure.rs"]
mod structure;
#[path = "selective_terms.rs"]
mod terms;
use emit::{lc_from_column, trace_error};
pub(super) use projected_decoder::SelectiveProjectedSourceResolution;
pub(crate) use projected_rows::{project_rows_with_alignment, project_rows_with_source_provenance_with_alignment};
pub use projected_rows::{
    SelectiveProjectedDerivedProductSum, SelectiveProjectedGeometricRun, SelectiveProjectedPort,
    SelectiveProjectedProductFactor, SelectiveProjectedPublicCoordinate, SelectiveProjectedPublicCoordinateSource,
    SelectiveProjectedRetainedStep, SelectiveProjectedRewriteOutput, SelectiveProjectedRewriteStep,
    SelectiveProjectedRowArtifact, SelectiveProjectedRowsAudit, SelectiveProjectedSourceDefinition,
    SelectiveProjectedSourceLinearCombination, SelectiveProjectedSourceProvenance, SelectiveProjectedSourceSlot,
    SelectiveProjectedSourceTerm, SelectiveProjectedTerm,
};
use rows::{skipped_selective_rows, PreparedSelectiveRows};
pub(crate) use shape::{
    audit_multi_branch_selective_low_norm_shape_with_alignment,
    audit_multi_branch_selective_low_norm_shape_with_shared_bit_prefix, SelectiveLowNormShape,
};

pub(super) const EVAL_GROUP_SIZE: usize = 5;
const BALANCED_FIELD_WIDTH: usize = 41;
const BINARY_FIELD_WIDTH: usize = 64;
const BIT: usize = 0;
const GENERAL_SELECTOR: usize = 1;
const A: usize = 2;
const B: usize = 3;
const C: usize = 4;
const SBOX_INPUT: usize = 5;
const CENTERED_UNIT: usize = 6;
const EVAL_SELECTOR: usize = 7;
// These ports are shared with the last three evaluation pairs. Canonical
// rows set GENERAL_SELECTOR and leave EVAL_SELECTOR zero; evaluation rows do
// the converse, so the two direct row families remain disjoint.
const CANON_DIGIT: usize = 8;
const CANON_BORROW: usize = 9;
const CANON_NEXT_BORROW: usize = 10;
const CANON_BOUND_DIGIT: usize = 11;
const EVAL_PAIRS: [(usize, usize); EVAL_GROUP_SIZE] =
    [(BIT, A), (B, SBOX_INPUT), (CENTERED_UNIT, 8), (9, 10), (11, 12)];
const SELECTIVE_ARITY: usize = 13;

struct SelectiveLayout {
    plans: Vec<SelectiveArmPlan>,
    slots: Vec<Vec<Option<(usize, usize)>>>,
    aliases: Vec<Vec<Option<(usize, usize)>>>,
    equal_aliases: Vec<Vec<Option<usize>>>,
    derived_product_sums: Vec<Vec<DerivedProductSumEncoding>>,
    selector_cols: Vec<usize>,
    public_padding_cols: Vec<usize>,
    private_padding_cols: Vec<usize>,
    public_input_len: usize,
    columns: usize,
    prepared_rows: PreparedSelectiveRows,
    compiler_audit: SelectiveCompilerAudit,
}
struct SelectiveArmPlan {
    widths: Vec<usize>,
    centered: Vec<bool>,
    source_boolean_rows: Vec<bool>,
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

/// Compile one-hot field-R1CS arms to selective degree-eight CCS.
///
/// Two distinct zero regions are inserted and accounted independently:
/// the logical public prefix is first completed to `modulus`, then selectors
/// are allocated, and a second region places shared private advice at
/// `residue (mod modulus)`. This ordering keeps branch selectors out of the
/// active SuperNeo public ring carrier.
pub fn build_multi_branch_selective_low_norm_r1cs_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<MultiBranchLowNormR1cs, LowNormR1csError> {
    build_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
        arms,
        shared_private_fields,
        shared_private_fields,
        modulus,
        residue,
    )
}

/// Project selected exact rows from the same emitter term stream used by the
/// full selective relation, without allocating arrays over every final column.
#[doc(hidden)]
pub fn audit_multi_branch_selective_rows_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
    selected_rows: &[usize],
) -> Result<SelectiveProjectedRowsAudit, LowNormR1csError> {
    project_rows_with_alignment(
        arms,
        shared_private_fields,
        shared_private_fields,
        modulus,
        residue,
        selected_rows,
    )
}

pub fn build_multi_branch_selective_low_norm_r1cs_with_shared_bit_prefix(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<MultiBranchLowNormR1cs, LowNormR1csError> {
    let layout = prepare_selective_layout(arms, shared_private_fields, shared_private_bit_fields, modulus, residue)?;
    let structure = structure::build_structure(
        arms,
        &layout.plans,
        &layout.slots,
        &layout.aliases,
        &layout.equal_aliases,
        shared_private_fields,
        &layout.derived_product_sums,
        &layout.selector_cols,
        &layout.public_padding_cols,
        &layout.private_padding_cols,
        layout.columns,
        &layout.prepared_rows,
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
        Some(layout.compiler_audit),
    ))
}

/// Compute selective-lowering width without constructing output matrices.
/// Public-carrier and private-alignment padding remain separate audit fields.
pub fn audit_multi_branch_selective_low_norm_width_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLowNormWidthAudit, LowNormR1csError> {
    audit_multi_branch_selective_low_norm_width_with_shared_bit_prefix(
        arms,
        shared_private_fields,
        shared_private_fields,
        modulus,
        residue,
    )
}

pub fn audit_multi_branch_selective_low_norm_width_with_shared_bit_prefix(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLowNormWidthAudit, LowNormR1csError> {
    Ok(
        prepare_selective_layout(arms, shared_private_fields, shared_private_bit_fields, modulus, residue)?
            .compiler_audit
            .into_width(),
    )
}

fn prepare_selective_layout(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLayout, LowNormR1csError> {
    if arms.len() < 2 {
        return Err(LowNormR1csError::TooFewArms(arms.len()));
    }
    if modulus == 0 {
        return Err(LowNormR1csError::ZeroAlignmentModulus);
    }
    if shared_private_bit_fields > shared_private_fields {
        return Err(trace_error("shared bit prefix exceeds shared private prefix"));
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
        .map(|arm| selective_arm_plan(arm, shared_private_fields, shared_private_bit_fields))
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
    let logical_public_input_len = cursor;
    let public_padding_len = (modulus - cursor % modulus) % modulus;
    let public_padding_cols = (cursor..cursor + public_padding_len).collect::<Vec<_>>();
    cursor += public_padding_len;
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
    let private_padding_cols = (cursor..cursor + padding_len).collect::<Vec<_>>();
    cursor += padding_len;
    let shared_private_start = cursor;

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
    let width_audit = SelectiveLowNormWidthAudit {
        constant_coordinate: 1,
        logical_public_coordinates: logical_public_input_len - 1,
        public_carrier_padding: public_padding_cols.len(),
        public_coordinates: public_input_len - 1,
        selector_coordinates: selector_cols.len(),
        alignment_padding: private_padding_cols.len(),
        shared_private_coordinates: branch_start - public_input_len - selector_cols.len() - private_padding_cols.len(),
        branch_start,
        arms: arms_audit,
        total_coordinates: cursor,
    };
    let layout_audit = SelectiveLayoutAudit::from_prepared_layout(
        logical_public_input_len,
        public_input_len,
        public_padding_cols.clone(),
        selector_cols.clone(),
        private_padding_cols.clone(),
        shared_private_start..branch_start,
        branch_start..cursor,
        cursor..cursor.next_multiple_of(D),
    );
    let prepared_rows = PreparedSelectiveRows::prepare(
        arms,
        &plans,
        &slots,
        &aliases,
        &equal_aliases,
        shared_private_fields,
        &derived_product_sums,
        selector_cols.len(),
        public_padding_cols.len(),
        private_padding_cols.len(),
        cursor,
    )?;
    let row_audit = prepared_rows.audit();
    Ok(SelectiveLayout {
        plans,
        slots,
        aliases,
        equal_aliases,
        derived_product_sums,
        selector_cols,
        public_padding_cols,
        private_padding_cols,
        public_input_len,
        columns: cursor,
        prepared_rows,
        compiler_audit: SelectiveCompilerAudit::new(
            layout_audit,
            width_audit,
            row_audit,
            arms.iter()
                .map(|arm| arm.physical_stage_ranges().to_vec())
                .collect(),
        ),
    })
}

fn selective_arm_plan(
    arm: &SparseR1cs,
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
) -> Result<SelectiveArmPlan, LowNormR1csError> {
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
    for trace in arm.shifted_ternary_canonical_traces() {
        let digit_end = trace.digit_columns_start + BALANCED_FIELD_WIDTH;
        let negative_end = trace.negative_columns_start + BALANCED_FIELD_WIDTH;
        let borrow_end = trace.borrow_columns_start + BALANCED_FIELD_WIDTH - 1;
        if digit_end > arm.m || negative_end > arm.m || borrow_end > arm.m {
            return Err(trace_error("shifted-ternary trace columns exceed the source arm"));
        }
        for column in trace.negative_columns_start..negative_end {
            eliminated[column] = true;
        }
        for column in trace.digit_columns_start..digit_end {
            eliminated[column] = false;
        }
        for column in trace.borrow_columns_start..borrow_end {
            eliminated[column] = false;
        }
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
    // The public prefix and the leading part of the shared private prefix are
    // verifier-owned bit surfaces. Remaining shared fields retain the widths
    // inferred from their identical per-arm relations.
    let shared_bit_end = arm.m_in + shared_private_bit_fields;
    widths[1..arm.m_in].fill(1);
    widths[arm.m_in..shared_bit_end].fill(1);
    centered[..shared_bit_end].fill(false);
    let equality_roots = propagate_low_norm_equalities(arm, &mut widths, &mut centered, shared_bit_end)?;
    let mut removed_rows = skipped_selective_rows(arm)?;
    for definition in &definitions.entries {
        if let Some(row) = definition.row {
            removed_rows[row] = true;
        }
    }
    let mut source_boolean_rows = vec![false; arm.m];
    for &(column, row) in arm.boolean_constraint_rows() {
        if column < arm.m && row < arm.n && !removed_rows[row] {
            source_boolean_rows[column] = true;
        }
    }
    Ok(SelectiveArmPlan {
        widths,
        centered,
        source_boolean_rows,
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
    if let Some((target, dependency)) = entries.iter().find_map(|definition| {
        definition.rhs.terms.iter().find_map(|&(column, _)| {
            (column >= definition.target || column >= directly_eliminated.len() || directly_eliminated[column])
                .then_some((definition.target, column))
        })
    }) {
        return Err(trace_error(&format!(
            "linear definition for column {target} is not acyclic over retained dependency {dependency}"
        )));
    }
    Ok(LinearDefinitions { by_column, entries })
}

fn for_each_explicit_term(matrix: &CcsMatrix<F>, mut visit: impl FnMut(usize, usize, F)) {
    let mut visit_csc = |csc: &CscMat<F>| {
        for column in 0..csc.ncols {
            for index in csc.column_range(column) {
                visit(csc.row_index(index), column, csc.vals[index]);
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
