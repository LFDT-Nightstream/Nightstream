//! Terminal delayed projection over the exact raw witnesses opened by Ajtai.
//!
//! This is the concrete one-fold-delay closure.  It consumes the verifier's
//! pending 19-coordinate old block and the fourteen ordered `FinalWitnessWires`
//! allocations.  It never reads a child `CeClaim.y_zcol` or a digest.
//!
//! | Stable leaf | Equation | Multiplicity |
//! |---|---|---|
//! | `terminal.raw_old_block_projection.tensor` | live-prefix `chi(old_block[..T])` recurrence | `5 * Σ_{j<T} min(B, 2^j)` rows |
//! | `terminal.raw_old_block_projection.products` | prefix weight times `Σ_i b^i Z_i[lane,block]` in two limbs | `2 * D * B` rows |
//! | `terminal.raw_old_block_projection.final_scale` | lane sum times common final low-bit factor | `5 * D` rows when profitable |
//! | `terminal.raw_old_block_projection.terminal` | product sums equal `pending.parent_y_zcol` | `2 * D` rows |
//!
//! `B = ceil(m / D)`.  The 10 lanes `D..64` are verifier-computed zero and
//! intentionally own no rows.  Commitment binding is owned separately by the
//! terminal Ajtai relation over the identical witness allocations.

use neo_math::D;

use crate::engine::r1cs_circuit::field_ext::{enforce_k_mul, KVar};
use crate::engine::r1cs_circuit::{
    R1csBuilder, RawOldBlockProjectionColumnMap, RawOldBlockProjectionPlan, RawOldBlockProjectionProgram,
    TerminalPendingProjectionAudit, RAW_OLD_BLOCK_CHILD_COUNT, RAW_OLD_BLOCK_PENDING_JOIN_ID,
};

use super::witness::FinalWitnessWires;

#[derive(Debug)]
pub(crate) struct RawOldBlockProjectionError {
    what: &'static str,
    expected: usize,
    got: usize,
}

impl RawOldBlockProjectionError {
    pub(crate) fn what(&self) -> &'static str {
        self.what
    }

    pub(crate) fn expected(&self) -> usize {
        self.expected
    }

    pub(crate) fn got(&self) -> usize {
        self.got
    }
}

/// Enforce the pending parent directly from the ordered raw witness family.
///
/// For every active lane `rho`, the terminal equation is
///
/// `parent[rho] = Σ_block χ_old(block) · Σ_child radix^child · Z_child[rho, block]`.
///
/// The equality tensor prefix is expanded once. At every prefix node only one child
/// is multiplied: when both are live, `high = parent * r` and
/// `low = parent - high`; when only the low child is live, it is multiplied by
/// `1-r` directly. When every block's last bit is zero and doing so saves
/// rows, the common final factor is multiplied once per lane after summation.
/// This gives the exact row count exposed by
/// [`RawOldBlockProjectionPlan`].
pub(crate) fn enforce_raw_old_block_projection(
    builder: &mut R1csBuilder,
    logical_columns: usize,
    old_block: &[KVar],
    parent_y_zcol: &[KVar],
    witnesses: &[FinalWitnessWires],
    radix: u32,
) -> Result<(), RawOldBlockProjectionError> {
    let placement = production_placement(builder, logical_columns, old_block, parent_y_zcol, witnesses, radix)?;
    let plan = placement.plan;
    let program = placement.program;
    let column_map = &placement.column_map;

    let row_start = placement.row_start;
    let tensor_row_start = builder.rows();
    let tensor_first_allocated_column = builder.cols();
    for round in 0..plan.tensor_variables() {
        let count = plan
            .tensor_round_mul_count(round)
            .expect("validated tensor round");
        for parent in 0..count {
            debug_assert_eq!(
                builder.rows() - row_start,
                plan.tensor_row_offset(round, parent, 0)
                    .expect("validated tensor operation")
            );
            let (left, right, _, _, _, canonical_output) = program
                .tensor_operation(round, parent)
                .expect("validated tensor operation");
            let actual_left = column_map
                .map_klc(&left)
                .expect("indexed tensor left operand maps to production wires");
            let actual_right = column_map
                .map_klc(&right)
                .expect("indexed tensor right operand maps to production wires");
            let actual_output = enforce_k_mul(builder, &actual_left, &actual_right);
            debug_assert_eq!(
                [actual_output.c0.col(), actual_output.c1.col()],
                [
                    column_map
                        .canonical_to_actual(canonical_output.c0.col())
                        .expect("indexed tensor c0 output maps to production wires"),
                    column_map
                        .canonical_to_actual(canonical_output.c1.col())
                        .expect("indexed tensor c1 output maps to production wires"),
                ]
            );
        }
    }
    let tensor_rows = tensor_row_start..builder.rows();
    builder.record_row_family("terminal.raw_old_block_projection.tensor", tensor_row_start);

    let weights = (0..plan.packed_columns())
        .map(|block| {
            let canonical = program.chi_terms(block).expect("validated packed block");
            column_map
                .map_klc_owned(canonical)
                .expect("indexed block weight maps to production wires")
        })
        .collect::<Vec<_>>();

    let projection_product_row_start = builder.rows();
    let projection_product_first_allocated_column = builder.cols();
    for lane in 0..plan.active_lanes() {
        for (block, weight) in weights.iter().enumerate() {
            let recomposed_digit = column_map
                .map_lc_owned(
                    program
                        .raw_terms(lane, block)
                        .expect("validated active lane and packed block"),
                )
                .expect("indexed raw-child recomposition maps to production wires");
            let term_c0 = builder.alloc_mul(&recomposed_digit, &weight.c0);
            let term_c1 = builder.alloc_mul(&recomposed_digit, &weight.c1);
            debug_assert_eq!(
                [term_c0.col(), term_c1.col()],
                [
                    column_map
                        .canonical_to_actual(
                            program
                                .layout()
                                .product_column(lane, block, 0)
                                .expect("validated product c0"),
                        )
                        .expect("indexed product c0 maps to production wires"),
                    column_map
                        .canonical_to_actual(
                            program
                                .layout()
                                .product_column(lane, block, 1)
                                .expect("validated product c1"),
                        )
                        .expect("indexed product c1 maps to production wires"),
                ]
            );
        }
    }
    let projection_product_rows = projection_product_row_start..builder.rows();
    builder.record_row_family(
        "terminal.raw_old_block_projection.products",
        projection_product_row_start,
    );

    let final_scale_row_start = builder.rows();
    let final_scale_first_allocated_column = builder.cols();
    for lane in 0..plan.final_scale_mul_count() {
        debug_assert_eq!(
            builder.rows() - row_start,
            plan.final_scale_row_offset(lane, 0)
                .expect("validated final-scale lane")
        );
        let (left, right, _, _, _, canonical_output) = program
            .final_scale_operation(lane)
            .expect("validated final-scale operation");
        let actual_left = column_map
            .map_klc(&left)
            .expect("indexed final-scale left operand maps to production wires");
        let actual_right = column_map
            .map_klc(&right)
            .expect("indexed final-scale right operand maps to production wires");
        let actual_output = enforce_k_mul(builder, &actual_left, &actual_right);
        debug_assert_eq!(
            [actual_output.c0.col(), actual_output.c1.col()],
            [
                column_map
                    .canonical_to_actual(canonical_output.c0.col())
                    .expect("indexed final-scale c0 output maps to production wires"),
                column_map
                    .canonical_to_actual(canonical_output.c1.col())
                    .expect("indexed final-scale c1 output maps to production wires"),
            ]
        );
    }
    let final_scale_rows = final_scale_row_start..builder.rows();
    if !final_scale_rows.is_empty() {
        builder.record_row_family("terminal.raw_old_block_projection.final_scale", final_scale_row_start);
    }

    let terminal_row_start = builder.rows();
    for lane in 0..plan.active_lanes() {
        for limb in 0..2 {
            debug_assert_eq!(
                builder.rows() - row_start,
                plan.terminal_row_offset(lane, limb)
                    .expect("validated terminal lane and limb")
            );
            let (canonical_parent, canonical_sum) = program
                .terminal_operands(lane, limb)
                .expect("validated terminal lane and limb");
            let actual_parent = column_map
                .map_lc(&canonical_parent)
                .expect("indexed terminal parent maps to production wires");
            let actual_sum = column_map
                .map_lc(&canonical_sum)
                .expect("indexed terminal sum maps to production wires");
            builder.enforce_eq(&actual_parent, &actual_sum);
        }
    }
    let terminal_rows = terminal_row_start..builder.rows();
    builder.record_row_family("terminal.raw_old_block_projection.terminal", terminal_row_start);

    let expected_column_stop = placement.final_scale_first_allocated_column + plan.final_scale_rows();
    if tensor_rows != placement.tensor_rows
        || projection_product_rows != placement.projection_product_rows
        || final_scale_rows != placement.final_scale_rows
        || terminal_rows != placement.terminal_rows
        || builder.rows() != placement.row_end
        || builder.cols() != expected_column_stop
        || tensor_first_allocated_column != placement.tensor_first_allocated_column
        || projection_product_first_allocated_column != placement.projection_product_first_allocated_column
        || final_scale_first_allocated_column != placement.final_scale_first_allocated_column
    {
        return Err(shape(
            "emitted raw-old-block placement matches its production plan",
            placement.row_end,
            builder.rows(),
        ));
    }

    builder.record_terminal_pending_projection(placement);
    builder.record_row_family("terminal.raw_old_block_projection", row_start);
    Ok(())
}

/// Construct the compact placement consumed by both the production emitter
/// and the placement-only artifact exporter. No row or assignment payload is
/// materialized here.
pub(crate) fn production_placement(
    builder: &R1csBuilder,
    logical_columns: usize,
    old_block: &[KVar],
    parent_y_zcol: &[KVar],
    witnesses: &[FinalWitnessWires],
    radix: u32,
) -> Result<TerminalPendingProjectionAudit, RawOldBlockProjectionError> {
    if witnesses.len() != RAW_OLD_BLOCK_CHILD_COUNT {
        return Err(shape(
            "ordered raw witness child count",
            RAW_OLD_BLOCK_CHILD_COUNT,
            witnesses.len(),
        ));
    }
    let plan = RawOldBlockProjectionPlan::new(logical_columns, witnesses.len()).map_err(|_| {
        shape(
            "logical columns within the fixed block domain",
            neo_reductions::block_projection::BLOCK_PROJECTION_DOMAIN_SIZE * D,
            logical_columns,
        )
    })?;
    if old_block.len() != plan.block_variables() {
        return Err(shape(
            "pending old-block coordinates",
            plan.block_variables(),
            old_block.len(),
        ));
    }
    if parent_y_zcol.len() != plan.active_lanes() {
        return Err(shape(
            "pending parent active lanes",
            plan.active_lanes(),
            parent_y_zcol.len(),
        ));
    }
    for witness in witnesses {
        if witness.rows != plan.packed_rows() || witness.cols != plan.packed_columns() {
            return Err(shape(
                "raw witness packed entries",
                plan.packed_rows() * plan.packed_columns(),
                witness.rows * witness.cols,
            ));
        }
    }

    if plan.logical_columns() != plan.active_lanes() * plan.packed_columns() {
        return Err(shape(
            "logical columns exactly fill active packed lanes",
            plan.active_lanes() * plan.packed_columns(),
            plan.logical_columns(),
        ));
    }
    if radix == 0 {
        return Err(shape("nonzero raw-witness radix", 1, 0));
    }

    let program = RawOldBlockProjectionProgram::new(plan, radix).map_err(|what| shape(what, 1, 0))?;
    let pending_old_block_cols = old_block
        .iter()
        .map(|value| [value.c0.col(), value.c1.col()])
        .collect::<Vec<_>>();
    let parent_y_zcol_cols = parent_y_zcol
        .iter()
        .map(|value| [value.c0.col(), value.c1.col()])
        .collect::<Vec<_>>();
    let child_witness_first_columns = witnesses
        .iter()
        .map(|witness| {
            let first = witness.values.first().ok_or_else(|| {
                shape(
                    "raw witness allocation has a first column",
                    plan.packed_rows() * plan.packed_columns(),
                    0,
                )
            })?;
            let first = first.col();
            if let Some((offset, wire)) = witness
                .values
                .iter()
                .enumerate()
                .find(|(offset, wire)| wire.col() != first + *offset)
            {
                return Err(shape(
                    "raw witness allocation is contiguous lane-major/block-minor",
                    first + offset,
                    wire.col(),
                ));
            }
            Ok(first)
        })
        .collect::<Result<Vec<_>, RawOldBlockProjectionError>>()?;

    let row_start = builder.rows();
    let first_allocated_column = builder.cols();
    let tensor_first_allocated_column = builder.cols();
    let projection_product_first_allocated_column = tensor_first_allocated_column + plan.tensor_rows();
    let final_scale_first_allocated_column = projection_product_first_allocated_column + plan.projection_product_rows();
    let column_map = RawOldBlockProjectionColumnMap::new(
        program.layout(),
        pending_old_block_cols.clone(),
        parent_y_zcol_cols.clone(),
        child_witness_first_columns.clone(),
        tensor_first_allocated_column,
        projection_product_first_allocated_column,
        final_scale_first_allocated_column,
    )
    .map_err(|what| shape(what, 1, 0))?;

    let tensor_rows = row_start..row_start + plan.tensor_rows();
    let projection_product_rows = tensor_rows.end..tensor_rows.end + plan.projection_product_rows();
    let final_scale_rows = projection_product_rows.end..projection_product_rows.end + plan.final_scale_rows();
    let terminal_rows = final_scale_rows.end..final_scale_rows.end + plan.terminal_rows();
    Ok(TerminalPendingProjectionAudit {
        pending_projection_join_id: RAW_OLD_BLOCK_PENDING_JOIN_ID,
        row_start,
        row_end: terminal_rows.end,
        first_allocated_column,
        plan,
        program,
        column_map,
        radix,
        tensor_rows,
        tensor_first_allocated_column,
        projection_product_rows,
        projection_product_first_allocated_column,
        final_scale_rows,
        final_scale_first_allocated_column,
        terminal_rows,
        pending_old_block_cols,
        parent_y_zcol_cols,
        projection_child_witness_first_columns: child_witness_first_columns,
        ajtai_child_witness_first_columns: Vec::new(),
    })
}

fn shape(what: &'static str, expected: usize, got: usize) -> RawOldBlockProjectionError {
    RawOldBlockProjectionError { what, expected, got }
}
