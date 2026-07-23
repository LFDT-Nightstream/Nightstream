//! Focused direct terminal raw-old-block projection regressions.

use super::*;

fn small_projection_fixture() -> (
    usize,
    [K; neo_reductions::block_projection::BLOCK_PROJECTION_POINT_LEN],
    Vec<Mat<F>>,
    [K; D],
) {
    let logical_columns = 2 * D;
    let witnesses = (0..14)
        .map(|child| {
            Mat::from_row_major(
                D,
                2,
                (0..D)
                    .flat_map(|lane| {
                        (0..2).map(move |block| F::from_u64((101 * child + 7 * lane + 3 * block + 1) as u64))
                    })
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let old_block = std::array::from_fn(|index| K::from(F::from_u64((index + 2) as u64)));
    let parent = neo_reductions::block_projection::radix_recompose_raw_witnesses_at_block_point(
        &witnesses,
        logical_columns,
        &old_block,
        K::from(F::from_u64(2)),
    )
    .expect("native raw-old-block recomposition");
    (logical_columns, old_block, witnesses, parent)
}

fn factorized_projection_fixture() -> (
    usize,
    [K; neo_reductions::block_projection::BLOCK_PROJECTION_POINT_LEN],
    Vec<Mat<F>>,
    [K; D],
) {
    let packed_columns = D + 1;
    let logical_columns = packed_columns * D;
    let witnesses = (0..14)
        .map(|child| {
            Mat::from_row_major(
                D,
                packed_columns,
                (0..D)
                    .flat_map(|lane| {
                        (0..packed_columns).map(move |block| {
                            if child == 0 && lane == 0 && block == 0 {
                                F::ONE
                            } else {
                                F::ZERO
                            }
                        })
                    })
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let old_block = std::array::from_fn(|index| K::from(F::from_u64((index + 2) as u64)));
    let parent = neo_reductions::block_projection::radix_recompose_raw_witnesses_at_block_point(
        &witnesses,
        logical_columns,
        &old_block,
        K::from(F::from_u64(2)),
    )
    .expect("native factorized raw-old-block recomposition");
    (logical_columns, old_block, witnesses, parent)
}

#[test]
fn terminal_pending_parent_is_direct_raw_witness_recomposition() {
    let (logical_columns, old_block, witnesses, parent) = small_projection_fixture();
    let honest = enforce_terminal_raw_old_block_projection_against(logical_columns, &old_block, &parent, &witnesses, 2)
        .expect("emit terminal raw-old-block projection");
    assert!(honest.builder.is_satisfied(), "honest projection must satisfy");
    let plan = neo_fold_clean::engine::r1cs_circuit::RawOldBlockProjectionPlan::new(logical_columns, 14)
        .expect("small projection plan");
    assert_eq!(honest.builder.rows(), plan.total_rows(), "exact plan row formula");
    let audit = &honest.builder.terminal_pending_projection_audits()[0];
    assert_eq!(audit.plan, plan);
    assert_eq!(audit.tensor_rows.len(), plan.tensor_rows());
    assert_eq!(audit.projection_product_rows.len(), plan.projection_product_rows());
    assert_eq!(audit.final_scale_rows.len(), plan.final_scale_rows());
    assert_eq!(audit.terminal_rows.len(), plan.terminal_rows());
    let lane = 1;
    let block = 0;
    let product_row = audit.row_start
        + plan
            .projection_product_row_offset(lane, block, 0)
            .expect("in-range product row");
    let (a_triplets, _, _) = honest.builder.sparse_triplets();
    let mut actual_columns = a_triplets
        .iter()
        .filter_map(|(row, column, _)| (*row == product_row).then_some(*column))
        .collect::<Vec<_>>();
    actual_columns.sort_unstable();
    let mut expected_columns = audit
        .projection_child_witness_first_columns
        .iter()
        .map(|first| first + lane * plan.packed_columns() + block)
        .collect::<Vec<_>>();
    expected_columns.sort_unstable();
    assert_eq!(
        actual_columns, expected_columns,
        "product row uses row-major raw columns"
    );
    assert_ne!(
        expected_columns[0],
        audit.projection_child_witness_first_columns[0] + block * plan.active_lanes() + lane,
        "row-major lane/block mapping must differ from the rejected block-major mapping",
    );

    let mut witness_tamper =
        enforce_terminal_raw_old_block_projection_against(logical_columns, &old_block, &parent, &witnesses, 2)
            .expect("emit witness mutation");
    let witness_column = witness_tamper.raw_witness_probe;
    witness_tamper.builder.tamper_witness(
        witness_column,
        witness_tamper.builder.witness()[witness_column] + F::ONE,
    );
    assert!(!witness_tamper.builder.is_satisfied(), "raw witness mutation must fail");

    let mut parent_tamper =
        enforce_terminal_raw_old_block_projection_against(logical_columns, &old_block, &parent, &witnesses, 2)
            .expect("emit parent mutation");
    let parent_column = parent_tamper.parent_c0_probe;
    parent_tamper
        .builder
        .tamper_witness(parent_column, parent_tamper.builder.witness()[parent_column] + F::ONE);
    assert!(
        !parent_tamper.builder.is_satisfied(),
        "pending parent mutation must fail"
    );

    let mut reordered = witnesses.clone();
    reordered.swap(0, 1);
    let reordered =
        enforce_terminal_raw_old_block_projection_against(logical_columns, &old_block, &parent, &reordered, 2)
            .expect("emit child-order mutation");
    assert!(!reordered.builder.is_satisfied(), "child-order mutation must fail");

    let mut changed_point = old_block;
    changed_point[0] += K::ONE;
    let changed_point =
        enforce_terminal_raw_old_block_projection_against(logical_columns, &changed_point, &parent, &witnesses, 2)
            .expect("emit old-point mutation");
    assert!(!changed_point.builder.is_satisfied(), "old-block mutation must fail");

    let changed_radix =
        enforce_terminal_raw_old_block_projection_against(logical_columns, &old_block, &parent, &witnesses, 3)
            .expect("emit radix mutation");
    assert!(!changed_radix.builder.is_satisfied(), "radix mutation must fail");
}

#[test]
fn production_plan_factors_the_common_final_block_coordinate() {
    let plan = neo_fold_clean::engine::r1cs_circuit::RawOldBlockProjectionPlan::new(11_437_038, 14)
        .expect("production raw-old-block plan");
    assert_eq!(plan.packed_columns(), 211_797);
    assert!(plan.factor_final_round());
    assert_eq!(plan.block_variables(), 19);
    assert_eq!(plan.tensor_variables(), 18);
    assert_eq!(plan.factored_variable(), Some(18));
    assert_eq!(plan.tensor_mul_count(), 262_143);
    assert_eq!(plan.tensor_rows(), 1_310_715);
    assert_eq!(plan.projection_product_rows(), 22_874_076);
    assert_eq!(plan.final_scale_mul_count(), 54);
    assert_eq!(plan.final_scale_rows(), 270);
    assert_eq!(plan.terminal_rows(), 108);
    assert_eq!(plan.total_rows(), 24_185_169);
    assert_eq!(
        plan.tensor_rows() + plan.projection_product_rows() + plan.final_scale_rows(),
        24_185_061,
        "exact derived-column count",
    );
}

#[test]
fn factorized_final_round_uses_raw_children_and_binds_the_pending_parent() {
    let (logical_columns, old_block, witnesses, parent) = factorized_projection_fixture();
    let honest = enforce_terminal_raw_old_block_projection_against(logical_columns, &old_block, &parent, &witnesses, 2)
        .expect("emit factorized projection");
    assert!(honest.builder.is_satisfied(), "honest factorized projection");
    let audit = &honest.builder.terminal_pending_projection_audits()[0];
    assert!(audit.plan.factor_final_round());
    assert_eq!(audit.plan.factored_variable(), Some(18));
    assert_eq!(audit.final_scale_rows.len(), 5 * D);

    let mut changed_point = old_block;
    changed_point[18] += K::ONE;
    let changed_point =
        enforce_terminal_raw_old_block_projection_against(logical_columns, &changed_point, &parent, &witnesses, 2)
            .expect("emit final-coordinate mutation");
    assert!(!changed_point.builder.is_satisfied(), "oldBlock[18] mutation must fail");

    let mut reordered = witnesses.clone();
    reordered.swap(0, 1);
    let reordered =
        enforce_terminal_raw_old_block_projection_against(logical_columns, &old_block, &parent, &reordered, 2)
            .expect("emit raw-child order mutation");
    assert!(!reordered.builder.is_satisfied(), "raw-child order mutation must fail");

    let mut child_tamper =
        enforce_terminal_raw_old_block_projection_against(logical_columns, &old_block, &parent, &witnesses, 2)
            .expect("emit raw-child trace mutation");
    let child_column = child_tamper.raw_witness_probe;
    child_tamper
        .builder
        .tamper_witness(child_column, child_tamper.builder.witness()[child_column] + F::ONE);
    assert!(!child_tamper.builder.is_satisfied(), "raw-child mutation must fail");

    let mut parent_tamper =
        enforce_terminal_raw_old_block_projection_against(logical_columns, &old_block, &parent, &witnesses, 2)
            .expect("emit pending-parent mutation");
    let parent_column = parent_tamper.parent_c0_probe;
    parent_tamper
        .builder
        .tamper_witness(parent_column, parent_tamper.builder.witness()[parent_column] + F::ONE);
    assert!(
        !parent_tamper.builder.is_satisfied(),
        "pending-parent mutation must fail"
    );

    let mut scale_tamper =
        enforce_terminal_raw_old_block_projection_against(logical_columns, &old_block, &parent, &witnesses, 2)
            .expect("emit final-scale output mutation");
    let scale_column = scale_tamper
        .final_scale_c0_probe
        .expect("factorized mode owns a final-scale output");
    scale_tamper
        .builder
        .tamper_witness(scale_column, scale_tamper.builder.witness()[scale_column] + F::ONE);
    assert!(
        !scale_tamper.builder.is_satisfied(),
        "final-scale output mutation must fail"
    );
}

#[test]
fn direct_projection_and_ajtai_consume_the_same_final_witness_wires() {
    let (prep, finished) = build_honest_finished_proof(2);
    let final_running = finished
        .proof
        .state
        .proof
        .running()
        .expect("finished proof has final running");
    let old_block = std::array::from_fn(|index| K::from(F::from_u64((index + 3) as u64)));
    let parent = neo_reductions::block_projection::radix_recompose_raw_witnesses_at_block_point(
        &final_running.witnesses,
        prep.structure().m,
        &old_block,
        K::from(F::from_u64(prep.params.b() as u64)),
    )
    .expect("bounded raw-old-block recomposition");
    let builder = enforce_ce_relations_many_with_raw_pending_against(
        &prep,
        &final_running.claims,
        &final_running.witnesses,
        &old_block,
        &parent,
    )
    .expect("emit shared projection and CE allocation");
    assert!(builder.is_satisfied(), "honest shared terminal rows must satisfy");

    let projection = &builder.terminal_pending_projection_audits()[0];
    let ce = builder.terminal_ce_claim_audits();
    assert_eq!(projection.projection_child_witness_first_columns.len(), ce.len());
    assert_eq!(
        projection.projection_child_witness_first_columns, projection.ajtai_child_witness_first_columns,
        "projection and terminal CE/Ajtai join audit must name the same absolute bases",
    );
    for (child, (first, claim)) in projection
        .projection_child_witness_first_columns
        .iter()
        .zip(ce)
        .enumerate()
    {
        assert_eq!(
            Some(*first),
            claim.witness_cols.first().copied(),
            "child {child} projection and Ajtai/CE must share one FinalWitnessWires allocation",
        );
    }
}
