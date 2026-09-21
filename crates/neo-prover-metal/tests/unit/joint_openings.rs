use super::*;
use neo_ccs::Mat;
use neo_reductions::superneo_eval::{SuperneoEvalCacheBuilder, SuperneoZBlocks};
use std::sync::Arc;

#[test]
fn compact_openings_match_cpu_across_parallel_and_tiled_lists() {
    // Cross the actual device list thresholds. All matrices share columns,
    // so each matrix must overwrite the reused form storage correctly.
    let rows = FORM_TILE_ENTRIES + 1;
    let mut builder = SuperneoEvalCacheBuilder::new(rows, 2 * D, 3).unwrap();
    for row in 0..rows {
        builder
            .push_row(0, row, [(1, F::from_u64(3)), (2, F::from_u64(5)), (D + 1, -F::ONE)])
            .unwrap();
        builder
            .push_row(
                1,
                row,
                (row < PARALLEL_FORM_LIST_THRESHOLD)
                    .then_some([(1, -F::ONE), (D + 1, F::ONE)])
                    .into_iter()
                    .flatten(),
            )
            .unwrap();
        builder
            .push_row(
                2,
                row,
                (0..D)
                    .filter(|_| row < 2)
                    .map(|local| (D + local, F::from_u64((local + 2) as u64))),
            )
            .unwrap();
    }
    let cache = Arc::new(builder.finish().unwrap());
    let mut first = Mat::zero(D, 2, F::ZERO);
    first[(1, 0)] = F::ONE;
    first[(2, 0)] = -F::ONE;
    first[(1, 1)] = F::ONE;
    let mut last = first.clone();
    last[(D - 1, 1)] = -F::ONE;
    let witnesses = [first, Mat::zero(D, 2, F::ZERO), last];
    let variables = rows.next_power_of_two().ilog2() as usize;
    let point: Vec<_> = (0..variables)
        .map(|i| K::from_coeffs([F::from_u64((i + 2) as u64), F::from_u64((i + 3) as u64)]))
        .collect();
    let blocks: Vec<_> = witnesses
        .iter()
        .map(|witness| SuperneoZBlocks::from_witness_mat(witness, 2 * D).unwrap())
        .collect();
    let expected = cache.eval_real_v1_1_openings(&point, &blocks).unwrap();
    let session = MetalSession::new().unwrap();
    let plan = session.prepare_joint_matrix_plan(cache).unwrap();
    assert!(plan.opening.parallel_form_list_count > 0);
    assert!(plan.opening.tiled_form_tile_count > 0);
    let actual = session
        .eval_joint_dec_openings(&plan, &witnesses, &point, 2 * D)
        .unwrap()
        .unwrap();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.eval_k, expected.eval_k);
        assert_eq!(actual.eval_a, expected.eval_a);
    }
}
