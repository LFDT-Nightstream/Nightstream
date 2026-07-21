use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_reductions::block_projection::{
    project_raw_witness_at_block_point, project_raw_witnesses_at_block_point,
    radix_recompose_raw_witnesses_at_block_point, BLOCK_PROJECTION_DOMAIN_SIZE, BLOCK_PROJECTION_POINT_LEN,
};
use p3_field::PrimeCharacteristicRing;

fn block_selector(block: usize) -> [K; BLOCK_PROJECTION_POINT_LEN] {
    std::array::from_fn(|bit| if (block >> bit) & 1 == 0 { K::ZERO } else { K::ONE })
}

#[test]
fn raw_projection_indexes_packed_blocks_and_preserves_lane_ownership() {
    let expected_m = 2 * D;
    let mut witness = Mat::zero(D, 2, F::ZERO);
    witness.set(7, 0, F::from_u64(3));
    witness.set(7, 1, F::from_u64(11));
    witness.set(8, 1, F::from_u64(13));

    let projected = project_raw_witness_at_block_point(&witness, expected_m, &block_selector(1))
        .expect("well-shaped packed witness");

    assert_eq!(projected[7], K::from(F::from_u64(11)));
    assert_eq!(projected[8], K::from(F::from_u64(13)));
    assert!(projected
        .iter()
        .enumerate()
        .all(|(lane, value)| lane == 7 || lane == 8 || *value == K::ZERO));
}

#[test]
fn raw_projection_mutation_changes_only_the_owned_lane() {
    let expected_m = 2 * D;
    let point = block_selector(1);
    let mut honest = Mat::zero(D, 2, F::ZERO);
    honest.set(12, 1, F::from_u64(5));
    let mut mutated = honest.clone();
    mutated.set(12, 1, F::from_u64(6));

    let rows = project_raw_witnesses_at_block_point(&[honest, mutated], expected_m, &point)
        .expect("both witnesses have the same valid packed shape");

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[1][12] - rows[0][12], K::ONE);
    assert!(rows[0]
        .iter()
        .zip(rows[1].iter())
        .enumerate()
        .all(|(lane, (left, right))| lane == 12 || left == right));
}

#[test]
fn inactive_block_domain_is_computed_as_zero() {
    let expected_m = 2 * D;
    let mut witness = Mat::zero(D, 2, F::ZERO);
    witness.set(0, 0, F::ONE);
    witness.set(1, 1, F::ONE);

    let projected = project_raw_witness_at_block_point(&witness, expected_m, &block_selector(2))
        .expect("selector lies in the fixed domain even though its block is inactive");

    assert!(projected.iter().all(|value| *value == K::ZERO));
}

#[test]
fn raw_projection_rejects_wrong_witness_shape_and_oversized_domain() {
    let point = block_selector(0);
    let wrong_rows = Mat::zero(D - 1, 1, F::ZERO);
    assert!(project_raw_witness_at_block_point(&wrong_rows, D, &point).is_err());

    let wrong_columns = Mat::zero(D, 2, F::ZERO);
    assert!(project_raw_witness_at_block_point(&wrong_columns, D, &point).is_err());

    let tiny_witness = Mat::zero(D, 1, F::ZERO);
    let oversized_m = (BLOCK_PROJECTION_DOMAIN_SIZE + 1) * D;
    let error = project_raw_witness_at_block_point(&tiny_witness, oversized_m, &point)
        .expect_err("the fixed 19-bit domain must reject an oversized packed width");
    assert!(error.to_string().contains("exceeding the fixed"));
}

#[test]
fn radix_recomposition_binds_ordered_raw_children() {
    let expected_m = 2 * D;
    let point = block_selector(1);
    let mut child_zero = Mat::zero(D, 2, F::ZERO);
    child_zero.set(4, 1, F::from_u64(3));
    let mut child_one = Mat::zero(D, 2, F::ZERO);
    child_one.set(4, 1, F::from_u64(5));
    let radix = K::from(F::from_u64(2));

    let ordered = radix_recompose_raw_witnesses_at_block_point(
        &[child_zero.clone(), child_one.clone()],
        expected_m,
        &point,
        radix,
    )
    .expect("ordered children have valid packed shapes");
    let swapped = radix_recompose_raw_witnesses_at_block_point(&[child_one, child_zero], expected_m, &point, radix)
        .expect("swapped children still have valid packed shapes");

    assert_eq!(ordered[4], K::from(F::from_u64(13)));
    assert_eq!(swapped[4], K::from(F::from_u64(11)));
    assert_ne!(ordered, swapped);
}
