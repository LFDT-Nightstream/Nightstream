use super::*;
use crate::nightstream_fprime_setup::{
    commit_production_signed_unit_prefix_matrices, commit_production_signed_unit_prefix_matrix,
};

#[test]
fn dense_refresh_fits_while_sparse_and_over_budget_batches_use_four_planes() {
    let columns = 73_281;
    let layout = BatchLayout::new(9, columns);
    let tables = four_plane::TableLayout::new(9, columns).unwrap();
    assert!(layout.fits(585_835, 73_174, &tables));
    assert!(!layout.fits(9, 1, &tables));

    let columns = 1 << 21;
    let layout = BatchLayout::new(9, columns);
    assert!(!layout.fits(9 * columns, columns, &four_plane::TableLayout::new(9, columns).unwrap()));
    assert!(four_plane::TableLayout::new(9, columns + 1).is_none());
}

#[test]
fn packed_batch_preserves_child_slots_and_falls_back_for_sparse_or_invalid_inputs() {
    let columns = 40_003;
    let _key = ExpandedKey::production(columns as u64);
    let zero = Mat::virtual_constant(D, columns, Fq::ZERO);
    let packed = |pos, neg| {
        Mat::compact_signed_unit_from_column_masks(D, columns, &vec![pos; columns], &vec![neg; columns]).unwrap()
    };
    let witnesses = [zero, packed(1, 0), packed(0, 2), packed(0, 0), packed(1 << 63, 1)];
    let actual = commit(&witnesses).expect("dense batch fits");
    for (witness, commitment) in witnesses.iter().zip(actual) {
        assert_eq!(
            commitment,
            commit_production_signed_unit_prefix_matrix(witness).unwrap()
        );
    }

    let mut positive = vec![0; columns];
    positive[columns - 1] = 1;
    let sparse = Mat::compact_signed_unit_from_column_masks(D, columns, &positive, &vec![0; columns]).unwrap();
    assert!(commit(&[&sparse; 3]).is_none());
    let batch = commit_production_signed_unit_prefix_matrices(&[&sparse; 3]).unwrap();
    assert_eq!(
        batch,
        vec![commit_production_signed_unit_prefix_matrix(&sparse).unwrap(); 3]
    );

    let invalid = Mat::virtual_constant(D, columns, Fq::from_u64(2));
    let error = commit_production_signed_unit_prefix_matrices(&[&witnesses[1], &witnesses[2], &witnesses[4], &invalid])
        .unwrap_err();
    assert_eq!(error.witness_index(), 3);
}

#[test]
fn toom4_concurrent_panels_fit_the_batch_scratch_budget() {
    for planes in 3..=four_plane::MAX_PLANES {
        let groups = planes.div_ceil(MAX_BATCH_PLANES);
        let width = planes.div_ceil(groups);
        let budget = ROWS
            * (0..planes)
                .step_by(width)
                .map(|first| child_batch_workspace_bytes(width.min(planes - first)))
                .sum::<usize>();
        let jobs = ROWS * groups;
        let per_task = ChildBatchKernel::Toom4.jobs_per_task(jobs, width, budget);
        let digit_lanes = width * 16;
        let panels = 7 * (128 * (32 + digit_lanes) + 32 * digit_lanes) * size_of::<f64>();
        assert!(jobs.div_ceil(per_task) * (panels + (64 << 10)) <= budget);
    }
}
