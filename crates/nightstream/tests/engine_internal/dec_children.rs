use super::*;
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrices;
use neo_ccs::Mat;

#[test]
fn lowest_child_commitment_follows_from_the_parent_and_higher_children() {
    // Distinct signed-unit children over three columns; the parent commitment
    // is the recomposition that the Pi_DEC check compares.
    let lanes = (1u64 << D) - 1;
    let children: Vec<Mat<F>> = (0..4u32)
        .map(|child| {
            let bits = |column: u32| 0x9e37_79b9_7f4a_7c15u64.rotate_left(7 * child + 3 * column) & lanes;
            let positive: Vec<u64> = (0..3)
                .map(|column| bits(column) & 0x5555_5555_5555_5555)
                .collect();
            let negative: Vec<u64> = (0..3)
                .map(|column| bits(column) & !0x5555_5555_5555_5555)
                .collect();
            Mat::compact_signed_unit_from_column_masks(D, 3, &positive, &negative).unwrap()
        })
        .collect();
    let commitments = commit_production_signed_unit_prefix_matrices(&children).unwrap();
    let parent = ajtai_dec_mixer(&commitments, 2);
    assert_ne!(commitments[0], commitments[1]);
    assert_eq!(lowest_child_commitment(&parent, &commitments[1..], 2), commitments[0]);
}
