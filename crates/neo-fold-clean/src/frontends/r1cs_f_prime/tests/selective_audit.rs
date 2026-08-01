use super::row_family_masks;

#[test]
fn row_family_mask_sweep_matches_exact_range_membership() {
    let families = vec![
        ("outer", vec![(1, 7), (3, 5), (9, 9)]),
        ("edges", vec![(0, 2), (7, 10)]),
        ("adjacent", vec![(2, 4), (4, 6)]),
    ];

    let masks = row_family_masks(10, &families);
    let expected = (0..10)
        .map(|row| {
            families
                .iter()
                .enumerate()
                .fold(0u64, |mask, (family_index, (_, ranges))| {
                    if ranges
                        .iter()
                        .any(|&(start, end)| (start..end).contains(&row))
                    {
                        mask | (1 << family_index)
                    } else {
                        mask
                    }
                })
        })
        .collect::<Vec<_>>();

    assert_eq!(masks, expected);
}

#[test]
#[should_panic(expected = "row-family range exceeds the relation")]
fn row_family_mask_sweep_rejects_out_of_bounds_ranges() {
    row_family_masks(3, &[("bad", vec![(0, 4)])]);
}
