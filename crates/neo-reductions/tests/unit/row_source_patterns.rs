use super::*;

#[test]
fn repeated_patterns_share_storage_but_hash_collisions_do_not() {
    let mut builder = SuperneoEvalCacheBuilder::new(3, 2 * D, 1).unwrap();
    let original = [F::from_u64(3), F::from_u64(5)];
    let changed = [F::from_u64(7), F::from_u64(11)];
    builder
        .push_row(0, 0, [(0, original[0]), (1, original[1])])
        .unwrap();
    builder
        .push_row(0, 1, [(D, original[0]), (D + 1, original[1])])
        .unwrap();
    // Force the changed pattern to find a different pattern in its hash bucket.
    builder.patterns[0].insert(pattern_hash(&[0, 1], &changed), vec![0]);
    builder
        .push_row(0, 2, [(0, changed[0]), (1, changed[1])])
        .unwrap();
    let cache = builder.finish().unwrap();
    let matrix = &cache.matrix_caches()[0];
    let parts = matrix.compact_device_parts().unwrap();
    assert_eq!(
        parts.dense_offsets.len(),
        3,
        "two distinct patterns, not three row copies"
    );
    assert_eq!(
        parts.dense_coefficients,
        &[original[0], original[1], changed[0], changed[1]]
    );
    for (row, expected) in [
        vec![(0, original[0]), (1, original[1])],
        vec![(D, original[0]), (D + 1, original[1])],
        vec![(0, changed[0]), (1, changed[1])],
    ]
    .into_iter()
    .enumerate()
    {
        let mut actual = Vec::new();
        matrix.for_each_compact_explicit_row_coefficient(row, |block, local, coefficient| {
            actual.push((block as usize * D + local as usize, coefficient));
        });
        assert_eq!(actual, expected);
    }
}
