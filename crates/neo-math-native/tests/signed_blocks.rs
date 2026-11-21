use neo_math_native::signed_ring_sum;

#[test]
fn dense_tiles_match_negacyclic_signed_products_at_the_boundaries() {
    let mut state = 0x7f76_ed92_a18b_c5d3u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let key: Vec<[u64; 64]> = (0..405)
        .map(|i| {
            std::array::from_fn(|j| match (i + j) % 7 {
                0 => u64::MAX,
                1 => 0,
                _ => next(),
            })
        })
        .collect();
    let masks: Vec<_> = (0..4096)
        .map(|i| [(i % key.len()) as u64, next(), next()])
        .collect();
    if signed_ring_sum(&key, &masks[..1], |b| *b).is_none() {
        return;
    }
    for n in [1, 3, 4, 5, 191, 192, 193, 319, 320, 321, 383, 384, 405, 4095, 4096] {
        let actual = signed_ring_sum(&key, &masks[..n], |b| *b).expect("supported tile");
        let mut expected = [0; 128];
        for &[column, plus, minus] in &masks[..n] {
            for lane in 0..64 {
                let digit = ((plus >> lane) & 1) as i64 - ((minus >> lane) & 1) as i64;
                for (k, &word) in key[column as usize].iter().enumerate() {
                    let degree = lane + k;
                    let sign = if degree < 64 { digit } else { -digit };
                    expected[2 * (degree % 64)] += sign * i64::from(word as u32);
                    expected[2 * (degree % 64) + 1] += sign * (word >> 32) as i64;
                }
            }
        }
        assert_eq!(actual, expected);
    }
    let mut invalid = masks[..321].to_vec();
    invalid[320][0] = u64::MAX;
    assert!(signed_ring_sum(&key, &invalid, |b| *b).is_none());
    assert!(signed_ring_sum(&key, &[[0; 3]; 4097], |b| *b).is_none());
}
