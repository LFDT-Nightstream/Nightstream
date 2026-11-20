use super::*;

fn scalar(i: usize) -> K {
    K::from_coeffs([F::from_usize(31 * i + 7), F::from_usize(19 * i + 11)])
}

fn bind(a: K, b: K, x: K) -> K {
    a + x * (b - a)
}

fn value(coefficients: &[K], x: K) -> K {
    coefficients.iter().rev().fold(K::ZERO, |v, c| v * x + *c)
}

#[test]
fn packed_rounds_and_handoff_match_direct_evaluation_with_partial_tiles() {
    let session = MetalSession::new().unwrap();
    let checked: Vec<[F; 4]> = (0..1027)
        .map(|i| std::array::from_fn(|j| F::from_usize((i * (j + 1) + j + 3) % 97)))
        .collect();
    let mut gates: Vec<[K; 4]> = checked.iter().map(|row| row.map(K::from)).collect();
    let width = 163 * 64;
    let (mut positive, mut negative) = (vec![0u64; width / 64], vec![0u64; width / 64]);
    let mut digits: Vec<_> = (0..width)
        .map(|i| match (i * 73 + i / 17) % 5 {
            0 => {
                positive[i / 64] |= 1 << (i % 64);
                K::ONE
            }
            1 => {
                negative[i / 64] |= 1 << (i % 64);
                -K::ONE
            }
            _ => K::ZERO,
        })
        .collect();
    let mut packed = Packed::new(&session, checked, &positive, &negative, width).unwrap();
    let mut point = Vec::new();
    for round in 0..=7 {
        let low = (0..8).map(scalar).collect::<Vec<_>>();
        let high = (0..digits.len().div_ceil(16))
            .map(|i| scalar(i + 13))
            .collect::<Vec<_>>();
        let weight = |i: usize| low[i % low.len()] * high[i / low.len()];
        let (gate, norm) = packed.evaluate(&point, (&low, &high)).unwrap();
        for sample in 0..8 {
            let x = scalar(sample + 101);
            let expected: K = gates
                .chunks(2)
                .enumerate()
                .map(|(i, pair)| {
                    let [a, b, c, s]: [K; 4] =
                        std::array::from_fn(|j| bind(pair[0][j], pair.get(1).map_or(K::ZERO, |right| right[j]), x));
                    weight(i) * (a * b - c + s.exp_u64(7))
                })
                .sum();
            assert_eq!(value(&gate, x), expected, "gate round {round}, sample {sample}");
            if sample < 4 {
                let expected: K = digits
                    .chunks(2)
                    .enumerate()
                    .map(|(i, pair)| {
                        let v = bind(pair[0], pair.get(1).copied().unwrap_or(K::ZERO), x);
                        weight(i) * (v * v * v - v)
                    })
                    .sum();
                assert_eq!(value(&norm, x), expected, "norm round {round}, sample {sample}");
            }
        }
        if round == 7 {
            let (columns, signed) = packed.take_tables(round).unwrap();
            let expected: Vec<_> = (0..4)
                .flat_map(|j| gates.iter().map(move |row| row[j]))
                .collect();
            assert_eq!(columns, expected);
            assert_eq!(signed, digits);
            break;
        }
        let coin = scalar(round + 41);
        gates = gates
            .chunks(2)
            .map(|pair| std::array::from_fn(|j| bind(pair[0][j], pair.get(1).map_or(K::ZERO, |right| right[j]), coin)))
            .collect();
        digits = digits
            .chunks(2)
            .map(|pair| bind(pair[0], pair.get(1).copied().unwrap_or(K::ZERO), coin))
            .collect();
        point.push(coin);
    }
}
