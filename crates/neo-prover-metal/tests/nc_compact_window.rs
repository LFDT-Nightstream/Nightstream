//! Algebra and layout checks for the mask-native compact NC kernel.

use neo_ccs::Mat;
use neo_math::{from_complex, D, F, K};
use p3_field::PrimeCharacteristicRing;

fn value(seed: usize) -> K {
    if seed.is_multiple_of(7) {
        return K::ZERO;
    }
    from_complex(F::from_u64((17 * seed + 3) as u64), F::from_u64((29 * seed + 5) as u64))
}

fn accumulate_constraint(inner: &mut [K; 4], weight: K, a: K, b: K) {
    let three = K::from(F::from_u64(3));
    let a2 = a * a;
    let b2 = b * b;
    inner[0] += weight * (a2 * a - a);
    inner[1] += weight * ((a2 * b) * three - b);
    inner[2] += weight * ((a * b2) * three);
    inner[3] += weight * (b2 * b);
}

fn accumulate_factored_constraint(inner: &mut [K; 4], weight: K, a: K, b: K) {
    let weighted_a = weight * a;
    let weighted_b = weight * b;
    let a2 = a * a;
    let b2 = b * b;
    inner[0] += weighted_a * (a2 - K::ONE);
    inner[1] += weighted_b * (a2 + a2 + a2 - K::ONE);
    inner[2] += weighted_a * (b2 + b2 + b2);
    inner[3] += weighted_b * b2;
}

fn accumulate_low_window_constraint(inner: &mut [K; 4], weight: K, a: K) {
    let weighted_a = weight * a;
    let weighted_a3 = weighted_a * (a * a);
    let three_weighted_a3 = weighted_a3 + weighted_a3 + weighted_a3;
    inner[0] += weighted_a3 - weighted_a;
    inner[1] += weighted_a - three_weighted_a3;
    inner[2] += three_weighted_a3;
    inner[3] -= weighted_a3;
}

fn accumulate_high_window_constraint(inner: &mut [K; 4], weight: K, b: K) {
    let weighted_b = weight * b;
    let weighted_b3 = weighted_b * (b * b);
    inner[1] -= weighted_b;
    inner[3] += weighted_b3;
}

fn accumulate_signed_low_constraint(inner: &mut [K; 4], signed_weight: K) {
    inner[1] -= signed_weight + signed_weight;
    inner[2] += signed_weight + signed_weight + signed_weight;
    inner[3] -= signed_weight;
}

fn accumulate_signed_high_constraint(inner: &mut [K; 4], signed_weight: K) {
    inner[1] -= signed_weight;
    inner[3] += signed_weight;
}

#[test]
fn factored_constraint_coefficients_match_the_canonical_cubic() {
    for seed in 1..64 {
        let weight = value(70_000 + 3 * seed);
        let a = value(70_001 + 3 * seed);
        let b = value(70_002 + 3 * seed);
        let mut canonical = [K::ZERO; 4];
        let mut factored = [K::ZERO; 4];
        accumulate_constraint(&mut canonical, weight, a, b);
        accumulate_factored_constraint(&mut factored, weight, a, b);
        assert_eq!(factored, canonical, "seed {seed}");
    }
}

#[test]
fn karatsuba_equality_product_matches_direct_convolution() {
    for seed in 1..64 {
        let cubic = [
            value(71_000 + 6 * seed),
            value(71_001 + 6 * seed),
            value(71_002 + 6 * seed),
            value(71_003 + 6 * seed),
        ];
        let eq_zero = value(71_004 + 6 * seed);
        let eq_one = value(71_005 + 6 * seed);
        let eq_slope = eq_one - eq_zero;
        let direct = [
            eq_zero * cubic[0],
            eq_zero * cubic[1] + eq_slope * cubic[0],
            eq_zero * cubic[2] + eq_slope * cubic[1],
            eq_zero * cubic[3] + eq_slope * cubic[2],
            eq_slope * cubic[3],
        ];

        let lo_zero = eq_zero * cubic[0];
        let lo_two = eq_slope * cubic[1];
        let lo_one = eq_one * (cubic[0] + cubic[1]) - lo_zero - lo_two;
        let hi_zero = eq_zero * cubic[2];
        let hi_two = eq_slope * cubic[3];
        let hi_one = eq_one * (cubic[2] + cubic[3]) - hi_zero - hi_two;
        let karatsuba = [lo_zero, lo_one, lo_two + hi_zero, hi_one, hi_two];

        assert_eq!(karatsuba, direct, "seed {seed}");
    }
}

fn compact_lane(values: &[K], rows: usize, width: usize, row: usize, lane: usize) -> K {
    debug_assert_eq!(values.len(), rows * width);
    let start = (row * width) % D;
    let slot = (lane + D - start) % D;
    if slot < width {
        values[row * width + slot]
    } else {
        K::ZERO
    }
}

fn dense_reference(tables: &[Vec<K>], weights: &[[K; D]], eq: &[K], rows: usize, width: usize) -> [K; 5] {
    let mut coefficients = [K::ZERO; 5];
    for pair in 0..rows / 2 {
        let lo_row = 2 * pair;
        let hi_row = lo_row + 1;
        let e0 = eq[lo_row];
        let e1 = eq[hi_row] - e0;
        let mut inner = [K::ZERO; 4];
        for (witness, table) in tables.iter().enumerate() {
            for lane in 0..D {
                let a = compact_lane(table, rows, width, lo_row, lane);
                let hi = compact_lane(table, rows, width, hi_row, lane);
                accumulate_constraint(&mut inner, weights[witness][lane], a, hi - a);
            }
        }
        coefficients[0] += e0 * inner[0];
        coefficients[1] += e0 * inner[1] + e1 * inner[0];
        coefficients[2] += e0 * inner[2] + e1 * inner[1];
        coefficients[3] += e0 * inner[3] + e1 * inner[2];
        coefficients[4] += e1 * inner[3];
    }
    coefficients
}

fn disjoint_window_candidate(tables: &[Vec<K>], weights: &[[K; D]], eq: &[K], rows: usize, width: usize) -> [K; 5] {
    assert!(2 * width <= D);
    let mut coefficients = [K::ZERO; 5];
    for pair in 0..rows / 2 {
        let lo_row = 2 * pair;
        let hi_row = lo_row + 1;
        let e0 = eq[lo_row];
        let e1 = eq[hi_row] - e0;
        let mut inner = [K::ZERO; 4];
        for (witness, table) in tables.iter().enumerate() {
            let lo_start = (lo_row * width) % D;
            let hi_start = (hi_row * width) % D;
            for slot in 0..width {
                let lane = (lo_start + slot) % D;
                let a = table[lo_row * width + slot];
                accumulate_low_window_constraint(&mut inner, weights[witness][lane], a);
            }
            for slot in 0..width {
                let lane = (hi_start + slot) % D;
                let hi = table[hi_row * width + slot];
                accumulate_high_window_constraint(&mut inner, weights[witness][lane], hi);
            }
        }
        coefficients[0] += e0 * inner[0];
        coefficients[1] += e0 * inner[1] + e1 * inner[0];
        coefficients[2] += e0 * inner[2] + e1 * inner[1];
        coefficients[3] += e0 * inner[3] + e1 * inner[2];
        coefficients[4] += e1 * inner[3];
    }
    coefficients
}

#[test]
fn disjoint_compact_windows_match_all_lane_evaluation() {
    let rows = 32;
    let witness_count = 3;
    let weights = (0..witness_count)
        .map(|witness| std::array::from_fn(|lane| value(10_000 + witness * D + lane)))
        .collect::<Vec<_>>();
    let eq = (0..rows).map(|row| value(20_000 + row)).collect::<Vec<_>>();

    for width in [1, 2, 4, 8, 16] {
        let tables = (0..witness_count)
            .map(|witness| {
                (0..rows * width)
                    .map(|entry| value(30_000 + witness * rows * width + entry))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            disjoint_window_candidate(&tables, &weights, &eq, rows, width),
            dense_reference(&tables, &weights, &eq, rows, width),
            "compact width {width}",
        );
    }
}

fn overlapping_window_candidate(tables: &[Vec<K>], weights: &[[K; D]], eq: &[K], rows: usize, width: usize) -> [K; 5] {
    assert!(width <= D && 2 * width > D);
    let mut coefficients = [K::ZERO; 5];
    for pair in 0..rows / 2 {
        let lo_row = 2 * pair;
        let hi_row = lo_row + 1;
        let e0 = eq[lo_row];
        let e1 = eq[hi_row] - e0;
        let lo_start = (lo_row * width) % D;
        let hi_start = (hi_row * width) % D;
        let mut inner = [K::ZERO; 4];
        for (witness, table) in tables.iter().enumerate() {
            for lane in 0..D {
                let lo_slot = (lane + D - lo_start) % D;
                let hi_slot = (lane + D - hi_start) % D;
                match (lo_slot < width, hi_slot < width) {
                    (true, true) => {
                        let a = table[lo_row * width + lo_slot];
                        let hi = table[hi_row * width + hi_slot];
                        accumulate_factored_constraint(&mut inner, weights[witness][lane], a, hi - a);
                    }
                    (true, false) => accumulate_low_window_constraint(
                        &mut inner,
                        weights[witness][lane],
                        table[lo_row * width + lo_slot],
                    ),
                    (false, true) => accumulate_high_window_constraint(
                        &mut inner,
                        weights[witness][lane],
                        table[hi_row * width + hi_slot],
                    ),
                    (false, false) => {}
                }
            }
        }
        coefficients[0] += e0 * inner[0];
        coefficients[1] += e0 * inner[1] + e1 * inner[0];
        coefficients[2] += e0 * inner[2] + e1 * inner[1];
        coefficients[3] += e0 * inner[3] + e1 * inner[2];
        coefficients[4] += e1 * inner[3];
    }
    coefficients
}

#[test]
fn overlapping_compact_windows_match_all_lane_evaluation() {
    let rows = 32;
    let width = 32;
    let witness_count = 3;
    let weights = (0..witness_count)
        .map(|witness| std::array::from_fn(|lane| value(40_000 + witness * D + lane)))
        .collect::<Vec<_>>();
    let eq = (0..rows).map(|row| value(50_000 + row)).collect::<Vec<_>>();
    let tables = (0..witness_count)
        .map(|witness| {
            (0..rows * width)
                .map(|entry| value(60_000 + witness * rows * width + entry))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert_eq!(
        overlapping_window_candidate(&tables, &weights, &eq, rows, width),
        dense_reference(&tables, &weights, &eq, rows, width),
    );
}

fn expand_masks(words: &[u64], witness_count: usize, blocks: usize, active_rows: usize, rows: usize) -> Vec<Vec<K>> {
    (0..witness_count)
        .map(|witness| {
            (0..rows)
                .map(|row| {
                    if row >= active_rows {
                        return K::ZERO;
                    }
                    let block = row / D;
                    let bit = 1u64 << (row % D);
                    let base = 2 * (witness * blocks + block);
                    if words[base] & bit != 0 {
                        K::ONE
                    } else if words[base + 1] & bit != 0 {
                        K::ZERO - K::ONE
                    } else {
                        K::ZERO
                    }
                })
                .collect()
        })
        .collect()
}

fn mask_digit(words: &[u64], witness: usize, blocks: usize, active_rows: usize, row: usize) -> K {
    if row >= active_rows {
        return K::ZERO;
    }
    let block = row / D;
    let bit = 1u64 << (row % D);
    let base = 2 * (witness * blocks + block);
    if words[base] & bit != 0 {
        K::ONE
    } else if words[base + 1] & bit != 0 {
        K::ZERO - K::ONE
    } else {
        K::ZERO
    }
}

fn cyclic_mask_lane(
    words: &[u64],
    blocks: usize,
    active_rows: usize,
    row: usize,
    width: usize,
    lane: usize,
    basis: &[K],
) -> K {
    let start = (row * width) % D;
    let first = (lane + D - start) % D;
    (first..width)
        .step_by(D)
        .map(|slot| mask_digit(words, 0, blocks, active_rows, row * width + slot) * basis[slot])
        .sum()
}

#[test]
fn delayed_mask_crossover_matches_direct_cyclic_accumulation() {
    let basis = (0..128)
        .map(|slot| value(80_000 + slot))
        .collect::<Vec<_>>();
    for active_rows in [1usize, 53, 54, 63, 64, 107, 108, 117, 128] {
        let blocks = active_rows.div_ceil(D);
        let mut words = vec![0u64; 2 * blocks];
        for row in 0..active_rows {
            let block = row / D;
            let bit = 1u64 << (row % D);
            if row.is_multiple_of(5) {
                words[2 * block] |= bit;
            } else if row.is_multiple_of(7) {
                words[2 * block + 1] |= bit;
            }
        }
        for (width, rows) in [(64, 2), (128, 1)] {
            let mut collision_counts = [0usize; 4];
            for row in 0..rows {
                for lane in 0..D {
                    let direct = (0..width)
                        .filter(|&slot| (row * width + slot) % D == lane)
                        .map(|slot| mask_digit(&words, 0, blocks, active_rows, row * width + slot) * basis[slot])
                        .sum::<K>();
                    assert_eq!(
                        cyclic_mask_lane(&words, blocks, active_rows, row, width, lane, &basis),
                        direct,
                        "active_rows={active_rows}, width={width}, row={row}, lane={lane}",
                    );
                    if row == 0 {
                        collision_counts[(0..width).filter(|&slot| slot % D == lane).count()] += 1;
                    }
                }
            }
            let expected = if width == 64 { [0, 44, 10, 0] } else { [0, 0, 34, 20] };
            assert_eq!(collision_counts, expected, "width={width}");
        }
    }
}

fn direct_mask_round(
    words: &[u64],
    witness_count: usize,
    blocks: usize,
    active_rows: usize,
    rows: usize,
    weights: &[[K; D]],
    eq: &[K],
) -> [K; 5] {
    let mut coefficients = [K::ZERO; 5];
    for pair in 0..rows / 2 {
        let lo_row = 2 * pair;
        let hi_row = lo_row + 1;
        let e0 = eq[lo_row];
        let e1 = eq[hi_row] - e0;
        let mut inner = [K::ZERO; 4];
        for witness in 0..witness_count {
            let lo = mask_digit(words, witness, blocks, active_rows, lo_row);
            if lo != K::ZERO {
                accumulate_signed_low_constraint(&mut inner, weights[witness][lo_row % D] * lo);
            }
            let hi = mask_digit(words, witness, blocks, active_rows, hi_row);
            if hi != K::ZERO {
                accumulate_signed_high_constraint(&mut inner, weights[witness][hi_row % D] * hi);
            }
        }
        coefficients[0] += e0 * inner[0];
        coefficients[1] += e0 * inner[1] + e1 * inner[0];
        coefficients[2] += e0 * inner[2] + e1 * inner[1];
        coefficients[3] += e0 * inner[3] + e1 * inner[2];
        coefficients[4] += e1 * inner[3];
    }
    coefficients
}

fn direct_mask_fold(
    words: &[u64],
    witness_count: usize,
    blocks: usize,
    active_rows: usize,
    rows: usize,
    challenge: K,
) -> Vec<Vec<K>> {
    (0..witness_count)
        .map(|witness| {
            (0..rows / 2)
                .flat_map(|pair| {
                    let lo = mask_digit(words, witness, blocks, active_rows, 2 * pair);
                    let hi = mask_digit(words, witness, blocks, active_rows, 2 * pair + 1);
                    [lo + challenge * (K::ZERO - lo), challenge * hi]
                })
                .collect()
        })
        .collect()
}

#[test]
fn signed_column_masks_expand_to_the_canonical_width_one_table() {
    let active_rows = 70usize;
    let rows = 128usize;
    let blocks = active_rows.div_ceil(D);
    let masks = [
        (vec![0b100101, 1 << 7], vec![0b010010, 1 << 15]),
        (vec![0b001010, 1 << 3], vec![0b100001, 1 << 11]),
    ];
    let matrices = masks
        .iter()
        .map(|(positive, negative)| {
            Mat::compact_signed_unit_from_column_masks(D, blocks, positive, negative).expect("valid signed masks")
        })
        .collect::<Vec<Mat<F>>>();
    let words = masks
        .iter()
        .flat_map(|(positive, negative)| {
            positive
                .iter()
                .copied()
                .zip(negative.iter().copied())
                .flat_map(|(positive, negative)| [positive, negative])
        })
        .collect::<Vec<_>>();

    let expanded = expand_masks(&words, matrices.len(), blocks, active_rows, rows);
    for (witness, matrix) in matrices.iter().enumerate() {
        for row in 0..rows {
            let expected = if row < active_rows {
                K::from(matrix[(row % D, row / D)])
            } else {
                K::ZERO
            };
            assert_eq!(expanded[witness][row], expected, "witness {witness}, row {row}");
        }
    }

    let weights = (0..matrices.len())
        .map(|witness| std::array::from_fn(|lane| value(40_000 + witness * D + lane)))
        .collect::<Vec<_>>();
    let eq = (0..rows).map(|row| value(50_000 + row)).collect::<Vec<_>>();
    assert_eq!(
        direct_mask_round(&words, matrices.len(), blocks, active_rows, rows, &weights, &eq,),
        dense_reference(&expanded, &weights, &eq, rows, 1),
    );

    let challenge = value(60_001);
    let canonical_fold = expanded
        .iter()
        .map(|table| {
            table
                .chunks_exact(2)
                .flat_map(|pair| [pair[0] + challenge * (K::ZERO - pair[0]), challenge * pair[1]])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        direct_mask_fold(&words, matrices.len(), blocks, active_rows, rows, challenge,),
        canonical_fold,
    );
}
