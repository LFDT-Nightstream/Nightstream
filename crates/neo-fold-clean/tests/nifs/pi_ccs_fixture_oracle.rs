//! Check cached scalar completion sums against direct finite MLE sums.
//! These small arithmetic cases are not positive Stage 1 fixtures.

use neo_ccs::{SparsePoly, Term};
use neo_math::{from_complex, superneo_bar_block, Rq, D, F, K};
use neo_reductions::engines::pi_ccs_joint_protocol::PaperJointRoundOracle;
use p3_field::PrimeCharacteristicRing;

#[path = "../../src/bin/generate_pi_ccs_fixture/oracle.rs"]
mod oracle;

#[allow(dead_code)]
#[path = "../../src/bin/generate_pi_ccs_fixture/ring_output.rs"]
mod ring_output;

#[allow(dead_code)]
#[path = "../../src/bin/generate_pi_ccs_fixture/running_prefix.rs"]
mod running_prefix;

#[allow(dead_code)]
#[path = "../../src/bin/generate_pi_ccs_fixture/folded_opening.rs"]
mod folded_opening;

use oracle::{Oracle, LIVE_MATRICES, MATRICES, ROUNDS};

fn scalar(value: u64) -> K {
    K::from(F::from_u64(value))
}

fn partial(values: &[K], fixed: &[K], tail: usize) -> K {
    values
        .iter()
        .enumerate()
        .filter(|(index, _)| *index >> fixed.len() == tail)
        .map(|(index, &value)| {
            fixed
                .iter()
                .enumerate()
                .fold(value, |value, (bit, &coordinate)| {
                    value
                        * if index & (1 << bit) == 0 {
                            K::ONE - coordinate
                        } else {
                            coordinate
                        }
                })
        })
        .sum()
}

struct RunningCase {
    values: Vec<K>,
    point: Vec<K>,
    assignments: [Vec<K>; 16],
}

fn equality_at(point: &[K], fixed: &[K], tail: usize) -> K {
    point
        .iter()
        .enumerate()
        .fold(K::ONE, |product, (bit, &target)| {
            let coordinate = if bit < fixed.len() {
                fixed[bit]
            } else if tail & (1 << (bit - fixed.len())) == 0 {
                K::ZERO
            } else {
                K::ONE
            };
            product * ((K::ONE - coordinate) * (K::ONE - target) + coordinate * target)
        })
}

fn direct_completion(
    values: &[K],
    images: &[[K; LIVE_MATRICES]],
    polynomial: &SparsePoly<F>,
    alpha: &[K],
    gamma: K,
    fixed: &[K],
    running: Option<&RunningCase>,
) -> K {
    let support = values
        .len()
        .max(images.len())
        .max(running.map_or(0, |r| r.values.len()))
        .max(running.map_or(0, |r| r.assignments.iter().map(Vec::len).max().unwrap_or(0)))
        .div_ceil(1 << fixed.len());
    let mut sum = K::ZERO;
    let mut evaluation_sum = K::ZERO;
    for tail in 0..support {
        let value = partial(values, fixed, tail);
        let matrix_values = std::array::from_fn::<_, MATRICES, _>(|matrix| {
            if matrix == LIVE_MATRICES {
                K::ZERO
            } else {
                partial(&images.iter().map(|row| row[matrix]).collect::<Vec<_>>(), fixed, tail)
            }
        });
        let mut norm = (value + K::ONE) * value * (value - K::ONE);
        if let Some(running) = running {
            let mut power = gamma;
            for assignment in &running.assignments {
                let source_value = partial(assignment, fixed, tail);
                norm += power * (source_value + K::ONE) * source_value * (source_value - K::ONE);
                power *= gamma;
            }
            evaluation_sum += equality_at(&running.point, fixed, tail) * partial(&running.values, fixed, tail);
        }
        sum += equality_at(alpha, fixed, tail) * (polynomial.eval_in_ext(&matrix_values) + gamma * norm);
    }
    let shift = (0..16 * 54 * (MATRICES + 1)).fold(K::ONE, |power, _| power * gamma);
    evaluation_sum + shift * sum
}

fn polynomial() -> SparsePoly<F> {
    let mut eighth = vec![0; MATRICES];
    eighth[0] = 8;
    let mut mixed = vec![0; MATRICES];
    mixed[1] = 1;
    mixed[2] = 2;
    mixed[6] = 3;
    let mut zero_port = vec![0; MATRICES];
    zero_port[LIVE_MATRICES] = 1;
    SparsePoly::new(
        MATRICES,
        vec![
            Term {
                coeff: F::ONE,
                exps: eighth,
            },
            Term {
                coeff: F::from_u64(7),
                exps: mixed,
            },
            Term {
                coeff: -F::ONE,
                exps: zero_port,
            },
        ],
    )
}

fn check_round_cache(with_running: bool) {
    // Singleton, odd, and even prefixes exercise all zero-completion paths.
    for length in [1, 5, 6] {
        let values = (0..length)
            .map(|index| match index % 3 {
                0 => -K::ONE,
                1 => K::ZERO,
                _ => K::ONE,
            })
            .collect::<Vec<_>>();
        let images = (0..length)
            .map(|row| std::array::from_fn(|matrix| scalar(((row + 1) * (matrix + 2)) as u64)))
            .collect::<Vec<_>>();
        let alpha = (0..ROUNDS)
            .map(|index| from_complex(F::from_u64(index as u64 + 11), F::from_u64(index as u64 + 31)))
            .collect::<Vec<_>>();
        let gamma = from_complex(F::from_u64(17), F::from_u64(23));
        let polynomial = polynomial();
        let mut oracle = Oracle::new(values.clone(), images.clone(), &polynomial, alpha.clone(), gamma);
        let running = with_running.then(|| RunningCase {
            values: (0..=length)
                .map(|index| from_complex(F::from_u64(index as u64 + 3), F::from_u64(index as u64 + 7)))
                .collect(),
            point: (0..ROUNDS)
                .map(|index| from_complex(F::from_u64(index as u64 + 5), F::from_u64(index as u64 + 17)))
                .collect(),
            assignments: std::array::from_fn(|source| {
                values
                    .iter()
                    .map(|&value| if source % 2 == 0 { value } else { -value })
                    .collect()
            }),
        });
        if let Some(running) = &running {
            let norm_weight = K::ONE
                + (0..16)
                    .map(|source| {
                        let sign = if source % 2 == 0 { K::ONE } else { -K::ONE };
                        sign * (0..=source).fold(K::ONE, |power, _| power * gamma)
                    })
                    .sum::<K>();
            oracle = oracle.with_running(running.values.clone(), running.point.clone(), norm_weight);
        }
        let points = [K::ZERO, K::ONE, from_complex(F::from_u64(2), F::from_u64(3))];
        let mut fixed = Vec::new();
        for round in 0..ROUNDS {
            let actual = oracle.evals_at(&points).expect("cached scalar sums");
            for (index, &point) in points.iter().enumerate() {
                let mut prefix = fixed.clone();
                prefix.push(point);
                assert_eq!(
                    actual[index],
                    direct_completion(&values, &images, &polynomial, &alpha, gamma, &prefix, running.as_ref()),
                    "prefix length {length}, round {round}, trial {index}"
                );
            }
            let challenge = match round {
                0 => K::ZERO,
                1 => K::ONE,
                _ => from_complex(F::from_u64(round as u64 + 3), F::from_u64(round as u64 + 7)),
            };
            oracle.fold(challenge).expect("one exact prefix fold");
            fixed.push(challenge);
        }
        let (value, matrix_values) = oracle.scalar_outputs();
        assert_eq!(value, partial(&values, &fixed, 0));
        for matrix in 0..LIVE_MATRICES {
            assert_eq!(
                matrix_values[matrix],
                partial(&images.iter().map(|row| row[matrix]).collect::<Vec<_>>(), &fixed, 0)
            );
        }
        assert_eq!(matrix_values[LIVE_MATRICES], K::ZERO);
        assert_eq!(
            oracle.terminal(),
            direct_completion(&values, &images, &polynomial, &alpha, gamma, &fixed, running.as_ref())
        );
    }
}

#[test]
fn scalar_round_cache_matches_direct_mle_through_all_twenty_eight_folds() {
    check_round_cache(false);
}

#[test]
fn signed_running_terms_match_direct_mle_through_all_twenty_eight_folds() {
    check_round_cache(true);
}

#[test]
fn distinct_running_norms_match_direct_mle_through_all_twenty_eight_folds() {
    let values = vec![K::ONE, -K::ONE, K::ZERO, K::ONE, -K::ONE];
    let images = (0..values.len())
        .map(|row| std::array::from_fn(|matrix| scalar(((row + 1) * (matrix + 2)) as u64)))
        .collect::<Vec<_>>();
    let sources: [Vec<u8>; 16] = std::array::from_fn(|source| {
        // A different basis coordinate for each running source. The last
        // source extends past both the fresh and linear-evaluation prefixes.
        let mut values = vec![0; 17];
        values[source + 1] = if source % 2 == 0 { 1 } else { 255 };
        values
    });
    let running = RunningCase {
        values: (0..7)
            .map(|index| from_complex(F::from_u64(index + 3), F::from_u64(index + 7)))
            .collect(),
        point: (0..ROUNDS)
            .map(|index| from_complex(F::from_u64(index as u64 + 5), F::from_u64(index as u64 + 17)))
            .collect(),
        assignments: sources.clone().map(|values| {
            values
                .into_iter()
                .map(|value| match value {
                    0 => K::ZERO,
                    1 => K::ONE,
                    255 => -K::ONE,
                    _ => unreachable!(),
                })
                .collect()
        }),
    };
    let alpha = (0..ROUNDS)
        .map(|index| from_complex(F::from_u64(index as u64 + 11), F::from_u64(index as u64 + 31)))
        .collect::<Vec<_>>();
    let gamma = from_complex(F::from_u64(17), F::from_u64(23));
    let polynomial = polynomial();
    let mut oracle = Oracle::new(values.clone(), images.clone(), &polynomial, alpha.clone(), gamma)
        .with_distinct_running(running.values.clone(), running.point.clone(), sources);
    let mut points = (0..=oracle.degree_bound())
        .map(|index| scalar(index as u64))
        .collect::<Vec<_>>();
    points.push(from_complex(F::from_u64(2), F::from_u64(3)));
    let mut fixed = Vec::new();
    for round in 0..ROUNDS {
        let actual = oracle
            .evals_at(&points)
            .expect("distinct-source completion sums");
        for (&point, actual) in points.iter().zip(actual) {
            let mut prefix = fixed.clone();
            prefix.push(point);
            assert_eq!(
                actual,
                direct_completion(&values, &images, &polynomial, &alpha, gamma, &prefix, Some(&running)),
                "distinct sources, round {round}"
            );
        }
        let challenge = from_complex(F::from_u64(round as u64 + 3), F::from_u64(round as u64 + 7));
        oracle
            .fold(challenge)
            .expect("all source assignments fold at the same challenge");
        fixed.push(challenge);
    }
    assert_eq!(
        oracle.terminal(),
        direct_completion(&values, &images, &polynomial, &alpha, gamma, &fixed, Some(&running))
    );
}

#[test]
#[should_panic(expected = "running opening is not a signed unit")]
fn distinct_running_source_rejects_an_out_of_range_coefficient() {
    let mut sources: [Vec<u8>; 16] = std::array::from_fn(|_| Vec::new());
    sources[15] = vec![2];
    let _ = Oracle::new(
        vec![K::ONE],
        vec![[K::ZERO; LIVE_MATRICES]],
        &polynomial(),
        vec![K::ZERO; ROUNDS],
        K::ONE,
    )
    .with_distinct_running(vec![K::ZERO], vec![K::ZERO; ROUNDS], sources);
}

#[test]
fn full_ring_kernel_matches_weighted_basis_products_in_every_coefficient() {
    let kernel = ring_output::RingKernel::new();
    let weights: [K; D] =
        std::array::from_fn(|lane| from_complex(F::from_u64(lane as u64 + 2), F::from_u64(lane as u64 + 5)));
    let sources: [[u8; D]; 3] = [
        std::array::from_fn(|lane| if lane == 0 { 1 } else { 0 }),
        std::array::from_fn(|lane| if lane == D - 1 { 255 } else { 0 }),
        std::array::from_fn(|lane| match lane % 3 {
            0 => 0,
            1 => 1,
            _ => 255,
        }),
    ];
    for source in sources {
        let actual = kernel.apply(&weights, &source);
        let mut expected = [K::ZERO; D];
        for (column, &weight) in weights.iter().enumerate() {
            let mut basis = [F::ZERO; D];
            basis[column] = F::ONE;
            let transformed = Rq(superneo_bar_block(basis));
            let mut product = Rq::zero();
            for (power, value) in source.iter().enumerate() {
                match value {
                    0 => {}
                    1 => product = product.add(&transformed.mul_by_monomial(power)),
                    255 => product = product.sub(&transformed.mul_by_monomial(power)),
                    _ => unreachable!("bounded test source"),
                }
            }
            for coefficient in 0..D {
                expected[coefficient] += weight * K::from(product.0[coefficient]);
            }
        }
        assert_eq!(actual, expected, "all 54 coefficients, including cyclotomic reduction");
        let scalar: K = weights
            .iter()
            .zip(source)
            .map(|(&weight, value)| match value {
                0 => K::ZERO,
                1 => weight,
                255 => -weight,
                _ => unreachable!("bounded test source"),
            })
            .sum();
        assert_eq!(actual[0], scalar, "constant coefficient is the scalar inner product");
    }
}

#[test]
fn running_kernels_match_full_ring_coefficient_sums_at_every_native_lane() {
    let gamma = from_complex(F::from_u64(17), F::from_u64(23));
    let kernel = running_prefix::RunningKernel::new(gamma);
    let full = ring_output::RingKernel::new();
    let pad_weights: [K; D] =
        std::array::from_fn(|coefficient| (0..16 * coefficient).fold(K::ONE, |weight, _| weight * gamma));
    let matrix_weights: [K; D] =
        std::array::from_fn(|coefficient| (0..16 * MATRICES * coefficient).fold(K::ONE, |weight, _| weight * gamma));
    let sources: [[u8; D]; 3] = [
        std::array::from_fn(|lane| if lane == 0 { 1 } else { 0 }),
        std::array::from_fn(|lane| if lane == D - 1 { 255 } else { 0 }),
        std::array::from_fn(|lane| match lane % 3 {
            0 => 0,
            1 => 1,
            _ => 255,
        }),
    ];
    for source in sources {
        let (pad, matrix) = kernel.apply(&source);
        for lane in 0..D {
            let mut native = [K::ZERO; D];
            native[lane] = K::ONE;
            let coefficients = full.apply(&native, &source);
            let expected_pad: K = coefficients
                .iter()
                .zip(pad_weights)
                .map(|(&value, weight)| value * weight)
                .sum();
            let expected_matrix: K = coefficients
                .iter()
                .zip(matrix_weights)
                .map(|(&value, weight)| value * weight)
                .sum();
            assert_eq!(
                pad[lane], expected_pad,
                "all I_K coefficient powers at native lane {lane}"
            );
            assert_eq!(
                matrix[lane], expected_matrix,
                "all I_A coefficient powers at native lane {lane}"
            );
        }
    }
}

#[test]
fn combined_child_kernel_equals_the_indexed_sum_of_separate_kernels() {
    let gamma = from_complex(F::from_u64(17), F::from_u64(23));
    let kernel = running_prefix::RunningKernel::new(gamma);
    let sources: [[u8; D]; 16] =
        std::array::from_fn(|source| std::array::from_fn(|lane| [0, 1, 255][(lane + source) % 3]));
    let mut combined = [K::ZERO; D];
    let mut expected_pad = [K::ZERO; D];
    let mut expected_matrix = [K::ZERO; D];
    let mut weight = K::ONE;
    for source in sources {
        for lane in 0..D {
            combined[lane] += weight
                * match source[lane] {
                    0 => K::ZERO,
                    1 => K::ONE,
                    255 => -K::ONE,
                    _ => unreachable!(),
                };
        }
        let (pad, matrix) = kernel.apply(&source);
        for lane in 0..D {
            expected_pad[lane] += weight * pad[lane];
            expected_matrix[lane] += weight * matrix[lane];
        }
        weight *= gamma;
    }
    let (pad, matrix) = kernel.apply_combination(&combined);
    assert_eq!(pad, expected_pad, "all indexed I_K powers");
    assert_eq!(matrix, expected_matrix, "all indexed I_A powers");
}

#[test]
fn packed_integer_fold_matches_the_field_ring_on_every_basis_pair() {
    let lift = |value: i16| {
        let magnitude = F::from_u64(u64::from(value.unsigned_abs()));
        if value < 0 {
            -magnitude
        } else {
            magnitude
        }
    };
    for rho_column in 0..D {
        for coefficient in -2i8..=2 {
            let mut rho = [0i8; D];
            rho[rho_column] = coefficient;
            let kernel = folded_opening::FoldKernel::new(rho);
            let ring = Rq(rho.map(|value| lift(i16::from(value))));
            for source_column in 0..D {
                let shifted = ring.mul_by_monomial(source_column);
                for sign in [-1i8, 0, 1] {
                    let mut source = [0u8; D];
                    source[source_column] = sign as u8;
                    let actual = kernel.apply(&source);
                    let expected = shifted.0.map(|value| value * lift(i16::from(sign)));
                    assert_eq!(actual.map(lift), expected);
                    assert!(actual
                        .iter()
                        .all(|value| value.abs() <= kernel.norm_bound()));
                }
            }
        }
    }
    let rho = std::array::from_fn(|lane| [-2, -1, 0, 1, 2][lane % 5]);
    let kernel = folded_opening::FoldKernel::new(rho);
    let ring = Rq(rho.map(|value| lift(i16::from(value))));
    for source in [[1u8; D], [255u8; D], std::array::from_fn(|lane| [0, 1, 255][lane % 3])] {
        let mut expected = Rq::zero();
        for (power, &value) in source.iter().enumerate() {
            match value {
                0 => {}
                1 => expected = expected.add(&ring.mul_by_monomial(power)),
                255 => expected = expected.sub(&ring.mul_by_monomial(power)),
                _ => unreachable!(),
            }
        }
        let actual = kernel.apply(&source);
        assert_eq!(actual.map(lift), expected.0);
        assert!(actual
            .iter()
            .all(|value| value.abs() <= kernel.norm_bound()));
    }
}

#[test]
fn folded_opening_digits_cover_the_strict_profile_bound_without_fallback() {
    use folded_opening::{signed_digits, DIGITS, PARENT_BOUND};
    for value in -(PARENT_BOUND - 1)..PARENT_BOUND {
        let digits = signed_digits(value).expect("strictly bounded parent");
        let mut remaining = value;
        for digit in digits {
            assert_eq!(i32::from(digit), remaining % 2, "canonical signed bit");
            remaining = (remaining - i32::from(digit)) / 2;
        }
        assert_eq!(remaining, 0);
        assert_eq!(
            (0..DIGITS)
                .map(|bit| (1i32 << bit) * i32::from(digits[bit]))
                .sum::<i32>(),
            value
        );
    }
    for value in [-PARENT_BOUND, PARENT_BOUND, i32::MIN, i32::MAX] {
        assert!(signed_digits(value).is_none(), "out-of-range input must fail");
    }
}
