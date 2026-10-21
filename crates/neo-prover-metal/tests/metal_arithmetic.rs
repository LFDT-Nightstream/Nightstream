#![cfg(all(feature = "metal", target_vendor = "apple", neo_metal_shaders))]

use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_math::{from_complex, KExtensions, Rq, D, F, K};
use neo_prover_metal::{GoldilocksMulVariant, KWords, MetalSession, PoseidonHashVariant};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::{Rng, SeedableRng};

const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;

fn k_from_words(words: KWords) -> K {
    from_complex(F::from_u64(words.c0), F::from_u64(words.c1))
}

fn words_from_k(value: K) -> KWords {
    let (c0, c1) = value.to_limbs_u64();
    KWords::new(c0, c1)
}

#[test]
fn metal_goldilocks_and_extension_arithmetic_match_cpu() {
    let session = MetalSession::new().expect("Metal session");
    let boundaries = [
        0,
        1,
        2,
        GOLDILOCKS_MODULUS - 2,
        GOLDILOCKS_MODULUS - 1,
        GOLDILOCKS_MODULUS,
        GOLDILOCKS_MODULUS + 1,
        u64::MAX - 1,
        u64::MAX,
    ];
    let mut lhs = Vec::new();
    let mut rhs = Vec::new();
    for &a in &boundaries {
        for &b in &boundaries {
            lhs.push(a);
            rhs.push(b);
        }
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x004d_4554_414c);
    for _ in 0..10_000 {
        lhs.push(rng.random());
        rhs.push(rng.random());
    }

    let limb_output = session
        .goldilocks_ops_variant(&lhs, &rhs, GoldilocksMulVariant::Limb32)
        .expect("limb Goldilocks Metal dispatch");
    let native_output = session
        .goldilocks_ops_variant(&lhs, &rhs, GoldilocksMulVariant::Native64)
        .expect("native Goldilocks Metal dispatch");
    assert_eq!(native_output, limb_output);
    for ((&lhs, &rhs), output) in lhs.iter().zip(&rhs).zip(limb_output) {
        let a = F::from_u64(lhs);
        let b = F::from_u64(rhs);
        assert_eq!(output.add, (a + b).as_canonical_u64());
        assert_eq!(output.sub, (a - b).as_canonical_u64());
        assert_eq!(output.mul, (a * b).as_canonical_u64());
    }

    let initial = (0..4_096)
        .map(|_| {
            KWords::new(
                rng.random::<u64>() % GOLDILOCKS_MODULUS,
                rng.random::<u64>() % GOLDILOCKS_MODULUS,
            )
        })
        .collect::<Vec<_>>();
    let multipliers = (0..initial.len())
        .map(|_| {
            KWords::new(
                rng.random::<u64>() % GOLDILOCKS_MODULUS,
                rng.random::<u64>() % GOLDILOCKS_MODULUS,
            )
        })
        .collect::<Vec<_>>();
    let rounds = 17;
    let (actual, stats) = session
        .kx_mul_add_chain(&initial, &multipliers, rounds)
        .expect("resident extension-field chain");
    assert_eq!(stats.dispatches, rounds + 1);
    let plan = session
        .prepare_kx_chain(&initial, &multipliers)
        .expect("prepare resident extension-field chain");
    let (reused, reused_stats) = session
        .kx_mul_add_chain_with_plan(&plan, rounds)
        .expect("reuse resident extension-field chain");
    assert_eq!(reused, actual);
    assert_eq!(reused_stats.dispatches, rounds + 1);
    for ((initial, multiplier), actual) in initial.iter().zip(&multipliers).zip(actual) {
        let mut expected = k_from_words(*initial);
        let multiplier = k_from_words(*multiplier);
        for _ in 0..rounds {
            expected = expected * multiplier + expected;
        }
        assert_eq!(actual, words_from_k(expected));
    }

    let states = (0..1_024)
        .map(|_| std::array::from_fn(|_| rng.random::<u64>()))
        .collect::<Vec<_>>();
    let actual = session
        .poseidon2_permute(&states)
        .expect("Poseidon2 Metal permutation");
    for (input, actual) in states.into_iter().zip(actual) {
        let expected = p2::permute_state(input.map(F::from_u64)).map(|value| value.as_canonical_u64());
        assert_eq!(actual, expected);
    }

    let inputs = [0, 1, 11, 12, 13, 24, 25, 64]
        .into_iter()
        .map(|len| (0..len).map(|_| rng.random::<u64>()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let scalar = session
        .poseidon2_hash_variant(&inputs, PoseidonHashVariant::Scalar)
        .expect("scalar Poseidon2 Metal hashes");
    let simd = session
        .poseidon2_hash_variant(&inputs, PoseidonHashVariant::SimdGroup)
        .expect("SIMD-group Poseidon2 Metal hashes");
    assert_eq!(simd, scalar);
    for (input, actual) in inputs.iter().zip(scalar) {
        let input = input.iter().copied().map(F::from_u64).collect::<Vec<_>>();
        let expected = p2::poseidon2_hash(&input).map(|value| value.as_canonical_u64());
        assert_eq!(actual, expected);
    }

    let uniform_fields = (0..1_024 * 8)
        .map(|_| rng.random::<u64>())
        .collect::<Vec<_>>();
    let uniform_plan = session
        .prepare_poseidon2_uniform(&uniform_fields, 8)
        .expect("prepare uniform Poseidon2 hashes");
    for variant in [PoseidonHashVariant::Scalar, PoseidonHashVariant::SimdGroup] {
        let actual = session
            .poseidon2_hash_uniform(&uniform_fields, 8, variant)
            .expect("uniform Poseidon2 Metal hashes");
        for (input, actual) in uniform_fields.chunks_exact(8).zip(actual) {
            let input = input.iter().copied().map(F::from_u64).collect::<Vec<_>>();
            let expected = p2::poseidon2_hash(&input).map(|value| value.as_canonical_u64());
            assert_eq!(actual, expected);
        }
        let reused = session
            .poseidon2_hash_uniform_with_plan(&uniform_plan, variant)
            .expect("reuse uniform Poseidon2 hashes");
        let direct = session
            .poseidon2_hash_uniform(&uniform_fields, 8, variant)
            .expect("direct uniform Poseidon2 hashes");
        assert_eq!(reused, direct);
    }

    // Merkle trees: leaves by the overwrite sponge (partial last blocks,
    // non-canonical words), nodes by the same sponge over both children.
    let sponge = |words: &[u64]| -> [u64; 4] {
        let mut state = [F::ZERO; p2::WIDTH];
        for block in words.chunks(p2::RATE) {
            for (lane, &word) in state.iter_mut().zip(block) {
                *lane = F::from_u64(word);
            }
            state = p2::permute_state(state);
        }
        std::array::from_fn(|lane| state[lane].as_canonical_u64())
    };
    for (rows, width) in [
        (256, 1),
        (256, 5),
        (256, 12),
        (256, 13),
        (512, 16),
        (128, 24),
        (256, 48),
        (1, 7),
    ] {
        let fields = (0..rows * width)
            .map(|_| rng.random::<u64>())
            .collect::<Vec<_>>();
        let layers = session
            .poseidon2_merkle_layers(rows, width, |words| words.copy_from_slice(&fields))
            .expect("Poseidon2 Metal Merkle tree");
        let mut expected: Vec<[u64; 4]> = fields.chunks_exact(width).map(sponge).collect();
        let leaves = session
            .poseidon2_merkle_leaves(&fields, width)
            .expect("Poseidon2 Metal Merkle leaves");
        assert_eq!(leaves, expected, "leaves of rows {rows} width {width}");
        for layer in &layers {
            assert_eq!(layer, &expected, "rows {rows} width {width}");
            expected = expected
                .chunks_exact(2)
                .map(|pair| sponge(&[pair[0], pair[1]].concat()))
                .collect();
        }
        assert_eq!(layers.last().map(Vec::len), Some(1), "rows {rows} width {width}");
    }
    // Leaves of rows on whole pages are read in place: only the width word
    // is uploaded. Rows one row off the page are copied.
    let page = std::alloc::Layout::from_size_align(1 << 16, 1 << 14).unwrap();
    let pointer = unsafe { std::alloc::alloc(page) }.cast::<u64>();
    assert!(!pointer.is_null());
    let words = unsafe { std::slice::from_raw_parts_mut(pointer, page.size() / 8) };
    words
        .iter_mut()
        .for_each(|word| *word = rng.random::<u64>());
    for (start, len, in_place) in [(0, words.len(), true), (16, words.len() / 2, false)] {
        let rows = &words[start..start + len];
        let uploaded = session.activity().uploaded_bytes;
        let leaves = session
            .poseidon2_merkle_leaves(rows, 16)
            .expect("Poseidon2 Metal Merkle leaves");
        let uploaded = session.activity().uploaded_bytes - uploaded;
        assert_eq!(leaves, rows.chunks_exact(16).map(sponge).collect::<Vec<_>>());
        assert_eq!(uploaded == 8, in_place, "uploaded {uploaded} bytes");
    }
    unsafe { std::alloc::dealloc(pointer.cast(), page) };

    let rows = 3;
    let cols = 5;
    let matrix = (0..rows * cols * D)
        .map(|_| rng.random::<u64>() % GOLDILOCKS_MODULUS)
        .collect::<Vec<_>>();
    let message = (0..cols * D)
        .map(|_| rng.random::<u64>() % GOLDILOCKS_MODULUS)
        .collect::<Vec<_>>();
    let actual = session
        .ajtai_mat_vec(&matrix, rows, cols, &message)
        .expect("Ajtai Metal ring mat-vec");
    let mut expected = Vec::with_capacity(rows * D);
    for row in 0..rows {
        let mut value = Rq::zero();
        for col in 0..cols {
            let matrix_start = (row * cols + col) * D;
            let message_start = col * D;
            let matrix_value = Rq::from_field_coeffs(
                &matrix[matrix_start..matrix_start + D]
                    .iter()
                    .copied()
                    .map(F::from_u64)
                    .collect::<Vec<_>>(),
            );
            let message_value = Rq::from_field_coeffs(
                &message[message_start..message_start + D]
                    .iter()
                    .copied()
                    .map(F::from_u64)
                    .collect::<Vec<_>>(),
            );
            value = value + matrix_value * message_value;
        }
        expected.extend(value.0.map(|coefficient| coefficient.as_canonical_u64()));
    }
    assert_eq!(actual, expected);

    let low_norm_message = (0..cols * D)
        .map(|_| rng.random_range(-1i8..=1))
        .collect::<Vec<_>>();
    let actual = session
        .ajtai_low_norm_mat_vec(&matrix, rows, cols, &low_norm_message)
        .expect("Ajtai low-norm Metal ring mat-vec");
    let low_norm_plan = session
        .prepare_ajtai_low_norm(&matrix, rows, cols)
        .expect("prepare Ajtai low-norm ring mat-vec");
    let reused = session
        .ajtai_low_norm_with_plan(&low_norm_plan, &low_norm_message)
        .expect("reuse Ajtai low-norm ring mat-vec");
    assert_eq!(reused, actual);
    let mut invalid_message = low_norm_message.clone();
    invalid_message[0] = 2;
    assert!(session
        .ajtai_low_norm_with_plan(&low_norm_plan, &invalid_message)
        .is_err());
    let mut expected = Vec::with_capacity(rows * D);
    for row in 0..rows {
        let mut value = Rq::zero();
        for col in 0..cols {
            let matrix_start = (row * cols + col) * D;
            let message_start = col * D;
            let matrix_value = Rq::from_field_coeffs(
                &matrix[matrix_start..matrix_start + D]
                    .iter()
                    .copied()
                    .map(F::from_u64)
                    .collect::<Vec<_>>(),
            );
            let message_value = Rq::from_field_coeffs(
                &low_norm_message[message_start..message_start + D]
                    .iter()
                    .map(|digit| match digit {
                        -1 => -F::ONE,
                        0 => F::ZERO,
                        1 => F::ONE,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>(),
            );
            value = value + matrix_value * message_value;
        }
        expected.extend(value.0.map(|coefficient| coefficient.as_canonical_u64()));
    }
    assert_eq!(actual, expected);

    let table = (0..8_192)
        .map(|_| {
            KWords::new(
                rng.random::<u64>() % GOLDILOCKS_MODULUS,
                rng.random::<u64>() % GOLDILOCKS_MODULUS,
            )
        })
        .collect::<Vec<_>>();
    let challenge = KWords::new(
        rng.random::<u64>() % GOLDILOCKS_MODULUS,
        rng.random::<u64>() % GOLDILOCKS_MODULUS,
    );
    let actual = session
        .fold_k_table(&table, challenge)
        .expect("extension-field table fold");
    let challenge = k_from_words(challenge);
    let expected = table
        .chunks_exact(2)
        .map(|pair| {
            let left = k_from_words(pair[0]);
            let right = k_from_words(pair[1]);
            words_from_k(left + challenge * (right - left))
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);

    let challenges = (0..table.len().ilog2())
        .map(|_| {
            KWords::new(
                rng.random::<u64>() % GOLDILOCKS_MODULUS,
                rng.random::<u64>() % GOLDILOCKS_MODULUS,
            )
        })
        .collect::<Vec<_>>();
    let (actual, stats) = session
        .fold_k_table_full(&table, &challenges)
        .expect("resident full extension-field reduction");
    assert_eq!(stats.dispatches, challenges.len());
    let mut expected = table;
    for challenge in challenges {
        let challenge = k_from_words(challenge);
        expected = expected
            .chunks_exact(2)
            .map(|pair| {
                let left = k_from_words(pair[0]);
                let right = k_from_words(pair[1]);
                words_from_k(left + challenge * (right - left))
            })
            .collect();
    }
    assert_eq!(actual, expected[0]);
}

#[test]
#[ignore = "physical Apple GPU throughput snapshot"]
fn metal_resident_kx_throughput_snapshot() {
    let session = MetalSession::new().expect("Metal session");
    let elements = 1 << 18;
    let rounds = 64;
    let initial = vec![KWords::new(3, 5); elements];
    let multipliers = vec![KWords::new(7, 11); elements];
    session
        .kx_mul_add_chain(&initial, &multipliers, 4)
        .expect("resident extension-field warm-up");
    let mut samples = (0..5)
        .map(|_| {
            session
                .kx_mul_add_chain(&initial, &multipliers, rounds)
                .expect("resident extension-field chain")
                .1
                .elapsed
        })
        .collect::<Vec<_>>();
    samples.sort_unstable();
    let elapsed = samples[samples.len() / 2];
    let operations = elements * rounds;
    println!(
        "METAL_KX_PROFILE_JSON={{\"elements\":{elements},\"dispatches\":{rounds},\"median_ms\":{:.3},\"min_ms\":{:.3},\"max_ms\":{:.3},\"element_dispatches_per_second\":{:.3}}}",
        elapsed.as_secs_f64() * 1e3,
        samples[0].as_secs_f64() * 1e3,
        samples[samples.len() - 1].as_secs_f64() * 1e3,
        operations as f64 / elapsed.as_secs_f64(),
    );

    let field_elements = 1 << 18;
    let lhs = vec![0x1234_5678_9abc_def0; field_elements];
    let rhs = vec![0xfedc_ba98_7654_3210; field_elements];
    for variant in [GoldilocksMulVariant::Limb32, GoldilocksMulVariant::Native64] {
        session
            .goldilocks_ops_variant(&lhs[..1_024], &rhs[..1_024], variant)
            .expect("Goldilocks warm-up");
        let mut samples = (0..5)
            .map(|_| {
                let started = std::time::Instant::now();
                std::hint::black_box(
                    session
                        .goldilocks_ops_variant(&lhs, &rhs, variant)
                        .expect("Goldilocks batch"),
                );
                started.elapsed()
            })
            .collect::<Vec<_>>();
        samples.sort_unstable();
        let elapsed = samples[samples.len() / 2];
        println!(
            "METAL_GOLDILOCKS_PROFILE_JSON={{\"variant\":\"{variant:?}\",\"elements\":{field_elements},\"median_ms\":{:.3},\"min_ms\":{:.3},\"max_ms\":{:.3},\"elements_per_second\":{:.3}}}",
            elapsed.as_secs_f64() * 1e3,
            samples[0].as_secs_f64() * 1e3,
            samples[samples.len() - 1].as_secs_f64() * 1e3,
            field_elements as f64 / elapsed.as_secs_f64(),
        );
    }

    let hash_count = 1 << 15;
    let inputs = vec![vec![1, 2, 3, 4, 5, 6, 7, 8]; hash_count];
    for variant in [PoseidonHashVariant::Scalar, PoseidonHashVariant::SimdGroup] {
        session
            .poseidon2_hash_variant(&inputs[..1_024], variant)
            .expect("Poseidon2 warm-up");
        let mut samples = (0..5)
            .map(|_| {
                let started = std::time::Instant::now();
                std::hint::black_box(
                    session
                        .poseidon2_hash_variant(&inputs, variant)
                        .expect("Poseidon2 batch"),
                );
                started.elapsed()
            })
            .collect::<Vec<_>>();
        samples.sort_unstable();
        let elapsed = samples[samples.len() / 2];
        println!(
            "METAL_POSEIDON_PROFILE_JSON={{\"variant\":\"{variant:?}\",\"hashes\":{hash_count},\"fields_per_hash\":8,\"median_ms\":{:.3},\"min_ms\":{:.3},\"max_ms\":{:.3},\"hashes_per_second\":{:.3}}}",
            elapsed.as_secs_f64() * 1e3,
            samples[0].as_secs_f64() * 1e3,
            samples[samples.len() - 1].as_secs_f64() * 1e3,
            hash_count as f64 / elapsed.as_secs_f64(),
        );
    }
}

#[test]
fn metal_key_weights_match_cpu_key_rows() {
    use neo_ajtai::nightstream_fprime_setup::{coefficient_block, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS};
    let prover = neo_prover_metal::MetalRowProver::new().expect("Metal prover");
    let mut rng = rand::rngs::StdRng::seed_from_u64(91);
    let rows = PRODUCTION_VERIFIER_ROWS as usize;
    let columns = 5;
    let powers: Vec<[u64; 3]> = (0..rows)
        .map(|_| std::array::from_fn(|_| rng.random::<u64>() % GOLDILOCKS_MODULUS))
        .collect();
    let actual = prover
        .production_key_weights(&powers, columns)
        .expect("Metal key weights");
    assert_eq!(actual.len(), columns * D);
    for column in 0..columns {
        let keys: Vec<[u64; D]> = (0..rows)
            .map(|row| coefficient_block(&PRODUCTION_SEED, row as u32, column as u64))
            .collect();
        for lane in 0..D {
            let expected: [u64; 3] = std::array::from_fn(|c| {
                (0..rows)
                    .map(|row| F::from_u64(powers[row][c]) * F::from_u64(keys[row][lane]))
                    .sum::<F>()
                    .as_canonical_u64()
            });
            assert_eq!(actual[column * D + lane], expected, "column {column} lane {lane}");
        }
    }
}

/// The radix-2 DFT of `values` at the powers of `omega`, recursively.
fn fft(values: &[F], omega: F) -> Vec<F> {
    if values.len() == 1 {
        return values.to_vec();
    }
    let half = values.len() / 2;
    let split = |parity: usize| {
        values
            .iter()
            .skip(parity)
            .step_by(2)
            .copied()
            .collect::<Vec<_>>()
    };
    let (even, odd) = (fft(&split(0), omega * omega), fft(&split(1), omega * omega));
    let mut out = vec![F::ZERO; values.len()];
    let mut twiddle = F::ONE;
    for k in 0..half {
        out[k] = even[k] + twiddle * odd[k];
        out[k + half] = even[k] - twiddle * odd[k];
        twiddle *= omega;
    }
    out
}

#[test]
fn metal_dft_matches_cpu_fft() {
    use p3_field::TwoAdicField;
    let session = MetalSession::new().expect("Metal session");
    let mut rng = rand::rngs::StdRng::seed_from_u64(17);
    // One pass, two passes and three passes; narrow and wide rows.
    for (log_rows, width) in [(1, 1), (3, 5), (8, 16), (9, 48), (10, 32), (17, 16)] {
        let rows = 1usize << log_rows;
        let values: Vec<u64> = (0..rows * width).map(|_| rng.random::<u64>()).collect();
        let omega = F::two_adic_generator(log_rows);
        let mut actual = values.clone();
        session
            .dft_columns(&mut actual, width, omega.as_canonical_u64())
            .expect("Metal DFT");
        for column in 0..width {
            let input: Vec<F> = (0..rows)
                .map(|r| F::from_u64(values[r * width + column]))
                .collect();
            for (row, expected) in fft(&input, omega).into_iter().enumerate() {
                assert_eq!(
                    actual[row * width + column],
                    expected.as_canonical_u64(),
                    "2^{log_rows} rows, width {width}, row {row}, column {column}"
                );
            }
        }
    }
    // Past 2^24 rows a pass of one stage would launch 2^32 threads; spot
    // check rows of a 2^25-row DFT by direct evaluation.
    let log_rows = 25;
    let values: Vec<u64> = (0..1u64 << log_rows).map(|_| rng.random::<u64>()).collect();
    let omega = F::two_adic_generator(log_rows);
    let mut actual = values.clone();
    session
        .dft_columns(&mut actual, 1, omega.as_canonical_u64())
        .expect("Metal DFT");
    for row in [0usize, 1, 3, 1 << 24, (1 << 25) - 1] {
        let point = omega.exp_u64(row as u64);
        let mut power = F::ONE;
        let mut expected = F::ZERO;
        for &value in &values {
            expected += F::from_u64(value) * power;
            power *= point;
        }
        assert_eq!(actual[row], expected.as_canonical_u64(), "2^25 rows, row {row}");
    }
}

type E3 = [F; 3];

fn e3_words(value: E3) -> [u64; 3] {
    value.map(|c| c.as_canonical_u64())
}

fn e3_add(a: E3, b: E3) -> E3 {
    std::array::from_fn(|c| a[c] + b[c])
}

fn e3_sub(a: E3, b: E3) -> E3 {
    std::array::from_fn(|c| a[c] - b[c])
}

/// `F[X] / (X^3 - X - 1)`.
fn e3_mul(a: E3, b: E3) -> E3 {
    let mut product = [F::ZERO; 5];
    for i in 0..3 {
        for j in 0..3 {
            product[i + j] += a[i] * b[j];
        }
    }
    // X^3 = X + 1, X^4 = X^2 + X.
    [
        product[0] + product[3],
        product[1] + product[3] + product[4],
        product[2] + product[4],
    ]
}

fn e3_scale(a: E3, x: u64) -> E3 {
    a.map(|c| c * F::from_u64(x))
}

/// One GKR layer on the CPU, as the prover computes it: the round messages,
/// then each tree's children.
fn gkr_layer_reference(
    tables: &mut [Vec<Vec<E3>>],
    factors: &[usize],
    point: &[E3],
    weights: &[E3],
    mu: E3,
    degree: usize,
    challenges: &[E3],
) -> (Vec<Vec<E3>>, Vec<Vec<E3>>) {
    let mut eq = vec![[F::ONE, F::ZERO, F::ZERO]];
    for &c in point {
        let high: Vec<E3> = eq.iter().map(|&v| e3_mul(v, c)).collect();
        let low: Vec<E3> = eq.iter().zip(&high).map(|(&v, &h)| e3_sub(v, h)).collect();
        eq = [low, high].concat();
    }
    let nodes: Vec<u64> = std::iter::once(0).chain(2..=degree as u64).collect();
    let line = |table: &[E3], at: usize, x: u64| e3_add(table[at], e3_scale(e3_sub(table[at + 1], table[at]), x));
    let mut messages = Vec::new();
    for &r in challenges {
        let message = nodes
            .iter()
            .map(|&x| {
                let mut sum = [F::ZERO; 3];
                for m in 0..eq.len() / 2 {
                    let mut total = [F::ZERO; 3];
                    for ((parts, &f), &weight) in tables.iter().zip(factors).zip(weights) {
                        let half = parts[0].len() / 2;
                        let child = |part: usize, b: usize| line(&parts[part], b * half + 2 * m, x);
                        let p =
                            |b: usize| (0..f).fold([F::ONE, F::ZERO, F::ZERO], |acc, part| e3_mul(acc, child(part, b)));
                        let (q0, q1) = (child(f, 0), child(f, 1));
                        let term = e3_add(e3_add(e3_mul(p(0), q1), e3_mul(p(1), q0)), e3_mul(e3_mul(mu, q0), q1));
                        total = e3_add(total, e3_mul(weight, term));
                    }
                    sum = e3_add(sum, e3_mul(line(&eq, 2 * m, x), total));
                }
                sum
            })
            .collect();
        messages.push(message);
        let fold = |table: &mut Vec<E3>| {
            *table = table
                .chunks_exact(2)
                .map(|pair| e3_add(pair[0], e3_mul(r, e3_sub(pair[1], pair[0]))))
                .collect();
        };
        fold(&mut eq);
        for parts in tables.iter_mut() {
            parts.iter_mut().for_each(fold);
        }
    }
    let children = tables
        .iter()
        .map(|parts| parts.iter().flat_map(|part| [part[0], part[1]]).collect())
        .collect();
    (messages, children)
}

#[test]
fn metal_gkr_layers_match_cpu() {
    let session = MetalSession::new().expect("Metal session");
    let mut rng = rand::rngs::StdRng::seed_from_u64(29);
    let mut random = || -> E3 { std::array::from_fn(|_| F::from_u64(rng.random::<u64>())) };
    // (depth, factors): trees ending at different layers, with zero to two
    // factors. Layers from 14 up have tables of whole pages.
    let shapes = [(16usize, 0usize), (16, 2), (15, 1), (14, 0)];
    let mut levels: Vec<Vec<Vec<Vec<E3>>>> = shapes
        .iter()
        .map(|&(depth, factors)| {
            let leaves: Vec<Vec<E3>> = (0..=factors)
                .map(|_| (0..1 << depth).map(|_| random()).collect())
                .collect();
            let mut tree = vec![leaves];
            while tree.last().unwrap()[0].len() > 1 {
                let child = tree.last().unwrap();
                let f = child.len() - 1;
                let half = child[0].len() / 2;
                let numerator =
                    |y: usize| (0..f).fold([F::ONE, F::ZERO, F::ZERO], |acc, part| e3_mul(acc, child[part][y]));
                let (p, q) = (0..half)
                    .map(|y| {
                        let (q0, q1) = (child[f][y], child[f][y + half]);
                        (
                            e3_add(e3_mul(numerator(y), q1), e3_mul(numerator(y + half), q0)),
                            e3_mul(q0, q1),
                        )
                    })
                    .unzip();
                tree.push(vec![p, q]);
            }
            tree.reverse();
            tree
        })
        .collect();
    let words = |values: &[E3]| {
        values
            .iter()
            .flat_map(|&v| e3_words(v))
            .collect::<Vec<u64>>()
    };
    for k in 14..=16 {
        let active: Vec<usize> = (0..shapes.len()).filter(|&t| shapes[t].0 >= k).collect();
        let factors: Vec<usize> = active
            .iter()
            .map(|&t| if shapes[t].0 == k { shapes[t].1 } else { 1 })
            .collect();
        let degree = factors.iter().map(|f| f + 2).max().unwrap().max(3);
        let point: Vec<E3> = (1..k).map(|_| random()).collect();
        let (mu, weights) = (random(), active.iter().map(|_| random()).collect::<Vec<_>>());
        let challenges: Vec<E3> = (1..k).map(|_| random()).collect();
        let mut tables: Vec<Vec<Vec<E3>>> = active
            .iter()
            .map(|&t| std::mem::take(&mut levels[t][k]))
            .collect();
        let mut host: Vec<Vec<Vec<u64>>> = tables
            .iter()
            .map(|parts| parts.iter().map(|part| words(part)).collect())
            .collect();
        let mut eq = vec![[F::ONE, F::ZERO, F::ZERO]];
        for &c in &point {
            let high: Vec<E3> = eq.iter().map(|&v| e3_mul(v, c)).collect();
            let low: Vec<E3> = eq.iter().zip(&high).map(|(&v, &h)| e3_sub(v, h)).collect();
            eq = [low, high].concat();
        }
        let mut eq_words = words(&eq);
        let (messages, children) =
            gkr_layer_reference(&mut tables, &factors, &point, &weights, mu, degree, &challenges);
        {
            let trees: Vec<Vec<&mut [u64]>> = host
                .iter_mut()
                .map(|parts| parts.iter_mut().map(Vec::as_mut_slice).collect())
                .collect();
            let mut layer = session
                .gkr_layer(trees, &mut eq_words)
                .expect("the device takes a layer of whole pages");
            let weight_words: Vec<[u64; 3]> = weights.iter().map(|&w| e3_words(w)).collect();
            for (round, (message, &r)) in messages.iter().zip(&challenges).enumerate() {
                let actual = session
                    .gkr_message(&layer, &weight_words, e3_words(mu), degree)
                    .expect("Metal GKR message");
                let expected: Vec<[u64; 3]> = message.iter().map(|&v| e3_words(v)).collect();
                assert_eq!(actual, expected, "layer {k} round {round}");
                session
                    .gkr_fold(&mut layer, e3_words(r))
                    .expect("Metal GKR fold");
            }
        }
        // The folded host tables hold logical entry j at element j·2^(k-1).
        let stride = 1 << (k - 1);
        for (parts, expected) in host.iter().zip(&children) {
            let actual: Vec<[u64; 3]> = parts
                .iter()
                .flat_map(|part| [0, stride].map(|at| std::array::from_fn(|c| part[3 * at + c])))
                .collect();
            let expected: Vec<[u64; 3]> = expected.iter().map(|&v| e3_words(v)).collect();
            assert_eq!(actual, expected, "layer {k} children");
        }
    }
}
