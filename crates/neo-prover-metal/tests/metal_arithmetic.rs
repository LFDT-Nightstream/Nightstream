#![cfg(all(target_vendor = "apple", neo_metal_shaders))]

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

    let inputs = [0, 1, 3, 4, 5, 8, 9, 17, 64]
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
