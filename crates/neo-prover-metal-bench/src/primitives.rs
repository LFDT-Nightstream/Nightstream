//! M1 primitive parity and same-device CPU/Metal timing.

use std::hint::black_box;
use std::time::{Duration, Instant};

use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_math::{from_complex, KExtensions, D, F, K};
use neo_prover_metal::{
    GoldilocksMulVariant, GoldilocksOps, KWords, MetalSession, PoseidonDigest, PoseidonHashVariant,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;

use crate::report::{ActivityReport, BenchmarkConfig, BenchmarkError, CandidateReport, PrimitiveReport, TimingSummary};

const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;

pub fn run_primitive_benchmarks(
    session: &MetalSession,
    config: &BenchmarkConfig,
) -> Result<Vec<PrimitiveReport>, BenchmarkError> {
    config.validate()?;
    Ok(vec![
        benchmark_goldilocks(session, config)?,
        benchmark_poseidon(session, config)?,
        benchmark_kx(session, config)?,
        benchmark_ajtai(session, config)?,
        benchmark_fe_fold(session, config)?,
    ])
}

fn benchmark_goldilocks(session: &MetalSession, config: &BenchmarkConfig) -> Result<PrimitiveReport, BenchmarkError> {
    let lhs = field_words(config.field_elements, 0x1234_5678_9abc_def0);
    let rhs = field_words(config.field_elements, 0xfedc_ba98_7654_3210);
    let cpu_output = cpu_goldilocks(&lhs, &rhs);
    let cpu = sample(config.samples, || black_box(cpu_goldilocks(&lhs, &rhs)));
    let mut candidates = Vec::new();
    for variant in [GoldilocksMulVariant::Limb32, GoldilocksMulVariant::Native64] {
        let output = session.goldilocks_ops_variant(&lhs, &rhs, variant)?;
        if output != cpu_output {
            return Err(BenchmarkError::Parity("Goldilocks"));
        }
        session.reset_activity();
        let timing = sample_result(config.samples, || session.goldilocks_ops_variant(&lhs, &rhs, variant))?;
        candidates.push(candidate(
            format!("{variant:?}"),
            &cpu,
            timing,
            config.field_elements,
            Some(session.activity().into()),
        ));
    }
    Ok(primitive("goldilocks_ops", config.field_elements, cpu, candidates))
}

fn benchmark_poseidon(session: &MetalSession, config: &BenchmarkConfig) -> Result<PrimitiveReport, BenchmarkError> {
    let words = field_words(
        config.poseidon_hashes * config.poseidon_fields_per_hash,
        0x504f_5345_4944_4f4e,
    );
    let cpu_output = cpu_poseidon_uniform(&words, config.poseidon_fields_per_hash);
    let cpu = sample(config.samples, || {
        black_box(cpu_poseidon_uniform(&words, config.poseidon_fields_per_hash))
    });
    let indexed_inputs = words
        .chunks_exact(config.poseidon_fields_per_hash)
        .map(<[u64]>::to_vec)
        .collect::<Vec<_>>();
    let setup_started = Instant::now();
    let uniform_plan = session.prepare_poseidon2_uniform(&words, config.poseidon_fields_per_hash)?;
    let resident_setup_ms = setup_started.elapsed().as_secs_f64() * 1e3;
    let mut candidates = Vec::new();
    for variant in [PoseidonHashVariant::Scalar, PoseidonHashVariant::SimdGroup] {
        let indexed_output = session.poseidon2_hash_variant(&indexed_inputs, variant)?;
        if indexed_output != cpu_output {
            return Err(BenchmarkError::Parity("Poseidon2"));
        }
        session.reset_activity();
        let indexed_timing = sample_result(config.samples, || {
            session.poseidon2_hash_variant(&indexed_inputs, variant)
        })?;
        candidates.push(candidate(
            format!("{variant:?}Indexed"),
            &cpu,
            indexed_timing,
            config.poseidon_hashes,
            Some(session.activity().into()),
        ));

        let uniform_output = session.poseidon2_hash_uniform(&words, config.poseidon_fields_per_hash, variant)?;
        if uniform_output != cpu_output {
            return Err(BenchmarkError::Parity("uniform Poseidon2"));
        }
        session.reset_activity();
        let uniform_timing = sample_result(config.samples, || {
            session.poseidon2_hash_uniform(&words, config.poseidon_fields_per_hash, variant)
        })?;
        candidates.push(candidate(
            format!("{variant:?}Uniform"),
            &cpu,
            uniform_timing,
            config.poseidon_hashes,
            Some(session.activity().into()),
        ));

        let resident_output = session.poseidon2_hash_uniform_with_plan(&uniform_plan, variant)?;
        if resident_output != cpu_output {
            return Err(BenchmarkError::Parity("resident uniform Poseidon2"));
        }
        session.reset_activity();
        let resident_timing = sample_result(config.samples, || {
            session.poseidon2_hash_uniform_with_plan(&uniform_plan, variant)
        })?;
        let mut resident = candidate(
            format!("{variant:?}UniformResident"),
            &cpu,
            resident_timing,
            config.poseidon_hashes,
            Some(session.activity().into()),
        );
        resident.setup_ms = resident_setup_ms;
        candidates.push(resident);
    }
    Ok(primitive("poseidon2_hash", config.poseidon_hashes, cpu, candidates))
}

fn benchmark_kx(session: &MetalSession, config: &BenchmarkConfig) -> Result<PrimitiveReport, BenchmarkError> {
    let initial = k_words(config.kx_elements, 0x4b58_494e_4954_4941);
    let multipliers = k_words(config.kx_elements, 0x4b58_4d55_4c54_4950);
    let cpu_output = cpu_kx_chain(&initial, &multipliers, config.kx_rounds);
    let cpu = sample(config.samples, || {
        black_box(cpu_kx_chain(&initial, &multipliers, config.kx_rounds))
    });
    let output = session
        .kx_mul_add_chain(&initial, &multipliers, config.kx_rounds)?
        .0;
    if output != cpu_output {
        return Err(BenchmarkError::Parity("K extension chain"));
    }
    session.reset_activity();
    let transfer_timing = sample_result(config.samples, || {
        session.kx_mul_add_chain(&initial, &multipliers, config.kx_rounds)
    })?;
    let work_items = config.kx_elements * config.kx_rounds;
    let transfer = candidate(
        "TransferInclusive".to_owned(),
        &cpu,
        transfer_timing,
        work_items,
        Some(session.activity().into()),
    );

    let setup_started = Instant::now();
    let plan = session.prepare_kx_chain(&initial, &multipliers)?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1e3;
    let output = session
        .kx_mul_add_chain_with_plan(&plan, config.kx_rounds)?
        .0;
    if output != cpu_output {
        return Err(BenchmarkError::Parity("resident K extension chain"));
    }
    session.reset_activity();
    let resident_timing = sample_result(config.samples, || {
        session.kx_mul_add_chain_with_plan(&plan, config.kx_rounds)
    })?;
    let mut resident = candidate(
        "ResidentPlan".to_owned(),
        &cpu,
        resident_timing,
        work_items,
        Some(session.activity().into()),
    );
    resident.setup_ms = setup_ms;
    Ok(primitive(
        "kx_resident_chain",
        work_items,
        cpu.clone(),
        vec![transfer, resident],
    ))
}

fn benchmark_ajtai(session: &MetalSession, config: &BenchmarkConfig) -> Result<PrimitiveReport, BenchmarkError> {
    let matrix = field_words(config.ajtai_rows * config.ajtai_cols * D, 0x414a_5441_495f_4d41);
    let message = low_norm_digits(config.ajtai_cols * D, 0x414a_5441_495f_5a5a);
    let cpu_output = cpu_ajtai_low_norm(&matrix, config.ajtai_rows, config.ajtai_cols, &message);
    let cpu = sample(config.samples, || {
        black_box(cpu_ajtai_low_norm(
            &matrix,
            config.ajtai_rows,
            config.ajtai_cols,
            &message,
        ))
    });
    let setup_started = Instant::now();
    let plan = session.prepare_ajtai_low_norm(&matrix, config.ajtai_rows, config.ajtai_cols)?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1e3;
    let output = session.ajtai_low_norm_with_plan(&plan, &message)?;
    if output != cpu_output {
        return Err(BenchmarkError::Parity("Ajtai low-norm ring mat-vec"));
    }
    session.reset_activity();
    let timing = sample_result(config.samples, || session.ajtai_low_norm_with_plan(&plan, &message))?;
    let work_items = config.ajtai_rows * config.ajtai_cols * D;
    let mut low_norm = candidate(
        "LowNorm".to_owned(),
        &cpu,
        timing,
        work_items,
        Some(session.activity().into()),
    );
    low_norm.setup_ms = setup_ms;
    Ok(primitive(
        "ajtai_low_norm_mat_vec",
        work_items,
        cpu.clone(),
        vec![low_norm],
    ))
}

fn benchmark_fe_fold(session: &MetalSession, config: &BenchmarkConfig) -> Result<PrimitiveReport, BenchmarkError> {
    let table = k_words(config.fe_table_elements, 0x4645_5f54_4142_4c45);
    let challenges = k_words(table.len().ilog2() as usize, 0x4645_5f43_4841_4c4c);
    let cpu_output = cpu_fe_fold_full(&table, &challenges);
    let cpu = sample(config.samples, || black_box(cpu_fe_fold_full(&table, &challenges)));
    let output = session.fold_k_table_full(&table, &challenges)?.0;
    if output != cpu_output {
        return Err(BenchmarkError::Parity("resident FE reduction"));
    }
    session.reset_activity();
    let timing = sample_result(config.samples, || session.fold_k_table_full(&table, &challenges))?;
    let work_items = table.len() - 1;
    Ok(primitive(
        "fe_resident_reduction",
        work_items,
        cpu.clone(),
        vec![candidate(
            "ResidentReduction".to_owned(),
            &cpu,
            timing,
            work_items,
            Some(session.activity().into()),
        )],
    ))
}

fn primitive(
    name: &'static str,
    work_items: usize,
    cpu: TimingSummary,
    candidates: Vec<CandidateReport>,
) -> PrimitiveReport {
    let selected = candidates
        .iter()
        .min_by(|lhs, rhs| lhs.timing.median_ms.total_cmp(&rhs.timing.median_ms))
        .expect("primitive has a Metal candidate");
    PrimitiveReport {
        name: name.to_owned(),
        work_items,
        parity_ok: true,
        crossover_required: name != "goldilocks_ops",
        cpu,
        selected_candidate: selected.name.clone(),
        selected_speedup_over_cpu: selected.speedup_over_cpu,
        crossover_gate_passed: selected.speedup_over_cpu >= 2.0,
        candidates,
    }
}

fn candidate(
    name: String,
    cpu: &TimingSummary,
    timing: TimingSummary,
    work_items: usize,
    activity: Option<ActivityReport>,
) -> CandidateReport {
    CandidateReport {
        name,
        setup_ms: 0.0,
        throughput_per_second: work_items as f64 / (timing.median_ms / 1e3),
        speedup_over_cpu: cpu.median_ms / timing.median_ms,
        timing,
        activity,
    }
}

fn sample<T>(samples: usize, mut run: impl FnMut() -> T) -> TimingSummary {
    black_box(run());
    let durations = (0..samples)
        .map(|_| {
            let started = Instant::now();
            black_box(run());
            started.elapsed()
        })
        .collect::<Vec<Duration>>();
    TimingSummary::from_durations(durations)
}

fn sample_result<T, E>(samples: usize, mut run: impl FnMut() -> Result<T, E>) -> Result<TimingSummary, E> {
    black_box(run()?);
    let mut durations = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        black_box(run()?);
        durations.push(started.elapsed());
    }
    Ok(TimingSummary::from_durations(durations))
}

fn field_words(len: usize, seed: u64) -> Vec<u64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % GOLDILOCKS_MODULUS
        })
        .collect()
}

fn k_words(len: usize, seed: u64) -> Vec<KWords> {
    let words = field_words(len * 2, seed);
    words
        .chunks_exact(2)
        .map(|pair| KWords::new(pair[0], pair[1]))
        .collect()
}

fn low_norm_digits(len: usize, seed: u64) -> Vec<i8> {
    field_words(len, seed)
        .into_iter()
        .map(|word| match word % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect()
}

fn cpu_goldilocks(lhs: &[u64], rhs: &[u64]) -> Vec<GoldilocksOps> {
    lhs.par_iter()
        .zip(rhs)
        .map(|(&lhs, &rhs)| {
            let lhs = F::from_u64(lhs);
            let rhs = F::from_u64(rhs);
            GoldilocksOps {
                add: (lhs + rhs).as_canonical_u64(),
                sub: (lhs - rhs).as_canonical_u64(),
                mul: (lhs * rhs).as_canonical_u64(),
            }
        })
        .collect()
}

fn cpu_poseidon_uniform(fields: &[u64], fields_per_hash: usize) -> Vec<PoseidonDigest> {
    fields
        .par_chunks_exact(fields_per_hash)
        .map(|input| {
            let input = input.iter().copied().map(F::from_u64).collect::<Vec<_>>();
            p2::poseidon2_hash(&input).map(|value| value.as_canonical_u64())
        })
        .collect()
}

fn cpu_kx_chain(initial: &[KWords], multipliers: &[KWords], rounds: usize) -> Vec<KWords> {
    initial
        .par_iter()
        .zip(multipliers)
        .map(|(&initial, &multiplier)| {
            let mut value = k_from_words(initial);
            let multiplier = k_from_words(multiplier);
            for _ in 0..rounds {
                value = value * multiplier + value;
            }
            words_from_k(value)
        })
        .collect()
}

fn cpu_ajtai_low_norm(matrix: &[u64], rows: usize, cols: usize, message: &[i8]) -> Vec<u64> {
    (0..rows)
        .into_par_iter()
        .flat_map_iter(|row| {
            let mut accumulator = [F::ZERO; D];
            for col in 0..cols {
                let matrix_base = (row * cols + col) * D;
                let message_base = col * D;
                for shift in 0..D {
                    let digit = message[message_base + shift];
                    if digit == 0 {
                        continue;
                    }
                    for source in 0..D {
                        let mut value = F::from_u64(matrix[matrix_base + source]);
                        if digit < 0 {
                            value = -value;
                        }
                        let exponent = source + shift;
                        if exponent < D {
                            accumulator[exponent] += value;
                        } else if exponent < D + 27 {
                            accumulator[exponent - D] -= value;
                            accumulator[exponent - 27] -= value;
                        } else {
                            accumulator[exponent - 81] += value;
                        }
                    }
                }
            }
            accumulator.map(|coefficient| coefficient.as_canonical_u64())
        })
        .collect()
}

fn cpu_fe_fold_full(table: &[KWords], challenges: &[KWords]) -> KWords {
    let mut current = table.to_vec();
    for challenge in challenges {
        let challenge = k_from_words(*challenge);
        current = current
            .par_chunks_exact(2)
            .map(|pair| {
                let left = k_from_words(pair[0]);
                let right = k_from_words(pair[1]);
                words_from_k(left + challenge * (right - left))
            })
            .collect();
    }
    current[0]
}

fn k_from_words(words: KWords) -> K {
    from_complex(F::from_u64(words.c0), F::from_u64(words.c1))
}

fn words_from_k(value: K) -> KWords {
    let (c0, c1) = value.to_limbs_u64();
    KWords::new(c0, c1)
}
