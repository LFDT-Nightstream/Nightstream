//! Full canonical ring output for one Pad or CCS matrix family.
//! Row weights are aggregated before the linear bar transform and ring
//! product. This keeps storage local to each row partition.

use std::{fs, path::Path, time::Instant};

use neo_math::{from_complex, superneo_bar_matrix, KExtensions, Rq, D, F, K};
use nightstream_fprime::{load_per_application_package, LoadedPerApplicationPackage, PackageError};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;
use serde_json::{json, Value};

use super::oracle::{EqualityWeights, MATRICES, ROUNDS};

const MODULUS: u64 = 0xffff_ffff_0000_0001;

pub struct RingKernel {
    bar: [Vec<(usize, F)>; D],
}

impl RingKernel {
    pub fn new() -> Self {
        Self {
            bar: std::array::from_fn(|row| {
                superneo_bar_matrix()[row]
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, coefficient)| *coefficient != F::ZERO)
                    .collect()
            }),
        }
    }

    pub fn apply(&self, weights: &[K; D], values: &[u8]) -> [K; D] {
        assert_eq!(values.len(), D);
        let z = Rq(std::array::from_fn(|lane| match values[lane] {
            0 => F::ZERO,
            1 => F::ONE,
            255 => -F::ONE,
            _ => panic!("bounded source coefficient"),
        }));
        if z.0.iter().all(|&value| value == F::ZERO) {
            return [K::ZERO; D];
        }
        let real = Rq(std::array::from_fn(|row| {
            self.bar[row]
                .iter()
                .map(|&(column, coefficient)| coefficient * weights[column].as_coeffs()[0])
                .sum()
        }));
        let imaginary = Rq(std::array::from_fn(|row| {
            self.bar[row]
                .iter()
                .map(|&(column, coefficient)| coefficient * weights[column].as_coeffs()[1])
                .sum()
        }));
        let real = real.mul(&z);
        let imaginary = imaginary.mul(&z);
        std::array::from_fn(|lane| K::from_coeffs([real.0[lane], imaginary.0[lane]]))
    }
}

fn extension(words: [u64; 2]) -> K {
    assert!(words.iter().all(|&word| word < MODULUS));
    from_complex(F::from_u64(words[0]), F::from_u64(words[1]))
}

fn add(left: [K; D], right: [K; D]) -> [K; D] {
    std::array::from_fn(|lane| left[lane] + right[lane])
}

pub(super) fn matrix_weights(
    package: &LoadedPerApplicationPackage,
    weights: &EqualityWeights,
    matrix: usize,
    rows: std::ops::Range<usize>,
) -> Vec<Option<Box<[K; D]>>> {
    let mut blocks: Vec<Option<Box<[K; D]>>> = vec![None; package.logical_column_count().div_ceil(D)];
    package
        .visit_matrix_rows(rows, |row, values| {
            let weight = weights.at(row);
            for entry in values
                .matrix(matrix)
                .ok_or(PackageError::Invalid("matrix output family"))?
            {
                let coefficients = blocks[entry.column() / D].get_or_insert_with(|| Box::new([K::ZERO; D]));
                coefficients[entry.column() % D] += weight * K::from(F::from_u64(entry.coefficient()));
            }
            Ok(())
        })
        .expect("exact selected matrix rows");
    blocks
}

pub(super) fn evaluate_family(
    package: &LoadedPerApplicationPackage,
    carrier: &[u8],
    weights: &EqualityWeights,
    matrix: Option<usize>,
) -> [K; D] {
    let kernel = RingKernel::new();
    match matrix {
        None => carrier
            .par_chunks_exact(D)
            .enumerate()
            .map(|(block, values)| {
                let coefficient_weights = std::array::from_fn(|lane| weights.at(block * D + lane));
                kernel.apply(&coefficient_weights, values)
            })
            .reduce(|| [K::ZERO; D], add),
        Some(matrix) => {
            let workers = rayon::current_num_threads();
            let rows_per_worker = package.row_count().div_ceil(workers);
            let partials = (0..workers)
                .into_par_iter()
                .map(|worker| {
                    let start = (worker * rows_per_worker).min(package.row_count());
                    let end = (start + rows_per_worker).min(package.row_count());
                    let blocks = matrix_weights(package, weights, matrix, start..end);
                    let mut output = [K::ZERO; D];
                    let mut active = 0usize;
                    for (block, coefficients) in blocks.into_iter().enumerate() {
                        if let Some(coefficients) = coefficients {
                            active += 1;
                            output = add(output, kernel.apply(&coefficients, &carrier[block * D..block * D + D]));
                        }
                    }
                    (output, active)
                })
                .collect::<Vec<_>>();
            println!(
                "ring family A{matrix} accumulated blocks={}",
                partials.iter().map(|(_, count)| count).sum::<usize>()
            );
            partials
                .into_iter()
                .map(|(value, _)| value)
                .fold([K::ZERO; D], add)
        }
    }
}

pub fn generate(candidate: &Path, identity: [u64; 4], cache: &Path, rounds: &Path, family: &str, output: &Path) {
    let started = Instant::now();
    assert!(!output.exists(), "use a fresh external family output");
    let selected = if family == "all" {
        std::iter::once(None)
            .chain((0..MATRICES).map(Some))
            .collect::<Vec<_>>()
    } else if family == "K" {
        vec![None]
    } else {
        let index = family
            .strip_prefix('A')
            .expect("family K or A0 through A13")
            .parse::<usize>()
            .expect("matrix index");
        assert!(index < MATRICES);
        vec![Some(index)]
    };
    let opening: Value =
        serde_json::from_slice(&fs::read(cache.join("opening.json")).expect("checked opening metadata"))
            .expect("opening cache JSON");
    assert_eq!(opening[0], json!(1));
    assert_eq!(opening[1], json!(identity));
    let round_result: Value =
        serde_json::from_slice(&fs::read(rounds).expect("honest scalar round result")).expect("scalar round JSON");
    assert_eq!(round_result[0], json!(1));
    assert_eq!(round_result[1], opening[1]);
    assert_eq!(round_result[2], opening[2]);
    assert_eq!(round_result[3], opening[9]);
    assert_eq!(round_result[4], opening[10]);
    let point: Vec<[u64; 2]> = serde_json::from_value(round_result[9].clone()).expect("round point");
    assert_eq!(point.len(), ROUNDS);
    let weights = EqualityWeights::new(&point.iter().copied().map(extension).collect::<Vec<_>>());
    let bytes = fs::read(candidate).expect("Lean candidate package");
    let package = load_per_application_package(&bytes, identity).expect("selected candidate identity");
    drop(bytes);
    let binding = package
        .production_verifier_binding()
        .expect("selected verifier binding");
    assert_eq!(opening[2], json!(binding.verifier_context().digest()));
    assert_eq!(opening[3], json!(package.logical_column_count()));
    assert_eq!(opening[5], json!(package.row_count()));
    let carrier = fs::read(cache.join("carrier.i8")).expect("checked carrier");
    assert_eq!(carrier.len(), package.logical_column_count().div_ceil(D) * D);
    assert_eq!(opening[4], json!(carrier.len()));
    assert!(carrier.iter().all(|value| matches!(value, 0 | 1 | 255)));
    assert!(carrier[package.logical_column_count()..]
        .iter()
        .all(|&value| value == 0));
    println!("ring family {family} selected input: {:?}", started.elapsed());

    let mut generated = Vec::with_capacity(selected.len());
    for matrix in selected {
        let label = matrix.map_or_else(|| "K".to_owned(), |matrix| format!("A{matrix}"));
        let value = evaluate_family(&package, &carrier, &weights, matrix);
        let expected_constant: [u64; 2] = serde_json::from_value(match matrix {
            None => round_result[12].clone(),
            Some(matrix) => round_result[13][matrix].clone(),
        })
        .expect("separately folded scalar value");
        assert_eq!(
            value[0],
            extension(expected_constant),
            "ring constant equals the scalar MLE"
        );
        if matrix == Some(MATRICES - 1) {
            assert_eq!(value, [K::ZERO; D], "canonical zero matrix output");
        }
        generated.push((matrix, value));
        println!("full ring family {label} checked: {:?}", started.elapsed());
    }
    let encoded = if family == "all" {
        assert_eq!(generated.len(), MATRICES + 1);
        assert_eq!(generated[0].0, None);
        let mut eval_k = vec![vec![[0u64; 2]; D]; 17];
        let mut eval_a = vec![vec![vec![[0u64; 2]; D]; MATRICES]; 17];
        eval_k[0] = generated[0]
            .1
            .iter()
            .map(|value| <[u64; 2]>::from(value.to_limbs_u64()))
            .collect();
        for matrix in 0..MATRICES {
            assert_eq!(generated[matrix + 1].0, Some(matrix));
            eval_a[0][matrix] = generated[matrix + 1]
                .1
                .iter()
                .map(|value| <[u64; 2]>::from(value.to_limbs_u64()))
                .collect();
        }
        json!([1, opening[10], opening[9], round_result[8], eval_k, eval_a])
    } else {
        assert_eq!(generated.len(), 1);
        let (matrix, value) = generated[0];
        let family_words = value.map(|value| <[u64; 2]>::from(value.to_limbs_u64()));
        json!([
            1,
            identity,
            opening[2],
            opening[9],
            opening[10],
            point,
            u64::from(matrix.is_some()),
            matrix.unwrap_or(0),
            family_words.as_slice()
        ])
    };
    let mut bytes = serde_json::to_vec(&encoded).expect("canonical full-family result");
    bytes.push(b'\n');
    fs::write(output, bytes).expect("complete family sink");
    println!(
        "full_ring_family={family} coefficients={} elapsed={:?}",
        D,
        started.elapsed()
    );
}
