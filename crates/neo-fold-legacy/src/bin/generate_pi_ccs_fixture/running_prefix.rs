//! Build the multilinear running term from the canonical matrices and
//! the checked signed-unit carrier. The generated prefix is prover data;
//! full phase and independent opening checks remain separate gates.

use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

use neo_math::{from_complex, superneo_bar_block, KExtensions, Rq, D, F, K};
use nightstream_fprime::{load_per_application_package, PackageError};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;
use serde_json::{json, Value};

use super::oracle::{EqualityWeights, MATRICES, ROUNDS};

const MODULUS: u64 = 0xffff_ffff_0000_0001;
const RUNNING: usize = 16;

fn field(word: u64) -> F {
    assert!(word < MODULUS, "canonical field word");
    F::from_u64(word)
}

fn extension(words: [u64; 2]) -> K {
    from_complex(field(words[0]), field(words[1]))
}

fn power(value: K, count: usize) -> K {
    (0..count).fold(K::ONE, |product, _| product * value)
}

/// Coefficient-weighted Bar(e_lane) * z for the I_K and I_A exponents.
/// Fixed kernels turn each signed source coefficient into additions only.
pub struct RunningKernel {
    by_power: [[(K, K); D]; D],
}

impl RunningKernel {
    pub fn new(gamma: K) -> Self {
        let pad_weights: [K; D] = std::array::from_fn(|coefficient| power(gamma, RUNNING * coefficient));
        let matrix_weights: [K; D] = std::array::from_fn(|coefficient| power(gamma, RUNNING * MATRICES * coefficient));
        let basis: [Rq; D] = std::array::from_fn(|lane| {
            let mut unit = [F::ZERO; D];
            unit[lane] = F::ONE;
            Rq(superneo_bar_block(unit))
        });
        Self {
            by_power: std::array::from_fn(|power| {
                std::array::from_fn(|lane| {
                    let product = basis[lane].mul_by_monomial(power);
                    let mut pad = K::ZERO;
                    let mut matrix = K::ZERO;
                    for coefficient in 0..D {
                        let value = K::from(product.0[coefficient]);
                        pad += value * pad_weights[coefficient];
                        matrix += value * matrix_weights[coefficient];
                    }
                    (pad, matrix)
                })
            }),
        }
    }

    pub fn apply(&self, source: &[u8]) -> ([K; D], [K; D]) {
        assert_eq!(source.len(), D);
        let mut pad = [K::ZERO; D];
        let mut matrix = [K::ZERO; D];
        for (power, &value) in source.iter().enumerate() {
            match value {
                0 => {}
                1 => {
                    for lane in 0..D {
                        pad[lane] += self.by_power[power][lane].0;
                        matrix[lane] += self.by_power[power][lane].1;
                    }
                }
                255 => {
                    for lane in 0..D {
                        pad[lane] -= self.by_power[power][lane].0;
                        matrix[lane] -= self.by_power[power][lane].1;
                    }
                }
                _ => panic!("signed-unit carrier coefficient"),
            }
        }
        (pad, matrix)
    }

    /// Linear extension to the gamma-weighted sum of distinct child openings.
    pub fn apply_combination(&self, source: &[K; D]) -> ([K; D], [K; D]) {
        let mut pad = [K::ZERO; D];
        let mut matrix = [K::ZERO; D];
        for (power, &value) in source.iter().enumerate() {
            if value == K::ZERO {
                continue;
            }
            for lane in 0..D {
                pad[lane] += value * self.by_power[power][lane].0;
                matrix[lane] += value * self.by_power[power][lane].1;
            }
        }
        (pad, matrix)
    }
}

/// Infer each sign from the full checked commitment and public input.
pub(super) fn sign_sum(prelude: &Value, commitment: &[u64], public: &[u64], gamma: K) -> K {
    let commitments: Vec<Vec<u64>> = serde_json::from_value(prelude[2][1].clone()).expect("running commitments");
    let public_inputs: Vec<Vec<u64>> = serde_json::from_value(prelude[2][2].clone()).expect("running public inputs");
    assert_eq!((commitments.len(), public_inputs.len()), (RUNNING, RUNNING));
    let negative_commitment = commitment
        .iter()
        .map(|&word| (-field(word)).as_canonical_u64())
        .collect::<Vec<_>>();
    let negative_public = public
        .iter()
        .map(|&word| (-field(word)).as_canonical_u64())
        .collect::<Vec<_>>();
    let mut weight = K::ONE;
    let mut sum = K::ZERO;
    for source in 0..RUNNING {
        let sign = if commitments[source] == commitment {
            assert_eq!(public_inputs[source], public);
            K::ONE
        } else {
            assert_eq!(
                commitments[source], negative_commitment,
                "signed copy of the checked opening"
            );
            assert_eq!(public_inputs[source], negative_public);
            -K::ONE
        };
        sum += weight * sign;
        weight *= gamma;
    }
    sum
}

pub fn generate(
    candidate: &Path,
    identity: [u64; 4],
    cache: &Path,
    lean_prelude: &Path,
    folded_children: Option<&Path>,
    output: &Path,
) {
    let started = Instant::now();
    assert!(!output.exists(), "use a fresh external running-prefix sink");
    let opening: Value =
        serde_json::from_slice(&fs::read(cache.join("opening.json")).expect("opening metadata")).expect("opening JSON");
    let prelude: Value = serde_json::from_slice(&fs::read(lean_prelude).expect("Lean prelude")).expect("prelude JSON");
    assert_eq!(opening[0], 1);
    assert_eq!(opening[1], json!(identity));
    assert_eq!(prelude[0], 1);
    assert_eq!(prelude[1][0], 2);
    assert_eq!(prelude[1][1], opening[10]);
    assert_eq!(prelude[1][2], opening[9]);
    let gamma = extension(serde_json::from_value(prelude[5][2].clone()).expect("Lean gamma"));
    let initial = extension(serde_json::from_value(prelude[5][7].clone()).expect("Lean initial claim"));
    let prior: Vec<[u64; 2]> = serde_json::from_value(prelude[2][0].clone()).expect("prior point");
    assert_eq!(prior.len(), ROUNDS);
    let commitment: Vec<u64> = serde_json::from_value(opening[10].clone()).expect("fresh commitment");
    let public: Vec<u64> = serde_json::from_value(opening[9].clone()).expect("fresh public input");
    let sign_sum = folded_children
        .is_none()
        .then(|| sign_sum(&prelude, &commitment, &public, gamma));
    let bytes = fs::read(candidate).expect("Lean candidate package");
    let package = load_per_application_package(&bytes, identity).expect("selected canonical package");
    drop(bytes);
    assert_eq!(
        opening[2],
        json!(package
            .production_verifier_binding()
            .expect("binding")
            .verifier_context()
            .digest())
    );
    assert_eq!(opening[3], json!(package.logical_column_count()));
    assert_eq!(opening[5], json!(package.row_count()));
    assert_eq!(
        (opening[6].as_u64(), opening[7].as_u64()),
        (Some(MATRICES as u64), Some(ROUNDS as u64))
    );
    let carrier = fs::read(cache.join("carrier.i8")).expect("checked carrier");
    assert_eq!(carrier.len(), package.logical_column_count().div_ceil(D) * D);
    assert!(carrier.len().max(package.row_count()) <= 1 << ROUNDS);
    assert!(carrier[package.logical_column_count()..]
        .iter()
        .all(|&value| value == 0));
    println!("running prefix input: {:?}", started.elapsed());

    let kernel = RunningKernel::new(gamma);
    let mut pad = vec![K::ZERO; carrier.len().max(package.row_count())];
    let mut matrix = vec![K::ZERO; carrier.len()];
    if let Some(children) = folded_children {
        let sources = super::folded_opening::child_assignments(
            children,
            identity,
            package
                .production_verifier_binding()
                .expect("selected context")
                .verifier_context()
                .digest(),
            package.logical_column_count(),
            carrier.len(),
            &prelude,
        );
        let powers: [K; RUNNING] = std::array::from_fn(|source| power(gamma, source));
        pad[..carrier.len()]
            .par_chunks_mut(D)
            .zip(matrix.par_chunks_mut(D))
            .enumerate()
            .for_each(|(block, (pad, matrix))| {
                let mut combined = [K::ZERO; D];
                for (source, weight) in sources.iter().zip(powers) {
                    if source.is_empty() {
                        continue;
                    }
                    for lane in 0..D {
                        match source[block * D + lane] {
                            0 => {}
                            1 => combined[lane] += weight,
                            255 => combined[lane] -= weight,
                            _ => unreachable!("checked child signed unit"),
                        }
                    }
                }
                let (p, m) = kernel.apply_combination(&combined);
                pad.copy_from_slice(&p);
                matrix.copy_from_slice(&m);
            });
    } else {
        pad[..carrier.len()]
            .par_chunks_mut(D)
            .zip(matrix.par_chunks_mut(D))
            .zip(carrier.par_chunks_exact(D))
            .for_each(|((pad, matrix), source)| {
                let (p, m) = kernel.apply(source);
                pad.copy_from_slice(&p);
                matrix.copy_from_slice(&m);
            });
    }
    println!("running coefficient kernels: {:?}", started.elapsed());
    let matrix_weights: [K; MATRICES] = std::array::from_fn(|matrix| power(gamma, RUNNING * matrix));
    let matrix_shift = power(gamma, RUNNING * D);
    let rows_per_worker = package.row_count().div_ceil(rayon::current_num_threads());
    pad[..package.row_count()]
        .par_chunks_mut(rows_per_worker)
        .enumerate()
        .try_for_each(|(worker, rows)| {
            let start = worker * rows_per_worker;
            package.visit_matrix_rows(start..start + rows.len(), |row, matrices| {
                let mut sum = K::ZERO;
                for family in 0..MATRICES {
                    let mut value = K::ZERO;
                    for entry in matrices
                        .matrix(family)
                        .ok_or(PackageError::Invalid("running matrix family"))?
                    {
                        value += K::from(field(entry.coefficient())) * matrix[entry.column()];
                    }
                    sum += matrix_weights[family] * value;
                }
                rows[row - start] += matrix_shift * sum;
                Ok(())
            })
        })
        .expect("canonical matrix running prefix");
    drop(matrix);
    if let Some(sign_sum) = sign_sum {
        pad.par_iter_mut().for_each(|value| *value *= sign_sum);
    }
    println!("running canonical matrix prefix: {:?}", started.elapsed());
    let weights = EqualityWeights::new(&prior.into_iter().map(extension).collect::<Vec<_>>());
    let evaluated: K = pad
        .par_iter()
        .enumerate()
        .map(|(row, &value)| weights.at(row) * value)
        .sum();
    assert_eq!(
        evaluated, initial,
        "actual running prefix evaluates to Lean's initial claim"
    );
    let mut sink = BufWriter::new(fs::File::create(output).expect("external running-prefix sink"));
    for value in pad {
        let words: [u64; 2] = value.to_limbs_u64().into();
        for word in words {
            sink.write_all(&word.to_le_bytes())
                .expect("canonical prefix word");
        }
    }
    sink.flush().expect("complete running-prefix sink");
    println!(
        "canonical_running_prefix={} elapsed={:?}",
        output.display(),
        started.elapsed()
    );
    println!("full_nonzero_running_phase_and_independent_openings=still_required");
}
