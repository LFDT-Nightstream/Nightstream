//! CE claims for the actual folded digits at the preceding PiCCS point.
//! Matrices come from the selected Lean package. These children need not
//! satisfy the fresh CCS polynomial; their complete openings are checked separately.

use std::{fs, path::Path, time::Instant};

use neo_math::{from_complex, KExtensions, D, F, K};
use nightstream_fprime::{load_per_application_package, LoadedPerApplicationPackage};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;
use serde_json::{json, Value};

use super::{
    folded_opening::{signed_digits, DIGITS, PARENT_BOUND},
    oracle::{EqualityWeights, MATRICES, ROUNDS},
    ring_output,
};

const MODULUS: u64 = 0xffff_ffff_0000_0001;
const PUBLIC: usize = 270;
const COMMITMENT: usize = 22 * D;

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("child evaluation input")).expect("child evaluation JSON")
}

struct Inputs {
    package: LoadedPerApplicationPackage,
    parent: Vec<u8>,
    context: [u64; 4],
    point: Vec<[u64; 2]>,
    public: Vec<Vec<i8>>,
    bound: u64,
}

impl Inputs {
    fn load(candidate: &Path, identity: [u64; 4], cache: &Path) -> Self {
        let bytes = fs::read(candidate).expect("canonical candidate package");
        let package = load_per_application_package(&bytes, identity).expect("selected package identity");
        drop(bytes);
        let context = package
            .production_verifier_binding()
            .expect("selected binding")
            .verifier_context()
            .digest();
        assert_eq!(
            (
                package.ccs_relation().cube_variables(),
                package.ccs_relation().matrix_sources().len()
            ),
            (ROUNDS, MATRICES)
        );
        let meta = read(&cache.join("folded.json"));
        assert_eq!(meta.as_array().expect("folded metadata").len(), 13);
        assert_eq!(meta[0], 1);
        assert_eq!(meta[1], json!(identity));
        assert_eq!(meta[2], json!(context));
        assert_eq!(meta[3], json!(package.logical_column_count()));
        let native_width = package.logical_column_count().div_ceil(D) * D;
        assert_eq!(meta[4], json!(native_width));
        assert!(native_width.max(package.row_count()) <= 1 << ROUNDS);
        let bound = meta[8].as_u64().expect("parent bound");
        assert!(bound < PARENT_BOUND as u64);
        let point: Vec<[u64; 2]> = serde_json::from_value(meta[12].clone()).expect("child point");
        assert_eq!(point.len(), ROUNDS);
        assert!(point.iter().flatten().all(|&word| word < MODULUS));
        let public: Vec<Vec<i8>> = serde_json::from_value(meta[11].clone()).expect("child public digits");
        assert_eq!(public.len(), DIGITS);
        assert!(public.iter().all(|values| values.len() == PUBLIC));
        let parent = fs::read(cache.join("folded.i16")).expect("full native folded carrier");
        assert_eq!(parent.len(), native_width * 2);
        parent.par_chunks_exact(2).for_each(|word| {
            let value = i16::from_le_bytes(word.try_into().unwrap());
            assert!(u64::from(value.unsigned_abs()) <= bound, "actual parent bound");
        });
        for column in 0..PUBLIC {
            let value = i16::from_le_bytes(parent[2 * column..2 * column + 2].try_into().unwrap());
            let digits = signed_digits(i32::from(value)).expect("strict public parent bound");
            for child in 0..DIGITS {
                assert_eq!(public[child][column], digits[child]);
            }
        }
        Self {
            package,
            parent,
            context,
            point,
            public,
            bound,
        }
    }

    fn child_record(&self, identity: [u64; 4], children: &Path, child: usize) -> Value {
        let record = read(&children.join(format!("child-{child}.json")));
        assert_eq!(record.as_array().expect("child commitment record").len(), 7);
        assert_eq!(record[0], 1);
        assert_eq!(record[1], json!(identity));
        assert_eq!(record[2], json!(self.context));
        assert_eq!(record[3], json!(child));
        assert_eq!(record[4], json!(self.point));
        let public: Vec<u64> = serde_json::from_value(record[5].clone()).expect("child public input");
        let expected: Vec<u64> = self.public[child]
            .iter()
            .map(|&value| match value {
                -1 => MODULUS - 1,
                0 => 0,
                1 => 1,
                _ => panic!("signed child public digit"),
            })
            .collect();
        assert_eq!(public, expected);
        let commitment: Vec<u64> = serde_json::from_value(record[6].clone()).expect("child commitment");
        assert_eq!(commitment.len(), COMMITMENT);
        assert!(commitment.iter().all(|&word| word < MODULUS));
        record
    }
}

type Values = [[K; D]; DIGITS];

fn merge(mut left: Box<Values>, right: Box<Values>) -> Box<Values> {
    for (left, right) in left.iter_mut().zip(right.iter()) {
        for (left, right) in left.iter_mut().zip(right) {
            *left += *right;
        }
    }
    left
}

fn add_digits(total: &mut Values, kernel: &ring_output::RingKernel, weights: &[K; D], parent: &[u8], active: usize) {
    let native: [i16; D] =
        std::array::from_fn(|lane| i16::from_le_bytes(parent[2 * lane..2 * lane + 2].try_into().unwrap()));
    if native.iter().all(|&value| value == 0) {
        return;
    }
    for child in 0..active {
        let source: [u8; D] = native.map(|value| signed_digits(i32::from(value)).expect("bounded parent")[child] as u8);
        if source.iter().all(|&value| value == 0) {
            continue;
        }
        let result = kernel.apply(weights, &source);
        for (target, value) in total[child].iter_mut().zip(result) {
            *target += value;
        }
    }
}

pub fn generate(
    candidate: &Path,
    identity: [u64; 4],
    cache: &Path,
    children: &Path,
    families: &str,
    rounds: Option<&Path>,
    output: &Path,
) {
    let started = Instant::now();
    assert!(!output.exists(), "use a fresh external child family sink");
    let selected: Vec<Option<usize>> = families
        .split(',')
        .map(|family| {
            if family == "K" {
                None
            } else {
                let matrix = family
                    .strip_prefix('A')
                    .expect("K or A0..A13")
                    .parse::<usize>()
                    .expect("matrix index");
                assert!(matrix < MATRICES);
                Some(matrix)
            }
        })
        .collect();
    for index in 0..selected.len() {
        assert!(!selected[..index].contains(&selected[index]), "duplicate family");
    }
    let inputs = Inputs::load(candidate, identity, cache);
    let records = (0..DIGITS)
        .map(|child| inputs.child_record(identity, children, child))
        .collect::<Vec<_>>();
    let active = (u64::BITS - inputs.bound.leading_zeros()) as usize;
    for record in &records[active..] {
        assert_eq!(record[5], json!(vec![0u64; PUBLIC]));
        assert_eq!(record[6], json!(vec![0u64; COMMITMENT]));
    }
    let output_point = if let Some(path) = rounds {
        let result = read(path);
        assert_eq!(result.as_array().expect("round result fields").len(), 14);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], json!(identity));
        assert_eq!(result[2], json!(inputs.context));
        let point: Vec<[u64; 2]> = serde_json::from_value(result[9].clone()).expect("derived round point");
        assert_eq!(point.len(), ROUNDS);
        assert!(point.iter().flatten().all(|&word| word < MODULUS));
        point
    } else {
        inputs.point.clone()
    };
    let point = output_point
        .iter()
        .map(|words| from_complex(F::from_u64(words[0]), F::from_u64(words[1])))
        .collect::<Vec<_>>();
    let weights = EqualityWeights::new(&point);
    let kernel = ring_output::RingKernel::new();
    let batch = selected.len() > 1;
    if batch {
        fs::create_dir(output).expect("fresh child family batch directory");
    }
    println!(
        "child_families={families} children={DIGITS} input_time={:?}",
        started.elapsed()
    );
    for selected in selected {
        let family = selected.map_or_else(|| "K".to_owned(), |matrix| format!("A{matrix}"));
        let target = if batch {
            output.join(format!("family-{family}.json"))
        } else {
            output.to_path_buf()
        };
        let values = match selected {
            None => inputs
                .parent
                .par_chunks_exact(2 * D)
                .enumerate()
                .fold(
                    || Box::new([[K::ZERO; D]; DIGITS]),
                    |mut total, (block, parent)| {
                        let coefficients = std::array::from_fn(|lane| weights.at(block * D + lane));
                        add_digits(&mut total, &kernel, &coefficients, parent, active);
                        total
                    },
                )
                .reduce(|| Box::new([[K::ZERO; D]; DIGITS]), merge),
            Some(matrix) => {
                let workers = rayon::current_num_threads();
                let rows_per_worker = inputs.package.row_count().div_ceil(workers);
                (0..workers)
                    .into_par_iter()
                    .map(|worker| {
                        let start = (worker * rows_per_worker).min(inputs.package.row_count());
                        let end = (start + rows_per_worker).min(inputs.package.row_count());
                        let blocks = ring_output::matrix_weights(&inputs.package, &weights, matrix, start..end);
                        let mut total = Box::new([[K::ZERO; D]; DIGITS]);
                        for (block, coefficients) in blocks.into_iter().enumerate() {
                            if let Some(coefficients) = coefficients {
                                add_digits(
                                    &mut total,
                                    &kernel,
                                    &coefficients,
                                    &inputs.parent[block * 2 * D..(block + 1) * 2 * D],
                                    active,
                                );
                            }
                        }
                        total
                    })
                    .reduce(|| Box::new([[K::ZERO; D]; DIGITS]), merge)
            }
        };
        if selected == Some(MATRICES - 1) {
            assert_eq!(*values, [[K::ZERO; D]; DIGITS], "canonical zero matrix");
        }
        let encoded_values = values
            .iter()
            .map(|child| {
                child
                    .iter()
                    .map(|value| <[u64; 2]>::from(value.to_limbs_u64()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let result = json!([
            1,
            identity,
            inputs.context,
            output_point,
            u64::from(selected.is_some()),
            selected.unwrap_or(0),
            records.iter().map(|record| &record[5]).collect::<Vec<_>>(),
            records.iter().map(|record| &record[6]).collect::<Vec<_>>(),
            encoded_values
        ]);
        let mut encoded = serde_json::to_vec(&result).expect("complete numeric child family");
        encoded.push(b'\n');
        fs::write(&target, encoded).expect("external child family sink");
        println!(
            "complete_child_family={family} children={DIGITS} coefficients={} elapsed={:?}",
            DIGITS * D,
            started.elapsed()
        );
        println!("independent_evaluation_and_matching_fprime_witness=still_required");
    }
}
