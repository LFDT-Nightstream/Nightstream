//! Integer packed fold of the checked base opening for a recursive fixture.
//! This produces witness input data. Package rows and independent opening
//! checks remain responsible for accepting the resulting parent and children.

use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

use neo_fold_clean::{engine::optimized, paper::params::Params};
use neo_math::{D, F};
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::load_per_application_package;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;
use serde_json::{json, Value};

pub const DIGITS: usize = 16;
pub const PARENT_BOUND: i32 = 1 << DIGITS;
const MODULUS: u64 = 0xffff_ffff_0000_0001;

pub struct FoldKernel {
    columns: [[i16; D]; D],
    norm_bound: i16,
}

impl FoldKernel {
    pub fn new(rho: [i8; D]) -> Self {
        assert!(rho.iter().all(|&value| (-2..=2).contains(&value)));
        let columns: [[i16; D]; D] = std::array::from_fn(|column| {
            let mut values = [0i16; D];
            for (coefficient, &value) in rho.iter().enumerate() {
                // Phi81=X^54+X^27+1, hence X^81=1 in the quotient.
                let power = (column + coefficient) % 81;
                if power < D {
                    values[power] += i16::from(value);
                } else {
                    values[power - 54] -= i16::from(value);
                    values[power - 27] -= i16::from(value);
                }
            }
            values
        });
        let norm_bound = (0..D)
            .map(|row| {
                columns
                    .iter()
                    .map(|column: &[i16; D]| column[row].abs())
                    .sum::<i16>()
            })
            .max()
            .unwrap();
        // The existing five-symbol profile has T=2*54*2=216.
        assert!(norm_bound <= 2 * D as i16 * 2);
        Self { columns, norm_bound }
    }

    pub fn norm_bound(&self) -> i16 {
        self.norm_bound
    }

    pub fn apply(&self, source: &[u8; D]) -> [i16; D] {
        let mut output = [0i16; D];
        for (column, &value) in source.iter().enumerate() {
            match value {
                0 => {}
                1 => {
                    for (out, &term) in output.iter_mut().zip(&self.columns[column]) {
                        *out += term;
                    }
                }
                255 => {
                    for (out, &term) in output.iter_mut().zip(&self.columns[column]) {
                        *out -= term;
                    }
                }
                _ => panic!("fresh opening is not a signed unit"),
            }
        }
        output
    }
}

pub fn signed_digits(value: i32) -> Option<[i8; DIGITS]> {
    if value.unsigned_abs() >= PARENT_BOUND as u32 {
        return None;
    }
    let sign = if value < 0 { -1 } else { 1 };
    Some(std::array::from_fn(|bit| {
        sign * ((value.unsigned_abs() >> bit) & 1) as i8
    }))
}

fn field(value: u64) -> F {
    assert!(value < MODULUS, "canonical field word");
    F::from_u64(value)
}

fn centered_challenge(value: F) -> i8 {
    let word = value.as_canonical_u64();
    match word {
        0..=2 => word as i8,
        value if value == MODULUS - 1 => -1,
        value if value == MODULUS - 2 => -2,
        _ => panic!("sampler coefficient is outside the five-symbol alphabet"),
    }
}

fn zero(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.iter().all(zero),
        Value::Number(value) => value.as_u64() == Some(0),
        _ => false,
    }
}

pub fn generate(candidate: &Path, expected: [u64; 4], cache: &Path, lean_path: &Path, output: &Path) {
    let started = Instant::now();
    assert!(!output.exists(), "use a fresh external folded-opening directory");
    let bytes = fs::read(candidate).expect("canonical candidate");
    let package = load_per_application_package(&bytes, expected).expect("selected candidate identity");
    drop(bytes);
    let logical_width = package.logical_column_count();
    let row_count = package.row_count();
    let context = package
        .production_verifier_binding()
        .expect("selected binding")
        .verifier_context()
        .digest();
    assert_eq!(package.ccs_relation().cube_variables(), 28);
    assert_eq!(package.ccs_relation().matrix_sources().len(), 14);
    drop(package);
    let opening: Value =
        serde_json::from_slice(&fs::read(cache.join("opening.json")).expect("opening metadata")).expect("opening JSON");
    let lean: Value =
        serde_json::from_slice(&fs::read(lean_path).expect("accepted PiCCS result")).expect("Lean result JSON");
    assert_eq!(opening[0], 1);
    assert_eq!(opening[1], json!(expected));
    assert_eq!(opening[2], json!(context));
    assert_eq!(opening[3], json!(logical_width));
    assert_eq!(opening[5], json!(row_count));
    assert_eq!((opening[6].as_u64(), opening[7].as_u64()), (Some(14), Some(28)));
    assert_eq!(lean[0], 1);
    assert_eq!(lean[1][0], 2);
    assert_eq!(lean[1][1], opening[10], "same checked fresh commitment");
    assert_eq!(lean[1][2], opening[9], "same checked fresh public input");
    assert!(zero(&lean[1][6]), "this first fold uses sixteen zero running openings");
    assert_eq!(lean[5].as_array().expect("complete phase result").len(), 15);
    assert_eq!(lean[5][0], 1, "accepted honest PiCCS proof");
    let state: [u64; 8] = serde_json::from_value(lean[5][14].clone()).expect("PiCCS outgoing state");
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(state.map(field), 0);
    let rhos = optimized::sample_rho_n(&mut transcript, &Params::production(), 17).expect("existing bounded sampler");
    let rho: [i8; D] = std::array::from_fn(|row| centered_challenge(rhos[0].as_mat()[(row, 0)]));
    let kernel = FoldKernel::new(rho);
    for column in 0..D {
        let mut basis = [0u8; D];
        basis[column] = 1;
        for (row, value) in kernel.apply(&basis).into_iter().enumerate() {
            let magnitude = F::from_u64(u64::from(value.unsigned_abs()));
            let lifted = if value < 0 { -magnitude } else { magnitude };
            assert_eq!(lifted, rhos[0].as_mat()[(row, column)], "sampled rotation entry");
        }
    }
    let coefficients: Vec<Vec<i8>> = rhos
        .iter()
        .map(|rho| {
            (0..D)
                .map(|row| centered_challenge(rho.as_mat()[(row, 0)]))
                .collect()
        })
        .collect();
    let carrier = fs::read(cache.join("carrier.i8")).expect("checked base carrier");
    let carrier_width = logical_width.div_ceil(D) * D;
    assert_eq!(carrier.len(), carrier_width);
    assert_eq!(opening[4], json!(carrier_width));
    assert!(
        carrier[logical_width..].iter().all(|&value| value == 0),
        "fresh alignment suffix"
    );
    let public: Vec<u64> = serde_json::from_value(opening[9].clone()).expect("fresh public projection");
    assert_eq!(public.len(), 270);
    assert_eq!(
        carrier[..270]
            .iter()
            .copied()
            .map(u64::from)
            .collect::<Vec<_>>(),
        public
    );
    println!("folded opening inputs: {:?}", started.elapsed());
    let folded: Vec<[i16; D]> = carrier
        .par_chunks_exact(D)
        .map(|block| kernel.apply(block.try_into().expect("complete ring block")))
        .collect();
    drop(carrier);
    let (max_norm, digit_mask) = folded
        .par_iter()
        .map(|block| {
            block.iter().fold((0u16, 0u16), |(bound, mask), value| {
                let magnitude = value.unsigned_abs();
                (bound.max(magnitude), mask | magnitude)
            })
        })
        .reduce(|| (0, 0), |left, right| (left.0.max(right.0), left.1 | right.1));
    assert!(max_norm <= kernel.norm_bound() as u16);
    assert!(i32::from(max_norm) < PARENT_BOUND);
    let parent_public: Vec<i16> = folded.iter().flatten().take(270).copied().collect();
    let child_public: Vec<Vec<i8>> = (0..DIGITS)
        .map(|child| {
            parent_public
                .iter()
                .map(|&value| signed_digits(i32::from(value)).expect("strict parent bound")[child])
                .collect()
        })
        .collect();
    println!(
        "packed integer fold: {:?}; max_norm={max_norm} exact_rho_bound={}",
        started.elapsed(),
        kernel.norm_bound()
    );
    fs::create_dir(output).expect("new external folded-opening directory");
    let mut writer = BufWriter::new(fs::File::create(output.join("folded.i16")).expect("folded carrier sink"));
    for block in &folded {
        let mut bytes = [0u8; D * 2];
        for (lane, value) in block.iter().enumerate() {
            bytes[lane * 2..lane * 2 + 2].copy_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes).expect("integer ring block");
    }
    writer.flush().expect("complete folded carrier");
    let metadata = json!([
        1,
        expected,
        context,
        logical_width,
        carrier_width,
        state,
        coefficients,
        transcript.state().map(|value| value.as_canonical_u64()),
        max_norm,
        digit_mask,
        parent_public,
        child_public,
        lean[5][6]
    ]);
    let mut encoded = serde_json::to_vec(&metadata).expect("folded metadata JSON");
    encoded.push(b'\n');
    fs::write(output.join("folded.json"), encoded).expect("folded metadata sink");
    println!(
        "folded_base_opening={} carrier={carrier_width} elapsed={:?}",
        output.display(),
        started.elapsed()
    );
    println!("child_openings_parent_assignment_and_next_positive_pi_ccs=still_required");
}

pub(super) fn is_zero(value: &Value) -> bool {
    match value {
        Value::Array(values) => values.iter().all(is_zero),
        Value::Number(value) => value.as_u64() == Some(0),
        _ => false,
    }
}

pub(super) fn child_assignments(
    cache: &Path,
    identity: [u64; 4],
    context: [u64; 4],
    logical_width: usize,
    carrier_width: usize,
    prelude: &Value,
) -> [Vec<u8>; DIGITS] {
    let metadata: Value = serde_json::from_slice(&fs::read(cache.join("folded.json")).expect("folded child metadata"))
        .expect("folded metadata JSON");
    assert_eq!(metadata.as_array().expect("folded metadata fields").len(), 13);
    assert_eq!(metadata[0], 1);
    assert_eq!(metadata[1], json!(identity));
    assert_eq!(metadata[2], json!(context));
    assert_eq!(metadata[3], json!(logical_width));
    assert_eq!(metadata[4], json!(carrier_width));
    assert_eq!(metadata[12], prelude[2][0], "exact child claim point");
    assert_eq!(prelude[2], prelude[1][6], "same complete running statement");
    let bound = metadata[8].as_u64().expect("actual folded bound");
    assert!(bound < PARENT_BOUND as u64);
    let bytes = fs::read(cache.join("folded.i16")).expect("complete native folded carrier");
    assert_eq!(bytes.len(), carrier_width * 2);
    let native: Vec<i16> = bytes
        .par_chunks_exact(2)
        .map(|word| {
            let value = i16::from_le_bytes(word.try_into().expect("folded coefficient"));
            assert!(
                u64::from(value.unsigned_abs()) <= bound,
                "actual folded coefficient bound"
            );
            value
        })
        .collect();
    drop(bytes);
    assert_eq!(
        metadata[11]
            .as_array()
            .expect("sixteen child public inputs")
            .len(),
        DIGITS
    );
    std::array::from_fn(|child| {
        let expected: Vec<i8> = serde_json::from_value(metadata[11][child].clone()).expect("signed child public input");
        assert_eq!(expected.len(), 270);
        let public: Vec<u64> = expected
            .iter()
            .map(|&value| match value {
                -1 => MODULUS - 1,
                0 => 0,
                1 => 1,
                _ => panic!("child public digit range"),
            })
            .collect();
        assert_eq!(json!(public), prelude[2][2][child], "same indexed child public input");
        if bound < 1u64 << child {
            assert!(expected.iter().all(|&value| value == 0));
            for family in [1, 3, 4] {
                assert!(
                    is_zero(&prelude[2][family][child]),
                    "zero child commitment and evaluations"
                );
            }
            return Vec::new();
        }
        let values: Vec<u8> = native
            .par_iter()
            .map(|&value| {
                let bit = ((value.unsigned_abs() >> child) & 1) as i8;
                (if value < 0 { -bit } else { bit }) as u8
            })
            .collect();
        for (column, &value) in expected.iter().enumerate() {
            assert_eq!(values[column] as i8, value, "actual child public coefficient");
        }
        values
    })
}
