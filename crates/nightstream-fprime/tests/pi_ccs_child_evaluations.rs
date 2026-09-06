//! Independent CE evaluations for all sixteen actual folded digits.
//! One selected family shares its canonical row weights across the children.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use nightstream_fprime::load_per_application_package;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};

#[allow(dead_code)]
#[path = "support/pi_ccs_opening.rs"]
mod opening;
#[allow(dead_code)]
#[path = "per_application_logical_matrix_conformance/reference/mod.rs"]
mod reference;

use opening::{EqualityTensor, Extension, Ring, DEGREE};
use reference::{matrix::MatrixProgram, source::SourcePackage};

const CHILDREN: usize = 16;
const MATRICES: usize = 14;
const PUBLIC: usize = 270;
type Values = [Ring; CHILDREN];

#[derive(Deserialize)]
struct Inputs {
    package: PathBuf,
    structural_identity: [u64; 4],
    folded_cache: PathBuf,
    commitments: PathBuf,
    family_result: PathBuf,
    family: String,
    lean_result: Option<PathBuf>,
}

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("child evaluation input")).expect("child evaluation JSON")
}

fn ring(value: &Value) -> Ring {
    serde_json::from_value::<Vec<[u64; 2]>>(value.clone())
        .expect("ring words")
        .into_iter()
        .map(|words| Extension::checked(words).expect("canonical extension word"))
        .collect::<Vec<_>>()
        .try_into()
        .expect("54 coefficients")
}

fn check_values(actual: &Values, encoded: &Value) -> Result<(), String> {
    let values: Vec<Vec<[u64; 2]>> = serde_json::from_value(encoded.clone()).map_err(|error| error.to_string())?;
    if values.len() != CHILDREN {
        return Err("child count".into());
    }
    for (child, values) in values.iter().enumerate() {
        if values.len() != DEGREE {
            return Err(format!("child {child} coefficient count"));
        }
        for (coefficient, &words) in values.iter().enumerate() {
            if words
                .iter()
                .any(|&word| word >= reference::GOLDILOCKS_MODULUS)
            {
                return Err(format!("child {child} coefficient {coefficient} encoding"));
            }
            if words != actual[child][coefficient].words() {
                return Err(format!("child {child} coefficient {coefficient} value"));
            }
        }
    }
    Ok(())
}

fn check_output_mutations(actual: &Values, encoded: &Value) {
    let modulus = reference::GOLDILOCKS_MODULUS;
    let mut mutated = encoded.clone();
    for child in 0..CHILDREN {
        for coefficient in 0..DEGREE {
            for lane in 0..2 {
                let original = encoded[child][coefficient][lane].as_u64().unwrap();
                mutated[child][coefficient][lane] = json!((original + 1) % modulus);
                assert!(check_values(actual, &mutated).is_err(), "changed child evaluation");
                mutated[child][coefficient][lane] = json!(modulus);
                assert!(check_values(actual, &mutated).is_err(), "noncanonical child evaluation");
                mutated[child][coefficient][lane] = json!(original);
            }
        }
    }
    // Adjacent digit weights differ by two. These changes preserve their
    // weighted sum, so a recombination-only check would accept them.
    for child in 0..CHILDREN - 1 {
        let left = encoded[child][0][0].as_u64().unwrap();
        let right = encoded[child + 1][0][0].as_u64().unwrap();
        let changed_left = ((u128::from(left) + 2) % u128::from(modulus)) as u64;
        let changed_right = if right == 0 { modulus - 1 } else { right - 1 };
        assert_eq!(
            (u128::from(left) + 2 * u128::from(right)) % u128::from(modulus),
            (u128::from(changed_left) + 2 * u128::from(changed_right)) % u128::from(modulus)
        );
        mutated[child][0][0] = json!(changed_left);
        mutated[child + 1][0][0] = json!(changed_right);
        assert!(check_values(actual, &mutated).is_err(), "cancelling child evaluations");
        mutated[child][0][0] = json!(left);
        mutated[child + 1][0][0] = json!(right);
    }
    mutated.as_array_mut().unwrap().pop();
    assert!(check_values(actual, &mutated).is_err(), "missing child");
    let mut truncated = encoded.clone();
    for child in 0..CHILDREN {
        truncated[child].as_array_mut().unwrap().pop();
        assert!(check_values(actual, &truncated).is_err(), "missing child coefficient");
        truncated[child] = encoded[child].clone();
    }
    println!(
        "child_evaluation_mutations=passed changed_words={} noncanonical_words={} cancelling_pairs={} missing_children=1 truncated_rings={CHILDREN}",
        CHILDREN * DEGREE * 2,
        CHILDREN * DEGREE * 2,
        CHILDREN - 1
    );
}

fn merge(mut left: Box<Values>, right: Box<Values>) -> Box<Values> {
    for (left, right) in left.iter_mut().zip(right.iter()) {
        for (left, right) in left.iter_mut().zip(right) {
            *left += *right;
        }
    }
    left
}

fn add_children(values: &mut Values, weights: &Ring, parent: &[u8], active_digits: usize) {
    assert_eq!(parent.len(), 2 * DEGREE);
    let native: [i16; DEGREE] =
        std::array::from_fn(|lane| i16::from_le_bytes(parent[2 * lane..2 * lane + 2].try_into().unwrap()));
    if native.iter().all(|&value| value == 0) {
        return;
    }
    let transformed = opening::transform(weights);
    for child in 0..active_digits {
        let source: [u8; DEGREE] = std::array::from_fn(|lane| {
            let digit = (u32::from(native[lane].unsigned_abs()) / (1u32 << child)) % 2;
            if digit == 0 {
                0
            } else if native[lane] < 0 {
                255
            } else {
                1
            }
        });
        if source.iter().all(|&value| value == 0) {
            continue;
        }
        let product = opening::multiply_signed(&transformed, &source);
        for (total, value) in values[child].iter_mut().zip(product) {
            *total += value;
        }
    }
}

fn evaluate(
    bytes: &[u8],
    logical_width: usize,
    row_count: usize,
    parent: &[u8],
    weights: &EqualityTensor,
    matrix: Option<usize>,
    active_digits: usize,
) -> Box<Values> {
    match matrix {
        None => parent
            .par_chunks_exact(2 * DEGREE)
            .enumerate()
            .fold(
                || Box::new([[Extension::ZERO; DEGREE]; CHILDREN]),
                |mut total, (block, values)| {
                    let coefficients = std::array::from_fn(|lane| weights.at(block * DEGREE + lane));
                    add_children(&mut total, &coefficients, values, active_digits);
                    total
                },
            )
            .reduce(|| Box::new([[Extension::ZERO; DEGREE]; CHILDREN]), merge),
        Some(matrix) => {
            let artifact = SourcePackage::decode(bytes).expect("independent canonical Lean decoder");
            assert_eq!(
                (artifact.logical_rows, artifact.logical_columns, artifact.cube_variables),
                (row_count, logical_width, 28)
            );
            let program = MatrixProgram::decode(&artifact.matrix_program, &artifact.sources, logical_width, row_count)
                .expect("independent canonical matrix program");
            let workers = rayon::current_num_threads();
            let rows_per_worker = row_count.div_ceil(workers);
            (0..workers)
                .into_par_iter()
                .map(|worker| {
                    let mut blocks: Vec<Option<Box<Ring>>> = vec![None; parent.len() / (2 * DEGREE)];
                    let start = (worker * rows_per_worker).min(row_count);
                    let end = (start + rows_per_worker).min(row_count);
                    let mut covered = 0;
                    program
                        .visit_rows(start, end, &artifact.sources, |row, forms| {
                            assert_eq!(row, start + covered, "canonical row order");
                            let weight = weights.at(row);
                            for entry in forms[matrix].entries() {
                                let block = entry.column / DEGREE;
                                let lane = entry.column % DEGREE;
                                let coefficients =
                                    blocks[block].get_or_insert_with(|| Box::new([Extension::ZERO; DEGREE]));
                                coefficients[lane] += weight.scale(entry.coefficient);
                            }
                            covered += 1;
                            Ok(())
                        })
                        .expect("independent canonical matrix rows");
                    assert_eq!(covered, end - start);
                    let mut result = Box::new([[Extension::ZERO; DEGREE]; CHILDREN]);
                    for (block, coefficients) in blocks.into_iter().enumerate() {
                        if let Some(coefficients) = coefficients {
                            add_children(
                                &mut result,
                                &coefficients,
                                &parent[block * 2 * DEGREE..(block + 1) * 2 * DEGREE],
                                active_digits,
                            );
                        }
                    }
                    result
                })
                .reduce(|| Box::new([[Extension::ZERO; DEGREE]; CHILDREN]), merge)
        }
    }
}

#[test]
#[ignore = "requires external child-opening paths and family K or A0..A13 as JSON on stdin; run under the 300-second cap"]
fn independent_actual_child_evaluation_family() {
    let started = Instant::now();
    let inputs: Inputs = serde_json::from_reader(std::io::stdin().lock()).expect("child evaluation paths");
    let selected = if inputs.family == "K" {
        None
    } else {
        let matrix = inputs
            .family
            .strip_prefix('A')
            .expect("family K or A0..A13")
            .parse::<usize>()
            .expect("matrix index");
        assert!(matrix < MATRICES);
        Some(matrix)
    };
    let bytes = fs::read(&inputs.package).expect("canonical package");
    let package = load_per_application_package(&bytes, inputs.structural_identity).expect("selected package identity");
    let logical_width = package.logical_column_count();
    let row_count = package.row_count();
    let native_width = logical_width.div_ceil(DEGREE) * DEGREE;
    assert_eq!(
        (
            package.ccs_relation().cube_variables(),
            package.ccs_relation().matrix_sources().len()
        ),
        (28, MATRICES)
    );
    let context = package
        .production_verifier_binding()
        .expect("selected binding")
        .verifier_context()
        .digest();
    let meta = read(&inputs.folded_cache.join("folded.json"));
    assert_eq!(meta.as_array().expect("folded metadata").len(), 13);
    assert_eq!(meta[0], 1);
    assert_eq!(meta[1], json!(inputs.structural_identity));
    assert_eq!(meta[2], json!(context));
    assert_eq!(meta[3], json!(logical_width));
    assert_eq!(meta[4], json!(native_width));
    let bound = meta[8].as_u64().expect("actual parent bound");
    assert!(bound < 1 << CHILDREN);
    let active_digits = (u64::BITS - bound.leading_zeros()) as usize;
    let prior_point: Vec<[u64; 2]> = serde_json::from_value(meta[12].clone()).expect("input CE point");
    assert_eq!(prior_point.len(), 28);
    let phase = inputs.lean_result.as_ref().map(|path| read(path));
    let point: Vec<[u64; 2]> = if let Some(phase) = &phase {
        assert_eq!(phase[0], 1);
        assert_eq!(phase[1][0], 2);
        assert_eq!(phase[5][0], 1, "accepted Lean phase result");
        assert_eq!(phase[2], phase[1][6], "same complete running statement");
        assert_eq!(phase[2][0], json!(prior_point), "actual prior claim point");
        serde_json::from_value(phase[5][6].clone()).expect("verifier-derived output point")
    } else {
        prior_point.clone()
    };
    assert_eq!(point.len(), 28);
    let weights = EqualityTensor::new(
        &point
            .iter()
            .map(|&words| Extension::checked(words).expect("canonical point"))
            .collect::<Vec<_>>(),
    );
    let parent = fs::read(inputs.folded_cache.join("folded.i16")).expect("complete raw folded carrier");
    assert_eq!(parent.len(), native_width * 2);
    assert!(parent
        .par_chunks_exact(2)
        .all(|word| u64::from(i16::from_le_bytes(word.try_into().unwrap()).unsigned_abs()) <= bound));
    let family = read(&inputs.family_result);
    assert_eq!(family.as_array().expect("complete child family").len(), 9);
    assert_eq!(family[0], 1);
    assert_eq!(family[1], json!(inputs.structural_identity));
    assert_eq!(family[2], json!(context));
    assert_eq!(family[3], json!(point));
    assert_eq!(family[4], json!(u64::from(selected.is_some())));
    assert_eq!(family[5], json!(selected.unwrap_or(0)));
    for index in [6, 7, 8] {
        assert_eq!(
            family[index]
                .as_array()
                .expect("sixteen child values")
                .len(),
            CHILDREN
        );
    }
    let mut expected = Box::new([[Extension::ZERO; DEGREE]; CHILDREN]);
    for child in 0..CHILDREN {
        let commitment = read(&inputs.commitments.join(format!("child-{child}.json")));
        assert_eq!(
            commitment
                .as_array()
                .expect("child commitment record")
                .len(),
            7
        );
        assert_eq!(commitment[0], 1);
        assert_eq!(commitment[1], json!(inputs.structural_identity));
        assert_eq!(commitment[2], json!(context));
        assert_eq!(commitment[3], json!(child));
        assert_eq!(commitment[4], json!(prior_point));
        assert_eq!(family[6][child], commitment[5], "same public projection");
        assert_eq!(family[7][child], commitment[6], "same independently checked commitment");
        if let Some(phase) = &phase {
            assert_eq!(family[6][child], phase[2][2][child], "same phase public input");
            assert_eq!(family[7][child], phase[2][1][child], "same phase commitment");
            let claimed = match selected {
                None => &phase[1][4][child + 1],
                Some(matrix) => &phase[1][5][child + 1][matrix],
            };
            assert_eq!(&family[8][child], claimed, "same complete phase output family");
        }
        let public: Vec<u64> = serde_json::from_value(family[6][child].clone()).expect("public projection");
        assert_eq!(public.len(), PUBLIC);
        for (column, &word) in public.iter().enumerate() {
            let value = i16::from_le_bytes(parent[column * 2..column * 2 + 2].try_into().unwrap());
            let digit = (u64::from(value.unsigned_abs()) / (1u64 << child)) % 2;
            let canonical = if value < 0 && digit != 0 {
                reference::GOLDILOCKS_MODULUS - digit
            } else {
                digit
            };
            assert_eq!(word, canonical, "actual child public coordinate");
        }
        let commitments: Vec<u64> = serde_json::from_value(family[7][child].clone()).expect("commitment words");
        assert_eq!(commitments.len(), 22 * DEGREE);
        assert!(commitments
            .iter()
            .all(|&word| word < reference::GOLDILOCKS_MODULUS));
        expected[child] = ring(&family[8][child]);
        if selected == Some(MATRICES - 1) {
            assert_eq!(
                expected[child],
                [Extension::ZERO; DEGREE],
                "canonical zero matrix claim"
            );
        }
    }
    drop(package);
    println!(
        "independent_child_evaluation_family={} children={CHILDREN} input_time={:?}",
        inputs.family,
        started.elapsed()
    );
    println!(
        "child_evaluation_point_source={}",
        if phase.is_some() {
            "accepted_lean_output"
        } else {
            "prior_claim"
        }
    );
    let actual = evaluate(
        &bytes,
        logical_width,
        row_count,
        &parent,
        &weights,
        selected,
        active_digits,
    );
    check_values(&actual, &family[8]).expect("every independently evaluated child coefficient");
    check_output_mutations(&actual, &family[8]);
    println!("independent_child_evaluation_family={} passed children={CHILDREN} coefficients={} carrier={native_width} elapsed={:?}", inputs.family, CHILDREN * DEGREE, started.elapsed());
}
