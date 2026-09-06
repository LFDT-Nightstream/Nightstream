//! Independent full opening values for the current positive PiCCS input.
//! Canonical Lean rows, raw signed-unit carrier, and verifier-derived point
//! are evaluated separately from the fixture generator.

use std::{fs, path::PathBuf, time::Instant};

use nightstream_fprime::load_per_application_package;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};

#[allow(dead_code)]
#[path = "per_application_logical_matrix_conformance/reference/mod.rs"]
mod reference;

#[path = "support/pi_ccs_opening.rs"]
mod opening;

#[path = "support/pi_ccs_commitment.rs"]
mod commitment;

#[path = "support/pi_ccs_opening_family.rs"]
mod opening_family;

use opening::{evaluate_block, EqualityTensor, Extension, Ring, DEGREE};
use reference::{matrix::MatrixProgram, source::SourcePackage};

#[test]
fn independent_phi81_product_and_dual_basis_cover_all_basis_pairs() {
    let u = Extension::checked([0, 1]).unwrap();
    assert_eq!((u * u).words(), [7, 0]);
    for left in 0..DEGREE {
        for right in 0..DEGREE {
            let mut basis = [Extension::ZERO; DEGREE];
            basis[left] = Extension::ONE;
            let mut source = [0u8; DEGREE];
            source[right] = 1;
            let mut expected = [Extension::ZERO; DEGREE];
            let power = (left + right) % 81;
            if power < DEGREE {
                expected[power] = Extension::ONE;
            } else {
                expected[power - 54] = -Extension::ONE;
                expected[power - 27] = -Extension::ONE;
            }
            assert_eq!(opening::multiply_signed(&basis, &source), expected);
            let product = evaluate_block(&basis, &source);
            assert_eq!(product[0], if left == right { Extension::ONE } else { Extension::ZERO });
            source[right] = 255;
            assert_eq!(
                opening::multiply_signed(&basis, &source),
                expected.map(std::ops::Neg::neg)
            );
            assert_eq!(evaluate_block(&basis, &source)[0], -product[0]);
        }
    }
}

#[test]
fn independent_tensor_preserves_every_point_coordinate_and_split_boundary() {
    let point: Vec<_> = (0..28)
        .map(|index| Extension::checked([index + 2, index + 11]).unwrap())
        .collect();
    let tensor = EqualityTensor::new(&point);
    let split = point.len() / 2;
    for row in
        (0..point.len())
            .map(|bit| 1usize << bit)
            .chain([0, (1 << split) - 1, 1 << split, (1 << point.len()) - 1])
    {
        let expected = point
            .iter()
            .enumerate()
            .fold(Extension::ONE, |product, (bit, &coordinate)| {
                product
                    * if row & (1 << bit) == 0 {
                        Extension::ONE + -coordinate
                    } else {
                        coordinate
                    }
            });
        assert_eq!(tensor.at(row), expected, "complete 28-variable weight at row {row}");
    }
}

#[derive(Deserialize)]
struct ExternalOpening {
    package: PathBuf,
    structural_identity: [u64; 4],
    cache: PathBuf,
    phase_input: PathBuf,
    lean_result: PathBuf,
    setup_fixture: Option<PathBuf>,
    family: String,
}

fn read_json(path: &PathBuf) -> Value {
    serde_json::from_slice(&fs::read(path).expect("opening input file")).expect("opening numeric JSON")
}

fn extension_array(value: &Value) -> Vec<Extension> {
    serde_json::from_value::<Vec<[u64; 2]>>(value.clone())
        .expect("extension array")
        .into_iter()
        .map(|words| Extension::checked(words).expect("canonical extension"))
        .collect()
}

fn running_signs(input: &Value) -> [i8; 16] {
    let fresh_commitment: Vec<u64> = serde_json::from_value(input[1].clone()).expect("fresh commitment");
    let fresh_public: Vec<u64> = serde_json::from_value(input[2].clone()).expect("fresh public input");
    assert_eq!((fresh_commitment.len(), fresh_public.len()), (22 * DEGREE, 270));
    assert!(fresh_commitment.iter().any(|&value| value != 0));
    let negative = |words: &[u64]| {
        words
            .iter()
            .map(|&word| (-reference::Field::checked(word, "canonical signed source word").unwrap()).canonical())
            .collect::<Vec<_>>()
    };
    let negative_commitment = negative(&fresh_commitment);
    let negative_public = negative(&fresh_public);
    let commitments: Vec<Vec<u64>> = serde_json::from_value(input[6][1].clone()).expect("running commitments");
    let public: Vec<Vec<u64>> = serde_json::from_value(input[6][2].clone()).expect("running public inputs");
    assert_eq!((commitments.len(), public.len()), (16, 16));
    std::array::from_fn(|source| {
        assert_eq!((commitments[source].len(), public[source].len()), (22 * DEGREE, 270));
        if commitments[source] == fresh_commitment {
            assert_eq!(public[source], fresh_public);
            1
        } else if commitments[source] == negative_commitment {
            assert_eq!(public[source], negative_public);
            -1
        } else {
            assert!(commitments[source].iter().all(|&value| value == 0));
            assert!(public[source].iter().all(|&value| value == 0));
            0
        }
    })
}

fn signed_ring(value: &Ring, sign: i8) -> Ring {
    match sign {
        1 => *value,
        -1 => value.map(std::ops::Neg::neg),
        0 => [Extension::ZERO; DEGREE],
        _ => unreachable!("checked source sign"),
    }
}

#[test]
#[ignore = "requires external candidate paths and one family as JSON on stdin; run under the 300-second cap"]
fn external_positive_opening_family() {
    check_positive_opening(OpeningSources::SignedRunning);
}

#[test]
#[ignore = "checks the fresh source only; running inputs and outputs require their separate child-opening gates"]
fn external_positive_fresh_opening_family() {
    check_positive_opening(OpeningSources::Fresh);
}

#[test]
#[ignore = "requires the selected package and valid nonzero raw opening paths on stdin; run under the 300-second cap"]
fn external_rows_reject_detached_pi_rlc_product_blocks() {
    check_positive_opening(OpeningSources::DetachedProducts);
}

enum OpeningSources {
    Fresh,
    SignedRunning,
    DetachedProducts,
}

fn check_positive_opening(sources: OpeningSources) {
    let started = Instant::now();
    let detached_products = matches!(sources, OpeningSources::DetachedProducts);
    let paths: ExternalOpening = serde_json::from_reader(std::io::stdin().lock()).expect("external opening paths");
    let check_ccs = paths.family == "CCS";
    assert!(
        !detached_products || check_ccs,
        "detached-product regression requires the raw CCS row gate"
    );
    let check_commitment = paths.family == "COMMITMENT";
    let commitment_row = if check_ccs || check_commitment {
        None
    } else {
        paths
            .family
            .strip_prefix('C')
            .map(|row| row.parse::<usize>().expect("commitment row"))
    };
    let selected = if paths.family == "K" || check_ccs || check_commitment || commitment_row.is_some() {
        None
    } else {
        let matrix: usize = paths
            .family
            .strip_prefix('A')
            .expect("family K or A0..A13")
            .parse()
            .expect("matrix index");
        assert!(matrix < 14);
        Some(matrix)
    };
    let bytes = fs::read(&paths.package).expect("Lean canonical package");
    let package = load_per_application_package(&bytes, paths.structural_identity).expect("selected candidate identity");
    let logical_width = package.logical_column_count();
    let row_count = package.row_count();
    let context = package
        .production_verifier_binding()
        .expect("selected binding")
        .verifier_context()
        .digest();
    assert_eq!(package.ccs_relation().cube_variables(), 28);
    let metadata = read_json(&paths.cache.join("opening.json"));
    let input = read_json(&paths.phase_input);
    let lean = read_json(&paths.lean_result);
    assert_eq!(metadata[0], 1);
    assert_eq!(metadata[1], json!(paths.structural_identity));
    assert_eq!(metadata[2], json!(context));
    assert_eq!(metadata[3], json!(logical_width));
    assert_eq!(metadata[5], json!(row_count));
    assert_eq!(metadata[6], 14);
    assert_eq!(metadata[7], 28);
    assert_eq!(metadata[9], input[2], "same public projection");
    assert_eq!(
        metadata[10], input[1],
        "same commitment claim; commitment evaluation is a separate gate"
    );
    assert_eq!(input[0], 2);
    let signs = match sources {
        OpeningSources::Fresh | OpeningSources::DetachedProducts => None,
        OpeningSources::SignedRunning => Some(running_signs(&input)),
    };
    assert_eq!(lean[0], 1);
    assert_eq!(lean[1], input, "same serialized positive proof and input");
    assert_eq!(lean[5][0], 1, "Lean positive acceptance");
    let point = extension_array(&lean[5][6]);
    let weights = EqualityTensor::new(&point);
    let expected: Ring = extension_array(match selected {
        None => &input[4][0],
        Some(matrix) => &input[5][0][matrix],
    })
    .try_into()
    .expect("54 claimed output coefficients");
    let mut carrier = fs::read(paths.cache.join("carrier.i8")).expect("raw opening carrier");
    assert_eq!(carrier.len(), logical_width.div_ceil(DEGREE) * DEGREE);
    assert_eq!(metadata[4], json!(carrier.len()));
    assert!(
        carrier.iter().all(|value| matches!(value, 0 | 1 | 255)),
        "strict b=2 bound"
    );
    assert!(
        carrier[logical_width..].iter().all(|&value| value == 0),
        "carrier alignment"
    );
    let public: Vec<u64> = serde_json::from_value(input[2].clone()).expect("public input");
    assert_eq!(public.len(), 270);
    for (index, &value) in public.iter().enumerate() {
        assert!(value <= 1);
        assert_eq!(u64::from(carrier[index]), value, "carrier public projection");
    }
    if detached_products {
        zero_pi_rlc_product_blocks(&bytes, &mut carrier);
    }
    assert_eq!(input[4].as_array().expect("17 Pad families").len(), 17);
    assert_eq!(input[5].as_array().expect("17 matrix families").len(), 17);
    if commitment_row.is_some() || check_commitment {
        let first_row = commitment_row.unwrap_or(0);
        let end_row = commitment_row.map_or(22, |row| row + 1);
        let setup = read_json(
            paths
                .setup_fixture
                .as_ref()
                .expect("current Lean setup fixture"),
        );
        assert_eq!(setup[0], 3, "wide256 setup schema");
        let authority: Vec<u64> = serde_json::from_value(setup[6].clone()).expect("complete Lean setup authority");
        assert_eq!(authority.len(), 73);
        assert_eq!(authority[0], 37);
        assert_eq!(
            &authority[1..38],
            b"nightstream-ajtai-chacha20-wide256-v1"
                .iter()
                .map(|&byte| u64::from(byte))
                .collect::<Vec<_>>()
        );
        assert_eq!(&authority[38..41], &[22, (carrier.len() / DEGREE) as u64, 32]);
        assert!(first_row < end_row && end_row <= authority[38] as usize);
        let binding = package
            .production_verifier_binding()
            .expect("selected key binding");
        assert_eq!(binding.verifier_context().commitment_key_words(), authority);
        let seed: [u8; 32] = authority[41..]
            .iter()
            .map(|&value| u8::try_from(value).expect("canonical seed byte"))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        assert_eq!(setup[4], json!(seed));
        let test_seed: [u8; 32] = serde_json::from_value(setup[2].clone()).expect("Lean RFC test seed");
        let expected_block: [u32; 16] = serde_json::from_value(setup[3].clone()).expect("Lean RFC block words");
        assert_eq!(
            commitment::block_words(&test_seed, 0x09000000, 0x4a000000, 1),
            expected_block
        );
        let samples: Vec<[u64; 4]> =
            serde_json::from_value(setup[5].clone()).expect("Lean indexed coefficient vectors");
        for [sample_row, block, lane, expected] in samples {
            assert_eq!(
                commitment::coefficient(&seed, sample_row.try_into().unwrap(), block, lane.try_into().unwrap()),
                expected
            );
        }
        let claimed: Vec<u64> = serde_json::from_value(input[1].clone()).expect("complete claimed commitment");
        assert_eq!(claimed.len(), 22 * DEGREE);
        assert!(claimed
            .iter()
            .all(|&word| word < reference::GOLDILOCKS_MODULUS));
        drop(package);
        println!(
            "independent_commitment_rows={first_row}..{end_row} decoded={:?}",
            started.elapsed()
        );
        for row in first_row..end_row {
            let actual = commitment::commitment_row(&seed, row as u32, &carrier);
            assert_eq!(
                actual.as_slice(),
                &claimed[row * DEGREE..(row + 1) * DEGREE],
                "complete selected-key commitment row {row}"
            );
        }
        println!(
            "independent_commitment_rows={first_row}..{end_row} coefficients={} carrier={} elapsed={:?}",
            (end_row - first_row) * DEGREE,
            carrier.len(),
            started.elapsed()
        );
        return;
    }
    drop(package);
    println!(
        "independent_opening_family={} decoded={:?}",
        paths.family,
        started.elapsed()
    );
    if check_ccs {
        let artifact = SourcePackage::decode(&bytes).expect("independent canonical Lean decoder");
        assert_eq!(
            (artifact.logical_rows, artifact.logical_columns, artifact.cube_variables),
            (row_count, logical_width, 28)
        );
        let program = MatrixProgram::decode(&artifact.matrix_program, &artifact.sources, logical_width, row_count)
            .expect("independent canonical matrix program");
        let relation = reference::relation::Relation::decode(&bytes).expect("independent exact CCS polynomial");
        let workers = rayon::current_num_threads();
        let rows_per_worker = row_count.div_ceil(workers);
        let results: Vec<_> = (0..workers)
            .into_par_iter()
            .map(|worker| {
                let start = (worker * rows_per_worker).min(row_count);
                let end = (start + rows_per_worker).min(row_count);
                reference::evaluation::verify_satisfaction_range_with(
                    &program,
                    &artifact.sources,
                    &relation,
                    start,
                    end,
                    |column| match carrier.get(column) {
                        Some(0) => Ok(reference::Field::ZERO),
                        Some(1) => Ok(reference::Field::ONE),
                        Some(255) => Ok(-reference::Field::ONE),
                        _ => Err(format!("invalid raw opening coordinate {column}")),
                    },
                )
            })
            .collect();
        if detached_products {
            assert!(
                results.iter().any(Result::is_err),
                "canonical rows accepted detached PiRLC product inputs, intermediates and outputs"
            );
            println!(
                "independent_detached_pi_rlc_products=rejected elapsed={:?}",
                started.elapsed()
            );
            return;
        }
        let checked: usize = results
            .into_iter()
            .map(|result| result.expect("every independent CCS row accepts the raw opening"))
            .sum();
        assert_eq!(checked, row_count);
        assert_eq!(relation.evaluate(&[reference::Field::ZERO; 14]), reference::Field::ZERO);
        for row in [row_count, (1 << 28) - 1, 1 << 28] {
            assert!(
                program.row(row, &artifact.sources).is_err(),
                "implicit zero padding is not an active row"
            );
        }
        println!(
            "independent_opening_CCS=passed active_rows={checked} carrier={} padding={} elapsed={:?}",
            carrier.len(),
            (1 << 28) - row_count,
            started.elapsed()
        );
        return;
    }
    let actual = opening_family::evaluate_family(&bytes, logical_width, row_count, &carrier, &weights, selected);
    assert_eq!(
        actual, expected,
        "all 54 independent opening coefficients for {}",
        paths.family
    );
    if selected == Some(13) {
        assert_eq!(actual, [Extension::ZERO; DEGREE]);
    }
    if let Some(signs) = signs {
        let prior = if signs.iter().any(|&sign| sign != 0) {
            let prior_point = extension_array(&input[6][0]);
            let prior_weights = EqualityTensor::new(&prior_point);
            Some(opening_family::evaluate_family(
                &bytes,
                logical_width,
                row_count,
                &carrier,
                &prior_weights,
                selected,
            ))
        } else {
            None
        };
        for (source, &sign) in signs.iter().enumerate() {
            let claimed_output = extension_array(match selected {
                None => &input[4][source + 1],
                Some(matrix) => &input[5][source + 1][matrix],
            });
            assert_eq!(
                claimed_output.as_slice(),
                signed_ring(&actual, sign).as_slice(),
                "running output source {source} has the actual signed opening"
            );
            let claimed_input = extension_array(match selected {
                None => &input[6][3][source],
                Some(matrix) => &input[6][4][source][matrix],
            });
            let expected_input = if sign == 0 {
                [Extension::ZERO; DEGREE]
            } else {
                signed_ring(prior.as_ref().expect("independent prior-point evaluation"), sign)
            };
            assert_eq!(
                claimed_input.as_slice(),
                expected_input.as_slice(),
                "running input source {source} has the actual signed opening at the prior point"
            );
        }
        println!(
            "independent_running_opening_sources=16 input_and_output_family={}",
            paths.family
        );
    } else {
        println!("independent_opening_source_scope=fresh_only");
    }
    println!(
        "independent_opening_family={} coefficients={} carrier={} elapsed={:?}",
        paths.family,
        DEGREE,
        carrier.len(),
        started.elapsed()
    );
}

// Read the Lean assignment plan directly. The production assignment encoder
// is not used to mutate or validate this arbitrary signed-unit assignment.
fn zero_pi_rlc_product_blocks(bytes: &[u8], carrier: &mut [u8]) {
    let package: Value = serde_json::from_slice(bytes).expect("raw Lean package");
    let mut offset = package[6].as_u64().expect("logical public width") as usize;
    let blocks = package[4][1].as_array().expect("Lean assignment blocks");
    let mut changed = 0;
    for (index, block) in blocks.iter().enumerate() {
        assert_eq!(block[0], json!(index), "canonical block order");
        let width = match block[1].as_u64().expect("slot kind") {
            0 | 1 => 1,
            2 => 41,
            _ => panic!("unsupported slot kind"),
        };
        let end = offset + width * block[2].as_u64().expect("slot count") as usize;
        // PerApplicationAssignmentPlan: productGroup, productInput, productOutput.
        if matches!(index, 3 | 9 | 10) {
            assert_eq!(width, 41, "field-encoded PiRLC product block");
            let values = &mut carrier[offset..end];
            let block_changes = values.iter().filter(|&&value| value != 0).count();
            changed += block_changes;
            values.fill(0);
            println!("detached_pi_rlc_block={index} start={offset} end={end} changed={block_changes}");
        }
        offset = end;
    }
    assert!(changed > 0, "the fixture must exercise nonzero PiRLC product values");
    assert!(
        carrier[offset..].iter().all(|&value| value == 0),
        "zero carrier alignment"
    );
}
