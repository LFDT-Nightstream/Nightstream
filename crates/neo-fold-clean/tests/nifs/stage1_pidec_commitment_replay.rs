//! Compare complete Rust child commitments with complete independent Lean values.

use std::{fs::File, io::BufReader, path::Path, time::Instant};

use neo_ajtai::{nightstream_fprime_setup::PRODUCTION_MESSAGE_COLUMNS, Commitment};
use neo_math::{D, F};
use nightstream_fprime::PI_DEC_V1_1_CHILD_COUNT as CHILDREN;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::Deserialize;

#[derive(Deserialize)]
struct SavedCommitments {
    schema: u64,
    commitments: [Commitment; CHILDREN],
}

type LeanCommitments = (u64, usize, usize, usize, Vec<Vec<Vec<u64>>>);

fn compare_values(actual: &[Commitment; CHILDREN], expected: &[Commitment; CHILDREN]) -> Result<(), String> {
    for child in 0..CHILDREN {
        if actual[child].d != D
            || actual[child].kappa != expected[child].kappa
            || actual[child].data.len() != expected[child].data.len()
        {
            return Err(format!("commitment shape mismatch at child {child}"));
        }
        for (coordinate, (actual, expected)) in actual[child]
            .data
            .iter()
            .zip(&expected[child].data)
            .enumerate()
        {
            if actual.as_canonical_u64() != expected.as_canonical_u64() {
                return Err(format!(
                    "commitment mismatch at child {child}, row {}, lane {}",
                    coordinate / D,
                    coordinate % D
                ));
            }
        }
    }
    Ok(())
}

/// Lean emits [1,blocks,0,blocks,rows[children[54 coefficients]]] after it
/// sums every contiguous range. Rust only decodes and compares those complete
/// values; it does not perform the expected commitment arithmetic.
pub fn compare(split_path: &Path, lean_path: &Path) {
    let started = Instant::now();
    let actual: SavedCommitments =
        serde_json::from_reader(BufReader::new(File::open(split_path).expect("Rust split file")))
            .expect("Rust split commitments");
    assert_eq!(actual.schema, 1);
    let rows = neo_ajtai::nightstream_fprime_setup::PRODUCTION_VERIFIER_ROWS as usize;
    let blocks = PRODUCTION_MESSAGE_COLUMNS as usize;
    let mut expected = std::array::from_fn(|_| Commitment::zeros(D, rows));
    let (schema, count, start, end, values): LeanCommitments =
        serde_json::from_reader(BufReader::new(File::open(lean_path).expect("Lean commitments")))
            .expect("complete Lean commitments");
    assert_eq!(schema, 1);
    assert_eq!(count, blocks);
    assert_eq!(
        (start, end),
        (0, blocks),
        "complete commitment coverage including tails"
    );
    assert_eq!(values.len(), rows, "all key rows");
    for (row, children) in values.iter().enumerate() {
        assert_eq!(children.len(), CHILDREN, "all child commitments");
        for (child, coefficients) in children.iter().enumerate() {
            assert_eq!(coefficients.len(), D, "all ring coefficients");
            for (lane, &coefficient) in coefficients.iter().enumerate() {
                assert!(coefficient < F::ORDER_U64, "canonical Lean field coefficient");
                expected[child].data[row * D + lane] = F::from_u64(coefficient);
            }
        }
    }
    compare_values(&actual.commitments, &expected).expect("all independent child commitments match");

    let child = CHILDREN - 1;
    let row = rows - 1;
    let lane = D - 1;
    let mut changed = actual.commitments;
    changed[child].data[row * D + lane] += F::ONE;
    assert_eq!(
        compare_values(&changed, &expected).unwrap_err(),
        format!("commitment mismatch at child {child}, row {row}, lane {lane}")
    );
    println!(
        "pidec_commitment_replay=passed children={CHILDREN} rows={rows} coefficients={} blocks={blocks} target_mutation=rejected child={child} row={row} lane={lane} elapsed={:?}",
        CHILDREN * rows * D,
        started.elapsed()
    );
}
