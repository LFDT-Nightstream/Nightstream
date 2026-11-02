//! The option-B step packages expand into the plan rows that Lean describes:
//! the relation rows match Lean's parity file, the S-box and binding rows
//! compute Rust's native Poseidon2 permutation, and the identity uses Lean's
//! identity encoding.

use std::fs;
use std::path::PathBuf;

use neo_ccs::crypto::poseidon2_goldilocks::permute_state;
use nightstream_fprime::{
    load_option_b_packages, option_b_identity, parse_atoms, OptionBPackages, PackageError, PlanRows,
    OPTION_B_PACKAGES_IDENTITY,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde_json::Value;

fn artifact(name: &str) -> Vec<u8> {
    fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../formal/nightstream-fprime/artifacts").join(name))
        .unwrap_or_else(|error| panic!("read {name}: {error}"))
}

fn parity() -> Vec<Value> {
    match parse_atoms(&artifact("nightstream-fprime-option-b-parity-v1.json")).expect("parity file") {
        Value::Array(fields) if fields.len() == 5 => fields,
        _ => panic!("the parity file has five fields"),
    }
}

fn packages() -> OptionBPackages {
    load_option_b_packages(
        &artifact("nightstream-fprime-option-b-light-v1.json"),
        &artifact("nightstream-fprime-option-b-refresh-v1.json"),
        OPTION_B_PACKAGES_IDENTITY,
    )
    .expect("option-B packages")
}

fn words(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|word| word.as_u64().expect("word"))
        .collect()
}

fn evaluate(rows: &PlanRows, row: usize, port: usize, assignment: &[Goldilocks]) -> Goldilocks {
    rows.port(row, port)
        .fold(Goldilocks::ZERO, |sum, (column, coefficient)| sum + coefficient * assignment[column])
}

#[test]
fn identity_code_matches_lean_on_the_probe_packages() {
    let fields = parity();
    let expected: [u64; 4] = words(&fields[4]).try_into().expect("four words");
    assert_eq!(option_b_identity(&fields[3]).expect("probe identity"), expected);
}

#[test]
fn relation_rows_match_the_lean_checksum() {
    let fields = parity();
    let packages = packages();
    for (package, summary, logical_rows) in [
        (packages.light(), &fields[1], 184_830),
        (packages.refresh(), &fields[2], 796_563),
    ] {
        let [ordinary, sbox, checksum] = words(summary)[..] else {
            panic!("a summary has three words");
        };
        let rows = package.plan_rows().expect("plan rows");
        let circuit = package.circuit();
        assert_eq!(rows.ordinary_row_count() as u64, ordinary, "relation rows of circuit {circuit}");
        assert_eq!((rows.row_count() - rows.ordinary_row_count()) as u64, sbox);
        assert_eq!(rows.row_count(), logical_rows, "census rows of circuit {circuit}");
        assert_eq!(rows.ordinary_checksum(), checksum, "relation-row checksum of circuit {circuit}");
    }
}

#[test]
fn loader_rejects_another_identity() {
    let mut other = OPTION_B_PACKAGES_IDENTITY;
    other[0] ^= 1;
    let loaded = load_option_b_packages(
        &artifact("nightstream-fprime-option-b-light-v1.json"),
        &artifact("nightstream-fprime-option-b-refresh-v1.json"),
        other,
    );
    assert!(matches!(loaded, Err(PackageError::IdentityMismatch { .. })));
}

/// Each block's output locals take the native permutation of its lanes, block
/// by block; the S-box rows fill the S-box columns; every binding row and every
/// S-box row then holds.
#[test]
fn block_rows_compute_the_native_permutation() {
    let packages = packages();
    let light = packages.light();
    let rows = light.plan_rows().expect("plan rows");
    let width = light.sbox_start() + 150 * light.block_count();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut assignment: Vec<Goldilocks> = (0..width)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            Goldilocks::from_u64(state >> 1)
        })
        .collect();
    assignment[0] = Goldilocks::ONE;
    for block in light.blocks() {
        let lanes: [Goldilocks; 16] = block.lane_values(&assignment).try_into().expect("16 lanes");
        assignment[block.outputs()..block.outputs() + 16].copy_from_slice(&permute_state(lanes));
    }
    for row in rows.ordinary_row_count()..rows.row_count() {
        let [(column, coefficient)] = rows.port(row, 2).collect::<Vec<_>>()[..] else {
            panic!("an S-box row's C port is one S-box column");
        };
        assert_eq!(coefficient, Goldilocks::ONE);
        assignment[column] = evaluate(&rows, row, 3, &assignment).exp_u64(7);
    }
    let binding = rows.ordinary_row_count() - 16 * light.block_count();
    for row in binding..rows.row_count() {
        let residual = evaluate(&rows, row, 0, &assignment) * evaluate(&rows, row, 1, &assignment)
            - evaluate(&rows, row, 2, &assignment)
            + evaluate(&rows, row, 3, &assignment).exp_u64(7);
        assert_eq!(residual.as_canonical_u64(), 0, "row {row}");
    }
}
