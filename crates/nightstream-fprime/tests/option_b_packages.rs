//! The option-B step packages expand into the plan rows that Lean describes:
//! the relation rows match Lean's parity file, the S-box and binding rows
//! compute Rust's native Poseidon2 permutation, and the identity uses Lean's
//! identity encoding.

use std::fs;
use std::path::PathBuf;

use neo_ccs::crypto::poseidon2_goldilocks::permute_state;
use nightstream_fprime::{
    load_option_b_packages, option_b_identity, parse_atoms, OptionBPackages, PackageError, OPTION_B_PACKAGES_IDENTITY,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as G;
use serde_json::Value;

fn artifact(name: &str) -> Vec<u8> {
    fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts")
            .join(name),
    )
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

/// Lean `Parity.sharedWord`.
fn shared_word(column: usize) -> G {
    if column == 0 {
        return G::ONE;
    }
    let q = u128::from(G::ORDER_U64);
    G::from_u64(((column as u128 * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) % q) as u64)
}

fn shared_columns(summary: &Value) -> Vec<G> {
    (0..words(summary)[3] as usize).map(shared_word).collect()
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
        let [ordinary, sbox, checksum, _, _, _] = words(summary)[..] else {
            panic!("a summary has six words");
        };
        let rows = package.plan_rows().expect("plan rows");
        let circuit = package.circuit();
        assert_eq!(
            rows.ordinary_row_count() as u64,
            ordinary,
            "relation rows of circuit {circuit}"
        );
        assert_eq!((rows.row_count() - rows.ordinary_row_count()) as u64, sbox);
        assert_eq!(rows.row_count(), logical_rows, "census rows of circuit {circuit}");
        assert_eq!(
            rows.ordinary_checksum(),
            checksum,
            "relation-row checksum of circuit {circuit}"
        );
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
    let mut assignment: Vec<G> = (0..width)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            G::from_u64(state >> 1)
        })
        .collect();
    assignment[0] = G::ONE;
    for block in light.blocks() {
        let lanes: [G; 16] = block.lane_values(&assignment).try_into().expect("16 lanes");
        assignment[block.outputs()..block.outputs() + 16].copy_from_slice(&permute_state(lanes));
    }
    for row in rows.ordinary_row_count()..rows.row_count() {
        let [(column, coefficient)] = rows.port(row, 2).collect::<Vec<_>>()[..] else {
            panic!("an S-box row's C port is one S-box column");
        };
        assert_eq!(coefficient, G::ONE);
        assignment[column] = rows.evaluate(row, 3, &assignment).exp_u64(7);
    }
    for row in rows.binding_rows().start..rows.row_count() {
        assert_eq!(rows.residual(row, &assignment).as_canonical_u64(), 0, "row {row}");
    }
}

/// Rust's witness program and native ranges give the columns that Lean's
/// recipes give, on the same shared columns.
#[test]
fn circuit_columns_match_the_lean_witness() {
    let fields = parity();
    let packages = packages();
    for (package, summary) in [(packages.light(), &fields[1]), (packages.refresh(), &fields[2])] {
        let columns = package
            .circuit_columns(&shared_columns(summary))
            .expect("circuit columns");
        assert_eq!(
            package.witness_checksum(&columns),
            words(summary)[4],
            "witness of circuit {}",
            package.circuit()
        );
    }
}

/// The fresh columns make every ring, binding and S-box row hold, whatever
/// the shared columns are.
#[test]
fn fresh_columns_satisfy_the_rows_that_define_them() {
    let fields = parity();
    let packages = packages();
    for (package, summary) in [(packages.light(), &fields[1]), (packages.refresh(), &fields[2])] {
        let rows = package.plan_rows().expect("plan rows");
        let columns = package
            .circuit_columns(&shared_columns(summary))
            .expect("circuit columns");
        let columns = package
            .relation_columns(columns, &rows)
            .expect("relation columns");
        let defining = rows
            .ring_rows()
            .iter()
            .cloned()
            .chain([rows.binding_rows(), rows.sbox_rows()]);
        for row in defining.flatten() {
            assert_eq!(
                rows.residual(row, &columns).as_canonical_u64(),
                0,
                "circuit {} row {row}",
                package.circuit()
            );
        }
    }
}

/// Rust's logical layout and digits give the logical assignment that Lean's
/// slot layout gives; every digit is in {-1, 0, 1} and every slot
/// recomposes to its source value.
#[test]
fn logical_assignment_matches_the_lean_layout() {
    let fields = parity();
    let packages = packages();
    for (package, summary) in [(packages.light(), &fields[1]), (packages.refresh(), &fields[2])] {
        let layout = package.layout().expect("layout");
        let rows = package.plan_rows().expect("plan rows");
        let columns = package
            .circuit_columns(&shared_columns(summary))
            .expect("circuit columns");
        let columns = package
            .relation_columns(columns, &rows)
            .expect("relation columns");
        let logical = layout.assignment(&columns).expect("logical assignment");
        assert_eq!(
            layout.checksum(&logical, package.first_fresh()),
            words(summary)[5],
            "logical assignment of circuit {}",
            package.circuit()
        );
        let unit = |value: &G| matches!(value.as_canonical_u64(), 0 | 1) || *value == -G::ONE;
        assert!(logical[270..].iter().all(unit), "a digit outside {{-1, 0, 1}}");
        for (column, value) in columns.iter().enumerate().skip(270) {
            if let Some((offset, kind)) = layout.slot(column) {
                let recomposed = (0..kind.width())
                    .rev()
                    .fold(G::ZERO, |sum, digit| sum * G::from_u64(3) + logical[offset + digit]);
                assert_eq!(recomposed, *value, "column {column} recomposes");
            }
        }
        println!("circuit {}: {} logical columns", package.circuit(), layout.width());
    }
}
