//! The gate of `compression_layout` against the Lean bound of the
//! compression argument, on every claim set of the Poseidon2 hash chain:
//! both circuits of the fresh instance and each pair of slot depths up to
//! the light fold cap. Lean emits, for each set, its layout (`layoutNew`)
//! and the exact rational `runError (productionCounts …)`, which bounds one
//! interactive run (`Spec.Compression.Rust.params_run_soundness`) and the
//! density `error` of the state-restoration bound
//! (`Spec.Compression.Rust.error_le_runError`). `error_bits` must be its
//! `−log2`, and the gate must pass a security minimum exactly when the exact
//! value is at most `2^−minimum`.

use neo_compress::{Layout, Norm};
use nightstream_fprime::{PI_CCS_V1_2_MATRIX_COUNT, PUBLIC_WORDS};
use serde_json::Value;

use super::support::artifact;
use super::{prover, LIGHT_FOLD_CAP};

fn lean() -> Value {
    serde_json::from_slice(&artifact("nightstream-fprime-compression-parity-v1.json"))
        .expect("Lean compression parity file")
}

fn number(value: &Value) -> u64 {
    value.as_u64().expect("a natural number")
}

fn numbers(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(number)
        .collect()
}

/// `log2` of a decimal natural number. The parse rounds to the nearest
/// `f64`, so the error is below `2^-52` bits.
fn log2(decimal: &Value) -> f64 {
    decimal
        .as_str()
        .expect("a decimal string")
        .parse::<f64>()
        .expect("a decimal number")
        .log2()
}

/// The claims of a Lean layout: the blocks and the table of each cell.
fn claims(layout: &Value) -> Vec<(usize, Norm)> {
    layout["cells"]
        .as_array()
        .expect("cells")
        .iter()
        .map(|cells| {
            let norm = match &cells["table"] {
                Value::Null => Norm::Caller,
                table => Norm::Table {
                    bound: number(&table[0]),
                },
            };
            (number(&cells["blocks"]) as usize, norm)
        })
        .collect()
}

#[test]
fn error_bits_and_the_gate_match_the_lean_bound_on_each_production_claim_set() {
    let lean = lean();
    let mut prover = prover(LIGHT_FOLD_CAP);
    assert_eq!(
        numbers(&lean["blocks"]),
        [0, 1].map(|relation| prover.relations[relation].blocks() as u64),
        "Lean's inputs are the blocks of the light and the refresh relation"
    );
    let key_blocks = neo_compress::key_blocks(prover.key_blocks()).expect("key blocks");
    assert_eq!(
        number(&lean["key_blocks"]),
        key_blocks as u64,
        "the key of the compression"
    );
    assert_eq!(
        numbers(&lean["bounds"]),
        (0..=LIGHT_FOLD_CAP)
            .map(|depth| prover.accumulator_bound(depth))
            .collect::<Vec<_>>(),
        "Lean accumulatorBound is the bound of each depth"
    );
    assert_eq!(number(&lean["public_cells"]), PUBLIC_WORDS as u64);
    assert_eq!(number(&lean["matrices"]), PI_CCS_V1_2_MATRIX_COUNT as u64);

    let entries = lean["production"].as_array().expect("production entries");
    assert_eq!(
        entries.len(),
        2 * (LIGHT_FOLD_CAP as usize + 1).pow(2),
        "each circuit and pair of depths"
    );
    let mut closest = f64::INFINITY;
    for entry in entries {
        let pc = number(&entry["pc"]);
        let depths: [u64; 2] = numbers(&entry["depths"]).try_into().expect("two depths");
        let context = format!("pc {pc}, depths {depths:?}");
        prover.minimum_security_bits = 0;
        let layout = prover
            .compression_layout(key_blocks, pc, depths)
            .unwrap_or_else(|error| panic!("{context}: {error}"));
        assert_eq!(
            layout,
            Layout::new(&claims(&entry["layout"]), key_blocks).expect("Lean's claims"),
            "{context}: the claims of compression_layout are Lean's productionClaims"
        );

        let error = &entry["error"];
        let exact = log2(&error["denominator"]) - log2(&error["numerator"]);
        let bits = neo_compress::error_bits(&layout, PUBLIC_WORDS, PI_CCS_V1_2_MATRIX_COUNT);
        assert!(
            (bits - exact).abs() < 1e-9,
            "{context}: error_bits {bits} and Lean {exact}"
        );

        let minimum = u32::try_from(number(&error["bits"])).expect("bits");
        assert!(
            exact >= f64::from(minimum) && exact < f64::from(minimum + 1),
            "{context}: Lean's floor"
        );
        closest = closest.min(exact - f64::from(minimum));
        prover.minimum_security_bits = minimum;
        assert!(
            prover.compression_layout(key_blocks, pc, depths).is_ok(),
            "{context}: the gate passes 2^-{minimum}"
        );
        prover.minimum_security_bits = minimum + 1;
        assert!(
            prover.compression_layout(key_blocks, pc, depths).is_err(),
            "{context}: the gate refuses 2^-{}",
            minimum + 1
        );
    }
    println!("the exact error is at least {closest:.4} bits above its integer floor on each set");
}
