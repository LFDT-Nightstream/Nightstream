//! The gate of `compression_layout` against the Lean bound of the
//! compression argument, on every claim set of the Poseidon2 hash chain:
//! both circuits of the fresh instance and each pair of slot depths up to
//! the light fold cap, and for the light compression each depth of slot 0
//! under the key of the light relation. Lean emits, for each set, its
//! layout (`layoutNew`),
//! the count `runCount (productionCounts …)` over `q²` and `runBits`, the
//! largest `k` with `2^k · runCount ≤ q²`. One interactive run is at most
//! `2^−runBits` (`Spec.Compression.Rust.params_run_soundness_bits`).
//! `error_bits` must be `runBits`, and the gate must pass a security minimum
//! exactly when it is at most `runBits`.

use nightstream_fprime::{PI_CCS_V1_2_MATRIX_COUNT, PUBLIC_WORDS};
use ns_compress::{Layout, Norm};
use serde_json::Value;

use super::super::{BoundedDepth, Compression};
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
    let key_blocks = ns_compress::key_blocks(prover.key_blocks()).expect("key blocks");
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
    let mut counts = (u64::MAX, 0);
    for entry in entries {
        let pc = number(&entry["pc"]);
        let depths: [u64; 2] = numbers(&entry["depths"]).try_into().expect("two depths");
        let count = check_entry(&mut prover, key_blocks, pc, &depths, entry, "productionClaims");
        counts = (counts.0.min(count), counts.1.max(count));
    }
    println!(
        "the count over q² is between {} and {} on these sets",
        counts.0, counts.1
    );
}

/// The light compression of the hash chain: the key of the light relation
/// alone, and slot 0 at each depth with the fresh instance of the light
/// circuit (Lean `lightClaims`, `layoutNew_light`).
#[test]
fn error_bits_and_the_gate_match_the_lean_bound_on_each_light_claim_set() {
    let lean = lean();
    let mut prover = prover(LIGHT_FOLD_CAP).with_compression(Compression::Light);
    let key_blocks = ns_compress::key_blocks(prover.key_blocks()).expect("key blocks");
    assert_eq!(
        number(&lean["light_key_blocks"]),
        key_blocks as u64,
        "the key of the light relation"
    );
    let entries = lean["light"].as_array().expect("light entries");
    assert_eq!(entries.len(), LIGHT_FOLD_CAP as usize + 1, "each depth");
    for entry in entries {
        let depth = number(&entry["depth"]);
        check_entry(&mut prover, key_blocks, 1, &[depth], entry, "lightClaims");
    }
}

/// `compression_layout` of the claims of `pc` and `depths` is Lean's layout
/// of `entry`, `error_bits` of it is Lean `runBits`, and the gate passes a
/// security minimum exactly when it is at most `runBits`. The count over
/// `q²` of Lean's bound.
fn check_entry(
    prover: &mut BoundedDepth,
    key_blocks: usize,
    pc: u64,
    depths: &[u64],
    entry: &Value,
    lean_claims: &str,
) -> u64 {
    let context = format!("pc {pc}, depths {depths:?}");
    prover.minimum_security_bits = 0;
    let layout = prover
        .compression_layout(key_blocks, pc, depths)
        .unwrap_or_else(|error| panic!("{context}: {error}"));
    assert_eq!(
        layout,
        Layout::new(&claims(&entry["layout"]), key_blocks).expect("Lean's claims"),
        "{context}: the claims of compression_layout are Lean's {lean_claims}"
    );
    let error = &entry["error"];
    let minimum = u32::try_from(number(&error["bits"])).expect("bits");
    assert_eq!(
        ns_compress::error_bits(&layout, PUBLIC_WORDS, PI_CCS_V1_2_MATRIX_COUNT),
        minimum,
        "{context}: error_bits is Lean runBits"
    );
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
    number(&error["count"])
}
