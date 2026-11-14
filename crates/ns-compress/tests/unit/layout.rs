//! `Layout::new` and `error_bits` against Lean `layoutNew` and `runBits`
//! (`Spec.Compression.Rust`), on each claim set of the Poseidon2 hash chain
//! and on generated claim sets. Lean proves that a layout of `layoutNew`
//! satisfies the premises of the soundness theorem
//! (`Spec.Compression.Rust.layoutNew_production`), and that the exact error
//! sum is at most `2^-runBits` (`Spec.Compression.Rust.runError_le_runBits`).

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use super::{Cells, Layout, Slice, LANE_VARS, LOOKUP_VARS};
use crate::akita::SCHEDULES;
use crate::key::ROW_VARS;
use crate::{error_bits, Error, Norm};

/// The production public cells and matrices, which Lean's values use.
const PUBLIC_CELLS: usize = 320;
const MATRICES: usize = 4;

fn lean() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-compression-parity-v1.json");
    serde_json::from_slice(&fs::read(path).expect("Lean compression parity file")).expect("JSON")
}

fn number(value: &Value) -> usize {
    value.as_u64().expect("a natural number") as usize
}

fn slice(value: &Value) -> Slice {
    Slice {
        offset: number(&value[0]),
        vars: number(&value[1]),
    }
}

fn layout(value: &Value) -> Layout {
    let cells = value["cells"]
        .as_array()
        .expect("cells")
        .iter()
        .map(|cells| Cells {
            witness: slice(&cells["witness"]),
            blocks: number(&cells["blocks"]),
            table: match &cells["table"] {
                Value::Null => None,
                table => Some((table[0].as_u64().expect("bound"), slice(&table[1]))),
            },
        })
        .collect();
    Layout {
        cells,
        key_vars: number(&value["key_vars"]),
        witness_vars: number(&value["witness_vars"]),
        witness_len: number(&value["witness_len"]),
        span_vars: number(&value["span_vars"]),
        span_len: number(&value["span_len"]),
        inverse_vars: number(&value["inverse_vars"]),
    }
}

fn claim(blocks: &Value, table: &Value) -> (usize, Norm) {
    let norm = match table {
        Value::Null => Norm::Caller,
        bound => Norm::Table {
            bound: bound.as_u64().expect("bound"),
        },
    };
    (number(blocks), norm)
}

/// Each claim set with its key: the production sets, whose claims are the
/// cells of their Lean layout, and the generated sets.
fn claim_sets(lean: &Value) -> Vec<(Vec<(usize, Norm)>, usize, &Value)> {
    let production = lean["production"]
        .as_array()
        .expect("production")
        .iter()
        .map(|entry| {
            let claims = entry["layout"]["cells"]
                .as_array()
                .expect("cells")
                .iter()
                .map(|cells| claim(&cells["blocks"], &cells["table"][0]))
                .collect();
            (claims, number(&lean["key_blocks"]), entry)
        });
    let generated = lean["generated"]
        .as_array()
        .expect("generated")
        .iter()
        .map(|entry| {
            let claims = entry["claims"]
                .as_array()
                .expect("claims")
                .iter()
                .map(|pair| claim(&pair[0], &pair[1]))
                .collect();
            (claims, number(&entry["key_blocks"]), entry)
        });
    production.chain(generated).collect()
}

/// The Lean name of a rejection of `Layout::new`.
fn rejection(error: &Error) -> &'static str {
    match error {
        Error::Layout("no Akita schedule row has a key of this size") => "no_key",
        Error::Layout("a claim has no block") => "empty_claim",
        Error::Layout("a claim has more blocks than the key") => "claim_above_key",
        Error::Layout("no claim has a table") => "no_table",
        Error::Layout("a table exceeds the code width") => "table_wide",
        Error::Layout("no Akita schedule row holds the cells") => "no_row",
        other => panic!("not a rejection of Layout::new: {other}"),
    }
}

#[test]
fn schedules_and_variables_are_lean_constants() {
    let lean = lean();
    let schedules: Vec<(usize, Vec<(usize, usize)>)> = lean["schedules"]
        .as_array()
        .expect("schedules")
        .iter()
        .map(|schedule| {
            let rows = schedule["rows"]
                .as_array()
                .expect("rows")
                .iter()
                .map(|row| (number(&row[0]), number(&row[1])))
                .collect();
            (number(&schedule["key_vars"]), rows)
        })
        .collect();
    let rust: Vec<(usize, Vec<(usize, usize)>)> = SCHEDULES
        .iter()
        .map(|schedule| (schedule.key_vars, schedule.rows.to_vec()))
        .collect();
    assert_eq!(rust, schedules, "Lean schedules is Rust SCHEDULES");
    assert_eq!(
        [LANE_VARS, ROW_VARS, LOOKUP_VARS],
        [&lean["lane_vars"], &lean["row_vars"], &lean["lookup_vars"]].map(number)
    );
}

#[test]
fn layout_new_is_lean_layout_new_on_each_claim_set() {
    let lean = lean();
    let sets = claim_sets(&lean);
    assert_eq!(sets.len(), 8450 + 3001, "the production and the generated claim sets");
    for (claims, key_blocks, entry) in sets {
        let context = format!("claims {claims:?} under {key_blocks} key blocks");
        match (Layout::new(&claims, key_blocks), &entry["rejected"]) {
            (Ok(rust), Value::Null) => {
                assert_eq!(rust, layout(&entry["layout"]), "{context}");
                assert_eq!(rust.key_block_vars(), rust.key_vars - LANE_VARS - ROW_VARS, "{context}");
            }
            (Err(error), Value::String(name)) => assert_eq!(rejection(&error), name, "{context}"),
            (rust, lean) => panic!("{context}: Rust {rust:?}, Lean {lean}"),
        }
    }
}

#[test]
fn error_bits_is_lean_run_bits_on_each_layout() {
    let lean = lean();
    let mut layouts = 0;
    for (claims, key_blocks, entry) in claim_sets(&lean) {
        let Ok(layout) = Layout::new(&claims, key_blocks) else {
            continue;
        };
        assert_eq!(
            error_bits(&layout, PUBLIC_CELLS, MATRICES) as usize,
            number(&entry["error"]["bits"]),
            "{claims:?}"
        );
        layouts += 1;
    }
    assert_eq!(layouts, 8450 + 508, "the claim sets with a layout");
}

/// Nine claims under the key of `2^19` blocks: the terms over `|K|` other
/// than the lookup are `512/|K| > 2^-119`, because `|K| < 2^128`. So the
/// error sum is above `2^-119`, and 118 is the largest minimum that it
/// meets. In `f64`, `-log2` of this sum was exactly 119.
#[test]
fn error_bits_is_118_where_the_error_sum_is_just_above_2_to_the_minus_119() {
    let mut claims = vec![
        (1, Norm::Table { bound: 0 }),
        (1 << 19, Norm::Caller),
        (1 << 19, Norm::Caller),
    ];
    claims.extend([(1, Norm::Caller); 6]);
    let layout = Layout::new(&claims, 1 << 19).expect("layout");
    assert_eq!(layout.witness_vars, 27);
    assert_eq!(error_bits(&layout, PUBLIC_CELLS, MATRICES), 118);
}

/// `((N + T)/q)^4 ≤ 1/|K|` needs `(N + T)² ≤ q`. The largest such
/// `N + T` is `2^32 − 1`. A table above it gives 0 bits.
#[test]
fn error_bits_is_zero_when_a_lookup_term_can_exceed_one_over_k() {
    let mut layout = Layout::new(&[(1, Norm::Table { bound: 0 }), (1, Norm::Caller)], 512).expect("layout");
    let witness = layout.cells[0].witness.len() as u64;
    let mut bits = |poles: u64| {
        let table = layout.cells[0].table.as_mut().expect("table");
        table.0 = (poles - witness - 1) / 2;
        error_bits(&layout, PUBLIC_CELLS, MATRICES)
    };
    assert!(bits((1 << 32) - 1) > 100);
    assert_eq!(bits((1 << 32) + 1), 0);
}
