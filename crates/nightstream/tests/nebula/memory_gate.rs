//! The memory gate against Lean (`Spec.Nebula.SegmentGate`). Lean emits the
//! bound of rule 6 of SPEC.md section 7.1, the geometry of each fixed WASM
//! size with `m_mem` and the bits of its segments
//! (`WasmSize.segmentBits_eq`), and its decisions `Geometry.Valid` and
//! `Geometry.Secure` on geometries that break one rule each. Rust keeps its
//! own code for each, and must give the same values.

use serde_json::Value;

use super::{compression, segments_security_bits, Geometry};
use crate::lifecycle::bounded_depth::Compression;

fn lean() -> Vec<Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../formal/nightstream-fprime/artifacts/nightstream-fprime-nebula-gate-parity-v1.json"
    );
    let file: Value =
        serde_json::from_slice(&std::fs::read(path).expect("Lean memory gate file")).expect("Lean memory gate JSON");
    let values = file.as_array().expect("schema-1 array").clone();
    assert_eq!(values[0].as_u64(), Some(1), "Lean memory gate schema");
    values
}

fn numbers(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .expect("numbers")
        .iter()
        .map(|number| usize::try_from(number.as_u64().expect("a natural number")).expect("usize"))
        .collect()
}

fn rows(value: &Value) -> Vec<Vec<usize>> {
    value
        .as_array()
        .expect("rows")
        .iter()
        .map(numbers)
        .collect()
}

fn geometry(row: &[usize]) -> Geometry {
    Geometry {
        r: row[0],
        mu: row[1],
        w_ts: row[2],
        b_ops: row[3],
        b_scan: row[4],
        n: row[5],
        s_max: row[6],
    }
}

#[test]
fn the_gate_gives_the_lean_bits_on_each_wasm_geometry() {
    let lean = lean();
    let bits = lean[2].as_u64().expect("bits");
    let sizes = rows(&lean[3]);
    assert_eq!(sizes.len(), 4);
    for row in sizes {
        let size = geometry(&row);
        assert_eq!(Geometry::new(row[0], row[1], row[3], row[5], row[6]), Some(size));
        assert_eq!(size.max_tuples(), row[7]);
        assert_eq!(u64::from(segments_security_bits(&size)), bits, "{size:?}");
        assert_eq!(row[8] as u64, bits);
    }
}

#[test]
fn is_valid_and_is_secure_are_the_lean_decisions() {
    let lean = lean();
    let bound = usize::try_from(lean[1].as_u64().expect("the bound of rule 6")).expect("usize");
    let cases = rows(&lean[4]);
    assert!(
        cases.iter().any(|row| row[7] == 1 && row[8] == 0),
        "a valid case that is not secure"
    );
    assert!(
        cases.iter().filter(|row| row[7] == 0).count() >= 6,
        "a case for each rule"
    );
    for row in cases {
        let case = geometry(&row);
        assert_eq!(case.is_valid(), row[7] == 1, "{case:?}");
        if case.is_valid() {
            assert_eq!(case.is_secure(), row[8] == 1, "{case:?}");
            assert_eq!(case.max_tuples() <= bound, row[8] == 1, "{case:?}");
        }
    }
}

/// A memory proof has at most `N · S_max` steps (SPEC.md Algorithm 8). The
/// final proof of a geometry with at most `cap + 1 = 65` of them carries slot
/// 0 alone (Algorithm 9), and each Lean WASM size is such a geometry
/// (`WasmSize.steps_le_light`). At 66 steps a proof reaches a refresh fold,
/// and the final proof carries both slots.
#[test]
fn a_geometry_whose_proofs_stay_light_compresses_slot_zero_alone() {
    let lean = lean();
    for row in rows(&lean[3]) {
        let size = geometry(&row);
        assert_eq!(row[10], 1, "{size:?}: Lean S_max · N ≤ cap + 1");
        assert_eq!(compression(&size), Compression::Light, "{size:?}");
    }
    let base = geometry(&rows(&lean[3])[0]);
    for (n, s_max, expected) in [
        (1, 65, Compression::Light),
        (5, 13, Compression::Light),
        (1, 66, Compression::Both),
        (8, 19, Compression::Both),
        (16, 9, Compression::Both),
    ] {
        let case = Geometry { n, s_max, ..base };
        assert_eq!(compression(&case), expected, "N = {n}, S_max = {s_max}");
    }
}
