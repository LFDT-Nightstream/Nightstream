//! The fixed WASM sizes against Lean (`Spec.Nebula.WasmSize`). Lean proves
//! that each size is valid and secure, that the memory gate certifies 109
//! bits for all its segments, and that one more segment certifies fewer
//! (`WasmSize.segmentBits_eq`, `WasmSize.segmentBits_succ_lt`), and that no
//! proof of a size reaches a refresh fold (`WasmSize.steps_le_light`). Lean
//! emits the geometry and the rows of a step of each size, and `WasmSize`
//! must have the same values.

use neo_wasm::nebula::WasmSize;
use nightstream::nebula::Geometry;
use serde_json::Value;

#[test]
fn each_wasm_size_has_the_lean_geometry_and_rows() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../formal/nightstream-fprime/artifacts/nightstream-fprime-nebula-gate-parity-v1.json"
    );
    let lean: Value =
        serde_json::from_slice(&std::fs::read(path).expect("Lean memory gate file")).expect("Lean memory gate JSON");
    assert_eq!(lean[0].as_u64(), Some(1), "Lean memory gate schema");
    let sizes = lean[3].as_array().expect("sizes");
    assert_eq!(sizes.len(), WasmSize::ALL.len());
    for (size, row) in WasmSize::ALL.into_iter().zip(sizes) {
        let row: Vec<usize> = row
            .as_array()
            .expect("numbers")
            .iter()
            .map(|number| usize::try_from(number.as_u64().expect("a natural number")).expect("usize"))
            .collect();
        let geometry = Geometry {
            r: row[0],
            mu: row[1],
            w_ts: row[2],
            b_ops: row[3],
            b_scan: row[4],
            n: row[5],
            s_max: row[6],
        };
        assert_eq!(size.geometry(), geometry, "{size:?}");
        assert_eq!(size.rows().get(), row[9], "{size:?}");
        assert_eq!(row[10], 1, "{size:?}: Lean S_max · N ≤ cap + 1");
    }
}
