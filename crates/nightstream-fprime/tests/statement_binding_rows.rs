//! The statement-binding leaf alone rejects every second split of a packed
//! prior word. The test evaluates only the leaf's 4,680 child rows of the
//! emitted package, so no PiCCS or PiRLC row can reject first.

use std::{fs, ops::Range, path::PathBuf};

#[allow(dead_code, unused_imports)]
#[path = "../src/bin/check_package_conformance/support.rs"]
mod conformance_support;

const MODULUS: u64 = 0xffff_ffff_0000_0001;
const PRIVATE_COLUMNS: usize = 11_470_080;
const PUBLIC_COLUMNS: usize = 278;
/// Lean `PilotPiCCS.cumulativeFootprints_eq`: the pilot ends at row 5,086,126,
/// and the leaf's 32 state-word rows come before its 4,680 child rows.
const CHILD_ROWS: Range<usize> = 5_086_158..5_090_838;
const PRIOR_PACKED_WORD: usize = 27_716;
const PRIOR_CHILD_DIGITS: usize = 55_638;
const SIGNS: usize = 5_156_678;
const PUBLIC_WIDTH: usize = 270;
const SIGN_ROW: usize = CHILD_ROWS.start;
const LANES: usize = 3;
const ROWS_PER_LANE: usize = 1 + 16;
const PACKING_ROW: usize = CHILD_ROWS.start + LANES * ROWS_PER_LANE;

fn assignment(child_0: u64, child_1: u64, sign: u64, packed: u64) -> Vec<u64> {
    let mut private = vec![0; PRIVATE_COLUMNS];
    private[PRIOR_CHILD_DIGITS] = child_0;
    private[PRIOR_CHILD_DIGITS + PUBLIC_WIDTH] = child_1;
    private[SIGNS] = sign;
    private[PRIOR_PACKED_WORD] = packed;
    private
}

#[test]
fn the_statement_binding_rows_reject_a_second_split() {
    let expanded = fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts")
            .join("nightstream-fprime-stage1-poseidon2-hash-chain-v1-expanded.json"),
    )
    .expect("Lean expanded package");
    let public = vec![0; PUBLIC_COLUMNS];
    let evaluate =
        |private: &[u64]| conformance_support::evaluate_canonical_rows(&expanded, private, &public, CHILD_ROWS);
    let minus_one = MODULUS - 1;

    let cases = [
        ("canonical split", 1, 0, 0, 1, Ok(CHILD_ROWS.len())),
        ("mixed signs with sign 0", minus_one, 1, 0, 1, Err(SIGN_ROW + 1)),
        ("mixed signs with sign 1", minus_one, 1, 1, 1, Err(SIGN_ROW + 2)),
        ("sign 1 on a positive digit", 1, 0, 1, 1, Err(SIGN_ROW + 1)),
        ("digits that do not recompose", 0, 0, 0, 1, Err(PACKING_ROW)),
    ];
    for (name, child_0, child_1, sign, packed, expected) in cases {
        assert_eq!(
            evaluate(&assignment(child_0, child_1, sign, packed)),
            expected,
            "{name}"
        );
    }
}
