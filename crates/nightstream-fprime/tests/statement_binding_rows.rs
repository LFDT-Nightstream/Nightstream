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
/// Spartan columns: the prior preimage starts at 0 (packed parent words at
/// 27,716), role 19 at 55,638 (child-major), and the 270 hinted signs at the
/// PiCCS local start.
const PRIOR_PACKED_WORD: usize = 27_716;
const PRIOR_CHILD_DIGITS: usize = 55_638;
const SIGNS: usize = 5_156_678;
const PUBLIC_WIDTH: usize = 270;
/// Word 0, lane 0: the sign row, then one digit row per child. The packing
/// row follows the three lanes.
const SIGN_ROW: usize = CHILD_ROWS.start;
const PACKING_ROW: usize = CHILD_ROWS.start + 3 * 17;

/// Parent coordinate 0 (packed word 0, lane 0) with children 0 and 1 set.
/// Every other lane is zero and satisfies its rows.
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

    // The canonical split of parent 1: digit 1 at child 0, sign 0.
    assert_eq!(evaluate(&assignment(1, 0, 0, 1)), Ok(CHILD_ROWS.len()));
    // 1 = -1 + 2 * 1 keeps the packed word but mixes the signs: one digit
    // row fails for either sign value.
    assert_eq!(evaluate(&assignment(minus_one, 1, 0, 1)), Err(SIGN_ROW + 1));
    assert_eq!(evaluate(&assignment(minus_one, 1, 1, 1)), Err(SIGN_ROW + 2));
    // The hinted sign is not free: sign 1 rejects a positive digit.
    assert_eq!(evaluate(&assignment(1, 0, 1, 1)), Err(SIGN_ROW + 1));
    // Valid digits that do not recompose the hashed word fail the packing row.
    assert_eq!(evaluate(&assignment(0, 0, 0, 1)), Err(PACKING_ROW));
}
