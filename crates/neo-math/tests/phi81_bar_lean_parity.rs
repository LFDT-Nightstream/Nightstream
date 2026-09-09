//! Compare the runtime transform with every entry emitted by active Lean.

use std::io::{BufRead, BufReader};

use neo_math::{superneo_bar_block, superneo_bar_matrix, superneo_bar_vec, Fq, D};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[test]
#[ignore = "fresh Lean vectors required; run scripts/check_fprime_foundation_parity.sh"]
fn active_lean_bar_matches_runtime() {
    let path: std::path::PathBuf = serde_json::from_reader(std::io::stdin()).expect("Lean fixture path as JSON");
    let mut lines = BufReader::new(std::fs::File::open(path).expect("open Lean bar vectors")).lines();
    let header: Vec<u64> =
        serde_json::from_str(&lines.next().expect("header").expect("read header")).expect("decode header");
    assert_eq!(header, [Fq::ORDER_U64, D as u64]);

    let mut expected = [[Fq::ZERO; D]; D];
    for (row, entries) in expected.iter_mut().enumerate() {
        let values: Vec<u64> =
            serde_json::from_str(&lines.next().expect("matrix row").expect("read row")).expect("decode matrix row");
        assert_eq!(values.len(), D);
        for (column, value) in values.into_iter().enumerate() {
            assert!(value < Fq::ORDER_U64, "canonical Lean field value");
            assert_eq!(
                superneo_bar_matrix()[row][column].as_canonical_u64(),
                value,
                "matrix ({row},{column})"
            );
            entries[column] = Fq::from_u64(value);
        }
    }
    assert!(lines.next().is_none(), "unexpected extra matrix row");

    // Exercise the block and vector consumers on the complete coefficient basis.
    let mut basis_blocks = Vec::with_capacity(D * D);
    let mut expected_blocks = Vec::with_capacity(D * D);
    for column in 0..D {
        let mut basis = [Fq::ZERO; D];
        basis[column] = Fq::ONE;
        let image: [Fq; D] = std::array::from_fn(|row| expected[row][column]);
        assert_eq!(superneo_bar_block(basis), image, "basis {column}");
        basis_blocks.extend(basis);
        expected_blocks.extend(image);
    }
    assert_eq!(superneo_bar_vec(&basis_blocks), expected_blocks);
}
