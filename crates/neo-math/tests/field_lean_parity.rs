//! Compare base and extension arithmetic with fresh active-Lean values.

use std::io::{BufRead, BufReader};

use neo_math::{Fq, KExtensions, K};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

#[test]
#[ignore = "fresh Lean vectors required; run scripts/check_fprime_foundation_parity.sh"]
fn active_lean_arithmetic_matches_runtime() {
    let directory: std::path::PathBuf = serde_json::from_reader(std::io::stdin()).expect("Lean vector directory");
    let mut lines = BufReader::new(std::fs::File::open(directory.join("field.jsonl")).unwrap()).lines();
    let header: [u64; 2] = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(header[0], Fq::ORDER_U64);
    let mut count = 0u64;
    for line in lines {
        let row: [u64; 9] = serde_json::from_str(&line.unwrap()).unwrap();
        let a = Fq::from_u64(row[0]);
        let b = Fq::from_u64(row[1]);
        let actual = [a, b, a + b, a - b, a * b, -a, a.try_inverse().unwrap_or(Fq::ZERO)];
        assert_eq!(
            actual.map(|value| value.as_canonical_u64()).as_slice(),
            &row[2..],
            "base inputs {}, {}",
            row[0],
            row[1]
        );
        count += 1;
    }
    assert_eq!(count, header[1].checked_mul(header[1]).unwrap());
    assert_ne!(count, 0);

    let mut lines = BufReader::new(std::fs::File::open(directory.join("extension.jsonl")).unwrap()).lines();
    let header: [u64; 3] = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(&header[..2], &[Fq::ORDER_U64, 7]);
    let mut count = 0u64;
    for line in lines {
        let row: [[u64; 2]; 8] = serde_json::from_str(&line.unwrap()).unwrap();
        assert!(row.iter().flatten().all(|word| *word < Fq::ORDER_U64));
        let a = K::from_coeffs(row[0].map(Fq::from_u64));
        let b = K::from_coeffs(row[1].map(Fq::from_u64));
        let actual = [a + b, a - b, a * b, -a, a.conj(), a.try_inverse().unwrap_or(K::ZERO)];
        assert_eq!(
            actual
                .map(|value| value
                    .as_coeffs()
                    .map(|coefficient| coefficient.as_canonical_u64()))
                .as_slice(),
            &row[2..],
            "extension inputs {:?}, {:?}",
            row[0],
            row[1]
        );
        count += 1;
    }
    assert_eq!(count, header[2].checked_mul(header[2]).unwrap());
    assert_ne!(count, 0);
}
