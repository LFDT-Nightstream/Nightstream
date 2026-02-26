use neo_math::Fq;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use std::fs;
use std::path::PathBuf;

fn parse_u64(field: &str, line_no: usize) -> u64 {
    field
        .parse::<u64>()
        .unwrap_or_else(|_| panic!("invalid u64 at line {line_no}: {field}"))
}

#[test]
fn goldilocks_matches_lean_golden_vectors() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/superneo-lean/SuperNeo/Generated/GoldilocksGolden.csv");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    let mut seen_modulus = false;
    let mut add_count = 0usize;
    let mut sub_count = 0usize;
    let mut mul_count = 0usize;
    let mut neg_count = 0usize;
    let mut inv_count = 0usize;

    for (idx, raw_line) in content.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        match parts.as_slice() {
            ["modulus", modulus] => {
                let got = parse_u64(modulus, line_no);
                assert_eq!(got, Fq::ORDER_U64, "modulus mismatch at line {line_no}");
                seen_modulus = true;
            }
            ["add", a, b, expected] => {
                let a = Fq::from_u64(parse_u64(a, line_no));
                let b = Fq::from_u64(parse_u64(b, line_no));
                let expected = parse_u64(expected, line_no);
                let got = (a + b).as_canonical_u64();
                assert_eq!(got, expected, "add mismatch at line {line_no}");
                add_count += 1;
            }
            ["sub", a, b, expected] => {
                let a = Fq::from_u64(parse_u64(a, line_no));
                let b = Fq::from_u64(parse_u64(b, line_no));
                let expected = parse_u64(expected, line_no);
                let got = (a - b).as_canonical_u64();
                assert_eq!(got, expected, "sub mismatch at line {line_no}");
                sub_count += 1;
            }
            ["mul", a, b, expected] => {
                let a = Fq::from_u64(parse_u64(a, line_no));
                let b = Fq::from_u64(parse_u64(b, line_no));
                let expected = parse_u64(expected, line_no);
                let got = (a * b).as_canonical_u64();
                assert_eq!(got, expected, "mul mismatch at line {line_no}");
                mul_count += 1;
            }
            ["neg", a, expected] => {
                let a = Fq::from_u64(parse_u64(a, line_no));
                let expected = parse_u64(expected, line_no);
                let got = (-a).as_canonical_u64();
                assert_eq!(got, expected, "neg mismatch at line {line_no}");
                neg_count += 1;
            }
            ["inv", a, expected] => {
                let a = Fq::from_u64(parse_u64(a, line_no));
                let expected = parse_u64(expected, line_no);
                let got = a.inverse().as_canonical_u64();
                assert_eq!(got, expected, "inv mismatch at line {line_no}");
                inv_count += 1;
            }
            _ => panic!("invalid golden-vector line {line_no}: {line}"),
        }
    }

    assert!(seen_modulus, "missing modulus line");
    assert_eq!(add_count, 128, "unexpected add case count");
    assert_eq!(sub_count, 128, "unexpected sub case count");
    assert_eq!(mul_count, 128, "unexpected mul case count");
    assert_eq!(neg_count, 128, "unexpected neg case count");
    assert_eq!(inv_count, 128, "unexpected inv case count");
}
