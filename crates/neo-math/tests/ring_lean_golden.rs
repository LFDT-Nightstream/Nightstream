use neo_math::{ct, Fq, Rq, D};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::fs;
use std::path::PathBuf;

fn parse_u64(field: &str, line_no: usize) -> u64 {
    field
        .parse::<u64>()
        .unwrap_or_else(|_| panic!("invalid u64 at line {line_no}: {field}"))
}

fn parse_coeffs(field: &str, line_no: usize) -> [Fq; D] {
    let parts: Vec<&str> = field.split(':').collect();
    assert_eq!(
        parts.len(),
        D,
        "expected {D} coefficients at line {line_no}, got {}",
        parts.len()
    );
    let mut out = [Fq::ZERO; D];
    for (i, part) in parts.iter().enumerate() {
        out[i] = Fq::from_u64(parse_u64(part, line_no));
    }
    out
}

#[test]
fn ring_mul_matches_lean_golden_vectors() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/superneo-lean/SuperNeo/Generated/RingGolden.csv");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    let mut seen_modulus = false;
    let mut seen_d = false;
    let mut expected_cases = None::<usize>;
    let mut mul_count = 0usize;

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
                assert_eq!(
                    got,
                    Fq::ORDER_U64,
                    "modulus mismatch at line {line_no}"
                );
                seen_modulus = true;
            }
            ["d", d] => {
                let got = parse_u64(d, line_no) as usize;
                assert_eq!(got, D, "D mismatch at line {line_no}");
                seen_d = true;
            }
            ["cases", n] => {
                expected_cases = Some(parse_u64(n, line_no) as usize);
            }
            ["mul", _case_idx, a, b, expected_c, expected_ct] => {
                let a = parse_coeffs(a, line_no);
                let b = parse_coeffs(b, line_no);
                let expected_c = parse_coeffs(expected_c, line_no);
                let expected_ct = parse_u64(expected_ct, line_no);

                let got = Rq(a).mul(&Rq(b));
                for i in 0..D {
                    let g = got.0[i].as_canonical_u64();
                    let e = expected_c[i].as_canonical_u64();
                    assert_eq!(
                        g, e,
                        "mul coeff mismatch at line {line_no}, coeff {i}"
                    );
                }

                let got_ct = ct(&got).as_canonical_u64();
                assert_eq!(got_ct, expected_ct, "ct mismatch at line {line_no}");
                mul_count += 1;
            }
            _ => panic!("invalid golden-vector line {line_no}: {line}"),
        }
    }

    assert!(seen_modulus, "missing modulus line");
    assert!(seen_d, "missing d line");
    let expected_cases = expected_cases.expect("missing cases line");
    assert_eq!(mul_count, expected_cases, "unexpected mul case count");
}
