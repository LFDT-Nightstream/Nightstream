use neo_math::{Fq, K};
use neo_memory::mle::{build_chi_table, chi_at_index, eq_points, mle_eval};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::fs;
use std::path::PathBuf;

fn parse_u64(field: &str, line_no: usize) -> u64 {
    field
        .parse::<u64>()
        .unwrap_or_else(|_| panic!("invalid u64 at line {line_no}: {field}"))
}

fn parse_usize(field: &str, line_no: usize) -> usize {
    field
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid usize at line {line_no}: {field}"))
}

fn parse_fq_vec(field: &str, line_no: usize) -> Vec<Fq> {
    if field.is_empty() {
        return Vec::new();
    }
    field
        .split(':')
        .map(|s| Fq::from_u64(parse_u64(s, line_no)))
        .collect()
}

#[test]
fn eq_mle_matches_lean_golden_vectors() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/superneo-lean/SuperNeo/Generated/EqMleGolden.csv");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    let mut seen_modulus = false;
    let mut expected_eq_cases = None::<usize>;
    let mut expected_mle_cases = None::<usize>;
    let mut eq_count = 0usize;
    let mut mle_count = 0usize;

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
            ["eq_cases", n] => {
                expected_eq_cases = Some(parse_usize(n, line_no));
            }
            ["mle_cases", n] => {
                expected_mle_cases = Some(parse_usize(n, line_no));
            }
            ["eq", _case_idx, width, bool_tag, x_csv, y_csv, expected_eq, expected_indicator] => {
                let width = parse_usize(width, line_no);
                let bool_tag = parse_u64(bool_tag, line_no);
                let x = parse_fq_vec(x_csv, line_no);
                let y = parse_fq_vec(y_csv, line_no);
                let expected_eq = parse_u64(expected_eq, line_no);
                let expected_indicator = parse_u64(expected_indicator, line_no);

                assert_eq!(x.len(), width, "x width mismatch at line {line_no}");
                assert_eq!(y.len(), width, "y width mismatch at line {line_no}");

                let xk: Vec<K> = x.iter().copied().map(K::from).collect();
                let yk: Vec<K> = y.iter().copied().map(K::from).collect();
                let got = eq_points(&xk, &yk);
                let expected_k = K::from(Fq::from_u64(expected_eq));
                assert_eq!(got, expected_k, "eq mismatch at line {line_no}");

                if bool_tag == 1 {
                    let expected_ind_k = K::from(Fq::from_u64(expected_indicator));
                    assert_eq!(
                        got, expected_ind_k,
                        "boolean indicator mismatch at line {line_no}"
                    );
                }

                eq_count += 1;
            }
            [
                "mle",
                _case_idx,
                ell,
                v_csv,
                r_csv,
                expected_inner,
                expected_fold,
                expected_chi_sum,
                probe_idx,
                expected_probe_weight,
            ] => {
                let ell = parse_usize(ell, line_no);
                let v = parse_fq_vec(v_csv, line_no);
                let r = parse_fq_vec(r_csv, line_no);
                let expected_inner = parse_u64(expected_inner, line_no);
                let expected_fold = parse_u64(expected_fold, line_no);
                let expected_chi_sum = parse_u64(expected_chi_sum, line_no);
                let probe_idx = parse_usize(probe_idx, line_no);
                let expected_probe_weight = parse_u64(expected_probe_weight, line_no);

                assert_eq!(r.len(), ell, "r length mismatch at line {line_no}");
                assert_eq!(
                    v.len(),
                    1usize << ell,
                    "v length mismatch at line {line_no}"
                );

                let got_eval = mle_eval::<Fq, Fq>(&v, &r);
                assert_eq!(
                    got_eval.as_canonical_u64(),
                    expected_inner,
                    "mle_eval mismatch at line {line_no}"
                );
                assert_eq!(
                    expected_inner, expected_fold,
                    "lean inner/fold mismatch at line {line_no}"
                );

                let chi = build_chi_table(&r);
                assert_eq!(chi.len(), v.len(), "chi length mismatch at line {line_no}");

                let chi_sum: Fq = chi.iter().copied().sum();
                assert_eq!(
                    chi_sum.as_canonical_u64(),
                    expected_chi_sum,
                    "chi sum mismatch at line {line_no}"
                );

                assert!(
                    probe_idx < chi.len(),
                    "probe idx out of range at line {line_no}"
                );
                let expected_probe = Fq::from_u64(expected_probe_weight);
                assert_eq!(chi[probe_idx], expected_probe, "chi probe mismatch at line {line_no}");
                assert_eq!(
                    chi_at_index(&r, probe_idx),
                    expected_probe,
                    "chi_at_index mismatch at line {line_no}"
                );

                let manual = v
                    .iter()
                    .zip(chi.iter())
                    .fold(Fq::ZERO, |acc, (val, weight)| acc + (*val * *weight));
                assert_eq!(manual, got_eval, "manual mle mismatch at line {line_no}");

                mle_count += 1;
            }
            _ => panic!("invalid golden-vector line {line_no}: {line}"),
        }
    }

    assert!(seen_modulus, "missing modulus line");
    assert_eq!(
        eq_count,
        expected_eq_cases.expect("missing eq_cases line"),
        "unexpected eq case count"
    );
    assert_eq!(
        mle_count,
        expected_mle_cases.expect("missing mle_cases line"),
        "unexpected mle case count"
    );
}
