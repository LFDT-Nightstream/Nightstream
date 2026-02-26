use neo_math::{cf, cf_inv, ct, superneo_bar_block, superneo_bar_vec, Fq, D};
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

fn parse_fq_blocks(field: &str, line_no: usize) -> Vec<Vec<Fq>> {
    if field.is_empty() {
        return Vec::new();
    }
    field.split('|').map(|blk| parse_fq_vec(blk, line_no)).collect()
}

fn parse_fq_matrix(field: &str, line_no: usize) -> Vec<Vec<Fq>> {
    if field.is_empty() {
        return Vec::new();
    }
    field.split(';').map(|row| parse_fq_vec(row, line_no)).collect()
}

fn to_block(xs: &[Fq], line_no: usize, label: &str) -> [Fq; D] {
    assert_eq!(
        xs.len(),
        D,
        "expected {D} elements for {label} at line {line_no}, got {}",
        xs.len()
    );
    let mut out = [Fq::ZERO; D];
    out.copy_from_slice(xs);
    out
}

fn vec_add(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    assert_eq!(a.len(), b.len(), "vec_add size mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}

fn vec_scale(s: Fq, a: &[Fq]) -> Vec<Fq> {
    a.iter().map(|x| s * *x).collect()
}

fn dot(a: &[Fq], b: &[Fq]) -> Fq {
    assert_eq!(a.len(), b.len(), "dot size mismatch");
    let mut acc = Fq::ZERO;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

fn row_ct_bar_prod(row: &[Fq], z: &[Fq]) -> Fq {
    assert_eq!(row.len(), z.len(), "row/z size mismatch");
    assert_eq!(row.len() % D, 0, "row length must be multiple of D");
    let mut acc = Fq::ZERO;
    for (a_blk, z_blk) in row.chunks_exact(D).zip(z.chunks_exact(D)) {
        let a_bar = superneo_bar_block(to_block(a_blk, 0, "row block"));
        let term = ct(&cf_inv(a_bar).mul(&cf_inv(to_block(z_blk, 0, "z block"))));
        acc += term;
    }
    acc
}

#[test]
fn p9_p11_p12_matches_lean_golden_vectors() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/superneo-lean/SuperNeo/Generated/P9P11P12Golden.csv");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    let mut seen_modulus = false;
    let mut seen_d = false;
    let mut expected_embed_cases = None::<usize>;
    let mut expected_barlift_cases = None::<usize>;
    let mut expected_matrix_cases = None::<usize>;
    let mut embed_count = 0usize;
    let mut barlift_count = 0usize;
    let mut matrix_count = 0usize;

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
            ["d", d] => {
                let got = parse_usize(d, line_no);
                assert_eq!(got, D, "D mismatch at line {line_no}");
                seen_d = true;
            }
            ["embed_cases", n] => {
                expected_embed_cases = Some(parse_usize(n, line_no));
            }
            ["barlift_cases", n] => {
                expected_barlift_cases = Some(parse_usize(n, line_no));
            }
            ["matrix_cases", n] => {
                expected_matrix_cases = Some(parse_usize(n, line_no));
            }
            ["embed", _case_idx, input_csv, blocks_csv, roundtrip] => {
                let input = parse_fq_vec(input_csv, line_no);
                let blocks = parse_fq_blocks(blocks_csv, line_no);
                let roundtrip = parse_u64(roundtrip, line_no);

                assert_eq!(
                    input.len() % D,
                    0,
                    "embed input length must be multiple of D at line {line_no}"
                );
                assert_eq!(
                    blocks.len(),
                    input.len() / D,
                    "embed block count mismatch at line {line_no}"
                );

                let mut reconstructed = Vec::with_capacity(input.len());
                for (bidx, blk) in blocks.iter().enumerate() {
                    let blk_arr = to_block(blk, line_no, "embed block");
                    let got_coeff = cf(cf_inv(blk_arr));
                    for i in 0..D {
                        assert_eq!(
                            got_coeff[i],
                            blk_arr[i],
                            "embed coeff mismatch at line {line_no}, block {bidx}, coeff {i}"
                        );
                    }
                    reconstructed.extend_from_slice(&got_coeff);
                }

                let got_roundtrip = if reconstructed == input { 1 } else { 0 };
                assert_eq!(got_roundtrip, roundtrip, "embed roundtrip mismatch at line {line_no}");
                embed_count += 1;
            }
            [
                "barlift",
                _case_idx,
                v_csv,
                w_csv,
                scalar,
                expected_lift_v,
                expected_lift_w,
                expected_lift_add,
                expected_lift_scale,
            ] => {
                let v = parse_fq_vec(v_csv, line_no);
                let w = parse_fq_vec(w_csv, line_no);
                let scalar = Fq::from_u64(parse_u64(scalar, line_no));
                let expected_lift_v = parse_fq_vec(expected_lift_v, line_no);
                let expected_lift_w = parse_fq_vec(expected_lift_w, line_no);
                let expected_lift_add = parse_fq_vec(expected_lift_add, line_no);
                let expected_lift_scale = parse_fq_vec(expected_lift_scale, line_no);

                assert_eq!(v.len(), w.len(), "barlift size mismatch at line {line_no}");
                assert_eq!(
                    v.len() % D,
                    0,
                    "barlift vector length must be multiple of D at line {line_no}"
                );

                let got_lift_v = superneo_bar_vec(&v);
                let got_lift_w = superneo_bar_vec(&w);
                let got_lift_add = superneo_bar_vec(&vec_add(&v, &w));
                let got_lift_scale = superneo_bar_vec(&vec_scale(scalar, &v));

                assert_eq!(got_lift_v, expected_lift_v, "lift(v) mismatch at line {line_no}");
                assert_eq!(got_lift_w, expected_lift_w, "lift(w) mismatch at line {line_no}");
                assert_eq!(
                    got_lift_add, expected_lift_add,
                    "lift(v+w) mismatch at line {line_no}"
                );
                assert_eq!(
                    got_lift_scale, expected_lift_scale,
                    "lift(s*v) mismatch at line {line_no}"
                );
                assert_eq!(
                    got_lift_add,
                    vec_add(&got_lift_v, &got_lift_w),
                    "lift add-linearity mismatch at line {line_no}"
                );
                assert_eq!(
                    got_lift_scale,
                    vec_scale(scalar, &got_lift_v),
                    "lift scale-linearity mismatch at line {line_no}"
                );

                barlift_count += 1;
            }
            [
                "matrix",
                _case_idx,
                rows,
                cols,
                matrix_csv,
                z_csv,
                expected_mz_csv,
                expected_ct_bar_csv,
                identity,
            ] => {
                let rows = parse_usize(rows, line_no);
                let cols = parse_usize(cols, line_no);
                let matrix = parse_fq_matrix(matrix_csv, line_no);
                let z = parse_fq_vec(z_csv, line_no);
                let expected_mz = parse_fq_vec(expected_mz_csv, line_no);
                let expected_ct_bar = parse_fq_vec(expected_ct_bar_csv, line_no);
                let identity = parse_u64(identity, line_no);

                assert_eq!(matrix.len(), rows, "matrix row count mismatch at line {line_no}");
                for (ridx, row) in matrix.iter().enumerate() {
                    assert_eq!(
                        row.len(),
                        cols,
                        "matrix col count mismatch at line {line_no}, row {ridx}"
                    );
                    assert_eq!(
                        row.len() % D,
                        0,
                        "matrix row width not multiple of D at line {line_no}, row {ridx}"
                    );
                }
                assert_eq!(z.len(), cols, "z size mismatch at line {line_no}");
                assert_eq!(
                    expected_mz.len(),
                    rows,
                    "expected Mz size mismatch at line {line_no}"
                );
                assert_eq!(
                    expected_ct_bar.len(),
                    rows,
                    "expected ct(bar(M)z) size mismatch at line {line_no}"
                );

                let got_mz: Vec<Fq> = matrix.iter().map(|row| dot(row, &z)).collect();
                let got_ct_bar: Vec<Fq> = matrix.iter().map(|row| row_ct_bar_prod(row, &z)).collect();
                let got_identity = if got_mz == got_ct_bar { 1 } else { 0 };

                assert_eq!(got_mz, expected_mz, "matrix direct mismatch at line {line_no}");
                assert_eq!(
                    got_ct_bar, expected_ct_bar,
                    "matrix ct(bar(M)z) mismatch at line {line_no}"
                );
                assert_eq!(got_identity, identity, "matrix identity mismatch at line {line_no}");

                matrix_count += 1;
            }
            _ => panic!("invalid golden-vector line {line_no}: {line}"),
        }
    }

    assert!(seen_modulus, "missing modulus line");
    assert!(seen_d, "missing d line");
    assert_eq!(
        embed_count,
        expected_embed_cases.expect("missing embed_cases line"),
        "unexpected embed case count"
    );
    assert_eq!(
        barlift_count,
        expected_barlift_cases.expect("missing barlift_cases line"),
        "unexpected barlift case count"
    );
    assert_eq!(
        matrix_count,
        expected_matrix_cases.expect("missing matrix_cases line"),
        "unexpected matrix case count"
    );
}
