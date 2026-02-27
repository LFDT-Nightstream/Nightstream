use neo_math::{cf_inv, superneo_bar_block, Fq, D};
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

fn dot_f(a: &[Fq], b: &[Fq]) -> Fq {
    if a.len() != b.len() {
        return Fq::ZERO;
    }
    let mut acc = Fq::ZERO;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

fn vec_add(a: &[Fq], b: &[Fq]) -> Vec<Fq> {
    if a.len() != b.len() {
        return Vec::new();
    }
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}

fn vec_scale(s: Fq, a: &[Fq]) -> Vec<Fq> {
    a.iter().map(|x| s * *x).collect()
}

fn row_bar_mz_ring(row: &[Fq], z: &[Fq]) -> Vec<Fq> {
    if row.len() != z.len() {
        return Vec::new();
    }
    if row.len() % D != 0 {
        return Vec::new();
    }
    let mut acc = vec![Fq::ZERO; D];
    for (a_blk, z_blk) in row.chunks_exact(D).zip(z.chunks_exact(D)) {
        let a_bar = superneo_bar_block(to_block(a_blk, 0, "row block"));
        let term = cf_inv(a_bar).mul(&cf_inv(to_block(z_blk, 0, "z block")));
        for (i, acc_i) in acc.iter_mut().enumerate().take(D) {
            *acc_i += term.0[i];
        }
    }
    acc
}

fn bar_mz_ring(m: &[Vec<Fq>], z: &[Fq]) -> Vec<Vec<Fq>> {
    m.iter().map(|row| row_bar_mz_ring(row, z)).collect()
}

fn coeff_rows_of_ring_vec(ys: &[Vec<Fq>]) -> Vec<Vec<Fq>> {
    (0..D)
        .map(|ell| ys.iter().map(|yi| *yi.get(ell).unwrap_or(&Fq::ZERO)).collect())
        .collect()
}

fn eval_coeff_rows(rows: &[Vec<Fq>], weights: &[Fq]) -> Vec<Fq> {
    rows.iter().map(|row| dot_f(row, weights)).collect()
}

fn eval_ring_vec(ys: &[Vec<Fq>], weights: &[Fq]) -> Vec<Fq> {
    if ys.len() != weights.len() {
        return Vec::new();
    }
    eval_coeff_rows(&coeff_rows_of_ring_vec(ys), weights)
}

fn ct_row(ys: &[Vec<Fq>]) -> Vec<Fq> {
    ys.iter().map(|yi| *yi.first().unwrap_or(&Fq::ZERO)).collect()
}

fn chi_weight(r: &[Fq], j: usize) -> Fq {
    let mut w = Fq::ONE;
    for (i, ri) in r.iter().enumerate() {
        let bit = (j >> i) & 1;
        let term = if bit == 1 { *ri } else { Fq::ONE - *ri };
        w *= term;
    }
    w
}

fn r_hat(r: &[Fq], n: usize) -> Vec<Fq> {
    (0..n).map(|j| chi_weight(r, j)).collect()
}

fn eval_link_for_matrix(m: &[Vec<Fq>], z: &[Fq], r: &[Fq]) -> bool {
    let ys = bar_mz_ring(m, z);
    let weights = r_hat(r, ys.len());
    if ys.len() != weights.len() {
        return false;
    }
    let y = eval_ring_vec(&ys, &weights);
    let coeff_side = eval_coeff_rows(&coeff_rows_of_ring_vec(&ys), &weights);
    let ct_side = dot_f(&ct_row(&ys), &weights);
    y == coeff_side && y.first().copied().unwrap_or(Fq::ZERO) == ct_side
}

fn eval_bar_mz_at(m: &[Vec<Fq>], z: &[Fq], r: &[Fq]) -> Vec<Fq> {
    let ys = bar_mz_ring(m, z);
    let weights = r_hat(r, ys.len());
    eval_ring_vec(&ys, &weights)
}

fn lin_comb_2_vec(rho1: Fq, rho2: Fq, z1: &[Fq], z2: &[Fq]) -> Vec<Fq> {
    vec_add(&vec_scale(rho1, z1), &vec_scale(rho2, z2))
}

fn eval_hom2(m: &[Vec<Fq>], z1: &[Fq], z2: &[Fq], r: &[Fq], rho1: Fq, rho2: Fq) -> bool {
    if z1.len() != z2.len() {
        return false;
    }
    if !m
        .iter()
        .all(|row| row.len() == z1.len() && row.len().is_multiple_of(D))
    {
        return false;
    }
    let y1 = eval_bar_mz_at(m, z1, r);
    let y2 = eval_bar_mz_at(m, z2, r);
    let z_star = lin_comb_2_vec(rho1, rho2, z1, z2);
    let y_lin = vec_add(&vec_scale(rho1, &y1), &vec_scale(rho2, &y2));
    let y_direct = eval_bar_mz_at(m, &z_star, r);
    y_lin == y_direct
        && y_lin.first().copied().unwrap_or(Fq::ZERO)
            == rho1 * y1.first().copied().unwrap_or(Fq::ZERO)
                + rho2 * y2.first().copied().unwrap_or(Fq::ZERO)
}

#[test]
fn p13_p14_matches_lean_golden_vectors() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/superneo-lean/SuperNeo/Generated/P13P14Golden.csv");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    let mut seen_modulus = false;
    let mut seen_d = false;
    let mut expected_link_cases = None::<usize>;
    let mut expected_hom_cases = None::<usize>;
    let mut link_count = 0usize;
    let mut hom_count = 0usize;

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
            ["eval_link_cases", n] => {
                expected_link_cases = Some(parse_usize(n, line_no));
            }
            ["eval_hom_cases", n] => {
                expected_hom_cases = Some(parse_usize(n, line_no));
            }
            [
                "link",
                _case_idx,
                rows,
                cols,
                matrix_csv,
                z_csv,
                r_csv,
                expected_y_csv,
                expected_coeff_side_csv,
                expected_ct_y,
                expected_ct_side,
                identity,
            ] => {
                let rows = parse_usize(rows, line_no);
                let cols = parse_usize(cols, line_no);
                let matrix = parse_fq_matrix(matrix_csv, line_no);
                let z = parse_fq_vec(z_csv, line_no);
                let r = parse_fq_vec(r_csv, line_no);
                let expected_y = parse_fq_vec(expected_y_csv, line_no);
                let expected_coeff_side = parse_fq_vec(expected_coeff_side_csv, line_no);
                let expected_ct_y = Fq::from_u64(parse_u64(expected_ct_y, line_no));
                let expected_ct_side = Fq::from_u64(parse_u64(expected_ct_side, line_no));
                let identity = parse_u64(identity, line_no);

                assert_eq!(matrix.len(), rows, "link rows mismatch at line {line_no}");
                assert_eq!(z.len(), cols, "link z size mismatch at line {line_no}");
                for (ridx, row) in matrix.iter().enumerate() {
                    assert_eq!(
                        row.len(),
                        cols,
                        "link row width mismatch at line {line_no}, row {ridx}"
                    );
                    assert_eq!(
                        row.len() % D,
                        0,
                        "link row width not multiple of D at line {line_no}, row {ridx}"
                    );
                }

                let ys = bar_mz_ring(&matrix, &z);
                let weights = r_hat(&r, ys.len());
                let got_y = eval_ring_vec(&ys, &weights);
                let got_coeff_side = eval_coeff_rows(&coeff_rows_of_ring_vec(&ys), &weights);
                let got_ct_y = got_y.first().copied().unwrap_or(Fq::ZERO);
                let got_ct_side = dot_f(&ct_row(&ys), &weights);
                let got_identity = if eval_link_for_matrix(&matrix, &z, &r) {
                    1
                } else {
                    0
                };

                assert_eq!(got_y, expected_y, "link y mismatch at line {line_no}");
                assert_eq!(
                    got_coeff_side, expected_coeff_side,
                    "link coeff-side mismatch at line {line_no}"
                );
                assert_eq!(got_ct_y, expected_ct_y, "link ct(y) mismatch at line {line_no}");
                assert_eq!(
                    got_ct_side, expected_ct_side,
                    "link ct-side mismatch at line {line_no}"
                );
                assert_eq!(
                    got_identity, identity,
                    "link identity mismatch at line {line_no}"
                );

                link_count += 1;
            }
            [
                "hom",
                _case_idx,
                rows,
                cols,
                matrix_csv,
                z1_csv,
                z2_csv,
                r_csv,
                rho1,
                rho2,
                expected_y1_csv,
                expected_y2_csv,
                expected_y_lin_csv,
                expected_y_direct_csv,
                expected_ct_lin,
                expected_ct_formula,
                identity,
            ] => {
                let rows = parse_usize(rows, line_no);
                let cols = parse_usize(cols, line_no);
                let matrix = parse_fq_matrix(matrix_csv, line_no);
                let z1 = parse_fq_vec(z1_csv, line_no);
                let z2 = parse_fq_vec(z2_csv, line_no);
                let r = parse_fq_vec(r_csv, line_no);
                let rho1 = Fq::from_u64(parse_u64(rho1, line_no));
                let rho2 = Fq::from_u64(parse_u64(rho2, line_no));
                let expected_y1 = parse_fq_vec(expected_y1_csv, line_no);
                let expected_y2 = parse_fq_vec(expected_y2_csv, line_no);
                let expected_y_lin = parse_fq_vec(expected_y_lin_csv, line_no);
                let expected_y_direct = parse_fq_vec(expected_y_direct_csv, line_no);
                let expected_ct_lin = Fq::from_u64(parse_u64(expected_ct_lin, line_no));
                let expected_ct_formula = Fq::from_u64(parse_u64(expected_ct_formula, line_no));
                let identity = parse_u64(identity, line_no);

                assert_eq!(matrix.len(), rows, "hom rows mismatch at line {line_no}");
                assert_eq!(z1.len(), cols, "hom z1 size mismatch at line {line_no}");
                assert_eq!(z2.len(), cols, "hom z2 size mismatch at line {line_no}");
                for (ridx, row) in matrix.iter().enumerate() {
                    assert_eq!(
                        row.len(),
                        cols,
                        "hom row width mismatch at line {line_no}, row {ridx}"
                    );
                    assert_eq!(
                        row.len() % D,
                        0,
                        "hom row width not multiple of D at line {line_no}, row {ridx}"
                    );
                }

                let got_y1 = eval_bar_mz_at(&matrix, &z1, &r);
                let got_y2 = eval_bar_mz_at(&matrix, &z2, &r);
                let got_y_lin = vec_add(&vec_scale(rho1, &got_y1), &vec_scale(rho2, &got_y2));
                let got_y_direct = eval_bar_mz_at(&matrix, &lin_comb_2_vec(rho1, rho2, &z1, &z2), &r);
                let got_ct_lin = got_y_lin.first().copied().unwrap_or(Fq::ZERO);
                let got_ct_formula = rho1 * got_y1.first().copied().unwrap_or(Fq::ZERO)
                    + rho2 * got_y2.first().copied().unwrap_or(Fq::ZERO);
                let got_identity = if eval_hom2(&matrix, &z1, &z2, &r, rho1, rho2) {
                    1
                } else {
                    0
                };

                assert_eq!(got_y1, expected_y1, "hom y1 mismatch at line {line_no}");
                assert_eq!(got_y2, expected_y2, "hom y2 mismatch at line {line_no}");
                assert_eq!(
                    got_y_lin, expected_y_lin,
                    "hom y_lin mismatch at line {line_no}"
                );
                assert_eq!(
                    got_y_direct, expected_y_direct,
                    "hom y_direct mismatch at line {line_no}"
                );
                assert_eq!(
                    got_ct_lin, expected_ct_lin,
                    "hom ct(y_lin) mismatch at line {line_no}"
                );
                assert_eq!(
                    got_ct_formula, expected_ct_formula,
                    "hom ct formula mismatch at line {line_no}"
                );
                assert_eq!(
                    got_identity, identity,
                    "hom identity mismatch at line {line_no}"
                );

                hom_count += 1;
            }
            _ => panic!("invalid golden-vector line {line_no}: {line}"),
        }
    }

    assert!(seen_modulus, "missing modulus line");
    assert!(seen_d, "missing d line");
    assert_eq!(
        link_count,
        expected_link_cases.expect("missing eval_link_cases line"),
        "unexpected eval-link case count"
    );
    assert_eq!(
        hom_count,
        expected_hom_cases.expect("missing eval_hom_cases line"),
        "unexpected eval-hom case count"
    );
}
