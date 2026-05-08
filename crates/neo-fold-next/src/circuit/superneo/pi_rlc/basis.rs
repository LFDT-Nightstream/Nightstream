use super::*;

pub(super) fn dense_index(row: usize, col: usize, cols: usize, column_major: bool) -> usize {
    if column_major {
        col * D + row
    } else {
        row * cols + col
    }
}

pub(super) fn build_goldilocks_rot_basis_mats() -> Vec<Mat<F>> {
    let neg_phi = neo_reductions::RotRing::goldilocks()
        .phi_coeffs
        .iter()
        .map(|coeff| {
            if *coeff >= 0 {
                F::ZERO - F::from_u64(*coeff as u64)
            } else {
                F::from_u64((-*coeff) as u64)
            }
        })
        .collect::<Vec<_>>();
    let mut mats = Vec::with_capacity(D);
    for coeff_idx in 0..D {
        let mut col = vec![F::ZERO; D];
        col[coeff_idx] = F::ONE;
        let mut mat = Mat::zero(D, D, F::ZERO);
        for j in 0..D {
            for row in 0..D {
                mat[(row, j)] = col[row];
            }
            let tail = col[D - 1];
            let mut next = vec![F::ZERO; D];
            next[0] = tail * neg_phi[0];
            for row in 1..D {
                next[row] = col[row - 1] + tail * neg_phi[row];
            }
            col = next;
        }
        mats.push(mat);
    }
    mats
}

pub(super) fn basis_dense_f_scale(
    row: usize,
    col: usize,
    cols: usize,
    column_major: bool,
    child: &[F],
    coeff_idx: usize,
) -> F {
    let basis = &GOLDILOCKS_ROT_BASIS_MATS[coeff_idx];
    let mut acc = F::ZERO;
    for src_row in 0..D {
        let basis_coeff = basis[(row, src_row)];
        if basis_coeff == F::ZERO {
            continue;
        }
        let child_idx = dense_index(src_row, col, cols, column_major);
        let value = child[child_idx];
        acc += basis_coeff * value;
    }
    acc
}

pub(super) fn basis_k_row_scale(row: usize, child: &[K], coeff_idx: usize) -> (F, F) {
    let basis = &GOLDILOCKS_ROT_BASIS_MATS[coeff_idx];
    let mut acc_c0 = F::ZERO;
    let mut acc_c1 = F::ZERO;
    for src_row in 0..D {
        let basis_coeff = basis[(row, src_row)];
        if basis_coeff == F::ZERO {
            continue;
        }
        let coeffs = child[src_row].as_coeffs();
        acc_c0 += basis_coeff * coeffs[0];
        acc_c1 += basis_coeff * coeffs[1];
    }
    (acc_c0, acc_c1)
}
