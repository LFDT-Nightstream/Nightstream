use core::cmp::min;

use neo_ccs::{CcsMatrix, CcsStructure};
use neo_math::{ct, superneo_bar_block, KExtensions, Rq, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};

#[inline]
fn matrix_entry<Ff: Field + PrimeCharacteristicRing + Copy>(mat: &CcsMatrix<Ff>, row: usize, col: usize) -> Ff {
    if row >= mat.rows() || col >= mat.cols() {
        return Ff::ZERO;
    }
    match mat {
        CcsMatrix::Identity { .. } => {
            if row == col {
                Ff::ONE
            } else {
                Ff::ZERO
            }
        }
        CcsMatrix::Csc(csc) => {
            let s = csc.col_ptr[col];
            let e = csc.col_ptr[col + 1];
            match csc.row_idx[s..e].binary_search(&row) {
                Ok(idx) => csc.vals[s + idx],
                Err(_) => Ff::ZERO,
            }
        }
    }
}

#[inline]
fn as_base_field<Ff>(v: Ff) -> F
where
    Ff: Field + PrimeCharacteristicRing + Copy + Sync,
    K: From<Ff>,
{
    K::from(v).real()
}

/// Row dot-product via on-the-fly SuperNeo lift from an original matrix row.
pub fn superneo_row_dot_from_original<Ff>(mat: &CcsMatrix<Ff>, row: usize, z: &[K]) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    assert_eq!(
        mat.cols(),
        z.len(),
        "superneo_row_dot_from_original: column/vector length mismatch"
    );
    if row >= mat.rows() {
        return K::ZERO;
    }

    let blocks = z.len().div_ceil(D);
    let mut acc_re = F::ZERO;
    let mut acc_im = F::ZERO;

    for blk in 0..blocks {
        let base = blk * D;
        let mut a = [F::ZERO; D];
        let mut z_re = [F::ZERO; D];
        let mut z_im = [F::ZERO; D];

        for i in 0..D {
            a[i] = as_base_field(matrix_entry(mat, row, base + i));
            if base + i < z.len() {
                let [re, im] = z[base + i].as_coeffs();
                z_re[i] = re;
                z_im[i] = im;
            }
        }

        let a_ring = Rq(superneo_bar_block(a));
        acc_re += ct(&a_ring.mul(&Rq(z_re)));
        acc_im += ct(&a_ring.mul(&Rq(z_im)));
    }

    K::from_coeffs([acc_re, acc_im])
}

/// Evaluate `\tilde{(M z)}(r)` using transformed matrix rows and `ct` products.
pub fn eval_mle_transformed_matrix(mat_bar: &CcsMatrix<F>, z: &[K], chi_r: &[K], n_eff: usize) -> K {
    let row_cap = min(min(mat_bar.rows(), n_eff), chi_r.len());
    let mut acc = K::ZERO;
    for (row, &w) in chi_r.iter().take(row_cap).enumerate() {
        if w == K::ZERO {
            continue;
        }
        acc += w * super::superneo_row_dot_transformed_matrix(mat_bar, row, z);
    }
    acc
}

/// Evaluate `\tilde{(M z)}(r)` by lifting original rows through the SuperNeo `bar` transform.
pub fn eval_mle_superneo_from_original<Ff>(mat: &CcsMatrix<Ff>, z: &[K], chi_r: &[K], n_eff: usize) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let row_cap = min(min(mat.rows(), n_eff), chi_r.len());
    let mut acc = K::ZERO;
    for (row, &w) in chi_r.iter().take(row_cap).enumerate() {
        if w == K::ZERO {
            continue;
        }
        acc += w * superneo_row_dot_from_original(mat, row, z);
    }
    acc
}

/// Evaluate `\tilde{(M z)}(r)` directly from `M`.
pub fn eval_mle_direct_matrix<Ff>(mat: &CcsMatrix<Ff>, z: &[K], chi_r: &[K], n_eff: usize) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    assert_eq!(
        mat.cols(),
        z.len(),
        "eval_mle_direct_matrix: column/vector length mismatch"
    );
    let row_cap = min(mat.rows(), n_eff);
    let mut mz = vec![K::ZERO; row_cap];
    mat.add_mul_into(z, &mut mz, row_cap);

    let mut acc = K::ZERO;
    for (row, &w) in chi_r.iter().take(min(row_cap, chi_r.len())).enumerate() {
        if w == K::ZERO {
            continue;
        }
        acc += w * mz[row];
    }
    acc
}

#[inline]
pub fn is_superneo_compatible_shape(cols: usize) -> bool {
    cols > 0
}

pub fn should_enable_superneo_cache_default<Ff>(s: &CcsStructure<Ff>, _b: u32) -> bool {
    is_superneo_compatible_shape(s.m) && !s.matrices.is_empty()
}

pub fn eval_all_mats_transformed(s_bar: &CcsStructure<F>, z: &[K], chi_r: &[K], n_eff: usize) -> Vec<K> {
    let mut out = Vec::with_capacity(s_bar.matrices.len());
    for m in &s_bar.matrices {
        out.push(eval_mle_transformed_matrix(m, z, chi_r, n_eff));
    }
    out
}

pub fn eval_all_mats_direct<Ff>(s: &CcsStructure<Ff>, z: &[K], chi_r: &[K], n_eff: usize) -> Vec<K>
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    let mut out = Vec::with_capacity(s.matrices.len());
    for m in &s.matrices {
        out.push(eval_mle_direct_matrix(m, z, chi_r, n_eff));
    }
    out
}

pub fn eval_all_mats_superneo<Ff>(s: &CcsStructure<Ff>, z: &[K], chi_r: &[K], n_eff: usize) -> Vec<K>
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    let mut out = Vec::with_capacity(s.matrices.len());
    for m in &s.matrices {
        out.push(eval_mle_superneo_from_original(m, z, chi_r, n_eff));
    }
    out
}
