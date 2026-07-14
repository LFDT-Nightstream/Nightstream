//! Optimized Π_RLC mixing owns the witness-side random linear combination path.
//!
//! It does not own paper-exact cross-check formulas or DEC.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

const RLC_RING_MUL_COL_THRESHOLD: usize = 256;
const RLC_RING_SPLIT: usize = D / 3;
const RLC_RING_CHUNK_OUT: usize = 2 * RLC_RING_SPLIT - 1;
const RLC_RING_SPARSE_RHS_THRESHOLD: usize = D / 4;

fn add_sparse_rows<Ff>(acc: &mut Mat<Ff>, rho_data: &[Ff], rows: &[Vec<(usize, Ff)>], m: usize)
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
{
    let neg_one = Ff::ZERO - Ff::ONE;
    let add_row = |rr: usize, row_out: &mut [Ff]| {
        for (kk, nonzeros) in rows.iter().enumerate() {
            let coeff = rho_data[rr * D + kk];
            if coeff == Ff::ZERO {
                continue;
            }
            for &(column, value) in nonzeros {
                if value == Ff::ONE {
                    row_out[column] += coeff;
                } else if value == neg_one {
                    row_out[column] -= coeff;
                } else {
                    row_out[column] += coeff * value;
                }
            }
        }
    };

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if rayon::current_num_threads() > 1 {
        acc.as_mut_slice()
            .par_chunks_exact_mut(m)
            .enumerate()
            .for_each(|(rr, row_out)| add_row(rr, row_out));
        return;
    }

    for (rr, row_out) in acc.as_mut_slice().chunks_exact_mut(m).enumerate() {
        add_row(rr, row_out);
    }
}

fn left_mul_acc_optimized<Ff>(acc: &mut Mat<Ff>, rho: &Mat<Ff>, a: &Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
{
    debug_assert_eq!(rho.rows(), D);
    debug_assert_eq!(rho.cols(), D);
    debug_assert_eq!(a.rows(), D);
    debug_assert_eq!(acc.rows(), D);
    debug_assert_eq!(a.cols(), acc.cols());

    let m = acc.cols();
    let rho_data = rho.as_slice();
    if a.is_packed_signed_unit() {
        let mut row_nonzeros = (0..D)
            .map(|_| Vec::new())
            .collect::<Vec<Vec<(usize, Ff)>>>();
        for (row, nonzeros) in row_nonzeros.iter_mut().enumerate() {
            for column in 0..m {
                let value = a[(row, column)];
                if value != Ff::ZERO {
                    nonzeros.push((column, value));
                }
            }
        }
        add_sparse_rows(acc, rho_data, &row_nonzeros, m);
        return;
    }
    let a_data = a.as_slice();
    let neg_one = Ff::ZERO - Ff::ONE;

    if m >= 1024 {
        let total = D * m;
        let mut row_counts = [0usize; D];
        for kk in 0..D {
            let row = &a_data[kk * m..(kk + 1) * m];
            row_counts[kk] = row.iter().filter(|&&value| value != Ff::ZERO).count();
        }
        let total_nnz: usize = row_counts.iter().sum();

        // Sparse witnesses are common after DEC. In that case the dense
        // row-wise loop scans the same mostly-zero matrix once per output
        // row; building row nonzero lists once cuts the work from D^2*m
        // zero checks to D*nnz updates. Keep the threshold conservative so
        // dense SHA/F' traces stay on the locality-friendly dense path.
        if total_nnz > 0 && total_nnz * 8 <= total {
            let mut row_nonzeros: Vec<Vec<(usize, Ff)>> = row_counts
                .iter()
                .map(|&count| Vec::with_capacity(count))
                .collect();
            for kk in 0..D {
                let row = &a_data[kk * m..(kk + 1) * m];
                for (col, &value) in row.iter().enumerate() {
                    if value != Ff::ZERO {
                        row_nonzeros[kk].push((col, value));
                    }
                }
            }

            add_sparse_rows(acc, rho_data, &row_nonzeros, m);
            return;
        }
    }

    if m >= RLC_RING_MUL_COL_THRESHOLD {
        left_mul_acc_rotation_ring(acc, rho_data, a_data, m);
        return;
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if rayon::current_num_threads() > 1 {
            let acc_data = acc.as_mut_slice();
            const BLOCK_COLS: usize = 1024;
            acc_data
                .par_chunks_exact_mut(m)
                .enumerate()
                .for_each(|(rr, row_out)| {
                    for col0 in (0..m).step_by(BLOCK_COLS) {
                        let len = core::cmp::min(BLOCK_COLS, m - col0);
                        for kk in 0..D {
                            let coeff = rho_data[rr * D + kk];
                            if coeff == Ff::ZERO {
                                continue;
                            }
                            let in_off = kk * m + col0;
                            for t in 0..len {
                                let value = a_data[in_off + t];
                                if value == Ff::ZERO {
                                    continue;
                                }
                                if value == Ff::ONE {
                                    row_out[col0 + t] += coeff;
                                } else if value == neg_one {
                                    row_out[col0 + t] -= coeff;
                                } else {
                                    row_out[col0 + t] += coeff * value;
                                }
                            }
                        }
                    }
                });
            return;
        }
    }

    let acc_data = acc.as_mut_slice();
    const BLOCK_COLS: usize = 1024;
    for rr in 0..D {
        let row_out = &mut acc_data[rr * m..(rr + 1) * m];
        for col0 in (0..m).step_by(BLOCK_COLS) {
            let len = core::cmp::min(BLOCK_COLS, m - col0);
            for kk in 0..D {
                let coeff = rho_data[rr * D + kk];
                if coeff == Ff::ZERO {
                    continue;
                }
                let in_off = kk * m + col0;
                for t in 0..len {
                    let value = a_data[in_off + t];
                    if value == Ff::ZERO {
                        continue;
                    }
                    if value == Ff::ONE {
                        row_out[col0 + t] += coeff;
                    } else if value == neg_one {
                        row_out[col0 + t] -= coeff;
                    } else {
                        row_out[col0 + t] += coeff * value;
                    }
                }
            }
        }
    }
}

#[inline]
fn left_mul_acc_rotation_ring<Ff>(acc: &mut Mat<Ff>, rho_data: &[Ff], a_data: &[Ff], cols: usize)
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
{
    let mut rho_coeffs = [Ff::ZERO; D];
    for row in 0..D {
        rho_coeffs[row] = rho_data[row * D];
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if rayon::current_num_threads() > 1 && cols >= 1024 {
            let products: Vec<[Ff; D]> = (0..cols)
                .into_par_iter()
                .map(|col| {
                    let mut rhs = [Ff::ZERO; D];
                    let mut nnz = 0usize;
                    for row in 0..D {
                        let value = a_data[row * cols + col];
                        rhs[row] = value;
                        if value != Ff::ZERO {
                            nnz += 1;
                        }
                    }
                    if nnz <= RLC_RING_SPARSE_RHS_THRESHOLD {
                        return mul_phi_81_sparse_rhs(&rho_coeffs, &rhs);
                    }
                    mul_phi_81_toom3(&rho_coeffs, &rhs)
                })
                .collect();

            let acc_data = acc.as_mut_slice();
            for (col, product) in products.iter().enumerate() {
                for row in 0..D {
                    acc_data[row * cols + col] += product[row];
                }
            }
            return;
        }
    }

    let acc_data = acc.as_mut_slice();
    let mut rhs = [Ff::ZERO; D];
    for col in 0..cols {
        let mut nnz = 0usize;
        for row in 0..D {
            let value = a_data[row * cols + col];
            rhs[row] = value;
            if value != Ff::ZERO {
                nnz += 1;
            }
        }
        let product = if nnz <= RLC_RING_SPARSE_RHS_THRESHOLD {
            mul_phi_81_sparse_rhs(&rho_coeffs, &rhs)
        } else {
            mul_phi_81_toom3(&rho_coeffs, &rhs)
        };
        for row in 0..D {
            acc_data[row * cols + col] += product[row];
        }
    }
}

#[inline]
fn mul_phi_81_sparse_rhs<Ff>(lhs: &[Ff; D], rhs: &[Ff; D]) -> [Ff; D]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut out = [Ff::ZERO; D];
    let mut rot_col = *lhs;
    let mut rot_pos = 0usize;
    let neg_one = Ff::ZERO - Ff::ONE;

    for (pos, &scale) in rhs.iter().enumerate() {
        if scale == Ff::ZERO {
            continue;
        }
        advance_rot_col_phi_81(&mut rot_col, pos - rot_pos);
        if scale == Ff::ONE {
            for lane in 0..D {
                out[lane] += rot_col[lane];
            }
        } else if scale == neg_one {
            for lane in 0..D {
                out[lane] -= rot_col[lane];
            }
        } else {
            for lane in 0..D {
                out[lane] += rot_col[lane] * scale;
            }
        }
        rot_pos = pos;
    }

    out
}

#[inline]
fn advance_rot_col_phi_81<Ff>(col: &mut [Ff; D], delta: usize)
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    match delta {
        0 => {}
        1 => {
            let last = col[D - 1];
            for idx in (1..D).rev() {
                col[idx] = col[idx - 1];
            }
            col[0] = Ff::ZERO - last;
            col[D / 2] -= last;
        }
        _ => *col = mul_coeffs_by_monomial_phi_81(col, delta),
    }
}

#[inline]
fn mul_coeffs_by_monomial_phi_81<Ff>(input: &[Ff; D], j: usize) -> [Ff; D]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    debug_assert!(j < D);
    if j == 0 {
        return *input;
    }

    let mut out = [Ff::ZERO; D];
    let first_reduced = D - j;
    let first_wrap = (D + D / 2).saturating_sub(j).min(D);

    for i in 0..first_reduced {
        out[i + j] = input[i];
    }

    for i in first_reduced..first_wrap {
        let reduced = i + j - D;
        out[reduced] -= input[i];
        out[reduced + D / 2] -= input[i];
    }

    for i in first_wrap..D {
        out[i + j - D - D / 2] += input[i];
    }

    out
}

#[inline]
fn mul_phi_81_toom3<Ff>(lhs: &[Ff; D], rhs: &[Ff; D]) -> [Ff; D]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut tmp = mul_3way_karatsuba_54(lhs, rhs);
    reduce_mod_phi_81(&mut tmp);
    let mut out = [Ff::ZERO; D];
    out.copy_from_slice(&tmp[..D]);
    out
}

#[inline]
fn mul_schoolbook_chunk<Ff>(lhs: &[Ff; RLC_RING_SPLIT], rhs: &[Ff; RLC_RING_SPLIT]) -> [Ff; RLC_RING_CHUNK_OUT]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut out = [Ff::ZERO; RLC_RING_CHUNK_OUT];
    for i in 0..RLC_RING_SPLIT {
        let ai = lhs[i];
        for j in 0..RLC_RING_SPLIT {
            out[i + j] += ai * rhs[j];
        }
    }
    out
}

#[inline]
fn mul_3way_karatsuba_54<Ff>(lhs: &[Ff; D], rhs: &[Ff; D]) -> [Ff; 2 * D - 1]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut a0 = [Ff::ZERO; RLC_RING_SPLIT];
    let mut a1 = [Ff::ZERO; RLC_RING_SPLIT];
    let mut a2 = [Ff::ZERO; RLC_RING_SPLIT];
    let mut b0 = [Ff::ZERO; RLC_RING_SPLIT];
    let mut b1 = [Ff::ZERO; RLC_RING_SPLIT];
    let mut b2 = [Ff::ZERO; RLC_RING_SPLIT];
    a0.copy_from_slice(&lhs[..RLC_RING_SPLIT]);
    a1.copy_from_slice(&lhs[RLC_RING_SPLIT..2 * RLC_RING_SPLIT]);
    a2.copy_from_slice(&lhs[2 * RLC_RING_SPLIT..3 * RLC_RING_SPLIT]);
    b0.copy_from_slice(&rhs[..RLC_RING_SPLIT]);
    b1.copy_from_slice(&rhs[RLC_RING_SPLIT..2 * RLC_RING_SPLIT]);
    b2.copy_from_slice(&rhs[2 * RLC_RING_SPLIT..3 * RLC_RING_SPLIT]);

    let two = Ff::from_u64(2);
    let four = Ff::from_u64(4);
    let six = Ff::from_u64(6);
    let sixteen = Ff::from_u64(16);
    let half = two.inverse();
    let sixth = six.inverse();

    let a01 = add_chunk(&a0, &a1);
    let b01 = add_chunk(&b0, &b1);
    let a012 = add_chunk(&a01, &a2);
    let b012 = add_chunk(&b01, &b2);
    let am1 = add_chunk(&sub_chunk(&a0, &a1), &a2);
    let bm1 = add_chunk(&sub_chunk(&b0, &b1), &b2);
    let a2eval = add_scaled_chunk(&add_scaled_chunk(&a0, &a1, two), &a2, four);
    let b2eval = add_scaled_chunk(&add_scaled_chunk(&b0, &b1, two), &b2, four);

    let p0 = mul_schoolbook_chunk(&a0, &b0);
    let p1 = mul_schoolbook_chunk(&a012, &b012);
    let pm1 = mul_schoolbook_chunk(&am1, &bm1);
    let p2 = mul_schoolbook_chunk(&a2eval, &b2eval);
    let p4 = mul_schoolbook_chunk(&a2, &b2);

    let c0 = p0;
    let c4 = p4;
    let mut c2 = scale_chunk(&add_chunk(&p1, &pm1), half);
    sub_assign_chunk(&mut c2, &c0);
    sub_assign_chunk(&mut c2, &c4);

    let s = scale_chunk(&sub_chunk(&p1, &pm1), half);

    let mut c3 = p2;
    sub_assign_chunk(&mut c3, &c0);
    for i in 0..RLC_RING_CHUNK_OUT {
        c3[i] -= c2[i] * four;
        c3[i] -= c4[i] * sixteen;
    }
    c3 = scale_chunk(&sub_chunk(&c3, &scale_chunk(&s, two)), sixth);

    let mut c1 = s;
    sub_assign_chunk(&mut c1, &c3);

    let mut out = [Ff::ZERO; 2 * D - 1];
    add_assign_chunk_at(&mut out, 0, &c0);
    add_assign_chunk_at(&mut out, RLC_RING_SPLIT, &c1);
    add_assign_chunk_at(&mut out, 2 * RLC_RING_SPLIT, &c2);
    add_assign_chunk_at(&mut out, 3 * RLC_RING_SPLIT, &c3);
    add_assign_chunk_at(&mut out, 4 * RLC_RING_SPLIT, &c4);
    out
}

#[inline]
fn reduce_mod_phi_81<Ff>(coeffs: &mut [Ff; 2 * D - 1])
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    for i in (D..(2 * D - 1)).rev() {
        let t = coeffs[i];
        coeffs[i] = Ff::ZERO;
        coeffs[i - D] -= t;
        let idx_27 = i - (D / 2);
        if idx_27 < D {
            coeffs[idx_27] -= t;
        } else {
            coeffs[idx_27 - D] += t;
            if idx_27 - (D / 2) < D {
                coeffs[idx_27 - (D / 2)] += t;
            }
        }
    }
}

#[inline]
fn add_chunk<Ff, const N: usize>(lhs: &[Ff; N], rhs: &[Ff; N]) -> [Ff; N]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut out = [Ff::ZERO; N];
    for i in 0..N {
        out[i] = lhs[i] + rhs[i];
    }
    out
}

#[inline]
fn sub_chunk<Ff, const N: usize>(lhs: &[Ff; N], rhs: &[Ff; N]) -> [Ff; N]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut out = [Ff::ZERO; N];
    for i in 0..N {
        out[i] = lhs[i] - rhs[i];
    }
    out
}

#[inline]
fn scale_chunk<Ff, const N: usize>(lhs: &[Ff; N], scale: Ff) -> [Ff; N]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut out = [Ff::ZERO; N];
    for i in 0..N {
        out[i] = lhs[i] * scale;
    }
    out
}

#[inline]
fn add_scaled_chunk<Ff, const N: usize>(lhs: &[Ff; N], rhs: &[Ff; N], scale: Ff) -> [Ff; N]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut out = [Ff::ZERO; N];
    for i in 0..N {
        out[i] = lhs[i] + rhs[i] * scale;
    }
    out
}

#[inline]
fn sub_assign_chunk<Ff, const N: usize>(dst: &mut [Ff; N], src: &[Ff; N])
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    for i in 0..N {
        dst[i] -= src[i];
    }
}

#[inline]
fn add_assign_chunk_at<Ff, const N: usize>(dst: &mut [Ff; 2 * D - 1], offset: usize, src: &[Ff; N])
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    for i in 0..N {
        dst[offset + i] += src[i];
    }
}

#[inline]
fn mat_is_zero<Ff>(m: &Mat<Ff>) -> bool
where
    Ff: Field + Copy,
{
    if let Some(value) = m.virtual_constant_value() {
        return *value == Ff::ZERO;
    }
    if let Some(nonzero) = m.packed_signed_unit_nonzero_count() {
        return nonzero == 0;
    }
    m.as_slice().iter().all(|&entry| entry == Ff::ZERO)
}

/// The witness half of Π_RLC: `Z_mix = Σ ρ_i · Z_i`. Split out so device
/// backends can own this (the bulk-data cost) while `rlc_combine_claims`
/// keeps the small claim algebra on the host.
pub fn rlc_mix_witnesses<Ff>(s_m: usize, rhos: &[Mat<Ff>], Zs: &[&Mat<Ff>]) -> Mat<Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
{
    assert!(!Zs.is_empty(), "Π_RLC(optimized): need at least one witness");
    assert_eq!(rhos.len(), Zs.len(), "Π_RLC: |rhos| must equal |Zs|");
    let z_cols = Zs[0].cols();
    for (idx, z) in Zs.iter().enumerate() {
        crate::common::validate_superneo_witness_mat(*z, s_m)
            .unwrap_or_else(|e| panic!("Π_RLC(optimized): invalid witness shape at input {idx}: {e}"));
        assert_eq!(
            z.cols(),
            z_cols,
            "Π_RLC(optimized): all witness mats must share packed width"
        );
    }

    let mut Z = Mat::zero(D, z_cols, Ff::ZERO);
    for (rho, z_in) in rhos.iter().zip(Zs.iter()) {
        if mat_is_zero(z_in) {
            continue;
        }
        left_mul_acc_optimized(&mut Z, rho, z_in);
    }
    Z
}

/// The claim half of Π_RLC: every combined-CE field except the witness and
/// the commitment (`out.c` is a placeholder copy of input 0's commitment;
/// callers overwrite it via their commitment mixer).
pub fn rlc_combine_claims<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    ell_d: usize,
) -> CeClaim<Cmt, Ff, K>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    assert!(!me_inputs.is_empty(), "Π_RLC(optimized): need at least one input");
    let k1 = me_inputs.len();
    assert_eq!(rhos.len(), k1, "Π_RLC: |rhos| must equal |inputs|");
    crate::common::validate_rhos_are_rotation_matrices(params, rhos, "Π_RLC(optimized): rhos")
        .unwrap_or_else(|e| panic!("Π_RLC(optimized): invalid rho set: {e}"));

    let d_pad = 1usize << ell_d;
    let t_core = s.t();
    let m_in = me_inputs[0].m_in;
    let r = me_inputs[0].r.clone();
    let aux_len = me_inputs[0].aux_openings.len();
    for (idx, inst) in me_inputs.iter().enumerate() {
        assert_eq!(
            inst.aux_openings.len(),
            aux_len,
            "Π_RLC: aux_openings.len mismatch at input {idx}"
        );
    }

    #[cfg(feature = "perf-timers")]
    let t_y_ring = std::time::Instant::now();
    let mut y_ring: Vec<Vec<K>> = Vec::with_capacity(t_core);
    for j in 0..t_core {
        let mut yj_acc = vec![K::ZERO; d_pad];
        for i in 0..k1 {
            let yi = &me_inputs[i].y_ring[j];
            debug_assert!(yi.len() >= D, "ME.y_ring[{j}] must have length >= D");
            let rho = &rhos[i];
            for rr in 0..D {
                let mut acc_rr = K::ZERO;
                for kk in 0..D {
                    acc_rr += K::from(rho[(rr, kk)]) * yi[kk];
                }
                yj_acc[rr] += acc_rr;
            }
        }
        y_ring.push(yj_acc);
    }
    #[cfg(feature = "perf-timers")]
    let y_ring_s = t_y_ring.elapsed().as_secs_f64();

    let wants_nc_channel = !(me_inputs[0].s_col.is_empty() && me_inputs[0].y_zcol.is_empty());
    if wants_nc_channel {
        assert!(
            !me_inputs[0].s_col.is_empty() && !me_inputs[0].y_zcol.is_empty(),
            "Π_RLC: incomplete NC channel on input 0 (expected both s_col and y_zcol)"
        );
        for (idx, inst) in me_inputs.iter().enumerate() {
            assert_eq!(inst.s_col, me_inputs[0].s_col, "Π_RLC: s_col mismatch at input {idx}");
            assert_eq!(
                inst.y_zcol.len(),
                d_pad,
                "Π_RLC: y_zcol len mismatch at input {idx} (expected {d_pad}, got {})",
                inst.y_zcol.len()
            );
        }
    }

    #[cfg(feature = "perf-timers")]
    let t_ct = std::time::Instant::now();
    let ct = crate::common::ct_from_y_ring_for_ccs_m(&y_ring, params, s.m);
    #[cfg(feature = "perf-timers")]
    let ct_s = t_ct.elapsed().as_secs_f64();

    #[cfg(feature = "perf-timers")]
    let t_aux = std::time::Instant::now();
    let mut aux_openings = vec![K::ZERO; aux_len];
    for (rho, inst) in rhos.iter().zip(me_inputs.iter()) {
        let w = K::from(rho[(0, 0)]);
        for (dst, src) in aux_openings.iter_mut().zip(inst.aux_openings.iter()) {
            *dst += w * *src;
        }
    }
    #[cfg(feature = "perf-timers")]
    let aux_s = t_aux.elapsed().as_secs_f64();

    #[cfg(feature = "perf-timers")]
    let t_x = std::time::Instant::now();
    let mut X = Mat::zero(D, m_in, Ff::ZERO);
    for (rho, inst) in rhos.iter().zip(me_inputs.iter()) {
        left_mul_acc_optimized(&mut X, rho, &inst.X);
    }
    #[cfg(feature = "perf-timers")]
    let x_s = t_x.elapsed().as_secs_f64();

    #[cfg(feature = "perf-timers")]
    let t_y_zcol = std::time::Instant::now();
    let y_zcol = if wants_nc_channel {
        let mut acc = vec![K::ZERO; d_pad];
        for i in 0..k1 {
            for rr in 0..D {
                let mut sum = K::ZERO;
                for kk in 0..D {
                    sum += K::from(rhos[i][(rr, kk)]) * me_inputs[i].y_zcol[kk];
                }
                acc[rr] += sum;
            }
        }
        acc
    } else {
        Vec::new()
    };
    #[cfg(feature = "perf-timers")]
    let y_zcol_s = t_y_zcol.elapsed().as_secs_f64();

    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-rlc] y_ring {:>7.2}s ct {:>7.2}s aux {:>7.2}s X_mix {:>7.2}s y_zcol {:>7.2}s",
        y_ring_s, ct_s, aux_s, x_s, y_zcol_s,
    );

    CeClaim::<Cmt, Ff, K> {
        adv: None,
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: me_inputs[0].c.clone(),
        X,
        r,
        s_col: me_inputs[0].s_col.clone(),
        y_ring,
        ct,
        aux_openings,
        y_zcol,
        m_in,
        fold_digest: me_inputs[0].fold_digest,
    }
}

fn rlc_reduction_optimized_from_refs<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    Zs: &[&Mat<Ff>],
    ell_d: usize,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    assert_eq!(Zs.len(), me_inputs.len(), "Π_RLC: |Zs| must equal |inputs|");
    let Z = rlc_mix_witnesses(s.m, rhos, Zs);
    let out = rlc_combine_claims(s, params, rhos, me_inputs, ell_d);
    (out, Z)
}

pub fn rlc_reduction_optimized<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    Zs: &[Mat<Ff>],
    ell_d: usize,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    let z_refs: Vec<&Mat<Ff>> = Zs.iter().collect();
    rlc_reduction_optimized_from_refs::<Ff>(s, params, rhos, me_inputs, &z_refs, ell_d)
}

pub fn rlc_reduction_optimized_with_commit_mix<Ff, Comb>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    Zs: &[&Mat<Ff>],
    ell_d: usize,
    combine_commit: Comb,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
    Comb: Fn(&[Mat<Ff>], &[Cmt]) -> Cmt,
{
    #[cfg(feature = "perf-timers")]
    let t_core = std::time::Instant::now();
    let (mut out, Z) = rlc_reduction_optimized_from_refs::<Ff>(s, params, rhos, me_inputs, Zs, ell_d);
    #[cfg(feature = "perf-timers")]
    let core_s = t_core.elapsed().as_secs_f64();
    #[cfg(feature = "perf-timers")]
    let t_commit = std::time::Instant::now();
    let inputs_c: Vec<Cmt> = me_inputs.iter().map(|m| m.c.clone()).collect();
    out.c = combine_commit(rhos, &inputs_c);
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-rlc] core {:>7.2}s commit_mix {:>7.2}s inputs={}",
        core_s,
        t_commit.elapsed().as_secs_f64(),
        me_inputs.len(),
    );
    (out, Z)
}
