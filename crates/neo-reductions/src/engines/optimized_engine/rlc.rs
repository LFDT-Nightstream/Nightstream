use crate::engines::RlcMixer;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

fn add_signed_unit_columns<Ff>(acc: &mut Mat<Ff>, rho_data: &[Ff], positive: &[u64], negative: &[u64])
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
{
    let m = acc.cols();
    let add_row = |rr: usize, row_out: &mut [Ff]| {
        let coefficients = &rho_data[rr * D..(rr + 1) * D];
        let mut nonzero_coefficients = 0u64;
        for (lane, &coefficient) in coefficients.iter().enumerate() {
            if coefficient != Ff::ZERO {
                nonzero_coefficients |= 1u64 << lane;
            }
        }
        for (column, value) in row_out.iter_mut().enumerate() {
            let mut mask = positive[column] & nonzero_coefficients;
            while mask != 0 {
                *value += coefficients[mask.trailing_zeros() as usize];
                mask &= mask - 1;
            }
            mask = negative[column] & nonzero_coefficients;
            while mask != 0 {
                *value -= coefficients[mask.trailing_zeros() as usize];
                mask &= mask - 1;
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
    if m == 0 {
        return;
    }
    let rho_data = rho.as_slice();
    if let Some(&constant) = a.virtual_constant_value() {
        if constant == Ff::ZERO {
            return;
        }
        let add_row = |rr: usize, row_out: &mut [Ff]| {
            let mut value = Ff::ZERO;
            for kk in 0..D {
                value += rho_data[rr * D + kk] * constant;
            }
            for entry in row_out {
                *entry += value;
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
        return;
    }
    if let Some((positive, negative)) = a.packed_signed_unit_column_masks() {
        add_signed_unit_columns(acc, rho_data, positive, negative);
        return;
    }
    left_mul_acc_rotation_ring(acc, rho_data, a.as_slice(), m);
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
        if rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none() {
            let products: Vec<[Ff; D]> = (0..cols)
                .into_par_iter()
                .map(|col| {
                    let mut rhs = [Ff::ZERO; D];
                    for row in 0..D {
                        rhs[row] = a_data[row * cols + col];
                    }
                    mul_negacyclic(&rho_coeffs, &rhs)
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
        for row in 0..D {
            rhs[row] = a_data[row * cols + col];
        }
        let product = mul_negacyclic(&rho_coeffs, &rhs);
        for row in 0..D {
            acc_data[row * cols + col] += product[row];
        }
    }
}

#[inline]
fn mul_negacyclic<Ff>(lhs: &[Ff; D], rhs: &[Ff; D]) -> [Ff; D]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut out = [Ff::ZERO; D];
    for (i, &left) in lhs.iter().enumerate() {
        if left == Ff::ZERO {
            continue;
        }
        for (j, &right) in rhs.iter().enumerate() {
            if i + j < D {
                out[i + j] += left * right;
            } else {
                out[i + j - D] -= left * right;
            }
        }
    }
    out
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

pub fn rlc_mix_witnesses<Ff>(s_m: usize, rhos: &[Mat<Ff>], Zs: &[&Mat<Ff>]) -> Mat<Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
{
    assert!(!Zs.is_empty(), "Π_RLC(optimized): need at least one witness");
    assert_eq!(rhos.len(), Zs.len(), "Π_RLC: |rhos| must equal |Zs|");
    let z_cols = Zs[0].cols();
    for (idx, z) in Zs.iter().enumerate() {
        crate::common::validate_superneo_witness_shape(*z, s_m)
            .unwrap_or_else(|e| panic!("Π_RLC(optimized): invalid witness shape at input {idx}: {e}"));
        assert_eq!(
            z.cols(),
            z_cols,
            "Π_RLC(optimized): all witness mats must share packed width"
        );
    }

    if let Some(witness) = light_popcount_mix(rhos, Zs) {
        return witness;
    }
    if let Some(witness) = packed_popcount_mix(rhos, Zs, z_cols) {
        return witness;
    }
    mix_validated(rhos, Zs, z_cols)
}

fn mix_validated<Ff>(rhos: &[Mat<Ff>], Zs: &[&Mat<Ff>], z_cols: usize) -> Mat<Ff>
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
{
    let mut Z = Mat::zero(D, z_cols, Ff::ZERO);
    for (rho, z_in) in rhos.iter().zip(Zs.iter()) {
        if mat_is_zero(z_in) {
            continue;
        }
        left_mul_acc_optimized(&mut Z, rho, z_in);
    }
    Z
}

const MAX_PACKED_SOURCES: usize = 32;
const _: () = assert!(D == u64::BITS as usize && MAX_PACKED_SOURCES * 2 * D <= i16::MAX as usize);

#[derive(Clone, Copy, Default)]
struct CoefficientMasks {
    negative: u64,
    units: u64,
    doubles: u64,
}

impl CoefficientMasks {
    fn rows<Ff: PrimeField64>(rho: &Mat<Ff>) -> Option<[Self; D]> {
        if rho.rows() != D || rho.cols() != D || rho.is_packed_signed_unit() || rho.is_virtual_constant() {
            return None;
        }
        let mut rows = [Self::default(); D];
        for (row, masks) in rho.as_slice().chunks_exact(D).zip(&mut rows) {
            for (lane, coefficient) in row.iter().enumerate() {
                let canonical = coefficient.as_canonical_u64();
                let complement = Ff::ORDER_U64 - canonical;
                let bit = 1u64 << lane;
                match canonical.min(complement) {
                    0 => continue,
                    1 => masks.units |= bit,
                    2 => masks.doubles |= bit,
                    _ => return None,
                }
                if complement < canonical {
                    masks.negative |= bit;
                }
            }
        }
        Some(rows)
    }

    fn dot(self, positive: u64, negative: u64) -> i16 {
        let occupied = positive | negative;
        let flipped = self.negative ^ negative;
        let count_signed = |mask: u64| mask.count_ones() as i16 - 2 * (mask & flipped).count_ones() as i16;
        count_signed(occupied & self.units) + 2 * count_signed(occupied & self.doubles)
    }
}

fn packed_popcount_mix<Ff: PrimeField64>(rhos: &[Mat<Ff>], witnesses: &[&Mat<Ff>], columns: usize) -> Option<Mat<Ff>> {
    if columns == 0 || !(3..=MAX_PACKED_SOURCES).contains(&rhos.len()) {
        return None;
    }
    let mut sources = Vec::new();
    for (rho, witness) in rhos.iter().zip(witnesses) {
        if witness.virtual_constant_value() == Some(&Ff::ZERO) {
            continue;
        }
        let (positive, negative) = witness.packed_signed_unit_column_masks()?;
        sources.push((positive, negative, CoefficientMasks::rows(rho)?));
    }

    let mut out = Mat::zero(D, columns, Ff::ZERO);
    let mix_row = |(row, output): (usize, &mut [Ff])| {
        const TILE: usize = 256;
        for (tile, chunk) in output.chunks_mut(TILE).enumerate() {
            let mut sums = [0i16; TILE];
            let start = tile * TILE;
            for (positive, negative, coefficients) in &sources {
                let masks = coefficients[row];
                for ((sum, &pos), &neg) in sums[..chunk.len()]
                    .iter_mut()
                    .zip(&positive[start..])
                    .zip(&negative[start..])
                {
                    *sum += masks.dot(pos, neg);
                }
            }
            for (value, sum) in chunk.iter_mut().zip(sums) {
                *value = Ff::from_i32(i32::from(sum));
            }
        }
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    out.as_mut_slice()
        .par_chunks_exact_mut(columns)
        .enumerate()
        .for_each(mix_row);
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    out.as_mut_slice()
        .chunks_exact_mut(columns)
        .enumerate()
        .for_each(mix_row);
    Some(out)
}

fn light_popcount_mix<Ff>(rhos: &[Mat<Ff>], witnesses: &[&Mat<Ff>]) -> Option<Mat<Ff>>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
{
    let ([rho, weight], [fresh, running]) = (rhos, witnesses) else {
        return None;
    };
    let columns = fresh.cols();
    if columns == 0 || weight.rows() != D || !weight.is_identity() {
        return None;
    }
    if running.is_packed_signed_unit() {
        return None;
    }
    let (positive, negative) = fresh.packed_signed_unit_column_masks()?;
    let coefficients = CoefficientMasks::rows(rho)?;
    let mut out = Mat::zero(D, columns, Ff::ZERO);
    let constant = running.virtual_constant_value().copied();
    let previous = if constant.is_none() { running.as_slice() } else { &[] };
    let add = |row: usize, output: &mut [Ff]| {
        let masks = coefficients[row];
        let prior = if constant.is_none() {
            &previous[row * columns..(row + 1) * columns]
        } else {
            &[]
        };
        for (column, entry) in output.iter_mut().enumerate() {
            let sum = masks.dot(positive[column], negative[column]);
            *entry = constant.unwrap_or_else(|| prior[column]) + Ff::from_i32(i32::from(sum));
        }
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    if rayon::current_num_threads() > 1 {
        out.as_mut_slice()
            .par_chunks_exact_mut(columns)
            .enumerate()
            .for_each(|(row, output)| add(row, output));
        return Some(out);
    }
    for (row, output) in out.as_mut_slice().chunks_exact_mut(columns).enumerate() {
        add(row, output);
    }
    Some(out)
}

#[cfg(test)]
#[path = "../../../tests/unit/rlc_popcount.rs"]
mod popcount_tests;

pub fn rlc_combine_claims<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    ell_d: usize,
    c: Cmt,
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
    let matrix_count = me_inputs[0].eval_a.len();
    assert_eq!(matrix_count, s.t(), "PiRLC Eval_A count mismatch");
    let m_in = me_inputs[0].m_in;
    let r = me_inputs[0].r.clone();
    for (idx, inst) in me_inputs.iter().enumerate() {
        assert_eq!(inst.eval_k.len(), d_pad, "PiRLC: Eval_K width mismatch at input {idx}");
        assert_eq!(
            inst.eval_a.len(),
            matrix_count,
            "PiRLC: Eval_A count mismatch at input {idx}"
        );
    }

    #[cfg(feature = "perf-timers")]
    let t_evaluations = std::time::Instant::now();
    let combine = |select: &dyn Fn(&CeClaim<Cmt, Ff, K>) -> &[K]| {
        let mut result = vec![K::ZERO; d_pad];
        for i in 0..k1 {
            let input = select(&me_inputs[i]);
            debug_assert!(input.len() >= D, "PiRLC evaluation must have length >= D");
            let rho = &rhos[i];
            for row in 0..D {
                let mut value = K::ZERO;
                for column in 0..D {
                    value += K::from(rho[(row, column)]) * input[column];
                }
                result[row] += value;
            }
        }
        result
    };
    let eval_k = combine(&|claim| &claim.eval_k);
    let mut eval_a = Vec::with_capacity(matrix_count);
    for matrix in 0..matrix_count {
        let mut yj_acc = vec![K::ZERO; d_pad];
        for i in 0..k1 {
            let yi = &me_inputs[i].eval_a[matrix];
            debug_assert!(yi.len() >= D, "PiRLC Eval_A[{matrix}] must have length >= D");
            let rho = &rhos[i];
            for rr in 0..D {
                let mut acc_rr = K::ZERO;
                for kk in 0..D {
                    acc_rr += K::from(rho[(rr, kk)]) * yi[kk];
                }
                yj_acc[rr] += acc_rr;
            }
        }
        eval_a.push(yj_acc);
    }
    #[cfg(feature = "perf-timers")]
    let evaluations_s = t_evaluations.elapsed().as_secs_f64();

    #[cfg(feature = "perf-timers")]
    let t_x = std::time::Instant::now();
    let mut X = Mat::zero(D, me_inputs[0].X.cols(), Ff::ZERO);
    for (rho, inst) in rhos.iter().zip(me_inputs.iter()) {
        left_mul_acc_optimized(&mut X, rho, &inst.X);
    }
    #[cfg(feature = "perf-timers")]
    let x_s = t_x.elapsed().as_secs_f64();

    #[cfg(feature = "perf-timers")]
    eprintln!("[pi-rlc] evaluations {:>7.2}s X_mix {:>7.2}s", evaluations_s, x_s,);

    CeClaim::<Cmt, Ff, K> {
        adv: None,
        c,
        X,
        r,
        eval_k,
        eval_a,
        m_in,
        fold_digest: me_inputs[0].fold_digest,
    }
}

pub fn rlc_reduction_optimized_with_commit_mix<Ff>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    Zs: &[&Mat<Ff>],
    ell_d: usize,
    combine_commit: RlcMixer<Ff>,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    rlc_reduction_optimized_with_mixers(
        s,
        params,
        rhos,
        me_inputs,
        Zs,
        ell_d,
        combine_commit,
        |rhos, witnesses| rlc_mix_witnesses(s.m(), rhos, witnesses),
    )
}

pub fn rlc_reduction_optimized_with_mixers<Ff, MixWitness>(
    s: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    me_inputs: &[CeClaim<Cmt, Ff, K>],
    Zs: &[&Mat<Ff>],
    ell_d: usize,
    combine_commit: RlcMixer<Ff>,
    mix_witnesses: MixWitness,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
    MixWitness: Fn(&[Mat<Ff>], &[&Mat<Ff>]) -> Mat<Ff>,
{
    #[cfg(feature = "perf-timers")]
    let t_core = std::time::Instant::now();
    assert_eq!(Zs.len(), me_inputs.len(), "Pi_RLC: |Zs| must equal |inputs|");
    #[cfg(feature = "perf-timers")]
    let t_commit = std::time::Instant::now();
    let inputs_c: Vec<Cmt> = me_inputs.iter().map(|m| m.c.clone()).collect();
    let c = combine_commit(rhos, &inputs_c);
    #[cfg(feature = "perf-timers")]
    let commit_s = t_commit.elapsed().as_secs_f64();
    let Z = mix_witnesses(rhos, Zs);
    let out = rlc_combine_claims(s, params, rhos, me_inputs, ell_d, c);
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[pi-rlc] core {:>7.2}s commit_mix {:>7.2}s inputs={}",
        t_core.elapsed().as_secs_f64() - commit_s,
        commit_s,
        me_inputs.len(),
    );
    (out, Z)
}
