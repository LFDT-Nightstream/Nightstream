//! Π_CCS Ajtai-tail coefficient kernel.
//!
//! Owns only the log-small tail after the row point is fixed. The row-phase
//! FE kernel remains in `pi_ccs_fe`; this module is loaded only by the tail
//! component path until the whole-phase Π_CCS hook consumes it.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::EmbeddedModuleError;

use crate::kernels::ajtai::RING_D;
use crate::kernels::goldilocks::Kx;

pub use pi_ccs_tail_kernels::LoadedModule as FeTailKernelModule;

pub fn load_fe_tail_kernels(ctx: &Arc<CudaContext>) -> Result<FeTailKernelModule, EmbeddedModuleError> {
    pi_ccs_tail_kernels::load(ctx)
}

#[allow(clippy::too_many_arguments)]
pub fn launch_ajtai_tail_round_coeffs(
    module: &FeTailKernelModule,
    stream: &Arc<CudaStream>,
    y_eval: &DeviceBuffer<u64>,
    header: &DeviceBuffer<u64>,
    mcs_meta: &DeviceBuffer<u64>,
    term_meta: &DeviceBuffer<u64>,
    term_vars: &DeviceBuffer<u64>,
    points: &DeviceBuffer<u64>,
    coeffs_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.ajtai_tail_round_coeffs(
        stream,
        LaunchConfig::for_num_elems(1),
        y_eval,
        header,
        mcs_meta,
        term_meta,
        term_vars,
        points,
        coeffs_out,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_ajtai_tail_round_partials_from_challenges(
    module: &FeTailKernelModule,
    stream: &Arc<CudaStream>,
    y_eval: &DeviceBuffer<u64>,
    header: &DeviceBuffer<u64>,
    header_offset: usize,
    points: &DeviceBuffer<u64>,
    challenges: &DeviceBuffer<u64>,
    partial_count: usize,
    partials: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    assert!(
        partial_count <= u32::MAX as usize,
        "Ajtai-tail partial count exceeds CUDA launch capacity"
    );
    if partial_count == 0 {
        return Ok(());
    }
    module.ajtai_tail_round_partials_from_challenges(
        stream,
        LaunchConfig::for_num_elems(partial_count as u32),
        y_eval,
        header,
        header_offset as u32,
        points,
        challenges,
        partial_count as u32,
        partials,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_ajtai_tail_round_reduce_from_challenges(
    module: &FeTailKernelModule,
    stream: &Arc<CudaStream>,
    y_eval: &DeviceBuffer<u64>,
    header: &DeviceBuffer<u64>,
    header_offset: usize,
    mcs_meta: &DeviceBuffer<u64>,
    term_meta: &DeviceBuffer<u64>,
    term_vars: &DeviceBuffer<u64>,
    points: &DeviceBuffer<u64>,
    challenges: &DeviceBuffer<u64>,
    inner_sums: &DeviceBuffer<u64>,
    coeffs_out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.ajtai_tail_round_reduce_from_challenges(
        stream,
        LaunchConfig::for_num_elems(1),
        y_eval,
        header,
        header_offset as u32,
        mcs_meta,
        term_meta,
        term_vars,
        points,
        challenges,
        inner_sums,
        coeffs_out,
    )
}

#[cuda_module]
pub mod pi_ccs_tail_kernels {
    use super::*;

    fn read_k_words(words: &[u64], word_offset: usize) -> Kx {
        if word_offset + 1 >= words.len() {
            return Kx::ZERO;
        }
        Kx::from_words(words[word_offset], words[word_offset + 1])
    }

    fn write_k_words(out: &mut DisjointSlice<u64>, k_index: usize, value: Kx) {
        let words = value.as_words();
        let base = k_index * 2;
        if base + 1 < out.len() {
            unsafe {
                *out.get_unchecked_mut(base) = words[0];
                *out.get_unchecked_mut(base + 1) = words[1];
            }
        }
    }

    fn write_tail_partial(out: &mut DisjointSlice<u64>, partial: usize, inner0: Kx, inner1: Kx) {
        let base = partial * 4;
        if base + 3 >= out.len() {
            return;
        }
        let a = inner0.as_words();
        let b = inner1.as_words();
        unsafe {
            *out.get_unchecked_mut(base) = a[0];
            *out.get_unchecked_mut(base + 1) = a[1];
            *out.get_unchecked_mut(base + 2) = b[0];
            *out.get_unchecked_mut(base + 3) = b[1];
        }
    }

    fn read_tail_partial(words: &[u64], partial: usize) -> (Kx, Kx) {
        let base = partial * 4;
        if base + 3 >= words.len() {
            return (Kx::ZERO, Kx::ZERO);
        }
        (
            Kx::from_words(words[base], words[base + 1]),
            Kx::from_words(words[base + 2], words[base + 3]),
        )
    }

    fn eq_lin(a: Kx, b: Kx) -> Kx {
        (Kx::ONE - a) * (Kx::ONE - b) + a * b
    }

    fn point_weight(points: &[u64], word_offset: usize, mask: usize, len: usize) -> Kx {
        let mut out = Kx::ONE;
        for bit in 0..len {
            let p = read_k_words(points, word_offset + 2 * bit);
            out = if ((mask >> bit) & 1) == 0 {
                out * (Kx::ONE - p)
            } else {
                out * p
            };
        }
        out
    }

    fn y_eval_at(y_eval: &[u64], witness: usize, mat: usize, lane: usize, t_mats: usize) -> Kx {
        if lane >= RING_D {
            return Kx::ZERO;
        }
        let base = witness * (2 * t_mats * RING_D) + 2 * mat * RING_D + lane;
        if base + RING_D >= y_eval.len() {
            return Kx::ZERO;
        }
        Kx::from_words(y_eval[base], y_eval[base + RING_D])
    }

    fn y_eval_lane_prefix_folded(
        y_eval: &[u64],
        witness: usize,
        mat: usize,
        lane_base: usize,
        prefix_len: usize,
        points: &[u64],
        prefix_word_offset: usize,
        t_mats: usize,
    ) -> Kx {
        let mut out = Kx::ZERO;
        let prefix_count = 1usize << prefix_len;
        for mask in 0..prefix_count {
            let lane = lane_base + mask;
            if lane < RING_D {
                let weight = point_weight(points, prefix_word_offset, mask, prefix_len);
                out = out + weight * y_eval_at(y_eval, witness, mat, lane, t_mats);
            }
        }
        out
    }

    fn tail_weighted_dot_affine(
        y_eval: &[u64],
        witness: usize,
        mat: usize,
        bit: usize,
        ell_d: usize,
        points: &[u64],
        alpha_word_offset: usize,
        prefix_word_offset: usize,
        t_mats: usize,
    ) -> (Kx, Kx) {
        let stride = 1usize << bit;
        let head_stride = stride << 1;
        let tail_len = 1usize << (ell_d - bit - 1);
        let tail_word_offset = alpha_word_offset + 2 * (bit + 1);
        let mut c0 = Kx::ZERO;
        let mut c1 = Kx::ZERO;
        for tail_mask in 0..tail_len {
            let weight = point_weight(points, tail_word_offset, tail_mask, ell_d - bit - 1);
            let lane0 = tail_mask * head_stride;
            let lane1 = lane0 + stride;
            let lo = y_eval_lane_prefix_folded(y_eval, witness, mat, lane0, bit, points, prefix_word_offset, t_mats);
            let hi = y_eval_lane_prefix_folded(y_eval, witness, mat, lane1, bit, points, prefix_word_offset, t_mats);
            c0 = c0 + weight * lo;
            c1 = c1 + weight * (hi - lo);
        }
        (c0, c1)
    }

    fn k_pow(base: Kx, exp: usize) -> Kx {
        let mut out = Kx::ONE;
        for _ in 0..exp {
            out = out * base;
        }
        out
    }

    fn eq_point(lhs: &[u64], lhs_offset: usize, rhs: &[u64], rhs_offset: usize, len: usize) -> Kx {
        let mut out = Kx::ONE;
        for i in 0..len {
            out = out
                * eq_lin(
                    read_k_words(lhs, lhs_offset + 2 * i),
                    read_k_words(rhs, rhs_offset + 2 * i),
                );
        }
        out
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn ajtai_tail_round_coeffs(
        y_eval: &[u64],
        header: &[u64],
        mcs_meta: &[u64],
        term_meta: &[u64],
        term_vars: &[u64],
        points: &[u64],
        mut coeffs_out: DisjointSlice<u64>,
    ) {
        if thread::index_1d().get() != 0 || header.len() < 15 {
            return;
        }
        let k_mcs = header[0] as usize;
        let k_total = header[1] as usize;
        let t_mats = header[2] as usize;
        let ell_d = header[3] as usize;
        let bit = header[4] as usize;
        let width = header[5] as usize;
        let has_inputs = header[6] != 0;
        if bit >= ell_d || width == 0 || t_mats == 0 || k_mcs > k_total {
            return;
        }

        for coeff in 0..width {
            write_k_words(&mut coeffs_out, coeff, Kx::ZERO);
        }

        let eq_beta_r = Kx::from_words(header[7], header[8]);
        let eq_r_inputs = Kx::from_words(header[9], header[10]);
        let gamma = Kx::from_words(header[11], header[12]);
        let gamma_to_k = Kx::from_words(header[13], header[14]);
        let alpha_word_offset = 0usize;
        let beta_word_offset = 2 * ell_d;
        let prefix_word_offset = 4 * ell_d;
        if points.len() < prefix_word_offset + 2 * bit {
            return;
        }

        let mut f_prime = Kx::ZERO;
        let num_terms = term_meta.len() / 4;
        for mcs in 0..k_mcs {
            let meta_base = mcs * 4;
            if meta_base + 1 >= mcs_meta.len() {
                return;
            }
            let gamma_mcs = Kx::from_words(mcs_meta[meta_base], mcs_meta[meta_base + 1]);
            if gamma_mcs == Kx::ZERO {
                continue;
            }
            let mut f_at_mcs = Kx::ZERO;
            for term in 0..num_terms {
                let tm_base = term * 4;
                let coeff = Kx::from_words(term_meta[tm_base], term_meta[tm_base + 1]);
                let var_off = term_meta[tm_base + 2] as usize;
                let var_count = term_meta[tm_base + 3] as usize;
                let mut term_acc = coeff;
                for v in 0..var_count {
                    let pair = (var_off + v) * 2;
                    if pair + 1 >= term_vars.len() {
                        return;
                    }
                    let mat = term_vars[pair] as usize;
                    let exp = term_vars[pair + 1] as usize;
                    if mat >= t_mats {
                        return;
                    }
                    term_acc = term_acc * k_pow(y_eval_at(y_eval, mcs, mat, 0, t_mats), exp);
                }
                f_at_mcs = f_at_mcs + term_acc;
            }
            f_prime = f_prime + gamma_mcs * f_at_mcs;
        }

        let mut eq_beta_pref = Kx::ONE;
        let mut eq_alpha_pref = Kx::ONE;
        for i in 0..bit {
            let prefix = read_k_words(points, prefix_word_offset + 2 * i);
            let beta = read_k_words(points, beta_word_offset + 2 * i);
            let alpha = read_k_words(points, alpha_word_offset + 2 * i);
            eq_beta_pref = eq_beta_pref * eq_lin(prefix, beta);
            eq_alpha_pref = eq_alpha_pref * eq_lin(prefix, alpha);
        }

        let beta_j = read_k_words(points, beta_word_offset + 2 * bit);
        let alpha_j = read_k_words(points, alpha_word_offset + 2 * bit);
        let beta0 = Kx::ONE - beta_j;
        let beta1 = beta_j + beta_j - Kx::ONE;
        let alpha0 = Kx::ONE - alpha_j;
        let alpha1 = alpha_j + alpha_j - Kx::ONE;

        let mut c0 = eq_beta_r * eq_beta_pref * beta0 * f_prime;
        let mut c1 = eq_beta_r * eq_beta_pref * beta1 * f_prime;
        let mut c2 = Kx::ZERO;

        if has_inputs && eq_r_inputs != Kx::ZERO && k_total > k_mcs {
            let mut inner0 = Kx::ZERO;
            let mut inner1 = Kx::ZERO;
            let mut gamma_k_pow_mat = Kx::ONE;
            for mat in 0..t_mats {
                let mut gamma_i = Kx::ONE;
                for witness in 0..k_total {
                    if witness >= k_mcs {
                        let coeff = gamma_i * gamma_k_pow_mat;
                        if coeff != Kx::ZERO {
                            let (dot0, dot1) = tail_weighted_dot_affine(
                                y_eval,
                                witness,
                                mat,
                                bit,
                                ell_d,
                                points,
                                alpha_word_offset,
                                prefix_word_offset,
                                t_mats,
                            );
                            inner0 = inner0 + coeff * dot0;
                            inner1 = inner1 + coeff * dot1;
                        }
                    }
                    gamma_i = gamma_i * gamma;
                }
                gamma_k_pow_mat = gamma_k_pow_mat * gamma_to_k;
            }
            let fac = eq_r_inputs * eq_alpha_pref * gamma_to_k;
            c0 = c0 + fac * alpha0 * inner0;
            c1 = c1 + fac * (alpha0 * inner1 + alpha1 * inner0);
            c2 = c2 + fac * alpha1 * inner1;
        }

        write_k_words(&mut coeffs_out, 0, c0);
        if width > 1 {
            write_k_words(&mut coeffs_out, 1, c1);
        }
        if width > 2 {
            write_k_words(&mut coeffs_out, 2, c2);
        }
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn ajtai_tail_round_partials_from_challenges(
        y_eval: &[u64],
        header: &[u64],
        header_offset: u32,
        points: &[u64],
        challenges: &[u64],
        partial_count: u32,
        mut partials_out: DisjointSlice<u64>,
    ) {
        let i = thread::index_1d().get();
        let partial_count = partial_count as usize;
        if i >= partial_count {
            return;
        }
        write_tail_partial(&mut partials_out, i, Kx::ZERO, Kx::ZERO);
        let header_offset = header_offset as usize;
        if header.len() < header_offset + 12 {
            return;
        }
        let header = &header[header_offset..];
        let k_mcs = header[0] as usize;
        let k_total = header[1] as usize;
        let t_mats = header[2] as usize;
        let ell_d = header[3] as usize;
        let ell_n = header[4] as usize;
        let bit = header[5] as usize;
        let has_inputs = header[7] != 0;
        if !has_inputs || bit >= ell_d || t_mats == 0 || k_mcs >= k_total || challenges.len() < 2 * (ell_n + bit) {
            return;
        }

        let gamma = Kx::from_words(header[8], header[9]);
        let gamma_to_k = Kx::from_words(header[10], header[11]);
        let alpha_word_offset = 0usize;
        let beta_r_word_offset = 4 * ell_d;
        let r_inputs_word_offset = beta_r_word_offset + 2 * ell_n;
        if points.len() < r_inputs_word_offset + 2 * ell_n {
            return;
        }

        let input_count = k_total - k_mcs;
        let prefix_len = 1usize << bit;
        let tail_len = 1usize << (ell_d - bit - 1);
        let lane_work = prefix_len * tail_len;
        if lane_work == 0 || i >= input_count * t_mats * lane_work {
            return;
        }
        let local = i % lane_work;
        let prefix_mask = local % prefix_len;
        let tail_mask = local / prefix_len;
        let mat = (i / lane_work) % t_mats;
        let witness = k_mcs + (i / (lane_work * t_mats));
        if witness >= k_total {
            return;
        }

        let prefix_word_offset = 2 * ell_n;
        let stride = 1usize << bit;
        let head_stride = stride << 1;
        let lane0 = tail_mask * head_stride + prefix_mask;
        let lane1 = lane0 + stride;
        let tail_word_offset = alpha_word_offset + 2 * (bit + 1);
        let tail_weight = point_weight(points, tail_word_offset, tail_mask, ell_d - bit - 1);
        let prefix_weight = point_weight(challenges, prefix_word_offset, prefix_mask, bit);
        let gamma_i = k_pow(gamma, witness);
        let gamma_j = k_pow(gamma_to_k, mat);
        let coeff = gamma_i * gamma_j * tail_weight * prefix_weight;
        if coeff == Kx::ZERO {
            return;
        }

        let lo = y_eval_at(y_eval, witness, mat, lane0, t_mats);
        let hi = y_eval_at(y_eval, witness, mat, lane1, t_mats);
        write_tail_partial(&mut partials_out, i, coeff * lo, coeff * (hi - lo));
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn ajtai_tail_round_reduce_from_challenges(
        y_eval: &[u64],
        header: &[u64],
        header_offset: u32,
        mcs_meta: &[u64],
        term_meta: &[u64],
        term_vars: &[u64],
        points: &[u64],
        challenges: &[u64],
        inner_sums: &[u64],
        mut coeffs_out: DisjointSlice<u64>,
    ) {
        let header_offset = header_offset as usize;
        if thread::index_1d().get() != 0 || header.len() < header_offset + 12 {
            return;
        }
        let header = &header[header_offset..];
        let k_mcs = header[0] as usize;
        let k_total = header[1] as usize;
        let t_mats = header[2] as usize;
        let ell_d = header[3] as usize;
        let ell_n = header[4] as usize;
        let bit = header[5] as usize;
        let width = header[6] as usize;
        let has_inputs = header[7] != 0;
        if bit >= ell_d || width == 0 || t_mats == 0 || k_mcs > k_total || challenges.len() < 2 * (ell_n + bit) {
            return;
        }

        for coeff in 0..width {
            write_k_words(&mut coeffs_out, coeff, Kx::ZERO);
        }

        let gamma_to_k = Kx::from_words(header[10], header[11]);
        let alpha_word_offset = 0usize;
        let beta_a_word_offset = 2 * ell_d;
        let beta_r_word_offset = 4 * ell_d;
        let r_inputs_word_offset = beta_r_word_offset + 2 * ell_n;
        if points.len() < r_inputs_word_offset + if has_inputs { 2 * ell_n } else { 0 } {
            return;
        }

        let eq_beta_r = eq_point(challenges, 0, points, beta_r_word_offset, ell_n);
        let eq_r_inputs = if has_inputs {
            eq_point(challenges, 0, points, r_inputs_word_offset, ell_n)
        } else {
            Kx::ZERO
        };

        let mut f_prime = Kx::ZERO;
        let num_terms = term_meta.len() / 4;
        for mcs in 0..k_mcs {
            let meta_base = mcs * 4;
            if meta_base + 1 >= mcs_meta.len() {
                return;
            }
            let gamma_mcs = Kx::from_words(mcs_meta[meta_base], mcs_meta[meta_base + 1]);
            if gamma_mcs == Kx::ZERO {
                continue;
            }
            let mut f_at_mcs = Kx::ZERO;
            for term in 0..num_terms {
                let tm_base = term * 4;
                let coeff = Kx::from_words(term_meta[tm_base], term_meta[tm_base + 1]);
                let var_off = term_meta[tm_base + 2] as usize;
                let var_count = term_meta[tm_base + 3] as usize;
                let mut term_acc = coeff;
                for v in 0..var_count {
                    let pair = (var_off + v) * 2;
                    if pair + 1 >= term_vars.len() {
                        return;
                    }
                    let mat = term_vars[pair] as usize;
                    let exp = term_vars[pair + 1] as usize;
                    if mat >= t_mats {
                        return;
                    }
                    term_acc = term_acc * k_pow(y_eval_at(y_eval, mcs, mat, 0, t_mats), exp);
                }
                f_at_mcs = f_at_mcs + term_acc;
            }
            f_prime = f_prime + gamma_mcs * f_at_mcs;
        }

        let prefix_word_offset = 2 * ell_n;
        let mut eq_beta_pref = Kx::ONE;
        let mut eq_alpha_pref = Kx::ONE;
        for i in 0..bit {
            let prefix = read_k_words(challenges, prefix_word_offset + 2 * i);
            let beta = read_k_words(points, beta_a_word_offset + 2 * i);
            let alpha = read_k_words(points, alpha_word_offset + 2 * i);
            eq_beta_pref = eq_beta_pref * eq_lin(prefix, beta);
            eq_alpha_pref = eq_alpha_pref * eq_lin(prefix, alpha);
        }

        let beta_j = read_k_words(points, beta_a_word_offset + 2 * bit);
        let alpha_j = read_k_words(points, alpha_word_offset + 2 * bit);
        let beta0 = Kx::ONE - beta_j;
        let beta1 = beta_j + beta_j - Kx::ONE;
        let alpha0 = Kx::ONE - alpha_j;
        let alpha1 = alpha_j + alpha_j - Kx::ONE;

        let mut c0 = eq_beta_r * eq_beta_pref * beta0 * f_prime;
        let mut c1 = eq_beta_r * eq_beta_pref * beta1 * f_prime;
        let mut c2 = Kx::ZERO;

        if has_inputs && eq_r_inputs != Kx::ZERO && k_total > k_mcs {
            let (inner0, inner1) = read_tail_partial(inner_sums, 0);
            let fac = eq_r_inputs * eq_alpha_pref * gamma_to_k;
            c0 = c0 + fac * alpha0 * inner0;
            c1 = c1 + fac * (alpha0 * inner1 + alpha1 * inner0);
            c2 = c2 + fac * alpha1 * inner1;
        }

        write_k_words(&mut coeffs_out, 0, c0);
        if width > 1 {
            write_k_words(&mut coeffs_out, 1, c1);
        }
        if width > 2 {
            write_k_words(&mut coeffs_out, 2, c2);
        }
    }
}
