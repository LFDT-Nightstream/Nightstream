//! The kernels of `gkr.metal`: the rounds of a batched fraction-sum GKR
//! layer over the cubic extension Gl[X] / (X^3 - X - 1). Values are three
//! words per element. Tables are reached through a list of device
//! addresses; a folded table keeps its entries in place with a stride that
//! doubles every round.

use cuda_device::{cuda_module, kernel, thread, SharedArray};

use crate::kernels::field::{gl_add, gl_from_word, gl_mul, gl_sub};

pub(crate) const GKR_PARTS: usize = 16;
pub(crate) const GKR_NODES: usize = 4;
pub(crate) const GKR_THREADS: usize = 256;
/// `[half, stride, trees, degree, μ (3)]`, then per tree `[factors, first
/// part, weight (3)]`.
pub(crate) const GKR_SHAPE_WORDS: usize = 7 + 5 * GKR_PARTS;

#[derive(Clone, Copy)]
struct Ext {
    c0: u64,
    c1: u64,
    c2: u64,
}

const EXT_ZERO: Ext = Ext { c0: 0, c1: 0, c2: 0 };

#[inline(always)]
fn ext_add(a: Ext, b: Ext) -> Ext {
    Ext {
        c0: gl_add(a.c0, b.c0),
        c1: gl_add(a.c1, b.c1),
        c2: gl_add(a.c2, b.c2),
    }
}

#[inline(always)]
fn ext_sub(a: Ext, b: Ext) -> Ext {
    Ext {
        c0: gl_sub(a.c0, b.c0),
        c1: gl_sub(a.c1, b.c1),
        c2: gl_sub(a.c2, b.c2),
    }
}

/// `X^3 = X + 1`.
#[inline(always)]
fn ext_mul(a: Ext, b: Ext) -> Ext {
    let b02 = gl_add(b.c0, b.c2);
    let b12 = gl_add(b.c1, b.c2);
    Ext {
        c0: gl_add(gl_add(gl_mul(a.c0, b.c0), gl_mul(a.c1, b.c2)), gl_mul(a.c2, b.c1)),
        c1: gl_add(gl_add(gl_mul(a.c0, b.c1), gl_mul(a.c1, b02)), gl_mul(a.c2, b12)),
        c2: gl_add(gl_add(gl_mul(a.c0, b.c2), gl_mul(a.c1, b.c1)), gl_mul(a.c2, b02)),
    }
}

/// # Safety
///
/// `table` holds element `index`.
#[inline(always)]
unsafe fn ext_load(table: *const u64, index: u64) -> Ext {
    let at = 3 * index as usize;
    // SAFETY: the caller's element.
    unsafe {
        Ext {
            c0: gl_from_word(*table.add(at)),
            c1: gl_from_word(*table.add(at + 1)),
            c2: gl_from_word(*table.add(at + 2)),
        }
    }
}

#[inline(always)]
fn ext_param(words: &[u64], at: usize) -> Ext {
    Ext {
        c0: words[at],
        c1: words[at + 1],
        c2: words[at + 2],
    }
}

/// Move each line in `value` from node `j - 1` to node `j` of 0, 2, 3, ...
#[inline(always)]
fn ext_step(value: &mut [Ext], slope: &[Ext], count: usize, j: usize) {
    if j == 0 {
        return;
    }
    for i in 0..count {
        value[i] = ext_add(value[i], slope[i]);
        if j == 1 {
            value[i] = ext_add(value[i], slope[i]);
        }
    }
}

#[cuda_module]
pub mod gkr_kernels {
    use super::*;

    /// Bind the lowest variable to `r`: `t[2k·s] = t[2k·s] + r·(t[(2k+1)·s]
    /// - t[2k·s])` for `k < count`. Shape: `[count, stride, r (3)]`.
    #[kernel]
    pub fn gkr_fold(table: *mut u64, shape: [u64; 5]) {
        let k = thread::index_1d().get() as u64;
        let count = shape[0];
        if k >= count {
            return;
        }
        let stride = shape[1];
        let low = 2 * k * stride;
        // SAFETY: the table holds `2·count` logical entries at the stride.
        unsafe {
            let a = ext_load(table, low);
            let b = ext_load(table, low + stride);
            let value = ext_add(a, ext_mul(ext_param(&shape, 2), ext_sub(b, a)));
            let at = 3 * low as usize;
            *table.add(at) = value.c0;
            *table.add(at + 1) = value.c1;
            *table.add(at + 2) = value.c2;
        }
    }

    /// The round message, as partial sums per block: for pair `m < half`,
    /// `Σ_j` over nodes 0, 2, ..., degree of `eq_m(x_j)·Σ_t w_t·(p_t0 q_t1 +
    /// p_t1 q_t0 + μ q_t0 q_t1)(x_j)`. Every table has `4·half` logical
    /// entries at the stride; tree `t` reads children `b` at `b·2·half + 2m`
    /// and `2m + 1`. The eq table has `2·half` logical entries.
    #[kernel]
    pub fn gkr_message(tables: [u64; GKR_PARTS], eq: *const u64, shape: [u64; GKR_SHAPE_WORDS], partials: *mut u64) {
        static mut SHARED: SharedArray<u64, { GKR_THREADS * GKR_NODES * 3 }> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let group = thread::blockIdx_x() as usize;
        let m = (group * GKR_THREADS + tid) as u64;
        let half_size = shape[0];
        let stride = shape[1];
        let trees = shape[2] as usize;
        let degree = shape[3] as usize;
        let mu = ext_param(&shape, 4);
        let mut sums = [EXT_ZERO; GKR_NODES];
        if m < half_size {
            let mut totals = [EXT_ZERO; GKR_NODES];
            for t in 0..trees {
                let factors = shape[7 + 5 * t] as usize;
                let first = shape[8 + 5 * t] as usize;
                let weight = ext_param(&shape, 9 + 5 * t);
                // value[2·part + b], slope[2·part + b] for parts ≤ 3.
                let mut value = [EXT_ZERO; 6];
                let mut slope = [EXT_ZERO; 6];
                for part in 0..=factors {
                    let table = tables[first + part] as *const u64;
                    for b in 0..2 {
                        let at = (b * 2 * half_size + 2 * m) * stride;
                        // SAFETY: each part holds `4·half` logical entries.
                        unsafe {
                            let low = ext_load(table, at);
                            value[2 * part + b as usize] = low;
                            slope[2 * part + b as usize] = ext_sub(ext_load(table, at + stride), low);
                        }
                    }
                }
                for (j, total) in totals.iter_mut().enumerate().take(degree) {
                    ext_step(&mut value, &slope, 2 * (factors + 1), j);
                    let q0 = value[2 * factors];
                    let q1 = value[2 * factors + 1];
                    let cross = if factors == 0 {
                        ext_add(q0, q1)
                    } else if factors == 1 {
                        ext_add(ext_mul(value[0], q1), ext_mul(value[1], q0))
                    } else {
                        ext_add(
                            ext_mul(ext_mul(value[0], value[2]), q1),
                            ext_mul(ext_mul(value[1], value[3]), q0),
                        )
                    };
                    let term = ext_add(cross, ext_mul(ext_mul(mu, q0), q1));
                    *total = ext_add(*total, ext_mul(weight, term));
                }
            }
            // SAFETY: the eq table holds `2·half` logical entries.
            let (low, high) = unsafe { (ext_load(eq, 2 * m * stride), ext_load(eq, (2 * m + 1) * stride)) };
            let mut e = [low];
            let de = [ext_sub(high, low)];
            for j in 0..degree {
                ext_step(&mut e, &de, 1, j);
                sums[j] = ext_mul(e[0], totals[j]);
            }
        }
        for (j, sum) in sums.iter().enumerate() {
            let at = (tid * GKR_NODES + j) * 3;
            // SAFETY: `at + 2` lies in the block's shared words.
            unsafe {
                SHARED[at] = sum.c0;
                SHARED[at + 1] = sum.c1;
                SHARED[at + 2] = sum.c2;
            }
        }
        thread::sync_threads();
        let mut width = GKR_THREADS / 2;
        while width > 0 {
            if tid < width {
                for w in 0..GKR_NODES * 3 {
                    let at = tid * GKR_NODES * 3 + w;
                    // SAFETY: both indices lie in the shared words.
                    unsafe { SHARED[at] = gl_add(SHARED[at], SHARED[at + width * GKR_NODES * 3]) };
                }
            }
            thread::sync_threads();
            width /= 2;
        }
        if tid == 0 {
            for w in 0..GKR_NODES * 3 {
                // SAFETY: the host gives one partial per block.
                unsafe { *partials.add(group * GKR_NODES * 3 + w) = SHARED[w] };
            }
        }
    }
}
