//! The kernels of `dft.metal`: a batched DFT over the columns of a
//! row-major Goldilocks matrix. Row `k` of the result is `Σ_i row_i·ω^{ik}`.
//! Radix-2 decimation-in-frequency stages run in passes of up to
//! `DFT_PASS_BITS` stages in shared memory, then a bit reversal puts the
//! rows in natural order.

use cuda_device::{cuda_module, kernel, thread, SharedArray};

use crate::kernels::field::{gl_add, gl_from_word, gl_mul, gl_sub};

pub(crate) const DFT_PASS_BITS: u32 = 8;
pub(crate) const DFT_COLUMNS: usize = 16;

#[cuda_module]
pub mod dft_kernels {
    use super::*;

    /// `twiddles[k] = ω^k` for `k < count`.
    #[kernel]
    pub fn dft_twiddles(twiddles: *mut u64, generator: u64, count: u64) {
        let k = thread::index_1d().get();
        if k as u64 >= count {
            return;
        }
        let mut base = generator;
        let mut power = 1;
        let mut exponent = k;
        while exponent != 0 {
            if exponent & 1 != 0 {
                power = gl_mul(power, base);
            }
            base = gl_mul(base, base);
            exponent >>= 1;
        }
        // SAFETY: the host gives `count` twiddles.
        unsafe { *twiddles.add(k) = power };
    }

    /// Stages `shape[2] .. shape[2] + shape[3]` of a `2^shape[0]`-row DFT
    /// over `shape[1]` columns. A block owns the `2^stages` rows that these
    /// stages mix, for `DFT_COLUMNS` columns (or all columns of a narrower
    /// matrix).
    #[kernel]
    pub fn dft_dif_pass(values: *mut u64, twiddles: *const u64, shape: [u64; 4]) {
        static mut TILE: SharedArray<u64, { (1usize << DFT_PASS_BITS) * DFT_COLUMNS }> = SharedArray::UNINIT;
        let group_index = u64::from(thread::blockIdx_x());
        let tid = thread::threadIdx_x() as usize;
        let threads = thread::blockDim_x() as usize;
        let log_rows = shape[0] as u32;
        let width = shape[1];
        let first = shape[2] as u32;
        let stages = shape[3] as u32;
        let columns = width.min(DFT_COLUMNS as u64) as usize;
        let groups = 1u64 << (log_rows - stages);
        let group = group_index % groups;
        let column0 = (group_index / groups) * columns as u64;
        // The mixed rows are `base + e·stride` for e < 2^stages.
        let stride = 1u64 << (log_rows - first - stages);
        let low = group % stride;
        let base = ((group / stride) << (log_rows - first)) | low;
        let elements = (1usize << stages) * columns;
        let at = |index: usize| {
            let row = base + (index / columns) as u64 * stride;
            (row * width + column0 + (index % columns) as u64) as usize
        };
        let mut index = tid;
        while index < elements {
            // SAFETY: the tile holds `elements` words and the matrix holds
            // every mixed row.
            unsafe { TILE[index] = gl_from_word(*values.add(at(index))) };
            index += threads;
        }
        thread::sync_threads();
        let pairs = elements / 2;
        for u in 0..stages {
            let half_size = 1usize << (stages - 1 - u);
            let mut p = tid;
            while p < pairs {
                let column = p % columns;
                let q = p / columns;
                let j = q % half_size;
                let e0 = (q / half_size) * 2 * half_size + j;
                let at0 = e0 * columns + column;
                let at1 = at0 + half_size * columns;
                // SAFETY: both entries lie in the tile; the twiddle index is
                // below `rows / 2`.
                unsafe {
                    let (a, b) = (TILE[at0], TILE[at1]);
                    // ω^{(i mod h)·2^t} for the row i of e0 at global stage t.
                    let twiddle = *twiddles.add((((j as u64) * stride + low) << (first + u)) as usize);
                    TILE[at0] = gl_add(a, b);
                    TILE[at1] = gl_mul(gl_sub(a, b), twiddle);
                }
                p += threads;
            }
            thread::sync_threads();
        }
        let mut index = tid;
        while index < elements {
            // SAFETY: as for the load.
            unsafe { *values.add(at(index)) = TILE[index] };
            index += threads;
        }
    }

    /// Swap rows `i` and `bitrev(i)` of a `2^log_rows`-row matrix of
    /// `width` columns.
    #[kernel]
    pub fn dft_bit_reverse(values: *mut u64, log_rows: u64, width: u64) {
        let row = thread::index_1d().get() as u64;
        if row >> log_rows != 0 {
            return;
        }
        let other = u64::from((row as u32).reverse_bits() >> (32 - log_rows as u32));
        if row < other {
            for column in 0..width {
                // SAFETY: both rows lie inside the matrix.
                unsafe {
                    let a = values.add((row * width + column) as usize);
                    let b = values.add((other * width + column) as usize);
                    // cuda-oxide cannot compile `core::ptr::swap`, which
                    // goes through `MaybeUninit`.
                    #[allow(clippy::manual_swap)]
                    {
                        let value = *a;
                        *a = *b;
                        *b = value;
                    }
                }
            }
        }
    }
}
