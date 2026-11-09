//! The kernels of `carried.metal`: the carried table of the running
//! sources. Each block's weighted lane sum `combined` is projected through
//! the basis forms, plus the row sums.

use cuda_device::{cuda_module, kernel, thread, SharedArray};

use crate::kernels::field::*;

/// Lanes of one block: `RING_DEGREE` rounded up to a power of two.
pub(crate) const CARRIED_THREADS: usize = 64;

/// # Safety
///
/// `combined` holds `RING_DEGREE` values written by this block, and the
/// buffers hold what `shape` names.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn joint_carried_project(
    combined: *const u64,
    shape: &[u64; 9],
    basis_re: *const u64,
    basis_im: *const u64,
    row_sums: *const u64,
    output: *mut u64,
    block: u64,
    lane: u64,
) {
    thread::sync_threads();
    let blocks = shape[1];
    let column = block * RING_DEGREE + lane;
    if lane >= RING_DEGREE || column >= shape[7] {
        return;
    }
    let mut value = KX_ZERO;
    let mut input = 0;
    // SAFETY: the caller's buffers.
    unsafe {
        while block < blocks && input < RING_DEGREE {
            let entry = (lane * RING_DEGREE + input) as usize;
            let basis = Kx {
                c0: gl_from_word(*basis_re.add(entry)),
                c1: gl_from_word(*basis_im.add(entry)),
            };
            value = kx_add(value, kx_mul(basis, load_k(combined, input)));
            input += 1;
        }
        if column < shape[8] {
            value = kx_add(value, load_k(row_sums, column));
        }
        store_k(output, column, value);
    }
}

#[cuda_module]
pub mod carried_kernels {
    use super::*;

    /// The carried table over masked running sources. Shape: `[count,
    /// blocks, matrices, rows, n, rows, magnitudes, width, row-sum length]`.
    #[kernel]
    pub fn joint_carried_projection(
        masks: *const u64,
        coefficients: *const u64,
        shape: [u64; 9],
        basis_re: *const u64,
        basis_im: *const u64,
        row_sums: *const u64,
        output: *mut u64,
    ) {
        static mut COMBINED: SharedArray<u64, { 2 * RING_DEGREE as usize }> = SharedArray::UNINIT;
        let block = u64::from(thread::blockIdx_x());
        let lane = u64::from(thread::threadIdx_x());
        let (count, blocks, magnitudes) = (shape[0], shape[1], shape[6]);
        let combined = &raw mut COMBINED as *mut u64;
        if lane < RING_DEGREE {
            let mut value = KX_ZERO;
            let bit = 1u64 << lane;
            let mut source = 0;
            // SAFETY: the host gives the masks and coefficients of `count`
            // sources; `lane` indexes the shared values.
            unsafe {
                while block < blocks && source < count {
                    let coefficient = load_k(coefficients, source);
                    let base = 2 * magnitudes * (source * blocks + block);
                    for magnitude in 1..=magnitudes {
                        let scaled = Kx {
                            c0: gl_mul(coefficient.c0, magnitude),
                            c1: gl_mul(coefficient.c1, magnitude),
                        };
                        let at = (base + 2 * (magnitude - 1)) as usize;
                        if *masks.add(at) & bit != 0 {
                            value = kx_add(value, scaled);
                        } else if *masks.add(at + 1) & bit != 0 {
                            value = kx_sub(value, scaled);
                        }
                    }
                    source += 1;
                }
                store_k(combined, lane, value);
            }
        }
        // SAFETY: the buffers hold what the shape names.
        unsafe { joint_carried_project(combined, &shape, basis_re, basis_im, row_sums, output, block, lane) };
    }

    /// The same table over running sources without norm terms, whose field
    /// values are stored source by source, block by block and lane by lane.
    #[kernel]
    pub fn joint_carried_projection_dense(
        values: *const u64,
        coefficients: *const u64,
        shape: [u64; 9],
        basis_re: *const u64,
        basis_im: *const u64,
        row_sums: *const u64,
        output: *mut u64,
    ) {
        static mut COMBINED: SharedArray<u64, { 2 * RING_DEGREE as usize }> = SharedArray::UNINIT;
        let block = u64::from(thread::blockIdx_x());
        let lane = u64::from(thread::threadIdx_x());
        let (count, blocks) = (shape[0], shape[1]);
        let combined = &raw mut COMBINED as *mut u64;
        if lane < RING_DEGREE {
            let mut value = KX_ZERO;
            let mut source = 0;
            // SAFETY: the host gives the values and coefficients of `count`
            // sources; `lane` indexes the shared values.
            unsafe {
                while block < blocks && source < count {
                    let coefficient = load_k(coefficients, source);
                    let entry = gl_from_word(*values.add(((source * blocks + block) * RING_DEGREE + lane) as usize));
                    value = kx_add(
                        value,
                        Kx {
                            c0: gl_mul(coefficient.c0, entry),
                            c1: gl_mul(coefficient.c1, entry),
                        },
                    );
                    source += 1;
                }
                store_k(combined, lane, value);
            }
        }
        // SAFETY: the buffers hold what the shape names.
        unsafe { joint_carried_project(combined, &shape, basis_re, basis_im, row_sums, output, block, lane) };
    }
}
