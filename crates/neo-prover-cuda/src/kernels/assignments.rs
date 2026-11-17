//! The kernels of `assignments.metal`: the folds of the norm-term
//! assignments, as masks, then small codes into shared values, then dense
//! `K` values.

use cuda_device::{cuda_module, kernel, thread};

use crate::kernels::field::*;
use crate::kernels::joint::joint_mask_value;

/// The code of assignment entry `index` of `source`, or `zero` past the
/// prefix.
///
/// # Safety
///
/// `data` and `sources` hold the prefix of the encoding.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn joint_assignment_code(
    data: *const u64,
    sources: *const u64,
    source: u64,
    index: u64,
    length: u64,
    encoding: u64,
    blocks: u64,
    magnitudes: u64,
    zero: u64,
) -> u64 {
    if index >= length {
        return zero;
    }
    let position = (source * length + index) as usize;
    // SAFETY: the caller's prefix.
    unsafe {
        if encoding == 1 {
            return u64::from(*(data as *const u8).add(position));
        }
        if encoding == 2 {
            return u64::from(*(data as *const u16).add(position));
        }
        let value = joint_mask_value(data, blocks, *sources.add(source as usize), index, magnitudes);
        if value <= magnitudes {
            zero + value
        } else {
            zero - gl_sub(0, value)
        }
    }
}

/// The `K` value of assignment entry `index` of `source`, or zero past the
/// prefix.
///
/// # Safety
///
/// `data`, `sources` and `values` hold the prefix of the encoding.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn joint_assignment_value(
    data: *const u64,
    sources: *const u64,
    values: *const u64,
    source: u64,
    index: u64,
    encoding: u64,
    blocks: u64,
    magnitudes: u64,
    length: u64,
) -> Kx {
    if index >= length {
        return KX_ZERO;
    }
    // SAFETY: the caller's prefix.
    unsafe {
        if encoding == 0 {
            return Kx {
                c0: joint_mask_value(data, blocks, *sources.add(source as usize), index, magnitudes),
                c1: 0,
            };
        }
        if encoding == 16 {
            return load_k(data, source * length + index);
        }
        let position = (source * length + index) as usize;
        let code = if encoding == 1 {
            u64::from(*(data as *const u8).add(position))
        } else {
            u64::from(*(data as *const u16).add(position))
        };
        load_k(values, code)
    }
}

#[cuda_module]
pub mod assignment_kernels {
    use super::*;

    /// The folded value of each code pair: `low + r·(high - low)` for
    /// `code < alphabet^2`, with `alphabet = shape[5]`.
    #[kernel]
    pub fn joint_fold_assignment_values(values: *const u64, challenge: [u64; 2], shape: [u64; 8], output: *mut u64) {
        let code = thread::index_1d().get() as u64;
        let alphabet = shape[5];
        if code >= alphabet * alphabet {
            return;
        }
        let challenge = Kx {
            c0: gl_from_word(challenge[0]),
            c1: gl_from_word(challenge[1]),
        };
        // SAFETY: the host gives `alphabet` values and `alphabet^2` outputs.
        unsafe {
            let low = load_k(values, code % alphabet);
            let high = load_k(values, code / alphabet);
            store_k(output, code, kx_add(low, kx_mul(challenge, kx_sub(high, low))));
        }
    }

    /// One fold of every assignment prefix. Shape: `[length, count, blocks,
    /// magnitudes, encoding, alphabet, zero, next encoding]`.
    #[kernel]
    pub fn joint_fold_assignments(
        data: *const u64,
        challenge: [u64; 2],
        shape: [u64; 8],
        output: *mut u64,
        sources: *const u64,
        values: *const u64,
    ) {
        let index = thread::index_1d().get() as u64;
        let (length, count, blocks, magnitudes) = (shape[0], shape[1], shape[2], shape[3]);
        let (encoding, alphabet, zero, next_encoding) = (shape[4], shape[5], shape[6], shape[7]);
        let next_len = length.div_ceil(2);
        if index >= count * next_len {
            return;
        }
        let source = index / next_len;
        let low_index = 2 * (index % next_len);
        // SAFETY: the host gives the prefix of `encoding` and `count·next_len`
        // outputs of `next_encoding`.
        unsafe {
            if next_encoding != 16 {
                let code =
                    |at| joint_assignment_code(data, sources, source, at, length, encoding, blocks, magnitudes, zero);
                let code = code(low_index) + alphabet * code(low_index + 1);
                if next_encoding == 1 {
                    *(output as *mut u8).add(index as usize) = code as u8;
                } else {
                    *(output as *mut u16).add(index as usize) = code as u16;
                }
            } else {
                let value = |at| {
                    joint_assignment_value(data, sources, values, source, at, encoding, blocks, magnitudes, length)
                };
                let (low, high) = (value(low_index), value(low_index + 1));
                let challenge = Kx {
                    c0: gl_from_word(challenge[0]),
                    c1: gl_from_word(challenge[1]),
                };
                store_k(output, index, kx_add(low, kx_mul(challenge, kx_sub(high, low))));
            }
        }
    }
}
