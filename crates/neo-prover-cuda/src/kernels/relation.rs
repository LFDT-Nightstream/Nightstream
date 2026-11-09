//! The kernel of `relation.metal`: the first row of a relation that the
//! application tables fail.

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{cuda_module, kernel, thread};

use crate::kernels::dec_forms::dec_pow;
use crate::kernels::field::*;

#[cuda_module]
pub mod relation_kernels {
    use super::*;

    /// `first_failure = min(first_failure, shape[2] + row)` for each row
    /// `< shape[0]` where `f(M_1 z, ..., M_t z) ≠ 0`; `shape[1]` terms.
    #[kernel]
    pub fn ccs_first_unsatisfied_row(
        tables: *const u64,
        term_headers: *const u64,
        term_variables: *const u64,
        shape: [u64; 3],
        first_failure: *mut u32,
    ) {
        let local_row = thread::index_1d().get() as u64;
        let rows = shape[0];
        if local_row >= rows {
            return;
        }
        let mut sum = 0;
        // SAFETY: the host gives the tables of every matrix, the term
        // metadata and the result word.
        unsafe {
            for term in 0..shape[1] {
                let header = 3 * term as usize;
                let mut value = gl_from_word(*term_headers.add(header));
                let start = *term_headers.add(header + 1);
                let end = start + *term_headers.add(header + 2);
                for variable in start..end {
                    let matrix = *term_variables.add(2 * variable as usize);
                    let exponent = *term_variables.add(2 * variable as usize + 1);
                    value = gl_mul(
                        value,
                        dec_pow(
                            gl_from_word(*tables.add((matrix * rows + local_row) as usize)),
                            exponent,
                        ),
                    );
                }
                sum = gl_add(sum, value);
            }
            if sum != 0 {
                DeviceAtomicU32::from_ptr(first_failure)
                    .fetch_min((shape[2] + local_row) as u32, AtomicOrdering::Relaxed);
            }
        }
    }
}
