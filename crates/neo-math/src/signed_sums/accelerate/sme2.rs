use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::OnceLock;

use p3_field::PrimeCharacteristicRing;

use super::{D, F};

const GROUP_COLUMNS: usize = 1 << 22;
const ROW_WORDS: usize = 204_800 / size_of::<u64>();
const _: () = assert!(GROUP_COLUMNS * u8::MAX as usize <= i32::MAX as usize);
const _: () = assert!(ROW_WORDS <= super::ROW_WORKSPACE_WORDS);

extern "C" {
    fn sysctlbyname(name: *const c_char, old: *mut c_void, len: *mut usize, new: *mut c_void, new_len: usize) -> c_int;
    fn ns_sme2_streaming_bytes() -> usize;
    fn ns_sme2_signed_dot(
        columns: usize,
        keys: *const u64,
        positive: *const u64,
        negative: *const u64,
        scratch: *mut u64,
        out: *mut i128,
    );
}

fn has_feature(name: &CStr) -> bool {
    let mut value: c_int = 0;
    let mut len = size_of_val(&value);
    // SAFETY: name is NUL-terminated and len bounds the writable integer.
    #[allow(unsafe_code)]
    let status = unsafe {
        sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    status == 0 && len == size_of_val(&value) && value == 1
}

fn available_on_this_thread() -> bool {
    static HARDWARE: OnceLock<bool> = OnceLock::new();
    let supported =
        HARDWARE.get_or_init(|| has_feature(c"hw.optional.arm.FEAT_SME2") && has_feature(c"hw.optional.arm.SME_I8I32"));
    // SAFETY: RDSVL runs only on an SME-capable CPU.
    #[allow(unsafe_code)]
    unsafe {
        *supported && ns_sme2_streaming_bytes() == 64
    }
}

pub(super) fn try_signed_ring_dot(keys: &[u64], positive: &[u64], negative: &[u64]) -> Option<[F; D]> {
    if !available_on_this_thread() {
        return None;
    }
    super::check_row(keys, positive, negative);
    let mut scratch = Vec::<u64>::with_capacity(ROW_WORDS);
    let mut sums = [0i128; D];
    for ((key, pos), neg) in keys
        .chunks(GROUP_COLUMNS * D)
        .zip(positive.chunks(GROUP_COLUMNS))
        .zip(negative.chunks(GROUP_COLUMNS))
    {
        let mut partial = [0i128; D];
        // SAFETY: The worker checked the ISA and vector length. Each mask has D key words.
        // C initializes its 204800 bytes of u64-aligned scratch and D output words before use.
        #[allow(unsafe_code)]
        unsafe {
            ns_sme2_signed_dot(
                pos.len(),
                key.as_ptr(),
                pos.as_ptr(),
                neg.as_ptr(),
                scratch.as_mut_ptr(),
                partial.as_mut_ptr(),
            );
        }
        for (sum, value) in sums.iter_mut().zip(partial) {
            *sum += value;
        }
    }
    Some(sums.map(F::from_i128))
}
