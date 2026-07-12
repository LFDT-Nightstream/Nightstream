//! Host/device field-word conversion.
//!
//! Device kernels exchange Goldilocks elements as `u64` words. This module is
//! the host-side boundary that canonicalizes those words before they enter
//! proof structs or transcripts.

use neo_math::{from_complex, F, K};
use p3_field::PrimeCharacteristicRing;

use crate::kernels::goldilocks::GOLDILOCKS_MODULUS;

#[inline]
pub fn f_from_device_word(word: u64) -> F {
    let canonical = if word >= GOLDILOCKS_MODULUS {
        word - GOLDILOCKS_MODULUS
    } else {
        word
    };
    F::from_u64(canonical)
}

#[inline]
pub fn k_from_device_words(c0: u64, c1: u64) -> K {
    from_complex(f_from_device_word(c0), f_from_device_word(c1))
}
