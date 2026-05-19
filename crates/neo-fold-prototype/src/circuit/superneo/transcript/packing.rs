//! Owns native packing of transcript byte and u64 inputs into Goldilocks limbs.

use crate::spartan_backend::SpartanF;

pub(super) fn pack_bytes(bytes: &[u8]) -> Vec<SpartanF> {
    const BYTES_PER_LIMB: usize = 7;
    let mut packed = Vec::with_capacity(bytes.len().div_ceil(BYTES_PER_LIMB));
    let mut i = 0usize;
    while i < bytes.len() {
        let end = (i + BYTES_PER_LIMB).min(bytes.len());
        let mut limb = [0u8; 8];
        limb[..(end - i)].copy_from_slice(&bytes[i..end]);
        packed.push(SpartanF::from_canonical_u64(u64::from_le_bytes(limb)));
        i = end;
    }
    packed
}

pub(super) fn pack_u64s(values: &[u64]) -> Vec<SpartanF> {
    let mut packed = Vec::with_capacity(values.len() * 2);
    for value in values {
        packed.push(SpartanF::from_canonical_u64(value & 0xFFFF_FFFF));
        packed.push(SpartanF::from_canonical_u64(value >> 32));
    }
    packed
}
