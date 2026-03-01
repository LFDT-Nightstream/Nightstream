use neo_ccs::crypto::poseidon2_goldilocks as p2;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

const MEMORY_PUBLIC_DIGEST_DOMAIN: &[u8] = b"neo/memory/public_digest/v2";

#[inline]
fn extend_packed_bytes_as_fields(dst: &mut Vec<Goldilocks>, bytes: &[u8]) {
    const BYTES_PER_LIMB: usize = 7;
    dst.push(Goldilocks::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        dst.push(Goldilocks::from_u64(u64::from_le_bytes(limb)));
    }
}

/// Public-memory digest helper used by Route-A transcript binding.
///
/// This is intentionally independent from Fiat-Shamir challenge code paths.
/// It hashes an injective packed encoding of `(domain, label, len(fs), fs...)`.
pub fn memory_public_digest_fields(label: &[u8], fs: &[Goldilocks]) -> [u8; 32] {
    let mut input = Vec::with_capacity(
        1 + MEMORY_PUBLIC_DIGEST_DOMAIN.len().div_ceil(7) + 1 + label.len().div_ceil(7) + 1 + fs.len(),
    );
    extend_packed_bytes_as_fields(&mut input, MEMORY_PUBLIC_DIGEST_DOMAIN);
    extend_packed_bytes_as_fields(&mut input, label);
    input.push(Goldilocks::from_u64(fs.len() as u64));
    input.extend_from_slice(fs);

    let digest = p2::poseidon2_hash(&input);
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..(i + 1) * 8].copy_from_slice(&digest[i].as_canonical_u64().to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn public_digest_label_packing_is_injective_against_modulus_wrap() {
        let fs = [Goldilocks::from_u64(123)];

        let label1 = [0u8; 8];
        let label2 = [0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF];

        let d1 = memory_public_digest_fields(&label1, &fs);
        let d2 = memory_public_digest_fields(&label2, &fs);

        assert_ne!(d1, d2, "public digest label encoding must be injective");
    }
}
