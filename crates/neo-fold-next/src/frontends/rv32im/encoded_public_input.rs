use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::f_prime::Rv32imEncodedPublicInput;

pub(crate) fn digest32_has_canonical_field_limb_bytes(digest: [u8; 32]) -> bool {
    digest.chunks_exact(8).all(|chunk| {
        let limb = u64::from_le_bytes(chunk.try_into().expect("digest32 has 8-byte limbs"));
        F::from_u64(limb).as_canonical_u64() == limb
    })
}

pub(crate) fn encoded_public_input_has_canonical_field_limb_bytes(input: &Rv32imEncodedPublicInput) -> bool {
    digest32_has_canonical_field_limb_bytes(input.bytes())
}
