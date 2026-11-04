#![forbid(unsafe_code)]

mod poseidon2;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

pub use poseidon2::{domain_chunk_v1_2, fold_domain_chunk_v1_2, FoldCoin, Poseidon2Transcript};

/// Lean `ProductionKey.priorDigest` of a v1.2 fresh public input. The PiCCS transcript absorbs
/// the complete input after the digest, so the digest adds no binding of its own.
pub fn prior_digest_v1_2(public_input: &[F]) -> [F; 4] {
    core::array::from_fn(|word| {
        (0..64).fold(F::ZERO, |value, bit| {
            let cell = public_input
                .get(1 + 64 * word + bit)
                .copied()
                .unwrap_or(F::ZERO);
            value + F::from_u64(1 << bit) * cell
        })
    })
}
