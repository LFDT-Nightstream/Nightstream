#![forbid(unsafe_code)]

#[cfg(feature = "debug-log")]
mod debug;
#[cfg(feature = "fs-guard")]
pub mod fs_guard;
pub mod labels;
mod poseidon2;
mod rng;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// Minimal, byte-first API + typed helpers (Merlin-inspired).
pub trait Transcript {
    fn new(app_label: &'static [u8]) -> Self;
    fn append_message(&mut self, label: &'static [u8], msg: &[u8]);
    fn append_fields(&mut self, label: &'static [u8], fs: &[F]);
    fn challenge_bytes(&mut self, label: &'static [u8], out: &mut [u8]);
    fn challenge_field(&mut self, label: &'static [u8]) -> F;
    fn challenge_fields(&mut self, label: &'static [u8], n: usize) -> Vec<F> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.challenge_field(label));
        }
        out
    }
    fn fork(&self, scope: &'static [u8]) -> Self;
    fn digest32(&mut self) -> [u8; 32];
}

pub trait TranscriptProtocol {
    fn absorb_ccs_header(&mut self, n: usize, m: usize, t: usize);
    fn absorb_poly_sparse(&mut self, label: &'static [u8], coeffs: &[(F, Vec<u32>)]);
    fn absorb_commit_coords(&mut self, coords: &[F]);
    fn absorb_public_fields(&mut self, label: &'static [u8], fs: &[F]);
}

pub use poseidon2::Poseidon2Transcript;

/// Lean `ProductionKey.priorDigest` (`decodeHash`) of a v1.1 fresh public
/// input: digest word `w` is `sum over bit < 64 of 2^bit * input[1 + 64 w + bit]`,
/// computed in the field. Cells past a shorter input read as zero; the PiCCS
/// transcript absorbs the complete input next, so this adds no binding of its
/// own.
pub fn prior_digest_v1_1(public_input: &[F]) -> [F; 4] {
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
pub use rng::{TranscriptRng, TranscriptRngBuilder};
