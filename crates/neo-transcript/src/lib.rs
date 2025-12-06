#![forbid(unsafe_code)]

#[cfg(feature = "debug-log")]
mod debug;
#[cfg(feature = "fs-guard")]
pub mod fs_guard;
pub mod labels;
mod poseidon2;
mod rng;

use neo_math::F;

/// Minimal, byte-first API + typed helpers (Merlin-inspired).
pub trait Transcript {
    fn new(app_label: &'static [u8]) -> Self;
    fn append_message(&mut self, label: &'static [u8], msg: &[u8]);
    fn append_fields(&mut self, label: &'static [u8], fs: &[F]);
    fn challenge_bytes(&mut self, label: &'static [u8], out: &mut [u8]);
    fn challenge_field(&mut self, label: &'static [u8]) -> F;
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
pub use rng::{TranscriptRng, TranscriptRngBuilder};
