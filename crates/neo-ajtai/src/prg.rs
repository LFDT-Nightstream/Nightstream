//! Ajtai PP seeded row expansion (PRG) — v1
//!
//! Deterministically expands a public seed into Ajtai rows using the unified
//! Poseidon2(Goldilocks, w=16, r=8, cap=8) sponge. Host-only utility for now.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use p3_field::PrimeField64;

/// Expand a single Ajtai row deterministically from a seed and row index.
///
/// - Domain tag: b"neo/ajtai/prg/v2"
/// - Input bytes: domain || seed || row_idx_le || len_le || ctr_le
/// - Output: `len` field elements in Goldilocks (as F), produced in chunks of 4.
///
/// Compared to v1, this version also absorbs the row length, removing any
/// ambiguity across circuits that might share the same seed and row indices
/// but differ in row length.
pub fn expand_row_v2(seed: &[u8; 32], row_idx: u64, len: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(len);
    let mut ctr: u64 = 0;
    while out.len() < len {
        let mut bytes = Vec::with_capacity(b"neo/ajtai/prg/v2".len() + 32 + 8 + 8 + 8);
        bytes.extend_from_slice(b"neo/ajtai/prg/v2");
        bytes.extend_from_slice(seed);
        bytes.extend_from_slice(&row_idx.to_le_bytes());
        bytes.extend_from_slice(&(len as u64).to_le_bytes());
        bytes.extend_from_slice(&ctr.to_le_bytes());
        let digest = p2::poseidon2_hash_packed_bytes(&bytes);
        for g in digest.iter() {
            if out.len() >= len { break; }
            out.push(F::from_u64(g.as_canonical_u64()));
        }
        ctr = ctr.wrapping_add(1);
    }
    out
}
